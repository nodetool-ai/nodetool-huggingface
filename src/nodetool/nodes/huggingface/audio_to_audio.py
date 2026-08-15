"""
Audio-to-audio nodes backed by SpeechBrain's SepFormer models.

Both nodes share a single inference path: ``SepformerSeparation`` from
``speechbrain.inference.separation``. Separation models emit one waveform per
speaker, the enhancement checkpoints emit a single cleaned waveform — the same
``separate_batch`` API covers both, which keeps the two nodes on identical
mechanics.

SpeechBrain is an optional dependency; it is imported lazily inside the nodes so
importing this module (for node discovery/metadata) never requires it.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import numpy as np
from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import AudioRef, HFAudioToAudio, HuggingFaceModel
from nodetool.nodes.huggingface._3d_common import MissingDependencyError
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    _pipeline_thread_pool,
)
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


SPEECHBRAIN_INSTALL_HINT = (
    "SpeechBrain is not installed. Install it with: pip install 'speechbrain>=1.0,<2'"
)

# Native sample rates of the supported checkpoints. SpeechBrain models are
# trained at a fixed rate and degrade badly when fed anything else, so the input
# audio is always resampled to the value resolved here.
_MODEL_SAMPLE_RATES: dict[str, int] = {
    "speechbrain/sepformer-wsj02mix": 8_000,
    "speechbrain/sepformer-wsj03mix": 8_000,
    "speechbrain/sepformer-whamr": 8_000,
    "speechbrain/sepformer-wham": 8_000,
    "speechbrain/sepformer-whamr16k": 16_000,
    "speechbrain/sepformer-wham16k-enhancement": 16_000,
    "speechbrain/sepformer-dns4-16k-enhancement": 16_000,
    "speechbrain/sepformer-libri2mix": 8_000,
    "speechbrain/sepformer-libri3mix": 8_000,
}

_DEFAULT_SAMPLE_RATE = 8_000


def _sample_rate_for_repo(repo_id: str) -> int:
    """Resolve a checkpoint's expected sample rate from its repo id.

    Known repos come from the table; anything else is inferred from a ``16k`` /
    ``8k`` marker in the name, falling back to 8 kHz (the SepFormer default).
    """
    repo_id = (repo_id or "").strip()
    known = _MODEL_SAMPLE_RATES.get(repo_id)
    if known:
        return known
    name = repo_id.lower()
    if re.search(r"(^|[^0-9])16k", name) or "16000" in name:
        return 16_000
    if re.search(r"(^|[^0-9])8k", name) or "8000" in name:
        return 8_000
    return _DEFAULT_SAMPLE_RATE


def _sample_rate_from_model(model: Any, repo_id: str) -> int:
    """Prefer the rate the loaded hparams declare, else infer from the repo id."""
    hparams = getattr(model, "hparams", None)
    rate = getattr(hparams, "sample_rate", None)
    if rate is None and isinstance(hparams, dict):
        rate = hparams.get("sample_rate")
    try:
        rate_int = int(rate)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _sample_rate_for_repo(repo_id)
    return rate_int if rate_int > 0 else _sample_rate_for_repo(repo_id)


def _hf_cache_dir() -> str:
    """The HF cache root, resolved the same way as the rest of the package."""
    return os.environ.get(
        "HUGGINGFACE_HUB_CACHE",
        os.environ.get(
            "HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        ),
    )


def _speechbrain_savedir(repo_id: str) -> str:
    """Checkpoint directory for a SpeechBrain repo, under the HF cache root.

    SpeechBrain wants a plain directory to symlink/copy the fetched checkpoint
    files into, so it gets its own subtree next to the hub cache rather than
    polluting the hub's own ``models--*`` layout.
    """
    safe = (repo_id or "unknown").replace("/", "--")
    return os.path.join(_hf_cache_dir(), "speechbrain", safe)


def _require_sepformer() -> Any:
    """Import ``SepformerSeparation`` or raise an actionable error."""
    try:
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MissingDependencyError(
            "Audio source separation requires the 'speechbrain' package, "
            "which is not installed in this environment.",
            install_hint=SPEECHBRAIN_INSTALL_HINT,
        ) from exc
    return SepformerSeparation


async def _load_separator(repo_id: str, device: str | None) -> Any:
    """Load a SepFormer checkpoint on the shared pipeline thread pool."""
    if not repo_id:
        raise ValueError("A model repository id is required")

    separation_cls = _require_sepformer()
    savedir = _speechbrain_savedir(repo_id)
    run_opts = {"device": device} if device else None

    def _load() -> Any:
        os.makedirs(savedir, exist_ok=True)
        return separation_cls.from_hparams(
            source=repo_id,
            savedir=savedir,
            run_opts=run_opts,
        )

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_pipeline_thread_pool, _load)


def _as_numpy(estimated: Any) -> np.ndarray:
    """Convert a ``separate_batch`` result to a ``(time, n_sources)`` array."""
    if hasattr(estimated, "detach"):
        estimated = estimated.detach().cpu().numpy()
    array = np.asarray(estimated, dtype=np.float32)
    if array.ndim == 3:  # (batch, time, n_sources)
        array = array[0]
    elif array.ndim == 2:  # (batch, time) — single-source enhancement output
        array = array[0][:, None]
    elif array.ndim == 1:
        array = array[:, None]
    else:
        raise ValueError(
            f"Unexpected separation output with shape {np.shape(estimated)}"
        )
    return array


async def _separate(model: Any, samples: np.ndarray) -> np.ndarray:
    """Run separation in the shared thread pool, returning ``(time, n_sources)``."""
    if model is None:
        raise ValueError("Separation model not initialized")
    if samples.size == 0:
        raise ValueError("Input audio is empty")

    mono = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))

    def _call() -> np.ndarray:
        import torch

        with torch.inference_mode():
            batch = torch.from_numpy(mono).float().unsqueeze(0)
            # SepformerSeparation.separate_batch moves the mixture onto the
            # model's own device, so no manual placement is needed here.
            estimated = model.separate_batch(batch)
        result = _as_numpy(estimated)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_pipeline_thread_pool, _call)


def _finalize(source: np.ndarray, normalize: bool) -> np.ndarray:
    """Return a float32 waveform safely inside ``[-1, 1]``."""
    wave = np.asarray(source, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if normalize and peak > 0.0:
        # SepFormer output amplitude is arbitrary; scale to a headroom-safe peak.
        wave = wave * (0.95 / peak)
    return np.clip(wave, -1.0, 1.0).astype(np.float32)


class SpeechSeparation(HuggingFacePipelineNode):
    """
    Separates overlapping speakers in a recording into one track per speaker.
    audio, separation, speech, speaker, cocktail-party, speechbrain, sepformer

    Use cases:
    - Split a two-person interview recorded on a single microphone
    - Isolate individual voices before transcribing a noisy meeting
    - Clean up overlapping speech for diarization or captioning pipelines
    - Extract a single speaker from a crowded, reverberant recording
    - Prepare training data by de-mixing multi-speaker audio

    **Note:** Requires the optional `speechbrain` package. Audio is resampled to
    the checkpoint's native rate (8 kHz for the wsj0 models, 16 kHz for the
    whamr16k/dns4 models) before separation.
    """

    model: HFAudioToAudio = Field(
        default=HFAudioToAudio(repo_id="speechbrain/sepformer-wsj02mix"),
        title="Model",
        description=(
            "The SpeechBrain SepFormer checkpoint. wsj02mix separates 2 speakers, "
            "wsj03mix separates 3, whamr16k handles 2 speakers with noise and reverb."
        ),
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        title="Audio",
        description="The mixed audio to separate. Converted to mono and resampled automatically.",
    )
    normalize: bool = Field(
        default=True,
        title="Normalize",
        description="Peak-normalize each separated track. Separated sources otherwise have arbitrary loudness.",
    )

    _model: Any = None
    _sample_rate: int = _DEFAULT_SAMPLE_RATE

    @classmethod
    def get_title(cls) -> str:
        return "Speech Separation"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "audio"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFAudioToAudio(
                repo_id="speechbrain/sepformer-wsj02mix",
                allow_patterns=["*.yaml", "*.ckpt", "*.txt", "*.json"],
            ),
            HFAudioToAudio(
                repo_id="speechbrain/sepformer-wsj03mix",
                allow_patterns=["*.yaml", "*.ckpt", "*.txt", "*.json"],
            ),
            HFAudioToAudio(
                repo_id="speechbrain/sepformer-whamr16k",
                allow_patterns=["*.yaml", "*.ckpt", "*.txt", "*.json"],
            ),
            HFAudioToAudio(
                repo_id="speechbrain/sepformer-dns4-16k-enhancement",
                allow_patterns=["*.yaml", "*.ckpt", "*.txt", "*.json"],
            ),
        ]

    def required_inputs(self):
        return ["audio"]

    def get_model_id(self) -> str:
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        self._model = await _load_separator(self.get_model_id(), context.device)
        self._sample_rate = _sample_rate_from_model(self._model, self.get_model_id())

    async def move_to_device(self, device: str):
        # No-op: SpeechBrain owns placement. The device is fixed at load time via
        # ``run_opts`` and ``separate_batch`` moves inputs onto it per call.
        pass

    async def process(self, context: ProcessingContext) -> list[AudioRef]:
        assert self._model is not None, "Separation model not initialized"

        sample_rate = int(self._sample_rate or _DEFAULT_SAMPLE_RATE)
        samples, _, _ = await context.audio_to_numpy(
            self.audio, sample_rate=sample_rate
        )
        sources = await _separate(self._model, samples)

        outputs: list[AudioRef] = []
        for index in range(sources.shape[1]):
            wave = _finalize(sources[:, index], self.normalize)
            outputs.append(await context.audio_from_numpy(wave, sample_rate))
        log.debug("Separated %d source(s) at %d Hz", len(outputs), sample_rate)
        return outputs


class SpeechEnhancement(HuggingFacePipelineNode):
    """
    Removes background noise and reverberation from a single-speaker recording.
    audio, enhancement, denoise, speech, cleanup, speechbrain, sepformer

    Use cases:
    - Clean up noisy voice memos and field recordings before transcription
    - Reduce room reverberation in podcast or interview takes
    - Improve intelligibility of call-center or conference audio
    - Preprocess speech for downstream ASR or speaker-recognition models
    - Denoise user-submitted voice notes in an automated pipeline

    **Note:** Requires the optional `speechbrain` package. Uses the same SepFormer
    inference path as Speech Separation and returns the first (cleaned) source.
    """

    model: HFAudioToAudio = Field(
        default=HFAudioToAudio(repo_id="speechbrain/sepformer-dns4-16k-enhancement"),
        title="Model",
        description=(
            "The SpeechBrain SepFormer enhancement checkpoint. dns4-16k-enhancement "
            "targets general noise, wham16k-enhancement targets noisy speech mixtures."
        ),
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        title="Audio",
        description="The noisy audio to clean up. Converted to mono and resampled automatically.",
    )
    normalize: bool = Field(
        default=True,
        title="Normalize",
        description="Peak-normalize the enhanced track. The model output otherwise has arbitrary loudness.",
    )

    _model: Any = None
    _sample_rate: int = 16_000

    @classmethod
    def get_title(cls) -> str:
        return "Speech Enhancement"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "audio"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFAudioToAudio(
                repo_id="speechbrain/sepformer-dns4-16k-enhancement",
                allow_patterns=["*.yaml", "*.ckpt", "*.txt", "*.json"],
            ),
            HFAudioToAudio(
                repo_id="speechbrain/sepformer-wham16k-enhancement",
                allow_patterns=["*.yaml", "*.ckpt", "*.txt", "*.json"],
            ),
        ]

    def required_inputs(self):
        return ["audio"]

    def get_model_id(self) -> str:
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        self._model = await _load_separator(self.get_model_id(), context.device)
        self._sample_rate = _sample_rate_from_model(self._model, self.get_model_id())

    async def move_to_device(self, device: str):
        # No-op: SpeechBrain owns placement (see SpeechSeparation.move_to_device).
        pass

    async def process(self, context: ProcessingContext) -> AudioRef:
        assert self._model is not None, "Enhancement model not initialized"

        sample_rate = int(self._sample_rate or 16_000)
        samples, _, _ = await context.audio_to_numpy(
            self.audio, sample_rate=sample_rate
        )
        sources = await _separate(self._model, samples)
        wave = _finalize(sources[:, 0], self.normalize)
        return await context.audio_from_numpy(wave, sample_rate)
