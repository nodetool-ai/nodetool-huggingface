from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    AudioChunk,
    AudioRef,
    HFVoiceActivityDetection,
    HuggingFaceModel,
)
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    _pipeline_thread_pool,
)
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

SAMPLE_RATE = 16_000

PYANNOTE_IMPORT_ERROR = (
    "pyannote.audio is required for speaker diarization but is not installed. "
    "Install it with `pip install 'pyannote.audio>=3.3,<4'`."
)


def _missing_token_error(repo_id: str) -> str:
    return (
        f"'{repo_id}' is a gated model. Please accept the model's conditions at "
        f"https://huggingface.co/{repo_id} (and at the repositories it depends on, "
        "e.g. https://huggingface.co/pyannote/segmentation-3.0) and set HF_TOKEN "
        "in Nodetool settings or in the environment."
    )


class SpeakerDiarization(HuggingFacePipelineNode):
    """
    Detects who spoke when in an audio recording using pyannote.audio.
    audio, diarization, speaker, segmentation, who-spoke-when, pyannote, huggingface

    Use cases:
    - Label meeting or interview recordings with per-speaker turns
    - Combine with Whisper chunks to build speaker-attributed transcripts
    - Split podcasts or calls into per-speaker segments for editing
    - Measure talk-time balance across participants
    - Pre-process training data that requires single-speaker audio

    **Note:** pyannote models are GATED on the Hugging Face Hub. Before running this
    node you must accept the conditions of `pyannote/speaker-diarization-3.1` and of
    the models it depends on (`pyannote/segmentation-3.0`), then set `HF_TOKEN` in
    Nodetool settings (or as an environment variable). Without a token the model
    cannot be downloaded.

    **Links:**
    - https://github.com/pyannote/pyannote-audio
    - https://huggingface.co/pyannote/speaker-diarization-3.1
    """

    model: HFVoiceActivityDetection = Field(
        default=HFVoiceActivityDetection(
            repo_id="pyannote/speaker-diarization-3.1",
        ),
        title="Model",
        description="The pyannote.audio diarization pipeline to use. Gated on the Hugging Face Hub: accept the model conditions and set HF_TOKEN.",
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        title="Audio",
        description="The audio file to diarize. Supports common formats like WAV, MP3, FLAC.",
    )
    num_speakers: int = Field(
        default=0,
        ge=0,
        title="Number of Speakers",
        description="Exact number of speakers in the recording. Use 0 to let the model decide automatically.",
    )
    min_speakers: int = Field(
        default=0,
        ge=0,
        title="Minimum Speakers",
        description="Lower bound on the number of speakers. Use 0 to leave it unset. Ignored when 'Number of Speakers' is set.",
    )
    max_speakers: int = Field(
        default=0,
        ge=0,
        title="Maximum Speakers",
        description="Upper bound on the number of speakers. Use 0 to leave it unset. Ignored when 'Number of Speakers' is set.",
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "Speaker Diarization"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "audio", "num_speakers"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFVoiceActivityDetection(
                repo_id="pyannote/speaker-diarization-3.1",
            ),
            HFVoiceActivityDetection(
                repo_id="pyannote/speaker-diarization-community-1",
            ),
            HFVoiceActivityDetection(
                repo_id="pyannote/segmentation-3.0",
            ),
        ]

    class OutputType(TypedDict):
        segments: list[AudioChunk]
        speakers: list[str]

    def required_inputs(self):
        return ["audio"]

    def requires_gpu(self) -> bool:
        # Diarization models are small and run acceptably on CPU.
        return False

    async def _resolve_token(self, context: ProcessingContext) -> str | None:
        """HF token from Nodetool settings, falling back to the environment."""
        token: str | None = None
        try:
            token = await context.get_secret("HF_TOKEN")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read HF_TOKEN from settings: %s", exc)
        return token or os.environ.get("HF_TOKEN") or None

    def _diarization_kwargs(self) -> dict[str, int]:
        """Only forward speaker-count hints that were actually set (> 0).

        ``num_speakers`` is exclusive with the min/max bounds in pyannote, so the
        exact count wins when both are provided.
        """
        if self.num_speakers > 0:
            return {"num_speakers": self.num_speakers}

        kwargs: dict[str, int] = {}
        if self.min_speakers > 0:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers > 0:
            kwargs["max_speakers"] = self.max_speakers
        return kwargs

    async def preload_model(self, context: ProcessingContext):
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ImportError(PYANNOTE_IMPORT_ERROR) from exc

        repo_id = self.model.repo_id or "pyannote/speaker-diarization-3.1"
        token = await self._resolve_token(context)
        if not token:
            raise ValueError(_missing_token_error(repo_id))

        logger.info("Loading pyannote diarization pipeline: %s", repo_id)

        def _load():
            return Pipeline.from_pretrained(repo_id, use_auth_token=token)

        loop = asyncio.get_event_loop()
        pipeline = await loop.run_in_executor(_pipeline_thread_pool, _load)

        if pipeline is None:
            # pyannote returns None instead of raising when access is denied.
            raise ValueError(_missing_token_error(repo_id))

        self._pipeline = pipeline
        await self.move_to_device(context.device)
        logger.info("pyannote diarization pipeline loaded.")

    async def move_to_device(self, device: str):
        if self._pipeline is None:
            return
        import torch

        self._pipeline.to(torch.device(device))
        logger.info("Moved diarization pipeline to device: %s", device)

    @staticmethod
    def _to_segments(diarization: Any) -> "SpeakerDiarization.OutputType":
        """Flatten a pyannote ``Annotation`` into chunks ordered by start time."""
        segments: list[AudioChunk] = []
        speakers: list[str] = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            label = str(speaker)
            segments.append(
                AudioChunk(
                    timestamp=(float(turn.start), float(turn.end)),
                    text=label,
                )
            )
            if label not in speakers:
                speakers.append(label)

        segments.sort(key=lambda chunk: chunk.timestamp[0])
        speakers.sort()
        return {"segments": segments, "speakers": speakers}

    async def process(self, context: ProcessingContext) -> OutputType:
        assert self._pipeline is not None, "Pipeline not initialized"

        import torch

        samples, _, _ = await context.audio_to_numpy(
            self.audio, sample_rate=SAMPLE_RATE
        )

        # pyannote expects a (channel, time) waveform tensor.
        waveform = torch.from_numpy(samples).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        diarization = await self.run_pipeline_in_thread(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            **self._diarization_kwargs(),
        )

        result = self._to_segments(diarization)
        logger.info(
            "Diarization produced %d segments across %d speakers.",
            len(result["segments"]),
            len(result["speakers"]),
        )
        return result
