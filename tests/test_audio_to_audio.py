"""Tests for the SpeechBrain-backed audio-to-audio nodes.

The separation model is mocked end to end — no SpeechBrain checkpoint is
downloaded and no real inference runs.
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.metadata.types import AudioRef, HFAudioToAudio
from nodetool.nodes.huggingface import audio_to_audio
from nodetool.nodes.huggingface.audio_to_audio import (
    SpeechEnhancement,
    SpeechSeparation,
    _as_numpy,
    _finalize,
    _sample_rate_for_repo,
    _sample_rate_from_model,
    _speechbrain_savedir,
)


class _FakeSeparator:
    """Stand-in for ``SepformerSeparation``.

    ``separate_batch`` ignores the mixture content and returns a deterministic
    ``(batch, time, n_sources)`` tensor of the same length as the input.
    """

    def __init__(self, n_sources: int = 2, sample_rate: int | None = None):
        self.n_sources = n_sources
        self.calls: list[tuple] = []
        if sample_rate is not None:
            self.hparams = type("H", (), {"sample_rate": sample_rate})()

    def separate_batch(self, mix):
        import torch

        self.calls.append(tuple(mix.shape))
        length = mix.shape[-1]
        sources = [
            torch.full((length,), float(i + 1) * 0.25) for i in range(self.n_sources)
        ]
        return torch.stack(sources, dim=-1).unsqueeze(0)


class _FakeContext:
    """Minimal ProcessingContext stand-in for audio I/O."""

    def __init__(self, samples: np.ndarray, device: str = "cpu"):
        self.device = device
        self._samples = samples
        self.requested_rates: list[int] = []
        self.produced: list[tuple[np.ndarray, int]] = []

    async def audio_to_numpy(self, audio_ref, sample_rate=16_000, mono=True):
        self.requested_rates.append(sample_rate)
        return self._samples, sample_rate, 1

    async def audio_from_numpy(self, data, sample_rate, num_channels=1, **kwargs):
        self.produced.append((np.asarray(data), sample_rate))
        return AudioRef(uri=f"memory://source-{len(self.produced)}")


# ---------------------------------------------------------------------------
# Metadata / conventions
# ---------------------------------------------------------------------------


def test_speech_separation_metadata():
    node = SpeechSeparation()
    assert node.model.repo_id == "speechbrain/sepformer-wsj02mix"
    assert isinstance(node.model, HFAudioToAudio)
    assert SpeechSeparation.get_title() == "Speech Separation"
    assert SpeechSeparation.get_basic_fields() == ["model", "audio"]
    assert node.required_inputs() == ["audio"]
    assert node.get_model_id() == "speechbrain/sepformer-wsj02mix"


def test_speech_enhancement_metadata():
    node = SpeechEnhancement()
    assert node.model.repo_id == "speechbrain/sepformer-dns4-16k-enhancement"
    assert SpeechEnhancement.get_title() == "Speech Enhancement"
    assert SpeechEnhancement.get_basic_fields() == ["model", "audio"]
    assert node.required_inputs() == ["audio"]


def test_recommended_models_are_audio_to_audio():
    repo_ids = [m.repo_id for m in SpeechSeparation.get_recommended_models()]
    assert "speechbrain/sepformer-wsj02mix" in repo_ids
    assert "speechbrain/sepformer-wsj03mix" in repo_ids
    assert "speechbrain/sepformer-whamr16k" in repo_ids
    assert all(
        isinstance(m, HFAudioToAudio) for m in SpeechSeparation.get_recommended_models()
    )
    assert all(
        isinstance(m, HFAudioToAudio)
        for m in SpeechEnhancement.get_recommended_models()
    )


# ---------------------------------------------------------------------------
# Sample-rate resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repo_id,expected",
    [
        ("speechbrain/sepformer-wsj02mix", 8_000),
        ("speechbrain/sepformer-wsj03mix", 8_000),
        ("speechbrain/sepformer-whamr16k", 16_000),
        ("speechbrain/sepformer-dns4-16k-enhancement", 16_000),
        ("someone/custom-sepformer-16k", 16_000),
        ("someone/mystery-checkpoint", 8_000),
    ],
)
def test_sample_rate_for_repo(repo_id, expected):
    assert _sample_rate_for_repo(repo_id) == expected


def test_sample_rate_prefers_hparams():
    model = _FakeSeparator(sample_rate=22_050)
    assert _sample_rate_from_model(model, "speechbrain/sepformer-wsj02mix") == 22_050


def test_sample_rate_falls_back_to_repo_id():
    model = _FakeSeparator()  # no hparams attribute
    assert _sample_rate_from_model(model, "speechbrain/sepformer-whamr16k") == 16_000


def test_savedir_is_under_hf_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    savedir = _speechbrain_savedir("speechbrain/sepformer-wsj02mix")
    assert savedir == str(tmp_path / "speechbrain" / "speechbrain--sepformer-wsj02mix")


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def test_as_numpy_shapes():
    assert _as_numpy(np.zeros((1, 100, 2))).shape == (100, 2)
    assert _as_numpy(np.zeros((1, 100))).shape == (100, 1)
    assert _as_numpy(np.zeros(100)).shape == (100, 1)


def test_as_numpy_rejects_unexpected_rank():
    with pytest.raises(ValueError):
        _as_numpy(np.zeros((1, 2, 3, 4)))


def test_finalize_normalizes_and_clips():
    loud = np.array([0.0, 4.0, -2.0], dtype=np.float32)
    normalized = _finalize(loud, normalize=True)
    assert normalized.dtype == np.float32
    assert pytest.approx(float(np.max(np.abs(normalized))), rel=1e-5) == 0.95

    clipped = _finalize(loud, normalize=False)
    assert float(np.max(np.abs(clipped))) <= 1.0


def test_finalize_handles_silence():
    silent = np.zeros(16, dtype=np.float32)
    assert not np.any(_finalize(silent, normalize=True))


# ---------------------------------------------------------------------------
# Missing-dependency handling
# ---------------------------------------------------------------------------


def test_missing_speechbrain_raises_actionable_error(monkeypatch):
    from nodetool.nodes.huggingface._3d_common import MissingDependencyError

    # A ``None`` entry in sys.modules makes the import machinery raise
    # ImportError, whether or not speechbrain is actually installed.
    monkeypatch.setitem(sys.modules, "speechbrain", None)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", None)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.separation", None)

    with pytest.raises(MissingDependencyError) as exc:
        audio_to_audio._require_sepformer()
    assert "speechbrain" in str(exc.value).lower()
    assert exc.value.install_hint and "pip install" in exc.value.install_hint


# ---------------------------------------------------------------------------
# process() with a fully mocked model
# ---------------------------------------------------------------------------


def test_speech_separation_process_returns_one_ref_per_source():
    node = SpeechSeparation(
        model=HFAudioToAudio(repo_id="speechbrain/sepformer-wsj02mix")
    )
    node._model = _FakeSeparator(n_sources=2)
    node._sample_rate = 8_000
    node.audio = AudioRef(uri="memory://mix")

    samples = np.random.default_rng(0).standard_normal(1_000).astype(np.float32)
    context = _FakeContext(samples)

    result = asyncio.run(node.process(context))

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(ref, AudioRef) for ref in result)
    # Input was resampled to the model's native rate, outputs carry it back out.
    assert context.requested_rates == [8_000]
    assert [rate for _, rate in context.produced] == [8_000, 8_000]
    # Each emitted waveform is float32, mono, and the same length as the input.
    for data, _ in context.produced:
        assert data.dtype == np.float32
        assert data.shape == (1_000,)
        assert float(np.max(np.abs(data))) <= 1.0
    # The mixture reached the model as a (batch, time) tensor.
    assert node._model.calls == [(1, 1_000)]


def test_speech_separation_supports_three_speakers():
    node = SpeechSeparation(
        model=HFAudioToAudio(repo_id="speechbrain/sepformer-wsj03mix")
    )
    node._model = _FakeSeparator(n_sources=3)
    node._sample_rate = 8_000
    context = _FakeContext(np.zeros(256, dtype=np.float32))

    result = asyncio.run(node.process(context))
    assert len(result) == 3


def test_speech_enhancement_process_returns_single_ref():
    node = SpeechEnhancement()
    node._model = _FakeSeparator(n_sources=1)
    node._sample_rate = 16_000
    context = _FakeContext(
        np.random.default_rng(1).standard_normal(512).astype(np.float32)
    )

    result = asyncio.run(node.process(context))

    assert isinstance(result, AudioRef)
    assert context.requested_rates == [16_000]
    assert len(context.produced) == 1
    data, rate = context.produced[0]
    assert rate == 16_000
    assert data.dtype == np.float32
    assert data.shape == (512,)


def test_process_rejects_empty_audio():
    node = SpeechEnhancement()
    node._model = _FakeSeparator(n_sources=1)
    node._sample_rate = 16_000
    context = _FakeContext(np.zeros(0, dtype=np.float32))

    with pytest.raises(ValueError):
        asyncio.run(node.process(context))


def test_preload_model_uses_context_device_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    captured: dict = {}

    class _FakeSepformerClass:
        @staticmethod
        def from_hparams(source, savedir, run_opts=None):
            captured.update(source=source, savedir=savedir, run_opts=run_opts)
            return _FakeSeparator(n_sources=2, sample_rate=8_000)

    monkeypatch.setattr(
        audio_to_audio, "_require_sepformer", lambda: _FakeSepformerClass
    )

    node = SpeechSeparation()
    context = _FakeContext(np.zeros(8, dtype=np.float32), device="cuda")
    asyncio.run(node.preload_model(context))

    assert captured["source"] == "speechbrain/sepformer-wsj02mix"
    assert captured["run_opts"] == {"device": "cuda"}
    assert captured["savedir"].startswith(str(tmp_path))
    assert Path(captured["savedir"]).is_dir()
    assert node._sample_rate == 8_000


def test_move_to_device_is_a_noop():
    node = SpeechSeparation()
    sentinel = _FakeSeparator()
    node._model = sentinel
    asyncio.run(node.move_to_device("cpu"))
    assert node._model is sentinel
