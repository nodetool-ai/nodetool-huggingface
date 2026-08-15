"""Unit tests for the pyannote-based SpeakerDiarization node.

pyannote.audio is an optional dependency and the diarization checkpoints are
gated on the Hugging Face Hub, so every test here mocks the pipeline: a fake
``pyannote.audio`` module is installed into ``sys.modules`` and a stub
annotation object stands in for the model output.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT.parent / "nodetool-core" / "src"
HF_SRC = ROOT / "src"
for p in (str(CORE_SRC), str(HF_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from nodetool.nodes.huggingface.speaker_diarization import (  # noqa: E402
    SpeakerDiarization,
)


class FakeTurn:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class FakeAnnotation:
    """Minimal stand-in for pyannote's ``Annotation`` result object."""

    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        assert yield_label, "the node always requests labels"
        for start, end, speaker in self._tracks:
            yield FakeTurn(start, end), None, speaker


class FakePipeline:
    """Records the call the node makes and returns a canned annotation."""

    def __init__(self, annotation):
        self._annotation = annotation
        self.calls: list[tuple[tuple, dict]] = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._annotation


class FakeContext:
    """Only the two ProcessingContext members the node touches."""

    def __init__(self, samples=None, secret=None, device="cpu"):
        self._samples = samples
        self._secret = secret
        self.device = device
        self.requested_sample_rate = None

    async def get_secret(self, key):
        assert key == "HF_TOKEN"
        return self._secret

    async def audio_to_numpy(self, audio, sample_rate=None):
        self.requested_sample_rate = sample_rate
        return self._samples, sample_rate, 1


@pytest.fixture
def fake_pyannote(monkeypatch):
    """Install a fake ``pyannote.audio`` exposing a controllable ``Pipeline``."""
    loaded: dict = {}

    class Pipeline:
        return_value: object = None

        @staticmethod
        def from_pretrained(repo_id, use_auth_token=None):
            loaded["repo_id"] = repo_id
            loaded["token"] = use_auth_token
            return Pipeline.return_value

    pyannote = types.ModuleType("pyannote")
    pyannote.__path__ = []  # mark as package
    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = Pipeline
    pyannote.audio = audio

    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    return types.SimpleNamespace(Pipeline=Pipeline, loaded=loaded)


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------


def test_default_model_is_speaker_diarization_31():
    node = SpeakerDiarization()
    assert node.model.repo_id == "pyannote/speaker-diarization-3.1"


def test_title():
    assert SpeakerDiarization.get_title() == "Speaker Diarization"


def test_basic_fields():
    assert SpeakerDiarization.get_basic_fields() == ["model", "audio", "num_speakers"]


def test_required_inputs():
    assert SpeakerDiarization.required_inputs(SpeakerDiarization()) == ["audio"]


def test_recommended_models_include_pyannote():
    repo_ids = [m.repo_id for m in SpeakerDiarization.get_recommended_models()]
    assert "pyannote/speaker-diarization-3.1" in repo_ids
    assert all(r and r.startswith("pyannote/") for r in repo_ids)


def test_docstring_documents_gating():
    doc = SpeakerDiarization.__doc__ or ""
    assert "GATED" in doc
    assert "HF_TOKEN" in doc


# --------------------------------------------------------------------------
# speaker-count kwargs
# --------------------------------------------------------------------------


def test_kwargs_empty_when_all_zero():
    assert SpeakerDiarization()._diarization_kwargs() == {}


def test_kwargs_exact_count():
    node = SpeakerDiarization(num_speakers=3)
    assert node._diarization_kwargs() == {"num_speakers": 3}


def test_kwargs_bounds():
    node = SpeakerDiarization(min_speakers=2, max_speakers=5)
    assert node._diarization_kwargs() == {"min_speakers": 2, "max_speakers": 5}


def test_kwargs_partial_bounds():
    assert SpeakerDiarization(max_speakers=4)._diarization_kwargs() == {
        "max_speakers": 4
    }


def test_exact_count_wins_over_bounds():
    """pyannote rejects num_speakers together with min/max bounds."""
    node = SpeakerDiarization(num_speakers=2, min_speakers=1, max_speakers=9)
    assert node._diarization_kwargs() == {"num_speakers": 2}


# --------------------------------------------------------------------------
# annotation -> segments
# --------------------------------------------------------------------------


def test_to_segments_maps_turns_to_audio_chunks():
    annotation = FakeAnnotation(
        [
            (0.5, 2.25, "SPEAKER_00"),
            (2.25, 4.0, "SPEAKER_01"),
        ]
    )
    result = SpeakerDiarization._to_segments(annotation)

    assert [c.timestamp for c in result["segments"]] == [(0.5, 2.25), (2.25, 4.0)]
    assert [c.text for c in result["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert result["speakers"] == ["SPEAKER_00", "SPEAKER_01"]


def test_to_segments_sorts_by_start_and_dedupes_speakers():
    annotation = FakeAnnotation(
        [
            (5.0, 6.0, "SPEAKER_00"),
            (1.0, 2.0, "SPEAKER_01"),
            (3.0, 4.0, "SPEAKER_00"),
        ]
    )
    result = SpeakerDiarization._to_segments(annotation)

    assert [c.timestamp[0] for c in result["segments"]] == [1.0, 3.0, 5.0]
    assert result["speakers"] == ["SPEAKER_00", "SPEAKER_01"]


def test_to_segments_empty_annotation():
    result = SpeakerDiarization._to_segments(FakeAnnotation([]))
    assert result == {"segments": [], "speakers": []}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preload_uses_settings_token(fake_pyannote):
    pipeline = FakePipeline(FakeAnnotation([]))
    fake_pyannote.Pipeline.return_value = pipeline

    node = SpeakerDiarization()
    await node.preload_model(FakeContext(secret="hf_secret_token"))

    assert fake_pyannote.loaded["repo_id"] == "pyannote/speaker-diarization-3.1"
    assert fake_pyannote.loaded["token"] == "hf_secret_token"
    assert node._pipeline is pipeline


@pytest.mark.asyncio
async def test_preload_falls_back_to_env_token(fake_pyannote, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env_token")
    fake_pyannote.Pipeline.return_value = FakePipeline(FakeAnnotation([]))

    node = SpeakerDiarization()
    await node.preload_model(FakeContext(secret=None))

    assert fake_pyannote.loaded["token"] == "env_token"


@pytest.mark.asyncio
async def test_preload_without_token_raises_gated_error(fake_pyannote, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake_pyannote.Pipeline.return_value = FakePipeline(FakeAnnotation([]))

    node = SpeakerDiarization()
    with pytest.raises(ValueError, match="gated"):
        await node.preload_model(FakeContext(secret=None))


@pytest.mark.asyncio
async def test_preload_none_pipeline_raises_gated_error(fake_pyannote):
    """pyannote returns None (instead of raising) when access is denied."""
    fake_pyannote.Pipeline.return_value = None

    node = SpeakerDiarization()
    with pytest.raises(ValueError, match="gated"):
        await node.preload_model(FakeContext(secret="tok"))


@pytest.mark.asyncio
async def test_preload_without_pyannote_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyannote", None)
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)

    node = SpeakerDiarization()
    with pytest.raises(ImportError, match="pyannote.audio is required"):
        await node.preload_model(FakeContext(secret="tok"))


@pytest.mark.asyncio
async def test_move_to_device_is_noop_without_pipeline():
    node = SpeakerDiarization()
    await node.move_to_device("cpu")  # must not raise


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_builds_waveform_dict_and_returns_segments():
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("torch")

    annotation = FakeAnnotation([(0.0, 1.5, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01")])
    pipeline = FakePipeline(annotation)

    node = SpeakerDiarization(num_speakers=2)
    node._pipeline = pipeline

    samples = numpy.zeros(16_000, dtype=numpy.float32)
    context = FakeContext(samples=samples)

    result = await node.process(context)

    assert context.requested_sample_rate == 16_000

    args, kwargs = pipeline.calls[0]
    payload = args[0]
    assert payload["sample_rate"] == 16_000
    assert tuple(payload["waveform"].shape) == (1, 16_000)
    assert kwargs == {"num_speakers": 2}

    assert result["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert result["segments"][0].text == "SPEAKER_00"
    assert result["segments"][0].timestamp == (0.0, 1.5)


@pytest.mark.asyncio
async def test_process_without_pipeline_raises():
    node = SpeakerDiarization()
    with pytest.raises(AssertionError):
        await node.process(FakeContext(samples=None))
