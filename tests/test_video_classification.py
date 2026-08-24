import asyncio
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.video_classification import VideoClassifier


def test_video_classifier_default_model():
    node = VideoClassifier()
    assert node.model.repo_id == "MCG-NJU/videomae-base-finetuned-kinetics"


def test_video_classifier_recommended_models():
    models = VideoClassifier.get_recommended_models()
    assert len(models) > 0
    repo_ids = [m.repo_id for m in models]
    assert any("videomae" in r for r in repo_ids)
    assert any("timesformer" in r for r in repo_ids)


def test_video_classifier_required_inputs():
    assert VideoClassifier.required_inputs(VideoClassifier()) == ["video"]


def test_video_classifier_title():
    assert VideoClassifier.get_title() == "HF Video Classifier"


class _FakeConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeModel:
    def __init__(self, config):
        self.config = config


class _FakePipeline:
    def __init__(self, config):
        self.model = _FakeModel(config)


class _FakeContext:
    device = "cpu"

    async def asset_to_bytes(self, video):
        return b"not-a-real-mp4"


def test_frames_expected_by_config_reads_num_frames():
    from nodetool.nodes.huggingface.video_classification import (
        frames_expected_by_config,
    )

    assert frames_expected_by_config(_FakeConfig(num_frames=16)) == 16


def test_frames_expected_by_config_falls_back_to_frames_per_clip():
    from nodetool.nodes.huggingface.video_classification import (
        frames_expected_by_config,
    )

    assert frames_expected_by_config(_FakeConfig(frames_per_clip=32)) == 32


def test_frames_expected_by_config_returns_none_without_either_key():
    from nodetool.nodes.huggingface.video_classification import (
        frames_expected_by_config,
    )

    assert frames_expected_by_config(_FakeConfig(image_size=224)) is None


def _run_with_config(monkeypatch, config, **props):
    """Run process() against a stub pipeline and return the kwargs it received."""
    recorded: dict[str, Any] = {}

    async def fake_run(self, *args, **kwargs):
        recorded.update(kwargs)
        return [{"label": "archery", "score": 1.0}]

    monkeypatch.setattr(VideoClassifier, "run_pipeline_in_thread", fake_run)
    node = VideoClassifier(**props)
    node._pipeline = _FakePipeline(config)
    asyncio.run(node.process(_FakeContext()))  # type: ignore[arg-type]
    return recorded


def test_process_uses_the_frame_count_the_model_config_pins(monkeypatch):
    """The default must follow the model, not a fixed 8.

    The node's default model (videomae-base-finetuned-kinetics) has fixed
    temporal position embeddings for 16 frames; sending 8 raises a tensor
    size mismatch inside the pipeline.
    """
    recorded = _run_with_config(monkeypatch, _FakeConfig(num_frames=16))
    assert recorded.get("num_frames") == 16


def test_process_uses_frames_per_clip_for_vjepa2(monkeypatch):
    recorded = _run_with_config(monkeypatch, _FakeConfig(frames_per_clip=32))
    assert recorded.get("num_frames") == 32


def test_process_passes_an_explicit_override_unchanged(monkeypatch):
    recorded = _run_with_config(monkeypatch, _FakeConfig(num_frames=16), num_frames=8)
    assert recorded.get("num_frames") == 8


def test_process_omits_num_frames_when_the_config_pins_nothing(monkeypatch):
    recorded = _run_with_config(monkeypatch, _FakeConfig(image_size=224))
    assert "num_frames" not in recorded
