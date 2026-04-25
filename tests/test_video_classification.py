import sys
from pathlib import Path

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
    assert VideoClassifier.get_title() == "Video Classifier"
