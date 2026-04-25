import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.summarization import Summarization


def test_summarization_default_model():
    node = Summarization()
    assert node.model.repo_id == "facebook/bart-large-cnn"


def test_summarization_recommended_models():
    models = Summarization.get_recommended_models()
    assert len(models) > 0
    repo_ids = [m.repo_id for m in models]
    assert any("bart" in r for r in repo_ids)
    assert any("pegasus" in r for r in repo_ids)


def test_summarization_required_inputs():
    assert Summarization.required_inputs(Summarization()) == ["text"]


def test_summarization_title():
    assert Summarization.get_title() == "Summarization"


def test_summarization_defaults():
    node = Summarization()
    assert node.max_length == 130
    assert node.min_length == 30
    assert node.do_sample is False
