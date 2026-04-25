import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.text2text_generation import Text2TextGeneration


def test_text2text_default_model():
    node = Text2TextGeneration()
    assert node.model.repo_id == "google/flan-t5-base"


def test_text2text_recommended_models():
    models = Text2TextGeneration.get_recommended_models()
    assert len(models) > 0
    repo_ids = [m.repo_id for m in models]
    assert any("flan-t5" in r for r in repo_ids)
    assert any("mt5" in r for r in repo_ids)


def test_text2text_required_inputs():
    assert Text2TextGeneration.required_inputs(Text2TextGeneration()) == ["text"]


def test_text2text_title():
    assert Text2TextGeneration.get_title() == "Text-to-Text Generation"


def test_text2text_defaults():
    node = Text2TextGeneration()
    assert node.max_new_tokens == 200
    assert node.do_sample is False
    assert node.temperature == 1.0
