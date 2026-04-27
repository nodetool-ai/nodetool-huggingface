import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.image_segmentation import MaskGeneration


def test_mask_generation_default_model():
    node = MaskGeneration()
    assert node.model.repo_id == "facebook/sam2.1-hiera-large"


def test_mask_generation_get_model_id():
    node = MaskGeneration()
    assert node.get_model_id() == "facebook/sam2.1-hiera-large"


def test_mask_generation_recommended_models():
    models = MaskGeneration.get_recommended_models()
    assert len(models) > 0
    repo_ids = [m.repo_id for m in models]
    assert any("sam2" in r for r in repo_ids)
    assert any("sam3" in r for r in repo_ids)


def test_mask_generation_required_inputs():
    assert MaskGeneration.required_inputs(MaskGeneration()) == ["image"]


def test_mask_generation_title():
    assert MaskGeneration.get_title() == "Mask Generation (SAM)"


def test_mask_generation_defaults():
    node = MaskGeneration()
    assert node.points_per_side == 32
    assert node.pred_iou_thresh == 0.88


def test_mask_generation_basic_fields():
    assert MaskGeneration.get_basic_fields() == [
        "model",
        "image",
        "points_per_side",
        "pred_iou_thresh",
    ]


