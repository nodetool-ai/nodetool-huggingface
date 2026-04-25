import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.image_to_video import Wan_FLF2V


def test_wan_flf2v_default_model():
    node = Wan_FLF2V()
    assert node.model_variant == Wan_FLF2V.WanFLF2VModel.WAN_2_2_FLF2V_14B_720P


def test_wan_flf2v_get_model_id():
    node = Wan_FLF2V()
    assert node.get_model_id() == "Wan-AI/Wan2.2-FLF2V-14B-720P-Diffusers"


def test_wan_flf2v_model_variant_2_1():
    node = Wan_FLF2V()
    node.model_variant = Wan_FLF2V.WanFLF2VModel.WAN_2_1_FLF2V_14B_720P
    assert node.get_model_id() == "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"


def test_wan_flf2v_recommended_models():
    models = Wan_FLF2V.get_recommended_models()
    assert len(models) == 2
    repo_ids = [m.repo_id for m in models]
    assert "Wan-AI/Wan2.2-FLF2V-14B-720P-Diffusers" in repo_ids
    assert "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers" in repo_ids


def test_wan_flf2v_required_inputs():
    assert Wan_FLF2V.required_inputs(Wan_FLF2V()) == ["first_image", "last_image"]


def test_wan_flf2v_title():
    assert Wan_FLF2V.get_title() == "Wan (First-Last-Frame to Video)"
