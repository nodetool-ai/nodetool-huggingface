import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.text_to_image import (
    StableDiffusionXL,
    StableDiffusionXLQuantization,
)


def test_sdxl_base_model_fp16_uses_full_repo():
    node = StableDiffusionXL()
    base_model = node._get_base_model(StableDiffusionXLQuantization.FP16)
    assert base_model.repo_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert base_model.allow_patterns is None


def test_sdxl_base_model_quantized_filters_transformer():
    node = StableDiffusionXL()
    base_model = node._get_base_model(StableDiffusionXLQuantization.INT4)
    assert base_model.repo_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert base_model.allow_patterns is not None
    assert "unet/config.json" in base_model.allow_patterns


def test_sdxl_resolve_nunchaku_model_fp4():
    node = StableDiffusionXL()
    model = node._resolve_transformer_model(
        StableDiffusionXLQuantization.FP4,
        use_legacy_transformer=False,
    )
    assert model is not None
    assert model.repo_id == "nunchaku-tech/nunchaku-sdxl"
    assert model.path == "svdq-fp4_r32-sdxl.safetensors"


def test_sdxl_detects_legacy_quantization_from_model_path():
    node = StableDiffusionXL()
    node.model.repo_id = "nunchaku-tech/nunchaku-sdxl"
    node.model.path = "svdq-int4_r32-sdxl.safetensors"
    legacy = node._detect_legacy_quantization()
    assert legacy == StableDiffusionXLQuantization.INT4
