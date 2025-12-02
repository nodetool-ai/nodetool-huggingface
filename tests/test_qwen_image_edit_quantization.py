import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.image_to_image import (
    QwenImageEdit,
    QwenImageEditQuantization,
)


def test_qwen_image_edit_resolve_fp16():
    node = QwenImageEdit()
    node.quantization = QwenImageEditQuantization.FP16

    model = node._resolve_model_config()

    assert model.repo_id == "Qwen/Qwen-Image-Edit"
    assert model.path is None


def test_qwen_image_edit_resolve_int4():
    node = QwenImageEdit()
    node.quantization = QwenImageEditQuantization.INT4

    model = node._resolve_model_config()

    assert model.repo_id == "nunchaku-tech/nunchaku-qwen-image-edit"
    assert model.path == "svdq-int4_r32-qwen-image-edit.safetensors"
