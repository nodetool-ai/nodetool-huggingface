import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.image_to_image import (
    FluxKontext,
    FluxKontextQuantization,
)


def test_flux_kontext_resolve_fp16():
    node = FluxKontext()
    node.quantization = FluxKontextQuantization.FP16

    transformer_model, text_encoder_model = node._resolve_model_config()

    assert transformer_model.repo_id == "black-forest-labs/FLUX.1-Kontext-dev"
    assert transformer_model.path is None
    assert text_encoder_model is None


def test_flux_kontext_resolve_int4():
    node = FluxKontext()
    node.quantization = FluxKontextQuantization.INT4

    transformer_model, text_encoder_model = node._resolve_model_config()

    assert transformer_model.repo_id == "nunchaku-tech/nunchaku-flux.1-kontext-dev"
    assert "svdq-int4" in (transformer_model.path or "")
    assert text_encoder_model is not None
    assert text_encoder_model.repo_id == "mit-han-lab/nunchaku-t5"
