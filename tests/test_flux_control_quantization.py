import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.text_to_image import (
    FluxControl,
    FluxControlQuantization,
)


def test_flux_control_resolve_fp16_depth():
    node = FluxControl()
    node.model.repo_id = "black-forest-labs/FLUX.1-Depth-dev"

    base_model, transformer_model, text_encoder_model = node._resolve_model_config(
        FluxControlQuantization.FP16
    )

    assert base_model.repo_id == "black-forest-labs/FLUX.1-Depth-dev"
    assert transformer_model is None
    assert text_encoder_model is None


def test_flux_control_resolve_int4_canny():
    node = FluxControl()
    node.model.repo_id = "black-forest-labs/FLUX.1-Canny-dev"

    base_model, transformer_model, text_encoder_model = node._resolve_model_config(
        FluxControlQuantization.INT4
    )

    assert base_model.repo_id == "black-forest-labs/FLUX.1-Canny-dev"
    assert transformer_model is not None
    assert transformer_model.repo_id == "nunchaku-tech/nunchaku-flux.1-canny-dev"
    assert "svdq-int4" in (transformer_model.path or "")
    assert text_encoder_model is not None
    assert text_encoder_model.repo_id == "mit-han-lab/nunchaku-t5"
