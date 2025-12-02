import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.image_to_image import (
    FluxPriorRedux,
    FluxPriorReduxQuantization,
)


def test_flux_prior_redux_resolve_fp16():
    node = FluxPriorRedux()
    node.quantization = FluxPriorReduxQuantization.FP16

    base_model, transformer_model = node._resolve_flux_model_config(
        FluxPriorReduxQuantization.FP16
    )

    assert base_model.repo_id == "black-forest-labs/FLUX.1-dev"
    assert transformer_model is None


def test_flux_prior_redux_resolve_int4():
    node = FluxPriorRedux()
    node.quantization = FluxPriorReduxQuantization.INT4

    base_model, transformer_model = node._resolve_flux_model_config(
        FluxPriorReduxQuantization.INT4
    )

    assert base_model.repo_id == "black-forest-labs/FLUX.1-dev"
    assert transformer_model is not None
    assert transformer_model.repo_id == "nunchaku-tech/nunchaku-flux.1-dev"
    assert "svdq-int4" in (transformer_model.path or "")
