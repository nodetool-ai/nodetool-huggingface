"""Nunchaku FLUX transformers must route to the base repo and pipeline they belong to.

The local text-to-image / image-to-image providers hand any nunchaku SVDQ FLUX
file to ``load_nunchaku_flux_pipeline`` without naming a pipeline class, so the
variant detected from the file name decides both which base repo is assembled
and which diffusers pipeline assembles it. Getting either wrong produces a
signature error deep in diffusers, or a silently wrong image.
"""

import pytest

from nodetool.huggingface.flux_utils import (
    detect_flux_variant,
    flux_variant_to_base_model_id,
    is_nunchaku_flux_transformer,
)


NUNCHAKU_FILES = [
    ("nunchaku-ai/nunchaku-flux.1-schnell", "svdq-int4_r32-flux.1-schnell.safetensors", "schnell", "black-forest-labs/FLUX.1-schnell"),
    ("nunchaku-ai/nunchaku-flux.1-dev", "svdq-int4_r32-flux.1-dev.safetensors", "dev", "black-forest-labs/FLUX.1-dev"),
    ("nunchaku-ai/nunchaku-flux.1-fill-dev", "svdq-int4_r32-flux.1-fill-dev.safetensors", "fill", "black-forest-labs/FLUX.1-Fill-dev"),
    ("nunchaku-ai/nunchaku-flux.1-canny-dev", "svdq-int4_r32-flux.1-canny-dev.safetensors", "canny", "black-forest-labs/FLUX.1-Canny-dev"),
    ("nunchaku-ai/nunchaku-flux.1-depth-dev", "svdq-int4_r32-flux.1-depth-dev.safetensors", "depth", "black-forest-labs/FLUX.1-Depth-dev"),
    ("nunchaku-ai/nunchaku-flux.1-kontext-dev", "svdq-int4_r32-flux.1-kontext-dev.safetensors", "kontext", "black-forest-labs/FLUX.1-Kontext-dev"),
]


@pytest.mark.parametrize("repo_id,path,variant,base", NUNCHAKU_FILES)
def test_shipped_nunchaku_files_detect_their_own_variant(repo_id, path, variant, base):
    assert is_nunchaku_flux_transformer(repo_id, path)
    assert detect_flux_variant(repo_id, path) == variant
    assert flux_variant_to_base_model_id(variant) == base


def test_derivative_repos_are_not_read_as_plain_dev():
    """"-dev" is a suffix on every derivative repo, so it must be matched last."""
    for repo_id, path, variant, _ in NUNCHAKU_FILES:
        if variant in ("schnell", "dev"):
            continue
        assert detect_flux_variant(repo_id, path) != "dev", repo_id


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("schnell", "FluxPipeline"),
        ("dev", "FluxPipeline"),
        ("fill", "FluxFillPipeline"),
        ("canny", "FluxControlPipeline"),
        ("depth", "FluxControlPipeline"),
        ("kontext", "FluxKontextPipeline"),
    ],
)
def test_variant_selects_its_pipeline_class(variant, expected):
    pytest.importorskip("diffusers")
    from nodetool.huggingface.flux_utils import flux_variant_to_pipeline_class

    assert flux_variant_to_pipeline_class(variant).__name__ == expected
