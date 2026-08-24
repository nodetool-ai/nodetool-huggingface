"""
Shared utilities for FLUX model detection and routing.
"""

from __future__ import annotations

from typing import Any


# Every FLUX derivative repo is also named "…-dev", so "dev" is tested last:
# checked earlier it swallows fill, canny, depth and kontext.
_VARIANT_MARKERS = ("schnell", "fill", "canny", "depth", "kontext", "dev")


def detect_flux_variant(repo_id: str, file_path: str | None) -> str:
    """Detect which FLUX base model should be used."""
    candidates = [repo_id, file_path or ""]
    for value in candidates:
        lower = value.lower()
        for marker in _VARIANT_MARKERS:
            if marker in lower:
                return marker
    return "dev"


def flux_variant_to_base_model_id(variant: str) -> str:
    """Map detected variant names to canonical FLUX repos."""
    mapping = {
        "schnell": "black-forest-labs/FLUX.1-schnell",
        "fill": "black-forest-labs/FLUX.1-Fill-dev",
        "canny": "black-forest-labs/FLUX.1-Canny-dev",
        "depth": "black-forest-labs/FLUX.1-Depth-dev",
        "kontext": "black-forest-labs/FLUX.1-Kontext-dev",
        "dev": "black-forest-labs/FLUX.1-dev",
    }
    return mapping.get(variant, "black-forest-labs/FLUX.1-dev")


def is_nunchaku_transformer(repo_id: str, file_path: str | None) -> bool:
    """Detect Nunchaku SVDQ transformer files (FLUX or Qwen)."""
    if not file_path:
        return False
    return "nunchaku" in repo_id.lower() and "svdq" in file_path.lower()


def is_nunchaku_qwen_transformer(repo_id: str, file_path: str | None) -> bool:
    """Detect Nunchaku SVDQ transformer files for Qwen-Image models."""
    if not is_nunchaku_transformer(repo_id, file_path):
        return False
    return "qwen" in repo_id.lower() or "qwen" in (file_path or "").lower()


def is_nunchaku_flux_transformer(repo_id: str, file_path: str | None) -> bool:
    """Detect Nunchaku SVDQ transformer files that belong to a FLUX pipeline."""
    if not is_nunchaku_transformer(repo_id, file_path):
        return False
    return not is_nunchaku_qwen_transformer(repo_id, file_path)


def flux_variant_to_pipeline_class(variant: str) -> Any:
    """Map a detected variant to the diffusers pipeline class its base repo needs.

    A FLUX derivative repo is not loadable by ``FluxPipeline``: Fill takes a mask,
    Canny and Depth take a control image, Kontext takes a reference image, and each
    ships a different ``model_index.json``. Assuming the plain pipeline turns a
    wrong-model bug into a confusing signature error deep inside diffusers.
    """
    if variant == "fill":
        from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline

        return FluxFillPipeline
    if variant in ("canny", "depth"):
        from diffusers.pipelines.flux.pipeline_flux_control import FluxControlPipeline

        return FluxControlPipeline
    if variant == "kontext":
        from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline

        return FluxKontextPipeline
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

    return FluxPipeline
