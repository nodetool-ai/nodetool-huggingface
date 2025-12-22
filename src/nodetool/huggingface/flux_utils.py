"""
Shared utilities for FLUX model detection and routing.
"""
from __future__ import annotations


def detect_flux_variant(repo_id: str, file_path: str | None) -> str:
    """Detect which FLUX base model should be used."""
    candidates = [repo_id, file_path or ""]
    for value in candidates:
        lower = value.lower()
        if "schnell" in lower:
            return "schnell"
        if "fill" in lower:
            return "fill"
        if "canny" in lower:
            return "canny"
        if "depth" in lower:
            return "depth"
        if "dev" in lower:
            return "dev"
    return "dev"


def flux_variant_to_base_model_id(variant: str) -> str:
    """Map detected variant names to canonical FLUX repos."""
    mapping = {
        "schnell": "black-forest-labs/FLUX.1-schnell",
        "fill": "black-forest-labs/FLUX.1-Fill-dev",
        "canny": "black-forest-labs/FLUX.1-Canny-dev",
        "depth": "black-forest-labs/FLUX.1-Depth-dev",
        "dev": "black-forest-labs/FLUX.1-dev",
    }
    return mapping.get(variant, "black-forest-labs/FLUX.1-dev")


def is_nunchaku_transformer(repo_id: str, file_path: str | None) -> bool:
    """Detect Nunchaku FLUX transformer files."""
    if not file_path:
        return False
    repo_lower = repo_id.lower()
    return (
        "nunchaku" in repo_lower
        and "flux" in repo_lower
        and "svdq" in file_path.lower()
    )

