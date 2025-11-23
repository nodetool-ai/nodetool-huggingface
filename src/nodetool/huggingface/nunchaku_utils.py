"""
Nunchaku model optimization utilities.

This module provides utilities for automatically detecting and loading quantized
nunchaku models (transformers and text encoders) when available in the HF cache.
"""

import asyncio
from pathlib import Path
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.hf_fast_cache import HfFastCache

log = get_logger(__name__)


async def find_nunchaku_t5_encoder() -> tuple[str, str] | None:
    """
    Find a nunchaku quantized T5 encoder in the HF cache.

    Returns:
        tuple[str, str] | None: (repo_id, file_path) if found, else None
    """
    from nunchaku.utils import get_precision

    cache = HfFastCache()
    precision = get_precision()

    t5_repo = "mit-han-lab/nunchaku-t5"
    t5_file = f"awq-{precision}-flux.1-t5xxl.safetensors"

    try:
        # Check if T5 repo is cached
        repo_root = await cache.repo_root(t5_repo, repo_type="model")
        if not repo_root:
            log.debug(f"Nunchaku T5 repo {t5_repo} not found in cache")
            return None

        # Check if the specific file exists
        files = await cache.list_files(t5_repo, repo_type="model")
        if t5_file in files:
            log.info(f"Found nunchaku T5 encoder: {t5_repo}/{t5_file}")
            return (t5_repo, t5_file)
        else:
            log.debug(f"T5 encoder {t5_file} not found in {t5_repo}")
            return None
    except Exception as exc:
        log.debug(f"Error checking for nunchaku T5 encoder: {exc}")
        return None


async def get_nunchaku_text_encoder(
    model_repo_id: str, precision: str | None = None
) -> Any | None:
    """
    Get text encoder kwargs when using a nunchaku model.

    This function checks if the model is a nunchaku FLUX variant and if so,
    attempts to find and load the nunchaku T5 text encoder from the HF cache.

    Args:
        model_repo_id: The model repo ID being used (could be nunchaku or base model)
        precision: Optional precision override ('int4' or 'fp4'), auto-detected if None

    Returns:
        dict: Pipeline kwargs with text_encoder_2 if nunchaku T5 encoder is found

    Example:
        >>> # Using nunchaku-flux model
        >>> kwargs = await get_nunchaku_text_encoder_kwargs("nunchaku-tech/nunchaku-flux.1-dev")
        >>> # kwargs = {"text_encoder_2": <NunchakuT5EncoderModel>}
        >>>
        >>> # Use with pipeline
        >>> transformer = NunchakuFluxTransformer2dModel.from_pretrained(...)
        >>> pipeline = FluxPipeline.from_pretrained(
        ...     "black-forest-labs/FLUX.1-dev",
        ...     transformer=transformer,
        ...     **kwargs,  # Adds text_encoder_2
        ... )
    """
    from nunchaku import NunchakuT5EncoderModel
    from nunchaku.utils import get_precision

    if precision is None:
        precision = get_precision()

    # Try to find nunchaku T5 encoder
    t5_info = await find_nunchaku_t5_encoder()
    if t5_info:
        # Lazy import to avoid dependency issues
        from nunchaku import NunchakuT5EncoderModel

        repo_id, file_path = t5_info
        full_path = f"{repo_id}/{file_path}"

        log.info(f"Nunchaku model detected - loading T5 encoder from {full_path}")
        text_encoder_2 = await asyncio.to_thread(
            NunchakuT5EncoderModel.from_pretrained, full_path
        )
        return text_encoder_2
    else:
        log.debug(f"Nunchaku T5 encoder not found in cache (looking for precision={precision})")

    return None
