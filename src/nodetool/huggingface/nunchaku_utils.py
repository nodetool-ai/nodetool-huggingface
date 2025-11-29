"""
Nunchaku model optimization utilities.

This module provides utilities for automatically detecting and loading quantized
nunchaku models (transformers and text encoders) when available in the HF cache.
"""

from typing import Any

import torch

from nodetool.config.logging_config import get_logger
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


async def get_nunchaku_text_encoder(
    context: ProcessingContext,
    node_id: str,
    repo_id: str | None = None,
    path: str | None = None,
) -> Any | None:
    """
    Get text encoder kwargs when using a nunchaku model.

    This function checks if the model is a nunchaku FLUX variant and if so,
    attempts to find and load the nunchaku T5 text encoder from the HF cache.

    Args:
        context: The context object
        node_id: The node ID
        repo_id: Optional repo_id override
        path: Optional path override

    Returns:
        dict: Pipeline kwargs with text_encoder_2 if nunchaku T5 encoder is found
    """
    from nunchaku import NunchakuT5EncoderModel
    from nunchaku.utils import get_precision
    from huggingface_hub import try_to_load_from_cache, hf_hub_download
    from nodetool.huggingface.huggingface_local_provider import (
        load_model,
    )

    if repo_id is None:
        repo_id = "mit-han-lab/nunchaku-t5"
    if path is None:
        path = f"awq-{get_precision()}-flux.1-t5xxl.safetensors"

    # Try to find nunchaku T5 encoder
    cache_path = try_to_load_from_cache(repo_id, path)
    if cache_path:
        return await load_model(
            context=context,
            model_id=cache_path,
            model_class=NunchakuT5EncoderModel,
            node_id=node_id,
            torch_dtype=torch.bfloat16,
        )
    else:
        log.info(
            "Downloading Nunchaku text_encoder %s/%s to cache",
            repo_id,
            path,
        )
        hf_hub_download(repo_id, path)
        cache_path = try_to_load_from_cache(repo_id, path)

        if not cache_path:
            raise ValueError(
                f"Downloading model {repo_id}/{path} from HuggingFace failed"
            )

    return await load_model(
        context=context,
        model_id=str(cache_path),
        model_class=NunchakuT5EncoderModel,
        node_id=node_id,
        torch_dtype=torch.bfloat16,
    )


async def get_nunchaku_transformer(
    context: ProcessingContext,
    model_class: type,
    node_id: str,
    repo_id: str,
    path: str
) -> Any | None:
    """
    Get transformer kwargs when using a nunchaku model.

    This function checks if the model is a nunchaku FLUX variant and if so,
    attempts to find and load the nunchaku SVDQ transformer from the HF cache.

    Args:
        context: The context object
        model_class: The model class to load
        node_id: The node ID
        precision: Optional precision override ('int4' or 'fp4'), auto-detected if None

    Returns:
        dict: Pipeline kwargs with transformer if nunchaku SVDQ transformer is found
    """
    from nunchaku.utils import get_precision
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from nodetool.huggingface.huggingface_local_provider import (
        load_model,
    )

    """Load FLUX pipeline using a Nunchaku SVDQ transformer file."""
    precision = get_precision()

    if "svdq" not in path.lower():
        raise ValueError(
            "Nunchaku Flux requires a transformer filename containing 'svdq'."
        )
    
    if precision not in path.lower():
        raise ValueError(
            f"Nunchaku Flux requires a transformer filename containing {precision}."
        )

    log.info(
        "Loading Nunchaku transformer from %s/%s (precision=%s)",
        repo_id,
        path,
        precision,
    )

    cache_path: str | None = None

    cache_path = try_to_load_from_cache(repo_id, path)
    if not cache_path:
        log.info(
            "Downloading Nunchaku transformer %s/%s to cache",
            repo_id,
            path,
        )
        hf_hub_download(
            repo_id,
            path,
        )
        cache_path = try_to_load_from_cache(
            repo_id,
            path,
        )
        if not cache_path:
            raise ValueError(
                f"Downloading model {repo_id}/{path} from HuggingFace failed"
            )

    transformer_identifier = cache_path or f"{repo_id}/{path}"

    transformer = await load_model(
        context=context,
        model_id=transformer_identifier,
        model_class=model_class,
        node_id=node_id,
        torch_dtype=torch.bfloat16,
    )
    return transformer
