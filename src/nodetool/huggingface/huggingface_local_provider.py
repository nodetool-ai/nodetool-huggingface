"""
HuggingFace local provider implementation.

This module implements the BaseProvider interface for locally cached HuggingFace models.
- Image models: Text2Image and ImageToImage using diffusion pipelines
  Supports both multi-file models (repo_id) and single-file models (repo_id:path.safetensors)
- TTS models: KokoroTTS and other HuggingFace TTS models
"""

import asyncio
import base64
import re
import threading
from queue import Queue
from typing import Any, AsyncGenerator, List, Literal, Set, Dict
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.integrations.huggingface.huggingface_models import (
    fetch_model_info,
    get_image_to_image_models_from_hf_cache,
    get_text_to_image_models_from_hf_cache,
    HF_FAST_CACHE,
)
from nodetool.types.job import JobUpdate
import numpy as np
from pydub import AudioSegment
import io
from io import BytesIO
from nodetool.types.model import UnifiedModel
from nodetool.workflows.processing_context import ProcessingContext
from PIL import Image
from nodetool.metadata.types import (
    ImageModel,
    Provider,
    TTSModel,
    Message,
)
from nodetool.workflows.types import Chunk, NodeProgress
from nodetool.metadata.types import ASRModel
from typing import List, Sequence, Any, AsyncIterator
from nodetool.config.logging_config import get_logger
import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from nodetool.ml.core.model_manager import ModelManager
from huggingface_hub import hf_hub_download, _CACHED_NO_EXIST
from nodetool.metadata.types import MessageTextContent, MessageImageContent
from nodetool.io.media_fetch import fetch_uri_bytes_and_mime_sync
from pydantic import BaseModel
import json


def _is_cuda_available() -> bool:
    """Safely check if CUDA is available, handling cases where PyTorch is not compiled with CUDA support."""
    try:
        # Check if cuda module exists
        if not hasattr(torch, "cuda"):
            return False
        # Try to check availability - this can raise RuntimeError if CUDA is not compiled
        return torch.cuda.is_available()
    except (RuntimeError, AttributeError):
        # PyTorch not compiled with CUDA support or other CUDA-related error
        return False


import os
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from nodetool.huggingface.nunchaku_utils import (
    get_nunchaku_text_encoder,
    get_nunchaku_transformer,
)
from nunchaku import NunchakuFluxTransformer2dModel

# Import specific pipeline classes
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import (
    StableDiffusion3Img2ImgPipeline,
)
from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    TextStreamer,
    pipeline as create_pipeline,
)
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
)
from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from nunchaku import NunchakuQwenImageTransformer2DModel

from nodetool.metadata.types import HFTextToSpeech
from pydub import AudioSegment
from io import BytesIO
from nodetool.metadata.types import HFTextToSpeech

from nodetool.workflows.recommended_models import get_recommended_models
from nodetool.metadata.types import LanguageModel, VideoRef
from transformers.pipelines import pipeline
from pathlib import Path
from typing import TypeVar
import os

ALLOW_DOWNLOAD_ENV = "NODETOOL_HF_ALLOW_DOWNLOAD"

T = TypeVar("T")

log = get_logger(__name__)
_PREFERRED_HF_DEVICE = "mps"
_ALLOW_DOWNLOAD_VALUES = {"1", "true", "yes", "on"}


def _is_mps_available() -> bool:
    """Detect whether the Apple Metal backend is available."""
    try:
        return (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def _is_node_model(model_id: str, model_path: str | None, node_cls: Any) -> bool:
    """Check if the model matches any of the node's recommended models."""
    for rec in node_cls.get_recommended_models():
        if rec.repo_id == model_id:
            # If recommended model has a specific path, it must match
            if rec.path:
                if rec.path == model_path:
                    return True
            # If recommended model has no path (repo-level),
            # it matches if model_path is None (dir) or if we accept any file in repo.
            # Usually strict match on repo_id is enough for repo-level recommendations.
            else:
                return True
    return False


def _resolve_hf_device(
    context: ProcessingContext,
    requested_device: str | None = None,
) -> str:
    """
    Force HuggingFace workloads onto the MPS device when available.

    Falls back to an explicitly requested device or CPU when Apple Metal is not
    present so execution can continue on other platforms.
    """
    if _is_mps_available():
        if requested_device and requested_device != _PREFERRED_HF_DEVICE:
            log.debug(
                "Ignoring requested device %s in favor of %s",
                requested_device,
                _PREFERRED_HF_DEVICE,
            )
        return _PREFERRED_HF_DEVICE

    fallback = None
    if requested_device and requested_device != _PREFERRED_HF_DEVICE:
        fallback = requested_device
    elif context.device and context.device != _PREFERRED_HF_DEVICE:
        fallback = context.device

    if fallback:
        return fallback

    fallback = "cuda" if _is_cuda_available() else "cpu"
    log.warning(
        "MPS backend unavailable; falling back to %s for HuggingFace execution",
        fallback,
    )
    return fallback


def _allow_downloads() -> bool:
    """Whether automatic Hugging Face downloads are permitted."""
    return os.getenv(ALLOW_DOWNLOAD_ENV, "").lower() in _ALLOW_DOWNLOAD_VALUES


async def _ensure_file_cached(
    repo_id: str,
    file_path: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """
    Guarantee a file exists locally, optionally downloading when allowed.
    Raises ValueError if downloads are blocked and the file is missing.
    """
    cache_path = await HF_FAST_CACHE.resolve(repo_id, file_path)
    if cache_path:
        return cache_path

    if _allow_downloads():
        log.info(
            "Downloading %s/%s because NODETOOL_HF_ALLOW_DOWNLOAD is enabled",
            repo_id,
            file_path,
        )
        # hf_hub_download is blocking
        await asyncio.to_thread(
            hf_hub_download,
            repo_id,
            file_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        # Check again using fast cache
        cache_path = await HF_FAST_CACHE.resolve(repo_id, file_path)
        if cache_path:
            return cache_path
        # If fast cache still doesn't see it (race condition or weird FS), fallback to standard
        return None

    raise ValueError(f"Model {repo_id}/{file_path} must be downloaded first")


def _ensure_model_on_device(model: Any, device: str | None) -> Any:
    """Move HF pipelines/models onto the requested device when possible."""
    if not device:
        return model

    move_fn = getattr(model, "to", None)
    if callable(move_fn):
        try:
            moved = move_fn(device)
            if moved is not None:
                return moved
        except Exception as exc:
            log.warning(
                "Failed to move %s to %s: %s",
                model.__class__.__name__,
                device,
                exc,
            )
    return model


def _is_vram_error(exc: Exception) -> bool:
    """Check if an exception indicates insufficient VRAM/GPU memory."""
    error_msg = str(exc).lower()
    vram_indicators = [
        "out of memory",
        "cuda out of memory",
        "not enough gpu ram",
        "some modules are dispatched on the cpu or the disk",
        "make sure you have enough gpu ram",
        "oom",
        "cudnn_status_not_supported",
        "cublas_status_alloc_failed",
    ]
    return any(indicator in error_msg for indicator in vram_indicators)


async def load_pipeline(
    node_id: str,
    context: ProcessingContext,
    pipeline_task: str,
    model_id: Any,
    device: str | None = None,
    torch_dtype: torch.dtype | None = None,
    skip_cache: bool = False,
    **kwargs: Any,
):
    """Load a HuggingFace pipeline model.

    If loading fails due to insufficient VRAM, this function will attempt to
    free cached models and retry once.
    """
    if model_id == "" or model_id is None:
        raise ValueError("Please select a model")

    cache_key = f"{model_id}_{pipeline_task}"

    cached_model = ModelManager.get_model(cache_key)
    if cached_model:
        target_device = _resolve_hf_device(context, device or context.device)
        return _ensure_model_on_device(cached_model, target_device)

    # target_device = _resolve_hf_device(context, device or context.device)

    if (
        isinstance(model_id, str)
        and not skip_cache
        and not Path(model_id).expanduser().exists()
    ):
        repo_id_for_cache = model_id
        revision = kwargs.get("revision")
        cache_dir = kwargs.get("cache_dir")

        if "@" in repo_id_for_cache and revision is None:
            repo_id_for_cache, revision = repo_id_for_cache.rsplit("@", 1)

        cache_checked = False
        for candidate in ("model_index.json", "config.json"):
            try:
                cache_path = await _ensure_file_cached(
                    repo_id_for_cache,
                    candidate,
                    revision=revision,
                    cache_dir=cache_dir,
                )
            except Exception:
                cache_path = None

            if cache_path:
                cache_checked = True
                break

        if not cache_checked:
            await _ensure_file_cached(
                repo_id_for_cache,
                "config.json",
                revision=revision,
                cache_dir=cache_dir,
            )

    context.post_message(
        JobUpdate(
            status="running",
            message=f"Loading pipeline {type(model_id) == str and model_id or pipeline_task} from HuggingFace",
        )
    )
    if "token" not in kwargs:
        kwargs["token"] = await context.get_secret("HF_TOKEN")

    def _create_pipeline():
        return pipeline(
            pipeline_task,  # type: ignore
            model=model_id,
            torch_dtype=torch_dtype,
            # device=target_device,
            **kwargs,
        )

    # First attempt to load the pipeline
    try:
        model = _create_pipeline()
    except (ValueError, RuntimeError) as exc:
        # Also catch torch.cuda.OutOfMemoryError if available
        if not _is_vram_error(exc):
            raise

        # VRAM error detected - try to free memory and retry
        log.warning(
            f"VRAM error while loading pipeline {model_id}: {exc}. "
            "Attempting to free cached models and retry..."
        )
        context.post_message(
            JobUpdate(
                status="running",
                message=f"Freeing VRAM and retrying load of {model_id}...",
            )
        )

        # Aggressively free VRAM by clearing cached models
        ModelManager.free_vram_if_needed(
            reason=f"VRAM error loading pipeline {model_id}",
            aggressive=True,
        )

        # Retry loading after freeing memory
        try:
            model = _create_pipeline()
            log.info(f"Successfully loaded pipeline {model_id} after freeing VRAM")
        except Exception as retry_exc:
            log.error(
                f"Failed to load pipeline {model_id} even after freeing VRAM: {retry_exc}"
            )
            raise

    #  model = _ensure_model_on_device(model, target_device)

    ModelManager.set_model(node_id, cache_key, model)
    return model  # type: ignore


async def load_model(
    node_id: str,
    context: ProcessingContext,
    model_class: type[T],
    model_id: str,
    variant: str | None = None,
    torch_dtype: torch.dtype | None = None,
    path: str | None = None,
    skip_cache: bool = False,
    cache_key: str | None = None,
    **kwargs: Any,
) -> T:
    """Load a HuggingFace model.

    If loading fails due to insufficient VRAM, this function will attempt to
    free cached models and retry once.
    """
    if model_id == "":
        raise ValueError("Please select a model")

    target_device = _resolve_hf_device(context, context.device)

    log.info(f"Loading model {model_id}/{path} from {target_device}")

    # Always ensure cache_key is set
    if cache_key is None:
        cache_key = f"{model_id}_{model_class.__name__}_{path}"

    if not skip_cache:
        cached_model = ModelManager.get_model(cache_key)
        if cached_model:
            return _ensure_model_on_device(cached_model, target_device)

    async def _do_load() -> T:
        """Inner function that performs the actual model loading."""
        if path:
            cache_path = await HF_FAST_CACHE.resolve(model_id, path)
            if not cache_path:
                raise ValueError(
                    f"Download model {model_id}/{path} first from recommended models"
                )

            log.info(f"Loading model {model_id} from {cache_path}")
            context.post_message(
                JobUpdate(
                    status="running",
                    message=f"Loading model {model_id}",
                )
            )

            if hasattr(model_class, "from_single_file"):
                model = model_class.from_single_file(  # type: ignore
                    cache_path,
                    torch_dtype=torch_dtype,
                    variant=variant,
                    **kwargs,
                )
            else:
                # Fallback to from_pretrained for classes without from_single_file
                model = model_class.from_pretrained(  # type: ignore
                    model_id,
                    torch_dtype=torch_dtype,
                    variant=variant,
                    **kwargs,
                )
        else:
            log.info(f"Loading model {model_id} from HuggingFace")
            context.post_message(
                JobUpdate(
                    status="running",
                    message=f"Loading model {model_id} from HuggingFace",
                )
            )
            if "token" not in kwargs:
                kwargs["token"] = await context.get_secret("HF_TOKEN")

            model = model_class.from_pretrained(  # type: ignore
                model_id,
                torch_dtype=torch_dtype,
                variant=variant,
                **kwargs,
            )
        return model

    # First attempt to load the model
    try:
        model = await _do_load()
    except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:  # type: ignore[attr-defined]
        if not _is_vram_error(exc):
            raise

        # VRAM error detected - try to free memory and retry
        log.warning(
            f"VRAM error while loading model {model_id}: {exc}. "
            "Attempting to free cached models and retry..."
        )
        context.post_message(
            JobUpdate(
                status="running",
                message=f"Freeing VRAM and retrying load of {model_id}...",
            )
        )

        # Aggressively free VRAM by clearing cached models
        ModelManager.free_vram_if_needed(
            reason=f"VRAM error loading {model_id}",
            aggressive=True,
        )

        # Retry loading after freeing memory
        try:
            model = await _do_load()
            log.info(f"Successfully loaded model {model_id} after freeing VRAM")
        except Exception as retry_exc:
            log.error(
                f"Failed to load model {model_id} even after freeing VRAM: {retry_exc}"
            )
            raise

    model = _ensure_model_on_device(model, target_device)
    ModelManager.set_model(node_id, cache_key, model)
    return model


async def _detect_cached_variant(repo_id: str) -> str | None:
    """Detect a cached diffusers variant (e.g., fp16) for a given repo.

    Heuristic: locate any cached file for the repo, then scan its snapshot
    folder for files containing ".fp16." in the filename. If found, return
    "fp16". Otherwise return None.

    This avoids passing a non-existent variant to diffusers.from_pretrained,
    which would error if the repo does not publish that variant.
    """
    # Try a few common files to retrieve the snapshot directory from cache
    probe_files = [
        "model_index.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
    ]
    snapshot_dir: str | None = None
    for fname in probe_files:
        p = await HF_FAST_CACHE.resolve(repo_id, fname)
        if isinstance(p, str):
            snapshot_dir = os.path.dirname(p)
            break
        if p is _CACHED_NO_EXIST:
            # This file not in cache; try next
            continue

    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        return None

    # Walk snapshot dir and look for any *.fp16.* file
    for root, _, files in os.walk(snapshot_dir):
        for f in files:
            if ".fp16." in f:
                return "fp16"

    return None


def _detect_flux_variant(repo_id: str, file_path: str | None) -> str:
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


def _flux_variant_to_base_model_id(variant: str) -> str:
    """Map detected variant names to canonical FLUX repos."""
    mapping = {
        "schnell": "black-forest-labs/FLUX.1-schnell",
        "fill": "black-forest-labs/FLUX.1-Fill-dev",
        "canny": "black-forest-labs/FLUX.1-Canny-dev",
        "depth": "black-forest-labs/FLUX.1-Depth-dev",
        "dev": "black-forest-labs/FLUX.1-dev",
    }
    return mapping.get(variant, "black-forest-labs/FLUX.1-dev")


def _is_nunchaku_transformer(repo_id: str, file_path: str | None) -> bool:
    """Detect Nunchaku FLUX transformer files."""
    if not file_path:
        return False
    repo_lower = repo_id.lower()
    return (
        "nunchaku" in repo_lower
        and "flux" in repo_lower
        and "svdq" in file_path.lower()
    )


def _node_identifier(node_id: str | None, repo_id: str) -> str:
    """Derive a unique identifier for caching based on node ID or repo."""
    return node_id or repo_id


async def load_nunchaku_flux_pipeline(
    context: ProcessingContext,
    repo_id: str,
    transformer_path: str,
    node_id: str | None,
    pipeline_class: Any | None = None,
    cache_key: str | None = None,
) -> Any:
    """Load a FLUX pipeline with a Nunchaku transformer/text encoder pair."""
    pipeline_class = pipeline_class or FluxPipeline
    pipeline_name = pipeline_class.__name__

    if cache_key:
        cached_pipeline = ModelManager.get_model(cache_key)
        if cached_pipeline:
            return cached_pipeline
    variant = _detect_flux_variant(repo_id, transformer_path)
    base_model_id = _flux_variant_to_base_model_id(variant)
    hf_token = await context.get_secret("HF_TOKEN")
    if variant != "schnell" and not hf_token:
        raise ValueError(
            f"Flux-{variant} is a gated model, please set the HF_TOKEN in Nodetool settings and accept the terms of use for the model: "
            f"https://huggingface.co/{base_model_id}"
        )

    torch_dtype = torch.bfloat16 if variant in ["schnell", "dev"] else torch.float16
    node_key = _node_identifier(node_id, repo_id)

    transformer = await get_nunchaku_transformer(
        context=context,
        model_class=NunchakuFluxTransformer2dModel,
        node_id=node_key,
        repo_id=repo_id,
        path=transformer_path,
    )
    transformer.set_attention_impl("nunchaku-fp16")

    text_encoder = await get_nunchaku_text_encoder(context, node_key)

    pipeline_kwargs: dict[str, Any] = {
        "transformer": transformer,
        "torch_dtype": torch_dtype,
        "token": hf_token,
    }
    if text_encoder is not None:
        pipeline_kwargs["text_encoder_2"] = text_encoder

    try:
        pipeline = pipeline_class.from_pretrained(base_model_id, **pipeline_kwargs)
        if cache_key and node_id:
            ModelManager.set_model(node_id, cache_key, pipeline)
        return pipeline
    except torch.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        raise ValueError(
            "VRAM out of memory while loading Flux with the Nunchaku transformer. "
            "Try enabling 'CPU offload' or reduce image size/steps."
        ) from exc


async def load_nunchaku_qwen_pipeline(
    context: ProcessingContext,
    repo_id: str,
    transformer_path: str,
    node_id: str | None,
    pipeline_class: Any,
    base_model_id: str,
    cache_key: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Load a Qwen pipeline with a Nunchaku transformer and quantized text encoder.

    Args:
        context: Processing context
        repo_id: Repository ID for the Nunchaku transformer
        transformer_path: Path to the transformer file
        node_id: Node ID for caching
        pipeline_class: The pipeline class to use (QwenImagePipeline or QwenImageEditPipeline)
        base_model_id: Base model ID for loading pipeline components
        cache_key: Optional cache key for the pipeline
        torch_dtype: The torch dtype to use
    """
    from nunchaku.utils import get_gpu_memory
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

    hf_token = await context.get_secret("HF_TOKEN")
    torch_dtype = torch.bfloat16
    if cache_key is None:
        cache_key = f"{repo_id}/{transformer_path}"

    cached_pipeline = ModelManager.get_model(cache_key)
    if cached_pipeline:
        return cached_pipeline

    try:
        # Load Nunchaku transformer
        transformer = await get_nunchaku_transformer(
            context=context,
            model_class=NunchakuQwenImageTransformer2DModel,
            node_id=node_id,
            repo_id=repo_id,
            path=transformer_path,
            device=context.device,
        )
        pipeline = pipeline_class.from_pretrained(
            base_model_id,
            transformer=transformer,
            torch_dtype=torch_dtype,
            token=hf_token,
        )
        pipeline.enable_model_cpu_offload()
        if get_gpu_memory() > 18:
            log.info("Enabling model CPU offload")
        else:
            log.info("Enabling model per-layer offloading")
        # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
        transformer.set_offload(
            True, use_pin_memory=True, num_blocks_on_gpu=1
        )  # increase num_blocks_on_gpu if you have more VRAM
        pipeline._exclude_from_cpu_offload.append("transformer")
        pipeline.enable_sequential_cpu_offload()

        if cache_key and node_id:
            ModelManager.set_model(node_id, text_encoder_repo, text_encoder)
        return pipeline
    except torch.OutOfMemoryError as exc:
        raise ValueError(
            "VRAM out of memory while loading Qwen with the Nunchaku transformer."
        ) from exc


@register_provider(Provider.HuggingFace)
class HuggingFaceLocalProvider(BaseProvider):
    """Local provider for HuggingFace models using cached diffusion pipelines."""

    provider_name = "hf_inference"

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables needed when running inside Docker."""
        # The nodes will handle HF_TOKEN internally
        return {}

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
    ) -> ImageBytes:
        """Generate an image from a text prompt using HuggingFace diffusion models.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            node_id: Optional node ID for progress tracking
        Returns:
            Raw image bytes as PNG

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        from nodetool.nodes.huggingface.image_to_image import pipeline_progress_callback

        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace image generation"
            )

        pipeline = ModelManager.get_model(model_id)

        if not pipeline:
            log.info(f"Loading text-to-image pipeline: {params.model.id}")
            use_cpu_offload = False

            # Check if model_id is in "repo_id:path" format (single-file model)
            if params.model.path:
                # Verify the file is cached locally
                cache_path = await HF_FAST_CACHE.resolve(
                    params.model.id, params.model.path
                )
                if not cache_path:
                    cache_path = await _ensure_file_cached(
                        params.model.id,
                        params.model.path,
                        revision=params.revision,
                        cache_dir=params.cache_dir,
                    )

                model_info = await fetch_model_info(params.model.id)
                if model_info is None:
                    raise ValueError(f"Model {params.model.id} not found")

                if model_info.pipeline_tag != "text-to-image":
                    raise ValueError(
                        f"Model {params.model.id} is not a text-to-image model"
                    )

                if not model_info.tags:
                    raise ValueError(f"Model {params.model.id} has no tags")

                from nodetool.nodes.huggingface.text_to_image import Flux, QwenImage

                if _is_node_model(params.model.id, params.model.path, Flux):
                    if _is_nunchaku_transformer(params.model.id, params.model.path):
                        pipeline = await load_nunchaku_flux_pipeline(
                            context=context,
                            repo_id=params.model.id,
                            transformer_path=params.model.path,
                            node_id=node_id,
                        )
                    else:
                        # Standard Flux loading
                        pipeline = FluxPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.bfloat16
                                if _is_cuda_available()
                                else torch.float32
                            ),
                        )
                elif _is_node_model(params.model.id, params.model.path, QwenImage):
                    if _is_nunchaku_transformer(params.model.id, params.model.path):
                        pipeline = await load_nunchaku_qwen_pipeline(
                            model_id=params.model.id,
                            path=params.model.path,
                            context=context,
                            torch_dtype=torch.bfloat16,
                            node_id=node_id,
                        )
                        use_cpu_offload = True
                    else:
                        pipeline = QwenImagePipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=torch.bfloat16,
                        )
                        use_cpu_offload = True
                else:
                    model_type = model_info.pipeline_tag or "unknown"

                    # Load pipeline from single file based on model type
                    if "diffusers:StableDiffusionXLPipeline" in model_info.tags:
                        pipeline = StableDiffusionXLPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16 if _is_cuda_available() else torch.float32
                            ),
                        )
                    elif "diffusers:StableDiffusionPipeline" in model_info.tags:
                        pipeline = StableDiffusionPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16 if _is_cuda_available() else torch.float32
                            ),
                        )
                    elif "diffusers:StableDiffusion3Pipeline" in model_info.tags:
                        pipeline = StableDiffusion3Pipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16 if _is_cuda_available() else torch.float32
                            ),
                        )
                    elif "flux" in model_info.tags:
                        pipeline = FluxPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.bfloat16
                                if _is_cuda_available()
                                else torch.float32
                            ),
                        )
                    else:
                        raise ValueError(
                            f"Unsupported single-file model type: {model_type}"
                        )
            else:
                # Load pipeline from multi-file model (standard format)
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    params.model.id,
                    torch_dtype=(
                        torch.float16 if _is_cuda_available() else torch.float32
                    ),
                    variant=await _detect_cached_variant(params.model.id),
                )

            if not use_cpu_offload:
                pipeline.to(context.device)

            # Cache the pipeline
            ModelManager.set_model(node_id, cache_key, pipeline)

        # Set up generator for reproducibility
        generator = None
        if params.seed is not None and params.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(params.seed)

        # Progress callback
        num_steps = params.num_inference_steps or 50

        # Generate the image off the event loop
        def _run_pipeline_sync():
            return pipeline(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or "",
                num_inference_steps=num_steps,
                guidance_scale=params.guidance_scale or 7.5,
                width=params.width or 512,
                height=params.height or 512,
                generator=generator,
                callback_on_step_end=pipeline_progress_callback(
                    node_id=node_id, total_steps=num_steps, context=context
                ),  # type: ignore
                callback_on_step_end_tensor_inputs=["latents"],
            )

        output = await asyncio.to_thread(_run_pipeline_sync)

        # Get the generated image
        pil_image = output.images[0]  # pyright: ignore[reportAttributeAccessIssue]

        # Convert PIL Image to bytes
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        return image_bytes

    async def image_to_image(
        self,
        image: ImageBytes,
        params: ImageToImageParams,
        context: ProcessingContext | None = None,
        timeout_s: int | None = None,
        node_id: str | None = None,
    ) -> ImageBytes:
        """Transform an image based on a text prompt using HuggingFace diffusion models.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            node_id: Optional node ID for progress tracking
        Returns:
            Raw image bytes as PNG

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace image generation"
            )

        # Convert input image bytes to PIL Image
        pil_image = Image.open(BytesIO(image))

        # Get or load the pipeline
        pipeline = ModelManager.get_model(model_id)

        if not pipeline:
            log.info(f"Loading image-to-image pipeline: {params.model.id}")
            use_cpu_offload = False

            # Check if model_id is in "repo_id:path" format (single-file model)
            if params.model.path:
                # Verify the file is cached locally
                cache_path = await HF_FAST_CACHE.resolve(
                    params.model.id, params.model.path
                )
                if not cache_path:
                    raise ValueError(
                        await _ensure_file_cached(
                            params.model.id,
                            params.model.path,
                            revision=params.revision,
                            cache_dir=params.cache_dir,
                        )
                    )

                from nodetool.nodes.huggingface.image_to_image import (
                    FluxFill,
                    QwenImageEdit,
                )

                if _is_node_model(params.model.id, params.model.path, FluxFill):
                    if _is_nunchaku_transformer(params.model.id, params.model.path):
                        # Flux Fill Nunchaku support
                        pipeline = await load_nunchaku_flux_pipeline(
                            context=context,
                            repo_id=params.model.id,
                            transformer_path=params.model.path,
                            node_id=node_id,
                            pipeline_class=FluxFillPipeline,
                        )
                    else:
                        # Standard Flux Fill
                        pipeline = FluxFillPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.bfloat16
                                if _is_cuda_available()
                                else torch.float32
                            ),
                        )
                elif _is_node_model(params.model.id, params.model.path, QwenImageEdit):
                    if _is_nunchaku_transformer(params.model.id, params.model.path):
                        pipeline = await load_nunchaku_qwen_pipeline(
                            context=context,
                            repo_id=params.model.id,
                            transformer_path=params.model.path,
                            node_id=node_id,
                            pipeline_class=QwenImageEditPipeline,
                        )
                    else:
                        pipeline = QwenImageEditPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=torch.float16,
                        )
                elif _is_nunchaku_transformer(params.model.id, params.model.path):
                    pipeline = await load_nunchaku_flux_pipeline(
                        context=context,
                        repo_id=params.model.id,
                        transformer_path=params.model.path,
                        node_id=node_id,
                    )
                else:
                    # Verify the file is cached locally
                    cache_path = await HF_FAST_CACHE.resolve(
                        params.model.id, params.model.path
                    )
                    if not cache_path:
                        raise ValueError(
                            _ensure_file_cached(
                                params.model.id,
                                params.model.path,
                                revision=params.revision,
                                cache_dir=params.cache_dir,
                            )
                        )

                    model_info = await fetch_model_info(params.model.id)
                    if model_info is None:
                        raise ValueError(f"Model {params.model.id} not found")

                    if model_info.pipeline_tag != "image-to-image":
                        # Some pipelines like FluxFill might be classified differently or we want to allow it
                        pass

                    if not model_info.tags:
                        raise ValueError(f"Model {params.model.id} has no tags")

                    # Load pipeline from single file based on model type
                    if "diffusers:StableDiffusionPipeline" in model_info.tags:
                        pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16 if _is_cuda_available() else torch.float32
                            ),
                        )
                    elif "diffusers:StableDiffusionXLPipeline" in model_info.tags:
                        pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16 if _is_cuda_available() else torch.float32
                            ),
                        )
                    elif "diffusers:StableDiffusion3Pipeline" in model_info.tags:
                        pipeline = StableDiffusion3Img2ImgPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16 if _is_cuda_available() else torch.float32
                            ),
                        )
                    elif "flux" in model_info.tags:
                        pipeline = FluxImg2ImgPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.bfloat16
                                if _is_cuda_available()
                                else torch.float32
                            ),
                        )
                    else:
                        raise ValueError(
                            f"Unsupported single-file model type: {model_info.pipeline_tag}"
                        )
            else:
                # Load pipeline from multi-file model (standard format)
                pipeline = AutoPipelineForImage2Image.from_pretrained(
                    params.model.id,
                    torch_dtype=(
                        torch.float16 if _is_cuda_available() else torch.float32
                    ),
                    variant=await _detect_cached_variant(params.model.id),
                )

            assert pipeline is not None
            if not use_cpu_offload:
                pipeline.to(context.device)

            # Cache the pipeline
            ModelManager.set_model(node_id, cache_key, pipeline)

        # Set up generator for reproducibility
        generator = None
        if params.seed is not None and params.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(params.seed)

        # Progress callback
        num_steps = params.num_inference_steps or 25

        # Generate the image off the event loop
        def _run_pipeline_sync():
            return pipeline(
                prompt=params.prompt,
                image=pil_image,
                negative_prompt=params.negative_prompt or "",
                strength=params.strength or 0.8,
                num_inference_steps=num_steps,
                guidance_scale=params.guidance_scale or 7.5,
                generator=generator,
                callback_on_step_end=pipeline_progress_callback(
                    node_id=node_id, total_steps=num_steps, context=context
                ),  # type: ignore
                callback_on_step_end_tensor_inputs=["latents"],
            )

        output = await asyncio.to_thread(_run_pipeline_sync)

        # Get the generated image
        pil_output = output.images[0]  # pyright: ignore[reportAttributeAccessIssue]

        # Convert PIL Image to bytes
        img_buffer = BytesIO()
        pil_output.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        self.usage["total_requests"] += 1
        self.usage["total_images"] += 1

        return image_bytes

    async def text_to_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        speed: float = 1.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:  # Returns np.ndarray[np.int16]
        """Generate speech audio from text using local HuggingFace TTS models.

        Supports TTS models:
        - Kokoro models (hexgrad/Kokoro-82M) - streaming with voice support

        Args:
            text: Text to convert to speech
            model: Model repository ID (e.g., "hexgrad/Kokoro-82M")
            voice: Voice preset (only for Kokoro models, e.g., "af_heart")
            speed: Speech speed multiplier (0.5 to 2.0, only for Kokoro)
            timeout_s: Optional timeout in seconds
            context: Processing context
            **kwargs: Additional arguments (lang_code for Kokoro)

        Yields:
            numpy.ndarray: Int16 audio chunks at 24kHz mono

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        from nodetool.nodes.huggingface.text_to_speech import KokoroTTS
        from nodetool.nodes.huggingface.text_to_speech import TextToSpeech

        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace TTS generation"
            )

        # Determine which TTS node to use based on model ID
        model_lower = model.lower()

        if "kokoro" in model_lower:
            # Map voice string to Voice enum
            voice_value = voice or "af_heart"  # Default voice
            lang_code = kwargs.get("lang_code", "a")  # Default to American English

            node = KokoroTTS(
                model=HFTextToSpeech(repo_id=model),
                text=text,
                voice=KokoroTTS.Voice(voice_value),
                speed=max(0.5, min(2.0, speed)),
                lang_code=KokoroTTS.LanguageCode(lang_code),
            )

            # Preload model
            await node.preload_model(context)

            # Stream chunks using gen_process
            async for output in node.gen_process(context):
                # Only yield chunk data (not the final AudioRef)
                chunk = output.get("chunk")
                if chunk and chunk.content and not chunk.done:
                    # Decode base64 chunk to numpy array
                    audio_bytes = base64.b64decode(chunk.content)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    yield audio_array

        else:
            node = TextToSpeech(
                model=HFTextToSpeech(repo_id=model),
                text=text,
            )

            # Preload model
            await node.preload_model(context)

            # Process to get audio
            audio_ref = await node.process(context)

            # Convert AudioRef to numpy array
            audio_bytes = await context.asset_to_bytes(audio_ref)
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Resample to 24kHz if needed
            if audio.frame_rate != 24000:
                audio = audio.set_frame_rate(24000)

            # Ensure 16-bit sample width
            if audio.sample_width != 2:
                audio = audio.set_sample_width(2)

            # Convert to numpy array
            audio_array = np.array(audio.get_array_of_samples(), dtype=np.int16)

            # Yield in chunks (4096 samples at a time)
            chunk_size = 4096
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i : i + chunk_size]
                yield chunk

    async def automatic_speech_recognition(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio to text using HuggingFace Whisper models.

        Args:
            audio: Input audio as bytes (various formats supported)
            model: Model repository ID (e.g., "openai/whisper-large-v3")
            language: Optional ISO-639-1 language code to improve accuracy
            prompt: Optional text to guide the model's style (initial_prompt)
            temperature: Sampling temperature between 0 and 1 (default 0)
            timeout_s: Optional timeout in seconds
            context: Processing context (required)
            **kwargs: Additional parameters (return_timestamps, chunk_length_s)

        Returns:
            Transcribed text from the audio

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If transcription fails
        """
        if context is None:
            raise ValueError("ProcessingContext is required for HuggingFace ASR")

        log.debug(f"Transcribing audio with HuggingFace Whisper model: {model}")

        # Get or load the pipeline
        asr_pipeline = ModelManager.get_model(model_id)

        if not asr_pipeline:
            log.info(f"Loading automatic speech recognition pipeline: {model}")

            # Determine torch dtype based on device
            torch_dtype = torch.float16 if _is_cuda_available() else torch.float32

            # Load model using helper
            hf_model = await load_model(
                node_id=cache_key,
                context=context,
                model_class=AutoModelForSpeechSeq2Seq,
                model_id=model,
                torch_dtype=torch_dtype,
                skip_cache=False,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )

            # Load processor
            processor = AutoProcessor.from_pretrained(model)

            # Create pipeline
            asr_pipeline = create_pipeline(
                "automatic-speech-recognition",
                model=hf_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=context.device,
            )

            # Cache the pipeline
            ModelManager.set_model(node_id, cache_key, asr_pipeline)

        audio_segment = AudioSegment.from_file(BytesIO(audio))
        # Whisper expects 16kHz mono audio
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

        # Convert to numpy array (float32)
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        # Normalize to [-1, 1] range
        samples = samples / (2**15)

        # Build pipeline kwargs
        pipeline_kwargs: dict[str, Any] = {
            "return_timestamps": kwargs.get("return_timestamps", False),
            "chunk_length_s": kwargs.get("chunk_length_s", 30.0),
            "generate_kwargs": {},
        }

        # Add language if specified
        if language:
            pipeline_kwargs["generate_kwargs"]["language"] = language

        # Add prompt if specified (Whisper uses initial_prompt)
        if prompt:
            pipeline_kwargs["generate_kwargs"]["initial_prompt"] = prompt

        # Add temperature if non-zero
        if temperature != 0.0:
            pipeline_kwargs["generate_kwargs"]["temperature"] = temperature

        # Run transcription in thread to avoid blocking
        def _transcribe():
            return asr_pipeline(samples, **pipeline_kwargs)

        result = await asyncio.to_thread(_transcribe)

        # Extract text from result
        if isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)

        log.debug(f"Transcription complete: {len(text)} characters")

        return text

    async def text_to_video(
        self,
        prompt: str,
        model: str,
        negative_prompt: str | None = None,
        num_frames: int = 49,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        height: int = 480,
        width: int = 720,
        fps: int = 16,
        seed: int | None = None,
        max_sequence_length: int = 512,
        enable_cpu_offload: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
        **kwargs: Any,
    ) -> VideoRef:
        """Generate a video from a text prompt using HuggingFace text-to-video models.

        Supports Wan text-to-video models (Wan-AI/Wan2.2-T2V-A14B-Diffusers, etc.).

        Args:
            prompt: Text description of the desired video
            model: Model repository ID (e.g., "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            negative_prompt: Text describing what to avoid in the video
            num_frames: Number of frames to generate (16-129)
            guidance_scale: Scale for classifier-free guidance (1.0-20.0)
            num_inference_steps: Number of denoising steps (1-100)
            height: Height of the generated video in pixels
            width: Width of the generated video in pixels
            fps: Frames per second for the output video
            seed: Random seed for generation (None for random)
            max_sequence_length: Maximum sequence length in encoded prompt
            enable_cpu_offload: Enable CPU offload to reduce VRAM usage
            enable_vae_slicing: Enable VAE slicing to reduce VRAM usage
            enable_vae_tiling: Enable VAE tiling for large videos
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            **kwargs: Additional arguments

        Returns:
            VideoRef to the generated video

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        from nodetool.nodes.huggingface.text_to_video import (
            AutoencoderKLWan,
            WanPipeline,
        )
        from nodetool.nodes.huggingface.text_to_video import pipeline_progress_callback

        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace text-to-video generation"
            )

        # Get or load the pipeline
        pipeline = ModelManager.get_model(model_id)

        if not pipeline:
            log.info(f"Loading text-to-video pipeline: {model_id}")

            # Load VAE first
            vae = await asyncio.to_thread(
                AutoencoderKLWan.from_pretrained,
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )

            # Load WanPipeline
            pipeline = await asyncio.to_thread(
                WanPipeline.from_pretrained,
                model_id,
                torch_dtype=torch.bfloat16,
                vae=vae,
            )

            # Apply memory optimization settings
            if enable_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
                if hasattr(pipeline, "enable_sequential_cpu_offload"):
                    pipeline.enable_sequential_cpu_offload()

            if hasattr(pipeline, "enable_attention_slicing"):
                try:
                    pipeline.enable_attention_slicing()
                except Exception:
                    pass

            if enable_vae_slicing and hasattr(pipeline, "vae"):
                try:
                    pipeline.vae.enable_slicing()
                except Exception:
                    pass

            if enable_vae_tiling and hasattr(pipeline, "vae"):
                try:
                    pipeline.vae.enable_tiling()
                except Exception:
                    pass

            try:
                if hasattr(pipeline, "unet") and hasattr(
                    pipeline.unet, "enable_xformers_memory_efficient_attention"
                ):
                    pipeline.unet.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            # Cache the pipeline
            ModelManager.set_model(node_id, cache_key, pipeline)

        # Set up generator for reproducibility
        generator = None
        if seed is not None and seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # Generate the video off the event loop
        def _run_pipeline_sync():
            return pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                max_sequence_length=max_sequence_length,
                callback_on_step_end=pipeline_progress_callback(
                    node_id=node_id, total_steps=num_inference_steps, context=context
                ),  # type: ignore
            )

        output = await asyncio.to_thread(_run_pipeline_sync)

        # Get the generated frames
        frames = output.frames[0]  # pyright: ignore[reportAttributeAccessIssue]

        # Convert frames to video
        video_ref = await context.video_from_frames(frames, fps=fps)  # type: ignore

        return video_ref

    async def get_available_language_models(self) -> List[LanguageModel]:
        """Get available HuggingFace language models.

        Returns models available in the local HuggingFace cache.

        Returns:
            List of LanguageModel instances for HuggingFace models
        """
        models = await get_hf_language_models_from_hf_cache()
        log.debug(f"Found {len(models)} HuggingFace models in HF cache")
        return models

    async def get_available_image_models(self) -> List[ImageModel]:
        """Get available HuggingFace image models.

        Returns both multi-file models and single-file models (.safetensors).
        Single-file models use format "repo_id:path".
        """
        # Get multi-file models
        text_to_image_models = await get_text_to_image_models_from_hf_cache()
        image_to_image_models = await get_image_to_image_models_from_hf_cache()

        # Get single-file models
        return text_to_image_models + image_to_image_models

    async def get_available_tts_models(self) -> List[TTSModel]:
        """Get available HuggingFace TTS models from recommended models.

        Returns TTS models based on the recommended models from the TTS nodes:
        - Bark models (general TTS)
        - KokoroTTS models (multi-language with voices)
        - Generic TextToSpeech models (MMS models for various languages)

        Returns:
            List of TTSModel instances for HuggingFace TTS
        """
        models: List[TTSModel] = []

        # KokoroTTS - 54 voices
        kokoro_voices = [
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_heart",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
            "bm_daniel",
            "bm_fable",
            "bm_george",
            "bm_lewis",
            "ef_dora",
            "em_alex",
            "em_santa",
            "ff_siwis",
            "hf_alpha",
            "hf_beta",
            "hm_omega",
            "hm_psi",
            "if_sara",
            "im_nicola",
            "jf_alpha",
            "jf_gongitsune",
            "jf_nezumi",
            "jf_tebukuro",
            "jm_kumo",
            "pf_dora",
            "pm_alex",
            "pm_santa",
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
        ]

        kokoro_model = TTSModel(
            id="hexgrad/Kokoro-82M",
            name="Kokoro TTS 82M",
            provider=Provider.HuggingFace,
            voices=kokoro_voices,
        )
        models.append(kokoro_model)

        log.debug(f"Returning {len(models)} HuggingFace TTS models")
        return models

    async def get_available_asr_models(self) -> List["ASRModel"]:
        """Get available HuggingFace ASR models from recommended models.

        Returns ASR models based on the recommended models from the Whisper node:
        - OpenAI Whisper models (large-v3, large-v3-turbo, large-v2, medium, small)
        - Faster-whisper models (optimized for speed)

        Returns:
            List of ASRModel instances for HuggingFace ASR
        """

        models = [
            ASRModel(
                id="openai/whisper-large-v3",
                name="Whisper Large V3",
                provider=Provider.HuggingFace,
            ),
            ASRModel(
                id="openai/whisper-large-v3-turbo",
                name="Whisper Large V3 Turbo",
                provider=Provider.HuggingFace,
            ),
            ASRModel(
                id="openai/whisper-large-v2",
                name="Whisper Large V2",
                provider=Provider.HuggingFace,
            ),
            ASRModel(
                id="openai/whisper-medium",
                name="Whisper Medium",
                provider=Provider.HuggingFace,
            ),
            ASRModel(
                id="openai/whisper-small",
                name="Whisper Small",
                provider=Provider.HuggingFace,
            ),
            ASRModel(
                id="Systran/faster-whisper-large-v3",
                name="Faster Whisper Large V3",
                provider=Provider.HuggingFace,
            ),
        ]

        log.debug(f"Returning {len(models)} HuggingFace ASR models")
        return models

    @staticmethod
    def _parse_model_spec(model: str) -> tuple[str, str | None]:
        """Return repo_id, optional filename from model spec."""
        if ":" not in model:
            return model, None
        parts = model.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid model spec: {model}")
        repo_id, filename = parts
        return repo_id, filename

    def _load_image_data(self, image_ref) -> bytes:
        """Load image data from an ImageRef."""
        if hasattr(image_ref, "data") and image_ref.data is not None:
            return image_ref.data

        uri = getattr(image_ref, "uri", "") if hasattr(image_ref, "uri") else ""
        if not uri:
            raise ValueError("ImageRef has no data or URI")

        _mime, data = fetch_uri_bytes_and_mime_sync(uri)
        return data

    def convert_message(self, message: Message) -> Dict[str, Any]:
        """
        Convert an internal message to HF dict format.
        Preserves PIL images in content list for further processing.
        """
        if message.role == "tool":
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            elif isinstance(message.content, dict):
                content = json.dumps(message.content)
            elif isinstance(message.content, list):
                content = json.dumps([part.model_dump() for part in message.content])
            elif isinstance(message.content, str):
                content = message.content
            else:
                content = json.dumps(message.content)

            return {"role": "tool", "content": content, "name": message.name}

        elif message.role == "system":
            if message.content is None:
                content = ""
            elif isinstance(message.content, str):
                content = message.content
            else:
                text_parts = [
                    part.text
                    for part in message.content
                    if isinstance(part, MessageTextContent)
                ]
                content = "\n".join(text_parts)
            return {"role": "system", "content": content}

        elif message.role == "user":
            if isinstance(message.content, str):
                return {"role": "user", "content": message.content}

            # Handle list content
            content_list = []
            for part in message.content:
                if isinstance(part, MessageTextContent):
                    content_list.append({"type": "text", "text": part.text})
                elif isinstance(part, MessageImageContent):
                    # Load image to PIL
                    data = self._load_image_data(part.image)
                    img = Image.open(io.BytesIO(data))
                    # Store PIL image directly; will be extracted later
                    content_list.append({"type": "image", "image": img})

            return {"role": "user", "content": content_list}

        elif message.role == "assistant":
            # For assistant, we mainly handle text content and tool calls if we supported them
            content = ""
            if message.content is None:
                content = ""
            elif isinstance(message.content, str):
                content = message.content
            else:
                text_parts = [
                    part.text
                    for part in message.content
                    if isinstance(part, MessageTextContent)
                ]
                content = "\n".join(text_parts)

            msg_dict = {"role": "assistant", "content": content}

            # TODO: Handle tool calls formatting if HF supports it in standard templates
            # For now, we omit complex tool_call structures as apply_chat_template conventions vary

            return msg_dict

        else:
            # Fallback
            content = str(message.content) if message.content else ""
            return {"role": message.role, "content": content}

        # Helper to convert messages to prompt is no longer needed as we use apply_chat_template
        return ""

    async def _stream_pipeline_generation(
        self,
        repo_id: str,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        context: ProcessingContext,
        node_id: str | None,
        quantization: str = "fp16",
    ) -> AsyncIterator[Chunk]:
        from transformers import BitsAndBytesConfig

        cached_pipeline = ModelManager.get_model(repo_id)

        if not cached_pipeline:
            log.info(f"Loading HuggingFace pipeline model {repo_id}")
            load_kwargs = {}
            if quantization == "nf4":
                load_kwargs["model_kwargs"] = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                }
            elif quantization == "nf8":
                load_kwargs["model_kwargs"] = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_quant_type="nf8",
                        bnb_8bit_use_double_quant=True,
                        bnb_8bit_compute_dtype=torch.bfloat16,
                    )
                }
            cached_pipeline = await load_pipeline(
                node_id=node_id,
                context=context,
                pipeline_task="text-generation",
                model_id=repo_id,
                **load_kwargs,
            )
            ModelManager.set_model(node_id, repo_id, cached_pipeline)

        tokenizer = cached_pipeline.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer missing from HuggingFace pipeline")

        # Apply chat template
        # Ensure messages are in the format expected by HF (list of dicts)
        hf_messages = [self.convert_message(msg) for msg in messages]

        prompt = tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True
        )

        token_queue: Queue = Queue()

        class AsyncTextStreamer(TextStreamer):
            def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                self.token_queue = token_queue

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("TextStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]

                if self.skip_prompt and self.next_tokens_are_prompt:
                    self.next_tokens_are_prompt = False
                    return

                text = self.tokenizer.decode(
                    value, skip_special_tokens=True
                )  # pyright: ignore[reportAttributeAccessIssue]
                if text:
                    self.token_queue.put(text)

            def end(self):
                self.token_queue.put(None)

        streamer = AsyncTextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def generate():
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "streamer": streamer,
                "return_full_text": False,
            }
            cached_pipeline(prompt, **generation_kwargs)  # type: ignore[reportArgumentType]

        thread = threading.Thread(target=generate)
        thread.start()

        try:
            while True:
                await asyncio.sleep(0.01)
                while not token_queue.empty():
                    token = token_queue.get_nowait()
                    if token is None:
                        return
                    yield Chunk(content=token, done=False, content_type="text")
                if not thread.is_alive():
                    while not token_queue.empty():
                        token = token_queue.get_nowait()
                        if token is None:
                            return
                        yield Chunk(content=token, done=False, content_type="text")
                    break
        finally:
            thread.join(timeout=1.0)

    @staticmethod
    def _extract_text_from_output(outputs: Any) -> str:
        """Normalize output across pipeline variants."""
        if isinstance(outputs, list) and len(outputs) > 0:
            first = outputs[0]
            if isinstance(first, dict):
                for key in [
                    "generated_text",
                    "answer",
                    "text",
                    "output_text",
                ]:
                    if key in first and isinstance(first[key], str):
                        return first[key]  # type: ignore
            if isinstance(first, str):
                return first
        return str(outputs)

    async def _stream_image_text_to_text(
        self,
        repo_id: str,
        messages: Sequence[Message],
        max_tokens: int,
        context: ProcessingContext,
        node_id: str | None,
        quantization: str = "fp16",
    ) -> AsyncIterator[Chunk]:
        """Stream generation for image-text-to-text models."""
        from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForCausalLM

        # Extract images and clean messages
        # Convert messages and extract images
        cleaned_messages = []
        pil_images = []

        for msg in messages:
            # Use convert_message to standardize
            converted = self.convert_message(msg)

            # Post-process for VLM: extract PIL images from content list
            if isinstance(converted.get("content"), list):
                new_content = []
                for item in converted["content"]:
                    if item.get("type") == "image" and "image" in item:
                        # Extract PIL image
                        pil_images.append(item["image"])
                        # Keep placeholder
                        new_content.append({"type": "image"})
                    else:
                        new_content.append(item)
                converted["content"] = new_content

            cleaned_messages.append(converted)

        # Load processor
        processor = await load_model(
            node_id=node_id,
            context=context,
            model_class=AutoProcessor,
            model_id=repo_id,
        )

        load_kwargs = {"device_map": "auto"}
        if quantization == "nf4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "nf8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )

        # Load model using AutoModelForCausalLM as requested for VL models
        model = await load_model(
            node_id=node_id,
            context=context,
            model_class=AutoModelForCausalLM,  # User guide suggests this for Qwen2.5-VL/LLaVA
            model_id=repo_id,
            **load_kwargs,
        )

        # Prepare inputs
        def _prepare_inputs():
            return processor.apply_chat_template(
                cleaned_messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                images=pil_images if pil_images else None,
            ).to(model.device)

        inputs = await asyncio.to_thread(_prepare_inputs)

        # Output streamer
        token_queue: Queue = Queue()

        # We need a streamer that puts into our queue
        # Reusing the class defined inside _stream_pipeline_generation is not possible cleanly unless we move it out or duplicate.
        # Duplicating for now since the other method has it inline.

        class AsyncTextStreamer(TextStreamer):
            def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                self.token_queue = token_queue

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("TextStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]

                if self.skip_prompt and self.next_tokens_are_prompt:
                    self.next_tokens_are_prompt = False
                    return

                text = self.tokenizer.decode(value, skip_special_tokens=True)
                if text:
                    self.token_queue.put(text)

            def end(self):
                self.token_queue.put(None)

        streamer = AsyncTextStreamer(
            processor.tokenizer,  # Processor should have tokenizer
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def generate():
            model.generate(
                **(
                    inputs if isinstance(inputs, dict) else {"input_ids": inputs}
                ),  # apply_chat_template returns tensor or dict
                max_new_tokens=max_tokens,
                streamer=streamer,
            )

        thread = threading.Thread(target=generate)
        thread.start()

        try:
            while True:
                await asyncio.sleep(0.01)
                while not token_queue.empty():
                    token = token_queue.get_nowait()
                    if token is None:
                        return
                    yield Chunk(content=token, done=False, content_type="text")
                if not thread.is_alive():
                    while not token_queue.empty():
                        token = token_queue.get_nowait()
                        if token is None:
                            return
                        yield Chunk(content=token, done=False, content_type="text")
                    break
        finally:
            thread.join(timeout=1.0)

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> Message:
        """Generate a single message completion using HuggingFace GGUF models.

        Args:
            messages: Conversation history to send
            model: Model spec in format "repo_id:filename.gguf" (e.g., "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF:qwen2.5-coder-0.5b-q8_0.gguf")
            tools: Optional tool definitions (not supported for HF GGUF models)
            max_tokens: Maximum tokens to generate
            context_window: Maximum tokens to consider for context
            response_format: Optional response schema (not supported for HF GGUF models)
            **kwargs: Additional arguments (temperature, top_p, do_sample, context)

        Returns:
            A Message object containing the model's response

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        if not messages:
            raise ValueError("messages must not be empty")

        context = kwargs.pop("context", None)
        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace text generation"
            )

        full_text = ""
        async for chunk in self.generate_messages(
            messages=messages,
            model=model,
            tools=tools,
            max_tokens=max_tokens,
            context_window=context_window,
            response_format=response_format,
            context=context,
            **kwargs,
        ):
            if chunk.content:
                full_text += chunk.content

        return Message(
            role="assistant",
            content=full_text,
            provider=Provider.HuggingFace,
            model=model,
        )

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        quantization: str = "fp16",
        **kwargs: Any,
    ) -> AsyncIterator[Chunk]:
        """Stream message completions using HuggingFace GGUF models.

        Args:
            messages: Conversation history to send
            model: Model spec in format "repo_id:filename.gguf" (e.g., "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF:qwen2.5-coder-0.5b-q8_0.gguf")
            tools: Optional tool definitions (not supported for HF GGUF models)
            max_tokens: Maximum tokens to generate
            context_window: Maximum tokens to consider for context
            response_format: Optional response schema (not supported for HF GGUF models)
            quantization: Quantization method to use (nf4 for 4-bit, nf8 for 8-bit, fp16 for default)
            **kwargs: Additional arguments (temperature, top_p, do_sample, context)

        Yields:
            Chunk objects containing text deltas

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        if not messages:
            raise ValueError("messages must not be empty")

        context = kwargs.get("context")
        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace text generation"
            )

        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 1.0)
        do_sample = kwargs.get("do_sample", True)
        node_id = kwargs.get("node_id")

        pipeline_task = kwargs.get("pipeline_task")
        if pipeline_task == "image-text-to-text":
            repo_id, _ = self._parse_model_spec(model)
            async for chunk in self._stream_image_text_to_text(
                repo_id=repo_id,
                messages=messages,
                max_tokens=max_tokens,
                context=context,
                node_id=node_id,
                quantization=quantization,
            ):
                yield chunk
            return

        repo_id, filename = self._parse_model_spec(model)

        async for chunk in self._stream_pipeline_generation(
            repo_id=repo_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            context=context,
            node_id=node_id,
            quantization=quantization,
        ):
            yield chunk


if __name__ == "__main__":
    import asyncio

    # Create provider instance
    provider = HuggingFaceLocalProvider()

    async def test_generate_messages():
        """Test the generate_messages method with streaming."""
        from nodetool.workflows.processing_context import ProcessingContext
        from nodetool.config.environment import Environment

        # Initialize environment
        env = Environment.get_environment()

        # Create a simple processing context (you may need to adjust this based on your setup)
        context = ProcessingContext()
        context.device = "mps"

        # Test messages
        messages = [
            Message(
                role="system",
                content="You are a helpful assistant that provides concise answers.",
            ),
            Message(
                role="user",
                content="What is the capital of France? Answer in one sentence.",
            ),
        ]

        # Model to test - using a small model for quick testing
        # Change this to any model you have cached locally
        models = await provider.get_available_language_models()
        model = models[0]

        # Stream the response
        full_response = ""
        async for chunk in provider.generate_messages(
            messages=messages,
            model=model.id,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            context=context,
        ):
            print(chunk)

    async def test_generate_message():
        """Test the generate_message method (non-streaming)."""
        from nodetool.workflows.processing_context import ProcessingContext

        # Create a simple processing context
        context = ProcessingContext()
        context.device = "mps"

        # Test messages
        messages = [
            Message(
                role="system",
                content="You are a helpful assistant that provides concise answers.",
            ),
            Message(
                role="user",
                content="What is 2+2? Answer in one sentence.",
            ),
        ]

        # Model to test
        models = await provider.get_available_language_models()
        model = models[0]

        # Get the response
        response = await provider.generate_message(
            messages=messages,
            model=model.id,
            max_tokens=50,
            temperature=0.7,
            context=context,
        )

    async def test_text_to_image():
        """Test the text_to_image method."""
        models = await provider.get_available_image_models()
        print(models)
        model = models[0]
        context = ProcessingContext()
        context.device = "mps"
        image = await provider.text_to_image(
            params=TextToImageParams(
                prompt="A beautiful sunset over a calm ocean",
                model=model,
                num_inference_steps=20,
            ),
            context=context,
        )
        open("image.png", "wb").write(image)

    async def test_image_to_image():
        """Test the image_to_image method."""
        models = await provider.get_available_image_models()
        print(models)
        model = models[0]
        provider = HuggingFaceLocalProvider()
        context = ProcessingContext()
        context.device = "mps"
        image_bytes = open("image.png", "rb").read()
        image = await provider.image_to_image(
            image=image_bytes,
            params=ImageToImageParams(
                prompt="a photo of an astronaut riding a horse",
                model=model,
                strength=0.8,
                num_inference_steps=20,
                guidance_scale=7.5,
            ),
            context=context,
        )
        open("image_to_image.png", "wb").write(image)

    async def test_available_language_models():
        """Test the available_language_models method."""
        provider = HuggingFaceLocalProvider()
        models = await provider.get_available_language_models()
        print(models)

    async def test_available_image_models():
        """Test the available_image_models method."""
        provider = HuggingFaceLocalProvider()
        models = await provider.get_available_image_models()
        print(models)

    async def test_available_tts_models():
        """Test the available_tts_models method."""
        provider = HuggingFaceLocalProvider()
        models = await provider.get_available_tts_models()
        print(models)

    async def test_available_asr_models():
        """Test the available_asr_models method."""
        provider = HuggingFaceLocalProvider()
        models = await provider.get_available_asr_models()
        print(models)

    # Run tests
    print("=" * 50)
    print("Testing HuggingFace Local Provider")
    print("=" * 50)

    asyncio.run(test_available_image_models())

    # # Test streaming
    # asyncio.run(test_generate_messages())

    # # Test non-streaming
    # asyncio.run(test_generate_message())
    # asyncio.run(test_text_to_image())
    # asyncio.run(test_image_to_image())
