"""
Shared helpers for the local HuggingFace provider and nodes.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, TypeVar, TYPE_CHECKING

from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE
from nodetool.ml.core.model_manager import ModelManager
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress
from huggingface_hub import _CACHED_NO_EXIST, hf_hub_download

if TYPE_CHECKING:
    import torch

T = TypeVar("T")

log = get_logger(__name__)

ALLOW_DOWNLOAD_ENV = "NODETOOL_HF_ALLOW_DOWNLOAD"
_ALLOW_DOWNLOAD_VALUES = {"1", "true", "yes", "on"}
_PREFERRED_HF_DEVICE = "mps"


def _get_torch():
    """Lazy import for torch."""
    import torch

    return torch


def _is_cuda_available() -> bool:
    """Safely check if CUDA is available, handling cases where PyTorch lacks CUDA."""
    try:
        torch = _get_torch()
        if not hasattr(torch, "cuda"):
            return False
        return torch.cuda.is_available()
    except (RuntimeError, AttributeError):
        return False


def _is_mps_available() -> bool:
    """Detect whether the Apple Metal backend is available."""
    try:
        import torch

        return (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
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
        await asyncio.to_thread(
            hf_hub_download,
            repo_id,
            file_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        cache_path = await HF_FAST_CACHE.resolve(repo_id, file_path)
        if cache_path:
            return cache_path
        raise ValueError(
            f"Failed to cache model file {repo_id}/{file_path} after download attempt"
        )

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
    torch_dtype: "torch.dtype" | None = None,
    skip_cache: bool = False,
    **kwargs: Any,
):
    """Load a HuggingFace pipeline model with optional VRAM recovery."""
    if model_id == "" or model_id is None:
        raise ValueError("Please select a model")

    cache_key = f"{model_id}_{pipeline_task}"

    cached_model = ModelManager.get_model(cache_key)
    if cached_model:
        target_device = _resolve_hf_device(context, device or context.device)
        return _ensure_model_on_device(cached_model, target_device)

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
        from transformers import pipeline

        return pipeline(
            pipeline_task,  # type: ignore
            model=model_id,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    try:
        model = _create_pipeline()
    except (ValueError, RuntimeError) as exc:
        if not _is_vram_error(exc):
            raise

        log.warning(
            "VRAM error while loading pipeline %s: %s. Attempting to free cached models and retry...",
            model_id,
            exc,
        )
        context.post_message(
            JobUpdate(
                status="running",
                message=f"Freeing VRAM and retrying load of {model_id}...",
            )
        )

        ModelManager.free_vram_if_needed(
            reason=f"VRAM error loading pipeline {model_id}",
            aggressive=True,
        )

        try:
            model = _create_pipeline()
            log.info("Successfully loaded pipeline %s after freeing VRAM", model_id)
        except Exception as retry_exc:
            log.error(
                "Failed to load pipeline %s even after freeing VRAM: %s",
                model_id,
                retry_exc,
            )
            raise

    ModelManager.set_model(node_id, cache_key, model)
    return model  # type: ignore


async def load_model(
    node_id: str,
    context: ProcessingContext,
    model_class: type[T],
    model_id: str,
    variant: str | None = None,
    torch_dtype: "torch.dtype" | None = None,
    path: str | None = None,
    skip_cache: bool = False,
    cache_key: str | None = None,
    **kwargs: Any,
) -> T:
    """Load a HuggingFace model with optional VRAM recovery."""
    if model_id == "":
        raise ValueError("Please select a model")

    target_device = _resolve_hf_device(context, context.device)

    log.info("Loading model %s/%s from %s", model_id, path, target_device)

    if cache_key is None:
        cache_key = f"{model_id}_{model_class.__name__}_{path}"

    if not skip_cache:
        cached_model = ModelManager.get_model(cache_key)
        if cached_model:
            return _ensure_model_on_device(cached_model, target_device)

    async def _do_load() -> T:
        if path:
            cache_path = await HF_FAST_CACHE.resolve(model_id, path)
            if not cache_path:
                raise ValueError(
                    f"Download model {model_id}/{path} first from recommended models"
                )

            log.info("Loading model %s from %s", model_id, cache_path)
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
                model = model_class.from_pretrained(  # type: ignore
                    model_id,
                    torch_dtype=torch_dtype,
                    variant=variant,
                    **kwargs,
                )
        else:
            log.info("Loading model %s from HuggingFace", model_id)
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

    try:
        model = await _do_load()
    except (ValueError, RuntimeError) as exc:
        if not _is_vram_error(exc):
            raise

        log.warning(
            "VRAM error while loading model %s: %s. Attempting to free cached models and retry...",
            model_id,
            exc,
        )
        context.post_message(
            JobUpdate(
                status="running",
                message=f"Freeing VRAM and retrying load of {model_id}...",
            )
        )

        ModelManager.free_vram_if_needed(
            reason=f"VRAM error loading {model_id}",
            aggressive=True,
        )

        try:
            model = await _do_load()
            log.info("Successfully loaded model %s after freeing VRAM", model_id)
        except Exception as retry_exc:
            log.error(
                "Failed to load model %s even after freeing VRAM: %s",
                model_id,
                retry_exc,
            )
            raise

    model = _ensure_model_on_device(model, target_device)
    ModelManager.set_model(node_id, cache_key, model)
    return model


async def _detect_cached_variant(repo_id: str) -> str | None:
    """Detect a cached diffusers variant (e.g., fp16) for a given repo."""
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
            continue

    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        return None

    for root, _, files in os.walk(snapshot_dir):
        for f in files:
            if ".fp16." in f:
                return "fp16"

    return None


def _is_node_model(model_id: str, model_path: str | None, node_cls: Any) -> bool:
    """Check if the model matches any of the node's recommended models."""
    for rec in node_cls.get_recommended_models():
        if rec.repo_id == model_id:
            if rec.path:
                if rec.path == model_path:
                    return True
            else:
                return True
    return False


def pipeline_progress_callback(
    node_id: str, total_steps: int, context: ProcessingContext
):
    def callback(
        pipeline: "Any", step: int, timestep: int, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        context.post_message(
            NodeProgress(
                node_id=node_id,
                progress=step,
                total=total_steps,
            )
        )
        return kwargs

    return callback


def _enable_pytorch2_attention(pipeline: Any, enabled: bool = True):
    """Enable PyTorch 2 scaled dot product attention to speed up inference."""
    if not enabled or pipeline is None:
        return

    enable_sdpa = getattr(pipeline, "enable_sdpa", None)

    if callable(enable_sdpa):
        try:
            enable_sdpa()
            pipeline_name = type(pipeline).__name__
            log.info(
                "Enabled PyTorch 2 scaled dot product attention for %s", pipeline_name
            )
        except Exception as e:
            log.warning("Failed to enable scaled dot product attention: %s", e)
    else:
        log.info("Scaled dot product attention not available on this pipeline")


def _apply_vae_optimizations(pipeline: Any):
    """Apply VAE slicing and channels_last layout when available."""
    if pipeline is None:
        return

    vae = getattr(pipeline, "vae", None)
    if vae is None:
        return

    if hasattr(vae, "enable_slicing"):
        try:
            vae.enable_slicing()
            log.debug("Enabled VAE slicing")
        except Exception as e:
            log.warning("Failed to enable VAE slicing: %s", e)

    try:
        torch = _get_torch()
        vae.to(memory_format=torch.channels_last)
        log.debug("Set VAE to channels_last memory format")
    except Exception as e:
        log.warning("Failed to set VAE channels_last memory format: %s", e)
