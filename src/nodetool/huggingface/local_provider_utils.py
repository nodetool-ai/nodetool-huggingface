"""
Shared helpers for the local HuggingFace provider and nodes.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar, TYPE_CHECKING

from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE
from nodetool.ml.core.model_manager import ModelManager
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import JobUpdate, NodeProgress
from huggingface_hub import _CACHED_NO_EXIST, hf_hub_download

if TYPE_CHECKING:
    import torch

T = TypeVar("T")

log = get_logger(__name__)

ALLOW_DOWNLOAD_ENV = "NODETOOL_HF_ALLOW_DOWNLOAD"
_ALLOW_DOWNLOAD_VALUES = {"1", "true", "yes", "on"}
_PREFERRED_HF_DEVICE = "mps"


def _loads_safetensors_via_pretrained(model_class: type) -> bool:
    """Nunchaku SVDQ weights load via ``from_pretrained(path.safetensors)``, not diffusers ``from_single_file``."""
    module = getattr(model_class, "__module__", "")
    return module.startswith("nunchaku.")


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
    """Move HF pipelines/models onto the requested device when possible.

    Cached pipelines may carry CPU offload hooks installed by
    ``_apply_memory_optimizations`` because their weights did not fit in VRAM.
    Moving those onto the GPU wholesale would undo the offload on every cache
    hit, so the move is skipped for them.
    """
    if not device:
        return model

    from nodetool.huggingface.memory_utils import offload_kind

    if str(device) != "cpu" and offload_kind(model) is not None:
        log.debug(
            "Keeping %s under CPU offload instead of moving it to %s",
            model.__class__.__name__,
            device,
        )
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


def _reclaim_vram_before_load(label: str) -> None:
    """Offload least-recently-used cached models when VRAM is already tight.

    Nothing trims the model cache between workflow runs, so without this the
    only reclaim ever attempted happens *after* a load has already raised OOM.
    Asking first turns that into an eviction of cold models before the new
    weights are allocated. ``free_vram_if_needed`` is a no-op when the device
    is below its pressure threshold, is throttled by its own cooldown, and only
    moves models to CPU, so a cache hit later still finds them.
    """
    try:
        ModelManager.free_vram_if_needed(reason=f"Preparing to load {label}")
    except Exception as exc:  # pragma: no cover - reclaim must never block a load
        log.debug("Proactive VRAM reclaim failed for %s: %s", label, exc)


def _release_exception(exc: BaseException) -> None:
    """Drop an exception's traceback and cause chain, then collect.

    A caught exception keeps every frame of the failed call alive through
    ``__traceback__``. When the failure was an OOM during model loading, those
    frames still hold the half-materialised GPU tensors, so any
    ``empty_cache()``/offload attempt made while the exception is reachable
    frees nothing. Callers that recover in-place must call this first.
    """
    import gc

    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    gc.collect()

    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch optional / CUDA absent
        pass


def _quantization_cache_suffix(kwargs: dict[str, Any]) -> str:
    """Derive a cache-key suffix from any quantization config in the load kwargs.

    Without it, a second load requesting a different quantization would hit the
    entry cached by the first load and the request would be silently ignored.
    """
    config = kwargs.get("quantization_config")
    if config is None:
        model_kwargs = kwargs.get("model_kwargs")
        if isinstance(model_kwargs, dict):
            config = model_kwargs.get("quantization_config")
    if config is None:
        return ""
    try:
        marker = repr(config)
    except Exception:
        marker = config.__class__.__name__
    return "_" + hashlib.sha1(marker.encode("utf-8")).hexdigest()[:8]


#: Component kwargs that swap out a pipeline's heavy weights. Two loads that
#: differ only in one of these produce genuinely different pipelines and must
#: not share a cache entry.
_COMPONENT_KWARGS = (
    "unet",
    "transformer",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "vae",
)


def _component_identity(component: Any) -> str:
    """Stable identity for a pre-built pipeline component.

    Components loaded through :func:`load_model` carry the cache key they were
    stored under, which already encodes repo, class, path and quantization.
    That is the only discriminator fine-grained enough to tell, say, an INT4
    Nunchaku UNet from an FP4 one — both share a class name.
    """
    marker = getattr(component, "_nodetool_cache_key", None)
    if isinstance(marker, str) and marker:
        return marker
    return f"{component.__class__.__name__}:{id(component):x}"


def _component_cache_suffix(kwargs: dict[str, Any]) -> str:
    """Derive a cache-key suffix from pre-built component kwargs.

    Without it a pipeline assembled with a quantized transformer collides with
    the full-precision pipeline for the same base repo: the second load returns
    the first one's pipeline and the freshly loaded transformer is stranded on
    the GPU with nothing referencing it.
    """
    parts = [
        f"{name}={_component_identity(kwargs[name])}"
        for name in _COMPONENT_KWARGS
        if kwargs.get(name) is not None
    ]
    if not parts:
        return ""
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8]
    return f"_c{digest}"


def _normalize_pipeline_task(pipeline_task: str) -> str:
    """Map legacy/NodeTool task names to the task names supported by installed transformers."""
    task = pipeline_task.strip().lower()
    if task == "image-to-text":
        return "image-text-to-text"
    return task


async def load_pipeline(
    node_id: str,
    context: ProcessingContext,
    pipeline_task: str,
    model_id: Any,
    device: str | None = None,
    torch_dtype: "torch.dtype" | None = None,
    skip_cache: bool = False,
    cache_key: str | None = None,
    cache_key_suffix: str = "",
    **kwargs: Any,
):
    """Load a HuggingFace pipeline model with optional VRAM recovery."""
    if model_id == "" or model_id is None:
        raise ValueError("Please select a model")

    normalized_pipeline_task = _normalize_pipeline_task(pipeline_task)

    if cache_key is None:
        cache_key = (
            f"{model_id}_{normalized_pipeline_task}{_quantization_cache_suffix(kwargs)}"
        )
    # Applied to explicit keys too: a caller that hands in its own key still
    # must not share an entry with an adapter-carrying variant of the same model.
    cache_key += cache_key_suffix

    if not skip_cache:
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
            message=f"Loading pipeline {type(model_id) == str and model_id or normalized_pipeline_task} from HuggingFace",
        )
    )
    if "token" not in kwargs:
        kwargs["token"] = await context.get_secret("HF_TOKEN")

    def _create_pipeline():
        from transformers import pipeline

        pipeline_kwargs = dict(kwargs)
        # Honor the requested device on fresh loads (previously only cache
        # hits were moved); accelerate handles placement when device_map is
        # given, and transformers rejects passing both.
        if device and "device_map" not in pipeline_kwargs:
            pipeline_kwargs["device"] = device

        return pipeline(
            normalized_pipeline_task,
            model=model_id,
            torch_dtype=torch_dtype,
            **pipeline_kwargs,
        )

    _reclaim_vram_before_load(f"pipeline {model_id}")

    vram_error: str | None = None
    model = None
    try:
        model = _create_pipeline()
    except (ValueError, RuntimeError) as exc:
        if not _is_vram_error(exc):
            raise
        # Release the traceback before retrying: its frames still reference the
        # partially loaded weights, so freeing VRAM while the exception is live
        # would leave the failed attempt resident and the retry would OOM again.
        vram_error = str(exc)
        _release_exception(exc)

    if vram_error is not None:
        log.warning(
            "VRAM error while loading pipeline %s: %s. Attempting to free cached models and retry...",
            model_id,
            vram_error,
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

    if not skip_cache:
        ModelManager.set_model(node_id, cache_key, model)
    return model


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
    cache_key_suffix: str = "",
    device: str | None = None,
    **kwargs: Any,
) -> T:
    """Load a HuggingFace model with optional VRAM recovery.

    ``device`` explicitly picks the placement: ``"cpu"`` keeps the weights on
    the host (callers that install CPU offload hooks afterwards rely on the
    model never transiting the GPU in full). When omitted, the context device
    is used. Models loaded with ``device_map`` are never moved — accelerate
    already dispatched them.
    """
    if model_id == "":
        raise ValueError("Please select a model")

    if device is not None:
        target_device = device
    else:
        target_device = _resolve_hf_device(context, context.device)
    skip_device_move = device == "cpu" or "device_map" in kwargs

    log.info("Loading model %s/%s from %s", model_id, path, target_device)

    if cache_key is None:
        cache_key = (
            f"{model_id}_{model_class.__name__}_{path}"
            f"{_quantization_cache_suffix(kwargs)}{_component_cache_suffix(kwargs)}"
        )
    cache_key += cache_key_suffix

    if not skip_cache:
        cached_model = ModelManager.get_model(cache_key)
        if cached_model:
            if skip_device_move:
                return cached_model
            return _ensure_model_on_device(cached_model, target_device)

    load_kwargs = dict(kwargs)
    cache_path: str | None = None
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
                message=f"Loading model {model_id}/{path}…",
            )
        )
    else:
        log.info("Loading model %s from HuggingFace", model_id)
        context.post_message(
            JobUpdate(
                status="running",
                message=f"Loading model {model_id} from HuggingFace…",
            )
        )
        if "token" not in load_kwargs:
            load_kwargs["token"] = await context.get_secret("HF_TOKEN")

    def _load_sync() -> T:
        if path:
            assert cache_path is not None
            if _loads_safetensors_via_pretrained(model_class):
                nunchaku_kwargs = dict(load_kwargs)
                if device is not None:
                    # Nunchaku loads its quantized weights directly onto the
                    # requested device.
                    nunchaku_kwargs["device"] = device
                return model_class.from_pretrained(  # type: ignore[attr-defined]
                    cache_path,
                    torch_dtype=torch_dtype,
                    **nunchaku_kwargs,
                )
            if hasattr(model_class, "from_single_file"):
                return model_class.from_single_file(  # type: ignore
                    cache_path,
                    torch_dtype=torch_dtype,
                    variant=variant,
                    **load_kwargs,
                )
            return model_class.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                variant=variant,
                **load_kwargs,
            )
        return model_class.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant=variant,
            **load_kwargs,
        )

    _reclaim_vram_before_load(f"model {model_id}")

    vram_error: str | None = None
    model = None
    try:
        model = await asyncio.to_thread(_load_sync)
    except (ValueError, RuntimeError) as exc:
        if not _is_vram_error(exc):
            raise
        # See the note in load_pipeline: the traceback pins the failed load's
        # GPU tensors, so drop it before trying to reclaim VRAM.
        vram_error = str(exc)
        _release_exception(exc)

    if vram_error is not None:
        log.warning(
            "VRAM error while loading model %s: %s. Attempting to free cached models and retry...",
            model_id,
            vram_error,
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
            model = await asyncio.to_thread(_load_sync)
            log.info("Successfully loaded model %s after freeing VRAM", model_id)
        except Exception as retry_exc:
            log.error(
                "Failed to load model %s even after freeing VRAM: %s",
                model_id,
                retry_exc,
            )
            raise

    if not skip_device_move:
        model = _ensure_model_on_device(model, target_device)
    # Record the key even when the cache is skipped: callers embed this model
    # into pipelines, and _component_cache_suffix uses the marker to keep those
    # pipelines' cache entries distinct.
    with suppress(AttributeError):
        model._nodetool_cache_key = cache_key  # type: ignore[attr-defined]
    if not skip_cache:
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


def _apply_vae_optimizations(pipeline: Any, enable_tiling: bool = False):
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

    if enable_tiling and hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling()
            log.debug("Enabled VAE tiling")
        except Exception as e:
            log.warning("Failed to enable VAE tiling: %s", e)

    try:
        torch = _get_torch()
        vae.to(memory_format=torch.channels_last)
        log.debug("Set VAE to channels_last memory format")
    except Exception as e:
        log.warning("Failed to set VAE channels_last memory format: %s", e)


# Activations, latents, text-encoder outputs and the CUDA context all need
# space beyond the raw weights, so keep this much VRAM free when deciding
# whether a pipeline fits on the device.
_VRAM_HEADROOM_GB = 2.0


def _cuda_free_vram_gb() -> float:
    """Free VRAM on the current CUDA device in GiB, 0.0 when CUDA is absent."""
    try:
        torch = _get_torch()
        if not _is_cuda_available():
            return 0.0
        free_bytes, _total = torch.cuda.mem_get_info()
        return free_bytes / (1024**3)
    except Exception:
        return 0.0


def _pipeline_component_sizes_gb(pipeline: Any) -> list[float]:
    """Per-component parameter footprints (GiB) of a diffusers pipeline."""
    sizes: list[float] = []
    components = getattr(pipeline, "components", None) or {}
    for component in components.values():
        parameters = getattr(component, "parameters", None)
        if not callable(parameters):
            continue
        try:
            size = sum(p.numel() * p.element_size() for p in component.parameters())
        except Exception:
            continue
        if size:
            sizes.append(size / (1024**3))
    return sizes


def _apply_memory_optimizations(
    pipeline: Any,
    device: str,
    force_cpu_offload: bool = False,
) -> bool:
    """Place a freshly loaded pipeline within the available VRAM budget.

    Picks the cheapest strategy that fits: full device placement when the
    summed weights fit in free VRAM, model-level CPU offload when they don't,
    or sequential (per-layer) offload when even the largest single component
    exceeds the budget (e.g. the 23 GiB bf16 Flux transformer on a 16-24 GiB
    GPU). Returns True when any CPU offload hook was installed, in which case
    the pipeline must not be moved via ``.to()`` afterwards.
    """
    if pipeline is None:
        return False

    from nodetool.huggingface.memory_utils import (
        apply_cpu_offload_if_needed,
        has_cpu_offload_enabled,
    )

    if has_cpu_offload_enabled(pipeline):
        # Nunchaku loaders install their own offload strategy; exactly one
        # strategy may exist per pipeline.
        _apply_vae_optimizations(pipeline)
        return True

    if not _is_cuda_available() or not str(device).startswith("cuda"):
        # On MPS the offloaded weights share unified memory with the device,
        # so offloading frees nothing; on CPU it is meaningless.
        _apply_vae_optimizations(pipeline)
        pipeline.to(device)
        return False

    sizes = _pipeline_component_sizes_gb(pipeline)
    total_gb = sum(sizes)
    largest_gb = max(sizes, default=0.0)
    budget_gb = max(_cuda_free_vram_gb() - _VRAM_HEADROOM_GB, 0.0)

    if not force_cpu_offload and total_gb <= budget_gb:
        _apply_vae_optimizations(pipeline)
        pipeline.to(device)
        return False

    if largest_gb <= budget_gb and hasattr(pipeline, "enable_model_cpu_offload"):
        log.info(
            "Pipeline weights (%.1f GiB) exceed the %.1f GiB VRAM budget; "
            "enabling model CPU offload",
            total_gb,
            budget_gb,
        )
        # Tiling keeps the VAE decode within budget at high resolutions.
        _apply_vae_optimizations(pipeline, enable_tiling=True)
        apply_cpu_offload_if_needed(pipeline, method="model")
        return True

    if hasattr(pipeline, "enable_sequential_cpu_offload"):
        log.info(
            "Largest pipeline component (%.1f GiB) exceeds the %.1f GiB VRAM "
            "budget; enabling sequential CPU offload",
            largest_gb,
            budget_gb,
        )
        _apply_vae_optimizations(pipeline, enable_tiling=True)
        apply_cpu_offload_if_needed(pipeline, method="sequential")
        return True

    _apply_vae_optimizations(pipeline)
    pipeline.to(device)
    return False
