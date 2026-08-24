"""
Nunchaku pipeline utilities for optimized model loading.

This module provides utilities for loading nunchaku-optimized models
(transformers and text encoders) when the nunchaku package is available.
All nunchaku-specific code is isolated here to allow the main huggingface
module to work without nunchaku installed.
"""

import asyncio
from typing import Any, TYPE_CHECKING

from nodetool.config.logging_config import get_logger
from nodetool.huggingface.flux_utils import (
    detect_flux_variant,
    flux_variant_to_base_model_id,
)
from nodetool.huggingface.local_provider_utils import _get_torch, load_model
from nodetool.huggingface.memory_utils import apply_cpu_offload_if_needed
from nodetool.workflows.memory_utils import log_memory, run_gc, MemoryTracker
from nodetool.workflows.types import JobUpdate
from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE

if TYPE_CHECKING:
    import torch
    from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


# nunchaku availability will be checked lazily
NUNCHAKU_AVAILABLE: bool | None = None


def _check_nunchaku_available() -> bool:
    """Check if the SVDQuant nunchaku runtime is available (lazy check)."""
    global NUNCHAKU_AVAILABLE
    if NUNCHAKU_AVAILABLE is None:
        try:
            from nunchaku import NunchakuFluxTransformer2dModel

            _ = NunchakuFluxTransformer2dModel
            NUNCHAKU_AVAILABLE = True
        except ImportError:
            NUNCHAKU_AVAILABLE = False
    return NUNCHAKU_AVAILABLE


def is_nunchaku_available() -> bool:
    """Check if the nunchaku package is available."""
    return _check_nunchaku_available()


def _require_nunchaku() -> None:
    """Raise an error if nunchaku is not available."""
    if not _check_nunchaku_available():
        raise ImportError(
            "The SVDQuant nunchaku runtime is required for this operation but is not installed. "
            "Install it from the NodeTool package manager (nunchaku-ai/nunchaku) or from the "
            "NodeTool registry wheel index. Do not install the unrelated PyPI 'nunchaku' package."
        )


async def get_nunchaku_text_encoder(
    context: "ProcessingContext",
    node_id: str,
    repo_id: str | None = None,
    path: str | None = None,
    allow_downloads: bool = True,
    torch_dtype: Any = None,  # Defaults to torch.bfloat16 at runtime
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
        torch_dtype: The torch dtype to use (must match the pipeline dtype)

    Returns:
        dict: Pipeline kwargs with text_encoder_2 if nunchaku T5 encoder is found
    """
    _require_nunchaku()

    from nunchaku import NunchakuT5EncoderModel
    from nunchaku.utils import get_precision
    from huggingface_hub import hf_hub_download

    if repo_id is None:
        repo_id = "nunchaku-ai/nunchaku-t5"
    if path is None:
        path = f"awq-{get_precision()}-flux.1-t5xxl.safetensors"

    # Try to find nunchaku T5 encoder
    cache_path = await HF_FAST_CACHE.resolve(repo_id, path)
    if not cache_path:
        if not allow_downloads:
            raise ValueError(
                f"Nunchaku text encoder {repo_id}/{path} is not downloaded. "
                "Download it to the local HF cache before running this node."
            )

        log.info(
            "Downloading Nunchaku text_encoder %s/%s to cache",
            repo_id,
            path,
        )
        await asyncio.to_thread(hf_hub_download, repo_id, path)
        cache_path = await HF_FAST_CACHE.resolve(repo_id, path)

        if not cache_path:
            raise ValueError(
                f"Downloading model {repo_id}/{path} from HuggingFace failed"
            )

    torch = _get_torch()
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    return await load_model(
        context=context,
        model_id=repo_id,
        path=path,
        model_class=NunchakuT5EncoderModel,
        node_id=node_id,
        torch_dtype=torch_dtype,
    )


async def get_nunchaku_transformer(
    context: "ProcessingContext",
    model_class: type,
    node_id: str,
    repo_id: str,
    path: str,
    torch_dtype: Any = None,  # Defaults to torch.bfloat16 at runtime
    device: str | None = None,
) -> Any | None:
    """
    Get transformer kwargs when using a nunchaku model.

    This function checks if the model is a nunchaku FLUX variant and if so,
    attempts to find and load the nunchaku SVDQ transformer from the HF cache.

    Args:
        context: The context object
        model_class: The model class to load
        node_id: The node ID
        repo_id: The repository ID
        path: The path to the transformer file
        torch_dtype: The torch dtype to use
        device: The device to use (defaults to context.device or cuda/cpu)

    Returns:
        dict: Pipeline kwargs with transformer if nunchaku SVDQ transformer is found
    """
    _require_nunchaku()

    from nunchaku.utils import get_precision

    # Resolve device - nunchaku requires a valid device, not None
    torch = _get_torch()
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    if device is None:
        device = (
            context.device
            if context.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
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

    cache_path = await HF_FAST_CACHE.resolve(repo_id, path)
    if not cache_path:
        raise ValueError(
            f"Nunchaku transformer {repo_id}/{path} is not downloaded. "
            "Download it from recommended models before running this node."
        )

    transformer = await load_model(
        context=context,
        model_id=repo_id,
        path=path,
        model_class=model_class,
        node_id=node_id,
        torch_dtype=torch_dtype,
        device=device,
    )
    return transformer


def _node_identifier(node_id: str | None, repo_id: str) -> str:
    """Derive a unique identifier for caching based on node ID or repo."""
    return node_id or repo_id


async def load_nunchaku_flux_pipeline(
    context: "ProcessingContext",
    repo_id: str,
    transformer_path: str,
    node_id: str | None,
    pipeline_class: Any | None = None,
    cache_key: str | None = None,
) -> Any:
    """Load a FLUX pipeline with a Nunchaku transformer/text encoder pair."""
    _require_nunchaku()

    from nunchaku import NunchakuFluxTransformer2dModel
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
    from nodetool.ml.core.model_manager import ModelManager

    log_memory("load_nunchaku_flux_pipeline - START")

    pipeline_class = pipeline_class or FluxPipeline
    pipeline_name = pipeline_class.__name__

    if cache_key:
        cached_pipeline = ModelManager.get_model(cache_key)
        if cached_pipeline:
            log.info(f"[MEMORY] Returning cached pipeline for {cache_key}")
            return cached_pipeline

    variant = detect_flux_variant(repo_id, transformer_path)
    base_model_id = flux_variant_to_base_model_id(variant)
    hf_token = await context.get_secret("HF_TOKEN")
    if variant != "schnell" and not hf_token:
        raise ValueError(
            f"Flux-{variant} is a gated model, please set the HF_TOKEN in Nodetool settings and accept the terms of use for the model: "
            f"https://huggingface.co/{base_model_id}"
        )

    torch = _get_torch()
    torch_dtype = torch.bfloat16 if variant in ["schnell", "dev"] else torch.float16
    node_key = _node_identifier(node_id, repo_id)

    # Run GC before loading transformer to free any unused memory
    run_gc("Before loading Nunchaku transformer")

    with MemoryTracker("Loading Nunchaku transformer", run_gc_after=False):
        transformer = await get_nunchaku_transformer(
            context=context,
            model_class=NunchakuFluxTransformer2dModel,
            node_id=node_key,
            repo_id=repo_id,
            path=transformer_path,
            torch_dtype=torch_dtype,
        )
        transformer.set_attention_impl("nunchaku-fp16")

    with MemoryTracker("Loading Nunchaku text encoder", run_gc_after=False):
        text_encoder = await get_nunchaku_text_encoder(
            context, node_key, torch_dtype=torch_dtype
        )

    pipeline_kwargs: dict[str, Any] = {
        "transformer": transformer,
        "torch_dtype": torch_dtype,
        "token": hf_token,
    }
    if text_encoder is not None:
        pipeline_kwargs["text_encoder_2"] = text_encoder

    base_cached = await HF_FAST_CACHE.resolve(base_model_id, "model_index.json")
    if base_cached:
        pipeline_kwargs["local_files_only"] = True

    context.post_message(
        JobUpdate(
            status="running",
            message=f"Assembling Flux pipeline from {base_model_id}…",
        )
    )

    def _build_pipeline():
        return pipeline_class.from_pretrained(base_model_id, **pipeline_kwargs)

    try:
        with MemoryTracker(
            f"Building {pipeline_name} from pretrained", run_gc_after=True
        ):
            pipeline = await asyncio.to_thread(_build_pipeline)

        if cache_key:
            ModelManager.set_model(node_id, cache_key, pipeline)

        log_memory("load_nunchaku_flux_pipeline - END")
        return pipeline
    except _get_torch().OutOfMemoryError as exc:
        run_gc("After OOM error")
        raise ValueError(
            "VRAM out of memory while loading Flux with the Nunchaku transformer. "
            "Try enabling 'CPU offload' or reduce image size/steps."
        ) from exc


def apply_qwen_rope_compat_shim() -> bool:
    """Let nunchaku's Qwen transformer call ``QwenEmbedRope`` on diffusers >= 0.39.

    diffusers PR #12702 removed the ``txt_seq_lens`` parameter, moving ``device``
    into the second positional slot::

        0.36            forward(video_fhw, txt_seq_lens, device)
        0.37, 0.38      forward(video_fhw, txt_seq_lens=None, device=None, max_txt_seq_len=None)
        0.39, 0.40      forward(video_fhw, device=None, max_txt_seq_len=None)

    nunchaku still calls the 0.36 form (``transformer_qwenimage.py:517``:
    ``self.pos_embed(img_shapes, txt_seq_lens, device=...)``), so ``txt_seq_lens``
    binds positionally to ``device`` and the call raises "got multiple values for
    argument 'device'". This is nunchaku-ai/nunchaku#884, closed by a stale bot
    rather than fixed; nunchaku ``main`` still carries the same line, so upgrading
    the wheel does not help. Pinning diffusers back is blocked by this package's
    own ``diffusers[torch]>=0.39.0`` floor.

    The conversion below is diffusers' own 0.38 deprecation branch, verbatim from
    ``transformer_qwenimage.py`` in v0.38.0::

        # Use max of txt_seq_lens for backward compatibility
        max_txt_seq_len = max(txt_seq_lens) if isinstance(txt_seq_lens, list) else txt_seq_lens

    so this restores removed behaviour rather than inventing semantics.

    The shim self-disables: it applies only while ``forward`` accepts
    ``max_txt_seq_len`` and no longer accepts ``txt_seq_lens``. If nunchaku fixes
    its call, or diffusers changes shape again, this becomes a no-op. Delete it
    once nunchaku calls ``max_txt_seq_len``.

    Returns:
        True if the shim is now installed, False if it was not needed.
    """
    import inspect

    from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope

    forward = QwenEmbedRope.forward
    if getattr(forward, "_nodetool_qwen_rope_shim", False):
        return True  # idempotent: already wrapped

    params = inspect.signature(forward).parameters
    if "max_txt_seq_len" not in params or "txt_seq_lens" in params:
        # Either an older diffusers that still accepts the legacy call, or a
        # future signature this translation would not match. Leave it alone.
        return False

    torch = _get_torch()

    def _forward(self, video_fhw, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Only a legacy positional `txt_seq_lens` is rewritten. A device passed
        # positionally (torch.device, a "cuda" string, or None) falls through.
        if args and (
            isinstance(args[0], (list, tuple, int)) or torch.is_tensor(args[0])
        ):
            legacy, *rest = args
            args = tuple(rest)
            if kwargs.get("max_txt_seq_len") is None:
                kwargs["max_txt_seq_len"] = (
                    max(legacy) if isinstance(legacy, list) else legacy
                )
        return forward(self, video_fhw, *args, **kwargs)

    _forward._nodetool_qwen_rope_shim = True  # type: ignore[attr-defined]
    QwenEmbedRope.forward = _forward  # type: ignore[method-assign]
    log.info(
        "Applied QwenEmbedRope compatibility shim for nunchaku on diffusers >= 0.39 "
        "(see nunchaku-ai/nunchaku#884)"
    )
    return True


async def load_nunchaku_qwen_pipeline(
    context: "ProcessingContext",
    repo_id: str,
    transformer_path: str,
    node_id: str | None,
    pipeline_class: Any,
    base_model_id: str,
    cache_key: str | None = None,
    torch_dtype: Any = None,  # Defaults to torch.bfloat16 at runtime
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
    _require_nunchaku()

    from nunchaku import NunchakuQwenImageTransformer2DModel
    from nunchaku.utils import get_gpu_memory
    from nodetool.ml.core.model_manager import ModelManager

    # nunchaku's Qwen transformer calls QwenEmbedRope with the pre-0.39
    # signature. Repair it before the transformer runs.
    apply_qwen_rope_compat_shim()

    log_memory("load_nunchaku_qwen_pipeline - START")

    hf_token = await context.get_secret("HF_TOKEN")
    torch = _get_torch()
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    if cache_key is None:
        cache_key = f"{repo_id}/{transformer_path}"

    cached_pipeline = ModelManager.get_model(cache_key)
    if cached_pipeline:
        log.info(f"[MEMORY] Returning cached pipeline for {cache_key}")
        return cached_pipeline

    # Run GC before loading to free any unused memory
    run_gc("Before loading Nunchaku Qwen pipeline")

    try:
        # Load Nunchaku transformer
        with MemoryTracker("Loading Nunchaku Qwen transformer", run_gc_after=False):
            transformer = await get_nunchaku_transformer(
                context=context,
                model_class=NunchakuQwenImageTransformer2DModel,
                node_id=node_id,
                repo_id=repo_id,
                path=transformer_path,
                device=context.device,
            )

        def _build_pipeline():
            return pipeline_class.from_pretrained(
                base_model_id,
                transformer=transformer,
                torch_dtype=torch_dtype,
                token=hf_token,
            )

        with MemoryTracker("Building Qwen pipeline from pretrained", run_gc_after=True):
            pipeline = await asyncio.to_thread(_build_pipeline)

        # Exactly one offload strategy must be applied to a pipeline. Going
        # through the helper records which one, so callers that later ask for
        # offload again don't tear this strategy down and replace it.
        if get_gpu_memory() > 18:
            log.info("Enabling model CPU offload")
            apply_cpu_offload_if_needed(pipeline, method="model")
        else:
            log.info("Enabling model per-layer offloading")
            # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
            transformer.set_offload(
                True, use_pin_memory=False, num_blocks_on_gpu=20
            )  # increase num_blocks_on_gpu if you have more VRAM
            pipeline._exclude_from_cpu_offload.append("transformer")
            apply_cpu_offload_if_needed(pipeline, method="sequential")

        if cache_key:
            ModelManager.set_model(node_id, cache_key, pipeline)

        log_memory("load_nunchaku_qwen_pipeline - END")
        return pipeline
    except _get_torch().OutOfMemoryError as exc:
        run_gc("After OOM error in Qwen pipeline")
        raise ValueError(
            "VRAM out of memory while loading Qwen with the Nunchaku transformer."
        ) from exc


def get_nunchaku_sdxl_unet_class() -> type:
    """Get the NunchakuSDXLUNet2DConditionModel class if available."""
    _require_nunchaku()
    from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel

    return NunchakuSDXLUNet2DConditionModel
