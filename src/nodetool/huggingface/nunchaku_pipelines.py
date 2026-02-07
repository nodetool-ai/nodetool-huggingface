"""
Nunchaku pipeline utilities for optimized model loading.

This module provides utilities for loading nunchaku-optimized models
(transformers and text encoders) when the nunchaku package is available.
All nunchaku-specific code is isolated here to allow the main huggingface
module to work without nunchaku installed.
"""

from typing import Any, TYPE_CHECKING

from nodetool.config.logging_config import get_logger
from nodetool.huggingface.flux_utils import (
    detect_flux_variant,
    flux_variant_to_base_model_id,
)
from nodetool.huggingface.local_provider_utils import _get_torch, load_model
from nodetool.huggingface.memory_utils import log_memory, run_gc, MemoryTracker
from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE

if TYPE_CHECKING:
    import torch
    from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


# Check if nunchaku is available
try:
    import nunchaku

    NUNCHAKU_AVAILABLE = True
except ImportError:
    NUNCHAKU_AVAILABLE = False


def is_nunchaku_available() -> bool:
    """Check if the nunchaku package is available."""
    return NUNCHAKU_AVAILABLE


def _require_nunchaku() -> None:
    """Raise an error if nunchaku is not available."""
    if not NUNCHAKU_AVAILABLE:
        raise ImportError(
            "nunchaku is required for this operation but is not installed. "
            "nunchaku is only available on non-Darwin platforms."
        )


async def get_nunchaku_text_encoder(
    context: "ProcessingContext",
    node_id: str,
    repo_id: str | None = None,
    path: str | None = None,
    allow_downloads: bool = True,
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
    _require_nunchaku()

    from nunchaku import NunchakuT5EncoderModel
    from nunchaku.utils import get_precision
    from huggingface_hub import hf_hub_download

    if repo_id is None:
        repo_id = "nunchaku-tech/nunchaku-t5"
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
        hf_hub_download(repo_id, path)
        cache_path = await HF_FAST_CACHE.resolve(repo_id, path)

        if not cache_path:
            raise ValueError(
                f"Downloading model {repo_id}/{path} from HuggingFace failed"
            )

    torch = _get_torch()
    return await load_model(
        context=context,
        model_id=str(cache_path),
        model_class=NunchakuT5EncoderModel,
        node_id=node_id,
        torch_dtype=torch.bfloat16,
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

    transformer_identifier = cache_path or f"{repo_id}/{path}"

    transformer = await load_model(
        context=context,
        model_id=transformer_identifier,
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
        )
        transformer.set_attention_impl("nunchaku-fp16")

    with MemoryTracker("Loading Nunchaku text encoder", run_gc_after=False):
        text_encoder = await get_nunchaku_text_encoder(context, node_key)

    pipeline_kwargs: dict[str, Any] = {
        "transformer": transformer,
        "torch_dtype": torch_dtype,
        "token": hf_token,
    }
    if text_encoder is not None:
        pipeline_kwargs["text_encoder_2"] = text_encoder

    try:
        with MemoryTracker(
            f"Building {pipeline_name} from pretrained", run_gc_after=True
        ):
            pipeline = pipeline_class.from_pretrained(base_model_id, **pipeline_kwargs)

        if cache_key and node_id:
            ModelManager.set_model(node_id, cache_key, pipeline)

        log_memory("load_nunchaku_flux_pipeline - END")
        return pipeline
    except _get_torch().OutOfMemoryError as exc:  # type: ignore[attr-defined]
        run_gc("After OOM error")
        raise ValueError(
            "VRAM out of memory while loading Flux with the Nunchaku transformer. "
            "Try enabling 'CPU offload' or reduce image size/steps."
        ) from exc


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

        with MemoryTracker("Building Qwen pipeline from pretrained", run_gc_after=True):
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
                True, use_pin_memory=False, num_blocks_on_gpu=20
            )  # increase num_blocks_on_gpu if you have more VRAM
            pipeline._exclude_from_cpu_offload.append("transformer")
            pipeline.enable_sequential_cpu_offload()

        if cache_key and node_id:
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
