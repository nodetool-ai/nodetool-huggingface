"""
Text-to-image pipeline loading for local HuggingFace models.

This module handles the loading of various HuggingFace diffusers pipelines for text-to-image generation.
It employs a robust strategy to determine the correct pipeline class:

1. **Explicit Node Matching (First Priority):**
   - It checks if the requested model (repo_id + path) matches any "recommended model" defined
     in our specific nodes (e.g., Flux, QwenImage).
   - If a match is found using `_is_node_model`, the model is loaded using the exact pipeline
     class and configuration required by that node (e.g., `FluxPipeline` or `QwenImagePipeline`).
   - This allows explicit overrides for models that might be generic but require specific handling.

2. **Metadata Tag Matching:**
   - If no node match is found, it fetches the model's metadata from HuggingFace.
   - It inspects the `tags` or `pipeline_tag` fields for specific markers like:
     - "diffusers:StableDiffusionXLPipeline"
     - "diffusers:StableDiffusionPipeline"
     - "flux"
   - These tags map directly to their corresponding diffusers pipeline classes.

3. **Fallback Heuristics:**
   - If the tags are generic (e.g., just "text-to-image") and lack specific diffusers markers:
     - It checks against `StableDiffusionXL` and `StableDiffusion` nodes again to see if we know this model.
     - If still unknown, it attempts to load as SDXL first (common for single-file safetensors).
     - If SDXL loading fails, it falls back to standard Stable Diffusion 1.5/2.x.

This multi-layered approach ensures we reliably load both well-tagged models and ambiguous custom checkpoints
using our internal knowledge base of node definitions.
"""

from __future__ import annotations

from typing import Any
from nodetool.huggingface.flux_utils import is_nunchaku_transformer
from nodetool.huggingface.local_provider_utils import (
    _detect_cached_variant,
    _ensure_file_cached,
    _get_torch,
    _is_cuda_available,
    _is_node_model,
    _resolve_hf_device,
)
from nodetool.huggingface.nunchaku_pipelines import (
    load_nunchaku_flux_pipeline,
    load_nunchaku_qwen_pipeline,
)
from nodetool.integrations.huggingface.huggingface_models import (
    HF_FAST_CACHE,
    fetch_model_info,
)
from nodetool.ml.core.model_manager import ModelManager
from nodetool.workflows.processing_context import ProcessingContext


async def load_text_to_image_pipeline(
    *,
    context: ProcessingContext,
    model_id: str,
    model_path: str | None,
    node_id: str | None,
    cache_key: str | None = None,
    device: str | None = None,
) -> tuple[Any, bool]:
    """
    Load a text-to-image pipeline with caching and model routing.

    Returns a tuple of (pipeline, use_cpu_offload).
    """
    if not model_id:
        raise ValueError("Please select a model")

    # Determine whether this model configuration should use CPU offload.
    use_cpu_offload = False
    if model_path:
        # Import here to avoid module-level dependency when not needed.
        from nodetool.nodes.huggingface.text_to_image import QwenImage

        if _is_node_model(model_id, model_path, QwenImage):
            use_cpu_offload = True

    cache_key = cache_key or f"text-to-image:{model_id}:{model_path or 'repo'}"
    cached = ModelManager.get_model(cache_key)
    if cached:
        return cached, use_cpu_offload

    pipeline: Any

    if model_path:
        cache_path = await HF_FAST_CACHE.resolve(model_id, model_path)
        if not cache_path:
            cache_path = await _ensure_file_cached(
                model_id,
                model_path,
            )

        model_info = await fetch_model_info(model_id)
        if model_info is None:
            raise ValueError(f"Model {model_id} not found")
        if model_info.pipeline_tag != "text-to-image":
            raise ValueError(f"Model {model_id} is not a text-to-image model")
        if not model_info.tags:
            raise ValueError(f"Model {model_id} has no tags")

        from nodetool.nodes.huggingface.text_to_image import Flux, QwenImage

        if _is_node_model(model_id, model_path, Flux):
            if is_nunchaku_transformer(model_id, model_path):
                pipeline = await load_nunchaku_flux_pipeline(
                    context=context,
                    repo_id=model_id,
                    transformer_path=model_path,
                    node_id=node_id,
                )
            else:
                from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

                torch = _get_torch()
                pipeline = FluxPipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=(
                        torch.bfloat16 if _is_cuda_available() else torch.float32
                    ),
                )
        elif _is_node_model(model_id, model_path, QwenImage):
            from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
                QwenImagePipeline,
            )

            if is_nunchaku_transformer(model_id, model_path):
                pipeline = await load_nunchaku_qwen_pipeline(
                    context=context,
                    repo_id=model_id,
                    transformer_path=model_path,
                    node_id=node_id,
                    pipeline_class=QwenImagePipeline,
                    base_model_id="Qwen/Qwen-Image",
                    torch_dtype=_get_torch().bfloat16,
                )
                use_cpu_offload = True
            else:
                torch = _get_torch()
                pipeline = QwenImagePipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=torch.bfloat16,
                )
                use_cpu_offload = True
        else:
            torch = _get_torch()
            # Check for Nunchaku Flux transformers before tag-based routing
            # Nunchaku models may not be in Flux node's recommended list, but they
            # still need to be loaded with the special Nunchaku pipeline
            if is_nunchaku_transformer(model_id, model_path):
                pipeline = await load_nunchaku_flux_pipeline(
                    context=context,
                    repo_id=model_id,
                    transformer_path=model_path,
                    node_id=node_id,
                )
            elif "diffusers:StableDiffusionXLPipeline" in model_info.tags:
                from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
                    StableDiffusionXLPipeline,
                )

                pipeline = StableDiffusionXLPipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=(
                        torch.float16 if _is_cuda_available() else torch.float32
                    ),
                )
            elif "diffusers:StableDiffusionPipeline" in model_info.tags:
                from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
                    StableDiffusionPipeline,
                )

                pipeline = StableDiffusionPipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=(
                        torch.float16 if _is_cuda_available() else torch.float32
                    ),
                )
            elif "diffusers:StableDiffusion3Pipeline" in model_info.tags:
                from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
                    StableDiffusion3Pipeline,
                )

                pipeline = StableDiffusion3Pipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=(
                        torch.float16 if _is_cuda_available() else torch.float32
                    ),
                )
            elif "flux" in model_info.tags:
                from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

                pipeline = FluxPipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=(
                        torch.bfloat16 if _is_cuda_available() else torch.float32
                    ),
                )
            elif model_info.pipeline_tag == "text-to-image":
                # Fallback for generic text-to-image models (likely SDXL or SD1.5) if no specific diffusers tag found
                # Check against known node models first
                from nodetool.nodes.huggingface.image_to_image import (
                    StableDiffusionControlNet,
                )
                from nodetool.nodes.huggingface.text_to_image import (
                    StableDiffusion,
                    StableDiffusionXL,
                )

                if _is_node_model(model_id, model_path, StableDiffusionXL):
                     from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
                        StableDiffusionXLPipeline,
                    )
                     pipeline = StableDiffusionXLPipeline.from_single_file(
                        str(cache_path),
                        torch_dtype=(
                            torch.float16 if _is_cuda_available() else torch.float32
                        ),
                    )
                elif _is_node_model(model_id, model_path, StableDiffusion) or _is_node_model(
                    model_id, model_path, StableDiffusionControlNet
                ):
                    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
                        StableDiffusionPipeline,
                    )
                    pipeline = StableDiffusionPipeline.from_single_file(
                        str(cache_path),
                        torch_dtype=(
                            torch.float16 if _is_cuda_available() else torch.float32
                        ),
                    )
                else:
                    # Attempt to load as SDXL first as it's common for single-file safetensors
                    try:
                        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
                            StableDiffusionXLPipeline,
                        )

                        pipeline = StableDiffusionXLPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16 if _is_cuda_available() else torch.float32
                            ),
                        )
                    except Exception:
                        # Fallback to standard SD
                        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
                            StableDiffusionPipeline,
                        )

                        pipeline = StableDiffusionPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16 if _is_cuda_available() else torch.float32
                            ),
                        )
            else:
                raise ValueError(
                    f"Unsupported single-file model type: {model_info.pipeline_tag}"
                )
    else:
        from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

        torch = _get_torch()
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if _is_cuda_available() else torch.float32,
            variant=await _detect_cached_variant(model_id),
        )

    if not use_cpu_offload:
        target_device = _resolve_hf_device(context, device or context.device)
        pipeline.to(target_device)

    ModelManager.set_model(node_id, cache_key, pipeline)
    return pipeline, use_cpu_offload
