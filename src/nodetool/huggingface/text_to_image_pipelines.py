"""
Text-to-image pipeline loading for local HuggingFace models.

This module handles the loading of various HuggingFace diffusers pipelines for text-to-image generation.
It determines the correct pipeline class as follows:

0. **Local Checkpoint (Before Anything Else):**
   - A model reference naming a file on disk is loaded straight from that file.
     Such a file has no Hub tags, so the family is read from its own tensor names
     by `single_file_models.detect_single_file_checkpoint`. Steps 1-3 below all
     depend on Hub metadata and do not apply.

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

import asyncio
from typing import Any

from nodetool.huggingface.flux_utils import (
    is_nunchaku_flux_transformer,
    is_nunchaku_qwen_transformer,
    is_nunchaku_transformer,
)
from nodetool.huggingface.local_provider_utils import (
    _apply_memory_optimizations,
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
from nodetool.huggingface.single_file_models import (
    SingleFileLoadPlan,
    detect_single_file_checkpoint,
    resolve_local_checkpoint,
)
from nodetool.integrations.huggingface.huggingface_models import (
    HF_FAST_CACHE,
    fetch_model_info,
)
from nodetool.ml.core.model_manager import ModelManager
from nodetool.workflows.processing_context import ProcessingContext

# diffusers `_class_name` (from a repo's model_index.json) -> (module, class) for
# full-repo text-to-image pipelines we load explicitly. AutoPipelineForText2Image
# does not cover all of these (e.g. Kandinsky5T2I, QwenImageLayered).
_EXPLICIT_T2I_PIPELINES: dict[str, tuple[str, str]] = {
    "Flux2Pipeline": ("diffusers.pipelines.flux2.pipeline_flux2", "Flux2Pipeline"),
    "Flux2KleinPipeline": (
        "diffusers.pipelines.flux2.pipeline_flux2_klein",
        "Flux2KleinPipeline",
    ),
    "GlmImagePipeline": (
        "diffusers.pipelines.glm_image.pipeline_glm_image",
        "GlmImagePipeline",
    ),
    "Kandinsky5T2IPipeline": (
        "diffusers.pipelines.kandinsky5.pipeline_kandinsky_t2i",
        "Kandinsky5T2IPipeline",
    ),
    "QwenImageLayeredPipeline": (
        "diffusers.pipelines.qwenimage.pipeline_qwenimage_layered",
        "QwenImageLayeredPipeline",
    ),
}


async def _resolve_full_repo_pipeline_class(model_id: str):
    """Resolve an explicit diffusers pipeline class for a cached full-repo model.

    Reads the repo's ``model_index.json`` ``_class_name`` and maps it to a known
    pipeline class. Returns ``None`` to fall back to ``AutoPipelineForText2Image``.
    """
    import importlib
    import json as _json

    try:
        path = await HF_FAST_CACHE.resolve(model_id, "model_index.json")
        if not isinstance(path, str):
            return None
        with open(path) as f:
            class_name = _json.load(f).get("_class_name")
    except Exception:
        return None

    entry = _EXPLICIT_T2I_PIPELINES.get(class_name or "")
    if entry is None:
        return None
    module_name, cls_name = entry
    return getattr(importlib.import_module(module_name), cls_name)


async def load_local_single_file_pipeline(
    checkpoint_path: str,
    *,
    plan: SingleFileLoadPlan | None = None,
) -> Any:
    """Build a text-to-image pipeline from one checkpoint file on disk.

    The family is read from the file's own tensor names, because a local file has
    no Hub tags to read.  Weights are never copied into the diffusers cache.
    """
    import importlib

    plan = plan or detect_single_file_checkpoint(checkpoint_path)
    torch = _get_torch()
    dtype = getattr(torch, plan.dtype) if _is_cuda_available() else torch.float32

    pipeline_cls = getattr(
        importlib.import_module(plan.pipeline_module), plan.pipeline_class
    )

    if plan.load_kind == "pipeline":
        return await asyncio.to_thread(
            pipeline_cls.from_single_file, checkpoint_path, torch_dtype=dtype
        )

    # The pipeline class has no `from_single_file`. Load the transformer from the
    # file and take the text encoder, VAE and scheduler from the base repo.
    assert plan.transformer_module and plan.transformer_class and plan.base_repo_id
    transformer_cls = getattr(
        importlib.import_module(plan.transformer_module), plan.transformer_class
    )
    transformer = await asyncio.to_thread(
        transformer_cls.from_single_file, checkpoint_path, torch_dtype=dtype
    )
    return await asyncio.to_thread(
        pipeline_cls.from_pretrained,
        plan.base_repo_id,
        transformer=transformer,
        torch_dtype=dtype,
    )


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

    # A model reference that names a file on disk is loaded from that file. It
    # has no Hub tags, so the family comes from the checkpoint's own keys.
    local_checkpoint = resolve_local_checkpoint(model_id, model_path)
    if local_checkpoint is not None:
        plan = detect_single_file_checkpoint(local_checkpoint)
        cache_key = cache_key or f"text-to-image:local:{local_checkpoint}"
        cached = ModelManager.get_model(cache_key)
        pipeline = cached or await load_local_single_file_pipeline(
            local_checkpoint, plan=plan
        )
        target_device = _resolve_hf_device(context, device or context.device)
        use_cpu_offload = _apply_memory_optimizations(
            pipeline,
            target_device,
            force_cpu_offload=plan.load_kind == "transformer",
        )
        ModelManager.set_model(node_id, cache_key, pipeline)
        return pipeline, use_cpu_offload

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
                pipeline = await asyncio.to_thread(
                    FluxPipeline.from_single_file,
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
                pipeline = await asyncio.to_thread(
                    QwenImagePipeline.from_single_file,
                    str(cache_path),
                    torch_dtype=torch.bfloat16,
                )
                use_cpu_offload = True
        else:
            torch = _get_torch()
            # Check for Nunchaku transformers before tag-based routing
            # Nunchaku models may not be in the Flux/QwenImage node's recommended
            # list, but they still need the special Nunchaku pipeline
            if is_nunchaku_qwen_transformer(model_id, model_path):
                from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
                    QwenImagePipeline,
                )

                pipeline = await load_nunchaku_qwen_pipeline(
                    context=context,
                    repo_id=model_id,
                    transformer_path=model_path,
                    node_id=node_id,
                    pipeline_class=QwenImagePipeline,
                    base_model_id="Qwen/Qwen-Image",
                    torch_dtype=torch.bfloat16,
                )
                use_cpu_offload = True
            elif is_nunchaku_flux_transformer(model_id, model_path):
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

                pipeline = await asyncio.to_thread(
                    StableDiffusionXLPipeline.from_single_file,
                    str(cache_path),
                    torch_dtype=(
                        torch.float16 if _is_cuda_available() else torch.float32
                    ),
                )
            elif "diffusers:StableDiffusionPipeline" in model_info.tags:
                from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
                    StableDiffusionPipeline,
                )

                pipeline = await asyncio.to_thread(
                    StableDiffusionPipeline.from_single_file,
                    str(cache_path),
                    torch_dtype=(
                        torch.float16 if _is_cuda_available() else torch.float32
                    ),
                )
            elif "diffusers:StableDiffusion3Pipeline" in model_info.tags:
                from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
                    StableDiffusion3Pipeline,
                )

                pipeline = await asyncio.to_thread(
                    StableDiffusion3Pipeline.from_single_file,
                    str(cache_path),
                    torch_dtype=(
                        torch.float16 if _is_cuda_available() else torch.float32
                    ),
                )
            elif "flux" in model_info.tags:
                from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

                pipeline = await asyncio.to_thread(
                    FluxPipeline.from_single_file,
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

                    pipeline = await asyncio.to_thread(
                        StableDiffusionXLPipeline.from_single_file,
                        str(cache_path),
                        torch_dtype=(
                            torch.float16 if _is_cuda_available() else torch.float32
                        ),
                    )
                elif _is_node_model(
                    model_id, model_path, StableDiffusion
                ) or _is_node_model(model_id, model_path, StableDiffusionControlNet):
                    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
                        StableDiffusionPipeline,
                    )

                    pipeline = await asyncio.to_thread(
                        StableDiffusionPipeline.from_single_file,
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

                        pipeline = await asyncio.to_thread(
                            StableDiffusionXLPipeline.from_single_file,
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

                        pipeline = await asyncio.to_thread(
                            StableDiffusionPipeline.from_single_file,
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
        torch = _get_torch()
        # Newer DiT pipelines (Flux.2, GLM-Image, Kandinsky 5, Qwen-Image-Layered)
        # aren't all covered by AutoPipelineForText2Image. Route them explicitly by
        # the diffusers `_class_name` recorded in the repo's model_index.json.
        explicit_cls = await _resolve_full_repo_pipeline_class(model_id)
        if explicit_cls is not None:
            pipeline = await asyncio.to_thread(
                explicit_cls.from_pretrained,
                model_id,
                torch_dtype=torch.bfloat16 if _is_cuda_available() else torch.float32,
            )
        else:
            from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

            pipeline = await asyncio.to_thread(
                AutoPipelineForText2Image.from_pretrained,
                model_id,
                torch_dtype=torch.float16 if _is_cuda_available() else torch.float32,
                variant=await _detect_cached_variant(model_id),
            )

    # Place the pipeline within the available VRAM budget. Large DiT pipelines
    # (Flux, SD3, Flux.2, Qwen-Image, ...) don't fit on GPUs with less than
    # 24GB VRAM in full precision, so CPU offload is installed automatically
    # when the weights exceed the budget instead of moving them wholesale.
    target_device = _resolve_hf_device(context, device or context.device)
    use_cpu_offload = _apply_memory_optimizations(
        pipeline, target_device, force_cpu_offload=use_cpu_offload
    )

    ModelManager.set_model(node_id, cache_key, pipeline)
    return pipeline, use_cpu_offload
