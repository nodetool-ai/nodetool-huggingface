"""
Image-to-image pipeline loading for local HuggingFace models.
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


async def load_image_to_image_pipeline(
    *,
    context: ProcessingContext,
    model_id: str,
    model_path: str | None,
    node_id: str | None,
    revision: str | None = None,
    cache_dir: str | None = None,
    cache_key: str | None = None,
    device: str | None = None,
) -> tuple[Any, bool]:
    """
    Load an image-to-image pipeline with caching and model routing.

    Returns a tuple of (pipeline, use_cpu_offload).
    """
    if not model_id:
        raise ValueError("Please select a model")

    use_cpu_offload = False

    # Determine expected CPU offload behavior based on model_id/model_path,
    # so that cached pipelines return the correct flag as well.
    if model_path:
        from nodetool.nodes.huggingface.image_to_image import QwenImageEdit

        if _is_node_model(model_id, model_path, QwenImageEdit):
            use_cpu_offload = True

    cache_key = cache_key or f"image-to-image:{model_id}:{model_path or 'repo'}"
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
                revision=revision,
                cache_dir=cache_dir,
            )

        from nodetool.nodes.huggingface.image_to_image import FluxFill, QwenImageEdit

        if _is_node_model(model_id, model_path, FluxFill):
            from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline

            if is_nunchaku_transformer(model_id, model_path):
                pipeline = await load_nunchaku_flux_pipeline(
                    context=context,
                    repo_id=model_id,
                    transformer_path=model_path,
                    node_id=node_id,
                    pipeline_class=FluxFillPipeline,
                )
            else:
                torch = _get_torch()
                pipeline = FluxFillPipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=(
                        torch.bfloat16 if _is_cuda_available() else torch.float32
                    ),
                )
        elif _is_node_model(model_id, model_path, QwenImageEdit):
            from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
                QwenImageEditPipeline,
            )

            if is_nunchaku_transformer(model_id, model_path):
                pipeline = await load_nunchaku_qwen_pipeline(
                    context=context,
                    repo_id=model_id,
                    transformer_path=model_path,
                    node_id=node_id,
                    pipeline_class=QwenImageEditPipeline,
                    base_model_id="Qwen/Qwen-Image-Edit",
                    torch_dtype=_get_torch().bfloat16,
                )
                use_cpu_offload = True
            else:
                torch = _get_torch()
                pipeline = QwenImageEditPipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=torch.float16,
                )
                use_cpu_offload = True
        elif is_nunchaku_transformer(model_id, model_path):
            pipeline = await load_nunchaku_flux_pipeline(
                context=context,
                repo_id=model_id,
                transformer_path=model_path,
                node_id=node_id,
            )
        else:
            model_info = await fetch_model_info(model_id)
            if model_info is None:
                raise ValueError(f"Model {model_id} not found")
            if not model_info.tags:
                raise ValueError(f"Model {model_id} has no tags")

            if "diffusers:StableDiffusionPipeline" in model_info.tags:
                from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
                    StableDiffusionImg2ImgPipeline,
                )

                pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=(
                        _get_torch().float16
                        if _is_cuda_available()
                        else _get_torch().float32
                    ),
                )
            elif "diffusers:StableDiffusionXLPipeline" in model_info.tags:
                from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
                    StableDiffusionXLImg2ImgPipeline,
                )

                pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=(
                        _get_torch().float16
                        if _is_cuda_available()
                        else _get_torch().float32
                    ),
                )
            elif "diffusers:StableDiffusion3Pipeline" in model_info.tags:
                from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import (
                    StableDiffusion3Img2ImgPipeline,
                )

                pipeline = StableDiffusion3Img2ImgPipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=(
                        _get_torch().float16
                        if _is_cuda_available()
                        else _get_torch().float32
                    ),
                )
            elif "flux" in model_info.tags:
                from diffusers.pipelines.flux.pipeline_flux_img2img import (
                    FluxImg2ImgPipeline,
                )

                pipeline = FluxImg2ImgPipeline.from_single_file(
                    str(cache_path),
                    torch_dtype=(
                        _get_torch().bfloat16
                        if _is_cuda_available()
                        else _get_torch().float32
                    ),
                )
            else:
                raise ValueError(
                    f"Unsupported single-file model type: {model_info.pipeline_tag}"
                )
    else:
        from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image

        torch = _get_torch()
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if _is_cuda_available() else torch.float32,
            variant=await _detect_cached_variant(model_id),
        )

    if not use_cpu_offload:
        target_device = _resolve_hf_device(context, device or context.device)
        pipeline.to(target_device)

    ModelManager.set_model(node_id, cache_key, pipeline)
    return pipeline, use_cpu_offload
