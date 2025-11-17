"""
HuggingFace local provider implementation.

This module implements the BaseProvider interface for locally cached HuggingFace models.
- Language models: Only GGUF models are supported, using transformers for inference
  with model IDs in format "repo_id:filename.gguf" (same as LlamaCpp provider)
- Image models: Text2Image and ImageToImage using diffusion pipelines
  Supports both multi-file models (repo_id) and single-file models (repo_id:path.safetensors)
- TTS models: KokoroTTS and other HuggingFace TTS models
"""

import asyncio
import base64
from queue import Queue
import threading
import re
from typing import Any, AsyncGenerator, List, Literal, Set
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.integrations.huggingface.huggingface_models import (
    fetch_model_info,
    get_image_to_image_models_from_hf_cache,
    get_text_to_image_models_from_hf_cache,
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
from huggingface_hub import hf_hub_download, try_to_load_from_cache, _CACHED_NO_EXIST


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
from diffusers.quantizers.quantization_config import GGUFQuantizationConfig
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

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
    T5EncoderModel,
    TextStreamer,
    pipeline as create_pipeline,
)
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
)

from nodetool.metadata.types import HFTextToSpeech
from pydub import AudioSegment
from io import BytesIO
from nodetool.metadata.types import HFTextToSpeech

from nodetool.workflows.recommended_models import get_recommended_models
from nodetool.metadata.types import LanguageModel, VideoRef
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from nodetool.integrations.huggingface.huggingface_models import (
    get_llamacpp_language_models_from_hf_cache,
)
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

log = get_logger(__name__)


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
    """Load a HuggingFace pipeline model."""
    if model_id == "" or model_id is None:
        raise ValueError("Please select a model")

    cached_model = ModelManager.get_model(model_id, pipeline_task)
    if cached_model:
        return cached_model

    if device is None:
        device = context.device

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
                cache_path = try_to_load_from_cache(
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
            raise ValueError(f"Model {model_id} must be downloaded first")

    context.post_message(
        JobUpdate(
            status="running",
            message=f"Loading pipeline {type(model_id) == str and model_id or pipeline_task} from HuggingFace",
        )
    )
    if not "token" in kwargs:
        kwargs["token"] = context.get_secret("HF_TOKEN")
    model = pipeline(
        pipeline_task,  # type: ignore
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        **kwargs,
    )  # type: ignore
    ModelManager.set_model(node_id, model_id, pipeline_task, model)
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
    **kwargs: Any,
) -> T:
    """Load a HuggingFace model."""
    if model_id == "":
        raise ValueError("Please select a model")

    if not skip_cache:
        cached_model = ModelManager.get_model(model_id, model_class.__name__, path)
        if cached_model:
            return cached_model

    if path:
        cache_path = try_to_load_from_cache(model_id, path)
        if not cache_path:
            context.post_message(
                JobUpdate(
                    status="running",
                    message=f"Downloading model {model_id}/{path} from HuggingFace",
                )
            )
            hf_hub_download(model_id, path)
            cache_path = try_to_load_from_cache(model_id, path)
            if not cache_path:
                raise ValueError(
                    f"Downloading model {model_id}/{path} from HuggingFace failed"
                )

        log.info(f"Loading model {model_id}/{path} from {cache_path}")
        context.post_message(
            JobUpdate(
                status="running",
                message=f"Loading model {model_id} from {cache_path}",
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
        if not "token" in kwargs:
            kwargs["token"] = context.get_secret("HF_TOKEN")

        model = model_class.from_pretrained(  # type: ignore
            model_id,
            torch_dtype=torch_dtype,
            variant=variant,
            **kwargs,
        )

    ModelManager.set_model(node_id, model_id, model_class.__name__, model, path)
    return model


def _detect_cached_variant(repo_id: str) -> str | None:
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
        p = try_to_load_from_cache(repo_id, fname)
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


def _is_qwen_image_gguf_model(repo_id: str, file_path: str | None) -> bool:
    """Check if the repo/path pair corresponds to a Qwen-Image GGUF model."""
    if not file_path:
        return False
    combined = f"{repo_id}:{file_path}".lower()
    return file_path.lower().endswith(".gguf") and "qwen-image" in combined


def _is_flux_gguf_model(repo_id: str, file_path: str | None) -> bool:
    """Check if the repo/path pair corresponds to a FLUX GGUF model."""
    if not file_path:
        return False
    combined = f"{repo_id}:{file_path}".lower()
    return file_path.lower().endswith(".gguf") and "flux" in combined


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


async def _load_flux_gguf_pipeline(
    repo_id: str,
    file_path: str,
    context: ProcessingContext,
    task: Literal["text2image", "image2image"],
    node_id: str | None = None,
):
    """Load a FLUX GGUF quantized transformer and wrap it in the requested pipeline."""

    cache_path = try_to_load_from_cache(repo_id, file_path)
    if not cache_path:
        raise ValueError(f"Model {repo_id}/{file_path} must be downloaded first")

    variant = _detect_flux_variant(repo_id, file_path)
    torch_dtype = torch.bfloat16 if variant in {"schnell", "dev"} else torch.float16

    transformer = await load_model(
        node_id=node_id,
        context=context,
        model_class=FluxTransformer2DModel,
        model_id=repo_id,
        path=file_path,
        torch_dtype=torch_dtype,
        skip_cache=False,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
    )

    base_model_id = _flux_variant_to_base_model_id(variant)
    log.info(
        f"Initializing FLUX {task} pipeline from {base_model_id} with quantized transformer..."
    )
    context.post_message(
        JobUpdate(
            status="running",
            message="Downloading FLUX pipelines components...",
        )
    )
    if task == "image2image":
        pipeline = FluxImg2ImgPipeline.from_pretrained(
            base_model_id,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )
    elif task == "text2image":
        pipeline = FluxPipeline.from_pretrained(
            base_model_id,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError(f"Unsupported FLUX gguf task: {task}")

    pipeline.enable_sequential_cpu_offload()
    return pipeline


async def load_qwen_image_gguf_pipeline(
    model_id: str,
    path: str,
    context: ProcessingContext,
    torch_dtype: torch.dtype,
    node_id: str | None = None,
):
    """Load Qwen-Image model with GGUF quantization."""
    log.info(f"Loading Qwen-Image model: {model_id}/{path}")

    cache_path = try_to_load_from_cache(model_id, path)
    if not cache_path:
        raise ValueError(f"Model {model_id}/{path} must be downloaded first")

    log.debug(f"Cache path: {cache_path}")
    log.debug(f"Torch dtype: {torch_dtype}")

    transformer = await load_model(
        node_id=node_id,
        context=context,
        model_class=QwenImageTransformer2DModel,
        model_id=model_id,
        path=path,
        torch_dtype=torch_dtype,
        skip_cache=False,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
        config="Qwen/Qwen-Image",
        subfolder="transformer",
        device="cpu",
    )

    # Create the pipeline with the quantized transformer
    log.info("Creating Qwen-Image pipeline with quantized transformer...")
    context.post_message(
        JobUpdate(
            status="running",
            message="Downloading Qwen-Image pipelines components...",
        )
    )

    pipeline = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image",
        transformer=transformer,
        torch_dtype=torch_dtype,
        device="cpu",
    )
    pipeline.enable_model_cpu_offload()
    pipeline.enable_attention_slicing()

    return pipeline


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

        cache_key = f"{params.model.id}:text2image"
        pipeline = ModelManager.get_model(cache_key, "text2image")

        if not pipeline:
            log.info(f"Loading text-to-image pipeline: {params.model.id}")
            use_cpu_offload = False

            # Check if model_id is in "repo_id:path" format (single-file model)
            if params.model.path:
                # Verify the file is cached locally
                cache_path = try_to_load_from_cache(params.model.id, params.model.path)
                if not cache_path:
                    raise ValueError(
                        f"Single-file model {params.model.id}/{params.model.path} must be downloaded first"
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

                if _is_flux_gguf_model(params.model.id, params.model.path):
                    pipeline = await _load_flux_gguf_pipeline(
                        repo_id=params.model.id,
                        file_path=params.model.path,
                        context=context,
                        task="text2image",
                        node_id=node_id,
                    )
                    use_cpu_offload = True
                elif _is_qwen_image_gguf_model(params.model.id, params.model.path):
                    pipeline = await load_qwen_image_gguf_pipeline(
                        model_id=params.model.id,
                        path=params.model.path,
                        context=context,
                        torch_dtype=torch.bfloat16,
                        node_id=node_id,
                    )
                    use_cpu_offload = True
                else:
                    model_type = model_info.pipeline_tag or "unknown"

                    # Load pipeline from single file based on model type
                    if "diffusers:StableDiffusionXLPipeline" in model_info.tags:
                        pipeline = StableDiffusionXLPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16
                                if _is_cuda_available()
                                else torch.float32
                            ),
                        )
                    elif "diffusers:StableDiffusionPipeline" in model_info.tags:
                        pipeline = StableDiffusionPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16
                                if _is_cuda_available()
                                else torch.float32
                            ),
                        )
                    elif "diffusers:StableDiffusion3Pipeline" in model_info.tags:
                        pipeline = StableDiffusion3Pipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16
                                if _is_cuda_available()
                                else torch.float32
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
                    variant=_detect_cached_variant(params.model.id),
                )

            if not use_cpu_offload:
                pipeline.to(context.device)

            # Cache the pipeline
            ModelManager.set_model(cache_key, cache_key, "text2image", pipeline)

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

        pipeline.to("cpu")

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
        cache_key = f"{params.model.id}:image2image"
        pipeline = ModelManager.get_model(cache_key, "image2image")

        if not pipeline:
            log.info(f"Loading image-to-image pipeline: {params.model.id}")
            use_cpu_offload = False

            # Check if model_id is in "repo_id:path" format (single-file model)
            if params.model.path:
                # Verify the file is cached locally
                cache_path = try_to_load_from_cache(params.model.id, params.model.path)
                if not cache_path:
                    raise ValueError(
                        f"Single-file model {params.model.id}/{params.model.path} must be downloaded first"
                    )

                if _is_flux_gguf_model(params.model.id, params.model.path):
                    pipeline = await _load_flux_gguf_pipeline(
                        repo_id=params.model.id,
                        file_path=params.model.path,
                        context=context,
                        task="image2image",
                        node_id=node_id,
                    )
                    use_cpu_offload = True
                else:
                    # Verify the file is cached locally
                    cache_path = try_to_load_from_cache(
                        params.model.id, params.model.path
                    )
                    if not cache_path:
                        raise ValueError(
                            f"Single-file model {params.model.id}/{params.model.path} must be downloaded first"
                        )

                    model_info = await fetch_model_info(params.model.id)
                    if model_info is None:
                        raise ValueError(f"Model {params.model.id} not found")

                    if model_info.pipeline_tag != "image-to-image":
                        raise ValueError(
                            f"Model {params.model.id} is not an image-to-image model"
                        )

                    if not model_info.tags:
                        raise ValueError(f"Model {params.model.id} has no tags")

                    # Load pipeline from single file based on model type
                    if "diffusers:StableDiffusionPipeline" in model_info.tags:
                        pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16
                                if _is_cuda_available()
                                else torch.float32
                            ),
                        )
                    elif "diffusers:StableDiffusionXLPipeline" in model_info.tags:
                        pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16
                                if _is_cuda_available()
                                else torch.float32
                            ),
                        )
                    elif "diffusers:StableDiffusion3Pipeline" in model_info.tags:
                        pipeline = StableDiffusion3Img2ImgPipeline.from_single_file(
                            str(cache_path),
                            torch_dtype=(
                                torch.float16
                                if _is_cuda_available()
                                else torch.float32
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
                    variant=_detect_cached_variant(params.model.id),
                )

            assert pipeline is not None
            if not use_cpu_offload:
                pipeline.to(context.device)

            # Cache the pipeline
            ModelManager.set_model(cache_key, cache_key, "image2image", pipeline)

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

        pipeline.to("cpu")

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
        cache_key = f"{model}:asr"
        asr_pipeline = ModelManager.get_model(cache_key, "automatic-speech-recognition")

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
            ModelManager.set_model(
                cache_key, cache_key, "automatic-speech-recognition", asr_pipeline
            )

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
        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace text-to-video generation"
            )

        # Get or load the pipeline
        model_id = model
        cache_key = f"{model_id}:text2video"
        pipeline = ModelManager.get_model(cache_key, "text2video")

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

            # Cache the pipeline
            ModelManager.set_model(cache_key, cache_key, "text2video", pipeline)

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
        """Get available HuggingFace GGUF language models.

        Returns GGUF models available in the local HuggingFace cache.
        Uses the same model ID syntax as LlamaCpp provider: "repo_id:filename.gguf"

        Returns:
            List of LanguageModel instances for HuggingFace GGUF models
        """
        models = await get_llamacpp_language_models_from_hf_cache()
        # Update provider to HuggingFace for these models
        hf_models = [
            LanguageModel(
                id=model.id,  # Already in "repo_id:filename" format
                name=model.name,
                provider=Provider.HuggingFace,
                supported_tasks=["text_generation"],
            )
            for model in models
        ]
        log.debug(f"Found {len(hf_models)} HuggingFace GGUF models in HF cache")
        return hf_models

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
    def _parse_model_spec(model: str) -> tuple[str, str | None, bool]:
        """Return repo_id, optional filename, and GGUF flag from model spec."""
        if ":" not in model:
            return model, None, False
        parts = model.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid model spec: {model}")
        repo_id, filename = parts
        is_gguf = filename.lower().endswith(".gguf")
        return repo_id, filename, is_gguf

    @staticmethod
    def _build_prompt_from_messages(messages: Sequence[Message]) -> str:
        """Convert simple text-only chat history into a prompt string."""
        system_parts: list[str] = []
        user_prompt: str | None = None

        for msg in messages:
            if isinstance(msg.content, str):
                content = msg.content.strip()
            elif msg.content is None:
                content = ""
            else:
                raise ValueError(
                    "HuggingFace local provider only supports text messages for local models"
                )

            if msg.role == "system" and content:
                system_parts.append(content)
            elif msg.role == "user" and content:
                user_prompt = content

        if user_prompt is None:
            raise ValueError(
                "HuggingFace text generation requires at least one user message with text content"
            )

        if system_parts:
            return "\n\n".join(system_parts + [user_prompt])
        return user_prompt

    async def _stream_gguf_generation(
        self,
        messages: Sequence[Message],
        repo_id: str,
        filename: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> AsyncIterator[Chunk]:
        if not filename:
            raise ValueError("GGUF model path is required for HuggingFace local models")

        chat: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                raise ValueError(
                    "HuggingFace GGUF models do not support multimodal content. "
                    "Please use text-only messages."
                )
            else:
                content = ""
            chat.append({"role": msg.role, "content": content})

        cache_key = f"{repo_id}:{filename}:text-generation"
        cached_pipeline = ModelManager.get_model(cache_key, "text-generation")

        if not cached_pipeline:
            cache_path = try_to_load_from_cache(
                repo_id, filename
            )  # pyright: ignore[reportArgumentType]
            if not cache_path:
                raise ValueError(f"Model {repo_id}/{filename} must be downloaded first")

            log.info(f"Loading GGUF model {repo_id}/{filename}")
            # Note: load_model doesn't support gguf_file parameter, so we load manually
            # but still use ModelManager for caching consistency
            cached_model = ModelManager.get_model(
                repo_id, AutoModelForCausalLM.__name__, filename
            )
            if not cached_model:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    gguf_file=filename,
                )
                ModelManager.set_model(
                    repo_id, AutoModelForCausalLM.__name__, filename, hf_model
                )
            else:
                hf_model = cached_model

            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                gguf_file=filename,
            )
            cached_pipeline = pipeline(
                "text-generation", model=hf_model, tokenizer=tokenizer
            )
            ModelManager.set_model(
                cache_key, cache_key, "text-generation", cached_pipeline
            )

        tokenizer = cached_pipeline.tokenizer
        assert tokenizer is not None
        formatted_prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
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
            cached_pipeline(formatted_prompt, **generation_kwargs)  # type: ignore[reportArgumentType]

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

    async def _stream_pipeline_generation(
        self,
        repo_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        context: ProcessingContext,
        node_id: str | None,
    ) -> AsyncIterator[Chunk]:
        # Check cache with our key format for backward compatibility
        cache_key = f"{repo_id}:text-generation"
        cached_pipeline = ModelManager.get_model(cache_key, "text-generation")

        # Also check with load_pipeline's cache format
        if not cached_pipeline:
            cached_pipeline = ModelManager.get_model(repo_id, "text-generation")

        if not cached_pipeline:
            log.info(f"Loading HuggingFace pipeline model {repo_id}")
            cached_pipeline = await load_pipeline(
                node_id=node_id,
                context=context,
                pipeline_task="text-generation",
                model_id=repo_id,
            )
            # Cache with our key format for backward compatibility
            ModelManager.set_model(
                cache_key, cache_key, "text-generation", cached_pipeline
            )

        tokenizer = cached_pipeline.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer missing from HuggingFace pipeline")

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

        repo_id, filename, is_gguf = self._parse_model_spec(model)

        if is_gguf:
            async for chunk in self._stream_gguf_generation(
                messages=messages,
                repo_id=repo_id,
                filename=filename,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            ):
                yield chunk
            return

        prompt = self._build_prompt_from_messages(messages)

        async for chunk in self._stream_pipeline_generation(
            repo_id=repo_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            context=context,
            node_id=node_id,
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

    # Run tests
    print("=" * 50)
    print("Testing HuggingFace Local Provider")
    print("=" * 50)

    # asyncio.run(test_available_language_models())

    # # Test streaming
    asyncio.run(test_generate_messages())

    # # Test non-streaming
    # asyncio.run(test_generate_message())
    # asyncio.run(test_text_to_image())
    # asyncio.run(test_image_to_image())
