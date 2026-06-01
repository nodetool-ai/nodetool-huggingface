from __future__ import annotations
from enum import Enum
import os
import asyncio
import logging
from typing import Any, TypedDict, TYPE_CHECKING, Optional

from pydantic import Field

from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE
from nodetool.config.logging_config import get_logger
from nodetool.workflows.memory_utils import log_memory, run_gc
from nodetool.metadata.types import (
    HFT5,
    HFFlux,
    HFQwenImage,
    HFStableDiffusionXL,
    HFTextToImage,
    HFControlNetFlux,
    HuggingFaceModel,
    ImageRef,
    TorchTensor,
)
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.huggingface.local_provider_utils import pipeline_progress_callback
from nodetool.nodes.huggingface.stable_diffusion_base import (
    available_torch_dtype,
    is_mps_device,
    maybe_enable_cpu_offload,
    StableDiffusionBaseNode,
    StableDiffusionXLBase,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, Notification, LogUpdate

import torch

if TYPE_CHECKING:
    import huggingface_hub
    from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
    from diffusers.pipelines.chroma.pipeline_chroma import ChromaPipeline
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
    from diffusers.pipelines.flux.pipeline_flux_control import FluxControlPipeline
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
        StableDiffusionPipeline,
    )
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
        StableDiffusionXLPipeline,
    )
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

log = get_logger(__name__)


def _get_torch():
    """Lazy import for torch."""
    import torch

    return torch


def _enable_pytorch2_attention(pipeline: Any):
    """Enable PyTorch 2 scaled dot product attention to speed up inference."""
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
        vae.to(memory_format=torch.channels_last)
        log.debug("Set VAE to channels_last memory format")
    except Exception as e:
        log.warning("Failed to set VAE channels_last memory format: %s", e)


class StableDiffusion(StableDiffusionBaseNode):
    """
    Generates images from text prompts using Stable Diffusion 1.x/2.x models.
    image, generation, AI, text-to-image, SD, creative

    Use cases:
    - Create custom illustrations and artwork from text descriptions
    - Generate concept art for games, films, and creative projects
    - Produce unique visual content for marketing and media
    - Explore AI-generated art with extensive community models
    - Build image generation applications with well-understood architecture
    """

    width: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Output image width in pixels. 512 is standard for SD 1.x.",
    )
    height: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Output image height in pixels. 512 is standard for SD 1.x.",
    )
    _pipeline: Any = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["width", "height"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion"

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
            StableDiffusionPipeline,
        )

        await super().preload_model(context)
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            config="Lykon/DreamShaper",
            torch_dtype=self.get_torch_dtype(),
            variant=None,
        )
        assert self._pipeline is not None
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        self._set_scheduler(self.scheduler)
        await self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> OutputType:
        result = await self.run_pipeline(context, width=self.width, height=self.height)
        return {
            "image": result if isinstance(result, ImageRef) else None,
            "latent": result if isinstance(result, TorchTensor) else None,
        }


class StableDiffusionXL(StableDiffusionXLBase):
    """
    Generates high-resolution images from text prompts using Stable Diffusion XL.
    image, generation, AI, text-to-image, SDXL, high-resolution

    Use cases:
    - Create detailed, high-resolution images (1024x1024) from text
    - Generate marketing visuals and product imagery
    - Produce concept art and illustrations with enhanced detail
    - Create stock imagery and visual content for publications
    - Build professional image generation applications
    """

    _pipeline: Any = None

    @classmethod
    def get_title(cls):
        return "Stable Diffusion XL"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
            StableDiffusionXLPipeline,
        )

        torch_dtype = self.get_torch_dtype()
        base_model, pipeline_model_id, transformer_model = self._prepare_sdxl_models()
        await self._load_sdxl_pipeline(
            context=context,
            pipeline_class=StableDiffusionXLPipeline,
            torch_dtype=torch_dtype,
            base_model=base_model,
            pipeline_model_id=pipeline_model_id,
            transformer_model=transformer_model,
            variant=None,
        )
        assert self._pipeline is not None
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        self._set_scheduler(self.scheduler)
        await self._load_ip_adapter()

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def process(self, context) -> OutputType:
        result = await self.run_pipeline(context)
        return {
            "image": result if isinstance(result, ImageRef) else None,
            "latent": result if isinstance(result, TorchTensor) else None,
        }


class LoadTextToImageModel(HuggingFacePipelineNode):
    """
    Loads and validates a Hugging Face text-to-image model for use in downstream nodes.
    model-loader, text-to-image, pipeline

    Use cases:
    - Pre-load text-to-image models before running pipelines
    - Validate model availability and compatibility
    - Configure model settings for Text2Image processing
    """

    repo_id: str = Field(
        default="",
        description="The Hugging Face repository ID for the text-to-image model.",
    )

    async def preload_model(self, context: ProcessingContext):
        torch_dtype = available_torch_dtype()
        await self.load_model(
            context=context,
            model_id=self.repo_id,
            model_class=AutoPipelineForText2Image,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=None,
        )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "repo_id",
        ]

    async def process(self, context: ProcessingContext) -> HFTextToImage:
        return HFTextToImage(
            repo_id=self.repo_id,
            variant=None,
        )


class Text2Image(HuggingFacePipelineNode):
    """
    Generates images from text prompts using AutoPipeline for automatic model detection.
    image, generation, AI, text-to-image, auto, flexible

    Use cases:
    - Generate images with automatic pipeline selection for any supported model
    - Quickly prototype with various text-to-image architectures
    - Build flexible workflows that adapt to different model types
    - Create images without needing pipeline-specific configuration
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(),
        description="The text-to-image model. AutoPipeline automatically selects the correct pipeline type.",
    )

    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="Text description of the image to generate. Be specific for better results.",
    )
    negative_prompt: str = Field(
        default="",
        description="Describe what to avoid in the image (e.g., 'blurry, low quality').",
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. 20-50 is typical; more steps = better quality but slower.",
        ge=1,
        le=100,
    )
    guidance_scale: float = Field(
        default=7.5,
        description="How strongly to follow the prompt. 7-9 is typical for SD models.",
        ge=1.0,
        le=20.0,
    )
    width: int = Field(
        default=512,
        description="Output image width in pixels.",
        ge=64,
        le=2048,
    )
    height: int = Field(
        default=512,
        description="Output image height in pixels.",
        ge=64,
        le=2048,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "HF Text to Image"

    def get_model_id(self) -> str:
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        torch_dtype = available_torch_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.get_model_id(),
            model_class=AutoPipelineForText2Image,
            torch_dtype=torch_dtype,
            variant=None,
        )
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            try:
                self._pipeline.to(device)
            except torch.OutOfMemoryError as e:
                raise ValueError(
                    "VRAM out of memory while moving TextToImage pipeline to device. "
                    "Enable 'CPU offload' in advanced node properties or reduce image size/steps."
                ) from e

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def process(self, context: ProcessingContext) -> OutputType:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # Generate the image off the event loop
        def _run_pipeline_sync():
            call_kwargs = {
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "width": self.width,
                "height": self.height,
                "generator": generator,
                "callback_on_step_end": pipeline_progress_callback(
                    self.id, self.num_inference_steps, context
                ),
            }
            with torch.inference_mode():
                return self._pipeline(**call_kwargs)

        output = await asyncio.to_thread(_run_pipeline_sync)

        image = output.images[0]
        result = await context.image_from_pil(image)

        return {
            "image": result,
            "latent": None,
        }

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "prompt",
            "height",
            "width",
            "seed",
        ]


class FluxVariant(Enum):
    SCHNELL = "schnell"
    DEV = "dev"


class FluxQuantization(Enum):
    FP16 = "fp16"
    FP4 = "fp4"
    INT4 = "int4"


class Flux(HuggingFacePipelineNode):
    """
    Generates high-quality images using Black Forest Labs' FLUX diffusion models with Nunchaku quantization.
    image, generation, AI, text-to-image, flux, quantization, high-quality

    Use cases:
    - Generate high-fidelity images with excellent text rendering
    - Create images with memory-efficient INT4/FP4 quantization
    - Fast generation with FLUX.1-schnell (4 steps)
    - High-quality generation with FLUX.1-dev
    - Build production image generation systems
    """

    variant: FluxVariant = Field(
        default=FluxVariant.DEV,
        description="FLUX variant: 'schnell' for fast 4-step generation, 'dev' for higher quality with more steps.",
    )
    quantization: FluxQuantization = Field(
        default=FluxQuantization.INT4,
        description="Quantization level: INT4/FP4 for lower VRAM, FP16 for full precision.",
    )
    enable_cpu_offload: bool = Field(
        default=False,
        description="Offload model components to CPU to reduce VRAM usage. "
        "Leave off for Nunchaku INT4/FP4 on 8GB+ GPUs; enable only if you hit OOM.",
    )
    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="Text description of the image to generate. FLUX excels at text rendering.",
    )
    guidance_scale: float = Field(
        default=3.5,
        description="Prompt adherence strength. Use 0.0 for schnell, 3-4 for dev.",
        ge=0.0,
        le=20.0,
    )
    width: int = Field(
        default=1024,
        description="Output image width in pixels. 1024 is recommended.",
        ge=64,
        le=2048,
    )
    height: int = Field(
        default=1024,
        description="Output image height in pixels. 1024 is recommended.",
        ge=64,
        le=2048,
    )
    num_inference_steps: int = Field(
        default=20,
        description="Denoising steps. Schnell uses 4 steps; dev uses 20-50.",
        ge=1,
        le=100,
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum prompt length. Use 256 for schnell, 512 for dev.",
        ge=1,
        le=512,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "Flux"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "variant",
            "quantization",
            "prompt",
            "height",
            "width",
            "seed",
        ]

    def _get_base_model(self, variant: FluxVariant) -> HFFlux:
        model_mapping = {
            FluxVariant.SCHNELL: HFFlux(
                repo_id="black-forest-labs/FLUX.1-schnell",
            ),
            FluxVariant.DEV: HFFlux(
                repo_id="black-forest-labs/FLUX.1-dev",
            ),
        }
        model = model_mapping.get(variant)
        if model is None:
            raise ValueError(f"Unknown variant: {variant}")
        return model

    @classmethod
    def get_recommended_models(cls) -> list[HFFlux]:
        allow_patterns = [
            "*.json",
            "*.txt",
            "scheduler/*",
            "vae/*",
            "text_encoder/*",
            "tokenizer/*",
            "tokenizer_2/*",
        ]
        return [
            HFFlux(
                repo_id="black-forest-labs/FLUX.1-schnell",
                allow_patterns=allow_patterns,
            ),
            HFFlux(
                repo_id="black-forest-labs/FLUX.1-dev",
                allow_patterns=allow_patterns,
            ),
            HFFlux(
                repo_id="nunchaku-tech/nunchaku-flux.1-schnell",
                path="svdq-int4_r32-flux.1-schnell.safetensors",
            ),
            HFFlux(
                repo_id="nunchaku-tech/nunchaku-flux.1-schnell",
                path="svdq-fp4_r32-flux.1-schnell.safetensors",
            ),
            HFFlux(
                repo_id="nunchaku-tech/nunchaku-flux.1-dev",
                path="svdq-int4_r32-flux.1-dev.safetensors",
            ),
            HFFlux(
                repo_id="nunchaku-tech/nunchaku-flux.1-dev",
                path="svdq-fp4_r32-flux.1-dev.safetensors",
            ),
            HFT5(
                repo_id="nunchaku-tech/nunchaku-t5",
                path="awq-int4-flux.1-t5xxl.safetensors",
            ),
        ]

    @classmethod
    def get_model_packs(cls):
        """Return curated Flux model packs for one-click download."""
        from nodetool.types.model import ModelPack, UnifiedModel

        FLUX_SCHNELL_ALLOW_PATTERNS = [
            "*.json",
            "*.txt",
            "scheduler/*",
            "vae/*",
            "text_encoder/*",
            "tokenizer/*",
            "tokenizer_2/*",
        ]
        FLUX_DEV_ALLOW_PATTERNS = FLUX_SCHNELL_ALLOW_PATTERNS

        return [
            ModelPack(
                id="flux_schnell_nunchaku_int4",
                title="Flux Schnell (Nunchaku INT4)",
                description="Fast 4-step Flux with INT4 quantization via Nunchaku. Requires base Schnell repo + quantized transformer + T5 encoder.",
                category="image_generation",
                tags=["flux", "text-to-image", "int4", "nunchaku", "fast", "4-step"],
                models=[
                    UnifiedModel(
                        id="black-forest-labs/FLUX.1-schnell",
                        type="hf.flux",
                        name="Flux Schnell Base (configs/VAE/tokenizer)",
                        repo_id="black-forest-labs/FLUX.1-schnell",
                        allow_patterns=FLUX_SCHNELL_ALLOW_PATTERNS,
                    ),
                    UnifiedModel(
                        # Composite ID (repo_id:filename) used to distinguish this specific file variant
                        # from other files in the same repo, ensuring unique identification in State.
                        id="nunchaku-tech/nunchaku-flux.1-schnell:svdq-int4_r32-flux.1-schnell.safetensors",
                        type="hf.flux",
                        name="Nunchaku Schnell Transformer (INT4)",
                        repo_id="nunchaku-tech/nunchaku-flux.1-schnell",
                        path="svdq-int4_r32-flux.1-schnell.safetensors",
                        size_on_disk=6400000000,
                    ),
                    UnifiedModel(
                        # Composite ID for T5 encoder variant
                        id="nunchaku-tech/nunchaku-t5:awq-int4-flux.1-t5xxl.safetensors",
                        type="hf.t5",
                        name="Nunchaku T5-XXL Encoder (INT4)",
                        repo_id="nunchaku-tech/nunchaku-t5",
                        path="awq-int4-flux.1-t5xxl.safetensors",
                        size_on_disk=5000000000,
                    ),
                ],
                total_size=11400000000,
            ),
            ModelPack(
                id="flux_dev_nunchaku_int4",
                title="Flux Dev (Nunchaku INT4)",
                description="High-quality Flux Dev with INT4 quantization via Nunchaku. Requires base Dev repo + quantized transformer + T5 encoder.",
                category="image_generation",
                tags=["flux", "text-to-image", "int4", "nunchaku", "high-quality"],
                models=[
                    UnifiedModel(
                        id="black-forest-labs/FLUX.1-dev",
                        type="hf.flux",
                        name="Flux Dev Base (configs/VAE/tokenizer)",
                        repo_id="black-forest-labs/FLUX.1-dev",
                        allow_patterns=FLUX_DEV_ALLOW_PATTERNS,
                    ),
                    UnifiedModel(
                        id="nunchaku-tech/nunchaku-flux.1-dev:svdq-int4_r32-flux.1-dev.safetensors",
                        type="hf.flux",
                        name="Nunchaku Dev Transformer (INT4)",
                        repo_id="nunchaku-tech/nunchaku-flux.1-dev",
                        path="svdq-int4_r32-flux.1-dev.safetensors",
                        size_on_disk=6400000000,
                    ),
                    UnifiedModel(
                        id="nunchaku-tech/nunchaku-t5:awq-int4-flux.1-t5xxl.safetensors",
                        type="hf.t5",
                        name="Nunchaku T5-XXL Encoder (INT4)",
                        repo_id="nunchaku-tech/nunchaku-t5",
                        path="awq-int4-flux.1-t5xxl.safetensors",
                        size_on_disk=5000000000,
                    ),
                ],
                total_size=11400000000,
            ),
        ]

    def _resolve_model_config(self) -> tuple[HFFlux, HFT5]:
        """
        Resolve flux and t5 models based on variant and quantization.
        Returns: (flux_model, t5_model)
        """
        if self.quantization == FluxQuantization.FP4:
            if self.variant == FluxVariant.SCHNELL:
                return (
                    HFFlux(
                        repo_id="nunchaku-tech/nunchaku-flux.1-schnell",
                        path="svdq-fp4_r32-flux.1-schnell.safetensors",
                    ),
                    HFT5(
                        repo_id="nunchaku-tech/nunchaku-t5",
                        path="awq-int4-flux.1-t5xxl.safetensors",
                    ),
                )
            else:
                return (
                    HFFlux(
                        repo_id="nunchaku-tech/nunchaku-flux.1-dev",
                        path="svdq-fp4_r32-flux.1-dev.safetensors",
                    ),
                    HFT5(
                        repo_id="nunchaku-tech/nunchaku-t5",
                        path="awq-int4-flux.1-t5xxl.safetensors",
                    ),
                )
        elif self.quantization == FluxQuantization.INT4:
            if self.variant == FluxVariant.SCHNELL:
                return (
                    HFFlux(
                        repo_id="nunchaku-tech/nunchaku-flux.1-schnell",
                        path="svdq-int4_r32-flux.1-schnell.safetensors",
                    ),
                    HFT5(
                        repo_id="nunchaku-tech/nunchaku-t5",
                        path="awq-int4-flux.1-t5xxl.safetensors",
                    ),
                )
            else:
                return (
                    HFFlux(
                        repo_id="nunchaku-tech/nunchaku-flux.1-dev",
                        path="svdq-int4_r32-flux.1-dev.safetensors",
                    ),
                    HFT5(
                        repo_id="nunchaku-tech/nunchaku-t5",
                        path="awq-int4-flux.1-t5xxl.safetensors",
                    ),
                )
        else:
            # FP16
            base_model = self._get_base_model(self.variant)
            return base_model, None

    def get_model_id(self) -> str:
        flux_model, _ = self._resolve_model_config()
        return flux_model.repo_id

    async def preload_model(self, context: ProcessingContext):
        # Diffusers' lazy Flux import chain hangs on Windows when triggered
        # from a worker thread (asyncio.to_thread). The stdio worker warm-imports
        # it on the main thread at startup, so this is just a cache lookup.
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

        transformer_model, text_encoder_model = self._resolve_model_config()

        torch_dtype = (
            torch.bfloat16
            if self.variant in [FluxVariant.SCHNELL, FluxVariant.DEV]
            else torch.float16
        )

        log.info(f"Using torch_dtype: {torch_dtype}")

        if (
            self.quantization == FluxQuantization.INT4
            or self.quantization == FluxQuantization.FP4
        ):
            assert transformer_model is not None
            assert text_encoder_model is not None

            # Ensure models are present in cache
            if not await HF_FAST_CACHE.resolve(
                transformer_model.repo_id, transformer_model.path
            ):
                raise ValueError(
                    f"Transformer model {transformer_model.repo_id}/{transformer_model.path} must be downloaded"
                )

            if not await HF_FAST_CACHE.resolve(
                text_encoder_model.repo_id, text_encoder_model.path
            ):
                raise ValueError(
                    f"Text encoder model {text_encoder_model.repo_id}/{text_encoder_model.path} must be downloaded"
                )

            base_model = self._get_base_model(self.variant)
            if not await HF_FAST_CACHE.resolve(base_model.repo_id, "model_index.json"):
                raise ValueError(
                    f"Base Flux model {base_model.repo_id} must be downloaded "
                    "(configs, VAE, CLIP tokenizer). Use the Flux Schnell (Nunchaku INT4) model pack."
                )

            from nodetool.huggingface.nunchaku_pipelines import (
                load_nunchaku_flux_pipeline,
            )

            cache_key = (
                f"{base_model.repo_id}:{self.variant.value}:{self.quantization.value}"
            )

            self._pipeline = await load_nunchaku_flux_pipeline(
                context=context,
                repo_id=transformer_model.repo_id,
                transformer_path=transformer_model.path,
                node_id=self.id,
                cache_key=cache_key,
            )

        else:
            # Standard loading (FP16)
            # transformer_model contains the base model for FP16 case
            repo_id = transformer_model.repo_id
            # Ensure model is present in cache
            if not await HF_FAST_CACHE.resolve(repo_id, "model_index.json"):
                raise ValueError(f"Model {repo_id} must be downloaded")

            hf_token = await context.get_secret("HF_TOKEN")
            log.info(f"Loading FLUX pipeline from {repo_id}...")
            self._pipeline = await self.load_model(
                context=context,
                model_id=repo_id,
                model_class=FluxPipeline,
                torch_dtype=torch_dtype,
                variant=None,
                device="cpu",
                token=hf_token,
            )

            _enable_pytorch2_attention(self._pipeline)
            _apply_vae_optimizations(self._pipeline)

        # Apply CPU offload if enabled and not already configured
        if self._pipeline is not None and self.enable_cpu_offload:
            from nodetool.huggingface.memory_utils import apply_cpu_offload_if_needed

            apply_cpu_offload_if_needed(self._pipeline, method="sequential")

    async def move_to_device(self, device: str):
        if self._pipeline is None:
            return

        pipeline_ref = self._pipeline

        # Pipeline.to() and CPU offload setup move multi-GB of weights — pure
        # CUDA/native work that blocks the asyncio event loop and starves the
        # stdio worker's stdin reader. Always run in a thread.
        def _do_move() -> None:
            if self.enable_cpu_offload:
                if device == "cpu":
                    pipeline_ref.to(device)
                elif device in ("cuda", "mps"):
                    from nodetool.huggingface.memory_utils import (
                        apply_cpu_offload_if_needed,
                    )

                    apply_cpu_offload_if_needed(pipeline_ref, method="sequential")
            else:
                pipeline_ref.to(device)
            _apply_vae_optimizations(pipeline_ref)

        try:
            await asyncio.to_thread(_do_move)
        except torch.OutOfMemoryError as e:
            raise ValueError(
                "VRAM out of memory while moving Flux to device. "
                "Try enabling 'CPU offload' in the advanced node properties, "
                "reduce image size, or lower steps."
            ) from e

    async def process(self, context: ProcessingContext) -> ImageRef:
        import threading

        import torch

        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        log_memory(
            f"Flux.process START (variant={self.variant.value}, {self.width}x{self.height})"
        )

        guidance_scale = self.guidance_scale
        num_inference_steps = self.num_inference_steps
        max_sequence_length = self.max_sequence_length

        if self.variant == FluxVariant.SCHNELL:
            guidance_scale = 0.0
            max_sequence_length = 256
            num_inference_steps = 4

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            gen_device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(self.seed)

        # Cancellation event for signalling the inference thread to stop.
        # Checked in the progress callback and used to set pipeline._interrupt.
        cancel_event = threading.Event()

        pipeline_ref = self._pipeline
        import time as _time

        step_timer = {"last": _time.perf_counter(), "started": False}

        def progress_callback(
            pipeline: Any, step: int, timestep: int, callback_kwargs: dict
        ) -> dict:
            # Use diffusers' native _interrupt to skip remaining steps gracefully
            if cancel_event.is_set():
                log.info("Flux inference interrupted via cancel_event")
                pipeline._interrupt = True
                return callback_kwargs
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=num_inference_steps,
                )
            )
            now = _time.perf_counter()
            dt = now - step_timer["last"]
            step_timer["last"] = now
            tag = "warmup" if not step_timer["started"] else "step"
            step_timer["started"] = True
            log.info(f"Flux {tag} {step}/{num_inference_steps} took {dt:.2f}s")
            if step % 5 == 0:
                log_memory(f"Flux inference step {step}/{num_inference_steps}")
            # Free the denoising loop's fragmented segments right before VAE
            # decode. Without this the large decode buffer can't find a
            # contiguous block and spills over PCIe on Windows/WDDM (5s → 10min).
            if step == num_inference_steps - 1:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return callback_kwargs

        # Run GC before inference to free any unused memory
        run_gc("Before Flux inference", log_before_after=True)

        # VAE tiling for >1MP images only — tiling on small images with the
        # Nunchaku-wrapped pipeline triggers a very slow code path on Windows.
        large_image = self.width * self.height > 1024 * 1024
        if large_image and hasattr(self._pipeline, "vae"):
            try:
                self._pipeline.vae.enable_tiling()
            except Exception as e:
                log.warning(f"Failed to enable VAE tiling: {e}")

        log.info(
            f"Starting Flux pipeline: {self.width}x{self.height}, {num_inference_steps} steps"
        )

        # Generate the image off the event loop
        def _run_pipeline_sync():
            with torch.inference_mode():
                t_pipe = _time.perf_counter()
                result = pipeline_ref(
                    prompt=self.prompt,
                    guidance_scale=guidance_scale,
                    height=self.height,
                    width=self.width,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=max_sequence_length,
                    generator=generator,
                    callback_on_step_end=progress_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
                t_vae = _time.perf_counter() - step_timer["last"]
                t_total = _time.perf_counter() - t_pipe
                log.info(
                    f"Flux VAE decoding completed in {t_vae:.2f}s "
                    f"(total pipeline: {t_total:.2f}s)"
                )
                return result

        # Use a per-step timeout: if no progress is made within this time,
        # the inference is considered stuck (e.g. nunchaku hang).
        # Allow generous time per step for slow GPUs.
        per_step_timeout = 120  # seconds per inference step
        total_timeout = (
            per_step_timeout * max(num_inference_steps, 4) + 60
        )  # extra buffer for VAE decode

        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(_run_pipeline_sync),
                timeout=total_timeout,
            )
        except asyncio.TimeoutError:
            cancel_event.set()
            # Also set _interrupt directly on the pipeline for immediate effect
            pipeline_ref._interrupt = True
            run_gc("After Flux timeout")
            raise RuntimeError(
                f"Flux inference timed out after {total_timeout}s. "
                "The pipeline may be stuck. Try restarting the server, "
                "reducing image size, or switching quantization mode."
            )
        except torch.OutOfMemoryError as e:
            cancel_event.set()
            pipeline_ref._interrupt = True
            run_gc("After Flux OOM error")
            raise ValueError(
                "VRAM out of memory while running Flux. "
                "Try enabling 'CPU offload' in the advanced node properties "
                "(Enable CPU offload), reduce image size, or lower steps."
            ) from e
        except asyncio.CancelledError:
            # Signal the inference thread to abort at the next step via
            # both the cancel_event and the diffusers native _interrupt flag.
            cancel_event.set()
            pipeline_ref._interrupt = True
            raise

        log_memory("Flux inference completed")

        image = output.images[0]

        # Run GC after inference to clean up intermediate tensors
        run_gc("After Flux inference", log_before_after=True)

        return await context.image_from_pil(image)


class Chroma(HuggingFacePipelineNode):
    """
    Generates high-quality images from text prompts using Chroma, a Flux-based architecture with enhanced color control.
    image, generation, AI, text-to-image, flux, chroma, transformer, artistic

    Use cases:
    - Generate professional-quality images with precise color control
    - Create artistic images with advanced attention mechanisms
    - Produce images with optimized memory usage via CPU offload
    - Build creative applications requiring high-fidelity output
    """

    prompt: str = Field(
        default="A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done.",
        description="Detailed text description of the image to generate.",
    )
    negative_prompt: str = Field(
        default="low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors",
        description="Describe what to avoid (e.g., 'blurry, low quality, distorted').",
    )
    guidance_scale: float = Field(
        default=3.0,
        description="Prompt adherence strength. 2-5 is typical for Chroma.",
        ge=0.0,
        le=30.0,
    )
    num_inference_steps: int = Field(
        default=40,
        description="Denoising steps. 30-50 is typical; more = better quality but slower.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=1024,
        description="Output image height in pixels.",
        ge=256,
        le=2048,
    )
    width: int = Field(
        default=1024,
        description="Output image width in pixels.",
        ge=256,
        le=2048,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum prompt length in tokens.",
        ge=1,
        le=512,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )
    enable_attention_slicing: bool = Field(
        default=True,
        description="Process attention in slices to reduce memory usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HFTextToImage]:
        return [
            HFTextToImage(
                repo_id="lodestones/Chroma",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Chroma"

    def get_model_id(self) -> str:
        return "lodestones/Chroma"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.chroma.pipeline_chroma import ChromaPipeline

        torch_dtype = available_torch_dtype()
        # Load the pipeline with reduced precision when available
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.get_model_id(),
            model_class=ChromaPipeline,
            torch_dtype=torch_dtype,
        )
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # Handle CPU offload case
            if self.enable_cpu_offload:
                # When moving to CPU, disable CPU offload and move all components to CPU
                if device == "cpu":
                    try:
                        self._pipeline.to(device)
                    except torch.OutOfMemoryError as e:
                        raise ValueError(
                            "VRAM out of memory while moving Chroma pipeline to device. "
                            "Enable 'CPU offload' in the advanced node properties or reduce image size/steps."
                        ) from e
                # When moving to GPU with CPU offload, re-enable CPU offload
                elif device in ["cuda", "mps"]:
                    self._pipeline.enable_model_cpu_offload()
            else:
                # Normal device movement without CPU offload
                try:
                    self._pipeline.to(device)
                except torch.OutOfMemoryError as e:
                    raise ValueError(
                        "VRAM out of memory while moving Chroma pipeline to device. "
                        "Try enabling 'CPU offload' in the advanced node properties, reduce image size, or lower steps."
                    ) from e

            if self.enable_attention_slicing and device != "cpu":
                self._pipeline.enable_attention_slicing()

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # Generate the image
        pipeline_kwargs = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "height": self.height,
            "width": self.width,
            "generator": generator,
            "max_sequence_length": self.max_sequence_length,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
            "callback_on_step_end_tensor_inputs": ["latents"],
        }

        # Generate the image off the event loop
        pipeline = self._pipeline
        assert pipeline is not None

        def _run_pipeline_sync():
            with torch.inference_mode():
                return pipeline(**pipeline_kwargs)

        output = await asyncio.to_thread(_run_pipeline_sync)

        image = output.images[0]

        return await context.image_from_pil(image)

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "prompt",
            "height",
            "width",
            "seed",
        ]


# Example structured JSON prompt for FIBO. FIBO is trained exclusively on
# structured JSON captions and will not produce good results with freeform text.
# Use the FIBO-VLM-prompt-to-JSON or FIBO-gemini-prompt-to-JSON helper models to
# convert a freeform description into this structured format.
_FIBO_DEFAULT_PROMPT = """{
  "scene_description": "A close-up portrait of a golden retriever sitting in a sunlit meadow.",
  "subjects": [
    {
      "type": "animal",
      "description": "a happy golden retriever with a glossy coat, tongue out",
      "position": "center"
    }
  ],
  "lighting": {
    "type": "natural",
    "direction": "backlit",
    "time_of_day": "golden hour",
    "mood": "warm and soft"
  },
  "composition": {
    "framing": "close-up",
    "aspect_ratio": "1:1",
    "focal_point": "the dog's face"
  },
  "color": {
    "palette": "warm golden tones with soft green background",
    "saturation": "medium-high"
  },
  "camera": {
    "lens": "85mm",
    "aperture": "f/1.8",
    "depth_of_field": "shallow"
  },
  "style": "photorealistic, professional photography"
}"""


class BriaFibo(HuggingFacePipelineNode):
    """
    Generates images from structured JSON prompts using Bria's FIBO model for precise visual control.
    image, generation, AI, text-to-image, bria, fibo, json, structured, control

    FIBO is trained exclusively on structured JSON captions (up to 1,000+ words)
    and is designed to understand and control visual parameters such as lighting,
    composition, color, and camera settings, enabling precise and reproducible
    outputs. With only 8 billion parameters it provides high image quality, strong
    prompt adherence and professional control.

    Note: FIBO will NOT work well with freeform text prompts. Provide a structured
    JSON prompt. Use the FIBO-VLM-prompt-to-JSON or FIBO-gemini-prompt-to-JSON
    helper models to convert a freeform description into structured JSON.

    The model is gated: accept the license at
    https://huggingface.co/briaai/FIBO and authenticate with your HF token.

    Use cases:
    - Generate images with precise, reproducible control over lighting and camera
    - Build pipelines that programmatically construct structured JSON prompts
    - Produce professional-quality images with fine-grained visual parameters
    """

    prompt: str = Field(
        default=_FIBO_DEFAULT_PROMPT,
        description=(
            "Structured JSON prompt describing the image. FIBO is trained on "
            "structured JSON captions and does not work well with freeform text."
        ),
    )
    negative_prompt: str = Field(
        default="",
        description="Describe what to avoid in the image (e.g., 'blurry, low quality').",
    )
    guidance_scale: float = Field(
        default=5.0,
        description="Prompt adherence strength. ~5 is typical for FIBO.",
        ge=0.0,
        le=30.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. 30-50 is typical; more = better quality but slower.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=1024,
        description="Output image height in pixels. 1024 gives the best results.",
        ge=256,
        le=2048,
    )
    width: int = Field(
        default=1024,
        description="Output image width in pixels. 1024 gives the best results.",
        ge=256,
        le=2048,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    max_sequence_length: int = Field(
        default=3000,
        description="Maximum prompt length in tokens. FIBO supports long structured prompts.",
        ge=1,
        le=4096,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HFTextToImage]:
        return [
            HFTextToImage(
                repo_id="briaai/FIBO",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "**/*.py",
                    "*.json",
                ],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Bria FIBO"

    def get_model_id(self) -> str:
        return "briaai/FIBO"

    async def preload_model(self, context: ProcessingContext):
        from diffusers import BriaFiboPipeline  # type: ignore

        torch_dtype = available_torch_dtype()
        # FIBO requires trust_remote_code as it ships a custom pipeline.
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.get_model_id(),
            model_class=BriaFiboPipeline,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

    async def move_to_device(self, device: str):
        if self._pipeline is None:
            return

        if self.enable_cpu_offload:
            # When moving to CPU, disable offload by moving everything to CPU.
            if device == "cpu":
                try:
                    self._pipeline.to(device)
                except torch.OutOfMemoryError as e:
                    raise ValueError(
                        "VRAM out of memory while moving FIBO pipeline to device. "
                        "Reduce image size/steps."
                    ) from e
            elif device in ["cuda", "mps"]:
                self._pipeline.enable_model_cpu_offload()
        else:
            try:
                self._pipeline.to(device)
            except torch.OutOfMemoryError as e:
                raise ValueError(
                    "VRAM out of memory while moving FIBO pipeline to device. "
                    "Try enabling 'CPU offload' in the advanced node properties, "
                    "reduce image size, or lower steps."
                ) from e

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        pipeline_kwargs = {
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "height": self.height,
            "width": self.width,
            "generator": generator,
            "max_sequence_length": self.max_sequence_length,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
            "callback_on_step_end_tensor_inputs": ["latents"],
        }
        # FIBO ignores negative prompts when guidance is disabled; only pass it
        # when it is set to avoid unnecessary encoding.
        if self.negative_prompt:
            pipeline_kwargs["negative_prompt"] = self.negative_prompt

        pipeline = self._pipeline
        assert pipeline is not None

        def _run_pipeline_sync():
            with torch.inference_mode():
                return pipeline(**pipeline_kwargs)

        output = await asyncio.to_thread(_run_pipeline_sync)

        image = output.images[0]

        return await context.image_from_pil(image)

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "prompt",
            "height",
            "width",
            "seed",
        ]


class QwenQuantization(str, Enum):
    FP16 = "fp16"
    FP4 = "fp4"
    INT4 = "int4"


class QwenTextEncoderQuantization(str, Enum):
    NF4 = "nf4"
    NF8 = "nf8"
    FP16 = "fp16"


class QwenImage(HuggingFacePipelineNode):
    """
    Generates images from text prompts using Alibaba's Qwen-Image model with Nunchaku quantization support.
    image, generation, AI, text-to-image, qwen, quantization, multilingual

    Use cases:
    - Generate high-quality images with strong multilingual prompt support
    - Memory-efficient generation using INT4/FP4 quantization
    - Create images with precise semantic understanding
    - Build production image generation systems
    """

    quantization: QwenQuantization = Field(
        default=QwenQuantization.INT4,
        description="Quantization level: INT4/FP4 for lower VRAM, FP16 for full precision.",
    )
    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="Text description of the image to generate.",
    )
    negative_prompt: str = Field(
        default="",
        description="Describe what to avoid in the image (e.g., 'blurry, low quality').",
    )
    true_cfg_scale: float = Field(
        default=1.0,
        description="True CFG scale for enhanced prompt following.",
        ge=0.0,
        le=10.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. 30-50 is typical.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=1024,
        description="Output image height in pixels.",
        ge=256,
        le=2048,
    )
    width: int = Field(
        default=1024,
        description="Output image width in pixels.",
        ge=256,
        le=2048,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )

    _pipeline: Any | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFQwenImage]:
        allow_patterns = [
            "*.json",
            "*.txt",
            "scheduler/*",
            "vae/*",
            "text_encoder/*",
            "text_encoder_2/*",
            "tokenizer/*",
            "tokenizer_2/*",
        ]
        return [
            HFQwenImage(
                repo_id="Qwen/Qwen-Image",
                allow_patterns=allow_patterns,
            ),
            HFQwenImage(
                repo_id="nunchaku-tech/nunchaku-qwen-image",
                path="svdq-int4_r32-qwen-image.safetensors",
            ),
            HFQwenImage(
                repo_id="nunchaku-tech/nunchaku-qwen-image",
                path="svdq-fp4_r32-qwen-image.safetensors",
            ),
        ]

    @classmethod
    def get_model_packs(cls):
        """Return curated Qwen-Image model packs for one-click download."""
        from nodetool.types.model import ModelPack, UnifiedModel

        QWEN_IMAGE_ALLOW_PATTERNS = [
            "*.json",
            "*.txt",
            "scheduler/*",
            "vae/*",
            "text_encoder/*",
            "text_encoder_2/*",
            "tokenizer/*",
            "tokenizer_2/*",
        ]

        return [
            ModelPack(
                id="qwen_image_nunchaku_int4",
                title="Qwen-Image (Nunchaku INT4)",
                description="Qwen-Image with INT4 quantization via Nunchaku. Requires base Qwen-Image repo + quantized transformer.",
                category="image_generation",
                tags=["qwen", "text-to-image", "int4", "nunchaku"],
                models=[
                    UnifiedModel(
                        id="Qwen/Qwen-Image",
                        type="hf.qwen_image",
                        name="Qwen-Image Base (configs/VAE/tokenizer/text_encoder)",
                        repo_id="Qwen/Qwen-Image",
                        allow_patterns=QWEN_IMAGE_ALLOW_PATTERNS,
                    ),
                    UnifiedModel(
                        id="nunchaku-tech/nunchaku-qwen-image:svdq-int4_r32-qwen-image.safetensors",
                        type="hf.qwen_image",
                        name="Nunchaku Qwen Transformer (INT4)",
                        repo_id="nunchaku-tech/nunchaku-qwen-image",
                        path="svdq-int4_r32-qwen-image.safetensors",
                        size_on_disk=6500000000,
                    ),
                ],
                total_size=6500000000,
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Qwen-Image"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "quantization",
            "prompt",
            "negative_prompt",
            "height",
            "width",
            "num_inference_steps",
        ]

    def _resolve_model_config(self) -> HFQwenImage:
        if self.quantization == QwenQuantization.FP4:
            return HFQwenImage(
                repo_id="nunchaku-tech/nunchaku-qwen-image",
                path="svdq-fp4_r32-qwen-image.safetensors",
            )
        elif self.quantization == QwenQuantization.INT4:
            return HFQwenImage(
                repo_id="nunchaku-tech/nunchaku-qwen-image",
                path="svdq-int4_r32-qwen-image.safetensors",
            )
        else:
            return HFQwenImage(repo_id="Qwen/Qwen-Image")

    def get_model_id(self) -> str:
        model = self._resolve_model_config()
        return model.repo_id

    def _is_nunchaku_model(self) -> bool:
        """Detect Nunchaku SVDQ transformers via repo or filename."""
        model = self._resolve_model_config()
        repo_has_svdq = model.repo_id and "svdq" in model.repo_id.lower()
        path_has_svdq = model.path and "svdq" in model.path.lower()
        return bool(repo_has_svdq or path_has_svdq)

    async def preload_model(self, context: ProcessingContext):
        if self._is_nunchaku_model():
            await self._load_nunchaku_model(context, available_torch_dtype())
        else:
            await self._load_full_precision_pipeline(context)

    async def _load_full_precision_pipeline(self, context: ProcessingContext):
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

        log.info(
            f"Loading Qwen-Image pipeline from {self.get_model_id()} without quantization..."
        )

        torch_dtype = available_torch_dtype()

        # Ensure model is present in cache
        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(f"Model {self.get_model_id()} must be downloaded")

        self._pipeline = await self.load_model(
            context=context,
            model_class=QwenImagePipeline,
            model_id=self.get_model_id(),
            torch_dtype=torch_dtype,
            device="cpu",  # Load on CPU first, then move to GPU in workflow runner
        )
        assert self._pipeline is not None

        # Apply memory optimizations after loading
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

    async def _load_nunchaku_model(
        self,
        context: ProcessingContext,
        torch_dtype: torch.dtype,
    ):
        """Load Qwen-Image pipeline using a Nunchaku SVDQ transformer file."""
        from nodetool.huggingface.nunchaku_pipelines import (
            load_nunchaku_qwen_pipeline,
        )
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

        model = self._resolve_model_config()

        self._pipeline = await load_nunchaku_qwen_pipeline(
            context=context,
            repo_id=model.repo_id,
            transformer_path=model.path,
            node_id=self.id,
            pipeline_class=QwenImagePipeline,
            base_model_id="Qwen/Qwen-Image",
            torch_dtype=torch_dtype,
        )

    async def move_to_device(self, device: str):
        pass

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device=context.device).manual_seed(self.seed)

        # Generate the image
        pipeline_kwargs = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "true_cfg_scale": self.true_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "height": self.height,
            "width": self.width,
            "generator": generator,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
            "callback_on_step_end_tensor_inputs": ["latents"],
        }

        # Generate the image off the event loop
        def _run_pipeline_sync():
            with torch.inference_mode():
                return self._pipeline(**pipeline_kwargs)  # type: ignore

        output = await asyncio.to_thread(_run_pipeline_sync)

        image = output.images[0]

        return await context.image_from_pil(image)

    def required_inputs(self):
        """Return list of required inputs that must be connected."""
        return []  # No required inputs - IP adapter image is optional


FLUX_CONTROL_BASE_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "scheduler/*",
    "vae/*",
    "text_encoder/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "controlnet/*",
    "transformer/config.json",
]


class FluxControlQuantization(Enum):
    FP16 = "fp16"
    FP4 = "fp4"
    INT4 = "int4"


class FluxControl(HuggingFacePipelineNode):
    """
    Generates images using FLUX Control models with depth or other control guidance.
    image, generation, AI, text-to-image, flux, control, depth, guidance

    Use cases:
    - Generate images with depth-based control guidance
    - Create images following structural guidance from control images
    - High-quality controlled generation with FLUX models
    - Depth-aware image generation
    """

    model: HFControlNetFlux = Field(
        default=HFControlNetFlux(repo_id="black-forest-labs/FLUX.1-Depth-dev"),
        description="The FLUX Control model to use for controlled image generation.",
    )
    prompt: str = Field(
        default="A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts.",
        description="A text prompt describing the desired image.",
    )
    control_image: ImageRef = Field(
        default=ImageRef(),
        description="The control image to guide the generation process.",
    )
    guidance_scale: float = Field(
        default=10.0,
        description="The scale for classifier-free guidance.",
        ge=0.0,
        le=30.0,
    )
    width: int = Field(
        default=1024, description="The width of the generated image.", ge=64, le=2048
    )
    height: int = Field(
        default=1024, description="The height of the generated image.", ge=64, le=2048
    )
    num_inference_steps: int = Field(
        default=30,
        description="The number of denoising steps.",
        ge=1,
        le=100,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Enable CPU offload to reduce VRAM usage.",
    )
    quantization: FluxControlQuantization = Field(
        default=FluxControlQuantization.INT4,
        description="Quantization level for the FLUX Control transformer.",
    )
    _pipeline: Any = None

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "quantization",
            "prompt",
            "control_image",
            "height",
            "width",
            "guidance_scale",
            "seed",
        ]

    @classmethod
    def get_recommended_models(cls) -> list[HFControlNetFlux | HFT5]:
        allow_patterns = [
            "*.json",
            "*.txt",
            "scheduler/*",
            "vae/*",
            "text_encoder/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "transformer/config.json",
        ]
        return [
            HFControlNetFlux(
                repo_id="black-forest-labs/FLUX.1-Depth-dev",
                allow_patterns=allow_patterns,
            ),
            HFControlNetFlux(
                repo_id="black-forest-labs/FLUX.1-Canny-dev",
                allow_patterns=allow_patterns,
            ),
            HFControlNetFlux(
                repo_id="nunchaku-tech/nunchaku-flux.1-depth-dev",
                path="svdq-int4_r32-flux.1-depth-dev.safetensors",
            ),
            HFControlNetFlux(
                repo_id="nunchaku-tech/nunchaku-flux.1-depth-dev",
                path="svdq-fp4_r32-flux.1-depth-dev.safetensors",
            ),
            HFControlNetFlux(
                repo_id="nunchaku-tech/nunchaku-flux.1-canny-dev",
                path="svdq-int4_r32-flux.1-canny-dev.safetensors",
            ),
            HFControlNetFlux(
                repo_id="nunchaku-tech/nunchaku-flux.1-canny-dev",
                path="svdq-fp4_r32-flux.1-canny-dev.safetensors",
            ),
            HFT5(
                repo_id="nunchaku-tech/nunchaku-t5",
                path="awq-int4-flux.1-t5xxl.safetensors",
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Flux Control"

    def get_model_id(self) -> str:
        quantization = self._resolve_effective_quantization()
        base_model, _, _ = self._resolve_model_config(quantization)
        return base_model.repo_id or "black-forest-labs/FLUX.1-Depth-dev"

    def required_inputs(self):
        return ["control_image"]

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.flux.pipeline_flux_control import FluxControlPipeline

        hf_token = await context.get_secret("HF_TOKEN")
        if not hf_token:
            model_url = f"https://huggingface.co/{self.get_model_id()}"
            raise ValueError(
                f"Flux Control is a gated model, please set the HF_TOKEN in Nodetool settings and accept the terms of use for the model: {model_url}"
            )

        torch_dtype = torch.bfloat16
        quantization = self._resolve_effective_quantization()
        base_model, transformer_model, text_encoder_model = self._resolve_model_config(
            quantization
        )

        log.info(
            "Loading FLUX Control pipeline from %s (quantization=%s)",
            base_model.repo_id,
            quantization.value,
        )
        if transformer_model is not None and text_encoder_model is not None:
            if not await HF_FAST_CACHE.resolve(base_model.repo_id, "model_index.json"):
                raise ValueError(
                    f"Base Flux Control model {base_model.repo_id} must be downloaded"
                )

            if not await HF_FAST_CACHE.resolve(
                transformer_model.repo_id, transformer_model.path
            ):
                raise ValueError(
                    f"Transformer model {transformer_model.repo_id}/{transformer_model.path} must be downloaded"
                )

            if not await HF_FAST_CACHE.resolve(
                text_encoder_model.repo_id, text_encoder_model.path
            ):
                raise ValueError(
                    f"Text encoder model {text_encoder_model.repo_id}/{text_encoder_model.path} must be downloaded"
                )

            from nodetool.huggingface.nunchaku_pipelines import (
                load_nunchaku_flux_pipeline,
            )
            from nodetool.ml.core.model_manager import ModelManager

            # Cache key for controlnet variant
            cache_key = f"{base_model.repo_id}:{quantization.value}:control-v1"

            self._pipeline = await load_nunchaku_flux_pipeline(
                context=context,
                repo_id=transformer_model.repo_id,
                transformer_path=transformer_model.path,
                node_id=self.id,
                pipeline_class=FluxControlPipeline,
                cache_key=cache_key,
            )
        else:
            if not await HF_FAST_CACHE.resolve(base_model.repo_id, "model_index.json"):
                raise ValueError(f"Model {base_model.repo_id} must be downloaded")

            self._pipeline = await self.load_model(
                context=context,
                model_class=FluxControlPipeline,
                model_id=base_model.repo_id,
                path=base_model.path,
                torch_dtype=torch_dtype,
                device="cpu",
                token=hf_token,
                local_files_only=True,
            )

        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        if self._pipeline is not None and self.enable_cpu_offload:
            from nodetool.huggingface.memory_utils import apply_cpu_offload_if_needed

            apply_cpu_offload_if_needed(self._pipeline, method="sequential")

    def _is_nunchaku_model(self) -> bool:
        return self._resolve_effective_quantization() != FluxControlQuantization.FP16

    def _detect_legacy_quantization(self) -> FluxControlQuantization | None:
        repo = (self.model.repo_id or "").lower()
        path = (self.model.path or "").lower()
        if "svdq" not in repo and "svdq" not in path:
            return None
        if "fp4" in repo or "fp4" in path:
            return FluxControlQuantization.FP4
        return FluxControlQuantization.INT4

    def _resolve_effective_quantization(self) -> FluxControlQuantization:
        quantization = self.quantization
        legacy_quantization = self._detect_legacy_quantization()
        if (
            quantization == FluxControlQuantization.FP16
            and legacy_quantization is not None
        ):
            quantization = legacy_quantization

        return quantization

    def _detect_flux_control_variant(self) -> tuple[str, str]:
        repo_id = (self.model.repo_id or "").lower()
        path = (self.model.path or "").lower()
        if "canny" in repo_id or "canny" in path:
            return "canny-dev", "black-forest-labs/FLUX.1-Canny-dev"
        return "depth-dev", "black-forest-labs/FLUX.1-Depth-dev"

    def _resolve_model_config(
        self, quantization: FluxControlQuantization
    ) -> tuple[HFControlNetFlux, HFControlNetFlux | None, HFT5 | None]:
        variant_key, base_model_id = self._detect_flux_control_variant()
        if quantization == FluxControlQuantization.FP16:
            if self.model.repo_id:
                return self.model, None, None
            return HFControlNetFlux(repo_id=base_model_id), None, None

        precision = "fp4" if quantization == FluxControlQuantization.FP4 else "int4"
        transformer_model = HFControlNetFlux(
            repo_id=f"nunchaku-tech/nunchaku-flux.1-{variant_key}",
            path=f"svdq-{precision}_r32-flux.1-{variant_key}.safetensors",
        )
        text_encoder_model = HFT5(
            repo_id="nunchaku-tech/nunchaku-t5",
            path="awq-int4-flux.1-t5xxl.safetensors",
        )
        base_model = HFControlNetFlux(
            repo_id=base_model_id,
            allow_patterns=FLUX_CONTROL_BASE_ALLOW_PATTERNS,
        )
        return base_model, transformer_model, text_encoder_model

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # If CPU offload is enabled, we need to handle device movement differently
            if self.enable_cpu_offload:
                # When moving to CPU, disable CPU offload and move all components to CPU
                if device == "cpu":
                    try:
                        self._pipeline.to(device)
                    except torch.OutOfMemoryError as e:
                        raise ValueError(
                            "VRAM out of memory while moving Flux Control to device. "
                            "Enable 'CPU offload' in the advanced node properties or reduce image size."
                        ) from e
                # When moving to GPU with CPU offload, re-enable CPU offload
                elif device in ["cuda", "mps"]:
                    from nodetool.huggingface.memory_utils import (
                        apply_cpu_offload_if_needed,
                    )

                    apply_cpu_offload_if_needed(self._pipeline, method="sequential")
            else:
                # Normal device movement without CPU offload
                try:
                    self._pipeline.to(device)
                except torch.OutOfMemoryError as e:
                    raise ValueError(
                        "VRAM out of memory while moving Flux Control to device. "
                        "Enable 'CPU offload' in the advanced node properties or reduce image size."
                    ) from e

            _apply_vae_optimizations(self._pipeline)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        log_memory(f"FluxControl.process START ({self.width}x{self.height})")

        # Set up the generator for reproducibility
        generator = torch.Generator(device=context.device)
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        # Load and preprocess control image
        control_image = await context.image_to_pil(self.control_image)

        # Run GC before inference to free any unused memory
        run_gc("Before FluxControl inference", log_before_after=True)

        # Prepare kwargs for the pipeline
        kwargs = {
            "prompt": self.prompt,
            "control_image": control_image,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "generator": generator,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
        }

        try:
            output = await self.run_pipeline_in_thread(**kwargs)
        except torch.OutOfMemoryError as e:
            run_gc("After FluxControl OOM error")
            raise ValueError(
                "VRAM out of memory while running Flux Control. "
                "Enable 'CPU offload' in the advanced node properties (if available), "
                "or reduce image size/steps."
            ) from e

        log_memory("FluxControl inference completed")

        image = output.images[0]

        # Run GC after inference to clean up intermediate tensors
        run_gc("After FluxControl inference", log_before_after=True)

        return await context.image_from_pil(image)


# Broad allow-patterns for multi-folder diffusers repos (configs + weights).
_DIFFUSERS_REPO_ALLOW_PATTERNS = [
    "**/*.safetensors",
    "**/*.json",
    "**/*.txt",
    "**/*.model",
    "*.json",
]


class Flux2(HuggingFacePipelineNode):
    """
    Generates images from text prompts using Black Forest Labs' FLUX.2 model.
    image, generation, AI, text-to-image, flux, flux2

    Use cases:
    - Generate high-fidelity images from detailed text prompts
    - Produce photorealistic and artistic imagery with the FLUX.2 architecture
    - Create assets for design, marketing, and creative work
    - Build production text-to-image systems
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(repo_id="black-forest-labs/FLUX.2-dev"),
        description="The FLUX.2 model to use for image generation.",
    )
    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="Text description of the image to generate.",
    )
    width: int = Field(
        default=1024, description="Output image width in pixels.", ge=256, le=2048
    )
    height: int = Field(
        default=1024, description="Output image height in pixels.", ge=256, le=2048
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. More steps = higher quality, slower generation.",
        ge=1,
        le=100,
    )
    guidance_scale: float = Field(
        default=4.0,
        description="How strongly to follow the prompt. 2.5-4.0 is typical.",
        ge=0.0,
        le=20.0,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToImage(
                repo_id="black-forest-labs/FLUX.2-dev",
                allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "FLUX.2"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "width", "height", "num_inference_steps"]

    def get_model_id(self) -> str:
        return self.model.repo_id or "black-forest-labs/FLUX.2-dev"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(
                f"Model {self.get_model_id()} must be downloaded first from the recommended models"
            )

        self._pipeline = await self.load_model(
            context=context,
            model_class=Flux2Pipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
            local_files_only=True,
        )
        maybe_enable_cpu_offload(self._pipeline, self.enable_cpu_offload)

    async def move_to_device(self, device: str):
        # On MPS we skip offload and load fully onto the device, so move here.
        if self._pipeline is not None and (
            not self.enable_cpu_offload or is_mps_device()
        ):
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            callback_on_step_end=pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
            callback_on_step_end_tensor_inputs=["latents"],
        )
        image = output.images[0]
        run_gc("After FLUX.2 inference", log_before_after=False)
        return await context.image_from_pil(image)


class Flux2Klein(HuggingFacePipelineNode):
    """
    Generates images from text prompts using Black Forest Labs' FLUX.2 Klein model.
    image, generation, AI, text-to-image, flux, flux2, klein

    Use cases:
    - Generate images with the open-weight FLUX.2 Klein variant
    - Fast, interactive text-to-image generation
    - Create artwork and design assets
    - Prototype image generation workflows
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(repo_id="black-forest-labs/FLUX.2-klein-base-9B"),
        description="The FLUX.2 Klein model to use for image generation.",
    )
    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="Text description of the image to generate.",
    )
    width: int = Field(
        default=1024, description="Output image width in pixels.", ge=256, le=2048
    )
    height: int = Field(
        default=1024, description="Output image height in pixels.", ge=256, le=2048
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. More steps = higher quality, slower generation.",
        ge=1,
        le=100,
    )
    guidance_scale: float = Field(
        default=4.0,
        description="How strongly to follow the prompt. Ignored for step-distilled variants.",
        ge=0.0,
        le=20.0,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToImage(
                repo_id="black-forest-labs/FLUX.2-klein-base-9B",
                allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "FLUX.2 Klein"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "width", "height", "num_inference_steps"]

    def get_model_id(self) -> str:
        return self.model.repo_id or "black-forest-labs/FLUX.2-klein-base-9B"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(
                f"Model {self.get_model_id()} must be downloaded first from the recommended models"
            )

        self._pipeline = await self.load_model(
            context=context,
            model_class=Flux2KleinPipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
            local_files_only=True,
        )
        maybe_enable_cpu_offload(self._pipeline, self.enable_cpu_offload)

    async def move_to_device(self, device: str):
        # On MPS we skip offload and load fully onto the device, so move here.
        if self._pipeline is not None and (
            not self.enable_cpu_offload or is_mps_device()
        ):
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            callback_on_step_end=pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
            callback_on_step_end_tensor_inputs=["latents"],
        )
        image = output.images[0]
        run_gc("After FLUX.2 Klein inference", log_before_after=False)
        return await context.image_from_pil(image)


class GlmImage(HuggingFacePipelineNode):
    """
    Generates images from text prompts using Zhipu AI's GLM-Image model.
    image, generation, AI, text-to-image, glm, text-rendering

    Use cases:
    - Generate knowledge-intensive images with accurate text rendering
    - Produce magazine-style layouts and infographics
    - Create high-detail imagery from descriptive prompts
    - Build multilingual image generation systems
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(repo_id="zai-org/GLM-Image"),
        description="The GLM-Image model to use for image generation.",
    )
    prompt: str = Field(
        default="A photo of an astronaut riding a horse on mars",
        description="Text description of the image to generate.",
    )
    width: int = Field(
        default=1024,
        description="Output image width in pixels. Should be a multiple of 32.",
        ge=256,
        le=2048,
    )
    height: int = Field(
        default=1024,
        description="Output image height in pixels. Should be a multiple of 32.",
        ge=256,
        le=2048,
    )
    num_inference_steps: int = Field(
        default=30,
        description="Denoising steps for the diffusion decoder. 30 is recommended.",
        ge=1,
        le=100,
    )
    guidance_scale: float = Field(
        default=1.5,
        description="How strongly to follow the prompt. 1.5 is recommended.",
        ge=0.0,
        le=20.0,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToImage(
                repo_id="zai-org/GLM-Image",
                allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "GLM-Image"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "width", "height", "num_inference_steps"]

    def get_model_id(self) -> str:
        return self.model.repo_id or "zai-org/GLM-Image"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.glm_image.pipeline_glm_image import GlmImagePipeline

        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(
                f"Model {self.get_model_id()} must be downloaded first from the recommended models"
            )

        self._pipeline = await self.load_model(
            context=context,
            model_class=GlmImagePipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
            local_files_only=True,
        )
        maybe_enable_cpu_offload(self._pipeline, self.enable_cpu_offload)

    async def move_to_device(self, device: str):
        # On MPS we skip offload and load fully onto the device, so move here.
        if self._pipeline is not None and (
            not self.enable_cpu_offload or is_mps_device()
        ):
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            callback_on_step_end=pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
            callback_on_step_end_tensor_inputs=["latents"],
        )
        image = output.images[0]
        run_gc("After GLM-Image inference", log_before_after=False)
        return await context.image_from_pil(image)


class QwenImageLayered(HuggingFacePipelineNode):
    """
    Generates layered images from text prompts using Qwen-Image-Layered.
    image, generation, AI, text-to-image, qwen, layered, transparent

    Use cases:
    - Generate images decomposed into separate layers
    - Produce assets with transparent foregrounds and backgrounds
    - Create compositable graphics for design workflows
    - Build layered content generation systems
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(repo_id="Qwen/Qwen-Image-Layered"),
        description="The Qwen-Image-Layered model to use.",
    )
    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="Text description of the image to generate.",
    )
    negative_prompt: str = Field(
        default="",
        description="Describe what to avoid in the image.",
    )
    layers: int = Field(
        default=4,
        description="Number of layers to decompose the image into.",
        ge=1,
        le=8,
    )
    resolution: int = Field(
        default=640,
        description="Working resolution for layer generation.",
        ge=256,
        le=1536,
    )
    true_cfg_scale: float = Field(
        default=4.0,
        description="True CFG scale for enhanced prompt following.",
        ge=0.0,
        le=10.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. 30-50 is typical.",
        ge=1,
        le=100,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToImage(
                repo_id="Qwen/Qwen-Image-Layered",
                allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Qwen-Image-Layered"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "layers", "resolution", "num_inference_steps"]

    def get_model_id(self) -> str:
        return self.model.repo_id or "Qwen/Qwen-Image-Layered"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_layered import (
            QwenImageLayeredPipeline,
        )

        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(
                f"Model {self.get_model_id()} must be downloaded first from the recommended models"
            )

        self._pipeline = await self.load_model(
            context=context,
            model_class=QwenImageLayeredPipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
            local_files_only=True,
        )
        maybe_enable_cpu_offload(self._pipeline, self.enable_cpu_offload)

    async def move_to_device(self, device: str):
        # On MPS we skip offload and load fully onto the device, so move here.
        if self._pipeline is not None and (
            not self.enable_cpu_offload or is_mps_device()
        ):
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            layers=self.layers,
            resolution=self.resolution,
            true_cfg_scale=self.true_cfg_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            callback_on_step_end=pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
            callback_on_step_end_tensor_inputs=["latents"],
        )
        images = output.images[0]
        # Layered pipelines may return a list of per-layer images; pick the first.
        if isinstance(images, (list, tuple)):
            images = images[0]
        run_gc("After Qwen-Image-Layered inference", log_before_after=False)
        return await context.image_from_pil(images)


class Kandinsky5Image(HuggingFacePipelineNode):
    """
    Generates images from text prompts using Kandinsky 5.0 Image Lite.
    image, generation, AI, text-to-image, kandinsky

    Use cases:
    - Generate high-quality images from text descriptions
    - Produce artwork with strong bilingual (incl. Russian) understanding
    - Create assets for creative and commercial projects
    - Build lightweight image generation systems
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(
            repo_id="kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"
        ),
        description="The Kandinsky 5.0 image model to use.",
    )
    prompt: str = Field(
        default="A cat and a dog baking a cake together in a kitchen.",
        description="Text description of the image to generate.",
    )
    negative_prompt: str = Field(
        default="",
        description="Describe what to avoid in the image.",
    )
    width: int = Field(
        default=1024, description="Output image width in pixels.", ge=256, le=2048
    )
    height: int = Field(
        default=1024, description="Output image height in pixels.", ge=256, le=2048
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. 50 is typical; distilled checkpoints use fewer.",
        ge=1,
        le=100,
    )
    guidance_scale: float = Field(
        default=3.5,
        description="How strongly to follow the prompt. Use 1.0 for distilled checkpoints.",
        ge=0.0,
        le=20.0,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            # SFT: highest generation quality.
            HFTextToImage(
                repo_id="kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers",
                allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
            ),
            # Pretrain: base model for research and fine-tuning.
            HFTextToImage(
                repo_id="kandinskylab/Kandinsky-5.0-T2I-Lite-pretrain-Diffusers",
                allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Kandinsky 5.0 Image"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "width", "height", "num_inference_steps"]

    def get_model_id(self) -> str:
        return self.model.repo_id or "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.kandinsky5.pipeline_kandinsky_t2i import (
            Kandinsky5T2IPipeline,
        )

        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(
                f"Model {self.get_model_id()} must be downloaded first from the recommended models"
            )

        self._pipeline = await self.load_model(
            context=context,
            model_class=Kandinsky5T2IPipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
            local_files_only=True,
        )
        maybe_enable_cpu_offload(self._pipeline, self.enable_cpu_offload)

    async def move_to_device(self, device: str):
        # On MPS we skip offload and load fully onto the device, so move here.
        if self._pipeline is not None and (
            not self.enable_cpu_offload or is_mps_device()
        ):
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            callback_on_step_end=pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
            callback_on_step_end_tensor_inputs=["latents"],
        )
        # Kandinsky output exposes generated images under .images/.image/.frames.
        images = (
            getattr(output, "images", None)
            or getattr(output, "image", None)
            or getattr(output, "frames", None)
        )
        image = images[0]
        if isinstance(image, (list, tuple)):
            image = image[0]
        run_gc("After Kandinsky 5.0 Image inference", log_before_after=False)
        return await context.image_from_pil(image)
