from enum import Enum
from typing import Any
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    HFTextToImage,
    HFImageToImage,
    HFLoraSD,
    HFFlux,
    HFQwenImage,
    ImageRef,
    TorchTensor,
)
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.nodes.huggingface.image_to_image import pipeline_progress_callback
from nodetool.nodes.huggingface.stable_diffusion_base import (
    StableDiffusionBaseNode,
    StableDiffusionXLBase,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, Notification, LogUpdate

import torch
import asyncio
import logging
from transformers.utils import logging as hf_logging
from diffusers.utils import logging as diffusers_logging
from diffusers.pipelines.pag.pipeline_pag_sd import StableDiffusionPAGPipeline
from diffusers.pipelines.pag.pipeline_pag_sd_xl import StableDiffusionXLPAGPipeline
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.kolors.pipeline_kolors import KolorsPipeline
from diffusers.pipelines.chroma.pipeline_chroma import ChromaPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.quantizers.quantization_config import GGUFQuantizationConfig
from pydantic import Field
from nodetool.workflows.base_node import BaseNode

log = get_logger(__name__)


class StableDiffusion(StableDiffusionBaseNode):
    """
    Generates images from text prompts using Stable Diffusion.
    image, generation, AI, text-to-image, SD

    Use cases:
    - Creating custom illustrations for various projects
    - Generating concept art for creative endeavors
    - Producing unique visual content for marketing materials
    - Exploring AI-generated art for personal or professional use
    """

    width: int = Field(
        default=512, ge=256, le=1024, description="Width of the generated image."
    )
    height: int = Field(
        default=512, ge=256, le=1024, description="Height of the generated image"
    )
    _pipeline: StableDiffusionPAGPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["width", "height"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion"

    @classmethod
    def return_type(cls):
        return {
            "image": ImageRef,
            "latent": TorchTensor,
        }

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionPAGPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            config="Lykon/DreamShaper",
            pag_scale=self.pag_scale,
        )
        assert self._pipeline is not None
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext):
        result = await self.run_pipeline(context, width=self.width, height=self.height)
        if self.output_type == self.StableDiffusionOutputType.IMAGE:
            return {
                "image": result,
                "latent": TorchTensor(),
            }
        else:
            return {
                "image": ImageRef(),
                "latent": result,
            }


class StableDiffusionXL(StableDiffusionXLBase):
    """
    Generates images from text prompts using Stable Diffusion XL.
    image, generation, AI, text-to-image, SDXL

    Use cases:
    - Creating custom illustrations for marketing materials
    - Generating concept art for game and film development
    - Producing unique stock imagery for websites and publications
    - Visualizing interior design concepts for clients
    """

    _pipeline: StableDiffusionXLPAGPipeline | None = None

    @classmethod
    def get_title(cls):
        return "Stable Diffusion XL"

    async def preload_model(self, context: ProcessingContext):
        if "playground" in self.model.repo_id:
            raise ValueError("Playground models are not supported in this node")

        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionXLPAGPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            variant="fp16",
        )
        assert self._pipeline is not None
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context) -> ImageRef:
        return await self.run_pipeline(context)


class ModelVariant(Enum):
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
    DEFAULT = "default"


class LoadTextToImageModel(HuggingFacePipelineNode):
    """
    Load HuggingFace model for image-to-image generation from a repo_id.

    Use cases:
    - Loads a pipeline directly from a repo_id
    - Used for AutoPipelineForImage2Image
    """

    repo_id: str = Field(
        default="",
        description="The repository ID of the model to use for image-to-image generation.",
    )

    variant: ModelVariant = Field(
        default=ModelVariant.DEFAULT,
        description="The variant of the model to use for text-to-image generation.",
    )

    async def preload_model(self, context: ProcessingContext):
        await self.load_model(
            context=context,
            model_id=self.repo_id,
            model_class=AutoPipelineForText2Image,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant=(
                self.variant.value if self.variant != ModelVariant.DEFAULT else None
            ),
        )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "repo_id",
        ]

    async def process(self, context: ProcessingContext) -> HFTextToImage:
        return HFTextToImage(
            repo_id=self.repo_id,
            variant=(
                self.variant.value if self.variant != ModelVariant.DEFAULT else None
            ),
        )


class Text2Image(HuggingFacePipelineNode):
    """
    Generates images from text prompts using AutoPipeline for automatic pipeline selection.
    image, generation, AI, text-to-image, auto

    Use cases:
    - Automatic selection of the best pipeline for a given model
    - Flexible image generation without pipeline-specific knowledge
    - Quick prototyping with various text-to-image models
    - Streamlined workflow for different model architectures
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(),
        description="The model to use for text-to-image generation.",
    )

    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="A text prompt describing the desired image.",
    )
    negative_prompt: str = Field(
        default="",
        description="A text prompt describing what to avoid in the image.",
    )
    num_inference_steps: int = Field(
        default=50,
        description="The number of denoising steps.",
        ge=1,
        le=100,
    )
    guidance_scale: float = Field(
        default=7.5,
        description="The scale for classifier-free guidance.",
        ge=1.0,
        le=20.0,
    )
    width: int = Field(
        default=512,
        description="The width of the generated image.",
        ge=64,
        le=2048,
    )
    height: int = Field(
        default=512,
        description="The height of the generated image.",
        ge=64,
        le=2048,
    )
    pag_scale: float = Field(
        default=3.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
        ge=0.0,
        le=10.0,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )

    _pipeline: AutoPipelineForText2Image | None = None

    @classmethod
    def get_title(cls) -> str:
        return "Text to Image"

    def get_model_id(self) -> str:
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.get_model_id(),
            model_class=AutoPipelineForText2Image,
            torch_dtype=torch.float16,
            variant=(
                self.model.variant
                if self.model.variant != ModelVariant.DEFAULT
                else None
            ),
            enable_pag=self.pag_scale > 0.0,
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # Generate the image off the event loop
        def _run_pipeline_sync():
            return self._pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                width=self.width,
                height=self.height,
                generator=generator,
                pag_scale=self.pag_scale,
                callback_on_step_end=pipeline_progress_callback(
                    self.id, self.num_inference_steps, context
                ),
            )  # type: ignore

        output = await asyncio.to_thread(_run_pipeline_sync)

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)

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
    FILL_DEV = "fill-dev"
    CANNY_DEV = "canny-dev"
    DEPTH_DEV = "depth-dev"


class Flux(HuggingFacePipelineNode):
    """
    Generates images using FLUX models with support for GGUF quantization for memory efficiency.
    image, generation, AI, text-to-image, flux, quantization

    Use cases:
    - High-quality image generation with FLUX models
    - Memory-efficient generation using GGUF quantization
    - Fast generation with FLUX.1-schnell
    - High-fidelity generation with FLUX.1-dev
    - Controlled generation with Fill, Canny, or Depth variants
    """

    model: HFFlux = Field(
        default=HFFlux(),
        description="The FLUX model to use for text-to-image generation.",
    )
    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="A text prompt describing the desired image.",
    )
    guidance_scale: float = Field(
        default=3.5,
        description="The scale for classifier-free guidance. Use 0.0 for schnell, 3.5 for dev.",
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
        default=20,
        description="The number of denoising steps. Use 4 for schnell, 20-50 for dev.",
        ge=1,
        le=100,
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum sequence length for the prompt. Use 256 for schnell, 512 for dev.",
        ge=1,
        le=512,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )
    enable_memory_efficient_attention: bool = Field(
        default=True,
        description="Enable memory efficient attention to reduce VRAM usage.",
    )
    enable_vae_tiling: bool = Field(
        default=False,
        description="Enable VAE tiling to reduce VRAM usage for large images.",
    )
    enable_vae_slicing: bool = Field(
        default=False,
        description="Enable VAE slicing to reduce VRAM usage.",
    )
    enable_cpu_offload: bool = Field(
        default=False,
        description="Enable CPU offload to reduce VRAM usage.",
    )

    _pipeline: FluxPipeline | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFFlux]:
        return [
            HFFlux(
                repo_id="black-forest-labs/FLUX.1-schnell",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HFFlux(
                repo_id="black-forest-labs/FLUX.1-dev",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HFFlux(
                repo_id="black-forest-labs/FLUX.1-Fill-dev",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HFFlux(
                repo_id="black-forest-labs/FLUX.1-Canny-dev",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HFFlux(
                repo_id="black-forest-labs/FLUX.1-Depth-dev",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            # GGUF quantized models
            HFFlux(
                repo_id="city96/FLUX.1-dev-gguf",
                path="flux1-dev-Q2_K.gguf",
            ),
            HFFlux(
                repo_id="city96/FLUX.1-dev-gguf",
                path="flux1-dev-Q3_K_S.gguf",
            ),
            HFFlux(
                repo_id="city96/FLUX.1-dev-gguf",
                path="flux1-dev-Q4_K_S.gguf",
            ),
            HFFlux(
                repo_id="city96/FLUX.1-dev-gguf",
                path="flux1-dev-Q5_K_S.gguf",
            ),
            # FLUX.1-schnell GGUF models
            HFFlux(
                repo_id="city96/FLUX.1-schnell-gguf",
                path="flux1-schnell-Q2_K.gguf",
            ),
            HFFlux(
                repo_id="city96/FLUX.1-schnell-gguf",
                path="flux1-schnell-Q3_K_S.gguf",
            ),
            HFFlux(
                repo_id="city96/FLUX.1-schnell-gguf",
                path="flux1-schnell-Q4_K_S.gguf",
            ),
            HFFlux(
                repo_id="city96/FLUX.1-schnell-gguf",
                path="flux1-schnell-Q5_K_S.gguf",
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Flux"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "prompt",
            "height",
            "width",
            "seed",
        ]

    def get_model_id(self) -> str:
        if self.model.repo_id:
            return self.model.repo_id
        # Fallback to detected variant
        detected_variant = self._detect_variant_from_repo_id()
        model_mapping = {
            FluxVariant.SCHNELL: "black-forest-labs/FLUX.1-schnell",
            FluxVariant.DEV: "black-forest-labs/FLUX.1-dev",
            FluxVariant.FILL_DEV: "black-forest-labs/FLUX.1-Fill-dev",
            FluxVariant.CANNY_DEV: "black-forest-labs/FLUX.1-Canny-dev",
            FluxVariant.DEPTH_DEV: "black-forest-labs/FLUX.1-Depth-dev",
        }
        return model_mapping[detected_variant]

    def _detect_variant_from_repo_id(self) -> FluxVariant:
        """Detect FLUX variant from the selected repo_id or path."""
        # Check repo_id first
        if self.model.repo_id:
            repo_id_lower = self.model.repo_id.lower()
            if "schnell" in repo_id_lower:
                return FluxVariant.SCHNELL
            elif "fill" in repo_id_lower:
                return FluxVariant.FILL_DEV
            elif "canny" in repo_id_lower:
                return FluxVariant.CANNY_DEV
            elif "depth" in repo_id_lower:
                return FluxVariant.DEPTH_DEV
            elif "dev" in repo_id_lower:
                return FluxVariant.DEV

        # Check path for GGUF models
        if self.model.path:
            path_lower = self.model.path.lower()
            if "schnell" in path_lower:
                return FluxVariant.SCHNELL
            elif "dev" in path_lower:
                return FluxVariant.DEV

        # Default fallback
        return FluxVariant.DEV

    def _detect_gguf_quantization_type(self) -> str | None:
        """Detect GGUF quantization type from model path."""
        if not self.model.path:
            return None

        # Extract quantization type from filename patterns like "flux1-dev-Q2_K.gguf"
        import re

        pattern = r"-(Q\d+_[A-Z]+|Q\d+)\.gguf$"
        match = re.search(pattern, self.model.path, re.IGNORECASE)
        return match.group(1) if match else None

    def _is_gguf_model(self) -> bool:
        """Check if the model is a GGUF model based on file extension."""
        return self.model.path is not None and self.model.path.lower().endswith(".gguf")

    async def preload_model(self, context: ProcessingContext):
        # Determine torch dtype based on variant
        # Auto-detect variant from model selection
        detected_variant = self._detect_variant_from_repo_id()

        torch_dtype = (
            torch.bfloat16
            if detected_variant in [FluxVariant.SCHNELL, FluxVariant.DEV]
            else torch.float16
        )

        # Check if this is a GGUF model based on file extension
        if self._is_gguf_model():
            await self._load_gguf_model(context, torch_dtype)
        else:
            # Load the full pipeline normally
            log.info(f"Loading FLUX pipeline from {self.get_model_id()}...")
            self._pipeline = await self.load_model(
                context=context,
                model_id=self.get_model_id(),
                path=self.model.path,
                model_class=FluxPipeline,
                torch_dtype=torch_dtype,
                variant=None,
                device="cpu",
            )

        # Apply CPU offload if enabled
        if self._pipeline is not None and self.enable_cpu_offload:
            self._pipeline.enable_sequential_cpu_offload()

    async def _load_gguf_model(self, context: ProcessingContext, torch_dtype):
        """Load FLUX model with GGUF quantization."""
        from huggingface_hub.file_download import try_to_load_from_cache

        quantization_type = self._detect_gguf_quantization_type()

        if quantization_type:
            log.info(
                f"Loading FLUX transformer with GGUF {quantization_type} quantization..."
            )
        else:
            log.info(
                "Loading GGUF model (quantization type not detected from filename)..."
            )

        # Get the cached file path
        assert self.model.path is not None
        cache_path = try_to_load_from_cache(self.get_model_id(), self.model.path)
        if not cache_path:
            raise ValueError(
                f"Model {self.get_model_id()}/{self.model.path} must be downloaded first"
            )

        # Load the transformer with GGUF quantization
        transformer = FluxTransformer2DModel.from_single_file(
            cache_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            torch_dtype=torch_dtype,
        )

        # Create the pipeline with the quantized transformer
        log.info("Creating FLUX pipeline...")

        # For GGUF models, always use the standard repos for the pipeline
        # Detect schnell vs dev model from the filename
        if "schnell" in self.model.path.lower():
            base_model_id = "black-forest-labs/FLUX.1-schnell"
        else:
            base_model_id = "black-forest-labs/FLUX.1-dev"

        log.info(
            f"Loading FLUX pipeline from {base_model_id} with quantized transformer..."
        )
        self._pipeline = FluxPipeline.from_pretrained(
            base_model_id,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )

        # Apply CPU offload if enabled
        if self.enable_cpu_offload:
            self._pipeline.enable_sequential_cpu_offload()

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # If CPU offload is enabled, we need to handle device movement differently
            if self.enable_cpu_offload:
                # With CPU offload, components are automatically managed
                # When moving to CPU, we should disable CPU offload and move everything to CPU
                if device == "cpu":
                    # Disable CPU offload and move all components to CPU
                    self._pipeline.to(device)
                # When moving to GPU with CPU offload, re-enable CPU offload
                elif device in ["cuda", "mps"]:
                    self._pipeline.enable_sequential_cpu_offload()
            else:
                # Normal device movement without CPU offload
                self._pipeline.to(device)

            # Apply memory optimizations only when on GPU
            if device != "cpu":
                if self.enable_vae_slicing:
                    self._pipeline.vae.enable_slicing()

                if self.enable_vae_tiling:
                    self._pipeline.vae.enable_tiling()

                if self.enable_memory_efficient_attention:
                    self._pipeline.enable_attention_slicing()

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # Adjust parameters based on detected variant
        detected_variant = self._detect_variant_from_repo_id()
        guidance_scale = self.guidance_scale
        num_inference_steps = self.num_inference_steps
        max_sequence_length = self.max_sequence_length

        if detected_variant == FluxVariant.SCHNELL:
            # For schnell, guidance_scale should be 0 and max_sequence_length <= 256
            if guidance_scale != 0.0:
                log.warning(
                    "For FLUX.1-schnell, guidance_scale should be 0.0. Adjusting automatically."
                )
                guidance_scale = 0.0
            if max_sequence_length > 256:
                log.warning(
                    "For FLUX.1-schnell, max_sequence_length should be <= 256. Adjusting to 256."
                )
                max_sequence_length = 256
            if num_inference_steps > 10:
                log.warning(
                    "For FLUX.1-schnell, fewer inference steps (4-8) are recommended for optimal performance."
                )

        def progress_callback(
            pipeline: Any, step: int, timestep: int, callback_kwargs: dict
        ) -> dict:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=num_inference_steps,
                )
            )
            return callback_kwargs

        # Generate the image off the event loop
        def _run_pipeline_sync():
            return self._pipeline(
                prompt=self.prompt,
                guidance_scale=guidance_scale,
                height=self.height,
                width=self.width,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                generator=generator,
                callback_on_step_end=progress_callback,  # type: ignore
                callback_on_step_end_tensor_inputs=["latents"],
            )

        output = await asyncio.to_thread(_run_pipeline_sync)

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)


class Kolors(HuggingFacePipelineNode):
    """
    Generates images from text prompts using Kolors, a large-scale text-to-image generation model.
    image, generation, AI, text-to-image, kolors, chinese, english

    Use cases:
    - Generate high-quality photorealistic images from text descriptions
    - Create images with Chinese text understanding and rendering
    - Produce images with complex semantic accuracy
    - Generate images with both Chinese and English text support
    - Create detailed images with strong text rendering capabilities
    """

    prompt: str = Field(
        default='A ladybug photo, macro, zoom, high quality, film, holding a sign that says "可图"',
        description="A text prompt describing the desired image. Supports both Chinese and English.",
    )
    negative_prompt: str = Field(
        default="",
        description="A text prompt describing what to avoid in the image.",
    )
    guidance_scale: float = Field(
        default=6.5,
        description="The scale for classifier-free guidance.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=25,
        description="The number of denoising steps.",
        ge=1,
        le=100,
    )
    width: int = Field(
        default=1024,
        description="The width of the generated image.",
        ge=64,
        le=2048,
    )
    height: int = Field(
        default=1024,
        description="The height of the generated image.",
        ge=64,
        le=2048,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )
    max_sequence_length: int = Field(
        default=256,
        description="Maximum sequence length for the prompt.",
        ge=1,
        le=512,
    )
    use_dpm_solver: bool = Field(
        default=True,
        description="Whether to use DPMSolverMultistepScheduler with Karras sigmas for better quality.",
    )

    _pipeline: KolorsPipeline | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFFlux]:
        return [
            HFFlux(
                repo_id="Kwai-Kolors/Kolors-diffusers",
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
        return "Kolors Text2Image"

    def get_model_id(self) -> str:
        return "Kwai-Kolors/Kolors-diffusers"

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.get_model_id(),
            model_class=KolorsPipeline,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Set up the scheduler as recommended in the docs
        if self._pipeline is not None and self.use_dpm_solver:
            self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self._pipeline.scheduler.config, use_karras_sigmas=True
            )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # Generate the image off the event loop
        def _run_pipeline_sync():
            return self._pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                height=self.height,
                width=self.width,
                generator=generator,
                max_sequence_length=self.max_sequence_length,
                callback_on_step_end=pipeline_progress_callback(self.id, self.num_inference_steps, context),  # type: ignore
                callback_on_step_end_tensor_inputs=["latents"],
            )

        output = await asyncio.to_thread(_run_pipeline_sync)

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "prompt",
            "height",
            "width",
            "seed",
        ]


class Chroma(HuggingFacePipelineNode):
    """
    Generates images from text prompts using Chroma, a text-to-image model based on Flux.
    image, generation, AI, text-to-image, flux, chroma, transformer

    Use cases:
    - Generate high-quality images with Flux-based architecture
    - Create images with advanced attention masking for enhanced fidelity
    - Produce images with IP adapter support for style control
    - Generate images with optimized memory usage
    - Create professional-quality images with precise color control
    """

    prompt: str = Field(
        default="A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done.",
        description="A text prompt describing the desired image.",
    )
    negative_prompt: str = Field(
        default="low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors",
        description="A text prompt describing what to avoid in the image.",
    )
    guidance_scale: float = Field(
        default=3.0,
        description="The scale for classifier-free guidance.",
        ge=0.0,
        le=30.0,
    )
    num_inference_steps: int = Field(
        default=40,
        description="The number of denoising steps.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=1024,
        description="The height of the generated image.",
        ge=256,
        le=2048,
    )
    width: int = Field(
        default=1024,
        description="The width of the generated image.",
        ge=256,
        le=2048,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum sequence length to use with the prompt.",
        ge=1,
        le=512,
    )
    ip_adapter_image: ImageRef | None = Field(
        default=None,
        description="Optional image input for IP Adapter style control.",
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Enable CPU offload to reduce VRAM usage.",
    )
    enable_vae_slicing: bool = Field(
        default=True,
        description="Enable VAE slicing to reduce VRAM usage.",
    )
    enable_vae_tiling: bool = Field(
        default=True,
        description="Enable VAE tiling to reduce VRAM usage for large images.",
    )
    enable_attention_slicing: bool = Field(
        default=True,
        description="Enable attention slicing to reduce memory usage.",
    )

    _pipeline: ChromaPipeline | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFFlux]:
        return [
            HFFlux(
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
        # Load the pipeline with bfloat16 as recommended
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.get_model_id(),
            model_class=ChromaPipeline,
            torch_dtype=torch.bfloat16,
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # Handle CPU offload case
            if self.enable_cpu_offload:
                # When moving to CPU, disable CPU offload and move all components to CPU
                if device == "cpu":
                    self._pipeline.to(device)
                # When moving to GPU with CPU offload, re-enable CPU offload
                elif device in ["cuda", "mps"]:
                    self._pipeline.enable_model_cpu_offload()
            else:
                # Normal device movement without CPU offload
                self._pipeline.to(device)

            # Apply memory optimizations only when on GPU
            if device != "cpu":
                if self.enable_vae_slicing:
                    self._pipeline.enable_vae_slicing()

                if self.enable_vae_tiling:
                    self._pipeline.enable_vae_tiling()

                if self.enable_attention_slicing:
                    self._pipeline.enable_attention_slicing()

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # Process IP adapter image if provided
        ip_adapter_image = None
        if self.ip_adapter_image is not None:
            ip_adapter_image = await context.image_to_pil(self.ip_adapter_image)

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
            "callback_on_step_end": pipeline_progress_callback(self.id, self.num_inference_steps, context),  # type: ignore
            "callback_on_step_end_tensor_inputs": ["latents"],
        }

        # Add IP adapter image if provided
        if ip_adapter_image is not None:
            pipeline_kwargs["ip_adapter_image"] = ip_adapter_image

        # Generate the image off the event loop
        pipeline = self._pipeline
        assert pipeline is not None

        def _run_pipeline_sync():
            return pipeline(**pipeline_kwargs)

        output = await asyncio.to_thread(_run_pipeline_sync)

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "prompt",
            "height",
            "width",
            "seed",
        ]


# Constants for FP8 text encoder
FP8_TEXT_ENCODER_REPO = "Comfy-Org/Qwen-Image_ComfyUI"
FP8_TEXT_ENCODER_PATH = (
    "split_files/text_encoders/qwen_image_text_encoder_fp8.safetensors"
)


class QwenImage(HuggingFacePipelineNode):
    """
    Generates images from text prompts using Qwen-Image with support for GGUF quantization.
    image, generation, AI, text-to-image, qwen, quantization

    Use cases:
    - High-quality, general-purpose text-to-image generation
    - Memory-efficient generation using GGUF quantization
    - Quick prototyping leveraging AutoPipeline
    - Works out-of-the-box with the official Qwen model
    """

    model: HFQwenImage = Field(
        default=HFQwenImage(),
        description="The Qwen-Image model to use for text-to-image generation.",
    )
    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="A text prompt describing the desired image.",
    )
    negative_prompt: str = Field(
        default="",
        description="A text prompt describing what to avoid in the image.",
    )
    true_cfg_scale: float = Field(
        default=1.0,
        description="True CFG scale for Qwen-Image models.",
        ge=0.0,
        le=10.0,
    )
    num_inference_steps: int = Field(
        default=20,
        description="The number of denoising steps.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=1024,
        description="The height of the generated image.",
        ge=256,
        le=2048,
    )
    width: int = Field(
        default=1024,
        description="The width of the generated image.",
        ge=256,
        le=2048,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )
    enable_memory_efficient_attention: bool = Field(
        default=True,
        description="Enable memory efficient attention to reduce VRAM usage.",
    )
    enable_vae_tiling: bool = Field(
        default=False,
        description="Enable VAE tiling to reduce VRAM usage for large images.",
    )
    enable_vae_slicing: bool = Field(
        default=False,
        description="Enable VAE slicing to reduce VRAM usage.",
    )
    enable_cpu_offload: bool = Field(
        default=False,
        description="Enable CPU offload to reduce VRAM usage.",
    )

    _pipeline: Any | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFQwenImage]:
        return [
            HFQwenImage(
                repo_id="Qwen/Qwen-Image",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            # GGUF quantized models
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-Q2_K.gguf",
            ),
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-Q3_K_S.gguf",
            ),
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-Q4_K_S.gguf",
            ),
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-Q5_K_S.gguf",
            ),
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-Q6_K.gguf",
            ),
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-Q8_0.gguf",
            ),
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-BF16.gguf",
            ),
            # FP8 Text Encoder for optimized performance
            HFQwenImage(
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                path="split_files/text_encoders/qwen_image_text_encoder_fp8.safetensors",
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Qwen-Image"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "prompt",
            "negative_prompt",
            "height",
            "width",
            "num_inference_steps",
        ]

    def get_model_id(self) -> str:
        if self.model.repo_id:
            return self.model.repo_id
        return "Qwen/Qwen-Image"

    def _detect_gguf_quantization_type(self) -> str | None:
        """Detect GGUF quantization type from model path."""
        if not self.model.path:
            return None

        # Extract quantization type from filename patterns like "qwen-image-Q2_K.gguf"
        import re

        pattern = r"-(Q\\d+_[A-Z]+|Q\\d+|BF16)\\.gguf$"
        match = re.search(pattern, self.model.path, re.IGNORECASE)
        return match.group(1) if match else None

    def _is_gguf_model(self) -> bool:
        """Check if the model is a GGUF model based on file extension."""
        return self.model.path is not None and self.model.path.lower().endswith(".gguf")

    async def _load_fp8_text_encoder(self, context: ProcessingContext):
        """Load FP8 text encoder using the base class's load_model method."""
        try:
            from transformers import T5EncoderModel

            log.info(
                f"Loading FP8 text encoder from {FP8_TEXT_ENCODER_REPO}/{FP8_TEXT_ENCODER_PATH}"
            )

            # Use the base class's load_model method
            text_encoder = await self.load_model(
                context=context,
                model_class=T5EncoderModel,
                model_id=FP8_TEXT_ENCODER_REPO,
                path=FP8_TEXT_ENCODER_PATH,
                torch_dtype=(
                    torch.float8_e4m3fn
                    if hasattr(torch, "float8_e4m3fn")
                    else torch.bfloat16
                ),
                skip_cache=False,
            )

            log.info("Successfully loaded FP8 text encoder")
            return text_encoder

        except Exception as e:
            log.warning(
                f"Failed to load FP8 text encoder: {e}. Falling back to default."
            )
            return None

    async def preload_model(self, context: ProcessingContext):
        # Load FP8 text encoder
        fp8_text_encoder = await self._load_fp8_text_encoder(context)

        # Check if this is a GGUF model based on file extension
        if self._is_gguf_model():
            await self._load_gguf_model(context, torch.bfloat16, fp8_text_encoder)
        else:
            # Load the full pipeline normally
            log.info(f"Loading Qwen-Image pipeline from {self.get_model_id()}...")

            # Prepare additional arguments for custom text encoder
            load_kwargs = {
                "context": context,
                "model_id": self.get_model_id(),
                "model_class": AutoPipelineForText2Image,
                "torch_dtype": torch.bfloat16,
                "device": "cpu",  # Load on CPU first, then move to GPU in workflow runner
            }

            # Add FP8 text encoder if available
            if fp8_text_encoder is not None:
                load_kwargs["text_encoder"] = fp8_text_encoder

            self._pipeline = await self.load_model(**load_kwargs)
            assert self._pipeline is not None

            # Apply memory optimizations after loading
            if self.enable_cpu_offload:
                self._pipeline.enable_model_cpu_offload()

            if self.enable_vae_slicing:
                self._pipeline.vae.enable_slicing()

            if self.enable_vae_tiling:
                self._pipeline.vae.enable_tiling()

            if self.enable_memory_efficient_attention:
                self._pipeline.enable_attention_slicing()

    async def _load_gguf_model(
        self, context: ProcessingContext, torch_dtype, fp8_text_encoder=None
    ):
        """Load Qwen-Image model with GGUF quantization."""
        from huggingface_hub.file_download import try_to_load_from_cache
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformer2DModel,
        )

        quantization_type = self._detect_gguf_quantization_type()

        if quantization_type:
            log.info(
                f"Loading Qwen-Image transformer with GGUF {quantization_type} quantization..."
            )
        else:
            log.info(
                "Loading GGUF model (quantization type not detected from filename)..."
            )

        # Get the cached file path
        assert self.model.path is not None
        cache_path = try_to_load_from_cache(self.get_model_id(), self.model.path)
        if not cache_path:
            raise ValueError(
                f"Model {self.get_model_id()}/{self.model.path} must be downloaded first"
            )

        # Load the transformer with GGUF quantization
        transformer = self.load_model(
            context=context,
            model_class=QwenImageTransformer2DModel,
            model_id=self.get_model_id(),
            path=cache_path,
            torch_dtype=torch_dtype,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            config="Qwen/Qwen-Image",
            subfolder="transformer",
        )

        # Create the pipeline with the quantized transformer
        log.info("Creating Qwen-Image pipeline with quantized transformer...")

        # Use DiffusionPipeline for GGUF models
        pipeline_kwargs = {
            "transformer": transformer,
            "torch_dtype": torch_dtype,
        }

        # Add FP8 text encoder if provided
        if fp8_text_encoder is not None:
            pipeline_kwargs["text_encoder"] = fp8_text_encoder

        self._pipeline = await self.load_model(
            context=context,
            model_class=DiffusionPipeline,
            model_id="Qwen/Qwen-Image",
            **pipeline_kwargs,
        )

        # Apply memory optimizations after loading
        if self.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()

        if self.enable_vae_slicing:
            self._pipeline.vae.enable_slicing()

        if self.enable_vae_tiling:
            self._pipeline.vae.enable_tiling()

        if self.enable_memory_efficient_attention:
            self._pipeline.enable_attention_slicing()

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # Handle CPU offload case
            if self.enable_cpu_offload:
                # When moving to CPU, disable CPU offload and move all components to CPU
                if device == "cpu":
                    self._pipeline.to(device)
                # When moving to GPU with CPU offload, re-enable CPU offload
                elif device in ["cuda", "mps"]:
                    self._pipeline.enable_model_cpu_offload()
            else:
                # Normal device movement without CPU offload
                self._pipeline.to(device)

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
            return self._pipeline(**pipeline_kwargs)  # type: ignore

        output = await asyncio.to_thread(_run_pipeline_sync)

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)

    def required_inputs(self):
        """Return list of required inputs that must be connected."""
        return []  # No required inputs - IP adapter image is optional
