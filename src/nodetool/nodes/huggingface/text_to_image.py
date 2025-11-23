from enum import Enum
import huggingface_hub
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from typing import Any, TypedDict
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    HFFlux,
    HFQwenImage,
    HFStableDiffusionXL,
    HFTextToImage,
    HFControlNetFlux,
    ImageRef,
    TorchTensor,
)
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.nodes.huggingface.image_to_image import pipeline_progress_callback
from nodetool.nodes.huggingface.stable_diffusion_base import (
    ModelVariant,
    _select_diffusion_dtype,
    StableDiffusionBaseNode,
    StableDiffusionXLBase,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, Notification, LogUpdate

import torch
import asyncio
import logging
from pathlib import Path
from nunchaku import (
    NunchakuFluxTransformer2dModel,
    NunchakuQwenImageTransformer2DModel,
)
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
from nunchaku.utils import get_gpu_memory, get_precision as get_nunchaku_precision

# The QwenImage import requires optional dependencies. Keep it near top-level to surface missing deps early.
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.chroma.pipeline_chroma import ChromaPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux_control import FluxControlPipeline
from diffusers.pipelines.kolors.pipeline_kolors import KolorsPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from diffusers.quantizers.quantization_config import GGUFQuantizationConfig
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from pydantic import Field
from transformers import T5EncoderModel
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


log = get_logger(__name__)


def _enable_pytorch2_attention(pipeline: Any):
    """Enable PyTorch 2 scaled dot product attention to speed up inference."""
    enable_sdpa = getattr(pipeline, "enable_sdpa", None)

    if callable(enable_sdpa):
        try:
            enable_sdpa()
            pipeline_name = type(pipeline).__name__
            log.info("Enabled PyTorch 2 scaled dot product attention for %s", pipeline_name)
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
    _pipeline: StableDiffusionPipeline | None = None

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
        self._load_ip_adapter()


    async def process(self, context: ProcessingContext) -> OutputType:
        result = await self.run_pipeline(context, width=self.width, height=self.height)
        return {
            "image": result if isinstance(result, ImageRef) else None,
            "latent": result if isinstance(result, TorchTensor) else None,
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

    _pipeline: StableDiffusionXLPipeline | DiffusionPipeline | None = None
    _using_playground_pipeline: bool = False

    @classmethod
    def get_title(cls):
        return "Stable Diffusion XL"

    async def preload_model(self, context: ProcessingContext):
        repo_id = (self.model.repo_id or "").lower()
        is_playground = "playground" in repo_id

        self._using_playground_pipeline = is_playground

        torch_dtype = self.get_torch_dtype()
        pipeline_model_id = self._resolve_sdxl_pipeline_model_id()

        if self._is_nunchaku_model():
            await self._load_nunchaku_model(context, torch_dtype)
        else:
            self._pipeline = await self.load_model(
                context=context,
                model_class=StableDiffusionXLPipeline,
                model_id=pipeline_model_id,
                path=self.model.path,
                torch_dtype=torch_dtype,
                variant=None,
            )
            _enable_pytorch2_attention(self._pipeline)
            _apply_vae_optimizations(self._pipeline)

        assert self._pipeline is not None
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    def _is_nunchaku_model(self) -> bool:
        repo_has_svdq = self.model.repo_id and "svdq" in self.model.repo_id.lower()
        path_has_svdq = self.model.path and "svdq" in self.model.path.lower()
        return bool(repo_has_svdq or path_has_svdq)

    def _resolve_sdxl_pipeline_model_id(self) -> str:
        if not self.model.repo_id:
            return "stabilityai/stable-diffusion-xl-base-1.0"
        repo_id_lower = self.model.repo_id.lower()
        if "nunchaku" in repo_id_lower and "sdxl" in repo_id_lower:
            return "stabilityai/stable-diffusion-xl-base-1.0"
        return self.model.repo_id

    async def _load_nunchaku_model(
        self,
        context: ProcessingContext,
        torch_dtype: torch.dtype,
    ):
        hf_token = await context.get_secret("HF_TOKEN")
        repo_id_lower = (self.model.repo_id or "").lower()
        transformer_repo_id = (
            self.model.repo_id
            if self.model.repo_id and "nunchaku" in repo_id_lower and "sdxl" in repo_id_lower
            else "nunchaku-tech/nunchaku-sdxl"
        )

        try:
            precision = get_nunchaku_precision()
        except Exception as exc:
            log.warning(
                "Failed to detect Nunchaku SDXL precision; defaulting to int4: %s", exc
            )
            precision = "int4"

        transformer_precision = precision
        if transformer_precision.startswith("nv"):
            transformer_precision = transformer_precision[2:]

        default_transformer_path = (
            self.model.path
            or f"svdq-{transformer_precision}_r32-sdxl.safetensors"
        )
        transformer_path = default_transformer_path.strip()

        transformer_file = Path(transformer_path).expanduser()
        transformer_identifier: str | None = None
        transformer_filename = transformer_path

        if transformer_file.is_file():
            transformer_identifier = str(transformer_file)
            transformer_filename = transformer_file.name
        else:
            repo_from_path: str | None = None
            if not transformer_path.startswith(("/", "~")):
                path_parts = transformer_path.split("/")
                if len(path_parts) >= 3:
                    repo_from_path = "/".join(path_parts[:2])
                    transformer_filename = "/".join(path_parts[2:])

            if repo_from_path:
                transformer_repo_id = repo_from_path

            if "svdq" not in transformer_filename.lower():
                raise ValueError(
                    "Nunchaku SDXL requires a transformer filename containing 'svdq'."
                )

            cache_path = try_to_load_from_cache(transformer_repo_id, transformer_filename)
            if not cache_path:
                log.info(
                    "Downloading Nunchaku SDXL transformer %s/%s (precision=%s)",
                    transformer_repo_id,
                    transformer_filename,
                    precision,
                )
                hf_hub_download(
                    transformer_repo_id,
                    transformer_filename,
                    token=hf_token,
                )
                cache_path = try_to_load_from_cache(
                    transformer_repo_id,
                    transformer_filename,
                )
                if not cache_path:
                    raise ValueError(
                        f"Downloading model {transformer_repo_id}/{transformer_filename} failed"
                    )

            transformer_identifier = cache_path

        assert transformer_identifier is not None

        log.info(
            "Loading Nunchaku SDXL UNet from %s/%s (precision=%s)",
            transformer_repo_id,
            transformer_filename,
            precision,
        )
        unet = await asyncio.to_thread(
            NunchakuSDXLUNet2DConditionModel.from_pretrained,
            transformer_identifier,
            torch_dtype=torch_dtype,
            device="cpu",
        )

        pipeline_model_id = self._resolve_sdxl_pipeline_model_id()
        log.info(
            "Creating StableDiffusionXL pipeline from %s with Nunchaku UNet",
            pipeline_model_id,
        )
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionXLPipeline,
            model_id=pipeline_model_id,
            torch_dtype=torch_dtype,
            unet=unet,
            use_safetensors=True,
            token=hf_token,
        )

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
    Load HuggingFace model for image-to-image generation from a repo_id.

    Use cases:
    - Loads a pipeline directly from a repo_id
    - Used for AutoPipelineForImage2Image
    """

    repo_id: str = Field(
        default="",
        description="The repository ID of the model to use for image-to-image generation.",
    )

    async def preload_model(self, context: ProcessingContext):
        torch_dtype = _select_diffusion_dtype()
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
        torch_dtype = _select_diffusion_dtype()
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
            except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                raise ValueError(
                    "VRAM out of memory while moving Kolors pipeline to device. "
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
                return self._pipeline(**call_kwargs)  # type: ignore

        output = await asyncio.to_thread(_run_pipeline_sync)

        image = output.images[0]  # type: ignore
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
        description="The number of denoising steps. 4 steps is forced for schnell models.",
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
    enable_cpu_offload: bool = Field(
        default=True,
        description="Enable CPU offload to reduce VRAM usage.",
    )

    _pipeline: FluxPipeline | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFFlux]:
        return [
            # Original Flux repos
            HFFlux(
                repo_id="black-forest-labs/FLUX.1-dev",
            ),
            HFFlux(
                repo_id="black-forest-labs/FLUX.1-schnell",
            ),
            # Nunchaku SVDQ transformers (auto-detected precision)
            HFFlux(
                repo_id="nunchaku-tech/nunchaku-flux.1-dev",
                path="svdq-int4_r32-flux.1-dev.safetensors",
            ),
            HFFlux(
                repo_id="nunchaku-tech/nunchaku-flux.1-dev",
                path="svdq-fp4_r32-flux.1-dev.safetensors",
            ),
            # GGUF quantized models
            HFFlux(
                repo_id="city96/FLUX.1-dev-gguf",
                path="flux1-dev-Q4_K_S.gguf",
            ),
            # FLUX.1-schnell GGUF models
            HFFlux(
                repo_id="city96/FLUX.1-schnell-gguf",
                path="flux1-schnell-Q4_K_S.gguf",
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

    def _get_base_model_id(self, variant: FluxVariant) -> str:
        model_mapping = {
            FluxVariant.SCHNELL: "black-forest-labs/FLUX.1-schnell",
            FluxVariant.DEV: "black-forest-labs/FLUX.1-dev",
            FluxVariant.FILL_DEV: "black-forest-labs/FLUX.1-Fill-dev",
            FluxVariant.CANNY_DEV: "black-forest-labs/FLUX.1-Canny-dev",
            FluxVariant.DEPTH_DEV: "black-forest-labs/FLUX.1-Depth-dev",
        }
        return model_mapping.get(variant, "black-forest-labs/FLUX.1-dev")

    def get_model_id(self) -> str:
        if self.model.repo_id:
            return self.model.repo_id
        # Fallback to detected variant
        detected_variant = self._detect_variant_from_repo_id()
        return self._get_base_model_id(detected_variant)

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

    def _is_gguf_model(self) -> bool:
        """Check if the model is a GGUF model based on file extension."""
        return self.model.path is not None and self.model.path.lower().endswith(".gguf")

    def _is_nunchaku_model(self) -> bool:
        """Detect Nunchaku SVDQ transformers via repo or filename."""
        repo_has_svdq = self.model.repo_id and "svdq" in self.model.repo_id.lower()
        path_has_svdq = self.model.path and "svdq" in self.model.path.lower()
        return bool(repo_has_svdq or path_has_svdq)

    async def preload_model(self, context: ProcessingContext):
        hf_token = await context.get_secret("HF_TOKEN")
        if not hf_token:
            model_url = f"https://huggingface.co/{self.get_model_id()}"
            raise ValueError(
                f"Flux is a gated model, please set the HF_TOKEN in Nodetool settings and accept the terms of use for the model: {model_url}"
            )

        # Determine torch dtype based on variant
        # Auto-detect variant from model selection
        detected_variant = self._detect_variant_from_repo_id()

        torch_dtype = (
            torch.bfloat16
            if detected_variant in [FluxVariant.SCHNELL, FluxVariant.DEV]
            else torch.float16
        )

        log.info(f"Using torch_dtype: {torch_dtype}")

        # Check if this is a GGUF model based on file extension
        if self._is_gguf_model():
            await self._load_gguf_model(context, torch_dtype, hf_token)
        elif self._is_nunchaku_model():
            await self._load_nunchaku_model(
                context=context,
                torch_dtype=torch_dtype,
                detected_variant=detected_variant,
                hf_token=hf_token,
            )
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
                token=hf_token,
            )
            _enable_pytorch2_attention(self._pipeline)
            _apply_vae_optimizations(self._pipeline)


        # Apply CPU offload if enabled
        if self._pipeline is not None and self.enable_cpu_offload:
            self._pipeline.enable_sequential_cpu_offload()

    async def _load_gguf_model(
        self,
        context: ProcessingContext,
        torch_dtype: torch.dtype,
        hf_token: str | None = None,
    ):
        """Load FLUX model with GGUF quantization."""
        from huggingface_hub.file_download import try_to_load_from_cache

        if hf_token:
            log.info(f"Using HF_TOKEN: {hf_token[:4]}...{hf_token[-4:]}")

        # Load the transformer with GGUF quantization
        transformer = await self.load_model(
            context=context,
            model_class=FluxTransformer2DModel,
            model_id=self.model.repo_id,
            path=self.model.path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            torch_dtype=torch_dtype,
            token=hf_token,
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
            token=hf_token,
        )

        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

        if self.enable_cpu_offload and self._pipeline is not None:
            self._pipeline.enable_sequential_cpu_offload()

    async def _load_nunchaku_model(
        self,
        context: ProcessingContext,
        torch_dtype: torch.dtype,
        detected_variant: FluxVariant,
        hf_token: str | None = None,
    ):
        """Load FLUX pipeline using a Nunchaku SVDQ transformer file."""
        repo_id_lower = (self.model.repo_id or "").lower()
        transformer_repo_id = self.model.repo_id or "nunchaku-tech/nunchaku-flux.1-dev"
        if "svdq" not in repo_id_lower and "nunchaku" not in repo_id_lower:
            transformer_repo_id = "nunchaku-tech/nunchaku-flux.1-dev"

        precision = get_nunchaku_precision()
        transformer_path = (
            self.model.path
            or f"svdq-{precision}_r32-flux.1-dev.safetensors"
        )

        if "svdq" not in transformer_path.lower():
            raise ValueError(
                "Nunchaku Flux requires a transformer filename containing 'svdq'."
            )

        log.info(
            "Loading Nunchaku transformer from %s/%s (precision=%s)",
            transformer_repo_id,
            transformer_path,
            precision,
        )

        transformer_file = Path(transformer_path).expanduser()
        cache_path: str | None = None

        if transformer_file.is_file():
            transformer_identifier = str(transformer_file)
        else:
            cache_path = try_to_load_from_cache(transformer_repo_id, transformer_path)
            if not cache_path:
                log.info(
                    "Downloading Nunchaku transformer %s/%s to cache",
                    transformer_repo_id,
                    transformer_path,
                )
                hf_hub_download(
                    transformer_repo_id,
                    transformer_path,
                    token=hf_token,
                )
                cache_path = try_to_load_from_cache(
                    transformer_repo_id,
                    transformer_path,
                )
                if not cache_path:
                    raise ValueError(
                        f"Downloading model {transformer_repo_id}/{transformer_path} from HuggingFace failed"
                    )

            transformer_identifier = cache_path or f"{transformer_repo_id}/{transformer_path}"

        transformer = await asyncio.to_thread(
            NunchakuFluxTransformer2dModel.from_pretrained,
            transformer_identifier,
            config=transformer_repo_id,
            torch_dtype=torch_dtype,
            token=hf_token,
        )

        base_model_id = self._get_base_model_id(detected_variant)
        log.info(
            "Creating FLUX pipeline from %s with Nunchaku transformer %s/%s",
            base_model_id,
            transformer_repo_id,
            transformer_path,
        )

        try:
            self._pipeline = FluxPipeline.from_pretrained(
                base_model_id,
                transformer=transformer,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while loading Flux with the Nunchaku transformer. "
                "Try enabling CPU offload or reduce image size/steps."
            ) from e

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # If CPU offload is enabled, we need to handle device movement differently
            if self.enable_cpu_offload:
                # With CPU offload, components are automatically managed
                # When moving to CPU, we should disable CPU offload and move everything to CPU
                if device == "cpu":
                    # Disable CPU offload and move all components to CPU
                    try:
                        self._pipeline.to(device)
                    except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                        raise ValueError(
                            "VRAM out of memory while moving Flux to device. "
                            "Enable 'CPU offload' in the advanced node properties or reduce image size/steps."
                        ) from e
                # When moving to GPU with CPU offload, re-enable CPU offload
                elif device in ["cuda", "mps"]:
                    self._pipeline.enable_sequential_cpu_offload()
            else:
                # Normal device movement without CPU offload
                try:
                    self._pipeline.to(device)
                except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                    raise ValueError(
                        "VRAM out of memory while moving Flux to device. "
                        "Try enabling 'CPU offload' in the advanced node properties, reduce image size, or lower steps."
                    ) from e

            _apply_vae_optimizations(self._pipeline)

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
            guidance_scale = 0.0
            max_sequence_length = 256
            num_inference_steps = 4

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
            with torch.inference_mode():
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

        try:
            output = await asyncio.to_thread(_run_pipeline_sync)
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while running Flux. "
                "Try enabling 'CPU offload' in the advanced node properties "
                "(Enable CPU offload), reduce image size, or lower steps."
            ) from e

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
    def get_recommended_models(cls) -> list[HFTextToImage]:
        return [
            HFTextToImage(
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
        torch_dtype = _select_diffusion_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.get_model_id(),
            model_class=KolorsPipeline,
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None,
        )

        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

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
            with torch.inference_mode():
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

        try:
            output = await asyncio.to_thread(_run_pipeline_sync)
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while running Qwen-Image. "
                "Enable 'CPU offload' in the advanced node properties (Enable CPU offload), "
                "or reduce image size/steps."
            ) from e

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
    enable_cpu_offload: bool = Field(
        default=True,
        description="Enable CPU offload to reduce VRAM usage.",
    )
    enable_attention_slicing: bool = Field(
        default=True,
        description="Enable attention slicing to reduce memory usage.",
    )

    _pipeline: ChromaPipeline | None = None

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
        torch_dtype = _select_diffusion_dtype()
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
                    except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
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
                except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
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
            "callback_on_step_end": pipeline_progress_callback(self.id, self.num_inference_steps, context),  # type: ignore
            "callback_on_step_end_tensor_inputs": ["latents"],
        }

        # Generate the image off the event loop
        pipeline = self._pipeline
        assert pipeline is not None

        def _run_pipeline_sync():
            with torch.inference_mode():
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


# TODO: Wait for diffusers release
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
        default=50,
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
    enable_cpu_offload: bool = Field(
        default=False,
        description="Enable CPU offload to reduce VRAM usage.",
    )

    _pipeline: Any | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFQwenImage]:
        return [
            # Nunchaku SVDQ transformers (auto-detected precision)
            HFQwenImage(
                repo_id="nunchaku-tech/nunchaku-qwen-image",
                path="svdq-int4_r32-qwen-image.safetensors",
            ),
            HFQwenImage(
                repo_id="nunchaku-tech/nunchaku-qwen-image",
                path="svdq-fp4_r32-qwen-image.safetensors",
            ),
            # GGUF quantized models
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-Q4_K_M.gguf",
            ),
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-Q8_0.gguf",
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

    def _is_gguf_model(self) -> bool:
        """Check if the model is a GGUF model based on file extension."""
        return self.model.path is not None and self.model.path.lower().endswith(".gguf")

    def _is_nunchaku_model(self) -> bool:
        """Detect Nunchaku SVDQ transformers via repo or filename."""
        repo_has_svdq = self.model.repo_id and "svdq" in self.model.repo_id.lower()
        path_has_svdq = self.model.path and "svdq" in self.model.path.lower()
        return bool(repo_has_svdq or path_has_svdq)

    async def preload_model(self, context: ProcessingContext):
        # Handle GGUF models separately to maintain existing behaviour
        if self._is_gguf_model():
            await self._load_gguf_model(context, _select_diffusion_dtype())
        elif self._is_nunchaku_model():
            await self._load_nunchaku_model(context, _select_diffusion_dtype())
        else:
            await self._load_full_precision_pipeline(context)

    async def _load_full_precision_pipeline(self, context: ProcessingContext):
        log.info(
            f"Loading Qwen-Image pipeline from {self.get_model_id()} without quantization..."
        )

        torch_dtype = _select_diffusion_dtype()

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

        self._pipeline.enable_model_cpu_offload()

    async def _load_nunchaku_model(
        self,
        context: ProcessingContext,
        torch_dtype,
    ):
        """Load Qwen-Image pipeline using a Nunchaku SVDQ transformer file."""
        repo_id_lower = (self.model.repo_id or "").lower()
        transformer_repo_id = self.model.repo_id or "nunchaku-tech/nunchaku-qwen-image"
        if "svdq" not in repo_id_lower and "nunchaku" not in repo_id_lower:
            transformer_repo_id = "nunchaku-tech/nunchaku-qwen-image"

        precision = get_nunchaku_precision()
        transformer_path = self.model.path or f"svdq-{precision}_r32-qwen-image.safetensors"

        if "svdq" not in transformer_path.lower():
            raise ValueError(
                "Nunchaku Qwen-Image requires a transformer filename containing 'svdq'."
            )

        log.info(
            "Loading Nunchaku Qwen-Image transformer from %s/%s (precision=%s)",
            transformer_repo_id,
            transformer_path,
            precision,
        )

        transformer_file = Path(transformer_path).expanduser()
        cache_path: str | None = None

        if transformer_file.is_file():
            transformer_identifier = str(transformer_file)
        else:
            cache_path = try_to_load_from_cache(transformer_repo_id, transformer_path)
            if not cache_path:
                log.info(
                    "Downloading Nunchaku Qwen-Image transformer %s/%s to cache",
                    transformer_repo_id,
                    transformer_path,
                )
                hf_token = await context.get_secret("HF_TOKEN")
                hf_hub_download(
                    transformer_repo_id,
                    transformer_path,
                    token=hf_token,
                )
                cache_path = try_to_load_from_cache(
                    transformer_repo_id,
                    transformer_path,
                )
                if not cache_path:
                    raise ValueError(
                        f"Downloading model {transformer_repo_id}/{transformer_path} from HuggingFace failed"
                    )

            transformer_identifier = cache_path or f"{transformer_repo_id}/{transformer_path}"

        hf_token = await context.get_secret("HF_TOKEN")

        transformer = await asyncio.to_thread(
            NunchakuQwenImageTransformer2DModel.from_pretrained,
            transformer_identifier,
            config="Qwen/Qwen-Image",
            torch_dtype=torch_dtype,
            token=hf_token,
        )

        log.info(
            "Creating Qwen-Image pipeline from Qwen/Qwen-Image with Nunchaku transformer %s/%s",
            transformer_repo_id,
            transformer_path,
        )

        try:
            self._pipeline = QwenImagePipeline.from_pretrained(
                "Qwen/Qwen-Image",
                transformer=transformer,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while loading Qwen-Image with the Nunchaku transformer. "
                "Try enabling CPU offload or reduce image size/steps."
            ) from e

        # Apply memory and offload optimizations
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

        if self.enable_cpu_offload and self._pipeline is not None:
            try:
                gpu_mem_gb = get_gpu_memory()
            except Exception as e:  # pragma: no cover - GPU utility may not be available in all envs
                log.warning(
                    "Failed to query GPU memory for Nunchaku Qwen-Image: %s. "
                    "Falling back to standard model CPU offload.",
                    e,
                )
                self._pipeline.enable_model_cpu_offload()
            else:
                if gpu_mem_gb > 18:
                    # High VRAM: offload entire pipeline to CPU with automatic GPU placement.
                    self._pipeline.enable_model_cpu_offload()
                else:
                    # Low VRAM: per-layer offloading requires ~3–4GB of VRAM.
                    try:
                        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
                    except Exception as e:
                        log.warning(
                            "Failed to enable per-layer offload on Nunchaku Qwen-Image transformer: %s",
                            e,
                        )
                    exclude = getattr(self._pipeline, "_exclude_from_cpu_offload", None)
                    if isinstance(exclude, list):
                        exclude.append("transformer")
                    self._pipeline.enable_sequential_cpu_offload()

    async def _load_gguf_model(
        self,
        context: ProcessingContext,
        torch_dtype,
    ):
        """Load Qwen-Image model with GGUF quantization."""
        # Get the cached file path
        assert self.model.path is not None

        # Load the transformer with GGUF quantization
        transformer = await self.load_model(
            context=context,
            model_class=QwenImageTransformer2DModel,
            model_id=self.get_model_id(),
            path=self.model.path,
            torch_dtype=torch_dtype,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            config="Qwen/Qwen-Image",
            subfolder="transformer",
            device="cpu",  # Load on CPU first, then move to GPU
        )

        # Create the pipeline with the quantized transformer
        log.info("Creating Qwen-Image pipeline with quantized transformer...")

        self._pipeline = await self.load_model(
            context=context,
            model_class=DiffusionPipeline,
            model_id="Qwen/Qwen-Image",
            torch_dtype=torch_dtype,
            transformer=transformer,
            device="cpu",  # Load on CPU first, then move to GPU
        )

        # Apply memory optimizations after loading
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

        self._pipeline.enable_model_cpu_offload()

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # Handle CPU offload case
            if self.enable_cpu_offload:
                # When moving to CPU, force all components to CPU
                if device == "cpu":
                    try:
                        self._pipeline.to(device)
                    except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                        raise ValueError(
                            "VRAM out of memory while moving Qwen-Image pipeline to CPU. "
                            "Reduce image size/steps."
                        ) from e
                # For GPU devices, rely on the offload configuration set up at load time
                # (model or sequential CPU offload) instead of reconfiguring here.
            else:
                # Normal device movement without CPU offload
                try:
                    self._pipeline.to(device)
                except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                    raise ValueError(
                        "VRAM out of memory while moving Qwen-Image pipeline to device. "
                        "Enable 'CPU offload' in advanced node properties or reduce image size/steps."
                    ) from e

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

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)

    def required_inputs(self):
        """Return list of required inputs that must be connected."""
        return []  # No required inputs - IP adapter image is optional

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
    _pipeline: FluxControlPipeline | None = None

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "prompt",
            "control_image",
            "height",
            "width",
            "guidance_scale",
            "seed",
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Flux Control"

    def get_model_id(self) -> str:
        if self.model.repo_id:
            return self.model.repo_id
        return "black-forest-labs/FLUX.1-Depth-dev"

    def required_inputs(self):
        return ["control_image"]

    async def preload_model(self, context: ProcessingContext):
        hf_token = await context.get_secret("HF_TOKEN")
        if not hf_token:
            model_url = f"https://huggingface.co/{self.get_model_id()}"
            raise ValueError(
                f"Flux Control is a gated model, please set the HF_TOKEN in Nodetool settings and accept the terms of use for the model: {model_url}"
            )

        log.info(f"Loading FLUX Control pipeline from {self.get_model_id()}...")
        torch_dtype = _select_diffusion_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_class=FluxControlPipeline,
            model_id=self.get_model_id(),
            torch_dtype=torch_dtype,
            device="cpu",
        )

        # Apply CPU offload if enabled
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        if self._pipeline is not None and self.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # If CPU offload is enabled, we need to handle device movement differently
            if self.enable_cpu_offload:
                # When moving to CPU, disable CPU offload and move all components to CPU
                if device == "cpu":
                    self._pipeline.to(device)
                # When moving to GPU with CPU offload, re-enable CPU offload
                elif device in ["cuda", "mps"]:
                    self._pipeline.enable_model_cpu_offload()
            else:
                # Normal device movement without CPU offload
                try:
                    self._pipeline.to(device)
                except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                    raise ValueError(
                        "VRAM out of memory while moving Flux Control to device. "
                        "Enable 'CPU offload' in the advanced node properties or reduce image size."
                    ) from e

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = torch.Generator(device=context.device)
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        # Load and preprocess control image
        control_image = await context.image_to_pil(self.control_image)

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
            output = await self.run_pipeline_in_thread(**kwargs)  # type: ignore
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while running Flux Control. "
                "Enable 'CPU offload' in the advanced node properties (if available), "
                "or reduce image size/steps."
            ) from e
        image = output.images[0]

        return await context.image_from_pil(image)
