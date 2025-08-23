from enum import Enum
from nodetool.common.environment import Environment
from nodetool.metadata.types import HFTextToImage, HFImageToImage, HFLoraSD, HuggingFaceModel, ImageRef, TorchTensor
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.nodes.huggingface.image_to_image import pipeline_progress_callback
from nodetool.nodes.huggingface.stable_diffusion_base import (
    StableDiffusionBaseNode,
    StableDiffusionXLBase,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, NodeUpdate

import torch
from diffusers.pipelines.pag.pipeline_pag_sd import StableDiffusionPAGPipeline
from diffusers.pipelines.pag.pipeline_pag_sd_xl import StableDiffusionXLPAGPipeline
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.kolors.pipeline_kolors import KolorsPipeline
from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import HunyuanDiTPipeline
from diffusers.pipelines.lumina.pipeline_lumina import LuminaPipeline
from diffusers.pipelines.chroma.pipeline_chroma import ChromaPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from transformers import T5EncoderModel
from transformers.utils.quantization_config import BitsAndBytesConfig
from pydantic import Field
from nodetool.workflows.base_node import BaseNode

log = Environment.get_logger()

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
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["width", "height"]

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
            variant=self.variant.value if self.variant != ModelVariant.DEFAULT else None,
        ) 

    async def process(self, context: ProcessingContext) -> HFTextToImage:
        return HFTextToImage(
            repo_id=self.repo_id,
            variant=self.variant.value if self.variant != ModelVariant.DEFAULT else None,
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
            variant=self.model.variant if self.model.variant != ModelVariant.DEFAULT else None,
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

        # Generate the image
        output = self._pipeline(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            width=self.width,
            height=self.height,
            generator=generator,
            pag_scale=self.pag_scale,
            callback_on_step_end=pipeline_progress_callback(self.id, self.num_inference_steps, context),
        )  # type: ignore

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)


class FluxVariant(Enum):
    SCHNELL = "schnell"
    DEV = "dev"
    FILL_DEV = "fill-dev"
    CANNY_DEV = "canny-dev"
    DEPTH_DEV = "depth-dev"


class QuantizationMethod(Enum):
    NONE = "none"
    BITSANDBYTES_8BIT = "bitsandbytes-8bit"
    BITSANDBYTES_4BIT = "bitsandbytes-4bit"


class FluxText2Image(HuggingFacePipelineNode):
    """
    Generates images using FLUX models with quantization support for memory efficiency.
    image, generation, AI, text-to-image, flux, quantization

    Use cases:
    - High-quality image generation with FLUX models
    - Memory-efficient generation using quantization
    - Fast generation with FLUX.1-schnell
    - High-fidelity generation with FLUX.1-dev
    - Controlled generation with Fill, Canny, or Depth variants
    """

    variant: FluxVariant = Field(
        default=FluxVariant.SCHNELL,
        description="The FLUX model variant to use.",
    )
    quantization: QuantizationMethod = Field(
        default=QuantizationMethod.NONE,
        description="Quantization method to reduce memory usage.",
    )
    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="A text prompt describing the desired image.",
    )
    guidance_scale: float = Field(
        default=0.0,
        description="The scale for classifier-free guidance. Use 0.0 for schnell, 3.5 for dev.",
        ge=0.0,
        le=30.0,
    )
    width: int = Field(
        default=1360, 
        description="The width of the generated image.", 
        ge=64, 
        le=2048
    )
    height: int = Field(
        default=768, 
        description="The height of the generated image.", 
        ge=64, 
        le=2048
    )
    num_inference_steps: int = Field(
        default=4, 
        description="The number of denoising steps. Use 4 for schnell, 50 for dev.", 
        ge=1, 
        le=100
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
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="black-forest-labs/FLUX.1-schnell",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HuggingFaceModel(
                repo_id="black-forest-labs/FLUX.1-dev",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HuggingFaceModel(
                repo_id="black-forest-labs/FLUX.1-Fill-dev",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HuggingFaceModel(
                repo_id="black-forest-labs/FLUX.1-Canny-dev",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HuggingFaceModel(
                repo_id="black-forest-labs/FLUX.1-Depth-dev",
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
        return "FLUX Text2Image"

    def get_model_id(self) -> str:
        model_mapping = {
            FluxVariant.SCHNELL: "black-forest-labs/FLUX.1-schnell",
            FluxVariant.DEV: "black-forest-labs/FLUX.1-dev",
            FluxVariant.FILL_DEV: "black-forest-labs/FLUX.1-Fill-dev",
            FluxVariant.CANNY_DEV: "black-forest-labs/FLUX.1-Canny-dev",
            FluxVariant.DEPTH_DEV: "black-forest-labs/FLUX.1-Depth-dev",
        }
        return model_mapping[self.variant]

    def _get_text_encoder_quantization_config(self):
        """Get quantization config for text encoder (uses transformers BitsAndBytesConfig)."""
        if self.quantization == QuantizationMethod.BITSANDBYTES_8BIT:
            try:
                return BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                raise ImportError("bitsandbytes is required for quantization. Install with: pip install bitsandbytes")
        
        elif self.quantization == QuantizationMethod.BITSANDBYTES_4BIT:
            try:
                return BitsAndBytesConfig(load_in_4bit=True)
            except ImportError:
                raise ImportError("bitsandbytes is required for quantization. Install with: pip install bitsandbytes")
        
        return None
    
    def _get_transformer_quantization_config(self):
        """Get quantization config for transformer (uses diffusers quantization)."""
        if self.quantization == QuantizationMethod.BITSANDBYTES_8BIT:
            try:
                # For diffusers transformer, we'll use the same config but it may need to be handled differently
                return BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                raise ImportError("bitsandbytes is required for quantization. Install with: pip install bitsandbytes")
        
        elif self.quantization == QuantizationMethod.BITSANDBYTES_4BIT:
            try:
                return BitsAndBytesConfig(load_in_4bit=True)
            except ImportError:
                raise ImportError("bitsandbytes is required for quantization. Install with: pip install bitsandbytes")
        
        return None

    async def preload_model(self, context: ProcessingContext):
        # Determine torch dtype based on variant
        torch_dtype = torch.bfloat16 if self.variant in [FluxVariant.SCHNELL, FluxVariant.DEV] else torch.float16
        
        # Load components separately if quantization is enabled
        if self.quantization != QuantizationMethod.NONE:
            await self._load_quantized_components(context, torch_dtype)
        else:
            # Load the full pipeline normally
            self._pipeline = await self.load_model(
                context=context,
                model_id=self.get_model_id(),
                model_class=FluxPipeline,
                torch_dtype=torch_dtype,
                variant=None,
                device="cpu",
            )
    
    async def _load_quantized_components(self, context: ProcessingContext, torch_dtype):
        """Load text encoder and transformer separately with quantization."""
        model_id = self.get_model_id()
        
        # Load text encoder with quantization
        text_encoder_quant_config = self._get_text_encoder_quantization_config()
        text_encoder = None
        if text_encoder_quant_config:
            text_encoder = T5EncoderModel.from_pretrained(
                model_id,
                subfolder="text_encoder_2",
                quantization_config=text_encoder_quant_config,
                torch_dtype=torch_dtype,
            )
        
        # Load transformer with quantization  
        transformer_quant_config = self._get_transformer_quantization_config()
        transformer = None
        if transformer_quant_config:
            transformer = FluxTransformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=transformer_quant_config,
                torch_dtype=torch_dtype,
            )
        
        # Load the full pipeline with quantized components
        self._pipeline = FluxPipeline.from_pretrained(
            model_id,
            text_encoder_2=text_encoder,
            transformer=transformer,
            torch_dtype=torch_dtype,
            device_map="balanced",
        ) # type: ignore

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            if not self.enable_cpu_offload:
                self._pipeline.to(device)
            
            # Apply memory optimization settings
            if self.enable_cpu_offload:
                self._pipeline.enable_sequential_cpu_offload()
            
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

        # Adjust parameters based on variant
        guidance_scale = self.guidance_scale
        num_inference_steps = self.num_inference_steps
        max_sequence_length = self.max_sequence_length
        
        if self.variant == FluxVariant.SCHNELL:
            # For schnell, guidance_scale should be 0 and max_sequence_length <= 256
            if guidance_scale != 0.0:
                log.warning("For FLUX.1-schnell, guidance_scale should be 0.0. Adjusting automatically.")
                guidance_scale = 0.0
            if max_sequence_length > 256:
                log.warning("For FLUX.1-schnell, max_sequence_length should be <= 256. Adjusting to 256.")
                max_sequence_length = 256
            if num_inference_steps > 10:
                log.warning("For FLUX.1-schnell, fewer inference steps (4-8) are recommended for optimal performance.")

        def progress_callback(step: int, timestep: int, callback_kwargs: dict) -> None:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=num_inference_steps,
                )
            )

        # Generate the image
        output = self._pipeline(
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
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
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

        # Generate the image
        output = self._pipeline(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            height=self.height,
            width=self.width,
            generator=generator,
            max_sequence_length=self.max_sequence_length,
            callback_on_step_end=pipeline_progress_callback(self.id, self.num_inference_steps, context), # type: ignore
            callback_on_step_end_tensor_inputs=["latents"],
        )

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)


class HunyuanDiT(HuggingFacePipelineNode):
    """
    Generates images from text prompts using Hunyuan-DiT, a powerful multi-resolution diffusion transformer.
    image, generation, AI, text-to-image, chinese, english, diffusion, transformer

    Use cases:
    - Generate high-quality images from Chinese and English text descriptions
    - Create images with fine-grained language understanding
    - Produce multi-resolution images with optimal aspect ratios
    - Generate images with both Chinese and English text support
    - Create detailed images with strong semantic accuracy
    """

    prompt: str = Field(
        default="一个宇航员在骑马",
        description="A text prompt describing the desired image. Supports both Chinese and English.",
    )
    negative_prompt: str = Field(
        default="",
        description="A text prompt describing what to avoid in the image.",
    )
    guidance_scale: float = Field(
        default=7.5,
        description="The scale for classifier-free guidance.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="The number of denoising steps.",
        ge=1,
        le=100,
    )
    width: int = Field(
        default=1024,
        description="The width of the generated image.",
        ge=512,
        le=2048,
    )
    height: int = Field(
        default=1024,
        description="The height of the generated image.",
        ge=512,
        le=2048,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )
    use_resolution_binning: bool = Field(
        default=True,
        description="Whether to use resolution binning. Maps input resolution to closest standard resolution.",
    )
    original_size: tuple[int, int] = Field(
        default=(1024, 1024),
        description="The original size of the image used to calculate time IDs.",
    )
    target_size: tuple[int, int] | None = Field(
        default=None,
        description="The target size of the image used to calculate time IDs. If None, uses (width, height).",
    )
    crops_coords_top_left: tuple[int, int] = Field(
        default=(0, 0),
        description="The top-left coordinates of the crop used to calculate time IDs.",
    )
    guidance_rescale: float = Field(
        default=0.0,
        description="Rescale the noise according to guidance_rescale.",
        ge=0.0,
        le=1.0,
    )
    enable_memory_optimization: bool = Field(
        default=True,
        description="Enable memory optimization with T5 encoder quantization.",
    )
    enable_forward_chunking: bool = Field(
        default=False,
        description="Enable forward chunking to reduce memory usage at the cost of inference speed.",
    )
    forward_chunk_size: int = Field(
        default=1,
        description="Chunk size for forward chunking.",
        ge=1,
        le=4,
    )

    _pipeline: HunyuanDiTPipeline | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="Tencent-Hunyuan/HunyuanDiT-Diffusers",
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
        return "Hunyuan-DiT"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "width", "height", "use_resolution_binning"]

    def get_model_id(self) -> str:
        return "Tencent-Hunyuan/HunyuanDiT-Diffusers"

    def _get_supported_resolutions(self) -> list[tuple[int, int]]:
        """Get list of supported resolutions for resolution binning."""
        return [
            (1024, 1024), (1280, 1280), (1024, 768), (1152, 864), (1280, 960),
            (768, 1024), (864, 1152), (960, 1280), (1280, 768), (768, 1280)
        ]

    def _find_closest_resolution(self, width: int, height: int) -> tuple[int, int]:
        """Find the closest supported resolution."""
        if not self.use_resolution_binning:
            return (width, height)
        
        target_ratio = width / height
        best_resolution = (1024, 1024)
        best_ratio_diff = float('inf')
        
        for res_width, res_height in self._get_supported_resolutions():
            ratio = res_width / res_height
            ratio_diff = abs(ratio - target_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_resolution = (res_width, res_height)
        
        return best_resolution

    async def preload_model(self, context: ProcessingContext):
        # Load with memory optimization if enabled
        if self.enable_memory_optimization:
            # Load T5 encoder in 8-bit for memory efficiency
            try:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                
                self._pipeline = await self.load_model(
                    context=context,
                    model_id=self.get_model_id(),
                    model_class=HunyuanDiTPipeline,
                    torch_dtype=torch.float16,
                    text_encoder_2_quantization_config=quantization_config,
                )
            except ImportError:
                log.warning("bitsandbytes not available, loading without quantization")
                self._pipeline = await self.load_model(
                    context=context,
                    model_id=self.get_model_id(),
                    model_class=HunyuanDiTPipeline,
                    torch_dtype=torch.float16,
                )
        else:
            self._pipeline = await self.load_model(
                context=context,
                model_id=self.get_model_id(),
                model_class=HunyuanDiTPipeline,
                torch_dtype=torch.float16,
            )

        # Apply forward chunking if enabled
        if self._pipeline is not None and self.enable_forward_chunking:
            self._pipeline.transformer.enable_forward_chunking(
                chunk_size=self.forward_chunk_size, 
                dim=1
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

        # Determine final resolution
        final_width, final_height = self._find_closest_resolution(self.width, self.height)
        
        if self.use_resolution_binning and (final_width != self.width or final_height != self.height):
            log.info(f"Resolution binning: {self.width}x{self.height} -> {final_width}x{final_height}")

        # Set target size if not specified
        target_size = self.target_size or (final_width, final_height)

        def progress_callback(step: int, timestep: int, callback_kwargs: dict) -> None:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )

        # Generate the image
        output = self._pipeline(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            height=final_height,
            width=final_width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            original_size=self.original_size,
            target_size=target_size,
            crops_coords_top_left=self.crops_coords_top_left,
            guidance_rescale=self.guidance_rescale,
            use_resolution_binning=self.use_resolution_binning,
            callback_on_step_end=progress_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)


class LuminaT2X(HuggingFacePipelineNode):
    """
    Generates images from text prompts using Lumina-T2X, a Flow-based Large Diffusion Transformer.
    image, generation, AI, text-to-image, diffusion, transformer, flow, quantization

    Use cases:
    - Generate high-quality images with improved sampling efficiency
    - Create images with Next-DiT architecture and 3D RoPE
    - Produce images with better resolution extrapolation capabilities
    - Generate images with multilingual support using decoder-based LLMs
    - Create images with advanced frequency and time-aware scaling
    """

    prompt: str = Field(
        default="Upper body of a young woman in a Victorian-era outfit with brass goggles and leather straps. Background shows an industrial revolution cityscape with smoky skies and tall, metal structures",
        description="A text prompt describing the desired image.",
    )
    negative_prompt: str = Field(
        default="",
        description="A text prompt describing what to avoid in the image. For Lumina-T2X, this should typically be empty.",
    )
    guidance_scale: float = Field(
        default=4.0,
        description="The scale for classifier-free guidance.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=30,
        description="The number of denoising steps. Lumina-T2X uses fewer steps for efficient generation.",
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
    clean_caption: bool = Field(
        default=True,
        description="Whether to clean the caption before creating embeddings. Requires beautifulsoup4 and ftfy.",
    )
    max_sequence_length: int = Field(
        default=256,
        description="Maximum sequence length to use with the prompt.",
        ge=1,
        le=512,
    )
    scaling_watershed: float = Field(
        default=1.0,
        description="Scaling watershed parameter for improved generation quality.",
        ge=0.1,
        le=2.0,
    )
    proportional_attn: bool = Field(
        default=True,
        description="Whether to use proportional attention for better quality.",
    )
    enable_quantization: bool = Field(
        default=True,
        description="Enable quantization for memory efficiency.",
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

    _pipeline: LuminaPipeline | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="Alpha-VLLM/Lumina-Next-SFT-diffusers",
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
        return "Lumina-T2X"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "height", "width", "num_inference_steps"]

    def get_model_id(self) -> str:
        return "Alpha-VLLM/Lumina-Next-SFT-diffusers"

    def _get_quantization_config(self):
        """Get quantization config for memory efficiency."""
        if not self.enable_quantization:
            return None
        
        try:
            # Use 8-bit quantization for memory efficiency
            return BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            log.warning("bitsandbytes not available, loading without quantization")
            return None

    async def preload_model(self, context: ProcessingContext):
        # Determine dtype - Lumina-T2X works best with bfloat16
        torch_dtype = torch.bfloat16
        
        # Load with quantization if enabled
        quantization_config = self._get_quantization_config()
        
        if quantization_config:
            try:
                self._pipeline = await self.load_model(
                    context=context,
                    model_id=self.get_model_id(),
                    model_class=LuminaPipeline,
                    torch_dtype=torch_dtype,
                    quantization_config=quantization_config,
                )
            except Exception as e:
                log.warning(f"Failed to load with quantization: {e}. Loading without quantization.")
                self._pipeline = await self.load_model(
                    context=context,
                    model_id=self.get_model_id(),
                    model_class=LuminaPipeline,
                    torch_dtype=torch_dtype,
                )
        else:
            self._pipeline = await self.load_model(
                context=context,
                model_id=self.get_model_id(),
                model_class=LuminaPipeline,
                torch_dtype=torch_dtype,
            )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            if not self.enable_cpu_offload:
                self._pipeline.to(device)
            
            # Apply memory optimization settings
            if self.enable_cpu_offload:
                self._pipeline.enable_model_cpu_offload()
            
            if self.enable_vae_slicing:
                self._pipeline.vae.enable_slicing()
            
            if self.enable_vae_tiling:
                self._pipeline.vae.enable_tiling()

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        def progress_callback(step: int, timestep: int, callback_kwargs: dict) -> None:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )

        # Generate the image
        try:
            output = self._pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                height=self.height,
                width=self.width,
                generator=generator,
                clean_caption=self.clean_caption,
                max_sequence_length=self.max_sequence_length,
                scaling_watershed=self.scaling_watershed,
                proportional_attn=self.proportional_attn,
                callback_on_step_end=progress_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        except Exception as e:
            # Handle cases where clean_caption dependencies might be missing
            if "beautifulsoup4" in str(e) or "ftfy" in str(e):
                log.warning("Missing dependencies for clean_caption. Retrying with clean_caption=False")
                output = self._pipeline(
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    height=self.height,
                    width=self.width,
                    generator=generator,
                    clean_caption=False,
                    max_sequence_length=self.max_sequence_length,
                    scaling_watershed=self.scaling_watershed,
                    proportional_attn=self.proportional_attn,
                    callback_on_step_end=progress_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
            else:
                raise e

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)


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
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
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

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "negative_prompt", "height", "width"]

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
            if not self.enable_cpu_offload:
                self._pipeline.to(device)
            
            # Apply memory optimization settings
            if self.enable_cpu_offload:
                self._pipeline.enable_model_cpu_offload()
            
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
            "callback_on_step_end": pipeline_progress_callback(self.id, self.num_inference_steps, context), # type: ignore
            "callback_on_step_end_tensor_inputs": ["latents"],
        }

        # Add IP adapter image if provided
        if ip_adapter_image is not None:
            pipeline_kwargs["ip_adapter_image"] = ip_adapter_image

        output = self._pipeline(**pipeline_kwargs)

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)

    def required_inputs(self):
        """Return list of required inputs that must be connected."""
        return []  # No required inputs - IP adapter image is optional


class QuantoFlux(HuggingFacePipelineNode):
    """
    Generates images using FLUX models with Optimum Quanto FP8 quantization for extreme memory efficiency.
    image, generation, AI, text-to-image, flux, quantization, fp8, quanto

    Use cases:
    - Ultra memory-efficient FLUX image generation using FP8 quantization
    - High-quality image generation on lower-end hardware
    - Faster inference with reduced memory footprint
    - Professional image generation with optimized resource usage
    """

    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="A text prompt describing the desired image.",
    )
    guidance_scale: float = Field(
        default=3.5,
        description="The scale for classifier-free guidance.",
        ge=0.0,
        le=30.0,
    )
    width: int = Field(
        default=1024, 
        description="The width of the generated image.", 
        ge=64, 
        le=2048
    )
    height: int = Field(
        default=1024, 
        description="The height of the generated image.", 
        ge=64, 
        le=2048
    )
    num_inference_steps: int = Field(
        default=20, 
        description="The number of denoising steps.", 
        ge=1, 
        le=100
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum sequence length for the prompt.",
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
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="Kijai/flux-fp8",
                path="flux1-dev-fp8.safetensors",
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Quanto FLUX FP8"

    def get_model_id(self) -> str:
        return "black-forest-labs/FLUX.1-dev"


    async def preload_model(self, context: ProcessingContext):
        from optimum.quanto import freeze, qfloat8, quantize
        
        dtype = torch.bfloat16
        
        # Load and quantize the transformer from single file
        transformer = await self.load_model(
            context,
            FluxTransformer2DModel,
            model_id="Kijai/flux-fp8",
            path="flux1-dev-fp8.safetensors",
            torch_dtype=dtype,
        )
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        
        # Load and quantize the text encoder
        log.info("Loading and quantizing text encoder with FP8...")
        context.post_message(
            NodeUpdate(
                node_id=self.id,
                node_name=self.get_title(),
                status="Loading and quantizing text encoder with FP8...",
            )
        )
        text_encoder_2 = await self.load_model(
            context,
            T5EncoderModel,
            model_id=self.get_model_id(),
            subfolder="text_encoder_2",
            torch_dtype=dtype,
        )
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        
        # Load the base pipeline without transformer and text_encoder_2
        log.info("Loading base pipeline...")
        pipe = FluxPipeline.from_pretrained(self.get_model_id(), transformer=None, text_encoder_2=None, torch_dtype=dtype)
        pipe.transformer = transformer
        pipe.text_encoder_2 = text_encoder_2

        pipe.enable_model_cpu_offload()
        log.info("QuantoFlux pipeline loaded successfully with FP8 quantization")

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            if self.enable_cpu_offload:
                self._pipeline.enable_model_cpu_offload()
            else:
                self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # Generate the image
        output = self._pipeline(
            prompt=self.prompt,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            generator=generator,
            callback_on_step_end=pipeline_progress_callback(self.id, self.num_inference_steps, context), # type: ignore
            callback_on_step_end_tensor_inputs=["latents"],
        )

        image = output.images[0]  # type: ignore

        return await context.image_from_pil(image)

