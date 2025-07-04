from enum import Enum
from huggingface_hub import try_to_load_from_cache
from nodetool.common.environment import Environment
from nodetool.metadata.types import HFTextToImage, HFImageToImage, HFLoraSD, HuggingFaceModel, ImageRef
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.nodes.huggingface.image_to_image import pipeline_progress_callback
from nodetool.nodes.huggingface.stable_diffusion_base import (
    StableDiffusionBaseNode,
    StableDiffusionXLBase,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress

import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import T5EncoderModel
from transformers.utils.quantization_config import BitsAndBytesConfig
from pydantic import Field
from nodetool.workflows.base_node import BaseNode

log = Environment.get_logger()

# class AuraFlow(BaseNode):
#     """
#     Generates images using the AuraFlow pipeline.
#     image, generation, AI, text-to-image

#     Use cases:
#     - Create unique images from text descriptions
#     - Generate illustrations for creative projects
#     - Produce visual content for digital media
#     """

#     prompt: str = Field(
#         default="A cat holding a sign that says hello world",
#         description="A text prompt describing the desired image.",
#     )
#     negative_prompt: str = Field(
#         default="", description="A text prompt describing what to avoid in the image."
#     )
#     guidance_scale: float = Field(
#         default=7.0, description="The guidance scale for the transformation.", ge=1.0
#     )
#     num_inference_steps: int = Field(
#         default=25, description="The number of denoising steps.", ge=1, le=100
#     )
#     width: int = Field(
#         default=768, description="The width of the generated image.", ge=128, le=1024
#     )
#     height: int = Field(
#         default=768, description="The height of the generated image.", ge=128, le=1024
#     )
#     seed: int = Field(
#         default=-1,
#         description="Seed for the random number generator. Use -1 for a random seed.",
#         ge=-1,
#     )

#     _pipeline: AuraFlowPipeline | None = None

#     async def preload_model(self, context: ProcessingContext):
#         self._pipeline = AuraFlowPipeline.from_pretrained(
#             "fal/AuraFlow", torch_dtype=torch.float16
#         )  # type: ignore

#     async def move_to_device(self, device: str):
#         pass

#     async def process(self, context: ProcessingContext) -> ImageRef:
#         if self._pipeline is None:
#             raise ValueError("Pipeline not initialized")

#         # Set up the generator for reproducibility if a seed is provided
#         generator = None
#         if self.seed != -1:
#             generator = torch.Generator(device=self._pipeline.device).manual_seed(
#                 self.seed
#             )

#         self._pipeline.enable_sequential_cpu_offload()

#         output = self._pipeline(
#             self.prompt,
#             negative_prompt=self.negative_prompt,
#             guidance_scale=self.guidance_scale,
#             num_inference_steps=self.num_inference_steps,
#             width=self.width,
#             height=self.height,
#             generator=generator,
#         )
#         image = output.images[0]  # type: ignore

#         return await context.image_from_pil(image)


# class Kandinsky2(BaseNode):
#     """
#     Generates images using the Kandinsky 2.2 model from text prompts.
#     image, generation, AI, text-to-image

#     Use cases:
#     - Create high-quality images from text descriptions
#     - Generate detailed illustrations for creative projects
#     - Produce visual content for digital media and art
#     - Explore AI-generated imagery for concept development
#     """

#     @classmethod
#     def get_title(cls) -> str:
#         return "Kandinsky 2.2"

#     prompt: str = Field(
#         default="A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background.",
#         description="A text prompt describing the desired image.",
#     )
#     negative_prompt: str = Field(
#         default="", description="A text prompt describing what to avoid in the image."
#     )
#     num_inference_steps: int = Field(
#         default=50, description="The number of denoising steps.", ge=1, le=100
#     )
#     width: int = Field(

#         default=768, description="The width of the generated image.", ge=128, le=1024
#     )
#     height: int = Field(
#         default=768, description="The height of the generated image.", ge=128, le=1024
#     )
#     seed: int = Field(
#         default=-1,
#         description="Seed for the random number generator. Use -1 for a random seed.",
#         ge=-1,
#     )

#     _prior_pipeline: KandinskyV22PriorPipeline | None = None
#     _pipeline: KandinskyV22Pipeline | None = None

#     async def preload_model(self, context: ProcessingContext):
#         self._prior_pipeline = KandinskyV22PriorPipeline.from_pretrained(
#             "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
#         )  # type: ignore
#         self._pipeline = KandinskyV22Pipeline.from_pretrained(
#             "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
#         )  # type: ignore

#     async def process(self, context: ProcessingContext) -> ImageRef:
#         if self._prior_pipeline is None or self._pipeline is None:
#             raise ValueError("Pipelines not initialized")

#         # Set up the generator for reproducibility
#         generator = torch.Generator(device="cpu")
#         if self.seed != -1:
#             generator = generator.manual_seed(self.seed)

#         # Enable sequential CPU offload for memory efficiency
#         self._pipeline.enable_sequential_cpu_offload()

#         # Generate image embeddings
#         prior_output = self._prior_pipeline(
#             self.prompt, negative_prompt=self.negative_prompt, generator=generator
#         )
#         image_emb, negative_image_emb = prior_output.to_tuple()  # type: ignore

#         output = self._pipeline(
#             image_embeds=image_emb,
#             negative_image_embeds=negative_image_emb,
#             height=self.height,
#             width=self.width,
#             num_inference_steps=self.num_inference_steps,
#             generator=generator,
#             callback=progress_callback(self.id, self.num_inference_steps, context),
#             callback_steps=1,
#         )

#         image = output.images[0]  # type: ignore

#         return await context.image_from_pil(image)


# class PixArtAlpha(HuggingFacePipelineNode):
#     """
#     Generates images from text prompts using the PixArt-Alpha model.
#     image, generation, AI, text-to-image

#     Use cases:
#     - Create unique images from detailed text descriptions
#     - Generate concept art for creative projects
#     - Produce visual content for digital media and marketing
#     - Explore AI-generated imagery for artistic inspiration
#     """

#     prompt: str = Field(
#         default="An astronaut riding a green horse",
#         description="A text prompt describing the desired image.",
#     )
#     negative_prompt: str = Field(
#         default="",
#         description="A text prompt describing what to avoid in the image.",
#     )
#     num_inference_steps: int = Field(
#         default=50,
#         description="The number of denoising steps.",
#         ge=1,
#         le=100,
#     )
#     guidance_scale: float = Field(
#         default=7.5,
#         description="The scale for classifier-free guidance.",
#         ge=1.0,
#         le=20.0,
#     )
#     width: int = Field(
#         default=768,
#         description="The width of the generated image.",
#         ge=128,
#         le=1024,
#     )
#     height: int = Field(
#         default=768,
#         description="The height of the generated image.",
#         ge=128,
#         le=1024,
#     )
#     seed: int = Field(
#         default=-1,
#         description="Seed for the random number generator. Use -1 for a random seed.",
#         ge=-1,
#     )

#     _pipeline: PixArtAlphaPipeline | None = None

#     @classmethod
#     def get_recommended_models(cls) -> list[HFImageToImage]:
#         return [
#             HFImageToImage(
#                 repo_id="PixArt-alpha/PixArt-XL-2-1024-MS",
#             ),
#         ]

#     async def preload_model(self, context: ProcessingContext):
#         self._pipeline = await self.load_model(
#             context=context,
#             model_id="PixArt-alpha/PixArt-XL-2-1024-MS",
#             model_class=PixArtAlphaPipeline,
#             variant=None,
#         )

#     async def move_to_device(self, device: str):
#         if self._pipeline is not None:
#             self._pipeline.to(device)

#     async def process(self, context: ProcessingContext) -> ImageRef:
#         if self._pipeline is None:
#             raise ValueError("Pipeline not initialized")

#         # Set up the generator for reproducibility
#         generator = None
#         if self.seed != -1:
#             generator = torch.Generator(device="cpu").manual_seed(self.seed)

#         def callback(step: int, timestep: int, latents: torch.Tensor) -> None:
#             context.post_message(
#                 NodeProgress(
#                     node_id=self.id,
#                     progress=step,
#                     total=self.num_inference_steps,
#                 )
#             )

#         # Generate the image
#         output = self._pipeline(
#             prompt=self.prompt,
#             negative_prompt=self.negative_prompt,
#             num_inference_steps=self.num_inference_steps,
#             guidance_scale=self.guidance_scale,
#             width=self.width,
#             height=self.height,
#             generator=generator,
#             callback=callback,
#             callback_steps=1,
#         )

#         image = output.images[0]  # type: ignore

#         return await context.image_from_pil(image)


# class PixArtSigma(HuggingFacePipelineNode):
#     """
#     Generates images from text prompts using the PixArt-Sigma model.
#     image, generation, AI, text-to-image

#     Use cases:
#     - Create unique images from detailed text descriptions
#     - Generate concept art for creative projects
#     - Produce visual content for digital media and marketing
#     - Explore AI-generated imagery for artistic inspiration
#     """

#     prompt: str = Field(
#         default="An astronaut riding a green horse",
#         description="A text prompt describing the desired image.",
#     )
#     negative_prompt: str = Field(
#         default="",
#         description="A text prompt describing what to avoid in the image.",
#     )
#     num_inference_steps: int = Field(
#         default=50,
#         description="The number of denoising steps.",
#         ge=1,
#         le=100,
#     )
#     guidance_scale: float = Field(
#         default=7.5,
#         description="The scale for classifier-free guidance.",
#         ge=1.0,
#         le=20.0,
#     )
#     width: int = Field(
#         default=768,
#         description="The width of the generated image.",
#         ge=128,
#         le=1024,
#     )
#     height: int = Field(
#         default=768,
#         description="The height of the generated image.",
#         ge=128,
#         le=1024,
#     )
#     seed: int = Field(
#         default=-1,
#         description="Seed for the random number generator. Use -1 for a random seed.",
#         ge=-1,
#     )

#     _pipeline: PixArtAlphaPipeline | None = None

#     @classmethod
#     def get_recommended_models(cls) -> list[HFImageToImage]:
#         return [
#             HFImageToImage(
#                 repo_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
#             ),
#         ]

#     async def preload_model(self, context: ProcessingContext):
#         self._pipeline = await self.load_model(
#             context=context,
#             model_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
#             model_class=PixArtAlphaPipeline,
#             variant=None,
#         )

#     async def process(self, context: ProcessingContext) -> ImageRef:
#         if self._pipeline is None:
#             raise ValueError("Pipeline not initialized")

#         # Set up the generator for reproducibility
#         generator = None
#         if self.seed != -1:
#             generator = torch.Generator(device="cpu").manual_seed(self.seed)

#         def callback(step: int, timestep: int, latents: torch.FloatTensor) -> None:
#             context.post_message(
#                 NodeProgress(
#                     node_id=self.id,
#                     progress=step,
#                     total=self.num_inference_steps,
#                 )
#             )

#         # Generate the image
#         output = self._pipeline(
#             prompt=self.prompt,
#             negative_prompt=self.negative_prompt,
#             num_inference_steps=self.num_inference_steps,
#             guidance_scale=self.guidance_scale,
#             width=self.width,
#             height=self.height,
#             generator=generator,
#             callback=callback,  # type: ignore
#             callback_steps=1,
#         )

#         image = output.images[0]  # type: ignore

#         return await context.image_from_pil(image)


# class Kandinsky3(HuggingFacePipelineNode):
#     """
#     Generates images using the Kandinsky-3 model from text prompts.
#     image, generation, AI, text-to-image

#     Use cases:
#     - Create detailed images from text descriptions
#     - Generate unique illustrations for creative projects
#     - Produce visual content for digital media and art
#     - Explore AI-generated imagery for concept development
#     """

#     prompt: str = Field(
#         default="A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background.",
#         description="A text prompt describing the desired image.",
#     )
#     num_inference_steps: int = Field(
#         default=25, description="The number of denoising steps.", ge=1, le=100
#     )
#     width: int = Field(
#         default=1024, description="The width of the generated image.", ge=64, le=2048
#     )
#     height: int = Field(
#         default=1024, description="The height of the generated image.", ge=64, le=2048
#     )
#     seed: int = Field(
#         default=0,
#         description="Seed for the random number generator. Use -1 for a random seed.",
#         ge=-1,
#     )

#     _pipeline: AutoPipelineForText2Image | None = None

#     @classmethod
#     def get_recommended_models(cls) -> list[HuggingFaceModel]:
#         return [
#             HuggingFaceModel(
#                 repo_id="kandinsky-community/kandinsky-3",
#             ),
#         ]

#     @classmethod
#     def get_title(cls) -> str:
#         return "Kandinsky 3"

#     def get_model_id(self):
#         return "kandinsky-community/kandinsky-3"

#     async def preload_model(self, context: ProcessingContext):
#         self._pipeline = await self.load_model(
#             context=context,
#             model_id="kandinsky-community/kandinsky-3",
#             model_class=AutoPipelineForText2Image,
#         )

#     async def move_to_device(self, device: str):
#         # Commented out as in the original class
#         # if self._pipeline is not None:
#         #     self._pipeline.to(device)
#         pass

#     async def process(self, context: ProcessingContext) -> ImageRef:
#         if self._pipeline is None:
#             raise ValueError("Pipeline not initialized")

#         # Set up the generator for reproducibility
#         generator = None
#         if self.seed != -1:
#             generator = torch.Generator(device="cpu").manual_seed(self.seed)

#         self._pipeline.enable_sequential_cpu_offload()

#         # Generate the image
#         output = self._pipeline(
#             prompt=self.prompt,
#             num_inference_steps=self.num_inference_steps,
#             generator=generator,
#             width=self.width,
#             height=self.height,
#             callback=progress_callback(self.id, self.num_inference_steps, context),
#             callback_steps=1,
#         )  # type: ignore

#         image = output.images[0]  # type: ignore

#         return await context.image_from_pil(image)


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

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            config="Lykon/DreamShaper",
        )
        assert self._pipeline is not None
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> ImageRef:
        return await self.run_pipeline(context, width=self.width, height=self.height)


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

    _pipeline: StableDiffusionXLPipeline | None = None

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
            model_class=StableDiffusionXLPipeline,
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

class AutoPipelineText2Image(HuggingFacePipelineNode):
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
        return "Auto Pipeline Text2Image"

    def get_model_id(self) -> str:
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.get_model_id(),
            model_class=AutoPipelineForText2Image,
            torch_dtype=torch.float16,
            variant=self.model.variant if self.model.variant != ModelVariant.DEFAULT else None,
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
            callback_on_step_end=pipeline_progress_callback(self.id, self.num_inference_steps, context),
            callback_steps=1,
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


# class FluxSchnell(HuggingFacePipelineNode):
