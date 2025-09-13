from enum import Enum
import re
from typing import Any
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from nodetool.workflows.types import NodeProgress
import numpy as np
import torch
from RealESRGAN import RealESRGAN
from huggingface_hub import try_to_load_from_cache
from nodetool.metadata.types import (
    HFControlNet,
    HFImageToImage,
    HFRealESRGAN,
    HFStableDiffusionUpscale,
    HFVAE,
    TorchTensor,
    HuggingFaceModel,
    ImageRef,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.nodes.huggingface.stable_diffusion_base import (
    HF_CONTROLNET_MODELS,
    StableDiffusionBaseNode,
    StableDiffusionXLBase,
)
from nodetool.nodes.huggingface.huggingface_node import progress_callback
from nodetool.workflows.processing_context import ProcessingContext

from diffusers.pipelines.omnigen.pipeline_omnigen import OmniGenPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.pag.pipeline_pag_controlnet_sd import (
    StableDiffusionControlNetPAGPipeline,
)
from diffusers.pipelines.pag.pipeline_pag_controlnet_sd_xl_img2img import (
    StableDiffusionXLControlNetPAGImg2ImgPipeline,
)
from diffusers.pipelines.pag.pipeline_pag_controlnet_sd_inpaint import (
    StableDiffusionControlNetPAGInpaintPipeline,
)
from diffusers.pipelines.pag.pipeline_pag_sd_img2img import (
    StableDiffusionPAGImg2ImgPipeline,
)
from diffusers.pipelines.pag.pipeline_pag_sd_inpaint import (
    StableDiffusionPAGInpaintPipeline,
)
from diffusers.pipelines.pag.pipeline_pag_controlnet_sd_xl import (
    StableDiffusionXLControlNetPAGPipeline,
)
from diffusers.pipelines.pag.pipeline_pag_sd_xl_img2img import (
    StableDiffusionXLPAGImg2ImgPipeline,
)
from diffusers.pipelines.pag.pipeline_pag_sd_xl_inpaint import (
    StableDiffusionXLPAGInpaintPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import (
    StableDiffusionControlNetImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import (
    StableDiffusionUpscalePipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_latent_upscale import (
    StableDiffusionLatentUpscalePipeline,
)
from pydantic import Field


class BaseImageToImage(HuggingFacePipelineNode):
    """
    Base class for image-to-image transformation tasks.
    image, transformation, generation, huggingface
    """

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not BaseImageToImage

    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The input image to transform",
    )
    prompt: str = Field(
        default="",
        title="Prompt",
        description="The text prompt to guide the image transformation (if applicable)",
    )

    def required_inputs(self):
        return ["image"]

    def get_model_id(self):
        raise NotImplementedError("Subclass must implement abstract method")

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context, "image-to-image", self.get_model_id(), device=context.device
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> ImageRef:
        image = await context.image_to_pil(self.image)
        result = self._pipeline(image, prompt=self.prompt)  # type: ignore
        return await context.image_from_pil(result)  # type: ignore


class RealESRGANNode(BaseNode):
    """
    Performs image super-resolution using the RealESRGAN model.
    image, super-resolution, enhancement, huggingface

    Use cases:
    - Enhance low-resolution images
    - Improve image quality for printing or display
    - Upscale images for better detail
    """

    """
    Performs image super-resolution using the RealESRGAN model.
    image, super-resolution, enhancement, huggingface

    Use cases:
    - Enhance low-resolution images
    - Improve image quality for printing or display
    - Upscale images for better detail
    """

    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The input image to transform",
    )
    model: HFRealESRGAN = Field(
        default=HFRealESRGAN(),
        title="RealESRGAN Model",
        description="The RealESRGAN model to use for image super-resolution",
    )
    _model: RealESRGAN | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFRealESRGAN]:
        return [
            HFRealESRGAN(
                repo_id="ai-forever/Real-ESRGAN",
                path="RealESRGAN_x2.pth",
            ),
            HFRealESRGAN(
                repo_id="ai-forever/Real-ESRGAN",
                path="RealESRGAN_x4.pth",
            ),
            HFRealESRGAN(
                repo_id="ai-forever/Real-ESRGAN",
                path="RealESRGAN_x8.pth",
            ),
            HFRealESRGAN(
                repo_id="ximso/RealESRGAN_x4plus_anime_6B",
                path="RealESRGAN_x4plus_anime_6B.pth",
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        assert self.model.path is not None, "Model is not set"

        model_path = try_to_load_from_cache(self.model.repo_id, self.model.path)

        if model_path is None:
            raise ValueError("Download the model first from RECOMMENDED_MODELS above")

        # parse scale from model path using regex, e.g. *x2*.pth -> 2
        match = re.search(r"x(\d+).*\.pth", self.model.path)
        if match is None:
            raise ValueError("Invalid model path, should be in the format *x2*.pth")
        scale = int(match.group(1))

        self._model = RealESRGAN(context.device, scale=scale)
        self._model.load_weights(model_path)

    async def move_to_device(self, device: str):
        if self._model is not None:
            self._model.device = device
            self._model.model.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        assert self._model is not None, "Model not initialized"
        assert self.image is not None, "Image not set"
        image = await context.image_to_pil(self.image)
        sr_image = self._model.predict(image)
        return await context.image_from_pil(sr_image)


class Swin2SR(BaseImageToImage):
    """
    Performs image super-resolution using the Swin2SR model.
    image, super-resolution, enhancement, huggingface

    Use cases:
    - Enhance low-resolution images
    - Improve image quality for printing or display
    - Upscale images for better detail
    """

    model: HFImageToImage = Field(
        default=HFImageToImage(),
        title="Model ID on Huggingface",
        description="The model ID to use for image super-resolution",
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFImageToImage]:
        return [
            HFImageToImage(
                repo_id="caidas/swin2SR-classical-sr-x2-64",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFImageToImage(
                repo_id="caidas/swin2SR-classical-sr-x4-48",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFImageToImage(
                repo_id="caidas/swin2SR-lightweight-sr-x2-64",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HFImageToImage(
                repo_id="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Swin2SR"

    def get_model_id(self):
        return self.model.repo_id


class ModelVariant(Enum):
    DEFAULT = "default"
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"


class LoadImageToImageModel(HuggingFacePipelineNode):
    """
    Load HuggingFace model for image-to-image generation from a repo_id.

    Use cases:
    - Loads a pipeline directly from a repo_id
    - Used for ImageToImage node
    """

    repo_id: str = Field(
        default="",
        description="The repository ID of the model to use for image-to-image generation.",
    )

    variant: ModelVariant = Field(
        default=ModelVariant.FP16,
        description="The variant of the model to use for image-to-image generation.",
    )

    async def preload_model(self, context: ProcessingContext):
        await self.load_model(
            context=context,
            model_id=self.repo_id,
            model_class=AutoPipelineForImage2Image,
            torch_dtype=(
                torch.float16 if self.variant == ModelVariant.FP16 else torch.float32
            ),
            use_safetensors=True,
            variant=(
                self.variant.value if self.variant != ModelVariant.DEFAULT else None
            ),
        )

    async def process(self, context: ProcessingContext) -> HFImageToImage:
        return HFImageToImage(
            repo_id=self.repo_id,
            variant=(
                self.variant.value if self.variant != ModelVariant.DEFAULT else None
            ),
        )


def pipeline_progress_callback(
    node_id: str, total_steps: int, context: ProcessingContext
):
    def callback(
        pipeline: DiffusionPipeline, step: int, timestep: int, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        context.post_message(
            NodeProgress(
                node_id=node_id,
                progress=step,
                total=total_steps,
            )
        )
        return kwargs

    return callback


class ImageToImage(HuggingFacePipelineNode):
    """
    Transforms existing images based on text prompts using AutoPipeline for Image-to-Image.
    This node automatically detects the appropriate pipeline class based on the model used.
    image, generation, image-to-image, autopipeline

    Use cases:
    - Transform existing images with any compatible model (Stable Diffusion, SDXL, Kandinsky, etc.)
    - Apply specific styles or concepts to photographs or artwork
    - Modify existing images based on text descriptions
    - Create variations of existing visual content with automatic pipeline selection
    """

    model: HFImageToImage = Field(
        default=HFImageToImage(),
        description="The HuggingFace model to use for image-to-image generation.",
    )
    prompt: str = Field(
        default="A beautiful landscape with mountains and a lake at sunset",
        description="Text prompt describing the desired image transformation.",
    )
    negative_prompt: str = Field(
        default="",
        description="Text prompt describing what should not appear in the generated image.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The input image to transform",
    )
    strength: float = Field(
        default=0.8,
        description="Strength of the transformation. Higher values allow for more deviation from the original image.",
        ge=0.0,
        le=1.0,
    )
    num_inference_steps: int = Field(
        default=25,
        description="Number of denoising steps.",
        ge=1,
        le=100,
    )
    guidance_scale: float = Field(
        default=7.5,
        description="Guidance scale for generation. Higher values follow the prompt more closely.",
        ge=1.0,
        le=20.0,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )

    _pipeline: AutoPipelineForImage2Image | None = None

    @classmethod
    def get_basic_fields(cls):
        return ["model", "image", "prompt", "negative_prompt", "strength"]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls) -> str:
        return "Image to Image"

    def get_model_id(self) -> str:
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.model.repo_id,
            path=self.model.path,
            model_class=AutoPipelineForImage2Image,
            torch_dtype=(
                torch.float16
                if self.model.variant == ModelVariant.FP16
                else torch.float32
            ),
            use_safetensors=True,
            variant=self.model.variant,
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        input_image = await context.image_to_pil(self.image)

        # Prepare kwargs for the pipeline
        kwargs = {
            "prompt": self.prompt,
            "image": input_image,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "generator": generator,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
        }

        # Add negative prompt if provided
        if self.negative_prompt:
            kwargs["negative_prompt"] = self.negative_prompt

        output = self._pipeline(**kwargs)  # type: ignore
        image = output.images[0]

        return await context.image_from_pil(image)


class Inpaint(HuggingFacePipelineNode):
    """
    Performs inpainting on images using AutoPipeline for Inpainting.
    This node automatically detects the appropriate pipeline class based on the model used.
    image, inpainting, autopipeline, stable-diffusion, SDXL, kandinsky

    Use cases:
    - Remove unwanted objects from images with any compatible model
    - Fill in missing parts of images using various diffusion models
    - Modify specific areas of images while preserving the rest
    - Automatic pipeline selection for different model architectures
    """

    model: HFImageToImage = Field(
        default=HFImageToImage(),
        description="The HuggingFace model to use for inpainting.",
    )
    prompt: str = Field(
        default="",
        description="Text prompt describing what should be generated in the masked area.",
    )
    negative_prompt: str = Field(
        default="",
        description="Text prompt describing what should not appear in the generated content.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The input image to inpaint",
    )
    mask_image: ImageRef = Field(
        default=ImageRef(),
        title="Mask Image",
        description="The mask image indicating areas to be inpainted (white areas will be inpainted)",
    )
    num_inference_steps: int = Field(
        default=25,
        description="Number of denoising steps.",
        ge=1,
        le=100,
    )
    guidance_scale: float = Field(
        default=7.5,
        description="Guidance scale for generation. Higher values follow the prompt more closely.",
        ge=1.0,
        le=20.0,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )

    _pipeline: AutoPipelineForInpainting | None = None

    @classmethod
    def get_basic_fields(cls):
        return ["model", "image", "mask_image", "prompt", "negative_prompt"]

    def required_inputs(self):
        return ["image", "mask_image"]

    @classmethod
    def get_title(cls) -> str:
        return "AutoPipeline Inpainting"

    def get_model_id(self) -> str:
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.model.repo_id,
            path=self.model.path,
            model_class=AutoPipelineForInpainting,
            torch_dtype=(
                torch.float16
                if self.model.variant == ModelVariant.FP16
                else (
                    torch.bfloat16
                    if self.model.variant == ModelVariant.BF16
                    else torch.float32
                )
            ),
            use_safetensors=True,
            variant=self.model.variant,
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        input_image = await context.image_to_pil(self.image)
        mask_image = await context.image_to_pil(self.mask_image)

        # Prepare kwargs for the pipeline
        kwargs = {
            "prompt": self.prompt,
            "image": input_image,
            "mask_image": mask_image,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "generator": generator,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
        }

        # Add negative prompt if provided
        if self.negative_prompt:
            kwargs["negative_prompt"] = self.negative_prompt

        output = self._pipeline(**kwargs)  # type: ignore

        image = output.images[0]

        return await context.image_from_pil(image)


class StableDiffusionControlNetNode(StableDiffusionBaseNode):
    """
    Generates images using Stable Diffusion with ControlNet guidance.
    image, generation, text-to-image, controlnet, SD

    Use cases:
    - Generate images with precise control over composition and structure
    - Create variations of existing images while maintaining specific features
    - Artistic image generation with guided outputs
    """

    controlnet: HFControlNet = Field(
        default=HFControlNet(),
        description="The ControlNet model to use for guidance.",
    )
    control_image: ImageRef = Field(
        default=ImageRef(),
        description="The control image to guide the generation process.",
    )
    controlnet_conditioning_scale: float = Field(
        default=1.0,
        description="The scale for ControlNet conditioning.",
        ge=0.0,
        le=2.0,
    )

    _pipeline: StableDiffusionControlNetPAGPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + [
            "controlnet",
            "control_image",
            "controlnet_conditioning_scale",
        ]

    @classmethod
    def get_recommended_models(cls):
        return HF_CONTROLNET_MODELS + super().get_recommended_models()

    def required_inputs(self):
        return ["control_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion ControlNet"

    @classmethod
    def return_type(cls):
        return {
            "image": ImageRef,
            "latent": TorchTensor,
        }

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.controlnet.to(device)
            self._pipeline.unet.to(device)
            self._pipeline.vae.to(device)
            self._pipeline.text_encoder.to(device)

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        # Use float32 for MPS compatibility with controlnet models
        controlnet_dtype = torch.float32 if context.device == "mps" else torch.float16
        controlnet_variant = None if context.device == "mps" else "fp16"
        controlnet = await self.load_model(
            context=context,
            model_class=ControlNetModel,
            model_id=self.controlnet.repo_id,
            torch_dtype=controlnet_dtype,
            variant=controlnet_variant,
        )
        # Align pipeline dtype with controlnet dtype to avoid mismatches
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionControlNetPAGPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            controlnet=controlnet,
            config="Lykon/DreamShaper",  # workaround for missing SD15 repo
            torch_dtype=controlnet_dtype,
        )
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_image = await context.image_to_pil(self.control_image)
        return await self.run_pipeline(
            context,
            image=control_image,
            width=control_image.width,
            height=control_image.height,
            controlnet_conditioning_scale=float(self.controlnet_conditioning_scale),
        )


class StableDiffusionImg2ImgNode(StableDiffusionBaseNode):
    """
    Transforms existing images based on text prompts using Stable Diffusion.
    image, generation, image-to-image, SD, img2img, style-transfer, ipadapter

    Use cases:
    - Modifying existing images to fit a specific style or theme
    - Enhancing or altering photographs
    - Creating variations of existing artwork
    - Applying text-guided edits to images
    """

    init_image: ImageRef = Field(
        default=ImageRef(),
        description="The initial image for Image-to-Image generation.",
    )
    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Strength for Image-to-Image generation. Higher values allow for more deviation from the original image.",
    )
    variant: ModelVariant = Field(
        default=ModelVariant.FP16,
        description="The variant of the model to use for Image-to-Image generation.",
    )
    _pipeline: StableDiffusionPAGImg2ImgPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["init_image", "strength"]

    def required_inputs(self):
        return ["init_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion (Img2Img)"

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
            model_class=StableDiffusionPAGImg2ImgPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            safety_checker=None,
            config="Lykon/DreamShaper",
            torch_dtype=(
                torch.float16 if self.variant == ModelVariant.FP16 else torch.float32
            ),
        )
        assert self._pipeline is not None
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> ImageRef:
        init_image = await context.image_to_pil(self.init_image)
        return await self.run_pipeline(
            context,
            image=init_image,
            strength=self.strength,
        )


class StableDiffusionControlNetModel(str, Enum):
    INPAINT = "lllyasviel/control_v11p_sd15_inpaint"


class StableDiffusionControlNetInpaintNode(StableDiffusionBaseNode):
    """
    Performs inpainting on images using Stable Diffusion with ControlNet guidance.
    image, inpainting, controlnet, SD, style-transfer, ipadapter

    Use cases:
    - Remove unwanted objects from images with precise control
    - Fill in missing parts of images guided by control images
    - Modify specific areas of images while preserving the rest and maintaining structure
    """

    controlnet: StableDiffusionControlNetModel = Field(
        default=StableDiffusionControlNetModel.INPAINT,
        description="The ControlNet model to use for guidance.",
    )
    init_image: ImageRef = Field(
        default=ImageRef(),
        description="The initial image to be inpainted.",
    )
    mask_image: ImageRef = Field(
        default=ImageRef(),
        description="The mask image indicating areas to be inpainted.",
    )
    control_image: ImageRef = Field(
        default=ImageRef(),
        description="The control image to guide the inpainting process.",
    )
    controlnet_conditioning_scale: float = Field(
        default=0.5,
        description="The scale for ControlNet conditioning.",
        ge=0.0,
        le=2.0,
    )

    _pipeline: StableDiffusionControlNetPAGInpaintPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + [
            "init_image",
            "mask_image",
            "control_image",
            "controlnet_conditioning_scale",
        ]

    def required_inputs(self):
        return ["init_image", "mask_image", "control_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion ControlNet Inpaint"

    @classmethod
    def return_type(cls):
        return {
            "image": ImageRef,
            "latent": TorchTensor,
        }

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        # Use float32 for MPS compatibility with controlnet models
        controlnet_dtype = torch.float32 if context.device == "mps" else torch.float16
        controlnet = await self.load_pipeline(
            context,
            "controlnet",
            self.controlnet.value,
            device=context.device,
            torch_dtype=controlnet_dtype,
        )
        # Align pipeline dtype with controlnet dtype to avoid mismatches
        self._pipeline = await self.load_model(
            context,
            model_class=StableDiffusionControlNetPAGInpaintPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            controlnet=controlnet,
            device=context.device,
            torch_dtype=controlnet_dtype,
        )  # type: ignore
        assert self._pipeline is not None
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> ImageRef:
        init_image = await context.image_to_pil(self.init_image)
        mask_image = await context.image_to_pil(self.mask_image)
        control_image = await context.image_to_pil(self.control_image)
        return await self.run_pipeline(
            context,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            width=init_image.width,
            height=init_image.height,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
        )


class StableDiffusionInpaintNode(StableDiffusionBaseNode):
    """
    Performs inpainting on images using Stable Diffusion.
    image, inpainting, SD

    Use cases:
    - Remove unwanted objects from images
    - Fill in missing parts of images
    - Modify specific areas of images while preserving the rest
    """

    init_image: ImageRef = Field(
        default=ImageRef(),
        description="The initial image to be inpainted.",
    )
    mask_image: ImageRef = Field(
        default=ImageRef(),
        description="The mask image indicating areas to be inpainted.",
    )
    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Strength for inpainting. Higher values allow for more deviation from the original image.",
    )
    variant: ModelVariant = Field(
        default=ModelVariant.FP16,
        description="The variant of the model to use for Image-to-Image generation.",
    )
    _pipeline: StableDiffusionPAGInpaintPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["init_image", "mask_image", "strength"]

    def required_inputs(self):
        return ["init_image", "mask_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion (Inpaint)"

    @classmethod
    def return_type(cls):
        return {
            "image": ImageRef,
            "latent": TorchTensor,
        }

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        if self._pipeline is None:
            self._pipeline = await self.load_model(
                context=context,
                model_class=StableDiffusionPAGInpaintPipeline,
                model_id=self.model.repo_id,
                path=self.model.path,
                safety_checker=None,
                config="Lykon/DreamShaper",
                torch_dtype=(
                    torch.float16
                    if self.variant == ModelVariant.FP16
                    else torch.float32
                ),
                variant=self.variant.value,
            )
            assert self._pipeline is not None
            self._load_ip_adapter()
            self._set_scheduler(self.scheduler)

    async def process(self, context: ProcessingContext) -> ImageRef:
        init_image = await context.image_to_pil(self.init_image)
        mask_image = await context.image_to_pil(self.mask_image)
        return await self.run_pipeline(
            context,
            image=init_image,
            mask_image=mask_image,
            width=init_image.width,
            height=init_image.height,
            strength=self.strength,
        )


class StableDiffusionControlNetImg2ImgNode(StableDiffusionBaseNode):
    """
    Transforms existing images using Stable Diffusion with ControlNet guidance.
    image, generation, image-to-image, controlnet, SD, style-transfer, ipadapter

    Use cases:
    - Modify existing images with precise control over composition and structure
    - Apply specific styles or concepts to photographs or artwork with guided transformations
    - Create variations of existing visual content while maintaining certain features
    - Enhance image editing capabilities with AI-guided transformations
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="The input image to be transformed.",
    )
    strength: float = Field(
        default=0.5,
        description="Similarity to the input image",
        ge=0.0,
        le=1.0,
    )
    controlnet: HFControlNet = Field(
        default=HFControlNet(),
        description="The ControlNet model to use for guidance.",
    )
    control_image: ImageRef = Field(
        default=ImageRef(),
        description="The control image to guide the transformation.",
    )

    _pipeline: StableDiffusionControlNetImg2ImgPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["image", "controlnet", "control_image"]

    def required_inputs(self):
        return ["image", "control_image"]

    @classmethod
    def get_recommended_models(cls):
        return HF_CONTROLNET_MODELS + super().get_recommended_models()

    @classmethod
    def get_title(cls):
        return "Stable Diffusion ControlNet (Img2Img)"

    @classmethod
    def return_type(cls):
        return {
            "image": ImageRef,
            "latent": TorchTensor,
        }

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        if not context.is_huggingface_model_cached(self.controlnet.repo_id):
            raise ValueError(
                f"ControlNet model {self.controlnet.repo_id} must be downloaded first"
            )
        if not context.is_huggingface_model_cached(self.model.repo_id):
            raise ValueError(f"Model {self.model.repo_id} must be downloaded first")

        # Use float32 for MPS compatibility with controlnet models
        controlnet_dtype = torch.float32 if context.device == "mps" else torch.float16
        controlnet_variant = None if context.device == "mps" else "fp16"
        controlnet = await self.load_model(
            context=context,
            model_class=ControlNetModel,
            model_id=self.controlnet.repo_id,
            torch_dtype=controlnet_dtype,
            variant=controlnet_variant,
        )
        # Align pipeline dtype with controlnet dtype to avoid mismatches
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionControlNetImg2ImgPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            controlnet=controlnet,
            torch_dtype=controlnet_dtype,
            config="Lykon/DreamShaper",  # workaround for missing SD15 repo
        )
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        input_image = await context.image_to_pil(self.image)
        control_image = await context.image_to_pil(self.control_image)

        return await self.run_pipeline(
            context,
            image=input_image,
            control_image=control_image,
            width=input_image.width,
            height=input_image.height,
            strength=self.strength,
        )


class StableDiffusionUpscale(HuggingFacePipelineNode):
    """
    Upscales an image using Stable Diffusion 4x upscaler.
    image, upscaling, stable-diffusion, SD

    Use cases:
    - Enhance low-resolution images
    - Improve image quality for printing or display
    - Create high-resolution versions of small images
    """

    prompt: str = Field(
        default="",
        description="The prompt for image generation.",
    )
    negative_prompt: str = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    num_inference_steps: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Number of upscaling steps.",
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="Guidance scale for generation.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="The initial image for Image-to-Image generation.",
    )
    scheduler: StableDiffusionBaseNode.StableDiffusionScheduler = Field(
        default=StableDiffusionBaseNode.StableDiffusionScheduler.HeunDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        le=2**32 - 1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    enable_tiling: bool = Field(
        default=False,
        description="Enable tiling to save VRAM",
    )

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "negative_prompt", "image"]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion 4x Upscale"

    _pipeline: StableDiffusionUpscalePipeline | None = None

    @classmethod
    def get_recommended_models(cls):
        return [
            HFStableDiffusionUpscale(
                repo_id="stabilityai/stable-diffusion-x4-upscaler",
                allow_patterns=[
                    "README.md",
                    "**/*.fp16.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            )
        ]

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionUpscalePipeline,
            model_id="stabilityai/stable-diffusion-x4-upscaler",
        )
        assert self._pipeline is not None
        self._set_scheduler(self.scheduler)

    def _set_scheduler(
        self, scheduler_type: StableDiffusionBaseNode.StableDiffusionScheduler
    ):
        if self._pipeline is not None:
            scheduler_class = StableDiffusionBaseNode.get_scheduler_class(
                scheduler_type
            )
            self._pipeline.scheduler = scheduler_class.from_config(
                self._pipeline.scheduler.config
            )
            if self.enable_tiling:
                self._pipeline.vae.enable_tiling()

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        input_image = await context.image_to_pil(self.image)

        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        upscaled_image = self._pipeline(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=input_image,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            callback=progress_callback(self.id, self.num_inference_steps, context),  # type: ignore
        ).images[  # type: ignore
            0
        ]

        return await context.image_from_pil(upscaled_image)


class StableDiffusionLatentUpscaler(HuggingFacePipelineNode):
    """
    Upscales Stable Diffusion latents (x2) using the SD Latent Upscaler pipeline.
    tensor, upscaling, stable-diffusion, latent, SD

    Input and output are tensors for chaining with latent-based workflows.
    """

    prompt: str = Field(
        default="",
        description="The prompt for upscaling guidance.",
    )
    negative_prompt: str = Field(
        default="",
        description="The negative prompt to guide what should not appear in the result.",
    )
    num_inference_steps: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of upscaling denoising steps.",
    )
    guidance_scale: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="Guidance scale for upscaling. 0 preserves content strongly.",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    latents: TorchTensor = Field(
        default=TorchTensor(),
        description="Low-resolution latents tensor to upscale.",
    )

    _pipeline: StableDiffusionLatentUpscalePipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return [
            "latents",
            "prompt",
            "negative_prompt",
            "num_inference_steps",
            "guidance_scale",
        ]

    def required_inputs(self):
        return ["latents"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion Latent Upscaler"

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionLatentUpscalePipeline,
            model_id="stabilityai/sd-x2-latent-upscaler",
            variant=None,
            torch_dtype=torch.float16,
        )
        assert self._pipeline is not None
        self._pipeline.to(context.device)

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> TorchTensor:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Prepare generator
        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        # Convert TorchTensor to torch.Tensor and ensure device
        low_res_latents = self.latents.to_tensor()
        low_res_latents = low_res_latents.to(context.device)

        # Run latent upscaler with progress callback
        upscaled = self._pipeline(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=low_res_latents,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            output_type="latent",
            callback=progress_callback(self.id, self.num_inference_steps, context),  # type: ignore
            callback_steps=1,
        ).images  # type: ignore

        # Convert back to TorchTensor
        return TorchTensor.from_tensor(upscaled)


class VAEEncode(HuggingFacePipelineNode):
    """
    Encodes an image into latents using a VAE.
    image -> tensor (TorchTensor)
    """

    model: HFVAE = Field(default=HFVAE(), description="The VAE model to use.")
    image: ImageRef = Field(default=ImageRef(), description="Input image to encode.")
    scale_factor: float = Field(
        default=0.18215,
        ge=0.0,
        le=10.0,
        description="Scaling factor applied to latents (e.g., 0.18215 for SD15)",
    )

    _vae: AutoencoderKL | None = None

    @classmethod
    def get_basic_fields(cls):
        return ["model", "image", "scale_factor"]

    @classmethod
    def get_recommended_models(cls) -> list[HFVAE]:
        return [
            HFVAE(repo_id="stabilityai/sd-vae-ft-mse"),
            HFVAE(repo_id="stabilityai/sd-vae-ft-ema"),
            HFVAE(repo_id="stabilityai/sdxl-vae"),
            HFVAE(repo_id="madebyollin/sdxl-vae-fp16-fix"),
        ]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls):
        return "VAE Encode"

    async def preload_model(self, context: ProcessingContext):
        dtype = torch.float32 if context.device == "mps" else torch.float16
        self._vae = await self.load_model(
            context=context,
            model_class=AutoencoderKL,
            model_id=self.model.repo_id,
            path=self.model.path,
            torch_dtype=dtype,
            subfolder=self.model.variant if self.model.variant else None,
        )
        assert self._vae is not None
        self._vae.to(context.device)

    async def move_to_device(self, device: str):
        if self._vae is not None:
            self._vae.to(device)

    async def process(self, context: ProcessingContext) -> TorchTensor:
        if self._vae is None:
            raise ValueError("VAE not initialized")

        # Convert to tensor HWC in [0,1]
        img = await context.image_to_tensor(self.image)
        # To NCHW, batch, and model dtype on correct device
        img = img.permute(2, 0, 1).unsqueeze(0)
        vae_dtype = next(self._vae.parameters()).dtype
        img = img.to(dtype=vae_dtype, device=context.device)
        # Normalize to [-1, 1]
        img = img * 2.0 - 1.0

        with torch.no_grad():
            posterior = self._vae.encode(img)
            latents = posterior.latent_dist.sample()
            if self.scale_factor > 0:
                latents = latents * self.scale_factor

        return TorchTensor.from_tensor(latents.cpu())


class VAEDecode(HuggingFacePipelineNode):
    """
    Decodes latents into an image using a VAE.
    tensor (TorchTensor) -> image
    """

    model: HFVAE = Field(default=HFVAE(), description="The VAE model to use.")
    latents: TorchTensor = Field(
        default=TorchTensor(), description="Latent tensor to decode."
    )
    scale_factor: float = Field(
        default=0.18215,
        ge=0.0,
        le=10.0,
        description="Scaling factor used for encoding (inverse is applied before decode)",
    )

    _vae: AutoencoderKL | None = None

    @classmethod
    def get_basic_fields(cls):
        return ["model", "latents", "scale_factor"]

    @classmethod
    def get_recommended_models(cls) -> list[HFVAE]:
        return [
            HFVAE(repo_id="stabilityai/sd-vae-ft-mse"),
            HFVAE(repo_id="stabilityai/sd-vae-ft-ema"),
            HFVAE(repo_id="stabilityai/sdxl-vae"),
            HFVAE(repo_id="madebyollin/sdxl-vae-fp16-fix"),
        ]

    def required_inputs(self):
        return ["latents"]

    @classmethod
    def get_title(cls):
        return "VAE Decode"

    async def preload_model(self, context: ProcessingContext):
        dtype = torch.float32 if context.device == "mps" else torch.float16
        self._vae = await self.load_model(
            context=context,
            model_class=AutoencoderKL,
            model_id=self.model.repo_id,
            path=self.model.path,
            torch_dtype=dtype,
            subfolder=self.model.variant if self.model.variant else None,
        )
        assert self._vae is not None
        self._vae.to(context.device)

    async def move_to_device(self, device: str):
        if self._vae is not None:
            self._vae.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._vae is None:
            raise ValueError("VAE not initialized")

        latents = self.latents.to_tensor().to(context.device)
        vae_dtype = next(self._vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)

        if self.scale_factor > 0:
            latents = latents / self.scale_factor

        with torch.no_grad():
            decoded = self._vae.decode(latents).sample
            # Map from [-1, 1] to [0, 1]
            decoded = (decoded + 1.0) / 2.0
            # NCHW -> NHWC for image_from_tensor
            decoded = decoded.permute(0, 2, 3, 1).contiguous()

        return await context.image_from_tensor(decoded.cpu())


class StableDiffusionXLImg2Img(StableDiffusionXLBase):
    """
    Transforms existing images based on text prompts using Stable Diffusion XL.
    image, generation, image-to-image, SDXL, style-transfer, ipadapter

    Use cases:
    - Modifying existing images to fit a specific style or theme
    - Enhancing or altering photographs
    - Creating variations of existing artwork
    - Applying text-guided edits to images
    """

    init_image: ImageRef = Field(
        default=ImageRef(),
        description="The initial image for Image-to-Image generation.",
    )
    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Strength for Image-to-Image generation. Higher values allow for more deviation from the original image.",
    )
    variant: ModelVariant = Field(
        default=ModelVariant.FP16,
        description="The variant of the model to use for Image-to-Image generation.",
    )
    _pipeline: StableDiffusionXLPAGImg2ImgPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["init_image", "strength"]

    def required_inputs(self):
        return ["init_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion XL (Img2Img)"

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionXLPAGImg2ImgPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            safety_checker=None,
            torch_dtype=(
                torch.float16 if self.variant == ModelVariant.FP16 else torch.float32
            ),
            variant=self.variant.value,
        )
        assert self._pipeline is not None
        self._pipeline.enable_model_cpu_offload()
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context) -> ImageRef:
        init_image = await context.image_to_pil(self.init_image)
        init_image = init_image.resize((self.width, self.height))
        return await self.run_pipeline(
            context, image=init_image, strength=self.strength
        )


class StableDiffusionXLInpainting(StableDiffusionXLBase):
    """
    Performs inpainting on images using Stable Diffusion XL.
    image, inpainting, SDXL

    Use cases:
    - Remove unwanted objects from images
    - Fill in missing parts of images
    - Modify specific areas of images while preserving the rest
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="The initial image to be inpainted.",
    )
    mask_image: ImageRef = Field(
        default=ImageRef(),
        description="The mask image indicating areas to be inpainted.",
    )
    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Strength for inpainting. Higher values allow for more deviation from the original image.",
    )
    variant: ModelVariant = Field(
        default=ModelVariant.FP16,
        description="The variant of the model to use for Image-to-Image generation.",
    )
    _pipeline: StableDiffusionXLPAGInpaintPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["image", "mask_image", "strength"]

    def required_inputs(self):
        return ["image", "mask_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion XL (Inpaint)"

    async def preload_model(self, context: ProcessingContext):
        if self._pipeline is None:
            self._pipeline = await self.load_model(
                context=context,
                model_class=StableDiffusionXLPAGInpaintPipeline,
                model_id=self.model.repo_id,
                path=self.model.path,
                safety_checker=None,
                torch_dtype=(
                    torch.float16
                    if self.variant == ModelVariant.FP16
                    else torch.float32
                ),
                variant=self.variant.value,
            )
            assert self._pipeline is not None
            self._load_ip_adapter()
            self._set_scheduler(self.scheduler)

    async def process(self, context: ProcessingContext) -> ImageRef:
        input_image = await context.image_to_pil(self.image)
        mask_image = await context.image_to_pil(self.mask_image)
        mask_image = mask_image.resize((self.width, self.height))
        return await self.run_pipeline(
            context,
            image=input_image,
            mask_image=mask_image,
            strength=self.strength,
            width=input_image.width,
            height=input_image.height,
        )


class StableDiffusionXLControlNetNode(StableDiffusionXLImg2Img):
    """
    Transforms existing images using Stable Diffusion XL with ControlNet guidance.
    image, generation, image-to-image, controlnet, SDXL

    Use cases:
    - Modify existing images with precise control over composition and structure
    - Apply specific styles or concepts to photographs or artwork with guided transformations
    - Create variations of existing visual content while maintaining certain features
    - Enhance image editing capabilities with AI-guided transformations
    """

    controlnet: HFControlNet = Field(
        default=HFControlNet(),
        description="The ControlNet model to use for guidance.",
    )
    control_image: ImageRef = Field(
        default=ImageRef(),
        description="The control image to guide the transformation.",
    )
    controlnet_conditioning_scale: float = Field(
        default=1.0,
        description="The scale for ControlNet conditioning.",
        ge=0.0,
        le=2.0,
    )

    _pipeline: StableDiffusionXLControlNetPAGPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + [
            "controlnet",
            "control_image",
            "controlnet_conditioning_scale",
        ]

    def required_inputs(self):
        return ["control_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion XL ControlNet"

    async def preload_model(self, context: ProcessingContext):
        # Use float32 for MPS compatibility with controlnet models
        controlnet_dtype = torch.float32 if context.device == "mps" else torch.float16
        controlnet = await self.load_model(
            context=context,
            model_class=ControlNetModel,
            model_id=self.controlnet.repo_id,
            path=self.controlnet.path,
            variant=None,
            torch_dtype=controlnet_dtype,
        )
        # Align pipeline dtype with controlnet dtype to avoid mismatches
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionXLControlNetPAGPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            controlnet=controlnet,
            torch_dtype=controlnet_dtype,
        )
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_image = await context.image_to_pil(self.control_image)
        init_image = None
        if not self.init_image.is_empty():
            init_image = await context.image_to_pil(self.init_image)
            init_image = init_image.resize((self.width, self.height))

        # Set up the generator for reproducibility
        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        return await self.run_pipeline(
            context,
            init_image=init_image,
            image=control_image,
            strength=self.strength,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            callback_on_step_end=self.progress_callback(context),
            callback_steps=1,
        )


class OmniGenNode(HuggingFacePipelineNode):
    """
    Generates and edits images using the OmniGen model, supporting multimodal inputs.
    image, generation, text-to-image, image-editing, multimodal, omnigen

    Use cases:
    - Generate images from text prompts
    - Edit existing images with text instructions
    - Controllable image generation with reference images
    - Visual reasoning and image manipulation
    - ID and object preserving generation
    """

    prompt: str = Field(
        default="A realistic photo of a young woman sitting on a sofa, holding a book and facing the camera.",
        description="The text prompt for image generation. Use <img><|image_1|></img> placeholders to reference input images.",
    )
    input_images: list[ImageRef] = Field(
        default=[],
        description="List of input images to use for editing or as reference. Referenced in prompt using <img><|image_1|></img>, <img><|image_2|></img>, etc.",
    )
    height: int = Field(
        default=1024, description="Height of the generated image.", ge=64, le=2048
    )
    width: int = Field(
        default=1024, description="Width of the generated image.", ge=64, le=2048
    )
    guidance_scale: float = Field(
        default=2.5,
        description="Guidance scale for generation. Higher values follow the prompt more closely.",
        ge=1.0,
        le=20.0,
    )
    img_guidance_scale: float = Field(
        default=1.6,
        description="Image guidance scale when using input images.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=25, description="Number of denoising steps.", ge=1, le=100
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )
    use_input_image_size_as_output: bool = Field(
        default=False,
        description="If True, use the input image size as output size. Recommended for image editing.",
    )
    max_input_image_size: int = Field(
        default=1024,
        description="Maximum input image size. Smaller values reduce memory usage but may affect quality.",
        ge=256,
        le=2048,
    )
    enable_model_cpu_offload: bool = Field(
        default=False,
        description="Enable CPU offload to reduce memory usage when using multiple images.",
    )

    _pipeline: Any | None = None

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "input_images", "height", "width", "guidance_scale"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="Shitao/OmniGen-v1-diffusers",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
        ]

    def required_inputs(self):
        # No required inputs since it can work with text-only prompts
        return []

    @classmethod
    def get_title(cls) -> str:
        return "OmniGen"

    def get_model_id(self) -> str:
        return "Shitao/OmniGen-v1-diffusers"

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_id="Shitao/OmniGen-v1-diffusers",
            model_class=OmniGenPipeline,
            torch_dtype=torch.bfloat16,
            variant=None,
        )

        if self.enable_model_cpu_offload and self._pipeline is not None:
            self._pipeline.enable_model_cpu_offload()

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        # Convert input images to PIL if provided
        input_images_pil = []
        if self.input_images:
            for img_ref in self.input_images:
                pil_img = await context.image_to_pil(img_ref)
                input_images_pil.append(pil_img)

        # Prepare kwargs for the pipeline
        kwargs = {
            "prompt": self.prompt,
            "height": self.height,
            "width": self.width,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "generator": generator,
            "max_input_image_size": self.max_input_image_size,
            "use_input_image_size_as_output": self.use_input_image_size_as_output,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
        }

        # Add input images if provided
        if input_images_pil:
            kwargs["input_images"] = input_images_pil
            kwargs["img_guidance_scale"] = self.img_guidance_scale

        output = self._pipeline(**kwargs)  # type: ignore
        image = output.images[0]

        return await context.image_from_pil(image)


class QwenImageEdit(HuggingFacePipelineNode):
    """
    Performs image editing using the Qwen Image Edit model.
    image, editing, semantic, appearance, qwen, multimodal
    
    Use cases:
    - Semantic editing (object rotation, style transfer)
    - Appearance editing (adding/removing elements) 
    - Precise text modifications in images
    - Background and clothing changes
    - Complex image transformations guided by text
    """
    
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The input image to edit"
    )
    prompt: str = Field(
        default="Change the object's color to blue",
        description="Text description of the desired edit to apply to the image"
    )
    negative_prompt: str = Field(
        default="",
        description="Text describing what should not appear in the edited image"
    )
    num_inference_steps: int = Field(
        default=50,
        description="Number of denoising steps for the editing process",
        ge=1,
        le=100
    )
    true_cfg_scale: float = Field(
        default=4.0,
        description="Guidance scale for editing. Higher values follow the prompt more closely",
        ge=1.0,
        le=20.0
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed",
        ge=-1
    )
    
    _pipeline: QwenImageEditPipeline | None = None
    
    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "negative_prompt", "true_cfg_scale"]
    
    def required_inputs(self):
        return ["image"]
    
    @classmethod
    def get_title(cls) -> str:
        return "Qwen Image Edit"
    
    def get_model_id(self) -> str:
        return "Qwen/Qwen-Image-Edit"
    
    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="Qwen/Qwen-Image-Edit",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"]
            )
        ]
    
    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_id="Qwen/Qwen-Image-Edit",
            model_class=QwenImageEditPipeline,
            torch_dtype=torch.bfloat16,
            variant=None
        )
    
    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.to(device)
    
    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")
        
        # Set up the generator for reproducibility
        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)
        
        input_image = await context.image_to_pil(self.image)
        
        # Prepare kwargs for the pipeline
        kwargs = {
            "image": input_image,
            "prompt": self.prompt,
            "generator": generator,
            "true_cfg_scale": self.true_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            )
        }
        
        # Add negative prompt if provided
        if self.negative_prompt:
            kwargs["negative_prompt"] = self.negative_prompt
        else:
            kwargs["negative_prompt"] = " "  # Default as shown in Qwen example
        
        output = self._pipeline(**kwargs)  # type: ignore
        image = output.images[0]
        
        return await context.image_from_pil(image)
