from enum import Enum
import platform
import re
import asyncio
from typing import Any, TypedDict
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from nodetool.config.logging_config import get_logger
from nodetool.huggingface.nunchaku_utils import (
    get_nunchaku_text_encoder,
    get_nunchaku_transformer,
)
from nodetool.workflows.types import NodeProgress
import torch
from RealESRGAN import RealESRGAN
from huggingface_hub import try_to_load_from_cache
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision as get_nunchaku_precision
from nodetool.metadata.types import (
    HFImageToImage,
    HFControlNet,
    HFQwenImageEdit,
    HFRealESRGAN,
    HFVAE,
    HFFluxKontext,
    HFFluxRedux,
    HFFlux,
    HFFluxFill,
    TorchTensor,
    HuggingFaceModel,
    ImageRef,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.nodes.huggingface.stable_diffusion_base import (
    HF_CONTROLNET_MODELS,
    HF_CONTROLNET_XL_MODELS,
    ModelVariant,
    _select_diffusion_dtype,
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
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
)
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_prior_redux import FluxPriorReduxPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.controlnet.pipeline_controlnet import (
    StableDiffusionControlNetPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import (
    StableDiffusionControlNetImg2ImgPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
    StableDiffusionControlNetInpaintPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import (
    StableDiffusionXLControlNetPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl_img2img import (
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import (
    StableDiffusionUpscalePipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_latent_upscale import (
    StableDiffusionLatentUpscalePipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
    StableDiffusionXLInpaintPipeline,
)
from pydantic import Field

log = get_logger(__name__)


async def _get_nunchaku_text_encoder_kwargs(
    context: ProcessingContext, node_id: str, model_repo_id: str
) -> dict[str, Any]:
    """Return kwargs for Flux pipelines when a Nunchaku text encoder is available."""
    try:
        text_encoder = await get_nunchaku_text_encoder(context, node_id)
    except Exception as exc:
        log.warning(
            "Unable to load Nunchaku text encoder for %s: %s", model_repo_id, exc
        )
        return {}

    if text_encoder is None:
        return {}

    return {"text_encoder_2": text_encoder}



def _enable_pytorch2_attention(pipeline: Any, enabled: bool = True):
    """Enable PyTorch 2 scaled dot product attention to speed up inference."""
    if not enabled or pipeline is None:
        return

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
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> ImageRef:
        assert self._pipeline is not None
        image = await context.image_to_pil(self.image)
        result = await self.run_pipeline_in_thread(image, prompt=self.prompt)  # type: ignore
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
        import torch

        def _predict():
            with torch.inference_mode():
                return self._model.predict(image)

        sr_image = await asyncio.to_thread(_predict)
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
                repo_id="caidas/swin2SR-classical-sr-x4-64",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFImageToImage(
                repo_id="caidas/swin2SR-lightweight-x2-64",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFImageToImage(
                repo_id="caidas/swin2SR-compressed-sr-x4-48",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
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

    async def preload_model(self, context: ProcessingContext):
        torch_dtype = _select_diffusion_dtype()
        await self.load_model(
            context=context,
            model_id=self.repo_id,
            model_class=AutoPipelineForImage2Image,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=None,
        )

    async def process(self, context: ProcessingContext) -> HFImageToImage:
        return HFImageToImage(
            repo_id=self.repo_id,
            variant=None,
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
        torch_dtype = _select_diffusion_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.model.repo_id,
            path=self.model.path,
            model_class=AutoPipelineForImage2Image,
            torch_dtype=torch_dtype,
        )
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            try:
                self._pipeline.to(device)
            except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                raise ValueError(
                    "VRAM out of memory while moving Qwen Image Edit pipeline to device. "
                    "Try enabling 'CPU offload' in the advanced node properties (if available), "
                    "or reduce image size/steps."
                ) from e

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

        try:
            output = await self.run_pipeline_in_thread(**kwargs)  # type: ignore
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while running Qwen Image Edit. "
                "Enable 'CPU offload' in the advanced node properties (if available), "
                "or reduce image size/steps."
            ) from e
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
        torch_dtype = _select_diffusion_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.model.repo_id,
            path=self.model.path,
            model_class=AutoPipelineForInpainting,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=self.model.variant,
        )
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

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

        output = await self.run_pipeline_in_thread(**kwargs)  # type: ignore

        image = output.images[0]

        return await context.image_from_pil(image)


class StableDiffusionControlNet(StableDiffusionBaseNode):
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

    _pipeline: StableDiffusionControlNetPipeline | None = None

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

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.controlnet.to(device)
            self._pipeline.unet.to(device)
            self._pipeline.vae.to(device)
            self._pipeline.text_encoder.to(device)

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        # Use PyTorch 2 attention optimizations while keeping MPS/controlnet compatibility
        controlnet_dtype = torch.float32 if context.device == "mps" else _select_diffusion_dtype()
        controlnet = await self.load_model(
            context=context,
            model_class=ControlNetModel,
            model_id=self.controlnet.repo_id,
            torch_dtype=controlnet_dtype,
            variant=None,
        )
        # Align pipeline dtype with controlnet dtype to avoid mismatches
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionControlNetPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            controlnet=controlnet,
            config="Lykon/DreamShaper",  # workaround for missing SD15 repo
            torch_dtype=controlnet_dtype,
        )
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> OutputType:
        control_image = await context.image_to_pil(self.control_image)
        output = await self.run_pipeline(
            context,
            image=control_image,
            width=control_image.width,
            height=control_image.height,
            controlnet_conditioning_scale=float(self.controlnet_conditioning_scale),
        )
        return {
            "image": output if isinstance(output, ImageRef) else None,
            "latent": output if isinstance(output, TorchTensor) else None,
        }


class StableDiffusionImg2ImgNode(StableDiffusionBaseNode):
    """
    Transforms existing images based on text prompts using Stable Diffusion.
    image, generation, image-to-image, SD, img2img, style-transfer

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
    _pipeline: StableDiffusionImg2ImgPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["init_image", "strength"]

    def required_inputs(self):
        return ["init_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion (Img2Img)"

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionImg2ImgPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            safety_checker=None,
            config="Lykon/DreamShaper",
            torch_dtype=_select_diffusion_dtype(),
        )
        assert self._pipeline is not None
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> OutputType:
        init_image = await context.image_to_pil(self.init_image)
        result = await self.run_pipeline(
            context,
            image=init_image,
            strength=self.strength,
        )
        return {
            "image": result if isinstance(result, ImageRef) else None,
            "latent": result if isinstance(result, TorchTensor) else None,
        }


class StableDiffusionControlNetModel(str, Enum):
    INPAINT = "lllyasviel/control_v11p_sd15_inpaint"


class StableDiffusionControlNetInpaintNode(StableDiffusionBaseNode):
    """
    Performs inpainting on images using Stable Diffusion with ControlNet guidance.
    image, inpainting, controlnet, SD, style-transfer

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

    _pipeline: StableDiffusionControlNetInpaintPipeline | None = None

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

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        controlnet_dtype = torch.float32 if context.device == "mps" else _select_diffusion_dtype()
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
            model_class=StableDiffusionControlNetInpaintPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            controlnet=controlnet,
            device=context.device,
            torch_dtype=controlnet_dtype,
        )  # type: ignore
        assert self._pipeline is not None
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> OutputType:
        init_image = await context.image_to_pil(self.init_image)
        mask_image = await context.image_to_pil(self.mask_image)
        control_image = await context.image_to_pil(self.control_image)
        result = await self.run_pipeline(
            context,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            width=init_image.width,
            height=init_image.height,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
        )
        return {
            "image": result if isinstance(result, ImageRef) else None,
            "latent": result if isinstance(result, TorchTensor) else None,
        }


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
    _pipeline: StableDiffusionInpaintPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["init_image", "mask_image", "strength"]

    def required_inputs(self):
        return ["init_image", "mask_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion (Inpaint)"

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        if self._pipeline is None:
            self._pipeline = await self.load_model(
                context=context,
                model_class=StableDiffusionInpaintPipeline,
                model_id=self.model.repo_id,
                path=self.model.path,
                safety_checker=None,
                config="Lykon/DreamShaper",
                torch_dtype=_select_diffusion_dtype(),
                variant=None,
            )
            assert self._pipeline is not None
            _enable_pytorch2_attention(self._pipeline)
            _apply_vae_optimizations(self._pipeline)
            self._set_scheduler(self.scheduler)
            self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> OutputType:
        init_image = await context.image_to_pil(self.init_image)
        mask_image = await context.image_to_pil(self.mask_image)
        result = await self.run_pipeline(
            context,
            image=init_image,
            mask_image=mask_image,
            width=init_image.width,
            height=init_image.height,
            strength=self.strength,
        )
        return {
            "image": result if isinstance(result, ImageRef) else None,
            "latent": result if isinstance(result, TorchTensor) else None,
        }


class StableDiffusionControlNetImg2ImgNode(StableDiffusionBaseNode):
    """
    Transforms existing images using Stable Diffusion with ControlNet guidance.
    image, generation, image-to-image, controlnet, SD, style-transfer

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

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        if not context.is_huggingface_model_cached(self.controlnet.repo_id):
            raise ValueError(
                f"ControlNet model {self.controlnet.repo_id} must be downloaded first"
            )
        if not context.is_huggingface_model_cached(self.model.repo_id):
            raise ValueError(f"Model {self.model.repo_id} must be downloaded first")

        # Use float32 for MPS compatibility with controlnet models
        controlnet_dtype = torch.float32 if context.device == "mps" else _select_diffusion_dtype()
        controlnet_variant = None
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
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> OutputType:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        input_image = await context.image_to_pil(self.image)
        control_image = await context.image_to_pil(self.control_image)

        result = await self.run_pipeline(
            context,
            image=input_image,
            control_image=control_image,
            width=input_image.width,
            height=input_image.height,
            strength=self.strength,
        )
        return {
            "image": result if isinstance(result, ImageRef) else None,
            "latent": result if isinstance(result, TorchTensor) else None,
        }


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
            HFImageToImage(
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
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
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

        def _run_pipeline_sync():
            with torch.inference_mode():
                return self._pipeline(
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt,
                    image=input_image,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    callback=progress_callback(
                        self.id, self.num_inference_steps, context
                    ),  # type: ignore
                )

        output = await asyncio.to_thread(_run_pipeline_sync)
        upscaled_image = output.images[0]  # type: ignore

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
            torch_dtype=_select_diffusion_dtype(),
        )
        assert self._pipeline is not None
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
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
        def _run_pipeline_sync():
            with torch.inference_mode():
                return self._pipeline(
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt,
                    image=low_res_latents,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                    output_type="latent",
                    callback=progress_callback(
                        self.id, self.num_inference_steps, context
                    ),  # type: ignore
                    callback_steps=1,
                )

        output = await asyncio.to_thread(_run_pipeline_sync)
        upscaled = output.images  # type: ignore

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
        self._vae.to(context.device)  # type: ignore

    async def move_to_device(self, device: str):
        if self._vae is not None:
            self._vae.to(device)  # type: ignore

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

        with torch.inference_mode():
            posterior = self._vae.encode(img)
            assert isinstance(posterior, AutoencoderKLOutput)
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
        self._vae.to(context.device)  # type: ignore

    async def move_to_device(self, device: str):
        if self._vae is not None:
            self._vae.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._vae is None:
            raise ValueError("VAE not initialized")

        latents = self.latents.to_tensor().to(context.device)
        vae_dtype = next(self._vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)

        if self.scale_factor > 0:
            latents = latents / self.scale_factor

        with torch.inference_mode():
            output = self._vae.decode(latents)
            assert isinstance(output, DecoderOutput)
            decoded = output.sample
            # Map from [-1, 1] to [0, 1]
            decoded = (decoded + 1.0) / 2.0
            # NCHW -> NHWC for image_from_tensor
            decoded = decoded.permute(0, 2, 3, 1).contiguous()

        return await context.image_from_tensor(decoded.cpu())


class StableDiffusionXLImg2Img(StableDiffusionXLBase):
    """
    Transforms existing images based on text prompts using Stable Diffusion XL.
    image, generation, image-to-image, SDXL, style-transfer

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
    _pipeline: StableDiffusionXLImg2ImgPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["init_image", "strength"]

    def required_inputs(self):
        return ["init_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion XL (Img2Img)"

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionXLImg2ImgPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            safety_checker=None,
            torch_dtype=_select_diffusion_dtype(),
            variant=None,
        )
        assert self._pipeline is not None
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        self._pipeline.enable_model_cpu_offload()
        self._set_scheduler(self.scheduler)
        self._load_ip_adapter()

    async def process(self, context) -> OutputType:
        init_image = await context.image_to_pil(self.init_image)
        init_image = init_image.resize((self.width, self.height))
        result = await self.run_pipeline(
            context, image=init_image, strength=self.strength
        )
        return {
            "image": result if isinstance(result, ImageRef) else None,
            "latent": result if isinstance(result, TorchTensor) else None,
        }


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
    _pipeline: StableDiffusionXLInpaintPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["image", "mask_image", "strength"]

    def required_inputs(self):
        return ["image", "mask_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion XL (Inpaint)"

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def preload_model(self, context: ProcessingContext):
        if self._pipeline is None:
            self._pipeline = await self.load_model(
                context=context,
                model_class=StableDiffusionXLInpaintPipeline,
                model_id=self.model.repo_id,
                path=self.model.path,
                safety_checker=None,
                torch_dtype=_select_diffusion_dtype(),
                variant=None,
            )
            assert self._pipeline is not None
            _enable_pytorch2_attention(self._pipeline)
            _apply_vae_optimizations(self._pipeline)
            self._set_scheduler(self.scheduler)
            self._load_ip_adapter()

    async def process(self, context: ProcessingContext) -> OutputType:
        input_image = await context.image_to_pil(self.image)
        mask_image = await context.image_to_pil(self.mask_image)
        mask_image = mask_image.resize((self.width, self.height))
        result = await self.run_pipeline(
            context,
            image=input_image,
            mask_image=mask_image,
            strength=self.strength,
            width=input_image.width,
            height=input_image.height,
        )
        return {
            "image": result if isinstance(result, ImageRef) else None,
            "latent": result if isinstance(result, TorchTensor) else None,
        }


class StableDiffusionXLControlNet(StableDiffusionXLBase):
    """
    Generates images using Stable Diffusion XL with ControlNet guidance.
    image, generation, text-to-image, controlnet, SDXL

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
    # Override default to disable attention slicing for better performance
    # Users can enable it if they need to save VRAM
    enable_attention_slicing: bool = Field(
        default=False,
        description="Enable attention slicing for the pipeline. This can reduce VRAM usage but may slow down generation.",
    )

    _pipeline: StableDiffusionXLControlNetPipeline | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFControlNet]:
        return HF_CONTROLNET_XL_MODELS

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

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # Move entire pipeline to device to ensure all components are on the same device
            self._pipeline.to(device)

    async def preload_model(self, context: ProcessingContext):
        await super().preload_model(context)
        controlnet_dtype = torch.float32 if context.device == "mps" else _select_diffusion_dtype()
        controlnet_variant = None

        controlnet = await self.load_model(
            context=context,
            model_class=ControlNetModel,
            model_id=self.controlnet.repo_id,
            path=self.controlnet.path,
            variant=controlnet_variant,
            torch_dtype=controlnet_dtype,
        )
        # Align pipeline dtype with controlnet dtype to avoid mismatches
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionXLControlNetPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            controlnet=controlnet,
            variant=controlnet_variant,
            torch_dtype=controlnet_dtype,
        )
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        self._set_scheduler(self.scheduler)
        # Ensure pipeline is on the correct device after loading
        if self._pipeline is not None:
            self._pipeline.to(context.device)

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def process(self, context: ProcessingContext) -> OutputType:
        control_image = await context.image_to_pil(self.control_image)
        output = await self.run_pipeline(
            context,
            image=control_image,
            width=control_image.width,
            height=control_image.height,
            controlnet_conditioning_scale=float(self.controlnet_conditioning_scale),
        )
        return {
            "image": output if isinstance(output, ImageRef) else None,
            "latent": output if isinstance(output, TorchTensor) else None,
        }


class StableDiffusionXLControlNetImg2ImgNode(StableDiffusionXLImg2Img):
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
    # Override default to disable attention slicing for better performance
    # Users can enable it if they need to save VRAM
    enable_attention_slicing: bool = Field(
        default=False,
        description="Enable attention slicing for the pipeline. This can reduce VRAM usage but may slow down generation.",
    )

    _pipeline: StableDiffusionXLControlNetImg2ImgPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + [
            "controlnet",
            "control_image",
            "controlnet_conditioning_scale",
        ]

    @classmethod
    def get_recommended_models(cls) -> list[HFControlNet]:
        return HF_CONTROLNET_XL_MODELS

    def required_inputs(self):
        return ["control_image", "init_image"]

    @classmethod
    def get_title(cls):
        return "Stable Diffusion XL ControlNet (Img2Img)"

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # Move entire pipeline to device to ensure all components are on the same device
            self._pipeline.to(device)

    async def preload_model(self, context: ProcessingContext):
        controlnet_dtype = torch.float32 if context.device == "mps" else _select_diffusion_dtype()
        controlnet_variant = None

        controlnet = await self.load_model(
            context=context,
            model_class=ControlNetModel,
            model_id=self.controlnet.repo_id,
            path=self.controlnet.path,
            variant=controlnet_variant,
            torch_dtype=controlnet_dtype,
        )
        # Align pipeline dtype with controlnet dtype to avoid mismatches
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableDiffusionXLControlNetImg2ImgPipeline,
            model_id=self.model.repo_id,
            path=self.model.path,
            controlnet=controlnet,
            variant=controlnet_variant,
            torch_dtype=controlnet_dtype,
        )
        # Ensure pipeline is on the correct device after loading
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)
        if self._pipeline is not None:
            self._pipeline.to(context.device)

    class OutputType(TypedDict):
        image: ImageRef | None
        latent: TorchTensor | None

    async def process(self, context: ProcessingContext) -> OutputType:
        control_image = await context.image_to_pil(self.control_image)
        init_image = await context.image_to_pil(self.init_image)
        init_image = init_image.resize((self.width, self.height))

        # Set up the generator for reproducibility
        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        result = await self.run_pipeline(
            context,
            image=init_image,
            control_image=control_image,
            strength=self.strength,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            callback_on_step_end=self.progress_callback(context),
            callback_steps=1,
        )
        return {
            "image": result if isinstance(result, ImageRef) else None,
            "latent": result if isinstance(result, TorchTensor) else None,
        }


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
        torch_dtype = _select_diffusion_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_id="Shitao/OmniGen-v1-diffusers",
            model_class=OmniGenPipeline,
            torch_dtype=torch_dtype,
            variant=None,
        )

        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

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

        output = await self.run_pipeline_in_thread(**kwargs)  # type: ignore
        image = output.images[0]

        return await context.image_from_pil(image)


class QwenImageEdit(HuggingFacePipelineNode):
    """
    Performs image editing using the Qwen Image Edit model with support for Nunchaku quantization.
    image, editing, semantic, appearance, qwen, multimodal, quantization

    Use cases:
    - Semantic editing (object rotation, style transfer)
    - Appearance editing (adding/removing elements)
    - Precise text modifications in images
    - Background and clothing changes
    - Complex image transformations guided by text
    - Memory-efficient editing using Nunchaku quantization
    """

    model: HFQwenImageEdit = Field(
        default=HFQwenImageEdit(
            repo_id="mit-han-lab/nunchaku-qwen-image-edit",
            path="awq-int4-qwen-image-edit.safetensors",
        ),
        description="The Qwen-Image-Edit model to use for image editing.",
    )

    image: ImageRef = Field(
        default=ImageRef(), title="Input Image", description="The input image to edit"
    )
    prompt: str = Field(
        default="Change the object's color to blue",
        description="Text description of the desired edit to apply to the image",
    )
    negative_prompt: str = Field(
        default="",
        description="Text describing what should not appear in the edited image",
    )
    num_inference_steps: int = Field(
        default=50,
        description="Number of denoising steps for the editing process",
        ge=1,
        le=100,
    )
    true_cfg_scale: float = Field(
        default=4.0,
        description="Guidance scale for editing. Higher values follow the prompt more closely",
        ge=1.0,
        le=20.0,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed",
        ge=-1,
    )
    enable_memory_efficient_attention: bool = Field(
        default=True,
        description="Enable memory efficient attention to reduce VRAM usage.",
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Enable CPU offload to reduce VRAM usage.",
    )

    _pipeline: QwenImageEditPipeline | DiffusionPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return ["model", "image", "prompt", "negative_prompt", "true_cfg_scale"]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls) -> str:
        return "Qwen Image Edit"

    def get_model_id(self) -> str:
        if self.model.repo_id:
            return self.model.repo_id
        return "Qwen/Qwen-Image-Edit"

    @classmethod
    def get_recommended_models(cls) -> list[HFQwenImageEdit]:
        return [
            HFQwenImageEdit(
                repo_id="mit-han-lab/nunchaku-qwen-image-edit",
                path="awq-int4-qwen-image-edit.safetensors",
            ),
        ]

    async def _load_full_precision_pipeline(self, context: ProcessingContext):
        log.info(
            f"Loading Qwen-Image-Edit pipeline from {self.get_model_id()} without quantization..."
        )

        torch_dtype = _select_diffusion_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_class=QwenImageEditPipeline,
            model_id=self.get_model_id(),
            torch_dtype=torch_dtype,
            device="cpu",  # Load on CPU first, then move to GPU in workflow runner
        )
        assert self._pipeline is not None

        # Apply memory optimizations after loading
        _enable_pytorch2_attention(
            self._pipeline, self.enable_memory_efficient_attention
        )
        _apply_vae_optimizations(self._pipeline)
        if self.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()

        if self.enable_memory_efficient_attention:
            self._pipeline.enable_attention_slicing()

    async def preload_model(self, context: ProcessingContext):
        await self._load_full_precision_pipeline(context)

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
                try:
                    self._pipeline.to(device)
                except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                    raise ValueError(
                        "VRAM out of memory while moving Qwen Image Edit pipeline to device. "
                        "Enable 'CPU offload' in advanced node properties or reduce image size/steps."
                    ) from e

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = torch.Generator(device=context.device)
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
            ),
        }

        # Add negative prompt if provided
        if self.negative_prompt:
            kwargs["negative_prompt"] = self.negative_prompt
        else:
            kwargs["negative_prompt"] = " "  # Default as shown in Qwen example

        try:
            output = await self.run_pipeline_in_thread(**kwargs)  # type: ignore
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while running Qwen Image Edit. "
                "Enable 'CPU offload' in the advanced node properties (if available), "
                "or reduce image size/steps."
            ) from e
        image = output.images[0]

        return await context.image_from_pil(image)


class FluxFill(HuggingFacePipelineNode):
    """
    Performs image inpainting/filling using FLUX Fill models with support for GGUF quantization.
    image, inpainting, fill, flux, quantization, mask

    Use cases:
    - Fill masked regions in images with high-quality content
    - Remove unwanted objects from images
    - Complete missing parts of images
    - Memory-efficient inpainting using GGUF quantization
    - High-quality image editing with FLUX models
    """

    model: HFFluxFill = Field(
        default=HFFluxFill(repo_id="black-forest-labs/FLUX.1-Fill-dev"),
        description="The FLUX Fill model to use for image inpainting.",
    )
    prompt: str = Field(
        default="a white paper cup",
        description="A text prompt describing what should fill the masked area.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The input image to fill/inpaint",
    )
    mask_image: ImageRef = Field(
        default=ImageRef(),
        title="Mask Image",
        description="The mask image indicating areas to be filled (white areas will be filled)",
    )
    height: int = Field(
        default=1024, description="The height of the generated image.", ge=64, le=2048
    )
    width: int = Field(
        default=1024, description="The width of the generated image.", ge=64, le=2048
    )
    guidance_scale: float = Field(
        default=30.0,
        description="Guidance scale for generation. Higher values follow the prompt more closely",
        ge=0.0,
        le=50.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Number of denoising steps",
        ge=1,
        le=100,
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

    _pipeline: FluxFillPipeline | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFFluxFill]:
        return [
            HFFluxFill(
                repo_id="black-forest-labs/FLUX.1-Fill-dev",
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Flux Fill"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "image",
            "mask_image",
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "num_inference_steps",
            "seed",
        ]

    def required_inputs(self):
        return ["image", "mask_image"]

    def get_model_id(self) -> str:
        if self.model.repo_id:
            return self.model.repo_id
        return "black-forest-labs/FLUX.1-Fill-dev"

    async def _load_nunchaku_model(self, context: ProcessingContext, torch_dtype):
        """Load FLUX Fill model with Nunchaku quantization."""
        hf_token = await context.get_secret("HF_TOKEN")
        if not hf_token:
            model_url = f"https://huggingface.co/{self.get_model_id()}"
            raise ValueError(
                f"Flux Fill is a gated model, please set the HF_TOKEN in Nodetool settings and accept the terms of use for the model: {model_url}"
            )

        transformer = FluxTransformer2DModel.from_single_file(
            cache_path,
            torch_dtype=torch_dtype,
            token=hf_token,
        )

        # Create the pipeline with the quantized transformer
        log.info("Creating FLUX Fill pipeline...")

        base_model_id = "black-forest-labs/FLUX.1-Fill-dev"

        log.info(
            f"Loading FLUX Fill pipeline from {base_model_id} with quantized transformer..."
        )
        self._pipeline = FluxFillPipeline.from_pretrained(
            base_model_id,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )

        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

    async def preload_model(self, context: ProcessingContext):
        # Use bfloat16 for FLUX Fill models
        torch_dtype = torch.bfloat16

        # Check if this is a GGUF model based on file extension
        # Load the full pipeline normally
        log.info(f"Loading FLUX Fill pipeline from {self.get_model_id()}...")
        self._pipeline = await self.load_model(
            context=context,
            model_id=self.get_model_id(),
            path=self.model.path,
            model_class=FluxFillPipeline,
            torch_dtype=torch_dtype,
            variant=None,
            device="cpu",
        )

        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

        # Apply CPU offload if enabled
        if self._pipeline is not None and self.enable_cpu_offload:
            self._pipeline.enable_sequential_cpu_offload()

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
                            "VRAM out of memory while moving Flux Fill to device. "
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
                        "VRAM out of memory while moving Flux Fill to device. "
                        "Try enabling 'CPU offload' in the advanced node properties, reduce image size, or lower steps."
                    ) from e

            _apply_vae_optimizations(self._pipeline)

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
            "height": self.height,
            "width": self.width,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "max_sequence_length": self.max_sequence_length,
            "generator": generator,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
            "callback_on_step_end_tensor_inputs": ["latents"],
        }

        try:
            output = await self.run_pipeline_in_thread(**kwargs)  # type: ignore
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while running Flux Fill. "
                "Try enabling 'CPU offload' in the advanced node properties "
                "(Enable CPU offload), reduce image size, or lower max_sequence_length."
            ) from e

        image = output.images[0]

        return await context.image_from_pil(image)


class FluxKontext(HuggingFacePipelineNode):
    """
    Performs image editing using FLUX Kontext models for context-aware image generation.
    image, editing, flux, kontext, context-aware, generation

    Use cases:
    - Edit images based on reference context
    - Add elements to images guided by prompts
    - Context-aware image modifications
    - High-quality image editing with FLUX models
    """

    model: HFFluxKontext = Field(
        default=HFFluxKontext(repo_id="black-forest-labs/FLUX.1-Kontext-dev"),
        description="The FLUX Kontext model to use for image editing.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The input image to edit",
    )
    prompt: str = Field(
        default="Add a hat to the cat",
        description="Text description of the desired edit to apply to the image",
    )
    guidance_scale: float = Field(
        default=2.5,
        description="Guidance scale for editing. Higher values follow the prompt more closely",
        ge=0.0,
        le=30.0,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Enable CPU offload to reduce VRAM usage.",
    )

    _pipeline: FluxKontextPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return ["model", "image", "prompt", "guidance_scale"]

    @classmethod
    def get_recommended_models(cls) -> list[HFFluxKontext]:
        return [
            HFFluxKontext(repo_id="black-forest-labs/FLUX.1-Kontext-dev"),
            HFFluxKontext(
                repo_id="nunchaku-tech/nunchaku-flux.1-kontext-dev",
                path="svdq-int4_r32-flux.1-kontext-dev.safetensors",
            ),
            HFFluxKontext(
                repo_id="nunchaku-tech/nunchaku-flux.1-kontext-dev",
                path="svdq-fp4_r32-flux.1-kontext-dev.safetensors",
            ),
            HuggingFaceModel(
                repo_id="mit-han-lab/nunchaku-t5",
                path="awq-int4-flux.1-t5xxl.safetensors",
            ),
        ]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls) -> str:
        return "Flux Kontext"

    def get_model_id(self) -> str:
        if self.model.repo_id:
            return self.model.repo_id
        return "black-forest-labs/FLUX.1-Kontext-dev"

    async def preload_model(self, context: ProcessingContext):
        hf_token = await context.get_secret("HF_TOKEN")
        if not hf_token:
            model_url = f"https://huggingface.co/{self.get_model_id()}"
            raise ValueError(
                f"Flux Kontext is a gated model, please set the HF_TOKEN in Nodetool settings and accept the terms of use for the model: {model_url}"
            )

        log.info(f"Loading FLUX Kontext pipeline from {self.get_model_id()}...")
        torch_dtype = torch.bfloat16
        if self._is_nunchaku_model():
            await self._load_nunchaku_model(
                context=context,
                torch_dtype=torch_dtype,
                hf_token=hf_token,
            )
        else:
            self._pipeline = await self.load_model(
                context=context,
                model_class=FluxKontextPipeline,
                model_id=self.get_model_id(),
                torch_dtype=torch_dtype,
                device="cpu",
                token=hf_token,
            )

        # Apply CPU offload if enabled
        _enable_pytorch2_attention(self._pipeline)
        if self._pipeline is not None and self.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()

    def _is_nunchaku_model(self) -> bool:
        repo_has_svdq = self.model.repo_id and "svdq" in self.model.repo_id.lower()
        path_has_svdq = self.model.path and "svdq" in self.model.path.lower()
        return bool(repo_has_svdq or path_has_svdq)

    async def _load_nunchaku_model(
        self,
        context: ProcessingContext,
        torch_dtype: torch.dtype,
        hf_token: str,
    ):
        repo_id_lower = (self.model.repo_id or "").lower()
        transformer_repo_id = (
            self.model.repo_id
            if self.model.repo_id and "nunchaku" in repo_id_lower and "kontext" in repo_id_lower
            else "nunchaku-tech/nunchaku-flux.1-kontext-dev"
        )

        precision = get_nunchaku_precision()
        transformer_path = (
            self.model.path
            or f"svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
        )

        if "svdq" not in transformer_path.lower():
            raise ValueError(
                "Nunchaku Flux Kontext requires a transformer filename containing 'svdq'."
            )

        log.info(
            "Loading Nunchaku Flux Kontext transformer from %s/%s (precision=%s)",
            transformer_repo_id,
            transformer_path,
            precision,
        )

        transformer = await get_nunchaku_transformer(
            context=context,
            model_class=NunchakuFluxTransformer2dModel,
            node_id=self.id,
            repo_id=transformer_repo_id,
            path=transformer_path,
        )

        log.info(
            "Creating FLUX Kontext pipeline from %s with Nunchaku transformer %s/%s",
            "black-forest-labs/FLUX.1-Kontext-dev",
            transformer_repo_id,
            transformer_path,
        )
        text_encoder_kwargs = await _get_nunchaku_text_encoder_kwargs(
            context,
            self.id,
            transformer_repo_id,
        )
        try:
            self._pipeline = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev",
                transformer=transformer,
                torch_dtype=torch_dtype,
                token=hf_token,
                **text_encoder_kwargs,
            )
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while loading Flux Kontext with the Nunchaku transformer. "
                "Try enabling CPU offload or reduce image size."
            ) from e

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
                        "VRAM out of memory while moving Flux Kontext to device. "
                        "Enable 'CPU offload' in the advanced node properties or reduce image size."
                    ) from e

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = torch.Generator(device=context.device)
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        input_image = await context.image_to_pil(self.image)

        # Prepare kwargs for the pipeline
        kwargs = {
            "image": input_image,
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "generator": generator,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, 20, context  # Default steps for Flux Kontext
            ),
        }

        try:
            output = await self.run_pipeline_in_thread(**kwargs)  # type: ignore
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while running Flux Kontext. "
                "Enable 'CPU offload' in the advanced node properties (if available), "
                "or reduce image size."
            ) from e
        image = output.images[0]

        return await context.image_from_pil(image)


class FluxPriorRedux(HuggingFacePipelineNode):
    """
    Performs image transformation using FLUX Prior Redux pipeline for image-conditioned generation.
    image, transformation, flux, redux, prior, image-conditioned, generation

    Use cases:
    - Transform images using FLUX Prior Redux for style transfer and image variations
    - Generate images conditioned on reference images without text prompts
    - High-quality image-to-image transformation with FLUX models
    - Create variations of existing images with FLUX Prior Redux guidance
    """

    prior_redux_model: HFFluxRedux = Field(
        default=HFFluxRedux(repo_id="black-forest-labs/FLUX.1-Redux-dev"),
        description="The FLUX Prior Redux model to use for image conditioning.",
    )
    flux_model: HFFlux = Field(
        default=HFFlux(repo_id="black-forest-labs/FLUX.1-dev"),
        description="The FLUX base model to use for generation.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The input image to transform",
    )
    guidance_scale: float = Field(
        default=2.5,
        description="Guidance scale for generation. Higher values follow the prior more closely",
        ge=0.0,
        le=30.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Number of denoising steps",
        ge=1,
        le=100,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Enable CPU offload to reduce VRAM usage.",
    )

    _prior_pipeline: FluxPriorReduxPipeline | None = None
    _pipeline: FluxPipeline | None = None

    @classmethod
    def get_basic_fields(cls):
        return [
            "prior_redux_model",
            "flux_model",
            "image",
            "guidance_scale",
            "num_inference_steps",
            "seed",
        ]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls) -> str:
        return "Flux Prior Redux"

    def get_prior_model_id(self) -> str:
        if self.prior_redux_model.repo_id:
            return self.prior_redux_model.repo_id
        return "black-forest-labs/FLUX.1-Redux-dev"

    def get_flux_model_id(self) -> str:
        if self.flux_model.repo_id:
            return self.flux_model.repo_id
        return "black-forest-labs/FLUX.1-dev"

    async def preload_model(self, context: ProcessingContext):
        hf_token = await context.get_secret("HF_TOKEN")
        if not hf_token:
            model_url = f"https://huggingface.co/{self.get_prior_model_id()}"
            raise ValueError(
                f"Flux Prior Redux is a gated model, please set the HF_TOKEN in Nodetool settings and accept the terms of use for the model: {model_url}"
            )

        log.info(
            f"Loading FLUX Prior Redux pipeline from {self.get_prior_model_id()}..."
        )
        torch_dtype = _select_diffusion_dtype()
        self._prior_pipeline = await self.load_model(
            context=context,
            model_class=FluxPriorReduxPipeline,
            model_id=self.get_prior_model_id(),
            torch_dtype=torch_dtype,
            device="cpu",
        )
        _enable_pytorch2_attention(self._prior_pipeline)
        _apply_vae_optimizations(self._prior_pipeline)

        log.info(f"Loading FLUX pipeline from {self.get_flux_model_id()}...")
        self._pipeline = await self.load_model(
            context=context,
            model_class=FluxPipeline,
            model_id=self.get_flux_model_id(),
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=torch_dtype,
            device="cpu",
        )
        _enable_pytorch2_attention(self._pipeline)
        _apply_vae_optimizations(self._pipeline)

        # Apply CPU offload if enabled
        if self._prior_pipeline is not None and self.enable_cpu_offload:
            self._prior_pipeline.enable_model_cpu_offload()
        if self._pipeline is not None and self.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()

    async def move_to_device(self, device: str):
        if self._prior_pipeline is not None:
            # If CPU offload is enabled, we need to handle device movement differently
            if self.enable_cpu_offload:
                if device == "cpu":
                    self._prior_pipeline.to(device)
                elif device in ["cuda", "mps"]:
                    self._prior_pipeline.enable_model_cpu_offload()
            else:
                try:
                    self._prior_pipeline.to(device)
                except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                    raise ValueError(
                        "VRAM out of memory while moving Flux Prior Redux to device. "
                        "Enable 'CPU offload' in the advanced node properties or reduce image size."
                    ) from e

        if self._pipeline is not None:
            # If CPU offload is enabled, we need to handle device movement differently
            if self.enable_cpu_offload:
                if device == "cpu":
                    self._pipeline.to(device)
                elif device in ["cuda", "mps"]:
                    self._pipeline.enable_model_cpu_offload()
            else:
                try:
                    self._pipeline.to(device)
                except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                    raise ValueError(
                        "VRAM out of memory while moving Flux pipeline to device. "
                        "Enable 'CPU offload' in the advanced node properties or reduce image size."
                    ) from e

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._prior_pipeline is None or self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Set up the generator for reproducibility
        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        input_image = await context.image_to_pil(self.image)

        # Process image through prior redux pipeline
        def _run_prior_redux():
            return self._prior_pipeline(input_image)  # type: ignore

        log.info("Running FLUX Prior Redux pipeline...")
        pipe_prior_output = await asyncio.to_thread(_run_prior_redux)

        # Convert prior output to dict if it's not already
        if isinstance(pipe_prior_output, dict):
            prior_dict = pipe_prior_output
        elif hasattr(pipe_prior_output, "_asdict"):  # NamedTuple
            prior_dict = pipe_prior_output._asdict()
        elif hasattr(pipe_prior_output, "__dict__"):
            prior_dict = pipe_prior_output.__dict__
        else:
            # Try to convert using vars() or create dict from attributes
            try:
                prior_dict = vars(pipe_prior_output)
            except TypeError:
                # Fallback: create dict from common attributes
                prior_dict = {}
                for attr in [
                    "prompt_embeds",
                    "pooled_prompt_embeds",
                    "negative_prompt_embeds",
                    "negative_pooled_prompt_embeds",
                ]:
                    if hasattr(pipe_prior_output, attr):
                        prior_dict[attr] = getattr(pipe_prior_output, attr)

        # Prepare kwargs for the main FluxPipeline
        kwargs = {
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "generator": generator,
            "callback_on_step_end": pipeline_progress_callback(
                self.id, self.num_inference_steps, context
            ),
            **prior_dict,  # Unpack prior redux output
        }

        try:
            output = await self.run_pipeline_in_thread(**kwargs)  # type: ignore
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            raise ValueError(
                "VRAM out of memory while running Flux Prior Redux. "
                "Enable 'CPU offload' in the advanced node properties (if available), "
                "or reduce image size/steps."
            ) from e
        image = output.images[0]

        return await context.image_from_pil(image)
