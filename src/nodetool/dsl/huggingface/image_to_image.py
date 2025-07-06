from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ImageToImage(GraphNode):
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

    model: types.HFImageToImage | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFImageToImage(
            type="hf.image_to_image",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The HuggingFace model to use for image-to-image generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="A beautiful landscape with mountains and a lake at sunset",
        description="Text prompt describing the desired image transformation.",
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Text prompt describing what should not appear in the generated image.",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image to transform",
    )
    strength: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.8,
        description="Strength of the transformation. Higher values allow for more deviation from the original image.",
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of denoising steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5,
        description="Guidance scale for generation. Higher values follow the prompt more closely.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.ImageToImage"


class Inpaint(GraphNode):
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

    model: types.HFImageToImage | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFImageToImage(
            type="hf.image_to_image",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The HuggingFace model to use for inpainting.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Text prompt describing what should be generated in the masked area.",
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Text prompt describing what should not appear in the generated content.",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image to inpaint",
    )
    mask_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The mask image indicating areas to be inpainted (white areas will be inpainted)",
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of denoising steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5,
        description="Guidance scale for generation. Higher values follow the prompt more closely.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.Inpaint"


import nodetool.nodes.huggingface.image_to_image


class LoadImageToImageModel(GraphNode):
    """
    Load HuggingFace model for image-to-image generation from a repo_id.

    Use cases:
    - Loads a pipeline directly from a repo_id
    - Used for ImageToImage node
    """

    ModelVariant: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.image_to_image.ModelVariant
    )
    repo_id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The repository ID of the model to use for image-to-image generation.",
    )
    variant: nodetool.nodes.huggingface.image_to_image.ModelVariant = Field(
        default=nodetool.nodes.huggingface.image_to_image.ModelVariant.FP16,
        description="The variant of the model to use for image-to-image generation.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.LoadImageToImageModel"


class OmniGenNode(GraphNode):
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

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="A realistic photo of a young woman sitting on a sofa, holding a book and facing the camera.",
        description="The text prompt for image generation. Use <img><|image_1|></img> placeholders to reference input images.",
    )
    input_images: list[types.ImageRef] | GraphNode | tuple[GraphNode, str] = Field(
        default=[],
        description="List of input images to use for editing or as reference. Referenced in prompt using <img><|image_1|></img>, <img><|image_2|></img>, etc.",
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Height of the generated image."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Width of the generated image."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=2.5,
        description="Guidance scale for generation. Higher values follow the prompt more closely.",
    )
    img_guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.6, description="Image guidance scale when using input images."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of denoising steps."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    use_input_image_size_as_output: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="If True, use the input image size as output size. Recommended for image editing.",
    )
    max_input_image_size: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024,
        description="Maximum input image size. Smaller values reduce memory usage but may affect quality.",
    )
    enable_model_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable CPU offload to reduce memory usage when using multiple images.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.OmniGen"


class RealESRGANNode(GraphNode):
    """
    Performs image super-resolution using the RealESRGAN model.
    image, super-resolution, enhancement, huggingface

    Use cases:
    - Enhance low-resolution images
    - Improve image quality for printing or display
    - Upscale images for better detail
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image to transform",
    )
    model: types.HFRealESRGAN | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFRealESRGAN(
            type="hf.real_esrgan",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The RealESRGAN model to use for image super-resolution",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.RealESRGAN"


import nodetool.nodes.huggingface.stable_diffusion_base
import nodetool.nodes.huggingface.stable_diffusion_base


class StableDiffusionControlNetImg2ImgNode(GraphNode):
    """
    Transforms existing images using Stable Diffusion with ControlNet guidance.
    image, generation, image-to-image, controlnet, SD, style-transfer, ipadapter

    Use cases:
    - Modify existing images with precise control over composition and structure
    - Apply specific styles or concepts to photographs or artwork with guided transformations
    - Create variations of existing visual content while maintaining certain features
    - Enhance image editing capabilities with AI-guided transformations
    """

    StableDiffusionScheduler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    )
    StableDiffusionUpscaler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler
    )
    model: types.HFStableDiffusion | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFStableDiffusion(
            type="hf.stable_diffusion",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model to use for image generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt for image generation."
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of denoising steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5, description="Guidance scale for generation."
    )
    scheduler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    loras: list[types.HFLoraSDConfig] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The LoRA models to use for image processing"
    )
    ip_adapter_model: types.HFIPAdapter | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFIPAdapter(
            type="hf.ip_adapter",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The IP adapter model to use for image processing",
    )
    ip_adapter_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="When provided the image will be fed into the IP adapter",
    )
    ip_adapter_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="The strength of the IP adapter"
    )
    pag_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    detail_level: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5,
        description="Level of detail for the hi-res pass. 0.0 is low detail, 1.0 is high detail.",
    )
    enable_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )
    upscaler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler.NONE,
        description="The upscaler to use for 2-pass generation.",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image to be transformed.",
    )
    strength: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="Similarity to the input image"
    )
    controlnet: types.HFControlNet | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFControlNet(
            type="hf.controlnet",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The ControlNet model to use for guidance.",
    )
    control_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The control image to guide the transformation.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.StableDiffusionControlNetImg2Img"


import nodetool.nodes.huggingface.stable_diffusion_base
import nodetool.nodes.huggingface.stable_diffusion_base
import nodetool.nodes.huggingface.image_to_image


class StableDiffusionControlNetInpaintNode(GraphNode):
    """
    Performs inpainting on images using Stable Diffusion with ControlNet guidance.
    image, inpainting, controlnet, SD, style-transfer, ipadapter

    Use cases:
    - Remove unwanted objects from images with precise control
    - Fill in missing parts of images guided by control images
    - Modify specific areas of images while preserving the rest and maintaining structure
    """

    StableDiffusionScheduler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    )
    StableDiffusionUpscaler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler
    )
    StableDiffusionControlNetModel: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.image_to_image.StableDiffusionControlNetModel
    )
    model: types.HFStableDiffusion | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFStableDiffusion(
            type="hf.stable_diffusion",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model to use for image generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt for image generation."
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of denoising steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5, description="Guidance scale for generation."
    )
    scheduler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    loras: list[types.HFLoraSDConfig] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The LoRA models to use for image processing"
    )
    ip_adapter_model: types.HFIPAdapter | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFIPAdapter(
            type="hf.ip_adapter",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The IP adapter model to use for image processing",
    )
    ip_adapter_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="When provided the image will be fed into the IP adapter",
    )
    ip_adapter_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="The strength of the IP adapter"
    )
    pag_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    detail_level: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5,
        description="Level of detail for the hi-res pass. 0.0 is low detail, 1.0 is high detail.",
    )
    enable_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )
    upscaler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler.NONE,
        description="The upscaler to use for 2-pass generation.",
    )
    controlnet: (
        nodetool.nodes.huggingface.image_to_image.StableDiffusionControlNetModel
    ) = Field(
        default=nodetool.nodes.huggingface.image_to_image.StableDiffusionControlNetModel.INPAINT,
        description="The ControlNet model to use for guidance.",
    )
    init_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The initial image to be inpainted.",
    )
    mask_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The mask image indicating areas to be inpainted.",
    )
    control_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The control image to guide the inpainting process.",
    )
    controlnet_conditioning_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="The scale for ControlNet conditioning."
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.StableDiffusionControlNetInpaint"


import nodetool.nodes.huggingface.stable_diffusion_base
import nodetool.nodes.huggingface.stable_diffusion_base


class StableDiffusionControlNetNode(GraphNode):
    """
    Generates images using Stable Diffusion with ControlNet guidance.
    image, generation, text-to-image, controlnet, SD

    Use cases:
    - Generate images with precise control over composition and structure
    - Create variations of existing images while maintaining specific features
    - Artistic image generation with guided outputs
    """

    StableDiffusionScheduler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    )
    StableDiffusionUpscaler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler
    )
    model: types.HFStableDiffusion | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFStableDiffusion(
            type="hf.stable_diffusion",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model to use for image generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt for image generation."
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of denoising steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5, description="Guidance scale for generation."
    )
    scheduler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    loras: list[types.HFLoraSDConfig] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The LoRA models to use for image processing"
    )
    ip_adapter_model: types.HFIPAdapter | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFIPAdapter(
            type="hf.ip_adapter",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The IP adapter model to use for image processing",
    )
    ip_adapter_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="When provided the image will be fed into the IP adapter",
    )
    ip_adapter_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="The strength of the IP adapter"
    )
    pag_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    detail_level: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5,
        description="Level of detail for the hi-res pass. 0.0 is low detail, 1.0 is high detail.",
    )
    enable_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )
    upscaler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler.NONE,
        description="The upscaler to use for 2-pass generation.",
    )
    controlnet: types.HFControlNet | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFControlNet(
            type="hf.controlnet",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The ControlNet model to use for guidance.",
    )
    control_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The control image to guide the generation process.",
    )
    controlnet_conditioning_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0, description="The scale for ControlNet conditioning."
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.StableDiffusionControlNet"


import nodetool.nodes.huggingface.stable_diffusion_base
import nodetool.nodes.huggingface.stable_diffusion_base


class StableDiffusionImg2ImgNode(GraphNode):
    """
    Transforms existing images based on text prompts using Stable Diffusion.
    image, generation, image-to-image, SD, img2img, style-transfer, ipadapter

    Use cases:
    - Modifying existing images to fit a specific style or theme
    - Enhancing or altering photographs
    - Creating variations of existing artwork
    - Applying text-guided edits to images
    """

    StableDiffusionScheduler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    )
    StableDiffusionUpscaler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler
    )
    model: types.HFStableDiffusion | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFStableDiffusion(
            type="hf.stable_diffusion",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model to use for image generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt for image generation."
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of denoising steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5, description="Guidance scale for generation."
    )
    scheduler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    loras: list[types.HFLoraSDConfig] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The LoRA models to use for image processing"
    )
    ip_adapter_model: types.HFIPAdapter | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFIPAdapter(
            type="hf.ip_adapter",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The IP adapter model to use for image processing",
    )
    ip_adapter_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="When provided the image will be fed into the IP adapter",
    )
    ip_adapter_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="The strength of the IP adapter"
    )
    pag_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    detail_level: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5,
        description="Level of detail for the hi-res pass. 0.0 is low detail, 1.0 is high detail.",
    )
    enable_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )
    upscaler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler.NONE,
        description="The upscaler to use for 2-pass generation.",
    )
    init_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The initial image for Image-to-Image generation.",
    )
    strength: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.8,
        description="Strength for Image-to-Image generation. Higher values allow for more deviation from the original image.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.StableDiffusionImg2Img"


import nodetool.nodes.huggingface.stable_diffusion_base
import nodetool.nodes.huggingface.stable_diffusion_base


class StableDiffusionInpaintNode(GraphNode):
    """
    Performs inpainting on images using Stable Diffusion.
    image, inpainting, SD

    Use cases:
    - Remove unwanted objects from images
    - Fill in missing parts of images
    - Modify specific areas of images while preserving the rest
    """

    StableDiffusionScheduler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    )
    StableDiffusionUpscaler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler
    )
    model: types.HFStableDiffusion | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFStableDiffusion(
            type="hf.stable_diffusion",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model to use for image generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt for image generation."
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of denoising steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5, description="Guidance scale for generation."
    )
    scheduler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    loras: list[types.HFLoraSDConfig] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The LoRA models to use for image processing"
    )
    ip_adapter_model: types.HFIPAdapter | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFIPAdapter(
            type="hf.ip_adapter",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The IP adapter model to use for image processing",
    )
    ip_adapter_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="When provided the image will be fed into the IP adapter",
    )
    ip_adapter_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="The strength of the IP adapter"
    )
    pag_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    detail_level: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5,
        description="Level of detail for the hi-res pass. 0.0 is low detail, 1.0 is high detail.",
    )
    enable_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )
    upscaler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionUpscaler.NONE,
        description="The upscaler to use for 2-pass generation.",
    )
    init_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The initial image to be inpainted.",
    )
    mask_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The mask image indicating areas to be inpainted.",
    )
    strength: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.8,
        description="Strength for inpainting. Higher values allow for more deviation from the original image.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.StableDiffusionInpaint"


import nodetool.nodes.huggingface.stable_diffusion_base


class StableDiffusionUpscale(GraphNode):
    """
    Upscales an image using Stable Diffusion 4x upscaler.
    image, upscaling, stable-diffusion, SD

    Use cases:
    - Enhance low-resolution images
    - Improve image quality for printing or display
    - Create high-resolution versions of small images
    """

    StableDiffusionScheduler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt for image generation."
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of upscaling steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5, description="Guidance scale for generation."
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The initial image for Image-to-Image generation.",
    )
    scheduler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionScheduler.HeunDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    enable_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Enable tiling to save VRAM"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.StableDiffusionUpscale"


import nodetool.nodes.huggingface.stable_diffusion_base


class StableDiffusionXLControlNetNode(GraphNode):
    """
    Transforms existing images using Stable Diffusion XL with ControlNet guidance.
    image, generation, image-to-image, controlnet, SDXL

    Use cases:
    - Modify existing images with precise control over composition and structure
    - Apply specific styles or concepts to photographs or artwork with guided transformations
    - Create variations of existing visual content while maintaining certain features
    - Enhance image editing capabilities with AI-guided transformations
    """

    StableDiffusionScheduler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler
    )
    model: types.HFStableDiffusionXL | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFStableDiffusionXL(
            type="hf.stable_diffusion_xl",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The Stable Diffusion XL model to use for generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt for image generation."
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Width of the generated image."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Height of the generated image"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Seed for the random number generator."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of inference steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.0, description="Guidance scale for generation."
    )
    scheduler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    pag_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    loras: list[types.HFLoraSDXLConfig] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The LoRA models to use for image processing"
    )
    lora_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="Strength of the LoRAs"
    )
    ip_adapter_model: types.HFIPAdapter | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFIPAdapter(
            type="hf.ip_adapter",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The IP adapter model to use for image processing",
    )
    ip_adapter_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="When provided the image will be fed into the IP adapter",
    )
    ip_adapter_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="Strength of the IP adapter image"
    )
    enable_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )
    init_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The initial image for Image-to-Image generation.",
    )
    strength: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.8,
        description="Strength for Image-to-Image generation. Higher values allow for more deviation from the original image.",
    )
    controlnet: types.HFControlNet | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFControlNet(
            type="hf.controlnet",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The ControlNet model to use for guidance.",
    )
    control_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The control image to guide the transformation.",
    )
    controlnet_conditioning_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0, description="The scale for ControlNet conditioning."
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.StableDiffusionXLControlNet"


import nodetool.nodes.huggingface.stable_diffusion_base


class StableDiffusionXLImg2Img(GraphNode):
    """
    Transforms existing images based on text prompts using Stable Diffusion XL.
    image, generation, image-to-image, SDXL, style-transfer, ipadapter

    Use cases:
    - Modifying existing images to fit a specific style or theme
    - Enhancing or altering photographs
    - Creating variations of existing artwork
    - Applying text-guided edits to images
    """

    StableDiffusionScheduler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler
    )
    model: types.HFStableDiffusionXL | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFStableDiffusionXL(
            type="hf.stable_diffusion_xl",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The Stable Diffusion XL model to use for generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt for image generation."
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Width of the generated image."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Height of the generated image"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Seed for the random number generator."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of inference steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.0, description="Guidance scale for generation."
    )
    scheduler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    pag_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    loras: list[types.HFLoraSDXLConfig] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The LoRA models to use for image processing"
    )
    lora_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="Strength of the LoRAs"
    )
    ip_adapter_model: types.HFIPAdapter | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFIPAdapter(
            type="hf.ip_adapter",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The IP adapter model to use for image processing",
    )
    ip_adapter_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="When provided the image will be fed into the IP adapter",
    )
    ip_adapter_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="Strength of the IP adapter image"
    )
    enable_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )
    init_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The initial image for Image-to-Image generation.",
    )
    strength: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.8,
        description="Strength for Image-to-Image generation. Higher values allow for more deviation from the original image.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.StableDiffusionXLImg2Img"


import nodetool.nodes.huggingface.stable_diffusion_base


class StableDiffusionXLInpainting(GraphNode):
    """
    Performs inpainting on images using Stable Diffusion XL.
    image, inpainting, SDXL

    Use cases:
    - Remove unwanted objects from images
    - Fill in missing parts of images
    - Modify specific areas of images while preserving the rest
    """

    StableDiffusionScheduler: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler
    )
    model: types.HFStableDiffusionXL | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFStableDiffusionXL(
            type="hf.stable_diffusion_xl",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The Stable Diffusion XL model to use for generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt for image generation."
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Width of the generated image."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Height of the generated image"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Seed for the random number generator."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of inference steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.0, description="Guidance scale for generation."
    )
    scheduler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    pag_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    loras: list[types.HFLoraSDXLConfig] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The LoRA models to use for image processing"
    )
    lora_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="Strength of the LoRAs"
    )
    ip_adapter_model: types.HFIPAdapter | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFIPAdapter(
            type="hf.ip_adapter",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The IP adapter model to use for image processing",
    )
    ip_adapter_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="When provided the image will be fed into the IP adapter",
    )
    ip_adapter_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="Strength of the IP adapter image"
    )
    enable_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The initial image to be inpainted.",
    )
    mask_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The mask image indicating areas to be inpainted.",
    )
    strength: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.8,
        description="Strength for inpainting. Higher values allow for more deviation from the original image.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.StableDiffusionXLInpainting"


class Swin2SR(GraphNode):
    """
    Performs image super-resolution using the Swin2SR model.
    image, super-resolution, enhancement, huggingface

    Use cases:
    - Enhance low-resolution images
    - Improve image quality for printing or display
    - Upscale images for better detail
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image to transform",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The text prompt to guide the image transformation (if applicable)",
    )
    model: types.HFImageToImage | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFImageToImage(
            type="hf.image_to_image",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for image super-resolution",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_image.Swin2SR"
