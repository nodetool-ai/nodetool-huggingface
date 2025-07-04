from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AutoPipelineText2Image(GraphNode):
    """
    Generates images from text prompts using AutoPipeline for automatic pipeline selection.
    image, generation, AI, text-to-image, auto

    Use cases:
    - Automatic selection of the best pipeline for a given model
    - Flexible image generation without pipeline-specific knowledge
    - Quick prototyping with various text-to-image models
    - Streamlined workflow for different model architectures
    """

    model: types.HFTextToImage | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFTextToImage(
            type="hf.text_to_image",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model to use for text-to-image generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="A cat holding a sign that says hello world",
        description="A text prompt describing the desired image.",
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="A text prompt describing what to avoid in the image."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=50, description="The number of denoising steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5, description="The scale for classifier-free guidance."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="The width of the generated image."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="The height of the generated image."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.AutoPipelineText2Image"


import nodetool.nodes.huggingface.text_to_image
import nodetool.nodes.huggingface.text_to_image


class FluxText2Image(GraphNode):
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

    FluxVariant: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.text_to_image.FluxVariant
    )
    QuantizationMethod: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.text_to_image.QuantizationMethod
    )
    variant: nodetool.nodes.huggingface.text_to_image.FluxVariant = Field(
        default=nodetool.nodes.huggingface.text_to_image.FluxVariant.SCHNELL,
        description="The FLUX model variant to use.",
    )
    quantization: nodetool.nodes.huggingface.text_to_image.QuantizationMethod = Field(
        default=nodetool.nodes.huggingface.text_to_image.QuantizationMethod.NONE,
        description="Quantization method to reduce memory usage.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="A cat holding a sign that says hello world",
        description="A text prompt describing the desired image.",
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0,
        description="The scale for classifier-free guidance. Use 0.0 for schnell, 3.5 for dev.",
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1360, description="The width of the generated image."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=768, description="The height of the generated image."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4,
        description="The number of denoising steps. Use 4 for schnell, 50 for dev.",
    )
    max_sequence_length: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512,
        description="Maximum sequence length for the prompt. Use 256 for schnell, 512 for dev.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    enable_memory_efficient_attention: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Enable memory efficient attention to reduce VRAM usage.",
    )
    enable_vae_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable VAE tiling to reduce VRAM usage for large images.",
    )
    enable_vae_slicing: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Enable VAE slicing to reduce VRAM usage."
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Enable CPU offload to reduce VRAM usage."
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.FluxText2Image"


import nodetool.nodes.huggingface.text_to_image


class LoadTextToImageModel(GraphNode):
    """
    Load HuggingFace model for image-to-image generation from a repo_id.

    Use cases:
    - Loads a pipeline directly from a repo_id
    - Used for AutoPipelineForImage2Image
    """

    ModelVariant: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.text_to_image.ModelVariant
    )
    repo_id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The repository ID of the model to use for image-to-image generation.",
    )
    variant: nodetool.nodes.huggingface.text_to_image.ModelVariant = Field(
        default=nodetool.nodes.huggingface.text_to_image.ModelVariant.DEFAULT,
        description="The variant of the model to use for text-to-image generation.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.LoadTextToImageModel"


import nodetool.nodes.huggingface.stable_diffusion_base
import nodetool.nodes.huggingface.stable_diffusion_base


class StableDiffusion(GraphNode):
    """
    Generates images from text prompts using Stable Diffusion.
    image, generation, AI, text-to-image, SD

    Use cases:
    - Creating custom illustrations for various projects
    - Generating concept art for creative endeavors
    - Producing unique visual content for marketing materials
    - Exploring AI-generated art for personal or professional use
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
        default="(blurry, low quality, deformed, mutated, bad anatomy, extra limbs, bad proportions, text, watermark, grainy, pixelated, disfigured face, missing fingers, cropped image, bad lighting",
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
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="Width of the generated image."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="Height of the generated image"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.StableDiffusion"


import nodetool.nodes.huggingface.stable_diffusion_base


class StableDiffusionXL(GraphNode):
    """
    Generates images from text prompts using Stable Diffusion XL.
    image, generation, AI, text-to-image, SDXL

    Use cases:
    - Creating custom illustrations for marketing materials
    - Generating concept art for game and film development
    - Producing unique stock imagery for websites and publications
    - Visualizing interior design concepts for clients
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
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Seed for the random number generator."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of inference steps."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.0, description="Guidance scale for generation."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Width of the generated image."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Height of the generated image"
    )
    scheduler: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionXLBase.StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
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

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.StableDiffusionXL"
