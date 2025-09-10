from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Chroma(GraphNode):
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

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done.",
        description="A text prompt describing the desired image.",
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors",
        description="A text prompt describing what to avoid in the image.",
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.0, description="The scale for classifier-free guidance."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=40, description="The number of denoising steps."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The height of the generated image."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The width of the generated image."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    max_sequence_length: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="Maximum sequence length to use with the prompt."
    )
    ip_adapter_image: (
        nodetool.metadata.types.ImageRef | None | GraphNode | tuple[GraphNode, str]
    ) = Field(
        default=None, description="Optional image input for IP Adapter style control."
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable CPU offload to reduce VRAM usage."
    )
    enable_vae_slicing: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable VAE slicing to reduce VRAM usage."
    )
    enable_vae_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Enable VAE tiling to reduce VRAM usage for large images.",
    )
    enable_attention_slicing: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable attention slicing to reduce memory usage."
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.Chroma"


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


class HunyuanDiT(GraphNode):
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

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="一个宇航员在骑马",
        description="A text prompt describing the desired image. Supports both Chinese and English.",
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="A text prompt describing what to avoid in the image."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5, description="The scale for classifier-free guidance."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=50, description="The number of denoising steps."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The width of the generated image."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The height of the generated image."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    use_resolution_binning: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Whether to use resolution binning. Maps input resolution to closest standard resolution.",
    )
    original_size: tuple[int, int] | GraphNode | tuple[GraphNode, str] = Field(
        default=(1024, 1024),
        description="The original size of the image used to calculate time IDs.",
    )
    target_size: tuple[int, int] | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None,
        description="The target size of the image used to calculate time IDs. If None, uses (width, height).",
    )
    crops_coords_top_left: tuple[int, int] | GraphNode | tuple[GraphNode, str] = Field(
        default=(0, 0),
        description="The top-left coordinates of the crop used to calculate time IDs.",
    )
    guidance_rescale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description="Rescale the noise according to guidance_rescale."
    )
    enable_memory_optimization: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Enable memory optimization with T5 encoder quantization.",
    )
    enable_forward_chunking: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable forward chunking to reduce memory usage at the cost of inference speed.",
    )
    forward_chunk_size: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Chunk size for forward chunking."
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.HunyuanDiT"


class Kolors(GraphNode):
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

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default='A ladybug photo, macro, zoom, high quality, film, holding a sign that says "可图"',
        description="A text prompt describing the desired image. Supports both Chinese and English.",
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="A text prompt describing what to avoid in the image."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=6.5, description="The scale for classifier-free guidance."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="The number of denoising steps."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The width of the generated image."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The height of the generated image."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    max_sequence_length: int | GraphNode | tuple[GraphNode, str] = Field(
        default=256, description="Maximum sequence length for the prompt."
    )
    use_dpm_solver: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Whether to use DPMSolverMultistepScheduler with Karras sigmas for better quality.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.Kolors"


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


class LuminaT2X(GraphNode):
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

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Upper body of a young woman in a Victorian-era outfit with brass goggles and leather straps. Background shows an industrial revolution cityscape with smoky skies and tall, metal structures",
        description="A text prompt describing the desired image.",
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="A text prompt describing what to avoid in the image. For Lumina-T2X, this should typically be empty.",
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=4.0, description="The scale for classifier-free guidance."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=30,
        description="The number of denoising steps. Lumina-T2X uses fewer steps for efficient generation.",
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The height of the generated image."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The width of the generated image."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    clean_caption: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Whether to clean the caption before creating embeddings. Requires beautifulsoup4 and ftfy.",
    )
    max_sequence_length: int | GraphNode | tuple[GraphNode, str] = Field(
        default=256, description="Maximum sequence length to use with the prompt."
    )
    scaling_watershed: float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0,
        description="Scaling watershed parameter for improved generation quality.",
    )
    proportional_attn: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Whether to use proportional attention for better quality.",
    )
    enable_quantization: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable quantization for memory efficiency."
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable CPU offload to reduce VRAM usage."
    )
    enable_vae_slicing: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable VAE slicing to reduce VRAM usage."
    )
    enable_vae_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Enable VAE tiling to reduce VRAM usage for large images.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.LuminaT2X"


class QuantoFlux(GraphNode):
    """
    Generates images using FLUX models with Optimum Quanto FP8 quantization for extreme memory efficiency.
    image, generation, AI, text-to-image, flux, quantization, fp8, quanto

    Use cases:
    - Ultra memory-efficient FLUX image generation using FP8 quantization
    - High-quality image generation on lower-end hardware
    - Faster inference with reduced memory footprint
    - Professional image generation with optimized resource usage
    """

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="A cat holding a sign that says hello world",
        description="A text prompt describing the desired image.",
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.5, description="The scale for classifier-free guidance."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The width of the generated image."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The height of the generated image."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=20, description="The number of denoising steps."
    )
    max_sequence_length: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="Maximum sequence length for the prompt."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable CPU offload to reduce VRAM usage."
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.QuantoFlux"


class QwenImage(GraphNode):
    """
    Generates images from text prompts using Qwen-Image via AutoPipelineForText2Image.
    image, generation, AI, text-to-image, qwen

    Use cases:
    - High-quality, general-purpose text-to-image generation
    - Quick prototyping leveraging AutoPipeline
    - Works out-of-the-box with the official Qwen model
    """

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="A cat holding a sign that says hello world",
        description="A text prompt describing the desired image.",
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="A text prompt describing what to avoid in the image."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.5, description="The scale for classifier-free guidance."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=20, description="The number of denoising steps."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The height of the generated image."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="The width of the generated image."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.QwenImage"


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
    StableDiffusionOutputType: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionOutputType
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
    enable_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )
    output_type: (
        nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionOutputType
    ) = Field(
        default=nodetool.nodes.huggingface.stable_diffusion_base.StableDiffusionBaseNode.StableDiffusionOutputType.IMAGE,
        description="The type of output to generate.",
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

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.StableDiffusionXL"


class Text2Image(GraphNode):
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
    pag_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=3.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_image.Text2Image"
