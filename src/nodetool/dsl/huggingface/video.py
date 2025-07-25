from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AnimateDiffNode(GraphNode):
    """
    Generates animated GIFs using the AnimateDiff pipeline.
    image, animation, generation, AI

    Use cases:
    - Create animated visual content from text descriptions
    - Generate dynamic visual effects for creative projects
    - Produce animated illustrations for digital media
    """

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
        default="masterpiece, bestquality, highlydetailed, ultradetailed, sunset, orange sky, warm lighting, fishing boats, ocean waves seagulls, rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, golden hour, coastal landscape, seaside scenery",
        description="A text prompt describing the desired animation.",
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="bad quality, worse quality",
        description="A text prompt describing what you don't want in the animation.",
    )
    num_frames: int | GraphNode | tuple[GraphNode, str] = Field(
        default=16, description="The number of frames in the animation."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=7.5, description="Scale for classifier-free guidance."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="The number of denoising steps."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=42, description="Seed for the random number generator."
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.video.AnimateDiff"


class CogVideoX(GraphNode):
    """
    Generates videos from text prompts using CogVideoX, a large diffusion transformer model.
    video, generation, AI, text-to-video, transformer, diffusion

    Use cases:
    - Create high-quality videos from text descriptions
    - Generate longer and more consistent videos
    - Produce cinematic content for creative projects
    - Create animated scenes for storytelling
    - Generate video content for marketing and media
    """

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.",
        description="A text prompt describing the desired video.",
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="A text prompt describing what to avoid in the video."
    )
    num_frames: int | GraphNode | tuple[GraphNode, str] = Field(
        default=49,
        description="The number of frames in the video. Must be divisible by 8 + 1 (e.g., 49, 81, 113).",
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=6.0, description="The scale for classifier-free guidance."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=50, description="The number of denoising steps."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=480, description="The height of the generated video in pixels."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=720, description="The width of the generated video in pixels."
    )
    fps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=8, description="Frames per second for the output video."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    max_sequence_length: int | GraphNode | tuple[GraphNode, str] = Field(
        default=226, description="Maximum sequence length in encoded prompt."
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable CPU offload to reduce VRAM usage."
    )
    enable_vae_slicing: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable VAE slicing to reduce VRAM usage."
    )
    enable_vae_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Enable VAE tiling to reduce VRAM usage for large videos.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.video.CogVideoX"


class StableVideoDiffusion(GraphNode):
    """
    Generates a video from a single image using the Stable Video Diffusion model.
    video, generation, AI, image-to-video, stable-diffusion, SD

    Use cases:
    - Create short animations from static images
    - Generate dynamic content for presentations or social media
    - Prototype video ideas from still concept art
    """

    input_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image to generate the video from, resized to 1024x576.",
    )
    num_frames: int | GraphNode | tuple[GraphNode, str] = Field(
        default=14, description="Number of frames to generate."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=25, description="Number of steps per generated frame"
    )
    fps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=7, description="Frames per second for the output video."
    )
    decode_chunk_size: int | GraphNode | tuple[GraphNode, str] = Field(
        default=8, description="Number of frames to decode at once."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=42, description="Random seed for generation."
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.video.StableVideoDiffusion"
