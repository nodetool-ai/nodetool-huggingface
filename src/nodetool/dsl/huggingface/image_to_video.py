from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.huggingface.image_to_video


class Wan_I2V(GraphNode):
    """
    Generates a video from an input image using Wan image-to-video pipelines.
    video, generation, AI, image-to-video, diffusion, Wan

    Use cases:
    - Turn a single image into a dynamic clip with prompt guidance
    - Choose between Wan 2.2 A14B, Wan 2.1 14B 480P, and Wan 2.1 14B 720P models
    """

    WanI2VModel: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.image_to_video.Wan_I2V.WanI2VModel
    )
    input_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image to generate the video from.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="An astronaut walking on the moon, cinematic lighting, high detail",
        description="A text prompt describing the desired video.",
    )
    model_variant: nodetool.nodes.huggingface.image_to_video.Wan_I2V.WanI2VModel = (
        Field(
            default=nodetool.nodes.huggingface.image_to_video.Wan_I2V.WanI2VModel.WAN_2_2_I2V_A14B,
            description="Select the Wan I2V model to use.",
        )
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="A text prompt describing what to avoid in the video."
    )
    num_frames: int | GraphNode | tuple[GraphNode, str] = Field(
        default=81, description="The number of frames in the video."
    )
    guidance_scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=5.0, description="The scale for classifier-free guidance."
    )
    num_inference_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=50, description="The number of denoising steps."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=480, description="The height of the generated video in pixels."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=832, description="The width of the generated video in pixels."
    )
    fps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=16, description="Frames per second for the output video."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    max_sequence_length: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="Maximum sequence length in encoded prompt."
    )
    enable_cpu_offload: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable CPU offload to reduce VRAM usage."
    )
    enable_vae_slicing: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Enable VAE slicing to reduce VRAM usage."
    )
    enable_vae_tiling: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Enable VAE tiling to reduce VRAM usage for large videos.",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_video.Wan_I2V"
