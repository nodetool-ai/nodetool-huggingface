from __future__ import annotations

from typing import Any, TYPE_CHECKING
from enum import Enum
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    HuggingFaceModel,
    HFTextToVideo,
    ImageRef,
    VideoRef,
)
from .huggingface_pipeline import HuggingFacePipelineNode
from nodetool.nodes.huggingface.stable_diffusion_base import (
    ModelVariant,
    _select_diffusion_dtype,
)
from nodetool.workflows.types import NodeProgress

if TYPE_CHECKING:
    import torch
    from diffusers.pipelines.animatediff.pipeline_animatediff import AnimateDiffPipeline
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.models.unets.unet_motion_model import MotionAdapter
    from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
        StableVideoDiffusionPipeline,
    )
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
    from diffusers.pipelines.wan.pipeline_wan import WanPipeline


class CogVideoX(HuggingFacePipelineNode):
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

    prompt: str = Field(
        default="A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.",
        description="A text prompt describing the desired video.",
    )
    negative_prompt: str = Field(
        default="",
        description="A text prompt describing what to avoid in the video.",
    )
    num_frames: int = Field(
        default=49,
        description="The number of frames in the video. Must be divisible by 8 + 1 (e.g., 49, 81, 113).",
        ge=49,
        le=113,
    )
    guidance_scale: float = Field(
        default=6.0,
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
    height: int = Field(
        default=480,
        description="The height of the generated video in pixels.",
        ge=256,
        le=1024,
    )
    width: int = Field(
        default=720,
        description="The width of the generated video in pixels.",
        ge=256,
        le=1024,
    )
    fps: int = Field(
        default=8,
        description="Frames per second for the output video.",
        ge=1,
        le=30,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )
    max_sequence_length: int = Field(
        default=226,
        description="Maximum sequence length in encoded prompt.",
        ge=1,
        le=512,
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
        description="Enable VAE tiling to reduce VRAM usage for large videos.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(
                repo_id="THUDM/CogVideoX-2b",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HFTextToVideo(
                repo_id="THUDM/CogVideoX-5b",
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
        return "CogVideoX"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "num_frames", "height", "width"]

    def get_model_id(self) -> str:
        return "THUDM/CogVideoX-2b"

    async def preload_model(self, context: ProcessingContext):
        import torch
        from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline

        torch_dtype = _select_diffusion_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_class=CogVideoXPipeline,
            model_id=self.get_model_id(),
            torch_dtype=torch_dtype,
            device="cpu",
        )

        # Apply memory optimization settings
        if self._pipeline is not None:
            if self.enable_cpu_offload:
                self._pipeline.enable_model_cpu_offload()

            if self.enable_vae_slicing:
                self._pipeline.vae.enable_slicing()

            if self.enable_vae_tiling:
                self._pipeline.vae.enable_tiling()

    async def move_to_device(self, device: str):
        if self._pipeline is not None and not self.enable_cpu_offload:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> VideoRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        def callback_on_step_end(
            step: int, timestep: int, callback_kwargs: dict
        ) -> None:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )

        # Generate the video
        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_frames=self.num_frames,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            height=self.height,
            width=self.width,
            generator=generator,
            max_sequence_length=self.max_sequence_length,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        frames = output.frames[0]  # type: ignore

        return await context.video_from_numpy(frames, fps=self.fps)  # type: ignore


class Wan_T2V(HuggingFacePipelineNode):
    """
    Generates videos from text prompts using Wan text-to-video pipeline.
    video, generation, AI, text-to-video, diffusion, Wan

    Use cases:
    - Create high-quality videos from text descriptions
    - Efficient 1.3B model for consumer GPUs or 14B for maximum quality
    """

    class WanModel(str, Enum):
        WAN_2_2_T2V_A14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        WAN_2_1_T2V_14B = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        WAN_2_2_TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    prompt: str = Field(
        default="A robot standing on a mountain top at sunset, cinematic lighting, high detail",
        description="A text prompt describing the desired video.",
    )
    model_variant: WanModel = Field(
        default=WanModel.WAN_2_2_T2V_A14B,
        description="Select the Wan model to use.",
    )
    negative_prompt: str = Field(
        default="",
        description="A text prompt describing what to avoid in the video.",
    )
    num_frames: int = Field(
        default=49,
        description="The number of frames in the video.",
        ge=16,
        le=129,
    )
    guidance_scale: float = Field(
        default=5.0,
        description="The scale for classifier-free guidance.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=30,
        description="The number of denoising steps.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=480,
        description="The height of the generated video in pixels.",
        ge=256,
        le=1080,
    )
    width: int = Field(
        default=720,
        description="The width of the generated video in pixels.",
        ge=256,
        le=1920,
    )
    fps: int = Field(
        default=16,
        description="Frames per second for the output video.",
        ge=1,
        le=60,
    )
    seed: int = Field(
        default=-1,
        description="Seed for the random number generator. Use -1 for a random seed.",
        ge=-1,
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum sequence length in encoded prompt.",
        ge=64,
        le=1024,
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
        default=False,
        description="Enable VAE tiling to reduce VRAM usage for large videos.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(
                repo_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
            HFTextToVideo(
                repo_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
            HFTextToVideo(
                repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Wan (Text-to-Video)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "num_frames", "height", "width", "model_variant"]

    def get_model_id(self) -> str:
        return self.model_variant.value

    async def preload_model(self, context: ProcessingContext):
        import torch
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
        from diffusers.pipelines.wan.pipeline_wan import WanPipeline

        vae = await self.load_model(
            context=context,
            model_class=AutoencoderKLWan,
            model_id=self.get_model_id(),
            subfolder="vae",
            torch_dtype=torch.float32,
            # Some Wan releases have VAE weight shapes that differ from the
            # default config; allow loading by ignoring mismatches as suggested
            # by diffusers docs/errors.
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )
        torch_dtype = _select_diffusion_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_class=WanPipeline,
            model_id=self.get_model_id(),
            torch_dtype=torch_dtype,
            device="cpu",
            vae=vae,
        )

        if self._pipeline is not None:
            if self.enable_cpu_offload and hasattr(
                self._pipeline, "enable_model_cpu_offload"
            ):
                self._pipeline.enable_model_cpu_offload()  # type: ignore
            # VAE memory helpers
            if self.enable_vae_slicing and hasattr(self._pipeline, "vae"):
                try:
                    self._pipeline.vae.enable_slicing()  # type: ignore
                except Exception:
                    pass
            if self.enable_vae_tiling and hasattr(self._pipeline, "vae"):
                try:
                    self._pipeline.vae.enable_tiling()  # type: ignore
                except Exception:
                    pass

    async def move_to_device(self, device: str):
        if self._pipeline is not None and not self.enable_cpu_offload:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> VideoRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        def callback_on_step_end(
            pipeline: Any, step_index: int, timesteps: int, callback_kwargs: dict
        ) -> dict:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step_index,
                    total=self.num_inference_steps,
                )
            )
            return callback_kwargs

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_frames=self.num_frames,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            height=self.height,
            width=self.width,
            generator=generator,
            max_sequence_length=self.max_sequence_length,
            callback_on_step_end=callback_on_step_end,  # type: ignore
        )

        return await context.video_from_frames(output.frames[0], fps=self.fps)  # type: ignore
