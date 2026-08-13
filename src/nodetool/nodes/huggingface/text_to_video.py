from __future__ import annotations

import importlib
from typing import Any, TypedDict, TYPE_CHECKING
from enum import Enum
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    AudioRef,
    HuggingFaceModel,
    HFTextToVideo,
    ImageRef,
    VideoRef,
)
from .huggingface_pipeline import HuggingFacePipelineNode, select_inference_dtype
from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE
from nodetool.nodes.huggingface.stable_diffusion_base import (
    available_torch_dtype,
    is_mps_device,
    maybe_enable_cpu_offload,
)
from nodetool.workflows.memory_utils import run_gc
from nodetool.workflows.types import NodeProgress
from nodetool.huggingface.video_utils import (
    video_from_frames,
    video_from_frames_with_audio,
)

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
    Generates high-quality videos from text prompts using the CogVideoX diffusion transformer.
    video, generation, AI, text-to-video, transformer, diffusion, cinematic

    Use cases:
    - Create videos from detailed text descriptions
    - Generate cinematic content for creative projects
    - Produce animated scenes for storytelling and marketing
    - Build AI video generation applications
    - Create visual content for social media
    """

    prompt: str = Field(
        default="A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.",
        description="Detailed text description of the video to generate. More descriptive prompts produce better results.",
    )
    negative_prompt: str = Field(
        default="",
        description="Describe what to avoid in the video (e.g., 'blurry, low quality, distorted').",
    )
    num_frames: int = Field(
        default=49,
        description="Total frames in the output. Must be 8n+1 (49, 81, 113). More frames = longer video.",
        ge=49,
        le=113,
    )
    guidance_scale: float = Field(
        default=6.0,
        description="How strongly to follow the prompt. 5-8 is typical; higher = more adherence.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. 50 is recommended; lower for faster generation.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=480,
        description="Output video height in pixels.",
        ge=256,
        le=1024,
    )
    width: int = Field(
        default=720,
        description="Output video width in pixels.",
        ge=256,
        le=1024,
    )
    fps: int = Field(
        default=8,
        description="Frames per second for the output video file.",
        ge=1,
        le=30,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    max_sequence_length: int = Field(
        default=226,
        description="Maximum prompt encoding length. Higher allows longer prompts.",
        ge=1,
        le=512,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )
    enable_vae_slicing: bool = Field(
        default=True,
        description="Process VAE in slices to reduce peak memory usage.",
    )
    enable_vae_tiling: bool = Field(
        default=True,
        description="Process VAE in tiles for large videos. Reduces memory but may affect quality.",
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

        torch_dtype = available_torch_dtype()
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

        frames = output.frames[0]

        run_gc("After CogVideoX inference", log_before_after=False)
        return await video_from_frames(context, frames, fps=self.fps)


class Wan_T2V(HuggingFacePipelineNode):
    """
    Generates videos from text prompts using Wan text-to-video diffusion models.
    video, generation, AI, text-to-video, diffusion, Wan, cinematic

    Use cases:
    - Create high-quality videos from text descriptions
    - Generate cinematic content for creative and commercial projects
    - Produce animated scenes for storytelling and marketing
    - Build AI video generation applications
    - Create visual content for social media and entertainment

    **Note:** Model variants offer different quality/resource tradeoffs. A14B is balanced; 14B offers maximum quality.
    """

    class WanModel(str, Enum):
        WAN_2_2_T2V_A14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        WAN_2_1_T2V_14B = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        WAN_2_2_TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    prompt: str = Field(
        default="A robot standing on a mountain top at sunset, cinematic lighting, high detail",
        description="Detailed text description of the video to generate.",
    )
    model_variant: WanModel = Field(
        default=WanModel.WAN_2_2_T2V_A14B,
        description="The Wan model variant. A14B is balanced; TI2V-5B is smaller/faster.",
    )
    negative_prompt: str = Field(
        default="",
        description="Describe what to avoid in the video (e.g., 'blurry, distorted').",
    )
    num_frames: int = Field(
        default=49,
        description="Total frames in the output video. More frames = longer duration.",
        ge=16,
        le=129,
    )
    guidance_scale: float = Field(
        default=5.0,
        description="How strongly to follow the prompt. 4-7 is typical.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=30,
        description="Denoising steps. 30 is fast; 50+ for higher quality.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=480,
        description="Output video height in pixels.",
        ge=256,
        le=1080,
    )
    width: int = Field(
        default=720,
        description="Output video width in pixels.",
        ge=256,
        le=1920,
    )
    fps: int = Field(
        default=16,
        description="Frames per second for the output video file.",
        ge=1,
        le=60,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum prompt encoding length. Higher allows longer prompts.",
        ge=64,
        le=1024,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )
    enable_vae_slicing: bool = Field(
        default=True,
        description="Process VAE in slices to reduce peak memory usage.",
    )
    enable_vae_tiling: bool = Field(
        default=False,
        description="Process VAE in tiles for large videos. May affect quality.",
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
            HFTextToVideo(
                repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
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
        # Wan models are huge and typically released with bfloat16 weights, so we
        # explicitly request that dtype to keep memory usage manageable.
        torch_dtype = torch.bfloat16
        self._pipeline = await self.load_model(
            context=context,
            model_class=WanPipeline,
            model_id=self.get_model_id(),
            torch_dtype=torch_dtype,
            device="cpu",
            vae=vae,
        )

        if self._pipeline is not None:
            if self.enable_cpu_offload:
                from nodetool.huggingface.memory_utils import (
                    apply_cpu_offload_if_needed,
                )

                # Use model CPU offload as primary, fall back to sequential
                if hasattr(self._pipeline, "enable_model_cpu_offload"):
                    apply_cpu_offload_if_needed(self._pipeline, method="model")
                elif hasattr(self._pipeline, "enable_sequential_cpu_offload"):
                    apply_cpu_offload_if_needed(self._pipeline, method="sequential")

            if hasattr(self._pipeline, "enable_attention_slicing"):
                try:
                    self._pipeline.enable_attention_slicing()
                except Exception:
                    pass

            # VAE memory helpers
            if self.enable_vae_slicing and hasattr(self._pipeline, "vae"):
                try:
                    self._pipeline.vae.enable_slicing()
                except Exception:
                    pass
            if self.enable_vae_tiling and hasattr(self._pipeline, "vae"):
                try:
                    self._pipeline.vae.enable_tiling()
                except Exception:
                    pass

            # Attempt to enable xformers if it is available
            try:
                if hasattr(self._pipeline, "unet") and hasattr(
                    self._pipeline.unet, "enable_xformers_memory_efficient_attention"
                ):
                    self._pipeline.unet.enable_xformers_memory_efficient_attention()
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
            callback_on_step_end=callback_on_step_end,
        )

        run_gc("After Wan T2V inference", log_before_after=False)
        return await video_from_frames(context, output.frames[0], fps=self.fps)


_DIFFUSERS_REPO_ALLOW_PATTERNS = [
    "**/*.safetensors",
    "**/*.json",
    "**/*.txt",
    "**/*.model",
    "*.json",
]


class LTX2(HuggingFacePipelineNode):
    """
    Generates videos from text prompts using Lightricks' LTX-2 diffusion model.
    video, generation, AI, text-to-video, ltx, cinematic

    Use cases:
    - Create high-quality videos from detailed text descriptions
    - Generate cinematic content for creative projects
    - Produce animated scenes for storytelling and marketing
    - Build AI video generation applications
    """

    model: HFTextToVideo = Field(
        default=HFTextToVideo(repo_id="Lightricks/LTX-2"),
        description="The LTX-2 model to use for video generation.",
    )
    prompt: str = Field(
        default="A majestic eagle soaring over snow-capped mountains at sunrise, cinematic, highly detailed.",
        description="Detailed text description of the video to generate.",
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Describe what to avoid in the video.",
    )
    num_frames: int = Field(
        default=121,
        description="Total frames in the output. More frames = longer video.",
        ge=9,
        le=257,
    )
    frame_rate: float = Field(
        default=24.0,
        description="Frame rate the model generates at (also used for the output file).",
        ge=1.0,
        le=60.0,
    )
    guidance_scale: float = Field(
        default=4.0,
        description="How strongly to follow the prompt. 3-5 is typical.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=40,
        description="Denoising steps. 40 is recommended; lower for faster generation.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=512,
        description="Output video height in pixels.",
        ge=256,
        le=1280,
    )
    width: int = Field(
        default=768,
        description="Output video width in pixels.",
        ge=256,
        le=1280,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(
                repo_id="Lightricks/LTX-2",
                allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "num_frames", "height", "width"]

    def get_model_id(self) -> str:
        return self.model.repo_id or "Lightricks/LTX-2"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline

        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(
                f"Model {self.get_model_id()} must be downloaded first from the recommended models"
            )

        self._pipeline = await self.load_model(
            context=context,
            model_class=LTX2Pipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
            local_files_only=True,
        )
        maybe_enable_cpu_offload(self._pipeline, self.enable_cpu_offload)

    async def move_to_device(self, device: str):
        # On MPS we skip offload and load fully onto the device, so move here.
        if self._pipeline is not None and (
            not self.enable_cpu_offload or is_mps_device()
        ):
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> VideoRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        def callback_on_step_end(
            pipeline: Any, step: int, timestep: int, callback_kwargs: dict
        ) -> dict:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )
            return callback_kwargs

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            frame_rate=self.frame_rate,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        # LTX-2 returns video frames under .frames (and may also produce audio).
        frames = output.frames[0] if hasattr(output, "frames") else output[0]
        run_gc("After LTX-2 inference", log_before_after=False)
        return await video_from_frames(context, frames, fps=int(self.frame_rate))


class LTXVideo(HuggingFacePipelineNode):
    """
    Generates high-quality videos from text prompts using LTX-Video diffusion models.
    video, generation, AI, text-to-video, diffusion, LTX, Lightricks, cinematic

    Use cases:
    - Create smooth, high-fidelity videos from detailed text descriptions
    - Generate cinematic and artistic video content
    - Produce short animated clips for social media and creative projects
    - Build AI-driven video generation pipelines
    - Create visual content with fine-grained temporal control

    **Note:** LTX-Video-0.9.5 is recommended for best quality; older versions are also supported.
    """

    class LTXModel(str, Enum):
        LTX_VIDEO_0_9_5 = "Lightricks/LTX-Video-0.9.5"
        LTX_VIDEO_0_9_1 = "Lightricks/LTX-Video-0.9.1"
        LTX_VIDEO = "Lightricks/LTX-Video"

    model_variant: LTXModel = Field(
        default=LTXModel.LTX_VIDEO_0_9_5,
        description="The LTX-Video model variant. 0.9.5 is the latest and recommended.",
    )
    prompt: str = Field(
        default="A serene mountain landscape at sunrise, misty valleys below, golden light on snow-capped peaks, cinematic quality",
        description="Detailed text description of the video to generate.",
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Describe what to avoid in the video.",
    )
    num_frames: int = Field(
        default=161,
        description="Total frames in the output video. Must be 8n+1 (e.g. 65, 97, 129, 161). More frames = longer video.",
        ge=9,
        le=257,
    )
    guidance_scale: float = Field(
        default=3.0,
        description="How strongly to follow the prompt. 3.0 is typical for LTX-Video.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. 50 is recommended; lower for faster generation.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=480,
        description="Output video height in pixels. Best results at 480 or 512.",
        ge=64,
        le=1024,
    )
    width: int = Field(
        default=704,
        description="Output video width in pixels. Best results at 704 or 768.",
        ge=64,
        le=1280,
    )
    frame_rate: int = Field(
        default=25,
        description="Frame rate for the output video file.",
        ge=1,
        le=60,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    max_sequence_length: int = Field(
        default=128,
        description="Maximum prompt encoding length. Higher allows longer prompts.",
        ge=32,
        le=512,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )
    enable_vae_slicing: bool = Field(
        default=True,
        description="Process VAE in slices to reduce peak memory usage.",
    )
    enable_vae_tiling: bool = Field(
        default=False,
        description="Process VAE in tiles for large videos. May affect quality.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(
                repo_id="Lightricks/LTX-Video-0.9.5",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
            HFTextToVideo(
                repo_id="Lightricks/LTX-Video-0.9.1",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
            HFTextToVideo(
                repo_id="Lightricks/LTX-Video",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "LTX-Video (Text-to-Video)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model_variant", "num_frames", "height", "width"]

    def get_model_id(self) -> str:
        return self.model_variant.value

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline

        torch_dtype = available_torch_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_class=LTXPipeline,
            model_id=self.get_model_id(),
            torch_dtype=torch_dtype,
            device="cpu",
        )

        if self._pipeline is not None:
            if self.enable_cpu_offload:
                self._pipeline.enable_model_cpu_offload()
            if self.enable_vae_slicing and hasattr(self._pipeline, "vae"):
                try:
                    self._pipeline.vae.enable_slicing()
                except Exception:
                    pass
            if self.enable_vae_tiling and hasattr(self._pipeline, "vae"):
                try:
                    self._pipeline.vae.enable_tiling()
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
            frame_rate=self.frame_rate,
            generator=generator,
            max_sequence_length=self.max_sequence_length,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        run_gc("After LTX-Video T2V inference", log_before_after=False)
        return await video_from_frames(context, output.frames[0], fps=self.frame_rate)


class LTX2Video(HuggingFacePipelineNode):
    """
    Generates videos from text prompts using LTX-2, the newest Lightricks video generation model.
    video, generation, AI, text-to-video, diffusion, LTX2, Lightricks, cinematic

    Use cases:
    - Create high-quality videos with the latest LTX-2 architecture
    - Generate videos with improved temporal consistency and visual fidelity
    - Produce cinematic content with detailed motion and scene description
    - Build next-generation AI video generation workflows
    - Create visual content with support for audio output

    **Note:** LTX-2 uses a Gemma3 text encoder and requires more VRAM than LTX-Video.
    """

    prompt: str = Field(
        default="A woman with long brown hair smiles warmly in soft golden sunlight, cinematic close-up shot, high quality",
        description="Detailed text description of the video to generate.",
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Describe what to avoid in the video.",
    )
    num_frames: int = Field(
        default=121,
        description="Total frames in the output video. 121 is the default (about 5 seconds at 24fps).",
        ge=9,
        le=257,
    )
    guidance_scale: float = Field(
        default=4.0,
        description="How strongly to follow the prompt. 4.0 is recommended for LTX-2.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=40,
        description="Denoising steps. 40 is recommended; lower for faster generation.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=512,
        description="Output video height in pixels.",
        ge=64,
        le=1024,
    )
    width: int = Field(
        default=768,
        description="Output video width in pixels.",
        ge=64,
        le=1280,
    )
    frame_rate: float = Field(
        default=24.0,
        description="Frame rate for the output video file.",
        ge=1.0,
        le=60.0,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    max_sequence_length: int = Field(
        default=1024,
        description="Maximum prompt encoding length. LTX-2 supports long prompts up to 1024 tokens.",
        ge=64,
        le=2048,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )
    enable_vae_slicing: bool = Field(
        default=True,
        description="Process VAE in slices to reduce peak memory usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(
                repo_id="Lightricks/LTX-2",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2 (Text-to-Video)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "num_frames", "height", "width"]

    def get_model_id(self) -> str:
        return "Lightricks/LTX-2"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline

        import torch

        self._pipeline = await self.load_model(
            context=context,
            model_class=LTX2Pipeline,
            model_id=self.get_model_id(),
            torch_dtype=torch.bfloat16,
            device="cpu",
        )

        if self._pipeline is not None:
            if self.enable_cpu_offload:
                self._pipeline.enable_model_cpu_offload()
            if self.enable_vae_slicing and hasattr(self._pipeline, "vae"):
                try:
                    self._pipeline.vae.enable_slicing()
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
            frame_rate=self.frame_rate,
            generator=generator,
            max_sequence_length=self.max_sequence_length,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="np",
        )

        fps = int(self.frame_rate)
        run_gc("After LTX-2 T2V inference", log_before_after=False)
        return await video_from_frames(context, output.frames[0], fps=fps)


class Kandinsky5Video(HuggingFacePipelineNode):
    """
    Generates videos from text prompts using Kandinsky 5.0 video models.
    video, generation, AI, text-to-video, kandinsky, cinematic

    Use cases:
    - Create high-quality short videos from text descriptions
    - Generate cinematic content with strong multilingual understanding
    - Produce animated scenes for creative projects
    - Build AI video generation applications
    """

    model: HFTextToVideo = Field(
        default=HFTextToVideo(
            repo_id="kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
        ),
        description="The Kandinsky 5.0 video model to use.",
    )
    prompt: str = Field(
        default="A cat and a dog baking a cake together in a kitchen.",
        description="Detailed text description of the video to generate.",
    )
    negative_prompt: str = Field(
        default="Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        description="Describe what to avoid in the video.",
    )
    num_frames: int = Field(
        default=121,
        description="Total frames in the output. 121 ~= 5 seconds at 24fps.",
        ge=25,
        le=241,
    )
    guidance_scale: float = Field(
        default=5.0,
        description="How strongly to follow the prompt. Use 1.0 for distilled checkpoints.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. 50 is typical; distilled checkpoints use 16.",
        ge=1,
        le=100,
    )
    height: int = Field(
        default=512,
        description="Output video height in pixels.",
        ge=256,
        le=1280,
    )
    width: int = Field(
        default=768,
        description="Output video width in pixels.",
        ge=256,
        le=1280,
    )
    fps: int = Field(
        default=24,
        description="Frames per second for the output video file.",
        ge=1,
        le=60,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        def _m(repo: str) -> HFTextToVideo:
            return HFTextToVideo(
                repo_id=repo, allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS
            )

        return [
            # Pro: 19B high-quality (use enable_cpu_offload on a single GPU).
            _m("kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers"),
            # Lite SFT: 2B, highest quality in its class.
            _m("kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"),
            _m("kandinskylab/Kandinsky-5.0-T2V-Lite-sft-10s-Diffusers"),
            # Lite no-CFG: 2x faster (run with guidance_scale=1.0).
            _m("kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers"),
            _m("kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-10s-Diffusers"),
            # Lite distilled: 16 steps, 6x faster.
            _m("kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers"),
            _m("kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers"),
            # Lite pretrain: base models for research and fine-tuning.
            _m("kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-5s-Diffusers"),
            _m("kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-10s-Diffusers"),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Kandinsky 5.0 Video"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "num_frames", "height", "width"]

    def get_model_id(self) -> str:
        return (
            self.model.repo_id or "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
        )

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.kandinsky5.pipeline_kandinsky import (
            Kandinsky5T2VPipeline,
        )

        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(
                f"Model {self.get_model_id()} must be downloaded first from the recommended models"
            )

        self._pipeline = await self.load_model(
            context=context,
            model_class=Kandinsky5T2VPipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
            local_files_only=True,
        )
        maybe_enable_cpu_offload(self._pipeline, self.enable_cpu_offload)

    async def move_to_device(self, device: str):
        # On MPS we skip offload and load fully onto the device, so move here.
        if self._pipeline is not None and (
            not self.enable_cpu_offload or is_mps_device()
        ):
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> VideoRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        def callback_on_step_end(
            pipeline: Any, step: int, timestep: int, callback_kwargs: dict
        ) -> dict:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )
            return callback_kwargs

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        frames = output.frames[0] if hasattr(output, "frames") else output[0]
        run_gc("After Kandinsky 5.0 Video inference", log_before_after=False)
        return await video_from_frames(context, frames, fps=self.fps)


# MiniMax-H3 generates at a fixed 24 fps and its video VAE only decodes frame
# counts of the form ``17 * n + 5``. The pipeline snaps a request up to the next
# such count, and the resulting duration has to stay between 5 and 15 seconds —
# so 124 (5.17s) is the shortest usable request and 345 (14.375s) the longest.
MINIMAX_H3_REPO_ID = "MiniMaxAI/MiniMax-H3"
MINIMAX_H3_FPS = 24
MINIMAX_H3_MIN_FRAMES = 120
MINIMAX_H3_MAX_FRAMES = 345
MINIMAX_H3_MODULE = "diffusers.modular_pipelines.minimax_h3"


def _load_minimax_h3_runtime() -> tuple[Any, Any, Any]:
    """Load the unreleased Diffusers H3 integration only when an H3 node runs."""
    diffusers = None
    try:
        diffusers = importlib.import_module("diffusers")
        integration = importlib.import_module(MINIMAX_H3_MODULE)
        components_manager = getattr(diffusers, "ComponentsManager")
        modular_pipeline = getattr(diffusers, "ModularPipeline")
    except (ImportError, AttributeError) as exc:
        version = getattr(diffusers, "__version__", "not installed")
        raise ImportError(
            "MiniMax-H3 is not available in the installed Diffusers version "
            f"({version}). Diffusers 0.39 does not include the H3 integration; "
            "install a newer Diffusers release once H3 is published. Other "
            "Hugging Face nodes remain available with Diffusers 0.39."
        ) from exc

    return integration, components_manager, modular_pipeline


def _minimax_h3_recommended_models() -> list[HuggingFaceModel]:
    return [
        HFTextToVideo(
            repo_id=MINIMAX_H3_REPO_ID,
            allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
        ),
    ]


class _MiniMaxH3Base(HuggingFacePipelineNode):
    """Shared loading and output handling for the MiniMax-H3 workflows.

    MiniMax-H3 is a Modular Diffusers integration: one repository holds two
    transformer partitions and three workflows (`t2va`, `fl2va`, `ref2va`), and
    selecting a workflow loads only the partition that workflow needs. The
    checkpoint is guidance-distilled, so there is no negative prompt and no
    guidance scale.
    """

    # Set by the concrete nodes: "fl2va" (text and keyframes, `transformer/`) or
    # "ref2va" (omni-references, `transformer_ref/`).
    _workflow: str = "fl2va"

    prompt: str = Field(
        default="A red fox trotting through a snowy pine forest, snow crunching underfoot",
        description="Text description of the video and its soundtrack.",
    )
    num_frames: int = Field(
        default=124,
        description=(
            "Frames to generate at the fixed 24 fps. Snapped up to the next 17n+5 "
            "the video VAE can decode; the result must stay between 5 and 15 seconds."
        ),
        ge=MINIMAX_H3_MIN_FRAMES,
        le=MINIMAX_H3_MAX_FRAMES,
    )
    height: int = Field(
        default=0,
        description="Output height in pixels, a multiple of 32. Use 0 for the model's own canvas.",
        ge=0,
        le=1344,
    )
    width: int = Field(
        default=0,
        description="Output width in pixels, a multiple of 32. Use 0 for the model's own canvas.",
        ge=0,
        le=1344,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. Counts sigma grid points, so it runs one model evaluation less.",
        ge=1,
        le=100,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description=(
            "Move components on and off the accelerator as the blocks reach them. "
            "The transformer alone is 61.7GB, so this is required on a single GPU."
        ),
    )

    _pipeline: Any = None

    class OutputType(TypedDict):
        video: VideoRef
        audio: AudioRef

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not _MiniMaxH3Base

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return _minimax_h3_recommended_models()

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "num_frames", "height", "width"]

    def get_model_id(self) -> str:
        return MINIMAX_H3_REPO_ID

    async def preload_model(self, context: ProcessingContext):
        import asyncio

        import torch
        from nodetool.ml.core.model_manager import ModelManager

        _, components_manager_class, modular_pipeline_class = _load_minimax_h3_runtime()

        offload = self.enable_cpu_offload and torch.cuda.is_available()
        cache_key = (
            f"{MINIMAX_H3_REPO_ID}_ModularPipeline_{self._workflow}_offload{offload}"
        )

        cached = ModelManager.get_model(cache_key)
        if cached is not None:
            self._pipeline = cached
            return

        dtype = select_inference_dtype()

        def _load() -> Any:
            manager = components_manager_class() if offload else None
            pipeline = modular_pipeline_class.from_pretrained(
                MINIMAX_H3_REPO_ID,
                workflow=self._workflow,
                components_manager=manager,
            )
            pipeline.load_components(dtype=dtype)
            if manager is not None:
                manager.enable_auto_cpu_offload(device="cuda")
            return pipeline

        self._pipeline = await asyncio.to_thread(_load)
        ModelManager.set_model(self.id, cache_key, self._pipeline)

    async def move_to_device(self, device: str):
        # With auto offload the components manager owns placement; without it the
        # whole pipeline lives on the device.
        if self._pipeline is None or self.enable_cpu_offload:
            return
        if hasattr(self._pipeline, "to"):
            self._pipeline.to(device)

    def _canvas_kwargs(self) -> dict[str, int]:
        """Only pass height/width when both are set, so the model resolves its own canvas."""
        if self.height > 0 and self.width > 0:
            return {"height": self.height, "width": self.width}
        return {}

    async def _generate(
        self, context: ProcessingContext, **workflow_kwargs: Any
    ) -> OutputType:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        results = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            output=["videos", "audio", "sampling_rate"],
            **self._canvas_kwargs(),
            **workflow_kwargs,
        )

        frames = results["videos"][0]
        # `audio` is a `(1, 2, num_samples)` batch; take the request's own waveform.
        waveform = results["audio"][0]
        sampling_rate = int(results["sampling_rate"])

        run_gc("After MiniMax-H3 inference", log_before_after=False)

        video = await video_from_frames_with_audio(
            context,
            frames,
            audio=waveform,
            audio_sample_rate=sampling_rate,
            fps=MINIMAX_H3_FPS,
        )
        # AudioSegment reads interleaved samples, so the channel axis goes last.
        samples = waveform.float().cpu().numpy().T
        audio = await context.audio_from_numpy(
            samples,
            sampling_rate,
            num_channels=samples.shape[1] if samples.ndim > 1 else 1,
        )
        return {"video": video, "audio": audio}


class MiniMaxH3(_MiniMaxH3Base):
    """
    Generates a video and its soundtrack together from a text prompt and optional keyframes using MiniMax-H3.
    video, audio, generation, AI, text-to-video, keyframe, soundtrack, minimax

    Use cases:
    - Create short clips that come with their own synchronized audio
    - Animate a still image into a video that starts from it
    - Generate a transition that ends on a given frame
    - Produce sounded video content for social media and storytelling

    **Note:** MiniMax-H3 is guidance-distilled, so there is no negative prompt and
    no guidance scale. It generates 5 to 15 seconds at 24 fps. Leave both keyframes
    empty for text-to-video-and-audio.
    """

    _workflow: str = "fl2va"

    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional keyframe the video starts from. It also sets the canvas aspect ratio.",
    )
    last_image: ImageRef = Field(
        default=ImageRef(),
        description="Optional keyframe the video ends on. Can be used on its own to generate up to a frame.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "MiniMax-H3"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "image", "num_frames"]

    async def process(self, context: ProcessingContext) -> _MiniMaxH3Base.OutputType:
        keyframes: dict[str, Any] = {}
        if not self.image.is_empty():
            keyframes["image"] = await context.image_to_pil(self.image)
        if not self.last_image.is_empty():
            keyframes["last_image"] = await context.image_to_pil(self.last_image)

        return await self._generate(context, **keyframes)


class MiniMaxH3Reference(_MiniMaxH3Base):
    """
    Generates a video and its soundtrack conditioned on image, video and audio references using MiniMax-H3.
    video, audio, generation, AI, reference, omni-reference, soundtrack, minimax

    Use cases:
    - Keep a subject, style or scene consistent across generated clips
    - Transfer the motion and camera work of a reference video
    - Drive lip movement and delivery from a reference voice recording
    - Continue a previous generation with the same character and soundtrack

    **Note:** References are read in order — images, then videos, then audio clips —
    and the order is part of the request. At most 9 images, 3 videos and 3 audio
    clips, 12 in total; audio references cannot be used on their own. References do
    not bind the canvas, which defaults to 16:9.
    """

    _workflow: str = "ref2va"

    image_references: list[ImageRef] = Field(
        default=[],
        description="Subject, style or scene references. At most 9.",
    )
    video_references: list[VideoRef] = Field(
        default=[],
        description="Motion and camera references, conditioned on with their own soundtrack. At most 3.",
    )
    audio_references: list[AudioRef] = Field(
        default=[],
        description="Voice or music references. At most 3, and never without an image or video reference.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "MiniMax-H3 Reference"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "image_references", "num_frames"]

    def _validate_references(self) -> None:
        images = len(self.image_references)
        videos = len(self.video_references)
        audios = len(self.audio_references)

        if images > 9:
            raise ValueError(f"At most 9 image references are supported, got {images}")
        if videos > 3:
            raise ValueError(f"At most 3 video references are supported, got {videos}")
        if audios > 3:
            raise ValueError(f"At most 3 audio references are supported, got {audios}")
        if images + videos + audios > 12:
            raise ValueError(
                f"At most 12 references are supported, got {images + videos + audios}"
            )
        if audios and not (images or videos):
            raise ValueError(
                "Audio references need at least one image or video reference"
            )
        if not (images or videos or audios):
            raise ValueError("Provide at least one reference")

    async def _build_references(self, context: ProcessingContext) -> list[Any]:
        import os
        import tempfile

        integration, _, _ = _load_minimax_h3_runtime()
        audio_reference_class = integration.MiniMaxH3AudioReference
        image_reference_class = integration.MiniMaxH3ImageReference
        video_reference_class = integration.MiniMaxH3VideoReference

        references: list[Any] = [
            image_reference_class(image=await context.image_to_pil(image))
            for image in self.image_references
        ]

        # Videos and audio are decoded through `from_file`, which is what carries
        # the frame rate and sample rate of the container onto the reference —
        # a rate lost on the way in conditions the request at the wrong speed.
        async def _from_file(reference_class: Any, asset: Any, suffix: str) -> Any:
            data = await context.asset_to_bytes(asset)
            handle, path = tempfile.mkstemp(suffix=suffix)
            try:
                with os.fdopen(handle, "wb") as media:
                    media.write(data)
                return reference_class.from_file(path)
            finally:
                os.unlink(path)

        for video in self.video_references:
            references.append(await _from_file(video_reference_class, video, ".mp4"))
        for audio in self.audio_references:
            references.append(await _from_file(audio_reference_class, audio, ".wav"))

        return references

    async def process(self, context: ProcessingContext) -> _MiniMaxH3Base.OutputType:
        self._validate_references()
        references = await self._build_references(context)
        return await self._generate(context, references=references)
