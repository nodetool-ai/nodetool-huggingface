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
    available_torch_dtype,
)
from nodetool.workflows.memory_utils import run_gc
from nodetool.workflows.types import NodeProgress
from nodetool.huggingface.video_utils import video_from_frames

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
    from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline
    from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline


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

        frames = output.frames[0]

        run_gc("After CogVideoX inference", log_before_after=False)
        return await context.video_from_numpy(frames, fps=self.fps)


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
                repo_id="Wan-AI/Wan2.2-T2V-14B-Diffusers",
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
