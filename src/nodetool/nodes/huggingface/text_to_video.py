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
    is_mps_device,
    maybe_enable_cpu_offload,
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

        self._pipeline = await self.load_model(
            context=context,
            model_class=LTX2Pipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
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

        self._pipeline = await self.load_model(
            context=context,
            model_class=Kandinsky5T2VPipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
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
