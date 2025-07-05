from typing import Any
import numpy as np
from pydantic import Field
from nodetool.nodes.huggingface.huggingface_node import progress_callback
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    HFStableDiffusion,
    HuggingFaceModel,
    HFTextToVideo,
    ImageRef,
    VideoRef,
)
import torch
from diffusers.pipelines.animatediff.pipeline_animatediff import AnimateDiffPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.unets.unet_motion_model import MotionAdapter
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    StableVideoDiffusionPipeline,
)
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
from diffusers.utils.export_utils import export_to_video
from .huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.types import NodeProgress


class AnimateDiffNode(HuggingFacePipelineNode):
    """
    Generates animated GIFs using the AnimateDiff pipeline.
    image, animation, generation, AI

    Use cases:
    - Create animated visual content from text descriptions
    - Generate dynamic visual effects for creative projects
    - Produce animated illustrations for digital media
    """

    model: HFStableDiffusion = Field(
        default=HFStableDiffusion(),
        description="The model to use for image generation.",
    )

    prompt: str = Field(
        default="masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery",
        description="A text prompt describing the desired animation.",
    )
    negative_prompt: str = Field(
        default="bad quality, worse quality",
        description="A text prompt describing what you don't want in the animation.",
    )
    num_frames: int = Field(
        default=16, description="The number of frames in the animation.", ge=1, le=60
    )
    guidance_scale: float = Field(
        default=7.5, description="Scale for classifier-free guidance.", ge=1.0, le=20.0
    )
    num_inference_steps: int = Field(
        default=25, description="The number of denoising steps.", ge=1, le=100
    )
    seed: int = Field(
        default=42, description="Seed for the random number generator.", ge=0
    )

    _pipeline: AnimateDiffPipeline | None = None

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(
                repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
                allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"],
            ),
            HFStableDiffusion(
                repo_id="Lykon/dreamshaper-8",
                allow_patterns=[
                    "**/*.fp16.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HFStableDiffusion(
                repo_id="Yntec/Deliberate2",
                allow_patterns=[
                    "**/*.fp16.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HFStableDiffusion(
                repo_id="imagepipeline/epiC-PhotoGasm",
                allow_patterns=[
                    "**/*.fp16.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HFStableDiffusion(
                repo_id="526christian/526mix-v1.5",
                allow_patterns=[
                    "**/*.fp16.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HFStableDiffusion(
                repo_id="stablediffusionapi/realistic-vision-v51",
                allow_patterns=[
                    "**/*.fp16.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
            HFStableDiffusion(
                repo_id="stablediffusionapi/anything-v5",
                allow_patterns=[
                    "**/*.fp16.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        adapter = await self.load_model(
            context=context,
            model_class=MotionAdapter,
            model_id="guoyww/animatediff-motion-adapter-v1-5-2",
        )
        self._pipeline = await self.load_model(
            context=context,
            model_class=AnimateDiffPipeline,
            model_id=self.model.repo_id,
            motion_adapter=adapter,
        )

        scheduler = DDIMScheduler.from_pretrained(
            self.model.repo_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        assert self._pipeline is not None
        self._pipeline.scheduler = scheduler
        self._pipeline.enable_vae_slicing()
        self._pipeline.enable_model_cpu_offload()

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> VideoRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        output = self._pipeline(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_frames=self.num_frames,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            callback=progress_callback(self.id, self.num_inference_steps, context),
            callback_steps=1,
            generator=generator,
        )

        frames = output.frames[0]  # type: ignore

        return await context.video_from_numpy(frames)  # type: ignore


class StableVideoDiffusion(HuggingFacePipelineNode):
    """
    Generates a video from a single image using the Stable Video Diffusion model.
    video, generation, AI, image-to-video, stable-diffusion, SD

    Use cases:
    - Create short animations from static images
    - Generate dynamic content for presentations or social media
    - Prototype video ideas from still concept art
    """

    input_image: ImageRef = Field(
        default=ImageRef(),
        description="The input image to generate the video from, resized to 1024x576.",
    )
    num_frames: int = Field(
        default=14, ge=1, le=50, description="Number of frames to generate."
    )
    num_inference_steps: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Number of steps per generated frame",
    )
    fps: int = Field(
        default=7, ge=1, le=30, description="Frames per second for the output video."
    )
    decode_chunk_size: int = Field(
        default=8, ge=1, le=16, description="Number of frames to decode at once."
    )
    seed: int = Field(
        default=42, ge=0, le=2**32 - 1, description="Random seed for generation."
    )

    _pipeline: StableVideoDiffusionPipeline | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFStableDiffusion(
                repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
                allow_patterns=[
                    "**/*.fp16.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                ],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_model(
            context=context,
            model_class=StableVideoDiffusionPipeline,
            model_id="stabilityai/stable-video-diffusion-img2vid-xt",
        )
        self._pipeline.enable_model_cpu_offload()  # type: ignore

    async def process(self, context: ProcessingContext) -> VideoRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        # Load and preprocess the input image
        input_image = await context.image_to_pil(self.input_image)
        input_image = input_image.resize((1024, 576))

        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        def callback(pipe: StableVideoDiffusionPipeline, step: int, *args):
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )
            return {}

        # Generate the video frames
        frames = self._pipeline(
            input_image,
            num_frames=self.num_frames,
            decode_chunk_size=self.decode_chunk_size,
            generator=generator,
            callback_on_step_end=callback,  # type: ignore
        ).frames[  # type: ignore
            0
        ]
        return await context.video_from_numpy(np.array(frames), fps=self.fps)  # type: ignore

    @classmethod
    def get_title(cls) -> str:
        return "Stable Video Diffusion"

    def required_inputs(self):
        return ["input_image"]


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

    _pipeline: CogVideoXPipeline | None = None

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
        self._pipeline = await self.load_model(
            context=context,
            model_class=CogVideoXPipeline,
            model_id=self.get_model_id(),
            torch_dtype=torch.bfloat16,
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

        # Set up the generator for reproducibility
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        def callback_on_step_end(step: int, timestep: int, callback_kwargs: dict) -> None:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )

        # Generate the video
        output = self._pipeline(
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
