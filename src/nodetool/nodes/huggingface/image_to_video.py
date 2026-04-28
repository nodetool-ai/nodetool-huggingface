from __future__ import annotations
from typing import Any, TYPE_CHECKING
from enum import Enum
from pydantic import Field

from nodetool.nodes.huggingface.stable_diffusion_base import (
    available_torch_dtype,
)
from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    HFTextToVideo,
    HuggingFaceModel,
    ImageRef,
    VideoRef,
)
from nodetool.workflows.types import NodeProgress
from nodetool.workflows.memory_utils import run_gc
from nodetool.huggingface.video_utils import video_from_frames
from .huggingface_pipeline import HuggingFacePipelineNode

if TYPE_CHECKING:
    import torch
    from transformers import CLIPVisionModel
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
    from diffusers.pipelines.ltx.pipeline_ltx_image2video import LTXImageToVideoPipeline
    from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import (
        LTX2ImageToVideoPipeline,
    )


class Wan_I2V(HuggingFacePipelineNode):
    """
    Transforms a static image into a dynamic video clip using Wan image-to-video diffusion models.
    video, generation, AI, image-to-video, diffusion, Wan, animation

    Use cases:
    - Animate photographs and artwork into short video clips
    - Create motion from still images with text-guided direction
    - Generate video content for social media from static images
    - Bring product images to life with realistic movement
    - Create dynamic visual effects from single frames

    **Note:** Model variants offer different quality/speed tradeoffs. A14B is balanced; 720P provides higher resolution.
    """

    class WanI2VModel(str, Enum):
        WAN_2_2_I2V_A14B = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        WAN_2_1_I2V_14B_480P = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        WAN_2_1_I2V_14B_720P = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

    input_image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to animate. Image content guides the video's appearance.",
    )
    prompt: str = Field(
        default="An astronaut walking on the moon, cinematic lighting, high detail",
        description="Text description guiding how the image should animate and move.",
    )
    model_variant: WanI2VModel = Field(
        default=WanI2VModel.WAN_2_2_I2V_A14B,
        description="The Wan I2V model variant. A14B is balanced; 720P offers higher resolution output.",
    )
    negative_prompt: str = Field(
        default="",
        description="Describe what to avoid in the generated video (e.g., 'blurry, distorted').",
    )
    num_frames: int = Field(
        default=81,
        description="Total frames in the output video. More frames = longer duration.",
        ge=16,
        le=129,
    )
    guidance_scale: float = Field(
        default=5.0,
        description="How strongly to follow the prompt. Higher values = more prompt adherence.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. More steps = better quality but slower generation.",
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
        default=832,
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
        description="Random seed for reproducibility. Use -1 for random generation.",
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
        description="Process VAE in tiles for very large videos. May affect quality.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(
                repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
            HFTextToVideo(
                repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
            HFTextToVideo(
                repo_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Wan (Image-to-Video)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "input_image",
            "prompt",
            "num_frames",
            "height",
            "width",
            "model_variant",
        ]

    def get_model_id(self) -> str:
        return self.model_variant.value

    async def preload_model(self, context: ProcessingContext):
        model_id = self.get_model_id()
        repo_id_for_cache = model_id
        revision = None
        if "@" in repo_id_for_cache:
            repo_id_for_cache, revision = repo_id_for_cache.rsplit("@", 1)

        cache_checked = False
        for candidate in ("model_index.json", "config.json"):
            try:
                cache_path = await HF_FAST_CACHE.resolve(
                    repo_id_for_cache,
                    candidate,
                    repo_type=None,
                )
            except Exception:
                cache_path = None

            if cache_path:
                cache_checked = True
                break

        if not cache_checked:
            raise ValueError(
                f"Model {model_id} must be downloaded before running this node."
            )

        import torch
        from transformers import CLIPVisionModel
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
        from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline

        image_encoder = await self.load_model(
            context=context,
            model_class=CLIPVisionModel,
            model_id=model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        )
        vae = await self.load_model(
            context=context,
            model_class=AutoencoderKLWan,
            model_id=model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        torch_dtype = available_torch_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_class=WanImageToVideoPipeline,
            model_id=model_id,
            torch_dtype=torch_dtype,
            device="cpu",
            vae=vae,
            image_encoder=image_encoder,
        )

        if self._pipeline is not None:
            if self.enable_cpu_offload and hasattr(
                self._pipeline, "enable_model_cpu_offload"
            ):
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

        input_image = await context.image_to_pil(self.input_image)

        generator = None
        if self.seed != -1:
            import torch

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
            image=input_image,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            max_sequence_length=self.max_sequence_length,
            callback_on_step_end=callback_on_step_end,
        )

        run_gc("After Wan I2V inference", log_before_after=False)
        return await video_from_frames(context, output.frames[0], fps=self.fps)

    def required_inputs(self):
        return ["input_image"]


class Wan_FLF2V(HuggingFacePipelineNode):
    """
    Generates smooth video transitions between a first and last frame using Wan FLF2V.
    video, generation, AI, image-to-video, first-last-frame, interpolation, diffusion, Wan

    Use cases:
    - Create seamless transitions between two keyframe images
    - Animate the journey from one scene to another
    - Generate motion-filled clips from start and end frames
    - Build scene interpolation workflows for film and VFX
    - Produce smooth morphing effects for creative projects
    """

    class WanFLF2VModel(str, Enum):
        WAN_2_2_FLF2V_14B_720P = "Wan-AI/Wan2.2-FLF2V-14B-720P-Diffusers"
        WAN_2_1_FLF2V_14B_720P = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"

    model_variant: WanFLF2VModel = Field(
        default=WanFLF2VModel.WAN_2_2_FLF2V_14B_720P,
        description="The Wan FLF2V model variant. 2.2 is the latest; 2.1 is the previous generation.",
    )
    first_image: ImageRef = Field(
        default=ImageRef(),
        description="The first frame (starting image) for the video transition.",
    )
    last_image: ImageRef = Field(
        default=ImageRef(),
        description="The last frame (ending image) for the video transition.",
    )
    prompt: str = Field(
        default="A smooth transition between the two scenes, cinematic motion",
        description="Text description guiding the motion and style of the generated transition.",
    )
    negative_prompt: str = Field(
        default="",
        description="Describe what to avoid in the generated video (e.g., 'blurry, distorted').",
    )
    num_frames: int = Field(
        default=81,
        description="Total frames in the output video. More frames = longer duration.",
        ge=16,
        le=129,
    )
    guidance_scale: float = Field(
        default=5.0,
        description="How strongly to follow the prompt.",
        ge=1.0,
        le=20.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. More steps = better quality but slower.",
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
        default=832,
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
        description="Random seed for reproducibility. Use -1 for random.",
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
                repo_id="Wan-AI/Wan2.2-FLF2V-14B-720P-Diffusers",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
            HFTextToVideo(
                repo_id="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Wan (First-Last-Frame to Video)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["first_image", "last_image", "prompt", "num_frames", "height", "width"]

    def get_model_id(self) -> str:
        return self.model_variant.value

    def required_inputs(self):
        return ["first_image", "last_image"]

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.wan.pipeline_wan_flf2v import WanFLF2VPipeline
        from nodetool.nodes.huggingface.stable_diffusion_base import (
            available_torch_dtype,
        )

        torch_dtype = available_torch_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_class=WanFLF2VPipeline,
            model_id=self.get_model_id(),
            torch_dtype=torch_dtype,
            variant=None,
        )
        if self.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()

    async def move_to_device(self, device: str):
        if self._pipeline is not None and not self.enable_cpu_offload:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> VideoRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        first_frame = await context.image_to_pil(self.first_image)
        last_frame = await context.image_to_pil(self.last_image)

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
            image=first_frame,
            last_image=last_frame,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
        )

        run_gc("After Wan FLF2V inference", log_before_after=False)
        return await video_from_frames(context, output.frames[0], fps=self.fps)


class LTXVideoI2V(HuggingFacePipelineNode):
    """
    Animates a static image into a video using LTX-Video image-to-video diffusion models.
    video, generation, AI, image-to-video, diffusion, LTX, Lightricks, animation

    Use cases:
    - Animate photographs or artwork into dynamic video clips
    - Create motion-driven video from a single input image with text guidance
    - Generate content for social media or creative projects from still images
    - Bring product images or illustrations to life
    - Produce smooth animated sequences with fine temporal control

    **Note:** LTX-Video-0.9.5 is recommended for best quality.
    """

    class LTXModel(str, Enum):
        LTX_VIDEO_0_9_5 = "Lightricks/LTX-Video-0.9.5"
        LTX_VIDEO_0_9_1 = "Lightricks/LTX-Video-0.9.1"
        LTX_VIDEO = "Lightricks/LTX-Video"

    model_variant: LTXModel = Field(
        default=LTXModel.LTX_VIDEO_0_9_5,
        description="The LTX-Video model variant. 0.9.5 is the latest and recommended.",
    )
    input_image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to animate. Image content guides the video's appearance.",
    )
    prompt: str = Field(
        default="The scene comes to life with smooth, natural motion, cinematic quality",
        description="Text description guiding how the image should animate and move.",
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Describe what to avoid in the generated video.",
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
        description="Output video height in pixels.",
        ge=64,
        le=1024,
    )
    width: int = Field(
        default=704,
        description="Output video width in pixels.",
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
        return "LTX-Video (Image-to-Video)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "input_image",
            "prompt",
            "model_variant",
            "num_frames",
            "height",
            "width",
        ]

    def get_model_id(self) -> str:
        return self.model_variant.value

    def required_inputs(self):
        return ["input_image"]

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.ltx.pipeline_ltx_image2video import (
            LTXImageToVideoPipeline,
        )

        torch_dtype = available_torch_dtype()
        self._pipeline = await self.load_model(
            context=context,
            model_class=LTXImageToVideoPipeline,
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

        input_image = await context.image_to_pil(self.input_image)

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
            image=input_image,
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

        run_gc("After LTX-Video I2V inference", log_before_after=False)
        return await video_from_frames(context, output.frames[0], fps=self.frame_rate)


class LTX2VideoI2V(HuggingFacePipelineNode):
    """
    Animates a static image into a video using LTX-2, the newest Lightricks image-to-video model.
    video, generation, AI, image-to-video, diffusion, LTX2, Lightricks, animation

    Use cases:
    - Animate images with the latest LTX-2 model for improved quality
    - Create smooth video transitions from a single reference image
    - Generate motion-driven video content with extended prompt support
    - Produce high-fidelity animated clips for creative and commercial use
    - Build next-generation image animation workflows

    **Note:** LTX-2 uses a Gemma3 text encoder and requires more VRAM than LTX-Video.
    """

    input_image: ImageRef = Field(
        default=ImageRef(),
        description="The source image to animate. Image content guides the video's appearance.",
    )
    prompt: str = Field(
        default="The scene comes to life with smooth, natural motion, cinematic quality",
        description="Text description guiding how the image should animate and move.",
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Describe what to avoid in the generated video.",
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
        return "LTX-2 (Image-to-Video)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["input_image", "prompt", "num_frames", "height", "width"]

    def get_model_id(self) -> str:
        return "Lightricks/LTX-2"

    def required_inputs(self):
        return ["input_image"]

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import (
            LTX2ImageToVideoPipeline,
        )

        import torch

        self._pipeline = await self.load_model(
            context=context,
            model_class=LTX2ImageToVideoPipeline,
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

        input_image = await context.image_to_pil(self.input_image)

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
            image=input_image,
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
        run_gc("After LTX-2 I2V inference", log_before_after=False)
        return await video_from_frames(context, output.frames[0], fps=fps)
