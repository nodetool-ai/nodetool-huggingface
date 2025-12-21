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
from .huggingface_pipeline import HuggingFacePipelineNode

if TYPE_CHECKING:
    import torch
    from transformers import CLIPVisionModel
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline


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
                self._pipeline.enable_model_cpu_offload()  # type: ignore
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
            callback_on_step_end=callback_on_step_end,  # type: ignore
        )

        return await context.video_from_frames(output.frames[0], fps=self.fps)  # type: ignore

    def required_inputs(self):
        return ["input_image"]
