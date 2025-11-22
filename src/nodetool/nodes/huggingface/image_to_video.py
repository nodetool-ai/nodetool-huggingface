from typing import Any
from enum import Enum
from pydantic import Field
import torch
from nodetool.nodes.huggingface.stable_diffusion_base import (
    ModelVariant,
    _select_diffusion_dtype,
)
from transformers import CLIPVisionModel
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from huggingface_hub.file_download import try_to_load_from_cache

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


class Wan_I2V(HuggingFacePipelineNode):
    """
    Generates a video from an input image using Wan image-to-video pipelines.
    video, generation, AI, image-to-video, diffusion, Wan

    Use cases:
    - Turn a single image into a dynamic clip with prompt guidance
    - Choose between Wan 2.2 A14B, Wan 2.1 14B 480P, and Wan 2.1 14B 720P models
    """

    class WanI2VModel(str, Enum):
        WAN_2_2_I2V_A14B = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        WAN_2_1_I2V_14B_480P = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        WAN_2_1_I2V_14B_720P = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

    input_image: ImageRef = Field(
        default=ImageRef(),
        description="The input image to generate the video from.",
    )
    prompt: str = Field(
        default="An astronaut walking on the moon, cinematic lighting, high detail",
        description="A text prompt describing the desired video.",
    )
    model_variant: WanI2VModel = Field(
        default=WanI2VModel.WAN_2_2_I2V_A14B,
        description="Select the Wan I2V model to use.",
    )
    negative_prompt: str = Field(
        default="",
        description="A text prompt describing what to avoid in the video.",
    )
    num_frames: int = Field(
        default=81,
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
        default=50,
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
        default=832,
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

    _pipeline: WanImageToVideoPipeline | None = None

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
                cache_path = try_to_load_from_cache(
                    repo_id_for_cache,
                    candidate,
                    revision=revision,
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
        torch_dtype = _select_diffusion_dtype()
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
