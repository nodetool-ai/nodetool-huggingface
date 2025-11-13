from enum import Enum
import os
from random import randint
import asyncio

from huggingface_hub import try_to_load_from_cache
from pydantic import Field
from typing import Any

import torch

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    HFCLIP,
    HFCLIPVision,
    HFControlNet,
    HFLoraSDConfig,
    HFLoraSDXLConfig,
    HFTextToImage,
    HFUnet,
    ImageRef,
    TorchTensor,
)
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
from diffusers.schedulers.scheduling_dpmsolver_singlestep import (
    DPMSolverSinglestepScheduler,
)
from diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete import (
    KDPM2AncestralDiscreteScheduler,
)

from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress


log = get_logger(__name__)


HF_STABLE_DIFFUSION_MODELS = [
    HFTextToImage(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        ignore_patterns=["v1-5-*.safetensors", "v1-5-*.ckpt"],
    ),
    HFTextToImage(
        repo_id="SG161222/Realistic_Vision_V5.1_noVAE",
        ignore_patterns=["Realistic_Vision_*.safetensors"],
    ),
    HFTextToImage(
        repo_id="Lykon/DreamShaper",
        ignore_patterns=["DreamShaper_*.safetensors"],
    ),
]

HF_STABLE_DIFFUSION_XL_MODELS = [
    HFTextToImage(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        ignore_patterns=["sd_xl_base_*.safetensors"],
    ),
    HFTextToImage(
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        ignore_patterns=["sd_xl_refiner_*.safetensors"],
    ),
    HFTextToImage(
        repo_id="fal-collab-models/dreamshaper-xl-1-0",
    )
]

HF_STABLE_DIFFUSION_3_MODELS = [
    HFTextToImage(
        repo_id="Comfy-Org/stable-diffusion-3.5-fp8",
        path="sd3.5_large_fp8_scaled.safetensors",
    ),
    HFTextToImage(
        repo_id="Comfy-Org/stable-diffusion-3.5-fp8",
        path="sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors",
    ),
]


HF_CONTROLNET_MODELS: list[HFControlNet] = [
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_canny",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_inpaint",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_mlsd",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_lineart",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_scribble",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_openpose",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
]

HF_CONTROL_NET_XL_MODELS: list[HFControlNet] = [
    HFControlNet(
        repo_id="diffusers/controlnet-canny-sdxl-1.0",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="diffusers/controlnet-depth-sdxl-1.0",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="diffusers/controlnet-zoe-depth-sdxl-1.0",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
]

HF_LTXV_MODELS = [
    HFTextToImage(
        repo_id="Lightricks/LTX-Video",
        path="ltx-video-2b-v0.9.safetensors",
    ),
]

HF_CLIP_MODELS = [
    HFCLIP(
        repo_id="Comfy-Org/mochi_preview_repackaged",
        path="split_files/text_encoders/t5xxl_fp16.safetensors",
    ),
    HFCLIP(
        repo_id="Comfy-Org/mochi_preview_repackaged",
        path="split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    HFCLIP(repo_id="comfyanonymous/flux_text_encoders", path="clip_l.safetensors"),
    HFCLIP(
        repo_id="comfyanonymous/flux_text_encoders",
        path="t5xxl_fp16.safetensors",
    ),
]

HF_CLIP_VISION_MODELS = [
    HFCLIPVision(
        repo_id="Comfy-Org/sigclip_vision_384",
        path="sigclip_vision_patch14_384.safetensors",
    ),
    HFCLIPVision(
        repo_id="h94/IP-Adapter",
        path="models/image_encoder/model.safetensors",
    ),
    HFCLIPVision(
        repo_id="h94/IP-Adapter",
        path="sdxl_models/image_encoder/model.safetensors",
    ),
]


class StableDiffusionDetailLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


def load_loras(pipeline: Any, loras: list[HFLoraSDConfig] | list[HFLoraSDXLConfig]):
    log.debug(f"Loading LoRAs. Total LoRAs provided: {len(loras)}")
    loras = [lora for lora in loras if lora.lora.is_set()]  # type: ignore
    log.debug(f"LoRAs after filtering (only set ones): {len(loras)}")

    if len(loras) == 0:
        log.debug("No LoRAs to load")
        return

    if not hasattr(pipeline, "load_lora_weights") or not hasattr(
        pipeline, "set_adapters"
    ):
        log.warning(
            "Skipping LoRA loading because the current pipeline does not support adapters"
        )
        return

    lora_names = []
    lora_weights = []
    for i, lora in enumerate(loras):
        log.debug(
            f"Processing LoRA {i+1}/{len(loras)}: {lora.lora.repo_id}/{lora.lora.path}"
        )
        cache_path = try_to_load_from_cache(
            lora.lora.repo_id,
            lora.lora.path or "",
        )
        if cache_path is None:
            log.error(f"LoRA cache not found for {lora.lora.repo_id}/{lora.lora.path}")
            raise ValueError(
                f"Install {lora.lora.repo_id}/{lora.lora.path} LORA to use it (Recommended Models above)"
            )
        base_name = os.path.basename(lora.lora.path or "").split(".")[0]
        log.debug(f"LoRA loaded from cache: {cache_path}")
        log.debug(f"LoRA base name: {base_name}, strength: {lora.strength}")
        lora_names.append(base_name)
        lora_weights.append(lora.strength)
        pipeline.load_lora_weights(cache_path, adapter_name=base_name)

    log.debug(f"Setting LoRA adapters: names={lora_names}, weights={lora_weights}")
    pipeline.set_adapters(lora_names, adapter_weights=lora_weights)
    log.debug("LoRA loading completed successfully")


def quantize_to_multiple_of_64(value):
    original = value
    quantized = round(value / 64) * 64
    log.debug(f"Quantizing {original} to multiple of 64: {quantized}")
    return quantized


def upscale_latents(latents: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
    """Upscale latents using torch interpolation.

    Args:
        latents: Input latents tensor of shape (B, C, H, W)
        scale_factor: Factor to scale dimensions by

    Returns:
        Upscaled latents tensor
    """
    log.debug(f"Upscaling latents with scale factor: {scale_factor}")

    # Ensure input is on correct device
    if not isinstance(latents, torch.Tensor):
        log.error("Input to upscale_latents must be a torch tensor")
        raise ValueError("Input must be a torch tensor")

    # Get current dimensions
    batch_size, channels, height, width = latents.shape
    log.debug(f"Original latent dimensions: {batch_size}x{channels}x{height}x{width}")

    # Calculate new dimensions
    new_height = height * scale_factor
    new_width = width * scale_factor
    log.debug(
        f"New latent dimensions: {batch_size}x{channels}x{new_height}x{new_width}"
    )

    # Use interpolate for upscaling
    upscaled = torch.nn.functional.interpolate(
        latents, size=(new_height, new_width), mode="bicubic", align_corners=False
    )

    log.debug("Latent upscaling completed")
    return upscaled


class ModelVariant(Enum):
    DEFAULT = "default"
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"


class StableDiffusionBaseNode(HuggingFacePipelineNode):

    async def preload_model(self, context: ProcessingContext):
        """Preload the Stable Diffusion model and set up pipeline."""
        log.debug(f"Preloading Stable Diffusion model: {self.model.repo_id}")
        log.debug(f"Model path: {self.model.path}")
        log.debug(f"Device context: {context.device}")
        await super().preload_model(context)
        log.debug("Stable Diffusion model preloaded successfully")

    class StableDiffusionScheduler(str, Enum):
        DPMSolverSDEScheduler = "DPMSolverSDEScheduler"
        EulerDiscreteScheduler = "EulerDiscreteScheduler"
        LMSDiscreteScheduler = "LMSDiscreteScheduler"
        DDIMScheduler = "DDIMScheduler"
        DDPMScheduler = "DDPMScheduler"
        HeunDiscreteScheduler = "HeunDiscreteScheduler"
        DPMSolverMultistepScheduler = "DPMSolverMultistepScheduler"
        DEISMultistepScheduler = "DEISMultistepScheduler"
        PNDMScheduler = "PNDMScheduler"
        EulerAncestralDiscreteScheduler = "EulerAncestralDiscreteScheduler"
        UniPCMultistepScheduler = "UniPCMultistepScheduler"
        KDPM2DiscreteScheduler = "KDPM2DiscreteScheduler"
        DPMSolverSinglestepScheduler = "DPMSolverSinglestepScheduler"
        KDPM2AncestralDiscreteScheduler = "KDPM2AncestralDiscreteScheduler"

    class StableDiffusionOutputType(str, Enum):
        IMAGE = "Image"
        LATENT = "Latent"

    @classmethod
    def get_scheduler_class(cls, scheduler: StableDiffusionScheduler):
        if scheduler == cls.StableDiffusionScheduler.DPMSolverSDEScheduler:
            return DPMSolverSDEScheduler
        elif scheduler == cls.StableDiffusionScheduler.EulerDiscreteScheduler:
            return EulerDiscreteScheduler
        elif scheduler == cls.StableDiffusionScheduler.LMSDiscreteScheduler:
            return LMSDiscreteScheduler
        elif scheduler == cls.StableDiffusionScheduler.DDIMScheduler:
            return DDIMScheduler
        elif scheduler == cls.StableDiffusionScheduler.DDPMScheduler:
            return DDPMScheduler
        elif scheduler == cls.StableDiffusionScheduler.HeunDiscreteScheduler:
            return HeunDiscreteScheduler
        elif scheduler == cls.StableDiffusionScheduler.DPMSolverMultistepScheduler:
            return DPMSolverMultistepScheduler
        elif scheduler == cls.StableDiffusionScheduler.DEISMultistepScheduler:
            return DEISMultistepScheduler
        elif scheduler == cls.StableDiffusionScheduler.PNDMScheduler:
            return PNDMScheduler
        elif scheduler == cls.StableDiffusionScheduler.EulerAncestralDiscreteScheduler:
            return EulerAncestralDiscreteScheduler
        elif scheduler == cls.StableDiffusionScheduler.UniPCMultistepScheduler:
            return UniPCMultistepScheduler
        elif scheduler == cls.StableDiffusionScheduler.KDPM2DiscreteScheduler:
            return KDPM2DiscreteScheduler
        elif scheduler == cls.StableDiffusionScheduler.DPMSolverSinglestepScheduler:
            return DPMSolverSinglestepScheduler
        elif scheduler == cls.StableDiffusionScheduler.KDPM2AncestralDiscreteScheduler:
            return KDPM2AncestralDiscreteScheduler

    model: HFTextToImage = Field(
        default=HFTextToImage(),
        description="The model to use for image generation.",
    )
    variant: ModelVariant = Field(
        default=ModelVariant.FP16,
        description="The variant of the model to use for generation.",
    )
    prompt: str = Field(default="", description="The prompt for image generation.")
    negative_prompt: str = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        le=2**32 - 1,
        description="Seed for the random number generator. Use -1 for a random seed.",
    )
    num_inference_steps: int = Field(
        default=25, ge=1, le=100, description="Number of denoising steps."
    )
    guidance_scale: float = Field(
        default=7.5, ge=1.0, le=20.0, description="Guidance scale for generation."
    )
    scheduler: StableDiffusionScheduler = Field(
        default=StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    loras: list[HFLoraSDConfig] = Field(
        default=[],
        description="The LoRA models to use for image processing",
    )
    pag_scale: float = Field(
        default=3.0,
        ge=0.0,
        le=10.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    latents: TorchTensor = Field(
        default=TorchTensor(),
        description="Optional initial latents to start generation from.",
    )
    enable_attention_slicing: bool = Field(
        default=True,
        description="Enable attention slicing for the pipeline. This can reduce VRAM usage.",
    )
    enable_tiling: bool = Field(
        default=True,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )
    output_type: StableDiffusionOutputType = Field(
        default=StableDiffusionOutputType.IMAGE,
        description="The type of output to generate.",
    )
    _loaded_adapters: set[str] = set()
    _pipeline: Any = None

    @classmethod
    def get_basic_fields(cls):
        return ["model", "prompt"]

    @classmethod
    def get_recommended_models(cls):
        return HF_STABLE_DIFFUSION_MODELS

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not StableDiffusionBaseNode

    async def pre_process(self, context: ProcessingContext):
        log.debug(f"Pre-processing Stable Diffusion node with seed: {self.seed}")
        if self.seed == -1:
            original_seed = self.seed
            self.seed = randint(0, 2**32 - 1)
            log.debug(f"Generated random seed: {original_seed} -> {self.seed}")
        log.debug(
            f"Pre-process complete. Model: {self.model.repo_id}, Seed: {self.seed}"
        )

    def should_skip_cache(self):
        return len(self.loras) > 0

    def _set_scheduler(self, scheduler_type: StableDiffusionScheduler):
        log.debug(f"Setting scheduler to: {scheduler_type}")
        scheduler_class = self.get_scheduler_class(scheduler_type)
        log.debug(f"Scheduler class: {scheduler_class.__name__}")
        scheduler = getattr(self._pipeline, "scheduler", None)
        if scheduler is None:
            log.warning(
                "Current pipeline does not expose a scheduler; skipping scheduler update"
            )
            return
        self._pipeline.scheduler = scheduler_class.from_config(scheduler.config)
        log.debug("Scheduler set successfully")

    async def move_to_device(self, device: str):
        log.debug(f"Moving pipeline to device: {device}")
        if self._pipeline is not None:
            log.debug("Pipeline found, moving to device")
            self._pipeline.to(device)
            log.debug(f"Pipeline moved to device: {device}")
        else:
            log.debug("No pipeline to move to device")

    def _setup_generator(self):
        log.debug("Setting up generator")
        log.debug(f"Current seed value: {self.seed}")
        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            log.debug(f"Setting manual seed: {self.seed}")
            generator = generator.manual_seed(self.seed)
        else:
            log.debug("Using random seed")
        log.debug("Generator setup complete")
        return generator

    def progress_callback(
        self,
        context: ProcessingContext,
        start: int,
        total: int,
    ):
        def callback(
            pipeline, step: int, timestep: int, kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=start + step,
                    total=total,
                )
            )
            return kwargs

        return callback

    async def run_pipeline(
        self, context: ProcessingContext, **kwargs
    ) -> ImageRef | TorchTensor:
        log.debug("Starting pipeline execution")
        if self._pipeline is None:
            log.error("Pipeline not initialized")
            raise ValueError("Pipeline not initialized")

        log.debug(f"Enable tiling: {self.enable_tiling}")
        if self.enable_tiling and hasattr(self._pipeline, "vae"):
            log.debug("Enabling VAE tiling")
            vae = getattr(self._pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()

        log.debug(f"Enable CPU offload: {self.enable_cpu_offload}")
        if self.enable_cpu_offload and hasattr(
            self._pipeline, "enable_model_cpu_offload"
        ):
            log.debug("Enabling model CPU offload")
            self._pipeline.enable_model_cpu_offload()

        log.debug(f"Enable attention slicing: {self.enable_attention_slicing}")
        if self.enable_attention_slicing and hasattr(
            self._pipeline, "enable_attention_slicing"
        ):
            log.debug("Enabling attention slicing")
            self._pipeline.enable_attention_slicing()

        loras = [
            lora for lora in self.loras if not lora.lora.path in self._loaded_adapters
        ]
        log.debug(f"New LoRAs to load: {len(loras)}")
        load_loras(self._pipeline, loras)
        self._loaded_adapters.update(lora.lora.path for lora in loras if lora.lora.path)
        log.debug(f"Total loaded adapters: {len(self._loaded_adapters)}")

        log.debug("Setting up generator")
        generator = self._setup_generator()

        width = kwargs.get("width", None)
        height = kwargs.get("height", None)
        log.debug(f"Original dimensions - width: {width}, height: {height}")

        if width is not None:
            original_width = width
            width = quantize_to_multiple_of_64(width)
            kwargs["width"] = width
            log.debug(f"Quantized width: {original_width} -> {width}")

        if height is not None:
            original_height = height
            height = quantize_to_multiple_of_64(height)
            kwargs["height"] = height
            log.debug(f"Quantized height: {original_height} -> {height}")

        log.debug("Starting pipeline inference")
        log.debug(
            f"Prompt: {self.prompt[:100]}{'...' if len(self.prompt) > 100 else ''}"
        )
        log.debug(
            f"Negative prompt: {self.negative_prompt[:100]}{'...' if len(self.negative_prompt) > 100 else ''}"
        )
        log.debug(f"Inference steps: {self.num_inference_steps}")
        log.debug(f"Guidance scale: {self.guidance_scale}")
        log.debug(f"PAG scale: {self.pag_scale}")
        log.debug(f"Output type: {self.output_type.value}")

        def _run_pipeline_sync():
            latents = None
            if self.latents.is_set():
                latents = self.latents.to_tensor().to(device=context.device)
                unet = getattr(self._pipeline, "unet", None)
                if unet is not None and hasattr(unet, "dtype"):
                    latents = latents.to(dtype=unet.dtype)

            call_kwargs: dict[str, Any] = {
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "generator": generator,
                "latents": latents,
                "callback_on_step_end": self.progress_callback(
                    context, 0, self.num_inference_steps
                ),
                "pag_scale": self.pag_scale,
                "output_type": self.output_type.value,
            }

            if self._loaded_adapters:
                call_kwargs["cross_attention_kwargs"] = {"scale": 1.0}

            call_kwargs.update(kwargs)
            return self._pipeline(**call_kwargs)

        output = await asyncio.to_thread(_run_pipeline_sync)
        image = output.images[0]

        log.debug("Pipeline inference completed")

        if self.output_type == self.StableDiffusionOutputType.IMAGE:
            log.debug("Converting PIL image to ImageRef")
            result = await context.image_from_pil(image)
            log.debug("Pipeline execution completed successfully")
            return result
        else:
            log.debug("Returning tensor output")
            result = TorchTensor.from_tensor(image)
            log.debug("Pipeline execution completed successfully")
            return result

    async def process(self, context: ProcessingContext) -> ImageRef:
        raise NotImplementedError("Subclasses must implement this method")


class StableDiffusionXLBase(HuggingFacePipelineNode):

    async def preload_model(self, context: ProcessingContext):
        """Preload the Stable Diffusion XL model and set up pipeline."""
        log.debug(f"Preloading Stable Diffusion XL model: {self.model.repo_id}")
        log.debug(f"Model path: {self.model.path}")
        log.debug(f"Device context: {context.device}")
        await super().preload_model(context)
        log.debug("Stable Diffusion XL model preloaded successfully")

    class StableDiffusionScheduler(str, Enum):
        DPMSolverSDEScheduler = "DPMSolverSDEScheduler"
        EulerDiscreteScheduler = "EulerDiscreteScheduler"
        LMSDiscreteScheduler = "LMSDiscreteScheduler"
        DDIMScheduler = "DDIMScheduler"
        DDPMScheduler = "DDPMScheduler"
        HeunDiscreteScheduler = "HeunDiscreteScheduler"
        DPMSolverMultistepScheduler = "DPMSolverMultistepScheduler"
        DEISMultistepScheduler = "DEISMultistepScheduler"
        PNDMScheduler = "PNDMScheduler"
        EulerAncestralDiscreteScheduler = "EulerAncestralDiscreteScheduler"
        UniPCMultistepScheduler = "UniPCMultistepScheduler"
        KDPM2DiscreteScheduler = "KDPM2DiscreteScheduler"
        DPMSolverSinglestepScheduler = "DPMSolverSinglestepScheduler"
        KDPM2AncestralDiscreteScheduler = "KDPM2AncestralDiscreteScheduler"

    @classmethod
    def get_scheduler_class(cls, scheduler: StableDiffusionScheduler):
        if scheduler == cls.StableDiffusionScheduler.DPMSolverSDEScheduler:
            return DPMSolverSDEScheduler
        elif scheduler == cls.StableDiffusionScheduler.EulerDiscreteScheduler:
            return EulerDiscreteScheduler
        elif scheduler == cls.StableDiffusionScheduler.LMSDiscreteScheduler:
            return LMSDiscreteScheduler
        elif scheduler == cls.StableDiffusionScheduler.DDIMScheduler:
            return DDIMScheduler
        elif scheduler == cls.StableDiffusionScheduler.DDPMScheduler:
            return DDPMScheduler
        elif scheduler == cls.StableDiffusionScheduler.HeunDiscreteScheduler:
            return HeunDiscreteScheduler
        elif scheduler == cls.StableDiffusionScheduler.DPMSolverMultistepScheduler:
            return DPMSolverMultistepScheduler
        elif scheduler == cls.StableDiffusionScheduler.DEISMultistepScheduler:
            return DEISMultistepScheduler
        elif scheduler == cls.StableDiffusionScheduler.PNDMScheduler:
            return PNDMScheduler
        elif scheduler == cls.StableDiffusionScheduler.EulerAncestralDiscreteScheduler:
            return EulerAncestralDiscreteScheduler
        elif scheduler == cls.StableDiffusionScheduler.UniPCMultistepScheduler:
            return UniPCMultistepScheduler
        elif scheduler == cls.StableDiffusionScheduler.KDPM2DiscreteScheduler:
            return KDPM2DiscreteScheduler
        elif scheduler == cls.StableDiffusionScheduler.DPMSolverSinglestepScheduler:
            return DPMSolverSinglestepScheduler
        elif scheduler == cls.StableDiffusionScheduler.KDPM2AncestralDiscreteScheduler:
            return KDPM2AncestralDiscreteScheduler

    model: HFTextToImage = Field(
        default=HFTextToImage(),
        description="The Stable Diffusion XL model to use for generation.",
    )
    variant: ModelVariant = Field(
        default=ModelVariant.FP16,
        description="The variant of the model to use for generation.",
    )
    prompt: str = Field(default="", description="The prompt for image generation.")
    negative_prompt: str = Field(
        default="",
        description="The negative prompt to guide what should not appear in the generated image.",
    )
    width: int = Field(
        default=1024, ge=64, le=2048, description="Width of the generated image."
    )
    height: int = Field(
        default=1024, ge=64, le=2048, description="Height of the generated image"
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        le=1000000,
        description="Seed for the random number generator.",
    )
    num_inference_steps: int = Field(
        default=25, ge=1, le=100, description="Number of inference steps."
    )
    guidance_scale: float = Field(
        default=7.0, ge=0.0, le=20.0, description="Guidance scale for generation."
    )
    scheduler: StableDiffusionScheduler = Field(
        default=StableDiffusionScheduler.EulerDiscreteScheduler,
        description="The scheduler to use for the diffusion process.",
    )
    pag_scale: float = Field(
        default=3.0,
        ge=0.0,
        le=10.0,
        description="Scale of the Perturbed-Attention Guidance applied to the image.",
    )
    loras: list[HFLoraSDXLConfig] = Field(
        default=[],
        description="The LoRA models to use for image processing",
    )
    lora_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of the LoRAs",
    )
    enable_attention_slicing: bool = Field(
        default=True,
        description="Enable attention slicing for the pipeline. This can reduce VRAM usage.",
    )
    enable_tiling: bool = Field(
        default=False,
        description="Enable tiling for the VAE. This can reduce VRAM usage.",
    )
    enable_cpu_offload: bool = Field(
        default=False,
        description="Enable CPU offload for the pipeline. This can reduce VRAM usage.",
    )

    class StableDiffusionOutputType(str, Enum):
        IMAGE = "Image"
        LATENT = "Latent"

    output_type: StableDiffusionOutputType = Field(
        default=StableDiffusionOutputType.IMAGE,
        description="The type of output to generate.",
    )
    _loaded_adapters: set[str] = set()
    _pipeline: Any = None

    @classmethod
    def get_basic_fields(cls):
        return ["model", "prompt", "width", "height"]

    @classmethod
    def get_recommended_models(cls):
        return HF_STABLE_DIFFUSION_XL_MODELS

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not StableDiffusionXLBase

    async def pre_process(self, context: ProcessingContext):
        log.debug(f"Pre-processing Stable Diffusion node with seed: {self.seed}")
        if self.seed == -1:
            original_seed = self.seed
            self.seed = randint(0, 2**32 - 1)
            log.debug(f"Generated random seed: {original_seed} -> {self.seed}")
        log.debug(
            f"Pre-process complete. Model: {self.model.repo_id}, Seed: {self.seed}"
        )

    def should_skip_cache(self):
        return len(self.loras) > 0

    def _set_scheduler(self, scheduler_type: StableDiffusionScheduler):
        log.debug(f"Setting scheduler to: {scheduler_type} (XL)")
        log.debug(f"Model repo_id: {self.model.repo_id}")
        scheduler_class = self.get_scheduler_class(scheduler_type)
        log.debug(f"Scheduler class: {scheduler_class.__name__}")
        scheduler = getattr(self._pipeline, "scheduler", None)
        if scheduler is None:
            log.warning(
                "Current pipeline does not expose a scheduler; skipping scheduler update (XL)"
            )
            return

        if "turbo" in self.model.repo_id:
            log.debug("Using turbo mode with trailing timestep spacing")
            self._pipeline.scheduler = scheduler_class.from_config(
                scheduler.config,
                timestep_spacing="trailing",
            )
        else:
            log.debug("Using standard timestep spacing")
            self._pipeline.scheduler = scheduler_class.from_config(
                scheduler.config,
            )
        log.debug("Scheduler set successfully (XL)")

    async def move_to_device(self, device: str):
        log.debug(f"Moving pipeline to device: {device}")
        if self._pipeline is not None:
            log.debug("Pipeline found, moving to device")
            self._pipeline.to(device)
            log.debug(f"Pipeline moved to device: {device}")
        else:
            log.debug("No pipeline to move to device")

    def _setup_generator(self):
        log.debug("Setting up generator (XL)")
        log.debug(f"Current seed value: {self.seed}")
        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            log.debug(f"Setting manual seed (XL): {self.seed}")
            generator = generator.manual_seed(self.seed)
        else:
            log.debug("Using random seed (XL)")
        log.debug("Generator setup complete (XL)")
        return generator

    def progress_callback(self, context: ProcessingContext):
        def callback(
            pipeline, step: int, timestep: int, kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )
            return kwargs

        return callback

    async def run_pipeline(
        self, context: ProcessingContext, **kwargs
    ) -> ImageRef | TorchTensor:
        log.debug("Starting pipeline execution (XL)")
        if self._pipeline is None:
            log.error("Pipeline not initialized")
            raise ValueError("Pipeline not initialized")

        log.debug(f"Enable attention slicing: {self.enable_attention_slicing}")
        if self.enable_attention_slicing and hasattr(
            self._pipeline, "enable_attention_slicing"
        ):
            log.debug("Enabling attention slicing")
            self._pipeline.enable_attention_slicing()

        log.debug(f"Enable tiling: {self.enable_tiling}")
        if self.enable_tiling and hasattr(self._pipeline, "vae"):
            log.debug("Enabling VAE tiling (XL)")
            vae = getattr(self._pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()

        log.debug(f"Enable CPU offload: {self.enable_cpu_offload}")
        if self.enable_cpu_offload and hasattr(
            self._pipeline, "enable_model_cpu_offload"
        ):
            log.debug("Enabling model CPU offload (XL)")
            self._pipeline.enable_model_cpu_offload()

        loras = [
            lora for lora in self.loras if not lora.lora.path in self._loaded_adapters
        ]
        log.debug(f"New LoRAs to load (XL): {len(loras)}")
        load_loras(self._pipeline, loras)
        self._loaded_adapters.update(lora.lora.path for lora in loras if lora.lora.path)
        log.debug(f"Total loaded adapters (XL): {len(self._loaded_adapters)}")

        log.debug("Setting up generator (XL)")
        generator = self._setup_generator()

        log.debug("Starting pipeline inference (XL)")
        log.debug(
            f"Prompt: {self.prompt[:100]}{'...' if len(self.prompt) > 100 else ''}"
        )
        log.debug(
            f"Negative prompt: {self.negative_prompt[:100]}{'...' if len(self.negative_prompt) > 100 else ''}"
        )
        log.debug(f"Inference steps: {self.num_inference_steps}")
        log.debug(f"Guidance scale: {self.guidance_scale}")
        log.debug(f"Dimensions: {self.width}x{self.height}")
        log.debug(f"LoRA scale: {self.lora_scale}")

        def _run_pipeline_sync_xl():
            call_kwargs: dict[str, Any] = {
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "width": self.width,
                "height": self.height,
                "callback_on_step_end": self.progress_callback(context),
                "generator": generator,
                "output_type": self.output_type.value,
            }

            if self._loaded_adapters:
                call_kwargs["cross_attention_kwargs"] = {"scale": self.lora_scale}

            call_kwargs.update(kwargs)
            return self._pipeline(**call_kwargs)

        output = await asyncio.to_thread(_run_pipeline_sync_xl)
        image = output.images[0]

        log.debug("Pipeline inference completed (XL)")

        if self.output_type == self.StableDiffusionOutputType.IMAGE:
            log.debug("Converting PIL image to ImageRef (XL)")
            result = await context.image_from_pil(image)
            log.debug("Pipeline execution completed successfully (XL)")
            return result

        log.debug("Returning tensor output (XL)")
        result = TorchTensor.from_tensor(image)
        log.debug("Pipeline execution completed successfully (XL)")
        return result

    async def process(self, context: ProcessingContext) -> ImageRef:
        raise NotImplementedError("Subclasses must implement this method")
