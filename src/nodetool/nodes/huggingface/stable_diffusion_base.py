from __future__ import annotations

from enum import Enum
import os
from random import randint
import asyncio
from typing import Any, TYPE_CHECKING, ClassVar


from pydantic import Field

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.huggingface.nunchaku_utils import get_nunchaku_transformer
from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE
from nodetool.metadata.types import (
    HFCLIP,
    HFCLIPVision,
    HFControlNet,
    HFIPAdapter,
    HFLoraSDConfig,
    HFLoraSDXLConfig,
    HFStableDiffusion,
    HFStableDiffusionXL,
    HFUnet,
    ImageRef,
    TorchTensor,
)
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel

from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress

import logging

if TYPE_CHECKING:
    import torch

log = get_logger(__name__)
log.setLevel(logging.DEBUG)


HF_IP_ADAPTER_MODELS = [
    HFIPAdapter(repo_id="h94/IP-Adapter", path="models/ip-adapter_sd15.bin"),
    HFIPAdapter(repo_id="h94/IP-Adapter", path="models/ip-adapter_sd15_light.bin"),
    HFIPAdapter(repo_id="h94/IP-Adapter", path="models/ip-adapter_sd15_vit-G.bin"),
]

HF_IP_ADAPTER_XL_MODELS = [
    HFIPAdapter(repo_id="h94/IP-Adapter", path="sdxl_models/ip-adapter_sdxl.bin"),
    HFIPAdapter(repo_id="h94/IP-Adapter", path="sdxl_models/ip-adapter_sdxl_vit-h.bin"),
    HFIPAdapter(
        repo_id="h94/IP-Adapter", path="sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
    ),
]


HF_STABLE_DIFFUSION_MODELS = [
    HFStableDiffusion(
        repo_id="Lykon/DreamShaper", path="DreamShaper_6.2_BakedVae_pruned.safetensors"
    ),
]

HF_STABLE_DIFFUSION_XL_MODELS = [
    HFStableDiffusionXL(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        path="sd_xl_base_1.0.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="Lykon/dreamshaper-xl-v2-turbo",
        path="DreamShaperXL_Turbo_v2_1.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="RunDiffusion/Juggernaut-XL-v9",
        path="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="dataautogpt3/ProteusV0.3",
        path="ProteusV0.3.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="John6666/prefect-illustrious-xl-v3-sdxl",
    ),
    HFStableDiffusionXL(
        repo_id="cagliostrolab/animagine-xl-4.0",
        path="animagine-xl-4.0-opt.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="SG161222/RealVisXL_V5.0",
        path="RealVisXL_V5.0_fp16.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="nunchaku-tech/nunchaku-sdxl",
        path="svdq-int4_r32-sdxl.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="nunchaku-tech/nunchaku-sdxl-turbo",
        path="svdq-int4_r32-sdxl-turbo.safetensors",
    ),
]

SDXL_BASE_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "vae/*",
    "unet/config.json",
]


HF_CONTROLNET_MODELS: list[HFControlNet] = [
    # Original ControlNet models
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
    # SD Control Collection - IP-Adapter SD15 models
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="ip-adapter_sd15_plus.pth",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="ip-adapter_sd15.pth",
    ),
    # SD Control Collection - Other SD15 models
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="ioclab_sd15_recolor.safetensors",
    ),
]

HF_CONTROLNET_XL_MODELS: list[HFControlNet] = [
    # Original SDXL ControlNet models
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
        path="diffusion_pytorch_model.safetensors",
    ),
    # SD Control Collection - SDXL Canny models
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="diffusers_xl_canny_full.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="diffusers_xl_canny_mid.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="diffusers_xl_canny_small.safetensors",
    ),
    # SD Control Collection - SDXL Depth models
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="diffusers_xl_depth_full.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="diffusers_xl_depth_mid.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="diffusers_xl_depth_small.safetensors",
    ),
    # SD Control Collection - T2I Adapter Diffusers XL models
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="t2i-adapter_diffusers_xl_canny.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="t2i-adapter_diffusers_xl_depth_midas.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="t2i-adapter_diffusers_xl_depth_zoe.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="t2i-adapter_diffusers_xl_lineart.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="t2i-adapter_diffusers_xl_openpose.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="t2i-adapter_diffusers_xl_sketch.safetensors",
    ),
    # SD Control Collection - T2I Adapter XL models
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="t2i-adapter_xl_canny.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="t2i-adapter_xl_openpose.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="t2i-adapter_xl_sketch.safetensors",
    ),
    # SD Control Collection - OpenPose models
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="thibaud_xl_openpose.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="thibaud_xl_openpose_256lora.safetensors",
    ),
    # SD Control Collection - SargeZT Depth models
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="sargezt_xl_depth_faid_vidit.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="sargezt_xl_depth_zeed.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="sargezt_xl_depth.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="sargezt_xl_softedge.safetensors",
    ),
    # SD Control Collection - IP-Adapter XL models
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="ip-adapter_xl.pth",
    ),
    # SD Control Collection - Kohya ControlLite XL models
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="kohya_controllllite_xl_depth_anime.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="kohya_controllllite_xl_canny_anime.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="kohya_controllllite_xl_scribble_anime.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="kohya_controllllite_xl_openpose_anime.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="kohya_controllllite_xl_openpose_anime_v2.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="kohya_controllllite_xl_blur_anime_beta.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="kohya_controllllite_xl_blur.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="kohya_controllllite_xl_blur_anime.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="kohya_controllllite_xl_canny.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/sd_control_collection",
        path="kohya_controllllite_xl_depth.safetensors",
    ),
    # Qinglong ControlNet-LLLite XL models
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_canny.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_depth.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_depth_V2.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_dw_openpose.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_lineart_anime_denoise.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_mlsd_V2.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_normal.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_normal_dsine.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_recolor_luminance.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_segment_animeface.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_segment_animeface_V2.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_sketch.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_softedge.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_t2i-adapter_color_shuffle.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_tile_anime_alpha.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_tile_anime_beta.safetensors",
    ),
    HFControlNet(
        repo_id="bdsqlsz/qinglong_controlnet-lllite",
        path="bdsqlsz_controlllite_xl_tile_realistic.safetensors",
    ),
]


class StableDiffusionDetailLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class StableDiffusionXLQuantization(Enum):
    FP16 = "fp16"
    FP4 = "fp4"
    INT4 = "int4"


async def load_loras(pipeline: Any, loras: list[HFLoraSDConfig] | list[HFLoraSDXLConfig]):
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
        cache_path = await HF_FAST_CACHE.resolve(
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
    import torch

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


def _select_diffusion_dtype(variant: ModelVariant | None = None) -> "torch.dtype":
    import torch

    if variant == ModelVariant.FP32:
        return torch.float32
    if variant == ModelVariant.FP16:
        return torch.float16
    if variant == ModelVariant.BF16:
        return torch.bfloat16

    # Prefer BF16 on capable GPUs (PyTorch 2 optimization path), otherwise fall back.
    try:
        if torch.cuda.is_available():
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(is_bf16_supported) and is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.float16
    except Exception:
        pass

    return torch.float32


class StableDiffusionBaseNode(HuggingFacePipelineNode):

    variant: ClassVar[ModelVariant] = ModelVariant.DEFAULT

    def get_torch_dtype(self) -> "torch.dtype":
        return _select_diffusion_dtype()

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
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )
        from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
        from diffusers.schedulers.scheduling_dpmsolver_singlestep import (
            DPMSolverSinglestepScheduler,
        )
        from diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete import (
            KDPM2AncestralDiscreteScheduler,
        )

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

    model: HFStableDiffusion = Field(
        default=HFStableDiffusion(),
        description="The model to use for image generation.",
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
    ip_adapter_model: HFIPAdapter = Field(
        default=HFIPAdapter(),
        description="The IP adapter model to use for image processing",
    )
    ip_adapter_image: ImageRef = Field(
        default=ImageRef(),
        description="When provided the image will be fed into the IP adapter",
    )
    ip_adapter_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The strength of the IP adapter",
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
        default=False,
        description="Legacy VAE tiling flag (disabled in favor of PyTorch 2 attention optimizations).",
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
        return HF_IP_ADAPTER_MODELS + HF_STABLE_DIFFUSION_MODELS

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
        if self.ip_adapter_model.repo_id != "":
            return True
        if len(self.loras) > 0:
            return True
        return False

    async def _load_ip_adapter(self):
        log.debug("Checking IP Adapter configuration")
        log.debug(f"IP Adapter model repo_id: {self.ip_adapter_model.repo_id}")
        log.debug(f"IP Adapter model path: {self.ip_adapter_model.path}")
        log.debug(f"IP Adapter scale: {self.ip_adapter_scale}")

        if self.ip_adapter_model.repo_id != "" and self.ip_adapter_model.path:
            if self._pipeline is None:
                log.error("Pipeline not initialized when loading IP Adapter")
                raise ValueError("Pipeline must be initialized before loading IP Adapter")

            if not hasattr(self._pipeline, "load_ip_adapter"):
                log.error("Current pipeline does not support IP Adapter loading")
                raise ValueError(
                    "The current pipeline does not support IP Adapter. "
                    "Use a Stable Diffusion pipeline with IP Adapter support."
                )

            log.debug("IP Adapter model is configured, loading from cache")
            cache_path = await HF_FAST_CACHE.resolve(
                self.ip_adapter_model.repo_id, self.ip_adapter_model.path
            )
            if cache_path is None:
                log.error(
                    f"IP Adapter cache not found for {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path}"
                )
                raise ValueError(
                    f"Install the {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path} "
                    "IP Adapter model to use it (Recommended Models above)"
                )
            path_parts = self.ip_adapter_model.path.split("/")
            subfolder = "/".join(path_parts[0:-1])
            weight_name = path_parts[-1]
            log.info(
                f"Loading IP Adapter {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path}"
            )
            log.debug(f"IP Adapter cache path: {cache_path}")
            log.debug(f"IP Adapter subfolder: {subfolder}")
            log.debug(f"IP Adapter weight name: {weight_name}")
            self._pipeline.load_ip_adapter(  # type: ignore[call-arg]
                self.ip_adapter_model.repo_id,
                subfolder=subfolder,
                weight_name=weight_name,
            )
            log.debug("IP Adapter loaded successfully")
        else:
            log.debug("No IP Adapter model configured")

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
        import torch

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
        await load_loras(self._pipeline, loras)
        self._loaded_adapters.update(lora.lora.path for lora in loras if lora.lora.path)
        log.debug(f"Total loaded adapters: {len(self._loaded_adapters)}")

        log.debug("Setting up generator")
        generator = self._setup_generator()

        log.debug(f"IP Adapter image set: {self.ip_adapter_image.is_set()}")
        if self.ip_adapter_image.is_set():
            if self.ip_adapter_model.repo_id == "":
                log.error("IP Adapter image provided but no model selected")
                raise ValueError("Select an IP Adapter model")
            log.debug("Converting IP Adapter image to PIL")
            ip_adapter_image = await context.image_to_pil(self.ip_adapter_image)
        else:
            ip_adapter_image = None

        if hasattr(self._pipeline, "set_ip_adapter_scale"):
            log.debug(f"Setting IP Adapter scale: {self.ip_adapter_scale}")
            self._pipeline.set_ip_adapter_scale(self.ip_adapter_scale)  # type: ignore[attr-defined]

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
        log.debug(f"Output type: {self.output_type.value}")

        def _run_pipeline_sync():
            import torch

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
                "output_type": self.output_type.value,
            }

            if ip_adapter_image is not None:
                call_kwargs["ip_adapter_image"] = ip_adapter_image

            if self._loaded_adapters:
                call_kwargs["cross_attention_kwargs"] = {"scale": 1.0}

            call_kwargs.update(kwargs)
            with torch.inference_mode():
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

    variant: ClassVar[ModelVariant] = ModelVariant.DEFAULT

    def get_torch_dtype(self) -> torch.dtype:
        return _select_diffusion_dtype()

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
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )
        from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
        from diffusers.schedulers.scheduling_dpmsolver_singlestep import (
            DPMSolverSinglestepScheduler,
        )
        from diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete import (
            KDPM2AncestralDiscreteScheduler,
        )

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

    model: HFStableDiffusionXL = Field(
        default=HFStableDiffusionXL(),
        description="The Stable Diffusion XL model to use for generation.",
    )
    quantization: StableDiffusionXLQuantization = Field(
        default=StableDiffusionXLQuantization.FP16,
        description="Quantization level for Stable Diffusion XL (enable INT4/FP4 to use a Nunchaku UNet).",
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
    ip_adapter_model: HFIPAdapter = Field(
        default=HFIPAdapter(),
        description="The IP adapter model to use for image processing",
    )
    ip_adapter_image: ImageRef = Field(
        default=ImageRef(),
        description="When provided the image will be fed into the IP adapter",
    )
    ip_adapter_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of the IP adapter image",
    )
    enable_attention_slicing: bool = Field(
        default=True,
        description="Enable attention slicing for the pipeline. This can reduce VRAM usage.",
    )
    enable_tiling: bool = Field(
        default=False,
        description="Legacy VAE tiling flag (disabled in favor of PyTorch 2 attention optimizations).",
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
        return ["model", "quantization", "prompt", "width", "height"]

    @classmethod
    def get_recommended_models(cls):
        return HF_IP_ADAPTER_XL_MODELS + HF_STABLE_DIFFUSION_XL_MODELS

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
        if self.ip_adapter_model.repo_id != "":
            return True
        if len(self.loras) > 0:
            return True
        return False

    async def _load_ip_adapter(self):
        log.debug("Checking IP Adapter configuration (XL)")
        log.debug(f"IP Adapter model repo_id: {self.ip_adapter_model.repo_id}")
        log.debug(f"IP Adapter model path: {self.ip_adapter_model.path}")
        log.debug(f"IP Adapter scale: {self.ip_adapter_scale}")

        if self.ip_adapter_model.repo_id != "" and self.ip_adapter_model.path:
            if self._pipeline is None:
                log.error("Pipeline not initialized when loading IP Adapter (XL)")
                raise ValueError(
                    "Pipeline must be initialized before loading IP Adapter (XL)"
                )

            if not hasattr(self._pipeline, "load_ip_adapter"):
                log.error("Current XL pipeline does not support IP Adapter loading")
                raise ValueError(
                    "The current XL pipeline does not support IP Adapter. "
                    "Use a Stable Diffusion XL pipeline with IP Adapter support."
                )

            log.debug("IP Adapter model is configured, loading from cache (XL)")
            cache_path = await HF_FAST_CACHE.resolve(
                self.ip_adapter_model.repo_id, self.ip_adapter_model.path
            )
            if cache_path is None:
                log.error(
                    f"IP Adapter cache not found for {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path}"
                )
                raise ValueError(
                    f"Install the {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path} "
                    "IP Adapter model to use it (Recommended Models above)"
                )
            path_parts = self.ip_adapter_model.path.split("/")
            subfolder = "/".join(path_parts[0:-1])
            weight_name = path_parts[-1]
            log.info(
                f"Loading IP Adapter {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path}"
            )
            log.debug(f"IP Adapter cache path: {cache_path}")
            log.debug(f"IP Adapter subfolder: {subfolder}")
            log.debug(f"IP Adapter weight name: {weight_name}")
            self._pipeline.load_ip_adapter(  # type: ignore[call-arg]
                self.ip_adapter_model.repo_id,
                subfolder=subfolder,
                weight_name=weight_name,
            )
            log.debug("IP Adapter loaded successfully (XL)")
        else:
            log.debug("No IP Adapter model configured (XL)")

    def _resolve_effective_quantization(self) -> StableDiffusionXLQuantization:
        quantization = self.quantization
        legacy_quantization = self._detect_legacy_quantization()
        if (
            quantization == StableDiffusionXLQuantization.FP16
            and legacy_quantization is not None
        ):
            quantization = legacy_quantization

        smoke_mode = os.getenv("NODETOOL_SMOKE_TEST", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if smoke_mode and quantization != StableDiffusionXLQuantization.FP16:
            log.info(
                "Smoke test detected, forcing Stable Diffusion XL quantization to fp16"
            )
            return StableDiffusionXLQuantization.FP16
        return quantization

    def _detect_legacy_quantization(self) -> StableDiffusionXLQuantization | None:
        repo = (self.model.repo_id or "").lower()
        path = (self.model.path or "").lower()
        if "svdq" not in repo and "svdq" not in path:
            return None
        if "fp4" in repo or "fp4" in path:
            return StableDiffusionXLQuantization.FP4
        return StableDiffusionXLQuantization.INT4

    def _get_base_model(
        self, quantization: StableDiffusionXLQuantization
    ) -> HFStableDiffusionXL:
        if quantization == StableDiffusionXLQuantization.FP16:
            if self.model.repo_id:
                return self.model
            return HFStableDiffusionXL(
                repo_id="stabilityai/stable-diffusion-xl-base-1.0",
                path="sd_xl_base_1.0.safetensors",
            )

        return HFStableDiffusionXL(
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            allow_patterns=SDXL_BASE_ALLOW_PATTERNS,
        )

    def _resolve_transformer_model(
        self,
        quantization: StableDiffusionXLQuantization,
        use_legacy_transformer: bool,
    ) -> HFStableDiffusionXL | None:
        if quantization == StableDiffusionXLQuantization.FP16:
            return None

        if use_legacy_transformer:
            if not self.model.repo_id or not self.model.path:
                raise ValueError(
                    "Legacy Nunchaku SDXL configuration requires model.repo_id and model.path"
                )
            return self.model

        is_turbo = "turbo" in (self.model.repo_id or "").lower()
        if is_turbo:
            path = (
                "svdq-fp4_r32-sdxl-turbo.safetensors"
                if quantization == StableDiffusionXLQuantization.FP4
                else "svdq-int4_r32-sdxl-turbo.safetensors"
            )
            return HFStableDiffusionXL(
                repo_id="nunchaku-tech/nunchaku-sdxl-turbo",
                path=path,
            )

        path = (
            "svdq-fp4_r32-sdxl.safetensors"
            if quantization == StableDiffusionXLQuantization.FP4
            else "svdq-int4_r32-sdxl.safetensors"
        )

        return HFStableDiffusionXL(
            repo_id="nunchaku-tech/nunchaku-sdxl",
            path=path,
        )

    def _resolve_sdxl_pipeline_model_id(
        self, base_model: HFStableDiffusionXL
    ) -> str:
        repo_id = base_model.repo_id or "stabilityai/stable-diffusion-xl-base-1.0"
        repo_id_lower = repo_id.lower()
        if "nunchaku" in repo_id_lower and "sdxl" in repo_id_lower:
            return "stabilityai/stable-diffusion-xl-base-1.0"
        return repo_id

    def _prepare_sdxl_models(
        self,
    ) -> tuple[HFStableDiffusionXL, str, HFStableDiffusionXL | None]:
        quantization = self._resolve_effective_quantization()
        legacy_quantization = self._detect_legacy_quantization()
        use_legacy_transformer = (
            legacy_quantization is not None
            and self.quantization == StableDiffusionXLQuantization.FP16
            and quantization != StableDiffusionXLQuantization.FP16
        )
        base_model = self._get_base_model(quantization)
        pipeline_model_id = self._resolve_sdxl_pipeline_model_id(base_model)
        transformer_model = self._resolve_transformer_model(
            quantization, use_legacy_transformer
        )
        return base_model, pipeline_model_id, transformer_model

    async def _load_sdxl_pipeline(
        self,
        context: ProcessingContext,
        pipeline_class: type,
        torch_dtype,
        base_model: HFStableDiffusionXL,
        pipeline_model_id: str,
        transformer_model: HFStableDiffusionXL | None,
        **pipeline_kwargs,
    ):
        if transformer_model is not None:
            if not transformer_model.path:
                raise ValueError("Nunchaku SDXL requires a transformer path to be set")

            transformer = await get_nunchaku_transformer(
                context=context,
                model_class=NunchakuSDXLUNet2DConditionModel,
                node_id=self.id,
                repo_id=transformer_model.repo_id,
                path=transformer_model.path,
            )
            hf_token = await context.get_secret("HF_TOKEN")
            self._pipeline = await self.load_model(
                context=context,
                model_class=pipeline_class,
                model_id=pipeline_model_id,
                path=base_model.path,
                torch_dtype=torch_dtype,
                unet=transformer,
                use_safetensors=True,
                token=hf_token,
                **pipeline_kwargs,
            )
        else:
            self._pipeline = await self.load_model(
                context=context,
                model_class=pipeline_class,
                model_id=pipeline_model_id,
                path=base_model.path,
                torch_dtype=torch_dtype,
                **pipeline_kwargs,
            )

        _apply_vae_optimizations(self._pipeline)
        self._set_scheduler(self.scheduler)
        await self._load_ip_adapter()

    async def _load_ip_adapter(self):
        if self.ip_adapter_model.repo_id != "" and self.ip_adapter_model.path:
            if self._pipeline is None:
                log.error("Pipeline not initialized when loading IP Adapter (XL)")
                raise ValueError(
                    "Pipeline must be initialized before loading IP Adapter (XL)"
                )

            if not hasattr(self._pipeline, "load_ip_adapter"):
                log.error("Current XL pipeline does not support IP Adapter loading")
                raise ValueError(
                    "The current XL pipeline does not support IP Adapter. "
                    "Use a Stable Diffusion XL pipeline with IP Adapter support."
                )

            log.debug("IP Adapter model is configured, loading from cache (XL)")
            cache_path = await HF_FAST_CACHE.resolve(
                self.ip_adapter_model.repo_id, self.ip_adapter_model.path
            )
            if cache_path is None:
                log.error(
                    f"IP Adapter cache not found for {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path}"
                )
                raise ValueError(
                    f"Install the {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path} "
                    "IP Adapter model to use it (Recommended Models above)"
                )
            path_parts = self.ip_adapter_model.path.split("/")
            subfolder = "/".join(path_parts[0:-1])
            weight_name = path_parts[-1]
            log.info(
                f"Loading IP Adapter {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path}"
            )
            log.debug(f"IP Adapter cache path: {cache_path}")
            log.debug(f"IP Adapter subfolder: {subfolder}")
            log.debug(f"IP Adapter weight name: {weight_name}")
            self._pipeline.load_ip_adapter(  # type: ignore[call-arg]
                self.ip_adapter_model.repo_id,
                subfolder=subfolder,
                weight_name=weight_name,
            )
            log.debug("IP Adapter loaded successfully (XL)")
        else:
            log.debug("No IP Adapter model configured (XL)")

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
        import torch

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
        await load_loras(self._pipeline, loras)
        self._loaded_adapters.update(lora.lora.path for lora in loras if lora.lora.path)
        log.debug(f"Total loaded adapters (XL): {len(self._loaded_adapters)}")

        log.debug("Setting up generator (XL)")
        generator = self._setup_generator()

        log.debug(f"IP Adapter image set (XL): {self.ip_adapter_image.is_set()}")
        ip_adapter_image = (
            await context.image_to_pil(self.ip_adapter_image)
            if self.ip_adapter_image.is_set()
            else None
        )

        if hasattr(self._pipeline, "set_ip_adapter_scale"):
            log.debug(f"Setting IP Adapter scale (XL): {self.ip_adapter_scale}")
            self._pipeline.set_ip_adapter_scale(self.ip_adapter_scale)  # type: ignore[attr-defined]

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
        log.debug(f"Model device: {self._pipeline.device}")

        def _run_pipeline_sync_xl():
            import torch

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
            if ip_adapter_image is not None:
                call_kwargs["ip_adapter_image"] = ip_adapter_image

            call_kwargs.update(kwargs)
            with torch.inference_mode():
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
