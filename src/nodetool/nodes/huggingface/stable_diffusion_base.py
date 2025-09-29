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
    HFLTXV,
    HFCLIPVision,
    HFControlNet,
    HFFlux,
    HFIPAdapter,
    HFLoraSDConfig,
    HFLoraSDXLConfig,
    HFStableDiffusion,
    HFStableDiffusion3,
    HFStableDiffusionXL,
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


class IPAdapter_SDXL_Model(str, Enum):
    NONE = ""
    IP_ADAPTER = "ip-adapter_sdxl.bin"
    IP_ADAPTER_PLUS = "ip-adapter-plus_sdxl_vit-h.bin"


class IPAdapter_SD15_Model(str, Enum):
    NONE = ""
    IP_ADAPTER = "ip-adapter_sd15.bin"
    IP_ADAPTER_LIGHT = "ip-adapter_sd15_light_v11.bin"
    IP_ADAPTER_PLUS = "ip-adapter-plus_sd15_vit-G.bin"


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
        repo_id="SG161222/Realistic_Vision_V5.1_noVAE",
        path="Realistic_Vision_V5.1_fp16-no-ema.safetensors",
    ),
    HFStableDiffusion(
        repo_id="digiplay/majicMIX_realistic_v7",
        path="majicmixRealistic_v7.safetensors",
    ),
    HFStableDiffusion(
        repo_id="philz1337x/epicrealism",
        path="epicrealism_naturalSinRC1VAE.safetensors",
    ),
    HFStableDiffusion(
        repo_id="Lykon/DreamShaper",
        path="DreamShaper_5_beta2_noVae_half_pruned.safetensors",
    ),
    HFStableDiffusion(
        repo_id="Lykon/DreamShaper",
        path="DreamShaper_4BakedVae_fp16.safetensors",
    ),
    HFStableDiffusion(
        repo_id="XpucT/Deliberate",
        path="Deliberate_v6.safetensors",
    ),
    HFStableDiffusion(
        repo_id="XpucT/Deliberate",
        path="Deliberate_v6-inpainting.safetensors",
    ),
    HFStableDiffusion(
        repo_id="Lykon/AbsoluteReality",
        path="AbsoluteReality_1.8.1_pruned.safetensors",
    ),
    HFStableDiffusion(
        repo_id="Lykon/AbsoluteReality",
        path="AbsoluteReality_1.8.1_INPAINTING.inpainting.safetensors",
    ),
    HFStableDiffusion(
        repo_id="gsdf/Counterfeit-V2.5",
        path="Counterfeit-V2.5_fp16.safetensors",
    ),
    # HFStableDiffusion(
    #     repo_id="Yntec/Deliberate2",
    #     path="Deliberate_v2.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/epiCPhotoGasm", path="epiCPhotoGasmVAE.safetensors"
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/YiffyMix",
    #     path="yiffymix_v31_vae.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/DreamPhotoGASM",
    #     path="DreamPhotoGASM.safetensors",
    # ),
    # HFStableDiffusion(repo_id="Yntec/epiCEpic", path="epiCEpic.safetensors"),
    # HFStableDiffusion(
    #     repo_id="Yntec/HyperRealism",
    #     path="Hyper_Realism_1.2_fp16.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/AbsoluteReality",
    #     path="absolutereality_v16.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/RealLife",
    #     path="reallife_v20.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/beLIEve",
    #     path="beLIEve.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/photoMovieXFinal",
    #     path="photoMovieXFinal.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/aMovieX",
    #     path="AmovieX.safetensors",
    # ),
    # HFStableDiffusion(repo_id="Yntec/Paramount", path="Paramount.safetensors"),
    # HFStableDiffusion(
    #     repo_id="Yntec/realisticStockPhoto3",
    #     path="realisticStockPhoto_v30SD15.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/Analog",
    #     path="Analog.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/insaneRealistic_v2",
    #     path="insaneRealistic_v20.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/CyberRealistic",
    #     path="CyberRealistic20.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/photoMovieRealistic",
    #     path="photoMovieRealistic-no-ema.safetensors",
    # ),
    # HFStableDiffusion(repo_id="Yntec/VisionVision", path="VisionVision.safetensors"),
    # HFStableDiffusion(
    #     repo_id="Yntec/Timeless",
    #     path="Timeless.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/HyperRemix",
    #     path="HyperRemix.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/HyperPhotoGASM",
    #     path="HyperPhotoGASM.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/ZootVision", path="zootvisionAlpha_v10Alpha.safetensors"
    # ),
    # HFStableDiffusion(repo_id="Yntec/ChunkyCat", path="ChunkyCat.safetensors"),
    # HFStableDiffusion(
    #     repo_id="Yntec/TickleYourFancy",
    #     path="TickleYourFancy.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/AllRoadsLeadToRetro",
    #     path="AllRoadsLeadToRetro.safetensors",
    # ),
    # HFStableDiffusion(repo_id="Yntec/ClayStyle", path="ClayStyle.safetensors"),
    # HFStableDiffusion(repo_id="Yntec/epiCDream", path="epicdream_lullaby.safetensors"),
    # HFStableDiffusion(
    #     repo_id="Yntec/Epsilon_Naught",
    #     path="Epsilon_Naught.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/BetterPonyDiffusion",
    #     path="betterPonyDiffusionV6_v20.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/ZootVisionEpsilon",
    #     path="zootvisionEpsilon_v50Epsilon.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/RevAnimatedV2Rebirth",
    #     path="revAnimated_v2RebirthVAE.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/AnimephilesAnonymous",
    #     path="AnimephilesAnonymous.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/GrandPrix",
    #     path="GrandPrix.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/InsaneSurreality",
    #     path="InsaneSurreality.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/DreamlikePhotoReal2",
    #     path="DreamlikePhotoReal2.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/DreamShaperRemix",
    #     path="DreamShaperRemix.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/epiCVision",
    #     path="epiCVision.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/IncredibleWorld2",
    #     path="incredibleWorld_v20.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/CrystalReality",
    #     path="CrystalReality.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/ZooFun",
    #     path="ZooFun.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/DreamWorks",
    #     path="DreamWorks.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/AnythingV7",
    #     path="AnythingV7.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/ICantBelieveItSNotPhotography",
    #     path="icbinpICantBelieveIts_v10_pruned.safetensors",
    # ),
    # HFStableDiffusion(repo_id="Yntec/fennPhoto", path="fennPhoto_v10.safetensors"),
    # HFStableDiffusion(
    #     repo_id="Yntec/Surreality",
    #     path="ChainGirl-Surreality.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/WinningBlunder",
    #     path="WinningBlunder.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/Neurogen",
    #     path="NeurogenVAE.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/Hyperlink",
    #     path="Hyperlink.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/Disneyify",
    #     path="Disneyify_v1.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/DisneyPixarCartoon768",
    #     path="disneyPixarCartoonVAE.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/Wonder",
    #     path="Wonder.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/Voxel",
    #     path="VoxelVAE.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/Vintage",
    #     path="Vintage.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/BeautyFoolRemix",
    #     path="BeautyFoolRemix.safetensors",
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/handpaintedRPGIcons",
    #     path="handpaintedRPGIcons_v1.safetensors",
    # ),
    # HFStableDiffusion(repo_id="Yntec/526Mix", path="526mixV15.safetensors"),
    # HFStableDiffusion(repo_id="Yntec/majicmixLux", path="majicmixLux_v1.safetensors"),
    # HFStableDiffusion(
    #     repo_id="Yntec/incha_re_zoro", path="inchaReZoro_v10.safetensors"
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/3DCartoonVision", path="3dCartoonVision_v10.safetensors"
    # ),
    # HFStableDiffusion(repo_id="Yntec/RetroRetro", path="RetroRetro.safetensors"),
    # HFStableDiffusion(repo_id="Yntec/ClassicToons", path="ClassicToons.safetensors"),
    # HFStableDiffusion(repo_id="Yntec/PixelKicks", path="PixelKicks.safetensors"),
    # HFStableDiffusion(
    #     repo_id="Yntec/NostalgicLife", path="NostalgicLifeVAE.safetensors"
    # ),
    # HFStableDiffusion(
    #     repo_id="Yntec/ArthemyComics",
    #     path="arthemyComics_v10Bakedvae.safetensors",
    # ),
]

HF_STABLE_DIFFUSION_XL_MODELS = [
    HFStableDiffusionXL(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        path="sd_xl_base_1.0.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        path="sd_xl_refiner_1.0.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="playgroundai/playground-v2.5-1024px-aesthetic",
        path="playground-v2.5-1024px-aesthetic.fp16.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="RunDiffusion/Juggernaut-XL-v9",
        path="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="dataautogpt3/ProteusV0.5",
        path="proteusV0.5.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="Lykon/dreamshaper-xl-lightning",
        path="DreamShaperXL_Lightning.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="Lykon/AAM_XL_AnimeMix",
        path="AAM_XL_Anime_Mix.safetensors",
    ),
    HFStableDiffusionXL(
        repo_id="stabilityai/sdxl-turbo", path="sd_xl_turbo_1.0_fp16.safetensors"
    ),
    HFStableDiffusionXL(
        repo_id="Lykon/dreamshaper-xl-v2-turbo",
        path="DreamShaperXL_Turbo_v2_1.safetensors",
    ),
]

HF_STABLE_DIFFUSION_3_MODELS = [
    HFStableDiffusion3(
        repo_id="Comfy-Org/stable-diffusion-3.5-fp8",
        path="sd3.5_large_fp8_scaled.safetensors",
    ),
    HFStableDiffusion3(
        repo_id="Comfy-Org/stable-diffusion-3.5-fp8",
        path="sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors",
    ),
]

HF_FLUX_MODELS = [
    HFFlux(
        repo_id="Comfy-Org/flux1-dev",
        path="flux1-dev-fp8.safetensors",
    ),
    HFFlux(
        repo_id="Comfy-Org/flux1-schnell",
        path="flux1-schnell-fp8.safetensors",
    ),
    HFFlux(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        path="flux1-fill-dev.safetensors",
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
        repo_id="lllyasviel/control_v11p_sd15_tile",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_shuffle",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_ip2p",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_lineart",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_lineart_anime",
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
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_scribble",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_seg",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_hed",
        path="diffusion_pytorch_model.fp16.safetensors",
    ),
    HFControlNet(
        repo_id="lllyasviel/control_v11p_sd15_normalbae",
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
    HFLTXV(
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

HF_UNET_MODELS = [
    HFUnet(
        repo_id="Comfy-Org/mochi_preview_repackaged",
        path="split_files/diffusion_models/mochi_preview_bf16.safetensors",
    ),
    HFUnet(
        repo_id="Comfy-Org/mochi_preview_repackaged",
        path="split_files/diffusion_models/mochi_preview_fp8_scaled.safetensors",
    ),
    HFUnet(
        repo_id="black-forest-labs/FLUX.1-dev",
        path="flux1-dev.safetensors",
    ),
    HFUnet(
        repo_id="black-forest-labs/FLUX.1-schnell",
        path="flux1-schnell.safetensors",
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

    model: HFStableDiffusion = Field(
        default=HFStableDiffusion(),
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
        return HF_IP_ADAPTER_MODELS + HF_STABLE_DIFFUSION_MODELS + HF_IP_ADAPTER_MODELS

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

    def _load_ip_adapter(self):
        log.debug("Checking IP Adapter configuration")
        log.debug(f"IP Adapter model repo_id: {self.ip_adapter_model.repo_id}")
        log.debug(f"IP Adapter model path: {self.ip_adapter_model.path}")
        log.debug(f"IP Adapter scale: {self.ip_adapter_scale}")

        if self.ip_adapter_model.repo_id != "" and self.ip_adapter_model.path:
            log.debug("IP Adapter model is configured, loading from cache")
            cache_path = try_to_load_from_cache(
                self.ip_adapter_model.repo_id, self.ip_adapter_model.path
            )
            if cache_path is None:
                log.error(
                    f"IP Adapter cache not found for {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path}"
                )
                raise ValueError(
                    f"Install the {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path} IP Adapter model to use it (Recommended Models above)"
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
            self._pipeline.load_ip_adapter(
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
        self._pipeline.scheduler = scheduler_class.from_config(
            self._pipeline.scheduler.config
        )
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
        if self.enable_tiling:
            log.debug("Enabling VAE tiling")
            self._pipeline.vae.enable_tiling()

        log.debug(f"Enable CPU offload: {self.enable_cpu_offload}")
        if self.enable_cpu_offload:
            log.debug("Enabling model CPU offload")
            self._pipeline.enable_model_cpu_offload()

        log.debug(f"Enable attention slicing: {self.enable_attention_slicing}")
        if self.enable_attention_slicing:
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

        log.debug(f"IP Adapter image set: {self.ip_adapter_image.is_set()}")
        if self.ip_adapter_image.is_set():
            if self.ip_adapter_model.repo_id == "":
                log.error("IP Adapter image provided but no model selected")
                raise ValueError("Select an IP Adapter model")
            log.debug("Converting IP Adapter image to PIL")
            ip_adapter_image = await context.image_to_pil(self.ip_adapter_image)
        else:
            ip_adapter_image = None

        log.debug(f"Setting IP Adapter scale: {self.ip_adapter_scale}")
        self._pipeline.set_ip_adapter_scale(self.ip_adapter_scale)

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

            return self._pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
                latents=latents,
                ip_adapter_image=ip_adapter_image,
                cross_attention_kwargs={"scale": 1.0},
                callback_on_step_end=self.progress_callback(
                    context, 0, self.num_inference_steps
                ),
                pag_scale=self.pag_scale,
                output_type=self.output_type.value,
                **kwargs,
            )

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

    model: HFStableDiffusionXL = Field(
        default=HFStableDiffusionXL(),
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

    def _set_scheduler(self, scheduler_type: StableDiffusionScheduler):
        log.debug(f"Setting scheduler to: {scheduler_type} (XL)")
        log.debug(f"Model repo_id: {self.model.repo_id}")
        scheduler_class = self.get_scheduler_class(scheduler_type)
        log.debug(f"Scheduler class: {scheduler_class.__name__}")

        if "turbo" in self.model.repo_id:
            log.debug("Using turbo mode with trailing timestep spacing")
            self._pipeline.scheduler = scheduler_class.from_config(
                self._pipeline.scheduler.config,
                timestep_spacing="trailing",
            )
        else:
            log.debug("Using standard timestep spacing")
            self._pipeline.scheduler = scheduler_class.from_config(
                self._pipeline.scheduler.config,
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

    def _load_ip_adapter(self):
        log.debug("Checking IP Adapter configuration (XL)")
        log.debug(f"IP Adapter model repo_id: {self.ip_adapter_model.repo_id}")
        log.debug(f"IP Adapter model path: {self.ip_adapter_model.path}")
        log.debug(f"IP Adapter scale: {self.ip_adapter_scale}")

        if self.ip_adapter_model.repo_id != "" and self.ip_adapter_model.path:
            log.debug("IP Adapter model is configured, loading from cache (XL)")
            cache_path = try_to_load_from_cache(
                self.ip_adapter_model.repo_id, self.ip_adapter_model.path
            )
            if cache_path is None:
                log.error(
                    f"IP Adapter cache not found for {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path}"
                )
                raise ValueError(
                    f"Install the {self.ip_adapter_model.repo_id}/{self.ip_adapter_model.path} IP Adapter model to use it (Recommended Models above)"
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
            self._pipeline.load_ip_adapter(
                self.ip_adapter_model.repo_id,
                subfolder=subfolder,
                weight_name=weight_name,
            )
            log.debug("IP Adapter loaded successfully (XL)")
        else:
            log.debug("No IP Adapter model configured (XL)")

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
        if self.enable_attention_slicing:
            log.debug("Enabling attention slicing")
            self._pipeline.enable_attention_slicing()

        log.debug(f"Enable tiling: {self.enable_tiling}")
        if self.enable_tiling:
            log.debug("Enabling VAE tiling (XL)")
            self._pipeline.vae.enable_tiling()

        log.debug(f"Enable CPU offload: {self.enable_cpu_offload}")
        if self.enable_cpu_offload:
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

        log.debug(f"IP Adapter image set (XL): {self.ip_adapter_image.is_set()}")
        ip_adapter_image = (
            await context.image_to_pil(self.ip_adapter_image)
            if self.ip_adapter_image.is_set()
            else None
        )

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
            return self._pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                width=self.width,
                height=self.height,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_scale=self.ip_adapter_scale,
                cross_attention_kwargs={"scale": self.lora_scale},
                callback_on_step_end=self.progress_callback(context),
                generator=generator,
                output_type=self.output_type.value,
                **kwargs,
            )

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
