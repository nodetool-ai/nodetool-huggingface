"""
HuggingFace image-to-3D generation nodes.

Provides local inference for image-to-3D model generation:
- Shap-E Image-to-3D: Using OpenAI's diffusers pipeline
- Hunyuan3D: Tencent Hunyuan3D-2
- StableFast3D (SF3D): Stability AI
- TripoSR: Stability AI / VAST
- Trellis2: Microsoft TRELLIS.2-4B
- TripoSG: VAST-AI

These nodes run locally and do not require API keys.
"""

from __future__ import annotations

import io
import logging
from enum import Enum
from typing import Any, ClassVar, TYPE_CHECKING

from pydantic import Field

from nodetool.metadata.types import HuggingFaceModel, ImageRef, Model3DRef
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.memory_utils import run_gc

from nodetool.nodes.huggingface._3d_common import (
    _model_revision,
    _check_disk_space,
    _resolve_device,
    _resolve_seed,
    _open_pil_image,
    _export_mesh,
)

if TYPE_CHECKING:
    import torch
    import trimesh
    from PIL import Image

log = logging.getLogger(__name__)


class ShapEImageTo3D(HuggingFacePipelineNode):
    """
    Generate 3D models from images using OpenAI Shap-E.
    3d, generation, image-to-3d, shap-e, mesh, local, reconstruction

    Use cases:
    - Convert images to 3D models locally
    - Create 3D assets from product photos
    - Generate 3D content without API costs
    - Reconstruct 3D objects from single images

    **Platforms:** all (CPU/MPS/CUDA).

    **Note:** Requires diffusers and torch. First run will download the model (~2.5GB).
    """

    # -- static metadata ---------------------------------------------------
    MIN_VRAM_GB: ClassVar[int] = 4
    ESTIMATED_DOWNLOAD_GB: ClassVar[float] = 2.5
    license_warning: ClassVar[str | None] = None  # MIT
    SUPPORTED_PLATFORMS: ClassVar[list[str]] = ["linux", "macos", "windows"]
    INSTALL_HINT: ClassVar[str | None] = None  # uses diffusers, already in core

    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image to convert to 3D",
    )
    guidance_scale: float = Field(
        default=3.0,
        ge=1.0,
        le=15.0,
        description="How strongly to follow the image. Higher = more image adherence.",
    )
    num_inference_steps: int = Field(
        default=64,
        ge=16,
        le=128,
        description="Number of denoising steps. More steps = better quality but slower.",
    )
    frame_size: int = Field(
        default=256,
        ge=64,
        le=512,
        description="Resolution of the internal representation. Higher = more detail.",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed for reproducibility. -1 for random.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="openai/shap-e-img2img",
                # Include safetensors + bin for shap_e_renderer (no safetensors available there)
                allow_patterns=[
                    "**/*.safetensors",
                    "shap_e_renderer/*.bin",
                    "**/*.json",
                    "**/*.txt",
                ],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Shap-E Image-to-3D"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "seed"]

    def requires_gpu(self) -> bool:
        return False

    async def preload_model(self, context: ProcessingContext):
        import torch
        from diffusers import ShapEImg2ImgPipeline

        device = _resolve_device()
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipeline = await self.load_model(
            context=context,
            model_class=ShapEImg2ImgPipeline,
            model_id="openai/shap-e-img2img",
            torch_dtype=torch_dtype,
        )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch

        if self.image.is_empty():
            raise ValueError("Input image is required")

        assert self._pipeline is not None, "Pipeline not initialized"

        # Load input image
        image_io = await context.asset_to_io(self.image)
        input_image = _open_pil_image(image_io, mode="RGB")

        device = str(self._pipeline.device)

        # Set seed – generator device must be "cpu" for MPS pipelines
        gen_device = "cpu" if device.startswith("mps") else device
        seed = _resolve_seed(self.seed)
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Generate
        images = self._pipeline(
            input_image,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            frame_size=self.frame_size,
            generator=generator,
            output_type="mesh",
        ).images

        if not images:
            raise RuntimeError("No 3D model generated")

        mesh = images[0]

        run_gc("After ShapE Image-to-3D inference", log_before_after=False)
        # Export to GLB
        import trimesh

        tri_mesh = trimesh.Trimesh(
            vertices=mesh.verts.cpu().numpy(),
            faces=mesh.faces.cpu().numpy(),
        )
        model_bytes = _export_mesh(tri_mesh, format="glb")

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"shap_e_img_{self.id}.glb",
            format="glb",
            metadata={"seed": seed, "source_model": "openai/shap-e-img2img"},
        )


class Hunyuan3D(HuggingFacePipelineNode):
    """
    Generate 3D meshes from images using Tencent Hunyuan3D-2.
    3d, generation, image-to-3d, hunyuan3d, mesh, local, high-quality

    Use cases:
    - Generate 3D models from images locally
    - Create 3D assets from product photos or concept art
    - Convert single images to 3D meshes
    - Prototype 3D content without API costs

    Produces untextured meshes (shape only). Use external tools for texturing.

    **Platforms:** Linux+CUDA, Windows+CUDA. Installable on macOS but does not run.

    **Requirements:** hy3dgen>=2.0.2 package, torch with CUDA.
    First run downloads ~5GB model (shape-only, not full 75GB repo).
    Standard model needs ~6GB VRAM, mini needs ~5GB. Use low_vram_mode on constrained GPUs.

    Models: https://huggingface.co/tencent/Hunyuan3D-2
    """

    # -- static metadata ---------------------------------------------------
    MIN_VRAM_GB: ClassVar[int] = 5
    ESTIMATED_DOWNLOAD_GB: ClassVar[float] = 5.0
    license_warning: ClassVar[str | None] = (
        "Tencent Hunyuan3D License — non-commercial use only. "
        "See https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE"
    )
    SUPPORTED_PLATFORMS: ClassVar[list[str]] = ["linux", "windows"]
    INSTALL_HINT: ClassVar[str | None] = (
        "Install with: pip install 'nodetool-huggingface[hunyuan3d]'"
    )

    class ModelVariant(str, Enum):
        STANDARD = "standard"
        MINI = "mini"

    class OutputFormat(str, Enum):
        GLB = "glb"
        OBJ = "obj"

    # Mapping from variant enum to HuggingFace repo and subfolder
    VARIANT_CONFIG: ClassVar[dict[str, dict[str, str]]] = {
        "standard": {
            "repo_id": "tencent/Hunyuan3D-2",
            "subfolder": "hunyuan3d-dit-v2-0",
        },
        "mini": {
            "repo_id": "tencent/Hunyuan3D-2mini",
            "subfolder": "hunyuan3d-dit-v2-mini",
        },
    }

    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image to convert to 3D",
    )
    model_variant: ModelVariant = Field(
        default=ModelVariant.STANDARD,
        description="Model variant. Standard (1.1B params, ~6GB VRAM) or Mini (0.6B params, ~5GB VRAM, faster).",
    )
    num_inference_steps: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Number of denoising steps. More steps = better quality but slower.",
    )
    guidance_scale: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="How strongly to follow the input. Higher = more adherence.",
    )
    octree_resolution: int = Field(
        default=384,
        ge=64,
        le=512,
        description="Resolution of the octree for mesh extraction. Higher = more detail.",
    )
    low_vram_mode: bool = Field(
        default=False,
        description="Enable CPU offloading for GPUs with less than 8GB VRAM.",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GLB,
        description="Output format for the 3D model. GLB is recommended; OBJ files are not previewable in the canvas viewer.",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed for reproducibility. -1 for random.",
    )

    _pipeline: Any = None
    _loaded_variant: str | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        # Only download shape-generation files (DiT model contains bundled VAE)
        # Standard model: ~5GB, Mini model: ~2GB
        return [
            HuggingFaceModel(
                repo_id="tencent/Hunyuan3D-2",
                # Only download the shape DiT subfolder (includes bundled VAE weights)
                allow_patterns=[
                    "config.json",
                    "hunyuan3d-dit-v2-0/*.yaml",
                    "hunyuan3d-dit-v2-0/*.safetensors",
                ],
            ),
            HuggingFaceModel(
                repo_id="tencent/Hunyuan3D-2mini",
                allow_patterns=[
                    "config.json",
                    "hunyuan3d-dit-v2-mini/*.yaml",
                    "hunyuan3d-dit-v2-mini/*.safetensors",
                ],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Hunyuan3D-2"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "model_variant", "output_format", "seed"]

    def requires_gpu(self) -> bool:
        return True

    def _ensure_model_downloaded(self, variant: str) -> str:
        """
        Pre-download only the shape-generation files (~5 GB standard, ~2 GB mini)
        to avoid downloading the full repo (which includes paint/texture models).
        Returns the repo_id for the model.
        """
        from huggingface_hub import snapshot_download

        config = self.VARIANT_CONFIG[variant]
        repo_id = config["repo_id"]
        subfolder = config["subfolder"]

        # Only download the specific subfolder needed for shape generation
        # This avoids downloading paint, delight, turbo variants etc.
        _check_disk_space(self.ESTIMATED_DOWNLOAD_GB)
        revision = _model_revision(repo_id)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[
                "config.json",  # Root config
                f"{subfolder}/*",  # DiT model (includes bundled VAE)
            ],
        )
        return repo_id

    def _load_pipeline(self, variant: str):
        """Load or retrieve the Hunyuan3D pipeline for the given variant."""
        from nodetool.ml.core.model_manager import ModelManager

        config = self.VARIANT_CONFIG[variant]
        cache_key = f"hunyuan3d_{config['repo_id']}_{config['subfolder']}"
        cached = ModelManager.get_model(cache_key)
        if cached is not None:
            self._pipeline = cached
        else:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            # Pre-download only shape files to avoid 75GB full repo download
            self._ensure_model_downloaded(variant)

            if self.license_warning:
                log.info("License notice: %s", self.license_warning)

            # Now load the pipeline - hy3dgen will find the cached files
            revision = _model_revision(config["repo_id"])
            self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                config["repo_id"],
                subfolder=config["subfolder"],
                revision=revision,
                use_safetensors=True,
                variant="fp16",
            )

            # Enable CPU offloading if requested
            if self.low_vram_mode:
                try:
                    # hy3dgen pipeline lacks the `components` property that
                    # enable_model_cpu_offload() expects (upstream bug).
                    if not hasattr(self._pipeline, "components"):
                        self._pipeline.components = {
                            "conditioner": self._pipeline.conditioner,
                            "model": self._pipeline.model,
                            "vae": self._pipeline.vae,
                            "scheduler": self._pipeline.scheduler,
                            "image_processor": self._pipeline.image_processor,
                        }
                    self._pipeline.enable_model_cpu_offload()
                except Exception as exc:
                    log.warning(
                        "low_vram_mode unavailable for this hy3dgen version "
                        "(enable_model_cpu_offload failed: %s). Continuing without offloading.",
                        exc,
                    )

            ModelManager.set_model(self.id, cache_key, self._pipeline)

        self._loaded_variant = variant

    async def preload_model(self, context: ProcessingContext):
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except ImportError:
            return
        self._load_pipeline(self.model_variant.value)

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if self.image.is_empty():
            raise ValueError("Input image is required")

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except ImportError:
            raise ImportError(
                "Hunyuan3D requires the hy3dgen package (>=2.0.2). "
                "Install with: pip install hy3dgen>=2.0.2"
            )

        # Load input image
        image_io = await context.asset_to_io(self.image)
        input_image = _open_pil_image(image_io, mode="RGB")

        # Load or reload pipeline if variant changed
        variant = self.model_variant.value
        if self._pipeline is None or self._loaded_variant != variant:
            self._load_pipeline(variant)

        # Set seed using per-call generator (avoids leaking global RNG state)
        seed = _resolve_seed(self.seed)
        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(seed)

        # Generate 3D mesh
        mesh = self._pipeline(
            image=input_image,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            octree_resolution=self.octree_resolution,
            generator=generator,
        )[0]

        run_gc("After Hunyuan3D inference", log_before_after=False)
        # Export mesh to bytes
        format_str = self.output_format.value
        model_bytes = _export_mesh(mesh, format=format_str)

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"hunyuan3d_{self.id}.{format_str}",
            format=format_str,
            metadata={
                "seed": seed,
                "source_model": self.VARIANT_CONFIG[variant]["repo_id"],
            },
        )


class StableFast3D(HuggingFacePipelineNode):
    """
    Generate textured 3D meshes from images in under 1 second using Stability AI SF3D.
    3d, generation, image-to-3d, sf3d, mesh, local, fast, textured

    Use cases:
    - Ultra-fast 3D asset generation from images
    - Create game-ready 3D assets with textures
    - Generate UV-unwrapped meshes with materials
    - Real-time 3D reconstruction workflows

    **Platforms:** Linux+CUDA, Windows+CUDA. macOS Metal experimental (untested).

    **Requirements:** sf3d package, rembg, torch with CUDA.
    First run downloads ~1GB model. Needs ~6GB VRAM.
    Generates textured meshes with UV maps, normal maps, and PBR materials.

    Model: https://huggingface.co/stabilityai/stable-fast-3d
    License: Free for <$1M revenue, enterprise license required above.
    """

    # -- static metadata ---------------------------------------------------
    MIN_VRAM_GB: ClassVar[int] = 6
    ESTIMATED_DOWNLOAD_GB: ClassVar[float] = 1.0
    license_warning: ClassVar[str | None] = (
        "Stability AI Community License — free under $1 M annual revenue, "
        "enterprise license required above. "
        "See https://huggingface.co/stabilityai/stable-fast-3d/blob/main/LICENSE.md"
    )
    SUPPORTED_PLATFORMS: ClassVar[list[str]] = ["linux", "windows"]
    INSTALL_HINT: ClassVar[str | None] = (
        "Install from: https://github.com/Stability-AI/stable-fast-3d"
    )

    class OutputFormat(str, Enum):
        GLB = "glb"
        OBJ = "obj"

    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image to convert to 3D",
    )
    foreground_ratio: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Ratio of foreground size to image size after background removal.",
    )
    texture_resolution: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Resolution of the texture atlas.",
    )
    remesh: bool = Field(
        default=False,
        description="Whether to remesh the output for cleaner topology.",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GLB,
        description="Output format for the 3D model. GLB is recommended; OBJ files are not previewable in the canvas viewer.",
    )

    _model: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="stabilityai/stable-fast-3d",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.yaml",
                    "**/*.json",
                    "**/*.txt",
                ],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "SF3D (Stable Fast 3D)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "output_format"]

    def requires_gpu(self) -> bool:
        return True

    def _load_model(self):
        """Load or retrieve the SF3D model from cache."""
        import torch
        from nodetool.ml.core.model_manager import ModelManager

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("SF3D requires a CUDA-capable GPU")

        cache_key = "stabilityai/stable-fast-3d_SF3D"
        cached = ModelManager.get_model(cache_key)
        if cached is not None:
            self._model = cached
        else:
            from sf3d.system import SF3D

            _check_disk_space(self.ESTIMATED_DOWNLOAD_GB)
            if self.license_warning:
                log.info("License notice: %s", self.license_warning)

            self._model = SF3D.from_pretrained(
                "stabilityai/stable-fast-3d",
                config_name="config.yaml",
                weight_name="model.safetensors",
            )
            self._model.to(device)
            self._model.eval()
            ModelManager.set_model(self.id, cache_key, self._model)

    async def preload_model(self, context: ProcessingContext):
        try:
            from sf3d.system import SF3D  # noqa: F401
        except ImportError:
            return
        self._load_model()

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if self.image.is_empty():
            raise ValueError("Input image is required")

        try:
            from sf3d.utils import remove_background, resize_foreground
            import rembg
        except ImportError:
            raise ImportError(
                "SF3D requires the sf3d package. "
                "Install from: https://github.com/Stability-AI/stable-fast-3d"
            )

        # Load input image
        image_io = await context.asset_to_io(self.image)
        input_image = _open_pil_image(image_io, mode="RGBA")

        # Load model
        if self._model is None:
            self._load_model()

        from nodetool.ml.core.model_manager import ModelManager

        # Remove background and resize (cache rembg session to avoid re-loading U²-Net)
        rembg_cache_key = "rembg_u2net_session"
        rembg_session = ModelManager.get_model(rembg_cache_key)
        if rembg_session is None:
            rembg_session = rembg.new_session()
            ModelManager.set_model(self.id, rembg_cache_key, rembg_session)
        image = remove_background(input_image, rembg_session)
        image = resize_foreground(image, self.foreground_ratio)

        # Generate 3D mesh (SF3D always requires CUDA)
        device = "cuda"
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                mesh, _ = self._model.run_image(
                    [image],
                    bake_resolution=self.texture_resolution,
                    remesh="triangle" if self.remesh else "none",
                )

        run_gc("After SF3D inference", log_before_after=False)
        # Export mesh
        format_str = self.output_format.value
        if isinstance(mesh, list):
            mesh = mesh[0]
        model_bytes = _export_mesh(mesh, format=format_str, include_normals=True)

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"sf3d_{self.id}.{format_str}",
            format=format_str,
            metadata={"source_model": "stabilityai/stable-fast-3d"},
        )


class TripoSR(HuggingFacePipelineNode):
    """
    Generate 3D meshes from images using Stability AI TripoSR.
    3d, generation, image-to-3d, triposr, mesh, local, fast

    Use cases:
    - Fast 3D reconstruction from single images
    - Generate 3D models from concept art
    - Create 3D assets for games and visualization
    - Prototype 3D content quickly

    **Platforms:** Linux+CUDA, Windows+CUDA. macOS CPU experimental (slow, untested).

    **Requirements:** tsr package (TripoSR), rembg, torch with CUDA.
    First run downloads ~1GB model. Needs ~6GB VRAM.

    Model: https://huggingface.co/stabilityai/TripoSR
    License: MIT
    """

    # -- static metadata ---------------------------------------------------
    MIN_VRAM_GB: ClassVar[int] = 6
    ESTIMATED_DOWNLOAD_GB: ClassVar[float] = 1.0
    license_warning: ClassVar[str | None] = None  # MIT
    SUPPORTED_PLATFORMS: ClassVar[list[str]] = ["linux", "windows"]
    INSTALL_HINT: ClassVar[str | None] = (
        "Install from: https://github.com/VAST-AI-Research/TripoSR"
    )

    class OutputFormat(str, Enum):
        GLB = "glb"
        OBJ = "obj"

    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image to convert to 3D",
    )
    foreground_ratio: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Ratio of foreground size to image size after background removal.",
    )
    mc_resolution: int = Field(
        default=256,
        ge=64,
        le=512,
        description="Marching cubes resolution for mesh extraction. Higher = more detail.",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GLB,
        description="Output format for the 3D model. GLB is recommended; OBJ files are not previewable in the canvas viewer.",
    )

    _model: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="stabilityai/TripoSR",
                allow_patterns=["**/*.ckpt", "**/*.yaml", "**/*.json", "**/*.txt"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "TripoSR"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "output_format"]

    def requires_gpu(self) -> bool:
        return True

    def _load_model(self):
        """Load or retrieve the TripoSR model from cache."""
        import torch
        from nodetool.ml.core.model_manager import ModelManager

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = "stabilityai/TripoSR_TSR"
        cached = ModelManager.get_model(cache_key)
        if cached is not None:
            self._model = cached
        else:
            from tsr.system import TSR

            _check_disk_space(self.ESTIMATED_DOWNLOAD_GB)
            self._model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self._model.to(device)
            ModelManager.set_model(self.id, cache_key, self._model)

    async def preload_model(self, context: ProcessingContext):
        try:
            from tsr.system import TSR  # noqa: F401
        except ImportError:
            return
        self._load_model()

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        import os
        import numpy as np
        from PIL import Image

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if self.image.is_empty():
            raise ValueError("Input image is required")

        try:
            from tsr.utils import remove_background, resize_foreground
            import rembg
        except ImportError:
            raise ImportError(
                "TripoSR requires the tsr package. "
                "Install from: https://github.com/VAST-AI-Research/TripoSR"
            )

        # Load input image
        image_io = await context.asset_to_io(self.image)
        input_image = _open_pil_image(image_io, mode="RGBA")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        if self._model is None:
            self._load_model()

        from nodetool.ml.core.model_manager import ModelManager

        # Remove background and resize (cache rembg session to avoid re-loading U²-Net)
        rembg_cache_key = "rembg_u2net_session"
        rembg_session = ModelManager.get_model(rembg_cache_key)
        if rembg_session is None:
            rembg_session = rembg.new_session()
            ModelManager.set_model(self.id, rembg_cache_key, rembg_session)
        image = remove_background(input_image, rembg_session)
        image = resize_foreground(image, self.foreground_ratio)

        # Convert to proper format
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))

        # Generate 3D mesh
        with torch.no_grad():
            scene_codes = self._model([image], device=device)
            meshes = self._model.extract_mesh(
                scene_codes, True, resolution=self.mc_resolution
            )

        if not meshes:
            raise RuntimeError("No mesh generated by TripoSR")

        mesh = meshes[0]

        run_gc("After TripoSR inference", log_before_after=False)
        # Export mesh
        format_str = self.output_format.value
        model_bytes = _export_mesh(mesh, format=format_str)

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"triposr_{self.id}.{format_str}",
            format=format_str,
            metadata={"source_model": "stabilityai/TripoSR"},
        )


class Trellis2(HuggingFacePipelineNode):
    """
    Generate high-fidelity 3D models from images using Microsoft TRELLIS.2-4B.
    3d, generation, image-to-3d, trellis, mesh, local, high-quality, PBR, o-voxel

    Use cases:
    - Generate state-of-the-art 3D models from single images
    - Create production-ready 3D assets with PBR materials
    - Handle complex topology including open surfaces and non-manifold geometry
    - Generate assets with transparency/translucency support

    **Platforms:** Linux+CUDA only, 24 GB+ VRAM.

    **Note:** Requires trellis2, o_voxel, torch, and 24GB+ GPU memory.
    Currently only tested on Linux systems.
    First run will download the model (~10GB+).

    Model: https://huggingface.co/microsoft/TRELLIS.2-4B
    Paper: https://arxiv.org/abs/2512.14692
    """

    # -- static metadata ---------------------------------------------------
    MIN_VRAM_GB: ClassVar[int] = 24
    ESTIMATED_DOWNLOAD_GB: ClassVar[float] = 10.0
    license_warning: ClassVar[str | None] = (
        "Microsoft Research License — non-commercial use only. "
        "See https://huggingface.co/microsoft/TRELLIS.2-4B/blob/main/LICENSE"
    )
    SUPPORTED_PLATFORMS: ClassVar[list[str]] = ["linux"]
    INSTALL_HINT: ClassVar[str | None] = (
        "Install trellis2 and o_voxel. "
        "See https://github.com/microsoft/TRELLIS.2 for instructions."
    )

    class Resolution(str, Enum):
        RES_512 = "512"  # ~3 seconds on H100
        RES_1024 = "1024"  # ~17 seconds on H100
        RES_1536 = "1536"  # ~60 seconds on H100

    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image to convert to 3D",
    )
    resolution: Resolution = Field(
        default=Resolution.RES_1024,
        description="Voxel grid resolution. Higher = more detail but slower. 512 (~3s), 1024 (~17s), 1536 (~60s) on H100.",
    )
    decimation_target: int = Field(
        default=1000000,
        ge=10000,
        le=16777216,
        description="Target number of faces for mesh decimation. Lower = smaller file size.",
    )
    texture_size: int = Field(
        default=4096,
        ge=512,
        le=8192,
        description="Texture resolution for PBR materials.",
    )
    remesh: bool = Field(
        default=True,
        description="Whether to remesh the output for cleaner topology.",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed for reproducibility. -1 for random.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="microsoft/TRELLIS.2-4B",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "*.json",
                    "**/*.bin",
                ],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "TRELLIS.2-4B"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "resolution", "seed"]

    def requires_gpu(self) -> bool:
        return True

    def _load_pipeline(self):
        """Load or retrieve the Trellis2 pipeline from cache."""
        import torch
        from nodetool.ml.core.model_manager import ModelManager

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError(
                "TRELLIS.2 requires a CUDA-capable GPU with at least 24GB memory"
            )

        cache_key = "microsoft/TRELLIS.2-4B_Trellis2"
        cached = ModelManager.get_model(cache_key)
        if cached is not None:
            self._pipeline = cached
        else:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            _check_disk_space(self.ESTIMATED_DOWNLOAD_GB)
            if self.license_warning:
                log.info("License notice: %s", self.license_warning)

            self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS.2-4B"
            )
            self._pipeline.cuda()
            ModelManager.set_model(self.id, cache_key, self._pipeline)

    async def preload_model(self, context: ProcessingContext):
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401
            import o_voxel  # noqa: F401
        except ImportError:
            return
        self._load_pipeline()

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        import os

        # Enable OpenEXR support for environment maps (optional but recommended)
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if self.image.is_empty():
            raise ValueError("Input image is required")

        # Load input image
        image_io = await context.asset_to_io(self.image)
        input_image = _open_pil_image(image_io, mode="RGB")

        try:
            import o_voxel  # noqa: F401
        except ImportError:
            raise ImportError(
                "TRELLIS.2 requires the trellis2 and o_voxel packages. "
                "See https://github.com/microsoft/TRELLIS.2 for installation instructions."
            )

        if self._pipeline is None:
            self._load_pipeline()

        # Set seed using per-call generator (avoids leaking global RNG state)
        seed = _resolve_seed(self.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Generate 3D model
        # The pipeline returns a list of meshes
        meshes = self._pipeline.run(input_image, resolution=int(self.resolution.value))

        if not meshes:
            raise RuntimeError("No mesh generated by TRELLIS.2")

        mesh = meshes[0]

        # Simplify if needed (nvdiffrast limit is 16M faces)
        if hasattr(mesh, "simplify"):
            mesh.simplify(min(self.decimation_target, 16777216))

        run_gc("After Trellis2 inference", log_before_after=False)
        # Export to GLB using o_voxel
        try:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=self.decimation_target,
                texture_size=self.texture_size,
                remesh=self.remesh,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )

            # Export GLB to bytes
            buffer = io.BytesIO()
            glb.export(buffer, extension_webp=True)
            buffer.seek(0)
            model_bytes = buffer.read()
            format_str = "glb"
        except Exception as e:
            # Fallback: try basic trimesh export if o_voxel fails
            try:
                model_bytes = _export_mesh(mesh, format="glb")
                format_str = "glb"
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Failed to export mesh: {e}. Fallback also failed: {fallback_error}"
                )

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"trellis2_{self.id}.{format_str}",
            format=format_str,
            metadata={"seed": seed, "source_model": "microsoft/TRELLIS.2-4B"},
        )


class TripoSG(HuggingFacePipelineNode):
    """
    Generate high-fidelity 3D meshes from images using VAST TripoSG.
    3d, generation, image-to-3d, triposg, mesh, local, high-fidelity, rectified-flow

    Use cases:
    - Generate detailed 3D models from single images
    - Create 3D assets from photos, cartoons, or sketches
    - Produce meshes with sharp geometric features and fine surface details
    - Handle complex topology including thin structures

    **Platforms:** Linux+CUDA, Windows+CUDA (build required).

    **Requirements:** CUDA GPU with at least 8GB VRAM.
    First run downloads ~3GB model weights.

    Model: https://huggingface.co/VAST-AI/TripoSG
    License: MIT
    """

    # -- static metadata ---------------------------------------------------
    MIN_VRAM_GB: ClassVar[int] = 8
    ESTIMATED_DOWNLOAD_GB: ClassVar[float] = 3.0
    license_warning: ClassVar[str | None] = None  # MIT
    SUPPORTED_PLATFORMS: ClassVar[list[str]] = ["linux", "windows"]
    INSTALL_HINT: ClassVar[str | None] = (
        "Install with: pip install 'nodetool-huggingface[triposg]'"
    )

    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image to convert to 3D",
    )
    num_inference_steps: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Number of denoising steps. More steps = better quality but slower.",
    )
    guidance_scale: float = Field(
        default=7.0,
        ge=1.0,
        le=20.0,
        description="Classifier-free guidance scale. Higher = more adherence to input.",
    )
    octree_depth: int = Field(
        default=7,
        ge=5,
        le=9,
        description="Octree depth for mesh extraction. Higher = finer detail but slower. "
        "7 is a good balance (~30s). 9 gives maximum detail but requires diso for speed.",
    )
    max_faces: int = Field(
        default=-1,
        ge=-1,
        le=500000,
        description="Maximum number of faces in output mesh. -1 for no limit.",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed for reproducibility. -1 for random.",
    )

    _pipeline: Any = None
    _rmbg_net: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="VAST-AI/TripoSG",
                allow_patterns=[
                    "**/*.safetensors",
                    "**/*.json",
                    "**/*.txt",
                    "**/*.bin",
                ],
            ),
            HuggingFaceModel(
                repo_id="briaai/RMBG-1.4",
                allow_patterns=["**/*.pth", "**/*.json", "**/*.py"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "TripoSG"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "octree_depth", "max_faces", "seed"]

    def requires_gpu(self) -> bool:
        return True

    def _load_models(self):
        """Load or retrieve the TripoSG pipeline and RMBG model from cache."""
        import torch
        from nodetool.ml.core.model_manager import ModelManager
        from huggingface_hub import snapshot_download
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline
        from triposg.briarmbg import BriaRMBG

        device = "cuda"

        # Load RMBG model for background removal
        if self._rmbg_net is None:
            cache_key = "briaai/RMBG-1.4_BriaRMBG"
            cached = ModelManager.get_model(cache_key)
            if cached is not None:
                self._rmbg_net = cached
            else:
                _check_disk_space(0.5)  # RMBG is ~0.2 GB
                rmbg_path = snapshot_download(
                    repo_id="briaai/RMBG-1.4",
                    revision=_model_revision("briaai/RMBG-1.4"),
                )
                self._rmbg_net = BriaRMBG.from_pretrained(rmbg_path).to(device)
                self._rmbg_net.eval()
                ModelManager.set_model(self.id, cache_key, self._rmbg_net)

        # Load TripoSG pipeline
        if self._pipeline is None:
            cache_key = "VAST-AI/TripoSG_Pipeline"
            cached = ModelManager.get_model(cache_key)
            if cached is not None:
                self._pipeline = cached
            else:
                _check_disk_space(self.ESTIMATED_DOWNLOAD_GB)
                triposg_path = snapshot_download(
                    repo_id="VAST-AI/TripoSG",
                    revision=_model_revision("VAST-AI/TripoSG"),
                )
                self._pipeline = TripoSGPipeline.from_pretrained(triposg_path).to(
                    device, torch.float16
                )
                ModelManager.set_model(self.id, cache_key, self._pipeline)

    async def preload_model(self, context: ProcessingContext):
        import torch

        if not torch.cuda.is_available():
            return
        self._load_models()

    def _prepare_image(
        self,
        img_pil: "Image.Image",
        rmbg_net: Any,
        *,
        device: str = "cuda",
    ) -> "Image.Image":
        """Remove background and center-crop the foreground."""
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        import torchvision.transforms.functional as TF
        from PIL import Image
        from skimage.measure import label
        from skimage.morphology import remove_small_objects

        bg_color = np.array([1.0, 1.0, 1.0])
        padding_ratio = 0.1
        img = np.array(img_pil)

        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            rgb_image = img[:, :, :3]
        else:
            rgb_image = img
            alpha = None

        height, width = rgb_image.shape[:2]
        scale = 2000 / max(height, width)
        if scale < 1:
            new_size = (int(width * scale), int(height * scale))
            rgb_image = cv2.resize(rgb_image, new_size, interpolation=cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(alpha, new_size, interpolation=cv2.INTER_AREA)

        rgb_gpu = (
            torch.from_numpy(rgb_image).to(device).float().permute(2, 0, 1) / 255.0
        )

        if alpha is not None:
            _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            alpha_gpu = torch.from_numpy(alpha).to(device).float().unsqueeze(0) / 255.0
        else:
            resize_transform = transforms.Resize((1024, 1024), antialias=True)
            rgb_resized = resize_transform(rgb_gpu)
            max_val = rgb_resized.flatten().max()
            if max_val < 1e-3:
                raise ValueError("Invalid image: pure black")

            rmbg_input = TF.normalize(rgb_resized, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            result = rmbg_net(rmbg_input.unsqueeze(0))
            alpha_gpu = result[0][0].squeeze(0)
            alpha_gpu = transforms.Resize(
                (rgb_gpu.shape[1], rgb_gpu.shape[2]), antialias=True
            )(alpha_gpu)
            ma, mi = alpha_gpu.max(), alpha_gpu.min()
            alpha_gpu = (alpha_gpu - mi) / (ma - mi)

            alpha_np = (alpha_gpu * 255).to(torch.uint8).squeeze().cpu().numpy()
            _, alpha_np = cv2.threshold(
                alpha_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            labeled = label(alpha_np)
            cleaned = (remove_small_objects(labeled, min_size=200) > 0).astype(np.uint8)
            alpha_gpu = torch.from_numpy(cleaned).to(device).float().unsqueeze(0)

        _, binary = cv2.threshold(
            (alpha_gpu.squeeze().cpu().numpy() * 255).astype(np.uint8),
            1,
            255,
            cv2.THRESH_BINARY,
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("No foreground found in image")
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        bg = (
            torch.from_numpy(bg_color)
            .float()
            .to(device)
            .repeat(alpha_gpu.shape[1], alpha_gpu.shape[2], 1)
            .permute(2, 0, 1)
        )
        rgb_gpu = rgb_gpu * alpha_gpu + bg * (1 - alpha_gpu)

        padding_size = [0] * 6
        if w > h:
            padding_size[0] = int(w * padding_ratio)
            padding_size[2] = int(padding_size[0] + (w - h) / 2)
        else:
            padding_size[2] = int(h * padding_ratio)
            padding_size[0] = int(padding_size[2] + (h - w) / 2)
        padding_size[1] = padding_size[0]
        padding_size[3] = padding_size[2]

        padded = F.pad(
            rgb_gpu[:, y : y + h, x : x + w],
            pad=tuple(padding_size),
            mode="constant",
            value=bg_color[0],
        )
        result_np = padded.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((result_np * 255).astype(np.uint8))

    def _simplify_mesh(self, mesh: "trimesh.Trimesh", n_faces: int):
        """Reduce mesh face count using quadric edge collapse."""
        import pymeshlab
        import trimesh

        if mesh.faces.shape[0] <= n_faces:
            return mesh
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
        result = ms.current_mesh()
        return trimesh.Trimesh(
            vertices=result.vertex_matrix(), faces=result.face_matrix()
        )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        import os
        import numpy as np
        import trimesh

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if self.image.is_empty():
            raise ValueError("Input image is required")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError(
                "TripoSG requires a CUDA-capable GPU with at least 8GB VRAM"
            )

        # Load models if not already loaded
        self._load_models()

        # Load and prepare input image
        image_io = await context.asset_to_io(self.image)
        input_image = _open_pil_image(image_io, mode="RGBA")
        prepared_image = self._prepare_image(input_image, self._rmbg_net, device=device)

        # Set seed
        seed = _resolve_seed(self.seed)
        generator = torch.Generator(device=device).manual_seed(seed)

        # Use flash decoder (diso) if available, fall back to marching cubes
        try:
            import diso  # noqa: F401

            use_flash_decoder = True
        except ImportError:
            use_flash_decoder = False

        # Run inference — only pass flash_octree_depth when flash decoder is active (#12)
        pipeline_kwargs: dict[str, Any] = {
            "image": prepared_image,
            "generator": generator,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "dense_octree_depth": self.octree_depth - 1,
            "hierarchical_octree_depth": self.octree_depth,
            "use_flash_decoder": use_flash_decoder,
        }
        if use_flash_decoder:
            pipeline_kwargs["flash_octree_depth"] = self.octree_depth

        outputs = self._pipeline(**pipeline_kwargs).samples[0]

        mesh = trimesh.Trimesh(
            outputs[0].astype(np.float32),
            np.ascontiguousarray(outputs[1]),
        )

        run_gc("After TripoSG inference", log_before_after=False)

        # Optionally simplify mesh
        if self.max_faces > 0:
            mesh = self._simplify_mesh(mesh, self.max_faces)

        # Export to GLB
        model_bytes = _export_mesh(mesh, format="glb")

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"triposg_{self.id}.glb",
            format="glb",
            metadata={"seed": seed, "source_model": "VAST-AI/TripoSG"},
        )
