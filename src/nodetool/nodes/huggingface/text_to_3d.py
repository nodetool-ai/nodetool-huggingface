"""
HuggingFace local 3D generation nodes.

Provides local inference for 3D model generation using models from HuggingFace:
- Shap-E: Text-to-3D and Image-to-3D using diffusers

These nodes run locally and do not require API keys.
"""

from __future__ import annotations

import io
from enum import Enum
from typing import Any, ClassVar, TYPE_CHECKING

from pydantic import Field

from nodetool.metadata.types import HuggingFaceModel, ImageRef, Model3DRef
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    import torch


class ShapETextTo3D(HuggingFacePipelineNode):
    """
    Generate 3D models from text descriptions using OpenAI Shap-E.
    3d, generation, text-to-3d, shap-e, mesh, local

    Use cases:
    - Generate 3D models from text descriptions locally
    - Create 3D assets without API costs
    - Prototype 3D content quickly
    - Generate simple 3D objects for games/visualization

    **Note:** Requires diffusers and torch. First run will download the model (~2.5GB).
    """

    prompt: str = Field(
        default="a shark",
        description="Text description of the 3D model to generate",
    )
    guidance_scale: float = Field(
        default=15.0,
        ge=1.0,
        le=30.0,
        description="How strongly to follow the prompt. Higher = more prompt adherence.",
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
                repo_id="openai/shap-e",
                # Include safetensors + bin for shap_e_renderer (no safetensors available there)
                allow_patterns=["**/*.safetensors", "shap_e_renderer/*.bin", "**/*.json", "**/*.txt"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Shap-E Text-to-3D"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "seed"]

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        from diffusers import ShapEPipeline
        
        if not self.prompt:
            raise ValueError("Prompt is required")

        # Load pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self._pipeline is None:
            self._pipeline = ShapEPipeline.from_pretrained(
                "openai/shap-e",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                
            ).to(device)

        # Set seed
        generator = None
        if self.seed >= 0:
            generator = torch.Generator(device=device).manual_seed(self.seed)

        # Generate
        images = self._pipeline(
            self.prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            frame_size=self.frame_size,
            generator=generator,
            output_type="mesh",
        ).images

        if not images:
            raise RuntimeError("No 3D model generated")

        mesh = images[0]

        # Export to GLB
        import trimesh
        tri_mesh = trimesh.Trimesh(
            vertices=mesh.verts.cpu().numpy(),
            faces=mesh.faces.cpu().numpy(),
        )
        buffer = io.BytesIO()
        tri_mesh.export(buffer, file_type="glb")
        buffer.seek(0)
        model_bytes = buffer.read()

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"shap_e_{self.id}.glb",
            format="glb",
        )


class ShapEImageTo3D(HuggingFacePipelineNode):
    """
    Generate 3D models from images using OpenAI Shap-E.
    3d, generation, image-to-3d, shap-e, mesh, local, reconstruction

    Use cases:
    - Convert images to 3D models locally
    - Create 3D assets from product photos
    - Generate 3D content without API costs
    - Reconstruct 3D objects from single images

    **Note:** Requires diffusers and torch. First run will download the model (~2.5GB).
    """

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
                allow_patterns=["**/*.safetensors", "shap_e_renderer/*.bin", "**/*.json", "**/*.txt"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Shap-E Image-to-3D"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "seed"]

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        from PIL import Image
        from diffusers import ShapEImg2ImgPipeline
        
        if self.image.is_empty():
            raise ValueError("Input image is required")

        # Load input image
        image_io = await context.asset_to_io(self.image)
        input_image = Image.open(image_io).convert("RGB")

        # Load pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self._pipeline is None:
            self._pipeline = ShapEImg2ImgPipeline.from_pretrained(
                "openai/shap-e-img2img",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                
            ).to(device)

        # Set seed
        generator = None
        if self.seed >= 0:
            generator = torch.Generator(device=device).manual_seed(self.seed)

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

        # Export to GLB
        import trimesh
        tri_mesh = trimesh.Trimesh(
            vertices=mesh.verts.cpu().numpy(),
            faces=mesh.faces.cpu().numpy(),
        )
        buffer = io.BytesIO()
        tri_mesh.export(buffer, file_type="glb")
        buffer.seek(0)
        model_bytes = buffer.read()

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"shap_e_img_{self.id}.glb",
            format="glb",
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

    **Requirements:** hy3dgen>=2.0.2 package, torch with CUDA.
    First run downloads ~5GB model (shape-only, not full 75GB repo).
    Standard model needs ~6GB VRAM, mini needs ~5GB. Use low_vram_mode on constrained GPUs.

    Models: https://huggingface.co/tencent/Hunyuan3D-2
    """

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
        description="Output format for the 3D model.",
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
        Pre-download only the shape-generation files to avoid downloading
        the entire 75GB repo (which includes paint/texture models).
        Returns the repo_id for the model.
        """
        from huggingface_hub import snapshot_download
        
        config = self.VARIANT_CONFIG[variant]
        repo_id = config["repo_id"]
        subfolder = config["subfolder"]
        
        # Only download the specific subfolder needed for shape generation
        # This avoids downloading paint, delight, turbo variants etc.
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                "config.json",  # Root config
                f"{subfolder}/*",  # DiT model (includes bundled VAE)
            ],
        )
        return repo_id

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        from PIL import Image

        if self.image.is_empty():
            raise ValueError("Input image is required")

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError:
            raise ImportError(
                "Hunyuan3D requires the hy3dgen package (>=2.0.2). "
                "Install with: pip install hy3dgen>=2.0.2"
            )

        # Load input image
        image_io = await context.asset_to_io(self.image)
        input_image = Image.open(image_io).convert("RGB")

        # Get variant config
        variant = self.model_variant.value
        config = self.VARIANT_CONFIG[variant]
        
        # Load or reload pipeline if variant changed
        if self._pipeline is None or self._loaded_variant != variant:
            # Pre-download only shape files to avoid 75GB full repo download
            self._ensure_model_downloaded(variant)
            
            # Now load the pipeline - hy3dgen will find the cached files
            self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                config["repo_id"],
                subfolder=config["subfolder"],
                use_safetensors=True,
                variant="fp16",
            )
            
            # Enable CPU offloading if requested
            if self.low_vram_mode:
                self._pipeline.enable_model_cpu_offload()
                
            self._loaded_variant = variant

        # Set seed
        if self.seed >= 0:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        # Generate 3D mesh
        mesh = self._pipeline(
            image=input_image,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            octree_resolution=self.octree_resolution,
        )[0]

        # Export mesh to bytes via trimesh
        format_str = self.output_format.value
        buffer = io.BytesIO()

        if hasattr(mesh, "export"):
            # hy3dgen returns trimesh objects directly
            mesh.export(buffer, file_type=format_str)
        else:
            import trimesh
            tri_mesh = trimesh.Trimesh(
                vertices=mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, "cpu") else mesh.vertices,
                faces=mesh.faces.cpu().numpy() if hasattr(mesh.faces, "cpu") else mesh.faces,
            )
            tri_mesh.export(buffer, file_type=format_str)

        buffer.seek(0)
        model_bytes = buffer.read()

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"hunyuan3d_{self.id}.{format_str}",
            format=format_str,
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

    **Requirements:** sf3d package, rembg, torch with CUDA.
    First run downloads ~1GB model. Needs ~6GB VRAM.
    Generates textured meshes with UV maps, normal maps, and PBR materials.

    Model: https://huggingface.co/stabilityai/stable-fast-3d
    License: Free for <$1M revenue, enterprise license required above.
    """

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
        description="Output format for the 3D model.",
    )

    _model: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="stabilityai/stable-fast-3d",
                allow_patterns=["**/*.safetensors", "**/*.yaml", "**/*.json", "**/*.txt"],
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

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        from PIL import Image

        if self.image.is_empty():
            raise ValueError("Input image is required")

        try:
            from sf3d.system import SF3D
            from sf3d.utils import remove_background, resize_foreground
            import rembg
        except ImportError:
            raise ImportError(
                "SF3D requires the sf3d package. "
                "Install from: https://github.com/Stability-AI/stable-fast-3d"
            )

        # Load input image
        image_io = await context.asset_to_io(self.image)
        input_image = Image.open(image_io).convert("RGBA")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("SF3D requires a CUDA-capable GPU")

        # Load model
        if self._model is None:
            self._model = SF3D.from_pretrained(
                "stabilityai/stable-fast-3d",
                config_name="config.yaml",
                weight_name="model.safetensors",
            )
            self._model.to(device)
            self._model.eval()

        # Remove background and resize
        rembg_session = rembg.new_session()
        image = remove_background(input_image, rembg_session)
        image = resize_foreground(image, self.foreground_ratio)

        # Generate 3D mesh
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                mesh, _ = self._model.run_image(
                    [image],
                    bake_resolution=self.texture_resolution,
                    remesh="triangle" if self.remesh else "none",
                )

        # Export mesh
        format_str = self.output_format.value
        buffer = io.BytesIO()
        
        if isinstance(mesh, list):
            mesh = mesh[0]
        mesh.export(buffer, file_type=format_str, include_normals=True)
        
        buffer.seek(0)
        model_bytes = buffer.read()

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"sf3d_{self.id}.{format_str}",
            format=format_str,
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

    **Requirements:** tsr package (TripoSR), rembg, torch with CUDA.
    First run downloads ~1GB model. Needs ~6GB VRAM.

    Model: https://huggingface.co/stabilityai/TripoSR
    License: MIT
    """

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
        description="Output format for the 3D model.",
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

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        import numpy as np
        from PIL import Image

        if self.image.is_empty():
            raise ValueError("Input image is required")

        try:
            from tsr.system import TSR
            from tsr.utils import remove_background, resize_foreground
            import rembg
        except ImportError:
            raise ImportError(
                "TripoSR requires the tsr package. "
                "Install from: https://github.com/VAST-AI-Research/TripoSR"
            )

        # Load input image
        image_io = await context.asset_to_io(self.image)
        input_image = Image.open(image_io).convert("RGBA")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        if self._model is None:
            self._model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self._model.to(device)

        # Remove background and resize
        rembg_session = rembg.new_session()
        image = remove_background(input_image, rembg_session)
        image = resize_foreground(image, self.foreground_ratio)
        
        # Convert to proper format
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))

        # Generate 3D mesh
        with torch.no_grad():
            scene_codes = self._model([image], device=device)
            meshes = self._model.extract_mesh(scene_codes, True, resolution=self.mc_resolution)

        if not meshes:
            raise RuntimeError("No mesh generated by TripoSR")

        mesh = meshes[0]

        # Export mesh
        format_str = self.output_format.value
        buffer = io.BytesIO()
        mesh.export(buffer, file_type=format_str)
        buffer.seek(0)
        model_bytes = buffer.read()

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"triposr_{self.id}.{format_str}",
            format=format_str,
        )

class Trellis2\(HuggingFacePipelineNode\):
    """
    Generate high-fidelity 3D models from images using Microsoft TRELLIS.2-4B.
    3d, generation, image-to-3d, trellis, mesh, local, high-quality, PBR, o-voxel

    Use cases:
    - Generate state-of-the-art 3D models from single images
    - Create production-ready 3D assets with PBR materials
    - Handle complex topology including open surfaces and non-manifold geometry
    - Generate assets with transparency/translucency support

    **Note:** Requires trellis2, o_voxel, torch, and 24GB+ GPU memory.
    Currently only tested on Linux systems.
    First run will download the model (~10GB+).
    
    Model: https://huggingface.co/microsoft/TRELLIS.2-4B
    Paper: https://arxiv.org/abs/2512.14692
    """

    class Resolution(str, Enum):
        RES_512 = "512"    # ~3 seconds on H100
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
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json", "**/*.bin"],
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

    def _ensure_model_downloaded(self, variant: str) -> str:
        """
        Pre-download only the shape-generation files to avoid downloading
        the entire 75GB repo (which includes paint/texture models).
        Returns the repo_id for the model.
        """
        from huggingface_hub import snapshot_download
        
        config = self.VARIANT_CONFIG[variant]
        repo_id = config["repo_id"]
        subfolder = config["subfolder"]
        
        # Only download the specific subfolder needed for shape generation
        # This avoids downloading paint, delight, turbo variants etc.
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                "config.json",  # Root config
                f"{subfolder}/*",  # DiT model (includes bundled VAE)
            ],
        )
        return repo_id

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        from PIL import Image
        import os

        # Enable OpenEXR support for environment maps (optional but recommended)
        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if self.image.is_empty():
            raise ValueError("Input image is required")

        # Load input image
        image_io = await context.asset_to_io(self.image)
        input_image = Image.open(image_io).convert("RGB")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("TRELLIS.2 requires a CUDA-capable GPU with at least 24GB memory")

        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            import o_voxel
        except ImportError:
            raise ImportError(
                "TRELLIS.2 requires the trellis2 and o_voxel packages. "
                "See https://github.com/microsoft/TRELLIS.2 for installation instructions."
            )

        if self._pipeline is None:
            self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS.2-4B"
            )
            self._pipeline.cuda()

        # Set seed
        if self.seed >= 0:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

        # Generate 3D model
        # The pipeline returns a list of meshes
        meshes = self._pipeline.run(input_image, resolution=int(self.resolution.value))
        
        if not meshes:
            raise RuntimeError("No mesh generated by TRELLIS.2")
        
        mesh = meshes[0]
        
        # Simplify if needed (nvdiffrast limit is 16M faces)
        if hasattr(mesh, "simplify"):
            mesh.simplify(min(self.decimation_target, 16777216))

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
                verbose=False
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
                import trimesh
                tri_mesh = trimesh.Trimesh(
                    vertices=mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, "cpu") else mesh.vertices,
                    faces=mesh.faces.cpu().numpy() if hasattr(mesh.faces, "cpu") else mesh.faces,
                )
                buffer = io.BytesIO()
                tri_mesh.export(buffer, file_type="glb")
                buffer.seek(0)
                model_bytes = buffer.read()
                format_str = "glb"
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to export mesh: {e}. Fallback also failed: {fallback_error}")

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"trellis2_{self.id}.{format_str}",
            format=format_str,
        )
