"""
HuggingFace text-to-3D generation nodes.

Provides local inference for text-to-3D model generation:
- Shap-E: Text-to-3D using OpenAI's diffusers pipeline

These nodes run locally and do not require API keys.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from nodetool.metadata.types import HuggingFaceModel, Model3DRef
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.memory_utils import run_gc

from nodetool.nodes.huggingface._3d_common import (
    _resolve_device,
    _resolve_seed,
    _export_mesh,
)


class ShapETextTo3D(HuggingFacePipelineNode):
    """
    Generate 3D models from text descriptions using OpenAI Shap-E.
    3d, generation, text-to-3d, shap-e, mesh, local

    Use cases:
    - Generate 3D models from text descriptions locally
    - Create 3D assets without API costs
    - Prototype 3D content quickly
    - Generate simple 3D objects for games/visualization

    **Platforms:** all (CPU/MPS/CUDA).

    **Note:** Requires diffusers and torch. First run will download the model (~2.5GB).
    """

    # -- static metadata ---------------------------------------------------
    MIN_VRAM_GB: ClassVar[int] = 4
    ESTIMATED_DOWNLOAD_GB: ClassVar[float] = 2.5
    license_warning: ClassVar[str | None] = None  # MIT
    SUPPORTED_PLATFORMS: ClassVar[list[str]] = ["linux", "macos", "windows"]
    INSTALL_HINT: ClassVar[str | None] = None  # uses diffusers, already in core

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

    # Note: self._pipeline is required by the HuggingFacePipelineNode base class
    # (used by move_to_device, run_pipeline_in_thread). Refactoring this
    # out requires upstream changes to nodetool-core's base class.
    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="openai/shap-e",
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
        return "Shap-E Text-to-3D"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "seed"]

    def requires_gpu(self) -> bool:
        return False

    async def preload_model(self, context: ProcessingContext):
        import torch
        from diffusers import ShapEPipeline

        device = _resolve_device()
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipeline = await self.load_model(
            context=context,
            model_class=ShapEPipeline,
            model_id="openai/shap-e",
            torch_dtype=torch_dtype,
        )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch

        if not self.prompt:
            raise ValueError("Prompt is required")

        assert self._pipeline is not None, "Pipeline not initialized"
        device = str(self._pipeline.device)

        # Set seed – generator device must be "cpu" for MPS pipelines
        gen_device = "cpu" if device.startswith("mps") else device
        seed = _resolve_seed(self.seed)
        generator = torch.Generator(device=gen_device).manual_seed(seed)

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

        run_gc("After ShapE Text-to-3D inference", log_before_after=False)
        # Export to GLB
        import trimesh

        tri_mesh = trimesh.Trimesh(
            vertices=mesh.verts.cpu().numpy(),
            faces=mesh.faces.cpu().numpy(),
        )
        model_bytes = _export_mesh(tri_mesh, format="glb")

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"shap_e_{self.id}.glb",
            format="glb",
            metadata={"seed": seed, "source_model": "openai/shap-e"},
        )
