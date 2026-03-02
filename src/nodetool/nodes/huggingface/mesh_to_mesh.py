"""
HuggingFace ML-based 3D mesh processing nodes.

Provides local inference for mesh-to-mesh operations:
- MeshAnything V2: AI retopology (dense mesh -> clean low-poly mesh)

These nodes run locally and require GPU.
"""

from __future__ import annotations

import io
from typing import Any, TYPE_CHECKING

from pydantic import Field

from nodetool.metadata.types import HuggingFaceModel, Model3DRef
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.memory_utils import run_gc

if TYPE_CHECKING:
    import numpy as np


class MeshAnythingV2(HuggingFacePipelineNode):
    """
    AI retopology: converts dense meshes into clean, low-poly artist-style meshes.
    3d, retopology, mesh, simplify, artist, low-poly, AI, local

    Use cases:
    - Convert dense scanned meshes to clean artist-style topology
    - Create game-ready low-poly assets from high-poly models
    - Retopologize AI-generated 3D models for production use
    - Generate clean meshes with up to 1600 faces

    **Requirements:** MeshAnythingV2 package (git clone + pip install).
    First run downloads ~1.5GB model. Needs GPU with ~6GB VRAM.
    Input meshes should have a +Y up vector for best results.

    Code: https://github.com/buaacyw/MeshAnythingV2
    """

    model_input: Model3DRef = Field(
        default=Model3DRef(),
        description="Dense 3D model to retopologize",
    )
    marching_cubes: bool = Field(
        default=False,
        description="Preprocess with Marching Cubes. Enable if input is NOT from Marching Cubes.",
    )
    mc_resolution: int = Field(
        default=7,
        ge=5,
        le=9,
        description="Marching Cubes octree depth (2^N resolution). 7=128, 8=256. Higher = more detail but slower.",
    )
    sampling: bool = Field(
        default=False,
        description="Use random sampling during generation. Try with different seeds if results are poor.",
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Random seed for reproducibility.",
    )

    _model: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="Yiwen-ntu/meshanythingv2",
                allow_patterns=["*.safetensors", "*.json"],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "MeshAnything V2"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model_input", "marching_cubes", "seed"]

    def requires_gpu(self) -> bool:
        return True

    def _mesh_to_point_cloud(self, mesh, sample_num: int = 8192) -> "np.ndarray":
        """Convert a trimesh to a normalized point cloud with normals."""
        import numpy as np

        if self.marching_cubes:
            import mesh2sdf.core
            import skimage.measure

            vertices = mesh.vertices.copy()
            bbmin, bbmax = vertices.min(0), vertices.max(0)
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * 0.95 / (bbmax - bbmin).max()
            scaled_vertices = (vertices - center) * scale

            size = 2 ** self.mc_resolution
            level = 2 / size
            sdf = mesh2sdf.core.compute(scaled_vertices, mesh.faces, size=size)
            mc_verts, mc_faces, mc_normals, _ = skimage.measure.marching_cubes(
                np.abs(sdf), level
            )
            mc_verts = mc_verts / size * 2 - 1
            mc_verts = mc_verts / scale + center
            import trimesh as _trimesh

            mesh = _trimesh.Trimesh(mc_verts, mc_faces, normals=mc_normals)

        points, face_idx = mesh.sample(sample_num, return_index=True)
        normals = mesh.face_normals[face_idx]
        return np.concatenate([points, normals], axis=-1, dtype=np.float16)

    async def process(self, context: ProcessingContext) -> Model3DRef:
        import torch
        import numpy as np

        if self.model_input.is_empty():
            raise ValueError("Input model is required")

        try:
            from MeshAnything.models.meshanything_v2 import (
                MeshAnythingV2 as MeshAnythingV2Model,
            )
        except ImportError:
            raise ImportError(
                "MeshAnythingV2 requires the MeshAnything package. "
                "Install from: https://github.com/buaacyw/MeshAnythingV2 "
                "(git clone, then pip install -r requirements.txt)"
            )

        import trimesh
        from accelerate.utils import set_seed

        set_seed(self.seed)

        # Load input mesh
        if self.model_input.data:
            model_data = self.model_input.data
        else:
            model_data = await context.asset_to_bytes(self.model_input)

        input_mesh = trimesh.load(
            io.BytesIO(model_data), file_type=self.model_input.format or "glb"
        )
        if isinstance(input_mesh, trimesh.Scene):
            input_mesh = input_mesh.dump(concatenate=True)

        # Load model
        if self._model is None:
            from nodetool.ml.core.model_manager import ModelManager

            cache_key = "Yiwen-ntu/meshanythingv2_MeshAnythingV2"
            cached = ModelManager.get_model(cache_key)
            if cached is not None:
                self._model = cached
            else:
                self._model = MeshAnythingV2Model.from_pretrained(
                    "Yiwen-ntu/meshanythingv2"
                )
                self._model.cuda()
                self._model.eval()
                ModelManager.set_model(self.id, cache_key, self._model)

        # Preprocess: mesh -> point cloud with normals
        pc_normal = self._mesh_to_point_cloud(input_mesh)

        # Normalize coordinates
        pc_coor = pc_normal[:, :3]
        normals = pc_normal[:, 3:]
        bounds = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
        pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
        pc_coor = pc_coor / np.abs(pc_coor).max() * 0.99
        normalized = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)

        input_tensor = torch.tensor(normalized, dtype=torch.float16, device="cuda")[
            None
        ]

        # Generate artist-style mesh
        with torch.cuda.amp.autocast():
            outputs = self._model(input_tensor, self.sampling)

        run_gc("After MeshAnythingV2 inference", log_before_after=False)

        recon_mesh = outputs[0]
        valid_mask = torch.all(~torch.isnan(recon_mesh.reshape((-1, 9))), dim=1)
        recon_mesh = recon_mesh[valid_mask]

        vertices = recon_mesh.reshape(-1, 3).cpu().numpy()
        faces = np.arange(len(vertices)).reshape(-1, 3)

        artist_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=True
        )
        artist_mesh.merge_vertices()
        artist_mesh.update_faces(artist_mesh.nondegenerate_faces())
        artist_mesh.update_faces(artist_mesh.unique_faces())
        artist_mesh.remove_unreferenced_vertices()
        artist_mesh.fix_normals()

        # Export
        buffer = io.BytesIO()
        artist_mesh.export(buffer, file_type="glb")
        buffer.seek(0)
        model_bytes = buffer.read()

        return await context.model3d_from_bytes(
            model_bytes,
            name=f"meshanything_{self.id}.glb",
            format="glb",
        )
