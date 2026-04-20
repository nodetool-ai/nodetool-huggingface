"""
Shared helpers for local 3D generation nodes.

Provides device resolution, seeding, mesh export, image validation,
disk-space pre-flight, and model-revision pinning shared by
``text_to_3d`` and ``image_to_3d`` modules.
"""

from __future__ import annotations

import io
import logging
import shutil
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model revision pinning (D9 / #19)
# ---------------------------------------------------------------------------
# Map from HuggingFace repo_id → known-good commit SHA.
# Pass these to ``from_pretrained(..., revision=...)`` and
# ``snapshot_download(..., revision=...)`` calls so upstream
# weight reorganizations don't silently break users.
#
# Values of ``None`` mean "use latest" — fill in verified SHAs
# and bump intentionally.  Refresh quarterly.
# ---------------------------------------------------------------------------
MODEL_REVISIONS: dict[str, str | None] = {
    "openai/shap-e": None,
    "openai/shap-e-img2img": None,
    "tencent/Hunyuan3D-2": None,
    "tencent/Hunyuan3D-2mini": None,
    "stabilityai/stable-fast-3d": None,
    "stabilityai/TripoSR": None,
    "microsoft/TRELLIS.2-4B": None,
    "VAST-AI/TripoSG": None,
    "briaai/RMBG-1.4": None,
}


def _model_revision(repo_id: str) -> str | None:
    """Return the pinned revision for *repo_id*, or ``None`` (latest)."""
    return MODEL_REVISIONS.get(repo_id)


# ---------------------------------------------------------------------------
# Disk-space pre-flight (#21)
# ---------------------------------------------------------------------------
_DISK_HEADROOM_GB = 2  # extra headroom beyond estimated model size


def _check_disk_space(estimated_gb: float, cache_dir: str | None = None) -> None:
    """Raise if the HF cache volume has less free space than *estimated_gb*.

    Parameters
    ----------
    estimated_gb:
        Approximate download size in GiB.
    cache_dir:
        Override for the cache directory to check.  When ``None``,
        resolves from ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE`` env vars,
        falling back to ``~/.cache/huggingface/hub``.
    """
    import os

    if cache_dir is None:
        cache_dir = os.environ.get(
            "HUGGINGFACE_HUB_CACHE",
            os.environ.get(
                "HF_HOME",
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
            ),
        )
    os.makedirs(cache_dir, exist_ok=True)
    usage = shutil.disk_usage(cache_dir)
    free_gb = usage.free / (1 << 30)
    needed = estimated_gb + _DISK_HEADROOM_GB
    if free_gb < needed:
        raise OSError(
            f"Not enough disk space to download model weights. "
            f"Need ~{needed:.1f} GB free, but only {free_gb:.1f} GB available "
            f"in {cache_dir}. Free up space or set HF_HOME / "
            f"HUGGINGFACE_HUB_CACHE to a volume with more room."
        )


def _resolve_device() -> str:
    """Return the best available torch device string (cuda > mps > cpu)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _warn_vram(min_vram_gb: int, node_name: str) -> None:
    """Log a warning if detected GPU VRAM is below *min_vram_gb* (D10).

    Per D10, VRAM guidance is a soft warning — never a hard block.
    Only checks the default CUDA device; silently skips on non-CUDA setups
    or when torch is not installed.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return

    if not torch.cuda.is_available():
        return
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_mem
        total_gb = total_bytes / (1 << 30)
        if total_gb < min_vram_gb:
            log.warning(
                "%s recommends at least %d GB VRAM but the current GPU "
                "reports %.1f GB. The model may run out of memory or fall "
                "back to CPU offloading.",
                node_name,
                min_vram_gb,
                total_gb,
            )
    except Exception:
        pass  # non-critical — skip silently


def _warn_platform(supported_platforms: list[str], node_name: str) -> None:
    """Log a warning if the current OS is not in *supported_platforms*.

    Uses ``sys.platform`` to detect the current operating system and maps
    it to the canonical platform names used by 3D nodes
    (``linux``, ``macos``, ``windows``).
    """
    import sys

    plat_map = {"linux": "linux", "darwin": "macos", "win32": "windows"}
    current = plat_map.get(sys.platform)
    if current and current not in supported_platforms:
        log.warning(
            "%s is not supported on %s. Supported platforms: %s. "
            "The node may fail or produce unexpected results.",
            node_name,
            current,
            ", ".join(supported_platforms),
        )


def _resolve_seed(seed: int) -> int:
    """Return *seed* unchanged when >= 0, otherwise a random 32-bit integer."""
    if seed >= 0:
        return seed
    import torch

    return torch.randint(0, 2**32, (1,)).item()


def _open_pil_image(image_bytes_io: Any, mode: str = "RGB") -> "Image.Image":
    """Open a PIL image from an IO buffer with friendly error messages.

    Wraps ``PIL.Image.open`` to convert common failure modes
    (corrupt file, unsupported format, truncated download, etc.)
    into a clear ``ValueError`` that surfaces in the node UI.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(image_bytes_io)
        img.load()  # force full decode to catch truncated files
        return img.convert(mode)
    except UnidentifiedImageError:
        raise ValueError(
            "Invalid input image: the file could not be identified as an image. "
            "Please provide a valid JPEG, PNG, or WebP file."
        )
    except OSError as exc:
        raise ValueError(f"Invalid input image: {exc}") from exc


def _export_mesh(
    mesh: Any,
    format: str = "glb",
    include_normals: bool = False,
) -> bytes:
    """Export a mesh (trimesh or raw vertices/faces object) to bytes.

    Handles the common export pattern shared across 3D generation nodes:
    - If *mesh* already has an ``.export()`` method (e.g. a trimesh object),
      call it directly.
    - Otherwise build a ``trimesh.Trimesh`` from ``.vertices`` / ``.faces``,
      converting GPU tensors to numpy when necessary.

    Parameters
    ----------
    mesh:
        A trimesh ``Trimesh``, or any object with ``.vertices`` and ``.faces``
        attributes.
    format:
        Target file type (``"glb"``, ``"obj"``, …).
    include_normals:
        If ``True``, pass ``include_normals=True`` to the trimesh exporter
        (used by SF3D).
    """
    import trimesh as _trimesh

    buffer = io.BytesIO()

    if hasattr(mesh, "export"):
        kwargs: dict[str, Any] = {"file_type": format}
        if include_normals:
            kwargs["include_normals"] = True
        mesh.export(buffer, **kwargs)
    else:
        verts = mesh.vertices
        faces = mesh.faces
        if hasattr(verts, "cpu"):
            verts = verts.cpu().numpy()
        if hasattr(faces, "cpu"):
            faces = faces.cpu().numpy()
        tri = _trimesh.Trimesh(vertices=verts, faces=faces)
        tri.export(buffer, file_type=format)

    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# Canonical orientation and units (D12)
# ---------------------------------------------------------------------------
# All local 3D generators use +Y up, matching glTF 2.0 spec and Three.js default.
ORIENTATION_UP = "+Y"
UNITS = "meters"  # nominal — models are not to real-world scale


def _normalize_mesh(mesh: Any) -> Any:
    """Center a trimesh ``Trimesh`` at the bounding-box center (D12).

    Operates in-place when possible and returns the mesh for chaining.
    Non-trimesh objects are returned unchanged (SF3D / Trellis2 textured
    meshes handle their own coordinate space).
    """
    import trimesh as _trimesh

    if not isinstance(mesh, _trimesh.Trimesh):
        return mesh

    centroid = mesh.bounding_box.centroid
    mesh.vertices -= centroid
    return mesh


def _build_standard_metadata(
    *,
    source_model: str,
    mesh: Any = None,
    seed: int | None = None,
    has_texture: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the minimum metadata dict specified by D12.

    Fields always present: ``source_model``, ``orientation``, ``units``.
    Fields present when available: ``seed``, ``vertex_count``, ``face_count``,
    ``has_texture``.
    """
    import trimesh as _trimesh

    meta: dict[str, Any] = {
        "source_model": source_model,
        "orientation": ORIENTATION_UP,
        "units": UNITS,
        "has_texture": has_texture,
    }

    if seed is not None:
        meta["seed"] = seed

    if mesh is not None:
        if isinstance(mesh, _trimesh.Trimesh):
            meta["vertex_count"] = int(mesh.vertices.shape[0])
            meta["face_count"] = int(mesh.faces.shape[0])
        elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            verts = mesh.vertices
            faces = mesh.faces
            if hasattr(verts, "shape"):
                meta["vertex_count"] = int(verts.shape[0])
            if hasattr(faces, "shape"):
                meta["face_count"] = int(faces.shape[0])

    if extra:
        meta.update(extra)

    return meta


async def _finalize_3d_output(
    context: Any,
    *,
    mesh: Any,
    source_model: str,
    node_id: str,
    name_prefix: str,
    format: str = "glb",
    seed: int | None = None,
    has_texture: bool = False,
    include_normals: bool = False,
    center: bool = True,
    extra_metadata: dict[str, Any] | None = None,
    raw_bytes: bytes | None = None,
) -> Any:
    """Shared finalization for all local 3D generators (D12).

    Steps:
    1. Optionally center the mesh at bounding-box origin.
    2. Export to bytes (or use pre-exported *raw_bytes* for Trellis2).
    3. Build standardized metadata.
    4. Call ``context.model3d_from_bytes``.

    Parameters
    ----------
    context:
        The ``ProcessingContext``.
    mesh:
        A trimesh ``Trimesh`` or any mesh-like object. Used for centering,
        metadata extraction, and export.  Ignored when *raw_bytes* is given
        (but still used for metadata).
    raw_bytes:
        Pre-exported bytes (e.g. Trellis2 o_voxel export).  When provided,
        the mesh is NOT re-exported; *raw_bytes* are used directly.
    """
    import trimesh as _trimesh

    if center and isinstance(mesh, _trimesh.Trimesh):
        _normalize_mesh(mesh)

    if raw_bytes is None:
        model_bytes = _export_mesh(mesh, format=format, include_normals=include_normals)
    else:
        model_bytes = raw_bytes

    metadata = _build_standard_metadata(
        source_model=source_model,
        mesh=mesh,
        seed=seed,
        has_texture=has_texture,
        extra=extra_metadata,
    )

    return await context.model3d_from_bytes(
        model_bytes,
        name=f"{name_prefix}_{node_id}.{format}",
        format=format,
        metadata=metadata,
    )
