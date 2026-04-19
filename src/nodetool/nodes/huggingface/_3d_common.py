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
