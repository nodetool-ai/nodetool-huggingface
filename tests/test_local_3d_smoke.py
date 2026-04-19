"""
No-GPU smoke tests for local 3D generation nodes.

These tests verify that every node class in local_3d.py can be imported,
instantiated with default values, and that basic metadata methods work
correctly.  No GPU or model weights are required.
"""

import pytest
from nodetool.metadata.types import HuggingFaceModel


def _load_node_classes():
    """Import all 3D node classes from the renamed module."""
    from nodetool.nodes.huggingface.local_3d import (
        ShapETextTo3D,
        ShapEImageTo3D,
        Hunyuan3D,
        StableFast3D,
        TripoSR,
        Trellis2,
        TripoSG,
    )

    return [
        ShapETextTo3D,
        ShapEImageTo3D,
        Hunyuan3D,
        StableFast3D,
        TripoSR,
        Trellis2,
        TripoSG,
    ]


@pytest.fixture(scope="module")
def node_classes():
    return _load_node_classes()


def test_import_module():
    """Module can be imported without error."""
    import nodetool.nodes.huggingface.local_3d  # noqa: F401


def test_all_classes_importable(node_classes):
    """Every expected node class exists and is importable."""
    assert len(node_classes) == 7


@pytest.mark.parametrize(
    "cls_name",
    [
        "ShapETextTo3D",
        "ShapEImageTo3D",
        "Hunyuan3D",
        "StableFast3D",
        "TripoSR",
        "Trellis2",
        "TripoSG",
    ],
)
def test_instantiate_with_defaults(cls_name):
    """Each node can be instantiated with default field values."""
    from nodetool.nodes.huggingface import local_3d

    cls = getattr(local_3d, cls_name)
    instance = cls()
    assert instance is not None


@pytest.mark.parametrize(
    "cls_name",
    [
        "ShapETextTo3D",
        "ShapEImageTo3D",
        "Hunyuan3D",
        "StableFast3D",
        "TripoSR",
        "Trellis2",
        "TripoSG",
    ],
)
def test_get_recommended_models(cls_name):
    """get_recommended_models returns a non-empty list of HuggingFaceModel."""
    from nodetool.nodes.huggingface import local_3d

    cls = getattr(local_3d, cls_name)
    models = cls.get_recommended_models()
    assert isinstance(models, list)
    assert len(models) > 0
    for m in models:
        assert isinstance(m, HuggingFaceModel)
        assert m.repo_id, f"{cls_name}: model repo_id must be non-empty"


@pytest.mark.parametrize(
    "cls_name",
    [
        "ShapETextTo3D",
        "ShapEImageTo3D",
        "Hunyuan3D",
        "StableFast3D",
        "TripoSR",
        "Trellis2",
        "TripoSG",
    ],
)
def test_get_basic_fields(cls_name):
    """get_basic_fields returns a non-empty list of strings."""
    from nodetool.nodes.huggingface import local_3d

    cls = getattr(local_3d, cls_name)
    fields = cls.get_basic_fields()
    assert isinstance(fields, list)
    assert len(fields) > 0
    for f in fields:
        assert isinstance(f, str)


@pytest.mark.parametrize(
    "cls_name",
    [
        "ShapETextTo3D",
        "ShapEImageTo3D",
        "Hunyuan3D",
        "StableFast3D",
        "TripoSR",
        "Trellis2",
        "TripoSG",
    ],
)
def test_get_title(cls_name):
    """get_title returns a non-empty string."""
    from nodetool.nodes.huggingface import local_3d

    cls = getattr(local_3d, cls_name)
    title = cls.get_title()
    assert isinstance(title, str)
    assert len(title) > 0


def test_shape_nodes_do_not_require_gpu():
    """Shap-E nodes should support CPU (requires_gpu returns False)."""
    from nodetool.nodes.huggingface.local_3d import ShapETextTo3D, ShapEImageTo3D

    assert ShapETextTo3D().requires_gpu() is False
    assert ShapEImageTo3D().requires_gpu() is False


def test_heavy_nodes_require_gpu():
    """Heavy pipeline nodes should require GPU."""
    from nodetool.nodes.huggingface.local_3d import (
        Hunyuan3D,
        StableFast3D,
        TripoSR,
        Trellis2,
        TripoSG,
    )

    for cls in [Hunyuan3D, StableFast3D, TripoSR, Trellis2, TripoSG]:
        assert cls().requires_gpu() is True, f"{cls.__name__} should require GPU"


def test_seed_defaults():
    """All nodes with a seed field default to -1 (random)."""
    from nodetool.nodes.huggingface.local_3d import (
        ShapETextTo3D,
        ShapEImageTo3D,
        Hunyuan3D,
        Trellis2,
        TripoSG,
    )

    for cls in [ShapETextTo3D, ShapEImageTo3D, Hunyuan3D, Trellis2, TripoSG]:
        instance = cls()
        assert instance.seed == -1, f"{cls.__name__} seed should default to -1"


def test_resolve_device_returns_string():
    """_resolve_device returns a non-empty string."""
    pytest.importorskip("torch")
    from nodetool.nodes.huggingface.local_3d import _resolve_device

    device = _resolve_device()
    assert isinstance(device, str)
    assert device in ("cuda", "mps", "cpu")


def test_resolve_seed_passthrough():
    """_resolve_seed returns the seed unchanged when >= 0."""
    pytest.importorskip("torch")
    from nodetool.nodes.huggingface.local_3d import _resolve_seed

    assert _resolve_seed(42) == 42
    assert _resolve_seed(0) == 0


def test_resolve_seed_random():
    """_resolve_seed returns a random uint32 when seed is -1."""
    pytest.importorskip("torch")
    from nodetool.nodes.huggingface.local_3d import _resolve_seed

    seed = _resolve_seed(-1)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


def test_open_pil_image_rejects_garbage():
    """_open_pil_image raises ValueError on non-image bytes."""
    import io
    from nodetool.nodes.huggingface.local_3d import _open_pil_image

    buf = io.BytesIO(b"this is not an image")
    with pytest.raises(ValueError, match="Invalid input image"):
        _open_pil_image(buf)


def test_open_pil_image_valid_png():
    """_open_pil_image succeeds on a minimal valid PNG."""
    import io
    from PIL import Image
    from nodetool.nodes.huggingface.local_3d import _open_pil_image

    # Create a tiny 2x2 red PNG in memory
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    result = _open_pil_image(buf, mode="RGBA")
    assert result.mode == "RGBA"
    assert result.size == (2, 2)


def test_export_mesh_roundtrip():
    """_export_mesh produces non-empty GLB bytes from a trimesh object."""
    trimesh = pytest.importorskip("trimesh")
    import numpy as np
    from nodetool.nodes.huggingface.local_3d import _export_mesh

    # Simple triangle
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    data = _export_mesh(mesh, format="glb")
    assert isinstance(data, bytes)
    assert len(data) > 0
    # GLB magic bytes: "glTF"
    assert data[:4] == b"glTF"


def test_export_mesh_include_normals():
    """_export_mesh with include_normals=True still produces valid output."""
    trimesh = pytest.importorskip("trimesh")
    import numpy as np
    from nodetool.nodes.huggingface.local_3d import _export_mesh

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    data = _export_mesh(mesh, format="glb", include_normals=True)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_all_nodes_have_platforms_in_docstring():
    """Every node class docstring includes a **Platforms:** line."""
    from nodetool.nodes.huggingface import local_3d

    for cls_name in [
        "ShapETextTo3D",
        "ShapEImageTo3D",
        "Hunyuan3D",
        "StableFast3D",
        "TripoSR",
        "Trellis2",
        "TripoSG",
    ]:
        cls = getattr(local_3d, cls_name)
        doc = cls.__doc__ or ""
        assert (
            "**Platforms:**" in doc
        ), f"{cls_name} docstring is missing a **Platforms:** line"


def test_model3d_from_bytes_accepts_metadata():
    """Verify model3d_from_bytes signature accepts metadata kwarg."""
    import inspect
    from nodetool.workflows.processing_context import ProcessingContext

    sig = inspect.signature(ProcessingContext.model3d_from_bytes)
    assert (
        "metadata" in sig.parameters
    ), "model3d_from_bytes must accept a 'metadata' parameter"


# ---------------------------------------------------------------------------
# ClassVar metadata tests (#20, #22a, D9/D11)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls_name",
    [
        "ShapETextTo3D",
        "ShapEImageTo3D",
        "Hunyuan3D",
        "StableFast3D",
        "TripoSR",
        "Trellis2",
        "TripoSG",
    ],
)
def test_min_vram_gb_present(cls_name):
    """Every node declares a positive MIN_VRAM_GB ClassVar."""
    from nodetool.nodes.huggingface import local_3d

    cls = getattr(local_3d, cls_name)
    assert hasattr(cls, "MIN_VRAM_GB"), f"{cls_name} missing MIN_VRAM_GB"
    assert isinstance(cls.MIN_VRAM_GB, int)
    assert cls.MIN_VRAM_GB > 0


@pytest.mark.parametrize(
    "cls_name",
    [
        "ShapETextTo3D",
        "ShapEImageTo3D",
        "Hunyuan3D",
        "StableFast3D",
        "TripoSR",
        "Trellis2",
        "TripoSG",
    ],
)
def test_estimated_download_gb_present(cls_name):
    """Every node declares a positive ESTIMATED_DOWNLOAD_GB ClassVar."""
    from nodetool.nodes.huggingface import local_3d

    cls = getattr(local_3d, cls_name)
    assert hasattr(
        cls, "ESTIMATED_DOWNLOAD_GB"
    ), f"{cls_name} missing ESTIMATED_DOWNLOAD_GB"
    assert isinstance(cls.ESTIMATED_DOWNLOAD_GB, float)
    assert cls.ESTIMATED_DOWNLOAD_GB > 0


@pytest.mark.parametrize(
    "cls_name",
    [
        "ShapETextTo3D",
        "ShapEImageTo3D",
        "Hunyuan3D",
        "StableFast3D",
        "TripoSR",
        "Trellis2",
        "TripoSG",
    ],
)
def test_license_warning_present(cls_name):
    """Every node has a license_warning ClassVar (str or None)."""
    from nodetool.nodes.huggingface import local_3d

    cls = getattr(local_3d, cls_name)
    assert hasattr(cls, "license_warning"), f"{cls_name} missing license_warning"
    val = cls.license_warning
    assert val is None or isinstance(val, str)


def test_license_warnings_correct():
    """Non-MIT nodes have a non-None license_warning; MIT nodes have None."""
    from nodetool.nodes.huggingface.local_3d import (
        ShapETextTo3D,
        ShapEImageTo3D,
        Hunyuan3D,
        StableFast3D,
        TripoSR,
        Trellis2,
        TripoSG,
    )

    # MIT nodes
    for cls in [ShapETextTo3D, ShapEImageTo3D, TripoSR, TripoSG]:
        assert cls.license_warning is None, f"{cls.__name__} should be None (MIT)"

    # Non-MIT nodes
    for cls in [Hunyuan3D, StableFast3D, Trellis2]:
        assert (
            cls.license_warning is not None
        ), f"{cls.__name__} should have a license warning"
        assert isinstance(cls.license_warning, str)
        assert len(cls.license_warning) > 10


# ---------------------------------------------------------------------------
# Disk-space pre-flight (#21)
# ---------------------------------------------------------------------------


def test_check_disk_space_passes_when_enough():
    """_check_disk_space does not raise when space is ample."""
    import tempfile
    from nodetool.nodes.huggingface.local_3d import _check_disk_space

    # Use system tmp — should have plenty of space
    with tempfile.TemporaryDirectory() as tmpdir:
        _check_disk_space(0.001, cache_dir=tmpdir)  # 1 MB


def test_check_disk_space_raises_when_tight():
    """_check_disk_space raises OSError when requesting absurd space."""
    import tempfile
    from nodetool.nodes.huggingface.local_3d import _check_disk_space

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(OSError, match="Not enough disk space"):
            _check_disk_space(999_999, cache_dir=tmpdir)  # 999 TB


# ---------------------------------------------------------------------------
# MODEL_REVISIONS table (#19 / D9)
# ---------------------------------------------------------------------------


def test_model_revisions_table_covers_all_repos():
    """MODEL_REVISIONS has an entry for every repo referenced by nodes."""
    from nodetool.nodes.huggingface.local_3d import MODEL_REVISIONS

    expected_repos = {
        "openai/shap-e",
        "openai/shap-e-img2img",
        "tencent/Hunyuan3D-2",
        "tencent/Hunyuan3D-2mini",
        "stabilityai/stable-fast-3d",
        "stabilityai/TripoSR",
        "microsoft/TRELLIS.2-4B",
        "VAST-AI/TripoSG",
        "briaai/RMBG-1.4",
    }
    assert expected_repos.issubset(MODEL_REVISIONS.keys())


def test_model_revision_returns_none_for_unset():
    """_model_revision returns None for repos not yet pinned."""
    from nodetool.nodes.huggingface.local_3d import _model_revision

    # All are None currently (placeholder SHAs to be filled in)
    assert _model_revision("openai/shap-e") is None
    # Unknown repo also returns None gracefully
    assert _model_revision("nonexistent/repo") is None
