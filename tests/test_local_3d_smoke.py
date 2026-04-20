"""
No-GPU smoke tests for local 3D generation nodes.

These tests verify that every node class in text_to_3d.py and image_to_3d.py
can be imported, instantiated with default values, and that basic metadata
methods work correctly.  No GPU or model weights are required.
"""

import pytest
from nodetool.metadata.types import HuggingFaceModel


def _load_node_classes():
    """Import all 3D node classes from the split modules."""
    from nodetool.nodes.huggingface.text_to_3d import ShapETextTo3D
    from nodetool.nodes.huggingface.image_to_3d import (
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


def test_import_modules():
    """Modules can be imported without error."""
    import nodetool.nodes.huggingface.text_to_3d  # noqa: F401
    import nodetool.nodes.huggingface.image_to_3d  # noqa: F401
    import nodetool.nodes.huggingface._3d_common  # noqa: F401


def test_all_classes_importable(node_classes):
    """Every expected node class exists and is importable."""
    assert len(node_classes) == 7


@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("ShapETextTo3D", "text_to_3d"),
        ("ShapEImageTo3D", "image_to_3d"),
        ("Hunyuan3D", "image_to_3d"),
        ("StableFast3D", "image_to_3d"),
        ("TripoSR", "image_to_3d"),
        ("Trellis2", "image_to_3d"),
        ("TripoSG", "image_to_3d"),
    ],
)
def test_instantiate_with_defaults(cls_name, module):
    """Each node can be instantiated with default field values."""
    import importlib

    mod = importlib.import_module(f"nodetool.nodes.huggingface.{module}")
    cls = getattr(mod, cls_name)
    instance = cls()
    assert instance is not None


@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("ShapETextTo3D", "text_to_3d"),
        ("ShapEImageTo3D", "image_to_3d"),
        ("Hunyuan3D", "image_to_3d"),
        ("StableFast3D", "image_to_3d"),
        ("TripoSR", "image_to_3d"),
        ("Trellis2", "image_to_3d"),
        ("TripoSG", "image_to_3d"),
    ],
)
def test_get_recommended_models(cls_name, module):
    """get_recommended_models returns a non-empty list of HuggingFaceModel."""
    import importlib

    mod = importlib.import_module(f"nodetool.nodes.huggingface.{module}")
    cls = getattr(mod, cls_name)
    models = cls.get_recommended_models()
    assert isinstance(models, list)
    assert len(models) > 0
    for m in models:
        assert isinstance(m, HuggingFaceModel)
        assert m.repo_id, f"{cls_name}: model repo_id must be non-empty"


@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("ShapETextTo3D", "text_to_3d"),
        ("ShapEImageTo3D", "image_to_3d"),
        ("Hunyuan3D", "image_to_3d"),
        ("StableFast3D", "image_to_3d"),
        ("TripoSR", "image_to_3d"),
        ("Trellis2", "image_to_3d"),
        ("TripoSG", "image_to_3d"),
    ],
)
def test_get_basic_fields(cls_name, module):
    """get_basic_fields returns a non-empty list of strings."""
    import importlib

    mod = importlib.import_module(f"nodetool.nodes.huggingface.{module}")
    cls = getattr(mod, cls_name)
    fields = cls.get_basic_fields()
    assert isinstance(fields, list)
    assert len(fields) > 0
    for f in fields:
        assert isinstance(f, str)


@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("ShapETextTo3D", "text_to_3d"),
        ("ShapEImageTo3D", "image_to_3d"),
        ("Hunyuan3D", "image_to_3d"),
        ("StableFast3D", "image_to_3d"),
        ("TripoSR", "image_to_3d"),
        ("Trellis2", "image_to_3d"),
        ("TripoSG", "image_to_3d"),
    ],
)
def test_get_title(cls_name, module):
    """get_title returns a non-empty string."""
    import importlib

    mod = importlib.import_module(f"nodetool.nodes.huggingface.{module}")
    cls = getattr(mod, cls_name)
    title = cls.get_title()
    assert isinstance(title, str)
    assert len(title) > 0


def test_shape_nodes_do_not_require_gpu():
    """Shap-E nodes should support CPU (requires_gpu returns False)."""
    from nodetool.nodes.huggingface.text_to_3d import ShapETextTo3D
    from nodetool.nodes.huggingface.image_to_3d import ShapEImageTo3D

    assert ShapETextTo3D().requires_gpu() is False
    assert ShapEImageTo3D().requires_gpu() is False


def test_heavy_nodes_require_gpu():
    """Heavy pipeline nodes should require GPU."""
    from nodetool.nodes.huggingface.image_to_3d import (
        Hunyuan3D,
        Trellis2,
        TripoSG,
    )

    for cls in [Hunyuan3D, Trellis2, TripoSG]:
        assert cls().requires_gpu() is True, f"{cls.__name__} should require GPU"


def test_experimental_apple_nodes_do_not_require_gpu():
    """SF3D and TripoSR allow non-GPU execution (experimental Apple path)."""
    from nodetool.nodes.huggingface.image_to_3d import (
        StableFast3D,
        TripoSR,
    )

    assert StableFast3D().requires_gpu() is False
    assert TripoSR().requires_gpu() is False


def test_seed_defaults():
    """All nodes with a seed field default to -1 (random)."""
    from nodetool.nodes.huggingface.text_to_3d import ShapETextTo3D
    from nodetool.nodes.huggingface.image_to_3d import (
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
    from nodetool.nodes.huggingface._3d_common import _resolve_device

    device = _resolve_device()
    assert isinstance(device, str)
    assert device in ("cuda", "mps", "cpu")


def test_resolve_seed_passthrough():
    """_resolve_seed returns the seed unchanged when >= 0."""
    pytest.importorskip("torch")
    from nodetool.nodes.huggingface._3d_common import _resolve_seed

    assert _resolve_seed(42) == 42
    assert _resolve_seed(0) == 0


def test_resolve_seed_random():
    """_resolve_seed returns a random uint32 when seed is -1."""
    pytest.importorskip("torch")
    from nodetool.nodes.huggingface._3d_common import _resolve_seed

    seed = _resolve_seed(-1)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


def test_open_pil_image_rejects_garbage():
    """_open_pil_image raises ValueError on non-image bytes."""
    import io
    from nodetool.nodes.huggingface._3d_common import _open_pil_image

    buf = io.BytesIO(b"this is not an image")
    with pytest.raises(ValueError, match="Invalid input image"):
        _open_pil_image(buf)


def test_open_pil_image_valid_png():
    """_open_pil_image succeeds on a minimal valid PNG."""
    import io
    from PIL import Image
    from nodetool.nodes.huggingface._3d_common import _open_pil_image

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
    from nodetool.nodes.huggingface._3d_common import _export_mesh

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
    from nodetool.nodes.huggingface._3d_common import _export_mesh

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    data = _export_mesh(mesh, format="glb", include_normals=True)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_all_nodes_have_platforms_in_docstring():
    """Every node class docstring includes a **Platforms:** line."""
    from nodetool.nodes.huggingface.text_to_3d import ShapETextTo3D
    from nodetool.nodes.huggingface.image_to_3d import (
        ShapEImageTo3D,
        Hunyuan3D,
        StableFast3D,
        TripoSR,
        Trellis2,
        TripoSG,
    )

    for cls in [
        ShapETextTo3D,
        ShapEImageTo3D,
        Hunyuan3D,
        StableFast3D,
        TripoSR,
        Trellis2,
        TripoSG,
    ]:
        doc = cls.__doc__ or ""
        assert (
            "**Platforms:**" in doc
        ), f"{cls.__name__} docstring is missing a **Platforms:** line"


def test_model3d_from_bytes_accepts_metadata():
    """Verify model3d_from_bytes signature accepts metadata kwarg."""
    import inspect
    from nodetool.workflows.processing_context import ProcessingContext

    sig = inspect.signature(ProcessingContext.model3d_from_bytes)
    assert (
        "metadata" in sig.parameters
    ), "model3d_from_bytes must accept a 'metadata' parameter"


# ---------------------------------------------------------------------------
# ClassVar metadata tests (#20, #22a, D9/D11, #8e)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("ShapETextTo3D", "text_to_3d"),
        ("ShapEImageTo3D", "image_to_3d"),
        ("Hunyuan3D", "image_to_3d"),
        ("StableFast3D", "image_to_3d"),
        ("TripoSR", "image_to_3d"),
        ("Trellis2", "image_to_3d"),
        ("TripoSG", "image_to_3d"),
    ],
)
def test_min_vram_gb_present(cls_name, module):
    """Every node declares a positive MIN_VRAM_GB ClassVar."""
    import importlib

    mod = importlib.import_module(f"nodetool.nodes.huggingface.{module}")
    cls = getattr(mod, cls_name)
    assert hasattr(cls, "MIN_VRAM_GB"), f"{cls_name} missing MIN_VRAM_GB"
    assert isinstance(cls.MIN_VRAM_GB, int)
    assert cls.MIN_VRAM_GB > 0


@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("ShapETextTo3D", "text_to_3d"),
        ("ShapEImageTo3D", "image_to_3d"),
        ("Hunyuan3D", "image_to_3d"),
        ("StableFast3D", "image_to_3d"),
        ("TripoSR", "image_to_3d"),
        ("Trellis2", "image_to_3d"),
        ("TripoSG", "image_to_3d"),
    ],
)
def test_estimated_download_gb_present(cls_name, module):
    """Every node declares a positive ESTIMATED_DOWNLOAD_GB ClassVar."""
    import importlib

    mod = importlib.import_module(f"nodetool.nodes.huggingface.{module}")
    cls = getattr(mod, cls_name)
    assert hasattr(
        cls, "ESTIMATED_DOWNLOAD_GB"
    ), f"{cls_name} missing ESTIMATED_DOWNLOAD_GB"
    assert isinstance(cls.ESTIMATED_DOWNLOAD_GB, float)
    assert cls.ESTIMATED_DOWNLOAD_GB > 0


@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("ShapETextTo3D", "text_to_3d"),
        ("ShapEImageTo3D", "image_to_3d"),
        ("Hunyuan3D", "image_to_3d"),
        ("StableFast3D", "image_to_3d"),
        ("TripoSR", "image_to_3d"),
        ("Trellis2", "image_to_3d"),
        ("TripoSG", "image_to_3d"),
    ],
)
def test_license_warning_present(cls_name, module):
    """Every node has a license_warning ClassVar (str or None)."""
    import importlib

    mod = importlib.import_module(f"nodetool.nodes.huggingface.{module}")
    cls = getattr(mod, cls_name)
    assert hasattr(cls, "license_warning"), f"{cls_name} missing license_warning"
    val = cls.license_warning
    assert val is None or isinstance(val, str)


def test_license_warnings_correct():
    """Non-MIT nodes have a non-None license_warning; MIT nodes have None."""
    from nodetool.nodes.huggingface.text_to_3d import ShapETextTo3D
    from nodetool.nodes.huggingface.image_to_3d import (
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
    from nodetool.nodes.huggingface._3d_common import _check_disk_space

    # Use system tmp — should have plenty of space
    with tempfile.TemporaryDirectory() as tmpdir:
        _check_disk_space(0.001, cache_dir=tmpdir)  # 1 MB


def test_check_disk_space_raises_when_tight():
    """_check_disk_space raises OSError when requesting absurd space."""
    import tempfile
    from nodetool.nodes.huggingface._3d_common import _check_disk_space

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(OSError, match="Not enough disk space"):
            _check_disk_space(999_999, cache_dir=tmpdir)  # 999 TB


# ---------------------------------------------------------------------------
# MODEL_REVISIONS table (#19 / D9)
# ---------------------------------------------------------------------------


def test_model_revisions_table_covers_all_repos():
    """MODEL_REVISIONS has an entry for every repo referenced by nodes."""
    from nodetool.nodes.huggingface._3d_common import MODEL_REVISIONS

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
    from nodetool.nodes.huggingface._3d_common import _model_revision

    # All are None currently (placeholder SHAs to be filled in)
    assert _model_revision("openai/shap-e") is None
    # Unknown repo also returns None gracefully
    assert _model_revision("nonexistent/repo") is None


# ---------------------------------------------------------------------------
# Static capability metadata (#8e)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("ShapETextTo3D", "text_to_3d"),
        ("ShapEImageTo3D", "image_to_3d"),
        ("Hunyuan3D", "image_to_3d"),
        ("StableFast3D", "image_to_3d"),
        ("TripoSR", "image_to_3d"),
        ("Trellis2", "image_to_3d"),
        ("TripoSG", "image_to_3d"),
    ],
)
def test_supported_platforms_present(cls_name, module):
    """Every node declares a non-empty SUPPORTED_PLATFORMS ClassVar."""
    import importlib

    mod = importlib.import_module(f"nodetool.nodes.huggingface.{module}")
    cls = getattr(mod, cls_name)
    assert hasattr(
        cls, "SUPPORTED_PLATFORMS"
    ), f"{cls_name} missing SUPPORTED_PLATFORMS"
    assert isinstance(cls.SUPPORTED_PLATFORMS, list)
    assert len(cls.SUPPORTED_PLATFORMS) > 0
    for p in cls.SUPPORTED_PLATFORMS:
        assert p in ("linux", "macos", "windows"), f"unexpected platform: {p}"


@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("ShapETextTo3D", "text_to_3d"),
        ("ShapEImageTo3D", "image_to_3d"),
        ("Hunyuan3D", "image_to_3d"),
        ("StableFast3D", "image_to_3d"),
        ("TripoSR", "image_to_3d"),
        ("Trellis2", "image_to_3d"),
        ("TripoSG", "image_to_3d"),
    ],
)
def test_install_hint_present(cls_name, module):
    """Every node has an INSTALL_HINT ClassVar (str or None)."""
    import importlib

    mod = importlib.import_module(f"nodetool.nodes.huggingface.{module}")
    cls = getattr(mod, cls_name)
    assert hasattr(cls, "INSTALL_HINT"), f"{cls_name} missing INSTALL_HINT"
    val = cls.INSTALL_HINT
    assert val is None or isinstance(val, str)


# ---------------------------------------------------------------------------
# Shared export-contract test (D12 — no GPU required)
# ---------------------------------------------------------------------------


def test_finalize_3d_output_contract():
    """Verify _finalize_3d_output produces standardized metadata and GLB bytes.

    Creates a trivial trimesh triangle, runs it through the shared export
    pipeline, and checks that the metadata contract (D12) is satisfied.
    No GPU or real model weights needed.
    """
    import asyncio

    trimesh = pytest.importorskip("trimesh")
    import numpy as np
    from unittest.mock import AsyncMock

    from nodetool.nodes.huggingface._3d_common import (
        _normalize_mesh,
        _build_standard_metadata,
        _finalize_3d_output,
        ORIENTATION_UP,
        UNITS,
    )

    # ------ _normalize_mesh centers the bounding box ------
    verts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy())
    _normalize_mesh(mesh)
    centroid = mesh.bounding_box.centroid
    np.testing.assert_allclose(centroid, [0.0, 0.0, 0.0], atol=1e-6)

    # ------ _build_standard_metadata has required fields ------
    meta = _build_standard_metadata(
        source_model="test/model",
        mesh=mesh,
        seed=42,
        has_texture=False,
    )
    assert meta["source_model"] == "test/model"
    assert meta["seed"] == 42
    assert meta["orientation"] == ORIENTATION_UP
    assert meta["units"] == UNITS
    assert meta["has_texture"] is False
    assert meta["vertex_count"] == 3
    assert meta["face_count"] == 1

    # ------ _finalize_3d_output calls model3d_from_bytes correctly ------
    mock_context = AsyncMock()
    mock_context.model3d_from_bytes = AsyncMock(return_value="mock_ref")

    mesh2 = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy())
    result = asyncio.run(
        _finalize_3d_output(
            mock_context,
            mesh=mesh2,
            source_model="test/model",
            node_id="node123",
            name_prefix="test",
            seed=7,
        )
    )

    assert result == "mock_ref"
    mock_context.model3d_from_bytes.assert_called_once()
    call_info = mock_context.model3d_from_bytes.call_args
    assert call_info.kwargs["name"] == "test_node123.glb"
    assert call_info.kwargs["format"] == "glb"
    passed_meta = call_info.kwargs["metadata"]
    assert passed_meta["source_model"] == "test/model"
    assert passed_meta["seed"] == 7
    assert passed_meta["orientation"] == "+Y"
    assert passed_meta["units"] == "meters"
    assert "vertex_count" in passed_meta
    assert "face_count" in passed_meta
    # model_bytes should be valid GLB
    model_bytes = call_info.args[0]
    assert isinstance(model_bytes, bytes)
    assert len(model_bytes) > 0


# ---------------------------------------------------------------------------
# VRAM and platform warning helpers (3D-A3 / D10)
# ---------------------------------------------------------------------------


def test_warn_vram_no_crash_without_cuda():
    """_warn_vram does not crash when CUDA is unavailable."""
    from nodetool.nodes.huggingface._3d_common import _warn_vram

    # Should silently return on CI (no GPU)
    _warn_vram(8, "TestNode")


def test_warn_platform_warns_on_unsupported(caplog):
    """_warn_platform logs a warning when platform is not in supported list."""
    import logging
    from nodetool.nodes.huggingface._3d_common import _warn_platform

    with caplog.at_level(logging.WARNING):
        # Use an empty list so any platform triggers the warning
        _warn_platform([], "TestNode")

    assert "TestNode" in caplog.text
    assert "not supported" in caplog.text


def test_warn_platform_silent_when_supported(caplog):
    """_warn_platform does not warn when the current platform is supported."""
    import logging
    import sys
    from nodetool.nodes.huggingface._3d_common import _warn_platform

    plat_map = {"linux": "linux", "darwin": "macos", "win32": "windows"}
    current = plat_map.get(sys.platform, "linux")

    with caplog.at_level(logging.WARNING):
        _warn_platform([current], "TestNode")

    assert "not supported" not in caplog.text


# ---------------------------------------------------------------------------
# Error taxonomy tests (GHF4)
# ---------------------------------------------------------------------------


def test_error_taxonomy_hierarchy():
    """All custom exceptions inherit from Local3DError."""
    from nodetool.nodes.huggingface._3d_common import (
        Local3DError,
        MissingDependencyError,
        InsufficientResourcesError,
        UnsupportedPlatformError,
        InvalidInputError,
        ModelLoadError,
        InferenceError,
    )

    for cls in [
        MissingDependencyError,
        InsufficientResourcesError,
        UnsupportedPlatformError,
        InvalidInputError,
        ModelLoadError,
        InferenceError,
    ]:
        assert issubclass(cls, Local3DError), f"{cls.__name__} should extend Local3DError"


def test_error_taxonomy_stdlib_compatibility():
    """Custom exceptions also inherit from appropriate stdlib types."""
    from nodetool.nodes.huggingface._3d_common import (
        MissingDependencyError,
        InsufficientResourcesError,
        UnsupportedPlatformError,
        InvalidInputError,
        ModelLoadError,
        InferenceError,
    )

    assert issubclass(MissingDependencyError, ImportError)
    assert issubclass(InsufficientResourcesError, OSError)
    assert issubclass(UnsupportedPlatformError, RuntimeError)
    assert issubclass(InvalidInputError, ValueError)
    assert issubclass(ModelLoadError, RuntimeError)
    assert issubclass(InferenceError, RuntimeError)


def test_missing_dependency_error_install_hint():
    """MissingDependencyError carries an install_hint attribute."""
    from nodetool.nodes.huggingface._3d_common import MissingDependencyError

    err = MissingDependencyError("need foo", install_hint="pip install foo")
    assert err.install_hint == "pip install foo"
    assert str(err) == "need foo"


# ---------------------------------------------------------------------------
# Runtime availability tests (GHF1)
# ---------------------------------------------------------------------------


def test_runtime_availability_returns_expected_keys():
    """_check_runtime_availability returns all documented keys."""
    from nodetool.nodes.huggingface._3d_common import _check_runtime_availability

    result = _check_runtime_availability(
        node_name="TestNode",
        supported_platforms=["linux", "macos", "windows"],
        requires_gpu=False,
        min_vram_gb=0,
    )
    for key in ["available", "platform_ok", "gpu_ok", "vram_ok", "missing_packages", "issues"]:
        assert key in result, f"missing key: {key}"


def test_runtime_availability_detects_missing_package():
    """_check_runtime_availability detects missing optional packages."""
    from nodetool.nodes.huggingface._3d_common import _check_runtime_availability

    result = _check_runtime_availability(
        node_name="TestNode",
        supported_platforms=["linux", "macos", "windows"],
        requires_gpu=False,
        min_vram_gb=0,
        optional_packages=["_nonexistent_package_12345"],
    )
    assert not result["available"]
    assert "_nonexistent_package_12345" in result["missing_packages"]
    assert any("Missing packages" in issue for issue in result["issues"])


def test_runtime_availability_classmethod_on_nodes():
    """All 3D nodes expose a runtime_availability classmethod."""
    from nodetool.nodes.huggingface.text_to_3d import ShapETextTo3D
    from nodetool.nodes.huggingface.image_to_3d import (
        ShapEImageTo3D,
        Hunyuan3D,
        StableFast3D,
        TripoSR,
        Trellis2,
        TripoSG,
    )

    for cls in [ShapETextTo3D, ShapEImageTo3D, Hunyuan3D, StableFast3D, TripoSR, Trellis2, TripoSG]:
        result = cls.runtime_availability()
        assert isinstance(result, dict), f"{cls.__name__}.runtime_availability() should return dict"
        assert "available" in result


# ---------------------------------------------------------------------------
# Progress-stage helper tests (GHF2)
# ---------------------------------------------------------------------------


def test_report_stage_sends_progress_message():
    """_report_stage posts a NodeProgress message to the context."""
    from unittest.mock import MagicMock
    from nodetool.nodes.huggingface._3d_common import _report_stage

    mock_context = MagicMock()
    _report_stage(mock_context, "node_123", "inference")

    mock_context.post_message.assert_called_once()
    msg = mock_context.post_message.call_args[0][0]
    assert msg.node_id == "node_123"
    assert msg.total == 100


def test_report_stage_custom_progress():
    """_report_stage respects explicit progress/total values."""
    from unittest.mock import MagicMock
    from nodetool.nodes.huggingface._3d_common import _report_stage

    mock_context = MagicMock()
    _report_stage(mock_context, "node_456", "inference", progress=50, total=200)

    msg = mock_context.post_message.call_args[0][0]
    assert msg.progress == 50
    assert msg.total == 200


# ---------------------------------------------------------------------------
# Warm / cold start visibility tests (GHF5)
# ---------------------------------------------------------------------------


def test_log_cache_status_warm(caplog):
    """_log_cache_status logs 'warm start' for cached models."""
    import logging
    from nodetool.nodes.huggingface._3d_common import _log_cache_status

    with caplog.at_level(logging.INFO):
        _log_cache_status("test_key", is_cached=True, node_name="TestNode")

    assert "warm start" in caplog.text


def test_log_cache_status_cold(caplog):
    """_log_cache_status logs 'cold start' for uncached models."""
    import logging
    from nodetool.nodes.huggingface._3d_common import _log_cache_status

    with caplog.at_level(logging.INFO):
        _log_cache_status("test_key", is_cached=False, node_name="TestNode", load_time_s=2.5)

    assert "cold start" in caplog.text
    assert "2.5s" in caplog.text
