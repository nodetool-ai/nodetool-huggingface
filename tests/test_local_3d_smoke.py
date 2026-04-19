"""
No-GPU smoke tests for local 3D generation nodes.

These tests verify that every node class in local_3d.py can be imported,
instantiated with default values, and that basic metadata methods work
correctly.  No GPU or model weights are required.
"""

import pytest
from nodetool.metadata.types import HuggingFaceModel

# All node classes under test
NODE_CLASSES = []


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
