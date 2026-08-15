import sys
from pathlib import Path

import pytest

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.background_removal import (  # noqa: E402
    BackgroundRemoval,
    parse_hex_color,
)


def test_background_removal_basics():
    node = BackgroundRemoval()

    assert node.model.repo_id == "briaai/RMBG-2.0"
    assert node.get_model_id() == "briaai/RMBG-2.0"
    assert BackgroundRemoval.get_title() == "Background Removal"
    assert BackgroundRemoval.required_inputs(node) == ["image"]
    assert BackgroundRemoval.get_basic_fields() == [
        "model",
        "image",
        "background_color",
    ]
    # Transparent background and the training resolution are the defaults.
    assert node.background_color == ""
    assert node.resolution == 1024


def test_background_removal_recommended_models():
    models = BackgroundRemoval.get_recommended_models()
    repo_ids = [m.repo_id for m in models]

    assert "briaai/RMBG-2.0" in repo_ids
    assert "ZhengPeng7/BiRefNet" in repo_ids
    assert "briaai/RMBG-1.4" in repo_ids

    # trust_remote_code needs the modeling code from each repo.
    for model in models:
        patterns = model.allow_patterns or []
        assert any(p.endswith("*.py") for p in patterns), model.repo_id


def test_background_removal_outputs_image_and_mask():
    annotations = BackgroundRemoval.OutputType.__annotations__
    assert set(annotations) == {"image", "mask"}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", None),
        ("   ", None),
        ("#ffffff", (255, 255, 255, 255)),
        ("ffffff", (255, 255, 255, 255)),
        ("#000", (0, 0, 0, 255)),
        ("#ff000080", (255, 0, 0, 128)),
        ("#0f08", (0, 255, 0, 136)),
    ],
)
def test_parse_hex_color(value, expected):
    assert parse_hex_color(value) == expected


@pytest.mark.parametrize("value", ["#12345", "not-a-color", "#gggggg"])
def test_parse_hex_color_rejects_garbage(value):
    with pytest.raises(ValueError):
        parse_hex_color(value)


def test_extract_matte_unwraps_nested_outputs():
    class FakeTensor:
        def __init__(self):
            self.sigmoid_called = False

        def sigmoid(self):
            self.sigmoid_called = True
            return self

        def float(self):
            return self

        def cpu(self):
            return self

    final = FakeTensor()

    # BiRefNet-style: list of supervision maps, last one is the prediction.
    assert BackgroundRemoval.extract_matte([object(), final]) is final
    # RMBG-1.4-style: tuple of lists.
    assert BackgroundRemoval.extract_matte(([object()], [object(), final])) is final
    # Plain tensor.
    assert BackgroundRemoval.extract_matte(final) is final
    assert final.sigmoid_called is True


def test_extract_matte_rejects_bad_output():
    with pytest.raises(ValueError):
        BackgroundRemoval.extract_matte([])
    with pytest.raises(ValueError):
        BackgroundRemoval.extract_matte("nonsense")


class FakeContext:
    """Minimal stand-in for ProcessingContext image helpers."""

    device = "cpu"

    def __init__(self, image):
        self._image = image
        self.produced = []

    async def image_to_pil(self, _ref):
        return self._image

    async def image_from_pil(self, image):
        self.produced.append(image)
        return image


def _patch_inference(monkeypatch, alpha):
    """Replace the torch-dependent parts of process() with pure-PIL fakes."""
    import numpy as np

    monkeypatch.setattr(BackgroundRemoval, "preprocess", lambda self, image: "tensor")

    async def fake_run(self, *_args, **_kwargs):
        return "raw-output"

    monkeypatch.setattr(BackgroundRemoval, "run_pipeline_in_thread", fake_run)

    class FakeMatte:
        def squeeze(self):
            return self

        def clamp(self, _lo, _hi):
            return self

        def numpy(self):
            return np.full((4, 4), alpha, dtype=np.float32)

    monkeypatch.setattr(
        BackgroundRemoval, "extract_matte", staticmethod(lambda _out: FakeMatte())
    )


@pytest.mark.asyncio
async def test_process_returns_transparent_cutout(monkeypatch):
    import PIL.Image

    node = BackgroundRemoval()
    node._pipeline = object()
    _patch_inference(monkeypatch, alpha=0.0)

    source = PIL.Image.new("RGB", (8, 6), (10, 20, 30))
    context = FakeContext(source)

    result = await node.process(context)  # type: ignore[arg-type]

    assert set(result) == {"image", "mask"}
    cutout = result["image"]
    mask = result["mask"]

    assert cutout.mode == "RGBA"
    assert cutout.size == (8, 6)
    assert mask.mode == "L"
    assert mask.size == (8, 6)
    # A zero matte means everything is background: fully transparent.
    assert cutout.getpixel((0, 0))[3] == 0
    assert mask.getpixel((0, 0)) == 0


@pytest.mark.asyncio
async def test_process_composites_onto_background_color(monkeypatch):
    import PIL.Image

    node = BackgroundRemoval(background_color="#ff0000")
    node._pipeline = object()
    _patch_inference(monkeypatch, alpha=0.0)

    source = PIL.Image.new("RGB", (8, 6), (10, 20, 30))
    context = FakeContext(source)

    result = await node.process(context)  # type: ignore[arg-type]

    # Fully transparent foreground over red leaves the plate colour.
    assert result["image"].getpixel((0, 0)) == (255, 0, 0, 255)


@pytest.mark.asyncio
async def test_process_keeps_foreground_when_matte_is_opaque(monkeypatch):
    import PIL.Image

    node = BackgroundRemoval(background_color="#ff0000")
    node._pipeline = object()
    _patch_inference(monkeypatch, alpha=1.0)

    source = PIL.Image.new("RGB", (8, 6), (10, 20, 30))
    context = FakeContext(source)

    result = await node.process(context)  # type: ignore[arg-type]

    assert result["image"].getpixel((0, 0)) == (10, 20, 30, 255)
    assert result["mask"].getpixel((0, 0)) == 255


@pytest.mark.asyncio
async def test_process_requires_loaded_model():
    node = BackgroundRemoval()
    with pytest.raises(ValueError):
        await node.process(FakeContext(None))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_move_to_device_moves_model_and_records_device():
    node = BackgroundRemoval()

    class MockModel:
        def __init__(self):
            self.kwargs = None

        def to(self, **kwargs):
            self.kwargs = kwargs

    node._pipeline = MockModel()
    await node.move_to_device("cpu")

    assert node._device == "cpu"
    assert node._pipeline.kwargs is not None
    assert node._pipeline.kwargs["device"] == "cpu"


@pytest.mark.asyncio
async def test_move_to_device_without_model_is_a_noop():
    node = BackgroundRemoval()
    node._pipeline = None
    await node.move_to_device("cuda")
