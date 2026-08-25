"""Output handling of the Qwen-Image-Layered node.

`QwenImageLayeredPipeline.__call__` reads `image.size` on its first line and
returns `images` as one list of per-layer RGBA images per batch item. These
tests drive the node against doubles of that contract — no GPU, no download.
"""

import pytest
from PIL import Image

from nodetool.metadata.types import ImageRef
from nodetool.nodes.huggingface.text_to_image import QwenImageLayered


class FakeContext:
    """Only the two conversions the node uses."""

    def __init__(self, source: Image.Image | None = None):
        self.source = source or Image.new("RGBA", (8, 8))
        self.saved: list[Image.Image] = []

    async def image_to_pil(self, image: ImageRef) -> Image.Image:
        return self.source

    async def image_from_pil(self, image: Image.Image) -> ImageRef:
        # Mirrors the real conversion, which reads `image.size`.
        width, height = image.size
        self.saved.append(image)
        return ImageRef(uri=f"memory://{width}x{height}")


class FakeOutput:
    def __init__(self, images):
        self.images = images


def make_run(output, calls: list):
    async def run_pipeline_in_thread(self, *args, **kwargs):
        calls.append(kwargs)
        return output

    return run_pipeline_in_thread


def make_node(monkeypatch, output, calls: list) -> QwenImageLayered:
    node = QwenImageLayered()
    node.image = ImageRef(uri="memory://input")
    node.layers = 2
    node.num_inference_steps = 6
    node.seed = 42
    node._pipeline = object()

    monkeypatch.setattr(
        QwenImageLayered, "run_pipeline_in_thread", make_run(output, calls)
    )
    return node


@pytest.mark.asyncio
async def test_returns_one_image_ref_per_layer(monkeypatch):
    layers = [Image.new("RGBA", (64, 64)), Image.new("RGBA", (64, 64))]
    node = make_node(monkeypatch, FakeOutput(images=[layers]), [])

    result = await node.process(FakeContext())  # type: ignore[arg-type]

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(ref, ImageRef) for ref in result)


@pytest.mark.asyncio
async def test_passes_the_input_image_to_the_pipeline(monkeypatch):
    calls: list = []
    source = Image.new("RGB", (32, 16))
    node = make_node(monkeypatch, FakeOutput(images=[[Image.new("RGBA", (64, 64))]]), calls)

    await node.process(FakeContext(source))  # type: ignore[arg-type]

    assert calls[0]["image"].size == (32, 16)
    assert calls[0]["layers"] == 2


@pytest.mark.asyncio
async def test_missing_input_image_is_reported_before_the_pipeline_runs(monkeypatch):
    calls: list = []
    node = make_node(monkeypatch, FakeOutput(images=[[Image.new("RGBA", (64, 64))]]), calls)
    node.image = ImageRef()

    with pytest.raises(ValueError, match="requires an input image"):
        await node.process(FakeContext())  # type: ignore[arg-type]

    assert calls == []


@pytest.mark.asyncio
async def test_none_layer_raises_naming_what_came_back(monkeypatch):
    node = make_node(monkeypatch, FakeOutput(images=[[None, None]]), [])

    with pytest.raises(ValueError) as excinfo:
        await node.process(FakeContext())  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "no usable layer images" in message
    assert "NoneType" in message


@pytest.mark.asyncio
async def test_empty_images_raises_naming_what_came_back(monkeypatch):
    node = make_node(monkeypatch, FakeOutput(images=[]), [])

    with pytest.raises(ValueError) as excinfo:
        await node.process(FakeContext())  # type: ignore[arg-type]

    assert "no usable layer images" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_saved_int_resolution_still_loads(monkeypatch):
    calls: list = []
    node = QwenImageLayered(resolution=640)  # type: ignore[arg-type]
    monkeypatch.setattr(
        QwenImageLayered,
        "run_pipeline_in_thread",
        make_run(FakeOutput(images=[[Image.new("RGBA", (64, 64))]]), calls),
    )
    node.image = ImageRef(uri="memory://input")
    node._pipeline = object()

    await node.process(FakeContext())  # type: ignore[arg-type]

    assert calls[0]["resolution"] == 640
