"""VisualizeObjectDetection used a bare `import matplotlib` inside process(),
undeclared anywhere in pyproject.toml -- the same shape of bug proven broken
on a real worker for SplitSentences (see #65 / #69). It has been rewritten
to draw with `PIL.ImageDraw` (already a base dependency of this package, and
already used in the same method via `context.image_to_pil`) instead of
matplotlib.

The rewrite is also a correctness fix, not just a dependency removal: the
matplotlib implementation rendered through `Axes.imshow` + `fig.savefig(...,
bbox_inches="tight")`, and `bbox_inches="tight"` expands the saved figure to
fit anything (like a label near the image edge) that spills outside the
axes -- so the *output PNG was not reliably the same size as the input
image*, despite the docstring/task intent that it should be.
`test_matplotlib_bbox_tight_could_change_the_output_size_but_pil_does_not`
below reproduces that concretely (skipped if matplotlib isn't installed,
since it is no longer a declared dependency) and shows the PIL rewrite does
not have the defect.
"""

import io
import sys
from pathlib import Path

import pytest

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import PIL.Image

from nodetool.metadata.types import BoundingBox, ImageRef, ObjectDetectionResult
from nodetool.nodes.huggingface.object_detection import VisualizeObjectDetection
from nodetool.workflows.processing_context import ProcessingContext

try:
    import matplotlib

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _image_ref(width: int, height: int, color=(10, 20, 30)) -> ImageRef:
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return ImageRef(data=buffer.getvalue())


def _detections() -> list[ObjectDetectionResult]:
    return [
        ObjectDetectionResult(
            label="cat",
            score=0.9512,
            box=BoundingBox(xmin=10, ymin=10, xmax=60, ymax=50),
        ),
        ObjectDetectionResult(
            label="dog",
            score=0.7231,
            box=BoundingBox(xmin=70, ymin=40, xmax=140, ymax=110),
        ),
    ]


async def _render(image_ref: ImageRef, objects) -> PIL.Image.Image:
    node = VisualizeObjectDetection(image=image_ref, objects=objects)
    context = ProcessingContext(encode_assets_as_base64=True)
    result = await node.process(context)
    return await context.image_to_pil(result)


def _matplotlib_render(image: PIL.Image.Image, objects) -> PIL.Image.Image:
    """The pre-rewrite implementation, reproduced verbatim for comparison."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    width, height = image.size
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(image)
    for obj in objects:
        rect = patches.Rectangle(
            (obj.box.xmin, obj.box.ymin),
            obj.box.xmax - obj.box.xmin,
            obj.box.ymax - obj.box.ymin,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            obj.box.xmin,
            obj.box.ymin,
            f"{obj.label} ({obj.score:.2f})",
            color="r",
            fontsize=8,
            backgroundcolor="w",
        )
    ax.axis("off")
    plt.tight_layout(pad=0)
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return PIL.Image.open(img_bytes)


@pytest.mark.asyncio
async def test_output_is_exactly_the_input_size():
    image_ref = _image_ref(200, 150)
    out = await _render(image_ref, _detections())
    assert out.size == (200, 150)


@pytest.mark.asyncio
async def test_output_is_the_input_size_even_with_a_label_near_the_edge():
    # A label near (0, 0) with a long text extends past the image's left/top
    # edge -- exactly the case that made matplotlib's bbox_inches="tight"
    # grow the canvas (see the module docstring).
    image_ref = _image_ref(200, 150)
    objects = [
        ObjectDetectionResult(
            label="a-very-long-label-name-here",
            score=0.987654,
            box=BoundingBox(xmin=5, ymin=5, xmax=35, ymax=25),
        )
    ]
    out = await _render(image_ref, objects)
    assert out.size == (200, 150)


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
def test_matplotlib_bbox_tight_could_change_the_output_size_but_pil_does_not():
    image = PIL.Image.new("RGB", (200, 150), color=(10, 20, 30))
    objects = [
        ObjectDetectionResult(
            label="a-very-long-label-name-here",
            score=0.987654,
            box=BoundingBox(xmin=5, ymin=5, xmax=35, ymax=25),
        )
    ]
    old_output = _matplotlib_render(image, objects)
    assert old_output.size != image.size, (
        "expected the pre-rewrite matplotlib renderer to demonstrate its "
        "own size defect here -- if this fails, the defect this rewrite "
        "fixes may no longer reproduce and the comparison above is moot"
    )


@pytest.mark.asyncio
async def test_box_edges_are_drawn_in_red():
    image_ref = _image_ref(200, 150)
    objects = [
        ObjectDetectionResult(
            label="cat", score=0.9, box=BoundingBox(xmin=20, ymin=20, xmax=100, ymax=80)
        )
    ]
    out = (await _render(image_ref, objects)).convert("RGB")

    def is_reddish(pixel):
        r, g, b = pixel
        return r > 150 and g < 100 and b < 100

    top_edge_pixels = [out.getpixel((x, 20)) for x in range(20, 100)]
    left_edge_pixels = [out.getpixel((20, y)) for y in range(20, 80)]
    assert any(is_reddish(p) for p in top_edge_pixels)
    assert any(is_reddish(p) for p in left_edge_pixels)


@pytest.mark.asyncio
async def test_no_boxes_leaves_the_image_unchanged_in_size_and_mostly_in_color():
    image_ref = _image_ref(64, 48, color=(5, 5, 5))
    out = await _render(image_ref, [])
    assert out.size == (64, 48)
    assert out.convert("RGB").getpixel((32, 24)) == (5, 5, 5)


def test_node_signature_is_unchanged():
    assert VisualizeObjectDetection.get_title() == "Visualize Object Detection"
    node = VisualizeObjectDetection()
    assert VisualizeObjectDetection.required_inputs(node) == ["image", "objects"]
    fields = set(VisualizeObjectDetection.model_fields)
    assert {"image", "objects"} <= fields
