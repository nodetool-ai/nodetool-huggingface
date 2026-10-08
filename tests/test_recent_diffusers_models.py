"""Z-Image, LongCat-Image and HunyuanVideo 1.5 nodes call their pipelines correctly.

Each node is run with ``run_pipeline_in_thread`` replaced, and the keyword
arguments it would pass are bound against the real diffusers ``__call__``
signature. LongCat-Image and HunyuanVideo 1.5 take no step callback and
HunyuanVideo 1.5 takes no ``guidance_scale``, so a node that passed the
arguments every other node passes would raise ``TypeError`` only after the
user had downloaded the weights.
"""

import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.huggingface import image_to_image, image_to_video
from nodetool.nodes.huggingface import text_to_image, text_to_video
from nodetool.nodes.huggingface.image_to_image import LongCatImageEdit, ZImageImg2Img
from nodetool.nodes.huggingface.image_to_video import HunyuanVideo15I2V
from nodetool.nodes.huggingface.text_to_image import LongCatImage, ZImage
from nodetool.nodes.huggingface.text_to_video import (
    HUNYUAN_VIDEO_15_GEOMETRY,
    HunyuanVideo15,
    apply_hunyuan_video_15_guidance,
    snap_video_geometry,
)

pytest.importorskip("diffusers")

PIPELINES = {
    ZImage: "diffusers.pipelines.z_image.pipeline_z_image:ZImagePipeline",
    ZImageImg2Img: "diffusers.pipelines.z_image.pipeline_z_image_img2img:ZImageImg2ImgPipeline",
    LongCatImage: "diffusers.pipelines.longcat_image.pipeline_longcat_image:LongCatImagePipeline",
    LongCatImageEdit: "diffusers.pipelines.longcat_image.pipeline_longcat_image_edit:LongCatImageEditPipeline",
    HunyuanVideo15: "diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5:HunyuanVideo15Pipeline",
    HunyuanVideo15I2V: "diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5_image2video:HunyuanVideo15ImageToVideoPipeline",
}


def _pipeline_class(node_cls):
    module, name = PIPELINES[node_cls].split(":")
    return getattr(importlib.import_module(module), name)


class _Context:
    def __init__(self):
        self.saved_image = None

    def post_message(self, _message):
        pass

    async def image_to_pil(self, _ref):
        return Image.new("RGBA", (1000, 750))

    async def image_from_pil(self, image):
        self.saved_image = image
        return ImageRef(uri="out.png")


class _Guider:
    """Stands in for ClassifierFreeGuidance: ``new`` copies with overrides."""

    def __init__(self, guidance_scale=6.0):
        self.guidance_scale = guidance_scale

    def new(self, **kwargs):
        return _Guider(**{"guidance_scale": self.guidance_scale, **kwargs})


async def _run(monkeypatch, node, module):
    """Run ``node.process`` and return the kwargs bound to the real signature."""
    captured = {}

    async def run_pipeline(_self, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            images=[Image.new("RGB", (8, 8))],
            frames=[np.zeros((5, 8, 8, 3), dtype=np.float32)],
        )

    async def frames_to_video(_context, frames, fps):
        captured["_fps"] = fps
        return VideoRef(uri="out.mp4")

    monkeypatch.setattr(type(node), "run_pipeline_in_thread", run_pipeline)
    monkeypatch.setattr(module, "run_gc", lambda *_a, **_k: None)
    if hasattr(module, "video_from_frames"):
        monkeypatch.setattr(module, "video_from_frames", frames_to_video)

    if node._pipeline is None:
        node._pipeline = SimpleNamespace(guider=_Guider())
    result = await node.process(_Context())

    call_kwargs = {k: v for k, v in captured.items() if not k.startswith("_")}
    signature = inspect.signature(_pipeline_class(type(node)).__call__)
    signature.bind(None, **call_kwargs)
    return result, captured


@pytest.mark.asyncio
@pytest.mark.parametrize("node_cls", list(PIPELINES), ids=lambda c: c.__name__)
async def test_node_kwargs_match_the_real_pipeline_signature(monkeypatch, node_cls):
    module = {
        ZImage: text_to_image,
        LongCatImage: text_to_image,
        ZImageImg2Img: image_to_image,
        LongCatImageEdit: image_to_image,
        HunyuanVideo15: text_to_video,
        HunyuanVideo15I2V: image_to_video,
    }[node_cls]
    result, _ = await _run(monkeypatch, node_cls(seed=3), module)
    assert result.uri in ("out.png", "out.mp4")


@pytest.mark.parametrize("node_cls", list(PIPELINES), ids=lambda c: c.__name__)
def test_default_model_is_recommended(node_cls):
    node = node_cls()
    repos = [m.repo_id for m in node_cls.get_recommended_models()]
    assert node.get_model_id() == repos[0]
    assert len(repos) == len(set(repos))
    for field in node_cls.get_basic_fields():
        assert field in node_cls.model_fields


@pytest.mark.asyncio
async def test_z_image_snaps_size_onto_its_16px_grid(monkeypatch):
    _, kwargs = await _run(monkeypatch, ZImage(width=1000, height=770), text_to_image)
    assert (kwargs["width"], kwargs["height"]) == (992, 768)
    assert kwargs["guidance_scale"] == 0.0
    assert kwargs["negative_prompt"] is None


@pytest.mark.asyncio
async def test_z_image_img2img_keeps_input_size_on_the_grid(monkeypatch):
    _, kwargs = await _run(monkeypatch, ZImageImg2Img(), image_to_image)
    assert (kwargs["width"], kwargs["height"]) == (992, 736)
    assert kwargs["image"].mode == "RGB"


@pytest.mark.asyncio
async def test_hunyuan_video_snaps_geometry_and_keeps_fps(monkeypatch):
    node = HunyuanVideo15(width=850, height=482, num_frames=120, fps=24)
    _, kwargs = await _run(monkeypatch, node, text_to_video)
    assert (kwargs["width"], kwargs["height"], kwargs["num_frames"]) == (848, 480, 121)
    assert kwargs["_fps"] == 24
    assert "guidance_scale" not in kwargs


def test_hunyuan_geometry_lattice():
    assert snap_video_geometry(
        1280, 720, 121, HUNYUAN_VIDEO_15_GEOMETRY, "HunyuanVideo 1.5"
    ) == (1280, 720, 121)
    assert snap_video_geometry(
        1280, 720, 124, HUNYUAN_VIDEO_15_GEOMETRY, "HunyuanVideo 1.5"
    ) == (1280, 720, 125)


def test_hunyuan_guidance_overrides_for_one_run_and_restores_the_checkpoint_guider():
    original = _Guider(guidance_scale=6.0)
    pipeline = SimpleNamespace(guider=original)

    apply_hunyuan_video_15_guidance(pipeline, 3.0)
    assert pipeline.guider.guidance_scale == 3.0
    assert original.guidance_scale == 6.0

    # A later run on the cached pipeline starts from the checkpoint's guider,
    # not from the previous run's override.
    apply_hunyuan_video_15_guidance(pipeline, 2.0)
    assert pipeline.guider.guidance_scale == 2.0

    apply_hunyuan_video_15_guidance(pipeline, -1.0)
    assert pipeline.guider is original


def test_hunyuan_guidance_works_with_the_real_guider():
    from diffusers.guiders import ClassifierFreeGuidance

    pipeline = SimpleNamespace(guider=ClassifierFreeGuidance(guidance_scale=6.0))
    apply_hunyuan_video_15_guidance(pipeline, 1.5)
    assert pipeline.guider.guidance_scale == 1.5
    apply_hunyuan_video_15_guidance(pipeline, -1.0)
    assert pipeline.guider.guidance_scale == 6.0
