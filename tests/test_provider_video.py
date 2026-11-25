"""HuggingFace local provider: image to video and MiniMax-H3 routing."""

from __future__ import annotations

import asyncio
import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from nodetool.huggingface import huggingface_local_provider as provider_module
from nodetool.huggingface.huggingface_local_provider import (
    HuggingFaceLocalProvider,
    ltx_canvas_for_image,
)
from nodetool.metadata.types import Provider, VideoModel, VideoRef

pytest.importorskip("torch")
pytest.importorskip("diffusers")

from nodetool.nodes.huggingface import image_to_video, text_to_video  # noqa: E402
from nodetool.providers.types import ImageToVideoParams  # noqa: E402

VIDEO = VideoRef(uri="memory://video")


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def _recording_node(calls: list[dict]):
    class FakeNode:
        class Speed(str):
            pass

        def __init__(self, **kwargs):
            calls.append(kwargs)

        async def preload_model(self, context):
            pass

        async def process(self, context):
            return {"video": VIDEO, "audio": object()}

    return FakeNode


def _params(model: str, **extra) -> ImageToVideoParams:
    return ImageToVideoParams(
        model=VideoModel(id=model, name=model, provider=Provider.HuggingFace),
        prompt="a fox",
        seed=7,
        **extra,
    )


def test_canvas_keeps_the_image_aspect_on_the_32_pixel_grid():
    assert ltx_canvas_for_image((1920, 1080), None) == (768, 448)
    assert ltx_canvas_for_image((1080, 1920), "720p") == (704, 1280)
    assert ltx_canvas_for_image((1000, 1000), "480p") == (480, 480)


def test_ltx25_image_to_video_passes_start_and_end_frames(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(image_to_video, "LTX25ImageToVideo", _recording_node(calls))
    params = _params("Lightricks/LTX-2.5", end_image=b"end", num_frames=97)

    result = asyncio.run(
        HuggingFaceLocalProvider().image_to_video(
            _png(1280, 720), params, context=MagicMock()
        )
    )

    assert result is VIDEO
    [kwargs] = calls
    assert kwargs["model"].repo_id == "Lightricks/LTX-2.5"
    assert kwargs["image"].data == _png(1280, 720)
    assert kwargs["last_image"].data == b"end"
    assert (kwargs["width"], kwargs["height"]) == (768, 448)
    assert kwargs["num_frames"] == 97
    assert kwargs["seed"] == 7


def test_ltx25_frame_count_follows_duration_when_no_frame_count(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(image_to_video, "LTX25ImageToVideo", _recording_node(calls))
    params = _params("Lightricks/LTX-2.5", duration_seconds=4)

    asyncio.run(
        HuggingFaceLocalProvider().image_to_video(
            _png(640, 640), params, context=MagicMock()
        )
    )

    assert calls[0]["num_frames"] == 96
    assert calls[0]["last_image"].data is None


@pytest.mark.parametrize(
    ("model", "speed"),
    [
        (text_to_video.MINIMAX_H3_REPO_ID, "best"),
        (text_to_video.MINIMAX_H3_REPO_ID + ":fast", "fast"),
    ],
)
def test_minimax_h3_image_to_video_selects_speed(monkeypatch, model, speed):
    calls: list[dict] = []
    original = text_to_video.MiniMaxH3
    fake = _recording_node(calls)
    fake.Speed = original.Speed
    monkeypatch.setattr(text_to_video, "MiniMaxH3", fake)

    result = asyncio.run(
        HuggingFaceLocalProvider().image_to_video(
            _png(64, 64), _params(model, end_image=b"end"), context=MagicMock()
        )
    )

    assert result is VIDEO
    [kwargs] = calls
    assert kwargs["speed"] == original.Speed(speed)
    assert kwargs["image"].data == _png(64, 64)
    assert kwargs["last_image"].data == b"end"
    assert kwargs["num_frames"] == 124


def test_minimax_h3_text_to_video_clamps_into_the_model_window(monkeypatch):
    calls: list[dict] = []
    original = text_to_video.MiniMaxH3
    fake = _recording_node(calls)
    fake.Speed = original.Speed
    monkeypatch.setattr(text_to_video, "MiniMaxH3", fake)

    result = asyncio.run(
        HuggingFaceLocalProvider().text_to_video(
            prompt="x",
            model=text_to_video.MINIMAX_H3_REPO_ID,
            num_frames=49,
            context=MagicMock(),
        )
    )

    assert result is VIDEO
    assert calls[0]["num_frames"] == text_to_video.MINIMAX_H3_MIN_FRAMES
    assert calls[0]["image"].data is None


def test_image_to_video_rejects_other_models():
    with pytest.raises(ValueError, match="LTX-2.5 or MiniMax-H3"):
        asyncio.run(
            HuggingFaceLocalProvider().image_to_video(
                _png(64, 64), _params("a/wan"), context=MagicMock()
            )
        )


def test_video_models_mark_ltx25_for_image_to_video_and_list_h3(monkeypatch):
    async def fake_cache():
        return [
            VideoModel(
                id="Lightricks/LTX-2.5",
                name="LTX-2.5",
                supported_tasks=["text_to_video"],
            ),
            VideoModel(id="a/wan", name="wan", supported_tasks=["text_to_video"]),
        ]

    async def class_name(self, model):
        return {"Lightricks/LTX-2.5": "LTX2Pipeline", "a/wan": "WanPipeline"}[model]

    cached = {
        (text_to_video.MINIMAX_H3_REPO_ID, "modular_model_index.json"),
        (
            text_to_video.MINIMAX_H3_FAST_LORA.repo_id,
            text_to_video.MINIMAX_H3_FAST_LORA.path,
        ),
    }

    async def resolve(repo_id, path):
        return "/cache/file" if (repo_id, path) in cached else None

    from nodetool.integrations.huggingface import huggingface_models

    monkeypatch.setattr(
        provider_module, "get_text_to_video_models_from_hf_cache", fake_cache
    )
    monkeypatch.setattr(
        HuggingFaceLocalProvider, "_read_model_index_class_name", class_name
    )
    monkeypatch.setattr(huggingface_models.HF_FAST_CACHE, "resolve", resolve)

    models = {
        m.id: m.supported_tasks
        for m in asyncio.run(HuggingFaceLocalProvider().get_available_video_models())
    }

    both = ["text_to_video", "image_to_video"]
    assert models == {
        "Lightricks/LTX-2.5": both,
        "a/wan": ["text_to_video"],
        text_to_video.MINIMAX_H3_REPO_ID: both,
        text_to_video.MINIMAX_H3_REPO_ID + ":fast": both,
    }
