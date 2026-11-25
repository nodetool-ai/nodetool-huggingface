from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from nodetool.metadata.types import AudioRef, VideoRef
from nodetool.nodes.huggingface import text_to_video, video_to_video
from nodetool.nodes.huggingface.text_to_video import ltx25_distilled_sigmas
from nodetool.nodes.huggingface.video_to_video import (
    LTX23_IC_LORAS,
    LTXControl,
    LTXVideoControl,
)


class Context:
    def __init__(self, frames):
        self.frames = frames

    def post_message(self, _message):
        pass

    async def video_to_frames(self, _video, fps=1):
        return self.frames

    async def audio_from_numpy(self, samples, sample_rate, num_channels=1, **_):
        return AudioRef(uri="audio.wav")


def test_metadata_points_at_the_distilled_23_checkpoint_and_both_loras():
    from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2InContextPipeline

    node = LTXVideoControl()
    assert node.get_model_id() == "diffusers/LTX-2.3-Distilled-Diffusers"
    assert node._pipeline_class() is LTX2InContextPipeline
    recommended = [
        (m.repo_id, m.path) for m in LTXVideoControl.get_recommended_models()
    ]
    assert recommended == [
        ("diffusers/LTX-2.3-Distilled-Diffusers", None),
        (
            "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
            "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        ),
        (
            "Lightricks/LTX-2.3-22b-IC-LoRA-Colorization",
            "ltx-2.3-22b-ic-lora-colorization-0.9.safetensors",
        ),
    ]


def test_each_control_gets_its_own_cached_pipeline():
    motion = LTXVideoControl(control=LTXControl.MOTION_CONTROL)
    color = LTXVideoControl(control=LTXControl.COLORIZE)
    assert motion.model_cache_suffix() != color.model_cache_suffix()
    assert (
        motion.model_cache_suffix()
        != LTXVideoControl(lora_strength=0.5).model_cache_suffix()
    )


def test_colorize_adds_its_trigger_once():
    node = LTXVideoControl(control=LTXControl.COLORIZE, prompt="a rabbit on rocks")
    assert node._prompt() == "COLORIZE a rabbit on rocks"
    node.prompt = "COLORIZE a rabbit"
    assert node._prompt() == "COLORIZE a rabbit"
    assert LTXVideoControl(prompt="a monk")._prompt() == "a monk"


def test_geometry_keeps_the_clip_aspect_and_halved_reference_lattice():
    node = LTXVideoControl(control=LTXControl.MOTION_CONTROL)
    # 1920x1080 at 768 on the long side is 768x432, snapped to multiples of 64.
    assert node._output_geometry((1920, 1080), 300) == (768, 448, 121)


def test_geometry_follows_a_shorter_clip_down_onto_the_lattice():
    node = LTXVideoControl(control=LTXControl.COLORIZE)
    assert node._output_geometry((640, 480), 60) == (768, 576, 57)


def test_geometry_refuses_a_clip_shorter_than_nine_frames():
    with pytest.raises(ValueError, match="at least 9"):
        LTXVideoControl()._output_geometry((640, 480), 5)


@pytest.mark.asyncio
async def test_bind_adapters_loads_the_lora_once(monkeypatch):
    calls = []

    class Pipeline:
        def load_lora_weights(self, path, adapter_name):
            calls.append(("load", path, adapter_name))

        def set_adapters(self, names, adapter_weights):
            calls.append(("set", names, adapter_weights))

    async def resolve(repo_id, path):
        return f"/cache/{repo_id}/{path}"

    monkeypatch.setattr(video_to_video.HF_FAST_CACHE, "resolve", resolve)
    node = LTXVideoControl(lora_strength=0.8)
    pipeline = Pipeline()

    await node._bind_adapters(pipeline)
    await node._bind_adapters(pipeline)

    lora = LTX23_IC_LORAS[LTXControl.MOTION_CONTROL]
    assert calls == [
        ("load", f"/cache/{lora.repo_id}/{lora.path}", "ic_lora"),
        ("set", ["ic_lora"], [0.8]),
    ]


@pytest.mark.asyncio
async def test_bind_adapters_requires_the_downloaded_lora(monkeypatch):
    async def resolve(_repo_id, _path):
        return None

    monkeypatch.setattr(video_to_video.HF_FAST_CACHE, "resolve", resolve)
    with pytest.raises(ValueError, match="Union-Control"):
        await LTXVideoControl()._bind_adapters(SimpleNamespace())


@pytest.mark.asyncio
async def test_process_passes_the_clip_as_a_reference(monkeypatch):
    from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition

    frames = [Image.new("L", (640, 360)) for _ in range(30)]
    node = LTXVideoControl(video=VideoRef(uri="clip.mp4"))
    node._pipeline = SimpleNamespace(
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    )
    captured = {}

    async def run_pipeline(_self, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            frames=[np.zeros((25, 4, 4, 3), dtype=np.float32)],
            audio=torch.zeros((1, 2, 100)),
        )

    async def mux_video(_context, _frames, **_kwargs):
        return VideoRef(uri="video.mp4")

    monkeypatch.setattr(LTXVideoControl, "run_pipeline_in_thread", run_pipeline)
    monkeypatch.setattr(text_to_video, "video_from_frames_with_audio", mux_video)
    monkeypatch.setattr(text_to_video, "run_gc", lambda *_a, **_k: None)

    result = await node.process(Context(frames))

    assert result["video"] == VideoRef(uri="video.mp4")
    assert (captured["width"], captured["height"], captured["num_frames"]) == (
        768,
        448,
        25,
    )
    assert captured["reference_downscale_factor"] == 2
    assert captured["sigmas"] == ltx25_distilled_sigmas()
    (reference,) = captured["reference_conditions"]
    assert isinstance(reference, LTX2ReferenceCondition)
    assert len(reference.frames) == 25
    assert all(frame.mode == "RGB" for frame in reference.frames)
