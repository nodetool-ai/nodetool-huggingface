from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from nodetool.metadata.types import AudioRef, ImageRef, VideoRef
from nodetool.nodes.huggingface import text_to_video
from nodetool.nodes.huggingface.image_to_video import (
    LTX25ImageToVideo,
    LTX25Keyframes,
)
from nodetool.nodes.huggingface.text_to_video import (
    LTX25Conditioned,
    ltx2_latent_index,
    ltx2_spread_latent_indices,
    ltx25_distilled_sigmas,
)
from nodetool.nodes.huggingface.video_to_video import LTX25Extend


class Context:
    def __init__(self, frames=None):
        self.images: dict[str, Image.Image] = {}
        self.frames = frames or []
        self.audio_samples = None

    def post_message(self, _message):
        pass

    async def image_to_pil(self, image: ImageRef) -> Image.Image:
        return self.images[image.uri]

    async def video_to_frames(self, _video, fps=1):
        self.fps = fps
        return self.frames

    async def audio_from_numpy(self, samples, sample_rate, num_channels=1, **_):
        self.audio_samples = samples
        return AudioRef(uri="audio.wav")


def _image(context: Context, name: str, color: int) -> ImageRef:
    context.images[name] = Image.new("RGB", (16, 16), (color, color, color))
    return ImageRef(uri=name)


def test_pixel_frames_map_onto_latent_indices():
    assert ltx2_latent_index(0) == 0
    assert ltx2_latent_index(1) == 1
    assert ltx2_latent_index(8) == 1
    assert ltx2_latent_index(9) == 2
    # The last frame of a 121-frame clip is the 16th latent.
    assert ltx2_latent_index(120) == 15


def test_keyframes_spread_from_first_to_last_latent():
    assert ltx2_spread_latent_indices(2, 121) == [0, 15]
    assert ltx2_spread_latent_indices(4, 121) == [0, 5, 10, 15]
    assert ltx2_spread_latent_indices(1, 121) == [0]


def test_conditioned_nodes_use_the_condition_pipeline_and_hide_the_base():
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2ConditionPipeline

    assert not LTX25Conditioned.is_visible()
    for node_class in (LTX25ImageToVideo, LTX25Keyframes, LTX25Extend):
        assert node_class.is_visible()
        assert node_class()._pipeline_class() is LTX2ConditionPipeline
        assert node_class().get_model_id() == "Lightricks/LTX-2.5-Diffusers"
        outputs = node_class.get_metadata(include_model_info=False).outputs
        assert [slot.name for slot in outputs] == ["video", "audio"]


@pytest.mark.asyncio
async def test_image_to_video_pins_the_first_frame_only():
    context = Context()
    node = LTX25ImageToVideo(image=_image(context, "a", 10))

    conditions = await node._conditions(context)

    assert [(c.index, c.strength) for c in conditions] == [(0, 1.0)]


@pytest.mark.asyncio
async def test_image_to_video_adds_the_last_frame_on_the_last_latent():
    context = Context()
    node = LTX25ImageToVideo(
        image=_image(context, "a", 10),
        last_image=_image(context, "b", 200),
        num_frames=120,
        strength=0.8,
    )

    conditions = await node._conditions(context)

    # 120 snaps to 121 frames, whose last frame sits on latent 15.
    assert [(c.index, c.strength) for c in conditions] == [(0, 0.8), (15, 0.8)]
    assert conditions[1].frames.getpixel((0, 0)) == (200, 200, 200)


@pytest.mark.asyncio
async def test_image_to_video_needs_a_start_image():
    with pytest.raises(ValueError, match="starts from"):
        await LTX25ImageToVideo()._conditions(Context())


@pytest.mark.asyncio
async def test_keyframes_are_spread_in_order():
    context = Context()
    images = [_image(context, str(i), i * 40) for i in range(4)]
    node = LTX25Keyframes(keyframes=images, num_frames=121)

    conditions = await node._conditions(context)

    assert [c.index for c in conditions] == [0, 5, 10, 15]
    assert [c.frames.getpixel((0, 0))[0] for c in conditions] == [0, 40, 80, 120]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 9])
async def test_keyframes_reject_counts_outside_two_to_eight(count):
    context = Context()
    images = [_image(context, str(i), i) for i in range(count)]
    with pytest.raises(ValueError, match="2 to 8"):
        await LTX25Keyframes(keyframes=images)._conditions(context)


@pytest.mark.asyncio
async def test_keyframes_refuse_more_keyframes_than_latents():
    context = Context()
    images = [_image(context, str(i), i) for i in range(8)]
    # 25 frames hold only 4 latents, so 8 keyframes would collide.
    with pytest.raises(ValueError, match="do not fit"):
        await LTX25Keyframes(keyframes=images, num_frames=25)._conditions(context)


@pytest.mark.asyncio
async def test_extend_conditions_on_the_source_tail():
    frames = [Image.new("RGB", (8, 8), (i, i, i)) for i in range(40)]
    context = Context(frames=frames)
    node = LTX25Extend(video=VideoRef(uri="clip.mp4"), context_frames=16)

    conditions = await node._conditions(context)

    assert context.fps == 24
    assert len(conditions) == 1
    assert conditions[0].index == 0
    # 16 snaps to 17: the last 17 frames of the source, in order.
    assert [f.getpixel((0, 0))[0] for f in conditions[0].frames] == list(range(23, 40))


@pytest.mark.asyncio
async def test_extend_refuses_a_source_shorter_than_the_context():
    context = Context(frames=[Image.new("RGB", (8, 8))] * 5)
    node = LTX25Extend(video=VideoRef(uri="clip.mp4"))
    with pytest.raises(ValueError, match="fewer than the 17 context frames"):
        await node._conditions(context)


def test_extend_needs_room_for_new_frames():
    node = LTX25Extend(context_frames=33, num_frames=33)
    with pytest.raises(ValueError, match="must be larger"):
        node._context_frame_count()


@pytest.mark.asyncio
async def test_extend_drops_the_context_frames_and_their_sound(monkeypatch):
    frames = [Image.new("RGB", (8, 8))] * 40
    context = Context(frames=frames)
    node = LTX25Extend(video=VideoRef(uri="clip.mp4"), num_frames=49)
    node._pipeline = SimpleNamespace(
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    )
    generated = np.arange(49, dtype=np.float32).reshape(49, 1, 1, 1)
    waveform = torch.zeros((1, 2, 49_000))
    captured = {}

    async def run_pipeline(_self, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(frames=[generated], audio=waveform)

    async def mux_video(_context, output_frames, **kwargs):
        captured["frames"] = output_frames
        captured["audio"] = kwargs["audio"]
        return VideoRef(uri="video.mp4")

    monkeypatch.setattr(LTX25Extend, "run_pipeline_in_thread", run_pipeline)
    monkeypatch.setattr(text_to_video, "video_from_frames_with_audio", mux_video)
    monkeypatch.setattr(text_to_video, "run_gc", lambda *_a, **_k: None)

    result = await node.process(context)

    assert result == {
        "video": VideoRef(uri="video.mp4"),
        "audio": AudioRef(uri="audio.wav"),
    }
    assert len(captured["conditions"]) == 1
    assert captured["sigmas"] == ltx25_distilled_sigmas()
    assert captured["num_frames"] == 49
    # 17 context frames dropped, 32 new ones kept.
    assert captured["frames"][0, 0, 0, 0] == 17
    assert len(captured["frames"]) == 32
    # 17 frames at 24 fps is 17000 samples at 24 kHz.
    assert captured["audio"].shape == (2, 32_000)
    assert context.audio_samples.shape == (32_000, 2)
