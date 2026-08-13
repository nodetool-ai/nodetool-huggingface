from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nodetool.metadata.types import AudioRef, VideoRef
from nodetool.nodes.huggingface import text_to_video
from nodetool.nodes.huggingface.text_to_video import LTX25


def test_ltx25_metadata_and_model():
    node = LTX25()

    assert LTX25.get_title() == "LTX-2.5"
    assert node.get_model_id() == "Lightricks/LTX-2.5-Diffusers"
    assert [model.repo_id for model in LTX25.get_recommended_models()] == [
        "Lightricks/LTX-2.5-Diffusers"
    ]
    assert LTX25.get_basic_fields() == [
        "model",
        "prompt",
        "num_frames",
        "height",
        "width",
    ]

    outputs = LTX25.get_metadata(include_model_info=False).outputs
    assert [(slot.name, slot.type.type) for slot in outputs] == [
        ("video", "video"),
        ("audio", "audio"),
    ]


@pytest.mark.asyncio
async def test_ltx25_process_returns_muxed_video_and_audio(monkeypatch):
    node = LTX25(seed=-1, frame_rate=24.0)
    node._pipeline = SimpleNamespace(
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    )
    frames = np.zeros((3, 8, 8, 3), dtype=np.float32)
    waveform = torch.zeros((1, 2, 100), dtype=torch.float32)
    pipeline_kwargs = {}

    async def run_pipeline(_self, **kwargs):
        pipeline_kwargs.update(kwargs)
        return SimpleNamespace(frames=[frames], audio=waveform)

    async def mux_video(context, output_frames, **kwargs):
        assert output_frames is frames
        assert kwargs["audio"].shape == (2, 100)
        assert kwargs["audio_sample_rate"] == 24000
        assert kwargs["fps"] == 24
        return VideoRef(uri="video.mp4")

    class Context:
        def post_message(self, _message):
            pass

        async def audio_from_numpy(
            self, samples, sample_rate, num_channels=1, **_kwargs
        ):
            assert samples.shape == (100, 2)
            assert sample_rate == 24000
            assert num_channels == 2
            return AudioRef(uri="audio.wav")

    monkeypatch.setattr(LTX25, "run_pipeline_in_thread", run_pipeline)
    monkeypatch.setattr(text_to_video, "video_from_frames_with_audio", mux_video)
    monkeypatch.setattr(text_to_video, "run_gc", lambda *_args, **_kwargs: None)

    result = await node.process(Context())

    assert result == {
        "video": VideoRef(uri="video.mp4"),
        "audio": AudioRef(uri="audio.wav"),
    }
    assert pipeline_kwargs["output_type"] == "np"
    assert pipeline_kwargs["frame_rate"] == 24.0
