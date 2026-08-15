from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nodetool.metadata.types import AudioRef, VideoRef
from nodetool.nodes.huggingface import text_to_video
from nodetool.nodes.huggingface.text_to_video import (
    LTX25,
    LTX25_DISTILLED_STEPS,
    LTX25_UNGUIDED_KWARGS,
    LTX2,
    ltx25_distilled_sigmas,
)


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


class _Pipeline25:
    """Stands in for the 2.5-era LTX2Pipeline, which names the guidance kwargs."""

    def __init__(self):
        self.vocoder = SimpleNamespace(
            config=SimpleNamespace(output_sampling_rate=24000)
        )

    def __call__(
        self,
        prompt=None,
        sigmas=None,
        guidance_scale=4.0,
        stg_scale=0.0,
        modality_scale=1.0,
        audio_guidance_scale=None,
        audio_stg_scale=None,
        audio_modality_scale=None,
        **kwargs,
    ):
        raise AssertionError("not called in tests")


class _Pipeline23:
    """Stands in for a 2.3-era pipeline without the STG/modality kwargs."""

    def __init__(self):
        self.vocoder = SimpleNamespace(
            config=SimpleNamespace(output_sampling_rate=24000)
        )

    def __call__(self, prompt=None, sigmas=None, guidance_scale=4.0, **kwargs):
        raise AssertionError("not called in tests")


def test_distilled_sigma_grid_has_eight_points():
    sigmas = ltx25_distilled_sigmas()
    assert len(sigmas) == LTX25_DISTILLED_STEPS == 8
    assert sigmas[0] == 1.0
    assert sigmas == sorted(sigmas, reverse=True)


def test_ltx25_pins_the_unguided_distilled_recipe():
    node = LTX25()
    assert node.guidance_scale == 1.0
    assert node.num_inference_steps == LTX25_DISTILLED_STEPS
    # The base node keeps the guided SFT recipe.
    assert LTX2().guidance_scale == 4.0
    assert LTX2().num_inference_steps == 40


def test_ltx25_schedules_on_sigmas_not_steps():
    node = LTX25()
    node._pipeline = _Pipeline25()

    kwargs = node._build_pipeline_kwargs(generator=None, callback_on_step_end=None)

    assert kwargs["sigmas"] == ltx25_distilled_sigmas()
    assert "num_inference_steps" not in kwargs
    assert kwargs["guidance_scale"] == 1.0
    assert kwargs["audio_guidance_scale"] == 1.0
    assert kwargs["stg_scale"] == 0.0
    assert kwargs["audio_stg_scale"] == 0.0
    assert kwargs["modality_scale"] == 1.0
    assert kwargs["audio_modality_scale"] == 1.0
    assert kwargs["output_type"] == "np"


def test_ltx25_drops_guidance_kwargs_older_pipelines_do_not_take():
    node = LTX25()
    node._pipeline = _Pipeline23()

    kwargs = node._build_pipeline_kwargs(generator=None, callback_on_step_end=None)

    assert kwargs["sigmas"] == ltx25_distilled_sigmas()
    assert kwargs["guidance_scale"] == 1.0
    for name in LTX25_UNGUIDED_KWARGS:
        assert name not in kwargs


def test_ltx2_keeps_step_based_scheduling():
    node = LTX2()
    kwargs = node._build_pipeline_kwargs(generator=None, callback_on_step_end=None)

    assert kwargs["num_inference_steps"] == 40
    assert kwargs["guidance_scale"] == 4.0
    assert "sigmas" not in kwargs


def test_ltx2_snaps_off_lattice_geometry():
    node = LTX2(width=770, height=500, num_frames=120)
    kwargs = node._build_pipeline_kwargs(generator=None, callback_on_step_end=None)

    assert kwargs["width"] == 768
    assert kwargs["height"] == 512
    assert kwargs["num_frames"] == 121


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
    assert pipeline_kwargs["sigmas"] == ltx25_distilled_sigmas()
    assert "num_inference_steps" not in pipeline_kwargs
