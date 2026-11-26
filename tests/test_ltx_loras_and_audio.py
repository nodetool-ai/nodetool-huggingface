"""LTX LoRAs, audio to video, ingredients and upscale on the Hugging Face nodes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from pydub import AudioSegment

from nodetool.metadata.types import AudioRef, ImageRef, VideoRef
from nodetool.nodes.huggingface import text_to_video, video_to_video
from nodetool.nodes.huggingface.image_to_video import LTX25ImageToVideo
from nodetool.nodes.huggingface.text_to_video import (
    LTX25,
    FrozenAudioScheduler,
    HFLoraLTX,
    HFLoraLTXConfig,
    LTX25AudioToVideo,
    ltx_audio_mel,
)
from nodetool.nodes.huggingface.video_to_video import (
    LTX23_INGREDIENTS_LORA,
    LTX25_UPSCALE_LORA,
    LTXIngredients,
    LTXUpscale,
    compose_ingredients_sheet,
    ingredients_prompt,
)


def lora(repo_id: str, path: str | None = "w.safetensors", strength: float = 1.0):
    return HFLoraLTXConfig(lora=HFLoraLTX(repo_id=repo_id, path=path), strength=strength)


class Pipeline:
    def __init__(self):
        self.calls = []

    def load_lora_weights(self, path, adapter_name):
        self.calls.append(("load", path, adapter_name))

    def set_adapters(self, names, adapter_weights):
        self.calls.append(("set", names, adapter_weights))


@pytest.fixture
def cache(monkeypatch):
    async def resolve(repo_id, path):
        return f"/cache/{repo_id}/{path}"

    monkeypatch.setattr(text_to_video.HF_FAST_CACHE, "resolve", resolve)


async def test_character_loras_bind_once_with_their_strengths(cache):
    node = LTX25(loras=[lora("me/hero", strength=0.7), lora("me/style")])
    pipeline = Pipeline()
    await node._bind_adapters(pipeline)
    await node._bind_adapters(pipeline)
    assert pipeline.calls == [
        ("load", "/cache/me/hero/w.safetensors", "lora_0"),
        ("load", "/cache/me/style/w.safetensors", "lora_1"),
        ("set", ["lora_0", "lora_1"], [0.7, 1.0]),
    ]


async def test_no_loras_bind_nothing(cache):
    pipeline = Pipeline()
    await LTX25()._bind_adapters(pipeline)
    assert pipeline.calls == []
    assert LTX25().model_cache_suffix() == ""


def test_loras_get_their_own_cached_pipeline():
    plain = LTX25ImageToVideo().model_cache_suffix()
    hero = LTX25ImageToVideo(loras=[lora("me/hero")]).model_cache_suffix()
    weaker = LTX25ImageToVideo(loras=[lora("me/hero", strength=0.5)])
    assert len({plain, hero, weaker.model_cache_suffix()}) == 3


def test_a_lora_needs_its_file_path():
    with pytest.raises(ValueError, match="path of its .safetensors"):
        LTX25(loras=[lora("me/hero", path=None)])._adapters()


async def test_ic_lora_nodes_bind_their_adapter_before_character_loras(cache):
    pipeline = Pipeline()
    await LTXIngredients(loras=[lora("me/hero")])._bind_adapters(pipeline)
    assert pipeline.calls[-1] == ("set", ["ic_lora", "lora_0"], [1.4, 1.0])
    assert pipeline.calls[0][1].endswith(LTX23_INGREDIENTS_LORA.path)


def test_mel_matches_the_audio_vae_input_layout():
    one_second = torch.zeros(1, 16000)
    mel = ltx_audio_mel(one_second)
    # Centered frames: 16000 / 160 + 1.
    assert mel.shape == (1, 2, 101, 64)
    assert torch.allclose(mel, torch.full_like(mel, np.log(1e-5)))


def test_mel_runs_through_a_real_ltx2_audio_vae():
    from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import (
        AutoencoderKLLTX2Audio,
    )

    vae = AutoencoderKLLTX2Audio().eval()
    with torch.no_grad():
        latents = vae.encode(ltx_audio_mel(torch.randn(2, 16000))).latent_dist.mode()
    assert latents.shape[:2] == (1, 8) and latents.shape[-1] == 16


def test_frozen_audio_scheduler_ends_on_the_clean_soundtrack():
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_timesteps
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    clean = torch.randn(1, 5, 128)
    scheduler = FrozenAudioScheduler(
        FlowMatchEulerDiscreteScheduler(), clean, torch.Generator().manual_seed(0)
    )
    sigmas = text_to_video.ltx25_distilled_sigmas()
    timesteps, _ = retrieve_timesteps(scheduler, len(sigmas), "cpu", sigmas=sigmas, mu=1.0)
    audio = torch.randn_like(clean)
    for t in timesteps:
        audio = scheduler.step(torch.randn_like(clean), t, audio, return_dict=False)[0]
        sigma = float(scheduler.sigmas[scheduler.step_index])
        # The held audio sits at the next noise level, around the clean latents.
        assert sigma == 0.0 or not torch.allclose(audio, clean)
    assert torch.allclose(audio, clean)


def test_audio_sets_the_frame_count_on_the_lattice():
    node = LTX25AudioToVideo()
    assert node.frames_for_audio(4000) == 89
    assert node.frames_for_audio(60_000) == 257
    with pytest.raises(ValueError, match="shorter than 9"):
        node.frames_for_audio(200)


class Context:
    def __init__(self, frames=None, segment=None, video_bytes=b""):
        self.frames = frames or []
        self.segment = segment
        self.video_bytes = video_bytes
        self.audio = None

    def post_message(self, _message):
        pass

    async def video_to_frames(self, _video, fps=1):
        return self.frames

    async def image_to_pil(self, _image):
        return Image.new("RGB", (64, 48), (255, 0, 0))

    async def audio_to_audio_segment(self, _audio):
        return self.segment

    async def asset_to_bytes(self, _ref):
        return self.video_bytes

    async def audio_from_numpy(self, samples, sample_rate, num_channels=1, **_):
        self.audio = (samples, sample_rate, num_channels)
        return AudioRef(uri="audio.wav")


def _patch_outputs(monkeypatch, node_class, frame_count):
    captured = {}

    async def run_pipeline(self, **kwargs):
        captured.update(kwargs)
        captured["audio_scheduler"] = getattr(self._pipeline, "audio_scheduler", None)
        return SimpleNamespace(
            frames=[np.zeros((frame_count, 4, 4, 3), dtype=np.float32)],
            audio=torch.zeros((1, 2, 24000 * 10)),
        )

    async def mux_video(_context, frames, **kwargs):
        captured["muxed_frames"] = len(frames)
        captured["muxed_audio"] = kwargs["audio"]
        captured["muxed_rate"] = kwargs["audio_sample_rate"]
        return VideoRef(uri="video.mp4")

    monkeypatch.setattr(node_class, "run_pipeline_in_thread", run_pipeline)
    monkeypatch.setattr(text_to_video, "video_from_frames_with_audio", mux_video)
    monkeypatch.setattr(text_to_video, "run_gc", lambda *_a, **_k: None)
    return captured


async def test_audio_to_video_holds_the_soundtrack(monkeypatch):
    original_scheduler = object()
    pipeline = SimpleNamespace(
        audio_scheduler=original_scheduler,
        scheduler=SimpleNamespace(),
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000)),
        prepare_audio_latents=lambda latents, noise_scale, dtype: latents * 2,
    )
    node = LTX25AudioToVideo(audio=AudioRef(uri="voice.wav"), image=ImageRef(uri="a.png"))
    node._pipeline = pipeline
    clean = torch.ones(1, 8, 23, 16)
    monkeypatch.setattr(node, "_encode_audio", lambda waveform: clean)
    captured = _patch_outputs(monkeypatch, LTX25AudioToVideo, 89)
    voice = AudioSegment.silent(duration=4000, frame_rate=44100)

    result = await node.process(Context(segment=voice))

    assert result["video"] == VideoRef(uri="video.mp4")
    assert captured["num_frames"] == 89
    assert captured["audio_latents"] is clean
    held = captured["audio_scheduler"]
    assert isinstance(held, FrozenAudioScheduler)
    assert torch.equal(held._clean, clean * 2)
    assert pipeline.audio_scheduler is original_scheduler
    [condition] = captured["conditions"]
    assert condition.index == 0
    # The output carries the input audio, cut to the clip.
    assert captured["muxed_rate"] == 44100
    assert captured["muxed_audio"].shape[-1] == round(89 * 44100 / 24)


def test_sheet_layout_and_prompt():
    red = Image.new("RGB", (40, 30), (255, 0, 0))
    assert compose_ingredients_sheet([red]).size == (40, 30)
    sheet = compose_ingredients_sheet([red] * 3)
    assert sheet.size == (1536, 896)
    assert sheet.getpixel((384, 224)) == (255, 0, 0)
    assert sheet.getpixel((1152, 672)) == (0, 0, 0)
    assert ingredients_prompt("a cat", "it jumps") == (
        "Reference sheet: a cat\n\nGenerated video: it jumps"
    )


async def test_ingredients_ride_the_sheet_as_the_reference(monkeypatch):
    node = LTXIngredients(
        images=[ImageRef(uri="a.png"), ImageRef(uri="b.png")],
        sheet_description="a cat, a hat",
        prompt="the cat puts on the hat",
        num_frames=49,
        width=768,
        height=448,
    )
    node._pipeline = SimpleNamespace(
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    )
    captured = _patch_outputs(monkeypatch, LTXIngredients, 49)

    await node.process(Context())

    assert (captured["width"], captured["height"], captured["num_frames"]) == (768, 448, 49)
    assert captured["prompt"].startswith("Reference sheet: a cat, a hat")
    (reference,) = captured["reference_conditions"]
    assert len(reference.frames) == 49 and reference.frames[0].size == (768, 448)
    assert captured["reference_downscale_factor"] == 1
    assert [m.repo_id for m in LTXIngredients.get_recommended_models()][1] == (
        LTX23_INGREDIENTS_LORA.repo_id
    )


def test_upscale_geometry_doubles_and_pads():
    assert LTXUpscale.upscale_geometry((768, 448), 100) == (1536, 896, 105)
    assert LTXUpscale.upscale_geometry((300, 200), 3) == (576, 384, 9)
    with pytest.raises(ValueError, match="past 1536"):
        LTXUpscale.upscale_geometry((1280, 720), 50)


async def test_upscale_pads_trims_and_keeps_the_source_sound(monkeypatch):
    frames = [Image.new("RGB", (320, 192)) for _ in range(30)]
    node = LTXUpscale(video=VideoRef(uri="clip.mp4"), reference_strength=0.6)
    node._pipeline = SimpleNamespace(
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    )
    captured = _patch_outputs(monkeypatch, LTXUpscale, 33)
    sound = AudioSegment.silent(duration=2000, frame_rate=48000)

    async def soundtrack(_context, _video):
        return sound

    monkeypatch.setattr(video_to_video, "source_soundtrack", soundtrack)

    await node.process(Context(frames=frames))

    assert (captured["width"], captured["height"], captured["num_frames"]) == (640, 384, 33)
    (reference,) = captured["reference_conditions"]
    assert len(reference.frames) == 33 and reference.strength == 0.6
    assert captured["conditioning_attention_strength"] == 0.6
    assert captured["reference_downscale_factor"] == LTX25_UPSCALE_LORA.reference_downscale
    assert captured["muxed_frames"] == 30
    assert captured["muxed_rate"] == 48000
