"""Focused tests for the local TTS provider and optional model adapters."""

import io
import base64
import os
import sys
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from nodetool.huggingface.huggingface_local_provider import HuggingFaceLocalProvider
from nodetool.metadata.types import AudioRef
from nodetool.nodes.huggingface.text_to_speech import F5TTS, SupertonicTTS
from nodetool.nodes.huggingface.text_to_speech import KokoroTTS, TextToSpeech
from nodetool.workflows.types import Chunk


class _FakeSupertonic:
    sample_rate = 44_100

    def __init__(self, *, auto_download: bool):
        assert auto_download is True

    def get_voice_style(self, voice_name: str):
        return {"voice": voice_name}

    def synthesize(self, text: str, **kwargs):
        assert text == "Hello"
        assert kwargs["voice_style"] == {"voice": "F2"}
        assert kwargs["lang"] == "de"
        assert kwargs["speed"] == 1.2
        return np.array([[0.0, 0.25, -0.25]], dtype=np.float32), np.array([3])


@pytest.mark.asyncio
async def test_supertonic_adapter_preserves_reported_sample_rate(monkeypatch):
    monkeypatch.setitem(sys.modules, "supertonic", SimpleNamespace(TTS=_FakeSupertonic))
    context = SimpleNamespace(
        audio_from_numpy=AsyncMock(return_value=SimpleNamespace(type="audio"))
    )
    node = SupertonicTTS(
        text="Hello",
        language="de",
        voice=SupertonicTTS.Voice.F2,
        speed=1.2,
    )

    await node.preload_model(context)
    await node.process(context)

    audio, sample_rate = context.audio_from_numpy.await_args.args
    np.testing.assert_array_equal(audio, np.array([0.0, 0.25, -0.25], dtype=np.float32))
    assert sample_rate == 44_100


@pytest.mark.asyncio
async def test_tts_model_discovery_advertises_capabilities():
    provider = HuggingFaceLocalProvider()
    models = await provider.get_available_tts_models()
    by_id = {model.id: model for model in models}

    supertonic = by_id["Supertone/supertonic-3"]
    assert supertonic.capabilities == ["preset_voice", "language_selection"]
    assert len(supertonic.languages) == 32  # 31 ISO languages plus `na` fallback
    assert supertonic.sample_rate == 44_100
    assert supertonic.voices == [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
    ]

    kokoro = by_id["hexgrad/Kokoro-82M"]
    assert "preset_voice" in kokoro.capabilities
    assert kokoro.sample_rate == 24_000
    assert "facebook/mms-tts-eng" in by_id
    assert "suno/bark" in by_id
    f5 = by_id["SWivid/F5-TTS"]
    assert f5.capabilities == [
        "voice_cloning",
        "reference_transcript",
        "language_selection",
    ]
    assert f5.requires_reference_text is True


def _wav_bytes(sample_rate: int = 16_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        samples = np.array([1000, -1000, 2000, -2000], dtype="<i2")
        wav.writeframes(samples.tobytes())
    return output.getvalue()


@pytest.mark.asyncio
async def test_provider_audio_normalization_yields_24khz_mono_pcm():
    context = SimpleNamespace(asset_to_bytes=AsyncMock(return_value=_wav_bytes()))
    chunks = [
        chunk
        async for chunk in HuggingFaceLocalProvider._audio_ref_as_pcm24k(
            context, SimpleNamespace(type="audio")
        )
    ]

    assert chunks
    assert all(chunk.dtype == np.int16 and chunk.ndim == 1 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) > 0


@pytest.mark.asyncio
async def test_f5_adapter_requires_transcript_and_cleans_temporary_audio():
    seen_path = None

    class FakeF5:
        def infer(self, **kwargs):
            nonlocal seen_path
            seen_path = kwargs["ref_file"]
            assert kwargs["ref_text"] == "Reference words."
            assert kwargs["gen_text"] == "Generated words."
            assert kwargs["nfe_step"] == 24
            with open(seen_path, "rb") as reference:
                assert reference.read() == b"RIFF-reference"
            return np.array([0.0, 0.5, -0.5]), 24_000, None

    context = SimpleNamespace(
        is_cancelled=False,
        asset_to_bytes=AsyncMock(return_value=b"RIFF-reference"),
        audio_from_numpy=AsyncMock(return_value=AudioRef()),
    )
    node = F5TTS(
        text="Generated words.",
        reference_audio=AudioRef(data=b"RIFF-reference"),
        reference_text="Reference words.",
        nfe_steps=24,
    )
    node._f5tts = FakeF5()

    await node.process(context)

    assert seen_path is not None
    from pathlib import Path

    assert not Path(seen_path).exists()
    context.audio_from_numpy.assert_awaited_once()


@pytest.mark.asyncio
async def test_f5_adapter_does_not_fall_back_to_asr():
    node = F5TTS(reference_audio=AudioRef(data=b"audio"), reference_text="")
    node._f5tts = object()

    with pytest.raises(ValueError, match="exact reference transcript"):
        await node.process(SimpleNamespace())


@pytest.mark.asyncio
async def test_kokoro_unified_provider_route_yields_pcm(monkeypatch):
    samples = np.array([100, -100, 200], dtype=np.int16)

    async def preload(_self, _context):
        return None

    async def generate(_self, _context):
        yield {
            "audio": None,
            "chunk": Chunk(
                content=base64.b64encode(samples.tobytes()).decode(),
                done=False,
                content_type="audio",
            ),
        }

    monkeypatch.setattr(KokoroTTS, "preload_model", preload)
    monkeypatch.setattr(KokoroTTS, "gen_process", generate)
    provider = HuggingFaceLocalProvider()

    chunks = [
        chunk
        async for chunk in provider.text_to_speech(
            text="Hello",
            model="hexgrad/Kokoro-82M",
            voice="af_heart",
            context=SimpleNamespace(),
        )
    ]

    np.testing.assert_array_equal(np.concatenate(chunks), samples)


@pytest.mark.asyncio
async def test_generic_unified_provider_route_yields_normalized_pcm(monkeypatch):
    async def preload(_self, _context):
        return None

    async def process(_self, _context):
        return AudioRef(uri="memory://generated.wav")

    monkeypatch.setattr(TextToSpeech, "preload_model", preload)
    monkeypatch.setattr(TextToSpeech, "process", process)
    context = SimpleNamespace(asset_to_bytes=AsyncMock(return_value=_wav_bytes()))
    provider = HuggingFaceLocalProvider()

    chunks = [
        chunk
        async for chunk in provider.text_to_speech(
            text="Hello", model="facebook/mms-tts-eng", context=context
        )
    ]

    assert chunks and all(chunk.dtype == np.int16 for chunk in chunks)


@pytest.mark.asyncio
async def test_f5_unified_provider_forwards_reference_bytes(monkeypatch):
    captured = None

    async def preload(self, _context):
        nonlocal captured
        captured = self

    async def process(_self, _context):
        return AudioRef(uri="memory://f5.wav")

    monkeypatch.setattr(F5TTS, "preload_model", preload)
    monkeypatch.setattr(F5TTS, "process", process)
    context = SimpleNamespace(asset_to_bytes=AsyncMock(return_value=_wav_bytes()))
    provider = HuggingFaceLocalProvider()

    chunks = [
        chunk
        async for chunk in provider.text_to_speech(
            text="Generated",
            model="SWivid/F5-TTS",
            reference_audio=b"reference",
            reference_text="Reference",
            context=context,
        )
    ]

    assert chunks
    assert captured is not None
    assert captured.reference_audio.data == b"reference"
    assert captured.reference_text == "Reference"


@pytest.mark.asyncio
async def test_provider_reuses_supertonic_engine_for_second_run(monkeypatch):
    preload_count = 0
    process_count = 0

    async def preload(self, _context):
        nonlocal preload_count
        preload_count += 1
        self._tts = object()

    async def process(self, _context):
        nonlocal process_count
        process_count += 1
        assert self._tts is not None
        return AudioRef(uri="memory://supertonic.wav")

    monkeypatch.setattr(SupertonicTTS, "preload_model", preload)
    monkeypatch.setattr(SupertonicTTS, "process", process)
    context = SimpleNamespace(asset_to_bytes=AsyncMock(return_value=_wav_bytes()))
    provider = HuggingFaceLocalProvider()

    for _ in range(2):
        chunks = [
            chunk
            async for chunk in provider.text_to_speech(
                text="Hello",
                model="Supertone/supertonic-3",
                voice="M1",
                context=context,
            )
        ]
        assert chunks

    assert preload_count == 1
    assert process_count == 2


@pytest.mark.skipif(
    os.environ.get("NODETOOL_TTS_SMOKE") != "1",
    reason="Set NODETOOL_TTS_SMOKE=1 to download and run Supertonic 3",
)
@pytest.mark.asyncio
async def test_supertonic_opt_in_cpu_synthesis_smoke():
    node = SupertonicTTS(text="Hello from NodeTool.")
    context = SimpleNamespace(
        device="cpu",
        audio_from_numpy=AsyncMock(return_value=AudioRef()),
    )
    await node.preload_model(context)
    await node.process(context)
    audio, sample_rate = context.audio_from_numpy.await_args.args
    assert len(audio) > 0
    assert sample_rate == 44_100


@pytest.mark.skipif(
    not all(
        os.environ.get(name)
        for name in ("NODETOOL_F5_REFERENCE_AUDIO", "NODETOOL_F5_REFERENCE_TEXT")
    ),
    reason="Set consented F5 reference fixture environment variables",
)
@pytest.mark.asyncio
async def test_f5_opt_in_gpu_cloning_smoke():
    reference_path = os.environ["NODETOOL_F5_REFERENCE_AUDIO"]
    reference_text = os.environ["NODETOOL_F5_REFERENCE_TEXT"]
    with open(reference_path, "rb") as reference_file:
        reference_bytes = reference_file.read()
    node = F5TTS(
        text="This is a voice cloning smoke test.",
        reference_audio=AudioRef(data=reference_bytes),
        reference_text=reference_text,
    )
    context = SimpleNamespace(
        device="cuda",
        is_cancelled=False,
        asset_to_bytes=AsyncMock(return_value=reference_bytes),
        audio_from_numpy=AsyncMock(return_value=AudioRef()),
    )
    await node.preload_model(context)
    await node.process(context)
    audio, sample_rate = context.audio_from_numpy.await_args.args
    assert len(audio) > 0
    assert sample_rate == 24_000
