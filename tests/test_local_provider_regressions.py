"""Regression tests for the local HuggingFace provider and shared helpers.

Every test replaces the model with a fake, so nothing is downloaded.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nodetool.huggingface import huggingface_local_provider as provider_module
from nodetool.huggingface.huggingface_local_provider import (
    HuggingFaceLocalProvider,
    _sampling_kwargs,
)
from nodetool.huggingface.local_provider_utils import pipeline_progress_callback
from nodetool.metadata.types import Message, VideoModel, AudioModel, VideoRef
from nodetool.ml.core.model_manager import ModelManager
from nodetool.workflows.types import NodeProgress

SRC = Path(__file__).resolve().parents[1] / "src" / "nodetool"


# HF2 ---------------------------------------------------------------------


def test_progress_callback_without_node_id_posts_nothing():
    context = MagicMock()
    callback = pipeline_progress_callback(node_id=None, total_steps=4, context=context)
    kwargs = {"latents": object()}
    assert callback(None, 0, 0, kwargs) is kwargs
    context.post_message.assert_not_called()


def test_progress_callback_with_node_id_posts_progress():
    context = MagicMock()
    callback = pipeline_progress_callback(node_id="n1", total_steps=4, context=context)
    callback(None, 2, 0, {})
    (message,), _ = context.post_message.call_args
    assert isinstance(message, NodeProgress)
    assert (message.node_id, message.progress, message.total) == ("n1", 2, 4)


# HF4 ---------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [0, 0.0, -1])
def test_zero_temperature_selects_greedy_decoding(temperature):
    assert _sampling_kwargs(temperature, 0.9, True) == {"do_sample": False}


def test_integer_temperature_is_cast_to_float():
    kwargs = _sampling_kwargs(1, 1, True)
    assert kwargs == {"do_sample": True, "temperature": 1.0, "top_p": 1.0}
    assert isinstance(kwargs["temperature"], float)


def test_do_sample_false_drops_sampling_settings():
    assert _sampling_kwargs(0.7, 0.9, False) == {"do_sample": False}


def test_none_sampling_values_use_defaults():
    assert _sampling_kwargs(None, None, True) == {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
    }


# HF4, HF10, HF12: the text-generation stream ------------------------------


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<bos>" + messages[-1]["content"]


class _FakeTextPipeline:
    """Emits one token per step until a stopping criterion fires."""

    def __init__(self, max_steps: int = 10_000):
        self.tokenizer = _FakeTokenizer()
        self.calls: list[dict] = []
        self.steps = 0
        self.max_steps = max_steps
        self.finished = threading.Event()

    def __call__(self, prompt, **kwargs):
        import torch

        self.calls.append({"prompt": prompt, **kwargs})
        streamer = kwargs["streamer"]
        criteria = kwargs["stopping_criteria"]
        ids = torch.zeros((1, 1), dtype=torch.long)
        try:
            for _ in range(self.max_steps):
                self.steps += 1
                streamer.on_finalized_text("tok ")
                if bool(criteria(ids, None).all()):
                    break
                threading.Event().wait(0.001)
            streamer.on_finalized_text("", stream_end=True)
        finally:
            self.finished.set()


@pytest.fixture
def fake_text_pipeline(monkeypatch):
    pytest.importorskip("torch")
    pipe = _FakeTextPipeline()
    key = "fake/llm_fp16"
    ModelManager._models[key] = pipe
    # The streamer only needs the TextStreamer base class behaviour we bypass.
    import transformers

    class _Streamer:
        def __init__(self, tokenizer, skip_prompt=True, **kwargs):
            pass

        def on_finalized_text(self, text, stream_end=False):
            raise NotImplementedError

    monkeypatch.setattr(transformers, "TextStreamer", _Streamer)
    try:
        yield pipe
    finally:
        ModelManager._models.pop(key, None)


async def _collect(gen, limit=None):
    out = []
    async for chunk in gen:
        out.append(chunk)
        if limit is not None and len(out) >= limit:
            break
    return out


def _stream(provider, **kwargs):
    from nodetool.workflows.processing_context import ProcessingContext

    return provider.generate_messages(
        messages=[Message(role="user", content="hi")],
        model="fake/llm",
        max_tokens=16,
        context=ProcessingContext(),
        **kwargs,
    )


def test_stream_with_zero_temperature_uses_greedy_and_no_extra_bos(
    fake_text_pipeline,
):
    fake_text_pipeline.max_steps = 3
    provider = HuggingFaceLocalProvider()
    chunks = asyncio.run(_collect(_stream(provider, temperature=0, top_p=1)))

    assert "".join(c.content for c in chunks) == "tok tok tok "
    call = fake_text_pipeline.calls[0]
    assert call["do_sample"] is False
    assert "temperature" not in call
    assert call["add_special_tokens"] is False
    assert call["prompt"] == "<bos>hi"


def test_closing_the_stream_stops_generation(fake_text_pipeline):
    provider = HuggingFaceLocalProvider()

    async def run():
        gen = _stream(provider, temperature=0.7)
        await _collect(gen, limit=1)
        await gen.aclose()

    asyncio.run(run())
    assert fake_text_pipeline.finished.wait(5)
    assert fake_text_pipeline.steps < 10_000


# HF6 ---------------------------------------------------------------------


def _patch_class_names(monkeypatch, names: dict[str, str]):
    async def fake_class_name(self, model):
        return names.get(model)

    monkeypatch.setattr(
        HuggingFaceLocalProvider, "_read_model_index_class_name", fake_class_name
    )


def test_video_models_list_only_loadable_pipelines(monkeypatch):
    names = {
        "a/wan": "WanPipeline",
        "a/ltx2": "LTX2Pipeline",
        "a/k5": "Kandinsky5T2VPipeline",
        "a/cog": "CogVideoXPipeline",
        "a/svd": "StableVideoDiffusionPipeline",
        "a/wan-i2v": "WanImageToVideoPipeline",
    }
    _patch_class_names(monkeypatch, names)

    async def fake_cache():
        return [VideoModel(id=repo, name=repo) for repo in names]

    monkeypatch.setattr(
        provider_module, "get_text_to_video_models_from_hf_cache", fake_cache
    )
    models = asyncio.run(HuggingFaceLocalProvider().get_available_video_models())
    assert [m.id for m in models] == ["a/wan", "a/ltx2", "a/k5"]


def test_audio_models_list_only_loadable_pipelines(monkeypatch):
    names = {
        "a/ace": "AceStepPipeline",
        "a/longcat": "LongCatAudioDiTPipeline",
        "a/ldm2": "AudioLDM2Pipeline",
        "a/stable": "StableAudioPipeline",
    }
    _patch_class_names(monkeypatch, names)

    async def fake_cache():
        return [AudioModel(id=repo, name=repo) for repo in names]

    monkeypatch.setattr(
        provider_module, "get_text_to_audio_models_from_hf_cache", fake_cache
    )
    models = asyncio.run(HuggingFaceLocalProvider().get_available_audio_models())
    assert [m.id for m in models] == ["a/ace", "a/longcat"]


def test_text_to_video_rejects_unroutable_pipeline(monkeypatch):
    pytest.importorskip("diffusers")
    _patch_class_names(monkeypatch, {"a/cog": "CogVideoXPipeline"})
    with pytest.raises(ValueError, match="CogVideoXPipeline"):
        asyncio.run(
            HuggingFaceLocalProvider().text_to_video(
                prompt="x", model="a/cog", context=MagicMock()
            )
        )


def test_text_to_audio_rejects_unroutable_pipeline(monkeypatch):
    _patch_class_names(monkeypatch, {"a/ldm2": "AudioLDM2Pipeline"})
    with pytest.raises(ValueError, match="AudioLDM2Pipeline"):
        asyncio.run(
            HuggingFaceLocalProvider().text_to_audio(
                prompt="x", model="a/ldm2", context=MagicMock()
            )
        )


# HF14 --------------------------------------------------------------------


def test_faster_whisper_is_not_offered():
    models = asyncio.run(HuggingFaceLocalProvider().get_available_asr_models())
    assert all("faster-whisper" not in m.id for m in models)

    from nodetool.nodes.huggingface.automatic_speech_recognition import Whisper

    assert all(
        "faster-whisper" not in m.repo_id for m in Whisper.get_recommended_models()
    )


# HF15, HF16 (provider) ---------------------------------------------------


class _FakeAsrPipeline:
    def __init__(self, result):
        self.result = result
        self.samples = None

    def __call__(self, samples, **kwargs):
        self.samples = samples
        return self.result


def test_asr_normalizes_32_bit_audio_and_closes_last_chunk():
    from pydub import AudioSegment

    pytest.importorskip("pydub")
    tone = (np.sin(np.linspace(0, 200, 16_000)) * 0.5 * (2**31 - 1)).astype(np.int32)
    segment = AudioSegment(
        tone.tobytes(), sample_width=4, frame_rate=16_000, channels=1
    )
    buffer = io.BytesIO()
    segment.export(buffer, format="wav")

    model_id = "fake/asr"
    pipe = _FakeAsrPipeline(
        {
            "text": "a b",
            "chunks": [
                {"timestamp": (0.0, 0.4), "text": "a"},
                {"timestamp": (0.4, None), "text": "b"},
            ],
        }
    )
    ModelManager._models[model_id] = pipe
    try:
        result = asyncio.run(
            HuggingFaceLocalProvider().automatic_speech_recognition(
                audio=buffer.getvalue(),
                model=model_id,
                context=MagicMock(),
                word_timestamps=True,
            )
        )
    finally:
        ModelManager._models.pop(model_id, None)

    assert np.abs(pipe.samples).max() <= 1.0
    assert np.abs(pipe.samples).max() > 0.3
    assert result["chunks"][1]["timestamp"] == [0.4, pytest.approx(1.0)]


# HF16 (Whisper node) -----------------------------------------------------


def test_whisper_node_keeps_last_chunk_without_end(monkeypatch):
    from nodetool.metadata.types import AudioRef
    from nodetool.nodes.huggingface.automatic_speech_recognition import (
        Timestamps,
        Whisper,
    )

    async def fake_run(self, *args, **kwargs):
        return {
            "text": "a b",
            "chunks": [
                {"timestamp": (0.0, 0.5), "text": "a"},
                {"timestamp": (0.5, None), "text": "b"},
            ],
        }

    monkeypatch.setattr(Whisper, "run_pipeline_in_thread", fake_run)
    node = Whisper(audio=AudioRef(uri="memory://x"), timestamps=Timestamps.WORD)
    node._pipeline = object()
    context = MagicMock()

    async def audio_to_numpy(*args, **kwargs):
        return np.zeros(32_000, dtype=np.float32), 16_000, 1

    context.audio_to_numpy = audio_to_numpy
    result = asyncio.run(node.process(context))
    assert [c.text for c in result["chunks"]] == ["a", "b"]
    assert tuple(result["chunks"][1].timestamp) == (0.5, 2.0)


# HF19 --------------------------------------------------------------------


def test_image_to_image_parameter_order_matches_base_provider():
    from nodetool.providers.base import BaseProvider

    ours = list(inspect.signature(HuggingFaceLocalProvider.image_to_image).parameters)
    base = list(inspect.signature(BaseProvider.image_to_image).parameters)
    assert ours[: len(base)] == base


# HF20 --------------------------------------------------------------------


def test_text_to_video_via_ltx25_returns_the_video_ref(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from nodetool.nodes.huggingface import text_to_video

    video = VideoRef(uri="memory://video")

    class FakeLTX25:
        def __init__(self, **kwargs):
            pass

        async def preload_model(self, context):
            pass

        async def process(self, context):
            return {"video": video, "audio": object()}

    monkeypatch.setattr(text_to_video, "LTX25", FakeLTX25)
    _patch_class_names(monkeypatch, {"Lightricks/LTX-2.5": "LTX2Pipeline"})
    result = asyncio.run(
        HuggingFaceLocalProvider().text_to_video(
            prompt="x", model="Lightricks/LTX-2.5", context=MagicMock()
        )
    )
    assert result is video


# HF3 ---------------------------------------------------------------------


def test_random_seed_gives_different_generators(monkeypatch):
    pytest.importorskip("torch")
    from nodetool.nodes.huggingface.text_to_audio import AudioLDM

    seeds: list[int] = []

    class _Stop(Exception):
        pass

    async def fake_run(self, *args, generator, **kwargs):
        seeds.append(generator.initial_seed())
        raise _Stop

    monkeypatch.setattr(AudioLDM, "run_pipeline_in_thread", fake_run)
    for seed in (-1, -1, 7):
        node = AudioLDM(seed=seed)
        node._pipeline = object()
        with pytest.raises(_Stop):
            asyncio.run(node.process(MagicMock()))

    assert seeds[0] != seeds[1]
    assert seeds[2] == 7


@pytest.mark.parametrize("module", ["image_to_image.py", "text_to_audio.py"])
def test_every_seeded_generator_is_reseeded_for_random(module):
    source = (SRC / "nodes" / "huggingface" / module).read_text()
    sites = re.findall(
        r"generator = generator\.manual_seed\(self\.seed\)\n(\s+)(\S+)", source
    )
    assert sites, "no seeded generators found"
    assert all(follow == "else:" for _, follow in sites)


# HF5 ---------------------------------------------------------------------


def test_sd15_checkpoint_with_real_shapes_is_detected(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from safetensors.torch import save_file

    from nodetool.huggingface.single_file_models import (
        FAMILY_SD15,
        detect_single_file_checkpoint,
        plan_for_family,
    )

    path = tmp_path / "sd15.safetensors"
    save_file(
        {
            # diffusers reads this tensor's shape before checking any marker.
            "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
            "model.diffusion_model.output_blocks.11.0.skip_connection.weight": (
                torch.zeros(4)
            ),
        },
        str(path),
    )
    assert detect_single_file_checkpoint(str(path)) == plan_for_family(FAMILY_SD15)


# HF7 ---------------------------------------------------------------------


def test_load_pipeline_builds_off_the_event_loop(monkeypatch, tmp_path):
    import transformers

    from nodetool.huggingface.local_provider_utils import load_pipeline

    threads: list[threading.Thread] = []

    def fake_pipeline(task, model, torch_dtype=None, **kwargs):
        threads.append(threading.current_thread())
        return object()

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr(
        "nodetool.huggingface.local_provider_utils._reclaim_vram_before_load",
        lambda label: None,
    )
    context = MagicMock()
    asyncio.run(
        load_pipeline(
            node_id="n",
            context=context,
            pipeline_task="text-generation",
            model_id=str(tmp_path),
            skip_cache=True,
            token=None,
        )
    )
    assert threads and threads[0] is not threading.main_thread()


# HF9 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "loader_path, key",
    [
        (
            "nodetool.huggingface.text_to_image_pipelines.load_text_to_image_pipeline",
            "text-to-image:fake/repo:repo",
        ),
        (
            "nodetool.huggingface.image_to_image_pipelines.load_image_to_image_pipeline",
            "image-to-image:fake/repo:repo",
        ),
    ],
)
def test_cached_pipeline_is_moved_back_to_the_device(monkeypatch, loader_path, key):
    import importlib

    module_name, func_name = loader_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_resolve_hf_device", lambda ctx, dev: "cuda")
    monkeypatch.setattr(
        "nodetool.huggingface.memory_utils.offload_kind", lambda pipeline: None
    )

    class FakePipeline:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

    cached = FakePipeline()
    ModelManager._models[key] = cached
    try:
        pipeline, _ = asyncio.run(
            getattr(module, func_name)(
                context=MagicMock(),
                model_id="fake/repo",
                model_path=None,
                node_id=None,
            )
        )
    finally:
        ModelManager._models.pop(key, None)
    assert pipeline is cached
    assert cached.device == "cuda"


# HF17 --------------------------------------------------------------------


def test_mp4_bytes_encodes_into_a_closed_file_and_removes_it():
    from nodetool.huggingface.video_utils import _mp4_bytes

    seen: list[str] = []

    def encode(path):
        seen.append(path)
        # Reopening by name is what fails on Windows while the file is open.
        with open(path, "wb") as handle:
            handle.write(b"mp4")

    assert asyncio.run(_mp4_bytes(encode)) == b"mp4"
    assert not Path(seen[0]).exists()


# HF18 --------------------------------------------------------------------


def test_split_sentences_reads_a_document_passed_by_uri(monkeypatch):
    from nodetool.metadata.types import DocumentRef
    from nodetool.nodes.huggingface import sentence_transformers as st

    class FakeTokenizer:
        def encode(self, text, **kwargs):
            return [0, *range(1, len(text.split()) + 1), 0]

        def decode(self, ids):
            return " ".join(f"w{i}" for i in ids)

    monkeypatch.setattr(st, "_split_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(st, "_model_max_seq_length", lambda: 384)

    context = MagicMock()

    async def asset_to_bytes(ref):
        assert ref.uri == "file:///doc.txt"
        return b"one two three"

    context.asset_to_bytes = asset_to_bytes
    node = st.SplitSentences(
        document=DocumentRef(uri="file:///doc.txt"), chunk_size=40, chunk_overlap=0
    )

    async def run():
        return [item async for item in node.gen_process(context)]

    items = asyncio.run(run())
    assert [i["text"] for i in items] == ["w1 w2 w3"]
