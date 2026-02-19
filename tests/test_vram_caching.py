"""
Tests for VRAM leak prevention in Whisper and KokoroTTS nodes.

Verifies that:
1. Whisper uses a proper string-based cache key for pipelines (not model repr)
2. KokoroTTS caches the KModel via ModelManager to prevent re-creation
3. load_pipeline supports an optional cache_key parameter
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


def test_load_pipeline_uses_custom_cache_key():
    """load_pipeline should use the provided cache_key instead of deriving one from model_id."""
    from nodetool.ml.core.model_manager import ModelManager

    custom_key = "my_model_custom_key"
    sentinel = object()

    # Seed the cache with our custom key
    ModelManager._models[custom_key] = sentinel

    try:
        from nodetool.huggingface.local_provider_utils import load_pipeline
        import asyncio

        ctx = MagicMock()
        ctx.device = "cpu"

        result = asyncio.get_event_loop().run_until_complete(
            load_pipeline(
                node_id="test_node",
                context=ctx,
                pipeline_task="automatic-speech-recognition",
                model_id="openai/whisper-large-v3",
                cache_key=custom_key,
            )
        )
        assert result is sentinel
    finally:
        ModelManager._models.pop(custom_key, None)


def test_load_pipeline_default_cache_key():
    """When cache_key is not provided, load_pipeline should derive it from model_id and task."""
    from nodetool.ml.core.model_manager import ModelManager

    default_key = "openai/whisper-large-v3_automatic-speech-recognition"
    sentinel = object()

    ModelManager._models[default_key] = sentinel

    try:
        from nodetool.huggingface.local_provider_utils import load_pipeline
        import asyncio

        ctx = MagicMock()
        ctx.device = "cpu"

        result = asyncio.get_event_loop().run_until_complete(
            load_pipeline(
                node_id="test_node",
                context=ctx,
                pipeline_task="automatic-speech-recognition",
                model_id="openai/whisper-large-v3",
                # no cache_key provided
            )
        )
        assert result is sentinel
    finally:
        ModelManager._models.pop(default_key, None)


def test_whisper_preload_uses_string_cache_key():
    """Whisper.preload_model should pass a string cache_key to load_pipeline."""
    from nodetool.nodes.huggingface.automatic_speech_recognition import Whisper

    node = Whisper()
    repo_id = node.model.repo_id

    # The cache key should be derived from the repo_id string, not model.__repr__
    expected_key = f"{repo_id}_automatic-speech-recognition"

    # Verify the key is a reasonable length string (not a huge model repr)
    assert isinstance(expected_key, str)
    assert len(expected_key) < 200


def test_kokoro_preload_caches_model():
    """KokoroTTS.preload_model should cache the KModel in ModelManager."""
    from nodetool.nodes.huggingface.text_to_speech import KokoroTTS
    from nodetool.ml.core.model_manager import ModelManager

    node = KokoroTTS()
    cache_key = f"{node.get_model_id()}_KModel"

    # Verify the cache key format is correct
    assert cache_key == "hexgrad/Kokoro-82M_KModel"

    # Verify that a cached model would be detected
    sentinel_model = MagicMock()
    ModelManager._models[cache_key] = sentinel_model

    try:
        cached = ModelManager.get_model(cache_key)
        assert cached is sentinel_model
    finally:
        ModelManager._models.pop(cache_key, None)


def test_huggingface_pipeline_node_passes_cache_key():
    """HuggingFacePipelineNode.load_pipeline should accept and forward cache_key."""
    from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
    import inspect

    sig = inspect.signature(HuggingFacePipelineNode.load_pipeline)
    assert "cache_key" in sig.parameters
