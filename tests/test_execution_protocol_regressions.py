import asyncio
import logging
from types import SimpleNamespace

import pytest

from nodetool.nodes.huggingface.huggingface_node import (
    HuggingFaceLogHandler,
    hf_log_route,
)
from nodetool.nodes.huggingface.speaker_diarization import SpeakerDiarization
from nodetool.nodes.huggingface.stable_diffusion_base import (
    StableDiffusionBaseNode,
    StableDiffusionXLBase,
)


def test_diarization_declares_forwarded_token():
    assert "HF_TOKEN" in SpeakerDiarization.required_settings()


@pytest.mark.asyncio
async def test_logging_isolated_between_tasks_and_threads():
    handler = HuggingFaceLogHandler()
    received = {"A": [], "B": []}

    async def execute(name):
        context = SimpleNamespace(post_message=received[name].append)
        token = hf_log_route.set((context, name, name))
        try:
            await asyncio.sleep(0)
            record = logging.LogRecord("transformers", logging.INFO, "", 0, name, (), None)
            await asyncio.to_thread(handler.emit, record)
        finally:
            hf_log_route.reset(token)

    await asyncio.gather(execute("A"), execute("B"))
    for name, messages in received.items():
        assert len(messages) == 1
        assert messages[0].node_id == name
        assert messages[0].content == name
    assert not hasattr(handler, "context")


@pytest.mark.parametrize("xl", [False, True])
def test_diffusion_callback_stops_after_cancellation(xl):
    messages = []
    cancelled = False

    def check():
        if cancelled:
            raise RuntimeError("cancelled")

    context = SimpleNamespace(raise_if_cancelled=check, post_message=messages.append)
    node = SimpleNamespace(id="test", num_inference_steps=3)
    if xl:
        callback = StableDiffusionXLBase.progress_callback(node, context)
    else:
        callback = StableDiffusionBaseNode.progress_callback(node, context, 0, 3)
    payload = {"latents": object()}
    assert callback(None, 0, 0, payload) is payload
    cancelled = True
    with pytest.raises(RuntimeError, match="cancelled"):
        callback(None, 1, 0, payload)
    assert len(messages) == 1
