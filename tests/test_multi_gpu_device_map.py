"""Regression: vision-language models must not be sharded across GPUs.

`device_map="auto"` spreads a model over every visible GPU. Every inference
path here then moves its inputs to a single device (the model's
first-parameter device), so on a multi-GPU host the run dies. Reproduced on a
RunPod 4xA40 pod (transformers 5.15.1, torch/CUDA 12.8) through
`huggingface.image_text_to_text.Qwen2_5_VL` with
`Qwen/Qwen2.5-VL-7B-Instruct`:

    Expected all tensors to be on the same device, but got mat2 is on cuda:3,
    different from other tensors on cuda:0 (when checking argument in method
    wrapper_CUDA_bmm)

Isolated outside the node with the node's own load kwargs, same machine, back
to back: 4 GPUs visible -> parameters on cuda:0..3 -> failed; with
CUDA_VISIBLE_DEVICES=0 -> parameters on cuda:0 -> generated tokens. Only the
visible GPU count differs.

The three load sites now cap accelerate's memory budget at one GPU, which
keeps the model on one card and still lets accelerate offload the overflow to
CPU -- the case `"auto"` served on a single-GPU host.

The GPU topology is mocked: these tests need no CUDA device, no network, and
no model weights.
"""

import sys
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import accelerate.utils
import pytest
import torch

from nodetool.huggingface import huggingface_local_provider as provider_module
from nodetool.huggingface import local_provider_utils
from nodetool.huggingface.huggingface_local_provider import HuggingFaceLocalProvider
from nodetool.metadata.types import HFQwen2_5_VL, HFQwen3_VL, Message, MessageTextContent
from nodetool.nodes.huggingface.image_text_to_text import Qwen2_5_VL, Qwen3_VL
from nodetool.workflows.processing_context import ProcessingContext

# What a 4xA40 pod reports: free VRAM per card, plus host RAM.
FOUR_A40 = {0: 47_000_000_000, 1: 47_000_000_000, 2: 47_000_000_000, 3: 47_000_000_000}
HOST_RAM = 400_000_000_000


def _mock_gpus(monkeypatch, count: int, current: int = 0) -> None:
    """Present `count` CUDA devices to the placement code."""
    fake_torch = SimpleNamespace(
        device=torch.device,
        cuda=SimpleNamespace(
            device_count=lambda: count,
            current_device=lambda: current,
        ),
    )
    monkeypatch.setattr(local_provider_utils, "_get_torch", lambda: fake_torch)
    monkeypatch.setattr(local_provider_utils, "_is_cuda_available", lambda: True)
    budget: dict[Any, Any] = {i: FOUR_A40[i] for i in range(count)}
    budget["cpu"] = HOST_RAM
    monkeypatch.setattr(accelerate.utils, "get_max_memory", lambda *a, **k: budget)


def _pinned_index(load_kwargs: dict[str, Any]) -> int:
    """The single GPU this load is confined to."""
    assert "max_memory" in load_kwargs, (
        f"model loaded without a memory budget, so accelerate is free to shard "
        f"it across every visible GPU: {load_kwargs}"
    )
    gpu_indices = [key for key in load_kwargs["max_memory"] if isinstance(key, int)]
    assert len(gpu_indices) == 1, load_kwargs["max_memory"]
    return gpu_indices[0]


class _StubInputs(UserDict):
    def __init__(self) -> None:
        super().__init__({"input_ids": "ids"})

    def to(self, device: Any) -> "_StubInputs":
        return self


class _StubProcessor:
    def __init__(self) -> None:
        self.tokenizer = MagicMock()

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        return _StubInputs() if kwargs.get("tokenize") else "prompt"

    def __call__(self, **kwargs: Any) -> _StubInputs:
        return _StubInputs()


async def _provider_load_kwargs(monkeypatch, device: str) -> dict[str, Any]:
    """Run the provider's image-text-to-text stream; report the model's load kwargs."""
    calls: list[dict[str, Any]] = []
    model = MagicMock()
    model.device = "cuda:0"

    async def fake_load_model(**kwargs: Any):
        calls.append(kwargs)
        return _StubProcessor() if len(calls) == 1 else model

    monkeypatch.setattr(provider_module, "load_model", fake_load_model)

    provider = HuggingFaceLocalProvider()
    messages = [Message(role="user", content=[MessageTextContent(text="Hello.")])]
    try:
        async for _chunk in provider._stream_image_text_to_text(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=messages,
            max_tokens=8,
            context=ProcessingContext(device=device),
            node_id="node-1",
        ):
            pass
    except Exception:
        # The placement decision is made before any input is built; a later
        # failure in the stubbed generation path says nothing about it.
        pass

    assert len(calls) == 2, calls
    return calls[1]


async def _node_load_kwargs(monkeypatch, node: Any, device: str) -> dict[str, Any]:
    """Run a Qwen VL node's preload_model; report the model's load kwargs."""
    calls: list[dict[str, Any]] = []

    async def fake_load_model(self, context, model_class, model_id, **kwargs: Any):
        calls.append(kwargs)
        return MagicMock()

    # The nodes are pydantic models, so the double goes on the class.
    monkeypatch.setattr(type(node), "load_model", fake_load_model)
    await node.preload_model(ProcessingContext(device=device))

    assert len(calls) == 2, calls
    return calls[0]


@pytest.mark.asyncio
async def test_provider_pins_the_model_to_one_gpu_on_a_multi_gpu_host(monkeypatch):
    _mock_gpus(monkeypatch, count=4)
    load_kwargs = await _provider_load_kwargs(monkeypatch, "cuda:0")
    assert load_kwargs["device_map"] == "auto"
    assert _pinned_index(load_kwargs) == 0


@pytest.mark.asyncio
async def test_provider_pins_to_the_context_device(monkeypatch):
    """A context that picked cuda:2 must not be relocated to cuda:0."""
    _mock_gpus(monkeypatch, count=4)
    load_kwargs = await _provider_load_kwargs(monkeypatch, "cuda:2")
    assert _pinned_index(load_kwargs) == 2


@pytest.mark.asyncio
async def test_a_pinned_model_can_still_offload_to_cpu(monkeypatch):
    """The budget keeps CPU, so a model too large for one card still loads."""
    _mock_gpus(monkeypatch, count=4)
    load_kwargs = await _provider_load_kwargs(monkeypatch, "cuda:0")
    assert load_kwargs["max_memory"]["cpu"] == HOST_RAM


@pytest.mark.asyncio
async def test_single_gpu_placement_is_unchanged(monkeypatch):
    """One visible GPU keeps the plain device_map it always had."""
    _mock_gpus(monkeypatch, count=1)
    load_kwargs = await _provider_load_kwargs(monkeypatch, "cuda:0")
    assert load_kwargs["device_map"] == "auto"
    assert "max_memory" not in load_kwargs


@pytest.mark.asyncio
async def test_qwen2_5_vl_node_pins_the_model_to_one_gpu(monkeypatch):
    """The node from the hardware report: Qwen2_5_VL loads its own model."""
    _mock_gpus(monkeypatch, count=4)
    node = Qwen2_5_VL(model=HFQwen2_5_VL(repo_id="Qwen/Qwen2.5-VL-7B-Instruct"))
    load_kwargs = await _node_load_kwargs(monkeypatch, node, "cuda:0")
    assert load_kwargs["device_map"] == "auto"
    assert _pinned_index(load_kwargs) == 0


@pytest.mark.asyncio
async def test_qwen3_vl_node_pins_the_model_to_one_gpu(monkeypatch):
    _mock_gpus(monkeypatch, count=4)
    node = Qwen3_VL(model=HFQwen3_VL(repo_id="Qwen/Qwen3-VL-8B-Instruct"))
    load_kwargs = await _node_load_kwargs(monkeypatch, node, "cuda:0")
    assert load_kwargs["device_map"] == "auto"
    assert _pinned_index(load_kwargs) == 0


def test_no_load_site_still_hardcodes_device_map_auto():
    """The three sites are the whole population; a new one must opt in too."""
    package_root = Path(local_provider_utils.__file__).resolve().parents[1]
    offenders = [
        path
        for path in package_root.rglob("*.py")
        if 'device_map="auto"' in path.read_text()
        and path.name != "local_provider_utils.py"
    ]
    assert offenders == [], offenders
