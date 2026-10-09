"""CPU offload targets the selected GPU, and ``cuda:<n>`` counts as CUDA.

``move_to_device`` used ``device in ("cuda", "mps")``, so ``cuda:1`` with CPU
offload enabled did nothing, and diffusers offload always ran on ``cuda:0``.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from nodetool.huggingface import local_provider_utils
from nodetool.huggingface.memory_utils import (
    apply_cpu_offload_if_needed,
    offload_gpu_id,
)
from nodetool.nodes.huggingface.text_to_image import BriaFibo
from nodetool.workflows import torch_support


class FakePipeline:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def enable_model_cpu_offload(self, **kwargs):
        self.calls.append(("model", kwargs))

    def enable_sequential_cpu_offload(self, **kwargs):
        self.calls.append(("sequential", kwargs))


@pytest.mark.parametrize(
    "device,expected",
    [("cuda:1", 1), ("cuda:0", 0), ("cuda", None), ("mps", None), ("cpu", None)],
)
def test_offload_gpu_id(device, expected):
    assert offload_gpu_id(device) == expected


def test_offload_gpu_id_defaults_to_the_resolved_device(monkeypatch):
    monkeypatch.setattr(torch_support, "resolve_torch_device", lambda *a: "cuda:2")
    assert offload_gpu_id() == 2


@pytest.mark.parametrize("method", ["model", "sequential"])
def test_offload_runs_on_the_selected_gpu(method):
    pipeline = FakePipeline()
    assert apply_cpu_offload_if_needed(pipeline, method=method, device="cuda:1")
    assert pipeline.calls == [(method, {"gpu_id": 1})]


def test_offload_without_an_index_keeps_the_diffusers_default():
    pipeline = FakePipeline()
    apply_cpu_offload_if_needed(pipeline, method="model", device="cuda")
    assert pipeline.calls == [("model", {})]


@pytest.mark.asyncio
async def test_move_to_indexed_cuda_device_applies_offload():
    node = BriaFibo(enable_cpu_offload=True)
    node._pipeline = FakePipeline()
    await node.move_to_device("cuda:1")
    assert node._pipeline.calls == [("model", {"gpu_id": 1})]


def test_free_vram_reads_the_selected_gpu(monkeypatch):
    import torch

    seen = []

    def mem_get_info(index=None):
        seen.append(index)
        return (8 * 1024**3, 16 * 1024**3)

    monkeypatch.setattr(local_provider_utils, "_is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    assert local_provider_utils._cuda_free_vram_gb("cuda:1") == 8.0
    assert seen == [1]


def test_no_device_membership_checks_exclude_indexed_cuda():
    root = pathlib.Path(local_provider_utils.__file__).parents[1]
    sources = list(root.rglob("*.py"))
    assert sources
    pattern = re.compile(r"device\s+in\s+[\(\[]\s*\"cuda\"")
    offenders = [
        f"{p}:{n}"
        for p in sources
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == []
