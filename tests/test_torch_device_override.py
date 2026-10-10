"""HF device and dtype choices follow ``NODETOOL_TORCH_DEVICE`` and the context.

``_resolve_hf_device`` used to return ``mps`` whenever Apple Metal existed, the
3D nodes picked ``cuda`` whenever CUDA existed, and dtypes followed the
hardware. A CPU override then still loaded on the GPU, in half precision.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nodetool.huggingface.local_provider_utils import _resolve_hf_device
from nodetool.nodes.huggingface._3d_common import _resolve_device
from nodetool.nodes.huggingface.huggingface_pipeline import select_inference_dtype
from nodetool.nodes.huggingface.stable_diffusion_base import available_torch_dtype
from nodetool.workflows import torch_support


@pytest.fixture
def gpus_present(monkeypatch):
    """Pretend MPS and two CUDA devices exist."""
    monkeypatch.setattr(torch_support, "_is_mps_available", lambda: True)
    monkeypatch.setattr(torch_support, "is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)


def test_hf_device_keeps_the_requested_device_when_mps_exists(gpus_present):
    context = SimpleNamespace(device="mps")
    assert _resolve_hf_device(context, "cpu") == "cpu"
    assert _resolve_hf_device(context, "cuda:1") == "cuda:1"


def test_hf_device_uses_the_context_device(gpus_present):
    assert _resolve_hf_device(SimpleNamespace(device="cpu")) == "cpu"


def test_hf_device_falls_back_to_the_override(gpus_present, monkeypatch):
    monkeypatch.setenv(torch_support.TORCH_DEVICE_ENV, "cpu")
    assert _resolve_hf_device(SimpleNamespace(device=None)) == "cpu"


def test_3d_device_follows_context_and_override(gpus_present, monkeypatch):
    assert _resolve_device(SimpleNamespace(device="cuda:1")) == "cuda:1"
    monkeypatch.setenv(torch_support.TORCH_DEVICE_ENV, "cpu")
    assert _resolve_device() == "cpu"


@pytest.mark.parametrize("select", [select_inference_dtype, available_torch_dtype])
@pytest.mark.parametrize(
    "device,expected",
    [("cpu", torch.float32), ("mps", torch.float16), ("cuda:1", torch.float16)],
)
def test_node_dtype_follows_the_device(monkeypatch, select, device, expected):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    assert select(device) is expected


def test_node_dtype_defaults_to_the_override(gpus_present, monkeypatch):
    monkeypatch.setenv(torch_support.TORCH_DEVICE_ENV, "cpu")
    assert select_inference_dtype() is torch.float32
    assert available_torch_dtype() is torch.float32
