"""Provider pipeline loads pick their dtype with the same rule as the nodes.

The provider used ``bfloat16 if cuda else float32``: float32 on Apple Silicon
(about 48 GB for the FLUX transformer) and bf16 on CUDA cards that emulate or
reject it. The nodes use fp16 on MPS, bf16 only where CUDA supports it, and
fp32 on CPU. The dtype follows the device the pipeline runs on, so a CPU run
on a GPU machine loads float32.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from nodetool.huggingface import local_provider_utils
from nodetool.huggingface.local_provider_utils import _select_pipeline_dtype

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_MODULES = [
    ROOT / "src/nodetool/huggingface/text_to_image_pipelines.py",
    ROOT / "src/nodetool/huggingface/image_to_image_pipelines.py",
]


@pytest.mark.parametrize(
    "device,bf16,prefer_bf16,expected",
    [
        ("cuda", True, True, torch.bfloat16),
        ("cuda", False, True, torch.float16),  # pre-Ampere
        ("cuda:1", True, False, torch.float16),
        ("mps", False, True, torch.float16),  # Apple Silicon
        ("mps", False, False, torch.float16),
        ("cpu", True, True, torch.float32),  # CPU, even with a GPU present
    ],
)
def test_select_pipeline_dtype(monkeypatch, device, bf16, prefer_bf16, expected):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)
    assert _select_pipeline_dtype(prefer_bf16=prefer_bf16, device=device) is expected


def test_select_pipeline_dtype_defaults_to_resolved_device(monkeypatch):
    monkeypatch.setattr(
        local_provider_utils, "resolve_torch_device", lambda *a: "cpu"
    )
    assert _select_pipeline_dtype(prefer_bf16=True) is torch.float32


@pytest.mark.parametrize("path", PIPELINE_MODULES, ids=lambda p: p.name)
def test_no_cuda_or_float32_dtype_choice_left(path):
    source = path.read_text(encoding="utf-8")
    assert "_select_pipeline_dtype(" in source
    leftover = re.findall(r"if\s+_is_cuda_available\(\)\s+else\s+\S*float32", source)
    assert not leftover, f"{path.name} still picks float32 for every non-CUDA device"
