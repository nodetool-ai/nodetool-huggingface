"""Provider pipeline loads pick their dtype with the same rule as the nodes.

The provider used ``bfloat16 if cuda else float32``: float32 on Apple Silicon
(about 48 GB for the FLUX transformer) and bf16 on CUDA cards that emulate or
reject it. The nodes use fp16 on MPS, bf16 only where CUDA supports it, and
fp32 on CPU.
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


def _hardware(monkeypatch, *, cuda: bool, mps: bool, bf16: bool = False) -> None:
    monkeypatch.setattr(local_provider_utils, "_is_cuda_available", lambda: cuda)
    monkeypatch.setattr(local_provider_utils, "_is_mps_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)


@pytest.mark.parametrize(
    "cuda,mps,bf16,prefer_bf16,expected",
    [
        (True, False, True, True, torch.bfloat16),
        (True, False, False, True, torch.float16),  # pre-Ampere
        (True, False, True, False, torch.float16),
        (False, True, False, True, torch.float16),  # Apple Silicon
        (False, True, False, False, torch.float16),
        (False, False, False, True, torch.float32),  # CPU
    ],
)
def test_select_pipeline_dtype(monkeypatch, cuda, mps, bf16, prefer_bf16, expected):
    _hardware(monkeypatch, cuda=cuda, mps=mps, bf16=bf16)
    assert _select_pipeline_dtype(prefer_bf16=prefer_bf16) is expected


@pytest.mark.parametrize("path", PIPELINE_MODULES, ids=lambda p: p.name)
def test_no_cuda_or_float32_dtype_choice_left(path):
    source = path.read_text(encoding="utf-8")
    assert "_select_pipeline_dtype(" in source
    leftover = re.findall(r"if\s+_is_cuda_available\(\)\s+else\s+\S*float32", source)
    assert not leftover, f"{path.name} still picks float32 for every non-CUDA device"
