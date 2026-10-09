"""Regression: fp16 Qwen-Image was left running on the CPU.

`QwenImage.move_to_device` was a blanket `pass`. The pipeline deliberately
loads with `device="cpu"` and relies on this call to place it, so a 20B model
ran on the CPU while the GPU sat idle.
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from nodetool.nodes.huggingface import text_to_image
from nodetool.nodes.huggingface.text_to_image import QwenImage


class _FakePipeline:
    """Stands in for a diffusers pipeline; records where it was asked to go."""

    def __init__(self) -> None:
        self.moved_to: list[str] = []


@pytest.fixture
def record_moves(monkeypatch):
    moves: list[tuple[object, str]] = []
    monkeypatch.setattr(
        text_to_image,
        "move_pipeline_to_device",
        lambda pipeline, device: moves.append((pipeline, device)),
    )
    return moves


@pytest.mark.asyncio
async def test_fp16_pipeline_is_moved_to_the_device(record_moves):
    """The whole bug: this list stayed empty, so the model never left the CPU."""
    node = QwenImage()
    node._pipeline = _FakePipeline()

    await node.move_to_device("cuda")

    assert [device for _, device in record_moves] == ["cuda"]


@pytest.mark.asyncio
async def test_no_pipeline_is_not_an_error(record_moves):
    node = QwenImage()
    node._pipeline = None

    await node.move_to_device("cuda")

    assert record_moves == []
