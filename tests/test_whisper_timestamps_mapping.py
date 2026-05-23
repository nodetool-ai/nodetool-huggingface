"""Unit tests for the Whisper node's return_timestamps mapping.

The HF AutomaticSpeechRecognitionPipeline only accepts:
    - return_timestamps=True   -> segment ("sentence") timestamps
    - return_timestamps="word" -> word-level timestamps
    - return_timestamps=None/False -> no timestamps

Passing the literal string "sentence" sneaks past the validator but
produces no usable chunks downstream.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT.parent / "nodetool-core" / "src"
HF_SRC = ROOT / "src"
for p in (str(CORE_SRC), str(HF_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from nodetool.nodes.huggingface.automatic_speech_recognition import (  # noqa: E402
    Timestamps,
    Whisper,
)


def _return_timestamps(node: Whisper):
    return node._build_pipeline_kwargs()["return_timestamps"]


def test_none_maps_to_none():
    node = Whisper(timestamps=Timestamps.NONE)
    assert _return_timestamps(node) is None


def test_word_maps_to_word_string():
    node = Whisper(timestamps=Timestamps.WORD)
    assert _return_timestamps(node) == "word"


def test_sentence_maps_to_true_not_string():
    """The HF pipeline expects True for segment-level timestamps;
    passing 'sentence' silently yields no chunks."""
    node = Whisper(timestamps=Timestamps.SENTENCE)
    assert _return_timestamps(node) is True
