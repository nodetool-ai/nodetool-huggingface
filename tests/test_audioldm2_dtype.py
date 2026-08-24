"""Regression: AudioLDM2 loaded its components in mixed dtypes, then failed.

`AudioLDM2Pipeline.from_pretrained` with no `torch_dtype` loads each component
in whatever dtype the repo stored it in. cvssp/audioldm2 ships its CLAP text
encoder as float64 while everything else is float32, so the encoder's output
reached the float32 projection model and torch raised:

    RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float

Read off a real A40 worker (diffusers 0.40.0), before the fix:

    vae                torch.float32
    unet               torch.float32
    text_encoder       torch.float64   <- the odd one out
    text_encoder_2     torch.float32
    projection_model   torch.float32
    vocoder            torch.float32

With torch_dtype=torch.float32 every component loads float32 and that error is
gone. A second, unrelated failure then surfaces from upstream, which
`_assert_audioldm2_language_model_can_generate` reports in plain words rather
than as a bare AttributeError from inside diffusers.

AudioLDM (v1) does not share either problem: its components are all float32 and
it runs.
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from nodetool.nodes.huggingface.text_to_audio import (
    AudioLDM2,
    _assert_audioldm2_language_model_can_generate,
)


class _Generates:
    """A language model that still carries the generation helper."""

    def _update_model_kwargs_for_generation(self, *args, **kwargs):
        return {}


class _DoesNot:
    """transformers 5's GPT2Model: a base model, no generation methods."""


class _Pipeline:
    def __init__(self, language_model) -> None:
        self.language_model = language_model


@pytest.mark.asyncio
async def test_preload_pins_float32(monkeypatch):
    """The whole bug: no torch_dtype meant a float64 text encoder."""
    captured: dict = {}

    async def fake_load_model(self, context, model_class, model_id, **kwargs):
        captured.update(kwargs)
        captured["model_id"] = model_id
        return _Pipeline(_Generates())

    monkeypatch.setattr(AudioLDM2, "load_model", fake_load_model, raising=False)

    node = AudioLDM2()
    await node.preload_model(context=None)

    import torch

    assert captured["model_id"] == "cvssp/audioldm2"
    assert captured.get("torch_dtype") is torch.float32, (
        "components must be pinned to one dtype, or the CLAP encoder loads float64"
    )


def test_a_generation_capable_language_model_is_accepted():
    _assert_audioldm2_language_model_can_generate(_Pipeline(_Generates()))


def test_a_base_language_model_is_reported_in_plain_words():
    """transformers 5 removed generation methods from base models."""
    with pytest.raises(RuntimeError) as excinfo:
        _assert_audioldm2_language_model_can_generate(_Pipeline(_DoesNot()))

    message = str(excinfo.value)
    assert "transformers" in message
    assert "upstream" in message
    # Name the method, so the reader can match it against the diffusers source.
    assert "_update_model_kwargs_for_generation" in message


def test_no_language_model_is_not_an_error():
    _assert_audioldm2_language_model_can_generate(_Pipeline(None))
