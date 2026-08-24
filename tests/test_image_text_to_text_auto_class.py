"""Regression: vision-language models must load with the auto class that
transformers registers for them.

`_stream_image_text_to_text` hardcoded `AutoModelForCausalLM`. In transformers
5 that mapping does not contain the vision-language configs, so
`HuggingFaceTB/SmolVLM-Instruct` (model_type `idefics3`) failed on an A40 with:

    Unrecognized configuration class
    <class 'transformers.models.idefics3.configuration_idefics3.Idefics3Config'>
    for this kind of AutoModel: AutoModelForCausalLM.

`idefics3` and `qwen3_vl` are absent from
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES and present in
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, so the failure is not
SmolVLM-specific. A few multimodal checkpoints this node recommends go the
other way -- `microsoft/Phi-4-multimodal-instruct` (model_type
`phi4_multimodal`) is causal-LM only -- so the class is chosen per checkpoint
rather than swapped outright.

Nothing here touches the network, a GPU, or model weights: the config lookup
and the model load are both stubbed.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from nodetool.huggingface import huggingface_local_provider as provider_module
from nodetool.huggingface.huggingface_local_provider import HuggingFaceLocalProvider
from nodetool.metadata.types import Message, MessageTextContent
from nodetool.workflows.processing_context import ProcessingContext


async def _model_class_used(monkeypatch, repo_id: str, model_type: str) -> type:
    """Run the image-text-to-text stream and report the auto class it loaded with."""
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: MagicMock(model_type=model_type),
    )

    loaded: list[type] = []

    async def fake_load_model(*, model_class: type, **kwargs: Any):
        loaded.append(model_class)
        return MagicMock()

    monkeypatch.setattr(provider_module, "load_model", fake_load_model)

    provider = HuggingFaceLocalProvider()
    messages = [Message(role="user", content=[MessageTextContent(text="What is this?")])]

    async for _chunk in provider._stream_image_text_to_text(
        repo_id=repo_id,
        messages=messages,
        max_tokens=8,
        context=ProcessingContext(),
        node_id="node-1",
    ):
        pass

    # The processor loads first, the model second.
    assert len(loaded) == 2, loaded
    return loaded[1]


@pytest.mark.asyncio
async def test_image_text_to_text_config_loads_with_image_text_to_text_class(monkeypatch):
    """An idefics3-style checkpoint must not go to AutoModelForCausalLM."""
    model_class = await _model_class_used(
        monkeypatch, "HuggingFaceTB/SmolVLM-Instruct", "idefics3"
    )
    assert model_class is AutoModelForImageTextToText


@pytest.mark.asyncio
async def test_causal_lm_only_config_keeps_the_causal_lm_class(monkeypatch):
    """phi4_multimodal is absent from the image-text-to-text mapping; it must
    keep the class that can load it."""
    model_class = await _model_class_used(
        monkeypatch, "microsoft/Phi-4-multimodal-instruct", "phi4_multimodal"
    )
    assert model_class is AutoModelForCausalLM


def test_mapping_membership_matches_what_the_fix_assumes():
    """Pin the transformers mappings the selection rule reads."""
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )

    for model_type in ("idefics3", "qwen3_vl"):
        assert model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        assert model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    assert "phi4_multimodal" in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert "phi4_multimodal" not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
