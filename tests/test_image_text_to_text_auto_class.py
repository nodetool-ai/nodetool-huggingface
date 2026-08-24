"""Regression: the local ImageTextToText path must load and call a
transformers 5 vision-language model the way transformers 5 expects.

Two defects, both reproduced on a RunPod A40 with transformers 5.15.1 and
`HuggingFaceTB/SmolVLM-Instruct` (model_type `idefics3`).

1. The model loaded with a hardcoded `AutoModelForCausalLM`:

       Unrecognized configuration class
       <class 'transformers.models.idefics3.configuration_idefics3.Idefics3Config'>
       for this kind of AutoModel: AutoModelForCausalLM.

   `idefics3` and `qwen3_vl` are absent from MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
   and present in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, so this is not
   SmolVLM-specific. A few multimodal checkpoints go the other way --
   `microsoft/Phi-4-multimodal-instruct` (model_type `phi4_multimodal`) is
   causal-LM only -- so the class is chosen per checkpoint.

2. The inputs were built by passing `images=` into `apply_chat_template`. In
   transformers 5 that method collects the conversation's images from the
   content blocks itself and forwards every other keyword to the processor
   call it makes, so `images` arrives twice and the call raises TypeError.
   The images here were lifted out of the content into a separate list, so
   they have to reach the processor through `processor(text=..., images=...)`
   instead.

Nothing here touches the network, a GPU, or model weights: the config lookup,
the processor, and the model are all stubbed.
"""

import sys
from collections import UserDict
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import PIL.Image
import pytest
import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from nodetool.huggingface import huggingface_local_provider as provider_module
from nodetool.huggingface.huggingface_local_provider import HuggingFaceLocalProvider
from nodetool.metadata.types import (
    ImageRef,
    Message,
    MessageImageContent,
    MessageTextContent,
)
from nodetool.workflows.processing_context import ProcessingContext


class _StubInputs(UserDict):
    """Stands in for the BatchFeature a processor returns."""

    def __init__(self) -> None:
        super().__init__({"input_ids": "ids", "pixel_values": "pixels"})
        self.moved_to: Any = None

    def to(self, device: Any) -> "_StubInputs":
        self.moved_to = device
        return self


class _StubProcessor:
    """A processor with the transformers 5 call contract.

    `ProcessorMixin.apply_chat_template` forwards a keyword it does not
    recognize to the processor call it makes internally -- and that call
    already passes `images`, gathered from the conversation's own content
    blocks. An `images=` keyword therefore arrives twice and Python raises
    TypeError, which is what an Idefics3Processor does on hardware.
    """

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.tokenizer = MagicMock()

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
        self.record["template_kwargs"] = kwargs
        if "images" in kwargs:
            raise TypeError(
                f"{type(self).__name__}.__call__() got multiple values for "
                "keyword argument 'images'"
            )
        return "<|im_start|>user<image>Describe this.<|im_end|>"

    def __call__(self, **kwargs: Any) -> _StubInputs:
        self.record["call_kwargs"] = kwargs
        return _StubInputs()


def _image_in_workspace(workspace_dir: str) -> ImageRef:
    path = Path(workspace_dir) / "input_image.png"
    buffer = BytesIO()
    PIL.Image.new("RGB", (4, 4), color="red").save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    return ImageRef(uri="file:///input_image.png")


async def _run_stream(
    monkeypatch, tmp_path, repo_id: str, model_type: str
) -> dict[str, Any]:
    """Run the image-text-to-text stream against stubs and report what it did."""
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: MagicMock(model_type=model_type),
    )

    record: dict[str, Any] = {"model_classes": [], "error": None}
    processor = _StubProcessor(record)
    model = MagicMock()
    model.device = "cpu"

    async def fake_load_model(*, model_class: type, **kwargs: Any):
        record["model_classes"].append(model_class)
        return processor if len(record["model_classes"]) == 1 else model

    monkeypatch.setattr(provider_module, "load_model", fake_load_model)

    context = ProcessingContext(workspace_dir=str(tmp_path))
    messages = [
        Message(
            role="user",
            content=[
                MessageImageContent(image=_image_in_workspace(str(tmp_path))),
                MessageTextContent(text="Describe this."),
            ],
        )
    ]

    provider = HuggingFaceLocalProvider()
    try:
        async for _chunk in provider._stream_image_text_to_text(
            repo_id=repo_id,
            messages=messages,
            max_tokens=8,
            context=context,
            node_id="node-1",
        ):
            pass
    except Exception as exc:  # reported as a test failure, not swallowed
        record["error"] = exc

    return record


@pytest.mark.asyncio
async def test_image_text_to_text_config_loads_with_image_text_to_text_class(
    monkeypatch, tmp_path
):
    """An idefics3-style checkpoint must not go to AutoModelForCausalLM."""
    record = await _run_stream(
        monkeypatch, tmp_path, "HuggingFaceTB/SmolVLM-Instruct", "idefics3"
    )
    # The processor loads first, the model second.
    assert len(record["model_classes"]) == 2, record["model_classes"]
    assert record["model_classes"][1] is AutoModelForImageTextToText


@pytest.mark.asyncio
async def test_causal_lm_only_config_keeps_the_causal_lm_class(monkeypatch, tmp_path):
    """phi4_multimodal is absent from the image-text-to-text mapping; it must
    keep the class that can load it."""
    record = await _run_stream(
        monkeypatch, tmp_path, "microsoft/Phi-4-multimodal-instruct", "phi4_multimodal"
    )
    assert len(record["model_classes"]) == 2, record["model_classes"]
    assert record["model_classes"][1] is AutoModelForCausalLM


@pytest.mark.asyncio
async def test_images_reach_the_processor_call_not_the_chat_template(
    monkeypatch, tmp_path
):
    """The input building must survive a transformers 5 processor, and the
    images must arrive through the processor call."""
    record = await _run_stream(
        monkeypatch, tmp_path, "HuggingFaceTB/SmolVLM-Instruct", "idefics3"
    )

    assert record["error"] is None, f"input building failed: {record['error']!r}"
    assert "images" not in record["template_kwargs"], record["template_kwargs"]

    call_kwargs = record["call_kwargs"]
    assert isinstance(call_kwargs["text"], str)
    assert call_kwargs["return_tensors"] == "pt"
    images = call_kwargs["images"]
    assert len(images) == 1
    assert isinstance(images[0], PIL.Image.Image)


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


def test_every_recommended_family_accepts_the_processor_call_shape():
    """Every vision-language processor in transformers 5 takes (text, images);
    the fix does not need a per-family convention."""
    import inspect
    import importlib

    processors = {
        "idefics3": "transformers.models.idefics3.processing_idefics3.Idefics3Processor",
        "smolvlm": "transformers.models.smolvlm.processing_smolvlm.SmolVLMProcessor",
        "qwen2_5_vl": "transformers.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor",
        "qwen3_vl": "transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor",
        "llava": "transformers.models.llava.processing_llava.LlavaProcessor",
        "llava_next": "transformers.models.llava_next.processing_llava_next.LlavaNextProcessor",
        "gemma3": "transformers.models.gemma3.processing_gemma3.Gemma3Processor",
        "internvl": "transformers.models.internvl.processing_internvl.InternVLProcessor",
        "florence2": "transformers.models.florence2.processing_florence2.Florence2Processor",
        "glm4v": "transformers.models.glm4v.processing_glm4v.Glm4vProcessor",
    }

    for family, path in processors.items():
        module_name, class_name = path.rsplit(".", 1)
        processor_class = getattr(importlib.import_module(module_name), class_name)
        params = inspect.signature(processor_class.__call__).parameters
        assert "text" in params, family
        assert "images" in params, family
