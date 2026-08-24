"""StableDiffusionControlNetInpaintNode must load its ControlNet as a diffusers model.

Its `controlnet` field defaults to `lllyasviel/control_v11p_sd15_inpaint`. That
repo is a diffusers ControlNet: its config has no `model_type` key, so routing
it through the transformers pipeline loader fails with

    Unrecognized model in lllyasviel/control_v11p_sd15_inpaint.
    Should have a `model_type` key in its config.json.

`fake_load_pipeline` below reproduces that refusal, so the node's own default
path fails until the ControlNet is loaded with `ControlNetModel`.

No network, no GPU, no weights: both loaders are stubbed.
"""

import asyncio
import sys
from pathlib import Path

import pytest

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.image_to_image import (
    StableDiffusionControlNetInpaintNode,
)
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode


class FakeContext:
    device = "cpu"


class FakePipeline:
    """Stands in for a loaded diffusers pipeline. Carries no scheduler, so
    `_set_scheduler` skips rather than touching real config."""


def run_preload(monkeypatch, node):
    """Run preload_model against stubbed loaders; return the recorded calls."""
    load_model_calls: list[dict] = []
    load_pipeline_calls: list[dict] = []

    async def fake_load_pipeline(self, context, pipeline_task, model_id, **kwargs):
        load_pipeline_calls.append(
            {"pipeline_task": pipeline_task, "model_id": model_id}
        )
        # What transformers' AutoConfig raises for a diffusers ControlNet repo.
        raise ValueError(
            f"Unrecognized model in {model_id}. "
            "Should have a `model_type` key in its config.json."
        )

    async def fake_load_model(self, context, model_class, model_id, **kwargs):
        load_model_calls.append({"model_class": model_class, "model_id": model_id})
        return FakePipeline()

    monkeypatch.setattr(HuggingFacePipelineNode, "load_pipeline", fake_load_pipeline)
    monkeypatch.setattr(HuggingFacePipelineNode, "load_model", fake_load_model)

    asyncio.run(node.preload_model(FakeContext()))  # type: ignore[arg-type]
    return load_model_calls, load_pipeline_calls


def test_default_controlnet_loads_without_the_transformers_pipeline(monkeypatch):
    from diffusers.models.controlnets.controlnet import ControlNetModel

    node = StableDiffusionControlNetInpaintNode()
    assert node.controlnet.value == "lllyasviel/control_v11p_sd15_inpaint"

    load_model_calls, load_pipeline_calls = run_preload(monkeypatch, node)

    assert load_pipeline_calls == [], (
        "a ControlNet repo has no `model_type`; it cannot go through the "
        "transformers pipeline loader"
    )

    controlnet_loads = [
        c
        for c in load_model_calls
        if c["model_id"] == "lllyasviel/control_v11p_sd15_inpaint"
    ]
    assert len(controlnet_loads) == 1
    assert controlnet_loads[0]["model_class"] is ControlNetModel


def test_preload_reaches_the_inpaint_pipeline(monkeypatch):
    from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
        StableDiffusionControlNetInpaintPipeline,
    )

    node = StableDiffusionControlNetInpaintNode()
    load_model_calls, _ = run_preload(monkeypatch, node)

    pipeline_loads = [
        c
        for c in load_model_calls
        if c["model_class"] is StableDiffusionControlNetInpaintPipeline
    ]
    assert len(pipeline_loads) == 1
    assert node._pipeline is not None
