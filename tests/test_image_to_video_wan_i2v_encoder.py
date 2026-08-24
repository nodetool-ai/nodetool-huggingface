"""Wan I2V must load the image encoder only when the model declares one.

Wan 2.2 (`Wan-AI/Wan2.2-I2V-A14B-Diffusers`, the node default) declares
`image_encoder: [None, None]` and ships no `image_encoder/` folder. Loading
that subfolder anyway fails with a misleading "does not appear to have a file
named pytorch_model.bin or model.safetensors".

No network, no GPU, no weights: the model_index.json read and every
`load_model` call are stubbed.
"""

import asyncio
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface import image_to_video as mod
from nodetool.nodes.huggingface.image_to_video import Wan_I2V
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode

WAN_2_2_MODEL_INDEX = {
    "_class_name": "WanImageToVideoPipeline",
    "image_encoder": [None, None],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}

WAN_2_1_MODEL_INDEX = {
    "_class_name": "WanImageToVideoPipeline",
    "image_encoder": ["transformers", "CLIPVisionModelWithProjection"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


def run_preload(monkeypatch, tmp_path, model_index, variant):
    """Run Wan_I2V.preload_model against a stubbed cache and loader.

    Returns the recorded `load_model` keyword bags, in call order.
    """
    index_path = tmp_path / "model_index.json"
    index_path.write_text(json.dumps(model_index))

    async def fake_resolve(repo_id, relpath, repo_type=None, **kwargs):
        return str(index_path) if relpath == "model_index.json" else None

    calls: list[dict] = []

    async def fake_load_model(self, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(mod.HF_FAST_CACHE, "resolve", fake_resolve)
    monkeypatch.setattr(HuggingFacePipelineNode, "load_model", fake_load_model)

    node = Wan_I2V(model_variant=variant, enable_cpu_offload=False)
    asyncio.run(node.preload_model(context=None))  # type: ignore[arg-type]
    return calls


def test_wan_2_2_skips_the_image_encoder(monkeypatch, tmp_path):
    calls = run_preload(
        monkeypatch,
        tmp_path,
        WAN_2_2_MODEL_INDEX,
        Wan_I2V.WanI2VModel.WAN_2_2_I2V_A14B,
    )

    encoder_loads = [c for c in calls if c.get("subfolder") == "image_encoder"]
    assert encoder_loads == [], "Wan 2.2 has no image_encoder subfolder to load"

    pipeline_calls = [c for c in calls if "subfolder" not in c]
    assert len(pipeline_calls) == 1
    assert "image_encoder" not in pipeline_calls[0]


def test_wan_2_1_loads_the_image_encoder(monkeypatch, tmp_path):
    calls = run_preload(
        monkeypatch,
        tmp_path,
        WAN_2_1_MODEL_INDEX,
        Wan_I2V.WanI2VModel.WAN_2_1_I2V_14B_720P,
    )

    encoder_loads = [c for c in calls if c.get("subfolder") == "image_encoder"]
    assert len(encoder_loads) == 1

    pipeline_calls = [c for c in calls if "subfolder" not in c]
    assert len(pipeline_calls) == 1
    assert pipeline_calls[0].get("image_encoder") is not None
