"""Regression: MusicLDM on transformers 5 failed with an error nobody can act on.

``MusicLDMPipeline.encode_prompt`` (diffusers 0.40.0) does::

    prompt_embeds = self.text_encoder.get_text_features(...)
    prompt_embeds = prompt_embeds.to(dtype=..., device=device)

transformers 4 returned a tensor from ``ClapModel.get_text_features``.
transformers 5 returns the whole ``BaseModelOutputWithPooling`` -- the
embeddings moved into ``pooler_output`` -- and a ``ModelOutput`` has no
``.to``. On an A40 with diffusers 0.40.0, transformers 5.15.1 and
ucsd-reach/musicldm the run died with::

    'BaseModelOutputWithPooling' object has no attribute 'to'

after the model had downloaded and loaded. The preflight below says what is
wrong, and says it before the download.

AudioLDM (v1) shares MusicLDM's ancestry but not this bug: its pipeline calls
``self.text_encoder(...)`` and reads ``.text_embeds`` off the result, and
transformers 5's ``ClapTextModelWithProjection`` still returns a
``ClapTextModelOutput`` carrying ``text_embeds``. It was run on the same A40,
same session, same versions, with cvssp/audioldm-s-full-v2 and finished in 9.5s.
It gets no check here: one that can never fire implies coverage that is not
there.
"""

import sys
import types
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from nodetool.nodes.huggingface.text_to_audio import MusicLDM


def _encode_prompt_transformers_4_only(self, prompt, device, *args, **kwargs):
    """diffusers 0.40.0's MusicLDM: the return value is used as a tensor."""
    prompt_embeds = self.text_encoder.get_text_features(prompt)
    prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.text_model.dtype)
    return prompt_embeds


def _encode_prompt_with_pooled_unwrap(self, prompt, device, *args, **kwargs):
    """What an upstream fix looks like -- the shape diffusers' AudioLDM2 uses."""
    prompt_embeds = self.text_encoder.get_text_features(prompt)
    if hasattr(prompt_embeds, "pooler_output"):
        prompt_embeds = prompt_embeds.pooler_output
    return prompt_embeds.to(dtype=self.text_encoder.text_model.dtype)


def _fake_environment(monkeypatch, *, transformers_version, encode_prompt):
    transformers = types.ModuleType("transformers")
    transformers.__version__ = transformers_version

    class _MusicLDMPipeline:
        pass

    _MusicLDMPipeline.encode_prompt = encode_prompt

    diffusers = types.ModuleType("diffusers")
    diffusers.MusicLDMPipeline = _MusicLDMPipeline

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)


def _record_loads(monkeypatch):
    loads: list[str] = []

    async def fake_load_model(self, context, model_class, model_id, **kwargs):
        loads.append(model_id)
        return object()

    monkeypatch.setattr(MusicLDM, "load_model", fake_load_model, raising=False)
    return loads


@pytest.mark.asyncio
async def test_transformers_5_is_reported_in_plain_words(monkeypatch):
    _fake_environment(
        monkeypatch,
        transformers_version="5.15.1",
        encode_prompt=_encode_prompt_transformers_4_only,
    )
    loads = _record_loads(monkeypatch)

    node = MusicLDM()
    with pytest.raises(RuntimeError) as excinfo:
        await node.preload_model(context=None)

    message = str(excinfo.value)
    assert "MusicLDM" in message
    assert "5.15.1" in message
    assert "upstream" in message
    # Name the call and the type, so the reader can match both against the
    # diffusers and transformers sources.
    assert "get_text_features" in message
    assert "BaseModelOutputWithPooling" in message
    assert loads == [], "the check must refuse before the model downloads"


@pytest.mark.asyncio
async def test_transformers_4_still_loads(monkeypatch):
    _fake_environment(
        monkeypatch,
        transformers_version="4.56.2",
        encode_prompt=_encode_prompt_transformers_4_only,
    )
    loads = _record_loads(monkeypatch)

    node = MusicLDM()
    node.model.repo_id = "ucsd-reach/musicldm"
    await node.preload_model(context=None)

    assert loads == ["ucsd-reach/musicldm"]


@pytest.mark.asyncio
async def test_an_upstream_fix_clears_the_check(monkeypatch):
    """A diffusers pipeline that unwraps pooler_output must not be blocked."""
    _fake_environment(
        monkeypatch,
        transformers_version="5.15.1",
        encode_prompt=_encode_prompt_with_pooled_unwrap,
    )
    loads = _record_loads(monkeypatch)

    node = MusicLDM()
    node.model.repo_id = "ucsd-reach/musicldm"
    await node.preload_model(context=None)

    assert loads == ["ucsd-reach/musicldm"]
