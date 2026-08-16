"""Tests for LoRA adapter naming and unload-before-load behavior."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.stable_diffusion_base import (  # noqa: E402
    load_loras,
    lora_adapter_name,
    unload_loras,
)

ADAPTER_NAME_RE = r"^[A-Za-z0-9_]+$"


def test_adapter_name_is_module_safe():
    """Adapter names become torch module names, so only [A-Za-z0-9_] is allowed."""
    import re

    for path in [
        "My LoRA v1.2.safetensors",
        "sub/dir/add_detail.safetensors",
        "weird-name (copy)!.pt",
        "nö.safetensors",
    ]:
        assert re.match(ADAPTER_NAME_RE, lora_adapter_name(path))


def test_version_suffixes_do_not_collide():
    """Only the final extension is stripped, so v1.2 and v1.3 stay distinct."""
    a = lora_adapter_name("My LoRA v1.2.safetensors")
    b = lora_adapter_name("My LoRA v1.3.safetensors")
    assert a != b
    assert "1_2" in a
    assert "1_3" in b


def test_adapter_name_is_deterministic():
    path = "loras/My LoRA v1.2.safetensors"
    assert lora_adapter_name(path) == lora_adapter_name(path)


def test_sanitization_collisions_are_disambiguated_by_hash():
    """Different filenames that sanitize identically still map to different names."""
    assert lora_adapter_name("my lora.safetensors") != lora_adapter_name(
        "my-lora.safetensors"
    )


def test_adapter_name_handles_empty_path():
    import re

    assert re.match(ADAPTER_NAME_RE, lora_adapter_name(""))


class MockPipeline:
    """Minimal pipeline exposing the adapter API used by load_loras."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.loaded: list[str] = []
        self.active: list[str] = []

    def load_lora_weights(self, path, adapter_name=None):
        self.calls.append(("load", adapter_name))
        self.loaded.append(adapter_name)

    def unload_lora_weights(self):
        self.calls.append(("unload", None))
        self.loaded.clear()
        self.active.clear()

    def set_adapters(self, names, adapter_weights=None):
        self.calls.append(("set", tuple(names)))
        self.active = list(names)


class NoUnloadPipeline:
    def __init__(self):
        self.calls: list[tuple] = []

    def load_lora_weights(self, path, adapter_name=None):
        self.calls.append(("load", adapter_name))

    def set_adapters(self, names, adapter_weights=None):
        self.calls.append(("set", tuple(names)))


def make_lora(path: str, strength: float = 1.0):
    return SimpleNamespace(
        lora=SimpleNamespace(
            repo_id="repo/loras",
            path=path,
            is_set=lambda: True,
        ),
        strength=strength,
    )


@pytest.fixture
def fake_cache(monkeypatch):
    async def resolve(repo_id, path):
        return f"/cache/{repo_id}/{path}"

    monkeypatch.setattr(
        "nodetool.nodes.huggingface.stable_diffusion_base.HF_FAST_CACHE",
        SimpleNamespace(resolve=resolve),
    )


@pytest.mark.asyncio
async def test_unload_happens_before_load(fake_cache):
    pipeline = MockPipeline()
    await load_loras(pipeline, [make_lora("a.safetensors")])

    assert pipeline.calls[0] == ("unload", None)
    assert [c[0] for c in pipeline.calls] == ["unload", "load", "set"]


@pytest.mark.asyncio
async def test_repeated_loads_do_not_stack(fake_cache):
    pipeline = MockPipeline()
    await load_loras(pipeline, [make_lora("a.safetensors")])
    await load_loras(pipeline, [make_lora("b.safetensors")])

    assert pipeline.loaded == [lora_adapter_name("b.safetensors")]
    assert pipeline.active == [lora_adapter_name("b.safetensors")]


@pytest.mark.asyncio
async def test_empty_lora_list_still_unloads(fake_cache):
    pipeline = MockPipeline()
    await load_loras(pipeline, [make_lora("a.safetensors")])
    await load_loras(pipeline, [])

    assert pipeline.calls[-1] == ("unload", None)
    assert pipeline.loaded == []


@pytest.mark.asyncio
async def test_two_versions_load_as_distinct_adapters(fake_cache):
    pipeline = MockPipeline()
    await load_loras(
        pipeline,
        [make_lora("My LoRA v1.2.safetensors"), make_lora("My LoRA v1.3.safetensors")],
    )

    assert len(set(pipeline.loaded)) == 2
    assert pipeline.active == pipeline.loaded


@pytest.mark.asyncio
async def test_pipeline_without_unload_support_still_loads(fake_cache):
    pipeline = NoUnloadPipeline()
    await load_loras(pipeline, [make_lora("a.safetensors")])

    assert [c[0] for c in pipeline.calls] == ["load", "set"]


@pytest.mark.asyncio
async def test_identical_set_is_not_rebound(fake_cache):
    """Cached pipelines are reused across runs; re-binding the same set is wasted work."""
    pipeline = MockPipeline()
    loras = [make_lora("a.safetensors")]

    await load_loras(pipeline, loras)
    calls_after_first = list(pipeline.calls)
    await load_loras(pipeline, loras)

    assert pipeline.calls == calls_after_first
    assert pipeline.active == [lora_adapter_name("a.safetensors")]


@pytest.mark.asyncio
async def test_changed_strength_forces_rebind(fake_cache):
    """Strength is part of the bound state, so a change must re-apply the adapters."""
    pipeline = MockPipeline()

    await load_loras(pipeline, [make_lora("a.safetensors", strength=0.5)])
    await load_loras(pipeline, [make_lora("a.safetensors", strength=0.9)])

    assert [c[0] for c in pipeline.calls] == [
        "unload",
        "load",
        "set",
        "unload",
        "load",
        "set",
    ]


def test_adapter_cache_suffix_distinguishes_configurations():
    """The suffix keeps adapter-carrying pipelines in their own cache entries."""
    from nodetool.metadata.types import HFIPAdapter
    from nodetool.nodes.huggingface.stable_diffusion_base import adapter_cache_suffix

    plain = adapter_cache_suffix([], None)
    with_lora = adapter_cache_suffix([make_lora("a.safetensors")], None)
    other_lora = adapter_cache_suffix([make_lora("b.safetensors")], None)
    other_strength = adapter_cache_suffix(
        [make_lora("a.safetensors", strength=0.3)], None
    )
    with_ip = adapter_cache_suffix(
        [],
        HFIPAdapter(repo_id="h94/IP-Adapter", path="models/ip-adapter_sd15.bin"),
    )

    assert plain == ""
    assert len({with_lora, other_lora, other_strength, with_ip}) == 4


def test_adapter_cache_suffix_ignores_lora_order():
    """The bound set is what matters, not the order it was listed in."""
    from nodetool.nodes.huggingface.stable_diffusion_base import adapter_cache_suffix

    a, b = make_lora("a.safetensors"), make_lora("b.safetensors")
    assert adapter_cache_suffix([a, b], None) == adapter_cache_suffix([b, a], None)


def test_adapter_cache_suffix_skips_unset_loras():
    """An unset LoRA slot leaves the pipeline unmodified, so it must not split the cache."""
    from nodetool.nodes.huggingface.stable_diffusion_base import adapter_cache_suffix

    unset = SimpleNamespace(
        lora=SimpleNamespace(repo_id="", path="", is_set=lambda: False),
        strength=1.0,
    )
    assert adapter_cache_suffix([unset], None) == ""


def test_unload_loras_swallows_errors():
    class Failing:
        def unload_lora_weights(self):
            raise RuntimeError("boom")

    unload_loras(Failing())
    unload_loras(object())
