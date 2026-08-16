"""
Tests for the VRAM-leak guards around CPU offload and cache keys.

Covers the three failure modes these guards exist to prevent:

1. Model-level CPU offload going undetected, so a second (or stacked) offload
   strategy tears down the first one's hooks.
2. ``pipeline.to("cuda")`` yanking an offloaded pipeline fully onto a GPU that
   was already judged too small for its weights.
3. Pipelines that differ only by a swapped-in component or by bound adapters
   sharing one cache entry, which strands the unused weights on the GPU.
"""

import sys
import types

import pytest


class _FakeCpuOffloadHook:
    pass


class _FakeAlignDevicesHook:
    pass


@pytest.fixture(autouse=True)
def fake_accelerate_hooks(monkeypatch):
    """Stand in for ``accelerate.hooks`` so the tests run without accelerate."""
    module = types.ModuleType("accelerate.hooks")
    module.CpuOffload = _FakeCpuOffloadHook
    module.AlignDevicesHook = _FakeAlignDevicesHook
    package = types.ModuleType("accelerate")
    package.hooks = module
    monkeypatch.setitem(sys.modules, "accelerate", package)
    monkeypatch.setitem(sys.modules, "accelerate.hooks", module)
    yield


class _Component:
    """A pipeline component that can carry an accelerate hook."""

    def __init__(self, hook=None, submodule_hook=None):
        if hook is not None:
            self._hf_hook = hook
        self._submodules = []
        if submodule_hook is not None:
            sub = _Component()
            sub._hf_hook = submodule_hook
            self._submodules.append(sub)

    def modules(self):
        yield self
        yield from self._submodules


class _Pipeline:
    """Minimal stand-in for a diffusers pipeline."""

    def __init__(self, **components):
        self.components = components
        self.moved_to = []
        self.model_offload_calls = 0
        self.sequential_offload_calls = 0

    def to(self, device):
        self.moved_to.append(device)
        return self

    def enable_model_cpu_offload(self):
        self.model_offload_calls += 1

    def enable_sequential_cpu_offload(self):
        self.sequential_offload_calls += 1


def test_offload_kind_detects_model_offload():
    """enable_model_cpu_offload installs CpuOffload, not AlignDevicesHook."""
    from nodetool.huggingface.memory_utils import offload_kind

    pipeline = _Pipeline(unet=_Component(hook=_FakeCpuOffloadHook()))
    assert offload_kind(pipeline) == "model"


def test_offload_kind_detects_sequential_offload():
    """Sequential offload dispatches per submodule, so detection must recurse."""
    from nodetool.huggingface.memory_utils import offload_kind

    pipeline = _Pipeline(transformer=_Component(submodule_hook=_FakeAlignDevicesHook()))
    assert offload_kind(pipeline) == "sequential"


def test_offload_kind_none_without_hooks():
    from nodetool.huggingface.memory_utils import offload_kind

    assert offload_kind(_Pipeline(unet=_Component())) is None


def test_apply_cpu_offload_is_idempotent():
    """Re-applying would remove and rebuild every hook on a cached pipeline."""
    from nodetool.huggingface.memory_utils import apply_cpu_offload_if_needed

    pipeline = _Pipeline(unet=_Component())

    assert apply_cpu_offload_if_needed(pipeline, method="model") is True
    assert apply_cpu_offload_if_needed(pipeline, method="model") is False
    assert pipeline.model_offload_calls == 1


def test_apply_cpu_offload_does_not_stack_strategies():
    """A pipeline may carry exactly one offload strategy."""
    from nodetool.huggingface.memory_utils import apply_cpu_offload_if_needed

    pipeline = _Pipeline(transformer=_Component())

    assert apply_cpu_offload_if_needed(pipeline, method="sequential") is True
    assert apply_cpu_offload_if_needed(pipeline, method="model") is False
    assert pipeline.model_offload_calls == 0
    assert pipeline.sequential_offload_calls == 1


def test_apply_cpu_offload_rejects_unknown_method():
    from nodetool.huggingface.memory_utils import apply_cpu_offload_if_needed

    with pytest.raises(ValueError):
        apply_cpu_offload_if_needed(_Pipeline(), method="nonsense")


def test_move_to_gpu_skipped_for_offloaded_pipeline():
    """Moving an offloaded pipeline to the GPU would undo the offload."""
    from nodetool.huggingface.memory_utils import move_pipeline_to_device

    pipeline = _Pipeline(unet=_Component(hook=_FakeCpuOffloadHook()))

    assert move_pipeline_to_device(pipeline, "cuda") is False
    assert pipeline.moved_to == []


def test_move_to_cpu_allowed_for_offloaded_pipeline():
    """Moving *off* the GPU is always safe."""
    from nodetool.huggingface.memory_utils import move_pipeline_to_device

    pipeline = _Pipeline(unet=_Component(hook=_FakeCpuOffloadHook()))

    assert move_pipeline_to_device(pipeline, "cpu") is True
    assert pipeline.moved_to == ["cpu"]


def test_move_to_gpu_allowed_without_offload():
    from nodetool.huggingface.memory_utils import move_pipeline_to_device

    pipeline = _Pipeline(unet=_Component())

    assert move_pipeline_to_device(pipeline, "cuda") is True
    assert pipeline.moved_to == ["cuda"]


def test_ensure_model_on_device_keeps_offload():
    """Cache hits must not pull an offloaded pipeline back onto the GPU."""
    from nodetool.huggingface.local_provider_utils import _ensure_model_on_device

    pipeline = _Pipeline(unet=_Component(hook=_FakeCpuOffloadHook()))

    assert _ensure_model_on_device(pipeline, "cuda") is pipeline
    assert pipeline.moved_to == []


def test_component_cache_suffix_separates_quantizations():
    """Two pipelines differing only by a swapped transformer need distinct keys."""
    from nodetool.huggingface.local_provider_utils import _component_cache_suffix

    int4 = _Component()
    int4._nodetool_cache_key = "repo_NunchakuUNet_svdq-int4.safetensors"
    fp4 = _Component()
    fp4._nodetool_cache_key = "repo_NunchakuUNet_svdq-fp4.safetensors"

    suffix_int4 = _component_cache_suffix({"unet": int4})
    suffix_fp4 = _component_cache_suffix({"unet": fp4})

    assert suffix_int4 and suffix_fp4
    assert suffix_int4 != suffix_fp4
    # A full-precision load passes no component at all.
    assert _component_cache_suffix({"use_safetensors": True}) == ""


def test_component_cache_suffix_is_stable():
    from nodetool.huggingface.local_provider_utils import _component_cache_suffix

    component = _Component()
    component._nodetool_cache_key = "repo_Class_path"

    assert _component_cache_suffix({"unet": component}) == _component_cache_suffix(
        {"unet": component}
    )


def test_cache_key_suffix_reaches_the_lookup():
    """A node's adapter suffix must actually change the key the cache is probed with."""
    import asyncio
    from unittest.mock import MagicMock

    from nodetool.huggingface.local_provider_utils import load_pipeline
    from nodetool.ml.core.model_manager import ModelManager

    plain_key = "repo/model_automatic-speech-recognition"
    suffixed_key = plain_key + "_aDEADBEEF"
    plain, suffixed = object(), object()
    ModelManager._models[plain_key] = plain
    ModelManager._models[suffixed_key] = suffixed

    ctx = MagicMock()
    ctx.device = "cpu"

    def run(suffix):
        return asyncio.run(
            load_pipeline(
                node_id="n",
                context=ctx,
                pipeline_task="automatic-speech-recognition",
                model_id="repo/model",
                cache_key_suffix=suffix,
            )
        )

    try:
        assert run("") is plain
        assert run("_aDEADBEEF") is suffixed
    finally:
        ModelManager._models.pop(plain_key, None)
        ModelManager._models.pop(suffixed_key, None)


def test_release_exception_drops_traceback():
    """The traceback pins the failed load's GPU tensors; it must be released."""
    from nodetool.huggingface.local_provider_utils import _release_exception

    try:
        raise RuntimeError("CUDA out of memory")
    except RuntimeError as exc:
        assert exc.__traceback__ is not None
        _release_exception(exc)
        assert exc.__traceback__ is None
        assert exc.__cause__ is None
        assert exc.__context__ is None
