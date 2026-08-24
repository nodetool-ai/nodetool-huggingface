"""Regression: every FLUX variant shared one nunchaku T5 encoder object.

``load_model`` derives its cache key from the repo id, the model class and the
file path. The nunchaku T5 encoder is the same file for schnell, dev, fill,
canny, depth and kontext, so one object served all six pipelines. Two separate
failures followed, and each hid the other on hardware:

* FluxKontext defaults ``enable_cpu_offload=True``. Its
  ``enable_sequential_cpu_offload`` stamped an ``AlignDevicesHook`` onto the
  shared encoder. The next pipeline inherited it, ``offload_kind`` reported
  "sequential" for a pipeline nobody had offloaded, and both
  ``apply_cpu_offload_if_needed`` and ``move_pipeline_to_device`` skipped their
  work — leaving CLIP on the CPU. FluxFill and FluxControl then died with
  "index is on cuda:0, different from other tensors on cpu".
* ``torch_dtype`` never reached the cache key, so the first loader's dtype won.
  With the hooks fixed, FluxFill instead died with "Input type (c10::BFloat16)
  and bias type (c10::Half) should be the same".

Reproduced on an A40: evict, run FluxKontext, then run FluxFill.
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from nodetool.huggingface.memory_utils import (
    apply_cpu_offload_if_needed,
    claim_pipeline_components,
    move_pipeline_to_device,
    offload_kind,
    owns_its_components,
)
from nodetool.huggingface.local_provider_utils import _dtype_cache_suffix


torch = pytest.importorskip("torch")


class _Module(torch.nn.Module):
    """A real nn.Module: accelerate walks children() to detach hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


class _Pipeline:
    def __init__(self, text_encoder_2) -> None:
        self.transformer = _Module()
        self.vae = _Module()
        self.text_encoder = _Module()
        self.text_encoder_2 = text_encoder_2
        self.moved_to: list[str] = []
        self.offloads: list[str] = []

    def to(self, device):
        self.moved_to.append(device)

    def enable_sequential_cpu_offload(self):
        self.offloads.append("sequential")

    def enable_model_cpu_offload(self):
        self.offloads.append("model")


def _install_sequential_offload(pipeline) -> None:
    """What ``enable_sequential_cpu_offload`` leaves behind on each submodule."""
    hooks = pytest.importorskip("accelerate.hooks")
    for name in ("transformer", "vae", "text_encoder", "text_encoder_2"):
        getattr(pipeline, name)._hf_hook = hooks.AlignDevicesHook()


def test_a_borrowed_hook_is_not_reported_as_the_new_pipelines_offload():
    """The whole bug: kontext's hook made fill look already-offloaded."""
    shared_t5 = _Module()

    kontext = _Pipeline(shared_t5)
    claim_pipeline_components(kontext)
    _install_sequential_offload(kontext)
    assert offload_kind(kontext) == "sequential"

    fill = _Pipeline(shared_t5)  # same encoder object, straight from the cache
    assert offload_kind(fill) == "sequential", "precondition: the hook is inherited"

    claim_pipeline_components(fill)

    assert offload_kind(fill) is None


def test_the_previous_owner_is_marked_stale_and_must_be_rebuilt():
    """Claiming strips kontext's hooks, so its cached pipeline is no longer valid."""
    shared_t5 = _Module()

    kontext = _Pipeline(shared_t5)
    claim_pipeline_components(kontext)
    _install_sequential_offload(kontext)
    assert owns_its_components(kontext)

    fill = _Pipeline(shared_t5)
    claim_pipeline_components(fill)

    assert owns_its_components(fill)
    assert not owns_its_components(kontext)


def test_claiming_an_unshared_pipeline_changes_nothing():
    pipeline = _Pipeline(_Module())
    claim_pipeline_components(pipeline)
    _install_sequential_offload(pipeline)

    assert claim_pipeline_components(pipeline) is False
    assert offload_kind(pipeline) == "sequential"


def test_a_pipeline_nobody_claimed_owns_itself():
    assert owns_its_components(_Pipeline(_Module()))


@pytest.mark.parametrize(
    "dtype_name, other_name",
    [("bfloat16", "float16"), ("float16", "float32")],
)
def test_the_cache_key_separates_dtypes(dtype_name, other_name):
    """schnell asks for bfloat16 and fill for float16; both must not collide."""
    torch = pytest.importorskip("torch")
    mine = _dtype_cache_suffix(getattr(torch, dtype_name))
    theirs = _dtype_cache_suffix(getattr(torch, other_name))

    assert mine != theirs
    assert dtype_name in mine


def test_no_dtype_keeps_the_key_unchanged():
    assert _dtype_cache_suffix(None) == ""


def test_the_inheriting_pipeline_is_placed_again():
    """Both placement paths were no-ops, which is what stranded CLIP on the CPU."""
    shared_t5 = _Module()

    kontext = _Pipeline(shared_t5)
    claim_pipeline_components(kontext)
    _install_sequential_offload(kontext)

    fill = _Pipeline(shared_t5)
    claim_pipeline_components(fill)

    assert move_pipeline_to_device(fill, "cuda") is True
    assert fill.moved_to == ["cuda"]


def test_the_inheriting_pipeline_can_still_install_offload():
    shared_t5 = _Module()

    kontext = _Pipeline(shared_t5)
    claim_pipeline_components(kontext)
    _install_sequential_offload(kontext)

    fill = _Pipeline(shared_t5)
    claim_pipeline_components(fill)

    assert apply_cpu_offload_if_needed(fill, method="sequential") is True
    assert fill.offloads == ["sequential"]
