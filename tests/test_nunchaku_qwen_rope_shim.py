"""Regression tests for the nunchaku Qwen compatibility shims.

diffusers PR #12702 removed ``txt_seq_lens`` from both ``QwenEmbedRope.forward``
and ``QwenImagePipeline``. nunchaku still passes the old form, and no longer
receives a value to pass. See ``apply_qwen_rope_compat_shim``.

The call these tests exercise is the one the pipeline really makes:
``pos_embed(img_shapes, None, device=...)``. An earlier version of this suite
used ``[16]`` there -- a value diffusers >= 0.39 never produces -- and passed
against a shim that still failed on the GPU.

These tests run on CPU and need no GPU and no network.
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.models.transformers.transformer_qwenimage import (  # noqa: E402
    QwenEmbedRope,
)

from nodetool.huggingface import nunchaku_pipelines  # noqa: E402
from nodetool.huggingface.nunchaku_pipelines import (  # noqa: E402
    apply_qwen_rope_compat_shim,
)

IMG_SHAPES = [(1, 32, 32)]
TEXT_SEQ_LEN = 16


@pytest.fixture(autouse=True)
def restore_forward():
    """Undo whatever a test did to the shared diffusers class."""
    original = QwenEmbedRope.forward
    yield
    QwenEmbedRope.forward = original


def make_rope():
    return QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)


def shim_applicable() -> bool:
    params = inspect.signature(QwenEmbedRope.forward).parameters
    return "max_txt_seq_len" in params and "txt_seq_lens" not in params


class FakeTransformer:
    """Stands in for NunchakuQwenImageTransformer2DModel.

    Same parameter order as nunchaku 1.2.1, and the same inner rope call.
    """

    def __init__(self, rope):
        self.pos_embed = rope
        self.seen_txt_seq_lens = "unset"

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        timestep=None,
        img_shapes=None,
        txt_seq_lens=None,
        guidance=None,
        return_dict=True,
    ):
        self.seen_txt_seq_lens = txt_seq_lens
        # nunchaku/models/transformers/transformer_qwenimage.py:517
        return self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)


def test_the_call_the_pipeline_actually_makes_works():
    """The regression: diffusers >= 0.39 never supplies txt_seq_lens.

    nunchaku therefore forwards its own default of None into the positional
    slot. This is the exact call that failed on the GPU with the first shim.
    """
    if not shim_applicable():
        pytest.skip("diffusers still accepts the legacy call; shim not applicable")

    rope = make_rope()
    device = torch.device("cpu")

    with pytest.raises(TypeError, match="multiple values for argument 'device'"):
        rope(IMG_SHAPES, None, device=device)

    assert apply_qwen_rope_compat_shim() is True

    # Still raises, but now for the honest reason: nothing supplied a length.
    # The transformer shim is what provides one; see the test below.
    with pytest.raises(ValueError, match="max_txt_seq_len"):
        rope(IMG_SHAPES, None, device=device)


def test_transformer_shim_supplies_the_length_so_the_whole_chain_runs():
    """End to end over the real call chain, with the weights stubbed out."""
    if not shim_applicable():
        pytest.skip("diffusers still accepts the legacy call; shim not applicable")

    apply_qwen_rope_compat_shim()

    rope = make_rope()
    transformer = FakeTransformer(rope)
    # Drive the real shim against a stand-in, so this test exercises the
    # shipped code rather than a copy of it.
    assert nunchaku_pipelines._shim_qwen_transformer_seq_lens(FakeTransformer) is True

    hidden_states = torch.zeros(1, 4, 8)
    encoder_hidden_states = torch.zeros(1, TEXT_SEQ_LEN, 8)

    got = transformer.forward(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        img_shapes=IMG_SHAPES,
    )

    assert transformer.seen_txt_seq_lens == [TEXT_SEQ_LEN]

    expected = make_rope()(
        IMG_SHAPES, max_txt_seq_len=TEXT_SEQ_LEN, device=hidden_states.device
    )
    for actual, want in zip(got, expected):
        assert torch.equal(actual, want)


def test_legacy_list_form_still_converts():
    """A caller that does supply lengths keeps working (diffusers' own rule)."""
    if not shim_applicable():
        pytest.skip("diffusers still accepts the legacy call; shim not applicable")

    rope = make_rope()
    device = torch.device("cpu")
    expected = rope(IMG_SHAPES, max_txt_seq_len=TEXT_SEQ_LEN, device=device)

    apply_qwen_rope_compat_shim()

    got = rope(IMG_SHAPES, [TEXT_SEQ_LEN], device=device)
    for actual, want in zip(got, expected):
        assert torch.equal(actual, want)


def test_modern_keyword_call_still_works_through_the_shim():
    """The shim must not disturb callers that already use the new contract."""
    if not shim_applicable():
        pytest.skip("diffusers still accepts the legacy call; shim not applicable")

    rope = make_rope()
    device = torch.device("cpu")
    expected = rope(IMG_SHAPES, max_txt_seq_len=TEXT_SEQ_LEN, device=device)

    apply_qwen_rope_compat_shim()

    got = rope(IMG_SHAPES, max_txt_seq_len=TEXT_SEQ_LEN, device=device)
    for actual, want in zip(got, expected):
        assert torch.equal(actual, want)


def test_is_a_no_op_when_the_signature_does_not_need_it():
    """Guard against silently corrupting a future fixed nunchaku or diffusers."""
    calls = []

    def legacy_forward(self, video_fhw, txt_seq_lens=None, device=None):
        calls.append((video_fhw, txt_seq_lens, device))
        return "legacy-sentinel"

    QwenEmbedRope.forward = legacy_forward

    assert nunchaku_pipelines._shim_qwen_rope_signature() is False
    assert QwenEmbedRope.forward is legacy_forward

    rope = make_rope()
    assert rope(IMG_SHAPES, [TEXT_SEQ_LEN], device=torch.device("cpu")) == (
        "legacy-sentinel"
    )
    assert calls == [(IMG_SHAPES, [TEXT_SEQ_LEN], torch.device("cpu"))]


def test_applying_twice_changes_nothing():
    """A second pipeline load must not double-wrap the class."""
    if not shim_applicable():
        pytest.skip("diffusers still accepts the legacy call; shim not applicable")

    assert apply_qwen_rope_compat_shim() is True
    once = QwenEmbedRope.forward

    assert apply_qwen_rope_compat_shim() is True
    assert QwenEmbedRope.forward is once
