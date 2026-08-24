"""Regression tests for the QwenEmbedRope compatibility shim.

nunchaku's Qwen transformer calls ``QwenEmbedRope.forward`` with the diffusers
0.36 signature. diffusers 0.39 removed the ``txt_seq_lens`` parameter, so the
call raises "got multiple values for argument 'device'". See
``apply_qwen_rope_compat_shim`` for the full account.

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

from nodetool.huggingface.nunchaku_pipelines import (  # noqa: E402
    apply_qwen_rope_compat_shim,
)

IMG_SHAPES = [(1, 32, 32)]
TXT_SEQ_LENS = [16]


@pytest.fixture(autouse=True)
def restore_forward():
    """Undo whatever a test did to the shared diffusers class."""
    original = QwenEmbedRope.forward
    yield
    QwenEmbedRope.forward = original


def make_rope():
    return QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)


def needs_shim() -> bool:
    params = inspect.signature(QwenEmbedRope.forward).parameters
    return "max_txt_seq_len" in params and "txt_seq_lens" not in params


def test_legacy_positional_call_matches_the_modern_keyword_call():
    """The whole point: nunchaku's call works, and returns the right answer.

    Correctness is asserted against the modern call rather than against a
    hardcoded shape, so this stays honest if diffusers changes what it returns.
    """
    if not needs_shim():
        pytest.skip("diffusers still accepts the legacy call; shim not applicable")

    rope = make_rope()
    device = torch.device("cpu")

    expected = rope(IMG_SHAPES, max_txt_seq_len=max(TXT_SEQ_LENS), device=device)

    # Without the shim this is the reported TypeError.
    with pytest.raises(TypeError, match="multiple values for argument 'device'"):
        rope(IMG_SHAPES, TXT_SEQ_LENS, device=device)

    assert apply_qwen_rope_compat_shim() is True

    got = rope(IMG_SHAPES, TXT_SEQ_LENS, device=device)

    assert len(got) == len(expected)
    for actual, want in zip(got, expected):
        assert actual.shape == want.shape
        assert torch.equal(actual, want)


def test_modern_keyword_call_still_works_through_the_shim():
    """The shim must not disturb callers that already use the new contract."""
    if not needs_shim():
        pytest.skip("diffusers still accepts the legacy call; shim not applicable")

    rope = make_rope()
    device = torch.device("cpu")
    expected = rope(IMG_SHAPES, max_txt_seq_len=16, device=device)

    apply_qwen_rope_compat_shim()

    got = rope(IMG_SHAPES, max_txt_seq_len=16, device=device)
    for actual, want in zip(got, expected):
        assert torch.equal(actual, want)


def test_is_a_no_op_when_the_signature_does_not_need_it():
    """Guard against silently corrupting a future fixed nunchaku or diffusers.

    Simulates the pre-0.39 signature, which still accepts ``txt_seq_lens``. The
    shim must decline to wrap it: rewriting the argument there would move
    ``device`` into the wrong slot.
    """
    calls = []

    def legacy_forward(self, video_fhw, txt_seq_lens=None, device=None):
        calls.append((video_fhw, txt_seq_lens, device))
        return "legacy-sentinel"

    QwenEmbedRope.forward = legacy_forward

    assert apply_qwen_rope_compat_shim() is False
    assert QwenEmbedRope.forward is legacy_forward

    rope = make_rope()
    assert rope(IMG_SHAPES, TXT_SEQ_LENS, device=torch.device("cpu")) == (
        "legacy-sentinel"
    )
    assert calls == [(IMG_SHAPES, TXT_SEQ_LENS, torch.device("cpu"))]


def test_applying_twice_changes_nothing():
    """A second pipeline load must not double-wrap the class."""
    if not needs_shim():
        pytest.skip("diffusers still accepts the legacy call; shim not applicable")

    assert apply_qwen_rope_compat_shim() is True
    once = QwenEmbedRope.forward

    assert apply_qwen_rope_compat_shim() is True
    assert QwenEmbedRope.forward is once

    rope = make_rope()
    device = torch.device("cpu")
    expected = rope(IMG_SHAPES, max_txt_seq_len=max(TXT_SEQ_LENS), device=device)
    got = rope(IMG_SHAPES, TXT_SEQ_LENS, device=device)
    for actual, want in zip(got, expected):
        assert torch.equal(actual, want)
