"""``load_model`` must keep loads that differ only in dtype in separate cache entries.

``torch_dtype`` is a named parameter, so it never reached the derived cache
key. Two pipelines asking for the same weights in different dtypes then shared
one object, and the later one fed bf16 hidden states into fp16 layers: "Input
type (c10::BFloat16) and bias type (c10::Half) should be the same".
"""

import pytest

from nodetool.huggingface.local_provider_utils import _dtype_cache_suffix


@pytest.mark.parametrize(
    "dtype_name, other_name",
    [("bfloat16", "float16"), ("float16", "float32")],
)
def test_the_cache_key_separates_dtypes(dtype_name, other_name):
    torch = pytest.importorskip("torch")
    mine = _dtype_cache_suffix(getattr(torch, dtype_name))
    theirs = _dtype_cache_suffix(getattr(torch, other_name))

    assert mine != theirs
    assert dtype_name in mine


def test_no_dtype_keeps_the_key_unchanged():
    assert _dtype_cache_suffix(None) == ""
