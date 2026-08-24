"""SplitSentences used to import langchain_text_splitters/langchain_core
straight inside gen_process, undeclared anywhere in pyproject.toml -- proven
broken on a real worker with `No module named 'langchain_text_splitters'`
(see #65 / #69). It has been rewritten to chunk directly against a
`transformers` tokenizer (already a base dependency) instead of LangChain's
SentenceTransformersTokenTextSplitter wrapper around it.

Two things had to be verified, not guessed, while building the rewrite:

1. `chunk_size` never sized a chunk in LangChain's implementation.
   `SentenceTransformersTokenTextSplitter.__init__` forwards `chunk_size` to
   `TextSplitter.__init__` only for its ">0 and >= chunk_overlap" validation;
   the real per-chunk token budget was always
   `SentenceTransformer("sentence-transformers/all-mpnet-base-v2").max_seq_length`
   (384 -- sentence_bert_config.json's own cap, below the tokenizer's
   `model_max_length` of 512).

2. `start_index` comes from `TextSplitter.create_documents`'s substring
   search (`text.find(chunk, max(0, index + previous_chunk_len -
   chunk_overlap))`), which can legitimately return -1 when the decoded chunk
   doesn't reappear verbatim in the source text -- not a bug to paper over.

BREAKING CHANGE from the first rewrite: `chunk_size` is no longer treated as
inert. It is now honored, capped at the model's real usable sequence length
(`_model_max_seq_length()`, read from the model's own
`sentence_bert_config.json` at runtime -- not hardcoded, since a bare
per-model constant would silently go stale if `_SPLIT_TOKENIZER_MODEL` ever
changed). This means:

- `chunk_size=40` (the field's default) now produces ~40-token chunks, where
  it previously produced 384-token chunks. Anyone already using this node
  with the default (or any `chunk_size < 384`) gets smaller chunks than
  before -- a real behavior change, not just an internal cleanup.
- `chunk_size >= 384` is unaffected: it was already being capped to 384 by
  the model, so those callers see identical output.

The tokenizer/start_index logic itself (point 2 above, and the encode/decode
round trip) is unchanged, so the two are tested separately:
`test_matches_langchain_*` below prove the shared logic still matches
LangChain byte-for-byte at chunk sizes >= the model's cap (the path that
hasn't changed); `test_small_chunk_size_produces_small_chunks` and
`test_chunk_size_is_honored_and_capped_at_the_model_limit` prove the new
sizing behavior.
"""

import sys
from pathlib import Path

import pytest

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.metadata.types import DocumentRef
from nodetool.nodes.huggingface.sentence_transformers import (
    SplitSentences,
    _FALLBACK_TOKENS_PER_CHUNK,
    _encode_without_special_tokens,
    _model_max_seq_length,
    _split_text_on_tokens,
    _split_tokenizer,
)
from nodetool.workflows.processing_context import ProcessingContext

try:
    from langchain_text_splitters import SentenceTransformersTokenTextSplitter
    from langchain_core.documents import Document

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def _long_text(num_words: int = 1000) -> str:
    # Long enough (well past the ~384-token chunk budget) to force multiple
    # chunks and exercise start_index across a chunk boundary.
    return " ".join(f"word{i}" for i in range(num_words))


async def _run_node(text: str, *, chunk_size: int, chunk_overlap: int):
    node = SplitSentences(
        document=DocumentRef(data=text, uri="doc://test"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    context = ProcessingContext(encode_assets_as_base64=True)
    return [item async for item in node.gen_process(context)]


def _run_langchain(text: str, *, chunk_size: int, chunk_overlap: int):
    splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    docs = splitter.split_documents([Document(page_content=text)])
    return [
        {
            "text": doc.page_content,
            "source_id": f"doc://test:{i}",
            "start_index": doc.metadata.get("start_index", 0),
        }
        for i, doc in enumerate(docs)
    ]


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain not installed")
@pytest.mark.asyncio
async def test_matches_langchain_implementation_at_a_chunk_size_above_the_cap():
    """chunk_size >= the model's cap is the path that hasn't changed: both
    the old and new implementation clamp to the model's max_seq_length, so
    the tokenizer/start_index logic should still match LangChain exactly."""
    text = _long_text()
    expected = _run_langchain(text, chunk_size=1000, chunk_overlap=5)
    actual = await _run_node(text, chunk_size=1000, chunk_overlap=5)

    assert actual == expected
    # Sanity: this document is long enough to actually exercise chunking.
    assert len(actual) > 1


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain not installed")
@pytest.mark.asyncio
async def test_matches_langchain_implementation_with_a_different_overlap():
    text = _long_text(1500)
    expected = _run_langchain(text, chunk_size=1000, chunk_overlap=50)
    actual = await _run_node(text, chunk_size=1000, chunk_overlap=50)

    assert actual == expected
    assert len(actual) > 1


def test_small_chunk_size_produces_small_chunks():
    """The behavior change: chunk_size below the model's cap now actually
    sizes the chunks, instead of being silently ignored in favor of the
    model's 384-token limit.

    Chunk *count* is checked against a sliding-window count computed
    independently from the raw token ids (not by re-encoding the decoded
    chunk text, which the wordpiece tokenizer doesn't round-trip losslessly
    token-for-token)."""
    text = _long_text()
    tokenizer = _split_tokenizer()
    input_ids = _encode_without_special_tokens(tokenizer, text)
    total_tokens = len(input_ids)

    chunk_size = 40
    chunk_overlap = 5
    splits = _split_text_on_tokens(text, tokenizer, chunk_size, chunk_overlap)

    expected_count = 0
    start = 0
    while start < total_tokens:
        expected_count += 1
        end = min(start + chunk_size, total_tokens)
        if end == total_tokens:
            break
        start += chunk_size - chunk_overlap
    assert len(splits) == expected_count
    assert len(splits) > 1

    # Small chunks means many more of them than the capped case.
    capped = _split_text_on_tokens(text, tokenizer, 10_000, chunk_overlap)
    assert len(splits) > len(capped)


@pytest.mark.asyncio
async def test_chunk_size_is_honored_and_capped_at_the_model_limit():
    text = _long_text(2000)
    model_max = _model_max_seq_length()

    at_cap = await _run_node(text, chunk_size=model_max, chunk_overlap=5)
    above_cap = await _run_node(text, chunk_size=model_max + 1000, chunk_overlap=5)
    below_cap = await _run_node(text, chunk_size=model_max - 50, chunk_overlap=5)

    # Anything at or above the model's limit is clamped to the same thing.
    assert at_cap == above_cap
    # Below the limit, chunk_size is honored: strictly more (smaller) chunks.
    assert len(below_cap) > len(at_cap)


@pytest.mark.asyncio
async def test_short_document_yields_a_single_chunk():
    # The MPNet tokenizer decodes lower-cased, so a short document with
    # capitals round-trips to a differently-cased string than the source --
    # which is also why its start_index search below misses (-1). Both are
    # genuine LangChain behavior, reproduced here, not artifacts of the
    # rewrite.
    text = "This is a short document with well under 40 tokens."
    result = await _run_node(text, chunk_size=40, chunk_overlap=5)

    assert len(result) == 1
    assert result[0]["text"] == text.lower()
    assert result[0]["start_index"] == -1
    assert result[0]["source_id"] == "doc://test:0"


@pytest.mark.asyncio
async def test_short_lowercase_document_finds_its_start_index():
    text = "this is a short document with well under 40 tokens."
    result = await _run_node(text, chunk_size=40, chunk_overlap=5)

    assert len(result) == 1
    assert result[0]["text"] == text
    assert result[0]["start_index"] == 0


@pytest.mark.asyncio
async def test_empty_document_yields_no_chunks():
    result = await _run_node("", chunk_size=40, chunk_overlap=5)
    assert result == []


@pytest.mark.asyncio
async def test_chunk_overlap_greater_than_chunk_size_raises():
    with pytest.raises(ValueError, match="effective chunk size"):
        await _run_node(_long_text(50), chunk_size=5, chunk_overlap=40)


@pytest.mark.asyncio
async def test_chunk_overlap_at_least_the_capped_effective_size_raises():
    """The classic infinite-loop bug in every token splitter: chunk_size
    above the model's cap (so the effective size is smaller than
    chunk_size), with chunk_overlap at or above that *effective* size. Must
    be rejected before any chunking loop runs, not hang."""
    model_max = _model_max_seq_length()
    with pytest.raises(ValueError, match="effective chunk size"):
        await _run_node(
            _long_text(2000),
            chunk_size=model_max + 200,
            chunk_overlap=model_max,
        )


@pytest.mark.asyncio
async def test_chunk_size_not_positive_raises():
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        await _run_node(_long_text(50), chunk_size=0, chunk_overlap=0)


def test_model_max_seq_length_reads_the_live_model_config():
    """Pin the runtime read to its actual source (the model's own
    sentence_bert_config.json), so it can't silently drift from what the
    model really supports."""
    from huggingface_hub import hf_hub_download
    import json

    path = hf_hub_download(
        "sentence-transformers/all-mpnet-base-v2", "sentence_bert_config.json"
    )
    config = json.loads(Path(path).read_text())
    assert _model_max_seq_length() == config["max_seq_length"]


def test_model_max_seq_length_falls_back_when_the_config_is_unreachable(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("simulated: model repo unreachable")

    # hf_hub_download is imported inside _model_max_seq_length at call time,
    # so patch it where it actually resolves from.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _raise)
    _model_max_seq_length.cache_clear()
    try:
        assert _model_max_seq_length() == _FALLBACK_TOKENS_PER_CHUNK
    finally:
        _model_max_seq_length.cache_clear()


def test_split_text_on_tokens_rejects_overlap_at_least_tokens_per_chunk():
    tokenizer = _split_tokenizer()
    with pytest.raises(ValueError, match="tokens_per_chunk must be greater"):
        _split_text_on_tokens("hello world", tokenizer, 40, 40)
