"""SplitSentences used to import langchain_text_splitters/langchain_core
straight inside gen_process, undeclared anywhere in pyproject.toml -- proven
broken on a real worker with `No module named 'langchain_text_splitters'`
(see #65 / #69). It has been rewritten to chunk directly against a
`transformers` tokenizer (already a base dependency) instead of LangChain's
SentenceTransformersTokenTextSplitter wrapper around it.

Two things had to be verified, not guessed, before the rewrite:

1. `chunk_size` never actually sized a chunk in the LangChain implementation.
   `SentenceTransformersTokenTextSplitter.__init__` forwards `chunk_size` to
   `TextSplitter.__init__` only for its ">0 and >= chunk_overlap" validation;
   the real per-chunk token budget is
   `SentenceTransformer("sentence-transformers/all-mpnet-base-v2").max_seq_length`
   (384 -- sentence_bert_config.json's own cap, below the tokenizer's
   `model_max_length` of 512). `test_chunk_size_field_was_always_a_no_op`
   below pins that this was real, observable behavior, not a misreading.

2. `start_index` comes from `TextSplitter.create_documents`'s substring
   search (`text.find(chunk, max(0, index + previous_chunk_len -
   chunk_overlap))`), which can legitimately return -1 when the decoded chunk
   doesn't reappear verbatim in the source text -- not a bug to paper over.

Both are reproduced exactly in `_split_text_on_tokens` /
`SplitSentences.gen_process`, and cross-checked here against the LangChain
implementation directly (skipped if langchain isn't installed, since it is no
longer a declared dependency of this package).
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
    _TOKENS_PER_CHUNK,
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
    # Long enough (well past the 384-token chunk budget) to force multiple
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
async def test_matches_langchain_implementation_on_a_long_document():
    text = _long_text()
    expected = _run_langchain(text, chunk_size=40, chunk_overlap=5)
    actual = await _run_node(text, chunk_size=40, chunk_overlap=5)

    assert actual == expected
    # Sanity: this document is long enough to actually exercise chunking.
    assert len(actual) > 1


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain not installed")
@pytest.mark.asyncio
async def test_matches_langchain_implementation_with_a_different_overlap():
    text = _long_text(1500)
    expected = _run_langchain(text, chunk_size=200, chunk_overlap=50)
    actual = await _run_node(text, chunk_size=200, chunk_overlap=50)

    assert actual == expected
    assert len(actual) > 1


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain not installed")
def test_chunk_size_field_was_always_a_no_op():
    """Documents the surprising real behavior the rewrite had to preserve:
    varying chunk_size alone changes nothing about the chunks LangChain
    produced, because SentenceTransformersTokenTextSplitter never reads it
    back out -- only the model's fixed max_seq_length (384) does."""
    text = _long_text()
    by_40 = _run_langchain(text, chunk_size=40, chunk_overlap=5)
    by_300 = _run_langchain(text, chunk_size=300, chunk_overlap=5)

    assert by_40 == by_300


@pytest.mark.asyncio
async def test_short_document_yields_a_single_chunk():
    # The MPNet tokenizer decodes lower-cased, so a short document with
    # capitals round-trips to a differently-cased string than the source --
    # which is also why its start_index search below misses (-1). Both are
    # genuine LangChain behavior, reproduced here, not artifacts of the
    # rewrite: see test_matches_langchain_implementation_on_a_long_document.
    text = "This is a short document with well under 384 tokens."
    result = await _run_node(text, chunk_size=40, chunk_overlap=5)

    assert len(result) == 1
    assert result[0]["text"] == text.lower()
    assert result[0]["start_index"] == -1
    assert result[0]["source_id"] == "doc://test:0"


@pytest.mark.asyncio
async def test_short_lowercase_document_finds_its_start_index():
    text = "this is a short document with well under 384 tokens."
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
    with pytest.raises(ValueError, match="larger chunk overlap"):
        await _run_node(_long_text(50), chunk_size=5, chunk_overlap=40)


@pytest.mark.asyncio
async def test_chunk_size_not_positive_raises():
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        await _run_node(_long_text(50), chunk_size=0, chunk_overlap=0)


def test_tokens_per_chunk_matches_the_model_sentence_bert_config():
    """Pin 384 to its actual source (the model's own
    sentence_bert_config.json), so this constant can't silently drift from
    what LangChain's SentenceTransformer-backed splitter really used."""
    from huggingface_hub import hf_hub_download
    import json

    path = hf_hub_download(
        "sentence-transformers/all-mpnet-base-v2", "sentence_bert_config.json"
    )
    config = json.loads(Path(path).read_text())
    assert config["max_seq_length"] == _TOKENS_PER_CHUNK


def test_split_text_on_tokens_rejects_overlap_at_least_tokens_per_chunk():
    tokenizer = _split_tokenizer()
    with pytest.raises(ValueError, match="tokens_per_chunk must be greater"):
        _split_text_on_tokens("hello world", tokenizer, _TOKENS_PER_CHUNK)
