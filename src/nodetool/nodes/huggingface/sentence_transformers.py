import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, AsyncGenerator

from nodetool.metadata.types import DocumentRef
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from pydantic import Field

# The model LangChain's SentenceTransformersTokenTextSplitter used by default
# (this node never exposed a model_name field, so it was always this one).
_SPLIT_TOKENIZER_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Last-known usable sequence length for _SPLIT_TOKENIZER_MODEL, used only as
# a fallback when the model's own config can't be read (offline, HF Hub
# unreachable) -- see _model_max_seq_length(). Not a claim that this holds
# for every model; a bare module-level constant would silently go stale if
# _SPLIT_TOKENIZER_MODEL ever changed, so the real value is read at runtime.
_FALLBACK_TOKENS_PER_CHUNK = 384


@lru_cache(maxsize=1)
def _split_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_SPLIT_TOKENIZER_MODEL)


@lru_cache(maxsize=1)
def _model_max_seq_length() -> int:
    """The model's real usable token budget.

    sentence-transformers caps this below the tokenizer's own
    `model_max_length` (512 for this model): the model repo carries a
    `sentence_bert_config.json` with its own `max_seq_length` (384), and
    that -- not the tokenizer's limit -- is what the model can actually
    embed. Read from the model repo at runtime rather than hardcoded, so
    this can't drift from the model; falls back to the last-known value if
    the config can't be fetched.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(_SPLIT_TOKENIZER_MODEL, "sentence_bert_config.json")
        return int(json.loads(Path(path).read_text())["max_seq_length"])
    except Exception:
        return _FALLBACK_TOKENS_PER_CHUNK


def _encode_without_special_tokens(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(
        text, max_length=2**32, truncation="do_not_truncate"
    )
    # Strip the leading/trailing special tokens (e.g. [CLS]/[SEP]) the same
    # way LangChain's SentenceTransformersTokenTextSplitter._encode does.
    return token_ids[1:-1]


def _split_text_on_tokens(
    text: str, tokenizer, tokens_per_chunk: int, chunk_overlap: int
) -> list[str]:
    if tokens_per_chunk <= chunk_overlap:
        raise ValueError("tokens_per_chunk must be greater than chunk_overlap")

    input_ids = _encode_without_special_tokens(tokenizer, text)
    splits: list[str] = []
    start_idx = 0
    while start_idx < len(input_ids):
        cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        if not chunk_ids:
            break
        decoded = tokenizer.decode(chunk_ids)
        if decoded:
            splits.append(decoded)
        if cur_idx == len(input_ids):
            break
        start_idx += tokens_per_chunk - chunk_overlap
    return splits


class SplitSentences(BaseNode):
    """
    Splits text into sentences using a sentence-transformer tokenizer's token count.
    sentences, split, nlp

    Use cases:
    - Natural sentence-based text splitting
    - Creating semantically meaningful chunks
    - Processing text for sentence-level analysis
    """

    document: DocumentRef = Field(default=DocumentRef())
    chunk_size: int = Field(
        default=40,
        description=(
            "Maximum number of tokens per chunk. Capped at the embedding "
            "model's maximum sequence length (384 tokens for the default "
            "model) -- a larger value is clamped down to that limit, since "
            "the model cannot embed more tokens than that in one pass."
        ),
    )
    chunk_overlap: int = Field(
        default=5,
        description="Number of tokens to overlap between chunks",
    )

    @classmethod
    def get_title(cls):
        return "Split into Sentences"

    class OutputType(TypedDict):
        text: str
        source_id: str
        start_index: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be >= 0, got {self.chunk_overlap}"
            )

        model_max = _model_max_seq_length()
        effective_chunk_size = min(self.chunk_size, model_max)

        if self.chunk_overlap >= effective_chunk_size:
            capped_note = (
                f" (chunk_size={self.chunk_size} was capped to the model's "
                f"{model_max}-token limit)"
                if effective_chunk_size < self.chunk_size
                else ""
            )
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"the effective chunk size ({effective_chunk_size}){capped_note}"
            )

        text = self.document.data
        tokenizer = _split_tokenizer()
        chunks = _split_text_on_tokens(
            text, tokenizer, effective_chunk_size, self.chunk_overlap
        )

        # Mirrors LangChain's TextSplitter.create_documents start_index search:
        # search from the previous chunk's end minus the overlap, forward.
        index = 0
        previous_chunk_len = 0
        for i, chunk in enumerate(chunks):
            offset = index + previous_chunk_len - self.chunk_overlap
            index = text.find(chunk, max(0, offset))
            previous_chunk_len = len(chunk)
            yield {
                "text": chunk,
                "source_id": f"{self.document.uri}:{i}",
                "start_index": index,
            }
