from functools import lru_cache
from typing import TypedDict, AsyncGenerator

from nodetool.metadata.types import DocumentRef
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from pydantic import Field

# The model LangChain's SentenceTransformersTokenTextSplitter used by default
# (this node never exposed a model_name field, so it was always this one).
_SPLIT_TOKENIZER_MODEL = "sentence-transformers/all-mpnet-base-v2"

# sentence-transformers caps this model's usable sequence length below the
# tokenizer's own `model_max_length` (512): the model's
# `sentence_bert_config.json` pins `max_seq_length` to 384, and that value --
# not the node's `chunk_size` field -- is what LangChain actually chunked on.
# `chunk_size` only gated a "must be > 0 and >= chunk_overlap" validation in
# LangChain's base TextSplitter; it never sized a chunk. Preserved here for
# behavioral parity with the pre-rewrite implementation.
_TOKENS_PER_CHUNK = 384


@lru_cache(maxsize=1)
def _split_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_SPLIT_TOKENIZER_MODEL)


def _encode_without_special_tokens(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(
        text, max_length=2**32, truncation="do_not_truncate"
    )
    # Strip the leading/trailing special tokens (e.g. [CLS]/[SEP]) the same
    # way LangChain's SentenceTransformersTokenTextSplitter._encode does.
    return token_ids[1:-1]


def _split_text_on_tokens(text: str, tokenizer, chunk_overlap: int) -> list[str]:
    if _TOKENS_PER_CHUNK <= chunk_overlap:
        raise ValueError("tokens_per_chunk must be greater than chunk_overlap")

    input_ids = _encode_without_special_tokens(tokenizer, text)
    splits: list[str] = []
    start_idx = 0
    while start_idx < len(input_ids):
        cur_idx = min(start_idx + _TOKENS_PER_CHUNK, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        if not chunk_ids:
            break
        decoded = tokenizer.decode(chunk_ids)
        if decoded:
            splits.append(decoded)
        if cur_idx == len(input_ids):
            break
        start_idx += _TOKENS_PER_CHUNK - chunk_overlap
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
        description="Maximum number of tokens per chunk",
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
        if self.chunk_overlap > self.chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({self.chunk_overlap}) than chunk "
                f"size ({self.chunk_size}), should be smaller."
            )

        text = self.document.data
        tokenizer = _split_tokenizer()
        chunks = _split_text_on_tokens(text, tokenizer, self.chunk_overlap)

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
