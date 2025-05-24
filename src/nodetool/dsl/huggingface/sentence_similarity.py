from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class SentenceSimilarity(GraphNode):
    """
    Compares the similarity between two sentences.
    text, sentence similarity, embeddings, natural language processing

    Use cases:
    - Duplicate detection in text data
    - Semantic search
    - Sentiment analysis
    """

    model: types.HFSentenceSimilarity | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFSentenceSimilarity(
            type="hf.sentence_similarity",
            repo_id="",
            path=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for sentence similarity",
    )
    inputs: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to compare"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.sentence_similarity.SentenceSimilarity"
