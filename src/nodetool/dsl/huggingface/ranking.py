from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Reranker(GraphNode):
    """
    Reranks pairs of text based on their semantic similarity.
    text, ranking, reranking, natural language processing

    Use cases:
    - Improve search results ranking
    - Question-answer pair scoring
    - Document relevance ranking
    """

    model: types.HFReranker | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFReranker(
            type="hf.reranker",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for reranking",
    )
    query: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The query text to compare against candidates"
    )
    candidates: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of candidate texts to rank"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.ranking.Reranker"
