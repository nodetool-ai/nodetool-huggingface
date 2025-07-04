from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class FeatureExtraction(GraphNode):
    """
    Extracts features from text using pre-trained models.
    text, feature extraction, embeddings, natural language processing

    Use cases:
    - Text similarity comparison
    - Clustering text documents
    - Input for machine learning models
    - Semantic search applications
    """

    model: types.HFFeatureExtraction | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFFeatureExtraction(
            type="hf.feature_extraction",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for feature extraction",
    )
    inputs: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to extract features from"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.feature_extraction.FeatureExtraction"
