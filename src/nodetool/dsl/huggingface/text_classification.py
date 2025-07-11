from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class TextClassifier(GraphNode):
    """
    Classifies text into predefined categories using a Hugging Face model.
    text, classification, zero-shot, natural language processing
    """

    model: types.HFTextClassification | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFTextClassification(
            type="hf.text_classification",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for the classification",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The input text to the model"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_classification.TextClassifier"


class ZeroShotTextClassifier(GraphNode):
    """
    Performs zero-shot classification on text.
    text, classification, zero-shot, natural language processing

    Use cases:
    - Classify text into custom categories without training
    - Topic detection in documents
    - Sentiment analysis with custom sentiment labels
    - Intent classification in conversational AI
    """

    model: types.HFZeroShotClassification | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFZeroShotClassification(
            type="hf.zero_shot_classification",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for zero-shot classification",
    )
    inputs: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to classify"
    )
    candidate_labels: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Comma-separated list of candidate labels for classification",
    )
    multi_label: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Whether to perform multi-label classification"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_classification.ZeroShotTextClassifier"
