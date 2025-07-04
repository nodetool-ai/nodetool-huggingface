from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Summarize(GraphNode):
    """
    Summarizes text using a Hugging Face model.
    text, summarization, AI, LLM
    """

    model: types.HFTextGeneration | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFTextGeneration(
            type="hf.text_generation",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for the text generation",
    )
    inputs: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The input text to summarize"
    )
    max_length: int | GraphNode | tuple[GraphNode, str] = Field(
        default=100, description="The maximum length of the generated text"
    )
    do_sample: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Whether to sample from the model"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.summarization.Summarize"
