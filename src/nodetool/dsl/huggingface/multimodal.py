from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ImageToText(GraphNode):
    """
    Generates text descriptions from images.
    image, text, captioning, vision-language

    Use cases:
    - Automatic image captioning
    - Assisting visually impaired users
    - Enhancing image search capabilities
    - Generating alt text for web images
    """

    model: types.HFImageToText | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFImageToText(
            type="hf.image_to_text",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for image-to-text generation",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to generate text from",
    )
    max_new_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=50, description="The maximum number of tokens to generate"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.multimodal.ImageToText"


class VisualQuestionAnswering(GraphNode):
    """
    Answers questions about images.
    image, text, question answering, multimodal

    Use cases:
    - Image content analysis
    - Automated image captioning
    - Visual information retrieval
    - Accessibility tools for visually impaired users
    """

    model: types.HFVisualQuestionAnswering | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFVisualQuestionAnswering(
            type="hf.visual_question_answering",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for visual question answering",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to analyze",
    )
    question: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The question to be answered about the image"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.multimodal.VisualQuestionAnswering"
