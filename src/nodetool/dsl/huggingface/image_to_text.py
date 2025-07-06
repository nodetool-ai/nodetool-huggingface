from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ImageToText(GraphNode):
    """
    Generates textual descriptions from images.
    image, captioning, OCR, image-to-text

    Use cases:
    - Generate captions for images
    - Extract text from images (OCR)
    - Describe image content for visually impaired users
    - Build accessibility features for visual content
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
        default=1024,
        description="The maximum number of tokens to generate (if supported by model)",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_text.ImageToText"


class LoadImageToTextModel(GraphNode):
    repo_id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The model ID to use for image-to-text generation"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.image_to_text.LoadImageToTextModel"
