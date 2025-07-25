from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ObjectDetection(GraphNode):
    """
    Detects and localizes objects in images.
    image, object detection, bounding boxes, huggingface

    Use cases:
    - Identify and count objects in images
    - Locate specific items in complex scenes
    - Assist in autonomous vehicle vision systems
    - Enhance security camera footage analysis
    """

    model: types.HFObjectDetection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFObjectDetection(
            type="hf.object_detection",
            repo_id="facebook/detr-resnet-50",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for object detection",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image for object detection",
    )
    threshold: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.9, description="Minimum confidence score for detected objects"
    )
    top_k: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="The number of top predictions to return"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.object_detection.ObjectDetection"


class VisualizeObjectDetection(GraphNode):
    """
    Visualizes object detection results on images.
    image, object detection, bounding boxes, visualization, mask
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image to visualize",
    )
    objects: list[types.ObjectDetectionResult] | GraphNode | tuple[GraphNode, str] = (
        Field(default={}, description="The detected objects to visualize")
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.object_detection.VisualizeObjectDetection"


class ZeroShotObjectDetection(GraphNode):
    """
    Detects objects in images without the need for training data.
    image, object detection, bounding boxes, zero-shot, mask

    Use cases:
    - Quickly detect objects in images without training data
    - Identify objects in images without predefined labels
    - Automate object detection for large datasets
    """

    model: types.HFZeroShotObjectDetection | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFZeroShotObjectDetection(
            type="hf.zero_shot_object_detection",
            repo_id="google/owlv2-base-patch16",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for object detection",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image for object detection",
    )
    threshold: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.1, description="Minimum confidence score for detected objects"
    )
    top_k: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="The number of top predictions to return"
    )
    candidate_labels: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The candidate labels to detect in the image, separated by commas",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.object_detection.ZeroShotObjectDetection"
