from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class FindSegment(GraphNode):
    """
    Extracts a specific segment from a list of segmentation masks.
    image, segmentation, object detection, mask
    """

    segments: list[types.ImageSegmentationResult] | GraphNode | tuple[GraphNode, str] = Field(default={}, description='The segmentation masks to search')
    segment_label: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='The label of the segment to extract')

    @classmethod
    def get_node_type(cls): return "huggingface.image_segmentation.FindSegment"



class SAM2Segmentation(GraphNode):
    """
    Performs semantic segmentation on images using SAM2 (Segment Anything Model 2).
    image, segmentation, object detection, scene parsing, mask

    Use cases:
    - Automatic segmentation of objects in images
    - Instance segmentation for computer vision tasks
    - Interactive segmentation with point prompts
    - Scene understanding and object detection
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(default=types.ImageRef(type='image', uri='', asset_id=None, data=None), description='The input image to segment')

    @classmethod
    def get_node_type(cls): return "huggingface.image_segmentation.SAM2Segmentation"



class Segmentation(GraphNode):
    """
    Performs semantic segmentation on images, identifying and labeling different regions.
    image, segmentation, object detection, scene parsing

    Use cases:
    - Segmenting objects in images
    - Segmenting facial features in images
    """

    model: types.HFImageSegmentation | GraphNode | tuple[GraphNode, str] = Field(default=types.HFImageSegmentation(type='hf.image_segmentation', repo_id='nvidia/segformer-b3-finetuned-ade-512-512', path=None, allow_patterns=None, ignore_patterns=None), description='The model ID to use for the segmentation')
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(default=types.ImageRef(type='image', uri='', asset_id=None, data=None), description='The input image to segment')

    @classmethod
    def get_node_type(cls): return "huggingface.image_segmentation.Segmentation"



class VisualizeSegmentation(GraphNode):
    """
    Visualizes segmentation masks on images with labels.
    image, segmentation, visualization, mask

    Use cases:
    - Visualize results of image segmentation models
    - Analyze and compare different segmentation techniques
    - Create labeled images for presentations or reports
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(default=types.ImageRef(type='image', uri='', asset_id=None, data=None), description='The input image to visualize')
    segments: list[types.ImageSegmentationResult] | GraphNode | tuple[GraphNode, str] = Field(default=[], description='The segmentation masks to visualize')

    @classmethod
    def get_node_type(cls): return "huggingface.image_segmentation.VisualizeSegmentation"


