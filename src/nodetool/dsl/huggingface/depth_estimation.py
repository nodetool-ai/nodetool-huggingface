from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class DepthEstimation(GraphNode):
    """
    Estimates depth from a single image.
    image, depth estimation, 3D, huggingface

    Use cases:
    - Generate depth maps for 3D modeling
    - Assist in augmented reality applications
    - Enhance computer vision systems for robotics
    - Improve scene understanding in autonomous vehicles
    """

    model: types.HFDepthEstimation | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFDepthEstimation(
            type="hf.depth_estimation",
            repo_id="LiheYoung/depth-anything-base-hf",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for depth estimation",
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image for depth estimation",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.depth_estimation.DepthEstimation"
