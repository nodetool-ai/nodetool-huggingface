from __future__ import annotations

import asyncio
from typing import Any, Sequence

from pydantic import Field

from nodetool.metadata.types import (
    HFComputerVision,
    HFObjectDetection,
    ImageRef,
)
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

# COCO 17-keypoint layout used by all ViTPose checkpoints trained on COCO.
COCO_KEYPOINT_LABELS: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Bone connections between COCO keypoint indices, used for skeleton rendering.
COCO_SKELETON: list[tuple[int, int]] = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]

# Per-bone colors (RGB) so left/right/torso limbs stay distinguishable.
_SKELETON_COLORS: list[tuple[int, int, int]] = [
    (255, 128, 0),
    (255, 153, 51),
    (255, 178, 102),
    (230, 230, 0),
    (255, 153, 255),
    (153, 204, 255),
    (255, 102, 255),
    (255, 51, 255),
    (102, 178, 255),
    (51, 153, 255),
    (255, 153, 153),
    (255, 102, 102),
    (255, 51, 51),
    (153, 255, 153),
    (102, 255, 102),
    (51, 255, 51),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
]


def to_list(value: Any) -> Any:
    """Convert torch tensors / numpy arrays into plain nested Python lists.

    Plain lists and tuples pass through unchanged, which keeps the pure helpers
    below testable without torch installed.
    """
    if value is None:
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, (list, tuple)):
        return [to_list(item) for item in value]
    return value


def xyxy_to_xywh(boxes: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convert ``(x1, y1, x2, y2)`` detector boxes to the COCO ``(x, y, w, h)``
    format expected by the ViTPose processor."""
    converted: list[list[float]] = []
    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in list(box)[:4])
        converted.append([x1, y1, x2 - x1, y2 - y1])
    return converted


def person_label_ids(config: Any) -> set[int]:
    """Return the label ids that mean "person" for a detector config.

    Falls back to ``{0}`` (the COCO person index) when the config carries no
    usable ``id2label`` mapping.
    """
    id2label = getattr(config, "id2label", None) or {}
    ids: set[int] = set()
    for key, label in id2label.items():
        try:
            label_id = int(key)
        except (TypeError, ValueError):
            continue
        if str(label).strip().lower() == "person":
            ids.add(label_id)
    return ids or {0}


def is_moe_model(model: Any, model_id: str = "") -> bool:
    """ViTPose+ checkpoints use a mixture-of-experts backbone whose forward pass
    requires an explicit ``dataset_index``."""
    if "plus" in (model_id or "").lower():
        return True
    backbone_config = getattr(getattr(model, "config", None), "backbone_config", None)
    try:
        return int(getattr(backbone_config, "num_experts", 1)) > 1
    except (TypeError, ValueError):
        return False


def pose_result_to_dict(
    result: Any,
    keypoint_threshold: float = 0.0,
    labels: Sequence[str] = COCO_KEYPOINT_LABELS,
) -> dict[str, Any]:
    """Normalize one entry of ``post_process_pose_estimation`` into plain data.

    The processor returns ``keypoints`` (N x 2), ``scores`` (N) and optionally
    ``labels`` (N) and ``bbox``, as tensors. The result is a JSON-serializable
    dict with one entry per keypoint above ``keypoint_threshold``.
    """
    keypoints = to_list(result.get("keypoints")) or []
    scores = to_list(result.get("scores")) or []
    keypoint_labels = to_list(result.get("labels")) or []

    points: list[dict[str, Any]] = []
    for index, point in enumerate(keypoints):
        score = float(scores[index]) if index < len(scores) else 1.0
        if score < keypoint_threshold:
            continue
        if index < len(keypoint_labels):
            label_index = int(keypoint_labels[index])
        else:
            label_index = index
        label = (
            labels[label_index] if 0 <= label_index < len(labels) else str(label_index)
        )
        points.append(
            {
                "index": label_index,
                "label": label,
                "x": float(point[0]),
                "y": float(point[1]),
                "score": score,
            }
        )

    pose: dict[str, Any] = {"keypoints": points}
    bbox = to_list(result.get("bbox"))
    if bbox is not None and len(bbox) >= 4:
        pose["bbox"] = {
            "x": float(bbox[0]),
            "y": float(bbox[1]),
            "width": float(bbox[2]),
            "height": float(bbox[3]),
        }
    if scores:
        pose["score"] = float(sum(float(s) for s in scores) / len(scores))
    return pose


class PoseEstimation(HuggingFacePipelineNode):
    """
    Detects human body keypoints in images using ViTPose top-down pose estimation.
    image, pose-estimation, keypoints, skeleton, human, vitpose, huggingface, computer-vision

    Use cases:
    - Analyze athletic form and movement in sports footage
    - Drive character animation and motion capture from ordinary photos
    - Monitor ergonomics, physiotherapy exercises or fall detection
    - Build gesture and action recognition on top of skeleton data
    - Generate pose conditioning images for ControlNet workflows
    """

    model: HFComputerVision = Field(
        default=HFComputerVision(repo_id="usyd-community/vitpose-base-simple"),
        title="Model",
        description="The ViTPose model used for keypoint estimation. The 'simple' variant is fast; the 'plus' variants are mixture-of-experts models with higher accuracy.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Image",
        description="The image to estimate human poses in. Supports common formats like JPEG, PNG.",
    )
    person_detector: HFObjectDetection = Field(
        default=HFObjectDetection(repo_id="PekingU/rtdetr_r50vd_coco_o365"),
        title="Person Detector",
        description="ViTPose is a top-down model, so persons are located first with this object detection model.",
    )
    detection_threshold: float = Field(
        default=0.3,
        title="Detection Threshold",
        description="Minimum confidence score (0-1) for a person detection to be passed to the pose model.",
    )
    keypoint_threshold: float = Field(
        default=0.3,
        title="Keypoint Threshold",
        description="Minimum confidence score (0-1) for an individual keypoint to be included in the output.",
    )

    _pipeline: Any = None
    _processor: Any = None
    _detector: Any = None
    _detector_processor: Any = None

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "image"]

    @classmethod
    def get_recommended_models(cls) -> list[HFComputerVision]:
        return [
            HFComputerVision(
                repo_id="usyd-community/vitpose-base-simple",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFComputerVision(
                repo_id="usyd-community/vitpose-plus-small",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFComputerVision(
                repo_id="usyd-community/vitpose-plus-base",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFComputerVision(
                repo_id="usyd-community/vitpose-plus-large",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFComputerVision(
                repo_id="usyd-community/vitpose-plus-huge",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls) -> str:
        return "HF Pose Estimation"

    def get_model_id(self) -> str:
        return self.model.repo_id

    def get_detector_model_id(self) -> str:
        return self.person_detector.repo_id

    async def preload_model(self, context: ProcessingContext):
        from transformers import (
            AutoModelForObjectDetection,
            AutoProcessor,
            VitPoseForPoseEstimation,
        )

        if not self.get_model_id():
            raise ValueError("Model ID is required")
        if not self.get_detector_model_id():
            raise ValueError("Person detector model ID is required")

        dtype = select_inference_dtype()

        self._pipeline = await self.load_model(
            context=context,
            model_class=VitPoseForPoseEstimation,
            model_id=self.get_model_id(),
            torch_dtype=dtype,
        )
        self._processor = await self.load_model(
            context=context,
            model_class=AutoProcessor,
            model_id=self.get_model_id(),
        )
        self._detector = await self.load_model(
            context=context,
            model_class=AutoModelForObjectDetection,
            model_id=self.get_detector_model_id(),
            torch_dtype=dtype,
        )
        self._detector_processor = await self.load_model(
            context=context,
            model_class=AutoProcessor,
            model_id=self.get_detector_model_id(),
        )

        await self.move_to_device(context.device)

    async def move_to_device(self, device: str):
        if not device:
            return
        if self._pipeline is not None:
            self._pipeline.to(device)
        if self._detector is not None:
            self._detector.to(device)

    async def detect_persons(self, image: Any) -> list[list[float]]:
        """Run the person detector and return boxes in COCO ``(x, y, w, h)`` form."""
        import torch

        assert self._detector is not None, "Person detector not initialized"
        assert (
            self._detector_processor is not None
        ), "Person detector processor not initialized"

        detector = self._detector
        processor = self._detector_processor

        def _detect() -> list[list[float]]:
            # Weights may be half precision, so the float inputs are cast to
            # match; BatchFeature.to only casts floating point tensors.
            inputs = processor(images=image, return_tensors="pt").to(
                device=detector.device, dtype=detector.dtype
            )
            with torch.inference_mode():
                outputs = detector(**inputs)
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([(image.height, image.width)]),
                threshold=self.detection_threshold,
            )
            result = results[0]
            allowed = person_label_ids(detector.config)
            boxes = to_list(result["boxes"])
            labels = to_list(result["labels"])
            return xyxy_to_xywh(
                [box for box, label in zip(boxes, labels) if int(label) in allowed]
            )

        return await asyncio.to_thread(_detect)

    async def process(self, context: ProcessingContext) -> list[dict[str, Any]]:
        import torch

        assert self._pipeline is not None, "Model not initialized"
        assert self._processor is not None, "Processor not initialized"

        image = await context.image_to_pil(self.image)
        image = image.convert("RGB")

        person_boxes = await self.detect_persons(image)
        if not person_boxes:
            return []

        inputs = self._processor(image, boxes=[person_boxes], return_tensors="pt").to(
            device=self._pipeline.device, dtype=self._pipeline.dtype
        )
        if is_moe_model(self._pipeline, self.get_model_id()):
            # ViTPose+ selects a backbone expert per dataset; 0 is COCO.
            dataset_index = torch.zeros(
                len(inputs["pixel_values"]),
                dtype=torch.int64,
                device=inputs["pixel_values"].device,
            )
            inputs["dataset_index"] = dataset_index

        outputs = await self.run_pipeline_in_thread(**inputs)

        pose_results = self._processor.post_process_pose_estimation(
            outputs, boxes=[person_boxes]
        )
        if not pose_results:
            return []
        return [
            pose_result_to_dict(result, self.keypoint_threshold)
            for result in pose_results[0]
        ]


class VisualizePoseEstimation(BaseNode):
    """
    Renders estimated human poses as keypoints and skeleton lines on the original image.
    image, pose-estimation, keypoints, skeleton, visualization, annotation

    Use cases:
    - Visually verify pose estimation output quality
    - Produce annotated stills for documentation or presentations
    - Debug keypoint confidence thresholds
    - Create pose overlays for coaching and physiotherapy feedback
    """

    image: ImageRef = Field(
        default=ImageRef(),
        title="Image",
        description="The original image to draw the poses on.",
    )
    poses: list[dict[str, Any]] = Field(
        default=[],
        title="Poses",
        description="List of poses from the Pose Estimation node.",
    )
    keypoint_threshold: float = Field(
        default=0.3,
        title="Keypoint Threshold",
        description="Minimum keypoint confidence (0-1) required to draw a point or bone.",
    )
    point_radius: int = Field(
        default=4,
        title="Point Radius",
        description="Radius in pixels of the drawn keypoint markers.",
    )
    line_width: int = Field(
        default=3,
        title="Line Width",
        description="Width in pixels of the drawn skeleton bones.",
    )
    draw_boxes: bool = Field(
        default=False,
        title="Draw Boxes",
        description="Also draw the person bounding box that produced each pose.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "poses"]

    def required_inputs(self):
        return ["image", "poses"]

    @classmethod
    def get_title(cls) -> str:
        return "Visualize Pose Estimation"

    async def process(self, context: ProcessingContext) -> ImageRef:
        from PIL import ImageDraw

        image = await context.image_to_pil(self.image)
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)

        for pose in self.poses:
            points: dict[int, tuple[float, float]] = {}
            for keypoint in pose.get("keypoints", []):
                score = float(keypoint.get("score", 1.0))
                if score < self.keypoint_threshold:
                    continue
                index = keypoint.get("index")
                if index is None:
                    label = keypoint.get("label")
                    index = (
                        COCO_KEYPOINT_LABELS.index(label)
                        if label in COCO_KEYPOINT_LABELS
                        else None
                    )
                if index is None:
                    continue
                points[int(index)] = (float(keypoint["x"]), float(keypoint["y"]))

            for bone_index, (start, end) in enumerate(COCO_SKELETON):
                if start in points and end in points:
                    draw.line(
                        [points[start], points[end]],
                        fill=_SKELETON_COLORS[bone_index % len(_SKELETON_COLORS)],
                        width=self.line_width,
                    )

            for index, (x, y) in points.items():
                radius = self.point_radius
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=_SKELETON_COLORS[index % len(_SKELETON_COLORS)],
                    outline=(255, 255, 255),
                )

            bbox = pose.get("bbox") if self.draw_boxes else None
            if bbox:
                x = float(bbox["x"])
                y = float(bbox["y"])
                draw.rectangle(
                    [x, y, x + float(bbox["width"]), y + float(bbox["height"])],
                    outline=(0, 255, 0),
                    width=max(1, self.line_width - 1),
                )

        return await context.image_from_pil(image)
