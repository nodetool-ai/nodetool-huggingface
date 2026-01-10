from nodetool.metadata.types import (
    BoundingBox,
    HFObjectDetection,
    HFZeroShotObjectDetection,
    ImageRef,
    ObjectDetectionResult,
)
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


class ObjectDetection(HuggingFacePipelineNode):
    """
    Detects and localizes objects in images with bounding boxes and confidence scores.
    image, object-detection, bounding-boxes, huggingface, computer-vision

    Use cases:
    - Count and identify objects in photographs and videos
    - Locate specific items in complex scenes for robotics
    - Analyze security camera footage for monitoring systems
    - Detect tables and structures in documents
    - Build automated inventory and inspection systems
    """

    model: HFObjectDetection = Field(
        default=HFObjectDetection(repo_id="facebook/detr-resnet-50"),
        title="Model",
        description="The object detection model. DETR models offer high accuracy; YOLOS variants are faster. Specialized models exist for tables and fashion items.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Image",
        description="The image to detect objects in. Supports common formats like JPEG, PNG.",
    )
    threshold: float = Field(
        default=0.9,
        title="Confidence Threshold",
        description="Minimum confidence score (0-1) for detected objects. Higher values return fewer but more certain detections.",
    )
    top_k: int = Field(
        default=5,
        title="Top K",
        description="Maximum number of detected objects to return, sorted by confidence.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "image"]

    @classmethod
    def get_recommended_models(cls) -> list[HFObjectDetection]:
        return [
            HFObjectDetection(
                repo_id="facebook/detr-resnet-50",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HFObjectDetection(
                repo_id="facebook/detr-resnet-101",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFObjectDetection(
                repo_id="hustvl/yolos-tiny",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFObjectDetection(
                repo_id="hustvl/yolos-small",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFObjectDetection(
                repo_id="microsoft/table-transformer-detection",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HFObjectDetection(
                repo_id="microsoft/table-transformer-structure-recognition-v1.1-all",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HFObjectDetection(
                repo_id="valentinafeve/yolos-fashionpedia",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls) -> str:
        return "Object Detection"

    def get_model_id(self):
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context,
            "object-detection",
            self.get_model_id(),
            device=context.device,
            torch_dtype=select_inference_dtype(),
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> list[ObjectDetectionResult]:
        assert self._pipeline is not None
        image = await context.image_to_pil(self.image)
        result = await self.run_pipeline_in_thread(image, threshold=self.threshold)
        if isinstance(result, list):
            return [
                ObjectDetectionResult(
                    label=item["label"],
                    score=item["score"],
                    box=BoundingBox(
                        xmin=item["box"]["xmin"],
                        ymin=item["box"]["ymin"],
                        xmax=item["box"]["xmax"],
                        ymax=item["box"]["ymax"],
                    ),
                )
                for item in result
            ]
        else:
            raise ValueError(f"Invalid result type: {type(result)}")


class VisualizeObjectDetection(BaseNode):
    """
    Renders object detection results as labeled bounding boxes overlaid on the original image.
    image, object-detection, bounding-boxes, visualization, annotation

    Use cases:
    - Visualize and verify object detection model outputs
    - Create annotated images for documentation and presentations
    - Debug and analyze detection accuracy and coverage
    - Generate labeled images for training data review
    """

    image: ImageRef = Field(
        default=ImageRef(),
        title="Image",
        description="The original image to draw detection boxes on.",
    )

    objects: list[ObjectDetectionResult] = Field(
        default={},
        title="Detected Objects",
        description="List of detected objects from ObjectDetection or ZeroShotObjectDetection nodes.",
    )

    def required_inputs(self):
        return ["image", "objects"]

    @classmethod
    def get_title(cls) -> str:
        return "Visualize Object Detection"

    async def process(self, context: ProcessingContext) -> ImageRef:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import io

        image = await context.image_to_pil(self.image)

        # Get the size of the input image
        width, height = image.size

        # Create figure with the same size as the input image
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100)
        )  # Convert pixels to inches
        ax.imshow(image)

        for obj in self.objects:
            xmin = obj.box.xmin
            ymin = obj.box.ymin
            xmax = obj.box.xmax
            ymax = obj.box.ymax

            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                ymin,
                f"{obj.label} ({obj.score:.2f})",
                color="r",
                fontsize=8,
                backgroundcolor="w",
            )

        ax.axis("off")

        # Remove padding around the image
        plt.tight_layout(pad=0)

        if fig is None:
            raise ValueError("Invalid plot")
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return await context.image_from_bytes(img_bytes.getvalue())


class ZeroShotObjectDetection(HuggingFacePipelineNode):
    """
    Detects objects in images using custom labels without requiring task-specific training.
    image, object-detection, bounding-boxes, zero-shot, flexible

    Use cases:
    - Detect custom objects without training a specialized model
    - Search for specific items described in natural language
    - Build flexible object detection systems with dynamic categories
    - Prototype detection applications with arbitrary object classes
    """

    model: HFZeroShotObjectDetection = Field(
        default=HFZeroShotObjectDetection(repo_id="google/owlv2-base-patch16"),
        title="Model",
        description="The zero-shot detection model. OWL-ViT/OWLv2 models use CLIP for flexible label matching; Grounding-DINO offers strong performance.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Image",
        description="The image to detect objects in.",
    )
    threshold: float = Field(
        default=0.1,
        title="Confidence Threshold",
        description="Minimum confidence score (0-1) for detections. Lower values find more objects but may include false positives.",
    )
    top_k: int = Field(
        default=5,
        title="Top K",
        description="Maximum number of detected objects to return per label.",
    )
    candidate_labels: str = Field(
        default="",
        title="Candidate Labels",
        description="Comma-separated list of object labels to detect (e.g., 'cat,dog,person,car'). Use descriptive phrases for better results.",
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFZeroShotObjectDetection]:
        return [
            HFZeroShotObjectDetection(
                repo_id="google/owlvit-base-patch32",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json", "txt"],
            ),
            HFZeroShotObjectDetection(
                repo_id="google/owlvit-large-patch14",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json", "txt"],
            ),
            HFZeroShotObjectDetection(
                repo_id="google/owlvit-base-patch16",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json", "txt"],
            ),
            HFZeroShotObjectDetection(
                repo_id="google/owlv2-base-patch16",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json", "txt"],
            ),
            HFZeroShotObjectDetection(
                repo_id="google/owlv2-base-patch16-ensemble",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json", "txt"],
            ),
            HFZeroShotObjectDetection(
                repo_id="IDEA-Research/grounding-dino-tiny",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json", "txt"],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls) -> str:
        return "Zero-Shot Object Detection"

    def get_model_id(self):
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context,
            "zero-shot-object-detection",
            self.get_model_id(),
            device=context.device,
            torch_dtype=select_inference_dtype(),
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> list[ObjectDetectionResult]:
        assert self._pipeline is not None
        image = await context.image_to_pil(self.image)
        result = await self.run_pipeline_in_thread(
            image,
            candidate_labels=self.candidate_labels.split(","),
            threshold=self.threshold,
        )
        return [
            ObjectDetectionResult(
                label=item.label,  # type: ignore
                score=item.score,  # type: ignore
                box=BoundingBox(
                    xmin=item.box.xmin,  # type: ignore
                    ymin=item.box.ymin,  # type: ignore
                    xmax=item.box.xmax,  # type: ignore
                    ymax=item.box.ymax,  # type: ignore
                ),
            )
            for item in result  # type: ignore
        ]
