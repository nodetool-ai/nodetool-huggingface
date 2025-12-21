from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nodetool.metadata.types import (
    HFImageClassification,
    HFZeroShotImageClassification,
    ImageRef,
)
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field

if TYPE_CHECKING:
    from transformers import ImageClassificationPipeline


class ImageClassifier(HuggingFacePipelineNode):
    """
    Classifies images into predefined categories using vision transformer models.
    image, classification, labeling, categorization, computer-vision

    Use cases:
    - Automatically tag and organize photo libraries
    - Detect inappropriate or NSFW content for moderation
    - Classify product images in e-commerce catalogs
    - Identify age, gender, or other attributes in photos
    - Sort images by scene type, object presence, or style
    """

    model: HFImageClassification = Field(
        default=HFImageClassification(),
        title="Model",
        description="The image classification model. ViT and ResNet models offer general classification; specialized models exist for NSFW detection, age estimation, etc.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Image",
        description="The image to classify. Supports common formats like JPEG, PNG, WebP.",
    )
    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HFImageClassification]:
        return [
            HFImageClassification(
                repo_id="google/vit-base-patch16-224",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFImageClassification(
                repo_id="microsoft/resnet-50",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFImageClassification(
                repo_id="microsoft/resnet-18",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFImageClassification(
                repo_id="apple/mobilevit-small",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HFImageClassification(
                repo_id="apple/mobilevit-xx-small",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HFImageClassification(
                repo_id="nateraw/vit-age-classifier",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFImageClassification(
                repo_id="Falconsai/nsfw_image_detection",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HFImageClassification(
                repo_id="rizvandwiki/gender-classification-2",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    def get_model_id(self):
        return self.model.repo_id

    @classmethod
    def get_title(cls) -> str:
        return "Image Classifier"

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)  # type: ignore

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="image-classification",
            model_id=self.get_model_id(),
            device=context.device,
            torch_dtype=select_inference_dtype(),
        )  # type: ignore

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        image = await context.image_to_pil(self.image)
        result = await self.run_pipeline_in_thread(image)  # type: ignore
        return {str(item["label"]): float(item["score"]) for item in result}  # type: ignore


class ZeroShotImageClassifier(HuggingFacePipelineNode):
    """
    Classifies images into custom categories without requiring task-specific training data.
    image, classification, labeling, categorization, zero-shot, flexible

    Use cases:
    - Categorize images with custom, user-defined labels on the fly
    - Quickly prototype image classification systems without training
    - Identify objects or scenes without predefined model categories
    - Build flexible image tagging workflows with dynamic categories
    - Test hypotheses about image content using natural language labels
    """

    model: HFZeroShotImageClassification = Field(
        default=HFZeroShotImageClassification(),
        title="Model",
        description="The zero-shot classification model. CLIP-based models (OpenAI, LAION) enable flexible label matching using vision-language understanding.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Image",
        description="The image to classify. Supports common formats like JPEG, PNG, WebP.",
    )
    candidate_labels: str = Field(
        default="",
        title="Candidate Labels",
        description="Comma-separated list of labels to classify against (e.g., 'cat,dog,bird,fish'). Use descriptive labels for better results.",
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFZeroShotImageClassification]:
        return [
            HFZeroShotImageClassification(
                repo_id="openai/clip-vit-base-patch32",
                allow_patterns=["README.md", "pytorch_model.bin", "*.json", "*.txt"],
            ),
            HFZeroShotImageClassification(
                repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                allow_patterns=["README.md", "pytorch_model.bin", "*.json", "*.txt"],
            ),
            HFZeroShotImageClassification(
                repo_id="laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
                allow_patterns=["README.md", "pytorch_model.bin", "*.json", "*.txt"],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls) -> str:
        return "Zero-Shot Image Classifier"

    def get_model_id(self):
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="zero-shot-image-classification",
            model_id=self.get_model_id(),
            device=context.device,
            torch_dtype=select_inference_dtype(),
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        image = await context.image_to_pil(self.image)
        result = await self.run_pipeline_in_thread(
            image, candidate_labels=self.candidate_labels.split(",")
        )  # type: ignore
        return {str(item["label"]): float(item["score"]) for item in result}  # type: ignore
