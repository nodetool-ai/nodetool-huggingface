from nodetool.metadata.types import HFImageToText, ImageRef
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


class LoadImageToTextModel(HuggingFacePipelineNode):
    repo_id: str = Field(
        default="",
        title="Model ID on Huggingface",
        description="The model ID to use for image-to-text generation",
    )

    async def preload_model(self, context: ProcessingContext):
        if not self.repo_id:
            raise ValueError("Model ID is required")
        self._pipeline = await self.load_pipeline(
            context, "image-to-text", self.repo_id
        )

    async def process(self, context: ProcessingContext) -> HFImageToText:
        return HFImageToText(
            repo_id=self.repo_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        )


class ImageToText(HuggingFacePipelineNode):
    """
    Generates textual descriptions from images.
    image, captioning, OCR, image-to-text

    Use cases:
    - Generate captions for images
    - Extract text from images (OCR)
    - Describe image content for visually impaired users
    - Build accessibility features for visual content
    """

    model: HFImageToText = Field(
        default=HFImageToText(),
        title="Model ID on Huggingface",
        description="The model ID to use for image-to-text generation",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The image to generate text from",
    )
    max_new_tokens: int = Field(
        default=1024,
        title="Max New Tokens",
        description="The maximum number of tokens to generate (if supported by model)",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "image"]

    @classmethod
    def get_recommended_models(cls):
        return [
            HFImageToText(
                repo_id="Salesforce/blip-image-captioning-base",
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            ),
            HFImageToText(
                repo_id="Salesforce/blip2-opt-2.7b",
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            ),
            HFImageToText(
                repo_id="microsoft/git-base",
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            ),
            HFImageToText(
                repo_id="nlpconnect/vit-gpt2-image-captioning",
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context, "image-to-text", self.model.repo_id
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            try:
                self._pipeline.model.to(device)  # type: ignore
            except AttributeError:
                # Some models might have a different structure
                try:
                    self._pipeline.to(device)  # type: ignore
                except:
                    # If both approaches fail, we'll continue without moving the model
                    pass

    async def process(self, context: ProcessingContext) -> str:
        assert self._pipeline is not None
        image = await context.image_to_pil(self.image)

        kwargs = {}
        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.max_new_tokens

        result = await self.run_pipeline_in_thread(image, **kwargs)  # type: ignore

        # Handle different output formats from different models
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                return result[0]["generated_text"]  # type: ignore
            elif isinstance(result[0], str):
                return result[0]  # type: ignore

        # Fallback for other formats
        return str(result)  # type: ignore
