from nodetool.metadata.types import HFImageToText, ImageRef
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


class LoadImageToTextModel(HuggingFacePipelineNode):
    """
    Loads and validates a Hugging Face image-to-text model for use in downstream nodes.
    model-loader, captioning, OCR, image-to-text

    Use cases:
    - Pre-load image captioning models before running pipelines
    - Validate model availability and compatibility
    - Configure model settings for ImageToText processing
    """

    repo_id: str = Field(
        default="Salesforce/blip-image-captioning-base",
        title="Model ID",
        description="The Hugging Face repository ID for the image-to-text model (e.g., BLIP, GIT, ViT-GPT2).",
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
    Generates textual descriptions and captions from images using vision-language models.
    image, captioning, OCR, image-to-text, accessibility

    Use cases:
    - Automatically generate captions for photos and artwork
    - Extract visible text from images (OCR-style functionality)
    - Create alt-text descriptions for web accessibility
    - Build image search engines with text-based queries
    - Generate descriptions for visually impaired users
    """

    model: HFImageToText = Field(
        default=HFImageToText(
            repo_id="Salesforce/blip-image-captioning-base",
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        ),
        title="Model",
        description="The image captioning model. BLIP models offer good quality; BLIP2 provides enhanced understanding; GIT is faster.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The image to generate text from. Supports JPEG, PNG, WebP and other common formats.",
    )
    max_new_tokens: int = Field(
        default=1024,
        title="Max New Tokens",
        description="Maximum length of the generated caption in tokens. Higher values allow longer descriptions.",
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
