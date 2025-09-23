from typing import Any

from pydantic import Field

from nodetool.metadata.types import HFImageTextToText, ImageRef
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext


class LoadImageTextToTextModel(HuggingFacePipelineNode):
    """
    Load a Hugging Face image-text-to-text model/pipeline by repo_id.

    Use cases:
    - Produces a configurable `HFImageTextToText` model reference for downstream nodes
    - Ensures the selected model can be loaded with the "image-text-to-text" task
    """

    repo_id: str = Field(
        default="",
        title="Model ID on Hugging Face",
        description="The model repository ID to use for image-text-to-text generation.",
    )

    async def preload_model(self, context: ProcessingContext):
        if not self.repo_id:
            raise ValueError("Model ID is required")
        self._pipeline = await self.load_pipeline(
            context, "image-text-to-text", self.repo_id
        )

    async def process(self, context: ProcessingContext) -> HFImageTextToText:
        return HFImageTextToText(
            repo_id=self.repo_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        )


class ImageTextToText(HuggingFacePipelineNode):
    """
    Answers questions or follows instructions given both an image and text.
    image, text, visual question answering, multimodal, VLM

    Use cases:
    - Visual question answering with free-form reasoning
    - Zero-shot object localization or structure extraction via instructions
    - OCR-free document understanding when combined with prompts
    - Multi-turn, instruction-following conversations grounded in an image
    """

    model: HFImageTextToText = Field(
        default=HFImageTextToText(),
        title="Model",
        description="The image-text-to-text model to use.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The image to analyze.",
    )
    prompt: str = Field(
        default="Describe this image.",
        title="Prompt",
        description="Instruction or question for the model about the image.",
    )
    max_new_tokens: int = Field(
        default=256,
        title="Max New Tokens",
        description="Maximum number of tokens to generate.",
        ge=1,
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "image", "prompt"]

    @classmethod
    def get_recommended_models(cls):
        return [
            HFImageTextToText(
                repo_id="HuggingFaceTB/SmolVLM-Instruct",
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            ),
            HFImageTextToText(
                repo_id="zai-org/GLM-4.5V",
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            ),
            HFImageTextToText(
                repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            ),
            HFImageTextToText(
                repo_id="llava-hf/llava-interleave-qwen-0.5b-hf",
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    async def preload_model(self, context: ProcessingContext):
        if not self.model.repo_id:
            raise ValueError("Model ID is required")
        self._pipeline = await self.load_pipeline(
            context, "image-text-to-text", self.model.repo_id
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            try:
                # Many pipelines expose an inner model attribute
                self._pipeline.model.to(device)  # type: ignore
            except AttributeError:
                try:
                    # Fallback for pipelines that implement .to()
                    self._pipeline.to(device)  # type: ignore
                except Exception:
                    pass

    async def process(self, context: ProcessingContext) -> str:
        assert self._pipeline is not None

        image_pil = await context.image_to_pil(self.image)

        # Follow HF task API: compose a single user message containing image + text
        # Reference: https://huggingface.co/tasks/image-text-to-text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        # Most models use `text=` for the messages payload per HF examples
        outputs: Any = self._pipeline(  # type: ignore
            text=messages,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )

        # Normalize output across variants
        if isinstance(outputs, list) and len(outputs) > 0:
            first = outputs[0]
            if isinstance(first, dict):
                for key in [
                    "generated_text",
                    "answer",
                    "text",
                    "output_text",
                ]:
                    if key in first and isinstance(first[key], str):
                        return first[key]  # type: ignore
            if isinstance(first, str):
                return first

        return str(outputs)
