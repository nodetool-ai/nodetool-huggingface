from __future__ import annotations

from nodetool.metadata.types import HFText2TextGeneration
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger
from pydantic import Field
from typing import Any

logger = get_logger(__name__)


class Text2TextGeneration(HuggingFacePipelineNode):
    """
    Transforms input text into output text using sequence-to-sequence transformer models.
    text, text2text, NLP, seq2seq, transformation

    Use cases:
    - Grammar correction and text refinement
    - Text paraphrasing and rewriting
    - Question generation from context
    - Text simplification for accessibility
    - Code conversion between programming languages
    - General-purpose text transformation tasks
    """

    model: HFText2TextGeneration = Field(
        default=HFText2TextGeneration(
            repo_id="google-t5/t5-base",
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        ),
        title="Model",
        description="The text-to-text generation model. T5 and FLAN-T5 models excel at various text transformation tasks.",
    )
    inputs: str = Field(
        default="",
        title="Input Text",
        description="The input text to transform. For T5 models, prefix with task description (e.g., 'summarize:', 'translate English to German:').",
    )
    max_length: int = Field(
        default=256,
        ge=1,
        le=1024,
        title="Max Length",
        description="Maximum length of the generated output in tokens.",
    )
    do_sample: bool = Field(
        default=False,
        title="Do Sample",
        description="Whether to use sampling instead of greedy decoding. Enable for more creative outputs.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        title="Temperature",
        description="Controls randomness when sampling. Lower values produce more focused outputs.",
    )

    _pipeline: Any = None

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "inputs"]

    @classmethod
    def get_recommended_models(cls) -> list[HFText2TextGeneration]:
        return [
            HFText2TextGeneration(
                repo_id="google-t5/t5-base",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFText2TextGeneration(
                repo_id="google-t5/t5-small",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFText2TextGeneration(
                repo_id="google-t5/t5-large",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFText2TextGeneration(
                repo_id="google/flan-t5-base",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFText2TextGeneration(
                repo_id="google/flan-t5-small",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFText2TextGeneration(
                repo_id="google/flan-t5-large",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        """Initialize the text2text pipeline by loading the specified model."""
        try:
            self._pipeline = await self.load_pipeline(
                context,
                "text2text-generation",
                self.model.repo_id,
                torch_dtype=select_inference_dtype(),
            )
            logger.info(f"Pipeline loaded with model {self.model.repo_id}")
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise RuntimeError(f"Failed to load pipeline: {e}") from e

    async def move_to_device(self, device: str):
        """Move the pipeline's model to the specified device."""
        import torch
        from transformers import PreTrainedModel

        if self._pipeline is None:
            raise ValueError("Pipeline is not initialized.")

        try:
            target_device = torch.device(device)
            self._pipeline.device = target_device

            pipeline_model = getattr(self._pipeline, "model", None)
            if isinstance(pipeline_model, PreTrainedModel):
                pipeline_model.to(target_device)
                logger.info(f"Model moved to {target_device}")
        except Exception as e:
            logger.error(f"Error moving pipeline to device {device}: {e}")
            raise RuntimeError(
                f"Failed to move pipeline to device {device}: {e}"
            ) from e

    async def process(self, context: ProcessingContext) -> str:
        """Process the input text through the text2text pipeline and return the generated text."""
        if self._pipeline is None:
            raise ValueError("Pipeline is not initialized.")

        if not self.inputs.strip():
            return ""

        try:
            kwargs: dict[str, Any] = {
                "max_length": self.max_length,
                "do_sample": self.do_sample,
            }

            if self.do_sample:
                kwargs["temperature"] = self.temperature

            result = await self.run_pipeline_in_thread(self.inputs, **kwargs)

            if result is None:
                raise ValueError("Text2text generation result is None.")

            # Handle different types of result
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if isinstance(first_result, dict):
                    generated_text = first_result.get("generated_text")
                    if generated_text:
                        return generated_text
                    raise ValueError("No 'generated_text' found in the result.")
                elif isinstance(first_result, str):
                    return first_result
            elif isinstance(result, dict):
                generated_text = result.get("generated_text")
                if generated_text:
                    return generated_text
                raise ValueError("No 'generated_text' found in the result.")

            raise TypeError(f"Unexpected result type: {type(result)}")

        except Exception as e:
            logger.error(f"Error during text2text generation: {e}")
            raise RuntimeError(f"Failed to process text2text generation: {e}") from e
