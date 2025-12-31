from __future__ import annotations

from nodetool.metadata.types import HFSummarization
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger
from pydantic import Field
from typing import Any

logger = get_logger(__name__)


class Summarize(HuggingFacePipelineNode):
    """
    Generates concise summaries of long text using sequence-to-sequence transformer models.
    text, summarization, NLP, abstractive, extractive

    Use cases:
    - Summarize long articles or documents into brief overviews
    - Create executive summaries of reports and papers
    - Generate social media-friendly summaries from long-form content
    - Condense meeting transcripts into action points
    - Build automated news summarization pipelines
    """

    model: HFSummarization = Field(
        default=HFSummarization(
            repo_id="facebook/bart-large-cnn",
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
        ),
        title="Model",
        description="The summarization model. BART and T5 models are commonly used for summarization tasks.",
    )
    inputs: str = Field(
        default="",
        title="Input Text",
        description="The long text to summarize.",
    )
    max_length: int = Field(
        default=150,
        ge=10,
        le=1024,
        title="Max Length",
        description="Maximum length of the summary in tokens.",
    )
    min_length: int = Field(
        default=30,
        ge=1,
        le=512,
        title="Min Length",
        description="Minimum length of the summary in tokens.",
    )
    do_sample: bool = Field(
        default=False,
        title="Do Sample",
        description="Whether to use sampling instead of greedy decoding.",
    )

    _pipeline: Any = None

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "inputs"]

    @classmethod
    def get_recommended_models(cls) -> list[HFSummarization]:
        return [
            HFSummarization(
                repo_id="facebook/bart-large-cnn",
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
            ),
            HFSummarization(
                repo_id="google/pegasus-xsum",
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
            ),
            HFSummarization(
                repo_id="philschmid/bart-large-cnn-samsum",
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
            ),
            HFSummarization(
                repo_id="google-t5/t5-base",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFSummarization(
                repo_id="sshleifer/distilbart-cnn-12-6",
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        """Initialize the summarization pipeline by loading the specified model."""
        try:
            self._pipeline = await self.load_pipeline(
                context,
                "summarization",
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
        """Process the input text through the summarization pipeline and return the summary."""
        if self._pipeline is None:
            raise ValueError("Pipeline is not initialized.")

        if not self.inputs.strip():
            return ""

        try:
            result = await self.run_pipeline_in_thread(
                self.inputs,
                max_length=self.max_length,
                min_length=self.min_length,
                do_sample=self.do_sample,
            )

            if result is None:
                raise ValueError("Summarization result is None.")

            # Handle different types of result
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if isinstance(first_result, dict):
                    summary_text = first_result.get("summary_text")
                    if summary_text:
                        return summary_text
                    raise ValueError("No 'summary_text' found in the result.")
                elif isinstance(first_result, str):
                    return first_result
            elif isinstance(result, dict):
                summary_text = result.get("summary_text")
                if summary_text:
                    return summary_text
                raise ValueError("No 'summary_text' found in the result.")

            raise TypeError(f"Unexpected result type: {type(result)}")

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise RuntimeError(f"Failed to process summarization: {e}") from e
