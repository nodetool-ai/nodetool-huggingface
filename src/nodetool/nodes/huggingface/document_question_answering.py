from __future__ import annotations

from nodetool.metadata.types import HFDocumentQuestionAnswering, ImageRef
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger
from pydantic import Field
from typing import Any, TypedDict

logger = get_logger(__name__)


class DocumentQuestionAnsweringOutput(TypedDict):
    answer: str
    score: float
    start: int
    end: int


class DocumentQuestionAnswering(HuggingFacePipelineNode):
    """
    Answers questions about document images by understanding both visual layout and text content.
    document, question-answering, OCR, multimodal, document-understanding

    Use cases:
    - Extract specific information from scanned documents
    - Answer questions about forms, invoices, and receipts
    - Automate document processing workflows
    - Build document search and retrieval systems
    - Enable accessibility features for document images
    """

    model: HFDocumentQuestionAnswering = Field(
        default=HFDocumentQuestionAnswering(
            repo_id="impira/layoutlm-document-qa",
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
        ),
        title="Model",
        description="The document QA model. LayoutLM models understand both text and document structure.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Document Image",
        description="The document image to analyze and answer questions about.",
    )
    question: str = Field(
        default="",
        title="Question",
        description="The question to answer about the document.",
    )
    top_k: int = Field(
        default=1,
        ge=1,
        le=10,
        title="Top K",
        description="Number of answer candidates to return.",
    )

    _pipeline: Any = None

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "image", "question"]

    @classmethod
    def get_recommended_models(cls) -> list[HFDocumentQuestionAnswering]:
        return [
            HFDocumentQuestionAnswering(
                repo_id="impira/layoutlm-document-qa",
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
            ),
            HFDocumentQuestionAnswering(
                repo_id="impira/layoutlm-invoices",
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
            ),
            HFDocumentQuestionAnswering(
                repo_id="naver-clova-ix/donut-base-finetuned-docvqa",
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    async def preload_model(self, context: ProcessingContext):
        """Initialize the document QA pipeline by loading the specified model."""
        try:
            self._pipeline = await self.load_pipeline(
                context,
                "document-question-answering",
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

    async def process(
        self, context: ProcessingContext
    ) -> list[DocumentQuestionAnsweringOutput]:
        """Process the document image and question through the pipeline and return answers."""
        if self._pipeline is None:
            raise ValueError("Pipeline is not initialized.")

        if not self.question.strip():
            raise ValueError("Question must not be empty.")

        try:
            # Load the document image
            image = await context.image_to_pil(self.image)

            result = await self.run_pipeline_in_thread(
                image,
                question=self.question,
                top_k=self.top_k,
            )

            if result is None:
                raise ValueError("Document QA result is None.")

            # Normalize result to list
            if isinstance(result, dict):
                result = [result]

            outputs: list[DocumentQuestionAnsweringOutput] = []
            for item in result:
                if isinstance(item, dict):
                    outputs.append(
                        DocumentQuestionAnsweringOutput(
                            answer=item.get("answer", ""),
                            score=item.get("score", 0.0),
                            start=item.get("start", 0),
                            end=item.get("end", 0),
                        )
                    )

            return outputs

        except Exception as e:
            logger.error(f"Error during document QA: {e}")
            raise RuntimeError(f"Failed to process document QA: {e}") from e
