from __future__ import annotations

from typing import Any

from nodetool.metadata.types import HuggingFaceModel, ImageRef
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


class DocumentQuestionAnswering(HuggingFacePipelineNode):
    """
    Answers natural language questions about document images (PDFs, receipts, forms, invoices).
    document, QA, question-answering, OCR, LayoutLM, Donut, vision-language

    Use cases:
    - Extract structured data from scanned invoices, receipts, and forms
    - Query PDF documents by natural language without manual parsing
    - Automate data entry from paper-based documents
    - Build intelligent document processing pipelines
    - Answer questions about tables, charts, and mixed-layout documents
    """

    model: HuggingFaceModel = Field(
        default=HuggingFaceModel(repo_id="impira/layoutlm-document-qa"),
        title="Model",
        description="The document QA model. LayoutLM variants combine text and layout understanding; Donut models are OCR-free and more robust to noisy scans.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Document Image",
        description="The document image to query (scan, screenshot, photo of a form or invoice).",
    )
    question: str = Field(
        default="",
        title="Question",
        description="The natural language question to answer about the document (e.g. 'What is the total amount?', 'Who is the sender?').",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="impira/layoutlm-document-qa",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="impira/layoutlm-invoices",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="naver-clova-ix/donut-base-finetuned-docvqa",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="naver-clova-ix/donut-base-finetuned-rvlcdip",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="microsoft/layoutlmv3-base",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="cloudqi/cqi_visual_question_answering_pt_v0",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
        ]

    def required_inputs(self):
        return ["image", "question"]

    @classmethod
    def get_title(cls) -> str:
        return "Document Question Answering"

    def get_model_id(self):
        return self.model.repo_id

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="document-question-answering",
            model_id=self.get_model_id(),
            device=context.device,
            torch_dtype=select_inference_dtype(),
        )

    async def process(self, context: ProcessingContext) -> str:
        assert self._pipeline is not None
        image = await context.image_to_pil(self.image)
        result = await self.run_pipeline_in_thread(
            image=image,
            question=self.question,
        )
        if isinstance(result, list) and len(result) > 0:
            return str(result[0].get("answer", ""))
        if isinstance(result, dict):
            return str(result.get("answer", ""))
        return str(result)
