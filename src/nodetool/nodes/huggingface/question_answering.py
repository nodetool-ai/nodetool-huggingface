from nodetool.metadata.types import (
    DataframeRef,
    HFQuestionAnswering,
    HFTableQuestionAnswering,
)
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


from typing import Any, TypedDict


class QuestionAnswering(HuggingFacePipelineNode):
    """
    Extracts answers to questions from a given text context using extractive QA models.
    text, question-answering, NLP, reading-comprehension

    Use cases:
    - Build automated FAQ and customer support systems
    - Extract specific information from documents and articles
    - Create reading comprehension and study tools
    - Enable natural language queries over text databases
    - Analyze contracts and legal documents for key details
    """

    model: HFQuestionAnswering = Field(
        default=HFQuestionAnswering(),
        title="Model",
        description="The extractive QA model. DistilBERT is fast; BERT-large and RoBERTa offer higher accuracy.",
    )
    context: str = Field(
        default="",
        title="Context",
        description="The text passage containing the information to answer questions from.",
    )
    question: str = Field(
        default="",
        title="Question",
        description="The question to answer based on the provided context.",
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFQuestionAnswering]:
        return [
            HFQuestionAnswering(
                repo_id="distilbert-base-cased-distilled-squad",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFQuestionAnswering(
                repo_id="bert-large-uncased-whole-word-masking-finetuned-squad",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFQuestionAnswering(
                repo_id="deepset/roberta-base-squad2",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFQuestionAnswering(
                repo_id="distilbert-base-uncased-distilled-squad",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context,
            "question-answering",
            self.model.repo_id,
            torch_dtype=select_inference_dtype(),
        )

    async def move_to_device(self, device: str):
        self._pipeline.model.to(device)  # type: ignore

    class OutputType(TypedDict):
        answer: str
        score: float
        start: int
        end: int

    async def process(self, context: ProcessingContext) -> OutputType:
        assert self._pipeline is not None
        inputs = {
            "question": self.question,
            "context": self.context,
        }

        result = await self.run_pipeline_in_thread(inputs)
        assert result is not None
        return {
            "answer": result["answer"],
            "score": result["score"],
            "start": result["start"],
            "end": result["end"],
        }


class TableQuestionAnswering(HuggingFacePipelineNode):
    """
    Answers natural language questions about tabular data using table QA models.
    table, question-answering, NLP, data-analysis

    Use cases:
    - Query spreadsheets and databases using natural language
    - Extract insights from financial reports and data tables
    - Build conversational interfaces for data exploration
    - Automate data analysis with question-based queries
    - Enable non-technical users to query structured data
    """

    @classmethod
    def get_recommended_models(cls) -> list[HFTableQuestionAnswering]:
        return [
            HFTableQuestionAnswering(
                repo_id="google/tapas-base-finetuned-wtq",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFTableQuestionAnswering(
                repo_id="google/tapas-large-finetuned-wtq",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFTableQuestionAnswering(
                repo_id="microsoft/tapex-large-finetuned-tabfact",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
        ]

    model: HFTableQuestionAnswering = Field(
        default=HFTableQuestionAnswering(),
        title="Model",
        description="The table QA model. TAPAS models handle complex queries; TAPEX offers fact verification.",
    )
    dataframe: DataframeRef = Field(
        default=DataframeRef(),
        title="Table",
        description="The table data to query. Columns should have clear, descriptive headers.",
    )
    question: str = Field(
        default="",
        title="Question",
        description="Your question about the table data (e.g., 'What is the total revenue?' or 'Which product sold the most?').",
    )

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context,
            "table-question-answering",
            self.model.repo_id,
            torch_dtype=select_inference_dtype(),
        )

    async def move_to_device(self, device: str):
        self._pipeline.model.to(device)  # type: ignore

    class OutputType(TypedDict):
        answer: str
        coordinates: list[tuple[int, int]]
        cells: list[str]
        aggregator: str

    async def process(self, context: ProcessingContext) -> OutputType:
        assert self._pipeline is not None
        table = await context.dataframe_to_pandas(self.dataframe)
        inputs = {
            "table": table.astype(str),
            "query": self.question,
        }

        result = await self.run_pipeline_in_thread(inputs)
        assert result is not None
        return {
            "answer": result["answer"],
            "coordinates": result["coordinates"],
            "cells": result["cells"],
            "aggregator": result["aggregator"],
        }
