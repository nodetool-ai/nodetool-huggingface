from nodetool.metadata.types import (
    DataframeRef,
    HFTableQuestionAnswering,
)
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


from typing import TypedDict


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
        self._pipeline.model.to(device)

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
        # Only TAPAS models emit coordinates/cells, and only aggregating TAPAS
        # heads emit an aggregator. Seq2seq models (e.g. TAPEX) return the
        # answer alone, so the extra keys are optional.
        return {
            "answer": result["answer"],
            "coordinates": result.get("coordinates", []),
            "cells": result.get("cells", []),
            "aggregator": result.get("aggregator", ""),
        }
