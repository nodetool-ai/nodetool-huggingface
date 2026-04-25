from __future__ import annotations

from typing import Any

from nodetool.metadata.types import HuggingFaceModel
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


class Summarization(HuggingFacePipelineNode):
    """
    Condenses long documents or articles into concise summaries using seq2seq models.
    text, summarization, NLP, BART, T5, PEGASUS, document-understanding

    Use cases:
    - Summarize news articles, research papers, or legal documents
    - Generate executive summaries for long reports
    - Create TL;DR versions of long-form content
    - Extract key points from meeting transcripts or call recordings
    - Build document digest pipelines for knowledge management
    """

    model: HuggingFaceModel = Field(
        default=HuggingFaceModel(repo_id="facebook/bart-large-cnn"),
        title="Model",
        description="The summarization model. BART-large-CNN excels at news; PEGASUS variants work well across domains; DistilBART is faster with comparable quality.",
    )
    text: str = Field(
        default="",
        title="Text",
        description="The text to summarize. Works best with at least a few sentences; very short inputs may produce poor results.",
    )
    max_length: int = Field(
        default=130,
        title="Max Length",
        description="Maximum token length of the generated summary.",
        ge=10,
        le=1024,
    )
    min_length: int = Field(
        default=30,
        title="Min Length",
        description="Minimum token length of the generated summary.",
        ge=5,
        le=512,
    )
    do_sample: bool = Field(
        default=False,
        title="Do Sample",
        description="Enable sampling for more varied outputs. Disable for deterministic, beam-search summaries.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="facebook/bart-large-cnn",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="facebook/bart-large-xsum",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="sshleifer/distilbart-cnn-12-6",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="sshleifer/distilbart-xsum-12-6",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="google/pegasus-xsum",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json", "*.txt"],
            ),
            HuggingFaceModel(
                repo_id="google/pegasus-cnn_dailymail",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json", "*.txt"],
            ),
            HuggingFaceModel(
                repo_id="t5-small",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="t5-base",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="philschmid/bart-large-cnn-samsum",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
        ]

    def required_inputs(self):
        return ["text"]

    @classmethod
    def get_title(cls) -> str:
        return "Summarization"

    def get_model_id(self):
        return self.model.repo_id

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="summarization",
            model_id=self.get_model_id(),
            device=context.device,
        )

    async def process(self, context: ProcessingContext) -> str:
        assert self._pipeline is not None
        result = await self.run_pipeline_in_thread(
            self.text,
            max_length=self.max_length,
            min_length=self.min_length,
            do_sample=self.do_sample,
        )
        return str(result[0]["summary_text"])
