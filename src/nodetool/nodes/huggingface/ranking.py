from __future__ import annotations

from typing import TYPE_CHECKING, List, Any
from pydantic import Field

from nodetool.metadata.types import HFReranker
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker(HuggingFacePipelineNode):
    """
    Reranks pairs of text based on their semantic similarity.
    text, ranking, reranking, natural language processing

    Use cases:
    - Improve search results ranking
    - Question-answer pair scoring
    - Document relevance ranking
    """

    model: HFReranker = Field(
        default=HFReranker(),
        title="Model ID on Huggingface",
        description="The model ID to use for reranking",
    )
    query: str = Field(
        default="",
        title="Query Text",
        description="The query text to compare against candidates",
    )
    candidates: list[str] = Field(
        default=[],
        title="Candidate Texts",
        description="List of candidate texts to rank",
    )

    _model: Any = None
    _tokenizer: Any = None

    @classmethod
    def get_recommended_models(cls):
        return [
            HFReranker(
                repo_id="BAAI/bge-reranker-v2-m3",
                allow_patterns=["*.safetensors", "*.txt", "*.json"],
            ),
            HFReranker(
                repo_id="BAAI/bge-reranker-base",
                allow_patterns=["*.safetensors", "*.txt", "*.json"],
            ),
            HFReranker(
                repo_id="BAAI/bge-reranker-large",
                allow_patterns=["*.safetensors", "*.txt", "*.json"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = await self.load_model(
            context=context,
            model_class=AutoTokenizer,
            model_id=self.model.repo_id,
        )
        self._model = await self.load_model(
            context=context,
            model_class=AutoModelForSequenceClassification,
            model_id=self.model.repo_id,
        )
        self._model.eval()  # type: ignore

    async def move_to_device(self, device: str):
        self._model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        import torch

        pairs = [[self.query, candidate] for candidate in self.candidates]

        with torch.inference_mode():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )  # type: ignore
            scores = (
                self._model(**inputs, return_dict=True)  # type: ignore
                .logits.view(
                    -1,
                )
                .float()
            )

        return dict(zip(self.candidates, scores.tolist()))
