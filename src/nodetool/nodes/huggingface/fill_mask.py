from nodetool.metadata.types import ColumnDef, DataframeRef, HFFillMask
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext


from pydantic import Field


from typing import Any


class FillMask(HuggingFacePipelineNode):
    """
    Predicts the most likely words to fill masked positions in text using language models.
    text, fill-mask, NLP, language-modeling, word-prediction

    Use cases:
    - Complete sentences with contextually appropriate words
    - Generate word suggestions for text editing tools
    - Test language understanding and word associations
    - Build autocomplete and text prediction features
    - Explore semantic relationships between words in context
    """

    model: HFFillMask = Field(
        default=HFFillMask(),
        title="Model",
        description="The masked language model to use. BERT, RoBERTa, and DistilBERT variants are supported.",
    )
    inputs: str = Field(
        default="The capital of France is [MASK].",
        title="Input Text",
        description="Text containing [MASK] token(s) to be predicted. Different models may use different mask tokens (e.g., BERT uses [MASK], RoBERTa uses <mask>).",
    )
    top_k: int = Field(
        default=5,
        title="Top K",
        description="Number of most likely predictions to return for each masked position.",
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFFillMask]:
        return [
            HFFillMask(
                repo_id="bert-base-uncased",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFFillMask(
                repo_id="roberta-base",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFFillMask(
                repo_id="distilbert-base-uncased",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFFillMask(
                repo_id="albert-base-v2",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context, "fill-mask", self.model.repo_id
        )

    async def move_to_device(self, device: str):
        self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        assert self._pipeline is not None
        result = await self.run_pipeline_in_thread(self.inputs, top_k=self.top_k)
        assert result is not None
        data = [[item["token_str"], item["score"]] for item in result]  # type: ignore
        columns = [
            ColumnDef(name="token", data_type="string"),
            ColumnDef(name="score", data_type="float"),
        ]
        return DataframeRef(columns=columns, data=data)  # type: ignore
