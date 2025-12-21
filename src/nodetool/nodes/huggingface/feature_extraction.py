from nodetool.metadata.types import HFFeatureExtraction, NPArray
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


class FeatureExtraction(HuggingFacePipelineNode):
    """
    Extracts dense vector embeddings from text using transformer models for downstream ML tasks.
    text, feature-extraction, embeddings, NLP, semantic-search

    Use cases:
    - Compute text embeddings for semantic similarity comparisons
    - Cluster documents by meaning rather than keywords
    - Generate input features for machine learning classifiers
    - Build semantic search engines and recommendation systems
    - Create vector databases for retrieval-augmented generation (RAG)
    """

    model: HFFeatureExtraction = Field(
        default=HFFeatureExtraction(),
        title="Model",
        description="The embedding model to use. mxbai-embed-large-v1 and BGE models offer excellent quality; smaller models trade accuracy for speed.",
    )
    inputs: str = Field(
        default="",
        title="Input Text",
        description="The text to extract embeddings from. Can be a sentence, paragraph, or document.",
    )

    @classmethod
    def get_recommended_models(cls):
        return [
            HFFeatureExtraction(
                repo_id="mixedbread-ai/mxbai-embed-large-v1",
                allow_patterns=["*.safetensors", "*.txt", "*,json"],
            ),
            HFFeatureExtraction(
                repo_id="BAAI/bge-base-en-v1.5",
                allow_patterns=["*.safetensors", "*.txt", "*,json"],
            ),
            HFFeatureExtraction(
                repo_id="BAAI/bge-large-en-v1.5",
                allow_patterns=["*.safetensors", "*.txt", "*,json"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="feature-extraction",
            model_id=self.model.repo_id,
            torch_dtype=select_inference_dtype(),
        )

    async def move_to_device(self, device: str):
        self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> NPArray:
        # The result is typically a list of lists, where each inner list represents the features for a token
        # We'll return the mean of these features to get a single vector for the entire input
        import numpy as np

        assert self._pipeline is not None

        result = await self.run_pipeline_in_thread(self.inputs)

        assert isinstance(result, list)

        return NPArray.from_numpy(np.mean(result[0], axis=0))
