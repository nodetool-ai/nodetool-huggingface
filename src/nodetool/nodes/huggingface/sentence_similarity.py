from nodetool.metadata.types import HFSentenceSimilarity, NPArray
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext


from pydantic import Field


class SentenceSimilarity(HuggingFacePipelineNode):
    """
    Generates dense vector embeddings from text for semantic similarity comparisons.
    text, sentence-similarity, embeddings, NLP, semantic-search

    Use cases:
    - Compute semantic similarity between sentences or documents
    - Build semantic search and recommendation engines
    - Detect duplicate or near-duplicate content
    - Cluster documents by meaning rather than keywords
    - Create text embeddings for downstream ML tasks
    """

    model: HFSentenceSimilarity = Field(
        default=HFSentenceSimilarity(),
        title="Model",
        description="The sentence embedding model. all-mpnet-base-v2 offers high quality; MiniLM variants are faster; BGE-m3 is multilingual.",
    )
    inputs: str = Field(
        default="",
        title="Input Text",
        description="The text to generate embeddings for. Can be a sentence, paragraph, or short document.",
    )

    @classmethod
    def get_recommended_models(cls):
        return [
            HFSentenceSimilarity(
                repo_id="sentence-transformers/all-mpnet-base-v2",
                allow_patterns=["*.safetensors", "*.txt", "*,json"],
            ),
            HFSentenceSimilarity(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                allow_patterns=["*.safetensors", "*.txt", "*,json"],
            ),
            HFSentenceSimilarity(
                repo_id="BAAI/bge-m3",
                allow_patterns=["*.safetensors", "*.txt", "*,json"],
            ),
            HFSentenceSimilarity(
                repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                allow_patterns=["*.safetensors", "*.txt", "*,json"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="feature-extraction",
            model_id=self.model.repo_id,
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
