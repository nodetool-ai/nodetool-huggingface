from nodetool.metadata.types import ColumnDef, DataframeRef, HFTokenClassification
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext


from pydantic import Field


from enum import Enum


class TokenClassification(HuggingFacePipelineNode):
    """
    Performs token-level classification tasks such as Named Entity Recognition (NER).
    text, token-classification, NER, NLP, entity-extraction

    Use cases:
    - Extract named entities (people, organizations, locations) from text
    - Identify parts of speech in sentences
    - Perform chunking and shallow parsing for text analysis
    - Extract structured information from unstructured documents
    - Build information extraction pipelines for documents
    """

    class AggregationStrategy(str, Enum):
        SIMPLE = "simple"
        FIRST = "first"
        AVERAGE = "average"
        MAX = "max"

    model: HFTokenClassification = Field(
        default=HFTokenClassification(
            repo_id="dbmdz/bert-large-cased-finetuned-conll03-english",
            allow_patterns=["*.bin", "*.json", "**/*.json", "*.safetensors"],
        ),
        title="Model",
        description="The token classification model. BERT-large-cased-finetuned-conll03 offers high-quality NER for English text.",
    )
    inputs: str = Field(
        default="",
        title="Input Text",
        description="The text to extract entities from.",
    )
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.SIMPLE,
        title="Aggregation Strategy",
        description="How to combine token predictions into entities: 'simple' merges adjacent tokens; 'first'/'average'/'max' control subword handling.",
    )

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context, "token-classification", self.model.repo_id
        )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        assert self._pipeline is not None
        result = await self.run_pipeline_in_thread(
            self.inputs, aggregation_strategy=self.aggregation_strategy.value
        )
        data = [
            [
                item["entity_group"],  # type: ignore
                item["word"],  # type: ignore
                item["start"],  # type: ignore
                item["end"],  # type: ignore
                float(item["score"]),  # type: ignore
            ]
            for item in result  # type: ignore
        ]
        columns = [
            ColumnDef(name="entity", data_type="string"),
            ColumnDef(name="word", data_type="string"),
            ColumnDef(name="start", data_type="int"),
            ColumnDef(name="end", data_type="int"),
            ColumnDef(name="score", data_type="float"),
        ]
        return DataframeRef(columns=columns, data=data)
