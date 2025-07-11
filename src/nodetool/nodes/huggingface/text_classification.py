from nodetool.metadata.types import HFTextClassification, HFZeroShotClassification
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext


from pydantic import Field


class TextClassifier(HuggingFacePipelineNode):
    """
    Classifies text into predefined categories using a Hugging Face model.
    text, classification, zero-shot, natural language processing
    """

    model: HFTextClassification = Field(
        default=HFTextClassification(),
        title="Model ID on Huggingface",
        description="The model ID to use for the classification",
    )
    prompt: str = Field(
        default="",
        title="Inputs",
        description="The input text to the model",
    )

    @classmethod
    def get_recommended_models(cls):
        return [
            HFTextClassification(
                repo_id=model, allow_patterns=["*.json", "*.txt", "*.bin"]
            )
            for model in [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "michellejieli/emotion_text_classifier",
            ]
        ]

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context, "text-classification", self.model.repo_id
        )

    async def move_to_device(self, device: str):
        self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        assert self._pipeline is not None
        result = self._pipeline(self.prompt)
        return {i["label"]: i["score"] for i in list(result)}  # type: ignore


class ZeroShotTextClassifier(HuggingFacePipelineNode):
    """
    Performs zero-shot classification on text.
    text, classification, zero-shot, natural language processing

    Use cases:
    - Classify text into custom categories without training
    - Topic detection in documents
    - Sentiment analysis with custom sentiment labels
    - Intent classification in conversational AI
    """

    @classmethod
    def get_recommended_models(cls) -> list[HFZeroShotClassification]:
        return [
            HFZeroShotClassification(
                repo_id="facebook/bart-large-mnli",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFZeroShotClassification(
                repo_id="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFZeroShotClassification(
                repo_id="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFZeroShotClassification(
                repo_id="tasksource/ModernBERT-base-nli",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFZeroShotClassification(
                repo_id="cross-encoder/nli-deberta-v3-base",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFZeroShotClassification(
                repo_id="microsoft/deberta-v2-xlarge-mnli",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFZeroShotClassification(
                repo_id="roberta-large-mnli",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
        ]

    model: HFZeroShotClassification = Field(
        default=HFZeroShotClassification(),
        title="Model ID on Huggingface",
        description="The model ID to use for zero-shot classification",
    )
    inputs: str = Field(
        default="",
        title="Input Text",
        description="The text to classify",
    )
    candidate_labels: str = Field(
        default="",
        title="Candidate Labels",
        description="Comma-separated list of candidate labels for classification",
    )
    multi_label: bool = Field(
        default=False,
        title="Multi-label Classification",
        description="Whether to perform multi-label classification",
    )

    async def preload_model(self, context: ProcessingContext):
        # load model directly onto device
        self._pipeline = await self.load_pipeline(
            context=context,
            model_id=self.model.repo_id,
            pipeline_task="zero-shot-classification",
            device=context.device,
        )

    async def move_to_device(self, device: str):
        pass

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        assert self._pipeline, "Pipeline not initialized"
        if self.candidate_labels == "":
            raise ValueError("Please provide candidate labels")
        if self.inputs.strip() == "":
            raise ValueError("Please provide input text")
        
        labels = self.candidate_labels.split(",")
        result = self._pipeline(
            self.inputs,
            candidate_labels=labels,
            multi_label=self.multi_label,
        )
        return dict(zip(result["labels"], result["scores"]))  # type: ignore
