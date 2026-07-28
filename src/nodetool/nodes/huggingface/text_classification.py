from nodetool.metadata.types import HFTextClassification, HFZeroShotClassification
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


class TextClassifier(HuggingFacePipelineNode):
    """
    Classifies text into predefined categories using fine-tuned transformer models.
    text, classification, sentiment, NLP, emotion

    Use cases:
    - Analyze sentiment in social media posts and reviews
    - Detect emotions in customer feedback and conversations
    - Classify support tickets by category or priority
    - Filter spam or inappropriate content
    - Categorize news articles by topic
    """

    model: HFTextClassification = Field(
        default=HFTextClassification(),
        title="Model",
        description="The text classification model. Use sentiment models for opinion analysis; emotion models for feeling detection; tabularisai and clapAI models cover many languages.",
    )
    prompt: str = Field(
        default="",
        title="Input Text",
        description="The text to classify.",
    )

    @classmethod
    def get_recommended_models(cls):
        return [
            HFTextClassification(
                repo_id=model,
                allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"],
            )
            for model in [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "michellejieli/emotion_text_classifier",
                "SamLowe/roberta-base-go_emotions",
                "tabularisai/multilingual-sentiment-analysis",
                "tabularisai/multilingual-emotion-classification",
                "clapAI/modernBERT-base-multilingual-sentiment",
            ]
        ]

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context,
            "text-classification",
            self.model.repo_id,
            torch_dtype=select_inference_dtype(),
        )

    async def move_to_device(self, device: str):
        self._pipeline.model.to(device)

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        assert self._pipeline is not None
        # Without an explicit top_k the pipeline falls back to its legacy mode and
        # emits only the winning label, which does not fill out the label -> score
        # mapping this node advertises. top_k=None returns every label.
        result = await self.run_pipeline_in_thread(self.prompt, top_k=None)
        return {i["label"]: i["score"] for i in list(result)}


class ZeroShotTextClassifier(HuggingFacePipelineNode):
    """
    Classifies text into custom categories without requiring task-specific training data.
    text, classification, zero-shot, NLP, flexible

    Use cases:
    - Classify text into custom, user-defined categories on the fly
    - Detect topics in documents without predefined training
    - Perform sentiment analysis with custom sentiment labels
    - Build flexible intent classification for conversational AI
    - Prototype classification systems with dynamic categories
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
                repo_id="tasksource/ModernBERT-large-nli",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFZeroShotClassification(
                repo_id="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFZeroShotClassification(
                repo_id="MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
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
        title="Model",
        description="The zero-shot classification model. BART-large-mnli is reliable; the zeroshot-v2.0 and ModernBERT-NLI checkpoints are more accurate and faster; mDeBERTa is multilingual.",
    )
    inputs: str = Field(
        default="",
        title="Input Text",
        description="The text to classify.",
    )
    candidate_labels: str = Field(
        default="",
        title="Candidate Labels",
        description="Comma-separated list of labels to classify against (e.g., 'positive,negative,neutral' or 'sports,politics,technology').",
    )
    multi_label: bool = Field(
        default=False,
        title="Multi-label Classification",
        description="Allow multiple labels to be assigned to the same text (useful when text can belong to multiple categories).",
    )

    async def preload_model(self, context: ProcessingContext):
        # load model directly onto device
        self._pipeline = await self.load_pipeline(
            context=context,
            model_id=self.model.repo_id,
            pipeline_task="zero-shot-classification",
            device=context.device,
            torch_dtype=select_inference_dtype(),
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
        result = await self.run_pipeline_in_thread(
            self.inputs,
            candidate_labels=labels,
            multi_label=self.multi_label,
        )
        return dict(zip(result["labels"], result["scores"]))
