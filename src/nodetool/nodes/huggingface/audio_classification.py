from __future__ import annotations

from nodetool.metadata.types import (
    AudioRef,
    HFAudioClassification,
    HFZeroShotAudioClassification,
    HuggingFaceModel,
)
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext

from typing import TYPE_CHECKING, Any
from pydantic import Field

if TYPE_CHECKING:
    import torch
    from transformers import (
        AudioClassificationPipeline,
        ZeroShotAudioClassificationPipeline,
    )


class AudioClassifier(HuggingFacePipelineNode):
    """
    Classifies audio into predefined categories using pretrained transformer models.
    audio, classification, labeling, categorization, sound-recognition

    Use cases:
    - Classify music by genre or mood
    - Detect speech vs. non-speech audio segments
    - Identify environmental sounds (e.g., car horn, dog bark)
    - Recognize emotions in speech recordings
    - Content moderation for audio platforms
    """

    model: HFAudioClassification = Field(
        default=HFAudioClassification(),
        title="Model",
        description="The Hugging Face model for audio classification. Recommended: MIT/ast-finetuned-audioset for general sounds, wav2vec2-lg-xlsr-en-speech-emotion-recognition for speech emotions.",
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        title="Audio",
        description="The audio file to classify. Supports common formats like WAV, MP3, FLAC.",
    )
    top_k: int = Field(
        default=10,
        title="Top K",
        description="Number of top classification results to return, ranked by confidence score.",
    )
    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFAudioClassification(
                repo_id="MIT/ast-finetuned-audioset-10-10-0.4593",
                allow_patterns=["*.safetensors", "*.json"],
            ),
            HFAudioClassification(
                repo_id="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                allow_patterns=["pytorch_model.bin", "*.json"],
            ),
        ]

    def required_inputs(self):
        return ["audio"]

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="audio-classification",
            model_id=self.model.repo_id,
            torch_dtype=select_inference_dtype(),
        )  # type: ignore

    async def move_to_device(self, device: str):
        assert self._pipeline is not None, "Pipeline not initialized"
        self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        samples, _, _ = await context.audio_to_numpy(self.audio)
        result = await self.run_pipeline_in_thread(
            samples,
            top_k=self.top_k,
        )  # type: ignore
        return {item["label"]: item["score"] for item in result}  # type: ignore


class ZeroShotAudioClassifier(HuggingFacePipelineNode):
    """
    Classifies audio into custom categories without requiring task-specific training data.
    audio, classification, labeling, categorization, zero-shot, flexible

    Use cases:
    - Categorize audio with custom, user-defined labels on the fly
    - Identify sounds or music genres without predefined model training
    - Quickly prototype audio classification systems
    - Automate tagging for large audio datasets with dynamic categories
    """

    model: HFZeroShotAudioClassification = Field(
        default=HFZeroShotAudioClassification(),
        title="Model",
        description="The Hugging Face model for zero-shot audio classification. Uses CLIP-based models for flexible label matching.",
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        title="Audio",
        description="The audio file to classify. Supports common formats like WAV, MP3, FLAC.",
    )
    candidate_labels: str = Field(
        default="",
        title="Candidate Labels",
        description="Comma-separated list of labels to classify against (e.g., 'music,speech,noise,silence').",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFZeroShotAudioClassification(
                repo_id="laion/clap-htsat-unfused",
                allow_patterns=["model.safetensors", "*.json", "*.txt"],
            ),
        ]

    def required_inputs(self):
        return ["audio"]

    def get_model_id(self):
        return self.model.repo_id

    @property
    def pipeline_task(self) -> str:
        return "zero-shot-audio-classification"

    async def preload_model(self, context: ProcessingContext):
        from transformers import ZeroShotAudioClassificationPipeline

        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="zero-shot-audio-classification",
            model_id=self.model.repo_id,
            torch_dtype=select_inference_dtype(),
        )

    def get_params(self):
        return {}

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        assert self._pipeline is not None, "Pipeline not initialized"
        samples, _, _ = await context.audio_to_numpy(self.audio)
        result = await self.run_pipeline_in_thread(
            samples, candidate_labels=self.candidate_labels.split(",")
        )
        return {item["label"]: item["score"] for item in result}  # type: ignore
