from __future__ import annotations

from typing import Any

from nodetool.metadata.types import HuggingFaceModel, VideoRef
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    select_inference_dtype,
)
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


class VideoClassifier(HuggingFacePipelineNode):
    """
    Classifies video clips into action or scene categories using video transformer models.
    video, classification, action-recognition, computer-vision, VideoMAE, TimeSformer

    Use cases:
    - Recognize human actions and activities in surveillance or sports footage
    - Moderate video content by detecting inappropriate scenes
    - Tag and organize video libraries by scene type or activity
    - Enable smart search over large video collections
    - Analyze movement patterns in sports, healthcare, or robotics
    """

    model: HuggingFaceModel = Field(
        default=HuggingFaceModel(repo_id="MCG-NJU/videomae-base-finetuned-kinetics"),
        title="Model",
        description="The video classification model. VideoMAE and TimeSformer are strong general-purpose models; task-specific fine-tuned variants are available for sports, cooking, etc.",
    )
    video: VideoRef = Field(
        default=VideoRef(),
        title="Video",
        description="The video clip to classify. Shorter clips (2-10s) tend to give better results.",
    )
    num_frames: int = Field(
        default=8,
        title="Num Frames",
        description="Number of frames to sample from the video for classification. Higher values capture more temporal detail but use more memory.",
        ge=4,
        le=32,
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="MCG-NJU/videomae-base-finetuned-kinetics",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="MCG-NJU/videomae-large-finetuned-kinetics",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="facebook/timesformer-base-finetuned-k400",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="facebook/timesformer-base-finetuned-k600",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="facebook/timesformer-hr-finetuned-k400",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="sayakpaul/videomae-base-finetuned-ucf101-subset",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
        ]

    def required_inputs(self):
        return ["video"]

    @classmethod
    def get_title(cls) -> str:
        return "Video Classifier"

    def get_model_id(self):
        return self.model.repo_id

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="video-classification",
            model_id=self.get_model_id(),
            device=context.device,
            torch_dtype=select_inference_dtype(),
        )

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        assert self._pipeline is not None

        from nodetool.nodes.huggingface.image_text_to_text import extract_video_frames

        video_bytes = await context.asset_to_bytes(self.video)
        frames = await context.run_worker(
            extract_video_frames, video_bytes, fps=1
        )

        # Uniformly sample num_frames from extracted frames
        if len(frames) > self.num_frames:
            indices = [
                int(i * (len(frames) - 1) / (self.num_frames - 1))
                for i in range(self.num_frames)
            ]
            frames = [frames[i] for i in indices]

        result = await self.run_pipeline_in_thread(frames)
        return {str(item["label"]): float(item["score"]) for item in result}
