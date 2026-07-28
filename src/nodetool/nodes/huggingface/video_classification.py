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
    video, classification, action-recognition, computer-vision, VideoMAE, TimeSformer, V-JEPA 2

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
        description="The video classification model. VideoMAE and TimeSformer are strong general-purpose Kinetics models; V-JEPA 2 checkpoints are newer and better at fine-grained motion (Something-Something v2, Diving48).",
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
                repo_id="facebook/vjepa2-vitl-fpc16-256-ssv2",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="facebook/vjepa2-vitl-fpc32-256-diving48",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
        ]

    def required_inputs(self):
        return ["video"]

    @classmethod
    def get_title(cls) -> str:
        return "HF Video Classifier"

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

        import os
        import tempfile

        # The video-classification pipeline decodes the video itself and only
        # accepts a local path or an http(s) URL, so materialize the asset.
        video_bytes = await context.asset_to_bytes(self.video)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            tmp.write(video_bytes)
            tmp.close()
            result = await self.run_pipeline_in_thread(
                tmp.name, num_frames=self.num_frames
            )
        finally:
            os.unlink(tmp.name)

        return {str(item["label"]): float(item["score"]) for item in result}
