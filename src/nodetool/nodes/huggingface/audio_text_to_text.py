from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

from pydantic import Field

from nodetool.metadata.types import AudioRef, HFAudioTextToText, HuggingFaceModel
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext


class AudioFlamingo(HuggingFacePipelineNode):
    """
    Understands and answers questions about speech, sound, and music using NVIDIA's Audio/Music Flamingo.
    audio, music, understanding, question-answering, captioning, audio-text-to-text, flamingo

    Use cases:
    - Caption or describe music, environmental sounds, and speech
    - Answer questions about an audio clip (genre, mood, instruments, events)
    - Reason over up to 10 minutes of audio with text instructions
    - Build audio analysis and tagging pipelines
    """

    model: HFAudioTextToText = Field(
        default=HFAudioTextToText(
            repo_id="nvidia/audio-flamingo-3-hf",
        ),
        title="Model",
        description="The Audio Flamingo model. audio-flamingo-3 handles speech/sound/music; music-flamingo specializes in music.",
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        title="Audio",
        description="Audio clip to analyze (max ~10 minutes; longer audio is truncated).",
    )
    prompt: str = Field(
        default="Describe this audio in detail.",
        title="Prompt",
        description="Instruction or question about the audio.",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        title="Max New Tokens",
        description="Maximum number of tokens to generate.",
    )

    _pipeline: Any = None
    _processor: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "Audio Flamingo"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "audio", "prompt"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFAudioTextToText(
                repo_id="nvidia/audio-flamingo-3-hf",
                # Repo ships sharded weights (canonical, used by from_pretrained)
                # plus a redundant 16.5GB consolidated model.safetensors; "think/*"
                # is a separate reasoning variant the node does not load.
                ignore_patterns=["model.safetensors", "think/*"],
            ),
            HFAudioTextToText(repo_id="nvidia/music-flamingo-hf"),
            HFAudioTextToText(repo_id="nvidia/music-flamingo-2601-hf"),
        ]

    def required_inputs(self):
        return ["audio"]

    def get_model_id(self) -> str:
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        from nodetool.nodes.huggingface.huggingface_pipeline import (
            select_inference_dtype,
        )

        if not self.model.repo_id:
            raise ValueError("Model ID is required")

        self._pipeline = await self.load_model(
            context=context,
            model_class=AudioFlamingo3ForConditionalGeneration,
            model_id=self.get_model_id(),
            torch_dtype=select_inference_dtype(),
        )
        self._processor = await self.load_model(
            context=context,
            model_class=AutoProcessor,
            model_id=self.get_model_id(),
        )

    async def process(self, context: ProcessingContext) -> str:
        import torch

        assert self._pipeline is not None, "Model not initialized"
        assert self._processor is not None, "Processor not initialized"

        # Materialize the input audio to a temp WAV so the processor's chat
        # template can load it as a file path (its supported audio input form).
        segment = await context.audio_to_audio_segment(self.audio)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        await asyncio.to_thread(segment.export, tmp.name, format="wav")

        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "audio", "path": tmp.name},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(self._pipeline.device, dtype=self._pipeline.dtype)

            prompt_len = inputs["input_ids"].shape[1]

            def _generate():
                with torch.inference_mode():
                    return self._pipeline.generate(
                        **inputs, max_new_tokens=self.max_new_tokens
                    )

            outputs = await asyncio.to_thread(_generate)
        finally:
            os.unlink(tmp.name)

        decoded = self._processor.decode(
            outputs[0, prompt_len:], skip_special_tokens=True
        )
        return decoded.strip()
