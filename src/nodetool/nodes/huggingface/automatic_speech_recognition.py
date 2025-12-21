from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, TypedDict, Any
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    AudioRef,
    AudioChunk,
    HFAutomaticSpeechRecognition,
    HuggingFaceModel,
)
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field
from enum import Enum

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from transformers.pipelines.automatic_speech_recognition import (
        AutomaticSpeechRecognitionPipeline,
    )

logger = get_logger(__name__)


def _is_cuda_available() -> bool:
    """Safely check if CUDA is available, handling cases where PyTorch is not compiled with CUDA support."""
    try:
        import torch

        # Check if cuda module exists
        if not hasattr(torch, "cuda"):
            return False
        # Try to check availability - this can raise RuntimeError if CUDA is not compiled
        return torch.cuda.is_available()
    except (RuntimeError, AttributeError):
        # PyTorch not compiled with CUDA support or other CUDA-related error
        return False


class WhisperLanguage(str, Enum):
    NONE = "auto_detect"
    SPANISH = "spanish"
    ITALIAN = "italian"
    KOREAN = "korean"
    PORTUGUESE = "portuguese"
    ENGLISH = "english"
    JAPANESE = "japanese"
    GERMAN = "german"
    RUSSIAN = "russian"
    DUTCH = "dutch"
    POLISH = "polish"
    CATALAN = "catalan"
    FRENCH = "french"
    INDONESIAN = "indonesian"
    UKRAINIAN = "ukrainian"
    TURKISH = "turkish"
    MALAY = "malay"
    SWEDISH = "swedish"
    MANDARIN = "mandarin"
    FINNISH = "finnish"
    NORWEGIAN = "norwegian"
    ROMANIAN = "romanian"
    THAI = "thai"
    VIETNAMESE = "vietnamese"
    SLOVAK = "slovak"
    ARABIC = "arabic"
    CZECH = "czech"
    CROATIAN = "croatian"
    GREEK = "greek"
    SERBIAN = "serbian"
    DANISH = "danish"
    BULGARIAN = "bulgarian"
    HUNGARIAN = "hungarian"
    FILIPINO = "filipino"
    BOSNIAN = "bosnian"
    GALICIAN = "galician"
    MACEDONIAN = "macedonian"
    HINDI = "hindi"
    ESTONIAN = "estonian"
    SLOVENIAN = "slovenian"
    TAMIL = "tamil"
    LATVIAN = "latvian"
    AZERBAIJANI = "azerbaijani"
    URDU = "urdu"
    LITHUANIAN = "lithuanian"
    HEBREW = "hebrew"
    WELSH = "welsh"
    PERSIAN = "persian"
    ICELANDIC = "icelandic"
    KAZAKH = "kazakh"
    AFRIKAANS = "afrikaans"
    KANNADA = "kannada"
    MARATHI = "marathi"
    SWAHILI = "swahili"
    TELUGU = "telugu"
    MAORI = "maori"
    NEPALI = "nepali"
    ARMENIAN = "armenian"
    BELARUSIAN = "belarusian"
    GUJARATI = "gujarati"
    PUNJABI = "punjabi"
    BENGALI = "bengali"


class Task(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class Timestamps(str, Enum):
    NONE = "none"
    WORD = "word"
    SENTENCE = "sentence"


class Whisper(HuggingFacePipelineNode):
    """
    Converts speech to text using OpenAI's Whisper models with multilingual support.
    asr, automatic-speech-recognition, speech-to-text, translate, transcribe, audio, huggingface

    Use cases:
    - Transcribe audio files into text for documentation or analysis
    - Enable voice input for chatbots and virtual assistants
    - Create subtitles and closed captions for videos
    - Translate speech from one language to English
    - Build voice-controlled applications

    **Note:** Language selection follows Whisper's FLEURS benchmark word error rate ranking.
    Multiple model variants are available, optimized for different speed/accuracy trade-offs.

    **Links:**
    - https://github.com/openai/whisper
    - https://platform.openai.com/docs/guides/speech-to-text/supported-languages
    """

    model: HFAutomaticSpeechRecognition = Field(
        default=HFAutomaticSpeechRecognition(),
        title="Model",
        description="The Whisper model variant to use. Larger models (large-v3) offer better accuracy; smaller models (small, tiny) are faster. Turbo variants balance speed and quality.",
    )
    audio: AudioRef = Field(
        default=AudioRef(),
        title="Audio Input",
        description="The audio file to transcribe. Supports WAV, MP3, FLAC and other common formats.",
    )

    task: Task = Field(
        default=Task.TRANSCRIBE,
        title="Task",
        description="Choose 'transcribe' for speech-to-text in the original language, or 'translate' to convert any language to English.",
    )
    language: WhisperLanguage = Field(
        default=WhisperLanguage.NONE,
        title="Language",
        description="Specify the audio's language for better accuracy, or use 'auto_detect' to let the model identify it automatically.",
    )
    timestamps: Timestamps = Field(
        default=Timestamps.NONE,
        title="Timestamps",
        description="Choose 'word' for word-level timing, 'sentence' for phrase-level timing, or 'none' to disable timestamps.",
    )

    _pipeline: Any = None

    @classmethod
    def get_basic_fields(cls):
        return ["model", "audio", "task"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFAutomaticSpeechRecognition(
                repo_id="openai/whisper-large-v3",
                allow_patterns=["model.safetensors", "*.json", "*.txt"],
            ),
            HFAutomaticSpeechRecognition(
                repo_id="openai/whisper-large-v3-turbo",
                allow_patterns=["model.safetensors", "*.json", "*.txt"],
            ),
            HFAutomaticSpeechRecognition(
                repo_id="openai/whisper-large-v2",
                allow_patterns=["model.safetensors", "*.json", "*.txt"],
            ),
            HFAutomaticSpeechRecognition(
                repo_id="openai/whisper-medium",
                allow_patterns=["model.safetensors", "*.json", "*.txt"],
            ),
            HFAutomaticSpeechRecognition(
                repo_id="openai/whisper-small",
                allow_patterns=["model.safetensors", "*.json", "*.txt"],
            ),
            HFAutomaticSpeechRecognition(
                repo_id="Systran/faster-whisper-large-v3",
                allow_patterns=["model.bin", "*.json", "*.txt"],
            ),
        ]

    class OutputType(TypedDict):
        text: str
        chunks: list[AudioChunk]

    def required_inputs(self):
        return ["audio"]

    async def preload_model(self, context: ProcessingContext):
        logger.info("Initializing Whisper model...")
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        torch_dtype = torch.float16 if _is_cuda_available() else torch.float32

        model = await self.load_model(
            context=context,
            model_class=AutoModelForSpeechSeq2Seq,
            model_id=self.model.repo_id,
            variant=None,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            torch_dtype=torch_dtype,
        )

        processor = AutoProcessor.from_pretrained(self.model.repo_id)

        if self.task == Task.TRANSCRIBE:
            pipeline_task = "automatic-speech-recognition"
        elif self.task == Task.TRANSLATE:
            pipeline_task = "translation"
        else:
            pipeline_task = "automatic-speech-recognition"

        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task=pipeline_task,
            model_id=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=context.device,
        )  # type: ignore

        logger.info("Whisper model initialized successfully.")

    async def move_to_device(self, device: str):
        assert self._pipeline
        self._pipeline.model.to(device)  # type: ignore
        logger.info(f"Moved Whisper model to device: {device}")

    async def process(self, context: ProcessingContext) -> OutputType:
        assert self._pipeline

        logger.info("Starting audio processing...")

        samples, _, _ = await context.audio_to_numpy(self.audio, sample_rate=16_000)

        pipeline_kwargs = {
            "return_timestamps": (
                self.timestamps.value if self.timestamps != Timestamps.NONE else None
            ),
            "chunk_length_s": 30.0,
            "generate_kwargs": {
                "language": (
                    None
                    if self.language.value == "auto_detect"
                    else self.language.value
                ),
            },
        }

        result = await self.run_pipeline_in_thread(samples, **pipeline_kwargs)

        assert isinstance(result, dict)

        text = result.get("text", "")

        chunks = []
        if self.timestamps != Timestamps.NONE:
            raw_chunks = result.get("chunks", [])
            SEGMENT_LENGTH = 30.0  # Whisper's default segment length in seconds

            for chunk in raw_chunks:
                try:
                    timestamp = chunk.get("timestamp")
                    if (
                        timestamp
                        and len(timestamp) == 2
                        and all(isinstance(t, (int, float)) for t in timestamp)
                    ):
                        chunks.append(
                            AudioChunk(
                                timestamp=timestamp,
                                text=chunk.get("text", ""),
                            )
                        )
                except ValueError as e:
                    logger.warning(f"Skipping invalid chunk: {e}")

        logger.info("Audio processing completed successfully.")
        return {
            "text": text,
            "chunks": chunks,
        }


class ChunksToSRT(BaseNode):
    """
    Converts Whisper audio chunks to SubRip Subtitle (SRT) format for video captioning.
    subtitle, srt, whisper, transcription, captions

    Use cases:
    - Generate .srt subtitle files for video players
    - Create closed captions for accessibility compliance
    - Convert Whisper transcription output to industry-standard format
    - Build automated subtitle generation pipelines
    """

    chunks: list[AudioChunk] = Field(
        default=[],
        title="Audio Chunks",
        description="List of timestamped audio chunks from Whisper transcription output.",
    )

    time_offset: float = Field(
        default=0.0,
        title="Time Offset",
        description="Offset in seconds to add to all timestamps (useful when audio is from a clip within a longer video).",
    )

    def required_inputs(self):
        return ["chunks"]

    def _format_time(self, seconds: float) -> str:
        time = datetime.timedelta(seconds=seconds)
        return (datetime.datetime.min + time).strftime("%H:%M:%S,%f")[:-3]

    async def process(self, context: ProcessingContext) -> str:
        srt_lines = []
        for index, chunk in enumerate(self.chunks, start=1):
            start_time = chunk.timestamp[0] + self.time_offset
            end_time = chunk.timestamp[1] + self.time_offset

            srt_lines.extend(
                [
                    f"{index}",
                    f"{self._format_time(start_time)} --> {self._format_time(end_time)}",
                    f"{chunk.text.strip()}",
                    "",
                ]
            )

        return "\n".join(srt_lines)
