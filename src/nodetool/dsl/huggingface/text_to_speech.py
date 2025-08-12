from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Bark(GraphNode):
    """
    Bark is a text-to-audio model created by Suno. Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying.
    tts, audio, speech, huggingface

    Use cases:
    - Create voice content for apps and websites
    - Develop voice assistants with natural-sounding speech
    - Generate automated announcements for public spaces
    """

    model: types.HFTextToSpeech | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFTextToSpeech(
            type="hf.text_to_speech",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for the image generation",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The input text to the model"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_speech.Bark"


import nodetool.nodes.huggingface.text_to_speech
import nodetool.nodes.huggingface.text_to_speech


class KokoroTTS(GraphNode):
    """
    Kokoro is an open-weight, fast, and lightweight TTS model (~82M params) with Apache-2.0 weights.
    It supports multiple languages via `misaki` and provides high-quality speech with selectable voices.
    tts, audio, speech, huggingface, kokoro

    Reference: https://huggingface.co/hexgrad/Kokoro-82M

    Use cases:
    - Natural-sounding speech synthesis for apps, assistants, and narration
    - Low-latency TTS in production or local projects
    - Multi-language TTS with configurable voices and speed
    """

    LanguageCode: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.text_to_speech.KokoroTTS.LanguageCode
    )
    Voice: typing.ClassVar[type] = (
        nodetool.nodes.huggingface.text_to_speech.KokoroTTS.Voice
    )
    model: types.HFTextToSpeech | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFTextToSpeech(
            type="hf.text_to_speech",
            repo_id="hexgrad/Kokoro-82M",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The Kokoro repo to use (e.g., hexgrad/Kokoro-82M)",
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Hello from Kokoro.", description="Input text to synthesize"
    )
    lang_code: nodetool.nodes.huggingface.text_to_speech.KokoroTTS.LanguageCode = Field(
        default=nodetool.nodes.huggingface.text_to_speech.KokoroTTS.LanguageCode.AMERICAN_ENGLISH,
        description="Language code for G2P. Examples: 'a' (American English), 'b' (British English), 'e' (es), 'f' (fr-fr), 'h' (hi), 'i' (it), 'p' (pt-br), 'j' (ja), 'z' (zh).",
    )
    voice: nodetool.nodes.huggingface.text_to_speech.KokoroTTS.Voice = Field(
        default=nodetool.nodes.huggingface.text_to_speech.KokoroTTS.Voice.AF_HEART,
        description="Voice name (see VOICES.md on the model page). Examples: af_heart, af_bella, af_jessica.",
    )
    speed: float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0, description="Speech speed multiplier (0.5–2.0)"
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_speech.KokoroTTS"


class TextToSpeech(GraphNode):
    """
    A generic Text-to-Speech node that can work with various Hugging Face TTS models.
    tts, audio, speech, huggingface, speak, voice

    Use cases:
    - Generate speech from text for various applications
    - Create voice content for apps, websites, or virtual assistants
    - Produce audio narrations for videos, presentations, or e-learning content
    """

    model: types.HFTextToSpeech | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFTextToSpeech(
            type="hf.text_to_speech",
            repo_id="",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="The model ID to use for text-to-speech generation",
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Hello, this is a test of the text-to-speech system.",
        description="The text to convert to speech",
    )

    @classmethod
    def get_node_type(cls):
        return "huggingface.text_to_speech.TextToSpeech"
