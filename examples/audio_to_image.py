import asyncio
import os
from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.huggingface.text_to_image import StableDiffusion
from nodetool.dsl.huggingface.automatic_speech_recognition import Whisper
from nodetool.dsl.nodetool.constant import Audio
from nodetool.metadata.types import (
    AudioRef,
    HFAutomaticSpeechRecognition,
    HFStableDiffusion,
)

dirname = os.path.dirname(__file__)
audio_path = os.path.join(dirname, "sample-0.mp3")
audio = Audio(value=AudioRef(uri=audio_path, type="audio"))
whisper = Whisper(
    audio=audio.output,
    model=HFAutomaticSpeechRecognition(
        repo_id="openai/whisper-small",
    ),
)

g = StableDiffusion(
    prompt=whisper.out.text,
    model=HFStableDiffusion(
        repo_id="SG161222/Realistic_Vision_V5.1_noVAE",
        path="Realistic_Vision_V5.1_fp16-no-ema.safetensors",
    ),
)

run_graph(create_graph(g))
