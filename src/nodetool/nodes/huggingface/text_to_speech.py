from __future__ import annotations

import base64
from enum import Enum
from nodetool.workflows.types import Chunk
from nodetool.metadata.types import AudioRef, HFTextToSpeech, HuggingFaceModel
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext

from typing import TYPE_CHECKING, Any, AsyncGenerator, Mapping, TypedDict
from pydantic import Field
import numpy as np

if TYPE_CHECKING:
    from kokoro.pipeline import KPipeline
    from transformers.pipelines.text_to_audio import TextToAudioPipeline
    from transformers.pipelines.base import Pipeline


class Bark(HuggingFacePipelineNode):
    """
    Generates realistic multilingual speech and non-verbal audio from text using Suno's Bark model.
    tts, audio, speech, huggingface, multilingual, voice-synthesis

    Use cases:
    - Create natural-sounding voice content for apps and videos
    - Generate multilingual speech for global applications
    - Produce expressive speech with emotions (laughing, sighing, crying)
    - Add realistic voice to chatbots and virtual assistants
    - Create audio content with background music and sound effects
    """

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(),
        title="Model",
        description="The Bark model variant. bark-small is faster; bark offers higher quality. Both support multilingual output.",
    )
    prompt: str = Field(
        default="",
        title="Text Prompt",
        description="The text to convert to speech. Can include non-verbal markers like [laughs] or [sighs].",
    )
    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToSpeech(
                repo_id="suno/bark",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
            HFTextToSpeech(
                repo_id="suno/bark-small",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
        ]

    def get_model_id(self):
        return self.model.repo_id

    async def move_to_device(self, device: str):
        pass

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="text-to-speech",
            model_id=self.get_model_id(),
            device=context.device,
        )  # type: ignore

    async def process(self, context: ProcessingContext) -> AudioRef:
        assert self._pipeline is not None, "Pipeline not initialized"
        result = await self.run_pipeline_in_thread(
            self.prompt, forward_params={"do_sample": True}
        )
        audio = await context.audio_from_numpy(result["audio"], 24_000)  # type: ignore
        return audio


class KokoroTTS(HuggingFacePipelineNode):
    """
    Generates high-quality speech from text using the lightweight Kokoro TTS model.
    tts, audio, speech, huggingface, kokoro, multilingual, voice

    Use cases:
    - Generate natural-sounding speech for applications and assistants
    - Create voice content with multiple voice options and styles
    - Build low-latency TTS systems for real-time applications
    - Produce multilingual speech in various languages
    - Generate narration for videos, presentations, and e-learning

    **Note:** Kokoro is a fast, lightweight model (~82M params) with Apache-2.0 license.
    See https://huggingface.co/hexgrad/Kokoro-82M for voice samples.
    """

    class LanguageCode(str, Enum):
        AMERICAN_ENGLISH = "a"
        BRITISH_ENGLISH = "b"
        ESPANOL = "e"
        FRENCH = "f"
        HINDI = "h"
        ITALIAN = "i"
        PORTUGUESE = "p"
        JAPANESE = "j"
        CHINESE = "z"
        KOREAN = "k"
        RUSSIAN = "r"
        TURKISH = "t"
        VIETNAMESE = "v"
        ARABIC = "a"
        GERMAN = "g"
        POLISH = "p"
        ROMANIAN = "r"
        UKRAINIAN = "u"

    class Voice(str, Enum):
        # af_*
        AF_ALLOY = "af_alloy"
        AF_AOEDE = "af_aoede"
        AF_BELLA = "af_bella"
        AF_HEART = "af_heart"
        AF_JESSICA = "af_jessica"
        AF_KORE = "af_kore"
        AF_NICOLE = "af_nicole"
        AF_NOVA = "af_nova"
        AF_RIVER = "af_river"
        AF_SARAH = "af_sarah"
        AF_SKY = "af_sky"
        # am_*
        AM_ADAM = "am_adam"
        AM_ECHO = "am_echo"
        AM_ERIC = "am_eric"
        AM_FENRIR = "am_fenrir"
        AM_LIAM = "am_liam"
        AM_MICHAEL = "am_michael"
        AM_ONYX = "am_onyx"
        AM_PUCK = "am_puck"
        AM_SANTA = "am_santa"
        # bf_*
        BF_ALICE = "bf_alice"
        BF_EMMA = "bf_emma"
        BF_ISABELLA = "bf_isabella"
        BF_LILY = "bf_lily"
        # bm_*
        BM_DANIEL = "bm_daniel"
        BM_FABLE = "bm_fable"
        BM_GEORGE = "bm_george"
        BM_LEWIS = "bm_lewis"
        # ef_*
        EF_DORA = "ef_dora"
        # em_*
        EM_ALEX = "em_alex"
        EM_SANTA = "em_santa"
        # ff_*
        FF_SIWIS = "ff_siwis"
        # hf_*
        HF_ALPHA = "hf_alpha"
        HF_BETA = "hf_beta"
        # hm_*
        HM_OMEGA = "hm_omega"
        HM_PSI = "hm_psi"
        # if_*
        IF_SARA = "if_sara"
        # im_*
        IM_NICOLA = "im_nicola"
        # jf_*
        JF_ALPHA = "jf_alpha"
        JF_GONGITSUNE = "jf_gongitsune"
        JF_NEZUMI = "jf_nezumi"
        JF_TEBUKURO = "jf_tebukuro"
        # jm_*
        JM_KUMO = "jm_kumo"
        # pf_*
        PF_DORA = "pf_dora"
        # pm_*
        PM_ALEX = "pm_alex"
        PM_SANTA = "pm_santa"
        # zf_*
        ZF_XIAOBEI = "zf_xiaobei"
        ZF_XIAONI = "zf_xiaoni"
        ZF_XIAOXIAO = "zf_xiaoxiao"
        ZF_XIAOYI = "zf_xiaoyi"

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id="hexgrad/Kokoro-82M"),
        title="Model",
        description="The Kokoro model repository.",
    )
    text: str = Field(
        default="Hello from Kokoro.",
        title="Text",
        description="The text to convert to speech.",
    )
    lang_code: LanguageCode = Field(
        default=LanguageCode.AMERICAN_ENGLISH,
        title="Language",
        description="Language for pronunciation. Choose based on your text's language.",
    )
    voice: Voice = Field(
        default=Voice.AF_HEART,
        title="Voice",
        description="The voice to use. af_* = American female, am_* = American male, bf_* = British female, etc.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        title="Speed",
        description="Speech speed multiplier: 0.5 = half speed, 1.0 = normal, 2.0 = double speed.",
    )

    _kpipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToSpeech(
                repo_id="hexgrad/Kokoro-82M",
                allow_patterns=["*.json", "*.pth", "voices/*.pt"],
            )
        ]

    def get_model_id(self):
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        from kokoro.pipeline import KPipeline

        # Initialize and cache the Kokoro pipeline
        device = context.device
        self._kpipeline = KPipeline(
            lang_code=self.lang_code,
            repo_id=self.get_model_id(),
            device=device if device else None,
        )

    async def move_to_device(self, device: str):
        if (
            self._kpipeline is not None
            and getattr(self._kpipeline, "model", None) is not None
        ):
            # KPipeline holds a torch.nn.Module in .model
            self._kpipeline.model.to(device)  # type: ignore

    class OutputType(TypedDict):
        audio: AudioRef | None
        chunk: Chunk

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        assert self._kpipeline is not None, "Kokoro pipeline not initialized"

        text = self.text.replace("\n", " ")

        generator = self._kpipeline(
            text,
            voice=self.voice.value,
            speed=self.speed,
        )

        audio_chunks: list[np.ndarray] = []
        for result in generator:
            audio = result.audio
            if audio is None:
                continue
            audio = audio.detach().cpu().numpy()
            # Convert to int16 for audio output
            if audio.dtype != np.int16:
                # Assume audio is float32 in [-1, 1], scale to int16
                audio_int16 = np.clip(audio, -1.0, 1.0)
                audio_int16 = (audio_int16 * 32767.0).astype(np.int16)
            else:
                audio_int16 = audio
            audio_chunks.append(audio_int16)
            audio_base64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
            chunk = Chunk(
                content=audio_base64,
                content_type="audio",
                content_metadata={
                    "sample_rate": 24_000,
                    "channels": 1,
                    "dtype": "int16",
                },
                done=False,
            )
            yield {"chunk": chunk, "audio": None}

        if not audio_chunks:
            raise ValueError("Kokoro did not produce any audio")

        if len(audio_chunks) == 1:
            combined = audio_chunks[0]
        else:
            combined = np.concatenate(audio_chunks)

        # Kokoro outputs 24kHz mono
        yield {
            "audio": await context.audio_from_numpy(combined, 24_000),
            "chunk": Chunk(content="", done=True, content_type="audio"),
        }

    def requires_gpu(self) -> bool:
        return True


# class ParlerTTS(HuggingFacePipelineNode):
#     """
#     Generates speech from text using the Parler TTS model.
#     tts, audio, speech, huggingface

#     Use cases:
#     - Create voice content for apps and websites
#     - Generate natural-sounding speech for various applications
#     - Produce audio narrations for videos or presentations
#     """

#     model: HFTextToSpeech = Field(
#         default=HFTextToSpeech(),
#         title="Model ID on Huggingface",
#         description="The model ID to use for text-to-speech generation",
#     )
#     prompt: str = Field(
#         default="Hey, how are you doing today?",
#         title="Prompt",
#         description="The text to convert to speech",
#     )
#     description: str = Field(
#         default="A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
#         title="Description",
#         description="A description of the desired speech characteristics",
#     )

#     _model: ParlerTTSForConditionalGeneration | None = None
#     _tokenizer: AutoTokenizer | None = None

#     @classmethod
#     def get_recommended_models(cls) -> list[HuggingFaceModel]:
#         return [
#             HFTextToSpeech(
#                 repo_id="parler-tts/parler-tts-mini-v1",
#             ),
#             HFTextToSpeech(
#                 repo_id="parler-tts/parler-tts-large-v1",
#             ),
#         ]

#     def get_model_id(self):
#         return self.model.repo_id

#     async def preload_model(self, context: ProcessingContext):
#         self._model = await self.load_model(
#             context=context,
#             model_class=ParlerTTSForConditionalGeneration,
#             model_id=self.get_model_id(),
#             variant=None,
#             torch_dtype=torch.float32,
#         )
#         self._tokenizer = AutoTokenizer.from_pretrained(self.get_model_id())  # type: ignore

#     async def move_to_device(self, device: str):
#         if self._model is not None:
#             self._model.to(device)  # type: ignore

#     async def process(self, context: ProcessingContext) -> AudioRef:
#         if self._model is None or self._tokenizer is None:
#             raise ValueError("Model or tokenizer not initialized")

#         device = context.device

#         input_ids = self._tokenizer(self.description, return_tensors="pt").input_ids.to(  # type: ignore
#             device
#         )
#         prompt_input_ids = self._tokenizer(
#             self.prompt, return_tensors="pt"
#         ).input_ids.to(  # type: ignore
#             device
#         )

#         generation = self._model.generate(
#             input_ids=input_ids, prompt_input_ids=prompt_input_ids
#         )
#         audio_arr = generation.cpu().numpy().squeeze()  # type: ignore

#         return await context.audio_from_numpy(
#             audio_arr, self._model.config.sampling_rate
#         )


class TextToSpeech(HuggingFacePipelineNode):
    """
    Converts text to speech using various Hugging Face TTS models for multiple languages.
    tts, audio, speech, huggingface, voice, speak

    Use cases:
    - Generate speech from text for applications and websites
    - Create voice content for virtual assistants and chatbots
    - Produce audio narrations for videos and presentations
    - Build accessibility features with text-to-speech output
    - Generate multilingual speech using MMS-TTS models
    """

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(),
        title="Model",
        description="The TTS model to use. facebook/mms-tts-* models support many languages (eng=English, fra=French, deu=German, kor=Korean, etc.).",
    )
    text: str = Field(
        default="Hello, this is a test of the text-to-speech system.",
        title="Input Text",
        description="The text to convert to speech.",
    )
    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToSpeech(
                repo_id="facebook/mms-tts-eng",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
            HFTextToSpeech(
                repo_id="facebook/mms-tts-kor",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
            HFTextToSpeech(
                repo_id="facebook/mms-tts-fra",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
            HFTextToSpeech(
                repo_id="facebook/mms-tts-deu",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
        ]

    def get_model_id(self):
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        import torch

        self._pipeline = await self.load_pipeline(
            context,
            "text-to-speech",
            self.get_model_id(),
            device=context.device,
            torch_dtype=torch.float32,
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> AudioRef:
        assert self._pipeline is not None, "Pipeline not initialized"

        result = await self.run_pipeline_in_thread(self.text)

        if isinstance(result, dict) and "audio" in result:
            audio_array = result["audio"]
        elif isinstance(result, tuple) and len(result) == 2:
            audio_array, sample_rate = result
        else:
            raise ValueError("Unexpected output format from the TTS pipeline")

        # Assuming a default sample rate of 16000 if not provided
        sample_rate = getattr(self._pipeline, "sampling_rate", 16000)

        return await context.audio_from_numpy(audio_array, sample_rate)


# class LoadSpeakerEmbedding(BaseNode):
#     """
#     Loads a speaker embedding from a dataset.
#     """

#     dataset_name: str = Field(
#         default="Matthijs/cmu-arctic-xvectors",
#         description="The name of the dataset containing speaker embeddings",
#     )
#     embedding_index: int = Field(
#         default=0, description="The index of the embedding to use"
#     )

#     async def process(self, context: ProcessingContext) -> Tensor:
#         from datasets import load_dataset

#         embeddings_dataset = load_dataset(self.dataset_name, split="validation")
#         speaker_embeddings = torch.tensor(
#             embeddings_dataset[self.embedding_index]["xvector"]  # type: ignore
#         ).unsqueeze(0)
#         return Tensor(value=speaker_embeddings.tolist(), dtype="float32")


# class SpeechT5(BaseNode):
#     """
#     A complete pipeline for text-to-speech using SpeechT5.
#     """

#     text: str = Field(default=str, description="The input text to convert to speech")
#     # speaker_embedding: Tensor = Field(
#     #     default=Tensor(), description="The index of the embedding to use"
#     # )

#     _processor: SpeechT5Processor | None = None
#     _model: SpeechT5ForTextToSpeech | None = None
#     # _vododer: SpeechT5HifiGan | None = None

#     async def preload_model(self, context: ProcessingContext):
#         model_name = "microsoft/speecht5_tts"
#         self._processor = SpeechT5Processor.from_pretrained(model_name)  # type: ignore
#         self._model = SpeechT5ForTextToSpeech.from_pretrained(model_name)  # type: ignore
#         # self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

#     async def process(self, context: ProcessingContext) -> AudioRef:
#         # inputs = self._processor(text=self.text, return_tensors="pt")
#         assert self._processor is not None
#         assert self._model is not None

#         # speaker_embedding = self.speaker_embedding.value
#         inputs = self._processor(self.text)

#         assert inputs is not None

#         speech = self._model.generate_speech(
#             inputs["input_ids"],
#             speaker_embedding=speaker_embedding,  # type: ignore
#         )

#         return await context.audio_from_numpy(speech.numpy(), sample_rate=16000)
