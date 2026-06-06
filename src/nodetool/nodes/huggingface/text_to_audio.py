from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field

from nodetool.metadata.types import (
    AudioRef,
    HFTextToAudio,
    HuggingFaceModel,
)
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE
from nodetool.nodes.huggingface.stable_diffusion_base import (
    is_mps_device,
    maybe_enable_cpu_offload,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.memory_utils import run_gc
from nodetool.workflows.types import NodeProgress

if TYPE_CHECKING:
    import torch
    import torchaudio
    from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline
    from diffusers.pipelines.audioldm.pipeline_audioldm import AudioLDMPipeline
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.pipelines.musicldm.pipeline_musicldm import MusicLDMPipeline
    from diffusers.pipelines.stable_audio.pipeline_stable_audio import (
        StableAudioPipeline,
    )
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    from transformers import AutoTokenizer
    from transformers import AutoFeatureExtractor, set_seed


class MusicGen(HuggingFacePipelineNode):
    """
    Generates music and audio from text descriptions using Meta's MusicGen models.
    audio, music, generation, huggingface, text-to-audio, soundtrack

    Use cases:
    - Create custom background music for videos, games, and podcasts
    - Generate sound effects from textual descriptions
    - Prototype musical ideas and compositions quickly
    - Produce royalty-free audio content for creative projects
    - Build AI-powered music generation applications
    """

    model: HFTextToAudio = Field(
        default=HFTextToAudio(),
        title="Model",
        description="The MusicGen model variant. Small is fastest; Large offers best quality; Melody can condition on audio input; Stereo produces 2-channel output.",
    )
    prompt: str = Field(
        default="",
        title="Text Prompt",
        description="Describe the music you want to generate (e.g., 'upbeat jazz piano with drums' or 'calm ambient soundscape').",
    )
    max_new_tokens: int = Field(
        default=1024,
        title="Max New Tokens",
        description="Controls audio length: ~256 tokens ≈ 5 seconds. Higher values produce longer audio.",
    )

    _processor: Any = None
    _model: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "MusicGen"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToAudio(
                repo_id="facebook/musicgen-small",
                allow_patterns=["*.safetensors", "*.json", "*.model"],
            ),
            HFTextToAudio(
                repo_id="facebook/musicgen-medium",
                allow_patterns=["*.safetensors", "*.json", "*.model"],
            ),
            HFTextToAudio(
                repo_id="facebook/musicgen-large",
                allow_patterns=["*.safetensors", "*.json", "*.model"],
            ),
            HFTextToAudio(
                repo_id="facebook/musicgen-melody",
                allow_patterns=["*.safetensors", "*.json", "*.model"],
            ),
            HFTextToAudio(
                repo_id="facebook/musicgen-stereo-small",
                allow_patterns=["*.safetensors", "*.json", "*.model"],
            ),
            HFTextToAudio(
                repo_id="facebook/musicgen-stereo-large",
                allow_patterns=["*.safetensors", "*.json", "*.model"],
            ),
        ]

    def get_model_id(self):
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        if not context.is_huggingface_model_cached(self.model.repo_id):
            raise ValueError(f"Download the model {self.model.repo_id} first")

        self._processor = await self.load_model(
            context, AutoProcessor, self.model.repo_id
        )
        self._model = await self.load_model(
            context=context,
            model_class=MusicgenForConditionalGeneration,
            model_id=self.model.repo_id,
            variant=None,
        )

    async def move_to_device(self, device: str):
        if self._model is not None:
            self._model.to(device)

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self._model is None:
            raise ValueError("Model not initialized")

        import torch

        inputs = self._processor(
            text=[self.prompt],
            padding=True,
            return_tensors="pt",
        )

        inputs["input_ids"] = inputs["input_ids"].to(context.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(context.device)

        with torch.inference_mode():
            audio_values = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
        sampling_rate = self._model.config.audio_encoder.sampling_rate

        run_gc("After MusicGen inference", log_before_after=False)
        return await context.audio_from_numpy(
            audio_values[0].cpu().numpy(), sampling_rate
        )


class MusicLDM(HuggingFacePipelineNode):
    """
    Generates music from text descriptions using latent diffusion models.
    audio, music, generation, huggingface, text-to-audio, diffusion

    Use cases:
    - Create custom background music for videos and games
    - Generate music clips based on textual mood descriptions
    - Produce audio content for multimedia projects
    - Explore AI-generated music for creative inspiration
    """

    model: HFTextToAudio = Field(
        default=HFTextToAudio(),
        title="Model",
        description="The MusicLDM model to use for audio generation.",
    )

    prompt: str = Field(
        default="",
        title="Text Prompt",
        description="Describe the music you want (e.g., 'electronic dance music with heavy bass').",
    )
    num_inference_steps: int = Field(
        default=10,
        title="Inference Steps",
        description="Number of denoising steps. More steps = better quality but slower generation.",
    )
    audio_length_in_s: float = Field(
        default=5.0,
        title="Audio Length",
        description="Duration of the generated audio in seconds.",
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "MusicLDM"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToAudio(
                repo_id="ucsd-reach/musicldm",
                allow_patterns=["*.safetensors", "*.json", "*.txt"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.musicldm.pipeline_musicldm import MusicLDMPipeline

        self._pipeline = await self.load_model(
            context, MusicLDMPipeline, self.model.repo_id
        )

    async def move_to_device(self, device: str):
        if self._pipeline:
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> AudioRef:
        assert self._pipeline is not None, "Pipeline not initialized"
        audio = await self.run_pipeline_in_thread(
            self.prompt,
            num_inference_steps=self.num_inference_steps,
            audio_length_in_s=self.audio_length_in_s,
        )
        audio = audio.audios[0]

        run_gc("After MusicLDM inference", log_before_after=False)
        return await context.audio_from_numpy(audio, 16_000)


class AudioLDM(HuggingFacePipelineNode):
    """
    Generates audio from text prompts using the AudioLDM latent diffusion model.
    audio, generation, AI, text-to-audio, sound-effects

    Use cases:
    - Create custom music clips from text descriptions
    - Generate sound effects for videos, games, and media
    - Produce background audio for creative projects
    - Explore AI-generated soundscapes and ambient audio
    """

    prompt: str = Field(
        default="Techno music with a strong, upbeat tempo and high melodic riffs",
        description="Text description of the audio you want to generate.",
    )
    num_inference_steps: int = Field(
        default=10,
        description="Denoising steps. More steps = better quality but slower. 10-50 is typical.",
        ge=1,
        le=100,
    )
    audio_length_in_s: float = Field(
        default=5.0,
        description="Duration of the generated audio in seconds (1-30).",
        ge=1.0,
        le=30.0,
    )
    seed: int = Field(
        default=0,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "AudioLDM"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToAudio(
                repo_id="cvssp/audioldm-s-full-v2",
                allow_patterns=["*.safetensors", "*.json", "*.txt"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.audioldm.pipeline_audioldm import AudioLDMPipeline

        self._pipeline = await self.load_model(
            context, AudioLDMPipeline, "cvssp/audioldm-s-full-v2"
        )

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        def progress_callback(
            step: int, timestep: int, latents: "torch.FloatTensor"
        ) -> None:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )

        audio = await self.run_pipeline_in_thread(
            self.prompt,
            num_inference_steps=self.num_inference_steps,
            audio_length_in_s=self.audio_length_in_s,
            generator=generator,
            callback=progress_callback,
            callback_steps=1,
        )
        audio = audio.audios[0]

        run_gc("After AudioLDM inference", log_before_after=False)
        return await context.audio_from_numpy(audio, 16000)


class AudioLDM2(HuggingFacePipelineNode):
    """
    Generates audio from text prompts using the improved AudioLDM2 model.
    audio, generation, AI, text-to-audio, sound-effects, sound-design

    Use cases:
    - Create realistic sound effects from text descriptions
    - Generate background audio for videos and games
    - Produce environmental soundscapes for multimedia
    - Explore creative AI-generated audio for sound design
    """

    prompt: str = Field(
        default="The sound of a hammer hitting a wooden surface.",
        description="Text description of the audio you want to generate.",
    )
    negative_prompt: str = Field(
        default="Low quality.",
        description="Describe what to avoid in the generated audio (e.g., 'noise, distortion').",
    )
    num_inference_steps: int = Field(
        default=200,
        description="Denoising steps. 200 is recommended for quality; lower for speed.",
        ge=50,
        le=500,
    )
    audio_length_in_s: float = Field(
        default=10.0,
        description="Duration of the generated audio in seconds (1-30).",
        ge=1.0,
        le=30.0,
    )
    num_waveforms_per_prompt: int = Field(
        default=3,
        description="Number of audio variations to generate. Best result is returned.",
        ge=1,
        le=5,
    )
    seed: int = Field(
        default=0,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "AudioLDM2"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToAudio(
                repo_id="cvssp/audioldm2",
                allow_patterns=["*.safetensors", "*.json", "*.txt"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline

        self._pipeline = await self.load_model(
            context, AudioLDM2Pipeline, "cvssp/audioldm2", variant=None
        )

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        def progress_callback(
            step: int, timestep: int, latents: "torch.FloatTensor"
        ) -> None:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )

        audio = await self.run_pipeline_in_thread(
            self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            audio_length_in_s=self.audio_length_in_s,
            num_waveforms_per_prompt=self.num_waveforms_per_prompt,
            generator=generator,
            callback=progress_callback,
            callback_steps=1,
        )
        audio = audio.audios[0]

        run_gc("After AudioLDM2 inference", log_before_after=False)
        return await context.audio_from_numpy(audio, 16000)


class DanceDiffusion(HuggingFacePipelineNode):
    """
    Generates AI-composed music using the DanceDiffusion unconditional audio model.
    audio, generation, AI, music, text-to-audio, unconditional

    Use cases:
    - Create AI-generated music samples and loops
    - Produce background music for videos and games
    - Generate experimental audio content
    - Explore AI-composed musical ideas and patterns
    """

    audio_length_in_s: float = Field(
        default=4.0,
        description="Duration of the generated audio in seconds (1-30).",
        ge=1.0,
        le=30.0,
    )
    num_inference_steps: int = Field(
        default=50,
        description="Denoising steps. More steps = better quality but slower. 50-200 is typical.",
        ge=1,
        le=1000,
    )
    seed: int = Field(
        default=0,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "Dance Diffusion"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToAudio(
                repo_id="harmonai/maestro-150k",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline

        self._pipeline = await self.load_model(
            context, DiffusionPipeline, "harmonai/maestro-150k"
        )

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        audio = await self.run_pipeline_in_thread(
            audio_length_in_s=self.audio_length_in_s,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        )
        audio = audio.audios[0]

        run_gc("After DanceDiffusion inference", log_before_after=False)
        return await context.audio_from_numpy(audio, 16000)


class StableAudioNode(HuggingFacePipelineNode):
    """
    Generates high-quality, long-form audio from text prompts using Stability AI's Stable Audio.
    audio, generation, synthesis, text-to-audio, music, sound-effects

    Use cases:
    - Create professional-quality music and soundtracks
    - Generate ambient sounds and environmental audio
    - Produce sound effects for multimedia projects
    - Create experimental and artistic audio content
    - Generate up to 5 minutes of continuous audio
    """

    prompt: str = Field(
        default="A peaceful piano melody.",
        description="Text description of the audio you want to generate.",
    )
    negative_prompt: str = Field(
        default="Low quality.",
        description="Describe what to avoid in the generated audio (e.g., 'noise, distortion').",
    )
    duration: float = Field(
        default=10.0,
        description="Duration of the generated audio in seconds. Stable Audio supports up to 300 seconds.",
        ge=1.0,
        le=300.0,
    )
    num_inference_steps: int = Field(
        default=200,
        description="Denoising steps. 200 is recommended for quality; lower values for faster generation.",
        ge=50,
        le=500,
    )
    seed: int = Field(
        default=0,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "Stable Audio"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToAudio(
                repo_id="stabilityai/stable-audio-open-1.0",
                allow_patterns=["*.safetensors", "*.json", "*.txt"],
            ),
        ]

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.stable_audio.pipeline_stable_audio import (
            StableAudioPipeline,
        )

        self._pipeline = await self.load_model(
            context=context,
            model_class=StableAudioPipeline,
            model_id="stabilityai/stable-audio-open-1.0",
            variant=None,
        )

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = torch.Generator(device="cpu")

        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        def progress_callback(
            step: int, timestep: int, latents: "torch.FloatTensor"
        ) -> None:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )

        audio = await self.run_pipeline_in_thread(
            self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            audio_end_in_s=self.duration,
            num_waveforms_per_prompt=1,
            generator=generator,
            callback=progress_callback,
        )
        audio = audio.audios[0]

        output = audio.T.float().cpu().numpy()
        sampling_rate = self._pipeline.vae.sampling_rate
        run_gc("After StableAudio inference", log_before_after=False)
        audio = await context.audio_from_numpy(output, sampling_rate)
        return audio


class AceStep(HuggingFacePipelineNode):
    """
    Generates music with optional lyrics from text prompts using ACE-Step 1.5.
    audio, music, generation, text-to-music, lyrics, song

    Use cases:
    - Generate commercial-grade stereo music from text descriptions
    - Create songs with structured lyrics ([verse], [chorus], etc.)
    - Produce backing tracks and instrumental pieces
    - Build AI music generation applications
    """

    model: HFTextToAudio = Field(
        default=HFTextToAudio(repo_id="ACE-Step/acestep-v15-xl-turbo-diffusers"),
        description="The ACE-Step model to use for music generation.",
    )
    prompt: str = Field(
        default="A beautiful piano piece with soft melodies and gentle rhythm",
        description="Describe the music style, instruments, mood, and tempo.",
    )
    lyrics: str = Field(
        default="[verse]\nSoft notes in the morning light\nDancing through the air so bright\n[chorus]\nMusic fills the air tonight\nEvery note feels just right",
        description="Lyrics for the music. Use structure tags like [verse], [chorus], [bridge].",
    )
    audio_duration: float = Field(
        default=30.0,
        description="Duration of the generated music in seconds.",
        ge=10.0,
        le=600.0,
    )
    vocal_language: str = Field(
        default="en",
        description="Language code for the lyrics (e.g. 'en', 'zh', 'ja').",
    )
    num_inference_steps: int = Field(
        default=8,
        description="Denoising steps. The turbo model is designed for 8 steps.",
        ge=1,
        le=200,
    )
    guidance_scale: float = Field(
        default=7.0,
        description="Guidance scale for CFG. Ignored by guidance-distilled turbo checkpoints.",
        ge=1.0,
        le=15.0,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "ACE-Step"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "lyrics", "audio_duration"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        allow = ["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"]
        return [
            # Turbo: guidance-distilled, 8 steps (CFG ignored).
            HFTextToAudio(
                repo_id="ACE-Step/acestep-v15-xl-turbo-diffusers",
                allow_patterns=allow,
            ),
            # Base: CFG enabled, higher quality with more steps.
            HFTextToAudio(
                repo_id="ACE-Step/acestep-v15-base",
                allow_patterns=allow,
            ),
            # SFT: supervised fine-tuned, CFG enabled.
            HFTextToAudio(
                repo_id="ACE-Step/acestep-v15-sft",
                allow_patterns=allow,
            ),
        ]

    def get_model_id(self) -> str:
        return self.model.repo_id or "ACE-Step/acestep-v15-xl-turbo-diffusers"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.ace_step.pipeline_ace_step import AceStepPipeline
        from nodetool.nodes.huggingface.stable_diffusion_base import (
            available_torch_dtype,
        )

        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(
                f"Model {self.get_model_id()} must be downloaded first from the recommended models"
            )

        self._pipeline = await self.load_model(
            context=context,
            model_class=AceStepPipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
            local_files_only=True,
        )
        maybe_enable_cpu_offload(self._pipeline, self.enable_cpu_offload)

    async def move_to_device(self, device: str):
        # On MPS we skip offload and load fully onto the device, so move here.
        if self._pipeline is not None and (
            not self.enable_cpu_offload or is_mps_device()
        ):
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        def progress_callback(step: int, timestep: int, latents: Any) -> None:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            lyrics=self.lyrics,
            audio_duration=self.audio_duration,
            vocal_language=self.vocal_language,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            callback=progress_callback,
        )

        # ACE-Step returns stereo tensors of shape [channels, samples] at pipe.sample_rate.
        audio = output.audios[0]
        waveform = audio.T.float().cpu().numpy()
        sampling_rate = int(getattr(self._pipeline, "sample_rate", 48000))
        run_gc("After ACE-Step inference", log_before_after=False)
        return await context.audio_from_numpy(waveform, sampling_rate)


class AceStepTaskBaseNode(HuggingFacePipelineNode):
    """
    Shared base for the ACE-Step 1.5 audio-to-audio task nodes.

    Mirrors the loading and offload conventions of the ``AceStep`` text-to-music
    node (model selector, CPU offload, cache guard) and adds the audio helpers
    used by the cover / repaint / extract / lego / complete tasks. Concrete task
    nodes add their task-specific inputs and call ``_run`` with the matching
    ``task_type``.
    """

    model: HFTextToAudio = Field(
        default=HFTextToAudio(repo_id="ACE-Step/acestep-v15-xl-turbo-diffusers"),
        description="The ACE-Step model to use for generation.",
    )
    prompt: str = Field(
        default="A beautiful piano piece with soft melodies and gentle rhythm",
        description="Describe the music style, instruments, mood, and tempo.",
    )
    lyrics: str = Field(
        default="",
        description="Lyrics for the music. Use structure tags like [verse], [chorus], [bridge]. Leave empty for instrumental.",
    )
    vocal_language: str = Field(
        default="en",
        description="Language code for the lyrics (e.g. 'en', 'zh', 'ja').",
    )
    num_inference_steps: int = Field(
        default=8,
        description="Denoising steps. The turbo model is designed for 8 steps.",
        ge=1,
        le=200,
    )
    guidance_scale: float = Field(
        default=7.0,
        description="Guidance scale for CFG. Ignored by guidance-distilled turbo checkpoints.",
        ge=1.0,
        le=15.0,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not AceStepTaskBaseNode

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        allow = ["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"]
        return [
            # Turbo: guidance-distilled, 8 steps (CFG ignored).
            HFTextToAudio(
                repo_id="ACE-Step/acestep-v15-xl-turbo-diffusers",
                allow_patterns=allow,
            ),
            # Base: CFG enabled, higher quality with more steps.
            HFTextToAudio(
                repo_id="ACE-Step/acestep-v15-base",
                allow_patterns=allow,
            ),
            # SFT: supervised fine-tuned, CFG enabled.
            HFTextToAudio(
                repo_id="ACE-Step/acestep-v15-sft",
                allow_patterns=allow,
            ),
        ]

    def get_model_id(self) -> str:
        return self.model.repo_id or "ACE-Step/acestep-v15-xl-turbo-diffusers"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.ace_step.pipeline_ace_step import AceStepPipeline
        from nodetool.nodes.huggingface.stable_diffusion_base import (
            available_torch_dtype,
        )

        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(
                f"Model {self.get_model_id()} must be downloaded first from the recommended models"
            )

        self._pipeline = await self.load_model(
            context=context,
            model_class=AceStepPipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
            local_files_only=True,
        )
        maybe_enable_cpu_offload(self._pipeline, self.enable_cpu_offload)

    async def move_to_device(self, device: str):
        # On MPS we skip offload and load fully onto the device, so move here.
        if self._pipeline is not None and (
            not self.enable_cpu_offload or is_mps_device()
        ):
            self._pipeline.to(device)

    async def _audio_to_tensor(
        self, context: ProcessingContext, audio: AudioRef
    ) -> "torch.Tensor":
        """Decode an AudioRef into a stereo ``[channels, samples]`` tensor at the
        pipeline sample rate (48 kHz for the released checkpoints)."""
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        sample_rate = int(getattr(self._pipeline, "sample_rate", 48000))
        samples, _, _ = await context.audio_to_numpy(audio, sample_rate=sample_rate)
        tensor = torch.from_numpy(samples).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        # ACE-Step expects stereo input; duplicate a mono channel.
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(2, 1)
        return tensor

    async def _run(
        self, context: ProcessingContext, *, task_type: str, **task_kwargs: Any
    ) -> AudioRef:
        """Run the pipeline with the shared conditioning plus task-specific
        keyword arguments and return the generated stereo audio."""
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        def progress_callback(step: int, timestep: int, latents: Any) -> None:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            lyrics=self.lyrics,
            vocal_language=self.vocal_language,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            task_type=task_type,
            generator=generator,
            callback=progress_callback,
            **task_kwargs,
        )

        # ACE-Step returns stereo tensors of shape [channels, samples] at pipe.sample_rate.
        audio = output.audios[0]
        waveform = audio.T.float().cpu().numpy()
        sampling_rate = int(getattr(self._pipeline, "sample_rate", 48000))
        run_gc(f"After ACE-Step {task_type} inference", log_before_after=False)
        return await context.audio_from_numpy(waveform, sampling_rate)


class AceStepCover(AceStepTaskBaseNode):
    """
    Re-records source audio in a new timbre using ACE-Step 1.5 cover mode.
    audio, music, cover, timbre-transfer, voice-conversion, ace-step

    Use cases:
    - Re-sing an existing song with a different voice or instrument timbre
    - Transfer the style of a reference recording onto source audio
    - Create covers that keep the melody but change the sound
    """

    src_audio: AudioRef = Field(
        default=AudioRef(),
        title="Source Audio",
        description="Source audio (stereo, 48 kHz) providing the musical content to cover.",
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(),
        title="Reference Audio",
        description="Reference audio whose timbre/style is transferred onto the source.",
    )
    audio_cover_strength: float = Field(
        default=1.0,
        title="Cover Strength",
        description="Strength of the cover blending (0.0 keeps the source, 1.0 fully re-records).",
        ge=0.0,
        le=1.0,
    )

    @classmethod
    def get_title(cls) -> str:
        return "ACE-Step Cover"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "src_audio", "reference_audio", "audio_cover_strength"]

    def required_inputs(self):
        return ["src_audio", "reference_audio"]

    async def process(self, context: ProcessingContext) -> AudioRef:
        src = await self._audio_to_tensor(context, self.src_audio)
        reference = await self._audio_to_tensor(context, self.reference_audio)
        return await self._run(
            context,
            task_type="cover",
            src_audio=src,
            reference_audio=reference,
            audio_cover_strength=self.audio_cover_strength,
        )


class AceStepRepaint(AceStepTaskBaseNode):
    """
    Regenerates a section of existing audio while keeping the rest with ACE-Step 1.5.
    audio, music, repaint, inpainting, edit, ace-step

    Use cases:
    - Replace a verse or chorus while preserving the surrounding song
    - Fix or rework a specific time range of a track
    - Generate variations of one section without re-rendering the whole piece
    """

    src_audio: AudioRef = Field(
        default=AudioRef(),
        title="Source Audio",
        description="Audio (stereo, 48 kHz) to repaint. The region outside the range is kept.",
    )
    repainting_start: float = Field(
        default=0.0,
        title="Repaint Start",
        description="Start time in seconds of the region to regenerate.",
        ge=0.0,
    )
    repainting_end: float = Field(
        default=-1.0,
        title="Repaint End",
        description="End time in seconds of the region to regenerate. Use -1 for until the end.",
        ge=-1.0,
    )

    @classmethod
    def get_title(cls) -> str:
        return "ACE-Step Repaint"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "src_audio", "repainting_start", "repainting_end"]

    def required_inputs(self):
        return ["src_audio"]

    async def process(self, context: ProcessingContext) -> AudioRef:
        src = await self._audio_to_tensor(context, self.src_audio)
        return await self._run(
            context,
            task_type="repaint",
            src_audio=src,
            repainting_start=self.repainting_start,
            repainting_end=self.repainting_end,
        )


class AceStepExtract(AceStepTaskBaseNode):
    """
    Extracts a single track (e.g. vocals, drums) from audio with ACE-Step 1.5.
    audio, music, extract, stem-separation, vocals, drums, ace-step

    Use cases:
    - Isolate vocals, drums, bass, or other stems from a mix
    - Build acapellas or instrumentals from a full track
    - Prepare stems for remixing and sampling
    """

    src_audio: AudioRef = Field(
        default=AudioRef(),
        title="Source Audio",
        description="Audio (stereo, 48 kHz) to extract a track from.",
    )
    track_name: str = Field(
        default="vocals",
        title="Track Name",
        description="Track to extract (e.g. 'vocals', 'drums', 'bass', 'guitar', 'piano').",
    )

    @classmethod
    def get_title(cls) -> str:
        return "ACE-Step Extract"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "src_audio", "track_name"]

    def required_inputs(self):
        return ["src_audio"]

    async def process(self, context: ProcessingContext) -> AudioRef:
        src = await self._audio_to_tensor(context, self.src_audio)
        return await self._run(
            context,
            task_type="extract",
            src_audio=src,
            track_name=self.track_name,
        )


class AceStepLego(AceStepTaskBaseNode):
    """
    Generates a specific track conditioned on existing audio with ACE-Step 1.5 lego mode.
    audio, music, lego, accompaniment, stem-generation, ace-step

    Use cases:
    - Add a new instrument track that fits an existing arrangement
    - Generate a drum or bass line for a given musical context
    - Regenerate a single track within a chosen time range
    """

    src_audio: AudioRef = Field(
        default=AudioRef(),
        title="Source Audio",
        description="Audio (stereo, 48 kHz) providing the musical context.",
    )
    track_name: str = Field(
        default="drums",
        title="Track Name",
        description="Track to generate (e.g. 'vocals', 'drums', 'bass', 'guitar', 'piano').",
    )
    repainting_start: float = Field(
        default=0.0,
        title="Region Start",
        description="Start time in seconds of the region to generate.",
        ge=0.0,
    )
    repainting_end: float = Field(
        default=-1.0,
        title="Region End",
        description="End time in seconds of the region to generate. Use -1 for until the end.",
        ge=-1.0,
    )

    @classmethod
    def get_title(cls) -> str:
        return "ACE-Step Lego"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "src_audio", "track_name", "repainting_start"]

    def required_inputs(self):
        return ["src_audio"]

    async def process(self, context: ProcessingContext) -> AudioRef:
        src = await self._audio_to_tensor(context, self.src_audio)
        return await self._run(
            context,
            task_type="lego",
            src_audio=src,
            track_name=self.track_name,
            repainting_start=self.repainting_start,
            repainting_end=self.repainting_end,
        )


class AceStepComplete(AceStepTaskBaseNode):
    """
    Completes input audio with additional tracks using ACE-Step 1.5.
    audio, music, complete, arrangement, accompaniment, ace-step

    Use cases:
    - Flesh out a sparse demo with extra instrument tracks
    - Add accompaniment around a lead vocal or melody
    - Turn a single stem into a fuller arrangement
    """

    src_audio: AudioRef = Field(
        default=AudioRef(),
        title="Source Audio",
        description="Audio (stereo, 48 kHz) to complete with additional tracks.",
    )
    complete_track_classes: list[str] = Field(
        default_factory=list,
        title="Track Classes",
        description="Track classes to add (e.g. ['drums', 'bass', 'guitar']). Empty lets the model decide.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "ACE-Step Complete"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "src_audio", "complete_track_classes"]

    def required_inputs(self):
        return ["src_audio"]

    async def process(self, context: ProcessingContext) -> AudioRef:
        src = await self._audio_to_tensor(context, self.src_audio)
        return await self._run(
            context,
            task_type="complete",
            src_audio=src,
            complete_track_classes=self.complete_track_classes or None,
        )


class LongCatAudioDiT(HuggingFacePipelineNode):
    """
    Generates audio (ambience, sound effects) from text prompts using Meituan's LongCat-AudioDiT.
    audio, generation, text-to-audio, sound-effects, ambience, longcat

    Use cases:
    - Generate ambient soundscapes from text descriptions
    - Create sound effects for multimedia and games
    - Produce background audio for videos
    - Experiment with text-conditioned audio generation
    """

    model: HFTextToAudio = Field(
        default=HFTextToAudio(repo_id="ruixiangma/LongCat-AudioDiT-1B-Diffusers"),
        description="The LongCat-AudioDiT model to use for audio generation.",
    )
    prompt: str = Field(
        default="A calm ocean wave ambience with soft wind in the background.",
        description="Text description of the audio to generate.",
    )
    negative_prompt: str = Field(
        default="",
        description="Describe what to avoid in the generated audio.",
    )
    audio_duration: float = Field(
        default=5.0,
        description="Target duration of the generated audio in seconds.",
        ge=1.0,
        le=60.0,
    )
    num_inference_steps: int = Field(
        default=16,
        description="Denoising steps. 16-20 is typical.",
        ge=1,
        le=100,
    )
    guidance_scale: float = Field(
        default=4.0,
        description="How strongly to follow the prompt.",
        ge=0.0,
        le=15.0,
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducible generation. Use -1 for random.",
        ge=-1,
    )
    enable_cpu_offload: bool = Field(
        default=True,
        description="Offload model components to CPU to reduce VRAM usage.",
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "LongCat-AudioDiT"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "audio_duration", "num_inference_steps"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToAudio(
                repo_id="ruixiangma/LongCat-AudioDiT-1B-Diffusers",
                allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt", "*.json"],
            ),
        ]

    def get_model_id(self) -> str:
        return self.model.repo_id or "ruixiangma/LongCat-AudioDiT-1B-Diffusers"

    async def preload_model(self, context: ProcessingContext):
        from diffusers.pipelines.longcat_audio_dit.pipeline_longcat_audio_dit import (
            LongCatAudioDiTPipeline,
        )
        from nodetool.nodes.huggingface.stable_diffusion_base import (
            available_torch_dtype,
        )

        if not await HF_FAST_CACHE.resolve(self.get_model_id(), "model_index.json"):
            raise ValueError(
                f"Model {self.get_model_id()} must be downloaded first from the recommended models"
            )

        self._pipeline = await self.load_model(
            context=context,
            model_class=LongCatAudioDiTPipeline,
            model_id=self.get_model_id(),
            torch_dtype=available_torch_dtype(),
            device="cpu",
            local_files_only=True,
        )
        maybe_enable_cpu_offload(self._pipeline, self.enable_cpu_offload)

    async def move_to_device(self, device: str):
        # On MPS we skip offload and load fully onto the device, so move here.
        if self._pipeline is not None and (
            not self.enable_cpu_offload or is_mps_device()
        ):
            self._pipeline.to(device)

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        import torch

        generator = torch.Generator(device="cpu")
        if self.seed != -1:
            generator = generator.manual_seed(self.seed)

        def callback_on_step_end(
            pipeline: Any, step: int, timestep: int, callback_kwargs: dict
        ) -> dict:
            context.post_message(
                NodeProgress(
                    node_id=self.id,
                    progress=step,
                    total=self.num_inference_steps,
                )
            )
            return callback_kwargs

        output = await self.run_pipeline_in_thread(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt or None,
            audio_duration_s=self.audio_duration,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        # Output shape is (batch, channels, samples); take the first mono sample.
        audio = output.audios[0, 0]
        waveform = np.asarray(audio, dtype=np.float32)
        sampling_rate = int(getattr(self._pipeline, "sample_rate", 24000))
        run_gc("After LongCat-AudioDiT inference", log_before_after=False)
        return await context.audio_from_numpy(waveform, sampling_rate)


# class ParlerTTSNode(HuggingFacePipelineNode):
#     """
#     Generates speech using the Parler TTS model based on a text prompt and description.
#     audio, generation, AI, text-to-speech, TTS

#     Use cases:
#     - Generate natural-sounding speech from text
#     - Create voiceovers for videos or presentations
#     - Produce audio content for accessibility purposes
#     - Explore AI-generated speech with customizable characteristics
#     """

#     model: HFTextToSpeech = Field(
#         default=HFTextToSpeech(),
#         title="Model ID on Huggingface",
#         description="The model ID to use for the audio generation, must be a Parler TTS model",
#     )

#     prompt: str = Field(
#         default="Hello, how are you doing today?",
#         description="The text to be converted to speech.",
#     )
#     description: str = Field(
#         default="A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
#         description="A description of the desired speech characteristics.",
#     )
#     reference_audio: AudioRef | None = Field(
#         default=None,
#         description="Optional reference audio file for voice cloning",
#     )
#     reference_text: str | None = Field(
#         default=None,
#         description="Transcript of the reference audio for voice cloning",
#     )
#     max_length: int = Field(
#         default=512,
#         description="The maximum length of the input text",
#     )
#     seed: int = Field(
#         default=0,
#         description="Seed for the random number generator. Use -1 for a random seed.",
#         ge=-1,
#     )

#     _model: ParlerTTSForConditionalGeneration | None = None
#     _tokenizer: AutoTokenizer | None = None
#     _feature_extractor: AutoFeatureExtractor | None = None

#     @classmethod
#     def get_title(cls) -> str:
#         return "Parler TTS"

#     @classmethod
#     def get_recommended_models(cls) -> list[HuggingFaceModel]:
#         return [
#             HFTextToSpeech(
#                 repo_id="parler-tts/parler-tts-mini-v1",
#                 allow_patterns=["*.bin", "*.json", "*.txt"],
#             ),
#             HFTextToSpeech(
#                 repo_id="parler-tts/parler-tts-large-v1",
#                 allow_patterns=["*.bin", "*.json", "*.txt"],
#             ),
#         ]

#     async def preload_model(self, context: ProcessingContext):
#         self._model = await self.load_model(
#             context=context,
#             model_class=ParlerTTSForConditionalGeneration,
#             model_id=self.model.repo_id,
#             variant=None,
#             torch_dtype=None,
#         )
#         self._tokenizer = await self.load_model(
#             context, AutoTokenizer, self.model.repo_id
#         )
#         self._feature_extractor = await self.load_model(
#             context, AutoFeatureExtractor, self.model.repo_id
#         )

#     async def move_to_device(self, device: str):
#         if self._model is not None:
#             self._model.to(device)

#     async def process(self, context: ProcessingContext) -> AudioRef:
#         if (
#             self._model is None
#             or self._tokenizer is None
#             or self._feature_extractor is None
#         ):
#             raise ValueError("Model, tokenizer, or feature extractor not initialized")

#         # Set seeds for reproducibility
#         if self.seed != -1:
#             set_seed(self.seed)
#             torch.manual_seed(self.seed)
#             if torch.cuda.is_available():
#                 torch.cuda.manual_seed(self.seed)
#                 torch.cuda.manual_seed_all(self.seed)
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False

#         sampling_rate = self._model.config.sampling_rate

#         # Prepare input IDs for the description
#         input_ids = self._tokenizer(
#             self.description,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#         ).input_ids.to(
#             context.device
#         )

#         # Handle voice cloning if reference audio is provided
#         input_values = None
#         if self.reference_audio is not None and self.reference_text is not None:
#             # Load and preprocess reference audio
#             ref_audio, ref_sample_rate, num_channels = await context.audio_to_numpy(
#                 self.reference_audio
#             )
#             ref_audio_tensor = torch.from_numpy(ref_audio).float()
#             if num_channels > 1:
#                 ref_audio_tensor = ref_audio_tensor.mean(0)

#             # Resample if necessary
#             if ref_sample_rate != sampling_rate:
#                 ref_audio_tensor = torchaudio.functional.resample(
#                     ref_audio_tensor, ref_sample_rate, sampling_rate
#                 )

#             # Process reference audio
#             input_values = self._feature_extractor(
#                 ref_audio_tensor,
#                 sampling_rate=sampling_rate,
#                 return_tensors="pt",
#                 padding=True,
#                 max_length=self.max_length,
#             ).input_values.to(
#                 context.device
#             )

#         # Process the full prompt in a single pass
#         prompt_input_ids = self._tokenizer(
#             self.prompt,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#         ).input_ids.to(
#             context.device
#         )

#         gen_kwargs = {
#             "input_ids": input_ids,
#             "prompt_input_ids": prompt_input_ids,
#             # "max_new_tokens": 1000,
#         }

#         if input_values is not None:
#             gen_kwargs["input_values"] = input_values

#         with torch.inference_mode():
#             generation = self._model.generate(**gen_kwargs)
#             audio_arr = generation.cpu().numpy().squeeze()

#         return await context.audio_from_numpy(audio_arr, sampling_rate)
