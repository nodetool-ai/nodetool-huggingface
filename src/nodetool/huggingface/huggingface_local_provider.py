"""
HuggingFace local provider implementation.

This module implements the BaseProvider interface for locally cached HuggingFace models.
- Image models: Text2Image and ImageToImage using diffusion pipelines
  Supports both multi-file models (repo_id) and single-file models (repo_id:path.safetensors)
- TTS models: KokoroTTS and other HuggingFace TTS models
"""

from __future__ import annotations

import asyncio
import base64
import re
import os
import threading
import json
from queue import Queue
from typing import (
    Any,
    AsyncGenerator,
    List,
    Dict,
    Sequence,
    AsyncIterator,
    TYPE_CHECKING,
)
from io import BytesIO

import numpy as np
from pydub import AudioSegment
from PIL import Image
from pydantic import BaseModel

from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.integrations.huggingface.huggingface_models import (
    get_image_to_image_models_from_hf_cache,
    get_text_to_image_models_from_hf_cache,
    get_hf_language_models_from_hf_cache,
)
from nodetool.huggingface.image_to_image_pipelines import (
    load_image_to_image_pipeline,
)
from nodetool.huggingface.local_provider_utils import (
    _get_torch,
    _is_cuda_available,
    load_model,
    load_pipeline,
    pipeline_progress_callback,
)
from nodetool.huggingface.text_to_image_pipelines import (
    load_text_to_image_pipeline,
)
from nodetool.types.job import JobUpdate
from nodetool.types.model import UnifiedModel
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    ImageModel,
    Provider,
    TTSModel,
    Message,
    ASRModel,
    MessageTextContent,
    MessageImageContent,
    HFTextToSpeech,
    LanguageModel,
    VideoRef,
)
from nodetool.workflows.types import Chunk, NodeProgress
from nodetool.config.logging_config import get_logger
from nodetool.ml.core.model_manager import ModelManager
from nodetool.io.media_fetch import fetch_uri_bytes_and_mime_sync
from nodetool.workflows.recommended_models import get_recommended_models

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, TextStreamer
    from transformers.pipelines import pipeline as transformers_pipeline

log = get_logger(__name__)


@register_provider(Provider.HuggingFace)
class HuggingFaceLocalProvider(BaseProvider):
    """Local provider for HuggingFace models using cached diffusion pipelines."""

    provider_name = "hf_inference"

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables needed when running inside Docker."""
        # The nodes will handle HF_TOKEN internally
        return {}

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
    ) -> ImageBytes:
        """Generate an image from a text prompt using HuggingFace diffusion models.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            node_id: Optional node ID for progress tracking
        Returns:
            Raw image bytes as PNG

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        from nodetool.nodes.huggingface.image_to_image import pipeline_progress_callback

        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace image generation"
            )

        model_id = params.model.id
        # Sanitize model_id for composite IDs used in Model Packs.
        # When checking out specific single-file models (e.g. quantized versions),
        # the model.id is often formatted as "repo_id:filename" to ensure uniqueness
        # in the UI/State, while model.path holds the filename.
        # We need to extract just the repo_id for the hugginface_hub/diffusers calls.
        if params.model.path and ":" in model_id:
            parts = model_id.split(":")
            if len(parts) == 2 and parts[1] == params.model.path:
                model_id = parts[0]

        pipeline, _ = await load_text_to_image_pipeline(
            context=context,
            model_id=model_id,
            model_path=params.model.path,
            node_id=node_id,
        )

        # Set up generator for reproducibility
        generator = None
        if params.seed is not None and params.seed != -1:
            torch = _get_torch()
            generator = torch.Generator(device="cpu").manual_seed(params.seed)

        # Progress callback
        num_steps = params.num_inference_steps or 50

        # Generate the image off the event loop
        def _run_pipeline_sync():
            return pipeline(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or "",
                num_inference_steps=num_steps,
                guidance_scale=params.guidance_scale or 7.5,
                width=params.width or 512,
                height=params.height or 512,
                generator=generator,
                callback_on_step_end=pipeline_progress_callback(
                    node_id=node_id, total_steps=num_steps, context=context
                ),  # type: ignore
                callback_on_step_end_tensor_inputs=["latents"],
            )

        output = await asyncio.to_thread(_run_pipeline_sync)

        # Get the generated image
        pil_image = output.images[0]  # pyright: ignore[reportAttributeAccessIssue]

        # Convert PIL Image to bytes
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        return image_bytes

    async def image_to_image(
        self,
        image: ImageBytes,
        params: ImageToImageParams,
        context: ProcessingContext | None = None,
        timeout_s: int | None = None,
        node_id: str | None = None,
    ) -> ImageBytes:
        """Transform an image based on a text prompt using HuggingFace diffusion models.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            node_id: Optional node ID for progress tracking
        Returns:
            Raw image bytes as PNG

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace image generation"
            )

        # Convert input image bytes to PIL Image
        pil_image = Image.open(BytesIO(image))

        model_id = params.model.id
        # Sanitize model_id for composite IDs used in Model Packs.
        # When checking out specific single-file models (e.g. quantized versions),
        # the model.id is often formatted as "repo_id:filename" to ensure uniqueness
        # in the UI/State, while model.path holds the filename.
        # We need to extract just the repo_id for the hugginface_hub/diffusers calls.
        if params.model.path and ":" in model_id:
            parts = model_id.split(":")
            if len(parts) == 2 and parts[1] == params.model.path:
                model_id = parts[0]

        pipeline, _ = await load_image_to_image_pipeline(
            context=context,
            model_id=model_id,
            model_path=params.model.path,
            node_id=node_id,
        )

        # Set up generator for reproducibility
        generator = None
        if params.seed is not None and params.seed != -1:
            torch = _get_torch()
            generator = torch.Generator(device="cpu").manual_seed(params.seed)

        # Progress callback
        num_steps = params.num_inference_steps or 25

        # Generate the image off the event loop
        def _run_pipeline_sync():
            return pipeline(
                prompt=params.prompt,
                image=pil_image,
                negative_prompt=params.negative_prompt or "",
                strength=params.strength or 0.8,
                num_inference_steps=num_steps,
                guidance_scale=params.guidance_scale or 7.5,
                generator=generator,
                callback_on_step_end=pipeline_progress_callback(
                    node_id=node_id, total_steps=num_steps, context=context
                ),  # type: ignore
                callback_on_step_end_tensor_inputs=["latents"],
            )

        output = await asyncio.to_thread(_run_pipeline_sync)

        # Get the generated image
        pil_output = output.images[0]  # pyright: ignore[reportAttributeAccessIssue]

        # Convert PIL Image to bytes
        img_buffer = BytesIO()
        pil_output.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        return image_bytes

    async def text_to_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        speed: float = 1.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:  # Returns np.ndarray[np.int16]
        """Generate speech audio from text using local HuggingFace TTS models.

        Supports TTS models:
        - Kokoro models (hexgrad/Kokoro-82M) - streaming with voice support

        Args:
            text: Text to convert to speech
            model: Model repository ID (e.g., "hexgrad/Kokoro-82M")
            voice: Voice preset (only for Kokoro models, e.g., "af_heart")
            speed: Speech speed multiplier (0.5 to 2.0, only for Kokoro)
            timeout_s: Optional timeout in seconds
            context: Processing context
            **kwargs: Additional arguments (lang_code for Kokoro)

        Yields:
            numpy.ndarray: Int16 audio chunks at 24kHz mono

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        from nodetool.nodes.huggingface.text_to_speech import KokoroTTS
        from nodetool.nodes.huggingface.text_to_speech import TextToSpeech

        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace TTS generation"
            )

        # Determine which TTS node to use based on model ID
        model_lower = model.lower()

        if "kokoro" in model_lower:
            # Map voice string to Voice enum
            voice_value = voice or "af_heart"  # Default voice
            lang_code = kwargs.get("lang_code", "a")  # Default to American English

            node = KokoroTTS(
                model=HFTextToSpeech(repo_id=model),
                text=text,
                voice=KokoroTTS.Voice(voice_value),
                speed=max(0.5, min(2.0, speed)),
                lang_code=KokoroTTS.LanguageCode(lang_code),
            )

            # Preload model
            await node.preload_model(context)

            # Stream chunks using gen_process
            async for output in node.gen_process(context):
                # Only yield chunk data (not the final AudioRef)
                chunk = output.get("chunk")
                if chunk and chunk.content and not chunk.done:
                    # Decode base64 chunk to numpy array
                    audio_bytes = base64.b64decode(chunk.content)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    yield audio_array

        else:
            node = TextToSpeech(
                model=HFTextToSpeech(repo_id=model),
                text=text,
            )

            # Preload model
            await node.preload_model(context)

            # Process to get audio
            audio_ref = await node.process(context)

            # Convert AudioRef to numpy array
            audio_bytes = await context.asset_to_bytes(audio_ref)
            audio = AudioSegment.from_file(BytesIO(audio_bytes))

            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Resample to 24kHz if needed
            if audio.frame_rate != 24000:
                audio = audio.set_frame_rate(24000)

            # Ensure 16-bit sample width
            if audio.sample_width != 2:
                audio = audio.set_sample_width(2)

            # Convert to numpy array
            audio_array = np.array(audio.get_array_of_samples(), dtype=np.int16)

            # Yield in chunks (4096 samples at a time)
            chunk_size = 4096
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i : i + chunk_size]
                yield chunk

    async def automatic_speech_recognition(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio to text using HuggingFace Whisper models.

        Args:
            audio: Input audio as bytes (various formats supported)
            model: Model repository ID (e.g., "openai/whisper-large-v3")
            language: Optional ISO-639-1 language code to improve accuracy
            prompt: Optional text to guide the model's style (initial_prompt)
            temperature: Sampling temperature between 0 and 1 (default 0)
            timeout_s: Optional timeout in seconds
            context: Processing context (required)
            **kwargs: Additional parameters (return_timestamps, chunk_length_s)

        Returns:
            Transcribed text from the audio

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If transcription fails
        """
        if context is None:
            raise ValueError("ProcessingContext is required for HuggingFace ASR")

        log.debug(f"Transcribing audio with HuggingFace Whisper model: {model}")

        # Get or load the pipeline
        asr_pipeline = ModelManager.get_model(model)

        if not asr_pipeline:
            log.info(f"Loading automatic speech recognition pipeline: {model}")

            import torch
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline as create_pipeline,
            )

            # Determine torch dtype based on device
            torch_dtype = torch.float16 if _is_cuda_available() else torch.float32

            # Load model using helper
            hf_model = await load_model(
                node_id=None,
                context=context,
                model_class=AutoModelForSpeechSeq2Seq,
                model_id=model,
                torch_dtype=torch_dtype,
                skip_cache=False,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )

            # Load processor
            processor = AutoProcessor.from_pretrained(model)

            # Create pipeline
            asr_pipeline = create_pipeline(
                "automatic-speech-recognition",
                model=hf_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=context.device,
            )

            # Cache the pipeline
            ModelManager.set_model(None, model, asr_pipeline)

        audio_segment = AudioSegment.from_file(BytesIO(audio))
        # Whisper expects 16kHz mono audio
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

        # Convert to numpy array (float32)
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        # Normalize to [-1, 1] range
        samples = samples / (2**15)

        # Build pipeline kwargs
        pipeline_kwargs: dict[str, Any] = {
            "return_timestamps": kwargs.get("return_timestamps", False),
            "chunk_length_s": kwargs.get("chunk_length_s", 30.0),
            "generate_kwargs": {},
        }

        # Add language if specified
        if language:
            pipeline_kwargs["generate_kwargs"]["language"] = language

        # Add prompt if specified (Whisper uses initial_prompt)
        if prompt:
            pipeline_kwargs["generate_kwargs"]["initial_prompt"] = prompt

        # Add temperature if non-zero
        if temperature != 0.0:
            pipeline_kwargs["generate_kwargs"]["temperature"] = temperature

        # Run transcription in thread to avoid blocking
        def _transcribe():
            return asr_pipeline(samples, **pipeline_kwargs)

        result = await asyncio.to_thread(_transcribe)

        # Extract text from result
        if isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)

        log.debug(f"Transcription complete: {len(text)} characters")

        return text

    async def text_to_video(
        self,
        prompt: str,
        model: str,
        negative_prompt: str | None = None,
        num_frames: int = 49,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        height: int = 480,
        width: int = 720,
        fps: int = 16,
        seed: int | None = None,
        max_sequence_length: int = 512,
        enable_cpu_offload: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
        **kwargs: Any,
    ) -> VideoRef:
        """Generate a video from a text prompt using HuggingFace text-to-video models.

        Supports Wan text-to-video models (Wan-AI/Wan2.2-T2V-A14B-Diffusers, etc.).

        Args:
            prompt: Text description of the desired video
            model: Model repository ID (e.g., "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            negative_prompt: Text describing what to avoid in the video
            num_frames: Number of frames to generate (16-129)
            guidance_scale: Scale for classifier-free guidance (1.0-20.0)
            num_inference_steps: Number of denoising steps (1-100)
            height: Height of the generated video in pixels
            width: Width of the generated video in pixels
            fps: Frames per second for the output video
            seed: Random seed for generation (None for random)
            max_sequence_length: Maximum sequence length in encoded prompt
            enable_cpu_offload: Enable CPU offload to reduce VRAM usage
            enable_vae_slicing: Enable VAE slicing to reduce VRAM usage
            enable_vae_tiling: Enable VAE tiling for large videos
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            **kwargs: Additional arguments

        Returns:
            VideoRef to the generated video

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        from nodetool.nodes.huggingface.text_to_video import (
            AutoencoderKLWan,
            WanPipeline,
        )
        from nodetool.nodes.huggingface.text_to_video import pipeline_progress_callback

        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace text-to-video generation"
            )

        # Get or load the pipeline
        pipeline = ModelManager.get_model(model)

        if not pipeline:
            import torch

            log.info(f"Loading text-to-video pipeline: {model}")

            # Load VAE first
            vae = await asyncio.to_thread(
                AutoencoderKLWan.from_pretrained,
                model,
                subfolder="vae",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )

            # Load WanPipeline
            pipeline = await asyncio.to_thread(
                WanPipeline.from_pretrained,
                model,
                torch_dtype=torch.bfloat16,
                vae=vae,
            )

            # Apply memory optimization settings
            if enable_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
                if hasattr(pipeline, "enable_sequential_cpu_offload"):
                    pipeline.enable_sequential_cpu_offload()

            if hasattr(pipeline, "enable_attention_slicing"):
                try:
                    pipeline.enable_attention_slicing()
                except Exception:
                    pass

            if enable_vae_slicing and hasattr(pipeline, "vae"):
                try:
                    pipeline.vae.enable_slicing()
                except Exception:
                    pass

            if enable_vae_tiling and hasattr(pipeline, "vae"):
                try:
                    pipeline.vae.enable_tiling()
                except Exception:
                    pass

            try:
                if hasattr(pipeline, "unet") and hasattr(
                    pipeline.unet, "enable_xformers_memory_efficient_attention"
                ):
                    pipeline.unet.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            # Cache the pipeline
            ModelManager.set_model(node_id, model, pipeline)

        # Set up generator for reproducibility
        import torch

        generator = None
        if seed is not None and seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # Generate the video off the event loop
        def _run_pipeline_sync():
            return pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                max_sequence_length=max_sequence_length,
                callback_on_step_end=pipeline_progress_callback(
                    node_id=node_id, total_steps=num_inference_steps, context=context
                ),  # type: ignore
            )

        output = await asyncio.to_thread(_run_pipeline_sync)

        # Get the generated frames
        frames = output.frames[0]  # pyright: ignore[reportAttributeAccessIssue]

        # Convert frames to video
        video_ref = await context.video_from_frames(frames, fps=fps)  # type: ignore

        return video_ref

    async def get_available_language_models(self) -> List[LanguageModel]:
        """Get available HuggingFace language models.

        Returns models available in the local HuggingFace cache.

        Returns:
            List of LanguageModel instances for HuggingFace models
        """
        models = await get_hf_language_models_from_hf_cache()
        log.debug(f"Found {len(models)} HuggingFace models in HF cache")
        return models

    async def get_available_image_models(self) -> List[ImageModel]:
        """Get available HuggingFace image models.

        Returns both multi-file models and single-file models (.safetensors).
        Single-file models use format "repo_id:path".
        """
        # Get multi-file models
        text_to_image_models = await get_text_to_image_models_from_hf_cache()
        image_to_image_models = await get_image_to_image_models_from_hf_cache()

        # Get single-file models
        return text_to_image_models + image_to_image_models

    async def get_available_tts_models(self) -> List[TTSModel]:
        """Get available HuggingFace TTS models from recommended models.

        Returns TTS models based on the recommended models from the TTS nodes:
        - Bark models (general TTS)
        - KokoroTTS models (multi-language with voices)
        - Generic TextToSpeech models (MMS models for various languages)

        Returns:
            List of TTSModel instances for HuggingFace TTS
        """
        models: List[TTSModel] = []

        # KokoroTTS - 54 voices
        kokoro_voices = [
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_heart",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
            "bm_daniel",
            "bm_fable",
            "bm_george",
            "bm_lewis",
            "ef_dora",
            "em_alex",
            "em_santa",
            "ff_siwis",
            "hf_alpha",
            "hf_beta",
            "hm_omega",
            "hm_psi",
            "if_sara",
            "im_nicola",
            "jf_alpha",
            "jf_gongitsune",
            "jf_nezumi",
            "jf_tebukuro",
            "jm_kumo",
            "pf_dora",
            "pm_alex",
            "pm_santa",
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
        ]

        kokoro_model = TTSModel(
            id="hexgrad/Kokoro-82M",
            name="Kokoro TTS 82M",
            provider=Provider.HuggingFace,
            voices=kokoro_voices,
        )
        models.append(kokoro_model)

        log.debug(f"Returning {len(models)} HuggingFace TTS models")
        return models

    async def get_available_asr_models(self) -> List["ASRModel"]:
        """Get available HuggingFace ASR models from recommended models.

        Returns ASR models based on the recommended models from the Whisper node:
        - OpenAI Whisper models (large-v3, large-v3-turbo, large-v2, medium, small)
        - Faster-whisper models (optimized for speed)

        Returns:
            List of ASRModel instances for HuggingFace ASR
        """

        models = [
            ASRModel(
                id="openai/whisper-large-v3",
                name="Whisper Large V3",
                provider=Provider.HuggingFace,
            ),
            ASRModel(
                id="openai/whisper-large-v3-turbo",
                name="Whisper Large V3 Turbo",
                provider=Provider.HuggingFace,
            ),
            ASRModel(
                id="openai/whisper-large-v2",
                name="Whisper Large V2",
                provider=Provider.HuggingFace,
            ),
            ASRModel(
                id="openai/whisper-medium",
                name="Whisper Medium",
                provider=Provider.HuggingFace,
            ),
            ASRModel(
                id="openai/whisper-small",
                name="Whisper Small",
                provider=Provider.HuggingFace,
            ),
            ASRModel(
                id="Systran/faster-whisper-large-v3",
                name="Faster Whisper Large V3",
                provider=Provider.HuggingFace,
            ),
        ]

        log.debug(f"Returning {len(models)} HuggingFace ASR models")
        return models

    @staticmethod
    def _parse_model_spec(model: str) -> tuple[str, str | None]:
        """Return repo_id, optional filename from model spec."""
        if ":" not in model:
            return model, None
        parts = model.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid model spec: {model}")
        repo_id, filename = parts
        return repo_id, filename

    def _load_image_data(self, image_ref) -> bytes:
        """Load image data from an ImageRef."""
        if hasattr(image_ref, "data") and image_ref.data is not None:
            return image_ref.data

        uri = getattr(image_ref, "uri", "") if hasattr(image_ref, "uri") else ""
        if not uri:
            raise ValueError("ImageRef has no data or URI")

        _mime, data = fetch_uri_bytes_and_mime_sync(uri)
        return data

    def convert_message(self, message: Message) -> Dict[str, Any]:
        """
        Convert an internal message to HF dict format.
        Preserves PIL images in content list for further processing.
        """
        if message.role == "tool":
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            elif isinstance(message.content, dict):
                content = json.dumps(message.content)
            elif isinstance(message.content, list):
                content = json.dumps([part.model_dump() for part in message.content])
            elif isinstance(message.content, str):
                content = message.content
            else:
                content = json.dumps(message.content)

            return {"role": "tool", "content": content, "name": message.name}

        elif message.role == "system":
            if message.content is None:
                content = ""
            elif isinstance(message.content, str):
                content = message.content
            else:
                text_parts = [
                    part.text
                    for part in message.content
                    if isinstance(part, MessageTextContent)
                ]
                content = "\n".join(text_parts)
            return {"role": "system", "content": content}

        elif message.role == "user":
            if isinstance(message.content, str):
                return {"role": "user", "content": message.content}

            # Handle list content
            content_list = []
            has_images = False
            for part in message.content:
                if isinstance(part, MessageTextContent):
                    content_list.append({"type": "text", "text": part.text})
                elif isinstance(part, MessageImageContent):
                    # Load image to PIL
                    data = self._load_image_data(part.image)
                    img = Image.open(BytesIO(data))
                    # Store PIL image directly; will be extracted later
                    content_list.append({"type": "image", "image": img})
                    has_images = True

            # For text-only models, content must be a string, not a list
            # Only use list format if there are images (for VLM models)
            if has_images:
                return {"role": "user", "content": content_list}
            else:
                # Extract text parts and join them
                text_parts = [
                    part["text"] for part in content_list if part.get("type") == "text"
                ]
                return {"role": "user", "content": "\n".join(text_parts)}

        elif message.role == "assistant":
            # For assistant, we mainly handle text content and tool calls if we supported them
            content = ""
            if message.content is None:
                content = ""
            elif isinstance(message.content, str):
                content = message.content
            else:
                text_parts = [
                    part.text
                    for part in message.content
                    if isinstance(part, MessageTextContent)
                ]
                content = "\n".join(text_parts)

            msg_dict = {"role": "assistant", "content": content}

            # TODO: Handle tool calls formatting if HF supports it in standard templates
            # For now, we omit complex tool_call structures as apply_chat_template conventions vary

            return msg_dict

        else:
            # Fallback
            content = str(message.content) if message.content else ""
            return {"role": message.role, "content": content}

        # Helper to convert messages to prompt is no longer needed as we use apply_chat_template
        return ""

    async def _stream_pipeline_generation(
        self,
        repo_id: str,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        context: ProcessingContext,
        node_id: str | None,
        quantization: str = "fp16",
    ) -> AsyncIterator[Chunk]:
        import torch
        from transformers import BitsAndBytesConfig, TextStreamer

        cached_pipeline = ModelManager.get_model(repo_id)

        if not cached_pipeline:
            log.info(f"Loading HuggingFace pipeline model {repo_id}")
            load_kwargs = {}
            if quantization == "nf4":
                load_kwargs["model_kwargs"] = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                }
            elif quantization == "nf8":
                load_kwargs["model_kwargs"] = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_quant_type="nf8",
                        bnb_8bit_use_double_quant=True,
                        bnb_8bit_compute_dtype=torch.bfloat16,
                    )
                }
            cached_pipeline = await load_pipeline(
                node_id=node_id,
                context=context,
                pipeline_task="text-generation",
                model_id=repo_id,
                **load_kwargs,
            )
            ModelManager.set_model(node_id, repo_id, cached_pipeline)

        tokenizer = cached_pipeline.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer missing from HuggingFace pipeline")

        # Apply chat template
        # Ensure messages are in the format expected by HF (list of dicts)
        hf_messages = [self.convert_message(msg) for msg in messages]

        prompt = tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True
        )

        token_queue: Queue = Queue()

        class AsyncTextStreamer(TextStreamer):
            def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                self.token_queue = token_queue

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("TextStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]

                if self.skip_prompt and self.next_tokens_are_prompt:
                    self.next_tokens_are_prompt = False
                    return

                text = self.tokenizer.decode(value, skip_special_tokens=True)  # pyright: ignore[reportAttributeAccessIssue]
                if text:
                    self.token_queue.put(text)

            def end(self):
                self.token_queue.put(None)

        streamer = AsyncTextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def generate():
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "streamer": streamer,
                "return_full_text": False,
            }
            cached_pipeline(prompt, **generation_kwargs)  # type: ignore[reportArgumentType]

        thread = threading.Thread(target=generate)
        thread.start()

        try:
            while True:
                await asyncio.sleep(0.01)
                while not token_queue.empty():
                    token = token_queue.get_nowait()
                    if token is None:
                        return
                    yield Chunk(content=token, done=False, content_type="text")
                if not thread.is_alive():
                    while not token_queue.empty():
                        token = token_queue.get_nowait()
                        if token is None:
                            return
                        yield Chunk(content=token, done=False, content_type="text")
                    break
        finally:
            thread.join(timeout=1.0)

    @staticmethod
    def _extract_text_from_output(outputs: Any) -> str:
        """Normalize output across pipeline variants."""
        if isinstance(outputs, list) and len(outputs) > 0:
            first = outputs[0]
            if isinstance(first, dict):
                for key in [
                    "generated_text",
                    "answer",
                    "text",
                    "output_text",
                ]:
                    if key in first and isinstance(first[key], str):
                        return first[key]  # type: ignore
            if isinstance(first, str):
                return first
        return str(outputs)

    async def _stream_image_text_to_text(
        self,
        repo_id: str,
        messages: Sequence[Message],
        max_tokens: int,
        context: ProcessingContext,
        node_id: str | None,
        quantization: str = "fp16",
    ) -> AsyncIterator[Chunk]:
        """Stream generation for image-text-to-text models."""
        import torch
        from transformers import (
            BitsAndBytesConfig,
            AutoProcessor,
            AutoModelForCausalLM,
            TextStreamer,
        )

        # Extract images and clean messages
        # Convert messages and extract images
        cleaned_messages = []
        pil_images = []

        for msg in messages:
            # Use convert_message to standardize
            converted = self.convert_message(msg)

            # Post-process for VLM: extract PIL images from content list
            if isinstance(converted.get("content"), list):
                new_content = []
                for item in converted["content"]:
                    if item.get("type") == "image" and "image" in item:
                        # Extract PIL image
                        pil_images.append(item["image"])
                        # Keep placeholder
                        new_content.append({"type": "image"})
                    else:
                        new_content.append(item)
                converted["content"] = new_content

            cleaned_messages.append(converted)

        # Load processor
        processor = await load_model(
            node_id=node_id,
            context=context,
            model_class=AutoProcessor,
            model_id=repo_id,
        )

        load_kwargs = {"device_map": "auto"}
        if quantization == "nf4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "nf8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )

        # Load model using AutoModelForCausalLM as requested for VL models
        model = await load_model(
            node_id=node_id,
            context=context,
            model_class=AutoModelForCausalLM,  # User guide suggests this for Qwen2.5-VL/LLaVA
            model_id=repo_id,
            **load_kwargs,
        )

        # Prepare inputs
        def _prepare_inputs():
            return processor.apply_chat_template(
                cleaned_messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                images=pil_images if pil_images else None,
            ).to(model.device)

        inputs = await asyncio.to_thread(_prepare_inputs)

        # Output streamer
        token_queue: Queue = Queue()

        # We need a streamer that puts into our queue
        # Reusing the class defined inside _stream_pipeline_generation is not possible cleanly unless we move it out or duplicate.
        # Duplicating for now since the other method has it inline.

        class AsyncTextStreamer(TextStreamer):
            def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                self.token_queue = token_queue

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("TextStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]

                if self.skip_prompt and self.next_tokens_are_prompt:
                    self.next_tokens_are_prompt = False
                    return

                text = self.tokenizer.decode(value, skip_special_tokens=True)
                if text:
                    self.token_queue.put(text)

            def end(self):
                self.token_queue.put(None)

        streamer = AsyncTextStreamer(
            processor.tokenizer,  # Processor should have tokenizer
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def generate():
            model.generate(
                **(
                    inputs if isinstance(inputs, dict) else {"input_ids": inputs}
                ),  # apply_chat_template returns tensor or dict
                max_new_tokens=max_tokens,
                streamer=streamer,
            )

        thread = threading.Thread(target=generate)
        thread.start()

        try:
            while True:
                await asyncio.sleep(0.01)
                while not token_queue.empty():
                    token = token_queue.get_nowait()
                    if token is None:
                        return
                    yield Chunk(content=token, done=False, content_type="text")
                if not thread.is_alive():
                    while not token_queue.empty():
                        token = token_queue.get_nowait()
                        if token is None:
                            return
                        yield Chunk(content=token, done=False, content_type="text")
                    break
        finally:
            thread.join(timeout=1.0)

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> Message:
        """Generate a single message completion using HuggingFace GGUF models.

        Args:
            messages: Conversation history to send
            model: Model spec in format "repo_id:filename.gguf" (e.g., "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF:qwen2.5-coder-0.5b-q8_0.gguf")
            tools: Optional tool definitions (not supported for HF GGUF models)
            max_tokens: Maximum tokens to generate
            context_window: Maximum tokens to consider for context
            response_format: Optional response schema (not supported for HF GGUF models)
            **kwargs: Additional arguments (temperature, top_p, do_sample, context)

        Returns:
            A Message object containing the model's response

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        if not messages:
            raise ValueError("messages must not be empty")

        context = kwargs.pop("context", None)
        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace text generation"
            )

        full_text = ""
        async for chunk in self.generate_messages(
            messages=messages,
            model=model,
            tools=tools,
            max_tokens=max_tokens,
            context_window=context_window,
            response_format=response_format,
            context=context,
            **kwargs,
        ):
            if chunk.content:
                full_text += chunk.content

        return Message(
            role="assistant",
            content=full_text,
            provider=Provider.HuggingFace,
            model=model,
        )

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        quantization: str = "fp16",
        **kwargs: Any,
    ) -> AsyncIterator[Chunk]:
        """Stream message completions using HuggingFace GGUF models.

        Args:
            messages: Conversation history to send
            model: Model spec in format "repo_id:filename.gguf" (e.g., "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF:qwen2.5-coder-0.5b-q8_0.gguf")
            tools: Optional tool definitions (not supported for HF GGUF models)
            max_tokens: Maximum tokens to generate
            context_window: Maximum tokens to consider for context
            response_format: Optional response schema (not supported for HF GGUF models)
            quantization: Quantization method to use (nf4 for 4-bit, nf8 for 8-bit, fp16 for default)
            **kwargs: Additional arguments (temperature, top_p, do_sample, context)

        Yields:
            Chunk objects containing text deltas

        Raises:
            ValueError: If required parameters are missing or context not provided
            RuntimeError: If generation fails
        """
        if not messages:
            raise ValueError("messages must not be empty")

        context = kwargs.get("context")
        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace text generation"
            )

        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 1.0)
        do_sample = kwargs.get("do_sample", True)
        node_id = kwargs.get("node_id")

        pipeline_task = kwargs.get("pipeline_task")
        if pipeline_task == "image-text-to-text":
            repo_id, _ = self._parse_model_spec(model)
            async for chunk in self._stream_image_text_to_text(
                repo_id=repo_id,
                messages=messages,
                max_tokens=max_tokens,
                context=context,
                node_id=node_id,
                quantization=quantization,
            ):
                yield chunk
            return

        repo_id, filename = self._parse_model_spec(model)

        async for chunk in self._stream_pipeline_generation(
            repo_id=repo_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            context=context,
            node_id=node_id,
            quantization=quantization,
        ):
            yield chunk


if __name__ == "__main__":
    import asyncio

    # Create provider instance
    provider = HuggingFaceLocalProvider()

    async def test_generate_messages():
        """Test the generate_messages method with streaming."""
        from nodetool.workflows.processing_context import ProcessingContext
        from nodetool.config.environment import Environment

        # Initialize environment
        env = Environment.get_environment()

        # Create a simple processing context (you may need to adjust this based on your setup)
        context = ProcessingContext()
        context.device = "mps"

        # Test messages
        messages = [
            Message(
                role="system",
                content="You are a helpful assistant that provides concise answers.",
            ),
            Message(
                role="user",
                content="What is the capital of France? Answer in one sentence.",
            ),
        ]

        # Model to test - using a small model for quick testing
        # Change this to any model you have cached locally
        models = await provider.get_available_language_models()
        model = models[0]

        # Stream the response
        full_response = ""
        async for chunk in provider.generate_messages(
            messages=messages,
            model=model.id,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            context=context,
        ):
            print(chunk)

    async def test_generate_message():
        """Test the generate_message method (non-streaming)."""
        from nodetool.workflows.processing_context import ProcessingContext

        # Create a simple processing context
        context = ProcessingContext()
        context.device = "mps"

        # Test messages
        messages = [
            Message(
                role="system",
                content="You are a helpful assistant that provides concise answers.",
            ),
            Message(
                role="user",
                content="What is 2+2? Answer in one sentence.",
            ),
        ]

        # Model to test
        models = await provider.get_available_language_models()
        model = models[0]

        # Get the response
        response = await provider.generate_message(
            messages=messages,
            model=model.id,
            max_tokens=50,
            temperature=0.7,
            context=context,
        )

    async def test_text_to_image():
        """Test the text_to_image method."""
        models = await provider.get_available_image_models()
        print(models)
        model = models[0]
        context = ProcessingContext()
        context.device = "mps"
        image = await provider.text_to_image(
            params=TextToImageParams(
                prompt="A beautiful sunset over a calm ocean",
                model=model,
                num_inference_steps=20,
            ),
            context=context,
        )
        open("image.png", "wb").write(image)

    async def test_image_to_image():
        """Test the image_to_image method."""
        provider = HuggingFaceLocalProvider()
        models = await provider.get_available_image_models()
        print(models)
        model = models[0]
        context = ProcessingContext()
        context.device = "mps"
        image_bytes = open("image.png", "rb").read()
        image = await provider.image_to_image(
            image=image_bytes,
            params=ImageToImageParams(
                prompt="a photo of an astronaut riding a horse",
                model=model,
                strength=0.8,
                num_inference_steps=20,
                guidance_scale=7.5,
            ),
            context=context,
        )
        open("image_to_image.png", "wb").write(image)

    async def test_available_language_models():
        """Test the available_language_models method."""
        provider = HuggingFaceLocalProvider()
        models = await provider.get_available_language_models()
        print(models)

    async def test_available_image_models():
        """Test the available_image_models method."""
        provider = HuggingFaceLocalProvider()
        models = await provider.get_available_image_models()
        print(models)

    async def test_available_tts_models():
        """Test the available_tts_models method."""
        provider = HuggingFaceLocalProvider()
        models = await provider.get_available_tts_models()
        print(models)

    async def test_available_asr_models():
        """Test the available_asr_models method."""
        provider = HuggingFaceLocalProvider()
        models = await provider.get_available_asr_models()
        print(models)

    # Run tests
    print("=" * 50)
    print("Testing HuggingFace Local Provider")
    print("=" * 50)

    asyncio.run(test_available_image_models())

    # # Test streaming
    # asyncio.run(test_generate_messages())

    # # Test non-streaming
    # asyncio.run(test_generate_message())
    # asyncio.run(test_text_to_image())
    # asyncio.run(test_image_to_image())
