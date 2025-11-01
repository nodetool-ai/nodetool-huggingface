"""
HuggingFace local provider implementation.

This module implements the BaseProvider interface for locally cached HuggingFace models.
- Language models: Only GGUF models are supported, using transformers for inference
  with model IDs in format "repo_id:filename.gguf" (same as LlamaCpp provider)
- Image models: Text2Image and ImageToImage using diffusion pipelines
  Supports both multi-file models (repo_id) and single-file models (repo_id:path.safetensors)
- TTS models: KokoroTTS and other HuggingFace TTS models
"""

import asyncio
import base64
from queue import Queue
import threading
from typing import Any, AsyncGenerator, List, Set
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from huggingface_hub import scan_cache_dir
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.integrations.huggingface.huggingface_models import (
    fetch_model_info,
    has_model_index,
    model_type_from_model_info,
)
import numpy as np
from pydub import AudioSegment
import io
from io import BytesIO
from nodetool.types.model import UnifiedModel
from nodetool.workflows.processing_context import ProcessingContext
from PIL import Image
from nodetool.metadata.types import (
    ImageModel,
    Provider,
    TTSModel,
    Message,
)
from nodetool.workflows.types import Chunk, NodeProgress
from nodetool.metadata.types import ASRModel
from typing import List, Sequence, Any, AsyncIterator
from nodetool.config.logging_config import get_logger
import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from nodetool.ml.core.model_manager import ModelManager
from huggingface_hub import try_to_load_from_cache
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

# Import specific pipeline classes
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import StableDiffusion3Img2ImgPipeline
from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    TextStreamer,
    pipeline as create_pipeline,
)
from nodetool.nodes.huggingface.text_to_speech import KokoroTTS
from nodetool.metadata.types import HFTextToSpeech
from pydub import AudioSegment
from io import BytesIO
from nodetool.nodes.huggingface.text_to_speech import TextToSpeech
from nodetool.metadata.types import HFTextToSpeech

from nodetool.workflows.recommended_models import get_recommended_models
from nodetool.metadata.types import LanguageModel, VideoRef
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import WanPipeline

log = get_logger(__name__)


def get_model_type_from_id(model_id: str) -> str | None:
    """
    Determine the model type from a model ID by looking it up in recommended models.

    Args:
        model_id: Model ID in format "repo_id" or "repo_id:path"

    Returns:
        Model type string (e.g., "hf.stable_diffusion") or None if not found
    """
    recommended_models = get_recommended_models()

    # Flatten all recommended models
    all_models = []
    for models_list in recommended_models.values():
        all_models.extend(models_list)

    # Look for model by ID
    for model in all_models:
        if model.id == model_id:
            return model.type

    return None


async def get_hf_cached_single_file_image_models() -> List[ImageModel]:
    """
    Get single-file image models from HF cache based on recommended models.

    Returns models in format "repo_id:path" for single-file models (.safetensors)
    that are cached locally.

    Returns:
        List of ImageModel instances for cached single-file models
    """
    # Image model types we want to include (SD, SDXL, SD3, Flux, QwenImage)
    IMAGE_MODEL_TYPES = {
        "hf.stable_diffusion",
        "hf.stable_diffusion_xl",
        "hf.stable_diffusion_3",
        "hf.flux",
        "hf.qwen_image",
    }

    models: list[ImageModel] = []
    seen: set[str] = set()

    def flatten(models: list[list[UnifiedModel]]) -> list[UnifiedModel]:
        return [model for sublist in models for model in sublist]

    # Get all recommended models and filter by image model types
    recommended_models: list[UnifiedModel] = flatten(list(get_recommended_models().values()))
    image_models = [
        model for model in recommended_models
        if model.type in IMAGE_MODEL_TYPES
    ]

    for model_config in image_models:
        # Check if the file is cached locally
        split_path = model_config.id.split(':')
        if len(split_path) != 2:
            continue
        repo_id = split_path[0]
        path = split_path[1]
        cache_path = try_to_load_from_cache(repo_id, path)
        if cache_path:
            # Create model ID in format "repo_id:path"
            model_id = f"{repo_id}:{path}"

            if model_id in seen:
                continue
            seen.add(model_id)

            models.append(
                ImageModel(
                    id=model_id,
                    name=model_config.name,
                    provider=Provider.HuggingFace,
                )
            )
            log.debug(f"Found cached single-file model: {model_id}")

    log.debug(f"Found {len(models)} cached single-file image models")
    return models


async def get_hf_cached_image_models() -> List[UnifiedModel]:
    """
    Scan the Hugging Face cache directory and return models that are compatible
    with image generation architectures: SD1.5, SDXL, SD3, Flux, QwenImage, and Chroma.

    Returns:
        List[UnifiedModel]: List of cached image models compatible with the supported architectures
    """
    # Model types we want to include (image generation models)
    COMPATIBLE_MODEL_TYPES = {
        "hf.stable_diffusion",  # SD1.5
        "hf.stable_diffusion_xl",  # SDXL
        "hf.stable_diffusion_3",  # SD3
        "hf.flux",  # Flux
        "hf.qwen_image",  # QwenImage
    }

    # Tags that indicate image generation models we want to include
    COMPATIBLE_TAGS = {
        "stable-diffusion",
        "stable-diffusion-xl",
        "stable-diffusion-3",
        "flux",
        "qwen",
        "chroma",
        "text-to-image",
        "diffusers",
    }

    # Scan HF cache directory
    cache_info = await asyncio.to_thread(scan_cache_dir)

    model_repos = [repo for repo in cache_info.repos if repo.repo_type == "model"]
    recommended_models = get_recommended_models()

    # Fetch model info for all cached repos
    model_infos = await asyncio.gather(
        *[fetch_model_info(repo.repo_id) for repo in model_repos]
    )

    models: list[UnifiedModel] = []
    for repo, model_info in zip(model_repos, model_infos):
        # Skip if we couldn't fetch model info
        if model_info is None:
            continue

        # Determine model type
        model_type = model_type_from_model_info(
            recommended_models, repo.repo_id, model_info
        )

        # Check if this is a compatible image model
        is_compatible = False

        # Check by model type
        if model_type:
            if model_type in COMPATIBLE_MODEL_TYPES:
                is_compatible = True
            else:
                continue

        # Check by tags
        if not is_compatible and model_info.tags:
            model_tags_lower = [tag.lower() for tag in model_info.tags]
            if any(
                compatible_tag in tag
                for tag in model_tags_lower
                for compatible_tag in COMPATIBLE_TAGS
            ):
                is_compatible = True

        # Check by pipeline tag
        if not is_compatible and model_info.pipeline_tag:
            if model_info.pipeline_tag in ["text-to-image", "image-to-image"]:
                is_compatible = True

        # Skip if not compatible
        if not is_compatible:
            continue

        # Use repo name as display name (last part of repo_id)
        display_name = (
            repo.repo_id.split("/")[-1] if "/" in repo.repo_id else repo.repo_id
        )

        models.append(
            UnifiedModel(
                id=repo.repo_id,
                type=model_type,
                name=display_name,
                cache_path=str(repo.repo_path),
                allow_patterns=None,
                ignore_patterns=None,
                description=None,
                readme=None,
                downloaded=repo.repo_path is not None,
                pipeline_tag=model_info.pipeline_tag,
                tags=model_info.tags,
                has_model_index=has_model_index(model_info),
                repo_id=repo.repo_id,
                path=None,
                size_on_disk=repo.size_on_disk,
                downloads=model_info.downloads,
                likes=model_info.likes,
                trending_score=model_info.trending_score,
            )
        )

    log.info(f"Found {len(models)} cached image models")
    return models

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
    ) -> ImageBytes:
        """Generate an image from a text prompt using HuggingFace diffusion models.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling

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

        # Get or load the pipeline
        model_id = params.model.id
        cache_key = f"{model_id}:text2image"
        pipeline = ModelManager.get_model(cache_key, "text2image")

        if not pipeline:
            log.info(f"Loading text-to-image pipeline: {model_id}")

            # Determine model type
            model_type = get_model_type_from_id(model_id)

            # Check if model_id is in "repo_id:path" format (single-file model)
            if ":" in model_id and model_id.count(":") == 1:
                repo_id, file_path = model_id.split(":", 1)

                # Verify the file is cached locally
                cache_path = try_to_load_from_cache(repo_id, file_path)
                if not cache_path:
                    raise ValueError(f"Single-file model {repo_id}/{file_path} must be downloaded first")

                # Load pipeline from single file based on model type
                if model_type == "hf.stable_diffusion":
                    pipeline = StableDiffusionPipeline.from_single_file(
                        str(cache_path),
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    )
                elif model_type == "hf.stable_diffusion_xl":
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        str(cache_path),
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    )
                elif model_type == "hf.stable_diffusion_3":
                    pipeline = StableDiffusion3Pipeline.from_single_file(
                        str(cache_path),
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    )
                elif model_type == "hf.flux":
                    pipeline = FluxPipeline.from_single_file(
                        str(cache_path),
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    )
                else:
                    raise ValueError(f"Unsupported single-file model type: {model_type}")
            else:
                # Load pipeline from multi-file model (standard format)
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                )

            pipeline.to(context.device)

            # Cache the pipeline
            ModelManager.set_model(cache_key, cache_key, "text2image", pipeline)

        # Set up generator for reproducibility
        generator = None
        if params.seed is not None and params.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(params.seed)

        # Progress callback
        num_steps = params.num_inference_steps or 50

        def progress_callback(pipe, step: int, timestep: int, callback_kwargs: dict):
            if context:
                context.post_message(
                    NodeProgress(
                        node_id="text_to_image",
                        progress=step,
                        total=num_steps,
                    )
                )
            return callback_kwargs

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
                callback_on_step_end=progress_callback,  # type: ignore
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
    ) -> ImageBytes:
        """Transform an image based on a text prompt using HuggingFace diffusion models.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling

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

        # Get or load the pipeline
        model_id = params.model.id
        cache_key = f"{model_id}:image2image"
        pipeline = ModelManager.get_model(cache_key, "image2image")

        if not pipeline:
            log.info(f"Loading image-to-image pipeline: {model_id}")

            # Determine model type
            model_type = get_model_type_from_id(model_id)

            # Check if model_id is in "repo_id:path" format (single-file model)
            if ":" in model_id and model_id.count(":") == 1:
                repo_id, file_path = model_id.split(":", 1)

                # Verify the file is cached locally
                cache_path = try_to_load_from_cache(repo_id, file_path)
                if not cache_path:
                    raise ValueError(f"Single-file model {repo_id}/{file_path} must be downloaded first")
                # Load pipeline from single file based on model type
                if model_type == "hf.stable_diffusion":
                    pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                        str(cache_path),
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    )
                elif model_type == "hf.stable_diffusion_xl":
                    pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
                        str(cache_path),
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    )
                elif model_type == "hf.stable_diffusion_3":
                    pipeline = StableDiffusion3Img2ImgPipeline.from_single_file(
                        str(cache_path),
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    )
                elif model_type == "hf.flux":
                    pipeline = FluxImg2ImgPipeline.from_single_file(
                        str(cache_path),
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    )
                else:
                    raise ValueError(f"Unsupported single-file model type: {model_type}")
            else:
                # Load pipeline from multi-file model (standard format)
                pipeline = AutoPipelineForImage2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                )

            assert pipeline is not None
            pipeline.to(context.device)

            # Cache the pipeline
            ModelManager.set_model(cache_key, cache_key, "image2image", pipeline)

        # Set up generator for reproducibility
        generator = None
        if params.seed is not None and params.seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(params.seed)

        # Progress callback
        num_steps = params.num_inference_steps or 25

        def progress_callback(pipe, step: int, timestep: int, callback_kwargs: dict):
            if context:
                context.post_message(
                    NodeProgress(
                        node_id="image_to_image",
                        progress=step,
                        total=num_steps,
                    )
                )
            return callback_kwargs

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
                callback_on_step_end=progress_callback,  # type: ignore
                callback_on_step_end_tensor_inputs=["latents"],
            )

        output = await asyncio.to_thread(_run_pipeline_sync)

        # Get the generated image
        pil_output = output.images[0]  # pyright: ignore[reportAttributeAccessIssue]

        # Convert PIL Image to bytes
        img_buffer = BytesIO()
        pil_output.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        self.usage["total_requests"] += 1
        self.usage["total_images"] += 1

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
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

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
            raise ValueError(
                "ProcessingContext is required for HuggingFace ASR"
            )

        log.debug(f"Transcribing audio with HuggingFace Whisper model: {model}")

        # Get or load the pipeline
        cache_key = f"{model}:asr"
        asr_pipeline = ModelManager.get_model(cache_key, "automatic-speech-recognition")

        if not asr_pipeline:
            log.info(f"Loading automatic speech recognition pipeline: {model}")

            # Determine torch dtype based on device
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Load model
            hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model,
                torch_dtype=torch_dtype,
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
            ModelManager.set_model(cache_key, cache_key, "automatic-speech-recognition", asr_pipeline)

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
        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace text-to-video generation"
            )

        # Get or load the pipeline
        model_id = model
        cache_key = f"{model_id}:text2video"
        pipeline = ModelManager.get_model(cache_key, "text2video")

        if not pipeline:
            log.info(f"Loading text-to-video pipeline: {model_id}")

            # Load VAE first
            vae = await asyncio.to_thread(
                AutoencoderKLWan.from_pretrained,
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )

            # Load WanPipeline
            pipeline = await asyncio.to_thread(
                WanPipeline.from_pretrained,
                model_id,
                torch_dtype=torch.bfloat16,
                vae=vae,
            )

            # Apply memory optimization settings
            if enable_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()

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

            # Cache the pipeline
            ModelManager.set_model(cache_key, cache_key, "text2video", pipeline)

        # Set up generator for reproducibility
        generator = None
        if seed is not None and seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # Progress callback
        def progress_callback(pipe, step: int, timestep: int, callback_kwargs: dict):
            if context:
                context.post_message(
                    NodeProgress(
                        node_id="text_to_video",
                        progress=step,
                        total=num_inference_steps,
                    )
                )
            return callback_kwargs

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
                callback_on_step_end=progress_callback,
            )

        output = await asyncio.to_thread(_run_pipeline_sync)

        # Get the generated frames
        frames = output.frames[0]  # pyright: ignore[reportAttributeAccessIssue]

        # Convert frames to video
        video_ref = await context.video_from_frames(frames, fps=fps)

        return video_ref

    async def get_available_language_models(self) -> List[LanguageModel]:
        """Get available HuggingFace GGUF language models.

        Returns GGUF models available in the local HuggingFace cache.
        Uses the same model ID syntax as LlamaCpp provider: "repo_id:filename.gguf"

        Returns:
            List of LanguageModel instances for HuggingFace GGUF models
        """
        # Import the function to get locally cached GGUF models
        from nodetool.integrations.huggingface.huggingface_models import (
            get_llamacpp_language_models_from_hf_cache,
        )

        models = await get_llamacpp_language_models_from_hf_cache()
        # Update provider to HuggingFace for these models
        hf_models = [
            LanguageModel(
                id=model.id,  # Already in "repo_id:filename" format
                name=model.name,
                provider=Provider.HuggingFace,
            )
            for model in models
        ]
        log.debug(f"Found {len(hf_models)} HuggingFace GGUF models in HF cache")
        return hf_models

    async def get_available_image_models(self) -> List[ImageModel]:
        """Get available HuggingFace image models.

        Returns both multi-file models and single-file models (.safetensors).
        Single-file models use format "repo_id:path".
        """
        # Get multi-file models
        unified_models = await get_hf_cached_image_models()
        image_models = [
            ImageModel(id=model.id, name=model.name, provider=Provider.HuggingFace)
            for model in unified_models
        ]

        # Get single-file models
        single_file_models = await get_hf_cached_single_file_image_models()
        image_models.extend(single_file_models)

        return image_models

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
    def _parse_model_spec(model: str) -> tuple[str, str | None, bool]:
        """Return repo_id, optional filename, and GGUF flag from model spec."""
        if ":" not in model:
            return model, None, False
        parts = model.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid model spec: {model}")
        repo_id, filename = parts
        is_gguf = filename.lower().endswith(".gguf")
        return repo_id, filename, is_gguf

    @staticmethod
    def _build_prompt_from_messages(messages: Sequence[Message]) -> str:
        """Convert simple text-only chat history into a prompt string."""
        system_parts: list[str] = []
        user_prompt: str | None = None

        for msg in messages:
            if isinstance(msg.content, str):
                content = msg.content.strip()
            elif msg.content is None:
                content = ""
            else:
                raise ValueError(
                    "HuggingFace local provider only supports text messages for local models"
                )

            if msg.role == "system" and content:
                system_parts.append(content)
            elif msg.role == "user" and content:
                user_prompt = content

        if user_prompt is None:
            raise ValueError(
                "HuggingFace text generation requires at least one user message with text content"
            )

        if system_parts:
            return "\n\n".join(system_parts + [user_prompt])
        return user_prompt

    async def _stream_gguf_generation(
        self,
        messages: Sequence[Message],
        repo_id: str,
        filename: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> AsyncIterator[Chunk]:
        if not filename:
            raise ValueError("GGUF model path is required for HuggingFace local models")

        chat: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                raise ValueError(
                    "HuggingFace GGUF models do not support multimodal content. "
                    "Please use text-only messages."
                )
            else:
                content = ""
            chat.append({"role": msg.role, "content": content})

        cache_key = f"{repo_id}:{filename}:text-generation"
        cached_pipeline = ModelManager.get_model(cache_key, "text-generation")

        if not cached_pipeline:
            cache_path = try_to_load_from_cache(repo_id, filename)  # pyright: ignore[reportArgumentType]
            if not cache_path:
                raise ValueError(f"Model {repo_id}/{filename} must be downloaded first")

            log.info(f"Loading GGUF model {repo_id}/{filename}")
            hf_model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=torch.float32,
                device_map="auto",
                gguf_file=filename,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                gguf_file=filename,
            )
            cached_pipeline = pipeline(
                "text-generation", model=hf_model, tokenizer=tokenizer
            )
            ModelManager.set_model(
                cache_key, cache_key, "text-generation", cached_pipeline
            )

        tokenizer = cached_pipeline.tokenizer
        assert tokenizer is not None
        formatted_prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
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
            cached_pipeline(formatted_prompt, **generation_kwargs)  # type: ignore[reportArgumentType]

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

    async def _stream_pipeline_generation(
        self,
        repo_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        context: ProcessingContext,
    ) -> AsyncIterator[Chunk]:
        _ = context
        cache_key = f"{repo_id}:text-generation"
        cached_pipeline = ModelManager.get_model(cache_key, "text-generation")

        if not cached_pipeline:
            log.info(f"Loading HuggingFace pipeline model {repo_id}")
            cached_pipeline = create_pipeline(
                "text-generation",
                model=repo_id,
            )
            ModelManager.set_model(cache_key, cache_key, "text-generation", cached_pipeline)

        tokenizer = cached_pipeline.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer missing from HuggingFace pipeline")

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

        return Message(role="assistant", content=full_text, provider=Provider.HuggingFace, model=model)

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
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

        repo_id, filename, is_gguf = self._parse_model_spec(model)

        if is_gguf:
            async for chunk in self._stream_gguf_generation(
                messages=messages,
                repo_id=repo_id,
                filename=filename,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            ):
                yield chunk
            return

        prompt = self._build_prompt_from_messages(messages)

        async for chunk in self._stream_pipeline_generation(
            repo_id=repo_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            context=context,
        ):
            yield chunk


if __name__ == "__main__":
    import asyncio

    async def test_generate_messages():
        """Test the generate_messages method with streaming."""
        from nodetool.workflows.processing_context import ProcessingContext
        from nodetool.config.environment import Environment

        # Initialize environment
        env = Environment.get_environment()

        # Create provider instance
        provider = HuggingFaceLocalProvider()

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

        # Create provider instance
        provider = HuggingFaceLocalProvider()

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
        models = await get_hf_cached_single_file_image_models()
        print(models)
        model = models[0]
        provider = HuggingFaceLocalProvider()
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
        models = await get_hf_cached_single_file_image_models()
        print(models)
        model = models[0]
        provider = HuggingFaceLocalProvider()
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

    # Run tests
    print("=" * 50)
    print("Testing HuggingFace Local Provider")
    print("=" * 50)

    # asyncio.run(test_available_language_models())

    # # Test streaming
    asyncio.run(test_generate_messages())

    # # Test non-streaming
    # asyncio.run(test_generate_message())
    # asyncio.run(test_text_to_image())
    # asyncio.run(test_image_to_image())
