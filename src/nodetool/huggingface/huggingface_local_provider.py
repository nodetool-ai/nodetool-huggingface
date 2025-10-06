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
from typing import Any, AsyncGenerator, List, Set
from huggingface_hub import CacheNotFound, scan_cache_dir
from nodetool.providers.base import BaseProvider, ProviderCapability, register_provider
from nodetool.providers.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.integrations.huggingface.huggingface_models import (
    fetch_model_info,
    has_model_index,
    model_type_from_model_info,
)
from nodetool.metadata.types import HFTextToImage, HFImageToImage, HFTextGeneration
from io import BytesIO
from nodetool.types.model import UnifiedModel
from nodetool.workflows.processing_context import ProcessingContext
from PIL import Image
from nodetool.metadata.types import (
    ImageModel,
    Provider,
    TTSModel,
    Message,
    MessageTextContent,
)
from nodetool.workflows.types import Chunk, NodeProgress
from typing import List, Sequence, Any, AsyncIterator
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

from nodetool.workflows.recommended_models import get_recommended_models
from nodetool.metadata.types import LanguageModel


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
    from huggingface_hub import try_to_load_from_cache

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

    try:
        # Scan HF cache directory
        cache_info = await asyncio.to_thread(scan_cache_dir)
    except CacheNotFound:
        log.debug(
            "Hugging Face cache directory not found; returning empty image model list"
        )
        return []

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

    def __init__(self):
        super().__init__()

    def get_capabilities(self) -> Set[ProviderCapability]:
        """HuggingFace provider supports text-to-image, image-to-image, and text-to-speech."""
        return {
            ProviderCapability.GENERATE_MESSAGE,
            ProviderCapability.GENERATE_MESSAGES,
            ProviderCapability.TEXT_TO_IMAGE,
            ProviderCapability.IMAGE_TO_IMAGE,
            ProviderCapability.TEXT_TO_SPEECH,
            ProviderCapability.AUTOMATIC_SPEECH_RECOGNITION,
        }

    def get_container_env(self) -> dict[str, str]:
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

        try:
            import torch
            from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
            from nodetool.ml.core.model_manager import ModelManager
            from huggingface_hub import try_to_load_from_cache

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

                    # Import specific pipeline classes
                    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
                    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
                    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
                    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

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
                    callback_on_step_end=progress_callback,
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

        except Exception as e:
            log.error(f"HuggingFace text-to-image generation failed: {e}")
            raise RuntimeError(f"HuggingFace text-to-image generation failed: {e}")

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

        try:
            import torch
            from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
            from nodetool.ml.core.model_manager import ModelManager
            from huggingface_hub import try_to_load_from_cache

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

                    # Import specific pipeline classes
                    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
                    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
                    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import StableDiffusion3Img2ImgPipeline
                    from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline

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
                    callback_on_step_end=progress_callback,
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

        except Exception as e:
            log.error(f"HuggingFace image-to-image generation failed: {e}")
            raise RuntimeError(f"HuggingFace image-to-image generation failed: {e}")

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

        try:
            import numpy as np

            # Determine which TTS node to use based on model ID
            model_lower = model.lower()

            if "kokoro" in model_lower:
                # Use KokoroTTS node - supports streaming and voices
                from nodetool.nodes.huggingface.text_to_speech import KokoroTTS
                from nodetool.metadata.types import HFTextToSpeech

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
                        import base64

                        audio_bytes = base64.b64decode(chunk.content)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        yield audio_array

            else:
                # Use generic TextToSpeech node (MMS models)
                from nodetool.nodes.huggingface.text_to_speech import TextToSpeech
                from nodetool.metadata.types import HFTextToSpeech

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
                from pydub import AudioSegment
                import io

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


        except Exception as e:
            log.error(f"HuggingFace TTS generation failed: {e}")
            raise RuntimeError(f"HuggingFace TTS generation failed: {e}")

    async def get_available_language_models(self) -> List[LanguageModel]:
        """Get available HuggingFace GGUF language models.

        Returns GGUF models available in the local HuggingFace cache.
        Uses the same model ID syntax as LlamaCpp provider: "repo_id:filename.gguf"

        Returns:
            List of LanguageModel instances for HuggingFace GGUF models
        """
        try:
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
        except Exception as e:
            log.error(f"Error getting HuggingFace GGUF models: {e}")
            return []

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

        # Get processing context from kwargs
        context = kwargs.get("context")
        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace text generation"
            )

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from nodetool.ml.core.model_manager import ModelManager
            from huggingface_hub.file_download import try_to_load_from_cache

            # Convert messages to chat format for template application
            chat = []
            for msg in messages:
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    # GGUF models do not support multimodal content
                    raise ValueError(
                        "HuggingFace GGUF models do not support multimodal content. "
                        "Please use text-only messages."
                    )
                else:
                    content = ""

                chat.append({"role": msg.role, "content": content})

            # Extract generation parameters from kwargs
            temperature = kwargs.get("temperature", 1.0)
            top_p = kwargs.get("top_p", 1.0)
            do_sample = kwargs.get("do_sample", True)

            # Parse model ID: "repo_id:filename.gguf" format
            if ":" in model and model.count(":") == 1:
                repo_id, filename = model.split(":", 1)
                if not filename.lower().endswith(".gguf"):
                    raise ValueError(f"HuggingFace provider only supports GGUF models, got: {filename}")
            else:
                raise ValueError(f"Model ID must be in format 'repo_id:filename.gguf', got: {model}")

            # Load or retrieve cached pipeline and tokenizer
            cache_key = f"{repo_id}:{filename}:text-generation"
            cached_pipeline = ModelManager.get_model(cache_key, "text-generation")

            if not cached_pipeline:
                # Check if model file is cached
                cache_path = try_to_load_from_cache(repo_id, filename)  # pyright: ignore[reportArgumentType]
                if not cache_path:
                    raise ValueError(f"Model {repo_id}/{filename} must be downloaded first")

                log.info(f"Loading GGUF model {repo_id}/{filename}")

                # Load model with GGUF support
                hf_model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    gguf_file=filename,
                )

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    repo_id,
                    gguf_file=filename,
                )

                # Create pipeline
                cached_pipeline = pipeline(
                    "text-generation", model=hf_model, tokenizer=tokenizer
                )
                ModelManager.set_model(
                    cache_key, cache_key, "text-generation", cached_pipeline
                )

            # Apply chat template to format the prompt properly
            tokenizer = cached_pipeline.tokenizer
            try:
                # Try to apply chat template with generation prompt
                assert tokenizer is not None
                formatted_prompt = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                # Fallback to simple concatenation if chat template not available
                log.warning(f"Chat template not available for {model}, using fallback: {e}")
                prompt_parts = [f"{msg['role']}: {msg['content']}" for msg in chat]
                formatted_prompt = "\n".join(prompt_parts)

            # Generate text using the pipeline
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "return_full_text": False,
            }

            # Run generation in thread to avoid blocking
            def _generate():
                return cached_pipeline(formatted_prompt, **generation_kwargs)  # type: ignore

            result = await asyncio.to_thread(_generate)

            # Extract generated text
            generated_text = result[0]["generated_text"] if result else ""

            # Return as a Message
            return Message(
                role="assistant",
                content=generated_text,
            )

        except Exception as e:
            log.error(f"HuggingFace text generation failed: {e}")
            raise RuntimeError(f"HuggingFace text generation failed: {e}")

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

        # Get processing context from kwargs
        context = kwargs.get("context")
        if context is None:
            raise ValueError(
                "ProcessingContext is required for HuggingFace text generation"
            )

        try:
            import torch
            import threading
            from queue import Queue
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
            from nodetool.ml.core.model_manager import ModelManager
            from huggingface_hub.file_download import try_to_load_from_cache

            # Convert messages to chat format for template application
            chat = []
            for msg in messages:
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    # GGUF models do not support multimodal content
                    raise ValueError(
                        "HuggingFace GGUF models do not support multimodal content. "
                        "Please use text-only messages."
                    )
                else:
                    content = ""

                chat.append({"role": msg.role, "content": content})

            # Extract generation parameters from kwargs
            temperature = kwargs.get("temperature", 1.0)
            top_p = kwargs.get("top_p", 1.0)
            do_sample = kwargs.get("do_sample", True)

            # Parse model ID: "repo_id:filename.gguf" format
            if ":" in model and model.count(":") == 1:
                repo_id, filename = model.split(":", 1)
                if not filename.lower().endswith(".gguf"):
                    raise ValueError(f"HuggingFace provider only supports GGUF models, got: {filename}")
            else:
                raise ValueError(f"Model ID must be in format 'repo_id:filename.gguf', got: {model}")

            # Load or retrieve cached pipeline and tokenizer
            cache_key = f"{repo_id}:{filename}:text-generation"
            cached_pipeline = ModelManager.get_model(cache_key, "text-generation")

            if not cached_pipeline:
                # Check if model file is cached
                cache_path = try_to_load_from_cache(repo_id, filename)  # pyright: ignore[reportArgumentType]
                if not cache_path:
                    raise ValueError(f"Model {repo_id}/{filename} must be downloaded first")

                log.info(f"Loading GGUF model {repo_id}/{filename}")

                # Load model with GGUF support
                hf_model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    gguf_file=filename,
                )

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    repo_id,
                    gguf_file=filename,
                )

                # Create pipeline
                cached_pipeline = pipeline(
                    "text-generation", model=hf_model, tokenizer=tokenizer
                )
                ModelManager.set_model(
                    cache_key, cache_key, "text-generation", cached_pipeline
                )

            # Apply chat template to format the prompt properly
            tokenizer = cached_pipeline.tokenizer
            try:
                # Try to apply chat template with generation prompt
                assert tokenizer is not None
                formatted_prompt = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                # Fallback to simple concatenation if chat template not available
                log.warning(f"Chat template not available for {model}, using fallback: {e}")
                prompt_parts = [f"{msg['role']}: {msg['content']}" for msg in chat]
                formatted_prompt = "\n".join(prompt_parts)

            # Create streaming setup (copied from TextGeneration node)
            token_queue: Queue = Queue()

            class AsyncTextStreamer(TextStreamer):
                def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
                    super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                    self.token_queue = token_queue

                def put(self, value):
                    """Override put to send tokens to queue instead of stdout"""
                    if len(value.shape) > 1 and value.shape[0] > 1:
                        raise ValueError("TextStreamer only supports batch size 1")
                    elif len(value.shape) > 1:
                        value = value[0]

                    if self.skip_prompt and self.next_tokens_are_prompt:
                        self.next_tokens_are_prompt = False
                        return

                    # Decode the token
                    text = self.tokenizer.decode(value, skip_special_tokens=True)  # pyright: ignore[reportAttributeAccessIssue]
                    if text:
                        self.token_queue.put(text)

                def end(self):
                    """Signal end of generation"""
                    self.token_queue.put(None)  # Sentinel value

            # Create the streaming tokenizer
            streamer = AsyncTextStreamer(
                tokenizer,  # pyright: ignore[reportArgumentType]
                skip_prompt=True,
                skip_special_tokens=True,
            )

            # Run generation in a separate thread
            def generate():
                try:
                    generation_kwargs = {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "do_sample": do_sample,
                        "streamer": streamer,
                        "return_full_text": False,
                    }

                    cached_pipeline(formatted_prompt, **generation_kwargs)  # type: ignore[reportArgumentType]
                except Exception as e:
                    token_queue.put(f"Error: {e}")
                    token_queue.put(None)

            # Start generation in background thread
            thread = threading.Thread(target=generate)
            thread.start()

            # Stream tokens as they become available
            try:
                while True:
                    # Check queue with timeout to avoid blocking
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

                    # Non-blocking queue check
                    while not token_queue.empty():
                        token = token_queue.get_nowait()
                        if token is None:  # Sentinel value indicating end
                            return

                        # Yield chunk for streaming
                        yield Chunk(content=token, done=False)

                    # Check if thread is still alive
                    if not thread.is_alive():
                        # Drain any remaining tokens
                        while not token_queue.empty():
                            token = token_queue.get_nowait()
                            if token is not None:
                                yield Chunk(content=token, done=False)
                        break

            finally:
                # Ensure thread completes
                thread.join(timeout=1.0)

        except Exception as e:
            log.error(f"HuggingFace streaming text generation failed: {e}")
            raise RuntimeError(f"HuggingFace streaming text generation failed: {e}")


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

        print(f"Testing generate_messages with model: {model}")
        print(f"Messages: {messages}")
        print("\nStreaming response:")
        print("-" * 50)

        try:
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

            print("\n" + "-" * 50)
            print(f"\nFull response: {full_response}")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()

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

        print(f"\nTesting generate_message with model: {model}")
        print(f"Messages: {messages}")
        print("\nResponse:")
        print("-" * 50)

        try:
            # Get the response
            response = await provider.generate_message(
                messages=messages,
                model=model.id,
                max_tokens=50,
                temperature=0.7,
                context=context,
            )

            print(response.content)
            print("-" * 50)

        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()

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

