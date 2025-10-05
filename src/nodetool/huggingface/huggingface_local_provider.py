"""
HuggingFace local provider implementation.

This module implements the BaseProvider interface for locally cached HuggingFace models.
Uses the Text2Image and ImageToImage nodes from the nodetool-huggingface package.
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
from nodetool.workflows.types import Chunk
from typing import List, Sequence, Any, AsyncIterator
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

from nodetool.workflows.recommended_models import get_recommended_models
from nodetool.metadata.types import LanguageModel

async def get_hf_cached_language_models() -> list[UnifiedModel]:
    """
    Scan the Hugging Face cache directory and return models that are compatible
    with language generation architectures.
    """
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

        if model_info.pipeline_tag in ["text-generation"]:
            models.append(
                UnifiedModel(
                    id=repo.repo_id,
                    type=model_type,
                    name=repo.repo_id,
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
        """Generate an image from a text prompt using HuggingFace models.

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
            # Import here to avoid circular dependencies
            from nodetool.nodes.huggingface.text_to_image import Text2Image

            # Create the Text2Image node with parameters
            node = Text2Image(
                model=HFTextToImage(
                    repo_id=params.model.id,
                ),
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or "",
                num_inference_steps=params.num_inference_steps or 50,
                guidance_scale=params.guidance_scale or 7.5,
                width=params.width or 512,
                height=params.height or 512,
                seed=params.seed if params.seed is not None else -1,
                pag_scale=0.0,  # Disable PAG by default for compatibility
            )

            # Preload the model
            await node.preload_model(context)

            # Process to generate the image
            output = await node.process(context)

            # The output is a dict with 'image' and 'latent' keys
            image_ref = output.get("image")
            if image_ref is None:
                raise RuntimeError("Node did not return an image")

            # Convert ImageRef to PIL Image
            pil_image = await context.image_to_pil(image_ref)

            # Convert PIL Image to bytes
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format="PNG")
            image_bytes = img_buffer.getvalue()

            return image_bytes

        except Exception as e:
            raise RuntimeError(f"HuggingFace text-to-image generation failed: {e}")

    async def image_to_image(
        self,
        image: ImageBytes,
        params: ImageToImageParams,
        context: ProcessingContext | None = None,
        timeout_s: int | None = None,
    ) -> ImageBytes:
        """Transform an image based on a text prompt using HuggingFace models.

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
            # Import here to avoid circular dependencies
            from nodetool.nodes.huggingface.image_to_image import (
                ImageToImage as ImageToImageNode,
            )

            # Convert input image bytes to PIL Image, then to ImageRef
            pil_image = Image.open(BytesIO(image))
            input_image_ref = await context.image_from_pil(pil_image)

            # Create the ImageToImage node with parameters
            node = ImageToImageNode(
                model=HFImageToImage(
                    repo_id=params.model.id,
                ),
                image=input_image_ref,
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or "",
                strength=params.strength or 0.8,
                num_inference_steps=params.num_inference_steps or 25,
                guidance_scale=params.guidance_scale or 7.5,
                seed=params.seed if params.seed is not None else -1,
            )

            # Preload the model
            await node.preload_model(context)

            # Process to transform the image
            output_image_ref = await node.process(context)

            # Convert ImageRef to PIL Image
            pil_output = await context.image_to_pil(output_image_ref)

            # Convert PIL Image to bytes
            img_buffer = BytesIO()
            pil_output.save(img_buffer, format="PNG")
            image_bytes = img_buffer.getvalue()

            self.usage["total_requests"] += 1
            self.usage["total_images"] += 1

            return image_bytes

        except Exception as e:
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
        """Get available HuggingFace language models."""
        unified_models = await get_hf_cached_language_models()
        return [LanguageModel(id=model.id, name=model.name, provider=Provider.HuggingFace) for model in unified_models]

    async def get_available_image_models(self) -> List[ImageModel]:
        """Get available HuggingFace image models."""
        unified_models = await get_hf_cached_image_models()

        # Convert UnifiedModel instances to ImageModel instances
        image_models = [
            ImageModel(id=model.id, name=model.name, provider=Provider.HuggingFace)
            for model in unified_models
        ]

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
        """Generate a single message completion using HuggingFace text generation models.

        Args:
            messages: Conversation history to send
            model: Model repository ID (e.g., "gpt2", "Qwen/Qwen2-0.5B-Instruct")
            tools: Optional tool definitions (not supported for HF models)
            max_tokens: Maximum tokens to generate
            context_window: Maximum tokens to consider for context
            response_format: Optional response schema (not supported for HF models)
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
                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    # Handle multimodal content - extract text only
                    text_parts = []
                    for content_item in msg.content:
                        if isinstance(content_item, MessageTextContent):
                            text_parts.append(content_item.text)
                    content = " ".join(text_parts)

                chat.append({"role": msg.role, "content": content})

            # Extract generation parameters from kwargs
            temperature = kwargs.get("temperature", 1.0)
            top_p = kwargs.get("top_p", 1.0)
            do_sample = kwargs.get("do_sample", True)
            path = kwargs.get("path")  # Optional GGUF file path

            # Check if this is a GGUF model
            is_gguf = path is not None and path.lower().endswith(".gguf")

            # Load or retrieve cached pipeline and tokenizer
            if is_gguf:
                # Load GGUF model
                cache_key = f"{model}:{path}:text-generation"
                cached_pipeline = ModelManager.get_model(cache_key, "text-generation")

                if not cached_pipeline:
                    # Check if model file is cached
                    cache_path = try_to_load_from_cache(model, path)  # pyright: ignore[reportArgumentType]
                    if not cache_path:
                        raise ValueError(f"Model {model}/{path} must be downloaded first")

                    log.info(f"Loading GGUF model {model}/{path}")

                    # Load model with GGUF support
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        model,
                        torch_dtype=torch.float32,
                        device_map="auto",
                        gguf_file=path,
                    )

                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        model,
                        gguf_file=path,
                    )

                    # Create pipeline
                    cached_pipeline = pipeline(
                        "text-generation", model=hf_model, tokenizer=tokenizer
                    )
                    ModelManager.set_model(
                        cache_key, cache_key, "text-generation", cached_pipeline
                    )
            else:
                # Regular model loading
                cached_pipeline = ModelManager.get_model(model, "text-generation")

                if not cached_pipeline:
                    log.info(f"Loading model {model}")
                    cached_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device=context.device,
                    )
                    ModelManager.set_model(model, model, "text-generation", cached_pipeline)

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
        """Stream message completions using HuggingFace text generation models.

        Args:
            messages: Conversation history to send
            model: Model repository ID (e.g., "gpt2", "Qwen/Qwen2-0.5B-Instruct")
            tools: Optional tool definitions (not supported for HF models)
            max_tokens: Maximum tokens to generate
            context_window: Maximum tokens to consider for context
            response_format: Optional response schema (not supported for HF models)
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
                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    # Handle multimodal content - extract text only
                    text_parts = []
                    for content_item in msg.content:
                        if isinstance(content_item, MessageTextContent):
                            text_parts.append(content_item.text)
                    content = " ".join(text_parts)

                chat.append({"role": msg.role, "content": content})

            # Extract generation parameters from kwargs
            temperature = kwargs.get("temperature", 1.0)
            top_p = kwargs.get("top_p", 1.0)
            do_sample = kwargs.get("do_sample", True)
            path = kwargs.get("path")  # Optional GGUF file path

            # Check if this is a GGUF model
            is_gguf = path is not None and path.lower().endswith(".gguf")

            # Load or retrieve cached pipeline and tokenizer
            if is_gguf:
                # Load GGUF model
                cache_key = f"{model}:{path}:text-generation"
                cached_pipeline = ModelManager.get_model(cache_key, "text-generation")

                if not cached_pipeline:
                    # Check if model file is cached
                    cache_path = try_to_load_from_cache(model, path)  # pyright: ignore[reportArgumentType]
                    if not cache_path:
                        raise ValueError(f"Model {model}/{path} must be downloaded first")

                    log.info(f"Loading GGUF model {model}/{path}")

                    # Load model with GGUF support
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        model,
                        torch_dtype=torch.float32,
                        device_map="auto",
                        gguf_file=path,
                    )

                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        model,
                        gguf_file=path,
                    )

                    # Create pipeline
                    cached_pipeline = pipeline(
                        "text-generation", model=hf_model, tokenizer=tokenizer
                    )
                    ModelManager.set_model(
                        cache_key, cache_key, "text-generation", cached_pipeline
                    )
            else:
                # Regular model loading
                cached_pipeline = ModelManager.get_model(model, "text-generation")

                if not cached_pipeline:
                    log.info(f"Loading model {model}")
                    cached_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device=context.device,
                    )
                    ModelManager.set_model(model, model, "text-generation", cached_pipeline)

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
        context = ProcessingContext(
            user_id="test_user",
            auth_token="test_token",
        )

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
        model = "ibm-granite/granite-4.0-h-small"

        print(f"Testing generate_messages with model: {model}")
        print(f"Messages: {messages}")
        print("\nStreaming response:")
        print("-" * 50)

        try:
            # Stream the response
            full_response = ""
            async for chunk in provider.generate_messages(
                messages=messages,
                model=model,
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                context=context,
            ):
                print(chunk.content, end="", flush=True)
                full_response += chunk.content

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
        context = ProcessingContext(
            user_id="test_user",
            auth_token="test_token",
        )

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
        model = "gpt2"

        print(f"\nTesting generate_message with model: {model}")
        print(f"Messages: {messages}")
        print("\nResponse:")
        print("-" * 50)

        try:
            # Get the response
            response = await provider.generate_message(
                messages=messages,
                model=model,
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

    # Run tests
    print("=" * 50)
    print("Testing HuggingFace Local Provider")
    print("=" * 50)

    # Test streaming
    asyncio.run(test_generate_messages())

    # Test non-streaming
    asyncio.run(test_generate_message())