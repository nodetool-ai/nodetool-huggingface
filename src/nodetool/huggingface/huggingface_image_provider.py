"""
HuggingFace local provider implementation.

This module implements the BaseProvider interface for locally cached HuggingFace models.
Uses the Text2Image and ImageToImage nodes from the nodetool-huggingface package.
"""

import asyncio
from typing import List, Set
from huggingface_hub import CacheNotFound, scan_cache_dir
from nodetool.chat.providers.base import BaseProvider, ProviderCapability
from nodetool.image.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.integrations.huggingface.huggingface_models import (
    fetch_model_info,
    has_model_index,
    model_type_from_model_info,
)
from nodetool.metadata.types import HFTextToImage, HFImageToImage
from io import BytesIO
from nodetool.types.model import UnifiedModel
from nodetool.workflows.processing_context import ProcessingContext
from PIL import Image
from nodetool.metadata.types import ImageModel, Provider
from typing import List
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

from nodetool.workflows.recommended_models import get_recommended_models


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


class HuggingFaceLocalProvider(BaseProvider):
    """Local provider for HuggingFace models using cached diffusion pipelines."""

    provider_name = "hf_inference"

    def __init__(self):
        super().__init__()

    def get_capabilities(self) -> Set[ProviderCapability]:
        """HuggingFace provider supports both text-to-image and image-to-image generation."""
        return {
            ProviderCapability.TEXT_TO_IMAGE,
            ProviderCapability.IMAGE_TO_IMAGE,
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

        self._log_api_request("text_to_image", params)

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

            self.usage["total_requests"] += 1
            self.usage["total_images"] += 1
            self._log_api_response("text_to_image", 1)

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

        self._log_api_request("image_to_image", params)

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
            self._log_api_response("image_to_image", 1)

            return image_bytes

        except Exception as e:
            raise RuntimeError(f"HuggingFace image-to-image generation failed: {e}")

    async def get_available_image_models(self) -> List[ImageModel]:
        """Get available HuggingFace image models."""
        unified_models = await get_hf_cached_image_models()

        # Convert UnifiedModel instances to ImageModel instances
        image_models = [
            ImageModel(id=model.id, name=model.name, provider=Provider.HuggingFace)
            for model in unified_models
        ]

        return image_models


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(get_hf_cached_image_models()))
