"""HuggingFace integration for NodeTool."""

# Register the image provider
from nodetool.image.providers import register_image_provider
from nodetool.huggingface.huggingface_image_provider import HuggingFaceLocalProvider

register_image_provider("huggingface", lambda: HuggingFaceLocalProvider())

__all__ = ["HuggingFaceLocalProvider"]
