"""HuggingFace integration for NodeTool."""

# Register the image provider
from nodetool.image.providers import register_image_provider
from nodetool.huggingface.huggingface_image_provider import HuggingFaceImageProvider

register_image_provider("huggingface", lambda: HuggingFaceImageProvider())

__all__ = ["HuggingFaceImageProvider"]
