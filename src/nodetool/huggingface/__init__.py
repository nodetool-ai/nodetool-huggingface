"""HuggingFace integration for NodeTool."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.huggingface.huggingface_local_provider import HuggingFaceLocalProvider

__all__ = ["HuggingFaceLocalProvider"]


def __getattr__(name: str):
    if name == "HuggingFaceLocalProvider":
        from nodetool.huggingface.huggingface_local_provider import HuggingFaceLocalProvider

        return HuggingFaceLocalProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
