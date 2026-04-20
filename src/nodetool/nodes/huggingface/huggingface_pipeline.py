from __future__ import annotations

from nodetool.workflows.graph import BaseNode
import asyncio
import concurrent.futures
from nodetool.config.logging_config import get_logger
from nodetool.nodes.huggingface.huggingface_node import (
    setup_hf_logging,
)
from nodetool.workflows.processing_context import ProcessingContext
from typing import TYPE_CHECKING, Any, TypeVar
from nodetool.huggingface.local_provider_utils import load_model, load_pipeline

if TYPE_CHECKING:
    import torch
    from transformers.pipelines.base import Pipeline
    from nodetool.types.model import UnifiedModel

T = TypeVar("T")

log = get_logger(__name__)

# Shared thread pool for all HF pipeline operations to minimize CUDA memory pool fragmentation
# Using a single thread ensures consistent CUDA memory pool usage
_pipeline_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf_pipeline")


def select_inference_dtype() -> "torch.dtype":
    """
    Prefer bfloat16 when supported; otherwise fall back to float16 on GPUs and
    float32 on CPU. Keeps pipelines on a safe dtype for the current hardware.
    """
    import torch

    if torch.cuda.is_available():
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported) and is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16

    return torch.float32


class HuggingFacePipelineNode(BaseNode):
    @classmethod
    def unified_recommended_models(
        cls, include_model_info: bool = False
    ) -> list["UnifiedModel"]:
        from nodetool.types.model import UnifiedModel

        recommended_models = cls.get_recommended_models()
        if not recommended_models:
            return []

        unified_models: list[UnifiedModel] = []
        for model in recommended_models:
            repo_id = getattr(model, "repo_id", None)
            if not repo_id:
                continue
            path = getattr(model, "path", None)
            model_id = f"{repo_id}:{path}" if path else repo_id
            unified_models.append(
                UnifiedModel(
                    id=model_id,
                    repo_id=repo_id,
                    path=path,
                    type=getattr(model, "type", None),
                    name=repo_id,
                    cache_path=None,
                    allow_patterns=getattr(model, "allow_patterns", None),
                    ignore_patterns=getattr(model, "ignore_patterns", None),
                )
            )
        return unified_models

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not HuggingFacePipelineNode

    async def pre_process(self, context: ProcessingContext):
        """Base pre_process that sets up HuggingFace logging for all HF nodes."""
        # Set up HuggingFace logging redirection automatically
        setup_hf_logging(context, self.id, self.get_title())
        # Call parent implementation

    _pipeline: Any = None

    def should_skip_cache(self):
        return False

    async def load_pipeline(
        self,
        context: ProcessingContext,
        pipeline_task: str,
        model_id: Any,
        cache_key: str | None = None,
        **kwargs: Any,
    ):
        """Load a HuggingFace pipeline model (instance method wrapper)."""
        return await load_pipeline(
            self.id,
            context,
            pipeline_task,
            model_id,
            skip_cache=self.should_skip_cache(),
            cache_key=cache_key,
            **kwargs,
        )

    async def load_model(
        self,
        context: ProcessingContext,
        model_class: type[T],
        model_id: str,
        skip_cache: bool = False,
        cache_key: str | None = None,
        **kwargs: Any,
    ) -> T:
        """Load a HuggingFace model (instance method wrapper)."""
        return await load_model(
            self.id,
            context,
            model_class,
            model_id,
            skip_cache=skip_cache or self.should_skip_cache(),
            cache_key=cache_key,
            **kwargs,
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None and hasattr(self._pipeline, "to"):
            self._pipeline.to(device)

    async def run_pipeline_in_thread(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the underlying HF pipeline in a background thread to avoid
        blocking the asyncio event loop.
        Uses a shared thread pool to minimize CUDA memory pool fragmentation.
        """
        import torch
        import gc

        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        pipeline = self._pipeline

        def _call():
            with torch.inference_mode():
                result = pipeline(*args, **kwargs)

            # Explicit cleanup: synchronize, clear any temporary allocations
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Force Python GC to collect any temporary tensors
                gc.collect()

            return result

        # Use shared thread pool instead of asyncio.to_thread to ensure
        # consistent CUDA memory pool usage
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_pipeline_thread_pool, _call)

    async def process(self, context: ProcessingContext) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def requires_gpu(self) -> bool:
        return True
