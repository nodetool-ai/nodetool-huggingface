from __future__ import annotations

from nodetool.workflows.graph import BaseNode
import asyncio
from nodetool.config.logging_config import get_logger
from nodetool.nodes.huggingface.huggingface_node import (
    setup_hf_logging,
)
from nodetool.workflows.processing_context import ProcessingContext
from typing import TYPE_CHECKING, Any, TypeVar
from nodetool.huggingface.huggingface_local_provider import (
    load_pipeline,
    load_model,
)

if TYPE_CHECKING:
    import torch
    from transformers.pipelines.base import Pipeline

T = TypeVar("T")

log = get_logger(__name__)


class HuggingFacePipelineNode(BaseNode):
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
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ):
        """Load a HuggingFace pipeline model (instance method wrapper)."""
        return await load_pipeline(
            self.id,
            context,
            pipeline_task,
            model_id,
            device=device,
            torch_dtype=torch_dtype,
            skip_cache=self.should_skip_cache(),
            **kwargs,
        )

    async def load_model(
        self,
        context: ProcessingContext,
        model_class: type[T],
        model_id: str,
        variant: str | None = None,
        torch_dtype: torch.dtype | None = None,
        path: str | None = None,
        skip_cache: bool = False,
        **kwargs: Any,
    ) -> T:
        """Load a HuggingFace model (instance method wrapper)."""
        return await load_model(
            self.id,
            context,
            model_class,
            model_id,
            variant=variant,
            torch_dtype=torch_dtype,
            path=path,
            skip_cache=skip_cache or self.should_skip_cache(),
            **kwargs,
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None and hasattr(self._pipeline, "to"):
            self._pipeline.to(device)  # type: ignore

    async def run_pipeline_in_thread(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the underlying HF pipeline in a background thread to avoid
        blocking the asyncio event loop.
        """
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")

        pipeline = self._pipeline

        def _call():
            return pipeline(*args, **kwargs)

        return await asyncio.to_thread(_call)

    async def process(self, context: ProcessingContext) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def requires_gpu(self) -> bool:
        return True
