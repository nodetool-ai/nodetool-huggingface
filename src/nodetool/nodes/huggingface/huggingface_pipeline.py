from pathlib import Path
from nodetool.workflows.graph import BaseNode
import torch
import asyncio
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.nodes.huggingface.huggingface_node import (
    setup_hf_logging,
)
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field
from transformers.pipelines.base import Pipeline
from transformers.pipelines import pipeline
from typing import Any
from nodetool.ml.core.model_manager import ModelManager
from huggingface_hub.file_download import try_to_load_from_cache
from typing import Any, TypeVar

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

    _pipeline: Pipeline | None = None

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
        if model_id == "" or model_id is None:
            raise ValueError("Please select a model")

        cached_model = ModelManager.get_model(model_id, pipeline_task)
        if cached_model:
            return cached_model

        # if not context.is_huggingface_model_cached(model_id):
        #     raise ValueError(f"Model {model_id} must be downloaded first")

        if device is None:
            device = context.device

        if (
            isinstance(model_id, str)
            and not self.should_skip_cache()
            and not Path(model_id).expanduser().exists()
        ):
            repo_id_for_cache = model_id
            revision = kwargs.get("revision")
            cache_dir = kwargs.get("cache_dir")

            if "@" in repo_id_for_cache and revision is None:
                repo_id_for_cache, revision = repo_id_for_cache.rsplit("@", 1)

            cache_checked = False
            for candidate in ("model_index.json", "config.json"):
                try:
                    cache_path = try_to_load_from_cache(
                        repo_id_for_cache,
                        candidate,
                        revision=revision,
                        cache_dir=cache_dir,
                    )
                except Exception:
                    cache_path = None

                if cache_path:
                    cache_checked = True
                    break

            if not cache_checked:
                raise ValueError(f"Model {model_id} must be downloaded first")

        context.post_message(
            JobUpdate(
                status="running",
                message=f"Loading pipeline {type(model_id) == str and model_id or pipeline_task} from HuggingFace",
            )
        )
        model = pipeline(
            pipeline_task,  # type: ignore
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs,
        )  # type: ignore
        ModelManager.set_model(self.id, model_id, pipeline_task, model)
        return model  # type: ignore

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
        if model_id == "":
            raise ValueError("Please select a model")

        if not skip_cache and not self.should_skip_cache():
            cached_model = ModelManager.get_model(model_id, model_class.__name__, path)
            if cached_model:
                return cached_model

        if path:
            cache_path = try_to_load_from_cache(model_id, path)
            if not cache_path:
                raise ValueError(f"Model {model_id}/{path} must be downloaded first")
            log.info(f"Loading model {model_id}/{path} from {cache_path}")
            context.post_message(
                JobUpdate(
                    status="running",
                    message=f"Loading model {model_id} from {cache_path}",
                )
            )

            if hasattr(model_class, "from_single_file"):
                model = model_class.from_single_file(  # type: ignore
                    cache_path,
                    torch_dtype=torch_dtype,
                    variant=variant,
                    **kwargs,
                )
            else:
                # Fallback to from_pretrained for classes without from_single_file
                model = model_class.from_pretrained(  # type: ignore
                    model_id,
                    torch_dtype=torch_dtype,
                    variant=variant,
                    **kwargs,
                )
        else:
            log.info(f"Loading model {model_id} from HuggingFace")
            context.post_message(
                JobUpdate(
                    status="running",
                    message=f"Loading model {model_id} from HuggingFace",
                )
            )
            model = model_class.from_pretrained(  # type: ignore
                model_id,
                torch_dtype=torch_dtype,
                variant=variant,
                **kwargs,
            )

        ModelManager.set_model(self.id, model_id, model_class.__name__, model, path)
        return model

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
