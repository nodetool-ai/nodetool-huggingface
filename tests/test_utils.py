"""
Shared smoke-test helpers to exercise Hugging Face DSL nodes in single-node graphs.

Mirrors the lightweight style used by nodetool-base DSL examples: instantiate a DSL
node with sensible defaults, create a graph, and run it.
"""

from __future__ import annotations

import asyncio
import sys
import gc
import uuid
from contextlib import suppress
import inspect
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from pydub.generators import Sine
import os

# Force lightweight, offline-friendly defaults for smoke runs
os.environ.setdefault("ENV", "test")
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_KEY", "")

import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode, create_graph, run_graph
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider as get_provider_async
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext

# Ensure repo and src on path when run directly (python tests/test_x.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

# Allow downloads for smoke runs explicitly (can be overridden by caller)
os.environ.setdefault("NODETOOL_HF_ALLOW_DOWNLOAD", "1")
# Mark smoke mode so heavy nodes can short-circuit expensive generation
os.environ.setdefault("NODETOOL_SMOKE_TEST", "1")


class SampleAssets:
    """Build minimal assets and model defaults so nodes can run with defaults."""

    def __init__(self, context: ProcessingContext) -> None:
        self.context = context
        self._image_ref: types.ImageRef | None = None
        self._audio_ref: types.AudioRef | None = None
        self._dataframe_ref: types.DataframeRef | None = None

    async def image_ref(self) -> types.ImageRef:
        if self._image_ref is None:
            img = Image.new("RGB", (64, 64), color=(200, 100, 50))
            ref = self.context.wrap_object(img)
            assert isinstance(ref, types.ImageRef)
            self._image_ref = ref
        return self._image_ref

    async def audio_ref(self) -> types.AudioRef:
        if self._audio_ref is None:
            tone = Sine(440).to_audio_segment(duration=500)
            ref = self.context.wrap_object(tone)
            assert isinstance(ref, types.AudioRef)
            self._audio_ref = ref
        return self._audio_ref

    async def dataframe_ref(self) -> types.DataframeRef:
        if self._dataframe_ref is None:
            df = pd.DataFrame({"city": ["Paris", "Berlin"], "pop": [2.1, 3.6]})
            ref = self.context.wrap_object(df)
            assert isinstance(ref, types.DataframeRef)
            self._dataframe_ref = ref
        return self._dataframe_ref

    def text(self) -> str:
        return "hello world"

    def candidate_labels(self) -> str:
        return "label_one, label_two"

    async def ensure_asset(self, value: types.AssetRef):
        if isinstance(value, types.ImageRef):
            return await self.image_ref()
        if isinstance(value, types.AudioRef):
            return await self.audio_ref()
        if isinstance(value, types.VideoRef):
            img_ref = await self.image_ref()
            return types.VideoRef(uri=img_ref.uri)
        if isinstance(value, types.DataframeRef):
            return await self.dataframe_ref()
        return value

    def audio_chunk(self) -> types.AudioChunk:
        return types.AudioChunk(timestamp=(0.0, 1.0), text="audio chunk")


async def populate_node_inputs(node: GraphNode, assets: SampleAssets):
    """Fill missing inputs with minimal defaults so the node can execute."""
    for name, field in node.model_fields.items():  # type: ignore[attr-defined]
        value = getattr(node, name)

        if isinstance(value, str) and not value:
            setattr(
                node,
                name,
                assets.candidate_labels() if "candidate_labels" in name else assets.text(),
            )
            continue

        # Speed up smoke runs by shrinking generation-heavy parameters
        if isinstance(value, int) and "num_inference_steps" in name:
            setattr(node, name, min(value, 2))
            continue
        if isinstance(value, float) and ("audio_length" in name or "duration" in name):
            setattr(node, name, min(value, 1.0))
            continue
        if isinstance(value, int) and "num_waveforms_per_prompt" in name:
            setattr(node, name, 1)
            continue
        if isinstance(value, int) and "max_new_tokens" in name:
            setattr(node, name, min(value, 32))
            continue

        if hasattr(value, "repo_id") and getattr(value, "repo_id", "") == "":
            raise Exception(f"No model default defined for {node}")

        if isinstance(value, types.AssetRef) and getattr(value, "uri", "") == "":
            new_value = await assets.ensure_asset(value)
            setattr(node, name, new_value)
            continue

        if isinstance(value, list) and not value:
            if "chunks" in name:
                setattr(node, name, [assets.audio_chunk()])
                continue
            if "labels" in name:
                setattr(node, name, assets.candidate_labels().split(","))
                continue


def _drop_node_model_references(node: GraphNode) -> None:
    """Remove common heavy attributes so GC can reclaim VRAM-heavy objects."""
    heavy_attrs = (
        "_pipeline",
        "_model",
        "_models",
        "_text_encoder",
        "_text_encoder_2",
        "_transformer",
        "_transformer2",
        "_transformer_2",
        "_vae",
        "_unet",
        "_controlnet",
        "_processor",
    )
    for attr in heavy_attrs:
        if hasattr(node, attr):
            with suppress(Exception):
                setattr(node, attr, None)

def _remove_accelerate_hooks(node: GraphNode) -> None:
    """Detach Accelerate CPU offload hooks to avoid reallocating on GPU during cleanup."""
    try:
        from accelerate.hooks import remove_hook_from_module  # type: ignore
        import torch
        Module = torch.nn.Module
    except Exception:
        return

    def _strip(obj: object):
        if obj is None:
            return
        if isinstance(obj, Module):
            with suppress(Exception):
                remove_hook_from_module(obj, recurse=True)
            return
        components = getattr(obj, "_components", None)
        if isinstance(components, dict):
            for comp in components.values():
                _strip(comp)

    for attr_name in ("_pipeline", "_model", "_models", "_unet", "_vae", "_transformer", "_text_encoder"):
        _strip(getattr(node, attr_name, None))


async def unload_models_from_vram(node: GraphNode | None, reason: str) -> None:
    """Best-effort cleanup to free VRAM between smoke runs."""
    if node is not None:
        _remove_accelerate_hooks(node)
        if hasattr(node, "move_to_device"):
            try:
                move_result = node.move_to_device("cpu")  # type: ignore[attr-defined]
                if inspect.isawaitable(move_result):
                    await move_result
            except Exception:
                pass
        _drop_node_model_references(node)

    try:
        from nodetool.ml.core.model_manager import ModelManager
    except Exception:
        ModelManager = None  # type: ignore[assignment]

    if ModelManager is not None:
        with suppress(Exception):
            ModelManager.free_vram_if_needed(reason=reason, aggressive=True)
        with suppress(Exception):
            ModelManager.clear()

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        with suppress(Exception):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    gc.collect()


async def run_nodes(nodes: Iterable[GraphNode]):
    """Run each provided node instance in its own single-node graph."""
    results: dict[str, dict[str, object]] = {}
    failures: list[tuple[str, Exception]] = []

    # Disable required-input validation for smoke tests
    try:
        from nodetool.workflows.base_node import BaseNode

        BaseNode.validate_inputs = lambda self, input_edges: []  # type: ignore[assignment]
        BaseNode.requires_gpu = lambda self: False  # type: ignore[assignment]
        try:
            from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
            HuggingFacePipelineNode.requires_gpu = lambda self: False  # type: ignore[assignment]
        except Exception:
            pass
    except Exception:
        pass

    test_mode = os.getenv("NODETOOL_SMOKE_TEST", "").lower() in {"1", "true", "yes", "on"}

    async with ResourceScope():
        provider_cache: dict[str, object] = {}

        for node in nodes:
            context = ProcessingContext(
                user_id="test_user",
                auth_token="test_token",
                job_id=f"smoke-{uuid.uuid4().hex}",
            )
            assets = SampleAssets(context)

            try:
                print(f"[smoke] Running {node.__class__.__name__}", flush=True)

                if node.__class__.__module__.endswith("text_generation"):
                    if "text_generation" not in provider_cache:
                        provider_cache["text_generation"] = await get_provider_async(
                            Provider.HuggingFace,
                            user_id=context.user_id or "test_user",
                        )
                    import nodetool.nodes.huggingface.text_generation as tg

                    tg.get_provider = lambda *_args, **_kwargs: provider_cache[
                        "text_generation"
                    ]

                await populate_node_inputs(node, assets)
                try:
                    node.__class__.requires_gpu = lambda self: False  # type: ignore[assignment]
                except Exception:
                    pass
                graph = create_graph(node)
                results[node.__class__.__name__] = await run_graph(graph, context=context)
                print(f"[smoke] Completed {node.__class__.__name__}", flush=True)
            except Exception as exc:  # pragma: no cover - smoke path
                failures.append((node.__class__.__name__, exc))
            finally:
                await unload_models_from_vram(
                    node=node, reason=f"Smoke cleanup after {node.__class__.__name__}"
                )

    if failures:
        joined = "; ".join(f"{name}: {err}" for name, err in failures)
        raise RuntimeError(f"Smoke run failed for: {joined}")

    return results


def main(nodes: Iterable[GraphNode]):
    """CLI entry point used by per-module smoke scripts."""
    return asyncio.run(run_nodes(nodes))
