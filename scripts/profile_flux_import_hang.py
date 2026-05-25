"""Reproduce worker import sequence with stdio guard + timing."""
from __future__ import annotations

import asyncio
import sys
import time


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _step(label: str) -> None:
    print(f"[{_ts()}] {label}", file=sys.stderr, flush=True)


async def main() -> None:
    from nodetool.worker.stdio_stdout_guard import install_stdio_stdout_guard

    _step("install_stdio_stdout_guard")
    install_stdio_stdout_guard()

    from nodetool.worker.node_loader import load_nodes

    _step("load_nodes begin")
    t0 = time.perf_counter()
    load_nodes(["huggingface"])
    _step(f"load_nodes done in {time.perf_counter() - t0:.1f}s")

    from nodetool.nodes.huggingface.text_to_image import Flux
    from nodetool.worker.context_stub import WorkerContext
    from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode

    node = Flux()
    node.assign_property("variant", "schnell")
    node.assign_property("quantization", "int4")
    node.assign_property("enable_cpu_offload", False)
    node.assign_property("width", 512)
    node.assign_property("height", 512)
    node.assign_property("prompt", "test")
    ctx = WorkerContext(secrets={}, cancel_event=None)

    _step("pre_process begin")
    t0 = time.perf_counter()
    await node.pre_process(ctx)
    _step(f"pre_process done in {time.perf_counter() - t0:.1f}s")

    def _import_flux_deps() -> None:
        _step("thread: import torch begin")
        import torch  # noqa: F401

        _step("thread: import torch done, FluxPipeline begin")
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline  # noqa: F401

        _step("thread: FluxPipeline import done")

    _step("preload import (to_thread) begin")
    t0 = time.perf_counter()
    await asyncio.to_thread(_import_flux_deps)
    _step(f"preload import done in {time.perf_counter() - t0:.1f}s")

    _step("preload_model begin")
    t0 = time.perf_counter()
    await node.preload_model(ctx)
    _step(f"preload_model done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
