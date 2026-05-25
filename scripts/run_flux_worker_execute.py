"""Run Flux node through worker executor (same path as NodeTool)."""
from __future__ import annotations

import asyncio
import time


async def main() -> None:
    from nodetool.worker.executor import execute_node
    from nodetool.worker.node_loader import load_nodes

    load_nodes(["huggingface"])

    fields = {
        "variant": "schnell",
        "quantization": "int4",
        "enable_cpu_offload": True,
        "guidance_scale": 0.0,
        "width": 512,
        "height": 512,
        "num_inference_steps": 9,
        "max_sequence_length": 512,
        "seed": 42,
        "prompt": "a red cube on white background",
    }

    t0 = time.perf_counter()
    print("execute_node BEGIN", flush=True)
    result = await execute_node(
        node_type="huggingface.text_to_image.Flux",
        fields=fields,
        secrets={},
        input_blobs={},
    )
    elapsed = time.perf_counter() - t0
    print(f"execute_node DONE in {elapsed:.1f}s", flush=True)
    print("outputs keys:", list(result.get("outputs", {}).keys()), flush=True)
    print("blobs:", list(result.get("blobs", {}).keys()), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
