"""Profile Flux Schnell Nunchaku INT4 load + inference. Run from nodetool-huggingface:
  python scripts/profile_flux_nunchaku_load.py
"""
from __future__ import annotations

import asyncio
import sys
import time


def stamp(label: str, t0: float) -> float:
    now = time.perf_counter()
    print(f"[{now - t0:7.1f}s] {label}", flush=True)
    return now


async def main() -> None:
    t0 = time.perf_counter()
    stamp("start", t0)

    import torch

    stamp(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}", t0)
    if torch.cuda.is_available():
        stamp(f"gpu={torch.cuda.get_device_name(0)}", t0)

    from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE
    from nodetool.huggingface.nunchaku_pipelines import load_nunchaku_flux_pipeline
    from nodetool.workflows.processing_context import ProcessingContext

    repo = "nunchaku-tech/nunchaku-flux.1-schnell"
    path = "svdq-int4_r32-flux.1-schnell.safetensors"
    base = "black-forest-labs/FLUX.1-schnell"

    for r, p in [
        (base, "model_index.json"),
        (repo, path),
        ("nunchaku-tech/nunchaku-t5", "awq-int4-flux.1-t5xxl.safetensors"),
    ]:
        resolved = await HF_FAST_CACHE.resolve(r, p)
        stamp(f"cache {r}/{p} -> {'OK' if resolved else 'MISSING'}", t0)

    ctx = ProcessingContext(device="cuda" if torch.cuda.is_available() else "cpu")

    stamp("load_nunchaku_flux_pipeline BEGIN", t0)
    pipeline = await load_nunchaku_flux_pipeline(
        context=ctx,
        repo_id=repo,
        transformer_path=path,
        node_id="profile-test",
        cache_key="profile-flux-schnell-int4",
    )
    stamp("load_nunchaku_flux_pipeline DONE", t0)

    stamp("inference BEGIN (512x512, 4 steps)", t0)
    with torch.inference_mode():
        out = pipeline(
            prompt="a red cube on white background",
            guidance_scale=0.0,
            height=512,
            width=512,
            num_inference_steps=4,
            max_sequence_length=256,
        )
    stamp(f"inference DONE, image={out.images[0].size}", t0)


if __name__ == "__main__":
    asyncio.run(main())
