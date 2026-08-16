"""
Memory utilities for HuggingFace pipelines.

This module provides HuggingFace-specific memory utilities built on top of
the core memory utilities from nodetool.workflows.memory_utils.

For general memory utilities (memory tracking, GC, etc.), use:
    from nodetool.workflows.memory_utils import (
        get_memory_usage_mb,
        get_gpu_memory_usage_mb,
        log_memory,
        run_gc,
        log_memory_summary,
        MemoryTracker,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nodetool.config.logging_config import get_logger
from nodetool.workflows.memory_utils import (
    get_memory_usage_mb,
    get_gpu_memory_usage_mb,
    log_memory,
    run_gc,
    log_memory_summary,
    MemoryTracker,
)

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


__all__ = [
    "get_memory_usage_mb",
    "get_gpu_memory_usage_mb",
    "log_memory",
    "run_gc",
    "log_memory_summary",
    "MemoryTracker",
    "has_cpu_offload_enabled",
    "apply_cpu_offload_if_needed",
    "preferred_offload_method",
    "offload_kind",
    "move_pipeline_to_device",
]


def preferred_offload_method(pipeline) -> str:
    """Pick the cheapest CPU offload strategy that fits the available VRAM.

    Model-level offload keeps one whole component on the GPU at a time, which
    still OOMs when a single component (e.g. a 14B Wan or 23 GiB bf16 Flux
    transformer) exceeds the free VRAM. In that case sequential (per-layer)
    offload is required — slower, but it runs within a few GiB of VRAM.

    Returns:
        "model" when every component fits the budget, "sequential" otherwise.
    """
    from nodetool.huggingface.local_provider_utils import (
        _VRAM_HEADROOM_GB,
        _cuda_free_vram_gb,
        _is_cuda_available,
        _pipeline_component_sizes_gb,
    )

    if not _is_cuda_available():
        return "model"

    sizes = _pipeline_component_sizes_gb(pipeline)
    budget = max(_cuda_free_vram_gb() - _VRAM_HEADROOM_GB, 0.0)
    if sizes and max(sizes) > budget:
        log.info(
            "Largest pipeline component (%.1f GiB) exceeds the %.1f GiB VRAM "
            "budget; preferring sequential CPU offload",
            max(sizes),
            budget,
        )
        return "sequential"
    return "model"


def _pipeline_modules(pipeline):
    """Yield the ``torch.nn.Module`` components of a pipeline.

    Prefers the diffusers ``components`` mapping and falls back to probing the
    usual attribute names, so non-diffusers pipelines (hy3dgen, nunchaku
    wrappers) are still inspected.
    """
    components = getattr(pipeline, "components", None)
    if isinstance(components, dict) and components:
        yield from (c for c in components.values() if c is not None)
        return

    for name in (
        "transformer",
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "image_encoder",
    ):
        component = getattr(pipeline, name, None)
        if component is not None:
            yield component


def offload_kind(pipeline) -> str | None:
    """Identify which CPU offload strategy is installed on ``pipeline``.

    diffusers uses two different accelerate hooks: ``enable_model_cpu_offload``
    installs :class:`accelerate.hooks.CpuOffload` on each component, while
    ``enable_sequential_cpu_offload`` installs per-submodule
    :class:`accelerate.hooks.AlignDevicesHook`. Detecting only the latter (as
    this helper originally did) reports "no offload" for model-level offload,
    which then lets a second strategy be stacked on top of the first.

    Returns:
        ``"model"``, ``"sequential"``, or ``None`` when no offload is installed.
    """
    # Prefer the live hooks over our own marker: diffusers' offload setters call
    # remove_all_hooks() first, so the hooks are the ground truth and a marker
    # left behind by an earlier strategy would be stale.
    try:
        from accelerate.hooks import AlignDevicesHook, CpuOffload
    except ImportError:
        AlignDevicesHook = CpuOffload = None  # type: ignore[assignment]

    if CpuOffload is not None:
        # Model-level offload hooks sit on the top-level components; sequential
        # offload dispatches per submodule, so recurse for the AlignDevicesHook.
        for component in _pipeline_modules(pipeline):
            if isinstance(getattr(component, "_hf_hook", None), CpuOffload):
                return "model"

        for component in _pipeline_modules(pipeline):
            if not callable(getattr(component, "modules", None)):
                continue
            try:
                for module in component.modules():
                    if isinstance(getattr(module, "_hf_hook", None), AlignDevicesHook):
                        return "sequential"
            except Exception:  # pragma: no cover - defensive, exotic components
                continue

    recorded = getattr(pipeline, "_nodetool_offload_kind", None)
    if recorded in ("model", "sequential"):
        return recorded

    return None


def has_cpu_offload_enabled(pipeline) -> bool:
    """
    Check if a pipeline already has CPU offload enabled via diffusers' methods.

    This avoids redundant calls to ``enable_*_cpu_offload()`` which tear down
    and rebuild every accelerate hook, and which can cause significant memory
    overhead (3GB+ RAM) when called repeatedly.

    Args:
        pipeline: A diffusers pipeline object.

    Returns:
        True if CPU offload appears to be already configured by diffusers.
    """
    if pipeline is None:
        return False
    if getattr(pipeline, "_nodetool_cpu_offload_applied", False):
        return True
    return offload_kind(pipeline) is not None


def apply_cpu_offload_if_needed(pipeline, method: str = "sequential") -> bool:
    """
    Apply CPU offload to a pipeline if not already configured.

    Args:
        pipeline: A diffusers pipeline object.
        method: "sequential" for enable_sequential_cpu_offload,
                "model" for enable_model_cpu_offload.

    Returns:
        True if offload was applied, False if already configured.
    """
    if pipeline is None:
        return False

    if method not in ("sequential", "model"):
        raise ValueError(f"Unknown offload method: {method}")

    existing = offload_kind(pipeline) or (
        "unknown" if getattr(pipeline, "_nodetool_cpu_offload_applied", False) else None
    )
    if existing is not None:
        # Exactly one strategy may be installed per pipeline; re-applying would
        # remove every hook and rebuild it, and stacking a different strategy on
        # top silently discards the first one's memory savings.
        log.debug(
            "Skipping %s CPU offload - %s offload already configured on pipeline",
            method,
            existing,
        )
        return False

    log_memory(f"Before {method}_cpu_offload")

    if method == "sequential":
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.enable_model_cpu_offload()

    pipeline._nodetool_cpu_offload_applied = True
    pipeline._nodetool_offload_kind = method

    log_memory(f"After {method}_cpu_offload")
    return True


def move_pipeline_to_device(pipeline, device: str | None) -> bool:
    """Move ``pipeline`` to ``device`` unless that would defeat CPU offload.

    ``pipeline.to("cuda")`` on an offloaded pipeline pulls every component onto
    the GPU at once: diffusers only logs a warning for model-level offload (and
    raises for sequential offload), so an automatically installed offload
    strategy would be silently undone and the full weights would become
    resident on a device already judged too small for them.

    Returns:
        True when the pipeline was moved, False when the move was skipped.
    """
    if pipeline is None or not device:
        return False

    move_fn = getattr(pipeline, "to", None)
    if not callable(move_fn):
        return False

    if str(device) != "cpu":
        kind = offload_kind(pipeline)
        if kind is not None:
            log.debug(
                "Skipping move to %s - %s CPU offload places components on demand",
                device,
                kind,
            )
            return False

    move_fn(device)
    return True
