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
]


def has_cpu_offload_enabled(pipeline) -> bool:
    """
    Check if a pipeline already has CPU offload enabled via diffusers' methods.

    This avoids redundant calls to enable_sequential_cpu_offload() which can
    cause significant memory overhead (3GB+ RAM) when called repeatedly.

    Args:
        pipeline: A diffusers pipeline object.

    Returns:
        True if CPU offload appears to be already configured by diffusers.
    """
    if getattr(pipeline, '_nodetool_cpu_offload_applied', False):
        return True

    try:
        from accelerate.hooks import AlignDevicesHook
        for name in ['transformer', 'unet', 'vae', 'text_encoder', 'text_encoder_2']:
            component = getattr(pipeline, name, None)
            if component is not None and hasattr(component, '_hf_hook'):
                if isinstance(component._hf_hook, AlignDevicesHook):
                    return True
    except ImportError:
        pass

    return False


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
    if has_cpu_offload_enabled(pipeline):
        log.debug("Skipping CPU offload - already configured on pipeline")
        return False

    log_memory(f"Before {method}_cpu_offload")

    if method == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif method == "model":
        pipeline.enable_model_cpu_offload()
    else:
        raise ValueError(f"Unknown offload method: {method}")

    pipeline._nodetool_cpu_offload_applied = True

    log_memory(f"After {method}_cpu_offload")
    return True
