"""
Memory logging and garbage collection utilities for HuggingFace pipelines.

This module provides utilities for tracking memory usage and performing
garbage collection to help diagnose memory leaks and reduce RAM usage.
"""

from __future__ import annotations

import gc
import psutil
import os
from typing import TYPE_CHECKING

from nodetool.config.logging_config import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_usage_mb() -> tuple[float, float] | None:
    """
    Get current GPU memory usage in MB.

    Returns:
        Tuple of (allocated_mb, reserved_mb) or None if CUDA unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return (allocated, reserved)
    except ImportError:
        pass
    return None


def log_memory(label: str, include_gpu: bool = True) -> None:
    """
    Log current memory usage with a label.

    Args:
        label: A descriptive label for this memory checkpoint.
        include_gpu: Whether to also log GPU memory usage.
    """
    ram_mb = get_memory_usage_mb()
    log.info(f"[MEMORY] {label}: RAM={ram_mb:.1f}MB")

    if include_gpu:
        gpu_mem = get_gpu_memory_usage_mb()
        if gpu_mem:
            allocated, reserved = gpu_mem
            log.info(f"[MEMORY] {label}: GPU allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")


def run_gc(label: str = "", log_before_after: bool = True) -> float:
    """
    Run garbage collection and optionally log memory before and after.

    Args:
        label: A descriptive label for this GC run.
        log_before_after: Whether to log memory usage before and after GC.

    Returns:
        Memory freed in MB (RAM only, approximate).
    """
    if log_before_after:
        before_mb = get_memory_usage_mb()
        log.info(f"[GC] {label} - Before GC: RAM={before_mb:.1f}MB")
    else:
        before_mb = get_memory_usage_mb()

    # Run full garbage collection (all generations)
    gc.collect()

    # Also clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    after_mb = get_memory_usage_mb()
    freed_mb = before_mb - after_mb

    if log_before_after:
        log.info(f"[GC] {label} - After GC: RAM={after_mb:.1f}MB (freed {freed_mb:.1f}MB)")
        gpu_mem = get_gpu_memory_usage_mb()
        if gpu_mem:
            allocated, reserved = gpu_mem
            log.info(f"[GC] {label} - GPU after: allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")

    return freed_mb


def log_memory_summary(label: str = "Summary") -> dict:
    """
    Log a comprehensive memory summary.

    Returns:
        Dictionary with memory stats.
    """
    stats = {
        "ram_mb": get_memory_usage_mb(),
    }

    gpu_mem = get_gpu_memory_usage_mb()
    if gpu_mem:
        stats["gpu_allocated_mb"] = gpu_mem[0]
        stats["gpu_reserved_mb"] = gpu_mem[1]

    # Log Python object counts for debugging
    gc.collect()
    obj_counts = {}
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        obj_counts[obj_type] = obj_counts.get(obj_type, 0) + 1

    # Get top 10 object types by count
    top_objects = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    stats["top_objects"] = top_objects

    log.info(f"[MEMORY SUMMARY] {label}")
    log.info(f"  RAM: {stats['ram_mb']:.1f}MB")
    if gpu_mem:
        log.info(f"  GPU allocated: {stats.get('gpu_allocated_mb', 0):.1f}MB")
        log.info(f"  GPU reserved: {stats.get('gpu_reserved_mb', 0):.1f}MB")
    log.info(f"  Top objects: {top_objects[:5]}")

    return stats


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
    # Check for the marker attribute that we set after applying CPU offload
    if getattr(pipeline, '_nodetool_cpu_offload_applied', False):
        return True

    # Check for accelerate AlignDevicesHook which is specifically set by
    # enable_sequential_cpu_offload / enable_model_cpu_offload
    # This is more reliable than checking _all_hooks which can have other hooks
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

    # Mark that we've applied CPU offload to avoid redundant calls
    pipeline._nodetool_cpu_offload_applied = True

    log_memory(f"After {method}_cpu_offload")
    return True


class MemoryTracker:
    """
    Context manager for tracking memory usage during a block of code.

    Usage:
        with MemoryTracker("Loading model"):
            # ... load model code ...
    """

    def __init__(self, label: str, run_gc_after: bool = True):
        self.label = label
        self.run_gc_after = run_gc_after
        self.start_ram_mb = 0.0
        self.start_gpu: tuple[float, float] | None = None

    def __enter__(self):
        self.start_ram_mb = get_memory_usage_mb()
        self.start_gpu = get_gpu_memory_usage_mb()
        log.info(f"[MEMORY TRACK] {self.label} - START: RAM={self.start_ram_mb:.1f}MB")
        if self.start_gpu:
            log.info(f"[MEMORY TRACK] {self.label} - START: GPU allocated={self.start_gpu[0]:.1f}MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_ram_mb = get_memory_usage_mb()
        end_gpu = get_gpu_memory_usage_mb()

        ram_delta = end_ram_mb - self.start_ram_mb
        log.info(f"[MEMORY TRACK] {self.label} - END: RAM={end_ram_mb:.1f}MB (delta: {ram_delta:+.1f}MB)")

        if end_gpu and self.start_gpu:
            gpu_delta = end_gpu[0] - self.start_gpu[0]
            log.info(f"[MEMORY TRACK] {self.label} - END: GPU allocated={end_gpu[0]:.1f}MB (delta: {gpu_delta:+.1f}MB)")

        if self.run_gc_after:
            run_gc(f"{self.label} cleanup", log_before_after=True)

        return False
