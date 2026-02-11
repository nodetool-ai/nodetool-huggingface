"""
Memory utilities for HuggingFace pipelines.

This module provides HuggingFace-specific memory utilities for tracking and
managing memory usage during model loading and inference.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from nodetool.config.logging_config import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


def get_memory_usage_mb() -> float:
    """
    Get current system RAM usage in MB.

    Returns:
        Current RAM usage in MB
    """
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)
    except Exception as exc:
        log.debug(f"Error getting RAM usage: {exc}")
        return 0.0


def get_gpu_memory_usage_mb(device: str = "cuda") -> float:
    """
    Get current GPU memory usage in MB.

    Args:
        device: Device to check (cuda, mps)

    Returns:
        Current GPU memory usage in MB
    """
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            return allocated / (1024 * 1024)
        elif device == "mps" and hasattr(torch, "mps"):
            if hasattr(torch.mps, "current_allocated_memory"):
                allocated = torch.mps.current_allocated_memory()
                return allocated / (1024 * 1024)
    except Exception as exc:
        log.debug(f"Error getting GPU memory usage: {exc}")

    return 0.0


def log_memory(context: str = "") -> None:
    """
    Log current memory usage.

    Args:
        context: Context string to include in log message
    """
    ram_mb = get_memory_usage_mb()
    gpu_mb = get_gpu_memory_usage_mb()

    if context:
        log.info(f"[{context}] Memory: RAM={ram_mb:.1f}MB, GPU={gpu_mb:.1f}MB")
    else:
        log.info(f"Memory: RAM={ram_mb:.1f}MB, GPU={gpu_mb:.1f}MB")


def run_gc(label: str = "", log_before_after: bool = False) -> float:
    """
    Run garbage collection and clear GPU cache if available.

    Args:
        label: Optional descriptive label for logging
        log_before_after: Whether to log memory usage before and after GC

    Returns:
        Memory freed in MB (RAM only, approximate)
    """
    if log_before_after:
        before_mb = get_memory_usage_mb()
        log.info(f"[GC] {label} - Before GC: RAM={before_mb:.1f}MB")
    else:
        before_mb = get_memory_usage_mb()
        if label:
            log.debug(f"[GC] {label}")

    gc.collect()

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
        gpu_mb = get_gpu_memory_usage_mb()
        if gpu_mb > 0:
            log.info(f"[GC] {label} - GPU after: {gpu_mb:.1f}MB")

    return freed_mb


def log_memory_summary(context: str = "") -> None:
    """
    Log a memory summary including garbage collection stats.

    Args:
        context: Context string to include in log message
    """
    log_memory(context)
    gc_stats = gc.get_stats()
    if gc_stats:
        log.debug(f"GC stats: {gc_stats}")


class MemoryTracker:
    """
    Context manager for tracking memory usage during a block of code.

    Usage:
        with MemoryTracker("Loading model"):
            # ... load model code ...

        # Or without context manager:
        tracker = MemoryTracker("loading model")
        # ... do work ...
        tracker.log_delta()
    """

    def __init__(self, context: str = "", run_gc_after: bool = True):
        """
        Initialize memory tracker.

        Args:
            context: Context string for logging
            run_gc_after: Whether to run garbage collection after the block
        """
        self.context = context
        self.run_gc_after = run_gc_after
        self.start_ram = 0.0
        self.start_gpu = 0.0

    def __enter__(self):
        """Start tracking memory."""
        self.start_ram = get_memory_usage_mb()
        self.start_gpu = get_gpu_memory_usage_mb()
        log.info(f"[MEMORY TRACK] {self.context} - START: RAM={self.start_ram:.1f}MB, GPU={self.start_gpu:.1f}MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log memory delta and optionally run GC."""
        self.log_delta()
        if self.run_gc_after:
            run_gc(f"{self.context} cleanup")
        return False

    def log_delta(self) -> None:
        """Log memory delta since initialization."""
        current_ram = get_memory_usage_mb()
        current_gpu = get_gpu_memory_usage_mb()

        delta_ram = current_ram - self.start_ram
        delta_gpu = current_gpu - self.start_gpu

        if self.context:
            log.info(
                f"[MEMORY TRACK] {self.context} - END: RAM={current_ram:.1f}MB (delta: {delta_ram:+.1f}MB), GPU={current_gpu:.1f}MB (delta: {delta_gpu:+.1f}MB)"
            )
        else:
            log.info(f"Memory delta: RAM +{delta_ram:.1f}MB, GPU +{delta_gpu:.1f}MB")


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
    if getattr(pipeline, "_nodetool_cpu_offload_applied", False):
        return True

    try:
        from accelerate.hooks import AlignDevicesHook

        for name in ["transformer", "unet", "vae", "text_encoder", "text_encoder_2"]:
            component = getattr(pipeline, name, None)
            if component is not None and hasattr(component, "_hf_hook"):
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
