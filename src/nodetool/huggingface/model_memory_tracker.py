"""
Model Memory Tracker for HuggingFace Models

This module provides accurate per-model memory tracking to power the ModelRegistry
and Memory API. It tracks memory at all critical lifecycle points:
- Load: Initial model loading
- Device placement: When model.to(device) is called
- Warmup: First inference to capture lazy allocations
- Peak usage: Maximum memory during inference

The tracker handles special cases:
- Shared components (VAEs, text encoders, tokenizers)
- CPU offload (Accelerate hooks, Nunchaku)
- Diffusion pipelines with multiple sub-models
- MPS (Apple Silicon) devices
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from nodetool.config.logging_config import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Environment variables for configuration
SKIP_WARMUP_ENV = "NODETOOL_SKIP_WARMUP"
SMOKE_TEST_ENV = "NODETOOL_SMOKE_TEST"
WARMUP_TIMEOUT = 5.0  # seconds


@dataclass
class MemoryStats:
    """Memory statistics for a loaded model."""

    # Total memory usage in MB
    total_mb: float

    # Peak memory usage in MB (with safety margin)
    peak_mb: float

    # VRAM usage in MB (GPU memory)
    vram_mb: float

    # System RAM usage in MB
    ram_mb: float

    # Offload status
    offload_status: Literal[
        "full_gpu", "cpu_offload", "sequential_offload", "cpu_only", "mps"
    ]

    # Shared components (e.g., ["vae", "text_encoder"])
    shared_components: list[str]

    # Additional metadata (e.g., quantization info)
    metadata: dict[str, Any]

    # Breakdown by component for pipelines
    component_memory: dict[str, float]


class ModelMemoryTracker:
    """
    Tracks memory usage for HuggingFace models throughout their lifecycle.

    Usage:
        tracker = ModelMemoryTracker(node_id, model_id, device)
        tracker.start()
        model = load_model(...)
        tracker.after_load()
        model = model.to(device)
        tracker.after_device_placement()
        await tracker.warmup_inference(model, model_class)
        stats = tracker.finalize()
    """

    def __init__(
        self,
        node_id: str,
        model_id: str,
        device: str | None = None,
    ):
        """
        Initialize memory tracker.

        Args:
            node_id: Unique identifier for the node loading the model
            model_id: Model identifier (repo_id or path)
            device: Target device (cuda, mps, cpu)
        """
        self.node_id = node_id
        self.model_id = model_id
        self.device = device or "cpu"

        # Memory measurements at different lifecycle points
        self._baseline_vram: float = 0.0
        self._baseline_ram: float = 0.0
        self._after_load_vram: float = 0.0
        self._after_load_ram: float = 0.0
        self._after_device_vram: float = 0.0
        self._after_device_ram: float = 0.0
        self._peak_vram: float = 0.0
        self._peak_ram: float = 0.0

        # Component tracking for pipelines
        self._component_memory: dict[str, float] = {}

        # Shared components detected
        self._shared_components: list[str] = []

        # Additional metadata
        self._metadata: dict[str, Any] = {}

        # Timing
        self._start_time: float = 0.0

    def start(self) -> None:
        """Initialize baseline memory measurements."""
        self._start_time = time.time()
        self._baseline_vram = self._get_vram_mb()
        self._baseline_ram = self._get_ram_mb()

        log.debug(
            f"[{self.node_id}] Memory baseline: VRAM={self._baseline_vram:.1f}MB, RAM={self._baseline_ram:.1f}MB"
        )

    def after_load(self) -> None:
        """Record memory after model loading."""
        self._after_load_vram = self._get_vram_mb()
        self._after_load_ram = self._get_ram_mb()

        vram_delta = self._after_load_vram - self._baseline_vram
        ram_delta = self._after_load_ram - self._baseline_ram

        log.info(
            f"[{self.node_id}] After load: +{vram_delta:.1f}MB VRAM, +{ram_delta:.1f}MB RAM"
        )

    def after_device_placement(self) -> None:
        """Record memory after moving model to device."""
        self._after_device_vram = self._get_vram_mb()
        self._after_device_ram = self._get_ram_mb()

        vram_delta = self._after_device_vram - self._after_load_vram
        ram_delta = self._after_device_ram - self._after_load_ram

        log.info(
            f"[{self.node_id}] After device placement: +{vram_delta:.1f}MB VRAM, +{ram_delta:.1f}MB RAM"
        )

    async def warmup_inference(
        self, model: Any, model_class: type, timeout: float = WARMUP_TIMEOUT
    ) -> None:
        """
        Run a lightweight warmup inference to capture lazy allocations.

        Args:
            model: The loaded model or pipeline
            model_class: The model class type
            timeout: Maximum time to spend on warmup (seconds)
        """
        # Skip warmup if configured
        if self._should_skip_warmup():
            log.debug(f"[{self.node_id}] Skipping warmup inference")
            return

        log.debug(f"[{self.node_id}] Running warmup inference...")

        try:
            # Run warmup with timeout
            await asyncio.wait_for(
                self._run_warmup(model, model_class), timeout=timeout
            )

            # Record peak memory after warmup
            self._peak_vram = self._get_peak_vram_mb()
            self._peak_ram = max(self._get_ram_mb(), self._peak_ram)

            log.info(
                f"[{self.node_id}] Warmup complete: peak VRAM={self._peak_vram:.1f}MB"
            )

        except asyncio.TimeoutError:
            log.warning(f"[{self.node_id}] Warmup inference timed out after {timeout}s")
        except Exception as exc:
            log.warning(
                f"[{self.node_id}] Warmup inference failed: {exc}", exc_info=True
            )

    async def _run_warmup(self, model: Any, model_class: type) -> None:
        """Execute model-specific warmup inference."""

        def _warmup():
            # Import torch inside the thread
            try:
                import torch
            except ImportError:
                log.warning("torch not available, skipping warmup")
                return

            # Detect model type and run appropriate warmup
            if hasattr(model, "__call__"):
                # Try to detect if it's a pipeline or transformers model
                model_name = model_class.__name__.lower()

                if "pipeline" in model_name or hasattr(model, "components"):
                    self._warmup_pipeline(model, torch)
                elif "diffusion" in model_name:
                    self._warmup_diffusion(model, torch)
                elif any(
                    x in model_name for x in ["llm", "gpt", "bert", "t5", "llama"]
                ):
                    self._warmup_text_model(model, torch)
                elif "whisper" in model_name:
                    self._warmup_audio_model(model, torch)
                else:
                    log.debug(
                        f"[{self.node_id}] No specific warmup for {model_class.__name__}"
                    )

        # Run in thread to avoid blocking
        await asyncio.to_thread(_warmup)

    def _warmup_pipeline(self, pipeline: Any, torch: Any) -> None:
        """Warmup for diffusion pipelines."""
        try:
            with torch.inference_mode():
                # Minimal image generation for diffusion pipelines
                if hasattr(pipeline, "transformer") or hasattr(pipeline, "unet"):
                    pipeline(
                        prompt="test",
                        num_inference_steps=1,
                        height=64,
                        width=64,
                        output_type="latent",
                    )
                # Other pipelines - just call with minimal input
                else:
                    pipeline("test")
        except Exception as exc:
            log.debug(f"[{self.node_id}] Pipeline warmup error: {exc}")

    def _warmup_diffusion(self, model: Any, torch: Any) -> None:
        """Warmup for standalone diffusion models."""
        try:
            with torch.inference_mode():
                # Create minimal latent tensor
                if hasattr(model, "forward"):
                    # Try to call forward with minimal input
                    dummy_input = torch.randn(
                        1, 4, 8, 8, device=self.device
                    )  # Minimal latent
                    model(dummy_input, timestep=torch.tensor([1]))
        except Exception as exc:
            log.debug(f"[{self.node_id}] Diffusion warmup error: {exc}")

    def _warmup_text_model(self, model: Any, torch: Any) -> None:
        """Warmup for text generation models."""
        try:
            with torch.inference_mode():
                # Minimal text input
                if hasattr(model, "generate"):
                    # Try tokenizer if available
                    input_ids = torch.tensor([[1]], device=self.device)  # Minimal token
                    model.generate(input_ids, max_new_tokens=1)
                elif hasattr(model, "__call__"):
                    model("test", max_new_tokens=1)
        except Exception as exc:
            log.debug(f"[{self.node_id}] Text model warmup error: {exc}")

    def _warmup_audio_model(self, model: Any, torch: Any) -> None:
        """Warmup for audio models."""
        try:
            with torch.inference_mode():
                # Minimal audio input
                if hasattr(model, "__call__"):
                    # Create minimal audio tensor
                    dummy_audio = torch.randn(1, 16000, device=self.device)  # 1 second
                    model(dummy_audio)
        except Exception as exc:
            log.debug(f"[{self.node_id}] Audio model warmup error: {exc}")

    def detect_offload_status(self, model: Any) -> str:
        """
        Detect CPU offload configuration.

        Returns:
            Offload status: "full_gpu", "cpu_offload", "sequential_offload", "cpu_only", "mps"
        """
        # Check for MPS device
        if self.device == "mps":
            return "mps"

        # Check for CPU-only
        if self.device == "cpu":
            return "cpu_only"

        # Check for Accelerate hooks
        try:
            from accelerate.hooks import AlignDevicesHook

            # Check pipeline components
            if hasattr(model, "components"):
                for name, component in model.components.items():
                    if component is not None and hasattr(component, "_hf_hook"):
                        if isinstance(component._hf_hook, AlignDevicesHook):
                            log.debug(
                                f"[{self.node_id}] Detected Accelerate offload on {name}"
                            )
                            return "sequential_offload"

            # Check model directly
            if hasattr(model, "_hf_hook") and isinstance(
                model._hf_hook, AlignDevicesHook
            ):
                return "cpu_offload"

        except ImportError:
            pass

        # Check for Nunchaku offload
        if self._detect_nunchaku_offload(model):
            return "cpu_offload"

        # Check for custom offload flag
        if getattr(model, "_nodetool_cpu_offload_applied", False):
            return "sequential_offload"

        return "full_gpu"

    def _detect_nunchaku_offload(self, model: Any) -> bool:
        """Detect Nunchaku-specific CPU offload patterns."""
        try:
            # Check if model has Nunchaku transformer
            if hasattr(model, "transformer"):
                transformer = model.transformer
                # Nunchaku transformers have rank_begin and rank_end attributes
                if hasattr(transformer, "rank_begin") and hasattr(
                    transformer, "rank_end"
                ):
                    rank_begin = getattr(transformer, "rank_begin", 0)
                    rank_end = getattr(transformer, "rank_end", 0)

                    if rank_begin > 0 or rank_end > 0:
                        log.debug(
                            f"[{self.node_id}] Detected Nunchaku offload: layers {rank_begin}-{rank_end}"
                        )
                        self._metadata["quantization"] = "FP4"
                        self._metadata["layers_on_gpu"] = rank_end - rank_begin
                        return True
        except Exception:
            pass

        return False

    def detect_shared_components(self, model: Any) -> list[str]:
        """
        Detect shared components that might be cached.

        Args:
            model: The loaded model or pipeline

        Returns:
            List of shared component names
        """
        shared = []

        try:
            from nodetool.ml.core.model_manager import ModelManager

            # Check pipeline components
            if hasattr(model, "components"):
                for name, component in model.components.items():
                    if component is not None:
                        # Check if component is already cached
                        component_id = f"{self.model_id}_{name}"
                        existing = ModelManager.get_model(component_id)
                        if existing is not None and existing is component:
                            shared.append(name)
                            log.debug(
                                f"[{self.node_id}] Detected shared component: {name}"
                            )

        except Exception as exc:
            log.debug(f"[{self.node_id}] Error detecting shared components: {exc}")

        self._shared_components = shared
        return shared

    def track_component_memory(self, model: Any) -> dict[str, float]:
        """
        Track memory usage by component for pipelines.

        Args:
            model: The loaded pipeline

        Returns:
            Dictionary mapping component name to memory in MB
        """
        component_memory = {}

        try:
            # Only for pipelines with components
            if not hasattr(model, "components"):
                return component_memory

            for name, component in model.components.items():
                if component is not None:
                    # Estimate component memory
                    mem_mb = self._estimate_component_memory(component)
                    if mem_mb > 0:
                        component_memory[name] = mem_mb
                        log.debug(f"[{self.node_id}] Component {name}: {mem_mb:.1f}MB")

        except Exception as exc:
            log.debug(f"[{self.node_id}] Error tracking component memory: {exc}")

        self._component_memory = component_memory
        return component_memory

    def _estimate_component_memory(self, component: Any) -> float:
        """Estimate memory usage of a single component."""
        try:
            import torch

            if isinstance(component, torch.nn.Module):
                # Count parameters
                total_params = sum(p.numel() for p in component.parameters())

                # Estimate bytes per parameter (depends on dtype)
                bytes_per_param = 4  # float32 default

                if hasattr(component, "dtype"):
                    dtype = component.dtype
                    if dtype == torch.float16 or dtype == torch.bfloat16:
                        bytes_per_param = 2
                    elif dtype == torch.float64:
                        bytes_per_param = 8

                # Convert to MB
                mem_mb = (total_params * bytes_per_param) / (1024 * 1024)
                return mem_mb

        except Exception:
            pass

        return 0.0

    def finalize(self) -> MemoryStats:
        """
        Finalize tracking and return memory statistics.

        Returns:
            MemoryStats object with all measurements
        """
        # Calculate total memory usage
        vram_mb = max(
            self._after_device_vram - self._baseline_vram,
            self._peak_vram - self._baseline_vram,
        )
        ram_mb = max(
            self._after_device_ram - self._baseline_ram,
            self._peak_ram - self._baseline_ram,
        )

        # Apply safety margin (1.2x)
        SAFETY_MARGIN = 1.2
        total_mb = (vram_mb + ram_mb) * SAFETY_MARGIN
        peak_mb = max(vram_mb, ram_mb) * SAFETY_MARGIN

        # Determine offload status
        offload_status = "full_gpu"  # Will be set by caller if needed

        elapsed = time.time() - self._start_time
        log.info(
            f"[{self.node_id}] Memory tracking complete in {elapsed:.2f}s: "
            f"total={total_mb:.1f}MB, peak={peak_mb:.1f}MB, vram={vram_mb:.1f}MB, ram={ram_mb:.1f}MB"
        )

        return MemoryStats(
            total_mb=total_mb,
            peak_mb=peak_mb,
            vram_mb=vram_mb * SAFETY_MARGIN,
            ram_mb=ram_mb * SAFETY_MARGIN,
            offload_status=offload_status,
            shared_components=self._shared_components,
            metadata=self._metadata,
            component_memory=self._component_memory,
        )

    def _get_vram_mb(self) -> float:
        """Get current VRAM usage in MB."""
        try:
            import torch

            if self.device.startswith("cuda"):
                # Synchronize to ensure all operations are complete
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                return allocated / (1024 * 1024)
            elif self.device == "mps":
                # MPS memory tracking if available
                if hasattr(torch.mps, "current_allocated_memory"):
                    allocated = torch.mps.current_allocated_memory()
                    return allocated / (1024 * 1024)
        except Exception as exc:
            log.debug(f"[{self.node_id}] Error getting VRAM: {exc}")

        return 0.0

    def _get_peak_vram_mb(self) -> float:
        """Get peak VRAM usage in MB."""
        try:
            import torch

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                return peak / (1024 * 1024)
            elif self.device == "mps":
                # MPS doesn't track peak, return current
                if hasattr(torch.mps, "current_allocated_memory"):
                    allocated = torch.mps.current_allocated_memory()
                    return allocated / (1024 * 1024)
        except Exception as exc:
            log.debug(f"[{self.node_id}] Error getting peak VRAM: {exc}")

        return 0.0

    def _get_ram_mb(self) -> float:
        """Get current system RAM usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except Exception as exc:
            log.debug(f"[{self.node_id}] Error getting RAM: {exc}")

        return 0.0

    def _should_skip_warmup(self) -> bool:
        """Check if warmup should be skipped based on environment."""
        # Skip in smoke test mode
        if os.getenv(SMOKE_TEST_ENV, "").lower() in ("1", "true", "yes", "on"):
            return True

        # Skip if explicitly disabled
        if os.getenv(SKIP_WARMUP_ENV, "").lower() in ("1", "true", "yes", "on"):
            return True

        return False
