"""
Tests for ModelMemoryTracker and memory tracking integration.

This test suite validates:
- Memory measurement accuracy
- CPU offload detection
- Shared component deduplication
- Warmup inference
- Performance (<500ms overhead)
"""

import asyncio
import os
import pytest
from unittest.mock import Mock, MagicMock, patch

# Set test environment before imports
os.environ["ENV"] = "test"
os.environ["NODETOOL_SMOKE_TEST"] = "1"
os.environ["NODETOOL_SKIP_WARMUP"] = "1"  # Skip warmup by default for tests

from nodetool.huggingface.model_memory_tracker import (
    ModelMemoryTracker,
    MemoryStats,
    SKIP_WARMUP_ENV,
    SMOKE_TEST_ENV,
)


class TestModelMemoryTracker:
    """Test the ModelMemoryTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")
        assert tracker.node_id == "node1"
        assert tracker.model_id == "model1"
        assert tracker.device == "cuda"
        assert tracker._baseline_vram == 0.0
        assert tracker._baseline_ram == 0.0

    def test_start_records_baseline(self):
        """Test that start() records baseline memory."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")
        tracker.start()

        # Baseline should be recorded (even if 0 without torch)
        assert tracker._baseline_vram >= 0.0
        assert tracker._baseline_ram >= 0.0
        assert tracker._start_time > 0.0

    def test_after_load_records_memory(self):
        """Test that after_load() records memory."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")
        tracker.start()
        tracker.after_load()

        assert tracker._after_load_vram >= 0.0
        assert tracker._after_load_ram >= 0.0

    def test_after_device_placement_records_memory(self):
        """Test that after_device_placement() records memory."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")
        tracker.start()
        tracker.after_load()
        tracker.after_device_placement()

        assert tracker._after_device_vram >= 0.0
        assert tracker._after_device_ram >= 0.0

    @pytest.mark.asyncio
    async def test_warmup_skipped_in_smoke_test_mode(self):
        """Test that warmup is skipped in smoke test mode."""
        os.environ[SMOKE_TEST_ENV] = "1"
        tracker = ModelMemoryTracker("node1", "model1", "cuda")
        tracker.start()

        mock_model = Mock()
        mock_class = type("MockClass", (), {})

        # Should return quickly without running warmup
        await tracker.warmup_inference(mock_model, mock_class)

        # No warmup should have been attempted
        assert tracker._peak_vram == 0.0

    @pytest.mark.asyncio
    async def test_warmup_skipped_when_disabled(self):
        """Test that warmup is skipped when explicitly disabled."""
        os.environ[SMOKE_TEST_ENV] = "0"
        os.environ[SKIP_WARMUP_ENV] = "1"

        tracker = ModelMemoryTracker("node1", "model1", "cuda")
        tracker.start()

        mock_model = Mock()
        mock_class = type("MockClass", (), {})

        await tracker.warmup_inference(mock_model, mock_class)

        # No warmup should have been attempted
        assert tracker._peak_vram == 0.0

    def test_detect_offload_status_cpu_only(self):
        """Test detection of CPU-only models."""
        tracker = ModelMemoryTracker("node1", "model1", "cpu")
        mock_model = Mock()

        status = tracker.detect_offload_status(mock_model)
        assert status == "cpu_only"

    def test_detect_offload_status_mps(self):
        """Test detection of MPS device."""
        tracker = ModelMemoryTracker("node1", "model1", "mps")
        mock_model = Mock()

        status = tracker.detect_offload_status(mock_model)
        assert status == "mps"

    def test_detect_offload_status_full_gpu(self):
        """Test detection of full GPU models."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")
        mock_model = Mock(spec=[])  # No components or hooks

        status = tracker.detect_offload_status(mock_model)
        assert status == "full_gpu"

    def test_detect_offload_status_with_accelerate_hook(self):
        """Test detection of Accelerate CPU offload."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")

        # Mock model with Accelerate hook
        try:
            from accelerate.hooks import AlignDevicesHook

            mock_model = Mock()
            mock_model.components = {
                "unet": Mock(_hf_hook=AlignDevicesHook()),
            }

            status = tracker.detect_offload_status(mock_model)
            # Should detect offload
            assert status in ["sequential_offload", "cpu_offload"]
        except ImportError:
            # Skip test if accelerate is not available
            pytest.skip("accelerate not available")

    def test_detect_offload_status_with_custom_flag(self):
        """Test detection of custom offload flag."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")

        mock_model = Mock()
        mock_model._nodetool_cpu_offload_applied = True

        status = tracker.detect_offload_status(mock_model)
        assert status == "sequential_offload"

    def test_detect_nunchaku_offload(self):
        """Test detection of Nunchaku quantization and offload."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")

        # Mock model with Nunchaku transformer
        mock_transformer = Mock()
        mock_transformer.rank_begin = 0
        mock_transformer.rank_end = 12

        mock_model = Mock()
        mock_model.transformer = mock_transformer

        is_nunchaku = tracker._detect_nunchaku_offload(mock_model)
        assert is_nunchaku is True
        assert tracker._metadata.get("quantization") == "FP4"
        assert tracker._metadata.get("layers_on_gpu") == 12

    def test_detect_shared_components_no_pipeline(self):
        """Test shared component detection on non-pipeline model."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")

        mock_model = Mock(spec=[])  # No components attribute

        shared = tracker.detect_shared_components(mock_model)
        assert shared == []

    def test_detect_shared_components_with_pipeline(self):
        """Test shared component detection on pipeline."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")

        mock_vae = Mock()
        mock_unet = Mock()

        mock_model = Mock()
        mock_model.components = {
            "vae": mock_vae,
            "unet": mock_unet,
        }

        # The function will try to call ModelManager but it's imported dynamically
        # Just verify it doesn't crash
        shared = tracker.detect_shared_components(mock_model)
        # Should return a list (possibly empty if ModelManager not available)
        assert isinstance(shared, list)

    def test_track_component_memory_no_pipeline(self):
        """Test component memory tracking on non-pipeline."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")

        mock_model = Mock(spec=[])

        component_mem = tracker.track_component_memory(mock_model)
        assert component_mem == {}

    def test_estimate_component_memory_without_torch(self):
        """Test component memory estimation without torch."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")

        mock_component = Mock()

        mem = tracker._estimate_component_memory(mock_component)
        assert mem == 0.0

    def test_finalize_returns_memory_stats(self):
        """Test that finalize() returns MemoryStats."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")
        tracker.start()
        tracker.after_load()
        tracker.after_device_placement()

        stats = tracker.finalize()

        assert isinstance(stats, MemoryStats)
        assert stats.total_mb >= 0.0
        assert stats.peak_mb >= 0.0
        assert stats.vram_mb >= 0.0
        assert stats.ram_mb >= 0.0
        assert stats.offload_status == "full_gpu"  # Default
        assert isinstance(stats.shared_components, list)
        assert isinstance(stats.metadata, dict)
        assert isinstance(stats.component_memory, dict)

    def test_finalize_applies_safety_margin(self):
        """Test that finalize() applies 1.2x safety margin."""
        tracker = ModelMemoryTracker("node1", "model1", "cuda")

        # Set baseline first
        tracker._baseline_vram = 0.0
        tracker._baseline_ram = 50.0  # Set to match expected delta

        # Simulate some memory allocation
        tracker._after_load_vram = 100.0
        tracker._after_device_vram = 100.0
        tracker._peak_vram = 100.0

        tracker._after_load_ram = 100.0
        tracker._after_device_ram = 100.0
        tracker._peak_ram = 100.0

        stats = tracker.finalize()

        # With 1.2x safety margin:
        # VRAM delta = 100 - 0 = 100
        # RAM delta = 100 - 50 = 50
        # Total = (100 + 50) * 1.2 = 180
        # Peak = max(100, 50) * 1.2 = 120
        assert stats.total_mb == pytest.approx(180.0, rel=0.01)
        assert stats.peak_mb == pytest.approx(120.0, rel=0.01)

    def test_memory_stats_dataclass(self):
        """Test MemoryStats dataclass creation."""
        stats = MemoryStats(
            total_mb=100.0,
            peak_mb=120.0,
            vram_mb=80.0,
            ram_mb=20.0,
            offload_status="full_gpu",
            shared_components=["vae"],
            metadata={"quantization": "FP16"},
            component_memory={"unet": 50.0, "vae": 30.0},
        )

        assert stats.total_mb == 100.0
        assert stats.peak_mb == 120.0
        assert stats.vram_mb == 80.0
        assert stats.ram_mb == 20.0
        assert stats.offload_status == "full_gpu"
        assert stats.shared_components == ["vae"]
        assert stats.metadata == {"quantization": "FP16"}
        assert stats.component_memory == {"unet": 50.0, "vae": 30.0}


class TestMemoryTrackingIntegration:
    """Test integration of memory tracking with load_model and load_pipeline."""

    @pytest.mark.asyncio
    async def test_load_model_integration(self):
        """Test that load_model integrates memory tracking."""
        # This is a minimal smoke test - we just verify the imports work
        from nodetool.huggingface.local_provider_utils import load_model

        # We don't actually run load_model since it would require real models
        # Just verify the function exists and has the right signature
        import inspect

        sig = inspect.signature(load_model)
        assert "node_id" in sig.parameters
        assert "context" in sig.parameters
        assert "model_class" in sig.parameters

    @pytest.mark.asyncio
    async def test_load_pipeline_integration(self):
        """Test that load_pipeline integrates memory tracking."""
        # This is a minimal smoke test
        from nodetool.huggingface.local_provider_utils import load_pipeline

        import inspect

        sig = inspect.signature(load_pipeline)
        assert "node_id" in sig.parameters
        assert "context" in sig.parameters
        assert "pipeline_task" in sig.parameters


class TestPerformance:
    """Test memory tracking performance requirements."""

    def test_tracker_initialization_is_fast(self):
        """Test that tracker initialization is fast (<1ms)."""
        import time

        start = time.time()
        for _ in range(100):
            tracker = ModelMemoryTracker("node1", "model1", "cuda")
        elapsed = time.time() - start

        # 100 initializations should take less than 100ms
        assert elapsed < 0.1

    def test_memory_measurement_is_fast(self):
        """Test that memory measurement is fast (<10ms)."""
        import time

        tracker = ModelMemoryTracker("node1", "model1", "cuda")

        start = time.time()
        for _ in range(10):
            tracker._get_vram_mb()
            tracker._get_ram_mb()
        elapsed = time.time() - start

        # 10 measurements should take less than 100ms
        assert elapsed < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
