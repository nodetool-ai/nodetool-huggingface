"""
Tests for runtime safety infrastructure.

These tests verify that the runtime capability inspection and safe dtype
selection work correctly across different platform configurations.
"""

import pytest
import platform
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
from nodetool.huggingface.runtime_safety import (
    inspect_runtime_capabilities,
    select_safe_dtype,
    _detect_wsl2,
    _detect_wddm,
    estimate_model_memory_mb,
    check_vram_sufficient,
    RuntimeCapabilities,
)

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")


class TestRuntimeCapabilityInspection:
    """Test comprehensive runtime environment detection."""

    def test_inspect_runtime_capabilities_basic(self):
        """Test that capability inspection completes without errors."""
        caps = inspect_runtime_capabilities()
        
        # Basic assertions that should always be true
        assert isinstance(caps, RuntimeCapabilities)
        assert caps.os_name in ("Windows", "Linux", "Darwin")
        assert isinstance(caps.cuda_available, bool)
        assert isinstance(caps.cpu_only, bool)
        
    def test_detect_wsl2_on_linux(self):
        """Test WSL2 detection logic."""
        with patch('platform.system', return_value='Linux'):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = \
                    'Linux version 5.10.16.3-microsoft-standard-WSL2'
                result = _detect_wsl2()
                assert result is True
                
    def test_detect_wsl2_on_native_linux(self):
        """Test that native Linux is not detected as WSL2."""
        with patch('platform.system', return_value='Linux'):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = \
                    'Linux version 5.15.0-generic'
                result = _detect_wsl2()
                assert result is False
                
    def test_detect_wsl2_on_windows(self):
        """Test that Windows is never detected as WSL2."""
        with patch('platform.system', return_value='Windows'):
            result = _detect_wsl2()
            assert result is False
    
    def test_detect_wddm_on_windows(self):
        """Test WDDM detection on Windows."""
        with patch('platform.system', return_value='Windows'):
            result = _detect_wddm()
            assert result is True
    
    def test_detect_wddm_on_linux(self):
        """Test WDDM detection on Linux."""
        with patch('platform.system', return_value='Linux'):
            result = _detect_wddm()
            assert result is False

    def test_capability_summary(self):
        """Test that capability summary produces a readable string."""
        caps = inspect_runtime_capabilities()
        summary = caps.summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert caps.os_name in summary


class TestSafeDtypeSelection:
    """Test safe dtype selection policy."""
    
    @requires_torch
    def test_select_safe_dtype_cpu_only(self):
        """Test that CPU-only systems get float32."""
        import torch
        
        caps = RuntimeCapabilities(
            os_name="Linux",
            is_wsl2=False,
            cuda_available=False,
            cuda_version=None,
            torch_cuda_version=None,
            gpu_count=0,
            gpu_name=None,
            compute_capability=None,
            total_vram_mb=None,
            available_vram_mb=None,
            bf16_hardware_support=False,
            is_wddm=False,
            cpu_only=True,
            mps_available=False,
        )
        
        dtype = select_safe_dtype(torch.bfloat16, caps)
        assert dtype == torch.float32
        
    @requires_torch
    def test_select_safe_dtype_windows_native(self):
        """Test that native Windows forces float16."""
        import torch
        
        caps = RuntimeCapabilities(
            os_name="Windows",
            is_wsl2=False,
            cuda_available=True,
            cuda_version="11.8",
            torch_cuda_version="11.8",
            gpu_count=1,
            gpu_name="NVIDIA GeForce RTX 3090",
            compute_capability=(8, 6),
            total_vram_mb=24000,
            available_vram_mb=20000,
            bf16_hardware_support=True,
            is_wddm=True,
            cpu_only=False,
            mps_available=False,
        )
        
        dtype = select_safe_dtype(torch.bfloat16, caps)
        assert dtype == torch.float16
    
    @requires_torch
    def test_select_safe_dtype_linux_with_ampere(self):
        """Test that Linux with Ampere GPU allows bfloat16."""
        import torch
        
        caps = RuntimeCapabilities(
            os_name="Linux",
            is_wsl2=False,
            cuda_available=True,
            cuda_version="11.8",
            torch_cuda_version="11.8",
            gpu_count=1,
            gpu_name="NVIDIA A100",
            compute_capability=(8, 0),
            total_vram_mb=40000,
            available_vram_mb=35000,
            bf16_hardware_support=True,
            is_wddm=False,
            cpu_only=False,
            mps_available=False,
        )
        
        dtype = select_safe_dtype(torch.bfloat16, caps)
        assert dtype == torch.bfloat16
    
    @requires_torch
    def test_select_safe_dtype_linux_with_volta(self):
        """Test that Linux with Volta GPU uses float16."""
        import torch
        
        caps = RuntimeCapabilities(
            os_name="Linux",
            is_wsl2=False,
            cuda_available=True,
            cuda_version="11.8",
            torch_cuda_version="11.8",
            gpu_count=1,
            gpu_name="NVIDIA V100",
            compute_capability=(7, 0),
            total_vram_mb=16000,
            available_vram_mb=14000,
            bf16_hardware_support=False,
            is_wddm=False,
            cpu_only=False,
            mps_available=False,
        )
        
        dtype = select_safe_dtype(torch.bfloat16, caps)
        assert dtype == torch.float16
    
    @requires_torch
    def test_select_safe_dtype_mps(self):
        """Test that MPS (Apple Metal) uses float16."""
        import torch
        
        caps = RuntimeCapabilities(
            os_name="Darwin",
            is_wsl2=False,
            cuda_available=False,
            cuda_version=None,
            torch_cuda_version=None,
            gpu_count=0,
            gpu_name=None,
            compute_capability=None,
            total_vram_mb=None,
            available_vram_mb=None,
            bf16_hardware_support=False,
            is_wddm=False,
            cpu_only=False,
            mps_available=True,
        )
        
        dtype = select_safe_dtype(torch.bfloat16, caps)
        assert dtype == torch.float16


class TestVRAMEstimation:
    """Test VRAM estimation and validation."""
    
    @requires_torch
    def test_estimate_model_memory_flux(self):
        """Test FLUX model memory estimation."""
        import torch
        
        # FLUX with bfloat16
        mem_mb = estimate_model_memory_mb("flux-schnell", torch.bfloat16)
        assert mem_mb > 10000  # Should be > 10GB
        assert mem_mb < 20000  # Should be < 20GB (with dtype reduction)
        
    @requires_torch
    def test_estimate_model_memory_sdxl(self):
        """Test SDXL model memory estimation."""
        import torch
        
        # SDXL with float16
        mem_mb = estimate_model_memory_mb("sdxl", torch.float16)
        assert mem_mb > 3000   # Should be > 3GB
        assert mem_mb < 8000   # Should be < 8GB
    
    def test_check_vram_sufficient_pass(self):
        """Test VRAM sufficiency check when there's enough memory."""
        caps = RuntimeCapabilities(
            os_name="Linux",
            is_wsl2=False,
            cuda_available=True,
            cuda_version="11.8",
            torch_cuda_version="11.8",
            gpu_count=1,
            gpu_name="NVIDIA A100",
            compute_capability=(8, 0),
            total_vram_mb=40000,
            available_vram_mb=35000,
            bf16_hardware_support=True,
            is_wddm=False,
            cpu_only=False,
            mps_available=False,
        )
        
        sufficient, msg = check_vram_sufficient(10000, caps, safety_margin_mb=1000)
        assert sufficient is True
        assert "sufficient" in msg.lower()
    
    def test_check_vram_sufficient_fail(self):
        """Test VRAM sufficiency check when there's not enough memory."""
        caps = RuntimeCapabilities(
            os_name="Linux",
            is_wsl2=False,
            cuda_available=True,
            cuda_version="11.8",
            torch_cuda_version="11.8",
            gpu_count=1,
            gpu_name="NVIDIA RTX 3060",
            compute_capability=(8, 6),
            total_vram_mb=12000,
            available_vram_mb=8000,
            bf16_hardware_support=True,
            is_wddm=False,
            cpu_only=False,
            mps_available=False,
        )
        
        sufficient, msg = check_vram_sufficient(20000, caps, safety_margin_mb=1000)
        assert sufficient is False
        assert "insufficient" in msg.lower()
    
    def test_check_vram_cpu_only(self):
        """Test VRAM check on CPU-only system."""
        caps = RuntimeCapabilities(
            os_name="Linux",
            is_wsl2=False,
            cuda_available=False,
            cuda_version=None,
            torch_cuda_version=None,
            gpu_count=0,
            gpu_name=None,
            compute_capability=None,
            total_vram_mb=None,
            available_vram_mb=None,
            bf16_hardware_support=False,
            is_wddm=False,
            cpu_only=True,
            mps_available=False,
        )
        
        sufficient, msg = check_vram_sufficient(10000, caps)
        assert sufficient is False
        assert "no gpu" in msg.lower()


class TestCaching:
    """Test capability caching functionality."""
    
    def test_get_cached_capabilities(self):
        """Test that cached capabilities are returned."""
        from nodetool.huggingface.runtime_safety import get_cached_capabilities
        
        # First call should inspect
        caps1 = get_cached_capabilities()
        assert isinstance(caps1, RuntimeCapabilities)
        
        # Second call should return cached
        caps2 = get_cached_capabilities()
        assert caps1 is caps2
        
    def test_force_refresh_capabilities(self):
        """Test forced refresh of capabilities."""
        from nodetool.huggingface.runtime_safety import get_cached_capabilities
        
        caps1 = get_cached_capabilities()
        caps2 = get_cached_capabilities(force_refresh=True)
        
        # After force refresh, we get a new object
        # (may have same values but is a new instance)
        assert isinstance(caps2, RuntimeCapabilities)


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
