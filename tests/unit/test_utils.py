"""
Unit tests for HuggingFace utility modules.
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
def test_memory_utils_exports():
    """Test that memory utils exports the expected functions."""
    from nodetool.huggingface import memory_utils
    
    assert hasattr(memory_utils, 'get_memory_usage_mb')
    assert hasattr(memory_utils, 'get_gpu_memory_usage_mb')
    assert hasattr(memory_utils, 'log_memory')
    assert hasattr(memory_utils, 'run_gc')
    assert hasattr(memory_utils, 'log_memory_summary')
    assert hasattr(memory_utils, 'MemoryTracker')
    assert hasattr(memory_utils, 'has_cpu_offload_enabled')
    assert hasattr(memory_utils, 'apply_cpu_offload_if_needed')


@pytest.mark.unit
def test_has_cpu_offload_enabled_returns_true_when_marked():
    """Test that has_cpu_offload_enabled returns True when pipeline is marked."""
    from nodetool.huggingface.memory_utils import has_cpu_offload_enabled
    
    mock_pipeline = MagicMock()
    mock_pipeline._nodetool_cpu_offload_applied = True
    
    assert has_cpu_offload_enabled(mock_pipeline) is True


@pytest.mark.unit
def test_flux_utils_imports():
    """Test that flux_utils can be imported."""
    from nodetool.huggingface import flux_utils
    
    assert hasattr(flux_utils, 'detect_flux_variant')
    assert hasattr(flux_utils, 'flux_variant_to_base_model_id')
    assert hasattr(flux_utils, 'is_nunchaku_transformer')


@pytest.mark.unit
def test_detect_flux_variant():
    """Test flux variant detection."""
    from nodetool.huggingface.flux_utils import detect_flux_variant
    
    assert detect_flux_variant("flux-schnell", None) == "schnell"
    assert detect_flux_variant("flux-dev", None) == "dev"
    assert detect_flux_variant("something-fill", None) == "fill"
    assert detect_flux_variant("random-model", None) == "dev"


@pytest.mark.unit
def test_flux_variant_to_base_model_id():
    """Test flux variant to base model ID mapping."""
    from nodetool.huggingface.flux_utils import flux_variant_to_base_model_id
    
    assert "schnell" in flux_variant_to_base_model_id("schnell")
    assert "dev" in flux_variant_to_base_model_id("dev")
    assert "Fill" in flux_variant_to_base_model_id("fill")


@pytest.mark.unit
def test_is_nunchaku_transformer():
    """Test nunchaku transformer detection."""
    from nodetool.huggingface.flux_utils import is_nunchaku_transformer
    
    assert is_nunchaku_transformer("nunchaku-flux", "model.svdq") is True
    assert is_nunchaku_transformer("regular-flux", "model.safetensors") is False
    assert is_nunchaku_transformer("nunchaku-flux", None) is False
