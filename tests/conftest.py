"""
Pytest configuration and shared fixtures for nodetool-huggingface tests.
"""
import os
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, AsyncMock
import uuid

# Add src to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))

# Set test environment variables
os.environ.setdefault("ENV", "test")
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_KEY", "")
os.environ.setdefault("NODETOOL_SMOKE_TEST", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Disable CUDA for tests


@pytest.fixture
def mock_processing_context():
    """Create a mock processing context for testing."""
    from nodetool.workflows.processing_context import ProcessingContext
    
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        job_id=f"test-{uuid.uuid4().hex}",
    )
    return context


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    from PIL import Image
    return Image.new("RGB", (64, 64), color=(200, 100, 50))


@pytest.fixture
def sample_audio():
    """Create a sample test audio."""
    from pydub.generators import Sine
    return Sine(440).to_audio_segment(duration=500)


@pytest.fixture
def sample_dataframe():
    """Create a sample test dataframe."""
    import pandas as pd
    return pd.DataFrame({"city": ["Paris", "Berlin"], "pop": [2.1, 3.6]})


@pytest.fixture
def mock_huggingface_model():
    """Create a mock HuggingFace model that doesn't require actual model loading."""
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    return mock_model


@pytest.fixture
def mock_pipeline():
    """Create a mock HuggingFace pipeline."""
    mock_pipe = MagicMock()
    mock_pipe.__call__ = MagicMock(return_value={"generated_text": "test output"})
    return mock_pipe


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests as requiring actual model download"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
