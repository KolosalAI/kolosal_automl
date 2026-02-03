"""
Unit test fixtures and configurations.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_model():
    """Create a mock machine learning model for testing."""
    model = Mock()
    model.predict.return_value = np.array([0, 1, 0, 1])
    model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
    model.feature_names_in_ = ["feature1", "feature2", "feature3"]
    model.n_features_in_ = 3
    return model

@pytest.fixture
def sample_features():
    """Generate sample feature data for testing."""
    np.random.seed(42)
    return np.random.randn(10, 3)

@pytest.fixture
def sample_2d_features():
    """Generate 2D sample feature data for testing."""
    np.random.seed(42)
    return np.random.randn(5, 4)

@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup is automatic with tempfile

@pytest.fixture
def config_dict():
    """Basic configuration dictionary for testing."""
    return {
        "enable_batching": True,
        "max_batch_size": 32,
        "enable_caching": True,
        "max_cache_entries": 1000,
        "cache_ttl_seconds": 300,
        "num_threads": 4,
        "enable_quantization": False,
        "debug_mode": False
    }