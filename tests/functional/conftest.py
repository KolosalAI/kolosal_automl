"""
Configuration and fixtures for functional/API tests.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add parent directory to import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

@pytest.fixture
def client():
    """Return a test client for the FastAPI app."""
    try:
        from fastapi.testclient import TestClient
        from modules.api.app import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI not available for API tests")

@pytest.fixture
def api_headers():
    """Common headers for API requests."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

@pytest.fixture
def mock_engine():
    """Mock inference engine for API tests."""
    from unittest.mock import Mock
    engine = Mock()
    engine.state = "READY"
    engine.model = Mock()
    engine.active_requests = 0
    engine.model_info = {"name": "test_model", "version": "1.0"}
    engine.feature_names = ["feature1", "feature2", "feature3"]
    engine.predict.return_value = (True, np.array([0.1, 0.9]), {"inference_time_ms": 5.0})
    engine.get_performance_metrics.return_value = {
        "avg_inference_time_ms": 5.0,
        "throughput_requests_per_second": 100.0,
        "total_requests": 1000,
        "error_rate": 0.01
    }
    return engine

@pytest.fixture
def ml_engine():
    """Return a configured ML Training Engine instance."""
    try:
        from modules.engine.train_engine import MLTrainingEngine
        from modules.configs import MLTrainingEngineConfig, TaskType
        config = MLTrainingEngineConfig()
        config.model_path = "./test_models"
        config.task_type = TaskType.CLASSIFICATION
        return MLTrainingEngine(config)
    except ImportError:
        pytest.skip("ML Training Engine not available")

@pytest.fixture
def regression_ml_engine():
    """Return a configured ML Training Engine instance for regression tasks."""
    try:
        from modules.engine.train_engine import MLTrainingEngine
        from modules.configs import MLTrainingEngineConfig, TaskType
        config = MLTrainingEngineConfig()
        config.model_path = "./test_models"
        config.task_type = TaskType.REGRESSION
        return MLTrainingEngine(config)
    except ImportError:
        pytest.skip("ML Training Engine not available")

@pytest.fixture
def sample_classification_data():
    """Generate a small classification dataset for testing."""
    # Create a simple dataset with two features and binary target
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    df['target'] = y
    
    # Save to CSV file
    test_data_dir = Path("./test_data")
    test_data_dir.mkdir(exist_ok=True)
    file_path = test_data_dir / "classification_data.csv"
    df.to_csv(file_path, index=False)
    
    return {
        "dataframe": df,
        "file_path": str(file_path),
        "X": df.iloc[:, :-1],
        "y": df.iloc[:, -1]
    }

@pytest.fixture
def sample_regression_data():
    """Generate a small regression dataset for testing."""
    # Create a simple dataset with two features and continuous target
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    y = 2 * X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    df['target'] = y
    
    # Save to CSV file
    test_data_dir = Path("./test_data")
    test_data_dir.mkdir(exist_ok=True)
    file_path = test_data_dir / "regression_data.csv"
    df.to_csv(file_path, index=False)
    
    return {
        "dataframe": df,
        "file_path": str(file_path),
        "X": df.iloc[:, :-1],
        "y": df.iloc[:, -1]
    }

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up temporary files after tests."""
    yield
    import shutil
    
    # Clean up test models directory
    test_models_dir = Path("./test_models")
    if test_models_dir.exists():
        shutil.rmtree(test_models_dir)
    
    # Clean up test data directory
    test_data_dir = Path("./test_data")
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
    
    # Clean up reports directory
    reports_dir = Path("./reports")
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
        
    # Clean up optimization_jobs directory
    opt_jobs_dir = Path("./optimization_jobs")
    if opt_jobs_dir.exists():
        shutil.rmtree(opt_jobs_dir)
        
    # Clean up streaming_results directory
    streaming_dir = Path("./streaming_results")
    if streaming_dir.exists():
        shutil.rmtree(streaming_dir)