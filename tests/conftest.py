# ---------------------------------------------------------------------
# tests/conftest.py - Root level fixtures available to all tests
# ---------------------------------------------------------------------
"""
Global fixtures and configuration for pytest.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to the Python path for all tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import shared configurations
from modules.configs import MLTrainingEngineConfig, TaskType

@pytest.fixture(scope="session")
def test_dir():
    """Create and return a temporary test directory."""
    test_dir = Path("./test_data")
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup can be handled by specific tests or in session-scoped autouse fixture

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Clean up temporary files after all tests have run."""
    yield  # Run all tests first
    
    # Then clean up directories
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

# ---------------------------------------------------------------------
# tests/functional/conftest.py - Fixtures for API/functional tests
# ---------------------------------------------------------------------
"""
Fixtures for functional tests of the API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

# Import API modules
from modules.api.app import app

@pytest.fixture
def client():
    """Return a test client for the FastAPI app."""
    return TestClient(app)

# ---------------------------------------------------------------------
# tests/integration/conftest.py - Fixtures for integration tests
# ---------------------------------------------------------------------
"""
Fixtures for integration tests that combine multiple components.
"""
import pytest
import numpy as np
import pandas as pd
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType

@pytest.fixture
def ml_engine():
    """Return a configured ML Training Engine instance."""
    config = MLTrainingEngineConfig()
    config.model_path = "./test_models"
    config.task_type = TaskType.CLASSIFICATION
    return MLTrainingEngine(config)

@pytest.fixture
def regression_ml_engine():
    """Return a configured ML Training Engine instance for regression tasks."""
    config = MLTrainingEngineConfig()
    config.model_path = "./test_models"
    config.task_type = TaskType.REGRESSION
    return MLTrainingEngine(config)

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