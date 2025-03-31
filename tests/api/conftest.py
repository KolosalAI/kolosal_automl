"""
Configuration and fixtures for pytest.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from pathlib import Path

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import API modules
from modules.api.app import app
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType

@pytest.fixture
def client():
    """Return a test client for the FastAPI app."""
    return TestClient(app)

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