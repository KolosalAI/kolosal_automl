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
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to the Python path for all tests
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging for tests
def setup_test_logging():
    """Set up logging configuration for tests."""
    # Create logs directory if it doesn't exist
    log_dir = Path("tests")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_dir / "test.log", mode='a'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific log levels for different modules
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Set up logging when conftest is imported
test_logger = setup_test_logging()

# Import shared configurations
try:
    from modules.configs import MLTrainingEngineConfig, TaskType
except ImportError:
    # Create minimal mock classes if import fails
    class TaskType:
        CLASSIFICATION = "classification"
        REGRESSION = "regression"
    
    class MLTrainingEngineConfig:
        def __init__(self):
            self.model_path = "./test_models"
            self.task_type = TaskType.CLASSIFICATION

@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Set up the test session with logging."""
    logger = logging.getLogger("test_session")
    logger.info("=" * 80)
    logger.info("Starting test session")
    logger.info("=" * 80)
    yield
    logger.info("=" * 80)
    logger.info("Test session completed")
    logger.info("=" * 80)

@pytest.fixture(scope="function", autouse=True)
def log_test_start_end(request):
    """Log the start and end of each test."""
    logger = logging.getLogger("test_function")
    test_name = request.node.name
    logger.info(f"Starting test: {test_name}")
    yield
    logger.info(f"Completed test: {test_name}")

@pytest.fixture(scope="session")
def test_dir():
    """Create and return a temporary test directory."""
    with tempfile.TemporaryDirectory(prefix="kolosal_test_") as temp_dir:
        test_dir = Path(temp_dir)
        yield test_dir
        # Automatic cleanup when context exits

@pytest.fixture(scope="function")
def temp_test_dir():
    """Create a temporary directory for individual test functions."""
    with tempfile.TemporaryDirectory(prefix="kolosal_test_func_") as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def models_dir():
    """Create temporary models directory for tests."""
    models_dir = Path("./test_models")
    models_dir.mkdir(exist_ok=True)
    yield models_dir
    # Cleanup in session cleanup

@pytest.fixture(scope="session")
def data_dir():
    """Create temporary data directory for tests."""
    data_dir = Path("./test_data")
    data_dir.mkdir(exist_ok=True)
    yield data_dir
    # Cleanup in session cleanup

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Clean up temporary files after all tests have run."""
    yield  # Run all tests first
    
    logger = logging.getLogger("test_cleanup")
    logger.info("Starting cleanup of test artifacts...")
    
    # Then clean up directories
    cleanup_dirs = [
        "./test_models",
        "./test_data", 
        "./reports",
        "./optimization_jobs",
        "./streaming_results",
        "./checkpoints",
        "./static/uploads",
        "./static/models",
        "./static/charts",
        "./static/reports"
    ]
    
    for dir_path in cleanup_dirs:
        path = Path(dir_path)
        if path.exists():
            try:
                shutil.rmtree(path)
                logger.info(f"Cleaned up directory: {dir_path}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not clean up {dir_path}: {e}")

@pytest.fixture
def test_logger():
    """Provide a logger for tests to use."""
    return logging.getLogger("test")

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "n_samples": n_samples,
        "n_features": n_features
    }

# ---------------------------------------------------------------------
# Common fixtures for API/functional tests
# ---------------------------------------------------------------------

@pytest.fixture
def client():
    """Return a test client for the FastAPI app."""
    try:
        from fastapi.testclient import TestClient
        from modules.api.app import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI not available for API tests")

# ---------------------------------------------------------------------
# Common fixtures for ML engine tests
# ---------------------------------------------------------------------

@pytest.fixture
def ml_engine():
    """Return a configured ML Training Engine instance."""
    try:
        from modules.engine.train_engine import MLTrainingEngine
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