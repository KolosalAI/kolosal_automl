"""
Test configuration and fixtures specifically for optimization system tests.

This module provides:
- Specialized fixtures for optimization testing
- Performance test utilities
- Mock data generators for different scenarios
- Test configuration constants
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock
import logging

# Test configuration constants
TEST_RANDOM_SEED = 42
DEFAULT_TEST_SIZES = {
    'tiny': 1_000,
    'small': 10_000, 
    'medium': 50_000,
    'large': 100_000
}

PERFORMANCE_THRESHOLDS = {
    'loading_speed_rows_per_sec': 5_000,
    'memory_optimization_max_time_sec': 10.0,
    'min_memory_reduction_percent': 5.0,
    'max_processing_time_small_sec': 5.0,
    'max_processing_time_medium_sec': 30.0
}

# Try to import optimization modules for availability checking
try:
    from modules.engine.optimization_integration import OptimizedDataPipeline
    from modules.engine.optimized_data_loader import OptimizedDataLoader
    from modules.engine.adaptive_preprocessing import ConfigOptimizer
    from modules.engine.memory_aware_processor import MemoryAwareDataProcessor
    OPTIMIZATION_MODULES_AVAILABLE = True
except ImportError:
    OPTIMIZATION_MODULES_AVAILABLE = False


@pytest.fixture(scope="session")
def optimization_test_logger():
    """Provide a logger specifically for optimization tests"""
    logger = logging.getLogger("optimization_tests")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Add console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


@pytest.fixture(scope="function")
def optimization_temp_dir():
    """Create a temporary directory for optimization tests"""
    with tempfile.TemporaryDirectory(prefix="optimization_test_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def performance_tracker():
    """Fixture to track performance metrics during tests"""
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
        
        def record_metric(self, name: str, value: float, unit: str = ""):
            self.metrics[name] = {'value': value, 'unit': unit}
        
        def get_metric(self, name: str):
            return self.metrics.get(name)
        
        def get_all_metrics(self):
            return self.metrics.copy()
        
        def assert_performance(self, name: str, threshold: float, operation: str = "less_than"):
            if name not in self.metrics:
                pytest.fail(f"Metric '{name}' not recorded")
            
            value = self.metrics[name]['value']
            
            if operation == "less_than":
                assert value < threshold, f"{name} ({value}) should be less than {threshold}"
            elif operation == "greater_than":
                assert value > threshold, f"{name} ({value}) should be greater than {threshold}"
            elif operation == "equals":
                assert abs(value - threshold) < 0.01, f"{name} ({value}) should equal {threshold}"
    
    return PerformanceTracker()


class DatasetGenerator:
    """Utility class for generating test datasets with specific characteristics"""
    
    def __init__(self, random_seed: int = TEST_RANDOM_SEED):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def create_basic_dataset(self, n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
        """Create a basic dataset with numerical and categorical features"""
        np.random.seed(self.random_seed)
        
        data = {}
        
        # Numerical features
        for i in range(n_features // 2):
            data[f'num_{i}'] = np.random.randn(n_samples)
        
        # Categorical features
        for i in range(n_features - n_features // 2):
            cardinality = min(10, max(3, n_samples // 100))
            data[f'cat_{i}'] = np.random.choice([f'cat_{j}' for j in range(cardinality)], n_samples)
        
        # Target variable
        data['target'] = np.random.randint(0, 2, n_samples)
        
        return pd.DataFrame(data)
    
    def create_memory_inefficient_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create a dataset with memory-inefficient dtypes"""
        np.random.seed(self.random_seed)
        
        return pd.DataFrame({
            'int64_small': pd.Series(range(n_samples), dtype='int64'),  # Could be int32 or smaller
            'float64_small': pd.Series(np.random.randn(n_samples), dtype='float64'),  # Could be float32
            'string_categorical': [f'category_{i % 5}' for i in range(n_samples)],  # Should be categorical
            'bool_as_object': pd.Series([True, False] * (n_samples // 2), dtype='object'),  # Should be bool
            'redundant_string': ['text_' + str(i) for i in range(n_samples)]  # High memory string data
        })
    
    def create_sparse_dataset(self, n_samples: int = 1000, sparsity: float = 0.8) -> pd.DataFrame:
        """Create a sparse dataset with many zero values"""
        np.random.seed(self.random_seed)
        n_features = 20
        
        # Create data with specified sparsity
        data = np.random.randn(n_samples, n_features)
        zero_mask = np.random.choice([True, False], size=data.shape, p=[sparsity, 1-sparsity])
        data[zero_mask] = 0
        
        df = pd.DataFrame(data, columns=[f'sparse_feature_{i}' for i in range(n_features)])
        df['target'] = np.random.randint(0, 2, n_samples)
        
        return df
    
    def create_high_cardinality_dataset(self, n_samples: int = 5000) -> pd.DataFrame:
        """Create dataset with high cardinality categorical features"""
        np.random.seed(self.random_seed)
        
        return pd.DataFrame({
            'id_unique': [f'id_{i}' for i in range(n_samples)],  # Unique values
            'high_cardinality': [f'item_{i % 1000}' for i in range(n_samples)],  # High cardinality
            'medium_cardinality': [f'category_{i % 100}' for i in range(n_samples)],  # Medium cardinality
            'low_cardinality': np.random.choice(['A', 'B', 'C'], n_samples),  # Low cardinality
            'numeric': np.random.randn(n_samples),
            'target': np.random.randint(0, 3, n_samples)
        })
    
    def create_outlier_dataset(self, n_samples: int = 1000, outlier_ratio: float = 0.1) -> pd.DataFrame:
        """Create dataset with outliers"""
        np.random.seed(self.random_seed)
        
        # Normal data
        normal_data = np.random.randn(n_samples)
        
        # Inject outliers
        n_outliers = int(n_samples * outlier_ratio)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        normal_data[outlier_indices] = np.random.randn(n_outliers) * 10 + 20  # Extreme values
        
        return pd.DataFrame({
            'normal_feature': np.random.randn(n_samples),
            'outlier_feature': normal_data,
            'skewed_feature': np.random.exponential(2, n_samples),  # Positively skewed
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
    
    def create_missing_data_dataset(self, n_samples: int = 1000, missing_ratio: float = 0.15) -> pd.DataFrame:
        """Create dataset with missing values"""
        np.random.seed(self.random_seed)
        
        df = pd.DataFrame({
            'complete_feature': np.random.randn(n_samples),
            'partial_missing': np.random.randn(n_samples),
            'high_missing': np.random.randn(n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # Add missing values
        n_missing_partial = int(n_samples * missing_ratio)
        n_missing_high = int(n_samples * missing_ratio * 2)
        
        partial_missing_indices = np.random.choice(n_samples, n_missing_partial, replace=False)
        high_missing_indices = np.random.choice(n_samples, n_missing_high, replace=False)
        
        df.loc[partial_missing_indices, 'partial_missing'] = np.nan
        df.loc[high_missing_indices, 'high_missing'] = np.nan
        
        return df
    
    def save_dataset_to_file(self, df: pd.DataFrame, file_path: Path, format: str = 'csv') -> Path:
        """Save dataset to file in specified format"""
        if format == 'csv':
            df.to_csv(file_path, index=False)
        elif format == 'xlsx':
            df.to_excel(file_path, index=False)
        elif format == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return file_path


@pytest.fixture
def dataset_generator():
    """Provide a dataset generator for tests"""
    return DatasetGenerator()


@pytest.fixture
def basic_test_dataset(dataset_generator):
    """Provide a basic test dataset"""
    return dataset_generator.create_basic_dataset(n_samples=1000, n_features=10)


@pytest.fixture
def memory_inefficient_dataset(dataset_generator):
    """Provide a memory-inefficient dataset for optimization testing"""
    return dataset_generator.create_memory_inefficient_dataset(n_samples=2000)


@pytest.fixture
def sparse_test_dataset(dataset_generator):
    """Provide a sparse dataset for testing"""
    return dataset_generator.create_sparse_dataset(n_samples=5000, sparsity=0.85)


@pytest.fixture
def high_cardinality_dataset(dataset_generator):
    """Provide a high cardinality dataset for testing"""
    return dataset_generator.create_high_cardinality_dataset(n_samples=3000)


@pytest.fixture
def outlier_dataset(dataset_generator):
    """Provide an outlier dataset for testing"""
    return dataset_generator.create_outlier_dataset(n_samples=2000, outlier_ratio=0.15)


@pytest.fixture
def missing_data_dataset(dataset_generator):
    """Provide a dataset with missing values"""
    return dataset_generator.create_missing_data_dataset(n_samples=1500, missing_ratio=0.2)


@pytest.fixture
def test_csv_files(optimization_temp_dir, dataset_generator):
    """Create multiple test CSV files for testing"""
    files = {}
    
    # Create different types of datasets
    datasets = {
        'basic': dataset_generator.create_basic_dataset(1000, 8),
        'memory_inefficient': dataset_generator.create_memory_inefficient_dataset(1500),
        'sparse': dataset_generator.create_sparse_dataset(2000, 0.8),
        'high_cardinality': dataset_generator.create_high_cardinality_dataset(2500),
        'outliers': dataset_generator.create_outlier_dataset(1200, 0.12),
        'missing_data': dataset_generator.create_missing_data_dataset(1800, 0.18)
    }
    
    # Save to files
    for name, df in datasets.items():
        file_path = optimization_temp_dir / f"{name}.csv"
        dataset_generator.save_dataset_to_file(df, file_path, 'csv')
        files[name] = file_path
    
    return files


@pytest.fixture(scope="session")
def optimization_performance_thresholds():
    """Provide performance thresholds for optimization tests"""
    return PERFORMANCE_THRESHOLDS.copy()


@pytest.fixture
def mock_optimization_components():
    """Provide mock optimization components for testing fallback behavior"""
    class MockOptimizationComponents:
        def __init__(self):
            self.data_loader = Mock()
            self.memory_processor = Mock()
            self.config_optimizer = Mock()
            self.adaptive_chunker = Mock()
        
        def setup_success_responses(self):
            """Setup mocks to return successful responses"""
            # Mock data loader
            self.data_loader.load_data.return_value = (
                pd.DataFrame({'x': range(100), 'y': range(100)}),
                Mock(rows=100, columns=2, loading_time=0.1)
            )
            
            # Mock memory processor
            self.memory_processor.optimize_dataframe_memory.return_value = pd.DataFrame({'x': range(100)})
            
            # Mock config optimizer
            self.config_optimizer.optimize_for_dataset.return_value = {
                'normalization': 'standard',
                'strategy': 'standard'
            }
        
        def setup_failure_responses(self):
            """Setup mocks to simulate failures"""
            self.data_loader.load_data.side_effect = Exception("Mock failure")
            self.memory_processor.optimize_dataframe_memory.side_effect = Exception("Mock failure")
            self.config_optimizer.optimize_for_dataset.side_effect = Exception("Mock failure")
    
    return MockOptimizationComponents()


# Pytest marks for categorizing tests
pytest_optimization_unit = pytest.mark.optimization_unit
pytest_optimization_integration = pytest.mark.optimization_integration  
pytest_optimization_performance = pytest.mark.optimization_performance
pytest_optimization_slow = pytest.mark.slow

# Skip markers for conditional testing
skip_if_no_optimization = pytest.mark.skipif(
    not OPTIMIZATION_MODULES_AVAILABLE,
    reason="Optimization modules not available"
)

# Parametrize fixtures for different dataset sizes
@pytest.fixture(params=['tiny', 'small', 'medium'])
def dataset_size_category(request):
    """Parametrized fixture for different dataset size categories"""
    return request.param

@pytest.fixture
def sized_dataset(dataset_size_category, dataset_generator):
    """Create dataset based on size category"""
    size = DEFAULT_TEST_SIZES[dataset_size_category]
    return dataset_generator.create_basic_dataset(n_samples=size, n_features=10)


# Custom assertions for optimization tests
class OptimizationAssertions:
    """Custom assertion methods for optimization testing"""
    
    @staticmethod
    def assert_memory_reduction(original_df: pd.DataFrame, optimized_df: pd.DataFrame, 
                              min_reduction_percent: float = 0):
        """Assert that memory optimization achieved minimum reduction"""
        original_memory = original_df.memory_usage(deep=True).sum()
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        reduction_percent = ((original_memory - optimized_memory) / original_memory) * 100
        
        assert reduction_percent >= min_reduction_percent, \
            f"Memory reduction {reduction_percent:.1f}% is below minimum {min_reduction_percent}%"
    
    @staticmethod
    def assert_data_integrity(original_df: pd.DataFrame, optimized_df: pd.DataFrame):
        """Assert that data integrity is preserved after optimization"""
        assert optimized_df.shape == original_df.shape, "DataFrame shape changed"
        
        # Check that data values are preserved (allowing for dtype changes)
        pd.testing.assert_frame_equal(
            optimized_df.astype(str),
            original_df.astype(str),
            check_dtype=False
        )
    
    @staticmethod
    def assert_performance_threshold(actual_time: float, threshold_time: float, operation: str):
        """Assert performance meets threshold"""
        if operation == "loading":
            assert actual_time < threshold_time, \
                f"Loading time {actual_time:.3f}s exceeds threshold {threshold_time:.3f}s"
        elif operation == "optimization":
            assert actual_time < threshold_time, \
                f"Optimization time {actual_time:.3f}s exceeds threshold {threshold_time:.3f}s"


@pytest.fixture
def optimization_assertions():
    """Provide optimization-specific assertions"""
    return OptimizationAssertions()


# Utility functions for test setup
def setup_test_environment():
    """Setup test environment for optimization tests"""
    # Set random seed for reproducibility
    np.random.seed(TEST_RANDOM_SEED)
    
    # Configure logging
    logging.getLogger("optimization_tests").setLevel(logging.INFO)
    
    return {
        'random_seed': TEST_RANDOM_SEED,
        'optimization_available': OPTIMIZATION_MODULES_AVAILABLE,
        'test_sizes': DEFAULT_TEST_SIZES,
        'performance_thresholds': PERFORMANCE_THRESHOLDS
    }


# Session-wide test environment setup
@pytest.fixture(scope="session", autouse=True)
def test_environment_setup():
    """Automatically setup test environment for all optimization tests"""
    env_info = setup_test_environment()
    
    # Log test environment info
    logger = logging.getLogger("optimization_tests")
    logger.info("=" * 60)
    logger.info("OPTIMIZATION SYSTEM TEST ENVIRONMENT")
    logger.info("=" * 60)
    logger.info(f"Optimization modules available: {env_info['optimization_available']}")
    logger.info(f"Random seed: {env_info['random_seed']}")
    logger.info(f"Test dataset sizes: {env_info['test_sizes']}")
    logger.info("=" * 60)
    
    yield env_info
    
    logger.info("=" * 60)
    logger.info("OPTIMIZATION TESTS COMPLETED")
    logger.info("=" * 60)
