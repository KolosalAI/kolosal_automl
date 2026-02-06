"""
Integration tests for Kolosal AutoML.

These tests verify that both Rust and Python backends work correctly
and produce consistent results.
"""

import os
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

# Test with both backends
BACKENDS = ["rust", "python"]


def set_backend(backend: str):
    """Set the backend via environment variable."""
    os.environ["KOLOSAL_USE_RUST"] = "1" if backend == "rust" else "0"


@pytest.fixture(params=BACKENDS)
def backend(request):
    """Fixture to test with both backends."""
    set_backend(request.param)
    yield request.param
    # Reset to default
    os.environ["KOLOSAL_USE_RUST"] = "1"


class TestDataPreprocessor:
    """Test DataPreprocessor with both backends."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        np.random.seed(42)
        return np.random.randn(100, 5)
    
    def test_standard_scaler_mean_std(self, backend, sample_data):
        """StandardScaler should produce mean≈0, std≈1."""
        # Import after setting backend
        from kolosal import DataPreprocessor
        
        preprocessor = DataPreprocessor(scaling="standard")
        transformed = preprocessor.fit_transform(sample_data)
        
        # Check mean and std
        col_means = np.mean(transformed, axis=0)
        col_stds = np.std(transformed, axis=0)
        
        assert_array_almost_equal(col_means, np.zeros(5), decimal=10)
        assert_array_almost_equal(col_stds, np.ones(5), decimal=10)
    
    def test_minmax_scaler_bounds(self, backend, sample_data):
        """MinMaxScaler should produce values in [0, 1]."""
        from kolosal import DataPreprocessor
        
        preprocessor = DataPreprocessor(scaling="minmax")
        transformed = preprocessor.fit_transform(sample_data)
        
        assert transformed.min() >= 0.0 - 1e-10
        assert transformed.max() <= 1.0 + 1e-10
    
    def test_inverse_transform_recovers_original(self, backend, sample_data):
        """Inverse transform should recover original data."""
        from kolosal import DataPreprocessor
        
        preprocessor = DataPreprocessor(scaling="standard")
        transformed = preprocessor.fit_transform(sample_data)
        recovered = preprocessor.inverse_transform(transformed)
        
        assert_array_almost_equal(recovered, sample_data, decimal=10)
    
    def test_transform_without_fit_raises(self, backend, sample_data):
        """Transform without fit should raise error."""
        from kolosal import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            preprocessor.transform(sample_data)


class TestTrainEngine:
    """Test TrainEngine with both backends."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
        return X, y
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(200) * 0.1
        return X, y
    
    def test_classification_accuracy(self, backend, classification_data):
        """Classification should achieve reasonable accuracy."""
        from kolosal import TrainEngine
        
        X, y = classification_data
        engine = TrainEngine(task_type="classification", cv_folds=3)
        
        result = engine.train(X, y, model_type="random_forest")
        
        # Should achieve better than random
        assert result.get("mean_score", result.get("accuracy", 0)) > 0.6
    
    def test_regression_r2(self, backend, regression_data):
        """Regression should achieve reasonable R²."""
        from kolosal import TrainEngine
        
        X, y = regression_data
        engine = TrainEngine(task_type="regression", cv_folds=3, metric="r2")
        
        result = engine.train(X, y, model_type="linear_regression")
        
        # Should achieve positive R²
        assert result.get("mean_score", result.get("r2", 0)) > 0.5


class TestBackendParity:
    """Test that both backends produce consistent results."""
    
    def test_preprocessing_parity(self):
        """Both backends should produce same preprocessing results."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        
        # Test with Rust backend
        os.environ["KOLOSAL_USE_RUST"] = "1"
        try:
            from kolosal import DataPreprocessor as RustPreprocessor
            rust_pp = RustPreprocessor(scaling="standard")
            rust_result = rust_pp.fit_transform(data.copy())
            rust_available = True
        except ImportError:
            rust_available = False
        
        # Test with Python backend
        os.environ["KOLOSAL_USE_RUST"] = "0"
        # Need to reload the module
        import importlib
        import kolosal
        importlib.reload(kolosal)
        from kolosal import DataPreprocessor as PythonPreprocessor
        python_pp = PythonPreprocessor(scaling="standard")
        python_result = python_pp.fit_transform(data.copy())
        
        if rust_available:
            # Results should be very close
            assert_array_almost_equal(rust_result, python_result, decimal=6)


class TestBackendInfo:
    """Test backend information utilities."""
    
    def test_get_backend_info(self, backend):
        """Should return valid backend info."""
        from kolosal import get_backend_info, BACKEND
        
        info = get_backend_info()
        
        assert "backend" in info
        assert "version" in info
        assert info["backend"] == BACKEND


# Benchmark tests (only run with --benchmark flag)
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks comparing backends."""
    
    def test_preprocessing_speed(self, benchmark):
        """Benchmark preprocessing speed."""
        from kolosal import DataPreprocessor
        
        np.random.seed(42)
        data = np.random.randn(10000, 20)
        
        preprocessor = DataPreprocessor(scaling="standard")
        
        def run():
            preprocessor.fit_transform(data.copy())
        
        result = benchmark(run)
        # Just verify it runs
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
