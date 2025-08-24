# ---------------------------------------------------------------------
# tests/benchmark/test_simple_inference_benchmark.py - Simple inference benchmark for testing
# ---------------------------------------------------------------------
"""
Simple inference benchmark to test model scaling behavior quickly.
"""
import pytest
import pandas as pd
import numpy as np
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.conftest import time_function, measure_memory_usage, configure_threading_for_ml

# Configure optimal threading for ML operations
_threading_backend = configure_threading_for_ml()

# Test if ML modules are available
try:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.inference
]

class TestSimpleInferenceBenchmark:
    """Simple inference benchmark tests."""
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_inference_with_different_model_sizes(self, benchmark_result, optimal_device_config):
        """Test inference performance with different model complexities using optimal device configuration."""
        benchmark_result.start()
        
        # Apply optimal configuration
        optimal_n_jobs = -1  # Use all cores by default
        if optimal_device_config:
            batch_config = optimal_device_config.get('batch_processor_config')
            if batch_config and hasattr(batch_config, 'num_workers'):
                optimal_n_jobs = batch_config.num_workers
        
        # Generate test dataset
        X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define model configurations with different complexities
        model_configs = {
            'small': {'n_estimators': 10, 'max_depth': 5, 'n_jobs': optimal_n_jobs},
            'medium': {'n_estimators': 50, 'max_depth': 10, 'n_jobs': optimal_n_jobs},
            'large': {'n_estimators': 100, 'max_depth': 15, 'n_jobs': optimal_n_jobs}
        }
        
        results = {}
        
        for size_name, config in model_configs.items():
            # Train model
            model = RandomForestClassifier(**config, random_state=42)
            model.fit(X_train, y_train)
            
            # Test single prediction performance
            single_times = []
            for i in range(50):  # Test with 50 samples
                sample = X_test[i:i+1]
                
                start_time = time.perf_counter()
                prediction = model.predict(sample)[0]
                end_time = time.perf_counter()
                
                single_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Test batch prediction performance
            batch_sizes = [1, 10, 50, 100]
            batch_results = {}
            
            for batch_size in batch_sizes:
                if batch_size > len(X_test):
                    continue
                    
                batch_X = X_test[:batch_size]
                
                start_time = time.perf_counter()
                batch_predictions = model.predict(batch_X)
                end_time = time.perf_counter()
                
                batch_time = (end_time - start_time) * 1000  # Convert to ms
                throughput = (batch_size * 1000) / batch_time  # predictions per second
                
                batch_results[f'batch_{batch_size}'] = {
                    'batch_time_ms': batch_time,
                    'throughput_per_sec': throughput,
                    'time_per_prediction_ms': batch_time / batch_size
                }
            
            # Calculate model size
            model_size_bytes = len(pickle.dumps(model))
            
            # Calculate accuracy
            test_predictions = model.predict(X_test[:100])
            accuracy = accuracy_score(y_test[:100], test_predictions)
            
            results[size_name] = {
                'model_config': config,
                'avg_single_prediction_ms': np.mean(single_times),
                'std_single_prediction_ms': np.std(single_times),
                'median_single_prediction_ms': np.median(single_times),
                'predictions_per_second': 1000 / np.mean(single_times),
                'model_size_bytes': model_size_bytes,
                'model_size_mb': model_size_bytes / (1024 * 1024),
                'accuracy': accuracy,
                'batch_results': batch_results
            }
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'inference_scaling_results': results,
            'dataset_info': {
                'samples': X.shape[0],
                'features': X.shape[1],
                'train_size': len(X_train),
                'test_size': len(X_test)
            },
            'optimization_config': {
                'n_jobs_used': optimal_n_jobs,
                'device_config_available': optimal_device_config is not None
            }
        }
        
        # Performance assertions
        for size_name, result in results.items():
            avg_time = result['avg_single_prediction_ms']
            
            # Set reasonable performance expectations
            if size_name == 'small':
                max_time = 50  # 50ms max for small models
            elif size_name == 'medium':
                max_time = 200  # 200ms max for medium models
            else:  # large
                max_time = 500  # 500ms max for large models
            
            assert avg_time < max_time, \
                f"{size_name} model avg prediction time {avg_time}ms exceeds {max_time}ms limit"
            
            # Check that larger models are generally slower (with some tolerance)
            if size_name == 'large' and 'small' in results:
                small_time = results['small']['avg_single_prediction_ms']
                assert avg_time > small_time * 0.5, \
                    f"Large model not significantly slower than small: {avg_time}ms vs {small_time}ms"
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_batch_vs_single_prediction_efficiency(self, benchmark_result, optimal_device_config):
        """Test that batch predictions are more efficient than single predictions using optimal configuration."""
        benchmark_result.start()
        
        # Apply optimal configuration
        optimal_n_jobs = -1
        if optimal_device_config:
            batch_config = optimal_device_config.get('batch_processor_config')
            if batch_config and hasattr(batch_config, 'num_workers'):
                optimal_n_jobs = batch_config.num_workers
        
        # Generate test dataset
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train a medium-complexity model with optimal configuration
        model = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=optimal_n_jobs)
        model.fit(X_train, y_train)
        
        # Test single predictions
        num_single_predictions = 100
        single_times = []
        
        for i in range(num_single_predictions):
            sample = X_test[i:i+1]
            
            start_time = time.perf_counter()
            prediction = model.predict(sample)[0]
            end_time = time.perf_counter()
            
            single_times.append((end_time - start_time) * 1000)
        
        avg_single_time = np.mean(single_times)
        single_throughput = 1000 / avg_single_time  # predictions per second
        
        # Test batch predictions
        batch_size = 100
        batch_X = X_test[:batch_size]
        
        start_time = time.perf_counter()
        batch_predictions = model.predict(batch_X)
        end_time = time.perf_counter()
        
        batch_time = (end_time - start_time) * 1000
        batch_throughput = (batch_size * 1000) / batch_time
        time_per_prediction_batch = batch_time / batch_size
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'single_prediction_stats': {
                'avg_time_ms': avg_single_time,
                'std_time_ms': np.std(single_times),
                'throughput_per_sec': single_throughput,
                'num_predictions': num_single_predictions
            },
            'batch_prediction_stats': {
                'batch_size': batch_size,
                'total_time_ms': batch_time,
                'time_per_prediction_ms': time_per_prediction_batch,
                'throughput_per_sec': batch_throughput
            },
            'efficiency_comparison': {
                'batch_vs_single_speedup': batch_throughput / single_throughput,
                'batch_faster': batch_throughput > single_throughput
            },
            'optimization_config': {
                'n_jobs_used': optimal_n_jobs,
                'device_config_available': optimal_device_config is not None
            }
        }
        
        # Assertions
        assert batch_throughput > single_throughput, \
            f"Batch processing not more efficient: {batch_throughput} vs {single_throughput} predictions/sec"
        
        # Batch should be at least 20% more efficient
        speedup = batch_throughput / single_throughput
        assert speedup > 1.2, f"Batch speedup too low: {speedup}x"
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_data_size_impact_on_inference(self, benchmark_result, optimal_device_config):
        """Test how different input data sizes affect inference performance with optimal configuration."""
        benchmark_result.start()
        
        # Apply optimal configuration
        optimal_n_jobs = -1
        if optimal_device_config:
            batch_config = optimal_device_config.get('batch_processor_config')
            if batch_config and hasattr(batch_config, 'num_workers'):
                optimal_n_jobs = batch_config.num_workers
        
        # Train a single model with optimal configuration
        X_train, y_train = make_classification(n_samples=1000, n_features=15, random_state=42)
        model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=42, n_jobs=optimal_n_jobs)
        model.fit(X_train, y_train)
        
        # Test with different input data sizes (different number of samples, same features)
        data_sizes = {
            'small': 50,
            'medium': 100, 
            'large': 200
        }
        
        results = {}
        
        for size_name, num_samples in data_sizes.items():
            # Generate test data with same number of features as training data
            X_test, _ = make_classification(
                n_samples=num_samples, 
                n_features=15,  # Same as training
                random_state=42
            )
            
            # Use all generated features
            X_test_model = X_test
            
            # Single prediction test
            single_times = []
            for i in range(20):
                sample = X_test_model[i:i+1]
                
                start_time = time.perf_counter()
                prediction = model.predict(sample)[0]
                end_time = time.perf_counter()
                
                single_times.append((end_time - start_time) * 1000)
            
            # Batch prediction test
            batch_size = min(50, num_samples)
            batch_X = X_test_model[:batch_size]
            
            start_time = time.perf_counter()
            batch_predictions = model.predict(batch_X)
            end_time = time.perf_counter()
            
            batch_time = (end_time - start_time) * 1000
            
            results[size_name] = {
                'data_samples': num_samples,
                'model_features_used': 15,
                'avg_single_time_ms': np.mean(single_times),
                'batch_time_ms': batch_time,
                'batch_throughput_per_sec': (batch_size * 1000) / batch_time,
                'data_size_mb': X_test.nbytes / (1024 * 1024)
            }
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'data_size_impact': results,
            'model_info': {
                'n_estimators': 20,
                'max_depth': 8,
                'features_trained_on': 15,
                'n_jobs': optimal_n_jobs
            },
            'optimization_config': {
                'n_jobs_used': optimal_n_jobs,
                'device_config_available': optimal_device_config is not None
            }
        }
        
        # All data sizes should have reasonable performance
        for size_name, result in results.items():
            avg_time = result['avg_single_time_ms']
            assert avg_time < 100, f"{size_name} data size prediction time {avg_time}ms too high"
            
            throughput = result['batch_throughput_per_sec']
            assert throughput > 50, f"{size_name} data size throughput {throughput} too low"
