# ---------------------------------------------------------------------
# tests/benchmark/test_inference_model_size_benchmark.py - Inference benchmarks for different model sizes
# ---------------------------------------------------------------------
"""
Benchmark tests for model inference performance across different model sizes and complexities.
Tests the same model architecture with varying parameters to measure scaling behavior.
"""

import pytest
import pandas as pd
import numpy as np
import time
import tempfile
import pickle
import json
from pathlib import Path
from unittest.mock import Mock, patch
import sys
from typing import Dict, List, Tuple, Any
import gc
import psutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.conftest import time_function, measure_memory_usage

# Test if ML modules are available
try:
    import os
    # Windows compatibility: Force single-threaded execution to avoid subprocess issues
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.inference
]

class InferenceModelConfig:
    """Configuration for different model sizes and complexities."""
    
    DATASET_SIZES = {
        'tiny': {'samples': 100, 'features': 5},
        'small': {'samples': 1000, 'features': 10}, 
        'medium': {'samples': 5000, 'features': 20},
        'large': {'samples': 10000, 'features': 50},
        'xl': {'samples': 50000, 'features': 100}
    }
    
    MODEL_COMPLEXITIES = {
        'simple': {
            'RandomForest': {'n_estimators': 10, 'max_depth': 5},
            'XGBoost': {'n_estimators': 10, 'max_depth': 3},
            'LightGBM': {'n_estimators': 10, 'max_depth': 3}
        },
        'medium': {
            'RandomForest': {'n_estimators': 50, 'max_depth': 10},
            'XGBoost': {'n_estimators': 50, 'max_depth': 6},
            'LightGBM': {'n_estimators': 50, 'max_depth': 6}
        },
        'complex': {
            'RandomForest': {'n_estimators': 100, 'max_depth': 15},
            'XGBoost': {'n_estimators': 100, 'max_depth': 8},
            'LightGBM': {'n_estimators': 100, 'max_depth': 8}
        },
        'very_complex': {
            'RandomForest': {'n_estimators': 200, 'max_depth': 20},
            'XGBoost': {'n_estimators': 200, 'max_depth': 10},
            'LightGBM': {'n_estimators': 200, 'max_depth': 10}
        }
    }
    
    BATCH_SIZES = [1, 5, 10, 25, 50, 100, 250, 500, 1000]

class ModelSizeBenchmarkRunner:
    """Run inference benchmarks across different model sizes."""
    
    def __init__(self):
        self.models = {}
        self.datasets = {}
        self.results = {}
        
    def generate_datasets(self, random_state: int = 42) -> Dict[str, Dict]:
        """Generate datasets of various sizes."""
        datasets = {}
        
        for size_name, config in InferenceModelConfig.DATASET_SIZES.items():
            # Generate classification dataset
            X_clf, y_clf = make_classification(
                n_samples=config['samples'],
                n_features=config['features'],
                n_classes=2,
                n_informative=max(2, config['features'] // 2),
                n_redundant=max(0, config['features'] // 4),
                random_state=random_state
            )
            
            # Generate regression dataset
            X_reg, y_reg = make_regression(
                n_samples=config['samples'],
                n_features=config['features'],
                noise=0.1,
                random_state=random_state
            )
            
            # Split datasets
            X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
                X_clf, y_clf, test_size=0.2, random_state=random_state
            )
            X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=random_state
            )
            
            datasets[size_name] = {
                'classification': {
                    'X_train': X_clf_train, 'X_test': X_clf_test,
                    'y_train': y_clf_train, 'y_test': y_clf_test
                },
                'regression': {
                    'X_train': X_reg_train, 'X_test': X_reg_test,
                    'y_train': y_reg_train, 'y_test': y_reg_test
                },
                'config': config
            }
            
        return datasets
    
    def train_models(self, datasets: Dict[str, Dict]) -> Dict[str, Dict]:
        """Train models of different complexities on various dataset sizes."""
        models = {}
        
        for size_name, data in datasets.items():
            models[size_name] = {}
            
            for complexity, model_configs in InferenceModelConfig.MODEL_COMPLEXITIES.items():
                models[size_name][complexity] = {}
                
                # Train classification models
                if HAS_SKLEARN:
                    # Random Forest Classifier with Windows-friendly settings
                    rf_clf = RandomForestClassifier(
                        **model_configs['RandomForest'],
                        random_state=42,
                        n_jobs=1  # Use single thread to avoid subprocess issues
                    )
                    rf_clf.fit(data['classification']['X_train'], data['classification']['y_train'])
                    models[size_name][complexity]['RandomForestClassifier'] = rf_clf
                    
                    # Logistic Regression
                    lr_clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=1)
                    lr_clf.fit(data['classification']['X_train'], data['classification']['y_train'])
                    models[size_name][complexity]['LogisticRegression'] = lr_clf
                    
                    # Random Forest Regressor with Windows-friendly settings
                    rf_reg = RandomForestRegressor(
                        **model_configs['RandomForest'],
                        random_state=42,
                        n_jobs=1  # Use single thread to avoid subprocess issues
                    )
                    rf_reg.fit(data['regression']['X_train'], data['regression']['y_train'])
                    models[size_name][complexity]['RandomForestRegressor'] = rf_reg
                    
                    # Linear Regression
                    lr_reg = LinearRegression(n_jobs=1)
                    lr_reg.fit(data['regression']['X_train'], data['regression']['y_train'])
                    models[size_name][complexity]['LinearRegression'] = lr_reg
                
                # XGBoost models with thread control
                if HAS_XGBOOST:
                    try:
                        xgb_clf = xgb.XGBClassifier(
                            **model_configs['XGBoost'],
                            random_state=42,
                            eval_metric='logloss',
                            nthread=1  # Use single thread
                        )
                        xgb_clf.fit(data['classification']['X_train'], data['classification']['y_train'])
                        models[size_name][complexity]['XGBClassifier'] = xgb_clf
                        
                        xgb_reg = xgb.XGBRegressor(
                            **model_configs['XGBoost'],
                            random_state=42,
                            nthread=1  # Use single thread
                        )
                        xgb_reg.fit(data['regression']['X_train'], data['regression']['y_train'])
                        models[size_name][complexity]['XGBRegressor'] = xgb_reg
                    except Exception as e:
                        print(f"Warning: XGBoost training failed for {size_name}/{complexity}: {e}")
                
                # LightGBM models with thread control
                if HAS_LIGHTGBM:
                    try:
                        lgb_clf = lgb.LGBMClassifier(
                            **model_configs['LightGBM'],
                            random_state=42,
                            verbose=-1,
                            num_threads=1  # Use single thread
                        )
                        lgb_clf.fit(data['classification']['X_train'], data['classification']['y_train'])
                        models[size_name][complexity]['LGBMClassifier'] = lgb_clf
                        
                        lgb_reg = lgb.LGBMRegressor(
                            **model_configs['LightGBM'],
                            random_state=42,
                            verbose=-1,
                            num_threads=1  # Use single thread
                        )
                        lgb_reg.fit(data['regression']['X_train'], data['regression']['y_train'])
                        models[size_name][complexity]['LGBMRegressor'] = lgb_reg
                    except Exception as e:
                        print(f"Warning: LightGBM training failed for {size_name}/{complexity}: {e}")
        
        return models
    
    def benchmark_single_predictions(self, models: Dict, datasets: Dict) -> Dict:
        """Benchmark single prediction performance."""
        results = {}
        
        for size_name, size_models in models.items():
            results[size_name] = {}
            data = datasets[size_name]
            
            for complexity, complexity_models in size_models.items():
                results[size_name][complexity] = {}
                
                for model_name, model in complexity_models.items():
                    # Determine test data type
                    if 'Classifier' in model_name or 'Logistic' in model_name:
                        X_test = data['classification']['X_test']
                        y_test = data['classification']['y_test']
                        task_type = 'classification'
                    else:
                        X_test = data['regression']['X_test']
                        y_test = data['regression']['y_test']
                        task_type = 'regression'
                    
                    # Single prediction timing
                    single_times = []
                    predictions = []
                    
                    # Test first 100 samples for single predictions
                    test_samples = min(100, len(X_test))
                    for i in range(test_samples):
                        sample = X_test[i:i+1]
                        
                        start_time = time.perf_counter()
                        pred = model.predict(sample)[0]
                        end_time = time.perf_counter()
                        
                        single_times.append((end_time - start_time) * 1000)  # Convert to ms
                        predictions.append(pred)
                    
                    # Calculate metrics
                    if task_type == 'classification':
                        accuracy = accuracy_score(y_test[:test_samples], predictions)
                        metric_value = accuracy
                        metric_name = 'accuracy'
                    else:
                        mse = mean_squared_error(y_test[:test_samples], predictions)
                        metric_value = mse
                        metric_name = 'mse'
                    
                    # Model size estimation
                    model_size_bytes = len(pickle.dumps(model))
                    
                    results[size_name][complexity][model_name] = {
                        'avg_single_prediction_ms': np.mean(single_times),
                        'std_single_prediction_ms': np.std(single_times),
                        'min_single_prediction_ms': np.min(single_times),
                        'max_single_prediction_ms': np.max(single_times),
                        'median_single_prediction_ms': np.median(single_times),
                        'predictions_per_second': 1000 / np.mean(single_times),
                        'model_size_bytes': model_size_bytes,
                        'model_size_mb': model_size_bytes / (1024 * 1024),
                        'test_samples': test_samples,
                        'task_type': task_type,
                        metric_name: metric_value,
                        'feature_count': X_test.shape[1]
                    }
        
        return results
    
    def benchmark_batch_predictions(self, models: Dict, datasets: Dict) -> Dict:
        """Benchmark batch prediction performance."""
        results = {}
        
        for size_name, size_models in models.items():
            results[size_name] = {}
            data = datasets[size_name]
            
            for complexity, complexity_models in size_models.items():
                results[size_name][complexity] = {}
                
                for model_name, model in complexity_models.items():
                    # Determine test data type
                    if 'Classifier' in model_name or 'Logistic' in model_name:
                        X_test = data['classification']['X_test']
                        y_test = data['classification']['y_test']
                        task_type = 'classification'
                    else:
                        X_test = data['regression']['X_test']
                        y_test = data['regression']['y_test']
                        task_type = 'regression'
                    
                    batch_results = {}
                    
                    for batch_size in InferenceModelConfig.BATCH_SIZES:
                        if batch_size > len(X_test):
                            continue
                        
                        batch_times = []
                        total_predictions = 0
                        
                        # Test multiple batches
                        num_batches = min(10, len(X_test) // batch_size)
                        for batch_idx in range(num_batches):
                            start_idx = batch_idx * batch_size
                            end_idx = start_idx + batch_size
                            batch_X = X_test[start_idx:end_idx]
                            
                            start_time = time.perf_counter()
                            predictions = model.predict(batch_X)
                            end_time = time.perf_counter()
                            
                            batch_time = (end_time - start_time) * 1000  # Convert to ms
                            batch_times.append(batch_time)
                            total_predictions += len(predictions)
                        
                        if batch_times:
                            avg_batch_time = np.mean(batch_times)
                            throughput = (batch_size * 1000) / avg_batch_time  # predictions per second
                            
                            batch_results[f'batch_size_{batch_size}'] = {
                                'avg_batch_time_ms': avg_batch_time,
                                'std_batch_time_ms': np.std(batch_times),
                                'throughput_predictions_per_sec': throughput,
                                'batch_size': batch_size,
                                'num_batches_tested': len(batch_times),
                                'total_predictions': total_predictions,
                                'time_per_prediction_ms': avg_batch_time / batch_size
                            }
                    
                    results[size_name][complexity][model_name] = batch_results
        
        return results
    
    def benchmark_memory_usage(self, models: Dict, datasets: Dict) -> Dict:
        """Benchmark memory usage during inference."""
        results = {}
        process = psutil.Process()
        
        for size_name, size_models in models.items():
            results[size_name] = {}
            data = datasets[size_name]
            
            for complexity, complexity_models in size_models.items():
                results[size_name][complexity] = {}
                
                for model_name, model in complexity_models.items():
                    # Determine test data type
                    if 'Classifier' in model_name or 'Logistic' in model_name:
                        X_test = data['classification']['X_test']
                    else:
                        X_test = data['regression']['X_test']
                    
                    # Measure memory before inference
                    gc.collect()
                    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
                    
                    # Perform batch prediction with memory monitoring
                    batch_size = min(100, len(X_test))
                    X_batch = X_test[:batch_size]
                    
                    memory_samples = []
                    start_time = time.perf_counter()
                    
                    # Monitor memory during prediction
                    for _ in range(5):  # Multiple iterations for stability
                        predictions = model.predict(X_batch)
                        memory_samples.append(process.memory_info().rss / (1024 * 1024))
                    
                    end_time = time.perf_counter()
                    total_time = (end_time - start_time) * 1000  # ms
                    
                    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                    peak_memory = max(memory_samples)
                    avg_memory = np.mean(memory_samples)
                    
                    results[size_name][complexity][model_name] = {
                        'memory_before_mb': memory_before,
                        'memory_after_mb': memory_after,
                        'memory_delta_mb': memory_after - memory_before,
                        'peak_memory_mb': peak_memory,
                        'avg_memory_during_inference_mb': avg_memory,
                        'memory_per_prediction_kb': (avg_memory - memory_before) * 1024 / batch_size,
                        'total_inference_time_ms': total_time,
                        'batch_size_tested': batch_size
                    }
        
        return results


class TestInferenceModelSizeBenchmarks:
    """Main benchmark test class for inference across different model sizes."""
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_inference_scaling_across_model_sizes(self, benchmark_result):
        """Test how inference performance scales across different model sizes and complexities."""
        benchmark_result.start()
        
        runner = ModelSizeBenchmarkRunner()
        
        # Generate datasets
        datasets = runner.generate_datasets()
        
        # Train models
        models = runner.train_models(datasets)
        
        # Run benchmarks
        single_prediction_results = runner.benchmark_single_predictions(models, datasets)
        batch_prediction_results = runner.benchmark_batch_predictions(models, datasets)
        memory_results = runner.benchmark_memory_usage(models, datasets)
        
        benchmark_result.stop()
        
        # Compile comprehensive results
        benchmark_result.metadata = {
            'single_prediction_results': single_prediction_results,
            'batch_prediction_results': batch_prediction_results,
            'memory_usage_results': memory_results,
            'dataset_configurations': InferenceModelConfig.DATASET_SIZES,
            'model_complexities': InferenceModelConfig.MODEL_COMPLEXITIES,
            'libraries_available': {
                'sklearn': HAS_SKLEARN,
                'xgboost': HAS_XGBOOST,
                'lightgbm': HAS_LIGHTGBM
            }
        }
        
        # Performance assertions
        self._validate_inference_performance(single_prediction_results, batch_prediction_results)
    
    def _validate_inference_performance(self, single_results, batch_results):
        """Validate that inference performance meets expected criteria."""
        
        # Check that single predictions are reasonably fast
        for size_name, size_results in single_results.items():
            for complexity, complexity_results in size_results.items():
                for model_name, model_results in complexity_results.items():
                    avg_time = model_results['avg_single_prediction_ms']
                    
                    # Set performance expectations based on model complexity
                    if complexity == 'simple':
                        max_time = 100  # 100ms max for simple models
                    elif complexity == 'medium':
                        max_time = 500  # 500ms max for medium models
                    elif complexity == 'complex':
                        max_time = 1000  # 1s max for complex models
                    else:  # very_complex
                        max_time = 2000  # 2s max for very complex models
                    
                    assert avg_time < max_time, \
                        f"{model_name} ({complexity}) avg prediction time {avg_time}ms exceeds {max_time}ms limit"
        
        # Check that batch predictions are more efficient than single predictions
        for size_name in single_results.keys():
            if size_name in batch_results:
                for complexity in single_results[size_name].keys():
                    if complexity in batch_results[size_name]:
                        for model_name in single_results[size_name][complexity].keys():
                            if model_name in batch_results[size_name][complexity]:
                                single_rate = single_results[size_name][complexity][model_name]['predictions_per_second']
                                
                                # Find best batch throughput
                                batch_model_results = batch_results[size_name][complexity][model_name]
                                best_batch_rate = max(
                                    result['throughput_predictions_per_sec']
                                    for result in batch_model_results.values()
                                    if 'throughput_predictions_per_sec' in result
                                )
                                
                                # Batch should be at least as fast as single predictions
                                assert best_batch_rate >= single_rate * 0.8, \
                                    f"{model_name} batch processing not efficient: {best_batch_rate} vs {single_rate} single"
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_model_complexity_impact_on_inference(self, benchmark_result):
        """Test how model complexity affects inference performance."""
        benchmark_result.start()
        
        runner = ModelSizeBenchmarkRunner()
        
        # Use medium dataset for complexity comparison
        dataset_size = 'medium'
        data_config = InferenceModelConfig.DATASET_SIZES[dataset_size]
        
        # Generate single dataset
        X, y = make_classification(
            n_samples=data_config['samples'],
            n_features=data_config['features'],
            n_classes=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        complexity_results = {}
        
        for complexity_name, model_configs in InferenceModelConfig.MODEL_COMPLEXITIES.items():
            complexity_results[complexity_name] = {}
            
            if HAS_SKLEARN:
                # Train Random Forest with different complexities
                rf_config = model_configs['RandomForest']
                rf_model = RandomForestClassifier(**rf_config, random_state=42)
                rf_model.fit(X_train, y_train)
                
                # Benchmark inference
                single_times = []
                for i in range(100):
                    sample = X_test[i:i+1]
                    start_time = time.perf_counter()
                    prediction = rf_model.predict(sample)
                    end_time = time.perf_counter()
                    single_times.append((end_time - start_time) * 1000)
                
                # Batch prediction test
                batch_size = 50
                batch_X = X_test[:batch_size]
                start_time = time.perf_counter()
                batch_predictions = rf_model.predict(batch_X)
                end_time = time.perf_counter()
                batch_time = (end_time - start_time) * 1000
                
                complexity_results[complexity_name]['RandomForest'] = {
                    'avg_single_time_ms': np.mean(single_times),
                    'batch_time_ms': batch_time,
                    'batch_throughput_per_sec': (batch_size * 1000) / batch_time,
                    'model_params': rf_config,
                    'model_size_mb': len(pickle.dumps(rf_model)) / (1024 * 1024),
                    'accuracy': accuracy_score(y_test[:100], rf_model.predict(X_test[:100]))
                }
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'complexity_comparison': complexity_results,
            'dataset_config': data_config,
            'test_setup': {
                'dataset_size': dataset_size,
                'single_prediction_samples': 100,
                'batch_size': 50
            }
        }
        
        # Validate complexity scaling
        self._validate_complexity_scaling(complexity_results)
    
    def _validate_complexity_scaling(self, complexity_results):
        """Validate that model complexity scaling is reasonable."""
        rf_results = {}
        
        # Extract RandomForest results for comparison
        for complexity, results in complexity_results.items():
            if 'RandomForest' in results:
                rf_results[complexity] = results['RandomForest']
        
        complexity_order = ['simple', 'medium', 'complex', 'very_complex']
        available_complexities = [c for c in complexity_order if c in rf_results]
        
        if len(available_complexities) >= 2:
            # Check that inference time generally increases with complexity
            prev_time = 0
            for complexity in available_complexities:
                current_time = rf_results[complexity]['avg_single_time_ms']
                
                # Allow some variance, but very complex models should be slower than simple ones
                if complexity == 'very_complex' and 'simple' in rf_results:
                    simple_time = rf_results['simple']['avg_single_time_ms']
                    assert current_time > simple_time * 0.5, \
                        f"Very complex model not significantly slower than simple: {current_time}ms vs {simple_time}ms"
                
                prev_time = current_time
    
    @pytest.mark.benchmark
    @pytest.mark.memory
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_inference_memory_scaling(self, benchmark_result, memory_monitor):
        """Test memory usage scaling during inference."""
        benchmark_result.start()
        
        memory_monitor.sample()  # Baseline
        
        runner = ModelSizeBenchmarkRunner()
        
        # Test memory scaling across different dataset sizes
        memory_scaling_results = {}
        
        for size_name, config in InferenceModelConfig.DATASET_SIZES.items():
            if config['samples'] > 20000:  # Skip very large datasets for memory test
                continue
                
            memory_monitor.sample()
            
            # Generate dataset
            X, y = make_classification(
                n_samples=config['samples'],
                n_features=config['features'],
                random_state=42
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            memory_monitor.sample()
            
            # Test inference memory usage
            process = psutil.Process()
            
            # Small batch inference
            small_batch_size = min(10, len(X_test))
            gc.collect()
            memory_before_small = process.memory_info().rss / (1024 * 1024)
            
            predictions_small = model.predict(X_test[:small_batch_size])
            memory_after_small = process.memory_info().rss / (1024 * 1024)
            
            # Large batch inference
            large_batch_size = min(100, len(X_test))
            gc.collect()
            memory_before_large = process.memory_info().rss / (1024 * 1024)
            
            predictions_large = model.predict(X_test[:large_batch_size])
            memory_after_large = process.memory_info().rss / (1024 * 1024)
            
            memory_scaling_results[size_name] = {
                'dataset_samples': config['samples'],
                'dataset_features': config['features'],
                'small_batch_memory_delta_mb': memory_after_small - memory_before_small,
                'large_batch_memory_delta_mb': memory_after_large - memory_before_large,
                'memory_per_small_prediction_kb': (memory_after_small - memory_before_small) * 1024 / small_batch_size,
                'memory_per_large_prediction_kb': (memory_after_large - memory_before_large) * 1024 / large_batch_size,
                'model_size_mb': len(pickle.dumps(model)) / (1024 * 1024)
            }
            
            memory_monitor.sample()
        
        benchmark_result.stop()
        
        monitor_stats = memory_monitor.get_stats()
        benchmark_result.metadata = {
            'memory_scaling_results': memory_scaling_results,
            'memory_monitor_stats': monitor_stats
        }
        
        # Validate memory usage is reasonable
        for size_name, results in memory_scaling_results.items():
            # Memory per prediction should be reasonable
            small_mem_per_pred = abs(results['memory_per_small_prediction_kb'])
            large_mem_per_pred = abs(results['memory_per_large_prediction_kb'])
            
            # Allow up to 1MB per prediction (very generous)
            assert small_mem_per_pred < 1024, f"{size_name} small batch uses {small_mem_per_pred}KB per prediction"
            assert large_mem_per_pred < 1024, f"{size_name} large batch uses {large_mem_per_pred}KB per prediction"
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_concurrent_inference_stress(self, benchmark_result):
        """Test inference performance under concurrent load."""
        benchmark_result.start()
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Generate test data
        X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Concurrent inference test
        num_threads = 5
        predictions_per_thread = 50
        
        def worker_inference(thread_id, model, X_test, num_predictions):
            """Worker function for concurrent inference."""
            thread_results = []
            
            for i in range(num_predictions):
                sample_idx = (thread_id * num_predictions + i) % len(X_test)
                sample = X_test[sample_idx:sample_idx+1]
                
                start_time = time.perf_counter()
                prediction = model.predict(sample)[0]
                end_time = time.perf_counter()
                
                thread_results.append({
                    'thread_id': thread_id,
                    'prediction_idx': i,
                    'inference_time_ms': (end_time - start_time) * 1000,
                    'prediction': prediction
                })
            
            return thread_results
        
        # Execute concurrent inference
        concurrent_results = []
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_inference, thread_id, model, X_test, predictions_per_thread)
                for thread_id in range(num_threads)
            ]
            
            for future in as_completed(futures):
                concurrent_results.extend(future.result())
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Analyze results
        inference_times = [r['inference_time_ms'] for r in concurrent_results]
        total_predictions = len(concurrent_results)
        
        # Group by thread for analysis
        thread_stats = {}
        for result in concurrent_results:
            thread_id = result['thread_id']
            if thread_id not in thread_stats:
                thread_stats[thread_id] = []
            thread_stats[thread_id].append(result['inference_time_ms'])
        
        thread_analysis = {}
        for thread_id, times in thread_stats.items():
            thread_analysis[f'thread_{thread_id}'] = {
                'avg_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times),
                'predictions_completed': len(times)
            }
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'concurrent_inference_stats': {
                'total_predictions': total_predictions,
                'total_time_ms': total_time,
                'avg_inference_time_ms': np.mean(inference_times),
                'std_inference_time_ms': np.std(inference_times),
                'overall_throughput_predictions_per_sec': (total_predictions * 1000) / total_time,
                'num_threads': num_threads,
                'predictions_per_thread': predictions_per_thread
            },
            'thread_analysis': thread_analysis,
            'all_inference_times': inference_times[:100]  # Sample for debugging
        }
        
        # Validate concurrent performance
        avg_time = np.mean(inference_times)
        throughput = (total_predictions * 1000) / total_time
        
        assert avg_time < 500, f"Average concurrent inference time too high: {avg_time}ms"
        assert throughput > 10, f"Concurrent throughput too low: {throughput} predictions/sec"
        
        # Check that all threads completed successfully
        assert total_predictions == num_threads * predictions_per_thread, \
            f"Not all predictions completed: {total_predictions} vs {num_threads * predictions_per_thread}"
