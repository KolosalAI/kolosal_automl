# ---------------------------------------------------------------------
# tests/benchmark/test_ml_benchmark.py - ML operations performance tests
# ---------------------------------------------------------------------
"""
Benchmark tests for ML operations and model training performance.
Tests consistency of ML pipelines and fallback behavior.
"""
import pytest
import pandas as pd
import numpy as np
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os
import pickle
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.conftest import time_function, measure_memory_usage

# Test if ML modules are available
try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.ml
]

class TestMLOperationBenchmarks:
    """Benchmark tests for ML operations."""
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_dataset_generation_performance(self, benchmark_result):
        """Benchmark synthetic dataset generation performance."""
        benchmark_result.start()
        
        dataset_configs = [
            ('small_classification', make_classification, {'n_samples': 1000, 'n_features': 10, 'n_classes': 2, 'n_informative': 8}),
            ('medium_classification', make_classification, {'n_samples': 10000, 'n_features': 20, 'n_classes': 2, 'n_informative': 15}),
            ('large_classification', make_classification, {'n_samples': 50000, 'n_features': 50, 'n_classes': 2, 'n_informative': 40}),
            ('small_regression', make_regression, {'n_samples': 1000, 'n_features': 10}),
            ('medium_regression', make_regression, {'n_samples': 10000, 'n_features': 20}),
            ('large_regression', make_regression, {'n_samples': 50000, 'n_features': 50}),
        ]
        
        generation_results = {}
        
        for dataset_name, generator_func, params in dataset_configs:
            result, generation_time = time_function(
                lambda g=generator_func, p=params: g(**p, random_state=42)
            )
            X, y = result
            
            # Calculate dataset statistics
            memory_usage = X.nbytes + y.nbytes
            
            generation_results[dataset_name] = {
                'generation_time_ms': generation_time,
                'samples': X.shape[0],
                'features': X.shape[1],
                'memory_bytes': memory_usage,
                'memory_mb': memory_usage / 1024**2,
                'generation_rate_samples_per_sec': X.shape[0] / (generation_time / 1000) if generation_time > 0 else 0
            }
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'dataset_generation_results': generation_results,
            'total_generation_time_ms': sum(r['generation_time_ms'] for r in generation_results.values())
        }
        
        # Dataset generation should be reasonably fast
        for dataset_name, result in generation_results.items():
            samples = result['samples']
            time_ms = result['generation_time_ms']
            
            # Should generate at least 1000 samples per second
            rate = result['generation_rate_samples_per_sec']
            assert rate > 1000, f"{dataset_name} generation too slow: {rate} samples/sec"
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_model_training_performance(self, benchmark_result):
        """Benchmark model training performance."""
        benchmark_result.start()
        
        # Generate test datasets
        X_clf, y_clf = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=42)
        X_reg, y_reg = make_regression(n_samples=5000, n_features=20, random_state=42)
        
        # Split datasets
        X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42
        )
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        training_results = {}
        
        # Test classification models
        clf_models = [
            ('rf_classifier_small', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('rf_classifier_medium', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('rf_classifier_large', RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
        
        for model_name, model in clf_models:
            # Train model
            trained_model, train_time = time_function(
                model.fit, X_clf_train, y_clf_train
            )
            
            # Predict
            predictions, predict_time = time_function(
                trained_model.predict, X_clf_test
            )
            
            # Calculate accuracy
            accuracy = accuracy_score(y_clf_test, predictions)
            
            training_results[model_name] = {
                'type': 'classification',
                'train_time_ms': train_time,
                'predict_time_ms': predict_time,
                'accuracy': accuracy,
                'train_samples': len(X_clf_train),
                'test_samples': len(X_clf_test),
                'features': X_clf_train.shape[1]
            }
        
        # Test regression models
        reg_models = [
            ('rf_regressor_small', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('rf_regressor_medium', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('rf_regressor_large', RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
        
        for model_name, model in reg_models:
            # Train model
            trained_model, train_time = time_function(
                model.fit, X_reg_train, y_reg_train
            )
            
            # Predict
            predictions, predict_time = time_function(
                trained_model.predict, X_reg_test
            )
            
            # Calculate MSE
            mse = mean_squared_error(y_reg_test, predictions)
            
            training_results[model_name] = {
                'type': 'regression',
                'train_time_ms': train_time,
                'predict_time_ms': predict_time,
                'mse': mse,
                'train_samples': len(X_reg_train),
                'test_samples': len(X_reg_test),
                'features': X_reg_train.shape[1]
            }
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'training_results': training_results,
            'total_train_time_ms': sum(r['train_time_ms'] for r in training_results.values()),
            'total_predict_time_ms': sum(r['predict_time_ms'] for r in training_results.values())
        }
        
        # Training should complete in reasonable time
        for model_name, result in training_results.items():
            train_time = result['train_time_ms']
            predict_time = result['predict_time_ms']
            
            # Training time limits based on model size
            if 'small' in model_name:
                assert train_time < 5000, f"{model_name} training took {train_time}ms"
            elif 'medium' in model_name:
                assert train_time < 15000, f"{model_name} training took {train_time}ms"
            elif 'large' in model_name:
                assert train_time < 30000, f"{model_name} training took {train_time}ms"
            
            # Prediction should be fast
            assert predict_time < 1000, f"{model_name} prediction took {predict_time}ms"
    
    @pytest.mark.benchmark
    @pytest.mark.memory
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_ml_memory_usage(self, benchmark_result, memory_monitor):
        """Test memory usage during ML operations."""
        benchmark_result.start()
        
        memory_monitor.sample()  # Baseline
        
        # Generate dataset with memory tracking
        result, dataset_memory = measure_memory_usage(
            lambda: make_classification(n_samples=10000, n_features=50, random_state=42)
        )
        X, y = result
        memory_monitor.sample()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        memory_monitor.sample()
        
        # Train model with memory tracking
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        trained_model, training_memory = measure_memory_usage(
            model.fit, X_train, y_train
        )
        memory_monitor.sample()
        
        # Predict with memory tracking
        predictions, prediction_memory = measure_memory_usage(
            trained_model.predict, X_test
        )
        memory_monitor.sample()
        
        # Model serialization memory
        serialized_model, serialization_memory = measure_memory_usage(
            lambda: pickle.dumps(trained_model)
        )
        memory_monitor.sample()
        
        benchmark_result.stop()
        
        monitor_stats = memory_monitor.get_stats()
        
        benchmark_result.metadata = {
            'dataset_memory_delta_mb': dataset_memory['delta_mb'],
            'training_memory_delta_mb': training_memory['delta_mb'],
            'prediction_memory_delta_mb': prediction_memory['delta_mb'],
            'serialization_memory_delta_mb': serialization_memory['delta_mb'],
            'serialized_model_size_mb': len(serialized_model) / 1024**2,
            'memory_monitor_stats': monitor_stats,
            'dataset_shape': X.shape,
            'model_params': model.get_params()
        }
        
        # Memory usage should be reasonable
        assert monitor_stats.get('peak_mb', 0) < 1000, f"Peak memory usage: {monitor_stats.get('peak_mb', 0)}MB"
        assert training_memory['delta_mb'] < 200, f"Training used {training_memory['delta_mb']}MB"
    
    @pytest.mark.benchmark
    @pytest.mark.consistency
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_ml_reproducibility(self, benchmark_result):
        """Test ML model training reproducibility."""
        benchmark_result.start()
        
        # Generate consistent dataset
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models with same parameters
        model_results = []
        
        for i in range(5):
            model = RandomForestClassifier(n_estimators=20, random_state=42)
            
            # Train and predict
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            # Get feature importances
            feature_importances = model.feature_importances_.tolist()
            
            model_results.append({
                'run': i,
                'accuracy': accuracy,
                'feature_importances': feature_importances,
                'n_estimators': model.n_estimators,
                'predictions_checksum': hash(tuple(predictions))
            })
        
        benchmark_result.stop()
        
        # Analyze consistency
        accuracies = [r['accuracy'] for r in model_results]
        checksums = [r['predictions_checksum'] for r in model_results]
        
        benchmark_result.metadata = {
            'model_results': model_results,
            'accuracy_variance': np.var(accuracies),
            'accuracy_std': np.std(accuracies),
            'consistent_predictions': len(set(checksums)) == 1,
            'unique_checksums': len(set(checksums))
        }
        
        # Results should be highly consistent with fixed random seed
        assert np.std(accuracies) < 0.01, f"Accuracy std deviation too high: {np.std(accuracies)}"
        assert len(set(checksums)) == 1, "Predictions not consistent across runs"


class TestMLPipelineBenchmarks:
    """Benchmark tests for complete ML pipelines."""
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_PANDAS or not HAS_SKLEARN, reason="Required ML libraries not available")
    def test_end_to_end_pipeline_performance(self, benchmark_result, test_data_generator):
        """Benchmark complete ML pipeline from data loading to model evaluation."""
        benchmark_result.start()
        
        # Generate test data and save to file
        test_data = test_data_generator('medium')
        data_file = test_data['file_path']
        
        pipeline_times = {}
        
        # Step 1: Data loading
        df, load_time = time_function(pd.read_csv, data_file)
        pipeline_times['data_loading_ms'] = load_time
        
        # Step 2: Data preprocessing
        def preprocess_data(dataframe):
            # Separate features and target
            X = dataframe.drop('target', axis=1)
            y = dataframe['target']
            
            # Handle categorical columns (simple encoding)
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col] = pd.Categorical(X[col]).codes
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            return X, y
        
        (X, y), preprocess_time = time_function(preprocess_data, df)
        pipeline_times['preprocessing_ms'] = preprocess_time
        
        # Step 3: Train-test split
        split_result, split_time = time_function(
            train_test_split, X, y, test_size=0.2, random_state=42
        )
        X_train, X_test, y_train, y_test = split_result
        pipeline_times['train_test_split_ms'] = split_time
        
        # Step 4: Model training
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        trained_model, train_time = time_function(model.fit, X_train, y_train)
        pipeline_times['model_training_ms'] = train_time
        
        # Step 5: Model prediction
        predictions, predict_time = time_function(trained_model.predict, X_test)
        pipeline_times['model_prediction_ms'] = predict_time
        
        # Step 6: Model evaluation
        def evaluate_model(y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            return {'accuracy': accuracy}
        
        evaluation, eval_time = time_function(evaluate_model, y_test, predictions)
        pipeline_times['model_evaluation_ms'] = eval_time
        
        # Step 7: Model serialization
        serialized_model, serialize_time = time_function(pickle.dumps, trained_model)
        pipeline_times['model_serialization_ms'] = serialize_time
        
        benchmark_result.stop()
        
        total_pipeline_time = sum(pipeline_times.values())
        
        benchmark_result.metadata = {
            'pipeline_times_ms': pipeline_times,
            'total_pipeline_time_ms': total_pipeline_time,
            'data_shape': df.shape,
            'model_evaluation': evaluation,
            'serialized_model_size_mb': len(serialized_model) / 1024**2,
            'pipeline_efficiency': {
                'samples_per_second': len(df) / (total_pipeline_time / 1000) if total_pipeline_time > 0 else 0,
                'features_per_second': (len(df) * df.shape[1]) / (total_pipeline_time / 1000) if total_pipeline_time > 0 else 0
            }
        }
        
        # Pipeline should complete in reasonable time
        assert total_pipeline_time < 30000, f"Complete pipeline took {total_pipeline_time}ms"
        
        # Each step should be reasonably fast
        assert pipeline_times['data_loading_ms'] < 5000, "Data loading too slow"
        assert pipeline_times['preprocessing_ms'] < 5000, "Preprocessing too slow"
        assert pipeline_times['model_training_ms'] < 15000, "Model training too slow"
        assert pipeline_times['model_prediction_ms'] < 2000, "Model prediction too slow"
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_repeated_training_stress(self, benchmark_result, benchmark_config):
        """Test repeated model training for stability and memory leaks."""
        benchmark_result.start()
        
        # Generate dataset once
        X, y = make_classification(n_samples=2000, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        iterations = min(benchmark_config['stress_iterations'], 20)
        training_results = []
        
        import psutil
        process = psutil.Process()
        
        for i in range(iterations):
            memory_before = process.memory_info().rss / 1024**2
            
            # Train model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = (time.perf_counter() - start_time) * 1000
            
            # Predict and evaluate
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            memory_after = process.memory_info().rss / 1024**2
            
            training_results.append({
                'iteration': i,
                'train_time_ms': train_time,
                'accuracy': accuracy,
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_delta_mb': memory_after - memory_before
            })
            
            # Cleanup
            del model, predictions
            
            # Periodic garbage collection
            if i % 5 == 0:
                import gc
                gc.collect()
        
        benchmark_result.stop()
        
        # Analyze results
        train_times = [r['train_time_ms'] for r in training_results]
        accuracies = [r['accuracy'] for r in training_results]
        memory_deltas = [r['memory_delta_mb'] for r in training_results]
        
        memory_growth = training_results[-1]['memory_after_mb'] - training_results[0]['memory_before_mb']
        
        benchmark_result.metadata = {
            'iterations': iterations,
            'training_results': training_results,
            'performance_stats': {
                'avg_train_time_ms': np.mean(train_times),
                'std_train_time_ms': np.std(train_times),
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'total_memory_growth_mb': memory_growth,
                'avg_memory_delta_mb': np.mean(memory_deltas)
            }
        }
        
        # Performance should be stable
        assert np.std(train_times) < np.mean(train_times) * 0.5, "Training time too variable"
        assert np.std(accuracies) < 0.05, "Accuracy too variable"
        assert memory_growth < 100, f"Memory grew by {memory_growth}MB"


@pytest.mark.benchmark
@pytest.mark.stress
class TestMLStressTests:
    """Stress tests for ML operations under high load."""
    
    @pytest.mark.stress
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_concurrent_training_simulation(self, benchmark_result):
        """Simulate concurrent ML training operations."""
        benchmark_result.start()
        
        # Generate multiple datasets
        datasets = []
        for i in range(5):
            X, y = make_classification(n_samples=1000, n_features=10, random_state=i)
            datasets.append((X, y))
        
        # Simulate concurrent training by rapidly switching between operations
        concurrent_results = []
        start_time = time.perf_counter()
        
        for iteration in range(20):
            dataset_idx = iteration % len(datasets)
            X, y = datasets[dataset_idx]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Quick training
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Quick prediction
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            concurrent_results.append({
                'iteration': iteration,
                'dataset_idx': dataset_idx,
                'accuracy': accuracy,
                'model_size': len(pickle.dumps(model))
            })
            
            # Cleanup
            del model, predictions
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'total_operations': len(concurrent_results),
            'total_time_ms': total_time,
            'avg_operation_time_ms': total_time / len(concurrent_results),
            'operations_per_second': len(concurrent_results) / (total_time / 1000) if total_time > 0 else 0,
            'concurrent_results': concurrent_results
        }
        
        # Should handle rapid operations
        assert total_time < 60000, f"Concurrent simulation took {total_time}ms"
        
        # All operations should produce reasonable results
        accuracies = [r['accuracy'] for r in concurrent_results]
        assert all(acc > 0.5 for acc in accuracies), "Some models performed poorly"
