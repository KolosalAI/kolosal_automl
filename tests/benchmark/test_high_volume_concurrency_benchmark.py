# ---------------------------------------------------------------------
# tests/benchmark/test_high_volume_concurrency_benchmark.py - High Volume Concurrency Benchmarks
# ---------------------------------------------------------------------
"""
High volume concurrency benchmark tests for testing system performance under
massive concurrent loads, high-throughput scenarios, and sustained concurrent operations.
"""

import pytest
import pandas as pd
import numpy as np
import time
import threading
import queue
import concurrent.futures
import multiprocessing
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os
import psutil
import gc
import tempfile
import json
import socket
import hashlib
from typing import List, Dict, Any, Callable, Generator, Tuple
import statistics
from dataclasses import dataclass
import itertools
import random
from collections import defaultdict, deque

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.conftest import time_function, measure_memory_usage

# Test if additional modules are available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.concurrency,
    pytest.mark.throughput,
    pytest.mark.large,
    pytest.mark.stress
]

@dataclass
class ConcurrencyMetrics:
    """Container for concurrency performance metrics."""
    concurrent_operations: int
    total_operations_completed: int
    successful_operations: int
    failed_operations: int
    operations_per_second: float
    requests_per_second: float
    average_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    throughput_mb_per_sec: float
    error_rate_percent: float
    concurrency_efficiency: float
    cpu_utilization_percent: float
    memory_peak_mb: float
    memory_average_mb: float
    test_duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'concurrent_operations': self.concurrent_operations,
            'total_operations_completed': self.total_operations_completed,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'operations_per_second': self.operations_per_second,
            'requests_per_second': self.requests_per_second,
            'average_response_time_ms': self.average_response_time_ms,
            'median_response_time_ms': self.median_response_time_ms,
            'p95_response_time_ms': self.p95_response_time_ms,
            'p99_response_time_ms': self.p99_response_time_ms,
            'max_response_time_ms': self.max_response_time_ms,
            'min_response_time_ms': self.min_response_time_ms,
            'throughput_mb_per_sec': self.throughput_mb_per_sec,
            'error_rate_percent': self.error_rate_percent,
            'concurrency_efficiency': self.concurrency_efficiency,
            'cpu_utilization_percent': self.cpu_utilization_percent,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_average_mb': self.memory_average_mb,
            'test_duration_seconds': self.test_duration_seconds
        }


class HighVolumeConcurrencyRunner:
    """Runner for high volume concurrency benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring_thread = None
        self.monitoring_active = False
        self.cpu_samples = []
        self.memory_samples = []
        
    def start_monitoring(self):
        """Start system resource monitoring."""
        self.monitoring_active = True
        self.cpu_samples = []
        self.memory_samples = []
        
        def monitor():
            while self.monitoring_active:
                try:
                    cpu_percent = self.process.cpu_percent(interval=None)
                    memory_mb = self.process.memory_info().rss / 1024**2
                    
                    self.cpu_samples.append(cpu_percent)
                    self.memory_samples.append(memory_mb)
                    
                    time.sleep(0.25)  # Sample every 250ms
                except:
                    break
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop system resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
    
    def measure_high_volume_concurrency(
        self,
        operation_func: Callable,
        operation_args: List[Any],
        max_concurrent: int,
        timeout_seconds: int = 300
    ) -> ConcurrencyMetrics:
        """Measure high volume concurrency performance."""
        
        # Start monitoring
        self.start_monitoring()
        self.process.cpu_percent()  # Reset CPU monitoring
        gc.collect()
        
        start_time = time.perf_counter()
        response_times = []
        successful_ops = 0
        failed_ops = 0
        total_data_bytes = 0
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                # Submit all operations
                futures = []
                submission_start = time.perf_counter()
                
                for args in operation_args:
                    future = executor.submit(self._timed_concurrent_operation, operation_func, args)
                    futures.append(future)
                
                submission_time = time.perf_counter() - submission_start
                
                # Collect results
                for future in concurrent.futures.as_completed(futures, timeout=timeout_seconds):
                    try:
                        response_time_ms, success, data_bytes = future.result()
                        response_times.append(response_time_ms)
                        total_data_bytes += data_bytes
                        
                        if success:
                            successful_ops += 1
                        else:
                            failed_ops += 1
                            
                    except Exception as e:
                        failed_ops += 1
                        print(f"Error in concurrent operation: {e}")
        
        except concurrent.futures.TimeoutError:
            print(f"Timeout reached for high volume concurrency test after {timeout_seconds}s")
            failed_ops += len(operation_args) - successful_ops - failed_ops
        
        end_time = time.perf_counter()
        test_duration = end_time - start_time
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Calculate metrics
        total_ops = successful_ops + failed_ops
        ops_per_second = total_ops / test_duration if test_duration > 0 else 0
        requests_per_second = successful_ops / test_duration if test_duration > 0 else 0
        error_rate = (failed_ops / total_ops * 100) if total_ops > 0 else 0
        throughput_mb_per_sec = (total_data_bytes / (1024**2)) / test_duration if test_duration > 0 else 0
        
        # Concurrency efficiency (how well we utilized the concurrent capacity)
        theoretical_max_ops = max_concurrent * test_duration  # If each operation took 1 second
        actual_ops = total_ops
        concurrency_efficiency = (actual_ops / theoretical_max_ops * 100) if theoretical_max_ops > 0 else 0
        
        # Response time metrics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = median_response_time = 0
            p95_response_time = p99_response_time = 0
            max_response_time = min_response_time = 0
        
        # System resource metrics
        peak_memory = max(self.memory_samples) if self.memory_samples else 0
        avg_memory = statistics.mean(self.memory_samples) if self.memory_samples else 0
        avg_cpu = statistics.mean(self.cpu_samples) if self.cpu_samples else 0
        
        return ConcurrencyMetrics(
            concurrent_operations=max_concurrent,
            total_operations_completed=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            operations_per_second=ops_per_second,
            requests_per_second=requests_per_second,
            average_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            max_response_time_ms=max_response_time,
            min_response_time_ms=min_response_time,
            throughput_mb_per_sec=throughput_mb_per_sec,
            error_rate_percent=error_rate,
            concurrency_efficiency=concurrency_efficiency,
            cpu_utilization_percent=avg_cpu,
            memory_peak_mb=peak_memory,
            memory_average_mb=avg_memory,
            test_duration_seconds=test_duration
        )
    
    def _timed_concurrent_operation(self, operation_func: Callable, args: Any) -> Tuple[float, bool, int]:
        """Wrapper to time concurrent operations and track data processed."""
        start = time.perf_counter()
        data_bytes = 0
        
        try:
            result = operation_func(args)
            success = True
            
            # Estimate data processed
            if isinstance(result, dict) and 'data_bytes' in result:
                data_bytes = result['data_bytes']
            elif isinstance(result, pd.DataFrame):
                data_bytes = result.memory_usage(deep=True).sum()
            elif isinstance(result, np.ndarray):
                data_bytes = result.nbytes
            elif isinstance(result, (list, tuple)):
                data_bytes = sys.getsizeof(result)
            elif hasattr(result, '__len__'):
                data_bytes = len(str(result))
                
        except Exception as e:
            result = None
            success = False
        
        end = time.perf_counter()
        response_time_ms = (end - start) * 1000
        
        return response_time_ms, success, data_bytes


class TestHighVolumeConcurrentOperations:
    """Test high volume concurrent operations."""
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    def test_massive_concurrent_data_processing(self, benchmark_result):
        """Test massive concurrent data processing operations."""
        benchmark_result.start()
        
        runner = HighVolumeConcurrencyRunner()
        
        def concurrent_data_processing(task_config):
            """Process data in concurrent manner."""
            task_id, data_size, operation_type = task_config
            
            # Generate data for processing
            if operation_type == 'numerical':
                data = np.random.randn(data_size, 20)
                
                # Numerical operations
                result = {
                    'task_id': task_id,
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'sum': np.sum(data),
                    'data_shape': data.shape,
                    'data_bytes': data.nbytes
                }
                
                # Additional computations
                if data_size > 1000:
                    result['correlation'] = np.corrcoef(data.T)
                    result['eigenvalues'] = np.linalg.eigvals(result['correlation'])
                
            elif operation_type == 'textual':
                # Text processing
                words = ['data', 'processing', 'concurrent', 'benchmark', 'performance', 'analysis']
                text_data = [' '.join(np.random.choice(words, 5)) for _ in range(data_size)]
                
                # Text operations
                total_words = sum(len(text.split()) for text in text_data)
                avg_length = np.mean([len(text) for text in text_data])
                unique_words = len(set(' '.join(text_data).split()))
                
                result = {
                    'task_id': task_id,
                    'texts_processed': len(text_data),
                    'total_words': total_words,
                    'avg_length': avg_length,
                    'unique_words': unique_words,
                    'data_bytes': sys.getsizeof(text_data)
                }
                
            elif operation_type == 'mixed':
                # Mixed data processing
                numerical_data = np.random.randn(data_size // 2, 10)
                categorical_data = np.random.choice(['A', 'B', 'C', 'D'], data_size)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'numerical': numerical_data.flatten()[:data_size],
                    'categorical': categorical_data
                })
                
                # Mixed operations
                num_stats = df['numerical'].describe()
                cat_counts = df['categorical'].value_counts()
                grouped = df.groupby('categorical')['numerical'].mean()
                
                result = {
                    'task_id': task_id,
                    'rows_processed': len(df),
                    'num_stats': num_stats.to_dict(),
                    'cat_distribution': cat_counts.to_dict(),
                    'grouped_means': grouped.to_dict(),
                    'data_bytes': df.memory_usage(deep=True).sum()
                }
                
                del df
            
            # Simulate processing time variation
            processing_delay = 0.001 + (task_id % 10) * 0.0005
            time.sleep(processing_delay)
            
            return result
        
        # Generate massive concurrent task configurations
        concurrent_tasks = []
        operation_types = ['numerical', 'textual', 'mixed']
        data_sizes = [500, 1000, 2000, 5000]
        
        # Generate many concurrent tasks
        for task_id in range(1000):  # 1000 concurrent tasks
            operation_type = operation_types[task_id % len(operation_types)]
            data_size = data_sizes[task_id % len(data_sizes)]
            concurrent_tasks.append((task_id, data_size, operation_type))
        
        # Test different concurrency levels
        concurrency_levels = [50, 100, 200, 500]
        concurrency_results = {}
        
        for max_concurrent in concurrency_levels:
            print(f"Testing concurrency level: {max_concurrent}")
            
            # Use subset of tasks for each concurrency level to manage execution time
            task_subset = concurrent_tasks[:max_concurrent * 2]  # 2x tasks to workers ratio
            
            metrics = runner.measure_high_volume_concurrency(
                concurrent_data_processing,
                task_subset,
                max_concurrent=max_concurrent,
                timeout_seconds=180  # 3 minutes timeout
            )
            
            concurrency_results[f'{max_concurrent}_concurrent'] = metrics.to_dict()
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'massive_concurrent_data_processing',
            'total_task_configurations': len(concurrent_tasks),
            'concurrency_levels_tested': concurrency_levels,
            'operation_types': operation_types,
            'concurrency_results': concurrency_results
        }
        
        # Assertions for massive concurrent processing
        for level, result in concurrency_results.items():
            assert result['error_rate_percent'] < 30, f"Error rate too high at {level}: {result['error_rate_percent']:.1f}%"
            assert result['operations_per_second'] > 5, f"Throughput too low at {level}: {result['operations_per_second']:.1f} ops/sec"
            assert result['concurrency_efficiency'] > 10, f"Concurrency efficiency too low at {level}: {result['concurrency_efficiency']:.1f}%"
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    def test_high_frequency_short_operations(self, benchmark_result):
        """Test high frequency short concurrent operations."""
        benchmark_result.start()
        
        runner = HighVolumeConcurrencyRunner()
        
        def short_operation(operation_config):
            """Perform short, quick operations at high frequency."""
            operation_id, operation_type, complexity = operation_config
            
            if operation_type == 'computation':
                # Quick mathematical computations
                x = operation_id
                result = 0
                for i in range(complexity):
                    result += x ** 2 + x + 1
                    x = result % 1000
                
                return {
                    'operation_id': operation_id,
                    'result': result,
                    'iterations': complexity,
                    'data_bytes': sys.getsizeof(result)
                }
                
            elif operation_type == 'hash':
                # Hash operations
                data = f"operation_{operation_id}_data_{'x' * complexity}"
                
                hashes = []
                for algorithm in ['md5', 'sha1', 'sha256']:
                    hash_obj = hashlib.new(algorithm)
                    hash_obj.update(data.encode())
                    hashes.append(hash_obj.hexdigest())
                
                return {
                    'operation_id': operation_id,
                    'hashes': hashes,
                    'data_length': len(data),
                    'data_bytes': len(data.encode())
                }
                
            elif operation_type == 'list_ops':
                # List operations
                data_list = list(range(complexity * 10))
                
                # Various list operations
                sorted_list = sorted(data_list, reverse=True)
                filtered_list = [x for x in data_list if x % 2 == 0]
                mapped_list = [x * 2 for x in data_list[:100]]  # Limit to prevent excessive computation
                
                return {
                    'operation_id': operation_id,
                    'original_length': len(data_list),
                    'sorted_length': len(sorted_list),
                    'filtered_length': len(filtered_list),
                    'mapped_length': len(mapped_list),
                    'data_bytes': sys.getsizeof(data_list) + sys.getsizeof(sorted_list)
                }
        
        # Generate high frequency operation configurations
        operation_configs = []
        operation_types = ['computation', 'hash', 'list_ops']
        complexities = [10, 25, 50, 100]
        
        # Generate many short operations
        for operation_id in range(2000):  # 2000 short operations
            operation_type = operation_types[operation_id % len(operation_types)]
            complexity = complexities[operation_id % len(complexities)]
            operation_configs.append((operation_id, operation_type, complexity))
        
        # Test with high concurrency for short operations
        max_concurrent = min(300, len(operation_configs))
        
        metrics = runner.measure_high_volume_concurrency(
            short_operation,
            operation_configs,
            max_concurrent=max_concurrent,
            timeout_seconds=120  # 2 minutes timeout
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'high_frequency_short_operations',
            'total_operations': len(operation_configs),
            'max_concurrent': max_concurrent,
            'operation_types': operation_types,
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for high frequency operations
        assert metrics.error_rate_percent < 10, f"Error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 50, f"Throughput too low for short operations: {metrics.operations_per_second:.1f} ops/sec"
        assert metrics.average_response_time_ms < 100, f"Average response time too high: {metrics.average_response_time_ms:.1f}ms"
        assert metrics.p95_response_time_ms < 500, f"P95 response time too high: {metrics.p95_response_time_ms:.1f}ms"
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_concurrent_ml_pipeline_operations(self, benchmark_result):
        """Test concurrent ML pipeline operations."""
        benchmark_result.start()
        
        runner = HighVolumeConcurrencyRunner()
        
        # Prepare base models and data
        base_models = {}
        model_types = ['rf_fast', 'rf_balanced', 'rf_accurate']
        
        for model_type in model_types:
            if model_type == 'rf_fast':
                model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=1)
            elif model_type == 'rf_balanced':
                model = RandomForestClassifier(n_estimators=25, max_depth=8, random_state=42, n_jobs=1)
            else:  # rf_accurate
                model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=1)
            
            # Train model
            X, y = make_classification(n_samples=2000, n_features=15, n_classes=2, random_state=42)
            model.fit(X, y)
            
            base_models[model_type] = {
                'model': model,
                'feature_count': X.shape[1],
                'training_samples': X.shape[0]
            }
        
        def ml_pipeline_operation(pipeline_config):
            """Perform ML pipeline operations concurrently."""
            operation_id, model_type, operation_type, data_config = pipeline_config
            
            model_info = base_models[model_type]
            model = model_info['model']
            feature_count = model_info['feature_count']
            
            if operation_type == 'batch_prediction':
                # Batch prediction operations
                batch_size, num_batches = data_config
                
                predictions = []
                probabilities = []
                processing_times = []
                
                for batch_idx in range(num_batches):
                    # Generate batch data
                    batch_data = np.random.randn(batch_size, feature_count)
                    
                    # Time individual batch prediction
                    batch_start = time.perf_counter()
                    batch_predictions = model.predict(batch_data)
                    batch_probabilities = model.predict_proba(batch_data)
                    batch_time = time.perf_counter() - batch_start
                    
                    predictions.extend(batch_predictions)
                    probabilities.extend(batch_probabilities.tolist())
                    processing_times.append(batch_time)
                
                result = {
                    'operation_id': operation_id,
                    'model_type': model_type,
                    'total_predictions': len(predictions),
                    'batch_count': num_batches,
                    'avg_batch_time': np.mean(processing_times),
                    'predictions_per_second': len(predictions) / sum(processing_times) if sum(processing_times) > 0 else 0,
                    'data_bytes': batch_size * num_batches * feature_count * 8  # Rough estimate
                }
                
            elif operation_type == 'feature_processing':
                # Feature processing operations
                sample_count, processing_steps = data_config
                
                # Generate raw data
                raw_data = np.random.randn(sample_count, feature_count)
                processed_data = raw_data.copy()
                
                processing_results = []
                
                for step in range(processing_steps):
                    step_start = time.perf_counter()
                    
                    if step % 3 == 0:
                        # Normalization
                        processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
                        step_type = 'normalization'
                    elif step % 3 == 1:
                        # Feature scaling
                        processed_data = (processed_data - np.min(processed_data)) / (np.max(processed_data) - np.min(processed_data))
                        step_type = 'scaling'
                    else:
                        # Feature transformation
                        processed_data = np.log1p(np.abs(processed_data))
                        step_type = 'transformation'
                    
                    step_time = time.perf_counter() - step_start
                    processing_results.append({'step': step, 'type': step_type, 'time': step_time})
                
                # Final prediction
                final_predictions = model.predict(processed_data)
                
                result = {
                    'operation_id': operation_id,
                    'model_type': model_type,
                    'samples_processed': sample_count,
                    'processing_steps': len(processing_results),
                    'final_predictions': len(final_predictions),
                    'total_processing_time': sum(r['time'] for r in processing_results),
                    'data_bytes': raw_data.nbytes + processed_data.nbytes
                }
                
            elif operation_type == 'model_evaluation':
                # Model evaluation operations
                test_size, evaluation_metrics = data_config
                
                # Generate test data
                X_test = np.random.randn(test_size, feature_count)
                y_test = np.random.randint(0, 2, test_size)
                
                # Evaluate model
                eval_start = time.perf_counter()
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)
                eval_time = time.perf_counter() - eval_start
                
                # Calculate metrics
                accuracy = np.mean(predictions == y_test)
                prediction_distribution = np.bincount(predictions)
                
                # Additional evaluations if requested
                evaluation_results = {'accuracy': accuracy}
                
                if 'feature_importance' in evaluation_metrics:
                    feature_importance = model.feature_importances_
                    evaluation_results['feature_importance'] = feature_importance.tolist()
                
                result = {
                    'operation_id': operation_id,
                    'model_type': model_type,
                    'test_samples': test_size,
                    'evaluation_time': eval_time,
                    'evaluation_results': evaluation_results,
                    'prediction_distribution': prediction_distribution.tolist(),
                    'data_bytes': X_test.nbytes + y_test.nbytes
                }
            
            return result
        
        # Generate ML pipeline operation configurations
        pipeline_configs = []
        
        # Batch prediction configs
        for i in range(100):
            model_type = model_types[i % len(model_types)]
            batch_size = [50, 100, 200][i % 3]
            num_batches = [5, 10, 15][i % 3]
            pipeline_configs.append((i, model_type, 'batch_prediction', (batch_size, num_batches)))
        
        # Feature processing configs
        for i in range(100, 200):
            model_type = model_types[i % len(model_types)]
            sample_count = [200, 500, 1000][i % 3]
            processing_steps = [3, 5, 8][i % 3]
            pipeline_configs.append((i, model_type, 'feature_processing', (sample_count, processing_steps)))
        
        # Model evaluation configs
        for i in range(200, 300):
            model_type = model_types[i % len(model_types)]
            test_size = [300, 500, 1000][i % 3]
            evaluation_metrics = [['accuracy'], ['accuracy', 'feature_importance']][i % 2]
            pipeline_configs.append((i, model_type, 'model_evaluation', (test_size, evaluation_metrics)))
        
        # Test concurrent ML pipeline operations
        max_concurrent = min(50, len(pipeline_configs) // 4)
        
        metrics = runner.measure_high_volume_concurrency(
            ml_pipeline_operation,
            pipeline_configs,
            max_concurrent=max_concurrent,
            timeout_seconds=300  # 5 minutes timeout
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'concurrent_ml_pipeline_operations',
            'total_pipeline_operations': len(pipeline_configs),
            'max_concurrent': max_concurrent,
            'model_types': model_types,
            'operation_types': ['batch_prediction', 'feature_processing', 'model_evaluation'],
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for ML pipeline operations
        assert metrics.error_rate_percent < 15, f"ML pipeline error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 1, f"ML pipeline throughput too low: {metrics.operations_per_second:.1f} ops/sec"
        assert metrics.p95_response_time_ms < 30000, f"P95 response time too high: {metrics.p95_response_time_ms:.1f}ms"


class TestSustainedHighThroughput:
    """Test sustained high throughput scenarios."""
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    def test_sustained_concurrent_load(self, benchmark_result):
        """Test sustained concurrent load over extended periods."""
        benchmark_result.start()
        
        runner = HighVolumeConcurrencyRunner()
        
        def sustained_operation(operation_config):
            """Perform operations for sustained load testing."""
            operation_id, phase, intensity, duration = operation_config
            
            phase_start = time.perf_counter()
            phase_end = phase_start + duration
            
            operations_performed = 0
            data_processed = 0
            
            while time.perf_counter() < phase_end:
                # Adjust operation complexity based on phase and intensity
                if phase == 'ramp_up':
                    # Gradually increasing complexity
                    elapsed = time.perf_counter() - phase_start
                    current_intensity = int(intensity * (elapsed / duration))
                elif phase == 'peak':
                    # Maximum intensity
                    current_intensity = intensity
                elif phase == 'steady':
                    # Steady state intensity
                    current_intensity = int(intensity * 0.7)
                else:  # ramp_down
                    # Gradually decreasing complexity
                    elapsed = time.perf_counter() - phase_start
                    current_intensity = int(intensity * (1 - elapsed / duration))
                
                current_intensity = max(1, current_intensity)  # Ensure minimum intensity
                
                # Perform operation based on current intensity
                data = np.random.randn(current_intensity * 100, 10)
                
                # Data processing operations
                result = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'sum': np.sum(data)
                }
                
                # Additional processing for higher intensities
                if current_intensity > 5:
                    correlation = np.corrcoef(data.T)
                    result['correlation_trace'] = np.trace(correlation)
                
                operations_performed += 1
                data_processed += data.nbytes
                
                # Simulate realistic operation timing
                operation_delay = 0.01 + (operations_performed % 10) * 0.001
                time.sleep(operation_delay)
            
            actual_duration = time.perf_counter() - phase_start
            
            return {
                'operation_id': operation_id,
                'phase': phase,
                'planned_duration': duration,
                'actual_duration': actual_duration,
                'operations_performed': operations_performed,
                'operations_per_second': operations_performed / actual_duration if actual_duration > 0 else 0,
                'data_bytes': data_processed
            }
        
        # Define sustained load phases
        sustained_phases = [
            # (operation_id, phase, intensity, duration_seconds)
            (1, 'ramp_up', 10, 30),      # 30s ramp up to intensity 10
            (2, 'peak', 15, 45),         # 45s peak load at intensity 15
            (3, 'steady', 12, 60),       # 60s steady state at intensity 12
            (4, 'peak', 18, 30),         # 30s second peak at intensity 18
            (5, 'steady', 10, 45),       # 45s sustained at intensity 10
            (6, 'ramp_down', 15, 20),    # 20s ramp down from intensity 15
        ]
        
        # Add more concurrent phases for higher load
        additional_phases = []
        for i in range(10):  # 10 additional concurrent sustained operations
            phase_type = ['steady', 'peak'][i % 2]
            intensity = 8 + (i % 5)
            duration = 40 + (i % 3) * 10
            additional_phases.append((10 + i, phase_type, intensity, duration))
        
        all_phases = sustained_phases + additional_phases
        
        # Test sustained concurrent load
        max_concurrent = min(20, len(all_phases))
        
        metrics = runner.measure_high_volume_concurrency(
            sustained_operation,
            all_phases,
            max_concurrent=max_concurrent,
            timeout_seconds=600  # 10 minutes timeout
        )
        
        benchmark_result.stop()
        
        total_planned_duration = sum(duration for _, _, _, duration in sustained_phases)
        
        benchmark_result.metadata = {
            'test_type': 'sustained_concurrent_load',
            'total_phases': len(all_phases),
            'main_phases': len(sustained_phases),
            'additional_phases': len(additional_phases),
            'max_concurrent': max_concurrent,
            'total_planned_duration_seconds': total_planned_duration,
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for sustained load
        assert metrics.error_rate_percent < 20, f"Sustained load error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 2, f"Sustained throughput too low: {metrics.operations_per_second:.1f} ops/sec"
        assert metrics.test_duration_seconds > 30, f"Test duration too short: {metrics.test_duration_seconds:.1f}s"
        assert metrics.concurrency_efficiency > 5, f"Concurrency efficiency too low: {metrics.concurrency_efficiency:.1f}%"
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    def test_burst_load_handling(self, benchmark_result):
        """Test system behavior under burst load scenarios."""
        benchmark_result.start()
        
        runner = HighVolumeConcurrencyRunner()
        
        def burst_operation(burst_config):
            """Perform operations during burst load scenarios."""
            operation_id, burst_type, burst_intensity, burst_duration = burst_config
            
            burst_start = time.perf_counter()
            burst_end = burst_start + burst_duration
            
            operations_in_burst = 0
            burst_data_processed = 0
            
            if burst_type == 'cpu_burst':
                # CPU-intensive burst
                while time.perf_counter() < burst_end:
                    # Intensive computation
                    matrix_size = min(200, burst_intensity * 10)
                    matrix = np.random.randn(matrix_size, matrix_size)
                    
                    # Matrix operations
                    result = np.dot(matrix, matrix.T)
                    eigenvals = np.linalg.eigvals(result)
                    
                    operations_in_burst += 1
                    burst_data_processed += matrix.nbytes + result.nbytes
                    
                    # Small delay to prevent system overload
                    time.sleep(0.005)
            
            elif burst_type == 'memory_burst':
                # Memory-intensive burst
                allocated_arrays = []
                
                while time.perf_counter() < burst_end:
                    # Allocate memory in bursts
                    array_size = burst_intensity * 1000
                    array = np.random.randn(array_size)
                    
                    # Process array
                    processed = array * 2 + np.random.randn(array_size)
                    result = np.mean(processed)
                    
                    # Keep some arrays to maintain memory pressure
                    if len(allocated_arrays) < 5:
                        allocated_arrays.append(array)
                    else:
                        # Replace oldest array
                        allocated_arrays[operations_in_burst % 5] = array
                    
                    operations_in_burst += 1
                    burst_data_processed += array.nbytes + processed.nbytes
                    
                    time.sleep(0.01)
                
                # Cleanup
                del allocated_arrays
            
            elif burst_type == 'io_burst':
                # I/O simulation burst
                while time.perf_counter() < burst_end:
                    # Simulate I/O operations
                    data_size = burst_intensity * 500
                    data = np.random.randn(data_size, 5)
                    
                    # Simulate file-like operations
                    temp_file_content = data.tobytes()
                    
                    # Simulate processing
                    processed_data = np.frombuffer(temp_file_content, dtype=np.float64).reshape(-1, 5)
                    summary = {
                        'shape': processed_data.shape,
                        'mean': np.mean(processed_data),
                        'std': np.std(processed_data)
                    }
                    
                    operations_in_burst += 1
                    burst_data_processed += len(temp_file_content)
                    
                    # Simulate I/O delay
                    time.sleep(0.008)
            
            actual_burst_duration = time.perf_counter() - burst_start
            
            return {
                'operation_id': operation_id,
                'burst_type': burst_type,
                'burst_intensity': burst_intensity,
                'planned_duration': burst_duration,
                'actual_duration': actual_burst_duration,
                'operations_in_burst': operations_in_burst,
                'burst_ops_per_second': operations_in_burst / actual_burst_duration if actual_burst_duration > 0 else 0,
                'data_bytes': burst_data_processed
            }
        
        # Generate burst load configurations
        burst_configs = []
        burst_types = ['cpu_burst', 'memory_burst', 'io_burst']
        
        # Multiple burst scenarios
        for burst_id in range(60):  # 60 burst operations
            burst_type = burst_types[burst_id % len(burst_types)]
            burst_intensity = 5 + (burst_id % 10)  # Intensity from 5 to 14
            burst_duration = 10 + (burst_id % 5) * 5  # Duration from 10 to 30 seconds
            
            burst_configs.append((burst_id, burst_type, burst_intensity, burst_duration))
        
        # Test burst load handling
        max_concurrent = min(25, len(burst_configs) // 3)
        
        metrics = runner.measure_high_volume_concurrency(
            burst_operation,
            burst_configs,
            max_concurrent=max_concurrent,
            timeout_seconds=400  # 6.67 minutes timeout
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'burst_load_handling',
            'total_burst_operations': len(burst_configs),
            'max_concurrent': max_concurrent,
            'burst_types': burst_types,
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for burst load handling
        assert metrics.error_rate_percent < 35, f"Burst load error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 1, f"Burst load throughput too low: {metrics.operations_per_second:.1f} ops/sec"
        assert metrics.p99_response_time_ms < 60000, f"P99 response time too high: {metrics.p99_response_time_ms:.1f}ms"


@pytest.mark.benchmark
@pytest.mark.stress
@pytest.mark.extreme
class TestExtremeScalabilityConcurrency:
    """Test extreme scalability and concurrency limits."""
    
    @pytest.mark.stress
    @pytest.mark.extreme
    def test_scalability_breaking_point(self, benchmark_result):
        """Test system behavior at scalability breaking points."""
        benchmark_result.start()
        
        runner = HighVolumeConcurrencyRunner()
        
        def scalability_test_operation(scale_config):
            """Perform operations to test scalability limits."""
            operation_id, scale_factor, operation_complexity = scale_config
            
            # Scale operation based on scale factor
            data_size = scale_factor * 100
            computation_rounds = scale_factor * 2
            
            total_computations = 0
            total_data_bytes = 0
            
            for round_idx in range(computation_rounds):
                # Generate data proportional to scale
                data = np.random.randn(data_size, 10)
                total_data_bytes += data.nbytes
                
                # Perform computations with increasing complexity
                if operation_complexity == 'light':
                    result = np.mean(data) + np.std(data)
                    total_computations += 2
                
                elif operation_complexity == 'medium':
                    # More complex operations
                    normalized = (data - np.mean(data)) / np.std(data)
                    correlation = np.corrcoef(data.T)
                    result = np.trace(correlation)
                    total_computations += 4
                    total_data_bytes += normalized.nbytes
                
                elif operation_complexity == 'heavy':
                    # Heavy computations
                    if data.shape[0] > 500:  # Limit size for heavy operations
                        data = data[:500]
                    
                    covariance = np.cov(data.T)
                    eigenvals, eigenvecs = np.linalg.eigh(covariance)
                    reconstruction = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                    result = np.sum(reconstruction)
                    total_computations += 6
                    total_data_bytes += covariance.nbytes + eigenvecs.nbytes
                
                # Simulate processing delay proportional to scale
                processing_delay = 0.001 * scale_factor
                time.sleep(processing_delay)
            
            return {
                'operation_id': operation_id,
                'scale_factor': scale_factor,
                'operation_complexity': operation_complexity,
                'total_computations': total_computations,
                'computation_rounds': computation_rounds,
                'data_bytes': total_data_bytes
            }
        
        # Generate scalability test configurations
        scale_configs = []
        complexities = ['light', 'medium', 'heavy']
        
        # Test different scale factors to find breaking point
        scale_factors = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]
        
        for scale_factor in scale_factors:
            for complexity in complexities:
                # Multiple operations per scale/complexity combination
                for op_id in range(5):
                    operation_id = len(scale_configs)
                    scale_configs.append((operation_id, scale_factor, complexity))
        
        # Test increasing concurrency levels to find breaking point
        concurrency_breaking_points = {}
        max_concurrency_levels = [10, 25, 50, 75, 100, 150, 200]
        
        for max_concurrent in max_concurrency_levels:
            print(f"Testing scalability at concurrency level: {max_concurrent}")
            
            # Use subset of configurations for each test
            test_configs = scale_configs[:max_concurrent * 2]  # 2x tasks to workers
            
            try:
                metrics = runner.measure_high_volume_concurrency(
                    scalability_test_operation,
                    test_configs,
                    max_concurrent=max_concurrent,
                    timeout_seconds=300  # 5 minutes timeout
                )
                
                concurrency_breaking_points[f'{max_concurrent}_concurrent'] = {
                    'metrics': metrics.to_dict(),
                    'breaking_point_reached': False,
                    'test_completed': True
                }
                
                # Check if we're approaching breaking point
                if (metrics.error_rate_percent > 50 or 
                    metrics.operations_per_second < 1 or 
                    metrics.p95_response_time_ms > 120000):  # 2 minutes
                    
                    concurrency_breaking_points[f'{max_concurrent}_concurrent']['breaking_point_reached'] = True
                    print(f"Breaking point indicators detected at {max_concurrent} concurrency")
                
            except Exception as e:
                concurrency_breaking_points[f'{max_concurrent}_concurrent'] = {
                    'metrics': None,
                    'breaking_point_reached': True,
                    'test_completed': False,
                    'error': str(e)
                }
                print(f"Test failed at {max_concurrent} concurrency: {e}")
                break  # Stop testing higher concurrency levels
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'scalability_breaking_point',
            'scale_factors_tested': scale_factors,
            'complexity_levels': complexities,
            'max_concurrency_levels_tested': max_concurrency_levels,
            'total_scale_configurations': len(scale_configs),
            'concurrency_breaking_points': concurrency_breaking_points
        }
        
        # Analyze breaking points
        successful_tests = [
            level for level, result in concurrency_breaking_points.items() 
            if result.get('test_completed', False) and not result.get('breaking_point_reached', False)
        ]
        
        # More lenient assertions for breaking point tests
        assert len(successful_tests) > 0, "No concurrency levels completed successfully"
        
        # Check that at least some level worked reasonably well
        best_performing_test = None
        best_ops_per_sec = 0
        
        for level, result in concurrency_breaking_points.items():
            if result.get('metrics') and result['metrics'].get('operations_per_second', 0) > best_ops_per_sec:
                best_ops_per_sec = result['metrics']['operations_per_second']
                best_performing_test = level
        
        assert best_ops_per_sec > 0.5, f"Best performing test had too low throughput: {best_ops_per_sec:.3f} ops/sec"
        
        print(f"Best performing concurrency level: {best_performing_test} with {best_ops_per_sec:.2f} ops/sec")
