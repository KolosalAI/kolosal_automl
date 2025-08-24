# ---------------------------------------------------------------------
# tests/benchmark/test_large_scale_benchmark.py - Large Scale Performance Benchmarks
# ---------------------------------------------------------------------
"""
Large scale benchmark tests for testing system performance under significantly increased load.
Tests processing of large datasets, high-volume operations, and extreme concurrency scenarios.
"""

import pytest
import pandas as pd
import numpy as np
import time
import threading
import queue
import concurrent.futures
import multiprocessing
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os
import psutil
import gc
import tempfile
import csv
import json
from typing import List, Dict, Any, Callable
import statistics
from dataclasses import dataclass
import asyncio
import aiofiles

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.conftest import time_function, measure_memory_usage

# Test if ML modules are available
try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.large,
    pytest.mark.stress,
    pytest.mark.concurrency,
    pytest.mark.throughput
]

@dataclass
class LargeScaleMetrics:
    """Container for large-scale performance metrics."""
    operations_per_second: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    memory_peak_mb: float
    memory_average_mb: float
    cpu_utilization_percent: float
    duration_seconds: float
    error_rate_percent: float
    throughput_mb_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'operations_per_second': self.operations_per_second,
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'average_latency_ms': self.average_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_average_mb': self.memory_average_mb,
            'cpu_utilization_percent': self.cpu_utilization_percent,
            'duration_seconds': self.duration_seconds,
            'error_rate_percent': self.error_rate_percent,
            'throughput_mb_per_second': self.throughput_mb_per_second
        }


class LargeScaleBenchmarkRunner:
    """Runner for large-scale performance benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def create_large_dataset(self, rows: int, cols: int, include_nulls: bool = True) -> pd.DataFrame:
        """Create a large dataset for testing."""
        np.random.seed(42)  # For reproducible results
        
        data = {}
        
        # Numeric columns
        for i in range(cols // 3):
            data[f'numeric_{i}'] = np.random.randn(rows)
            if include_nulls:
                # Add some null values
                null_indices = np.random.choice(rows, size=rows // 100, replace=False)
                data[f'numeric_{i}'][null_indices] = np.nan
        
        # Categorical columns
        categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for i in range(cols // 3):
            data[f'categorical_{i}'] = np.random.choice(categories, rows)
            if include_nulls:
                null_indices = np.random.choice(rows, size=rows // 200, replace=False)
                for idx in null_indices:
                    data[f'categorical_{i}'][idx] = None
        
        # Text columns
        words = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew']
        for i in range(cols - 2 * (cols // 3)):
            data[f'text_{i}'] = [' '.join(np.random.choice(words, 3)) for _ in range(rows)]
            if include_nulls:
                null_indices = np.random.choice(rows, size=rows // 150, replace=False)
                for idx in null_indices:
                    data[f'text_{i}'][idx] = None
        
        return pd.DataFrame(data)
    
    def create_large_csv_file(self, filepath: str, rows: int, cols: int) -> str:
        """Create a large CSV file for testing I/O performance."""
        df = self.create_large_dataset(rows, cols)
        df.to_csv(filepath, index=False)
        return filepath
    
    def measure_large_scale_operation(
        self,
        operation_func: Callable,
        operation_args: List[Any],
        max_workers: int = None,
        timeout_seconds: int = 300
    ) -> LargeScaleMetrics:
        """Measure performance of large-scale operations."""
        
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 4) * 2)
        
        # Initialize monitoring
        self.process.cpu_percent()  # Reset CPU monitoring
        gc.collect()
        memory_before = self.process.memory_info().rss / 1024**2
        memory_samples = [memory_before]
        
        start_time = time.perf_counter()
        latencies = []
        successful_ops = 0
        failed_ops = 0
        total_bytes_processed = 0
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # Submit all operations
                for args in operation_args:
                    future = executor.submit(self._timed_operation_wrapper, operation_func, args)
                    futures.append(future)
                
                # Collect results
                for future in concurrent.futures.as_completed(futures, timeout=timeout_seconds):
                    try:
                        latency_ms, success, bytes_processed = future.result()
                        latencies.append(latency_ms)
                        total_bytes_processed += bytes_processed
                        
                        if success:
                            successful_ops += 1
                        else:
                            failed_ops += 1
                            
                        # Sample memory usage
                        current_memory = self.process.memory_info().rss / 1024**2
                        memory_samples.append(current_memory)
                        
                    except Exception as e:
                        failed_ops += 1
                        print(f"Error in large-scale operation: {e}")
        
        except concurrent.futures.TimeoutError:
            print(f"Timeout reached for large-scale operation after {timeout_seconds}s")
            failed_ops += len(operation_args) - successful_ops - failed_ops
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Final measurements
        cpu_usage = self.process.cpu_percent()
        memory_peak = max(memory_samples)
        memory_avg = statistics.mean(memory_samples)
        
        # Calculate metrics
        total_ops = successful_ops + failed_ops
        ops_per_second = total_ops / duration if duration > 0 else 0
        error_rate = (failed_ops / total_ops * 100) if total_ops > 0 else 0
        throughput_mb_per_sec = (total_bytes_processed / (1024**2)) / duration if duration > 0 else 0
        
        return LargeScaleMetrics(
            operations_per_second=ops_per_second,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            average_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            memory_peak_mb=memory_peak,
            memory_average_mb=memory_avg,
            cpu_utilization_percent=cpu_usage,
            duration_seconds=duration,
            error_rate_percent=error_rate,
            throughput_mb_per_second=throughput_mb_per_sec
        )
    
    def _timed_operation_wrapper(self, operation_func: Callable, args: Any) -> tuple:
        """Wrapper to time individual operations and track data processed."""
        start = time.perf_counter()
        bytes_processed = 0
        
        try:
            result = operation_func(args)
            success = True
            
            # Try to estimate bytes processed
            if isinstance(result, pd.DataFrame):
                bytes_processed = result.memory_usage(deep=True).sum()
            elif isinstance(result, dict) and 'bytes_processed' in result:
                bytes_processed = result['bytes_processed']
            elif hasattr(result, '__len__'):
                bytes_processed = len(str(result))  # Rough estimate
            
        except Exception as e:
            result = None
            success = False
            print(f"Operation failed: {e}")
        
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        
        return latency_ms, success, bytes_processed


class TestLargeDatasetProcessing:
    """Test processing of large datasets."""
    
    @pytest.mark.benchmark
    @pytest.mark.large
    def test_very_large_csv_processing(self, benchmark_result):
        """Test processing of very large CSV files."""
        benchmark_result.start()
        
        runner = LargeScaleBenchmarkRunner()
        
        # Create multiple large CSV files
        temp_dir = tempfile.mkdtemp()
        large_files = []
        
        # Create files of different sizes
        file_configs = [
            (50000, 20),   # 50K rows, 20 columns
            (100000, 15),  # 100K rows, 15 columns
            (200000, 10),  # 200K rows, 10 columns
            (300000, 8),   # 300K rows, 8 columns
        ]
        
        for i, (rows, cols) in enumerate(file_configs):
            filepath = os.path.join(temp_dir, f'large_dataset_{i}.csv')
            runner.create_large_csv_file(filepath, rows, cols)
            large_files.append(filepath)
        
        def process_large_csv(filepath):
            """Process a large CSV file with comprehensive operations."""
            df = pd.read_csv(filepath)
            
            # Comprehensive processing operations
            results = {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'dtypes': len(df.dtypes),
                'null_counts': df.isnull().sum().sum(),
                'bytes_processed': df.memory_usage(deep=True).sum()
            }
            
            # Statistical analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                results['numeric_stats'] = df[numeric_cols].describe().to_dict()
            
            # String operations
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                results['text_lengths'] = {col: df[col].astype(str).str.len().mean() for col in text_cols[:3]}
            
            # Data transformations
            if len(numeric_cols) > 0:
                # Normalize numeric columns
                scaler_cols = numeric_cols[:5]  # Limit to first 5 to avoid excessive computation
                df[scaler_cols] = (df[scaler_cols] - df[scaler_cols].mean()) / df[scaler_cols].std()
            
            # Group operations
            if len(df.columns) > 0:
                first_col = df.columns[0]
                if df[first_col].nunique() < 100:  # Only if reasonable number of groups
                    results['group_stats'] = df.groupby(first_col).size().to_dict()
            
            return results
        
        # Measure large-scale processing
        metrics = runner.measure_large_scale_operation(
            process_large_csv,
            large_files,
            max_workers=min(8, len(large_files)),  # Reasonable concurrency for I/O
            timeout_seconds=180  # 3 minutes timeout
        )
        
        benchmark_result.stop()
        
        # Cleanup
        for filepath in large_files:
            try:
                os.remove(filepath)
            except:
                pass
        os.rmdir(temp_dir)
        
        benchmark_result.metadata = {
            'test_type': 'very_large_csv_processing',
            'file_configurations': file_configs,
            'files_processed': len(large_files),
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for large-scale processing
        assert metrics.error_rate_percent < 5, f"Error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.1, f"Throughput too low: {metrics.operations_per_second:.3f} ops/sec"
        assert metrics.memory_peak_mb < 8000, f"Memory usage too high: {metrics.memory_peak_mb:.1f}MB"  # 8GB limit
    
    @pytest.mark.benchmark
    @pytest.mark.large
    def test_massive_in_memory_processing(self, benchmark_result):
        """Test processing of massive in-memory datasets."""
        benchmark_result.start()
        
        runner = LargeScaleBenchmarkRunner()
        
        # Create very large datasets
        dataset_configs = [
            (100000, 25),   # 100K rows, 25 columns
            (200000, 20),   # 200K rows, 20 columns
            (500000, 15),   # 500K rows, 15 columns
            (1000000, 10),  # 1M rows, 10 columns
        ]
        
        datasets = []
        for rows, cols in dataset_configs:
            df = runner.create_large_dataset(rows, cols, include_nulls=True)
            datasets.append(df)
        
        def process_massive_dataset(df):
            """Process a massive dataset with intensive operations."""
            results = {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'bytes_processed': df.memory_usage(deep=True).sum()
            }
            
            # Data cleaning operations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Fill missing values
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                
                # Outlier detection (using IQR method)
                Q1 = df[numeric_cols].quantile(0.25)
                Q3 = df[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                          (df[numeric_cols] > (Q3 + 1.5 * IQR))).sum().sum()
                results['outliers_detected'] = outliers
            
            # String processing
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                # Process first few text columns to avoid excessive computation
                for col in text_cols[:3]:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna('missing')
                        df[col] = df[col].astype(str).str.lower().str.strip()
            
            # Aggregation operations
            if len(numeric_cols) > 0:
                agg_results = df[numeric_cols].agg(['mean', 'std', 'min', 'max']).to_dict()
                results['aggregations'] = len(agg_results)
            
            # Complex calculations
            if len(numeric_cols) >= 2:
                # Correlation matrix for subset of columns
                corr_cols = numeric_cols[:10]  # Limit to first 10 columns
                corr_matrix = df[corr_cols].corr()
                results['correlations_calculated'] = corr_matrix.shape[0] * corr_matrix.shape[1]
            
            # Sorting operations
            if len(df) > 0:
                first_numeric_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
                sorted_df = df.sort_values(by=first_numeric_col)
                results['sorted_rows'] = len(sorted_df)
            
            return results
        
        # Measure massive dataset processing
        metrics = runner.measure_large_scale_operation(
            process_massive_dataset,
            datasets,
            max_workers=min(4, len(datasets)),  # Limited workers for memory-intensive operations
            timeout_seconds=300  # 5 minutes timeout
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'massive_in_memory_processing',
            'dataset_configurations': dataset_configs,
            'datasets_processed': len(datasets),
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for massive processing
        assert metrics.error_rate_percent < 10, f"Error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.05, f"Throughput too low: {metrics.operations_per_second:.3f} ops/sec"
        assert metrics.memory_peak_mb < 12000, f"Memory usage too high: {metrics.memory_peak_mb:.1f}MB"  # 12GB limit


class TestExtremeConcurrency:
    """Test extreme concurrency scenarios."""
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    def test_extreme_thread_concurrency(self, benchmark_result):
        """Test system behavior under extreme thread concurrency."""
        benchmark_result.start()
        
        runner = LargeScaleBenchmarkRunner()
        
        def cpu_intensive_task(task_data):
            """CPU-intensive task for extreme concurrency testing."""
            task_id, complexity = task_data
            
            # Prime number calculation
            def calculate_primes(n):
                primes = []
                for num in range(2, n):
                    for i in range(2, int(num ** 0.5) + 1):
                        if num % i == 0:
                            break
                    else:
                        primes.append(num)
                return primes
            
            # Mathematical computations
            result = {
                'task_id': task_id,
                'primes_count': len(calculate_primes(complexity)),
                'fibonacci': [1, 1],
                'bytes_processed': complexity * 8  # Rough estimate
            }
            
            # Fibonacci sequence
            for i in range(min(20, complexity // 100)):
                result['fibonacci'].append(result['fibonacci'][-1] + result['fibonacci'][-2])
            
            # Matrix operations
            if complexity > 500:
                matrix = np.random.randn(min(50, complexity // 100), min(50, complexity // 100))
                eigenvalues = np.linalg.eigvals(matrix)
                result['eigenvalues_count'] = len(eigenvalues)
                result['bytes_processed'] += matrix.nbytes
            
            return result
        
        # Test different levels of extreme concurrency
        concurrency_tests = [
            (50, 1000),   # 50 threads, complexity 1000
            (100, 800),   # 100 threads, complexity 800
            (200, 600),   # 200 threads, complexity 600
            (500, 400),   # 500 threads, complexity 400
        ]
        
        concurrency_results = {}
        
        for max_workers, task_complexity in concurrency_tests:
            print(f"Testing extreme concurrency: {max_workers} workers, complexity {task_complexity}")
            
            # Generate tasks
            tasks = [(i, task_complexity) for i in range(max_workers * 2)]  # More tasks than workers
            
            metrics = runner.measure_large_scale_operation(
                cpu_intensive_task,
                tasks,
                max_workers=max_workers,
                timeout_seconds=120  # 2 minutes timeout
            )
            
            concurrency_results[f'{max_workers}_workers'] = metrics.to_dict()
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'extreme_thread_concurrency',
            'concurrency_tests': concurrency_tests,
            'concurrency_results': concurrency_results
        }
        
        # Validate extreme concurrency results
        for test_name, result in concurrency_results.items():
            assert result['error_rate_percent'] < 20, f"High error rate in {test_name}: {result['error_rate_percent']:.1f}%"
            assert result['operations_per_second'] > 5, f"Low throughput in {test_name}: {result['operations_per_second']:.1f} ops/sec"
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_concurrent_ml_training(self, benchmark_result):
        """Test concurrent ML model training."""
        benchmark_result.start()
        
        runner = LargeScaleBenchmarkRunner()
        
        def train_ml_model(model_config):
            """Train an ML model with given configuration."""
            model_type, n_samples, n_features, model_params = model_config
            
            # Generate training data
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=2,
                random_state=42
            )
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create model
            if model_type == 'random_forest':
                model = RandomForestClassifier(**model_params, random_state=42)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(**model_params, random_state=42)
            elif model_type == 'logistic_regression':
                model = LogisticRegression(**model_params, random_state=42, max_iter=1000)
            
            # Train model
            start_train = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start_train
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            return {
                'model_type': model_type,
                'train_time_seconds': train_time,
                'train_score': train_score,
                'test_score': test_score,
                'n_samples': n_samples,
                'n_features': n_features,
                'bytes_processed': X.nbytes + y.nbytes
            }
        
        # Define multiple model configurations for concurrent training
        model_configs = [
            ('random_forest', 5000, 10, {'n_estimators': 50, 'max_depth': 10}),
            ('random_forest', 10000, 15, {'n_estimators': 30, 'max_depth': 8}),
            ('gradient_boosting', 3000, 8, {'n_estimators': 50, 'learning_rate': 0.1}),
            ('gradient_boosting', 7000, 12, {'n_estimators': 30, 'learning_rate': 0.05}),
            ('logistic_regression', 15000, 20, {'C': 1.0, 'solver': 'lbfgs'}),
            ('logistic_regression', 20000, 15, {'C': 0.5, 'solver': 'liblinear'}),
            ('random_forest', 8000, 12, {'n_estimators': 40, 'max_depth': 12}),
            ('gradient_boosting', 5000, 10, {'n_estimators': 40, 'learning_rate': 0.08}),
        ]
        
        # Test concurrent ML training
        metrics = runner.measure_large_scale_operation(
            train_ml_model,
            model_configs,
            max_workers=min(8, len(model_configs)),  # Reasonable concurrency for ML training
            timeout_seconds=300  # 5 minutes timeout
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'concurrent_ml_training',
            'model_configurations': model_configs,
            'models_trained': len(model_configs),
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for concurrent ML training
        assert metrics.error_rate_percent < 15, f"ML training error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.02, f"ML training throughput too low: {metrics.operations_per_second:.3f} ops/sec"
        assert metrics.memory_peak_mb < 16000, f"Memory usage too high for ML training: {metrics.memory_peak_mb:.1f}MB"


class TestHighVolumeOperations:
    """Test high-volume operations and data processing."""
    
    @pytest.mark.benchmark
    @pytest.mark.large
    def test_high_volume_data_generation(self, benchmark_result):
        """Test generation and processing of high-volume data."""
        benchmark_result.start()
        
        runner = LargeScaleBenchmarkRunner()
        
        def generate_and_process_data(generation_config):
            """Generate and process large amounts of data."""
            rows, cols, operation_type = generation_config
            
            start_time = time.perf_counter()
            
            if operation_type == 'structured':
                # Generate structured DataFrame
                df = runner.create_large_dataset(rows, cols)
                
                # Process data
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
                
                result = {
                    'shape': df.shape,
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    'processing_time': time.perf_counter() - start_time
                }
                
            elif operation_type == 'numerical':
                # Generate large numerical arrays
                arrays = []
                for _ in range(cols):
                    arr = np.random.randn(rows)
                    arrays.append(arr)
                
                # Process arrays
                processed_arrays = []
                for arr in arrays:
                    # Mathematical operations
                    normalized = (arr - np.mean(arr)) / np.std(arr)
                    squared = normalized ** 2
                    filtered = squared[squared > 0.5]
                    processed_arrays.append(filtered)
                
                result = {
                    'arrays_generated': len(arrays),
                    'total_elements': rows * cols,
                    'processed_elements': sum(len(arr) for arr in processed_arrays),
                    'processing_time': time.perf_counter() - start_time
                }
                
            elif operation_type == 'text':
                # Generate text data
                words = ['data', 'science', 'machine', 'learning', 'artificial', 'intelligence', 'algorithm', 'model']
                text_data = []
                
                for _ in range(rows):
                    sentence = ' '.join(np.random.choice(words, size=cols))
                    text_data.append(sentence)
                
                # Process text
                processed_text = []
                for text in text_data:
                    cleaned = text.lower().strip()
                    tokens = cleaned.split()
                    processed_text.append(tokens)
                
                result = {
                    'text_entries': len(text_data),
                    'total_tokens': sum(len(tokens) for tokens in processed_text),
                    'avg_tokens_per_entry': np.mean([len(tokens) for tokens in processed_text]),
                    'processing_time': time.perf_counter() - start_time
                }
            
            # Calculate bytes processed
            if 'memory_usage' in result:
                result['bytes_processed'] = result['memory_usage']
            else:
                result['bytes_processed'] = rows * cols * 8  # Rough estimate
            
            return result
        
        # High-volume generation configurations
        generation_configs = [
            (500000, 20, 'structured'),
            (1000000, 15, 'numerical'),
            (300000, 25, 'structured'),
            (2000000, 10, 'numerical'),
            (100000, 50, 'text'),
            (750000, 18, 'structured'),
        ]
        
        # Measure high-volume operations
        metrics = runner.measure_large_scale_operation(
            generate_and_process_data,
            generation_configs,
            max_workers=min(6, len(generation_configs)),
            timeout_seconds=240  # 4 minutes timeout
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'high_volume_data_generation',
            'generation_configurations': generation_configs,
            'operations_completed': len(generation_configs),
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for high-volume operations
        assert metrics.error_rate_percent < 10, f"High-volume generation error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.1, f"High-volume generation throughput too low: {metrics.operations_per_second:.3f} ops/sec"
        assert metrics.throughput_mb_per_second > 50, f"Data throughput too low: {metrics.throughput_mb_per_second:.1f} MB/sec"
    
    @pytest.mark.benchmark
    @pytest.mark.large
    def test_batch_processing_scalability(self, benchmark_result):
        """Test scalability of batch processing operations."""
        benchmark_result.start()
        
        runner = LargeScaleBenchmarkRunner()
        
        def process_data_batch(batch_config):
            """Process a batch of data with specified configuration."""
            batch_size, num_batches, operation_complexity = batch_config
            
            total_processed = 0
            batch_results = []
            
            for batch_idx in range(num_batches):
                # Generate batch data
                batch_data = np.random.randn(batch_size, operation_complexity)
                
                # Process batch based on complexity
                if operation_complexity <= 10:
                    # Simple operations
                    result = {
                        'mean': np.mean(batch_data),
                        'std': np.std(batch_data),
                        'sum': np.sum(batch_data)
                    }
                elif operation_complexity <= 20:
                    # Medium complexity operations
                    result = {
                        'correlation': np.corrcoef(batch_data.T),
                        'eigenvalues': np.linalg.eigvals(np.cov(batch_data.T)),
                        'percentiles': np.percentile(batch_data, [25, 50, 75, 95], axis=0)
                    }
                else:
                    # High complexity operations
                    U, s, Vt = np.linalg.svd(batch_data, full_matrices=False)
                    result = {
                        'svd_components': len(s),
                        'explained_variance': np.sum(s**2),
                        'rank': np.linalg.matrix_rank(batch_data)
                    }
                
                batch_results.append(result)
                total_processed += batch_size
            
            return {
                'batch_size': batch_size,
                'num_batches': num_batches,
                'total_processed': total_processed,
                'operation_complexity': operation_complexity,
                'bytes_processed': batch_size * num_batches * operation_complexity * 8
            }
        
        # Different batch configurations to test scalability
        batch_configs = [
            (1000, 50, 5),     # Small batches, low complexity
            (5000, 20, 10),    # Medium batches, medium complexity
            (10000, 10, 15),   # Large batches, medium complexity
            (20000, 5, 8),     # Very large batches, low complexity
            (2000, 25, 25),    # Medium batches, high complexity
            (50000, 2, 12),    # Huge batches, medium complexity
        ]
        
        # Measure batch processing scalability
        metrics = runner.measure_large_scale_operation(
            process_data_batch,
            batch_configs,
            max_workers=min(8, len(batch_configs)),
            timeout_seconds=300  # 5 minutes timeout
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'batch_processing_scalability',
            'batch_configurations': batch_configs,
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for batch processing scalability
        assert metrics.error_rate_percent < 5, f"Batch processing error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.5, f"Batch processing throughput too low: {metrics.operations_per_second:.3f} ops/sec"
        assert metrics.memory_peak_mb < 20000, f"Memory usage too high for batch processing: {metrics.memory_peak_mb:.1f}MB"


@pytest.mark.benchmark
@pytest.mark.stress
@pytest.mark.large
class TestSystemLimits:
    """Test system behavior at its limits."""
    
    @pytest.mark.stress
    @pytest.mark.large
    def test_memory_stress_limits(self, benchmark_result):
        """Test system behavior under extreme memory pressure."""
        benchmark_result.start()
        
        runner = LargeScaleBenchmarkRunner()
        
        def memory_intensive_operation(memory_config):
            """Perform memory-intensive operations."""
            target_mb, operation_type = memory_config
            
            target_bytes = target_mb * 1024 * 1024
            
            if operation_type == 'numpy_arrays':
                # Allocate large numpy arrays
                arrays = []
                allocated_bytes = 0
                
                while allocated_bytes < target_bytes:
                    # Allocate in chunks to avoid single allocation limits
                    chunk_size = min(100 * 1024 * 1024, target_bytes - allocated_bytes)  # 100MB chunks
                    array_elements = chunk_size // 8  # 8 bytes per float64
                    arr = np.random.randn(array_elements)
                    arrays.append(arr)
                    allocated_bytes += arr.nbytes
                
                # Perform operations on arrays
                result = {
                    'arrays_created': len(arrays),
                    'total_bytes': allocated_bytes,
                    'mean_values': [np.mean(arr) for arr in arrays[:5]],  # Sample first 5
                    'operation_type': operation_type
                }
                
                # Cleanup
                del arrays
                gc.collect()
                
            elif operation_type == 'dataframe':
                # Create large DataFrame
                rows = int(target_bytes / (50 * 8))  # Estimate ~50 columns of float64
                cols = min(50, rows)  # Limit columns to avoid excessive computation
                rows = max(1, rows // cols)  # Recalculate rows
                
                df = runner.create_large_dataset(rows, cols)
                
                # Perform operations
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                operations_performed = []
                
                if len(numeric_cols) > 0:
                    operations_performed.append('statistics')
                    stats = df[numeric_cols].describe()
                
                if len(df) > 1:
                    operations_performed.append('sorting')
                    df_sorted = df.sort_values(by=df.columns[0])
                
                result = {
                    'dataframe_shape': df.shape,
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    'operations_performed': operations_performed,
                    'operation_type': operation_type
                }
                
                # Cleanup
                del df
                if 'df_sorted' in locals():
                    del df_sorted
                gc.collect()
            
            result['bytes_processed'] = target_bytes
            return result
        
        # Memory stress configurations (in MB)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        max_safe_memory_mb = int(available_memory_gb * 1024 * 0.3)  # Use max 30% of available memory
        
        memory_configs = [
            (min(500, max_safe_memory_mb), 'numpy_arrays'),
            (min(1000, max_safe_memory_mb), 'dataframe'),
            (min(1500, max_safe_memory_mb), 'numpy_arrays'),
            (min(2000, max_safe_memory_mb), 'dataframe'),
        ]
        
        # Filter out configs that exceed safe memory limits
        memory_configs = [(mb, op_type) for mb, op_type in memory_configs if mb >= 100]  # Minimum 100MB
        
        if not memory_configs:
            memory_configs = [(100, 'numpy_arrays'), (200, 'dataframe')]  # Fallback minimal configs
        
        # Measure memory stress operations
        metrics = runner.measure_large_scale_operation(
            memory_intensive_operation,
            memory_configs,
            max_workers=2,  # Limited workers for memory-intensive operations
            timeout_seconds=180  # 3 minutes timeout
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'memory_stress_limits',
            'memory_configurations': memory_configs,
            'available_memory_gb': available_memory_gb,
            'max_safe_memory_mb': max_safe_memory_mb,
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for memory stress limits
        assert metrics.error_rate_percent < 25, f"Memory stress error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.02, f"Memory stress throughput too low: {metrics.operations_per_second:.3f} ops/sec"
    
    @pytest.mark.stress
    @pytest.mark.large
    def test_cpu_stress_limits(self, benchmark_result):
        """Test system behavior under extreme CPU load."""
        benchmark_result.start()
        
        runner = LargeScaleBenchmarkRunner()
        
        def cpu_intensive_operation(cpu_config):
            """Perform CPU-intensive operations."""
            operation_type, complexity_level = cpu_config
            
            if operation_type == 'mathematical':
                # Mathematical computations
                result = {'operation_type': operation_type, 'complexity': complexity_level}
                
                # Matrix operations
                size = min(500, complexity_level)  # Limit matrix size
                matrix_a = np.random.randn(size, size)
                matrix_b = np.random.randn(size, size)
                
                # Perform operations
                matrix_mult = np.dot(matrix_a, matrix_b)
                eigenvals = np.linalg.eigvals(matrix_mult)
                determinant = np.linalg.det(matrix_mult)
                
                result.update({
                    'matrix_size': size,
                    'eigenvalues_count': len(eigenvals),
                    'determinant': determinant,
                    'bytes_processed': matrix_a.nbytes + matrix_b.nbytes + matrix_mult.nbytes
                })
                
            elif operation_type == 'algorithmic':
                # Algorithmic computations
                result = {'operation_type': operation_type, 'complexity': complexity_level}
                
                # Prime calculation
                def sieve_of_eratosthenes(n):
                    primes = [True] * (n + 1)
                    primes[0] = primes[1] = False
                    
                    for i in range(2, int(n**0.5) + 1):
                        if primes[i]:
                            for j in range(i*i, n + 1, i):
                                primes[j] = False
                    
                    return [i for i in range(2, n + 1) if primes[i]]
                
                # Sorting algorithms
                def merge_sort(arr):
                    if len(arr) <= 1:
                        return arr
                    
                    mid = len(arr) // 2
                    left = merge_sort(arr[:mid])
                    right = merge_sort(arr[mid:])
                    
                    return merge(left, right)
                
                def merge(left, right):
                    result = []
                    i = j = 0
                    
                    while i < len(left) and j < len(right):
                        if left[i] <= right[j]:
                            result.append(left[i])
                            i += 1
                        else:
                            result.append(right[j])
                            j += 1
                    
                    result.extend(left[i:])
                    result.extend(right[j:])
                    return result
                
                # Execute algorithms
                primes = sieve_of_eratosthenes(min(10000, complexity_level * 100))
                
                random_array = np.random.randint(0, 1000, min(5000, complexity_level * 50))
                sorted_array = merge_sort(random_array.tolist())
                
                result.update({
                    'primes_found': len(primes),
                    'array_sorted_size': len(sorted_array),
                    'largest_prime': max(primes) if primes else 0,
                    'bytes_processed': len(primes) * 4 + len(sorted_array) * 4
                })
            
            return result
        
        # CPU stress configurations
        cpu_configs = [
            ('mathematical', 100),
            ('algorithmic', 150),
            ('mathematical', 200),
            ('algorithmic', 250),
            ('mathematical', 300),
            ('algorithmic', 200),
        ]
        
        # Measure CPU stress operations
        max_workers = min(os.cpu_count() or 4, 16)  # Use up to 16 workers or CPU count
        metrics = runner.measure_large_scale_operation(
            cpu_intensive_operation,
            cpu_configs,
            max_workers=max_workers,
            timeout_seconds=240  # 4 minutes timeout
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'cpu_stress_limits',
            'cpu_configurations': cpu_configs,
            'max_workers_used': max_workers,
            'cpu_count': os.cpu_count(),
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for CPU stress limits
        assert metrics.error_rate_percent < 15, f"CPU stress error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.1, f"CPU stress throughput too low: {metrics.operations_per_second:.3f} ops/sec"
        assert metrics.cpu_utilization_percent > 50, f"CPU utilization too low during stress test: {metrics.cpu_utilization_percent:.1f}%"
