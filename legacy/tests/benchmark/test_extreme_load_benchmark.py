# ---------------------------------------------------------------------
# tests/benchmark/test_extreme_load_benchmark.py - Extreme Load and Volume Benchmarks
# ---------------------------------------------------------------------
"""
Extreme load benchmark tests for testing system performance under massive data volumes,
extreme numerical operations, and sustained high-throughput scenarios.
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
import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Any, Callable, Generator
import statistics
from dataclasses import dataclass
import itertools
import math

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.conftest import time_function, measure_memory_usage

# Test if additional modules are available
try:
    import scipy.sparse as sp
    from scipy import linalg
    from scipy.stats import norm, gamma
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.extreme,
    pytest.mark.large,
    pytest.mark.stress,
    pytest.mark.concurrency
]

@dataclass
class ExtremeLoadMetrics:
    """Container for extreme load performance metrics."""
    total_operations: int
    operations_per_second: float
    data_processed_gb: float
    data_throughput_gb_per_sec: float
    peak_memory_gb: float
    average_memory_gb: float
    peak_cpu_percent: float
    average_cpu_percent: float
    duration_minutes: float
    error_count: int
    error_rate_percent: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    max_latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_operations': self.total_operations,
            'operations_per_second': self.operations_per_second,
            'data_processed_gb': self.data_processed_gb,
            'data_throughput_gb_per_sec': self.data_throughput_gb_per_sec,
            'peak_memory_gb': self.peak_memory_gb,
            'average_memory_gb': self.average_memory_gb,
            'peak_cpu_percent': self.peak_cpu_percent,
            'average_cpu_percent': self.average_cpu_percent,
            'duration_minutes': self.duration_minutes,
            'error_count': self.error_count,
            'error_rate_percent': self.error_rate_percent,
            'latency_p50_ms': self.latency_p50_ms,
            'latency_p95_ms': self.latency_p95_ms,
            'latency_p99_ms': self.latency_p99_ms,
            'max_latency_ms': self.max_latency_ms
        }


class ExtremeLoadRunner:
    """Runner for extreme load benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring_active = False
        self.cpu_samples = []
        self.memory_samples = []
        
    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring_active = True
        self.cpu_samples = []
        self.memory_samples = []
        
        def monitor():
            while self.monitoring_active:
                try:
                    cpu_percent = self.process.cpu_percent()
                    memory_mb = self.process.memory_info().rss / 1024**2
                    
                    self.cpu_samples.append(cpu_percent)
                    self.memory_samples.append(memory_mb)
                    
                    time.sleep(0.5)  # Sample every 500ms
                except:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
    def create_massive_dataset(self, rows: int, cols: int, dtype_mix: bool = True) -> pd.DataFrame:
        """Create a massive dataset with mixed data types."""
        np.random.seed(42)
        
        data = {}
        
        if dtype_mix:
            # Mixed data types
            num_numeric = cols // 3
            num_categorical = cols // 3
            num_text = cols - num_numeric - num_categorical
        else:
            # Mostly numeric for performance
            num_numeric = int(cols * 0.8)
            num_categorical = int(cols * 0.15)
            num_text = cols - num_numeric - num_categorical
        
        # Numeric columns with different distributions
        for i in range(num_numeric):
            if i % 3 == 0:
                data[f'normal_{i}'] = np.random.normal(0, 1, rows)
            elif i % 3 == 1:
                data[f'uniform_{i}'] = np.random.uniform(-10, 10, rows)
            else:
                data[f'exponential_{i}'] = np.random.exponential(2, rows)
        
        # Categorical columns
        categories = [f'cat_{j}' for j in range(20)]  # 20 different categories
        for i in range(num_categorical):
            data[f'category_{i}'] = np.random.choice(categories, rows)
        
        # Text columns
        words = ['data', 'science', 'machine', 'learning', 'artificial', 'intelligence', 
                'algorithm', 'model', 'prediction', 'analysis', 'processing', 'computing']
        for i in range(num_text):
            data[f'text_{i}'] = [' '.join(np.random.choice(words, size=np.random.randint(3, 8))) 
                                for _ in range(rows)]
        
        return pd.DataFrame(data)
    
    def measure_extreme_load(
        self,
        operation_func: Callable,
        operation_args: List[Any],
        max_workers: int = None,
        timeout_minutes: int = 10
    ) -> ExtremeLoadMetrics:
        """Measure performance under extreme load."""
        
        if max_workers is None:
            max_workers = min(64, (os.cpu_count() or 4) * 4)  # Higher concurrency for extreme tests
        
        # Start monitoring
        self.start_monitoring()
        gc.collect()
        
        start_time = time.perf_counter()
        latencies = []
        successful_ops = 0
        failed_ops = 0
        total_data_bytes = 0
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # Submit all operations
                for args in operation_args:
                    future = executor.submit(self._timed_extreme_operation, operation_func, args)
                    futures.append(future)
                
                # Collect results with timeout
                timeout_seconds = timeout_minutes * 60
                for future in concurrent.futures.as_completed(futures, timeout=timeout_seconds):
                    try:
                        latency_ms, success, data_bytes = future.result()
                        latencies.append(latency_ms)
                        total_data_bytes += data_bytes
                        
                        if success:
                            successful_ops += 1
                        else:
                            failed_ops += 1
                            
                    except Exception as e:
                        failed_ops += 1
                        print(f"Error in extreme operation: {e}")
        
        except concurrent.futures.TimeoutError:
            print(f"Timeout reached for extreme load test after {timeout_minutes} minutes")
            failed_ops += len(operation_args) - successful_ops - failed_ops
        
        end_time = time.perf_counter()
        duration_seconds = end_time - start_time
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Calculate metrics
        total_ops = successful_ops + failed_ops
        ops_per_second = total_ops / duration_seconds if duration_seconds > 0 else 0
        error_rate = (failed_ops / total_ops * 100) if total_ops > 0 else 0
        data_gb = total_data_bytes / (1024**3)
        data_throughput = data_gb / duration_seconds if duration_seconds > 0 else 0
        
        # Memory and CPU metrics
        peak_memory_gb = max(self.memory_samples) / 1024 if self.memory_samples else 0
        avg_memory_gb = statistics.mean(self.memory_samples) / 1024 if self.memory_samples else 0
        peak_cpu = max(self.cpu_samples) if self.cpu_samples else 0
        avg_cpu = statistics.mean(self.cpu_samples) if self.cpu_samples else 0
        
        # Latency metrics
        p50_latency = np.percentile(latencies, 50) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        
        return ExtremeLoadMetrics(
            total_operations=total_ops,
            operations_per_second=ops_per_second,
            data_processed_gb=data_gb,
            data_throughput_gb_per_sec=data_throughput,
            peak_memory_gb=peak_memory_gb,
            average_memory_gb=avg_memory_gb,
            peak_cpu_percent=peak_cpu,
            average_cpu_percent=avg_cpu,
            duration_minutes=duration_seconds / 60,
            error_count=failed_ops,
            error_rate_percent=error_rate,
            latency_p50_ms=p50_latency,
            latency_p95_ms=p95_latency,
            latency_p99_ms=p99_latency,
            max_latency_ms=max_latency
        )
    
    def _timed_extreme_operation(self, operation_func: Callable, args: Any) -> tuple:
        """Wrapper to time extreme operations and track data processed."""
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
            elif hasattr(result, '__len__'):
                data_bytes = len(str(result))  # Rough estimate
                
        except Exception as e:
            result = None
            success = False
            print(f"Extreme operation failed: {e}")
        
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        
        return latency_ms, success, data_bytes


class TestExtremeDataVolumes:
    """Test processing of extreme data volumes."""
    
    @pytest.mark.benchmark
    @pytest.mark.extreme
    def test_multi_million_row_processing(self, benchmark_result):
        """Test processing datasets with millions of rows."""
        benchmark_result.start()
        
        runner = ExtremeLoadRunner()
        
        def process_million_rows(dataset_config):
            """Process a dataset with millions of rows."""
            rows, cols, operation_type = dataset_config
            
            # Generate massive dataset
            df = runner.create_massive_dataset(rows, cols, dtype_mix=False)  # Numeric-heavy for performance
            
            start_processing = time.perf_counter()
            
            if operation_type == 'statistical':
                # Statistical operations
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Basic statistics
                stats = df[numeric_cols].describe()
                
                # Correlations (sample subset if too large)
                if len(numeric_cols) > 20:
                    sample_cols = numeric_cols[:20]
                else:
                    sample_cols = numeric_cols
                
                if len(sample_cols) > 1:
                    corr_matrix = df[sample_cols].corr()
                
                # Percentiles
                percentiles = df[numeric_cols].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                
                processing_time = time.perf_counter() - start_processing
                
                result = {
                    'operation_type': operation_type,
                    'rows_processed': len(df),
                    'columns_processed': len(numeric_cols),
                    'processing_time': processing_time,
                    'stats_computed': stats.shape,
                    'correlations_computed': len(sample_cols)**2 if len(sample_cols) > 1 else 0,
                    'data_bytes': df.memory_usage(deep=True).sum()
                }
                
            elif operation_type == 'aggregation':
                # Aggregation operations
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                # Group by operations (sample data if too large)
                if len(categorical_cols) > 0 and len(df) > 100000:
                    # Sample for large datasets to avoid excessive computation
                    sample_size = min(100000, len(df))
                    df_sample = df.sample(n=sample_size, random_state=42)
                    
                    groupby_col = categorical_cols[0]
                    grouped = df_sample.groupby(groupby_col)[numeric_cols[:5]].agg(['mean', 'sum', 'count'])
                    
                else:
                    # Full aggregation for smaller datasets
                    if len(categorical_cols) > 0:
                        groupby_col = categorical_cols[0]
                        grouped = df.groupby(groupby_col)[numeric_cols[:5]].agg(['mean', 'sum', 'count'])
                
                # Window operations
                if len(numeric_cols) > 0:
                    first_numeric = numeric_cols[0]
                    rolling_mean = df[first_numeric].rolling(window=1000).mean()
                
                processing_time = time.perf_counter() - start_processing
                
                result = {
                    'operation_type': operation_type,
                    'rows_processed': len(df),
                    'aggregations_computed': len(categorical_cols) if len(categorical_cols) > 0 else 1,
                    'rolling_operations': 1 if len(numeric_cols) > 0 else 0,
                    'processing_time': processing_time,
                    'data_bytes': df.memory_usage(deep=True).sum()
                }
                
            elif operation_type == 'transformation':
                # Data transformation operations
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Normalization
                if len(numeric_cols) > 0:
                    normalized = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
                
                # Binning
                if len(numeric_cols) > 0:
                    first_col = numeric_cols[0]
                    binned = pd.cut(df[first_col], bins=10, labels=False)
                
                # Sorting
                if len(df.columns) > 0:
                    sorted_df = df.sort_values(by=df.columns[0])
                
                processing_time = time.perf_counter() - start_processing
                
                result = {
                    'operation_type': operation_type,
                    'rows_processed': len(df),
                    'transformations_applied': 3,  # normalization, binning, sorting
                    'processing_time': processing_time,
                    'data_bytes': df.memory_usage(deep=True).sum()
                }
            
            # Cleanup to free memory
            del df
            if 'sorted_df' in locals():
                del sorted_df
            if 'normalized' in locals():
                del normalized
            gc.collect()
            
            return result
        
        # Extreme dataset configurations
        extreme_configs = [
            (2000000, 10, 'statistical'),    # 2M rows, 10 cols
            (3000000, 8, 'aggregation'),     # 3M rows, 8 cols
            (5000000, 6, 'transformation'),  # 5M rows, 6 cols
            (1500000, 15, 'statistical'),    # 1.5M rows, 15 cols
            (4000000, 5, 'aggregation'),     # 4M rows, 5 cols
        ]
        
        # Measure extreme volume processing
        metrics = runner.measure_extreme_load(
            process_million_rows,
            extreme_configs,
            max_workers=min(6, len(extreme_configs)),  # Limited workers for memory-intensive operations
            timeout_minutes=15  # 15 minutes timeout
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'multi_million_row_processing',
            'dataset_configurations': extreme_configs,
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for extreme volume processing
        assert metrics.error_rate_percent < 20, f"Error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.02, f"Throughput too low: {metrics.operations_per_second:.3f} ops/sec"
        assert metrics.peak_memory_gb < 32, f"Memory usage too high: {metrics.peak_memory_gb:.1f}GB"
    
    @pytest.mark.benchmark
    @pytest.mark.extreme
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_massive_numerical_computations(self, benchmark_result):
        """Test massive numerical computations."""
        benchmark_result.start()
        
        runner = ExtremeLoadRunner()
        
        def massive_numerical_operation(computation_config):
            """Perform massive numerical computations."""
            operation_type, size_param, complexity = computation_config
            
            if operation_type == 'matrix_operations':
                # Large matrix operations
                matrix_size = min(2000, size_param)  # Limit to prevent excessive memory usage
                
                # Generate matrices
                A = np.random.randn(matrix_size, matrix_size)
                B = np.random.randn(matrix_size, matrix_size)
                
                start_comp = time.perf_counter()
                
                # Matrix multiplication
                C = np.dot(A, B)
                
                # Eigenvalue decomposition
                if matrix_size <= 1000:  # Limit eigenvalue computation for performance
                    eigenvals, eigenvecs = np.linalg.eigh(A)
                else:
                    eigenvals = np.linalg.eigvals(A)
                    eigenvecs = None
                
                # SVD decomposition
                if matrix_size <= 800:
                    U, s, Vt = np.linalg.svd(A[:800, :800])
                else:
                    U = s = Vt = None
                
                comp_time = time.perf_counter() - start_comp
                
                result = {
                    'operation_type': operation_type,
                    'matrix_size': matrix_size,
                    'operations_performed': ['multiplication', 'eigenvalues'] + (['svd'] if U is not None else []),
                    'computation_time': comp_time,
                    'data_bytes': A.nbytes + B.nbytes + C.nbytes + (eigenvals.nbytes if eigenvals is not None else 0)
                }
                
                # Cleanup
                del A, B, C
                if eigenvals is not None:
                    del eigenvals
                if eigenvecs is not None:
                    del eigenvecs
                if U is not None:
                    del U, s, Vt
                
            elif operation_type == 'statistical_distributions':
                # Statistical distribution operations
                sample_size = min(10000000, size_param * 1000)  # Up to 10M samples
                
                start_comp = time.perf_counter()
                
                # Generate samples from different distributions
                normal_samples = np.random.normal(0, 1, sample_size)
                gamma_samples = np.random.gamma(2, 2, sample_size)
                uniform_samples = np.random.uniform(-5, 5, sample_size)
                
                # Statistical computations
                distributions_stats = {}
                for name, samples in [('normal', normal_samples), ('gamma', gamma_samples), ('uniform', uniform_samples)]:
                    distributions_stats[name] = {
                        'mean': np.mean(samples),
                        'std': np.std(samples),
                        'skewness': self._calculate_skewness(samples),
                        'kurtosis': self._calculate_kurtosis(samples),
                        'percentiles': np.percentile(samples, [1, 5, 25, 50, 75, 95, 99])
                    }
                
                # Hypothesis testing simulation
                test_results = []
                for _ in range(complexity):
                    sample1 = np.random.choice(normal_samples, 1000)
                    sample2 = np.random.choice(gamma_samples, 1000)
                    t_stat = (np.mean(sample1) - np.mean(sample2)) / np.sqrt(np.var(sample1)/len(sample1) + np.var(sample2)/len(sample2))
                    test_results.append(t_stat)
                
                comp_time = time.perf_counter() - start_comp
                
                result = {
                    'operation_type': operation_type,
                    'sample_size': sample_size,
                    'distributions_analyzed': 3,
                    'hypothesis_tests': len(test_results),
                    'computation_time': comp_time,
                    'data_bytes': normal_samples.nbytes + gamma_samples.nbytes + uniform_samples.nbytes
                }
                
                # Cleanup
                del normal_samples, gamma_samples, uniform_samples
                
            elif operation_type == 'signal_processing':
                # Signal processing operations
                signal_length = min(1000000, size_param * 10000)  # Up to 1M samples
                
                start_comp = time.perf_counter()
                
                # Generate signal
                t = np.linspace(0, 100, signal_length)
                signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.1, signal_length)
                
                # Signal processing operations
                # Windowing
                windowed_signal = signal * np.hanning(len(signal))
                
                # Filtering (simple moving average)
                window_size = min(1000, signal_length // 100)
                filtered_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
                
                # Frequency analysis (limited size for performance)
                if signal_length <= 100000:
                    fft_signal = np.fft.fft(signal)
                    power_spectrum = np.abs(fft_signal)**2
                else:
                    # Sample for large signals
                    sample_indices = np.random.choice(signal_length, 50000, replace=False)
                    sample_signal = signal[sample_indices]
                    fft_signal = np.fft.fft(sample_signal)
                    power_spectrum = np.abs(fft_signal)**2
                
                comp_time = time.perf_counter() - start_comp
                
                result = {
                    'operation_type': operation_type,
                    'signal_length': signal_length,
                    'operations_performed': ['windowing', 'filtering', 'fft'],
                    'computation_time': comp_time,
                    'data_bytes': signal.nbytes + windowed_signal.nbytes + filtered_signal.nbytes + power_spectrum.nbytes
                }
                
                # Cleanup
                del signal, windowed_signal, filtered_signal, fft_signal, power_spectrum
            
            gc.collect()
            return result
        
        # Massive computation configurations
        computation_configs = [
            ('matrix_operations', 1000, 10),
            ('statistical_distributions', 5000, 50),
            ('signal_processing', 100, 20),
            ('matrix_operations', 1500, 15),
            ('statistical_distributions', 8000, 75),
            ('signal_processing', 200, 30),
        ]
        
        # Measure massive computations
        metrics = runner.measure_extreme_load(
            massive_numerical_operation,
            computation_configs,
            max_workers=min(6, len(computation_configs)),
            timeout_minutes=12
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'massive_numerical_computations',
            'computation_configurations': computation_configs,
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for massive computations
        assert metrics.error_rate_percent < 15, f"Computation error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.05, f"Computation throughput too low: {metrics.operations_per_second:.3f} ops/sec"
        assert metrics.peak_memory_gb < 24, f"Memory usage too high: {metrics.peak_memory_gb:.1f}GB"
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0


class TestExtremeConcurrencyLoad:
    """Test extreme concurrency scenarios."""
    
    @pytest.mark.benchmark
    @pytest.mark.extreme
    def test_thousand_thread_simulation(self, benchmark_result):
        """Test system behavior with simulation of thousand-thread-like load."""
        benchmark_result.start()
        
        runner = ExtremeLoadRunner()
        
        def simulate_high_concurrency_task(task_config):
            """Simulate high concurrency tasks."""
            task_type, task_id, workload_size = task_config
            
            if task_type == 'io_simulation':
                # Simulate I/O-bound tasks
                start_task = time.perf_counter()
                
                # Simulate file operations
                temp_data = np.random.randn(workload_size, 10)
                
                # Simulate data processing
                processed_data = []
                for i in range(0, len(temp_data), 100):
                    batch = temp_data[i:i+100]
                    batch_result = {
                        'mean': np.mean(batch),
                        'std': np.std(batch),
                        'sum': np.sum(batch)
                    }
                    processed_data.append(batch_result)
                
                # Simulate I/O delay
                time.sleep(0.001 * (task_id % 10 + 1))
                
                task_time = time.perf_counter() - start_task
                
                result = {
                    'task_type': task_type,
                    'task_id': task_id,
                    'workload_size': workload_size,
                    'batches_processed': len(processed_data),
                    'task_time': task_time,
                    'data_bytes': temp_data.nbytes
                }
                
                del temp_data
                
            elif task_type == 'cpu_simulation':
                # Simulate CPU-bound tasks
                start_task = time.perf_counter()
                
                # CPU-intensive computation
                result_data = []
                for i in range(workload_size):
                    # Mathematical computation
                    value = math.sin(i) * math.cos(i) + math.sqrt(abs(i % 1000))
                    if i % 100 == 0:
                        value = math.factorial(min(i % 10 + 1, 10))  # Limit factorial
                    result_data.append(value)
                
                # Additional computation
                final_result = sum(result_data) / len(result_data) if result_data else 0
                
                task_time = time.perf_counter() - start_task
                
                result = {
                    'task_type': task_type,
                    'task_id': task_id,
                    'workload_size': workload_size,
                    'computations_performed': len(result_data),
                    'final_result': final_result,
                    'task_time': task_time,
                    'data_bytes': len(result_data) * 8  # Rough estimate
                }
                
            elif task_type == 'mixed_workload':
                # Mixed I/O and CPU tasks
                start_task = time.perf_counter()
                
                # CPU phase
                cpu_results = []
                for i in range(workload_size // 2):
                    cpu_results.append(math.pow(i, 2) + math.log(i + 1))
                
                # I/O simulation phase
                io_data = np.random.randn(workload_size // 4, 5)
                io_summary = {
                    'shape': io_data.shape,
                    'mean': np.mean(io_data),
                    'std': np.std(io_data)
                }
                
                # Simulate mixed delay
                time.sleep(0.0005 * (task_id % 5 + 1))
                
                task_time = time.perf_counter() - start_task
                
                result = {
                    'task_type': task_type,
                    'task_id': task_id,
                    'workload_size': workload_size,
                    'cpu_computations': len(cpu_results),
                    'io_operations': 1,
                    'task_time': task_time,
                    'data_bytes': io_data.nbytes + len(cpu_results) * 8
                }
                
                del io_data
            
            gc.collect()
            return result
        
        # Generate high concurrency task configurations
        high_concurrency_tasks = []
        
        # Generate many tasks to simulate high concurrency
        task_types = ['io_simulation', 'cpu_simulation', 'mixed_workload']
        workload_sizes = [100, 200, 500, 1000]
        
        for task_id in range(500):  # 500 concurrent tasks
            task_type = task_types[task_id % len(task_types)]
            workload_size = workload_sizes[task_id % len(workload_sizes)]
            high_concurrency_tasks.append((task_type, task_id, workload_size))
        
        # Measure extreme concurrency
        max_workers = min(100, (os.cpu_count() or 4) * 8)  # High concurrency level
        metrics = runner.measure_extreme_load(
            simulate_high_concurrency_task,
            high_concurrency_tasks,
            max_workers=max_workers,
            timeout_minutes=10
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'thousand_thread_simulation',
            'total_tasks': len(high_concurrency_tasks),
            'max_workers': max_workers,
            'task_types': task_types,
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for extreme concurrency
        assert metrics.error_rate_percent < 30, f"High concurrency error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 10, f"High concurrency throughput too low: {metrics.operations_per_second:.1f} ops/sec"
        assert metrics.peak_memory_gb < 20, f"Memory usage too high: {metrics.peak_memory_gb:.1f}GB"
    
    @pytest.mark.benchmark
    @pytest.mark.extreme
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_concurrent_model_inference_load(self, benchmark_result):
        """Test extreme load on concurrent model inference."""
        benchmark_result.start()
        
        runner = ExtremeLoadRunner()
        
        # Prepare multiple models for concurrent inference
        models = {}
        model_configs = [
            ('rf_small', RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)),
            ('rf_medium', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
            ('rf_large', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)),
        ]
        
        # Train models
        for model_name, model in model_configs:
            X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=42)
            model.fit(X, y)
            models[model_name] = {
                'model': model,
                'feature_count': X.shape[1]
            }
        
        def concurrent_inference_task(inference_config):
            """Perform concurrent model inference."""
            model_name, batch_size, num_predictions = inference_config
            
            model_info = models[model_name]
            model = model_info['model']
            feature_count = model_info['feature_count']
            
            start_inference = time.perf_counter()
            
            # Generate prediction batches
            predictions = []
            probabilities = []
            total_data_bytes = 0
            
            for batch_idx in range(num_predictions // batch_size):
                # Generate batch data
                batch_data = np.random.randn(batch_size, feature_count)
                total_data_bytes += batch_data.nbytes
                
                # Perform inference
                batch_predictions = model.predict(batch_data)
                batch_probabilities = model.predict_proba(batch_data)
                
                predictions.extend(batch_predictions)
                probabilities.extend(batch_probabilities)
                
                # Small delay to simulate real-world inference latency
                time.sleep(0.0001)
            
            inference_time = time.perf_counter() - start_inference
            
            result = {
                'model_name': model_name,
                'batch_size': batch_size,
                'total_predictions': len(predictions),
                'inference_time': inference_time,
                'predictions_per_second': len(predictions) / inference_time if inference_time > 0 else 0,
                'data_bytes': total_data_bytes
            }
            
            return result
        
        # Generate extreme inference workload
        inference_configs = []
        
        for model_name, _ in model_configs:
            # Different batch sizes and prediction counts
            batch_configs = [
                (10, 1000),    # Small batches, many predictions
                (50, 2000),    # Medium batches
                (100, 3000),   # Large batches
                (25, 1500),    # Mixed load
            ]
            
            for batch_size, num_predictions in batch_configs:
                inference_configs.append((model_name, batch_size, num_predictions))
        
        # Add more configurations for extreme load
        for _ in range(20):  # Additional concurrent inference tasks
            model_name = model_configs[_ % len(model_configs)][0]
            batch_size = [10, 25, 50, 100][_ % 4]
            num_predictions = [500, 1000, 1500, 2000][_ % 4]
            inference_configs.append((model_name, batch_size, num_predictions))
        
        # Measure concurrent inference load
        metrics = runner.measure_extreme_load(
            concurrent_inference_task,
            inference_configs,
            max_workers=min(20, len(inference_configs)),
            timeout_minutes=8
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'concurrent_model_inference_load',
            'model_configurations': [(name, type(model).__name__) for name, model in model_configs],
            'inference_configurations': len(inference_configs),
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for concurrent inference load
        assert metrics.error_rate_percent < 10, f"Inference error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 2, f"Inference throughput too low: {metrics.operations_per_second:.1f} ops/sec"
        assert metrics.peak_memory_gb < 16, f"Memory usage too high: {metrics.peak_memory_gb:.1f}GB"


@pytest.mark.benchmark
@pytest.mark.stress
@pytest.mark.extreme
class TestSustainedExtremeLoad:
    """Test sustained extreme load scenarios."""
    
    @pytest.mark.stress
    @pytest.mark.extreme
    def test_sustained_extreme_throughput(self, benchmark_result):
        """Test sustained extreme throughput over extended periods."""
        benchmark_result.start()
        
        runner = ExtremeLoadRunner()
        
        def sustained_workload_task(task_config):
            """Perform sustained workload tasks."""
            task_phase, duration_seconds, intensity = task_config
            
            start_phase = time.perf_counter()
            end_phase = start_phase + duration_seconds
            
            operations_count = 0
            data_processed = 0
            
            if task_phase == 'warm_up':
                # Warm-up phase with gradually increasing load
                while time.perf_counter() < end_phase:
                    current_time = time.perf_counter() - start_phase
                    current_intensity = int(intensity * (current_time / duration_seconds))
                    
                    # Generate data based on current intensity
                    data = np.random.randn(current_intensity * 100, 10)
                    result = np.mean(data) + np.std(data)
                    
                    operations_count += 1
                    data_processed += data.nbytes
                    
                    time.sleep(0.01)  # Small delay
                    del data
                
            elif task_phase == 'peak_load':
                # Peak load phase with maximum intensity
                while time.perf_counter() < end_phase:
                    # High intensity operations
                    data = np.random.randn(intensity * 200, 15)
                    
                    # Complex operations
                    normalized = (data - np.mean(data)) / np.std(data)
                    filtered = normalized[np.abs(normalized) < 2]  # Remove outliers
                    result = {
                        'mean': np.mean(filtered) if len(filtered) > 0 else 0,
                        'count': len(filtered)
                    }
                    
                    operations_count += 1
                    data_processed += data.nbytes
                    
                    time.sleep(0.005)  # Shorter delay for peak load
                    del data, normalized, filtered
                
            elif task_phase == 'sustained':
                # Sustained load phase with consistent intensity
                while time.perf_counter() < end_phase:
                    # Consistent operations
                    data = np.random.randn(intensity * 150, 12)
                    
                    # Mixed operations
                    stats = {
                        'mean': np.mean(data),
                        'std': np.std(data),
                        'min': np.min(data),
                        'max': np.max(data),
                        'median': np.median(data)
                    }
                    
                    # Transformation
                    transformed = data * 2 + 1
                    
                    operations_count += 1
                    data_processed += data.nbytes + transformed.nbytes
                    
                    time.sleep(0.008)  # Consistent delay
                    del data, transformed
            
            actual_duration = time.perf_counter() - start_phase
            
            result = {
                'task_phase': task_phase,
                'planned_duration': duration_seconds,
                'actual_duration': actual_duration,
                'operations_completed': operations_count,
                'operations_per_second': operations_count / actual_duration if actual_duration > 0 else 0,
                'data_bytes': data_processed
            }
            
            gc.collect()
            return result
        
        # Sustained load configuration (phases)
        sustained_phases = [
            ('warm_up', 30, 5),      # 30 seconds warm-up, intensity 5
            ('peak_load', 60, 10),   # 60 seconds peak load, intensity 10
            ('sustained', 120, 7),   # 120 seconds sustained, intensity 7
            ('peak_load', 45, 12),   # 45 seconds second peak, intensity 12
            ('sustained', 90, 8),    # 90 seconds sustained, intensity 8
        ]
        
        # Measure sustained extreme load
        metrics = runner.measure_extreme_load(
            sustained_workload_task,
            sustained_phases,
            max_workers=min(8, len(sustained_phases)),
            timeout_minutes=15  # 15 minutes total timeout
        )
        
        benchmark_result.stop()
        
        total_planned_duration = sum(duration for _, duration, _ in sustained_phases)
        
        benchmark_result.metadata = {
            'test_type': 'sustained_extreme_throughput',
            'sustained_phases': sustained_phases,
            'total_planned_duration_seconds': total_planned_duration,
            'total_planned_duration_minutes': total_planned_duration / 60,
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for sustained extreme load
        assert metrics.error_rate_percent < 25, f"Sustained load error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.5, f"Sustained throughput too low: {metrics.operations_per_second:.1f} ops/sec"
        assert metrics.duration_minutes > 3, f"Test duration too short: {metrics.duration_minutes:.1f} minutes"
        assert metrics.peak_memory_gb < 30, f"Memory usage too high: {metrics.peak_memory_gb:.1f}GB"
    
    @pytest.mark.stress
    @pytest.mark.extreme
    def test_memory_pressure_resilience(self, benchmark_result):
        """Test system resilience under extreme memory pressure."""
        benchmark_result.start()
        
        runner = ExtremeLoadRunner()
        
        def memory_pressure_task(pressure_config):
            """Perform tasks under memory pressure."""
            pressure_type, target_memory_mb, operation_count = pressure_config
            
            allocated_objects = []
            total_data_bytes = 0
            
            try:
                if pressure_type == 'gradual_pressure':
                    # Gradually increase memory usage
                    chunk_size_mb = target_memory_mb // operation_count
                    
                    for i in range(operation_count):
                        # Allocate memory chunk
                        chunk_size = chunk_size_mb * 1024 * 1024
                        chunk_elements = chunk_size // 8
                        
                        chunk = np.random.randn(chunk_elements)
                        allocated_objects.append(chunk)
                        total_data_bytes += chunk.nbytes
                        
                        # Perform operation on chunk
                        result = np.mean(chunk) + np.std(chunk)
                        
                        # Small delay
                        time.sleep(0.01)
                
                elif pressure_type == 'spike_pressure':
                    # Create memory spikes
                    base_size_mb = target_memory_mb // 4
                    spike_size_mb = target_memory_mb
                    
                    for i in range(operation_count):
                        if i % 3 == 0:  # Every 3rd operation is a spike
                            size_mb = spike_size_mb
                        else:
                            size_mb = base_size_mb
                        
                        chunk_size = size_mb * 1024 * 1024
                        chunk_elements = chunk_size // 8
                        
                        chunk = np.random.randn(chunk_elements)
                        
                        # Process chunk immediately and release
                        result = {
                            'mean': np.mean(chunk),
                            'std': np.std(chunk),
                            'size_mb': size_mb
                        }
                        
                        total_data_bytes += chunk.nbytes
                        
                        # Keep only small chunks
                        if size_mb == base_size_mb:
                            allocated_objects.append(chunk)
                        else:
                            del chunk
                        
                        time.sleep(0.005)
                
                elif pressure_type == 'sustained_pressure':
                    # Sustained high memory usage
                    chunk_size_mb = target_memory_mb // 2
                    
                    # Allocate base memory
                    for i in range(2):  # Two large chunks
                        chunk_size = chunk_size_mb * 1024 * 1024
                        chunk_elements = chunk_size // 8
                        
                        chunk = np.random.randn(chunk_elements)
                        allocated_objects.append(chunk)
                        total_data_bytes += chunk.nbytes
                    
                    # Perform operations while maintaining memory pressure
                    for i in range(operation_count):
                        # Small temporary allocations
                        temp_data = np.random.randn(10000, 10)
                        
                        # Operations on existing data
                        if allocated_objects:
                            sample_chunk = allocated_objects[i % len(allocated_objects)]
                            result = np.correlate(sample_chunk[:1000], temp_data.flatten()[:1000], mode='valid')
                        
                        total_data_bytes += temp_data.nbytes
                        del temp_data
                        
                        time.sleep(0.01)
                
                result = {
                    'pressure_type': pressure_type,
                    'target_memory_mb': target_memory_mb,
                    'operations_completed': operation_count,
                    'objects_allocated': len(allocated_objects),
                    'success': True,
                    'data_bytes': total_data_bytes
                }
                
            except MemoryError:
                result = {
                    'pressure_type': pressure_type,
                    'target_memory_mb': target_memory_mb,
                    'operations_completed': operation_count,
                    'objects_allocated': len(allocated_objects),
                    'success': False,
                    'error': 'MemoryError',
                    'data_bytes': total_data_bytes
                }
            
            except Exception as e:
                result = {
                    'pressure_type': pressure_type,
                    'target_memory_mb': target_memory_mb,
                    'operations_completed': operation_count,
                    'objects_allocated': len(allocated_objects),
                    'success': False,
                    'error': str(e),
                    'data_bytes': total_data_bytes
                }
            
            finally:
                # Cleanup
                for obj in allocated_objects:
                    del obj
                allocated_objects.clear()
                gc.collect()
            
            return result
        
        # Memory pressure configurations
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        safe_memory_mb = int(available_memory_gb * 1024 * 0.2)  # Use 20% of available memory
        
        pressure_configs = [
            ('gradual_pressure', min(500, safe_memory_mb), 10),
            ('spike_pressure', min(800, safe_memory_mb), 15),
            ('sustained_pressure', min(1000, safe_memory_mb), 20),
            ('gradual_pressure', min(600, safe_memory_mb), 12),
        ]
        
        # Filter out configs that might cause system issues
        pressure_configs = [(ptype, min(mb, 2000), ops) for ptype, mb, ops in pressure_configs if mb >= 100]
        
        # Measure memory pressure resilience
        metrics = runner.measure_extreme_load(
            memory_pressure_task,
            pressure_configs,
            max_workers=2,  # Limited workers for memory pressure tests
            timeout_minutes=8
        )
        
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'test_type': 'memory_pressure_resilience',
            'pressure_configurations': pressure_configs,
            'available_memory_gb': available_memory_gb,
            'safe_memory_limit_mb': safe_memory_mb,
            'performance_metrics': metrics.to_dict()
        }
        
        # Assertions for memory pressure resilience
        # More lenient assertions for memory pressure tests
        assert metrics.error_rate_percent < 50, f"Memory pressure error rate too high: {metrics.error_rate_percent:.1f}%"
        assert metrics.operations_per_second > 0.1, f"Memory pressure throughput too low: {metrics.operations_per_second:.1f} ops/sec"
