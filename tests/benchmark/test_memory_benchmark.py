# ---------------------------------------------------------------------
# tests/benchmark/test_memory_benchmark.py - Comprehensive memory usage testing
# ---------------------------------------------------------------------
"""
Comprehensive memory usage benchmark tests.
Tests memory leaks, peak usage, garbage collection efficiency, and memory patterns.
"""
import pytest
import pandas as pd
import numpy as np
import time
import gc
import sys
import os
import threading
from pathlib import Path
from unittest.mock import Mock, patch
import psutil
from typing import List, Dict, Any, Optional
import tracemalloc
import weakref

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.conftest import time_function, measure_memory_usage

# Test if components are available
try:
    from app import MLSystemUI
    HAS_APP = True
except (ImportError, SystemExit):
    HAS_APP = False

try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.memory
]

class MemoryProfiler:
    """Advanced memory profiling utility."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = None
        self.peak_memory = 0
        self.samples = []
        self.gc_stats = []
        self.tracemalloc_started = False
        
    def start_profiling(self, enable_tracemalloc=True):
        """Start memory profiling."""
        gc.collect()  # Clean start
        
        if enable_tracemalloc and not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
            
        self.start_memory = self.process.memory_info().rss / 1024**2
        self.peak_memory = self.start_memory
        self.samples = [self.start_memory]
        self.gc_stats = [self._get_gc_stats()]
        
    def sample(self, label: str = ""):
        """Take a memory sample."""
        current_memory = self.process.memory_info().rss / 1024**2
        self.samples.append(current_memory)
        self.peak_memory = max(self.peak_memory, current_memory)
        
        sample_data = {
            'timestamp': time.time(),
            'memory_mb': current_memory,
            'label': label
        }
        
        if self.tracemalloc_started:
            try:
                current, peak = tracemalloc.get_traced_memory()
                sample_data['tracemalloc_current_mb'] = current / 1024**2
                sample_data['tracemalloc_peak_mb'] = peak / 1024**2
            except:
                pass
                
        return sample_data
        
    def force_gc(self):
        """Force garbage collection and record stats."""
        before_memory = self.process.memory_info().rss / 1024**2
        
        # Multiple GC passes to ensure cleanup
        for _ in range(3):
            collected = gc.collect()
            
        after_memory = self.process.memory_info().rss / 1024**2
        
        gc_result = {
            'before_mb': before_memory,
            'after_mb': after_memory,
            'freed_mb': before_memory - after_memory,
            'gc_stats': self._get_gc_stats()
        }
        
        self.gc_stats.append(gc_result)
        self.sample("after_gc")
        
        return gc_result
        
    def stop_profiling(self):
        """Stop profiling and get final stats."""
        final_memory = self.process.memory_info().rss / 1024**2
        
        stats = {
            'start_memory_mb': self.start_memory,
            'final_memory_mb': final_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_growth_mb': final_memory - self.start_memory,
            'peak_growth_mb': self.peak_memory - self.start_memory,
            'total_samples': len(self.samples),
            'gc_cycles': len(self.gc_stats),
            'average_memory_mb': np.mean(self.samples) if self.samples else 0
        }
        
        if self.tracemalloc_started:
            try:
                current, peak = tracemalloc.get_traced_memory()
                stats['tracemalloc_final_mb'] = current / 1024**2
                stats['tracemalloc_peak_mb'] = peak / 1024**2
                tracemalloc.stop()
                self.tracemalloc_started = False
            except:
                pass
                
        return stats
        
    def _get_gc_stats(self):
        """Get garbage collection statistics."""
        return {
            'collections': gc.get_stats(),
            'counts': gc.get_count(),
            'threshold': gc.get_threshold()
        }


class TestMemoryLeakDetection:
    """Test for memory leaks in various operations."""
    
    @pytest.mark.memory
    def test_data_loading_memory_leaks(self, test_data_generator, benchmark_result):
        """Test for memory leaks in repeated data loading."""
        benchmark_result.start()
        
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        # Generate test file
        test_data = test_data_generator('medium')
        file_path = test_data['file_path']
        
        memory_samples = []
        iterations = 50
        
        for i in range(iterations):
            # Load data
            df = pd.read_csv(file_path)
            
            # Process data
            summary = df.describe()
            preview = df.head(10)
            memory_usage = df.memory_usage(deep=True).sum()
            
            # Sample memory
            sample = profiler.sample(f"iteration_{i}")
            memory_samples.append(sample)
            
            # Cleanup
            del df, summary, preview
            
            # Force GC every 10 iterations
            if i % 10 == 0:
                gc_result = profiler.force_gc()
            
        final_stats = profiler.stop_profiling()
        benchmark_result.stop()
        
        # Analyze memory growth pattern
        memory_values = [s['memory_mb'] for s in memory_samples]
        early_avg = np.mean(memory_values[:10])
        late_avg = np.mean(memory_values[-10:])
        memory_trend = late_avg - early_avg
        
        benchmark_result.metadata = {
            'profiler_stats': final_stats,
            'memory_samples': memory_samples[-10:],  # Last 10 samples
            'iterations': iterations,
            'memory_trend_mb': memory_trend,
            'early_avg_mb': early_avg,
            'late_avg_mb': late_avg,
            'leak_detected': memory_trend > 50  # >50MB growth indicates leak
        }
        
        # Assertions
        assert memory_trend < 100, f"Possible memory leak detected: {memory_trend}MB growth"
        assert final_stats['peak_growth_mb'] < 500, f"Peak memory growth too high: {final_stats['peak_growth_mb']}MB"
    
    @pytest.mark.memory
    @pytest.mark.skipif(not HAS_APP, reason="App not available")
    def test_ui_component_memory_leaks(self, test_data_generator, benchmark_result):
        """Test for memory leaks in UI components."""
        benchmark_result.start()
        
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        ui_instances = []
        memory_samples = []
        
        # Create and destroy UI instances
        for i in range(20):
            # Create UI instance
            ui = MLSystemUI(inference_only=True)
            
            # Load some data
            test_data = test_data_generator('small')
            mock_file = Mock()
            mock_file.name = str(test_data['file_path'])
            
            ui.load_data(mock_file)
            ui.refresh_data_preview()
            
            # Keep weak reference to check cleanup
            weak_ref = weakref.ref(ui)
            ui_instances.append(weak_ref)
            
            sample = profiler.sample(f"ui_instance_{i}")
            memory_samples.append(sample)
            
            # Delete UI instance
            del ui
            
            # Force cleanup every 5 iterations
            if i % 5 == 0:
                profiler.force_gc()
        
        # Final cleanup
        profiler.force_gc()
        final_stats = profiler.stop_profiling()
        
        # Check if UI instances were properly cleaned up
        alive_instances = sum(1 for ref in ui_instances if ref() is not None)
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'profiler_stats': final_stats,
            'ui_instances_created': len(ui_instances),
            'ui_instances_alive': alive_instances,
            'memory_samples': memory_samples[-5:],
            'cleanup_efficiency_percent': (len(ui_instances) - alive_instances) / len(ui_instances) * 100
        }
        
        # Assertions
        assert alive_instances < len(ui_instances) * 0.1, f"Too many UI instances not cleaned up: {alive_instances}"
        assert final_stats['memory_growth_mb'] < 200, f"UI memory growth too high: {final_stats['memory_growth_mb']}MB"
    
    @pytest.mark.memory
    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not available")
    def test_ml_model_memory_leaks(self, benchmark_result):
        """Test for memory leaks in ML model operations."""
        benchmark_result.start()
        
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        model_instances = []
        memory_samples = []
        
        # Create and train multiple models
        for i in range(15):
            # Generate data
            X, y = make_classification(n_samples=1000, n_features=20, random_state=i)
            
            # Create and train model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X[:100])
            
            # Keep weak reference
            weak_ref = weakref.ref(model)
            model_instances.append(weak_ref)
            
            sample = profiler.sample(f"model_{i}")
            memory_samples.append(sample)
            
            # Cleanup
            del model, X, y, predictions
            
            if i % 5 == 0:
                profiler.force_gc()
        
        profiler.force_gc()
        final_stats = profiler.stop_profiling()
        
        # Check model cleanup
        alive_models = sum(1 for ref in model_instances if ref() is not None)
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'profiler_stats': final_stats,
            'models_created': len(model_instances),
            'models_alive': alive_models,
            'memory_samples': memory_samples[-5:],
            'model_cleanup_efficiency_percent': (len(model_instances) - alive_models) / len(model_instances) * 100
        }
        
        # Assertions
        assert alive_models < len(model_instances) * 0.2, f"Too many models not cleaned up: {alive_models}"
        assert final_stats['memory_growth_mb'] < 300, f"ML model memory growth too high: {final_stats['memory_growth_mb']}MB"


class TestMemoryUsagePatterns:
    """Test memory usage patterns under different scenarios."""
    
    @pytest.mark.memory
    def test_large_dataset_memory_usage(self, test_data_generator, benchmark_result):
        """Test memory usage with large datasets."""
        benchmark_result.start()
        
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        # Test different dataset sizes
        sizes = ['medium', 'large']
        size_results = {}
        
        for size in sizes:
            size_profiler = MemoryProfiler()
            size_profiler.start_profiling()
            
            test_data = test_data_generator(size)
            size_profiler.sample("after_generation")
            
            # Load dataset
            df = pd.read_csv(test_data['file_path'])
            size_profiler.sample("after_loading")
            
            # Memory-intensive operations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                correlation_matrix = df[numeric_cols].corr()
                size_profiler.sample("after_correlation")
            
            # Statistical operations
            summary_stats = df.describe()
            size_profiler.sample("after_stats")
            
            # Data transformation
            df_copy = df.copy()
            size_profiler.sample("after_copy")
            
            # Cleanup
            del df, df_copy, summary_stats
            if 'correlation_matrix' in locals():
                del correlation_matrix
            
            size_profiler.force_gc()
            size_stats = size_profiler.stop_profiling()
            
            size_results[size] = {
                'stats': size_stats,
                'data_shape': test_data['dataframe'].shape,
                'expected_memory_mb': test_data['memory_mb']
            }
        
        final_stats = profiler.stop_profiling()
        benchmark_result.stop()
        
        benchmark_result.metadata = {
            'overall_stats': final_stats,
            'size_specific_results': size_results
        }
        
        # Memory usage should be proportional to data size
        for size, result in size_results.items():
            actual_growth = result['stats']['peak_growth_mb']
            expected_memory = result['expected_memory_mb']
            
            # Memory usage should not exceed 5x the expected size
            assert actual_growth < expected_memory * 5, \
                f"{size} dataset used {actual_growth}MB vs expected ~{expected_memory}MB"
    
    @pytest.mark.memory
    def test_concurrent_memory_usage(self, test_data_generator, benchmark_result):
        """Test memory usage under concurrent operations."""
        benchmark_result.start()
        
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        import threading
        import queue
        
        # Shared results queue
        results_queue = queue.Queue()
        memory_samples = []
        
        def worker_function(worker_id, iterations):
            """Worker function for concurrent testing."""
            worker_memory = []
            
            for i in range(iterations):
                try:
                    # Generate and process data
                    test_data = test_data_generator('small')
                    df = pd.read_csv(test_data['file_path'])
                    
                    # Process data
                    summary = df.describe()
                    preview = df.head(10)
                    
                    # Sample memory
                    current_memory = psutil.Process().memory_info().rss / 1024**2
                    worker_memory.append(current_memory)
                    
                    # Cleanup
                    del df, summary, preview
                    
                except Exception as e:
                    worker_memory.append(f"Error: {e}")
            
            results_queue.put((worker_id, worker_memory))
        
        # Start multiple workers
        workers = []
        num_workers = 4
        iterations_per_worker = 10
        
        profiler.sample("before_workers")
        
        for worker_id in range(num_workers):
            worker = threading.Thread(
                target=worker_function, 
                args=(worker_id, iterations_per_worker)
            )
            workers.append(worker)
            worker.start()
        
        # Monitor memory during concurrent execution
        monitoring_start = time.time()
        while any(w.is_alive() for w in workers) and time.time() - monitoring_start < 30:
            sample = profiler.sample("concurrent_execution")
            memory_samples.append(sample)
            time.sleep(0.5)
        
        # Wait for all workers to complete
        for worker in workers:
            worker.join(timeout=10)
        
        profiler.sample("after_workers")
        
        # Collect worker results
        worker_results = {}
        while not results_queue.empty():
            worker_id, worker_memory = results_queue.get()
            worker_results[f'worker_{worker_id}'] = worker_memory
        
        profiler.force_gc()
        final_stats = profiler.stop_profiling()
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'profiler_stats': final_stats,
            'worker_results': worker_results,
            'num_workers': num_workers,
            'iterations_per_worker': iterations_per_worker,
            'concurrent_memory_samples': memory_samples[-10:],
            'peak_concurrent_memory_mb': max(s['memory_mb'] for s in memory_samples) if memory_samples else 0
        }
        
        # Concurrent operations shouldn't cause excessive memory growth
        peak_memory = max(s['memory_mb'] for s in memory_samples) if memory_samples else 0
        base_memory = profiler.start_memory
        
        assert peak_memory - base_memory < 1000, \
            f"Concurrent operations caused excessive memory growth: {peak_memory - base_memory}MB"
    
    @pytest.mark.memory
    def test_memory_fragmentation(self, benchmark_result):
        """Test for memory fragmentation patterns."""
        benchmark_result.start()
        
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        # Create and delete objects of varying sizes to test fragmentation
        fragmentation_test_results = []
        
        for cycle in range(10):
            cycle_start_memory = psutil.Process().memory_info().rss / 1024**2
            
            # Allocate objects of different sizes
            large_objects = []
            small_objects = []
            
            # Large objects
            for i in range(5):
                large_array = np.random.randn(10000, 100)
                large_objects.append(large_array)
            
            profiler.sample(f"cycle_{cycle}_large_allocated")
            
            # Small objects
            for i in range(100):
                small_array = np.random.randn(100, 10)
                small_objects.append(small_array)
                
            profiler.sample(f"cycle_{cycle}_small_allocated")
            
            # Delete in fragmented pattern (every other object)
            for i in range(0, len(large_objects), 2):
                if i < len(large_objects):
                    del large_objects[i]
                    
            for i in range(0, len(small_objects), 3):
                if i < len(small_objects):
                    del small_objects[i]
            
            profiler.sample(f"cycle_{cycle}_partial_cleanup")
            
            # Force GC
            gc_result = profiler.force_gc()
            
            cycle_end_memory = psutil.Process().memory_info().rss / 1024**2
            
            fragmentation_test_results.append({
                'cycle': cycle,
                'start_memory_mb': cycle_start_memory,
                'end_memory_mb': cycle_end_memory,
                'memory_delta_mb': cycle_end_memory - cycle_start_memory,
                'gc_freed_mb': gc_result['freed_mb']
            })
            
            # Cleanup remaining objects
            del large_objects, small_objects
        
        final_stats = profiler.stop_profiling()
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'profiler_stats': final_stats,
            'fragmentation_test_results': fragmentation_test_results,
            'avg_cycle_memory_delta': np.mean([r['memory_delta_mb'] for r in fragmentation_test_results]),
            'total_gc_freed_mb': sum(r['gc_freed_mb'] for r in fragmentation_test_results)
        }
        
        # Memory should be efficiently managed across cycles
        avg_delta = np.mean([r['memory_delta_mb'] for r in fragmentation_test_results])
        assert avg_delta < 50, f"Average memory delta per cycle too high: {avg_delta}MB"


@pytest.mark.memory
@pytest.mark.stress
class TestMemoryStress:
    """Stress tests for memory management."""
    
    @pytest.mark.stress
    def test_memory_pressure_handling(self, test_data_generator, benchmark_result):
        """Test system behavior under memory pressure."""
        benchmark_result.start()
        
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        # Gradually increase memory pressure
        pressure_levels = []
        memory_hogs = []
        
        try:
            for pressure_level in range(1, 11):
                pressure_start = psutil.Process().memory_info().rss / 1024**2
                
                # Allocate increasingly large arrays
                size = pressure_level * 1000000  # 1M elements per level
                memory_hog = np.random.randn(size)
                memory_hogs.append(memory_hog)
                
                # Try to perform normal operations under pressure
                try:
                    test_data = test_data_generator('small')
                    df = pd.read_csv(test_data['file_path'])
                    summary = df.describe()
                    operations_successful = True
                    del df, summary
                except Exception:
                    operations_successful = False
                
                pressure_end = psutil.Process().memory_info().rss / 1024**2
                
                sample = profiler.sample(f"pressure_level_{pressure_level}")
                
                pressure_levels.append({
                    'level': pressure_level,
                    'start_memory_mb': pressure_start,
                    'end_memory_mb': pressure_end,
                    'allocated_mb': pressure_end - pressure_start,
                    'operations_successful': operations_successful,
                    'sample': sample
                })
                
                # Stop if we hit memory limits
                if pressure_end > 2000:  # 2GB limit
                    break
                    
        except MemoryError:
            # Expected behavior under extreme pressure
            pass
        finally:
            # Cleanup
            del memory_hogs
            profiler.force_gc()
        
        final_stats = profiler.stop_profiling()
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'profiler_stats': final_stats,
            'pressure_levels': pressure_levels,
            'max_pressure_level': len(pressure_levels),
            'operations_success_rate': sum(1 for p in pressure_levels if p['operations_successful']) / len(pressure_levels) * 100 if pressure_levels else 0
        }
        
        # System should handle some memory pressure gracefully
        success_rate = benchmark_result.metadata['operations_success_rate']
        assert success_rate > 50, f"Operations failed too often under memory pressure: {success_rate}%"
