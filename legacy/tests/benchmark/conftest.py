# ---------------------------------------------------------------------
# tests/benchmark/conftest.py - Benchmark-specific fixtures
# ---------------------------------------------------------------------
"""
Benchmark-specific fixtures and utilities.
"""
import os
import sys
import time
import psutil
import gc
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock
from typing import Dict, Any, Generator, Tuple
import json
import logging

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import device optimizer for optimal configurations
try:
    from modules.device_optimizer import DeviceOptimizer, OptimizationMode
    from modules.configs import OptimizationMode as ConfigOptimizationMode
    DEVICE_OPTIMIZER_AVAILABLE = True
except ImportError:
    DEVICE_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
        self.memory_before_mb = None
        self.memory_after_mb = None
        self.memory_peak_mb = None
        self.memory_delta_mb = None
        self.cpu_percent = None
        self.metadata = {}
        self._process = psutil.Process()
        self._cpu_samples = []
        self._memory_samples = []
        
    def start(self):
        """Start timing and memory measurement."""
        gc.collect()  # Clean up before measurement
        self.start_time = time.perf_counter()
        self.memory_before_mb = self._process.memory_info().rss / 1024**2
        self.memory_peak_mb = self.memory_before_mb  # Initialize peak with starting value
        
        # Initialize CPU monitoring (first call establishes baseline)
        self._process.cpu_percent()
        self._cpu_samples = []
        self._memory_samples = [self.memory_before_mb]
        
    def stop(self):
        """Stop timing and finalize measurements."""
        self.end_time = time.perf_counter()
        
        # Final memory measurement
        self.memory_after_mb = self._process.memory_info().rss / 1024**2
        self._memory_samples.append(self.memory_after_mb)
        
        # Calculate memory metrics
        self.memory_delta_mb = self.memory_after_mb - self.memory_before_mb
        self.memory_peak_mb = max(self._memory_samples) if self._memory_samples else self.memory_after_mb
        
        # Final CPU measurement with interval to get meaningful reading
        try:
            final_cpu = self._process.cpu_percent(interval=0.1)
            if final_cpu is not None and final_cpu >= 0:
                self._cpu_samples.append(final_cpu)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        # Calculate average CPU usage, ensuring we have valid samples
        valid_cpu_samples = [cpu for cpu in self._cpu_samples if cpu is not None and cpu >= 0]
        self.cpu_percent = sum(valid_cpu_samples) / len(valid_cpu_samples) if valid_cpu_samples else None
        
        # Calculate duration
        self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def sample_metrics(self):
        """Take a sample of current memory and CPU usage during benchmark execution."""
        try:
            # Sample memory
            current_memory = self._process.memory_info().rss / 1024**2
            self._memory_samples.append(current_memory)
            self.memory_peak_mb = max(self.memory_peak_mb or 0, current_memory)
            
            # Sample CPU (with short interval to get meaningful reading)
            current_cpu = self._process.cpu_percent(interval=0.05)
            if current_cpu is not None and current_cpu >= 0:
                self._cpu_samples.append(current_cpu)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process might have changed, ignore sampling errors
            pass
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def clean_for_json(obj):
            """Recursively clean object for JSON serialization."""
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj]
            elif obj is None:
                return None
            elif isinstance(obj, (str, bool)):
                return obj
            else:
                # Try to convert to string for other types
                try:
                    return str(obj)
                except:
                    return repr(obj)
        
        return {
            'name': self.name,
            'duration_ms': round(self.duration_ms, 2) if self.duration_ms else None,
            'memory_before_mb': round(self.memory_before_mb, 2) if self.memory_before_mb else None,
            'memory_after_mb': round(self.memory_after_mb, 2) if self.memory_after_mb else None,
            'memory_delta_mb': round(self.memory_delta_mb, 2) if self.memory_delta_mb else None,
            'memory_peak_mb': round(self.memory_peak_mb, 2) if self.memory_peak_mb else None,
            'cpu_percent': round(self.cpu_percent, 2) if self.cpu_percent else None,
            'metadata': clean_for_json(self.metadata)
        }

class BenchmarkCollector:
    """Collects and stores benchmark results."""
    
    def __init__(self):
        self.results = []
        self.session_start = time.time()
        
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
        
    def save_results(self, output_dir: Path):
        """Save all results to JSON file with comprehensive structure."""
        output_dir.mkdir(exist_ok=True)
        
        # Categorize results by test type
        results_by_category = {
            'throughput_tests': [],
            'latency_tests': [],
            'inference_tests': [],
            'memory_tests': [],
            'concurrency_tests': [],
            'data_loading_tests': [],
            'api_tests': [],
            'ml_tests': [],
            'stress_tests': [],
            'other_tests': []
        }
        
        # Categorize each test result
        for r in self.results:
            result_dict = r.to_dict()
            test_name = result_dict['name'].lower()
            
            # Determine category based on test name and metadata
            if 'throughput' in test_name:
                results_by_category['throughput_tests'].append(result_dict)
            elif 'latency' in test_name:
                results_by_category['latency_tests'].append(result_dict)
            elif 'inference' in test_name:
                results_by_category['inference_tests'].append(result_dict)
            elif 'memory' in test_name:
                results_by_category['memory_tests'].append(result_dict)
            elif 'concurrent' in test_name or 'concurrency' in test_name:
                results_by_category['concurrency_tests'].append(result_dict)
            elif 'csv' in test_name or 'data' in test_name or 'loading' in test_name:
                results_by_category['data_loading_tests'].append(result_dict)
            elif 'api' in test_name:
                results_by_category['api_tests'].append(result_dict)
            elif 'ml' in test_name:
                results_by_category['ml_tests'].append(result_dict)
            elif 'stress' in test_name:
                results_by_category['stress_tests'].append(result_dict)
            else:
                results_by_category['other_tests'].append(result_dict)
        
        # Calculate summary statistics
        summary_stats = {
            'total_tests_run': len(self.results),
            'tests_by_category': {k: len(v) for k, v in results_by_category.items() if v},
            'session_duration_minutes': (time.time() - self.session_start) / 60,
            'avg_test_duration_ms': None,
            'total_memory_usage_mb': None,
            'performance_summary': {}
        }
        
        # Calculate averages
        if self.results:
            durations = [r.duration_ms for r in self.results if r.duration_ms]
            memory_deltas = [r.memory_delta_mb for r in self.results if r.memory_delta_mb is not None]
            
            if durations:
                summary_stats['avg_test_duration_ms'] = sum(durations) / len(durations)
                summary_stats['performance_summary']['fastest_test_ms'] = min(durations)
                summary_stats['performance_summary']['slowest_test_ms'] = max(durations)
            
            if memory_deltas:
                summary_stats['total_memory_usage_mb'] = sum(memory_deltas)
                summary_stats['performance_summary']['max_memory_usage_mb'] = max(memory_deltas)
                summary_stats['performance_summary']['avg_memory_usage_mb'] = sum(memory_deltas) / len(memory_deltas)
        
        # Create comprehensive session data
        session_data = {
            'benchmark_session': {
                'session_start': self.session_start,
                'session_end': time.time(),
                'session_duration_seconds': time.time() - self.session_start,
                'session_duration_minutes': (time.time() - self.session_start) / 60,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'summary': summary_stats
            },
            'system_info': {
                'cpu_count_logical': psutil.cpu_count(),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'memory_available_gb': psutil.virtual_memory().available / 1024**3,
                'python_version': sys.version,
                'platform': sys.platform
            },
            'test_results_by_category': results_by_category,
            'all_test_results': []  # Flat list for compatibility
        }
        
        # Add all results for backward compatibility and detailed analysis
        for r in self.results:
            try:
                session_data['all_test_results'].append(r.to_dict())
            except Exception as e:
                logger.warning(f"Failed to serialize result {r.name}: {e}")
                # Add a minimal entry
                session_data['all_test_results'].append({
                    'name': r.name,
                    'duration_ms': None,
                    'error': str(e),
                    'metadata': {'serialization_failed': True}
                })
        
        # Save with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_results_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(session_data, f, indent=2, cls=NumpyJSONEncoder)
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
            # Try to save a minimal version
            try:
                minimal_data = {
                    'benchmark_session': {
                        'session_start': self.session_start,
                        'error': str(e),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    },
                    'summary': {
                        'total_tests': len(self.results),
                        'error': 'Failed to save full results'
                    },
                    'all_test_results': [{'name': r.name, 'error': 'serialization_failed'} for r in self.results]
                }
                with open(output_file, 'w') as f:
                    json.dump(minimal_data, f, indent=2)
            except Exception as e2:
                logger.error(f"Failed to save even minimal results: {e2}")
                return None
            
        logger.info(f"Benchmark results saved to: {output_file}")
        print(f"\n✅ Complete benchmark results saved to: {output_file}")
        print(f"📊 Summary: {summary_stats['total_tests_run']} tests completed in {summary_stats['session_duration_minutes']:.2f} minutes")
        
        # Print category breakdown
        print("📈 Tests by category:")
        for category, count in summary_stats['tests_by_category'].items():
            if count > 0:
                print(f"   • {category.replace('_', ' ').title()}: {count} tests")
        
        return output_file

@pytest.fixture(scope="session")
def benchmark_collector():
    """Session-wide benchmark result collector."""
    collector = BenchmarkCollector()
    yield collector
    
    # Save results at end of session
    try:
        output_dir = Path("tests/benchmark/results")
        collector.save_results(output_dir)
    except Exception as e:
        logger.warning(f"Failed to save benchmark results: {e}")

@pytest.fixture(scope="session")
def optimal_device_config():
    """Create optimal device configuration for benchmarks."""
    if not DEVICE_OPTIMIZER_AVAILABLE:
        logger.warning("Device optimizer not available, using default configuration")
        return {}
    
    try:
        # Create device optimizer for performance mode
        optimizer = DeviceOptimizer(
            optimization_mode=OptimizationMode.PERFORMANCE,
            workload_type="inference",  # Benchmarks are primarily inference-like
            environment="auto",
            enable_specialized_accelerators=True,
            memory_reservation_percent=5.0,  # Use more memory for benchmarks
            power_efficiency=False,  # Prioritize performance over power
            auto_tune=True,
            debug_mode=False
        )
        
        # Get optimal configurations
        config = {
            'system_info': optimizer.get_system_info(),
            'quantization_config': optimizer.get_optimal_quantization_config(),
            'batch_processor_config': optimizer.get_optimal_batch_processor_config(),
            'preprocessor_config': optimizer.get_optimal_preprocessor_config(),
            'inference_engine_config': optimizer.get_optimal_inference_engine_config(),
            'training_engine_config': optimizer.get_optimal_training_engine_config()
        }
        
        logger.info("Optimal device configuration created for benchmarks")
        logger.info(f"CPU cores: {optimizer.cpu_count_logical}, Memory: {optimizer.total_memory_gb:.1f}GB")
        logger.info(f"Accelerators: {[acc.name for acc in optimizer.accelerators]}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to create optimal device configuration: {e}")
        return {}

@pytest.fixture
def benchmark_result(request, benchmark_collector, optimal_device_config, memory_monitor):
    """Create a benchmark result for the current test with optimal device configuration."""
    test_name = request.node.name
    result = BenchmarkResult(test_name)
    
    # Add device configuration metadata
    if optimal_device_config:
        result.metadata['device_config'] = {
            'cpu_cores': optimal_device_config.get('system_info', {}).get('cpu_count_logical', 0),
            'memory_gb': optimal_device_config.get('system_info', {}).get('total_memory_gb', 0),
            'accelerators': optimal_device_config.get('system_info', {}).get('accelerators', []),
            'optimization_mode': 'PERFORMANCE',
            'workload_type': 'inference'
        }
    
    # Enhanced monitoring: start benchmark with integrated memory monitoring
    original_start = result.start
    original_stop = result.stop
    
    def enhanced_start():
        original_start()
        # Additional monitoring setup
        memory_monitor.start_monitoring()
    
    def enhanced_stop():
        # Integrate memory monitor results
        monitor_stats = memory_monitor.get_stats()
        if monitor_stats.get('peak_mb'):
            result.memory_peak_mb = monitor_stats['peak_mb']
        
        # Take final samples for accuracy
        result.sample_metrics()
        original_stop()
        
        # Add monitoring metadata
        result.metadata['memory_monitoring'] = monitor_stats
    
    # Replace methods with enhanced versions
    result.start = enhanced_start
    result.stop = enhanced_stop
    
    yield result
    
    benchmark_collector.add_result(result)

@pytest.fixture
def memory_monitor():
    """Monitor memory usage during test execution."""
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.peak_memory = 0
            self.samples = []
            self.baseline_memory = None
            
        def start_monitoring(self):
            """Start memory monitoring with baseline."""
            gc.collect()
            self.baseline_memory = self.process.memory_info().rss / 1024**2
            self.peak_memory = self.baseline_memory
            self.samples = [self.baseline_memory]
            
        def sample(self):
            """Take a memory sample."""
            try:
                current = self.process.memory_info().rss / 1024**2
                self.samples.append(current)
                self.peak_memory = max(self.peak_memory, current)
                return current
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return self.peak_memory
            
        def get_stats(self):
            """Get memory statistics."""
            if not self.samples:
                return {}
            return {
                'peak_mb': self.peak_memory,
                'avg_mb': sum(self.samples) / len(self.samples),
                'min_mb': min(self.samples),
                'max_mb': max(self.samples),
                'baseline_mb': self.baseline_memory,
                'delta_from_baseline_mb': self.peak_memory - (self.baseline_memory or 0),
                'samples': len(self.samples)
            }
    
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    return monitor

def generate_test_csv(path: Path, rows: int, cols: int, seed: int = 42) -> Path:
    """Generate a test CSV file with specified dimensions."""
    np.random.seed(seed)
    
    # Generate data
    data = {}
    for i in range(cols - 1):
        if i % 3 == 0:
            # Numeric columns
            data[f'numeric_{i}'] = np.random.randn(rows)
        elif i % 3 == 1:
            # Integer columns
            data[f'integer_{i}'] = np.random.randint(0, 100, rows)
        else:
            # Categorical columns
            categories = [f'cat_{j}' for j in range(10)]
            data[f'category_{i}'] = np.random.choice(categories, rows)
    
    # Add target column
    data['target'] = np.random.randint(0, 2, rows)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    
    return path

@pytest.fixture
def test_data_generator():
    """Generate test datasets of various sizes."""
    def _generate(size: str = "small") -> Dict[str, Any]:
        """Generate test data based on size specification."""
        sizes = {
            "tiny": (100, 5),
            "small": (1000, 10),
            "medium": (10000, 50),
            "large": (100000, 100),
            "wide": (50000, 1000),
            "huge": (1000000, 50)  # This should trigger size limits
        }
        
        if size not in sizes:
            raise ValueError(f"Unknown size: {size}. Available: {list(sizes.keys())}")
            
        rows, cols = sizes[size]
        
        # Generate in-memory DataFrame
        np.random.seed(42)
        data = {}
        
        for i in range(cols - 1):
            if i % 4 == 0:
                data[f'float_{i}'] = np.random.randn(rows).astype(np.float32)
            elif i % 4 == 1:
                data[f'int_{i}'] = np.random.randint(0, 1000, rows)
            elif i % 4 == 2:
                categories = [f'cat_{j}' for j in range(min(20, rows // 10))]
                data[f'category_{i}'] = np.random.choice(categories, rows)
            else:
                data[f'bool_{i}'] = np.random.choice([True, False], rows)
        
        data['target'] = np.random.randint(0, 2, rows)
        df = pd.DataFrame(data)
        
        # Create temporary file
        temp_dir = Path(tempfile.gettempdir()) / "kolosal_benchmark"
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"test_data_{size}_{rows}x{cols}.csv"
        df.to_csv(temp_file, index=False)
        
        return {
            'dataframe': df,
            'file_path': temp_file,
            'size': size,
            'rows': rows,
            'cols': cols,
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
    
    return _generate

@pytest.fixture
def ui_test_environment():
    """Set up UI testing environment with fallback implementations."""
    # Force fallback implementations for controlled testing
    original_env = os.environ.get('FORCE_FALLBACK_IMPLEMENTATIONS', '0')
    os.environ['FORCE_FALLBACK_IMPLEMENTATIONS'] = '1'
    
    try:
        # Import UI components with fallback mode
        sys.path.insert(0, str(PROJECT_ROOT))
        
        # Mock some heavy imports to speed up tests
        import sys
        sys.modules['torch'] = Mock()
        sys.modules['transformers'] = Mock()
        sys.modules['sklearn'] = Mock()
        
        yield
        
    finally:
        os.environ['FORCE_FALLBACK_IMPLEMENTATIONS'] = original_env

@pytest.fixture
def benchmark_config(optimal_device_config):
    """Configuration for benchmark tests with optimal device settings."""
    base_config = {
        'timeout_seconds': 300,
        'memory_limit_mb': 2048,
        'max_file_size_mb': 100,
        'sample_sizes': ['tiny', 'small', 'medium'],
        'stress_iterations': 50,
        'concurrency_levels': [1, 5, 10]
    }
    
    # Apply device-specific optimizations
    if optimal_device_config:
        system_info = optimal_device_config.get('system_info', {})
        batch_config = optimal_device_config.get('batch_processor_config')
        
        # Adjust based on available resources
        cpu_cores = system_info.get('cpu_count_logical', 1)
        memory_gb = system_info.get('total_memory_gb', 4)
        
        # Optimize concurrency levels based on CPU cores
        base_config['concurrency_levels'] = [1, min(cpu_cores // 2, 5), min(cpu_cores, 10)]
        
        # Adjust memory limits based on available memory
        base_config['memory_limit_mb'] = int(memory_gb * 1024 * 0.8)  # Use 80% of available memory
        
        # Use optimal batch sizes from device config
        if batch_config and hasattr(batch_config, 'max_batch_size'):
            base_config['optimal_batch_size'] = batch_config.max_batch_size
            base_config['optimal_workers'] = getattr(batch_config, 'num_workers', cpu_cores)
        
        # Adjust stress iterations based on performance capabilities
        if memory_gb > 16 and cpu_cores >= 8:
            base_config['stress_iterations'] = 100  # More iterations for powerful systems
        elif memory_gb < 8 or cpu_cores < 4:
            base_config['stress_iterations'] = 25   # Fewer iterations for limited systems
    
    return base_config

@pytest.fixture(scope="session")
def benchmark_output_dir():
    """Create output directory for benchmark results."""
    output_dir = Path("tests/benchmark/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir

# Utility functions for benchmark tests
def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function execution and return result + duration in ms."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000
    return result, duration_ms

def measure_memory_usage(func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
    """Measure memory usage of a function call."""
    process = psutil.Process()
    
    # Clean up before measurement
    gc.collect()
    before = process.memory_info().rss / 1024**2
    
    # Execute function
    result = func(*args, **kwargs)
    
    # Measure after
    after = process.memory_info().rss / 1024**2
    delta = after - before
    
    return result, {
        'before_mb': before,
        'after_mb': after,
        'delta_mb': delta
    }

def monitor_benchmark_execution(benchmark_result, func, *args, sample_interval=0.2, **kwargs):
    """Execute a function while continuously monitoring memory and CPU usage."""
    import threading
    import time
    
    monitoring_active = threading.Event()
    monitoring_active.set()
    
    def monitoring_thread():
        """Background thread to sample metrics during execution."""
        sample_count = 0
        while monitoring_active.is_set():
            if hasattr(benchmark_result, 'sample_metrics'):
                try:
                    benchmark_result.sample_metrics()
                    sample_count += 1
                except Exception:
                    pass  # Ignore sampling errors
            time.sleep(sample_interval)
        
        # Log how many samples we collected
        if hasattr(benchmark_result, 'metadata'):
            benchmark_result.metadata['monitoring_samples'] = sample_count
    
    try:
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitoring_thread)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Small delay to ensure initial CPU baseline is established
        time.sleep(0.1)
        
        # Execute the actual function
        result = func(*args, **kwargs)
        
        return result
        
    finally:
        # Stop monitoring
        monitoring_active.clear()
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=1.0)  # Wait up to 1 second for thread to finish

def profile_cpu_usage(func, *args, duration_seconds=5, **kwargs):
    """Profile CPU usage during function execution."""
    process = psutil.Process()
    
    # Start monitoring CPU
    process.cpu_percent()  # Reset CPU measurement
    
    start_time = time.time()
    result = func(*args, **kwargs)
    
    # Sample CPU usage
    cpu_samples = []
    while time.time() - start_time < duration_seconds:
        cpu_sample = process.cpu_percent(interval=0.1)
        if cpu_sample is not None:
            cpu_samples.append(cpu_sample)
    
    return result, {
        'avg_cpu_percent': sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
        'max_cpu_percent': max(cpu_samples) if cpu_samples else 0,
        'samples': len(cpu_samples)
    }

def optimize_cpu_performance():
    """Apply CPU performance optimizations for benchmarks."""
    if not DEVICE_OPTIMIZER_AVAILABLE:
        return
    
    try:
        import os
        import threading
        
        # Set CPU affinity to use all available cores
        if hasattr(os, 'sched_setaffinity'):
            # Linux
            available_cpus = os.sched_getaffinity(0)
            os.sched_setaffinity(0, available_cpus)
        
        # Set thread count for NumPy operations
        try:
            import numpy as np
            if hasattr(np, '__config__'):
                # Try to set optimal thread count
                cpu_count = psutil.cpu_count(logical=True)
                os.environ['OMP_NUM_THREADS'] = str(cpu_count)
                os.environ['MKL_NUM_THREADS'] = str(cpu_count)
                os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
        except ImportError:
            pass
        
        # Set high process priority on Windows
        if os.name == 'nt':
            try:
                import psutil
                p = psutil.Process()
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            except (ImportError, AttributeError, PermissionError):
                pass
                
    except Exception as e:
        logger.debug(f"Failed to apply CPU optimizations: {e}")

def configure_threading_for_ml():
    """Configure optimal threading for ML operations."""
    try:
        import os
        
        # Get CPU info
        cpu_count = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False) or cpu_count
        
        # Set environment variables for various threading libraries
        # Use physical cores for compute-intensive operations
        os.environ['OMP_NUM_THREADS'] = str(physical_cores)
        os.environ['MKL_NUM_THREADS'] = str(physical_cores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(physical_cores)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(physical_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(physical_cores)
        
        # Configure scikit-learn threading
        try:
            from sklearn.utils import parallel_backend
            return parallel_backend('threading', n_jobs=physical_cores)
        except ImportError:
            pass
            
    except Exception as e:
        logger.debug(f"Failed to configure ML threading: {e}")
        
    return None

# Apply optimizations at module import
optimize_cpu_performance()
