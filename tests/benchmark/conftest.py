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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        
    def start(self):
        """Start timing and memory measurement."""
        gc.collect()  # Clean up before measurement
        self.start_time = time.perf_counter()
        self.memory_before_mb = psutil.Process().memory_info().rss / 1024**2
        
    def stop(self):
        """Stop timing and finalize measurements."""
        self.end_time = time.perf_counter()
        self.memory_after_mb = psutil.Process().memory_info().rss / 1024**2
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.memory_delta_mb = self.memory_after_mb - self.memory_before_mb
        self.cpu_percent = psutil.Process().cpu_percent()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'duration_ms': round(self.duration_ms, 2) if self.duration_ms else None,
            'memory_before_mb': round(self.memory_before_mb, 2) if self.memory_before_mb else None,
            'memory_after_mb': round(self.memory_after_mb, 2) if self.memory_after_mb else None,
            'memory_delta_mb': round(self.memory_delta_mb, 2) if self.memory_delta_mb else None,
            'memory_peak_mb': round(self.memory_peak_mb, 2) if self.memory_peak_mb else None,
            'cpu_percent': round(self.cpu_percent, 2) if self.cpu_percent else None,
            'metadata': self.metadata
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
        """Save all results to JSON file."""
        output_dir.mkdir(exist_ok=True)
        
        # Create session summary
        session_data = {
            'session_start': self.session_start,
            'session_duration': time.time() - self.session_start,
            'total_tests': len(self.results),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'python_version': sys.version,
                'platform': sys.platform
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        # Save with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        logger.info(f"Benchmark results saved to: {output_file}")
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

@pytest.fixture
def benchmark_result(request, benchmark_collector):
    """Create a benchmark result for the current test."""
    test_name = request.node.name
    result = BenchmarkResult(test_name)
    
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
            
        def sample(self):
            """Take a memory sample."""
            current = self.process.memory_info().rss / 1024**2
            self.samples.append(current)
            self.peak_memory = max(self.peak_memory, current)
            return current
            
        def get_stats(self):
            """Get memory statistics."""
            if not self.samples:
                return {}
            return {
                'peak_mb': self.peak_memory,
                'avg_mb': sum(self.samples) / len(self.samples),
                'min_mb': min(self.samples),
                'max_mb': max(self.samples),
                'samples': len(self.samples)
            }
    
    return MemoryMonitor()

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
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        'timeout_seconds': 300,
        'memory_limit_mb': 2048,
        'max_file_size_mb': 100,
        'sample_sizes': ['tiny', 'small', 'medium'],
        'stress_iterations': 50,
        'concurrency_levels': [1, 5, 10]
    }

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
        cpu_samples.append(process.cpu_percent(interval=0.1))
    
    return result, {
        'avg_cpu_percent': sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
        'max_cpu_percent': max(cpu_samples) if cpu_samples else 0,
        'samples': len(cpu_samples)
    }
