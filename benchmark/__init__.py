"""
Kolosal AutoML Benchmark Package

This package contains all benchmark-related modules and scripts for comparing
Kolosal AutoML against standard ML approaches.

Structure:
- benchmark_comparison.py: Main comparison framework
- training_engine_benchmark.py: Training engine benchmarks
- benchmark.py: Main benchmark runner
- scripts/: Benchmark runner scripts
- configs/: Configuration files
- results/: Benchmark results and reports
"""

# Make key classes available at package level
from .benchmark_comparison import (
    ComparisonBenchmarkRunner,
    BenchmarkResult,
    DatasetManager,
    StandardMLBenchmark,
    KolosalMLBenchmark
)

try:
    from .training_engine_benchmark import BenchmarkRunner
    _training_engine_available = True
except ImportError:
    _training_engine_available = False
    BenchmarkRunner = None

__version__ = "2.0.0"
__author__ = "Kolosal AutoML Team"

__all__ = [
    "ComparisonBenchmarkRunner",
    "BenchmarkResult", 
    "DatasetManager",
    "StandardMLBenchmark",
    "KolosalMLBenchmark"
]

if _training_engine_available:
    __all__.append("BenchmarkRunner")
