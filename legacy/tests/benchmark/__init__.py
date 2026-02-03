# ---------------------------------------------------------------------
# tests/benchmark/__init__.py - Benchmark test suite
# ---------------------------------------------------------------------
"""
Performance and stress testing suite for kolosal-automl.

This package contains benchmark tests that are excluded from default pytest runs.
Use specific markers to run benchmarks:

Examples:
  # Run all benchmarks
  pytest -m benchmark tests/benchmark/

  # Run specific benchmark categories
  pytest -m "benchmark and data_loading" tests/benchmark/
  pytest -m "benchmark and ui" tests/benchmark/
  pytest -m "benchmark and ml" tests/benchmark/

  # Run stress tests
  pytest -m stress tests/benchmark/

  # Run memory tests
  pytest -m memory tests/benchmark/

  # Run consistency tests
  pytest -m consistency tests/benchmark/
"""

__version__ = "1.0.0"
__author__ = "Kolosal AutoML Team"
