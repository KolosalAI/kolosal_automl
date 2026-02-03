# ---------------------------------------------------------------------
# tests/benchmark/README.md - Benchmark Test Suite Documentation
# ---------------------------------------------------------------------

# Benchmark Test Suite for Kolosal AutoML

This directory contains comprehensive benchmark tests for performance, consistency, and reliability testing of the Kolosal AutoML system.

## Overview

The benchmark suite is designed to:
- Test performance of critical system components
- Validate consistency across multiple runs
- Stress test under high load conditions
- Monitor memory usage and detect leaks
- Benchmark startup and import times

## Test Categories

### 🔧 Data Loading Benchmarks (`test_data_loading_benchmark.py`)
- CSV/Excel/JSON loading performance
- Sample data loader performance
- Data preview generation speed
- File validation performance
- Memory usage during data operations
- Stress testing with repeated operations

### 🎨 UI Performance Benchmarks (`test_ui_performance_benchmark.py`)
- UI component initialization time
- Data loading through UI interfaces
- Data preview refresh performance
- Unified data loading function tests
- UI memory usage monitoring
- Rapid operation stress tests

### 📦 Import & Startup Benchmarks (`test_import_benchmark.py`)
- Cold import performance testing
- Pydantic v2 compatibility patch overhead
- Fallback implementation performance
- Gradio import with patches
- UI initialization memory usage
- Import order independence

### 🤖 ML Operations Benchmarks (`test_ml_benchmark.py`)
- Dataset generation performance
- Model training and prediction speed
- Memory usage during ML operations
- ML reproducibility testing
- End-to-end pipeline performance
- Concurrent training simulation

### 🚀 Throughput Benchmarks (`test_throughput_benchmark.py`)
- Data processing throughput measurement
- Concurrent data loading performance
- Batch processing efficiency
- Inference engine throughput testing
- API endpoint throughput under load
- Sustained load handling

### 🧠 Memory Benchmarks (`test_memory_benchmark.py`)
- Memory leak detection in all components
- Peak memory usage monitoring
- Garbage collection efficiency
- Memory fragmentation analysis
- Memory usage under concurrent operations
- Memory pressure handling

## Running Benchmarks

### Quick Start

```powershell
# Run all benchmarks
pytest -m benchmark tests/benchmark/

# Run specific benchmark categories
pytest -m "benchmark and data_loading" tests/benchmark/
pytest -m "benchmark and ui" tests/benchmark/
pytest -m "benchmark and imports" tests/benchmark/
pytest -m "benchmark and ml" tests/benchmark/
pytest -m "benchmark and throughput" tests/benchmark/
pytest -m "benchmark and memory" tests/benchmark/
pytest -m "benchmark and inference" tests/benchmark/
```

### Advanced Usage

```powershell
# Run stress tests
pytest -m stress tests/benchmark/

# Run memory tests
pytest -m memory tests/benchmark/

# Run consistency tests
pytest -m consistency tests/benchmark/

# Run with verbose output and timing
pytest -m benchmark tests/benchmark/ -v --durations=10

# Save results to specific directory
pytest -m benchmark tests/benchmark/ --benchmark-output-dir=benchmark_results/
```

### Performance Testing

```powershell
# Quick performance check (fast tests only)
pytest -m "benchmark and not stress and not memory" tests/benchmark/ --durations=0

# Full stress testing (long running)
pytest -m "benchmark and stress" tests/benchmark/ --timeout=600

# Memory leak detection
pytest -m "benchmark and memory" tests/benchmark/ -v
```

## Configuration

### Pytest Markers

The tests use custom pytest markers:

- `benchmark` - All benchmark tests (excluded from default runs)
- `stress` - High-load stress tests
- `memory` - Memory usage and leak tests
- `consistency` - Data consistency and reproducibility tests
- `data_loading` - Data I/O performance tests
- `ui` - User interface performance tests
- `imports` - Import and startup performance tests
- `ml` - Machine learning operation tests
- `throughput` - Throughput and concurrent processing tests
- `memory` - Memory usage and leak tests
- `inference` - Inference engine performance tests

### Environment Variables

Set these environment variables for specific test behaviors:

```powershell
# Force fallback implementations for testing
$env:FORCE_FALLBACK_IMPLEMENTATIONS = "1"

# Enable Pydantic v1 compatibility mode
$env:PYDANTIC_V1 = "1"

# Set benchmark timeout
$env:PYTEST_TIMEOUT = "300"
```

### Benchmark Configuration

Default benchmark configuration can be found in `conftest.py`:

```python
benchmark_config = {
    'timeout_seconds': 300,
    'memory_limit_mb': 2048,
    'max_file_size_mb': 100,
    'sample_sizes': ['tiny', 'small', 'medium'],
    'stress_iterations': 50,
    'concurrency_levels': [1, 5, 10]
}
```

## Test Data Generation

The benchmark suite automatically generates test data of various sizes:

- **Tiny**: 100 rows × 5 columns
- **Small**: 1,000 rows × 10 columns  
- **Medium**: 10,000 rows × 50 columns
- **Large**: 100,000 rows × 100 columns
- **Wide**: 50,000 rows × 1,000 columns
- **Huge**: 1,000,000 rows × 50 columns (tests size limits)

## Results and Reporting

### Automatic Results Storage

Benchmark results are automatically saved to:
- `tests/benchmark/results/benchmark_results_YYYYMMDD_HHMMSS.json`

### Result Format

```json
{
  "session_start": 1640995200.0,
  "session_duration": 45.67,
  "total_tests": 25,
  "system_info": {
    "cpu_count": 8,
    "memory_total_gb": 16.0,
    "python_version": "3.10.0",
    "platform": "win32"
  },
  "results": [
    {
      "name": "test_csv_loading_performance",
      "duration_ms": 156.78,
      "memory_before_mb": 245.6,
      "memory_after_mb": 267.4,
      "memory_delta_mb": 21.8,
      "cpu_percent": 12.5,
      "metadata": {
        "file_size_mb": 5.2,
        "rows": 10000,
        "cols": 50
      }
    }
  ]
}
```

## Performance Baselines

### Expected Performance Ranges

| Operation | Small Data | Medium Data | Large Data |
|-----------|------------|-------------|------------|
| CSV Loading | < 100ms | < 500ms | < 2000ms |
| Data Preview | < 50ms | < 200ms | < 1000ms |
| UI Loading | < 200ms | < 1000ms | < 3000ms |
| Model Training | < 1000ms | < 5000ms | < 15000ms |
| Inference (Single) | < 10ms | < 50ms | < 100ms |
| Inference (Batch) | < 100ms | < 500ms | < 1000ms |
| Throughput (ops/sec) | > 100 | > 50 | > 10 |
| App Startup | < 5000ms | - | - |

### Memory Usage Guidelines

| Component | Expected Memory | Warning Threshold |
|-----------|----------------|-------------------|
| Data Loading | 2-3x file size | 5x file size |
| UI Components | < 100MB | > 200MB |
| ML Training | 3-5x data size | 10x data size |
| Inference Engine | < 50MB | > 100MB |
| App Startup | < 200MB | > 500MB |
| Memory Leaks | < 10MB/hour | > 50MB/hour |

## Troubleshooting

### Common Issues

1. **Tests Timeout**
   ```powershell
   # Increase timeout
   pytest -m benchmark tests/benchmark/ --timeout=600
   ```

2. **Memory Errors**
   ```powershell
   # Run with memory monitoring
   pytest -m "benchmark and not memory" tests/benchmark/
   ```

3. **Import Failures**
   ```powershell
   # Test with fallback implementations
   $env:FORCE_FALLBACK_IMPLEMENTATIONS = "1"
   pytest -m benchmark tests/benchmark/
   ```

4. **Slow Performance**
   ```powershell
   # Run quick tests only
   pytest -m "benchmark and not stress" tests/benchmark/ -k "not large and not huge"
   ```

### Debug Mode

Enable detailed debug output:

```powershell
pytest -m benchmark tests/benchmark/ -v -s --log-cli-level=DEBUG
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Benchmark Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Run nightly
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-benchmark
      
      - name: Run benchmarks
        run: |
          pytest -m "benchmark and not stress" tests/benchmark/
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: tests/benchmark/results/
```

## Contributing

### Adding New Benchmarks

1. Create test class with appropriate markers:
   ```python
   @pytest.mark.benchmark
   @pytest.mark.your_category
   class TestYourBenchmarks:
       pass
   ```

2. Use the benchmark utilities:
   ```python
   def test_your_benchmark(self, benchmark_result):
       benchmark_result.start()
       # Your test code here
       benchmark_result.stop()
       benchmark_result.metadata = {'key': 'value'}
   ```

3. Add appropriate assertions for performance expectations.

### Performance Regression Detection

Monitor key metrics across benchmark runs:
- Loading times should not increase by >20%
- Memory usage should not increase by >50%
- Success rates should remain >95%

For questions or issues, refer to the main project documentation or create an issue in the repository.
