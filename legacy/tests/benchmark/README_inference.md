# Inference Benchmarks

This directory contains comprehensive inference benchmarks for testing model performance with different sizes and complexities.

## Files Created

1. **`test_simple_inference_benchmark.py`** - Lightweight benchmark for quick testing
2. **`test_inference_model_size_benchmark.py`** - Comprehensive benchmark for detailed analysis  
3. **`docs/technical/inference-benchmarks.md`** - Complete documentation

## Quick Start

### Run Simple Benchmarks (Fast)
```bash
# Run all simple inference benchmarks
python -m pytest tests/benchmark/test_simple_inference_benchmark.py -v -m "benchmark and inference"

# Or use the benchmark runner
python run_benchmarks.py --category inference --quick
```

### Run Comprehensive Benchmarks (Detailed)
```bash
# Run comprehensive analysis (takes longer)
python -m pytest tests/benchmark/test_inference_model_size_benchmark.py -v -m "benchmark and inference"

# Or use the benchmark runner
python run_benchmarks.py --category inference
```

## What the Benchmarks Test

### Model Size Impact
- **Small Models**: 10 estimators, depth 5
- **Medium Models**: 50 estimators, depth 10  
- **Large Models**: 100 estimators, depth 15

### Performance Metrics
- Single prediction latency (ms)
- Batch prediction throughput (predictions/sec)
- Memory usage during inference
- Model serialization size (MB)

### Scaling Analysis
- Different dataset sizes (100 to 50K samples)
- Various batch sizes (1 to 1000 predictions)
- Multiple model types (Random Forest, XGBoost, LightGBM)
- Concurrent inference performance

## Example Results

The benchmarks clearly show the performance trade-offs between model complexity and inference speed:

```
Small Model:  0.43ms per prediction, 2342 predictions/sec, 0.03MB size
Medium Model: 1.28ms per prediction,  782 predictions/sec, 0.66MB size  
Large Model:  2.40ms per prediction,  417 predictions/sec, 1.66MB size
```

Batch processing shows significant efficiency gains:
- Single predictions: ~714 predictions/sec
- Batch of 100: ~59,053 predictions/sec (82x speedup)

## Integration

The benchmarks integrate with the existing benchmark infrastructure:
- Results saved automatically to `tests/benchmark/results/`
- Compatible with CI/CD pipelines  
- Uses existing benchmark markers and fixtures
- Supports the `run_benchmarks.py` script

## Usage in CI/CD

For continuous integration, use the simple benchmark:

```yaml
- name: Run Inference Benchmarks
  run: python -m pytest tests/benchmark/test_simple_inference_benchmark.py -v -m "benchmark and inference" --timeout=300
```

## Performance Expectations

### Acceptable Ranges
- **Simple models**: <100ms per prediction
- **Medium models**: <500ms per prediction
- **Complex models**: <1000ms per prediction
- **Batch efficiency**: >20% speedup over single predictions

### Hardware Dependencies
Results will vary based on:
- CPU cores and speed
- Available memory
- System load
- Python/library versions

## See Also

- Full documentation: `docs/technical/inference-benchmarks.md`
- Existing ML benchmarks: `test_ml_benchmark.py`
- Throughput benchmarks: `test_throughput_benchmark.py`

## Platform Compatibility

### Windows Users ✅
The inference benchmarks include automatic Windows compatibility fixes for joblib subprocess issues. Tests run reliably with single-threaded model configurations.

### Quick Validation
```bash
# Test that everything works on your platform
python -m pytest tests/benchmark/test_simple_inference_benchmark.py -m "benchmark" -v

# For comprehensive testing (takes longer)
python -m pytest tests/benchmark/test_inference_model_size_benchmark.py -m "benchmark" -v
```
