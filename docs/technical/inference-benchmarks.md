# Inference Model Size Benchmarks

This document describes the new inference benchmarks that test the same model architecture with several different sizes of machine learning models.

## Overview

The inference benchmarks have been designed to measure how model complexity and size affect inference performance. They test the same model types (Random Forest, XGBoost, LightGBM) with different configurations to understand scaling behavior.

## Available Benchmark Tests

### 1. Simple Inference Benchmark (`test_simple_inference_benchmark.py`)

A lightweight benchmark suitable for quick testing and CI/CD pipelines.

#### Tests Included:

- **`test_inference_with_different_model_sizes`**: Tests small, medium, and large Random Forest models
- **`test_batch_vs_single_prediction_efficiency`**: Compares single vs batch prediction performance
- **`test_data_size_impact_on_inference`**: Tests how different input data sizes affect performance

### 2. Comprehensive Inference Benchmark (`test_inference_model_size_benchmark.py`)

A comprehensive benchmark for detailed performance analysis across multiple model types and complexities.

#### Tests Included:

- **`test_inference_scaling_across_model_sizes`**: Full matrix of model types × complexities × dataset sizes
- **`test_model_complexity_impact_on_inference`**: Focused analysis of complexity impact
- **`test_inference_memory_scaling`**: Memory usage analysis during inference
- **`test_concurrent_inference_stress`**: Concurrent inference performance testing

## Model Configurations

### Dataset Sizes
- **Tiny**: 100 samples, 5 features
- **Small**: 1,000 samples, 10 features  
- **Medium**: 5,000 samples, 20 features
- **Large**: 10,000 samples, 50 features
- **XL**: 50,000 samples, 100 features

### Model Complexities
- **Simple**: 10 estimators, shallow depth
- **Medium**: 50 estimators, medium depth
- **Complex**: 100 estimators, deep trees
- **Very Complex**: 200 estimators, very deep trees

### Supported Models
- Random Forest (Classifier & Regressor)
- XGBoost (if available)
- LightGBM (if available)  
- Logistic/Linear Regression

## Running the Benchmarks

### Quick Test (Simple Benchmark)
```bash
# Run all simple inference benchmarks
python -m pytest tests/benchmark/test_simple_inference_benchmark.py -v -m "benchmark"

# Run specific test
python -m pytest tests/benchmark/test_simple_inference_benchmark.py::TestSimpleInferenceBenchmark::test_inference_with_different_model_sizes -v -m "benchmark"
```

### Comprehensive Analysis
```bash
# Run comprehensive inference benchmarks (takes longer)
python -m pytest tests/benchmark/test_inference_model_size_benchmark.py -v -m "benchmark"

# Run specific comprehensive test
python -m pytest tests/benchmark/test_inference_model_size_benchmark.py::TestInferenceModelSizeBenchmarks::test_inference_scaling_across_model_sizes -v -m "benchmark"
```

### Using the Benchmark Runner Script
```bash
# Run inference-specific benchmarks
python run_benchmarks.py --category inference

# Run quick inference benchmarks only
python run_benchmarks.py --category inference --quick
```

## Benchmark Metrics

### Performance Metrics
- **Single Prediction Time**: Average time for one prediction (ms)
- **Batch Prediction Time**: Time for batch predictions (ms)
- **Throughput**: Predictions per second
- **Memory Usage**: Memory consumption during inference
- **Model Size**: Serialized model size (MB)

### Batch Size Testing
The benchmarks test various batch sizes: 1, 5, 10, 25, 50, 100, 250, 500, 1000

### Performance Assertions
- Single predictions should complete within reasonable time limits
- Batch processing should be more efficient than single predictions
- Memory usage should scale reasonably with model complexity
- Concurrent inference should maintain performance

## Example Results

### Model Size Impact
```json
{
  "small": {
    "avg_single_prediction_ms": 0.43,
    "predictions_per_second": 2342,
    "model_size_mb": 0.033,
    "accuracy": 0.92
  },
  "medium": {
    "avg_single_prediction_ms": 1.28,
    "predictions_per_second": 782,
    "model_size_mb": 0.66,
    "accuracy": 0.91
  },
  "large": {
    "avg_single_prediction_ms": 2.40,
    "predictions_per_second": 417,
    "model_size_mb": 1.66,
    "accuracy": 0.94
  }
}
```

### Batch Efficiency
```json
{
  "single_prediction_stats": {
    "avg_time_ms": 1.4,
    "throughput_per_sec": 714
  },
  "batch_prediction_stats": {
    "batch_size": 100,
    "time_per_prediction_ms": 0.017,
    "throughput_per_sec": 59053
  },
  "efficiency_comparison": {
    "batch_vs_single_speedup": 82.7,
    "batch_faster": true
  }
}
```

## Interpreting Results

### Expected Patterns
1. **Model Size vs Speed**: Larger models should be slower but potentially more accurate
2. **Batch Efficiency**: Batch processing should show significant speedup over single predictions
3. **Memory Scaling**: Memory usage should increase with model complexity
4. **Throughput**: Should decrease as model complexity increases

### Performance Thresholds
- **Simple Models**: <100ms per prediction
- **Medium Models**: <500ms per prediction  
- **Complex Models**: <1000ms per prediction
- **Batch Speedup**: At least 20% improvement over single predictions

## Continuous Integration

The simple benchmark is designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Run Inference Benchmarks
  run: |
    python -m pytest tests/benchmark/test_simple_inference_benchmark.py -v -m "benchmark" --maxfail=1
```

## Extending the Benchmarks

### Adding New Model Types
1. Add model configuration to `InferenceModelConfig.MODEL_COMPLEXITIES`
2. Update `train_models()` method in `ModelSizeBenchmarkRunner`
3. Handle new model type in benchmark methods

### Adding New Metrics
1. Add metric calculation in benchmark methods
2. Update result dictionaries
3. Add validation assertions if needed

### Custom Dataset Sizes
Modify `InferenceModelConfig.DATASET_SIZES` to test with your specific data sizes.

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce dataset sizes or model complexity
2. **Timeout Errors**: Increase pytest timeout or reduce test scope  
3. **Library Missing**: Install optional dependencies (xgboost, lightgbm)
4. **Performance Failures**: Adjust assertion thresholds for your hardware

### Performance Tuning
- Use fewer estimators for faster tests
- Reduce dataset sizes for CI/CD
- Skip complex models if not needed
- Use parallel processing where available

## Results Storage

Benchmark results are automatically saved to `tests/benchmark/results/` with timestamps:
- JSON format for programmatic analysis
- Include system information and metadata
- Can be used for performance regression detection

## Platform Compatibility

### Windows Support

The inference benchmarks have been optimized for Windows compatibility:

**Known Issue & Solution:**
- **Problem**: Windows may encounter joblib subprocess errors during model training due to multiprocessing limitations
- **Solution**: The benchmarks automatically apply Windows-specific fixes:
  ```python
  # Windows compatibility: Force single-threaded execution
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'
  os.environ['OPENBLAS_NUM_THREADS'] = '1'
  ```
- **Impact**: Models use `n_jobs=1` to avoid subprocess issues, ensuring reliable execution

**Validated Compatibility:**
- ✅ Simple inference benchmarks: All tests pass
- ✅ Comprehensive model size benchmarks: Successfully tested with Windows-specific fixes
- ✅ sklearn, XGBoost, LightGBM: All work with single-threaded configuration

### Linux/macOS Support

All benchmarks run natively without modifications on Unix-based systems.

## Troubleshooting

### Common Issues

1. **"ValueError: cannot find context for 'threading'"**
   - **Cause**: Windows joblib multiprocessing incompatibility
   - **Solution**: Already fixed in benchmark files with environment variable settings
   - **Workaround**: Ensure `n_jobs=1` is used in all sklearn models

2. **Test timeouts during comprehensive benchmarks**
   - **Cause**: Large model training can be slow
   - **Solution**: Run with smaller model complexities first
   - **Command**: Use `-k "simple"` to run only simple complexity tests

3. **Memory issues with large datasets**
   - **Cause**: XL dataset size may exceed available memory
   - **Solution**: Reduce dataset sizes in the configuration
   - **Workaround**: Focus on small to medium sized datasets for development
