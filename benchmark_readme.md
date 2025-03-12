# Genta AutoML Benchmarking Suite

This benchmarking suite allows you to evaluate and optimize the performance of the Genta AutoML system, with a focus on training and inference engines.

## Features

- **Comprehensive Benchmarking**: Evaluate both training and inference performance
- **Configuration Testing**: Test different parameter configurations to find optimal settings
- **Quantization Analysis**: Measure the impact of different quantization strategies
- **Device Optimization**: Leverage automatic device optimization for best performance
- **Detailed Reporting**: Generate comprehensive reports with performance metrics and visualizations

## Installation

Ensure you have all required dependencies installed:

```bash
pip install numpy pandas matplotlib scikit-learn tqdm psutil
```

## Benchmarking Tools

The suite includes three main benchmarking tools:

### 1. Comprehensive Benchmark (`benchmark_script.py`)

This script provides a comprehensive benchmark of both training and inference engines.

```bash
python benchmark_script.py --mode both --models RandomForestClassifier LogisticRegression --task-types CLASSIFICATION --samples 10000 --features 20 --iterations 2 --batch-sizes 1 16 64 256 --output-dir ./benchmark_results
```

**Key Parameters:**
- `--mode`: Benchmark mode (`training`, `inference`, or `both`)
- `--models`: Models to benchmark
- `--task-types`: Task types to benchmark (`CLASSIFICATION`, `REGRESSION`)
- `--samples`: Number of samples in synthetic data
- `--features`: Number of features in synthetic data
- `--iterations`: Number of iterations for training benchmark
- `--batch-sizes`: Batch sizes for inference benchmark
- `--output-dir`: Directory to save benchmark results

### 2. Inference Engine Benchmark (`inference_benchmark.py`)

This script focuses on benchmarking the inference engine with various configurations.

```bash
python inference_benchmark.py --model-path ./models/RandomForestClassifier_CLASSIFICATION.pkl --model-type SKLEARN --samples 10000 --features 20 --batch-sizes 1 16 64 128 256 --iterations 10 --enable-quantization --enable-optimization --output-dir ./inference_benchmark
```

**Key Parameters:**
- `--model-path`: Path to the trained model file
- `--model-type`: Type of the model (`SKLEARN`, `XGBOOST`, `LIGHTGBM`, `CUSTOM`, `ENSEMBLE`)
- `--samples`: Number of samples for synthetic data
- `--features`: Number of features for synthetic data
- `--batch-sizes`: Batch sizes to benchmark
- `--iterations`: Number of iterations for each batch size
- `--enable-quantization`: Enable quantization for inference
- `--enable-optimization`: Enable Intel and memory optimizations
- `--adaptive-batching`: Enable adaptive batching
- `--output-dir`: Directory to save benchmark results

### 3. Quantization Benchmark (`quantization_benchmark.py`)

This script evaluates different quantization configurations to find the optimal strategy.

```bash
python quantization_benchmark.py --samples 10000 --features 100 --iterations 10 --quantization-types INT8 UINT8 INT16 --quantization-modes SYMMETRIC ASYMMETRIC DYNAMIC_PER_BATCH --output-dir ./quantization_benchmark
```

**Key Parameters:**
- `--samples`: Number of samples for synthetic data
- `--features`: Number of features for synthetic data
- `--iterations`: Number of iterations for each configuration
- `--quantization-types`: Quantization types to benchmark (`INT8`, `UINT8`, `INT16`)
- `--quantization-modes`: Quantization modes to benchmark (`SYMMETRIC`, `ASYMMETRIC`, `DYNAMIC_PER_BATCH`, `DYNAMIC_PER_CHANNEL`)
- `--output-dir`: Directory to save benchmark results

## Example Workflows

### 1. Optimize Training Configuration

```bash
# Run training benchmark with different models
python benchmark_script.py --mode training --models RandomForestClassifier LogisticRegression DecisionTreeClassifier --task-types CLASSIFICATION --samples 10000 --features 20 --iterations 3 --output-dir ./training_benchmark
```

### 2. Find Optimal Batch Size for Inference

```bash
# Run inference benchmark with different batch sizes
python inference_benchmark.py --model-path ./models/RandomForestClassifier_CLASSIFICATION.pkl --model-type SKLEARN --samples 5000 --features 20 --batch-sizes 1 4 8 16 32 64 128 256 --iterations 20 --output-dir ./batch_size_benchmark
```

### 3. Evaluate Quantization Impact on Accuracy

```bash
# Run quantization benchmark with focus on accuracy
python quantization_benchmark.py --samples 10000 --features 100 --iterations 10 --output-dir ./quantization_accuracy_benchmark
```

## Interpreting Results

The benchmark tools generate three types of output files:

1. **JSON Results**: Raw benchmark data for further analysis
2. **Markdown Reports**: Human-readable reports with key findings and recommendations
3. **Visualization Plots**: Performance charts comparing different configurations

### Key Metrics to Consider

#### Training Metrics:
- Training time
- Memory usage
- Model accuracy metrics (e.g., F1 score, RMSE)
- Feature selection efficiency

#### Inference Metrics:
- Latency (average, P95, P99)
- Throughput (samples/second)
- Memory usage per sample
- Batch size scaling efficiency

#### Quantization Metrics:
- Quantization/dequantization time
- Throughput
- Accuracy loss (mean absolute error)
- Compression ratio

## Best Practices

1. **Start with Default Configurations**: Begin with optimized configurations from `DeviceOptimizer`
2. **Benchmark Real Workloads**: When possible, use real data that matches your production scenarios
3. **Prioritize Your Requirements**: Focus on metrics that matter most for your use case (latency vs. throughput)
4. **Consider Memory Constraints**: Enable memory optimization for memory-constrained environments
5. **Test Multiple Batch Sizes**: Batch size can have significant impacts on throughput and memory usage

## Troubleshooting

If you encounter issues during benchmarking:

1. **Memory Errors**: Reduce data size or batch size
2. **Slow Performance**: Check if you've enabled optimizations appropriate for your hardware
3. **Error Messages**: Check the logs for specific error messages
4. **Inconsistent Results**: Increase the number of iterations for more reliable measurements

## Contributing

Contributions to improve the benchmarking suite are welcome! Please consider:

1. Adding support for new model types
2. Enhancing visualization capabilities
3. Adding more sophisticated analysis of benchmark results
4. Improving device-specific optimizations

## License

This benchmarking suite is released under the same license as the Genta AutoML system.