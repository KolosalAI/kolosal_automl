# Kolosal AutoML Benchmark Suite

This directory contains the complete benchmark suite for Kolosal AutoML, including fair comparison frameworks, training engine benchmarks, and automated testing scripts.

## Directory Structure

```
benchmark/
├── __init__.py                        # Package initialization
├── standard_ml_benchmark.py           # Pure scikit-learn baseline benchmark
├── kolosal_automl_benchmark.py       # Fair comparison Kolosal AutoML benchmark
├── benchmark_comparison.py            # Legacy comprehensive comparison framework
├── training_engine_benchmark.py       # Original training engine benchmarks
├── configs/                          # Configuration files
│   ├── example_comparison_config.json # Example comparison configuration
│   └── benchmark_comparison_config.json # Legacy configuration
└── results/                          # Benchmark results and reports
    └── training_engine_benchmarks/   # Historical benchmark results
```

## New Fair Comparison System

The benchmark suite now includes a fair comparison system with these key components:

### Core Benchmark Scripts

- **standard_ml_benchmark.py**: Pure scikit-learn implementation without any optimizations
- **kolosal_automl_benchmark.py**: Kolosal AutoML with optimizations disabled for fair comparison
- **benchmark_comparison.py**: Legacy comprehensive comparison (still available)
- **training_engine_benchmark.py**: Original training engine benchmarks

### Main Entry Points (Root Directory)

- **run_kolosal_comparison.py**: Main comparison runner with multiple modes
- **run_standard_ml.py**: Run only standard ML benchmarks
- **run_kolosal_automl.py**: Run only Kolosal AutoML benchmarks (fair mode)
- **test_comparison_system.py**: Test system setup and validation

## Key Components

### Core Modules

- **benchmark_comparison.py**: Comprehensive comparison framework between Kolosal AutoML and standard ML approaches
- **training_engine_benchmark.py**: Dedicated training engine performance benchmarks
- **benchmark.py**: Main benchmark runner with configuration-based execution

### Scripts

- **run_comparison.py**: Execute predefined or custom comparison configurations
- **run_kolosal_comparison.py**: Quick Kolosal AutoML vs Standard ML comparisons

## Usage

### From Root Directory (Recommended)

Use the convenience scripts in the root directory:

```bash
# Run comparison benchmarks
python run_comparison.py --config quick_comparison

# Run Kolosal-specific comparisons  
python run_kolosal_comparison.py --mode comprehensive

# Run training engine benchmarks
python benchmark_runner.py config.json
```

### From Benchmark Directory

```bash
cd benchmark

# Run comparison scripts
python scripts/run_comparison.py --config comprehensive_small
python scripts/run_kolosal_comparison.py --mode quick

# Run benchmark directly
python benchmark.py config.json
```

### Programmatic Usage

```python
from benchmark import ComparisonBenchmarkRunner, BenchmarkRunner

# Comparison benchmarks
runner = ComparisonBenchmarkRunner("./results")
runner.run_single_comparison("iris", "random_forest")

# Training engine benchmarks
benchmark_runner = BenchmarkRunner("./results")
benchmark_runner.run_multiple_benchmarks(configurations)
```

## Configuration

Benchmark configurations are stored in `configs/benchmark_comparison_config.json`. This file defines:

- Dataset configurations (small, medium, large, synthetic)
- Model parameters and hyperparameter grids
- Optimization strategies
- Device and compute configurations
- Predefined comparison scenarios

## Results

All benchmark results are saved to the `results/` directory with timestamped subdirectories. Results include:

- JSON data files with detailed metrics
- HTML reports with visualizations
- CSV summaries for analysis
- Performance charts and plots

## Key Features

### Comparison Framework
- Side-by-side comparison of Kolosal AutoML vs Standard ML
- Multiple dataset sizes and types
- Statistical significance testing
- Performance metrics (speed, accuracy, memory)
- Comprehensive reporting with visualizations

### Training Engine Benchmarks
- Configurable benchmark execution
- Memory usage tracking
- Performance analytics
- Error handling and reporting
- Graceful shutdown and cleanup

### Scalability Testing
- Datasets from small (150 samples) to massive (10M+ samples)
- Performance analysis across different scales
- Memory efficiency comparisons
- Speed improvement measurements

## Running Benchmarks

### Quick Comparison
```bash
python run_comparison.py --config quick_comparison
```

### Comprehensive Testing
```bash
python run_comparison.py --config comprehensive_small --trials 3
```

### Custom Comparisons
```bash
python run_comparison.py --custom --datasets iris wine --models random_forest logistic_regression
```

### Large Scale Testing
```bash
python run_comparison.py --config massive_scale_test
```

## Configuration Examples

### Predefined Configurations
- `quick_comparison`: Fast test on small datasets
- `comprehensive_small`: Thorough test on small/medium datasets
- `performance_test`: Performance evaluation on larger datasets
- `scalability_test`: Scalability analysis across dataset sizes
- `massive_scale_test`: Large scale testing up to 10M samples

### Custom Configuration
```python
configurations = [
    {
        "dataset": "iris",
        "model": "random_forest",
        "device_config": "cpu_only",
        "optimizer_config": "random_search_small"
    }
]
```

## Output Files

Each benchmark run generates:
- `comparison_results_TIMESTAMP.json`: Raw benchmark data
- `comparison_report_TIMESTAMP.html`: Interactive HTML report
- `comparison_charts_TIMESTAMP.png`: Performance visualization charts
- `comparison_benchmark_TIMESTAMP.log`: Detailed execution log

## Dependencies

- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- psutil>=5.8.0

## Migration from Root Directory

The benchmark code has been restructured from the root directory to this organized structure. Legacy scripts in the root directory now act as convenience wrappers that forward to the benchmark package.

## Performance Tips

1. Use `--trials 1` for quick testing
2. Start with `quick_comparison` before running larger benchmarks
3. Monitor system resources during large scale tests
4. Use `--verbose` for detailed logging during debugging
