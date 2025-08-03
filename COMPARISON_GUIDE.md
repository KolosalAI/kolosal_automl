# Kolosal AutoML vs Standard ML Comparison System

This system provides comprehensive benchmarking and comparison between Kolosal AutoML and standard ML approaches. It ensures fair comparison by running Kolosal AutoML with optimizations disabled to match the baseline performance characteristics of standard scikit-learn implementations.

## Overview

The comparison system consists of three main components:

1. **Standard ML Benchmark** - Pure scikit-learn implementation without optimizations
2. **Kolosal AutoML Benchmark** - Kolosal AutoML with optimizations disabled for fair comparison  
3. **Comparison Runner** - Unified interface that runs both benchmarks and generates comparison reports

## Quick Start

### Run a Quick Comparison
```bash
python run_kolosal_comparison.py --mode quick
```

### Run a Comprehensive Comparison
```bash
python run_kolosal_comparison.py --mode comprehensive
```

### Run Standard ML Benchmark Only
```bash
python run_standard_ml.py
```

### Run Kolosal AutoML Benchmark Only
```bash
python run_kolosal_automl.py
```

## Comparison Modes

### Predefined Modes

- **quick**: Fast comparison on small datasets (iris, wine)
- **comprehensive**: Thorough comparison across multiple datasets and models
- **optimization_strategies**: Compare different optimization strategies (grid search, random search, Bayesian, ASHT)
- **scalability**: Test performance across different dataset sizes
- **large_scale**: Large scale testing with bigger synthetic datasets

### Custom Mode

```bash
python run_kolosal_comparison.py --mode custom --datasets iris wine --models random_forest logistic_regression --optimization random_search
```

### Configuration File Mode

```bash
python run_kolosal_comparison.py --mode custom --config benchmark/configs/example_comparison_config.json
```

## Fair Comparison Approach

To ensure fair comparison, the Kolosal AutoML benchmark disables all optimizations:

### Disabled Optimizations in Kolosal AutoML:
- ✋ **Single-threaded execution** (n_jobs=1)
- ✋ **No Intel optimizations**
- ✋ **No memory optimizations** 
- ✋ **No adaptive batching**
- ✋ **No parallel preprocessing**
- ✋ **No feature selection**
- ✋ **No early stopping**
- ✋ **No experiment tracking**
- ✋ **Fixed batch sizes**
- ✋ **Simple processing strategies**

### What Remains Enabled:
- ✅ **Advanced optimization strategies** (ASHT, Bayesian optimization)
- ✅ **Same hyperparameter search space**
- ✅ **Same preprocessing steps**
- ✅ **Same cross-validation approach**
- ✅ **Framework architecture benefits**

## File Structure

```
kolosal-automl/
├── run_kolosal_comparison.py          # Main comparison runner
├── run_standard_ml.py                 # Standard ML benchmark runner
├── run_kolosal_automl.py             # Kolosal AutoML benchmark runner
├── benchmark/
│   ├── standard_ml_benchmark.py       # Pure scikit-learn implementation
│   ├── kolosal_automl_benchmark.py   # Fair comparison Kolosal AutoML
│   ├── benchmark_comparison.py        # Legacy comprehensive comparison
│   ├── training_engine_benchmark.py   # Original training engine benchmarks
│   └── configs/
│       └── example_comparison_config.json
└── comparison_results/                # Output directory
    ├── standard_ml/                   # Standard ML results
    ├── kolosal_automl/               # Kolosal AutoML results
    ├── comparison_report_*.html       # Comparison reports
    └── comparison_summary_*.json      # Comparison summaries
```

## Output Files

Each comparison run generates:

### Standard ML Results
- `standard_ml_results_TIMESTAMP.json` - Raw benchmark data
- `standard_ml_report_TIMESTAMP.html` - Standard ML performance report

### Kolosal AutoML Results  
- `kolosal_automl_results_TIMESTAMP.json` - Raw benchmark data
- `kolosal_automl_report_TIMESTAMP.html` - Kolosal AutoML performance report

### Comparison Results
- `comparison_report_TIMESTAMP.html` - Side-by-side comparison report
- `comparison_summary_TIMESTAMP.json` - Complete comparison data

## Key Metrics Compared

### Performance Metrics
- **Training Time** - Time to train the model including hyperparameter optimization
- **Prediction Time** - Time to make predictions on test set
- **Memory Usage** - Peak memory consumption during training

### ML Metrics
- **Test Score** - Accuracy for classification, R² for regression
- **Cross-validation Score** - Mean and standard deviation across folds
- **Training Score** - Performance on training set

### Additional Metrics
- **Dataset Size** - Number of samples and features
- **Best Parameters** - Optimal hyperparameters found
- **Success Rate** - Percentage of successful benchmarks

## Example Usage Scenarios

### 1. Basic Performance Comparison
```bash
# Quick test on small datasets
python run_kolosal_comparison.py --mode quick

# Output: Shows speed, accuracy, and memory comparison
```

### 2. Algorithm Strategy Comparison
```bash
# Compare optimization strategies
python run_kolosal_comparison.py --mode optimization_strategies

# Tests: random_search, grid_search, bayesian_optimization, asht
```

### 3. Scalability Analysis
```bash
# Test performance across dataset sizes
python run_kolosal_comparison.py --mode scalability

# Datasets: small (1K), medium (5K), large (20K) samples
```

### 4. Custom Dataset/Model Testing
```bash
# Test specific combinations
python run_kolosal_comparison.py --mode custom \
  --datasets breast_cancer diabetes \
  --models random_forest gradient_boosting \
  --optimization asht
```

### 5. Individual Benchmark Testing
```bash
# Test only standard ML
python run_standard_ml.py --datasets iris wine --models random_forest

# Test only Kolosal AutoML (fair comparison mode)
python run_kolosal_automl.py --datasets iris wine --models random_forest --optimization asht
```

## Understanding Results

### Speed Comparison
- **Positive %**: Kolosal AutoML is faster
- **Negative %**: Standard ML is faster

### Accuracy Comparison  
- **Positive %**: Kolosal AutoML is more accurate
- **Negative %**: Standard ML is more accurate

### Memory Comparison
- **Positive %**: Kolosal AutoML uses less memory
- **Negative %**: Standard ML uses less memory

## Expected Results

Since optimizations are disabled in Kolosal AutoML for fair comparison:

1. **Similar Performance**: Training times should be comparable
2. **Framework Overhead**: Kolosal AutoML may have slight overhead from framework architecture
3. **Algorithm Benefits**: Advanced optimization strategies (ASHT, Bayesian) may show accuracy improvements
4. **Consistent Results**: Multiple runs should show consistent patterns

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Kolosal AutoML modules are properly installed
2. **Memory Issues**: Use smaller datasets or reduce batch sizes
3. **Timeout Issues**: Increase timeout values for large datasets
4. **Module Not Found**: Check Python path and module imports

### Debug Mode
```bash
# Add verbose logging
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_kolosal_comparison.py --mode quick --verbose
```

## Configuration Options

### Supported Datasets
- **Real datasets**: iris, wine, breast_cancer, digits, diabetes, california_housing
- **Synthetic datasets**: synthetic_small_*, synthetic_medium_*, synthetic_large_*

### Supported Models  
- **Classification**: random_forest, gradient_boosting, logistic_regression
- **Regression**: random_forest, gradient_boosting, ridge, lasso

### Optimization Strategies
- **grid_search**: Exhaustive grid search
- **random_search**: Random parameter sampling  
- **bayesian_optimization**: Bayesian optimization (Kolosal enhancement)
- **asht**: Adaptive Sampling Hyperparameter Tuning (Kolosal proprietary)

## Advanced Usage

### Custom Configuration File
Create `my_config.json`:
```json
{
  "datasets": ["iris", "wine", "breast_cancer"],
  "models": ["random_forest", "gradient_boosting"], 
  "optimization": "asht",
  "description": "Custom test configuration"
}
```

Run with:
```bash
python run_kolosal_comparison.py --mode custom --config my_config.json
```

### Batch Processing
```bash
# Run multiple comparisons
for mode in quick comprehensive scalability; do
  python run_kolosal_comparison.py --mode $mode --output-dir results_$mode
done
```

This comparison system provides a robust, fair, and comprehensive way to evaluate Kolosal AutoML against standard ML approaches while ensuring meaningful and actionable insights.
