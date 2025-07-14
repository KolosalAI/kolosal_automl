# Genta AutoML vs Standard ML Comparison Benchmark

This benchmarking suite provides comprehensive comparison between Genta AutoML training engine and standard scikit-learn approaches. It evaluates performance metrics including training speed, accuracy, memory usage, and scalability across multiple datasets and models.

## 🎯 Features

- **Fair Comparison**: Genta AutoML optimizations are disabled to ensure fair comparison
- **Multiple Datasets**: Supports real-world and synthetic datasets of various sizes
- **Model Diversity**: Tests different algorithms (Random Forest, Gradient Boosting, Linear models)
- **Comprehensive Metrics**: Training time, accuracy, memory usage, model size
- **Visual Reports**: Generates detailed HTML reports with charts and statistics
- **Statistical Analysis**: Includes significance testing and performance improvements
- **Configurable**: Predefined configurations for different testing scenarios

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Available disk space for results

### Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn psutil
```

### Genta AutoML Modules
The comparison requires Genta AutoML modules to be properly installed:
```bash
# Ensure Genta AutoML modules are in your Python path
# The benchmark will run standard ML only if Genta modules are unavailable
```

## 🚀 Quick Start

### 1. Basic Comparison
Run a quick comparison on small datasets:
```bash
python run_comparison.py
```

### 2. List Available Configurations
```bash
python run_comparison.py --list-configs
```

### 3. Run Specific Configuration
```bash
python run_comparison.py --config comprehensive_small
```

### 4. Custom Comparison
```bash
python run_comparison.py --custom --datasets iris wine breast_cancer --models random_forest logistic_regression
```

## 📊 Available Configurations

### `quick_comparison` (Default)
- **Datasets**: iris, wine, diabetes
- **Models**: random_forest, logistic_regression, ridge
- **Purpose**: Fast validation of setup and basic comparison
- **Runtime**: ~2-5 minutes

### `comprehensive_small`
- **Datasets**: iris, wine, breast_cancer, diabetes, synthetic_small_*
- **Models**: random_forest, gradient_boosting, logistic_regression, ridge
- **Purpose**: Comprehensive evaluation on manageable datasets
- **Runtime**: ~10-20 minutes

### `performance_test`
- **Datasets**: breast_cancer, digits, california_housing, synthetic_medium_*
- **Models**: random_forest, gradient_boosting
- **Purpose**: Performance evaluation on larger datasets
- **Runtime**: ~20-45 minutes

### `scalability_test`
- **Datasets**: synthetic_small_*, synthetic_medium_*, synthetic_large_*
- **Models**: random_forest
- **Purpose**: Scalability analysis across dataset sizes
- **Runtime**: ~15-30 minutes

### `algorithm_comparison`
- **Datasets**: breast_cancer, wine
- **Models**: random_forest, gradient_boosting, logistic_regression
- **Purpose**: Compare different algorithms on same datasets
- **Runtime**: ~5-10 minutes

## 🔧 Advanced Usage

### Direct Script Usage
You can also run the comparison script directly:

```bash
# Basic usage
python benchmark_comparison.py

# With custom parameters
python benchmark_comparison.py --output-dir ./my_results --datasets iris wine --models random_forest gradient_boosting --optimization random_search
```

### Configuration File
Modify `benchmark_comparison_config.json` to:
- Add new datasets
- Define custom model parameters
- Adjust comparison settings
- Change reporting options

## 📈 Understanding Results

### Output Files
Each comparison run generates:
- `comparison_results_TIMESTAMP.json`: Raw results data
- `comparison_report_TIMESTAMP.html`: Visual report with charts
- `comparison_charts_TIMESTAMP.png`: Performance comparison charts
- `comparison_benchmark_TIMESTAMP.log`: Detailed execution log

### Key Metrics

#### Performance Metrics
- **Training Time**: Time to train model including hyperparameter optimization
- **Prediction Time**: Time to make predictions on test set
- **Memory Usage**: Peak memory consumption during training
- **Model Size**: Estimated size of trained model in memory

#### Accuracy Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: R² Score, MSE, MAE, RMSE

#### Comparison Analysis
- **Speed Improvement**: Percentage improvement in training time
- **Memory Efficiency**: Percentage improvement in memory usage  
- **Accuracy Gain**: Percentage improvement in model accuracy
- **Statistical Significance**: P-values for performance differences

### Sample Results Interpretation

```
Training Speed: +15.3% compared to standard ML
```
This means Genta AutoML is 15.3% faster than standard scikit-learn.

```
Memory Usage: -8.2% compared to standard ML  
```
This means Genta AutoML uses 8.2% less memory than standard scikit-learn.

## 🎨 Report Features

### HTML Report Includes:
- **Executive Summary**: Key performance improvements
- **Detailed Metrics**: Side-by-side comparison tables
- **Visual Charts**: Box plots, scatter plots, trend analysis
- **Individual Results**: Detailed breakdown per dataset/model
- **Statistical Analysis**: Significance tests and confidence intervals
- **Conclusions**: Automated insights and recommendations

### Chart Types:
- Training time comparison (box plots)
- Test score comparison (box plots)  
- Memory usage comparison (box plots)
- Training time vs dataset size (scatter plot)
- Performance trends across datasets

## 🔍 Troubleshooting

### Common Issues

#### "Genta AutoML modules not available"
- Ensure Genta modules are properly installed
- Check Python path includes module directory
- Comparison will run with standard ML only

#### Memory Errors on Large Datasets
- Reduce dataset sizes in configuration
- Use `scalability_test` instead of `performance_test`
- Increase system RAM or use smaller batch sizes

#### Slow Performance
- Start with `quick_comparison` configuration
- Use fewer datasets/models in custom runs
- Check system resources and close other applications

#### Import Errors
```bash
# Install missing dependencies
pip install numpy pandas scikit-learn matplotlib seaborn psutil

# For development dependencies
pip install jupyter notebook ipython
```

### Performance Tips
- Run comparisons during off-peak hours
- Use SSD storage for faster I/O
- Close unnecessary applications to free memory
- Use `--optimization random_search` for faster results

## 📁 File Structure

```
comparison_benchmark/
├── benchmark_comparison.py          # Main comparison script
├── run_comparison.py               # Configuration-based runner
├── benchmark_comparison_config.json # Configuration file
├── comparison_results_*/           # Results directory
│   ├── comparison_results_*.json   # Raw results
│   ├── comparison_report_*.html    # HTML report
│   ├── comparison_charts_*.png     # Charts
│   └── *.log                      # Log files
└── README.md                      # This file
```

## 🤝 Contributing

To add new datasets or models:

1. **Add Dataset**: Update `benchmark_comparison_config.json`:
```json
"new_dataset": {
  "name": "my_dataset",
  "description": "Description of dataset",
  "expected_task": "classification|regression",
  "compatible_models": ["random_forest", "logistic_regression"]
}
```

2. **Add Model**: Implement model loading in `DatasetManager.load_dataset()`

3. **Add Configuration**: Define new comparison configuration in config file

## 📜 License

This comparison benchmark is part of the Genta AutoML project. Please refer to the main project license.

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review log files for detailed error information
- Ensure all dependencies are properly installed
- Verify Genta AutoML modules are accessible

---

**Happy Benchmarking! 🚀**
