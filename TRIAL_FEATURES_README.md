# Enhanced Trial and Experiment Features

This document describes the new trial and experiment features added to the Kolosal AutoML comparison system.

## New Features Overview

### 1. Experiment Numbering System
- Each experiment now has a unique identifier in the format: `EXP_001_dataset_model_device_optimizer`
- Experiments are automatically numbered sequentially
- Easy tracking and identification of specific test runs

### 2. Multiple Trials per Experiment
- Support for running multiple trials of the same experiment configuration
- Configurable number of trials via `--trials` argument or config file
- Statistical analysis across trials to measure consistency and variance

### 3. Device Configuration Support
- Multiple device configurations: `cpu_only`, `cpu_multi_thread`, `cpu_high_memory`
- Ability to test performance across different hardware setups
- Memory limit and thread count specifications

### 4. Optimizer Configuration Support
- Multiple optimizer configurations: `random_search_small`, `random_search_medium`, `grid_search_small`
- Different hyperparameter search strategies and intensities
- Configurable cross-validation folds and iteration counts

### 5. Trial Result Plotting
- Automatic generation of trial analysis plots
- Four types of plots:
  - Training time across trials
  - Test score across trials  
  - Memory usage across trials
  - Score variability analysis (box plots)

### 6. Enhanced Statistics and Reporting
- Comprehensive trial statistics with mean, std, min, max
- Coefficient of variation analysis for consistency measurement
- Improved comparison reports with trial-level insights

## Configuration Updates

### benchmark_comparison_config.json

New sections added:

```json
{
  "comparison_settings": {
    "num_trials": 5,
    "enable_trial_plotting": true
  },
  
  "device_configurations": [
    {
      "name": "cpu_only",
      "description": "CPU only processing",
      "device": "cpu",
      "num_threads": 1,
      "memory_limit_gb": null
    },
    {
      "name": "cpu_multi_thread", 
      "description": "Multi-threaded CPU processing",
      "device": "cpu",
      "num_threads": 4,
      "memory_limit_gb": null
    }
  ],
  
  "optimizer_configurations": [
    {
      "name": "random_search_small",
      "strategy": "random_search",
      "max_iter": 10,
      "cv_folds": 3,
      "scoring": "auto",
      "random_state": 42
    },
    {
      "name": "grid_search_small",
      "strategy": "grid_search", 
      "cv_folds": 3,
      "scoring": "auto",
      "exhaustive": false
    }
  ]
}
```

## Usage Examples

### Basic Usage with Trials
```bash
# Run quick comparison with 3 trials per experiment
python run_comparison.py --config quick_comparison --trials 3

# Run custom comparison with 5 trials
python run_comparison.py --custom --datasets iris wine --models random_forest --trials 5
```

### Running Predefined Configurations
```bash
# All predefined configs now support multiple trials based on config file settings
python run_comparison.py --config comprehensive_small
python run_comparison.py --config massive_scale_test
```

## Output Structure

### Enhanced Results File
Results now include additional metadata:

```json
{
  "metadata": {
    "timestamp": "20250107_143000",
    "total_results": 30,
    "total_experiments": 6,
    "successful_results": 28
  },
  "results": [
    {
      "experiment_id": "EXP_001_iris_random_forest_cpu_only_random_search_small",
      "trial_number": 1,
      "approach": "standard",
      "dataset_name": "iris",
      "model_name": "random_forest",
      "device_config": "cpu_only",
      "optimizer_config": "random_search_small",
      "training_time": 1.23,
      "test_score": 0.95,
      "memory_peak_mb": 67.8,
      "success": true
    }
  ],
  "trial_statistics": {
    "total_trials": 30,
    "total_experiments": 6,
    "metrics_by_experiment": {...},
    "overall_comparison": {...},
    "variability_analysis": {...}
  }
}
```

### Generated Files
- `comparison_results_TIMESTAMP.json` - Enhanced results with trial statistics
- `plots/trial_plots_TIMESTAMP.png` - Trial analysis visualizations  
- `comparison_report_TIMESTAMP.html` - Updated comparison report
- `comparison_charts_TIMESTAMP.png` - Performance comparison charts

## Trial Statistics Analysis

### Metrics Tracked per Trial
- **Training Time**: Mean, std, min, max across trials
- **Test Score**: Performance consistency measurement
- **Memory Usage**: Peak and final memory consumption
- **Cross-validation Scores**: Model stability assessment

### Variability Analysis
- **Coefficient of Variation**: Relative variability measure
- **Range**: Max - min values across trials
- **IQR**: Interquartile range for outlier identification

## Performance Impact

### Multi-trial Benefits
- **Reliability**: Reduces impact of random initialization
- **Consistency**: Measures algorithmic stability
- **Statistical Significance**: More robust performance comparisons
- **Outlier Detection**: Identifies unusual performance cases

### Computational Cost
- Linear scaling with number of trials
- Recommended 3-5 trials for reliable results
- Use fewer trials for large datasets or extensive experiments

## Testing

Run the test script to verify all features:

```bash
python test_trial_features.py
```

This will:
1. Validate configuration loading
2. Test configuration generation
3. Simulate trial results and generate plots
4. Verify statistics generation

## Migration Notes

### Breaking Changes
- `BenchmarkResult` dataclass has new required fields
- `run_benchmark` methods have additional parameters
- Configuration combinations may generate more experiments

### Backwards Compatibility
- Default values provided for new parameters
- Existing configurations will work with defaults
- Single trial mode preserved as default behavior

## Best Practices

### Trial Configuration
- Use 3-5 trials for most experiments
- Increase trials for critical comparisons
- Consider computational cost vs. reliability trade-off

### Device/Optimizer Selection
- Test key combinations that match your deployment
- Limit combinations to avoid exponential explosion
- Focus on 1-2 device configs and 1-2 optimizer configs initially

### Result Analysis
- Review trial plots for consistency patterns
- Check coefficient of variation for reliability
- Compare mean performance across approaches
- Investigate high-variance experiments
