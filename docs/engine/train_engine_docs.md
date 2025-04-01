# ML Training Engine Documentation

## Overview

The ML Training Engine is a powerful and comprehensive framework for training, optimizing, and deploying machine learning models. The system provides end-to-end capabilities from data preprocessing to model optimization, evaluation, and inference.

## Key Components

### MLTrainingEngine

The core class that orchestrates the entire training and optimization process. It provides:

- Hyperparameter optimization with multiple strategies (Grid Search, Random Search, Bayesian)
- Automated feature selection
- Pipeline creation with preprocessing
- Model evaluation and comparison
- Batch inference capabilities
- Experiment tracking and reporting

### ExperimentTracker

Tracks experiments and metrics during model training:

- Records metrics, configurations, and feature importance
- Generates reports and visualizations
- Provides experiment logging
- Maintains history of experiments for comparison

### Supporting Components

- **DataPreprocessor**: Handles data preparation
- **BatchProcessor**: Manages batch processing for large datasets
- **InferenceEngine**: Optimized prediction interface
- **Quantizer**: Model compression for deployment

## Configuration

The engine is highly configurable through the `MLTrainingEngineConfig` class:

```python
from modules.configs import TaskType, OptimizationStrategy, MLTrainingEngineConfig

config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
    feature_selection=True,
    feature_selection_k=20,
    cv_folds=5,
    model_path="./models",
    experiment_tracking=True
)
```

## Usage Examples

### Basic Training Flow

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.datasets import load_iris

# Initialize the engine
engine = MLTrainingEngine(config)

# Load data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Define model and hyperparameter grid
model = RandomForestClassifier()
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

# Train model
best_model, metrics = engine.train_model(
    model=model,
    model_name="random_forest",
    param_grid=param_grid,
    X=X,
    y=y
)

# Save model
engine.save_model()

# Make predictions
predictions = engine.predict(X_test)
```

### Multiple Model Comparison

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Train multiple models
models = {
    "rf": RandomForestClassifier(),
    "gb": GradientBoostingClassifier(),
    "lr": LogisticRegression()
}

param_grids = {
    "rf": {...},  # RandomForest params
    "gb": {...},  # GradientBoosting params
    "lr": {...}   # LogisticRegression params
}

for name, model in models.items():
    engine.train_model(
        model=model,
        model_name=name,
        param_grid=param_grids[name],
        X=X,
        y=y
    )

# Evaluate all models
results = engine.evaluate_all_models(X_test, y_test)

# Generate comparison report
engine.generate_report("model_comparison.html")
```

## Advanced Features

### Feature Selection

The engine supports automated feature selection:

```python
config = MLTrainingEngineConfig(
    feature_selection=True,
    feature_selection_method="mutual_info",  # or "f_classif"
    feature_selection_k=15  # Select top 15 features
)
```

### Experiment Tracking

Track experiments and generate reports:

```python
# Experiments are automatically tracked when enabled
engine = MLTrainingEngine(config)

# After training
# Generate experiment report
if engine.tracker:
    report = engine.tracker.generate_report(include_plots=True)
```

### Batch Processing for Large Datasets

```python
def data_generator():
    # Generator function yielding batches of data
    for i in range(0, len(data), 1000):
        yield data[i:i+1000]

results = engine.run_batch_inference(data_generator(), batch_size=1000)
```

## API Reference

### MLTrainingEngine

- **train_model(model, model_name, param_grid, X, y, X_test=None, y_test=None)**: Trains and optimizes a model
- **predict(X, model_name=None)**: Makes predictions using specified or best model
- **save_model(model_name=None, filepath=None)**: Saves model to disk
- **load_model(filepath)**: Loads model from disk
- **evaluate_all_models(X_test, y_test)**: Evaluates all trained models
- **run_batch_inference(data_generator, batch_size=None, model_name=None)**: Runs inference in batches
- **generate_report(output_file=None)**: Generates performance report
- **shutdown()**: Releases resources

### ExperimentTracker

- **start_experiment(config, model_info)**: Starts a new experiment
- **log_metrics(metrics, step=None)**: Records metrics
- **log_feature_importance(feature_names, importance)**: Records feature importance
- **end_experiment()**: Finalizes the experiment
- **generate_report(include_plots=True)**: Creates experiment report

## Troubleshooting

### Common Issues

1. **Memory Issues**: For large datasets, enable memory optimization in config
2. **Slow Training**: Use batch processing and adjust optimization parameters
3. **Serialization Errors**: Ensure all components are serializable

### Logging

The system includes comprehensive logging:

```python
config = MLTrainingEngineConfig(
    log_level="DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR
)
```

## Best Practices

1. Always run `shutdown()` when finished to release resources
2. Use stratification for imbalanced classification tasks
3. Leverage experiment tracking for reproducible research
4. Save models after optimizing for deployment