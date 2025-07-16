# Experiment Tracker (`modules/engine/experiment_tracker.py`)

## Overview

The Experiment Tracker provides comprehensive experiment tracking capabilities for machine learning workflows with multiple backend support including MLflow integration. It enables systematic tracking of experiments, metrics, parameters, and artifacts across different runs.

## Features

- **Multi-Backend Support**: Local storage and MLflow integration
- **Comprehensive Tracking**: Metrics, parameters, artifacts, and model versions
- **Visualization Support**: Automatic plot generation and comparison charts
- **Experiment Comparison**: Side-by-side comparison of different runs
- **Model Versioning**: Track model evolution and performance
- **Artifact Management**: Store and retrieve experiment artifacts
- **Performance Analytics**: Statistical analysis of experiment results

## Core Classes

### ExperimentTracker

Main experiment tracking class with multiple backend support:

```python
class ExperimentTracker:
    def __init__(
        self,
        output_dir: str = "./experiments",
        experiment_name: Optional[str] = None,
        backend: str = "local",
        mlflow_tracking_uri: Optional[str] = None,
        auto_log: bool = True
    )
```

**Parameters:**
- `output_dir`: Directory for local experiment storage
- `experiment_name`: Name of the experiment
- `backend`: Tracking backend ("local", "mlflow", "both")
- `mlflow_tracking_uri`: MLflow tracking server URI
- `auto_log`: Enable automatic logging of common metrics

## Usage Examples

### Basic Experiment Tracking

```python
from modules.engine.experiment_tracker import ExperimentTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Initialize experiment tracker
tracker = ExperimentTracker(
    output_dir="./ml_experiments",
    experiment_name="random_forest_classification",
    backend="local"
)

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start experiment run
run_id = tracker.start_run(run_name="rf_baseline")

# Log parameters
tracker.log_params({
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'n_features': X.shape[1]
})

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log metrics
tracker.log_metrics({
    'accuracy': accuracy,
    'train_accuracy': model.score(X_train, y_train),
    'test_accuracy': accuracy
})

# Log model
tracker.log_model(model, "random_forest_model")

# Log artifacts
classification_rep = classification_report(y_test, y_pred, output_dict=True)
tracker.log_artifact_data(classification_rep, "classification_report.json")

# Log plots
tracker.log_confusion_matrix(y_test, y_pred)
tracker.log_feature_importance(model, feature_names=[f"feature_{i}" for i in range(X.shape[1])])

# End run
tracker.end_run()

print(f"Experiment completed. Run ID: {run_id}")
```

### MLflow Integration

```python
from modules.engine.experiment_tracker import ExperimentTracker
import mlflow

# Initialize with MLflow backend
tracker = ExperimentTracker(
    experiment_name="automl_comparison",
    backend="mlflow",
    mlflow_tracking_uri="http://localhost:5000",  # MLflow server
    auto_log=True
)

# Set MLflow experiment
mlflow.set_experiment("automl_comparison")

# Multiple runs for hyperparameter tuning
hyperparams_grid = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 15}
]

best_accuracy = 0
best_run_id = None

for i, params in enumerate(hyperparams_grid):
    run_id = tracker.start_run(run_name=f"rf_experiment_{i}")
    
    # Log parameters
    tracker.log_params(params)
    
    # Train model with current parameters
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    # Log metrics
    tracker.log_metrics({
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'accuracy_diff': train_acc - test_acc
    })
    
    # Log model if it's the best so far
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_run_id = run_id
        tracker.log_model(model, "best_model")
        
        # Log additional artifacts for best model
        tracker.log_feature_importance(model)
        tracker.log_learning_curve(model, X_train, y_train)
    
    tracker.end_run()

print(f"Best run: {best_run_id} with accuracy: {best_accuracy:.4f}")

# Get experiment summary
summary = tracker.get_experiment_summary()
print("Experiment Summary:", summary)
```

### Advanced Experiment Management

```python
from modules.engine.experiment_tracker import ExperimentTracker
import numpy as np
import time

# Initialize tracker with comprehensive logging
tracker = ExperimentTracker(
    experiment_name="deep_learning_experiments",
    backend="both",  # Log to both local and MLflow
    auto_log=True
)

# Custom metrics logging
class CustomMetricsTracker:
    def __init__(self, tracker):
        self.tracker = tracker
        self.epoch_metrics = []
        
    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        """Log metrics for each epoch"""
        metrics = {
            f'epoch_{epoch}/train_loss': train_loss,
            f'epoch_{epoch}/val_loss': val_loss,
            f'epoch_{epoch}/train_accuracy': train_acc,
            f'epoch_{epoch}/val_accuracy': val_acc,
            f'epoch_{epoch}/learning_rate': lr
        }
        
        self.tracker.log_metrics(metrics, step=epoch)
        self.epoch_metrics.append(metrics)
    
    def log_training_summary(self):
        """Log summary statistics of training"""
        if not self.epoch_metrics:
            return
            
        # Calculate summary metrics
        train_losses = [m[k] for m in self.epoch_metrics for k in m.keys() if 'train_loss' in k]
        val_losses = [m[k] for m in self.epoch_metrics for k in m.keys() if 'val_loss' in k]
        
        summary = {
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_val_loss': val_losses[-1] if val_losses else 0,
            'min_val_loss': min(val_losses) if val_losses else 0,
            'training_stability': np.std(train_losses[-10:]) if len(train_losses) >= 10 else 0,
            'convergence_epoch': len(train_losses)
        }
        
        self.tracker.log_metrics(summary)

# Start comprehensive experiment
run_id = tracker.start_run(run_name="neural_network_v1")

# Log system information
import platform
import psutil

system_info = {
    'platform': platform.platform(),
    'python_version': platform.python_version(),
    'cpu_count': psutil.cpu_count(),
    'memory_gb': psutil.virtual_memory().total / (1024**3)
}
tracker.log_params(system_info)

# Simulate neural network training with custom metrics
custom_tracker = CustomMetricsTracker(tracker)

# Training simulation
n_epochs = 50
for epoch in range(n_epochs):
    # Simulate training metrics
    train_loss = 2.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.05)
    val_loss = 2.2 * np.exp(-epoch * 0.09) + np.random.normal(0, 0.08)
    train_acc = 1 - np.exp(-epoch * 0.15) + np.random.normal(0, 0.02)
    val_acc = 1 - np.exp(-epoch * 0.12) + np.random.normal(0, 0.03)
    lr = 0.01 * (0.95 ** epoch)
    
    # Log epoch metrics
    custom_tracker.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, lr)
    
    # Log additional metrics every 10 epochs
    if epoch % 10 == 0:
        tracker.log_metrics({
            'gradient_norm': np.random.uniform(0.1, 1.0),
            'weight_norm': np.random.uniform(10, 50),
            'memory_usage_mb': np.random.uniform(500, 1500)
        }, step=epoch)
    
    time.sleep(0.1)  # Simulate training time

# Log training summary
custom_tracker.log_training_summary()

# Log final model artifacts
tracker.log_artifact_data({
    'model_architecture': 'dense_network',
    'layers': [128, 64, 32, 10],
    'activation': 'relu',
    'optimizer': 'adam'
}, 'model_config.json')

tracker.end_run()
```

## Experiment Comparison and Analysis

### Compare Multiple Experiments

```python
from modules.engine.experiment_tracker import ExperimentComparator

# Initialize comparator
comparator = ExperimentComparator(tracker)

# Get all runs from an experiment
runs = tracker.get_experiment_runs("random_forest_classification")

# Compare runs
comparison = comparator.compare_runs(
    run_ids=[run['run_id'] for run in runs[-5:]],  # Last 5 runs
    metrics=['accuracy', 'train_accuracy', 'test_accuracy'],
    include_params=True
)

print("Run Comparison:")
for run_id, data in comparison.items():
    print(f"Run {run_id}:")
    print(f"  Parameters: {data['params']}")
    print(f"  Metrics: {data['metrics']}")
    print(f"  Best Metric: {data['best_metric']}")

# Generate comparison plots
comparator.plot_metric_comparison(
    run_ids=[run['run_id'] for run in runs],
    metric='accuracy',
    save_path="accuracy_comparison.png"
)

# Statistical analysis
stats = comparator.get_statistical_summary(
    run_ids=[run['run_id'] for run in runs],
    metric='accuracy'
)

print(f"Statistical Summary for accuracy:")
print(f"  Mean: {stats['mean']:.4f}")
print(f"  Std: {stats['std']:.4f}")
print(f"  Min: {stats['min']:.4f}")
print(f"  Max: {stats['max']:.4f}")
print(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
```

### Hyperparameter Optimization Analysis

```python
from modules.engine.experiment_tracker import HyperparameterAnalyzer

# Analyze hyperparameter importance
analyzer = HyperparameterAnalyzer(tracker)

# Get runs for analysis
experiment_runs = tracker.get_experiment_runs("random_forest_classification")

# Analyze parameter importance
importance_analysis = analyzer.analyze_parameter_importance(
    runs=experiment_runs,
    target_metric='accuracy',
    parameters=['n_estimators', 'max_depth', 'min_samples_split']
)

print("Parameter Importance Analysis:")
for param, importance in importance_analysis.items():
    print(f"  {param}: {importance:.4f}")

# Generate parameter interaction plots
analyzer.plot_parameter_interactions(
    runs=experiment_runs,
    parameters=['n_estimators', 'max_depth'],
    target_metric='accuracy',
    save_path="parameter_interactions.png"
)

# Find optimal parameter ranges
optimal_ranges = analyzer.find_optimal_ranges(
    runs=experiment_runs,
    target_metric='accuracy',
    percentile=90  # Top 10% of runs
)

print("Optimal Parameter Ranges (top 10% runs):")
for param, range_info in optimal_ranges.items():
    print(f"  {param}: {range_info['min']:.2f} - {range_info['max']:.2f}")
```

## Advanced Logging Features

### Custom Artifact Logging

```python
# Log custom artifacts
run_id = tracker.start_run("custom_artifacts_demo")

# Log numpy arrays
feature_importance = np.random.rand(20)
tracker.log_numpy_array(feature_importance, "feature_importance.npy")

# Log pandas DataFrames
import pandas as pd
results_df = pd.DataFrame({
    'epoch': range(50),
    'train_loss': np.random.rand(50),
    'val_loss': np.random.rand(50)
})
tracker.log_dataframe(results_df, "training_history.csv")

# Log custom plots
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(results_df['epoch'], results_df['train_loss'], label='Train Loss')
plt.plot(results_df['epoch'], results_df['val_loss'], label='Val Loss')
plt.legend()
plt.title('Training History')
tracker.log_plot(plt, "training_curves.png")
plt.close()

# Log text files
training_log = """
Training Configuration:
- Model: RandomForest
- Features: 20
- Training samples: 800
- Validation samples: 200

Training completed successfully.
Best validation accuracy: 0.92
"""
tracker.log_text(training_log, "training_log.txt")

tracker.end_run()
```

### Metric Streaming and Real-time Monitoring

```python
from modules.engine.experiment_tracker import RealTimeTracker
import threading
import time

# Real-time tracking for long-running experiments
realtime_tracker = RealTimeTracker(tracker)

def long_running_experiment():
    """Simulate a long-running experiment with real-time updates"""
    run_id = realtime_tracker.start_run("long_running_experiment")
    
    # Initialize real-time dashboard
    realtime_tracker.start_dashboard(port=8080)
    
    for iteration in range(1000):
        # Simulate some computation
        time.sleep(0.1)
        
        # Generate metrics
        loss = 2.0 * np.exp(-iteration * 0.001) + np.random.normal(0, 0.1)
        accuracy = 1 - np.exp(-iteration * 0.002) + np.random.normal(0, 0.02)
        
        # Stream metrics in real-time
        realtime_tracker.stream_metrics({
            'iteration': iteration,
            'loss': loss,
            'accuracy': accuracy,
            'memory_usage': np.random.uniform(100, 500)
        })
        
        # Log checkpoints every 100 iterations
        if iteration % 100 == 0:
            realtime_tracker.save_checkpoint({
                'iteration': iteration,
                'model_state': f"checkpoint_{iteration}",
                'best_accuracy': max(accuracy, realtime_tracker.get_best_metric('accuracy'))
            })
    
    realtime_tracker.end_run()

# Run experiment in background thread
experiment_thread = threading.Thread(target=long_running_experiment)
experiment_thread.start()

# Monitor progress
print("Experiment running... Check dashboard at http://localhost:8080")
experiment_thread.join()
```

## Integration with AutoML

### AutoML Experiment Tracking

```python
from modules.engine.experiment_tracker import AutoMLTracker
from modules.engine.train_engine import MLTrainingEngine

# Specialized tracker for AutoML experiments
automl_tracker = AutoMLTracker(
    experiment_name="automl_benchmark",
    track_hyperparameter_search=True,
    track_model_selection=True,
    track_ensemble_creation=True
)

# Initialize training engine with tracker
engine = MLTrainingEngine(
    experiment_tracker=automl_tracker,
    enable_automl=True
)

# Run AutoML experiment
run_id = automl_tracker.start_run("automl_full_pipeline")

# The engine will automatically log:
# - Algorithm selection process
# - Hyperparameter optimization trials
# - Cross-validation results
# - Feature selection steps
# - Model ensemble creation
# - Final model evaluation

results = engine.fit(X_train, y_train)

# Get comprehensive AutoML report
automl_report = automl_tracker.get_automl_report()
print("AutoML Experiment Report:")
print(f"  Total algorithms tested: {automl_report['algorithms_tested']}")
print(f"  Best algorithm: {automl_report['best_algorithm']}")
print(f"  Hyperparameter trials: {automl_report['hyperparameter_trials']}")
print(f"  Feature selection iterations: {automl_report['feature_selection_iterations']}")
print(f"  Final ensemble size: {automl_report['ensemble_size']}")
print(f"  Cross-validation score: {automl_report['cv_score']:.4f}")

automl_tracker.end_run()
```

## Best Practices

### 1. Experiment Organization

```python
# Organize experiments hierarchically
tracker = ExperimentTracker(
    experiment_name="model_development/random_forest/hyperparameter_tuning",
    backend="mlflow"
)

# Use consistent naming conventions
run_name = f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_v{version}"
run_id = tracker.start_run(run_name=run_name)

# Tag runs for easy filtering
tracker.set_tags({
    'model_type': 'random_forest',
    'experiment_phase': 'hyperparameter_tuning',
    'dataset_version': 'v2.1',
    'developer': 'data_scientist_1'
})
```

### 2. Comprehensive Logging

```python
# Log everything that might be relevant
tracker.log_params({
    # Model parameters
    **model_params,
    
    # Data parameters
    'train_size': len(X_train),
    'test_size': len(X_test),
    'feature_count': X_train.shape[1],
    'class_balance': dict(zip(*np.unique(y_train, return_counts=True))),
    
    # System parameters
    'random_seed': random_seed,
    'cpu_count': os.cpu_count(),
    'python_version': platform.python_version(),
    
    # Preprocessing parameters
    'scaling_method': 'standard',
    'feature_selection': 'variance_threshold',
})
```

### 3. Error Handling and Recovery

```python
try:
    run_id = tracker.start_run("experiment_with_error_handling")
    
    # Your experiment code here
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    
    tracker.log_metrics(metrics)
    tracker.log_model(model, "final_model")
    
except Exception as e:
    # Log error information
    tracker.log_params({
        'error_occurred': True,
        'error_type': type(e).__name__,
        'error_message': str(e)
    })
    
    # Log stack trace as artifact
    tracker.log_text(traceback.format_exc(), "error_traceback.txt")
    
    # Mark run as failed
    tracker.set_run_status("FAILED")
    
    raise  # Re-raise the exception
    
finally:
    # Always end the run
    tracker.end_run()
```

## Related Documentation

- [Training Engine Documentation](train_engine.md)
- [Model Manager Documentation](../model_manager.md)
- [Configuration System Documentation](../configs.md)
- [Performance Metrics Documentation](performance_metrics.md)
