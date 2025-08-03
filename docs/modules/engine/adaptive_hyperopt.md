# Adaptive Hyperparameter Optimization (`modules/engine/adaptive_hyperopt.py`)

## Overview

The Adaptive Hyperparameter Optimization module provides intelligent, dynamic hyperparameter tuning with adaptive search space adjustment. It integrates multiple optimization backends (Optuna, HyperOpt, Scikit-Optimize) and automatically adjusts search strategies based on optimization progress.

## Features

- **Multi-backend Support**: Optuna, HyperOpt, Scikit-Optimize integration
- **Adaptive Search Space**: Dynamic adjustment based on optimization progress
- **Early Stopping**: Intelligent pruning of unpromising trials
- **Parallel Optimization**: Multi-threaded trial execution
- **Caching System**: Persistent trial results and warm starts
- **Progress Tracking**: Real-time optimization monitoring
- **Ensemble Methods**: Multiple sampler combination strategies

## Core Classes

### AdaptiveHyperparameterOptimizer

Main optimizer class with adaptive capabilities:

```python
class AdaptiveHyperparameterOptimizer:
    def __init__(
        self,
        backend: str = "optuna",
        n_trials: int = 100,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        cache_dir: Optional[str] = None,
        enable_pruning: bool = True,
        adaptation_interval: int = 20,
        min_improvement_threshold: float = 0.001
    )
```

**Parameters:**
- `backend`: Optimization backend ("optuna", "hyperopt", "skopt")
- `n_trials`: Maximum number of optimization trials
- `timeout`: Maximum optimization time in seconds
- `n_jobs`: Number of parallel jobs
- `random_state`: Random seed for reproducibility
- `cache_dir`: Directory for caching trial results
- `enable_pruning`: Enable early stopping of trials
- `adaptation_interval`: Trials between adaptation checks
- `min_improvement_threshold`: Minimum improvement for adaptation

### OptimizationConfig

Configuration for optimization parameters:

```python
@dataclass
class OptimizationConfig:
    backend: str = "optuna"
    n_trials: int = 100
    timeout: Optional[float] = None
    n_jobs: int = 1
    random_state: Optional[int] = None
    enable_pruning: bool = True
    pruning_patience: int = 5
    adaptation_interval: int = 20
    min_improvement_threshold: float = 0.001
    sampler_config: Dict[str, Any] = None
    study_direction: str = "maximize"
```

## Usage Examples

### Basic Hyperparameter Optimization

```python
from modules.engine.adaptive_hyperopt import AdaptiveHyperparameterOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Define search space
search_space = {
    'n_estimators': ('int', 10, 200),
    'max_depth': ('int', 3, 20),
    'min_samples_split': ('int', 2, 20),
    'min_samples_leaf': ('int', 1, 10),
    'max_features': ('categorical', ['sqrt', 'log2', None])
}

# Define objective function
def objective(params):
    model = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()

# Initialize optimizer
optimizer = AdaptiveHyperparameterOptimizer(
    backend="optuna",
    n_trials=100,
    n_jobs=4,
    enable_pruning=True
)

# Run optimization
best_params, best_score, trials_history = optimizer.optimize(
    objective=objective,
    search_space=search_space,
    direction="maximize"
)

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score:.4f}")
```

### Advanced Configuration with Multiple Backends

```python
from modules.engine.adaptive_hyperopt import (
    AdaptiveHyperparameterOptimizer,
    OptimizationConfig
)

# Configure optimization
config = OptimizationConfig(
    backend="optuna",
    n_trials=200,
    timeout=3600,  # 1 hour
    n_jobs=8,
    enable_pruning=True,
    pruning_patience=10,
    adaptation_interval=25,
    min_improvement_threshold=0.001,
    sampler_config={
        'multivariate': True,
        'n_startup_trials': 20,
        'n_ei_candidates': 24
    }
)

# Initialize with configuration
optimizer = AdaptiveHyperparameterOptimizer.from_config(config)

# Run optimization with callbacks
def progress_callback(trial_number, best_score, current_score):
    print(f"Trial {trial_number}: Current={current_score:.4f}, Best={best_score:.4f}")

best_params, best_score, history = optimizer.optimize(
    objective=objective,
    search_space=search_space,
    direction="maximize",
    callbacks=[progress_callback]
)
```

### Multi-Objective Optimization

```python
# Define multi-objective function
def multi_objective(params):
    model = RandomForestClassifier(**params, random_state=42)
    
    # Accuracy score
    accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    accuracy = accuracy_scores.mean()
    
    # Model complexity (inverse of n_estimators for minimization)
    complexity = 1.0 / params['n_estimators']
    
    return accuracy, complexity

# Multi-objective optimization
optimizer = AdaptiveHyperparameterOptimizer(
    backend="optuna",
    n_trials=150,
    enable_pruning=True
)

pareto_front = optimizer.multi_objective_optimize(
    objective=multi_objective,
    search_space=search_space,
    directions=["maximize", "maximize"],  # maximize accuracy, maximize simplicity
    n_trials=150
)

print(f"Pareto optimal solutions: {len(pareto_front)}")
```

### Ensemble Optimization

```python
# Ensemble of optimizers
ensemble_config = {
    'optimizers': [
        {'backend': 'optuna', 'weight': 0.4},
        {'backend': 'hyperopt', 'weight': 0.3},
        {'backend': 'skopt', 'weight': 0.3}
    ],
    'combination_strategy': 'weighted_average',
    'n_trials_per_optimizer': 50
}

ensemble_optimizer = AdaptiveHyperparameterOptimizer.create_ensemble(
    ensemble_config
)

best_params, best_score, ensemble_history = ensemble_optimizer.optimize(
    objective=objective,
    search_space=search_space,
    direction="maximize"
)
```

## Search Space Definition

### Supported Parameter Types

```python
# Numeric parameters
search_space = {
    'learning_rate': ('float', 0.001, 0.1, 'log'),  # Log scale
    'n_estimators': ('int', 10, 1000),
    'regularization': ('float', 1e-6, 1e-2, 'log'),
}

# Categorical parameters
search_space = {
    'algorithm': ('categorical', ['auto', 'ball_tree', 'kd_tree']),
    'metric': ('categorical', ['euclidean', 'manhattan', 'minkowski']),
}

# Conditional parameters
search_space = {
    'algorithm': ('categorical', ['svm', 'random_forest']),
    'C': ('float', 0.1, 10.0, 'when', 'algorithm', 'svm'),
    'n_estimators': ('int', 10, 200, 'when', 'algorithm', 'random_forest'),
}
```

### Dynamic Search Space Adaptation

```python
# Enable adaptive search space
optimizer = AdaptiveHyperparameterOptimizer(
    backend="optuna",
    n_trials=200,
    adaptation_interval=25,
    min_improvement_threshold=0.001
)

# The optimizer will automatically:
# 1. Narrow search ranges around promising regions
# 2. Expand ranges if no improvement is found
# 3. Add/remove parameters based on importance
# 4. Adjust sampling strategies
```

## Advanced Features

### Custom Samplers and Pruners

```python
# Custom sampler configuration
sampler_config = {
    'sampler_type': 'tpe',
    'multivariate': True,
    'n_startup_trials': 20,
    'n_ei_candidates': 24,
    'gamma': 0.25,
    'weights': lambda x: min(x, 25)
}

# Custom pruner configuration
pruner_config = {
    'pruner_type': 'hyperband',
    'min_resource': 1,
    'max_resource': 'auto',
    'reduction_factor': 3
}

optimizer = AdaptiveHyperparameterOptimizer(
    backend="optuna",
    sampler_config=sampler_config,
    pruner_config=pruner_config
)
```

### Warm Start from Previous Optimization

```python
# Save optimization state
optimizer.save_study("optimization_study.pkl")

# Load and continue optimization
new_optimizer = AdaptiveHyperparameterOptimizer.load_study(
    "optimization_study.pkl"
)

# Continue optimization
best_params, best_score, _ = new_optimizer.continue_optimization(
    additional_trials=50
)
```

### Custom Adaptation Strategies

```python
# Define custom adaptation strategy
def custom_adaptation_strategy(optimizer, trial_history):
    """Custom strategy for search space adaptation"""
    recent_trials = trial_history[-20:]
    
    if len(recent_trials) < 10:
        return optimizer.search_space  # No adaptation yet
    
    # Calculate improvement trend
    scores = [trial.value for trial in recent_trials]
    improvement = (scores[-1] - scores[0]) / len(scores)
    
    if improvement < 0.001:
        # Narrow search space around best parameters
        return optimizer._narrow_search_space()
    else:
        # Expand search space for exploration
        return optimizer._expand_search_space()

# Use custom adaptation
optimizer.set_adaptation_strategy(custom_adaptation_strategy)
```

## Performance Optimization

### Parallel Optimization

```python
# Multi-process optimization
optimizer = AdaptiveHyperparameterOptimizer(
    backend="optuna",
    n_trials=500,
    n_jobs=8,  # Use 8 parallel processes
    storage="sqlite:///optimization.db"  # Shared storage for parallel trials
)

# Distributed optimization (Redis backend)
optimizer = AdaptiveHyperparameterOptimizer(
    backend="optuna",
    n_trials=1000,
    n_jobs=-1,  # Use all available cores
    storage="redis://localhost:6379/0"
)
```

### Memory-Efficient Optimization

```python
# Enable memory optimization
optimizer = AdaptiveHyperparameterOptimizer(
    backend="optuna",
    n_trials=1000,
    memory_efficient=True,
    max_memory_usage="4GB",
    enable_gc=True,  # Enable garbage collection
    trial_timeout=300  # 5-minute timeout per trial
)
```

## Integration with ML Pipelines

### Scikit-learn Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Define search space for pipeline
search_space = {
    'scaler__with_mean': ('categorical', [True, False]),
    'scaler__with_std': ('categorical', [True, False]),
    'classifier__n_estimators': ('int', 10, 200),
    'classifier__max_depth': ('int', 3, 20),
}

def pipeline_objective(params):
    pipeline.set_params(**params)
    scores = cross_val_score(pipeline, X, y, cv=5)
    return scores.mean()

# Optimize pipeline
best_params, best_score, _ = optimizer.optimize(
    objective=pipeline_objective,
    search_space=search_space,
    direction="maximize"
)
```

### Deep Learning Integration

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def neural_network_objective(params):
    # Define model architecture based on parameters
    model = nn.Sequential(
        nn.Linear(input_size, params['hidden_size_1']),
        nn.ReLU(),
        nn.Dropout(params['dropout_1']),
        nn.Linear(params['hidden_size_1'], params['hidden_size_2']),
        nn.ReLU(),
        nn.Dropout(params['dropout_2']),
        nn.Linear(params['hidden_size_2'], output_size)
    )
    
    # Train and evaluate model
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    # ... training loop ...
    
    return validation_accuracy

# Neural network search space
nn_search_space = {
    'hidden_size_1': ('int', 32, 512),
    'hidden_size_2': ('int', 16, 256),
    'dropout_1': ('float', 0.1, 0.5),
    'dropout_2': ('float', 0.1, 0.5),
    'learning_rate': ('float', 1e-5, 1e-2, 'log'),
    'batch_size': ('categorical', [16, 32, 64, 128]),
}
```

## Monitoring and Visualization

### Real-time Progress Tracking

```python
# Enable detailed monitoring
optimizer = AdaptiveHyperparameterOptimizer(
    backend="optuna",
    n_trials=200,
    enable_monitoring=True,
    log_level="INFO"
)

# Custom progress callback
def detailed_progress_callback(study, trial):
    print(f"Trial {trial.number}:")
    print(f"  Parameters: {trial.params}")
    print(f"  Value: {trial.value}")
    print(f"  Best so far: {study.best_value}")
    
    # Log to file or external monitoring system
    logging.info(f"Trial {trial.number}: {trial.value}")

optimizer.add_callback(detailed_progress_callback)
```

### Optimization Visualization

```python
# Generate optimization plots
import matplotlib.pyplot as plt

# Plot optimization history
optimizer.plot_optimization_history()

# Plot parameter importance
optimizer.plot_param_importance()

# Plot parallel coordinate plot
optimizer.plot_parallel_coordinate()

# Plot hyperparameter relationships
optimizer.plot_param_relationships()

# Save plots
optimizer.save_plots("optimization_plots/")
```

## Error Handling and Robustness

### Fault-Tolerant Optimization

```python
# Configure error handling
optimizer = AdaptiveHyperparameterOptimizer(
    backend="optuna",
    n_trials=200,
    max_failures=10,  # Maximum allowed failed trials
    retry_failed_trials=True,
    timeout_per_trial=300,  # 5-minute timeout per trial
    catch_exceptions=True
)

def robust_objective(params):
    try:
        # Your optimization logic here
        result = train_and_evaluate_model(params)
        return result
    except Exception as e:
        # Log error and return worst possible score
        logging.error(f"Trial failed with error: {e}")
        return float('-inf')  # For maximization problems

# The optimizer will automatically handle failed trials
best_params, best_score, _ = optimizer.optimize(
    objective=robust_objective,
    search_space=search_space,
    direction="maximize"
)
```

## Best Practices

### 1. Search Space Design
```python
# Good: Well-defined ranges based on domain knowledge
search_space = {
    'learning_rate': ('float', 1e-5, 1e-1, 'log'),  # Log scale for learning rate
    'batch_size': ('categorical', [16, 32, 64, 128]),  # Powers of 2
    'n_layers': ('int', 2, 8),  # Reasonable range for layers
}

# Avoid: Too wide or inappropriate ranges
search_space = {
    'learning_rate': ('float', 0, 1),  # Too wide, linear scale
    'batch_size': ('int', 1, 1000),  # Too wide range
}
```

### 2. Objective Function Design
```python
# Good: Robust objective with proper validation
def robust_objective(params):
    try:
        # Use cross-validation for robust evaluation
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        return scores.mean()
    except Exception as e:
        logging.warning(f"Trial failed: {e}")
        return float('-inf')

# Good: Multi-metric objective
def multi_metric_objective(params):
    model = create_model(params)
    accuracy = evaluate_accuracy(model)
    inference_time = measure_inference_time(model)
    
    # Balance accuracy and speed
    return accuracy - 0.1 * inference_time
```

### 3. Resource Management
```python
# Configure resource limits
optimizer = AdaptiveHyperparameterOptimizer(
    n_trials=100,
    timeout=3600,  # 1 hour total
    n_jobs=min(4, os.cpu_count()),  # Don't oversubscribe
    memory_limit="8GB"
)
```

## Related Documentation

- [Training Engine Documentation](train_engine.md)
- [Configuration System Documentation](../configs.md)
- [ASHT Optimizer Documentation](../optimizer/asht.md)
- [HyperOptX Optimizer Documentation](../optimizer/hyperoptx.md)
