# ASHT Optimizer (`modules/optimizer/asht.py`)

## Overview

The ASHT (Adaptive Surrogate-Assisted Hyperparameter Tuning) Optimizer provides a sophisticated hyperparameter optimization algorithm that combines surrogate modeling, adaptive resource allocation, and intelligent parameter space exploration. It's designed to efficiently find optimal hyperparameters for machine learning models with minimal computational cost.

## Features

- **Surrogate-Assisted Optimization**: Uses Random Forest surrogate models to predict parameter performance
- **Adaptive Resource Allocation**: Dynamically adjusts evaluation budgets based on configuration promise
- **Multi-Fidelity Evaluation**: Supports different evaluation budgets for faster exploration
- **Parameter Space Refinement**: Intelligently focuses search on promising regions
- **Batch Optimization**: Evaluates multiple configurations in parallel
- **Cross-Validation Integration**: Built-in support for robust model evaluation

## Core Classes

### ASHTOptimizer

Main optimization class implementing the ASHT algorithm:

```python
class ASHTOptimizer:
    def __init__(
        self,
        estimator,                    # ML model to optimize
        param_space: dict,           # Parameter search space
        max_iter: int = 50,          # Maximum optimization iterations
        cv: int = 5,                 # Cross-validation folds
        scoring: str = None,         # Scoring metric
        random_state: int = None,    # Random seed
        n_jobs: int = 1,             # Parallel jobs
        verbose: int = 0             # Verbosity level
    )
```

**Key Attributes:**
- `best_params_`: Best parameters found
- `best_score_`: Best cross-validation score
- `best_estimator_`: Trained model with best parameters
- `cv_results_`: Detailed optimization history

## Usage Examples

### Basic Hyperparameter Optimization

```python
from modules.optimizer.asht import ASHTOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Create sample dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15, 
    n_redundant=5, 
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define parameter space for Random Forest
param_space = {
    'n_estimators': (10, 200),           # Numerical range
    'max_depth': (3, 20),                # Numerical range
    'min_samples_split': (2, 20),        # Numerical range
    'min_samples_leaf': (1, 10),         # Numerical range
    'max_features': ['sqrt', 'log2', None],  # Categorical choices
    'bootstrap': [True, False],           # Boolean choices
    'criterion': ['gini', 'entropy']     # Categorical choices
}

# Initialize base estimator
rf = RandomForestClassifier(random_state=42)

# Create ASHT optimizer
optimizer = ASHTOptimizer(
    estimator=rf,
    param_space=param_space,
    max_iter=30,           # 30 optimization iterations
    cv=5,                  # 5-fold cross-validation
    scoring='accuracy',    # Optimize for accuracy
    random_state=42,
    verbose=1              # Show progress
)

print("Starting ASHT hyperparameter optimization...")

# Run optimization
optimizer.fit(X_train, y_train)

print("Optimization completed!")
print(f"Best parameters: {optimizer.best_params_}")
print(f"Best cross-validation score: {optimizer.best_score_:.4f}")

# Train final model with best parameters
best_model = RandomForestClassifier(**optimizer.best_params_, random_state=42)
best_model.fit(X_train, y_train)

# Evaluate on test set
test_score = best_model.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.4f}")

# Access optimization history
print(f"\nOptimization History:")
print(f"  Total evaluations: {len(optimizer.cv_results_['params'])}")
print(f"  Score progression: {optimizer.cv_results_['mean_test_score'][:5]}...")  # First 5 scores
```

### Advanced Multi-Model Optimization

```python
from modules.optimizer.asht import ASHTOptimizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import time

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

class MultiModelOptimizer:
    """Optimize multiple models and compare results"""
    
    def __init__(self, models_config: dict):
        self.models_config = models_config
        self.results = {}
    
    def optimize_all_models(self, X_train, y_train, max_iter_per_model=25):
        """Optimize all configured models"""
        
        for model_name, (estimator, param_space) in self.models_config.items():
            print(f"\n{'='*50}")
            print(f"Optimizing {model_name}")
            print(f"{'='*50}")
            
            start_time = time.time()
            
            # Create optimizer for this model
            optimizer = ASHTOptimizer(
                estimator=estimator,
                param_space=param_space,
                max_iter=max_iter_per_model,
                cv=5,
                scoring='roc_auc',  # Use AUC for binary classification
                random_state=42,
                verbose=1
            )
            
            # Run optimization
            optimizer.fit(X_train, y_train)
            
            optimization_time = time.time() - start_time
            
            # Store results
            self.results[model_name] = {
                'optimizer': optimizer,
                'best_params': optimizer.best_params_,
                'best_score': optimizer.best_score_,
                'optimization_time': optimization_time,
                'total_evaluations': len(optimizer.cv_results_['params'])
            }
            
            print(f"Best {model_name} score: {optimizer.best_score_:.4f}")
            print(f"Optimization time: {optimization_time:.2f} seconds")
    
    def get_best_model(self):
        """Get the overall best performing model"""
        if not self.results:
            return None
        
        best_model_name = max(self.results.keys(), 
                            key=lambda k: self.results[k]['best_score'])
        return best_model_name, self.results[best_model_name]
    
    def compare_models(self):
        """Compare all optimized models"""
        print(f"\n{'='*70}")
        print("MODEL COMPARISON RESULTS")
        print(f"{'='*70}")
        
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['best_score'], 
            reverse=True
        )
        
        for rank, (model_name, result) in enumerate(sorted_results, 1):
            print(f"{rank}. {model_name}:")
            print(f"   Best CV Score: {result['best_score']:.4f}")
            print(f"   Optimization Time: {result['optimization_time']:.2f}s")
            print(f"   Evaluations: {result['total_evaluations']}")
            print(f"   Best Params: {result['best_params']}")
            print()

# Define models and their parameter spaces
models_config = {
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': (50, 300),
            'max_depth': (5, 25),
            'min_samples_split': (2, 15),
            'min_samples_leaf': (1, 8),
            'max_features': ['sqrt', 'log2', 0.5, 0.8]
        }
    ),
    
    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=42),
        {
            'n_estimators': (50, 200),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'min_samples_split': (2, 15),
            'min_samples_leaf': (1, 8),
            'subsample': (0.6, 1.0)
        }
    ),
    
    'SVM': (
        SVC(random_state=42, probability=True),
        {
            'C': (0.1, 100.0),
            'gamma': (0.001, 10.0),
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
    ),
    
    'Logistic Regression': (
        LogisticRegression(random_state=42, max_iter=1000),
        {
            'C': (0.01, 100.0),
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': (0.0, 1.0)  # Only used with elasticnet
        }
    )
}

# Run multi-model optimization
multi_optimizer = MultiModelOptimizer(models_config)
multi_optimizer.optimize_all_models(X_train, y_train, max_iter_per_model=20)

# Compare results
multi_optimizer.compare_models()

# Get best overall model
best_name, best_result = multi_optimizer.get_best_model()
print(f"Best overall model: {best_name}")
print(f"Best score: {best_result['best_score']:.4f}")

# Train and evaluate the best model
best_optimizer = best_result['optimizer']
best_estimator = best_optimizer.estimator.set_params(**best_result['best_params'])
best_estimator.fit(X_train, y_train)

# Test set evaluation
from sklearn.metrics import classification_report, roc_auc_score

y_pred = best_estimator.predict(X_test)
y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nTest Set Performance:")
print(f"AUC Score: {test_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

### Custom Optimization with Early Stopping

```python
from modules.optimizer.asht import ASHTOptimizer
import numpy as np
from sklearn.model_selection import validation_curve

class EarlyStoppingASHT(ASHTOptimizer):
    """ASHT with early stopping based on convergence criteria"""
    
    def __init__(self, *args, patience=5, min_improvement=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_scores_history = []
        self.no_improvement_count = 0
    
    def _check_early_stopping(self):
        """Check if optimization should stop early"""
        if len(self.best_scores_history) < self.patience:
            return False
        
        # Check if there's been significant improvement in the last 'patience' iterations
        recent_scores = self.best_scores_history[-self.patience:]
        improvement = max(recent_scores) - min(recent_scores)
        
        if improvement < self.min_improvement:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                return True
        else:
            self.no_improvement_count = 0
        
        return False
    
    def fit(self, X, y):
        """Modified fit with early stopping"""
        self.X = X
        self.y = y
        
        # Track best scores for early stopping
        iteration = 0
        
        # Initial exploration phase
        R = 1.0
        B = R * 0.1
        N = min(self.max_iter // 4, 10)
        
        if self.verbose:
            print(f"Starting early-stopping ASHT optimization (patience={self.patience})")
        
        # Initial random sampling
        initial_configs = self._sample_random_configs(N)
        
        for config in initial_configs:
            if iteration >= self.max_iter:
                break
                
            score = self._objective_func(config, budget=B)
            self.best_scores_history.append(self.best_score_)
            iteration += 1
            
            if self.verbose:
                print(f"Iteration {iteration}: Score = {score:.4f}, Best = {self.best_score_:.4f}")
            
            # Check early stopping
            if self._check_early_stopping():
                if self.verbose:
                    print(f"Early stopping at iteration {iteration} (no improvement for {self.patience} iterations)")
                break
        
        # Focused search phase
        while iteration < self.max_iter:
            # Sample new configuration using existing logic
            config = self._sample_random_configs(1)[0]
            score = self._objective_func(config, budget=R)  # Use full budget
            
            self.best_scores_history.append(self.best_score_)
            iteration += 1
            
            if self.verbose:
                print(f"Iteration {iteration}: Score = {score:.4f}, Best = {self.best_score_:.4f}")
            
            if self._check_early_stopping():
                if self.verbose:
                    print(f"Early stopping at iteration {iteration}")
                break
        
        if self.verbose:
            print(f"Optimization completed after {iteration} iterations")
            print(f"Best score: {self.best_score_:.4f}")
            print(f"Best parameters: {self.best_params_}")

# Example usage with early stopping
def early_stopping_example():
    """Demonstrate early stopping optimization"""
    
    # Create a challenging optimization problem
    X, y = make_classification(
        n_samples=2000, 
        n_features=30, 
        n_informative=20, 
        n_redundant=10,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Large parameter space
    param_space = {
        'n_estimators': (10, 500),
        'max_depth': (1, 30),
        'min_samples_split': (2, 50),
        'min_samples_leaf': (1, 20),
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    # Compare regular ASHT vs early stopping ASHT
    print("Comparing regular ASHT vs early stopping ASHT\n")
    
    # Regular ASHT
    print("Running regular ASHT...")
    start_time = time.time()
    regular_optimizer = ASHTOptimizer(
        estimator=RandomForestClassifier(random_state=42),
        param_space=param_space,
        max_iter=50,
        cv=3,
        scoring='accuracy',
        random_state=42,
        verbose=1
    )
    regular_optimizer.fit(X_train, y_train)
    regular_time = time.time() - start_time
    
    print(f"Regular ASHT completed in {regular_time:.2f} seconds")
    print(f"Best score: {regular_optimizer.best_score_:.4f}")
    print(f"Total evaluations: {len(regular_optimizer.cv_results_['params'])}\n")
    
    # Early stopping ASHT
    print("Running early stopping ASHT...")
    start_time = time.time()
    early_optimizer = EarlyStoppingASHT(
        estimator=RandomForestClassifier(random_state=42),
        param_space=param_space,
        max_iter=50,
        cv=3,
        scoring='accuracy',
        patience=5,
        min_improvement=0.001,
        random_state=42,
        verbose=1
    )
    early_optimizer.fit(X_train, y_train)
    early_time = time.time() - start_time
    
    print(f"Early stopping ASHT completed in {early_time:.2f} seconds")
    print(f"Best score: {early_optimizer.best_score_:.4f}")
    print(f"Total evaluations: {len(early_optimizer.cv_results_['params'])}")
    
    # Compare efficiency
    efficiency_gain = (regular_time - early_time) / regular_time * 100
    score_difference = abs(regular_optimizer.best_score_ - early_optimizer.best_score_)
    
    print(f"\nComparison:")
    print(f"Time saved with early stopping: {efficiency_gain:.1f}%")
    print(f"Score difference: {score_difference:.4f}")
    print(f"Early stopping efficiency: {'✓ More efficient' if efficiency_gain > 0 and score_difference < 0.01 else '✗ Less efficient'}")

# Run early stopping example
early_stopping_example()
```

## Advanced Features

### Custom Parameter Space Definition

```python
from scipy.stats import uniform, randint, choice

# Advanced parameter space with distributions
advanced_param_space = {
    # Continuous distributions
    'learning_rate': uniform(0.01, 0.29),  # Uniform between 0.01 and 0.3
    'regularization': uniform(0.0001, 0.9999),  # Uniform between 0.0001 and 1.0
    
    # Discrete distributions
    'n_estimators': randint(10, 201),  # Random integers between 10 and 200
    'max_depth': randint(3, 21),       # Random integers between 3 and 20
    
    # Categorical with probabilities
    'activation': ['relu', 'tanh', 'sigmoid'],
    'optimizer_type': ['adam', 'sgd', 'rmsprop'],
    
    # Log-scale parameters
    'alpha': (1e-5, 1e-1),  # Will be sampled in log space
}
```

### Performance Analysis and Visualization

```python
import matplotlib.pyplot as plt

def analyze_optimization_performance(optimizer):
    """Analyze and visualize optimization performance"""
    
    results = optimizer.cv_results_
    
    # Plot optimization progress
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Score progression
    plt.subplot(1, 3, 1)
    scores = results['mean_test_score']
    best_scores = np.maximum.accumulate(scores)
    
    plt.plot(scores, 'b-', alpha=0.6, label='Individual scores')
    plt.plot(best_scores, 'r-', linewidth=2, label='Best score so far')
    plt.xlabel('Iteration')
    plt.ylabel('CV Score')
    plt.title('Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Score distribution
    plt.subplot(1, 3, 2)
    plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(optimizer.best_score_, color='red', linestyle='--', 
                label=f'Best: {optimizer.best_score_:.4f}')
    plt.xlabel('CV Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Parameter importance (for numerical parameters)
    plt.subplot(1, 3, 3)
    numerical_params = {}
    for i, params in enumerate(results['params']):
        for key, value in params.items():
            if isinstance(value, (int, float)):
                if key not in numerical_params:
                    numerical_params[key] = []
                numerical_params[key].append((value, scores[i]))
    
    if numerical_params:
        param_correlations = {}
        for param, values in numerical_params.items():
            param_values = [v[0] for v in values]
            param_scores = [v[1] for v in values]
            correlation = np.corrcoef(param_values, param_scores)[0, 1]
            if not np.isnan(correlation):
                param_correlations[param] = abs(correlation)
        
        if param_correlations:
            params_sorted = sorted(param_correlations.items(), 
                                 key=lambda x: x[1], reverse=True)
            param_names = [p[0] for p in params_sorted[:8]]  # Top 8
            correlations = [p[1] for p in params_sorted[:8]]
            
            plt.barh(param_names, correlations)
            plt.xlabel('Absolute Correlation with Score')
            plt.title('Parameter Importance')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Optimization Summary:")
    print(f"  Total evaluations: {len(scores)}")
    print(f"  Best score: {optimizer.best_score_:.4f}")
    print(f"  Score std: {np.std(scores):.4f}")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Improvement: {(optimizer.best_score_ - min(scores)):.4f}")

# Example usage
# analyze_optimization_performance(optimizer)
```

## Best Practices

### 1. Parameter Space Design

```python
# Good parameter space design
param_space = {
    # Use appropriate ranges
    'n_estimators': (50, 300),  # Not too small, not too large
    
    # Use log scale for parameters that vary by orders of magnitude
    'learning_rate': (0.001, 0.1),
    
    # Include reasonable categorical options
    'criterion': ['gini', 'entropy'],  # Don't include too many options
    
    # Avoid conflicting parameters
    'solver': ['liblinear'],  # Fix solver if using specific penalty
    'penalty': ['l1', 'l2']   # Compatible with liblinear
}
```

### 2. Resource Allocation

```python
# Balance exploration vs exploitation
optimizer = ASHTOptimizer(
    max_iter=30,     # Reasonable number of iterations
    cv=5,            # Good balance of reliability vs speed
    verbose=1        # Monitor progress
)
```

### 3. Scoring Metrics

```python
# Choose appropriate scoring for your problem
binary_classification_scoring = 'roc_auc'
multiclass_classification_scoring = 'f1_macro'
regression_scoring = 'neg_mean_squared_error'
```

### 4. Random State Management

```python
# Always set random state for reproducibility
optimizer = ASHTOptimizer(
    estimator=model,
    param_space=params,
    random_state=42  # Ensures reproducible results
)
```

## Related Documentation

- [HyperoptX Optimizer Documentation](hyperoptx.md)
- [Adaptive Hyperopt Documentation](../engine/adaptive_hyperopt.md)
- [Training Engine Documentation](../engine/train_engine.md)
- [Configuration System Documentation](../configs.md)
