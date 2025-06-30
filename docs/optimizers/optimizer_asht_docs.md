# ASHTOptimizer Documentation

## Overview

The **ASHTOptimizer** (Adaptive Surrogate-Assisted Hyperparameter Tuning) class is designed to efficiently search for optimal hyperparameter configurations for a given estimator. It leverages a surrogate model (by default, a RandomForestRegressor) to approximate the performance of different hyperparameter settings and uses an Expected Improvement acquisition function to guide the search. The optimization process is performed in two phases:

1. **Phase 1**: Initial exploration with a subset of the resource (budget).  
2. **Phase 2**: Iterative focused search guided by the surrogate model, refining the parameter space progressively.

## Class: `ASHTOptimizer`

### Initialization

```python
def __init__(
    self,
    estimator,
    param_space,
    max_iter=50,
    cv=5,
    scoring=None,
    random_state=None,
    n_jobs=1,
    verbose=0
):
    ...
```

- **estimator**: A scikit-learn estimator (e.g., `RandomForestClassifier`, `LogisticRegression`, etc.).  
- **param_space**: A dictionary defining the hyperparameter search space. Each key is a hyperparameter name, and the value can be:
  - A numeric range `(low, high)` (tuple),
  - A list of categorical values,
  - A distribution object (e.g., from `scipy.stats`) implementing `.rvs()`.
- **max_iter** (default=50): The total number of iterations (evaluations) for the optimization.
- **cv** (default=5): Number of cross-validation folds to use for performance estimation.
- **scoring** (default=None): A string or callable that defines the metric to optimize (e.g., `'accuracy'` for classifiers). If `None`, estimator’s default scorer is used.
- **random_state** (default=None): Controls the random seed for reproducibility.
- **n_jobs** (default=1): Number of parallel jobs for cross-validation.
- **verbose** (default=0): Controls the amount of logging. Higher values produce more output.

### Attributes

- **best_params_**: The best hyperparameter set found so far.
- **best_score_**: The best cross-validation score observed.
- **best_estimator_**: The estimator refit with the best found hyperparameters.
- **cv_results_**: A dictionary tracking intermediate results across all evaluated configurations. Keys include:
  - `'params'`: List of evaluated parameter dictionaries.
  - `'mean_test_score'`: Mean CV score per configuration.
  - `'std_test_score'`: Standard deviation of CV scores.
  - `'split0_test_score'`: Score from the first CV split for quick reference.
  - `'budget'`: The fraction of the data resource used.

### Public Methods

#### `fit(X, y) -> ASHTOptimizer`
Runs the ASHT optimization process on the provided data.

```python
def fit(self, X, y):
    ...
```

- **X**: Feature matrix (array-like or DataFrame) for training.
- **y**: Target vector (array-like).
- **Returns**: `self`, the fitted `ASHTOptimizer` instance.

**Key steps** in `fit`:
1. **Phase 1 (Initial Exploration)**  
   - Samples a number of random configurations (`N`) from the parameter space.  
   - Evaluates them with a reduced budget (e.g., 10% of the resource).  
   - Identifies top-performing configurations and trains the surrogate model.
2. **Refining Parameter Space**  
   - Uses feature importances (for tree-based surrogates) to narrow down less important parameters or categories.
3. **Phase 2 (Focused Search)**  
   - Iteratively proposes new candidate configurations based on the acquisition function (Expected Improvement).
   - Each iteration refines the surrogate model, increases the resource budget, and continues until `max_iter` is reached.
4. **Best Estimator**  
   - After all iterations, refits a fresh clone of the estimator with the best found hyperparameters on the entire dataset.

### Typical Usage

```python
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from asht_optimizer import ASHTOptimizer

# Load data
X, y = load_boston(return_X_y=True)

# Define parameter space
param_space = {
    'n_estimators': (10, 200),     # Numeric range
    'max_depth': (2, 20),         # Numeric range
    'criterion': ['mse', 'mae'],  # Categorical
}

# Create a RandomForest estimator
estimator = RandomForestRegressor(random_state=42)

# Initialize and run ASHT
optimizer = ASHTOptimizer(
    estimator=estimator,
    param_space=param_space,
    max_iter=30,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=1
)

optimizer.fit(X, y)

# Inspect best results
print("Best Score:", optimizer.best_score_)
print("Best Params:", optimizer.best_params_)

# Use the best estimator
best_model = optimizer.best_estimator_
preds = best_model.predict(X)
```

### Internal/Utility Methods

Below are brief descriptions of the core internal methods used by `ASHTOptimizer`. Although these methods are marked as internal (via leading `_`), they provide insights into the internal logic.

- **`_analyze_param_space()`**  
  Determines how parameters should be handled (numeric, categorical, or distribution).

- **`_get_param_bounds()`**  
  Extracts bounds for numeric parameters for optimization steps.

- **`_validate_params(params)`**  
  Fixes incompatible or invalid parameter values (e.g., invalid solver-penalty combinations in certain estimators).

- **`_objective_func(params, budget)`**  
  Evaluates a given hyperparameter configuration with a specified fraction of the total resource (`budget`). Performs cross-validation and returns the mean score.

- **`_sample_random_configs(n)`**  
  Samples `n` random hyperparameter configurations from the parameter space.

- **`_encode_config(config)` / `_configs_to_features(configs)`**  
  Converts a hyperparameter dictionary (or a list of them) into a numeric feature vector suitable for the surrogate model. Categorical variables are one-hot encoded.

- **`_decode_vector_to_config(vector, param_names)`**  
  Inverse operation of `_encode_config`; reconstructs a hyperparameter configuration from a numeric vector.

- **`_train_surrogate_model(configs, scores)`**  
  Trains or updates the surrogate model (e.g., `RandomForestRegressor`) based on already evaluated configurations and their performance scores.

- **`_expected_improvement(x, surrogate, best_f, param_names, xi=0.01)`**  
  Computes the Expected Improvement (EI) acquisition function at a given point `x` in the feature space.

- **`_optimize_acquisition(surrogate, best_f, param_names, n_restarts=5)`**  
  Performs local optimization (using L-BFGS-B) over the acquisition function to find a promising configuration to evaluate next.

- **`_refine_param_space(param_space, surrogate)`**  
  Optionally shrinks or refines the parameter search space based on feature importances from the surrogate model.

- **`_propose_using_surrogate(surrogate, param_space)`**  
  Generates a single promising candidate configuration using the surrogate model and the acquisition function.

- **`_propose_batch_using_surrogate(surrogate, param_space, batch_size=5)`**  
  Similar to `_propose_using_surrogate`, but proposes multiple diverse candidate configurations in one step.

---

## Notes

- **Resource Budgeting**: The `budget` is interpreted in this implementation as a fraction of cross-validation folds. For example, if `cv=5` and `budget=0.6`, then `int(cv * budget) = 3` folds will be used for evaluation.  
- **Refined Search**: When using a tree-based surrogate, feature importances may shrink numeric ranges or limit categorical choices. This helps focus the search on more promising hyperparameters.  
- **Performance**: The surrogate model is retrained periodically with all evaluated configurations. This adds overhead for high-dimensional parameter spaces but often reduces the total number of real evaluations needed.  
- **Caching**: The class includes an internal `evaluated_configs` dictionary to avoid re-evaluating identical hyperparameter sets at the same budget.  

## Example Walkthrough

1. **Specify Parameter Space**  
   Define ranges, lists, or distributions for hyperparameters.

2. **Initialize `ASHTOptimizer`**  
   Provide an estimator (or pipeline), the parameter space, and relevant settings (e.g., `max_iter`, `cv`).

3. **Call `.fit(X, y)`**  
   - Phase 1: Samples random hyperparameters and evaluates them with a partial resource (`budget=0.1`).  
   - Surrogate is trained on these evaluations.  
   - Parameter space is refined (optional).  
   - Phase 2: Iteratively proposes new hyperparameters based on Expected Improvement, increasing the resource budget until `max_iter` is exhausted.

4. **Retrieve Best**  
   - Inspect `best_score_`, `best_params_`, and `best_estimator_`.  
   - The best estimator is already trained on the full dataset using the best found parameters.

---

### Summary

`ASHTOptimizer` provides a robust solution for hyperparameter tuning, especially for medium-to-large search spaces. By adaptively refining the search space and leveraging surrogate modeling, it aims to find high-performing configurations with fewer full-fidelity evaluations than a naive random or grid search.