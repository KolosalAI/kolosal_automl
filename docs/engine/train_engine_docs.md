# Detailed Function Documentation for ML Training Engine

## ExperimentTracker Class

### `__init__(output_dir="./experiments", experiment_name=None)`
- **Purpose**: Initializes the experiment tracker.
- **Parameters**:
  - `output_dir`: Directory to store experiment results
  - `experiment_name`: Name for this experiment series (defaults to timestamp-based name)
- **Returns**: ExperimentTracker instance

### `start_experiment(config, model_info)`
- **Purpose**: Begin a new experiment run and attach configuration and model metadata.
- **Parameters**:
  - `config`: Dictionary of configuration parameters
  - `model_info`: Dictionary of model metadata
- **Returns**: None

### `_make_json_serializable(obj)`
- **Purpose**: Recursively convert objects into JSON-serializable structures.
- **Parameters**:
  - `obj`: Object to convert
- **Returns**: JSON-serializable version of the input object

### `log_metrics(metrics, step=None)`
- **Purpose**: Log metrics for the current experiment.
- **Parameters**:
  - `metrics`: Dictionary of metric names and values
  - `step`: Optional step identifier to organize metrics by stages
- **Returns**: None

### `log_feature_importance(feature_names, importance)`
- **Purpose**: Log and visualize feature importance scores.
- **Parameters**:
  - `feature_names`: List of feature names
  - `importance`: Array of importance scores
- **Returns**: None

### `log_model(model, model_name, path=None)`
- **Purpose**: Save a trained model and add it to experiment artifacts.
- **Parameters**:
  - `model`: Trained model object
  - `model_name`: Name for the model
  - `path`: Optional custom save path
- **Returns**: None

### `log_confusion_matrix(y_true, y_pred, class_names=None)`
- **Purpose**: Generate and save confusion matrix for classification tasks.
- **Parameters**:
  - `y_true`: True labels
  - `y_pred`: Predicted labels
  - `class_names`: Optional list of class names for better visualization
- **Returns**: None

### `end_experiment()`
- **Purpose**: Finalize the experiment, write metadata to disk, and close MLflow run.
- **Returns**: Dictionary containing the serialized experiment data

### `generate_report(report_path=None, include_plots=True)`
- **Purpose**: Create a comprehensive markdown report of the experiment.
- **Parameters**:
  - `report_path`: Path to save the report
  - `include_plots`: Whether to include plots in the report
- **Returns**: String path to the generated report

## MLTrainingEngine Class

### `__init__(config)`
- **Purpose**: Initialize the training engine with the given configuration.
- **Parameters**:
  - `config`: Configuration object for the training engine
- **Returns**: MLTrainingEngine instance

### `_init_components()`
- **Purpose**: Initialize all engine components based on configuration.
- **Returns**: None

### `_register_shutdown_handlers()`
- **Purpose**: Register handlers for proper cleanup during shutdown.
- **Returns**: None

### `_signal_handler(signum, frame)`
- **Purpose**: Handle signals for graceful shutdown.
- **Parameters**:
  - `signum`: Signal number
  - `frame`: Current stack frame
- **Returns**: None

### `_cleanup_on_shutdown()`
- **Purpose**: Perform cleanup operations before shutdown.
- **Returns**: None

### `_register_model_types()`
- **Purpose**: Register built-in model types for automatic discovery.
- **Returns**: None

### `_get_feature_selector(X, y)`
- **Purpose**: Create appropriate feature selector based on configuration.
- **Parameters**:
  - `X`: Feature matrix
  - `y`: Target variable
- **Returns**: Configured feature selector or None

### `_create_pipeline(model)`
- **Purpose**: Create a scikit-learn pipeline with preprocessing and model.
- **Parameters**:
  - `model`: The model estimator
- **Returns**: sklearn.pipeline.Pipeline object

### `_get_cv_splitter(y=None)`
- **Purpose**: Get appropriate cross-validation splitter based on task type.
- **Parameters**:
  - `y`: Target variable for stratification
- **Returns**: KFold or StratifiedKFold object

### `_get_optimization_search(model, param_grid)`
- **Purpose**: Configure hyperparameter optimization based on strategy.
- **Parameters**:
  - `model`: The model estimator
  - `param_grid`: Parameter grid/space for optimization
- **Returns**: Configured optimization object (GridSearchCV, RandomizedSearchCV, etc.)

### `_get_scoring_metric()`
- **Purpose**: Determine the appropriate scoring metric based on task type.
- **Returns**: String identifier for sklearn scoring metric

### `_extract_feature_names(X)`
- **Purpose**: Extract feature names from input data.
- **Parameters**:
  - `X`: Input data (DataFrame, array, etc.)
- **Returns**: List of feature names

### `_get_feature_importance(model)`
- **Purpose**: Extract feature importance from a trained model.
- **Parameters**:
  - `model`: Trained model
- **Returns**: Array of feature importance values or None

### `_get_default_param_grid(model)`
- **Purpose**: Generate a reasonable default hyperparameter grid.
- **Parameters**:
  - `model`: Model instance
- **Returns**: Dictionary of parameter grids

### `_compare_metrics(new_metric, current_best)`
- **Purpose**: Compare metrics to determine if new model is better.
- **Parameters**:
  - `new_metric`: Metric value of new model
  - `current_best`: Current best metric value
- **Returns**: Boolean indicating if new model is better

### `_get_best_metric_value(metrics)`
- **Purpose**: Extract the best metric value based on task type.
- **Parameters**:
  - `metrics`: Dictionary of metrics
- **Returns**: Float value of the most relevant metric

### `train_model(X, y, model_type=None, custom_model=None, param_grid=None, model_name=None, X_val=None, y_val=None)`
- **Purpose**: Train a machine learning model with hyperparameter optimization.
- **Parameters**:
  - `X`: Feature matrix
  - `y`: Target variable
  - `model_type`: Type of model to train (e.g., "random_forest")
  - `custom_model`: Custom pre-initialized model
  - `param_grid`: Hyperparameter grid for optimization
  - `model_name`: Custom name for the model
  - `X_val`, `y_val`: Optional validation data
- **Returns**: Dictionary with training results and metrics
- **Example output**:
  ```python
  {
      "model_name": "random_forest_1621478921",
      "model": <RandomForestClassifier object>,
      "params": {"n_estimators": 200, "max_depth": 10, ...},
      "metrics": {"accuracy": 0.92, "f1": 0.91, "precision": 0.90, "recall": 0.93},
      "feature_importance": [0.2, 0.15, 0.1, ...],
      "training_time": 15.6
  }
  ```

### `get_performance_comparison()`
- **Purpose**: Compare performance across all trained models.
- **Returns**: Dictionary with model comparisons
- **Example output**:
  ```python
  {
      "models": [
          {
              "name": "random_forest_1621478921",
              "type": "RandomForestClassifier",
              "training_time": 15.6,
              "is_best": True,
              "metrics": {"accuracy": 0.92, "f1": 0.91, ...}
          },
          {
              "name": "xgboost_1621479045",
              "type": "XGBClassifier",
              "training_time": 25.2,
              "is_best": False,
              "metrics": {"accuracy": 0.90, "f1": 0.89, ...}
          }
      ],
      "best_model": "random_forest_1621478921",
      "primary_metric": "f1"
  }
  ```

### `_determine_primary_metric(available_metrics)`
- **Purpose**: Determine the most appropriate metric for model comparison.
- **Parameters**:
  - `available_metrics`: Set of available metric names
- **Returns**: String name of the primary metric

### `generate_report(output_file=None)`
- **Purpose**: Generate a comprehensive report of all models.
- **Parameters**:
  - `output_file`: Path to save the report
- **Returns**: String path to the generated report

### `shutdown()`
- **Purpose**: Explicitly shut down the engine and release resources.
- **Returns**: None

### `get_best_model()`
- **Purpose**: Get the current best model and its metrics.
- **Returns**: Tuple of (model_name, model_info)
- **Example output**:
  ```python
  ("random_forest_1621478921", {
      "model": <RandomForestClassifier object>,
      "params": {"n_estimators": 200, "max_depth": 10, ...},
      "feature_names": ["feature1", "feature2", ...],
      "metrics": {"accuracy": 0.92, "f1": 0.91, ...},
      "feature_importance": [0.2, 0.15, 0.1, ...]
  })
  ```

### `evaluate_model(model_name=None, X_test=None, y_test=None, detailed=False)`
- **Purpose**: Evaluate a model with comprehensive metrics.
- **Parameters**:
  - `model_name`: Name of the model to evaluate
  - `X_test`, `y_test`: Test data
  - `detailed`: Whether to compute additional detailed metrics
- **Returns**: Dictionary of evaluation metrics
- **Example output**:
  ```python
  {
      "accuracy": 0.92,
      "precision": 0.90,
      "recall": 0.93,
      "f1": 0.91,
      "roc_auc": 0.95,
      "prediction_time": 0.15,
      "detailed_report": {...},  # Only if detailed=True
      "confusion_matrix": [[45, 5], [3, 47]]  # Only if detailed=True
  }
  ```

### `save_model(model_name, path=None, include_preprocessor=True)`
- **Purpose**: Persist a model to disk.
- **Parameters**:
  - `model_name`: Name of the model to save
  - `path`: Custom save path
  - `include_preprocessor`: Whether to include preprocessor and metadata
- **Returns**: String path to the saved model

### `load_model(path, model_name=None)`
- **Purpose**: Load a saved model from disk.
- **Parameters**:
  - `path`: Path to the saved model
  - `model_name`: Name to give the loaded model
- **Returns**: Tuple of (success flag, model or error message)
- **Example output**:
  ```python
  (True, <RandomForestClassifier object>)  # Success
  (False, "Model file not found: ./models/my_model.pkl")  # Error
  ```

### `predict(X, model_name=None, return_proba=False)`
- **Purpose**: Make predictions using a trained model.
- **Parameters**:
  - `X`: Features to predict
  - `model_name`: Name of the model to use
  - `return_proba`: Whether to return probabilities for classification
- **Returns**: Tuple of (success flag, predictions or error message)
- **Example output**:
  ```python
  (True, np.array([0, 1, 0, 1, ...]))  # Classification
  (True, np.array([0.2, 0.8, 0.3, ...]))  # Regression or probabilities
  (False, "Model not found")  # Error
  ```

### `generate_explainability(model_name=None, X=None, method="shap")`
- **Purpose**: Generate model explainability visualizations.
- **Parameters**:
  - `model_name`: Name of the model to explain
  - `X`: Data for explanations
  - `method`: Explainability method (shap, permutation)
- **Returns**: Dictionary with explainability results
- **Example output**:
  ```python
  {
      "method": "shap",
      "importance": {"feature1": 0.25, "feature2": 0.18, ...},
      "plot_path": "./models/explanations/shap_summary_model1.png"
  }
  ```

### `get_model_summary(model_name=None)`
- **Purpose**: Get a summary of a model's information.
- **Parameters**:
  - `model_name`: Name of the model
- **Returns**: Dictionary with model summary
- **Example output**:
  ```python
  {
      "model_name": "random_forest_1621478921",
      "model_type": "RandomForestClassifier",
      "feature_count": 15,
      "metrics": {"accuracy": 0.92, "f1": 0.91, ...},
      "training_time": 15.6,
      "is_best_model": True,
      "top_features": {"feature1": 0.25, "feature2": 0.18, ...}
  }
  ```

### `_evaluate_model(model, X, y, X_test=None, y_test=None)`
- **Purpose**: Evaluate model performance with appropriate metrics.
- **Parameters**:
  - `model`: Trained model
  - `X`, `y`: Training data
  - `X_test`, `y_test`: Test data
- **Returns**: Dictionary of evaluation metrics
- **Example output**:
  ```python
  {
      "accuracy": 0.92,
      "precision": 0.90,
      "recall": 0.93,
      "f1": 0.91,
      "prediction_time": 0.15
  }
  ```