# Train Engine API (`modules/api/train_engine_api.py`)

## Overview

The Train Engine API provides a comprehensive RESTful interface for machine learning model training operations. It supports multiple ML frameworks, automated hyperparameter optimization, model evaluation, and experiment tracking through HTTP endpoints.

## Features

- **Multi-Framework Support**: Scikit-learn, XGBoost, LightGBM, Neural Networks
- **Automated Hyperparameter Optimization**: Grid search, random search, Bayesian optimization
- **Model Selection**: Automatic algorithm selection and comparison
- **Experiment Tracking**: MLflow integration for experiment management
- **Data Processing**: Automated data preprocessing and feature engineering
- **Model Evaluation**: Comprehensive evaluation metrics and validation
- **Async Training**: Background training with progress monitoring
- **Model Export**: Multiple export formats and deployment-ready models

## API Configuration

```python
# Environment Variables
TRAIN_API_HOST=0.0.0.0
TRAIN_API_PORT=8001
TRAIN_API_DEBUG=False
REQUIRE_API_KEY=False
API_KEYS=key1,key2,key3
MAX_WORKERS=4
STATIC_DIR=./static
MODELS_DIR=./static/models
REPORTS_DIR=./static/reports
MLFLOW_TRACKING_URI=./mlruns
```

## Data Models

### TrainingRequest
```python
{
    "task_type": "classification",                    # classification, regression, clustering, time_series
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],     # Training data
    "target": [0, 1],                                # Target values
    "test_size": 0.2,                                # Train/test split ratio
    "validation_split": 0.1,                         # Validation split
    "feature_names": ["feature1", "feature2", "feature3"],
    "target_name": "target",
    "optimization_strategy": "bayesian",             # grid_search, random_search, bayesian, auto
    "max_trials": 50,                                # Maximum optimization trials
    "timeout_minutes": 60,                           # Training timeout
    "cross_validation_folds": 5,                     # CV folds
    "scoring_metric": "accuracy",                     # Primary metric
    "preprocessing": {                                # Preprocessing options
        "normalize": true,
        "handle_missing": true,
        "feature_selection": true
    },
    "algorithms": ["random_forest", "xgboost", "svm"], # Algorithms to try
    "ensemble_methods": ["voting", "stacking"],        # Ensemble methods
    "model_selection_criteria": "best_score"          # Model selection criteria
}
```

### HyperparameterSpace
```python
{
    "random_forest": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [3, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "xgboost": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.8, 0.9, 1.0]
    }
}
```

## API Endpoints

### Engine Management

#### Initialize Engine
```http
POST /api/train/engine/init
Content-Type: application/json
X-API-Key: your-api-key

{
    "config": {
        "task_type": "classification",
        "optimization_strategy": "bayesian",
        "max_trials": 100,
        "cv_folds": 5,
        "enable_preprocessing": true,
        "enable_feature_selection": true,
        "mlflow_experiment_name": "production_experiment"
    }
}
```

**Response:**
```json
{
    "message": "Engine initialized successfully",
    "engine_id": "engine_12345",
    "config": {...},
    "mlflow_experiment_id": "1",
    "initialized_at": "2025-01-15T10:30:00Z"
}
```

#### Get Engine Status
```http
GET /api/train/engine/status
X-API-Key: your-api-key
```

**Response:**
```json
{
    "status": "ready",                    # ready, training, idle
    "current_task": null,
    "uptime": "2h 30m 45s",
    "total_trainings": 15,
    "successful_trainings": 14,
    "failed_trainings": 1,
    "average_training_time": "5m 23s",
    "memory_usage_mb": 512.5,
    "cpu_usage": 45.2
}
```

#### Update Engine Configuration
```http
PUT /api/train/engine/config
Content-Type: application/json
X-API-Key: your-api-key

{
    "max_trials": 200,
    "timeout_minutes": 120,
    "enable_early_stopping": true
}
```

### Training Operations

#### Start Training
```http
POST /api/train/start
Content-Type: application/json
X-API-Key: your-api-key

{
    "task_type": "classification",
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    "target": [0, 1, 0],
    "feature_names": ["feature1", "feature2", "feature3"],
    "optimization_strategy": "bayesian",
    "max_trials": 50,
    "algorithms": ["random_forest", "xgboost", "lightgbm"],
    "preprocessing": {
        "normalize": true,
        "handle_missing": true,
        "feature_selection": true,
        "outlier_detection": true
    },
    "cross_validation_folds": 5,
    "scoring_metric": "f1_score"
}
```

**Response:**
```json
{
    "message": "Training started successfully",
    "training_id": "training_12345",
    "task_type": "classification",
    "estimated_duration": "15 minutes",
    "algorithms_to_try": ["random_forest", "xgboost", "lightgbm"],
    "max_trials": 50,
    "started_at": "2025-01-15T10:30:00Z",
    "mlflow_run_id": "run_12345"
}
```

#### Upload Training Data (File)
```http
POST /api/train/upload-data
Content-Type: multipart/form-data
X-API-Key: your-api-key

file: [CSV file]
target_column: target
feature_columns: feature1,feature2,feature3
```

**Response:**
```json
{
    "message": "Data uploaded successfully",
    "data_id": "data_12345",
    "shape": [1000, 4],
    "feature_count": 3,
    "target_column": "target",
    "missing_values": 5,
    "data_types": {
        "feature1": "float64",
        "feature2": "float64", 
        "feature3": "int64",
        "target": "int64"
    }
}
```

#### Start Training with Uploaded Data
```http
POST /api/train/start-with-data
Content-Type: application/json
X-API-Key: your-api-key

{
    "data_id": "data_12345",
    "task_type": "classification",
    "test_size": 0.2,
    "optimization_strategy": "bayesian",
    "max_trials": 100
}
```

#### Get Training Status
```http
GET /api/train/status/{training_id}
X-API-Key: your-api-key
```

**Response:**
```json
{
    "training_id": "training_12345",
    "status": "training",              # queued, training, completed, failed, cancelled
    "progress": 65.5,                  # Progress percentage
    "current_trial": 33,
    "total_trials": 50,
    "current_algorithm": "xgboost",
    "best_score": 0.945,
    "best_algorithm": "random_forest",
    "elapsed_time": "8m 32s",
    "estimated_remaining": "5m 15s",
    "started_at": "2025-01-15T10:30:00Z",
    "last_updated": "2025-01-15T10:38:32Z"
}
```

#### Get Training Results
```http
GET /api/train/results/{training_id}
X-API-Key: your-api-key
```

**Response:**
```json
{
    "training_id": "training_12345",
    "status": "completed",
    "best_model": {
        "algorithm": "random_forest",
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5
        },
        "performance": {
            "accuracy": 0.945,
            "precision": 0.935,
            "recall": 0.950,
            "f1_score": 0.942,
            "roc_auc": 0.968
        },
        "cross_validation_scores": [0.94, 0.95, 0.93, 0.96, 0.94],
        "feature_importance": {
            "feature1": 0.45,
            "feature2": 0.32,
            "feature3": 0.23
        }
    },
    "all_trials": [
        {
            "trial": 1,
            "algorithm": "random_forest",
            "hyperparameters": {...},
            "score": 0.932,
            "training_time": 23.5
        }
    ],
    "training_summary": {
        "total_trials": 50,
        "successful_trials": 48,
        "failed_trials": 2,
        "total_time": "12m 45s",
        "best_trial": 33
    },
    "model_path": "/static/models/training_12345_best_model.pkl",
    "report_path": "/static/reports/training_12345_report.html"
}
```

#### Cancel Training
```http
DELETE /api/train/cancel/{training_id}
X-API-Key: your-api-key
```

### Model Evaluation

#### Evaluate Model
```http
POST /api/train/evaluate/{training_id}
Content-Type: application/json
X-API-Key: your-api-key

{
    "test_data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "test_target": [0, 1],
    "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
    "generate_plots": true,
    "detailed_report": true
}
```

**Response:**
```json
{
    "evaluation_results": {
        "accuracy": 0.945,
        "precision": 0.935,
        "recall": 0.950,
        "f1_score": 0.942,
        "roc_auc": 0.968
    },
    "confusion_matrix": [[85, 5], [3, 107]],
    "classification_report": {
        "0": {"precision": 0.97, "recall": 0.94, "f1-score": 0.95},
        "1": {"precision": 0.96, "recall": 0.97, "f1-score": 0.97}
    },
    "plots": {
        "confusion_matrix": "/static/charts/confusion_matrix_12345.png",
        "roc_curve": "/static/charts/roc_curve_12345.png",
        "precision_recall": "/static/charts/pr_curve_12345.png"
    },
    "detailed_report": "/static/reports/evaluation_12345.html"
}
```

#### Cross-Validate Model
```http
POST /api/train/cross-validate/{training_id}
Content-Type: application/json
X-API-Key: your-api-key

{
    "cv_folds": 10,
    "scoring_metrics": ["accuracy", "f1_score", "roc_auc"],
    "stratified": true
}
```

#### Compare Models
```http
POST /api/train/compare
Content-Type: application/json
X-API-Key: your-api-key

{
    "training_ids": ["training_12345", "training_67890"],
    "comparison_metrics": ["accuracy", "f1_score", "training_time"],
    "statistical_tests": true
}
```

**Response:**
```json
{
    "comparison": {
        "training_12345": {
            "algorithm": "random_forest",
            "accuracy": 0.945,
            "f1_score": 0.942,
            "training_time": "12m 45s"
        },
        "training_67890": {
            "algorithm": "xgboost",
            "accuracy": 0.938,
            "f1_score": 0.935,
            "training_time": "8m 32s"
        }
    },
    "best_model": {
        "by_accuracy": "training_12345",
        "by_f1_score": "training_12345",
        "by_speed": "training_67890"
    },
    "statistical_significance": {
        "accuracy_difference": 0.007,
        "p_value": 0.032,
        "significant": true
    }
}
```

### Model Management

#### Save Model
```http
POST /api/train/save-model/{training_id}
Content-Type: application/json
X-API-Key: your-api-key

{
    "model_name": "production_classifier_v1",
    "version": "1.0.0",
    "description": "Production-ready classification model",
    "tags": ["production", "classification", "v1"],
    "export_formats": ["pickle", "joblib", "onnx"],
    "include_preprocessor": true
}
```

**Response:**
```json
{
    "message": "Model saved successfully",
    "model_name": "production_classifier_v1",
    "model_paths": {
        "pickle": "/static/models/production_classifier_v1.pkl",
        "joblib": "/static/models/production_classifier_v1.joblib",
        "onnx": "/static/models/production_classifier_v1.onnx"
    },
    "model_size_mb": 15.6,
    "preprocessor_included": true,
    "saved_at": "2025-01-15T10:30:00Z"
}
```

#### Load Model for Prediction
```http
POST /api/train/load-model
Content-Type: application/json
X-API-Key: your-api-key

{
    "model_path": "/static/models/production_classifier_v1.pkl",
    "load_preprocessor": true
}
```

#### Make Prediction
```http
POST /api/train/predict
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "model_id": "loaded_model_12345",
    "return_probabilities": true,
    "apply_preprocessing": true
}
```

**Response:**
```json
{
    "predictions": [0, 1],
    "probabilities": [[0.85, 0.15], [0.25, 0.75]],
    "model_id": "loaded_model_12345",
    "preprocessing_applied": true,
    "prediction_time": 0.045
}
```

### Experiment Tracking

#### List Experiments
```http
GET /api/train/experiments
X-API-Key: your-api-key
```

**Response:**
```json
{
    "experiments": [
        {
            "experiment_id": "1",
            "name": "production_experiment",
            "creation_time": "2025-01-15T10:00:00Z",
            "last_updated": "2025-01-15T10:30:00Z",
            "run_count": 5,
            "active_runs": 0
        }
    ],
    "total_experiments": 1
}
```

#### Get Experiment Runs
```http
GET /api/train/experiments/{experiment_id}/runs
X-API-Key: your-api-key
```

**Response:**
```json
{
    "runs": [
        {
            "run_id": "run_12345",
            "run_name": "training_12345",
            "status": "FINISHED",
            "start_time": "2025-01-15T10:30:00Z",
            "end_time": "2025-01-15T10:42:45Z",
            "metrics": {
                "accuracy": 0.945,
                "f1_score": 0.942
            },
            "parameters": {
                "algorithm": "random_forest",
                "n_estimators": 200
            }
        }
    ],
    "total_runs": 1
}
```

#### Get Run Details
```http
GET /api/train/runs/{run_id}
X-API-Key: your-api-key
```

### Hyperparameter Optimization

#### Define Custom Search Space
```http
POST /api/train/hyperparameter-space
Content-Type: application/json
X-API-Key: your-api-key

{
    "space_name": "custom_rf_space",
    "algorithm": "random_forest",
    "hyperparameters": {
        "n_estimators": {
            "type": "choice",
            "values": [50, 100, 200, 500, 1000]
        },
        "max_depth": {
            "type": "range",
            "low": 3,
            "high": 20,
            "step": 1
        },
        "min_samples_split": {
            "type": "uniform",
            "low": 0.01,
            "high": 0.1
        },
        "bootstrap": {
            "type": "choice",
            "values": [true, false]
        }
    }
}
```

#### Start Advanced Optimization
```http
POST /api/train/optimize
Content-Type: application/json
X-API-Key: your-api-key

{
    "data_id": "data_12345",
    "algorithm": "random_forest",
    "search_space": "custom_rf_space",
    "optimization_algorithm": "bayesian",  # bayesian, random, grid, evolutionary
    "max_trials": 100,
    "max_time_minutes": 60,
    "early_stopping": {
        "enabled": true,
        "patience": 10,
        "min_delta": 0.001
    },
    "parallel_trials": 4
}
```

### AutoML and Model Selection

#### Start AutoML
```http
POST /api/train/automl
Content-Type: application/json
X-API-Key: your-api-key

{
    "data_id": "data_12345",
    "task_type": "classification",
    "time_budget_minutes": 120,
    "algorithms": "auto",              # auto, or specific list
    "ensemble_methods": ["voting", "stacking", "blending"],
    "feature_engineering": {
        "polynomial_features": true,
        "interaction_features": true,
        "feature_selection": true,
        "dimensionality_reduction": true
    },
    "model_selection_criteria": "best_score",
    "interpretability_required": true
}
```

**Response:**
```json
{
    "message": "AutoML started successfully",
    "automl_id": "automl_12345",
    "estimated_duration": "2 hours",
    "algorithms_to_try": ["random_forest", "xgboost", "lightgbm", "neural_network"],
    "ensemble_methods": ["voting", "stacking"],
    "feature_engineering_enabled": true,
    "started_at": "2025-01-15T10:30:00Z"
}
```

#### Get AutoML Results
```http
GET /api/train/automl/{automl_id}/results
X-API-Key: your-api-key
```

### Data Processing and Analysis

#### Analyze Dataset
```http
POST /api/train/analyze-data
Content-Type: application/json
X-API-Key: your-api-key

{
    "data_id": "data_12345",
    "generate_profile": true,
    "correlation_analysis": true,
    "outlier_detection": true,
    "feature_importance": true
}
```

**Response:**
```json
{
    "data_profile": {
        "shape": [1000, 4],
        "missing_values": 5,
        "duplicate_rows": 2,
        "numeric_features": 3,
        "categorical_features": 0,
        "target_distribution": {"0": 450, "1": 550}
    },
    "correlation_matrix": {...},
    "outliers": {
        "feature1": [5, 15, 25],
        "feature2": [10, 20]
    },
    "feature_statistics": {...},
    "recommendations": [
        "Consider removing outliers in feature1",
        "Feature2 shows high correlation with target"
    ],
    "profile_report": "/static/reports/data_profile_12345.html"
}
```

#### Feature Engineering
```http
POST /api/train/feature-engineering
Content-Type: application/json
X-API-Key: your-api-key

{
    "data_id": "data_12345",
    "operations": {
        "polynomial_features": {
            "degree": 2,
            "interaction_only": false
        },
        "feature_selection": {
            "method": "mutual_info",
            "k_best": 10
        },
        "scaling": {
            "method": "standard"
        },
        "encoding": {
            "categorical_method": "onehot"
        }
    }
}
```

## Usage Examples

### Basic Training Workflow

```python
import requests
import json
import pandas as pd

# API configuration
API_BASE = "http://localhost:8001"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# 1. Upload training data
df = pd.read_csv("training_data.csv")
with open("training_data.csv", "rb") as f:
    files = {"file": f}
    data = {
        "target_column": "target",
        "feature_columns": ",".join(df.columns[:-1])
    }
    response = requests.post(f"{API_BASE}/api/train/upload-data",
                           headers={"X-API-Key": API_KEY},
                           files=files, data=data)
data_id = response.json()["data_id"]

# 2. Start training
training_config = {
    "data_id": data_id,
    "task_type": "classification",
    "test_size": 0.2,
    "optimization_strategy": "bayesian",
    "max_trials": 50,
    "algorithms": ["random_forest", "xgboost", "lightgbm"],
    "scoring_metric": "f1_score"
}
response = requests.post(f"{API_BASE}/api/train/start-with-data",
                        headers=headers, json=training_config)
training_id = response.json()["training_id"]

# 3. Monitor training progress
import time
while True:
    response = requests.get(f"{API_BASE}/api/train/status/{training_id}",
                           headers={"X-API-Key": API_KEY})
    status = response.json()
    
    print(f"Status: {status['status']}, Progress: {status['progress']:.1f}%")
    
    if status["status"] in ["completed", "failed", "cancelled"]:
        break
    
    time.sleep(10)

# 4. Get results
response = requests.get(f"{API_BASE}/api/train/results/{training_id}",
                       headers={"X-API-Key": API_KEY})
results = response.json()

print("Best model:", results["best_model"]["algorithm"])
print("Best score:", results["best_model"]["performance"]["f1_score"])
```

### Advanced AutoML Pipeline

```python
# Start comprehensive AutoML
automl_config = {
    "data_id": data_id,
    "task_type": "classification", 
    "time_budget_minutes": 120,
    "algorithms": "auto",
    "ensemble_methods": ["voting", "stacking"],
    "feature_engineering": {
        "polynomial_features": True,
        "feature_selection": True,
        "dimensionality_reduction": True
    },
    "interpretability_required": True
}
response = requests.post(f"{API_BASE}/api/train/automl",
                        headers=headers, json=automl_config)
automl_id = response.json()["automl_id"]

# Monitor AutoML progress
while True:
    response = requests.get(f"{API_BASE}/api/train/automl/{automl_id}/results",
                           headers={"X-API-Key": API_KEY})
    
    if response.status_code == 200:
        results = response.json()
        if results.get("status") == "completed":
            break
    
    time.sleep(30)

print("AutoML completed!")
print("Best ensemble model:", results["best_ensemble"])
```

### Model Comparison and Selection

```python
# Train multiple models for comparison
training_configs = [
    {"algorithms": ["random_forest"], "max_trials": 20},
    {"algorithms": ["xgboost"], "max_trials": 20},
    {"algorithms": ["lightgbm"], "max_trials": 20}
]

training_ids = []
for config in training_configs:
    config.update({"data_id": data_id, "task_type": "classification"})
    response = requests.post(f"{API_BASE}/api/train/start-with-data",
                           headers=headers, json=config)
    training_ids.append(response.json()["training_id"])

# Wait for all trainings to complete
# ... (monitoring code)

# Compare models
comparison_request = {
    "training_ids": training_ids,
    "comparison_metrics": ["accuracy", "f1_score", "training_time"],
    "statistical_tests": True
}
response = requests.post(f"{API_BASE}/api/train/compare",
                        headers=headers, json=comparison_request)
comparison = response.json()

print("Model comparison:", comparison["comparison"])
print("Best by accuracy:", comparison["best_model"]["by_accuracy"])
```

### Custom Hyperparameter Optimization

```python
# Define custom search space
custom_space = {
    "space_name": "advanced_rf_space",
    "algorithm": "random_forest",
    "hyperparameters": {
        "n_estimators": {"type": "choice", "values": [100, 200, 500, 1000]},
        "max_depth": {"type": "range", "low": 5, "high": 25, "step": 1},
        "min_samples_split": {"type": "uniform", "low": 0.01, "high": 0.2},
        "max_features": {"type": "choice", "values": ["sqrt", "log2", 0.5, 0.8]}
    }
}
requests.post(f"{API_BASE}/api/train/hyperparameter-space",
             headers=headers, json=custom_space)

# Start optimization with custom space
optimization_config = {
    "data_id": data_id,
    "algorithm": "random_forest",
    "search_space": "advanced_rf_space",
    "optimization_algorithm": "bayesian",
    "max_trials": 100,
    "early_stopping": {"enabled": True, "patience": 15}
}
response = requests.post(f"{API_BASE}/api/train/optimize",
                        headers=headers, json=optimization_config)
```

## Error Handling

### Common Error Codes

- **400 Bad Request**: Invalid training configuration or data format
- **401 Unauthorized**: Missing or invalid API key
- **404 Not Found**: Training ID, data ID, or model not found
- **409 Conflict**: Training already in progress or resource conflict
- **422 Unprocessable Entity**: Data validation or training errors
- **500 Internal Server Error**: Training engine or system errors
- **503 Service Unavailable**: Engine not initialized or overloaded

### Error Response Format

```json
{
    "error": "TrainingError",
    "message": "Training failed due to insufficient data",
    "details": {
        "training_id": "training_12345",
        "data_shape": [10, 3],
        "minimum_samples_required": 50
    },
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_12345"
}
```

## Best Practices

### Training Configuration

1. **Data Quality**: Ensure clean, well-preprocessed data
2. **Cross-Validation**: Use appropriate CV strategies for your data
3. **Time Budgets**: Set realistic time limits for optimization
4. **Algorithm Selection**: Choose algorithms appropriate for your data size
5. **Metrics**: Select evaluation metrics that align with business objectives

### Performance Optimization

1. **Parallel Processing**: Use multiple workers for hyperparameter optimization
2. **Early Stopping**: Enable early stopping to save computation time
3. **Resource Monitoring**: Monitor memory and CPU usage during training
4. **Incremental Learning**: Use incremental learning for large datasets

### Model Management

1. **Version Control**: Maintain proper model versioning
2. **Experiment Tracking**: Use MLflow for comprehensive experiment tracking
3. **Model Documentation**: Include detailed metadata and descriptions
4. **Validation**: Always validate models on holdout test sets

### Production Deployment

1. **Model Testing**: Thoroughly test models before production deployment
2. **Performance Monitoring**: Monitor model performance in production
3. **A/B Testing**: Implement A/B testing for model comparisons
4. **Rollback Strategy**: Have rollback plans for failed deployments

## Advanced Features

### Custom Algorithms

Support for custom ML algorithms:

```python
class CustomAlgorithm:
    def fit(self, X, y):
        # Custom training logic
        pass
    
    def predict(self, X):
        # Custom prediction logic
        pass
```

### Multi-Objective Optimization

Optimize for multiple objectives simultaneously:

```python
multi_objective_config = {
    "objectives": [
        {"metric": "accuracy", "weight": 0.7},
        {"metric": "model_size", "weight": 0.2, "minimize": True},
        {"metric": "inference_time", "weight": 0.1, "minimize": True}
    ]
}
```

### Distributed Training

Support for distributed training across multiple nodes:

```python
distributed_config = {
    "distributed": True,
    "worker_nodes": ["node1:8001", "node2:8001"],
    "strategy": "parameter_server"
}
```

## Related Documentation

- [Training Engine](../engine/train_engine.md) - Core training engine
- [Hyperparameter Optimizer](../optimizer/asht.md) - ASHT optimizer
- [Configuration System](../configs.md) - Configuration management
- [Model Manager API](model_manager_api.md) - Model management API
- [Experiment Tracker](../engine/experiment_tracker.md) - MLflow integration

---

*The Train Engine API provides a comprehensive interface for automated machine learning with advanced features including hyperparameter optimization, ensemble methods, and experiment tracking.*
