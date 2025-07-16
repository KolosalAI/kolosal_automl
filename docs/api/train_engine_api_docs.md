# ML Training Engine API Documentation

## Overview
The ML Training Engine API provides a RESTful interface for training, evaluating, managing, and making predictions with machine learning models. It serves as a comprehensive interface to the MLTrainingEngine module, allowing users to upload data, train various machine learning models, optimize hyperparameters, visualize model performance, and generate predictions through a simple HTTP API.

The API integrates with the complete kolosal AutoML system and provides production-ready endpoints for automated machine learning workflows.

## Prerequisites
- Python ≥3.10
- FastAPI framework
- Core ML dependencies:
  ```bash
  pip install fastapi uvicorn pandas numpy scikit-learn
  ```
- MLTrainingEngine module from kolosal AutoML
- Additional requirements based on model types used (XGBoost, LightGBM, etc.)

## Installation
```bash
# Install required dependencies
pip install -r requirements.txt

# Start the API server
python start_api.py

# Or using uvicorn directly
uvicorn modules.api.train_engine_api:app --host 0.0.0.0 --port 8000 --reload
```

## Usage
The API can be accessed once the server is running:
```
http://localhost:8000
```

API documentation is available at:
```
http://localhost:8000/docs
```

## Configuration
The API creates several directories for storing files:
- `static/models`: Trained models
- `static/reports`: Generated reports
- `static/uploads`: Uploaded datasets
- `static/charts`: Visualization charts

## Architecture
The ML Training Engine API is built using FastAPI and provides a REST interface to the MLTrainingEngine for machine learning operations. The API follows a modular architecture:

1. **API Layer**: FastAPI routes and endpoints
2. **Engine Layer**: MLTrainingEngine for model training and evaluation
3. **File Storage**: Local filesystem for models, reports, and data
4. **Background Tasks**: Asynchronous processing for time-consuming operations

---

## Endpoints

### Root
```
GET /
```

Returns basic API information and initialization status.

**Response:**
```json
{
  "name": "ML Training Engine API",
  "version": "1.0.0",
  "documentation": "/docs",
  "status": "engine_initialized" or "engine_not_initialized"
}
```

### Initialize Engine
```
POST /api/initialize
```

Initializes the ML Training Engine with the specified configuration.

**Request Body:**
```json
{
  "engine_config": {
    "task_type": "classification",
    "model_path": "models",
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
    "n_jobs": -1,
    "verbose": 1,
    "optimization_strategy": "random_search",
    "optimization_iterations": 20,
    "model_selection_criteria": "f1",
    "feature_selection": true,
    "feature_selection_method": "mutual_info",
    "feature_selection_k": 10,
    "early_stopping": true,
    "early_stopping_rounds": 10,
    "auto_save": true,
    "checkpointing": true,
    "checkpoint_path": "checkpoints",
    "experiment_tracking": true,
    "generate_model_summary": true,
    "log_level": "INFO"
  },
  "preprocessing_config": {
    "handle_nan": true,
    "nan_strategy": "mean",
    "detect_outliers": true,
    "outlier_handling": "clip",
    "categorical_encoding": "one_hot",
    "normalization": "standard",
    "feature_interactions": false,
    "polynomial_features": false,
    "polynomial_degree": 2
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "ML Engine initialized successfully",
  "config": {
    "task_type": "classification",
    "model_path": "models",
    "experiment_tracking": true
  }
}
```

### Upload File
```
POST /api/upload
```

Uploads a data file for training, evaluation, or prediction.

**Request:**
- Form data with file in `file` field

**Response:**
```json
{
  "status": "success",
  "filename": "dataset_1620000000.csv",
  "original_filename": "dataset.csv",
  "file_path": "static/uploads/dataset_1620000000.csv",
  "file_size": 102400,
  "row_count": 1000,
  "column_count": 10,
  "columns": ["feature1", "feature2", "target"],
  "column_types": {
    "feature1": "float64",
    "feature2": "object",
    "target": "int64"
  },
  "sample_data": [
    {"feature1": 0.5, "feature2": "A", "target": 1},
    {"feature1": 0.7, "feature2": "B", "target": 0}
  ]
}
```

### Train Model
```
POST /api/train
```

Trains a machine learning model with the specified parameters.

**Request Body:**
```json
{
  "model_type": "random_forest",
  "model_name": "rf_model_1",
  "param_grid": {
    "n_estimators": [100, 200, 300],
    "max_depth": [null, 10, 20, 30],
    "min_samples_split": [2, 5, 10]
  },
  "train_data_file": "train_data.csv",
  "target_column": "target",
  "feature_columns": ["feature1", "feature2", "feature3"],
  "test_size": 0.2,
  "validation_data_file": "validation_data.csv"
}
```

**Response:**
```json
{
  "status": "training_started",
  "task_id": "train_1620000000",
  "message": "Model training started in the background"
}
```

### Get Training Status
```
GET /api/train/status/{task_id}
```

Gets the status of a training task.

**Response:**
```json
{
  "status": "running",
  "model_name": "rf_model_1",
  "progress": 0.5,
  "eta": 120,
  "started_at": "2025-05-11T12:00:00",
  "completed_at": null,
  "error": null
}
```

### List Models
```
GET /api/models
```

Lists all trained models.

**Response:**
```json
{
  "models": [
    {
      "name": "rf_model_1",
      "type": "RandomForestClassifier",
      "metrics": {
        "accuracy": 0.95,
        "f1": 0.94
      },
      "training_time": 10.5,
      "is_best": true,
      "feature_count": 10,
      "loaded_from": "models/rf_model_1.pkl"
    }
  ],
  "best_model": "rf_model_1",
  "count": 1
}
```

### Predict
```
GET /api/models/{model_name}/predict
```

Makes predictions using a trained model.

**Request Body:**
```json
{
  "data_file": "predict_data.csv",
  "feature_columns": ["feature1", "feature2", "feature3"],
  "return_probabilities": true
}
```

**Response:**
```json
{
  "model_name": "rf_model_1",
  "predictions": [0, 1, 0, 1],
  "prediction_count": 4,
  "probabilities": true,
  "timestamp": "2025-05-11T12:00:00"
}
```

### Explain Model
```
POST /api/models/{model_name}/explain
```

Generates model explainability visualizations.

**Request Body:**
```json
{
  "data_file": "explain_data.csv",
  "feature_columns": ["feature1", "feature2", "feature3"],
  "method": "shap"
}
```

**Response:**
```json
{
  "model_name": "rf_model_1",
  "method": "shap",
  "importance": {
    "feature1": 0.5,
    "feature2": 0.3,
    "feature3": 0.2
  },
  "plot_url": "/static/charts/shap_rf_model_1_1620000000.png",
  "timestamp": "2025-05-11T12:00:00"
}
```

### Save Model
```
POST /api/models/save/{model_name}
```

Saves a model to disk.

**Query Parameters:**
- `include_preprocessor`: Whether to include the preprocessor (default: true)

**Response:**
```json
{
  "model_name": "rf_model_1",
  "save_path": "models/rf_model_1.pkl",
  "include_preprocessor": true,
  "timestamp": "2025-05-11T12:00:00"
}
```

### Load Model
```
POST /api/models/load
```

Loads a model from disk or an uploaded file.

**Request:**
- Form data with `file` field (optional)
- Form data with `path` field (optional)
- Form data with `model_name` field (optional)

**Response:**
```json
{
  "status": "success",
  "model_name": "rf_model_1",
  "model_type": "RandomForestClassifier",
  "is_best": true,
  "timestamp": "2025-05-11T12:00:00"
}
```

### Compare Models
```
GET /api/models/compare
```

Compares performance across all trained models.

**Response:**
```json
{
  "models": {
    "rf_model_1": {
      "accuracy": 0.95,
      "f1": 0.94
    },
    "xgb_model_1": {
      "accuracy": 0.92,
      "f1": 0.91
    }
  },
  "best_model": "rf_model_1",
  "metric_names": ["accuracy", "f1"]
}
```

### Generate Report
```
POST /api/reports/generate
```

Generates a comprehensive report of all models.

**Response:**
```json
{
  "report_path": "static/reports/model_report_1620000000.md",
  "download_url": "/api/reports/download/model_report_1620000000.md",
  "timestamp": "2025-05-11T12:00:00"
}
```

### Download Report
```
GET /api/reports/download/{filename}
```

Downloads a generated report.

**Response:**
- Markdown file

### Engine Status
```
POST /api/engine/status
```

Gets the status of the ML Engine.

**Response:**
```json
{
  "initialized": true,
  "task_type": "classification",
  "model_count": 2,
  "best_model": "rf_model_1",
  "training_complete": true,
  "config": {
    "task_type": "classification",
    "model_path": "models"
  },
  "preprocessor": {
    "type": "StandardScaler",
    "configured": true
  }
}
```

### Shutdown Engine
```
POST /api/engine/shutdown
```

Shuts down the ML Engine and releases resources.

**Response:**
```json
{
  "status": "success",
  "message": "Engine shut down successfully",
  "timestamp": "2025-05-11T12:00:00"
}
```

### Evaluate Model
```
POST /api/models/{model_name}/evaluate
```

Evaluates a model on test data.

**Request Body:**
```json
{
  "test_data_file": "test_data.csv",
  "target_column": "target",
  "feature_columns": ["feature1", "feature2", "feature3"],
  "detailed": true
}
```

**Response:**
```json
{
  "model_name": "rf_model_1",
  "metrics": {
    "accuracy": 0.95,
    "f1": 0.94,
    "precision": 0.96,
    "recall": 0.93,
    "confusion_matrix": [[100, 5], [7, 95]]
  },
  "timestamp": "2025-05-11T12:00:00",
  "detailed": true
}
```

---

## Data Models

### TaskTypeEnum
- `classification`
- `regression`
- `clustering`
- `time_series`

### OptimizationStrategyEnum
- `grid_search`
- `random_search`
- `bayesian_optimization`
- `optuna`
- `asht`
- `hyperx`

### ModelSelectionCriteriaEnum
- `accuracy`
- `f1`
- `precision`
- `recall`
- `roc_auc`
- `mean_squared_error`
- `root_mean_squared_error`
- `mean_absolute_error`
- `r2`
- `explained_variance`

### NormalizationTypeEnum
- `standard`
- `minmax`
- `robust`
- `none`

### EngineConfigRequest
Configuration for the ML Training Engine.

### PreprocessingConfigRequest
Configuration for data preprocessing.

### InitializeEngineRequest
Request to initialize the engine with configuration.

### TrainModelRequest
Request to train a machine learning model.

### EvaluateModelRequest
Request to evaluate a trained model.

### PredictRequest
Request to make predictions with a trained model.

### ExplainabilityRequest
Request to generate model explainability.

### ModelInfoResponse
Response with model information.

### TrainingStatusResponse
Response with training task status.

---

## Security & Compliance
- API allows cross-origin resource sharing (CORS) with all origins.
- Files are saved locally in the specified directories.
- No authentication mechanism is implemented by default.

> Last Updated: 2025-05-11