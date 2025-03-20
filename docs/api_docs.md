# Genta AutoML API Documentation

## Overview

This documentation covers a FastAPI-based Machine Learning API designed for model training, inference, and management. The API provides a comprehensive set of endpoints for machine learning operations, including data preprocessing, model training, inference, quantization, error analysis, and more.

## Table of Contents

1. [Architecture](#architecture)
2. [Key Features](#key-features)
3. [Prerequisites and Dependencies](#prerequisites-and-dependencies)
4. [API Endpoints](#api-endpoints)
   - [Authentication](#authentication)
   - [Model Management](#model-management)
   - [Training](#training)
   - [Inference](#inference)
   - [Data Processing](#data-processing)
   - [Analysis](#analysis)
   - [Security](#security)
   - [User Management](#user-management)
5. [Data Models](#data-models)
6. [Security Considerations](#security-considerations)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)
9. [Deployment Guide](#deployment-guide)

## Architecture

The API follows a modular architecture with several key components:

- **FastAPI Application**: Core application handling HTTP requests and responses
- **Training Engine**: Handles model training and optimization
- **Inference Engine**: Manages prediction and model serving
- **Data Preprocessor**: Handles data cleaning, normalization, and preparation
- **Secure Model Manager**: Provides encryption and access control for models
- **Quantizer**: Optimizes models for deployment efficiency

The system uses a lazy initialization pattern to efficiently manage resources and prevent redundant initialization.

## Key Features

1. **Authentication and Authorization**
   - JWT-based authentication
   - Role-based access control (user/admin)
   - Secure password hashing with bcrypt

2. **Model Training and Management**
   - Training with hyperparameter optimization
   - Model versioning and metadata
   - Model export in multiple formats (sklearn, ONNX, PMML, etc.)
   - Model comparison tools

3. **Inference Pipeline**
   - Batch and individual predictions
   - Metrics tracking and logging
   - Input validation

4. **Data Processing**
   - Missing value handling
   - Normalization and standardization
   - Outlier detection

5. **Advanced Analytics**
   - Data drift detection
   - Error analysis
   - Performance metrics and visualization

6. **Optimization and Security**
   - Model quantization
   - Encrypted model storage
   - Model integrity verification

## Prerequisites and Dependencies

### Core Dependencies
- FastAPI
- Uvicorn
- NumPy
- Pandas
- scikit-learn
- PyJWT
- bcrypt

### Environment Variables
- `MODEL_PATH`: Directory for storing models (default: "./models")
- `TEMP_UPLOAD_FOLDER`: Directory for temporary files (default: "./uploads")
- `SECRET_KEY`: JWT secret key (required in production)
- `TOKEN_EXPIRATION`: Token lifetime in seconds (default: 86400)
- `CORS_ORIGINS`: Allowed CORS origins (default: "*")
- `HOST`: Host to bind server (default: "0.0.0.0")
- `PORT`: Port to run server (default: 5000)

## API Endpoints

### Authentication

#### POST `/api/login`
Authenticates a user and returns a JWT token.

**Request**:
```json
{
  "username": "user",
  "password": "user123"
}
```

**Response**:
```json
{
  "token": "eyJhbGc...",
  "username": "user",
  "roles": ["user"],
  "expires_in": 86400
}
```

### Model Management

#### GET `/api/models`
Lists all available models.

**Response**:
```json
{
  "models": [
    {
      "name": "model1",
      "path": "./models/model1.pkl",
      "size": 123456,
      "modified": "2023-01-01T12:00:00"
    }
  ],
  "count": 1
}
```

#### GET `/api/models/{model_name}`
Gets detailed information about a specific model.

**Response**:
```json
{
  "name": "model1",
  "path": "./models/model1.pkl",
  "size": 123456,
  "modified": "2023-01-01T12:00:00",
  "model_type": "RandomForestClassifier",
  "features": ["feature1", "feature2"],
  "metrics": {
    "accuracy": 0.95
  }
}
```

#### DELETE `/api/models/{model_name}`
Deletes a model (admin only).

**Response**:
```json
{
  "message": "Model model1 successfully deleted",
  "name": "model1"
}
```

#### POST `/api/models/compare`
Compares multiple models' performance.

**Request**:
```json
{
  "models": ["model1", "model2"],
  "metrics": ["accuracy", "f1_score"]
}
```

**Response**:
```json
{
  "models": ["model1", "model2"],
  "metrics": ["accuracy", "f1_score"],
  "best_model": "model1",
  "metric_ranges": {
    "accuracy": [0.9, 0.95],
    "f1_score": [0.88, 0.94]
  },
  "timestamp": "2023-01-01T12:00:00",
  "accuracy_plot_base64": "...",
  "f1_score_plot_base64": "..."
}
```

#### GET `/api/models/export/{model_name}`
Exports a model in different formats.

**Query Parameters**:
- `format`: Export format (sklearn, onnx, pmml, tf, torchscript)
- `include_pipeline`: Include preprocessing pipeline

**Response**: File download

### Training

#### POST `/api/train`
Trains a new model with uploaded data.

**Form Data**:
- `file`: Training data file
- `model_type`: "classification" or "regression"
- `model_name`: Name for the model
- `target_column`: Target variable column name

**Response**:
```json
{
  "status": "success",
  "message": "Model model1 trained successfully",
  "model_name": "model1",
  "model_path": "./models/model1.pkl",
  "metrics": {
    "accuracy": 0.95,
    "f1_score": 0.94
  }
}
```

### Inference

#### POST `/api/predict`
Makes predictions using a loaded model.

**Query Parameters**:
- `model`: Model name
- `batch_size`: Batch size for large datasets

**Request Options**:
- JSON data: `{ "model": "model1", "data": [[1.0, 2.0], [3.0, 4.0]] }`
- File upload: CSV, JSON, etc.

**Response**:
```json
{
  "predictions": [0, 1],
  "model": "model1",
  "sample_count": 2,
  "processing_time_ms": 15,
  "metadata": {
    "model_version": "1.0"
  }
}
```

### Data Processing

#### POST `/api/preprocess`
Preprocesses data using the data preprocessor.

**Form Data**:
- `file`: Data file
- `normalize`: Whether to normalize data
- `handle_missing`: Whether to handle missing values
- `detect_outliers`: Whether to detect outliers

**Response**: Preprocessed file download

### Analysis

#### POST `/api/drift-detection`
Detects data drift between reference and new data.

**Form Data**:
- `reference_file`: Reference data file
- `new_file`: New data file
- `threshold`: Drift threshold

**Response**:
```json
{
  "dataset_drift": 0.15,
  "drift_detected": true,
  "drifted_features_count": 3,
  "total_features": 10,
  "drift_threshold": 0.1,
  "drifted_features": ["feature1", "feature3", "feature7"],
  "timestamp": "2023-01-01T12:00:00",
  "drift_plot_base64": "...",
  "distribution_plot_base64": "..."
}
```

#### POST `/api/error-analysis/{model_name}`
Performs detailed error analysis on model predictions.

**Form Data**:
- `file`: Test data file
- `target_column`: Target variable column name
- `n_samples`: Number of samples to analyze

**Response**:
```json
{
  "model_name": "model1",
  "dataset_size": 1000,
  "error_count": 50,
  "error_rate": 0.05,
  "class_metrics": { ... },
  "confusion_matrix_plot_base64": "..."
}
```

### Security

#### POST `/api/models/secure/{model_name}`
Creates a secure, encrypted version of a model.

**Request**:
```json
{
  "access_code": "secret123"
}
```

**Response**:
```json
{
  "message": "Model model1 secured successfully",
  "secure_path": "./models/model1_secure.pkl"
}
```

#### GET `/api/models/verify/{model_name}`
Verifies the integrity of a secured model.

**Response**:
```json
{
  "status": "valid",
  "message": "Model model1 integrity verification successful"
}
```

#### POST `/api/quantize/{model_name}`
Quantizes a model for improved deployment efficiency.

**Request**:
```json
{
  "quantization_type": "int8",
  "quantization_mode": "dynamic_per_batch"
}
```

**Response**:
```json
{
  "message": "Model model1 quantized successfully",
  "original_size_bytes": 123456,
  "quantized_size_bytes": 30864,
  "compression_ratio": 4.0,
  "quantized_path": "./models/model1_quantized.pkl"
}
```

### User Management

#### GET `/api/users`
Lists all users (admin only).

**Response**:
```json
{
  "users": [
    {
      "username": "admin",
      "roles": ["admin"]
    },
    {
      "username": "user",
      "roles": ["user"]
    }
  ],
  "count": 2
}
```

#### POST `/api/users`
Creates a new user (admin only).

**Request**:
```json
{
  "username": "newuser",
  "password": "password123",
  "roles": ["user"]
}
```

**Response**:
```json
{
  "message": "User newuser created successfully",
  "username": "newuser",
  "roles": ["user"]
}
```

#### DELETE `/api/users/{username}`
Deletes a user (admin only).

**Response**:
```json
{
  "message": "User newuser deleted successfully"
}
```

## Data Models

### User Models
- `UserLogin`: Login credentials
- `UserResponse`: User information
- `TokenResponse`: JWT token response
- `UserCreate`: New user creation

### Model Management Models
- `ModelInfo`: Model metadata
- `ModelsList`: List of models
- `ModelMetrics`: Model performance metrics
- `ModelComparison`: Model comparison request

### Inference Models
- `PredictionData`: Data for prediction
- `PredictionResponse`: Prediction results

### Analysis Models
- `DriftResult`: Data drift analysis results
- `QuantizationOptions`: Model quantization options
- `QuantizationResult`: Quantization results

## Security Considerations

### Authentication and Authorization
- JWT tokens for stateless authentication
- Role-based access control
- Token expiration
- bcrypt for password hashing

### Data Security
- File validation to prevent unsafe uploads
- Secure temporary file handling
- Path validation to prevent directory traversal
- Model encryption

### API Security
- CORS protection
- Request logging
- Rate limiting (recommended for production)

## Error Handling

The API uses consistent error responses with appropriate HTTP status codes:

```json
{
  "error": "Detailed error message"
}
```

Common status codes:
- 400: Bad Request (invalid input)
- 401: Unauthorized (authentication failure)
- 403: Forbidden (insufficient permissions)
- 404: Not Found (resource doesn't exist)
- 409: Conflict (resource already exists)
- 500: Internal Server Error

## Best Practices

### Model Management
1. Use meaningful, descriptive model names
2. Keep model versions consistent
3. Regularly clean up unused models
4. Track model lineage and metadata

### Performance Optimization
1. Use batch prediction for large datasets
2. Consider quantizing models for faster inference
3. Monitor memory usage with large models
4. Implement caching for frequent predictions

### Security
1. Always use HTTPS in production
2. Rotate JWT secret keys periodically
3. Implement rate limiting for authentication endpoints
4. Follow the principle of least privilege for user roles

## Deployment Guide

### Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export MODEL_PATH=./models
   export TEMP_UPLOAD_FOLDER=./uploads
   export SECRET_KEY=your_secret_key
   export CORS_ORIGINS=https://yourdomain.com
   ```

### Running the Server

Development:
```bash
python api.py
```

Production (with Gunicorn):
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```
