# Kolosal AutoML API

A powerful and comprehensive Automated Machine Learning API that streamlines the entire machine learning workflow - from data preprocessing to model deployment.

## Overview

Kolosal AutoML API is an end-to-end machine learning platform designed to automate the process of building, optimizing, and deploying machine learning models. Built on top of a robust ML Training Engine with advanced optimization capabilities, it provides a RESTful API interface to make machine learning accessible to developers of all skill levels.

![Kolosal AutoML Architecture](https://via.placeholder.com/800x400?text=Kolosal+AutoML+Architecture)

## Key Features

- **Automated Model Training**: Train classification and regression models with intelligent hyperparameter optimization
- **Resource-Aware Optimization**: Automatically adapts to the available computational resources with different optimization modes
- **Advanced Preprocessing**: Automated data cleaning, normalization, feature selection, and outlier detection
- **Model Management**: Version, store, compare, and manage models through a unified API
- **Real-time Inference**: Fast, efficient prediction API with batch processing support
- **Performance Analysis**: Comprehensive error analysis, feature importance, and data drift detection
- **Model Quantization**: Optimize models for deployment with intelligent compression
- **Multiple Export Formats**: Export trained models in various formats (sklearn, ONNX, PMML, TensorFlow, PyTorch)
- **Secure Model Storage**: Encryption and access control for sensitive models
- **Python Client Library**: Simple, intuitive client library for Python developers

## Installation

### Server Setup

```bash
# Clone the repository
git clone https://github.com/kolosal/automl-api.git
cd automl-api

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (optional)
export MODEL_PATH="./models"
export PORT=5000
export HOST="0.0.0.0"

# Start the server
python kolosal_automl_api.py
```

### Client Installation

```bash
pip install kolosal-automl-client
```

## Quick Start

### Python Client Usage

```python
from kolosal_client import KolosalAutoML

# Initialize client
client = KolosalAutoML(
    base_url="http://localhost:5000",
    username="admin",
    password="admin123"
)

# Check API status
status = client.check_status()
print(f"API Status: {status['status']}, Version: {status['version']}")

# Train a model with Iris dataset
from sklearn.datasets import load_iris
import pandas as pd

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Train the model
result = client.train_model(
    data=df,
    target_column='target',
    model_type='classification',
    model_name='iris_classifier'
)

print(f"Model trained: {result['model_name']}")
print(f"Accuracy: {result['metrics']['accuracy']:.4f}")

# Make predictions
test_data = df.drop('target', axis=1).iloc[:5]
predictions = client.predict(model='iris_classifier', data=test_data)
print(f"Predictions: {predictions['predictions']}")

# Analyze feature importance
importance = client.feature_importance('iris_classifier')
print("Top features by importance:")
for feature, score in importance['top_features'].items():
    print(f"- {feature}: {score:.4f}")
```

## API Documentation

### Authentication

The API uses JWT token-based authentication. First, obtain a token by logging in:

```http
POST /api/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

Response:

```json
{
  "token": "eyJhbGc...",
  "username": "admin",
  "roles": ["admin"],
  "expires_in": 86400
}
```

Use this token in subsequent requests:

```http
GET /api/models
Authorization: Bearer eyJhbGc...
```

### API Endpoints

#### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List all available models |
| `/api/models/{model_name}` | GET | Get detailed information about a specific model |
| `/api/models/{model_name}` | DELETE | Delete a model |
| `/api/models/compare` | POST | Compare multiple models' performance |
| `/api/models/export/{model_name}` | GET | Export a model in different formats |
| `/api/models/{model_name}/download` | GET | Download a trained model |
| `/api/models/{model_name}/metadata` | POST | Update model metadata |
| `/api/models/{model_name}/metrics` | GET | Get model performance metrics |

#### Training & Inference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/train` | POST | Train a new model with uploaded data |
| `/api/predict` | POST | Make predictions using a trained model |
| `/api/preprocess` | POST | Preprocess data using the data preprocessor |
| `/api/quantize/{model_name}` | POST | Quantize a model for improved deployment efficiency |

#### Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/error-analysis/{model_name}` | POST | Perform detailed error analysis on model predictions |
| `/api/drift-detection/{model_name}` | POST | Detect data drift between reference and new data |
| `/api/feature-importance/{model_name}` | POST | Generate a feature importance report for a model |

#### Utility

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Get API status and version information |
| `/api/config` | GET | Get the current AutoML configuration |

## Configuration Options

### Optimization Modes

The API offers several optimization modes that adapt to different computational resources and requirements:

- **BALANCED**: Default mode, balances performance and resource usage (uses ~75% of CPU cores)
- **CONSERVATIVE**: Minimizes resource usage, suitable for shared environments (uses ~50% of CPU cores)
- **PERFORMANCE**: Prioritizes performance over resource efficiency (uses ~90% of CPU cores)
- **FULL_UTILIZATION**: Maximizes performance using all available resources (uses 100% of CPU cores)
- **MEMORY_SAVING**: Minimizes memory usage, suitable for constrained environments

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Directory for storing models | `./models` |
| `TEMP_UPLOAD_FOLDER` | Directory for temporary files | `./uploads` |
| `SECRET_KEY` | JWT secret key | Random (generated) |
| `TOKEN_EXPIRATION` | Token lifetime in seconds | `86400` (24 hours) |
| `CORS_ORIGINS` | Allowed CORS origins | `*` |
| `HOST` | Host to bind server | `0.0.0.0` |
| `PORT` | Port to run server | `5000` |
| `DEFAULT_OPTIMIZATION_MODE` | Default optimization mode | `BALANCED` |

## Python Client Reference

### Initialization

```python
from kolosal_client import KolosalAutoML

# With username/password
client = KolosalAutoML(
    base_url="http://localhost:5000",
    username="admin",
    password="admin123"
)

# With API key
client = KolosalAutoML(
    base_url="http://localhost:5000",
    api_key="your-api-key"
)
```

### Model Training

```python
# Basic training
result = client.train_model(
    data=dataframe,  # DataFrame or file path
    target_column='target',
    model_type='classification'  # or 'regression'
)

# Advanced training
result = client.train_model(
    data="data.csv",
    target_column='target',
    model_type='classification',
    model_name='custom_model_name',
    task_type='CLASSIFICATION',  # Optional, inferred from model_type if not provided
    test_size=0.2,
    optimization_strategy='BAYESIAN_OPTIMIZATION',
    optimization_iterations=50,
    feature_selection=True,
    cv_folds=5,
    random_state=42,
    optimization_mode='PERFORMANCE'
)
```

### Model Prediction

```python
# Predict with DataFrame
predictions = client.predict(
    model='my_model',
    data=test_df
)

# Predict with file
predictions = client.predict(
    model='my_model',
    data='test_data.csv'
)

# Predict with feature values list
predictions = client.predict(
    model='my_model',
    data=[[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]
)

# Get probabilities (for classification)
predictions = client.predict(
    model='my_model',
    data=test_df,
    return_proba=True
)
```

### Model Analysis

```python
# Get model metrics
metrics = client.get_model_metrics('my_model')

# Compare multiple models
comparison = client.compare_models(
    models=['model1', 'model2', 'model3'],
    metrics=['accuracy', 'f1', 'precision', 'recall'],
    include_plot=True
)

# Analyze feature importance
importance = client.feature_importance(
    model_name='my_model',
    top_n=20,
    include_plot=True
)

# Perform error analysis
analysis = client.error_analysis(
    model_name='my_model',
    test_data=test_df,  # Optional
    n_samples=100,
    include_plots=True
)

# Detect data drift
drift = client.detect_drift(
    model_name='my_model',
    new_data=new_df,
    reference_dataset=None,  # Uses training data as reference if None
    drift_threshold=0.1
)
```

### Model Management

```python
# List all models
models = client.list_models()

# Get model info
info = client.get_model_info('my_model')

# Quantize model for efficient deployment
quantized = client.quantize_model(
    model_name='my_model',
    quantization_type='int8',
    quantization_mode='dynamic_per_batch'
)

# Export model to different formats
client.export_model(
    model_name='my_model',
    format='onnx',  # 'sklearn', 'onnx', 'pmml', 'tf', 'torchscript'
    include_pipeline=True,
    output_path='exported_model.onnx'
)

# Delete model
client.delete_model('my_model')
```

### Data Processing

```python
# Preprocess data
processed_df = client.preprocess_data(
    data=df,
    normalize='standard',  # 'standard', 'minmax', 'robust', 'none'
    handle_missing=True,
    detect_outliers=True
)

# Save preprocessed data to file
output_path = client.preprocess_data(
    data=df,
    normalize='standard',
    handle_missing=True,
    detect_outliers=True,
    output_path='processed_data.csv'
)
```

## Advanced Usage

### Custom Optimization Strategy

Control the hyperparameter optimization process based on your needs:

```python
# For quick exploration with limited resources
result = client.train_model(
    data=df,
    target_column='target',
    model_type='classification',
    optimization_strategy='RANDOM_SEARCH',
    optimization_iterations=20,
    optimization_mode='MEMORY_SAVING'
)

# For thorough optimization when resources are available
result = client.train_model(
    data=df,
    target_column='target',
    model_type='regression',
    optimization_strategy='BAYESIAN_OPTIMIZATION',
    optimization_iterations=100,
    optimization_mode='FULL_UTILIZATION'
)
```

### Performance Tuning

```python
# Get current configuration
config = client.get_automl_config()
print(f"Current config: {config}")

# Change optimization mode
config = client.get_automl_config(optimization_mode='PERFORMANCE')
```

### Error Analysis with Test Data

```python
# Perform detailed error analysis with custom test data
error_analysis = client.error_analysis(
    model_name='my_classification_model',
    test_data=test_df_with_ground_truth,
    n_samples=200,
    include_plots=True
)

# Examine most frequent misclassifications
for error_type, details in error_analysis['top_misclassifications'].items():
    print(f"Error: {error_type}, Count: {details['count']}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For support, please open an issue on the GitHub repository or contact support@kolosal.ai.