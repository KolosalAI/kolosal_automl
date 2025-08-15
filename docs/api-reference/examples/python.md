# 🐍 Python API Examples

Complete examples for integrating Kolosal AutoML with Python applications.

## 📚 Table of Contents

1. [🔧 Setup & Authentication](#-setup--authentication)
2. [🚂 Training Models](#-training-models)
3. [⚡ Making Predictions](#-making-predictions)
4. [📦 Batch Processing](#-batch-processing)
5. [🗄️ Model Management](#️-model-management)
6. [🔄 Data Processing](#-data-processing)
7. [📊 Monitoring & Metrics](#-monitoring--metrics)
8. [🛠️ Advanced Usage](#️-advanced-usage)

## 🔧 Setup & Authentication

### Basic Client Setup

```python
import requests
import json
from typing import Dict, Any, Optional, List
import time

class KolosalClient:
    """Python client for Kolosal AutoML API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
            
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response with error checking"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_data = response.json() if response.content else {"error": {"message": str(e)}}
            raise Exception(f"API Error: {error_data.get('error', {}).get('message', str(e))}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        return self._handle_response(response)

# Initialize client
client = KolosalClient(
    base_url="http://localhost:8000",
    api_key="genta_your_api_key_here"
)

# Test connection
health = client.health_check()
print(f"✅ API Status: {health['status']}")
```

### Environment-Based Configuration

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class KolosalConfig:
    """Configuration management"""
    
    def __init__(self):
        self.api_url = os.getenv("KOLOSAL_API_URL", "http://localhost:8000")
        self.api_key = os.getenv("KOLOSAL_API_KEY")
        self.timeout = int(os.getenv("KOLOSAL_TIMEOUT", "30"))
        
        if not self.api_key:
            raise ValueError("KOLOSAL_API_KEY environment variable is required")

# Usage
config = KolosalConfig()
client = KolosalClient(config.api_url, config.api_key)
```

## 🚂 Training Models

### Simple Model Training

```python
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
import numpy as np

def train_classification_model():
    """Train a classification model with iris dataset"""
    
    # Load sample data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Prepare training request
    training_request = {
        "data": X_train.tolist(),
        "target": y_train.tolist(),
        "task_type": "classification",
        "model_type": "random_forest",
        "optimization_strategy": "bayesian",
        "config": {
            "cv_folds": 5,
            "enable_automl": True,
            "test_size": 0.2,
            "max_iter": 1000
        },
        "metadata": {
            "name": "iris_classifier",
            "description": "Iris flower classification model"
        }
    }
    
    # Start training
    response = client.session.post(
        f"{client.base_url}/api/train-engine/train",
        json=training_request
    )
    
    result = client._handle_response(response)
    job_id = result["job_id"]
    
    print(f"🚂 Training started with job ID: {job_id}")
    return job_id, X_test, y_test

# Train the model
job_id, X_test, y_test = train_classification_model()
```

### Monitor Training Progress

```python
def monitor_training(client: KolosalClient, job_id: str, poll_interval: int = 10):
    """Monitor training progress with real-time updates"""
    
    print(f"📊 Monitoring training job: {job_id}")
    
    while True:
        response = client.session.get(f"{client.base_url}/api/train-engine/status/{job_id}")
        status_data = client._handle_response(response)
        
        status = status_data["status"]
        progress = status_data.get("progress", 0)
        
        if status == "completed":
            print("✅ Training completed successfully!")
            break
        elif status == "failed":
            error_msg = status_data.get("error", "Unknown error")
            print(f"❌ Training failed: {error_msg}")
            return None
        else:
            current_step = status_data.get("current_step", "unknown")
            best_score = status_data.get("best_score", "N/A")
            remaining_time = status_data.get("estimated_remaining", "unknown")
            
            print(f"⏳ Progress: {progress}% | Step: {current_step} | Best Score: {best_score} | ETA: {remaining_time}")
            
        time.sleep(poll_interval)
    
    # Get final results
    response = client.session.get(f"{client.base_url}/api/train-engine/results/{job_id}")
    return client._handle_response(response)

# Monitor training
results = monitor_training(client, job_id)

if results:
    print(f"🎯 Final Accuracy: {results['performance_metrics']['accuracy']:.3f}")
    print(f"🏆 Best Model: {results['best_model']}")
    model_id = results["model_id"]
```

### Upload Dataset for Training

```python
import pandas as pd
from pathlib import Path

def upload_and_train_from_csv(client: KolosalClient, csv_path: str, target_column: str):
    """Upload CSV file and train model"""
    
    # Upload dataset
    with open(csv_path, 'rb') as file:
        files = {"file": file}
        data = {
            "target_column": target_column,
            "task_type": "classification",  # or "regression"
            "description": f"Dataset from {Path(csv_path).name}"
        }
        
        response = client.session.post(
            f"{client.base_url}/api/train-engine/upload-dataset",
            files=files,
            data=data
        )
    
    dataset_info = client._handle_response(response)
    dataset_id = dataset_info["dataset_id"]
    
    print(f"📊 Dataset uploaded: {dataset_info['rows']} rows, {dataset_info['columns']} columns")
    
    # Start training with uploaded dataset
    training_request = {
        "dataset_id": dataset_id,
        "task_type": dataset_info["task_type"],
        "optimization_strategy": "bayesian",
        "config": {
            "cv_folds": 5,
            "enable_automl": True,
            "algorithms": ["random_forest", "xgboost", "lightgbm"]
        }
    }
    
    response = client.session.post(
        f"{client.base_url}/api/train-engine/train",
        json=training_request
    )
    
    return client._handle_response(response)

# Example usage
# result = upload_and_train_from_csv(client, "data/my_dataset.csv", "target")
```

### Advanced Training Configuration

```python
def train_with_advanced_config():
    """Train model with comprehensive configuration"""
    
    training_config = {
        "data": X_train.tolist(),
        "target": y_train.tolist(),
        "task_type": "classification",
        "optimization_strategy": "asht",  # Kolosal's proprietary optimizer
        "config": {
            # Cross-validation settings
            "cv_folds": 10,
            "cv_strategy": "stratified",
            
            # Optimization settings
            "max_trials": 100,
            "optimization_timeout": 3600,  # 1 hour
            "early_stopping_rounds": 10,
            
            # Performance settings
            "enable_automl": True,
            "enable_jit_compilation": True,
            "enable_mixed_precision": True,
            "memory_optimization": True,
            "n_jobs": -1,  # Use all CPU cores
            
            # Data preprocessing
            "normalize_features": True,
            "handle_outliers": True,
            "feature_selection": True,
            "feature_selection_k": 10,
            
            # Model ensemble
            "enable_ensemble": True,
            "ensemble_methods": ["voting", "stacking"],
            
            # Algorithms to try
            "algorithms": [
                "random_forest",
                "xgboost", 
                "lightgbm",
                "catboost",
                "neural_network"
            ]
        },
        "metadata": {
            "name": "advanced_classifier",
            "description": "Advanced classification with full optimization",
            "tags": ["production", "optimized", "ensemble"]
        }
    }
    
    response = client.session.post(
        f"{client.base_url}/api/train-engine/train",
        json=training_config
    )
    
    return client._handle_response(response)

# Advanced training
# advanced_result = train_with_advanced_config()
```

## ⚡ Making Predictions

### Single Prediction

```python
def make_single_prediction(client: KolosalClient, model_id: str, features: List[float]):
    """Make a single prediction"""
    
    prediction_request = {
        "data": [features],
        "return_probabilities": True,
        "return_confidence": True,
        "return_feature_importance": True
    }
    
    response = client.session.post(
        f"{client.base_url}/api/inference/predict/{model_id}",
        json=prediction_request
    )
    
    result = client._handle_response(response)
    
    print(f"🔮 Prediction: {result['predictions'][0]}")
    print(f"📊 Probabilities: {result['probabilities'][0]}")
    print(f"🎯 Confidence: {result['confidence_scores'][0]:.3f}")
    
    return result

# Example prediction
sample_features = [5.1, 3.5, 1.4, 0.2]  # Iris sample
prediction = make_single_prediction(client, model_id, sample_features)
```

### Batch Predictions

```python
def make_batch_predictions(client: KolosalClient, model_id: str, features_batch: List[List[float]]):
    """Make predictions on multiple samples efficiently"""
    
    batch_request = {
        "data": features_batch,
        "batch_size": 32,
        "return_probabilities": True,
        "priority": "high",
        "enable_parallel": True
    }
    
    response = client.session.post(
        f"{client.base_url}/api/inference/predict-batch/{model_id}",
        json=batch_request
    )
    
    result = client._handle_response(response)
    
    print(f"📦 Processed {len(features_batch)} samples")
    print(f"⚡ Processing time: {result['batch_stats']['processing_time_ms']:.2f}ms")
    print(f"🚀 Throughput: {result['batch_stats']['throughput_per_second']:.0f} samples/sec")
    
    return result

# Batch prediction example
batch_features = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2]
]

batch_results = make_batch_predictions(client, model_id, batch_features)
```

### Asynchronous Predictions

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncPredictionClient:
    """Async prediction client for high-throughput scenarios"""
    
    def __init__(self, client: KolosalClient):
        self.client = client
        
    def submit_async_prediction(self, model_id: str, data: List[List[float]], 
                              callback_url: Optional[str] = None):
        """Submit prediction job for async processing"""
        
        async_request = {
            "data": data,
            "priority": "normal",
            "callback_url": callback_url,
            "metadata": {
                "submitted_at": time.time(),
                "batch_size": len(data)
            }
        }
        
        response = self.client.session.post(
            f"{self.client.base_url}/api/inference/predict-async/{model_id}",
            json=async_request
        )
        
        return self.client._handle_response(response)
    
    def get_async_results(self, job_id: str):
        """Retrieve async prediction results"""
        
        response = self.client.session.get(
            f"{self.client.base_url}/api/inference/results/{job_id}"
        )
        
        return self.client._handle_response(response)
    
    def wait_for_async_results(self, job_id: str, timeout: int = 300, poll_interval: int = 5):
        """Wait for async results with timeout"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.get_async_results(job_id)
            
            if result["status"] == "completed":
                print(f"✅ Async prediction completed in {time.time() - start_time:.1f}s")
                return result
            elif result["status"] == "failed":
                raise Exception(f"Async prediction failed: {result.get('error', 'Unknown error')}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Async prediction timed out after {timeout}s")

# Async prediction example
async_client = AsyncPredictionClient(client)

# Submit async job
large_batch = [[np.random.random(4) for _ in range(4)] for _ in range(1000)]
async_job = async_client.submit_async_prediction(model_id, large_batch)

print(f"🚀 Async job submitted: {async_job['job_id']}")

# Wait for results
async_results = async_client.wait_for_async_results(async_job['job_id'])
print(f"📊 Async results: {len(async_results['predictions'])} predictions")
```

## 📦 Batch Processing

### Large Dataset Processing

```python
def process_large_dataset(client: KolosalClient, dataset_path: str, operation: str = "inference"):
    """Process large datasets with batch operations"""
    
    batch_job_request = {
        "operation": operation,
        "data_source": {
            "type": "file",
            "path": dataset_path,
            "format": "csv"
        },
        "config": {
            "batch_size": 1000,
            "parallel_workers": 4,
            "priority": "high",
            "memory_limit_gb": 8,
            "enable_checkpointing": True
        },
        "output": {
            "format": "parquet",
            "compression": "gzip",
            "path": "output/batch_results.parquet"
        },
        "callback_url": "https://your-app.com/batch-callback"
    }
    
    response = client.session.post(
        f"{client.base_url}/api/batch/submit",
        json=batch_job_request
    )
    
    result = client._handle_response(response)
    return result["batch_job_id"]

def monitor_batch_job(client: KolosalClient, batch_job_id: str):
    """Monitor batch processing job"""
    
    print(f"📊 Monitoring batch job: {batch_job_id}")
    
    while True:
        response = client.session.get(f"{client.base_url}/api/batch/status/{batch_job_id}")
        status_data = client._handle_response(response)
        
        if status_data["status"] in ["completed", "failed"]:
            break
            
        progress = status_data.get("progress", {})
        completed = progress.get("completed_batches", 0)
        total = progress.get("total_batches", 0)
        percentage = progress.get("percentage", 0)
        
        print(f"⏳ Progress: {percentage:.1f}% ({completed}/{total} batches)")
        print(f"🚀 Throughput: {status_data.get('current_throughput', 'N/A')}")
        
        time.sleep(30)  # Check every 30 seconds for batch jobs
    
    return status_data

# Process large dataset
# batch_job_id = process_large_dataset(client, "data/large_dataset.csv", "inference") 
# final_status = monitor_batch_job(client, batch_job_id)
```

## 🗄️ Model Management

### List and Search Models

```python
def list_models(client: KolosalClient, task_type: Optional[str] = None, 
                status: str = "active", limit: int = 50):
    """List available models with filtering"""
    
    params = {
        "limit": limit,
        "status": status
    }
    
    if task_type:
        params["task_type"] = task_type
    
    response = client.session.get(
        f"{client.base_url}/api/models",
        params=params
    )
    
    result = client._handle_response(response)
    
    print(f"📚 Found {result['total']} models:")
    for model in result["models"]:
        accuracy = model.get("accuracy", "N/A")
        print(f"  🤖 {model['name']} ({model['algorithm']}) - Accuracy: {accuracy}")
    
    return result["models"]

def get_model_details(client: KolosalClient, model_id: str):
    """Get comprehensive model information"""
    
    response = client.session.get(f"{client.base_url}/api/models/{model_id}")
    model_info = client._handle_response(response)
    
    print(f"🤖 Model: {model_info['name']}")
    print(f"🔬 Algorithm: {model_info['algorithm']}")
    print(f"📊 Task: {model_info['task_type']}")
    print(f"🎯 Accuracy: {model_info['performance_metrics']['accuracy']:.3f}")
    print(f"📁 Size: {model_info['metadata']['file_size_mb']:.1f} MB")
    
    return model_info

# List models
classification_models = list_models(client, task_type="classification")

# Get detailed info for a specific model
if classification_models:
    model_details = get_model_details(client, classification_models[0]["model_id"])
```

### Model Comparison

```python
def compare_models(client: KolosalClient, model_ids: List[str]):
    """Compare multiple models side by side"""
    
    models_info = []
    for model_id in model_ids:
        model_info = get_model_details(client, model_id)
        models_info.append(model_info)
    
    # Create comparison table
    print("\n📊 Model Comparison:")
    print(f"{'Model Name':<20} {'Algorithm':<20} {'Accuracy':<10} {'Size (MB)':<10}")
    print("-" * 70)
    
    for model in models_info:
        name = model['name'][:19]
        algorithm = model['algorithm'][:19] 
        accuracy = f"{model['performance_metrics']['accuracy']:.3f}"
        size = f"{model['metadata']['file_size_mb']:.1f}"
        print(f"{name:<20} {algorithm:<20} {accuracy:<10} {size:<10}")
    
    return models_info

# Compare multiple models
# model_comparison = compare_models(client, ["model_1", "model_2", "model_3"])
```

### Model Deployment and Export

```python
def deploy_model(client: KolosalClient, model_id: str, deployment_config: Dict[str, Any]):
    """Deploy model to production environment"""
    
    deployment_request = {
        "model_id": model_id,
        "deployment_type": "api_endpoint",
        "config": deployment_config,
        "scaling": {
            "min_instances": 1,
            "max_instances": 10,
            "target_cpu_percent": 70
        },
        "monitoring": {
            "enable_logging": True,
            "log_level": "INFO",
            "enable_metrics": True
        }
    }
    
    response = client.session.post(
        f"{client.base_url}/api/models/{model_id}/deploy",
        json=deployment_request
    )
    
    return client._handle_response(response)

def export_model(client: KolosalClient, model_id: str, format: str = "pickle"):
    """Export model in various formats"""
    
    export_request = {
        "format": format,  # "pickle", "onnx", "pmml", "joblib"
        "include_preprocessor": True,
        "optimize_for_inference": True
    }
    
    response = client.session.post(
        f"{client.base_url}/api/models/{model_id}/export",
        json=export_request
    )
    
    result = client._handle_response(response)
    
    # Download the exported model
    download_response = client.session.get(result["download_url"])
    
    filename = f"exported_model_{model_id}.{format}"
    with open(filename, 'wb') as f:
        f.write(download_response.content)
    
    print(f"📦 Model exported to {filename}")
    return filename

# Deploy model
# deployment_config = {"endpoint_name": "iris_classifier_v1"}
# deployment_result = deploy_model(client, model_id, deployment_config)

# Export model
# exported_file = export_model(client, model_id, "onnx")
```

## 🔄 Data Processing

### Data Preprocessing Pipeline

```python
def preprocess_data(client: KolosalClient, raw_data: List[List], config: Dict[str, Any]):
    """Preprocess data using Kolosal's preprocessing pipeline"""
    
    preprocessing_request = {
        "data": raw_data,
        "config": {
            "handle_missing": config.get("handle_missing", "impute"),
            "missing_strategy": "median",
            "normalize": config.get("normalize", "standard"),
            "remove_outliers": config.get("remove_outliers", True),
            "outlier_method": "isolation_forest",
            "feature_selection": config.get("feature_selection", False),
            "feature_selection_k": config.get("feature_selection_k", 10),
            "encode_categorical": True,
            "handle_imbalanced": False
        },
        "return_transformer": True
    }
    
    response = client.session.post(
        f"{client.base_url}/api/data-processor/preprocess",
        json=preprocessing_request
    )
    
    result = client._handle_response(response)
    
    print(f"🔄 Preprocessing completed:")
    print(f"  📊 Original shape: {len(raw_data)}x{len(raw_data[0]) if raw_data else 0}")
    print(f"  ✨ Processed shape: {len(result['processed_data'])}x{len(result['processed_data'][0]) if result['processed_data'] else 0}")
    print(f"  🛠️ Transformations: {', '.join(result['transformations_applied'])}")
    
    return result

def validate_data_quality(client: KolosalClient, data: List[List], schema: Dict[str, Any]):
    """Validate data quality and get recommendations"""
    
    validation_request = {
        "data": data,
        "schema": schema,
        "checks": [
            "missing_values",
            "data_types",
            "outliers",
            "duplicates",
            "consistency"
        ]
    }
    
    response = client.session.post(
        f"{client.base_url}/api/data-processor/validate",
        json=validation_request
    )
    
    result = client._handle_response(response)
    
    print(f"✅ Data Quality Score: {result['data_quality_score']:.2f}")
    
    if result['recommendations']:
        print("💡 Recommendations:")
        for rec in result['recommendations']:
            print(f"  - {rec}")
    
    return result

# Example preprocessing
sample_data = [
    [1.0, 2.0, None, 4.0],
    [2.0, None, 3.0, 5.0],
    [3.0, 4.0, 5.0, None]
]

preprocessing_config = {
    "handle_missing": "impute",
    "normalize": "standard",
    "remove_outliers": True
}

# processed_result = preprocess_data(client, sample_data, preprocessing_config)
```

## 📊 Monitoring & Metrics

### System Monitoring

```python
def get_system_metrics(client: KolosalClient):
    """Get comprehensive system metrics"""
    
    response = client.session.get(f"{client.base_url}/api/metrics")
    metrics = client._handle_response(response)
    
    print("📊 System Metrics:")
    print(f"  🌐 API Requests/sec: {metrics['api_metrics']['requests_per_second']:.1f}")
    print(f"  ⚡ Avg Response Time: {metrics['api_metrics']['average_response_time_ms']:.1f}ms")
    print(f"  🎯 Error Rate: {metrics['api_metrics']['error_rate_percent']:.2f}%")
    print(f"  🧠 Models Trained: {metrics['ml_metrics']['models_trained']}")
    print(f"  🔮 Predictions Made: {metrics['ml_metrics']['predictions_made']:,}")
    print(f"  💻 CPU Usage: {metrics['system_metrics']['cpu_usage']:.1f}%")
    print(f"  💾 Memory Usage: {metrics['system_metrics']['memory_usage_mb']} MB")
    
    return metrics

def monitor_model_performance(client: KolosalClient, model_id: str, time_range: str = "24h"):
    """Monitor specific model performance"""
    
    params = {
        "time_range": time_range,
        "metrics": ["accuracy", "latency", "throughput", "errors"]
    }
    
    response = client.session.get(
        f"{client.base_url}/api/models/{model_id}/metrics",
        params=params
    )
    
    metrics = client._handle_response(response)
    
    print(f"📈 Model Performance ({time_range}):")
    print(f"  🎯 Predictions: {metrics['prediction_count']:,}")
    print(f"  ⚡ Avg Latency: {metrics['average_latency_ms']:.1f}ms")
    print(f"  🚀 Throughput: {metrics['throughput_per_second']:.1f}/sec")
    print(f"  ❌ Error Rate: {metrics['error_rate_percent']:.2f}%")
    
    return metrics

# Get system metrics
system_metrics = get_system_metrics(client)

# Monitor specific model
# model_metrics = monitor_model_performance(client, model_id, "7d")
```

## 🛠️ Advanced Usage

### Custom Model Pipeline

```python
class MLPipeline:
    """Advanced ML pipeline with Kolosal AutoML"""
    
    def __init__(self, client: KolosalClient):
        self.client = client
        self.models = {}
        self.preprocessing_pipelines = {}
        
    def create_experiment(self, name: str, description: str):
        """Create ML experiment for tracking multiple models"""
        
        experiment_request = {
            "name": name,
            "description": description,
            "tags": ["pipeline", "experiment"],
            "config": {
                "track_metrics": True,
                "enable_comparison": True,
                "auto_versioning": True
            }
        }
        
        response = self.client.session.post(
            f"{self.client.base_url}/api/experiments",
            json=experiment_request
        )
        
        result = self.client._handle_response(response)
        self.experiment_id = result["experiment_id"]
        return result
    
    def train_ensemble(self, X_train, y_train, algorithms: List[str]):
        """Train ensemble of models"""
        
        ensemble_request = {
            "data": X_train,
            "target": y_train,
            "algorithms": algorithms,
            "ensemble_method": "stacking",
            "base_models_config": {
                "cv_folds": 5,
                "optimization_strategy": "bayesian"
            },
            "meta_model": "logistic_regression",
            "experiment_id": getattr(self, 'experiment_id', None)
        }
        
        response = self.client.session.post(
            f"{self.client.base_url}/api/train-engine/ensemble",
            json=ensemble_request
        )
        
        return self.client._handle_response(response)
    
    def auto_feature_engineering(self, data: List[List], config: Dict[str, Any]):
        """Automatic feature engineering"""
        
        feature_eng_request = {
            "data": data,
            "config": {
                "polynomial_features": config.get("polynomial_features", True),
                "interaction_features": config.get("interaction_features", True),
                "feature_selection": True,
                "dimensionality_reduction": config.get("pca", False),
                "text_features": config.get("text_features", False),
                "datetime_features": config.get("datetime_features", False)
            }
        }
        
        response = self.client.session.post(
            f"{self.client.base_url}/api/feature-engineering/auto",
            json=feature_eng_request
        )
        
        return self.client._handle_response(response)

# Advanced pipeline usage
pipeline = MLPipeline(client)

# Create experiment
# experiment = pipeline.create_experiment(
#     name="Advanced Classification Pipeline",
#     description="Ensemble model with feature engineering"
# )

# Train ensemble
# ensemble_result = pipeline.train_ensemble(
#     X_train, y_train, 
#     ["random_forest", "xgboost", "lightgbm", "neural_network"]
# )
```

### Real-time Prediction Streaming

```python
import asyncio
import websockets
import json

class RealTimePredictionStreamer:
    """Stream real-time predictions using WebSocket"""
    
    def __init__(self, client: KolosalClient, model_id: str):
        self.client = client
        self.model_id = model_id
        self.ws_url = client.base_url.replace("http", "ws") + f"/ws/predict/{model_id}"
        
    async def stream_predictions(self, data_stream):
        """Stream predictions in real-time"""
        
        headers = {"X-API-Key": self.client.session.headers["X-API-Key"]}
        
        async with websockets.connect(self.ws_url, extra_headers=headers) as websocket:
            print("🌊 Connected to prediction stream")
            
            for data_point in data_stream:
                # Send data
                request = {
                    "data": data_point,
                    "timestamp": time.time()
                }
                
                await websocket.send(json.dumps(request))
                
                # Receive prediction
                response = await websocket.recv()
                prediction_result = json.loads(response)
                
                print(f"📡 Prediction: {prediction_result['prediction']} "
                      f"(confidence: {prediction_result['confidence']:.3f})")
                
                # Simulate real-time data
                await asyncio.sleep(0.1)

# Real-time streaming example
# streamer = RealTimePredictionStreamer(client, model_id)
# data_stream = [[np.random.random(4) for _ in range(4)] for _ in range(100)]
# asyncio.run(streamer.stream_predictions(data_stream))
```

### A/B Testing Framework

```python
class ModelABTesting:
    """A/B testing framework for model comparison"""
    
    def __init__(self, client: KolosalClient):
        self.client = client
        
    def setup_ab_test(self, model_a_id: str, model_b_id: str, 
                      traffic_split: float = 0.5, duration_hours: int = 24):
        """Setup A/B test between two models"""
        
        ab_test_config = {
            "name": f"AB_Test_{model_a_id}_vs_{model_b_id}",
            "model_a": model_a_id,
            "model_b": model_b_id,
            "traffic_split": traffic_split,
            "duration_hours": duration_hours,
            "metrics_to_track": [
                "accuracy", "latency", "throughput", 
                "user_satisfaction", "business_metrics"
            ],
            "auto_winner_selection": True,
            "confidence_threshold": 0.95
        }
        
        response = self.client.session.post(
            f"{self.client.base_url}/api/ab-testing/setup",
            json=ab_test_config
        )
        
        return self.client._handle_response(response)
    
    def get_ab_test_results(self, test_id: str):
        """Get A/B test results and statistical significance"""
        
        response = self.client.session.get(
            f"{self.client.base_url}/api/ab-testing/{test_id}/results"
        )
        
        results = self.client._handle_response(response)
        
        print(f"🧪 A/B Test Results:")
        print(f"  📊 Model A Performance: {results['model_a_metrics']}")
        print(f"  📊 Model B Performance: {results['model_b_metrics']}")
        print(f"  🏆 Winner: {results['winner']}")
        print(f"  📈 Confidence: {results['confidence']:.2%}")
        print(f"  📊 Statistical Significance: {results['statistically_significant']}")
        
        return results

# A/B testing example
# ab_tester = ModelABTesting(client)
# test_setup = ab_tester.setup_ab_test("model_v1", "model_v2", 0.5, 48)
# ab_results = ab_tester.get_ab_test_results(test_setup["test_id"])
```

## 🎉 Complete Example: End-to-End ML Workflow

```python
def complete_ml_workflow():
    """Complete end-to-end machine learning workflow"""
    
    print("🚀 Starting Complete ML Workflow")
    
    # 1. Load and preprocess data
    from sklearn.datasets import load_wine
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    
    # 2. Data preprocessing
    preprocessing_config = {
        "normalize": "standard",
        "remove_outliers": True,
        "feature_selection": True
    }
    
    preprocessing_result = preprocess_data(client, X_train.tolist(), preprocessing_config)
    X_train_processed = preprocessing_result["processed_data"]
    
    # 3. Train multiple models
    training_request = {
        "data": X_train_processed,
        "target": y_train.tolist(),
        "task_type": "classification",
        "optimization_strategy": "bayesian",
        "config": {
            "cv_folds": 5,
            "enable_automl": True,
            "algorithms": ["random_forest", "xgboost", "lightgbm"],
            "enable_ensemble": True
        }
    }
    
    training_response = client.session.post(
        f"{client.base_url}/api/train-engine/train",
        json=training_request
    )
    
    training_result = client._handle_response(training_response)
    job_id = training_result["job_id"]
    
    # 4. Monitor training
    final_results = monitor_training(client, job_id, poll_interval=10)
    
    if not final_results:
        print("❌ Training failed")
        return
    
    model_id = final_results["model_id"]
    print(f"✅ Training completed! Model ID: {model_id}")
    
    # 5. Make predictions on test set
    test_predictions = make_batch_predictions(client, model_id, X_test.tolist())
    
    # 6. Evaluate performance
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = test_predictions["predictions"]
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Test Accuracy: {accuracy:.3f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))
    
    # 7. Model management
    model_details = get_model_details(client, model_id)
    print(f"📦 Model Size: {model_details['metadata']['file_size_mb']:.1f} MB")
    
    # 8. Export model
    exported_file = export_model(client, model_id, "pickle")
    print(f"💾 Model exported to: {exported_file}")
    
    print("🎉 Workflow completed successfully!")
    return model_id, accuracy

# Run complete workflow
# model_id, accuracy = complete_ml_workflow()
```

---

## 🎯 Next Steps

This comprehensive guide covers all aspects of Python integration with Kolosal AutoML. For more information:

- 🌐 **[JavaScript Examples](javascript.md)** - Browser and Node.js integration
- ⚡ **[cURL Examples](curl.md)** - Command-line interface examples  
- 📚 **[API Reference](../README.md)** - Complete API documentation
- 🚀 **[User Guides](../../user-guides/)** - Task-oriented tutorials

Happy coding with Kolosal AutoML! 🚀

*Python Examples v1.0 | Last updated: January 2025*
