# Inference Engine API (`modules/api/inference_engine_api.py`)

## Overview

The Inference Engine API provides a high-performance RESTful interface for model inference operations. It supports multiple model types, batch processing, real-time inference, and comprehensive monitoring through HTTP endpoints.

## Features

- **Multi-Framework Support**: PyTorch, TensorFlow, Scikit-learn, XGBoost, LightGBM
- **High-Performance Inference**: Optimized for low-latency and high-throughput
- **Batch Processing**: Dynamic batching with priority queues
- **Model Management**: Load, unload, and manage multiple models
- **Real-time Monitoring**: Performance metrics and health monitoring
- **Security**: API key authentication and input validation
- **Async Processing**: Non-blocking operations with background tasks
- **Quantization Support**: Model quantization for optimization

## API Configuration

```python
# Environment Variables
INFERENCE_API_HOST=0.0.0.0
INFERENCE_API_PORT=8003
INFERENCE_API_DEBUG=False
REQUIRE_API_KEY=False
API_KEYS=key1,key2,key3
MAX_WORKERS=4
MODEL_DIR=./models
ENABLE_QUANTIZATION=True
ENABLE_COMPILATION=True
BATCH_TIMEOUT_MS=100
MAX_BATCH_SIZE=64
```

## Data Models

### ModelLoadRequest
```python
{
    "model_path": "/path/to/model.pkl",       # Path to model file
    "model_type": "sklearn",                  # sklearn, xgboost, lightgbm, ensemble, custom
    "compile_model": true                     # Whether to compile for faster inference
}
```

### InferenceRequest
```python
{
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Input data for inference
    "model_id": "model_v1",                        # Optional model identifier
    "return_probabilities": false,                 # Return class probabilities
    "batch_size": 32,                             # Batch size for processing
    "timeout_ms": 5000,                           # Request timeout
    "priority": "normal"                          # low, normal, high, urgent
}
```

### BatchInferenceRequest
```python
{
    "requests": [
        {
            "request_id": "req_001",
            "data": [[1.0, 2.0, 3.0]],
            "model_id": "model_v1"
        },
        {
            "request_id": "req_002", 
            "data": [[4.0, 5.0, 6.0]],
            "model_id": "model_v2"
        }
    ],
    "batch_timeout_ms": 1000,
    "priority": "high"
}
```

## API Endpoints

### Model Management

#### Load Model
```http
POST /api/inference/models/load
Content-Type: application/json
X-API-Key: your-api-key

{
    "model_path": "./models/my_model.pkl",
    "model_type": "sklearn",
    "compile_model": true
}
```

**Response:**
```json
{
    "message": "Model loaded successfully",
    "model_id": "model_12345",
    "model_type": "sklearn",
    "compiled": true,
    "load_time": 1.234,
    "model_size_mb": 15.6,
    "loaded_at": "2025-01-15T10:30:00Z"
}
```

#### Get Model Info
```http
GET /api/inference/models/{model_id}/info
X-API-Key: your-api-key
```

**Response:**
```json
{
    "model_id": "model_12345",
    "model_type": "sklearn",
    "model_path": "./models/my_model.pkl",
    "compiled": true,
    "loaded_at": "2025-01-15T10:30:00Z",
    "last_used": "2025-01-15T10:35:00Z",
    "inference_count": 150,
    "average_inference_time": 0.045,
    "model_size_mb": 15.6,
    "memory_usage_mb": 20.3
}
```

#### List Models
```http
GET /api/inference/models/list
X-API-Key: your-api-key
```

**Response:**
```json
{
    "models": [
        {
            "model_id": "model_12345",
            "model_type": "sklearn",
            "compiled": true,
            "loaded_at": "2025-01-15T10:30:00Z",
            "inference_count": 150
        }
    ],
    "total_models": 1,
    "total_memory_usage_mb": 20.3
}
```

#### Unload Model
```http
DELETE /api/inference/models/{model_id}
X-API-Key: your-api-key
```

**Response:**
```json
{
    "message": "Model unloaded successfully",
    "model_id": "model_12345",
    "memory_freed_mb": 20.3
}
```

#### Reload Model
```http
POST /api/inference/models/{model_id}/reload
X-API-Key: your-api-key
```

#### Get Model Metrics
```http
GET /api/inference/models/{model_id}/metrics
X-API-Key: your-api-key
```

**Response:**
```json
{
    "model_id": "model_12345",
    "inference_count": 150,
    "average_inference_time": 0.045,
    "total_inference_time": 6.75,
    "last_inference_time": 0.042,
    "error_count": 2,
    "error_rate": 0.013,
    "throughput_per_second": 22.2,
    "memory_usage_mb": 20.3
}
```

### Inference Operations

#### Single Prediction
```http
POST /api/inference/predict
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[1.0, 2.0, 3.0, 4.0, 5.0]],
    "model_id": "model_12345",
    "return_probabilities": false
}
```

**Response:**
```json
{
    "predictions": [0.85],
    "model_id": "model_12345",
    "inference_time": 0.045,
    "request_id": "req_12345",
    "timestamp": "2025-01-15T10:30:00Z"
}
```

#### Batch Prediction
```http
POST /api/inference/predict/batch
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0]
    ],
    "model_id": "model_12345",
    "batch_size": 32,
    "return_probabilities": true
}
```

**Response:**
```json
{
    "predictions": [0.85, 0.72, 0.91],
    "probabilities": [
        [0.15, 0.85],
        [0.28, 0.72], 
        [0.09, 0.91]
    ],
    "model_id": "model_12345",
    "batch_size": 3,
    "total_inference_time": 0.123,
    "average_inference_time": 0.041,
    "request_id": "batch_12345",
    "timestamp": "2025-01-15T10:30:00Z"
}
```

#### Multi-Model Batch Prediction
```http
POST /api/inference/predict/multi-batch
Content-Type: application/json
X-API-Key: your-api-key

{
    "requests": [
        {
            "request_id": "req_001",
            "data": [[1.0, 2.0, 3.0, 4.0, 5.0]],
            "model_id": "model_12345"
        },
        {
            "request_id": "req_002",
            "data": [[6.0, 7.0, 8.0, 9.0, 10.0]],
            "model_id": "model_67890"
        }
    ],
    "batch_timeout_ms": 1000
}
```

**Response:**
```json
{
    "results": [
        {
            "request_id": "req_001",
            "predictions": [0.85],
            "model_id": "model_12345",
            "inference_time": 0.045,
            "status": "success"
        },
        {
            "request_id": "req_002", 
            "predictions": [0.72],
            "model_id": "model_67890",
            "inference_time": 0.038,
            "status": "success"
        }
    ],
    "batch_id": "batch_12345",
    "total_requests": 2,
    "successful_requests": 2,
    "failed_requests": 0,
    "total_time": 0.156
}
```

#### Async Prediction
```http
POST /api/inference/predict/async
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[1.0, 2.0, 3.0, 4.0, 5.0]],
    "model_id": "model_12345",
    "callback_url": "https://your-app.com/callback",
    "priority": "high"
}
```

**Response:**
```json
{
    "task_id": "task_12345",
    "status": "queued",
    "estimated_completion": "2025-01-15T10:30:05Z",
    "queue_position": 3
}
```

#### Get Async Result
```http
GET /api/inference/predict/async/{task_id}
X-API-Key: your-api-key
```

**Response:**
```json
{
    "task_id": "task_12345",
    "status": "completed",
    "result": {
        "predictions": [0.85],
        "inference_time": 0.045
    },
    "completed_at": "2025-01-15T10:30:03Z"
}
```

### Streaming Inference

#### Start Streaming Session
```http
POST /api/inference/stream/start
Content-Type: application/json
X-API-Key: your-api-key

{
    "stream_id": "stream_12345",
    "model_id": "model_12345",
    "buffer_size": 100,
    "batch_timeout_ms": 50
}
```

#### Send Stream Data
```http
POST /api/inference/stream/{stream_id}/send
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[1.0, 2.0, 3.0, 4.0, 5.0]],
    "sequence_id": 1
}
```

#### Get Stream Results
```http
GET /api/inference/stream/{stream_id}/results
X-API-Key: your-api-key
```

#### Close Stream
```http
DELETE /api/inference/stream/{stream_id}
X-API-Key: your-api-key
```

### Engine Management

#### Get Engine Status
```http
GET /api/inference/engine/status
X-API-Key: your-api-key
```

**Response:**
```json
{
    "status": "running",
    "uptime": "2h 30m 45s",
    "loaded_models": 3,
    "active_requests": 5,
    "total_predictions": 1000,
    "average_response_time": 0.045,
    "memory_usage": {
        "total_mb": 1024.0,
        "used_mb": 512.5,
        "free_mb": 511.5
    },
    "cpu_usage": 45.2,
    "gpu_usage": 78.5
}
```

#### Get Engine Configuration
```http
GET /api/inference/engine/config
X-API-Key: your-api-key
```

**Response:**
```json
{
    "max_batch_size": 64,
    "batch_timeout_ms": 100,
    "enable_quantization": true,
    "enable_compilation": true,
    "max_concurrent_requests": 100,
    "model_cache_size": 5,
    "memory_limit_mb": 2048
}
```

#### Update Engine Configuration
```http
PUT /api/inference/engine/config
Content-Type: application/json
X-API-Key: your-api-key

{
    "max_batch_size": 128,
    "batch_timeout_ms": 50,
    "max_concurrent_requests": 200
}
```

#### Restart Engine
```http
POST /api/inference/engine/restart
X-API-Key: your-api-key
```

#### Clear Engine Cache
```http
DELETE /api/inference/engine/cache
X-API-Key: your-api-key
```

### Performance and Monitoring

#### Get Performance Metrics
```http
GET /api/inference/metrics/performance
X-API-Key: your-api-key
```

**Response:**
```json
{
    "request_metrics": {
        "total_requests": 1000,
        "successful_requests": 995,
        "failed_requests": 5,
        "average_response_time": 0.045,
        "p95_response_time": 0.120,
        "p99_response_time": 0.200,
        "requests_per_second": 22.5
    },
    "resource_metrics": {
        "cpu_usage": 45.2,
        "memory_usage_mb": 512.5,
        "gpu_usage": 78.5,
        "gpu_memory_mb": 2048.0
    },
    "model_metrics": {
        "active_models": 3,
        "total_model_memory_mb": 256.8,
        "cache_hit_rate": 0.85
    }
}
```

#### Get System Health
```http
GET /api/inference/health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-01-15T10:30:00Z",
    "checks": {
        "engine_status": "healthy",
        "memory_usage": "healthy",
        "model_loading": "healthy",
        "inference_performance": "healthy"
    },
    "uptime": "2h 30m 45s",
    "version": "1.0.0"
}
```

#### Get Resource Usage
```http
GET /api/inference/metrics/resources
X-API-Key: your-api-key
```

**Response:**
```json
{
    "cpu": {
        "usage_percent": 45.2,
        "cores": 8,
        "load_average": [1.2, 1.5, 1.8]
    },
    "memory": {
        "total_mb": 16384,
        "used_mb": 8192,
        "free_mb": 8192,
        "usage_percent": 50.0
    },
    "gpu": {
        "count": 1,
        "usage_percent": 78.5,
        "memory_total_mb": 8192,
        "memory_used_mb": 6144,
        "temperature": 65
    },
    "disk": {
        "usage_percent": 75.0,
        "free_gb": 100.5
    }
}
```

### Quantization and Optimization

#### Quantize Model
```http
POST /api/inference/models/{model_id}/quantize
Content-Type: application/json
X-API-Key: your-api-key

{
    "quantization_type": "INT8",
    "calibration_data": [[...]],  # Optional calibration data
    "quantization_mode": "DYNAMIC"
}
```

**Response:**
```json
{
    "message": "Model quantized successfully",
    "model_id": "model_12345",
    "quantization_type": "INT8",
    "size_reduction_percent": 75.0,
    "speed_improvement_percent": 40.0,
    "accuracy_loss_percent": 2.1
}
```

#### Compile Model
```http
POST /api/inference/models/{model_id}/compile
X-API-Key: your-api-key
```

#### Optimize Model
```http
POST /api/inference/models/{model_id}/optimize
Content-Type: application/json
X-API-Key: your-api-key

{
    "optimization_level": "aggressive",  # conservative, balanced, aggressive
    "target_latency_ms": 10,
    "target_throughput": 100
}
```

## Usage Examples

### Basic Inference Workflow

```python
import requests
import json

# API configuration
API_BASE = "http://localhost:8003"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# 1. Load model
model_config = {
    "model_path": "./models/my_model.pkl",
    "model_type": "sklearn",
    "compile_model": True
}
response = requests.post(f"{API_BASE}/api/inference/models/load",
                        headers=headers, json=model_config)
model_id = response.json()["model_id"]

# 2. Make prediction
prediction_data = {
    "data": [[1.0, 2.0, 3.0, 4.0, 5.0]],
    "model_id": model_id,
    "return_probabilities": True
}
response = requests.post(f"{API_BASE}/api/inference/predict",
                        headers=headers, json=prediction_data)
result = response.json()

print("Prediction:", result["predictions"])
print("Probabilities:", result["probabilities"])
```

### Batch Processing

```python
# Batch prediction
batch_data = {
    "data": [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0]
    ],
    "model_id": model_id,
    "batch_size": 32
}
response = requests.post(f"{API_BASE}/api/inference/predict/batch",
                        headers=headers, json=batch_data)
results = response.json()

print("Batch predictions:", results["predictions"])
print("Batch processing time:", results["total_inference_time"])
```

### Multi-Model Inference

```python
# Load multiple models
models = []
for i, model_path in enumerate(["model1.pkl", "model2.pkl"]):
    response = requests.post(f"{API_BASE}/api/inference/models/load",
                           headers=headers, 
                           json={"model_path": model_path, "model_type": "sklearn"})
    models.append(response.json()["model_id"])

# Multi-model batch prediction
multi_batch_data = {
    "requests": [
        {
            "request_id": "req_001",
            "data": [[1.0, 2.0, 3.0, 4.0, 5.0]],
            "model_id": models[0]
        },
        {
            "request_id": "req_002",
            "data": [[6.0, 7.0, 8.0, 9.0, 10.0]],
            "model_id": models[1]
        }
    ]
}
response = requests.post(f"{API_BASE}/api/inference/predict/multi-batch",
                        headers=headers, json=multi_batch_data)
results = response.json()
```

### Async Processing

```python
import time

# Submit async prediction
async_data = {
    "data": [[1.0, 2.0, 3.0, 4.0, 5.0]],
    "model_id": model_id,
    "priority": "high"
}
response = requests.post(f"{API_BASE}/api/inference/predict/async",
                        headers=headers, json=async_data)
task_id = response.json()["task_id"]

# Poll for results
while True:
    response = requests.get(f"{API_BASE}/api/inference/predict/async/{task_id}",
                           headers={"X-API-Key": API_KEY})
    status = response.json()["status"]
    
    if status == "completed":
        result = response.json()["result"]
        print("Async prediction:", result["predictions"])
        break
    elif status == "failed":
        print("Prediction failed")
        break
    
    time.sleep(0.1)
```

### Model Optimization

```python
# Quantize model for better performance
quantization_config = {
    "quantization_type": "INT8",
    "quantization_mode": "DYNAMIC"
}
response = requests.post(f"{API_BASE}/api/inference/models/{model_id}/quantize",
                        headers=headers, json=quantization_config)
print("Quantization result:", response.json())

# Compile model for faster inference
response = requests.post(f"{API_BASE}/api/inference/models/{model_id}/compile",
                        headers={"X-API-Key": API_KEY})
print("Compilation result:", response.json())
```

### Monitoring and Metrics

```python
# Get performance metrics
response = requests.get(f"{API_BASE}/api/inference/metrics/performance",
                       headers={"X-API-Key": API_KEY})
metrics = response.json()

print("Average response time:", metrics["request_metrics"]["average_response_time"])
print("Requests per second:", metrics["request_metrics"]["requests_per_second"])
print("CPU usage:", metrics["resource_metrics"]["cpu_usage"])

# Get model-specific metrics
response = requests.get(f"{API_BASE}/api/inference/models/{model_id}/metrics",
                       headers={"X-API-Key": API_KEY})
model_metrics = response.json()

print("Model inference count:", model_metrics["inference_count"])
print("Model average time:", model_metrics["average_inference_time"])
```

## Error Handling

### Common Error Codes

- **400 Bad Request**: Invalid input data or parameters
- **401 Unauthorized**: Missing or invalid API key
- **404 Not Found**: Model not found or endpoint not found
- **409 Conflict**: Model already loaded or resource conflict
- **422 Unprocessable Entity**: Model loading or inference errors
- **500 Internal Server Error**: Internal engine errors
- **503 Service Unavailable**: Engine not initialized or overloaded

### Error Response Format

```json
{
    "error": "ModelNotFoundError",
    "message": "Model with ID 'model_12345' not found",
    "details": {
        "model_id": "model_12345",
        "available_models": ["model_67890"]
    },
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_12345"
}
```

## Best Practices

### Performance Optimization

1. **Model Compilation**: Enable compilation for frequently used models
2. **Quantization**: Use quantization for better performance with minimal accuracy loss
3. **Batch Processing**: Use batch prediction for multiple samples
4. **Async Processing**: Use async for non-time-critical predictions
5. **Caching**: Keep frequently used models loaded

### Resource Management

1. **Memory Monitoring**: Monitor memory usage regularly
2. **Model Lifecycle**: Unload unused models to free memory
3. **Batch Size Tuning**: Optimize batch size based on available resources
4. **Request Limits**: Set appropriate timeout and concurrency limits

### Security

1. **API Keys**: Always use API keys in production
2. **Input Validation**: Validate input data before inference
3. **Rate Limiting**: Implement rate limiting for production use
4. **Secure Endpoints**: Use HTTPS in production

### Monitoring

1. **Health Checks**: Implement regular health checks
2. **Performance Metrics**: Monitor response times and throughput
3. **Error Tracking**: Track and analyze error patterns
4. **Resource Usage**: Monitor CPU, memory, and GPU usage

## Advanced Features

### Custom Model Types

Support for custom model implementations:

```python
class CustomModel:
    def predict(self, X):
        # Custom prediction logic
        return predictions
    
    def predict_proba(self, X):
        # Custom probability prediction
        return probabilities
```

### Model Ensembles

Load and use ensemble models:

```python
ensemble_config = {
    "model_path": "./models/ensemble_model.pkl",
    "model_type": "ensemble",
    "ensemble_method": "voting"  # voting, stacking, bagging
}
```

### A/B Testing

Support for A/B testing with multiple model versions:

```python
ab_test_config = {
    "requests": [
        {"data": data, "model_id": "model_v1", "weight": 0.5},
        {"data": data, "model_id": "model_v2", "weight": 0.5}
    ]
}
```

## Related Documentation

- [Inference Engine](../engine/inference_engine.md) - Core inference engine
- [Quantizer](../engine/quantizer.md) - Model quantization
- [Batch Processor API](batch_processor_api.md) - Batch processing API
- [Model Manager API](model_manager_api.md) - Model management API
- [Performance Metrics](../engine/performance_metrics.md) - Performance monitoring

---

*The Inference Engine API provides a high-performance, scalable interface for ML model inference with enterprise-grade features including monitoring, optimization, and security.*
