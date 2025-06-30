# Inference Engine API Documentation

## Overview
The Inference Engine API is a high-performance RESTful API for machine learning model inference. It provides endpoints for model loading, inference (both synchronous and asynchronous), batch processing, and performance monitoring. The API is designed for production-grade deployment of machine learning models with features like dynamic batching, request deduplication, and performance optimization.

## Prerequisites
- Python ≥3.8
- Required packages:
  - FastAPI
  - Uvicorn
  - NumPy
  - Pydantic
  - InferenceEngine (custom package)

## Installation
```bash
# Clone the repository
git clone https://github.com/yourorg/inference-engine-api.git

# Install dependencies
pip install -r requirements.txt

# Start the server
python inference_api.py
```

## Configuration
The API can be configured using environment variables:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL_DIR` | `./models` | Default directory for model files |
| `API_KEYS` | `` | Comma-separated list of valid API keys |
| `MAX_WORKERS` | `4` | Maximum number of worker threads |
| `ENABLE_ASYNC` | `1` | Enable/disable asynchronous processing (0/1) |
| `API_HOST` | `0.0.0.0` | Host to bind the server |
| `API_PORT` | `8000` | Port to bind the server |
| `API_DEBUG` | `0` | Enable/disable debug mode (0/1) |
| `JOB_TTL_SECONDS` | `3600` | Time-to-live for async jobs in seconds |
| `REQUIRE_API_KEY` | `0` | Require API key for authentication (0/1) |
| `ENGINE_ENABLE_BATCHING` | `1` | Enable batch processing (0/1) |
| `ENGINE_MAX_BATCH_SIZE` | `64` | Maximum batch size |
| `ENGINE_BATCH_TIMEOUT` | `0.01` | Batch collection timeout in seconds |
| `ENGINE_MAX_CONCURRENT_REQUESTS` | `100` | Maximum concurrent requests |
| `ENGINE_ENABLE_CACHE` | `1` | Enable request deduplication cache (0/1) |
| `ENGINE_MAX_CACHE_ENTRIES` | `1000` | Maximum cache entries |
| `ENGINE_CACHE_TTL_SECONDS` | `3600` | Cache time-to-live in seconds |
| `ENGINE_ENABLE_QUANTIZATION` | `0` | Enable model quantization (0/1) |
| `ENGINE_NUM_THREADS` | `4` | Number of inference threads |
| `ENGINE_ENABLE_THROTTLING` | `0` | Enable request throttling (0/1) |
| `DEFAULT_MODEL_PATH` | `` | Path to auto-load model on startup |
| `DEFAULT_MODEL_TYPE` | `` | Type of model to auto-load |
| `DEFAULT_COMPILE_MODEL` | `0` | Compile auto-loaded model (0/1) |

## Authentication
If `REQUIRE_API_KEY` is set to `1`, all protected endpoints require an API key to be provided in the `X-API-Key` header.

## API Endpoints

### Core Endpoints

#### GET `/`
Returns basic information about the API.

**Response:**
```json
{
  "name": "Inference Engine API",
  "version": "1.0.0",
  "description": "A high-performance API for machine learning model inference",
  "status": "running",
  "docs_url": "/docs"
}
```

#### GET `/health`
Health check endpoint to verify the API and engine status.

**Response:**
```json
{
  "status": "healthy",
  "state": "RUNNING",
  "uptime_seconds": 3600,
  "model_loaded": true,
  "active_requests": 0,
  "model_info": { "model_type": "sklearn", "input_features": 10 },
  "timestamp": "2025-05-11T12:00:00"
}
```

### Model Management

#### POST `/models/load`
Load a model into the inference engine.

**Request Body:**
```json
{
  "model_path": "models/my_model.pkl",
  "model_type": "sklearn",
  "compile_model": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | string | Yes | Path to the model file (relative to `MODEL_DIR` or absolute) |
| `model_type` | string | No | Type of model: "sklearn", "xgboost", "lightgbm", "ensemble", "custom" |
| `compile_model` | boolean | No | Whether to compile model for faster inference |

**Response:**
```json
{
  "success": true,
  "message": "Model loaded successfully",
  "model_path": "/path/to/models/my_model.pkl",
  "model_type": "sklearn",
  "model_info": {
    "input_features": 10,
    "output_shape": [1]
  }
}
```

#### DELETE `/models`
Unload the current model from the engine.

**Response:**
```json
{
  "success": true,
  "message": "Model unloaded successfully"
}
```

#### POST `/validate`
Validate the loaded model to ensure it's functioning correctly.

**Response:**
```json
{
  "success": true,
  "model_type": "sklearn",
  "results": {
    "valid": true,
    "input_features": 10,
    "output_shape": [1]
  },
  "timestamp": "2025-05-11T12:00:00"
}
```

### Inference Endpoints

#### POST `/predict`
Make a prediction using the loaded model.

**Request Body:**
```json
{
  "features": [[0.1, 0.2, 0.3, 0.4]],
  "request_id": "optional-client-request-id"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `features` | array of arrays | Yes | Input features as a 2D array |
| `request_id` | string | No | Optional client-provided request ID |

**Response:**
```json
{
  "request_id": "client-id-or-generated-uuid",
  "success": true,
  "predictions": [[0.95]],
  "error": null,
  "metadata": {
    "inference_time_ms": 5.2
  },
  "timestamp": "2025-05-11T12:00:00"
}
```

#### POST `/predict/batch`
Process a batch of prediction requests.

**Request Body:**
```json
{
  "batch": [[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]],
  "request_ids": ["id1", "id2", "id3"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `batch` | array of 2D arrays | Yes | List of feature arrays for batch processing |
| `request_ids` | array of strings | No | Optional client-provided request IDs |

**Response:**
```json
{
  "batch_id": "generated-batch-uuid",
  "results": [
    {
      "request_id": "id1",
      "success": true,
      "predictions": [[0.1]],
      "error": null,
      "metadata": {},
      "timestamp": "2025-05-11T12:00:00"
    },
    {
      "request_id": "id2",
      "success": true,
      "predictions": [[0.2]],
      "error": null,
      "metadata": {},
      "timestamp": "2025-05-11T12:00:00"
    },
    {
      "request_id": "id3",
      "success": true,
      "predictions": [[0.3]],
      "error": null,
      "metadata": {},
      "timestamp": "2025-05-11T12:00:00"
    }
  ],
  "metadata": {
    "total_time_ms": 15.5,
    "batch_size": 3,
    "success_rate": 1.0,
    "avg_time_per_item_ms": 5.16
  },
  "timestamp": "2025-05-11T12:00:00"
}
```

#### POST `/predict/async`
Submit an asynchronous prediction request.

**Request Body:**
```json
{
  "features": [[0.1, 0.2, 0.3, 0.4]],
  "request_id": "optional-client-request-id",
  "priority": "high",
  "timeout_ms": 5000
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `features` | array of arrays | Yes | Input features as a 2D array |
| `request_id` | string | No | Optional client-provided request ID |
| `priority` | string | No | Processing priority: "high", "normal", "low" (default: "normal") |
| `timeout_ms` | number | No | Optional timeout in milliseconds |

**Response:**
```json
{
  "job_id": "generated-job-uuid",
  "status": "pending",
  "eta_seconds": 0.5,
  "timestamp": "2025-05-11T12:00:00"
}
```

#### GET `/jobs/{job_id}`
Get the status of an asynchronous job.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | path | Yes | ID of the job to retrieve |

**Response (pending):**
```json
{
  "job_id": "job-uuid",
  "status": "pending",
  "eta_seconds": 0.2,
  "timestamp": "2025-05-11T12:00:00"
}
```

**Response (completed):**
```json
{
  "job_id": "job-uuid",
  "status": "completed",
  "result": {
    "request_id": "client-id-or-generated-uuid",
    "success": true,
    "predictions": [[0.95]],
    "metadata": {},
    "timestamp": "2025-05-11T12:00:00"
  },
  "timestamp": "2025-05-11T12:00:00"
}
```

#### GET `/jobs`
List all asynchronous jobs with optional filtering.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No | Maximum number of jobs to return (default: 20, max: 100) |
| `status` | string | No | Filter by job status ("pending", "completed", "failed") |

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job-uuid-1",
      "status": "completed",
      "created_at": "2025-05-11T11:59:30",
      "age_seconds": 30
    },
    {
      "job_id": "job-uuid-2",
      "status": "pending",
      "created_at": "2025-05-11T11:59:45",
      "age_seconds": 15
    }
  ],
  "total_count": 2,
  "timestamp": "2025-05-11T12:00:00"
}
```

### Analysis Endpoints

#### POST `/feature-importance`
Calculate feature importance for a specific input.

**Request Body:**
```json
{
  "features": [[0.1, 0.2, 0.3, 0.4]]
}
```

**Response:**
```json
{
  "success": true,
  "feature_importance": {
    "feature_2": 0.45,
    "feature_1": 0.3,
    "feature_3": 0.15,
    "feature_0": 0.1
  },
  "baseline_prediction": [0.75],
  "timestamp": "2025-05-11T12:00:00"
}
```

### Configuration and Monitoring

#### POST `/config`
Update engine configuration parameters.

**Request Body:**
```json
{
  "enable_batching": true,
  "max_batch_size": 128,
  "batch_timeout": 0.02,
  "enable_cache": true,
  "max_cache_entries": 2000,
  "cache_ttl_seconds": 7200,
  "enable_quantization": false,
  "num_threads": 8,
  "enable_throttling": true,
  "max_concurrent_requests": 50
}
```

**Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "updated_parameters": {
    "max_batch_size": 128,
    "batch_timeout": 0.02
  },
  "current_config": {
    "enable_batching": true,
    "max_batch_size": 128,
    "batch_timeout": 0.02,
    "enable_cache": true,
    "max_cache_entries": 1000,
    "cache_ttl_seconds": 3600,
    "enable_quantization": false,
    "num_threads": 4,
    "enable_throttling": false,
    "max_concurrent_requests": 100
  }
}
```

#### GET `/metrics`
Get engine performance metrics.

**Response:**
```json
{
  "metrics": {
    "avg_inference_time_ms": 5.2,
    "throughput_per_second": 192.3,
    "memory_usage_mb": 256,
    "cache_hit_rate": 0.75,
    "batch_utilization": 0.85,
    "requests_processed": 15000,
    "errors": 12,
    "error_rate": 0.0008
  },
  "timestamp": "2025-05-11T12:00:00"
}
```

#### GET `/cache/stats`
Get statistics about cache usage.

**Response:**
```json
{
  "result_cache": {
    "size": 532,
    "capacity": 1000,
    "hit_count": 2340,
    "miss_count": 780,
    "hit_rate": 0.75,
    "eviction_count": 120
  },
  "feature_cache": {
    "size": 245,
    "capacity": 1000,
    "hit_count": 1200,
    "miss_count": 500,
    "hit_rate": 0.7,
    "eviction_count": 50
  },
  "timestamp": "2025-05-11T12:00:00"
}
```

#### POST `/cache/clear`
Clear the prediction and feature caches.

**Response:**
```json
{
  "success": true,
  "result_cache_cleared": true,
  "feature_cache_cleared": true,
  "message": "Caches cleared successfully"
}
```

#### POST `/restart`
Restart the inference engine with current configuration.

**Response:**
```json
{
  "success": true,
  "message": "Engine restarted successfully",
  "state": "READY"
}
```

## Data Models

### ModelLoadRequest
```python
class ModelLoadRequest(BaseModel):
    model_path: str
    model_type: Optional[str] = None  # "sklearn", "xgboost", "lightgbm", "ensemble", "custom"
    compile_model: Optional[bool] = None
```

### InferenceRequest
```python
class InferenceRequest(BaseModel):
    features: List[List[float]]
    request_id: Optional[str] = None
```

### BatchInferenceRequest
```python
class BatchInferenceRequest(BaseModel):
    batch: List[List[List[float]]]
    request_ids: Optional[List[str]] = None
```

### AsyncInferenceRequest
```python
class AsyncInferenceRequest(InferenceRequest):
    priority: Optional[str] = "normal"  # "high", "normal", "low"
    timeout_ms: Optional[float] = None
```

### EngineConfigRequest
```python
class EngineConfigRequest(BaseModel):
    enable_batching: Optional[bool] = None
    max_batch_size: Optional[int] = None
    batch_timeout: Optional[float] = None
    enable_cache: Optional[bool] = None
    max_cache_entries: Optional[int] = None
    cache_ttl_seconds: Optional[int] = None
    enable_quantization: Optional[bool] = None
    num_threads: Optional[int] = None
    enable_throttling: Optional[bool] = None
    max_concurrent_requests: Optional[int] = None
```

## Security
- Optional API key authentication via `X-API-Key` header
- CORS middleware enabled for all origins
- GZip compression for responses over 1000 bytes

## Architecture
The API is built on FastAPI with the following components:
- **InferenceEngine**: Core engine for model loading and inference
- **Dynamic Batcher**: Optimizes throughput by dynamically batching requests
- **Result Cache**: Deduplicates requests for improved performance
- **Feature Cache**: Stores preprocessed features to reduce overhead
- **Thread Pool**: Manages asynchronous processing of requests

## Testing
```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_api.py
```

## Deployment
The API can be deployed using various methods:
- Docker container
- Kubernetes deployment
- Directly on a server with Uvicorn or Gunicorn

### Docker Example
```bash
# Build the image
docker build -t inference-api .

# Run the container
docker run -p 8000:8000 \
  -e REQUIRE_API_KEY=1 \
  -e API_KEYS=key1,key2 \
  -e DEFAULT_MODEL_PATH=/models/model.pkl \
  -v /path/to/models:/models \
  inference-api
```

## Security & Compliance
- API key authentication for protected endpoints
- Request validation using Pydantic models
- Proper error handling and logging

> Last Updated: 2025-04-28
> Author: Evint Leovonzko