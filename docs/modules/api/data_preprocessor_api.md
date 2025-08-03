# Data Preprocessor API (`modules/api/data_preprocessor_api.py`)

## Overview

The Data Preprocessor API provides a comprehensive RESTful interface for advanced data preprocessing operations. It offers multiple data format support, batch and streaming processing capabilities, performance monitoring, and memory optimization controls through HTTP endpoints.

## Features

- **Advanced Configuration Management**: Type-safe configuration for all preprocessing operations
- **Multiple Data Formats**: Support for JSON, CSV, NumPy arrays, and file uploads
- **Batch and Streaming Processing**: Handle both batch and real-time data streams
- **Performance Monitoring**: Real-time statistics and health monitoring
- **Memory Optimization**: Intelligent memory management and caching
- **Async Processing**: Non-blocking operations with background task support
- **Security**: API key authentication and input validation
- **Persistence**: Save and load preprocessor states

## API Configuration

```python
# Environment Variables
PREPROCESSOR_API_HOST=0.0.0.0
PREPROCESSOR_API_PORT=8002
PREPROCESSOR_API_DEBUG=False
REQUIRE_API_KEY=False
API_KEYS=key1,key2,key3
MAX_WORKERS=4
MODEL_DIR=./models
TEMP_DATA_DIR=./temp_data
```

## Data Models

### PreprocessorConfigRequest
```python
{
    "normalization": "STANDARD",           # NONE, STANDARD, MINMAX, ROBUST, LOG, QUANTILE, POWER, CUSTOM
    "handle_nan": true,
    "handle_inf": true,
    "detect_outliers": false,
    "nan_strategy": "MEAN",               # MEAN, MEDIAN, MOST_FREQUENT, CONSTANT, ZERO
    "inf_strategy": "MAX_VALUE",          # MAX_VALUE, MEDIAN, CONSTANT, NAN
    "outlier_method": "ZSCORE",           # IQR, ZSCORE, PERCENTILE
    "outlier_handling": "CLIP",           # CLIP, REMOVE, WINSORIZE, MEAN, MEDIAN
    "robust_percentiles": [25.0, 75.0],
    "outlier_iqr_multiplier": 1.5,
    "outlier_zscore_threshold": 3.0,
    "outlier_percentiles": [1.0, 99.0],
    "epsilon": 1e-8,
    "clip_values": false,
    "clip_min": -inf,
    "clip_max": inf,
    "nan_fill_value": 0.0,
    "copy_X": true,
    "dtype": "float32",                   # float32, float64, int32, int64
    "debug_mode": false,
    "parallel_processing": false,
    "n_jobs": -1,
    "chunk_size": 10000,
    "cache_enabled": true,
    "enable_input_validation": true,
    "input_size_limit": null,
    "version": "1.0.0"
}
```

### DataRequest
```python
{
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # 2D array of numerical data
    "feature_names": ["feature1", "feature2", "feature3"]  # Optional feature names
}
```

## API Endpoints

### Preprocessor Management

#### Create Preprocessor
```http
POST /api/preprocessor/create
Content-Type: application/json
X-API-Key: your-api-key

{
    "config": {
        "normalization": "STANDARD",
        "handle_nan": true,
        "handle_inf": true,
        "detect_outliers": true,
        "outlier_method": "ZSCORE",
        "parallel_processing": true
    }
}
```

**Response:**
```json
{
    "preprocessor_id": "prep_12345",
    "status": "created",
    "config": {...},
    "created_at": "2025-01-15T10:30:00Z"
}
```

#### Get Preprocessor Info
```http
GET /api/preprocessor/{preprocessor_id}/info
X-API-Key: your-api-key
```

**Response:**
```json
{
    "preprocessor_id": "prep_12345",
    "config": {...},
    "is_fitted": false,
    "statistics": null,
    "created_at": "2025-01-15T10:30:00Z",
    "last_used": null,
    "fit_time": null,
    "transform_count": 0
}
```

#### List Preprocessors
```http
GET /api/preprocessor/list
X-API-Key: your-api-key
```

**Response:**
```json
{
    "preprocessors": [
        {
            "preprocessor_id": "prep_12345",
            "config": {...},
            "is_fitted": true,
            "created_at": "2025-01-15T10:30:00Z"
        }
    ],
    "total_count": 1
}
```

#### Delete Preprocessor
```http
DELETE /api/preprocessor/{preprocessor_id}
X-API-Key: your-api-key
```

**Response:**
```json
{
    "message": "Preprocessor deleted successfully",
    "preprocessor_id": "prep_12345"
}
```

### Data Processing

#### Fit Preprocessor (Inline Data)
```http
POST /api/preprocessor/{preprocessor_id}/fit
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    "feature_names": ["feature1", "feature2", "feature3"]
}
```

**Response:**
```json
{
    "message": "Preprocessor fitted successfully",
    "statistics": {
        "feature_stats": {...},
        "outlier_count": 0,
        "nan_count": 0,
        "inf_count": 0
    },
    "fit_time": 0.123,
    "fitted_at": "2025-01-15T10:30:00Z"
}
```

#### Fit Preprocessor (File Upload)
```http
POST /api/preprocessor/{preprocessor_id}/fit-file
Content-Type: multipart/form-data
X-API-Key: your-api-key

file: [CSV/JSON file]
```

#### Transform Data (Inline)
```http
POST /api/preprocessor/{preprocessor_id}/transform
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
    "feature_names": ["feature1", "feature2", "feature3"]
}
```

**Response:**
```json
{
    "transformed_data": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    "statistics": {
        "outliers_detected": 0,
        "nan_handled": 0,
        "inf_handled": 0
    },
    "transform_time": 0.045,
    "transformed_at": "2025-01-15T10:30:00Z"
}
```

#### Transform Data (File Upload)
```http
POST /api/preprocessor/{preprocessor_id}/transform-file
Content-Type: multipart/form-data
X-API-Key: your-api-key

file: [CSV/JSON file]
```

#### Fit and Transform (Combined)
```http
POST /api/preprocessor/{preprocessor_id}/fit-transform
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    "feature_names": ["feature1", "feature2", "feature3"]
}
```

**Response:**
```json
{
    "transformed_data": [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    "statistics": {
        "feature_stats": {...},
        "outlier_count": 0,
        "nan_count": 0,
        "inf_count": 0,
        "outliers_detected": 0,
        "nan_handled": 0,
        "inf_handled": 0
    },
    "fit_time": 0.123,
    "transform_time": 0.045,
    "total_time": 0.168
}
```

#### Inverse Transform
```http
POST /api/preprocessor/{preprocessor_id}/inverse-transform
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    "feature_names": ["feature1", "feature2", "feature3"]
}
```

### Batch Processing

#### Submit Batch Job
```http
POST /api/preprocessor/{preprocessor_id}/batch
Content-Type: application/json
X-API-Key: your-api-key

{
    "operation": "transform",  # fit, transform, fit_transform, inverse_transform
    "data": [[...]],
    "feature_names": [...],
    "batch_size": 1000,
    "priority": "normal"  # low, normal, high, urgent
}
```

**Response:**
```json
{
    "job_id": "job_12345",
    "status": "queued",
    "operation": "transform",
    "estimated_completion": "2025-01-15T10:35:00Z",
    "queue_position": 2
}
```

#### Get Batch Job Status
```http
GET /api/preprocessor/batch/{job_id}/status
X-API-Key: your-api-key
```

**Response:**
```json
{
    "job_id": "job_12345",
    "status": "completed",  # queued, processing, completed, failed, cancelled
    "progress": 100.0,
    "started_at": "2025-01-15T10:30:00Z",
    "completed_at": "2025-01-15T10:32:00Z",
    "result": {
        "transformed_data": [...],
        "statistics": {...}
    }
}
```

#### Get Batch Job Result
```http
GET /api/preprocessor/batch/{job_id}/result
X-API-Key: your-api-key
```

#### Cancel Batch Job
```http
DELETE /api/preprocessor/batch/{job_id}
X-API-Key: your-api-key
```

### Streaming Processing

#### Create Stream
```http
POST /api/preprocessor/{preprocessor_id}/stream/create
Content-Type: application/json
X-API-Key: your-api-key

{
    "stream_id": "stream_12345",
    "operation": "transform",
    "chunk_size": 1000,
    "buffer_size": 5000
}
```

#### Send Stream Data
```http
POST /api/preprocessor/stream/{stream_id}/send
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[...]],
    "is_final": false
}
```

#### Get Stream Results
```http
GET /api/preprocessor/stream/{stream_id}/results
X-API-Key: your-api-key
```

#### Close Stream
```http
DELETE /api/preprocessor/stream/{stream_id}
X-API-Key: your-api-key
```

### Statistics and Information

#### Get Preprocessor Statistics
```http
GET /api/preprocessor/{preprocessor_id}/statistics
X-API-Key: your-api-key
```

**Response:**
```json
{
    "feature_stats": {
        "feature1": {
            "mean": 5.0,
            "std": 2.45,
            "min": 1.0,
            "max": 9.0,
            "outlier_count": 0
        }
    },
    "global_stats": {
        "total_samples": 1000,
        "total_features": 3,
        "outlier_count": 5,
        "nan_count": 0,
        "inf_count": 0
    },
    "processing_stats": {
        "fit_time": 0.123,
        "average_transform_time": 0.045,
        "total_transforms": 25
    }
}
```

#### Get Feature Information
```http
GET /api/preprocessor/{preprocessor_id}/features
X-API-Key: your-api-key
```

**Response:**
```json
{
    "features": [
        {
            "name": "feature1",
            "index": 0,
            "dtype": "float32",
            "statistics": {...}
        }
    ],
    "total_features": 3
}
```

### Persistence

#### Save Preprocessor
```http
POST /api/preprocessor/{preprocessor_id}/save
Content-Type: application/json
X-API-Key: your-api-key

{
    "filepath": "./models/preprocessor_v1.pkl",
    "include_config": true,
    "include_statistics": true
}
```

**Response:**
```json
{
    "message": "Preprocessor saved successfully",
    "filepath": "./models/preprocessor_v1.pkl",
    "file_size": 1024,
    "saved_at": "2025-01-15T10:30:00Z"
}
```

#### Load Preprocessor
```http
POST /api/preprocessor/load
Content-Type: application/json
X-API-Key: your-api-key

{
    "filepath": "./models/preprocessor_v1.pkl",
    "preprocessor_id": "prep_loaded"
}
```

**Response:**
```json
{
    "preprocessor_id": "prep_loaded",
    "config": {...},
    "statistics": {...},
    "loaded_at": "2025-01-15T10:30:00Z"
}
```

### Performance and Monitoring

#### Get Performance Metrics
```http
GET /api/preprocessor/metrics
X-API-Key: your-api-key
```

**Response:**
```json
{
    "system_metrics": {
        "cpu_usage": 45.2,
        "memory_usage": 1024.5,
        "active_preprocessors": 3,
        "active_jobs": 5
    },
    "performance_metrics": {
        "avg_fit_time": 0.123,
        "avg_transform_time": 0.045,
        "total_operations": 1000,
        "operations_per_second": 22.5
    }
}
```

#### Health Check
```http
GET /api/preprocessor/health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-01-15T10:30:00Z",
    "uptime": "2h 30m 45s",
    "active_preprocessors": 3,
    "system_resources": {
        "cpu_usage": 45.2,
        "memory_usage": 1024.5,
        "disk_usage": 512.3
    }
}
```

### Configuration Management

#### Update Preprocessor Configuration
```http
PUT /api/preprocessor/{preprocessor_id}/config
Content-Type: application/json
X-API-Key: your-api-key

{
    "config": {
        "parallel_processing": true,
        "n_jobs": 8,
        "chunk_size": 5000
    }
}
```

#### Get Default Configuration
```http
GET /api/preprocessor/config/default
X-API-Key: your-api-key
```

#### Validate Configuration
```http
POST /api/preprocessor/config/validate
Content-Type: application/json
X-API-Key: your-api-key

{
    "config": {...}
}
```

## Usage Examples

### Basic Preprocessing Workflow

```python
import requests
import json

# API configuration
API_BASE = "http://localhost:8002"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# 1. Create preprocessor
config = {
    "normalization": "STANDARD",
    "handle_nan": True,
    "detect_outliers": True,
    "parallel_processing": True
}
response = requests.post(f"{API_BASE}/api/preprocessor/create", 
                        headers=headers, json={"config": config})
preprocessor_id = response.json()["preprocessor_id"]

# 2. Fit on training data
train_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
response = requests.post(f"{API_BASE}/api/preprocessor/{preprocessor_id}/fit",
                        headers=headers, json={"data": train_data})

# 3. Transform new data
test_data = [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]
response = requests.post(f"{API_BASE}/api/preprocessor/{preprocessor_id}/transform",
                        headers=headers, json={"data": test_data})
transformed_data = response.json()["transformed_data"]

print("Transformed data:", transformed_data)
```

### File Upload Processing

```python
import requests

# Upload and process CSV file
with open("data.csv", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{API_BASE}/api/preprocessor/{preprocessor_id}/fit-file",
                           headers={"X-API-Key": API_KEY}, files=files)
```

### Batch Processing

```python
# Submit large batch job
batch_data = {
    "operation": "transform",
    "data": large_dataset,  # Large 2D array
    "batch_size": 1000,
    "priority": "high"
}
response = requests.post(f"{API_BASE}/api/preprocessor/{preprocessor_id}/batch",
                        headers=headers, json=batch_data)
job_id = response.json()["job_id"]

# Check job status
response = requests.get(f"{API_BASE}/api/preprocessor/batch/{job_id}/status",
                       headers={"X-API-Key": API_KEY})
status = response.json()["status"]

# Get results when completed
if status == "completed":
    response = requests.get(f"{API_BASE}/api/preprocessor/batch/{job_id}/result",
                           headers={"X-API-Key": API_KEY})
    results = response.json()
```

### Streaming Processing

```python
# Create stream
stream_config = {
    "stream_id": "my_stream",
    "operation": "transform",
    "chunk_size": 1000
}
response = requests.post(f"{API_BASE}/api/preprocessor/{preprocessor_id}/stream/create",
                        headers=headers, json=stream_config)

# Send data chunks
for chunk in data_chunks:
    chunk_data = {
        "data": chunk,
        "is_final": False  # Set to True for last chunk
    }
    requests.post(f"{API_BASE}/api/preprocessor/stream/my_stream/send",
                 headers=headers, json=chunk_data)

# Get accumulated results
response = requests.get(f"{API_BASE}/api/preprocessor/stream/my_stream/results",
                       headers={"X-API-Key": API_KEY})
results = response.json()
```

## Error Handling

### Common Error Codes

- **400 Bad Request**: Invalid input data or configuration
- **401 Unauthorized**: Missing or invalid API key
- **404 Not Found**: Preprocessor not found
- **409 Conflict**: Preprocessor already fitted (for fit operations)
- **422 Unprocessable Entity**: Validation errors
- **500 Internal Server Error**: Processing errors

### Error Response Format

```json
{
    "error": "ValidationError",
    "message": "Data validation failed",
    "details": {
        "field": "data",
        "issue": "Data cannot be empty"
    },
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_12345"
}
```

## Best Practices

### Performance Optimization

1. **Use Parallel Processing**: Enable `parallel_processing` for large datasets
2. **Optimize Chunk Size**: Set appropriate `chunk_size` based on available memory
3. **Enable Caching**: Use caching for repeated operations
4. **Batch Operations**: Use batch processing for large datasets

### Memory Management

1. **Set Input Limits**: Configure `input_size_limit` to prevent memory overflow
2. **Use Streaming**: Use streaming processing for very large datasets
3. **Monitor Resources**: Regularly check performance metrics
4. **Clean Up**: Delete unused preprocessors to free memory

### Security

1. **API Keys**: Always use API keys in production
2. **Input Validation**: Enable input validation to prevent malicious data
3. **File Upload Limits**: Set appropriate file size limits
4. **Error Handling**: Implement proper error handling in client code

### Configuration

1. **Environment-Specific**: Use different configurations for dev/prod
2. **Resource Allocation**: Adjust `n_jobs` based on available CPU cores
3. **Monitoring**: Enable debug mode for troubleshooting
4. **Persistence**: Save important preprocessor states

## Advanced Features

### Custom Normalization

The API supports custom normalization methods by extending the configuration:

```python
config = {
    "normalization": "CUSTOM",
    "custom_params": {
        "method": "quantile_uniform",
        "output_distribution": "normal"
    }
}
```

### Feature-Specific Processing

Apply different preprocessing to specific features:

```python
config = {
    "feature_specific_config": {
        "feature1": {"normalization": "MINMAX"},
        "feature2": {"normalization": "ROBUST"},
        "feature3": {"handle_outliers": False}
    }
}
```

### Pipeline Integration

The API can be integrated with ML pipelines:

```python
# Save preprocessor state
requests.post(f"{API_BASE}/api/preprocessor/{preprocessor_id}/save",
             headers=headers, json={"filepath": "pipeline_prep.pkl"})

# Load in production
requests.post(f"{API_BASE}/api/preprocessor/load",
             headers=headers, 
             json={"filepath": "pipeline_prep.pkl", "preprocessor_id": "prod_prep"})
```

## Related Documentation

- [Data Preprocessor Engine](../engine/data_preprocessor.md) - Core preprocessing engine
- [Configuration System](../configs.md) - Configuration management
- [Batch Processor API](batch_processor_api.md) - Batch processing API
- [Performance Metrics](../engine/performance_metrics.md) - Performance monitoring

---

*The Data Preprocessor API provides a comprehensive interface for advanced data preprocessing with enterprise-grade features including security, monitoring, and scalability.*
