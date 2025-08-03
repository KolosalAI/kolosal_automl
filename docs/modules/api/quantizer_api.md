# Quantizer API (`modules/api/quantizer_api.py`)

## Overview

The Quantizer API provides a high-performance RESTful interface for data and model quantization operations. It offers advanced quantization techniques, multiple precision modes, and comprehensive optimization features for reducing model size and improving inference speed.

## Features

- **Multiple Quantization Types**: INT8, UINT8, INT16, FLOAT16, Mixed Precision
- **Dynamic and Static Quantization**: Support for both calibrated and dynamic quantization
- **Per-Channel Quantization**: Fine-grained quantization control
- **Mixed Precision**: Selective quantization for optimal performance
- **Caching System**: Intelligent caching for repeated operations
- **Outlier Handling**: Advanced outlier detection and handling
- **Memory Optimization**: Efficient memory usage with buffer management
- **Performance Monitoring**: Real-time quantization metrics

## API Configuration

```python
# Environment Variables
QUANTIZER_API_HOST=0.0.0.0
QUANTIZER_API_PORT=8005
QUANTIZER_API_DEBUG=False
REQUIRE_API_KEY=False
API_KEYS=key1,key2,key3
MAX_WORKERS=4
CACHE_SIZE=1000
ENABLE_MIXED_PRECISION=True
DEFAULT_QUANTIZATION_TYPE=INT8
```

## Data Models

### QuantizerConfigModel
```python
{
    "quantization_type": "INT8",                    # INT8, UINT8, INT16, FLOAT16, NONE, MIXED
    "quantization_mode": "DYNAMIC",                 # DYNAMIC, DYNAMIC_PER_BATCH, CALIBRATED, STATIC
    "num_bits": 8,                                  # Number of bits for quantization
    "symmetric": false,                             # Whether to use symmetric quantization
    "enable_cache": true,                           # Whether to enable caching
    "cache_size": 1000,                            # Size of the cache
    "enable_mixed_precision": false,                # Whether to enable mixed precision
    "per_channel": false,                          # Whether to use per-channel quantization
    "buffer_size": 0,                              # Size of preallocated buffers
    "use_percentile": false,                       # Whether to use percentiles for range calculation
    "min_percentile": 0.01,                        # Minimum percentile for range calculation
    "max_percentile": 99.99,                       # Maximum percentile for range calculation
    "error_on_nan": true,                          # Whether to error on NaN values
    "error_on_inf": true,                          # Whether to error on Inf values
    "optimize_memory": false,                      # Whether to optimize memory usage
    "enable_requantization": false,                # Whether to enable requantization
    "requantization_threshold": 0.01,              # Threshold for requantization
    "outlier_threshold": null,                     # Threshold for outlier removal
    "skip_layers": [],                             # Layers to skip during quantization
    "quantize_bias": true,                         # Whether to quantize bias layers
    "quantize_weights_only": false,                # Whether to quantize weights only
    "mixed_precision_layers": [],                  # Layers to use mixed precision for
    "custom_quantization_config": {}               # Custom quantization configuration
}
```

### QuantizationRequest
```python
{
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],   # 2D array of data to quantize
    "validate": true,                               # Whether to validate input
    "channel_dim": null,                           # Dimension index for per-channel quantization
    "layer_name": null                             # Layer name for mixed precision handling
}
```

### CalibrationRequest
```python
{
    "data": [                                      # List of 2D arrays for calibration
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ]
}
```

## API Endpoints

### Quantizer Management

#### Create Quantizer
```http
POST /api/quantizer/create
Content-Type: application/json
X-API-Key: your-api-key

{
    "quantizer_id": "quantizer_prod",
    "config": {
        "quantization_type": "INT8",
        "quantization_mode": "DYNAMIC",
        "symmetric": false,
        "enable_cache": true,
        "per_channel": true,
        "enable_mixed_precision": true
    }
}
```

**Response:**
```json
{
    "message": "Quantizer created successfully",
    "quantizer_id": "quantizer_prod",
    "config": {...},
    "created_at": "2025-07-16T10:30:00Z"
}
```

#### Get Quantizer Info
```http
GET /api/quantizer/{quantizer_id}/info
X-API-Key: your-api-key
```

**Response:**
```json
{
    "quantizer_id": "quantizer_prod",
    "config": {...},
    "statistics": {
        "total_quantizations": 150,
        "cache_hits": 120,
        "cache_misses": 30,
        "average_quantization_time": 0.045,
        "total_data_processed": 1000000
    },
    "created_at": "2025-07-16T10:30:00Z",
    "last_used": "2025-07-16T10:35:00Z"
}
```

#### List Quantizers
```http
GET /api/quantizer/list
X-API-Key: your-api-key
```

**Response:**
```json
{
    "quantizers": [
        {
            "quantizer_id": "quantizer_prod",
            "quantization_type": "INT8",
            "quantization_mode": "DYNAMIC", 
            "created_at": "2025-07-16T10:30:00Z",
            "total_quantizations": 150
        }
    ],
    "total_quantizers": 1
}
```

#### Update Quantizer Configuration
```http
PUT /api/quantizer/{quantizer_id}/config
Content-Type: application/json
X-API-Key: your-api-key

{
    "config": {
        "cache_size": 2000,
        "enable_mixed_precision": true,
        "outlier_threshold": 3.0
    }
}
```

#### Delete Quantizer
```http
DELETE /api/quantizer/{quantizer_id}
X-API-Key: your-api-key
```

### Quantization Operations

#### Quantize Data
```http
POST /api/quantizer/{quantizer_id}/quantize
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [
        [1.5, 2.8, 3.2, 4.7],
        [5.1, 6.3, 7.9, 8.4],
        [9.2, 10.6, 11.1, 12.8]
    ],
    "validate": true,
    "channel_dim": 1,
    "layer_name": "conv1"
}
```

**Response:**
```json
{
    "quantized_data": [
        [15, 28, 32, 47],
        [51, 63, 79, 84],
        [92, 106, 111, 128]
    ],
    "scale_factors": [0.1, 0.1, 0.1, 0.1],
    "zero_points": [0, 0, 0, 0],
    "quantization_info": {
        "quantization_type": "INT8",
        "symmetric": false,
        "per_channel": true,
        "channel_dim": 1
    },
    "statistics": {
        "input_range": [-1.0, 12.8],
        "output_range": [-128, 127],
        "compression_ratio": 4.0,
        "quantization_error": 0.05
    },
    "quantization_time": 0.023,
    "cache_hit": false
}
```

#### Dequantize Data
```http
POST /api/quantizer/{quantizer_id}/dequantize
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [
        [15, 28, 32, 47],
        [51, 63, 79, 84],
        [92, 106, 111, 128]
    ],
    "channel_dim": 1,
    "layer_name": "conv1"
}
```

**Response:**
```json
{
    "dequantized_data": [
        [1.5, 2.8, 3.2, 4.7],
        [5.1, 6.3, 7.9, 8.4],
        [9.2, 10.6, 11.1, 12.8]
    ],
    "dequantization_info": {
        "scale_factors_used": [0.1, 0.1, 0.1, 0.1],
        "zero_points_used": [0, 0, 0, 0]
    },
    "reconstruction_error": 0.02,
    "dequantization_time": 0.018
}
```

#### Calibrate Quantizer
```http
POST /api/quantizer/{quantizer_id}/calibrate
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
    ]
}
```

**Response:**
```json
{
    "message": "Quantizer calibrated successfully",
    "calibration_info": {
        "samples_processed": 3,
        "data_points": 18,
        "min_values": [1.0, 2.0, 3.0],
        "max_values": [16.0, 17.0, 18.0],
        "scale_factors": [0.063, 0.059, 0.055],
        "zero_points": [0, 0, 0]
    },
    "calibration_time": 0.156,
    "calibrated_at": "2025-07-16T10:30:00Z"
}
```

### Batch Operations

#### Batch Quantize
```http
POST /api/quantizer/{quantizer_id}/batch-quantize
Content-Type: application/json
X-API-Key: your-api-key

{
    "batch_data": [
        {
            "data": [[1.0, 2.0], [3.0, 4.0]],
            "layer_name": "layer1"
        },
        {
            "data": [[5.0, 6.0], [7.0, 8.0]],
            "layer_name": "layer2"
        }
    ],
    "batch_size": 32,
    "parallel_processing": true
}
```

**Response:**
```json
{
    "batch_results": [
        {
            "layer_name": "layer1",
            "quantized_data": [[10, 20], [30, 40]],
            "scale_factors": [0.1, 0.1],
            "quantization_time": 0.023
        },
        {
            "layer_name": "layer2", 
            "quantized_data": [[50, 60], [70, 80]],
            "scale_factors": [0.1, 0.1],
            "quantization_time": 0.021
        }
    ],
    "total_time": 0.067,
    "items_processed": 2,
    "parallel_execution": true
}
```

#### Batch Dequantize
```http
POST /api/quantizer/{quantizer_id}/batch-dequantize
Content-Type: application/json
X-API-Key: your-api-key

{
    "batch_data": [
        {
            "data": [[10, 20], [30, 40]],
            "layer_name": "layer1"
        },
        {
            "data": [[50, 60], [70, 80]], 
            "layer_name": "layer2"
        }
    ]
}
```

### Advanced Quantization

#### Mixed Precision Quantization
```http
POST /api/quantizer/{quantizer_id}/mixed-precision
Content-Type: application/json
X-API-Key: your-api-key

{
    "layers_data": {
        "conv1": {
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "precision": "INT8"
        },
        "conv2": {
            "data": [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            "precision": "FLOAT16"
        },
        "fc1": {
            "data": [[13.0, 14.0], [15.0, 16.0]],
            "precision": "INT16"
        }
    }
}
```

**Response:**
```json
{
    "mixed_precision_results": {
        "conv1": {
            "quantized_data": [[10, 20, 30], [40, 50, 60]],
            "precision": "INT8",
            "compression_ratio": 4.0
        },
        "conv2": {
            "quantized_data": [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            "precision": "FLOAT16", 
            "compression_ratio": 2.0
        },
        "fc1": {
            "quantized_data": [[1300, 1400], [1500, 1600]],
            "precision": "INT16",
            "compression_ratio": 2.0
        }
    },
    "overall_compression_ratio": 2.67,
    "total_quantization_time": 0.089
}
```

#### Adaptive Quantization
```http
POST /api/quantizer/{quantizer_id}/adaptive-quantize
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "target_compression": 4.0,
    "max_error_threshold": 0.05,
    "optimize_for": "speed"  # speed, memory, accuracy
}
```

**Response:**
```json
{
    "adaptive_result": {
        "selected_precision": "INT8",
        "quantized_data": [[10, 20, 30], [40, 50, 60]],
        "achieved_compression": 4.0,
        "quantization_error": 0.03,
        "reasoning": "INT8 selected for optimal speed with acceptable error"
    },
    "alternatives": [
        {
            "precision": "INT16",
            "compression": 2.0,
            "error": 0.01,
            "speed_score": 0.8
        }
    ]
}
```

### Performance and Analysis

#### Analyze Data for Quantization
```http
POST /api/quantizer/analyze
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    "analysis_options": {
        "distribution_analysis": true,
        "outlier_detection": true,
        "quantization_sensitivity": true,
        "compression_estimation": true
    }
}
```

**Response:**
```json
{
    "data_analysis": {
        "shape": [3, 3],
        "statistics": {
            "min": 1.0,
            "max": 9.0,
            "mean": 5.0,
            "std": 2.58,
            "range": 8.0
        },
        "distribution": {
            "histogram": [...],
            "is_normal": false,
            "skewness": 0.0,
            "kurtosis": -1.2
        },
        "outliers": {
            "count": 0,
            "indices": [],
            "threshold_used": 3.0
        },
        "quantization_recommendations": {
            "recommended_type": "INT8",
            "expected_compression": 4.0,
            "estimated_error": 0.03,
            "per_channel_beneficial": false
        }
    },
    "compression_estimates": {
        "INT8": {"compression": 4.0, "error": 0.03},
        "INT16": {"compression": 2.0, "error": 0.01},
        "FLOAT16": {"compression": 2.0, "error": 0.001}
    }
}
```

#### Compare Quantization Methods
```http
POST /api/quantizer/compare-methods
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "methods": [
        {"type": "INT8", "mode": "DYNAMIC"},
        {"type": "INT8", "mode": "STATIC"},
        {"type": "FLOAT16", "mode": "DYNAMIC"}
    ],
    "evaluation_metrics": ["compression_ratio", "quantization_error", "speed"]
}
```

**Response:**
```json
{
    "method_comparison": {
        "INT8_DYNAMIC": {
            "compression_ratio": 4.0,
            "quantization_error": 0.03,
            "quantization_speed": 0.023,
            "memory_usage": "25%",
            "score": 8.5
        },
        "INT8_STATIC": {
            "compression_ratio": 4.0,
            "quantization_error": 0.02,
            "quantization_speed": 0.018,
            "memory_usage": "25%",
            "score": 9.0
        },
        "FLOAT16_DYNAMIC": {
            "compression_ratio": 2.0,
            "quantization_error": 0.001,
            "quantization_speed": 0.015,
            "memory_usage": "50%",
            "score": 7.5
        }
    },
    "best_method": "INT8_STATIC",
    "ranking": ["INT8_STATIC", "INT8_DYNAMIC", "FLOAT16_DYNAMIC"]
}
```

### Model Quantization

#### Quantize Model Weights
```http
POST /api/quantizer/model/quantize-weights
Content-Type: application/json
X-API-Key: your-api-key

{
    "model_layers": {
        "conv1.weight": {
            "data": [[[...]], [[...]]],  # 4D weight tensor
            "quantization_type": "INT8"
        },
        "conv2.weight": {
            "data": [[[...]], [[...]]],
            "quantization_type": "INT8"
        },
        "fc1.weight": {
            "data": [[...], [...]],      # 2D weight matrix
            "quantization_type": "INT16"
        }
    },
    "per_channel": true,
    "symmetric": false
}
```

**Response:**
```json
{
    "quantized_model": {
        "conv1.weight": {
            "quantized_data": [...],
            "scale_factors": [...],
            "zero_points": [...],
            "compression_ratio": 4.0
        },
        "conv2.weight": {
            "quantized_data": [...],
            "scale_factors": [...],
            "zero_points": [...],
            "compression_ratio": 4.0
        },
        "fc1.weight": {
            "quantized_data": [...],
            "scale_factors": [...],
            "zero_points": [...],
            "compression_ratio": 2.0
        }
    },
    "overall_compression": 3.33,
    "model_size_reduction": "70%",
    "quantization_time": 1.234
}
```

#### Post-Training Quantization
```http
POST /api/quantizer/model/post-training-quantize
Content-Type: application/json
X-API-Key: your-api-key

{
    "model_path": "/path/to/model.onnx",
    "calibration_data": [...],
    "quantization_config": {
        "quantization_type": "INT8",
        "quantization_mode": "STATIC",
        "per_channel": true,
        "calibration_samples": 100
    }
}
```

### Cache and Performance

#### Get Cache Statistics
```http
GET /api/quantizer/{quantizer_id}/cache/stats
X-API-Key: your-api-key
```

**Response:**
```json
{
    "cache_statistics": {
        "total_requests": 200,
        "cache_hits": 150,
        "cache_misses": 50,
        "hit_rate": 0.75,
        "cache_size": 1000,
        "used_entries": 450,
        "memory_usage_mb": 25.6
    },
    "performance_impact": {
        "average_hit_time": 0.001,
        "average_miss_time": 0.025,
        "time_saved": 3.75
    }
}
```

#### Clear Cache
```http
DELETE /api/quantizer/{quantizer_id}/cache
X-API-Key: your-api-key
```

#### Optimize Cache
```http
POST /api/quantizer/{quantizer_id}/cache/optimize
X-API-Key: your-api-key
```

### Health and Monitoring

#### Health Check
```http
GET /api/quantizer/health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-07-16T10:30:00Z",
    "checks": {
        "quantizer_engine": "healthy",
        "memory_usage": "healthy",
        "cache_system": "healthy",
        "performance": "healthy"
    },
    "uptime": "5d 12h 30m",
    "active_quantizers": 3
}
```

#### Get Performance Metrics
```http
GET /api/quantizer/metrics
X-API-Key: your-api-key
```

**Response:**
```json
{
    "performance_metrics": {
        "total_quantizations": 1500,
        "average_quantization_time": 0.025,
        "throughput_ops_per_second": 40.0,
        "total_data_processed_gb": 15.6,
        "cache_hit_rate": 0.75
    },
    "system_metrics": {
        "cpu_usage": 45.2,
        "memory_usage_mb": 512.5,
        "active_quantizers": 3,
        "queued_operations": 0
    },
    "compression_metrics": {
        "average_compression_ratio": 3.8,
        "total_size_reduction_gb": 45.2,
        "average_quantization_error": 0.025
    }
}
```

## Usage Examples

### Basic Quantization Workflow

```python
import requests
import json
import numpy as np

# API configuration
API_BASE = "http://localhost:8005"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# 1. Create quantizer with specific configuration
quantizer_config = {
    "quantizer_id": "prod_quantizer",
    "config": {
        "quantization_type": "INT8",
        "quantization_mode": "DYNAMIC",
        "symmetric": False,
        "enable_cache": True,
        "per_channel": True
    }
}
response = requests.post(f"{API_BASE}/api/quantizer/create",
                        headers=headers, json=quantizer_config)
quantizer_id = quantizer_config["quantizer_id"]

# 2. Quantize data
data = np.random.randn(100, 50).tolist()  # Convert numpy to list
quantization_request = {
    "data": data,
    "validate": True,
    "channel_dim": 1
}
response = requests.post(f"{API_BASE}/api/quantizer/{quantizer_id}/quantize",
                        headers=headers, json=quantization_request)
result = response.json()

print("Quantized data shape:", np.array(result["quantized_data"]).shape)
print("Compression ratio:", result["statistics"]["compression_ratio"])
print("Quantization error:", result["statistics"]["quantization_error"])

# 3. Dequantize for verification
dequantization_request = {
    "data": result["quantized_data"],
    "channel_dim": 1
}
response = requests.post(f"{API_BASE}/api/quantizer/{quantizer_id}/dequantize",
                        headers=headers, json=dequantization_request)
dequantized = response.json()

print("Reconstruction error:", dequantized["reconstruction_error"])
```

### Model Quantization

```python
# Simulate model weights (normally loaded from actual model)
model_weights = {
    "conv1.weight": np.random.randn(64, 3, 3, 3).tolist(),  # Conv layer
    "conv2.weight": np.random.randn(128, 64, 3, 3).tolist(),
    "fc1.weight": np.random.randn(512, 2048).tolist()       # FC layer
}

# Quantize model weights
quantize_request = {
    "model_layers": {
        "conv1.weight": {
            "data": model_weights["conv1.weight"],
            "quantization_type": "INT8"
        },
        "conv2.weight": {
            "data": model_weights["conv2.weight"],
            "quantization_type": "INT8"
        },
        "fc1.weight": {
            "data": model_weights["fc1.weight"],
            "quantization_type": "INT16"  # Higher precision for FC layer
        }
    },
    "per_channel": True,
    "symmetric": False
}

response = requests.post(f"{API_BASE}/api/quantizer/model/quantize-weights",
                        headers=headers, json=quantize_request)
quantized_model = response.json()

print("Overall compression:", quantized_model["overall_compression"])
print("Model size reduction:", quantized_model["model_size_reduction"])
```

### Calibration and Static Quantization

```python
# Prepare calibration data
calibration_data = []
for _ in range(10):  # 10 calibration samples
    calibration_data.append(np.random.randn(32, 50).tolist())

# Calibrate quantizer
calibration_request = {
    "data": calibration_data
}
response = requests.post(f"{API_BASE}/api/quantizer/{quantizer_id}/calibrate",
                        headers=headers, json=calibration_request)
calibration_result = response.json()

print("Calibration completed:")
print("Scale factors:", calibration_result["calibration_info"]["scale_factors"])
print("Zero points:", calibration_result["calibration_info"]["zero_points"])

# Now use static quantization mode
config_update = {
    "config": {
        "quantization_mode": "STATIC"
    }
}
requests.put(f"{API_BASE}/api/quantizer/{quantizer_id}/config",
            headers=headers, json=config_update)
```

### Mixed Precision Quantization

```python
# Define different precision requirements for different layers
mixed_precision_data = {
    "layers_data": {
        "feature_extractor": {
            "data": np.random.randn(64, 128).tolist(),
            "precision": "INT8"  # Lower precision for feature extraction
        },
        "attention_weights": {
            "data": np.random.randn(512, 512).tolist(),
            "precision": "FLOAT16"  # Medium precision for attention
        },
        "classifier_head": {
            "data": np.random.randn(10, 512).tolist(),
            "precision": "INT16"  # Higher precision for final classification
        }
    }
}

response = requests.post(f"{API_BASE}/api/quantizer/{quantizer_id}/mixed-precision",
                        headers=headers, json=mixed_precision_data)
mixed_result = response.json()

print("Mixed precision results:")
for layer, result in mixed_result["mixed_precision_results"].items():
    print(f"{layer}: {result['precision']}, compression: {result['compression_ratio']}")
```

### Batch Processing

```python
# Prepare batch data for multiple layers
batch_data = []
for i in range(5):
    layer_data = {
        "data": np.random.randn(32, 64).tolist(),
        "layer_name": f"layer_{i}"
    }
    batch_data.append(layer_data)

# Batch quantize
batch_request = {
    "batch_data": batch_data,
    "batch_size": 32,
    "parallel_processing": True
}
response = requests.post(f"{API_BASE}/api/quantizer/{quantizer_id}/batch-quantize",
                        headers=headers, json=batch_request)
batch_result = response.json()

print("Batch processing results:")
print(f"Total time: {batch_result['total_time']}")
print(f"Items processed: {batch_result['items_processed']}")
for result in batch_result["batch_results"]:
    print(f"Layer {result['layer_name']}: {result['quantization_time']}s")
```

### Performance Analysis

```python
# Analyze data for optimal quantization strategy
test_data = np.random.randn(1000, 256).tolist()
analysis_request = {
    "data": test_data,
    "analysis_options": {
        "distribution_analysis": True,
        "outlier_detection": True,
        "quantization_sensitivity": True,
        "compression_estimation": True
    }
}
response = requests.post(f"{API_BASE}/api/quantizer/analyze",
                        headers=headers, json=analysis_request)
analysis = response.json()

print("Data analysis results:")
print("Recommended type:", analysis["data_analysis"]["quantization_recommendations"]["recommended_type"])
print("Expected compression:", analysis["data_analysis"]["quantization_recommendations"]["expected_compression"])
print("Estimated error:", analysis["data_analysis"]["quantization_recommendations"]["estimated_error"])

# Compare different quantization methods
comparison_request = {
    "data": test_data[:100],  # Use smaller subset for comparison
    "methods": [
        {"type": "INT8", "mode": "DYNAMIC"},
        {"type": "INT8", "mode": "STATIC"},
        {"type": "INT16", "mode": "DYNAMIC"},
        {"type": "FLOAT16", "mode": "DYNAMIC"}
    ],
    "evaluation_metrics": ["compression_ratio", "quantization_error", "speed"]
}
response = requests.post(f"{API_BASE}/api/quantizer/compare-methods",
                        headers=headers, json=comparison_request)
comparison = response.json()

print("\nMethod comparison:")
for method, metrics in comparison["method_comparison"].items():
    print(f"{method}: Score {metrics['score']}, Compression {metrics['compression_ratio']}")
print("Best method:", comparison["best_method"])
```

### Monitoring and Optimization

```python
# Get performance metrics
response = requests.get(f"{API_BASE}/api/quantizer/metrics",
                       headers={"X-API-Key": API_KEY})
metrics = response.json()

print("Performance metrics:")
print(f"Total quantizations: {metrics['performance_metrics']['total_quantizations']}")
print(f"Average time: {metrics['performance_metrics']['average_quantization_time']}")
print(f"Throughput: {metrics['performance_metrics']['throughput_ops_per_second']} ops/sec")
print(f"Cache hit rate: {metrics['performance_metrics']['cache_hit_rate']}")

# Get cache statistics
response = requests.get(f"{API_BASE}/api/quantizer/{quantizer_id}/cache/stats",
                       headers={"X-API-Key": API_KEY})
cache_stats = response.json()

print("\nCache statistics:")
print(f"Hit rate: {cache_stats['cache_statistics']['hit_rate']}")
print(f"Time saved: {cache_stats['performance_impact']['time_saved']}s")

# Optimize cache if hit rate is low
if cache_stats['cache_statistics']['hit_rate'] < 0.5:
    requests.post(f"{API_BASE}/api/quantizer/{quantizer_id}/cache/optimize",
                 headers={"X-API-Key": API_KEY})
    print("Cache optimization triggered")
```

## Error Handling

### Common Error Codes

- **400 Bad Request**: Invalid quantization parameters or data format
- **401 Unauthorized**: Missing or invalid API key
- **404 Not Found**: Quantizer not found
- **422 Unprocessable Entity**: Data validation or quantization errors
- **500 Internal Server Error**: Quantization engine or system errors

### Error Response Format

```json
{
    "error": "QuantizationError",
    "message": "Data contains NaN values",
    "details": {
        "quantizer_id": "prod_quantizer",
        "data_shape": [100, 50],
        "nan_count": 5,
        "nan_positions": [[10, 5], [25, 12], [67, 34], [89, 7], [95, 23]]
    },
    "timestamp": "2025-07-16T10:30:00Z",
    "request_id": "req_12345"
}
```

## Best Practices

### Quantization Strategy

1. **Data Analysis**: Always analyze data before choosing quantization type
2. **Calibration**: Use representative calibration data for static quantization
3. **Mixed Precision**: Use mixed precision for optimal performance/accuracy trade-off
4. **Per-Channel**: Enable per-channel quantization for better accuracy
5. **Outlier Handling**: Set appropriate outlier thresholds

### Performance Optimization

1. **Caching**: Enable caching for repeated quantization operations
2. **Batch Processing**: Use batch operations for multiple layers
3. **Parallel Processing**: Enable parallel processing for large batches
4. **Memory Management**: Monitor memory usage and optimize buffer sizes

### Accuracy Preservation

1. **Calibration Data**: Use diverse, representative calibration data
2. **Precision Selection**: Choose precision based on layer sensitivity
3. **Error Monitoring**: Monitor quantization errors regularly
4. **Validation**: Always validate quantized models before deployment

### Production Deployment

1. **Performance Testing**: Benchmark quantized models thoroughly
2. **A/B Testing**: Compare quantized vs. full-precision models
3. **Monitoring**: Monitor inference performance and accuracy
4. **Rollback Plans**: Have plans for reverting to full-precision if needed

## Advanced Features

### Custom Quantization Schemes

Support for custom quantization implementations:

```python
custom_config = {
    "custom_quantization_config": {
        "algorithm": "logarithmic",
        "base": 2,
        "custom_scale_computation": "adaptive"
    }
}
```

### Quantization-Aware Training Integration

Integration with quantization-aware training:

```python
qat_config = {
    "qat_mode": True,
    "fake_quantization": True,
    "gradient_scaling": True
}
```

### Hardware-Specific Optimization

Optimization for specific hardware targets:

```python
hardware_config = {
    "target_hardware": "arm_neon",  # arm_neon, avx2, cuda
    "optimization_level": "aggressive"
}
```

## Related Documentation

- [Quantizer Engine](../engine/quantizer.md) - Core quantization engine
- [Configuration System](../configs.md) - Configuration management
- [Inference Engine API](inference_engine_api.md) - Model inference API
- [Mixed Precision](../engine/mixed_precision.md) - Mixed precision training

---

*The Quantizer API provides comprehensive quantization capabilities for optimizing model size and inference speed while maintaining accuracy through advanced techniques and monitoring.*
