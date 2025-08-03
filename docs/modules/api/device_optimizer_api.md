# Device Optimizer API (`modules/api/device_optimizer_api.py`)

## Overview

The Device Optimizer API provides a comprehensive RESTful interface for hardware detection, configuration generation, and optimization based on system capabilities. It automatically detects system resources and generates optimized configurations for different computing environments and workloads.

## Features

- **Automatic Hardware Detection**: CPU, GPU, memory, and specialized accelerator detection
- **Environment-Specific Optimization**: Cloud, desktop, edge computing optimizations
- **Workload-Aware Configuration**: Training, inference, and mixed workload optimizations
- **Resource Management**: Memory reservation and power efficiency controls
- **Configuration Generation**: Automated generation of optimized configurations
- **Performance Tuning**: Automatic parameter tuning and optimization
- **Fault Tolerance**: Configurable resilience levels for different environments
- **Debug and Monitoring**: Comprehensive debugging and monitoring capabilities

## API Configuration

```python
# Environment Variables
DEVICE_OPTIMIZER_API_HOST=0.0.0.0
DEVICE_OPTIMIZER_API_PORT=8006
DEVICE_OPTIMIZER_API_DEBUG=False
REQUIRE_API_KEY=False
API_KEYS=key1,key2,key3
CONFIG_PATH=./configs
CHECKPOINT_PATH=./checkpoints
MODEL_REGISTRY_PATH=./model_registry
ENABLE_GPU_DETECTION=True
ENABLE_SPECIALIZED_ACCELERATORS=True
```

## Data Models

### OptimizerRequest
```python
{
    "config_path": "./configs",                          # Path to save configuration files
    "checkpoint_path": "./checkpoints",                  # Path for model checkpoints
    "model_registry_path": "./model_registry",          # Path for model registry
    "optimization_mode": "BALANCED",                     # PERFORMANCE, MEMORY, BALANCED, POWER_EFFICIENT
    "workload_type": "mixed",                           # mixed, inference, training
    "environment": "auto",                              # auto, cloud, desktop, edge
    "enable_specialized_accelerators": true,            # Whether to enable detection of specialized hardware
    "memory_reservation_percent": 10.0,                # Percentage of memory to reserve for the system
    "power_efficiency": false,                          # Whether to optimize for power efficiency
    "resilience_level": 1,                             # Level of fault tolerance (0-3)
    "auto_tune": true,                                  # Whether to enable automatic parameter tuning
    "config_id": null,                                  # Optional identifier for the configuration set
    "debug_mode": false                                 # Enable debug mode for more verbose logging
}
```

### SystemInfoRequest
```python
{
    "enable_specialized_accelerators": true             # Whether to detect specialized hardware
}
```

### LoadConfigRequest
```python
{
    "config_path": "./configs",                         # Path where configuration files are stored
    "config_id": "config_12345"                        # Identifier for the configuration set
}
```

### ApplyConfigRequest
```python
{
    "configs": {                                        # Dictionary with configurations to apply
        "quantization_config": {...},
        "batch_processor_config": {...},
        "preprocessor_config": {...}
    }
}
```

## API Endpoints

### System Information

#### Get System Information
```http
POST /api/device-optimizer/system-info
Content-Type: application/json
X-API-Key: your-api-key

{
    "enable_specialized_accelerators": true
}
```

**Response:**
```json
{
    "system_info": {
        "cpu": {
            "model": "Intel(R) Core(TM) i9-12900K",
            "cores": 16,
            "threads": 24,
            "base_frequency": 3.2,
            "max_frequency": 5.2,
            "cache_l1": "80KB",
            "cache_l2": "1280KB", 
            "cache_l3": "30MB",
            "architecture": "x86_64",
            "features": ["AVX2", "AVX512", "FMA3", "SSE4.2"],
            "numa_nodes": 1
        },
        "memory": {
            "total_gb": 32.0,
            "available_gb": 28.5,
            "type": "DDR4",
            "speed_mhz": 3200,
            "channels": 2,
            "ecc_supported": false
        },
        "gpu": [
            {
                "name": "NVIDIA GeForce RTX 4090",
                "memory_gb": 24.0,
                "cuda_cores": 16384,
                "tensor_cores": 512,
                "compute_capability": "8.9",
                "driver_version": "535.98",
                "cuda_version": "12.2"
            }
        ],
        "storage": [
            {
                "device": "/dev/nvme0n1",
                "type": "NVMe SSD",
                "size_gb": 1000,
                "read_speed_mbps": 7000,
                "write_speed_mbps": 6500
            }
        ],
        "specialized_accelerators": [
            {
                "type": "Intel Neural Compute Stick",
                "name": "Intel Movidius",
                "supported_frameworks": ["OpenVINO", "ONNX"]
            }
        ],
        "network": {
            "interfaces": ["ethernet", "wifi"],
            "bandwidth_mbps": 1000
        },
        "os": {
            "name": "Ubuntu",
            "version": "22.04 LTS",
            "kernel": "5.15.0-75-generic",
            "architecture": "x86_64"
        }
    },
    "detection_time": 2.345,
    "timestamp": "2025-07-16T10:30:00Z"
}
```

#### Get Optimized Configuration
```http
POST /api/device-optimizer/optimize
Content-Type: application/json
X-API-Key: your-api-key

{
    "optimization_mode": "PERFORMANCE",
    "workload_type": "training",
    "environment": "cloud",
    "memory_reservation_percent": 15.0,
    "power_efficiency": false,
    "resilience_level": 2,
    "auto_tune": true,
    "config_id": "training_config_v1"
}
```

**Response:**
```json
{
    "optimization_result": {
        "config_id": "training_config_v1",
        "environment_detected": "cloud",
        "optimization_mode": "PERFORMANCE",
        "configurations": {
            "quantization_config": {
                "quantization_type": "MIXED",
                "enable_dynamic_quantization": true,
                "per_channel_quantization": true,
                "optimization_level": "aggressive"
            },
            "batch_processor_config": {
                "max_batch_size": 128,
                "min_batch_size": 16,
                "batch_timeout_ms": 50,
                "processing_strategy": "ADAPTIVE_BATCHING",
                "enable_priority_queue": true,
                "max_queue_size": 1000
            },
            "preprocessor_config": {
                "parallel_processing": true,
                "n_jobs": 16,
                "chunk_size": 10000,
                "enable_caching": true,
                "cache_size_mb": 512
            },
            "inference_config": {
                "enable_gpu": true,
                "gpu_memory_fraction": 0.8,
                "enable_mixed_precision": true,
                "optimize_for_throughput": true,
                "concurrent_requests": 64
            },
            "training_config": {
                "enable_distributed_training": true,
                "gradient_accumulation_steps": 4,
                "mixed_precision_training": true,
                "data_parallel": true,
                "model_parallel": false,
                "optimizer_settings": {
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "gradient_clipping": 1.0
                }
            }
        },
        "resource_allocation": {
            "cpu_cores_allocated": 14,
            "memory_allocated_gb": 27.2,
            "gpu_memory_allocated_gb": 19.2,
            "reserved_memory_gb": 4.8
        },
        "performance_estimates": {
            "expected_throughput_improvement": "40-60%",
            "memory_efficiency": "85%",
            "power_consumption": "high",
            "thermal_considerations": "active_cooling_recommended"
        },
        "optimizations_applied": [
            "GPU acceleration enabled",
            "Mixed precision training configured",
            "Optimal batch sizes calculated",
            "Memory pooling enabled",
            "NUMA-aware memory allocation",
            "CPU affinity optimization"
        ]
    },
    "config_files_created": [
        "./configs/quantization_config_training_config_v1.json",
        "./configs/batch_processor_config_training_config_v1.json",
        "./configs/preprocessor_config_training_config_v1.json"
    ],
    "optimization_time": 5.678,
    "created_at": "2025-07-16T10:30:00Z"
}
```

### Configuration Management

#### Save Configuration
```http
POST /api/device-optimizer/config/save
Content-Type: application/json
X-API-Key: your-api-key

{
    "config_id": "production_config",
    "config_path": "./configs/production",
    "configurations": {
        "quantization_config": {...},
        "batch_processor_config": {...},
        "preprocessor_config": {...}
    },
    "metadata": {
        "environment": "production",
        "workload_type": "inference",
        "created_by": "system_admin",
        "description": "Production inference configuration"
    }
}
```

**Response:**
```json
{
    "message": "Configuration saved successfully",
    "config_id": "production_config",
    "config_path": "./configs/production",
    "files_saved": [
        "./configs/production/quantization_config.json",
        "./configs/production/batch_processor_config.json",
        "./configs/production/preprocessor_config.json",
        "./configs/production/metadata.json"
    ],
    "total_size_kb": 45.6,
    "saved_at": "2025-07-16T10:30:00Z"
}
```

#### Load Configuration
```http
POST /api/device-optimizer/config/load
Content-Type: application/json
X-API-Key: your-api-key

{
    "config_path": "./configs/production",
    "config_id": "production_config"
}
```

**Response:**
```json
{
    "message": "Configuration loaded successfully",
    "config_id": "production_config",
    "configurations": {
        "quantization_config": {...},
        "batch_processor_config": {...},
        "preprocessor_config": {...}
    },
    "metadata": {
        "environment": "production",
        "workload_type": "inference",
        "created_at": "2025-07-16T10:30:00Z"
    },
    "loaded_at": "2025-07-16T10:35:00Z"
}
```

#### List Configurations
```http
GET /api/device-optimizer/config/list
X-API-Key: your-api-key
?config_path=./configs
```

**Response:**
```json
{
    "configurations": [
        {
            "config_id": "production_config",
            "environment": "production",
            "workload_type": "inference",
            "created_at": "2025-07-16T10:30:00Z",
            "size_kb": 45.6
        },
        {
            "config_id": "training_config_v1", 
            "environment": "cloud",
            "workload_type": "training",
            "created_at": "2025-07-16T09:15:00Z",
            "size_kb": 52.3
        }
    ],
    "total_configurations": 2,
    "config_path": "./configs"
}
```

#### Delete Configuration
```http
DELETE /api/device-optimizer/config/{config_id}
X-API-Key: your-api-key
?config_path=./configs
```

### Optimization Strategies

#### Optimize for Environment
```http
POST /api/device-optimizer/optimize-environment
Content-Type: application/json
X-API-Key: your-api-key

{
    "environment": "edge",                              # cloud, desktop, edge
    "constraints": {
        "max_memory_mb": 2048,
        "max_power_watts": 15,
        "thermal_limit": "passive_cooling",
        "network_bandwidth_mbps": 100
    },
    "priorities": {
        "latency": 0.8,
        "power_efficiency": 0.9,
        "accuracy": 0.7,
        "throughput": 0.3
    }
}
```

**Response:**
```json
{
    "environment_optimization": {
        "environment": "edge",
        "optimized_configs": {
            "quantization_config": {
                "quantization_type": "INT8",
                "aggressive_quantization": true,
                "model_compression": true
            },
            "batch_processor_config": {
                "max_batch_size": 8,
                "low_latency_mode": true,
                "memory_efficient_batching": true
            },
            "inference_config": {
                "cpu_only": true,
                "thread_count": 4,
                "memory_pool_size_mb": 256,
                "enable_model_caching": false
            }
        },
        "resource_allocation": {
            "memory_allocated_mb": 1536,
            "cpu_cores_used": 4,
            "power_budget_watts": 12
        },
        "performance_trade_offs": {
            "accuracy_loss_percent": 2.5,
            "latency_reduction_percent": 60,
            "power_savings_percent": 70,
            "model_size_reduction_percent": 75
        }
    }
}
```

#### Optimize for Workload
```http
POST /api/device-optimizer/optimize-workload
Content-Type: application/json
X-API-Key: your-api-key

{
    "workload_type": "inference",                       # training, inference, mixed
    "workload_characteristics": {
        "request_rate_per_second": 1000,
        "model_size_mb": 500,
        "input_size": [224, 224, 3],
        "batch_processing": true,
        "real_time_requirements": true
    },
    "performance_targets": {
        "max_latency_ms": 100,
        "min_throughput_rps": 800,
        "target_accuracy": 0.95
    }
}
```

**Response:**
```json
{
    "workload_optimization": {
        "workload_type": "inference",
        "optimized_pipeline": {
            "preprocessing": {
                "parallel_workers": 8,
                "batch_preprocessing": true,
                "cache_preprocessed_data": true
            },
            "inference": {
                "batch_size": 32,
                "gpu_inference": true,
                "tensor_rt_optimization": true,
                "dynamic_batching": true
            },
            "postprocessing": {
                "parallel_postprocessing": true,
                "result_caching": true
            }
        },
        "resource_requirements": {
            "gpu_memory_gb": 8,
            "system_memory_gb": 16,
            "cpu_cores": 8,
            "storage_iops": 1000
        },
        "performance_predictions": {
            "expected_latency_ms": 85,
            "expected_throughput_rps": 950,
            "expected_accuracy": 0.952,
            "confidence_level": 0.9
        }
    }
}
```

### Applied Configuration

#### Apply Configuration to Pipeline
```http
POST /api/device-optimizer/apply-config
Content-Type: application/json
X-API-Key: your-api-key

{
    "configs": {
        "quantization_config": {
            "quantization_type": "INT8",
            "enable_dynamic_quantization": true
        },
        "batch_processor_config": {
            "max_batch_size": 64,
            "processing_strategy": "ADAPTIVE_BATCHING"
        },
        "preprocessor_config": {
            "parallel_processing": true,
            "n_jobs": 8
        }
    }
}
```

**Response:**
```json
{
    "message": "Configuration applied successfully",
    "applied_configs": {
        "quantization_config": {
            "status": "applied",
            "changes": ["quantization_type updated", "dynamic_quantization enabled"]
        },
        "batch_processor_config": {
            "status": "applied", 
            "changes": ["batch_size increased", "strategy changed to adaptive"]
        },
        "preprocessor_config": {
            "status": "applied",
            "changes": ["parallel processing enabled", "worker count set to 8"]
        }
    },
    "pipeline_restart_required": false,
    "estimated_performance_impact": "+25% throughput improvement",
    "applied_at": "2025-07-16T10:30:00Z"
}
```

### Benchmarking and Validation

#### Benchmark Configuration
```http
POST /api/device-optimizer/benchmark
Content-Type: application/json
X-API-Key: your-api-key

{
    "config_id": "production_config",
    "benchmark_type": "performance",                    # performance, accuracy, resource_usage
    "test_data_size": 1000,
    "iterations": 10,
    "metrics": ["latency", "throughput", "memory_usage", "cpu_usage"]
}
```

**Response:**
```json
{
    "benchmark_results": {
        "config_id": "production_config",
        "benchmark_type": "performance",
        "metrics": {
            "latency": {
                "mean_ms": 45.6,
                "median_ms": 43.2,
                "p95_ms": 78.9,
                "p99_ms": 112.5,
                "std_ms": 15.3
            },
            "throughput": {
                "mean_rps": 875.3,
                "peak_rps": 950.2,
                "sustained_rps": 820.1
            },
            "resource_usage": {
                "cpu_usage_percent": 68.5,
                "memory_usage_mb": 2048.3,
                "gpu_usage_percent": 85.2,
                "gpu_memory_usage_mb": 6144.7
            }
        },
        "test_conditions": {
            "test_data_size": 1000,
            "iterations": 10,
            "duration_seconds": 120.5
        },
        "benchmark_time": "2025-07-16T10:30:00Z"
    }
}
```

#### Validate Configuration
```http
POST /api/device-optimizer/validate
Content-Type: application/json
X-API-Key: your-api-key

{
    "config_id": "production_config",
    "validation_checks": [
        "resource_constraints",
        "compatibility",
        "performance_requirements",
        "security_compliance"
    ]
}
```

**Response:**
```json
{
    "validation_results": {
        "config_id": "production_config",
        "overall_status": "valid",
        "checks": {
            "resource_constraints": {
                "status": "passed",
                "details": "All resource requirements within system limits"
            },
            "compatibility": {
                "status": "passed",
                "details": "Configuration compatible with system hardware"
            },
            "performance_requirements": {
                "status": "warning",
                "details": "Expected latency slightly above target (45ms vs 40ms target)",
                "recommendations": ["Consider reducing batch size", "Enable GPU optimization"]
            },
            "security_compliance": {
                "status": "passed",
                "details": "Security settings meet compliance requirements"
            }
        },
        "warnings": 1,
        "errors": 0,
        "validated_at": "2025-07-16T10:30:00Z"
    }
}
```

### Monitoring and Analytics

#### Get Optimization Analytics
```http
GET /api/device-optimizer/analytics
X-API-Key: your-api-key
?config_id=production_config
&time_range=24h
```

**Response:**
```json
{
    "analytics": {
        "config_id": "production_config",
        "time_range": "24h",
        "performance_metrics": {
            "average_latency_ms": 47.3,
            "throughput_trend": "+5.2% improvement",
            "resource_utilization": {
                "cpu": "68%",
                "memory": "75%",
                "gpu": "85%"
            },
            "efficiency_score": 8.7
        },
        "optimization_opportunities": [
            {
                "area": "memory_usage",
                "potential_improvement": "15% reduction",
                "recommendation": "Enable memory pooling",
                "priority": "medium"
            },
            {
                "area": "batch_processing",
                "potential_improvement": "20% throughput increase",
                "recommendation": "Increase batch size to 96",
                "priority": "high"
            }
        ],
        "alerts": [
            {
                "type": "performance_degradation",
                "severity": "low",
                "message": "Latency increased by 8% in last hour",
                "timestamp": "2025-07-16T09:30:00Z"
            }
        ]
    }
}
```

#### Health Check
```http
GET /api/device-optimizer/health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-07-16T10:30:00Z",
    "checks": {
        "system_detection": "healthy",
        "configuration_system": "healthy",
        "optimization_engine": "healthy",
        "file_system": "healthy"
    },
    "uptime": "7d 14h 22m",
    "active_configurations": 5,
    "last_optimization": "2025-07-16T10:25:00Z"
}
```

## Usage Examples

### Basic System Optimization

```python
import requests
import json

# API configuration
API_BASE = "http://localhost:8006"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# 1. Get system information
system_info_request = {
    "enable_specialized_accelerators": True
}
response = requests.post(f"{API_BASE}/api/device-optimizer/system-info",
                        headers=headers, json=system_info_request)
system_info = response.json()

print("System detected:")
print(f"CPU: {system_info['system_info']['cpu']['model']}")
print(f"Memory: {system_info['system_info']['memory']['total_gb']}GB")
print(f"GPU: {system_info['system_info']['gpu'][0]['name']}")

# 2. Get optimized configuration
optimization_request = {
    "optimization_mode": "PERFORMANCE",
    "workload_type": "training",
    "environment": "auto",
    "memory_reservation_percent": 10.0,
    "auto_tune": True,
    "config_id": "my_training_config"
}
response = requests.post(f"{API_BASE}/api/device-optimizer/optimize",
                        headers=headers, json=optimization_request)
optimization_result = response.json()

print("\nOptimized configuration:")
print(f"Config ID: {optimization_result['optimization_result']['config_id']}")
print(f"Environment: {optimization_result['optimization_result']['environment_detected']}")
print(f"Expected improvement: {optimization_result['optimization_result']['performance_estimates']['expected_throughput_improvement']}")

# 3. Apply configuration
apply_request = {
    "configs": optimization_result['optimization_result']['configurations']
}
response = requests.post(f"{API_BASE}/api/device-optimizer/apply-config",
                        headers=headers, json=apply_request)
apply_result = response.json()

print(f"\nConfiguration applied: {apply_result['message']}")
print(f"Performance impact: {apply_result['estimated_performance_impact']}")
```

### Environment-Specific Optimization

```python
# Edge computing optimization
edge_optimization = {
    "environment": "edge",
    "constraints": {
        "max_memory_mb": 4096,
        "max_power_watts": 25,
        "thermal_limit": "passive_cooling"
    },
    "priorities": {
        "latency": 0.9,
        "power_efficiency": 0.8,
        "accuracy": 0.6,
        "throughput": 0.4
    }
}
response = requests.post(f"{API_BASE}/api/device-optimizer/optimize-environment",
                        headers=headers, json=edge_optimization)
edge_config = response.json()

print("Edge optimization results:")
print(f"Power savings: {edge_config['environment_optimization']['performance_trade_offs']['power_savings_percent']}%")
print(f"Latency reduction: {edge_config['environment_optimization']['performance_trade_offs']['latency_reduction_percent']}%")
print(f"Model size reduction: {edge_config['environment_optimization']['performance_trade_offs']['model_size_reduction_percent']}%")

# Cloud optimization
cloud_optimization = {
    "environment": "cloud",
    "constraints": {
        "max_memory_mb": 32768,
        "max_power_watts": 500,
        "thermal_limit": "active_cooling"
    },
    "priorities": {
        "throughput": 0.9,
        "accuracy": 0.8,
        "latency": 0.6,
        "power_efficiency": 0.3
    }
}
response = requests.post(f"{API_BASE}/api/device-optimizer/optimize-environment",
                        headers=headers, json=cloud_optimization)
cloud_config = response.json()

print("\nCloud optimization results:")
print("Optimized for maximum throughput and accuracy")
```

### Workload-Specific Optimization

```python
# High-throughput inference optimization
inference_optimization = {
    "workload_type": "inference",
    "workload_characteristics": {
        "request_rate_per_second": 2000,
        "model_size_mb": 1000,
        "input_size": [512, 512, 3],
        "batch_processing": True,
        "real_time_requirements": False
    },
    "performance_targets": {
        "min_throughput_rps": 1500,
        "max_latency_ms": 200,
        "target_accuracy": 0.95
    }
}
response = requests.post(f"{API_BASE}/api/device-optimizer/optimize-workload",
                        headers=headers, json=inference_optimization)
workload_config = response.json()

print("Inference workload optimization:")
print(f"Expected throughput: {workload_config['workload_optimization']['performance_predictions']['expected_throughput_rps']} RPS")
print(f"Expected latency: {workload_config['workload_optimization']['performance_predictions']['expected_latency_ms']} ms")
print(f"GPU memory required: {workload_config['workload_optimization']['resource_requirements']['gpu_memory_gb']} GB")

# Training workload optimization
training_optimization = {
    "workload_type": "training",
    "workload_characteristics": {
        "dataset_size_gb": 100,
        "model_size_mb": 500,
        "batch_size": 64,
        "epochs": 100,
        "distributed_training": True
    },
    "performance_targets": {
        "training_time_hours": 24,
        "target_accuracy": 0.97,
        "memory_efficiency": 0.8
    }
}
response = requests.post(f"{API_BASE}/api/device-optimizer/optimize-workload",
                        headers=headers, json=training_optimization)
training_config = response.json()

print("\nTraining workload optimization:")
print("Distributed training configuration generated")
```

### Configuration Management

```python
# Save configuration for reuse
save_config_request = {
    "config_id": "production_inference_v2",
    "config_path": "./configs/production",
    "configurations": optimization_result['optimization_result']['configurations'],
    "metadata": {
        "environment": "production",
        "workload_type": "inference",
        "created_by": "ml_engineer",
        "description": "Optimized configuration for production inference workload"
    }
}
response = requests.post(f"{API_BASE}/api/device-optimizer/config/save",
                        headers=headers, json=save_config_request)
save_result = response.json()

print(f"Configuration saved: {save_result['config_id']}")
print(f"Files created: {len(save_result['files_saved'])}")

# List all configurations
response = requests.get(f"{API_BASE}/api/device-optimizer/config/list",
                       headers={"X-API-Key": API_KEY},
                       params={"config_path": "./configs"})
config_list = response.json()

print(f"\nAvailable configurations: {config_list['total_configurations']}")
for config in config_list['configurations']:
    print(f"- {config['config_id']}: {config['workload_type']} ({config['environment']})")

# Load a specific configuration
load_config_request = {
    "config_path": "./configs/production", 
    "config_id": "production_inference_v2"
}
response = requests.post(f"{API_BASE}/api/device-optimizer/config/load",
                        headers=headers, json=load_config_request)
loaded_config = response.json()

print(f"\nLoaded configuration: {loaded_config['config_id']}")
```

### Benchmarking and Validation

```python
# Benchmark configuration performance
benchmark_request = {
    "config_id": "production_inference_v2",
    "benchmark_type": "performance",
    "test_data_size": 1000,
    "iterations": 10,
    "metrics": ["latency", "throughput", "memory_usage", "cpu_usage"]
}
response = requests.post(f"{API_BASE}/api/device-optimizer/benchmark",
                        headers=headers, json=benchmark_request)
benchmark_results = response.json()

print("Benchmark results:")
print(f"Mean latency: {benchmark_results['benchmark_results']['metrics']['latency']['mean_ms']} ms")
print(f"Mean throughput: {benchmark_results['benchmark_results']['metrics']['throughput']['mean_rps']} RPS")
print(f"CPU usage: {benchmark_results['benchmark_results']['metrics']['resource_usage']['cpu_usage_percent']}%")

# Validate configuration
validate_request = {
    "config_id": "production_inference_v2",
    "validation_checks": [
        "resource_constraints",
        "compatibility", 
        "performance_requirements",
        "security_compliance"
    ]
}
response = requests.post(f"{API_BASE}/api/device-optimizer/validate",
                        headers=headers, json=validate_request)
validation_results = response.json()

print(f"\nValidation status: {validation_results['validation_results']['overall_status']}")
print(f"Warnings: {validation_results['validation_results']['warnings']}")
print(f"Errors: {validation_results['validation_results']['errors']}")

for check, result in validation_results['validation_results']['checks'].items():
    print(f"- {check}: {result['status']}")
```

### Monitoring and Analytics

```python
# Get optimization analytics
response = requests.get(f"{API_BASE}/api/device-optimizer/analytics",
                       headers={"X-API-Key": API_KEY},
                       params={
                           "config_id": "production_inference_v2",
                           "time_range": "24h"
                       })
analytics = response.json()

print("Analytics summary:")
print(f"Average latency: {analytics['analytics']['performance_metrics']['average_latency_ms']} ms")
print(f"Throughput trend: {analytics['analytics']['performance_metrics']['throughput_trend']}")
print(f"Efficiency score: {analytics['analytics']['performance_metrics']['efficiency_score']}/10")

print("\nOptimization opportunities:")
for opportunity in analytics['analytics']['optimization_opportunities']:
    print(f"- {opportunity['area']}: {opportunity['potential_improvement']} ({opportunity['priority']} priority)")
    print(f"  Recommendation: {opportunity['recommendation']}")

# Check system health
response = requests.get(f"{API_BASE}/api/device-optimizer/health")
health = response.json()

print(f"\nSystem health: {health['status']}")
print(f"Uptime: {health['uptime']}")
print(f"Active configurations: {health['active_configurations']}")
```

## Error Handling

### Common Error Codes

- **400 Bad Request**: Invalid optimization parameters or configuration format
- **401 Unauthorized**: Missing or invalid API key
- **404 Not Found**: Configuration or resource not found
- **409 Conflict**: Configuration conflict or resource unavailable
- **422 Unprocessable Entity**: System detection or optimization errors
- **500 Internal Server Error**: System detection or optimization engine errors

### Error Response Format

```json
{
    "error": "OptimizationError",
    "message": "Unable to optimize for specified constraints",
    "details": {
        "constraint": "max_memory_mb",
        "specified_value": 1024,
        "minimum_required": 2048,
        "recommendation": "Increase memory constraint or reduce model complexity"
    },
    "timestamp": "2025-07-16T10:30:00Z",
    "request_id": "req_12345"
}
```

## Best Practices

### System Detection

1. **Regular Updates**: Refresh system information periodically
2. **Comprehensive Detection**: Enable specialized accelerator detection
3. **Performance Profiling**: Benchmark system capabilities regularly
4. **Resource Monitoring**: Monitor resource utilization continuously

### Configuration Management

1. **Version Control**: Maintain configuration versions and metadata
2. **Environment Separation**: Use different configs for dev/staging/prod
3. **Documentation**: Document configuration purposes and trade-offs
4. **Validation**: Always validate configurations before deployment

### Optimization Strategy

1. **Iterative Optimization**: Start with balanced mode, then fine-tune
2. **Benchmarking**: Benchmark all optimizations thoroughly
3. **Trade-off Analysis**: Understand performance vs. resource trade-offs
4. **Monitoring**: Continuously monitor optimized systems

### Production Deployment

1. **Gradual Rollout**: Deploy optimizations gradually
2. **A/B Testing**: Compare optimized vs. baseline configurations
3. **Rollback Plans**: Have rollback strategies for failed optimizations
4. **Performance Monitoring**: Monitor production performance continuously

## Advanced Features

### Custom Optimization Algorithms

Support for custom optimization strategies:

```python
custom_optimizer = {
    "algorithm": "genetic_algorithm",
    "parameters": {
        "population_size": 50,
        "generations": 100,
        "mutation_rate": 0.1
    }
}
```

### Multi-Objective Optimization

Optimize for multiple objectives simultaneously:

```python
multi_objective = {
    "objectives": [
        {"metric": "latency", "weight": 0.4, "minimize": True},
        {"metric": "throughput", "weight": 0.4, "minimize": False},
        {"metric": "power_consumption", "weight": 0.2, "minimize": True}
    ]
}
```

### Dynamic Optimization

Runtime optimization based on current system state:

```python
dynamic_config = {
    "enable_runtime_optimization": True,
    "optimization_interval_minutes": 30,
    "adaptation_threshold": 0.1
}
```

## Related Documentation

- [Device Optimizer](../device_optimizer.md) - Core device optimizer
- [Configuration System](../configs.md) - Configuration management
- [Quantizer API](quantizer_api.md) - Model quantization API
- [Inference Engine API](inference_engine_api.md) - Model inference API
- [Performance Metrics](../engine/performance_metrics.md) - Performance monitoring

---

*The Device Optimizer API provides comprehensive hardware-aware optimization for ML workloads with automatic system detection, intelligent configuration generation, and continuous performance monitoring.*
