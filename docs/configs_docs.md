# Configs Module Documentation

This document provides detailed information about the configuration components used for machine learning model training, preprocessing, inference, and batch processing.

## Table of Contents

1. [Overview](#overview)
2. [Constants](#constants)
3. [Quantization Configuration](#quantization-configuration)
4. [Batch Processing Configuration](#batch-processing-configuration)
5. [Data Preprocessor Configuration](#data-preprocessor-configuration)
6. [Inference Engine Configuration](#inference-engine-configuration)
7. [Training Engine Configuration](#training-engine-configuration)

## Overview

The `configs.py` module provides a comprehensive set of configuration classes that control the behavior of various components in the machine learning pipeline. Each configuration class is designed to be modular, customizable, and serializable for easy persistence and sharing.

## Constants

```python
CHECKPOINT_PATH = "./checkpoints"
MODEL_REGISTRY_PATH = "./model_registry"
```

These constants define default paths for model checkpoints and the model registry.

## Quantization Configuration

Quantization configurations control how numerical values are quantized for improved performance and reduced memory usage.

### `QuantizationType` (Enum)

Defines supported quantization data types:

- `INT8`: 8-bit signed integer quantization
- `UINT8`: 8-bit unsigned integer quantization
- `INT16`: 16-bit signed integer quantization

### `QuantizationMode` (Enum)

Defines supported quantization modes:

- `SYMMETRIC`: Symmetric quantization around zero
- `ASYMMETRIC`: Asymmetric quantization with separate zero point
- `DYNAMIC_PER_BATCH`: Dynamic quantization applied per batch
- `DYNAMIC_PER_CHANNEL`: Dynamic quantization applied per channel

### `QuantizationConfig` (Dataclass)

Configuration for controlling quantization behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantization_type` | str | "int8" | Type of quantization to use |
| `quantization_mode` | str | "dynamic_per_batch" | Mode of quantization |
| `enable_cache` | bool | True | Whether to cache quantization results |
| `cache_size` | int | 1024 | Size of the quantization cache |
| `buffer_size` | int | 0 | Size of the buffer (0 means no buffer) |
| `use_percentile` | bool | False | Whether to use percentile for range calculation |
| `min_percentile` | float | 0.1 | Minimum percentile for range calculation |
| `max_percentile` | float | 99.9 | Maximum percentile for range calculation |
| `error_on_nan` | bool | False | Whether to raise an error on NaN values |
| `error_on_inf` | bool | False | Whether to raise an error on infinity values |
| `outlier_threshold` | Optional[float] | None | Threshold for handling outliers |
| `num_bits` | int | 8 | Number of bits for quantization |
| `optimize_memory` | bool | True | Whether to optimize memory usage |

#### Example Usage

```python
from configs import QuantizationConfig, QuantizationType, QuantizationMode

# Create a default quantization configuration
default_config = QuantizationConfig()

# Create a custom quantization configuration
custom_config = QuantizationConfig(
    quantization_type=QuantizationType.INT16.value,
    quantization_mode=QuantizationMode.SYMMETRIC.value,
    use_percentile=True,
    min_percentile=1.0,
    max_percentile=99.0,
    num_bits=16,
    cache_size=2048
)
```

## Batch Processing Configuration

Configurations for controlling how data is processed in batches.

### `BatchProcessingStrategy` (Enum)

Defines strategies for batch processing:

- `FIXED`: Use a fixed batch size
- `ADAPTIVE`: Dynamically adjust batch size based on system load
- `GREEDY`: Process as many items as available up to maximum batch size

### `BatchPriority` (Enum)

Defines priority levels for batch processing:

- `CRITICAL`: Highest priority (0)
- `HIGH`: High priority (1)
- `NORMAL`: Default priority (2)
- `LOW`: Low priority (3)
- `BACKGROUND`: Lowest priority (4)

### `PrioritizedItem` (NamedTuple)

Represents an item with priority information for queue ordering.

| Field | Type | Description |
|-------|------|-------------|
| `priority` | int | Priority level (lower number = higher priority) |
| `timestamp` | float | Time when item was added |
| `item` | Any | The actual item to process |

### `BatchProcessorConfig` (Dataclass)

Configuration for controlling batch processing behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_batch_size` | int | 1 | Minimum batch size |
| `max_batch_size` | int | 64 | Maximum batch size |
| `initial_batch_size` | int | 16 | Initial batch size |
| `max_queue_size` | int | 1000 | Maximum queue size |
| `enable_priority_queue` | bool | False | Whether to enable priority-based queuing |
| `batch_timeout` | float | 0.1 | Maximum time to wait for batch formation (seconds) |
| `item_timeout` | float | 10.0 | Maximum time to wait for single item processing (seconds) |
| `min_batch_interval` | float | 0.0 | Minimum time between batch processing (seconds) |
| `processing_strategy` | BatchProcessingStrategy | ADAPTIVE | Strategy for batch processing |
| `enable_adaptive_batching` | bool | True | Whether to adaptively adjust batch size |
| `max_retries` | int | 2 | Maximum number of retries for failed batches |
| `retry_delay` | float | 0.1 | Delay between retries (seconds) |
| `reduce_batch_on_failure` | bool | True | Whether to reduce batch size after failure |
| `max_batch_memory_mb` | Optional[float] | None | Maximum memory per batch in MB |
| `enable_memory_optimization` | bool | True | Whether to optimize memory usage |
| `gc_batch_threshold` | int | 32 | Run garbage collection after processing batches larger than this |
| `enable_monitoring` | bool | True | Whether to enable monitoring |
| `monitoring_window` | int | 100 | Number of batches to keep statistics for |
| `max_workers` | int | 4 | Maximum number of worker threads |
| `enable_health_monitoring` | bool | True | Whether to monitor system health |
| `health_check_interval` | float | 5.0 | Interval between health checks (seconds) |
| `memory_warning_threshold` | float | 70.0 | Memory usage percentage to trigger warning |
| `memory_critical_threshold` | float | 85.0 | Memory usage percentage to trigger critical actions |
| `queue_warning_threshold` | int | 100 | Queue size to trigger warning |
| `queue_critical_threshold` | int | 500 | Queue size to trigger critical actions |
| `debug_mode` | bool | False | Whether to enable debug mode |

#### Example Usage

```python
from configs import BatchProcessorConfig, BatchProcessingStrategy

# Create a default batch processor configuration
default_config = BatchProcessorConfig()

# Create a high-throughput batch processor configuration
high_throughput_config = BatchProcessorConfig(
    min_batch_size=32,
    max_batch_size=256,
    initial_batch_size=64,
    processing_strategy=BatchProcessingStrategy.GREEDY,
    max_workers=8,
    enable_priority_queue=True,
    gc_batch_threshold=64
)

# Create a low-latency batch processor configuration
low_latency_config = BatchProcessorConfig(
    min_batch_size=1,
    max_batch_size=16,
    batch_timeout=0.05,
    processing_strategy=BatchProcessingStrategy.FIXED,
    enable_adaptive_batching=False
)
```

## Data Preprocessor Configuration

Configurations for controlling data preprocessing operations.

### `NormalizationType` (Enum)

Defines supported normalization strategies:

- `NONE`: No normalization
- `STANDARD`: Z-score normalization ((x - mean) / std)
- `MINMAX`: Min-max scaling ((x - min) / (max - min))
- `ROBUST`: Robust scaling using percentiles
- `CUSTOM`: Custom normalization function

### `PreprocessorConfig` (Dataclass)

Configuration for controlling data preprocessing behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalization` | NormalizationType | STANDARD | Normalization strategy |
| `robust_percentiles` | Tuple[float, float] | (25.0, 75.0) | Percentiles for robust scaling |
| `handle_nan` | bool | True | Whether to handle NaN values |
| `handle_inf` | bool | True | Whether to handle infinity values |
| `nan_strategy` | str | "mean" | Strategy for handling NaN values |
| `inf_strategy` | str | "mean" | Strategy for handling infinity values |
| `detect_outliers` | bool | False | Whether to detect outliers |
| `outlier_method` | str | "iqr" | Method for outlier detection |
| `outlier_params` | Dict[str, Any] | {...} | Parameters for outlier detection |
| `clip_values` | bool | False | Whether to clip values |
| `clip_range` | Tuple[float, float] | (-np.inf, np.inf) | Range for clipping values |
| `enable_input_validation` | bool | True | Whether to validate input data |
| `input_size_limit` | Optional[int] | None | Maximum number of samples |
| `parallel_processing` | bool | False | Whether to use parallel processing |
| `n_jobs` | int | -1 | Number of jobs for parallel processing |
| `chunk_size` | Optional[int] | None | Size of chunks for processing large data |
| `cache_enabled` | bool | True | Whether to enable caching |
| `cache_size` | int | 128 | LRU cache size |
| `dtype` | np.dtype | np.float64 | Data type for numerical precision |
| `epsilon` | float | 1e-10 | Small value to avoid division by zero |
| `debug_mode` | bool | False | Whether to enable debug mode |
| `custom_normalization_fn` | Optional[Callable] | None | Custom normalization function |
| `custom_transform_fn` | Optional[Callable] | None | Custom transformation function |
| `version` | str | "1.0.0" | Version information |

#### Example Usage

```python
from configs import PreprocessorConfig, NormalizationType
import numpy as np

# Create a default preprocessor configuration
default_config = PreprocessorConfig()

# Create a robust preprocessing configuration
robust_config = PreprocessorConfig(
    normalization=NormalizationType.ROBUST,
    robust_percentiles=(10.0, 90.0),
    handle_nan=True,
    handle_inf=True,
    nan_strategy="median",
    inf_strategy="median",
    detect_outliers=True,
    outlier_method="isolation_forest",
    clip_values=True,
    clip_range=(-10.0, 10.0),
    parallel_processing=True,
    n_jobs=4,
    dtype=np.float32
)
```

## Inference Engine Configuration

Configurations for controlling model inference operations.

### `ModelType` (Enum)

Defines supported model types:

- `SKLEARN`: scikit-learn models
- `XGBOOST`: XGBoost models
- `LIGHTGBM`: LightGBM models
- `CUSTOM`: Custom models
- `ENSEMBLE`: Ensemble models

### `EngineState` (Enum)

Defines possible states of the inference engine:

- `INITIALIZING`: Engine is initializing
- `READY`: Engine is ready for inference
- `RUNNING`: Engine is currently running inference
- `LOADING`: Engine is loading a model
- `STOPPING`: Engine is in the process of stopping
- `STOPPED`: Engine is stopped
- `ERROR`: Engine encountered an error

### `InferenceEngineConfig` (Dataclass)

Configuration for controlling inference engine behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_version` | str | "1.0" | Model version |
| `debug_mode` | bool | False | Whether to enable debug mode |
| `num_threads` | int | 4 | Number of threads for inference |
| `set_cpu_affinity` | bool | False | Whether to set CPU affinity |
| `enable_intel_optimization` | bool | False | Whether to enable Intel optimizations |
| `enable_quantization` | bool | False | Whether to enable quantization |
| `enable_model_quantization` | bool | False | Whether to quantize the model |
| `enable_input_quantization` | bool | False | Whether to quantize inputs |
| `quantization_dtype` | str | "int8" | Data type for quantization |
| `quantization_config` | Optional[QuantizationConfig] | None | Configuration for quantization |
| `enable_request_deduplication` | bool | True | Whether to deduplicate requests |
| `max_cache_entries` | int | 1000 | Maximum number of cache entries |
| `cache_ttl_seconds` | int | 300 | Time-to-live for cache entries (seconds) |
| `monitoring_window` | int | 100 | Number of requests to keep statistics for |
| `enable_monitoring` | bool | True | Whether to enable monitoring |
| `monitoring_interval` | float | 10.0 | Interval between monitoring checks (seconds) |
| `throttle_on_high_cpu` | bool | True | Whether to throttle on high CPU usage |
| `cpu_threshold_percent` | float | 90.0 | CPU usage threshold in percent |
| `memory_high_watermark_mb` | float | 1024.0 | Memory high watermark in MB |
| `memory_limit_gb` | Optional[float] | None | Absolute memory limit in GB |
| `enable_batching` | bool | True | Whether to enable batching |
| `batch_processing_strategy` | str | "adaptive" | Strategy for batch processing |
| `batch_timeout` | float | 0.1 | Timeout for batch formation (seconds) |
| `max_concurrent_requests` | int | 8 | Maximum number of concurrent requests |
| `initial_batch_size` | int | 16 | Initial batch size |
| `min_batch_size` | int | 1 | Minimum batch size |
| `max_batch_size` | int | 64 | Maximum batch size |
| `enable_adaptive_batching` | bool | True | Whether to adaptively adjust batch size |
| `enable_memory_optimization` | bool | True | Whether to optimize memory usage |
| `enable_feature_scaling` | bool | False | Whether to enable feature scaling |
| `enable_warmup` | bool | True | Whether to perform warmup |
| `enable_quantization_aware_inference` | bool | False | Whether to use quantization-aware inference |
| `enable_throttling` | bool | False | Whether to enable throttling |

#### Example Usage

```python
from configs import InferenceEngineConfig, QuantizationConfig

# Create a default inference engine configuration
default_config = InferenceEngineConfig()

# Create a high-performance inference engine configuration
high_performance_config = InferenceEngineConfig(
    num_threads=8,
    set_cpu_affinity=True,
    enable_intel_optimization=True,
    enable_quantization=True,
    enable_model_quantization=True,
    enable_batching=True,
    max_batch_size=128,
    enable_adaptive_batching=True,
    enable_warmup=True,
    quantization_config=QuantizationConfig(
        quantization_mode="dynamic_per_channel",
        optimize_memory=True
    )
)
```

## Training Engine Configuration

Configurations for controlling model training operations.

### `TaskType` (Enum)

Defines supported machine learning task types:

- `CLASSIFICATION`: Classification tasks
- `REGRESSION`: Regression tasks
- `CLUSTERING`: Clustering tasks
- `ANOMALY_DETECTION`: Anomaly detection tasks

### `OptimizationStrategy` (Enum)

Defines supported hyperparameter optimization strategies:

- `GRID_SEARCH`: Grid search optimization
- `RANDOM_SEARCH`: Random search optimization
- `BAYESIAN_OPTIMIZATION`: Bayesian optimization
- `EVOLUTIONARY`: Evolutionary algorithm optimization
- `HYPERBAND`: Hyperband optimization

### `MLTrainingEngineConfig` (Class)

Configuration for controlling machine learning training engine behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | TaskType | CLASSIFICATION | Type of machine learning task |
| `random_state` | int | 42 | Random seed for reproducibility |
| `n_jobs` | int | -1 | Number of jobs for parallel processing |
| `verbose` | int | 1 | Verbosity level |
| `cv_folds` | int | 5 | Number of cross-validation folds |
| `test_size` | float | 0.2 | Proportion of data to use for testing |
| `stratify` | bool | True | Whether to use stratified sampling |
| `optimization_strategy` | OptimizationStrategy | RANDOM_SEARCH | Strategy for hyperparameter optimization |
| `optimization_iterations` | int | 50 | Number of optimization iterations |
| `early_stopping` | bool | True | Whether to use early stopping |
| `feature_selection` | bool | True | Whether to perform feature selection |
| `feature_selection_method` | str | "mutual_info" | Method for feature selection |
| `feature_selection_k` | Optional[int] | None | Number of features to select |
| `feature_importance_threshold` | float | 0.01 | Threshold for feature importance |
| `preprocessing_config` | Optional[PreprocessorConfig] | PreprocessorConfig(...) | Configuration for preprocessing |
| `batch_processing_config` | Optional[BatchProcessorConfig] | BatchProcessorConfig(...) | Configuration for batch processing |
| `inference_config` | Optional[InferenceEngineConfig] | InferenceEngineConfig(...) | Configuration for inference |
| `quantization_config` | Optional[QuantizationConfig] | QuantizationConfig(...) | Configuration for quantization |
| `model_path` | str | "./models" | Path to save models |
| `experiment_tracking` | bool | True | Whether to track experiments |
| `use_intel_optimization` | bool | True | Whether to use Intel optimizations |
| `memory_optimization` | bool | True | Whether to optimize memory usage |
| `enable_distributed` | bool | False | Whether to enable distributed training |
| `log_level` | str | "INFO" | Logging level |

#### Example Usage

```python
from configs import (
    MLTrainingEngineConfig, 
    TaskType, 
    OptimizationStrategy,
    PreprocessorConfig,
    NormalizationType
)

# Create a default training engine configuration
default_config = MLTrainingEngineConfig()

# Create a configuration for a regression task with Bayesian optimization
regression_config = MLTrainingEngineConfig(
    task_type=TaskType.REGRESSION,
    optimization_strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
    optimization_iterations=100,
    cv_folds=10,
    feature_selection_method="recursive_elimination",
    preprocessing_config=PreprocessorConfig(
        normalization=NormalizationType.ROBUST,
        detect_outliers=True
    ),
    experiment_tracking=True,
    enable_distributed=True
)
```