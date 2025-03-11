# DeviceOptimizer Documentation

## Overview

The `DeviceOptimizer` is a utility class that automatically configures machine learning pipeline settings based on the capabilities of the host device. It analyzes system resources such as CPU, memory, and architecture to generate optimized configurations for various components of an ML pipeline.

## Key Features

- Automatic detection of system capabilities (CPU, memory, architecture)
- Optimization of quantization settings based on available resources
- Configuration of batch processing parameters for efficient data handling
- Customization of preprocessing settings for optimal performance
- Tuning of inference and training engine parameters
- Serialization and storage of configurations for reproducibility

## Class: DeviceOptimizer

### Initialization

```python
DeviceOptimizer(
    config_path: str = "./configs",
    checkpoint_path: str = "./checkpoints",
    model_registry_path: str = "./model_registry"
)
```

**Parameters:**
- `config_path`: Directory to save configuration files
- `checkpoint_path`: Directory for model checkpoints
- `model_registry_path`: Directory for model registry

### Methods

#### `get_optimal_quantization_config()`

Creates an optimized quantization configuration based on device capabilities.

**Returns:** `QuantizationConfig` object with settings for:
- Quantization type (INT8, etc.)
- Quantization mode (dynamic per batch/channel)
- Cache and buffer sizes
- Percentile settings
- Error handling options

#### `get_optimal_batch_processor_config()`

Creates an optimized batch processor configuration based on device capabilities.

**Returns:** `BatchProcessorConfig` object with settings for:
- Batch size limits (min, max, initial)
- Queue sizes and thresholds
- Processing strategies
- Memory optimization parameters
- Worker count
- Health monitoring thresholds

#### `get_optimal_preprocessor_config()`

Creates an optimized preprocessor configuration based on device capabilities.

**Returns:** `PreprocessorConfig` object with settings for:
- Normalization type
- NaN/Inf handling strategies
- Outlier detection parameters
- Parallelization settings
- Cache configuration
- Data type optimization

#### `get_optimal_inference_engine_config()`

Creates an optimized inference engine configuration based on device capabilities.

**Returns:** `InferenceEngineConfig` object with settings for:
- Thread count and CPU affinity
- Hardware-specific optimizations
- Quantization settings
- Caching parameters
- Batching configuration
- Memory thresholds
- Monitoring options

#### `get_optimal_training_engine_config()`

Creates an optimized training engine configuration based on device capabilities.

**Returns:** `MLTrainingEngineConfig` object with settings for:
- Task type and parallelization
- Cross-validation settings
- Optimization strategy and iterations
- Feature selection parameters
- Integration with other configurations
- Memory optimization options
- Distributed training settings

#### `save_configs(config_id: Optional[str] = None)`

Generates and saves all optimized configurations to disk.

**Parameters:**
- `config_id`: Optional identifier for the configuration set (UUID generated if not provided)

**Returns:** Dictionary with paths to saved configuration files

#### `_serialize_config_dict(config_dict)`

Helper method to convert Enum values to strings for JSON serialization.

**Parameters:**
- `config_dict`: Dictionary that may contain Enum values

**Returns:** Dictionary with Enum values converted to strings

## Utility Functions

### `load_config(config_path: Union[str, Path])`

Loads a configuration from a JSON file.

**Parameters:**
- `config_path`: Path to the configuration file

**Returns:** Configuration as a dictionary

### `safe_dict_serializer(obj, ignore_types=None, max_depth=10, current_depth=0)`

Converts an object to a serializable dictionary, handling non-serializable types.

**Parameters:**
- `obj`: The object to convert
- `ignore_types`: List of types to ignore (replaced with string representation)
- `max_depth`: Maximum recursion depth
- `current_depth`: Current recursion depth (used internally)

**Returns:** A JSON-serializable representation of the object

### `save_serializable_json(obj, file_path, indent=2)`

Saves an object to a JSON file, handling non-serializable types.

**Parameters:**
- `obj`: The object to save
- `file_path`: Path to the output JSON file
- `indent`: Indentation level for the JSON file

### `create_optimized_configs(config_path, checkpoint_path, model_registry_path, config_id)`

Creates optimized configurations based on the current device.

**Parameters:**
- `config_path`: Path to save configuration files
- `checkpoint_path`: Path for model checkpoints
- `model_registry_path`: Path for model registry
- `config_id`: Optional identifier for the configuration set

**Returns:** Dictionary with paths to saved configuration files

## Usage Example

```python
# Create a device optimizer with default paths
optimizer = DeviceOptimizer()

# Get optimized quantization configuration
quant_config = optimizer.get_optimal_quantization_config()

# Get optimized batch processor configuration
batch_config = optimizer.get_optimal_batch_processor_config()

# Save all configurations to disk
master_config = optimizer.save_configs("my_config_v1")

# Or use the utility function to create and save all configs at once
master_config = create_optimized_configs(
    config_path="./my_configs",
    checkpoint_path="./my_checkpoints",
    model_registry_path="./my_models",
    config_id="production_v1"
)
```

## Dependencies

- `os`, `platform`, `psutil`: System information gathering
- `json`, `logging`: Data serialization and logging
- `numpy`: Numerical operations
- `pathlib`: Path handling
- `typing`: Type annotations
- `multiprocessing`, `socket`, `uuid`: System utilities
- `dataclasses`: Data structure utilities
- `enum`: Enumeration support

## Notes

- The optimizer detects Intel CPUs and AVX/AVX2 support for specialized optimizations
- Memory usage is carefully managed based on available system resources
- Configurations are saved as JSON files for easy loading and sharing
- A master configuration file links to all individual configuration files