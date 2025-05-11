# Quantizer Module Documentation

## Overview
The `Quantizer` module provides a high-performance, thread-safe implementation for numerical data quantization in machine learning applications. It supports various quantization types (int8, uint8, int16, float16), modes (symmetric, asymmetric, per-channel), and includes advanced features like caching, calibration, mixed-precision quantization, and performance optimizations.

This module is particularly useful for model compression, inference acceleration, and memory optimization in deep learning applications where reducing the precision of numerical data while preserving critical information is essential.

## Prerequisites
- Python ≥3.6
- Required packages:
  ```bash
  pip install numpy>=1.19.0 logging>=0.5.0
  ```
- Custom dependencies from `modules.configs` (QuantizationConfig, QuantizationType, QuantizationMode)

## Installation
The module is likely part of a larger package. Install the entire package:
```bash
pip install -e .
```

## Usage
```python
import numpy as np
from modules.configs import QuantizationConfig, QuantizationType, QuantizationMode
from quantizer import Quantizer

# Create with default configuration
quantizer = Quantizer()

# Create with custom configuration
config = QuantizationConfig(
    quantization_type=QuantizationType.INT8,
    quantization_mode=QuantizationMode.SYMMETRIC,
    num_bits=8,
    enable_cache=True
)
quantizer = Quantizer(config)

# Quantize data
data = np.random.randn(100, 100).astype(np.float32)
quantized_data = quantizer.quantize(data)

# Dequantize back to floating-point
dequantized_data = quantizer.dequantize(quantized_data)

# Calibrate quantizer with representative data
calibration_datasets = [np.random.randn(50, 50).astype(np.float32) for _ in range(5)]
quantizer.calibrate(calibration_datasets)

# Export parameters for later use
params = quantizer.export_import_parameters()

# Update configuration
new_config = QuantizationConfig(quantization_type=QuantizationType.INT16)
quantizer.update_config(new_config)
```

## Configuration
The quantizer is configured through the `QuantizationConfig` class with these primary parameters:

| Parameter | Description |
|-----------|-------------|
| `quantization_type` | Type of quantization (INT8, UINT8, INT16, FLOAT16, NONE, MIXED) |
| `quantization_mode` | Mode of quantization (SYMMETRIC, ASYMMETRIC, DYNAMIC, CALIBRATED, etc.) |
| `num_bits` | Number of bits to use for quantization (typically 8 or 16) |
| `symmetric` | Whether to use symmetric quantization around zero |
| `enable_cache` | Enable value caching for repeated dequantization |
| `cache_size` | Maximum number of entries in the cache |
| `buffer_size` | Size of preallocated buffers for performance |
| `enable_mixed_precision` | Enable mixed precision quantization |
| `per_channel` | Enable per-channel quantization |

## Architecture
The Quantizer uses a general workflow of:
1. **Configuration**: Sets up quantization parameters, ranges, and buffers
2. **Calibration** (optional): Analyzes representative data to determine optimal scales
3. **Quantization**: Converts float32 data to lower precision (INT8, UINT8, INT16, FLOAT16)
4. **Dequantization**: Converts quantized data back to float32
5. **Optimization**: Employs caching, threading, and vectorization for performance

## Class: Quantizer

```python
class Quantizer:
```

### Description
High-performance data quantizer that converts between floating-point and quantized integer representations of numerical data. Features thread-safety, caching, calibration, and various quantization modes.

### Attributes
- `config (QuantizationConfig)`: Configuration settings for quantization
- `_lock (threading.RLock)`: Reentrant lock for thread safety
- `_stats_lock (threading.Lock)`: Lock for statistics access
- `_cache_lock (threading.Lock)`: Lock for cache operations
- `_scales (Dict[int, float])`: Scale factors for per-channel quantization
- `_zero_points (Dict[int, float])`: Zero points for per-channel quantization
- `_global_scale (float)`: Global scale factor
- `_global_zero_point (float)`: Global zero point offset
- `_calibration_data (List)`: Stored calibration data
- `_is_calibrated (bool)`: Whether the quantizer has been calibrated
- `_quantization_stats (Dict)`: Statistics about quantization operations
- `_cache_enabled (bool)`: Whether value caching is enabled
- `_cache_size (int)`: Maximum size of value cache
- `_value_cache (OrderedDict)`: LRU cache for dequantization values
- `_transform_cache (Dict)`: Cache for transform results
- `_qmin (int)`: Minimum representable value in quantized range
- `_qmax (int)`: Maximum representable value in quantized range
- `_qrange (int)`: Range of representable values
- `_qtype (np.dtype)`: NumPy dtype for quantized values
- `_buffer_size (int)`: Size of preallocated buffers
- `_input_buffer (np.ndarray)`: Preallocated input buffer
- `_output_buffer (np.ndarray)`: Preallocated output buffer
- `_layer_precision (Dict)`: Mixed precision settings per layer

### Constructor
```python
def __init__(self, config: Optional[QuantizationConfig] = None):
```

- **Description**:  
  Initializes the quantizer with the given configuration or default settings.

- **Parameters**:  
  - `config (Optional[QuantizationConfig])`: Configuration for the quantizer. If None, defaults are used.

- **Raises**:  
  - None explicitly, but may raise exceptions during internal initialization.

## Methods

### `hash_array`
```python
@staticmethod
def hash_array(arr: NDArray) -> str:
```

- **Description**:  
  Creates a hashable representation of a numpy array for caching purposes.

- **Parameters**:  
  - `arr (NDArray)`: NumPy array to hash.

- **Returns**:  
  - `str`: String hash representation of the array.

- **Raises**:  
  - `TypeError`: If input is not a numpy array.

- **Example**:
  ```python
  arr = np.array([1, 2, 3])
  hash_str = Quantizer.hash_array(arr)
  ```

### `_setup_quantization_ranges`
```python
def _setup_quantization_ranges(self) -> None:
```

- **Description**:  
  Sets up the range values based on quantization type and bit width.

- **Parameters**:  
  - None

- **Returns**:  
  - None

- **Raises**:  
  - `ValueError`: If unsupported quantization type is specified.

### `_initialize_buffers`
```python
def _initialize_buffers(self) -> None:
```

- **Description**:  
  Initializes preallocated buffers for improved performance during quantization.

- **Parameters**:  
  - None

- **Returns**:  
  - None

### `_init_mixed_precision`
```python
def _init_mixed_precision(self) -> None:
```

- **Description**:  
  Initializes settings for mixed precision quantization, allowing different layers to use different precision.

- **Parameters**:  
  - None

- **Returns**:  
  - None

### `scale` (property)
```python
@property
def scale(self) -> float:
```

- **Description**:  
  Gets the current global scale factor in a thread-safe manner.

- **Returns**:  
  - `float`: Current global scale factor.

### `zero_point` (property)
```python
@property
def zero_point(self) -> float:
```

- **Description**:  
  Gets the current global zero point in a thread-safe manner.

- **Returns**:  
  - `float`: Current global zero point.

### `update_config`
```python
def update_config(self, new_config: QuantizationConfig) -> None:
```

- **Description**:  
  Updates the quantizer configuration with thread safety.

- **Parameters**:  
  - `new_config (QuantizationConfig)`: A new configuration object to update parameters.

- **Returns**:  
  - None

- **Example**:
  ```python
  new_config = QuantizationConfig(quantization_type=QuantizationType.INT16)
  quantizer.update_config(new_config)
  ```

### `get_config`
```python
def get_config(self) -> Dict[str, Any]:
```

- **Description**:  
  Retrieves the current configuration as a dictionary.

- **Returns**:  
  - `Dict[str, Any]`: Dictionary representation of the configuration.

- **Example**:
  ```python
  config_dict = quantizer.get_config()
  print(f"Current quantization type: {config_dict['quantization_type']}")
  ```

### `get_stats`
```python
def get_stats(self) -> Dict[str, Any]:
```

- **Description**:  
  Retrieves quantization statistics such as number of operations, clipped values, and cache hits.

- **Returns**:  
  - `Dict[str, Any]`: Dictionary with quantization statistics.

- **Example**:
  ```python
  stats = quantizer.get_stats()
  print(f"Cache hit ratio: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])}")
  ```

### `clear_cache`
```python
def clear_cache(self) -> None:
```

- **Description**:  
  Clears the dequantization cache to free memory.

- **Returns**:  
  - None

- **Example**:
  ```python
  quantizer.clear_cache()
  ```

### `export_import_parameters`
```python
def export_import_parameters(self) -> Dict[str, Any]:
```

- **Description**:  
  Exports quantization parameters for saving or transferring to another quantizer.

- **Returns**:  
  - `Dict[str, Any]`: Dictionary with quantization parameters.

- **Example**:
  ```python
  params = quantizer.export_import_parameters()
  # Save params to file or transfer to another quantizer
  ```

### `load_parameters`
```python
def load_parameters(self, params: Dict[str, Any]) -> None:
```

- **Description**:  
  Loads quantization parameters from a dictionary created by export_import_parameters.

- **Parameters**:  
  - `params (Dict[str, Any])`: Dictionary with parameters from export_import_parameters.

- **Returns**:  
  - None

- **Example**:
  ```python
  # Load previously saved parameters
  quantizer.load_parameters(params)
  ```

### `calibrate`
```python
def calibrate(self, calibration_data: List[NDArray[np.float32]]) -> None:
```

- **Description**:  
  Calibrates the quantizer using representative data to optimize scale and zero_point values.

- **Parameters**:  
  - `calibration_data (List[NDArray[np.float32]])`: List of numpy arrays with representative data.

- **Returns**:  
  - None

- **Example**:
  ```python
  # Calibrate using sample data
  sample_data = [np.random.randn(100, 100).astype(np.float32) for _ in range(5)]
  quantizer.calibrate(sample_data)
  ```

### `reset_stats`
```python
def reset_stats(self) -> None:
```

- **Description**:  
  Resets all quantization statistics to initial values.

- **Returns**:  
  - None

- **Example**:
  ```python
  quantizer.reset_stats()
  ```

### `_validate_input`
```python
def _validate_input(self, data: NDArray) -> None:
```

- **Description**:  
  Validates input data for quantization, checking for NaN or Inf values.

- **Parameters**:  
  - `data (NDArray)`: Input array to validate.

- **Returns**:  
  - None

- **Raises**:  
  - `ValueError`: If input data contains NaN or Inf values and config disallows them.

### `compute_scale_and_zero_point`
```python
def compute_scale_and_zero_point(
    self, 
    data: NDArray[np.float32], 
    channel_idx: Optional[int] = None,
    layer_name: Optional[str] = None
) -> Tuple[float, float]:
```

- **Description**:  
  Computes optimal scale and zero_point based on current quantization mode and input data.

- **Parameters**:  
  - `data (NDArray[np.float32])`: Input float32 array.
  - `channel_idx (Optional[int])`: Optional channel index for per-channel quantization.
  - `layer_name (Optional[str])`: Optional layer name for mixed precision quantization.

- **Returns**:  
  - `Tuple[float, float]`: Computed (scale, zero_point) values.

- **Example**:
  ```python
  scale, zero_point = quantizer.compute_scale_and_zero_point(data)
  ```

### `get_layer_quantization_params`
```python
def get_layer_quantization_params(self, layer_name: str) -> Dict[str, Any]:
```

- **Description**:  
  Gets quantization parameters for a specific layer, particularly useful with mixed precision.

- **Parameters**:  
  - `layer_name (str)`: Name of the layer.

- **Returns**:  
  - `Dict[str, Any]`: Dictionary with quantization parameters for the layer.

- **Example**:
  ```python
  params = quantizer.get_layer_quantization_params("conv1")
  print(f"Layer will use {params['bits']} bits")
  ```

### `should_quantize_layer`
```python
def should_quantize_layer(self, layer_name: str) -> bool:
```

- **Description**:  
  Determines if a layer should be quantized based on configuration and layer name.

- **Parameters**:  
  - `layer_name (str)`: Name of the layer.

- **Returns**:  
  - `bool`: True if the layer should be quantized, False otherwise.

- **Example**:
  ```python
  if quantizer.should_quantize_layer("fc_bias"):
      # Apply quantization
  ```

### `_apply_requantization`
```python
def _apply_requantization(self, quantized_data: NDArray, original_data: NDArray) -> NDArray:
```

- **Description**:  
  Applies requantization to reduce quantization error by feeding back error information.

- **Parameters**:  
  - `quantized_data (NDArray)`: Already quantized data.
  - `original_data (NDArray)`: Original unquantized data for reference.

- **Returns**:  
  - `NDArray`: Requantized data with reduced error.

### `_quantize_mixed_precision`
```python
def _quantize_mixed_precision(
    self, 
    data: NDArray[np.float32], 
    layer_name: str,
    channel_dim: Optional[int] = None
) -> NDArray:
```

- **Description**:  
  Performs mixed precision quantization for a specific layer with its own precision settings.

- **Parameters**:  
  - `data (NDArray[np.float32])`: Input float32 array.
  - `layer_name (str)`: Name of the layer for precision lookup.
  - `channel_dim (Optional[int])`: Optional dimension for per-channel quantization.

- **Returns**:  
  - `NDArray`: Quantized array with the precision specified for this layer.

### `_dequantize_mixed_precision`
```python
def _dequantize_mixed_precision(
    self, 
    data: NDArray, 
    layer_name: str,
    channel_dim: Optional[int] = None
) -> NDArray[np.float32]:
```

- **Description**:  
  Performs mixed precision dequantization for a specific layer with custom precision.

- **Parameters**:  
  - `data (NDArray)`: Quantized input array.
  - `layer_name (str)`: Name of the layer for precision lookup.
  - `channel_dim (Optional[int])`: Optional dimension for per-channel dequantization.

- **Returns**:  
  - `NDArray[np.float32]`: Dequantized array as float32.

### `quantize`
```python
def quantize(
    self, 
    data: Union[NDArray[np.float32], Any], 
    validate: bool = True,
    channel_dim: Optional[int] = None,
    layer_name: Optional[str] = None
) -> NDArray:
```

- **Description**:  
  High-performance vectorized quantization that converts float32 data to lower precision.

- **Parameters**:  
  - `data (Union[NDArray[np.float32], Any])`: Input float32 array.
  - `validate (bool)`: If True, performs input validation.
  - `channel_dim (Optional[int])`: Dimension index for per-channel quantization (if enabled).
  - `layer_name (Optional[str])`: Optional layer name for mixed precision handling.

- **Returns**:  
  - `NDArray`: Quantized array of the configured type.

- **Raises**:  
  - `TypeError`: If input is not a numpy array (when validate=True).
  - `ValueError`: If input contains NaN/Inf and config prohibits them.

- **Example**:
  ```python
  data = np.random.randn(100, 100).astype(np.float32)
  quantized = quantizer.quantize(data, layer_name="conv1")
  ```

### `dequantize`
```python
def dequantize(
    self, 
    data: NDArray, 
    channel_dim: Optional[int] = None,
    layer_name: Optional[str] = None
) -> NDArray[np.float32]:
```

- **Description**:  
  Dequantizes data from quantized format back to float32, with optional caching.

- **Parameters**:  
  - `data (NDArray)`: Quantized input array.
  - `channel_dim (Optional[int])`: Dimension index for per-channel dequantization.
  - `layer_name (Optional[str])`: Optional layer name for mixed precision handling.

- **Returns**:  
  - `NDArray[np.float32]`: Dequantized array as float32.

- **Example**:
  ```python
  dequantized = quantizer.dequantize(quantized_data)
  ```

### `quantize_dequantize`
```python
def quantize_dequantize(
    self, 
    data: NDArray[np.float32],
    channel_dim: Optional[int] = None,
    layer_name: Optional[str] = None
) -> NDArray[np.float32]:
```

- **Description**:  
  Quantizes and immediately dequantizes data to simulate quantization effects.

- **Parameters**:  
  - `data (NDArray[np.float32])`: Float32 input array.
  - `channel_dim (Optional[int])`: Dimension index for per-channel quantization.
  - `layer_name (Optional[str])`: Optional layer name for mixed precision handling.

- **Returns**:  
  - `NDArray[np.float32]`: Data after quantization and dequantization.

- **Example**:
  ```python
  # Simulate quantization effects without changing data type
  simulated = quantizer.quantize_dequantize(data)
  ```

### `_quantize_per_channel`
```python
def _quantize_per_channel(self, data: NDArray[np.float32], channel_dim: Optional[int] = None) -> NDArray:
```

- **Description**:  
  Performs per-channel quantization, computing separate scale/zero_point for each channel.

- **Parameters**:  
  - `data (NDArray[np.float32])`: Input float32 array.
  - `channel_dim (Optional[int])`: Dimension along which to perform per-channel quantization (if None, uses the first dimension).

- **Returns**:  
  - `NDArray`: Quantized array with per-channel scale factors.

### `_dequantize_per_channel`
```python
def _dequantize_per_channel(self, data: NDArray, channel_dim: Optional[int] = None) -> NDArray[np.float32]:
```

- **Description**:  
  Performs per-channel dequantization, using separate scale/zero_point for each channel.

- **Parameters**:  
  - `data (NDArray)`: Quantized input array.
  - `channel_dim (Optional[int])`: Dimension along which to perform per-channel dequantization (if None, uses the first dimension).

- **Returns**:  
  - `NDArray[np.float32]`: Dequantized array with per-channel scale factors.

## Testing
Tests for this module should verify:
- Thread-safety with concurrent operations
- Correctness of quantization and dequantization
- Performance with large arrays
- Proper caching behavior
- Calibration with representative data

```bash
python -m unittest tests/test_quantizer.py
```

## Security & Compliance
- No explicit encryption or security features
- Care should be taken when quantizing sensitive numerical data to prevent information leakage from quantization artifacts
- Thread-safety mechanisms are provided for multi-threaded environments

> Last Updated: 2025-05-11