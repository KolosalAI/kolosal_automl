# Quantizer Documentation

## Overview

The `Quantizer` class is a high-performance data quantizer designed for efficient quantization and dequantization of numerical data. It supports multiple quantization types (int8, uint8, int16), various quantization modes (symmetric, asymmetric, per-channel), and includes advanced features such as thread-safe operations, optimized cache mechanisms, statistical tracking, and vectorized operations.

## Features

- **Multiple Quantization Types**: Supports `int8`, `uint8`, and `int16` quantization.
- **Various Quantization Modes**: Includes symmetric, asymmetric, and per-channel quantization.
- **Thread-Safe Operations**: Ensures safe concurrent access with reentrant locks.
- **Optimized Cache Mechanisms**: Implements LRU caching for frequent values to improve performance.
- **Statistical Tracking**: Tracks quantization statistics such as clipped values, cache hits, and processing time.
- **Vectorized Operations**: Utilizes NumPy for efficient, vectorized operations.
- **Calibration**: Calibrates quantization parameters using representative data for optimal performance.
- **Export/Import Parameters**: Allows saving and loading quantization parameters for consistent results.

## Class: `Quantizer`

### Initialization

```python
def __init__(self, config: Optional[QuantizationConfig] = None):
```

Initializes the quantizer with optional configuration. If no configuration is provided, default values are used.

### Methods

#### `hash_array(arr: NDArray) -> str`

Creates a hashable representation of a NumPy array.

#### `_setup_quantization_ranges() -> None`

Sets up range values based on the quantization type and bit width.

#### `_initialize_buffers() -> None`

Initializes preallocated buffers for performance optimization.

#### `update_config(new_config: QuantizationConfig) -> None`

Updates the quantizer configuration with thread safety.

#### `get_config() -> Dict[str, Any]`

Retrieves the current configuration as a dictionary.

#### `compute_scale_and_zero_point(data: NDArray[np.float32], channel_idx: Optional[int] = None) -> Tuple[float, float]`

Computes the optimal scale and zero point based on the current quantization mode.

#### `quantize(data: Union[NDArray[np.float32], Any], validate: bool = True, channel_dim: Optional[int] = None) -> NDArray`

Quantizes the input data with optional validation and per-channel quantization.

#### `_quantize_per_channel(data: NDArray[np.float32], channel_dim: int) -> NDArray`

Performs per-channel quantization for improved accuracy.

#### `_get_cached_value(quantized_value: int, scale: float, zero_point: float) -> Optional[float]`

Thread-safe cache lookup for dequantized values.

#### `_cache_value(quantized_value: int, scale: float, zero_point: float, float_value: float) -> None`

Thread-safe cache update with LRU behavior.

#### `dequantize(data: NDArray, channel_dim: Optional[int] = None, use_cache: Optional[bool] = None) -> NDArray[np.float32]`

Dequantizes the input data with optional per-channel dequantization and cache usage.

#### `_dequantize_per_channel(data: NDArray, channel_dim: int) -> NDArray[np.float32]`

Performs per-channel dequantization.

#### `quantize_dequantize(data: NDArray[np.float32]) -> NDArray[np.float32]`

Performs quantization followed by dequantization to simulate quantization loss.

#### `get_quantization_stats() -> Dict[str, Union[int, float]]`

Returns comprehensive statistics about quantization operations.

#### `_validate_input(data: NDArray[np.float32]) -> None`

Validates input data with better error reporting and handling.

#### `reset_stats() -> None`

Resets all quantization statistics.

#### `clear_cache() -> None`

Clears the dequantization value cache.

#### `calibrate(calibration_data: NDArray[np.float32]) -> Dict[str, float]`

Calibrates the quantizer using representative data to find optimal parameters.

#### `get_parameters() -> Dict[str, Dict[str, float]]`

Retrieves current quantization parameters, including per-channel parameters if applicable.

#### `export_parameters(file_path: str) -> None`

Exports quantization parameters to a file for later use.

#### `import_parameters(file_path: str) -> None`

Imports quantization parameters from a file.

### Properties

#### `scale: float`

Gets the current global scale factor.

#### `zero_point: float`

Gets the current global zero point.

## Example Usage

```python
# Create a default quantizer with default config
config = QuantizationConfig(
    quantization_type=QuantizationType.INT8.value,
    quantization_mode=QuantizationMode.DYNAMIC_PER_BATCH.value,
    enable_cache=True,
    cache_size=256,
    min_percentile=0.1,
    max_percentile=99.9,
    buffer_size=1024,
    outlier_threshold=3.0,  # Filter outliers beyond 3 std deviations
)
quantizer = Quantizer(config)
print("Quantizer config:", quantizer.get_config())

# Example data
data = np.random.uniform(-1, 1, size=(100, 10)).astype(np.float32)

# Calibrate the quantizer
calibration_results = quantizer.calibrate(data)
print("Calibration results:")
for key, value in calibration_results.items():
    if key != 'error_hist':  # Skip histogram for cleaner output
        print(f"  {key}: {value}")

# Quantize and dequantize
q_data = quantizer.quantize(data)
deq_data = quantizer.dequantize(q_data)

# Check reconstruction error
mse = np.mean((data - deq_data) ** 2)
print(f"Mean squared error: {mse:.8f}")

# Get quantization statistics
stats = quantizer.get_quantization_stats()
print("Quantization stats:", stats)
```

## Conclusion

The `Quantizer` class provides a robust and efficient solution for quantization and dequantization tasks, with support for various quantization types and modes. Its advanced features, such as thread safety, caching, and statistical tracking, make it suitable for high-performance applications.