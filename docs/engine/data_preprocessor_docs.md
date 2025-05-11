# Module: `DataPreprocessor`

## Overview
A high-performance, production-ready data preprocessor for machine learning pipelines with extensive functionality including multiple normalization strategies, outlier detection, thread-safety, memory optimization, comprehensive error handling, monitoring, and serialization capabilities.

## Prerequisites
- Python ≥3.7
- Required packages:
  ```bash
  pip install numpy>=1.19.0
  ```
- Optional dependencies: `threading`, `logging`, `pickle`, `dataclasses`

## Installation
The module is part of a larger package structure. To use it within your project:

```python
from modules.preprocessor import DataPreprocessor
from modules.configs import PreprocessorConfig
```

## Usage
```python
import numpy as np
from modules.preprocessor import DataPreprocessor
from modules.configs import PreprocessorConfig, NormalizationType

# Create a config with standard normalization
config = PreprocessorConfig(
    normalization=NormalizationType.STANDARD,
    handle_nan=True,
    detect_outliers=True
)

# Initialize and fit the preprocessor
preprocessor = DataPreprocessor(config)
X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
preprocessor.fit(X_train)

# Transform new data
X_test = np.array([[2.0, 3.0], [4.0, 5.0]])
X_transformed = preprocessor.transform(X_test)

# Save the fitted preprocessor
preprocessor.serialize("preprocessor.pkl")

# Load later
loaded_preprocessor = DataPreprocessor.deserialize("preprocessor.pkl")
```

## Configuration
The `DataPreprocessor` is configured via a `PreprocessorConfig` dataclass with these key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `normalization` | `NormalizationType.NONE` | Normalization strategy (STANDARD, MINMAX, ROBUST, etc.) |
| `handle_nan` | `False` | Whether to handle NaN values |
| `nan_strategy` | `"mean"` | Strategy for NaN replacement (mean, median, constant) |
| `handle_inf` | `False` | Whether to handle infinite values |
| `detect_outliers` | `False` | Whether to detect and handle outliers |
| `outlier_method` | `"zscore"` | Method for outlier detection (zscore, iqr, percentile) |
| `chunk_size` | `None` | Size of chunks for processing large datasets |
| `parallel_processing` | `False` | Whether to use parallel processing |
| `debug_mode` | `False` | Enable detailed logging |
| `cache_enabled` | `False` | Enable caching for transform operations |

## Architecture
The preprocessor uses a pipeline architecture where multiple processing steps are applied sequentially:

1. Data validation and conversion
2. NaN/Inf handling (if enabled)
3. Outlier detection and handling (if enabled)
4. Normalization (if enabled)
5. Value clipping (if enabled)

Each step is encapsulated in a separate method and the pipeline is constructed based on the provided configuration.

---

## Exception Classes

### `PreprocessingError`
```python
class PreprocessingError(Exception):
```
- **Description**:  
  Base class for all preprocessing exceptions. Serves as a parent class for more specific exception types.

### `InputValidationError`
```python
class InputValidationError(PreprocessingError):
```
- **Description**:  
  Exception raised for input validation errors, such as incompatible shapes, data types, or out-of-range values.

### `StatisticsError`
```python
class StatisticsError(PreprocessingError):
```
- **Description**:  
  Exception raised for errors in computing statistics, such as division by zero or other numerical issues.

### `SerializationError`
```python
class SerializationError(PreprocessingError):
```
- **Description**:  
  Exception raised for errors in serialization or deserialization operations, such as file I/O problems or data integrity issues.

---

## Utility Functions

### `log_operation(func)`
```python
def log_operation(func):
```
- **Description**:  
  Decorator for logging operation timing and handling exceptions.

- **Parameters**:  
  - `func (Callable)`: The function to be decorated.

- **Returns**:  
  - `Callable`: Wrapped function that logs timing and handles exceptions.

---

## Main Class: DataPreprocessor

### `__init__(config=None)`
```python
def __init__(self, config: Optional[PreprocessorConfig] = None):
```
- **Description**:  
  Initializes the preprocessor with the specified configuration.

- **Parameters**:  
  - `config (Optional[PreprocessorConfig])`: Configuration object, or None to use defaults.

### `fit(X, y=None, feature_names=None, **fit_params)`
```python
def fit(self, X, y=None, feature_names=None, **fit_params):
```
- **Description**:  
  Fits the preprocessor to the data, computing necessary statistics for transformations.

- **Parameters**:  
  - `X (Union[np.ndarray, List])`: Input data to fit.
  - `y (Any)`: Target values (unused, for API compatibility).
  - `feature_names (Optional[List[str]])`: Optional list of feature names.
  - `**fit_params`: Additional parameters.

- **Returns**:  
  - `DataPreprocessor`: The fitted preprocessor instance.

- **Raises**:  
  - `InputValidationError`: If input data is invalid.
  - `StatisticsError`: If statistics computation fails.

### `transform(X, copy=True)`
```python
def transform(self, X: Union[np.ndarray, List], copy: bool = True) -> np.ndarray:
```
- **Description**:  
  Applies preprocessing transformations to the input data.

- **Parameters**:  
  - `X (Union[np.ndarray, List])`: Input data to transform.
  - `copy (bool)`: If True, ensures input is not modified in-place.

- **Returns**:  
  - `np.ndarray`: Transformed data array.

- **Raises**:  
  - `RuntimeError`: If preprocessor is not fitted.
  - `InputValidationError`: If input data is invalid.

### `fit_transform(X, y=None, feature_names=None, copy=True)`
```python
def fit_transform(self, X: Union[np.ndarray, List], y=None, feature_names=None, copy: bool = True) -> np.ndarray:
```
- **Description**:  
  Combines fit and transform operations efficiently.

- **Parameters**:  
  - `X (Union[np.ndarray, List])`: Input data.
  - `y (Any)`: Target values (unused, for API compatibility).
  - `feature_names (Optional[List[str]])`: Optional feature names.
  - `copy (bool)`: Whether to copy the input data.

- **Returns**:  
  - `np.ndarray`: Transformed data.

### `partial_fit(X, feature_names=None)`
```python
def partial_fit(self, X: np.ndarray, feature_names=None) -> 'DataPreprocessor':
```
- **Description**:  
  Updates the preprocessor with new data incrementally.

- **Parameters**:  
  - `X (np.ndarray)`: New data to incorporate.
  - `feature_names (Optional[List[str]])`: Optional feature names (ignored if already fitted).

- **Returns**:  
  - `DataPreprocessor`: The updated preprocessor.

- **Raises**:  
  - `InputValidationError`: If dimensions don't match.

### `reverse_transform(X, copy=True)`
```python
def reverse_transform(self, X: np.ndarray, copy: bool = True) -> np.ndarray:
```
- **Description**:  
  Reverses the preprocessing transformations to recover original data scale.

- **Parameters**:  
  - `X (np.ndarray)`: Transformed data to reverse.
  - `copy (bool)`: Whether to copy the input data.

- **Returns**:  
  - `np.ndarray`: Data with transformations reversed.

- **Raises**:  
  - `RuntimeError`: If not fitted.
  - `StatisticsError`: If required statistics are missing.

### `serialize(path)`
```python
def serialize(self, path: str) -> bool:
```
- **Description**:  
  Serializes the preprocessor to disk.

- **Parameters**:  
  - `path (str)`: File path to save the serialized preprocessor.

- **Returns**:  
  - `bool`: True if successful, False otherwise.

### `deserialize(path)`
```python
@classmethod
def deserialize(cls, path: str) -> 'DataPreprocessor':
```
- **Description**:  
  Deserializes a preprocessor from disk.

- **Parameters**:  
  - `path (str)`: File path to load the serialized preprocessor from.

- **Returns**:  
  - `DataPreprocessor`: Deserialized DataPreprocessor instance.

- **Raises**:  
  - `SerializationError`: If deserialization fails.

### `get_statistics()`
```python
def get_statistics(self) -> Dict[str, Any]:
```
- **Description**:  
  Gets the current statistics computed by the preprocessor.

- **Returns**:  
  - `Dict[str, Any]`: Dictionary of statistics.

- **Raises**:  
  - `RuntimeError`: If preprocessor is not fitted.

### `get_feature_names()`
```python
def get_feature_names(self) -> List[str]:
```
- **Description**:  
  Gets the feature names used by the preprocessor.

- **Returns**:  
  - `List[str]`: List of feature names.

### `get_performance_metrics()`
```python
def get_performance_metrics(self) -> Dict[str, List[float]]:
```
- **Description**:  
  Gets performance metrics collected during preprocessing operations.

- **Returns**:  
  - `Dict[str, List[float]]`: Dictionary of performance metrics.

### `reset()`
```python
def reset(self) -> None:
```
- **Description**:  
  Resets the preprocessor to its initialized state, clearing all fitted statistics.

### `update_config(new_config)`
```python
def update_config(self, new_config: PreprocessorConfig) -> None:
```
- **Description**:  
  Updates the preprocessor configuration, resetting state if pipeline changes.

- **Parameters**:  
  - `new_config (PreprocessorConfig)`: New configuration to apply.

---

## Internal Methods

### `_configure_logging()`
```python
def _configure_logging(self) -> None:
```
- **Description**:  
  Configures logging based on config settings.

### `_init_parallel_executor()`
```python
def _init_parallel_executor(self) -> None:
```
- **Description**:  
  Initializes parallel processing executor if parallel processing is enabled.

### `_init_processing_functions()`
```python
def _init_processing_functions(self) -> None:
```
- **Description**:  
  Initializes preprocessing functions based on configuration, creating the processing pipeline.

### `_error_context(operation)`
```python
@contextmanager
def _error_context(self, operation: str):
```
- **Description**:  
  Context manager for handling errors during preprocessing operations.

- **Parameters**:  
  - `operation (str)`: Name of the operation being performed.

### `_validate_and_convert_input(X, operation, copy)`
```python
def _validate_and_convert_input(self, X: Union[np.ndarray, List], operation: str = "transform", copy: bool = False) -> np.ndarray:
```
- **Description**:  
  Validates and converts input data efficiently with comprehensive checks.

- **Parameters**:  
  - `X (Union[np.ndarray, List])`: Input data.
  - `operation (str)`: Operation being performed ('fit' or 'transform').
  - `copy (bool)`: Whether to copy the input data.

- **Returns**:  
  - `np.ndarray`: Validated and converted numpy array.

- **Raises**:  
  - `InputValidationError`: If validation fails.

### `_fit_full(X)`
```python
def _fit_full(self, X: np.ndarray) -> None:
```
- **Description**:  
  Computes statistics on the full dataset.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

### `_fit_in_chunks(X)`
```python
def _fit_in_chunks(self, X: np.ndarray) -> None:
```
- **Description**:  
  Fits on large datasets by processing in chunks to reduce memory usage.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

### `_init_running_stats()`
```python
def _init_running_stats(self) -> None:
```
- **Description**:  
  Initializes running statistics for incremental fitting.

### `_update_stats_with_chunk(chunk)`
```python
def _update_stats_with_chunk(self, chunk: np.ndarray) -> None:
```
- **Description**:  
  Updates running statistics with a new data chunk.

- **Parameters**:  
  - `chunk (np.ndarray)`: Data chunk to process.

### `_finalize_running_stats()`
```python
def _finalize_running_stats(self) -> None:
```
- **Description**:  
  Finalizes statistics after processing all chunks.

### `_transform_parallel(X)`
```python
def _transform_parallel(self, X: np.ndarray) -> np.ndarray:
```
- **Description**:  
  Transforms data using parallel processing for large datasets.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

- **Returns**:  
  - `np.ndarray`: Transformed data array.

### `_hash_array(arr)`
```python
def _hash_array(self, arr: np.ndarray) -> str:
```
- **Description**:  
  Generates a hash for a numpy array for caching purposes.

- **Parameters**:  
  - `arr (np.ndarray)`: Input array to hash.

- **Returns**:  
  - `str`: Hash string for the array.

### `_transform_raw(X)`
```python
def _transform_raw(self, X: np.ndarray) -> np.ndarray:
```
- **Description**:  
  Applies preprocessing operations without caching.

- **Parameters**:  
  - `X (np.ndarray)`: Input data.

- **Returns**:  
  - `np.ndarray`: Transformed data.

### `_transform_in_chunks(X)`
```python
def _transform_in_chunks(self, X: np.ndarray) -> np.ndarray:
```
- **Description**:  
  Transforms large datasets by processing in chunks.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

- **Returns**:  
  - `np.ndarray`: Transformed data array.

### `_compute_standard_stats(X)`
```python
def _compute_standard_stats(self, X: np.ndarray) -> None:
```
- **Description**:  
  Computes statistics for standard normalization (mean, std).

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

### `_compute_minmax_stats(X)`
```python
def _compute_minmax_stats(self, X: np.ndarray) -> None:
```
- **Description**:  
  Computes statistics for min-max normalization.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

### `_compute_robust_stats(X)`
```python
def _compute_robust_stats(self, X: np.ndarray) -> None:
```
- **Description**:  
  Computes robust statistics for normalization using percentiles.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

### `_compute_custom_stats(X)`
```python
def _compute_custom_stats(self, X: np.ndarray) -> None:
```
- **Description**:  
  Computes custom statistics for normalization.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

### `_compute_outlier_stats(X)`
```python
def _compute_outlier_stats(self, X: np.ndarray) -> None:
```
- **Description**:  
  Computes statistics for outlier detection.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

### `_clip_data(X)`
```python
def _clip_data(self, X: np.ndarray) -> np.ndarray:
```
- **Description**:  
  Clips data to specified min/max values.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

- **Returns**:  
  - `np.ndarray`: Clipped data array.

### `_normalize_data(X)`
```python
def _normalize_data(self, X: np.ndarray) -> np.ndarray:
```
- **Description**:  
  Applies normalization to the data according to the configured method.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

- **Returns**:  
  - `np.ndarray`: Normalized data array.

- **Raises**:  
  - `StatisticsError`: If required statistics are not available.

### `_get_norm_type(normalization_type)`
```python
def _get_norm_type(self, normalization_type: Union[str, NormalizationType]) -> NormalizationType:
```
- **Description**:  
  Converts string normalization type to enum if needed and validates.

- **Parameters**:  
  - `normalization_type (Union[str, NormalizationType])`: Normalization type.

- **Returns**:  
  - `NormalizationType`: Validated normalization type enum.

### `_handle_nan_values(X)`
```python
def _handle_nan_values(self, X: np.ndarray) -> np.ndarray:
```
- **Description**:  
  Handles NaN values in the data according to the configured strategy.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

- **Returns**:  
  - `np.ndarray`: Data array with NaN values handled.

### `_handle_infinite_values(X)`
```python
def _handle_infinite_values(self, X: np.ndarray) -> np.ndarray:
```
- **Description**:  
  Handles infinite values in the data according to the configured strategy.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

- **Returns**:  
  - `np.ndarray`: Data array with infinite values handled.

### `_handle_outliers(X)`
```python
def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
```
- **Description**:  
  Handles outliers in the data according to the configured method.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

- **Returns**:  
  - `np.ndarray`: Data array with outliers handled.

### `_update_stats_incrementally(X)`
```python
def _update_stats_incrementally(self, X: np.ndarray) -> None:
```
- **Description**:  
  Updates statistics incrementally with new data.

- **Parameters**:  
  - `X (np.ndarray)`: New data to incorporate.

### `_update_outlier_stats(X)`
```python
def _update_outlier_stats(self, X: np.ndarray) -> None:
```
- **Description**:  
  Updates outlier statistics incrementally with new data.

- **Parameters**:  
  - `X (np.ndarray)`: New data to incorporate.

### `_compute_most_frequent_values(X)`
```python
def _compute_most_frequent_values(self, X: np.ndarray) -> None:
```
- **Description**:  
  Computes most frequent value for each feature for NaN imputation.

- **Parameters**:  
  - `X (np.ndarray)`: Input data array.

---

## Security & Compliance
- Thread-safe implementation using reentrant locks
- Protection against numerical instabilities
- Data validation to prevent corrupt processing
- Serialization with data integrity verification

## Testing
```bash
python -m unittest tests/test_preprocessor.py
```

> Last Updated: 2025-05-11