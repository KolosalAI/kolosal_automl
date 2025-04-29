# Module: `data_preprocessor`

## Overview
The `data_preprocessor` module provides an advanced, scalable, and production-ready data
preprocessing system optimized for large-scale machine learning deployments. It supports:
- Multiple normalization strategies (standard, min-max, robust, custom)
- Outlier detection and handling
- NaN and infinite value handling
- Chunked processing for large datasets
- Caching and serialization
- Parallelized transformations

It is designed for high performance, modularity, and robustness, suitable for real-world industrial applications.

---

## Prerequisites
- **Python Version**: ≥3.8
- **Required packages**:
  ```bash
  pip install numpy
  ```
- **Optional** (recommended for better performance):
  - `psutil` (for memory monitoring)
  - `concurrent.futures` (for parallelism, included in Python 3.8+)

## Installation
```bash
pip install numpy
```

## Usage
```python
from modules.configs import PreprocessorConfig
from data_preprocessor import DataPreprocessor

# Initialize preprocessor with default config
preprocessor = DataPreprocessor()

# Fit on training data
preprocessor.fit(X_train)

# Transform data
X_train_transformed = preprocessor.transform(X_train)

# Save preprocessor
preprocessor.serialize("preprocessor.pkl")

# Load preprocessor later
loaded_preprocessor = DataPreprocessor.deserialize("preprocessor.pkl")
X_test_transformed = loaded_preprocessor.transform(X_test)
```

## Configuration
| Config Option | Default | Description |
|---|---|---|
| `normalization` | `NONE` | Normalization strategy (STANDARD, MINMAX, ROBUST, CUSTOM) |
| `handle_nan` | `True` | Whether to handle NaN values |
| `nan_strategy` | `"mean"` | Strategy for handling NaNs (mean, median, most_frequent, constant) |
| `handle_inf` | `True` | Whether to handle infinite values |
| `inf_strategy` | `"max_value"` | Strategy for handling inf values |
| `detect_outliers` | `False` | Whether to detect and handle outliers |
| `outlier_method` | `"zscore"` | Outlier detection method (iqr, zscore, percentile) |
| `cache_enabled` | `False` | Enable caching of small transforms |
| `parallel_processing` | `False` | Enable parallel transformation |
| `chunk_size` | `None` | Chunk size for processing large datasets |
| `debug_mode` | `False` | Enable detailed debug logging |
| `clip_values` | `False` | Whether to clip values during transformation |
| `scale_shift` | `False` | Whether to shift scaling |
| `input_size_limit` | `None` | Limit input size for validation |

---

## Architecture
**Data Flow**:
```
Input Data → Validate → Handle NaN/Inf → Handle Outliers → Normalize → Clip → Output
```
Optional caching and parallel execution is integrated based on configuration.

---

## Testing
```bash
pytest tests/
```

Unit tests should cover:
- Fitting and transforming small datasets
- Serialization/deserialization
- Chunked processing for large datasets
- NaN/Inf handling
- Different normalization types
- Parallel transformation correctness

---

## Security & Compliance
- **Pickle** is used for serialization; use in a trusted environment.
- No direct user input execution risk.

---

> Last Updated: 2025-04-28

---
# Classes

---

## `PreprocessingError`
```python
class PreprocessingError(Exception)
```
**Description**:  
Base class for all preprocessing-related exceptions.

---

## `InputValidationError`
```python
class InputValidationError(PreprocessingError)
```
**Description**:  
Raised for invalid input data types, shapes, or sizes during preprocessing.

---

## `StatisticsError`
```python
class StatisticsError(PreprocessingError)
```
**Description**:  
Raised when there are errors in statistical computations (e.g., missing mean/std).

---

## `SerializationError`
```python
class SerializationError(PreprocessingError)
```
**Description**:  
Raised for errors during serialization or deserialization of the preprocessor.

---

## `DataPreprocessor`
```python
class DataPreprocessor:
```

**Description**:  
A highly optimized and configurable data preprocessing engine for industrial ML pipelines.
Supports various preprocessing steps with thread-safety, incremental updates, and error handling.

### Attributes
| Attribute | Type | Description |
|---|---|---|
| `_stats` | `Dict[str, Any]` | Stored statistics (mean, std, etc.) |
| `_feature_names` | `List[str]` | List of feature names |
| `_fitted` | `bool` | Whether the preprocessor is fitted |
| `_n_features` | `int` | Number of features |
| `_n_samples_seen` | `int` | Number of samples seen during fitting |
| `_metrics` | `Dict[str, List[float]]` | Performance metrics (fit time, transform time) |
| `_transform_cache` | `Dict[str, np.ndarray]` | Cache for fast repeated transformations |
| `_parallel_executor` | `ThreadPoolExecutor` | Executor for parallel processing |
| `_running_stats` | `Dict[str, Any]` | Statistics for chunked fitting |

---

## Methods (Function-by-Function)

---

### `__init__`
```python
def __init__(self, config: Optional[PreprocessorConfig] = None)
```
**Description**:  
Initializes a `DataPreprocessor` with optional configuration.

- **Parameters**:
  - `config (Optional[PreprocessorConfig])`: Configuration object or `None` for defaults.

- **Returns**:  
  - `None`

---

### `fit`
```python
@log_operation
def fit(self, X, y=None, feature_names=None, **fit_params)
```
**Description**:  
Fits the preprocessor on training data, calculating required statistics.

- **Parameters**:
  - `X (array-like)`: Input data.
  - `y (ignored)`: Placeholder for compatibility.
  - `feature_names (list, optional)`: Custom feature names.
  - `fit_params`: Additional parameters.

- **Returns**:
  - `self`

- **Raises**:
  - `InputValidationError`
  - `StatisticsError`

---

### `transform`
```python
@log_operation
def transform(self, X: Union[np.ndarray, List], copy: bool = True) -> np.ndarray
```
**Description**:  
Applies preprocessing transformations to the input data.

- **Parameters**:
  - `X (array-like)`: Data to transform.
  - `copy (bool)`: If `True`, operates on a copy.

- **Returns**:
  - `np.ndarray`: Transformed data.

- **Raises**:
  - `InputValidationError`
  - `StatisticsError`

---

### `fit_transform`
```python
@log_operation
def fit_transform(self, X, y=None, feature_names=None, copy: bool = True) -> np.ndarray
```
**Description**:  
Convenience method: fits and transforms the data in one step.

- **Returns**:
  - `np.ndarray`

---

### `partial_fit`
```python
def partial_fit(self, X: np.ndarray, feature_names=None) -> 'DataPreprocessor'
```
**Description**:  
Updates the preprocessor incrementally with new data.

- **Returns**:
  - `self`

- **Raises**:
  - `InputValidationError`

---

### `serialize`
```python
def serialize(self, path: str) -> bool
```
**Description**:  
Saves the preprocessor state to disk.

- **Returns**:
  - `True` if successful, `False` otherwise.

- **Raises**:
  - `SerializationError`

---

### `deserialize`
```python
@classmethod
def deserialize(cls, path: str) -> 'DataPreprocessor'
```
**Description**:  
Loads a preprocessor from disk.

- **Returns**:
  - `DataPreprocessor` instance

- **Raises**:
  - `SerializationError`

---

### `reverse_transform`
```python
def reverse_transform(self, X: np.ndarray, copy: bool = True) -> np.ndarray
```
**Description**:  
Reverse the transformations to approximate original data.

- **Returns**:
  - `np.ndarray`

---

### `reset`
```python
def reset(self) -> None
```
**Description**:  
Resets internal state, statistics, and caches.

---

### `update_config`
```python
def update_config(self, new_config: PreprocessorConfig) -> None
```
**Description**:  
Updates the configuration and resets if pipeline settings change.

---

### `get_statistics`
```python
def get_statistics(self) -> Dict[str, Any]
```
**Description**:  
Retrieves the current computed statistics.

- **Returns**:
  - `Dict[str, Any]`

---

### `get_feature_names`
```python
def get_feature_names(self) -> List[str]
```
**Description**:  
Retrieves the feature names used during fitting.

---

### `get_performance_metrics`
```python
def get_performance_metrics(self) -> Dict[str, List[float]]
```
**Description**:  
Retrieves performance metrics like fit and transform time.

---
Perfect —  
I’ll continue now and **fully expand** the documentation for all **private and helper methods** into the same industrial-grade format.

Continuing from where we left:

---

# Private Helper Functions

---

### `_configure_logging`
```python
def _configure_logging(self) -> None
```
**Description**:  
Configures the logger for the preprocessor based on the debug mode setting.

- **Parameters**:  
  - None

- **Returns**:  
  - None

---

### `_init_parallel_executor`
```python
def _init_parallel_executor(self) -> None
```
**Description**:  
Initializes a thread pool executor for parallel transformation if enabled.

- **Parameters**:  
  - None

- **Returns**:  
  - None

- **Raises**:
  - Automatically disables parallelism if `concurrent.futures` is not available.

---

### `_init_processing_functions`
```python
def _init_processing_functions(self) -> None
```
**Description**:  
Initializes a pipeline list of preprocessing functions based on configuration flags.

- **Parameters**:  
  - None

- **Returns**:  
  - None

---

### `_error_context`
```python
@contextmanager
def _error_context(self, operation: str)
```
**Description**:  
Context manager for standardized error handling during operations.

- **Parameters**:
  - `operation (str)`: Operation name (e.g., "fit", "transform").

- **Yields**:
  - None

- **Raises**:
  - Reraises categorized custom errors based on operation type.

---

### `_validate_and_convert_input`
```python
def _validate_and_convert_input(self, X, operation: str = "transform", copy: bool = False) -> np.ndarray
```
**Description**:  
Validates input data type, shape, and properties, then converts it to a numpy array.

- **Parameters**:
  - `X (array-like)`: Input data.
  - `operation (str)`: Operation type (fit, transform, etc.).
  - `copy (bool)`: Copy data if required.

- **Returns**:
  - `np.ndarray`: Validated input array.

- **Raises**:
  - `InputValidationError`

---

### `_fit_full`
```python
def _fit_full(self, X: np.ndarray) -> None
```
**Description**:  
Fits statistics on the full input dataset in one pass.

---

### `_fit_in_chunks`
```python
def _fit_in_chunks(self, X: np.ndarray) -> None
```
**Description**:  
Processes large datasets in chunks for memory efficiency during fitting.

---

### `_init_running_stats`
```python
def _init_running_stats(self) -> None
```
**Description**:  
Initializes structures to accumulate running statistics for incremental fitting.

---

### `_update_stats_with_chunk`
```python
def _update_stats_with_chunk(self, chunk: np.ndarray) -> None
```
**Description**:  
Updates running statistics with a processed data chunk.

---

### `_finalize_running_stats`
```python
def _finalize_running_stats(self) -> None
```
**Description**:  
Finalizes the accumulated running statistics after all chunks are processed.

---

### `_transform_parallel`
```python
def _transform_parallel(self, X: np.ndarray) -> np.ndarray
```
**Description**:  
Transforms data using parallel execution.

---

### `_transform_raw`
```python
def _transform_raw(self, X: np.ndarray) -> np.ndarray
```
**Description**:  
Applies transformations sequentially to the input data without caching.

---

### `_transform_in_chunks`
```python
def _transform_in_chunks(self, X: np.ndarray) -> np.ndarray
```
**Description**:  
Processes transformation of large datasets in manageable chunks.

---

### `_hash_array`
```python
def _hash_array(self, arr: np.ndarray) -> str
```
**Description**:  
Generates a unique hash for a numpy array for caching purposes.

---

### `_compute_standard_stats`
```python
def _compute_standard_stats(self, X: np.ndarray) -> None
```
**Description**:  
Computes mean and standard deviation for standard normalization.

---

### `_compute_minmax_stats`
```python
def _compute_minmax_stats(self, X: np.ndarray) -> None
```
**Description**:  
Computes minimum and maximum for min-max normalization.

---

### `_compute_robust_stats`
```python
def _compute_robust_stats(self, X: np.ndarray) -> None
```
**Description**:  
Computes lower and upper percentiles for robust normalization.

---

### `_compute_custom_stats`
```python
def _compute_custom_stats(self, X: np.ndarray) -> None
```
**Description**:  
Computes custom center and scaling statistics using user-provided functions.

---

### `_compute_outlier_stats`
```python
def _compute_outlier_stats(self, X: np.ndarray) -> None
```
**Description**:  
Computes statistics for outlier detection (bounds based on IQR, z-score, or percentiles).

---

### `_handle_nan_values`
```python
def _handle_nan_values(self, X: np.ndarray) -> np.ndarray
```
**Description**:  
Fills NaN values according to the configured strategy (mean, median, etc.).

---

### `_handle_infinite_values`
```python
def _handle_infinite_values(self, X: np.ndarray) -> np.ndarray
```
**Description**:  
Replaces or handles infinite values according to configured strategy.

---

### `_handle_outliers`
```python
def _handle_outliers(self, X: np.ndarray) -> np.ndarray
```
**Description**:  
Handles detected outliers by clipping, replacing, winsorizing, or setting to NaN.

---

### `_clip_data`
```python
def _clip_data(self, X: np.ndarray) -> np.ndarray
```
**Description**:  
Clips input data values to a specified min/max range.

---

### `_normalize_data`
```python
def _normalize_data(self, X: np.ndarray) -> np.ndarray
```
**Description**:  
Applies normalization (standard, min-max, robust, etc.) to input data.

---

### `_get_norm_type`
```python
def _get_norm_type(self, normalization_type: Union[str, NormalizationType]) -> NormalizationType
```
**Description**:  
Converts string normalization types to their enum equivalents.

---

### `_update_stats_incrementally`
```python
def _update_stats_incrementally(self, X: np.ndarray) -> None
```
**Description**:  
Updates normalization statistics incrementally when new data is received.

---

### `_update_outlier_stats`
```python
def _update_outlier_stats(self, X: np.ndarray) -> None
```
**Description**:  
Updates outlier detection statistics incrementally when new data is received.

---
