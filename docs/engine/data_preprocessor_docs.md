# DataPreprocessor Documentation

The `DataPreprocessor` class is a comprehensive tool for preprocessing data in machine learning workflows. It supports multiple normalization strategies, outlier detection, and handling, along with advanced features like thread safety, memory optimization, and serialization for model persistence.

## Features

- **Multiple Normalization Strategies**: Supports standard, min-max, robust, and custom normalization.
- **Outlier Detection and Handling**: Detects and handles outliers using IQR, Z-score, or Isolation Forest methods.
- **Thread Safety**: Ensures safe concurrent processing with reentrant locks.
- **Memory Optimization**: Processes large datasets in chunks to reduce memory usage.
- **Comprehensive Error Handling**: Includes custom exceptions for input validation, statistics computation, and serialization errors.
- **Monitoring and Metrics**: Tracks performance metrics for fit and transform operations.
- **Serialization**: Supports saving and loading the preprocessor state for model persistence.
- **Caching**: Implements caching for repeated operations to improve performance.

## Class Methods

### `__init__(self, config: Optional[PreprocessorConfig] = None)`
Initializes the preprocessor with the specified configuration.

- **Parameters**:
  - `config`: Configuration object, or `None` to use defaults.

### `fit(self, X: Union[np.array, List], feature_names: Optional[List[str]] = None) -> 'DataPreprocessor'`
Computes preprocessing statistics with optimized memory usage and validation.

- **Parameters**:
  - `X`: Input data array or list.
  - `feature_names`: Optional list of feature names for better logging and monitoring.
- **Returns**: `self` for method chaining.

### `transform(self, X: Union[np.ndarray, List], copy: bool = True) -> np.ndarray`
Applies preprocessing transformations with minimal memory overhead.

- **Parameters**:
  - `X`: Input data.
  - `copy`: If `True`, ensures input is not modified in-place.
- **Returns**: Transformed data array.

### `fit_transform(self, X: Union[np.array, List], feature_names: Optional[List[str]] = None, copy: bool = True) -> np.array`
Combines fit and transform operations efficiently.

- **Parameters**:
  - `X`: Input data.
  - `feature_names`: Optional feature names.
  - `copy`: Whether to copy the input data.
- **Returns**: Transformed data.

### `partial_fit(self, X: Union[np.array, List], feature_names: Optional[List[str]] = None) -> 'DataPreprocessor'`
Updates preprocessing statistics with a new batch of data.

- **Parameters**:
  - `X`: New data batch.
  - `feature_names`: Optional feature names (must match previous names if already fitted).
- **Returns**: `self` for method chaining.

### `inverse_transform(self, X: Union[np.array, List], copy: bool = True) -> np.array`
Reverses the transformation process to get original scale data.

- **Parameters**:
  - `X`: Transformed data.
  - `copy`: Whether to copy the input data.
- **Returns**: Data in original scale.

### `get_stats(self) -> Dict[str, Any]`
Returns computed statistics with safeguards.

- **Returns**: Dictionary of computed statistics.

### `get_feature_names(self) -> List[str]`
Returns the feature names if available.

- **Returns**: List of feature names.

### `get_metrics(self) -> Dict[str, Any]`
Returns performance metrics for monitoring.

- **Returns**: Dictionary of performance metrics.

### `reset(self) -> None`
Resets the preprocessor state.

### `save(self, filepath: str) -> None`
Saves the preprocessor to a file with version information.

- **Parameters**:
  - `filepath`: Path to save the preprocessor.

### `load(cls, filepath: str) -> 'DataPreprocessor'`
Loads a preprocessor from a file.

- **Parameters**:
  - `filepath`: Path to load the preprocessor from.
- **Returns**: Loaded preprocessor instance.

### `get_memory_usage(self) -> Dict[str, Any]`
Estimates the memory usage of the preprocessor.

- **Returns**: Dictionary with memory usage information.

### `copy(self) -> 'DataPreprocessor'`
Creates a copy of the preprocessor.

- **Returns**: A new instance of `DataPreprocessor` with the same state.

## Custom Exceptions

- **`PreprocessingError`**: Base class for all preprocessing exceptions.
- **`InputValidationError`**: Raised for input validation errors.
- **`StatisticsError`**: Raised for errors in computing statistics.
- **`SerializationError`**: Raised for errors in serialization/deserialization.

## Example Usage

```python
from modules.configs import PreprocessorConfig, NormalizationType
from data_preprocessor import DataPreprocessor
import numpy as np

# Initialize preprocessor with custom configuration
config = PreprocessorConfig(
    normalization=NormalizationType.STANDARD,
    handle_nan=True,
    handle_inf=True,
    detect_outliers=True,
    parallel_processing=True
)
preprocessor = DataPreprocessor(config)

# Fit the preprocessor on training data
X_train = np.array([[1, 2], [3, 4], [5, 6]])
preprocessor.fit(X_train)

# Transform new data
X_test = np.array([[7, 8], [9, 10]])
X_transformed = preprocessor.transform(X_test)

# Save the preprocessor for later use
preprocessor.save('preprocessor.pkl')

# Load the preprocessor
loaded_preprocessor = DataPreprocessor.load('preprocessor.pkl')

# Inverse transform to get original data
X_original = loaded_preprocessor.inverse_transform(X_transformed)
```