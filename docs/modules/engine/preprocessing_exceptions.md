# Preprocessing Exceptions (`modules/engine/preprocessing_exceptions.py`)

## Overview

The Preprocessing Exceptions module defines custom exception classes for the data preprocessing pipeline. It provides a structured hierarchy of exceptions that enables proper error handling and debugging throughout the preprocessing workflow.

## Features

- **Exception Hierarchy**: Structured exception inheritance for categorized error handling
- **Specific Error Types**: Targeted exceptions for different preprocessing stages
- **Debug-Friendly**: Clear exception names that indicate the source of problems
- **Pipeline Integration**: Seamless integration with preprocessing workflows

## Exception Classes

### PreprocessingError

Base exception class for all preprocessing-related errors:

```python
class PreprocessingError(Exception):
    """Base class for all preprocessing exceptions."""
    pass
```

This serves as the root exception that can catch any preprocessing-related error.

### InputValidationError

Exception for input validation failures:

```python
class InputValidationError(PreprocessingError):
    """Exception raised for input validation errors."""
    pass
```

Raised when input data fails validation checks (shape, type, range, etc.).

### StatisticsError

Exception for statistical computation errors:

```python
class StatisticsError(PreprocessingError):
    """Exception raised for errors in computing statistics."""
    pass
```

Raised when statistical calculations fail (mean, std, percentiles, etc.).

### SerializationError

Exception for serialization/deserialization failures:

```python
class SerializationError(PreprocessingError):
    """Exception raised for errors in serialization/deserialization."""
    pass
```

Raised when saving/loading preprocessors or statistical metadata fails.

## Usage Examples

### Basic Exception Handling

```python
from modules.engine.preprocessing_exceptions import (
    PreprocessingError,
    InputValidationError,
    StatisticsError,
    SerializationError
)
import numpy as np
import pandas as pd

def validate_input_data(data):
    """Example input validation with custom exceptions"""
    try:
        # Check if data is empty
        if data is None:
            raise InputValidationError("Input data cannot be None")
        
        # Check data type
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            raise InputValidationError(
                f"Expected numpy array or pandas DataFrame, got {type(data)}"
            )
        
        # Check for NaN values
        if isinstance(data, np.ndarray):
            if np.isnan(data).any():
                raise InputValidationError("Input data contains NaN values")
        elif isinstance(data, pd.DataFrame):
            if data.isnull().any().any():
                raise InputValidationError("Input data contains null values")
        
        # Check data shape
        if isinstance(data, np.ndarray):
            if data.ndim not in [1, 2]:
                raise InputValidationError(
                    f"Expected 1D or 2D array, got {data.ndim}D array"
                )
        
        print("Input validation passed")
        return True
        
    except InputValidationError as e:
        print(f"Input validation failed: {e}")
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise InputValidationError(f"Unexpected validation error: {e}")

# Test with valid data
valid_data = np.random.rand(100, 5)
validate_input_data(valid_data)

# Test with invalid data
try:
    invalid_data = None
    validate_input_data(invalid_data)
except InputValidationError as e:
    print(f"Caught expected error: {e}")
```

### Statistical Computation Error Handling

```python
from modules.engine.preprocessing_exceptions import StatisticsError
import numpy as np
import warnings

def compute_robust_statistics(data, feature_names=None):
    """Compute statistics with proper error handling"""
    try:
        if data.size == 0:
            raise StatisticsError("Cannot compute statistics on empty data")
        
        # Attempt to compute statistics
        statistics = {}
        
        try:
            statistics['mean'] = np.mean(data, axis=0)
        except Exception as e:
            raise StatisticsError(f"Failed to compute mean: {e}")
        
        try:
            statistics['std'] = np.std(data, axis=0)
            if np.any(statistics['std'] == 0):
                warnings.warn("Zero standard deviation detected in some features")
        except Exception as e:
            raise StatisticsError(f"Failed to compute standard deviation: {e}")
        
        try:
            statistics['median'] = np.median(data, axis=0)
        except Exception as e:
            raise StatisticsError(f"Failed to compute median: {e}")
        
        try:
            statistics['percentiles'] = {
                'p25': np.percentile(data, 25, axis=0),
                'p75': np.percentile(data, 75, axis=0)
            }
        except Exception as e:
            raise StatisticsError(f"Failed to compute percentiles: {e}")
        
        # Validate computed statistics
        for stat_name, stat_values in statistics.items():
            if stat_name == 'percentiles':
                for percentile_name, percentile_values in stat_values.items():
                    if np.isnan(percentile_values).any():
                        raise StatisticsError(f"NaN values in {percentile_name}")
            else:
                if np.isnan(stat_values).any():
                    raise StatisticsError(f"NaN values in {stat_name}")
        
        return statistics
        
    except StatisticsError:
        raise
    except Exception as e:
        raise StatisticsError(f"Unexpected error in statistics computation: {e}")

# Example usage
try:
    # Valid data
    data = np.random.randn(1000, 10)
    stats = compute_robust_statistics(data)
    print("Statistics computed successfully")
    print(f"Mean shape: {stats['mean'].shape}")
    
    # Invalid data (empty)
    empty_data = np.array([]).reshape(0, 10)
    stats = compute_robust_statistics(empty_data)
    
except StatisticsError as e:
    print(f"Statistics computation failed: {e}")
```

### Serialization Error Handling

```python
from modules.engine.preprocessing_exceptions import SerializationError
import pickle
import json
import numpy as np
from pathlib import Path

class PreprocessorSerializer:
    """Example serializer with proper error handling"""
    
    @staticmethod
    def save_preprocessor(preprocessor, filepath, format='pickle'):
        """Save preprocessor with error handling"""
        try:
            filepath = Path(filepath)
            
            # Validate input
            if preprocessor is None:
                raise SerializationError("Cannot serialize None preprocessor")
            
            # Create directory if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'pickle':
                try:
                    with open(filepath, 'wb') as f:
                        pickle.dump(preprocessor, f)
                except Exception as e:
                    raise SerializationError(f"Failed to pickle preprocessor: {e}")
                    
            elif format == 'json':
                try:
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_preprocessor = {}
                    for key, value in preprocessor.items():
                        if isinstance(value, np.ndarray):
                            serializable_preprocessor[key] = {
                                'data': value.tolist(),
                                'dtype': str(value.dtype),
                                'shape': value.shape
                            }
                        else:
                            serializable_preprocessor[key] = value
                    
                    with open(filepath, 'w') as f:
                        json.dump(serializable_preprocessor, f, indent=2)
                except Exception as e:
                    raise SerializationError(f"Failed to save as JSON: {e}")
            else:
                raise SerializationError(f"Unsupported format: {format}")
            
            print(f"Preprocessor saved to {filepath}")
            
        except SerializationError:
            raise
        except Exception as e:
            raise SerializationError(f"Unexpected serialization error: {e}")
    
    @staticmethod
    def load_preprocessor(filepath, format='pickle'):
        """Load preprocessor with error handling"""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise SerializationError(f"File does not exist: {filepath}")
            
            if format == 'pickle':
                try:
                    with open(filepath, 'rb') as f:
                        preprocessor = pickle.load(f)
                except Exception as e:
                    raise SerializationError(f"Failed to unpickle preprocessor: {e}")
                    
            elif format == 'json':
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Reconstruct numpy arrays
                    preprocessor = {}
                    for key, value in data.items():
                        if isinstance(value, dict) and 'data' in value:
                            # This is a serialized numpy array
                            array_data = np.array(value['data'], dtype=value['dtype'])
                            preprocessor[key] = array_data.reshape(value['shape'])
                        else:
                            preprocessor[key] = value
                            
                except Exception as e:
                    raise SerializationError(f"Failed to load from JSON: {e}")
            else:
                raise SerializationError(f"Unsupported format: {format}")
            
            return preprocessor
            
        except SerializationError:
            raise
        except Exception as e:
            raise SerializationError(f"Unexpected deserialization error: {e}")

# Example usage
try:
    # Create sample preprocessor
    sample_preprocessor = {
        'scaler_mean': np.random.rand(10),
        'scaler_std': np.random.rand(10),
        'feature_names': ['feature_' + str(i) for i in range(10)],
        'version': '1.0'
    }
    
    # Save successfully
    PreprocessorSerializer.save_preprocessor(
        sample_preprocessor, 
        'models/preprocessor.pkl',
        format='pickle'
    )
    
    # Load successfully
    loaded_preprocessor = PreprocessorSerializer.load_preprocessor(
        'models/preprocessor.pkl',
        format='pickle'
    )
    print("Serialization test passed")
    
    # Test with invalid file
    invalid_preprocessor = PreprocessorSerializer.load_preprocessor(
        'nonexistent/file.pkl'
    )
    
except SerializationError as e:
    print(f"Serialization error (expected): {e}")
```

### Comprehensive Error Handling Pipeline

```python
from modules.engine.preprocessing_exceptions import (
    PreprocessingError,
    InputValidationError,
    StatisticsError,
    SerializationError
)
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessingPipeline:
    """Example preprocessing pipeline with comprehensive error handling"""
    
    def __init__(self):
        self.statistics = None
        self.fitted = False
    
    def preprocess(self, data, save_path=None):
        """Complete preprocessing pipeline with error handling"""
        try:
            # Step 1: Input validation
            logger.info("Starting input validation...")
            self._validate_input(data)
            logger.info("Input validation completed")
            
            # Step 2: Compute statistics
            logger.info("Computing statistics...")
            self.statistics = self._compute_statistics(data)
            logger.info("Statistics computation completed")
            
            # Step 3: Transform data
            logger.info("Transforming data...")
            transformed_data = self._transform_data(data)
            logger.info("Data transformation completed")
            
            # Step 4: Save preprocessor (optional)
            if save_path:
                logger.info(f"Saving preprocessor to {save_path}...")
                self._save_preprocessor(save_path)
                logger.info("Preprocessor saved successfully")
            
            self.fitted = True
            return transformed_data
            
        except InputValidationError as e:
            logger.error(f"Input validation failed: {e}")
            raise
        except StatisticsError as e:
            logger.error(f"Statistics computation failed: {e}")
            raise
        except SerializationError as e:
            logger.error(f"Serialization failed: {e}")
            raise
        except PreprocessingError as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in preprocessing: {e}")
            raise PreprocessingError(f"Unexpected preprocessing error: {e}")
    
    def _validate_input(self, data):
        """Validate input data"""
        if data is None:
            raise InputValidationError("Input data is None")
        
        if not isinstance(data, np.ndarray):
            raise InputValidationError(f"Expected numpy array, got {type(data)}")
        
        if data.ndim != 2:
            raise InputValidationError(f"Expected 2D array, got {data.ndim}D")
        
        if data.size == 0:
            raise InputValidationError("Input data is empty")
        
        if np.isnan(data).any():
            raise InputValidationError("Input data contains NaN values")
    
    def _compute_statistics(self, data):
        """Compute preprocessing statistics"""
        try:
            stats = {
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0),
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0)
            }
            
            # Check for zero standard deviation
            if np.any(stats['std'] == 0):
                raise StatisticsError("Zero standard deviation detected")
            
            return stats
            
        except Exception as e:
            raise StatisticsError(f"Failed to compute statistics: {e}")
    
    def _transform_data(self, data):
        """Transform data using computed statistics"""
        if self.statistics is None:
            raise PreprocessingError("Statistics not computed. Call preprocess() first.")
        
        try:
            # Standard scaling
            transformed = (data - self.statistics['mean']) / self.statistics['std']
            
            # Validate transformation
            if np.isnan(transformed).any():
                raise PreprocessingError("Transformation produced NaN values")
            
            return transformed
            
        except Exception as e:
            raise PreprocessingError(f"Data transformation failed: {e}")
    
    def _save_preprocessor(self, filepath):
        """Save preprocessor state"""
        try:
            state = {
                'statistics': self.statistics,
                'fitted': self.fitted
            }
            
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
        except Exception as e:
            raise SerializationError(f"Failed to save preprocessor: {e}")

# Example usage with comprehensive error handling
pipeline = DataPreprocessingPipeline()

# Test cases
test_cases = [
    ("Valid data", np.random.randn(100, 5)),
    ("Invalid data (None)", None),
    ("Invalid data (1D)", np.random.randn(100)),
    ("Invalid data (NaN)", np.array([[1, 2, np.nan], [4, 5, 6]])),
    ("Invalid data (empty)", np.array([]).reshape(0, 5))
]

for test_name, test_data in test_cases:
    try:
        print(f"\nTesting: {test_name}")
        result = pipeline.preprocess(test_data, save_path=f"models/preprocessor_{test_name.replace(' ', '_')}.pkl")
        print(f"  Success: Output shape {result.shape}")
        
    except InputValidationError as e:
        print(f"  Input validation error: {e}")
    except StatisticsError as e:
        print(f"  Statistics error: {e}")
    except SerializationError as e:
        print(f"  Serialization error: {e}")
    except PreprocessingError as e:
        print(f"  Preprocessing error: {e}")
```

## Error Recovery Patterns

### Graceful Degradation

```python
from modules.engine.preprocessing_exceptions import (
    PreprocessingError,
    InputValidationError,
    StatisticsError
)
import numpy as np
import warnings

def robust_preprocessing_with_fallback(data):
    """Preprocessing with graceful degradation and fallback strategies"""
    try:
        # Primary preprocessing strategy
        return primary_preprocessing(data)
        
    except InputValidationError as e:
        warnings.warn(f"Input validation failed, trying fallback: {e}")
        try:
            return fallback_preprocessing(data)
        except Exception as fallback_error:
            raise PreprocessingError(f"Both primary and fallback failed: {e}, {fallback_error}")
    
    except StatisticsError as e:
        warnings.warn(f"Statistics computation failed, using defaults: {e}")
        return default_preprocessing(data)
    
    except PreprocessingError as e:
        # Log error but don't fail completely
        warnings.warn(f"Preprocessing failed, returning raw data: {e}")
        return data

def primary_preprocessing(data):
    """Primary preprocessing strategy"""
    if data is None or data.size == 0:
        raise InputValidationError("Invalid input data")
    
    # Complex preprocessing that might fail
    if np.std(data) == 0:
        raise StatisticsError("Zero variance data")
    
    return (data - np.mean(data)) / np.std(data)

def fallback_preprocessing(data):
    """Fallback preprocessing strategy"""
    # Simple min-max scaling
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max == data_min:
        return np.zeros_like(data)
    
    return (data - data_min) / (data_max - data_min)

def default_preprocessing(data):
    """Default preprocessing when all else fails"""
    # Just return normalized data
    return data / np.linalg.norm(data)

# Example usage
test_data = [
    np.random.randn(100),      # Normal case
    np.ones(100),              # Zero variance case
    np.array([]),              # Empty case
    None                       # None case
]

for i, data in enumerate(test_data):
    try:
        result = robust_preprocessing_with_fallback(data)
        print(f"Test {i}: Success")
    except PreprocessingError as e:
        print(f"Test {i}: Failed - {e}")
```

## Best Practices

### 1. Exception Chaining

```python
# Preserve original exception information
try:
    # Some operation that might fail
    problematic_operation()
except OriginalError as e:
    raise PreprocessingError("Preprocessing failed") from e
```

### 2. Specific Exception Types

```python
# Use most specific exception type
if data.dtype not in [np.float32, np.float64]:
    raise InputValidationError(f"Unsupported data type: {data.dtype}")

# Not just:
# raise PreprocessingError("Bad data type")
```

### 3. Error Context

```python
# Provide helpful context in exception messages
try:
    compute_statistics(data)
except Exception as e:
    raise StatisticsError(
        f"Failed to compute statistics for data shape {data.shape}, "
        f"dtype {data.dtype}: {e}"
    )
```

### 4. Logging Integration

```python
import logging

logger = logging.getLogger(__name__)

try:
    risky_operation()
except PreprocessingError as e:
    logger.error(f"Preprocessing failed: {e}", exc_info=True)
    raise
```

## Related Documentation

- [Data Preprocessor Documentation](data_preprocessor.md)
- [Training Engine Documentation](train_engine.md)
- [Inference Engine Documentation](inference_engine.md)
- [Configuration System Documentation](../configs.md)
