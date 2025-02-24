from typing import Union, Dict
import numpy as np
from numpy.typing import NDArray
import warnings
from dataclasses import dataclass
from enum import Enum
import threading
'''
class NormalizationType(Enum):
    STANDARD = "standard"  # (x - mean) / std
    MINMAX = "minmax"     # (x - min) / (max - min)
    ROBUST = "robust"     # Using percentiles instead of min/max
    NONE = "none"

@dataclass
class PreprocessorConfig:
    """Configuration for data preprocessing."""
    normalization: NormalizationType = NormalizationType.STANDARD
    epsilon: float = 1e-8
    clip_values: bool = True
    clip_range: tuple[float, float] = (-5.0, 5.0)
    robust_percentiles: tuple[float, float] = (1.0, 99.0)
    handle_inf: bool = True
    handle_nan: bool = True
    dtype: np.dtype = np.float32
'''

class DataPreprocessor:
    """
    Optimized data preprocessor with enhanced functionality and performance.
    Supports multiple normalization methods, robust statistics, and parallel processing.
    """

    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self._stats: Dict[str, NDArray] = {}
        self._fitted = False
        self._lock = threading.Lock()
        
        # Initialize processing functions based on config
        self._init_processing_functions()

    def _init_processing_functions(self) -> None:
        """Initialize preprocessing functions based on configuration."""
        self._processors = []
        
        if self.config.handle_inf:
            self._processors.append(self._handle_infinite_values)
        if self.config.handle_nan:
            self._processors.append(self._handle_nan_values)
            
        if self.config.normalization != NormalizationType.NONE:
            self._processors.append(self._normalize_data)
            
        if self.config.clip_values:
            self._processors.append(self._clip_data)

    def fit(self, X: Union[np.ndarray, list]) -> 'DataPreprocessor':
        """
        Compute preprocessing statistics with optimized memory usage.
        
        Args:
            X: Input data array or list
            
        Returns:
            self for method chaining
        """
        with self._lock:
            # Convert input to correct dtype if necessary
            X = self._validate_and_convert_input(X)
            
            # Compute statistics based on normalization type
            if self.config.normalization == NormalizationType.STANDARD:
                self._compute_standard_stats(X)
            elif self.config.normalization == NormalizationType.MINMAX:
                self._compute_minmax_stats(X)
            elif self.config.normalization == NormalizationType.ROBUST:
                self._compute_robust_stats(X)
                
            self._fitted = True
            return self

    def transform(self, X: Union[np.ndarray, list], copy: bool = True) -> np.ndarray:
        """
        Apply preprocessing transformations with minimal memory overhead.
        
        Args:
            X: Input data
            copy: If True, ensures input is not modified in-place
            
        Returns:
            Transformed data array
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
            
        X = self._validate_and_convert_input(X, copy=copy)
        
        # Apply all preprocessing functions in sequence
        for processor in self._processors:
            X = processor(X)
            
        return X

    def fit_transform(self, X: Union[np.ndarray, list], copy: bool = True) -> np.ndarray:
        """Combine fit and transform operations efficiently."""
        return self.fit(X).transform(X, copy=copy)

    def _validate_and_convert_input(
        self, X: Union[np.ndarray, list], copy: bool = False
    ) -> np.ndarray:
        """Validate and convert input data efficiently."""
        if isinstance(X, list):
            X = np.array(X, dtype=self.config.dtype)
        elif not isinstance(X, np.ndarray):
            raise TypeError(f"Expected numpy array or list, got {type(X)}")
        
        if X.dtype != self.config.dtype:
            X = X.astype(self.config.dtype, copy=copy)
        elif copy:
            X = X.copy()
            
        return X

    def _compute_standard_stats(self, X: np.ndarray) -> None:
        """Compute statistics for standard normalization."""
        self._stats['mean'] = np.mean(X, axis=0, dtype=self.config.dtype)
        self._stats['std'] = np.std(X, axis=0, dtype=self.config.dtype)
        # Avoid division by zero
        self._stats['std'] = np.maximum(self._stats['std'], self.config.epsilon)
        # Precompute scale
        self._stats['scale'] = 1.0 / self._stats['std']

    def _compute_minmax_stats(self, X: np.ndarray) -> None:
        """Compute statistics for min-max normalization."""
        self._stats['min'] = np.min(X, axis=0)
        self._stats['max'] = np.max(X, axis=0)
        self._stats['range'] = self._stats['max'] - self._stats['min']
        # Avoid division by zero
        self._stats['range'] = np.maximum(self._stats['range'], self.config.epsilon)
        self._stats['scale'] = 1.0 / self._stats['range']

    def _compute_robust_stats(self, X: np.ndarray) -> None:
        """Compute robust statistics using percentiles."""
        lower, upper = self.config.robust_percentiles
        self._stats['lower'] = np.percentile(X, lower, axis=0)
        self._stats['upper'] = np.percentile(X, upper, axis=0)
        self._stats['range'] = self._stats['upper'] - self._stats['lower']
        self._stats['range'] = np.maximum(self._stats['range'], self.config.epsilon)
        self._stats['scale'] = 1.0 / self._stats['range']

    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization based on configured method."""
        if self.config.normalization == NormalizationType.STANDARD:
            return (X - self._stats['mean']) * self._stats['scale']
        elif self.config.normalization == NormalizationType.MINMAX:
            return (X - self._stats['min']) * self._stats['scale']
        elif self.config.normalization == NormalizationType.ROBUST:
            return (X - self._stats['lower']) * self._stats['scale']
        return X

    def _handle_infinite_values(self, X: np.ndarray) -> np.ndarray:
        """Handle infinite values in data."""
        mask = ~np.isfinite(X)
        if np.any(mask):
            X[mask] = 0.0
            warnings.warn(f"Found {np.sum(mask)} infinite values, replaced with 0.0")
        return X

    def _handle_nan_values(self, X: np.ndarray) -> np.ndarray:
        """Handle NaN values in data."""
        mask = np.isnan(X)
        if np.any(mask):
            X[mask] = 0.0
            warnings.warn(f"Found {np.sum(mask)} NaN values, replaced with 0.0")
        return X

    def _clip_data(self, X: np.ndarray) -> np.ndarray:
        """Clip data to specified range."""
        return np.clip(X, self.config.clip_range[0], self.config.clip_range[1])

    def get_stats(self) -> Dict[str, NDArray]:
        """Return computed statistics."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted yet")
        return self._stats.copy()

    def reset(self) -> None:
        """Reset the preprocessor state."""
        self._stats.clear()
        self._fitted = False