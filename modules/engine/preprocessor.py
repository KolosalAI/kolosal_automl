from typing import Union, Dict
import numpy as np
from numpy.typing import NDArray
import warnings
from dataclasses import dataclass
from enum import Enum
import threading
import logging
from typing import Tuple
from modules.configs import PreprocessorConfig, NormalizationType  # Removed relative import

# Set up logging
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Optimized data preprocessor with enhanced functionality and performance.
    Supports multiple normalization methods, robust statistics, and parallel processing.
    """

    def __init__(self, config: 'PreprocessorConfig'):  # Forward reference for type hint
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

        if self.config.normalization != 'NormalizationType.NONE':  # Use string comparison
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
            normalization_type = self.config.normalization
            if normalization_type == 'NormalizationType.STANDARD':
                self._compute_standard_stats(X)
            elif normalization_type == 'NormalizationType.MINMAX':
                self._compute_minmax_stats(X)
            elif normalization_type == 'NormalizationType.ROBUST':
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
        mean = np.mean(X, axis=0, dtype=self.config.dtype)
        std = np.std(X, axis=0, dtype=self.config.dtype)
        std = np.maximum(std, self.config.epsilon)
        scale = 1.0 / std

        self._stats['mean'] = mean
        self._stats['std'] = std
        self._stats['scale'] = scale

    def _compute_minmax_stats(self, X: np.ndarray) -> None:
        """Compute statistics for min-max normalization."""
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        range_val = max_val - min_val
        range_val = np.maximum(range_val, self.config.epsilon)
        scale = 1.0 / range_val

        self._stats['min'] = min_val
        self._stats['max'] = max_val
        self._stats['range'] = range_val
        self._stats['scale'] = scale

    def _compute_robust_stats(self, X: np.ndarray) -> None:
        """Compute robust statistics using percentiles."""
        lower, upper = self.config.robust_percentiles
        lower_percentile = np.percentile(X, lower, axis=0)
        upper_percentile = np.percentile(X, upper, axis=0)
        range_val = upper_percentile - lower_percentile
        range_val = np.maximum(range_val, self.config.epsilon)
        scale = 1.0 / range_val

        self._stats['lower'] = lower_percentile
        self._stats['upper'] = upper_percentile
        self._stats['range'] = range_val
        self._stats['scale'] = scale

    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization based on configured method."""
        normalization_type = self.config.normalization
        if normalization_type == 'NormalizationType.STANDARD':
            return (X - self._stats['mean']) * self._stats['scale']
        elif normalization_type == 'NormalizationType.MINMAX':
            return (X - self._stats['min']) * self._stats['scale']
        elif normalization_type == 'NormalizationType.ROBUST':
            return (X - self._stats['lower']) * self._stats['scale']
        return X

    def _handle_infinite_values(self, X: np.ndarray) -> np.ndarray:
        """Handle infinite values in data."""
        mask = ~np.isfinite(X)
        if np.any(mask):
            num_inf = np.sum(mask)
            X[mask] = 0.0
            logger.warning(f"Found {num_inf} infinite values, replaced with 0.0")  # Use logger
        return X

    def _handle_nan_values(self, X: np.ndarray) -> np.ndarray:
        """Handle NaN values in data."""
        mask = np.isnan(X)
        if np.any(mask):
            num_nan = np.sum(mask)
            X[mask] = 0.0
            logger.warning(f"Found {num_nan} NaN values, replaced with 0.0")  # Use logger
        return X

    def _clip_data(self, X: np.ndarray) -> np.ndarray:
        """Clip data to specified range."""
        min_clip, max_clip = self.config.clip_range
        return np.clip(X, min_clip, max_clip, out=X)  # In-place clipping

    def get_stats(self) -> Dict[str, NDArray]:
        """Return computed statistics."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted yet")
        return {k: v.copy() for k, v in self._stats.items()}  # Return copies

    def reset(self) -> None:
        """Reset the preprocessor state."""
        self._stats.clear()
        self._fitted = False
