import os
import logging
import threading
import numpy as np
import hashlib
import json
import pickle
import traceback
import warnings
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from functools import lru_cache, wraps
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
import time


class NormalizationType(Enum):
    """Enumeration of supported normalization strategies."""
    NONE = auto()
    STANDARD = auto()  # z-score normalization: (x - mean) / std
    MINMAX = auto()    # Min-max scaling: (x - min) / (max - min)
    ROBUST = auto()    # Robust scaling using percentiles
    CUSTOM = auto()    # Custom normalization function


class PreprocessingError(Exception):
    """Base class for all preprocessing exceptions."""
    pass


class InputValidationError(PreprocessingError):
    """Exception raised for input validation errors."""
    pass


class StatisticsError(PreprocessingError):
    """Exception raised for errors in computing statistics."""
    pass


class SerializationError(PreprocessingError):
    """Exception raised for errors in serialization/deserialization."""
    pass


@dataclass
class PreprocessorConfig:
    """Configuration for the data preprocessor."""
    # Normalization settings
    normalization: NormalizationType = NormalizationType.STANDARD
    robust_percentiles: Tuple[float, float] = (25.0, 75.0)
    
    # Data handling settings
    handle_nan: bool = True
    handle_inf: bool = True
    nan_strategy: str = "mean"  # Options: "mean", "median", "most_frequent", "zero"
    inf_strategy: str = "mean"  # Options: "mean", "median", "zero"
    
    # Outlier handling
    detect_outliers: bool = False
    outlier_method: str = "iqr"  # Options: "iqr", "zscore", "isolation_forest"
    outlier_params: Dict[str, Any] = field(default_factory=lambda: {
        "threshold": 1.5,  # For IQR: threshold * IQR, for zscore: threshold * std
        "clip": True,      # Whether to clip or replace outliers
        "n_estimators": 100,  # For isolation_forest
        "contamination": "auto"  # For isolation_forest
    })
    
    # Data clipping
    clip_values: bool = False
    clip_range: Tuple[float, float] = (-np.inf, np.inf)
    
    # Data validation
    enable_input_validation: bool = True
    input_size_limit: Optional[int] = None  # Max number of samples
    
    # Performance settings
    parallel_processing: bool = False
    n_jobs: int = -1  # -1 for all cores
    chunk_size: Optional[int] = None  # Process large data in chunks of this size
    cache_enabled: bool = True
    cache_size: int = 128  # LRU cache size
    
    # Numeric precision
    dtype: np.dtype = np.float64
    epsilon: float = 1e-10  # Small value to avoid division by zero
    
    # Debugging and logging
    debug_mode: bool = False
    
    # Custom functions
    custom_normalization_fn: Optional[Callable] = None
    custom_transform_fn: Optional[Callable] = None
    
    # Version info
    version: str = "1.0.0"


def log_operation(func):
    """
    Decorator to log function calls and measure execution time.
    Also handles updating metrics for the decorated function.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start_time
            
            # Add debug logging if enabled
            if hasattr(self, 'logger') and hasattr(self, 'config'):
                if self.config.debug_mode:
                    self.logger.debug(f"{func.__name__} completed in {elapsed:.4f} seconds")
            
            # Update metrics if the method exists
            if hasattr(self, '_update_metrics'):
                self._update_metrics(func.__name__, elapsed)
                
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            if hasattr(self, 'logger'):
                self.logger.error(f"{func.__name__} failed after {elapsed:.4f} seconds: {str(e)}")
            raise
    return wrapper


class DataPreprocessor:
    """
    Advanced data preprocessor with enhanced functionality, performance optimizations,
    and production-ready features for machine learning deployments.
    
    Features:
    - Multiple normalization strategies with robust statistics
    - Outlier detection and handling
    - Thread-safety for concurrent processing
    - Memory optimization for large datasets
    - Comprehensive error handling and validation
    - Monitoring and metrics collection
    - Serialization for model persistence
    - Caching for repeated operations
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        """
        Initialize the preprocessor with the specified configuration.
        
        Args:
            config: Configuration object, or None to use defaults
        """
        self.config = config or PreprocessorConfig()
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.DataPreprocessor")
        self._configure_logging()
        
        # Initialize state variables
        self._stats: Dict[str, Any] = {}
        self._feature_names: List[str] = []
        self._fitted = False
        self._n_features = 0
        self._n_samples_seen = 0
        
        # Thread safety
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        
        # Performance monitoring
        self._metrics: Dict[str, List[float]] = {
            "fit_time": [],
            "transform_time": [],
            "process_chunk_time": []
        }
        
        # Initialize processing functions
        self._init_processing_functions()
        
        # Configure LRU cache if enabled
        if self.config.cache_enabled:
            self._cached_transform = lru_cache(maxsize=self.config.cache_size)(self._transform_raw)
        else:
            self._cached_transform = self._transform_raw
            
        # Set the processor version
        self._version = self.config.version
        
        # Initialize parallel processing if enabled
        self._parallel_executor = None
        if self.config.parallel_processing:
            self._init_parallel_executor()
        
        # Initialize running stats
        self._running_stats = None
        
        self.logger.info(f"DataPreprocessor initialized with version {self._version}")

    def _configure_logging(self) -> None:
        """Configure logging based on config settings."""
        level = logging.DEBUG if self.config.debug_mode else logging.INFO
        self.logger.setLevel(level)
        
        # Only add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _init_parallel_executor(self) -> None:
        """Initialize parallel processing executor."""
        try:
            from concurrent.futures import ThreadPoolExecutor
            n_jobs = self.config.n_jobs
            if n_jobs < 1:
                # Use all cores if n_jobs is -1
                import multiprocessing
                n_jobs = multiprocessing.cpu_count()
            
            self._parallel_executor = ThreadPoolExecutor(
                max_workers=n_jobs,
                thread_name_prefix="Preprocessor"
            )
            self.logger.info(f"Parallel processing initialized with {n_jobs} workers")
        except ImportError:
            self.logger.warning("ThreadPoolExecutor not available, parallel processing disabled")
            self.config.parallel_processing = False

    def _init_processing_functions(self) -> None:
        """Initialize preprocessing functions based on configuration."""
        self._processors = []

        # Add data quality processors first
        if self.config.handle_inf:
            self._processors.append(self._handle_infinite_values)
        if self.config.handle_nan:
            self._processors.append(self._handle_nan_values)

        # Add outlier detection if enabled
        if self.config.detect_outliers:
            self._processors.append(self._handle_outliers)

        # Add normalization if enabled
        if self.config.normalization != NormalizationType.NONE:
            self._processors.append(self._normalize_data)

        # Add clipping if enabled
        if self.config.clip_values:
            self._processors.append(self._clip_data)

    @contextmanager
    def _error_context(self, operation: str):
        """Context manager for error handling and reporting."""
        try:
            yield
        except Exception as e:
            error_msg = f"Error during {operation}: {str(e)}"
            self.logger.error(error_msg)
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            
            # Convert to appropriate exception type
            if "validation" in operation.lower():
                raise InputValidationError(error_msg) from e
            elif "statistics" in operation.lower():
                raise StatisticsError(error_msg) from e
            elif "serialization" in operation.lower():
                raise SerializationError(error_msg) from e
            else:
                raise PreprocessingError(error_msg) from e

    @log_operation
    def fit(self, X: Union[np.ndarray, List], feature_names: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Compute preprocessing statistics with optimized memory usage and validation.

        Args:
            X: Input data array or list
            feature_names: Optional list of feature names for better logging and monitoring

        Returns:
            self for method chaining
        """
        with self._lock, self._error_context("fit operation"):
            # Validate and convert input
            X = self._validate_and_convert_input(X, operation="fit")
            
            # Store feature count and names
            self._n_features = X.shape[1] if X.ndim > 1 else 1
            if feature_names:
                if len(feature_names) != self._n_features:
                    raise InputValidationError(
                        f"Feature names length ({len(feature_names)}) doesn't match data dimensions ({self._n_features})"
                    )
                self._feature_names = feature_names
            else:
                self._feature_names = [f"feature_{i}" for i in range(self._n_features)]
            
            # Process in chunks if specified and data is large
            if self.config.chunk_size and X.shape[0] > self.config.chunk_size:
                self._fit_in_chunks(X)
            else:
                self._fit_full(X)
            
            self._fitted = True
            self._n_samples_seen = X.shape[0]
            
            # Calculate and store median for each feature (used for imputation)
            if self.config.nan_strategy == "median" or self.config.inf_strategy == "median":
                self._stats['median'] = np.median(X, axis=0)
                
            # Calculate and store most frequent values if needed for imputation
            if self.config.nan_strategy == "most_frequent":
                self._compute_most_frequent_values(X)
            
            self.logger.info(f"Fitted preprocessor on {self._n_samples_seen} samples, {self._n_features} features")
            return self

    def _compute_most_frequent_values(self, X: np.ndarray) -> None:
        """Compute most frequent value for each feature."""
        most_frequent = np.zeros(X.shape[1], dtype=self.config.dtype)
        
        # Find most frequent value for each column
        for i in range(X.shape[1]):
            # Get unique values and counts
            unique_vals, counts = np.unique(X[:, i], return_counts=True)
            # Get the most frequent value
            if len(unique_vals) > 0:
                most_frequent[i] = unique_vals[np.argmax(counts)]
            else:
                most_frequent[i] = 0.0
                
        self._stats['most_frequent'] = most_frequent

    def _fit_full(self, X: np.ndarray) -> None:
        """Compute statistics on the full dataset."""
        norm_type = self.config.normalization
        
        if norm_type == NormalizationType.STANDARD:
            self._compute_standard_stats(X)
        elif norm_type == NormalizationType.MINMAX:
            self._compute_minmax_stats(X)
        elif norm_type == NormalizationType.ROBUST:
            self._compute_robust_stats(X)
        elif norm_type == NormalizationType.CUSTOM:
            self._compute_custom_stats(X)
            
        # Compute outlier statistics if needed
        if self.config.detect_outliers:
            self._compute_outlier_stats(X)

    def _fit_in_chunks(self, X: np.ndarray) -> None:
        """
        Fit on large datasets by processing in chunks to reduce memory usage.
        Incrementally updates statistics.
        """
        self.logger.info(f"Processing {X.shape[0]} samples in chunks of {self.config.chunk_size}")
        
        # Initialize running statistics based on normalization type
        self._init_running_stats()
        
        # Process each chunk
        chunk_size = self.config.chunk_size
        n_samples = X.shape[0]
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = X[start_idx:end_idx]
            
            # Update statistics with this chunk
            self._update_stats_with_chunk(chunk)
            
            if self.config.debug_mode and (start_idx // chunk_size) % 10 == 0:
                self.logger.debug(f"Processed chunk {start_idx//chunk_size + 1}, "
                                 f"{end_idx}/{n_samples} samples")
        
        # Finalize statistics
        self._finalize_running_stats()

    def _init_running_stats(self) -> None:
        """Initialize running statistics for incremental fitting."""
        norm_type = self.config.normalization
        
        if norm_type == NormalizationType.STANDARD:
            self._running_stats = {
                'n_samples': 0,
                'sum': 0,
                'sum_squared': 0
            }
        elif norm_type == NormalizationType.MINMAX:
            self._running_stats = {
                'min': None,
                'max': None
            }
        elif norm_type == NormalizationType.ROBUST:
            # For robust statistics, we need to collect values for percentiles
            # Use reservoir sampling for memory efficiency
            self._running_stats = {
                'reservoir': [],
                'reservoir_size': 10000,  # Maximum samples to store
                'n_seen': 0
            }

    def _update_stats_with_chunk(self, chunk: np.ndarray) -> None:
        """Update running statistics with a new data chunk."""
        norm_type = self.config.normalization
        
        if norm_type == NormalizationType.STANDARD:
            # Update mean and std incrementally
            n_samples = chunk.shape[0]
            self._running_stats['n_samples'] += n_samples
            self._running_stats['sum'] += np.sum(chunk, axis=0)
            self._running_stats['sum_squared'] += np.sum(chunk ** 2, axis=0)
            
        elif norm_type == NormalizationType.MINMAX:
            # Update min and max
            if self._running_stats['min'] is None:
                self._running_stats['min'] = np.min(chunk, axis=0)
                self._running_stats['max'] = np.max(chunk, axis=0)
            else:
                self._running_stats['min'] = np.minimum(self._running_stats['min'], np.min(chunk, axis=0))
                self._running_stats['max'] = np.maximum(self._running_stats['max'], np.max(chunk, axis=0))
                
        elif norm_type == NormalizationType.ROBUST:
            # Implement reservoir sampling to maintain a representative sample
            # while keeping memory usage bounded
            reservoir = self._running_stats['reservoir']
            reservoir_size = self._running_stats['reservoir_size']
            n_seen = self._running_stats['n_seen']
            
            # If reservoir isn't full yet, add all rows from this chunk
            if len(reservoir) < reservoir_size:
                # How many more samples can fit in the reservoir
                space_left = reservoir_size - len(reservoir)
                # Take either all or as many as can fit
                samples_to_add = min(chunk.shape[0], space_left)
                reservoir.extend(chunk[:samples_to_add].tolist())
            else:
                # Reservoir is full, use reservoir sampling
                for row in chunk:
                    n_seen += 1
                    # Randomly replace items with decreasing probability
                    j = np.random.randint(0, n_seen)
                    if j < reservoir_size:
                        reservoir[j] = row.tolist()
            
            # Update the count of samples seen
            self._running_stats['n_seen'] = n_seen + chunk.shape[0]

    def _finalize_running_stats(self) -> None:
        """Finalize statistics after processing all chunks."""
        if self._running_stats is None:
            self.logger.warning("No running stats to finalize")
            return
            
        norm_type = self.config.normalization
        
        if norm_type == NormalizationType.STANDARD:
            # Compute final mean and std
            n_samples = self._running_stats['n_samples']
            sum_values = self._running_stats['sum']
            sum_squared = self._running_stats['sum_squared']
            
            mean = sum_values / n_samples
            # Use numerically stable method to compute variance
            var = (sum_squared / n_samples) - (mean ** 2)
            # Ensure variance is positive
            var = np.maximum(var, 0.0)
            std = np.sqrt(np.maximum(var, self.config.epsilon))
            scale = 1.0 / std
            
            self._stats['mean'] = mean
            self._stats['std'] = std
            self._stats['scale'] = scale
            
        elif norm_type == NormalizationType.MINMAX:
            # Set final min and max
            min_val = self._running_stats['min']
            max_val = self._running_stats['max']
            range_val = max_val - min_val
            range_val = np.maximum(range_val, self.config.epsilon)
            scale = 1.0 / range_val
            
            self._stats['min'] = min_val
            self._stats['max'] = max_val
            self._stats['range'] = range_val
            self._stats['scale'] = scale
            
        elif norm_type == NormalizationType.ROBUST:
            # Compute percentiles from collected samples
            if 'reservoir' in self._running_stats and self._running_stats['reservoir']:
                try:
                    # Convert list of lists to numpy array
                    samples = np.array(self._running_stats['reservoir'])
                    lower, upper = self.config.robust_percentiles
                    lower_percentile = np.percentile(samples, lower, axis=0)
                    upper_percentile = np.percentile(samples, upper, axis=0)
                    
                    range_val = upper_percentile - lower_percentile
                    range_val = np.maximum(range_val, self.config.epsilon)
                    scale = 1.0 / range_val
                    
                    self._stats['lower'] = lower_percentile
                    self._stats['upper'] = upper_percentile
                    self._stats['range'] = range_val
                    self._stats['scale'] = scale
                except Exception as e:
                    self.logger.error(f"Error computing percentiles: {e}")
                    # Fall back to min-max as a safer alternative
                    min_val = np.min(samples, axis=0)
                    max_val = np.max(samples, axis=0)
                    range_val = max_val - min_val
                    range_val = np.maximum(range_val, self.config.epsilon)
                    scale = 1.0 / range_val
                    
                    self._stats['min'] = min_val
                    self._stats['max'] = max_val
                    self._stats['range'] = range_val
                    self._stats['scale'] = scale
                    self.logger.warning("Fell back to min-max scaling due to error in percentile calculation")
            else:
                raise StatisticsError("No samples collected for robust statistics")
        
        # Clean up running stats to free memory
        self._running_stats = None

    @log_operation
    def transform(self, X: Union[np.ndarray, List], copy: bool = True) -> np.ndarray:
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

        with self._error_context("transform operation"):
            # Validate and convert input
            X = self._validate_and_convert_input(X, operation="transform", copy=copy)
            
            # Use parallel processing if enabled and data is large enough
            if (self.config.parallel_processing and self._parallel_executor and 
                X.shape[0] > 1000 and self.config.chunk_size):
                return self._transform_parallel(X)
            
            # Generate hash for caching if enabled
            if self.config.cache_enabled and X.shape[0] <= 1000:  # Only cache smaller arrays
                # Use a faster hashing approach for numpy arrays
                X_hash = self._hash_array(X)
                return self._cached_transform(X, X_hash)
            else:
                return self._transform_raw(X)

    def _transform_parallel(self, X: np.ndarray) -> np.ndarray:
        """Transform data using parallel processing for large datasets."""
        # Split data into chunks
        chunk_size = self.config.chunk_size
        n_samples = X.shape[0]
        chunks = []
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunks.append(X[start_idx:end_idx].copy())
        
        # Submit chunks for parallel processing
        futures = []
        for chunk in chunks:
            futures.append(self._parallel_executor.submit(self._transform_raw, chunk))
            
        # Collect results
        results = []
        for future in futures:
            results.append(future.result())
            
        # Combine results
        return np.vstack(results)

    def _hash_array(self, arr: np.ndarray) -> str:
        """Generate a hash for a numpy array for caching purposes."""
        # For large arrays, hash a sample to improve performance
        if arr.size > 1000:
            # Sample specific elements and shape for a representative hash
            sample_size = min(1000, arr.size)
            step = max(1, arr.size // sample_size)
            
            # Include shape and dtype in hash
            shape_str = str(arr.shape)
            dtype_str = str(arr.dtype)
            
            # Sample values at regular intervals
            flat_indices = np.arange(0, arr.size, step)[:sample_size]
            flat_arr = arr.ravel()
            sampled_values = flat_arr[flat_indices]
            
            # Create hash from shape, dtype and sampled values
            hash_input = f"{shape_str}_{dtype_str}_{sampled_values.tobytes()}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        else:
            # For small arrays, hash the entire array
            return hashlib.md5(arr.tobytes()).hexdigest()

    def _transform_raw(self, X: np.ndarray, X_hash: Optional[str] = None) -> np.ndarray:
        """
        Apply preprocessing operations without caching.
        
        Args:
            X: Input data
            X_hash: Optional hash for identification (not used in raw transform)
            
        Returns:
            Transformed data
        """
        # Process in chunks if enabled and data is large
        if self.config.chunk_size and X.shape[0] > self.config.chunk_size:
            return self._transform_in_chunks(X)
        
        # Apply each processor in sequence
        for processor in self._processors:
            X = processor(X)
            
        return X

    def _transform_in_chunks(self, X: np.ndarray) -> np.ndarray:
        """Transform large datasets by processing in chunks."""
        chunk_size = self.config.chunk_size
        n_samples = X.shape[0]
        result_chunks = []
        
        # Process each chunk
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = X[start_idx:end_idx].copy()  # Make a copy to avoid modifying original
            
            # Apply processors to the chunk
            for processor in self._processors:
                chunk = processor(chunk)
                
            result_chunks.append(chunk)
            
            if self.config.debug_mode and (start_idx // chunk_size) % 10 == 0:
                self.logger.debug(f"Transformed chunk {start_idx//chunk_size + 1}, "
                                 f"{end_idx}/{n_samples} samples")
        
        # Combine the transformed chunks
        return np.vstack(result_chunks)

    @log_operation
    def fit_transform(self, X: Union[np.ndarray, List], 
                      feature_names: Optional[List[str]] = None,
                      copy: bool = True) -> np.ndarray:
        """
        Combine fit and transform operations efficiently.
        
        Args:
            X: Input data
            feature_names: Optional feature names
            copy: Whether to copy the input data
            
        Returns:
            Transformed data
        """
        self.fit(X, feature_names)
        return self.transform(X, copy=copy)

    def _validate_and_convert_input(self, X: Union[np.ndarray, List], 
                                   operation: str = "transform",
                                   copy: bool = False) -> np.ndarray:
        """
        Validate and convert input data efficiently with comprehensive checks.
        
        Args:
            X: Input data
            operation: Operation being performed ('fit' or 'transform')
            copy: Whether to copy the input data
            
        Returns:
            Validated and converted numpy array
        """
        if self.config.enable_input_validation:
            # Check input type
            if not isinstance(X, (list, np.ndarray, tuple)):
                raise InputValidationError(f"Expected numpy array, list, or tuple, got {type(X)}")
                
            # Convert to numpy array if needed
            if not isinstance(X, np.ndarray):
                try:
                    X = np.array(X, dtype=self.config.dtype)
                except Exception as e:
                    raise InputValidationError(f"Failed to convert input to numpy array: {e}")
            
            # Check for empty input
            if X.size == 0:
                raise InputValidationError("Input is empty")
                
            # Check for NaNs or infs if we don't handle them
            if not self.config.handle_nan and np.any(np.isnan(X)):
                raise InputValidationError("Input contains NaN values but handle_nan=False")
                
            if not self.config.handle_inf and np.any(~np.isfinite(X)):
                raise InputValidationError("Input contains infinite values but handle_inf=False")
                
            # Check dimensions
            if X.ndim not in (1, 2):
                raise InputValidationError(f"Expected 1D or 2D array, got {X.ndim}D")
                
            # Reshape 1D arrays to 2D
            if X.ndim == 1:
                X = X.reshape(-1, 1)
                
            # Check size limits
            if self.config.input_size_limit and X.shape[0] > self.config.input_size_limit:
                raise InputValidationError(
                    f"Input size {X.shape[0]} exceeds limit {self.config.input_size_limit}"
                )
                
            # If transforming, check feature dimensions match fitted dimensions
            if operation == "transform" and self._fitted:
                if X.shape[1] != self._n_features:
                    raise InputValidationError(
                        f"Expected {self._n_features} features, got {X.shape[1]}"
                    )
        
        else:
            # Minimal conversion without validation
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=self.config.dtype)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
                
        # Convert type if needed
        if X.dtype != self.config.dtype:
            X = X.astype(self.config.dtype, copy=copy)
        elif copy:
            X = X.copy()
            
        return X

    def _compute_standard_stats(self, X: np.ndarray) -> None:
        """Compute statistics for standard normalization."""
        self.logger.debug("Computing standard normalization statistics")
        mean = np.mean(X, axis=0, dtype=self.config.dtype)
        std = np.std(X, axis=0, dtype=self.config.dtype)
        std = np.maximum(std, self.config.epsilon)
        scale = 1.0 / std

        # Ensure all statistics are stored
        self._stats['mean'] = mean
        self._stats['std'] = std
        self._stats['scale'] = scale
        
        # Log feature statistics if in debug mode
        if self.config.debug_mode:
            for i, name in enumerate(self._feature_names):
                self.logger.debug(f"Feature {name}: mean={mean[i]:.6f}, std={std[i]:.6f}")


    def _compute_minmax_stats(self, X: np.ndarray) -> None:
        """Compute statistics for min-max normalization."""
        self.logger.debug("Computing min-max normalization statistics")
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        range_val = max_val - min_val
        range_val = np.maximum(range_val, self.config.epsilon)
        scale = 1.0 / range_val

        self._stats['min'] = min_val
        self._stats['max'] = max_val
        self._stats['range'] = range_val
        self._stats['scale'] = scale
        
        # Log feature statistics if in debug mode
        if self.config.debug_mode:
            for i, name in enumerate(self._feature_names):
                self.logger.debug(f"Feature {name}: min={min_val[i]:.6f}, max={max_val[i]:.6f}, range={range_val[i]:.6f}")

    def _compute_robust_stats(self, X: np.ndarray) -> None:
        """Compute robust statistics using percentiles."""
        self.logger.debug("Computing robust normalization statistics")
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
        
        # Log feature statistics if in debug mode
        if self.config.debug_mode:
            for i, name in enumerate(self._feature_names):
                self.logger.debug(f"Feature {name}: lower={lower_percentile[i]:.6f}, "
                                 f"upper={upper_percentile[i]:.6f}, range={range_val[i]:.6f}")

    def _compute_custom_stats(self, X: np.ndarray) -> None:
        """Compute custom statistics using a user-defined method."""
        self.logger.debug("Computing custom normalization statistics")
        
        # Check if custom normalization function is provided
        if hasattr(self.config, 'custom_normalization_fn') and callable(self.config.custom_normalization_fn):
            try:
                # Call the custom function with data and store results
                custom_stats = self.config.custom_normalization_fn(X)
                if isinstance(custom_stats, dict):
                    for key, value in custom_stats.items():
                        self._stats[key] = value
                else:
                    raise ValueError("Custom normalization function must return a dictionary")
            except Exception as e:
                raise StatisticsError(f"Error in custom normalization function: {e}")
        else:
            raise NotImplementedError(
                "Custom normalization is enabled but no custom_normalization_fn is defined in config"
            )

    def _compute_outlier_stats(self, X: np.ndarray) -> None:
        """Compute statistics for outlier detection."""
        outlier_method = self.config.outlier_method.lower()
        self.logger.debug(f"Computing outlier statistics using {outlier_method} method")
        
        self._compute_outlier_stats_with_method(X, outlier_method)

    def _compute_outlier_stats_with_method(self, X: np.ndarray, method: str) -> None:
        """
        Compute outlier statistics using the specified method.
        
        Args:
            X: Input data array
            method: Outlier detection method ('iqr', 'zscore', or 'isolation_forest')
        """
        self.logger.debug(f"Computing outlier statistics using {method} method")
        
        if method == "iqr":
            # IQR-based outlier detection (robust to outliers)
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            threshold = self.config.outlier_params.get("threshold", 1.5)
            
            self._stats['outlier_lower'] = q1 - threshold * iqr
            self._stats['outlier_upper'] = q3 + threshold * iqr
            self._stats['outlier_method'] = "iqr"
            
        elif method == "zscore":
            # Z-score based outlier detection
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            threshold = self.config.outlier_params.get("threshold", 3.0)
            
            self._stats['outlier_mean'] = mean
            self._stats['outlier_std'] = std
            self._stats['outlier_threshold'] = threshold
            self._stats['outlier_method'] = "zscore"
            
        elif method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest
                
                # Get params or use defaults
                n_estimators = self.config.outlier_params.get("n_estimators", 100)
                contamination = self.config.outlier_params.get("contamination", "auto")
                random_state = self.config.outlier_params.get("random_state", 42)
                
                model = IsolationForest(
                    n_estimators=n_estimators, 
                    contamination=contamination,
                    random_state=random_state
                )
                
                # Fit the model
                model.fit(X)
                
                # Store the model
                self._stats['outlier_model'] = model
                self._stats['outlier_method'] = "isolation_forest"
                
            except ImportError:
                self.logger.warning("sklearn not available, falling back to IQR for outlier detection")
                # Fall back to IQR method
                self._compute_outlier_stats_with_method(X, "iqr")
            except Exception as e:
                self.logger.warning(f"Error in isolation forest: {e}, falling back to IQR")
                self._compute_outlier_stats_with_method(X, "iqr")
        else:
            self.logger.warning(f"Unknown outlier method '{method}', falling back to IQR")
            self._compute_outlier_stats_with_method(X, "iqr")

    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization based on configured method."""
        normalization_type = self.config.normalization
        
        if normalization_type == NormalizationType.STANDARD:
            return (X - self._stats['mean']) * self._stats['scale']
            
        elif normalization_type == NormalizationType.MINMAX:
            return (X - self._stats['min']) * self._stats['scale']
            
        elif normalization_type == NormalizationType.ROBUST:
            return (X - self._stats['lower']) * self._stats['scale']
            
        elif normalization_type == NormalizationType.CUSTOM:
            # This would call a custom normalization function
            if hasattr(self.config, 'custom_transform_fn') and callable(self.config.custom_transform_fn):
                try:
                    return self.config.custom_transform_fn(X, self._stats)
                except Exception as e:
                    self.logger.error(f"Error in custom transform function: {e}")
                    self.logger.warning("Returning original data due to custom transform error")
                    return X
            else:
                self.logger.warning("Custom normalization selected but no transform function implemented")
            
        return X

    def _handle_infinite_values(self, X: np.ndarray) -> np.ndarray:
        """Handle infinite values in data with configurable strategies."""
        mask = ~np.isfinite(X)
        if np.any(mask):
            num_inf = np.sum(mask)
            
            # Count inf values per feature for detailed reporting
            if self.config.debug_mode:
                feature_inf_counts = np.sum(mask, axis=0)
                for i, count in enumerate(feature_inf_counts):
                    if count > 0:
                        feature_name = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
                        self.logger.debug(f"Feature {feature_name}: {count} infinite values")
            
            # Replace inf values based on strategy
            inf_strategy = self.config.inf_strategy.lower()
            if inf_strategy == "zero":
                X[mask] = 0.0
                replacement = "0.0"
            elif inf_strategy == "mean" and self._fitted and 'mean' in self._stats:
                # Replace with feature means if available
                for i in range(X.shape[1]):
                    col_mask = mask[:, i]
                    if np.any(col_mask):
                        X[col_mask, i] = self._stats['mean'][i]
                replacement = "feature means"
            elif inf_strategy == "median" and 'median' in self._stats:
                # Replace with feature medians if available
                for i in range(X.shape[1]):
                    col_mask = mask[:, i]
                    if np.any(col_mask):
                        X[col_mask, i] = self._stats['median'][i]
                replacement = "feature medians"
            else:
                # Default to zero
                X[mask] = 0.0
                replacement = "0.0"
                
            self.logger.warning(f"Found {num_inf} infinite values, replaced with {replacement}")
            
        return X

    def _handle_nan_values(self, X: np.ndarray) -> np.ndarray:
        """Handle NaN values in data with advanced strategies."""
        mask = np.isnan(X)
        if np.any(mask):
            num_nan = np.sum(mask)
            
            # Count NaN values per feature for detailed reporting
            if self.config.debug_mode:
                feature_nan_counts = np.sum(mask, axis=0)
                for i, count in enumerate(feature_nan_counts):
                    if count > 0:
                        feature_name = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
                        self.logger.debug(f"Feature {feature_name}: {count} NaN values")
            
            # Replace NaN values based on strategy
            nan_strategy = self.config.nan_strategy.lower()
            if nan_strategy == "mean" and self._fitted and 'mean' in self._stats:
                # Use mean imputation if we have statistics
                for i in range(X.shape[1]):
                    col_mask = mask[:, i]
                    if np.any(col_mask):
                        X[col_mask, i] = self._stats['mean'][i]
                replacement = "feature means"
            elif nan_strategy == "median" and 'median' in self._stats:
                # Use median imputation if we have statistics
                for i in range(X.shape[1]):
                    col_mask = mask[:, i]
                    if np.any(col_mask):
                        X[col_mask, i] = self._stats['median'][i]
                replacement = "feature medians"
            elif nan_strategy == "most_frequent" and 'most_frequent' in self._stats:
                # Use most frequent value imputation if we have statistics
                for i in range(X.shape[1]):
                    col_mask = mask[:, i]
                    if np.any(col_mask):
                        X[col_mask, i] = self._stats['most_frequent'][i]
                replacement = "most frequent values"
            else:
                # Default to zero
                X[mask] = 0.0
                replacement = "0.0"
                
            self.logger.warning(f"Found {num_nan} NaN values, replaced with {replacement}")
            
        return X

    def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
        """Detect and handle outliers based on fitted statistics."""
        if not self.config.detect_outliers:
            return X
            
        outlier_method = self._stats.get('outlier_method')
        if outlier_method is None:
            return X
            
        if outlier_method == "iqr":
            # IQR-based outlier detection
            lower_bound = self._stats['outlier_lower']
            upper_bound = self._stats['outlier_upper']
            
            # Detect outliers
            outliers_mask = (X < lower_bound) | (X > upper_bound)
            
            if np.any(outliers_mask):
                n_outliers = np.sum(outliers_mask)
                
                # Either clip or replace outliers based on config
                if self.config.outlier_params.get("clip", True):
                    # Clip outliers to boundaries
                    X = np.maximum(X, lower_bound)
                    X = np.minimum(X, upper_bound)
                    action = "clipped"
                else:
                    # Replace with boundary values
                    for i in range(X.shape[1]):
                        col_mask_lower = outliers_mask[:, i] & (X[:, i] < lower_bound[i])
                        col_mask_upper = outliers_mask[:, i] & (X[:, i] > upper_bound[i])
                        X[col_mask_lower, i] = lower_bound[i]
                        X[col_mask_upper, i] = upper_bound[i]
                    action = "replaced"
                    
                self.logger.info(f"Found {n_outliers} outliers, {action} to boundary values")
                
        elif outlier_method == "zscore":
            # Z-score based outlier detection
            mean = self._stats['outlier_mean']
            std = self._stats['outlier_std']
            threshold = self._stats['outlier_threshold']
            
            # Calculate z-scores
            z_scores = np.abs((X - mean) / np.maximum(std, self.config.epsilon))
            outliers_mask = z_scores > threshold
            
            if np.any(outliers_mask):
                n_outliers = np.sum(outliers_mask)
                
                # Handle outliers
                if self.config.outlier_params.get("clip", True):
                    # Clip outliers to threshold * std
                    upper_bound = mean + threshold * std
                    lower_bound = mean - threshold * std
                    X = np.maximum(X, lower_bound)
                    X = np.minimum(X, upper_bound)
                    action = "clipped"
                else:
                    # Replace with mean
                    for i in range(X.shape[1]):
                        col_mask = outliers_mask[:, i]
                        if np.any(col_mask):
                            X[col_mask, i] = mean[i]
                    action = "replaced with mean"
                    
                self.logger.info(f"Found {n_outliers} z-score outliers, {action}")
                
        elif outlier_method == "isolation_forest":
            # Use Isolation Forest for outlier detection
            if 'outlier_model' in self._stats:
                try:
                    model = self._stats['outlier_model']
                    
                    # Predict outliers (-1 for outliers, 1 for inliers)
                    outlier_predictions = model.predict(X)
                    outliers_mask = outlier_predictions == -1
                    
                    if np.any(outliers_mask):
                        n_outliers = np.sum(outliers_mask)
                        
                        # Handle outliers
                        if 'mean' in self._stats:
                            # Replace with feature means if available
                            for idx in np.where(outliers_mask)[0]:
                                X[idx] = self._stats['mean']
                            action = "replaced with feature means"
                        else:
                            # Calculate feature means on the fly for this batch
                            inliers = X[~outliers_mask]
                            if inliers.size > 0:
                                means = np.mean(inliers, axis=0)
                                
                                # Replace outliers with means
                                for idx in np.where(outliers_mask)[0]:
                                    X[idx] = means
                                action = "replaced with batch means"
                            else:
                                # All samples are outliers, use global statistics if available
                                if 'outlier_mean' in self._stats:
                                    for idx in np.where(outliers_mask)[0]:
                                        X[idx] = self._stats['outlier_mean']
                                    action = "replaced with global means"
                                else:
                                    # No good replacement available
                                    action = "not replaced (all samples are outliers)"
                            
                        self.logger.info(f"Found {n_outliers} isolation forest outliers, {action}")
                except Exception as e:
                    self.logger.error(f"Error applying isolation forest: {e}")
                
        return X

    def _clip_data(self, X: np.ndarray) -> np.ndarray:
        """Clip data to specified range."""
        min_clip, max_clip = self.config.clip_range
        np.clip(X, min_clip, max_clip, out=X)  # In-place clipping for efficiency
        return X

    def _update_metrics(self, operation: str, elapsed_time: float) -> None:
        """Update performance metrics for monitoring."""
        with self._lock:  # Protect metrics updates with lock
            if operation == 'fit':
                self._metrics['fit_time'].append(elapsed_time)
            elif operation == 'transform':
                self._metrics['transform_time'].append(elapsed_time)
            elif operation == '_transform_raw':
                self._metrics['transform_time'].append(elapsed_time)
            elif 'chunk' in operation:
                self._metrics['process_chunk_time'].append(elapsed_time)

    @log_operation
    def get_stats(self) -> Dict[str, Any]:
        """Return computed statistics with safeguards."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted yet")
            
        # Return copies of arrays to prevent modification
        result = {}
        for k, v in self._stats.items():
            if isinstance(v, np.ndarray):
                result[k] = v.copy()
            elif hasattr(v, 'copy'):
                result[k] = v.copy()
            elif k == 'outlier_model':
                # Skip complex objects like sklearn models
                result[k] = "Model object (not JSON serializable)"
            else:
                result[k] = v
                
        return result

    def get_feature_names(self) -> List[str]:
        """Return the feature names if available."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted yet")
        return self._feature_names.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for monitoring."""
        with self._lock:  # Protect metrics access with lock
            metrics = {}
            
            for metric_name, values in self._metrics.items():
                if values:
                    metrics[f"avg_{metric_name}"] = np.mean(values)
                    metrics[f"max_{metric_name}"] = np.max(values)
                    metrics[f"min_{metric_name}"] = np.min(values)
                    if len(values) > 1:
                        metrics[f"std_{metric_name}"] = np.std(values)
            
            metrics["n_samples_seen"] = self._n_samples_seen
            metrics["n_features"] = self._n_features
            metrics["fitted"] = self._fitted
            metrics["version"] = self._version
            
            return metrics

    def reset(self) -> None:
        """Reset the preprocessor state."""
        with self._lock:
            self._stats.clear()
            self._fitted = False
            self._n_samples_seen = 0
            
            # Clear caches if enabled
            if self.config.cache_enabled and hasattr(self._cached_transform, 'cache_clear'):
                self._cached_transform.cache_clear()
                
            self.logger.info("Preprocessor state reset")

    @log_operation
    def save(self, filepath: str) -> None:
        """
        Save the preprocessor to a file with version information.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
                
        with self._lock, self._error_context("serialization operation"):  # Ensure thread safety during save
            # Create directory if it doesn't exist
            directory = os.path.dirname(os.path.abspath(filepath))
            os.makedirs(directory, exist_ok=True)
            
            # Use a temporary file for atomic writing
            temp_filepath = f"{filepath}.tmp"
            model_files = []
            
            try:
                # Prepare data for serialization
                data = {
                    "version": self._version,
                    "config": self.config.to_dict(),  # Now properly implemented
                    "feature_names": self._feature_names,
                    "n_features": self._n_features,
                    "n_samples_seen": self._n_samples_seen,
                    "fitted": self._fitted,
                    "stats": {}
                }
                
                # Get base filepath for models without extension
                base_filepath = os.path.splitext(filepath)[0]
                
                # Handle stats with special cases for non-serializable objects
                for k, v in self._stats.items():
                    if isinstance(v, np.ndarray):
                        data["stats"][k] = {
                            "type": "ndarray",
                            "dtype": str(v.dtype),
                            "shape": v.shape,
                            "data": v.tolist()
                        }
                    elif k == "outlier_model" or (hasattr(v, '__module__') and 'sklearn' in getattr(v, '__module__', '')):
                        # For sklearn models or other complex objects, we need to use pickle
                        model_path = f"{base_filepath}_model_{k}.pkl"
                        model_files.append(model_path)
                        
                        try:
                            with open(model_path, 'wb') as f:
                                pickle.dump(v, f)
                            data["stats"][k] = {"type": "model", "path": model_path, "serialized": True}
                            self.logger.info(f"Model {k} saved to {model_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to pickle model {k}: {e}")
                            data["stats"][k] = {"type": "model", "serialized": False, "error": str(e)}
                    elif isinstance(v, (np.integer, np.floating, np.bool_)):
                        # Convert numpy scalars to Python scalars
                        data["stats"][k] = v.item()
                    elif isinstance(v, type):
                        # Handle Python types
                        data["stats"][k] = str(v)
                    else:
                        # Test JSON serialization first to avoid errors during final save
                        try:
                            json.dumps({k: v})
                            data["stats"][k] = v
                        except (TypeError, OverflowError):
                            # If not JSON serializable, convert to string representation
                            data["stats"][k] = str(v)
                            
                # Save the main data as JSON to temporary file first
                with open(temp_filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                # If we got here, atomically rename temp file to target file
                os.replace(temp_filepath, filepath)
                self.logger.info(f"Preprocessor saved to {filepath}")
                
            except Exception as e:
                # Clean up temporary files on error
                self.logger.error(f"Failed to save preprocessor: {e}")
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                for model_file in model_files:
                    if os.path.exists(model_file) and os.path.basename(filepath) not in model_file:
                        os.remove(model_file)
                
                # Re-raise with appropriate error type
                if isinstance(e, SerializationError):
                    raise e
                else:
                    raise SerializationError(f"Failed to save preprocessor: {e}")


    @classmethod
    @log_operation
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """
        Load a preprocessor from a file.
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor instance
        """
        logger = logging.getLogger(f"{__name__}.DataPreprocessor")
        logger.info(f"Loading preprocessor from {filepath}")
        
        # Create a error context for class method
        @contextmanager
        def error_context(operation):
            try:
                yield
            except Exception as e:
                error_msg = f"Error during {operation}: {str(e)}"
                logger.error(error_msg)
                # Convert to appropriate exception type
                if "validation" in operation.lower():
                    raise InputValidationError(error_msg) from e
                elif "serialization" in operation.lower():
                    raise SerializationError(error_msg) from e
                else:
                    raise PreprocessingError(error_msg) from e
        
        with error_context("serialization operation"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load preprocessor file: {e}")
                raise SerializationError(f"Failed to load preprocessor: {e}")
                
            # Extract version and verify compatibility
            version = data.get("version", "unknown")
            if version != data.get("config", {}).get("version", "unknown"):
                logger.warning(f"Version mismatch: file version {version}, "
                            f"config version {data.get('config', {}).get('version', 'unknown')}")
                
            # Create config from saved data
            config_dict = data.get("config", {})
            
            # Convert string enum values to actual enum values
            if "normalization" in config_dict and isinstance(config_dict["normalization"], str):
                try:
                    config_dict["normalization"] = NormalizationType[config_dict["normalization"]]
                except KeyError:
                    logger.warning(f"Unknown normalization type: {config_dict['normalization']}, "
                                f"defaulting to STANDARD")
                    config_dict["normalization"] = NormalizationType.STANDARD
                    
            # Create config object
            config = PreprocessorConfig(**config_dict)
            
            # Create preprocessor
            preprocessor = cls(config)
            
            # Restore stats with special handling for numpy arrays
            stats = {}
            for k, v in data.get("stats", {}).items():
                if isinstance(v, dict) and v.get("type") == "ndarray":
                    # Reconstruct numpy array
                    try:
                        arr = np.array(v["data"], dtype=np.dtype(v["dtype"]))
                        stats[k] = arr
                    except Exception as e:
                        logger.error(f"Failed to reconstruct array {k}: {e}")
                        # Use a default empty array as fallback
                        stats[k] = np.array([], dtype=np.float64)
                elif isinstance(v, dict) and v.get("type") == "model" and v.get("serialized", False):
                    # Load pickled model
                    model_path = v.get("path")
                    if model_path and os.path.exists(model_path):
                        try:
                            with open(model_path, 'rb') as f:
                                stats[k] = pickle.load(f)
                            logger.info(f"Model {k} loaded from {model_path}")
                        except Exception as e:
                            logger.error(f"Failed to load model {k}: {e}")
                    else:
                        logger.warning(f"Model file {model_path} not found")
                else:
                    stats[k] = v
                    
            # Restore instance variables
            preprocessor._stats = stats
            preprocessor._feature_names = data.get("feature_names", [])
            preprocessor._n_features = data.get("n_features", 0)
            preprocessor._n_samples_seen = data.get("n_samples_seen", 0)
            preprocessor._fitted = data.get("fitted", False)
            preprocessor._version = version
            
            logger.info(f"Preprocessor loaded: version={version}, fitted={preprocessor._fitted}, "
                    f"features={preprocessor._n_features}")
            
            return preprocessor

    def inverse_transform(self, X: Union[np.ndarray, List], copy: bool = True) -> np.ndarray:
        """
        Reverse the transformation process to get original scale data.
        
        Args:
            X: Transformed data
            copy: Whether to copy the input data
            
        Returns:
            Data in original scale
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
            
        with self._error_context("inverse transform operation"):
            # Validate and convert input
            X = self._validate_and_convert_input(X, operation="transform", copy=copy)
            
            # Apply inverse normalization
            norm_type = self.config.normalization
            
            if norm_type == NormalizationType.STANDARD:
                if 'mean' in self._stats and 'scale' in self._stats:
                    # Fixed implementation for standard normalization
                    return (X / self._stats['scale']) + self._stats['mean']
                    
            elif norm_type == NormalizationType.MINMAX:
                if 'min' in self._stats and 'scale' in self._stats:
                    # Fixed implementation for min-max scaling
                    return (X / self._stats['scale']) + self._stats['min']
                    
            elif norm_type == NormalizationType.ROBUST:
                if 'lower' in self._stats and 'scale' in self._stats:
                    # Fixed implementation for robust scaling
                    return (X / self._stats['scale']) + self._stats['lower']
                    
            elif norm_type == NormalizationType.CUSTOM:
                # Handle custom inverse transform
                if hasattr(self.config, 'custom_inverse_transform_fn') and callable(self.config.custom_inverse_transform_fn):
                    try:
                        return self.config.custom_inverse_transform_fn(X, self._stats)
                    except Exception as e:
                        self.logger.error(f"Error in custom inverse transform: {e}")
                        raise
                else:
                    raise NotImplementedError("Custom inverse transform not implemented")
                    
            # If we reach here with STANDARD normalization, use the stats we have
            if norm_type == NormalizationType.STANDARD and 'mean' in self._stats and 'std' in self._stats:
                return X * self._stats['std'] + self._stats['mean']
                
            # If we still get here, we couldn't invert the transform
            raise NotImplementedError(f"Inverse transform not implemented for {norm_type}")



    def partial_fit(self, X: Union[np.ndarray, List], feature_names: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Update preprocessing statistics with a new batch of data.
        
        Args:
            X: New data batch
            feature_names: Optional feature names (must match previous names if already fitted)
            
        Returns:
            self for method chaining
        """
        with self._lock, self._error_context("partial fit operation"):
            # Validate and convert input
            X = self._validate_and_convert_input(X, operation="fit")
            
            if not self._fitted:
                # First batch - do a regular fit
                return self.fit(X, feature_names)
            else:
                # Validate that feature dimensions match
                if X.shape[1] != self._n_features:
                    raise InputValidationError(
                        f"Input has {X.shape[1]} features, but preprocessor was fitted with {self._n_features} features"
                    )
                
                # Validate feature names if provided
                if feature_names:
                    if feature_names != self._feature_names:
                        raise InputValidationError("Feature names don't match previously fitted names")
                
                # Update statistics incrementally
                self._update_stats_incrementally(X)
                
                # Add to count of samples seen
                self._n_samples_seen += X.shape[0]
                
                self.logger.info(f"Updated preprocessor with {X.shape[0]} additional samples, total seen: {self._n_samples_seen}")
                return self

    def _update_stats_incrementally(self, X: np.ndarray) -> None:
        """Update statistics incrementally with new data."""
        norm_type = self.config.normalization
        
        if norm_type == NormalizationType.STANDARD:
            # For standard normalization, we update mean and variance incrementally
            # using Welford's online algorithm
            if 'mean' in self._stats and 'std' in self._stats:
                old_n = self._n_samples_seen
                new_n = X.shape[0]
                total_n = old_n + new_n
                
                # Get current mean and update with new data
                old_mean = self._stats['mean']
                new_mean = np.mean(X, axis=0)
                
                # Calculate batch variance for new data
                new_var = np.var(X, axis=0)
                
                # Compute combined mean
                combined_mean = (old_n * old_mean + new_n * new_mean) / total_n
                
                # Compute combined variance using parallel variance formula
                old_var = self._stats['std'] ** 2
                combined_var = (old_n * old_var + new_n * new_var + 
                                (old_n * new_n) / total_n * (old_mean - new_mean) ** 2) / total_n
                
                # Update statistics
                self._stats['mean'] = combined_mean
                self._stats['std'] = np.sqrt(np.maximum(combined_var, self.config.epsilon))
                self._stats['scale'] = 1.0 / self._stats['std']
                
                self.logger.debug("Updated standard normalization statistics incrementally")
                
        elif norm_type == NormalizationType.MINMAX:
            # For min-max normalization, we update min and max
            if 'min' in self._stats and 'max' in self._stats:
                # Update min and max with new data
                new_min = np.min(X, axis=0)
                new_max = np.max(X, axis=0)
                
                # Update global min and max
                self._stats['min'] = np.minimum(self._stats['min'], new_min)
                self._stats['max'] = np.maximum(self._stats['max'], new_max)
                
                # Recalculate range and scale
                range_val = self._stats['max'] - self._stats['min']
                range_val = np.maximum(range_val, self.config.epsilon)
                self._stats['range'] = range_val
                self._stats['scale'] = 1.0 / range_val
                
                self.logger.debug("Updated min-max normalization statistics incrementally")
                
        elif norm_type == NormalizationType.ROBUST:
            # For robust normalization, we need to recalculate percentiles
            # This is more challenging to do incrementally, so we'll use a simplified approach
            # that approximates the results
            if 'lower' in self._stats and 'upper' in self._stats:
                # Get current bounds
                old_lower = self._stats['lower']
                old_upper = self._stats['upper']
                
                # Calculate percentiles for new data
                lower, upper = self.config.robust_percentiles
                new_lower = np.percentile(X, lower, axis=0)
                new_upper = np.percentile(X, upper, axis=0)
                
                # Use a weighted average based on sample counts (approximation)
                weight_old = self._n_samples_seen / (self._n_samples_seen + X.shape[0])
                weight_new = X.shape[0] / (self._n_samples_seen + X.shape[0])
                
                # Update bounds (this is an approximation, not exact percentile calculation)
                combined_lower = weight_old * old_lower + weight_new * new_lower
                combined_upper = weight_old * old_upper + weight_new * new_upper
                
                # Ensure bounds are reasonable by taking min/max as safety
                combined_lower = np.minimum(combined_lower, np.minimum(old_lower, new_lower))
                combined_upper = np.maximum(combined_upper, np.maximum(old_upper, new_upper))
                
                # Update statistics
                self._stats['lower'] = combined_lower
                self._stats['upper'] = combined_upper
                range_val = combined_upper - combined_lower
                range_val = np.maximum(range_val, self.config.epsilon)
                self._stats['range'] = range_val
                self._stats['scale'] = 1.0 / range_val
                
                self.logger.debug("Updated robust normalization statistics incrementally (approximate)")
                
        elif norm_type == NormalizationType.CUSTOM:
            # For custom normalization, we need to call the user-defined function
            if hasattr(self.config, 'custom_incremental_update_fn') and callable(self.config.custom_incremental_update_fn):
                try:
                    # Call custom function with current stats, new data, and sample counts
                    self._stats = self.config.custom_incremental_update_fn(
                        self._stats, X, self._n_samples_seen, X.shape[0]
                    )
                    self.logger.debug("Updated custom normalization statistics incrementally")
                except Exception as e:
                    self.logger.error(f"Error in custom incremental update: {e}")
                    raise
            else:
                self.logger.warning("No custom_incremental_update_fn defined, statistics not updated")
        
        # Update outlier statistics if needed
        if self.config.detect_outliers:
            self._update_outlier_stats(X)
            
        # Update median if used for imputation
        if self.config.nan_strategy == "median" or self.config.inf_strategy == "median":
            if 'median' in self._stats:
                # Approximate update of median (not mathematically exact)
                old_weight = self._n_samples_seen / (self._n_samples_seen + X.shape[0])
                new_weight = X.shape[0] / (self._n_samples_seen + X.shape[0])
                new_median = np.median(X, axis=0)
                self._stats['median'] = old_weight * self._stats['median'] + new_weight * new_median
            else:
                # Calculate median if not present
                self._stats['median'] = np.median(X, axis=0)
                
        # Update most frequent values if needed
        if self.config.nan_strategy == "most_frequent" and 'most_frequent' in self._stats:
            # This is just an approximation, as keeping track of frequencies across batches
            # would require additional memory
            new_most_frequent = np.zeros(X.shape[1], dtype=self.config.dtype)
            for i in range(X.shape[1]):
                unique_vals, counts = np.unique(X[:, i], return_counts=True)
                if len(unique_vals) > 0:
                    new_most_frequent[i] = unique_vals[np.argmax(counts)]
                else:
                    new_most_frequent[i] = 0.0
            
            # Keep old values if they have higher frequency (approximation)
            # In a real implementation, we would track actual frequencies
            self._stats['most_frequent'] = self._stats['most_frequent']

    def _update_outlier_stats(self, X: np.ndarray) -> None:
        """Update outlier detection statistics with new data."""
        outlier_method = self._stats.get('outlier_method')
        if outlier_method is None:
            return
            
        if outlier_method == "iqr":
            # For IQR, we update quantiles (this is an approximation)
            if 'outlier_lower' in self._stats and 'outlier_upper' in self._stats:
                # Calculate IQR for new data
                q1 = np.percentile(X, 25, axis=0)
                q3 = np.percentile(X, 75, axis=0)
                iqr = q3 - q1
                threshold = self.config.outlier_params.get("threshold", 1.5)
                
                new_lower = q1 - threshold * iqr
                new_upper = q3 + threshold * iqr
                
                # Combine with existing bounds (approximation)
                weight_old = self._n_samples_seen / (self._n_samples_seen + X.shape[0])
                weight_new = X.shape[0] / (self._n_samples_seen + X.shape[0])
                
                combined_lower = weight_old * self._stats['outlier_lower'] + weight_new * new_lower
                combined_upper = weight_old * self._stats['outlier_upper'] + weight_new * new_upper
                
                # Update statistics
                self._stats['outlier_lower'] = combined_lower
                self._stats['outlier_upper'] = combined_upper
                
        elif outlier_method == "zscore":
            # For z-score, we update mean and std
            if 'outlier_mean' in self._stats and 'outlier_std' in self._stats:
                # Update mean and std incrementally (same as standard normalization)
                old_n = self._n_samples_seen
                new_n = X.shape[0]
                total_n = old_n + new_n
                
                old_mean = self._stats['outlier_mean']
                new_mean = np.mean(X, axis=0)
                
                new_var = np.var(X, axis=0)
                
                combined_mean = (old_n * old_mean + new_n * new_mean) / total_n
                
                old_var = self._stats['outlier_std'] ** 2
                combined_var = (old_n * old_var + new_n * new_var + 
                                (old_n * new_n) / total_n * (old_mean - new_mean) ** 2) / total_n
                
                self._stats['outlier_mean'] = combined_mean
                self._stats['outlier_std'] = np.sqrt(np.maximum(combined_var, self.config.epsilon))
                
        elif outlier_method == "isolation_forest":
            # For Isolation Forest, we could retrain the model with a sample of old and new data
            # This is more complex and would require keeping a sample of previous data
            # For simplicity, we'll skip this and rely on periodic full refitting
            self.logger.warning("Partial fit with Isolation Forest not fully supported - "
                              "model not updated incrementally")

    def __repr__(self) -> str:
        """Return string representation of the preprocessor."""
        status = "fitted" if self._fitted else "not fitted"
        features_info = f"{self._n_features} features" if self._fitted else "unknown features"
        samples_info = f"{self._n_samples_seen} samples seen" if self._fitted else ""
        return (f"DataPreprocessor(normalization={self.config.normalization}, "
                f"{features_info}, {samples_info}, {status}, version={self._version})")

    def __del__(self) -> None:
        """Cleanup resources when the object is deleted."""
        # Clean up parallel executor if exists
        if hasattr(self, '_parallel_executor') and self._parallel_executor:
            try:
                self._parallel_executor.shutdown(wait=False)
            except Exception:
                # Ignore errors during cleanup
                pass

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Estimate the memory usage of the preprocessor.
        
        Returns:
            Dictionary with memory usage information
        """
        import sys
        
        memory_info = {}
        
        # Estimate stats memory usage
        stats_size = 0
        for k, v in self._stats.items():
            if isinstance(v, np.ndarray):
                stats_size += v.nbytes
            elif hasattr(v, '__sizeof__'):
                stats_size += sys.getsizeof(v)
            else:
                stats_size += sys.getsizeof(str(v))
        
        memory_info['stats_bytes'] = stats_size
        memory_info['stats_mb'] = stats_size / (1024 * 1024)
        
        # Estimate cache size if enabled
        if self.config.cache_enabled and hasattr(self._cached_transform, 'cache_info'):
            cache_info = self._cached_transform.cache_info()
            memory_info['cache_hits'] = cache_info.hits
            memory_info['cache_misses'] = cache_info.misses
            memory_info['cache_size'] = cache_info.currsize
            memory_info['cache_maxsize'] = cache_info.maxsize
        
        # Total object size (approximate)
        memory_info['total_bytes'] = sys.getsizeof(self)
        memory_info['total_mb'] = sys.getsizeof(self) / (1024 * 1024)
        
        return memory_info

    def copy(self) -> 'DataPreprocessor':
        """
        Create a copy of this preprocessor.
        
        Returns:
            A new preprocessor with the same state
        """
        import io
        import tempfile
        import pickle
        
        if not self._fitted:
            # If not fitted, just create a new instance with the same config
            return DataPreprocessor(self.config)
        
        # For fitted preprocessors, use pickle instead of JSON serialization
        try:
            with tempfile.NamedTemporaryFile(suffix='.pkl') as temp_file:
                # Use pickle to serialize the object
                with open(temp_file.name, 'wb') as f:
                    pickle.dump(self, f)
                
                # Load from the pickle file
                with open(temp_file.name, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error copying preprocessor: {e}")
            # Fallback: create a new instance and manually copy attributes
            new_preprocessor = DataPreprocessor(self.config)
            new_preprocessor._stats = {k: (v.copy() if hasattr(v, 'copy') else v) 
                                    for k, v in self._stats.items()}
            new_preprocessor._feature_names = self._feature_names.copy()
            new_preprocessor._fitted = self._fitted
            new_preprocessor._n_features = self._n_features
            new_preprocessor._n_samples_seen = self._n_samples_seen
            new_preprocessor._version = self._version
            return new_preprocessor

