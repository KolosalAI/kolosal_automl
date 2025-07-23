import os
import logging
import threading
import numpy as np
import hashlib
import json
import pickle
import traceback
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from functools import wraps
from contextlib import contextmanager
from dataclasses import asdict
import time

from ..configs import PreprocessorConfig, NormalizationType
from .preprocessing_exceptions import (
    PreprocessingError, InputValidationError, 
    StatisticsError, SerializationError
)


def log_operation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            # Optionally log the success
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            # Optionally log the error details here.
            # If it's one of our custom exceptions, re-raise it unchanged.
            if isinstance(e, (InputValidationError, StatisticsError, SerializationError, PreprocessingError)):
                raise e
            else:
                raise Exception(f"{func.__name__} failed after {elapsed_time:.4f} seconds: {e}") from e
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
        
        # Initialize caching based on config
        if self.config.cache_enabled:
            self._transform_cache = {}
        else:
            self._transform_cache = None

        # Set the processor version
        self._version = self.config.version
        
        # Initialize parallel processing if enabled
        self._parallel_executor = None
        if self.config.parallel_processing:
            self._init_parallel_executor()
        
        # Initialize running stats
        self._running_stats = None
        
        # Get feature outlier handling mask
        self.feature_outlier_mask = getattr(self.config, 'feature_outlier_mask', None)
        
        # Initialize nan and inf handling configurations
        self.nan_fill_value = getattr(self.config, 'nan_fill_value', 0.0)
        self.pos_inf_fill_value = getattr(self.config, 'pos_inf_fill_value', float('nan'))
        self.neg_inf_fill_value = getattr(self.config, 'neg_inf_fill_value', float('nan'))
        self.inf_buffer = getattr(self.config, 'inf_buffer', 1.0)
        
        # Init clipping values
        self.clip_min = getattr(self.config, 'clip_min', float('-inf'))
        self.clip_max = getattr(self.config, 'clip_max', float('inf'))
        
        # Handle scale shift
        self.scale_shift = getattr(self.config, 'scale_shift', False)
        self.scale_offset = getattr(self.config, 'scale_offset', 0.0)
        
        # Initialize custom functions
        self.custom_center_func = getattr(self.config, 'custom_center_func', None)
        self.custom_scale_func = getattr(self.config, 'custom_scale_func', None)
        
        # Initialize outlier params
        self.outlier_iqr_multiplier = getattr(self.config, 'outlier_iqr_multiplier', 1.5)
        self.outlier_zscore_threshold = getattr(self.config, 'outlier_zscore_threshold', 3.0)
        self.outlier_percentiles = getattr(self.config, 'outlier_percentiles', (1.0, 99.0))
        
        self.logger.info(f"DataPreprocessor initialized with version {self._version}")

    def _configure_logging(self) -> None:
        """Configure logging based on config settings."""
        level = logging.DEBUG if self.config.debug_mode else logging.INFO
        
        try:
            from modules.logging_config import get_logger
            self.logger = get_logger(
                name="DataPreprocessor",
                level=level,
                log_file="data_preprocessor.log",
                enable_console=True
            )
        except ImportError:
            # Fallback to basic logging if centralized logging not available
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

        # Add data quality processors FIRST (this is critical)
        if self.config.handle_nan:
            self._processors.append(self._handle_nan_values)
        if self.config.handle_inf:
            self._processors.append(self._handle_infinite_values)

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
        try:
            yield
        except (InputValidationError, StatisticsError, SerializationError) as e:
            raise e
        except Exception as e:
            error_msg = f"Error during {operation}: {str(e)}"
            self.logger.error(error_msg)
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            if "validation" in operation.lower():
                raise InputValidationError(error_msg) from e
            elif "statistics" in operation.lower():
                raise StatisticsError(error_msg) from e
            elif "serialization" in operation.lower():
                raise SerializationError(error_msg) from e
            else:
                raise PreprocessingError(error_msg) from e

    @log_operation
    def fit(self, X, y=None, feature_names=None, **fit_params):
        """
        Fit the preprocessor to the data
        
        Args:
            X: Input data to fit
            y: Target values (unused, for API compatibility)
            feature_names: Optional list of feature names
            **fit_params: Additional parameters
            
        Returns:
            self: The fitted preprocessor
        """
        with self._lock, self._error_context("fit operation"):
            start_time = time.time()
            
            # Store the input shape
            self.input_shape_ = X.shape
            
            # Extract feature names if available
            if feature_names is None and hasattr(X, 'columns') and hasattr(X.columns, 'tolist'):
                feature_names = X.columns.tolist()
            
            # Validate and convert input
            X = self._validate_and_convert_input(X, operation="fit")
            
            # Store feature count and names
            self._n_features = X.shape[1] if X.ndim > 1 else 1
            
            # Process feature names
            if feature_names is not None and len(feature_names) > 0:
                if len(feature_names) != self._n_features:
                    raise InputValidationError(
                        f"Feature names length ({len(feature_names)}) doesn't match data dimensions ({self._n_features})"
                    )
                self._feature_names = feature_names
            else:
                # Create default feature names
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
            
            # Ensure standard stats are computed for standard normalization if not already computed
            norm_type = self._get_norm_type(self.config.normalization)
            if norm_type == NormalizationType.STANDARD and 'mean' not in self._stats:
                self._compute_standard_stats(X)
            
            # Track fit time for metrics
            fit_time = time.time() - start_time
            self._metrics['fit_time'].append(fit_time)
            
            self.logger.info(f"Fitted preprocessor on {self._n_samples_seen} samples, {self._n_features} features in {fit_time:.4f}s")
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
        norm_type = self._get_norm_type(self.config.normalization)
        
        if norm_type == NormalizationType.STANDARD:
            self._compute_standard_stats(X)
        elif norm_type == NormalizationType.MINMAX:
            self._compute_minmax_stats(X)
        elif norm_type == NormalizationType.ROBUST:
            self._compute_robust_stats(X)
        else:  # Handle custom or other normalization types
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
        norm_type = self._get_norm_type(self.config.normalization)
        
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
        norm_type = self._get_norm_type(self.config.normalization)
        
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
        
        norm_type = self._get_norm_type(self.config.normalization)
        
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

        start_time = time.time()
        
        with self._error_context("transform operation"):
            # Validate and convert input
            X = self._validate_and_convert_input(X, operation="transform", copy=copy)

            # Use parallel processing if enabled and data is large enough
            if (
                self.config.parallel_processing 
                and self._parallel_executor 
                and X.shape[0] > 1000 
                and self.config.chunk_size
            ):
                result = self._transform_parallel(X)
            else:
                # Manual caching for smaller arrays
                if self.config.cache_enabled and X.shape[0] <= 1000:
                    # Use a hash of the array bytes as the dictionary key
                    X_hash = self._hash_array(X)
                    if X_hash in self._transform_cache:
                        # Cache hit
                        result = self._transform_cache[X_hash]
                    else:
                        # Cache miss
                        result = self._transform_raw(X)
                        self._transform_cache[X_hash] = result
                else:
                    # Large arrays or caching disabled
                    result = self._transform_raw(X)
            
            # Track transform time for metrics
            transform_time = time.time() - start_time
            self._metrics['transform_time'].append(transform_time)
            
            if self.config.debug_mode:
                self.logger.debug(f"Transformed {X.shape[0]} samples in {transform_time:.4f}s")
                
            return result

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

    def _transform_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing operations without caching.
        
        Args:
            X: Input data
            
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
            start_chunk_time = time.time()
            
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = X[start_idx:end_idx].copy()  # Make a copy to avoid modifying original
            
            # Apply processors to the chunk
            for processor in self._processors:
                chunk = processor(chunk)
                
            result_chunks.append(chunk)
            
            chunk_time = time.time() - start_chunk_time
            self._metrics['process_chunk_time'].append(chunk_time)
            
            if self.config.debug_mode and (start_idx // chunk_size) % 10 == 0:
                self.logger.debug(f"Transformed chunk {start_idx//chunk_size + 1}, "
                                 f"{end_idx}/{n_samples} samples in {chunk_time:.4f}s")
        
        # Combine the transformed chunks
        return np.vstack(result_chunks)

    @log_operation
    def fit_transform(self, X: Union[np.ndarray, List], y=None, feature_names=None, copy: bool = True) -> np.ndarray:
        """
        Combine fit and transform operations efficiently.
        
        Args:
            X: Input data
            y: Target values (unused, for API compatibility)
            feature_names: Optional feature names
            copy: Whether to copy the input data
            
        Returns:
            Transformed data
        """
        self.fit(X, feature_names=feature_names)
        return self.transform(X, copy=copy)

    def _validate_and_convert_input(self, 
                                   X: Union[np.ndarray, List], 
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
            if not isinstance(X, (list, tuple, np.ndarray)):
                raise InputValidationError(
                    f"Expected numpy array, list, or tuple, got {type(X)}"
                )
            
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
        """Compute robust statistics for normalization."""
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
                self.logger.debug(f"Feature {name}: lower={lower_percentile[i]:.6f}, upper={upper_percentile[i]:.6f}, range={range_val[i]:.6f}")
    
    def _compute_custom_stats(self, X: np.ndarray) -> None:
        """Compute custom statistics for normalization."""
        self.logger.debug("Computing custom normalization statistics")
        
        # Check if custom functions are provided
        if self.custom_center_func is not None and self.custom_scale_func is not None:
            try:
                center = self.custom_center_func(X)
                scale = self.custom_scale_func(X)
                
                self._stats['center'] = center
                self._stats['scale'] = scale
                
                # Log feature statistics if in debug mode
                if self.config.debug_mode:
                    for i, name in enumerate(self._feature_names):
                        self.logger.debug(f"Feature {name}: center={center[i]:.6f}, scale={scale[i]:.6f}")
            except Exception as e:
                self.logger.error(f"Error computing custom statistics: {e}")
                # Fall back to standard stats as a safer alternative
                self._compute_standard_stats(X)
        else:
            # Default to standard stats if no custom functions
            self._compute_standard_stats(X)
    
    def _compute_outlier_stats(self, X: np.ndarray) -> None:
        """Compute statistics for outlier detection."""
        self.logger.debug("Computing outlier detection statistics")
        
        outlier_method = self.config.outlier_method.lower()
        
        if outlier_method == "iqr":
            # Compute IQR-based bounds
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            
            lower_bound = q1 - (self.outlier_iqr_multiplier * iqr)
            upper_bound = q3 + (self.outlier_iqr_multiplier * iqr)
            
        elif outlier_method == "zscore":
            # Compute Z-score based bounds
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std = np.maximum(std, self.config.epsilon)
            
            lower_bound = mean - (self.outlier_zscore_threshold * std)
            upper_bound = mean + (self.outlier_zscore_threshold * std)
            
        elif outlier_method == "percentile":
            # Compute percentile-based bounds
            lower, upper = self.outlier_percentiles
            lower_bound = np.percentile(X, lower, axis=0)
            upper_bound = np.percentile(X, upper, axis=0)
            
        else:
            # Default to Z-score method
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std = np.maximum(std, self.config.epsilon)
            
            lower_bound = mean - (3.0 * std)  # Default to 3 sigma
            upper_bound = mean + (3.0 * std)
        
        # Store outlier bounds
        self._stats['outlier_lower'] = lower_bound
        self._stats['outlier_upper'] = upper_bound
        
        # Log feature statistics if in debug mode
        if self.config.debug_mode:
            for i, name in enumerate(self._feature_names):
                self.logger.debug(f"Feature {name}: outlier bounds [{lower_bound[i]:.6f}, {upper_bound[i]:.6f}]")
                
    def _clip_data(self, X: np.ndarray) -> np.ndarray:
        """Clip data to specified min/max values."""
        if not self.config.clip_values:
            return X
            
        # Make a copy if needed to avoid modifying the original
        if self.config.copy_X:
            X = X.copy()
            
        # Get clip values from config or use defaults
        clip_min = self.clip_min
        clip_max = self.clip_max
        
        # Apply clipping
        np.clip(X, clip_min, clip_max, out=X)
        
        return X
    
    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """
        Apply normalization to the data according to the configured method.
        
        Args:
            X: Input data array
            
        Returns:
            Normalized data array
        """
        if self.config.normalization == NormalizationType.NONE:
            return X
            
        # Make a copy if needed to avoid modifying the original
        if self.config.copy_X:
            X = X.copy()
        
        norm_type = self._get_norm_type(self.config.normalization)
        
        if norm_type == NormalizationType.STANDARD:
            if 'mean' not in self._stats or 'scale' not in self._stats:
                raise StatisticsError("Standard normalization stats not available")
                
            # Apply standard scaling: (X - mean) / std
            for i in range(X.shape[1]):
                X[:, i] = (X[:, i] - self._stats['mean'][i]) * self._stats['scale'][i]
                
        elif norm_type == NormalizationType.MINMAX:
            if 'min' not in self._stats or 'scale' not in self._stats:
                raise StatisticsError("Min-max normalization stats not available")
                
            # Apply min-max scaling: (X - min) / (max - min)
            for i in range(X.shape[1]):
                X[:, i] = (X[:, i] - self._stats['min'][i]) * self._stats['scale'][i]
                
        elif norm_type == NormalizationType.ROBUST:
            if 'lower' not in self._stats or 'scale' not in self._stats:
                raise StatisticsError("Robust normalization stats not available")
                
            # Apply robust scaling: (X - q_lower) / (q_upper - q_lower)
            for i in range(X.shape[1]):
                X[:, i] = (X[:, i] - self._stats['lower'][i]) * self._stats['scale'][i]
                
        elif norm_type == NormalizationType.QUANTILE:
            # Quantile transformation logic here
            if 'quantile_transformer' not in self._stats:
                raise StatisticsError("Quantile transformation stats not available")
            # This would typically use a fitted quantile transformer
            # Implementation would depend on additional libraries
            pass
                
        elif norm_type == NormalizationType.LOG:
            # Log transformation
            # Ensure all values are positive before log transform
            min_vals = np.min(X, axis=0)
            for i in range(X.shape[1]):
                if min_vals[i] <= 0:
                    # Add offset to make all values positive
                    offset = abs(min_vals[i]) + 1.0
                    X[:, i] = np.log(X[:, i] + offset)
                else:
                    X[:, i] = np.log(X[:, i])
                
        elif norm_type == NormalizationType.POWER:
            # Power transformation (Box-Cox like)
            # Typically would use scipy.stats.boxcox or similar
            pass
                
        else:  # Custom normalization
            if self.config.custom_normalization_fn is not None:
                try:
                    X = self.config.custom_normalization_fn(X, self._stats)
                except Exception as e:
                    self.logger.error(f"Error in custom normalization function: {e}")
                    # Fall back to standard normalization if possible
                    if 'mean' in self._stats and 'scale' in self._stats:
                        for i in range(X.shape[1]):
                            X[:, i] = (X[:, i] - self._stats['mean'][i]) * self._stats['scale'][i]
                    else:
                        raise StatisticsError("Custom normalization failed and no fallback available")
                
        # Apply scale shift if configured (e.g., to [0,1] range)
        if self.scale_shift:
            X = X + self.scale_offset
        
        return X

    def _get_norm_type(self, normalization_type: Union[str, NormalizationType]) -> NormalizationType:
        """Convert string normalization type to enum if needed and validate."""
        if isinstance(normalization_type, str):
            try:
                return NormalizationType[normalization_type.upper()]
            except KeyError:
                self.logger.warning(f"Unknown normalization type '{normalization_type}', using 'NONE'")
                return NormalizationType.NONE
        return normalization_type

    def _handle_nan_values(self, X: np.ndarray) -> np.ndarray:
        """
        Handle NaN values in the data according to the configured strategy.
        
        Args:
            X: Input data array
            
        Returns:
            Data array with NaN values handled
        """
        if not self.config.handle_nan or not np.any(np.isnan(X)):
            return X
            
        # Make a copy if needed to avoid modifying the original
        if self.config.copy_X:
            X = X.copy()
        
        # Get strategy from config
        strategy = self.config.nan_strategy.lower()
        
        # Replace NaN values based on strategy
        if strategy == "mean":
            if 'mean' not in self._stats:
                # Calculate mean on the fly if not available
                self._stats['mean'] = np.nanmean(X, axis=0)
                
            for i in range(X.shape[1]):
                nan_mask = np.isnan(X[:, i])
                if np.any(nan_mask):
                    X[nan_mask, i] = self._stats['mean'][i]
                    
        elif strategy == "median":
            if 'median' not in self._stats:
                # Calculate median on the fly if not available
                self._stats['median'] = np.nanmedian(X, axis=0)
                
            for i in range(X.shape[1]):
                nan_mask = np.isnan(X[:, i])
                if np.any(nan_mask):
                    X[nan_mask, i] = self._stats['median'][i]
                    
        elif strategy == "most_frequent":
            if 'most_frequent' not in self._stats:
                # Calculate most frequent values if not available
                self._compute_most_frequent_values(X)
                
            for i in range(X.shape[1]):
                nan_mask = np.isnan(X[:, i])
                if np.any(nan_mask):
                    X[nan_mask, i] = self._stats['most_frequent'][i]
                    
        elif strategy == "constant":
            # Use configured fill value
            nan_fill_value = self.nan_fill_value
            X[np.isnan(X)] = nan_fill_value
            
        else:  # Default to zero
            X[np.isnan(X)] = 0.0
        
        return X

    def _handle_infinite_values(self, X: np.ndarray) -> np.ndarray:
        """
        Handle infinite values in the data according to the configured strategy.
        
        Args:
            X: Input data array
            
        Returns:
            Data array with infinite values handled
        """
        if not self.config.handle_inf or not np.any(~np.isfinite(X)):
            return X
            
        # Make a copy if needed to avoid modifying the original
        if self.config.copy_X:
            X = X.copy()
        
        # Get strategy from config
        strategy = self.config.inf_strategy.lower()
        
        # Get masks for positive and negative infinities
        pos_inf_mask = np.isposinf(X)
        neg_inf_mask = np.isneginf(X)
        
        # Replace infinite values based on strategy
        if strategy == "max_value":
            # Replace with a large value based on data max and buffer
            if 'max' not in self._stats:
                # Calculate max on the fly if not available (ignoring infs)
                finite_X = X[np.isfinite(X)] if np.any(np.isfinite(X)) else np.zeros(1)
                self._stats['max'] = np.max(finite_X) if finite_X.size > 0 else 1.0
                
            # Use max + buffer for pos inf, min - buffer for neg inf
            pos_value = self._stats['max'] * self.inf_buffer
            neg_value = -pos_value if 'min' not in self._stats else self._stats['min'] * self.inf_buffer
            
            X[pos_inf_mask] = pos_value
            X[neg_inf_mask] = neg_value
            
        elif strategy == "median":
            if 'median' not in self._stats:
                # Calculate median on the fly if not available (ignoring infs)
                finite_X = X[np.isfinite(X)]
                self._stats['median'] = np.median(finite_X)
                
            # Use median for all infinities
            X[pos_inf_mask | neg_inf_mask] = self._stats['median']
            
        elif strategy == "constant":
            # Use configured fill values
            X[pos_inf_mask] = self.pos_inf_fill_value
            X[neg_inf_mask] = self.neg_inf_fill_value
            
        else:  # Default to NaN
            X[~np.isfinite(X)] = np.nan
            # Then handle the NaN values
            X = self._handle_nan_values(X)
        
        return X

    def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
        """
        Handle outliers in the data according to the configured method.
        
        Args:
            X: Input data array
            
        Returns:
            Data array with outliers handled
        """
        if not self.config.detect_outliers or 'outlier_lower' not in self._stats or 'outlier_upper' not in self._stats:
            return X
            
        # Make a copy if needed to avoid modifying the original
        if self.config.copy_X:
            X = X.copy()
        
        # Get outlier handling strategy
        strategy = self.config.outlier_handling.lower()
        
        # Get outlier bounds
        lower_bound = self._stats['outlier_lower']
        upper_bound = self._stats['outlier_upper']
        
        # Check feature-specific outlier handling if configured
        feature_specific = self.feature_outlier_mask is not None
        
        # Validate feature_outlier_mask length if it's being used
        if feature_specific and len(self.feature_outlier_mask) < X.shape[1]:
            self.logger.warning(
                f"feature_outlier_mask length ({len(self.feature_outlier_mask)}) is smaller than number of features ({X.shape[1]}). "
                "Using available mask values and treating remaining features as non-masked."
            )
        
        # Handle outliers differently based on strategy
        if strategy == "clip":
            # Clip values to the bounds
            for i in range(X.shape[1]):
                # Check if index is valid in the mask before accessing it
                if not feature_specific or (i < len(self.feature_outlier_mask) and self.feature_outlier_mask[i]):
                    col_lower = lower_bound[i]
                    col_upper = upper_bound[i]
                    X[:, i] = np.clip(X[:, i], col_lower, col_upper)
                    
        elif strategy == "remove":
            # This would typically return a mask or indices to remove rows
            # But since we need to return same-shape array, we'll set to NaN instead
            outlier_mask = np.zeros_like(X, dtype=bool)
            
            for i in range(X.shape[1]):
                # Check if index is valid in the mask before accessing it
                if not feature_specific or (i < len(self.feature_outlier_mask) and self.feature_outlier_mask[i]):
                    col_mask = (X[:, i] < lower_bound[i]) | (X[:, i] > upper_bound[i])
                    outlier_mask[:, i] = col_mask
                    
            # Set outliers to NaN
            X[outlier_mask] = np.nan
            
            # Handle the NaN values
            X = self._handle_nan_values(X)
                    
        elif strategy == "winsorize":
            # Replace outliers with the boundary values
            for i in range(X.shape[1]):
                # Check if index is valid in the mask before accessing it
                if not feature_specific or (i < len(self.feature_outlier_mask) and self.feature_outlier_mask[i]):
                    col = X[:, i]
                    # Replace low outliers
                    low_mask = col < lower_bound[i]
                    if np.any(low_mask):
                        col[low_mask] = lower_bound[i]
                    
                    # Replace high outliers
                    high_mask = col > upper_bound[i]
                    if np.any(high_mask):
                        col[high_mask] = upper_bound[i]
                        
        elif strategy == "mean":
            # Replace outliers with the mean
            if 'mean' not in self._stats:
                self._stats['mean'] = np.mean(X, axis=0)
                
            for i in range(X.shape[1]):
                # Check if index is valid in the mask before accessing it
                if not feature_specific or (i < len(self.feature_outlier_mask) and self.feature_outlier_mask[i]):
                    col = X[:, i]
                    outlier_mask = (col < lower_bound[i]) | (col > upper_bound[i])
                    if np.any(outlier_mask):
                        col[outlier_mask] = self._stats['mean'][i]
        
        return X
    def serialize(self, path: str) -> bool:
        """
        Serialize the preprocessor to disk.
        
        Args:
            path: File path to save the serialized preprocessor
            
        Returns:
            True if successful, False otherwise
        """
        with self._error_context("serialization"):
            try:
                # Create serializable state
                state = {
                    'config': asdict(self.config),
                    'stats': self._stats,
                    'feature_names': self._feature_names,
                    'fitted': self._fitted,
                    'n_features': self._n_features,
                    'n_samples_seen': self._n_samples_seen,
                    'version': self._version
                }
                
                # Add hash of data for verification
                state['hash'] = hashlib.md5(json.dumps(str(state)).encode('utf-8')).hexdigest()
                
                # Save to disk
                with open(path, 'wb') as f:
                    pickle.dump(state, f)
                    
                self.logger.info(f"Preprocessor serialized to {path}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to serialize preprocessor: {e}")
                return False

    @classmethod
    def deserialize(cls, path: str) -> 'DataPreprocessor':
        """
        Deserialize a preprocessor from disk.
        
        Args:
            path: File path to load the serialized preprocessor from
            
        Returns:
            Deserialized DataPreprocessor instance
        """
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            # Create config from dict
            from ..configs import PreprocessorConfig
            config = PreprocessorConfig(**state['config'])
            
            # Create instance
            instance = cls(config)
            
            # Restore state
            instance._stats = state['stats']
            instance._feature_names = state['feature_names']
            instance._fitted = state['fitted']
            instance._n_features = state['n_features']
            instance._n_samples_seen = state['n_samples_seen']
            instance._version = state.get('version', '0.1.4')
            
            # Verify hash for data integrity
            saved_hash = state.get('hash', None)
            if saved_hash:
                # Create a new hash without the hash field
                verify_state = state.copy()
                verify_state.pop('hash', None)
                verify_hash = hashlib.md5(json.dumps(str(verify_state)).encode('utf-8')).hexdigest()
                
                if saved_hash != verify_hash:
                    raise SerializationError("Data integrity check failed. The model file may be corrupted.")
            
            return instance
            
        except Exception as e:
            raise SerializationError(f"Failed to deserialize preprocessor: {e}") from e

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get the current statistics computed by the preprocessor.
        
        Returns:
            Dictionary of statistics
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before getting statistics")
            
        return self._stats.copy()

    def get_feature_names(self) -> List[str]:
        """
        Get the feature names used by the preprocessor.
        
        Returns:
            List of feature names
        """
        return self._feature_names.copy()

    def get_performance_metrics(self) -> Dict[str, List[float]]:
        """
        Get performance metrics collected during preprocessing operations.
        
        Returns:
            Dictionary of performance metrics
        """
        return self._metrics.copy()

    def reverse_transform(self, X: np.ndarray, copy: bool = True) -> np.ndarray:
        """
        Reverse the preprocessing transformations to recover original data scale.
        
        Args:
            X: Transformed data to reverse
            copy: Whether to copy the input data
            
        Returns:
            Data with transformations reversed
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before reverse transform")
            
        with self._error_context("inverse transform operation"):
            # Validate and convert input
            X = self._validate_and_convert_input(X, operation="reverse_transform", copy=copy)
            
            # Apply inverse transformations in reverse order of the forward transformations
            norm_type = self._get_norm_type(self.config.normalization)
            
            # Handle scale shift first if applied
            if self.scale_shift:
                X = X - self.scale_offset
            
            # Reverse normalization
            if norm_type != NormalizationType.NONE:
                if norm_type == NormalizationType.STANDARD:
                    if 'mean' not in self._stats or 'scale' not in self._stats:
                        raise StatisticsError("Standard normalization stats not available")
                        
                    # Reverse standard scaling: X / scale + mean
                    for i in range(X.shape[1]):
                        X[:, i] = (X[:, i] / self._stats['scale'][i]) + self._stats['mean'][i]
                        
                elif norm_type == NormalizationType.MINMAX:
                    if 'min' not in self._stats or 'scale' not in self._stats or 'range' not in self._stats:
                        raise StatisticsError("Min-max normalization stats not available")
                        
                    # Reverse min-max scaling: X / scale + min
                    for i in range(X.shape[1]):
                        X[:, i] = (X[:, i] / self._stats['scale'][i]) + self._stats['min'][i]
                        
                elif norm_type == NormalizationType.ROBUST:
                    if 'lower' not in self._stats or 'scale' not in self._stats:
                        raise StatisticsError("Robust normalization stats not available")
                        
                    # Reverse robust scaling: X / scale + lower
                    for i in range(X.shape[1]):
                        X[:, i] = (X[:, i] / self._stats['scale'][i]) + self._stats['lower'][i]
                        
                elif norm_type == NormalizationType.LOG:
                    # Reverse log transformation
                    X = np.exp(X)
                    
                    # Handle any offsets that were applied during the forward transform
                    if 'log_offset' in self._stats:
                        for i in range(X.shape[1]):
                            if self._stats['log_offset'][i] > 0:
                                X[:, i] = X[:, i] - self._stats['log_offset'][i]
                    
                elif norm_type == NormalizationType.CUSTOM:
                    if self.config.custom_inverse_fn is not None:
                        try:
                            X = self.config.custom_inverse_fn(X, self._stats)
                        except Exception as e:
                            self.logger.error(f"Error in custom inverse function: {e}")
                            raise
            
            # We don't reverse outlier handling or nan/inf handling
            # as that information is generally lost in the forward transform
            
            return X

    def reset(self) -> None:
        """
        Reset the preprocessor to its initialized state.
        
        This clears all fitted statistics and state.
        """
        self.logger.info("Resetting preprocessor state")
        
        # Clear state variables
        self._stats = {}
        self._fitted = False
        self._n_samples_seen = 0
        
        # Clear metrics
        for key in self._metrics:
            self._metrics[key] = []
        
        # Clear cache if enabled
        if self.config.cache_enabled:
            self._transform_cache = {}

    def update_config(self, new_config: PreprocessorConfig) -> None:
        """
        Update the preprocessor configuration.
        
        Args:
            new_config: New configuration to apply
        
        Note:
            This will reset the preprocessor state if the configuration changes
            affect the preprocessing pipeline.
        """
        pipeline_changed = (
            new_config.normalization != self.config.normalization or
            new_config.detect_outliers != self.config.detect_outliers or
            new_config.handle_nan != self.config.handle_nan or
            new_config.handle_inf != self.config.handle_inf or
            new_config.clip_values != self.config.clip_values
        )
        
        # Update configuration
        self.config = new_config
        
        # Update instance variables from config
        self.feature_outlier_mask = getattr(new_config, 'feature_outlier_mask', None)
        self.nan_fill_value = getattr(new_config, 'nan_fill_value', 0.0)
        self.pos_inf_fill_value = getattr(new_config, 'pos_inf_fill_value', float('nan'))
        self.neg_inf_fill_value = getattr(new_config, 'neg_inf_fill_value', float('nan'))
        self.inf_buffer = getattr(new_config, 'inf_buffer', 1.0)
        self.clip_min = getattr(new_config, 'clip_min', float('-inf'))
        self.clip_max = getattr(new_config, 'clip_max', float('inf'))
        self.scale_shift = getattr(new_config, 'scale_shift', False)
        self.scale_offset = getattr(new_config, 'scale_offset', 0.0)
        self.custom_center_func = getattr(new_config, 'custom_center_func', None)
        self.custom_scale_func = getattr(new_config, 'custom_scale_func', None)
        self.outlier_iqr_multiplier = getattr(new_config, 'outlier_iqr_multiplier', 1.5)
        self.outlier_zscore_threshold = getattr(new_config, 'outlier_zscore_threshold', 3.0)
        self.outlier_percentiles = getattr(new_config, 'outlier_percentiles', (1.0, 99.0))
        
        # Re-configure logging
        self._configure_logging()
        
        # Re-initialize processing functions
        self._init_processing_functions()
        
        # Re-initialize parallel executor if needed
        if self.config.parallel_processing:
            self._init_parallel_executor()
        else:
            self._parallel_executor = None
        
        # Reset state if pipeline has changed
        if pipeline_changed and self._fitted:
            self.logger.warning("Pipeline configuration changed, resetting preprocessor state")
            self.reset()

    def partial_fit(self, X: np.ndarray, feature_names=None) -> 'DataPreprocessor':
        """
        Update the preprocessor with new data incrementally.
        
        Args:
            X: New data to incorporate
            feature_names: Optional feature names (ignored if already fitted)
            
        Returns:
            self: The updated preprocessor
        """
        with self._lock, self._error_context("partial fit operation"):
            start_time = time.time()
            
            # Validate and convert input
            X = self._validate_and_convert_input(X, operation="partial_fit")
            
            # If this is the first call, initialize as a regular fit
            if not self._fitted:
                return self.fit(X, feature_names=feature_names)
            
            # Verify dimensions match
            if X.shape[1] != self._n_features:
                raise InputValidationError(
                    f"Expected {self._n_features} features, got {X.shape[1]}"
                )
            
            # Update statistics incrementally
            self._update_stats_incrementally(X)
            
            # Update samples seen
            self._n_samples_seen += X.shape[0]
            
            # Track fit time for metrics
            fit_time = time.time() - start_time
            self._metrics['fit_time'].append(fit_time)
            
            self.logger.info(f"Partially fitted on additional {X.shape[0]} samples in {fit_time:.4f}s")
            return self

    def _update_stats_incrementally(self, X: np.ndarray) -> None:
        """
        Update statistics incrementally with new data.
        
        Args:
            X: New data to incorporate
        """
        norm_type = self._get_norm_type(self.config.normalization)
        
        # Update normalization statistics
        if norm_type == NormalizationType.STANDARD:
            # Update mean and std incrementally using Welford's algorithm
            if 'mean' not in self._stats or 'std' not in self._stats:
                self._compute_standard_stats(X)
                return
                
            n_prev = self._n_samples_seen
            n_new = X.shape[0]
            n_total = n_prev + n_new
            
            # Get previous statistics
            prev_mean = self._stats['mean']
            prev_var = self._stats['std'] ** 2
            
            # Compute statistics for new data
            new_mean = np.mean(X, axis=0)
            new_var = np.var(X, axis=0)
            
            # Combine statistics
            delta = new_mean - prev_mean
            m_a = prev_var * n_prev
            m_b = new_var * n_new
            M2 = m_a + m_b + delta ** 2 * n_prev * n_new / n_total
            
            # Update statistics
            updated_mean = prev_mean + delta * n_new / n_total
            updated_var = M2 / n_total
            updated_std = np.sqrt(np.maximum(updated_var, self.config.epsilon))
            updated_scale = 1.0 / updated_std
            
            # Store updated statistics
            self._stats['mean'] = updated_mean
            self._stats['std'] = updated_std
            self._stats['scale'] = updated_scale
            
        elif norm_type == NormalizationType.MINMAX:
            # Update min and max
            if 'min' not in self._stats or 'max' not in self._stats:
                self._compute_minmax_stats(X)
                return
                
            prev_min = self._stats['min']
            prev_max = self._stats['max']
            
            # Compute min and max for new data
            new_min = np.min(X, axis=0)
            new_max = np.max(X, axis=0)
            
            # Update min and max
            updated_min = np.minimum(prev_min, new_min)
            updated_max = np.maximum(prev_max, new_max)
            
            # Update range and scale
            updated_range = updated_max - updated_min
            updated_range = np.maximum(updated_range, self.config.epsilon)
            updated_scale = 1.0 / updated_range
            
            # Store updated statistics
            self._stats['min'] = updated_min
            self._stats['max'] = updated_max
            self._stats['range'] = updated_range
            self._stats['scale'] = updated_scale
            
        elif norm_type == NormalizationType.ROBUST:
            # For robust statistics, we need to recalculate percentiles
            # This is an approximation as exact incremental percentiles are complex
            
            # Get previous statistics or initialize
            if 'lower' not in self._stats or 'upper' not in self._stats:
                self._compute_robust_stats(X)
                return
                
            # For now, we'll recompute on a sample of old + new data
            # In a production system, you might use a more sophisticated algorithm
            
            # Get percentiles and sample size
            lower, upper = self.config.robust_percentiles
            sample_size = min(10000, self._n_samples_seen)
            
            # Generate random indices from previous data
            if hasattr(self, '_prev_sample_indices') and hasattr(self, '_prev_sample'):
                prev_indices = self._prev_sample_indices
                prev_sample = self._prev_sample
            else:
                # No previous sample, use current statistics as approximation
                prev_sample = None
                
            # Combine with new data
            if prev_sample is not None:
                combined = np.vstack([prev_sample, X])
            else:
                combined = X
                
            # Compute percentiles
            lower_percentile = np.percentile(combined, lower, axis=0)
            upper_percentile = np.percentile(combined, upper, axis=0)
            
            # Update range and scale
            updated_range = upper_percentile - lower_percentile
            updated_range = np.maximum(updated_range, self.config.epsilon)
            updated_scale = 1.0 / updated_range
            
            # Store updated statistics
            self._stats['lower'] = lower_percentile
            self._stats['upper'] = upper_percentile
            self._stats['range'] = updated_range
            self._stats['scale'] = updated_scale
            
            # Store a sample of this data for future updates
            if X.shape[0] > sample_size:
                indices = np.random.choice(X.shape[0], sample_size, replace=False)
                self._prev_sample = X[indices].copy()
            else:
                self._prev_sample = X.copy()
            self._prev_sample_indices = np.arange(len(self._prev_sample))
            
        # Update outlier statistics if needed
        if self.config.detect_outliers:
            self._update_outlier_stats(X)

    def _update_outlier_stats(self, X: np.ndarray) -> None:
        """
        Update outlier statistics incrementally with new data.
        
        Args:
            X: New data to incorporate
        """
        outlier_method = self.config.outlier_method.lower()
        
        # Initialize if not already present
        if 'outlier_lower' not in self._stats or 'outlier_upper' not in self._stats:
            self._compute_outlier_stats(X)
            return
            
        if outlier_method == "iqr":
            # For IQR, we'll recalculate on a sample as exact incremental IQR is complex
            if hasattr(self, '_prev_sample'):
                combined = np.vstack([self._prev_sample, X])
            else:
                combined = X
                
            # Compute IQR-based bounds
            q1 = np.percentile(combined, 25, axis=0)
            q3 = np.percentile(combined, 75, axis=0)
            iqr = q3 - q1
            
            lower_bound = q1 - (self.outlier_iqr_multiplier * iqr)
            upper_bound = q3 + (self.outlier_iqr_multiplier * iqr)
            
        elif outlier_method == "zscore":
            # For Z-score, we can use our incrementally updated mean and std
            if 'mean' not in self._stats or 'std' not in self._stats:
                self._update_stats_incrementally(X)
                
            mean = self._stats['mean']
            std = self._stats['std']
            
            lower_bound = mean - (self.outlier_zscore_threshold * std)
            upper_bound = mean + (self.outlier_zscore_threshold * std)
            
        elif outlier_method == "percentile":
            # For percentile bounds, we'll recalculate on a sample
            if hasattr(self, '_prev_sample'):
                combined = np.vstack([self._prev_sample, X])
            else:
                combined = X
                
            lower, upper = self.outlier_percentiles
            lower_bound = np.percentile(combined, lower, axis=0)
            upper_bound = np.percentile(combined, upper, axis=0)
            
        else:
            # Default to Z-score method
            if 'mean' not in self._stats or 'std' not in self._stats:
                self._update_stats_incrementally(X)
                
            mean = self._stats['mean']
            std = self._stats['std']
            
            lower_bound = mean - (3.0 * std)  # Default to 3 sigma
            upper_bound = mean + (3.0 * std)
        
        # Store updated outlier bounds
        self._stats['outlier_lower'] = lower_bound
        self._stats['outlier_upper'] = upper_bound