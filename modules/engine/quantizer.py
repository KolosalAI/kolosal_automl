import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict, Union, Any
from dataclasses import asdict
import logging
import threading
import time
import collections
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

from modules.configs import QuantizationConfig, QuantizationType, QuantizationMode
# Define enums that would be in the configs.py module


class Quantizer:
    """
    High-performance data quantizer with advanced features:
    - Multiple quantization types (int8, uint8, int16)
    - Various quantization modes (symmetric, asymmetric, per-channel)
    - Thread-safe operations for concurrent environments
    - Optimized cache mechanisms for frequent values
    - Statistical tracking and calibration
    - Vectorized operations with memory optimization
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize quantizer with configurable parameters.

        Args:
            config: Optional QuantizerConfig object. If not provided, defaults are used.
        """
        self.config = config if config is not None else QuantizationConfig()
        
        # Thread safety mechanisms with more granular locking
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._stats_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        
        # Initialize quantization parameters
        self._scales: Dict[int, float] = {}  # For per-channel quantization
        self._zero_points: Dict[int, float] = {}  # For per-channel quantization
        self._global_scale: float = 1.0
        self._global_zero_point: float = 0.0
        
        # Statistics
        self._quantization_stats = {
            'clipped_values': 0,
            'total_values': 0,
            'quantize_calls': 0,
            'dequantize_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_scale': 1.0,
            'last_zero_point': 0.0,
            'processing_time_ms': 0.0,
        }
        
        # Cache for dequantization - use OrderedDict for efficient LRU implementation
        self._cache_enabled = self.config.enable_cache
        self._cache_size = self.config.cache_size
        self._value_cache = collections.OrderedDict()
        
        # Determine quantization type and range values
        self._setup_quantization_ranges()
        
        # Performance optimization: preallocate buffers if needed
        self._initialize_buffers()
        
        logger.info(f"Quantizer initialized with {self.config.quantization_type} type and "
                    f"{self.config.quantization_mode} mode")

    def hash_array(arr: NDArray) -> str:
        """
        Create a hashable representation of a numpy array.
        
        Args:
            arr: NumPy array to hash
            
        Returns:
            String hash representation of the array
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(arr)}")
            
        # For small arrays, use a simple approach
        if arr.size <= 100:
            # Convert to bytes and then hash
            return hashlib.md5(arr.tobytes()).hexdigest()
        else:
            # For larger arrays, compute a hash from stats and a sample of values
            # This is faster but still reasonably unique
            stats_str = f"{arr.shape}_{arr.dtype}_{np.mean(arr)}_{np.std(arr)}"
            # Sample some values from the array
            sample_indices = np.linspace(0, arr.size-1, 20, dtype=int)
            samples = arr.flat[sample_indices]
            sample_str = '_'.join(str(x) for x in samples)
            return hashlib.md5((stats_str + sample_str).encode()).hexdigest()

    def _setup_quantization_ranges(self) -> None:
        """Set up range values based on quantization type and bit width."""
        q_type = self.config.quantization_type
        num_bits = self.config.num_bits
        
        # Calculate range based on bit width
        if q_type == QuantizationType.INT8.value:
            self._qmin = -(2 ** (num_bits - 1))
            self._qmax = 2 ** (num_bits - 1) - 1
            self._qrange = 2 ** num_bits - 1
            self._qtype = np.int8
        elif q_type == QuantizationType.UINT8.value:
            self._qmin = 0
            self._qmax = 2 ** num_bits - 1
            self._qrange = 2 ** num_bits - 1
            self._qtype = np.uint8
        elif q_type == QuantizationType.INT16.value:
            self._qmin = -(2 ** (num_bits - 1))
            self._qmax = 2 ** (num_bits - 1) - 1
            self._qrange = 2 ** num_bits - 1
            self._qtype = np.int16
        else:
            raise ValueError(f"Unsupported quantization type: {q_type}")
        
        logger.debug(f"Quantization range: [{self._qmin}, {self._qmax}], range: {self._qrange}")
    
    def _initialize_buffers(self) -> None:
        """Initialize preallocated buffers for performance."""
        self._buffer_size = self.config.buffer_size
        if self._buffer_size > 0:
            # Only preallocate if buffer size is specified
            self._input_buffer = np.zeros((self._buffer_size,), dtype=np.float32)
            self._output_buffer = np.zeros((self._buffer_size,), dtype=self._qtype)
            logger.debug(f"Initialized buffers with size {self._buffer_size}")

    @property
    def scale(self) -> float:
        """Get the current global scale factor"""
        with self._lock:
            return self._global_scale

    @property
    def zero_point(self) -> float:
        """Get the current global zero point"""
        with self._lock:
            return self._global_zero_point
    
    def update_config(self, new_config: QuantizationConfig) -> None:
        """
        Update the quantizer configuration with thread safety.

        Args:
            new_config: A new QuantizerConfig object to update parameters.
        """
        with self._lock:
            old_type = self.config.quantization_type
            old_bits = self.config.num_bits
            self.config = new_config
            self._cache_enabled = new_config.enable_cache
            self._cache_size = new_config.cache_size
            
            # If quantization type or bit width changed, update ranges
            if old_type != new_config.quantization_type or old_bits != new_config.num_bits:
                self._setup_quantization_ranges()
                # Reinitialize buffers if needed
                self._initialize_buffers()
            
            # Clear caches
            with self._cache_lock:
                self._value_cache.clear()
            
            logger.info(f"Quantizer config updated: type={new_config.quantization_type}, "
                       f"mode={new_config.quantization_mode}, bits={new_config.num_bits}")

    def get_config(self) -> Dict[str, Any]:
        """
        Retrieve the current configuration as a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        # Thread-safe access to config
        with self._lock:
            return asdict(self.config)

    def compute_scale_and_zero_point(
        self, 
        data: NDArray[np.float32], 
        channel_idx: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Compute optimal scale and zero_point based on current quantization mode.

        Args:
            data: Input float32 array
            channel_idx: Optional channel index for per-channel quantization

        Returns:
            Tuple of (scale, zero_point)
        """
        # Check if data is empty
        if data.size == 0:
            logger.warning("Cannot compute scale and zero point for empty data")
            return 1.0, 0.0
        
        # Early exit if not using dynamic range
        if self.config.quantization_mode not in [
            QuantizationMode.DYNAMIC_PER_BATCH.value,
            QuantizationMode.DYNAMIC_PER_CHANNEL.value
        ]:
            with self._lock:
                return self._global_scale, self._global_zero_point

        # Handle outliers if threshold is set
        if self.config.outlier_threshold is not None:
            # Compute mean and std for outlier removal
            data_mean = float(np.mean(data))
            data_std = float(np.std(data))
            threshold = self.config.outlier_threshold
            # Create a copy of data with outliers clipped
            filtered_data = np.clip(
                data, 
                data_mean - threshold * data_std,
                data_mean + threshold * data_std
            )
        else:
            filtered_data = data

        # Get data min/max with robustness against outliers
        if self.config.use_percentile:
            # Use percentiles to avoid extreme outliers
            data_min = float(np.percentile(filtered_data, self.config.min_percentile))
            data_max = float(np.percentile(filtered_data, self.config.max_percentile))
        else:
            # Use absolute min/max
            data_min = float(np.min(filtered_data))
            data_max = float(np.max(filtered_data))

        # Ensure range is not too small (avoid division by zero)
        data_range = max(data_max - data_min, 1e-7)
        
        if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
            # Symmetric quantization centers around zero
            abs_max = max(abs(data_min), abs(data_max))
            # Add a small epsilon to improve accuracy
            abs_max = abs_max * 1.01  # Add 1% margin to reduce clipping
            scale = abs_max / max(abs(self._qmin), abs(self._qmax))
            zero_point = 0.0
        else:
            # Asymmetric quantization uses full range
            # Add a small margin to improve accuracy
            data_range = data_range * 1.01  # Add 1% margin
            scale = data_range / self._qrange
            
            if self._qtype == np.uint8:
                # For uint8, calculate zero point directly
                zero_point_uint8 = round(self._qmin - data_min / scale)
                zero_point_uint8 = max(self._qmin, min(self._qmax, zero_point_uint8))
                zero_point = data_min
            else:
                # For int8/int16, calculate zero point
                zero_point = round(data_min / scale) * scale  # Align to quantization grid
        
        # Store the computed values
        with self._lock:
            if channel_idx is not None:
                # Store per-channel values
                self._scales[channel_idx] = scale
                self._zero_points[channel_idx] = zero_point
            else:
                # Store global values
                self._global_scale = scale
                self._global_zero_point = zero_point
                # Update stats
                with self._stats_lock:
                    self._quantization_stats['last_scale'] = scale
                    self._quantization_stats['last_zero_point'] = zero_point
        
        return scale, zero_point

    def quantize(
        self, 
        data: Union[NDArray[np.float32], Any], 
        validate: bool = True,
        channel_dim: Optional[int] = None
    ) -> NDArray:
        """
        High-performance vectorized quantization with multiple modes.

        Args:
            data: Input float32 array
            validate: If True, performs input validation
            channel_dim: Dimension index for per-channel quantization (if enabled)

        Returns:
            Quantized array of the configured type
        """
        start_time = time.time()
        
        # Input validation (if enabled)
        if validate:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"Input must be a numpy array, got {type(data)}")
            self._validate_input(data)
        
        # Record stats
        with self._stats_lock:
            self._quantization_stats['quantize_calls'] += 1
            self._quantization_stats['total_values'] += data.size
        
        # Memory optimization: work with a view if possible to avoid copies
        if self.config.optimize_memory and data.flags.c_contiguous:
            input_data = data
        else:
            # Create a contiguous copy for better performance
            input_data = np.ascontiguousarray(data)
        
        # Handle per-channel quantization if requested
        if channel_dim is not None and self.config.quantization_mode == QuantizationMode.DYNAMIC_PER_CHANNEL.value:
            result = self._quantize_per_channel(input_data, channel_dim)
        else:
            # Standard quantization path
            # Update scale and zero_point if using dynamic mode
            if self.config.quantization_mode in [
                QuantizationMode.DYNAMIC_PER_BATCH.value, 
                QuantizationMode.DYNAMIC_PER_CHANNEL.value
            ]:
                self.compute_scale_and_zero_point(input_data)
            
            # Get current scale and zero point (thread-safe)
            with self._lock:
                scale = self._global_scale
                zero_point = self._global_zero_point
            
            # Implement quantization based on quantization type
            if self._qtype == np.int8:
                # INT8 quantization formula
                if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                    # Symmetric: map [-max, max] to [-127, 127] (preserving 0)
                    quantized = np.clip(np.round(input_data / scale), self._qmin, self._qmax)
                else:
                    # Asymmetric: map [min, max] to [-128, 127]
                    quantized = np.clip(np.round((input_data - zero_point) / scale), self._qmin, self._qmax)
            
            elif self._qtype == np.uint8:
                # UINT8 quantization formula (always asymmetric)
                zero_point_uint8 = int(round(self._qmin - zero_point / scale))
                zero_point_uint8 = max(self._qmin, min(self._qmax, zero_point_uint8))
                quantized = np.clip(np.round(input_data / scale) + zero_point_uint8, self._qmin, self._qmax)
            
            elif self._qtype == np.int16:
                # INT16 quantization with higher precision
                if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                    quantized = np.clip(np.round(input_data / scale), self._qmin, self._qmax)
                else:
                    quantized = np.clip(np.round((input_data - zero_point) / scale), self._qmin, self._qmax)
            
            # Convert to target type efficiently
            result = quantized.astype(self._qtype)
            
            # Count clipped values for statistics
            if validate:
                with self._stats_lock:
                    self._quantization_stats['clipped_values'] += np.sum(
                        (input_data < self._qmin * scale) | (input_data > self._qmax * scale)
                    )
        
        # Record processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        with self._stats_lock:
            self._quantization_stats['processing_time_ms'] += processing_time
        
        return result

    def _quantize_per_channel(
        self, 
        data: NDArray[np.float32], 
        channel_dim: int
    ) -> NDArray:
        """
        Perform per-channel quantization for improved accuracy.

        Args:
            data: Input float32 array
            channel_dim: Dimension index for channels

        Returns:
            Quantized array with per-channel scaling
        """
        if channel_dim < 0 or channel_dim >= data.ndim:
            raise ValueError(f"Channel dimension {channel_dim} out of bounds for tensor with {data.ndim} dimensions")
            
        # Calculate number of channels
        n_channels = data.shape[channel_dim]
        
        # Create output array of the same shape
        output = np.zeros(data.shape, dtype=self._qtype)
        
        # Optimize: Transpose data to make channel_dim the first dimension for efficient processing
        # This allows us to process each channel in one contiguous memory block
        if channel_dim != 0:
            # Create permutation that puts channel_dim first
            perm = list(range(data.ndim))
            perm.remove(channel_dim)
            perm.insert(0, channel_dim)
            
            # Inverse permutation to restore original shape
            inv_perm = list(range(data.ndim))
            for i, p in enumerate(perm):
                inv_perm[p] = i
            
            # Transpose data
            data_t = np.transpose(data, perm)
            output_t = np.zeros_like(data_t, dtype=self._qtype)
            
            # Pre-compute scales and zero points for all channels to improve accuracy
            for c in range(n_channels):
                channel_data = data_t[c]
                # Use a slightly higher outlier threshold for per-channel quantization
                with self._lock:
                    old_threshold = self.config.outlier_threshold
                    self.config.outlier_threshold = 3.5 if old_threshold is None else old_threshold * 1.1
                    # Compute scale and zero point for this channel
                    self.compute_scale_and_zero_point(channel_data, channel_idx=c)
                    # Restore original threshold
                    self.config.outlier_threshold = old_threshold
                    
            # Process each channel in transposed data
            for c in range(n_channels):
                # Get the pre-computed scale and zero point
                with self._lock:
                    scale = self._scales.get(c, self._global_scale)
                    zero_point = self._zero_points.get(c, self._global_zero_point)
                
                channel_data = data_t[c]
                
                # Quantize this channel with improved precision
                if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                    quantized = np.clip(np.round(channel_data / scale), self._qmin, self._qmax)
                else:
                    quantized = np.clip(np.round((channel_data - zero_point) / scale), 
                                        self._qmin, self._qmax)
                
                # Store result in transposed output
                output_t[c] = quantized.astype(self._qtype)
            
            # Transpose back to original shape
            output = np.transpose(output_t, inv_perm)
        else:
            # Channel is already the first dimension, process directly
            # First pre-compute all scales and zero points
            for c in range(n_channels):
                channel_data = data[c]
                # Use a slightly higher outlier threshold for per-channel
                with self._lock:
                    old_threshold = self.config.outlier_threshold
                    self.config.outlier_threshold = 3.5 if old_threshold is None else old_threshold * 1.1
                    # Compute scale and zero point
                    self.compute_scale_and_zero_point(channel_data, channel_idx=c)
                    # Restore threshold
                    self.config.outlier_threshold = old_threshold
                    
            # Then process each channel
            for c in range(n_channels):
                with self._lock:
                    scale = self._scales.get(c, self._global_scale)
                    zero_point = self._zero_points.get(c, self._global_zero_point)
                    
                channel_data = data[c]
                
                if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                    quantized = np.clip(np.round(channel_data / scale), self._qmin, self._qmax)
                else:
                    quantized = np.clip(np.round((channel_data - zero_point) / scale), 
                                        self._qmin, self._qmax)
                
                output[c] = quantized.astype(self._qtype)
        
        return output

    def _get_cached_value(self, quantized_value: int, scale: float, zero_point: float) -> Optional[float]:
        """Thread-safe cache lookup with proper key construction"""
        # Ensure values are primitive types, not numpy types
        cache_key = (int(quantized_value), float(scale), float(zero_point))
        with self._cache_lock:
            if cache_key in self._value_cache:
                # Move item to end (most recently used)
                value = self._value_cache.pop(cache_key)
                self._value_cache[cache_key] = value
                with self._stats_lock:
                    self._quantization_stats['cache_hits'] += 1
                return value
            
            with self._stats_lock:
                self._quantization_stats['cache_misses'] += 1
            return None
    
    def _cache_value(self, quantized_value: int, scale: float, zero_point: float, float_value: float) -> None:
        """Thread-safe cache update with LRU behavior"""
        if not self._cache_enabled:
            return
            
        # Ensure values are primitive types, not numpy types
        cache_key = (int(quantized_value), float(scale), float(zero_point))
        with self._cache_lock:
            # Implement LRU: remove oldest entry if cache full
            if len(self._value_cache) >= self._cache_size:
                self._value_cache.popitem(last=False)  # Remove oldest item
            
            # Add new value to cache (will be at the end, most recent)
            self._value_cache[cache_key] = float(float_value)

    def dequantize(
        self, 
        data: NDArray, 
        channel_dim: Optional[int] = None,
        use_cache: Optional[bool] = None
    ) -> NDArray[np.float32]:
        """
        Optimized dequantization with support for different quantization types and modes.

        Args:
            data: Input quantized array
            channel_dim: Dimension index for per-channel dequantization
            use_cache: Override cache setting from config

        Returns:
            Dequantized float32 array
        """
        start_time = time.time()
        
        # Record stats
        with self._stats_lock:
            self._quantization_stats['dequantize_calls'] += 1
        
        # Determine whether to use cache
        use_cache = self._cache_enabled if use_cache is None else use_cache
        
        # Handle per-channel dequantization if requested
        if channel_dim is not None and self.config.quantization_mode == QuantizationMode.DYNAMIC_PER_CHANNEL.value:
            return self._dequantize_per_channel(data, channel_dim)
        
        # Get current scale and zero point (thread-safe)
        with self._lock:
            scale = self._global_scale
            zero_point = self._global_zero_point
        
        # Optimization for scalar or small arrays: use cache
        if use_cache and data.size <= 10:  # Only use cache for small data
            # For small arrays, check cache for each value
            output = np.zeros_like(data, dtype=np.float32)
            for idx, val in np.ndenumerate(data):
                # Convert to Python int to ensure proper hashing
                val_int = int(val)
                cached_value = self._get_cached_value(val_int, scale, zero_point)
                if cached_value is not None:
                    output[idx] = cached_value
                else:
                    if self._qtype == np.int8:
                        if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                            float_val = float(val_int) * scale
                        else:
                            float_val = float(val_int) * scale + zero_point
                    elif self._qtype == np.uint8:
                        zero_point_uint8 = int(round(self._qmin - zero_point / scale))
                        float_val = (float(val_int) - zero_point_uint8) * scale
                    else:  # int16
                        if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                            float_val = float(val_int) * scale
                        else:
                            float_val = float(val_int) * scale + zero_point
                            
                    output[idx] = float_val
                    # Cache the computed value
                    self._cache_value(val_int, scale, zero_point, float_val)
            
            result = output
        else:
            # Vectorized implementation for larger arrays
            # Implement dequantization based on quantization type
            if self._qtype == np.int8:
                if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                    # Symmetric dequantization
                    result = data.astype(np.float32) * scale
                else:
                    # Asymmetric dequantization
                    result = data.astype(np.float32) * scale + zero_point
            
            elif self._qtype == np.uint8:
                # UINT8 dequantization
                zero_point_uint8 = int(round(self._qmin - zero_point / scale))
                result = (data.astype(np.float32) - zero_point_uint8) * scale
            
            elif self._qtype == np.int16:
                # INT16 dequantization
                if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                    result = data.astype(np.float32) * scale
                else:
                    result = data.astype(np.float32) * scale + zero_point
        
        # Record processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        with self._stats_lock:
            self._quantization_stats['processing_time_ms'] += processing_time
        
        return result

    def cache_quantized_array(self, quantized_array: NDArray, original_array: NDArray[np.float32]) -> None:
        """
        Cache a mapping between a quantized array and its original float array.
        Useful for frequent operations on the same data.
        
        Args:
            quantized_array: The quantized array
            original_array: The original float32 array
        """
        if not self._cache_enabled:
            return
            
        # Create a hash for the quantized array
        array_hash = self.hash_array(quantized_array)
        
        # Store the mapping in a separate cache
        with self._cache_lock:
            # Implement a simple LRU for array cache
            if not hasattr(self, '_array_cache'):
                self._array_cache = collections.OrderedDict()
                
            # Limit array cache size
            array_cache_limit = max(10, self._cache_size // 100)  # Smaller than value cache
            if len(self._array_cache) >= array_cache_limit:
                self._array_cache.popitem(last=False)
                
            # Store the array
            self._array_cache[array_hash] = original_array.copy()

    def get_cached_array(self, quantized_array: NDArray) -> Optional[NDArray[np.float32]]:
        """
        Retrieve a cached float array for a given quantized array.
        
        Args:
            quantized_array: The quantized array to look up
            
        Returns:
            The cached original float32 array if found, None otherwise
        """
        if not hasattr(self, '_array_cache') or not self._cache_enabled:
            return None
            
        # Create a hash for the quantized array
        array_hash = self.hash_array(quantized_array)
        
        # Look up the array
        with self._cache_lock:
            if array_hash in self._array_cache:
                # Move to end (most recently used)
                cached_array = self._array_cache.pop(array_hash)
                self._array_cache[array_hash] = cached_array
                return cached_array
                
        return None

    def _dequantize_per_channel(
        self, 
        data: NDArray, 
        channel_dim: int
    ) -> NDArray[np.float32]:
        """
        Perform per-channel dequantization.

        Args:
            data: Input quantized array
            channel_dim: Dimension index for channels

        Returns:
            Dequantized float32 array with per-channel scaling
        """
        if channel_dim < 0 or channel_dim >= data.ndim:
            raise ValueError(f"Channel dimension {channel_dim} out of bounds for tensor with {data.ndim} dimensions")
            
        # Calculate number of channels
        n_channels = data.shape[channel_dim]
        
        # Create output array of the same shape
        output = np.zeros(data.shape, dtype=np.float32)
        
        # Optimize: Transpose data to make channel_dim the first dimension for efficient processing
        if channel_dim != 0:
            # Create permutation that puts channel_dim first
            perm = list(range(data.ndim))
            perm.remove(channel_dim)
            perm.insert(0, channel_dim)
            
            # Inverse permutation to restore original shape
            inv_perm = list(range(data.ndim))
            for i, p in enumerate(perm):
                inv_perm[p] = i
            
            # Transpose data
            data_t = np.transpose(data, perm)
            output_t = np.zeros_like(data_t, dtype=np.float32)
            
            # Process each channel in transposed data
            for c in range(n_channels):
                # Get scale and zero point for this channel
                with self._lock:
                    if c in self._scales:
                        scale = self._scales[c]
                        zero_point = self._zero_points[c]
                    else:
                        # Fall back to global values if not found
                        scale = self._global_scale
                        zero_point = self._global_zero_point
                
                # Dequantize based on quantization mode
                if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                    output_t[c] = data_t[c].astype(np.float32) * scale
                else:
                    output_t[c] = data_t[c].astype(np.float32) * scale + zero_point
            
            # Transpose back to original shape
            output = np.transpose(output_t, inv_perm)
        else:
            # Channel is already the first dimension, process directly
            for c in range(n_channels):
                # Get scale and zero point for this channel
                with self._lock:
                    if c in self._scales:
                        scale = self._scales[c]
                        zero_point = self._zero_points[c]
                    else:
                        # Fall back to global values if not found
                        scale = self._global_scale
                        zero_point = self._global_zero_point
                
                # Dequantize based on quantization mode
                if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                    output[c] = data[c].astype(np.float32) * scale
                else:
                    output[c] = data[c].astype(np.float32) * scale + zero_point
        
        return output

    def quantize_dequantize(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Perform quantization followed by dequantization to simulate quantization loss.
        Useful for accuracy testing and model fine-tuning with quantization awareness.

        Args:
            data: Input float32 array

        Returns:
            Float32 array after quantization and dequantization
        """
        # Save current configuration
        with self._lock:
            old_outlier_threshold = self.config.outlier_threshold
            old_enable_cache = self.config.enable_cache
            
            # Temporarily adjust settings for better precision
            if self.config.outlier_threshold is None:
                self.config.outlier_threshold = 3.0
            else:
                self.config.outlier_threshold *= 1.1  # Increase threshold slightly
                
            # Enable cache for best performance
            self.config.enable_cache = True
            self._cache_enabled = True
        
        try:
            # For small data, handle differently to improve accuracy
            if data.size < 1000:
                # Determine best scale and zero point for this specific data
                scale, zero_point = self.compute_scale_and_zero_point(data)
                
                # Quantize with calculated parameters
                if self._qtype == np.int8:
                    if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                        quantized = np.clip(np.round(data / scale), self._qmin, self._qmax)
                    else:
                        quantized = np.clip(np.round((data - zero_point) / scale), self._qmin, self._qmax)
                elif self._qtype == np.uint8:
                    zero_point_uint8 = int(round(self._qmin - zero_point / scale))
                    zero_point_uint8 = max(self._qmin, min(self._qmax, zero_point_uint8))
                    quantized = np.clip(np.round(data / scale) + zero_point_uint8, self._qmin, self._qmax)
                else:  # int16
                    if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                        quantized = np.clip(np.round(data / scale), self._qmin, self._qmax)
                    else:
                        quantized = np.clip(np.round((data - zero_point) / scale), self._qmin, self._qmax)
                        
                # Convert to target type
                quantized_array = quantized.astype(self._qtype)
                
                # Dequantize directly without using self.dequantize to ensure consistency
                if self._qtype == np.int8:
                    if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                        result = quantized_array.astype(np.float32) * scale
                    else:
                        result = quantized_array.astype(np.float32) * scale + zero_point
                elif self._qtype == np.uint8:
                    result = (quantized_array.astype(np.float32) - zero_point_uint8) * scale
                else:  # int16
                    if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
                        result = quantized_array.astype(np.float32) * scale
                    else:
                        result = quantized_array.astype(np.float32) * scale + zero_point
            else:
                # For larger data, use standard functions
                # Quantize the data
                quantized_array = self.quantize(data)
                
                # Dequantize back to floating point
                result = self.dequantize(quantized_array)
        
        finally:
            # Restore original configuration
            with self._lock:
                self.config.outlier_threshold = old_outlier_threshold
                self.config.enable_cache = old_enable_cache
                self._cache_enabled = old_enable_cache
        
        return result

    def get_quantization_stats(self) -> Dict[str, Union[int, float]]:
        """
        Returns comprehensive statistics about quantization operations.
        
        Returns:
            Dictionary with quantization statistics
        """
        with self._stats_lock:
            stats = dict(self._quantization_stats)
            
        # Calculate derived statistics
        if stats['total_values'] > 0:
            stats['clipping_ratio'] = stats['clipped_values'] / stats['total_values']
        else:
            stats['clipping_ratio'] = 0.0
            
        if stats['quantize_calls'] > 0:
            stats['avg_elements_per_call'] = stats['total_values'] / stats['quantize_calls']
            stats['avg_processing_time_ms'] = stats['processing_time_ms'] / (stats['quantize_calls'] + stats['dequantize_calls'])
        else:
            stats['avg_elements_per_call'] = 0
            stats['avg_processing_time_ms'] = 0
            
        # Calculate cache efficiency if enabled
        cache_total = stats.get('cache_hits', 0) + stats.get('cache_misses', 0)
        if cache_total > 0:
            stats['cache_hit_ratio'] = stats['cache_hits'] / cache_total
        else:
            stats['cache_hit_ratio'] = 0.0
            
        # Add current number of cached values
        with self._cache_lock:
            stats['cached_values_count'] = len(self._value_cache)
            
        return stats

    def _validate_input(self, data: NDArray[np.float32]) -> None:
        """
        Validate input data with better error reporting and handling.
        
        Args:
            data: Input data to validate
            
        Raises:
            TypeError: If input is not a numpy array
            ValueError: If input has invalid values
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Input must be a numpy array, got {type(data)}")
            
        # Check data type without modifying input
        if data.dtype != np.float32:
            logger.warning(f"Input data type is {data.dtype}, expected float32. "
                          "This may affect quantization precision.")
        
        # Check for NaN or Inf values
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        if nan_count > 0:
            msg = f"Input contains {nan_count} NaN values"
            if self.config.error_on_nan:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                
        if inf_count > 0:
            msg = f"Input contains {inf_count} infinite values"
            if self.config.error_on_inf:
                raise ValueError(msg)
            else:
                logger.warning(msg)

    def reset_stats(self) -> None:
        """Reset all quantization statistics."""
        with self._stats_lock:
            self._quantization_stats = {
                'clipped_values': 0,
                'total_values': 0,
                'quantize_calls': 0,
                'dequantize_calls': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'last_scale': self._global_scale,
                'last_zero_point': self._global_zero_point,
                'processing_time_ms': 0.0,
            }
        logger.debug("Quantization statistics reset")

    def clear_cache(self) -> None:
        """Clear the dequantization value cache."""
        with self._cache_lock:
            self._value_cache.clear()
        logger.debug("Quantization cache cleared")
        
    def calibrate(self, calibration_data: NDArray[np.float32]) -> Dict[str, float]:
        """
        Calibrate the quantizer using representative data to find optimal parameters.
        
        Args:
            calibration_data: Representative data for calibration
            
        Returns:
            Dictionary with calibration results
        """
        logger.info(f"Calibrating quantizer with {calibration_data.size} values")
        
        # Validate calibration data
        self._validate_input(calibration_data)
        
        # Calculate optimal parameters based on data distribution
        data_min = float(np.min(calibration_data))
        data_max = float(np.max(calibration_data))
        data_mean = float(np.mean(calibration_data))
        data_std = float(np.std(calibration_data))
        
        # Handle outliers if threshold is set
        if self.config.outlier_threshold is not None:
            threshold = self.config.outlier_threshold
            # Create a copy of data with outliers clipped
            filtered_data = np.clip(
                calibration_data, 
                data_mean - threshold * data_std,
                data_mean + threshold * data_std
            )
            # Recalculate min/max on filtered data
            filtered_min = float(np.min(filtered_data))
            filtered_max = float(np.max(filtered_data))
            logger.info(f"Outlier filtering: original range [{data_min:.4f}, {data_max:.4f}], "
                        f"filtered range [{filtered_min:.4f}, {filtered_max:.4f}]")
            data_min = filtered_min
            data_max = filtered_max
        
        # Calculate optimal scale and zero point based on quantization mode
        if self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
            # For symmetric, use the maximum absolute value
            abs_max = max(abs(data_min), abs(data_max))
            scale = abs_max / max(abs(self._qmin), abs(self._qmax))
            zero_point = 0.0
        else:
            # For asymmetric, use the full range
            data_range = max(data_max - data_min, 1e-7)
            scale = data_range / self._qrange
            if self._qtype == np.uint8:
                zero_point = self._qmin - data_min / scale
            else:
                zero_point = data_min
        
        # Update the parameters
        with self._lock:
            self._global_scale = scale
            self._global_zero_point = zero_point
        
        # Quantize and dequantize to measure error
        q_data = self.quantize(calibration_data)
        dq_data = self.dequantize(q_data)
        
        # Calculate error metrics
        mse = float(np.mean((calibration_data - dq_data) ** 2))
        mae = float(np.mean(np.abs(calibration_data - dq_data)))
        max_error = float(np.max(np.abs(calibration_data - dq_data)))
        
        # Calculate Signal-to-Noise Ratio (SNR)
        signal_power = np.mean(calibration_data ** 2)
        noise_power = np.mean((calibration_data - dq_data) ** 2)
        snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
        
        # Calculate histogram of errors for better analysis
        errors = (calibration_data - dq_data).flatten()
        hist, bin_edges = np.histogram(errors, bins=20)
        hist_data = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
        # Create and return calibration report
        calibration_results = {
            'data_min': data_min,
            'data_max': data_max,
            'data_mean': data_mean,
            'data_std': data_std,
            'scale': scale,
            'zero_point': zero_point,
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'snr_db': snr,
            'error_hist': hist_data,
        }
        
        logger.info(f"Calibration complete: scale={scale:.6f}, zero_point={zero_point:.6f}, "
                    f"MSE={mse:.6f}, SNR={snr:.2f}dB")
        return calibration_results
    
    def get_parameters(self) -> Dict[str, Dict[str, float]]:
        """
        Get current quantization parameters, including per-channel parameters if applicable.
        
        Returns:
            Dictionary containing global and channel-specific parameters
        """
        with self._lock:
            params = {
                'global': {
                    'scale': self._global_scale,
                    'zero_point': self._global_zero_point
                },
                'per_channel': {}
            }
            
            # Add per-channel parameters if there are any
            if self._scales:
                channel_params = {}
                for channel_idx in sorted(self._scales.keys()):
                    channel_params[str(channel_idx)] = {
                        'scale': self._scales[channel_idx],
                        'zero_point': self._zero_points[channel_idx]
                    }
                params['per_channel'] = channel_params
        
        return params
    
    def export_parameters(self, file_path: str) -> None:
        """
        Export quantization parameters to a file for later use.
        
        Args:
            file_path: Path to save the parameters
        """
        import json
        
        params = self.get_parameters()
        with open(file_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Quantization parameters exported to {file_path}")
    
    def import_parameters(self, file_path: str) -> None:
        """
        Import quantization parameters from a file.
        
        Args:
            file_path: Path to load the parameters from
        """
        import json
        
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
            
            with self._lock:
                # Set global parameters
                if 'global' in params:
                    self._global_scale = float(params['global']['scale'])
                    self._global_zero_point = float(params['global']['zero_point'])
                
                # Set per-channel parameters
                if 'per_channel' in params:
                    for channel_idx_str, channel_params in params['per_channel'].items():
                        channel_idx = int(channel_idx_str)
                        self._scales[channel_idx] = float(channel_params['scale'])
                        self._zero_points[channel_idx] = float(channel_params['zero_point'])
            
            logger.info(f"Quantization parameters imported from {file_path}")
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.error(f"Failed to import quantization parameters: {e}")
            raise

    def transform(self, X):
        with self._error_context():
            # Create a hashable identifier for the input data
            if isinstance(X, np.ndarray):
                # Convert array to a hashable representation
                X_hash = hash(X.tobytes())  # Convert array bytes to a hash
            else:
                # For other types, try using the object directly or its hash
                try:
                    X_hash = hash(X)
                except TypeError:
                    # If input isn't hashable, generate a unique identifier
                    X_hash = id(X)  # Use object ID as fallback
                    
            return self._cached_transform(X, X_hash)

    def _cached_transform(self, X, X_hash):
        # X_hash is now already a hashable type
        if X_hash in self._transform_cache:
            return self._transform_cache[X_hash]
        
        # Perform the actual transformation
        result = self._perform_transformation(X)
        
        # Cache the result
        self._transform_cache[X_hash] = result
        return result




# Example usage:
if __name__ == "__main__":
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
    
    # Try per-channel quantization
    channel_config = QuantizationConfig(
        quantization_type=QuantizationType.INT8.value,
        quantization_mode=QuantizationMode.DYNAMIC_PER_CHANNEL.value,
        enable_cache=True,
        use_percentile=True,
        min_percentile=0.1,
        max_percentile=99.9,
    )
    channel_quantizer = Quantizer(channel_config)
    
    # Create data with different ranges per channel
    channel_data = np.zeros((5, 10), dtype=np.float32)
    for i in range(5):
        channel_data[i] = np.random.uniform(-i-1, i+1, size=(10,))
    
    # Quantize with per-channel mode
    q_channel_data = channel_quantizer.quantize(channel_data, channel_dim=0)
    deq_channel_data = channel_quantizer.dequantize(q_channel_data, channel_dim=0)
    
    # Compare error with and without per-channel quantization
    mse_channel = np.mean((channel_data - deq_channel_data) ** 2)
    
    # Quantize the same data with standard quantizer
    q_std = quantizer.quantize(channel_data)
    deq_std = quantizer.dequantize(q_std)
    mse_std = np.mean((channel_data - deq_std) ** 2)
    
    print(f"Per-channel MSE: {mse_channel:.8f}")
    print(f"Standard MSE: {mse_std:.8f}")
    print(f"Error reduction: {(1 - mse_channel/mse_std)*100:.2f}%")
    
    # Demonstrate different quantization types
    uint8_config = QuantizationConfig(
        quantization_type=QuantizationType.UINT8.value,
        quantization_mode=QuantizationMode.ASYMMETRIC.value,
    )
    uint8_quantizer = Quantizer(uint8_config)
    
    int16_config = QuantizationConfig(
        quantization_type=QuantizationType.INT16.value,
        quantization_mode=QuantizationMode.SYMMETRIC.value,
    )
    int16_quantizer = Quantizer(int16_config)
    
    # Compare different quantization types
    test_data = np.random.normal(0, 1, size=(1000,)).astype(np.float32)
    
    q_int8 = quantizer.quantize(test_data)
    q_uint8 = uint8_quantizer.quantize(test_data)
    q_int16 = int16_quantizer.quantize(test_data)
    
    deq_int8 = quantizer.dequantize(q_int8)
    deq_uint8 = uint8_quantizer.dequantize(q_uint8)
    deq_int16 = int16_quantizer.dequantize(q_int16)
    
    mse_int8 = np.mean((test_data - deq_int8) ** 2)
    mse_uint8 = np.mean((test_data - deq_uint8) ** 2)
    mse_int16 = np.mean((test_data - deq_int16) ** 2)
    
    print("\nQuantization Type Comparison:")
    print(f"INT8 MSE: {mse_int8:.10f}")
    print(f"UINT8 MSE: {mse_uint8:.10f}")
    print(f"INT16 MSE: {mse_int16:.10f}")
    
    # Simulate quantization-aware training by using quantize_dequantize
    # This helps models adapt to quantization effects during training
    qat_data = quantizer.quantize_dequantize(test_data)
    qat_error = np.mean((test_data - qat_data) ** 2)
    print(f"\nQuantize-dequantize MSE (for quantization-aware training): {qat_error:.10f}")
    
    # Export and import parameters
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        param_file = tmp.name
    
    # Export parameters
    quantizer.export_parameters(param_file)
    
    # Create a new quantizer and import the parameters
    new_quantizer = Quantizer(config)
    new_quantizer.import_parameters(param_file)
    
    # Verify the imported parameters work the same
    q_new = new_quantizer.quantize(test_data)
    deq_new = new_quantizer.dequantize(q_new)
    mse_new = np.mean((test_data - deq_new) ** 2)
    print(f"\nMSE after parameter export/import: {mse_new:.10f}")
    
    # Benchmark performance
    print("\nPerformance benchmark:")
    large_data = np.random.uniform(-10, 10, size=(10000, 100)).astype(np.float32)
    
    # Warm-up
    quantizer.reset_stats()
    _ = quantizer.quantize(large_data[:100])
    
    # Benchmark
    start_time = time.time()
    q_large = quantizer.quantize(large_data)
    quantize_time = time.time() - start_time
    
    start_time = time.time()
    deq_large = quantizer.dequantize(q_large)
    dequantize_time = time.time() - start_time
    
    print(f"Quantize {large_data.size} elements: {quantize_time*1000:.2f}ms")
    print(f"Dequantize {q_large.size} elements: {dequantize_time*1000:.2f}ms")
    print(f"Throughput: {large_data.size/quantize_time/1e6:.2f} Melem/s")
    
    # Thread-safety demonstration
    def quantize_thread_function(thread_id, data):
        print(f"Thread {thread_id} starting")
        for i in range(5):
            q = quantizer.quantize(data)
            dq = quantizer.dequantize(q)
            error = np.mean((data - dq) ** 2)
            print(f"Thread {thread_id}, iteration {i}, error: {error:.10f}")
            time.sleep(0.01)  # Small delay to allow thread interleaving
        print(f"Thread {thread_id} finished")
    
    print("\nTesting thread safety with concurrent quantization:")
    threads = []
    for i in range(3):
        thread_data = np.random.uniform(-5, 5, size=(1000, 10)).astype(np.float32)
        t = threading.Thread(target=quantize_thread_function, args=(i, thread_data))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Show final stats
    final_stats = quantizer.get_quantization_stats()
    print("\nFinal quantization statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
