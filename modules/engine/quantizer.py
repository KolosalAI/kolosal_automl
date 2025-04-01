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
    - Multiple quantization types (int8, uint8, int16, float16)
    - Various quantization modes (symmetric, asymmetric, per-channel)
    - Thread-safe operations for concurrent environments
    - Optimized cache mechanisms for frequent values
    - Statistical tracking and calibration
    - Vectorized operations with memory optimization
    - Support for mixed precision quantization
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
        
        # For caching transform results
        self._transform_cache = {}
        
        # Determine quantization type and range values
        self._setup_quantization_ranges()
        
        # Performance optimization: preallocate buffers if needed
        self._initialize_buffers()
        
        # Initialize mixed precision settings if enabled
        if self.config.enable_mixed_precision:
            self._init_mixed_precision()
        
        logger.info(f"Quantizer initialized with {self.config.quantization_type} type and "
                    f"{self.config.quantization_mode} mode")

    @staticmethod
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
        elif q_type == QuantizationType.FLOAT16.value:
            # For float16, we don't need traditional quantization ranges
            self._qmin = -65504  # Min representable normal value in float16
            self._qmax = 65504   # Max representable normal value in float16
            self._qrange = self._qmax - self._qmin
            self._qtype = np.float16
        elif q_type == QuantizationType.NONE.value:
            # For no quantization, use float32
            self._qmin = np.finfo(np.float32).min
            self._qmax = np.finfo(np.float32).max
            self._qrange = self._qmax - self._qmin
            self._qtype = np.float32
        elif q_type == QuantizationType.MIXED.value:
            # For mixed precision, default to int8 initially
            # Specific layers will use different precision as configured
            self._qmin = -(2 ** 7)
            self._qmax = 2 ** 7 - 1
            self._qrange = 2 ** 8 - 1
            self._qtype = np.int8
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

    def _init_mixed_precision(self) -> None:
        """Initialize settings for mixed precision quantization."""
        if not self.config.enable_mixed_precision:
            return
            
        # Map of layer names to their precision configurations
        self._layer_precision = {}
        
        # Process mixed precision layers config
        for layer_name in self.config.mixed_precision_layers:
            # Default to higher precision for specified layers
            self._layer_precision[layer_name] = {
                "bits": 16,
                "type": QuantizationType.INT16.value
            }
            
        # Process custom quantization config if provided
        if self.config.custom_quantization_config:
            for layer_name, layer_config in self.config.custom_quantization_config.items():
                if "bits" in layer_config:
                    # Store custom precision configuration
                    self._layer_precision[layer_name] = {
                        "bits": layer_config.get("bits", 8),
                        "type": layer_config.get("type", QuantizationType.INT8.value)
                    }
        
        logger.debug(f"Mixed precision settings initialized for {len(self._layer_precision)} layers")

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
            old_mixed = self.config.enable_mixed_precision
            
            self.config = new_config
            self._cache_enabled = new_config.enable_cache
            self._cache_size = new_config.cache_size
            
            # If quantization type or bit width changed, update ranges
            if (old_type != new_config.quantization_type or 
                old_bits != new_config.num_bits):
                self._setup_quantization_ranges()
                # Reinitialize buffers if needed
                self._initialize_buffers()
            
            # Check if mixed precision settings need update
            if (not old_mixed and new_config.enable_mixed_precision) or (
                old_mixed and new_config.enable_mixed_precision and 
                new_config.mixed_precision_layers):
                self._init_mixed_precision()
            
            # Clear caches
            with self._cache_lock:
                self._value_cache.clear()
                self._transform_cache.clear()
            
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
        channel_idx: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Compute optimal scale and zero_point based on current quantization mode.

        Args:
            data: Input float32 array
            channel_idx: Optional channel index for per-channel quantization
            layer_name: Optional layer name for mixed precision quantization

        Returns:
            Tuple of (scale, zero_point)
        """
        # Check if data is empty
        if data.size == 0:
            logger.warning("Cannot compute scale and zero point for empty data")
            return 1.0, 0.0
        
        # Early exit if not using dynamic range
        if self.config.quantization_mode not in [
            QuantizationMode.DYNAMIC.value,
            QuantizationMode.DYNAMIC_PER_BATCH.value,
            QuantizationMode.DYNAMIC_PER_CHANNEL.value
        ]:
            with self._lock:
                return self._global_scale, self._global_zero_point

        # Handle mixed precision if specified
        qtype = self.config.quantization_type
        qmin = self._qmin
        qmax = self._qmax
        qrange = self._qrange
        
        if self.config.enable_mixed_precision and layer_name and layer_name in self._layer_precision:
            layer_config = self._layer_precision[layer_name]
            bits = layer_config["bits"]
            
            # Adjust range values for this layer
            if layer_config["type"] == QuantizationType.INT8.value:
                qmin = -(2 ** (bits - 1))
                qmax = 2 ** (bits - 1) - 1
            elif layer_config["type"] == QuantizationType.UINT8.value:
                qmin = 0
                qmax = 2 ** bits - 1
            elif layer_config["type"] == QuantizationType.INT16.value:
                qmin = -(2 ** (bits - 1))
                qmax = 2 ** (bits - 1) - 1
            
            qrange = qmax - qmin + 1
        
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
        
        # Determine if we should use symmetric quantization
        use_symmetric = self.config.symmetric
        if self.config.quantization_mode in [QuantizationMode.SYMMETRIC.value]:
            use_symmetric = True
        
        if use_symmetric:
            # Symmetric quantization centers around zero
            abs_max = max(abs(data_min), abs(data_max))
            # Add a small epsilon to improve accuracy
            abs_max = abs_max * 1.01  # Add 1% margin to reduce clipping
            scale = abs_max / max(abs(qmin), abs(qmax))
            zero_point = 0.0
        else:
            # Asymmetric quantization uses full range
            # Add a small margin to improve accuracy
            data_range = data_range * 1.01  # Add 1% margin
            scale = data_range / qrange
            
            if qtype == QuantizationType.UINT8.value:
                # For uint8, calculate zero point directly
                zero_point_uint8 = round(qmin - data_min / scale)
                zero_point_uint8 = max(qmin, min(qmax, zero_point_uint8))
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

    def get_layer_quantization_params(self, layer_name: str) -> Dict[str, Any]:
        """
        Get quantization parameters for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Dictionary with quantization parameters for the layer
        """
        with self._lock:
            if self.config.enable_mixed_precision and layer_name in self._layer_precision:
                layer_config = self._layer_precision[layer_name]
                return {
                    "bits": layer_config["bits"],
                    "type": layer_config["type"],
                    "skip": layer_name in self.config.skip_layers
                }
            
            # Default parameters based on global config
            return {
                "bits": self.config.num_bits,
                "type": self.config.quantization_type,
                "skip": layer_name in self.config.skip_layers
            }

    def should_quantize_layer(self, layer_name: str) -> bool:
        """
        Determine if a layer should be quantized based on configuration.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            True if the layer should be quantized, False otherwise
        """
        if self.config.quantization_type == QuantizationType.NONE.value:
            return False
            
        # Check if layer is in skip list
        if layer_name in self.config.skip_layers:
            return False
        
        # Check for bias quantization setting
        if "bias" in layer_name.lower() and not self.config.quantize_bias:
            return False
            
        # Check for weights-only setting
        if "activation" in layer_name.lower() and self.config.quantize_weights_only:
            return False
            
        return True

    def quantize(
        self, 
        data: Union[NDArray[np.float32], Any], 
        validate: bool = True,
        channel_dim: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> NDArray:
        """
        High-performance vectorized quantization with multiple modes.

        Args:
            data: Input float32 array
            validate: If True, performs input validation
            channel_dim: Dimension index for per-channel quantization (if enabled)
            layer_name: Optional layer name for mixed precision handling

        Returns:
            Quantized array of the configured type
        """
        start_time = time.time()
        
        # Fast path for no quantization
        if self.config.quantization_type == QuantizationType.NONE.value:
            return data
            
        # Check if layer should be skipped
        if layer_name is not None and not self.should_quantize_layer(layer_name):
            return data
        
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
        
        # Handle mixed precision if specified
        if self.config.enable_mixed_precision and layer_name and layer_name in self._layer_precision:
            return self._quantize_mixed_precision(input_data, layer_name, channel_dim)
        
        # Handle float16 quantization separately (simple cast)
        if self.config.quantization_type == QuantizationType.FLOAT16.value:
            result = input_data.astype(np.float16)
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            with self._stats_lock:
                self._quantization_stats['processing_time_ms'] += processing_time
                
            return result
        
        # Handle per-channel quantization if requested
        if self.config.per_channel or (channel_dim is not None and 
                                      self.config.quantization_mode == QuantizationMode.DYNAMIC_PER_CHANNEL.value):
            result = self._quantize_per_channel(input_data, channel_dim)
        else:
            # Standard quantization path
            # Update scale and zero_point if using dynamic mode
            if self.config.quantization_mode in [
                QuantizationMode.DYNAMIC.value,
                QuantizationMode.DYNAMIC_PER_BATCH.value
            ]:
                self.compute_scale_and_zero_point(input_data, layer_name=layer_name)
            
            # Get current scale and zero point (thread-safe)
            with self._lock:
                scale = self._global_scale
                zero_point = self._global_zero_point
            
            # Implement quantization based on quantization type
            if self._qtype == np.int8:
                # INT8 quantization formula
                if self.config.symmetric or self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
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
                if self.config.symmetric or self.config.quantization_mode == QuantizationMode.SYMMETRIC.value:
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
        
        # Apply requantization if enabled
        if self.config.enable_requantization:
            result = self._apply_requantization(result, data)
        
        # Record processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        with self._stats_lock:
            self._quantization_stats['processing_time_ms'] += processing_time
            self._quantization_stats['dequantize_calls'] += 1
        
        return result

    def _quantize_mixed_precision(
        self,
        data: NDArray[np.float32],
        layer_name: str,
        channel_dim: Optional[int] = None
    ) -> NDArray:
        """
        Quantize data with mixed precision for a specific layer.
        
        Args:
            data: Input float32 array
            layer_name: Name of the layer
            channel_dim: Optional dimension for per-channel quantization
            
        Returns:
            Quantized array with appropriate precision
        """
        layer_config = self._layer_precision[layer_name]
        bits = layer_config["bits"]
        qtype = layer_config["type"]
        
        # Determine quantization type and range for this layer
        if qtype == QuantizationType.INT8.value:
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
            qrange = 2 ** bits - 1
            nptype = np.int8
        elif qtype == QuantizationType.UINT8.value:
            qmin = 0
            qmax = 2 ** bits - 1
            qrange = 2 ** bits - 1
            nptype = np.uint8
        elif qtype == QuantizationType.INT16.value:
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
            qrange = 2 ** bits - 1
            nptype = np.int16
        elif qtype == QuantizationType.FLOAT16.value:
            # Simple cast for float16
            return data.astype(np.float16)
        else:
            # Fallback to default
            qmin = self._qmin
            qmax = self._qmax
            qrange = self._qrange
            nptype = self._qtype
        
        # Handle per-channel quantization if requested
        if channel_dim is not None and self.config.per_channel:
            # Store original types
            orig_qmin, orig_qmax, orig_qrange, orig_qtype = self._qmin, self._qmax, self._qrange, self._qtype
            
            # Temporarily update types for this layer
            self._qmin, self._qmax, self._qrange, self._qtype = qmin, qmax, qrange, nptype
            
            # Quantize with per-channel approach
            result = self._quantize_per_channel(data, channel_dim)
            
            # Restore original types
            self._qmin, self._qmax, self._qrange, self._qtype = orig_qmin, orig_qmax, orig_qrange, orig_qtype
            
            return result
        
        # Standard quantization for this precision
        # Update scale and zero_point if using dynamic mode
        if self.config.quantization_mode in [
            QuantizationMode.DYNAMIC.value,
            QuantizationMode.DYNAMIC_PER_BATCH.value
        ]:
            # Calculate scale and zero point based on data range
            data_min = float(np.min(data))
            data_max = float(np.max(data))
            data_range = max(data_max - data_min, 1e-7)
            
            if self.config.symmetric:
                # Symmetric quantization
                abs_max = max(abs(data_min), abs(data_max)) * 1.01
                scale = abs_max / max(abs(qmin), abs(qmax))
                zero_point = 0.0
            else:
                # Asymmetric quantization
                data_range = data_range * 1.01
                scale = data_range / qrange
                
                if qtype == QuantizationType.UINT8.value:
                    zero_point_uint8 = round(qmin - data_min / scale)
                    zero_point_uint8 = max(qmin, min(qmax, zero_point_uint8))
                    zero_point = data_min
                else:
                    zero_point = round(data_min / scale) * scale
        else:
            # Use global scale and zero point
            with self._lock:
                scale = self._global_scale
                zero_point = self._global_zero_point
        
        # Perform quantization with the determined parameters
        if qtype == QuantizationType.INT8.value or qtype == QuantizationType.INT16.value:
            if self.config.symmetric:
                quantized = np.clip(np.round(data / scale), qmin, qmax)
            else:
                quantized = np.clip(np.round((data - zero_point) / scale), qmin, qmax)
        elif qtype == QuantizationType.UINT8.value:
            zero_point_uint8 = int(round(qmin - zero_point / scale))
            zero_point_uint8 = max(qmin, min(qmax, zero_point_uint8))
            quantized = np.clip(np.round(data / scale) + zero_point_uint8, qmin, qmax)
        
        # Convert to target type
        return quantized.astype(nptype)