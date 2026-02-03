import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict, Union, Any, List
from dataclasses import asdict
import logging
import threading
import time
import collections
import hashlib

# Set up centralized logging
try:
    from modules.logging_config import get_logger
    logger = get_logger(
        name="quantizer",
        level=logging.INFO,
        log_file="quantizer.log",
        enable_console=True
    )
except ImportError:
    # Fallback to basic logging if centralized logging not available
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)

from ..configs import QuantizationConfig, QuantizationType, QuantizationMode


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
        
        # Parameters for calibration
        self._calibration_data = []
        self._is_calibrated = False
        
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
        if q_type == QuantizationType.INT8:
            self._qmin = -(2 ** (num_bits - 1))
            self._qmax = 2 ** (num_bits - 1) - 1
            self._qrange = 2 ** num_bits
            self._qtype = np.int8
        elif q_type == QuantizationType.UINT8:
            self._qmin = 0
            self._qmax = 2 ** num_bits - 1
            self._qrange = 2 ** num_bits
            self._qtype = np.uint8
        elif q_type == QuantizationType.INT16:
            self._qmin = -(2 ** (num_bits - 1))
            self._qmax = 2 ** (num_bits - 1) - 1
            self._qrange = 2 ** num_bits
            self._qtype = np.int16
        elif q_type == QuantizationType.FLOAT16:
            # For float16, we don't need traditional quantization ranges
            self._qmin = -65504  # Min representable normal value in float16
            self._qmax = 65504   # Max representable normal value in float16
            self._qrange = self._qmax - self._qmin
            self._qtype = np.float16
        elif q_type == QuantizationType.NONE:
            # For no quantization, use float32
            self._qmin = np.finfo(np.float32).min
            self._qmax = np.finfo(np.float32).max
            self._qrange = self._qmax - self._qmin
            self._qtype = np.float32
        elif q_type == QuantizationType.MIXED:
            # For mixed precision, default to int8 initially
            # Specific layers will use different precision as configured
            self._qmin = -(2 ** 7)
            self._qmax = 2 ** 7 - 1
            self._qrange = 2 ** 8
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
        if hasattr(self.config, 'mixed_precision_layers') and self.config.mixed_precision_layers:
            for layer_name in self.config.mixed_precision_layers:
                # Default to higher precision for specified layers
                self._layer_precision[layer_name] = {
                    "bits": 16,
                    "type": QuantizationType.INT16
                }
            
        # Process custom quantization config if provided
        if hasattr(self.config, 'custom_quantization_config') and self.config.custom_quantization_config:
            for layer_name, layer_config in self.config.custom_quantization_config.items():
                if "bits" in layer_config:
                    # Store custom precision configuration
                    self._layer_precision[layer_name] = {
                        "bits": layer_config.get("bits", 8),
                        "type": layer_config.get("type", QuantizationType.INT8)
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
            if ((not old_mixed and new_config.enable_mixed_precision) or 
                (old_mixed and new_config.enable_mixed_precision and 
                 hasattr(new_config, 'mixed_precision_layers') and 
                 new_config.mixed_precision_layers)):
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
            return self.config.to_dict()
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve quantization statistics.
        
        Returns:
            Dictionary with quantization statistics
        """
        with self._stats_lock:
            return self._quantization_stats.copy()
            
    def clear_cache(self) -> None:
        """Clear the dequantization cache."""
        with self._cache_lock:
            self._value_cache.clear()
            self._transform_cache.clear()
            logger.debug("Quantizer cache cleared")
            
    def export_import_parameters(self) -> Dict[str, Any]:
        """
        Export quantization parameters for saving/loading.
        
        Returns:
            Dictionary with quantization parameters
        """
        with self._lock:
            params = {
                "global_scale": self._global_scale,
                "global_zero_point": self._global_zero_point,
                "per_channel_scales": self._scales,
                "per_channel_zero_points": self._zero_points,
                "quantization_type": self.config.quantization_type,
                "quantization_mode": self.config.quantization_mode,
                "is_calibrated": self._is_calibrated,
                "qmin": self._qmin,
                "qmax": self._qmax,
            }
            return params
            
    def load_parameters(self, params: Dict[str, Any]) -> None:
        """
        Load quantization parameters from dictionary.
        
        Args:
            params: Dictionary with parameters from export_import_parameters
        """
        with self._lock:
            self._global_scale = params.get("global_scale", 1.0)
            self._global_zero_point = params.get("global_zero_point", 0.0)
            self._scales = params.get("per_channel_scales", {})
            self._zero_points = params.get("per_channel_zero_points", {})
            self._is_calibrated = params.get("is_calibrated", False)
            
            # Only update these if they match current configuration
            if params.get("quantization_type") == self.config.quantization_type:
                self._qmin = params.get("qmin", self._qmin)
                self._qmax = params.get("qmax", self._qmax)
                
        logger.info(f"Loaded quantization parameters: scale={self._global_scale}, "
                   f"zero_point={self._global_zero_point}, calibrated={self._is_calibrated}")
    
    def calibrate(self, calibration_data: List[NDArray[np.float32]]) -> None:
        """
        Calibrate the quantizer using representative data.
        
        Args:
            calibration_data: List of numpy arrays with representative data
        """
        if self.config.quantization_mode != QuantizationMode.CALIBRATED:
            logger.warning("Calibration is only used with CALIBRATED quantization mode. Setting mode to CALIBRATED.")
            # Update mode to CALIBRATED
            self.config.quantization_mode = QuantizationMode.CALIBRATED
            
        # Store calibration data
        self._calibration_data = calibration_data
        
        # Skip if no data provided
        if not calibration_data or len(calibration_data) == 0:
            logger.warning("No calibration data provided")
            return
            
        # Combine all calibration data
        combined_data = np.concatenate([array.flatten() for array in calibration_data])
        
        # Compute overall statistics
        if self.config.use_percentile:
            # Use percentiles to find robust min/max
            data_min = float(np.percentile(combined_data, self.config.min_percentile))
            data_max = float(np.percentile(combined_data, self.config.max_percentile))
        else:
            data_min = float(np.min(combined_data))
            data_max = float(np.max(combined_data))
            
        # Compute scale and zero point based on determined range
        data_range = max(data_max - data_min, 1e-7)
        
        if self.config.symmetric:
            # Symmetric calibration
            abs_max = max(abs(data_min), abs(data_max))
            abs_max = abs_max * 1.01  # Add margin
            
            with self._lock:
                self._global_scale = abs_max / max(abs(self._qmin), abs(self._qmax))
                self._global_zero_point = 0.0
        else:
            # Asymmetric calibration
            data_range = data_range * 1.01  # Add margin
            
            with self._lock:
                self._global_scale = data_range / self._qrange
                
                if self.config.quantization_type == QuantizationType.UINT8:
                    zero_point_uint8 = int(round(self._qmin - data_min / self._global_scale))
                    zero_point_uint8 = max(self._qmin, min(self._qmax, zero_point_uint8))
                    self._global_zero_point = float(zero_point_uint8)
                else:
                    # For int types, align zero point to quantization grid
                    if self.config.quantization_type in [QuantizationType.INT8, QuantizationType.INT16]:
                        self._global_zero_point = round(data_min / self._global_scale) * self._global_scale
                    else:
                        self._global_zero_point = data_min
                    
        # Mark as calibrated
        self._is_calibrated = True
        logger.info(f"Quantizer calibrated with {len(calibration_data)} data samples. "
                f"Scale={self._global_scale}, zero_point={self._global_zero_point}")
            
    def reset_stats(self) -> None:
        """Reset all quantization statistics to initial values."""
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

    def _validate_input(self, data: NDArray) -> None:
        """
        Validate input data for quantization.
        
        Args:
            data: Input array to validate
        
        Raises:
            ValueError: If input data contains NaN or Inf values and config disallows them
        """
        # Check for NaN values if configured to error
        if self.config.error_on_nan and np.isnan(data).any():
            raise ValueError("Input data contains NaN values")
        
        # Check for Inf values if configured to error
        if self.config.error_on_inf and np.isinf(data).any():
            raise ValueError("Input data contains Inf values")

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
            QuantizationMode.DYNAMIC,
            QuantizationMode.DYNAMIC_PER_BATCH,
            QuantizationMode.DYNAMIC_PER_CHANNEL
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
            if layer_config["type"] == QuantizationType.INT8:
                qmin = -(2 ** (bits - 1))
                qmax = 2 ** (bits - 1) - 1
            elif layer_config["type"] == QuantizationType.UINT8:
                qmin = 0
                qmax = 2 ** bits - 1
            elif layer_config["type"] == QuantizationType.INT16:
                qmin = -(2 ** (bits - 1))
                qmax = 2 ** (bits - 1) - 1
            
            qrange = qmax - qmin + 1
        
        # Handle outliers if threshold is set
        if hasattr(self.config, 'outlier_threshold') and self.config.outlier_threshold is not None:
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
        if hasattr(self.config, 'use_percentile') and self.config.use_percentile:
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
        if self.config.quantization_mode == QuantizationMode.SYMMETRIC:
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
            
            if qtype == QuantizationType.UINT8:
                # For uint8, calculate zero point directly
                zero_point_uint8 = round(qmin - data_min / scale)
                zero_point_uint8 = max(qmin, min(qmax, zero_point_uint8))
                zero_point = float(zero_point_uint8)
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
                    "skip": layer_name in self.config.skip_layers if hasattr(self.config, 'skip_layers') else False
                }
            
            # Default parameters based on global config
            return {
                "bits": self.config.num_bits,
                "type": self.config.quantization_type,
                "skip": layer_name in self.config.skip_layers if hasattr(self.config, 'skip_layers') else False
            }

    def should_quantize_layer(self, layer_name: str) -> bool:
        """
        Determine if a layer should be quantized based on configuration.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            True if the layer should be quantized, False otherwise
        """
        if self.config.quantization_type == QuantizationType.NONE:
            return False
            
        # Check if layer is in skip list
        if hasattr(self.config, 'skip_layers') and self.config.skip_layers and layer_name in self.config.skip_layers:
            return False
        
        # Check for bias quantization setting
        if "bias" in layer_name.lower() and hasattr(self.config, 'quantize_bias') and not self.config.quantize_bias:
            return False
            
        # Check for weights-only setting
        if "activation" in layer_name.lower() and hasattr(self.config, 'quantize_weights_only') and self.config.quantize_weights_only:
            return False
            
        return True
        
    def _apply_requantization(self, quantized_data: NDArray, original_data: NDArray) -> NDArray:
        """
        Apply requantization to reduce quantization error.
        
        Args:
            quantized_data: Already quantized data
            original_data: Original unquantized data for reference
            
        Returns:
            Requantized data with reduced error
        """
        if not hasattr(self.config, 'enable_requantization') or not self.config.enable_requantization:
            return quantized_data
            
        # Dequantize the quantized data
        dequantized = self.dequantize(quantized_data)
        
        # Calculate the quantization error
        error = original_data - dequantized
        
        # Check if error is significant enough for requantization
        max_error = np.max(np.abs(error))
        
        if not hasattr(self.config, 'requantization_threshold'):
            threshold = 0.01  # Default threshold
        else:
            threshold = self.config.requantization_threshold
        
        if max_error <= threshold:
            return quantized_data
            
        # Requantize with error feedback
        corrected_data = original_data + error * 0.5
        
        # Quantize the corrected data
        requantized = self.quantize(corrected_data, validate=False)
        
        return requantized

    def _quantize_mixed_precision(
        self, 
        data: NDArray[np.float32], 
        layer_name: str,
        channel_dim: Optional[int] = None
    ) -> NDArray:
        """
        Perform mixed precision quantization for a specific layer.
        
        Args:
            data: Input float32 array
            layer_name: Name of the layer for precision lookup
            channel_dim: Optional dimension for per-channel quantization
            
        Returns:
            Quantized array with the precision specified for this layer
        """
        # Get layer-specific precision settings
        layer_config = self._layer_precision[layer_name]
        
        # Temporarily adjust quantization parameters
        original_type = self.config.quantization_type
        original_bits = self.config.num_bits
        
        # Set layer-specific parameters
        self.config.quantization_type = layer_config["type"]
        self.config.num_bits = layer_config["bits"]
        
        # Re-setup quantization ranges
        old_qtype = self._qtype
        old_qmin = self._qmin
        old_qmax = self._qmax
        old_qrange = self._qrange
        
        # Set up new ranges for this layer
        self._setup_quantization_ranges()
        
        # Perform quantization with new parameters
        result = self.quantize(data, validate=False, channel_dim=channel_dim)
        
        # Restore original parameters
        self.config.quantization_type = original_type
        self.config.num_bits = original_bits
        self._qtype = old_qtype
        self._qmin = old_qmin
        self._qmax = old_qmax
        self._qrange = old_qrange
        
        return result

    def _dequantize_mixed_precision(
        self, 
        data: NDArray, 
        layer_name: str,
        channel_dim: Optional[int] = None
    ) -> NDArray[np.float32]:
        """
        Perform mixed precision dequantization for a specific layer.
        
        Args:
            data: Quantized input array
            layer_name: Name of the layer for precision lookup
            channel_dim: Optional dimension for per-channel dequantization
            
        Returns:
            Dequantized array as float32
        """
        # Get layer-specific precision settings
        layer_config = self._layer_precision[layer_name]
        
        # Temporarily adjust quantization parameters
        original_type = self.config.quantization_type
        original_bits = self.config.num_bits
        
        # Set layer-specific parameters
        self.config.quantization_type = layer_config["type"]
        self.config.num_bits = layer_config["bits"]
        
        # Re-setup quantization ranges
        old_qtype = self._qtype
        old_qmin = self._qmin
        old_qmax = self._qmax
        old_qrange = self._qrange
        
        # Set up new ranges for this layer
        self._setup_quantization_ranges()
        
        # Perform dequantization with new parameters
        result = self.dequantize(data, channel_dim=channel_dim)
        
        # Restore original parameters
        self.config.quantization_type = original_type
        self.config.num_bits = original_bits
        self._qtype = old_qtype
        self._qmin = old_qmin
        self._qmax = old_qmax
        self._qrange = old_qrange
        
        return result

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
        if self.config.quantization_type == QuantizationType.NONE:
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
        if hasattr(self.config, 'optimize_memory') and self.config.optimize_memory and data.flags.c_contiguous:
            input_data = data
        else:
            # Create a contiguous copy for better performance
            input_data = np.ascontiguousarray(data)
        
        # Handle mixed precision if specified
        if self.config.enable_mixed_precision and layer_name and layer_name in self._layer_precision:
            return self._quantize_mixed_precision(input_data, layer_name, channel_dim)
        
        # Handle float16 quantization separately (simple cast)
        if self.config.quantization_type == QuantizationType.FLOAT16:
            result = input_data.astype(np.float16)
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            with self._stats_lock:
                self._quantization_stats['processing_time_ms'] += processing_time
                
            return result
        
        # Handle per-channel quantization if requested
        if hasattr(self.config, 'per_channel') and self.config.per_channel or (channel_dim is not None and 
                                      self.config.quantization_mode == QuantizationMode.DYNAMIC_PER_CHANNEL):
            result = self._quantize_per_channel(input_data, channel_dim)
        else:
            # Standard quantization path
            # Update scale and zero_point if using dynamic mode
            if self.config.quantization_mode in [
                QuantizationMode.DYNAMIC,
                QuantizationMode.DYNAMIC_PER_BATCH,
                QuantizationMode.DYNAMIC_PER_CHANNEL
            ]:
                self.compute_scale_and_zero_point(input_data, layer_name=layer_name)
            
            # Get current scale and zero point (thread-safe)
            with self._lock:
                scale = self._global_scale
                zero_point = self._global_zero_point
            
            # Implement quantization based on quantization type
            if self._qtype == np.int8:
                # INT8 quantization formula
                if self.config.symmetric or self.config.quantization_mode == QuantizationMode.SYMMETRIC:
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
                if self.config.symmetric or self.config.quantization_mode == QuantizationMode.SYMMETRIC:
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
        if hasattr(self.config, 'enable_requantization') and self.config.enable_requantization:
            result = self._apply_requantization(result, data)
        
        # Record processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        with self._stats_lock:
            self._quantization_stats['processing_time_ms'] += processing_time
        
        return result

    def dequantize(
        self, 
        data: NDArray, 
        channel_dim: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> NDArray[np.float32]:
        """
        Dequantize data from quantized format back to float32.
        
        Args:
            data: Quantized input array
            channel_dim: Dimension index for per-channel dequantization
            layer_name: Optional layer name for mixed precision handling
            
        Returns:
            Dequantized array as float32
        """
        start_time = time.time()
        
        # Record stats
        with self._stats_lock:
            self._quantization_stats['dequantize_calls'] += 1
            self._quantization_stats['total_values'] += data.size
        
        # Fast path for no quantization or float16
        if self.config.quantization_type == QuantizationType.NONE:
            return data
        
        if self.config.quantization_type == QuantizationType.FLOAT16:
            return data.astype(np.float32)
        
        # Check cache if enabled
        if self._cache_enabled:
            cache_key = self.hash_array(data)
            with self._cache_lock:
                if cache_key in self._value_cache:
                    # Cache hit
                    with self._stats_lock:
                        self._quantization_stats['cache_hits'] += 1
                    # Move to end for LRU behavior
                    self._value_cache.move_to_end(cache_key)
                    return self._value_cache[cache_key]
                else:
                    # Cache miss
                    with self._stats_lock:
                        self._quantization_stats['cache_misses'] += 1
        
        # Handle mixed precision if specified
        if self.config.enable_mixed_precision and layer_name and layer_name in self._layer_precision:
            result = self._dequantize_mixed_precision(data, layer_name, channel_dim)
        elif hasattr(self.config, 'per_channel') and self.config.per_channel or (channel_dim is not None and 
                                            self.config.quantization_mode == QuantizationMode.DYNAMIC_PER_CHANNEL):
            # Handle per-channel dequantization
            result = self._dequantize_per_channel(data, channel_dim)
        else:
            # Standard dequantization
            with self._lock:
                scale = self._global_scale
                zero_point = self._global_zero_point
                
            # Convert based on quantization type
            if self._qtype == np.int8:
                if self.config.symmetric or self.config.quantization_mode == QuantizationMode.SYMMETRIC:
                    # Symmetric dequantization
                    result = data.astype(np.float32) * scale
                else:
                    # Asymmetric dequantization
                    result = data.astype(np.float32) * scale + zero_point
                    
            elif self._qtype == np.uint8:
                # UINT8 dequantization (always asymmetric)
                zero_point_uint8 = int(round(self._qmin - zero_point / scale))
                result = (data.astype(np.float32) - zero_point_uint8) * scale
                
            elif self._qtype == np.int16:
                if self.config.symmetric or self.config.quantization_mode == QuantizationMode.SYMMETRIC:
                    result = data.astype(np.float32) * scale
                else:
                    result = data.astype(np.float32) * scale + zero_point
            else:
                # Fallback for other types
                result = data.astype(np.float32)
        
        # Update cache if enabled
        if self._cache_enabled:
            with self._cache_lock:
                self._value_cache[cache_key] = result
                # Implement LRU cache eviction policy
                if len(self._value_cache) > self._cache_size:
                    self._value_cache.popitem(last=False)  # Remove oldest item
        
        # Record processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        with self._stats_lock:
            self._quantization_stats['processing_time_ms'] += processing_time
            
        return result

    def _dequantize_per_channel(self, data: NDArray, channel_dim: Optional[int] = None) -> NDArray[np.float32]:
        """
        Perform per-channel dequantization.
        
        Args:
            data: Quantized input array
            channel_dim: Dimension along which to perform per-channel dequantization
                        (if None, uses the first dimension)
                        
        Returns:
            Dequantized array with per-channel scale factors
        """
        if channel_dim is None:
            channel_dim = 0
            
        # Get the number of channels
        num_channels = data.shape[channel_dim]
        
        # Initialize result array with same shape as input but float32 type
        result = np.zeros_like(data, dtype=np.float32)
        
        # For each channel, retrieve scale/zero_point and dequantize
        for c in range(num_channels):
            # Extract the channel data
            if channel_dim == 0:
                channel_data = data[c]
            elif channel_dim == 1:
                channel_data = data[:, c]
            elif channel_dim == 2:
                channel_data = data[:, :, c]
            elif channel_dim == 3:
                channel_data = data[:, :, :, c]
            else:
                # General case for any dimension
                idx = [slice(None)] * data.ndim
                idx[channel_dim] = c
                channel_data = data[tuple(idx)]
                
            # Get scale and zero point for this channel
            with self._lock:
                scale = self._scales.get(c, self._global_scale)
                zero_point = self._zero_points.get(c, self._global_zero_point)
            
            # Dequantize the channel data
            if self._qtype == np.int8 or self._qtype == np.int16:
                if self.config.symmetric or self.config.quantization_mode == QuantizationMode.SYMMETRIC:
                    dequantized_channel = channel_data.astype(np.float32) * scale
                else:
                    dequantized_channel = channel_data.astype(np.float32) * scale + zero_point
            elif self._qtype == np.uint8:
                zero_point_uint8 = int(round(self._qmin - zero_point / scale))
                dequantized_channel = (channel_data.astype(np.float32) - zero_point_uint8) * scale
            else:
                # Fallback for other types
                dequantized_channel = channel_data.astype(np.float32)
                
            # Place the dequantized data back into the result array
            if channel_dim == 0:
                result[c] = dequantized_channel
            elif channel_dim == 1:
                result[:, c] = dequantized_channel
            elif channel_dim == 2:
                result[:, :, c] = dequantized_channel
            elif channel_dim == 3:
                result[:, :, :, c] = dequantized_channel
            else:
                # General case for any dimension
                idx = [slice(None)] * data.ndim
                idx[channel_dim] = c
                result[tuple(idx)] = dequantized_channel
                
        return result

    def quantize_dequantize(
        self, 
        data: NDArray[np.float32],
        channel_dim: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> NDArray[np.float32]:
        """
        Quantize and immediately dequantize data to simulate quantization effects.
        
        Args:
            data: Float32 input array
            channel_dim: Dimension index for per-channel quantization
            layer_name: Optional layer name for mixed precision handling
            
        Returns:
            Data after quantization and dequantization
        """
        # First quantize
        quantized = self.quantize(data, channel_dim=channel_dim, layer_name=layer_name)
        
        # Then dequantize
        return self.dequantize(quantized, channel_dim=channel_dim, layer_name=layer_name)

    def _quantize_per_channel(self, data: NDArray[np.float32], channel_dim: Optional[int] = None) -> NDArray:
        """
        Perform per-channel quantization.
        
        Args:
            data: Input float32 array
            channel_dim: Dimension along which to perform per-channel quantization
                        (if None, uses the first dimension)
                        
        Returns:
            Quantized array with per-channel scale factors
        """
        if channel_dim is None:
            channel_dim = 0
            
        # Get the number of channels
        num_channels = data.shape[channel_dim]
        
        # Initialize result array with same shape as input
        result = np.zeros_like(data, dtype=self._qtype)
        
        # For each channel, compute scale/zero_point and quantize
        for c in range(num_channels):
            # Extract the channel data
            if channel_dim == 0:
                channel_data = data[c]
            elif channel_dim == 1:
                channel_data = data[:, c]
            elif channel_dim == 2:
                channel_data = data[:, :, c]
            elif channel_dim == 3:
                channel_data = data[:, :, :, c]
            else:
                # General case for any dimension
                idx = [slice(None)] * data.ndim
                idx[channel_dim] = c
                channel_data = data[tuple(idx)]
            
            # Compute scale and zero point for this channel
            scale, zero_point = self.compute_scale_and_zero_point(channel_data, channel_idx=c)
            
            # Quantize the channel data
            if self._qtype == np.int8 or self._qtype == np.int16:
                if self.config.symmetric or self.config.quantization_mode == QuantizationMode.SYMMETRIC:
                    quantized_channel = np.clip(np.round(channel_data / scale), self._qmin, self._qmax)
                else:
                    quantized_channel = np.clip(np.round((channel_data - zero_point) / scale), self._qmin, self._qmax)
            elif self._qtype == np.uint8:
                zero_point_uint8 = int(round(self._qmin - zero_point / scale))
                zero_point_uint8 = max(self._qmin, min(self._qmax, zero_point_uint8))
                quantized_channel = np.clip(np.round(channel_data / scale) + zero_point_uint8, self._qmin, self._qmax)
            else:
                # Fallback for other types
                quantized_channel = channel_data
                
            # Convert to target type
            quantized_channel = quantized_channel.astype(self._qtype)
            
            # Place the quantized data back into the result array
            if channel_dim == 0:
                result[c] = quantized_channel
            elif channel_dim == 1:
                result[:, c] = quantized_channel
            elif channel_dim == 2:
                result[:, :, c] = quantized_channel
            elif channel_dim == 3:
                result[:, :, :, c] = quantized_channel
            else:
                # General case for any dimension
                idx = [slice(None)] * data.ndim
                idx[channel_dim] = c
                result[tuple(idx)] = quantized_channel
                
        return result