import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from functools import lru_cache
import warnings
from modules.configs import QuantizationConfig
from dataclasses import asdict
import logging

# Set up logging (consider moving this to a central logging configuration)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Quantizer:
    """
    Optimized data quantizer for int8 operations with improved performance and memory efficiency.
    Supports vectorized operations, caching, and optional error checking.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize quantizer with configurable parameters.

        Args:
            config: Optional QuantizerConfig object. If not provided, defaults are used.
        """
        self.config = config if config is not None else QuantizationConfig()
        self._scale: float = 1.0
        self._zero_point: float = 0.0
        self._dynamic_range = self.config.dynamic_range
        self._cache_size = self.config.cache_size

        # Pre-compute common constants from configuration
        self._INT8_MIN = self.config.int8_min
        self._INT8_MAX = self.config.int8_max
        self._UINT8_RANGE = self.config.uint8_range

        # Statistics for monitoring
        self._quantization_stats = {
            'clipped_values': 0,
            'total_values': 0
        }

        # Create a lock for thread-safe updates to stats (if needed in a multithreaded context)
        self._stats_lock = threading.Lock()

        # Pre-allocate memory for quantized data (potential optimization for fixed-size quantization)
        self._preallocated_int8: Optional[NDArray[np.int8]] = None  # Initialize as None

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def zero_point(self) -> float:
        return self._zero_point

    def update_config(self, new_config: QuantizationConfig) -> None:
        """
        Update the quantizer configuration.

        Args:
            new_config: A new QuantizerConfig object to update parameters.
        """
        self.config = new_config
        self._dynamic_range = new_config.dynamic_range
        self._cache_size = new_config.cache_size
        self._INT8_MIN = new_config.int8_min
        self._INT8_MAX = new_config.int8_max
        self._UINT8_RANGE = new_config.uint8_range
        # Reset pre-allocated memory if configuration changes significantly
        self._preallocated_int8 = None
        # Clear the cache when config changes
        self._cached_dequantize.cache_clear()


    def get_config(self) -> dict:
        """
        Retrieve the current configuration as a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self.config)

    def compute_scale_and_zero_point(self, data: NDArray[np.float32]) -> Tuple[float, float]:
        """
        Compute optimal scale and zero_point using improved statistical methods.

        Args:
            data: Input float32 array

        Returns:
            Tuple of (scale, zero_point)
        """
        if not self._dynamic_range:
            return self._scale, self._zero_point

        # Use robust statistics to handle outliers
        percentile_min = np.percentile(data, 1)
        percentile_max = np.percentile(data, 99)

        # Ensure minimum range to avoid division by zero
        value_range = max(percentile_max - percentile_min, 1e-8)

        # Optimize scale computation
        self._scale = value_range / self._UINT8_RANGE
        self._zero_point = percentile_min

        return self._scale, self._zero_point

    def quantize(self, data: NDArray[np.float32], validate: bool = False) -> NDArray[np.int8]:
        """
        Optimized vectorized quantization with optional validation.

        Args:
            data: Input float32 array
            validate: If True, performs input validation

        Returns:
            Quantized int8 array
        """
        if validate:
            self._validate_input(data)

        # Update scale and zero_point if using dynamic range
        if self._dynamic_range:
            self.compute_scale_and_zero_point(data)

        # Vectorized computation with minimal type conversions
        quantized = np.rint((data - self._zero_point) / self._scale - 128)

        # Optimize clipping operation and type conversion
        quantized = np.clip(quantized, self._INT8_MIN, self._INT8_MAX).astype(np.int8)

        # Track clipping statistics
        if validate:
            with self._stats_lock:  # Use lock if stats are accessed from multiple threads
                self._quantization_stats['total_values'] += data.size
                self._quantization_stats['clipped_values'] += np.sum(
                    (quantized < self._INT8_MIN) | (quantized > self._INT8_MAX)
                )

        return quantized

    @lru_cache(maxsize=128)
    def _cached_dequantize(self, quantized_value: int) -> float:
        """Cached dequantization for frequently occurring values."""
        return float((quantized_value + 128) * self._scale + self._zero_point)

    def dequantize(self, data: NDArray[np.int8], use_cache: bool = True) -> NDArray[np.float32]:
        """
        Optimized dequantization with optional caching.

        Args:
            data: Input int8 array
            use_cache: If True, uses cached values for common quantized values

        Returns:
            Dequantized float32 array
        """

        if use_cache:
            # Vectorize the cache lookup for better performance
            vectorized_cached_dequantize = np.vectorize(self._cached_dequantize)
            return vectorized_cached_dequantize(data)

        # Fast path for large arrays (no caching)
        return ((data.astype(np.int32) + 128) * self._scale + self._zero_point).astype(np.float32)

    def get_quantization_stats(self) -> dict:
        """Returns statistics about quantization operations."""
        with self._stats_lock:  # Use lock for thread-safe access
            stats = self._quantization_stats.copy()
        if stats['total_values'] > 0:
            stats['clipping_ratio'] = stats['clipped_values'] / stats['total_values']
        return stats

    def _validate_input(self, data: NDArray[np.float32]) -> None:
        """Validate input data for potential issues."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if data.dtype != np.float32:
            logger.warning("Input data is not float32, converting automatically")  # Use logger
            data = data.astype(np.float32)  # Convert data
        if np.isnan(data).any():
            logger.warning("Input contains NaN values")  # Use logger
        if np.isinf(data).any():
            logger.warning("Input contains infinite values")  # Use logger

    def reset_stats(self) -> None:
        """Reset quantization statistics."""
        with self._stats_lock:  # Use lock for thread-safe access
            self._quantization_stats = {
                'clipped_values': 0,
                'total_values': 0
            }

    def clear_cache(self) -> None:
        """Clears the dequantization cache."""
        self._cached_dequantize.cache_clear()


# Example usage:
if __name__ == "__main__":
    # Create a default quantizer with default config
    quantizer = Quantizer()
    print("Default config:", quantizer.get_config())

    # Update configuration if needed
    new_config = QuantizationConfig(dynamic_range=False, cache_size=256, int8_min=-128, int8_max=127,
                                     uint8_range=255.0)
    quantizer.update_config(new_config)
    print("Updated config:", quantizer.get_config())

    # Example data
    data = np.random.uniform(-1, 1, size=(1000,)).astype(np.float32)
    q_data = quantizer.quantize(data, validate=True)
    deq_data = quantizer.dequantize(q_data)
    stats = quantizer.get_quantization_stats()

    print("Quantization stats:", stats)
