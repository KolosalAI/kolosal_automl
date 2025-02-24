from dataclasses import dataclass
from enum import Enum
import numpy as np
import os


class OptimizationLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"

@dataclass
class ModelConfig:
    """Enhanced configuration for CPU-accelerated model."""
    # Processing configuration
    num_threads: int = os.cpu_count() or 4
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    
    # Batch processing settings
    enable_batching: bool = True
    initial_batch_size: int = 64
    max_batch_size: int = 512
    batch_timeout: float = 0.1
    
    # Memory and performance optimizations
    enable_memory_optimization: bool = True
    enable_intel_optimization: bool = True
    enable_quantization: bool = False
    
    # Monitoring and debugging
    enable_monitoring: bool = True
    debug_mode: bool = False
    monitoring_window: int = 100

class BatchProcessingStrategy(Enum):
    ADAPTIVE = "adaptive"  # Dynamically adjusts batch size
    FIXED = "fixed"       # Uses fixed batch size
    TIMEOUT = "timeout"   # Primarily driven by timeout

@dataclass
class BatchProcessorConfig:
    """Configuration for batch processing."""
    max_queue_size: int = 1000
    initial_batch_size: int = 64
    min_batch_size: int = 8
    max_batch_size: int = 512
    batch_timeout: float = 0.1
    processing_strategy: BatchProcessingStrategy = BatchProcessingStrategy.ADAPTIVE
    max_retries: int = 3
    retry_delay: float = 0.1
    enable_monitoring: bool = True
    monitoring_window: int = 100  # Number of batches to keep statistics for

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

@dataclass
class QuantizationConfig:
    """
    Configuration for the Quantizer.
    
    Attributes:
        dynamic_range: If True, automatically adjusts range based on data distribution.
        cache_size: Size of the LRU cache for dequantization operations.
        int8_min: Minimum value for int8 clipping.
        int8_max: Maximum value for int8 clipping.
        uint8_range: The range for uint8 normalization.
    """
    dynamic_range: bool = True
    cache_size: int = 128
    int8_min: int = -128
    int8_max: int = 127
    uint8_range: float = 255.0