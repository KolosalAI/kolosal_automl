from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import os
from typing import Optional

class BatchProcessingStrategy(Enum):
    ADAPTIVE = "adaptive"  # Dynamically adjusts batch size
    FIXED = "fixed"       # Uses fixed batch size
    TIMEOUT = "timeout"   # Primarily driven by timeout
class OptimizationLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"

@dataclass
class ModelConfig:
    """Enhanced configuration for CPU-accelerated model."""
    # Processing configuration
    num_threads: int = field(default_factory=lambda: os.cpu_count() or 4)
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

    # Feature Flags
    enable_request_deduplication: bool = False
    enable_data_versioning: bool = False
    model_version: str = "1.0.0" # Example
    monitoring_interval: int = 60 # seconds

    # Advanced parameters
    quantization_dtype: str = "int8" # Example: Supports int8, uint8, int16
    batch_processing_strategy: BatchProcessingStrategy = BatchProcessingStrategy.ADAPTIVE

    # Resource limits
    memory_limit_gb: Optional[float] = None # Example: Limit memory usage

@dataclass
class BatchProcessorConfig:
    def __init__(self,
                 batch_timeout: float,
                 max_queue_size: int,
                 initial_batch_size: int,
                 min_batch_size: int,
                 max_batch_size: int,
                 enable_monitoring: bool,
                 monitoring_window: int,
                 max_retries: int,
                 retry_delay: float,
                 processing_strategy: BatchProcessingStrategy,
                 max_workers: int = 4  # Add max_workers here
                 ):
        self.batch_timeout = batch_timeout
        self.max_queue_size = max_queue_size
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.enable_monitoring = enable_monitoring
        self.monitoring_window = monitoring_window
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.processing_strategy = processing_strategy
        self.max_workers = max_workers

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