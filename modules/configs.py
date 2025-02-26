from dataclasses import dataclass, field
from enum import Enum
import os
import numpy as np
from typing import Optional, List, Dict, Tuple

# -----------------------------------------------------------------------------
# Enums for Configuration Domains
# -----------------------------------------------------------------------------

class TaskType(Enum):
    """Defines types of machine learning tasks."""
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    CLUSTERING = "Clustering"

    def __str__(self):
        return self.value

class BatchProcessingStrategy(Enum):
    """Defines strategies for batch processing."""
    ADAPTIVE = "adaptive"  # Dynamically adjusts batch size
    FIXED = "fixed"        # Uses fixed batch size
    TIMEOUT = "timeout"    # Driven primarily by timeout

class OptimizationLevel(Enum):
    """Defines levels of CPU optimization."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"

class NormalizationType(Enum):
    """Defines types of data normalization."""
    STANDARD = "standard"  # (x - mean) / std
    MINMAX = "minmax"      # (x - min) / (max - min)
    ROBUST = "robust"      # Using percentiles instead of min/max
    NONE = "none"

# -----------------------------------------------------------------------------
# Hyperparameter Configuration
# -----------------------------------------------------------------------------

@dataclass
class HyperParameterConfig:
    """Configuration for a single hyperparameter."""
    name: str
    min_val: float
    max_val: float
    step: float
    default: float

# Default hyperparameters for selected models (e.g., for clustering)
DEFAULT_HYPERPARAMETERS: Dict[str, List[HyperParameterConfig]] = {
    "kmeans": [
        HyperParameterConfig("n_clusters", 2, 10, 1, 3),
        HyperParameterConfig("max_iter", 100, 500, 100, 300)
    ],
    "dbscan": [
        HyperParameterConfig("eps", 0.1, 1.0, 0.1, 0.5),
        HyperParameterConfig("min_samples", 2, 10, 1, 5)
    ],
    # Additional model hyperparameters can be added here.
}

# -----------------------------------------------------------------------------
# Target Metrics Definition
# -----------------------------------------------------------------------------

@dataclass
class TargetMetrics:
    """Records evaluation results for a given metric."""
    metric_name: str
    target_score: float
    achieved_value: float
    is_achieved: bool

# -----------------------------------------------------------------------------
# AutoML Model Configurations
# -----------------------------------------------------------------------------

@dataclass
class AutoMLModelConfig:
    """Configuration for the AutoML model."""
    target_column: str
    task_type: TaskType
    models: List[str]
    time_budget: int
    n_clusters: int
    random_seed: int
    verbosity: int
    show_shap: bool
    metric_name: str
    target_score: float
    hyperparameter_configs: Dict[str, List[HyperParameterConfig]]
    auto_model_selection: bool
    use_cv: bool

# Alternate naming (ModelConfig) can be used interchangeably if needed.
ModelConfig = AutoMLModelConfig

# -----------------------------------------------------------------------------
# CPU-Accelerated Model Configuration
# -----------------------------------------------------------------------------

@dataclass
class CPUAcceleratedModelConfig:
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
# -----------------------------------------------------------------------------
# Batch Processor Configuration
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Preprocessor Configuration
# -----------------------------------------------------------------------------

@dataclass
class PreprocessorConfig:
    """Configuration for data preprocessing."""
    normalization: NormalizationType = NormalizationType.STANDARD
    epsilon: float = 1e-8
    clip_values: bool = True
    clip_range: Tuple[float, float] = (-5.0, 5.0)
    robust_percentiles: Tuple[float, float] = (1.0, 99.0)
    handle_inf: bool = True
    handle_nan: bool = True
    dtype: np.dtype = np.float32

# -----------------------------------------------------------------------------
# Quantization Configuration
# -----------------------------------------------------------------------------

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    dynamic_range: bool = True
    cache_size: int = 128
    int8_min: int = -128
    int8_max: int = 127
    uint8_range: float = 255.0
