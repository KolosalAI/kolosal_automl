from dataclasses import dataclass, field
from enum import Enum
import os
import numpy as np
from typing import Optional, List, Dict, Tuple, NamedTuple, Callable, Any
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum, auto
import os

# -----------------------------------------------------------------------------
# Quantization Configuration
# -----------------------------------------------------------------------------
class QuantizationType(Enum):
    INT8 = "int8"
    UINT8 = "uint8"
    INT16 = "int16"

class QuantizationMode(Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    DYNAMIC_PER_BATCH = "dynamic_per_batch"
    DYNAMIC_PER_CHANNEL = "dynamic_per_channel"

@dataclass
class QuantizationConfig:
    """Configuration for the quantizer."""
    quantization_type: str = QuantizationType.INT8.value
    quantization_mode: str = QuantizationMode.DYNAMIC_PER_BATCH.value
    enable_cache: bool = True
    cache_size: int = 1024
    buffer_size: int = 0  # 0 means no buffer
    use_percentile: bool = False
    min_percentile: float = 0.1
    max_percentile: float = 99.9
    error_on_nan: bool = False
    error_on_inf: bool = False
    # New options for better control
    outlier_threshold: Optional[float] = None  # For handling outliers
    num_bits: int = 8  # Allow custom bit widths
    optimize_memory: bool = True  # Enable memory optimization


# -----------------------------------------------------------------------------
# Configuration for Batch Processing
# -----------------------------------------------------------------------------

class BatchProcessingStrategy(Enum):
    """Strategy for batch processing."""
    FIXED = auto()      # Use fixed batch size
    ADAPTIVE = auto()   # Dynamically adjust batch size based on system load
    GREEDY = auto()     # Process as many items as available up to max_batch_size

class BatchPriority(Enum):
    """Priority levels for batch processing."""
    CRITICAL = 0    # Highest priority
    HIGH = 1
    NORMAL = 2      # Default priority
    LOW = 3
    BACKGROUND = 4  # Lowest priority

class PrioritizedItem(NamedTuple):
    """Item with priority for queue ordering."""
    priority: int       # Lower number = higher priority
    timestamp: float    # Time when item was added (for FIFO within same priority)
    item: Any           # The actual item
    
    def __lt__(self, other):
        # Compare first by priority, then by timestamp
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

@dataclass
class BatchProcessorConfig:
    """Configuration for BatchProcessor."""
    
    # Batch sizing parameters
    min_batch_size: int = 1
    max_batch_size: int = 64
    initial_batch_size: int = 16
    
    # Queue parameters
    max_queue_size: int = 1000
    enable_priority_queue: bool = False
    
    # Processing parameters
    batch_timeout: float = 0.1  # Max time to wait for batch formation (seconds)
    item_timeout: float = 10.0  # Max time to wait for single item processing (seconds)
    min_batch_interval: float = 0.0  # Min time between batch processing (seconds)
    
    # Processing strategy
    processing_strategy: BatchProcessingStrategy = BatchProcessingStrategy.ADAPTIVE
    enable_adaptive_batching: bool = True
    
    # Retry parameters
    max_retries: int = 2
    retry_delay: float = 0.1  # Seconds between retries
    reduce_batch_on_failure: bool = True
    
    # Memory management
    max_batch_memory_mb: Optional[float] = None  # Max memory per batch in MB
    enable_memory_optimization: bool = True
    gc_batch_threshold: int = 32  # Run GC after processing batches larger than this
    
    # Monitoring and statistics
    enable_monitoring: bool = True
    monitoring_window: int = 100  # Number of batches to keep statistics for
    
    # Thread pool configuration
    max_workers: int = 4
    
    # Health monitoring
    enable_health_monitoring: bool = True
    health_check_interval: float = 5.0  # Seconds between health checks
    memory_warning_threshold: float = 70.0  # Memory usage percentage to trigger warning
    memory_critical_threshold: float = 85.0  # Memory usage percentage to trigger critical actions
    queue_warning_threshold: int = 100  # Queue size to trigger warning
    queue_critical_threshold: int = 500  # Queue size to trigger critical actions
    
    # Debug mode
    debug_mode: bool = False

# -----------------------------------------------------------------------------
# Configuration for Data Preprocessor
# -----------------------------------------------------------------------------

class NormalizationType(Enum):
    """Enumeration of supported normalization strategies."""
    NONE = auto()
    STANDARD = auto()  # z-score normalization: (x - mean) / std
    MINMAX = auto()    # Min-max scaling: (x - min) / (max - min)
    ROBUST = auto()    # Robust scaling using percentiles
    CUSTOM = auto()    # Custom normalization function

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the configuration
        """
        # Convert the configuration to a dictionary
        config_dict = asdict(self)
        
        # Handle special types that need custom serialization
        if 'normalization' in config_dict:
            # Convert enum to string for JSON serialization
            config_dict['normalization'] = self.normalization.name if self.normalization else 'NONE'
        
        # Remove non-serializable callable objects
        if 'custom_normalization_fn' in config_dict:
            config_dict['custom_normalization_fn'] = None
        if 'custom_transform_fn' in config_dict:
            config_dict['custom_transform_fn'] = None
        
        # Convert numpy dtypes to strings
        if 'dtype' in config_dict and isinstance(config_dict['dtype'], np.dtype):
            config_dict['dtype'] = str(config_dict['dtype'])
        
        return config_dict


# -----------------------------------------------------------------------------
# Configuration for Inference Engine
# -----------------------------------------------------------------------------
class ModelType(Enum):
    SKLEARN = auto()
    XGBOOST = auto()
    LIGHTGBM = auto()
    CUSTOM = auto()
    ENSEMBLE = auto()

class EngineState(Enum):
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    LOADING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()

@dataclass
class InferenceEngineConfig:
    # General engine settings
    model_version: str = "1.0"
    debug_mode: bool = False
    num_threads: int = 4
    set_cpu_affinity: bool = False

    # Intel optimizations flag
    enable_intel_optimization: bool = False

    # Quantization settings
    enable_quantization: bool = False
    enable_model_quantization: bool = False
    enable_input_quantization: bool = False
    quantization_dtype: str = "int8"  # e.g., "int8" or "float16"
    quantization_config: Optional[QuantizationConfig] = None

    # Caching and deduplication
    enable_request_deduplication: bool = True
    max_cache_entries: int = 1000
    cache_ttl_seconds: int = 300  # 5 minutes

    # Monitoring settings
    monitoring_window: int = 100
    enable_monitoring: bool = True
    monitoring_interval: float = 10.0  # seconds between checks
    throttle_on_high_cpu: bool = True
    cpu_threshold_percent: float = 90.0  # CPU usage threshold in percent
    memory_high_watermark_mb: float = 1024.0  # memory high watermark in MB
    memory_limit_gb: Optional[float] = None  # optional absolute limit in GB

    # Batch processing settings
    enable_batching: bool = True
    batch_processing_strategy: str = "adaptive"  # e.g., "adaptive", "fixed", etc.
    batch_timeout: float = 0.1  # seconds to wait for additional requests
    max_concurrent_requests: int = 8
    initial_batch_size: int = 16
    min_batch_size: int = 1
    max_batch_size: int = 64
    enable_adaptive_batching: bool = True
    enable_memory_optimization: bool = True

    # Preprocessing settings
    enable_feature_scaling: bool = False

    # Warmup settings
    enable_warmup: bool = True

    # Inference settings
    enable_quantization_aware_inference: bool = False

    # Throttling flag (dynamically updated based on CPU usage)
    enable_throttling: bool = False

    def to_dict(self) -> dict:
        """Return configuration as a dictionary."""
        return {
            "model_version": self.model_version,
            "debug_mode": self.debug_mode,
            "num_threads": self.num_threads,
            "set_cpu_affinity": self.set_cpu_affinity,
            "enable_intel_optimization": self.enable_intel_optimization,
            "enable_quantization": self.enable_quantization,
            "enable_model_quantization": self.enable_model_quantization,
            "enable_input_quantization": self.enable_input_quantization,
            "quantization_dtype": self.quantization_dtype,
            "quantization_config": self.quantization_config,
            "enable_request_deduplication": self.enable_request_deduplication,
            "max_cache_entries": self.max_cache_entries,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "monitoring_window": self.monitoring_window,
            "enable_monitoring": self.enable_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "throttle_on_high_cpu": self.throttle_on_high_cpu,
            "cpu_threshold_percent": self.cpu_threshold_percent,
            "memory_high_watermark_mb": self.memory_high_watermark_mb,
            "memory_limit_gb": self.memory_limit_gb,
            "enable_batching": self.enable_batching,
            "batch_processing_strategy": self.batch_processing_strategy,
            "batch_timeout": self.batch_timeout,
            "max_concurrent_requests": self.max_concurrent_requests,
            "initial_batch_size": self.initial_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "enable_adaptive_batching": self.enable_adaptive_batching,
            "enable_memory_optimization": self.enable_memory_optimization,
            "enable_feature_scaling": self.enable_feature_scaling,
            "enable_warmup": self.enable_warmup,
            "enable_quantization_aware_inference": self.enable_quantization_aware_inference,
            "enable_throttling": self.enable_throttling,
        }

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


def optimize_for_inference(model: Any, model_type: ModelType) -> Any:
    """
    Apply optimizations to model for faster inference.
    
    Args:
        model: The model to optimize
        model_type: Type of the model
        
    Returns:
        Optimized model
    """
    # Model-specific optimizations
    if model_type == ModelType.SKLEARN:
        # Some sklearn models can be optimized
        if hasattr(model, 'n_jobs'):
            # Set parallel jobs to 1 for inference (avoid overhead)
            model.n_jobs = 1
    
    elif model_type == ModelType.XGBOOST:
        # XGBoost optimizations
        try:
            # Use predictor type that is faster for CPU
            model.set_param('predictor', 'cpu_predictor')
            # Set number of threads
            model.set_param('nthread', os.cpu_count() or 4)
        except:
            pass
    
    elif model_type == ModelType.LIGHTGBM:
        # LightGBM optimizations
        try:
            # Set number of threads for inference
            model.params['num_threads'] = os.cpu_count() or 4
        except:
            pass
    
    return model