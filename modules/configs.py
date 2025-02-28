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
# Configuration for CPU-accelerated Models Inference
# -----------------------------------------------------------------------------

@dataclass(order=True)
class PrioritizedItem:
    """Wrapper for prioritized queue items."""
    priority: int
    timestamp: float = field(compare=False)
    item: Any = field(compare=False)
# Define enums
class OptimizationLevel(Enum):
    NONE = auto()
    BASIC = auto()
    ADVANCED = auto()
    EXTREME = auto()

class ModelType(Enum):
    SKLEARN = auto()
    XGBOOST = auto()
    LIGHTGBM = auto()
    CUSTOM = auto()
    ENSEMBLE = auto()

class EngineState(Enum):
    INITIALIZING = auto()
    READY = auto()
    LOADING = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()
    STOPPING = auto()
    STOPPED = auto()

# -----------------------------------------------------------------------------
# Configuration for CPU-accelerated Models
# -----------------------------------------------------------------------------

@dataclass
class CPUAcceleratedModelConfig:
    """
    Enhanced configuration for CPU-accelerated model with advanced features.
    
    Configuration parameters are organized by functional category:
    - Processing: Core processing settings
    - Batching: Batch processing configuration
    - Optimization: Performance optimization flags
    - Caching: Result caching configuration
    - Monitoring: Telemetry and observability
    - Resource Management: System resource controls
    - Advanced Features: Additional functionality
    - Quantization: Model and input quantization settings
    """
    # Processing configuration
    num_threads: int = field(default_factory=lambda: os.cpu_count() or 4)
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    model_version: str = "1.0.0"
    
    # Batch processing settings
    enable_batching: bool = True
    initial_batch_size: int = 64
    max_batch_size: int = 512
    min_batch_size: int = 4
    batch_timeout: float = 0.1
    batch_processing_strategy: str = "adaptive"
    enable_adaptive_batching: bool = True
    target_latency_ms: float = 50.0  # Target latency for adaptive batching
    
    # Memory and performance optimizations
    enable_memory_optimization: bool = True
    enable_intel_optimization: bool = True
    enable_quantization: bool = False
    quantization_dtype: str = "int8"  # Options: int8, uint8, int16
    enable_feature_scaling: bool = False  # Enable feature scaling in preprocessing
    
    # Caching and deduplication
    enable_request_deduplication: bool = True
    cache_ttl_seconds: int = 300  # Time-to-live for cache entries (5 minutes)
    max_cache_entries: int = 1000  # Maximum number of cached entries
    
    # Monitoring and debugging
    enable_monitoring: bool = True
    debug_mode: bool = False
    monitoring_window: int = 100
    monitoring_interval: float = 5.0  # Monitor update interval in seconds
    
    # Resource management
    memory_limit_gb: Optional[float] = None
    memory_high_watermark_mb: float = 1024.0  # Trigger GC when memory exceeds this
    enable_throttling: bool = False
    max_concurrent_requests: int = 100
    throttle_on_high_cpu: bool = False
    cpu_threshold_percent: float = 80.0  # CPU threshold for throttling
    set_cpu_affinity: bool = False  # Set CPU affinity for process
    
    # Advanced Features
    enable_warmup: bool = True  # Perform model warmup to stabilize first-inference latency
    enable_data_versioning: bool = False  # Track data version for cache invalidation
    
    # Quantization settings
    quantization_config: Optional[QuantizationConfig] = None
    enable_model_quantization: bool = False  # Quantize model weights
    enable_input_quantization: bool = False  # Quantize input data
    enable_quantization_aware_inference: bool = False  # Simulate quantization during inference
    
    # Extension points
    custom_extensions: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Ensure min_batch_size is not greater than initial_batch_size
        if self.min_batch_size > self.initial_batch_size:
            self.min_batch_size = self.initial_batch_size
            
        # Ensure initial_batch_size is not greater than max_batch_size
        if self.initial_batch_size > self.max_batch_size:
            self.initial_batch_size = self.max_batch_size
            
        # Set reasonable defaults for CPU thread count
        if self.num_threads <= 0:
            self.num_threads = os.cpu_count() or 4
            
        # Convert batch_processing_strategy to lowercase for case insensitivity
        if isinstance(self.batch_processing_strategy, str):
            self.batch_processing_strategy = self.batch_processing_strategy.lower()
            
        # Set default quantization config if not provided but quantization is enabled
        if (self.enable_quantization or self.enable_model_quantization or 
            self.enable_input_quantization) and self.quantization_config is None:
            self.quantization_config = QuantizationConfig(
                quantization_type=self.quantization_dtype,
                quantization_mode=QuantizationMode.DYNAMIC_PER_BATCH.value
            )
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary for serialization."""
        config_dict = {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
        
        # Handle nested QuantizationConfig
        if self.quantization_config is not None:
            config_dict['quantization_config'] = asdict(self.quantization_config)
        
        return config_dict
                
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CPUAcceleratedModelConfig':
        """Create config from a dictionary."""
        # Handle nested QuantizationConfig if present
        if 'quantization_config' in config_dict and isinstance(config_dict['quantization_config'], dict):
            config_dict['quantization_config'] = QuantizationConfig(**config_dict['quantization_config'])
        
        return cls(**config_dict)


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