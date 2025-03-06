from dataclasses import dataclass, field
from enum import Enum
import os
import numpy as np
from typing import Optional, List, Dict, Tuple, NamedTuple, Callable, Any
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum, auto
import json
import os

# Constants definitions
CHECKPOINT_PATH = "./checkpoints"
MODEL_REGISTRY_PATH = "./model_registry"

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
# Configuration for Training Engine
# -----------------------------------------------------------------------------
class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


class OptimizationStrategy(Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"


class MLTrainingEngineConfig:
    """Configuration for the ML Training Engine"""
    
    def __init__(
        self,
        task_type: TaskType = TaskType.CLASSIFICATION,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1,
        cv_folds: int = 5,
        test_size: float = 0.2,
        stratify: bool = True,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH,
        optimization_iterations: int = 50,
        early_stopping: bool = True,
        feature_selection: bool = True,
        feature_selection_method: str = "mutual_info",
        feature_selection_k: Optional[int] = None,
        feature_importance_threshold: float = 0.01,
        preprocessing_config: Optional[PreprocessorConfig] = None,
        batch_processing_config: Optional[BatchProcessorConfig] = None,
        inference_config: Optional[InferenceEngineConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        model_path: str = "./models",
        experiment_tracking: bool = True,
        use_intel_optimization: bool = True,
        memory_optimization: bool = True,
        enable_distributed: bool = False,
        log_level: str = "INFO",
    ):
        self.task_type = task_type
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.stratify = stratify
        self.optimization_strategy = optimization_strategy
        self.optimization_iterations = optimization_iterations
        self.early_stopping = early_stopping
        self.feature_selection = feature_selection
        self.feature_selection_method = feature_selection_method
        self.feature_selection_k = feature_selection_k
        self.feature_importance_threshold = feature_importance_threshold
        
        # Set default configurations if none provided
        if preprocessing_config is None:
            self.preprocessing_config = PreprocessorConfig(
                normalization=NormalizationType.STANDARD,
                handle_nan=True,
                handle_inf=True,
                detect_outliers=True,
                parallel_processing=True
            )
        else:
            self.preprocessing_config = preprocessing_config
            
        if batch_processing_config is None:
            self.batch_processing_config = BatchProcessorConfig(
                initial_batch_size=100,
                min_batch_size=50,
                max_batch_size=200,
                max_queue_size=1000,
                batch_timeout=1.0,
                enable_priority_queue=True,
                enable_monitoring=True,
                enable_memory_optimization=True
            )
        else:
            self.batch_processing_config = batch_processing_config
            
        if inference_config is None:
            self.inference_config = InferenceEngineConfig(
                enable_intel_optimization=use_intel_optimization,
                enable_batching=True,
                enable_quantization=True,
                debug_mode=False
            )
        else:
            self.inference_config = inference_config
            
        if quantization_config is None:
            self.quantization_config = QuantizationConfig(
                quantization_type=QuantizationType.INT8.value,
                quantization_mode=QuantizationMode.DYNAMIC_PER_BATCH.value,
                enable_cache=True,
                cache_size=256
            )
        else:
            self.quantization_config = quantization_config
            
        self.model_path = model_path
        self.experiment_tracking = experiment_tracking
        self.use_intel_optimization = use_intel_optimization
        self.memory_optimization = memory_optimization
        self.enable_distributed = enable_distributed
        self.log_level = log_level
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            "task_type": self.task_type.value,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "cv_folds": self.cv_folds,
            "test_size": self.test_size,
            "stratify": self.stratify,
            "optimization_strategy": self.optimization_strategy.value,
            "optimization_iterations": self.optimization_iterations,
            "early_stopping": self.early_stopping,
            "feature_selection": self.feature_selection,
            "feature_selection_method": self.feature_selection_method,
            "feature_selection_k": self.feature_selection_k,
            "feature_importance_threshold": self.feature_importance_threshold,
            "model_path": self.model_path,
            "experiment_tracking": self.experiment_tracking,
            "use_intel_optimization": self.use_intel_optimization,
            "memory_optimization": self.memory_optimization,
            "enable_distributed": self.enable_distributed,
            "log_level": self.log_level
        }
