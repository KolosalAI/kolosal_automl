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
    NONE = "none"
    INT8 = "int8"
    INT16 = "int16"
    FLOAT16 = "float16"
    MIXED = "mixed"

class QuantizationMode(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    DYNAMIC_PER_BATCH = "dynamic_per_batch"
    CALIBRATED = "calibrated"

@dataclass
class QuantizationConfigBase:
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
class BatchProcessorConfigBase:
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
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"
    LOG = "log"
    POWER = "power"

@dataclass
class PreprocessorConfigBase:
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
class InferenceEngineConfigBase:
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
    quantization_config: Optional[QuantizationConfigBase] = None

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
from enum import Enum
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"  # Added time series task type
    MULTI_LABEL = "multi_label"  # Added multi-label classification
    RANKING = "ranking"          # Added ranking task type


class OptimizationStrategy(Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"
    ASHT = "adaptive_surrogate_assisted_hyperparameter_tuning"
    HYPERX = "hyper_optimization_x"
    OPTUNA = "optuna"  # Added popular Optuna framework
    SUCCESSIVE_HALVING = "successive_halving"  # Added ASHA variant
    BOHB = "bayesian_optimization_hyperband"  # Added BOHB combination




class ModelSelectionCriteria(Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    ROC_AUC = "roc_auc"
    Matthews_CORRELATION = "matthews_correlation"
    ROOT_MEAN_SQUARED_ERROR = "rmse"
    MEAN_ABSOLUTE_ERROR = "mae"
    R2 = "r2"
    SILHOUETTE = "silhouette"
    EXPLAINED_VARIANCE = "explained_variance"
    BIC = "bic"
    AIC = "aic"
    CUSTOM = "custom"


class AutoMLMode(Enum):
    DISABLED = "disabled"
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    EXPERIMENTAL = "experimental"


class PreprocessorConfig(PreprocessorConfigBase):
    """Configuration for data preprocessing"""
    
    def __init__(
        self,
        normalization: Union[NormalizationType, str] = NormalizationType.STANDARD,
        handle_nan: bool = True,
        handle_inf: bool = True,
        detect_outliers: bool = True,
        outlier_method: str = "isolation_forest",
        outlier_contamination: float = 0.05,
        categorical_encoding: str = "one_hot",
        categorical_max_categories: int = 20,
        auto_feature_selection: bool = True,
        numeric_transformations: List[str] = None,
        text_vectorization: Optional[str] = "tfidf",
        text_max_features: int = 5000,
        dimension_reduction: Optional[str] = None,
        dimension_reduction_target: Optional[int] = None,
        datetime_features: bool = True,
        image_preprocessing: Optional[Dict] = None,
        handle_imbalance: bool = False,
        imbalance_strategy: str = "smote",
        feature_interaction: bool = False,
        feature_interaction_max: int = 10,
        custom_transformers: List = None,
        transformation_pipeline: List = None,
        parallel_processing: bool = True,
        cache_preprocessing: bool = True,
        verbosity: int = 1
    ):
        # Handle enum as string
        if isinstance(normalization, str):
            try:
                self.normalization = NormalizationType[normalization.upper()]
            except KeyError:
                self.normalization = NormalizationType.STANDARD
        else:
            self.normalization = normalization
            
        self.handle_nan = handle_nan
        self.handle_inf = handle_inf
        self.detect_outliers = detect_outliers
        self.outlier_method = outlier_method
        self.outlier_contamination = outlier_contamination
        self.categorical_encoding = categorical_encoding
        self.categorical_max_categories = categorical_max_categories
        self.auto_feature_selection = auto_feature_selection
        self.numeric_transformations = numeric_transformations or []
        self.text_vectorization = text_vectorization
        self.text_max_features = text_max_features
        self.dimension_reduction = dimension_reduction
        self.dimension_reduction_target = dimension_reduction_target
        self.datetime_features = datetime_features
        self.image_preprocessing = image_preprocessing or {}
        self.handle_imbalance = handle_imbalance
        self.imbalance_strategy = imbalance_strategy
        self.feature_interaction = feature_interaction
        self.feature_interaction_max = feature_interaction_max
        self.custom_transformers = custom_transformers or []
        self.transformation_pipeline = transformation_pipeline or []
        self.parallel_processing = parallel_processing
        self.cache_preprocessing = cache_preprocessing
        self.verbosity = verbosity
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            "normalization": self.normalization.value,
            "handle_nan": self.handle_nan,
            "handle_inf": self.handle_inf,
            "detect_outliers": self.detect_outliers,
            "outlier_method": self.outlier_method,
            "outlier_contamination": self.outlier_contamination,
            "categorical_encoding": self.categorical_encoding,
            "categorical_max_categories": self.categorical_max_categories,
            "auto_feature_selection": self.auto_feature_selection,
            "numeric_transformations": self.numeric_transformations,
            "text_vectorization": self.text_vectorization,
            "text_max_features": self.text_max_features,
            "dimension_reduction": self.dimension_reduction,
            "dimension_reduction_target": self.dimension_reduction_target,
            "datetime_features": self.datetime_features,
            "image_preprocessing": self.image_preprocessing,
            "handle_imbalance": self.handle_imbalance,
            "imbalance_strategy": self.imbalance_strategy,
            "feature_interaction": self.feature_interaction,
            "feature_interaction_max": self.feature_interaction_max,
            "parallel_processing": self.parallel_processing,
            "cache_preprocessing": self.cache_preprocessing,
            "verbosity": self.verbosity
        }


class BatchProcessorConfig(BatchProcessorConfigBase):
    """Configuration for batch processing"""
    
    def __init__(
        self,
        initial_batch_size: int = 100,
        min_batch_size: int = 50,
        max_batch_size: int = 200,
        max_queue_size: int = 1000,
        batch_timeout: float = 1.0,
        num_workers: int = 4,
        adaptive_batching: bool = True,
        batch_allocation_strategy: str = "dynamic",
        enable_priority_queue: bool = True,
        priority_levels: int = 3,
        enable_monitoring: bool = True,
        monitoring_interval: float = 2.0,
        enable_memory_optimization: bool = True,
        enable_prefetching: bool = True,
        prefetch_batches: int = 2,
        checkpoint_batches: bool = False,
        checkpoint_interval: int = 100,
        error_handling: str = "retry",
        max_retries: int = 3,
        retry_delay: float = 0.5,
        distributed_processing: bool = False,
        resource_allocation: Dict = None
    ):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.batch_timeout = batch_timeout
        self.num_workers = num_workers
        self.adaptive_batching = adaptive_batching
        self.batch_allocation_strategy = batch_allocation_strategy
        self.enable_priority_queue = enable_priority_queue
        self.priority_levels = priority_levels
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_prefetching = enable_prefetching
        self.prefetch_batches = prefetch_batches
        self.checkpoint_batches = checkpoint_batches
        self.checkpoint_interval = checkpoint_interval
        self.error_handling = error_handling
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.distributed_processing = distributed_processing
        self.resource_allocation = resource_allocation or {}
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            "initial_batch_size": self.initial_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "max_queue_size": self.max_queue_size,
            "batch_timeout": self.batch_timeout,
            "num_workers": self.num_workers,
            "adaptive_batching": self.adaptive_batching,
            "batch_allocation_strategy": self.batch_allocation_strategy,
            "enable_priority_queue": self.enable_priority_queue,
            "priority_levels": self.priority_levels,
            "enable_monitoring": self.enable_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "enable_memory_optimization": self.enable_memory_optimization,
            "enable_prefetching": self.enable_prefetching,
            "prefetch_batches": self.prefetch_batches,
            "checkpoint_batches": self.checkpoint_batches,
            "checkpoint_interval": self.checkpoint_interval,
            "error_handling": self.error_handling,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "distributed_processing": self.distributed_processing,
            "resource_allocation": self.resource_allocation
        }


class InferenceEngineConfig (InferenceEngineConfigBase):
    """Configuration for the inference engine"""
    
    def __init__(
        self,
        enable_intel_optimization: bool = True,
        enable_batching: bool = True,
        enable_quantization: bool = True,
        model_cache_size: int = 5,
        model_precision: str = "fp32",
        max_batch_size: int = 64,
        timeout_ms: int = 100,
        enable_jit: bool = True,
        enable_onnx: bool = False,
        onnx_opset: int = 13,
        enable_tensorrt: bool = False,
        runtime_optimization: bool = True,
        fallback_to_cpu: bool = True,
        thread_count: int = 0,
        warmup: bool = True,
        warmup_iterations: int = 10,
        profiling: bool = False,
        batching_strategy: str = "dynamic",
        output_streaming: bool = False,
        debug_mode: bool = False,
        memory_growth: bool = True,
        custom_ops: List = None,
        use_platform_accelerator: bool = True,
        platform_accelerator_config: Dict = None
    ):
        self.enable_intel_optimization = enable_intel_optimization
        self.enable_batching = enable_batching
        self.enable_quantization = enable_quantization
        self.model_cache_size = model_cache_size
        self.model_precision = model_precision
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.enable_jit = enable_jit
        self.enable_onnx = enable_onnx
        self.onnx_opset = onnx_opset
        self.enable_tensorrt = enable_tensorrt
        self.runtime_optimization = runtime_optimization
        self.fallback_to_cpu = fallback_to_cpu
        self.thread_count = thread_count
        self.warmup = warmup
        self.warmup_iterations = warmup_iterations
        self.profiling = profiling
        self.batching_strategy = batching_strategy
        self.output_streaming = output_streaming
        self.debug_mode = debug_mode
        self.memory_growth = memory_growth
        self.custom_ops = custom_ops or []
        self.use_platform_accelerator = use_platform_accelerator
        self.platform_accelerator_config = platform_accelerator_config or {}
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            "enable_intel_optimization": self.enable_intel_optimization,
            "enable_batching": self.enable_batching,
            "enable_quantization": self.enable_quantization,
            "model_cache_size": self.model_cache_size,
            "model_precision": self.model_precision,
            "max_batch_size": self.max_batch_size,
            "timeout_ms": self.timeout_ms,
            "enable_jit": self.enable_jit,
            "enable_onnx": self.enable_onnx,
            "onnx_opset": self.onnx_opset,
            "enable_tensorrt": self.enable_tensorrt,
            "runtime_optimization": self.runtime_optimization,
            "fallback_to_cpu": self.fallback_to_cpu,
            "thread_count": self.thread_count,
            "warmup": self.warmup,
            "warmup_iterations": self.warmup_iterations,
            "profiling": self.profiling,
            "batching_strategy": self.batching_strategy,
            "output_streaming": self.output_streaming,
            "debug_mode": self.debug_mode,
            "memory_growth": self.memory_growth,
            "use_platform_accelerator": self.use_platform_accelerator,
            "platform_accelerator_config": self.platform_accelerator_config
        }


class QuantizationConfig(QuantizationConfigBase):
    """Configuration for model quantization"""
    
    def __init__(
        self,
        quantization_type: Union[str, QuantizationType] = QuantizationType.INT8.value,
        quantization_mode: Union[str, QuantizationMode] = QuantizationMode.DYNAMIC_PER_BATCH.value,
        per_channel: bool = False,
        symmetric: bool = True,
        enable_cache: bool = True,
        cache_size: int = 256,
        calibration_samples: int = 100,
        calibration_method: str = "percentile",
        percentile: float = 99.99,
        skip_layers: List[str] = None,
        quantize_weights_only: bool = False,
        quantize_activations: bool = True,
        weight_bits: int = 8,
        activation_bits: int = 8,
        quantize_bias: bool = True,
        bias_bits: int = 32,
        enable_mixed_precision: bool = False,
        mixed_precision_layers: List[str] = None,
        optimize_for: str = "performance",  # "performance", "memory", "balanced"
        custom_quantization_config: Dict = None,
        enable_requantization: bool = False,
        requantization_threshold: float = 0.1
    ):
        # Handle quantization_type as string or enum
        if isinstance(quantization_type, str):
            try:
                self.quantization_type = QuantizationType[quantization_type.upper()].value
            except KeyError:
                self.quantization_type = QuantizationType.INT8.value
        else:
            self.quantization_type = quantization_type
            
        # Handle quantization_mode as string or enum
        if isinstance(quantization_mode, str):
            try:
                self.quantization_mode = QuantizationMode[quantization_mode.upper()].value
            except KeyError:
                self.quantization_mode = QuantizationMode.DYNAMIC_PER_BATCH.value
        else:
            self.quantization_mode = quantization_mode
            
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.calibration_samples = calibration_samples
        self.calibration_method = calibration_method
        self.percentile = percentile
        self.skip_layers = skip_layers or []
        self.quantize_weights_only = quantize_weights_only
        self.quantize_activations = quantize_activations
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.quantize_bias = quantize_bias
        self.bias_bits = bias_bits
        self.enable_mixed_precision = enable_mixed_precision
        self.mixed_precision_layers = mixed_precision_layers or []
        self.optimize_for = optimize_for
        self.custom_quantization_config = custom_quantization_config or {}
        self.enable_requantization = enable_requantization
        self.requantization_threshold = requantization_threshold
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            "quantization_type": self.quantization_type,
            "quantization_mode": self.quantization_mode,
            "per_channel": self.per_channel,
            "symmetric": self.symmetric,
            "enable_cache": self.enable_cache,
            "cache_size": self.cache_size,
            "calibration_samples": self.calibration_samples,
            "calibration_method": self.calibration_method,
            "percentile": self.percentile,
            "skip_layers": self.skip_layers,
            "quantize_weights_only": self.quantize_weights_only,
            "quantize_activations": self.quantize_activations,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "quantize_bias": self.quantize_bias,
            "bias_bits": self.bias_bits,
            "enable_mixed_precision": self.enable_mixed_precision,
            "mixed_precision_layers": self.mixed_precision_layers,
            "optimize_for": self.optimize_for,
            "enable_requantization": self.enable_requantization,
            "requantization_threshold": self.requantization_threshold
        }


class ExplainabilityConfig:
    """Configuration for model explainability"""
    
    def __init__(
        self,
        enable_explainability: bool = True,
        methods: List[str] = None,
        default_method: str = "shap",
        shap_algorithm: str = "auto",
        shap_samples: int = 100,
        lime_samples: int = 1000,
        feature_importance_permutations: int = 10,
        partial_dependence_features: List[str] = None,
        partial_dependence_grid_points: int = 20,
        eli5_permutation_samples: int = 100,
        generate_summary: bool = True,
        generate_plots: bool = True,
        explainability_report_format: str = "html",
        store_explanations: bool = True,
        explanation_cache_size: int = 100,
        enable_counterfactuals: bool = False,
        counterfactual_method: str = "dice",
        custom_explainers: List = None
    ):
        self.enable_explainability = enable_explainability
        self.methods = methods or ["shap", "feature_importance", "partial_dependence"]
        self.default_method = default_method
        self.shap_algorithm = shap_algorithm
        self.shap_samples = shap_samples
        self.lime_samples = lime_samples
        self.feature_importance_permutations = feature_importance_permutations
        self.partial_dependence_features = partial_dependence_features or []
        self.partial_dependence_grid_points = partial_dependence_grid_points
        self.eli5_permutation_samples = eli5_permutation_samples
        self.generate_summary = generate_summary
        self.generate_plots = generate_plots
        self.explainability_report_format = explainability_report_format
        self.store_explanations = store_explanations
        self.explanation_cache_size = explanation_cache_size
        self.enable_counterfactuals = enable_counterfactuals
        self.counterfactual_method = counterfactual_method
        self.custom_explainers = custom_explainers or []
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            "enable_explainability": self.enable_explainability,
            "methods": self.methods,
            "default_method": self.default_method,
            "shap_algorithm": self.shap_algorithm,
            "shap_samples": self.shap_samples,
            "lime_samples": self.lime_samples,
            "feature_importance_permutations": self.feature_importance_permutations,
            "partial_dependence_features": self.partial_dependence_features,
            "partial_dependence_grid_points": self.partial_dependence_grid_points,
            "eli5_permutation_samples": self.eli5_permutation_samples,
            "generate_summary": self.generate_summary,
            "generate_plots": self.generate_plots,
            "explainability_report_format": self.explainability_report_format,
            "store_explanations": self.store_explanations,
            "explanation_cache_size": self.explanation_cache_size,
            "enable_counterfactuals": self.enable_counterfactuals,
            "counterfactual_method": self.counterfactual_method
        }


class MonitoringConfig:
    """Configuration for model monitoring in production"""
    
    def __init__(
        self,
        enable_monitoring: bool = True,
        drift_detection: bool = True,
        drift_detection_method: str = "ks_test",
        drift_threshold: float = 0.05,
        drift_check_interval: str = "1h",
        performance_tracking: bool = True,
        performance_metrics: List[str] = None,
        alert_on_drift: bool = True,
        alert_on_performance_drop: bool = True,
        alert_channels: List[str] = None,
        monitoring_report_interval: str = "1d",
        data_sampling_rate: float = 1.0,
        store_predictions: bool = True,
        prediction_store_size: int = 10000,
        enable_auto_retraining: bool = False,
        retraining_trigger_drift_threshold: float = 0.1,
        retraining_trigger_performance_drop: float = 0.05,
        export_metrics: bool = False,
        metrics_export_format: str = "prometheus",
        custom_monitors: List = None
    ):
        self.enable_monitoring = enable_monitoring
        self.drift_detection = drift_detection
        self.drift_detection_method = drift_detection_method
        self.drift_threshold = drift_threshold
        self.drift_check_interval = drift_check_interval
        self.performance_tracking = performance_tracking
        self.performance_metrics = performance_metrics or ["accuracy", "latency"]
        self.alert_on_drift = alert_on_drift
        self.alert_on_performance_drop = alert_on_performance_drop
        self.alert_channels = alert_channels or ["log"]
        self.monitoring_report_interval = monitoring_report_interval
        self.data_sampling_rate = data_sampling_rate
        self.store_predictions = store_predictions
        self.prediction_store_size = prediction_store_size
        self.enable_auto_retraining = enable_auto_retraining
        self.retraining_trigger_drift_threshold = retraining_trigger_drift_threshold
        self.retraining_trigger_performance_drop = retraining_trigger_performance_drop
        self.export_metrics = export_metrics
        self.metrics_export_format = metrics_export_format
        self.custom_monitors = custom_monitors or []
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            "enable_monitoring": self.enable_monitoring,
            "drift_detection": self.drift_detection,
            "drift_detection_method": self.drift_detection_method,
            "drift_threshold": self.drift_threshold,
            "drift_check_interval": self.drift_check_interval,
            "performance_tracking": self.performance_tracking,
            "performance_metrics": self.performance_metrics,
            "alert_on_drift": self.alert_on_drift,
            "alert_on_performance_drop": self.alert_on_performance_drop,
            "alert_channels": self.alert_channels,
            "monitoring_report_interval": self.monitoring_report_interval,
            "data_sampling_rate": self.data_sampling_rate,
            "store_predictions": self.store_predictions,
            "prediction_store_size": self.prediction_store_size,
            "enable_auto_retraining": self.enable_auto_retraining,
            "retraining_trigger_drift_threshold": self.retraining_trigger_drift_threshold,
            "retraining_trigger_performance_drop": self.retraining_trigger_performance_drop,
            "export_metrics": self.export_metrics,
            "metrics_export_format": self.metrics_export_format
        }


class MLTrainingEngineConfig:
    """Enhanced configuration for the ML Training Engine"""
    
    def __init__(
        self,
        task_type: Union[TaskType, str] = TaskType.CLASSIFICATION,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1,
        cv_folds: int = 5,
        test_size: float = 0.2,
        stratify: bool = True,
        optimization_strategy: Union[OptimizationStrategy, str] = OptimizationStrategy.HYPERX,
        optimization_iterations: int = 50,
        optimization_timeout: Optional[int] = None,
        optimization_metric: Union[str, ModelSelectionCriteria] = None,
        early_stopping: bool = True,
        early_stopping_rounds: int = 10,
        early_stopping_metric: str = None,
        feature_selection: bool = True,
        feature_selection_method: str = "mutual_info",
        feature_selection_k: Optional[int] = None,
        feature_importance_threshold: float = 0.01,
        preprocessing_config: Optional[PreprocessorConfig] = None,
        batch_processing_config: Optional[BatchProcessorConfig] = None,
        inference_config: Optional[InferenceEngineConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        explainability_config: Optional[ExplainabilityConfig] = None,
        monitoring_config: Optional[MonitoringConfig] = None,
        model_path: str = "./models",
        model_registry_url: Optional[str] = None,
        auto_version_models: bool = True,
        experiment_tracking: bool = True,
        experiment_tracking_platform: str = "mlflow",
        experiment_tracking_config: Dict = None,
        use_intel_optimization: bool = True,
        use_gpu: bool = True,
        gpu_memory_fraction: float = 0.8,
        memory_optimization: bool = True,
        enable_distributed: bool = False,
        distributed_strategy: str = "mirrored",
        distributed_config: Dict = None,
        checkpointing: bool = True,
        checkpoint_interval: int = 10,
        checkpoint_path: Optional[str] = None,
        enable_pruning: bool = False,
        auto_ml: Union[bool, AutoMLMode] = AutoMLMode.DISABLED,
        auto_ml_time_budget: Optional[int] = None,
        ensemble_models: bool = False,
        ensemble_method: str = "stacking",
        ensemble_size: int = 3,
        hyperparameter_tuning_cv: bool = True,
        model_selection_criteria: Union[str, ModelSelectionCriteria] = ModelSelectionCriteria.ACCURACY,
        auto_save: bool = True,
        auto_save_on_shutdown: bool = True,
        save_state_on_shutdown: bool = False,
        load_best_model_after_train: bool = True,
        enable_quantization: bool = False,
        enable_model_compression: bool = False,
        compression_method: str = "pruning",
        compute_permutation_importance: bool = True,
        generate_feature_importance_report: bool = True,
        generate_model_summary: bool = True,
        generate_prediction_explanations: bool = False,
        enable_model_export: bool = True,
        export_formats: List[str] = None,
        auto_deploy: bool = False,
        deployment_platform: str = None,
        deployment_config: Dict = None,
        log_level: str = "INFO",
        debug_mode: bool = False,
        enable_telemetry: bool = False,
        backend: str = "sklearn",
        custom_callbacks: List = None,
        enable_data_validation: bool = True,
        enable_security: bool = False,
        security_config: Dict = None,
        metadata: Dict = None
    ):
        # Handle task_type as string or enum
        if isinstance(task_type, str):
            try:
                self.task_type = TaskType[task_type.upper()]
            except KeyError:
                self.task_type = TaskType.CLASSIFICATION
        else:
            self.task_type = task_type
        
        # Handle optimization_strategy as string or enum
        if isinstance(optimization_strategy, str):
            try:
                self.optimization_strategy = OptimizationStrategy[optimization_strategy.upper()]
            except KeyError:
                self.optimization_strategy = OptimizationStrategy.HYPERX
        else:
            self.optimization_strategy = optimization_strategy
            
        # Handle model_selection_criteria as string or enum
        if isinstance(model_selection_criteria, str):
            try:
                self.model_selection_criteria = ModelSelectionCriteria[model_selection_criteria.upper()]
            except KeyError:
                self.model_selection_criteria = ModelSelectionCriteria.ACCURACY
        else:
            self.model_selection_criteria = model_selection_criteria
            
        # Handle auto_ml as bool or enum
        if isinstance(auto_ml, bool):
            self.auto_ml = AutoMLMode.BASIC if auto_ml else AutoMLMode.DISABLED
        else:
            self.auto_ml = auto_ml
            
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.stratify = stratify
        self.optimization_iterations = optimization_iterations
        self.optimization_timeout = optimization_timeout
        self.optimization_metric = optimization_metric
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_metric = early_stopping_metric
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
                debug_mode=debug_mode
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
            
        if explainability_config is None:
            self.explainability_config = ExplainabilityConfig(
                enable_explainability=generate_prediction_explanations,
                methods=["shap", "feature_importance"],
                default_method="shap"
            )
        else:
            self.explainability_config = explainability_config
            
        if monitoring_config is None:
            self.monitoring_config = MonitoringConfig(
                enable_monitoring=True,
                drift_detection=True,
                performance_tracking=True
            )
        else:
            self.monitoring_config = monitoring_config
        
        # Other configuration parameters
        self.model_path = model_path
        self.model_registry_url = model_registry_url
        self.auto_version_models = auto_version_models
        self.experiment_tracking = experiment_tracking
        self.experiment_tracking_platform = experiment_tracking_platform
        self.experiment_tracking_config = experiment_tracking_config or {}
        self.use_intel_optimization = use_intel_optimization
        self.use_gpu = use_gpu
        self.gpu_memory_fraction = gpu_memory_fraction
        self.memory_optimization = memory_optimization
        self.enable_distributed = enable_distributed
        self.distributed_strategy = distributed_strategy
        self.distributed_config = distributed_config or {}
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path or os.path.join(model_path, "checkpoints")
        self.enable_pruning = enable_pruning
        self.auto_ml = auto_ml
        self.auto_ml_time_budget = auto_ml_time_budget
        self.ensemble_models = ensemble_models
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.hyperparameter_tuning_cv = hyperparameter_tuning_cv
        self.auto_save = auto_save
        self.auto_save_on_shutdown = auto_save_on_shutdown
        self.save_state_on_shutdown = save_state_on_shutdown
        self.load_best_model_after_train = load_best_model_after_train
        self.enable_quantization = enable_quantization
        self.enable_model_compression = enable_model_compression
        self.compression_method = compression_method
        self.compute_permutation_importance = compute_permutation_importance
        self.generate_feature_importance_report = generate_feature_importance_report
        self.generate_model_summary = generate_model_summary
        self.generate_prediction_explanations = generate_prediction_explanations
        self.enable_model_export = enable_model_export
        self.export_formats = export_formats or ["sklearn", "onnx"]
        self.auto_deploy = auto_deploy
        self.deployment_platform = deployment_platform
        self.deployment_config = deployment_config or {}
        self.log_level = log_level
        self.debug_mode = debug_mode
        self.enable_telemetry = enable_telemetry
        self.backend = backend
        self.custom_callbacks = custom_callbacks or []
        self.enable_data_validation = enable_data_validation
        self.enable_security = enable_security
        self.security_config = security_config or {}
        self.metadata = metadata or {}
# -----------------------------------------------------------------------------
# Configuration for Device Optimization 
# -----------------------------------------------------------------------------
class OptimizationMode(str, Enum):
    """
    Defines the optimization mode for device configuration.
    """
    BALANCED = "balanced"      # Balance between performance and resource usage
    CONSERVATIVE = "conservative"  # Prioritize stability and minimal resource usage
    PERFORMANCE = "performance"    # Prioritize maximum performance
    FULL_UTILIZATION = "full_utilization"  # Use all available resources
    MEMORY_SAVING = "memory_saving"  # Prioritize memory efficiency