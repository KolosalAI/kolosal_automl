from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import os
import numpy as np
from typing import Optional, List, Dict, Tuple, NamedTuple, Callable, Any, Union

# Constants definitions
CHECKPOINT_PATH = "./checkpoints"
MODEL_REGISTRY_PATH = "./model_registry"

# -----------------------------------------------------------------------------
# Quantization Configuration
# -----------------------------------------------------------------------------
from enum import Enum
from typing import List, Dict, Union, Optional
from dataclasses import dataclass, field, asdict
import copy


class QuantizationType(Enum):
    NONE = "none"
    INT8 = "int8"
    UINT8 = "uint8"  # Added the missing UINT8 type
    INT16 = "int16"
    FLOAT16 = "float16"
    MIXED = "mixed"


class QuantizationMode(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    DYNAMIC_PER_BATCH = "dynamic_per_batch"
    DYNAMIC_PER_CHANNEL = "dynamic_per_channel"  # Added the missing mode
    SYMMETRIC = "symmetric"  # Added symmetric mode
    CALIBRATED = "calibrated"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    quantization_type: Union[str, QuantizationType] = QuantizationType.INT8
    quantization_mode: Union[str, QuantizationMode] = QuantizationMode.DYNAMIC_PER_BATCH
    per_channel: bool = False
    symmetric: bool = True
    enable_cache: bool = True
    cache_size: int = 256
    calibration_samples: int = 100
    calibration_method: str = "percentile"
    percentile: float = 99.99
    skip_layers: List[str] = field(default_factory=list)
    quantize_weights_only: bool = False
    quantize_activations: bool = True
    weight_bits: int = 8
    activation_bits: int = 8
    quantize_bias: bool = True
    bias_bits: int = 32
    enable_mixed_precision: bool = False
    mixed_precision_layers: List[str] = field(default_factory=list)
    optimize_for: str = "performance"  # "performance", "memory", "balanced"
    custom_quantization_config: Dict = field(default_factory=dict)
    enable_requantization: bool = False
    requantization_threshold: float = 0.1
    use_percentile: bool = False
    min_percentile: float = 0.1
    max_percentile: float = 99.9
    error_on_nan: bool = False
    error_on_inf: bool = False
    outlier_threshold: Optional[float] = None  # For handling outliers
    num_bits: int = 8  # Allow custom bit widths
    optimize_memory: bool = True  # Enable memory optimization
    buffer_size: int = 0  # 0 means no buffer
    
    def __post_init__(self):
        # Handle quantization_type as string or enum
        if isinstance(self.quantization_type, str):
            try:
                self.quantization_type = QuantizationType[self.quantization_type.upper()]
            except KeyError:
                self.quantization_type = QuantizationType.INT8
                
        # Handle quantization_mode as string or enum
        if isinstance(self.quantization_mode, str):
            try:
                self.quantization_mode = QuantizationMode[self.quantization_mode.upper()]
            except KeyError:
                self.quantization_mode = QuantizationMode.DYNAMIC_PER_BATCH
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        config_dict = asdict(self)
        
        # Convert enum values to their string representation for serialization
        if isinstance(config_dict["quantization_type"], QuantizationType):
            config_dict["quantization_type"] = config_dict["quantization_type"].value
            
        if isinstance(config_dict["quantization_mode"], QuantizationMode):
            config_dict["quantization_mode"] = config_dict["quantization_mode"].value
            
        # Ensure custom_quantization_config is properly serialized
        if config_dict["custom_quantization_config"] is None:
            config_dict["custom_quantization_config"] = {}
            
        # Make a deep copy to avoid modifying the original
        return copy.deepcopy(config_dict)
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'QuantizationConfig':
        """Create a QuantizationConfig instance from a dictionary"""
        # Make a copy to avoid modifying the input
        config_copy = copy.deepcopy(config_dict)
        
        # Convert string values to enum instances if needed
        if "quantization_type" in config_copy and isinstance(config_copy["quantization_type"], str):
            try:
                config_copy["quantization_type"] = QuantizationType[config_copy["quantization_type"].upper()]
            except KeyError:
                # Try to match by value
                for qt in QuantizationType:
                    if qt.value == config_copy["quantization_type"]:
                        config_copy["quantization_type"] = qt
                        break
        
        if "quantization_mode" in config_copy and isinstance(config_copy["quantization_mode"], str):
            try:
                config_copy["quantization_mode"] = QuantizationMode[config_copy["quantization_mode"].upper()]
            except KeyError:
                # Try to match by value
                for qm in QuantizationMode:
                    if qm.value == config_copy["quantization_mode"]:
                        config_copy["quantization_mode"] = qm
                        break
                        
        # Filter out any keys that aren't in the dataclass
        valid_keys = {f.name for f in field(cls)}
        filtered_dict = {k: v for k, v in config_copy.items() if k in valid_keys}
        
        return cls(**filtered_dict)

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
    """Configuration for batch processing"""
    initial_batch_size: int = 100
    min_batch_size: int = 50
    max_batch_size: int = 200
    max_queue_size: int = 1000
    batch_timeout: float = 1.0
    num_workers: int = 4
    adaptive_batching: bool = True
    batch_allocation_strategy: str = "dynamic"
    enable_priority_queue: bool = True
    priority_levels: int = 3
    enable_monitoring: bool = True
    monitoring_interval: float = 2.0
    enable_memory_optimization: bool = True
    enable_prefetching: bool = True
    prefetch_batches: int = 2
    checkpoint_batches: bool = False
    checkpoint_interval: int = 100
    error_handling: str = "retry"
    max_retries: int = 3
    retry_delay: float = 0.5
    distributed_processing: bool = False
    resource_allocation: Dict = None
    item_timeout: float = 10.0  # Max time to wait for single item processing (seconds)
    min_batch_interval: float = 0.0  # Min time between batch processing (seconds)
    processing_strategy: BatchProcessingStrategy = BatchProcessingStrategy.ADAPTIVE
    enable_adaptive_batching: bool = True
    reduce_batch_on_failure: bool = True
    max_batch_memory_mb: Optional[float] = None  # Max memory per batch in MB
    gc_batch_threshold: int = 32  # Run GC after processing batches larger than this
    monitoring_window: int = 100  # Number of batches to keep statistics for
    max_workers: int = 4
    enable_health_monitoring: bool = True
    health_check_interval: float = 5.0  # Seconds between health checks
    memory_warning_threshold: float = 70.0  # Memory usage percentage to trigger warning
    memory_critical_threshold: float = 85.0  # Memory usage percentage to trigger critical actions
    queue_warning_threshold: int = 100  # Queue size to trigger warning
    queue_critical_threshold: int = 500  # Queue size to trigger critical actions
    debug_mode: bool = False
    
    def __post_init__(self):
        if self.resource_allocation is None:
            self.resource_allocation = {}
    
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
            "resource_allocation": self.resource_allocation,
            "item_timeout": self.item_timeout,
            "min_batch_interval": self.min_batch_interval,
            "processing_strategy": self.processing_strategy.name,
            "enable_adaptive_batching": self.enable_adaptive_batching,
            "reduce_batch_on_failure": self.reduce_batch_on_failure,
            "max_batch_memory_mb": self.max_batch_memory_mb,
            "gc_batch_threshold": self.gc_batch_threshold,
            "monitoring_window": self.monitoring_window,
            "max_workers": self.max_workers,
            "enable_health_monitoring": self.enable_health_monitoring,
            "health_check_interval": self.health_check_interval,
            "memory_warning_threshold": self.memory_warning_threshold,
            "memory_critical_threshold": self.memory_critical_threshold,
            "queue_warning_threshold": self.queue_warning_threshold,
            "queue_critical_threshold": self.queue_critical_threshold,
            "debug_mode": self.debug_mode
        }

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
class PreprocessorConfig:
    """Configuration for data preprocessing"""
    # Normalization settings
    normalization: Union[NormalizationType, str] = NormalizationType.STANDARD
    robust_percentiles: Tuple[float, float] = (25.0, 75.0)
    
    # Data handling settings
    handle_nan: bool = True
    handle_inf: bool = True
    nan_strategy: str = "mean"  # Options: "mean", "median", "most_frequent", "constant"
    nan_fill_value: float = 0.0  # Value to use for constant strategy
    inf_strategy: str = "max_value"  # Options: "max_value", "median", "constant"
    pos_inf_fill_value: float = np.nan  # Value to use for positive infinity
    neg_inf_fill_value: float = np.nan  # Value to use for negative infinity
    inf_buffer: float = 1.0  # Multiplier for max/min when replacing infinities
    copy_X: bool = True  # Whether to copy data for transformations
    
    # Outlier detection and handling settings
    detect_outliers: bool = True
    outlier_method: str = "iqr"  # Options: "iqr", "zscore", "percentile", "isolation_forest"
    outlier_handling: str = "clip"  # Options: "clip", "remove", "winsorize", "mean"
    outlier_contamination: float = 0.05
    outlier_iqr_multiplier: float = 1.5  # Multiplier for IQR method
    outlier_zscore_threshold: float = 3.0  # Threshold for zscore method
    outlier_percentiles: Tuple[float, float] = (1.0, 99.0)  # Percentiles for outlier detection
    feature_outlier_mask: Optional[List[bool]] = None  # Mask of which features to apply outlier detection to
    
    # Clipping settings
    clip_values: bool = False
    clip_min: float = float('-inf')
    clip_max: float = float('inf')
    
    # Scaling settings
    scale_shift: bool = False  # Whether to apply an offset after scaling
    scale_offset: float = 0.0  # Offset to apply after scaling
    
    # Custom functions
    custom_center_func: Optional[Callable] = None  # Custom function to compute centering values
    custom_scale_func: Optional[Callable] = None  # Custom function to compute scaling values
    custom_normalization_fn: Optional[Callable] = None  # Custom normalization function
    custom_transform_fn: Optional[Callable] = None  # Custom transform function
    custom_inverse_fn: Optional[Callable] = None  # Custom inverse function
    
    # Categorical features
    categorical_encoding: str = "one_hot"
    categorical_max_categories: int = 20
    
    # Feature engineering
    auto_feature_selection: bool = True
    numeric_transformations: List[str] = None
    text_vectorization: Optional[str] = "tfidf"
    text_max_features: int = 5000
    dimension_reduction: Optional[str] = None
    dimension_reduction_target: Optional[int] = None
    datetime_features: bool = True
    image_preprocessing: Optional[Dict] = None
    handle_imbalance: bool = False
    imbalance_strategy: str = "smote"
    feature_interaction: bool = False
    feature_interaction_max: int = 10
    custom_transformers: List = None
    transformation_pipeline: List = None
    
    # Performance settings
    parallel_processing: bool = True
    cache_preprocessing: bool = True
    verbosity: int = 1
    enable_input_validation: bool = True
    input_size_limit: Optional[int] = None  # Max number of samples
    n_jobs: int = -1  # -1 for all cores
    chunk_size: Optional[int] = None  # Process large data in chunks of this size
    cache_enabled: bool = True
    cache_size: int = 128  # LRU cache size
    
    # Technical settings
    dtype: np.dtype = np.float64
    epsilon: float = 1e-10  # Small value to avoid division by zero
    debug_mode: bool = False
    version: str = "1.0.0"
    
    def __post_init__(self):
        # Handle normalization as string or enum
        if isinstance(self.normalization, str):
            try:
                self.normalization = NormalizationType[self.normalization.upper()]
            except KeyError:
                self.normalization = NormalizationType.STANDARD
                
        # Initialize empty lists
        if self.numeric_transformations is None:
            self.numeric_transformations = []
        if self.image_preprocessing is None:
            self.image_preprocessing = {}
        if self.custom_transformers is None:
            self.custom_transformers = []
        if self.transformation_pipeline is None:
            self.transformation_pipeline = []
            
        # Convert feature outlier mask if needed
        if self.feature_outlier_mask is None:
            self.feature_outlier_mask = []
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        # Use dataclasses.asdict to convert to dictionary
        config_dict = asdict(self)
        
        # Handle special types
        if isinstance(self.normalization, NormalizationType):
            config_dict['normalization'] = self.normalization.value
            
        if isinstance(self.dtype, np.dtype):
            config_dict['dtype'] = str(self.dtype)
            
        # Remove non-serializable callables
        config_dict['custom_normalization_fn'] = None
        config_dict['custom_transform_fn'] = None
        config_dict['custom_center_func'] = None
        config_dict['custom_scale_func'] = None
        config_dict['custom_inverse_fn'] = None
        
        return config_dict
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PreprocessorConfig':
        """Create config from dictionary"""
        # Remove any unknown keys
        valid_fields = {field.name for field in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        # Handle special types
        if 'dtype' in filtered_dict and isinstance(filtered_dict['dtype'], str):
            filtered_dict['dtype'] = np.dtype(filtered_dict['dtype'])
            
        # Create instance
        return cls(**filtered_dict)
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

class OptimizationMode(str, Enum):
    """
    Defines the optimization mode for device configuration.
    """
    BALANCED = "balanced"      # Balance between performance and resource usage
    CONSERVATIVE = "conservative"  # Prioritize stability and minimal resource usage
    PERFORMANCE = "performance"    # Prioritize maximum performance
    FULL_UTILIZATION = "full_utilization"  # Use all available resources
    MEMORY_SAVING = "memory_saving"  # Prioritize memory efficiency

@dataclass
class InferenceEngineConfig:
    """Configuration for the inference engine"""
    enable_intel_optimization: bool = True
    enable_batching: bool = True
    enable_quantization: bool = True
    model_cache_size: int = 5
    model_precision: str = "fp32"
    max_batch_size: int = 64
    timeout_ms: int = 100
    enable_jit: bool = True
    enable_onnx: bool = False
    onnx_opset: int = 13
    enable_tensorrt: bool = False
    runtime_optimization: bool = True
    fallback_to_cpu: bool = True
    thread_count: int = 0
    warmup: bool = True
    warmup_iterations: int = 10
    profiling: bool = False
    batching_strategy: str = "dynamic"
    output_streaming: bool = False
    debug_mode: bool = False
    memory_growth: bool = True
    custom_ops: List = None
    use_platform_accelerator: bool = True
    platform_accelerator_config: Dict = None
    # From base class
    model_version: str = "1.0"
    num_threads: int = 4
    set_cpu_affinity: bool = False
    enable_model_quantization: bool = False
    enable_input_quantization: bool = False
    quantization_dtype: str = "int8"  # e.g., "int8" or "float16"
    quantization_config: Optional[QuantizationConfig] = None
    enable_request_deduplication: bool = True
    max_cache_entries: int = 1000
    cache_ttl_seconds: int = 300  # 5 minutes
    monitoring_window: int = 100
    enable_monitoring: bool = True
    monitoring_interval: float = 10.0  # seconds between checks
    throttle_on_high_cpu: bool = True
    cpu_threshold_percent: float = 90.0  # CPU usage threshold in percent
    memory_high_watermark_mb: float = 1024.0  # memory high watermark in MB
    memory_limit_gb: Optional[float] = None  # optional absolute limit in GB
    batch_timeout: float = 0.1  # seconds to wait for additional requests
    max_concurrent_requests: int = 8
    initial_batch_size: int = 16
    min_batch_size: int = 1
    enable_adaptive_batching: bool = True
    enable_memory_optimization: bool = True
    enable_feature_scaling: bool = False
    enable_warmup: bool = True
    enable_quantization_aware_inference: bool = False
    enable_throttling: bool = False
    
    def __post_init__(self):
        if self.custom_ops is None:
            self.custom_ops = []
        if self.platform_accelerator_config is None:
            self.platform_accelerator_config = {}
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        config_dict = {
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
            "custom_ops": self.custom_ops,
            "use_platform_accelerator": self.use_platform_accelerator,
            "platform_accelerator_config": self.platform_accelerator_config,
            "model_version": self.model_version,
            "num_threads": self.num_threads,
            "set_cpu_affinity": self.set_cpu_affinity,
            "enable_model_quantization": self.enable_model_quantization,
            "enable_input_quantization": self.enable_input_quantization,
            "quantization_dtype": self.quantization_dtype,
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
            "batch_timeout": self.batch_timeout,
            "max_concurrent_requests": self.max_concurrent_requests,
            "initial_batch_size": self.initial_batch_size,
            "min_batch_size": self.min_batch_size,
            "enable_adaptive_batching": self.enable_adaptive_batching,
            "enable_memory_optimization": self.enable_memory_optimization,
            "enable_feature_scaling": self.enable_feature_scaling,
            "enable_warmup": self.enable_warmup,
            "enable_quantization_aware_inference": self.enable_quantization_aware_inference,
            "enable_throttling": self.enable_throttling
        }
        
        # Add quantization config if present
        if self.quantization_config:
            config_dict["quantization_config"] = self.quantization_config.to_dict()
        
        return config_dict

# -----------------------------------------------------------------------------
# Configuration for Training Engine
# -----------------------------------------------------------------------------
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
            "counterfactual_method": self.counterfactual_method,
            "custom_explainers": self.custom_explainers
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
        monitoring_interval: str = "1h",
        performance_tracking: bool = True,
        performance_metrics: List[str] = None,
        performance_threshold: float = 0.1,  # Added this parameter
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
        self.monitoring_interval = monitoring_interval
        self.performance_tracking = performance_tracking
        self.performance_metrics = performance_metrics or ["accuracy", "latency"]
        self.performance_threshold = performance_threshold  # Initialize the new parameter
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
            "monitoring_interval": self.monitoring_interval,
            "performance_tracking": self.performance_tracking,
            "performance_metrics": self.performance_metrics,
            "performance_threshold": self.performance_threshold,  # Add to dictionary
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
            "metrics_export_format": self.metrics_export_format,
            "custom_monitors": self.custom_monitors
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
        self.preprocessing_config = preprocessing_config or PreprocessorConfig(
            normalization=NormalizationType.STANDARD,
            handle_nan=True,
            handle_inf=True,
            detect_outliers=True,
            parallel_processing=True
        )
            
        self.batch_processing_config = batch_processing_config or BatchProcessorConfig(
            initial_batch_size=100,
            min_batch_size=50,
            max_batch_size=200,
            max_queue_size=1000,
            batch_timeout=1.0,
            enable_priority_queue=True,
            enable_monitoring=True,
            enable_memory_optimization=True
        )
            
        self.inference_config = inference_config or InferenceEngineConfig(
            enable_intel_optimization=use_intel_optimization,
            enable_batching=True,
            enable_quantization=True,
            debug_mode=debug_mode
        )
            
        self.quantization_config = quantization_config or QuantizationConfig(
            quantization_type=QuantizationType.INT8.value,
            quantization_mode=QuantizationMode.DYNAMIC_PER_BATCH.value,
            enable_cache=True,
            cache_size=256
        )
            
        self.explainability_config = explainability_config or ExplainabilityConfig(
            enable_explainability=generate_prediction_explanations,
            methods=["shap", "feature_importance"],
            default_method="shap"
        )
            
        self.monitoring_config = monitoring_config or MonitoringConfig(
            enable_monitoring=True,
            drift_detection=True,
            performance_tracking=True
        )
        
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
        
    def to_dict(self) -> Dict:
        """Convert the full configuration to a dictionary for serialization"""
        config_dict = {
            "task_type": self.task_type.value,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "cv_folds": self.cv_folds,
            "test_size": self.test_size,
            "stratify": self.stratify,
            "optimization_strategy": self.optimization_strategy.value,
            "optimization_iterations": self.optimization_iterations,
            "optimization_timeout": self.optimization_timeout,
            "early_stopping": self.early_stopping,
            "early_stopping_rounds": self.early_stopping_rounds,
            "early_stopping_metric": self.early_stopping_metric,
            "feature_selection": self.feature_selection,
            "feature_selection_method": self.feature_selection_method,
            "feature_selection_k": self.feature_selection_k,
            "feature_importance_threshold": self.feature_importance_threshold,
            "model_path": self.model_path,
            "model_registry_url": self.model_registry_url,
            "auto_version_models": self.auto_version_models,
            "experiment_tracking": self.experiment_tracking,
            "experiment_tracking_platform": self.experiment_tracking_platform,
            "experiment_tracking_config": self.experiment_tracking_config,
            "use_intel_optimization": self.use_intel_optimization,
            "use_gpu": self.use_gpu,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "memory_optimization": self.memory_optimization,
            "enable_distributed": self.enable_distributed,
            "distributed_strategy": self.distributed_strategy,
            "distributed_config": self.distributed_config,
            "checkpointing": self.checkpointing,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_path": self.checkpoint_path,
            "enable_pruning": self.enable_pruning,
            "auto_ml": self.auto_ml.value,
            "auto_ml_time_budget": self.auto_ml_time_budget,
            "ensemble_models": self.ensemble_models,
            "ensemble_method": self.ensemble_method,
            "ensemble_size": self.ensemble_size,
            "hyperparameter_tuning_cv": self.hyperparameter_tuning_cv,
            "model_selection_criteria": self.model_selection_criteria.value,
            "auto_save": self.auto_save,
            "auto_save_on_shutdown": self.auto_save_on_shutdown,
            "save_state_on_shutdown": self.save_state_on_shutdown,
            "load_best_model_after_train": self.load_best_model_after_train,
            "enable_quantization": self.enable_quantization,
            "enable_model_compression": self.enable_model_compression,
            "compression_method": self.compression_method,
            "compute_permutation_importance": self.compute_permutation_importance,
            "generate_feature_importance_report": self.generate_feature_importance_report,
            "generate_model_summary": self.generate_model_summary,
            "generate_prediction_explanations": self.generate_prediction_explanations,
            "enable_model_export": self.enable_model_export,
            "export_formats": self.export_formats,
            "auto_deploy": self.auto_deploy,
            "deployment_platform": self.deployment_platform,
            "deployment_config": self.deployment_config,
            "log_level": self.log_level,
            "debug_mode": self.debug_mode,
            "enable_telemetry": self.enable_telemetry,
            "backend": self.backend,
            "enable_data_validation": self.enable_data_validation,
            "enable_security": self.enable_security,
            "security_config": self.security_config,
            "metadata": self.metadata,
        }
        
        # Add nested configurations
        if self.optimization_metric:
            if isinstance(self.optimization_metric, ModelSelectionCriteria):
                config_dict["optimization_metric"] = self.optimization_metric.value
            else:
                config_dict["optimization_metric"] = self.optimization_metric
                
        # Convert nested config objects
        config_dict["preprocessing_config"] = self.preprocessing_config.to_dict()
        config_dict["batch_processing_config"] = self.batch_processing_config.to_dict()
        config_dict["inference_config"] = self.inference_config.to_dict()
        config_dict["quantization_config"] = self.quantization_config.to_dict()
        config_dict["explainability_config"] = self.explainability_config.to_dict()
        config_dict["monitoring_config"] = self.monitoring_config.to_dict()
        
        return config_dict