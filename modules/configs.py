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

# Define enums
class TrainingMode(Enum):
    STANDARD = auto()
    INCREMENTAL = auto()
    TRANSFER = auto()
    DISTRIBUTED = auto()
    HYPERPARAMETER_SEARCH = auto()
    AUTO_ML = auto()

class TrainingState(Enum):
    INITIALIZING = auto()
    READY = auto()
    PREPROCESSING = auto()
    FEATURE_ENGINEERING = auto()
    TRAINING = auto()
    EVALUATING = auto()
    OPTIMIZING = auto()
    SAVING = auto()
    PAUSED = auto()
    ERROR = auto()
    STOPPING = auto()
    STOPPED = auto()
    COMPLETED = auto()

class OptimizationStrategy(Enum):
    GRID_SEARCH = auto()
    RANDOM_SEARCH = auto()
    BAYESIAN = auto()
    GENETIC = auto()
    CUSTOM = auto()

class DataSplitStrategy(Enum):
    RANDOM = auto()
    STRATIFIED = auto()
    GROUP = auto()
    TIME_SERIES = auto()
    CUSTOM = auto()

class ModelType(Enum):
    SKLEARN = auto()
    XGBOOST = auto()
    LIGHTGBM = auto()
    CUSTOM = auto()
    ENSEMBLE = auto()

@dataclass
class DatasetConfig:
    """Configuration for dataset handling"""
    train_path: Optional[str] = None
    validation_path: Optional[str] = None
    test_path: Optional[str] = None
    
    # Dataset splitting parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    split_strategy: DataSplitStrategy = DataSplitStrategy.RANDOM
    random_seed: int = 42
    stratify_column: Optional[str] = None
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    
    # Data processing options
    handle_missing: bool = True
    handle_outliers: bool = True
    handle_categorical: bool = True
    handle_imbalance: bool = False
    imbalance_strategy: str = "oversample"  # oversample, undersample, smote, etc.
    
    # Data format 
    csv_separator: str = ","
    encoding: str = "utf-8"
    has_header: bool = True
    
    # In-memory vs out-of-memory processing
    enable_chunking: bool = False
    chunk_size: int = 10000
    
    # Feature columns configuration
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    weight_column: Optional[str] = None
    id_column: Optional[str] = None
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)
    
    # Dataset versioning
    enable_versioning: bool = False
    version: str = "1.0.0"
    compute_dataset_hash: bool = False

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering"""
    enable_feature_selection: bool = False
    feature_selection_method: str = "importance"  # importance, correlation, recursive
    max_features: Optional[int] = None
    min_feature_importance: float = 0.01
    
    enable_polynomial_features: bool = False
    polynomial_degree: int = 2
    interaction_only: bool = True
    
    enable_feature_encoding: bool = True
    encoding_method: str = "auto"  # auto, onehot, label, target, frequency, etc.
    max_onehot_cardinality: int = 10
    
    enable_dimensionality_reduction: bool = False
    dimensionality_reduction_method: str = "pca"  # pca, t-sne, umap, etc.
    target_dimensions: Optional[int] = None
    variance_threshold: float = 0.95
    
    enable_feature_generation: bool = False
    datetime_features: bool = True
    text_features: bool = False
    aggregation_features: bool = False
    
    enable_scaling: bool = True
    scaling_method: str = "standard"  # standard, minmax, robust, etc.
    
    custom_transformers: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelTrainingConfig:
    """Configuration for model training"""
    model_type: ModelType = ModelType.SKLEARN
    model_class: Optional[str] = None  # e.g., "sklearn.ensemble.RandomForestClassifier"
    problem_type: str = "classification"  # classification, regression, multi-label, etc.
    
    # Model hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Training parameters
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_metric: str = "auto"  # auto, accuracy, f1, mse, etc.
    
    # Cross-validation
    enable_cross_validation: bool = False
    cv_folds: int = 5
    cv_strategy: str = "kfold"  # kfold, stratified, group, etc.
    
    # Multi-threading and parallelism
    n_jobs: int = -1
    enable_threading: bool = True
    use_gpu: bool = False

    # Checkpointing
    enable_checkpointing: bool = False
    checkpoint_interval: int = 1  # checkpoint every N epochs
    keep_checkpoint_max: int = 3  # keep last N checkpoints
    
    # Class weights for imbalanced classification
    class_weight: str = "balanced"  # balanced, balanced_subsample, None or custom dict
    
    # Objective function
    objective: Optional[str] = None  # custom objective for XGBoost/LightGBM
    
    # Regularization
    l1_regularization: Optional[float] = None
    l2_regularization: Optional[float] = None
    
    # Custom metrics to monitor
    metrics: List[str] = field(default_factory=list)
    
    # Advanced options
    enable_calibration: bool = False
    calibration_method: str = "isotonic"  # isotonic, sigmoid
    
    # Feature importance
    compute_feature_importance: bool = True
    importance_type: str = "auto"  # gain, split, permutation, etc.

@dataclass
class HyperparameterOptimizationConfig:
    """Configuration for hyperparameter optimization"""
    enabled: bool = False
    strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH
    
    # Optimization parameters
    max_trials: int = 10
    max_time_minutes: Optional[int] = None
    metric: str = "auto"  # auto, accuracy, f1, mse, etc.
    direction: str = "maximize"  # maximize, minimize
    
    # Search space definition
    param_grid: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-validation settings during optimization
    cv_folds: int = 3
    
    # Parallel optimization
    n_parallel_trials: int = 1
    
    # Bayesian optimization settings (for optuna)
    pruning: bool = True
    pruner: str = "median"  # median, hyperband, etc.
    sampler: str = "tpe"  # tpe, random, grid, etc.
    
    # Early stopping for hyperparameter search
    early_stopping: bool = True
    early_stopping_patience: int = 5

@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation"""
    metrics: List[str] = field(default_factory=lambda: ["auto"])
    
    # Classification specific metrics
    classification_threshold: float = 0.5
    average_method: str = "weighted"  # weighted, macro, micro, etc.
    
    # Evaluation datasets
    evaluate_on_train: bool = True
    evaluate_on_validation: bool = True
    evaluate_on_test: bool = True
    
    # Additional evaluation processes
    confusion_matrix: bool = True
    classification_report: bool = True
    roc_curve: bool = True
    precision_recall_curve: bool = True
    calibration_curve: bool = False
    
    # Feature importance evaluation
    compute_permutation_importance: bool = False
    n_repeats: int = 10
    
    # Cross-validation evaluation
    use_cross_validation: bool = False
    cv_folds: int = 5
    
    # Saving evaluation results
    save_predictions: bool = True
    save_evaluation_report: bool = True
    
    # Model explainability
    enable_explainability: bool = False
    explainability_method: str = "shap"  # shap, lime, etc.
    max_samples_to_explain: int = 100

@dataclass
class ModelRegistryConfig:
    """Configuration for model registry and versioning"""
    registry_path: str = MODEL_REGISTRY_PATH
    
    # Model metadata
    model_name: str = "model"
    model_version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Versioning
    auto_version: bool = True
    version_strategy: str = "semver"  # semver, timestamp, incremental, etc.
    
    # Artifacts to save with the model
    save_preprocessor: bool = True
    save_feature_names: bool = True
    save_metrics: bool = True
    save_hyperparameters: bool = True
    save_training_history: bool = True
    save_feature_importance: bool = True
    
    # Model formats
    save_formats: List[str] = field(default_factory=lambda: ["pickle"])
    
    # AWS S3/GCS/Azure Blob integration
    enable_cloud_storage: bool = False
    cloud_provider: str = "aws"  # aws, gcp, azure
    cloud_bucket: str = ""
    cloud_path: str = ""
    
    # Model deployment
    auto_deploy: bool = False
    deployment_target: str = "local"  # local, docker, kubernetes, etc.
    
    # Model lifecycle
    enable_lifecycle_management: bool = False
    archive_old_versions: bool = True
    max_versions_to_keep: int = 5

@dataclass
class CPUTrainingConfig:
    """
    Complete configuration for CPU-optimized training with advanced features.
    
    Configuration parameters are organized by functional category:
    - Training: Core training settings
    - Dataset: Dataset handling and splitting
    - Feature Engineering: Feature transformations and selection
    - Model: Model-specific parameters
    - Optimization: Hyperparameter optimization settings
    - Evaluation: Model evaluation and metrics
    - Registry: Model versioning and storage
    - Resource Management: System resource controls
    - Monitoring: Telemetry and observability
    """
    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Feature engineering configuration
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    
    # Model training configuration
    model_training: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)
    
    # Hyperparameter optimization configuration
    hyperparameter_optimization: HyperparameterOptimizationConfig = field(default_factory=HyperparameterOptimizationConfig)
    
    # Model evaluation configuration
    model_evaluation: ModelEvaluationConfig = field(default_factory=ModelEvaluationConfig)
    
    # Model registry configuration
    model_registry: ModelRegistryConfig = field(default_factory=ModelRegistryConfig)
    
    # Core training settings
    training_mode: TrainingMode = TrainingMode.STANDARD
    random_seed: int = 42
    
    # Resource management
    num_threads: int = field(default_factory=lambda: os.cpu_count() or 4)
    memory_limit_gb: Optional[float] = None
    enable_intel_optimization: bool = True
    use_disk_offloading: bool = False
    temp_directory: str = "./tmp"
    
    # Monitoring and debugging
    enable_monitoring: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"
    monitoring_interval: float = 5.0  # seconds
    
    # Advanced options
    enable_distributed_training: bool = False
    distributed_backend: str = "multiprocessing"  # multiprocessing, dask, ray, etc.
    num_workers: int = 1
    
    enable_auto_resume: bool = True
    resume_from_checkpoint: Optional[str] = None
    
    # Callbacks and hooks
    callbacks: Dict[str, Any] = field(default_factory=dict)
    
    # Development options
    experimental_features: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Set reasonable defaults for CPU thread count
        if self.num_threads <= 0:
            self.num_threads = os.cpu_count() or 4
            
        # Configure model_training.n_jobs based on num_threads if not set
        if self.model_training.n_jobs <= 0:
            self.model_training.n_jobs = self.num_threads
            
        # Create directories if they don't exist
        os.makedirs(self.model_registry.registry_path, exist_ok=True)
        os.makedirs(self.temp_directory, exist_ok=True)
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        
        # For distributed training, adjust worker count
        if self.enable_distributed_training and self.num_workers <= 0:
            self.num_workers = max(1, (os.cpu_count() or 4) - 1)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CPUTrainingConfig':
        """Create config from a dictionary."""
        # Handle nested configurations
        if 'dataset' in config_dict and isinstance(config_dict['dataset'], dict):
            config_dict['dataset'] = DatasetConfig(**config_dict['dataset'])
        
        if 'feature_engineering' in config_dict and isinstance(config_dict['feature_engineering'], dict):
            config_dict['feature_engineering'] = FeatureEngineeringConfig(**config_dict['feature_engineering'])
        
        if 'model_training' in config_dict and isinstance(config_dict['model_training'], dict):
            config_dict['model_training'] = ModelTrainingConfig(**config_dict['model_training'])
        
        if 'hyperparameter_optimization' in config_dict and isinstance(config_dict['hyperparameter_optimization'], dict):
            config_dict['hyperparameter_optimization'] = HyperparameterOptimizationConfig(**config_dict['hyperparameter_optimization'])
        
        if 'model_evaluation' in config_dict and isinstance(config_dict['model_evaluation'], dict):
            config_dict['model_evaluation'] = ModelEvaluationConfig(**config_dict['model_evaluation'])
        
        if 'model_registry' in config_dict and isinstance(config_dict['model_registry'], dict):
            config_dict['model_registry'] = ModelRegistryConfig(**config_dict['model_registry'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'CPUTrainingConfig':
        """Load configuration from a JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to a JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)