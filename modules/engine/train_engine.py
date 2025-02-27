import os
import sys
import time
import logging
import pickle
import hashlib
import threading
import warnings
import gc
import json
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, TypeVar, Generic, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta
from functools import wraps, lru_cache, partial
from contextlib import contextmanager
import traceback
import signal
import psutil
from collections import deque, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future

# Optional imports for specific optimizations
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import sklearn
    from sklearn.model_selection import (
        train_test_split, KFold, StratifiedKFold, GroupKFold, 
        cross_val_score, GridSearchCV, RandomizedSearchCV
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, mean_squared_error, r2_score
    )
    SKLEARN_AVAILABLE = True
    # Try to enable Intel optimizations for scikit-learn
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        SKLEARN_OPTIMIZED = True
    except ImportError:
        SKLEARN_OPTIMIZED = False
except ImportError:
    SKLEARN_AVAILABLE = False
    SKLEARN_OPTIMIZED = False

try:
    # Check for Intel's Python Distribution
    import intel
    INTEL_PYTHON = True
except ImportError:
    INTEL_PYTHON = False

try:
    import mkl
    MKL_AVAILABLE = True
except ImportError:
    MKL_AVAILABLE = False

# Optional ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import local modules (from inference engine)
from data_preprocessor import DataPreprocessor, PreprocessorConfig, NormalizationType
from configs import QuantizationConfig

# Define constants
MODEL_REGISTRY_PATH = os.environ.get("MODEL_REGISTRY_PATH", "./models")
DEFAULT_LOG_PATH = os.environ.get("LOG_PATH", "./logs")
DEFAULT_DATA_PATH = os.environ.get("DATA_PATH", "./data")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "./checkpoints")
MAX_RETRY_ATTEMPTS = 3
MEMORY_CHECK_INTERVAL = 30  # seconds

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


class TrainingMetrics:
    """Class to track and report model training performance metrics with thread safety."""
    
    def __init__(self):
        self.lock = threading.RLock()
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            # Training progress
            self.current_epoch = 0
            self.total_epochs = 0
            self.start_time = None
            self.end_time = None
            
            # Performance metrics
            self.train_metrics = {}
            self.validation_metrics = {}
            self.test_metrics = {}
            
            # Training history for epochs
            self.history = defaultdict(list)
            
            # Hyperparameter optimization tracking
            self.hpo_trials = []
            self.best_hpo_trial = None
            
            # Resource usage tracking
            self.memory_usage = []
            self.cpu_usage = []
            
            # Feature importance
            self.feature_importance = {}
            
            # Cross-validation results
            self.cv_results = None
            
            # Preprocessing stats
            self.preprocessing_stats = {}
            
            # Errors and warnings
            self.error_count = 0
            self.warning_count = 0
            self.errors = []
            self.warnings = []
    
    def start_training(self, total_epochs: int) -> None:
        """Record the start of training."""
        with self.lock:
            self.start_time = time.time()
            self.total_epochs = total_epochs
            self.current_epoch = 0
    
    def end_training(self) -> None:
        """Record the end of training."""
        with self.lock:
            self.end_time = time.time()
    
    def update_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                    validation_metrics: Optional[Dict[str, float]] = None) -> None:
        """Update metrics for an epoch."""
        with self.lock:
            self.current_epoch = epoch
            
            # Store current metrics
            self.train_metrics = train_metrics
            if validation_metrics:
                self.validation_metrics = validation_metrics
            
            # Update history
            for k, v in train_metrics.items():
                self.history[f"train_{k}"].append(v)
            
            if validation_metrics:
                for k, v in validation_metrics.items():
                    self.history[f"val_{k}"].append(v)
            
            # Add epoch to history
            self.history["epoch"].append(epoch)
            
            # Add timestamp
            self.history["timestamp"].append(time.time())
    
    def set_test_metrics(self, test_metrics: Dict[str, float]) -> None:
        """Set metrics from final model evaluation on test set."""
        with self.lock:
            self.test_metrics = test_metrics
    
    def add_hpo_trial(self, trial_id: str, params: Dict[str, Any], 
                     metrics: Dict[str, float], duration: float) -> None:
        """Record a hyperparameter optimization trial."""
        with self.lock:
            trial_info = {
                "trial_id": trial_id,
                "params": params,
                "metrics": metrics,
                "duration": duration,
                "timestamp": time.time()
            }
            self.hpo_trials.append(trial_info)
            
            # Update best trial if this is the first or better than current best
            primary_metric = next(iter(metrics.keys())) if metrics else None
            
            if primary_metric and (
                self.best_hpo_trial is None or 
                metrics[primary_metric] > self.best_hpo_trial["metrics"][primary_metric]
            ):
                self.best_hpo_trial = trial_info
    
    def set_feature_importance(self, importance_dict: Dict[str, float]) -> None:
        """Set feature importance values."""
        with self.lock:
            self.feature_importance = importance_dict
    
    def set_cv_results(self, cv_results: Dict[str, List[float]]) -> None:
        """Set cross-validation results."""
        with self.lock:
            self.cv_results = cv_results
    
    def record_resource_usage(self, memory_mb: float, cpu_percent: float) -> None:
        """Record system resource usage."""
        with self.lock:
            self.memory_usage.append((time.time(), memory_mb))
            self.cpu_usage.append((time.time(), cpu_percent))
    
    def set_preprocessing_stats(self, stats: Dict[str, Any]) -> None:
        """Set statistics from data preprocessing."""
        with self.lock:
            self.preprocessing_stats = stats
    
    def record_error(self, error_msg: str) -> None:
        """Record an error that occurred during training."""
        with self.lock:
            self.error_count += 1
            self.errors.append({
                "timestamp": time.time(),
                "message": error_msg
            })
    
    def record_warning(self, warning_msg: str) -> None:
        """Record a warning that occurred during training."""
        with self.lock:
            self.warning_count += 1
            self.warnings.append({
                "timestamp": time.time(),
                "message": warning_msg
            })
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current training progress information."""
        with self.lock:
            progress = {
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "progress_percent": (self.current_epoch / max(1, self.total_epochs)) * 100
            }
            
            # Calculate time stats if training has started
            if self.start_time:
                elapsed = time.time() - self.start_time
                progress["elapsed_time"] = elapsed
                
                # Estimate remaining time if we have at least one epoch
                if self.current_epoch > 0:
                    time_per_epoch = elapsed / self.current_epoch
                    remaining_epochs = self.total_epochs - self.current_epoch
                    progress["estimated_remaining_time"] = time_per_epoch * remaining_epochs
                    progress["estimated_total_time"] = time_per_epoch * self.total_epochs
            
            return progress
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics."""
        with self.lock:
            metrics = {
                "train": self.train_metrics,
                "validation": self.validation_metrics,
                "test": self.test_metrics,
                "epoch": self.current_epoch
            }
            
            # Add performance metrics if available
            if self.history and self.history.get("epoch"):
                # Get metrics for current epoch
                for key in self.history:
                    if key not in ["epoch", "timestamp"] and self.history[key]:
                        metrics[key] = self.history[key][-1]
            
            return metrics
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get the best metrics achieved during training."""
        with self.lock:
            best_metrics = {}
            
            # Find best validation metrics if available
            for key in self.history:
                if key.startswith("val_") and self.history[key]:
                    # For metrics where higher is better (accuracy, f1, etc.)
                    if any(metric in key.lower() for metric in 
                          ["accuracy", "precision", "recall", "f1", "auc", "r2"]):
                        best_idx = np.argmax(self.history[key])
                        best_value = max(self.history[key])
                        best_metrics[key] = best_value
                    
                    # For metrics where lower is better (loss, error, etc.)
                    elif any(metric in key.lower() for metric in 
                            ["loss", "error", "mse", "mae", "rmse"]):
                        best_idx = np.argmin(self.history[key])
                        best_value = min(self.history[key])
                        best_metrics[key] = best_value
            
            # Add corresponding train metrics from the same epoch if available
            if best_metrics and "val_" in next(iter(best_metrics.keys())) and best_idx is not None:
                for key in self.history:
                    if key.startswith("train_") and len(self.history[key]) > best_idx:
                        best_metrics[key] = self.history[key][best_idx]
                        
                # Add epoch number
                if len(self.history["epoch"]) > best_idx:
                    best_metrics["best_epoch"] = self.history["epoch"][best_idx]
            
            return best_metrics
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics and statistics."""
        with self.lock:
            return {
                "training_history": dict(self.history),
                "test_metrics": self.test_metrics,
                "feature_importance": self.feature_importance,
                "cv_results": self.cv_results,
                "hpo_trials": self.hpo_trials,
                "best_hpo_trial": self.best_hpo_trial,
                "preprocessing_stats": self.preprocessing_stats,
                "resource_usage": {
                    "memory": self.memory_usage,
                    "cpu": self.cpu_usage
                },
                "errors": {
                    "count": self.error_count,
                    "details": self.errors
                },
                "warnings": {
                    "count": self.warning_count,
                    "details": self.warnings
                },
                "timing": {
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                    "total_duration": (self.end_time - self.start_time) if self.end_time else None
                }
            }


class ModelCheckpoint:
    """Manages model checkpointing during training."""
    
    def __init__(self, 
                checkpoint_dir: str = CHECKPOINT_PATH,
                max_checkpoints: int = 3, 
                save_best_only: bool = True,
                metric: str = "val_loss", 
                mode: str = "min",
                verbose: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of recent checkpoints to keep
            save_best_only: Whether to save only when monitored metric improves
            metric: Metric to monitor for improvement
            mode: 'min' or 'max' to determine if higher or lower values are better
            verbose: Whether to print checkpoint messages
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric = metric
        self.mode = mode
        self.verbose = verbose
        
        # Ensure directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Track metrics for determining best model
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_files = []
        
        # Load existing checkpoints list if present
        self._load_checkpoint_metadata()
    
    def _load_checkpoint_metadata(self) -> None:
        """Load metadata about existing checkpoints."""
        metadata_path = os.path.join(self.checkpoint_dir, "checkpoint_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                self.checkpoint_files = metadata.get('checkpoint_files', [])
                self.best_value = metadata.get('best_value', 
                                             float('inf') if self.mode == 'min' else float('-inf'))
            except Exception as e:
                print(f"Error loading checkpoint metadata: {e}")
                self.checkpoint_files = []
    
    def _save_checkpoint_metadata(self) -> None:
        """Save metadata about existing checkpoints."""
        metadata_path = os.path.join(self.checkpoint_dir, "checkpoint_metadata.json")
        try:
            metadata = {
                'checkpoint_files': self.checkpoint_files,
                'best_value': self.best_value,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving checkpoint metadata: {e}")
    
    def save(self, model, epoch: int, metrics: Dict[str, float]) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: The model to save
            epoch: Current epoch number
            metrics: Dictionary of metrics
            
        Returns:
            Path to the saved checkpoint
        """
        # Skip if save_best_only and metric is not present or not improved
        if self.save_best_only:
            if self.metric not in metrics:
                if self.verbose:
                    print(f"Warning: Metric '{self.metric}' not found in metrics")
                return None
            
            current_value = metrics[self.metric]
            if (self.mode == 'min' and current_value >= self.best_value) or \
               (self.mode == 'max' and current_value <= self.best_value):
                return None
            
            # Update best value
            self.best_value = current_value
        
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch:03d}_{timestamp}.pkl"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save the model
        try:
            # Create metadata to save with model
            checkpoint_data = {
                'model': model,
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': timestamp
            }
            
            # Use joblib if available for better performance, otherwise pickle
            if JOBLIB_AVAILABLE:
                joblib.dump(checkpoint_data, checkpoint_path)
            else:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
            
            # Add to checkpoint files list
            self.checkpoint_files.append({
                'filename': checkpoint_name,
                'path': checkpoint_path,
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': timestamp
            })
            
            # Keep only max_checkpoints
            if len(self.checkpoint_files) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_files.pop(0)
                old_path = old_checkpoint['path']
                if os.path.exists(old_path):
                    os.remove(old_path)
            
            # Update metadata
            self._save_checkpoint_metadata()
            
            if self.verbose:
                print(f"Saved checkpoint: {checkpoint_path}")
                
            return checkpoint_path
        
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None
    
    def load_latest(self) -> Dict[str, Any]:
        """
        Load the latest checkpoint.
        
        Returns:
            Dictionary with model and metadata, or None if no checkpoints
        """
        if not self.checkpoint_files:
            return None
        
        # Get the latest checkpoint
        latest_checkpoint = self.checkpoint_files[-1]
        checkpoint_path = latest_checkpoint['path']
        
        return self.load_checkpoint(checkpoint_path)
    
    def load_best(self) -> Dict[str, Any]:
        """
        Load the best checkpoint based on monitored metric.
        
        Returns:
            Dictionary with model and metadata, or None if no checkpoints
        """
        if not self.checkpoint_files:
            return None
        
        # Find the best checkpoint
        if self.mode == 'min':
            best_idx = np.argmin([c['metrics'].get(self.metric, float('inf')) 
                                 for c in self.checkpoint_files])
        else:
            best_idx = np.argmax([c['metrics'].get(self.metric, float('-inf')) 
                                 for c in self.checkpoint_files])
        
        best_checkpoint = self.checkpoint_files[best_idx]
        checkpoint_path = best_checkpoint['path']
        
        return self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a specific checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary with model and metadata, or None if error
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            # Load the checkpoint
            if JOBLIB_AVAILABLE:
                checkpoint_data = joblib.load(checkpoint_path)
            else:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
            
            if self.verbose:
                print(f"Loaded checkpoint: {checkpoint_path}")
                
            return checkpoint_data
        
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """
        Get information about available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        # Return a copy of checkpoint files without the full paths
        return [{k: v for k, v in c.items() if k != 'path'} 
                for c in self.checkpoint_files]


class TrainingEngine:
    """
    High-performance CPU-optimized training engine with Intel optimizations,
    comprehensive monitoring, and advanced model training features.
    """
    
    def __init__(self, config: Optional[CPUTrainingConfig] = None):
        """
        Initialize the training engine.
        
        Args:
            config: Configuration for the training engine
        """
        self.config = config or CPUTrainingConfig()
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize state
        self.state = TrainingState.INITIALIZING
        self.model = None
        self.preprocessor = None
        self.feature_engineerer = None
        
        # Data containers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.sample_weights = None
        
        # Feature info
        self.feature_names = []
        self.target_names = []
        
        # Training metrics and monitoring
        self.metrics = TrainingMetrics()
        
        # Create checkpoint manager if checkpointing is enabled
        if self.config.model_training.enable_checkpointing:
            self.checkpoint_manager = ModelCheckpoint(
                checkpoint_dir=CHECKPOINT_PATH,
                max_checkpoints=self.config.model_training.keep_checkpoint_max,
                save_best_only=True,
                metric=self.config.model_training.early_stopping_metric 
                       if self.config.model_training.early_stopping_metric != 'auto' else 'val_loss',
                mode='min' if any(x in self.config.model_training.early_stopping_metric.lower()
                                for x in ['loss', 'error', 'mse', 'mae', 'rmse']) else 'max',
                verbose=self.config.debug_mode
            )
        else:
            self.checkpoint_manager = None
        
        # Thread safety
        self.state_lock = threading.RLock()
        
        # Set up threading
        self._setup_threading()
        
        # Set up monitoring and resource management
        self._setup_monitoring()
        
        # Configure Intel optimizations if enabled
        if self.config.enable_intel_optimization:
            self._setup_intel_optimizations()
            
        # Signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Change state to ready
        self.state = TrainingState.READY
        self.logger.info(f"Training engine initialized with version {self.config.model_registry.model_version}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the training engine."""
        logger = logging.getLogger(f"{__name__}.TrainingEngine")
        
        # Configure logging based on debug mode
        log_level_str = self.config.log_level.upper()
        try:
            level = getattr(logging, log_level_str)
        except AttributeError:
            level = logging.INFO
            print(f"Invalid log level: {log_level_str}, using INFO")
        
        logger.setLevel(level)
        
        # Ensure log directory exists
        os.makedirs(DEFAULT_LOG_PATH, exist_ok=True)
        
        # Only add handlers if there are none
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            
            # File handler
            model_name = self.config.model_registry.model_name
            model_version = self.config.model_registry.model_version
            file_handler = logging.FileHandler(
                os.path.join(DEFAULT_LOG_PATH, f"training_{model_name}_{model_version}.log")
            )
            file_handler.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_threading(self) -> None:
        """Configure threading and parallelism."""
        # Set number of threads for numpy/MKL if available
        thread_count = self.config.num_threads
        
        # Set environment variables for better thread control
        os.environ["OMP_NUM_THREADS"] = str(thread_count)
        os.environ["MKL_NUM_THREADS"] = str(thread_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
        
        # Set thread count for numpy if using OpenBLAS/MKL
        try:
            import numpy as np
            np.set_num_threads(thread_count)
        except (ImportError, AttributeError):
            pass
        
        # Configure MKL settings if available
        if MKL_AVAILABLE:
            try:
                mkl.set_num_threads(thread_count)
                self.logger.info(f"MKL configured with {thread_count} threads")
            except Exception as e:
                self.logger.warning(f"Failed to configure MKL: {str(e)}")
        
        # Create thread pool for parallel tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=thread_count,
            thread_name_prefix="TrainingWorker"
        )
        
        # Create process pool for distributed processing if enabled
        if self.config.enable_distributed_training:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.config.num_workers,
                thread_name_prefix="DistributedWorker"
            )
        
        self.logger.info(f"Threading configured with {thread_count} threads")
    
    def _setup_intel_optimizations(self) -> None:
        """Configure Intel-specific optimizations."""
        if not self.config.enable_intel_optimization:
            return
        
        # Log status of Intel optimizations
        optimizations = []
        
        # Configure Intel MKL if available
        if MKL_AVAILABLE:
            try:
                # Configure MKL for best performance
                mkl.set_threading_layer("intel")
                mkl.set_dynamic(False)
                optimizations.append("MKL")
            except Exception as e:
                self.logger.warning(f"Failed to configure MKL optimizations: {str(e)}")
        
        # Check for scikit-learn optimizations
        if SKLEARN_AVAILABLE and SKLEARN_OPTIMIZED:
            optimizations.append("scikit-learn-intelex")
        
        # Check if we're running Intel Python Distribution
        if INTEL_PYTHON:
            optimizations.append("Intel Python Distribution")
        
        if optimizations:
            self.logger.info(f"Intel optimizations enabled: {', '.join(optimizations)}")
        else:
            self.logger.warning("No Intel optimizations are available")
    
    def _setup_monitoring(self) -> None:
        """Set up monitoring and resource management."""
        if not self.config.enable_monitoring:
            return
        
        # Create monitoring thread
        self.monitor_stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="TrainingMonitor"
        )
        
        # Start monitoring thread
        self.monitor_thread.start()
        self.logger.info("Monitoring thread started")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.logger.info("Signal handlers registered for graceful shutdown")
        except (AttributeError, ValueError) as e:
            # Signal handling might not work in certain environments (e.g., Windows)
            self.logger.warning(f"Could not set up signal handlers: {e}")
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle termination signals."""
        self.logger.info(f"Received signal {sig}, initiating graceful shutdown")
        self.stop_training()
    
    def _monitoring_loop(self) -> None:
        """Background thread for monitoring system resources."""
        check_interval = self.config.monitoring_interval
        
        while not self.monitor_stop_event.is_set():
            try:
                # Check memory usage
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                cpu_percent = process.cpu_percent(interval=0.1)
                
                # Update metrics
                self.metrics.record_resource_usage(memory_mb, cpu_percent)
                
                # Check if we've exceeded memory limit
                if self.config.memory_limit_gb and memory_mb > (self.config.memory_limit_gb * 1024):
                    self.logger.warning(
                        f"Memory usage ({memory_mb:.1f} MB) exceeds limit "
                        f"({self.config.memory_limit_gb * 1024:.1f} MB), triggering GC"
                    )
                    gc.collect()
                
                # Sleep until next check
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
                
                # Sleep with reduced interval on error
                time.sleep(max(1.0, check_interval / 2))
    
    def load_data(self) -> bool:
        """
        Load and split data according to configuration.
        
        Returns:
            Success flag
        """
        self.logger.info("Loading data...")
        
        with self.state_lock:
            self.state = TrainingState.PREPROCESSING
        
        try:
            dataset_config = self.config.dataset
            
            # Check if paths are provided
            if dataset_config.train_path:
                # Load from separate train/val/test files if provided
                self._load_data_from_files()
            else:
                # Otherwise assume data will be set programmatically
                self.logger.warning("No data path provided. Use set_data() to provide training data.")
                return False
            
            self.logger.info(f"Data loaded successfully: {self.X_train.shape[0]} training samples")
            
            if self.X_val is not None:
                self.logger.info(f"Validation data: {self.X_val.shape[0]} samples")
            
            if self.X_test is not None:
                self.logger.info(f"Test data: {self.X_test.shape[0]} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            
            self.state = TrainingState.ERROR
            return False
    
    def _load_data_from_files(self) -> None:
        """Load data from files specified in configuration."""
        dataset_config = self.config.dataset
        
        # Determine file format and load accordingly
        train_path = dataset_config.train_path
        file_ext = os.path.splitext(train_path)[1].lower()
        
        # Load train data
        if file_ext == '.csv':
            train_data = pd.read_csv(
                train_path, 
                sep=dataset_config.csv_separator,
                encoding=dataset_config.encoding,
                header=0 if dataset_config.has_header else None
            )
        elif file_ext == '.parquet':
            train_data = pd.read_parquet(train_path)
        elif file_ext == '.jsonl' or file_ext == '.json':
            train_data = pd.read_json(train_path, lines=(file_ext == '.jsonl'))
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Extract features and target
        if not dataset_config.feature_columns and not dataset_config.target_column:
            # Auto-detect: assume last column is target if not specified
            feature_cols = train_data.columns[:-1]
            target_col = train_data.columns[-1]
        else:
            # Use specified columns
            feature_cols = dataset_config.feature_columns
            if not feature_cols:  # If still empty, use all non-target columns
                feature_cols = [c for c in train_data.columns if c != dataset_config.target_column]
            
            target_col = dataset_config.target_column
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Extract feature dataframe and target series
        X_all = train_data[feature_cols]
        y_all = train_data[target_col]
        
        # Store target name or names
        if isinstance(y_all, pd.DataFrame):
            self.target_names = y_all.columns.tolist()
        else:
            self.target_names = [target_col]
        
        # Handle weight column if specified
        if dataset_config.weight_column and dataset_config.weight_column in train_data:
            self.sample_weights = train_data[dataset_config.weight_column].values
        
        # Check if separate validation and test files are provided
        if dataset_config.validation_path and dataset_config.test_path:
            # Load validation data
            if os.path.exists(dataset_config.validation_path):
                val_data = self._load_dataset_file(dataset_config.validation_path)
                self.X_val = val_data[feature_cols]
                self.y_val = val_data[target_col]
            
            # Load test data
            if os.path.exists(dataset_config.test_path):
                test_data = self._load_dataset_file(dataset_config.test_path)
                self.X_test = test_data[feature_cols]
                self.y_test = test_data[target_col]
                
            # Use loaded train data as-is
            self.X_train = X_all
            self.y_train = y_all
        else:
            # Split the data according to configuration
            self._split_data(X_all, y_all)
    
    def _load_dataset_file(self, file_path: str) -> pd.DataFrame:
        """Load a dataset file in the appropriate format."""
        dataset_config = self.config.dataset
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(
                file_path, 
                sep=dataset_config.csv_separator,
                encoding=dataset_config.encoding,
                header=0 if dataset_config.has_header else None
            )
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path)
        elif file_ext == '.jsonl' or file_ext == '.json':
            return pd.read_json(file_path, lines=(file_ext == '.jsonl'))
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Split data into train, validation, and test sets."""
        dataset_config = self.config.dataset
        split_strategy = dataset_config.split_strategy
        
        # First split off test data
        if dataset_config.test_size > 0:
            if split_strategy == DataSplitStrategy.STRATIFIED and dataset_config.stratify_column:
                # For classification, use stratified split
                stratify = y if dataset_config.stratify_column == dataset_config.target_column else X[dataset_config.stratify_column]
                X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                    X, y, 
                    test_size=dataset_config.test_size,
                    random_state=dataset_config.random_seed,
                    stratify=stratify
                )
            elif split_strategy == DataSplitStrategy.GROUP and dataset_config.group_column:
                # Group-based split (e.g., for data with groups/clusters)
                groups = X[dataset_config.group_column]
                
                # We need to use GroupKFold for splitting with groups
                gkf = GroupKFold(n_splits=int(1 / dataset_config.test_size))
                train_idx, test_idx = next(gkf.split(X, y, groups))
                
                X_temp, self.X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_temp, self.y_test = y.iloc[train_idx], y.iloc[test_idx]
                
            elif split_strategy == DataSplitStrategy.TIME_SERIES and dataset_config.time_column:
                # Time-based split (latest data as test)
                time_col = dataset_config.time_column
                # Sort by time
                temp_df = pd.concat([X, y], axis=1)
                temp_df = temp_df.sort_values(by=time_col)
                
                # Split off latest data as test
                split_idx = int(len(temp_df) * (1 - dataset_config.test_size))
                train_temp = temp_df.iloc[:split_idx]
                test_temp = temp_df.iloc[split_idx:]
                
                # Extract features and target
                X_temp = train_temp[X.columns]
                y_temp = train_temp[y.name]
                self.X_test = test_temp[X.columns]
                self.y_test = test_temp[y.name]
                
            else:
                # Regular random split
                X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                    X, y, 
                    test_size=dataset_config.test_size,
                    random_state=dataset_config.random_seed
                )
        else:
            X_temp, y_temp = X, y
        
        # Then split validation data from remaining training data
        if dataset_config.validation_size > 0:
            validation_ratio = dataset_config.validation_size / (1 - dataset_config.test_size)
            
            if split_strategy == DataSplitStrategy.STRATIFIED and dataset_config.stratify_column:
                # For classification, use stratified split for validation too
                stratify = y_temp if dataset_config.stratify_column == dataset_config.target_column else X_temp[dataset_config.stratify_column]
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    X_temp, y_temp, 
                    test_size=validation_ratio,
                    random_state=dataset_config.random_seed,
                    stratify=stratify
                )
            elif split_strategy == DataSplitStrategy.GROUP and dataset_config.group_column:
                # Group-based split for validation
                groups = X_temp[dataset_config.group_column]
                
                # Use GroupKFold for validation split too
                gkf = GroupKFold(n_splits=int(1 / validation_ratio))
                train_idx, val_idx = next(gkf.split(X_temp, y_temp, groups))
                
                self.X_train, self.X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
                self.y_train, self.y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
                
            elif split_strategy == DataSplitStrategy.TIME_SERIES and dataset_config.time_column:
                # Time-based split for validation (latest remaining data)
                time_col = dataset_config.time_column
                # Sort by time
                temp_df = pd.concat([X_temp, y_temp], axis=1)
                temp_df = temp_df.sort_values(by=time_col)
                
                # Split off latest data as validation
                split_idx = int(len(temp_df) * (1 - validation_ratio))
                train_temp = temp_df.iloc[:split_idx]
                val_temp = temp_df.iloc[split_idx:]
                
                # Extract features and target
                self.X_train = train_temp[X.columns]
                self.y_train = train_temp[y.name]
                self.X_val = val_temp[X.columns]
                self.y_val = val_temp[y.name]
                
            else:
                # Regular random split for validation
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    X_temp, y_temp, 
                    test_size=validation_ratio,
                    random_state=dataset_config.random_seed
                )
        else:
            # No validation set
            self.X_train, self.y_train = X_temp, y_temp
            self.X_val, self.y_val = None, None
    
    def set_data(self, X_train: Union[np.ndarray, pd.DataFrame], 
                y_train: Union[np.ndarray, pd.Series],
                X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                y_val: Optional[Union[np.ndarray, pd.Series]] = None,
                X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                y_test: Optional[Union[np.ndarray, pd.Series]] = None,
                sample_weights: Optional[np.ndarray] = None,
                feature_names: Optional[List[str]] = None) -> None:
        """
        Set data for training directly.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target
            X_test: Optional test features
            y_test: Optional test target
            sample_weights: Optional sample weights
            feature_names: Optional feature names
        """
        self.logger.info("Setting training data...")
        
        try:
            # Store data
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test
            self.sample_weights = sample_weights
            
            # Set feature names if provided
            if feature_names is not None:
                self.feature_names = feature_names
            elif isinstance(X_train, pd.DataFrame):
                self.feature_names = X_train.columns.tolist()
            
            # Set target names
            if isinstance(y_train, pd.DataFrame):
                self.target_names = y_train.columns.tolist()
            elif isinstance(y_train, pd.Series):
                self.target_names = [y_train.name] if y_train.name else ['target']
            else:
                self.target_names = ['target']
            
            self.logger.info(f"Data set successfully: {X_train.shape[0]} training samples")
            
            if X_val is not None:
                self.logger.info(f"Validation data: {X_val.shape[0]} samples")
            
            if X_test is not None:
                self.logger.info(f"Test data: {X_test.shape[0]} samples")
            
        except Exception as e:
            self.logger.error(f"Error setting data: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            
            self.state = TrainingState.ERROR
    
    def preprocess_data(self) -> bool:
        """
        Preprocess data according to configuration.
        
        Returns:
            Success flag
        """
        self.logger.info("Preprocessing data...")
        
        with self.state_lock:
            self.state = TrainingState.PREPROCESSING
        
        try:
            if self.X_train is None:
                self.logger.error("No training data available for preprocessing")
                return False
            
            dataset_config = self.config.dataset
            
            # Create preprocessing stats dictionary
            preprocessing_stats = {
                "original_shape": self.X_train.shape,
                "missing_values_before": {},
                "categorical_counts": {},
                "numerical_stats": {}
            }
            
            # Convert pandas to numpy if needed for consistent processing
            X_train_values = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
            
            # Track original types and shapes
            if isinstance(self.X_train, pd.DataFrame):
                preprocessing_stats["original_dtypes"] = self.X_train.dtypes.to_dict()
                
                # Count missing values before preprocessing
                preprocessing_stats["missing_values_before"] = self.X_train.isna().sum().to_dict()
                
                # Get stats for categorical features
                for col in dataset_config.categorical_columns:
                    if col in self.X_train.columns:
                        preprocessing_stats["categorical_counts"][col] = self.X_train[col].value_counts().to_dict()
                
                # Get stats for numerical features
                for col in dataset_config.numerical_columns:
                    if col in self.X_train.columns:
                        preprocessing_stats["numerical_stats"][col] = {
                            "mean": self.X_train[col].mean(),
                            "std": self.X_train[col].std(),
                            "min": self.X_train[col].min(),
                            "max": self.X_train[col].max(),
                            "median": self.X_train[col].median()
                        }
            
            # Create preprocessor configuration
            preprocessor_config = PreprocessorConfig(
                handle_missing=dataset_config.handle_missing,
                handle_outliers=dataset_config.handle_outliers,
                handle_categorical=dataset_config.handle_categorical,
                scaling_method=self.config.feature_engineering.scaling_method 
                    if self.config.feature_engineering.enable_scaling else None,
                categorical_encoding=self.config.feature_engineering.encoding_method 
                    if self.config.feature_engineering.enable_feature_encoding else None,
                max_onehot_cardinality=self.config.feature_engineering.max_onehot_cardinality,
                categorical_columns=dataset_config.categorical_columns,
                numerical_columns=dataset_config.numerical_columns,
                datetime_columns=dataset_config.datetime_columns,
                text_columns=dataset_config.text_columns
            )
            
            # Create preprocessor and fit on training data
            self.preprocessor = DataPreprocessor(preprocessor_config)
            X_train_processed = self.preprocessor.fit_transform(self.X_train)
            
            # Process validation data if available
            X_val_processed = None
            if self.X_val is not None:
                X_val_processed = self.preprocessor.transform(self.X_val)
                
            # Process test data if available
            X_test_processed = None
            if self.X_test is not None:
                X_test_processed = self.preprocessor.transform(self.X_test)
                
            # Update data with processed versions
            self.X_train = X_train_processed
            self.X_val = X_val_processed
            self.X_test = X_test_processed
            
            # Update feature names if they've changed
            self.feature_names = self.preprocessor.get_feature_names()
            
            # Add preprocessing results to stats
            preprocessing_stats["processed_shape"] = self.X_train.shape
            preprocessing_stats["feature_names"] = self.feature_names
            
            # Feature engineering if configured
            if self.config.feature_engineering.enable_feature_generation:
                self._generate_features()
            
            if self.config.feature_engineering.enable_feature_selection:
                self._select_features()
                
            # Update metrics with preprocessing stats
            self.metrics.set_preprocessing_stats(preprocessing_stats)
            
            self.logger.info(f"Data preprocessing completed. Shape after preprocessing: {self.X_train.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            
            self.state = TrainingState.ERROR
            return False

    def _generate_features(self) -> None:
        """Generate additional features based on configuration."""
        self.logger.info("Generating additional features...")
        
        fe_config = self.config.feature_engineering
        dataset_config = self.config.dataset
        
        try:
            # Skip if no data or feature generation is disabled
            if self.X_train is None or not fe_config.enable_feature_generation:
                return
                
            # Convert to DataFrame if numpy array
            X_train_df = self.X_train
            if not isinstance(X_train_df, pd.DataFrame):
                X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
                
            X_val_df = None
            if self.X_val is not None:
                if not isinstance(self.X_val, pd.DataFrame):
                    X_val_df = pd.DataFrame(self.X_val, columns=self.feature_names)
                else:
                    X_val_df = self.X_val
                    
            X_test_df = None
            if self.X_test is not None:
                if not isinstance(self.X_test, pd.DataFrame):
                    X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
                else:
                    X_test_df = self.X_test
            
            # Datetime features
            if fe_config.datetime_features and dataset_config.datetime_columns:
                for col in dataset_config.datetime_columns:
                    if col in X_train_df.columns:
                        # Ensure datetime type
                        X_train_df[col] = pd.to_datetime(X_train_df[col])
                        
                        # Extract date components
                        X_train_df[f"{col}_year"] = X_train_df[col].dt.year
                        X_train_df[f"{col}_month"] = X_train_df[col].dt.month
                        X_train_df[f"{col}_day"] = X_train_df[col].dt.day
                        X_train_df[f"{col}_dayofweek"] = X_train_df[col].dt.dayofweek
                        X_train_df[f"{col}_quarter"] = X_train_df[col].dt.quarter
                        X_train_df[f"{col}_is_weekend"] = X_train_df[col].dt.dayofweek >= 5
                        
                        # Apply to validation and test as well
                        if X_val_df is not None and col in X_val_df.columns:
                            X_val_df[col] = pd.to_datetime(X_val_df[col])
                            X_val_df[f"{col}_year"] = X_val_df[col].dt.year
                            X_val_df[f"{col}_month"] = X_val_df[col].dt.month
                            X_val_df[f"{col}_day"] = X_val_df[col].dt.day
                            X_val_df[f"{col}_dayofweek"] = X_val_df[col].dt.dayofweek
                            X_val_df[f"{col}_quarter"] = X_val_df[col].dt.quarter
                            X_val_df[f"{col}_is_weekend"] = X_val_df[col].dt.dayofweek >= 5
                        
                        if X_test_df is not None and col in X_test_df.columns:
                            X_test_df[col] = pd.to_datetime(X_test_df[col])
                            X_test_df[f"{col}_year"] = X_test_df[col].dt.year
                            X_test_df[f"{col}_month"] = X_test_df[col].dt.month
                            X_test_df[f"{col}_day"] = X_test_df[col].dt.day
                            X_test_df[f"{col}_dayofweek"] = X_test_df[col].dt.dayofweek
                            X_test_df[f"{col}_quarter"] = X_test_df[col].dt.quarter
                            X_test_df[f"{col}_is_weekend"] = X_test_df[col].dt.dayofweek >= 5
            
            # Polynomial features if configured
            if fe_config.enable_polynomial_features:
                if SKLEARN_AVAILABLE:
                    from sklearn.preprocessing import PolynomialFeatures
                    
                    # Identify numeric columns
                    numeric_cols = dataset_config.numerical_columns
                    if not numeric_cols and isinstance(X_train_df, pd.DataFrame):
                        numeric_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if numeric_cols:
                        # Select only numeric features for polynomial generation
                        X_numeric = X_train_df[numeric_cols].values
                        
                        # Create polynomial features
                        poly = PolynomialFeatures(
                            degree=fe_config.polynomial_degree,
                            interaction_only=fe_config.interaction_only,
                            include_bias=False
                        )
                        
                        # Transform data
                        X_poly = poly.fit_transform(X_numeric)
                        
                        # Create feature names
                        poly_feature_names = poly.get_feature_names_out(numeric_cols)
                        
                        # Add new features to original dataframe
                        poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
                        # Only keep the interaction terms, not the original features
                        poly_df = poly_df.iloc[:, len(numeric_cols):]
                        X_train_df = pd.concat([X_train_df, poly_df], axis=1)
                        
                        # Apply to validation and test sets
                        if X_val_df is not None:
                            X_val_numeric = X_val_df[numeric_cols].values
                            X_val_poly = poly.transform(X_val_numeric)
                            val_poly_df = pd.DataFrame(X_val_poly, columns=poly_feature_names)
                            val_poly_df = val_poly_df.iloc[:, len(numeric_cols):]
                            X_val_df = pd.concat([X_val_df, val_poly_df], axis=1)
                        
                        if X_test_df is not None:
                            X_test_numeric = X_test_df[numeric_cols].values
                            X_test_poly = poly.transform(X_test_numeric)
                            test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names)
                            test_poly_df = test_poly_df.iloc[:, len(numeric_cols):]
                            X_test_df = pd.concat([X_test_df, test_poly_df], axis=1)
            
            # Update feature names and dataframes
            self.feature_names = X_train_df.columns.tolist()
            self.X_train = X_train_df
            self.X_val = X_val_df
            self.X_test = X_test_df
            
            self.logger.info(f"Feature generation complete. New feature count: {len(self.feature_names)}")
            
        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())

    def _select_features(self) -> None:
        """Select features based on importance or other criteria."""
        self.logger.info("Performing feature selection...")
        
        fe_config = self.config.feature_engineering
        
        try:
            # Skip if disabled or no data
            if not fe_config.enable_feature_selection or self.X_train is None:
                return
                
            # Ensure we have sklearn
            if not SKLEARN_AVAILABLE:
                self.logger.warning("Feature selection requires scikit-learn, which is not available")
                return
                
            # Convert to numpy arrays if needed
            X_train = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
            y_train = self.y_train.values if isinstance(self.y_train, pd.DataFrame) or isinstance(self.y_train, pd.Series) else self.y_train
            
            feature_names = self.feature_names
            
            # Different selection methods
            method = fe_config.feature_selection_method.lower()
            
            # Track selected features and their scores
            selected_features = []
            feature_scores = {}
            
            if method == 'importance':
                # Use a tree-based model to calculate feature importances
                if self.config.model_training.problem_type == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=50, n_jobs=self.config.num_threads)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=50, n_jobs=self.config.num_threads)
                    
                # Fit the model
                model.fit(X_train, y_train)
                
                # Get feature importances
                importances = model.feature_importances_
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                # Create feature scores dictionary
                for i, idx in enumerate(indices):
                    feature_scores[feature_names[idx]] = importances[idx]
                
                # Apply threshold or take top N
                if fe_config.max_features:
                    selected_indices = indices[:fe_config.max_features]
                else:
                    selected_indices = indices[importances[indices] >= fe_config.min_feature_importance]
                    
                # Get selected feature names
                selected_features = [feature_names[i] for i in selected_indices]
                
            elif method == 'correlation':
                # Use correlation for feature selection
                from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                
                # Choose proper statistical test
                if self.config.model_training.problem_type == 'classification':
                    score_func = f_classif
                else:
                    score_func = f_regression
                    
                # Determine number of features to select
                k = fe_config.max_features if fe_config.max_features else int(X_train.shape[1] * 0.5)
                
                # Create and fit selector
                selector = SelectKBest(score_func=score_func, k=k)
                selector.fit(X_train, y_train)
                
                # Get selected feature mask and scores
                selected_mask = selector.get_support()
                scores = selector.scores_
                
                # Get selected feature names and scores
                for i, (name, selected) in enumerate(zip(feature_names, selected_mask)):
                    if selected:
                        selected_features.append(name)
                    feature_scores[name] = scores[i]
                    
            elif method == 'recursive':
                # Use Recursive Feature Elimination
                from sklearn.feature_selection import RFECV
                
                # Create a base model
                if self.config.model_training.problem_type == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=50, n_jobs=self.config.num_threads)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=50, n_jobs=self.config.num_threads)
                
                # Create RFE selector with cross-validation
                min_features = 1
                if fe_config.max_features:
                    max_features = min(fe_config.max_features, X_train.shape[1])
                else:
                    max_features = X_train.shape[1]
                    
                selector = RFECV(
                    estimator=model,
                    step=1,
                    cv=5,
                    scoring='accuracy' if self.config.model_training.problem_type == 'classification' else 'neg_mean_squared_error',
                    min_features_to_select=min_features,
                    n_jobs=self.config.num_threads
                )
                
                # Fit selector
                selector.fit(X_train, y_train)
                
                # Get selected features
                selected_mask = selector.support_
                ranking = selector.ranking_
                
                # Create feature scores based on ranking (lower is better)
                max_rank = np.max(ranking)
                
                # Get selected feature names and scores
                for i, (name, selected, rank) in enumerate(zip(feature_names, selected_mask, ranking)):
                    if selected:
                        selected_features.append(name)
                    # Convert ranking to a score (higher is better)
                    feature_scores[name] = (max_rank - rank + 1) / max_rank
            
            # Apply feature selection if features were selected
            if selected_features:
                self.logger.info(f"Selected {len(selected_features)} features out of {len(feature_names)}")
                
                # Filter features in dataframes
                if isinstance(self.X_train, pd.DataFrame):
                    self.X_train = self.X_train[selected_features]
                    if self.X_val is not None and isinstance(self.X_val, pd.DataFrame):
                        self.X_val = self.X_val[selected_features]
                    if self.X_test is not None and isinstance(self.X_test, pd.DataFrame):
                        self.X_test = self.X_test[selected_features]
                else:
                    # For numpy arrays, create a boolean mask
                    mask = np.array([name in selected_features for name in feature_names])
                    self.X_train = self.X_train[:, mask]
                    if self.X_val is not None:
                        self.X_val = self.X_val[:, mask]
                    if self.X_test is not None:
                        self.X_test = self.X_test[:, mask]
                
                # Update feature names
                self.feature_names = selected_features
                
                # Store feature importance scores
                self.metrics.set_feature_importance({name: feature_scores.get(name, 0.0) for name in selected_features})
                
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())

    def train_model(self) -> bool:
        """
        Train the model according to configuration.
        
        Returns:
            Success flag
        """
        self.logger.info("Starting model training...")
        
        with self.state_lock:
            if self.state not in [TrainingState.READY, TrainingState.PREPROCESSING]:
                self.logger.error(f"Cannot start training from state {self.state}")
                return False
            self.state = TrainingState.TRAINING
        
        try:
            # Ensure we have training data
            if self.X_train is None or self.y_train is None:
                self.logger.error("No training data available")
                self.state = TrainingState.ERROR
                return False
            
            # Reset metrics for tracking training progress
            self.metrics.reset()
            
            # Handle hyperparameter optimization if enabled
            if self.config.hyperparameter_optimization.enabled:
                self.state = TrainingState.OPTIMIZING
                self.logger.info("Starting hyperparameter optimization...")
                success = self.optimize_hyperparameters()
                if not success:
                    self.logger.error("Hyperparameter optimization failed")
                    self.state = TrainingState.ERROR
                    return False
            
            # Get total epochs, using 1 for non-epoch based models
            total_epochs = self.config.model_training.num_epochs or 1
            
            # Start timing and metrics tracking
            self.metrics.start_training(total_epochs)
            
            # Initialize model
            success = self._initialize_model()
            if not success:
                self.logger.error("Model initialization failed")
                self.state = TrainingState.ERROR
                return False
            
            # Training loop
            with self.state_lock:
                self.state = TrainingState.TRAINING
            
            self.logger.info(f"Training model with {total_epochs} epochs...")
            
            # Actual training implementation differs for each model type
            if self.config.model_training.model_type == ModelType.SKLEARN:
                success = self._train_sklearn_model()
            elif self.config.model_training.model_type == ModelType.XGBOOST:
                success = self._train_xgboost_model()
            elif self.config.model_training.model_type == ModelType.LIGHTGBM:
                success = self._train_lightgbm_model()
            else:
                success = self._train_custom_model()
            
            if not success:
                self.logger.error("Model training failed")
                self.state = TrainingState.ERROR
                return False
            
            # Calculate feature importance if enabled
            if self.config.model_training.compute_feature_importance:
                self._compute_feature_importance()
            
            # Evaluate the model
            with self.state_lock:
                self.state = TrainingState.EVALUATING
            
            success = self.evaluate_model()
            if not success:
                self.logger.warning("Model evaluation had issues")
                # Don't fail the whole training because of evaluation issues
            
            # Record the end of training
            self.metrics.end_training()
            
            # Save model if configured
            if self.config.model_registry.save_formats:
                with self.state_lock:
                    self.state = TrainingState.SAVING
                
                success = self.save_model()
                if not success:
                    self.logger.error("Failed to save model")
                    self.state = TrainingState.ERROR
                    return False
            
            with self.state_lock:
                self.state = TrainingState.COMPLETED
            
            training_time = self.metrics.get_progress().get("elapsed_time", 0)
            self.logger.info(f"Model training completed in {training_time:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            
            with self.state_lock:
                self.state = TrainingState.ERROR
                
            return False

    def _initialize_model(self) -> bool:
        """Initialize the model based on configuration."""
        try:
            model_config = self.config.model_training
            model_type = model_config.model_type
            
            # Get appropriate parameters from configuration
            params = model_config.hyperparameters.copy()
            
            if model_type == ModelType.SKLEARN:
                # Import the appropriate model class
                if model_config.model_class:
                    module_path, class_name = model_config.model_class.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    model_class = getattr(module, class_name)
                    
                    # Set n_jobs parameter if applicable
                    if hasattr(model_class, 'n_jobs'):
                        params['n_jobs'] = model_config.n_jobs
                    
                    # Create the model instance
                    self.model = model_class(**params)
                    
                else:
                    # Default models based on problem type if no class specified
                    if model_config.problem_type == 'classification':
                        from sklearn.ensemble import RandomForestClassifier
                        self.model = RandomForestClassifier(
                            n_estimators=100,
                            n_jobs=model_config.n_jobs,
                            **params
                        )
                    else:
                        from sklearn.ensemble import RandomForestRegressor
                        self.model = RandomForestRegressor(
                            n_estimators=100,
                            n_jobs=model_config.n_jobs,
                            **params
                        )
                        
            elif model_type == ModelType.XGBOOST:
                if not XGBOOST_AVAILABLE:
                    self.logger.error("XGBoost not available but required for model type")
                    return False
                    
                # Set objective based on problem type if not specified
                if 'objective' not in params and model_config.objective is None:
                    if model_config.problem_type == 'classification':
                        # Check if binary or multi-class
                        if np.unique(self.y_train).size <= 2:
                            params['objective'] = 'binary:logistic'
                        else:
                            params['objective'] = 'multi:softprob'
                            params['num_class'] = np.unique(self.y_train).size
                    else:
                        params['objective'] = 'reg:squarederror'
                
                # Add model parameters
                if model_config.n_jobs > 0:
                    params['nthread'] = model_config.n_jobs
                    
                if model_config.use_gpu:
                    params['tree_method'] = 'gpu_hist'
                else:
                    params['tree_method'] = 'hist'  # Fast histogram-based algorithm
                    
                # Create the model
                self.model = xgb.XGBModel(**params)
                
            elif model_type == ModelType.LIGHTGBM:
                if not LIGHTGBM_AVAILABLE:
                    self.logger.error("LightGBM not available but required for model type")
                    return False
                    
                # Set objective based on problem type if not specified
                if 'objective' not in params and model_config.objective is None:
                    if model_config.problem_type == 'classification':
                        # Check if binary or multi-class
                        if np.unique(self.y_train).size <= 2:
                            params['objective'] = 'binary'
                        else:
                            params['objective'] = 'multiclass'
                            params['num_class'] = np.unique(self.y_train).size
                    else:
                        params['objective'] = 'regression'
                
                # Add model parameters
                if model_config.n_jobs > 0:
                    params['num_threads'] = model_config.n_jobs
                    
                if model_config.use_gpu:
                    params['device'] = 'gpu'
                    
                # Create the model
                if model_config.problem_type == 'classification':
                    self.model = lgb.LGBMClassifier(**params)
                else:
                    self.model = lgb.LGBMRegressor(**params)
                    
            else:
                # Custom model implementation
                if not model_config.model_class:
                    self.logger.error("Model class must be specified for custom models")
                    return False
                    
                # Import and instantiate the custom model
                module_path, class_name = model_config.model_class.rsplit('.', 1)
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
                
                # Create the model instance
                self.model = model_class(**params)
            
            self.logger.info(f"Model initialized: {type(self.model).__name__}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False

    def _train_sklearn_model(self) -> bool:
        """Train a scikit-learn model."""
        model_config = self.config.model_training
        
        try:
            # Convert to numpy arrays if needed
            X_train = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
            y_train = self.y_train.values if isinstance(self.y_train, pd.DataFrame) or isinstance(self.y_train, pd.Series) else self.y_train
            
            X_val = None
            y_val = None
            if self.X_val is not None and self.y_val is not None:
                X_val = self.X_val.values if isinstance(self.X_val, pd.DataFrame) else self.X_val
                y_val = self.y_val.values if isinstance(self.y_val, pd.DataFrame) or isinstance(self.y_val, pd.Series) else self.y_val
            
            # For scikit-learn models, there's typically just one "epoch"
            # but we can use cross-validation or multiple fits for ensemble building
            
            # Handle cross-validation if enabled
            if model_config.enable_cross_validation:
                self.logger.info("Training with cross-validation...")
                
                # Setup cross-validation strategy
                if model_config.cv_strategy == 'stratified' and model_config.problem_type == 'classification':
                    from sklearn.model_selection import StratifiedKFold
                    cv = StratifiedKFold(n_splits=model_config.cv_folds, shuffle=True, random_state=self.config.random_seed)
                else:
                    from sklearn.model_selection import KFold
                    cv = KFold(n_splits=model_config.cv_folds, shuffle=True, random_state=self.config.random_seed)
                
                # Scores to track during CV
                cv_scores = defaultdict(list)
                
                # Perform cross-validation manually to keep track of per-fold metrics
                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                    self.logger.info(f"Training fold {fold+1}/{model_config.cv_folds}")
                    
                    # Split data for this fold
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    # Clone the model for each fold
                    from sklearn.base import clone
                    fold_model = clone(self.model)
                    
                    # Train the model
                    fold_model.fit(X_fold_train, y_fold_train)
                    
                    # Evaluate on validation fold
                    fold_metrics = self._evaluate_sklearn_model(fold_model, X_fold_val, y_fold_val)
                    
                    # Store metrics for this fold
                    for metric_name, metric_value in fold_metrics.items():
                        cv_scores[metric_name].append(metric_value)
                    
                    # Log progress for this fold
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in fold_metrics.items()])
                    self.logger.info(f"Fold {fold+1} - {metrics_str}")
                    
                    # Update overall metrics with fold progress
                    epoch_metrics = {
                        "fold": fold + 1,
                        "train_loss": 0.0,  # We don't typically get loss from sklearn models
                    }
                    for k, v in fold_metrics.items():
                        epoch_metrics[f"val_{k}"] = v
                    
                    self.metrics.update_epoch(fold + 1, {"loss": 0.0}, epoch_metrics)
                
                # Compute mean and std of CV scores
                cv_results = {}
                for metric_name, values in cv_scores.items():
                    cv_results[f"{metric_name}_mean"] = np.mean(values)
                    cv_results[f"{metric_name}_std"] = np.std(values)
                    
                # Log final CV results
                self.logger.info("Cross-validation results:")
                for name, value in cv_results.items():
                    self.logger.info(f"{name}: {value:.4f}")
                
                # Store CV results in metrics
                self.metrics.set_cv_results(cv_results)
                
                # After CV, train final model on all data
                self.logger.info("Training final model on all data")
                
            # Actual model training
            sample_weights = self.sample_weights
            
            # Check if we should use class weights for imbalanced classification
            if model_config.problem_type == 'classification' and model_config.class_weight and hasattr(self.model, 'class_weight'):
                if model_config.class_weight == 'balanced' or model_config.class_weight == 'balanced_subsample':
                    self.model.class_weight = model_config.class_weight
                elif isinstance(model_config.class_weight, dict):
                    self.model.class_weight = model_config.class_weight
                    
            # Fit the model
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Compute training metrics
            train_metrics = self._evaluate_sklearn_model(self.model, X_train, y_train)
            self.logger.info(f"Training metrics: {train_metrics}")
            
            # Compute validation metrics if validation data exists
            val_metrics = None
            if X_val is not None and y_val is not None:
                val_metrics = self._evaluate_sklearn_model(self.model, X_val, y_val)
                self.logger.info(f"Validation metrics: {val_metrics}")
            
            # Update final epoch metrics
            self.metrics.update_epoch(1, train_metrics, val_metrics)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training scikit-learn model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False

    def _train_xgboost_model(self) -> bool:
        """Train an XGBoost model."""
        if not XGBOOST_AVAILABLE:
            self.logger.error("XGBoost not available but required for model type")
            return False
        
        model_config = self.config.model_training
        
        try:
            # Convert to appropriate format for XGBoost
            if isinstance(self.X_train, pd.DataFrame):
                X_train = self.X_train
                feature_names = self.X_train.columns.tolist()
            else:
                X_train = self.X_train
                feature_names = self.feature_names
            
            y_train = self.y_train
            
            # Prepare validation data if available
            eval_sets = []
            eval_names = []
            
            if self.X_val is not None and self.y_val is not None:
                X_val = self.X_val
                y_val = self.y_val
                eval_sets.append((X_val, y_val))
                eval_names.append('validation')
            
            # Add training data as an evaluation set
            eval_sets.append((X_train, y_train))
            eval_names.append('train')
            
            # Prepare sample weights if available
            sample_weights = self.sample_weights
            
            # Setup early stopping if enabled
            early_stopping_rounds = None
            if model_config.early_stopping and len(eval_sets) > 1:
                early_stopping_rounds = model_config.early_stopping_patience
            
            # Set number of training rounds (epochs)
            num_boost_round = model_config.num_epochs or 100
            
            # Callbacks for XGBoost
            callbacks = []
            
            # Add checkpointing callback if enabled
            if model_config.enable_checkpointing:
                # XGBoost has a built-in checkpointing mechanism
                checkpoint_callback = xgb.callback.TrainingCheckPoint(
                    directory=CHECKPOINT_PATH,
                    iterations=model_config.checkpoint_interval,
                    name=f"xgb_checkpoint_{self.config.model_registry.model_name}"
                )
                callbacks.append(checkpoint_callback)
            
            # Custom callback for metrics tracking
            class MetricsCallback(xgb.callback.TrainingCallback):
                def __init__(self, training_engine):
                    self.training_engine = training_engine
                    
                def after_iteration(self, model, epoch, evals_log):
                    # Get metrics from evals_log
                    train_metrics = {}
                    val_metrics = {}
                    
                    # Extract metrics for training set
                    for metric, values in evals_log.get('train', {}).items():
                        train_metrics[metric] = values[-1]
                    
                    # Extract metrics for validation set if available
                    for metric, values in evals_log.get('validation', {}).items():
                        val_metrics[metric] = values[-1]
                    
                    # Update metrics in the training engine
                    self.training_engine.metrics.update_epoch(epoch, train_metrics, val_metrics)
                    
                    # Log progress
                    self.training_engine.logger.info(f"Epoch {epoch}/{num_boost_round} - "
                                                f"Train: {train_metrics}, "
                                                f"Validation: {val_metrics}")
                    
                    return False  # Continue training
            
            # Add metrics callback
            callbacks.append(MetricsCallback(self))
            
            # Start metrics tracking
            self.metrics.start_training(num_boost_round)
            
            # Convert to DMatrix for better performance
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, feature_names=feature_names)
            
            evals = [(dtrain, 'train')]
            if len(eval_sets) > 1:
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
                evals.append((dval, 'validation'))
            
            # Create parameters dictionary
            params = self.model.get_params()
            
            # Remove non-booster parameters that are passed separately to train
            for param in ['n_estimators', 'early_stopping_rounds', 'verbose']:
                if param in params:
                    del params[param]
            
            # Train the model
            self.logger.info(f"Training XGBoost model with {num_boost_round} boosting rounds")
            
            # Use Booster.train API for more flexibility
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                callbacks=callbacks,
                verbose_eval=False  # We handle logging in the callback
            )
            
            # Set feature names in the model for later use
            self.model.feature_names = feature_names
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False

    def _train_lightgbm_model(self) -> bool:
        """Train a LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            self.logger.error("LightGBM not available but required for model type")
            return False
        
        model_config = self.config.model_training
        
        try:
            # Prepare data for LightGBM
            if isinstance(self.X_train, pd.DataFrame):
                X_train = self.X_train
                feature_names = self.X_train.columns.tolist()
            else:
                X_train = self.X_train
                feature_names = self.feature_names
            
            y_train = self.y_train
            
            # Prepare validation data if available
            eval_sets = []
            eval_names = []
            
            if self.X_val is not None and self.y_val is not None:
                X_val = self.X_val
                y_val = self.y_val
                eval_sets.append((X_val, y_val))
                eval_names.append('validation')
            
            # Prepare sample weights if available
            sample_weights = self.sample_weights
            
            # Set number of training rounds (epochs)
            num_boost_round = model_config.num_epochs or 100
            
            # Setup early stopping if enabled
            early_stopping_rounds = None
            if model_config.early_stopping and len(eval_sets) > 0:
                early_stopping_rounds = model_config.early_stopping_patience
            
            # Create LightGBM datasets
            lgb_train = lgb.Dataset(
                X_train, 
                y_train, 
                weight=sample_weights,
                feature_name=feature_names
            )
            
            lgb_eval = None
            if len(eval_sets) > 0:
                lgb_eval = lgb.Dataset(
                    X_val, 
                    y_val, 
                    reference=lgb_train,
                    feature_name=feature_names
                )
            
            # Create parameters dictionary
            params = self.model.get_params()
            
            # Remove non-booster parameters that are passed separately to train
            for param in ['n_estimators', 'early_stopping_rounds', 'verbose']:
                if param in params:
                    del params[param]
            
            # Add objective and metric if not specified
            if 'objective' not in params:
                if model_config.problem_type == 'classification':
                    if np.unique(y_train).size <= 2:
                        params['objective'] = 'binary'
                    else:
                        params['objective'] = 'multiclass'
                        params['num_class'] = np.unique(y_train).size
                else:
                    params['objective'] = 'regression'
            
            if 'metric' not in params:
                if model_config.problem_type == 'classification':
                    if np.unique(y_train).size <= 2:
                        params['metric'] = 'binary_logloss'
                    else:
                        params['metric'] = 'multi_logloss'
                else:
                    params['metric'] = 'rmse'
            
            # Setup callbacks
            callbacks = []
            
            # Add checkpointing callback if enabled
            if model_config.enable_checkpointing:
                checkpoint_callback = lgb.callback.create_checkpoint(
                    checkpoint_interval=model_config.checkpoint_interval,
                    directory=CHECKPOINT_PATH,
                    name=f"lgb_checkpoint_{self.config.model_registry.model_name}"
                )
                callbacks.append(checkpoint_callback)
            
            # Custom callback for metrics tracking
            class MetricsCallback:
                def __init__(self, training_engine):
                    self.training_engine = training_engine
                    
                def __call__(self, env):
                    # Extract iteration (epoch)
                    epoch = env.iteration + 1
                    
                    # Extract metrics
                    train_metrics = {}
                    val_metrics = {}
                    
                    # For training metrics
                    for name, value in zip(env.evaluation_result_list[0][1::2], env.evaluation_result_list[0][2::2]):
                        train_metrics[name] = value
                    
                    # For validation metrics if available
                    if len(env.evaluation_result_list) > 1:
                        for name, value in zip(env.evaluation_result_list[1][1::2], env.evaluation_result_list[1][2::2]):
                            val_metrics[name] = value
                    
                    # Update metrics in the training engine
                    self.training_engine.metrics.update_epoch(epoch, train_metrics, val_metrics)
                    
                    # Log progress
                    self.training_engine.logger.info(f"Epoch {epoch}/{num_boost_round} - "
                                                f"Train: {train_metrics}, "
                                                f"Validation: {val_metrics}")
            
            # Add metrics callback
            callbacks.append(MetricsCallback(self))
            
            # Start metrics tracking
            self.metrics.start_training(num_boost_round)
            
            # Train the model
            self.logger.info(f"Training LightGBM model with {num_boost_round} boosting rounds")
            
            self.model = lgb.train(
                params=params,
                train_set=lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=[lgb_train, lgb_eval] if lgb_eval else [lgb_train],
                valid_names=['train', 'validation'] if lgb_eval else ['train'],
                callbacks=callbacks,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False  # We handle logging in the callback
            )
            
            # Attach feature names to the model for later use
            self.model.feature_names = feature_names
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False

    def _train_custom_model(self) -> bool:
        """Train a custom model."""
        self.logger.info("Training custom model...")
        
        model_config = self.config.model_training
        
        try:
            # Check if model has the required train method
            if not hasattr(self.model, 'train') and not hasattr(self.model, 'fit'):
                self.logger.error("Custom model must implement either train() or fit() method")
                return False
            
            # Determine which method to use
            train_method = getattr(self.model, 'train', None) or getattr(self.model, 'fit')
            
            # Prepare data
            X_train = self.X_train
            y_train = self.y_train
            
            # Prepare validation data if available
            validation_data = None
            if self.X_val is not None and self.y_val is not None:
                validation_data = (self.X_val, self.y_val)
            
            # Get number of epochs
            num_epochs = model_config.num_epochs or 1
            
            # Start metrics tracking
            self.metrics.start_training(num_epochs)
            
            # Define callback for tracking metrics
            def metrics_callback(epoch, train_metrics, val_metrics=None):
                self.metrics.update_epoch(epoch, train_metrics, val_metrics)
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
                if val_metrics:
                    val_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                    metrics_str += f" | Validation: {val_str}"
                self.logger.info(f"Epoch {epoch}/{num_epochs} - {metrics_str}")
            
            # Create training parameters
            train_params = {
                'validation_data': validation_data,
                'epochs': num_epochs,
                'batch_size': model_config.batch_size,
                'callbacks': [metrics_callback],
                'early_stopping': model_config.early_stopping,
                'early_stopping_patience': model_config.early_stopping_patience
            }
            
            # Call the training method with appropriate parameters
            train_method(X_train, y_train, **train_params)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training custom model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False

    def _evaluate_sklearn_model(self, model, X, y) -> Dict[str, float]:
        """
        Evaluate a scikit-learn model and return metrics.
        
        Args:
            model: The model to evaluate
            X: Features
            y: Target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"error": 1.0}
        
        metrics = {}
        model_config = self.config.model_training
        problem_type = model_config.problem_type
        
        try:
            # Make predictions
            if hasattr(model, 'predict_proba') and problem_type == 'classification':
                y_pred_proba = model.predict_proba(X)
                if y_pred_proba.shape[1] == 2:  # Binary classification
                    y_pred_proba = y_pred_proba[:, 1]  # Get positive class probabilities
                
                y_pred = (y_pred_proba > self.config.model_evaluation.classification_threshold).astype(int) \
                    if y_pred_proba.ndim == 1 else np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X)
            
            # Calculate metrics based on problem type
            if problem_type == 'classification':
                # Classification metrics
                metrics['accuracy'] = accuracy_score(y, y_pred)
                
                # For binary classification
                if len(np.unique(y)) <= 2:
                    metrics['precision'] = precision_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['recall'] = recall_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['f1'] = f1_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    
                    # Add ROC AUC if we have probabilities
                    if 'y_pred_proba' in locals():
                        metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
                else:
                    # Multi-class metrics
                    metrics['precision'] = precision_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['recall'] = recall_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['f1'] = f1_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
            
            else:
                # Regression metrics
                metrics['mse'] = mean_squared_error(y, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y, y_pred)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating scikit-learn model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"error": 1.0}

    def evaluate_model(self) -> bool:
        """
        Evaluate the trained model on training, validation, and test sets.
        
        Returns:
            Success flag
        """
        self.logger.info("Evaluating model...")
        
        with self.state_lock:
            if self.state not in [TrainingState.TRAINING, TrainingState.EVALUATING, TrainingState.COMPLETED]:
                self.logger.error(f"Cannot evaluate model from state {self.state}")
                return False
            self.state = TrainingState.EVALUATING
        
        try:
            # Ensure model is trained
            if self.model is None:
                self.logger.error("No model available for evaluation")
                return False
            
            model_config = self.config.model_training
            eval_config = self.config.model_evaluation
            
            # Evaluate on training set if enabled
            if eval_config.evaluate_on_train and self.X_train is not None:
                self.logger.info("Evaluating on training data...")
                
                if model_config.model_type == ModelType.SKLEARN:
                    train_metrics = self._evaluate_sklearn_model(self.model, self.X_train, self.y_train)
                elif model_config.model_type == ModelType.XGBOOST:
                    train_metrics = self._evaluate_xgboost_model(self.X_train, self.y_train)
                elif model_config.model_type == ModelType.LIGHTGBM:
                    train_metrics = self._evaluate_lightgbm_model(self.X_train, self.y_train)
                else:
                    train_metrics = self._evaluate_custom_model(self.X_train, self.y_train)
                    
                self.logger.info(f"Training metrics: {train_metrics}")
            
            # Evaluate on validation set if enabled and available
            if eval_config.evaluate_on_validation and self.X_val is not None:
                self.logger.info("Evaluating on validation data...")
                
                if model_config.model_type == ModelType.SKLEARN:
                    val_metrics = self._evaluate_sklearn_model(self.model, self.X_val, self.y_val)
                elif model_config.model_type == ModelType.XGBOOST:
                    val_metrics = self._evaluate_xgboost_model(self.X_val, self.y_val)
                elif model_config.model_type == ModelType.LIGHTGBM:
                    val_metrics = self._evaluate_lightgbm_model(self.X_val, self.y_val)
                else:
                    val_metrics = self._evaluate_custom_model(self.X_val, self.y_val)
                    
                self.logger.info(f"Validation metrics: {val_metrics}")
            
            # Evaluate on test set if enabled and available
            if eval_config.evaluate_on_test and self.X_test is not None:
                self.logger.info("Evaluating on test data...")
                
                if model_config.model_type == ModelType.SKLEARN:
                    test_metrics = self._evaluate_sklearn_model(self.model, self.X_test, self.y_test)
                elif model_config.model_type == ModelType.XGBOOST:
                    test_metrics = self._evaluate_xgboost_model(self.X_test, self.y_test)
                elif model_config.model_type == ModelType.LIGHTGBM:
                    test_metrics = self._evaluate_lightgbm_model(self.X_test, self.y_test)
                else:
                    test_metrics = self._evaluate_custom_model(self.X_test, self.y_test)
                    
                self.logger.info(f"Test metrics: {test_metrics}")
                
                # Store test metrics
                self.metrics.set_test_metrics(test_metrics)
            
            # Generate additional evaluation artifacts if enabled
            if model_config.problem_type == 'classification':
                if eval_config.confusion_matrix and self.X_test is not None:
                    self._generate_confusion_matrix(self.X_test, self.y_test)
                    
                if eval_config.roc_curve and self.X_test is not None:
                    self._generate_roc_curve(self.X_test, self.y_test)
                    
                if eval_config.precision_recall_curve and self.X_test is not None:
                    self._generate_precision_recall_curve(self.X_test, self.y_test)
            
            # Calculate permutation importance if enabled
            if eval_config.compute_permutation_importance and self.X_test is not None:
                self._compute_permutation_importance()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            
            with self.state_lock:
                self.state = TrainingState.ERROR
                
            return False

    def _evaluate_xgboost_model(self, X, y) -> Dict[str, float]:
        """Evaluate an XGBoost model."""
        metrics = {}
        model_config = self.config.model_training
        problem_type = model_config.problem_type
        
        try:
            # Create DMatrix for prediction
            dtest = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
            
            # Make predictions
            if problem_type == 'classification':
                y_pred_proba = self.model.predict(dtest)
                
                # For binary classification
                if len(np.unique(y)) <= 2:
                    y_pred = (y_pred_proba > self.config.model_evaluation.classification_threshold).astype(int)
                    
                    # Classification metrics
                    metrics['accuracy'] = accuracy_score(y, y_pred)
                    metrics['precision'] = precision_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['recall'] = recall_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['f1'] = f1_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
                    
                else:
                    # Multi-class
                    y_pred = np.argmax(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else np.round(y_pred_proba).astype(int)
                    
                    # Classification metrics
                    metrics['accuracy'] = accuracy_score(y, y_pred)
                    metrics['precision'] = precision_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['recall'] = recall_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['f1'] = f1_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    
            else:
                # Regression
                y_pred = self.model.predict(dtest)
                
                # Regression metrics
                metrics['mse'] = mean_squared_error(y, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y, y_pred)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating XGBoost model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"error": 1.0}

    def _evaluate_lightgbm_model(self, X, y) -> Dict[str, float]:
        """Evaluate a LightGBM model."""
        metrics = {}
        model_config = self.config.model_training
        problem_type = model_config.problem_type
        
        try:
            # Make predictions
            if problem_type == 'classification':
                if hasattr(self.model, 'predict_proba'):
                    y_pred_proba = self.model.predict_proba(X)
                else:
                    y_pred_proba = self.model.predict(X, raw_score=False, pred_leaf=False, pred_contrib=False, num_iteration=None)
                
                # For binary classification
                if len(np.unique(y)) <= 2:
                    if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim > 1:
                        y_pred_proba = y_pred_proba[:, 1]  # Get positive class probabilities
                    
                    y_pred = (y_pred_proba > self.config.model_evaluation.classification_threshold).astype(int)
                    
                    # Classification metrics
                    metrics['accuracy'] = accuracy_score(y, y_pred)
                    metrics['precision'] = precision_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['recall'] = recall_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['f1'] = f1_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
                    
                else:
                    # Multi-class
                    y_pred = np.argmax(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else np.round(y_pred_proba).astype(int)
                    
                    # Classification metrics
                    metrics['accuracy'] = accuracy_score(y, y_pred)
                    metrics['precision'] = precision_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['recall'] = recall_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    metrics['f1'] = f1_score(y, y_pred, average=self.config.model_evaluation.average_method, zero_division=0)
                    
            else:
                # Regression
                y_pred = self.model.predict(X)
                
                # Regression metrics
                metrics['mse'] = mean_squared_error(y, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y, y_pred)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating LightGBM model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"error": 1.0}

    def _evaluate_custom_model(self, X, y) -> Dict[str, float]:
        """Evaluate a custom model."""
        metrics = {}
        model_config = self.config.model_training
        problem_type = model_config.problem_type
        
        try:
            # Check if model has a custom evaluate method
            if hasattr(self.model, 'evaluate'):
                return self.model.evaluate(X, y)
            
            # Otherwise implement standard evaluation based on problem type
            if problem_type == 'classification':
                # Check for predict_proba method
                if hasattr(self.model, 'predict_proba'):
                    y_pred_proba = self.model.predict_proba(X)
                    
                    # For binary classification
                    if len(np.unique(y)) <= 2:
                        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                            y_pred_proba = y_pred_proba[:, 1]  # Get positive class probabilities
                        
                        threshold = self.config.model_evaluation.classification_threshold
                        y_pred = (y_pred_proba > threshold).astype(int)
                        
                        # Classification metrics
                        metrics['accuracy'] = accuracy_score(y, y_pred)
                        metrics['precision'] = precision_score(y, y_pred, 
                                                            average=self.config.model_evaluation.average_method, 
                                                            zero_division=0)
                        metrics['recall'] = recall_score(y, y_pred, 
                                                        average=self.config.model_evaluation.average_method, 
                                                        zero_division=0)
                        metrics['f1'] = f1_score(y, y_pred, 
                                                average=self.config.model_evaluation.average_method, 
                                                zero_division=0)
                        metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
                    else:
                        # Multi-class
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        
                        # Classification metrics
                        metrics['accuracy'] = accuracy_score(y, y_pred)
                        metrics['precision'] = precision_score(y, y_pred, 
                                                            average=self.config.model_evaluation.average_method, 
                                                            zero_division=0)
                        metrics['recall'] = recall_score(y, y_pred, 
                                                        average=self.config.model_evaluation.average_method, 
                                                        zero_division=0)
                        metrics['f1'] = f1_score(y, y_pred, 
                                                average=self.config.model_evaluation.average_method, 
                                                zero_division=0)
                else:
                    # Use predict method for hard classification
                    y_pred = self.model.predict(X)
                    
                    # Classification metrics
                    metrics['accuracy'] = accuracy_score(y, y_pred)
                    metrics['precision'] = precision_score(y, y_pred, 
                                                        average=self.config.model_evaluation.average_method, 
                                                        zero_division=0)
                    metrics['recall'] = recall_score(y, y_pred, 
                                                    average=self.config.model_evaluation.average_method, 
                                                    zero_division=0)
                    metrics['f1'] = f1_score(y, y_pred, 
                                            average=self.config.model_evaluation.average_method, 
                                            zero_division=0)
            else:
                # Regression
                y_pred = self.model.predict(X)
                
                # Regression metrics
                metrics['mse'] = mean_squared_error(y, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y, y_pred)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating custom model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"error": 1.0}

    def _generate_confusion_matrix(self, X, y_true) -> bool:
        """
        Generate and save a confusion matrix visualization.
        
        Args:
            X: Features
            y_true: True labels
            
        Returns:
            Success flag
        """
        try:
            self.logger.info("Generating confusion matrix...")
            
            # Make predictions
            if hasattr(self.model, 'predict_proba') and self.config.model_training.problem_type == 'classification':
                y_proba = self.model.predict_proba(X)
                if y_proba.ndim > 1 and y_proba.shape[1] == 2:  # Binary classification
                    threshold = self.config.model_evaluation.classification_threshold
                    y_pred = (y_proba[:, 1] > threshold).astype(int)
                else:  # Multi-class
                    y_pred = np.argmax(y_proba, axis=1)
            else:
                y_pred = self.model.predict(X)
                
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Get class names if available
            class_names = self.config.model_evaluation.class_names
            if class_names is None:
                if hasattr(self, 'encoder') and hasattr(self.encoder, 'classes_'):
                    class_names = list(self.encoder.classes_)
                else:
                    class_names = [str(i) for i in range(len(np.unique(y_true)))]
            
            # Create figure
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, 
                        yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            # Save figure
            artifact_path = os.path.join(self.artifacts_path, 'confusion_matrix.png')
            plt.savefig(artifact_path)
            plt.close()
            
            self.logger.info(f"Confusion matrix saved to {artifact_path}")
            
            # Add to artifacts registry
            self.artifacts.register('confusion_matrix', artifact_path, 'image/png')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating confusion matrix: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False

    def _generate_roc_curve(self, X, y_true) -> bool:
        """
        Generate and save a ROC curve visualization.
        
        Args:
            X: Features
            y_true: True labels
            
        Returns:
            Success flag
        """
        try:
            self.logger.info("Generating ROC curve...")
            
            # Only applicable for classification
            if self.config.model_training.problem_type != 'classification':
                self.logger.warning("ROC curve only applicable for classification problems")
                return False
            
            # Check if model has predict_proba method
            if not hasattr(self.model, 'predict_proba'):
                self.logger.warning("Model does not have predict_proba method, required for ROC curve")
                return False
            
            # Make predictions
            y_proba = self.model.predict_proba(X)
            
            plt.figure(figsize=(10, 8))
            
            # Binary classification
            if len(np.unique(y_true)) <= 2:
                if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                    y_proba = y_proba[:, 1]  # Get positive class probabilities
                
                # Compute ROC curve and ROC area
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                
            else:
                # Multi-class ROC curve
                n_classes = len(np.unique(y_true))
                
                # Binarize the labels for one-vs-rest ROC
                lb = LabelBinarizer()
                lb.fit(y_true)
                y_true_bin = lb.transform(y_true)
                
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    plt.plot(fpr[i], tpr[i], lw=2, 
                            label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Multi-class Receiver Operating Characteristic')
                plt.legend(loc="lower right")
            
            # Save figure
            artifact_path = os.path.join(self.artifacts_path, 'roc_curve.png')
            plt.savefig(artifact_path)
            plt.close()
            
            self.logger.info(f"ROC curve saved to {artifact_path}")
            
            # Add to artifacts registry
            self.artifacts.register('roc_curve', artifact_path, 'image/png')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating ROC curve: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False

    def _generate_precision_recall_curve(self, X, y_true) -> bool:
        """
        Generate and save a Precision-Recall curve visualization.
        
        Args:
            X: Features
            y_true: True labels
            
        Returns:
            Success flag
        """
        try:
            self.logger.info("Generating Precision-Recall curve...")
            
            # Only applicable for classification
            if self.config.model_training.problem_type != 'classification':
                self.logger.warning("Precision-Recall curve only applicable for classification problems")
                return False
            
            # Check if model has predict_proba method
            if not hasattr(self.model, 'predict_proba'):
                self.logger.warning("Model does not have predict_proba method, required for Precision-Recall curve")
                return False
            
            # Make predictions
            y_proba = self.model.predict_proba(X)
            
            plt.figure(figsize=(10, 8))
            
            # Binary classification
            if len(np.unique(y_true)) <= 2:
                if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                    y_proba = y_proba[:, 1]  # Get positive class probabilities
                
                # Compute Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                pr_auc = auc(recall, precision)
                
                # Plot Precision-Recall curve
                plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title('Precision-Recall curve')
                plt.legend(loc="lower left")
                
            else:
                # Multi-class Precision-Recall curve
                n_classes = len(np.unique(y_true))
                
                # Binarize the labels for one-vs-rest PR curve
                lb = LabelBinarizer()
                lb.fit(y_true)
                y_true_bin = lb.transform(y_true)
                
                # Compute PR curve for each class
                for i in range(n_classes):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                    pr_auc = auc(recall, precision)
                    plt.plot(recall, precision, lw=2, 
                            label=f'PR curve of class {i} (area = {pr_auc:.2f})')
                
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title('Multi-class Precision-Recall curve')
                plt.legend(loc="lower left")
            
            # Save figure
            artifact_path = os.path.join(self.artifacts_path, 'precision_recall_curve.png')
            plt.savefig(artifact_path)
            plt.close()
            
            self.logger.info(f"Precision-Recall curve saved to {artifact_path}")
            
            # Add to artifacts registry
            self.artifacts.register('precision_recall_curve', artifact_path, 'image/png')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating Precision-Recall curve: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False

    def _compute_permutation_importance(self) -> bool:
        """
        Calculate and save permutation importance of features.
        
        Returns:
            Success flag
        """
        try:
            self.logger.info("Computing permutation importance...")
            
            # Get test data
            X_test = self.X_test
            y_test = self.y_test
            
            if X_test is None or y_test is None:
                self.logger.warning("Test data not available for permutation importance")
                return False
            
            # Get feature names
            feature_names = self.feature_names
            if feature_names is None and isinstance(X_test, pd.DataFrame):
                feature_names = X_test.columns.tolist()
            
            # Define scoring metric
            if self.config.model_training.problem_type == 'classification':
                if len(np.unique(y_test)) <= 2:
                    scorer = make_scorer(roc_auc_score, needs_proba=True)
                else:
                    scorer = make_scorer(accuracy_score)
            else:
                scorer = make_scorer(r2_score)
            
            # Calculate permutation importance
            r = permutation_importance(
                self.model, X_test, y_test, 
                n_repeats=self.config.model_evaluation.permutation_importance_repeats,
                random_state=self.config.random_seed,
                scoring=scorer
            )
            
            # Create DataFrame with importance results
            importance_df = pd.DataFrame({
                'Feature': feature_names if feature_names else [f'Feature_{i}' for i in range(X_test.shape[1])],
                'Importance': r.importances_mean,
                'StdDev': r.importances_std
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Save to CSV
            csv_path = os.path.join(self.artifacts_path, 'feature_importance.csv')
            importance_df.to_csv(csv_path, index=False)
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            plt.barh(
                importance_df['Feature'][:20],  # Show top 20 features for readability
                importance_df['Importance'][:20], 
                xerr=importance_df['StdDev'][:20],
                alpha=0.8
            )
            plt.xlabel('Permutation Importance')
            plt.title('Feature Importance (Permutation)')
            plt.tight_layout()
            
            # Save figure
            plot_path = os.path.join(self.artifacts_path, 'feature_importance.png')
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Feature importance data saved to {csv_path}")
            self.logger.info(f"Feature importance plot saved to {plot_path}")
            
            # Add to artifacts registry
            self.artifacts.register('feature_importance_data', csv_path, 'text/csv')
            self.artifacts.register('feature_importance_plot', plot_path, 'image/png')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error computing permutation importance: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False
