# ML Configuration System Documentation

## Overview
This module provides a comprehensive configuration system for machine learning model optimization, preprocessing, inference, and training. It defines a set of dataclasses and enums that enable fine-grained control over various aspects of machine learning workflows, from data preprocessing to model deployment, with special focus on quantization, batch processing, and model optimization.

## Prerequisites
- Python ≥3.6
- Required packages:
  ```bash
  pip install numpy dataclasses typing enum34
  ```

## Installation
No specific installation is required beyond the dependencies. Import the module directly in your Python code.

## Usage
```python
from ml_config import QuantizationConfig, BatchProcessorConfig, PreprocessorConfig, InferenceEngineConfig

# Create a quantization configuration for int8
quant_config = QuantizationConfig(
    quantization_type="int8",
    quantization_mode="dynamic_per_batch",
    symmetric=True
)

# Configure a batch processor
batch_config = BatchProcessorConfig(
    initial_batch_size=64,
    max_batch_size=128,
    enable_monitoring=True
)

# Create an inference engine configuration
inference_config = InferenceEngineConfig(
    enable_intel_optimization=True,
    enable_batching=True,
    quantization_config=quant_config,
    batch_processor_config=batch_config
)

# Export configuration to dict for serialization
config_dict = inference_config.to_dict()
```

## Configuration
The configuration system is organized into several modules, each handling a specific aspect of the machine learning pipeline.

---

## Constants

```python
CHECKPOINT_PATH = "./checkpoints"
MODEL_REGISTRY_PATH = "./model_registry"
```

- **CHECKPOINT_PATH**: Default path for model checkpoints.
- **MODEL_REGISTRY_PATH**: Default path for the model registry.

---

## Quantization Configuration

### Enums

#### `QuantizationType`
```python
class QuantizationType(Enum):
    NONE = "none"
    INT8 = "int8"
    UINT8 = "uint8"
    INT16 = "int16"
    FLOAT16 = "float16"
    MIXED = "mixed"
```

- **Description**:  
  Defines the available quantization types for model optimization.

- **Values**:
  - `NONE`: No quantization
  - `INT8`: 8-bit signed integer quantization
  - `UINT8`: 8-bit unsigned integer quantization
  - `INT16`: 16-bit signed integer quantization
  - `FLOAT16`: 16-bit floating point quantization
  - `MIXED`: Mixed precision quantization

#### `QuantizationMode`
```python
class QuantizationMode(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    DYNAMIC_PER_BATCH = "dynamic_per_batch"
    DYNAMIC_PER_CHANNEL = "dynamic_per_channel"
    SYMMETRIC = "symmetric"
    CALIBRATED = "calibrated"
```

- **Description**:  
  Defines the available quantization modes for model optimization.

- **Values**:
  - `STATIC`: Static quantization with fixed scales
  - `DYNAMIC`: Dynamic quantization at runtime
  - `DYNAMIC_PER_BATCH`: Dynamic quantization computed per batch
  - `DYNAMIC_PER_CHANNEL`: Dynamic quantization computed per channel
  - `SYMMETRIC`: Symmetric quantization around zero
  - `CALIBRATED`: Calibrated quantization using sample data

### Classes

#### `QuantizationConfig`
```python
@dataclass
class QuantizationConfig:
```

- **Description**:  
  Configuration class for model quantization parameters. Controls how models are quantized to reduce memory footprint and improve inference speed.

- **Attributes**:
  - `quantization_type (Union[str, QuantizationType])`: Type of quantization. Default is `QuantizationType.INT8`.
  - `quantization_mode (Union[str, QuantizationMode])`: Mode of quantization. Default is `QuantizationMode.DYNAMIC_PER_BATCH`.
  - `per_channel (bool)`: Whether to use per-channel quantization. Default is `False`.
  - `symmetric (bool)`: Whether to use symmetric quantization. Default is `True`.
  - `enable_cache (bool)`: Whether to cache quantized values. Default is `True`.
  - `cache_size (int)`: Size of the quantization cache. Default is `256`.
  - `calibration_samples (int)`: Number of samples to use for calibration. Default is `100`.
  - `calibration_method (str)`: Method for calibration. Default is `"percentile"`.
  - `percentile (float)`: Percentile value for calibration. Default is `99.99`.
  - `skip_layers (List[str])`: Layers to skip during quantization. Default is empty list.
  - `quantize_weights_only (bool)`: Whether to quantize only weights. Default is `False`.
  - `quantize_activations (bool)`: Whether to quantize activations. Default is `True`.
  - `weight_bits (int)`: Number of bits for weight quantization. Default is `8`.
  - `activation_bits (int)`: Number of bits for activation quantization. Default is `8`.
  - `quantize_bias (bool)`: Whether to quantize bias values. Default is `True`.
  - `bias_bits (int)`: Number of bits for bias quantization. Default is `32`.
  - `enable_mixed_precision (bool)`: Whether to enable mixed precision. Default is `False`.
  - `mixed_precision_layers (List[str])`: Layers to use mixed precision. Default is empty list.
  - `optimize_for (str)`: Optimization target. Default is `"performance"`.
  - `custom_quantization_config (Dict)`: Custom quantization parameters. Default is empty dict.
  - `enable_requantization (bool)`: Whether to enable requantization. Default is `False`.
  - `requantization_threshold (float)`: Threshold for requantization. Default is `0.1`.
  - `use_percentile (bool)`: Whether to use percentile for range calculation. Default is `False`.
  - `min_percentile (float)`: Minimum percentile for range calculation. Default is `0.1`.
  - `max_percentile (float)`: Maximum percentile for range calculation. Default is `99.9`.
  - `error_on_nan (bool)`: Whether to raise error on NaN values. Default is `False`.
  - `error_on_inf (bool)`: Whether to raise error on Inf values. Default is `False`.
  - `outlier_threshold (Optional[float])`: Threshold for outlier detection. Default is `None`.
  - `num_bits (int)`: Custom bit width for quantization. Default is `8`.
  - `optimize_memory (bool)`: Whether to optimize memory usage. Default is `True`.
  - `buffer_size (int)`: Size of buffer for quantization. Default is `0` (no buffer).

- **Methods**:

  ##### `__post_init__(self) -> None`
  ```python
  def __post_init__(self)
  ```
  - **Description**:  
    Initializes the object after dataclass initialization. Converts string values to corresponding enum values.
  
  - **Returns**:  
    - `None`

  ##### `to_dict(self) -> Dict`
  ```python
  def to_dict(self) -> Dict
  ```
  - **Description**:  
    Converts the configuration to a dictionary for serialization. Includes nested configurations.
    
  - **Returns**:  
    - `Dict`: Dictionary representation of the configuration.

  ##### `from_dict(cls, config_dict: Dict) -> 'InferenceEngineConfig'`
  ```python
  @classmethod
  def from_dict(cls, config_dict: Dict) -> 'InferenceEngineConfig'
  ```
  - **Description**:  
    Creates an InferenceEngineConfig instance from a dictionary.
    
  - **Parameters**:  
    - `config_dict (Dict)`: Dictionary containing configuration parameters.
    
  - **Returns**:  
    - `InferenceEngineConfig`: New InferenceEngineConfig instance.

  ##### `check_compatibility(self) -> Dict[str, bool]`
  ```python
  def check_compatibility(self) -> Dict[str, bool]
  ```
  - **Description**:  
    Checks if the configuration is compatible with available libraries.
    
  - **Returns**:  
    - `Dict[str, bool]`: Dictionary with compatibility results for various optimizations and libraries.

---

## Training Engine Configuration

### Enums

#### `TaskType`
```python
class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    MULTI_LABEL = "multi_label"
    RANKING = "ranking"
```

- **Description**:  
  Defines the supported task types for machine learning models.

- **Values**:
  - `CLASSIFICATION`: Binary or multi-class classification
  - `REGRESSION`: Regression task
  - `CLUSTERING`: Clustering task
  - `ANOMALY_DETECTION`: Anomaly or outlier detection
  - `TIME_SERIES`: Time series forecasting or analysis
  - `MULTI_LABEL`: Multi-label classification
  - `RANKING`: Ranking or recommendation task

#### `OptimizationStrategy`
```python
class OptimizationStrategy(Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"
    ASHT = "adaptive_surrogate_assisted_hyperparameter_tuning"
    HYPERX = "hyper_optimization_x"
    OPTUNA = "optuna"
    SUCCESSIVE_HALVING = "successive_halving"
    BOHB = "bayesian_optimization_hyperband"
```

- **Description**:  
  Defines strategies for hyperparameter optimization.

- **Values**:
  - `GRID_SEARCH`: Exhaustive search over specified parameter values
  - `RANDOM_SEARCH`: Random sampling from parameter distributions
  - `BAYESIAN_OPTIMIZATION`: Sequential model-based optimization
  - `EVOLUTIONARY`: Evolutionary algorithms for optimization
  - `HYPERBAND`: Bandit-based optimization with adaptive resource allocation
  - `ASHT`: Adaptive surrogate-assisted hyperparameter tuning
  - `HYPERX`: Custom hybrid optimization algorithm
  - `OPTUNA`: Optuna framework for hyperparameter optimization
  - `SUCCESSIVE_HALVING`: Adaptive successive halving algorithm
  - `BOHB`: Combination of Bayesian optimization and Hyperband

#### `ModelSelectionCriteria`
```python
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
```

- **Description**:  
  Defines criteria for model selection during training.

- **Values**:
  - `ACCURACY`: Classification accuracy
  - `F1`: F1 score (harmonic mean of precision and recall)
  - `PRECISION`: Precision (positive predictive value)
  - `RECALL`: Recall (sensitivity)
  - `ROC_AUC`: Area under ROC curve
  - `Matthews_CORRELATION`: Matthews correlation coefficient
  - `ROOT_MEAN_SQUARED_ERROR`: Root mean squared error
  - `MEAN_ABSOLUTE_ERROR`: Mean absolute error
  - `R2`: R-squared (coefficient of determination)
  - `SILHOUETTE`: Silhouette coefficient for clustering
  - `EXPLAINED_VARIANCE`: Explained variance
  - `BIC`: Bayesian information criterion
  - `AIC`: Akaike information criterion
  - `CUSTOM`: Custom evaluation metric

#### `AutoMLMode`
```python
class AutoMLMode(Enum):
    DISABLED = "disabled"
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    EXPERIMENTAL = "experimental"
```

- **Description**:  
  Defines modes for automated machine learning.

- **Values**:
  - `DISABLED`: AutoML is disabled
  - `BASIC`: Basic AutoML with default settings
  - `COMPREHENSIVE`: Comprehensive AutoML with extensive model search
  - `EXPERIMENTAL`: Experimental AutoML with advanced techniques

### Classes

#### `ExplainabilityConfig`
```python
class ExplainabilityConfig:
```

- **Description**:  
  Configuration class for model explainability. Controls how model decisions are explained and interpreted.

- **Constructor**:
  ```python
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
  )
  ```
  - **Parameters**:
    - `enable_explainability (bool)`: Whether to enable model explainability. Default is `True`.
    - `methods (List[str])`: Explainability methods to use. Default is `["shap", "feature_importance", "partial_dependence"]`.
    - `default_method (str)`: Default explainability method. Default is `"shap"`.
    - `shap_algorithm (str)`: Algorithm for SHAP values calculation. Default is `"auto"`.
    - `shap_samples (int)`: Number of samples for SHAP analysis. Default is `100`.
    - `lime_samples (int)`: Number of samples for LIME analysis. Default is `1000`.
    - `feature_importance_permutations (int)`: Number of permutations for feature importance. Default is `10`.
    - `partial_dependence_features (List[str])`: Features for partial dependence plots. Default is empty list.
    - `partial_dependence_grid_points (int)`: Number of grid points for partial dependence. Default is `20`.
    - `eli5_permutation_samples (int)`: Number of samples for ELI5 permutation importance. Default is `100`.
    - `generate_summary (bool)`: Whether to generate summary of explanations. Default is `True`.
    - `generate_plots (bool)`: Whether to generate explanation plots. Default is `True`.
    - `explainability_report_format (str)`: Format for explainability reports. Default is `"html"`.
    - `store_explanations (bool)`: Whether to store explanations. Default is `True`.
    - `explanation_cache_size (int)`: Size of explanation cache. Default is `100`.
    - `enable_counterfactuals (bool)`: Whether to enable counterfactual explanations. Default is `False`.
    - `counterfactual_method (str)`: Method for counterfactual generation. Default is `"dice"`.
    - `custom_explainers (List)`: Custom explainer objects. Default is empty list.

- **Methods**:

  ##### `to_dict(self) -> Dict`
  ```python
  def to_dict(self) -> Dict
  ```
  - **Description**:  
    Converts the configuration to a dictionary for serialization.
    
  - **Returns**:  
    - `Dict`: Dictionary representation of the configuration.

#### `MonitoringConfig`
```python
class MonitoringConfig:
```

- **Description**:  
  Configuration class for model monitoring in production. Controls how model performance and data drift are monitored.

- **Constructor**:
  ```python
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
      performance_threshold: float = 0.1,
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
  )
  ```
  - **Parameters**:
    - `enable_monitoring (bool)`: Whether to enable model monitoring. Default is `True`.
    - `drift_detection (bool)`: Whether to enable drift detection. Default is `True`.
    - `drift_detection_method (str)`: Method for drift detection. Default is `"ks_test"`.
    - `drift_threshold (float)`: Threshold for drift detection. Default is `0.05`.
    - `drift_check_interval (str)`: Interval for drift checks. Default is `"1h"`.
    - `monitoring_interval (str)`: Interval for monitoring. Default is `"1h"`.
    - `performance_tracking (bool)`: Whether to track model performance. Default is `True`.
    - `performance_metrics (List[str])`: Performance metrics to track. Default is `["accuracy", "latency"]`.
    - `performance_threshold (float)`: Threshold for performance degradation. Default is `0.1`.
    - `alert_on_drift (bool)`: Whether to alert on drift detection. Default is `True`.
    - `alert_on_performance_drop (bool)`: Whether to alert on performance drop. Default is `True`.
    - `alert_channels (List[str])`: Channels for alerts. Default is `["log"]`.
    - `monitoring_report_interval (str)`: Interval for monitoring reports. Default is `"1d"`.
    - `data_sampling_rate (float)`: Rate for data sampling. Default is `1.0`.
    - `store_predictions (bool)`: Whether to store model predictions. Default is `True`.
    - `prediction_store_size (int)`: Size of prediction store. Default is `10000`.
    - `enable_auto_retraining (bool)`: Whether to enable automatic model retraining. Default is `False`.
    - `retraining_trigger_drift_threshold (float)`: Drift threshold for retraining. Default is `0.1`.
    - `retraining_trigger_performance_drop (float)`: Performance drop threshold for retraining. Default is `0.05`.
    - `export_metrics (bool)`: Whether to export metrics. Default is `False`.
    - `metrics_export_format (str)`: Format for metrics export. Default is `"prometheus"`.
    - `custom_monitors (List)`: Custom monitoring objects. Default is empty list.

- **Methods**:

  ##### `to_dict(self) -> Dict`
  ```python
  def to_dict(self) -> Dict
  ```
  - **Description**:  
    Converts the configuration to a dictionary for serialization.
    
  - **Returns**:  
    - `Dict`: Dictionary representation of the configuration.

#### `MLTrainingEngineConfig`
```python
class MLTrainingEngineConfig:
```

- **Description**:  
  Configuration class for the ML training engine. Controls the entire training process for machine learning models.

- **Constructor**:
  ```python
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
  )
  ```
  - **Parameters**: [Selected key parameters due to length constraints]
    - `task_type (Union[TaskType, str])`: Type of machine learning task. Default is `TaskType.CLASSIFICATION`.
    - `random_state (int)`: Random seed for reproducibility. Default is `42`.
    - `n_jobs (int)`: Number of jobs for parallel processing. Default is `-1` (all cores).
    - `cv_folds (int)`: Number of cross-validation folds. Default is `5`.
    - `test_size (float)`: Fraction of data to use for testing. Default is `0.2`.
    - `stratify (bool)`: Whether to use stratified sampling. Default is `True`.
    - `optimization_strategy (Union[OptimizationStrategy, str])`: Strategy for hyperparameter optimization. Default is `OptimizationStrategy.HYPERX`.
    - `optimization_iterations (int)`: Number of optimization iterations. Default is `50`.
    - `early_stopping (bool)`: Whether to use early stopping. Default is `True`.
    - `feature_selection (bool)`: Whether to perform feature selection. Default is `True`.
    - `preprocessing_config (Optional[PreprocessorConfig])`: Configuration for data preprocessing. Default is automatically created.
    - `batch_processing_config (Optional[BatchProcessorConfig])`: Configuration for batch processing. Default is automatically created.
    - `inference_config (Optional[InferenceEngineConfig])`: Configuration for inference. Default is automatically created.
    - `quantization_config (Optional[QuantizationConfig])`: Configuration for quantization. Default is automatically created.
    - `explainability_config (Optional[ExplainabilityConfig])`: Configuration for explainability. Default is automatically created.
    - `monitoring_config (Optional[MonitoringConfig])`: Configuration for monitoring. Default is automatically created.
    - `auto_ml (Union[bool, AutoMLMode])`: Whether to use AutoML and which mode. Default is `AutoMLMode.DISABLED`.
    - `ensemble_models (bool)`: Whether to use ensemble models. Default is `False`.
    - `model_selection_criteria (Union[str, ModelSelectionCriteria])`: Criteria for model selection. Default is `ModelSelectionCriteria.ACCURACY`.

- **Methods**:

  ##### `to_dict(self) -> Dict`
  ```python
  def to_dict(self) -> Dict
  ```
  - **Description**:  
    Converts the configuration to a dictionary for serialization. Includes nested configurations.
    
  - **Returns**:  
    - `Dict`: Dictionary representation of the configuration.

## Architecture

The configuration system is designed with a hierarchical structure:

```
ML Configuration System
│
├── Quantization Configuration
│   ├── QuantizationType (Enum)
│   ├── QuantizationMode (Enum)
│   └── QuantizationConfig (Dataclass)
│
├── Batch Processing Configuration
│   ├── BatchProcessingStrategy (Enum)
│   ├── BatchPriority (Enum)
│   ├── PrioritizedItem (NamedTuple)
│   └── BatchProcessorConfig (Dataclass)
│
├── Data Preprocessor Configuration
│   ├── NormalizationType (Enum)
│   └── PreprocessorConfig (Dataclass)
│
├── Inference Engine Configuration
│   ├── ModelType (Enum)
│   ├── EngineState (Enum)
│   ├── OptimizationMode (Enum)
│   └── InferenceEngineConfig (Dataclass)
│
└── Training Engine Configuration
    ├── TaskType (Enum)
    ├── OptimizationStrategy (Enum)
    ├── ModelSelectionCriteria (Enum)
    ├── AutoMLMode (Enum)
    ├── ExplainabilityConfig (Class)
    ├── MonitoringConfig (Class)
    └── MLTrainingEngineConfig (Class)
```

## Testing
```bash
python -m unittest tests/test_config.py
```

## Security & Compliance
- The configuration system supports secure serialization and deserialization.
- Custom configurations are sanitized to prevent injection attacks.
- Configuration validation helps prevent misconfigurations.

> Last Updated: 2025-05-11ization.
    
  - **Returns**:  
    - `Dict`: Dictionary representation of the configuration.

  ##### `from_dict(cls, config_dict: Dict) -> 'QuantizationConfig'`
  ```python
  @classmethod
  def from_dict(cls, config_dict: Dict) -> 'QuantizationConfig'
  ```
  - **Description**:  
    Creates a QuantizationConfig instance from a dictionary.
    
  - **Parameters**:  
    - `config_dict (Dict)`: Dictionary containing configuration parameters.
    
  - **Returns**:  
    - `QuantizationConfig`: New QuantizationConfig instance.

---

## Batch Processing Configuration

### Enums

#### `BatchProcessingStrategy`
```python
class BatchProcessingStrategy(Enum):
    FIXED = auto()
    ADAPTIVE = auto()
    GREEDY = auto()
```

- **Description**:  
  Defines strategies for batch processing.

- **Values**:
  - `FIXED`: Use fixed batch size
  - `ADAPTIVE`: Dynamically adjust batch size based on system load
  - `GREEDY`: Process as many items as available up to max_batch_size

#### `BatchPriority`
```python
class BatchPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4
```

- **Description**:  
  Defines priority levels for batch processing.

- **Values**:
  - `CRITICAL`: Highest priority (0)
  - `HIGH`: High priority (1)
  - `NORMAL`: Default priority (2)
  - `LOW`: Low priority (3)
  - `BACKGROUND`: Lowest priority (4)

### Classes

#### `PrioritizedItem`
```python
class PrioritizedItem(NamedTuple):
```

- **Description**:  
  A named tuple representing an item with priority for queue ordering.

- **Attributes**:
  - `priority (int)`: Priority level (lower number = higher priority)
  - `timestamp (float)`: Time when the item was added (for FIFO within same priority)
  - `item (Any)`: The actual item

- **Methods**:

  ##### `__lt__(self, other) -> bool`
  ```python
  def __lt__(self, other)
  ```
  - **Description**:  
    Comparison method for priority ordering. Compares first by priority, then by timestamp.
    
  - **Parameters**:  
    - `other (PrioritizedItem)`: Another PrioritizedItem to compare with.
    
  - **Returns**:  
    - `bool`: True if this item has higher priority (lower priority number or earlier timestamp)

#### `BatchProcessorConfig`
```python
@dataclass
class BatchProcessorConfig:
```

- **Description**:  
  Configuration class for batch processing parameters. Controls how data is batched and processed.

- **Attributes**:
  - `initial_batch_size (int)`: Initial size of batches. Default is `100`.
  - `min_batch_size (int)`: Minimum size of batches. Default is `50`.
  - `max_batch_size (int)`: Maximum size of batches. Default is `200`.
  - `max_queue_size (int)`: Maximum size of the processing queue. Default is `1000`.
  - `batch_timeout (float)`: Timeout for batch formation in seconds. Default is `1.0`.
  - `num_workers (int)`: Number of worker threads. Default is `4`.
  - `adaptive_batching (bool)`: Whether to adapt batch size dynamically. Default is `True`.
  - `batch_allocation_strategy (str)`: Strategy for allocating batches. Default is `"dynamic"`.
  - `enable_priority_queue (bool)`: Whether to enable priority queue. Default is `True`.
  - `priority_levels (int)`: Number of priority levels. Default is `3`.
  - `enable_monitoring (bool)`: Whether to enable performance monitoring. Default is `True`.
  - `monitoring_interval (float)`: Interval for monitoring in seconds. Default is `2.0`.
  - `enable_memory_optimization (bool)`: Whether to optimize memory usage. Default is `True`.
  - `enable_prefetching (bool)`: Whether to prefetch batches. Default is `True`.
  - `prefetch_batches (int)`: Number of batches to prefetch. Default is `2`.
  - `checkpoint_batches (bool)`: Whether to checkpoint batches. Default is `False`.
  - `checkpoint_interval (int)`: Interval for checkpointing in batches. Default is `100`.
  - `error_handling (str)`: Strategy for handling errors. Default is `"retry"`.
  - `max_retries (int)`: Maximum number of retries. Default is `3`.
  - `retry_delay (float)`: Delay between retries in seconds. Default is `0.5`.
  - `distributed_processing (bool)`: Whether to use distributed processing. Default is `False`.
  - `resource_allocation (Dict)`: Resource allocation parameters. Default is `None`.
  - `item_timeout (float)`: Maximum time to wait for item processing in seconds. Default is `10.0`.
  - `min_batch_interval (float)`: Minimum time between batch processing in seconds. Default is `0.0`.
  - `processing_strategy (BatchProcessingStrategy)`: Strategy for batch processing. Default is `BatchProcessingStrategy.ADAPTIVE`.
  - `enable_adaptive_batching (bool)`: Whether to enable adaptive batching. Default is `True`.
  - `reduce_batch_on_failure (bool)`: Whether to reduce batch size on failure. Default is `True`.
  - `max_batch_memory_mb (Optional[float])`: Maximum memory per batch in MB. Default is `None`.
  - `gc_batch_threshold (int)`: Number of batches after which to run garbage collection. Default is `32`.
  - `monitoring_window (int)`: Number of batches to keep statistics for. Default is `100`.
  - `max_workers (int)`: Maximum number of worker threads. Default is `4`.
  - `enable_health_monitoring (bool)`: Whether to enable health monitoring. Default is `True`.
  - `health_check_interval (float)`: Interval for health checks in seconds. Default is `5.0`.
  - `memory_warning_threshold (float)`: Memory usage percentage to trigger warning. Default is `70.0`.
  - `memory_critical_threshold (float)`: Memory usage percentage to trigger critical actions. Default is `85.0`.
  - `queue_warning_threshold (int)`: Queue size to trigger warning. Default is `100`.
  - `queue_critical_threshold (int)`: Queue size to trigger critical actions. Default is `500`.
  - `debug_mode (bool)`: Whether to enable debug mode. Default is `False`.

- **Methods**:

  ##### `__post_init__(self) -> None`
  ```python
  def __post_init__(self)
  ```
  - **Description**:  
    Initializes default values for resource allocation if None.
  
  - **Returns**:  
    - `None`

  ##### `to_dict(self) -> Dict`
  ```python
  def to_dict(self) -> Dict
  ```
  - **Description**:  
    Converts the configuration to a dictionary for serialization.
    
  - **Returns**:  
    - `Dict`: Dictionary representation of the configuration.

---

## Data Preprocessor Configuration

### Enums

#### `NormalizationType`
```python
class NormalizationType(Enum):
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"
    LOG = "log"
    POWER = "power"
```

- **Description**:  
  Defines the available normalization types for data preprocessing.

- **Values**:
  - `NONE`: No normalization
  - `STANDARD`: Standardization (z-score normalization)
  - `MINMAX`: Min-max scaling
  - `ROBUST`: Robust scaling using percentiles
  - `QUANTILE`: Quantile transformation
  - `LOG`: Logarithmic transformation
  - `POWER`: Power transformation

### Classes

#### `PreprocessorConfig`
```python
@dataclass
class PreprocessorConfig:
```

- **Description**:  
  Configuration class for data preprocessing parameters. Controls how data is preprocessed before model training or inference.

- **Attributes**:
  - `normalization (Union[NormalizationType, str])`: Type of normalization to apply. Default is `NormalizationType.STANDARD`.
  - `robust_percentiles (Tuple[float, float])`: Percentiles for robust scaling. Default is `(25.0, 75.0)`.
  - `handle_nan (bool)`: Whether to handle NaN values. Default is `True`.
  - `handle_inf (bool)`: Whether to handle infinite values. Default is `True`.
  - `nan_strategy (str)`: Strategy for handling NaN values. Default is `"mean"`.
  - `nan_fill_value (float)`: Value to use for filling NaN values. Default is `0.0`.
  - `inf_strategy (str)`: Strategy for handling infinite values. Default is `"max_value"`.
  - `pos_inf_fill_value (float)`: Value to use for positive infinity. Default is `np.nan`.
  - `neg_inf_fill_value (float)`: Value to use for negative infinity. Default is `np.nan`.
  - `inf_buffer (float)`: Multiplier for max/min when replacing infinities. Default is `1.0`.
  - `copy_X (bool)`: Whether to copy data for transformations. Default is `True`.
  - `detect_outliers (bool)`: Whether to detect outliers. Default is `True`.
  - `outlier_method (str)`: Method for outlier detection. Default is `"iqr"`.
  - `outlier_handling (str)`: Strategy for handling outliers. Default is `"clip"`.
  - `outlier_contamination (float)`: Expected proportion of outliers. Default is `0.05`.
  - `outlier_iqr_multiplier (float)`: Multiplier for IQR method. Default is `1.5`.
  - `outlier_zscore_threshold (float)`: Threshold for z-score method. Default is `3.0`.
  - `outlier_percentiles (Tuple[float, float])`: Percentiles for outlier detection. Default is `(1.0, 99.0)`.
  - `feature_outlier_mask (Optional[List[bool]])`: Mask for which features to apply outlier detection. Default is `None`.
  - `clip_values (bool)`: Whether to clip values to range. Default is `False`.
  - `clip_min (float)`: Minimum value for clipping. Default is `-inf`.
  - `clip_max (float)`: Maximum value for clipping. Default is `inf`.
  - `scale_shift (bool)`: Whether to apply shift after scaling. Default is `False`.
  - `scale_offset (float)`: Offset to apply after scaling. Default is `0.0`.
  - `custom_center_func (Optional[Callable])`: Custom function for computing center values. Default is `None`.
  - `custom_scale_func (Optional[Callable])`: Custom function for computing scale values. Default is `None`.
  - `custom_normalization_fn (Optional[Callable])`: Custom normalization function. Default is `None`.
  - `custom_transform_fn (Optional[Callable])`: Custom transform function. Default is `None`.
  - `custom_inverse_fn (Optional[Callable])`: Custom inverse transform function. Default is `None`.
  - `categorical_encoding (str)`: Method for encoding categorical features. Default is `"one_hot"`.
  - `categorical_max_categories (int)`: Maximum number of categories to encode. Default is `20`.
  - `auto_feature_selection (bool)`: Whether to enable automatic feature selection. Default is `True`.
  - `numeric_transformations (List[str])`: Transformations to apply to numeric features. Default is `None`.
  - `text_vectorization (Optional[str])`: Method for vectorizing text. Default is `"tfidf"`.
  - `text_max_features (int)`: Maximum number of features for text vectorization. Default is `5000`.
  - `dimension_reduction (Optional[str])`: Method for dimension reduction. Default is `None`.
  - `dimension_reduction_target (Optional[int])`: Target dimension for reduction. Default is `None`.
  - `datetime_features (bool)`: Whether to extract features from datetime. Default is `True`.
  - `image_preprocessing (Optional[Dict])`: Parameters for image preprocessing. Default is `None`.
  - `handle_imbalance (bool)`: Whether to handle class imbalance. Default is `False`.
  - `imbalance_strategy (str)`: Strategy for handling imbalance. Default is `"smote"`.
  - `feature_interaction (bool)`: Whether to create feature interactions. Default is `False`.
  - `feature_interaction_max (int)`: Maximum number of feature interactions. Default is `10`.
  - `custom_transformers (List)`: Custom transformer objects. Default is `None`.
  - `transformation_pipeline (List)`: Custom transformation pipeline. Default is `None`.
  - `parallel_processing (bool)`: Whether to use parallel processing. Default is `True`.
  - `cache_preprocessing (bool)`: Whether to cache preprocessing results. Default is `True`.
  - `verbosity (int)`: Verbosity level. Default is `1`.
  - `enable_input_validation (bool)`: Whether to validate input data. Default is `True`.
  - `input_size_limit (Optional[int])`: Maximum number of samples to process. Default is `None`.
  - `n_jobs (int)`: Number of jobs for parallel processing. Default is `-1` (all cores).
  - `chunk_size (Optional[int])`: Size of chunks for processing large data. Default is `None`.
  - `cache_enabled (bool)`: Whether to enable caching. Default is `True`.
  - `cache_size (int)`: Size of LRU cache. Default is `128`.
  - `dtype (np.dtype)`: Data type for preprocessed data. Default is `np.float64`.
  - `epsilon (float)`: Small value to avoid division by zero. Default is `1e-10`.
  - `debug_mode (bool)`: Whether to enable debug mode. Default is `False`.
  - `version (str)`: Version of the preprocessor. Default is `"1.0.0"`.

- **Methods**:

  ##### `__post_init__(self) -> None`
  ```python
  def __post_init__(self)
  ```
  - **Description**:  
    Initializes default values and converts string values to corresponding enum values.
  
  - **Returns**:  
    - `None`

  ##### `to_dict(self) -> Dict`
  ```python
  def to_dict(self) -> Dict
  ```
  - **Description**:  
    Converts the configuration to a dictionary for serialization. Handles special types and removes non-serializable callables.
    
  - **Returns**:  
    - `Dict`: Dictionary representation of the configuration.

  ##### `from_dict(cls, config_dict: Dict) -> 'PreprocessorConfig'`
  ```python
  @classmethod
  def from_dict(cls, config_dict: Dict) -> 'PreprocessorConfig'
  ```
  - **Description**:  
    Creates a PreprocessorConfig instance from a dictionary.
    
  - **Parameters**:  
    - `config_dict (Dict)`: Dictionary containing configuration parameters.
    
  - **Returns**:  
    - `PreprocessorConfig`: New PreprocessorConfig instance.

---

## Inference Engine Configuration

### Enums

#### `ModelType`
```python
class ModelType(Enum):
    SKLEARN = auto()
    XGBOOST = auto()
    LIGHTGBM = auto()
    CUSTOM = auto()
    ENSEMBLE = auto()
    ONNX = auto()
    PYTORCH = auto()
    TENSORFLOW = auto()
```

- **Description**:  
  Defines the supported model types for the inference engine.

- **Values**:
  - `SKLEARN`: scikit-learn models
  - `XGBOOST`: XGBoost models
  - `LIGHTGBM`: LightGBM models
  - `CUSTOM`: Custom model implementations
  - `ENSEMBLE`: Ensemble models
  - `ONNX`: ONNX format models
  - `PYTORCH`: PyTorch models
  - `TENSORFLOW`: TensorFlow models

#### `EngineState`
```python
class EngineState(Enum):
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    LOADING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()
```

- **Description**:  
  Defines the possible states of the inference engine.

- **Values**:
  - `INITIALIZING`: Engine is initializing
  - `READY`: Engine is ready for inference
  - `RUNNING`: Engine is currently processing inference requests
  - `LOADING`: Engine is loading a model
  - `STOPPING`: Engine is in the process of stopping
  - `STOPPED`: Engine is stopped
  - `ERROR`: Engine encountered an error

#### `OptimizationMode`
```python
class OptimizationMode(str, Enum):
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    PERFORMANCE = "performance"
    FULL_UTILIZATION = "full_utilization"
    MEMORY_SAVING = "memory_saving"
```

- **Description**:  
  Defines the optimization modes for the inference engine.

- **Values**:
  - `BALANCED`: Balance between performance and resource usage
  - `CONSERVATIVE`: Prioritize stability and minimal resource usage
  - `PERFORMANCE`: Prioritize maximum performance
  - `FULL_UTILIZATION`: Use all available resources
  - `MEMORY_SAVING`: Prioritize memory efficiency

### Classes

#### `InferenceEngineConfig`
```python
@dataclass
class InferenceEngineConfig:
```

- **Description**:  
  Configuration class for the inference engine. Controls how models are loaded and used for inference.

- **Attributes**:
  - `enable_intel_optimization (bool)`: Whether to enable Intel optimizations. Default is `True`.
  - `enable_batching (bool)`: Whether to enable request batching. Default is `True`.
  - `enable_quantization (bool)`: Whether to enable model quantization. Default is `False`.
  - `model_cache_size (int)`: Number of models to cache. Default is `5`.
  - `model_precision (str)`: Precision for model computation. Default is `"fp32"`.
  - `max_batch_size (int)`: Maximum batch size for inference. Default is `64`.
  - `timeout_ms (int)`: Timeout for inference in milliseconds. Default is `100`.
  - `enable_jit (bool)`: Whether to enable just-in-time compilation. Default is `True`.
  - `enable_onnx (bool)`: Whether to enable ONNX runtime. Default is `False`.
  - `onnx_opset (int)`: ONNX opset version. Default is `15`.
  - `enable_tensorrt (bool)`: Whether to enable TensorRT acceleration. Default is `False`.
  - `runtime_optimization (bool)`: Whether to enable runtime optimizations. Default is `True`.
  - `fallback_to_cpu (bool)`: Whether to fallback to CPU if GPU fails. Default is `True`.
  - `thread_count (int)`: Number of threads for inference. Default is `0` (auto-detect).
  - `warmup (bool)`: Whether to warmup the model before inference. Default is `True`.
  - `warmup_iterations (int)`: Number of warmup iterations. Default is `10`.
  - `profiling (bool)`: Whether to enable performance profiling. Default is `False`.
  - `batching_strategy (str)`: Strategy for batching requests. Default is `"dynamic"`.
  - `output_streaming (bool)`: Whether to stream output results. Default is `False`.
  - `debug_mode (bool)`: Whether to enable debug mode. Default is `False`.
  - `memory_growth (bool)`: Whether to enable memory growth for GPU. Default is `True`.
  - `custom_ops (List)`: Custom operations for the inference engine. Default is empty list.
  - `use_platform_accelerator (bool)`: Whether to use platform-specific accelerators. Default is `True`.
  - `platform_accelerator_config (Dict)`: Configuration for platform accelerators. Default is empty dict.
  - `model_version (str)`: Version of the model. Default is `"1.0"`.
  - `num_threads (int)`: Number of threads for computation. Default is `4`.
  - `set_cpu_affinity (bool)`: Whether to set CPU affinity for threads. Default is `False`.
  - `enable_model_quantization (bool)`: Whether to quantize the model. Default is `False`.
  - `enable_input_quantization (bool)`: Whether to quantize input data. Default is `False`.
  - `quantization_dtype (str)`: Data type for quantization. Default is `"int8"`.
  - `quantization_config (Optional[QuantizationConfig])`: Configuration for quantization. Default is `None`.
  - `enable_request_deduplication (bool)`: Whether to deduplicate identical requests. Default is `True`.
  - `max_cache_entries (int)`: Maximum number of cache entries. Default is `1000`.
  - `cache_ttl_seconds (int)`: Time-to-live for cache entries in seconds. Default is `300`.
  - `monitoring_window (int)`: Number of requests to track for monitoring. Default is `100`.
  - `enable_monitoring (bool)`: Whether to enable performance monitoring. Default is `True`.
  - `monitoring_interval (float)`: Interval for monitoring in seconds. Default is `10.0`.
  - `throttle_on_high_cpu (bool)`: Whether to throttle requests on high CPU usage. Default is `True`.
  - `cpu_threshold_percent (float)`: CPU usage threshold for throttling. Default is `90.0`.
  - `memory_high_watermark_mb (float)`: Memory high watermark in MB. Default is `1024.0`.
  - `memory_limit_gb (Optional[float])`: Memory limit in GB. Default is `None`.
  - `batch_timeout (float)`: Timeout for batch formation in seconds. Default is `0.1`.
  - `max_concurrent_requests (int)`: Maximum number of concurrent requests. Default is `8`.
  - `initial_batch_size (int)`: Initial batch size. Default is `16`.
  - `min_batch_size (int)`: Minimum batch size. Default is `1`.
  - `enable_adaptive_batching (bool)`: Whether to adapt batch size dynamically. Default is `True`.
  - `enable_memory_optimization (bool)`: Whether to optimize memory usage. Default is `True`.
  - `enable_feature_scaling (bool)`: Whether to scale input features. Default is `False`.
  - `enable_warmup (bool)`: Whether to warmup the model. Default is `True`.
  - `enable_quantization_aware_inference (bool)`: Whether to use quantization-aware inference. Default is `False`.
  - `enable_throttling (bool)`: Whether to enable request throttling. Default is `False`.
  - `optimization_mode (OptimizationMode)`: Mode for optimization. Default is `OptimizationMode.BALANCED`.
  - `enable_fp16_optimization (bool)`: Whether to enable FP16 optimization. Default is `False`.
  - `enable_compiler_optimization (bool)`: Whether to enable compiler optimizations. Default is `True`.
  - `preprocessor_config (Optional[PreprocessorConfig])`: Configuration for preprocessing. Default is `None`.
  - `batch_processor_config (Optional[BatchProcessorConfig])`: Configuration for batch processing. Default is `None`.
  - `threadpoolctl_available (bool)`: Whether threadpoolctl is available. Default is `False`.
  - `onnx_available (bool)`: Whether ONNX runtime is available. Default is `False`.
  - `treelite_available (bool)`: Whether Treelite is available. Default is `False`.

- **Methods**:

  ##### `__post_init__(self) -> None`
  ```python
  def __post_init__(self)
  ```
  - **Description**:  
    Initializes default collections and validates configuration. Sets up nested configurations if needed.
  
  - **Returns**:  
    - `None`

  ##### `to_dict(self) -> Dict`
  ```python
  def to_dict(self) -> Dict
  ```
  - **Description**:  
    Converts the configuration to a dictionary for serial