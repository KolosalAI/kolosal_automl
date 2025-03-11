import os
import platform
import psutil
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import multiprocessing
import socket
import uuid

# Import the configuration classes from your module
from dataclasses import asdict
from enum import Enum

from modules.configs import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("config_optimizer")

class DeviceOptimizer:
    """
    Automatically configures ML pipeline settings based on the device capabilities.
    Optimizes configurations for:
    - Quantization
    - Batch processing
    - Preprocessing
    - Inference engine
    - Training engine
    """
    
    def __init__(self, 
                 config_path: str = "./configs",
                 checkpoint_path: str = "./checkpoints",
                 model_registry_path: str = "./model_registry"):
        """
        Initialize the device optimizer.
        
        Args:
            config_path: Path to save configuration files
            checkpoint_path: Path for model checkpoints
            model_registry_path: Path for model registry
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_registry_path = Path(model_registry_path)
        
        # Create directories if they don't exist
        for path in [self.config_path, self.checkpoint_path, self.model_registry_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # System information
        self.cpu_count = multiprocessing.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.system = platform.system()
        self.machine = platform.machine()
        self.processor = platform.processor()
        self.hostname = socket.gethostname()
        
        # Check for Intel CPU
        self.is_intel_cpu = "Intel" in self.processor
        
        # Check for AVX/AVX2 support on x86 processors
        self.has_avx = False
        self.has_avx2 = False
        
        if self.system == "Linux" and (self.machine == "x86_64" or self.machine == "AMD64"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    self.has_avx = "avx" in cpuinfo.lower()
                    self.has_avx2 = "avx2" in cpuinfo.lower()
            except Exception as e:
                logger.warning(f"Failed to check AVX support: {e}")
        
        # Log system information
        logger.info(f"System: {self.system} {self.machine}")
        logger.info(f"Processor: {self.processor}")
        logger.info(f"CPU Count: {self.cpu_count}")
        logger.info(f"Total Memory: {self.total_memory_gb:.2f} GB")
        logger.info(f"Intel CPU: {self.is_intel_cpu}")
        logger.info(f"AVX Support: {self.has_avx}")
        logger.info(f"AVX2 Support: {self.has_avx2}")
    
    # Add this helper function to handle Enum serialization
    def _serialize_config_dict(self, config_dict):
        """
        Convert Enum values to strings for JSON serialization.
        
        Args:
            config_dict: Dictionary that may contain Enum values
            
        Returns:
            Dictionary with Enum values converted to strings
        """
        result = {}
        for key, value in config_dict.items():
            if isinstance(value, Enum):
                result[key] = value.value if hasattr(value, 'value') else value.name
            elif isinstance(value, dict):
                result[key] = self._serialize_config_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self._serialize_config_dict(item) if isinstance(item, dict) 
                    else (item.value if isinstance(item, Enum) and hasattr(item, 'value') 
                        else (item.name if isinstance(item, Enum) else item))
                    for item in value
                ]
            else:
                result[key] = value
        return 

    def get_optimal_quantization_config(self) -> QuantizationConfig:
        """
        Create an optimized quantization configuration based on device capabilities.
        
        Returns:
            Optimized QuantizationConfig
        """
        # Default to INT8 quantization for most systems
        quant_type = QuantizationType.INT8.value
        
        # For systems with limited memory, use more aggressive quantization
        if self.total_memory_gb < 4:
            quant_mode = QuantizationMode.DYNAMIC_PER_BATCH.value
            cache_size = 256  # Smaller cache for low memory systems
        elif self.total_memory_gb < 8:
            quant_mode = QuantizationMode.DYNAMIC_PER_BATCH.value
            cache_size = 512
        else:
            quant_mode = QuantizationMode.DYNAMIC_PER_CHANNEL.value
            cache_size = 1024
        
        # Determine buffer size based on available memory
        buffer_size = max(int(self.total_memory_gb * 32), 64)  # 32MB per GB of RAM, minimum 64
        
        # Create and return the config
        return QuantizationConfig(
            quantization_type=quant_type,
            quantization_mode=quant_mode,
            enable_cache=True,
            cache_size=cache_size,
            buffer_size=buffer_size,
            use_percentile=True,
            min_percentile=0.1,
            max_percentile=99.9,
            error_on_nan=True,
            error_on_inf=True,
            outlier_threshold=3.0,  # 3 standard deviations
            num_bits=8,
            optimize_memory=self.total_memory_gb < 16  # Optimize memory on systems with less than 16GB
        )
    
    def get_optimal_batch_processor_config(self) -> BatchProcessorConfig:
        """
        Create an optimized batch processor configuration based on device capabilities.
        
        Returns:
            Optimized BatchProcessorConfig
        """
        # Calculate optimal batch sizes based on system resources
        max_batch_size = min(int(self.cpu_count * 16), 256)
        initial_batch_size = max(min(int(max_batch_size / 2), 64), 8)
        min_batch_size = max(int(initial_batch_size / 4), 1)
        
        # Calculate max workers based on CPU count
        max_workers = max(int(self.cpu_count * 0.75), 1)
        
        # Calculate memory thresholds
        max_batch_memory_mb = self.total_memory_gb * 128  # 128MB per GB of RAM
        
        # Determine queue sizes based on available memory
        max_queue_size = min(int(self.total_memory_gb * 100), 5000)
        queue_warning = max_queue_size // 5
        queue_critical = max_queue_size // 2
        
        # Create and return the config
        return BatchProcessorConfig(
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            initial_batch_size=initial_batch_size,
            max_queue_size=max_queue_size,
            enable_priority_queue=True,
            batch_timeout=0.1,
            item_timeout=10.0,
            min_batch_interval=0.01,
            processing_strategy=BatchProcessingStrategy.ADAPTIVE,
            enable_adaptive_batching=True,
            max_retries=2,
            retry_delay=0.1,
            reduce_batch_on_failure=True,
            max_batch_memory_mb=max_batch_memory_mb,
            enable_memory_optimization=True,
            gc_batch_threshold=max_batch_size // 2,
            enable_monitoring=True,
            monitoring_window=100,
            max_workers=max_workers,
            enable_health_monitoring=True,
            health_check_interval=5.0,
            memory_warning_threshold=70.0,
            memory_critical_threshold=85.0,
            queue_warning_threshold=queue_warning,
            queue_critical_threshold=queue_critical,
            debug_mode=False
        )
    
    def get_optimal_preprocessor_config(self) -> PreprocessorConfig:
        """
        Create an optimized preprocessor configuration based on device capabilities.
        
        Returns:
            Optimized PreprocessorConfig
        """
        # Determine optimal settings based on system resources
        parallel_processing = self.cpu_count > 2
        n_jobs = max(self.cpu_count - 1, 1) if parallel_processing else 1
        
        # Determine cache size based on available memory
        cache_size = min(int(self.total_memory_gb * 32), 512)
        
        # Choose appropriate dtype based on memory constraints
        dtype = np.float32 if self.total_memory_gb < 8 else np.float64
        
        # Create and return the config
        return PreprocessorConfig(
            normalization=NormalizationType.STANDARD,
            robust_percentiles=(25.0, 75.0),
            handle_nan=True,
            handle_inf=True,
            nan_strategy="mean",
            inf_strategy="mean",
            detect_outliers=True,
            outlier_method="iqr",
            outlier_params={
                "threshold": 1.5,
                "clip": True,
                "n_estimators": 100,
                "contamination": "auto"
            },
            clip_values=False,
            clip_range=(-np.inf, np.inf),
            enable_input_validation=True,
            input_size_limit=None,
            parallel_processing=parallel_processing,
            n_jobs=n_jobs,
            chunk_size=10000 if self.total_memory_gb < 8 else None,
            cache_enabled=True,
            cache_size=cache_size,
            dtype=dtype,
            epsilon=1e-10,
            debug_mode=False,
            custom_normalization_fn=None,
            custom_transform_fn=None,
            version="1.0.0"
        )
    
    def get_optimal_inference_engine_config(self) -> InferenceEngineConfig:
        """
        Create an optimized inference engine configuration based on device capabilities.
        
        Returns:
            Optimized InferenceEngineConfig
        """
        # Determine optimal thread count (leave some cores for system)
        num_threads = max(int(self.cpu_count * 0.8), 1)
        
        # Determine batch sizes based on system resources
        max_batch_size = min(int(self.cpu_count * 16), 256)
        initial_batch_size = max(min(int(max_batch_size / 2), 64), 8)
        min_batch_size = max(int(initial_batch_size / 4), 1)
        
        # Determine memory thresholds
        memory_high_watermark_mb = min(self.total_memory_gb * 256, 4096)
        memory_limit_gb = self.total_memory_gb * 0.8 if self.total_memory_gb > 4 else None
        
        # Get quantization config
        quantization_config = self.get_optimal_quantization_config()
        
        # Create and return the config
        return InferenceEngineConfig(
            model_version="1.0",
            debug_mode=False,
            num_threads=num_threads,
            set_cpu_affinity=self.cpu_count > 4,
            enable_intel_optimization=self.is_intel_cpu and (self.has_avx or self.has_avx2),
            enable_quantization=True,
            enable_model_quantization=True,
            enable_input_quantization=True,
            quantization_dtype="int8",
            quantization_config=quantization_config,
            enable_request_deduplication=True,
            max_cache_entries=min(int(self.total_memory_gb * 100), 2000),
            cache_ttl_seconds=300,
            monitoring_window=100,
            enable_monitoring=True,
            monitoring_interval=10.0,
            throttle_on_high_cpu=True,
            cpu_threshold_percent=90.0,
            memory_high_watermark_mb=memory_high_watermark_mb,
            memory_limit_gb=memory_limit_gb,
            enable_batching=True,
            batch_processing_strategy="adaptive",
            batch_timeout=0.1,
            max_concurrent_requests=max(num_threads, 2),
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            enable_adaptive_batching=True,
            enable_memory_optimization=True,
            enable_feature_scaling=True,
            enable_warmup=True,
            enable_quantization_aware_inference=self.is_intel_cpu and self.has_avx2,
            enable_throttling=False
        )
    
    def get_optimal_training_engine_config(self) -> MLTrainingEngineConfig:
        """
        Create an optimized training engine configuration based on device capabilities.
        
        Returns:
            Optimized MLTrainingEngineConfig
        """
        # Determine optimal job count (leave some cores for system)
        n_jobs = max(int(self.cpu_count * 0.8), 1)
        
        # Choose optimization strategy based on system resources
        if self.total_memory_gb > 16 and self.cpu_count > 8:
            optimization_strategy = OptimizationStrategy.ASHT
            optimization_iterations = 50
        elif self.total_memory_gb > 8 and self.cpu_count > 4:
            optimization_strategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
            optimization_iterations = 30
        else:
            optimization_strategy = OptimizationStrategy.RANDOM_SEARCH
            optimization_iterations = 20
        
        # Get other configs
        preprocessing_config = self.get_optimal_preprocessor_config()
        batch_processing_config = self.get_optimal_batch_processor_config()
        inference_config = self.get_optimal_inference_engine_config()
        quantization_config = self.get_optimal_quantization_config()
        
        # Create and return the config
        return MLTrainingEngineConfig(
            task_type=TaskType.CLASSIFICATION,
            random_state=42,
            n_jobs=n_jobs,
            verbose=1,
            cv_folds=5,
            test_size=0.2,
            stratify=True,
            optimization_strategy=optimization_strategy,
            optimization_iterations=optimization_iterations,
            early_stopping=True,
            feature_selection=True,
            feature_selection_method="mutual_info",
            feature_selection_k=None,
            feature_importance_threshold=0.01,
            preprocessing_config=preprocessing_config,
            batch_processing_config=batch_processing_config,
            inference_config=inference_config,
            quantization_config=quantization_config,
            model_path=str(self.model_registry_path),
            experiment_tracking=True,
            use_intel_optimization=self.is_intel_cpu and (self.has_avx or self.has_avx2),
            memory_optimization=self.total_memory_gb < 16,
            enable_distributed=self.cpu_count > 8,
            log_level="INFO"
        )
    


    # Update the save_configs method to use the serialization helper
    def save_configs(self, config_id: Optional[str] = None) -> Dict[str, str]:
        """
        Generate and save all optimized configurations.
        
        Args:
            config_id: Optional identifier for the configuration set
        
        Returns:
            Dictionary with paths to saved configuration files
        """
        if config_id is None:
            config_id = str(uuid.uuid4())[:8]
        
        # Create configs directory if it doesn't exist
        configs_dir = self.config_path / config_id
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all configs
        quantization_config = self.get_optimal_quantization_config()
        batch_config = self.get_optimal_batch_processor_config()
        preprocessor_config = self.get_optimal_preprocessor_config()
        inference_config = self.get_optimal_inference_engine_config()
        training_config = self.get_optimal_training_engine_config()
        
        # Save system info
        system_info = {
            "system": self.system,
            "machine": self.machine,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
            "total_memory_gb": self.total_memory_gb,
            "is_intel_cpu": self.is_intel_cpu,
            "has_avx": self.has_avx,
            "has_avx2": self.has_avx2,
            "hostname": self.hostname,
            "timestamp": str(np.datetime64('now'))
        }
        
        system_info_path = configs_dir / "system_info.json"
        with open(system_info_path, "w") as f:
            json.dump(system_info, f, indent=2)
        
        # Save quantization config - Use serialization helper
        quant_config_dict = self._serialize_config_dict(asdict(quantization_config))
        quant_config_path = configs_dir / "quantization_config.json"
        with open(quant_config_path, "w") as f:
            json.dump(quant_config_dict, f, indent=2)
        
        # Save batch processor config - Use serialization helper
        batch_config_dict = self._serialize_config_dict(asdict(batch_config))
        batch_config_path = configs_dir / "batch_processor_config.json"
        with open(batch_config_path, "w") as f:
            json.dump(batch_config_dict, f, indent=2)
        
        # Save preprocessor config - Use serialization helper
        # Instead of using to_dict(), use asdict() and then serialize
        preprocessor_config_dict = self._serialize_config_dict(asdict(preprocessor_config))
        preprocessor_config_path = configs_dir / "preprocessor_config.json"
        with open(preprocessor_config_path, "w") as f:
            json.dump(preprocessor_config_dict, f, indent=2)
        
        # Save inference engine config - Use serialization helper
        inference_config_dict = self._serialize_config_dict(inference_config.to_dict())
        inference_config_path = configs_dir / "inference_engine_config.json"
        with open(inference_config_path, "w") as f:
            json.dump(inference_config_dict, f, indent=2)
        
        # Save training engine config - Use serialization helper
        # For non-dataclass objects, we need to handle differently
        if hasattr(training_config, '__dict__'):
            training_config_dict = self._serialize_config_dict(training_config.__dict__)
        else:
            training_config_dict = self._serialize_config_dict(training_config.to_dict())
        
        training_config_path = configs_dir / "training_engine_config.json"
        with open(training_config_path, "w") as f:
            json.dump(training_config_dict, f, indent=2)
        
        # Create a master config file with paths to all configs
        master_config = {
            "config_id": config_id,
            "system_info": str(system_info_path),
            "quantization_config": str(quant_config_path),
            "batch_processor_config": str(batch_config_path),
            "preprocessor_config": str(preprocessor_config_path),
            "inference_engine_config": str(inference_config_path),
            "training_engine_config": str(training_config_path),
            "checkpoint_path": str(self.checkpoint_path),
            "model_registry_path": str(self.model_registry_path)
        }
        
        master_config_path = configs_dir / "master_config.json"
        with open(master_config_path, "w") as f:
            json.dump(master_config, f, indent=2)
        
        logger.info(f"All configurations saved to {configs_dir}")
        return master_config


# Fix the load_config function to properly check file existence
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration as a dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config

def safe_dict_serializer(obj, ignore_types=None, max_depth=10, current_depth=0):
    """
    Convert an object to a serializable dictionary, ignoring non-serializable types.
    
    Args:
        obj: The object to convert
        ignore_types: List of types to ignore (will be replaced with their string representation)
        max_depth: Maximum recursion depth to prevent infinite recursion
        current_depth: Current recursion depth (used internally)
        
    Returns:
        A JSON-serializable representation of the object
    """
    if ignore_types is None:
        ignore_types = [type, types.FunctionType, types.MethodType, types.ModuleType, 
                        types.BuiltinFunctionType, types.BuiltinMethodType]
    
    # Prevent infinite recursion
    if current_depth > max_depth:
        return str(obj)
    
    # Handle None
    if obj is None:
        return None
    
    # Handle basic types that are already serializable
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Handle Enum objects
    if isinstance(obj, Enum):
        return obj.value if hasattr(obj, 'value') else obj.name
    
    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.dtype):
        return str(obj)
    
    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)
    
    # Handle datetime objects
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    
    # Handle types to ignore
    if any(isinstance(obj, t) for t in ignore_types):
        return str(obj)
    
    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Skip keys that start with underscore (private attributes)
            if isinstance(key, str) and key.startswith('_'):
                continue
            
            # Convert key to string if it's not a string
            if not isinstance(key, str):
                key = str(key)
                
            # Recursively process the value
            result[key] = safe_dict_serializer(
                value, ignore_types, max_depth, current_depth + 1
            )
        return result
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [safe_dict_serializer(item, ignore_types, max_depth, current_depth + 1) 
                for item in obj]
    
    # Handle sets
    if isinstance(obj, set):
        return [safe_dict_serializer(item, ignore_types, max_depth, current_depth + 1) 
                for item in obj]
    
    # Try to convert to dictionary if object has __dict__
    if hasattr(obj, '__dict__'):
        return safe_dict_serializer(obj.__dict__, ignore_types, max_depth, current_depth + 1)
    
    # Try to convert dataclasses
    if dataclasses.is_dataclass(obj):
        return safe_dict_serializer(asdict(obj), ignore_types, max_depth, current_depth + 1)
    
    # Try to convert objects with __slots__
    if hasattr(obj, '__slots__'):
        result = {}
        for slot in obj.__slots__:
            if hasattr(obj, slot):
                result[slot] = safe_dict_serializer(
                    getattr(obj, slot), ignore_types, max_depth, current_depth + 1
                )
        return result
    
    # Last resort: convert to string
    return str(obj)

def save_serializable_json(obj, file_path, indent=2):
    """
    Save an object to a JSON file, handling non-serializable types.
    
    Args:
        obj: The object to save
        file_path: Path to the output JSON file
        indent: Indentation level for the JSON file
    """
    serializable_obj = safe_dict_serializer(obj)
    
    with open(file_path, 'w') as f:
        json.dump(serializable_obj, f, indent=indent)



def create_optimized_configs(
    config_path: str = "./configs",
    checkpoint_path: str = "./checkpoints",
    model_registry_path: str = "./model_registry",
    config_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Create optimized configurations based on the current device.
    
    Args:
        config_path: Path to save configuration files
        checkpoint_path: Path for model checkpoints
        model_registry_path: Path for model registry
        config_id: Optional identifier for the configuration set
    
    Returns:
        Dictionary with paths to saved configuration files
    """
    optimizer = DeviceOptimizer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model_registry_path=model_registry_path
    )
    
    return optimizer.save_configs(config_id)
