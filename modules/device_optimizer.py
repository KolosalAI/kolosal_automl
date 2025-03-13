import os
import platform
import psutil
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, TypeVar
import multiprocessing
import socket
import uuid
import datetime
import types

# Import the configuration classes from your module
from dataclasses import asdict, is_dataclass
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
                 model_registry_path: str = "./model_registry",
                 optimization_mode: OptimizationMode = OptimizationMode.BALANCED):
        """
        Initialize the device optimizer.
        
        Args:
            config_path: Path to save configuration files
            checkpoint_path: Path for model checkpoints
            model_registry_path: Path for model registry
            optimization_mode: Mode for optimization strategy
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_registry_path = Path(model_registry_path)
        self.optimization_mode = optimization_mode
        
        logger.info(f"Using optimization mode: {self.optimization_mode}")
        
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
    
    # Fixed _serialize_config_dict method to return a result
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
        return result  # Fixed: Added return statement here

    def _apply_optimization_mode_factors(self) -> Dict[str, float]:
        """
        Get scaling factors for different resource parameters based on the optimization mode.
        
        Returns:
            Dictionary of scaling factors for CPU, memory, and other resources
        """
        # Define resource utilization factors for each mode
        # Format: {mode: {resource: factor}}
        mode_factors = {
            OptimizationMode.BALANCED: {
                "cpu": 0.75,       # Use 75% of CPU cores
                "memory": 0.7,     # Use 70% of memory
                "batch_size": 0.8,  # 80% of max batch size
                "cache": 0.7,      # 70% of max cache size
                "workers": 0.75    # 75% of max workers
            },
            OptimizationMode.CONSERVATIVE: {
                "cpu": 0.5,        # Use 50% of CPU cores
                "memory": 0.5,     # Use 50% of memory
                "batch_size": 0.5,  # 50% of max batch size
                "cache": 0.5,      # 50% of max cache size
                "workers": 0.5     # 50% of max workers
            },
            OptimizationMode.PERFORMANCE: {
                "cpu": 0.9,        # Use 90% of CPU cores
                "memory": 0.8,     # Use 80% of memory
                "batch_size": 1.0,  # 100% of max batch size
                "cache": 0.9,      # 90% of max cache size
                "workers": 0.9     # 90% of max workers
            },
            OptimizationMode.FULL_UTILIZATION: {
                "cpu": 1.0,        # Use 100% of CPU cores
                "memory": 0.95,    # Use 95% of memory
                "batch_size": 1.2,  # 120% of max batch size
                "cache": 1.0,      # 100% of max cache size
                "workers": 1.0     # 100% of max workers
            },
            OptimizationMode.MEMORY_SAVING: {
                "cpu": 0.7,        # Use 70% of CPU cores
                "memory": 0.4,     # Use 40% of memory
                "batch_size": 0.6,  # 60% of max batch size
                "cache": 0.4,      # 40% of max cache size
                "workers": 0.7     # 70% of max workers
            }
        }
        
        return mode_factors.get(self.optimization_mode, mode_factors[OptimizationMode.BALANCED])

    def get_optimal_quantization_config(self) -> QuantizationConfig:
        """
        Create an optimized quantization configuration based on device capabilities.
        
        Returns:
            Optimized QuantizationConfig
        """
        # Get scaling factors based on optimization mode
        factors = self._apply_optimization_mode_factors()
        
        # Default to INT8 quantization for most systems
        quant_type = QuantizationType.INT8.value
        
        # Higher memory utilization in full utilization mode
        memory_factor = factors["memory"]
        
        # For systems with limited memory, use more aggressive quantization
        if self.total_memory_gb < 4 or self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            quant_mode = QuantizationMode.DYNAMIC_PER_BATCH.value
            cache_size = max(int(256 * memory_factor), 128)  # Smaller cache for low memory systems
        elif self.total_memory_gb < 8:
            quant_mode = QuantizationMode.DYNAMIC_PER_BATCH.value
            cache_size = max(int(512 * memory_factor), 256)
        else:
            quant_mode = QuantizationMode.DYNAMIC_PER_CHANNEL.value
            cache_size = max(int(1024 * memory_factor), 512)
        
        # Determine buffer size based on available memory
        buffer_size = max(int(self.total_memory_gb * 32 * memory_factor), 64)  # 32MB per GB of RAM, minimum 64
        
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
            optimize_memory=self.total_memory_gb < 16 or self.optimization_mode in [OptimizationMode.MEMORY_SAVING, OptimizationMode.CONSERVATIVE]
        )
    
    def get_optimal_batch_processor_config(self) -> BatchProcessorConfig:
        """
        Create an optimized batch processor configuration based on device capabilities.
        
        Returns:
            Optimized BatchProcessorConfig
        """
        # Get scaling factors based on optimization mode
        factors = self._apply_optimization_mode_factors()
        
        # Calculate optimal batch sizes based on system resources and optimization mode
        batch_factor = factors["batch_size"]
        max_batch_size = min(int(self.cpu_count * 16 * batch_factor), 512 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 256)
        initial_batch_size = max(min(int(max_batch_size / 2), 128 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 64), 8)
        min_batch_size = max(int(initial_batch_size / 4), 1)
        
        # Calculate max workers based on CPU count and optimization mode
        worker_factor = factors["workers"]
        max_workers = max(int(self.cpu_count * worker_factor), 1)
        
        # Calculate memory thresholds based on optimization mode
        memory_factor = factors["memory"]
        max_batch_memory_mb = self.total_memory_gb * 128 * memory_factor  # 128MB per GB of RAM
        
        # Determine queue sizes based on available memory and optimization mode
        cache_factor = factors["cache"]
        max_queue_size = min(int(self.total_memory_gb * 100 * cache_factor), 
                            10000 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 5000)
        queue_warning = max_queue_size // 5
        queue_critical = max_queue_size // 2
        
        # Adjust adaptive behavior based on optimization mode
        adaptive_batching = True
        if self.optimization_mode == OptimizationMode.FULL_UTILIZATION:
            # In full utilization mode, we might want to be less aggressive with adaptive behaviors
            adaptive_batching = False
        
        # Create and return the config
        return BatchProcessorConfig(
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            initial_batch_size=initial_batch_size,
            max_queue_size=max_queue_size,
            enable_priority_queue=True,
            batch_timeout=0.05 if self.optimization_mode == OptimizationMode.PERFORMANCE else 0.1,
            item_timeout=5.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else 10.0,
            min_batch_interval=0.005 if self.optimization_mode == OptimizationMode.PERFORMANCE else 0.01,
            processing_strategy=BatchProcessingStrategy.ADAPTIVE,
            enable_adaptive_batching=adaptive_batching,
            max_retries=3 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 2,
            retry_delay=0.05 if self.optimization_mode == OptimizationMode.PERFORMANCE else 0.1,
            reduce_batch_on_failure=True,
            max_batch_memory_mb=max_batch_memory_mb,
            enable_memory_optimization=self.optimization_mode != OptimizationMode.FULL_UTILIZATION,
            gc_batch_threshold=max_batch_size // 2,
            enable_monitoring=True,
            monitoring_window=50 if self.optimization_mode == OptimizationMode.PERFORMANCE else 100,
            max_workers=max_workers,
            enable_health_monitoring=True,
            health_check_interval=2.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else 5.0,
            memory_warning_threshold=85.0 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 70.0,
            memory_critical_threshold=95.0 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 85.0,
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
        # Get scaling factors based on optimization mode
        factors = self._apply_optimization_mode_factors()
        
        # Determine optimal settings based on system resources and optimization mode
        cpu_factor = factors["cpu"]
        parallel_processing = (self.cpu_count > 2) and (self.optimization_mode != OptimizationMode.CONSERVATIVE)
        n_jobs = max(int(self.cpu_count * cpu_factor), 1) if parallel_processing else 1
        
        # Determine cache size based on available memory and optimization mode
        memory_factor = factors["memory"]
        cache_size = min(int(self.total_memory_gb * 32 * memory_factor), 
                        1024 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 512)
        
        # Choose appropriate dtype based on memory constraints and optimization mode
        dtype = np.float16 if (self.total_memory_gb < 8 or self.optimization_mode == OptimizationMode.MEMORY_SAVING) else \
               (np.float32 if self.optimization_mode != OptimizationMode.FULL_UTILIZATION else np.float64)
        
        # Adjust chunk size based on optimization mode
        chunk_size = None
        if self.total_memory_gb < 8 or self.optimization_mode in [OptimizationMode.MEMORY_SAVING, OptimizationMode.CONSERVATIVE]:
            chunk_size = 5000 if self.optimization_mode == OptimizationMode.MEMORY_SAVING else 10000
        
        # Create and return the config
        return PreprocessorConfig(
            normalization=NormalizationType.STANDARD,
            robust_percentiles=(25.0, 75.0),
            handle_nan=True,
            handle_inf=True,
            nan_strategy="mean",
            inf_strategy="mean",
            detect_outliers=self.optimization_mode != OptimizationMode.PERFORMANCE,  # Disable for performance mode
            outlier_method="iqr" if self.optimization_mode != OptimizationMode.FULL_UTILIZATION else "isolation_forest",
            outlier_params={
                "threshold": 1.5,
                "clip": True,
                "n_estimators": 100 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 50,
                "contamination": "auto"
            },
            clip_values=self.optimization_mode in [OptimizationMode.MEMORY_SAVING, OptimizationMode.CONSERVATIVE],
            clip_range=(-np.inf, np.inf),
            enable_input_validation=self.optimization_mode != OptimizationMode.PERFORMANCE,
            input_size_limit=None,
            parallel_processing=parallel_processing,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
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
        # Get scaling factors based on optimization mode
        factors = self._apply_optimization_mode_factors()
        
        # Determine optimal thread count based on optimization mode
        cpu_factor = factors["cpu"]
        num_threads = max(int(self.cpu_count * cpu_factor), 1)
        
        # Determine batch sizes based on system resources and optimization mode
        batch_factor = factors["batch_size"]
        max_batch_size = min(int(self.cpu_count * 16 * batch_factor), 
                            512 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 256)
        initial_batch_size = max(min(int(max_batch_size / 2), 
                                    128 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 64), 8)
        min_batch_size = max(int(initial_batch_size / 4), 1)
        
        # Determine memory thresholds based on optimization mode
        memory_factor = factors["memory"]
        memory_high_watermark_mb = min(self.total_memory_gb * 256 * memory_factor, 
                                    8192 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 4096)
        memory_limit_gb = self.total_memory_gb * memory_factor if self.total_memory_gb > 4 else None
        
        # Get cache factor for cache settings
        cache_factor = factors["cache"]
        max_cache_entries = min(int(self.total_memory_gb * 100 * cache_factor), 
                            5000 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 2000)
        
        # Determine CPU thresholds based on optimization mode
        cpu_threshold = 95.0 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 90.0
        
        # Get quantization config
        quantization_config = self.get_optimal_quantization_config()
        
        # Enable optimizations based on optimization mode
        enable_intel_optimizations = self.is_intel_cpu and (self.has_avx or self.has_avx2)
        enable_quantization = self.optimization_mode != OptimizationMode.PERFORMANCE or self.total_memory_gb < 16
        enable_quantization_aware = enable_intel_optimizations and self.has_avx2 and self.optimization_mode != OptimizationMode.CONSERVATIVE
        
        # Create and return the config
        return InferenceEngineConfig(
            model_version="1.0",
            debug_mode=False,
            num_threads=num_threads,
            set_cpu_affinity=self.cpu_count > 4 and self.optimization_mode not in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING],
            enable_intel_optimization=enable_intel_optimizations,
            enable_quantization=enable_quantization,
            enable_model_quantization=enable_quantization,
            enable_input_quantization=enable_quantization,
            quantization_dtype="int8",
            quantization_config=quantization_config,
            enable_request_deduplication=self.optimization_mode != OptimizationMode.PERFORMANCE,
            max_cache_entries=max_cache_entries,
            cache_ttl_seconds=150 if self.optimization_mode == OptimizationMode.PERFORMANCE else 300,
            monitoring_window=50 if self.optimization_mode == OptimizationMode.PERFORMANCE else 100,
            enable_monitoring=True,
            monitoring_interval=5.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else 10.0,
            throttle_on_high_cpu=self.optimization_mode != OptimizationMode.FULL_UTILIZATION,
            cpu_threshold_percent=cpu_threshold,
            memory_high_watermark_mb=memory_high_watermark_mb,
            memory_limit_gb=memory_limit_gb,
            enable_batching=True,
            batch_processing_strategy="adaptive" if self.optimization_mode != OptimizationMode.FULL_UTILIZATION else "greedy",
            batch_timeout=0.05 if self.optimization_mode == OptimizationMode.PERFORMANCE else 0.1,
            max_concurrent_requests=max(num_threads * 2 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else num_threads, 2),
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            enable_adaptive_batching=self.optimization_mode != OptimizationMode.FULL_UTILIZATION,
            enable_memory_optimization=self.optimization_mode != OptimizationMode.FULL_UTILIZATION,
            enable_feature_scaling=True,
            enable_warmup=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            enable_quantization_aware_inference=enable_quantization_aware,
            enable_throttling=self.optimization_mode != OptimizationMode.FULL_UTILIZATION
        )
    
    def get_optimal_training_engine_config(self) -> MLTrainingEngineConfig:
        """
        Create an optimized training engine configuration based on device capabilities.
        
        Returns:
            Optimized MLTrainingEngineConfig
        """
        # Get scaling factors based on optimization mode
        factors = self._apply_optimization_mode_factors()
        
        # Determine optimal job count based on optimization mode
        cpu_factor = factors["cpu"]
        n_jobs = max(int(self.cpu_count * cpu_factor), 1)
        
        # Choose optimization strategy based on system resources and optimization mode
        if self.total_memory_gb > 16 and self.cpu_count > 8 and self.optimization_mode in [OptimizationMode.PERFORMANCE, OptimizationMode.FULL_UTILIZATION]:
            optimization_strategy = OptimizationStrategy.ASHT
            optimization_iterations = 100 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 50
        elif self.total_memory_gb > 8 and self.cpu_count > 4 and self.optimization_mode not in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING]:
            optimization_strategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
            optimization_iterations = 50 if self.optimization_mode == OptimizationMode.PERFORMANCE else 30
        else:
            optimization_strategy = OptimizationStrategy.RANDOM_SEARCH
            optimization_iterations = 30 if self.optimization_mode == OptimizationMode.PERFORMANCE else 20
        
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
            verbose=2 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 1,
            cv_folds=10 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 5,
            test_size=0.2,
            stratify=True,
            optimization_strategy=optimization_strategy,
            optimization_iterations=optimization_iterations,
            early_stopping=self.optimization_mode != OptimizationMode.FULL_UTILIZATION,
            feature_selection=self.optimization_mode != OptimizationMode.PERFORMANCE,
            feature_selection_method="mutual_info",
            feature_selection_k=None,
            feature_importance_threshold=0.005 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 0.01,
            preprocessing_config=preprocessing_config,
            batch_processing_config=batch_processing_config,
            inference_config=inference_config,
            quantization_config=quantization_config,
            model_path=str(self.model_registry_path),
            experiment_tracking=True,
            use_intel_optimization=self.is_intel_cpu and (self.has_avx or self.has_avx2),
            memory_optimization=self.optimization_mode != OptimizationMode.FULL_UTILIZATION,
            enable_distributed=self.cpu_count > 8 and self.optimization_mode in [OptimizationMode.PERFORMANCE, OptimizationMode.FULL_UTILIZATION],
            log_level="DEBUG" if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else "INFO"
        )

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
            "optimization_mode": self.optimization_mode.value,
            "timestamp": str(datetime.datetime.now())
        }
        
        system_info_path = configs_dir / "system_info.json"
        with open(system_info_path, "w") as f:
            json.dump(system_info, f, indent=2)
        
        # Create a safe method for serializing configurations
        def safe_serialize_config(config):
            """Convert dataclass or object to JSON serializable dictionary."""
            if hasattr(config, '__dict__'):
                return safe_dict_serializer(config.__dict__)
            elif is_dataclass(config):
                return safe_dict_serializer(asdict(config))
            return config
        
        # Save configurations using safe serialization
        configs_to_save = [
            ("quantization_config", quantization_config, quant_config_path := configs_dir / "quantization_config.json"),
            ("batch_config", batch_config, batch_config_path := configs_dir / "batch_processor_config.json"),
            ("preprocessor_config", preprocessor_config, preprocessor_config_path := configs_dir / "preprocessor_config.json"),
            ("inference_config", inference_config, inference_config_path := configs_dir / "inference_engine_config.json"),
            ("training_config", training_config, training_config_path := configs_dir / "training_engine_config.json")
        ]
        
        for name, config, path in configs_to_save:
            try:
                serialized_config = safe_serialize_config(config)
                with open(path, "w") as f:
                    json.dump(serialized_config, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")
        
        # Create a master config file with paths to all configs
        master_config = {
            "config_id": config_id,
            "optimization_mode": self.optimization_mode.value,
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

def create_optimized_configs(
    config_path: str = "./configs", 
    checkpoint_path: str = "./checkpoints", 
    model_registry_path: str = "./model_registry",
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED,
    **kwargs  # Added to handle additional arguments
) -> Dict[str, str]:
    """
    Create and save optimized configurations for the ML pipeline.
    
    Args:
        config_path: Path to save configuration files
        checkpoint_path: Path for model checkpoints
        model_registry_path: Path for model registry
        optimization_mode: Mode for optimization strategy
        **kwargs: Additional keyword arguments (for flexibility)
    
    Returns:
        Dictionary with paths to saved configuration files
    """
    # Extract config_id if present in kwargs
    config_id = kwargs.get('config_id')
    
    optimizer = DeviceOptimizer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model_registry_path=model_registry_path,
        optimization_mode=optimization_mode
    )
    
    # If config_id is provided, pass it to save_configs
    return optimizer.save_configs(config_id)

def create_configs_for_all_modes(
    config_path: str = "./configs", 
    checkpoint_path: str = "./checkpoints", 
    model_registry_path: str = "./model_registry",
    **kwargs  # Added to handle additional arguments
) -> Dict[OptimizationMode, Dict[str, str]]:
    """
    Create and save configurations for all optimization modes.
    
    Args:
        config_path: Path to save configuration files
        checkpoint_path: Path for model checkpoints
        model_registry_path: Path for model registry
        **kwargs: Additional keyword arguments (for flexibility)
    
    Returns:
        Dictionary of configurations for each optimization mode
    """
    # Get all optimization modes
    all_modes = list(OptimizationMode)
    
    # Create configurations for each mode
    configs = {}
    for mode in all_modes:
        # Pass along any additional keyword arguments
        configs[mode] = create_optimized_configs(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model_registry_path=model_registry_path,
            optimization_mode=mode,
            **kwargs
        )
    
    return configs

# Generic type for configuration classes
ConfigT = TypeVar('ConfigT')

def load_config(
    config_path: Union[str, Path], 
    config_class: Optional[Type[ConfigT]] = None
) -> Union[Dict[str, Any], ConfigT]:
    """
    Load a configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        config_class: Optional configuration class to convert the loaded dict to
    
    Returns:
        Loaded configuration as a dictionary or an instance of the specified config class
    """
    # Ensure path is a Path object
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Read the JSON file
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    # If no specific config class is provided, return the dictionary
    if config_class is None:
        return config_dict
    
    # Convert dictionary to config class instance
    try:
        # Special handling for Enum values
        def convert_enum_values(config_class, data):
            # Get the class's field names
            field_names = getattr(config_class, '__annotations__', {}).keys()
            
            # Create a new dictionary to store converted values
            converted_data = {}
            for key, value in data.items():
                if key in field_names:
                    field_type = config_class.__annotations__.get(key)
                    
                    # Convert Enum values
                    if isinstance(field_type, type) and issubclass(field_type, Enum):
                        converted_data[key] = field_type(value)
                    else:
                        converted_data[key] = value
                else:
                    converted_data[key] = value
            
            return converted_data
        
        # Convert enum values
        converted_data = convert_enum_values(config_class, config_dict)
        
        # Create an instance of the config class
        return config_class(**converted_data)
    except Exception as e:
        raise ValueError(f"Failed to convert config to {config_class.__name__}: {e}")

def safe_dict_serializer(
    data: Any, 
    max_depth: int = 10, 
    current_depth: int = 0
) -> Any:
    """
    Safely serialize complex objects into a JSON-compatible dictionary.
    
    Args:
        data: The data to be serialized
        max_depth: Maximum recursion depth to prevent infinite recursion
        current_depth: Current recursion depth
    
    Returns:
        JSON-compatible representation of the input data
    """
    # Check recursion depth
    if current_depth > max_depth:
        return str(data)
    
    # Handle None
    if data is None:
        return None
    
    # Handle primitive types
    if isinstance(data, (str, int, float, bool)):
        return data
    
    # Handle Enum
    if isinstance(data, Enum):
        return data.value
    
    # Handle datetime
    if isinstance(data, (datetime.datetime, datetime.date)):
        return data.isoformat()
    
    # Handle numpy types
    if isinstance(data, (np.integer, np.floating, np.ndarray)):
        # Convert numpy types to standard Python types
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data.item()
    
    # Handle dataclasses
    if is_dataclass(data):
        return safe_dict_serializer(asdict(data), max_depth, current_depth + 1)
    
    # Handle paths 
    if isinstance(data, (Path, os.PathLike)):
        # Normalize the path to use forward slashes for consistency
        return str(data).replace('\\', '/')
    
    # Handle dictionaries
    if isinstance(data, dict):
        return {
            str(k): safe_dict_serializer(v, max_depth, current_depth + 1) 
            for k, v in data.items()
        }
    
    # Handle lists, tuples, and other iterables
    if isinstance(data, (list, tuple, set)):
        return [
            safe_dict_serializer(item, max_depth, current_depth + 1) 
            for item in data
        ]
    
    # Handle objects with __dict__ attribute
    if hasattr(data, '__dict__'):
        return safe_dict_serializer(data.__dict__, max_depth, current_depth + 1)
    
    # Handle type objects
    if isinstance(data, type):
        return data.__name__
    
    # Fallback to string representation
    return str(data)