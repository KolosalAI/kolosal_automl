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
    
    # Other methods remain the same...

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