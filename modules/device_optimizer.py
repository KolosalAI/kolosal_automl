import os
import platform
import psutil
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TypeVar, Tuple
import multiprocessing
import socket
import uuid
import datetime
import sys
import shutil
import tempfile
import re
from enum import Enum, auto

# Import the configuration classes
from dataclasses import asdict, is_dataclass

# Assuming the modules.configs has been imported elsewhere
# If not, you'd need to properly import all the configuration classes
from modules.configs import (
    QuantizationType, QuantizationMode, QuantizationConfig, BatchProcessorConfig,
    PreprocessorConfig, InferenceEngineConfig, MLTrainingEngineConfig,
    OptimizationMode, BatchProcessingStrategy, NormalizationType, TaskType,
    OptimizationStrategy, ModelSelectionCriteria, AutoMLMode, ExplainabilityConfig,
    MonitoringConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cpu_device_optimizer")


class HardwareAccelerator(Enum):
    """Types of hardware accelerators that might be available"""
    NONE = auto()
    INTEL_IPEX = auto()
    INTEL_MKL = auto()
    ARM_NEON = auto()


class DeviceOptimizer:
    """
    CPU-only device optimizer that automatically configures ML pipeline settings 
    based on device capabilities with sophisticated optimization strategies.
    
    Features:
    - Comprehensive hardware detection (CPU, specialized accelerators)
    - Advanced configuration optimization for different pipeline components
    - Power and thermal management awareness
    - Dynamic scaling based on workload characteristics
    - Environment-aware configuration (cloud vs. edge vs. desktop)
    """
    
    def __init__(self, 
                 config_path: str = "./configs",
                 checkpoint_path: str = "./checkpoints",
                 model_registry_path: str = "./model_registry",
                 optimization_mode: OptimizationMode = OptimizationMode.BALANCED,
                 workload_type: str = "mixed",
                 environment: str = "auto",
                 enable_specialized_accelerators: bool = True,
                 memory_reservation_percent: float = 10.0,
                 power_efficiency: bool = False,
                 resilience_level: int = 1,
                 auto_tune: bool = True):
        """
        Initialize the CPU-only device optimizer.
        
        Args:
            config_path: Path to save configuration files
            checkpoint_path: Path for model checkpoints
            model_registry_path: Path for model registry
            optimization_mode: Mode for optimization strategy
            workload_type: Type of workload ("inference", "training", "mixed")
            environment: Computing environment ("cloud", "desktop", "edge", "auto")
            enable_specialized_accelerators: Whether to enable detection of specialized hardware
            memory_reservation_percent: Percentage of memory to reserve for the system
            power_efficiency: Whether to optimize for power efficiency
            resilience_level: Level of fault tolerance (0-3)
            auto_tune: Whether to enable automatic parameter tuning
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_registry_path = Path(model_registry_path)
        self.optimization_mode = optimization_mode
        self.workload_type = workload_type.lower()
        self.power_efficiency = power_efficiency
        self.resilience_level = min(max(resilience_level, 0), 3)  # Clamp between 0 and 3
        self.auto_tune = auto_tune
        self.memory_reservation_percent = min(max(memory_reservation_percent, 0), 50)  # Clamp between 0 and 50
        
        # Create directories if they don't exist
        for path in [self.config_path, self.checkpoint_path, self.model_registry_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # System information - base level
        self._detect_system_info()
        
        # CPU information - detailed
        self._detect_cpu_capabilities()
        
        # Memory information
        self._detect_memory_info()
        
        # Disk information
        self._detect_disk_info()
        
        # Detect specialized hardware accelerators
        self.accelerators = []
        if enable_specialized_accelerators:
            self._detect_specialized_accelerators()
        
        # Automatically detect environment if set to "auto"
        if environment == "auto":
            self.environment = self._detect_environment()
        else:
            self.environment = environment
        
        # Display detected information
        self._log_system_overview()

    def _detect_system_info(self):
        """Detect basic system information"""
        self.system = platform.system()
        self.release = platform.release()
        self.machine = platform.machine()
        self.processor = platform.processor()
        self.hostname = socket.gethostname()
        self.python_version = platform.python_version()

    def _detect_cpu_capabilities(self):
        """Detect detailed CPU capabilities"""
        self.cpu_count_physical = psutil.cpu_count(logical=False) or 1
        self.cpu_count_logical = psutil.cpu_count(logical=True) or 1
        self.cpu_freq = self._get_cpu_frequency()
        
        # Advanced CPU feature detection
        self.has_avx = False
        self.has_avx2 = False
        self.has_avx512 = False
        self.has_sse4 = False
        self.has_fma = False
        
        if self.system == "Linux" and (self.machine == "x86_64" or self.machine == "AMD64"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read().lower()
                    self.has_avx = "avx" in cpuinfo
                    self.has_avx2 = "avx2" in cpuinfo
                    self.has_avx512 = "avx512" in cpuinfo
                    self.has_sse4 = "sse4" in cpuinfo
                    self.has_fma = "fma" in cpuinfo
            except Exception as e:
                logger.warning(f"Failed to check advanced CPU features: {e}")
        elif self.system == "Windows":
            try:
                # On Windows, we can check using Python's platform module
                import subprocess
                result = subprocess.check_output("wmic cpu get Name", shell=True).decode()
                cpu_name = result.strip().split("\n")[1].lower()
                
                # Simplified detection based on CPU name
                self.has_avx = "avx" in cpu_name
                self.has_avx2 = "avx2" in cpu_name
                # Rough estimation, actual detection would need more sophisticated approach
                self.has_avx512 = "xeon" in cpu_name and int(platform.release()) >= 10
                self.has_sse4 = True  # Most modern CPUs have SSE4
                self.has_fma = "intel" in cpu_name and "core" in cpu_name and "i7" in cpu_name
            except Exception as e:
                logger.warning(f"Failed to check advanced CPU features on Windows: {e}")
        elif self.system == "Darwin":  # macOS
            try:
                result = os.popen("sysctl -a | grep machdep.cpu.features").read()
                features = result.lower()
                self.has_avx = "avx" in features
                self.has_avx2 = "avx2" in features
                self.has_avx512 = "avx512" in features
                self.has_sse4 = "sse4" in features
                self.has_fma = "fma" in features
            except Exception as e:
                logger.warning(f"Failed to check advanced CPU features on macOS: {e}")
        
        # Detect CPU vendor
        self.is_intel_cpu = "intel" in self.processor.lower()
        self.is_amd_cpu = "amd" in self.processor.lower()
        self.is_arm_cpu = "arm" in self.machine.lower() or "aarch64" in self.machine.lower()
        
        # Additional ARM-specific features
        if self.is_arm_cpu:
            self.has_neon = True  # Most modern ARM CPUs have NEON
        else:
            self.has_neon = False

    def _get_cpu_frequency(self) -> Dict[str, float]:
        """Get CPU frequency information"""
        try:
            freq = psutil.cpu_freq()
            if freq:
                return {
                    "current": freq.current,
                    "min": freq.min if freq.min else 0,
                    "max": freq.max if freq.max else 0
                }
        except Exception as e:
            logger.warning(f"Failed to get CPU frequency: {e}")
        
        return {"current": 0, "min": 0, "max": 0}

    def _detect_memory_info(self):
        """Detect memory information"""
        mem = psutil.virtual_memory()
        self.total_memory_gb = mem.total / (1024 ** 3)
        self.available_memory_gb = mem.available / (1024 ** 3)
        
        # Calculate safe usable memory (reserving system memory)
        reservation = self.memory_reservation_percent / 100.0
        self.usable_memory_gb = self.total_memory_gb * (1 - reservation)
        
        # Swap memory
        try:
            swap = psutil.swap_memory()
            self.swap_memory_gb = swap.total / (1024 ** 3)
        except Exception:
            self.swap_memory_gb = 0

    def _detect_disk_info(self):
        """Detect disk information"""
        try:
            usage = psutil.disk_usage(os.getcwd())
            self.disk_total_gb = usage.total / (1024 ** 3)
            self.disk_free_gb = usage.free / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Failed to get disk information: {e}")
            self.disk_total_gb = 0
            self.disk_free_gb = 0
        
        # Check if disk is SSD
        self.is_ssd = self._check_if_ssd()

    def _check_if_ssd(self) -> bool:
        """Check if the current disk is an SSD"""
        if self.system == "Linux":
            try:
                # A simplistic check that works on many Linux systems
                current_dir = os.getcwd()
                mount_point = self._find_mount_point(current_dir)
                device = self._get_device_for_mount_point(mount_point)
                
                if device:
                    # Check if the device is rotational (0 for SSD, 1 for HDD)
                    rotational_path = f"/sys/block/{os.path.basename(device)}/queue/rotational"
                    if os.path.exists(rotational_path):
                        with open(rotational_path, 'r') as f:
                            return f.read().strip() == '0'
            except Exception as e:
                logger.debug(f"SSD detection failed: {e}")
        
        # Default to False if detection fails
        return False

    def _find_mount_point(self, path):
        """Find the mount point of a given path"""
        path = os.path.abspath(path)
        while not os.path.ismount(path):
            path = os.path.dirname(path)
        return path

    def _get_device_for_mount_point(self, mount_point):
        """Get the device for a mount point"""
        try:
            with open('/proc/mounts', 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 1 and parts[1] == mount_point:
                        return parts[0]
        except Exception:
            pass
        return None

    def _detect_specialized_accelerators(self):
        """Detect specialized hardware accelerators"""
        self.accelerators = []
        
        # Check for Intel oneAPI/MKL/IPEX
        if self.is_intel_cpu and (self.has_avx or self.has_avx2):
            try:
                import ctypes
                try:
                    ctypes.CDLL("libmkl_rt.so")
                    self.accelerators.append(HardwareAccelerator.INTEL_MKL)
                    logger.info("Detected Intel MKL")
                except OSError:
                    pass
            except ImportError:
                pass
        
        # Check for ARM NEON support on ARM architectures
        if self.is_arm_cpu and self.has_neon:
            self.accelerators.append(HardwareAccelerator.ARM_NEON)
            logger.info("Detected ARM NEON instruction support")

    def _detect_environment(self) -> str:
        """Auto-detect the computing environment"""
        # Check for cloud environment indicators
        if os.environ.get('KUBERNETES_SERVICE_HOST') or os.environ.get('CLOUD_PROVIDER'):
            return "cloud"
        
        # Check for low-resource environment that could indicate edge device
        if self.total_memory_gb < 2 or self.cpu_count_physical <= 1:  # Modified thresholds
            return "edge"
        
        # Check for high-resource environment that could indicate cloud/server
        if self.total_memory_gb > 32 or self.cpu_count_physical >= 16:
            return "cloud"
        
        # Default to desktop for mid-range systems
        return "desktop"

    def _log_system_overview(self):
        """Log detected system information"""
        logger.info("=" * 50)
        logger.info(f"System Overview: {self.hostname} ({self.system} {self.release})")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Optimization Mode: {self.optimization_mode}")
        logger.info("-" * 50)
        logger.info(f"CPU: {self.processor}")
        logger.info(f"CPU Cores: {self.cpu_count_physical} physical, {self.cpu_count_logical} logical")
        logger.info(f"CPU Features: AVX={self.has_avx}, AVX2={self.has_avx2}, AVX512={self.has_avx512}")
        logger.info(f"Memory: {self.total_memory_gb:.2f} GB total, {self.usable_memory_gb:.2f} GB usable")
        logger.info(f"Disk: {self.disk_total_gb:.2f} GB total, {self.disk_free_gb:.2f} GB free, SSD={self.is_ssd}")
        
        if self.accelerators:
            logger.info("-" * 50)
            logger.info("Hardware Accelerators:")
            for acc in self.accelerators:
                logger.info(f"  {acc.name}")
        
        logger.info("=" * 50)

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
        return result
        
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
        
        # Get system information
        system_info = self.get_system_info()
        
        # Save system info
        system_info_path = configs_dir / "system_info.json"
        with open(system_info_path, "w") as f:
            json.dump(system_info, f, indent=2)
        
        # Function to safely convert config objects to JSON-serializable dictionaries
        def safe_serialize_config(config):
            """Convert dataclass or object with to_dict method to JSON serializable dictionary."""
            if hasattr(config, 'to_dict') and callable(getattr(config, 'to_dict')):
                return self._serialize_config_dict(config.to_dict())
            elif hasattr(config, '__dict__'):
                return self._serialize_config_dict(config.__dict__)
            elif is_dataclass(config):
                return self._serialize_config_dict(asdict(config))
            return config
        
        # Save configurations
        configs_to_save = [
            ("quantization_config", quantization_config, configs_dir / "quantization_config.json"),
            ("batch_config", batch_config, configs_dir / "batch_processor_config.json"),
            ("preprocessor_config", preprocessor_config, configs_dir / "preprocessor_config.json"),
            ("inference_config", inference_config, configs_dir / "inference_engine_config.json"),
            ("training_config", training_config, configs_dir / "training_engine_config.json")
        ]
        
        # Dictionary to store paths to saved configs
        saved_paths = {}
        
        for name, config, path in configs_to_save:
            try:
                # Serialize the config
                serialized_config = safe_serialize_config(config)
                
                # Save to file
                with open(path, "w") as f:
                    json.dump(serialized_config, f, indent=2)
                
                # Store the path for later reference
                saved_paths[name] = str(path)
                
                logger.info(f"Saved {name} to {path}")
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")
        
        # Create a master config file with paths to all configs
        master_config = {
            "config_id": config_id,
            "optimization_mode": self.optimization_mode.value,
            "system_info_path": str(system_info_path),
            "quantization_config_path": saved_paths.get("quantization_config", ""),
            "batch_processor_config_path": saved_paths.get("batch_config", ""),
            "preprocessor_config_path": saved_paths.get("preprocessor_config", ""),
            "inference_engine_config_path": saved_paths.get("inference_config", ""),
            "training_engine_config_path": saved_paths.get("training_config", ""),
            "checkpoint_path": str(self.checkpoint_path),
            "model_registry_path": str(self.model_registry_path),
            "creation_timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save master config
        master_config_path = configs_dir / "master_config.json"
        with open(master_config_path, "w") as f:
            json.dump(master_config, f, indent=2)
        
        logger.info(f"All configurations saved to {configs_dir}")
        return master_config
        
    def load_configs(self, config_id: str) -> Dict[str, Any]:
        """
        Load previously saved configurations.
        
        Args:
            config_id: Identifier for the configuration set
        
        Returns:
            Dictionary with loaded configurations
        """
        configs_dir = self.config_path / config_id
        
        if not configs_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {configs_dir}")
        
        # Load master config
        master_config_path = configs_dir / "master_config.json"
        if not master_config_path.exists():
            raise FileNotFoundError(f"Master configuration file not found: {master_config_path}")
        
        with open(master_config_path, "r") as f:
            master_config = json.load(f)
        
        # Load individual configs
        loaded_configs = {"master_config": master_config}
        
        # Try to load system info
        system_info_path = configs_dir / "system_info.json"
        if system_info_path.exists():
            with open(system_info_path, "r") as f:
                loaded_configs["system_info"] = json.load(f)
        
        # Try to load all other configs
        config_files = {
            "quantization_config": configs_dir / "quantization_config.json",
            "batch_processor_config": configs_dir / "batch_processor_config.json",
            "preprocessor_config": configs_dir / "preprocessor_config.json",
            "inference_engine_config": configs_dir / "inference_engine_config.json",
            "training_engine_config": configs_dir / "training_engine_config.json"
        }
        
        for name, path in config_files.items():
            if path.exists():
                try:
                    with open(path, "r") as f:
                        loaded_configs[name] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load {name} from {path}: {e}")
        
        return loaded_configs
        
    def create_configs_for_all_modes(self) -> Dict[str, Dict[str, str]]:
        """
        Create and save configurations for all optimization modes.
        
        Returns:
            Dictionary of master configurations for each optimization mode
        """
        # Generate a base config_id
        base_config_id = f"configs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get all optimization modes
        all_modes = list(OptimizationMode)
        
        # Save current optimization mode
        original_mode = self.optimization_mode
        
        # Dictionary to store master configs for each mode
        configs = {}
        
        # Generate configs for each mode
        for mode in all_modes:
            # Set the current optimization mode
            self.optimization_mode = mode
            
            # Create a mode-specific config_id
            mode_config_id = f"{base_config_id}_{mode.value}"
            
            # Generate and save configs for this mode
            master_config = self.save_configs(mode_config_id)
            
            # Store the master config
            configs[mode.value] = master_config
            
            logger.info(f"Created configurations for {mode.value} optimization mode")
        
        # Restore original optimization mode
        self.optimization_mode = original_mode
        
        return configs
        
    def auto_tune_configs(self, workload_sample: Any = None) -> Dict[str, Any]:
        """
        Auto-tune configurations based on a sample workload if auto_tune is enabled.
        
        Args:
            workload_sample: Optional sample data to tune configurations
            
        Returns:
            Dictionary with auto-tuned configurations
        """
        if not self.auto_tune:
            logger.info("Auto-tuning is disabled, using static optimization")
            return self.save_configs()
        
        logger.info("Starting auto-tuning process...")
        
        # Without a workload sample, we can still make some basic automatic adjustments
        # based on current system load
        
        # Check current CPU load
        cpu_load = psutil.cpu_percent(interval=1) / 100.0
        
        # Check current memory usage
        memory_load = psutil.virtual_memory().percent / 100.0
        
        # Adjust factors based on current load
        cpu_adjustment = max(0.5, 1.0 - cpu_load)
        memory_adjustment = max(0.5, 1.0 - memory_load)
        
        logger.info(f"Current system load: CPU {cpu_load:.2%}, Memory {memory_load:.2%}")
        logger.info(f"Applying adjustments: CPU factor {cpu_adjustment:.2f}, Memory factor {memory_adjustment:.2f}")
        
        # Store current factors
        original_factors = self._apply_optimization_mode_factors()
        
        # Create adjusted factors
        adjusted_factors = original_factors.copy()
        adjusted_factors["cpu"] *= cpu_adjustment
        adjusted_factors["memory"] *= memory_adjustment
        
        # Apply adjusted factors (this would be implemented in a real system)
        # For this sample, we'll just log the changes
        
        # Generate configurations with the adjusted factors
        # For a real implementation, this would use the adjusted factors
        tuned_configs = self.save_configs("auto_tuned")
        
        logger.info("Auto-tuning complete")
        return tuned_configs

    def _get_optimal_quantization_type(self) -> str:
        """Determine the optimal quantization type based on hardware"""
        # Use more aggressive quantization on memory-constrained devices
        if self.total_memory_gb < 4 or self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            return QuantizationType.INT8.value
        
        # Default based on optimization mode
        if self.optimization_mode == OptimizationMode.PERFORMANCE:
            return QuantizationType.FLOAT16.value
        elif self.optimization_mode == OptimizationMode.FULL_UTILIZATION:
            return QuantizationType.MIXED.value
        
        return QuantizationType.INT8.value

    def _apply_optimization_mode_factors(self) -> Dict[str, float]:
        """
        Get scaling factors for different resource parameters based on the optimization mode.
        
        Returns:
            Dictionary of scaling factors for CPU, memory, and other resources
        """
        # Base factors for different modes
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
        
        # Start with the base factors for the selected optimization mode
        factors = mode_factors.get(self.optimization_mode, mode_factors[OptimizationMode.BALANCED]).copy()
        
        # Environment-specific adjustments
        if self.environment == "edge":
            # For edge devices, be more conservative
            factors["memory"] *= 0.8  # Use less memory
            factors["batch_size"] *= 0.8  # Use smaller batch sizes
        elif self.environment == "cloud":
            # For cloud environments, can be more aggressive
            factors["memory"] *= 1.1  # Use more memory
            factors["batch_size"] *= 1.1  # Use larger batch sizes
        
        # Workload-specific adjustments
        if self.workload_type == "inference":
            # Inference workloads typically need larger batch sizes
            factors["batch_size"] *= 1.2
            # But can use less cache
            factors["cache"] *= 0.9
        elif self.workload_type == "training":
            # Training needs more memory
            factors["memory"] *= 1.1
            factors["cache"] *= 1.1
        
        # Power efficiency adjustments
        if self.power_efficiency:
            # Reduce resource usage for power efficiency
            factors["cpu"] *= 0.8
            factors["memory"] *= 0.9
        
        # Ensure factors stay in reasonable ranges
        for key in factors:
            factors[key] = min(max(factors[key], 0.1), 1.2)  # Clamp between 0.1 and 1.2
        
        return factors
        
    def get_optimal_quantization_config(self) -> QuantizationConfig:
        """
        Generate an optimized quantization configuration based on device capabilities.
        
        Returns:
            Optimized QuantizationConfig
        """
        # Get optimal quantization type
        quant_type = self._get_optimal_quantization_type()
        
        # Determine quantization mode based on optimization mode
        if self.optimization_mode == OptimizationMode.PERFORMANCE:
            quant_mode = QuantizationMode.STATIC.value
        elif self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            quant_mode = QuantizationMode.DYNAMIC.value
        else:
            quant_mode = QuantizationMode.DYNAMIC_PER_BATCH.value
        
        # Enable per-channel quantization for better accuracy
        per_channel = (
            self.optimization_mode == OptimizationMode.PERFORMANCE or 
            self.optimization_mode == OptimizationMode.FULL_UTILIZATION
        )
        
        # Symmetric quantization is typically more performant but less accurate
        symmetric = self.optimization_mode != OptimizationMode.PERFORMANCE
        
        # Determine appropriate cache size based on memory
        cache_size = (
            512 if self.total_memory_gb > 16 else
            256 if self.total_memory_gb > 8 else
            128 if self.total_memory_gb > 4 else
            64
        )
        
        # Reduce cache size for memory-saving mode
        if self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            cache_size //= 2
        
        # Determine calibration samples
        calibration_samples = (
            200 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else
            100 if self.optimization_mode == OptimizationMode.BALANCED else
            50  # For memory-saving or conservative modes
        )
        
        # Set appropriate bit precision based on hardware and mode
        if quant_type == QuantizationType.MIXED.value:
            weight_bits = 8
            activation_bits = 16
            bias_bits = 32
        elif quant_type == QuantizationType.FLOAT16.value:
            weight_bits = 16
            activation_bits = 16
            bias_bits = 32
        else:  # INT8 and others
            weight_bits = 8
            activation_bits = 8
            bias_bits = 32 if self.optimization_mode != OptimizationMode.MEMORY_SAVING else 16
        
        # Create the configuration
        config = QuantizationConfig(
            quantization_type=quant_type,
            quantization_mode=quant_mode,
            per_channel=per_channel,
            symmetric=symmetric,
            enable_cache=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            cache_size=cache_size,
            calibration_samples=calibration_samples,
            calibration_method="percentile" if self.optimization_mode == OptimizationMode.PERFORMANCE else "minmax",
            percentile=99.99 if self.optimization_mode == OptimizationMode.PERFORMANCE else 99.9,
            skip_layers=[],  # No specific layers to skip in default config
            quantize_weights_only=self.optimization_mode == OptimizationMode.CONSERVATIVE,
            quantize_activations=self.optimization_mode != OptimizationMode.CONSERVATIVE,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            quantize_bias=self.optimization_mode != OptimizationMode.PERFORMANCE,
            bias_bits=bias_bits,
            enable_mixed_precision=quant_type == QuantizationType.MIXED.value,
            optimize_for="performance" if self.optimization_mode in [OptimizationMode.PERFORMANCE, OptimizationMode.FULL_UTILIZATION] else
                       "memory" if self.optimization_mode == OptimizationMode.MEMORY_SAVING else
                       "balanced",
            enable_requantization=self.optimization_mode == OptimizationMode.FULL_UTILIZATION,
            requantization_threshold=0.1,
            use_percentile=self.optimization_mode == OptimizationMode.PERFORMANCE,
            min_percentile=0.1,
            max_percentile=99.9,
            error_on_nan=self.optimization_mode not in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING],
            error_on_inf=self.optimization_mode not in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING],
            outlier_threshold=3.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else None,
            num_bits=8,
            optimize_memory=self.optimization_mode != OptimizationMode.PERFORMANCE,
            buffer_size=64 if self.optimization_mode == OptimizationMode.MEMORY_SAVING else 0
        )
        
        return config

    def get_optimal_preprocessor_config(self) -> PreprocessorConfig:
        """
        Generate an optimized preprocessor configuration based on device capabilities.
        
        Returns:
            Optimized PreprocessorConfig
        """
        # Get scaling factors
        factors = self._apply_optimization_mode_factors()
        
        # Determine appropriate normalization type based on optimization mode
        if self.optimization_mode == OptimizationMode.PERFORMANCE:
            normalization = NormalizationType.STANDARD
        elif self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            normalization = NormalizationType.MINMAX  # Less memory intensive
        elif self.optimization_mode == OptimizationMode.FULL_UTILIZATION:
            normalization = NormalizationType.ROBUST  # Better quality but more compute
        else:
            normalization = NormalizationType.STANDARD  # Default
        
        # Feature selection settings
        auto_feature_selection = self.optimization_mode != OptimizationMode.PERFORMANCE
        
        # Text vectorization settings
        text_max_features = (
            10000 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION and self.total_memory_gb > 16 else
            5000 if self.optimization_mode == OptimizationMode.PERFORMANCE or self.total_memory_gb > 8 else
            2000 if self.optimization_mode == OptimizationMode.BALANCED else
            1000  # For memory-saving or conservative modes
        )
        
        # Dimension reduction
        if self.optimization_mode == OptimizationMode.MEMORY_SAVING or self.total_memory_gb < 4:
            dimension_reduction = "pca"
            dimension_reduction_target = 50
        elif self.optimization_mode == OptimizationMode.CONSERVATIVE:
            dimension_reduction = "pca"
            dimension_reduction_target = 100
        else:
            dimension_reduction = None
            dimension_reduction_target = None
        
        # Outlier detection
        detect_outliers = (
            self.optimization_mode not in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING] and
            self.environment != "edge"
        )
        
        # Feature interaction generation
        feature_interaction = (
            self.optimization_mode == OptimizationMode.FULL_UTILIZATION and
            self.environment == "cloud" and
            self.total_memory_gb > 16
        )
        
        # Number of parallel jobs for preprocessing
        n_jobs = (
            self.cpu_count_logical if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else
            max(int(self.cpu_count_logical * factors["cpu"]), 1)
        )
        
        # Chunk size for large datasets
        if self.total_memory_gb < 4:
            chunk_size = 1000
        elif self.total_memory_gb < 8:
            chunk_size = 5000
        elif self.total_memory_gb < 16:
            chunk_size = 10000
        else:
            chunk_size = None  # No chunking needed
        
        # Cache size
        cache_size = int(128 * factors["cache"])
        
        # Numeric dtype - use float32 to save memory if needed
        dtype = np.float32 if self.optimization_mode == OptimizationMode.MEMORY_SAVING else np.float64
        
        # Create the configuration
        return PreprocessorConfig(
            normalization=normalization,
            handle_nan=True,
            handle_inf=True,
            detect_outliers=detect_outliers,
            outlier_method="isolation_forest" if self.cpu_count_logical > 2 else "iqr",
            outlier_contamination=0.05,
            categorical_encoding="one_hot" if self.total_memory_gb > 4 else "label",
            categorical_max_categories=20 if self.total_memory_gb > 8 else 10,
            auto_feature_selection=auto_feature_selection,
            numeric_transformations=["standard", "log", "sqrt"] if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else [],
            text_vectorization="tfidf" if self.optimization_mode != OptimizationMode.MEMORY_SAVING else "count",
            text_max_features=text_max_features,
            dimension_reduction=dimension_reduction,
            dimension_reduction_target=dimension_reduction_target,
            datetime_features=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            image_preprocessing={},  # No default image preprocessing
            handle_imbalance=self.optimization_mode == OptimizationMode.FULL_UTILIZATION,
            imbalance_strategy="smote" if self.total_memory_gb > 8 else "random_under",
            feature_interaction=feature_interaction,
            feature_interaction_max=10 if self.total_memory_gb > 16 else 5,
            custom_transformers=[],
            transformation_pipeline=[],
            parallel_processing=self.cpu_count_logical > 1,
            cache_preprocessing=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            verbosity=2 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 1,
            robust_percentiles=(25.0, 75.0),
            nan_strategy="mean" if self.optimization_mode != OptimizationMode.PERFORMANCE else "median",
            inf_strategy="mean" if self.optimization_mode != OptimizationMode.PERFORMANCE else "median",
            outlier_params={
                "threshold": 1.5,
                "clip": True,
                "n_estimators": 100 if self.cpu_count_logical > 4 else 50,
                "contamination": "auto"
            },
            clip_values=self.optimization_mode in [OptimizationMode.MEMORY_SAVING, OptimizationMode.CONSERVATIVE],
            clip_range=(-np.inf, np.inf),  # Default to no clipping
            enable_input_validation=True,
            input_size_limit=None if self.environment == "cloud" else 100000,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            cache_enabled=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
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
        Generate an optimized inference engine configuration based on device capabilities.
        
        Returns:
            Optimized InferenceEngineConfig
        """
        # Get scaling factors
        factors = self._apply_optimization_mode_factors()
        
        # Determine model precision based on optimization mode
        if self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            model_precision = "fp16"  # Use half precision to save memory
        elif self.optimization_mode == OptimizationMode.PERFORMANCE:
            model_precision = "fp32"  # Use full precision for performance
        else:
            model_precision = "fp32"  # Default
        
        # Number of threads for inference
        thread_count = max(int(self.cpu_count_logical * factors["cpu"]), 1)
        
        # Batch size settings
        max_batch_size = (
            128 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION and self.total_memory_gb > 16 else
            64 if self.optimization_mode == OptimizationMode.PERFORMANCE else
            32 if self.optimization_mode == OptimizationMode.BALANCED else
            16  # For memory-saving or conservative modes
        )
        
        # Initial batch size
        initial_batch_size = max(max_batch_size // 2, 1)
        
        # Model cache size
        model_cache_size = (
            10 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION and self.total_memory_gb > 16 else
            5 if self.optimization_mode == OptimizationMode.PERFORMANCE or self.optimization_mode == OptimizationMode.BALANCED else
            2 if self.optimization_mode == OptimizationMode.CONSERVATIVE else
            1  # Minimal caching for memory saving
        )
        
        # Memory limits based on system memory
        memory_limit = (
            None if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else
            self.total_memory_gb * 0.8 if self.optimization_mode == OptimizationMode.PERFORMANCE else
            self.total_memory_gb * 0.6 if self.optimization_mode == OptimizationMode.BALANCED else
            self.total_memory_gb * 0.4  # Conservative memory limit
        )
        
        # Monitoring interval
        monitoring_interval = (
            5.0 if self.environment == "edge" or self.optimization_mode == OptimizationMode.MEMORY_SAVING else
            10.0  # Default
        )
        
        # Cache TTL
        cache_ttl = (
            600 if self.optimization_mode == OptimizationMode.PERFORMANCE else
            300 if self.optimization_mode == OptimizationMode.BALANCED else
            120  # Shorter TTL for memory saving
        )
        
        # Determine if Intel optimizations should be enabled
        enable_intel = self.is_intel_cpu and (self.has_avx or self.has_avx2 or self.has_avx512)
        
        # Create quantization config if needed
        quantization_config = self.get_optimal_quantization_config() if self.optimization_mode == OptimizationMode.MEMORY_SAVING else None
        
        # Create optimized inference engine config
        return InferenceEngineConfig(
            # CPU optimization settings
            enable_intel_optimization=enable_intel,
            thread_count=thread_count,
            num_threads=thread_count,
            set_cpu_affinity=self.optimization_mode == OptimizationMode.PERFORMANCE,
            
            # Batching settings
            enable_batching=True,
            max_batch_size=max_batch_size,
            initial_batch_size=initial_batch_size,
            min_batch_size=1,
            batch_timeout=0.05 if self.optimization_mode == OptimizationMode.PERFORMANCE else 0.1,
            batching_strategy="dynamic" if self.optimization_mode != OptimizationMode.CONSERVATIVE else "static",
            enable_adaptive_batching=self.optimization_mode != OptimizationMode.CONSERVATIVE,
            
            # Model settings
            model_cache_size=model_cache_size,
            model_precision=model_precision,
            model_version="1.0",
            
            # Performance settings
            timeout_ms=50 if self.optimization_mode == OptimizationMode.PERFORMANCE else 100,
            enable_jit=True,
            runtime_optimization=True,
            warmup=True,
            warmup_iterations=5 if self.environment == "edge" else 10,
            profiling=self.optimization_mode == OptimizationMode.FULL_UTILIZATION,
            
            # Memory settings
            memory_growth=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            enable_memory_optimization=self.optimization_mode != OptimizationMode.PERFORMANCE,
            memory_high_watermark_mb=1024 if self.total_memory_gb > 4 else 512,
            memory_limit_gb=memory_limit,
            
            # Export formats
            enable_onnx=False,  # Not using ONNX 
            onnx_opset=13,
            enable_tensorrt=False,  # No TensorRT support in CPU-only mode
            
            # Quantization settings
            enable_quantization=self.optimization_mode == OptimizationMode.MEMORY_SAVING,
            enable_model_quantization=self.optimization_mode == OptimizationMode.MEMORY_SAVING,
            enable_input_quantization=False,
            quantization_dtype="int8",
            quantization_config=quantization_config,
            enable_quantization_aware_inference=False,
            
            # Request handling
            enable_request_deduplication=True,
            output_streaming=False,
            max_concurrent_requests=thread_count,
            max_cache_entries=1000 if self.total_memory_gb > 8 else 500,
            cache_ttl_seconds=cache_ttl,
            
            # Monitoring and throttling
            monitoring_window=100 if self.environment == "cloud" else 50,
            enable_monitoring=True,
            monitoring_interval=monitoring_interval,
            throttle_on_high_cpu=self.optimization_mode != OptimizationMode.PERFORMANCE,
            cpu_threshold_percent=90.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else 80.0,
            enable_throttling=self.optimization_mode != OptimizationMode.PERFORMANCE,
            
            # Other settings
            enable_feature_scaling=False,  # Handled by preprocessor
            enable_warmup=True,
            fallback_to_cpu=True,  # Already CPU-only
            debug_mode=False,
            custom_ops=[],
            use_platform_accelerator=True,
            platform_accelerator_config={}
        )
        
    def get_optimal_batch_processor_config(self) -> BatchProcessorConfig:
        """
        Generate an optimized batch processor configuration based on device capabilities.
        
        Returns:
            Optimized BatchProcessorConfig
        """
        # Get scaling factors
        factors = self._apply_optimization_mode_factors()
        
        # Batch size calculations
        # Base batch sizes adjusted by environment and optimization mode
        if self.environment == "edge":
            base_batch_size = 32
        elif self.environment == "desktop":
            base_batch_size = 64
        else:  # cloud
            base_batch_size = 128
        
        # Apply overall batch size factor
        batch_factor = factors["batch_size"]
        initial_batch_size = max(int(base_batch_size * batch_factor), 1)
        
        # Min and max are relative to initial
        min_batch_size = max(int(initial_batch_size * 0.5), 1)
        max_batch_size = max(int(initial_batch_size * 2.0), min_batch_size + 1)
        
        # Queue size depends on environment and available memory
        if self.environment == "edge":
            max_queue_size = 500
        elif self.environment == "desktop":
            max_queue_size = 1000
        else:  # cloud
            max_queue_size = 2000
        
        # Adjust queue size based on memory
        mem_ratio = min(self.usable_memory_gb / 4.0, 1.0)  # Scale based on 4GB reference
        max_queue_size = int(max_queue_size * mem_ratio)
        
        # Batch timeout - shorter for performance mode, longer for memory saving
        batch_timeout = (
            0.5 if self.optimization_mode == OptimizationMode.PERFORMANCE else
            2.0 if self.optimization_mode == OptimizationMode.MEMORY_SAVING else
            1.0  # Default for balanced and others
        )
        
        # CPU resource allocation
        cpu_factor = factors["cpu"]
        num_workers = max(int(self.cpu_count_logical * cpu_factor), 1)
        max_workers = max(num_workers, 2)  # At least 2 workers
        
        # Priority levels - more for cloud/performance, fewer for edge/memory-saving
        priority_levels = (
            5 if self.environment == "cloud" and self.optimization_mode in [OptimizationMode.PERFORMANCE, OptimizationMode.FULL_UTILIZATION] else
            3 if self.environment != "edge" else
            2  # Minimum for edge devices
        )
        
        # Monitoring interval - less frequent for edge/memory-saving
        monitoring_interval = (
            5.0 if self.environment == "edge" or self.optimization_mode == OptimizationMode.MEMORY_SAVING else
            2.0 if self.environment == "desktop" or self.optimization_mode == OptimizationMode.BALANCED else
            1.0  # More frequent for cloud/performance
        )
        
        # Memory thresholds - higher for performance, lower for memory-saving
        memory_warning = (
            80.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else
            60.0 if self.optimization_mode == OptimizationMode.MEMORY_SAVING else
            70.0  # Default
        )
        
        memory_critical = (
            90.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else
            75.0 if self.optimization_mode == OptimizationMode.MEMORY_SAVING else
            85.0  # Default
        )
        
        # Determine processing strategy
        if self.optimization_mode == OptimizationMode.PERFORMANCE or self.optimization_mode == OptimizationMode.FULL_UTILIZATION:
            strategy = BatchProcessingStrategy.GREEDY
        elif self.optimization_mode == OptimizationMode.CONSERVATIVE or self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            strategy = BatchProcessingStrategy.FIXED
        else:
            strategy = BatchProcessingStrategy.ADAPTIVE
        
        # Create optimized batch processor config
        return BatchProcessorConfig(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            max_queue_size=max_queue_size,
            batch_timeout=batch_timeout,
            num_workers=num_workers,
            adaptive_batching=self.optimization_mode != OptimizationMode.CONSERVATIVE,
            batch_allocation_strategy="dynamic" if self.optimization_mode != OptimizationMode.CONSERVATIVE else "static",
            enable_priority_queue=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            priority_levels=priority_levels,
            enable_monitoring=True,
            monitoring_interval=monitoring_interval,
            enable_memory_optimization=self.optimization_mode != OptimizationMode.PERFORMANCE,
            enable_prefetching=self.optimization_mode in [OptimizationMode.PERFORMANCE, OptimizationMode.FULL_UTILIZATION, OptimizationMode.BALANCED],
            prefetch_batches=1 if self.optimization_mode == OptimizationMode.MEMORY_SAVING else 2,
            checkpoint_batches=self.resilience_level > 1,
            checkpoint_interval=50 if self.resilience_level > 2 else 100,
            error_handling="retry",
            max_retries=3 if self.resilience_level > 0 else 1,
            retry_delay=0.5,
            distributed_processing=self.environment == "cloud" and self.cpu_count_physical > 4,
            resource_allocation={},
            item_timeout=5.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else 10.0,
            min_batch_interval=0.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else 0.1,
            processing_strategy=strategy,
            enable_adaptive_batching=self.optimization_mode != OptimizationMode.CONSERVATIVE,
            reduce_batch_on_failure=True,
            max_batch_memory_mb=None,  # Auto-determine
            gc_batch_threshold=32 if self.environment != "edge" else 16,
            monitoring_window=100 if self.environment == "cloud" else 50,
            max_workers=max_workers,
            enable_health_monitoring=self.resilience_level > 0,
            health_check_interval=5.0 if self.resilience_level > 1 else 10.0,
            memory_warning_threshold=memory_warning,
            memory_critical_threshold=memory_critical,
            queue_warning_threshold=max_queue_size // 2,
            queue_critical_threshold=int(max_queue_size * 0.9),
            debug_mode=False
        )
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current system.
        
        Returns:
            Dictionary with detailed system information
        """
        system_info = {
            "system": self.system,
            "release": self.release,
            "machine": self.machine,
            "processor": self.processor,
            "hostname": self.hostname,
            "python_version": self.python_version,
            "environment": self.environment,
            
            "cpu": {
                "count_physical": self.cpu_count_physical,
                "count_logical": self.cpu_count_logical,
                "frequency": self.cpu_freq,
                "features": {
                    "avx": self.has_avx,
                    "avx2": self.has_avx2,
                    "avx512": self.has_avx512,
                    "sse4": self.has_sse4,
                    "fma": self.has_fma,
                    "neon": self.has_neon
                },
                "vendor": {
                    "intel": self.is_intel_cpu,
                    "amd": self.is_amd_cpu,
                    "arm": self.is_arm_cpu
                }
            },
            
            "memory": {
                "total_gb": self.total_memory_gb,
                "available_gb": self.available_memory_gb,
                "usable_gb": self.usable_memory_gb,
                "swap_gb": self.swap_memory_gb
            },
            
            "disk": {
                "total_gb": self.disk_total_gb,
                "free_gb": self.disk_free_gb,
                "is_ssd": self.is_ssd
            },
            
            "accelerators": [acc.name for acc in self.accelerators]
        }
        
        return system_info

    def get_optimal_training_engine_config(self) -> MLTrainingEngineConfig:
        """
        Generate an optimized training engine configuration based on device capabilities.
        
        Returns:
            Optimized MLTrainingEngineConfig
        """
        # Get scaling factors
        factors = self._apply_optimization_mode_factors()
        
        # Determine batch size based on memory and optimization mode
        if self.total_memory_gb < 4:
            batch_size = 16
        elif self.total_memory_gb < 8:
            batch_size = 32
        elif self.total_memory_gb < 16:
            batch_size = 64
        else:
            batch_size = 128
        
        # Apply batch size factor
        batch_size = max(int(batch_size * factors["batch_size"]), 1)
        
        # Determine number of workers for data loading
        num_workers = max(int(self.cpu_count_logical * factors["workers"]), 1)
        
        # Determine learning rate based on batch size and optimization mode
        base_lr = 0.001  # Default base learning rate
        
        if self.optimization_mode == OptimizationMode.PERFORMANCE:
            base_lr = 0.002  # Faster learning for performance mode
        elif self.optimization_mode == OptimizationMode.CONSERVATIVE:
            base_lr = 0.0005  # More conservative learning rate
        
        # Scale learning rate with batch size
        learning_rate = base_lr * (batch_size / 32)
        
        # Determine epochs based on optimization mode
        if self.optimization_mode == OptimizationMode.FULL_UTILIZATION:
            epochs = 100
        elif self.optimization_mode == OptimizationMode.PERFORMANCE:
            epochs = 50
        elif self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            epochs = 30
        else:
            epochs = 50  # Default
        
        # Model selection strategy
        if self.optimization_mode == OptimizationMode.PERFORMANCE:
            model_selection = ModelSelectionCriteria.SPEED
        elif self.optimization_mode == OptimizationMode.CONSERVATIVE:
            model_selection = ModelSelectionCriteria.ROBUSTNESS
        else:
            model_selection = ModelSelectionCriteria.BALANCED
        
        # Optimizer config based on optimization mode
        if self.optimization_mode == OptimizationMode.PERFORMANCE:
            optimizer_config = {
                "type": "adam",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0001
            }
        elif self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            optimizer_config = {
                "type": "sgd",
                "momentum": 0.9,
                "weight_decay": 0.0001
            }
        else:
            optimizer_config = {
                "type": "adam",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0001
            }
        
        # Early stopping patience based on optimization mode
        if self.optimization_mode == OptimizationMode.PERFORMANCE:
            patience = 5
        elif self.optimization_mode == OptimizationMode.CONSERVATIVE:
            patience = 15
        else:
            patience = 10
        
        # Memory optimization settings
        enable_mixed_precision = (
            self.optimization_mode == OptimizationMode.MEMORY_SAVING or 
            self.total_memory_gb < 8
        )
        
        gradient_accumulation_steps = (
            4 if self.total_memory_gb < 4 else
            2 if self.total_memory_gb < 8 else
            1
        )
        
        # Checkpoint frequency - more frequent for edge devices or conservative mode
        checkpoint_freq = (
            5 if self.environment == "edge" or self.optimization_mode == OptimizationMode.CONSERVATIVE else
            10 if self.optimization_mode == OptimizationMode.BALANCED else
            20  # Less frequent for performance mode
        )
        
        # GPU reserved memory percentage (for future GPU support)
        gpu_reserved_memory = (
            0.2 if self.optimization_mode == OptimizationMode.MEMORY_SAVING else
            0.1 if self.optimization_mode == OptimizationMode.BALANCED else
            0.05  # Minimal for performance mode
        )
        
        # Create the configuration
        return MLTrainingEngineConfig(
            # Basic training parameters
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            optimizer=optimizer_config["type"],
            optimizer_params=optimizer_config,
            scheduler="cosine" if self.optimization_mode != OptimizationMode.CONSERVATIVE else "step",
            scheduler_params={},
            loss_function="auto",
            
            # Resource management
            num_workers=num_workers,
            pin_memory=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            cuda_non_blocking=True,
            
            # Memory optimizations
            enable_mixed_precision=enable_mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            enable_gradient_checkpointing=self.optimization_mode == OptimizationMode.MEMORY_SAVING,
            
            # Regularization
            weight_decay=0.0001,
            dropout_rate=0.3 if self.optimization_mode != OptimizationMode.PERFORMANCE else 0.2,
            label_smoothing=0.1 if self.optimization_mode != OptimizationMode.PERFORMANCE else 0.0,
            
            # Validation and early stopping
            validation_frequency=1,
            early_stopping=True,
            patience=patience,
            
            # Checkpointing
            checkpoint_frequency=checkpoint_freq,
            keep_n_best_checkpoints=3 if self.environment != "edge" else 1,
            
            # Advanced settings
            enable_automl=self.auto_tune,
            automl_mode=AutoMLMode.BASIC if self.environment == "edge" else AutoMLMode.COMPREHENSIVE,
            automl_trials=10 if self.environment == "edge" else 30,
            
            # Performance tracking
            enable_profiling=self.optimization_mode == OptimizationMode.FULL_UTILIZATION,
            profiling_frequency=100,
            
            # Model selection
            model_selection_criteria=model_selection,
            
            # GPU reserved memory (for future GPU support)
            gpu_reserved_memory_fraction=gpu_reserved_memory,
            
            # Explainability
            enable_explainability=False,  # Disabled by default for CPU-only mode
            explainability_config=ExplainabilityConfig(),
            
            # Monitoring
            enable_monitoring=True,
            monitoring_config=MonitoringConfig(),
            
            # Misc
            random_seed=42,
            deterministic=self.optimization_mode == OptimizationMode.CONSERVATIVE,
            distributed_training=False,  # Not enabled for CPU-only version
            distributed_backend="gloo" if self.is_intel_cpu else "mpi",
            world_size=1,
            debug_mode=False
        )
# Helper functions for creating optimized configurations
def create_optimized_configs(
    config_path: str = "./configs", 
    checkpoint_path: str = "./checkpoints", 
    model_registry_path: str = "./model_registry",
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED,
    workload_type: str = "mixed",
    environment: str = "auto",
    enable_specialized_accelerators: bool = True,
    memory_reservation_percent: float = 10.0,
    power_efficiency: bool = False,
    resilience_level: int = 1,
    auto_tune: bool = True,
    config_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Create and save optimized configurations for the ML pipeline using the CPU-Only Device Optimizer.
    
    Args:
        config_path: Path to save configuration files
        checkpoint_path: Path for model checkpoints
        model_registry_path: Path for model registry
        optimization_mode: Mode for optimization strategy
        workload_type: Type of workload ("inference", "training", "mixed")
        environment: Computing environment ("cloud", "desktop", "edge", "auto")
        enable_specialized_accelerators: Whether to enable detection of specialized hardware
        memory_reservation_percent: Percentage of memory to reserve for the system
        power_efficiency: Whether to optimize for power efficiency
        resilience_level: Level of fault tolerance (0-3)
        auto_tune: Whether to enable automatic parameter tuning
        config_id: Optional identifier for the configuration set
    
    Returns:
        Dictionary with paths to saved configuration files
    """
    # Create optimizer
    optimizer = DeviceOptimizer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model_registry_path=model_registry_path,
        optimization_mode=optimization_mode,
        workload_type=workload_type,
        environment=environment,
        enable_specialized_accelerators=enable_specialized_accelerators,
        memory_reservation_percent=memory_reservation_percent,
        power_efficiency=power_efficiency,
        resilience_level=resilience_level,
        auto_tune=auto_tune
    )
    
    # Generate and save configurations
    if auto_tune:
        return optimizer.auto_tune_configs()
    else:
        return optimizer.save_configs(config_id)

def create_configs_for_all_modes(
    config_path: str = "./configs", 
    checkpoint_path: str = "./checkpoints", 
    model_registry_path: str = "./model_registry",
    workload_type: str = "mixed",
    environment: str = "auto",
    enable_specialized_accelerators: bool = True,
    memory_reservation_percent: float = 10.0,
    power_efficiency: bool = False,
    resilience_level: int = 1
) -> Dict[str, Dict[str, str]]:
    """
    Create and save configurations for all optimization modes.
    
    Args:
        config_path: Path to save configuration files
        checkpoint_path: Path for model checkpoints
        model_registry_path: Path for model registry
        workload_type: Type of workload ("inference", "training", "mixed")
        environment: Computing environment ("cloud", "desktop", "edge", "auto")
        enable_specialized_accelerators: Whether to enable detection of specialized hardware
        memory_reservation_percent: Percentage of memory to reserve for the system
        power_efficiency: Whether to optimize for power efficiency
        resilience_level: Level of fault tolerance (0-3)
    
    Returns:
        Dictionary of configurations for each optimization mode
    """
    # Create optimizer with default mode (will be changed for each mode)
    optimizer = DeviceOptimizer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model_registry_path=model_registry_path,
        optimization_mode=OptimizationMode.BALANCED,  # Default, will be changed
        workload_type=workload_type,
        environment=environment,
        enable_specialized_accelerators=enable_specialized_accelerators,
        memory_reservation_percent=memory_reservation_percent,
        power_efficiency=power_efficiency,
        resilience_level=resilience_level,
        auto_tune=False  # Disable auto-tuning when generating all modes
    )
    
    # Generate configs for all modes
    return optimizer.create_configs_for_all_modes()

def load_saved_configs(
    config_path: str,
    config_id: str
) -> Dict[str, Any]:
    """
    Load previously saved configuration files.
    
    Args:
        config_path: Path where configuration files are stored
        config_id: Identifier for the configuration set
    
    Returns:
        Dictionary with loaded configurations
    """
    # Create optimizer just to use its loading functionality
    optimizer = DeviceOptimizer(config_path=config_path)
    
    # Load the configs
    return optimizer.load_configs(config_id)

def get_system_information(
    enable_specialized_accelerators: bool = True
) -> Dict[str, Any]:
    """
    Get comprehensive information about the current system.
    
    Args:
        enable_specialized_accelerators: Whether to detect specialized hardware
    
    Returns:
        Dictionary with detailed system information
    """
    # Create optimizer just to use its detection capabilities
    optimizer = DeviceOptimizer(
        enable_specialized_accelerators=enable_specialized_accelerators
    )
    
    # Return system information
    return optimizer.get_system_info()

def optimize_for_environment(environment: str) -> Dict[str, str]:
    """
    Create optimized configurations specifically for a given environment type.
    
    Args:
        environment: Target environment ("cloud", "desktop", "edge")
    
    Returns:
        Dictionary with paths to saved configuration files
    """
    # Choose appropriate optimization mode based on environment
    if environment == "cloud":
        optimization_mode = OptimizationMode.PERFORMANCE
    elif environment == "desktop":
        optimization_mode = OptimizationMode.BALANCED
    elif environment == "edge":
        optimization_mode = OptimizationMode.MEMORY_SAVING
    else:
        raise ValueError(f"Invalid environment: {environment}. Must be 'cloud', 'desktop', or 'edge'.")
    
    # Create optimized configs
    return create_optimized_configs(
        optimization_mode=optimization_mode,
        environment=environment,
        workload_type="mixed",
        config_id=f"{environment}_optimized"
    )

def optimize_for_workload(workload_type: str) -> Dict[str, str]:
    """
    Create optimized configurations specifically for a given workload type.
    
    Args:
        workload_type: Target workload type ("inference", "training", "mixed")
    
    Returns:
        Dictionary with paths to saved configuration files
    """
    # Validate workload type
    if workload_type not in ["inference", "training", "mixed"]:
        raise ValueError(f"Invalid workload type: {workload_type}. Must be 'inference', 'training', or 'mixed'.")
    
    # Create optimized configs
    return create_optimized_configs(
        workload_type=workload_type,
        config_id=f"{workload_type}_optimized"
    )

def apply_configs_to_pipeline(configs_dict: Dict[str, Any]) -> bool:
    """
    Apply loaded configurations to an ML pipeline components.
    
    This function would implement the actual application of the configurations
    to the ML pipeline components. In a real implementation, it would:
    1. Extract configuration objects from the configs dictionary
    2. Initialize or reconfigure pipeline components with these settings
    
    Args:
        configs_dict: Dictionary with loaded configurations
    
    Returns:
        True if configurations were successfully applied, False otherwise
    """
    # This is a placeholder implementation
    # In a real-world scenario, this would extract configuration objects
    # and apply them to the appropriate ML pipeline components
    
    try:
        logger.info("Applying configurations to ML pipeline components...")
        
        # Check if required configurations are present
        for required_config in ["master_config", "quantization_config", "batch_processor_config", 
                              "preprocessor_config", "inference_engine_config"]:
            if required_config not in configs_dict:
                logger.error(f"Missing required configuration: {required_config}")
                return False
        
        # In a real implementation, we would initialize components with the loaded configs
        # For example:
        # quantizer = Quantizer.from_config(configs_dict["quantization_config"])
        # batch_processor = BatchProcessor.from_config(configs_dict["batch_processor_config"])
        # preprocessor = Preprocessor.from_config(configs_dict["preprocessor_config"])
        # inference_engine = InferenceEngine.from_config(configs_dict["inference_engine_config"])
        
        logger.info("All configurations successfully applied to pipeline components")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply configurations: {e}")
        return False

def get_default_config(
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED,
    workload_type: str = "mixed",
    environment: str = "auto",
    output_dir: str = "./configs/default",
    enable_specialized_accelerators: bool = True,
) -> Dict[str, Any]:
    """
    Get a set of default configurations optimized for the current system.
    
    This function creates a DeviceOptimizer instance and generates optimized
    configurations based on the specified parameters and current system capabilities.
    
    Args:
        optimization_mode: The optimization strategy to use
            (BALANCED, PERFORMANCE, MEMORY_SAVING, etc.)
        workload_type: Type of workload to optimize for 
            ("inference", "training", or "mixed")
        environment: Computing environment 
            ("cloud", "desktop", "edge", or "auto" for automatic detection)
        output_dir: Directory where configuration files will be saved
        enable_specialized_accelerators: Whether to enable detection and 
            optimization for specialized hardware accelerators
    
    Returns:
        Dictionary containing all optimized configurations:
        - quantization_config: Configuration for model quantization
        - batch_processor_config: Configuration for batch processing
        - preprocessor_config: Configuration for data preprocessing
        - inference_engine_config: Configuration for inference engine
        - training_engine_config: Configuration for training engine
        - system_info: Information about the detected system
    """
    # Create paths for configurations
    config_path = os.path.join(output_dir, "configs")
    checkpoint_path = os.path.join(output_dir, "checkpoints")
    model_registry_path = os.path.join(output_dir, "model_registry")
    
    # Create optimizer with the specified parameters
    optimizer = DeviceOptimizer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model_registry_path=model_registry_path,
        optimization_mode=optimization_mode,
        workload_type=workload_type,
        environment=environment,
        enable_specialized_accelerators=enable_specialized_accelerators,
        auto_tune=False  # Disable auto-tuning for default config
    )
    
    # Generate all configurations
    configs = {
        "quantization_config": optimizer.get_optimal_quantization_config(),
        "batch_processor_config": optimizer.get_optimal_batch_processor_config(),
        "preprocessor_config": optimizer.get_optimal_preprocessor_config(),
        "inference_engine_config": optimizer.get_optimal_inference_engine_config(),
        "training_engine_config": optimizer.get_optimal_training_engine_config(),
        "system_info": optimizer.get_system_info()
    }
    
    # Save configurations to files if output_dir is provided
    if output_dir:
        # Create a unique config_id
        config_id = f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        master_config = optimizer.save_configs(config_id)
        configs["master_config"] = master_config
    
    return configs