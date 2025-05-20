import os
import platform
import psutil
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple # Removed TypeVar
import multiprocessing
import socket
import uuid
import datetime
import sys
import shutil
import tempfile
import re
from enum import Enum, auto # Keep for HardwareAccelerator
from dataclasses import asdict, is_dataclass # Keep for generic dataclass handling

# Import from the new configs.py
# Assuming configs.py is in the same directory or accessible via Python path
from .configs import (
    QuantizationType, QuantizationMode, QuantizationConfig,
    BatchProcessorConfig, BatchProcessingStrategy,
    PreprocessorConfig, NormalizationType,
    InferenceEngineConfig, OptimizationMode, # Using OptimizationMode from new configs
    MLTrainingEngineConfig, TaskType, OptimizationStrategy as TrainingOptimizationStrategy,
    ModelSelectionCriteria, AutoMLMode, ExplainabilityConfig, MonitoringConfig
    # SecurityConfig is handled as a Dict in MLTrainingEngineConfig
)

# Setup logging - (assuming this is already correctly set up as per previous)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cpu_device_optimizer")


class HardwareAccelerator(Enum):
    NONE = auto()
    INTEL_IPEX = auto()
    INTEL_MKL = auto()
    ARM_NEON = auto()


class DeviceOptimizer:
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
                 auto_tune: bool = True,
                 debug_mode: bool = False):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_registry_path = Path(model_registry_path)
        
        # Ensure optimization_mode is an instance of the imported OptimizationMode enum
        if isinstance(optimization_mode, str):
            self.optimization_mode = OptimizationMode(optimization_mode)
        else:
            self.optimization_mode = optimization_mode

        self.workload_type = workload_type.lower()
        self.power_efficiency = power_efficiency
        self.resilience_level = min(max(resilience_level, 0), 3)
        self.auto_tune = auto_tune
        self.memory_reservation_percent = min(max(memory_reservation_percent, 0), 50)
        self.needs_feature_scaling = False # Placeholder
        
        self.onnx_available = False
        try:
            import onnxruntime
            self.onnx_available = True
        except ImportError:
            pass

        self.threadpoolctl_available = False
        try:
            import threadpoolctl
            self.threadpoolctl_available = True
        except ImportError:
            pass
        self.treelite_available = False # Placeholder

        self.has_gpu = False # CPU-only optimizer
        self.debug_mode = debug_mode

        for path_dir in [self.config_path, self.checkpoint_path, self.model_registry_path]:
            path_dir.mkdir(parents=True, exist_ok=True)

        self._detect_system_info()
        self._detect_cpu_capabilities()
        self._detect_memory_info()
        self._detect_disk_info()
        self.accelerators = []
        if enable_specialized_accelerators:
            self._detect_specialized_accelerators()
        if environment == "auto":
            self.environment = self._detect_environment()
        else:
            self.environment = environment
        self._log_system_overview()

    def _detect_system_info(self):
        self.system = platform.system()
        self.release = platform.release()
        self.machine = platform.machine()
        self.processor = platform.processor()
        if not self.processor and self.system == "Darwin":
             try:
                self.processor = subprocess.check_output(['sysctl', "-n", "machdep.cpu.brand_string"]).strip().decode()
             except Exception:
                self.processor = "N/A"
        self.hostname = socket.gethostname()
        self.python_version = platform.python_version()

    def _detect_cpu_capabilities(self):
        self.cpu_count_physical = psutil.cpu_count(logical=False) or 1
        self.cpu_count_logical = psutil.cpu_count(logical=True) or 1
        self.cpu_freq = self._get_cpu_frequency()
        self.has_avx = False
        self.has_avx2 = False
        self.has_avx512 = False
        self.has_sse4 = False
        self.has_fma = False
        if self.system == "Linux" and (self.machine == "x86_64" or self.machine == "AMD64"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read().lower()
                    self.has_avx = "avx " in cpuinfo
                    self.has_avx2 = "avx2 " in cpuinfo
                    self.has_avx512 = "avx512f " in cpuinfo # Common AVX512 flag
                    self.has_sse4 = "sse4_1 " in cpuinfo or "sse4_2 " in cpuinfo
                    self.has_fma = "fma " in cpuinfo
            except Exception as e:
                logger.warning(f"Failed to check advanced CPU features on Linux: {e}")
        elif self.system == "Windows":
            try:
                import subprocess
                result = subprocess.check_output("wmic cpu get Name", shell=True, text=True)
                cpu_name = result.strip().split("\n")[-1].lower() # Get the last non-empty line
                self.has_avx = "avx" in cpu_name
                self.has_avx2 = "avx2" in cpu_name
                self.has_avx512 = "xeon" in cpu_name and "avx-512" in cpu_name # More specific check
                self.has_sse4 = True # Assume modern CPUs
                self.has_fma = "fma" in cpu_name or ("intel" in cpu_name and "core" in cpu_name) # Broad assumption
            except Exception as e:
                logger.warning(f"Failed to check advanced CPU features on Windows: {e}")
        elif self.system == "Darwin":  # macOS
            try:
                import subprocess
                result = subprocess.check_output(['sysctl', "-a"], text=True)
                features = result.lower()
                self.has_avx = "machdep.cpu.features: avx1.0" in features or "avx" in features # Broader check
                self.has_avx2 = "machdep.cpu.features: avx2" in features
                self.has_avx512 = "machdep.cpu.features: avx512" in features # Check for various AVX512 flags
                self.has_sse4 = "machdep.cpu.features: sse4.1" in features or "machdep.cpu.features: sse4.2" in features
                self.has_fma = "machdep.cpu.features: fma" in features
            except Exception as e:
                logger.warning(f"Failed to check advanced CPU features on macOS: {e}")

        self.is_intel_cpu = "intel" in (self.processor.lower() if self.processor else "")
        self.is_amd_cpu = "amd" in (self.processor.lower() if self.processor else "")
        self.is_arm_cpu = "arm" in self.machine.lower() or "aarch64" in self.machine.lower()
        self.has_neon = self.is_arm_cpu # Simplified assumption, NEON is common on ARMv7+

    def _get_cpu_frequency(self) -> Dict[str, float]:
        try:
            freq = psutil.cpu_freq()
            if freq: # freq can be None or fields can be 0
                return {
                    "current": freq.current if freq.current else 0,
                    "min": freq.min if freq.min else 0,
                    "max": freq.max if freq.max else 0
                }
        except (AttributeError, NotImplementedError, Exception) as e: # Catch specific errors if psutil.cpu_freq() is not supported
            logger.warning(f"Failed to get CPU frequency: {e}")
        return {"current": 0, "min": 0, "max": 0}

    def _detect_memory_info(self):
        mem = psutil.virtual_memory()
        self.total_memory_gb = mem.total / (1024 ** 3)
        self.available_memory_gb = mem.available / (1024 ** 3)
        reservation = self.memory_reservation_percent / 100.0
        self.usable_memory_gb = self.total_memory_gb * (1 - reservation)
        try:
            swap = psutil.swap_memory()
            self.swap_memory_gb = swap.total / (1024 ** 3)
        except Exception: # psutil.swap_memory() can fail on some systems (e.g. no swap)
            self.swap_memory_gb = 0

    def _detect_disk_info(self):
        try:
            disk_path = Path.home() # Check home directory's disk by default
            if not disk_path.exists(): # Fallback if home dir is weird
                disk_path = Path("/") if self.system != "Windows" else Path("C:\\")
            usage = psutil.disk_usage(str(disk_path))
            self.disk_total_gb = usage.total / (1024 ** 3)
            self.disk_free_gb = usage.free / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Failed to get disk information for '{disk_path}': {e}")
            self.disk_total_gb = 0
            self.disk_free_gb = 0
        self.is_ssd = self._check_if_ssd()

    def _check_if_ssd(self) -> bool:
        if self.system == "Linux":
            try:
                # Try to find the root mount point's device
                mount_point = "/"
                device_name = None
                partitions = psutil.disk_partitions()
                for p in partitions:
                    if p.mountpoint == mount_point:
                        # device path is like /dev/sda1, we need sda
                        device_name = os.path.basename(p.device).rstrip('0123456789')
                        break
                
                if device_name and device_name.startswith('nvme'): # NVMe drives are SSDs
                    return True
                if device_name:
                    rotational_path = f"/sys/block/{device_name}/queue/rotational"
                    if os.path.exists(rotational_path):
                        with open(rotational_path, 'r') as f:
                            return f.read().strip() == '0'
            except Exception as e:
                logger.debug(f"SSD detection failed on Linux: {e}")
        elif self.system == "Darwin": # macOS
            try:
                import subprocess
                # Check for non-rotational media for the root device
                result = subprocess.check_output(['diskutil', 'info', '/'], text=True)
                if "Solid State" in result or "APPLE SSD" in result: # Common indicators
                    return True
                if "Rotational Rate:" in result and "Solid State" not in result: # Explicitly check for non-rotational
                     # This part is tricky, as "Not Applicable" can mean SSD or other non-HDD
                     pass # Could add more checks here
            except Exception as e:
                logger.debug(f"SSD detection failed on macOS: {e}")
        # Windows SSD detection is more complex, often relying on PowerShell or WMI
        # For simplicity, we'll default to False if not Linux/macOS with clear indication
        return False

    def _find_mount_point(self, path_to_check): # Not directly used by _check_if_ssd anymore
        path_to_check = os.path.abspath(path_to_check)
        while not os.path.ismount(path_to_check):
            parent = os.path.dirname(path_to_check)
            if parent == path_to_check:
                break
            path_to_check = parent
        return path_to_check

    def _get_device_for_mount_point(self, mount_point): # Not directly used by _check_if_ssd anymore
        try:
            partitions = psutil.disk_partitions()
            for p in partitions:
                if p.mountpoint == mount_point:
                    return p.device
        except Exception:
            pass
        return None

    def _detect_specialized_accelerators(self):
        self.accelerators = []
        if self.is_intel_cpu: # MKL is primarily for Intel CPUs
            try:
                # Check for MKL by trying to import a common symbol or library
                # This is a heuristic and might not be foolproof
                import ctypes
                # Try common MKL library names
                lib_names = ["libmkl_rt.so", "mkl_rt.dll", "libmkl_rt.dylib", "mkl_rt"]
                mkl_found = False
                for lib_name in lib_names:
                    try:
                        ctypes.CDLL(lib_name)
                        mkl_found = True
                        break
                    except OSError:
                        continue
                if mkl_found:
                    self.accelerators.append(HardwareAccelerator.INTEL_MKL)
                    logger.info("Detected Intel MKL")
            except ImportError: # ctypes might not be available in some stripped-down envs
                pass
            except Exception as e: # Catch any other errors during MKL detection
                logger.debug(f"Error during MKL detection: {e}")

        if self.is_arm_cpu and self.has_neon: # has_neon is a broad assumption
            self.accelerators.append(HardwareAccelerator.ARM_NEON)
            logger.info("Detected ARM NEON instruction support (assumed for ARM CPU)")

    def _detect_environment(self) -> str:
        if any(key in os.environ for key in ['KUBERNETES_SERVICE_HOST', 'AWS_EXECUTION_ENV', 'AZURE_FUNCTIONS_ENVIRONMENT', 'GOOGLE_CLOUD_PROJECT', 'FUNCTION_NAME']): # Added more cloud indicators
            return "cloud"
        # More robust edge detection
        is_edge_like = (
            self.total_memory_gb < 2 or
            self.cpu_count_physical <= 1 or
            (self.is_arm_cpu and self.total_memory_gb < 4) # ARM devices often edge
        )
        if is_edge_like:
            # Check for common edge device indicators in hostname or platform
            hostname_lower = self.hostname.lower()
            processor_lower = self.processor.lower() if self.processor else ""
            if any(indicator in hostname_lower for indicator in ["raspberrypi", "jetson", "coral", "edge"]):
                return "edge"
            if any(indicator in processor_lower for indicator in ["cortex-a", "snapdragon"]): # Common edge CPU types
                 return "edge"
            return "edge" # Generic edge if low resources

        if self.total_memory_gb > 32 or self.cpu_count_physical >= 16:
            return "cloud" # Could also be a powerful server/desktop
        return "desktop"

    def _log_system_overview(self):
        logger.info("=" * 50)
        logger.info(f"System Overview: {self.hostname} ({self.system} {self.release})")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Optimization Mode: {self.optimization_mode.value}") # Use .value for str enums
        logger.info("-" * 50)
        logger.info(f"CPU: {self.processor if self.processor else 'N/A'}")
        logger.info(f"CPU Cores: {self.cpu_count_physical} physical, {self.cpu_count_logical} logical")
        logger.info(f"CPU Freq (MHz): Current={self.cpu_freq['current']:.0f}, Min={self.cpu_freq['min']:.0f}, Max={self.cpu_freq['max']:.0f}")
        logger.info(f"CPU Features: AVX={self.has_avx}, AVX2={self.has_avx2}, AVX512={self.has_avx512}, SSE4={self.has_sse4}, FMA={self.has_fma}, NEON={self.has_neon}")
        logger.info(f"Memory: {self.total_memory_gb:.2f} GB total, {self.usable_memory_gb:.2f} GB usable, {self.available_memory_gb:.2f} GB available")
        logger.info(f"Swap Memory: {self.swap_memory_gb:.2f} GB")
        logger.info(f"Disk (@{Path.home()}): {self.disk_total_gb:.2f} GB total, {self.disk_free_gb:.2f} GB free, SSD={self.is_ssd}")
        if self.accelerators:
            logger.info("-" * 50)
            logger.info("Hardware Accelerators:")
            for acc in self.accelerators:
                logger.info(f"  {acc.name}") # .name for simple Enums
        logger.info("=" * 50)

    def _serialize_config_dict(self, config_dict_input: Dict) -> Dict:
        """Recursively serialize a dictionary, converting Enums and Path objects."""
        # This function is mainly for system_info or other generic dicts.
        # Config objects should use their own to_dict() methods.
        result = {}
        for key, value in config_dict_input.items():
            if isinstance(value, Enum):
                result[key] = value.value if hasattr(value, 'value') else value.name
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, dict):
                result[key] = self._serialize_config_dict(value) # Recurse for nested dicts
            elif isinstance(value, list):
                # Process items in a list
                result[key] = [
                    self._serialize_config_dict(item) if isinstance(item, dict)
                    else (item.value if isinstance(item, Enum) and hasattr(item, 'value')
                          else (item.name if isinstance(item, Enum)
                                else (str(item) if isinstance(item, Path) else item)))
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def save_configs(self, config_id: Optional[str] = None) -> Dict[str, str]:
        if config_id is None:
            config_id = f"config_{str(uuid.uuid4())[:8]}" # More descriptive default
        configs_dir = self.config_path / config_id
        configs_dir.mkdir(parents=True, exist_ok=True)

        quant_config = self.get_optimal_quantization_config()
        batch_config = self.get_optimal_batch_processor_config()
        preproc_config = self.get_optimal_preprocessor_config()
        infer_config = self.get_optimal_inference_engine_config()
        train_config = self.get_optimal_training_engine_config()
        system_info = self.get_system_info()

        system_info_path = configs_dir / "system_info.json"
        with open(system_info_path, "w") as f:
            # Use _serialize_config_dict for system_info as it's a plain dict
            json.dump(self._serialize_config_dict(system_info), f, indent=2)

        # Config objects now have their own to_dict methods that handle serialization
        configs_to_save = [
            ("quantization_config", quant_config, configs_dir / "quantization_config.json"),
            ("batch_processor_config", batch_config, configs_dir / "batch_processor_config.json"),
            ("preprocessor_config", preproc_config, configs_dir / "preprocessor_config.json"),
            ("inference_engine_config", infer_config, configs_dir / "inference_engine_config.json"),
            ("training_engine_config", train_config, configs_dir / "training_engine_config.json")
        ]
        saved_paths = {}
        for name, config_obj, path in configs_to_save:
            try:
                if hasattr(config_obj, 'to_dict') and callable(getattr(config_obj, 'to_dict')):
                    serialized_config = config_obj.to_dict()
                elif is_dataclass(config_obj): # Fallback for other dataclasses
                    serialized_config = self._serialize_config_dict(asdict(config_obj))
                else: # Should not happen for main configs
                    serialized_config = self._serialize_config_dict(config_obj.__dict__ if hasattr(config_obj, '__dict__') else {})

                with open(path, "w") as f:
                    json.dump(serialized_config, f, indent=2)
                saved_paths[name] = str(path)
                logger.info(f"Saved {name} to {path}")
            except Exception as e:
                logger.error(f"Failed to save {name} to {path}: {e}", exc_info=self.debug_mode)


        master_config = {
            "config_id": config_id,
            "optimization_mode": self.optimization_mode.value, # Use .value
            "system_info_path": str(system_info_path),
            "quantization_config_path": saved_paths.get("quantization_config", ""),
            "batch_processor_config_path": saved_paths.get("batch_processor_config", ""),
            "preprocessor_config_path": saved_paths.get("preprocessor_config", ""),
            "inference_engine_config_path": saved_paths.get("inference_engine_config", ""),
            "training_engine_config_path": saved_paths.get("training_engine_config", ""),
            "checkpoint_path": str(self.checkpoint_path),
            "model_registry_path": str(self.model_registry_path),
            "creation_timestamp": datetime.datetime.now().isoformat()
        }
        master_config_path = configs_dir / "master_config.json"
        with open(master_config_path, "w") as f:
            json.dump(master_config, f, indent=2)
        logger.info(f"All configurations saved to {configs_dir}")
        return master_config

    def load_configs(self, config_id: str) -> Dict[str, Any]:
        """
        Load previously saved configurations as dictionaries.
        For full object reconstruction, use from_dict methods of config classes
        on the loaded dictionaries.
        """
        configs_dir = self.config_path / config_id
        if not configs_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {configs_dir}")
        master_config_path = configs_dir / "master_config.json"
        if not master_config_path.exists():
            raise FileNotFoundError(f"Master configuration file not found: {master_config_path}")
        
        with open(master_config_path, "r") as f:
            master_config = json.load(f)
        
        loaded_configs = {"master_config": master_config}
        
        # Load system info
        system_info_path_str = master_config.get("system_info_path")
        if system_info_path_str and Path(system_info_path_str).exists():
            with open(system_info_path_str, "r") as f:
                loaded_configs["system_info"] = json.load(f)
        elif (configs_dir / "system_info.json").exists(): # Fallback
             with open(configs_dir / "system_info.json", "r") as f:
                loaded_configs["system_info"] = json.load(f)


        config_file_keys_to_paths = {
            "quantization_config": master_config.get("quantization_config_path"),
            "batch_processor_config": master_config.get("batch_processor_config_path"),
            "preprocessor_config": master_config.get("preprocessor_config_path"),
            "inference_engine_config": master_config.get("inference_engine_config_path"),
            "training_engine_config": master_config.get("training_engine_config_path"),
        }
        default_filenames = {
            "quantization_config": "quantization_config.json",
            "batch_processor_config": "batch_processor_config.json",
            "preprocessor_config": "preprocessor_config.json",
            "inference_engine_config": "inference_engine_config.json",
            "training_engine_config": "training_engine_config.json",
        }

        for name, path_str in config_file_keys_to_paths.items():
            actual_path = None
            if path_str and Path(path_str).exists():
                actual_path = Path(path_str)
            else:
                fallback_path = configs_dir / default_filenames[name]
                if fallback_path.exists():
                    actual_path = fallback_path
            
            if actual_path:
                try:
                    with open(actual_path, "r") as f:
                        # Here you could use ConfigClass.from_dict(json.load(f))
                        # For now, returning dicts as per original behavior.
                        loaded_configs[name] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load {name} from {actual_path}: {e}", exc_info=self.debug_mode)
            elif path_str: # Path was specified in master but not found
                 logger.warning(f"Configuration file for {name} not found at specified path: {path_str}")
        
        return loaded_configs

    def create_configs_for_all_modes(self) -> Dict[str, Dict[str, str]]:
        base_config_id = f"all_modes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_modes = list(OptimizationMode) # Use imported OptimizationMode
        original_mode = self.optimization_mode
        configs_summary = {}
        for mode_enum_instance in all_modes:
            self.optimization_mode = mode_enum_instance # Assign enum instance
            mode_config_id = f"{base_config_id}_{mode_enum_instance.value}" # Use .value for filename
            master_config = self.save_configs(mode_config_id)
            configs_summary[mode_enum_instance.value] = master_config # Use .value for key
            logger.info(f"Created configurations for {mode_enum_instance.value} optimization mode")
        self.optimization_mode = original_mode
        return configs_summary

    def auto_tune_configs(self, workload_sample: Any = None) -> Dict[str, Any]:
        if not self.auto_tune:
            logger.info("Auto-tuning is disabled, using static optimization")
            return self.save_configs() # Returns master_config dict
        logger.info("Starting auto-tuning process (placeholder)...")
        # Placeholder for actual auto-tuning logic
        cpu_load = psutil.cpu_percent(interval=1) / 100.0
        memory_load = psutil.virtual_memory().percent / 100.0
        logger.info(f"Current system load: CPU {cpu_load:.2%}, Memory {memory_load:.2%}")
        # Actual tuning logic would adjust parameters before saving
        tuned_master_config = self.save_configs(f"auto_tuned_{str(uuid.uuid4())[:4]}")
        logger.info("Auto-tuning complete (placeholder implementation)")
        return tuned_master_config # Return the master_config dict

    def _get_optimal_quantization_type(self) -> str: # Returns string value of enum
        if self.total_memory_gb < 4 or self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            return QuantizationType.INT8.value
        if self.optimization_mode == OptimizationMode.PERFORMANCE:
            # FLOAT16 might not always be faster on all CPUs without specific hardware support.
            # INT8 is often a good balance for performance on CPU.
            # Consider if AVX512_VNNI or similar is present for INT8 perf.
            return QuantizationType.INT8.value if self.has_avx512 else QuantizationType.FLOAT16.value
        elif self.optimization_mode == OptimizationMode.FULL_UTILIZATION:
            return QuantizationType.MIXED.value
        return QuantizationType.INT8.value

    def _apply_optimization_mode_factors(self) -> Dict[str, float]:
        # Keys are OptimizationMode enum instances
        mode_factors_map = {
            OptimizationMode.BALANCED: {"cpu": 0.75, "memory": 0.7, "batch_size": 0.8, "cache": 0.7, "workers": 0.75},
            OptimizationMode.CONSERVATIVE: {"cpu": 0.5, "memory": 0.5, "batch_size": 0.5, "cache": 0.5, "workers": 0.5},
            OptimizationMode.PERFORMANCE: {"cpu": 0.9, "memory": 0.8, "batch_size": 1.0, "cache": 0.9, "workers": 0.9},
            OptimizationMode.FULL_UTILIZATION: {"cpu": 1.0, "memory": 0.95, "batch_size": 1.2, "cache": 1.0, "workers": 1.0},
            OptimizationMode.MEMORY_SAVING: {"cpu": 0.7, "memory": 0.4, "batch_size": 0.6, "cache": 0.4, "workers": 0.7}
        }
        factors = mode_factors_map.get(self.optimization_mode, mode_factors_map[OptimizationMode.BALANCED]).copy()
        
        if self.environment == "edge":
            factors["memory"] *= 0.8
            factors["batch_size"] *= 0.7 # Smaller batches for edge
            factors["workers"] *= 0.7
        elif self.environment == "cloud":
            factors["memory"] *= 1.1 # Can use more memory
            factors["batch_size"] *= 1.1
        
        if self.workload_type == "inference":
            factors["batch_size"] *= 1.2
            factors["cache"] *= 0.9
        elif self.workload_type == "training":
            factors["memory"] *= 1.1
            factors["cache"] *= 1.1
            factors["workers"] *= 1.1 # Training can benefit from more workers for data loading

        if self.power_efficiency:
            factors["cpu"] *= 0.7
            factors["memory"] *= 0.8
            factors["workers"] *= 0.7

        for key in factors: # Clamp factors
            factors[key] = min(max(factors[key], 0.1), 1.5) # Allow slightly > 1 for full_utilization cases
        return factors

    def get_optimal_quantization_config(self) -> QuantizationConfig:
        quant_type_str = self._get_optimal_quantization_type() # This returns string value
        
        # Determine quantization mode based on optimization mode
        if self.optimization_mode == OptimizationMode.PERFORMANCE:
            quant_mode_enum = QuantizationMode.STATIC
        elif self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            quant_mode_enum = QuantizationMode.DYNAMIC
        else: # BALANCED, FULL_UTILIZATION, CONSERVATIVE
            quant_mode_enum = QuantizationMode.DYNAMIC_PER_BATCH
        
        per_channel_val = (self.optimization_mode == OptimizationMode.PERFORMANCE or
                           self.optimization_mode == OptimizationMode.FULL_UTILIZATION)
        symmetric_val = self.optimization_mode != OptimizationMode.PERFORMANCE
        
        base_cache_size = (512 if self.total_memory_gb > 16 else
                           256 if self.total_memory_gb > 8 else
                           128 if self.total_memory_gb > 4 else 64)
        cache_size_val = base_cache_size
        if self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            cache_size_val //= 2
        
        calibration_samples_val = (200 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else
                                   100 if self.optimization_mode == OptimizationMode.BALANCED else 50)
        
        # Default bits based on type, can be overridden by QuantizationConfig defaults
        weight_bits_val, activation_bits_val, bias_bits_val = 8, 8, 32
        if quant_type_str == QuantizationType.MIXED.value:
            weight_bits_val, activation_bits_val = 8, 16 # Example for mixed
        elif quant_type_str == QuantizationType.FLOAT16.value:
            weight_bits_val, activation_bits_val, bias_bits_val = 16, 16, 16 # fp16 bias
        elif quant_type_str == QuantizationType.INT16.value:
            weight_bits_val, activation_bits_val, bias_bits_val = 16, 16, 32

        if self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            bias_bits_val = 16 if bias_bits_val == 32 else bias_bits_val


        return QuantizationConfig(
            quantization_type=quant_type_str, # Pass string value, __post_init__ handles it
            quantization_mode=quant_mode_enum, # Pass enum, __post_init__ handles it
            per_channel=per_channel_val,
            symmetric=symmetric_val,
            enable_cache=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            cache_size=cache_size_val,
            calibration_samples=calibration_samples_val,
            calibration_method="percentile" if self.optimization_mode == OptimizationMode.PERFORMANCE else "minmax",
            percentile=99.99 if self.optimization_mode == OptimizationMode.PERFORMANCE else 99.9,
            quantize_weights_only=self.optimization_mode == OptimizationMode.CONSERVATIVE,
            quantize_activations=self.optimization_mode != OptimizationMode.CONSERVATIVE,
            weight_bits=weight_bits_val,
            activation_bits=activation_bits_val,
            quantize_bias=self.optimization_mode != OptimizationMode.PERFORMANCE,
            bias_bits=bias_bits_val,
            enable_mixed_precision=quant_type_str == QuantizationType.MIXED.value,
            optimize_for=(self.optimization_mode.value if self.optimization_mode != OptimizationMode.CONSERVATIVE else OptimizationMode.BALANCED.value),
            enable_requantization=self.optimization_mode == OptimizationMode.FULL_UTILIZATION,
            use_percentile=self.optimization_mode == OptimizationMode.PERFORMANCE, # Aligns with calibration_method
            error_on_nan=self.debug_mode, # More strict in debug mode
            error_on_inf=self.debug_mode,
            outlier_threshold=3.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else None,
            num_bits=weight_bits_val, # Default num_bits to weight_bits
            optimize_memory=self.optimization_mode != OptimizationMode.PERFORMANCE,
            buffer_size=64 if self.optimization_mode == OptimizationMode.MEMORY_SAVING else 0
        )

    def get_optimal_batch_processor_config(self) -> BatchProcessorConfig:
        factors = self._apply_optimization_mode_factors()
        
        # Base batch sizes considering memory and CPU
        if self.total_memory_gb > 16 and self.cpu_count_logical >= 8:
            base_max_bs = 256
        elif self.total_memory_gb > 8 and self.cpu_count_logical >= 4:
            base_max_bs = 128
        elif self.total_memory_gb > 4:
            base_max_bs = 64
        else:
            base_max_bs = 32

        max_bs = max(int(base_max_bs * factors["batch_size"]), 8) # Ensure at least 8
        initial_bs = max(max_bs // (2 if self.optimization_mode != OptimizationMode.PERFORMANCE else 1), 1)
        min_bs = max(initial_bs // 4, 1)
        
        num_workers_val = max(int(self.cpu_count_logical * factors["workers"]), 1)
        # Cap workers based on physical cores for some modes to avoid over-subscription
        if self.optimization_mode in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING]:
            num_workers_val = min(num_workers_val, self.cpu_count_physical if self.cpu_count_physical else 1)

        max_q_size = max(num_workers_val * 8, max_bs * 4) # Generous queue size

        processing_strat = BatchProcessingStrategy.ADAPTIVE
        if self.optimization_mode == OptimizationMode.CONSERVATIVE:
            processing_strat = BatchProcessingStrategy.FIXED
        
        return BatchProcessorConfig(
            initial_batch_size=initial_bs,
            min_batch_size=min_bs,
            max_batch_size=max_bs,
            max_queue_size=max_q_size,
            batch_timeout=0.01 if self.optimization_mode == OptimizationMode.PERFORMANCE else 0.05,
            num_workers=num_workers_val,
            max_workers=num_workers_val, # Align num_workers and max_workers initially
            adaptive_batching=self.optimization_mode not in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING],
            enable_adaptive_batching=self.optimization_mode not in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING], # Redundant with adaptive_batching
            processing_strategy=processing_strat, # Pass Enum
            enable_priority_queue=self.workload_type != "training", # Priority queue more for inference
            enable_memory_optimization=self.optimization_mode != OptimizationMode.PERFORMANCE,
            enable_monitoring=self.debug_mode or self.optimization_mode == OptimizationMode.FULL_UTILIZATION,
            debug_mode=self.debug_mode,
            # New fields from BatchProcessorConfig
            max_batch_memory_mb= (self.usable_memory_gb * 1024 * 0.05) if self.optimization_mode == OptimizationMode.MEMORY_SAVING else None # 5% of usable for mem saving
        )

    def get_optimal_preprocessor_config(self) -> PreprocessorConfig:
        norm_type_enum = NormalizationType.STANDARD
        if self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            norm_type_enum = NormalizationType.MINMAX # MinMax can be slightly lighter
        elif self.optimization_mode == OptimizationMode.PERFORMANCE:
            norm_type_enum = NormalizationType.NONE # Fastest if data is already sane

        return PreprocessorConfig(
            normalization=norm_type_enum, # Pass Enum
            handle_nan=True,
            nan_strategy="mean",
            detect_outliers=self.optimization_mode not in [OptimizationMode.PERFORMANCE, OptimizationMode.MEMORY_SAVING],
            outlier_handling="clip",
            parallel_processing=self.cpu_count_logical > 1 and self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            n_jobs=max(1, int(self.cpu_count_logical * 0.5)) if self.optimization_mode != OptimizationMode.MEMORY_SAVING else 1,
            cache_preprocessing=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            debug_mode=self.debug_mode,
            # New fields
            enable_input_validation=self.debug_mode or self.optimization_mode == OptimizationMode.CONSERVATIVE
        )

    def get_optimal_inference_engine_config(self) -> InferenceEngineConfig:
        factors = self._apply_optimization_mode_factors()
        
        model_prec = "fp16" if self.optimization_mode == OptimizationMode.MEMORY_SAVING and self.has_avx2 else "fp32" # fp16 needs some support
        enable_fp16 = model_prec == "fp16"

        # Use calculated batch processor config for consistent batch sizes
        # This ensures InferenceEngineConfig is aligned with how batches might be prepared
        # Note: InferenceEngineConfig also has its own batch size params, which can be a bit redundant.
        # We'll use the batch_processor_config's values to set the InferenceEngine's batch params.
        batch_proc_conf = self.get_optimal_batch_processor_config()

        thread_cnt = max(int(self.cpu_count_logical * factors["cpu"]), 1)
        if self.optimization_mode == OptimizationMode.CONSERVATIVE:
             thread_cnt = min(thread_cnt, self.cpu_count_physical if self.cpu_count_physical else 1, 4) # Cap for conservative

        model_cache_sz = max(int(10 * factors["cache"]), 1)
        if self.environment == "edge":
            model_cache_sz = min(model_cache_sz, 2)

        mem_limit_factor = 0.8 if self.optimization_mode == OptimizationMode.PERFORMANCE else \
                           0.6 if self.optimization_mode == OptimizationMode.BALANCED else \
                           0.4 # Conservative or memory saving
        mem_limit_gb_val = self.usable_memory_gb * mem_limit_factor if self.optimization_mode != OptimizationMode.FULL_UTILIZATION else None
        
        enable_intel_opt = self.is_intel_cpu and (self.has_avx2 or self.has_avx512 or HardwareAccelerator.INTEL_MKL in self.accelerators)
        
        quant_conf = self.get_optimal_quantization_config() if self.optimization_mode == OptimizationMode.MEMORY_SAVING else None
        preproc_conf = self.get_optimal_preprocessor_config() if self.needs_feature_scaling else None # Assuming needs_feature_scaling is set elsewhere

        return InferenceEngineConfig(
            enable_intel_optimization=enable_intel_opt,
            enable_batching=True, # Default, actual batching managed by BatchProcessorConfig if used
            enable_quantization=quant_conf is not None,
            model_cache_size=model_cache_sz,
            model_precision=model_prec,
            max_batch_size=batch_proc_conf.max_batch_size, # From BatchProcessorConfig
            timeout_ms=50 if self.optimization_mode == OptimizationMode.PERFORMANCE else (200 if self.environment == "edge" else 100),
            enable_jit=True,
            enable_onnx=self.onnx_available,
            onnx_opset=15, # From new config default
            enable_tensorrt=False, # CPU-only
            runtime_optimization=self.optimization_mode != OptimizationMode.CONSERVATIVE,
            thread_count=thread_cnt,
            num_threads=thread_cnt, # Align
            warmup=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            warmup_iterations=3 if self.environment == "edge" else (10 if self.optimization_mode == OptimizationMode.PERFORMANCE else 5),
            profiling=self.debug_mode and self.optimization_mode == OptimizationMode.FULL_UTILIZATION,
            batching_strategy="dynamic", # Default, can be influenced by BatchProcessorConfig
            debug_mode=self.debug_mode,
            memory_growth=self.optimization_mode != OptimizationMode.MEMORY_SAVING, # Allow growth unless strictly saving memory
            set_cpu_affinity=self.optimization_mode == OptimizationMode.PERFORMANCE and self.system != "Darwin",
            enable_model_quantization=quant_conf is not None, # Redundant with enable_quantization
            quantization_dtype=quant_conf.quantization_type.value if quant_conf else QuantizationType.INT8.value,
            quantization_config=quant_conf,
            enable_request_deduplication=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            max_cache_entries=max(int(1000 * factors["cache"]), 50) if self.total_memory_gb > 4 else 20,
            cache_ttl_seconds=int(600 * factors["cache"]) if self.optimization_mode == OptimizationMode.PERFORMANCE else int(120 * factors["cache"]),
            enable_monitoring=self.debug_mode or self.optimization_mode == OptimizationMode.FULL_UTILIZATION,
            monitoring_interval=5.0 if self.environment == "edge" else 10.0,
            throttle_on_high_cpu=self.optimization_mode != OptimizationMode.PERFORMANCE,
            cpu_threshold_percent=85.0 if self.optimization_mode == OptimizationMode.PERFORMANCE else 75.0,
            memory_limit_gb=mem_limit_gb_val,
            batch_timeout=batch_proc_conf.batch_timeout, # From BatchProcessorConfig
            max_concurrent_requests=thread_cnt * 2, # Allow some queuing
            initial_batch_size=batch_proc_conf.initial_batch_size, # From BatchProcessorConfig
            min_batch_size=batch_proc_conf.min_batch_size, # From BatchProcessorConfig
            enable_adaptive_batching=batch_proc_conf.adaptive_batching, # From BatchProcessorConfig
            enable_memory_optimization=self.optimization_mode != OptimizationMode.PERFORMANCE,
            enable_feature_scaling=preproc_conf is not None,
            optimization_mode=self.optimization_mode, # Pass the enum instance
            enable_fp16_optimization=enable_fp16,
            enable_compiler_optimization=self.optimization_mode != OptimizationMode.CONSERVATIVE,
            preprocessor_config=preproc_conf,
            batch_processor_config=batch_proc_conf, # Assign the generated batch processor config
            threadpoolctl_available=self.threadpoolctl_available,
            onnx_available=self.onnx_available,
            treelite_available=self.treelite_available
        )

    def get_optimal_training_engine_config(self) -> MLTrainingEngineConfig:
        factors = self._apply_optimization_mode_factors()

        enable_intel_opt = self.is_intel_cpu and \
                           (self.has_avx2 or self.has_avx512 or HardwareAccelerator.INTEL_MKL in self.accelerators)

        n_jobs_val = -1 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else \
                     max(1, int(self.cpu_count_logical * factors["cpu"]))
        if self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            n_jobs_val = max(1, int((self.cpu_count_physical or 1) * 0.5))

        cv_folds_val = 3 if self.optimization_mode == OptimizationMode.MEMORY_SAVING or self.workload_type == "inference" else 5
        
        opt_strategy_enum = TrainingOptimizationStrategy.HYPERX
        if self.optimization_mode == OptimizationMode.CONSERVATIVE:
            opt_strategy_enum = TrainingOptimizationStrategy.RANDOM_SEARCH

        opt_iters = 50
        if self.optimization_mode == OptimizationMode.PERFORMANCE: opt_iters = 75
        elif self.optimization_mode == OptimizationMode.MEMORY_SAVING: opt_iters = 25
        elif self.optimization_mode == OptimizationMode.CONSERVATIVE: opt_iters = 15

        # Default to a common metric. The user can change this on the config object
        # based on their specific MLTrainingEngineConfig.task_type.
        model_sel_criteria_enum = ModelSelectionCriteria.ACCURACY
        opt_metric_val = ModelSelectionCriteria.ACCURACY.value # Pass string value

        early_stop_rounds = 10
        if self.optimization_mode == OptimizationMode.MEMORY_SAVING: early_stop_rounds = 5

        feat_sel = self.optimization_mode not in [OptimizationMode.PERFORMANCE, OptimizationMode.CONSERVATIVE]

        preproc_conf_train = self.get_optimal_preprocessor_config()
        batch_proc_conf_train = self.get_optimal_batch_processor_config()
        batch_proc_conf_train.initial_batch_size = max(batch_proc_conf_train.initial_batch_size, 32 if self.total_memory_gb > 4 else 16)
        batch_proc_conf_train.max_batch_size = max(batch_proc_conf_train.max_batch_size, 64 if self.total_memory_gb > 4 else 32)
        if self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            batch_proc_conf_train.initial_batch_size = min(batch_proc_conf_train.initial_batch_size, 32)
            batch_proc_conf_train.max_batch_size = min(batch_proc_conf_train.max_batch_size, 64)

        quant_conf_train = self.get_optimal_quantization_config() if self.optimization_mode == OptimizationMode.MEMORY_SAVING else None
        
        expl_conf = ExplainabilityConfig(
            enable_explainability=self.optimization_mode not in [OptimizationMode.MEMORY_SAVING, OptimizationMode.PERFORMANCE],
            methods=["shap", "feature_importance"] if self.optimization_mode != OptimizationMode.CONSERVATIVE else ["feature_importance"],
            shap_samples=50 if self.optimization_mode == OptimizationMode.CONSERVATIVE else 100,
            generate_plots=self.optimization_mode != OptimizationMode.MEMORY_SAVING
        )
        mon_conf = MonitoringConfig(
            enable_monitoring=self.debug_mode or self.environment == "cloud",
            drift_detection=self.environment == "cloud",
            performance_tracking=True,
            alert_on_drift=False
        )

        exp_platform_val = "mlflow" if self.total_memory_gb >= 4 and self.disk_free_gb > 1 else "csv"
        mem_opt_train = self.optimization_mode == OptimizationMode.MEMORY_SAVING or self.total_memory_gb < 8

        enable_ens = self.optimization_mode not in [OptimizationMode.MEMORY_SAVING, OptimizationMode.CONSERVATIVE] and self.total_memory_gb >= 8
        ens_method = "stacking" if self.total_memory_gb >= 16 and self.optimization_mode == OptimizationMode.FULL_UTILIZATION else "voting"
        ens_size = 3 if self.total_memory_gb < 16 or self.optimization_mode != OptimizationMode.FULL_UTILIZATION else 5

        export_fmts = ["sklearn"]
        if self.onnx_available: export_fmts.append("onnx")
        if self.optimization_mode == OptimizationMode.MEMORY_SAVING and self.onnx_available:
            export_fmts = ["onnx"]
        elif self.optimization_mode == OptimizationMode.MEMORY_SAVING:
             export_fmts = []

        sec_config_dict = {}
        if self.optimization_mode != OptimizationMode.PERFORMANCE:
            sec_config_dict = {
                "enable_input_sanitization": True,
                "enable_output_filtering": False
            }
        
        meta_dict = {
            "device_info": {
                "cpu_count_physical": self.cpu_count_physical,
                "cpu_count_logical": self.cpu_count_logical,
                "total_memory_gb": self.total_memory_gb,
                "is_intel_cpu": self.is_intel_cpu,
                "has_gpu": self.has_gpu
            },
            "optimizer_settings": {
                 "optimization_mode": self.optimization_mode.value,
                 "environment": self.environment,
                 "workload_type": self.workload_type,
                 "auto_tune_optimizer": self.auto_tune
            }
        }

        # MLTrainingEngineConfig defaults task_type to CLASSIFICATION.
        # The optimization_metric and model_selection_criteria set here
        # are general defaults.
        return MLTrainingEngineConfig(
            # task_type will use its default (CLASSIFICATION) or what user provides
            random_state=42,
            n_jobs=n_jobs_val,
            verbose=1 if not self.debug_mode and self.optimization_mode != OptimizationMode.PERFORMANCE else (2 if self.debug_mode else 0),
            cv_folds=cv_folds_val,
            stratify=True, # Default, relevant for classification
            optimization_strategy=opt_strategy_enum,
            optimization_iterations=opt_iters,
            optimization_metric=opt_metric_val, # General default metric
            early_stopping=self.optimization_mode != OptimizationMode.FULL_UTILIZATION,
            early_stopping_rounds=early_stop_rounds,
            feature_selection=feat_sel,
            preprocessing_config=preproc_conf_train,
            batch_processing_config=batch_proc_conf_train,
            inference_config=None,
            quantization_config=quant_conf_train,
            explainability_config=expl_conf,
            monitoring_config=mon_conf,
            model_path=str(self.model_registry_path / "training_models"),
            checkpoint_path=str(self.checkpoint_path / "training_checkpoints"),
            experiment_tracking_platform=exp_platform_val,
            use_intel_optimization=enable_intel_opt,
            use_gpu=False,
            gpu_memory_fraction=0.0,
            memory_optimization=mem_opt_train,
            enable_distributed=False,
            checkpointing=self.optimization_mode != OptimizationMode.PERFORMANCE,
            checkpoint_interval=5 if self.optimization_mode == OptimizationMode.CONSERVATIVE else 10,
            enable_pruning=self.optimization_mode == OptimizationMode.MEMORY_SAVING,
            auto_ml=AutoMLMode.BASIC if self.auto_tune and self.optimization_mode != OptimizationMode.CONSERVATIVE else AutoMLMode.DISABLED,
            ensemble_models=enable_ens,
            ensemble_method=ens_method,
            ensemble_size=ens_size,
            hyperparameter_tuning_cv=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            model_selection_criteria=model_sel_criteria_enum, # General default criteria
            enable_quantization=quant_conf_train is not None,
            enable_model_compression=self.optimization_mode == OptimizationMode.MEMORY_SAVING,
            compute_permutation_importance=expl_conf.enable_explainability and "feature_importance" in expl_conf.methods,
            generate_prediction_explanations=expl_conf.enable_explainability and self.optimization_mode == OptimizationMode.FULL_UTILIZATION,
            export_formats=export_fmts,
            log_level="DEBUG" if self.debug_mode else "INFO",
            debug_mode=self.debug_mode,
            enable_security=self.optimization_mode != OptimizationMode.PERFORMANCE,
            security_config=sec_config_dict,
            metadata=meta_dict
        )



    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current system."""
        # This method returns a dictionary, which is fine.
        # _serialize_config_dict will handle enums within this dict if any.
        return {
            "system": self.system, "release": self.release, "machine": self.machine,
            "processor": self.processor, "hostname": self.hostname,
            "python_version": self.python_version,
            "cpu_count_physical": self.cpu_count_physical,
            "cpu_count_logical": self.cpu_count_logical,
            "cpu_freq_mhz": self.cpu_freq,
            "cpu_features": {
                "avx": self.has_avx, "avx2": self.has_avx2, "avx512": self.has_avx512,
                "sse4": self.has_sse4, "fma": self.has_fma, "neon": self.has_neon
            },
            "is_intel_cpu": self.is_intel_cpu, "is_amd_cpu": self.is_amd_cpu, "is_arm_cpu": self.is_arm_cpu,
            "total_memory_gb": self.total_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "usable_memory_gb": self.usable_memory_gb,
            "swap_memory_gb": self.swap_memory_gb,
            "disk_total_gb": self.disk_total_gb, "disk_free_gb": self.disk_free_gb, "is_ssd": self.is_ssd,
            "accelerators": [acc.name for acc in self.accelerators], # .name for simple Enum
            "detected_environment": self.environment,
            "optimizer_settings": { # Added a sub-dict for clarity
                "optimization_mode": self.optimization_mode.value, # Use .value for str Enum
                "workload_type": self.workload_type,
                "power_efficiency": self.power_efficiency,
                "auto_tune": self.auto_tune,
                "debug_mode": self.debug_mode
            },
            "library_availability": {
                "onnx_available": self.onnx_available,
                "threadpoolctl_available": self.threadpoolctl_available,
                "treelite_available": self.treelite_available # Assuming this is set
            }
        }
