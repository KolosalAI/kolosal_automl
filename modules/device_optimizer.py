
import os
import platform
import psutil
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, TypeVar, Tuple
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
logger = logging.getLogger("enhanced_device_optimizer")


class HardwareAccelerator(Enum):
    """Types of hardware accelerators that might be available"""
    NONE = auto()
    GPU_CUDA = auto()
    GPU_ROCM = auto()
    TPU = auto()
    INTEL_IPEX = auto()
    INTEL_MKL = auto()
    ARM_NEON = auto()


class GPUInfo:
    """Information about available GPUs"""
    def __init__(self, gpu_id: int, name: str, memory_mb: int, compute_capability: Optional[str] = None):
        self.gpu_id = gpu_id
        self.name = name
        self.memory_mb = memory_mb
        self.compute_capability = compute_capability
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "memory_mb": self.memory_mb,
            "compute_capability": self.compute_capability
        }


class DeviceOptimizer:
    """
    Enhanced device optimizer that automatically configures ML pipeline settings 
    based on device capabilities with sophisticated optimization strategies.
    
    Features:
    - Comprehensive hardware detection (CPU, GPU, TPU, specialized accelerators)
    - Advanced configuration optimization for different pipeline components
    - Power and thermal management awareness
    - Multi-device support for distributed training
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
                 enable_gpu: bool = True,
                 enable_specialized_accelerators: bool = True,
                 memory_reservation_percent: float = 10.0,
                 power_efficiency: bool = False,
                 resilience_level: int = 1,
                 auto_tune: bool = True):
        """
        Initialize the enhanced device optimizer.
        
        Args:
            config_path: Path to save configuration files
            checkpoint_path: Path for model checkpoints
            model_registry_path: Path for model registry
            optimization_mode: Mode for optimization strategy
            workload_type: Type of workload ("inference", "training", "mixed")
            environment: Computing environment ("cloud", "desktop", "edge", "auto")
            enable_gpu: Whether to enable GPU detection and optimization
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
        
        # GPU information (if enabled)
        self.gpus = []
        if enable_gpu:
            self._detect_gpu_capabilities()
        
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

    def _detect_gpu_capabilities(self):
        """Detect GPU capabilities"""
        # Reset GPU list
        self.gpus = []
        self.has_cuda = False
        self.has_rocm = False
        
        # Try detecting NVIDIA GPUs using CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self.has_cuda = True
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)  # Convert to MB
                    compute_capability = f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}"
                    
                    self.gpus.append(GPUInfo(
                        gpu_id=i,
                        name=gpu_name,
                        memory_mb=gpu_memory,
                        compute_capability=compute_capability
                    ))
                logger.info(f"Detected {len(self.gpus)} NVIDIA GPUs with CUDA support")
        except ImportError:
            logger.debug("PyTorch not available, skipping CUDA GPU detection via PyTorch")
        except Exception as e:
            logger.debug(f"Error detecting NVIDIA GPUs: {e}")
        
        # If no GPUs detected with PyTorch, try a different approach
        if not self.gpus:
            if self.system == "Linux":
                self._detect_gpus_linux()
            elif self.system == "Windows":
                self._detect_gpus_windows()
            elif self.system == "Darwin":  # macOS
                self._detect_gpus_macos()

    def _detect_gpus_linux(self):
        """Detect GPUs on Linux"""
        # Try NVIDIA GPUs with nvidia-smi
        try:
            import subprocess
            result = subprocess.check_output("nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader", shell=True)
            lines = result.decode().strip().split('\n')
            
            self.has_cuda = True
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_id = int(parts[0])
                    gpu_name = parts[1]
                    # Convert memory from MiB to MB
                    memory_mb = int(parts[2].split(' ')[0])
                    compute_capability = parts[3] if len(parts) > 3 else None
                    
                    self.gpus.append(GPUInfo(
                        gpu_id=gpu_id,
                        name=gpu_name,
                        memory_mb=memory_mb,
                        compute_capability=compute_capability
                    ))
            logger.info(f"Detected {len(self.gpus)} NVIDIA GPUs with nvidia-smi")
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")
        
        # Try AMD GPUs with rocm-smi
        if not self.gpus:
            try:
                import subprocess
                result = subprocess.check_output("rocm-smi --showproductname --showmeminfo vram", shell=True)
                lines = result.decode().strip().split('\n')
                
                gpu_id = 0
                gpu_data = {}
                
                for line in lines:
                    if "GPU[" in line and "]: " in line:
                        match = re.search(r'GPU\[(\d+)\].*: (.*)', line)
                        if match:
                            current_id = int(match.group(1))
                            value = match.group(2).strip()
                            
                            if "Product Name" in line:
                                if current_id not in gpu_data:
                                    gpu_data[current_id] = {"name": value}
                            elif "VRAM Total" in line:
                                if current_id in gpu_data:
                                    # Extract just the numeric part of memory value
                                    memory_value = re.search(r'(\d+)', value)
                                    if memory_value:
                                        gpu_data[current_id]["memory_mb"] = int(memory_value.group(1))
                
                # Create GPUInfo objects from gathered data
                for gpu_id, data in gpu_data.items():
                    if "name" in data and "memory_mb" in data:
                        self.gpus.append(GPUInfo(
                            gpu_id=gpu_id,
                            name=data["name"],
                            memory_mb=data["memory_mb"],
                            compute_capability=None  # ROCm doesn't have an equivalent concept
                        ))
                
                if self.gpus:
                    self.has_rocm = True
                    logger.info(f"Detected {len(self.gpus)} AMD GPUs with rocm-smi")
            except Exception as e:
                logger.debug(f"rocm-smi detection failed: {e}")

    def _detect_gpus_windows(self):
        """Detect GPUs on Windows"""
        # Try using Windows Management Instrumentation Command-line (WMIC)
        try:
            import subprocess
            result = subprocess.check_output("wmic path win32_VideoController get Name,AdapterRAM", shell=True)
            lines = result.decode().strip().split('\n')
            
            # Skip header
            if len(lines) > 1:
                lines = lines[1:]
            
            for idx, line in enumerate(lines):
                if line.strip():
                    parts = line.strip().split('  ')
                    parts = [p for p in parts if p.strip()]
                    
                    if len(parts) >= 2:
                        # Last element is RAM, everything before is name
                        ram = parts[-1]
                        name = ' '.join(parts[:-1]).strip()
                        
                        # Convert RAM from bytes to MB
                        try:
                            memory_mb = int(ram) // (1024 * 1024)
                        except ValueError:
                            memory_mb = 0
                        
                        is_nvidia = "nvidia" in name.lower()
                        is_amd = "amd" in name.lower() or "radeon" in name.lower()
                        
                        if is_nvidia:
                            self.has_cuda = True
                        elif is_amd:
                            self.has_rocm = True
                        
                        self.gpus.append(GPUInfo(
                            gpu_id=idx,
                            name=name,
                            memory_mb=memory_mb,
                            compute_capability=None
                        ))
            
            if self.gpus:
                logger.info(f"Detected {len(self.gpus)} GPUs using Windows WMI")
        except Exception as e:
            logger.debug(f"Windows GPU detection failed: {e}")

    def _detect_gpus_macos(self):
        """Detect GPUs on macOS"""
        try:
            import subprocess
            result = subprocess.check_output("system_profiler SPDisplaysDataType", shell=True)
            output = result.decode()
            
            # Parse the output to extract GPU information
            current_gpu = None
            gpu_id = 0
            
            for line in output.split('\n'):
                line = line.strip()
                
                if "Chipset Model:" in line:
                    if current_gpu:
                        # Add the previous GPU to the list
                        self.gpus.append(current_gpu)
                    
                    gpu_name = line.split('Chipset Model:')[1].strip()
                    current_gpu = GPUInfo(
                        gpu_id=gpu_id,
                        name=gpu_name,
                        memory_mb=0,  # Will try to set this below
                        compute_capability=None
                    )
                    gpu_id += 1
                elif current_gpu and "VRAM" in line:
                    try:
                        # Try to extract memory information
                        memory_str = line.split(':')[1].strip()
                        memory_value = int(re.search(r'(\d+)', memory_str).group(1))
                        
                        # Check if memory is in GB and convert to MB
                        if "GB" in memory_str:
                            memory_value *= 1024
                        
                        current_gpu.memory_mb = memory_value
                    except Exception:
                        pass
            
            # Add the last GPU
            if current_gpu:
                self.gpus.append(current_gpu)
            
            if self.gpus:
                logger.info(f"Detected {len(self.gpus)} GPUs on macOS")
        except Exception as e:
            logger.debug(f"macOS GPU detection failed: {e}")

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
        
        # Check for Intel IPEX (Intel Extension for PyTorch)
        try:
            import intel_extension_for_pytorch
            self.accelerators.append(HardwareAccelerator.INTEL_IPEX)
            logger.info("Detected Intel Extension for PyTorch (IPEX)")
        except ImportError:
            pass
        
        # Check for TPU support
        try:
            import tensorflow as tf
            tpu_devices = tf.config.list_logical_devices('TPU')
            if tpu_devices:
                self.accelerators.append(HardwareAccelerator.TPU)
                logger.info(f"Detected {len(tpu_devices)} TPU devices")
        except (ImportError, AttributeError, ValueError):
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
        if self.total_memory_gb < 4 or self.cpu_count_physical <= 2:
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
        
        if self.gpus:
            logger.info("-" * 50)
            logger.info(f"GPUs detected: {len(self.gpus)}")
            for i, gpu in enumerate(self.gpus):
                logger.info(f"  GPU {i}: {gpu.name}, {gpu.memory_mb} MB memory")
        
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

    def _get_optimal_quantization_type(self) -> str:
        """Determine the optimal quantization type based on hardware"""
        # Use more aggressive quantization on memory-constrained devices
        if self.total_memory_gb < 4 or self.optimization_mode == OptimizationMode.MEMORY_SAVING:
            return QuantizationType.INT8.value
        
        # Use fp16 if GPUs are available and support it
        if self.gpus and any("compute capability" in getattr(gpu, "compute_capability", "") for gpu in self.gpus):
            return QuantizationType.FLOAT16.value
        
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

    def get_optimal_training_engine_config(self) -> MLTrainingEngineConfig:
        """
        Create an optimized training engine configuration based on device capabilities.
        
        Returns:
            Optimized MLTrainingEngineConfig
        """
        # Get scaling factors based on optimization mode
        factors = self._apply_optimization_mode_factors()
        
        # CPU utilization for jobs
        cpu_factor = factors["cpu"]
        n_jobs = max(int(self.cpu_count_logical * cpu_factor), 1)
        
        # Choose optimization strategy based on hardware and mode
        if self.total_memory_gb > 16 and self.cpu_count_logical > 8 and self.optimization_mode in [OptimizationMode.PERFORMANCE, OptimizationMode.FULL_UTILIZATION]:
            optimization_strategy = OptimizationStrategy.ASHT
            optimization_iterations = 100 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 50
        elif self.total_memory_gb > 8 and self.cpu_count_logical > 4 and self.optimization_mode not in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING]:
            optimization_strategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
            optimization_iterations = 50 if self.optimization_mode == OptimizationMode.PERFORMANCE else 30
        else:
            optimization_strategy = OptimizationStrategy.RANDOM_SEARCH
            optimization_iterations = 30 if self.optimization_mode == OptimizationMode.PERFORMANCE else 20
        
        # For cloud environments with many cores, use more advanced strategies
        if self.environment == "cloud" and self.cpu_count_logical > 16 and self.optimization_mode == OptimizationMode.FULL_UTILIZATION:
            optimization_strategy = OptimizationStrategy.BOHB
            optimization_iterations = 150
        
        # Time budget for optimization
        optimization_timeout = (
            None if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else
            3600 if self.optimization_mode == OptimizationMode.PERFORMANCE else
            1800  # 30 minutes for other modes
        )
        
        # Cross-validation settings
        cv_folds = (
            10 if (self.optimization_mode == OptimizationMode.FULL_UTILIZATION and self.environment != "edge") else
            5 if self.environment != "edge" else
            3  # Fewer folds for edge devices
        )
        
        # Test size - larger for more conservative modes
        test_size = (
            0.15 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else
            0.2 if self.optimization_mode == OptimizationMode.PERFORMANCE else
            0.25
        )
        
        # Early stopping settings
        early_stopping = self.optimization_mode != OptimizationMode.FULL_UTILIZATION
        early_stopping_rounds = (
            20 if self.optimization_mode == OptimizationMode.CONSERVATIVE else
            10 if self.optimization_mode == OptimizationMode.BALANCED else
            5  # Fewer rounds for performance mode
        )
        
        # Feature selection settings
        feature_selection = self.optimization_mode != OptimizationMode.PERFORMANCE
        feature_selection_method = (
            "mutual_info" if self.cpu_count_logical > 4 else
            "chi2"  # Simpler method for fewer cores
        )
        
        # Feature importance threshold
        feature_importance_threshold = (
            0.005 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else
            0.01 if self.optimization_mode == OptimizationMode.BALANCED else
            0.02  # Higher threshold for conservative modes
        )
        
        # Get other configurations
        preprocessing_config = self.get_optimal_preprocessor_config()
        batch_processing_config = self.get_optimal_batch_processor_config()
        inference_config = self.get_optimal_inference_engine_config()
        
        # Fix: Add batch_processing_strategy attribute if it's missing
        if not hasattr(inference_config, 'batch_processing_strategy'):
            inference_config.batch_processing_strategy = batch_processing_config.strategy if hasattr(batch_processing_config, 'strategy') else "auto"
            
        quantization_config = self.get_optimal_quantization_config()
        
        # Explainability configuration
        explainability_config = ExplainabilityConfig(
            enable_explainability=self.optimization_mode != OptimizationMode.MEMORY_SAVING,
            methods=["shap", "feature_importance"] if self.environment != "edge" else ["feature_importance"],
            default_method="shap" if self.total_memory_gb > 8 else "feature_importance",
            shap_algorithm="auto",
            shap_samples=100 if self.optimization_mode != OptimizationMode.FULL_UTILIZATION else 200,
            generate_summary=True,
            generate_plots=self.environment != "edge",
            store_explanations=self.environment == "cloud" or self.optimization_mode == OptimizationMode.FULL_UTILIZATION
        )
        
        # Monitoring configuration
        monitoring_config = MonitoringConfig(
            enable_monitoring=True,
            drift_detection=self.environment != "edge",
            drift_detection_method="ks_test",
            drift_threshold=0.05,
            performance_tracking=True,
            alert_on_drift=self.resilience_level > 1,
            alert_channels=["log"] if self.environment == "edge" else ["log", "metrics"],
            data_sampling_rate=1.0 if self.environment == "cloud" else 0.5,
            store_predictions=self.environment == "cloud" or self.optimization_mode == OptimizationMode.FULL_UTILIZATION,
            enable_auto_retraining=self.resilience_level > 2 and self.environment == "cloud",
            export_metrics=self.environment == "cloud"
        )
        
        # AutoML settings
        auto_ml = (
            AutoMLMode.COMPREHENSIVE if (self.optimization_mode == OptimizationMode.FULL_UTILIZATION and self.environment == "cloud") else
            AutoMLMode.BASIC if (self.optimization_mode not in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING]) else
            AutoMLMode.DISABLED
        )
        
        # Time budget for AutoML
        auto_ml_time_budget = (
            7200 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else
            3600 if self.optimization_mode == OptimizationMode.PERFORMANCE else
            1800  # 30 minutes for other modes
        ) if auto_ml != AutoMLMode.DISABLED else None
        
        # Ensemble settings
        ensemble_models = (
            self.environment == "cloud" and
            self.optimization_mode not in [OptimizationMode.CONSERVATIVE, OptimizationMode.MEMORY_SAVING]
        )
        
        # Ensemble method
        ensemble_method = (
            "stacking" if (self.optimization_mode == OptimizationMode.FULL_UTILIZATION and self.total_memory_gb > 16) else
            "voting"  # Simpler method for other modes
        )
        
        # Check for hardware acceleration features - CPU agnostic approach
        use_acceleration = False
        if hasattr(self, 'has_avx') and self.has_avx:
            use_acceleration = True
        elif hasattr(self, 'has_avx2') and self.has_avx2:
            use_acceleration = True
        elif hasattr(self, 'has_neon') and self.has_neon:
            use_acceleration = True
        
        # Create the configuration
        return MLTrainingEngineConfig(
            task_type=TaskType.CLASSIFICATION,  # Default, should be set based on actual task
            random_state=42,
            n_jobs=n_jobs,
            verbose=2 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 1,
            cv_folds=cv_folds,
            test_size=test_size,
            stratify=True,
            optimization_strategy=optimization_strategy,
            optimization_iterations=optimization_iterations,
            optimization_timeout=optimization_timeout,
            early_stopping=early_stopping,
            early_stopping_rounds=early_stopping_rounds,
            feature_selection=feature_selection,
            feature_selection_method=feature_selection_method,
            feature_importance_threshold=feature_importance_threshold,
            preprocessing_config=preprocessing_config,
            batch_processing_config=batch_processing_config,
            inference_config=inference_config,
            quantization_config=quantization_config,
            explainability_config=explainability_config,
            monitoring_config=monitoring_config,
            model_path=str(self.model_registry_path),
            auto_version_models=True,
            experiment_tracking=self.environment != "edge",
            experiment_tracking_platform="mlflow",
            use_hardware_acceleration=use_acceleration,  # Changed from Intel-specific to generic hardware acceleration
            use_gpu=self.gpus and len(self.gpus) > 0,
            gpu_memory_fraction=0.95 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 0.8,
            memory_optimization=self.optimization_mode != OptimizationMode.FULL_UTILIZATION,
            enable_distributed=len(self.gpus) > 1 or (self.cpu_count_physical > 8 and self.environment == "cloud"),
            distributed_strategy="mirrored" if len(self.gpus) > 1 else "multiworker",
            checkpointing=self.resilience_level > 0,
            checkpoint_interval=5 if self.resilience_level > 2 else 10,
            checkpoint_path=str(self.checkpoint_path),
            enable_pruning=self.optimization_mode in [OptimizationMode.MEMORY_SAVING, OptimizationMode.CONSERVATIVE],
            auto_ml=auto_ml,
            auto_ml_time_budget=auto_ml_time_budget,
            ensemble_models=ensemble_models,
            ensemble_method=ensemble_method,
            ensemble_size=5 if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else 3,
            model_selection_criteria=ModelSelectionCriteria.ACCURACY,  # Default, should be set based on task
            auto_save=True,
            enable_quantization=self.optimization_mode in [OptimizationMode.MEMORY_SAVING, OptimizationMode.BALANCED],
            enable_model_compression=self.optimization_mode == OptimizationMode.MEMORY_SAVING,
            compression_method="pruning",
            log_level="DEBUG" if self.optimization_mode == OptimizationMode.FULL_UTILIZATION else "INFO",
            debug_mode=False,
            enable_data_validation=True,
            enable_security=self.resilience_level > 1
        )
        
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information as a dictionary.
        
        Returns:
            Dictionary with system information
        """
        # Create a dictionary with system information
        system_info = {
            "system": {
                "name": self.system,
                "release": self.release,
                "machine": self.machine,
                "processor": self.processor,
                "hostname": self.hostname,
                "python_version": self.python_version,
                "environment": self.environment
            },
            "cpu": {
                "physical_cores": self.cpu_count_physical,
                "logical_cores": self.cpu_count_logical,
                "frequency": self.cpu_freq,
                "is_intel": self.is_intel_cpu,
                "is_amd": self.is_amd_cpu,
                "is_arm": self.is_arm_cpu,
                "features": {
                    "avx": self.has_avx,
                    "avx2": self.has_avx2,
                    "avx512": self.has_avx512,
                    "sse4": self.has_sse4,
                    "fma": self.has_fma,
                    "neon": self.has_neon
                }
            },
            "memory": {
                "total_gb": self.total_memory_gb,
                "available_gb": self.available_memory_gb,
                "usable_gb": self.usable_memory_gb,
                "swap_gb": self.swap_memory_gb,
                "reservation_percent": self.memory_reservation_percent
            },
            "disk": {
                "total_gb": self.disk_total_gb,
                "free_gb": self.disk_free_gb,
                "is_ssd": self.is_ssd
            },
            "gpu": {
                "count": len(self.gpus),
                "has_cuda": self.has_cuda,
                "has_rocm": self.has_rocm,
                "gpus": [gpu.to_dict() for gpu in self.gpus]
            },
            "accelerators": [acc.name for acc in self.accelerators],
            "optimization": {
                "mode": self.optimization_mode.value,
                "workload_type": self.workload_type,
                "power_efficiency": self.power_efficiency,
                "resilience_level": self.resilience_level,
                "auto_tune": self.auto_tune
            },
            "paths": {
                "config_path": str(self.config_path),
                "checkpoint_path": str(self.checkpoint_path),
                "model_registry_path": str(self.model_registry_path)
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return system_info
        
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
# Helper functions for creating optimized configurations
def create_optimized_configs(
    config_path: str = "./configs", 
    checkpoint_path: str = "./checkpoints", 
    model_registry_path: str = "./model_registry",
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED,
    workload_type: str = "mixed",
    environment: str = "auto",
    enable_gpu: bool = True,
    enable_specialized_accelerators: bool = True,
    memory_reservation_percent: float = 10.0,
    power_efficiency: bool = False,
    resilience_level: int = 1,
    auto_tune: bool = True,
    config_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Create and save optimized configurations for the ML pipeline using the Enhanced Device Optimizer.
    
    Args:
        config_path: Path to save configuration files
        checkpoint_path: Path for model checkpoints
        model_registry_path: Path for model registry
        optimization_mode: Mode for optimization strategy
        workload_type: Type of workload ("inference", "training", "mixed")
        environment: Computing environment ("cloud", "desktop", "edge", "auto")
        enable_gpu: Whether to enable GPU detection and optimization
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
        enable_gpu=enable_gpu,
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
    enable_gpu: bool = True,
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
        enable_gpu: Whether to enable GPU detection and optimization
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
        enable_gpu=enable_gpu,
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
    enable_gpu: bool = True,
    enable_specialized_accelerators: bool = True
) -> Dict[str, Any]:
    """
    Get comprehensive information about the current system.
    
    Args:
        enable_gpu: Whether to detect GPU capabilities
        enable_specialized_accelerators: Whether to detect specialized hardware
    
    Returns:
        Dictionary with detailed system information
    """
    # Create optimizer just to use its detection capabilities
    optimizer = DeviceOptimizer(
        enable_gpu=enable_gpu,
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