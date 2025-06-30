# CPU Device Optimizer

## Overview
The CPU Device Optimizer is a sophisticated module for automatically configuring machine learning pipeline settings based on CPU capabilities. It performs hardware detection and generates optimized configurations for various ML pipeline components, including quantization, batch processing, preprocessing, inference, and training engines.

## Prerequisites
- Python ≥3.6
- Required packages:
  ```bash
  pip install psutil numpy pathlib dataclasses
  ```
- Dependencies from `modules.configs` (with classes for configuration components)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from cpu_device_optimizer import DeviceOptimizer, OptimizationMode

# Create an optimizer with default settings
optimizer = DeviceOptimizer()

# Get optimized configurations
quantization_config = optimizer.get_optimal_quantization_config()
batch_config = optimizer.get_optimal_batch_processor_config()
preprocessor_config = optimizer.get_optimal_preprocessor_config()
inference_config = optimizer.get_optimal_inference_engine_config()
training_config = optimizer.get_optimal_training_engine_config()

# Save all configurations to disk
config_paths = optimizer.save_configs()

# Create configurations for performance mode
performance_optimizer = DeviceOptimizer(
    optimization_mode=OptimizationMode.PERFORMANCE,
    workload_type="inference"
)
performance_configs = performance_optimizer.save_configs("perf_configs")

# Create configurations for all optimization modes
all_mode_configs = create_configs_for_all_modes()
```

## Configuration
| Parameter                       | Default          | Description                                     |
|---------------------------------|------------------|-------------------------------------------------|
| `config_path`                   | `"./configs"`    | Path to save configuration files                |
| `checkpoint_path`               | `"./checkpoints"`| Path for model checkpoints                      |
| `model_registry_path`           | `"./model_registry"` | Path for model registry                     |
| `optimization_mode`             | `OptimizationMode.BALANCED` | Mode for optimization strategy       |
| `workload_type`                 | `"mixed"`        | Type of workload ("inference", "training", "mixed") |
| `environment`                   | `"auto"`         | Computing environment ("cloud", "desktop", "edge", "auto") |
| `enable_specialized_accelerators`| `True`          | Whether to enable detection of specialized hardware |
| `memory_reservation_percent`     | `10.0`          | Percentage of memory to reserve for the system  |
| `power_efficiency`               | `False`         | Whether to optimize for power efficiency        |
| `resilience_level`               | `1`             | Level of fault tolerance (0-3)                  |
| `auto_tune`                      | `True`          | Whether to enable automatic parameter tuning    |

---

## Classes

### `HardwareAccelerator`
```python
class HardwareAccelerator(Enum):
```
- **Description**:  
  An enumeration class defining types of hardware accelerators that might be available on the system.

- **Attributes**:  
  - `NONE`: No accelerator
  - `INTEL_IPEX`: Intel IPEX accelerator
  - `INTEL_MKL`: Intel MKL accelerator
  - `ARM_NEON`: ARM NEON accelerator

---

### `DeviceOptimizer`
```python
class DeviceOptimizer:
```
- **Description**:  
  CPU-only device optimizer that automatically configures ML pipeline settings based on device capabilities with sophisticated optimization strategies.

- **Attributes**:  
  - `config_path (Path)`: Path to save configuration files
  - `checkpoint_path (Path)`: Path for model checkpoints
  - `model_registry_path (Path)`: Path for model registry
  - `optimization_mode (OptimizationMode)`: Mode for optimization strategy
  - `workload_type (str)`: Type of workload ("inference", "training", "mixed")
  - `power_efficiency (bool)`: Whether to optimize for power efficiency
  - `resilience_level (int)`: Level of fault tolerance (0-3)
  - `auto_tune (bool)`: Whether to enable automatic parameter tuning
  - `memory_reservation_percent (float)`: Percentage of memory to reserve for the system
  - `needs_feature_scaling (bool)`: Whether feature scaling is required
  - `onnx_available (bool)`: Whether ONNX is available
  - `threadpoolctl_available (bool)`: Whether threadpoolctl is available
  - `treelite_available (bool)`: Whether Treelite is available
  - `system (str)`: Operating system name
  - `release (str)`: Operating system release
  - `machine (str)`: Machine type
  - `processor (str)`: Processor type
  - `hostname (str)`: System hostname
  - `python_version (str)`: Python version
  - `cpu_count_physical (int)`: Number of physical CPU cores
  - `cpu_count_logical (int)`: Number of logical CPU cores
  - `cpu_freq (Dict[str, float])`: CPU frequency information
  - `has_avx (bool)`: Whether AVX is supported
  - `has_avx2 (bool)`: Whether AVX2 is supported
  - `has_avx512 (bool)`: Whether AVX512 is supported
  - `has_sse4 (bool)`: Whether SSE4 is supported
  - `has_fma (bool)`: Whether FMA is supported
  - `is_intel_cpu (bool)`: Whether the CPU is Intel
  - `is_amd_cpu (bool)`: Whether the CPU is AMD
  - `is_arm_cpu (bool)`: Whether the CPU is ARM
  - `has_neon (bool)`: Whether ARM NEON is supported
  - `total_memory_gb (float)`: Total system memory in GB
  - `available_memory_gb (float)`: Available system memory in GB
  - `usable_memory_gb (float)`: Usable system memory in GB
  - `swap_memory_gb (float)`: Swap memory in GB
  - `disk_total_gb (float)`: Total disk space in GB
  - `disk_free_gb (float)`: Free disk space in GB
  - `is_ssd (bool)`: Whether the disk is an SSD
  - `accelerators (List[HardwareAccelerator])`: Detected hardware accelerators
  - `environment (str)`: Computing environment

- **Constructor**:
  ```python
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
  ```
  - **Parameters**:
    - `config_path (str)`: Path to save configuration files. Default is "./configs".
    - `checkpoint_path (str)`: Path for model checkpoints. Default is "./checkpoints".
    - `model_registry_path (str)`: Path for model registry. Default is "./model_registry".
    - `optimization_mode (OptimizationMode)`: Mode for optimization strategy. Default is OptimizationMode.BALANCED.
    - `workload_type (str)`: Type of workload ("inference", "training", "mixed"). Default is "mixed".
    - `environment (str)`: Computing environment ("cloud", "desktop", "edge", "auto"). Default is "auto".
    - `enable_specialized_accelerators (bool)`: Whether to enable detection of specialized hardware. Default is True.
    - `memory_reservation_percent (float)`: Percentage of memory to reserve for the system. Default is 10.0.
    - `power_efficiency (bool)`: Whether to optimize for power efficiency. Default is False.
    - `resilience_level (int)`: Level of fault tolerance (0-3). Default is 1.
    - `auto_tune (bool)`: Whether to enable automatic parameter tuning. Default is True.
  - **Raises**:
    - No explicit exceptions raised.

- **Methods**:  
  - `_detect_system_info(self) -> None`: Detect basic system information
  - `_detect_cpu_capabilities(self) -> None`: Detect detailed CPU capabilities
  - `_get_cpu_frequency(self) -> Dict[str, float]`: Get CPU frequency information
  - `_detect_memory_info(self) -> None`: Detect memory information
  - `_detect_disk_info(self) -> None`: Detect disk information
  - `_check_if_ssd(self) -> bool`: Check if the current disk is an SSD
  - `_find_mount_point(self, path) -> str`: Find the mount point of a given path
  - `_get_device_for_mount_point(self, mount_point) -> Optional[str]`: Get the device for a mount point
  - `_detect_specialized_accelerators(self) -> None`: Detect specialized hardware accelerators
  - `_detect_environment(self) -> str`: Auto-detect the computing environment
  - `_log_system_overview(self) -> None`: Log detected system information
  - `_serialize_config_dict(self, config_dict) -> Dict`: Convert Enum values to strings for JSON serialization
  - `save_configs(self, config_id: Optional[str] = None) -> Dict[str, str]`: Generate and save all optimized configurations
  - `load_configs(self, config_id: str) -> Dict[str, Any]`: Load previously saved configurations
  - `create_configs_for_all_modes(self) -> Dict[str, Dict[str, str]]`: Create and save configurations for all optimization modes
  - `auto_tune_configs(self, workload_sample: Any = None) -> Dict[str, Any]`: Auto-tune configurations based on a sample workload
  - `_get_optimal_quantization_type(self) -> str`: Determine the optimal quantization type based on hardware
  - `_apply_optimization_mode_factors(self) -> Dict[str, float]`: Get scaling factors for different resource parameters based on the optimization mode
  - `get_optimal_quantization_config(self) -> QuantizationConfig`: Generate an optimized quantization configuration
  - `get_optimal_preprocessor_config(self) -> PreprocessorConfig`: Generate an optimized preprocessor configuration
  - `get_optimal_inference_engine_config(self) -> InferenceEngineConfig`: Generate an optimized inference engine configuration
  - `get_optimal_batch_processor_config(self) -> BatchProcessorConfig`: Generate an optimized batch processor configuration
  - `get_system_info(self) -> Dict[str, Any]`: Get comprehensive information about the current system
  - `get_optimal_training_engine_config(self) -> MLTrainingEngineConfig`: Generate an optimized training engine configuration

#### Method: `_detect_system_info`
```python
def _detect_system_info(self) -> None:
```
- **Description**:  
  Detects basic system information such as OS, machine type, processor, hostname, and Python version.

- **Parameters**:  
  - None

- **Returns**:  
  - None

#### Method: `_detect_cpu_capabilities`
```python
def _detect_cpu_capabilities(self) -> None:
```
- **Description**:  
  Detects detailed CPU capabilities including core count, frequency, and advanced CPU features like AVX, AVX2, AVX512, SSE4, and FMA.

- **Parameters**:  
  - None

- **Returns**:  
  - None

#### Method: `_get_cpu_frequency`
```python
def _get_cpu_frequency(self) -> Dict[str, float]:
```
- **Description**:  
  Gets CPU frequency information including current, minimum, and maximum frequencies.

- **Parameters**:  
  - None

- **Returns**:  
  - `Dict[str, float]`: Dictionary containing current, min, and max CPU frequencies

#### Method: `_detect_memory_info`
```python
def _detect_memory_info(self) -> None:
```
- **Description**:  
  Detects memory information including total memory, available memory, usable memory, and swap memory.

- **Parameters**:  
  - None

- **Returns**:  
  - None

#### Method: `_detect_disk_info`
```python
def _detect_disk_info(self) -> None:
```
- **Description**:  
  Detects disk information including total disk space, free disk space, and whether the disk is an SSD.

- **Parameters**:  
  - None

- **Returns**:  
  - None

#### Method: `_check_if_ssd`
```python
def _check_if_ssd(self) -> bool:
```
- **Description**:  
  Checks if the current disk is an SSD by examining system information.

- **Parameters**:  
  - None

- **Returns**:  
  - `bool`: True if the disk is an SSD, False otherwise

#### Method: `_find_mount_point`
```python
def _find_mount_point(self, path) -> str:
```
- **Description**:  
  Finds the mount point of a given path by traversing up the directory tree.

- **Parameters**:  
  - `path (str)`: File system path to find the mount point for

- **Returns**:  
  - `str`: Mount point path

#### Method: `_get_device_for_mount_point`
```python
def _get_device_for_mount_point(self, mount_point) -> Optional[str]:
```
- **Description**:  
  Gets the device for a mount point by examining system mount information.

- **Parameters**:  
  - `mount_point (str)`: Mount point to find the device for

- **Returns**:  
  - `Optional[str]`: Device path or None if not found

#### Method: `_detect_specialized_accelerators`
```python
def _detect_specialized_accelerators(self) -> None:
```
- **Description**:  
  Detects specialized hardware accelerators like Intel MKL or ARM NEON that can be used for optimization.

- **Parameters**:  
  - None

- **Returns**:  
  - None

#### Method: `_detect_environment`
```python
def _detect_environment(self) -> str:
```
- **Description**:  
  Auto-detects the computing environment (cloud, desktop, or edge) based on system characteristics.

- **Parameters**:  
  - None

- **Returns**:  
  - `str`: Detected environment ("cloud", "desktop", or "edge")

#### Method: `_log_system_overview`
```python
def _log_system_overview(self) -> None:
```
- **Description**:  
  Logs detected system information for diagnostic purposes.

- **Parameters**:  
  - None

- **Returns**:  
  - None

#### Method: `_serialize_config_dict`
```python
def _serialize_config_dict(self, config_dict) -> Dict:
```
- **Description**:  
  Converts Enum values to strings for JSON serialization to ensure configurations can be saved to files.

- **Parameters**:  
  - `config_dict (Dict)`: Dictionary that may contain Enum values

- **Returns**:  
  - `Dict`: Dictionary with Enum values converted to strings

#### Method: `save_configs`
```python
def save_configs(self, config_id: Optional[str] = None) -> Dict[str, str]:
```
- **Description**:  
  Generates and saves all optimized configurations to disk.

- **Parameters**:  
  - `config_id (Optional[str])`: Optional identifier for the configuration set. Default is None.

- **Returns**:  
  - `Dict[str, str]`: Dictionary with paths to saved configuration files

#### Method: `load_configs`
```python
def load_configs(self, config_id: str) -> Dict[str, Any]:
```
- **Description**:  
  Loads previously saved configurations from disk.

- **Parameters**:  
  - `config_id (str)`: Identifier for the configuration set

- **Returns**:  
  - `Dict[str, Any]`: Dictionary with loaded configurations

#### Method: `create_configs_for_all_modes`
```python
def create_configs_for_all_modes(self) -> Dict[str, Dict[str, str]]:
```
- **Description**:  
  Creates and saves configurations for all optimization modes (BALANCED, PERFORMANCE, MEMORY_SAVING, etc.).

- **Parameters**:  
  - None

- **Returns**:  
  - `Dict[str, Dict[str, str]]`: Dictionary of master configurations for each optimization mode

#### Method: `auto_tune_configs`
```python
def auto_tune_configs(self, workload_sample: Any = None) -> Dict[str, Any]:
```
- **Description**:  
  Auto-tunes configurations based on a sample workload if auto_tune is enabled.

- **Parameters**:  
  - `workload_sample (Any)`: Optional sample data to tune configurations. Default is None.

- **Returns**:  
  - `Dict[str, Any]`: Dictionary with auto-tuned configurations

#### Method: `_get_optimal_quantization_type`
```python
def _get_optimal_quantization_type(self) -> str:
```
- **Description**:  
  Determines the optimal quantization type based on hardware capabilities and optimization mode.

- **Parameters**:  
  - None

- **Returns**:  
  - `str`: Optimal quantization type value

#### Method: `_apply_optimization_mode_factors`
```python
def _apply_optimization_mode_factors(self) -> Dict[str, float]:
```
- **Description**:  
  Gets scaling factors for different resource parameters based on the optimization mode.

- **Parameters**:  
  - None

- **Returns**:  
  - `Dict[str, float]`: Dictionary of scaling factors for CPU, memory, and other resources

#### Method: `get_optimal_quantization_config`
```python
def get_optimal_quantization_config(self) -> QuantizationConfig:
```
- **Description**:  
  Generates an optimized quantization configuration based on device capabilities.

- **Parameters**:  
  - None

- **Returns**:  
  - `QuantizationConfig`: Optimized quantization configuration

#### Method: `get_optimal_preprocessor_config`
```python
def get_optimal_preprocessor_config(self) -> PreprocessorConfig:
```
- **Description**:  
  Generates an optimized preprocessor configuration based on device capabilities.

- **Parameters**:  
  - None

- **Returns**:  
  - `PreprocessorConfig`: Optimized preprocessor configuration

#### Method: `get_optimal_inference_engine_config`
```python
def get_optimal_inference_engine_config(self) -> InferenceEngineConfig:
```
- **Description**:  
  Generates an optimized inference engine configuration based on device capabilities.

- **Parameters**:  
  - None

- **Returns**:  
  - `InferenceEngineConfig`: Optimized inference engine configuration

#### Method: `get_optimal_batch_processor_config`
```python
def get_optimal_batch_processor_config(self) -> BatchProcessorConfig:
```
- **Description**:  
  Generates an optimized batch processor configuration based on device capabilities.

- **Parameters**:  
  - None

- **Returns**:  
  - `BatchProcessorConfig`: Optimized batch processor configuration

#### Method: `get_system_info`
```python
def get_system_info(self) -> Dict[str, Any]:
```
- **Description**:  
  Gets comprehensive information about the current system including CPU, memory, disk, and accelerator details.

- **Parameters**:  
  - None

- **Returns**:  
  - `Dict[str, Any]`: Dictionary with detailed system information

#### Method: `get_optimal_training_engine_config`
```python
def get_optimal_training_engine_config(self) -> MLTrainingEngineConfig:
```
- **Description**:  
  Generates an optimized training engine configuration based on device capabilities.

- **Parameters**:  
  - None

- **Returns**:  
  - `MLTrainingEngineConfig`: Optimized training engine configuration

---

## Helper Functions

### `create_optimized_configs`
```python
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
```
- **Description**:  
  Creates and saves optimized configurations for the ML pipeline using the CPU-Only Device Optimizer.

- **Parameters**:  
  - `config_path (str)`: Path to save configuration files. Default is "./configs".
  - `checkpoint_path (str)`: Path for model checkpoints. Default is "./checkpoints".
  - `model_registry_path (str)`: Path for model registry. Default is "./model_registry".
  - `optimization_mode (OptimizationMode)`: Mode for optimization strategy. Default is OptimizationMode.BALANCED.
  - `workload_type (str)`: Type of workload ("inference", "training", "mixed"). Default is "mixed".
  - `environment (str)`: Computing environment ("cloud", "desktop", "edge", "auto"). Default is "auto".
  - `enable_specialized_accelerators (bool)`: Whether to enable detection of specialized hardware. Default is True.
  - `memory_reservation_percent (float)`: Percentage of memory to reserve for the system. Default is 10.0.
  - `power_efficiency (bool)`: Whether to optimize for power efficiency. Default is False.
  - `resilience_level (int)`: Level of fault tolerance (0-3). Default is 1.
  - `auto_tune (bool)`: Whether to enable automatic parameter tuning. Default is True.
  - `config_id (Optional[str])`: Optional identifier for the configuration set. Default is None.

- **Returns**:  
  - `Dict[str, str]`: Dictionary with paths to saved configuration files

---

### `create_configs_for_all_modes`
```python
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
```
- **Description**:  
  Creates and saves configurations for all optimization modes.

- **Parameters**:  
  - `config_path (str)`: Path to save configuration files. Default is "./configs".
  - `checkpoint_path (str)`: Path for model checkpoints. Default is "./checkpoints".
  - `model_registry_path (str)`: Path for model registry. Default is "./model_registry".
  - `workload_type (str)`: Type of workload ("inference", "training", "mixed"). Default is "mixed".
  - `environment (str)`: Computing environment ("cloud", "desktop", "edge", "auto"). Default is "auto".
  - `enable_specialized_accelerators (bool)`: Whether to enable detection of specialized hardware. Default is True.
  - `memory_reservation_percent (float)`: Percentage of memory to reserve for the system. Default is 10.0.
  - `power_efficiency (bool)`: Whether to optimize for power efficiency. Default is False.
  - `resilience_level (int)`: Level of fault tolerance (0-3). Default is 1.

- **Returns**:  
  - `Dict[str, Dict[str, str]]`: Dictionary of configurations for each optimization mode

---

### `load_saved_configs`
```python
def load_saved_configs(
    config_path: str,
    config_id: str
) -> Dict[str, Any]:
```
- **Description**:  
  Loads previously saved configuration files.

- **Parameters**:  
  - `config_path (str)`: Path where configuration files are stored
  - `config_id (str)`: Identifier for the configuration set

- **Returns**:  
  - `Dict[str, Any]`: Dictionary with loaded configurations

---

### `get_system_information`
```python
def get_system_information(
    enable_specialized_accelerators: bool = True
) -> Dict[str, Any]:
```
- **Description**:  
  Gets comprehensive information about the current system.

- **Parameters**:  
  - `enable_specialized_accelerators (bool)`: Whether to detect specialized hardware. Default is True.

- **Returns**:  
  - `Dict[str, Any]`: Dictionary with detailed system information

---

### `optimize_for_environment`
```python
def optimize_for_environment(environment: str) -> Dict[str, str]:
```
- **Description**:  
  Creates optimized configurations specifically for a given environment type.

- **Parameters**:  
  - `environment (str)`: Target environment ("cloud", "desktop", "edge")

- **Returns**:  
  - `Dict[str, str]`: Dictionary with paths to saved configuration files

- **Raises**:
  - `ValueError`: If the environment is not "cloud", "desktop", or "edge"

---

### `optimize_for_workload`
```python
def optimize_for_workload(workload_type: str) -> Dict[str, str]:
```
- **Description**:  
  Creates optimized configurations specifically for a given workload type.

- **Parameters**:  
  - `workload_type (str)`: Target workload type ("inference", "training", "mixed")

- **Returns**:  
  - `Dict[str, str]`: Dictionary with paths to saved configuration files

- **Raises**:
  - `ValueError`: If the workload type is not "inference", "training", or "mixed"

---

### `apply_configs_to_pipeline`
```python
def apply_configs_to_pipeline(configs_dict: Dict[str, Any]) -> bool:
```
- **Description**:  
  Applies loaded configurations to ML pipeline components.

- **Parameters**:  
  - `configs_dict (Dict[str, Any])`: Dictionary with loaded configurations

- **Returns**:  
  - `bool`: True if configurations were successfully applied, False otherwise

---

### `get_default_config`
```python
def get_default_config(
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED,
    workload_type: str = "mixed",
    environment: str = "auto",
    output_dir: str = "./configs/default",
    enable_specialized_accelerators: bool = True,
) -> Dict[str, Any]:
```
- **Description**:  
  Gets a set of default configurations optimized for the current system.

- **Parameters**:  
  - `optimization_mode (OptimizationMode)`: The optimization strategy to use. Default is OptimizationMode.BALANCED.
  - `workload_type (str)`: Type of workload to optimize for ("inference", "training", or "mixed"). Default is "mixed".
  - `environment (str)`: Computing environment ("cloud", "desktop", "edge", or "auto" for automatic detection). Default is "auto".
  - `output_dir (str)`: Directory where configuration files will be saved. Default is "./configs/default".
  - `enable_specialized_accelerators (bool)`: Whether to enable detection and optimization for specialized hardware accelerators. Default is True.

- **Returns**:  
  - `Dict[str, Any]`: Dictionary containing all optimized configurations

---

## Architecture
The CPU Device Optimizer includes several key components that work together to provide optimized configurations for machine learning pipelines:

1. **Hardware Detection**: Detects CPU capabilities, memory, disk, and specialized accelerators
2. **Environment Analysis**: Determines the computing environment (cloud, desktop, edge)
3. **Optimization Engines**: Generates optimized configurations for each pipeline component
4. **Configuration Management**: Saves, loads, and applies configurations to ML pipelines

**Data Flow**: System Hardware → Device Optimizer → Optimized Configurations → ML Pipeline Components

---

## Testing
```bash
python -m unittest tests/test_device_optimizer.py
```

## Security & Compliance
- No sensitive data is collected or transmitted
- All configurations are stored locally
- Hardware detection is performed securely using system APIs

> Last Updated: 2025-05-11