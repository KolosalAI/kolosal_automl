# Module: `device_optimizer`

## Overview

This module defines the `DeviceOptimizer`, a comprehensive CPU-only optimization engine
for machine learning pipelines. It adapts configurations across various pipeline
components (quantization, preprocessing, batching, inference, training) based on
detected hardware specs, workload, and optimization goals (performance, memory, etc.).

It dynamically generates and saves configurations using dataclass-based schemas
defined in `modules.configs`, and is capable of system-aware tuning for edge, desktop,
or cloud environments.

## Prerequisites
- Python ≥3.8
- Dependencies:
  ```bash
  pip install psutil numpy
  ```
- Internal dependencies: `modules.configs` (must be implemented or imported)

## Installation
```bash
# Clone the project and install dependencies
pip install -r requirements.txt
```

## Usage
```python
from device_optimizer import create_optimized_configs

configs = create_optimized_configs(
    optimization_mode=OptimizationMode.PERFORMANCE,
    workload_type="training",
    environment="cloud"
)
```

## Configuration Options
| Parameter                    | Description                                        |
|-----------------------------|----------------------------------------------------|
| `optimization_mode`         | Strategy: BALANCED, PERFORMANCE, MEMORY_SAVING... |
| `workload_type`             | Type of load: inference, training, mixed          |
| `environment`               | Target: auto, desktop, cloud, edge                |
| `enable_specialized_accelerators` | Detect AVX, NEON, MKL, etc.                |

## Architecture
```
+---------------------+
|   DeviceOptimizer   |
+---------------------+
       |       |
       v       v
+------------+  +----------------+
| System Info|  | Config Generators |
+------------+  +----------------+
       |                |
       v                v
+------------+     +--------------+
| Save/Load  |     | Auto-tune     |
+------------+     +--------------+
```

## Testing
```bash
pytest tests/
```

## Security & Compliance
- No external data transmission
- Designed for privacy-preserving local execution

> Last Updated: 2025-04-28

---

## Class: `DeviceOptimizer`
```python
class DeviceOptimizer:
```
### Description
Optimizes configuration settings for a CPU-only ML pipeline. Automatically adapts to
hardware capabilities, workload types, and environmental context (cloud, edge, etc.).

### Attributes
- `system`, `processor`, `hostname` (str): System-level metadata
- `cpu_count_physical`, `cpu_count_logical`: Core counts
- `total_memory_gb`, `usable_memory_gb`: Memory metrics
- `disk_total_gb`, `is_ssd`: Disk metrics
- `accelerators`: List of detected `HardwareAccelerator`

### Constructor
```python
def __init__(self, config_path: str = "./configs", ...):
```
#### Parameters
- `config_path` (str): Where config files will be saved
- `optimization_mode` (OptimizationMode): Strategy for tuning (BALANCED, etc.)
- `workload_type` (str): Type of ML workload (inference/training/mixed)
- `environment` (str): Execution context ('cloud', 'edge', etc.)
- `enable_specialized_accelerators` (bool): Detect AVX, MKL, NEON, etc.
- `memory_reservation_percent` (float): Reserve part of RAM
- `power_efficiency` (bool): Whether to reduce usage for energy
- `resilience_level` (int): How fault-tolerant configs should be
- `auto_tune` (bool): If true, dynamically adjust configs using system load

---

## Functions

### `save_configs()`
```python
def save_configs(self, config_id: Optional[str] = None) -> Dict[str, str]:
```
- **Description**: Saves all auto-optimized configs to disk, including system snapshot
- **Returns**: Dictionary with config filenames

---

### `load_configs()`
```python
def load_configs(self, config_id: str) -> Dict[str, Any]:
```
- **Description**: Loads configurations from disk given an ID
- **Returns**: Dict of configs including system_info and subcomponent configs

---

### `get_system_info()`
```python
def get_system_info(self) -> Dict[str, Any]:
```
- **Description**: Returns system hardware and software metadata

---

### `auto_tune_configs()`
```python
def auto_tune_configs(self, workload_sample: Any = None) -> Dict[str, Any]:
```
- **Description**: Adjusts resource allocations dynamically based on CPU & memory load

---

### Configuration Generators
Each of the following methods returns a specific config dataclass:

```python
def get_optimal_quantization_config(self) -> QuantizationConfig
```
```python
def get_optimal_preprocessor_config(self) -> PreprocessorConfig
```
```python
def get_optimal_batch_processor_config(self) -> BatchProcessorConfig
```
```python
def get_optimal_inference_engine_config(self) -> InferenceEngineConfig
```
```python
def get_optimal_training_engine_config(self) -> MLTrainingEngineConfig
```

---

## Helper Functions

### `create_optimized_configs()`
- **Description**: Main entry point for generating and saving configurations
- **Returns**: Dictionary with file paths of all saved configs

### `create_configs_for_all_modes()`
- **Description**: Generates configs for every `OptimizationMode` variant

### `load_saved_configs()`
- **Description**: Loads configuration files using a given `config_id`

### `get_system_information()`
- **Description**: Retrieves system metrics without saving anything

### `optimize_for_environment()`
- **Description**: Convenience function to generate configs for 'edge', 'cloud', etc.

### `optimize_for_workload()`
- **Description**: Same as above, but by workload type (e.g., 'inference')

### `apply_configs_to_pipeline()`
- **Description**: Placeholder for real integration with the actual ML pipeline

### `get_default_config()`
- **Description**: Returns configs + system info for default settings, saves if given
