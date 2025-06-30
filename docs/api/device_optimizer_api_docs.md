# CPU Device Optimizer API

## Overview
The CPU Device Optimizer API provides RESTful endpoints to detect hardware capabilities, generate optimal configurations, and optimize machine learning pipelines based on the underlying system. It supports various optimization strategies tailored to different environments (cloud, desktop, edge) and workload types (inference, training, mixed).

## Prerequisites
- Python ≥3.7
- FastAPI
- DeviceOptimizer module and dependencies
- Required packages:
  ```bash
  pip install fastapi uvicorn pydantic
  ```

## Installation
```bash
# Clone the repository
git clone https://github.com/example/cpu-device-optimizer-api

# Install dependencies
pip install -r requirements.txt

# Set API key (for production)
export API_KEY=your_secure_api_key
```

## Usage
```bash
# Start the API server
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Configuration
| Env Variable | Default | Description |
|--------------|---------|-------------|
| `API_KEY` | `dev_key_12345` | API key for authentication |

## Architecture
The API is built on FastAPI and provides endpoints for:
1. System information detection
2. Configuration generation for different optimization modes
3. Environment-specific optimizations
4. Workload-specific optimizations
5. Configuration management (loading, applying, listing, deleting)

---

## API Endpoints

### Root Endpoint

#### `GET /`

Provides basic information about the API.

- **Parameters**: None
- **Returns**: 
  - Information about the API, version, and description
- **Example Response**:
  ```json
  {
    "api": "CPU Device Optimizer API",
    "version": "1.0.0",
    "description": "API for hardware detection, configuration generation, and optimization"
  }
  ```

### System Information

#### `GET /system-info`

Retrieves comprehensive information about the current system hardware capabilities.

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - `enable_specialized_accelerators` (query, optional): Whether to detect specialized hardware. Default: `true`
- **Returns**: 
  - Detailed system information including CPU, memory, and accelerator details
- **Errors**:
  - `401 Unauthorized`: Invalid API Key
  - `500 Internal Server Error`: Error getting system information

### Configuration Generation

#### `POST /optimize`

Generates and saves optimized configurations based on device capabilities.

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - Request body (required):
    ```json
    {
      "config_path": "./configs",
      "checkpoint_path": "./checkpoints",
      "model_registry_path": "./model_registry",
      "optimization_mode": "BALANCED",
      "workload_type": "mixed",
      "environment": "auto",
      "enable_specialized_accelerators": true,
      "memory_reservation_percent": 10.0,
      "power_efficiency": false,
      "resilience_level": 1,
      "auto_tune": true,
      "config_id": null
    }
    ```
- **Returns**:
  - Status message and generated master configuration
- **Errors**:
  - `401 Unauthorized`: Invalid API Key
  - `500 Internal Server Error`: Failed to create optimized configurations

#### `POST /optimize/all-modes`

Generates and saves configurations for all optimization modes (BALANCED, PERFORMANCE, MEMORY_SAVING, etc.).

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - Request body (required): Same as `/optimize` endpoint
- **Returns**:
  - Status message and all generated configurations
- **Errors**:
  - `401 Unauthorized`: Invalid API Key
  - `500 Internal Server Error`: Failed to create configurations for all modes

#### `POST /optimize/environment/{environment}`

Creates optimized configurations for a specific environment type.

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - `environment` (path, required): Target environment (`cloud`, `desktop`, or `edge`)
- **Returns**:
  - Status message and optimized configuration for the specified environment
- **Errors**:
  - `400 Bad Request`: Invalid environment
  - `401 Unauthorized`: Invalid API Key
  - `500 Internal Server Error`: Failed to create optimized configurations for environment

#### `POST /optimize/workload/{workload_type}`

Creates optimized configurations for a specific workload type.

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - `workload_type` (path, required): Target workload type (`inference`, `training`, or `mixed`)
- **Returns**:
  - Status message and optimized configuration for the specified workload
- **Errors**:
  - `400 Bad Request`: Invalid workload type
  - `401 Unauthorized`: Invalid API Key
  - `500 Internal Server Error`: Failed to create optimized configurations for workload

### Configuration Management

#### `POST /configs/load`

Loads previously saved configurations.

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - Request body (required):
    ```json
    {
      "config_path": "./configs",
      "config_id": "your-config-id"
    }
    ```
- **Returns**:
  - Status message and loaded configurations
- **Errors**:
  - `401 Unauthorized`: Invalid API Key
  - `404 Not Found`: Configuration not found
  - `500 Internal Server Error`: Failed to load configurations

#### `POST /configs/apply`

Applies loaded configurations to ML pipeline components.

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - Request body (required):
    ```json
    {
      "configs": {
        "component1": {"param1": "value1"},
        "component2": {"param2": "value2"}
      }
    }
    ```
- **Returns**:
  - Status message indicating successful application
- **Errors**:
  - `401 Unauthorized`: Invalid API Key
  - `500 Internal Server Error`: Failed to apply configurations

#### `POST /configs/default`

Gets a set of default configurations optimized for the current system.

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - Request body (required):
    ```json
    {
      "optimization_mode": "BALANCED",
      "workload_type": "mixed",
      "environment": "auto",
      "output_dir": "./configs/default",
      "enable_specialized_accelerators": true
    }
    ```
- **Returns**:
  - Status message and default configurations
- **Errors**:
  - `401 Unauthorized`: Invalid API Key
  - `500 Internal Server Error`: Failed to get default configurations

#### `GET /configs/list`

Lists all available configuration sets.

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - `config_path` (query, optional): Path where configuration files are stored. Default: `./configs`
- **Returns**:
  - List of available configurations with their IDs and creation dates
- **Errors**:
  - `401 Unauthorized`: Invalid API Key
  - `500 Internal Server Error`: Failed to list configurations

#### `DELETE /configs/{config_id}`

Deletes a specific configuration set.

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - `config_id` (path, required): Identifier for the configuration set to delete
  - `config_path` (query, optional): Path where configuration files are stored. Default: `./configs`
- **Returns**:
  - Status message indicating successful deletion
- **Errors**:
  - `401 Unauthorized`: Invalid API Key
  - `404 Not Found`: Configuration not found
  - `500 Internal Server Error`: Failed to delete configuration

### Maintenance

#### `POST /maintenance/cleanup`

Schedules a background task to clean up old configuration files.

- **Parameters**:
  - `api_key` (query, required): API Key for authentication
  - `older_than_days` (query, optional): Delete configurations older than this many days. Default: `30`
  - `config_path` (query, optional): Path where configuration files are stored. Default: `./configs`
- **Returns**:
  - Status message indicating scheduled cleanup
- **Errors**:
  - `401 Unauthorized`: Invalid API Key
  - `500 Internal Server Error`: Failed to schedule cleanup task

---

## Data Models

### Environment (Enum)
- `auto`: Automatic environment detection
- `cloud`: Cloud environment
- `desktop`: Desktop environment
- `edge`: Edge device environment

### WorkloadType (Enum)
- `mixed`: Mixed workloads
- `inference`: Inference-focused workloads
- `training`: Training-focused workloads

### OptimizerRequest
```python
{
    "config_path": str,            # Path to save configuration files
    "checkpoint_path": str,        # Path for model checkpoints
    "model_registry_path": str,    # Path for model registry
    "optimization_mode": str,      # Mode for optimization strategy
    "workload_type": WorkloadType, # Type of workload
    "environment": Environment,    # Computing environment
    "enable_specialized_accelerators": bool, # Whether to enable detection of specialized hardware
    "memory_reservation_percent": float,     # Percentage of memory to reserve for the system
    "power_efficiency": bool,      # Whether to optimize for power efficiency
    "resilience_level": int,       # Level of fault tolerance
    "auto_tune": bool,             # Whether to enable automatic parameter tuning
    "config_id": Optional[str]     # Optional identifier for the configuration set
}
```

### SystemInfoRequest
```python
{
    "enable_specialized_accelerators": bool  # Whether to detect specialized hardware
}
```

### LoadConfigRequest
```python
{
    "config_path": str,  # Path where configuration files are stored
    "config_id": str     # Identifier for the configuration set
}
```

### ApplyConfigRequest
```python
{
    "configs": Dict[str, Any]  # Dictionary with configurations to apply
}
```

### DefaultConfigRequest
```python
{
    "optimization_mode": str,        # The optimization strategy to use
    "workload_type": WorkloadType,   # Type of workload to optimize for
    "environment": Environment,      # Computing environment
    "output_dir": str,               # Directory where configuration files will be saved
    "enable_specialized_accelerators": bool  # Whether to enable detection of specialized hardware
}
```

## Security
- The API uses a simple API key authentication mechanism
- For production use, implement a more robust authentication method
- All endpoints require an API key validation

## Testing
```bash
# Run tests
pytest tests/
```

## Security & Compliance
- API key authentication should be replaced with OAuth2 or JWT for production
- Consider TLS/HTTPS for data in transit
- Implement proper access control for sensitive system information

> Last Updated: 2025-05-11