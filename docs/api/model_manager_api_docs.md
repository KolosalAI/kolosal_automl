# Secure Model Manager API

## Overview
The Secure Model Manager API provides a robust interface for managing machine learning models with advanced security features. It allows users to create manager instances, save and load models, verify model integrity, and handle model encryption with security best practices.

## Prerequisites
- Python 3.6+
- FastAPI
- Uvicorn
- Dependencies from the modules package (SecureModelManager, TaskType)

## Installation
```bash
# Install required packages
pip install fastapi uvicorn pydantic

# Clone the repository (if applicable)
git clone https://github.com/your-org/secure-model-manager.git
cd secure-model-manager

# Set environment variables (recommended for production)
export API_KEYS="your_key_1,your_key_2"
export MODEL_PATH="/path/to/models"
export JWT_SECRET="your-secure-jwt-secret"
```

## Usage
```bash
# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000

# Or run directly from the script
python main.py
```

## Configuration
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `API_KEYS` | `"dev_key"` | Comma-separated list of valid API keys |
| `MODEL_PATH` | `"./models"` | Default path where models will be stored |
| `JWT_SECRET` | `"change-this-in-production"` | Secret key for JWT token validation |

## Security
The API implements two security mechanisms:
1. **API Key Authentication**: Required for manager creation and administrative endpoints
2. **Bearer Token Authentication**: Required for model operations

## API Endpoints

### Manager Operations

#### Create Manager
```
POST /api/managers
```
Creates a new model manager instance with the specified configuration.

**Authentication**: API Key (X-API-Key header)

**Request Body**:
```json
{
  "model_path": "./models/project1",
  "task_type": "regression",
  "enable_encryption": true,
  "use_scrypt": true,
  "primary_metric": "mse"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Manager created with ID: {manager_id}",
  "details": {
    "manager_id": "12345678-1234-5678-1234-567812345678"
  }
}
```

#### List Managers
```
GET /api/managers
```
Lists all available model manager instances.

**Authentication**: API Key (X-API-Key header)

**Response**:
```json
{
  "manager_id_1": {
    "encryption_enabled": true,
    "model_path": "./models/project1",
    "task_type": "REGRESSION",
    "models_count": 2,
    "best_model": "model_name"
  },
  "manager_id_2": {
    "encryption_enabled": false,
    "model_path": "./models/project2",
    "task_type": "CLASSIFICATION",
    "models_count": 1,
    "best_model": null
  }
}
```

#### Delete Manager
```
DELETE /api/managers/{manager_id}
```
Deletes a model manager instance.

**Authentication**: API Key (X-API-Key header)

**Parameters**:
- `manager_id` (path): ID of the manager to delete

**Response**:
```json
{
  "success": true,
  "message": "Manager {manager_id} deleted successfully",
  "details": {}
}
```

### Model Operations

#### Save Model
```
POST /api/managers/{manager_id}/models/save
```
Saves a model using the specified manager.

**Authentication**: Bearer Token

**Parameters**:
- `manager_id` (path): ID of the manager to use

**Request Body**:
```json
{
  "model_name": "random_forest_v1",
  "filepath": null,
  "access_code": "secure_password",
  "compression_level": 5
}
```

**Response**:
```json
{
  "success": true,
  "message": "Model random_forest_v1 saved successfully",
  "details": {
    "model_name": "random_forest_v1",
    "filepath": "./models/project1/random_forest_v1.pkl",
    "encrypted": true
  }
}
```

#### Load Model
```
POST /api/managers/{manager_id}/models/load
```
Loads a model using the specified manager.

**Authentication**: Bearer Token

**Parameters**:
- `manager_id` (path): ID of the manager to use

**Request Body**:
```json
{
  "filepath": "./models/random_forest_v1.pkl",
  "access_code": "secure_password"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Model random_forest_v1 loaded successfully",
  "details": {
    "model_name": "random_forest_v1",
    "is_best_model": true
  }
}
```

#### List Models
```
GET /api/managers/{manager_id}/models
```
Lists all models in the specified manager.

**Authentication**: Bearer Token

**Parameters**:
- `manager_id` (path): ID of the manager to use

**Response**:
```json
{
  "models": ["random_forest_v1", "xgboost_v2"],
  "best_model": "random_forest_v1"
}
```

#### Verify Model
```
POST /api/managers/{manager_id}/verify
```
Verifies the integrity of a model file.

**Authentication**: Bearer Token

**Parameters**:
- `manager_id` (path): ID of the manager to use
- `filepath` (query): Path to the model file to verify

**Response**:
```json
{
  "filepath": "./models/random_forest_v1.pkl",
  "is_valid": true,
  "encryption_status": "encrypted"
}
```

#### Rotate Encryption Key
```
POST /api/managers/{manager_id}/rotate-key
```
Rotates the encryption key for a manager.

**Authentication**: Bearer Token

**Parameters**:
- `manager_id` (path): ID of the manager to use

**Request Body**:
```json
{
  "new_password": "new_secure_password"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Encryption key rotated successfully",
  "details": {
    "using_password": true,
    "timestamp": 1651234567
  }
}
```

#### Upload Model
```
POST /api/managers/{manager_id}/upload-model
```
Uploads a model file and loads it using the specified manager.

**Authentication**: Bearer Token

**Parameters**:
- `manager_id` (path): ID of the manager to use
- `model_file` (form): The model file to upload
- `access_code` (query, optional): Access code if the model is protected

**Response**:
```json
{
  "success": true,
  "message": "Model model_name_1651234567.pkl uploaded successfully and queued for loading",
  "details": {
    "filepath": "./models/project1/model_name_1651234567.pkl",
    "model_name": "model_name"
  }
}
```

### Health and Info Endpoints

#### Health Check
```
GET /health
```
Provides a health check status for the API.

**Authentication**: None

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1651234567,
  "version": "1.0.0"
}
```

#### API Information
```
GET /
```
Provides general information about the API.

**Authentication**: None

**Response**:
```json
{
  "name": "Secure Model Manager API",
  "version": "1.0.0",
  "description": "API for managing machine learning models with advanced security features",
  "docs_url": "/docs",
  "total_managers": 2
}
```

## Data Models

### ManagerConfigModel
Configuration for creating a new model manager.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_path` | string | No | `DEFAULT_MODEL_PATH` | Path where models will be stored |
| `task_type` | string (enum) | Yes | - | Type of ML task (regression, classification, clustering, anomaly_detection) |
| `enable_encryption` | boolean | No | `true` | Enable model encryption |
| `key_iterations` | integer | No | `200000` | Key derivation iterations |
| `hash_algorithm` | string | No | `"sha512"` | Hash algorithm for password derivation |
| `use_scrypt` | boolean | No | `true` | Use Scrypt instead of PBKDF2 |
| `enable_quantization` | boolean | No | `false` | Enable model quantization |
| `primary_metric` | string | No | `null` | Primary metric for model evaluation |

### ModelSaveRequest
Request to save a model.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_name` | string | Yes | - | Name of the model to save |
| `filepath` | string | No | `null` | Custom filepath to save the model |
| `access_code` | string | No | `null` | Access code to secure the model |
| `compression_level` | integer | No | `5` | Compression level (0-9) |

### ModelLoadRequest
Request to load a model.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filepath` | string | Yes | - | Path to the model file |
| `access_code` | string | No | `null` | Access code if the model is protected |

### RotateKeyRequest
Request to rotate the encryption key.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `new_password` | string | No | `null` | New password for key derivation. If not provided, a random key will be generated |

## Architecture
The API interacts with the SecureModelManager class, which handles model operations including encryption, compression, and integrity verification. The API server provides a RESTful interface to this functionality with robust error handling and authentication.

## Security & Compliance
- API uses both API key and Bearer token authentication
- Models can be encrypted with password protection
- Files are saved with 0600 permissions (only owner can read/write)
- JWT tokens should be properly validated in production
- CORS is configured but should be restricted to specific origins in production
- Passwords and access codes are never logged

## Testing
```bash
# Run the test suite
pytest tests/
```

> Last Updated: 2025-05-11