# Model Manager API (`modules/api/model_manager_api.py`)

## Overview

The Model Manager API provides a secure RESTful interface for comprehensive model lifecycle management. It offers advanced features including model encryption, versioning, metadata management, and secure storage with enterprise-grade security features.

## Features

- **Secure Model Management**: Advanced encryption and authentication
- **Model Lifecycle**: Complete model versioning and deployment management
- **Metadata Management**: Comprehensive model information tracking
- **Model Registry**: Centralized model storage and discovery
- **Security Features**: Encryption, digital signatures, and access control
- **Performance Tracking**: Model performance monitoring and comparison
- **Backup and Recovery**: Automated backup and disaster recovery
- **Multi-Format Support**: Support for various ML frameworks and formats

## API Configuration

```python
# Environment Variables
MODEL_MANAGER_API_HOST=0.0.0.0
MODEL_MANAGER_API_PORT=8004
API_KEYS=key1,key2,key3
MODEL_PATH=./models
JWT_SECRET=your-secret-key
ENABLE_ENCRYPTION=True
BACKUP_ENABLED=True
BACKUP_INTERVAL_HOURS=24
```

## Data Models

### ManagerConfigModel
```python
{
    "model_path": "./models/project1",            # Path for model storage
    "task_type": "regression",                    # regression, classification, clustering, anomaly_detection
    "enable_encryption": true,                    # Enable model encryption
    "key_iterations": 200000,                     # Key derivation iterations
    "hash_algorithm": "sha512",                   # Hash algorithm for passwords
    "use_scrypt": true,                          # Use Scrypt instead of PBKDF2
    "enable_quantization": false,                # Enable model quantization
    "primary_metric": "mse"                      # Primary evaluation metric
}
```

### ModelSaveRequest
```python
{
    "model_name": "my_model_v1",                 # Unique model name
    "model_data": "base64_encoded_model_data",   # Serialized model data
    "password": "secure_password",               # Encryption password
    "metadata": {                                # Model metadata
        "version": "1.0.0",
        "framework": "sklearn",
        "algorithm": "RandomForest",
        "performance_metrics": {
            "accuracy": 0.95,
            "f1_score": 0.93
        },
        "training_date": "2025-01-15",
        "author": "Data Scientist",
        "description": "Production model for classification"
    },
    "tags": ["production", "v1", "classification"],
    "replace_existing": false
}
```

### ModelLoadRequest
```python
{
    "model_name": "my_model_v1",                 # Model name to load
    "password": "secure_password",               # Decryption password
    "verify_integrity": true                     # Verify model integrity
}
```

## API Endpoints

### Manager Lifecycle

#### Create Manager
```http
POST /api/model-manager/create
Content-Type: application/json
X-API-Key: your-api-key

{
    "manager_id": "manager_prod",
    "config": {
        "model_path": "./models/production",
        "task_type": "classification",
        "enable_encryption": true,
        "use_scrypt": true,
        "primary_metric": "f1_score"
    }
}
```

**Response:**
```json
{
    "message": "Manager created successfully",
    "manager_id": "manager_prod",
    "config": {...},
    "created_at": "2025-01-15T10:30:00Z",
    "storage_path": "./models/production"
}
```

#### Get Manager Info
```http
GET /api/model-manager/{manager_id}/info
X-API-Key: your-api-key
```

**Response:**
```json
{
    "manager_id": "manager_prod",
    "config": {...},
    "created_at": "2025-01-15T10:30:00Z",
    "model_count": 5,
    "total_size_mb": 125.6,
    "last_backup": "2025-01-15T09:00:00Z",
    "encryption_enabled": true
}
```

#### List Managers
```http
GET /api/model-manager/list
X-API-Key: your-api-key
```

**Response:**
```json
{
    "managers": [
        {
            "manager_id": "manager_prod",
            "task_type": "classification",
            "model_count": 5,
            "created_at": "2025-01-15T10:30:00Z"
        }
    ],
    "total_managers": 1
}
```

#### Delete Manager
```http
DELETE /api/model-manager/{manager_id}
X-API-Key: your-api-key
```

### Model Operations

#### Save Model
```http
POST /api/model-manager/{manager_id}/models/save
Content-Type: application/json
X-API-Key: your-api-key

{
    "model_name": "classifier_v2.1",
    "model_data": "base64_encoded_pickle_data",
    "password": "secure_password_123",
    "metadata": {
        "version": "2.1.0",
        "framework": "sklearn",
        "algorithm": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        "performance_metrics": {
            "accuracy": 0.96,
            "precision": 0.94,
            "recall": 0.95,
            "f1_score": 0.945
        },
        "training_data_hash": "sha256_hash_of_training_data",
        "feature_importance": {...},
        "training_time_seconds": 45.6,
        "author": "john.doe@company.com",
        "description": "Improved classifier with better feature engineering"
    },
    "tags": ["production", "v2.1", "high-accuracy"]
}
```

**Response:**
```json
{
    "message": "Model saved successfully",
    "model_name": "classifier_v2.1",
    "model_id": "model_12345",
    "encrypted": true,
    "file_size_mb": 15.6,
    "hash": "sha256_model_hash",
    "saved_at": "2025-01-15T10:30:00Z",
    "backup_created": true
}
```

#### Load Model
```http
POST /api/model-manager/{manager_id}/models/load
Content-Type: application/json
X-API-Key: your-api-key

{
    "model_name": "classifier_v2.1",
    "password": "secure_password_123",
    "verify_integrity": true
}
```

**Response:**
```json
{
    "message": "Model loaded successfully",
    "model_name": "classifier_v2.1",
    "model_data": "base64_encoded_model_data",
    "metadata": {...},
    "integrity_verified": true,
    "load_time": 0.234,
    "loaded_at": "2025-01-15T10:30:00Z"
}
```

#### Get Model Info
```http
GET /api/model-manager/{manager_id}/models/{model_name}/info
X-API-Key: your-api-key
```

**Response:**
```json
{
    "model_name": "classifier_v2.1",
    "model_id": "model_12345",
    "metadata": {...},
    "file_info": {
        "size_mb": 15.6,
        "created_at": "2025-01-15T10:30:00Z",
        "modified_at": "2025-01-15T10:30:00Z",
        "hash": "sha256_model_hash"
    },
    "encryption_info": {
        "encrypted": true,
        "algorithm": "AES-256-GCM",
        "key_derivation": "scrypt"
    },
    "access_info": {
        "access_count": 15,
        "last_accessed": "2025-01-15T10:25:00Z"
    }
}
```

#### List Models
```http
GET /api/model-manager/{manager_id}/models/list
X-API-Key: your-api-key
```

**Response:**
```json
{
    "models": [
        {
            "model_name": "classifier_v2.1",
            "model_id": "model_12345",
            "version": "2.1.0",
            "framework": "sklearn",
            "size_mb": 15.6,
            "created_at": "2025-01-15T10:30:00Z",
            "tags": ["production", "v2.1", "high-accuracy"],
            "performance_metrics": {
                "accuracy": 0.96,
                "f1_score": 0.945
            }
        }
    ],
    "total_models": 1,
    "total_size_mb": 15.6
}
```

#### Delete Model
```http
DELETE /api/model-manager/{manager_id}/models/{model_name}
Content-Type: application/json
X-API-Key: your-api-key

{
    "confirm_deletion": true,
    "backup_before_delete": true
}
```

**Response:**
```json
{
    "message": "Model deleted successfully",
    "model_name": "classifier_v2.1",
    "backup_created": true,
    "backup_path": "./backups/classifier_v2.1_20250115_103000.backup"
}
```

### Model Versioning

#### List Model Versions
```http
GET /api/model-manager/{manager_id}/models/{model_name}/versions
X-API-Key: your-api-key
```

**Response:**
```json
{
    "model_name": "classifier",
    "versions": [
        {
            "version": "2.1.0",
            "model_name": "classifier_v2.1",
            "created_at": "2025-01-15T10:30:00Z",
            "performance_metrics": {
                "accuracy": 0.96,
                "f1_score": 0.945
            },
            "is_active": true
        },
        {
            "version": "2.0.0",
            "model_name": "classifier_v2.0",
            "created_at": "2025-01-10T15:20:00Z",
            "performance_metrics": {
                "accuracy": 0.94,
                "f1_score": 0.925
            },
            "is_active": false
        }
    ],
    "total_versions": 2
}
```

#### Compare Models
```http
POST /api/model-manager/{manager_id}/models/compare
Content-Type: application/json
X-API-Key: your-api-key

{
    "model_names": ["classifier_v2.1", "classifier_v2.0"],
    "comparison_metrics": ["accuracy", "f1_score", "precision", "recall"]
}
```

**Response:**
```json
{
    "comparison": {
        "classifier_v2.1": {
            "accuracy": 0.96,
            "f1_score": 0.945,
            "precision": 0.94,
            "recall": 0.95,
            "size_mb": 15.6,
            "training_time": 45.6
        },
        "classifier_v2.0": {
            "accuracy": 0.94,
            "f1_score": 0.925,
            "precision": 0.92,
            "recall": 0.93,
            "size_mb": 14.2,
            "training_time": 38.4
        }
    },
    "best_model": {
        "by_accuracy": "classifier_v2.1",
        "by_f1_score": "classifier_v2.1",
        "by_size": "classifier_v2.0"
    },
    "improvements": {
        "accuracy": 0.02,
        "f1_score": 0.02
    }
}
```

### Model Search and Discovery

#### Search Models
```http
GET /api/model-manager/{manager_id}/models/search
X-API-Key: your-api-key
?query=classification
&framework=sklearn
&min_accuracy=0.9
&tags=production
&sort_by=accuracy
&sort_order=desc
&limit=10
```

**Response:**
```json
{
    "models": [
        {
            "model_name": "classifier_v2.1",
            "relevance_score": 0.95,
            "metadata": {...},
            "match_criteria": ["framework", "tags", "accuracy"]
        }
    ],
    "total_results": 1,
    "query_time": 0.045
}
```

#### Get Model by Tags
```http
GET /api/model-manager/{manager_id}/models/by-tags
X-API-Key: your-api-key
?tags=production,high-accuracy
```

#### Get Best Models
```http
GET /api/model-manager/{manager_id}/models/best
X-API-Key: your-api-key
?metric=f1_score
&limit=5
```

**Response:**
```json
{
    "best_models": [
        {
            "model_name": "classifier_v2.1",
            "f1_score": 0.945,
            "rank": 1
        },
        {
            "model_name": "classifier_v2.0",
            "f1_score": 0.925,
            "rank": 2
        }
    ],
    "metric": "f1_score",
    "total_models": 2
}
```

### Backup and Recovery

#### Create Backup
```http
POST /api/model-manager/{manager_id}/backup/create
Content-Type: application/json
X-API-Key: your-api-key

{
    "backup_name": "daily_backup_20250115",
    "include_models": ["classifier_v2.1", "regressor_v1.0"],
    "compress": true,
    "encrypt_backup": true
}
```

**Response:**
```json
{
    "message": "Backup created successfully",
    "backup_name": "daily_backup_20250115",
    "backup_path": "./backups/daily_backup_20250115.tar.gz",
    "backup_size_mb": 45.2,
    "models_included": 2,
    "created_at": "2025-01-15T10:30:00Z"
}
```

#### List Backups
```http
GET /api/model-manager/{manager_id}/backup/list
X-API-Key: your-api-key
```

**Response:**
```json
{
    "backups": [
        {
            "backup_name": "daily_backup_20250115",
            "created_at": "2025-01-15T10:30:00Z",
            "size_mb": 45.2,
            "model_count": 2,
            "encrypted": true
        }
    ],
    "total_backups": 1
}
```

#### Restore from Backup
```http
POST /api/model-manager/{manager_id}/backup/restore
Content-Type: application/json
X-API-Key: your-api-key

{
    "backup_name": "daily_backup_20250115",
    "restore_password": "backup_password",
    "overwrite_existing": false,
    "models_to_restore": ["classifier_v2.1"]
}
```

### Security and Authentication

#### Update Manager Password
```http
PUT /api/model-manager/{manager_id}/security/password
Content-Type: application/json
X-API-Key: your-api-key

{
    "old_password": "old_secure_password",
    "new_password": "new_secure_password",
    "rotate_model_keys": true
}
```

#### Get Security Status
```http
GET /api/model-manager/{manager_id}/security/status
X-API-Key: your-api-key
```

**Response:**
```json
{
    "encryption_enabled": true,
    "encryption_algorithm": "AES-256-GCM",
    "key_derivation": "scrypt",
    "models_encrypted": 5,
    "models_unencrypted": 0,
    "last_key_rotation": "2025-01-10T12:00:00Z",
    "backup_encryption": true
}
```

#### Audit Trail
```http
GET /api/model-manager/{manager_id}/audit/trail
X-API-Key: your-api-key
?start_date=2025-01-01
&end_date=2025-01-15
&action_type=model_access
```

**Response:**
```json
{
    "audit_entries": [
        {
            "timestamp": "2025-01-15T10:30:00Z",
            "action": "model_load",
            "model_name": "classifier_v2.1",
            "user_id": "api_key_hash",
            "ip_address": "192.168.1.100",
            "success": true
        }
    ],
    "total_entries": 1
}
```

### Performance and Monitoring

#### Get Manager Statistics
```http
GET /api/model-manager/{manager_id}/stats
X-API-Key: your-api-key
```

**Response:**
```json
{
    "storage_stats": {
        "total_models": 5,
        "total_size_mb": 125.6,
        "average_model_size_mb": 25.1,
        "encrypted_models": 5,
        "compressed_models": 3
    },
    "access_stats": {
        "total_accesses": 150,
        "unique_models_accessed": 3,
        "most_accessed_model": "classifier_v2.1",
        "last_access": "2025-01-15T10:25:00Z"
    },
    "performance_stats": {
        "average_save_time": 0.456,
        "average_load_time": 0.234,
        "cache_hit_rate": 0.85
    }
}
```

#### Health Check
```http
GET /api/model-manager/{manager_id}/health
X-API-Key: your-api-key
```

**Response:**
```json
{
    "status": "healthy",
    "checks": {
        "storage_accessible": true,
        "encryption_working": true,
        "backup_system": true,
        "model_integrity": true
    },
    "last_check": "2025-01-15T10:30:00Z",
    "uptime": "5d 12h 30m"
}
```

## Usage Examples

### Basic Model Management

```python
import requests
import base64
import pickle

# API configuration
API_BASE = "http://localhost:8004"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# 1. Create manager
manager_config = {
    "manager_id": "production_manager",
    "config": {
        "model_path": "./models/production",
        "task_type": "classification",
        "enable_encryption": True,
        "primary_metric": "f1_score"
    }
}
response = requests.post(f"{API_BASE}/api/model-manager/create",
                        headers=headers, json=manager_config)
manager_id = manager_config["manager_id"]

# 2. Save a model
model = train_your_model()  # Your trained model
model_data = base64.b64encode(pickle.dumps(model)).decode()

save_request = {
    "model_name": "classifier_v1.0",
    "model_data": model_data,
    "password": "secure_password_123",
    "metadata": {
        "version": "1.0.0",
        "framework": "sklearn",
        "algorithm": "RandomForestClassifier",
        "performance_metrics": {
            "accuracy": 0.95,
            "f1_score": 0.93
        }
    },
    "tags": ["production", "v1.0"]
}
response = requests.post(f"{API_BASE}/api/model-manager/{manager_id}/models/save",
                        headers=headers, json=save_request)
print("Model saved:", response.json())

# 3. Load the model
load_request = {
    "model_name": "classifier_v1.0",
    "password": "secure_password_123",
    "verify_integrity": True
}
response = requests.post(f"{API_BASE}/api/model-manager/{manager_id}/models/load",
                        headers=headers, json=load_request)
model_data = response.json()["model_data"]
loaded_model = pickle.loads(base64.b64decode(model_data))
```

### Model Versioning and Comparison

```python
# Save multiple versions
versions = ["1.0", "1.1", "2.0"]
for version in versions:
    model = train_model_version(version)
    model_data = base64.b64encode(pickle.dumps(model)).decode()
    
    save_request = {
        "model_name": f"classifier_v{version}",
        "model_data": model_data,
        "password": "secure_password_123",
        "metadata": {
            "version": version,
            "framework": "sklearn",
            "performance_metrics": get_performance_metrics(model)
        }
    }
    requests.post(f"{API_BASE}/api/model-manager/{manager_id}/models/save",
                 headers=headers, json=save_request)

# Compare model versions
comparison_request = {
    "model_names": ["classifier_v1.0", "classifier_v1.1", "classifier_v2.0"],
    "comparison_metrics": ["accuracy", "f1_score", "precision", "recall"]
}
response = requests.post(f"{API_BASE}/api/model-manager/{manager_id}/models/compare",
                        headers=headers, json=comparison_request)
comparison = response.json()

print("Best model by accuracy:", comparison["best_model"]["by_accuracy"])
print("Performance comparison:", comparison["comparison"])
```

### Model Search and Discovery

```python
# Search for production models
response = requests.get(
    f"{API_BASE}/api/model-manager/{manager_id}/models/search",
    headers={"X-API-Key": API_KEY},
    params={
        "query": "classification",
        "tags": "production", 
        "min_accuracy": 0.9,
        "sort_by": "f1_score",
        "sort_order": "desc"
    }
)
search_results = response.json()

# Get best performing models
response = requests.get(
    f"{API_BASE}/api/model-manager/{manager_id}/models/best",
    headers={"X-API-Key": API_KEY},
    params={"metric": "f1_score", "limit": 3}
)
best_models = response.json()["best_models"]
```

### Backup and Recovery

```python
# Create backup
backup_request = {
    "backup_name": f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "include_models": ["classifier_v2.0", "regressor_v1.5"],
    "compress": True,
    "encrypt_backup": True
}
response = requests.post(f"{API_BASE}/api/model-manager/{manager_id}/backup/create",
                        headers=headers, json=backup_request)
backup_info = response.json()

# List all backups
response = requests.get(f"{API_BASE}/api/model-manager/{manager_id}/backup/list",
                       headers={"X-API-Key": API_KEY})
backups = response.json()["backups"]

# Restore from backup if needed
restore_request = {
    "backup_name": backup_info["backup_name"],
    "restore_password": "backup_password",
    "overwrite_existing": False
}
response = requests.post(f"{API_BASE}/api/model-manager/{manager_id}/backup/restore",
                        headers=headers, json=restore_request)
```

## Error Handling

### Common Error Codes

- **400 Bad Request**: Invalid request parameters or malformed data
- **401 Unauthorized**: Missing or invalid API key
- **403 Forbidden**: Insufficient permissions or wrong password
- **404 Not Found**: Manager or model not found
- **409 Conflict**: Model already exists or version conflict
- **422 Unprocessable Entity**: Model validation or encryption errors
- **500 Internal Server Error**: Storage or system errors

### Error Response Format

```json
{
    "error": "ModelNotFoundError",
    "message": "Model 'classifier_v1.0' not found in manager 'production_manager'",
    "details": {
        "manager_id": "production_manager",
        "model_name": "classifier_v1.0",
        "available_models": ["classifier_v2.0", "regressor_v1.0"]
    },
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_12345"
}
```

## Best Practices

### Security

1. **Strong Passwords**: Use strong, unique passwords for model encryption
2. **API Key Management**: Rotate API keys regularly
3. **Access Control**: Implement proper access controls in production
4. **Audit Logging**: Monitor and audit all model access
5. **Backup Encryption**: Always encrypt backups

### Performance

1. **Model Size**: Optimize model size for faster loading
2. **Caching**: Implement model caching for frequently accessed models
3. **Compression**: Use compression for large models
4. **Batch Operations**: Use batch operations for multiple models

### Organization

1. **Naming Conventions**: Use consistent model naming conventions
2. **Version Control**: Implement semantic versioning for models
3. **Metadata**: Include comprehensive metadata for searchability
4. **Tags**: Use tags for model organization and filtering
5. **Documentation**: Document model purpose and usage

### Backup Strategy

1. **Regular Backups**: Schedule regular automated backups
2. **Incremental Backups**: Use incremental backups for efficiency
3. **Off-site Storage**: Store backups in separate locations
4. **Recovery Testing**: Regularly test backup recovery procedures

## Advanced Features

### Custom Encryption

Support for custom encryption algorithms:

```python
config = {
    "encryption_algorithm": "ChaCha20Poly1305",
    "custom_key_derivation": {
        "algorithm": "Argon2",
        "memory_cost": 65536,
        "time_cost": 3
    }
}
```

### Model Signing

Digital signatures for model integrity:

```python
save_request = {
    "model_name": "signed_model",
    "model_data": model_data,
    "digital_signature": True,
    "signing_key": "private_key_pem"
}
```

### Model Deployment Integration

Integration with deployment systems:

```python
deployment_config = {
    "deploy_to": "kubernetes",
    "namespace": "ml-models",
    "service_name": "classifier-api",
    "auto_deploy": True
}
```

## Related Documentation

- [Secure Model Manager](../model_manager.md) - Core model manager
- [Configuration System](../configs.md) - Configuration management
- [Inference Engine API](inference_engine_api.md) - Model inference API
- [Train Engine API](train_engine_api.md) - Model training API

---

*The Model Manager API provides enterprise-grade model lifecycle management with advanced security, versioning, and organizational features for production ML systems.*
