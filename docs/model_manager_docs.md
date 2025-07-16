# SecureModelManager Documentation

## Overview
`SecureModelManager` is a robust security-focused model management system that provides advanced encryption capabilities for machine learning models. It enables secure storage, loading, and management of models with features such as password-based encryption, integrity verification, quantization support, and access control.

The manager supports multiple encryption algorithms, automatic model versioning, and seamless integration with the kolosal AutoML training pipeline.

## Prerequisites
- Python ≥3.10
- Required packages:
  ```bash
  pip install joblib numpy cryptography
  ```
- Optional: A quantizer implementation for model compression

## Installation
Install the required dependencies:
```bash
pip install joblib numpy cryptography
```

## Usage
```python
from modules.configs import TaskType
from modules.model_manager import SecureModelManager
import joblib

# Create a configuration object
class Config:
    task_type = TaskType.CLASSIFICATION
    model_path = "./models"
    enable_encryption = True
    key_iterations = 200000
    hash_algorithm = "sha512"
    use_scrypt = True
    
config = Config()

# Initialize the secure model manager
manager = SecureModelManager(config)

# Train a simple model (example)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save a model with encryption
manager.save_model(model, "my_model", "./models/my_model.pkl")

# Load a model (will prompt for password if encrypted)
loaded_model = manager.load_model("./models/my_model.pkl")

# Verify model integrity
is_valid = manager.verify_model_integrity("./models/my_model.pkl")
print(f"Model integrity: {'Valid' if is_valid else 'Invalid'}")

# Add model to registry for tracking
manager.add_model(model, "my_model", score=0.95, metadata={"algorithm": "RandomForest"})

# Get best model
best_model = manager.get_best_model()
print(f"Best model score: {manager.best_score}")

# Rotate encryption key
manager.rotate_encryption_key("new_password")

# Export model with metadata
export_path = manager.export_model("my_model", "./exports/my_model_export.pkl")
print(f"Model exported to: {export_path}")
```

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_type` | Required | Type of ML task (REGRESSION or CLASSIFICATION) |
| `model_path` | Required | Directory path for storing models |
| `enable_encryption` | `True` | Whether to encrypt model files |
| `key_iterations` | `200000` | Number of iterations for key derivation |
| `hash_algorithm` | `sha512` | Hash algorithm for passwords and verification |
| `use_scrypt` | `True` | Whether to use Scrypt (stronger) instead of PBKDF2 |
| `enable_quantization` | - | Optional: Enable model compression |
| `quantizer` | - | Optional: Instance of a quantizer for compression |
| `primary_metric` | - | Optional: Main metric for model comparison |

## Security & Compliance
- Uses AES-128-CBC encryption via Fernet
- Provides SHA-512 based integrity verification
- Implements secure key derivation with Scrypt or PBKDF2
- Restricts file permissions to owner-only (0o600)
- Supports compliance with regulations requiring data protection

## Architecture
The system implements a multi-layered security approach:
1. **Encryption Layer**: Secures model data using symmetric encryption
2. **Access Control**: Optional password protection for models
3. **Integrity Verification**: Checksums to detect tampering
4. **Backup Protection**: Automatic backup creation before overwriting
5. **Quantization**: Optional model compression

---

## Classes

### `SecureModelManager`
```python
class SecureModelManager:
```
- **Description**:  
  A secure model manager that provides encryption, access control, and integrity verification for machine learning models.

- **Attributes**:  
  - `DEFAULT_KEY_ITERATIONS (int)`: Default number of iterations for key derivation (200000).
  - `DEFAULT_HASH_ALGORITHM (str)`: Default hash algorithm for passwords ("sha512").
  - `VERSION (str)`: Version string for the manager ("2.0.0").
  - `config (Any)`: Configuration object for the manager.
  - `models (Dict)`: Dictionary of loaded models.
  - `best_model (Dict)`: Reference to the best model based on metrics.
  - `best_score (float)`: Score of the best model.
  - `logger (logging.Logger)`: Logger instance.
  - `quantizer (Any)`: Optional quantizer for model compression.
  - `encryption_enabled (bool)`: Whether encryption is enabled.
  - `key_iterations (int)`: Number of iterations for key derivation.
  - `hash_algorithm (str)`: Algorithm used for hashing.
  - `use_scrypt (bool)`: Whether to use Scrypt instead of PBKDF2.
  - `key (bytes)`: Encryption key.
  - `cipher (Fernet)`: Fernet cipher instance.

- **Constructor**:
  ```python
  def __init__(self, config, logger=None, secret_key=None)
  ```
  - **Parameters**:
    - `config (Any)`: Configuration object containing model parameters and security settings.
    - `logger (logging.Logger, optional)`: Logger instance. If None, creates a new logger.
    - `secret_key (bytes, optional)`: Secret key for encryption. If None, will be generated or loaded.
  - **Raises**:  
    - `Exception`: If encryption initialization fails.

- **Methods**:  

  #### `_initialize_encryption(secret_key=None)`
  ```python
  def _initialize_encryption(self, secret_key=None)
  ```
  - **Description**:  
    Initializes the encryption system with a provided key or generates a new one.
  
  - **Parameters**:  
    - `secret_key (bytes, optional)`: Secret key for encryption. If None, attempts to load from environment or file, or prompts for password.
  
  - **Raises**:  
    - `Exception`: If encryption initialization fails.

  #### `_encrypt_data(data)`
  ```python
  def _encrypt_data(self, data)
  ```
  - **Description**:  
    Encrypts data with integrity verification.
  
  - **Parameters**:  
    - `data (Any)`: Data to be encrypted.
  
  - **Returns**:  
    - `Dict`: Encrypted package with metadata, or original data if encryption fails.
  
  - **Raises**:  
    - `Exception`: If encryption fails.

  #### `_decrypt_data(encrypted_package)`
  ```python
  def _decrypt_data(self, encrypted_package)
  ```
  - **Description**:  
    Decrypts data with integrity verification.
  
  - **Parameters**:  
    - `encrypted_package (Dict)`: Encrypted package to decrypt.
  
  - **Returns**:  
    - `Any`: Decrypted data, or None if decryption fails.
  
  - **Raises**:  
    - `InvalidToken`: If the decryption token is invalid.
    - `Exception`: If decryption fails.

  #### `save_model(model_name=None, filepath=None, access_code=None, compression_level=5)`
  ```python
  def save_model(self, model_name: Optional[str] = None, filepath: Optional[str] = None, 
                 access_code: Optional[str] = None, compression_level: int = 5) -> bool
  ```
  - **Description**:  
    Saves a model to disk with optional encryption, access control, and compression.
  
  - **Parameters**:  
    - `model_name (str, optional)`: Name of the model to save. If None, uses the best model.
    - `filepath (str, optional)`: Path to save the model. If None, uses config.model_path.
    - `access_code (str, optional)`: Password for access control.
    - `compression_level (int)`: Compression level for joblib (0-9). Default is 5.
  
  - **Returns**:  
    - `bool`: True if successful, False otherwise.
  
  - **Raises**:  
    - `Exception`: If saving fails.
  
  - **Example**:
    ```python
    manager.save_model("random_forest", "./models/rf_model.pkl", access_code="secure123")
    ```

  #### `_save_quantized_model(model_data, model_name)`
  ```python
  def _save_quantized_model(self, model_data, model_name)
  ```
  - **Description**:  
    Helper method to save a quantized version of the model for reduced size.
  
  - **Parameters**:  
    - `model_data (Dict)`: Model data to quantize.
    - `model_name (str)`: Name of the model.
  
  - **Raises**:  
    - `Exception`: If quantization fails.

  #### `_safe_config_export()`
  ```python
  def _safe_config_export(self) -> Dict[str, Any]
  ```
  - **Description**:  
    Exports configuration safely, removing sensitive information.
  
  - **Returns**:  
    - `Dict[str, Any]`: Safe configuration dictionary.

  #### `load_model(filepath, access_code=None)`
  ```python
  def load_model(self, filepath: str, access_code: Optional[str] = None) -> Any
  ```
  - **Description**:  
    Loads a model from disk with decryption and access control verification.
  
  - **Parameters**:  
    - `filepath (str)`: Path to the model file.
    - `access_code (str, optional)`: Password for access control.
  
  - **Returns**:  
    - `Any`: Loaded model, or None if loading fails.
  
  - **Raises**:  
    - `Exception`: If loading fails.
  
  - **Example**:
    ```python
    model = manager.load_model("./models/rf_model.pkl", access_code="secure123")
    ```

  #### `_update_best_model(model_name)`
  ```python
  def _update_best_model(self, model_name)
  ```
  - **Description**:  
    Updates the best model tracking after loading a new model.
  
  - **Parameters**:  
    - `model_name (str)`: Name of the model to check.

  #### `_verify_access_control(model_package, access_code)`
  ```python
  def _verify_access_control(self, model_package: Dict[str, Any], access_code: Optional[str]) -> bool
  ```
  - **Description**:  
    Verifies access control for the model.
  
  - **Parameters**:  
    - `model_package (Dict[str, Any])`: Model package containing access control info.
    - `access_code (str, optional)`: Password for access control.
  
  - **Returns**:  
    - `bool`: True if verification succeeds, False otherwise.
  
  - **Raises**:  
    - `Exception`: If verification fails.

  #### `_get_model_score(metrics)`
  ```python
  def _get_model_score(self, metrics: Dict[str, float]) -> float
  ```
  - **Description**:  
    Calculates a score for the model based on metrics.
  
  - **Parameters**:  
    - `metrics (Dict[str, float])`: Dictionary of model metrics.
  
  - **Returns**:  
    - `float`: Score value (lower is better for regression, higher is better for classification).

  #### `rotate_encryption_key(new_password=None)`
  ```python
  def rotate_encryption_key(self, new_password: Optional[str] = None) -> bool
  ```
  - **Description**:  
    Rotates encryption key for added security, re-encrypting all models.
  
  - **Parameters**:  
    - `new_password (str, optional)`: New password for key derivation. If None, generates random key.
  
  - **Returns**:  
    - `bool`: True if successful, False otherwise.
  
  - **Raises**:  
    - `Exception`: If key rotation fails.
  
  - **Example**:
    ```python
    success = manager.rotate_encryption_key("new_secure_password")
    ```

  #### `_find_model_files()`
  ```python
  def _find_model_files(self)
  ```
  - **Description**:  
    Finds all model files in the model directory.
  
  - **Returns**:  
    - `List[str]`: List of file paths.

  #### `verify_model_integrity(filepath)`
  ```python
  def verify_model_integrity(self, filepath: str) -> bool
  ```
  - **Description**:  
    Verifies the integrity of a model file without loading it completely.
  
  - **Parameters**:  
    - `filepath (str)`: Path to the model file.
  
  - **Returns**:  
    - `bool`: True if model has integrity, False otherwise.
  
  - **Raises**:  
    - `Exception`: If verification fails.
  
  - **Example**:
    ```python
    is_valid = manager.verify_model_integrity("./models/rf_model.pkl")
    ```

## Testing
```bash
python -m unittest tests/test_secure_model_manager.py
```

## Versioning and Metadata
> Last Updated: 2025-07-17
> Version: 0.1.4