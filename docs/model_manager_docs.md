# Module: `secure_model_manager`

## Overview
The `SecureModelManager` class manages trained machine learning models with 
robust support for secure encryption, integrity verification, quantization, access 
control, and versioning. It offers enhanced capabilities for model serialization, 
decryption, validation, and rotation of encryption keys.

---

## Prerequisites
- Python >= 3.8
- Dependencies:
  ```bash
  pip install cryptography joblib numpy
  ```
- Custom module dependency: `modules.configs` with at least the `TaskType` definition

---

## Configuration
| Config Option        | Type     | Default       | Description |
|----------------------|----------|---------------|-------------|
| `enable_encryption`  | `bool`   | `True`        | Enables encryption of model files |
| `model_path`         | `str`    | required      | Directory where models are saved |
| `task_type`          | `Enum`   | Regression or Classification | Defines task type for scoring |
| `primary_metric`     | `str`    | Optional      | Primary metric used to rank models |
| `use_scrypt`         | `bool`   | `True`        | Whether to use `Scrypt` over `PBKDF2` |
| `key_iterations`     | `int`    | `200000`      | Number of iterations for PBKDF2 |
| `hash_algorithm`     | `str`    | `sha512`      | Hash algorithm for PBKDF2 |
| `enable_quantization`| `bool`   | Optional      | If True, enables saving quantized models |
| `quantizer`          | `object` | Optional      | A quantizer instance with `quantize()` |

---

## Usage
```python
from secure_model_manager import SecureModelManager
from modules.configs import MyModelConfig

config = MyModelConfig(model_path='models/')
manager = SecureModelManager(config)
model = SomeTrainedModel()

# Save model securely
manager.models['my_model'] = {"model": model, "metrics": {"accuracy": 0.91}}
manager.save_model('my_model')

# Load model
loaded_model = manager.load_model('models/my_model.pkl')
```

---

## Classes

### `SecureModelManager`
```python
class SecureModelManager:
```
- **Description**: 
  Manages secure saving, loading, quantizing, and validating machine learning models. 
  Supports encryption using Fernet, access control with password hashing (PBKDF2/Scrypt),
  and configuration-aware score tracking.

#### Constructor
```python
def __init__(self, config, logger=None, secret_key=None):
```
- **Parameters**:
  - `config`: Object with attributes like `model_path`, `task_type`, `enable_encryption`
  - `logger`: Optional logger instance
  - `secret_key`: Optional base64 key for Fernet encryption

---

## Methods

### `_initialize_encryption`
```python
def _initialize_encryption(self, secret_key=None):
```
- Initializes Fernet encryption either via:
  - Secret key from environment or argument
  - Password-derived key using Scrypt or PBKDF2
  - Auto-generated secure random key if no password provided

---

### `_encrypt_data`
```python
def _encrypt_data(self, data):
```
- **Encrypts and serializes model data with integrity hash and metadata.**
- **Returns**: dict with `encrypted_data`, `checksum`, and metadata.

---

### `_decrypt_data`
```python
def _decrypt_data(self, encrypted_package):
```
- **Decrypts model content** and verifies its checksum using SHA-512.
- Returns `pickle`-loaded model object on success or `None` on failure.

---

### `save_model`
```python
def save_model(self, model_name=None, filepath=None, access_code=None, compression_level=5):
```
- **Saves model securely** to disk, with:
  - Encrypted and versioned format
  - Optional password protection (Scrypt/PBKDF2)
  - Compression via joblib (if encryption is off)
- Creates `.bak` backup before overwriting

---

### `load_model`
```python
def load_model(self, filepath: str, access_code: Optional[str] = None):
```
- **Loads model from disk**, decrypts and validates access if necessary.
- **Returns**: Model object or `None`

---

### `_save_quantized_model`
```python
def _save_quantized_model(self, model_data, model_name):
```
- Converts model bytes to quantized form using a provided quantizer
- Saves it encrypted (if enabled) with metadata

---

### `_safe_config_export`
```python
def _safe_config_export(self):
```
- Exports the config dictionary omitting sensitive fields like keys and tokens

---

### `_verify_access_control`
```python
def _verify_access_control(self, model_package: Dict[str, Any], access_code: Optional[str]) -> bool:
```
- Checks the correctness of access password using hash verification (PBKDF2/Scrypt)

---

### `rotate_encryption_key`
```python
def rotate_encryption_key(self, new_password: Optional[str] = None) -> bool:
```
- Rotates the current encryption key
- Re-encrypts all `.pkl` model files in `model_path`

---

### `verify_model_integrity`
```python
def verify_model_integrity(self, filepath: str) -> bool:
```
- Verifies the SHA-512 checksum of an encrypted model file
- Returns `True` if verified, else `False`

---

### `_get_model_score`
```python
def _get_model_score(self, metrics: Dict[str, float]) -> float:
```
- Returns a numeric score depending on task type:
  - Regression: lower is better (MAE, MSE, etc.)
  - Classification: higher is better (Accuracy, F1, etc.)

---

### `_update_best_model`
```python
def _update_best_model(self, model_name):
```
- Updates the internal best model tracker based on performance score

---

### `_find_model_files`
```python
def _find_model_files(self):
```
- Finds `.pkl` model files in `model_path`

---

## Security & Compliance
- AES-128 (via Fernet) encryption
- SHA-512 checksum verification
- Secure password hashing with `Scrypt` or `PBKDF2`
- Password-protected access control
- File system permissions set to owner-only (`chmod 600`)

---

## Versioning & Metadata
> Last Updated: 2025-04-28  
> Version: 0.1.0  
> Encryption format includes timestamp and algorithm metadata for compatibility

---
