# SecureModelManager Documentation

## Overview

`SecureModelManager` is a security-focused model management system designed to safely store, encrypt, and manage machine learning models. It provides robust encryption, access control mechanisms, and integrity verification to protect sensitive model data.

## Key Features

- **Strong Encryption**: Uses Fernet symmetric encryption (AES-128-CBC) with key derivation
- **Access Control**: Supports password-protected models with strong key derivation
- **Integrity Verification**: Checksum validation to detect tampering
- **Key Rotation**: Ability to change encryption keys while preserving model access
- **Model Quantization**: Optional model compression support
- **Best Model Tracking**: Automatic tracking of the best performing model

## Installation Requirements

The `SecureModelManager` requires the following dependencies:

```
cryptography
numpy
joblib
```

## Initialization

### Basic Initialization

```python
from modules.configs import TaskType
from secure_model_manager import SecureModelManager

# Create a configuration object
class Config:
    model_path = "./models"
    task_type = TaskType.CLASSIFICATION
    enable_encryption = True
    
config = Config()
manager = SecureModelManager(config)
```

### Advanced Initialization

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create configuration with enhanced security
class AdvancedConfig:
    model_path = "./secure_models"
    task_type = TaskType.REGRESSION
    enable_encryption = True
    key_iterations = 300000  # Higher than default for stronger security
    use_scrypt = True  # Use Scrypt instead of PBKDF2
    primary_metric = "mae"  # Specify which metric to use for model comparison
    enable_quantization = True  # Enable model compression
    
config = AdvancedConfig()
manager = SecureModelManager(config, logger=logger)
```

## Saving Models

### Basic Model Saving

```python
# Assuming 'model' is your trained model
model_data = {
    "name": "my_model",
    "model": model,
    "params": model.get_params(),
    "metrics": {"accuracy": 0.95, "f1": 0.94}
}

manager.models["my_model"] = model_data
manager.save_model("my_model")
```

### Saving with Access Controls

```python
# Save model with password protection
access_code = "secure_password_123"
manager.save_model("my_model", access_code=access_code)
```

### Saving with Custom Path

```python
# Save to a specific location
custom_path = "/secure/location/my_special_model.pkl"
manager.save_model("my_model", filepath=custom_path)
```

## Loading Models

### Basic Model Loading

```python
# Load a model
model = manager.load_model("./models/my_model.pkl")

# Use the loaded model
if model is not None:
    predictions = model.predict(X_test)
```

### Loading Password-Protected Models

```python
# Load a password-protected model
access_code = "secure_password_123"
model = manager.load_model("./models/protected_model.pkl", access_code=access_code)
```

## Security Management

### Key Rotation

```python
# Rotate encryption keys (reencrypts all models)
manager.rotate_encryption_key()

# Rotate keys with a new password
manager.rotate_encryption_key(new_password="new_secure_password")
```

### Integrity Verification

```python
# Verify model hasn't been tampered with
is_valid = manager.verify_model_integrity("./models/my_model.pkl")
if not is_valid:
    print("Warning: Model may have been compromised!")
```

## Best Model Management

The manager automatically tracks the best model based on metrics:

```python
# Get the current best model
best_model = manager.best_model["model"]

# Check the best score
print(f"Best model score: {manager.best_score}")

# Save the best model
manager.save_model()  # No model_name needed when saving the best model
```

## Encryption Details

The `SecureModelManager` supports two key derivation methods:

1. **Scrypt** (default, more resistant to hardware acceleration attacks)
2. **PBKDF2** with SHA-512 (still strong with sufficient iterations)

When encryption is enabled:

- A secret key is either provided, loaded from environment variables, derived from a password, or generated randomly
- All model data is encrypted before being saved to disk
- File permissions are set to restrict access (0o600 - owner only)
- Integrity verification is added via SHA-512 checksums

## Advanced Usage Patterns

### Working with Quantized Models

If quantization is enabled and a quantizer is provided:

```python
# Configure a quantizer
class SimpleQuantizer:
    def quantize(self, data):
        # Implement quantization logic
        return quantized_data
        
    def get_config(self):
        return {"quantization_method": "simple"}

config.quantizer = SimpleQuantizer()
manager = SecureModelManager(config)

# Save a model (will also save quantized version)
manager.save_model("my_model")
```

### Environment Variable Configuration

You can set encryption keys via environment variables:

```bash
# Set in environment
export MODEL_ENCRYPTION_KEY="your-base64-key"
```

```python
# The manager will use the environment variable key
manager = SecureModelManager(config)
```

## Error Handling

The manager includes extensive error handling and fallback mechanisms:

- Creates backups before overwriting files
- Falls back to unencrypted mode if encryption fails
- Logs detailed error information
- Returns meaningful boolean results from operations

## Security Best Practices

1. Store models in a directory with restricted permissions
2. Use strong, unique passwords for access codes
3. Rotate encryption keys periodically
4. Verify model integrity before using models from untrusted sources
5. Consider using Scrypt key derivation for stronger security
6. Do not share encryption keys or access codes in code repositories
7. Set up proper backup procedures for encrypted models

## Performance Considerations

- Encryption and decryption add overhead to save/load operations
- Higher key iteration counts increase security but slow down operations
- Quantization can significantly reduce model size at some cost to accuracy

## Limitations

- Encryption is only as strong as the key management
- Requires maintaining access to encryption keys to use models
- Not designed for distributed or multi-user environments without additional access controls