import os
import time
import joblib
import pickle
import numpy as np
import hashlib
import base64
import logging
import enum
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from getpass import getpass

from modules.configs import TaskType

class SecureModelManager:
    """
    Enhanced secure model manager with advanced encryption capabilities
    and improved model handling.
    """
    
    DEFAULT_KEY_ITERATIONS = 200000
    DEFAULT_HASH_ALGORITHM = "sha512"
    VERSION = "2.0.0"
    
    def __init__(self, config, logger=None, secret_key=None):
        """Initialize the secure model manager with encryption capabilities"""
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = float('inf') if config.task_type == TaskType.REGRESSION else float('-inf')
        self.logger = logger or logging.getLogger(__name__)
        self.quantizer = getattr(self.config, 'quantizer', None)
        
        # Security configurations
        self.encryption_enabled = getattr(self.config, 'enable_encryption', True)
        self.key_iterations = getattr(self.config, 'key_iterations', self.DEFAULT_KEY_ITERATIONS)
        self.hash_algorithm = getattr(self.config, 'hash_algorithm', self.DEFAULT_HASH_ALGORITHM)
        self.use_scrypt = getattr(self.config, 'use_scrypt', True)
        
        # Create model directory if it doesn't exist
        Path(self.config.model_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        self.key = None
        self.cipher = None
        if self.encryption_enabled:
            self._initialize_encryption(secret_key)
            
    def _initialize_encryption(self, secret_key=None):
        """Initialize encryption with a secret key or generate one"""
        try:
            # Get or create a secret key
            if secret_key is None:
                # Check if key exists in environment variable
                env_key = os.environ.get('MODEL_ENCRYPTION_KEY')
                if env_key:
                    secret_key = env_key
                else:
                    # Check if key file exists
                    key_path = os.path.join(self.config.model_path, '.enc_key')
                    if os.path.exists(key_path):
                        with open(key_path, 'rb') as key_file:
                            self.key = key_file.read()
                        self.logger.info(f"Loaded encryption key from {key_path}")
                    else:
                        # Ask for a password to derive the key from
                        password = getpass("Enter encryption password (or leave empty to generate random key): ")
                        if not password:
                            # Generate a random key if no password provided
                            self.key = Fernet.generate_key()
                            # Save the key to a secure file with restricted permissions
                            with open(key_path, 'wb') as key_file:
                                key_file.write(self.key)
                            os.chmod(key_path, 0o600)  # Only owner can read/write
                            self.logger.info(f"Generated encryption key stored at {key_path}")
                        else:
                            # Derive key from password using either Scrypt (preferred) or PBKDF2
                            salt = os.urandom(32)
                            salt_path = os.path.join(self.config.model_path, '.salt')
                            with open(salt_path, 'wb') as salt_file:
                                salt_file.write(salt)
                            os.chmod(salt_path, 0o600)  # Only owner can read/write
                            
                            if self.use_scrypt:
                                # Scrypt is more resistant to hardware acceleration attacks
                                kdf = Scrypt(
                                    salt=salt,
                                    length=32,
                                    n=2**14,  # Reduced from 2**20 for practical performance
                                    r=8,
                                    p=1,
                                    backend=default_backend()
                                )
                                key_bytes = kdf.derive(password.encode())
                            else:
                                # PBKDF2 with SHA-512 is still strong with enough iterations
                                kdf = PBKDF2HMAC(
                                    algorithm=getattr(hashes, self.hash_algorithm.upper())(),
                                    length=32,
                                    salt=salt,
                                    iterations=self.key_iterations,
                                    backend=default_backend()
                                )
                                key_bytes = kdf.derive(password.encode())
                                
                            self.key = base64.urlsafe_b64encode(key_bytes)
            else:
                self.key = secret_key
                
            # Initialize the Fernet cipher for symmetric encryption
            self.cipher = Fernet(self.key)
            self.logger.info("Encryption initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {str(e)}")
            self.encryption_enabled = False
            self.logger.warning("Encryption has been disabled due to initialization failure")
            
    def _encrypt_data(self, data):
        """Encrypt the given data with improved integrity verification"""
        if not self.encryption_enabled or self.cipher is None:
            return data
            
        try:
            # Serialize the data
            serialized_data = pickle.dumps(data)
            
            # Encrypt the serialized data
            encrypted_data = self.cipher.encrypt(serialized_data)
            
            # Calculate checksum for integrity verification using SHA-512
            checksum = hashlib.sha512(serialized_data).digest()
            
            # Add time-based versioning to detect replays
            timestamp = int(time.time())
            
            return {
                "encrypted_data": encrypted_data,
                "checksum": checksum,
                "encryption_metadata": {
                    "algorithm": "Fernet (AES-128-CBC)",
                    "checksum_algorithm": "SHA-512",
                    "timestamp": timestamp,
                    "is_encrypted": True,
                    "version": self.VERSION
                }
            }
        except Exception as e:
            self.logger.error(f"Encryption failed: {str(e)}")
            return data  # Return original data on failure instead of None
            
    def _decrypt_data(self, encrypted_package):
        """Decrypt the given encrypted data with enhanced verification"""
        if not self.encryption_enabled or self.cipher is None or not isinstance(encrypted_package, dict) or "encrypted_data" not in encrypted_package:
            return encrypted_package
            
        try:
            # Check version compatibility
            metadata = encrypted_package.get("encryption_metadata", {})
            package_version = metadata.get("version", "1.0.0")
            
            if package_version > self.VERSION:
                self.logger.warning(f"Model was encrypted with a newer version ({package_version}) than current ({self.VERSION})")
            
            # Decrypt the data
            decrypted_data = self.cipher.decrypt(encrypted_package["encrypted_data"])
            
            # Verify integrity with checksum
            if "checksum" in encrypted_package:
                calculated_checksum = None
                if metadata.get("checksum_algorithm") == "SHA-512":
                    calculated_checksum = hashlib.sha512(decrypted_data).digest()
                else:
                    # Fallback for old format using SHA-256
                    calculated_checksum = hashlib.sha256(decrypted_data).digest()
                    
                if calculated_checksum != encrypted_package["checksum"]:
                    self.logger.error("Data integrity verification failed. The model may have been tampered with.")
                    return None
                    
            # Deserialize the data
            deserialized_data = pickle.loads(decrypted_data)
            
            return deserialized_data
        except InvalidToken:
            self.logger.error("Invalid encryption token. Key may be incorrect or data corrupted.")
            return None
        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            return None
    
    def save_model(self, model_name: Optional[str] = None, filepath: Optional[str] = None, 
                   access_code: Optional[str] = None, compression_level: int = 5) -> bool:
        """Save the model to disk with encryption and optional compression"""
        if model_name is None and self.best_model is not None:
            model_name = self.best_model["name"]
            model_data = self.best_model
        elif model_name in self.models:
            model_data = self.models[model_name]
        else:
            self.logger.error(f"Model {model_name} not found")
            return False
            
        if filepath is None:
            filepath = os.path.join(self.config.model_path, f"{model_name}.pkl")
            
        try:
            # Prepare the model package with metadata
            model_package = {
                "model": model_data["model"],
                "params": model_data.get("params", {}),
                "metrics": model_data.get("metrics", {}),
                "config": self._safe_config_export(),
                "timestamp": int(time.time()),
                "version": self.VERSION,
                "model_name": model_name
            }
            
            # Apply access control if provided
            if access_code:
                # Use Scrypt for password hashing
                salt = os.urandom(32)
                if self.use_scrypt:
                    kdf = Scrypt(
                        salt=salt,
                        length=64,
                        n=2**14,  # Reduced from 2**20 for practical performance
                        r=8,
                        p=1,
                        backend=default_backend()
                    )
                    password_hash = kdf.derive(access_code.encode())
                else:
                    password_hash = hashlib.pbkdf2_hmac(
                        self.hash_algorithm, 
                        access_code.encode(), 
                        salt, 
                        self.key_iterations
                    )
                
                model_package["access_control"] = {
                    "salt": salt,
                    "password_hash": password_hash,
                    "method": "scrypt" if self.use_scrypt else "pbkdf2",
                    "iterations": self.key_iterations if not self.use_scrypt else None,
                    "algorithm": self.hash_algorithm
                }
            
            # Create backup of existing file before overwriting
            if os.path.exists(filepath):
                backup_path = f"{filepath}.bak"
                try:
                    os.replace(filepath, backup_path)
                    self.logger.info(f"Created backup at {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to create backup: {str(e)}")
            
            if self.encryption_enabled and self.cipher is not None:
                # Encrypt the entire model package
                encrypted_package = self._encrypt_data(model_package)
                if encrypted_package is None:
                    self.logger.error("Failed to encrypt model data")
                    return False
                    
                # Save the encrypted package
                with open(filepath, 'wb') as f:
                    pickle.dump(encrypted_package, f)
            else:
                # Use joblib for efficient serialization without encryption
                joblib.dump(model_package, filepath, compress=compression_level)
                
            # Set restrictive file permissions
            os.chmod(filepath, 0o600)  # Only owner can read/write
            
            self.logger.info(f"Model saved to {filepath}")
            
            # Optionally quantize and save a quantized version
            if hasattr(self.config, 'enable_quantization') and self.config.enable_quantization and self.quantizer:
                try:
                    self._save_quantized_model(model_data, model_name)
                except Exception as e:
                    self.logger.warning(f"Failed to save quantized model: {str(e)}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            # Store backup in case of failure
            backup_path = f"{filepath}.bak"
            self.logger.info(f"Attempting to save backup to {backup_path}")
            try:
                joblib.dump(model_data, backup_path)
                self.logger.info(f"Backup saved to {backup_path}")
            except Exception as backup_error:
                self.logger.error(f"Failed to save backup: {str(backup_error)}")
            return False
    
    def _save_quantized_model(self, model_data, model_name):
        """Helper method to save quantized version of the model"""
        if self.quantizer is None:
            self.logger.error("Quantizer not initialized but quantization was requested")
            return
            
        # Prepare model for quantization - exclude non-numeric parts
        model_bytes = pickle.dumps(model_data["model"])
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        
        # Quantize the model bytes
        quantized_data = self.quantizer.quantize(model_array)
        
        # Prepare quantized package
        quantized_package = {
            "quantized_data": quantized_data,
            "metadata": {
                "original_size": len(model_bytes),
                "quantized_size": len(quantized_data),
                "config": self.quantizer.get_config(),
                "model_name": model_name,
                "timestamp": int(time.time())
            }
        }
        
        # Create quantized filepath
        quantized_filepath = os.path.join(self.config.model_path, f"{model_name}_quantized.pkl")
        
        # Create backup of existing quantized file if it exists
        if os.path.exists(quantized_filepath):
            backup_path = f"{quantized_filepath}.bak"
            try:
                os.replace(quantized_filepath, backup_path)
                self.logger.info(f"Created backup of quantized model at {backup_path}")
            except Exception as e:
                self.logger.warning(f"Failed to create backup of quantized model: {str(e)}")
        
        # Encrypt quantized package if encryption is enabled
        if self.encryption_enabled and self.cipher is not None:
            encrypted_quantized = self._encrypt_data(quantized_package)
            with open(quantized_filepath, 'wb') as f:
                pickle.dump(encrypted_quantized, f)
        else:
            with open(quantized_filepath, 'wb') as f:
                pickle.dump(quantized_package, f)
                
        # Set restrictive file permissions
        os.chmod(quantized_filepath, 0o600)
        
        self.logger.info(f"Quantized model saved to {quantized_filepath}")
        
    def _safe_config_export(self) -> Dict[str, Any]:
        """Export config safely, removing sensitive information"""
        if hasattr(self.config, 'to_dict'):
            config_dict = self.config.to_dict()
        else:
            # Try to convert to dictionary if possible
            config_dict = {}
            for key, value in vars(self.config).items():
                if not key.startswith('_'):
                    try:
                        # Try JSON serialization to ensure exportability
                        json.dumps({key: value})
                        config_dict[key] = value
                    except (TypeError, OverflowError, ValueError):
                        config_dict[key] = str(value)
        
        # Remove any potentially sensitive information
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
        for key in list(config_dict.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                del config_dict[key]
                
        return config_dict
            
    def load_model(self, filepath: str, access_code: Optional[str] = None) -> Any:
        """Load a model from disk with decryption and access control"""
        try:
            # First, check if the file exists and can be accessed
            if not os.path.exists(filepath):
                self.logger.error(f"Model file {filepath} does not exist")
                return None
                
            # Load the raw file content
            try:
                with open(filepath, 'rb') as f:
                    file_content = pickle.load(f)
            except Exception as e:
                self.logger.error(f"Failed to read model file: {str(e)}")
                return None
                
            # Check if the model is encrypted
            is_encrypted = isinstance(file_content, dict) and "encrypted_data" in file_content
            
            if is_encrypted and self.encryption_enabled and self.cipher is not None:
                # Decrypt the model package
                model_package = self._decrypt_data(file_content)
                if model_package is None:
                    self.logger.error("Failed to decrypt model data")
                    return None
            elif is_encrypted and (not self.encryption_enabled or self.cipher is None):
                self.logger.error("Encrypted model detected but encryption is not enabled or initialized")
                return None
            else:
                # Model is not encrypted
                try:
                    # First try to read it as our standard model package format
                    if isinstance(file_content, dict) and "model" in file_content:
                        model_package = file_content
                    else:
                        # Try loading it using joblib instead
                        try:
                            model_package = joblib.load(filepath)
                        except Exception as joblib_err:
                            self.logger.error(f"Failed to load with joblib: {str(joblib_err)}")
                            return None
                except Exception as e:
                    self.logger.error(f"Failed to load unencrypted model: {str(e)}")
                    return None
                
            # Verify access control if present in model
            if "access_control" in model_package and not self._verify_access_control(model_package, access_code):
                return None
            
            # Verify model package has required fields
            if "model" not in model_package:
                self.logger.error("Invalid model package - missing 'model' field")
                return None
                    
            # Extract model data
            model_name = model_package.get("model_name", os.path.basename(filepath).split('.')[0])
            self.models[model_name] = {
                "name": model_name,
                "model": model_package["model"],
                "params": model_package.get("params", {}),
                "metrics": model_package.get("metrics", {})
            }
            
            # Update best model tracking
            self._update_best_model(model_name)
                
            self.logger.info(f"Model loaded from {filepath}")
            return model_package["model"]
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return None
    
    def _update_best_model(self, model_name):
        """Update best model tracking after loading a new model"""
        if model_name not in self.models:
            return
            
        metrics = self.models[model_name].get("metrics", {})
        model_score = self._get_model_score(metrics)
        
        is_better = False
        if self.config.task_type == TaskType.REGRESSION:
            # For regression, lower score is better
            if model_score < self.best_score:
                is_better = True
        else:
            # For classification, higher score is better
            if model_score > self.best_score:
                is_better = True
                
        if is_better or self.best_model is None:
            self.best_score = model_score
            self.best_model = self.models[model_name].copy()
            
    def _verify_access_control(self, model_package: Dict[str, Any], access_code: Optional[str]) -> bool:
        """Verify access control for the model"""
        if "access_control" not in model_package:
            return True
            
        if access_code is None:
            self.logger.error("This model requires an access code to load")
            return False
            
        ac = model_package["access_control"]
        method = ac.get("method", "pbkdf2")
        
        try:
            if method == "scrypt":
                # Verify with Scrypt
                kdf = Scrypt(
                    salt=ac["salt"],
                    length=len(ac["password_hash"]),
                    n=2**14,  # Reduced from 2**20 for practical performance
                    r=8,
                    p=1,
                    backend=default_backend()
                )
                # This will raise an exception if verification fails
                kdf.verify(access_code.encode(), ac["password_hash"])
            else:
                # Verify with PBKDF2
                provided_hash = hashlib.pbkdf2_hmac(
                    ac.get("algorithm", self.hash_algorithm),
                    access_code.encode(),
                    ac["salt"],
                    ac.get("iterations", self.key_iterations)
                )
                if not isinstance(provided_hash, bytes) or not isinstance(ac["password_hash"], bytes):
                    raise ValueError("Hash type mismatch")
                    
                if provided_hash != ac["password_hash"]:
                    raise ValueError("Invalid access code")
                    
            return True
        except Exception as e:
            self.logger.error(f"Access verification failed: {str(e)}")
            return False

    def _get_model_score(self, metrics: Dict[str, float]) -> float:
        """Calculate a score for the model based on metrics"""
        if not metrics:
            # Default scores when no metrics are available
            return float('inf') if self.config.task_type == TaskType.REGRESSION else float('-inf')
        
        # If we have a primary metric configured, use that
        primary_metric = getattr(self.config, 'primary_metric', None)
        if primary_metric and primary_metric in metrics:
            raw_value = metrics[primary_metric]
            
            # Handle the primary metric based on task type
            if self.config.task_type == TaskType.REGRESSION:
                # For regression, we want to MINIMIZE metrics like MSE, RMSE, MAE (lower is better)
                # But MAXIMIZE metrics like R2 (higher is better)
                if primary_metric.lower() in ['r2', 'r2_score', 'variance_explained']:
                    # For R2, higher is better, so we negate to convert to a minimization problem
                    return -raw_value
                else:
                    # For error metrics, lower is better
                    return raw_value
            else:
                # For classification, we want to MAXIMIZE metrics like accuracy, f1, auc (higher is better)
                return raw_value
        
        # If no primary metric is defined or it's not in the metrics, use heuristics
        if self.config.task_type == TaskType.REGRESSION:
            # For regression tasks, prioritize certain metrics
            # First, check for R2 (where higher is better)
            for r2_metric in ['r2', 'r2_score', 'variance_explained']:
                if r2_metric in metrics:
                    # Negate R2 so that higher R2 becomes a lower score (better)
                    return -metrics[r2_metric]
                    
            # Then check for error metrics (where lower is better)
            for err_metric in ['mae', 'mean_absolute_error', 'rmse', 'root_mean_squared_error', 'mse', 'mean_squared_error']:
                if err_metric in metrics:
                    return metrics[err_metric]
            
            # If no recognized metric is found, use the first one
            # Assume it's an error metric where lower is better
            if metrics:
                return next(iter(metrics.values()))
            return float('inf')  # Default if metrics dict is empty
        else:
            # For classification, prioritize accuracy metrics (higher is better)
            for cls_metric in ['accuracy', 'balanced_accuracy', 'f1', 'f1_score', 'auc', 'roc_auc', 'precision', 'recall']:
                if cls_metric in metrics:
                    return metrics[cls_metric]
                
            # If no recognized metric is found, use the first one
            # Assume higher is better for classification
            if metrics:
                return next(iter(metrics.values()))
            return float('-inf')  # Default if metrics dict is empty
            
    def rotate_encryption_key(self, new_password: Optional[str] = None) -> bool:
        """Rotate encryption key for added security"""
        if not self.encryption_enabled or self.cipher is None:
            self.logger.warning("Encryption is not enabled or initialized, cannot rotate key")
            return False
            
        try:
            # Remember old key
            old_key = self.key
            old_cipher = self.cipher
            
            # Generate or derive new key
            if new_password:
                # Use Scrypt for better security (if enabled)
                salt = os.urandom(32)
                salt_path = os.path.join(self.config.model_path, '.salt')
                with open(salt_path, 'wb') as salt_file:
                    salt_file.write(salt)
                os.chmod(salt_path, 0o600)
                
                if self.use_scrypt:
                    kdf = Scrypt(
                        salt=salt,
                        length=32,
                        n=2**14,  # Reduced from 2**20 for practical performance
                        r=8,
                        p=1,
                        backend=default_backend()
                    )
                    key_bytes = kdf.derive(new_password.encode())
                else:
                    kdf = PBKDF2HMAC(
                        algorithm=getattr(hashes, self.hash_algorithm.upper())(),
                        length=32,
                        salt=salt,
                        iterations=self.key_iterations,
                        backend=default_backend()
                    )
                    key_bytes = kdf.derive(new_password.encode())
                
                self.key = base64.urlsafe_b64encode(key_bytes)
            else:
                self.key = Fernet.generate_key()
                key_path = os.path.join(self.config.model_path, '.enc_key')
                with open(key_path, 'wb') as key_file:
                    key_file.write(self.key)
                os.chmod(key_path, 0o600)
                
            # Initialize new cipher
            self.cipher = Fernet(self.key)
            
            # Reencrypt all models with new key
            model_files = self._find_model_files()
            for filepath in model_files:
                try:
                    # Load with old key
                    with open(filepath, 'rb') as f:
                        file_content = pickle.load(f)
                    
                    if isinstance(file_content, dict) and "encrypted_data" in file_content:
                        # Set up temporary old cipher
                        temp_cipher = old_cipher
                        
                        # Decrypt with old cipher
                        decrypted_data = temp_cipher.decrypt(file_content["encrypted_data"])
                        model_package = pickle.loads(decrypted_data)
                        
                        # Re-encrypt with new cipher
                        encrypted_package = self._encrypt_data(model_package)
                        
                        # Save back with backup
                        backup_path = f"{filepath}.bak"
                        os.rename(filepath, backup_path)
                        
                        with open(filepath, 'wb') as f:
                            pickle.dump(encrypted_package, f)
                            
                        # Remove backup after successful save
                        os.remove(backup_path)
                        
                        model_name = os.path.basename(filepath).split('.')[0]
                        self.logger.info(f"Re-encrypted model {model_name} with new key")
                except Exception as e:
                    self.logger.error(f"Failed to re-encrypt {filepath}: {str(e)}")
            
            self.logger.info("Encryption key rotation completed successfully")
            return True
        except Exception as e:
            # Restore old key on failure
            self.key = old_key
            self.cipher = old_cipher
            self.logger.error(f"Key rotation failed: {str(e)}")
            return False
            
    def _find_model_files(self):
        """Find all model files in the model directory"""
        model_path = Path(self.config.model_path)
        return [str(f) for f in model_path.glob("*.pkl") if not f.name.startswith('.')]
        
    def verify_model_integrity(self, filepath: str) -> bool:
        """Verify the integrity of a model file without loading it completely"""
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"Model file {filepath} does not exist")
                return False
                
            # Load just the encrypted package
            with open(filepath, 'rb') as f:
                file_content = pickle.load(f)
                
            # Check if the model is encrypted
            if not isinstance(file_content, dict) or "encrypted_data" not in file_content:
                self.logger.warning(f"Model {filepath} is not encrypted, cannot verify integrity")
                return True  # Can't verify, but not necessarily invalid
                
            # Verify checksum without full decryption
            if "checksum" not in file_content:
                self.logger.warning(f"Model {filepath} does not contain integrity verification information")
                return True
                
            # Check if encryption is enabled and cipher is initialized
            if not self.encryption_enabled or self.cipher is None:
                self.logger.error("Cannot verify integrity: encryption not enabled or cipher not initialized")
                return False
                
            # Attempt partial decryption to verify
            try:
                decrypted_data = self.cipher.decrypt(file_content["encrypted_data"])
                
                # Calculate checksum
                metadata = file_content.get("encryption_metadata", {})
                calculated_checksum = None
                if metadata.get("checksum_algorithm") == "SHA-512":
                    calculated_checksum = hashlib.sha512(decrypted_data).digest()
                else:
                    calculated_checksum = hashlib.sha256(decrypted_data).digest()
                    
                if calculated_checksum != file_content["checksum"]:
                    self.logger.error(f"Integrity check failed for {filepath}. The file may have been tampered with.")
                    return False
                    
                self.logger.info(f"Model {filepath} integrity verified successfully")
                return True
            except InvalidToken:
                self.logger.error(f"Cannot decrypt {filepath}. Invalid token or wrong encryption key.")
                return False
            except Exception as e:
                self.logger.error(f"Integrity check failed with error: {str(e)}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to verify model integrity: {str(e)}")
            return False