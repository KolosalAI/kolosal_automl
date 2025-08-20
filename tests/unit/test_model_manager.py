import unittest
import os
import tempfile
import shutil
import pickle
import numpy as np
from pathlib import Path
from cryptography.fernet import Fernet
import hashlib
import base64
import logging
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
# Add missing imports for the key rotation test
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from modules.model_manager import SecureModelManager
from modules.configs import TaskType

class Config:
    """Configuration class for testing"""
    def __init__(self, task_type=TaskType.CLASSIFICATION, enable_encryption=True, 
                 hash_algorithm="sha512", key_iterations=200000, use_scrypt=True):
        self.task_type = task_type
        self.enable_encryption = enable_encryption
        self.hash_algorithm = hash_algorithm
        self.key_iterations = key_iterations
        self.use_scrypt = use_scrypt
        self.model_path = None  # Set in setUp
        self.enable_quantization = False
        self.primary_metric = "accuracy" if task_type == TaskType.CLASSIFICATION else "mse"
        self.quantizer = None
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "task_type": self.task_type,
            "enable_encryption": self.enable_encryption,
            "hash_algorithm": self.hash_algorithm,
            "key_iterations": self.key_iterations,
            "use_scrypt": self.use_scrypt,
            "model_path": self.model_path,
            "enable_quantization": self.enable_quantization,
            "primary_metric": self.primary_metric
        }


class TestSecureModelManagerWithRealModels(unittest.TestCase):
    """Unit tests for the SecureModelManager class using real sklearn models"""
    
    def setUp(self):
        """Set up test environment with real data and models"""
        # Create temporary directory for model storage
        self.test_dir = tempfile.mkdtemp()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create classification dataset
        X_cls, y_cls = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_informative=10, 
            n_classes=2, 
            random_state=42
        )
        self.X_cls_train, self.X_cls_test, self.y_cls_train, self.y_cls_test = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42
        )
        
        # Create regression dataset
        X_reg, y_reg = make_regression(
            n_samples=1000, 
            n_features=20, 
            n_informative=10, 
            noise=0.1, 
            random_state=42
        )
        self.X_reg_train, self.X_reg_test, self.y_reg_train, self.y_reg_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Train classification models
        self.log_reg = LogisticRegression(max_iter=1000, random_state=42)
        self.log_reg.fit(self.X_cls_train, self.y_cls_train)
        log_reg_preds = self.log_reg.predict(self.X_cls_test)
        log_reg_acc = accuracy_score(self.y_cls_test, log_reg_preds)
        log_reg_f1 = f1_score(self.y_cls_test, log_reg_preds)
        
        self.rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_cls.fit(self.X_cls_train, self.y_cls_train)
        rf_cls_preds = self.rf_cls.predict(self.X_cls_test)
        rf_cls_acc = accuracy_score(self.y_cls_test, rf_cls_preds)
        rf_cls_f1 = f1_score(self.y_cls_test, rf_cls_preds)
        
        # Train regression models
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.X_reg_train, self.y_reg_train)
        lin_reg_preds = self.lin_reg.predict(self.X_reg_test)
        lin_reg_mse = mean_squared_error(self.y_reg_test, lin_reg_preds)
        lin_reg_mae = mean_absolute_error(self.y_reg_test, lin_reg_preds)
        
        self.rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_reg.fit(self.X_reg_train, self.y_reg_train)
        rf_reg_preds = self.rf_reg.predict(self.X_reg_test)
        rf_reg_mse = mean_squared_error(self.y_reg_test, rf_reg_preds)
        rf_reg_mae = mean_absolute_error(self.y_reg_test, rf_reg_preds)
        
        # Record metrics
        self.cls_metrics = {
            "logistic_regression": {"accuracy": log_reg_acc, "f1": log_reg_f1},
            "random_forest": {"accuracy": rf_cls_acc, "f1": rf_cls_f1}
        }
        
        self.reg_metrics = {
            "linear_regression": {"mse": lin_reg_mse, "mae": lin_reg_mae},
            "random_forest": {"mse": rf_reg_mse, "mae": rf_reg_mae}
        }
        
        # Create configurations
        self.cls_config = Config(task_type=TaskType.CLASSIFICATION)
        self.cls_config.model_path = os.path.join(self.test_dir, "classification")
        os.makedirs(self.cls_config.model_path, exist_ok=True)
        
        self.reg_config = Config(task_type=TaskType.REGRESSION)
        self.reg_config.model_path = os.path.join(self.test_dir, "regression")
        os.makedirs(self.reg_config.model_path, exist_ok=True)
        
        # Use deterministic encryption key for testing
        self.test_key = Fernet.generate_key()
        
        # Initialize manager instances
        self.cls_manager = SecureModelManager(
            self.cls_config, 
            logger=self.logger,
            secret_key=self.test_key
        )
        
        self.reg_manager = SecureModelManager(
            self.reg_config, 
            logger=self.logger,
            secret_key=self.test_key
        )
        
        # Add models to respective managers
        self.cls_manager.models["logistic_regression"] = {
            "name": "logistic_regression",
            "model": self.log_reg,
            "params": {"max_iter": 1000, "random_state": 42},
            "metrics": self.cls_metrics["logistic_regression"]
        }
        
        self.cls_manager.models["random_forest"] = {
            "name": "random_forest",
            "model": self.rf_cls,
            "params": {"n_estimators": 100, "random_state": 42},
            "metrics": self.cls_metrics["random_forest"]
        }
        
        self.reg_manager.models["linear_regression"] = {
            "name": "linear_regression",
            "model": self.lin_reg,
            "params": {},
            "metrics": self.reg_metrics["linear_regression"]
        }
        
        self.reg_manager.models["random_forest"] = {
            "name": "random_forest",
            "model": self.rf_reg,
            "params": {"n_estimators": 100, "random_state": 42},
            "metrics": self.reg_metrics["random_forest"]
        }
        
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_save_and_load_classification_models(self):
        """Test saving and loading real classification models"""
        # Save both classification models
        log_reg_path = os.path.join(self.cls_config.model_path, "logistic_regression.pkl")
        rf_cls_path = os.path.join(self.cls_config.model_path, "random_forest.pkl")
        
        self.cls_manager.save_model("logistic_regression", filepath=log_reg_path)
        self.cls_manager.save_model("random_forest", filepath=rf_cls_path)
        
        # Check files exist
        self.assertTrue(os.path.exists(log_reg_path))
        self.assertTrue(os.path.exists(rf_cls_path))
        
        # Create new manager instance for loading
        new_cls_manager = SecureModelManager(
            self.cls_config,
            logger=self.logger,
            secret_key=self.test_key
        )
        
        # Load models
        loaded_log_reg = new_cls_manager.load_model(log_reg_path)
        loaded_rf_cls = new_cls_manager.load_model(rf_cls_path)
        
        # Check models loaded correctly
        self.assertIsNotNone(loaded_log_reg)
        self.assertIsNotNone(loaded_rf_cls)
        
        # Verify predictions match original models
        log_reg_orig_preds = self.log_reg.predict(self.X_cls_test)
        log_reg_loaded_preds = loaded_log_reg.predict(self.X_cls_test)
        np.testing.assert_array_equal(log_reg_orig_preds, log_reg_loaded_preds)
        
        rf_cls_orig_preds = self.rf_cls.predict(self.X_cls_test)
        rf_cls_loaded_preds = loaded_rf_cls.predict(self.X_cls_test)
        np.testing.assert_array_equal(rf_cls_orig_preds, rf_cls_loaded_preds)
        
        # Verify best model tracking (RF should be better)
        self.assertEqual(new_cls_manager.best_model["name"], "random_forest")
    
    def test_save_and_load_regression_models(self):
        """Test saving and loading real regression models"""
        # Save both regression models
        lin_reg_path = os.path.join(self.reg_config.model_path, "linear_regression.pkl")
        rf_reg_path = os.path.join(self.reg_config.model_path, "random_forest.pkl")
        
        self.reg_manager.save_model("linear_regression", filepath=lin_reg_path)
        self.reg_manager.save_model("random_forest", filepath=rf_reg_path)
        
        # Check files exist
        self.assertTrue(os.path.exists(lin_reg_path))
        self.assertTrue(os.path.exists(rf_reg_path))
        
        # Create new manager instance for loading
        new_reg_manager = SecureModelManager(
            self.reg_config,
            logger=self.logger,
            secret_key=self.test_key
        )
        
        # Load models
        loaded_lin_reg = new_reg_manager.load_model(lin_reg_path)
        loaded_rf_reg = new_reg_manager.load_model(rf_reg_path)
        
        # Check models loaded correctly
        self.assertIsNotNone(loaded_lin_reg)
        self.assertIsNotNone(loaded_rf_reg)
        
        # Verify predictions are correct (floating point, so use almost equal)
        lin_reg_orig_preds = self.lin_reg.predict(self.X_reg_test)
        lin_reg_loaded_preds = loaded_lin_reg.predict(self.X_reg_test)
        np.testing.assert_array_almost_equal(lin_reg_orig_preds, lin_reg_loaded_preds)
        
        rf_reg_orig_preds = self.rf_reg.predict(self.X_reg_test)
        rf_reg_loaded_preds = loaded_rf_reg.predict(self.X_reg_test)
        np.testing.assert_array_almost_equal(rf_reg_orig_preds, rf_reg_loaded_preds)
        
        # Verify best model tracking - either could be best depending on data
        best_model_name = new_reg_manager.best_model["name"]
        self.assertIn(best_model_name, ["linear_regression", "random_forest"])
    
    def test_model_with_access_code(self):
        """Test saving and loading models with access code protection"""
        # Save model with access code
        access_code = "secure_password_123"
        protected_path = os.path.join(self.cls_config.model_path, "protected_model.pkl")
        
        self.cls_manager.save_model("random_forest", filepath=protected_path, access_code=access_code)
        
        # Check file exists
        self.assertTrue(os.path.exists(protected_path))
        
        # Create new manager for loading
        new_manager = SecureModelManager(
            self.cls_config,
            logger=self.logger,
            secret_key=self.test_key
        )
        
        # Try to load without access code (should fail)
        loaded_model_fail = new_manager.load_model(protected_path)
        self.assertIsNone(loaded_model_fail)
        
        # Load with correct access code
        loaded_model = new_manager.load_model(protected_path, access_code=access_code)
        self.assertIsNotNone(loaded_model)
        
        # Verify predictions
        rf_orig_preds = self.rf_cls.predict(self.X_cls_test)
        rf_loaded_preds = loaded_model.predict(self.X_cls_test)
        np.testing.assert_array_equal(rf_orig_preds, rf_loaded_preds)
    
    def test_encryption_disabled(self):
        """Test functionality with encryption disabled"""
        # Create config with encryption disabled
        unencrypted_config = Config(enable_encryption=False)
        unencrypted_config.model_path = os.path.join(self.test_dir, "unencrypted")
        os.makedirs(unencrypted_config.model_path, exist_ok=True)
        
        # Create manager
        unencrypted_manager = SecureModelManager(
            unencrypted_config,
            logger=self.logger
        )
        
        # Add a model
        unencrypted_manager.models["unencrypted_model"] = {
            "name": "unencrypted_model",
            "model": self.log_reg,
            "params": {"max_iter": 1000},
            "metrics": {"accuracy": 0.85}
        }
        
        # Save model
        unenc_path = os.path.join(unencrypted_config.model_path, "unencrypted_model.pkl")
        result = unencrypted_manager.save_model("unencrypted_model", filepath=unenc_path)
        
        # Check save was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(unenc_path))
        
        # Load model with new manager
        new_unenc_manager = SecureModelManager(
            unencrypted_config,
            logger=self.logger
        )
        
        loaded_model = new_unenc_manager.load_model(unenc_path)
        
        # Verify model loaded correctly
        self.assertIsNotNone(loaded_model)
        
        # Verify predictions
        orig_preds = self.log_reg.predict(self.X_cls_test)
        loaded_preds = loaded_model.predict(self.X_cls_test)
        np.testing.assert_array_equal(orig_preds, loaded_preds)
    
    def test_multiple_models_best_tracking(self):
        """Test tracking the best model when loading multiple models"""
        # First, create models with different performance
        # For classification - different levels of performance
        X_cls, y_cls = make_classification(
            n_samples=1000, n_features=10, n_informative=5, 
            n_redundant=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42
        )
        
        # Create models with different performance levels
        models = {
            "model_good": LogisticRegression(C=1.0, random_state=42),
            "model_medium": LogisticRegression(C=0.1, random_state=42),
            "model_bad": LogisticRegression(C=0.01, random_state=42)
        }
        
        # Train models and record metrics
        model_metrics = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            model_metrics[name] = {"accuracy": accuracy, "f1": f1}
        
        # Create manager
        tracking_config = Config(task_type=TaskType.CLASSIFICATION)
        tracking_config.model_path = os.path.join(self.test_dir, "tracking")
        os.makedirs(tracking_config.model_path, exist_ok=True)
        
        tracking_manager = SecureModelManager(
            tracking_config,
            logger=self.logger,
            secret_key=self.test_key
        )
        
        # Add models to manager
        for name, model in models.items():
            tracking_manager.models[name] = {
                "name": name,
                "model": model,
                "params": {},
                "metrics": model_metrics[name]
            }
        
        # Save all models
        for name in models.keys():
            model_path = os.path.join(tracking_config.model_path, f"{name}.pkl")
            tracking_manager.save_model(name, filepath=model_path)
        
        # Create new manager to load models
        new_tracking_manager = SecureModelManager(
            tracking_config,
            logger=self.logger,
            secret_key=self.test_key
        )
        
        # Load models in a random order
        model_paths = [
            os.path.join(tracking_config.model_path, "model_medium.pkl"),
            os.path.join(tracking_config.model_path, "model_bad.pkl"),
            os.path.join(tracking_config.model_path, "model_good.pkl")
        ]
        
        for path in model_paths:
            new_tracking_manager.load_model(path)
        
        # Verify the best model tracking works (check that a best model is selected)
        self.assertIsNotNone(new_tracking_manager.best_model)
        self.assertIn("name", new_tracking_manager.best_model)
        
        # Find the model with the highest accuracy to verify it's selected as best
        best_accuracy = 0
        expected_best_name = None
        for name in models.keys():
            accuracy = model_metrics[name]["accuracy"]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                expected_best_name = name
        
        # Verify the best model is tracked correctly
        self.assertEqual(new_tracking_manager.best_model["name"], expected_best_name)
        
        # Verify that metrics were preserved
        for name in models.keys():
            self.assertIn(name, new_tracking_manager.models)
            self.assertAlmostEqual(
                new_tracking_manager.models[name]["metrics"]["accuracy"],
                model_metrics[name]["accuracy"]
            )
    
    def test_key_rotation(self):
        """Test rotation of encryption keys with real models"""
        # Save a model
        model_path = os.path.join(self.cls_config.model_path, "rotation_test.pkl")
        save_result = self.cls_manager.save_model("logistic_regression", filepath=model_path)
        
        # Check that save worked
        self.assertTrue(save_result, "Model save should succeed")
        self.assertTrue(os.path.exists(model_path), "Model file should exist after save")
        
        # Check encryption status before rotation
        self.assertTrue(self.cls_manager.encryption_enabled, "Encryption should be enabled")
        self.assertIsNotNone(self.cls_manager.cipher, "Cipher should be initialized")
        
        # Rotate the key
        new_password = "new_secure_password"
        result = self.cls_manager.rotate_encryption_key(new_password=new_password)
        
        # Check rotation succeeded
        self.assertIsNotNone(result, "rotate_encryption_key should not return None")
        self.assertTrue(result, "Key rotation should succeed")
        
        # Try to load the model with a new manager using the rotated key
        # We need to re-derive the key from the password
        salt_path = os.path.join(self.cls_config.model_path, '.salt')
        with open(salt_path, 'rb') as salt_file:
            salt = salt_file.read()
        
        # Create manager with manually derived key (simulating loading with new password)
        # This is implementation-specific - adjust based on actual implementation
        if self.cls_manager.use_scrypt:
            kdf = Scrypt(
                salt=salt,
                length=32,
                n=2**14,  # Match the value used in rotate_encryption_key method
                r=8,
                p=1,
                backend=default_backend()
            )
            key_bytes = kdf.derive(new_password.encode())
        else:
            kdf = PBKDF2HMAC(
                algorithm=getattr(hashes, self.cls_manager.hash_algorithm.upper())(),
                length=32,
                salt=salt,
                iterations=self.cls_manager.key_iterations,
                backend=default_backend()
            )
            key_bytes = kdf.derive(new_password.encode())
            
        new_key = base64.urlsafe_b64encode(key_bytes)
        
        # Create new manager with rotated key
        new_manager = SecureModelManager(
            self.cls_config,
            logger=self.logger,
            secret_key=new_key
        )
        
        # Load model with new key
        loaded_model = new_manager.load_model(model_path)
        
        # Verify model loaded correctly
        self.assertIsNotNone(loaded_model)
        
        # Verify predictions
        orig_preds = self.log_reg.predict(self.X_cls_test)
        loaded_preds = loaded_model.predict(self.X_cls_test)
        np.testing.assert_array_equal(orig_preds, loaded_preds)
    
    def test_verify_model_integrity(self):
        """Test verification of model integrity"""
        # Save a model
        model_path = os.path.join(self.cls_config.model_path, "integrity_test.pkl")
        self.cls_manager.save_model("random_forest", filepath=model_path)
        
        # Verify integrity (should pass)
        integrity_result = self.cls_manager.verify_model_integrity(model_path)
        self.assertTrue(integrity_result)
        
        # Tamper with the file
        with open(model_path, 'rb+') as f:
            content = bytearray(f.read())
            # Modify a byte near the middle of the file
            midpoint = len(content) // 2
            content[midpoint] = (content[midpoint] + 1) % 256
            f.seek(0)
            f.write(content)
        
        # Verify integrity again (should fail)
        integrity_result = self.cls_manager.verify_model_integrity(model_path)
        self.assertFalse(integrity_result)


class TestSecureModelManagerCustomModels(unittest.TestCase):
    """Tests using custom model classes"""
    
    class SimpleCustomModel:
        """A simple custom model class for testing"""
        def __init__(self, weights=None):
            self.weights = weights if weights is not None else np.random.random(10)
            
        def predict(self, X):
            """Simple prediction function"""
            # Just multiply input by weights and sum
            return np.dot(X, self.weights)
            
        def __eq__(self, other):
            """Check if two model instances are equal"""
            if not isinstance(other, TestSecureModelManagerCustomModels.SimpleCustomModel):
                return False
            return np.array_equal(self.weights, other.weights)
    
    def setUp(self):
        """Set up test environment with custom models"""
        # Create temporary directory for model storage
        self.test_dir = tempfile.mkdtemp()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create configuration
        self.config = Config(task_type=TaskType.REGRESSION)
        self.config.model_path = self.test_dir
        
        # Create test data
        np.random.seed(42)
        self.X = np.random.random((100, 10))
        
        # Create custom models
        self.model1 = self.SimpleCustomModel(weights=np.ones(10))
        self.model2 = self.SimpleCustomModel(weights=np.arange(10))
        
        # Calculate metrics
        y1 = self.model1.predict(self.X)
        y2 = self.model2.predict(self.X)
        y_true = np.random.random(100)  # Random ground truth
        
        mse1 = np.mean((y1 - y_true) ** 2)
        mse2 = np.mean((y2 - y_true) ** 2)
        
        # Create manager
        self.test_key = Fernet.generate_key()
        self.manager = SecureModelManager(
            self.config,
            logger=self.logger,
            secret_key=self.test_key
        )
        
        # Add models to manager
        self.manager.models["model1"] = {
            "name": "model1",
            "model": self.model1,
            "params": {},
            "metrics": {"mse": mse1}
        }
        
        self.manager.models["model2"] = {
            "name": "model2",
            "model": self.model2,
            "params": {},
            "metrics": {"mse": mse2}
        }
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_save_load_custom_models(self):
        """Test saving and loading custom model classes"""
        # Save models
        model1_path = os.path.join(self.config.model_path, "model1.pkl")
        model2_path = os.path.join(self.config.model_path, "model2.pkl")
        
        self.manager.save_model("model1", filepath=model1_path)
        self.manager.save_model("model2", filepath=model2_path)
        
        # Check files exist
        self.assertTrue(os.path.exists(model1_path))
        self.assertTrue(os.path.exists(model2_path))
        
        # Create new manager
        new_manager = SecureModelManager(
            self.config,
            logger=self.logger,
            secret_key=self.test_key
        )
        
        # Load models
        loaded_model1 = new_manager.load_model(model1_path)
        loaded_model2 = new_manager.load_model(model2_path)
        
        # Check models loaded correctly
        self.assertIsNotNone(loaded_model1)
        self.assertIsNotNone(loaded_model2)
        
        # Verify models are equal to original
        self.assertEqual(loaded_model1, self.model1)
        self.assertEqual(loaded_model2, self.model2)
        
        # Verify predictions
        np.testing.assert_array_equal(
            loaded_model1.predict(self.X),
            self.model1.predict(self.X)
        )
        
        np.testing.assert_array_equal(
            loaded_model2.predict(self.X),
            self.model2.predict(self.X)
        )


if __name__ == '__main__':
    unittest.main()