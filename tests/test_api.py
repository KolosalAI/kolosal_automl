import unittest
import json
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient
from fastapi import UploadFile
from io import BytesIO

# Import the main application
from api import app, get_training_engine, get_inference_engine, get_secure_manager, get_preprocessor, get_quantizer

# Create test client
client = TestClient(app)

class MockUser:
    def __init__(self, username="testuser", roles=None):
        self.username = username
        self.roles = roles or ["user"]

class MockFile:
    def __init__(self, content, filename="test.csv"):
        self.file = BytesIO(content)
        self.filename = filename
    
    def read(self):
        return self.file.read()
    
    def seek(self, pos):
        return self.file.seek(pos)

class TestMLAPI(unittest.TestCase):
    """Base test class with setup and utility methods"""
    
    @classmethod
    def setUpClass(cls):
        """Set up mock engines and test data once for all tests"""
        # Create mock engines
        cls.mock_train_engine = MagicMock()
        cls.mock_inference_engine = MagicMock()
        cls.mock_secure_manager = MagicMock()
        cls.mock_preprocessor = MagicMock()
        cls.mock_quantizer = MagicMock()
        
        # Create test data
        cls.test_csv_content = b"col1,col2,target\n1,2,0\n3,4,1\n5,6,0"
        
        # Create test models data
        cls.test_models = [
            {"name": "model1", "path": "/models/model1.pkl", "size": 1000, "modified": "2023-01-01T00:00:00"},
            {"name": "model2", "path": "/models/model2.pkl", "size": 2000, "modified": "2023-01-02T00:00:00"}
        ]
    
    def setUp(self):
        """Set up before each test case"""
        # Apply patches for engines
        self.train_engine_patcher = patch('main.get_training_engine', return_value=self.mock_train_engine)
        self.inference_engine_patcher = patch('main.get_inference_engine', return_value=self.mock_inference_engine)
        self.secure_manager_patcher = patch('main.get_secure_manager', return_value=self.mock_secure_manager)
        self.preprocessor_patcher = patch('main.get_preprocessor', return_value=self.mock_preprocessor)
        self.quantizer_patcher = patch('main.get_quantizer', return_value=self.mock_quantizer)
        
        # Start patches
        self.mock_get_train_engine = self.train_engine_patcher.start()
        self.mock_get_inference_engine = self.inference_engine_patcher.start()
        self.mock_get_secure_manager = self.secure_manager_patcher.start()
        self.mock_get_preprocessor = self.preprocessor_patcher.start()
        self.mock_get_quantizer = self.quantizer_patcher.start()
        
        # Mock file operations
        self.file_patcher = patch('builtins.open', mock_open())
        self.mock_file = self.file_patcher.start()
        
        # Mock OS path operations
        self.os_path_exists_patcher = patch('os.path.exists', return_value=True)
        self.mock_os_path_exists = self.os_path_exists_patcher.start()
        
        self.os_path_getsize_patcher = patch('os.path.getsize', return_value=1000)
        self.mock_os_path_getsize = self.os_path_getsize_patcher.start()
        
        self.os_path_getmtime_patcher = patch('os.path.getmtime', return_value=1672531200) # 2023-01-01
        self.mock_os_path_getmtime = self.os_path_getmtime_patcher.start()
        
        # Create valid JWT token for testing
        self.valid_token = self.get_auth_token()
    
    def tearDown(self):
        """Clean up after each test case"""
        # Stop all patches
        self.train_engine_patcher.stop()
        self.inference_engine_patcher.stop()
        self.secure_manager_patcher.stop()
        self.preprocessor_patcher.stop()
        self.quantizer_patcher.stop()
        self.file_patcher.stop()
        self.os_path_exists_patcher.stop()
        self.os_path_getsize_patcher.stop()
        self.os_path_getmtime_patcher.stop()
    
    def get_auth_token(self):
        """Helper to get an auth token for testing"""
        with patch('main.USERS', {"testuser": {"password": "$2b$12$TestHashedPassword", "roles": ["user"]}}):
            with patch('bcrypt.checkpw', return_value=True):
                response = client.post(
                    "/api/login",
                    json={"username": "testuser", "password": "password123"}
                )
                return response.json()["token"]
    
    def get_auth_header(self, admin=False):
        """Helper to get auth headers for testing"""
        with patch('main.USERS', {
                "testuser": {"password": "$2b$12$TestHashedPassword", "roles": ["user"]},
                "adminuser": {"password": "$2b$12$TestHashedPassword", "roles": ["admin"]}
            }):
            with patch('bcrypt.checkpw', return_value=True):
                username = "adminuser" if admin else "testuser"
                response = client.post(
                    "/api/login",
                    json={"username": username, "password": "password123"}
                )
                token = response.json()["token"]
                return {"Authorization": f"Bearer {token}"}
    
    def mock_jwt_decode(self, roles=None):
        """Helper to mock JWT decode"""
        if roles is None:
            roles = ["user"]
        
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": roles}):
            return True

class TestAuthentication(TestMLAPI):
    """Test authentication and authorization endpoints"""
    
    def test_login_success(self):
        """Test successful login"""
        with patch('main.USERS', {"testuser": {"password": "$2b$12$TestHashedPassword", "roles": ["user"]}}):
            with patch('bcrypt.checkpw', return_value=True):
                response = client.post(
                    "/api/login",
                    json={"username": "testuser", "password": "password123"}
                )
                
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn("token", data)
                self.assertEqual(data["username"], "testuser")
                self.assertEqual(data["roles"], ["user"])
    
    def test_login_invalid_user(self):
        """Test login with invalid username"""
        with patch('main.USERS', {"testuser": {"password": "$2b$12$TestHashedPassword", "roles": ["user"]}}):
            response = client.post(
                "/api/login",
                json={"username": "invaliduser", "password": "password123"}
            )
            
            self.assertEqual(response.status_code, 401)
            self.assertIn("Invalid username or password", response.json()["detail"])
    
    def test_login_invalid_password(self):
        """Test login with invalid password"""
        with patch('main.USERS', {"testuser": {"password": "$2b$12$TestHashedPassword", "roles": ["user"]}}):
            with patch('bcrypt.checkpw', return_value=False):
                response = client.post(
                    "/api/login",
                    json={"username": "testuser", "password": "wrongpassword"}
                )
                
                self.assertEqual(response.status_code, 401)
                self.assertIn("Invalid username or password", response.json()["detail"])
    
    def test_protected_endpoint_unauthorized(self):
        """Test accessing protected endpoint without auth token"""
        response = client.get("/api/models")
        
        self.assertEqual(response.status_code, 401)
        self.assertIn("Not authenticated", response.json()["detail"])
    
    def test_protected_endpoint_with_token(self):
        """Test accessing protected endpoint with auth token"""
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            response = client.get(
                "/api/models",
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 200)
    
    def test_admin_endpoint_with_user_token(self):
        """Test accessing admin endpoint with user token"""
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            response = client.get(
                "/api/users",
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 403)
            self.assertIn("don't have permission", response.json()["detail"])
    
    def test_admin_endpoint_with_admin_token(self):
        """Test accessing admin endpoint with admin token"""
        with patch('jwt.decode', return_value={"sub": "adminuser", "roles": ["admin"]}):
            response = client.get(
                "/api/users",
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 200)

class TestModelManagement(TestMLAPI):
    """Test model management endpoints"""
    
    def test_list_models(self):
        """Test listing models"""
        with patch('pathlib.Path.glob', return_value=[
                MagicMock(name="model1.pkl", stem="model1", stat=lambda: MagicMock(st_size=1000, st_mtime=1672531200)),
                MagicMock(name="model2.pkl", stem="model2", stat=lambda: MagicMock(st_size=2000, st_mtime=1672617600))
            ]):
            with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
                response = client.get(
                    "/api/models",
                    headers={"Authorization": f"Bearer {self.valid_token}"}
                )
                
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(len(data["models"]), 2)
                self.assertEqual(data["count"], 2)
                self.assertEqual(data["models"][0]["name"], "model1")
                self.assertEqual(data["models"][1]["name"], "model2")
    
    def test_get_model_info(self):
        """Test getting model information"""
        self.mock_inference_engine.load_model.return_value = True
        self.mock_inference_engine.get_model_info.return_value = {
            "model_type": "RandomForestClassifier",
            "features": 10,
            "classes": 2
        }
        
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            response = client.get(
                "/api/models/model1",
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["name"], "model1")
            self.assertEqual(data["model_type"], "RandomForestClassifier")
            self.assertEqual(data["features"], 10)
            self.assertEqual(data["classes"], 2)
    
    def test_delete_model(self):
        """Test deleting a model"""
        with patch('jwt.decode', return_value={"sub": "adminuser", "roles": ["admin"]}):
            with patch('os.remove') as mock_remove:
                response = client.delete(
                    "/api/models/model1",
                    headers={"Authorization": f"Bearer {self.valid_token}"}
                )
                
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["name"], "model1")
                self.assertIn("successfully deleted", data["message"])
                mock_remove.assert_called()
    
    def test_quantize_model(self):
        """Test quantizing a model"""
        self.mock_quantizer.quantize.return_value = (np.array([1, 2, 3]), {})
        
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            with patch('pickle.load'):
                with patch('pickle.dumps', return_value=b'test_bytes'):
                    response = client.post(
                        "/api/quantize/model1",
                        json={"quantization_type": "int8", "quantization_mode": "dynamic_per_batch"},
                        headers={"Authorization": f"Bearer {self.valid_token}"}
                    )
                    
                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertEqual(data["message"], "Model model1 quantized successfully")
                    self.assertTrue(data["compression_ratio"] > 0)

class TestPrediction(TestMLAPI):
    """Test prediction endpoints"""
    
    def test_predict_with_json_data(self):
        """Test prediction with JSON data"""
        self.mock_inference_engine.load_model.return_value = True
        self.mock_inference_engine.predict.return_value = (True, [0, 1, 0], {"confidence": [0.8, 0.7, 0.9]})
        
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            response = client.post(
                "/api/predict?model=model1",
                json={"model": "model1", "data": [[1, 2], [3, 4], [5, 6]]},
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["model"], "model1")
            self.assertEqual(data["predictions"], [0, 1, 0])
            self.assertEqual(data["sample_count"], 3)
            self.assertEqual(data["metadata"]["confidence"], [0.8, 0.7, 0.9])
    
    def test_predict_with_file_upload(self):
        """Test prediction with file upload"""
        self.mock_inference_engine.load_model.return_value = True
        self.mock_inference_engine.predict.return_value = (True, [0, 1, 0], {"confidence": [0.8, 0.7, 0.9]})
        
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            with patch('main.parse_data', return_value=np.array([[1, 2], [3, 4], [5, 6]])):
                with patch('main.temp_file_manager') as mock_temp_manager:
                    mock_temp_manager.return_value.__enter__.return_value = "/tmp/test_file.csv"
                    mock_temp_manager.return_value.__exit__.return_value = None
                    
                    files = {"file": ("test.csv", self.test_csv_content, "text/csv")}
                    response = client.post(
                        "/api/predict?model=model1",
                        files=files,
                        headers={"Authorization": f"Bearer {self.valid_token}"}
                    )
                    
                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertEqual(data["model"], "model1")
                    self.assertEqual(data["predictions"], [0, 1, 0])
    
    def test_predict_batch(self):
        """Test batch prediction"""
        mock_future = MagicMock()
        mock_future.result.return_value = ([0, 1, 0], {"confidence": [0.8, 0.7, 0.9]})
        
        self.mock_inference_engine.load_model.return_value = True
        self.mock_inference_engine.predict_batch.return_value = mock_future
        
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            response = client.post(
                "/api/predict?model=model1&batch_size=2",
                json={"model": "model1", "data": [[1, 2], [3, 4], [5, 6]]},
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["model"], "model1")
            self.assertEqual(data["predictions"], [0, 1, 0])
            self.assertEqual(data["sample_count"], 3)
            
            self.mock_inference_engine.predict_batch.assert_called_with(
                np.array([[1, 2], [3, 4], [5, 6]]), batch_size=2
            )

class TestTraining(TestMLAPI):
    """Test training endpoints"""
    
    def test_train_model(self):
        """Test training a model"""
        mock_model = MagicMock()
        mock_metrics = {
            "accuracy": 0.85,
            "precision": 0.8,
            "recall": 0.75,
            "f1": 0.77
        }
        
        self.mock_train_engine.train_model.return_value = (mock_model, mock_metrics)
        self.mock_train_engine.save_model.return_value = (True, "/models/new_model.pkl")
        
        with patch('jwt.decode', return_value={"sub": "adminuser", "roles": ["admin"]}):
            with patch('main.parse_data') as mock_parse:
                mock_df = pd.DataFrame({
                    "feature1": [1, 2, 3],
                    "feature2": [4, 5, 6],
                    "target": [0, 1, 0]
                })
                mock_parse.return_value = mock_df
                
                with patch('main.temp_file_manager') as mock_temp_manager:
                    mock_temp_manager.return_value.__enter__.return_value = "/tmp/test_file.csv"
                    mock_temp_manager.return_value.__exit__.return_value = None
                    
                    files = {"file": ("test.csv", self.test_csv_content, "text/csv")}
                    response = client.post(
                        "/api/train",
                        files=files,
                        data={"model_type": "classification", "model_name": "new_model", "target_column": "target"},
                        headers={"Authorization": f"Bearer {self.valid_token}"}
                    )
                    
                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertEqual(data["model_name"], "new_model")
                    self.assertIn("trained successfully", data["message"])
                    self.assertEqual(data["metrics"]["accuracy"], 0.85)

class TestDataProcessing(TestMLAPI):
    """Test data processing endpoints"""
    
    def test_preprocess_data(self):
        """Test data preprocessing endpoint"""
        mock_preprocessed_data = pd.DataFrame({
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [0.4, 0.5, 0.6]
        })
        mock_stats = {
            "missing_values": 0,
            "outliers": 0
        }
        
        self.mock_preprocessor.fit_transform.return_value = mock_preprocessed_data
        self.mock_preprocessor.get_stats.return_value = mock_stats
        
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            with patch('main.parse_data', return_value=pd.DataFrame({
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6]
            })):
                with patch('main.temp_file_manager') as mock_temp_manager:
                    mock_temp_manager.return_value.__enter__.return_value = "/tmp/test_file.csv"
                    mock_temp_manager.return_value.__exit__.return_value = None
                    
                    files = {"file": ("test.csv", self.test_csv_content, "text/csv")}
                    
                    # Mock the FileResponse
                    with patch('fastapi.responses.FileResponse', return_value=MagicMock(status_code=200)):
                        response = client.post(
                            "/api/preprocess",
                            files=files,
                            data={"normalize": "true", "handle_missing": "true", "detect_outliers": "true"},
                            headers={"Authorization": f"Bearer {self.valid_token}"}
                        )
                        
                        self.assertEqual(response.status_code, 200)
    
    def test_drift_detection(self):
        """Test drift detection endpoint"""
        mock_drift_results = {
            "dataset_drift": 0.15,
            "drift_detected": True,
            "drifted_features": ["feature1"],
            "feature_drift": {"feature1": 0.3, "feature2": 0.05}
        }
        
        self.mock_train_engine.detect_data_drift.return_value = mock_drift_results
        
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            with patch('main.parse_data', side_effect=[
                pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]}),  # reference
                pd.DataFrame({"feature1": [2, 3, 4], "feature2": [4, 5, 6]})   # new
            ]):
                with patch('main.temp_file_manager') as mock_temp_manager:
                    # Need to return different paths for different files
                    mock_temp_manager.return_value.__enter__.side_effect = ["/tmp/ref.csv", "/tmp/new.csv"]
                    mock_temp_manager.return_value.__exit__.return_value = None
                    
                    ref_file = {"reference_file": ("ref.csv", self.test_csv_content, "text/csv")}
                    new_file = {"new_file": ("new.csv", self.test_csv_content, "text/csv")}
                    
                    response = client.post(
                        "/api/drift-detection",
                        files={**ref_file, **new_file},
                        data={"threshold": 0.1},
                        headers={"Authorization": f"Bearer {self.valid_token}"}
                    )
                    
                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertEqual(data["dataset_drift"], 0.15)
                    self.assertTrue(data["drift_detected"])
                    self.assertEqual(data["drifted_features"], ["feature1"])

class TestSecurity(TestMLAPI):
    """Test security endpoints"""
    
    def test_secure_model(self):
        """Test securing a model"""
        self.mock_secure_manager.load_model.return_value = MagicMock()
        self.mock_secure_manager.save_model.return_value = True
        
        with patch('jwt.decode', return_value={"sub": "adminuser", "roles": ["admin"]}):
            response = client.post(
                "/api/models/secure/model1",
                json={"access_code": "secret123"},
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("secured successfully", data["message"])
            self.assertIn("secure_path", data)
    
    def test_verify_model(self):
        """Test verifying a model's integrity"""
        self.mock_secure_manager.verify_model_integrity.return_value = True
        
        with patch('jwt.decode', return_value={"sub": "adminuser", "roles": ["admin"]}):
            response = client.get(
                "/api/models/verify/model1",
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "valid")
            self.assertIn("verification successful", data["message"])

class TestErrorHandling(TestMLAPI):
    """Test error handling in the API"""
    
    def test_invalid_model_name(self):
        """Test providing an invalid model name"""
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            response = client.get(
                "/api/models/invalid/../model",
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 400)
            self.assertIn("Invalid model name", response.json()["detail"])
    
    def test_model_not_found(self):
        """Test requesting a model that doesn't exist"""
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            with patch('os.path.exists', return_value=False):
                response = client.get(
                    "/api/models/nonexistent",
                    headers={"Authorization": f"Bearer {self.valid_token}"}
                )
                
                self.assertEqual(response.status_code, 404)
                self.assertIn("Model not found", response.json()["detail"])
    
    def test_invalid_file_type(self):
        """Test uploading an invalid file type"""
        with patch('jwt.decode', return_value={"sub": "adminuser", "roles": ["admin"]}):
            files = {"file": ("test.exe", b"invalid content", "application/octet-stream")}
            response = client.post(
                "/api/train",
                files=files,
                data={"model_type": "classification", "target_column": "target"},
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 400)
            self.assertIn("File type not allowed", response.json()["detail"])
    
    def test_missing_data(self):
        """Test prediction without providing data"""
        with patch('jwt.decode', return_value={"sub": "testuser", "roles": ["user"]}):
            response = client.post(
                "/api/predict?model=model1",
                headers={"Authorization": f"Bearer {self.valid_token}"}
            )
            
            self.assertEqual(response.status_code, 400)
            self.assertIn("No data provided", response.json()["detail"])

class TestUserManagement(TestMLAPI):
    """Test user management endpoints"""
    
    def test_list_users(self):
        """Test listing users"""
        with patch('main.USERS', {
                "user1": {"password": "hashed1", "roles": ["user"]},
                "admin": {"password": "hashed2", "roles": ["admin"]}
            }):
            with patch('jwt.decode', return_value={"sub": "adminuser", "roles": ["admin"]}):
                response = client.get(
                    "/api/users",
                    headers={"Authorization": f"Bearer {self.valid_token}"}
                )
                
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(len(data["users"]), 2)
                self.assertEqual(data["count"], 2)
    
    def test_create_user(self):
        """Test creating a new user"""
        users = {
            "user1": {"password": "hashed1", "roles": ["user"]}
        }
        
        with patch('main.USERS', users):
            with patch('jwt.decode', return_value={"sub": "adminuser", "roles": ["admin"]}):
                with patch('bcrypt.hashpw', return_value=b"newhashed") as mock_hashpw:
                    with patch('bcrypt.gensalt', return_value=b"salt"):
                        response = client.post(
                            "/api/users",
                            json={"username": "newuser", "password": "password123", "roles": ["user"]},
                            headers={"Authorization": f"Bearer {self.valid_token}"}
                        )
                        
                        self.assertEqual(response.status_code, 201)
                        data = response.json()
                        self.assertEqual(data["username"], "newuser")
                        self.assertEqual(data["roles"], ["user"])
                        self.assertIn("newuser", users)
                        mock_hashpw.assert_called_once()
    
    def test_delete_user(self):
        """Test deleting a user"""
        users = {
            "user1": {"password": "hashed1", "roles": ["user"]},
            "user2": {"password": "hashed2", "roles": ["user"]},
            "admin": {"password": "hashed3", "roles": ["admin"]}
        }
        
        with patch('main.USERS', users):
            with patch('jwt.decode', return_value={"sub": "admin", "roles": ["admin"]}):
                response = client.delete(
                    "/api/users/user2",
                    headers={"Authorization": f"Bearer {self.valid_token}"}
                )
                
                self.assertEqual(response.status_code, 200)
                self.assertIn("deleted successfully", response.json()["message"])
                self.assertNotIn("user2", users)

if __name__ == "__main__":
    unittest.main()