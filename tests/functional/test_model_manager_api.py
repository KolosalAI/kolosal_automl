import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import os
import uuid
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Set environment variables before importing the app
os.environ["API_KEYS"] = "test_key"

# Import the FastAPI app and related classes
from modules.api.model_manager_api import (
    app, ModelManagerException, get_or_create_manager, 
    validate_manager_exists, manager_instances, TaskType,
    ManagerConfigModel, ModelSaveRequest, ModelLoadRequest
)
from modules.model_manager import SecureModelManager

class TestModelManagerAPI(unittest.TestCase):
    """Tests for the Secure Model Manager API"""

    def setUp(self):
        """Set up test client and mocks before each test"""
        self.client = TestClient(app)
        # Clear manager instances between tests
        manager_instances.clear()
        # Common headers for authenticated requests
        self.api_key_headers = {"X-API-Key": "test_key"}
        self.bearer_headers = {"Authorization": "Bearer test_token"}
        
    def tearDown(self):
        """Clean up after each test"""
        # Clear manager instances
        manager_instances.clear()
        
    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        
    def test_api_info(self):
        """Test the API info endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Secure Model Manager API"
        assert data["version"] == "0.1.4"
        assert "total_managers" in data

    @patch("modules.api.model_manager_api.SecureModelManager")
    def test_create_manager(self, mock_manager_class):
        """Test creating a new manager"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager_class.VERSION = "0.1.4"
        
        # Test data
        config_data = {
            "model_path": "./test_models",
            "task_type": "regression",
            "enable_encryption": True,
            "use_scrypt": True,
            "primary_metric": "mse"
        }
        
        # Make request
        response = self.client.post(
            "/api/managers", 
            json=config_data,
            headers=self.api_key_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "manager_id" in data["details"]
        
        # Verify manager was created with correct config
        manager_id = data["details"]["manager_id"]
        assert manager_id in manager_instances
        
        # Verify SecureModelManager was instantiated with correct params
        mock_manager_class.assert_called_once()
        
    def test_create_manager_without_api_key(self):
        """Test that creating a manager without API key fails"""
        config_data = {
            "model_path": "./test_models",
            "task_type": "regression"
        }
        
        response = self.client.post("/api/managers", json=config_data)
        assert response.status_code == 401
        
    @patch("modules.api.model_manager_api.SecureModelManager")
    def test_create_manager_invalid_task_type(self, mock_manager_class):
        """Test creating a manager with invalid task type"""
        config_data = {
            "model_path": "./test_models",
            "task_type": "invalid_type",
            "enable_encryption": True
        }
        
        response = self.client.post(
            "/api/managers", 
            json=config_data,
            headers=self.api_key_headers
        )
        
        assert response.status_code == 422  # FastAPI validation error
        response_data = response.json()
        assert "detail" in response_data
        
    def test_list_managers_empty(self):
        """Test listing managers when none exist"""
        response = self.client.get(
            "/api/managers",
            headers=self.api_key_headers
        )
        
        assert response.status_code == 200
        assert response.json() == {}
        
    @patch("modules.api.model_manager_api.SecureModelManager")
    def test_list_managers(self, mock_manager_class):
        """Test listing managers with existing managers"""
        # Create a mock manager first
        mock_manager = MagicMock()
        mock_manager.encryption_enabled = True
        mock_manager.config.model_path = "./test_path"
        mock_manager.config.task_type = TaskType.REGRESSION
        mock_manager.models = {"model1": {}, "model2": {}}
        # Create a mock best_model dict with a name key (to match API expectation)
        mock_manager.best_model = {"name": "model1"}
        mock_manager_class.return_value = mock_manager
        
        # Add a manager to the instances
        manager_id = str(uuid.uuid4())
        manager_instances[manager_id] = mock_manager
        
        # Make request
        response = self.client.get(
            "/api/managers",
            headers=self.api_key_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert manager_id in data
        manager_data = data[manager_id]
        assert manager_data["encryption_enabled"]
        assert manager_data["model_path"] == "./test_path"
        assert manager_data["task_type"] == "regression"  # API returns lowercase
        assert manager_data["models_count"] == 2
        assert manager_data["best_model"] == "model1"
        
    def test_validate_manager_exists_nonexistent(self):
        """Test validate_manager_exists with non-existent ID"""
        with self.assertRaises(ModelManagerException) as context:
            validate_manager_exists("nonexistent_id")
            
        assert context.exception.status_code == 404
        assert "not found" in context.exception.detail
        
    @patch("modules.api.model_manager_api.SecureModelManager")
    def test_save_model(self, mock_manager_class):
        """Test saving a model"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.save_model.return_value = True
        mock_manager.encryption_enabled = True
        mock_manager.config.model_path = "./test_path"
        mock_manager_class.return_value = mock_manager
        
        # Add a manager to the instances
        manager_id = str(uuid.uuid4())
        manager_instances[manager_id] = mock_manager
        
        # Test data
        save_request = {
            "model_name": "test_model",
            "access_code": "test_password",
            "compression_level": 5
        }
        
        # Make request
        response = self.client.post(
            f"/api/managers/{manager_id}/models/save",
            json=save_request,
            headers=self.bearer_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "test_model" in data["message"]
        
        # Verify save_model was called with correct params
        mock_manager.save_model.assert_called_once_with(
            model_name="test_model",
            filepath=None,
            access_code="test_password",
            compression_level=5
        )
        
    @patch("modules.api.model_manager_api.SecureModelManager")
    def test_save_model_failure(self, mock_manager_class):
        """Test saving a model with failure"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.save_model.return_value = False
        mock_manager_class.return_value = mock_manager
        
        # Add a manager to the instances
        manager_id = str(uuid.uuid4())
        manager_instances[manager_id] = mock_manager
        
        # Test data
        save_request = {
            "model_name": "test_model",
            "access_code": "test_password"
        }
        
        # Make request
        response = self.client.post(
            f"/api/managers/{manager_id}/models/save",
            json=save_request,
            headers=self.bearer_headers
        )
        
        # Assertions
        assert response.status_code == 500
        assert "Failed to save model" in response.json()["detail"]
        
    @patch("modules.api.model_manager_api.SecureModelManager")
    @patch("os.path.exists")
    def test_load_model(self, mock_exists, mock_manager_class):
        """Test loading a model"""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.load_model.return_value = MagicMock()  # Return a model
        # Create a mock best_model object with a name attribute
        mock_best_model = MagicMock()
        mock_best_model.name = "other_model"
        mock_manager.best_model = mock_best_model
        mock_manager_class.return_value = mock_manager
        mock_exists.return_value = True
        
        # Add a manager to the instances
        manager_id = str(uuid.uuid4())
        manager_instances[manager_id] = mock_manager
        
        # Test data
        load_request = {
            "filepath": "./test_path/test_model.pkl",
            "access_code": "test_password"
        }
        
        # Make request
        response = self.client.post(
            f"/api/managers/{manager_id}/models/load",
            json=load_request,
            headers=self.bearer_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "test_model" in data["message"]
        assert data["details"]["is_best_model"] == False
        
        # Verify load_model was called with correct params
        mock_manager.load_model.assert_called_once_with(
            filepath="./test_path/test_model.pkl",
            access_code="test_password"
        )
        
    @patch("modules.api.model_manager_api.SecureModelManager")
    @patch("os.path.exists")
    def test_load_model_file_not_found(self, mock_exists, mock_manager_class):
        """Test loading a model with file not found"""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_exists.return_value = False
        
        # Add a manager to the instances
        manager_id = str(uuid.uuid4())
        manager_instances[manager_id] = mock_manager
        
        # Test data
        load_request = {
            "filepath": "./test_path/nonexistent_model.pkl",
            "access_code": "test_password"
        }
        
        # Make request
        response = self.client.post(
            f"/api/managers/{manager_id}/models/load",
            json=load_request,
            headers=self.bearer_headers
        )
        
        # Assertions
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
        
    @patch("modules.api.model_manager_api.SecureModelManager")
    def test_list_models(self, mock_manager_class):
        """Test listing models"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.models = {"model1": {}, "model2": {}}
        # Create a mock best_model object with a name attribute
        mock_best_model = MagicMock()
        mock_best_model.name = "model1"
        mock_manager.best_model = mock_best_model
        mock_manager_class.return_value = mock_manager
        
        # Add a manager to the instances
        manager_id = str(uuid.uuid4())
        manager_instances[manager_id] = mock_manager
        
        # Make request
        response = self.client.get(
            f"/api/managers/{manager_id}/models",
            headers=self.bearer_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == ["model1", "model2"]
        assert data["best_model"] == "model1"
        
    @patch("modules.api.model_manager_api.SecureModelManager")
    @patch("os.path.exists")
    def test_verify_model(self, mock_exists, mock_manager_class):
        """Test verifying a model"""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.verify_model_integrity.return_value = True
        mock_manager_class.return_value = mock_manager
        mock_exists.return_value = True
        
        # Mock open and pickle to simulate reading file
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch("pickle.load") as mock_pickle_load:
                mock_pickle_load.return_value = {"encrypted_data": "encrypted"}
                
                # Add a manager to the instances
                manager_id = str(uuid.uuid4())
                manager_instances[manager_id] = mock_manager
                
                # Make request
                response = self.client.post(
                    f"/api/managers/{manager_id}/verify",
                    params={"filepath": "./test_path/test_model.pkl"},
                    headers=self.bearer_headers
                )
                
                # Assertions
                assert response.status_code == 200
                data = response.json()
                assert data["filepath"] == "./test_path/test_model.pkl"
                assert data["is_valid"]
                assert data["encryption_status"] == "encrypted"
                
                # Verify verify_model_integrity was called
                mock_manager.verify_model_integrity.assert_called_once_with(
                    "./test_path/test_model.pkl"
                )
    
    @patch("modules.api.model_manager_api.SecureModelManager")
    def test_rotate_encryption_key(self, mock_manager_class):
        """Test rotating encryption key"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.encryption_enabled = True
        mock_manager.rotate_encryption_key.return_value = True
        mock_manager_class.return_value = mock_manager
        
        # Add a manager to the instances
        manager_id = str(uuid.uuid4())
        manager_instances[manager_id] = mock_manager
        
        # Test data
        rotate_request = {
            "new_password": "new_test_password"
        }
        
        # Make request
        response = self.client.post(
            f"/api/managers/{manager_id}/rotate-key",
            json=rotate_request,
            headers=self.bearer_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "rotated successfully" in data["message"]
        assert data["details"]["using_password"]
        
        # Verify rotate_encryption_key was called
        mock_manager.rotate_encryption_key.assert_called_once_with(
            new_password="new_test_password"
        )
        
    @patch("modules.api.model_manager_api.SecureModelManager")
    def test_rotate_encryption_key_encryption_disabled(self, mock_manager_class):
        """Test rotating encryption key when encryption is disabled"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.encryption_enabled = False
        mock_manager_class.return_value = mock_manager
        
        # Add a manager to the instances
        manager_id = str(uuid.uuid4())
        manager_instances[manager_id] = mock_manager
        
        # Test data
        rotate_request = {
            "new_password": "new_test_password"
        }
        
        # Make request
        response = self.client.post(
            f"/api/managers/{manager_id}/rotate-key",
            json=rotate_request,
            headers=self.bearer_headers
        )
        
        # Assertions
        assert response.status_code == 400
        assert "not enabled" in response.json()["detail"]
        
    @patch("modules.api.model_manager_api.SecureModelManager")
    @patch("os.makedirs")
    @patch("os.chmod")
    def test_upload_model(self, mock_chmod, mock_makedirs, mock_manager_class):
        """Test uploading a model file"""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.config.model_path = "./test_path"
        mock_manager_class.return_value = mock_manager
        
        # Add a manager to the instances
        manager_id = str(uuid.uuid4())
        manager_instances[manager_id] = mock_manager
        
        # Create mock file data
        file_content = b"mock file content"
        
        # Make request with file upload
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            response = self.client.post(
                f"/api/managers/{manager_id}/upload-model",
                files={"model_file": ("test_model.pkl", file_content)},
                params={"access_code": "test_password"},
                headers=self.bearer_headers
            )
            
            # Assertions
            assert response.status_code == 200
            data = response.json()
            assert data["success"]
            assert "uploaded successfully" in data["message"]
            assert data["details"]["model_name"] == "test_model"
            
            # Verify directories were created
            mock_makedirs.assert_called_once_with(mock_manager.config.model_path, exist_ok=True)
            
            # Verify file permissions were set
            mock_chmod.assert_called_once()
            
    @patch("modules.api.model_manager_api.SecureModelManager")
    def test_delete_manager(self, mock_manager_class):
        """Test deleting a manager"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        # Add a manager to the instances
        manager_id = str(uuid.uuid4())
        manager_instances[manager_id] = mock_manager
        
        # Make request
        response = self.client.delete(
            f"/api/managers/{manager_id}",
            headers=self.api_key_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "deleted successfully" in data["message"]
        
        # Verify manager was removed
        assert manager_id not in manager_instances
        
    def test_get_or_create_manager_missing_config(self):
        """Test get_or_create_manager with missing config"""
        manager_id = str(uuid.uuid4())
        
        with self.assertRaises(ModelManagerException) as context:
            get_or_create_manager(manager_id)
            
        assert context.exception.status_code == 400
        assert "no configuration was provided" in context.exception.detail

if __name__ == "__main__":
    unittest.main()