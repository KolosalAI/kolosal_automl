import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from fastapi import status

try:
    from fastapi.testclient import TestClient
    from modules.api.inference_engine_api import app, InferenceEngine, ModelType, BatchPriority, EngineState
except ImportError as e:
    pytest.skip(f"API modules not available: {e}", allow_module_level=True)


@pytest.mark.functional
class TestInferenceEngineAPI:
    """Tests for the InferenceEngine API functionality"""

    @pytest.fixture(autouse=True)
    def setup_api_test(self):
        """Set up test environment before each test case"""
        # Create test client
        self.client = TestClient(app)
        
        # Mock the InferenceEngine
        self.mock_engine = Mock(spec=InferenceEngine)
        self.mock_engine.model = Mock()
        self.mock_engine.state = EngineState.READY
        self.mock_engine.active_requests = 0
        self.mock_engine.model_info = {"name": "test_model", "version": "1.0"}
        self.mock_engine.feature_names = ["feature1", "feature2", "feature3"]
        
        # Add missing methods to mock
        self.mock_engine.predict_batch = Mock()
        self.mock_engine.validate_model = Mock()
        
        # Patch app state to use our mock engine
        app.state.engine = self.mock_engine
        app.state.thread_pool = Mock()

    def test_root_endpoint(self):
        """Test the root endpoint returns expected API information"""
        response = self.client.get("/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert data["model_loaded"]  # Our mock has a model

    @patch("os.path.exists")
    def test_load_model_success(self, mock_exists):
        """Test successfully loading a model"""
        # Setup
        mock_exists.return_value = True
        self.mock_engine.load_model.return_value = True
        
        # Test
        response = self.client.post(
            "/models/load",
            json={"model_path": "/models/test_model.pkl", "model_type": "sklearn"}
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"]
        self.mock_engine.load_model.assert_called_once()

    @patch("os.path.exists")
    def test_load_model_not_found(self, mock_exists):
        """Test loading a non-existent model"""
        # Setup
        mock_exists.return_value = False
        
        # Test
        response = self.client.post(
            "/models/load",
            json={"model_path": "/models/nonexistent.pkl"}
        )
        
        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"]

    def test_predict_success(self):
        """Test making a successful prediction"""
        # Setup
        test_features = [[1.0, 2.0, 3.0]]
        test_predictions = np.array([[0.9, 0.1]])
        
        self.mock_engine.predict.return_value = (True, test_predictions, {})
        
        # Test
        response = self.client.post(
            "/predict",
            json={"features": test_features}
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"]
        assert data["predictions"] == test_predictions.tolist()
        self.mock_engine.predict.assert_called_once()

    def test_predict_no_model(self):
        """Test prediction when no model is loaded"""
        # Setup
        self.mock_engine.model = None
        
        # Test
        response = self.client.post(
            "/predict",
            json={"features": [[1.0, 2.0, 3.0]]}
        )
        
        # Assert
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "No model loaded" in response.json()["detail"]
        
        # Reset for other tests
        self.mock_engine.model = Mock()

    def test_predict_invalid_features(self):
        """Test prediction with invalid feature format"""
        # Test
        response = self.client.post(
            "/predict",
            json={"features": [["invalid", "features"]]}
        )
        
        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_batch_success(self):
        """Test batch prediction"""
        # Setup
        test_batch = [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]
        test_predictions = [
            (True, np.array([[0.9, 0.1]]), {}),
            (True, np.array([[0.2, 0.8]]), {})
        ]
        
        self.mock_engine.predict_batch.return_value = test_predictions
        
        # Test
        response = self.client.post(
            "/predict/batch",
            json={"batch": test_batch}
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["success"]
        assert data["results"][1]["success"]
        self.mock_engine.predict_batch.assert_called_once()

    def test_predict_async(self):
        """Test asynchronous prediction"""
        # Setup
        test_features = [[1.0, 2.0, 3.0]]
        mock_future = MagicMock()
        
        self.mock_engine.dynamic_batcher = Mock()
        self.mock_engine.metrics = Mock()
        self.mock_engine.metrics.get_metrics.return_value = {"avg_inference_time_ms": 10}
        self.mock_engine.dynamic_batcher.get_stats.return_value = {"current_queue_size": 5}
        self.mock_engine.enqueue_prediction.return_value = mock_future
        
        # Test
        response = self.client.post(
            "/predict/async",
            json={"features": test_features, "priority": "high"}
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "pending"
        assert "job_id" in data
        self.mock_engine.enqueue_prediction.assert_called_once()

    def test_update_config(self):
        """Test updating engine configuration"""
        # Setup
        self.mock_engine.config = Mock()
        self.mock_engine.config.enable_batching = True
        self.mock_engine.config.max_batch_size = 64
        self.mock_engine.config.batch_timeout = 0.01
        self.mock_engine.config.enable_request_deduplication = True
        self.mock_engine.config.max_cache_entries = 1000
        self.mock_engine.config.cache_ttl_seconds = 3600
        self.mock_engine.config.enable_quantization = False
        self.mock_engine.config.num_threads = 4
        self.mock_engine.config.enable_throttling = False
        self.mock_engine.config.max_concurrent_requests = 100
        
        self.mock_engine.result_cache = Mock()
        self.mock_engine.feature_cache = Mock()
        
        # Test
        response = self.client.post(
            "/config",
            json={"max_batch_size": 128, "enable_cache": False}
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"]
        assert data["updated_parameters"]["max_batch_size"] == 128
        assert data["updated_parameters"]["enable_cache"] == False

    def test_unload_model(self):
        """Test unloading the model"""
        # Setup
        self.mock_engine.result_cache = Mock()
        self.mock_engine.feature_cache = Mock()
        
        # Test
        response = self.client.delete("/models")
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"]
        
        # Check that model was unloaded
        assert self.mock_engine.model is None

    def test_clear_cache(self):
        """Test clearing the cache"""
        # Setup
        self.mock_engine.result_cache = Mock()
        self.mock_engine.feature_cache = Mock()
        
        # Test
        response = self.client.post("/cache/clear")
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"]
        self.mock_engine.result_cache.clear.assert_called_once()
        self.mock_engine.feature_cache.clear.assert_called_once()

    def test_feature_importance(self):
        """Test feature importance calculation"""
        # Setup
        test_features = [[1.0, 2.0, 3.0]]
        baseline_pred = np.array([[0.7, 0.3]])
        
        # Mock predict to return different values for baseline and perturbed inputs
        self.mock_engine.predict.side_effect = [
            (True, baseline_pred, {}),  # Baseline prediction
            (True, np.array([[0.6, 0.4]]), {}),  # Feature 0 perturbed
            (True, np.array([[0.5, 0.5]]), {}),  # Feature 1 perturbed
            (True, np.array([[0.4, 0.6]]), {})   # Feature 2 perturbed
        ]
        
        # Test
        response = self.client.post(
            "/feature-importance",
            json={"features": test_features}
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"]
        assert "feature_importance" in data
        # Check all feature names are included
        for feature in self.mock_engine.feature_names:
            assert feature in data["feature_importance"]

    def test_validate_model(self):
        """Test model validation"""
        # Setup
        self.mock_engine.validate_model.return_value = {
            "valid": True,
            "model_type": "sklearn",
            "input_shape": [1, 3],
            "output_shape": [1, 2]
        }
        
        # Test
        response = self.client.post("/validate")
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"]
        assert data["model_type"] == "sklearn"
        self.mock_engine.validate_model.assert_called_once()

    def test_cache_stats(self):
        """Test getting cache statistics"""
        # Setup
        self.mock_engine.result_cache = Mock()
        self.mock_engine.feature_cache = Mock()
        
        self.mock_engine.result_cache.get_stats.return_value = {
            "size": 100,
            "hits": 50,
            "misses": 25,
            "hit_rate": 0.67
        }
        
        self.mock_engine.feature_cache.get_stats.return_value = {
            "size": 50,
            "hits": 30,
            "misses": 10,
            "hit_rate": 0.75
        }
        
        # Test
        response = self.client.get("/cache/stats")
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "result_cache" in data
        assert "feature_cache" in data
        assert data["result_cache"]["hits"] == 50
        assert data["feature_cache"]["hits"] == 30

    def test_restart_engine(self):
        """Test restarting the engine"""
        # Setup
        self.mock_engine.config = Mock()
        
        # Test
        with patch("modules.api.inference_engine_api.InferenceEngine") as mock_engine_class:
            response = self.client.post("/restart")
            
            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"]
            mock_engine_class.assert_called_once()
            self.mock_engine.shutdown.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])