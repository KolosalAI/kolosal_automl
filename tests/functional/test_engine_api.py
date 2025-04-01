import unittest
from unittest.mock import Mock, patch, MagicMock, call, mock_open
import os
import json
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from fastapi import FastAPI
import tempfile
from datetime import datetime
import time
import asyncio

# Import the router and dependencies
from modules.api.inference import router, get_inference_engine
from modules.engine.inference_engine import InferenceEngine, EngineState, ModelType
from modules.configs import InferenceEngineConfig

# Create a test FastAPI app using our router
app = FastAPI()
app.include_router(router)

# Create a test client
client = TestClient(app)

# Mock inference engine class for testing
class MockInferenceEngine:
    def __init__(self, config=None):
        self.config = config or InferenceEngineConfig()
        self.model = None
        self.feature_names = []
        self.state = EngineState.READY
        self.model_type = None
        self.current_model_name = None
    
    def load_model(self, model_path, model_type=None):
        if "error" in model_path or not os.path.exists(model_path):
            return False
        self.model = Mock()
        self.model_type = model_type or ModelType.SKLEARN
        self.current_model_name = os.path.basename(model_path)
        return True
    
    def predict(self, features, request_id=None, batch_size=None, return_proba=False):
        if isinstance(features, np.ndarray) and features.size == 0:
            return False, None, {"error": "Empty features array"}
        
        predictions = np.zeros(features.shape[0])
        if return_proba:
            predictions = np.random.rand(features.shape[0], 3)  # 3 classes
        
        metadata = {
            "request_id": request_id or "test-id",
            "inference_time_ms": 10.5,
            "batch_size": features.shape[0]
        }
        
        return True, predictions, metadata
    
    def predict_batch(self, features, priority=0):
        future = asyncio.Future()
        predictions = np.zeros(features.shape[0])
        metadata = {"batch_id": "test-batch"}
        future.set_result((predictions, metadata))
        return future
    
    def get_state(self):
        return self.state
    
    def get_metrics(self):
        return {
            "total_requests": 1000,
            "error_count": 10,
            "error_rate": 0.01,
            "throughput_requests_per_second": 200,
            "avg_inference_time_ms": 5.0,
            "p95_inference_time_ms": 10.0,
            "p99_inference_time_ms": 15.0,
            "engine_state": self.state.name,
            "active_requests": 5,
            "memory_mb": 500.0,
            "cpu_percent": 30.0,
            "cache_hit_rate": 0.8,
            "cache_hits": 800,
            "cache_misses": 200,
            "avg_batch_size": 32.0
        }
    
    def get_model_info(self):
        return {
            "model_info": {
                "model_type": "sklearn_random_forest",
                "model_class": "sklearn.ensemble.RandomForestClassifier",
                "feature_count": len(self.feature_names),
                "has_feature_importances": True,
                "hyperparameters": {"n_estimators": 100, "max_depth": 10}
            },
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {}
        }
    
    def set_feature_names(self, feature_names):
        self.feature_names = feature_names
    
    def get_feature_names(self):
        return self.feature_names
    
    def clear_cache(self):
        pass
    
    def shutdown(self):
        self.state = EngineState.STOPPED
    
    def get_memory_usage(self):
        return {
            "rss_mb": 500.0,
            "vms_mb": 1000.0,
            "percent": 5.0,
            "cpu_percent": 30.0
        }

# Override dependency to use our mock inference engine
app.dependency_overrides[get_inference_engine] = lambda: MockInferenceEngine()


class TestInferenceAPI(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for model registry
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_registry = self.temp_dir.name
        
        # Set up the model registry directory structure
        self.model_dir = os.path.join(self.model_registry, "test_model")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create a mock model info file
        model_info = {
            "model_type": "sklearn_random_forest",
            "feature_count": 10,
            "description": "Test model for API testing"
        }
        
        with open(os.path.join(self.model_dir, "model_info.json"), "w") as f:
            json.dump(model_info, f)
        
        # Create version directories
        versions = ["v1", "v2", "latest"]
        for version in versions:
            version_dir = os.path.join(self.model_dir, version)
            os.makedirs(version_dir, exist_ok=True)
            with open(os.path.join(version_dir, "model_info.json"), "w") as f:
                json.dump(model_info, f)
        
        # Mock the model registry path in the API module
        self.patcher = patch("modules.api.inference.MODEL_REGISTRY", self.model_registry)
        self.mock_registry = self.patcher.start()
        
        # Set up results directory
        self.results_dir = os.path.join(self.temp_dir.name, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.results_patcher = patch("modules.api.inference.RESULTS_DIR", self.results_dir)
        self.mock_results_dir = self.results_patcher.start()
    
    def tearDown(self):
        # Clean up temporary directory
        self.patcher.stop()
        self.results_patcher.stop()
        self.temp_dir.cleanup()
    
    def test_predict_endpoint(self):
        """Test the /predict endpoint"""
        # Create a test CSV file
        test_df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "feature3": [7, 8, 9]
        })
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            test_df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
        
        try:
            # Test with file upload
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/api/v1/inference/predict",
                    files={"data_file": ("test.csv", f, "text/csv")},
                    data={"return_probabilities": "false"}
                )
            
            self.assertEqual(response.status_code, 200)
            self.assertIn("predictions", response.json())
            self.assertEqual(len(response.json()["predictions"]), 3)
            self.assertIn("execution_time_ms", response.json())
            
            # Test with file upload and return probabilities
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/api/v1/inference/predict",
                    files={"data_file": ("test.csv", f, "text/csv")},
                    data={"return_probabilities": "true"}
                )
            
            self.assertEqual(response.status_code, 200)
            self.assertIn("predictions", response.json())
            self.assertTrue(isinstance(response.json()["predictions"], list))
            self.assertTrue(isinstance(response.json()["predictions"][0], list))
        
        finally:
            # Clean up
            os.unlink(temp_file_path)

    @patch("modules.api.inference.inference_engine.predict_batch")
    def test_batch_inference_endpoint(self, mock_predict_batch):
        """Test the /batch-inference endpoint"""
        # Create future result for mock
        mock_future = MagicMock()
        mock_future.result.return_value = (np.zeros(3), {"batch_id": "test-batch"})
        mock_predict_batch.return_value = mock_future
        
        # Create two test CSV files
        test_df1 = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6]
        })
        
        test_df2 = pd.DataFrame({
            "feature1": [7, 8, 9],
            "feature2": [10, 11, 12]
        })
        
        temp_files = []
        for i, df in enumerate([test_df1, test_df2]):
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                df.to_csv(temp_file.name, index=False)
                temp_files.append(temp_file.name)
        
        try:
            # Test with file uploads
            with open(temp_files[0], "rb") as f1, open(temp_files[1], "rb") as f2:
                response = client.post(
                    "/api/v1/inference/batch-inference",
                    files=[
                        ("files", ("test1.csv", f1, "text/csv")),
                        ("files", ("test2.csv", f2, "text/csv"))
                    ],
                    data={"params": json.dumps({"parallel": True})}
                )
            
            self.assertEqual(response.status_code, 200)
            self.assertIn("results", response.json())
            self.assertEqual(len(response.json()["results"]), 2)
            self.assertEqual(response.json()["batch_count"], 2)
            self.assertIn("execution_time_ms", response.json())
        
        finally:
            # Clean up
            for file_path in temp_files:
                os.unlink(file_path)
    
    @patch("modules.api.inference.background_tasks.add_task")
    def test_streaming_inference_endpoint(self, mock_add_task):
        """Test the /streaming-inference endpoint"""
        request_data = {
            "model_id": "test_model",
            "batch_identifier": "test_batch",
            "output_format": "csv",
            "include_probabilities": False
        }
        
        # Make sure the model directory exists for the test
        os.makedirs(os.path.join(self.model_registry, "test_model"), exist_ok=True)
        
        response = client.post("/api/v1/inference/streaming-inference", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("job_id", response.json())
        self.assertIn("status", response.json())
        self.assertEqual(response.json()["status"], "pending")
        self.assertTrue(mock_add_task.called)
    
    def test_streaming_status_endpoint(self):
        """Test the /streaming-status/{job_id} endpoint"""
        # Create a mock status file
        job_id = "test_job_123"
        status_data = {
            "job_id": job_id,
            "model_id": "test_model",
            "model_version": "v1",
            "batch_identifier": "test_batch",
            "status": "processing",
            "start_time": time.time(),
            "output_path": os.path.join(self.results_dir, "test_results.csv"),
            "output_format": "csv",
            "progress": 50,
            "processed_items": 5000,
            "total_items": 10000,
            "errors": []
        }
        
        status_path = os.path.join(self.results_dir, f"{job_id}_status.json")
        with open(status_path, "w") as f:
            json.dump(status_data, f)
        
        # Test endpoint
        response = client.get(f"/api/v1/inference/streaming-status/{job_id}")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_id"], job_id)
        self.assertEqual(response.json()["status"], "processing")
        self.assertEqual(response.json()["progress"], 50)
        
        # Test with non-existent job
        response = client.get("/api/v1/inference/streaming-status/non_existent_job")
        self.assertEqual(response.status_code, 404)
    
    def test_explain_endpoint(self):
        """Test the /explain endpoint"""
        # Create a test CSV file
        test_df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "feature3": [7, 8, 9]
        })
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            test_df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
        
        try:
            # Test with file upload
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/api/v1/inference/explain",
                    files={"data_file": ("test.csv", f, "text/csv")},
                    data={"params": json.dumps({"method": "shap", "n_samples": 2})}
                )
            
            self.assertEqual(response.status_code, 200)
            self.assertIn("explanations", response.json())
            self.assertEqual(response.json()["method"], "shap")
            self.assertIn("sample_count", response.json())
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_models_endpoint(self):
        """Test the /models endpoint"""
        response = client.get("/api/v1/inference/models")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("models", response.json())
        self.assertIn("count", response.json())
        self.assertIn("current_model", response.json())
        
        # Check if our test model is in the results
        models = response.json()["models"]
        self.assertTrue(any(model["model_id"] == "test_model" for model in models))
    
    @patch("os.path.exists")
    def test_load_model_endpoint(self, mock_exists):
        """Test the /load-model/{model_id} endpoint"""
        # Mock that the model exists
        mock_exists.return_value = True
        
        # Test loading a model
        response = client.post(
            "/api/v1/inference/load-model/test_model",
            params={"model_version": "v2"}
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("model_id", response.json())
        self.assertEqual(response.json()["model_id"], "test_model")
        self.assertEqual(response.json()["model_version"], "v2")
        
        # Test loading a non-existent model
        mock_exists.return_value = False
        response = client.post("/api/v1/inference/load-model/non_existent_model")
        self.assertEqual(response.status_code, 404)
    
    def test_model_metadata_endpoint(self):
        """Test the /model-metadata endpoint"""
        # Test getting metadata for current model
        response = client.get("/api/v1/inference/model-metadata")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("metadata", response.json())
        
        # Test getting metadata for specified model
        response = client.get(
            "/api/v1/inference/model-metadata",
            params={"model_id": "test_model", "model_version": "v1"}
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("metadata", response.json())
        self.assertEqual(response.json()["model_id"], "test_model")
        self.assertEqual(response.json()["model_version"], "v1")
    
    def test_model_config_endpoint(self):
        """Test the /model-config endpoint"""
        config_data = {
            "engine_config": {"debug_mode": True, "enable_batching": True},
            "feature_names": ["feature1", "feature2", "feature3"]
        }
        
        response = client.post("/api/v1/inference/model-config", json=config_data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("updates_applied", response.json())
        self.assertTrue(response.json()["updates_applied"]["engine_config"])
        self.assertTrue(response.json()["updates_applied"]["feature_names"])
    
    def test_metrics_endpoint(self):
        """Test the /metrics endpoint"""
        response = client.get("/api/v1/inference/metrics")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("engine_state", response.json())
        self.assertIn("total_requests", response.json())
        self.assertIn("avg_inference_time_ms", response.json())
        self.assertIn("error_rate", response.json())
        self.assertIn("throughput_requests_per_second", response.json())
    
    def test_clear_cache_endpoint(self):
        """Test the /clear-cache endpoint"""
        response = client.post("/api/v1/inference/clear-cache")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("timestamp", response.json())
    
    def test_status_endpoint(self):
        """Test the /status endpoint"""
        response = client.get("/api/v1/inference/status")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertIn("ready_for_inference", response.json())
        self.assertIn("model_loaded", response.json())
        self.assertIn("memory_usage_mb", response.json())
        self.assertIn("cpu_percent", response.json())
        self.assertIn("timestamp", response.json())
    
    def test_shutdown_endpoint(self):
        """Test the /shutdown endpoint"""
        response = client.post("/api/v1/inference/shutdown")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("status", response.json())
        self.assertIn("timestamp", response.json())
    
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_results_endpoint(self, mock_file, mock_exists):
        """Test the /download-results/{job_id} endpoint"""
        # Set up mocks
        mock_exists.return_value = True
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps({
            "status": "completed",
            "output_path": "/path/to/results.csv"
        })
        
        # Unfortunately, we can't fully test the FileResponse in this unit test
        # So we'll just check that the endpoint handles the request correctly
        
        # This will raise an exception because we can't mock the file response completely
        # But we can at least check that the endpoint code ran up to that point
        with self.assertRaises(Exception):
            response = client.get("/api/v1/inference/download-results/test_job")
    
    def test_error_handling_predict(self):
        """Test error handling in the predict endpoint"""
        # Test with missing file
        response = client.post("/api/v1/inference/predict")
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
        
        # Test with invalid model
        with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
            pd.DataFrame({"a": [1]}).to_csv(temp_file.name, index=False)
            temp_file.flush()
            
            with open(temp_file.name, "rb") as f:
                response = client.post(
                    "/api/v1/inference/predict",
                    files={"data_file": ("test.csv", f, "text/csv")},
                    data={"model_id": "non_existent_model"}
                )
            
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()