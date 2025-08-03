import unittest
import json
import os
import pandas as pd
import numpy as np
from unittest import mock
from fastapi.testclient import TestClient
from fastapi import UploadFile
from io import BytesIO

# Import the module being tested
from modules.api.train_engine_api import (
    app, ml_engine, get_ml_engine, 
    map_config_to_engine_config, map_config_to_preprocessor_config,
    load_data_from_file, train_model_task, training_tasks,
    EngineConfigRequest, PreprocessingConfigRequest, TrainModelRequest
)

# Import necessary classes for mocking
from modules.configs import (
    TaskType, OptimizationStrategy, MLTrainingEngineConfig,
    PreprocessorConfig, NormalizationType, ModelSelectionCriteria
)
from modules.engine.train_engine import MLTrainingEngine

class TestMLTrainingEngineAPI(unittest.TestCase):
    """Tests for the ML Training Engine API."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create test client
        self.client = TestClient(app)
        
        # Mock global ml_engine
        self.ml_engine_mock = mock.MagicMock(spec=MLTrainingEngine)
        
        # Create test directories if they don't exist
        os.makedirs("static/uploads", exist_ok=True)
        os.makedirs("static/models", exist_ok=True)
        os.makedirs("static/reports", exist_ok=True)
        os.makedirs("static/charts", exist_ok=True)
        
        # Create a small test CSV file
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        self.test_file_path = "static/uploads/test_data.csv"
        self.test_df.to_csv(self.test_file_path, index=False)
        
    def tearDown(self):
        """Clean up after each test."""
        # Reset global ml_engine
        global ml_engine
        ml_engine = None
        
        # Clear training tasks
        training_tasks.clear()
        
        # Remove test file if it exists
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        # Ensure ml_engine is None for this test
        global ml_engine
        ml_engine = None
        # Also reset in the API module 
        from modules.api import train_engine_api
        train_engine_api.ml_engine = None
        
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "ML Training Engine API")
        self.assertEqual(data["status"], "engine_not_initialized")
    
    @mock.patch('modules.api.train_engine_api.MLTrainingEngine')
    def test_initialize_engine(self, mock_ml_engine_class):
        """Test engine initialization endpoint."""
        # Setup mock
        mock_ml_engine_instance = mock.MagicMock()
        mock_ml_engine_class.return_value = mock_ml_engine_instance
        
        # Prepare request data
        request_data = {
            "engine_config": {
                "task_type": "classification",
                "model_path": "models",
                "random_state": 42,
                "test_size": 0.2,
                "cv_folds": 5,
                "n_jobs": -1,
                "verbose": 1,
                "optimization_strategy": "random_search",
                "optimization_iterations": 20,
                "model_selection_criteria": "f1",
                "feature_selection": True,
                "feature_selection_method": "mutual_info",
                "feature_selection_k": 10,
                "early_stopping": True,
                "early_stopping_rounds": 10,
                "auto_save": True,
                "experiment_tracking": True
            },
            "preprocessing_config": {
                "handle_nan": True,
                "nan_strategy": "mean",
                "detect_outliers": True,
                "outlier_handling": "clip",
                "categorical_encoding": "one_hot",
                "normalization": "standard"
            }
        }
        
        # Make request
        response = self.client.post("/api/initialize", json=request_data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["config"]["task_type"], "classification")
        
        # Verify MLTrainingEngine was created with correct parameters
        mock_ml_engine_class.assert_called_once()
        
        # Reset global ml_engine
        global ml_engine
        ml_engine = None
    
    def test_map_config_to_engine_config(self):
        """Test mapping from API config to engine config."""
        # Create a sample EngineConfigRequest
        config_request = EngineConfigRequest(
            task_type="classification",
            model_path="test_models",
            random_state=123,
            test_size=0.25,
            cv_folds=3,
            optimization_strategy="grid_search",
            model_selection_criteria="accuracy"
        )
        
        # Map to engine config
        engine_config = map_config_to_engine_config(config_request)
        
        # Check mapping
        self.assertIsInstance(engine_config, MLTrainingEngineConfig)
        self.assertEqual(engine_config.task_type, TaskType.CLASSIFICATION)
        self.assertEqual(engine_config.model_path, "test_models")
        self.assertEqual(engine_config.random_state, 123)
        self.assertEqual(engine_config.test_size, 0.25)
        self.assertEqual(engine_config.cv_folds, 3)
        self.assertEqual(engine_config.optimization_strategy, OptimizationStrategy.GRID_SEARCH)
        self.assertEqual(engine_config.model_selection_criteria, ModelSelectionCriteria.ACCURACY)
    
    def test_map_config_to_preprocessor_config(self):
        """Test mapping from API config to preprocessor config."""
        # Create a sample PreprocessingConfigRequest
        config_request = PreprocessingConfigRequest(
            handle_nan=True,
            nan_strategy="median",
            detect_outliers=True,
            outlier_handling="clip",
            categorical_encoding="label",
            normalization="minmax"
        )
        
        # Map to preprocessor config
        preprocessor_config = map_config_to_preprocessor_config(config_request)
        
        # Check mapping
        self.assertIsInstance(preprocessor_config, PreprocessorConfig)
        self.assertEqual(preprocessor_config.handle_nan, True)
        self.assertEqual(preprocessor_config.nan_strategy, "median")
        self.assertEqual(preprocessor_config.detect_outliers, True)
        self.assertEqual(preprocessor_config.outlier_handling, "clip")
        self.assertEqual(preprocessor_config.categorical_encoding, "label")
        self.assertEqual(preprocessor_config.normalization, NormalizationType.MINMAX)
    
    def test_get_ml_engine_not_initialized(self):
        """Test get_ml_engine when engine is not initialized."""
        with self.assertRaises(Exception):
            get_ml_engine()
    
    def test_load_data_from_file(self):
        """Test loading data from a file."""
        # Load data from test file
        df = load_data_from_file(self.test_file_path)
        
        # Check DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertListEqual(list(df.columns), ['feature1', 'feature2', 'target'])
        
        # Test with column selection
        df = load_data_from_file(self.test_file_path, columns=['feature1', 'target'])
        self.assertListEqual(list(df.columns), ['feature1', 'target'])
        
        # Test with missing file
        with self.assertRaises(ValueError):
            load_data_from_file("nonexistent_file.csv")
    
    @mock.patch('modules.api.train_engine_api.ml_engine')
    def test_upload_file(self, mock_ml_engine):
        """Test file upload endpoint."""
        # Create a CSV file in memory
        content = b"feature1,feature2,target\n1,2,0\n3,4,1\n5,6,0"
        file = BytesIO(content)
        
        # Make request
        response = self.client.post(
            "/api/upload",
            files={"file": ("test_upload.csv", file, "text/csv")}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue("filename" in data)
        self.assertEqual(data["original_filename"], "test_upload.csv")
        self.assertEqual(data["row_count"], 3)
        self.assertEqual(data["column_count"], 3)
        
        # Clean up uploaded file
        uploaded_path = data["file_path"]
        if os.path.exists(uploaded_path):
            os.remove(uploaded_path)
    
    @mock.patch('modules.api.train_engine_api.get_ml_engine')
    @mock.patch('modules.api.train_engine_api.load_data_from_file')
    def test_train_model(self, mock_load_data, mock_get_ml_engine):
        """Test train model endpoint."""
        # Setup mocks
        mock_engine = mock.MagicMock()
        mock_get_ml_engine.return_value = mock_engine
        
        # Mock loading data
        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        mock_load_data.return_value = mock_df
        
        # Prepare request data
        request_data = {
            "model_type": "random_forest",
            "model_name": "test_model",
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20]
            },
            "train_data_file": "test_data.csv",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2"],
            "test_size": 0.2
        }
        
        # Make request
        response = self.client.post("/api/train", json=request_data)
        
        # Check response
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertEqual(data["status"], "training_started")
        self.assertTrue("task_id" in data)
        
        # Check that task was created
        task_id = data["task_id"]
        self.assertTrue(task_id in training_tasks)
    
    @mock.patch('modules.api.train_engine_api.get_ml_engine')
    def test_get_training_status(self, mock_get_ml_engine):
        """Test getting training status endpoint."""
        # Setup mock
        mock_engine = mock.MagicMock()
        mock_get_ml_engine.return_value = mock_engine
        
        # Create a task
        task_id = "test_task_123"
        training_tasks[task_id] = {
            "status": "running",
            "progress": 0.5,
            "started_at": "2025-01-01T12:00:00",
            "model_name": "test_model",
            "error": None
        }
        
        # Make request
        response = self.client.get(f"/api/train/status/{task_id}")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["progress"], 0.5)
        self.assertEqual(data["model_name"], "test_model")
        
        # Test with nonexistent task
        response = self.client.get("/api/train/status/nonexistent_task")
        self.assertEqual(response.status_code, 404)
    
    @mock.patch('modules.api.train_engine_api.get_ml_engine')
    def test_list_models(self, mock_get_ml_engine):
        """Test listing models endpoint."""
        # Setup mock
        mock_engine = mock.MagicMock()
        mock_engine.models = {
            "model1": {
                "model": mock.MagicMock(),
                "metrics": {"accuracy": 0.9},
                "training_time": 10.5,
                "feature_names": ["f1", "f2"]
            },
            "model2": {
                "model": mock.MagicMock(),
                "metrics": {"accuracy": 0.85},
                "training_time": 8.2,
                "feature_names": ["f1", "f2", "f3"]
            }
        }
        mock_engine.best_model_name = "model1"
        mock_get_ml_engine.return_value = mock_engine
        
        # Make request
        response = self.client.get("/api/models")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["models"]), 2)
        self.assertEqual(data["best_model"], "model1")
        self.assertEqual(data["count"], 2)
        
        # Check model details
        models = {model["name"]: model for model in data["models"]}
        self.assertTrue("model1" in models)
        self.assertTrue("model2" in models)
        self.assertEqual(models["model1"]["feature_count"], 2)
        self.assertEqual(models["model2"]["feature_count"], 3)
        self.assertEqual(models["model1"]["is_best"], True)
        self.assertEqual(models["model2"]["is_best"], False)
    
    @mock.patch('modules.api.train_engine_api.get_ml_engine')
    @mock.patch('modules.api.train_engine_api.load_data_from_file')
    def test_model_predict(self, mock_load_data, mock_get_ml_engine):
        """Test model prediction endpoint."""
        # Setup mocks
        mock_engine = mock.MagicMock()
        mock_engine.models = {"test_model": mock.MagicMock()}
        mock_engine.best_model_name = "test_model"
        mock_engine.predict.return_value = (True, [0, 1, 0])
        mock_get_ml_engine.return_value = mock_engine
        
        # Mock loading data
        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [10, 20, 30]
        })
        mock_load_data.return_value = mock_df
        
        # Test with data file - use POST instead of GET
        response = self.client.post(
            "/api/models/test_model/predict",
            json={
                "data_file": "test_data.csv",
                "return_probabilities": False
            }
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "test_model")
        self.assertEqual(data["predictions"], [0, 1, 0])
        self.assertEqual(data["prediction_count"], 3)
        
        # Test with "best" model - use POST
        response = self.client.post(
            "/api/models/best/predict",
            json={
                "data_file": "test_data.csv",
                "return_probabilities": False
            }
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "test_model")  # Should use best model name
        
        # Test with direct data - use POST
        response = self.client.post(
            "/api/models/test_model/predict",
            json={
                "data": [
                    {"feature1": 1, "feature2": 10},
                    {"feature1": 2, "feature2": 20}
                ],
                "return_probabilities": False
            }
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        # Test with nonexistent model - use POST
        response = self.client.post(
            "/api/models/nonexistent_model/predict",
            json={"data_file": "test_data.csv"}
        )
        self.assertEqual(response.status_code, 404)
    
    @mock.patch('modules.api.train_engine_api.get_ml_engine')
    def test_save_model(self, mock_get_ml_engine):
        """Test saving model endpoint."""
        # Setup mock
        mock_engine = mock.MagicMock()
        mock_engine.models = {"test_model": mock.MagicMock()}
        mock_engine.best_model_name = "test_model"
        mock_engine.save_model.return_value = "models/test_model.pkl"
        mock_get_ml_engine.return_value = mock_engine
        
        # Make request
        response = self.client.post("/api/models/save/test_model?include_preprocessor=true")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "test_model")
        self.assertEqual(data["save_path"], "models/test_model.pkl")
        self.assertEqual(data["include_preprocessor"], True)
        
        # Test with "best" model
        response = self.client.post("/api/models/save/best?include_preprocessor=false")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "test_model")  # Should use best model name
        self.assertEqual(data["include_preprocessor"], False)
        
        # Test with nonexistent model
        response = self.client.post("/api/models/save/nonexistent_model")
        self.assertEqual(response.status_code, 404)
    
    @mock.patch('modules.api.train_engine_api.get_ml_engine')
    @mock.patch('modules.api.train_engine_api.load_data_from_file')
    def test_evaluate_model(self, mock_load_data, mock_get_ml_engine):
        """Test model evaluation endpoint."""
        # Setup mocks
        mock_engine = mock.MagicMock()
        mock_engine.models = {"test_model": mock.MagicMock()}
        mock_engine.evaluate_model.return_value = {
            "accuracy": 0.9,
            "precision": 0.85,
            "recall": 0.88,
            "f1": 0.86
        }
        mock_get_ml_engine.return_value = mock_engine
        
        # Mock loading data
        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        mock_load_data.return_value = mock_df
        
        # Prepare request data
        request_data = {
            "test_data_file": "test_data.csv",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2"],
            "detailed": True
        }
        
        # Make request
        response = self.client.post(f"/api/models/test_model/evaluate", json=request_data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "test_model")
        self.assertEqual(data["metrics"]["accuracy"], 0.9)
        self.assertEqual(data["metrics"]["f1"], 0.86)
        self.assertEqual(data["detailed"], True)
        
        # Test with nonexistent model
        response = self.client.post("/api/models/nonexistent_model/evaluate", json=request_data)
        self.assertEqual(response.status_code, 404)
    
    @mock.patch('modules.api.train_engine_api.get_ml_engine')
    def test_engine_status(self, mock_get_ml_engine):
        """Test engine status endpoint."""
        # Setup mock
        mock_engine = mock.MagicMock()
        mock_engine.models = {"model1": mock.MagicMock(), "model2": mock.MagicMock()}
        mock_engine.best_model_name = "model1"
        mock_engine.training_complete = True
        mock_engine.config.task_type = TaskType.CLASSIFICATION
        mock_engine.config.model_path = "models"
        mock_engine.config.test_size = 0.2
        mock_engine.preprocessor = mock.MagicMock()
        mock_get_ml_engine.return_value = mock_engine
        
        # Make request
        response = self.client.post("/api/engine/status")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["initialized"], True)
        self.assertEqual(data["task_type"], "classification")
        self.assertEqual(data["model_count"], 2)
        self.assertEqual(data["best_model"], "model1")
        self.assertEqual(data["training_complete"], True)
        self.assertTrue("preprocessor" in data)
        self.assertEqual(data["preprocessor"]["configured"], True)
    
    @mock.patch('modules.api.train_engine_api.get_ml_engine')
    def test_generate_report(self, mock_get_ml_engine):
        """Test report generation endpoint."""
        # Setup mock
        mock_engine = mock.MagicMock()
        mock_engine.generate_report.return_value = "static/reports/model_report_123.md"
        mock_get_ml_engine.return_value = mock_engine
        
        # Make request
        response = self.client.post("/api/reports/generate")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue("report_path" in data)
        self.assertTrue("download_url" in data)
        self.assertTrue("timestamp" in data)
    
    def test_shutdown_engine(self):
        """Test engine shutdown endpoint."""
        # Set global ml_engine in the API module
        from modules.api import train_engine_api
        mock_engine = mock.MagicMock()
        train_engine_api.ml_engine = mock_engine
        
        # Make request
        response = self.client.post("/api/engine/shutdown")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["message"], "Engine shut down successfully")
        
        # Check that ml_engine was reset
        self.assertIsNone(train_engine_api.ml_engine)
        
        # Test with already shut down engine
        response = self.client.post("/api/engine/shutdown")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "Engine already shut down")

if __name__ == "__main__":
    unittest.main()