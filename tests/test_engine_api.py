import unittest
import os
import sys
import tempfile
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from fastapi import FastAPI
import io
import joblib
from datetime import datetime
import time
import json

# Import the router to test
from modules.api.engine import router, get_ml_engine, MLTrainingEngine, MLTrainingEngineConfig, TaskType, OptimizationStrategy

# Create a FastAPI app and register the router for testing
app = FastAPI()
app.include_router(router)

# Create a test client
client = TestClient(app)

class TestMLEngineRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and ML engine once for all tests."""
        # Create ML engine instance for testing
        config = MLTrainingEngineConfig(
            task_type=TaskType.CLASSIFICATION,
            optimization_strategy=OptimizationStrategy.ACCURACY,
            random_state=42
        )
        cls.ml_engine = MLTrainingEngine(config)
        
        # Create test data for classification
        cls.X_train, cls.y_train = cls._create_test_data()
        cls.test_data_path = cls._create_test_csv(cls.X_train, cls.y_train)
        
        # Training a simple model to use in tests
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(cls.X_train, cls.y_train)
        
        # Add model to engine
        cls.model_name = "test_model"
        cls.ml_engine.models[cls.model_name] = {
            "model": model,
            "params": {"n_estimators": 10, "random_state": 42},
            "metrics": {"accuracy": 0.95},
            "timestamp": time.time(),
            "training_time": 1.0,
            "feature_names": [f"feature_{i}" for i in range(cls.X_train.shape[1])],
            "dataset_shape": {"X_train": cls.X_train.shape}
        }
        cls.ml_engine.best_model = cls.model_name
        cls.ml_engine.best_score = 0.95
        
        # Monkey patch the get_ml_engine dependency
        app.dependency_overrides[get_ml_engine] = lambda: cls.ml_engine

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove test files
        if os.path.exists(cls.test_data_path):
            os.remove(cls.test_data_path)
        
        # Remove dependency override
        app.dependency_overrides = {}

    @staticmethod
    def _create_test_data(n_samples=100, n_features=5):
        """Create synthetic data for testing."""
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, size=n_samples)
        return X, y

    @classmethod
    def _create_test_csv(cls, X, y):
        """Create a CSV file with test data."""
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        
        # Save to temp file
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        df.to_csv(path, index=False)
        return path
    
    def _create_csv_file(self, df):
        """Create a CSV file from a dataframe."""
        csv_file = io.BytesIO()
        df.to_csv(csv_file, index=False)
        csv_file.seek(0)
        return csv_file

    def test_list_models(self):
        """Test GET /ml-engine/models endpoint."""
        response = client.get("/ml-engine/models")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("models", data)
        self.assertIn("count", data)
        self.assertIn("best_model", data)
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["best_model"], self.model_name)
        self.assertIn(self.model_name, data["models"])

    def test_get_model_details(self):
        """Test GET /ml-engine/models/{model_name} endpoint."""
        response = client.get(f"/ml-engine/models/{self.model_name}")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["name"], self.model_name)
        self.assertEqual(data["type"], "RandomForestClassifier")
        self.assertIn("metrics", data)
        self.assertIn("params", data)
        self.assertTrue(data["is_best"])

    def test_get_nonexistent_model(self):
        """Test GET for a model that doesn't exist."""
        response = client.get("/ml-engine/models/nonexistent_model")
        self.assertEqual(response.status_code, 404)

    def test_train_model(self):
        """Test POST /ml-engine/train-model endpoint."""
        # Create test data file
        df = pd.DataFrame(self.X_train)
        df["target"] = self.y_train
        csv_file = self._create_csv_file(df)
        
        # Set up request parameters
        params = {
            "model_type": "random_forest",
            "model_name": "new_test_model",
            "param_grid": {"n_estimators": 10, "max_depth": 5},
            "test_size": 0.2
        }
        
        # Make request
        response = client.post(
            "/ml-engine/train-model",
            data={"params": json.dumps(params)},
            files={"data_file": ("test.csv", csv_file, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["status"], "training_started")
        self.assertEqual(data["model_name"], "new_test_model")

    def test_predict(self):
        """Test POST /ml-engine/predict endpoint."""
        # Create test data file (without target)
        df = pd.DataFrame(self.X_train)
        csv_file = self._create_csv_file(df)
        
        # Make request
        response = client.post(
            "/ml-engine/predict",
            data={"params.model_name": self.model_name, "params.return_proba": "false"},
            files={"batch_data": ("test.csv", csv_file, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("predictions", data)
        self.assertEqual(data["model_used"], self.model_name)
        self.assertEqual(data["row_count"], len(self.X_train))

    def test_evaluate_model(self):
        """Test POST /ml-engine/evaluate/{model_name} endpoint."""
        # Create test data file
        df = pd.DataFrame(self.X_train)
        df["target"] = self.y_train
        csv_file = self._create_csv_file(df)
        
        # Make request
        response = client.post(
            f"/ml-engine/evaluate/{self.model_name}",
            data={"detailed": "false"},
            files={"test_data": ("test.csv", csv_file, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["model_name"], self.model_name)
        self.assertIn("metrics", data)
        self.assertEqual(data["test_samples"], len(self.X_train))

    def test_feature_importance(self):
        """Test POST /ml-engine/feature-importance endpoint."""
        params = {
            "model_name": self.model_name,
            "top_n": 5,
            "include_plot": False
        }
        
        response = client.post("/ml-engine/feature-importance", json=params)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("importances", data)
        self.assertEqual(len(data["importances"]), min(5, self.X_train.shape[1]))

    def test_error_analysis(self):
        """Test POST /ml-engine/error-analysis endpoint."""
        # Create test data file
        df = pd.DataFrame(self.X_train)
        df["target"] = self.y_train
        csv_file = self._create_csv_file(df)
        
        # Make request
        response = client.post(
            "/ml-engine/error-analysis",
            data={"params.model_name": self.model_name, "params.n_samples": "10", "params.include_plot": "false"},
            files={"test_data": ("test.csv", csv_file, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("error_samples", data)
        self.assertIn("confusion_matrix", data)

    def test_data_drift(self):
        """Test POST /ml-engine/data-drift endpoint."""
        # Create test data file with slight drift
        X_drift = self.X_train * 1.1  # Add some drift
        df = pd.DataFrame(X_drift)
        df["target"] = self.y_train
        csv_file = self._create_csv_file(df)
        
        # Make request with just new data
        response = client.post(
            "/ml-engine/data-drift",
            data={"params.drift_threshold": "0.5"},
            files={"new_data": ("test.csv", csv_file, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("drift_detected", data)
        self.assertIn("dataset_drift", data)
        self.assertIn("drifted_features", data)

    def test_compare_models(self):
        """Test POST /ml-engine/compare-models endpoint."""
        params = {
            "model_names": [self.model_name],
            "metrics": ["accuracy"],
            "include_plot": False
        }
        
        response = client.post("/ml-engine/compare-models", json=params)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("comparison_table", data)
        self.assertIn(self.model_name, data["comparison_table"])

    def test_export_model(self):
        """Test POST /ml-engine/export-model/{model_name} endpoint."""
        params = {
            "format": "sklearn",
            "include_pipeline": True
        }
        
        response = client.post(f"/ml-engine/export-model/{self.model_name}", json=params)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["model_name"], self.model_name)
        self.assertEqual(data["format"], "sklearn")
        self.assertIn("file_size_bytes", data)
        self.assertIn("download_url", data)

    def test_delete_model(self):
        """Test DELETE /ml-engine/models/{model_name} endpoint."""
        # First create a model to delete
        model_to_delete = "model_to_delete"
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        self.ml_engine.models[model_to_delete] = {
            "model": model,
            "params": {"n_estimators": 5, "random_state": 42},
            "metrics": {"accuracy": 0.9},
            "timestamp": time.time(),
            "training_time": 0.5
        }
        
        # Delete the model
        response = client.delete(f"/ml-engine/models/{model_to_delete}")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("message", data)
        self.assertNotIn(model_to_delete, self.ml_engine.models)

    def test_generate_report(self):
        """Test POST /ml-engine/generate-report endpoint."""
        response = client.post("/ml-engine/generate-report", params={"include_plots": False})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("report", data)
        self.assertIn("model_count", data)
        self.assertIn("best_model", data)

    def test_save_and_load_model(self):
        """Test save and load model endpoints."""
        # Test save model
        response_save = client.post(
            f"/ml-engine/save-model/{self.model_name}",
            params={"version_tag": "v1", "include_preprocessor": True}
        )
        
        self.assertEqual(response_save.status_code, 200)
        save_data = response_save.json()
        self.assertIn("filepath", save_data)
        
        # Load the saved model
        with open(save_data["filepath"], "rb") as f:
            model_bytes = f.read()
        
        # Test load model
        response_load = client.post(
            "/ml-engine/load-model",
            files={"model_file": ("model.pkl", io.BytesIO(model_bytes), "application/octet-stream")},
            params={"validate_metrics": True}
        )
        
        self.assertEqual(response_load.status_code, 200)
        load_data = response_load.json()
        
        self.assertIn("model_name", load_data)
        self.assertIn("model_type", load_data)
        self.assertIn("metrics", load_data)
        
        # Clean up the saved model file
        if os.path.exists(save_data["filepath"]):
            os.remove(save_data["filepath"])

    def test_ensemble_models(self):
        """Test POST /ml-engine/ensemble-models endpoint."""
        # First create another model for the ensemble
        second_model_name = "second_test_model"
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        
        self.ml_engine.models[second_model_name] = {
            "model": model,
            "params": {"random_state": 42},
            "metrics": {"accuracy": 0.88},
            "timestamp": time.time(),
            "training_time": 0.2
        }
        
        # Create ensemble
        params = {
            "model_names": [self.model_name, second_model_name],
            "ensemble_name": "test_ensemble",
            "voting_type": "soft"
        }
        
        response = client.post("/ml-engine/ensemble-models", json=params)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["ensemble_name"], "test_ensemble")
        self.assertEqual(len(data["base_models"]), 2)
        self.assertIn("test_ensemble", self.ml_engine.models)

    def test_calibrate_model(self):
        """Test POST /ml-engine/models/{model_name}/calibrate endpoint."""
        # Create test data file
        df = pd.DataFrame(self.X_train)
        df["target"] = self.y_train
        csv_file = self._create_csv_file(df)
        
        # Make request
        response = client.post(
            f"/ml-engine/models/{self.model_name}/calibrate",
            data={"method": "isotonic", "cv": "3"},
            files={"calibration_data": ("test.csv", csv_file, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["original_model"], self.model_name)
        self.assertIn("calibrated_model", data)
        self.assertEqual(data["calibration_method"], "isotonic")

    def test_interpret_model(self):
        """Test POST /ml-engine/interpret-model endpoint."""
        # Create test data file
        df = pd.DataFrame(self.X_train[:5])  # Just use a few samples for efficiency
        csv_file = self._create_csv_file(df)
        
        # Make request
        response = client.post(
            "/ml-engine/interpret-model",
            data={"model_name": self.model_name, "method": "eli5"},
            files={"sample_data": ("test.csv", csv_file, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["model_name"], self.model_name)
        self.assertEqual(data["method"], "eli5")
        self.assertIn("feature_weights", data)

    def test_explain_prediction(self):
        """Test POST /ml-engine/explain-prediction endpoint."""
        # Create test data file with just one sample
        df = pd.DataFrame(self.X_train[:1])
        csv_file = self._create_csv_file(df)
        
        # Make request
        response = client.post(
            "/ml-engine/explain-prediction",
            data={"model_name": self.model_name, "method": "eli5"},
            files={"sample_data": ("test.csv", csv_file, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["model_name"], self.model_name)
        self.assertIn("feature_weights", data)

    def test_model_health_check(self):
        """Test POST /ml-engine/health-check endpoint."""
        # Create test data file
        df = pd.DataFrame(self.X_train)
        df["target"] = self.y_train
        csv_file = self._create_csv_file(df)
        
        # Make request with test data
        response = client.post(
            "/ml-engine/health-check",
            data={"model_name": self.model_name},
            files={"test_data": ("test.csv", csv_file, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["model_name"], self.model_name)
        self.assertIn("status", data)
        self.assertIn("checks", data)
        self.assertTrue(len(data["checks"]) > 0)

    def test_batch_process_pipeline(self):
        """Test POST /ml-engine/batch-process-pipeline endpoint."""
        # Create test data file
        df = pd.DataFrame(self.X_train)
        csv_file = self._create_csv_file(df)
        
        # Make request
        response = client.post(
            "/ml-engine/batch-process-pipeline",
            data={"model_name": self.model_name, "steps": '["preprocess", "predict", "postprocess"]'},
            files={"input_data": ("test.csv", csv_file, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["model_name"], self.model_name)
        self.assertEqual(data["input_rows"], len(self.X_train))
        self.assertIn("steps", data)
        self.assertIn("execution_time", data)

    def test_batch_inference(self):
        """Test POST /ml-engine/batch-inference endpoint."""
        # Create two test data files
        df1 = pd.DataFrame(self.X_train[:50])
        df2 = pd.DataFrame(self.X_train[50:])
        
        csv_file1 = self._create_csv_file(df1)
        csv_file2 = self._create_csv_file(df2)
        
        # Make request
        response = client.post(
            "/ml-engine/batch-inference",
            data={
                "model_name": self.model_name,
                "batch_size": "10",
                "return_proba": "false",
                "parallel": "true"
            },
            files=[
                ("batch_data", ("test1.csv", csv_file1, "text/csv")),
                ("batch_data", ("test2.csv", csv_file2, "text/csv"))
            ]
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["model_used"], self.model_name)
        self.assertEqual(data["batch_count"], 2)
        self.assertEqual(data["total_samples"], len(self.X_train))
        self.assertEqual(len(data["results"]), 2)

    def test_quantize_model(self):
        """Test POST /ml-engine/models/{model_name}/quantize endpoint."""
        response = client.post(f"/ml-engine/models/{self.model_name}/quantize")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["model_name"], self.model_name)
        self.assertIn("original_size_bytes", data)
        self.assertIn("quantized_size_bytes", data)
        self.assertIn("compression_ratio", data)
        
        # Clean up quantized model file
        if os.path.exists(data["quantized_filepath"]):
            os.remove(data["quantized_filepath"])

    def test_transfer_learning(self):
        """Test POST /ml-engine/models/{model_name}/transfer-learning endpoint."""
        # Create test data file
        df = pd.DataFrame(self.X_train)
        df["target"] = self.y_train
        csv_file = self._create_csv_file(df)
        
        # Make request
        response = client.post(
            f"/ml-engine/models/{self.model_name}/transfer-learning",
            data={"learning_rate": "0.01", "epochs": "5"},
            files={"new_data": ("test.csv", csv_file, "text/csv")}
        )
        
        # This test might fail if the original model doesn't support warm_start
        # But RandomForestClassifier does support it
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["original_model"], self.model_name)
        self.assertIn("new_model", data)
        self.assertIn(data["new_model"], self.ml_engine.models)

    def test_shutdown(self):
        """Test POST /ml-engine/shutdown endpoint."""
        response = client.post("/ml-engine/shutdown")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("message", data)
        self.assertIn("shut down successfully", data["message"])


if __name__ == "__main__":
    unittest.main()