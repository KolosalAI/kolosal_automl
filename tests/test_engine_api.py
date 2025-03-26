import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import asyncio
from io import BytesIO

# Import router from engine_api to access all endpoints
from modules.engine_api import router

# Extract API functions from router endpoints for easier access
# This creates a dictionary mapping function names to their handler functions
api_functions = {route.name: route.endpoint for route in router.routes if hasattr(route, 'name')}

# Access functions by name
get_ml_engine = api_functions.get('get_ml_engine', None)
list_models = api_functions.get('list_models', None)
get_model_details = api_functions.get('get_model_details', None)
train_model = api_functions.get('train_model', None)
predict = api_functions.get('predict', None)
evaluate_model = api_functions.get('evaluate_model', None)
feature_importance = api_functions.get('feature_importance', None)
error_analysis = api_functions.get('error_analysis', None)
data_drift = api_functions.get('data_drift', None)
compare_models = api_functions.get('compare_models', None)
save_model = api_functions.get('save_model', None)
load_model = api_functions.get('load_model', None)
delete_model = api_functions.get('delete_model', None)
generate_report = api_functions.get('generate_report', None)
batch_inference = api_functions.get('batch_inference', None)
interpret_model = api_functions.get('interpret_model', None)
explain_prediction = api_functions.get('explain_prediction', None)
health_check = api_functions.get('health_check', None)
create_ensemble = api_functions.get('create_ensemble', None)
batch_process_pipeline = api_functions.get('batch_process_pipeline', None)
calibrate_model = api_functions.get('calibrate_model', None)
transfer_learning = api_functions.get('transfer_learning', None)
model_quantization = api_functions.get('model_quantization', None)
shutdown_engine = api_functions.get('shutdown_engine', None)

# Import config and engine
from modules.configs import MLTrainingEngineConfig, TaskType
from modules.engine.train_engine import MLTrainingEngine

# Create a mock UploadFile class for testing
class MockUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content
        
    async def read(self):
        return self.content

# Create a mock BackgroundTasks class for testing
class MockBackgroundTasks:
    def __init__(self):
        pass
        
    def add_task(self, func, *args, **kwargs):
        # Actually run the task immediately for testing
        func(*args, **kwargs)

class TestMLEngineAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and models once for all tests"""
        # Create a classification dataset
        X_cls, y_cls = make_classification(
            n_samples=500, 
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        # Create a regression dataset
        X_reg, y_reg = make_regression(
            n_samples=500,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        
        # Split data
        cls.X_train_cls, cls.X_test_cls, cls.y_train_cls, cls.y_test_cls = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42
        )
        
        cls.X_train_reg, cls.X_test_reg, cls.y_train_reg, cls.y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Save test data to temporary CSV files
        cls.temp_dir = tempfile.TemporaryDirectory()
        
        # Classification data
        cls.train_cls_file = os.path.join(cls.temp_dir.name, "train_cls.csv")
        cls.test_cls_file = os.path.join(cls.temp_dir.name, "test_cls.csv")
        
        # Create DataFrames
        train_cls_df = pd.DataFrame(cls.X_train_cls, columns=[f"feature_{i}" for i in range(10)])
        train_cls_df["target"] = cls.y_train_cls
        train_cls_df.to_csv(cls.train_cls_file, index=False)
        
        test_cls_df = pd.DataFrame(cls.X_test_cls, columns=[f"feature_{i}" for i in range(10)])
        test_cls_df["target"] = cls.y_test_cls
        test_cls_df.to_csv(cls.test_cls_file, index=False)
        
        # Regression data
        cls.train_reg_file = os.path.join(cls.temp_dir.name, "train_reg.csv")
        cls.test_reg_file = os.path.join(cls.temp_dir.name, "test_reg.csv")
        
        train_reg_df = pd.DataFrame(cls.X_train_reg, columns=[f"feature_{i}" for i in range(10)])
        train_reg_df["target"] = cls.y_train_reg
        train_reg_df.to_csv(cls.train_reg_file, index=False)
        
        test_reg_df = pd.DataFrame(cls.X_test_reg, columns=[f"feature_{i}" for i in range(10)])
        test_reg_df["target"] = cls.y_test_reg
        test_reg_df.to_csv(cls.test_reg_file, index=False)
        
        # Create engines with different task types
        cls.engine_cls = MLTrainingEngine(
            MLTrainingEngineConfig(
                task_type=TaskType.CLASSIFICATION,
                model_path=cls.temp_dir.name,
                experiment_tracking=True
            )
        )
        
        cls.engine_reg = MLTrainingEngine(
            MLTrainingEngineConfig(
                task_type=TaskType.REGRESSION,
                model_path=cls.temp_dir.name,
                experiment_tracking=True
            )
        )
        
        # Train sample models directly on the engines
        cls.cls_model = RandomForestClassifier(n_estimators=10, random_state=42)
        cls.reg_model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Train classification model
        cls.engine_cls.train_model(
            model=cls.cls_model, 
            model_name="test_rf_classifier",
            param_grid={"n_estimators": [10]},
            X=cls.X_train_cls,
            y=cls.y_train_cls,
            X_test=cls.X_test_cls,
            y_test=cls.y_test_cls
        )
        
        # Train regression model
        cls.engine_reg.train_model(
            model=cls.reg_model,
            model_name="test_rf_regressor",
            param_grid={"n_estimators": [10]},
            X=cls.X_train_reg,
            y=cls.y_train_reg,
            X_test=cls.X_test_reg,
            y_test=cls.y_test_reg
        )
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        cls.temp_dir.cleanup()
        
    # Helper method to run async functions in tests
    def run_async(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)
        
    def test_list_models(self):
        """Test listing all models"""
        # Mock get_ml_engine
        def mock_get_engine():
            return self.engine_cls
            
        result = self.run_async(list_models(engine=mock_get_engine()))
        
        self.assertIn("models", result)
        self.assertIn("count", result)
        self.assertIn("best_model", result)
        self.assertGreaterEqual(result["count"], 1)
        
    def test_get_model_details(self):
        """Test getting details of a specific model"""
        def mock_get_engine():
            return self.engine_cls
            
        # Test classification model
        result = self.run_async(get_model_details(model_name="test_rf_classifier", engine=mock_get_engine()))
        
        self.assertEqual(result["name"], "test_rf_classifier")
        self.assertIn("metrics", result)
        self.assertIn("params", result)
        
        # Test for a non-existent model
        with self.assertRaises(Exception):
            self.run_async(get_model_details(model_name="nonexistent_model", engine=mock_get_engine()))
        
    def test_train_model(self):
        """Test training a new model"""
        def mock_get_engine():
            return self.engine_cls
            
        # Create mock file
        with open(self.train_cls_file, "rb") as f:
            file_content = f.read()
            
        # Create TrainModelParams object
        from pydantic import BaseModel
        
        class TrainModelParams(BaseModel):
            model_type: str
            model_name: str
            param_grid: dict
            test_size: float = 0.2
            stratify: bool = True
            
        params = TrainModelParams(
            model_type="random_forest",
            model_name="test_new_classifier",
            param_grid={"n_estimators": [10], "max_depth": [5]}
        )
        
        mock_file = MockUploadFile("train_data.csv", file_content)
        
        result = self.run_async(
            train_model(
                background_tasks=MockBackgroundTasks(),
                params=params,
                data_file=mock_file,
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["model_name"], "test_new_classifier")
        self.assertEqual(result["status"], "training_started")
        
        # Verify the model exists in the engine
        self.assertIn("test_new_classifier", self.engine_cls.models)
        
    def test_predict(self):
        """Test making predictions"""
        def mock_get_engine():
            return self.engine_cls
            
        # Create mock file
        with open(self.test_cls_file, "rb") as f:
            file_content = f.read()
            
        mock_file = MockUploadFile("test_data.csv", file_content)
        
        # Create BatchPredictionParams
        from pydantic import BaseModel
        
        class BatchPredictionParams(BaseModel):
            model_name: str = "test_rf_classifier"
            batch_size: int = None
            return_proba: bool = False
            
        params = BatchPredictionParams()
        
        result = self.run_async(
            predict(
                batch_data=mock_file,
                params=params,
                engine=mock_get_engine()
            )
        )
        
        self.assertIn("predictions", result)
        self.assertIn("model_used", result)
        self.assertIn("row_count", result)
        self.assertEqual(result["model_used"], "test_rf_classifier")
        
    def test_evaluate_model(self):
        """Test evaluating a model with test data"""
        def mock_get_engine():
            return self.engine_cls
            
        with open(self.test_cls_file, "rb") as f:
            file_content = f.read()
            
        mock_file = MockUploadFile("test_data.csv", file_content)
        
        result = self.run_async(
            evaluate_model(
                model_name="test_rf_classifier",
                test_data=mock_file,
                detailed=True,
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["model_name"], "test_rf_classifier")
        self.assertIn("metrics", result)
        self.assertIn("accuracy", result["metrics"])
        self.assertIn("precision", result["metrics"])
        self.assertIn("recall", result["metrics"])
        self.assertIn("f1", result["metrics"])
        
    def test_feature_importance(self):
        """Test generating feature importance"""
        def mock_get_engine():
            return self.engine_cls
            
        # Create FeatureImportanceParams
        from pydantic import BaseModel
        
        class FeatureImportanceParams(BaseModel):
            model_name: str = "test_rf_classifier"
            top_n: int = 10
            include_plot: bool = True
            
        params = FeatureImportanceParams()
        
        result = self.run_async(
            feature_importance(
                params=params,
                engine=mock_get_engine()
            )
        )
        
        self.assertNotIn("error", result)
        self.assertIn("feature_importance", result)
        
    def test_error_analysis(self):
        """Test error analysis"""
        def mock_get_engine():
            return self.engine_cls
            
        with open(self.test_cls_file, "rb") as f:
            file_content = f.read()
            
        mock_file = MockUploadFile("test_data.csv", file_content)
        
        # Create ErrorAnalysisParams
        from pydantic import BaseModel
        
        class ErrorAnalysisParams(BaseModel):
            model_name: str = "test_rf_classifier"
            n_samples: int = 10
            include_plot: bool = True
            
        params = ErrorAnalysisParams()
        
        result = self.run_async(
            error_analysis(
                test_data=mock_file,
                params=params,
                engine=mock_get_engine()
            )
        )
        
        self.assertNotIn("error", result)
        
    def test_data_drift(self):
        """Test data drift detection"""
        def mock_get_engine():
            return self.engine_cls
            
        with open(self.test_cls_file, "rb") as f:
            new_data_content = f.read()
            
        with open(self.train_cls_file, "rb") as f:
            ref_data_content = f.read()
            
        new_data = MockUploadFile("new_data.csv", new_data_content)
        ref_data = MockUploadFile("ref_data.csv", ref_data_content)
        
        # Create DataDriftParams
        from pydantic import BaseModel
        
        class DataDriftParams(BaseModel):
            drift_threshold: float = 0.1
            
        params = DataDriftParams()
        
        result = self.run_async(
            data_drift(
                new_data=new_data,
                reference_data=ref_data,
                params=params,
                engine=mock_get_engine()
            )
        )
        
        self.assertNotIn("error", result)
        self.assertIn("drift_detected", result)
        
    def test_compare_models(self):
        """Test comparing models"""
        def mock_get_engine():
            return self.engine_cls
            
        # Create ModelComparisonParams
        from pydantic import BaseModel
        
        class ModelComparisonParams(BaseModel):
            model_names: list = ["test_rf_classifier"]
            metrics: list = None
            include_plot: bool = True
            
        params = ModelComparisonParams()
        
        result = self.run_async(
            compare_models(
                params=params,
                engine=mock_get_engine()
            )
        )
        
        self.assertNotIn("error", result)
        self.assertIn("comparison_table", result)
        
    def test_save_model(self):
        """Test saving a model"""
        def mock_get_engine():
            return self.engine_cls
            
        result = self.run_async(
            save_model(
                model_name="test_rf_classifier",
                version_tag="v1",
                include_preprocessor=True,
                include_metadata=True,
                compression_level=5,
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["model_name"], "test_rf_classifier")
        self.assertIn("filepath", result)
        self.assertIn("size_bytes", result)
        
    def test_load_model(self):
        """Test loading a model"""
        def mock_get_engine():
            return self.engine_cls
            
        # First, save a model to get a file to load
        save_result = self.run_async(
            save_model(
                model_name="test_rf_classifier",
                engine=mock_get_engine()
            )
        )
        
        # Read the saved file
        with open(save_result["filepath"], "rb") as f:
            file_content = f.read()
            
        mock_file = MockUploadFile("model.pkl", file_content)
        
        result = self.run_async(
            load_model(
                model_file=mock_file,
                validate_metrics=True,
                engine=mock_get_engine()
            )
        )
        
        self.assertIn("model_name", result)
        self.assertIn("model_type", result)
        self.assertIn("is_best", result)
        
    def test_delete_model(self):
        """Test deleting a model"""
        def mock_get_engine():
            return self.engine_cls
            
        # First, ensure the model exists
        self.assertIn("test_rf_classifier", self.engine_cls.models)
        
        # Create a new model for deletion to avoid affecting other tests
        with open(self.train_cls_file, "rb") as f:
            file_content = f.read()
            
        from pydantic import BaseModel
        
        class TrainModelParams(BaseModel):
            model_type: str
            model_name: str
            param_grid: dict
            test_size: float = 0.2
            stratify: bool = True
            
        params = TrainModelParams(
            model_type="random_forest",
            model_name="model_to_delete",
            param_grid={"n_estimators": [10]}
        )
        
        mock_file = MockUploadFile("train_data.csv", file_content)
        
        # Train a model to delete
        self.run_async(
            train_model(
                background_tasks=MockBackgroundTasks(),
                params=params,
                data_file=mock_file,
                engine=mock_get_engine()
            )
        )
        
        # Verify the model exists
        self.assertIn("model_to_delete", self.engine_cls.models)
        
        # Delete the model
        result = self.run_async(
            delete_model(
                model_name="model_to_delete",
                engine=mock_get_engine()
            )
        )
        
        self.assertIn("message", result)
        self.assertNotIn("model_to_delete", self.engine_cls.models)
        
    def test_generate_report(self):
        """Test generating a comprehensive report"""
        def mock_get_engine():
            return self.engine_cls
            
        result = self.run_async(
            generate_report(
                include_plots=True,
                engine=mock_get_engine()
            )
        )
        
        self.assertIn("report", result)
        self.assertIn("model_count", result)
        self.assertIn("best_model", result)
        
    def test_batch_inference(self):
        """Test running batch inference"""
        def mock_get_engine():
            return self.engine_cls
            
        # Create multiple test files
        files = []
        for i in range(2):
            with open(self.test_cls_file, "rb") as f:
                file_content = f.read()
                files.append(MockUploadFile(f"test_data_{i}.csv", file_content))
        
        result = self.run_async(
            batch_inference(
                batch_data=files,
                model_name="test_rf_classifier",
                batch_size=None,
                return_proba=False,
                parallel=True,
                timeout=None,
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["model_used"], "test_rf_classifier")
        self.assertEqual(result["batch_count"], 2)
        self.assertIn("results", result)
        
    def test_interpret_model(self):
        """Test model interpretation"""
        def mock_get_engine():
            return self.engine_cls
            
        with open(self.test_cls_file, "rb") as f:
            file_content = f.read()
            
        mock_file = MockUploadFile("test_data.csv", file_content)
        
        result = self.run_async(
            interpret_model(
                model_name="test_rf_classifier",
                sample_data=mock_file,
                method="built_in",  # Using built-in for simplicity in tests
                background_samples=10,
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["model_name"], "test_rf_classifier")
        self.assertIn("method", result)
        
    def test_explain_prediction(self):
        """Test prediction explanation"""
        def mock_get_engine():
            return self.engine_cls
            
        with open(self.test_cls_file, "rb") as f:
            file_content = f.read()
            
        mock_file = MockUploadFile("test_data.csv", file_content)
        
        result = self.run_async(
            explain_prediction(
                sample_data=mock_file,
                model_name="test_rf_classifier",
                method="built_in",  # Using built-in for simplicity in tests
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["model_name"], "test_rf_classifier")
        self.assertIn("method", result)
        
    def test_health_check(self):
        """Test model health check"""
        def mock_get_engine():
            return self.engine_cls
            
        with open(self.test_cls_file, "rb") as f:
            file_content = f.read()
            
        mock_file = MockUploadFile("test_data.csv", file_content)
        
        result = self.run_async(
            health_check(
                test_data=mock_file,
                model_name="test_rf_classifier",
                engine=mock_get_engine()
            )
        )
        
        self.assertIn("status", result)
        self.assertIn("checks", result)
        self.assertIn("model_name", result)
        
    def test_transfer_learning(self):
        """Test transfer learning"""
        def mock_get_engine():
            return self.engine_cls
            
        with open(self.train_cls_file, "rb") as f:
            file_content = f.read()
            
        mock_file = MockUploadFile("train_data.csv", file_content)
        
        result = self.run_async(
            transfer_learning(
                model_name="test_rf_classifier",
                new_data=mock_file,
                learning_rate=0.01,
                epochs=2,
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["original_model"], "test_rf_classifier")
        self.assertIn("new_model", result)
        self.assertIn(result["new_model"], self.engine_cls.models)
        
    def test_create_ensemble(self):
        """Test creating ensemble models"""
        def mock_get_engine():
            return self.engine_cls
            
        # Create a second model to ensemble with
        with open(self.train_cls_file, "rb") as f:
            file_content = f.read()
            
        from pydantic import BaseModel
        
        class TrainModelParams(BaseModel):
            model_type: str
            model_name: str
            param_grid: dict
            test_size: float = 0.2
            stratify: bool = True
            
        params = TrainModelParams(
            model_type="decision_tree",
            model_name="test_dt_classifier",
            param_grid={"max_depth": [3]}
        )
        
        mock_file = MockUploadFile("train_data.csv", file_content)
        
        # Train a second model
        self.run_async(
            train_model(
                background_tasks=MockBackgroundTasks(),
                params=params,
                data_file=mock_file,
                engine=mock_get_engine()
            )
        )
        
        # Create ensemble
        result = self.run_async(
            create_ensemble(
                model_names=["test_rf_classifier", "test_dt_classifier"],
                ensemble_name="test_ensemble",
                voting_type="soft",
                weights=None,
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["ensemble_name"], "test_ensemble")
        self.assertIn("test_ensemble", self.engine_cls.models)
        
    def test_calibrate_model(self):
        """Test model calibration"""
        def mock_get_engine():
            return self.engine_cls
            
        with open(self.train_cls_file, "rb") as f:
            file_content = f.read()
            
        mock_file = MockUploadFile("train_data.csv", file_content)
        
        result = self.run_async(
            calibrate_model(
                model_name="test_rf_classifier",
                calibration_data=mock_file,
                method="isotonic",
                cv=2,
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["original_model"], "test_rf_classifier")
        self.assertIn("calibrated_model", result)
        self.assertIn(result["calibrated_model"], self.engine_cls.models)
        
    def test_model_quantization(self):
        """Test model quantization"""
        def mock_get_engine():
            return self.engine_cls
            
        result = self.run_async(
            model_quantization(
                model_name="test_rf_classifier",
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["model_name"], "test_rf_classifier")
        self.assertIn("original_size_bytes", result)
        self.assertIn("quantized_size_bytes", result)
        self.assertIn("compression_ratio", result)
        
    def test_batch_process_pipeline(self):
        """Test batch processing pipeline"""
        def mock_get_engine():
            return self.engine_cls
            
        with open(self.test_cls_file, "rb") as f:
            file_content = f.read()
            
        mock_file = MockUploadFile("test_data.csv", file_content)
        
        result = self.run_async(
            batch_process_pipeline(
                input_data=mock_file,
                model_name="test_rf_classifier",
                steps=["preprocess", "predict", "postprocess"],
                batch_size=None,
                engine=mock_get_engine()
            )
        )
        
        self.assertEqual(result["model_name"], "test_rf_classifier")
        self.assertIn("steps", result)
        self.assertIn("execution_time", result)
        
    def test_shutdown_engine(self):
        """Test shutting down the engine"""
        def mock_get_engine():
            return self.engine_cls
            
        result = self.run_async(
            shutdown_engine(
                engine=mock_get_engine()
            )
        )
        
        self.assertIn("message", result)

if __name__ == "__main__":
    unittest.main()
