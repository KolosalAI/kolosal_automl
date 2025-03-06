import unittest
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

# Import our ML Training Engine components
from modules.configs import InferenceEngineConfig, BatchProcessorConfig, PreprocessorConfig
from modules.configs import NormalizationType, QuantizationConfig, QuantizationType, QuantizationMode
from modules.engine.inference_engine import InferenceEngine
from modules.engine.batch_processor import BatchProcessor
from modules.engine.data_preprocessor import DataPreprocessor
from modules.engine.quantizer import Quantizer

# The main training engine components
from modules.engine.train_engine import (
    MLTrainingEngine, 
    MLTrainingEngineConfig, 
    TaskType, 
    OptimizationStrategy
)


class TestMLTrainingEngine(unittest.TestCase):
    """Test suite for the ML Training Engine"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across test methods"""
        # Load datasets
        cls.iris = load_iris()
        cls.X_iris, cls.y_iris = cls.iris.data, cls.iris.target
        
        cls.diabetes = load_diabetes()
        cls.X_diabetes, cls.y_diabetes = cls.diabetes.data, cls.diabetes.target
        
        # Create temporary directory for model storage
        cls.test_dir = tempfile.mkdtemp()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove temporary directory
        shutil.rmtree(cls.test_dir)
        
    def setUp(self):
        """Set up before each test method"""
        # Create basic configs for testing
        self.classification_config = MLTrainingEngineConfig(
            task_type=TaskType.CLASSIFICATION,
            random_state=42,
            n_jobs=1,  # Use 1 job for testing to avoid parallel issues
            verbose=0,  # Suppress verbose output during tests
            cv_folds=3,  # Use fewer folds for faster testing
            test_size=0.2,
            stratify=True,
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
            optimization_iterations=2,  # Fewer iterations for faster testing
            feature_selection=True,
            feature_selection_method="mutual_info",
            model_path=os.path.join(self.test_dir, "classification"),
            experiment_tracking=True
        )
        
        self.regression_config = MLTrainingEngineConfig(
            task_type=TaskType.REGRESSION,
            random_state=42,
            n_jobs=1,
            verbose=0,
            cv_folds=3,
            test_size=0.2,
            stratify=False,
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
            optimization_iterations=2,
            feature_selection=True,
            feature_selection_method="mutual_info",
            model_path=os.path.join(self.test_dir, "regression"),
            experiment_tracking=True
        )
        
        # Create model definitions
        self.classification_models = {
            "random_forest": {
                "model": RandomForestClassifier(random_state=42, n_estimators=10),  # Small model for testing
                "params": {
                    "model__n_estimators": [10, 20],
                    "model__max_depth": [3, 5]
                }
            },
            "logistic_regression": {
                "model": LogisticRegression(random_state=42),
                "params": {
                    "model__C": [0.1, 1.0],
                    "model__solver": ["liblinear"]
                }
            }
        }
        
        self.regression_models = {
            "gradient_boosting": {
                "model": GradientBoostingRegressor(random_state=42, n_estimators=10),
                "params": {
                    "model__n_estimators": [10, 20],
                    "model__max_depth": [2, 3]
                }
            },
            "ridge": {
                "model": Ridge(random_state=42),
                "params": {
                    "model__alpha": [0.1, 1.0]
                }
            }
        }
        
    def test_engine_initialization(self):
        """Test that the engine initializes correctly"""
        # Classification engine
        cls_engine = MLTrainingEngine(self.classification_config)
        self.assertIsInstance(cls_engine, MLTrainingEngine)
        self.assertEqual(cls_engine.config.task_type, TaskType.CLASSIFICATION)
        
        # Regression engine
        reg_engine = MLTrainingEngine(self.regression_config)
        self.assertIsInstance(reg_engine, MLTrainingEngine)
        self.assertEqual(reg_engine.config.task_type, TaskType.REGRESSION)
        
        # Clean up
        cls_engine.shutdown()
        reg_engine.shutdown()
        
    def test_classification_training(self):
        """Test training a classification model"""
        # Initialize engine
        engine = MLTrainingEngine(self.classification_config)
        
        # Train a model
        model_info = self.classification_models["random_forest"]
        model, metrics = engine.train_model(
            model=model_info["model"],
            model_name="test_rf",
            param_grid=model_info["params"],
            X=self.X_iris,
            y=self.y_iris
        )
        
        # Check if training was successful
        self.assertIsNotNone(model)
        self.assertIsNotNone(metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("test_rf", engine.models)
        
        # Check if model was stored correctly
        self.assertIsNotNone(engine.models["test_rf"]["model"])
        
        # Clean up
        engine.shutdown()
        
    def test_regression_training(self):
        """Test training a regression model"""
        # Initialize engine
        engine = MLTrainingEngine(self.regression_config)
        
        # Train a model
        model_info = self.regression_models["ridge"]
        model, metrics = engine.train_model(
            model=model_info["model"],
            model_name="test_ridge",
            param_grid=model_info["params"],
            X=self.X_diabetes,
            y=self.y_diabetes
        )
        
        # Check if training was successful
        self.assertIsNotNone(model)
        self.assertIsNotNone(metrics)
        self.assertIn("mse", metrics)
        self.assertIn("test_ridge", engine.models)
        
        # Clean up
        engine.shutdown()
        
    def test_multiple_models_training(self):
        """Test training multiple models and comparing them"""
        # Initialize engine
        engine = MLTrainingEngine(self.classification_config)
        
        # Train multiple models
        for name, model_info in self.classification_models.items():
            engine.train_model(
                model=model_info["model"],
                model_name=name,
                param_grid=model_info["params"],
                X=self.X_iris,
                y=self.y_iris
            )
            
        # Check if all models were trained
        self.assertEqual(len(engine.models), len(self.classification_models))
        
        # Check if best model was selected
        self.assertIsNotNone(engine.best_model)
        self.assertIn(engine.best_model["name"], self.classification_models.keys())
        
        # Clean up
        engine.shutdown()
        
    def test_model_saving_and_loading(self):
        """Test model serialization and deserialization"""
        # Initialize engine
        engine = MLTrainingEngine(self.classification_config)
        
        # Train a model
        model_info = self.classification_models["random_forest"]
        engine.train_model(
            model=model_info["model"],
            model_name="save_test",
            param_grid=model_info["params"],
            X=self.X_iris,
            y=self.y_iris
        )
        
        # Save the model
        saved = engine.save_model("save_test")
        self.assertTrue(saved)
        
        # Create a new engine
        new_engine = MLTrainingEngine(self.classification_config)
        
        # Load the model
        model_path = os.path.join(self.classification_config.model_path, "save_test.pkl")
        loaded_model = new_engine.load_model(model_path)
        
        # Check if model was loaded correctly
        self.assertIsNotNone(loaded_model)
        self.assertIn("save_test", new_engine.models)
        
        # Make a prediction with loaded model
        X_sample = self.X_iris[:5]
        predictions = new_engine.predict(X_sample, "save_test")
        
        # Check predictions
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), 5)
        
        # Clean up
        engine.shutdown()
        new_engine.shutdown()
        
    def test_model_evaluation(self):
        """Test model evaluation functionality"""
        # Initialize engine
        engine = MLTrainingEngine(self.classification_config)
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_iris, self.y_iris, 
            test_size=0.3, 
            random_state=42, 
            stratify=self.y_iris
        )
        
        # Train multiple models
        for name, model_info in self.classification_models.items():
            engine.train_model(
                model=model_info["model"],
                model_name=name,
                param_grid=model_info["params"],
                X=X_train,
                y=y_train
            )
            
        # Evaluate all models
        results = engine.evaluate_all_models(X_test, y_test)
        
        # Check evaluation results
        self.assertIsNotNone(results)
        self.assertEqual(len(results), len(self.classification_models))
        
        for model_name, metrics in results.items():
            self.assertIn("accuracy", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1", metrics)
        
        # Clean up
        engine.shutdown()
    
    def test_batch_inference(self):
        """Test batch inference functionality"""
        # Initialize engine
        engine = MLTrainingEngine(self.regression_config)
        
        # Train a model
        model_info = self.regression_models["ridge"]
        engine.train_model(
            model=model_info["model"],
            model_name="batch_test",
            param_grid=model_info["params"],
            X=self.X_diabetes,
            y=self.y_diabetes
        )
        
        # Create data generator for batch processing
        def data_generator():
            batch_size = 10
            for i in range(0, len(self.X_diabetes), batch_size):
                yield self.X_diabetes[i:i+batch_size]
        
        # Run batch inference
        batch_predictions = engine.run_batch_inference(
            data_generator(), 
            batch_size=10,
            model_name="batch_test"
        )
        
        # Check batch predictions
        self.assertIsNotNone(batch_predictions)
        self.assertEqual(len(batch_predictions), len(self.X_diabetes))
        
        # Compare with direct predictions
        direct_predictions = engine.predict(self.X_diabetes, "batch_test")
        np.testing.assert_allclose(batch_predictions, direct_predictions, rtol=1e-5)
        
        # Clean up
        engine.shutdown()
    
    def test_report_generation(self):
        """Test report generation functionality"""
        # Initialize engine
        engine = MLTrainingEngine(self.classification_config)
        
        # Train a couple of models
        for name, model_info in list(self.classification_models.items())[:1]:  # Just train one for speed
            engine.train_model(
                model=model_info["model"],
                model_name=name,
                param_grid=model_info["params"],
                X=self.X_iris,
                y=self.y_iris
            )
            
        # Generate report
        report_path = engine.generate_report()
        
        # Check if report was generated
        self.assertIsNotNone(report_path)
        self.assertTrue(os.path.exists(report_path))
        
        # Check report contents (basic check)
        with open(report_path, 'r') as f:
            report_content = f.read()
            self.assertIn("ML Training Engine Report", report_content)
            self.assertIn("Model Performance Summary", report_content)
            
        # Clean up
        engine.shutdown()
    
    def test_feature_selection(self):
        """Test feature selection functionality"""
        # Create a config with feature selection enabled
        config = MLTrainingEngineConfig(
            task_type=TaskType.CLASSIFICATION,
            random_state=42,
            n_jobs=1,
            verbose=0,
            cv_folds=3,
            test_size=0.2,
            stratify=True,
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
            optimization_iterations=2,
            feature_selection=True,
            feature_selection_method="mutual_info",
            feature_selection_k=2,  # Select only 2 features
            model_path=os.path.join(self.test_dir, "feature_selection"),
            experiment_tracking=True
        )
        
        # Initialize engine
        engine = MLTrainingEngine(config)
        
        # Train a model
        model_info = self.classification_models["logistic_regression"]
        model, metrics = engine.train_model(
            model=model_info["model"],
            model_name="feature_test",
            param_grid=model_info["params"],
            X=self.X_iris,
            y=self.y_iris
        )
        
        # Check if feature selection was applied
        self.assertIsNotNone(engine.feature_selector)
        
        # The pipeline should transform data with only selected features
        pipeline = engine.models["feature_test"]["model"]
        
        # For logistic regression, check the coefficients shape
        # If feature selection worked, we should only have 2 coefficients per class
        lr_model = pipeline.named_steps['model']
        
        # The number of coefficients should match the number of selected features x number of classes
        n_classes = len(np.unique(self.y_iris))
        n_selected_features = 2
        
        # Check coefficient shape - if multiclass, shape is (n_classes, n_features)
        coef = lr_model.coef_
        
        # Some sklearn versions flatten the coefficients for binary classification
        if n_classes == 2 and coef.ndim == 1:
            expected_shape = (n_selected_features,)
        else:
            expected_shape = (n_classes, n_selected_features)
            
        if coef.ndim > 1:  # Check the number of features in multi-dimensional case
            self.assertEqual(coef.shape[1], n_selected_features)
        else:
            self.assertEqual(coef.shape[0], n_selected_features)
        
        # Clean up
        engine.shutdown()
        
    def test_edge_cases(self):
        """Test various edge cases"""
        # Initialize engine
        engine = MLTrainingEngine(self.classification_config)
        
        # Test predicting with non-existent model
        predictions = engine.predict(self.X_iris[:5], "nonexistent_model")
        self.assertIsNone(predictions)
        
        # Test loading non-existent model
        nonexistent_path = os.path.join(self.test_dir, "nonexistent.pkl")
        loaded_model = engine.load_model(nonexistent_path)
        self.assertIsNone(loaded_model)
        
        # Clean up
        engine.shutdown()


if __name__ == '__main__':
    unittest.main()
