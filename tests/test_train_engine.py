import unittest
import pytest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import make_classification, make_regression, load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configure pytest-asyncio to use function scope for fixtures
pytestmark = pytest.mark.asyncio_default_fixture_loop_scope("function")

# Import the ML Training Engine components
from modules.configs import TaskType, OptimizationStrategy, MLTrainingEngineConfig
from modules.configs import InferenceEngineConfig, BatchProcessorConfig, PreprocessorConfig
from modules.configs import NormalizationType, QuantizationConfig, QuantizationType, QuantizationMode
from modules.engine.inference_engine import InferenceEngine
from modules.engine.batch_processor import BatchProcessor
from modules.engine.data_preprocessor import DataPreprocessor
from modules.engine.quantizer import Quantizer
from modules.optimizer.asht import ASHTOptimizer
from modules.engine.train_engine import MLTrainingEngine


class MockPreprocessor:
    """Mock preprocessor for testing"""
    
    def __init__(self):
        self.is_fitted = False
        
    def fit(self, X):
        self.is_fitted = True
        return self
        
    def transform(self, X):
        return X
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class TestMLTrainingEngine:
    """Comprehensive test suite for the ML Training Engine"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(self, request):
        """Set up test data that can be reused across test methods"""
        # Suppress warnings during tests
        warnings.filterwarnings("ignore")
        
        # Load standard datasets
        iris = load_iris()
        request.cls.X_iris, request.cls.y_iris = iris.data, iris.target
        
        diabetes = load_diabetes()
        request.cls.X_diabetes, request.cls.y_diabetes = diabetes.data, diabetes.target
        
        # Create synthetic datasets
        X_class, y_class = make_classification(
            n_samples=200, n_features=10, n_informative=5, 
            n_redundant=2, random_state=42
        )
        request.cls.X_class = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(X_class.shape[1])])
        request.cls.y_class = pd.Series(y_class, name='target')
        
        X_reg, y_reg = make_regression(
            n_samples=200, n_features=10, n_informative=5,
            noise=0.1, random_state=42
        )
        request.cls.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
        request.cls.y_reg = pd.Series(y_reg, name='target')
        
        # Create temporary directory for model storage
        request.cls.test_dir = tempfile.mkdtemp()
        
        yield
        
        # Clean up after all tests - remove temporary directory
        shutil.rmtree(request.cls.test_dir)
    
    @pytest.fixture(autouse=True)
    def setup_method(self, request):
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
            experiment_tracking=True,
            log_level="INFO"
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
            experiment_tracking=True,
            log_level="INFO"
        )
        
        # Create ASHT-specific config for testing
        self.asht_config = MLTrainingEngineConfig(
            task_type=TaskType.CLASSIFICATION,
            random_state=42,
            n_jobs=1,
            verbose=0,
            cv_folds=3,
            test_size=0.2,
            stratify=True,
            optimization_strategy=OptimizationStrategy.ASHT,  # Use the ASHT strategy
            optimization_iterations=5,  # More iterations for ASHT
            feature_selection=True,
            feature_selection_method="mutual_info",
            model_path=os.path.join(self.test_dir, "asht"),
            experiment_tracking=True,
            log_level="INFO"
        )
        
        # Create model definitions for testing
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
            },
            "linear_regression": {
                "model": LinearRegression(),
                "params": {
                    "model__fit_intercept": [True, False]
                }
            }
        }
        
        # Mock preprocessing config
        self.preprocessing_config = {
            "numeric_features": ["feature_0", "feature_1", "feature_2"],
            "categorical_features": [],
            "standardize_numeric": True,
            "handle_missing": "mean"
        }
    
    def test_engine_initialization(self):
        """Test that the engine initializes correctly"""
        # Classification engine
        cls_engine = MLTrainingEngine(self.classification_config)
        assert isinstance(cls_engine, MLTrainingEngine)
        assert cls_engine.config.task_type == TaskType.CLASSIFICATION
        
        # Regression engine
        reg_engine = MLTrainingEngine(self.regression_config)
        assert isinstance(reg_engine, MLTrainingEngine)
        assert reg_engine.config.task_type == TaskType.REGRESSION
        
        # ASHT engine
        asht_engine = MLTrainingEngine(self.asht_config)
        assert isinstance(asht_engine, MLTrainingEngine)
        assert asht_engine.config.optimization_strategy == OptimizationStrategy.ASHT
        
        # Check if the model directory is created
        assert os.path.exists(self.test_dir)
        
        # Check if experiment tracker is initialized
        assert cls_engine.tracker is not None
        
        # Clean up
        cls_engine.shutdown()
        reg_engine.shutdown()
        asht_engine.shutdown()
    
    def test_train_model_classification(self):
        """Test training a classification model"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Create a simple model and parameter grid
        model_info = self.classification_models["logistic_regression"]
        
        # Train the model
        best_model, metrics = engine.train_model(
            model=model_info["model"],
            model_name="test_logistic",
            param_grid=model_info["params"],
            X=self.X_class,
            y=self.y_class
        )
        
        # Check if model is stored correctly
        assert "test_logistic" in engine.models
        assert engine.best_model is not None
        
        # Check if metrics are computed
        assert "accuracy" in metrics
        assert "f1" in metrics
        
        # Check if best model is selected
        assert engine.best_model["name"] == "test_logistic"
        
        # Clean up
        engine.shutdown()
    
    def test_train_model_regression(self):
        """Test training a regression model"""
        engine = MLTrainingEngine(self.regression_config)
        
        # Create a simple model and parameter grid
        model_info = self.regression_models["linear_regression"]
        
        # Train the model
        best_model, metrics = engine.train_model(
            model=model_info["model"],
            model_name="test_linear",
            param_grid=model_info["params"],
            X=self.X_reg,
            y=self.y_reg
        )
        
        # Check if model is stored correctly
        self.assertIn("test_linear", engine.models)
        self.assertIsNotNone(engine.best_model)
        
        # Check if metrics are computed
        self.assertIn("mse", metrics)
        self.assertIn("r2", metrics)
        
        # Check if best model is selected
        self.assertEqual(engine.best_model["name"], "test_linear")
        
        # Clean up
        engine.shutdown()
    
    def test_asht_optimization(self):
        """Test the ASHT optimization strategy"""
        # Initialize engine with ASHT config
        engine = MLTrainingEngine(self.asht_config)
        
        # Train a model using ASHT
        model_info = self.classification_models["random_forest"]
        model, metrics = engine.train_model(
            model=model_info["model"],
            model_name="asht_rf",
            param_grid=model_info["params"],
            X=self.X_iris,
            y=self.y_iris
        )
        
        # Check if training was successful
        self.assertIsNotNone(model)
        self.assertIsNotNone(metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("asht_rf", engine.models)
        
        # Check if model was stored correctly
        self.assertIsNotNone(engine.models["asht_rf"]["model"])
        
        # Check if the model performs reasonably well
        self.assertGreater(metrics["accuracy"], 0.7)  # Basic sanity check
        
        # Clean up
        engine.shutdown()
        
    def test_asht_vs_random_search(self):
        """Compare ASHT with random search on the same problem"""
        # Train with ASHT
        asht_engine = MLTrainingEngine(self.asht_config)
        model_info = self.classification_models["random_forest"]
        _, asht_metrics = asht_engine.train_model(
            model=model_info["model"],
            model_name="asht_compare",
            param_grid=model_info["params"],
            X=self.X_iris,
            y=self.y_iris
        )
        asht_engine.shutdown()
        
        # Train with Random Search (using same iterations for fair comparison)
        random_config = MLTrainingEngineConfig(
            task_type=TaskType.CLASSIFICATION,
            random_state=42,
            n_jobs=1,
            verbose=0,
            cv_folds=3,
            test_size=0.2,
            stratify=True,
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
            optimization_iterations=5,  # Same as ASHT
            feature_selection=True,
            feature_selection_method="mutual_info",
            model_path=os.path.join(self.test_dir, "random"),
            experiment_tracking=True,
            log_level="INFO"
        )
        
        random_engine = MLTrainingEngine(random_config)
        _, random_metrics = random_engine.train_model(
            model=model_info["model"],
            model_name="random_compare",
            param_grid=model_info["params"],
            X=self.X_iris,
            y=self.y_iris
        )
        random_engine.shutdown()
        
        # Both should produce valid results
        self.assertIn("accuracy", asht_metrics)
        self.assertIn("accuracy", random_metrics)
        
        # Note: We don't assert which is better since that's probabilistic
    
    def test_evaluate_model(self):
        """Test model evaluation functionality"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Train a simple model first
        model_info = self.classification_models["logistic_regression"]
        
        engine.train_model(
            model=model_info["model"],
            model_name="eval_test",
            param_grid=model_info["params"],
            X=self.X_class,
            y=self.y_class
        )
        
        # Evaluate model with new data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_class, self.y_class, test_size=0.3, random_state=42
        )
        
        metrics = engine.evaluate_model(
            model_name="eval_test", 
            X_test=X_test, 
            y_test=y_test, 
            detailed=True
        )
        
        # Check if evaluation metrics are returned
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Test detailed evaluation
        self.assertIn("detailed_report", metrics)
        self.assertIn("confusion_matrix", metrics)
        
        # Test evaluation for non-existent model
        error_result = engine.evaluate_model(model_name="non_existent")
        self.assertIn("error", error_result)
        
        # Clean up
        engine.shutdown()
    
    def test_evaluate_all_models(self):
        """Test evaluating all trained models on test data"""
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
    
    def test_predict(self):
        """Test model prediction functionality"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Train a simple model first
        model_info = self.classification_models["logistic_regression"]
        
        engine.train_model(
            model=model_info["model"],
            model_name="predict_test",
            param_grid=model_info["params"],
            X=self.X_class,
            y=self.y_class
        )
        
        # Generate predictions
        predictions = engine.predict(self.X_class)
        
        # Check if predictions are generated
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.X_class))
        
        # Test probability predictions
        proba_predictions = engine.predict(self.X_class, return_proba=True)
        self.assertEqual(proba_predictions.shape[1], 2)  # Binary classification
        
        # Test batch prediction
        batch_predictions = engine.predict(self.X_class, batch_size=50)
        self.assertIsNotNone(batch_predictions)
        self.assertEqual(len(batch_predictions), len(self.X_class))
        
        # Test prediction with non-existent model
        none_predictions = engine.predict(self.X_class, model_name="non_existent")
        self.assertIsNone(none_predictions)
        
        # Clean up
        engine.shutdown()
    
    def test_batch_inference(self):
        """Test batch inference functionality"""
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
        batch_results = engine.run_batch_inference(
            data_generator(), 
            batch_size=10,
            model_name="batch_test"
        )
        
        # Check batch predictions
        self.assertIsNotNone(batch_results)
        self.assertEqual(len(batch_results), len(list(data_generator())))
        
        # Compare with direct predictions for one batch
        sample_batch = self.X_diabetes[:10]
        direct_prediction = engine.predict(sample_batch, "batch_test")
        batch_prediction = batch_results[0]
        
        np.testing.assert_allclose(batch_prediction, direct_prediction, rtol=1e-5)
        
        # Clean up
        engine.shutdown()
    
    def test_save_load_model(self):
        """Test model saving and loading functionality"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Train a simple model first
        model_info = self.classification_models["random_forest"]
        
        engine.train_model(
            model=model_info["model"],
            model_name="save_test",
            param_grid=model_info["params"],
            X=self.X_class,
            y=self.y_class
        )
        
        # Save the model
        success, filepath = engine.save_model(model_name="save_test")
        
        # Check if save was successful
        self.assertTrue(success)
        self.assertTrue(os.path.exists(filepath))
        
        # Create a new engine and load the model
        new_engine = MLTrainingEngine(self.classification_config)
        loaded_model = new_engine.load_model(filepath)
        
        # Check if model was loaded successfully
        self.assertIsNotNone(loaded_model)
        self.assertIn("save_test", new_engine.models)
        
        # Generate predictions with loaded model and compare
        orig_predictions = engine.predict(self.X_class)
        loaded_predictions = new_engine.predict(self.X_class)
        
        np.testing.assert_array_equal(orig_predictions, loaded_predictions)
        
        # Clean up
        engine.shutdown()
        new_engine.shutdown()
    
    def test_feature_importance(self):
        """Test feature importance calculation and reporting"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Train a model with built-in feature importance
        model_info = self.classification_models["random_forest"]
        
        engine.train_model(
            model=model_info["model"],
            model_name="importance_test",
            param_grid=model_info["params"],
            X=self.X_class,
            y=self.y_class
        )
        
        # Generate feature importance report
        result = engine.generate_feature_importance_report(
            model_name="importance_test",
            top_n=5
        )
        
        # Check if report contains expected data
        self.assertIn("model_name", result)
        self.assertIn("feature_importance", result)
        self.assertIn("top_features", result)
        
        # Ensure top features are sorted by importance
        importances = list(result["top_features"].values())
        self.assertEqual(importances, sorted(importances, reverse=True))
        
        # Clean up
        engine.shutdown()
    
    def test_feature_selection(self):
        """Test feature selection functionality"""
        # Create a config with explicit feature selection
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
            experiment_tracking=True,
            log_level="INFO"
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
        lr_model = pipeline.named_steps['model']
        
        n_classes = len(np.unique(self.y_iris))
        n_selected_features = 2
        coef = lr_model.coef_
        
        # If multiclass, shape is (n_classes, n_features)
        # Some sklearn versions flatten the coefficients for binary classification
        if n_classes == 2 and coef.ndim == 1:
            self.assertEqual(coef.shape[0], n_selected_features)
        else:
            self.assertEqual(coef.shape[1], n_selected_features)
        
        # Check the feature selector's functionality directly
        selected_features = engine._perform_feature_selection(self.X_iris, self.y_iris)
        self.assertIsInstance(selected_features, list)
        self.assertEqual(len(selected_features), 2)  # Should have 2 features
        
        # Clean up
        engine.shutdown()
    
    def test_compare_models(self):
        """Test model comparison functionality"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Train multiple models
        for name, model_info in self.classification_models.items():
            engine.train_model(
                model=model_info["model"],
                model_name=name,
                param_grid=model_info["params"],
                X=self.X_class,
                y=self.y_class
            )
        
        # Compare models
        comparison = engine.compare_models(
            model_names=list(self.classification_models.keys()),
            metrics=["accuracy", "f1"]
        )
        
        # Check if comparison contains expected data
        self.assertIn("models", comparison)
        self.assertIn("metrics", comparison)
        self.assertIn("data", comparison)
        
        # Check if both models are included
        for model_name in self.classification_models.keys():
            self.assertIn(model_name, comparison["data"])
        
        # Check if metrics are included
        for model_name in self.classification_models.keys():
            self.assertIn("accuracy", comparison["data"][model_name]["metrics"])
            self.assertIn("f1", comparison["data"][model_name]["metrics"])
        
        # Clean up
        engine.shutdown()
    
    def test_error_analysis(self):
        """Test error analysis functionality"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Train a model
        model_info = self.classification_models["logistic_regression"]
        
        engine.train_model(
            model=model_info["model"],
            model_name="error_test",
            param_grid=model_info["params"],
            X=self.X_class,
            y=self.y_class
        )
        
        # Prepare test data with some errors
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_class, self.y_class, test_size=0.3, random_state=0
        )
        
        # Perform error analysis
        analysis = engine.perform_error_analysis(
            model_name="error_test",
            X_test=X_test,
            y_test=y_test,
            n_samples=10
        )
        
        # Check if analysis contains expected data
        self.assertIn("model_name", analysis)
        self.assertIn("dataset_size", analysis)
        self.assertIn("error_count", analysis)
        self.assertIn("error_rate", analysis)
        
        # Check if confusion matrix is included
        self.assertIn("confusion_matrix", analysis)
        
        # Check if class metrics are included
        self.assertIn("class_metrics", analysis)
        
        # Check if detailed samples are included
        self.assertIn("detailed_samples", analysis)
        
        # Clean up
        engine.shutdown()
    
    def test_data_drift(self):
        """Test data drift detection functionality"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Create reference data
        reference_data = self.X_class.copy()
        
        # Create new data with drift
        drifted_data = self.X_class.copy()
        drifted_data['feature_0'] = drifted_data['feature_0'] + 2.0  # Add shift
        
        # Detect drift
        drift_results = engine.detect_data_drift(
            new_data=drifted_data,
            reference_data=reference_data,
            drift_threshold=0.1
        )
        
        # Check if drift detection contains expected data
        self.assertIn("feature_drift", drift_results)
        self.assertIn("dataset_drift", drift_results)
        self.assertIn("drifted_features", drift_results)
        self.assertIn("drift_detected", drift_results)
        
        # Check if feature_0 is detected as drifted
        self.assertIn("feature_0", drift_results["drifted_features"])
        
        # Overall drift should be detected
        self.assertTrue(drift_results["drift_detected"])
        
        # Clean up
        engine.shutdown()
    
    def test_preprocessing_integration(self):
        """Test integration with preprocessor"""
        # Update config with preprocessing
        config = self.classification_config
        config.preprocessing_config = self.preprocessing_config
        
        engine = MLTrainingEngine(config)
        
        # Check if preprocessor is initialized
        self.assertIsNotNone(engine.preprocessor)
        
        # Train a model with preprocessing
        model_info = self.classification_models["logistic_regression"]
        
        engine.train_model(
            model=model_info["model"],
            model_name="preproc_test",
            param_grid=model_info["params"],
            X=self.X_class,
            y=self.y_class
        )
        
        # Check if model training succeeded
        self.assertIn("preproc_test", engine.models)
        
        # Test prediction with preprocessing
        predictions = engine.predict(self.X_class)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.X_class))
        
        # Clean up
        engine.shutdown()
    
    def test_report_generation(self):
        """Test report generation functionality"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Train multiple models
        models = [
            (LogisticRegression(random_state=42), "logistic_report", {'C': [1.0]}),
            (RandomForestClassifier(random_state=42, n_estimators=10), "rf_report", {'n_estimators': [10]})
        ]
        
        for model, name, params in models:
            engine.train_model(
                model=model,
                model_name=name,
                param_grid=params,
                X=self.X_class,
                y=self.y_class
            )
        
        # Generate reports - use both generate_report and generate_reports methods
        standard_report_path = engine.generate_report()
        markdown_report_path = engine.generate_reports(include_plots=True)
        
        # Check if report files were created
        self.assertTrue(os.path.exists(standard_report_path))
        self.assertTrue(os.path.exists(markdown_report_path))
        
        # Check if reports contain key information
        with open(markdown_report_path, 'r') as f:
            content = f.read()
            self.assertIn("Model Performance Summary", content)
            self.assertIn("logistic_report", content)
            self.assertIn("rf_report", content)
        
        # Clean up
        engine.shutdown()
    
    def test_export_model(self):
        """Test model exporting functionality"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Train a model
        model_info = self.classification_models["logistic_regression"]
        
        engine.train_model(
            model=model_info["model"],
            model_name="export_test",
            param_grid=model_info["params"],
            X=self.X_class,
            y=self.y_class
        )
        
        # Export model to sklearn format
        export_path = engine.export_model(
            model_name="export_test",
            format="sklearn",
            include_pipeline=True
        )
        
        # Check if export file was created
        self.assertTrue(os.path.exists(export_path))
        
        # Check if metadata file was also created
        metadata_path = export_path + ".json"
        self.assertTrue(os.path.exists(metadata_path))
        
        # Clean up
        engine.shutdown()
    
    def test_shutdown(self):
        """Test engine shutdown functionality"""
        engine = MLTrainingEngine(self.classification_config)
        
        # Train a model
        model_info = self.classification_models["logistic_regression"]
        
        engine.train_model(
            model=model_info["model"],
            model_name="shutdown_test",
            param_grid=model_info["params"],
            X=self.X_class,
            y=self.y_class
        )
        
        # Test shutdown with auto-save
        engine.config.auto_save_on_shutdown = True
        engine.shutdown()
        
        # Check if a model file was created during shutdown
        model_files = [f for f in os.listdir(engine.config.model_path) if f.startswith("shutdown_test")]
        self.assertTrue(len(model_files) > 0)
    
    def test_asht_surrogate_model(self):
        """Test that ASHT creates and uses a surrogate model"""
        # Initialize engine with ASHT config
        engine = MLTrainingEngine(self.asht_config)
        
        # Get the optimization search object
        model_info = self.classification_models["random_forest"]
        search = engine._get_optimization_search(model_info["model"], model_info["params"])
        
        # Verify it's an ASHTOptimizer
        self.assertEqual(search.__class__.__name__, "ASHTOptimizer")
        
        # Verify it has a surrogate model
        self.assertTrue(hasattr(search, 'surrogate_model'))
        
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
    
    def test_edge_cases(self):
        """Test various edge cases"""
        # Initialize engine
        engine = MLTrainingEngine(self.classification_config)
        
        # Test predicting with non-existent model
        predictions = engine.predict(self.X_iris[:5], "nonexistent_model")
        assert predictions is None
        
        # Test loading non-existent model
        nonexistent_path = os.path.join(self.test_dir, "nonexistent.pkl")
        loaded_model = engine.load_model(nonexistent_path)
        assert loaded_model is None
        
        # Test ASHT with empty parameter grid
        asht_engine = MLTrainingEngine(self.asht_config)
        empty_params = {}
        model_info = self.classification_models["random_forest"]
        
        # This should handle gracefully without errors, but we expect an exception
        # because the optimizer won't have any parameters to vary.
        with pytest.raises(Exception):
            asht_engine._get_optimization_search(
                model_info["model"], 
                empty_params
            ).fit(self.X_iris, self.y_iris)
        
        # Test with invalid data
        with pytest.raises(ValueError):
            engine._validate_training_data(None, None)
        
        # Test with mismatched data sizes
        with pytest.raises(ValueError):
            engine._validate_training_data(self.X_iris[:10], self.y_iris[:5])
        
        # Clean up
        engine.shutdown()
        asht_engine.shutdown()
    
    def test_hyperoptx_optimization(self):
        """Test the HyperOptX optimization strategy if available"""
        try:
            # Create HyperOptX-specific config
            hyperx_config = MLTrainingEngineConfig(
                task_type=TaskType.CLASSIFICATION,
                random_state=42,
                n_jobs=1,
                verbose=0,
                cv_folds=3,
                test_size=0.2,
                stratify=True,
                optimization_strategy=OptimizationStrategy.HYPERX,  # Use HyperOptX strategy
                optimization_iterations=5,
                feature_selection=True,
                feature_selection_method="mutual_info",
                model_path=os.path.join(self.test_dir, "hyperx"),
                experiment_tracking=True,
                log_level="INFO"
            )
            
            # Initialize engine
            engine = MLTrainingEngine(hyperx_config)
            
            # Get the optimization search object
            model_info = self.classification_models["random_forest"]
            search = engine._get_optimization_search(model_info["model"], model_info["params"])
            
            # Verify it's a HyperOptX optimizer
            self.assertEqual(search.__class__.__name__, "HyperOptX")
            
            # Train a model
            model, metrics = engine.train_model(
                model=model_info["model"],
                model_name="hyperx_rf",
                param_grid=model_info["params"],
                X=self.X_iris,
                y=self.y_iris
            )
            
            # Check if training was successful
            self.assertIsNotNone(model)
            self.assertIsNotNone(metrics)
            self.assertIn("accuracy", metrics)
            
            # Clean up
            engine.shutdown()
        except (ImportError, AttributeError, NotImplementedError):
            # Skip this test if HyperOptX is not available
            self.skipTest("HyperOptX optimizer not available")

if __name__ == "__main__":
    unittest.main()