import unittest
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, Mock
import pickle
import tempfile
import shutil
from sklearn.datasets import make_classification, make_regression

# Import the module under test
from modules.configs import TaskType, OptimizationStrategy, ModelSelectionCriteria, MLTrainingEngineConfig
from modules.engine.train_engine import MLTrainingEngine, ExperimentTracker


class TestMLTrainingEngine(unittest.TestCase):
    """Test suite for MLTrainingEngine class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a basic configuration
        self.config = MLTrainingEngineConfig()
        self.config.task_type = TaskType.CLASSIFICATION
        self.config.model_path = "./test_models"
        self.config.random_state = 42
        self.config.n_jobs = 1  # Use 1 job for testing
        self.config.test_size = 0.2
        self.config.cv_folds = 2  # Use 2 folds for faster testing
        self.config.experiment_tracking = False
        self.config.optimization_strategy = OptimizationStrategy.RANDOM_SEARCH
        self.config.optimization_iterations = 2  # Small number for testing
        self.config.model_selection_criteria = ModelSelectionCriteria.F1
        
        # Create test data
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        
        # For regression tests
        self.X_reg, self.y_reg = make_regression(
            n_samples=100,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42
        )
        
        # Create pandas DataFrame versions
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(self.X.shape[1])])
        self.y_df = pd.Series(self.y, name='target')
        
        # Initialize engine
        self.engine = MLTrainingEngine(self.config)
        
        # Create temp directory for test artifacts
        self.test_dir = tempfile.mkdtemp()
        self.config.model_path = os.path.join(self.test_dir, "models")
        os.makedirs(self.config.model_path, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Clean up test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
        # Clean up engine
        if hasattr(self, 'engine') and self.engine:
            self.engine.shutdown()

    def test_initialization(self):
        """Test engine initialization with different configs."""
        # Test basic initialization
        engine = MLTrainingEngine(self.config)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.config.task_type, TaskType.CLASSIFICATION)
        
        # Test with regression config
        reg_config = MLTrainingEngineConfig()
        reg_config.task_type = TaskType.REGRESSION
        reg_engine = MLTrainingEngine(reg_config)
        self.assertEqual(reg_engine.config.task_type, TaskType.REGRESSION)
        
        # Clean up
        reg_engine.shutdown()

    def test_register_model_types(self):
        """Test model type registration."""
        # Check if common model types are registered
        self.assertTrue(hasattr(self.engine, '_model_registry'))
        self.assertIn('classification', self.engine._model_registry)
        self.assertIn('regression', self.engine._model_registry)
        
        # Check specific models
        self.assertIn('random_forest', self.engine._model_registry['classification'])
        self.assertIn('logistic_regression', self.engine._model_registry['classification'])
        self.assertIn('linear_regression', self.engine._model_registry['regression'])

    def test_train_model_classification(self):
        """Test model training with classification data."""
        # Basic training with random forest
        result = self.engine.train_model(self.X, self.y, model_type='random_forest')
        
        # Verify results
        self.assertIn('model', result)
        self.assertIn('metrics', result)
        self.assertIn('model_name', result)
        self.assertGreater(len(self.engine.models), 0)
        self.assertIsNotNone(self.engine.best_model)
        
        # Check metrics format
        metrics = result['metrics']
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        
    def test_train_model_regression(self):
        """Test model training with regression data."""
        # Set up regression config
        reg_config = MLTrainingEngineConfig()
        reg_config.task_type = TaskType.REGRESSION
        reg_config.model_path = os.path.join(self.test_dir, "reg_models")
        reg_config.cv_folds = 2
        reg_config.optimization_iterations = 2
        
        # Create engine
        reg_engine = MLTrainingEngine(reg_config)
        
        # Train model
        result = reg_engine.train_model(self.X_reg, self.y_reg, model_type='random_forest')
        
        # Verify results
        self.assertIn('model', result)
        self.assertIn('metrics', result)
        
        # Check metrics format for regression
        metrics = result['metrics']
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mae', metrics)
        
        # Clean up
        reg_engine.shutdown()

    def test_train_model_with_dataframe(self):
        """Test model training with pandas DataFrame input."""
        result = self.engine.train_model(self.X_df, self.y_df, model_type='random_forest')
        
        # Verify results with DataFrame input
        self.assertIsNotNone(result['model'])
        self.assertIn('metrics', result)
        
        # Check if feature names were extracted
        model_name = result['model_name']
        if model_name in self.engine.models:
            self.assertTrue(len(self.engine.models[model_name].get('feature_names', [])) > 0)

    def test_multiple_model_training(self):
        """Test training multiple models and comparing them."""
        # Train two different models
        self.engine.train_model(self.X, self.y, model_type='random_forest', model_name='rf_model')
        self.engine.train_model(self.X, self.y, model_type='logistic_regression', model_name='lr_model')
        
        # Get performance comparison
        comparison = self.engine.get_performance_comparison()
        
        # Verify comparison structure
        self.assertIn('models', comparison)
        self.assertIn('best_model', comparison)
        self.assertGreaterEqual(len(comparison['models']), 2)
        
        # Check that models are properly ranked
        for model in comparison['models']:
            self.assertIn('name', model)
            self.assertIn('metrics', model)
            self.assertIn('is_best', model)

    def test_save_and_load_model(self):
        """Test saving and loading models."""
        # Train a model
        result = self.engine.train_model(self.X, self.y, model_type='random_forest', model_name='save_test')
        
        # Save the model
        save_path = self.engine.save_model('save_test')
        self.assertIsNotNone(save_path)
        self.assertTrue(os.path.exists(save_path))
        
        # Create a new engine
        new_engine = MLTrainingEngine(self.config)
        
        # Load the model
        success, loaded_model = new_engine.load_model(save_path, 'loaded_model')
        
        # Verify loading
        self.assertTrue(success)
        self.assertIsNotNone(loaded_model)
        self.assertIn('loaded_model', new_engine.models)
        
        # Test prediction with loaded model
        success, predictions = new_engine.predict(self.X, model_name='loaded_model')
        self.assertTrue(success)
        self.assertEqual(len(predictions), len(self.y))
        
        # Clean up
        new_engine.shutdown()

    def test_evaluation_methods(self):
        """Test model evaluation methods."""
        # Train a model
        self.engine.train_model(self.X, self.y, model_type='random_forest')
        
        # Test basic evaluation
        metrics = self.engine.evaluate_model(X_test=self.X, y_test=self.y)
        
        # Verify metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        
        # Test detailed evaluation
        detailed_metrics = self.engine.evaluate_model(X_test=self.X, y_test=self.y, detailed=True)
        
        # Verify detailed metrics include additional information
        self.assertIn('confusion_matrix', detailed_metrics)
        self.assertIn('detailed_report', detailed_metrics)

    def test_prediction(self):
        """Test prediction functionality."""
        # Train a model
        self.engine.train_model(self.X, self.y, model_type='random_forest')
        
        # Test standard prediction
        success, predictions = self.engine.predict(self.X)
        self.assertTrue(success)
        self.assertEqual(len(predictions), len(self.y))
        
        # Test probability prediction
        success, probabilities = self.engine.predict(self.X, return_proba=True)
        self.assertTrue(success)
        self.assertEqual(probabilities.shape[0], len(self.y))
        self.assertEqual(probabilities.shape[1], 2)  # Binary classification

    @patch('modules.engine.train_engine.plt')
    def test_feature_importance(self, mock_plt):
        """Test feature importance calculation."""
        # Skip if matplotlib not available
        try:
            import matplotlib
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False
            self.skipTest("Matplotlib not available")
        
        # Mock the PLOTTING_AVAILABLE flag
        with patch('modules.engine.train_engine.PLOTTING_AVAILABLE', True):
            # Train a model with feature names
            self.engine.train_model(self.X_df, self.y_df, model_type='random_forest')
            
            # Test feature importance in the trained model
            model_info = self.engine.best_model
            self.assertIn('feature_importance', model_info)
            self.assertIsNotNone(model_info['feature_importance'])

    def test_get_model_summary(self):
        """Test model summary generation."""
        # Train a model
        self.engine.train_model(self.X, self.y, model_type='random_forest', model_name='summary_test')
        
        # Get model summary
        summary = self.engine.get_model_summary('summary_test')
        
        # Verify summary structure
        self.assertIn('model_name', summary)
        self.assertIn('model_type', summary)
        self.assertIn('metrics', summary)
        self.assertIn('is_best_model', summary)
        
        # Check summary content
        self.assertEqual(summary['model_name'], 'summary_test')
        self.assertIn('RandomForest', summary['model_type'])

    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.end_run')
    def test_experiment_tracking(self, mock_end_run, mock_log_metric, mock_log_param, mock_start_run):
        """Test experiment tracking with mocked MLflow."""
        # Set up config with experiment tracking
        track_config = MLTrainingEngineConfig()
        track_config.task_type = TaskType.CLASSIFICATION
        track_config.model_path = os.path.join(self.test_dir, "tracked_models")
        track_config.experiment_tracking = True
        
        # Define mock mlflow class to be returned in import check
        class MockMLflow:
            @staticmethod
            def start_run(*args, **kwargs):
                return mock_start_run(*args, **kwargs)
                
            @staticmethod
            def log_param(*args, **kwargs):
                return mock_log_param(*args, **kwargs)
                
            @staticmethod
            def log_metric(*args, **kwargs):
                return mock_log_metric(*args, **kwargs)
                
            @staticmethod
            def end_run(*args, **kwargs):
                return mock_end_run(*args, **kwargs)
        
        # Mock the import check for MLflow
        with patch('modules.engine.train_engine.MLFLOW_AVAILABLE', True), \
             patch('modules.engine.train_engine.mlflow', MockMLflow):
            # Create engine with tracking
            tracked_engine = MLTrainingEngine(track_config)
            
            # Train model with tracking
            tracked_engine.train_model(self.X, self.y, model_type='random_forest')
            
            # Verify MLflow interactions
            mock_start_run.assert_called()
            mock_log_param.assert_called()
            mock_log_metric.assert_called()
            mock_end_run.assert_called()
            
            # Clean up
            tracked_engine.shutdown()

    def test_error_handling(self):
        """Test error handling in the training engine."""
        # Test with invalid model type
        with self.assertRaises(ValueError):
            self.engine.train_model(self.X, self.y, model_type='invalid_model_type')
        
        # Test loading non-existent model
        success, error = self.engine.load_model('nonexistent_model.pkl')
        self.assertFalse(success)
        self.assertIsInstance(error, str)
        
        # Test prediction with non-existent model
        success, error = self.engine.predict(self.X, model_name='nonexistent_model')
        self.assertFalse(success)
        self.assertIsInstance(error, str)


class TestExperimentTracker(unittest.TestCase):
    """Test suite for ExperimentTracker class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temp directory for test artifacts
        self.test_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(output_dir=self.test_dir)
        
        # Create sample data
        self.X, self.y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        # Mock model
        self.mock_model = Mock()
        self.mock_model.feature_importances_ = np.random.rand(5)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Clean up test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ExperimentTracker(output_dir=self.test_dir, experiment_name="test_exp")
        self.assertEqual(tracker.experiment_name, "test_exp")
        self.assertEqual(tracker.output_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_experiment_workflow(self):
        """Test a complete experiment tracking workflow."""
        # Start experiment
        config = {"task_type": "classification", "cv_folds": 5}
        model_info = {"model_type": "random_forest", "model_class": "sklearn.ensemble.RandomForestClassifier"}
        self.tracker.start_experiment(config, model_info)
        
        # Log metrics
        metrics = {"accuracy": 0.95, "f1": 0.94, "precision": 0.93, "recall": 0.92}
        self.tracker.log_metrics(metrics)
        
        # Log feature importance
        feature_names = [f"feature_{i}" for i in range(5)]
        importance = np.random.rand(5)
        self.tracker.log_feature_importance(feature_names, importance)
        
        # Log model
        model_path = os.path.join(self.test_dir, "test_model.pkl")
        self.tracker.log_model(self.mock_model, "test_model", model_path)
        
        # End experiment
        result = self.tracker.end_experiment()
        
        # Verify results
        self.assertIn("metrics", result)
        self.assertIn("feature_importance", result)
        self.assertIn("artifacts", result)
        self.assertEqual(result["config"], config)
        self.assertEqual(result["model_info"], model_info)
        
        # Check experiment file was created
        exp_files = [f for f in os.listdir(self.test_dir) if f.startswith("experiment_")]
        self.assertTrue(len(exp_files) > 0)

    @patch('modules.engine.train_engine.plt')
    def test_confusion_matrix(self, mock_plt):
        """Test confusion matrix generation."""
        # Skip if matplotlib not available
        try:
            import matplotlib
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False
            self.skipTest("Matplotlib not available")
            
        # Mock the PLOTTING_AVAILABLE flag
        with patch('modules.engine.train_engine.PLOTTING_AVAILABLE', True):
            # Start experiment
            self.tracker.start_experiment({}, {})
            
            # Create sample data
            y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
            y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
            
            # Log confusion matrix
            self.tracker.log_confusion_matrix(y_true, y_pred)
            
            # Verify artifact was created
            self.assertIn("confusion_matrix", self.tracker.artifacts)
            self.assertTrue(os.path.exists(self.tracker.artifacts["confusion_matrix"]))

    def test_generate_report(self):
        """Test report generation."""
        # Start experiment
        config = {"task_type": "classification", "cv_folds": 5}
        model_info = {"model_type": "random_forest"}
        self.tracker.start_experiment(config, model_info)
        
        # Log metrics
        metrics = {"accuracy": 0.95, "f1": 0.94}
        self.tracker.log_metrics(metrics)
        
        # End experiment
        self.tracker.end_experiment()
        
        # Generate report
        report_path = self.tracker.generate_report()
        
        # Verify report
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn("Experiment Report", content)
            self.assertIn("accuracy", content)
            self.assertIn("0.95", content)


if __name__ == "__main__":
    unittest.main()