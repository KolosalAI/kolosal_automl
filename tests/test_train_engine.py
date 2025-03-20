import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import joblib
import matplotlib.pyplot as plt

# Import the classes we want to test
from modules.engine.train_engine import MLTrainingEngine, ExperimentTracker
from modules.configs import MLTrainingEngineConfig, TaskType, OptimizationStrategy


class TestMLTrainingEngine(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create a basic configuration for testing
        self.config = MLTrainingEngineConfig(
            model_path=self.test_dir,
            task_type=TaskType.CLASSIFICATION,
            feature_selection=True,
            feature_selection_k=5,
            cv_folds=2,
            test_size=0.2,
            random_state=42,
            optimization_strategy=OptimizationStrategy.GRID_SEARCH,
            optimization_iterations=2,
            experiment_tracking=True,
            auto_save=False,
            verbose=0,
            n_jobs=1
        )
        
        # Create the training engine
        self.engine = MLTrainingEngine(self.config)
        
        # Create some dummy data for testing
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'feature4': np.random.rand(100),
            'feature5': np.random.rand(100),
        })
        self.y = np.random.randint(0, 2, 100)  # Binary classification targets
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
    def tearDown(self):
        """Clean up after each test"""
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
        
        # Close any open matplotlib figures
        plt.close('all')

    def test_initialization(self):
        """Test the initialization of MLTrainingEngine"""
        self.assertIsInstance(self.engine, MLTrainingEngine)
        self.assertEqual(self.engine.config.model_path, self.test_dir)
        self.assertEqual(self.engine.config.task_type, TaskType.CLASSIFICATION)
        self.assertIsNotNone(self.engine.preprocessor)
        self.assertIsNotNone(self.engine.tracker)
        self.assertEqual(len(self.engine.models), 0)
        self.assertIsNone(self.engine.best_model)

    def test_train_model(self):
        """Test training a basic model"""
        # Create a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model_name = "test_decision_tree"
        param_grid = {"max_depth": [2, 3]}
        
        # Mock GridSearchCV to avoid actual training
        with patch('sklearn.model_selection.GridSearchCV') as MockGridSearchCV:
            # Configure the mock
            mock_grid_search = MockGridSearchCV.return_value
            mock_grid_search.best_estimator_ = model
            mock_grid_search.best_params_ = {"max_depth": 3}
            mock_grid_search.best_score_ = 0.9
            mock_grid_search.best_index_ = 0
            mock_grid_search.cv_results_ = {
                'mean_test_score': [0.9],
                'std_test_score': [0.05],
                'split0_test_score': [0.85],
                'split1_test_score': [0.95]
            }
            
            # Train the model
            trained_model, metrics = self.engine.train_model(
                model, model_name, param_grid, 
                self.X_train, self.y_train, 
                self.X_test, self.y_test
            )
        
        # Check that the model was stored
        self.assertIn(model_name, self.engine.models)
        self.assertEqual(self.engine.models[model_name]["model"], model)
        self.assertEqual(self.engine.models[model_name]["params"]["max_depth"], 3)

    def test_evaluate_model(self):
        """Test model evaluation"""
        # Create a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model_name = "test_eval_model"
        
        # Mock the trained model
        with patch.object(self.engine, '_evaluate_model') as mock_evaluate:
            mock_metrics = {
                "accuracy": 0.85,
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.85
            }
            mock_evaluate.return_value = mock_metrics
            
            # Store a mock model
            self.engine.models[model_name] = {
                "name": model_name,
                "model": model,
                "params": {},
                "metrics": mock_metrics
            }
            
            # Set as best model
            self.engine.best_model = model_name
            
            # Evaluate the model
            metrics = self.engine.evaluate_model(model_name, self.X_test, self.y_test)
        
        # Check metrics
        self.assertEqual(metrics["accuracy"], 0.85)
        self.assertEqual(metrics["precision"], 0.9)
        self.assertEqual(metrics["recall"], 0.8)
        self.assertEqual(metrics["f1"], 0.85)

    def test_feature_importance(self):
        """Test feature importance extraction"""
        # Create a model with known feature importance
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Set importance attributes
        importances = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        model.feature_importances_ = importances
        
        # Test the feature importance extraction
        extracted_importance = self.engine._get_feature_importance(model)
        
        # Check that importance values are retrieved and normalized
        self.assertIsNotNone(extracted_importance)
        self.assertEqual(len(extracted_importance), 5)
        self.assertAlmostEqual(np.sum(extracted_importance), 1.0)

    def test_save_load_model(self):
        """Test saving and loading a model"""
        # Create and train a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        model_name = "test_save_model"
        
        # Store the model
        self.engine.models[model_name] = {
            "name": model_name,
            "model": model,
            "params": {"max_depth": 3},
            "metrics": {"accuracy": 0.85}
        }
        
        # Save the model
        success, filepath = self.engine.save_model(model_name)
        
        # Check that save was successful
        self.assertTrue(success)
        self.assertTrue(os.path.exists(filepath))
        
        # Create a new engine and load the model
        new_engine = MLTrainingEngine(self.config)
        loaded_model = new_engine.load_model(filepath)
        
        # Check that the model was loaded correctly
        self.assertIsNotNone(loaded_model)
        self.assertEqual(type(loaded_model), type(model))
        self.assertIn(model_name, new_engine.models)
        self.assertEqual(new_engine.models[model_name]["params"]["max_depth"], 3)

    def test_error_analysis(self):
        """Test error analysis functionality"""
        # Create and train a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        model_name = "test_error_model"
        
        # Store the model
        self.engine.models[model_name] = {
            "name": model_name,
            "model": model,
            "params": {},
            "metrics": {}
        }
        
        # Mock predictions for error analysis
        with patch.object(model, 'predict') as mock_predict:
            # Create some errors in the predictions
            y_pred = self.y_test.copy()
            error_indices = np.random.choice(len(y_pred), 5, replace=False)
            for idx in error_indices:
                y_pred[idx] = 1 - y_pred[idx]  # Flip prediction to create error
            
            mock_predict.return_value = y_pred
            
            # Perform error analysis
            analysis = self.engine.perform_error_analysis(model_name, self.X_test, self.y_test, include_plot=False)
        
        # Check analysis results
        self.assertIn("model_name", analysis)
        self.assertEqual(analysis["model_name"], model_name)
        self.assertIn("error_count", analysis)
        self.assertEqual(analysis["error_count"], 5)
        self.assertIn("error_rate", analysis)
        self.assertAlmostEqual(analysis["error_rate"], 5/len(self.y_test))

    def test_data_drift_detection(self):
        """Test data drift detection"""
        # Create reference data
        reference_data = self.X_train.copy()
        
        # Create new data with drift
        new_data = self.X_train.copy()
        new_data['feature1'] = new_data['feature1'] + 0.5  # Add drift to one feature
        
        # Detect drift
        drift_results = self.engine.detect_data_drift(new_data, reference_data, include_plot=False)
        
        # Check results
        self.assertIn("feature_drift", drift_results)
        self.assertIn("feature1", drift_results["feature_drift"])
        self.assertIn("drift_detected", drift_results)
        self.assertTrue("feature1" in drift_results["drifted_features"])

    def test_comparison_report(self):
        """Test model comparison reporting"""
        # Create two models
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        model1 = DecisionTreeClassifier()
        model2 = RandomForestClassifier()
        
        # Store models
        self.engine.models["model1"] = {
            "model": model1,
            "params": {"max_depth": 3},
            "metrics": {"accuracy": 0.80, "f1": 0.79}
        }
        
        self.engine.models["model2"] = {
            "model": model2,
            "params": {"n_estimators": 10},
            "metrics": {"accuracy": 0.85, "f1": 0.84}
        }
        
        # Set best model
        self.engine.best_model = "model2"
        
        # Generate comparison report
        comparison = self.engine.compare_models(include_plot=False)
        
        # Check results
        self.assertIn("models", comparison)
        self.assertEqual(len(comparison["models"]), 2)
        self.assertIn("metrics", comparison)
        self.assertIn("best_model", comparison)
        self.assertEqual(comparison["best_model"], "model2")

    def test_prediction(self):
        """Test model prediction"""
        # Create and train a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        model_name = "test_predict_model"
        
        # Store the model
        self.engine.models[model_name] = {
            "model": model,
            "params": {},
            "metrics": {}
        }
        
        # Set as best model
        self.engine.best_model = model_name
        
        # Test prediction
        predictions = self.engine.predict(self.X_test, model_name=model_name)
        
        # Check results
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))  # Binary classification

    def test_run_batch_inference(self):
        """Test batch inference"""
        # Create and train a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        model_name = "test_batch_model"
        
        # Store the model
        self.engine.models[model_name] = {
            "model": model,
            "params": {},
            "metrics": {}
        }
        
        # Create batch data
        batch_size = 20
        batches = []
        for i in range(0, len(self.X_test), batch_size):
            batches.append(self.X_test.iloc[i:i+batch_size])
            
        # Run batch inference
        results = self.engine.run_batch_inference(
            batches, 
            model_name=model_name,
            parallel=False  # To avoid threading issues in testing
        )
        
        # Check results
        self.assertEqual(len(results), len(batches))
        for batch_result in results:
            self.assertTrue(np.all((batch_result == 0) | (batch_result == 1)))  # Binary classification
            
    def test_generate_report(self):
        """Test report generation"""
        # Create and store a model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model_name = "report_model"
        
        self.engine.models[model_name] = {
            "model": model,
            "params": {"max_depth": 3},
            "metrics": {"accuracy": 0.85, "f1": 0.84}
        }
        
        # Set as best model
        self.engine.best_model = model_name
        
        # Generate report
        report_path = self.engine.generate_report()
        
        # Check that report was generated
        self.assertIsNotNone(report_path)
        self.assertTrue(os.path.exists(report_path))


class TestExperimentTracker(unittest.TestCase):
    """Test cases for the ExperimentTracker class"""
    
    def setUp(self):
        # Create temporary directory for experiments
        self.test_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(output_dir=self.test_dir)
        
    def tearDown(self):
        # Clean up
        shutil.rmtree(self.test_dir)
        plt.close('all')
        
    def test_initialization(self):
        """Test ExperimentTracker initialization"""
        self.assertEqual(self.tracker.output_dir, self.test_dir)
        self.assertEqual(len(self.tracker.metrics_history), 0)
        self.assertEqual(self.tracker.current_experiment, {})
        
    def test_experiment_lifecycle(self):
        """Test starting, logging, and ending an experiment"""
        # Start experiment
        config = {"test_param": 1}
        model_info = {"model_type": "DecisionTree"}
        self.tracker.start_experiment(config, model_info)
        
        # Check experiment was created
        self.assertIn("experiment_id", self.tracker.current_experiment)
        self.assertIn("config", self.tracker.current_experiment)
        self.assertEqual(self.tracker.current_experiment["config"], config)
        
        # Log metrics
        metrics = {"accuracy": 0.9, "f1": 0.85}
        self.tracker.log_metrics(metrics)
        
        # Check metrics were logged
        self.assertIn("metrics", self.tracker.current_experiment)
        self.assertEqual(self.tracker.current_experiment["metrics"]["accuracy"], 0.9)
        
        # Log feature importance
        feature_names = ["f1", "f2", "f3"]
        importance = np.array([0.5, 0.3, 0.2])
        self.tracker.log_feature_importance(feature_names, importance)
        
        # Check feature importance was logged
        self.assertIn("feature_importance", self.tracker.current_experiment)
        self.assertEqual(self.tracker.current_experiment["feature_importance"]["f1"], 0.5)
        
        # End experiment
        result = self.tracker.end_experiment()
        
        # Check experiment was ended and saved
        self.assertEqual(len(self.tracker.metrics_history), 1)
        self.assertEqual(result["config"], config)
        self.assertEqual(result["metrics"]["accuracy"], 0.9)
        
        # Check experiment file was created
        exp_id = self.tracker.experiment_id
        self.assertTrue(os.path.exists(f"{self.test_dir}/experiment_{exp_id}.json"))


if __name__ == '__main__':
    unittest.main()