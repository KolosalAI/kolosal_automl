import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import joblib
import matplotlib.pyplot as plt
import json
import logging
import time
from sklearn.pipeline import Pipeline

# Import the classes we want to test
from modules.engine.train_engine import MLTrainingEngine, ExperimentTracker
from modules.configs import (
    MLTrainingEngineConfig, 
    TaskType, 
    OptimizationStrategy,
    PreprocessorConfig,
    BatchProcessorConfig,
    InferenceEngineConfig,
    NormalizationType,
    ModelSelectionCriteria,
    MonitoringConfig,
    ExplainabilityConfig
)


class TestMLTrainingEngine(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        
        # Create monitoring and explainability configs
        monitoring_config = MonitoringConfig(
            enable_monitoring=True,
            monitoring_interval=10,
            performance_threshold=0.75,
            drift_detection_method="ks_test"
        )
        
        explainability_config = ExplainabilityConfig(
            enable_explainability=True,
            default_method="shap",
            store_explanations=True
        )
        
        preprocessor_config = PreprocessorConfig(
            handle_missing_values=True,
            normalization_type=NormalizationType.STANDARD,
            one_hot_encode_categoricals=True,
            text_vectorization_method="tfidf"
        )
        
        batch_config = BatchProcessorConfig(
            batch_size=32,
            use_parallel_processing=True,
            max_workers=2
        )
        
        # Create inference config
        inference_config = InferenceEngineConfig(
            batch_size=64,
            prediction_timeout=30,
            enable_caching=True
        )

        # Create a comprehensive configuration for testing
        self.config = MLTrainingEngineConfig(
            model_path=self.test_dir,
            checkpoint_path=self.checkpoint_dir,
            task_type=TaskType.CLASSIFICATION,
            feature_selection=True,
            feature_selection_method="mutual_info",
            feature_selection_k=5,
            cv_folds=2,
            test_size=0.2,
            random_state=42,
            optimization_strategy=OptimizationStrategy.GRID_SEARCH,
            optimization_iterations=2,
            experiment_tracking=True,
            experiment_tracking_platform="local",
            experiment_tracking_config={"local_dir": os.path.join(self.test_dir, "experiments")},
            auto_save=True,
            checkpointing=True,
            model_selection_criteria=ModelSelectionCriteria.F1,
            preprocessing_config=preprocessor_config,
            batch_processing_config=batch_config,
            inference_config=inference_config,
            monitoring_config=monitoring_config,
            explainability_config=explainability_config,
            compute_permutation_importance=True,
            verbose=0,
            n_jobs=1,
            debug_mode=True,
            use_gpu=False,
            log_level="INFO"
        )
        
        # Mock InferenceEngine to avoid initialization issues
        with patch('modules.engine.inference_engine.InferenceEngine') as MockInferenceEngine:
            # Configure mock to return a dummy inference engine
            mock_engine = MockInferenceEngine.return_value
            
            # Create the training engine
            self.engine = MLTrainingEngine(self.config)
            
            # Replace the actual inference engine with our mock
            self.engine.inference_engine = mock_engine
        
        # Create some dummy data for testing
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'categorical1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical2': np.random.choice(['X', 'Y', 'Z'], 100),
            'text_feature': [f"This is sample text {i}" for i in range(100)]
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
        
        # Test version attribute exists
        self.assertTrue(hasattr(self.engine, 'VERSION'))
        self.assertIsNotNone(self.engine.VERSION)
        
        # Test model registry initialization
        self.assertTrue(hasattr(self.engine, '_model_registry'))
        self.assertIn("classification", self.engine._model_registry)
        self.assertIn("regression", self.engine._model_registry)
        
        # Test components initialization
        self.assertIsNotNone(self.engine.preprocessor)
        self.assertIsNotNone(self.engine.batch_processor)
        self.assertIsNotNone(self.engine.inference_engine)

    def test_train_model_standard(self):
        """Test training a basic model with standard optimization"""
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

    def test_train_model_asht(self):
        """Test training a model with ASHT optimization strategy"""
        # Update config to use ASHT
        self.engine.config.optimization_strategy = OptimizationStrategy.ASHT
        
        # Create a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model_name = "test_asht_model"
        param_grid = {"max_depth": [2, 3, 4]}
        
        # Mock ASHTOptimizer
        with patch('modules.optimizer.asht.ASHTOptimizer') as MockASHT:
            # Configure the mock
            mock_optimizer = MockASHT.return_value
            mock_optimizer.best_estimator_ = model
            mock_optimizer.best_params_ = {"max_depth": 3}
            mock_optimizer.best_score_ = 0.9
            
            # Mock fit method
            mock_optimizer.fit = Mock(return_value=mock_optimizer)
            
            # Mock results attributes
            mock_optimizer.cv_results_ = {
                'mean_test_score': [0.9],
                'std_test_score': [0.05]
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

    def test_train_model_hyperx(self):
        """Test training a model with HyperOptX optimization strategy"""
        # Update config to use HyperOptX
        self.engine.config.optimization_strategy = OptimizationStrategy.HYPERX
        
        # Create a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model_name = "test_hyperx_model"
        param_grid = {"max_depth": [2, 3, 4]}
        
        # Mock HyperOptX
        with patch('modules.optimizer.hyperoptx.HyperOptX') as MockHyperOptX:
            # Configure the mock
            mock_optimizer = MockHyperOptX.return_value
            mock_optimizer.best_estimator_ = model
            mock_optimizer.best_params_ = {"max_depth": 4}
            mock_optimizer.best_score_ = 0.92
            
            # Mock fit method
            mock_optimizer.fit = Mock(return_value=mock_optimizer)
            
            # Mock results attributes
            mock_optimizer.cv_results_ = {
                'mean_test_score': [0.92],
                'std_test_score': [0.03]
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
        self.assertEqual(self.engine.models[model_name]["params"]["max_depth"], 4)

    def test_feature_selection(self):
        """Test feature selection functionality"""
        # Mock feature selector
        with patch('sklearn.feature_selection.SelectKBest') as MockSelector:
            # Configure the mock
            mock_selector = MockSelector.return_value
            mock_selector.fit = Mock(return_value=mock_selector)
            mock_selector.transform = Mock(return_value=np.random.rand(80, 5))
            mock_selector.get_support = Mock(return_value=np.array([True, True, False, True, False, True]))
            
            # Get feature selector
            selector = self.engine._get_feature_selector(self.X_train, self.y_train)
            
            # Apply it to data
            selector.fit(self.X_train, self.y_train)
            X_transformed = selector.transform(self.X_train)
            
        # Check that feature selection was applied
        self.assertIsNotNone(X_transformed)
        MockSelector.assert_called_once()

    def test_model_pipeline(self):
        """Test creating a pipeline with preprocessor, feature selector, and model"""
        # Create components
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        
        # Mock preprocessor and feature selector
        self.engine.preprocessor = Mock()
        self.engine.feature_selector = Mock()
        
        # Create pipeline
        pipeline = self.engine._create_pipeline(model)
        
        # Check pipeline structure
        self.assertEqual(len(pipeline.steps), 3)
        self.assertEqual(pipeline.steps[0][0], 'preprocessor')
        self.assertEqual(pipeline.steps[1][0], 'feature_selector')
        self.assertEqual(pipeline.steps[2][0], 'model')
        self.assertEqual(pipeline.steps[2][1], model)

    def test_evaluate_model_classification(self):
        """Test detailed model evaluation for classification"""
        # Create and train a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        model_name = "test_eval_classification"
        
        # Store the model
        self.engine.models[model_name] = {
            "name": model_name,
            "model": model,
            "params": {},
            "metrics": {}
        }
        
        # Cache test data
        self.engine._last_X_test = self.X_test
        self.engine._last_y_test = self.y_test
        
        # Evaluate model with detailed=True
        metrics = self.engine.evaluate_model(model_name, detailed=True)
        
        # Check metrics
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Check detailed metrics
        self.assertIn("detailed_report", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("per_class", metrics)

    def test_evaluate_model_regression(self):
        """Test model evaluation for regression"""
        # Change task type to regression
        self.engine.config.task_type = TaskType.REGRESSION
        
        # Create regression targets
        y_regression = np.random.rand(100) * 10
        from sklearn.model_selection import train_test_split
        _, _, y_train_reg, y_test_reg = train_test_split(
            self.X, y_regression, test_size=0.2, random_state=42
        )
        
        # Create and train a regression model
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=42)
        model.fit(self.X_train, y_train_reg)
        model_name = "test_eval_regression"
        
        # Store the model
        self.engine.models[model_name] = {
            "name": model_name,
            "model": model,
            "params": {},
            "metrics": {}
        }
        
        # Evaluate model
        metrics = self.engine.evaluate_model(model_name, self.X_test, y_test_reg)
        
        # Check regression metrics
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)

    def test_feature_importance_methods(self):
        """Test different methods of extracting feature importance"""
        # Test with model having feature_importances_
        from sklearn.ensemble import RandomForestClassifier
        model1 = RandomForestClassifier(n_estimators=5, random_state=42)
        model1.fit(self.X_train.select_dtypes(include=np.number), self.y_train)
        importances1 = np.array([0.2, 0.3, 0.5])
        model1.feature_importances_ = importances1
        
        # Extract importance
        extracted1 = self.engine._get_feature_importance(model1)
        self.assertIsNotNone(extracted1)
        self.assertEqual(len(extracted1), len(importances1))
        
        # Test with model having coef_
        from sklearn.linear_model import LogisticRegression
        model2 = LogisticRegression(random_state=42)
        X_numeric = self.X_train.select_dtypes(include=np.number)
        model2.fit(X_numeric, self.y_train)
        
        # Extract importance
        extracted2 = self.engine._get_feature_importance(model2)
        self.assertIsNotNone(extracted2)
        self.assertEqual(len(extracted2), X_numeric.shape[1])
        
        # Test permutation importance
        with patch('sklearn.inspection.permutation_importance') as MockPermutation:
            # Configure mock
            mock_result = Mock()
            mock_result.importances_mean = np.array([0.1, 0.2, 0.3])
            MockPermutation.return_value = mock_result
            
            # Cache training data
            self.engine._last_X_train = X_numeric
            self.engine._last_y_train = self.y_train
            
            # Create model without built-in importance
            model3 = Mock()
            model3.predict = Mock()
            
            # Extract importance
            extracted3 = self.engine._get_feature_importance(model3)
            self.assertIsNotNone(extracted3)
            self.assertEqual(len(extracted3), 3)

    def test_save_load_model(self):
        """Test saving and loading a model"""
        # Create and train a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        model_name = "test_save_model"
        
        # Store the model
        self.engine.models[model_name] = {
            "name": model_name,
            "model": model,
            "params": {"max_depth": 3},
            "metrics": {"accuracy": 0.85},
            "created_at": "2023-01-01"
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

    def test_checkpointing(self):
        """Test model checkpointing functionality"""
        # Ensure checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # Create model data
        model_data = {
            "model": Mock(),
            "params": {"max_depth": 3},
            "metrics": {"accuracy": 0.85}
        }
        
        # Mock joblib.dump
        with patch('joblib.dump') as mock_dump:
            # Create checkpoint
            checkpoint_path = self.engine._create_checkpoint(
                "test_checkpoint", 
                model_data, 
                epoch=5, 
                score=0.85
            )
            
            # Check that dump was called
            mock_dump.assert_called_once()
            
            # Check checkpoint path format
            self.assertTrue("test_checkpoint" in checkpoint_path)
            self.assertTrue("epoch_5" in checkpoint_path)
            self.assertTrue("score_0.85" in checkpoint_path)

    def test_detect_data_drift(self):
        """Test data drift detection with various methods"""
        # Create reference and new data
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        
        # New data with drift in feature1
        new_data = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1.2, 100),  # Drift
            'feature2': np.random.normal(0, 1, 100)       # No drift
        })
        
        # Test drift detection with KS test
        drift_results_ks = self.engine.detect_data_drift(
            new_data, 
            reference_data,
            method="ks_test",
            include_plot=False
        )
        
        # Check results
        self.assertIn("feature_drift", drift_results_ks)
        self.assertIn("feature1", drift_results_ks["feature_drift"])
        self.assertIn("drift_detected", drift_results_ks)
        
        # Test drift detection with Jensen-Shannon divergence
        drift_results_js = self.engine.detect_data_drift(
            new_data, 
            reference_data,
            method="js_divergence",
            include_plot=False
        )
        
        # Check results
        self.assertIn("feature_drift", drift_results_js)
        self.assertIn("drift_detected", drift_results_js)

    def test_error_analysis(self):
        """Test comprehensive error analysis"""
        # Create and train a simple model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        model_name = "test_error_model"
        
        # Store the model
        self.engine.models[model_name] = {
            "name": model_name,
            "model": model,
            "params": {},
            "metrics": {}
        }
        
        # Test error analysis with segmentation
        analysis = self.engine.perform_error_analysis(
            model_name, 
            self.X_test, 
            self.y_test,
            segment_by=['categorical1'],
            include_plot=False
        )
        
        # Check analysis results
        self.assertIn("error_count", analysis)
        self.assertIn("error_rate", analysis)
        self.assertIn("segments", analysis)
        
        # Ensure segments were created for each category
        self.assertTrue(any('A' in str(seg) for seg in analysis["segments"]))
        self.assertTrue(any('B' in str(seg) for seg in analysis["segments"]))
        self.assertTrue(any('C' in str(seg) for seg in analysis["segments"]))

    def test_batch_processing(self):
        """Test batch processing with parallel execution"""
        # Create batches
        batch_size = 20
        num_batches = 5
        batches = [self.X.iloc[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
        
        # Mock batch processor
        with patch.object(self.engine.batch_processor, 'process_batch') as mock_process:
            # Configure mock
            mock_process.side_effect = lambda batch, **kwargs: np.random.rand(len(batch))
            
            # Process batches
            results = self.engine.run_batch_inference(
                batches, 
                model_name=None,  # Use best model
                parallel=True,
                batch_args={'additional_arg': True}
            )
            
        # Check results
        self.assertEqual(len(results), num_batches)
        for batch_result in results:
            self.assertEqual(len(batch_result), batch_size)

    def test_run_optimization(self):
        """Test running optimization without training for various strategies"""
        # Create sample models and parameters
        models = {
            "model1": {
                "model": Mock(),
                "params": {"param1": [1, 2], "param2": [3, 4]}
            },
            "model2": {
                "model": Mock(),
                "params": {"param3": [5, 6], "param4": [7, 8]}
            }
        }
        
        # Test with different optimization strategies
        for strategy in [
            OptimizationStrategy.GRID_SEARCH,
            OptimizationStrategy.RANDOM_SEARCH,
            OptimizationStrategy.BAYESIAN_OPTIMIZATION,
            OptimizationStrategy.ASHT,
            OptimizationStrategy.HYPERX
        ]:
            # Update config
            self.engine.config.optimization_strategy = strategy
            
            # Mock appropriate optimizer
            optimizer_class = f"modules.optimizer.{strategy.value.lower()}.{strategy.value.title()}Optimizer" if strategy in [OptimizationStrategy.ASHT, OptimizationStrategy.HYPERX] else f"sklearn.model_selection.{strategy.value.title()}SearchCV"
            
            with patch(optimizer_class.replace('.', '.')) as MockOptimizer:
                # Configure mock
                mock_opt = MockOptimizer.return_value
                mock_opt.fit = Mock(return_value=mock_opt)
                mock_opt.best_estimator_ = Mock()
                mock_opt.best_params_ = {"best_param": "value"}
                mock_opt.best_score_ = 0.9
                
                # Run optimization
                best_model = self.engine.run_optimization(
                    models,
                    self.X_train,
                    self.y_train
                )
                
                # Check results
                self.assertIsNotNone(best_model)
                self.assertIn("model", best_model)
                self.assertIn("params", best_model)
                self.assertIn("score", best_model)

    def test_generate_report(self):
        """Test comprehensive report generation"""
        # Create and store multiple models
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        model1 = DecisionTreeClassifier(random_state=42)
        model2 = RandomForestClassifier(random_state=42)
        
        self.engine.models["model1"] = {
            "model": model1,
            "params": {"max_depth": 3},
            "metrics": {"accuracy": 0.82, "f1": 0.81, "precision": 0.80},
            "feature_importance": {"feature1": 0.5, "feature2": 0.3, "feature3": 0.2}
        }
        
        self.engine.models["model2"] = {
            "model": model2,
            "params": {"n_estimators": 10, "max_depth": 5},
            "metrics": {"accuracy": 0.88, "f1": 0.87, "precision": 0.86},
            "feature_importance": {"feature1": 0.4, "feature2": 0.4, "feature3": 0.2}
        }
        
        # Set best model
        self.engine.best_model = "model2"
        
        # Generate comprehensive report
        report_path = self.engine.generate_report()
        
        # Check report exists
        self.assertTrue(os.path.exists(report_path))
        
        # Read report content
        with open(report_path, 'r') as f:
            report_content = f.read()
            
        # Check key sections are in the report
        self.assertIn("# ML Training Engine Report", report_content)
        self.assertIn("## Configuration", report_content)
        self.assertIn("## Model Performance Summary", report_content)
        self.assertIn("## Model Details", report_content)
        self.assertIn("### model1", report_content)
        self.assertIn("### model2", report_content)
        self.assertIn("## Conclusion", report_content)
        self.assertIn("**model2** **[BEST]**", report_content)

    def test_explainability(self):
        """Test explainability functionality"""
        # Create and train a simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(self.X_train.select_dtypes(include=np.number), self.y_train)
        model_name = "explainable_model"
        
        # Store the model
        self.engine.models[model_name] = {
            "name": model_name,
            "model": model,
            "params": {},
            "metrics": {}
        }
        
        # Test with SHAP explainer
        with patch('shap.Explainer') as MockExplainer:
            # Configure mock
            mock_explainer = MockExplainer.return_value
            mock_values = Mock()
            mock_values.values = np.random.rand(20, 3)  # 20 samples, 3 features
            mock_explainer.shap_values = Mock(return_value=mock_values)
            
            # Generate explanations
            explanations = self.engine.explain_predictions(
                model_name,
                self.X_test.select_dtypes(include=np.number).iloc[:20],
                method="shap"
            )
            
            # Check explanations
            self.assertIsNotNone(explanations)
            self.assertIn("method", explanations)
            self.assertEqual(explanations["method"], "shap")
            self.assertIn("explanations", explanations)
            self.assertIn("global_importance", explanations)

    def test_model_comparison_and_selection(self):
        """Test comparing multiple models and selecting the best"""
        # Create and store multiple models with metrics
        self.engine.models = {
            "model1": {
                "model": Mock(),
                "params": {"param1": 1},
                "metrics": {"accuracy": 0.8, "f1": 0.75}
            },
            "model2": {
                "model": Mock(),
                "params": {"param1": 2},
                "metrics": {"accuracy": 0.85, "f1": 0.83}
            },
            "model3": {
                "model": Mock(),
                "params": {"param1": 3},
                "metrics": {"accuracy": 0.82, "f1": 0.9}  # Highest F1
            }
        }
        
        # Set criteria to F1
        self.engine.config.model_selection_criteria = ModelSelectionCriteria.F1
        
        # Find best model
        best_model = self.engine._find_best_model()
        
        # Check best model is the one with highest F1
        self.assertEqual(best_model, "model3")
        
        # Change criteria to accuracy
        self.engine.config.model_selection_criteria = ModelSelectionCriteria.ACCURACY
        
        # Find best model with new criteria
        best_model = self.engine._find_best_model()
        
        # Check best model is now the one with highest accuracy
        self.assertEqual(best_model, "model2")

    def test_ensemble_creation(self):
        """Test creating ensemble models"""
        # Create multiple base models
        base_models = {
            "model1": {
                "model": Mock(),
                "metrics": {"accuracy": 0.82}
            },
            "model2": {
                "model": Mock(),
                "metrics": {"accuracy": 0.85}
            },
            "model3": {
                "model": Mock(),
                "metrics": {"accuracy": 0.80}
            }
        }
        
        # Mock VotingClassifier or VotingRegressor
        with patch('sklearn.ensemble.VotingClassifier') as MockVoting:
            # Configure mock
            mock_ensemble = MockVoting.return_value
            mock_ensemble.fit = Mock(return_value=mock_ensemble)
            
            # Create ensemble
            ensemble_name = "test_ensemble"
            ensemble_model = self.engine.create_ensemble(
                base_models=list(base_models.keys()),
                ensemble_name=ensemble_name,
                ensemble_type="voting",
                weights=[1, 2, 1],  # Weight second model higher
                X_train=self.X_train,
                y_train=self.y_train
            )
            
            # Check ensemble was created
            self.assertIsNotNone(ensemble_model)
            self.assertIn(ensemble_name, self.engine.models)
            self.assertEqual(self.engine.models[ensemble_name]["model"], mock_ensemble)
            self.assertEqual(self.engine.models[ensemble_name]["ensemble_type"], "voting")
            
            # Verify correct models were passed to ensemble
            args, kwargs = MockVoting.call_args
            self.assertEqual(len(kwargs.get("estimators", [])), 3)

    def test_cross_validation_evaluation(self):
        """Test cross-validation evaluation"""
        # Create model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        model_name = "cv_model"
        
        # Store model
        self.engine.models[model_name] = {
            "model": model,
            "params": {},
            "metrics": {}
        }
        
        # Mock cross_val_score
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            # Configure mock
            mock_cv.return_value = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
            
            # Run CV evaluation
            cv_results = self.engine.evaluate_with_cross_validation(
                model_name,
                self.X_train,
                self.y_train,
                cv=5,
                scoring="accuracy"
            )
            
            # Check results
            self.assertIn("cv_scores", cv_results)
            self.assertEqual(len(cv_results["cv_scores"]), 5)
            self.assertIn("mean_score", cv_results)
            self.assertIn("std_score", cv_results)
            self.assertAlmostEqual(cv_results["mean_score"], 0.85, places=2)

    def test_learning_curve_analysis(self):
        """Test learning curve analysis"""
        # Create model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        model_name = "learning_curve_model"
        
        # Store model
        self.engine.models[model_name] = {
            "model": model,
            "params": {},
            "metrics": {}
        }
        
        # Mock learning_curve
        with patch('sklearn.model_selection.learning_curve') as mock_lc:
            # Configure mock
            mock_lc.return_value = (
                np.array([0.2, 0.4, 0.6, 0.8, 1.0]),  # Train sizes
                [np.array([0.95, 0.93, 0.92, 0.91, 0.90])],  # Train scores for each CV split
                [np.array([0.80, 0.82, 0.84, 0.85, 0.86])]   # Test scores for each CV split
            )
            
            # Generate learning curve
            lc_results = self.engine.generate_learning_curve(
                model_name,
                self.X_train,
                self.y_train,
                train_sizes=np.linspace(0.2, 1.0, 5),
                cv=2,
                include_plot=False
            )
            
            # Check results
            self.assertIn("train_sizes", lc_results)
            self.assertIn("train_scores", lc_results)
            self.assertIn("test_scores", lc_results)
            self.assertIn("mean_train_scores", lc_results)
            self.assertIn("mean_test_scores", lc_results)
            
            # Check for overfitting analysis
            self.assertIn("gap_analysis", lc_results)
            self.assertIn("overfitting_risk", lc_results)

    def test_feature_importance_plot(self):
        """Test feature importance visualization"""
        # Create model with feature importance
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model_name = "feature_imp_model"
        
        # Mock fitting
        with patch.object(model, 'fit') as mock_fit:
            mock_fit.return_value = model
            model.feature_importances_ = np.array([0.3, 0.2, 0.5])
            model.feature_names_in_ = np.array(['feature1', 'feature2', 'feature3'])
            
            # Store model
            self.engine.models[model_name] = {
                "model": model,
                "params": {},
                "metrics": {},
                "feature_importance": {
                    'feature1': 0.3,
                    'feature2': 0.2,
                    'feature3': 0.5
                },
                "feature_names": ['feature1', 'feature2', 'feature3']
            }
            
            # Generate feature importance plot
            with patch('matplotlib.pyplot.savefig') as mock_save:
                plot_path = self.engine.plot_feature_importance(
                    model_name,
                    top_n=3,
                    output_path=os.path.join(self.test_dir, "feat_imp.png")
                )
                
                # Check plot was saved
                self.assertIsNotNone(plot_path)
                mock_save.assert_called_once()

    def test_model_monitoring(self):
        """Test model monitoring functionality"""
        # Setup monitoring
        model_name = "monitored_model"
        self.engine.models[model_name] = {
            "model": Mock(),
            "metrics": {"accuracy": 0.85},
            "created_at": time.time()
        }
        
        # Setup benchmark data
        benchmark_data = {
            "X": self.X_test,
            "y": self.y_test,
            "metrics": {"accuracy": 0.85}
        }
        
        # Register model for monitoring
        self.engine.register_model_for_monitoring(
            model_name,
            benchmark_data,
            monitoring_interval=24,  # hours
            alert_threshold=0.05     # 5% degradation
        )
        
        # Check model is registered
        self.assertIn(model_name, getattr(self.engine, "_monitored_models", {}))
        
        # Test monitoring check
        with patch.object(self.engine, 'evaluate_model') as mock_eval:
            # Configure mock to show degradation
            mock_eval.return_value = {"accuracy": 0.79}  # 6% drop
            
            # Run monitoring check
            monitoring_results = self.engine.check_model_health(model_name)
            
            # Check results indicate degradation
            self.assertIn("status", monitoring_results)
            self.assertEqual(monitoring_results["status"], "degraded")
            self.assertIn("metric_changes", monitoring_results)
            self.assertAlmostEqual(monitoring_results["metric_changes"]["accuracy"], -0.06, places=2)
            self.assertIn("alerts", monitoring_results)
            self.assertTrue(monitoring_results["alerts"]["accuracy_threshold_alert"])

    def test_pipeline_optimization(self):
        """Test optimizing a full preprocessing and model pipeline"""
        # Define pipeline components and parameters
        pipeline_components = {
            "preprocessor": {
                "imputer": ["simple", "knn", "iterative"],
                "scaler": ["standard", "minmax", None]
            },
            "feature_selection": {
                "method": ["mutual_info", "chi2", None],
                "k": [3, 5, "all"]
            },
            "model": {
                "type": ["decision_tree", "random_forest"],
                "params": {
                    "decision_tree": {
                        "max_depth": [3, 5, 7]
                    },
                    "random_forest": {
                        "n_estimators": [10, 50],
                        "max_depth": [3, 5]
                    }
                }
            }
        }
        
        # Mock optimizer classes
        with patch('sklearn.model_selection.RandomizedSearchCV') as mock_random_search:
            # Configure mock
            mock_optimizer = mock_random_search.return_value
            mock_optimizer.fit = Mock(return_value=mock_optimizer)
            mock_optimizer.best_estimator_ = Pipeline([('preprocessor', Mock()), ('model', Mock())])
            mock_optimizer.best_params_ = {
                'preprocessor__imputer': 'knn',
                'preprocessor__scaler': 'standard',
                'feature_selection__method': 'mutual_info',
                'feature_selection__k': 5,
                'model__max_depth': 5,
                'model__type': 'random_forest',
                'model__n_estimators': 50
            }
            mock_optimizer.best_score_ = 0.88
            
            # Optimize pipeline
            pipeline_results = self.engine.optimize_pipeline(
                pipeline_components,
                self.X_train,
                self.y_train,
                n_iter=10,
                cv=3
            )
            
            # Check results
            self.assertIn("best_pipeline", pipeline_results)
            self.assertIn("best_params", pipeline_results)
            self.assertIn("best_score", pipeline_results)
            self.assertEqual(pipeline_results["best_score"], 0.88)
            self.assertEqual(
                pipeline_results["best_params"]["model__type"], 
                "random_forest"
            )

    def test_auto_ml_end_to_end(self):
        """Test end-to-end automated ML workflow"""
        # Mock model registry
        mock_registry = {
            "classification": {
                "decision_tree": Mock(),
                "random_forest": Mock()
            }
        }
        self.engine._model_registry = mock_registry
        
        # Mock all the training functions
        with patch.object(self.engine, 'train_model') as mock_train:
            # Configure mocks
            mock_train.side_effect = lambda model, name, params, X_train, y_train, X_test=None, y_test=None: (
                model, {"accuracy": 0.8 + (0.05 * len(name))}  # Make later models slightly better
            )
            
            # Run auto ML
            with patch.object(self.engine, '_get_feature_selector'):
                with patch.object(self.engine, 'preprocessor'):
                    auto_ml_results = self.engine.auto_ml(
                        self.X_train,
                        self.y_train,
                        self.X_test,
                        self.y_test,
                        models=["decision_tree", "random_forest"],
                        time_budget=10,  # seconds
                        optimization_level="light"  # Less intensive for test
                    )
            
            # Check results
            self.assertIn("best_model", auto_ml_results)
            self.assertIn("leaderboard", auto_ml_results)
            self.assertEqual(len(auto_ml_results["leaderboard"]), 2)
            
            # Verify the calls to train_model
            self.assertEqual(mock_train.call_count, 2)

    def test_time_series_support(self):
        """Test time series specific functionality"""
        # Change task type to time series
        self.engine.config.task_type = TaskType.TIME_SERIES
        
        # Create time series data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        ts_data = pd.DataFrame({
            'date': dates,
            'value': np.sin(np.arange(100) * 0.1) + np.random.normal(0, 0.1, 100)
        })
        ts_data.set_index('date', inplace=True)
        
        # Mock time series cross validator
        with patch('sklearn.model_selection.TimeSeriesSplit') as mock_ts_cv:
            # Configure mock
            mock_splitter = mock_ts_cv.return_value
            mock_splitter.split = Mock(return_value=[
                (np.arange(0, 70), np.arange(70, 80)),
                (np.arange(0, 80), np.arange(80, 90)),
                (np.arange(0, 90), np.arange(90, 100))
            ])
            
            # Test time series CV
            ts_cv = self.engine._get_cv_splitter(ts_data['value'])
            
            # Verify TimeSeriesSplit was used
            mock_ts_cv.assert_called_once()
            
        # Test time series feature engineering
        with patch.object(self.engine, '_create_time_features') as mock_time_features:
            # Configure mock to add features
            mock_time_features.return_value = pd.DataFrame({
                'value': ts_data['value'],
                'lag_1': ts_data['value'].shift(1),
                'lag_7': ts_data['value'].shift(7),
                'rolling_mean_7': ts_data['value'].rolling(7).mean()
            })
            
            # Generate features
            ts_features = self.engine.prepare_time_series_features(
                ts_data,
                target_col='value',
                lags=[1, 7],
                rolling_windows=[7],
                rolling_funcs=['mean']
            )
            
            # Check features
            self.assertIsNotNone(ts_features)
            mock_time_features.assert_called_once()


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
        self.assertIsNotNone(self.tracker.logger)
        
    def test_experiment_lifecycle(self):
        """Test starting, logging, and ending an experiment"""
        # Start experiment
        config = {"test_param": 1, "model_type": "RandomForest"}
        model_info = {"model_type": "RandomForest", "parameters": {"n_estimators": 100}}
        self.tracker.start_experiment(config, model_info)
        
        # Check experiment was created
        self.assertIn("experiment_id", self.tracker.current_experiment)
        self.assertIn("config", self.tracker.current_experiment)
        self.assertEqual(self.tracker.current_experiment["config"], config)
        
        # Log metrics
        metrics = {"accuracy": 0.92, "f1": 0.90, "precision": 0.89, "recall": 0.91}
        self.tracker.log_metrics(metrics)
        
        # Log step metrics
        cv_metrics = {
            "fold_1": 0.91,
            "fold_2": 0.93,
            "fold_3": 0.90,
            "mean": 0.913,
            "std": 0.015
        }
        self.tracker.log_metrics(cv_metrics, step="cross_validation")
        
        # Check metrics were logged
        self.assertIn("metrics", self.tracker.current_experiment)
        self.assertEqual(self.tracker.current_experiment["metrics"]["accuracy"], 0.92)
        self.assertIn("steps", self.tracker.current_experiment)
        self.assertIn("cross_validation", self.tracker.current_experiment["steps"])
        
        # Log feature importance
        feature_names = ["feature1", "feature2", "feature3", "feature4"]
        importance = np.array([0.4, 0.3, 0.2, 0.1])
        self.tracker.log_feature_importance(feature_names, importance)
        
        # Check feature importance was logged
        self.assertIn("feature_importance", self.tracker.current_experiment)
        self.assertEqual(self.tracker.current_experiment["feature_importance"]["feature1"], 0.4)
        
        # End experiment
        result = self.tracker.end_experiment()
        
        # Check experiment was ended and saved
        self.assertEqual(len(self.tracker.metrics_history), 1)
        self.assertEqual(result["config"], config)
        self.assertEqual(result["metrics"]["accuracy"], 0.92)
        
        # Check experiment file was created
        exp_id = self.tracker.experiment_id
        experiment_file = f"{self.test_dir}/experiment_{exp_id}.json"
        self.assertTrue(os.path.exists(experiment_file))
        
        # Verify JSON content
        with open(experiment_file, 'r') as f:
            saved_experiment = json.load(f)
            
        self.assertEqual(saved_experiment["config"], config)
        self.assertEqual(saved_experiment["metrics"]["accuracy"], 0.92)
        self.assertEqual(saved_experiment["feature_importance"]["feature1"], 0.4)
        
    def test_generate_plots(self):
        """Test plot generation"""
        # Setup experiment data
        self.tracker.current_experiment = {
            "experiment_id": self.tracker.experiment_id,
            "feature_importance": {
                "feature1": 0.4,
                "feature2": 0.3,
                "feature3": 0.2,
                "feature4": 0.1
            },
            "steps": {
                "cv": {
                    "fold_1": 0.91,
                    "fold_2": 0.89,
                    "fold_3": 0.93,
                }
            }
        }
        
        # Mock plt.savefig to prevent actual file creation
        with patch('matplotlib.pyplot.savefig') as mock_save:
            # Generate plots
            self.tracker._generate_plots()
            
            # Check that savefig was called twice (once for each plot)
            self.assertEqual(mock_save.call_count, 2)
            
    def test_multiple_experiments(self):
        """Test tracking multiple experiments"""
        # Run first experiment
        self.tracker.start_experiment({"exp": 1}, {"model": "RF"})
        self.tracker.log_metrics({"accuracy": 0.9})
        self.tracker.end_experiment()
        
        # Run second experiment
        self.tracker.start_experiment({"exp": 2}, {"model": "SVM"})
        self.tracker.log_metrics({"accuracy": 0.85})
        self.tracker.end_experiment()
        
        # Check both experiments are tracked
        self.assertEqual(len(self.tracker.metrics_history), 2)
        self.assertEqual(self.tracker.metrics_history[0]["config"]["exp"], 1)
        self.assertEqual(self.tracker.metrics_history[1]["config"]["exp"], 2)
        
        # Check both experiment files exist
        exp_id_1 = self.tracker.metrics_history[0]["experiment_id"]
        exp_id_2 = self.tracker.metrics_history[1]["experiment_id"]
        
        self.assertTrue(os.path.exists(f"{self.test_dir}/experiment_{exp_id_1}.json"))
        self.assertTrue(os.path.exists(f"{self.test_dir}/experiment_{exp_id_2}.json"))

    def test_generate_report(self):
        """Test report generation functionality"""
        # Mock models data for report generation
        self.tracker.models = {
            "model1": {
                "model": Mock(),
                "metrics": {"accuracy": 0.9, "f1": 0.89},
                "params": {"max_depth": 5},
                "feature_importance": {"f1": 0.5, "f2": 0.3, "f3": 0.2}
            },
            "model2": {
                "model": Mock(),
                "metrics": {"accuracy": 0.92, "f1": 0.91},
                "params": {"n_estimators": 100},
                "feature_importance": {"f1": 0.4, "f2": 0.4, "f3": 0.2}
            }
        }
        
        self.tracker.best_model = "model2"
        self.tracker.config = Mock()
        self.tracker.config.model_path = self.test_dir
        self.tracker.config.task_type = TaskType.CLASSIFICATION
        self.tracker.config.to_dict = Mock(return_value={"task_type": "classification"})
        
        # Generate report
        report_path = self.tracker.generate_report()
        
        # Check report exists
        self.assertTrue(os.path.exists(report_path))
        
        # Read report content
        with open(report_path, 'r') as f:
            report_content = f.read()
            
        # Check key sections
        self.assertIn("# ML Training Engine Report", report_content)
        self.assertIn("## Model Performance Summary", report_content)
        self.assertIn("model2 **[BEST]**", report_content)
        self.assertIn("### model1", report_content)
        self.assertIn("### model2", report_content)


if __name__ == '__main__':
    unittest.main()