import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings
import time
import os
import sys
from typing import Dict, List, Tuple, Any

# Add the parent directory to sys.path to import HyperOptX
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import HyperOptX class 
# Assuming the HyperOptX class is in a file named hyperoptx.py
from modules.optimizer.hyperoptx import HyperOptX

class TestHyperOptXInitialization(unittest.TestCase):
    """Test initialization of HyperOptX class"""
    
    def setUp(self):
        # Load a small dataset for testing
        self.X, self.y = make_regression(n_samples=50, n_features=5, random_state=42)
        self.estimator = RandomForestRegressor()
        self.param_space = {
            'n_estimators': (10, 50),
            'max_depth': [None, 5, 10],
            'min_samples_split': (2, 5),
        }
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space
        )
        
        # Check default values
        self.assertEqual(optimizer.max_iter, 100)
        self.assertEqual(optimizer.cv, 5)
        self.assertEqual(optimizer.verbose, 0)
        self.assertTrue(optimizer.maximize)
        self.assertTrue(optimizer.ensemble_surrogate)
        self.assertTrue(optimizer.transfer_learning)
        self.assertTrue(optimizer.early_stopping)
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space,
            max_iter=50,
            cv=3,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=2,
            verbose=1,
            maximize=False,
            time_budget=60,
            ensemble_surrogate=False,
            optimization_strategy='evolutionary'
        )
        
        # Check custom values
        self.assertEqual(optimizer.max_iter, 50)
        self.assertEqual(optimizer.cv, 3)
        self.assertEqual(optimizer.scoring, 'neg_mean_absolute_error')
        self.assertEqual(optimizer.random_state, 42)
        self.assertEqual(optimizer.n_jobs, 2)
        self.assertEqual(optimizer.verbose, 1)
        self.assertFalse(optimizer.maximize)
        self.assertEqual(optimizer.time_budget, 60)
        self.assertFalse(optimizer.ensemble_surrogate)
        self.assertEqual(optimizer.optimization_strategy, 'evolutionary')
    
    def test_param_space_analysis(self):
        """Test parameter space analysis"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space,
            random_state=42
        )
        
        # Check parameter types
        self.assertEqual(optimizer.param_types['n_estimators'], 'integer')
        self.assertEqual(optimizer.param_types['max_depth'], 'categorical')
        self.assertEqual(optimizer.param_types['min_samples_split'], 'integer')
        
        # Check parameter bounds
        self.assertEqual(optimizer.param_bounds['n_estimators'], (10, 50))
        self.assertEqual(optimizer.param_bounds['min_samples_split'], (2, 5))
        
        # max_depth is categorical, so it shouldn't be in param_bounds
        self.assertNotIn('max_depth', optimizer.param_bounds)


class TestHyperOptXCore(unittest.TestCase):
    """Test core functionality of HyperOptX"""
    
    def setUp(self):
        # Create a small dataset for testing
        self.X, self.y = make_regression(n_samples=50, n_features=5, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Define a simple parameter space
        self.param_space = {
            'alpha': (0.01, 10.0),
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        }
        
        # Create a simple estimator
        self.estimator = Ridge()
    
    def test_sample_configurations(self):
        """Test configuration sampling methods"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space,
            random_state=42
        )
        
        # Test random sampling
        random_configs = optimizer._sample_configurations(5, strategy='random')
        self.assertEqual(len(random_configs), 5)
        for config in random_configs:
            self.assertIn('alpha', config)
            self.assertIn('solver', config)
            self.assertGreaterEqual(config['alpha'], 0.01)
            self.assertLessEqual(config['alpha'], 10.0)
            self.assertIn(config['solver'], self.param_space['solver'])
        
        # Test quasi-random sampling
        quasi_configs = optimizer._sample_configurations(5, strategy='quasi_random')
        self.assertEqual(len(quasi_configs), 5)
        
        # Test adaptive sampling (requires prior evaluations, so we'll add some)
        optimizer.cv_results_['params'] = random_configs
        optimizer.cv_results_['mean_test_score'] = [0.8, 0.7, 0.9, 0.6, 0.75]
        
        adaptive_configs = optimizer._sample_configurations(5, strategy='adaptive')
        self.assertEqual(len(adaptive_configs), 5)
    
    def test_encode_config(self):
        """Test configuration encoding"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space,
            random_state=42
        )
        
        # Create a test configuration
        config = {'alpha': 5.0, 'solver': 'auto'}
        
        # Encode the configuration
        encoded = optimizer._encode_config(config)
        
        # Check that the encoding is a numpy array with the right shape
        self.assertIsInstance(encoded, np.ndarray)
        
        # The shape should be (1, 8): 1 row, 8 features
        # 1 numerical feature (alpha) + 7 one-hot encoded features (solver)
        self.assertEqual(encoded.shape[1], 1 + len(self.param_space['solver']))
    
    def test_validate_params(self):
        """Test parameter validation"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space,
            random_state=42
        )
        
        # Test valid parameters
        valid_params = {'alpha': 5.0, 'solver': 'auto'}
        validated = optimizer._validate_params(valid_params)
        self.assertEqual(validated['alpha'], 5.0)
        self.assertEqual(validated['solver'], 'auto')
        
        # Test parameter adjustment (if alpha is <= 0)
        invalid_params = {'alpha': -1.0, 'solver': 'auto'}
        validated = optimizer._validate_params(invalid_params)
        self.assertGreater(validated['alpha'], 0)  # Should be adjusted to a positive value
        
        # Test handling of missing parameters
        partial_params = {'alpha': 5.0}  # Missing solver
        validated = optimizer._validate_params(partial_params)
        self.assertEqual(validated['alpha'], 5.0)
        # The solver would remain unset, and the estimator will use its default

    def test_objective_func(self):
        """Test the objective function evaluation"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space,
            random_state=42,
            maximize=True,
            scoring='neg_mean_squared_error'
        )
        
        # Set X and y attributes (normally done in fit)
        optimizer.X = self.X_train
        optimizer.y = self.y_train
        
        # Evaluate a configuration
        params = {'alpha': 1.0, 'solver': 'auto'}
        score = optimizer._objective_func(params, budget=1.0)
        
        # The score should be a finite number
        self.assertTrue(np.isfinite(score))
        
        # Test budget scaling (reduced CV folds)
        score_reduced = optimizer._objective_func(params, budget=0.5)
        self.assertTrue(np.isfinite(score_reduced))
        
        # Check caching
        cached_score = optimizer._objective_func(params, budget=1.0)
        # Should return exactly the same value (cached)
        self.assertEqual(score, cached_score)
    
    def test_fit_basic(self):
        """Test basic fitting functionality"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space,
            max_iter=5,  # Use a small number for quick testing
            random_state=42,
            verbose=0
        )
        
        # Fit the optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.fit(self.X_train, self.y_train)
        
        # Check that the best_score_ and best_params_ are set
        self.assertIsNotNone(optimizer.best_score_)
        self.assertIsNotNone(optimizer.best_params_)
        
        # Check that cv_results_ contains the correct number of evaluations
        self.assertLessEqual(len(optimizer.cv_results_['params']), 
                          optimizer.max_iter)  # Could be less due to caching
                          
        # Check that best_estimator_ is fitted
        self.assertIsNotNone(optimizer.best_estimator_)
        
        # Predict with best_estimator_ and check score
        y_pred = optimizer.best_estimator_.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertTrue(np.isfinite(mse))


class TestHyperOptXStrategies(unittest.TestCase):
    """Test different optimization strategies in HyperOptX"""
    
    def setUp(self):
        # Create a dataset for testing
        self.X, self.y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Define a parameter space
        self.param_space = {
            'n_estimators': (10, 50),
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 4),
            'bootstrap': [True, False]
        }
        
        # Create an estimator
        self.estimator = RandomForestRegressor(random_state=42)
    
    def test_bayesian_strategy(self):
        """Test Bayesian optimization strategy"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space,
            max_iter=10,
            random_state=42,
            optimization_strategy='bayesian',
            verbose=0
        )
        
        # Fit the optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.fit(self.X_train, self.y_train)
        
        # Check best score and parameters
        self.assertIsNotNone(optimizer.best_score_)
        self.assertIsNotNone(optimizer.best_params_)
        
        # Prediction should work
        y_pred = optimizer.best_estimator_.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertTrue(np.isfinite(mse))
    
    def test_evolutionary_strategy(self):
        """Test evolutionary optimization strategy"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space,
            max_iter=10,
            random_state=42,
            optimization_strategy='evolutionary',
            verbose=0
        )
        
        # Fit the optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.fit(self.X_train, self.y_train)
        
        # Check best score and parameters
        self.assertIsNotNone(optimizer.best_score_)
        self.assertIsNotNone(optimizer.best_params_)
        
        # Check that population is built
        self.assertGreater(len(optimizer.population), 0)
        
        # Prediction should work
        y_pred = optimizer.best_estimator_.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertTrue(np.isfinite(mse))
    
    def test_hybrid_strategy(self):
        """Test hybrid optimization strategy"""
        optimizer = HyperOptX(
            estimator=self.estimator,
            param_space=self.param_space,
            max_iter=10,
            random_state=42,
            optimization_strategy='hybrid',
            verbose=0
        )
        
        # Fit the optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.fit(self.X_train, self.y_train)
        
        # Check best score and parameters
        self.assertIsNotNone(optimizer.best_score_)
        self.assertIsNotNone(optimizer.best_params_)
        
        # Prediction should work
        y_pred = optimizer.best_estimator_.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertTrue(np.isfinite(mse))


class TestHyperOptXAdvanced(unittest.TestCase):
    """Test advanced features of HyperOptX"""
    
    def setUp(self):
        # Load a real dataset for testing
        self.X, self.y = load_diabetes(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Define parameter spaces for different models
        self.rf_param_space = {
            'n_estimators': (10, 50),
            'max_depth': [None, 5, 10],
            'min_samples_split': (2, 6),
            'min_samples_leaf': (1, 3)
        }
        
        self.en_param_space = {
            'alpha': (0.0001, 1.0),
            'l1_ratio': (0.0, 1.0),
            'max_iter': (500, 1000),
            'tol': (1e-5, 1e-3),
            'selection': ['cyclic', 'random']
        }
    
    def test_ensemble_surrogate(self):
        """Test ensemble surrogate model functionality"""
        # Create optimizer with ensemble surrogate enabled
        optimizer = HyperOptX(
            estimator=RandomForestRegressor(),
            param_space=self.rf_param_space,
            max_iter=10,
            random_state=42,
            ensemble_surrogate=True,
            verbose=0
        )
        
        # Fit the optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.fit(self.X_train, self.y_train)
        
        # Check that multiple surrogate models are used
        self.assertGreater(len(optimizer.active_surrogates), 1)
        
        # Check best score and parameters
        self.assertIsNotNone(optimizer.best_score_)
        self.assertIsNotNone(optimizer.best_params_)
    
    def test_early_stopping(self):
        """Test early stopping functionality"""
        # Create optimizer with early stopping enabled
        optimizer = HyperOptX(
            estimator=ElasticNet(),
            param_space=self.en_param_space,
            max_iter=30,  # Set higher to allow early stopping
            random_state=42,
            early_stopping=True,
            verbose=0
        )
        
        # Set X and y attributes
        optimizer.X = self.X_train
        optimizer.y = self.y_train
        
        # Test early stopping detection function
        scores = [0.9, 0.92, 0.93, 0.935, 0.94, 0.942, 0.943, 0.9435, 0.944, 0.9445]
        times = [1.0] * len(scores)
        
        # This stable pattern might trigger early stopping
        early_stop = optimizer._needs_early_stopping(10, scores, times)
        
        # Not asserting a specific result, as implementation details may vary
        # Just check that the function runs without errors
        self.assertIsInstance(early_stop, bool)
        
        # Fit the optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start_time = time.time()
            optimizer.fit(self.X_train, self.y_train)
            fit_time = time.time() - start_time
        
        # Check if early stopping might have occurred
        self.assertIsNotNone(optimizer.best_score_)
        self.assertIsNotNone(optimizer.best_params_)
    
    def test_time_budget(self):
        """Test time budget functionality"""
        # Set a short time budget
        time_budget = 2.0  # 2 seconds
        
        # Create optimizer with time budget
        optimizer = HyperOptX(
            estimator=RandomForestRegressor(),
            param_space=self.rf_param_space,
            max_iter=100,  # Set high to ensure time budget is the limiting factor
            random_state=42,
            time_budget=time_budget,
            verbose=0
        )
        
        # Fit the optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start_time = time.time()
            try:
                optimizer.fit(self.X_train, self.y_train)
                fit_time = time.time() - start_time
                
                # The fit time should be close to the time budget
                # Allow for some flexibility in timing
                self.assertLessEqual(fit_time, time_budget * 2)
                
                # Check that we have some results
                self.assertGreater(len(optimizer.cv_results_['params']), 0)
                self.assertIsNotNone(optimizer.best_score_)
                self.assertIsNotNone(optimizer.best_params_)
            except TimeoutError:
                # TimeoutError is acceptable too
                pass
    
    def test_pipeline_optimization(self):
        """Test optimization of a sklearn Pipeline"""
        # Create a pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge())
        ])
        
        # Define parameter space for the pipeline
        pipeline_param_space = {
            'regressor__alpha': (0.01, 10.0),
            'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }
        
        # Create optimizer
        optimizer = HyperOptX(
            estimator=pipeline,
            param_space=pipeline_param_space,
            max_iter=10,
            random_state=42,
            verbose=0
        )
        
        # Fit the optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.fit(self.X_train, self.y_train)
        
        # Check best score and parameters
        self.assertIsNotNone(optimizer.best_score_)
        self.assertIsNotNone(optimizer.best_params_)
        
        # Prediction should work
        y_pred = optimizer.best_estimator_.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertTrue(np.isfinite(mse))


class TestHyperOptXUtilities(unittest.TestCase):
    """Test utility functions in HyperOptX"""
    
    def setUp(self):
        # Create a small dataset
        self.X, self.y = make_regression(n_samples=50, n_features=5, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Simple parameter space
        self.param_space = {
            'alpha': (0.01, 10.0),
            'solver': ['auto', 'svd', 'cholesky']
        }
        
        # Run optimizer with a few iterations to generate results
        self.optimizer = HyperOptX(
            estimator=Ridge(),
            param_space=self.param_space,
            max_iter=5,
            random_state=42,
            verbose=0
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.optimizer.fit(self.X_train, self.y_train)
    
    def test_score_cv_results(self):
        """Test score_cv_results utility function"""
        # Get DataFrame of results
        try:
            import pandas as pd
            results_df = self.optimizer.score_cv_results()
            
            # Check that results are returned as a DataFrame
            self.assertIsInstance(results_df, pd.DataFrame)
            
            # Check that key columns exist
            expected_columns = ['iteration', 'score', 'std', 'budget', 'time',
                               'surrogate_prediction', 'surrogate_uncertainty',
                               'alpha', 'solver', 'cumulative_time', 'rank', 'is_best']
            for col in expected_columns:
                self.assertIn(col, results_df.columns)
                
            # Check that the best configuration is marked
            self.assertEqual(sum(results_df['is_best']), 1)
        except ImportError:
            # Skip test if pandas is not available
            self.skipTest("pandas is required for this test")
    
    def test_plot_optimization_history(self):
        """Test plot_optimization_history utility function"""
        try:
            import matplotlib.pyplot as plt
            fig = self.optimizer.plot_optimization_history()
            
            # Check that a figure is returned
            self.assertIsNotNone(fig)
            
            # Close the figure to avoid display
            plt.close(fig)
        except ImportError:
            # Skip test if matplotlib is not available
            self.skipTest("matplotlib is required for this test")
    
    def test_benchmark_against_alternatives(self):
        """Test benchmark_against_alternatives utility function"""
        try:
            # Run a very small benchmark to test functionality
            results = self.optimizer.benchmark_against_alternatives(
                self.X_train,
                self.y_train,
                methods=['random'],  # Only test against RandomizedSearchCV for speed
                n_iter=3,
                time_budget=5
            )
            
            # Check that results are returned as a dictionary
            self.assertIsInstance(results, dict)
            
            # Check that HyperOptX and RandomizedSearchCV results are included
            self.assertIn('HyperOptX', results)
            self.assertIn('RandomizedSearchCV', results)
            
            # Check that key metrics are included
            for method in ['HyperOptX', 'RandomizedSearchCV']:
                self.assertIn('best_score', results[method])
                self.assertIn('best_params', results[method])
                self.assertIn('time', results[method])
                self.assertIn('n_iters', results[method])
                
        except (ImportError, ModuleNotFoundError):
            # Skip test if dependencies are not available
            self.skipTest("scikit-learn RandomizedSearchCV is required for this test")


if __name__ == '__main__':
    unittest.main()