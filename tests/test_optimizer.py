import unittest
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the ASHTOptimizer class
from modules.engine.optimizer import ASHTOptimizer  # Assuming the class is in a file called asht_optimizer.py

class TestASHTOptimizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and common parameters"""
        # Create regression dataset
        self.X_reg, self.y_reg = make_regression(
            n_samples=100, n_features=5, noise=0.1, random_state=42
        )
        
        # Create classification dataset
        self.X_clf, self.y_clf = make_classification(
            n_samples=100, n_features=5, n_informative=3, random_state=42
        )
        
        # Split data
        self.X_reg_train, self.X_reg_test, self.y_reg_train, self.y_reg_test = train_test_split(
            self.X_reg, self.y_reg, test_size=0.2, random_state=42
        )
        
        self.X_clf_train, self.X_clf_test, self.y_clf_train, self.y_clf_test = train_test_split(
            self.X_clf, self.y_clf, test_size=0.2, random_state=42
        )
        
        # Scale data
        scaler = StandardScaler()
        self.X_reg_train_scaled = scaler.fit_transform(self.X_reg_train)
        self.X_reg_test_scaled = scaler.transform(self.X_reg_test)
        
        scaler = StandardScaler()
        self.X_clf_train_scaled = scaler.fit_transform(self.X_clf_train)
        self.X_clf_test_scaled = scaler.transform(self.X_clf_test)
    
    def test_initialization(self):
        """Test that the optimizer initializes correctly"""
        # Define parameter space
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False]
        }
        
        # Initialize optimizer
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            max_iter=10,
            random_state=42
        )
        
        # Check that attributes are set correctly
        self.assertEqual(optimizer.max_iter, 10)
        self.assertEqual(optimizer.random_state, 42)
        self.assertIsNone(optimizer.best_params_)
        self.assertEqual(optimizer.best_score_, -float('inf'))
        self.assertIsNone(optimizer.best_estimator_)
        
        # Check parameter space analysis
        self.assertEqual(optimizer.param_types['alpha'], 'numerical')
        self.assertEqual(optimizer.param_types['fit_intercept'], 'categorical')
        
        # Check parameter bounds
        self.assertEqual(optimizer.param_bounds['alpha'], (0.1, 10.0))
    
    def test_sample_random_configs(self):
        """Test random configuration sampling"""
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False]
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            random_state=42
        )
        
        # Sample configurations
        configs = optimizer._sample_random_configs(5)
        
        # Check that we get the right number of configurations
        self.assertEqual(len(configs), 5)
        
        # Check that configurations have the right structure
        for config in configs:
            self.assertIn('alpha', config)
            self.assertIn('fit_intercept', config)
            self.assertTrue(0.1 <= config['alpha'] <= 10.0)
            self.assertIn(config['fit_intercept'], [True, False])
    
    def test_encode_decode_config(self):
        """Test encoding and decoding of configurations"""
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky']
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            random_state=42
        )
        
        # Create a test configuration
        config = {
            'alpha': 1.5,
            'fit_intercept': True,
            'solver': 'svd'
        }
        
        # Encode the configuration
        encoded = optimizer._encode_config(config)
        
        # Check that encoding has the right shape
        # 1 numerical + 2 categorical (with 2 and 3 options) = 1 + 2 + 3 = 6 features
        self.assertEqual(encoded.shape, (1, 6))
        
        # Decode the encoded vector back to a configuration
        param_names = sorted(list(param_space.keys()))
        decoded = optimizer._decode_vector_to_config(encoded[0], param_names)
        
        # Check that decoding preserves the original values
        self.assertAlmostEqual(decoded['alpha'], config['alpha'])
        self.assertEqual(decoded['fit_intercept'], config['fit_intercept'])
        self.assertEqual(decoded['solver'], config['solver'])
    
    def test_configs_to_features(self):
        """Test conversion of multiple configurations to feature matrix"""
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False]
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            random_state=42
        )
        
        # Create test configurations
        configs = [
            {'alpha': 1.0, 'fit_intercept': True},
            {'alpha': 2.0, 'fit_intercept': False},
            {'alpha': 3.0, 'fit_intercept': True}
        ]
        
        # Convert to features
        features = optimizer._configs_to_features(configs)
        
        # Check shape (3 configs, 3 features: 1 numerical + 2 categorical)
        self.assertEqual(features.shape, (3, 3))
        
        # Check values
        self.assertAlmostEqual(features[0, 0], 1.0)  # alpha for first config
        self.assertEqual(features[0, 1], 1.0)  # fit_intercept=True for first config
        self.assertEqual(features[0, 2], 0.0)  # fit_intercept=False for first config
        
        self.assertAlmostEqual(features[1, 0], 2.0)  # alpha for second config
        self.assertEqual(features[1, 1], 0.0)  # fit_intercept=True for second config
        self.assertEqual(features[1, 2], 1.0)  # fit_intercept=False for second config
    
    def test_train_surrogate_model(self):
        """Test training of surrogate model"""
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False]
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            random_state=42
        )
        
        # Create test configurations and scores
        configs = [
            {'alpha': 1.0, 'fit_intercept': True},
            {'alpha': 2.0, 'fit_intercept': False},
            {'alpha': 3.0, 'fit_intercept': True},
            {'alpha': 4.0, 'fit_intercept': False},
            {'alpha': 5.0, 'fit_intercept': True}
        ]
        
        scores = [0.8, 0.6, 0.9, 0.7, 0.5]
        
        # Train surrogate model
        surrogate = optimizer._train_surrogate_model(configs, scores)
        
        # Check that surrogate model is trained
        self.assertTrue(hasattr(surrogate, 'tree_'))
    
    def test_expected_improvement(self):
        """Test expected improvement calculation"""
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False]
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            random_state=42
        )
        
        # Create test configurations and scores
        configs = [
            {'alpha': 1.0, 'fit_intercept': True},
            {'alpha': 2.0, 'fit_intercept': False},
            {'alpha': 3.0, 'fit_intercept': True},
            {'alpha': 4.0, 'fit_intercept': False},
            {'alpha': 5.0, 'fit_intercept': True}
        ]
        
        scores = [0.8, 0.6, 0.9, 0.7, 0.5]
        
        # Train surrogate model
        surrogate = optimizer._train_surrogate_model(configs, scores)
        
        # Test point
        x = np.array([2.5, 1.0, 0.0])  # alpha=2.5, fit_intercept=True
        
        # Calculate expected improvement
        param_names = sorted(list(param_space.keys()))
        ei = optimizer._expected_improvement(x, surrogate, 0.9, param_names)
        
        # Check that EI is a negative number (for minimization)
        self.assertLessEqual(ei, 0)
    
    def test_optimize_acquisition(self):
        """Test optimization of acquisition function"""
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False]
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            random_state=42
        )
        
        # Create test configurations and scores
        configs = [
            {'alpha': 1.0, 'fit_intercept': True},
            {'alpha': 2.0, 'fit_intercept': False},
            {'alpha': 3.0, 'fit_intercept': True},
            {'alpha': 4.0, 'fit_intercept': False},
            {'alpha': 5.0, 'fit_intercept': True}
        ]
        
        scores = [0.8, 0.6, 0.9, 0.7, 0.5]
        
        # Train surrogate model
        surrogate = optimizer._train_surrogate_model(configs, scores)
        
        # Set best score
        optimizer.best_score_ = 0.9
        
        # Optimize acquisition function
        param_names = sorted(list(param_space.keys()))
        config = optimizer._optimize_acquisition(surrogate, 0.9, param_names, n_restarts=2)
        
        # Check that we get a valid configuration
        self.assertIn('alpha', config)
        self.assertIn('fit_intercept', config)
        self.assertTrue(0.1 <= config['alpha'] <= 10.0)
        self.assertIn(config['fit_intercept'], [True, False])
    
    def test_propose_using_surrogate(self):
        """Test proposal of configurations using surrogate model"""
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False]
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            random_state=42
        )
        
        # Create test configurations and scores
        configs = [
            {'alpha': 1.0, 'fit_intercept': True},
            {'alpha': 2.0, 'fit_intercept': False},
            {'alpha': 3.0, 'fit_intercept': True},
            {'alpha': 4.0, 'fit_intercept': False},
            {'alpha': 5.0, 'fit_intercept': True}
        ]
        
        scores = [0.8, 0.6, 0.9, 0.7, 0.5]
        
        # Train surrogate model
        surrogate = optimizer._train_surrogate_model(configs, scores)
        
        # Set best score
        optimizer.best_score_ = 0.9
        
        # Propose configuration
        config = optimizer._propose_using_surrogate(surrogate, param_space)
        
        # Check that we get a valid configuration
        self.assertIn('alpha', config)
        self.assertIn('fit_intercept', config)
        self.assertTrue(0.1 <= config['alpha'] <= 10.0)
        self.assertIn(config['fit_intercept'], [True, False])
    
    def test_propose_batch_using_surrogate(self):
        """Test batch proposal of configurations using surrogate model"""
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False]
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            random_state=42
        )
        
        # Create test configurations and scores
        configs = [
            {'alpha': 1.0, 'fit_intercept': True},
            {'alpha': 2.0, 'fit_intercept': False},
            {'alpha': 3.0, 'fit_intercept': True},
            {'alpha': 4.0, 'fit_intercept': False},
            {'alpha': 5.0, 'fit_intercept': True}
        ]
        
        scores = [0.8, 0.6, 0.9, 0.7, 0.5]
        
        # Train surrogate model
        surrogate = optimizer._train_surrogate_model(configs, scores)
        
        # Set best score
        optimizer.best_score_ = 0.9
        
        # Propose batch of configurations
        batch_size = 3
        configs = optimizer._propose_batch_using_surrogate(surrogate, param_space, batch_size)
        
        # Check that we get the right number of configurations
        self.assertEqual(len(configs), batch_size)
        
        # Check that configurations are valid
        for config in configs:
            self.assertIn('alpha', config)
            self.assertIn('fit_intercept', config)
            self.assertTrue(0.1 <= config['alpha'] <= 10.0)
            self.assertIn(config['fit_intercept'], [True, False])
    
    def test_refine_param_space(self):
        """Test refinement of parameter space"""
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            random_state=42
        )
        
        # Create test configurations and scores
        configs = [
            {'alpha': 1.0, 'fit_intercept': True, 'solver': 'auto'},
            {'alpha': 2.0, 'fit_intercept': False, 'solver': 'svd'},
            {'alpha': 3.0, 'fit_intercept': True, 'solver': 'cholesky'},
            {'alpha': 4.0, 'fit_intercept': False, 'solver': 'lsqr'},
            {'alpha': 5.0, 'fit_intercept': True, 'solver': 'sparse_cg'}
        ]
        
        scores = [0.8, 0.6, 0.9, 0.7, 0.5]
        
        # Train surrogate model
        surrogate = optimizer._train_surrogate_model(configs, scores)
        
        # Refine parameter space
        refined_space = optimizer._refine_param_space(param_space, surrogate)
        
        # Check that refined space is a valid parameter space
        self.assertIn('alpha', refined_space)
        self.assertIn('fit_intercept', refined_space)
        self.assertIn('solver', refined_space)
        
        # Check that numerical ranges are narrowed
        if isinstance(refined_space['alpha'], tuple):
            low, high = refined_space['alpha']
            self.assertTrue(low >= 0.1)
            self.assertTrue(high <= 10.0)
        
        # Check that categorical options might be reduced
        if isinstance(refined_space['solver'], list):
            self.assertTrue(len(refined_space['solver']) <= 5)
    
    def test_objective_func(self):
        """Test objective function evaluation"""
        param_space = {
            'alpha': (0.1, 10.0),
            'fit_intercept': [True, False]
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            random_state=42
        )
        
        # Set data
        optimizer.X = self.X_reg_train_scaled
        optimizer.y = self.y_reg_train
        
        # Evaluate a configuration
        config = {'alpha': 1.0, 'fit_intercept': True}
        score = optimizer._objective_func(config, budget=0.5)
        
        # Check that score is a number
        self.assertIsInstance(score, float)
        
        # Check that results are stored
        self.assertEqual(len(optimizer.cv_results_['params']), 1)
        self.assertEqual(len(optimizer.cv_results_['mean_test_score']), 1)
        self.assertEqual(len(optimizer.cv_results_['std_test_score']), 1)
        self.assertEqual(len(optimizer.cv_results_['budget']), 1)
        
        # Check that best score and params are updated
        self.assertEqual(optimizer.best_score_, score)
        self.assertEqual(optimizer.best_params_, config)
    
    def test_fit_regression(self):
        """Test full optimization process on regression problem"""
        param_space = {
            'alpha': (0.01, 10.0),
            'fit_intercept': [True, False]
        }
        
        optimizer = ASHTOptimizer(
            estimator=Ridge(),
            param_space=param_space,
            max_iter=10,  # Small number for testing
            random_state=42
        )
        
        # Fit optimizer
        optimizer.fit(self.X_reg_train_scaled, self.y_reg_train)
        
        # Check that best parameters are found
        self.assertIsNotNone(optimizer.best_params_)
        self.assertIn('alpha', optimizer.best_params_)
        self.assertIn('fit_intercept', optimizer.best_params_)
        
        # Check that best estimator is set
        self.assertIsNotNone(optimizer.best_estimator_)
        
        # Check that best estimator performs reasonably
        y_pred = optimizer.best_estimator_.predict(self.X_reg_test_scaled)
        mse = mean_squared_error(self.y_reg_test, y_pred)
        self.assertLess(mse, 10.0)  # Arbitrary threshold for this test
    
    def test_fit_classification(self):
        """Test full optimization process on classification problem"""
        param_space = {
            'C': (0.01, 10.0),
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
        
        optimizer = ASHTOptimizer(
            estimator=LogisticRegression(max_iter=1000),
            param_space=param_space,
            max_iter=10,  # Small number for testing
            random_state=42
        )
        
        # Fit optimizer
        optimizer.fit(self.X_clf_train_scaled, self.y_clf_train)
        
        # Check that best parameters are found
        self.assertIsNotNone(optimizer.best_params_)
        self.assertIn('C', optimizer.best_params_)
        self.assertIn('penalty', optimizer.best_params_)
        
        # Check that best estimator is set
        self.assertIsNotNone(optimizer.best_estimator_)
        
        # Check that best estimator performs reasonably
        y_pred = optimizer.best_estimator_.predict(self.X_clf_test_scaled)
        accuracy = accuracy_score(self.y_clf_test, y_pred)
        self.assertGreater(accuracy, 0.6)  # Arbitrary threshold for this test
    
    def test_complex_parameter_space(self):
        """Test with a more complex parameter space including nested parameters"""
        param_space = {
            'n_estimators': (10, 100),
            'max_depth': (3, 10),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 4),
            'bootstrap': [True, False]
        }
        
        optimizer = ASHTOptimizer(
            estimator=RandomForestRegressor(random_state=42),
            param_space=param_space,
            max_iter=10,  # Small number for testing
            random_state=42
        )
        
        # Fit optimizer
        optimizer.fit(self.X_reg_train, self.y_reg_train)
        
        # Check that best parameters are found
        self.assertIsNotNone(optimizer.best_params_)
        for param in param_space:
            self.assertIn(param, optimizer.best_params_)
        
        # Check that best estimator is set
        self.assertIsNotNone(optimizer.best_estimator_)
        
        # Check that best estimator performs reasonably
        y_pred = optimizer.best_estimator_.predict(self.X_reg_test)
        mse = mean_squared_error(self.y_reg_test, y_pred)
        self.assertLess(mse, 10.0)  # Arbitrary threshold for this test

if __name__ == '__main__':
    unittest.main()
