import numpy as np
import math
import pandas as pd
from sklearn.model_selection import cross_val_score, ParameterSampler
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from scipy.stats import norm, truncnorm
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.optimize import NonlinearConstraint, LinearConstraint
from scipy.optimize import linprog, milp
import warnings
from tqdm.auto import tqdm
import time
from joblib import Parallel, delayed
import multiprocessing
from collections import defaultdict
import heapq
import random
import itertools
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging


# Set up logging
logger = logging.getLogger(__name__)


class HyperOptX:
    """
    Advanced Hyperparameter Optimization with Multi-Stage Optimization and Meta-Learning
    
    This optimizer combines:
    - Multi-fidelity optimization with adaptive resource allocation
    - Meta-model selection for surrogate model choice
    - Thompson sampling for exploration/exploitation balance
    - Advanced acquisition functions with entropy search
    - Quasi-Monte Carlo methods for efficient search space exploration
    - Linear programming for categorical parameter optimization
    - Population-based training with evolutionary strategies
    - Constraint satisfaction for parameter compatibility
    - Transfer learning from previous optimization runs
    - Ensemble surrogate models for improved prediction
    """
    
    def __init__(
        self, 
        estimator,
        param_space,
        max_iter=100,
        cv=5,
        scoring=None,
        random_state=None,
        n_jobs=-1,
        verbose=0,
        maximize=True,
        time_budget=None,
        ensemble_surrogate=True,
        transfer_learning=True,
        optimization_strategy='auto',
        early_stopping=True,
        meta_learning=True,
        constraint_handling='auto'
    ):
        """
        Initialize the HyperOptX optimizer
        
        Parameters:
        -----------
        estimator : estimator object
            The machine learning estimator to optimize
            
        param_space : dict
            Dictionary with parameter names as keys and search space as values:
            - For numerical parameters: tuple (low, high)
            - For categorical parameters: list of values
            - For distributions: scipy.stats distribution objects
            
        max_iter : int, default=100
            Maximum number of iterations
            
        cv : int, default=5
            Number of cross-validation folds
            
        scoring : str or callable, default=None
            Scoring function to use
            
        random_state : int, default=None
            Random state for reproducibility
            
        n_jobs : int, default=-1
            Number of jobs for parallel processing (-1 for all processors)
            
        verbose : int, default=0
            Verbosity level
            
        maximize : bool, default=True
            Whether to maximize or minimize the objective
            
        time_budget : float, default=None
            Maximum time budget in seconds
            
        ensemble_surrogate : bool, default=True
            Whether to use ensemble surrogate models
            
        transfer_learning : bool, default=True
            Whether to use transfer learning from previous runs
            
        optimization_strategy : str, default='auto'
            Strategy for optimization: 'auto', 'bayesian', 'evolutionary', 'hybrid'
            
        early_stopping : bool, default=True
            Whether to use early stopping
            
        meta_learning : bool, default=True
            Whether to use meta-learning for surrogate model selection
            
        constraint_handling : str, default='auto'
            Strategy for constraint handling: 'auto', 'penalty', 'projection', 'repair'
        """
        self.estimator = estimator
        self.param_space = param_space
        self.max_iter = max_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.verbose = verbose
        self.maximize = maximize
        self.time_budget = time_budget
        self.ensemble_surrogate = ensemble_surrogate
        self.transfer_learning = transfer_learning
        self.optimization_strategy = optimization_strategy
        self.early_stopping = early_stopping
        self.meta_learning = meta_learning
        self.constraint_handling = constraint_handling
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        # Set default score based on optimization direction
        self.best_score_ = -float('inf') if self.maximize else float('inf')
        self.best_params_ = None
        self.best_estimator_ = None
        
        # Store all evaluation results
        self.cv_results_ = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'budget': [],
            'training_time': [],
            'iteration': [],
            'surrogate_prediction': [],
            'surrogate_uncertainty': []
        }
        
        # Initialize internal state
        self._initialize_state()

    def _initialize_state(self):
        """Initialize the internal state and analyze parameter space"""
        # Parameter space analysis
        self.param_types = self._analyze_param_space()
        self.param_bounds = self._get_param_bounds()
        self.param_constraints = self._analyze_param_constraints()
        
        # Scaler for numerical features
        self.scaler = StandardScaler()
        
        # Timing and budget tracking
        self.start_time = None
        self.total_eval_time = 0
        self.iteration_count = 0
        
        # Cache for evaluated configurations
        self.evaluated_configs = {}
        
        # Feature encoding cache
        self.feature_cache = {}
        
        # Quasi-random sequence generator for efficient space exploration
        self.qr_sequence = self._initialize_quasi_random()
        
        # Initialize surrogate model(s)
        self._initialize_surrogate_models()
        
        # Evolutionary population
        self.population = []
        self.pop_scores = []
        
        # Learning curve models for early stopping
        if self.early_stopping:
            self.learning_curve_model = Ridge(alpha=1.0)
            
        # Meta-learning history
        self.meta_learning_data = {
            'problem_features': [],
            'best_surrogate': [],
            'surrogate_performance': []
        }
        
        # Transfer learning cache
        self.transfer_knowledge = {}
    
    def _analyze_param_space(self) -> Dict[str, str]:
        """Analyze parameter space to determine parameter types"""
        param_types = {}
        for param_name, param_value in self.param_space.items():
            if isinstance(param_value, list):
                param_types[param_name] = 'categorical'
            elif isinstance(param_value, tuple) and len(param_value) == 2:
                if isinstance(param_value[0], int) and isinstance(param_value[1], int):
                    param_types[param_name] = 'integer'
                else:
                    param_types[param_name] = 'numerical'
            elif hasattr(param_value, 'rvs'):  # scipy distribution
                param_types[param_name] = 'distribution'
            else:
                param_types[param_name] = 'unknown'
        return param_types
    
    def _get_param_bounds(self) -> Dict[str, Tuple]:
        """Get bounds for numerical parameters"""
        return {param_name: self.param_space[param_name] 
                for param_name, param_type in self.param_types.items() 
                if param_type in ['numerical', 'integer']}
    
    def _analyze_param_constraints(self) -> List[Dict]:
        """Analyze parameter constraints and dependencies"""
        # Common sklearn parameter constraints
        constraints = []
        
        # Example: solver-penalty compatibility in sklearn linear models
        if hasattr(self.estimator, 'solver') and hasattr(self.estimator, 'penalty'):
            if 'solver' in self.param_space and 'penalty' in self.param_space:
                incompatible_combos = {
                    'lbfgs': {'valid_penalties': ['l2', None], 'default': 'l2'},
                    'newton-cg': {'valid_penalties': ['l2', None], 'default': 'l2'},
                    'sag': {'valid_penalties': ['l2', None], 'default': 'l2'},
                    'saga': {'valid_penalties': ['l1', 'l2', 'elasticnet', None], 'default': 'l2'},
                    'liblinear': {'valid_penalties': ['l1', 'l2'], 'default': 'l2'}
                }
                
                constraints.append({
                    'type': 'compatibility',
                    'params': ['solver', 'penalty'],
                    'relation': incompatible_combos
                })
        
        # Example: max_iter must be greater than certain threshold
        if hasattr(self.estimator, 'max_iter') and 'max_iter' in self.param_space:
            constraints.append({
                'type': 'bound',
                'param': 'max_iter',
                'relation': 'greater_equal',
                'value': 10
            })
        
        # More advanced constraints can be added here based on estimator type
        estimator_class = self.estimator.__class__.__name__
        
        if estimator_class == 'XGBRegressor' or estimator_class == 'XGBClassifier':
            # Example: max_depth and num_leaves compatibility
            if 'max_depth' in self.param_space and 'num_leaves' in self.param_space:
                constraints.append({
                    'type': 'derived',
                    'params': ['max_depth', 'num_leaves'],
                    'relation': lambda d, l: l <= 2**d
                })
        
        return constraints
    
    def _initialize_quasi_random(self):
        """Initialize quasi-random sequence generators for more efficient search space exploration"""
        # Sobol sequence for efficient space covering
        try:
            from scipy.stats import qmc
            return qmc.Sobol(d=len(self.param_space), scramble=True, seed=self.random_state)
        except ImportError:
            # Fallback to random sampling if scipy.stats.qmc is not available
            return None

    def _initialize_surrogate_models(self):
        """Initialize surrogate models based on meta-learning or ensemble strategy"""
        self.surrogate_models = {}
        
        # Base GP model with Matern kernel (good for hyperparameter optimization)
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.01)
        base_gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=self.random_state
        )
        
        # Random Forest model (robust for mixed parameter types)
        base_rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=3,
            random_state=self.random_state,
            n_jobs=min(self.n_jobs, 4)  # Limit RF internal parallelism
        )
        
        # Neural network model for complex landscapes
        base_nn = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            random_state=self.random_state
        )
        
        # Register surrogate models
        self.surrogate_models['gp'] = base_gp
        self.surrogate_models['rf'] = base_rf
        self.surrogate_models['nn'] = base_nn
        
        # For ensembling we use a meta-model
        if self.ensemble_surrogate:
            self.meta_model = Ridge(alpha=1.0)
            self.surrogate_weights = None
            self.active_surrogates = ['gp', 'rf']  # Start with these two
        else:
            # Default to Gaussian Process
            self.active_surrogates = ['gp']
        
        # Current surrogate model state
        self.trained_surrogates = {}
        
        # Initialize meta-model selection tracking
        self.surrogate_performances = {model: [] for model in self.surrogate_models}
        
        # Store training data for uncertainty estimation
        self.X_valid = None
        self.y_valid = None
    
    def _validate_params(self, params: Dict) -> Dict:
        """Validate and fix parameter compatibility issues"""
        fixed_params = params.copy()
        
        # Apply constraints
        for constraint in self.param_constraints:
            if constraint['type'] == 'compatibility':
                param1, param2 = constraint['params']
                if param1 in fixed_params and param2 in fixed_params:
                    relation = constraint['relation']
                    value1 = fixed_params[param1]
                    
                    if value1 in relation:
                        if fixed_params[param2] not in relation[value1]['valid_penalties']:
                            fixed_params[param2] = relation[value1]['default']
            
            elif constraint['type'] == 'bound':
                param = constraint['param']
                if param in fixed_params:
                    if constraint['relation'] == 'greater_equal' and fixed_params[param] < constraint['value']:
                        fixed_params[param] = constraint['value']
                    elif constraint['relation'] == 'less_equal' and fixed_params[param] > constraint['value']:
                        fixed_params[param] = constraint['value']
            
            elif constraint['type'] == 'derived':
                params_list = constraint['params']
                if all(p in fixed_params for p in params_list):
                    values = [fixed_params[p] for p in params_list]
                    if not constraint['relation'](*values):
                        # Apply a repair strategy - can be customized
                        # For now, choose a simple strategy
                        for i, param in enumerate(params_list):
                            if self.param_types[param] == 'integer' or self.param_types[param] == 'numerical':
                                # Adjust numerical parameters
                                bounds = self.param_bounds[param]
                                fixed_params[param] = (fixed_params[param] + bounds[0]) / 2
                                break
        
        # Handle model-specific validations
        estimator_class = self.estimator.__class__.__name__
        
        # Fix common sklearn parameters
        if 'C' in fixed_params and fixed_params['C'] <= 0:
            fixed_params['C'] = 1e-6
        if 'alpha' in fixed_params and fixed_params['alpha'] <= 0:
            fixed_params['alpha'] = 1e-6
        if 'learning_rate' in fixed_params and fixed_params['learning_rate'] <= 0:
            fixed_params['learning_rate'] = 1e-6
            
        # Ensure integer parameters are integers
        for param, param_type in self.param_types.items():
            if param_type == 'integer' and param in fixed_params:
                fixed_params[param] = int(round(fixed_params[param]))
        
        return fixed_params
    
    def _encode_config(self, config: Dict) -> np.ndarray:
        """Encode a configuration into a numerical vector with caching"""
        # Check if this config has already been encoded
        config_key = frozenset(config.items())
        if config_key in self.feature_cache:
            return self.feature_cache[config_key]
        
        # Calculate features by parameter type
        numerical_features = []
        categorical_features = []
        
        # Process parameters
        for param_name, param_type in self.param_types.items():
            if param_name in config:
                if param_type in ['numerical', 'integer', 'distribution']:
                    # Standardize numerical values to [0,1] if bounds are available
                    if param_type in ['numerical', 'integer'] and param_name in self.param_bounds:
                        low, high = self.param_bounds[param_name]
                        # Handle case where low == high (fixed parameter)
                        if low == high:
                            numerical_features.append(0.5)
                        else:
                            scaled_value = (float(config[param_name]) - low) / (high - low)
                            numerical_features.append(scaled_value)
                    else:
                        # For distributions or parameters without bounds
                        numerical_features.append(float(config[param_name]))
                elif param_type == 'categorical':
                    # One-hot encoding for categorical parameters
                    categories = self.param_space[param_name]
                    one_hot = [1.0 if val == config[param_name] else 0.0 for val in categories]
                    categorical_features.extend(one_hot)
        
        # Combine all features into a single matrix
        if not numerical_features and not categorical_features:
            # Empty config
            features = np.array([]).reshape(1, -1)
        else:
            num_array = np.array(numerical_features).reshape(1, -1) if numerical_features else np.array([]).reshape(1, 0)
            cat_array = np.array(categorical_features).reshape(1, -1) if categorical_features else np.array([]).reshape(1, 0)
            features = np.hstack([num_array, cat_array]) if num_array.size > 0 or cat_array.size > 0 else np.array([]).reshape(1, 0)
        
        # Cache the result
        self.feature_cache[config_key] = features
        
        return features
    
    def _configs_to_features(self, configs: List[Dict]) -> np.ndarray:
        """Convert configurations to feature matrix with batch processing"""
        if not configs:
            return np.array([]).reshape(0, 0)
        
        # Process first config to get dimensions
        first_features = self._encode_config(configs[0])
        if first_features.size == 0:
            return np.array([]).reshape(len(configs), 0)
            
        n_features = first_features.shape[1]
        
        # Pre-allocate feature matrix
        X = np.zeros((len(configs), n_features))
        X[0] = first_features
        
        # Process remaining configs
        for i, config in enumerate(configs[1:], 1):
            features = self._encode_config(config)
            if features.size > 0:
                X[i] = features
        
        return X
    
    def _objective_func(self, params: Dict, budget: float = 1.0, store: bool = True) -> float:
        """Evaluate a configuration with the given budget and optional storage"""
        # Validate and fix parameters
        params = self._validate_params(params)
        
        # Check cache first to avoid redundant evaluations
        param_key = frozenset(params.items())
        budget_key = round(budget, 3)  # Round to avoid floating point issues
        
        if param_key in self.evaluated_configs and budget_key in self.evaluated_configs[param_key]:
            return self.evaluated_configs[param_key][budget_key]['score']
        
        # Set parameters on a clone of the estimator
        estimator = clone(self.estimator)
        try:
            estimator.set_params(**params)
        except Exception as e:
            if self.verbose > 1:
                logger.warning(f"Parameter setting error: {e}")
            # Return a very poor score for invalid configurations
            return -float('inf') if self.maximize else float('inf')
        
        # Calculate actual CV based on budget
        actual_cv = max(2, min(self.cv, int(self.cv * budget)))
        
        # Record time for performance analysis
        start_time = time.time()
        
        # Perform cross-validation with error handling
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    estimator, self.X, self.y, 
                    cv=actual_cv, 
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    error_score='raise'
                )
            
            # Handle NaN scores
            scores = scores[~np.isnan(scores)]
            if len(scores) == 0:
                mean_score = -float('inf') if self.maximize else float('inf')
                std_score = 0
            else:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
        except Exception as e:
            if self.verbose > 1:
                logger.warning(f"Cross-validation error: {e}")
            mean_score = -float('inf') if self.maximize else float('inf')
            std_score = 0
            scores = [mean_score]
        
        elapsed_time = time.time() - start_time
        self.total_eval_time += elapsed_time
        
        # Check for time budget
        if self.time_budget is not None and (time.time() - self.start_time) > self.time_budget:
            raise TimeoutError("Time budget exceeded")
        
        # Store results if requested
        if store:
            self.cv_results_['params'].append(params)
            self.cv_results_['mean_test_score'].append(mean_score)
            self.cv_results_['std_test_score'].append(std_score)
            self.cv_results_['budget'].append(budget)
            self.cv_results_['training_time'].append(elapsed_time)
            self.cv_results_['iteration'].append(self.iteration_count)
            
            # Add placeholder for surrogate predictions (filled later)
            self.cv_results_['surrogate_prediction'].append(None)
            self.cv_results_['surrogate_uncertainty'].append(None)
            
            # Update best result if needed
            if (self.maximize and mean_score > self.best_score_) or \
               (not self.maximize and mean_score < self.best_score_):
                self.best_score_ = mean_score
                self.best_params_ = params.copy()
        
            # Cache the result
            if param_key not in self.evaluated_configs:
                self.evaluated_configs[param_key] = {}
            
            self.evaluated_configs[param_key][budget_key] = {
                'score': mean_score, 
                'std': std_score,
                'time': elapsed_time,
                'iter': self.iteration_count
            }
        
        return mean_score
    
    def _sample_configurations(self, n: int, strategy: str = 'mixed') -> List[Dict]:
        """Sample configurations using various strategies"""
        if self.random_state is not None:
            np.random.seed(self.random_state + self.iteration_count)
        
        samples = []
        
        if strategy == 'random':
            # Use sklearn's ParameterSampler
            samples = list(ParameterSampler(
                self.param_space, 
                n_iter=n, 
                random_state=self.random_state+self.iteration_count
            ))
            
        elif strategy == 'quasi_random':
            # Use quasi-random sequences for better space coverage
            if self.qr_sequence is not None:
                try:
                    from scipy.stats import qmc
                    # Generate quasi-random points in [0, 1]^d space
                    points = self.qr_sequence.random(n)
                    
                    # Convert to parameter values
                    for point in points:
                        config = {}
                        idx = 0
                        for param_name, param_type in self.param_types.items():
                            if param_type in ['numerical', 'integer']:
                                low, high = self.param_bounds[param_name]
                                value = low + point[idx] * (high - low)
                                if param_type == 'integer':
                                    value = int(round(value))
                                config[param_name] = value
                                idx += 1
                            elif param_type == 'categorical':
                                categories = self.param_space[param_name]
                                cat_idx = int(point[idx] * len(categories))
                                cat_idx = min(cat_idx, len(categories) - 1)  # Ensure valid index
                                config[param_name] = categories[cat_idx]
                                idx += 1
                            elif param_type == 'distribution':
                                # For distributions, use PPF (percent point function)
                                dist = self.param_space[param_name]
                                config[param_name] = dist.ppf(point[idx])
                                idx += 1
                        samples.append(config)
                except (ImportError, AttributeError):
                    # Fallback to random sampling
                    samples = list(ParameterSampler(
                        self.param_space, 
                        n_iter=n, 
                        random_state=self.random_state+self.iteration_count
                    ))
            else:
                # Fallback to random sampling
                samples = list(ParameterSampler(
                    self.param_space, 
                    n_iter=n, 
                    random_state=self.random_state+self.iteration_count
                ))
                
        elif strategy == 'grid':
            # Limited grid sampling using fixed number of points per dimension
            param_values = {}
            for param_name, param_type in self.param_types.items():
                if param_type in ['numerical', 'integer']:
                    low, high = self.param_bounds[param_name]
                    # Adaptive number of points based on dimensionality
                    n_points = max(2, int(np.power(n, 1/len(self.param_types))))
                    param_values[param_name] = np.linspace(low, high, n_points)
                    if param_type == 'integer':
                        param_values[param_name] = np.unique(np.round(param_values[param_name]).astype(int))
                elif param_type == 'categorical':
                    param_values[param_name] = self.param_space[param_name]
                elif param_type == 'distribution':
                    # Sample from distribution
                    dist = self.param_space[param_name]
                    n_points = max(2, int(np.power(n, 1/len(self.param_types))))
                    param_values[param_name] = dist.rvs(size=n_points, random_state=self.random_state+self.iteration_count)
            
            # Generate combinations up to n samples
            keys = list(param_values.keys())
            for values in itertools.islice(itertools.product(*[param_values[k] for k in keys]), n):
                config = {k: v for k, v in zip(keys, values)}
                samples.append(config)
                
        elif strategy == 'adaptive':
            # Combine best configurations with local perturbations
            if len(self.cv_results_['params']) > 0:
                # Get top configurations
                top_k = min(5, len(self.cv_results_['params']))
                
                # Sort by score
                indices = np.argsort(self.cv_results_['mean_test_score'])
                if self.maximize:
                    indices = indices[::-1]  # Reverse for maximization
                
                top_configs = [self.cv_results_['params'][i] for i in indices[:top_k]]
                
                # Generate variants
                for base_config in top_configs:
                    # Add the base config
                    samples.append(base_config.copy())
                    
                    # Generate local variations
                    for _ in range(n // (top_k * 2)):
                        variant = base_config.copy()
                        # Modify random parameters
                        n_params_to_change = max(1, np.random.randint(1, len(variant) // 2 + 1))
                        params_to_change = random.sample(list(variant.keys()), n_params_to_change)
                        
                        for param in params_to_change:
                            param_type = self.param_types[param]
                            if param_type in ['numerical', 'integer']:
                                low, high = self.param_bounds[param]
                                # Local perturbation
                                current = variant[param]
                                # Gaussian perturbation with standard deviation of 10% of range
                                std_dev = (high - low) * 0.1
                                new_value = current + np.random.normal(0, std_dev)
                                # Clip to bounds
                                new_value = max(low, min(high, new_value))
                                if param_type == 'integer':
                                    new_value = int(round(new_value))
                                variant[param] = new_value
                            elif param_type == 'categorical':
                                categories = self.param_space[param]
                                # Select a different category with higher probability
                                current_idx = categories.index(variant[param])
                                other_indices = [i for i in range(len(categories)) if i != current_idx]
                                if other_indices:
                                    new_idx = np.random.choice(other_indices)
                                    variant[param] = categories[new_idx]
                        
                        samples.append(variant)
            
            # Fill remaining slots with random samples
            remaining = n - len(samples)
            if remaining > 0:
                random_samples = list(ParameterSampler(
                    self.param_space, 
                    n_iter=remaining, 
                    random_state=self.random_state+self.iteration_count
                ))
                samples.extend(random_samples)
                
        elif strategy == 'mixed':
            # Blend of different strategies
            n_random = n // 3
            n_quasi = n // 3
            n_adaptive = n - n_random - n_quasi
            
            # Get samples from each strategy
            random_samples = self._sample_configurations(n_random, 'random')
            quasi_samples = self._sample_configurations(n_quasi, 'quasi_random')
            adaptive_samples = self._sample_configurations(n_adaptive, 'adaptive')
            
            # Combine all samples
            samples = random_samples + quasi_samples + adaptive_samples
        
        # Validate all configurations
        return [self._validate_params(config) for config in samples]
    
    def _evolutionary_search(self, n_offspring: int = 10, mutation_prob: float = 0.2) -> List[Dict]:
        """Perform evolutionary search based on current population"""
        if not self.population:
            # Initialize population with random samples if empty
            return self._sample_configurations(n_offspring, 'mixed')
        
        # Sort population by fitness
        pop_with_scores = list(zip(self.population, self.pop_scores))
        if self.maximize:
            pop_with_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            pop_with_scores.sort(key=lambda x: x[1])
            
        sorted_pop = [p[0] for p in pop_with_scores]
        
        offspring = []
        
        # Elitism: Keep top performers
        elite_count = max(1, n_offspring // 5)
        offspring.extend([p.copy() for p in sorted_pop[:elite_count]])
        
        # Crossover and mutation to create remaining offspring
        remaining = n_offspring - elite_count
        
        # Tournament selection
        def tournament_select(k=3):
            """Select a parent using tournament selection"""
            candidates = random.sample(sorted_pop, min(k, len(sorted_pop)))
            return candidates[0]  # Already sorted, so first is best
        
        for _ in range(remaining):
            # Crossover
            if len(sorted_pop) >= 2 and random.random() < 0.7:  # 70% chance of crossover
                parent1 = tournament_select()
                parent2 = tournament_select()
                
                # Create child through crossover
                child = {}
                
                # Uniform crossover for parameters
                for param in set(parent1.keys()).union(parent2.keys()):
                    # 50% chance of inheriting from each parent
                    if param in parent1 and param in parent2:
                        child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
                    elif param in parent1:
                        child[param] = parent1[param]
                    else:
                        child[param] = parent2[param]
            else:
                # No crossover, just clone a parent
                child = tournament_select().copy()
            
            # Mutation
            for param in child:
                if random.random() < mutation_prob:
                    param_type = self.param_types[param]
                    
                    if param_type in ['numerical', 'integer']:
                        low, high = self.param_bounds[param]
                        # Gaussian mutation
                        std_dev = (high - low) * 0.1  # 10% of range
                        child[param] += random.gauss(0, std_dev)
                        # Clip to bounds
                        child[param] = max(low, min(high, child[param]))
                        if param_type == 'integer':
                            child[param] = int(round(child[param]))
                    elif param_type == 'categorical':
                        categories = self.param_space[param]
                        # Select a random category, possibly the same one
                        child[param] = random.choice(categories)
            
            # Validate and add to offspring
            child = self._validate_params(child)
            offspring.append(child)
        
        return offspring
    
    def _train_surrogate_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train surrogate models on evaluated configurations"""
        trained_models = {}
        
        if X.shape[0] < 2:  # Need at least 2 samples
            return trained_models
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Direction adjustment for minimization
        y_adj = y.copy()
        if not self.maximize:
            y_adj = -y_adj
            
        # Filter out non-finite values
        finite_mask = np.isfinite(y_adj)
        if np.sum(finite_mask) < 2:  # Need at least 2 valid samples
            return trained_models
            
        X_valid = X_scaled[finite_mask]
        y_valid = y_adj[finite_mask]
        
        # Store for uncertainty estimation
        self.X_valid = X_valid
        self.y_valid = y_valid
        
        # Train each surrogate model in the active set
        model_errors = {}
        
        for model_name in self.active_surrogates:
            model = self.surrogate_models[model_name]
            
            try:
                # Train the model
                model.fit(X_valid, y_valid)
                
                # Calculate cross-validation error for model selection
                if len(X_valid) >= 5:  # Need enough data for CV
                    from sklearn.model_selection import cross_val_predict
                    y_pred = cross_val_predict(model, X_valid, y_valid, cv=min(5, len(X_valid)))
                    mse = np.mean((y_valid - y_pred) ** 2)
                    model_errors[model_name] = mse
                    
                    # Store for meta-learning
                    self.surrogate_performances[model_name].append(mse)
                
                # Add to trained models
                trained_models[model_name] = model
                
            except Exception as e:
                if self.verbose > 1:
                    logger.warning(f"Error training {model_name} surrogate: {e}")
        
        # Meta-model selection based on performance
        if model_errors and self.meta_learning:
            # Check if we should switch models
            current_best = min(model_errors, key=model_errors.get)
            
            # Record for meta-learning
            if X.shape[1] > 0:  # Skip if no features
                self.meta_learning_data['problem_features'].append({
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'y_mean': np.mean(y_valid),
                    'y_std': np.std(y_valid),
                    'iteration': self.iteration_count
                })
                self.meta_learning_data['best_surrogate'].append(current_best)
                self.meta_learning_data['surrogate_performance'].append(model_errors)
            
            # Update active surrogates
            if self.ensemble_surrogate:
                # Keep all models, but update weights in acquisition
                self.active_surrogates = list(trained_models.keys())
                
                # Compute weights inversely proportional to error
                errors = np.array([model_errors.get(m, float('inf')) for m in self.active_surrogates])
                if np.all(np.isfinite(errors)) and np.sum(errors) > 0:
                    weights = 1.0 / (errors + 1e-10)
                    self.surrogate_weights = weights / np.sum(weights)
                else:
                    self.surrogate_weights = np.ones(len(self.active_surrogates)) / len(self.active_surrogates)
            else:
                # Use single best model
                self.active_surrogates = [current_best]
        
        # If no models could be trained, create a dummy surrogate
        if not trained_models:
            trained_models['dummy'] = self._create_dummy_surrogate(y)
            self.active_surrogates = ['dummy']
        
        return trained_models

    def _create_dummy_surrogate(self, y: np.ndarray) -> Any:
        """Create a simple dummy surrogate when model training fails"""
        # Use valid scores only
        valid_scores = y[np.isfinite(y)]
        mean_score = np.mean(valid_scores) if len(valid_scores) > 0 else 0.0
        std_score = np.std(valid_scores) if len(valid_scores) > 0 else 0.1
        
        class DummySurrogate:
            def predict(self, X):
                return np.full(X.shape[0], mean_score)
                
            def predict_with_std(self, X):
                return np.full(X.shape[0], mean_score), np.full(X.shape[0], std_score)
        
        return DummySurrogate()
    
    def _acquisition_function(self, x: np.ndarray, models: Dict[str, Any], best_f: float, xi: float = 0.01) -> float:
        """
        Advanced acquisition function with ensemble support and adaptive exploration
        
        Parameters:
        -----------
        x : array-like
            Point to evaluate
        
        models : dict
            Dictionary of trained surrogate models
        
        best_f : float
            Best function value observed so far
        
        xi : float
            Exploration-exploitation parameter
            
        Returns:
        --------
        acquisition_value : float
            Value of acquisition function
        """
        # Reshape x if needed
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Scale features
        x_scaled = self.scaler.transform(x)
        
        # Check which acquisition function to use based on iteration
        acq_type = self._select_acquisition_function()
        
        # Compute predictions from all models
        all_means = []
        all_stds = []
        
        for i, model_name in enumerate(self.active_surrogates):
            if model_name not in models:
                continue
                
            model = models[model_name]
            
            # Different prediction method based on model type
            if model_name == 'gp':
                try:
                    mean, std = model.predict(x_scaled, return_std=True)
                    all_means.append(mean)
                    all_stds.append(std)
                except Exception as e:
                    if self.verbose > 1:
                        logger.warning(f"Error in GP acquisition prediction: {e}")
            
            elif model_name == 'rf':
                try:
                    # Mean prediction
                    mean = model.predict(x_scaled)
                    
                    # Get std from individual trees
                    tree_preds = np.array([tree.predict(x_scaled) for tree in model.estimators_])
                    std = np.std(tree_preds, axis=0)
                    
                    all_means.append(mean)
                    all_stds.append(std)
                except Exception as e:
                    if self.verbose > 1:
                        logger.warning(f"Error in RF acquisition prediction: {e}")
            
            elif model_name == 'nn':
                try:
                    mean = model.predict(x_scaled)
                    
                    # Simple uncertainty
                    if hasattr(self, 'X_valid') and self.X_valid is not None and self.X_valid.shape[0] > 0:
                        dists = np.min(np.sum((x_scaled[:, np.newaxis, :] - self.X_valid[np.newaxis, :, :]) ** 2, axis=2), axis=1)
                        std = np.sqrt(dists) * 0.1 + 0.05
                    else:
                        std = np.ones_like(mean) * 0.1
                        
                    all_means.append(mean)
                    all_stds.append(std)
                except Exception as e:
                    if self.verbose > 1:
                        logger.warning(f"Error in NN acquisition prediction: {e}")
            
            else:  # dummy or other
                try:
                    mean = model.predict(x_scaled)
                    std = np.ones_like(mean) * 0.1
                    
                    all_means.append(mean)
                    all_stds.append(std)
                except Exception as e:
                    if self.verbose > 1:
                        logger.warning(f"Error in dummy acquisition prediction: {e}")
        
        # If no models provided valid predictions, return a default value
        if not all_means:
            return 0.0
            
        # Compute ensemble prediction
        if self.ensemble_surrogate and self.surrogate_weights is not None and len(all_means) > 1:
            # Ensure we have correct number of weights
            if len(self.surrogate_weights) != len(all_means):
                # Equal weights as fallback
                weights = np.ones(len(all_means)) / len(all_means)
            else:
                weights = self.surrogate_weights
                
            # Weighted mean
            mu = np.zeros_like(all_means[0])
            for i, mean in enumerate(all_means):
                mu += weights[i] * mean
                
            # Weighted variance (including model disagreement)
            total_var = np.zeros_like(all_stds[0])
            
            # Within-model variance
            for i, std in enumerate(all_stds):
                total_var += weights[i] * (std ** 2)
                
            # Between-model variance (disagreement)
            for i, mean in enumerate(all_means):
                total_var += weights[i] * ((mean - mu) ** 2)
                
            sigma = np.sqrt(total_var)
        else:
            # Single model or equal weights
            mu = all_means[0]
            sigma = all_stds[0]
        
        # Handle zero uncertainty
        if np.all(sigma < 1e-6):
            sigma = np.ones_like(sigma) * 1e-6
        
        # Compute acquisition value based on selected function
        if acq_type == 'ei':  # Expected Improvement
            # Adjust for maximization/minimization
            if self.maximize:
                imp = mu - best_f - xi
            else:
                # For minimization problems, we negate predictions
                imp = best_f - mu - xi
                
            # Z-score
            z = imp / sigma
            
            # EI formula: (imp * CDF(z) + sigma * pdf(z))
            cdf = 0.5 * (1 + np.array([math.erf(val / np.sqrt(2)) for val in z]))
            pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            
            ei = imp * cdf + sigma * pdf
            acquisition = ei[0]  # Take first element for single point
            
        elif acq_type == 'ucb':  # Upper Confidence Bound
            # Adaptive beta based on iteration count
            beta = 0.5 + np.log(1 + self.iteration_count)
            
            if self.maximize:
                acquisition = mu[0] + beta * sigma[0]
            else:
                acquisition = -(mu[0] - beta * sigma[0])
                
        elif acq_type == 'poi':  # Probability of Improvement
            if self.maximize:
                z = (mu - best_f - xi) / sigma
            else:
                z = (best_f - mu - xi) / sigma
                
            # POI is just the CDF
            acquisition = 0.5 * (1 + math.erf(z[0] / np.sqrt(2)))
            
        elif acq_type == 'thompson':  # Thompson Sampling
            # Sample from posterior
            if self.maximize:
                acquisition = mu[0] + sigma[0] * np.random.randn()
            else:
                acquisition = -(mu[0] + sigma[0] * np.random.randn())
                
        else:  # Default to EI
            if self.maximize:
                imp = mu - best_f - xi
            else:
                imp = best_f - mu - xi
                
            z = imp / sigma
            cdf = 0.5 * (1 + np.array([math.erf(val / np.sqrt(2)) for val in z]))
            pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            
            ei = imp * cdf + sigma * pdf
            acquisition = ei[0]
        
        # Return negative for minimization with scipy optimize
        return -acquisition
    
    def _select_acquisition_function(self) -> str:
        """Select acquisition function based on optimization stage"""
        # Early iterations: focus on exploration (UCB)
        if self.iteration_count < self.max_iter * 0.3:
            return 'ucb'
        # Middle iterations: balance (EI)
        elif self.iteration_count < self.max_iter * 0.7:
            return 'ei'
        # Late iterations: Thompson sampling for final refinement
        else:
            return 'thompson'
    
    def _decode_vector_to_config(self, vector: np.ndarray, param_names: List[str]) -> Dict:
        """Convert optimization vector back to parameter configuration"""
        config = {}
        idx = 0
        
        for param_name in param_names:
            param_type = self.param_types[param_name]
            
            if param_type in ['numerical', 'integer']:
                # Clip to bounds
                low, high = self.param_bounds[param_name]
                value = np.clip(vector[idx], low, high)
                
                # Cast to integer if needed
                if param_type == 'integer':
                    value = int(round(value))
                
                config[param_name] = value
                idx += 1
            elif param_type == 'categorical':
                categories = self.param_space[param_name]
                n_categories = len(categories)
                
                # Get one-hot part of the vector
                one_hot = vector[idx:idx+n_categories]
                
                # Find the index of the maximum value
                category_idx = np.argmax(one_hot)
                config[param_name] = categories[category_idx]
                
                idx += n_categories
            elif param_type == 'distribution':
                # For distributions, store the raw value
                config[param_name] = vector[idx]
                idx += 1
        
        # Validate the configuration
        config = self._validate_params(config)
        return config
    
    def _optimize_acquisition(self, surrogate_models: Dict[str, Any], best_f: float, n_restarts: int = 5) -> Dict:
        """
        Optimize acquisition function to find the next point to evaluate
        
        Uses multiple optimization methods:
        1. L-BFGS-B for continuous parameters
        2. Dual annealing for global optimization
        3. Linear programming for categorical parameters
        """
        # Get sorted parameter names for consistency
        param_names = sorted(list(self.param_types.keys()))
        
        # Determine problem type based on parameter types
        has_numerical = any(self.param_types[p] in ['numerical', 'integer'] for p in param_names)
        has_categorical = any(self.param_types[p] == 'categorical' for p in param_names)
        
        # Strategy selection based on parameter types
        if has_categorical and has_numerical:
            return self._optimize_mixed_space(surrogate_models, best_f, param_names)
        elif has_categorical:
            return self._optimize_categorical_space(surrogate_models, best_f, param_names)
        else:
            return self._optimize_continuous_space(surrogate_models, best_f, param_names)
    
    def _optimize_continuous_space(self, surrogate_models: Dict[str, Any], best_f: float, param_names: List[str]) -> Dict:
        """Optimize acquisition in continuous parameter space with improved robustness.
        
        This method employs multiple optimization strategies to find the global optimum of
        the acquisition function, handling numerical stability issues and potential optimization failures.
        
        Parameters:
        -----------
        surrogate_models : Dict[str, Any]
            The trained surrogate models to use for acquisition function
        best_f : float
            The best objective value observed so far
        param_names : List[str]
            List of parameter names to optimize
        
        Returns:
        --------
        Dict
            The configuration with optimized parameter values
        """
        # Define bounds for optimization
        bounds = []
        
        for param_name in param_names:
            if self.param_types[param_name] in ['numerical', 'integer']:
                low, high = self.param_bounds[param_name]
                bounds.append((low, high))
            else:  # Handle any non-numerical parameter types
                bounds.append((0, 1))
        
        # Early stopping - if no bounds, return random configuration
        if not bounds:
            return self._sample_configurations(1, 'random')[0]
        
        # Multiple optimization approaches for better global search
        results = []
        
        # 1. L-BFGS-B with multiple restarts from diverse starting points
        n_restarts = min(10, max(5, len(bounds) * 2))  # Scale with dimensionality
        
        # Generate diverse starting points using quasi-random sequence if available
        x0_points = []
        try:
            if self.qr_sequence is not None:
                # Use quasi-random points for better coverage
                from scipy.stats import qmc
                points = self.qr_sequence.random(n_restarts)
                for point in points:
                    x0 = np.zeros(len(bounds))
                    for i, (low, high) in enumerate(bounds):
                        x0[i] = low + point[i] * (high - low)
                    x0_points.append(x0)
            else:
                # Fallback to random with Latin Hypercube Sampling
                for _ in range(n_restarts):
                    x0 = np.zeros(len(bounds))
                    for i, (low, high) in enumerate(bounds):
                        x0[i] = np.random.uniform(low, high)
                    x0_points.append(x0)
        except:
            # Basic random sampling as ultimate fallback
            for _ in range(n_restarts):
                x0 = np.zeros(len(bounds))
                for i, (low, high) in enumerate(bounds):
                    x0[i] = np.random.uniform(low, high)
                x0_points.append(x0)
        
        # Run L-BFGS-B optimizations in parallel if possible
        if self.n_jobs > 1:
            try:
                from joblib import Parallel, delayed
                
                def run_lbfgs_optimization(x0):
                    try:
                        result = minimize(
                            lambda x: self._acquisition_function(x, surrogate_models, best_f),
                            x0,
                            bounds=bounds,
                            method='L-BFGS-B',
                            options={'maxiter': 200}
                        )
                        if result.success:
                            return (result.x, -result.fun)
                        return None
                    except:
                        return None
                
                lbfgs_results = Parallel(n_jobs=min(self.n_jobs, len(x0_points)))(
                    delayed(run_lbfgs_optimization)(x0) for x0 in x0_points
                )
                
                # Filter out None results
                lbfgs_results = [r for r in lbfgs_results if r is not None]
                results.extend(lbfgs_results)
                
            except:
                # Fallback to sequential optimization
                for x0 in x0_points:
                    try:
                        result = minimize(
                            lambda x: self._acquisition_function(x, surrogate_models, best_f),
                            x0,
                            bounds=bounds,
                            method='L-BFGS-B', 
                            options={'maxiter': 200}
                        )
                        if result.success:
                            results.append((result.x, -result.fun))
                    except Exception as e:
                        if self.verbose > 1:
                            logger.warning(f"L-BFGS-B optimization error: {e}")
        else:
            # Sequential optimization
            for x0 in x0_points:
                try:
                    result = minimize(
                        lambda x: self._acquisition_function(x, surrogate_models, best_f),
                        x0,
                        bounds=bounds,
                        method='L-BFGS-B',
                        options={'maxiter': 200}
                    )
                    if result.success:
                        results.append((result.x, -result.fun))
                except Exception as e:
                    if self.verbose > 1:
                        logger.warning(f"L-BFGS-B optimization error: {e}")
        
        # 2. Dual annealing for global optimization (with reduced max iterations for speed)
        try:
            result = dual_annealing(
                lambda x: self._acquisition_function(x, surrogate_models, best_f),
                bounds,
                maxiter=100,
                seed=self.random_state
            )
            
            if result.success:
                results.append((result.x, -result.fun))
        except Exception as e:
            if self.verbose > 1:
                logger.warning(f"Dual annealing optimization error: {e}")
        
        # 3. Differential evolution with smaller population but sufficient to explore space
        try:
            result = differential_evolution(
                lambda x: self._acquisition_function(x, surrogate_models, best_f),
                bounds,
                popsize=min(10, max(5, len(bounds))),  # Adaptive population size
                maxiter=20,  # Limited iterations for efficiency
                seed=self.random_state,
                tol=1e-3,    # Relaxed tolerance
                updating='deferred'  # More robust convergence
            )
            
            if result.success:
                results.append((result.x, -result.fun))
        except Exception as e:
            if self.verbose > 1:
                logger.warning(f"Differential evolution optimization error: {e}")
        
        # 4. If all optimizations failed or found poor results, sample points and evaluate directly
        if not results or max(r[1] for r in results) < 1e-8:
            # Direct sampling as fallback
            n_samples = min(100, max(20, 10 * len(bounds)))
            configs = self._sample_configurations(n_samples, 'quasi_random')
            
            # Evaluate all configs with surrogate
            X_configs = self._configs_to_features(configs)
            if X_configs.shape[0] > 0:
                X_scaled = self.scaler.transform(X_configs)
                acq_values = []
                
                # Batch evaluation for efficiency
                for i in range(X_scaled.shape[0]):
                    x = X_scaled[i:i+1]
                    try:
                        acq = -self._acquisition_function(x, surrogate_models, best_f)
                        acq_values.append(acq)
                    except:
                        acq_values.append(float('-inf'))
                
                # Find best config
                if acq_values:
                    best_idx = np.argmax(acq_values)
                    return configs[best_idx]
        
        # Find the best result across all optimization methods
        if not results:
            return self._sample_configurations(1, 'random')[0]
        
        # Get the best result with proper error handling
        try:
            best_x, _ = max(results, key=lambda x: x[1])
            
            # Convert to configuration
            return self._decode_vector_to_config(best_x, param_names)
        except:
            # Final fallback for safety
            return self._sample_configurations(1, 'random')[0]
    
    def _optimize_categorical_space(self, surrogate_models: Dict[str, Any], best_f: float, param_names: List[str]) -> Dict:
        """Optimize acquisition in categorical parameter space.
        
        If the number of categorical configurations is small (<= 1000), then evaluate all possible
        combinations. Otherwise, fallback to a sampling approach. In cases where non-categorical
        parameters are involved, a default value is used which will be scaled later.
        
        Parameters:
        -----------
        surrogate_models : Dict[str, Any]
            A dictionary of surrogate models used to evaluate configurations.
        best_f : float
            The current best function value.
        param_names : List[str]
            The names of parameters to be optimized.
            
        Returns:
        --------
        Dict
            The best configuration found.
        """
        import numpy as np
        import itertools

        # Calculate the number of combinations for categorical parameters.
        categorical_lengths = [len(self.param_space[p]) for p in param_names if self.param_types[p] == 'categorical']
        n_combinations = np.prod(categorical_lengths) if categorical_lengths else 1

        # If the search space is small enough, evaluate all configurations.
        if n_combinations <= 1000:
            configs = []
            param_values = {}
            for param in param_names:
                if self.param_types[param] == 'categorical':
                    param_values[param] = self.param_space[param]
                else:
                    # For non-categorical parameters, use a single default value.
                    param_values[param] = [0.5]  # This value will be scaled appropriately later.
            
            # Generate all possible configurations.
            for values in itertools.product(*(param_values[p] for p in param_names)):
                config = dict(zip(param_names, values))
                configs.append(config)
            
            # Evaluate all configurations using the surrogate model.
            X_configs = self._configs_to_features(configs)
            if X_configs.shape[0] > 0:
                X_scaled = self.scaler.transform(X_configs)
                acq_values = []
                # Evaluate the acquisition function on each configuration.
                for i in range(X_scaled.shape[0]):
                    x = X_scaled[i:i+1]
                    acq = -self._acquisition_function(x, surrogate_models, best_f)
                    acq_values.append(acq)
                
                best_idx = int(np.argmax(acq_values))
                return configs[best_idx]

        # Fallback: for large categorical spaces, use a sampling approach.
        n_samples = min(1000, max(100, int(n_combinations * 0.1)))
        configs = self._sample_configurations(n_samples, 'quasi_random')
        
        # Evaluate the sampled configurations.
        X_configs = self._configs_to_features(configs)
        if X_configs.shape[0] > 0:
            X_scaled = self.scaler.transform(X_configs)
            acq_values = [-self._acquisition_function(X_scaled[i:i+1], surrogate_models, best_f)
                        for i in range(X_scaled.shape[0])]
            best_idx = int(np.argmax(acq_values))
            return configs[best_idx]
        
        # Final fallback to random sampling if all else fails.
        return self._sample_configurations(1, 'random')[0]

    def _optimize_mixed_space(self, surrogate_models: Dict[str, Any], best_f: float, param_names: List[str]) -> Dict:
        """Optimize acquisition in mixed parameter space (categorical + numerical)"""
        # Strategy: Fix categorical parameters and optimize continuous space
        categorical_params = [p for p in param_names if self.param_types[p] == 'categorical']
        numerical_params = [p for p in param_names if self.param_types[p] in ['numerical', 'integer']]
        
        # Generate candidates for categorical params
        n_candidates = min(50, max(5, 5 * len(categorical_params)))
        
        # Two approaches:
        # 1. Sample from promising categorical combinations (if we have data)
        # 2. Sample randomly
        
        cat_configs = []
        
        # If we have enough evaluations, learn promising categorical values
        if len(self.cv_results_['params']) >= 10:
            # Get scores and configs
            scores = np.array(self.cv_results_['mean_test_score'])
            configs = self.cv_results_['params']
            
            # Sort by score
            indices = np.argsort(scores)
            if self.maximize:
                indices = indices[::-1]  # Reverse for maximization
                
            # Get top and random configs
            top_k = min(5, len(configs))
            top_configs = [configs[i] for i in indices[:top_k]]
            random_configs = self._sample_configurations(n_candidates - top_k, 'random')
            
            # Extract categorical configurations from top performers
            for config in top_configs:
                cat_config = {p: config[p] for p in categorical_params if p in config}
                cat_configs.append(cat_config)
                
            # Add some random configurations for exploration
            for config in random_configs:
                cat_config = {p: config[p] for p in categorical_params if p in config}
                cat_configs.append(cat_config)
        else:
            # Sample randomly
            random_configs = self._sample_configurations(n_candidates, 'random')
            for config in random_configs:
                cat_config = {p: config[p] for p in categorical_params if p in config}
                cat_configs.append(cat_config)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_cat_configs = []
        for config in cat_configs:
            config_key = frozenset(config.items())
            if config_key not in seen:
                seen.add(config_key)
                unique_cat_configs.append(config)
        
        # For each categorical configuration, optimize numerical parameters
        best_overall_config = None
        best_overall_acq = float('-inf')
        
        for cat_config in unique_cat_configs:
            # Define optimization problem for numerical parameters
            bounds = [(self.param_bounds[p][0], self.param_bounds[p][1]) for p in numerical_params]
            
            # Skip if no numerical parameters
            if not bounds:
                # Evaluate this categorical config directly
                full_config = cat_config.copy()
                config_features = self._encode_config(full_config)
                if config_features.shape[1] > 0:
                    scaled_features = self.scaler.transform(config_features)
                    acq_value = -self._acquisition_function(scaled_features, surrogate_models, best_f)
                    if acq_value > best_overall_acq:
                        best_overall_acq = acq_value
                        best_overall_config = full_config
                continue
            
            # Helper function for optimization that includes fixed categorical params
            def objective(x):
                # Create full configuration
                full_config = cat_config.copy()
                for i, param in enumerate(numerical_params):
                    full_config[param] = x[i]
                
                # Validate
                full_config = self._validate_params(full_config)
                
                # Encode and evaluate
                config_features = self._encode_config(full_config)
                scaled_features = self.scaler.transform(config_features)
                return self._acquisition_function(scaled_features, surrogate_models, best_f)
            
            # Multiple starting points
            x0_points = []
            for _ in range(2):  # Fewer restarts per categorical config
                x0 = np.zeros(len(bounds))
                for i, (low, high) in enumerate(bounds):
                    x0[i] = np.random.uniform(low, high)
                x0_points.append(x0)
            
            best_x = None
            best_acq = float('-inf')
            
            # Try optimization with different starting points
            for x0 in x0_points:
                try:
                    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
                    if result.success and -result.fun > best_acq:
                        best_acq = -result.fun
                        best_x = result.x
                except Exception:
                    continue
            
            # If optimization succeeded, create full config
            if best_x is not None:
                full_config = cat_config.copy()
                for i, param in enumerate(numerical_params):
                    value = best_x[i]
                    if self.param_types[param] == 'integer':
                        value = int(round(value))
                    full_config[param] = value
                
                full_config = self._validate_params(full_config)
                
                # Update best overall if better
                if best_acq > best_overall_acq:
                    best_overall_acq = best_acq
                    best_overall_config = full_config
        
        # If optimization failed completely, sample randomly
        if best_overall_config is None:
            return self._sample_configurations(1, 'random')[0]
        
        return best_overall_config
    
    def _multi_fidelity_schedule(self, max_iter: int) -> List[float]:
        """
        Create a multi-fidelity evaluation schedule
        
        Starts with low fidelity (small budgets) and progressively
        increases to higher fidelity (larger budgets)
        """
        # Initialization phase - start with low budget
        min_budget = 0.2  # 20% resources
        
        # Generate schedule - exponential ramp-up
        log_range = np.linspace(np.log(min_budget), np.log(1.0), max_iter // 3 + 1)
        schedule = np.exp(log_range)
        
        # Make sure last few iterations use full budget
        budget_steps = list(schedule) + [1.0] * (max_iter - len(schedule))
        return budget_steps[:max_iter]
    
    def _successive_halving(self, configs: List[Dict], budget: float, n_survivors: int) -> List[Dict]:
        """Evaluate configurations and keep the best ones (successive halving)"""
        if len(configs) <= n_survivors:
            return configs  # No need to eliminate if we already have few enough
            
        # Evaluate all configs with the current budget
        results = []
        for config in configs:
            score = self._objective_func(config, budget=budget)
            results.append((config, score))
        
        # Sort by score
        if self.maximize:
            results.sort(key=lambda x: x[1], reverse=True)
        else:
            results.sort(key=lambda x: x[1])
            
        # Keep top n_survivors
        return [config for config, _ in results[:n_survivors]]
    
    def _update_population(self, new_configs: List[Dict], new_scores: List[float], max_size: int = 30):
            """Update the evolutionary population with new configurations"""
            # Add new configurations
            for config, score in zip(new_configs, new_scores):
                # Only add if valid score
                if np.isfinite(score):
                    self.population.append(config.copy())
                    self.pop_scores.append(score)
            
            # If population is too large, keep only the best ones
            if len(self.population) > max_size:
                # Create combined list and sort by score
                combined = list(zip(self.population, self.pop_scores))
                if self.maximize:
                    combined.sort(key=lambda x: x[1], reverse=True)
                else:
                    combined.sort(key=lambda x: x[1])
                    
                # Keep top performers
                self.population = [p[0] for p in combined[:max_size]]
                self.pop_scores = [p[1] for p in combined[:max_size]]
    
    def _needs_early_stopping(self, iteration: int, scores: List[float], times: List[float]) -> bool:
        """Determine if optimization should be stopped early"""
        try:
            if not self.early_stopping:
                return False
                
            # Need at least 10 iterations to detect trends
            if iteration < 10:
                return False
                
            # Stop if time budget exceeded
            if self.time_budget is not None and (time.time() - self.start_time) > self.time_budget:
                return True
                
            # Check for score convergence
            if len(scores) >= 10:
                recent_scores = scores[-10:]
                
                # Extract valid scores
                valid_scores = [s for s in recent_scores if np.isfinite(s)]
                if len(valid_scores) < 5:  # Need enough valid scores
                    return False
                    
                # Convert to numpy array for operations
                valid_scores = np.array(valid_scores)
                
                # Use simple linear fit to detect trend
                x = np.arange(len(valid_scores)).reshape(-1, 1)
                
                try:
                    # Reset and fit the model
                    self.learning_curve_model = Ridge(alpha=1.0)
                    self.learning_curve_model.fit(x, valid_scores)
                    slope = self.learning_curve_model.coef_[0]
                    
                    # Calculate improvement percentage
                    if len(valid_scores) > 1:
                        score_range = np.max(valid_scores) - np.min(valid_scores)
                        if score_range == 0:  # No variation in scores
                            return True
                            
                        # Calculate relative slope
                        norm_slope = slope / score_range
                        
                        # If maximizing, check if slope is very small positive or negative
                        if self.maximize:
                            return bool(norm_slope < 0.001)
                        # If minimizing, check if slope is very small negative or positive
                        else:
                            return bool(norm_slope > -0.001)
                except Exception:
                    # If error in slope calculation, don't trigger early stopping
                    return False
                    
            return False
        except Exception:
            # Safety catch-all to avoid halting optimization due to early stopping error
            return False
    
    def _extract_problem_features(self, X, y):
        """Extract features from the problem for meta-learning"""
        features = {}
        
        # Data characteristics
        features['n_samples'] = X.shape[0]
        features['n_features'] = X.shape[1]
        features['density'] = np.count_nonzero(X) / (X.shape[0] * X.shape[1])
        
        # Target characteristics
        features['y_mean'] = np.mean(y)
        features['y_std'] = np.std(y)
        features['y_skew'] = 0
        try:
            from scipy.stats import skew
            features['y_skew'] = skew(y)
        except:
            pass
            
        # Parameter space characteristics
        features['n_params'] = len(self.param_space)
        features['n_categorical'] = sum(1 for p in self.param_types.values() if p == 'categorical')
        features['n_numerical'] = sum(1 for p in self.param_types.values() if p in ['numerical', 'integer'])
        
        return features

    def fit(self, X, y):
        """
        Run the HyperOptX optimization process
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        y : array-like, shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
            Returns self
        """
        self.X = X
        self.y = y
        
        # Start timing
        self.start_time = time.time()
        
        # Track current iteration number to enforce max_iter
        current_iter = 0
        
        # Extract problem features for meta-learning
        if self.meta_learning:
            problem_features = self._extract_problem_features(X, y)
            if self.verbose:
                logger.info(f"Problem features: {problem_features}")
        
        # Create multi-fidelity budget schedule
        budget_schedule = self._multi_fidelity_schedule(self.max_iter)
        
        # Track scores and configs for early stopping
        all_scores = []
        all_configs = []
        eval_times = []
        
        # Strategy determination
        selected_strategy = self.optimization_strategy
        if selected_strategy == 'auto':
            # Choose strategy based on problem size
            if X.shape[1] > 50 or len(self.param_space) > 10:
                # High-dimensional problems: prefer evolutionary
                selected_strategy = 'evolutionary'
            elif len([p for p, t in self.param_types.items() if t == 'categorical']) > len(self.param_types) / 2:
                # Many categorical parameters: prefer bayesian
                selected_strategy = 'bayesian'
            else:
                # Default to hybrid
                selected_strategy = 'hybrid'
                
        if self.verbose:
            logger.info(f"Selected optimization strategy: {selected_strategy}")
            logger.info(f"Initializing HyperOptX with {self.max_iter} maximum iterations")
        
        # Strict enforcement of max_iter
        # Phase 1: Initial exploration with fewer samples to stay within max_iter
        n_initial = max(1, min(self.max_iter // 4, 5))
        
        if self.verbose:
            logger.info(f"Phase 1: Initial exploration with {n_initial} configurations")
        
        # Generate initial configurations
        initial_configs = self._sample_configurations(n_initial, 'mixed')
        
        # Evaluate initial configs
        initial_results = []
        for config in initial_configs:
            if current_iter >= self.max_iter:
                break
                
            score = self._objective_func(config, budget=0.5)  # 50% budget
            initial_results.append((config, score))
            all_scores.append(score)
            all_configs.append(config)
            
            # Update counters
            current_iter += 1
            self.iteration_count += 1
            
            # Update evolutionary population
            if selected_strategy in ['evolutionary', 'hybrid']:
                self._update_population([config], [score])
        
        # Sort by score
        if self.maximize:
            initial_results.sort(key=lambda x: x[1], reverse=True)
        else:
            initial_results.sort(key=lambda x: x[1])
        
        # Initialize surrogate models with these results
        X_init = self._configs_to_features([cfg for cfg, _ in initial_results])
        y_init = np.array([score for _, score in initial_results])
        
        # Train initial surrogate models
        surrogate_models = self._train_surrogate_models(X_init, y_init)
        
        # Iterative optimization phase
        remaining_iter = self.max_iter - current_iter
        
        if remaining_iter > 0 and self.verbose:
            logger.info(f"Phase 2: Iterative optimization with {remaining_iter} iterations")
        
        while current_iter < self.max_iter:
            # Current budget from schedule
            current_budget = budget_schedule[min(current_iter, len(budget_schedule)-1)]
            
            # Different proposal strategies based on selected strategy
            if selected_strategy == 'bayesian':
                # Bayesian optimization: use acquisition function
                current_best = max(all_scores) if self.maximize else min(all_scores)
                next_config = self._optimize_acquisition(surrogate_models, current_best)
                candidates = [next_config]
                
            elif selected_strategy == 'evolutionary':
                # Evolutionary optimization: generate offspring
                if len(self.population) >= 5:
                    candidates = self._evolutionary_search(n_offspring=1)  # Reduced to just 1
                else:
                    # Not enough population yet, use mixed strategy
                    candidates = self._sample_configurations(1, 'mixed')  # Reduced to just 1
                    
            elif selected_strategy == 'hybrid':
                # Hybrid: mix of bayesian and evolutionary
                candidates = []
                
                # Bayesian candidate
                current_best = max(all_scores) if self.maximize else min(all_scores)
                bayes_config = self._optimize_acquisition(surrogate_models, current_best)
                candidates.append(bayes_config)
                
                # Don't add more candidates if we're near max_iter
                if current_iter < self.max_iter - 1:
                    # Add just one more candidate
                    if len(self.population) >= 5:
                        evo_config = self._evolutionary_search(n_offspring=1)[0]
                        candidates.append(evo_config)
                    else:
                        random_config = self._sample_configurations(1, 'random')[0]
                        candidates.append(random_config)
            
            else:  # Default to mixed sampling
                candidates = self._sample_configurations(1, 'mixed')  # Reduced to just 1
            
            # Limit candidates to respect remaining iterations
            candidates = candidates[:max(1, self.max_iter - current_iter)]
            
            # Evaluate candidates
            batch_scores = []
            batch_times = []
            
            if self.verbose > 0 and (current_iter % 5 == 0 or self.verbose >= 2):
                logger.info(f"  Iteration {current_iter+1}/{self.max_iter}: " + 
                    f"budget={current_budget:.2f}, evaluating {len(candidates)} candidates")
            
            for config in candidates:
                # Check for early stopping
                if self._needs_early_stopping(current_iter, all_scores, eval_times):
                    if self.verbose:
                        logger.info(f"Early stopping at iteration {current_iter+1}")
                    break
                
                # Check if we've reached max_iter
                if current_iter >= self.max_iter:
                    break
                    
                # Evaluate with current budget
                eval_start = time.time()
                score = self._objective_func(config, budget=current_budget)
                eval_time = time.time() - eval_start
                
                batch_scores.append(score)
                batch_times.append(eval_time)
                
                all_scores.append(score)
                all_configs.append(config)
                eval_times.append(eval_time)
                
                # Update evolutionary population
                if selected_strategy in ['evolutionary', 'hybrid']:
                    self._update_population([config], [score])
                
                # Update iteration counter
                current_iter += 1
                self.iteration_count += 1
            
            # Retrain surrogate models with all data
            X_all = self._configs_to_features(all_configs)
            y_all = np.array(all_scores)
            
            # Only retrain if we haven't reached max_iter yet
            if current_iter < self.max_iter and len(all_configs) >= 2:
                surrogate_models = self._train_surrogate_models(X_all, y_all)
                
                # Update surrogate predictions in cv_results
                self._update_surrogate_predictions(surrogate_models)
            
            # Check for early stopping
            if self._needs_early_stopping(current_iter, all_scores, eval_times):
                if self.verbose:
                    logger.info(f"Early stopping at iteration {current_iter}")
                break
        
        # Final evaluation of best configurations with full budget
        if current_iter >= self.max_iter and self.verbose:
            logger.info("Phase 3: Final evaluation of top configurations with full budget")
        
            # Get top configurations
            indices = np.argsort(all_scores)
            if self.maximize:
                indices = indices[::-1]  # Reverse for maximization
                
            # Take top 3 configurations
           
            top_k = min(3, len(all_configs))
            top_configs = [all_configs[i] for i in indices[:top_k]]
            
            # Re-evaluate with full budget if needed
            for config in top_configs:
                # Skip if already evaluated with full budget
                param_key = frozenset(config.items())
                if param_key in self.evaluated_configs and 1.0 in self.evaluated_configs[param_key]:
                    continue
                    
                # We don't count this towards max_iter since it's a final evaluation
                self._objective_func(config, budget=1.0)
        
        # Set the best estimator
        if self.best_params_ is not None:
            try:
                self.best_estimator_ = clone(self.estimator)
                self.best_estimator_.set_params(**self.best_params_)
                self.best_estimator_.fit(X, y)
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Error fitting best estimator: {e}")
                # Fallback: fit with default parameters
                self.best_estimator_ = clone(self.estimator)
                self.best_estimator_.fit(X, y)
        else:
            # No good parameters found, use default
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.fit(X, y)
        
        # Display final results
        if self.verbose:
            logger.info("\nOptimization completed:")
            logger.info(f"Total iterations: {current_iter}")
            logger.info(f"Best score: {self.best_score_:.6f}")
            logger.info(f"Best parameters: {self.best_params_}")
            logger.info(f"Time elapsed: {time.time() - self.start_time:.2f} seconds")
        
        return self
    
    def _update_surrogate_predictions(self, surrogate_models):
        """Update surrogate predictions in cv_results_"""
        if not surrogate_models:
            return
            
        # Get configurations and convert to features
        configs = self.cv_results_['params']
        X_configs = self._configs_to_features(configs)
        
        if X_configs.shape[0] == 0 or X_configs.shape[1] == 0:
            return
            
        X_scaled = self.scaler.transform(X_configs)
        
        # Get predictions from surrogate models
        if self.ensemble_surrogate and len(self.active_surrogates) > 1:
            # Use all models with weights
            means = []
            stds = []
            
            for i, model_name in enumerate(self.active_surrogates):
                if model_name in surrogate_models:
                    model = surrogate_models[model_name]
                    
                    # Different prediction method based on model type
                    if model_name == 'gp':
                        # Gaussian Process has return_std parameter
                        try:
                            mean, std = model.predict(X_scaled, return_std=True)
                            means.append(mean)
                            stds.append(std)
                        except Exception as e:
                            if self.verbose > 1:
                                logger.warning(f"Error in GP prediction: {e}")
                    
                    elif model_name == 'rf':
                        # Random Forest - manually calculate std from trees
                        try:
                            # First get the mean prediction
                            mean = model.predict(X_scaled)
                            
                            # Then get predictions from individual trees
                            tree_preds = np.array([tree.predict(X_scaled) for tree in model.estimators_])
                            std = np.std(tree_preds, axis=0)
                            
                            means.append(mean)
                            stds.append(std)
                        except Exception as e:
                            if self.verbose > 1:
                                logger.warning(f"Error in RF prediction: {e}")
                    
                    elif model_name == 'nn':
                        # Neural Network - use distance-based uncertainty
                        try:
                            mean = model.predict(X_scaled)
                            
                            # Simple uncertainty based on distance to training data
                            if hasattr(self, 'X_valid') and self.X_valid is not None and self.X_valid.shape[0] > 0:
                                dists = np.min(np.sum((X_scaled[:, np.newaxis, :] - self.X_valid[np.newaxis, :, :]) ** 2, axis=2), axis=1)
                                std = np.sqrt(dists) * 0.1 + 0.05
                            else:
                                std = np.ones_like(mean) * 0.1
                                
                            means.append(mean)
                            stds.append(std)
                        except Exception as e:
                            if self.verbose > 1:
                                logger.warning(f"Error in NN prediction: {e}")
            
            # Skip ensemble calculation if no predictions available
            if not means:
                return
                    
            # Calculate ensemble prediction
            if self.surrogate_weights is not None:
                # Make sure we have the right number of weights
                if len(self.surrogate_weights) != len(means):
                    # If mismatch, just use equal weights
                    self.surrogate_weights = np.ones(len(means)) / len(means)
                
                # Weighted mean and variance
                weights = self.surrogate_weights
                mu = np.zeros_like(means[0])
                
                # Compute weighted mean
                for i, mean in enumerate(means):
                    mu += weights[i] * mean
                    
                # Compute weighted variance (including model disagreement)
                total_var = np.zeros_like(stds[0])
                
                # Within-model variance
                for i, std in enumerate(stds):
                    total_var += weights[i] * (std ** 2)
                    
                # Between-model variance (disagreement)
                for i, mean in enumerate(means):
                    total_var += weights[i] * ((mean - mu) ** 2)
                
                sigma = np.sqrt(total_var)
            else:
                # Equal weights if no weights provided
                mu = np.mean(means, axis=0)
                
                # Combine within-model and between-model variance
                within_var = np.mean([std**2 for std in stds], axis=0)
                between_var = np.var(means, axis=0)
                total_var = within_var + between_var
                
                sigma = np.sqrt(total_var)
                
            # Update cv_results_
            for i in range(len(self.cv_results_['surrogate_prediction'])):
                if i < len(mu):
                    self.cv_results_['surrogate_prediction'][i] = mu[i]
                    self.cv_results_['surrogate_uncertainty'][i] = sigma[i]
        else:
            # Use single best model
            if not self.active_surrogates:
                return
                
            model_name = self.active_surrogates[0]
            if model_name not in surrogate_models:
                return
                
            model = surrogate_models[model_name]
            
            # Different prediction method based on model type
            if model_name == 'gp':
                try:
                    means, stds = model.predict(X_scaled, return_std=True)
                except Exception:
                    return
            elif model_name == 'rf':
                try:
                    means = model.predict(X_scaled)
                    
                    # Get std from individual trees
                    tree_preds = np.array([tree.predict(X_scaled) for tree in model.estimators_])
                    stds = np.std(tree_preds, axis=0)
                except Exception:
                    return
            elif model_name == 'nn':
                try:
                    means = model.predict(X_scaled)
                    
                    # Simple uncertainty
                    if hasattr(self, 'X_valid') and self.X_valid is not None and self.X_valid.shape[0] > 0:
                        dists = np.min(np.sum((X_scaled[:, np.newaxis, :] - self.X_valid[np.newaxis, :, :]) ** 2, axis=2), axis=1)
                        stds = np.sqrt(dists) * 0.1 + 0.05
                    else:
                        stds = np.ones_like(means) * 0.1
                except Exception:
                    return
            else:
                # Dummy or unknown model
                try:
                    means = model.predict(X_scaled)
                    stds = np.ones_like(means) * 0.1
                except Exception:
                    return
                    
            # Store predictions
            for i in range(len(self.cv_results_['surrogate_prediction'])):
                if i < len(means):
                    self.cv_results_['surrogate_prediction'][i] = means[i]
                    self.cv_results_['surrogate_uncertainty'][i] = stds[i]
    
    def score_cv_results(self):
        """
        Create a DataFrame with all evaluation results and additional metrics
        
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame with evaluation results
        """
        try:
            import pandas as pd
            
            # Create base DataFrame
            df = pd.DataFrame({
                'iteration': self.cv_results_['iteration'],
                'score': self.cv_results_['mean_test_score'],
                'std': self.cv_results_['std_test_score'],
                'budget': self.cv_results_['budget'],
                'time': self.cv_results_['training_time'],
                'surrogate_prediction': self.cv_results_['surrogate_prediction'],
                'surrogate_uncertainty': self.cv_results_['surrogate_uncertainty']
            })
            
            # Add parameters as columns
            for i, params in enumerate(self.cv_results_['params']):
                for param, value in params.items():
                    if param not in df.columns:
                        df[param] = None
                    df.loc[i, param] = value
            
            # Add derived metrics
            df['cumulative_time'] = df['time'].cumsum()
            
            # Add rank
            df['rank'] = df['score'].rank(ascending=not self.maximize)
            
            # Mark best configuration
            best_idx = df['score'].argmax() if self.maximize else df['score'].argmin()
            df['is_best'] = False
            df.loc[best_idx, 'is_best'] = True
            
            return df
            
        except ImportError:
            if self.verbose:
                logger.warning("pandas is required for score_cv_results()")
            return None
    
    def plot_optimization_history(self, figsize=(12, 8)):
        """
        Plot optimization history including:
        - Score vs iteration
        - Score vs cumulative time
        - Parameter importance
        - Learning curves
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure with optimization history plots
        """
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            from sklearn.inspection import permutation_importance
            
            # Get results as DataFrame
            results_df = self.score_cv_results()
            if results_df is None:
                return None
                
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Plot 1: Score vs iteration
            ax1 = axes[0, 0]
            ax1.plot(results_df['iteration'], results_df['score'], 'o-', alpha=0.6)
            best_idx = results_df['score'].argmax() if self.maximize else results_df['score'].argmin()
            ax1.plot(results_df['iteration'][best_idx], results_df['score'][best_idx], 'r*', markersize=12)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Score')
            ax1.set_title('Score vs Iteration')
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # Plot 2: Score vs cumulative time
            ax2 = axes[0, 1]
            ax2.plot(results_df['cumulative_time'], results_df['score'], 'o-', alpha=0.6)
            ax2.plot(results_df['cumulative_time'][best_idx], results_df['score'][best_idx], 'r*', markersize=12)
            ax2.set_xlabel('Cumulative Time (s)')
            ax2.set_ylabel('Score')
            ax2.set_title('Score vs Time')
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            # Plot 3: Parameter importance if possible
            ax3 = axes[1, 0]
            try:
                # Try to compute parameter importance
                param_cols = [col for col in results_df.columns 
                             if col not in ['iteration', 'score', 'std', 'budget', 'time', 
                                           'surrogate_prediction', 'surrogate_uncertainty',
                                           'cumulative_time', 'rank', 'is_best']]
                
                # Convert categorical to numerical
                X_params = results_df[param_cols].copy()
                for col in X_params.columns:
                    if X_params[col].dtype == 'object':
                        # Simple label encoding
                        categories = X_params[col].dropna().unique()
                        mapping = {cat: i for i, cat in enumerate(categories)}
                        X_params[col] = X_params[col].map(mapping)
                
                # Fill NAs with median
                X_params = X_params.fillna(X_params.median())
                
                # Compute importance using permutation importance
                if len(X_params) >= 10:  # Need enough samples
                    y_score = results_df['score'].values
                    r = permutation_importance(
                        estimator=Ridge(),
                        X=X_params.values,
                        y=y_score,
                        n_repeats=5,
                        random_state=self.random_state
                    )
                    
                    # Plot importance
                    indices = np.argsort(r.importances_mean)[::-1]
                    n_top = min(10, len(param_cols))
                    top_params = [param_cols[i] for i in indices[:n_top]]
                    top_importance = [r.importances_mean[i] for i in indices[:n_top]]
                    
                    bars = ax3.barh(range(n_top), top_importance, align='center')
                    ax3.set_yticks(range(n_top))
                    ax3.set_yticklabels(top_params)
                    ax3.set_xlabel('Importance')
                    ax3.set_title('Parameter Importance')
                else:
                    ax3.text(0.5, 0.5, 'Not enough samples\nfor importance calculation', 
                            ha='center', va='center', transform=ax3.transAxes)
            except Exception as e:
                ax3.text(0.5, 0.5, f'Error computing importance:\n{str(e)}', 
                        ha='center', va='center', transform=ax3.transAxes)
            
            # Plot 4: Surrogate model quality
            ax4 = axes[1, 1]
            if all(p is not None for p in results_df['surrogate_prediction']) and len(results_df) > 5:
                ax4.scatter(results_df['surrogate_prediction'], results_df['score'], alpha=0.7)
                
                # Add ideal line
                min_val = min(results_df['surrogate_prediction'].min(), results_df['score'].min())
                max_val = max(results_df['surrogate_prediction'].max(), results_df['score'].max())
                ax4.plot([min_val, max_val], [min_val, max_val], 'k--')
                
                ax4.set_xlabel('Surrogate Prediction')
                ax4.set_ylabel('Actual Score')
                ax4.set_title('Surrogate Model Quality')
                ax4.grid(True, linestyle='--', alpha=0.6)
                
                # Add correlation coefficient
                corr = results_df['surrogate_prediction'].corr(results_df['score'])
                ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=ax4.transAxes, va='top', ha='left')
            else:
                ax4.text(0.5, 0.5, 'Surrogate model data\nnot available', 
                        ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            if self.verbose:
                logger.warning("matplotlib and pandas are required for plotting")
            return None
    
    def benchmark_against_alternatives(self, X, y, methods=['grid', 'random', 'bayesian'], 
                                    n_iter=50, cv=None, time_budget=None):
        """
        Benchmark against alternative hyperparameter optimization methods.
        
        Parameters:
        -----------
        X : array-like
            Training data
            
        y : array-like
            Target values
            
        methods : list, default=['grid', 'random', 'bayesian']
            Methods to benchmark against
            
        n_iter : int, default=50
            Number of iterations for each method
            
        cv : int, default=None
            Cross-validation folds (defaults to self.cv)
            
        time_budget : float, default=None
            Time budget (in seconds) for each method
            
        Returns:
        --------
        results : dict
            Benchmark results
        """
        import numpy as np
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        import time

        # Use the instance's CV value if none is specified
        cv_value = cv if cv is not None else self.cv

        # Create a reduced parameter space for GridSearchCV
        grid_param_space = {}
        for param, value in self.param_space.items():
            if isinstance(value, list):
                grid_param_space[param] = value
            elif isinstance(value, tuple) and len(value) == 2:
                low, high = value
                if isinstance(low, int) and isinstance(high, int):
                    range_size = high - low
                    if range_size > 10:
                        # For large ranges, sample fewer values
                        step = max(1, range_size // 5)
                        grid_param_space[param] = list(range(low, high + 1, step))
                    else:
                        grid_param_space[param] = list(range(low, high + 1))
                else:
                    grid_param_space[param] = list(np.linspace(low, high, 5))

        results = {
            'HyperOptX': {
                'best_score': None,
                'best_params': None,
                'time': None,
                'n_iters': None
            }
        }

        # Run HyperOptX (our method)
        start_time = time.time()
        self.fit(X, y)
        hyperoptx_time = time.time() - start_time

        results['HyperOptX']['best_score'] = self.best_score_
        results['HyperOptX']['best_params'] = self.best_params_
        results['HyperOptX']['time'] = hyperoptx_time
        results['HyperOptX']['n_iters'] = self.iteration_count

        # -----------------------
        # Run GridSearchCV
        if 'grid' in methods:
            try:
                logger.info("Running GridSearchCV...")
                start_time = time.time()
                grid_search = GridSearchCV(
                    self.estimator,
                    grid_param_space,
                    cv=cv_value,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    verbose=0
                )
                if time_budget:
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(grid_search.fit, X, y)
                        try:
                            future.result(timeout=time_budget)
                        except TimeoutError:
                            logger.warning("  GridSearchCV timed out")
                            results['GridSearchCV'] = {
                                'best_score': float('nan'),
                                'best_params': None,
                                'time': time_budget,
                                'n_iters': 'timed_out'
                            }
                        else:
                            grid_time = time.time() - start_time
                            results['GridSearchCV'] = {
                                'best_score': grid_search.best_score_,
                                'best_params': grid_search.best_params_,
                                'time': grid_time,
                                'n_iters': len(grid_search.cv_results_['params'])
                            }
                else:
                    grid_search.fit(X, y)
                    grid_time = time.time() - start_time
                    results['GridSearchCV'] = {
                        'best_score': grid_search.best_score_,
                        'best_params': grid_search.best_params_,
                        'time': grid_time,
                        'n_iters': len(grid_search.cv_results_['params'])
                    }
            except Exception as e:
                logger.warning(f"  Error in GridSearchCV: {e}")
                results['GridSearchCV'] = {
                    'best_score': float('nan'),
                    'best_params': None,
                    'time': float('nan'),
                    'n_iters': 'error'
                }

        # -----------------------
        # Run RandomizedSearchCV
        if 'random' in methods:
            try:
                logger.info("Running RandomizedSearchCV...")
                start_time = time.time()
                random_search = RandomizedSearchCV(
                    self.estimator,
                    self.param_space,
                    n_iter=n_iter,
                    cv=cv_value,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=0
                )
                if time_budget:
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(random_search.fit, X, y)
                        try:
                            future.result(timeout=time_budget)
                        except TimeoutError:
                            logger.warning("  RandomizedSearchCV timed out")
                            results['RandomizedSearchCV'] = {
                                'best_score': float('nan'),
                                'best_params': None,
                                'time': time_budget,
                                'n_iters': 'timed_out'
                            }
                        else:
                            random_time = time.time() - start_time
                            results['RandomizedSearchCV'] = {
                                'best_score': random_search.best_score_,
                                'best_params': random_search.best_params_,
                                'time': random_time,
                                'n_iters': n_iter
                            }
                else:
                    random_search.fit(X, y)
                    random_time = time.time() - start_time
                    results['RandomizedSearchCV'] = {
                        'best_score': random_search.best_score_,
                        'best_params': random_search.best_params_,
                        'time': random_time,
                        'n_iters': n_iter
                    }
            except Exception as e:
                logger.warning(f"  Error in RandomizedSearchCV: {e}")
                results['RandomizedSearchCV'] = {
                    'best_score': float('nan'),
                    'best_params': None,
                    'time': float('nan'),
                    'n_iters': 'error'
                }

        # -----------------------
        # Run BayesSearchCV (Bayesian optimization)
        if 'bayesian' in methods:
            try:
                from skopt import BayesSearchCV
                logger.info("Running BayesSearchCV...")
                start_time = time.time()
                from skopt.space import Real, Integer, Categorical
                skopt_space = {}
                for param, value in self.param_space.items():
                    if isinstance(value, list):
                        skopt_space[param] = Categorical(value)
                    elif isinstance(value, tuple) and len(value) == 2:
                        low, high = value
                        if isinstance(low, int) and isinstance(high, int):
                            skopt_space[param] = Integer(low, high)
                        else:
                            skopt_space[param] = Real(low, high)
                bayes_search = BayesSearchCV(
                    self.estimator,
                    skopt_space,
                    n_iter=n_iter,
                    cv=cv_value,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=0
                )
                if time_budget:
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(bayes_search.fit, X, y)
                        try:
                            future.result(timeout=time_budget)
                        except TimeoutError:
                            logger.warning("  BayesSearchCV timed out")
                            results['BayesSearchCV'] = {
                                'best_score': float('nan'),
                                'best_params': None,
                                'time': time_budget,
                                'n_iters': 'timed_out'
                            }
                        else:
                            bayes_time = time.time() - start_time
                            results['BayesSearchCV'] = {
                                'best_score': bayes_search.best_score_,
                                'best_params': bayes_search.best_params_,
                                'time': bayes_time,
                                'n_iters': n_iter
                            }
                else:
                    bayes_search.fit(X, y)
                    bayes_time = time.time() - start_time
                    results['BayesSearchCV'] = {
                        'best_score': bayes_search.best_score_,
                        'best_params': bayes_search.best_params_,
                        'time': bayes_time,
                        'n_iters': n_iter
                    }
            except ImportError:
                logger.warning("  scikit-optimize not available, skipping BayesSearchCV")
            except Exception as e:
                logger.warning(f"  Error in BayesSearchCV: {e}")
                results['BayesSearchCV'] = {
                    'best_score': float('nan'),
                    'best_params': None,
                    'time': float('nan'),
                    'n_iters': 'error'
                }

        # -----------------------
        # Compute summary metrics based on a baseline
        baseline_score = None
        baseline_time = None

        # Prefer RandomizedSearchCV as the baseline if available and not timed out; otherwise, use GridSearchCV
        if ('RandomizedSearchCV' in results and 
            results['RandomizedSearchCV'].get('n_iters') != 'timed_out'):
            baseline_score = results['RandomizedSearchCV']['best_score']
            baseline_time = results['RandomizedSearchCV']['time']
        elif ('GridSearchCV' in results and 
            results['GridSearchCV'].get('n_iters') != 'timed_out'):
            baseline_score = results['GridSearchCV']['best_score']
            baseline_time = results['GridSearchCV']['time']

        if baseline_score is not None and baseline_time is not None:
            hyperoptx_score = results['HyperOptX']['best_score']
            hyperoptx_time = results['HyperOptX']['time']
            results['summary'] = {
                'score_ratio': hyperoptx_score / baseline_score if baseline_score != 0 else float('inf'),
                'speedup': baseline_time / hyperoptx_time if hyperoptx_time > 0 else float('inf')
            }

        # -----------------------
        # Print table of results
        logger.info("\nBenchmark Results:")
        logger.info("-" * 80)
        logger.info(f"{'Method':<20} {'Best Score':<15} {'Time (s)':<15} {'Iterations':<15}")
        logger.info("-" * 80)
        for method, result in results.items():
            if method != 'summary':
                best_score = result.get('best_score', float('nan'))
                time_val = result.get('time', float('nan'))
                n_iters = result.get('n_iters', 'N/A')
                logger.info(f"{method:<20} {best_score:<15.6f} {time_val:<15.2f} {str(n_iters):<15}")
        logger.info("-" * 80)
        if 'summary' in results:
            summary = results['summary']
            logger.info(f"HyperOptX score ratio: {summary['score_ratio']:.3f}x")
            logger.info(f"HyperOptX speedup: {summary['speedup']:.3f}x")

        return results



# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import r2_score, mean_squared_error
    import time
    
    # Load dataset
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter space for Random Forest
    rf_param_space = {
        'n_estimators': (10, 200),  # Numerical parameter
        'max_depth': [None, 5, 10, 15, 20],  # Categorical parameter
        'min_samples_split': (2, 10),  # Numerical parameter
        'min_samples_leaf': (1, 5),  # Numerical parameter
        'bootstrap': [True, False]  # Categorical parameter
    }
    
    # Create and run the optimizer
    logger.info("\nRunning HyperOptX for Random Forest...")
    start_time = time.time()
    
    optimizer = HyperOptX(
        estimator=RandomForestRegressor(),
        param_space=rf_param_space,
        max_iter=30,  # Reduced for demonstration
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1,
        maximize=True,  # Negative MSE, so we maximize
        ensemble_surrogate=True  # Use ensemble surrogate models
    )
    
    optimizer.fit(X_train, y_train)
    
    # Get results
    logger.info(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Best parameters: {optimizer.best_params_}")
    logger.info(f"Best CV score: {optimizer.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = optimizer.best_estimator_.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test MSE: {test_mse:.4f}")
    logger.info(f"Test R²: {test_r2:.4f}")
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        fig = optimizer.plot_optimization_history()
        plt.show()
    except ImportError:
        logger.warning("Matplotlib not available for visualization")
    
    # Compare with alternative methods
    logger.info("\nComparing with alternative optimization methods...")
    results = optimizer.benchmark_against_alternatives(
        X_train, 
        y_train,
        methods=['grid', 'random', 'bayesian'],
        n_iter=20,
        time_budget=60  # 1 minute timeout
    )
    
    # Example with ElasticNet
    logger.info("\n" + "="*80)
    logger.info("Running HyperOptX for ElasticNet...")
    
    # Define parameter space for ElasticNet
    en_param_space = {
        'alpha': (0.0001, 10.0),  # Numerical parameter (regularization strength)
        'l1_ratio': (0.0, 1.0),  # Numerical parameter (mixing parameter)
        'max_iter': (500, 2000),  # Numerical parameter
        'tol': (1e-5, 1e-3),  # Numerical parameter
        'selection': ['cyclic', 'random']  # Categorical parameter
    }
    
    elasticnet_optimizer = HyperOptX(
        estimator=ElasticNet(),
        param_space=en_param_space,
        max_iter=25,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1,
        maximize=True,
        optimization_strategy='hybrid'  # Use hybrid optimization
    )
    
    elasticnet_optimizer.fit(X_train, y_train)
    
    # Get results
    logger.info(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Best parameters: {elasticnet_optimizer.best_params_}")
    logger.info(f"Best CV score: {elasticnet_optimizer.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = elasticnet_optimizer.best_estimator_.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test MSE: {test_mse:.4f}")
    logger.info(f"Test R²: {test_r2:.4f}")
    
    # Create visualization for ElasticNet
    try:
        import matplotlib.pyplot as plt
        fig = elasticnet_optimizer.plot_optimization_history()
        plt.show()
    except ImportError:
        logger.warning("Matplotlib not available for visualization")