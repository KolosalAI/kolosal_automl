"""
Adaptive Hyperparameter Optimization
Provides adaptive hyperparameter optimization with dynamic search space adjustment.
"""

import logging
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import threading
from pathlib import Path
from collections import defaultdict, deque
import hashlib

# Core optimization libraries
try:
    import optuna
    from optuna.integration import OptunaSearchCV
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# Machine learning libraries
try:
    from sklearn.model_selection import cross_val_score
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_trial_number: int
    total_trials: int
    optimization_time: float
    convergence_iteration: Optional[int]
    search_space_adaptations: int


@dataclass
class SearchSpaceConfig:
    """Configuration for adaptive search space."""
    param_name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    initial_range: Union[Tuple[float, float], List[Any]]
    current_range: Union[Tuple[float, float], List[Any]]
    adaptation_factor: float = 0.1
    exploration_rate: float = 0.2
    lock_after_trials: int = 50


class PerformanceTracker:
    """Tracks performance patterns for adaptive optimization."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.scores = deque(maxlen=window_size)
        self.params_history = deque(maxlen=window_size)
        self.convergence_threshold = 1e-4
        self.stagnation_patience = 10
        
    def add_result(self, score: float, params: Dict[str, Any]):
        """Add a new result to the tracker."""
        self.scores.append(score)
        self.params_history.append(params.copy())
    
    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        if len(self.scores) < self.window_size:
            return False
        
        recent_scores = list(self.scores)[-self.stagnation_patience:]
        if len(recent_scores) < self.stagnation_patience:
            return False
        
        score_std = np.std(recent_scores)
        return score_std < self.convergence_threshold
    
    def get_promising_regions(self) -> Dict[str, Any]:
        """Identify promising parameter regions."""
        if len(self.scores) < 5:
            return {}
        
        # Find top performing trials
        scores_array = np.array(list(self.scores))
        top_indices = np.argsort(scores_array)[-5:]  # Top 5 trials
        
        promising_regions = {}
        for param_name in self.params_history[0].keys():
            param_values = []
            for idx in top_indices:
                if idx < len(self.params_history):
                    param_values.append(self.params_history[idx][param_name])
            
            if param_values:
                if isinstance(param_values[0], (int, float)):
                    promising_regions[param_name] = {
                        'min': min(param_values),
                        'max': max(param_values),
                        'mean': np.mean(param_values),
                        'std': np.std(param_values)
                    }
                else:
                    # Categorical parameters
                    from collections import Counter
                    value_counts = Counter(param_values)
                    promising_regions[param_name] = {
                        'most_common': value_counts.most_common(3)
                    }
        
        return promising_regions


class AdaptiveSearchSpace:
    """Manages adaptive search space for hyperparameter optimization."""
    
    def __init__(self, initial_search_space: Dict[str, SearchSpaceConfig]):
        self.search_spaces = initial_search_space.copy()
        self.adaptation_history = []
        self.lock = threading.RLock()
        
    def adapt_search_space(self, performance_tracker: PerformanceTracker) -> bool:
        """Adapt search space based on performance patterns."""
        with self.lock:
            promising_regions = performance_tracker.get_promising_regions()
            if not promising_regions:
                return False
            
            adaptations_made = False
            for param_name, config in self.search_spaces.items():
                if param_name in promising_regions and config.param_type in ['continuous', 'discrete']:
                    region_info = promising_regions[param_name]
                    
                    # Adapt continuous/discrete parameters
                    current_min, current_max = config.current_range
                    promising_min = region_info['min']
                    promising_max = region_info['max']
                    promising_mean = region_info['mean']
                    promising_std = region_info['std']
                    
                    # Calculate new range centered on promising region
                    range_size = current_max - current_min
                    new_range_size = max(
                        range_size * (1 - config.adaptation_factor),
                        promising_std * 4  # Ensure some exploration
                    )
                    
                    new_min = max(
                        current_min,
                        promising_mean - new_range_size / 2
                    )
                    new_max = min(
                        current_max,
                        promising_mean + new_range_size / 2
                    )
                    
                    if new_min != current_min or new_max != current_max:
                        config.current_range = (new_min, new_max)
                        adaptations_made = True
                        
                        self.adaptation_history.append({
                            'param_name': param_name,
                            'old_range': (current_min, current_max),
                            'new_range': (new_min, new_max),
                            'timestamp': time.time()
                        })
            
            return adaptations_made
    
    def get_optuna_search_space(self) -> Dict[str, Any]:
        """Convert to Optuna search space format."""
        optuna_space = {}
        
        for param_name, config in self.search_spaces.items():
            if config.param_type == 'continuous':
                low, high = config.current_range
                optuna_space[param_name] = optuna.distributions.FloatDistribution(low, high)
            elif config.param_type == 'discrete':
                low, high = config.current_range
                optuna_space[param_name] = optuna.distributions.IntDistribution(int(low), int(high))
            elif config.param_type == 'categorical':
                optuna_space[param_name] = optuna.distributions.CategoricalDistribution(config.current_range)
        
        return optuna_space


class AdaptiveHyperparameterOptimizer:
    """
    Adaptive hyperparameter optimizer with dynamic search space adjustment.
    
    Features:
    - Multiple optimization backends (Optuna, Hyperopt, Scikit-Optimize)
    - Adaptive search space that narrows based on promising regions
    - Early stopping and convergence detection
    - Performance pattern analysis
    - Multi-objective optimization support
    - Warm starting from previous optimizations
    """
    
    def __init__(self,
                 optimization_backend: str = 'optuna',
                 n_trials: int = 100,
                 timeout: Optional[float] = None,
                 enable_adaptive_search: bool = True,
                 adaptation_frequency: int = 10,
                 early_stopping_patience: int = 20,
                 cache_dir: Optional[str] = None):
        """
        Initialize the adaptive hyperparameter optimizer.
        
        Args:
            optimization_backend: Backend to use ('optuna', 'hyperopt', 'skopt')
            n_trials: Maximum number of trials
            timeout: Maximum optimization time in seconds
            enable_adaptive_search: Whether to enable adaptive search space
            adaptation_frequency: How often to adapt search space (in trials)
            early_stopping_patience: Patience for early stopping
            cache_dir: Directory to cache optimization results
        """
        self.optimization_backend = optimization_backend
        self.n_trials = n_trials
        self.timeout = timeout
        self.enable_adaptive_search = enable_adaptive_search
        self.adaptation_frequency = adaptation_frequency
        self.early_stopping_patience = early_stopping_patience
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.current_study = None
        self.performance_tracker = PerformanceTracker()
        self.adaptive_search_space = None
        self.optimization_history = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Validate backend availability
        self._validate_backend()
        
        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Adaptive hyperparameter optimizer initialized with {optimization_backend} backend")
    
    def _validate_backend(self):
        """Validate that the selected backend is available."""
        # Re-check imports in case they were installed after module import
        global OPTUNA_AVAILABLE, HYPEROPT_AVAILABLE, SKOPT_AVAILABLE
        
        if self.optimization_backend == 'optuna':
            try:
                import optuna
                OPTUNA_AVAILABLE = True
            except ImportError:
                OPTUNA_AVAILABLE = False
                raise ImportError("Optuna is not available. Install with: pip install optuna")
        elif self.optimization_backend == 'hyperopt':
            try:
                import hyperopt
                HYPEROPT_AVAILABLE = True
            except ImportError:
                HYPEROPT_AVAILABLE = False
                raise ImportError("Hyperopt is not available. Install with: pip install hyperopt")
        elif self.optimization_backend == 'skopt':
            try:
                import skopt
                SKOPT_AVAILABLE = True
            except ImportError:
                SKOPT_AVAILABLE = False
                raise ImportError("Scikit-Optimize is not available. Install with: pip install scikit-optimize")
    
    def optimize(self,
                 objective_function: Callable,
                 search_space: Dict[str, Any],
                 direction: str = 'maximize',
                 study_name: Optional[str] = None,
                 warm_start_trials: Optional[List[Dict[str, Any]]] = None) -> OptimizationResult:
        """
        Run adaptive hyperparameter optimization.
        
        Args:
            objective_function: Function to optimize
            search_space: Parameter search space
            direction: Optimization direction ('maximize' or 'minimize')
            study_name: Name for the optimization study
            warm_start_trials: Previous trials for warm starting
            
        Returns:
            Optimization result with best parameters and statistics
        """
        start_time = time.perf_counter()
        
        # Setup adaptive search space
        if self.enable_adaptive_search:
            adaptive_configs = self._create_adaptive_search_space(search_space)
            self.adaptive_search_space = AdaptiveSearchSpace(adaptive_configs)
        
        # Load cached results if available
        if self.cache_dir and study_name:
            cached_result = self._load_cached_result(study_name)
            if cached_result:
                self.logger.info(f"Loaded cached optimization result for {study_name}")
                return cached_result
        
        # Run optimization with selected backend
        if self.optimization_backend == 'optuna':
            result = self._optimize_with_optuna(
                objective_function, search_space, direction, study_name, warm_start_trials
            )
        elif self.optimization_backend == 'hyperopt':
            result = self._optimize_with_hyperopt(
                objective_function, search_space, direction
            )
        elif self.optimization_backend == 'skopt':
            result = self._optimize_with_skopt(
                objective_function, search_space, direction
            )
        else:
            raise ValueError(f"Unsupported optimization backend: {self.optimization_backend}")
        
        # Add timing information
        result.optimization_time = time.perf_counter() - start_time
        
        # Cache result
        if self.cache_dir and study_name:
            self._cache_result(study_name, result)
        
        # Add to history
        self.optimization_history.append(result)
        
        self.logger.info(f"Optimization completed in {result.optimization_time:.2f}s with score {result.best_score:.6f}")
        
        return result
    
    def _create_adaptive_search_space(self, search_space: Dict[str, Any]) -> Dict[str, SearchSpaceConfig]:
        """Create adaptive search space configurations."""
        adaptive_configs = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, tuple) and len(param_config) == 2:
                # Continuous parameter (min, max)
                min_val, max_val = param_config
                adaptive_configs[param_name] = SearchSpaceConfig(
                    param_name=param_name,
                    param_type='continuous',
                    initial_range=(min_val, max_val),
                    current_range=(min_val, max_val)
                )
            elif isinstance(param_config, list):
                # Categorical parameter
                adaptive_configs[param_name] = SearchSpaceConfig(
                    param_name=param_name,
                    param_type='categorical',
                    initial_range=param_config.copy(),
                    current_range=param_config.copy()
                )
            elif isinstance(param_config, dict) and 'type' in param_config:
                # Advanced parameter configuration
                param_type = param_config['type']
                if param_type in ['int', 'discrete']:
                    min_val, max_val = param_config['range']
                    adaptive_configs[param_name] = SearchSpaceConfig(
                        param_name=param_name,
                        param_type='discrete',
                        initial_range=(min_val, max_val),
                        current_range=(min_val, max_val)
                    )
                elif param_type in ['float', 'continuous']:
                    min_val, max_val = param_config['range']
                    adaptive_configs[param_name] = SearchSpaceConfig(
                        param_name=param_name,
                        param_type='continuous',
                        initial_range=(min_val, max_val),
                        current_range=(min_val, max_val)
                    )
                elif param_type == 'categorical':
                    choices = param_config['choices']
                    adaptive_configs[param_name] = SearchSpaceConfig(
                        param_name=param_name,
                        param_type='categorical',
                        initial_range=choices.copy(),
                        current_range=choices.copy()
                    )
        
        return adaptive_configs
    
    def _optimize_with_optuna(self,
                             objective_function: Callable,
                             search_space: Dict[str, Any],
                             direction: str,
                             study_name: Optional[str],
                             warm_start_trials: Optional[List[Dict[str, Any]]]) -> OptimizationResult:
        """Run optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not available")
        
        # Import Optuna components locally to ensure fresh imports
        try:
            import optuna
            from optuna.samplers import TPESampler
            from optuna.pruners import MedianPruner
        except ImportError as e:
            raise ImportError(f"Failed to import Optuna components: {e}")
        
        # Create study
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Add warm start trials
        if warm_start_trials:
            for trial_params in warm_start_trials:
                study.enqueue_trial(trial_params)
        
        # Objective function wrapper for adaptive optimization
        def adaptive_objective(trial):
            # Get parameters from current search space
            if self.enable_adaptive_search and self.adaptive_search_space:
                optuna_space = self.adaptive_search_space.get_optuna_search_space()
                params = {}
                for param_name, distribution in optuna_space.items():
                    if isinstance(distribution, optuna.distributions.FloatDistribution):
                        params[param_name] = trial.suggest_float(param_name, distribution.low, distribution.high)
                    elif isinstance(distribution, optuna.distributions.IntDistribution):
                        params[param_name] = trial.suggest_int(param_name, distribution.low, distribution.high)
                    elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                        params[param_name] = trial.suggest_categorical(param_name, distribution.choices)
            else:
                # Use original search space - convert sklearn format to Optuna format
                params = {}
                for param_name, param_config in search_space.items():
                    if isinstance(param_config, tuple) and len(param_config) == 2:
                        min_val, max_val = param_config
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                        else:
                            params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                    elif isinstance(param_config, list):
                        # Check if this is a numeric range that should be continuous
                        if (len(param_config) == 2 and 
                            all(isinstance(x, (int, float)) for x in param_config) and
                            param_config[0] < param_config[1]):
                            min_val, max_val = param_config
                            # For integer parameters, use integer suggestion
                            if (param_name in ['n_estimators', 'max_depth', 'min_samples_split', 
                                             'min_samples_leaf', 'max_features'] or
                                all(isinstance(x, int) for x in param_config)):
                                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                            else:
                                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                        else:
                            # True categorical parameter
                            params[param_name] = trial.suggest_categorical(param_name, param_config)
            
            # Ensure parameter types are correct before evaluation
            self._ensure_parameter_types(params, search_space)
            
            # Evaluate objective
            score = objective_function(params)
            
            # Track performance
            self.performance_tracker.add_result(score, params)
            
            # Adapt search space periodically
            if (self.enable_adaptive_search and 
                self.adaptive_search_space and 
                trial.number % self.adaptation_frequency == 0 and 
                trial.number > 0):
                
                adapted = self.adaptive_search_space.adapt_search_space(self.performance_tracker)
                if adapted:
                    self.logger.info(f"Adapted search space at trial {trial.number}")
            
            return score
        
        # Run optimization
        study.optimize(
            adaptive_objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            catch=(Exception,)
        )
        
        # Detect convergence
        convergence_iteration = None
        if self.performance_tracker.is_converged():
            convergence_iteration = len(study.trials) - self.performance_tracker.stagnation_patience
        
        # Count search space adaptations
        adaptations = len(self.adaptive_search_space.adaptation_history) if self.adaptive_search_space else 0
        
        # Convert parameters to correct types
        best_params = study.best_params.copy()
        self._ensure_parameter_types(best_params, search_space)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=study.best_value,
            best_trial_number=study.best_trial.number,
            total_trials=len(study.trials),
            optimization_time=0.0,  # Will be set by caller
            convergence_iteration=convergence_iteration,
            search_space_adaptations=adaptations
        )
    
    def _optimize_with_hyperopt(self,
                               objective_function: Callable,
                               search_space: Dict[str, Any],
                               direction: str) -> OptimizationResult:
        """Run optimization using Hyperopt."""
        if not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt is not available")
        
        # Convert search space to Hyperopt format
        hyperopt_space = {}
        for param_name, param_config in search_space.items():
            if isinstance(param_config, tuple) and len(param_config) == 2:
                min_val, max_val = param_config
                if isinstance(min_val, int) and isinstance(max_val, int):
                    hyperopt_space[param_name] = hp.randint(param_name, min_val, max_val + 1)
                else:
                    hyperopt_space[param_name] = hp.uniform(param_name, min_val, max_val)
            elif isinstance(param_config, list):
                hyperopt_space[param_name] = hp.choice(param_name, param_config)
        
        # Objective function wrapper
        def hyperopt_objective(params):
            score = objective_function(params)
            self.performance_tracker.add_result(score, params)
            
            # Hyperopt minimizes, so negate for maximization
            return -score if direction == 'maximize' else score
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=hyperopt_objective,
            space=hyperopt_space,
            algo=tpe.suggest,
            max_evals=self.n_trials,
            trials=trials,
            timeout=self.timeout
        )
        
        # Get best score
        best_score = -trials.best_trial['result']['loss'] if direction == 'maximize' else trials.best_trial['result']['loss']
        
        return OptimizationResult(
            best_params=space_eval(hyperopt_space, best),
            best_score=best_score,
            best_trial_number=trials.best_trial['tid'],
            total_trials=len(trials.trials),
            optimization_time=0.0,
            convergence_iteration=None,
            search_space_adaptations=0
        )
    
    def _optimize_with_skopt(self,
                            objective_function: Callable,
                            search_space: Dict[str, Any],
                            direction: str) -> OptimizationResult:
        """Run optimization using Scikit-Optimize."""
        if not SKOPT_AVAILABLE:
            raise ImportError("Scikit-Optimize is not available")
        
        # Convert search space to skopt format
        dimensions = []
        param_names = []
        
        for param_name, param_config in search_space.items():
            param_names.append(param_name)
            if isinstance(param_config, tuple) and len(param_config) == 2:
                min_val, max_val = param_config
                if isinstance(min_val, int) and isinstance(max_val, int):
                    dimensions.append(Integer(min_val, max_val, name=param_name))
                else:
                    dimensions.append(Real(min_val, max_val, name=param_name))
            elif isinstance(param_config, list):
                dimensions.append(Categorical(param_config, name=param_name))
        
        # Objective function wrapper
        @use_named_args(dimensions)
        def skopt_objective(**params):
            score = objective_function(params)
            self.performance_tracker.add_result(score, params)
            
            # Skopt minimizes, so negate for maximization
            return -score if direction == 'maximize' else score
        
        # Run optimization
        result = gp_minimize(
            func=skopt_objective,
            dimensions=dimensions,
            n_calls=self.n_trials,
            random_state=42
        )
        
        # Convert result back to named parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun if direction == 'maximize' else result.fun
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_trial_number=len(result.x_iters) - 1,
            total_trials=len(result.x_iters),
            optimization_time=0.0,
            convergence_iteration=None,
            search_space_adaptations=0
        )
    
    def _cache_result(self, study_name: str, result: OptimizationResult):
        """Cache optimization result to disk."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{study_name}_optimization_result.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {str(e)}")
    
    def _load_cached_result(self, study_name: str) -> Optional[OptimizationResult]:
        """Load cached optimization result from disk."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{study_name}_optimization_result.pkl"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cached result: {str(e)}")
            return None
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all optimizations."""
        return [asdict(result) for result in self.optimization_history]
    
    def clear_cache(self):
        """Clear all cached optimization results."""
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*_optimization_result.pkl"):
                cache_file.unlink()
    
    def _ensure_parameter_types(self, params, search_space):
        """
        Ensure parameters have the correct types for scikit-learn models.
        Convert float values to integers for parameters that must be integers.
        """
        # Parameters that must be integers
        integer_params = {
            'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
            'max_features', 'n_neighbors', 'leaf_size', 'n_components',
            'max_iter', 'degree', 'random_state', 'n_jobs'
        }
        
        for param_name, value in params.items():
            # Check if this parameter should be an integer
            if param_name in integer_params and isinstance(value, float):
                params[param_name] = int(round(value))
            
            # Also check based on the search space definition
            elif param_name in search_space:
                param_config = search_space[param_name]
                if isinstance(param_config, (list, tuple)) and len(param_config) >= 2:
                    # If the search space contains only integers, convert to int
                    sample_values = param_config[:2] if isinstance(param_config, (list, tuple)) else [param_config]
                    if all(isinstance(x, int) for x in sample_values if x is not None):
                        if isinstance(value, float):
                            params[param_name] = int(round(value))


# Global adaptive optimizer instance
_global_adaptive_optimizer = None

def get_global_adaptive_optimizer() -> AdaptiveHyperparameterOptimizer:
    """Get or create the global adaptive hyperparameter optimizer."""
    global _global_adaptive_optimizer
    if _global_adaptive_optimizer is None:
        _global_adaptive_optimizer = AdaptiveHyperparameterOptimizer()
    return _global_adaptive_optimizer
