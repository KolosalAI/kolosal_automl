import numpy as np
import os
import pickle
import joblib
import time
import logging
import json
import traceback
import gc
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
from contextlib import contextmanager
from queue import Queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
from types import MethodType
# Import scikit-learn components
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    StratifiedKFold, KFold, train_test_split, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error, roc_auc_score,
    explained_variance_score, matthews_corrcoef, confusion_matrix,
    classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

# Import new optimization modules
from .jit_compiler import JITCompiler, get_global_jit_compiler
from .mixed_precision import MixedPrecisionManager, get_global_mixed_precision_manager
from .adaptive_hyperopt import AdaptiveHyperparameterOptimizer, get_global_adaptive_optimizer
from .streaming_pipeline import StreamingDataPipeline, get_global_streaming_pipeline

# For optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import engine components if needed
# This section would need to be adjusted based on your specific project structure
from ..configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig,
    PreprocessorConfig,
    BatchProcessorConfig,
    InferenceEngineConfig,
    NormalizationType,
    ModelSelectionCriteria,
    MonitoringConfig,
    ExplainabilityConfig
)
from .inference_engine import InferenceEngine
from .batch_processor import BatchProcessor
from .data_preprocessor import DataPreprocessor
from .experiment_tracker import ExperimentTracker
from .utils import _json_safe, _scrub, _patch_pickle_for_locks

# Optional imports for optimizers
try:
    from ..optimizer.asht import ASHTOptimizer
    ASHT_AVAILABLE = True
except ImportError:
    ASHT_AVAILABLE = False

try:
    from ..optimizer.hyperoptx import HyperOptX
    HYPEROPTX_AVAILABLE = True
except ImportError:
    HYPEROPTX_AVAILABLE = False


class MLTrainingEngine:
    """
    Advanced training engine for machine learning models with optimization.
    
    Features:
    - Comprehensive model training workflow
    - Hyperparameter optimization with multiple strategies
    - Feature selection and preprocessing
    - Advanced model evaluation and metrics
    - Experiment tracking and reporting
    - Model serialization and management
    - Integration with InferenceEngine for deployment
    """
    
    VERSION = "0.1.4"
    
    def __init__(self, config: MLTrainingEngineConfig):
        """
        Initialize the training engine with the given configuration.
        
        Args:
            config: Configuration object for the training engine
        """
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('-inf')
        self.preprocessor = None
        self.feature_selector = None
        self.training_complete = False
        self._shutdown_handlers_registered = False
        
        # Set up logging using centralized configuration
        try:
            from modules.logging_config import get_logger
            self.logger = get_logger(
                name="MLTrainingEngine",
                level=getattr(logging, config.log_level),
                log_file="ml_training_engine.log",
                enable_console=True
            )
        except ImportError:
            # Fallback to basic logging if centralized logging not available
            logging.basicConfig(
                level=getattr(logging, config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger("MLTrainingEngine")
            self.logger.setLevel(getattr(logging, config.log_level))
        
        # Initialize experiment tracker if enabled
        if config.experiment_tracking:
            self.tracker = ExperimentTracker(os.path.join(config.model_path, "experiments"))
            self.logger.info(f"Experiment tracking enabled")
        else:
            self.tracker = None
            
        # Create model directory if it doesn't exist
        os.makedirs(config.model_path, exist_ok=True)
            
        # Create checkpoint directory if needed
        if config.checkpointing and not os.path.exists(config.checkpoint_path):
            os.makedirs(config.checkpoint_path, exist_ok=True)
            
        # Initialize components
        self._init_components()
        
        # Register shutdown handlers
        self._register_shutdown_handlers()
        
        # Log initialization
        self.logger.info(f"ML Training Engine v{self.VERSION} initialized with task type: {config.task_type}")
        if hasattr(config, 'use_gpu') and config.use_gpu:
            self.logger.info(f"GPU usage enabled with memory fraction: {config.gpu_memory_fraction}")
            
        # Register model types for automatic discovery
        self._register_model_types()
        
    def _init_components(self):
        """Initialize all engine components."""
        try:
            # Initialize preprocessor with the provided configuration
            if hasattr(self.config, 'preprocessing_config'):
                self.preprocessor = DataPreprocessor(self.config.preprocessing_config)
                self.logger.info("Data preprocessor initialized")
            else:
                self.logger.info("No preprocessing configuration provided")
            
            # Initialize batch processor for efficient data handling
            if hasattr(self.config, 'batch_processing_config'):
                self.batch_processor = BatchProcessor(self.config.batch_processing_config)
                self.logger.info("Batch processor initialized")
            
            # Initialize inference engine for prediction serving
            if hasattr(self.config, 'inference_config'):
                self.inference_engine = InferenceEngine(self.config.inference_config)
                self.logger.info("Inference engine initialized")
            
            # Initialize monitoring if enabled
            if hasattr(self.config, 'monitoring_config') and self.config.monitoring_config.enable_monitoring:
                self.logger.info("Model monitoring initialized")
                
            # Initialize explainability tools if enabled
            if hasattr(self.config, 'explainability_config') and self.config.explainability_config.enable_explainability:
                self.logger.info(f"Explainability initialized with method: {self.config.explainability_config.default_method}")
                
                # Check if SHAP is available
                if self.config.explainability_config.default_method == "shap" and not SHAP_AVAILABLE:
                    self.logger.warning("SHAP is configured as the default explainer but is not installed")
                    
            # Initialize optimization components
            self._init_optimization_components()
            
            # Initialize thread pool for parallel operations
            if self.config.n_jobs != 1:
                n_jobs = self.config.n_jobs if self.config.n_jobs > 0 else None  # None = all CPUs
                self.thread_pool = ThreadPoolExecutor(max_workers=n_jobs, thread_name_prefix="MLTrain")
                self.process_pool = ProcessPoolExecutor(max_workers=n_jobs)
                self.logger.debug(f"Thread and process pools initialized with {n_jobs} workers")
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            
            # Initialize fallback optimization components to ensure they exist
            try:
                if not hasattr(self, 'jit_compiler') or self.jit_compiler is None:
                    self.jit_compiler = get_global_jit_compiler()
                if not hasattr(self, 'mixed_precision_manager') or self.mixed_precision_manager is None:
                    self.mixed_precision_manager = get_global_mixed_precision_manager()
                if not hasattr(self, 'adaptive_optimizer') or self.adaptive_optimizer is None:
                    self.adaptive_optimizer = get_global_adaptive_optimizer()
                if not hasattr(self, 'streaming_pipeline') or self.streaming_pipeline is None:
                    self.streaming_pipeline = get_global_streaming_pipeline()
                self.logger.info("Fallback optimization components initialized")
            except Exception as fallback_error:
                self.logger.error(f"Failed to initialize fallback optimization components: {str(fallback_error)}")
                # Set to None if even fallback fails
                if not hasattr(self, 'jit_compiler'):
                    self.jit_compiler = None
                if not hasattr(self, 'mixed_precision_manager'):
                    self.mixed_precision_manager = None
                if not hasattr(self, 'adaptive_optimizer'):
                    self.adaptive_optimizer = None
                if not hasattr(self, 'streaming_pipeline'):
                    self.streaming_pipeline = None
    
    def _register_shutdown_handlers(self):
        """Register handlers for proper cleanup during shutdown."""
        if not self._shutdown_handlers_registered:
            import atexit
            import signal
            
            # Register atexit handler
            atexit.register(self._cleanup_on_shutdown)
            
            # Register signal handlers
            try:
                signal.signal(signal.SIGTERM, self._signal_handler)
                signal.signal(signal.SIGINT, self._signal_handler)
            except (ValueError, AttributeError):
                # This can happen in environments where signal is not available
                self.logger.debug("Could not register signal handlers")
                
            self._shutdown_handlers_registered = True
            
    def _signal_handler(self, signum, frame):
        """Handle signals for graceful shutdown."""
        self._safe_log("info", f"Received signal {signum}, cleaning up...")
        self._cleanup_on_shutdown()
        
    def _safe_log(self, level, message):
        """Safely log a message, avoiding I/O errors during shutdown."""
        try:
            if hasattr(self, 'logger') and self.logger:
                # Check if logger handlers are still valid
                for handler in self.logger.handlers:
                    if hasattr(handler, 'stream') and hasattr(handler.stream, 'closed'):
                        if handler.stream.closed:
                            return  # Skip logging if stream is closed
                
                getattr(self.logger, level)(message)
        except (ValueError, OSError, AttributeError):
            # Silently ignore logging errors during shutdown
            pass
    
    def _cleanup_on_shutdown(self):
        """Perform cleanup operations before shutdown."""
        if hasattr(self, 'config') and hasattr(self.config, 'auto_save_on_shutdown') and self.config.auto_save_on_shutdown and hasattr(self, 'best_model') and self.best_model is not None:
            self._safe_log("info", "Auto-saving best model before shutdown")
            try:
                self.save_model(self.best_model_name)
            except Exception as e:
                self._safe_log("error", f"Error saving model during shutdown: {str(e)}")
                
        # Save experiment state if requested
        if hasattr(self, 'config') and hasattr(self.config, 'save_state_on_shutdown') and self.config.save_state_on_shutdown and hasattr(self, 'tracker') and self.tracker:
            try:
                self._safe_log("info", "Saving experiment state before shutdown")
                if hasattr(self.tracker, 'end_experiment'):
                    self.tracker.end_experiment()
            except Exception as e:
                self._safe_log("error", f"Error saving experiment state during shutdown: {str(e)}")
                
        # Signal components to clean up
        try:
            if hasattr(self, 'inference_engine'):
                self.inference_engine.shutdown()
                
            if hasattr(self, 'batch_processor'):
                self.batch_processor.stop()
        except Exception as e:
            self.logger.error(f"Error shutting down components: {str(e)}")
        
        # Clean up logging resources
        try:
            from modules.logging_config import cleanup_logging
            cleanup_logging()
        except Exception as e:
            # Use print since logging might be shutting down
            print(f"Error cleaning up logging: {e}")
            
        self.logger.info("Cleanup complete")
        
    def _register_model_types(self):
        """Register built-in model types for auto discovery."""
        self._model_registry = {}
        
        # Register common sklearn model types
        try:
            from sklearn.ensemble import (
                RandomForestClassifier, RandomForestRegressor,
                GradientBoostingClassifier, GradientBoostingRegressor,
                AdaBoostClassifier, AdaBoostRegressor,
                StackingClassifier, StackingRegressor,
                VotingClassifier, VotingRegressor
            )
            from sklearn.linear_model import (
                LogisticRegression, LinearRegression,
                Ridge, Lasso, ElasticNet,
                SGDClassifier, SGDRegressor
            )
            from sklearn.svm import SVC, SVR
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.naive_bayes import GaussianNB, MultinomialNB
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            
            # Classification models
            self._model_registry["classification"] = {
                "random_forest": RandomForestClassifier,
                "gradient_boosting": GradientBoostingClassifier,
                "logistic_regression": LogisticRegression,
                "svm": SVC,
                "knn": KNeighborsClassifier,
                "decision_tree": DecisionTreeClassifier,
                "adaboost": AdaBoostClassifier,
                "sgd": SGDClassifier,
                "naive_bayes": GaussianNB,
                "multinomial_nb": MultinomialNB,
                "mlp": MLPClassifier,
                "stacking": StackingClassifier,
                "voting": VotingClassifier
            }
            
            # Regression models
            self._model_registry["regression"] = {
                "random_forest": RandomForestRegressor,
                "gradient_boosting": GradientBoostingRegressor,
                "linear_regression": LinearRegression,
                "ridge": Ridge,
                "lasso": Lasso,
                "elastic_net": ElasticNet,
                "svr": SVR,
                "knn": KNeighborsRegressor,
                "decision_tree": DecisionTreeRegressor,
                "adaboost": AdaBoostRegressor,
                "sgd": SGDRegressor,
                "mlp": MLPRegressor,
                "stacking": StackingRegressor,
                "voting": VotingRegressor
            }
            
            # Try to register other model types if available
            try:
                import xgboost as xgb
                self._model_registry["classification"]["xgboost"] = xgb.XGBClassifier
                self._model_registry["regression"]["xgboost"] = xgb.XGBRegressor
            except ImportError:
                self.logger.debug("XGBoost not available")
                
            try:
                import lightgbm as lgb
                self._model_registry["classification"]["lightgbm"] = lgb.LGBMClassifier
                self._model_registry["regression"]["lightgbm"] = lgb.LGBMRegressor
            except ImportError:
                self.logger.debug("LightGBM not available")
                
            try:
                import catboost as cb
                self._model_registry["classification"]["catboost"] = cb.CatBoostClassifier
                self._model_registry["regression"]["catboost"] = cb.CatBoostRegressor
            except ImportError:
                self.logger.debug("CatBoost not available")
                
        except ImportError as e:
            self.logger.warning(f"Failed to register model types: {str(e)}")
            
        self.logger.info("Model types registered for auto discovery")
        self.logger.debug(f"Registered {len(self._model_registry.get('classification', {}))} classification models and "
                         f"{len(self._model_registry.get('regression', {}))} regression models")
                         
    def _get_feature_selector(self, X, y):
        """
        Get appropriate feature selector based on configuration.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Configured feature selector or None
        """
        if not hasattr(self.config, 'feature_selection') or not self.config.feature_selection:
            return None
        
        # For classification tasks
        if self.config.task_type == TaskType.CLASSIFICATION:
            if hasattr(self.config, 'feature_selection_method') and self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_classif, k=self.config.feature_selection_k)
            elif hasattr(self.config, 'feature_selection_method') and self.config.feature_selection_method == "chi2":
                from sklearn.feature_selection import chi2
                # Ensure non-negative values for chi2
                if isinstance(X, np.ndarray) and np.any(X < 0):
                    self.logger.warning("Chi2 requires non-negative values. Using f_classif instead.")
                    selector = SelectKBest(f_classif, k=self.config.feature_selection_k)
                else:
                    selector = SelectKBest(chi2, k=self.config.feature_selection_k)
            elif hasattr(self.config, 'feature_selection_method') and self.config.feature_selection_method == "rfe":
                # Recursive feature elimination requires a base estimator
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestClassifier
                base_estimator = RandomForestClassifier(n_estimators=10, random_state=self.config.random_state)
                selector = RFE(base_estimator, n_features_to_select=self.config.feature_selection_k)
            else:
                # Default to f_classif
                selector = SelectKBest(f_classif, k=self.config.feature_selection_k)
        
        # For regression tasks
        elif self.config.task_type == TaskType.REGRESSION:
            from sklearn.feature_selection import mutual_info_regression, f_regression
            if hasattr(self.config, 'feature_selection_method') and self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_regression, k=self.config.feature_selection_k)
            elif hasattr(self.config, 'feature_selection_method') and self.config.feature_selection_method == "rfe":
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestRegressor
                base_estimator = RandomForestRegressor(n_estimators=10, random_state=self.config.random_state)
                selector = RFE(base_estimator, n_features_to_select=self.config.feature_selection_k)
            else:
                selector = SelectKBest(f_regression, k=self.config.feature_selection_k)
        
        # For other task types, default to mutual information
        else:
            from sklearn.feature_selection import mutual_info_regression
            selector = SelectKBest(mutual_info_regression, k=self.config.feature_selection_k)
                
        # If k is not specified, select based on percentile or threshold
        if hasattr(self.config, 'feature_selection_k') and self.config.feature_selection_k is None:
            # Use all features but get their scores for later filtering
            selector.k = 'all'
            
        return selector
        
    def _create_pipeline(self, model):
        """
        Create a pipeline with preprocessing and model.
        
        Args:
            model: The model estimator
            
        Returns:
            Configured scikit-learn Pipeline
        """
        steps = []
        
        # Add preprocessor if available
        if self.preprocessor and hasattr(self.preprocessor, 'transform'):
            steps.append(('preprocessor', self.preprocessor))
            
        # Add feature selector if available
        if self.feature_selector:
            steps.append(('feature_selector', self.feature_selector))
            
        # Add final model
        steps.append(('model', model))
        
        return Pipeline(steps)
    
    def _get_cv_splitter(self, y=None):
        """
        Get appropriate cross-validation splitter based on task type.
        
        Args:
            y: Target variable for stratification
            
        Returns:
            Configured cross-validation splitter
        """
        if self.config.task_type == TaskType.CLASSIFICATION and hasattr(self.config, 'stratify') and self.config.stratify and y is not None:
            return StratifiedKFold(
                n_splits=self.config.cv_folds, 
                shuffle=True, 
                random_state=self.config.random_state
            )
        else:
            return KFold(
                n_splits=self.config.cv_folds, 
                shuffle=True, 
                random_state=self.config.random_state
            )

    def _get_optimization_search(self, model, param_grid):
        """
        Get the appropriate hyperparameter search based on strategy.
        
        Args:
            model: The model estimator
            param_grid: Parameter grid for optimization
            
        Returns:
            Configured hyperparameter optimization object
        """
        cv = self._get_cv_splitter()
        scoring = self._get_scoring_metric()
        
        if not hasattr(self.config, 'optimization_strategy'):
            # Default to random search if not specified
            return RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=10,  # Default value
                cv=cv,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                random_state=self.config.random_state,
                scoring=scoring,
                refit=True,
                return_train_score=True
            )
        
        if self.config.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
            return GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                scoring=scoring,
                refit=True,
                return_train_score=True
            )
        elif self.config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
            return RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=self.config.optimization_iterations,
                cv=cv,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                random_state=self.config.random_state,
                scoring=scoring,
                refit=True,
                return_train_score=True
            )
        elif self.config.optimization_strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
            # Check if scikit-optimize is available
            try:
                from skopt import BayesSearchCV
                return BayesSearchCV(
                    estimator=model,
                    search_spaces=param_grid,
                    n_iter=self.config.optimization_iterations,
                    cv=cv,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state,
                    scoring=scoring,
                    refit=True,
                    return_train_score=True
                )
            except ImportError:
                self.logger.warning("scikit-optimize not installed. Falling back to RandomizedSearchCV.")
                return RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=self.config.optimization_iterations,
                    cv=cv,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state,
                    scoring=scoring,
                    refit=True,
                    return_train_score=True
                )
        elif self.config.optimization_strategy == OptimizationStrategy.ASHT:
            # Use ASHT optimizer if available
            if ASHT_AVAILABLE:
                return ASHTOptimizer(
                    estimator=model,
                    param_space=param_grid,
                    max_iter=self.config.optimization_iterations,
                    cv=self.config.cv_folds,
                    scoring=scoring,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose
                )
            else:
                self.logger.warning("ASHT optimizer not available. Falling back to RandomizedSearchCV.")
                return RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=self.config.optimization_iterations,
                    cv=cv,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state,
                    scoring=scoring,
                    refit=True,
                    return_train_score=True
                )
        elif self.config.optimization_strategy == OptimizationStrategy.HYPERX:
            # Use HyperOptX if available
            if HYPEROPTX_AVAILABLE:
                return HyperOptX(
                    estimator=model,
                    param_space=param_grid,
                    max_iter=self.config.optimization_iterations,
                    cv=self.config.cv_folds,
                    scoring=scoring,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose
                )
            else:
                self.logger.warning("HyperOptX optimizer not available. Falling back to RandomizedSearchCV.")
                return RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=self.config.optimization_iterations,
                    cv=cv,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state,
                    scoring=scoring,
                    refit=True,
                    return_train_score=True
                )
        elif self.config.optimization_strategy == OptimizationStrategy.ADAPTIVE:
            # Use adaptive hyperparameter optimization
            try:
                from .adaptive_hyperopt import OptimizationResult
                
                # Convert param_grid to adaptive format
                adaptive_search_space = {}
                for param_name, param_values in param_grid.items():
                    if isinstance(param_values, list):
                        if all(isinstance(v, (int, float)) for v in param_values):
                            # Numeric parameter - convert to range
                            adaptive_search_space[param_name] = (min(param_values), max(param_values))
                        else:
                            # Categorical parameter
                            adaptive_search_space[param_name] = param_values
                    else:
                        # Assume it's already in the right format
                        adaptive_search_space[param_name] = param_values
                
                # Create objective function for sklearn model
                def sklearn_objective(params):
                    try:
                        # Create model with current parameters
                        temp_model = model.set_params(**params)
                        
                        # Apply JIT compilation if enabled
                        if hasattr(self, 'jit_compiler'):
                            scores = self.jit_compiler.compile_if_hot(
                                cross_val_score,
                                temp_model, self._current_X, self._current_y,
                                cv=cv, scoring=scoring, n_jobs=1  # Use single job for adaptive opt
                            )
                        else:
                            scores = cross_val_score(temp_model, self._current_X, self._current_y,
                                                   cv=cv, scoring=scoring, n_jobs=1)
                        
                        return np.mean(scores)
                    except Exception as e:
                        self.logger.warning(f"Error in objective function: {str(e)}")
                        return float('-inf')  # Return very low score on error
                
                # Run adaptive optimization
                result = self.adaptive_optimizer.optimize(
                    objective_function=sklearn_objective,
                    search_space=adaptive_search_space,
                    direction='maximize',
                    study_name=f"{model.__class__.__name__}_{int(time.time())}"
                )
                
                # Create a mock sklearn search object for compatibility
                class AdaptiveSearchResult:
                    def __init__(self, best_params, best_score):
                        self.best_params_ = best_params
                        self.best_score_ = best_score
                        self.best_estimator_ = model.set_params(**best_params)
                        self.cv_results_ = {'mean_test_score': [best_score]}
                    
                    def fit(self, X, y):
                        # Already optimized, just fit the best estimator
                        self.best_estimator_.fit(X, y)
                        return self
                
                return AdaptiveSearchResult(result.best_params, result.best_score)
                
            except Exception as e:
                self.logger.warning(f"Adaptive optimization failed: {str(e)}. Falling back to RandomizedSearchCV.")
                return RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=self.config.optimization_iterations,
                    cv=cv,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state,
                    scoring=scoring,
                    refit=True,
                    return_train_score=True
                )
        elif self.config.optimization_strategy == OptimizationStrategy.OPTUNA:
            # Check if Optuna is available
            try:
                from sklearn.model_selection import BaseSearchCV
                import optuna
                from optuna.integration import OptunaSearchCV
                
                return OptunaSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_trials=self.config.optimization_iterations,
                    cv=cv,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state,
                    scoring=scoring,
                    refit=True,
                    return_train_score=True
                )
            except ImportError:
                self.logger.warning("Optuna not installed. Falling back to RandomizedSearchCV.")
                return RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=self.config.optimization_iterations,
                    cv=cv,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state,
                    scoring=scoring,
                    refit=True,
                    return_train_score=True
                )
        else:
            self.logger.warning(f"Unsupported optimization strategy: {self.config.optimization_strategy}. Using RandomizedSearchCV.")
            return RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=self.config.optimization_iterations if hasattr(self.config, 'optimization_iterations') else 10,
                cv=cv,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                random_state=self.config.random_state,
                scoring=scoring,
                refit=True,
                return_train_score=True
            )
    
    def _get_scoring_metric(self):
        """
        Get appropriate scoring metric based on task type and configuration.
        
        Returns:
            String identifier of the sklearn scoring metric
        """
        # If a specific optimization metric is set in the config, use it
        if hasattr(self.config, 'optimization_metric') and self.config.optimization_metric:
            if isinstance(self.config.optimization_metric, ModelSelectionCriteria):
                metric_name = self.config.optimization_metric.value
            else:
                metric_name = self.config.optimization_metric
                
            # Map metric names to sklearn scoring strings
            metric_mapping = {
                'accuracy': 'accuracy',
                'f1': 'f1_weighted',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted',
                'roc_auc': 'roc_auc',
                'matthews_correlation': 'matthews_corrcoef',
                'rmse': 'neg_root_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'r2': 'r2',
                'explained_variance': 'explained_variance',
                'silhouette': 'silhouette'
            }
            
            if metric_name.lower() in metric_mapping:
                return metric_mapping[metric_name.lower()]
            else:
                self.logger.warning(f"Unrecognized metric: {metric_name}. Using default for task type.")
        
        # Default metrics based on task type
        if self.config.task_type == TaskType.CLASSIFICATION:
            if hasattr(self.config, 'model_selection_criteria'):
                if self.config.model_selection_criteria == ModelSelectionCriteria.F1:
                    return "f1_weighted"
                elif self.config.model_selection_criteria == ModelSelectionCriteria.PRECISION:
                    return "precision_weighted"
                elif self.config.model_selection_criteria == ModelSelectionCriteria.RECALL:
                    return "recall_weighted"
                elif self.config.model_selection_criteria == ModelSelectionCriteria.ROC_AUC:
                    return "roc_auc"
            return "f1_weighted"  # Default for classification
            
        elif self.config.task_type == TaskType.REGRESSION:
            if hasattr(self.config, 'model_selection_criteria'):
                if self.config.model_selection_criteria == ModelSelectionCriteria.MEAN_ABSOLUTE_ERROR:
                    return "neg_mean_absolute_error"
                elif self.config.model_selection_criteria == ModelSelectionCriteria.R2:
                    return "r2"
                elif self.config.model_selection_criteria == ModelSelectionCriteria.EXPLAINED_VARIANCE:
                    return "explained_variance"
            return "neg_mean_squared_error"  # Default for regression
            
        elif self.config.task_type == TaskType.CLUSTERING:
            return "silhouette"
            
        elif self.config.task_type == TaskType.TIME_SERIES:
            # For time series, typically use MSE or MAE
            return "neg_mean_squared_error"
            
        # Default fallback
        return "accuracy"
        
    def _extract_feature_names(self, X):
        """
        Extract feature names from input data.
        
        Args:
            X: Input data (DataFrame, array, etc.)
            
        Returns:
            List of feature names
        """
        # Try to get feature names from DataFrame
        if hasattr(X, 'columns'):
            return list(X.columns)
            
        # Try to get feature names from numpy array
        if hasattr(X, 'dtype') and hasattr(X.dtype, 'names') and X.dtype.names:
            return list(X.dtype.names)
            
        # Generate default feature names
        if hasattr(X, 'shape') and len(X.shape) > 1:
            return [f"feature_{i}" for i in range(X.shape[1])]
            
        # Fallback
        return [f"feature_{i}" for i in range(10)]  # Assume 10 features as a safe fallback
    
    def _get_feature_importance(self, model):
        """
        Extract feature importance from the model.
        
        Args:
            model: Trained model
            
        Returns:
            Array of feature importance values or None
        """
        # Try different attributes that might contain feature importance
        for attr in ['feature_importances_', 'coef_', 'feature_importance_']:
            if hasattr(model, attr):
                importance = getattr(model, attr)
                
                # Convert to numpy array if it's not already
                if not isinstance(importance, np.ndarray):
                    try:
                        importance = np.array(importance)
                    except Exception:
                        continue
                
                # Handle different shapes
                if attr == 'coef_':
                    if importance.ndim > 1:
                        # For multi-class models, take the mean absolute coefficient
                        importance = np.mean(np.abs(importance), axis=0)
                    elif importance.ndim == 1:
                        # For binary classification or regression, take absolute values
                        importance = np.abs(importance)
                
                # Normalize importance scores to sum to 1
                if importance.sum() != 0:
                    importance = importance / importance.sum()
                    
                return importance
                
        # Try permutation importance for models without built-in feature importance
        if hasattr(self.config, 'compute_permutation_importance') and self.config.compute_permutation_importance and hasattr(model, 'predict'):
            try:
                from sklearn.inspection import permutation_importance
                if hasattr(self, '_last_X_train') and hasattr(self, '_last_y_train'):
                    # Using cached data if available
                    result = permutation_importance(
                        model, self._last_X_train, self._last_y_train,
                        n_repeats=5, random_state=self.config.random_state
                    )
                    return result.importances_mean
            except Exception as e:
                self.logger.warning(f"Permutation importance calculation failed: {str(e)}")
                    
        # If we reach here, model doesn't have built-in feature importance
        self.logger.warning("Model doesn't provide feature importance.")
        return None
        
    def _get_default_param_grid(self, model):
        """Return a reasonable, *small* default grid for quick exploration."""
        model_name = model.__class__.__name__.lower()

        default_grids = {
            "randomforest": dict(
                n_estimators=[100, 300],
                max_depth=[None, 10, 30],
                min_samples_split=[2, 5],
            ),
            "gradientboosting": dict(
                n_estimators=[100, 300],
                learning_rate=[0.03, 0.1],
                max_depth=[3, 5],
            ),
            "xgb": dict(
                n_estimators=[200, 400],
                learning_rate=[0.03, 0.1],
                max_depth=[4, 6],
                subsample=[0.8, 1.0],
                colsample_bytree=[0.8, 1.0],
            ),
            "lgbm": dict(
                n_estimators=[200, 400],
                learning_rate=[0.03, 0.1],
                max_depth=[-1, 6],
                num_leaves=[31, 63],
            ),
            "logistic": dict(
                C=[0.01, 0.1, 1, 10],
                solver=["lbfgs", "liblinear"],
                penalty=["l2"],          # 'none' not accepted by liblinear
            ),
            "svc": dict(
                C=[0.1, 1, 10],
                kernel=["rbf", "linear"],
                gamma=["scale", 0.1],
            ),
            "svr": dict(
                C=[0.1, 1, 10],
                kernel=["rbf", "linear"],
                gamma=["scale", 0.1],
            ),
            "kneighbors": dict(
                n_neighbors=[3, 5, 11],
                weights=["uniform", "distance"],
                p=[1, 2],
            ),
            "decisiontree": dict(
                max_depth=[None, 10, 30],
                min_samples_split=[2, 5],
                min_samples_leaf=[1, 3],
            ),
            "ridge": dict(alpha=[0.1, 1.0, 10.0]),
            "lasso": dict(alpha=[0.001, 0.01, 0.1]),
            "elasticnet": dict(
                alpha=[0.001, 0.01, 0.1],
                l1_ratio=[0.2, 0.5, 0.8],
            ),
        }

        for key, grid in default_grids.items():
            if key in model_name:
                return grid

        # Generic fallback: take a *very* small random subset of numeric hyper‑parameters
        try:
            numeric_params = {k: v for k, v in model.get_params().items()
                            if isinstance(v, (int, float)) and k != "random_state"}
            return {k: [v, v * 2] if isinstance(v, (int, float)) and v else [v]
                    for k, v in list(numeric_params.items())[:4]}
        except Exception:
            self.logger.warning("No default grid found; using empty grid")
            return {}
    
    def _compare_metrics(self, new_metric, current_best):
        """
        Compare metrics to determine if new model is better.
        
        Args:
            new_metric: Metric value of new model
            current_best: Current best metric value
            
        Returns:
            True if new model is better, False otherwise
        """
        # Handle regression metrics that are better when lower
        if self.config.task_type == TaskType.REGRESSION:
            if hasattr(self.config, 'model_selection_criteria'):
                criteria = self.config.model_selection_criteria
                
                if isinstance(criteria, ModelSelectionCriteria):
                    key = criteria.value
                else:
                    key = criteria
                    
                # For these metrics, lower is better
                if key in ["rmse", "mse", "mae", "mape", "median_absolute_error"]:
                    return new_metric < current_best
        
        # For most metrics, higher is better
        return new_metric > current_best
    
    def _get_best_metric_value(self, metrics):
        """Extract the best metric value based on task type."""
        if not metrics:
            return 0
            
        # Determine which metric to use based on task and config
        if hasattr(self.config, 'model_selection_criteria'):
            criteria = self.config.model_selection_criteria
            
            if isinstance(criteria, ModelSelectionCriteria):
                key = criteria.value
            else:
                key = criteria
                
            # Check if the metric exists
            if key in metrics:
                return metrics[key]
                
        # Default metrics by task
        if self.config.task_type == TaskType.CLASSIFICATION:
            if "f1" in metrics:
                return metrics["f1"]
            elif "accuracy" in metrics:
                return metrics["accuracy"]
        elif self.config.task_type == TaskType.REGRESSION:
            # For regression, lower RMSE/MSE is better, but higher R2 is better
            if "r2" in metrics:
                return metrics["r2"]
            elif "rmse" in metrics:
                return -metrics["rmse"]  # Negative because lower is better
            elif "mse" in metrics:
                return -metrics["mse"]  # Negative because lower is better
                
        # Default to the first metric
        return next(iter(metrics.values()))
    
    def train_model(self, X, y, model_type: str = None, custom_model=None, 
                param_grid: Dict = None, model_name: str = None, X_val=None, y_val=None):
        """
        Optimized: Train a machine learning model with hyperparameter optimization, focusing on speed and memory usage.
        """
        start_time = time.time()
        gc.collect()  # Clean up memory before starting

        # Store current training data for adaptive optimization
        self._current_X = X
        self._current_y = y
        
        # Handle large datasets with streaming pipeline
        use_streaming = (hasattr(self.config, 'enable_streaming') and self.config.enable_streaming and 
                        isinstance(X, pd.DataFrame) and len(X) > getattr(self.config, 'streaming_threshold', 10000) and
                        hasattr(self, 'streaming_pipeline') and self.streaming_pipeline is not None)
        
        if use_streaming:
            self.logger.info(f"Using streaming pipeline for large dataset with {len(X)} samples")
            
            # Use streaming preprocessing if enabled
            def preprocess_chunk(chunk):
                if self.preprocessor:
                    return self.preprocessor.fit_transform(chunk)
                return chunk
            
            # Process data in streaming fashion
            try:
                with self.streaming_pipeline.monitoring_context():
                    processed_chunks = list(self.streaming_pipeline.process_dataframe_stream(
                        X, preprocess_chunk
                    ))
                    X_processed = pd.concat(processed_chunks, ignore_index=True)
            except Exception as e:
                self.logger.warning(f"Streaming processing failed: {str(e)}. Using regular processing.")
                X_processed = X
        else:
            X_processed = X
        
        # Apply mixed precision optimization for numerical data
        if (hasattr(self.config, 'enable_mixed_precision') and self.config.enable_mixed_precision and
            hasattr(self, 'mixed_precision_manager') and self.mixed_precision_manager is not None):
            if isinstance(X_processed, np.ndarray):
                try:
                    X_processed = self.mixed_precision_manager.optimize_numpy_precision(X_processed)
                    self.logger.info("Applied mixed precision optimization to training data")
                except Exception as e:
                    self.logger.warning(f"Mixed precision optimization failed: {str(e)}")

        # Extract feature names if available
        feature_names = self._extract_feature_names(X_processed)
        self._last_feature_names = feature_names

        # Determine model type based on task
        task_key = self.config.task_type.value

        # Validate inputs
        if custom_model is None and model_type is None:
            if task_key == "classification":
                model_type = "random_forest"
            elif task_key == "regression":
                model_type = "random_forest"
            else:
                raise ValueError(f"Please specify model_type or custom_model for task type: {task_key}")

        # Set model name
        if model_name is None:
            if custom_model is not None:
                model_name = f"{custom_model.__class__.__name__}_{int(time.time())}"
            else:
                model_name = f"{model_type}_{int(time.time())}"

        # Get model class
        if custom_model is not None:
            model = custom_model
            self.logger.info(f"Using custom model: {model.__class__.__name__}")
        else:
            if task_key not in self._model_registry:
                raise ValueError(f"No models registered for task type: {task_key}")
            if model_type not in self._model_registry[task_key]:
                raise ValueError(f"Model type '{model_type}' not found for {task_key}. " 
                            f"Available models: {', '.join(self._model_registry[task_key].keys())}")
            model_class = self._model_registry[task_key][model_type]
            # Set n_jobs=-1 if supported for parallelism
            default_params = {"random_state": self.config.random_state} if hasattr(model_class, "random_state") else {}
            if hasattr(model_class, "n_jobs"):
                default_params["n_jobs"] = -1
            model = model_class(**default_params)
            self.logger.info(f"Initialized {model_type} model for {task_key}")

        # Get default parameter grid if not provided
        if param_grid is None:
            param_grid = self._get_default_param_grid(model)

        # Start experiment tracking if enabled
        if self.tracker:
            model_info = {
                "model_type": model_type or model.__class__.__name__,
                "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "task_type": task_key
            }
            self.tracker.start_experiment(
                config=self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
                model_info=model_info
            )

        # Set up the feature selector if enabled
        if hasattr(self.config, 'feature_selection') and self.config.feature_selection:
            self.feature_selector = self._get_feature_selector(X, y)
            if self.feature_selector:
                self.logger.info(f"Feature selection enabled: {getattr(self.config, 'feature_selection_method', 'default')}")

        # Create training pipeline
        pipeline = self._create_pipeline(model)

        # Split data for validation if not provided
        if (X_val is None or y_val is None) and hasattr(self.config, 'test_size') and self.config.test_size > 0:
            split_kwargs = dict(test_size=self.config.test_size, random_state=self.config.random_state)
            if self.config.task_type == TaskType.CLASSIFICATION and getattr(self.config, 'stratify', False):
                split_kwargs['stratify'] = y
            X_train, X_val, y_train, y_val = train_test_split(X, y, **split_kwargs)
            self.logger.info(f"Data split: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
        else:
            X_train, y_train = X, y
            self.logger.info(f"Using provided validation data: {X_val.shape[0]} validation samples" if X_val is not None else "No validation data")

        # Convert data to numpy arrays (no copy if possible)
        def to_numpy_safe(arr):
            if hasattr(arr, 'to_numpy'):
                return arr.to_numpy(copy=False)
            elif hasattr(arr, 'values'):
                return arr.values
            return arr
        X_train = to_numpy_safe(X_train)
        y_train = to_numpy_safe(y_train)
        X_val = to_numpy_safe(X_val) if X_val is not None else None
        y_val = to_numpy_safe(y_val) if y_val is not None else None

        # Fit preprocessor if available
        if self.preprocessor:
            try:
                self.logger.info("Fitting preprocessor...")
                self.preprocessor.fit(X_train, y_train)
                X_train_processed = self.preprocessor.transform(X_train)
                X_val_processed = self.preprocessor.transform(X_val) if X_val is not None else None
                self.logger.info("Preprocessor fitted successfully")
            except Exception as e:
                self.logger.error(f"Error fitting preprocessor: {str(e)}")
                if getattr(self.config, 'debug_mode', False):
                    self.logger.error(traceback.format_exc())
                X_train_processed = X_train
                X_val_processed = X_val
        else:
            X_train_processed = X_train
            X_val_processed = X_val
        del X_train, X_val  # Free memory
        gc.collect()

        # Fit feature selector if available
        selected_feature_names = feature_names
        if self.feature_selector:
            try:
                self.logger.info("Fitting feature selector...")
                self.feature_selector.fit(X_train_processed, y_train)
                X_train_processed = self.feature_selector.transform(X_train_processed)
                if X_val_processed is not None:
                    X_val_processed = self.feature_selector.transform(X_val_processed)
                if hasattr(self.feature_selector, 'get_support'):
                    feature_indices = self.feature_selector.get_support(indices=True)
                    selected_feature_names = [feature_names[i] for i in feature_indices] if feature_names else None
                    self.logger.info(f"Selected {len(feature_indices)} features out of {len(feature_names)}")
                    if self.tracker and selected_feature_names:
                        self.tracker.log_metrics({
                            "selected_feature_count": len(feature_indices),
                            "total_feature_count": len(feature_names)
                        }, step="feature_selection")
            except Exception as e:
                self.logger.error(f"Error fitting feature selector: {str(e)}")
                if getattr(self.config, 'debug_mode', False):
                    self.logger.error(traceback.format_exc())
                selected_feature_names = feature_names
        # del y_train  # Free memory (moved to after training)
        gc.collect()

        # Configure hyperparameter optimization
        if param_grid and len(param_grid) > 0:
            self.logger.info(f"Starting hyperparameter optimization with {getattr(self.config, 'optimization_strategy', 'default strategy')}")
            try:
                optimizer = self._get_optimization_search(model, param_grid)
                if hasattr(optimizer, 'n_jobs'):
                    optimizer.n_jobs = -1
                if self.tracker:
                    self.tracker.log_metrics({
                        "param_combinations": getattr(optimizer, 'n_iter', 'grid'),
                        "cv_folds": self.config.cv_folds,
                        "scoring": str(getattr(optimizer, 'scoring', 'default'))
                    }, step="optimization_setup")
                fit_params = {}
                if getattr(self.config, 'early_stopping', False) and hasattr(model, 'early_stopping') and X_val_processed is not None:
                    fit_params['eval_set'] = [(X_train_processed, y_train), (X_val_processed, y_val)]
                    fit_params['early_stopping_rounds'] = getattr(self.config, 'early_stopping_rounds', 10)
                    fit_params['verbose'] = bool(self.config.verbose)
                optimization_start = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    optimizer.fit(X_train_processed, y_train, **fit_params)
                optimization_time = time.time() - optimization_start
                best_model = getattr(optimizer, 'best_estimator_', optimizer.estimator)
                best_params = getattr(optimizer, 'best_params_', optimizer.get_params())
                if hasattr(optimizer, 'cv_results_'):
                    cv_results = optimizer.cv_results_
                    mean_test_score = np.mean(cv_results['mean_test_score'])
                    std_test_score = np.mean(cv_results['std_test_score'])
                    if self.tracker:
                        self.tracker.log_metrics({
                            "mean_cv_score": mean_test_score,
                            "std_cv_score": std_test_score,
                            "optimization_time": optimization_time
                        }, step="optimization")
                    self.logger.info(f"Best CV score: {mean_test_score:.4f} ± {std_test_score:.4f}")
                    self.logger.info(f"Best parameters: {best_params}")
                best_model_info = {
                    "model": best_model,
                    "params": best_params,
                    "feature_names": selected_feature_names,
                    "training_time": time.time() - start_time,
                    "metrics": {},
                    "feature_importance": None
                }
            except Exception as e:
                self.logger.error(f"Error during hyperparameter optimization: {str(e)}")
                if getattr(self.config, 'debug_mode', False):
                    self.logger.error(traceback.format_exc())
                self.logger.info("Falling back to basic training without optimization")
                model.fit(X_train_processed, y_train)
                best_model_info = {
                    "model": model,
                    "params": model.get_params(),
                    "feature_names": selected_feature_names,
                    "training_time": time.time() - start_time,
                    "metrics": {},
                    "feature_importance": None
                }
        else:
            self.logger.info("Training model without hyperparameter optimization")
            fit_params = {}
            if getattr(self.config, 'early_stopping', False) and hasattr(model, 'early_stopping') and X_val_processed is not None:
                fit_params['eval_set'] = [(X_train_processed, y_train), (X_val_processed, y_val)]
                fit_params['early_stopping_rounds'] = getattr(self.config, 'early_stopping_rounds', 10)
                fit_params['verbose'] = bool(self.config.verbose)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_processed, y_train, **fit_params)
            best_model_info = {
                "model": model,
                "params": model.get_params(),
                "feature_names": selected_feature_names,
                "training_time": time.time() - start_time,
                "metrics": {},
                "feature_importance": None
            }
        del X_train_processed  # Free memory
        gc.collect()

        # Evaluate the best model on validation data if available
        if X_val_processed is not None and y_val is not None:
            self.logger.info("Evaluating model on validation data...")
            val_metrics = self._evaluate_model(best_model_info["model"], None, None, X_val_processed, y_val)
            best_model_info["metrics"] = val_metrics
            if self.tracker:
                self.tracker.log_metrics(val_metrics, step="validation")
            for metric, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"Validation {metric}: {value:.4f}")
                else:
                    self.logger.info(f"Validation {metric}: {value}")
        del X_val_processed, y_val  # Free memory
        gc.collect()

        # Extract feature importance if available
        feature_importance = self._get_feature_importance(best_model_info["model"])
        if feature_importance is not None:
            best_model_info["feature_importance"] = feature_importance
            feature_importance_dict = {selected_feature_names[i] if i < len(selected_feature_names) else f"feature_{i}": float(importance) for i, importance in enumerate(feature_importance)}
            if self.tracker:
                self.tracker.log_feature_importance(
                    feature_names=list(feature_importance_dict.keys()),
                    importance=np.array(list(feature_importance_dict.values()))
                )
            top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            self.logger.info("Top 10 features by importance:")
            for feature, importance in top_features:
                self.logger.info(f"  {feature}: {importance:.4f}")
        del feature_importance  # Free memory
        gc.collect()

        # Save the model to the models registry
        self.models[model_name] = best_model_info

        # Check if this is the best model so far
        best_metric = self._get_best_metric_value(best_model_info["metrics"]) if "metrics" in best_model_info and best_model_info["metrics"] else 0
        if self._compare_metrics(best_metric, self.best_score):
            self.best_model = best_model_info
            self.best_model_name = model_name
            self.best_score = best_metric
            self.logger.info(f"New best model: {model_name} with score {best_metric:.4f}")
            if getattr(self.config, 'auto_save', False):
                self.save_model(model_name)

        # Generate confusion matrix for classification tasks
        if self.config.task_type == TaskType.CLASSIFICATION and best_model_info["metrics"] and self.tracker:
            try:
                y_pred = best_model_info["model"].predict(best_model_info["metrics"].get('X_val', None))
                class_names = list(map(str, np.unique(y)))
                self.tracker.log_confusion_matrix(y_val, y_pred, class_names=class_names)
            except Exception as e:
                self.logger.warning(f"Failed to generate confusion matrix: {str(e)}")

        # End experiment tracking if enabled
        if self.tracker:
            self.tracker.end_experiment()
            if getattr(self.config, 'generate_model_summary', False):
                try:
                    report_path = self.tracker.generate_report()
                    self.logger.info(f"Model report generated: {report_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to generate model report: {str(e)}")

        self.training_complete = True
        total_time = time.time() - start_time
        self.logger.info(f"Model training completed in {total_time:.2f} seconds")
        gc.collect()
        return {
            "model_name": model_name,
            "model": best_model_info["model"],
            "params": best_model_info["params"],
            "metrics": best_model_info.get("metrics", {}),
            "feature_importance": best_model_info.get("feature_importance", None),
            "training_time": total_time
        }
    
    def save_model(self, model_name: str, model_path: Optional[str] = None) -> Union[str, bool]:
        """Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            model_path: Optional path to save the model. If None, uses default path.
            
        Returns:
            Path to saved model on success, False on failure
        """
        if model_name not in self.models:
            self.logger.error(f"Model '{model_name}' not found")
            return False
        
        model_info = self.models[model_name]
        model = model_info["model"]
        
        try:
            if model_path is None:
                model_path = os.path.join(self.config.model_path, f"{model_name}.pkl")
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            model_info["save_path"] = model_path
            self._safe_log("info", f"Model '{model_name}' saved to {model_path}")
            return model_path
            
        except Exception as e:
            self._safe_log("error", f"Failed to save model: {str(e)}")
            return False

    def load_model(self, model_path: str, model_name: Optional[str] = None) -> Tuple[bool, Optional[Any]]:
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model file
            model_name: Optional name for the loaded model
            
        Returns:
            Tuple of (success_status, loaded_model)
        """
        try:
            # Validate input path
            if not isinstance(model_path, str) or not model_path.strip():
                self.logger.error(f"Invalid model path provided: {model_path}")
                return False, None
            
            # Check if file exists
            if not os.path.exists(model_path):
                self.logger.error(f"Model file does not exist: {model_path}")
                return False, None
            
            # Load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Generate model name if not provided
            if model_name is None:
                model_name = f"loaded_model_{int(time.time())}"
            
            # Create model info
            model_info = {
                "model": model,
                "params": model.get_params() if hasattr(model, 'get_params') else {},
                "load_path": model_path,
                "load_timestamp": time.time(),
                "metrics": {},
                "feature_importance": None
            }
            
            # Store in registry
            self.models[model_name] = model_info
            
            # Update best model if this is the first model
            if self.best_model is None:
                self.best_model = model
                self.best_model_name = model_name
            
            self.logger.info(f"Model loaded from {model_path} as '{model_name}'")
            return True, model
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {str(e)}")
            return False, None

    def _init_optimization_components(self):
        """Initialize high-impact optimization components."""
        # Initialize all components to defaults first
        self.jit_compiler = get_global_jit_compiler()
        self.mixed_precision_manager = get_global_mixed_precision_manager()
        self.adaptive_optimizer = get_global_adaptive_optimizer()
        self.streaming_pipeline = get_global_streaming_pipeline()
        
        try:
            # Initialize JIT compiler for hot path compilation
            if hasattr(self.config, 'enable_jit_compilation') and self.config.enable_jit_compilation:
                self.jit_compiler.initialize()
            else:
                self.jit_compiler = None
            
            # Initialize mixed precision manager
            if hasattr(self.config, 'enable_mixed_precision') and self.config.enable_mixed_precision:
                self.mixed_precision_manager.initialize()
            else:
                self.mixed_precision_manager = None
            
            # Initialize adaptive hyperparameter optimizer
            if hasattr(self.config, 'enable_adaptive_hyperopt') and self.config.enable_adaptive_hyperopt:
                self.adaptive_optimizer.initialize()
            else:
                self.adaptive_optimizer = None
                
        except Exception as e:
            self.logger.error(f"Error initializing optimization components: {str(e)}")
            # Fallback to None values
            self.jit_compiler = None
            self.mixed_precision_manager = None
            self.adaptive_optimizer = None
    
    def _get_best_metric_value(self, metrics):
        """Extract the best metric value based on task type."""
        if not metrics:
            return 0
            
        # Determine which metric to use based on task and config
        if hasattr(self.config, 'model_selection_criteria'):
            criteria = self.config.model_selection_criteria
            
            if isinstance(criteria, ModelSelectionCriteria):
                key = criteria.value
            else:
                key = criteria
                
            # Check if the metric exists
            if key in metrics:
                return metrics[key]
                
        # Default metrics by task
        if self.config.task_type == TaskType.CLASSIFICATION:
            if "f1" in metrics:
                return metrics["f1"]
            elif "accuracy" in metrics:
                return metrics["accuracy"]
        elif self.config.task_type == TaskType.REGRESSION:
            # For regression, lower RMSE/MSE is better, but higher R2 is better
            if "r2" in metrics:
                return metrics["r2"]
            elif "rmse" in metrics:
                return -metrics["rmse"]  # Negative because lower is better
            elif "mse" in metrics:
                return -metrics["mse"]  # Negative because lower is better
                
        # Default to the first metric
        return next(iter(metrics.values()))
    
    def train_model(self, X, y, model_type: str = None, custom_model=None, 
                param_grid: Dict = None, model_name: str = None, X_val=None, y_val=None):
        """
        Optimized: Train a machine learning model with hyperparameter optimization, focusing on speed and memory usage.
        """
        start_time = time.time()
        gc.collect()  # Clean up memory before starting

        # Store current training data for adaptive optimization
        self._current_X = X
        self._current_y = y
        
        # Handle large datasets with streaming pipeline
        use_streaming = (hasattr(self.config, 'enable_streaming') and self.config.enable_streaming and 
                        isinstance(X, pd.DataFrame) and len(X) > getattr(self.config, 'streaming_threshold', 10000) and
                        hasattr(self, 'streaming_pipeline') and self.streaming_pipeline is not None)
        
        if use_streaming:
            self.logger.info(f"Using streaming pipeline for large dataset with {len(X)} samples")
            
            # Use streaming preprocessing if enabled
            def preprocess_chunk(chunk):
                if self.preprocessor:
                    return self.preprocessor.fit_transform(chunk)
                return chunk
            
            # Process data in streaming fashion
            try:
                with self.streaming_pipeline.monitoring_context():
                    processed_chunks = list(self.streaming_pipeline.process_dataframe_stream(
                        X, preprocess_chunk
                    ))
                    X_processed = pd.concat(processed_chunks, ignore_index=True)
            except Exception as e:
                self.logger.warning(f"Streaming processing failed: {str(e)}. Using regular processing.")
                X_processed = X
        else:
            X_processed = X
        
        # Apply mixed precision optimization for numerical data
        if (hasattr(self.config, 'enable_mixed_precision') and self.config.enable_mixed_precision and
            hasattr(self, 'mixed_precision_manager') and self.mixed_precision_manager is not None):
            if isinstance(X_processed, np.ndarray):
                try:
                    X_processed = self.mixed_precision_manager.optimize_numpy_precision(X_processed)
                    self.logger.info("Applied mixed precision optimization to training data")
                except Exception as e:
                    self.logger.warning(f"Mixed precision optimization failed: {str(e)}")

        # Extract feature names if available
        feature_names = self._extract_feature_names(X_processed)
        self._last_feature_names = feature_names

        # Determine model type based on task
        task_key = self.config.task_type.value

        # Validate inputs
        if custom_model is None and model_type is None:
            if task_key == "classification":
                model_type = "random_forest"
            elif task_key == "regression":
                model_type = "random_forest"
            else:
                raise ValueError(f"Please specify model_type or custom_model for task type: {task_key}")

        # Set model name
        if model_name is None:
            if custom_model is not None:
                model_name = f"{custom_model.__class__.__name__}_{int(time.time())}"
            else:
                model_name = f"{model_type}_{int(time.time())}"

        # Get model class
        if custom_model is not None:
            model = custom_model
            self.logger.info(f"Using custom model: {model.__class__.__name__}")
        else:
            if task_key not in self._model_registry:
                raise ValueError(f"No models registered for task type: {task_key}")
            if model_type not in self._model_registry[task_key]:
                raise ValueError(f"Model type '{model_type}' not found for {task_key}. " 
                            f"Available models: {', '.join(self._model_registry[task_key].keys())}")
            model_class = self._model_registry[task_key][model_type]
            # Set n_jobs=-1 if supported for parallelism
            default_params = {"random_state": self.config.random_state} if hasattr(model_class, "random_state") else {}
            if hasattr(model_class, "n_jobs"):
                default_params["n_jobs"] = -1
            model = model_class(**default_params)
            self.logger.info(f"Initialized {model_type} model for {task_key}")

        # Get default parameter grid if not provided
        if param_grid is None:
            param_grid = self._get_default_param_grid(model)

        # Start experiment tracking if enabled
        if self.tracker:
            model_info = {
                "model_type": model_type or model.__class__.__name__,
                "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "task_type": task_key
            }
            self.tracker.start_experiment(
                config=self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
                model_info=model_info
            )

        # Set up the feature selector if enabled
        if hasattr(self.config, 'feature_selection') and self.config.feature_selection:
            self.feature_selector = self._get_feature_selector(X, y)
            if self.feature_selector:
                self.logger.info(f"Feature selection enabled: {getattr(self.config, 'feature_selection_method', 'default')}")

        # Create training pipeline
        pipeline = self._create_pipeline(model)

        # Split data for validation if not provided
        if (X_val is None or y_val is None) and hasattr(self.config, 'test_size') and self.config.test_size > 0:
            split_kwargs = dict(test_size=self.config.test_size, random_state=self.config.random_state)
            if self.config.task_type == TaskType.CLASSIFICATION and getattr(self.config, 'stratify', False):
                split_kwargs['stratify'] = y
            X_train, X_val, y_train, y_val = train_test_split(X, y, **split_kwargs)
            self.logger.info(f"Data split: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
        else:
            X_train, y_train = X, y
            self.logger.info(f"Using provided validation data: {X_val.shape[0]} validation samples" if X_val is not None else "No validation data")

        # Convert data to numpy arrays (no copy if possible)
        def to_numpy_safe(arr):
            if hasattr(arr, 'to_numpy'):
                return arr.to_numpy(copy=False)
            elif hasattr(arr, 'values'):
                return arr.values
            return arr
        X_train = to_numpy_safe(X_train)
        y_train = to_numpy_safe(y_train)
        X_val = to_numpy_safe(X_val) if X_val is not None else None
        y_val = to_numpy_safe(y_val) if y_val is not None else None

        # Fit preprocessor if available
        if self.preprocessor:
            try:
                self.logger.info("Fitting preprocessor...")
                self.preprocessor.fit(X_train, y_train)
                X_train_processed = self.preprocessor.transform(X_train)
                X_val_processed = self.preprocessor.transform(X_val) if X_val is not None else None
                self.logger.info("Preprocessor fitted successfully")
            except Exception as e:
                self.logger.error(f"Error fitting preprocessor: {str(e)}")
                if getattr(self.config, 'debug_mode', False):
                    self.logger.error(traceback.format_exc())
                X_train_processed = X_train
                X_val_processed = X_val
        else:
            X_train_processed = X_train
            X_val_processed = X_val
        del X_train, X_val  # Free memory
        gc.collect()

        # Fit feature selector if available
        selected_feature_names = feature_names
        if self.feature_selector:
            try:
                self.logger.info("Fitting feature selector...")
                self.feature_selector.fit(X_train_processed, y_train)
                X_train_processed = self.feature_selector.transform(X_train_processed)
                if X_val_processed is not None:
                    X_val_processed = self.feature_selector.transform(X_val_processed)
                if hasattr(self.feature_selector, 'get_support'):
                    feature_indices = self.feature_selector.get_support(indices=True)
                    selected_feature_names = [feature_names[i] for i in feature_indices] if feature_names else None
                    self.logger.info(f"Selected {len(feature_indices)} features out of {len(feature_names)}")
                    if self.tracker and selected_feature_names:
                        self.tracker.log_metrics({
                            "selected_feature_count": len(feature_indices),
                            "total_feature_count": len(feature_names)
                        }, step="feature_selection")
            except Exception as e:
                self.logger.error(f"Error fitting feature selector: {str(e)}")
                if getattr(self.config, 'debug_mode', False):
                    self.logger.error(traceback.format_exc())
                selected_feature_names = feature_names
        # del y_train  # Free memory (moved to after training)
        gc.collect()

        # Configure hyperparameter optimization
        if param_grid and len(param_grid) > 0:
            self.logger.info(f"Starting hyperparameter optimization with {getattr(self.config, 'optimization_strategy', 'default strategy')}")
            try:
                optimizer = self._get_optimization_search(model, param_grid)
                if hasattr(optimizer, 'n_jobs'):
                    optimizer.n_jobs = -1
                if self.tracker:
                    self.tracker.log_metrics({
                        "param_combinations": getattr(optimizer, 'n_iter', 'grid'),
                        "cv_folds": self.config.cv_folds,
                        "scoring": str(getattr(optimizer, 'scoring', 'default'))
                    }, step="optimization_setup")
                fit_params = {}
                if getattr(self.config, 'early_stopping', False) and hasattr(model, 'early_stopping') and X_val_processed is not None:
                    fit_params['eval_set'] = [(X_train_processed, y_train), (X_val_processed, y_val)]
                    fit_params['early_stopping_rounds'] = getattr(self.config, 'early_stopping_rounds', 10)
                    fit_params['verbose'] = bool(self.config.verbose)
                optimization_start = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    optimizer.fit(X_train_processed, y_train, **fit_params)
                optimization_time = time.time() - optimization_start
                best_model = getattr(optimizer, 'best_estimator_', optimizer.estimator)
                best_params = getattr(optimizer, 'best_params_', optimizer.get_params())
                if hasattr(optimizer, 'cv_results_'):
                    cv_results = optimizer.cv_results_
                    mean_test_score = np.mean(cv_results['mean_test_score'])
                    std_test_score = np.mean(cv_results['std_test_score'])
                    if self.tracker:
                        self.tracker.log_metrics({
                            "mean_cv_score": mean_test_score,
                            "std_cv_score": std_test_score,
                            "optimization_time": optimization_time
                        }, step="optimization")
                    self.logger.info(f"Best CV score: {mean_test_score:.4f} ± {std_test_score:.4f}")
                    self.logger.info(f"Best parameters: {best_params}")
                best_model_info = {
                    "model": best_model,
                    "params": best_params,
                    "feature_names": selected_feature_names,
                    "training_time": time.time() - start_time,
                    "metrics": {},
                    "feature_importance": None
                }
            except Exception as e:
                self.logger.error(f"Error during hyperparameter optimization: {str(e)}")
                if getattr(self.config, 'debug_mode', False):
                    self.logger.error(traceback.format_exc())
                self.logger.info("Falling back to basic training without optimization")
                model.fit(X_train_processed, y_train)
                best_model_info = {
                    "model": model,
                    "params": model.get_params(),
                    "feature_names": selected_feature_names,
                    "training_time": time.time() - start_time,
                    "metrics": {},
                    "feature_importance": None
                }
        else:
            self.logger.info("Training model without hyperparameter optimization")
            fit_params = {}
            if getattr(self.config, 'early_stopping', False) and hasattr(model, 'early_stopping') and X_val_processed is not None:
                fit_params['eval_set'] = [(X_train_processed, y_train), (X_val_processed, y_val)]
                fit_params['early_stopping_rounds'] = getattr(self.config, 'early_stopping_rounds', 10)
                fit_params['verbose'] = bool(self.config.verbose)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_processed, y_train, **fit_params)
            best_model_info = {
                "model": model,
                "params": model.get_params(),
                "feature_names": selected_feature_names,
                "training_time": time.time() - start_time,
                "metrics": {},
                "feature_importance": None
            }
        del X_train_processed  # Free memory
        gc.collect()

        # Evaluate the best model on validation data if available
        if X_val_processed is not None and y_val is not None:
            self.logger.info("Evaluating model on validation data...")
            val_metrics = self._evaluate_model(best_model_info["model"], None, None, X_val_processed, y_val)
            best_model_info["metrics"] = val_metrics
            if self.tracker:
                self.tracker.log_metrics(val_metrics, step="validation")
            for metric, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"Validation {metric}: {value:.4f}")
                else:
                    self.logger.info(f"Validation {metric}: {value}")
        del X_val_processed, y_val  # Free memory
        gc.collect()

        # Extract feature importance if available
        feature_importance = self._get_feature_importance(best_model_info["model"])
        if feature_importance is not None:
            best_model_info["feature_importance"] = feature_importance
            feature_importance_dict = {selected_feature_names[i] if i < len(selected_feature_names) else f"feature_{i}": float(importance) for i, importance in enumerate(feature_importance)}
            if self.tracker:
                self.tracker.log_feature_importance(
                    feature_names=list(feature_importance_dict.keys()),
                    importance=np.array(list(feature_importance_dict.values()))
                )
            top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            self.logger.info("Top 10 features by importance:")
            for feature, importance in top_features:
                self.logger.info(f"  {feature}: {importance:.4f}")
        del feature_importance  # Free memory
        gc.collect()

        # Save the model to the models registry
        self.models[model_name] = best_model_info

        # Check if this is the best model so far
        best_metric = self._get_best_metric_value(best_model_info["metrics"]) if "metrics" in best_model_info and best_model_info["metrics"] else 0
        if self._compare_metrics(best_metric, self.best_score):
            self.best_model = best_model_info
            self.best_model_name = model_name
            self.best_score = best_metric
            self.logger.info(f"New best model: {model_name} with score {best_metric:.4f}")
            if getattr(self.config, 'auto_save', False):
                self.save_model(model_name)

        # Generate confusion matrix for classification tasks
        if self.config.task_type == TaskType.CLASSIFICATION and best_model_info["metrics"] and self.tracker:
            try:
                y_pred = best_model_info["model"].predict(best_model_info["metrics"].get('X_val', None))
                class_names = list(map(str, np.unique(y)))
                self.tracker.log_confusion_matrix(y_val, y_pred, class_names=class_names)
            except Exception as e:
                self.logger.warning(f"Failed to generate confusion matrix: {str(e)}")

        # End experiment tracking if enabled
        if self.tracker:
            self.tracker.end_experiment()
            if getattr(self.config, 'generate_model_summary', False):
                try:
                    report_path = self.tracker.generate_report()
                    self.logger.info(f"Model report generated: {report_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to generate model report: {str(e)}")



        self.training_complete = True
        total_time = time.time() - start_time
        self.logger.info(f"Model training completed in {total_time:.2f} seconds")
        gc.collect()
        return {
            "model_name": model_name,
            "model": best_model_info["model"],
            "params": best_model_info["params"],
            "metrics": best_model_info.get("metrics", {}),
            "feature_importance": best_model_info.get("feature_importance", None),
            "training_time": total_time
        }
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """
        Compare performance across all trained models.
        
        Returns:
            Dictionary with model comparisons and best model information
        """
        try:
            if not self.models:
                return {
                    "models": [],
                    "best_model": None,
                    "primary_metric": None,
                    "total_models": 0,
                    "error": "No models trained yet"
                }
            
            models_list = []
            best_model_name = self.best_model_name
            primary_metric = None
            
            for model_name, model_info in self.models.items():
                try:
                    # Extract model type safely
                    model_type = "Unknown"
                    model_obj = model_info.get("model", None)
                    if model_obj and hasattr(model_obj, '__class__'):
                        model_type = model_obj.__class__.__name__
                    
                    # Get metrics safely
                    metrics = model_info.get("metrics", {})
                    if not isinstance(metrics, dict):
                        metrics = {}
                    
                    # Get training time safely
                    training_time = model_info.get("training_time", 0)
                    if not isinstance(training_time, (int, float)):
                        training_time = 0
                    
                    # Determine if this is the best model
                    is_best = (model_name == self.best_model_name)
                    
                    # Determine primary metric from the best model
                    if is_best and not primary_metric:
                        if hasattr(self.config, 'model_selection_criteria'):
                            criteria = self.config.model_selection_criteria
                            if hasattr(criteria, 'value'):
                                primary_metric = criteria.value
                            else:
                                primary_metric = str(criteria)
                        else:
                            # Default primary metrics by task
                            task_type = getattr(self.config.task_type, 'value', 'unknown')
                            if task_type == "classification":
                                primary_metric = "f1"
                            elif task_type == "regression":
                                primary_metric = "r2"
                            else:
                                primary_metric = list(metrics.keys())[0] if metrics else "score"
                    
                    models_list.append({
                        "name": model_name,
                        "type": model_type,
                        "training_time": training_time,
                        "is_best": is_best,
                        "metrics": metrics
                    })
                    
                except Exception as model_error:
                    self.logger.warning(f"Error processing model {model_name}: {str(model_error)}")
                    # Add a minimal entry for the problematic model
                    models_list.append({
                        "name": model_name,
                        "type": "Error",
                        "training_time": 0,
                        "is_best": False,
                        "metrics": {}
                    })
            
            return {
                "models": models_list,
                "best_model": best_model_name,
                "primary_metric": primary_metric,
                "total_models": len(models_list)
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_performance_comparison: {str(e)}")
            # Always return a valid structure even on error
            return {
                "models": [],
                "best_model": None,
                "primary_metric": None,
                "total_models": 0,
                "error": str(e)
            }
    
    def evaluate_model(self, model_name: str = None, X_test=None, y_test=None, detailed: bool = False) -> Dict[str, Any]:
        """
        Evaluate a model with comprehensive metrics.
        
        Args:
            model_name: Name of the model to evaluate (uses best model if None)
            X_test: Test features
            y_test: Test targets  
            detailed: Whether to compute additional detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Determine which model to use
            if model_name is None or model_name == "best":
                if self.best_model is None:
                    return {"error": "No best model available"}
                model = self.best_model.get("model") if isinstance(self.best_model, dict) else self.best_model
                actual_model_name = self.best_model_name
            else:
                if model_name not in self.models:
                    return {"error": f"Model '{model_name}' not found"}
                model = self.models[model_name]["model"]
                actual_model_name = model_name
            
            if X_test is None or y_test is None:
                return {"error": "Test data (X_test, y_test) must be provided"}
            
            # Perform evaluation
            start_time = time.time()
            metrics = self._evaluate_model(model, None, None, X_test, y_test)
            prediction_time = time.time() - start_time
            metrics["prediction_time"] = prediction_time
            
            # Add detailed metrics if requested
            if detailed:
                try:
                    y_pred = model.predict(X_test)
                    
                    if self.config.task_type.value == "classification":
                        # Confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        metrics["confusion_matrix"] = cm.tolist()
                        
                        # Classification report
                        report = classification_report(y_test, y_pred, output_dict=True)
                        metrics["detailed_report"] = report
                        
                    elif self.config.task_type.value == "regression":
                        # Additional regression metrics
                        explained_var = explained_variance_score(y_test, y_pred)
                        metrics["explained_variance"] = explained_var
                        
                        # Add residual statistics
                        residuals = y_test - y_pred
                        metrics["residual_stats"] = {
                            "mean": float(np.mean(residuals)),
                            "std": float(np.std(residuals)),
                            "min": float(np.min(residuals)),
                            "max": float(np.max(residuals))
                        }
                        
                except Exception as detail_error:
                    self.logger.warning(f"Error computing detailed metrics: {str(detail_error)}")
                    metrics["detailed_error"] = str(detail_error)
            
            return {
                "model_name": actual_model_name,
                "metrics": metrics,
                "detailed": detailed
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {"error": str(e)}
    
    def get_best_model(self) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Get the current best model and its information.
        
        Returns:
            Tuple of (model_name, model_info)
        """
        if self.best_model_name and self.best_model_name in self.models:
            return self.best_model_name, self.models[self.best_model_name]
        return None, None
    
    def generate_report(self, output_file: Optional[str] = None) -> Optional[str]:
        """
        Generate a comprehensive report of all models.
        
        Args:
            output_file: Path to save the report
            
        Returns:
            Path to the generated report or None if failed
        """
        try:
            if not self.models:
                self.logger.warning("No models to report on")
                return None
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Start building the report
            report = f"# ML Training Engine Report\n\n"
            report += f"**Generated**: {timestamp}\n\n"
            report += f"**Total Models Trained**: {len(self.models)}\n"
            report += f"**Best Model**: {self.best_model_name or 'None'}\n\n"
            
            # Add configuration summary
            report += "## Configuration\n\n"
            report += f"- **Task Type**: {self.config.task_type.value}\n"
            report += f"- **Random State**: {getattr(self.config, 'random_state', 'Not set')}\n"
            report += f"- **CV Folds**: {getattr(self.config, 'cv_folds', 'Not set')}\n"
            report += f"- **Optimization Strategy**: {getattr(self.config, 'optimization_strategy', 'Not set')}\n\n"
            
            # Add model comparison
            comparison = self.get_performance_comparison()
            if "error" not in comparison:
                report += "## Model Performance Comparison\n\n"
                report += "| Model Name | Type | Training Time (s) | Best | Metrics |\n"
                report += "|------------|------|------------------|------|----------|\n"
                
                for model in comparison["models"]:
                    metrics_str = ""
                    if model["metrics"]:
                        # Show top 3 metrics
                        metric_items = list(model["metrics"].items())[:3]
                        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                               for k, v in metric_items])
                    
                    best_indicator = "✅" if model["is_best"] else ""
                    report += f"| {model['name']} | {model['type']} | {model['training_time']:.2f} | {best_indicator} | {metrics_str} |\n"
                
                report += "\n"
            
            # Add detailed model information
            report += "## Detailed Model Information\n\n"
            for model_name, model_info in self.models.items():
                report += f"### {model_name}\n\n"
                
                # Model details
                model = model_info.get("model")
                if model:
                    report += f"**Model Type**: {model.__class__.__name__}\n\n"
                
                # Parameters
                params = model_info.get("params", {})
                if params:
                    report += "**Parameters**:\n"
                    for param, value in params.items():
                        report += f"- {param}: {value}\n"
                    report += "\n"
                
                # Metrics
                metrics = model_info.get("metrics", {})
                if metrics:
                    report += "**Performance Metrics**:\n"
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            report += f"- {metric}: {value:.6f}\n"
                        else:
                            report += f"- {metric}: {value}\n"
                    report += "\n"
                
                # Feature importance
                feature_importance = model_info.get("feature_importance")
                feature_names = model_info.get("feature_names")
                if feature_importance is not None and feature_names:
                    report += "**Top 10 Feature Importance**:\n"
                    # Create feature importance pairs and sort
                    importance_pairs = list(zip(feature_names, feature_importance))
                    importance_pairs.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (feature, importance) in enumerate(importance_pairs[:10]):
                        report += f"{i+1}. {feature}: {importance:.4f}\n"
                    report += "\n"
                
                report += "---\n\n"
            
            # Save to file if specified
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"Report saved to {output_file}")
                return str(output_file)
            else:
                # Return the report content
                return report
                
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None
    
    def shutdown(self):
        """Explicitly shut down the engine and release resources."""
        try:
            self.logger.info("Shutting down ML Training Engine...")
            
            # Clear models to free memory
            self.models.clear()
            self.best_model = None
            self.best_model_name = None
            
            # Clean up optimization components
            if hasattr(self, 'jit_compiler') and self.jit_compiler:
                try:
                    self.jit_compiler.clear_cache()
                except:
                    pass
                    
            if hasattr(self, 'streaming_pipeline') and self.streaming_pipeline:
                try:
                    self.streaming_pipeline.shutdown()
                except:
                    pass
            
            # Clear any cached data
            if hasattr(self, '_current_X'):
                delattr(self, '_current_X')
            if hasattr(self, '_current_y'):
                delattr(self, '_current_y')
            if hasattr(self, '_last_feature_names'):
                delattr(self, '_last_feature_names')
                
            # Force garbage collection
            gc.collect()
            
            self.logger.info("ML Training Engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    def _evaluate_model(self, model, X_train=None, y_train=None, X_test=None, y_test=None):
        """Evaluate model performance on test data."""
        if X_test is None or y_test is None:
            return {}
        
        metrics = {}
        try:
            y_pred = model.predict(X_test)
            
            if self.config.task_type == TaskType.CLASSIFICATION:
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                
                if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                    try:
                        y_proba = model.predict_proba(X_test)
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                    except Exception:
                        pass
                        
            elif self.config.task_type == TaskType.REGRESSION:
                metrics['mse'] = mean_squared_error(y_test, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                metrics['r2'] = r2_score(y_test, y_pred)
                
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
                
        return metrics