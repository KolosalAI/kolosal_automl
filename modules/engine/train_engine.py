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
    
    VERSION = "1.0.0"
    
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
        
        # Set up logging
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
        self.logger.info(f"Received signal {signum}, cleaning up...")
        self._cleanup_on_shutdown()
        
    def _cleanup_on_shutdown(self):
        """Perform cleanup operations before shutdown."""
        if hasattr(self, 'config') and hasattr(self.config, 'auto_save_on_shutdown') and self.config.auto_save_on_shutdown and hasattr(self, 'best_model') and self.best_model is not None:
            self.logger.info("Auto-saving best model before shutdown")
            try:
                self.save_model(self.best_model_name)
            except Exception as e:
                self.logger.error(f"Error saving model during shutdown: {str(e)}")
                
        # Save experiment state if requested
        if hasattr(self, 'config') and hasattr(self.config, 'save_state_on_shutdown') and self.config.save_state_on_shutdown and hasattr(self, 'tracker') and self.tracker:
            try:
                self.logger.info("Saving experiment state before shutdown")
                if hasattr(self.tracker, 'end_experiment'):
                    self.tracker.end_experiment()
            except Exception as e:
                self.logger.error(f"Error saving experiment state during shutdown: {str(e)}")
                
        # Signal components to clean up
        try:
            if hasattr(self, 'inference_engine'):
                self.inference_engine.shutdown()
                
            if hasattr(self, 'batch_processor'):
                self.batch_processor.stop()
        except Exception as e:
            self.logger.error(f"Error shutting down components: {str(e)}")
            
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

        # Extract feature names if available
        feature_names = self._extract_feature_names(X)
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
    def get_performance_comparison(self):
        """
        Compare performance across all trained models.
        
        Returns:
            Dictionary with model comparisons
        """
        if not self.models:
            return {"error": "No models trained yet"}
            
        comparison = {
            "models": [],
            "best_model": self.best_model_name
        }
        
        # Find common metrics across all models
        common_metrics = set()
        for model_name, model_info in self.models.items():
            if "metrics" in model_info:
                if not common_metrics:
                    common_metrics = set(model_info["metrics"].keys())
                else:
                    common_metrics = common_metrics.intersection(set(model_info["metrics"].keys()))
        
        # Create comparison data
        for model_name, model_info in self.models.items():
            model_data = {
                "name": model_name,
                "type": type(model_info["model"]).__name__,
                "training_time": model_info.get("training_time", 0),
                "is_best": (model_name == self.best_model_name),
                "metrics": {}
            }
            
            # Add metrics
            if "metrics" in model_info:
                for metric in common_metrics:
                    if metric in model_info["metrics"]:
                        model_data["metrics"][metric] = model_info["metrics"][metric]
                        
            comparison["models"].append(model_data)
            
        # Sort models by performance on the primary metric
        if comparison["models"] and common_metrics:
            primary_metric = self._determine_primary_metric(common_metrics)
            
            # For metrics where lower is better, sort in reverse
            reverse = True  # Default - higher is better
            if primary_metric in ["mse", "rmse", "mae", "mape", "median_absolute_error"]:
                reverse = False  # Lower is better
                
            comparison["models"].sort(
                key=lambda x: x["metrics"].get(primary_metric, float('-inf') if reverse else float('inf')),
                reverse=reverse
            )
            
            comparison["primary_metric"] = primary_metric
            
        return comparison
        
    def _determine_primary_metric(self, available_metrics):
        """Determine the primary metric to use for model comparison."""
        # Task-specific default metrics
        if self.config.task_type == TaskType.CLASSIFICATION:
            candidates = ["f1", "accuracy", "precision", "recall", "roc_auc"]
        elif self.config.task_type == TaskType.REGRESSION:
            candidates = ["r2", "rmse", "mse", "mae"]
        else:
            candidates = ["score"]
            
        # Check if configured metric is available
        if hasattr(self.config, 'model_selection_criteria'):
            metric = self.config.model_selection_criteria
            if isinstance(metric, ModelSelectionCriteria):
                metric = metric.value
                
            if metric in available_metrics:
                return metric
                
        # Find the first available metric from candidates
        for metric in candidates:
            if metric in available_metrics:
                return metric
                
        # If no preferred metric is available, use the first available
        return next(iter(available_metrics))
        
    def generate_report(self, output_file=None):
        """
        Generate a comprehensive report of all models.
        
        Args:
            output_file: Path to save the report
            
        Returns:
            Path to the generated report
        """
        if not self.models:
            self.logger.warning("No models to generate report")
            return None
                
        if output_file is None:
            output_file = os.path.join(self.config.model_path, "model_report.md")
                
        # Create basic report
        report = f"# ML Training Engine Report\n\n"
        report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add configuration section
        report += "## Configuration\n\n"
        report += "| Parameter | Value |\n"
        report += "| --- | --- |\n"
        
        # Add configuration
        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        for key, value in config_dict.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                report += f"| {key} | {value} |\n"
        
        report += "\n## Model Performance Summary\n\n"
        
        # Collect all metrics across models
        all_metrics = set()
        for model_data in self.models.values():
            all_metrics.update(model_data.get("metrics", {}).keys())
        
        # Create metrics table header
        report += "| Model | " + " | ".join(sorted(all_metrics)) + " |\n"
        report += "| --- | " + " | ".join(["---" for _ in all_metrics]) + " |\n"
        
        # Add model rows
        for model_name, model_data in self.models.items():
            is_best = self.best_model_name and self.best_model_name == model_name
            model_label = f"{model_name} **[BEST]**" if is_best else model_name
            
            row = f"| {model_label} |"
            for metric in sorted(all_metrics):
                value = model_data.get("metrics", {}).get(metric, "N/A")
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
                row += f" {value} |"
            
            report += row + "\n"
        
        # Add model details section
        report += "\n## Model Details\n\n"
        
        for model_name, model_data in self.models.items():
            report += f"### {model_name}\n\n"
            
            # Add model type
            report += f"- **Type**: {type(model_data.get('model', '')).__name__}\n"
            
            # Add if it's the best model
            is_best = self.best_model_name and self.best_model_name == model_name
            report += f"- **Best Model**: {'Yes' if is_best else 'No'}\n"
            
            # Add training time if available
            if "training_time" in model_data:
                report += f"- **Training Time**: {model_data['training_time']:.2f}s\n"
            
            # Add hyperparameters if available
            if "params" in model_data and model_data["params"]:
                report += "\n#### Hyperparameters\n\n"
                report += "| Parameter | Value |\n"
                report += "| --- | --- |\n"
                
                for param, value in model_data["params"].items():
                    report += f"| {param} | {value} |\n"
                
                report += "\n"
            
            # Add feature importance if available
            if "feature_importance" in model_data and model_data["feature_importance"] is not None:
                feature_importance = model_data["feature_importance"]
                if isinstance(feature_importance, dict):
                    # Sort by importance and get top 10
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    report += "\n#### Top 10 Features by Importance\n\n"
                    report += "| Feature | Importance |\n"
                    report += "| --- | --- |\n"
                    
                    for feature, importance in sorted_features:
                        report += f"| {feature} | {importance:.4f} |\n"
                    
                    report += "\n"
                elif isinstance(feature_importance, np.ndarray):
                    # For numpy array, we need feature names
                    feature_names = model_data.get("feature_names", [f"feature_{i}" for i in range(len(feature_importance))])
                    
                    # Sort by importance and get top 10
                    indices = np.argsort(feature_importance)[::-1][:10]
                    
                    report += "\n#### Top 10 Features by Importance\n\n"
                    report += "| Feature | Importance |\n"
                    report += "| --- | --- |\n"
                    
                    for idx in indices:
                        feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                        report += f"| {feature_name} | {feature_importance[idx]:.4f} |\n"
                    
                    report += "\n"
        
        # Add conclusion
        report += "\n## Conclusion\n\n"
        
        if self.best_model_name:
            best_model_name = self.best_model_name
            best_metrics = self.models[best_model_name].get("metrics", {})
            
            report += f"The best performing model is **{best_model_name}** with metrics:\n\n"
            
            # List key metrics
            key_metrics = []
            if self.config.task_type == TaskType.CLASSIFICATION:
                key_metrics = ["accuracy", "f1", "precision", "recall"]
            elif self.config.task_type == TaskType.REGRESSION:
                key_metrics = ["rmse", "mse", "r2", "mae"]
                
            for metric in key_metrics:
                if metric in best_metrics:
                    report += f"- **{metric}**: {best_metrics[metric]:.4f}\n"
        else:
            report += "No models have been trained or evaluated yet."
        
        # Write report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"Report generated: {output_file}")
        return output_file
        
    def __del__(self):
        """Clean up resources when object is deleted."""
        self._cleanup_on_shutdown()
        
    def shutdown(self):
        """Explicitly shut down the engine and release resources."""
        self._cleanup_on_shutdown()
        
        # Set state to indicate shutdown
        self.logger.info("MLTrainingEngine shut down")
        
    def get_best_model(self):
        """
        Get the current best model and its metrics.
        
        Returns:
            Tuple containing (model_name, model_info) if a best model exists, 
            otherwise (None, None)
        """
        if self.best_model is None:
            return None, None
            
        return self.best_model_name, self.best_model
    
    def evaluate_model(self, model_name=None, X_test=None, y_test=None, detailed=False):
        """
        Evaluate a model with comprehensive metrics and analysis.
        
        Args:
            model_name: Name of the model to evaluate (uses best model if None)
            X_test: Test features (uses cached test data if None)
            y_test: Test targets (uses cached test data if None)
            detailed: Whether to perform detailed evaluation with additional metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Determine which model to use
        if model_name is None and self.best_model is not None:
            model = self.best_model["model"]
            model_name = self.best_model_name
        elif model_name in self.models:
            model = self.models[model_name]["model"]
        else:
            self.logger.error(f"Model {model_name} not found")
            return {"error": f"Model {model_name} not found"}
            
        # Use provided test data or fall back to cached data
        if X_test is None or y_test is None:
            if hasattr(self, '_last_X_test') and hasattr(self, '_last_y_test'):
                X_test = self._last_X_test
                y_test = self._last_y_test
                self.logger.info("Using cached test data for evaluation")
            else:
                self.logger.error("No test data provided and no cached test data available")
                return {"error": "No test data available"}
                
        self.logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Apply preprocessing if needed
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                try:
                    X_test = self.preprocessor.transform(X_test)
                except Exception as e:
                    self.logger.error(f"Preprocessing failed during evaluation: {str(e)}")
                    return {"error": f"Preprocessing error: {str(e)}"}
            
            # Apply feature selection if needed
            if self.feature_selector and hasattr(self.feature_selector, 'transform'):
                try:
                    X_test = self.feature_selector.transform(X_test)
                except Exception as e:
                    self.logger.error(f"Feature selection failed during evaluation: {str(e)}")
                    return {"error": f"Feature selection error: {str(e)}"}
            
            # Time the prediction
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Basic metrics based on task type
            metrics = {"prediction_time": pred_time}
            
            if self.config.task_type == TaskType.CLASSIFICATION:
                # Classification metrics
                metrics.update({
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted"),
                    "recall": recall_score(y_test, y_pred, average="weighted"),
                    "f1": f1_score(y_test, y_pred, average="weighted")
                })
                
                # Add ROC AUC if binary classification
                if len(np.unique(y_test)) == 2:
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_prob = model.predict_proba(X_test)[:, 1]
                            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
                    except (AttributeError, IndexError) as e:
                        self.logger.warning(f"Could not calculate ROC AUC: {str(e)}")
                
                # Add detailed classification metrics if requested
                if detailed:
                    try:
                        report = classification_report(y_test, y_pred, output_dict=True)
                        metrics["detailed_report"] = report
                        
                        cm = confusion_matrix(y_test, y_pred)
                        metrics["confusion_matrix"] = cm.tolist()
                        
                        # Calculate per-class metrics
                        class_metrics = {}
                        classes = np.unique(y_test)
                        for cls in classes:
                            class_metrics[str(cls)] = {
                                "precision": precision_score(y_test, y_pred, labels=[cls], average=None)[0],
                                "recall": recall_score(y_test, y_pred, labels=[cls], average=None)[0],
                                "f1": f1_score(y_test, y_pred, labels=[cls], average=None)[0],
                                "support": np.sum(y_test == cls)
                            }
                        metrics["per_class"] = class_metrics
                    except Exception as e:
                        self.logger.warning(f"Could not calculate detailed metrics: {str(e)}")
                        
            elif self.config.task_type == TaskType.REGRESSION:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                metrics.update({
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred)
                })
                
                # Add detailed regression metrics if requested
                if detailed:
                    try:
                        # Calculate median absolute error
                        from sklearn.metrics import median_absolute_error, explained_variance_score
                        metrics["median_absolute_error"] = median_absolute_error(y_test, y_pred)
                        metrics["explained_variance"] = explained_variance_score(y_test, y_pred)
                        
                        # Calculate absolute percentage error
                        if not np.any(y_test == 0):  # Avoid division by zero
                            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                            metrics["mape"] = mape
                        
                        # Calculate residuals for analysis
                        residuals = y_test - y_pred
                        metrics["residuals_mean"] = float(np.mean(residuals))
                        metrics["residuals_std"] = float(np.std(residuals))
                    except Exception as e:
                        self.logger.warning(f"Could not calculate detailed metrics: {str(e)}")
            
            # Update model metrics
            if model_name in self.models:
                self.models[model_name]["metrics"] = metrics
                self.models[model_name]["last_evaluated"] = time.time()
                
                # Check if this affects best model ranking
                best_metric = self._get_best_metric_value(metrics)
                if self._compare_metrics(best_metric, self.best_score):
                    self.best_model = self.models[model_name]
                    self.best_model_name = model_name
                    self.best_score = best_metric
                    self.logger.info(f"Updated best model to {model_name} with score {best_metric:.4f}")
                
            # Log evaluation results
            self.logger.info(f"Evaluation results for {model_name}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {metric}: {value:.4f}")
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def save_model(self, model_name: str, path: str = None, include_preprocessor: bool = True):
        """
        Persist a model (optionally with preprocessor / selector) to disk.
        Now bullet‑proof against threading.Lock / RLock objects.
        """
        _patch_pickle_for_locks()          # ensure lock reducer is active

        if model_name not in self.models:
            self.logger.error("Model %s not found", model_name)
            return None

        info = self.models[model_name]
        path = path or os.path.join(self.config.model_path, f"{model_name}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        bundle = {"model": info["model"]}
        if include_preprocessor:
            bundle.update(
                preprocessor=self.preprocessor,
                feature_selector=self.feature_selector,
                feature_names=info.get("feature_names", []),
                metrics=info.get("metrics", {}),
                params=info.get("params", {}),
                task_type=self.config.task_type,
                timestamp=datetime.now().isoformat(),
                engine_version=self.VERSION,
            )

        try:                                    # 1‑st choice: joblib (fast for arrays)
            joblib.dump(bundle, path, compress=("gzip", 3))
        except Exception as e:
            self.logger.warning("joblib dump failed (%s). Falling back to pickle.", e)
            with open(path, "wb") as fh:       # 2‑nd choice: std‑lib pickle
                pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info("Model %s saved to %s", model_name, path)
        return path
        
    def load_model(self, path: str, model_name: str = None):
        """
        Load a saved model from disk.
        
        Args:
            path: Path to the saved model
            model_name: Name to give the loaded model (defaults to filename)
            
        Returns:
            Tuple of (success flag, model object or error message)
        """
        if not os.path.exists(path):
            self.logger.error(f"Model file not found: {path}")
            return False, f"Model file not found: {path}"
            
        # Determine model name if not provided
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(path))[0]
            
        try:
            # Try to load with joblib first
            if hasattr(joblib, 'load'):
                try:
                    loaded = joblib.load(path)
                except:
                    # Fall back to pickle
                    with open(path, 'rb') as f:
                        loaded = pickle.load(f)
            else:
                # Use pickle directly
                with open(path, 'rb') as f:
                    loaded = pickle.load(f)
                    
            # Check if it's a bundle or direct model
            if isinstance(loaded, dict) and "model" in loaded:
                # Extract from bundle
                model = loaded["model"]
                
                # Update preprocessor if included
                if "preprocessor" in loaded and loaded["preprocessor"] is not None:
                    self.preprocessor = loaded["preprocessor"]
                    self.logger.info("Updated preprocessor from loaded model bundle")
                    
                # Update feature selector if included
                if "feature_selector" in loaded and loaded["feature_selector"] is not None:
                    self.feature_selector = loaded["feature_selector"]
                    self.logger.info("Updated feature selector from loaded model bundle")
                    
                # Extract other metadata
                feature_names = loaded.get("feature_names", [])
                metrics = loaded.get("metrics", {})
                params = loaded.get("params", {})
                
                # Store in models registry
                self.models[model_name] = {
                    "model": model,
                    "params": params,
                    "feature_names": feature_names,
                    "metrics": metrics,
                    "loaded_from": path,
                    "loaded_at": datetime.now().isoformat()
                }
                
                # Calculate feature importance if available
                feature_importance = self._get_feature_importance(model)
                if feature_importance is not None:
                    self.models[model_name]["feature_importance"] = feature_importance
                
                self.logger.info(f"Loaded model bundle from {path} as {model_name}")
                
                # Check if this should be the best model
                if not self.best_model or (metrics and self._compare_metrics(
                    self._get_best_metric_value(metrics), self.best_score)):
                    self.best_model = self.models[model_name]
                    self.best_model_name = model_name
                    self.best_score = self._get_best_metric_value(metrics) if metrics else 0
                    self.logger.info(f"Set {model_name} as the best model")
                
                return True, model
                
            else:
                # Direct model object
                model = loaded
                
                # Store in models registry with minimal info
                self.models[model_name] = {
                    "model": model,
                    "params": model.get_params() if hasattr(model, 'get_params') else {},
                    "feature_names": [],
                    "metrics": {},
                    "loaded_from": path,
                    "loaded_at": datetime.now().isoformat()
                }
                
                # Calculate feature importance if available
                feature_importance = self._get_feature_importance(model)
                if feature_importance is not None:
                    self.models[model_name]["feature_importance"] = feature_importance
                
                self.logger.info(f"Loaded model from {path} as {model_name}")
                
                # If no best model set, use this one
                if not self.best_model:
                    self.best_model = self.models[model_name]
                    self.best_model_name = model_name
                    self.best_score = 0
                    self.logger.info(f"Set {model_name} as the best model (no metrics available)")
                
                return True, model
                
        except Exception as e:
            self.logger.error(f"Failed to load model from {path}: {str(e)}")
            if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False, str(e)
    
    def predict(self, X, model_name=None, return_proba=False):
        """
        Make predictions using a trained model.
        
        Args:
            X: Features to predict
            model_name: Name of the model to use (uses best model if None)
            return_proba: Whether to return probabilities for classification
            
        Returns:
            Tuple of (success flag, predictions or error message)
        """
        # Determine which model to use
        if model_name is None:
            if self.best_model is None:
                self.logger.error("No best model available for prediction")
                return False, "No model available"
            model = self.best_model["model"]
            model_name = self.best_model_name
        elif model_name in self.models:
            model = self.models[model_name]["model"]
        else:
            self.logger.error(f"Model {model_name} not found")
            return False, f"Model {model_name} not found"
        
        try:
            # Apply preprocessing if needed
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                try:
                    X = self.preprocessor.transform(X)
                except Exception as e:
                    self.logger.error(f"Preprocessing failed during prediction: {str(e)}")
                    return False, f"Preprocessing error: {str(e)}"
            
            # Apply feature selection if needed
            if self.feature_selector and hasattr(self.feature_selector, 'transform'):
                try:
                    X = self.feature_selector.transform(X)
                except Exception as e:
                    self.logger.error(f"Feature selection failed during prediction: {str(e)}")
                    return False, f"Feature selection error: {str(e)}"
            
            # Make predictions
            start_time = time.time()
            
            if return_proba and self.config.task_type == TaskType.CLASSIFICATION and hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)
            else:
                predictions = model.predict(X)
                
            prediction_time = time.time() - start_time
            
            self.logger.info(f"Made predictions with model {model_name} in {prediction_time:.4f}s")
            
            # Return predictions
            return True, predictions
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False, f"Prediction error: {str(e)}"
            
    def generate_explainability(
        self,
        model_name=None,
        X=None,
        y=None,
        method="drop_column",
        max_samples=500,
        n_repeats=3,
        random_state=None,
        create_plots=True,
        timeout=60
    ):
        import time
        import numpy as np
        import os
        import traceback
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.metrics import get_scorer
        from sklearn.inspection import permutation_importance

        start_time = time.time()

        def check_timeout():
            if timeout > 0 and time.time() - start_time > timeout:
                raise TimeoutError(f"Explanation generation exceeded timeout of {timeout} seconds")

        def sample_data(X, y):
            if X.shape[0] > max_samples:
                indices = np.random.RandomState(random_state).choice(X.shape[0], max_samples, replace=False)
                return X[indices], y[indices] if y is not None else None
            return X, y

        def safe_get_feature_names(preprocessor):
            try:
                return list(preprocessor.get_feature_names_out())
            except Exception:
                return None

        def drop_column_importance(model, X, y, feature_names, scorer):
            from sklearn.base import clone

            baseline_score = scorer(model, X, y)
            importance_dict = {}

            X_df = pd.DataFrame(X, columns=feature_names)

            for i, feature in enumerate(feature_names):
                if i % 5 == 0:
                    check_timeout()

                X_dropped = X_df.drop(columns=[feature])
                try:
                    model_clone = clone(model)
                    model_clone.fit(X_dropped, y)
                    score_dropped = scorer(model_clone, X_dropped, y)
                    importance = baseline_score - score_dropped
                    importance_dict[feature] = float(importance)
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate feature {feature}: {str(e)}")
                    importance_dict[feature] = 0.0

            return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))


        def permutation_feature_importance(model, X, y, feature_names):
            check_timeout()
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats,
                random_state=random_state, n_jobs=-1
            )
            check_timeout()
            importance_dict = {
                name: float(perm_importance.importances_mean[i])
                for i, name in enumerate(feature_names)
            }
            return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

        def plot_importance(importance_dict, title, xlabel, plot_name):
            if not create_plots or not hasattr(self.config, 'model_path'):
                return None
            plot_dir = os.path.join(self.config.model_path, "explanations")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, plot_name)
            try:
                plt.figure(figsize=(12, 8))
                features = list(importance_dict.keys())[:15]
                importances = [importance_dict[f] for f in features]
                plt.barh(range(len(features)), importances, align='center')
                plt.yticks(range(len(features)), features)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                return plot_path
            except Exception as e:
                self.logger.warning(f"Failed to create plot: {str(e)}")
                return None

        if random_state is None:
            random_state = getattr(getattr(self, 'config', None), 'random_state', 42)

        # Determine model
        if model_name is None and hasattr(self, 'best_model'):
            model = self.best_model["model"]
            model_name = self.best_model_name
            feature_names = self.best_model.get("feature_names", None)
        elif model_name and hasattr(self, 'models') and model_name in self.models:
            model = self.models[model_name]["model"]
            feature_names = self.models[model_name].get("feature_names", None)
        else:
            return {"error": f"Model {model_name} not found", "method": method}

        if X is None:
            X = getattr(self, '_last_X_train', None)
        if y is None and method in ["drop_column", "permutation"]:
            y = getattr(self, '_last_y_train', None)

        if X is None or (y is None and method in ["drop_column", "permutation"]):
            return {"error": "No data available for explanations", "method": method}

        X, y = sample_data(X, y)

        if hasattr(self, 'preprocessor') and self.preprocessor and hasattr(self.preprocessor, 'transform'):
            try:
                X = self.preprocessor.transform(X)
                feature_names = safe_get_feature_names(self.preprocessor) or feature_names
            except Exception as e:
                return {"error": f"Preprocessing error: {str(e)}", "method": method}

        if hasattr(self, 'feature_selector') and self.feature_selector and hasattr(self.feature_selector, 'transform'):
            try:
                if hasattr(self.feature_selector, 'get_support'):
                    selected_indices = self.feature_selector.get_support(indices=True)
                    if feature_names:
                        feature_names = [feature_names[i] for i in selected_indices]
                X = self.feature_selector.transform(X)
            except Exception as e:
                return {"error": f"Feature selection error: {str(e)}", "method": method}

        if feature_names is None or len(feature_names) != X.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        try:
            if method == "drop_column":
                scorer_name = "accuracy" if getattr(self, 'task_type', 'regression') != 'regression' else "r2"
                scorer = get_scorer(scorer_name)
                importance = drop_column_importance(model, X, y, feature_names, scorer)
                plot_path = plot_importance(importance, "Drop Column Importance", f"Decrease in {scorer_name}", f"drop_column_importance_{model_name}.png")
                return {
                    "method": method,
                    "importance": importance,
                    "plot_path": plot_path,
                    "execution_time": time.time() - start_time
                }

            elif method == "permutation":
                importance = permutation_feature_importance(model, X, y, feature_names)
                plot_path = plot_importance(importance, "Permutation Importance", "Mean Decrease in Performance", f"permutation_importance_{model_name}.png")
                return {
                    "method": method,
                    "importance": importance,
                    "plot_path": plot_path,
                    "execution_time": time.time() - start_time
                }

            else:
                return {"error": f"Unknown or unsupported method: {method}", "method": method}

        except TimeoutError as te:
            return {"error": str(te), "method": method}
        except Exception as e:
            if getattr(getattr(self, 'config', None), 'debug_mode', False):
                self.logger.error(traceback.format_exc())
            return {"error": f"Explanation generation failed: {str(e)}", "method": method}

    
    def get_model_summary(self, model_name=None):
        """
        Get a summary of a model's information.
        
        Args:
            model_name: Name of the model (uses best model if None)
            
        Returns:
            Dictionary with model summary
        """
        if model_name is None and self.best_model is not None:
            model_info = self.best_model
            model_name = self.best_model_name
        elif model_name in self.models:
            model_info = self.models[model_name]
        else:
            self.logger.error(f"Model {model_name} not found")
            return {"error": f"Model {model_name} not found"}
            
        # Extract relevant info for summary
        summary = {
            "model_name": model_name,
            "model_type": type(model_info["model"]).__name__,
            "feature_count": len(model_info.get("feature_names", [])),
            "metrics": model_info.get("metrics", {}),
            "training_time": model_info.get("training_time", 0),
            "is_best_model": (model_name == self.best_model_name)
        }
        
        # Add top features by importance if available
        if "feature_importance" in model_info and model_info["feature_importance"] is not None:
            feature_importance = model_info["feature_importance"]
            feature_names = model_info.get("feature_names", [])
            
            # Create feature importance dictionary
            importance_dict = {}
            if isinstance(feature_importance, dict):
                importance_dict = feature_importance
            else:
                for i, importance in enumerate(feature_importance):
                    if i < len(feature_names):
                        importance_dict[feature_names[i]] = float(importance)
                    else:
                        importance_dict[f"feature_{i}"] = float(importance)
                    
            # Sort and get top 10
            top_features = {k: v for k, v in sorted(importance_dict.items(), 
                                                key=lambda item: item[1], reverse=True)[:10]}
            
            summary["top_features"] = top_features
            
        return summary

    def _evaluate_model(self, model, X, y, X_test=None, y_test=None):
        """
        Evaluate model performance with appropriate metrics.
        
        Args:
            model: Trained model to evaluate
            X: Training features
            y: Training targets
            X_test: Testing features
            y_test: Testing targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if X_test is not None and y_test is not None:
            # Get predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            metrics = {"prediction_time": pred_time}
            
            # Calculate metrics based on task type
            if self.config.task_type == TaskType.CLASSIFICATION:
                # Classification metrics
                metrics["accuracy"] = accuracy_score(y_test, y_pred)
                metrics["precision"] = precision_score(y_test, y_pred, average="weighted")
                metrics["recall"] = recall_score(y_test, y_pred, average="weighted")
                metrics["f1"] = f1_score(y_test, y_pred, average="weighted")
                
                # Add Matthews correlation coefficient
                metrics["matthews_correlation"] = matthews_corrcoef(y_test, y_pred)
                
                # Add ROC AUC if binary classification
                if len(np.unique(y)) == 2:
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_prob = model.predict_proba(X_test)[:, 1]
                            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
                    except (AttributeError, IndexError):
                        pass
                    
            elif self.config.task_type == TaskType.REGRESSION:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                metrics["mse"] = mse
                metrics["rmse"] = np.sqrt(mse)
                metrics["mae"] = mean_absolute_error(y_test, y_pred)
                metrics["r2"] = r2_score(y_test, y_pred)
                
                # Add explained variance
                metrics["explained_variance"] = explained_variance_score(y_test, y_pred)
                
                # Calculate median absolute error
                metrics["median_absolute_error"] = np.median(np.abs(y_test - y_pred))
                
                # Calculate MAPE if no zeros in y_test
                if not np.any(y_test == 0):
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                    metrics["mape"] = mape
            else:
                # Default metrics
                metrics["score"] = model.score(X_test, y_test)
                
            return metrics
        else:
            # Use cross-validation if no test set provided
            cv = self._get_cv_splitter(y)
            
            try:
                # Use cross_validate to get multiple metrics
                if self.config.task_type == TaskType.CLASSIFICATION:
                    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
                    if len(np.unique(y)) == 2:
                        scoring.append('roc_auc')
                elif self.config.task_type == TaskType.REGRESSION:
                    scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2', 'explained_variance']
                else:
                    scoring = None
                    
                # Run cross-validation
                cv_results = cross_validate(
                    model, X, y, 
                    cv=cv, 
                    scoring=scoring,
                    return_train_score=False
                )
                
                # Extract and rename metrics
                metrics = {}
                
                for key, values in cv_results.items():
                    if key.startswith('test_'):
                        metric_name = key[5:]  # Remove 'test_' prefix
                        
                        # Convert negative metrics to positive
                        if metric_name == 'neg_mean_squared_error':
                            metrics['mse'] = -np.mean(values)
                            metrics['rmse'] = np.sqrt(-np.mean(values))
                        elif metric_name == 'neg_mean_absolute_error':
                            metrics['mae'] = -np.mean(values)
                        else:
                            metrics[metric_name] = np.mean(values)
                
                return metrics
                
            except Exception as e:
                self.logger.warning(f"Error during cross-validation: {str(e)}")
                
                # Fallback to basic scoring
                try:
                    score = model.score(X, y)
                    return {"score": score}
                except:
                    return {}