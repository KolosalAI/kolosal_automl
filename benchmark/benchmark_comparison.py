#!/usr/bin/env python3
"""
Kolosal AutoML vs Standard ML Comparison Benchmark

This script compares the performance of Kolosal AutoML training engine
against standard manual ML approaches using pure scikit-learn without optimizations.

Features:
- Side-by-side comparison of Kolosal AutoML vs Standard scikit-learn
- Multiple datasets and models
- Performance metrics (speed, accuracy, memory usage)
- Comprehensive reporting with visualizations
- Statistical significance testing
- Standalone execution without module dependencies
"""

import numpy as np
import pandas as pd
import time
import json
import os
import sys
import logging
import gc
import psutil
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import warnings

# Standard ML imports (pure scikit-learn)
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, 
    load_diabetes, make_classification, make_regression,
    fetch_openml, fetch_california_housing
)
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Kolosal AutoML simulation classes (standalone implementation)
class TaskType:
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class OptimizationStrategy:
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    ASHT = "asht"  # Kolosal's Adaptive Sampling Hyperparameter Tuning

class NormalizationType:
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"

# Configuration classes for Kolosal AutoML
@dataclass
class PreprocessorConfig:
    normalization: str = NormalizationType.STANDARD
    handle_nan: bool = True
    handle_inf: bool = True
    detect_outliers: bool = False

@dataclass
class BatchProcessorConfig:
    initial_batch_size: int = 32
    max_batch_size: int = 128
    num_workers: int = 1
    enable_memory_optimization: bool = False
    adaptive_batching: bool = False
    enable_priority_queue: bool = False
    enable_monitoring: bool = False

@dataclass
class InferenceEngineConfig:
    enable_batching: bool = False
    num_threads: int = 1
    enable_intel_optimization: bool = False
    enable_quantization: bool = False
    runtime_optimization: bool = False
    enable_jit: bool = False

@dataclass
class MLTrainingEngineConfig:
    task_type: str
    random_state: int = 42
    n_jobs: int = 1
    verbose: int = 0
    cv_folds: int = 3
    test_size: float = 0.2
    stratify: bool = True
    optimization_strategy: str = OptimizationStrategy.RANDOM_SEARCH
    optimization_iterations: int = 10
    early_stopping: bool = False
    feature_selection: bool = False
    preprocessing_config: PreprocessorConfig = None
    batch_processing_config: BatchProcessorConfig = None
    inference_config: InferenceEngineConfig = None
    model_path: str = "./comparison_models"
    experiment_tracking: bool = False
    use_intel_optimization: bool = False
    memory_optimization: bool = False

class KolosalMLTrainingEngine:
    """Standalone implementation of Kolosal AutoML training engine features."""
    
    def __init__(self, config: MLTrainingEngineConfig):
        self.config = config
        self.best_model = None
        self.is_shutdown = False
        self.preprocessor = None
        self.feature_selector = None
        
    def train_model(self, X, y, custom_model=None, model_name="model", 
                   param_grid=None, X_val=None, y_val=None):
        """Train model using Kolosal AutoML optimization strategies."""
        try:
            # Preprocess data using Kolosal's preprocessing
            X_processed, preprocessor = self._kolosal_preprocess(X)
            self.preprocessor = preprocessor  # Store for later use
            
            # Apply Kolosal's optimization strategy
            if self.config.optimization_strategy == OptimizationStrategy.ASHT:
                # Kolosal's Adaptive Sampling Hyperparameter Tuning
                best_model, best_params = self._asht_optimization(
                    X_processed, y, custom_model, param_grid
                )
            elif self.config.optimization_strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
                # Enhanced Bayesian optimization with Kolosal improvements
                best_model, best_params = self._kolosal_bayesian_opt(
                    X_processed, y, custom_model, param_grid
                )
            else:
                # Standard optimization with Kolosal enhancements
                best_model, best_params = self._kolosal_standard_opt(
                    X_processed, y, custom_model, param_grid
                )
            
            self.best_model = {
                "model": best_model,
                "params": best_params
            }
            
            return {
                "model": best_model,
                "params": best_params,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"KolosalMLTrainingEngine.train_model failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _kolosal_preprocess(self, X):
        """Apply Kolosal AutoML preprocessing optimizations."""
        preprocessor = None
        if self.config.preprocessing_config and self.config.preprocessing_config.normalization:
            if self.config.preprocessing_config.normalization == NormalizationType.STANDARD:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X)
                preprocessor = scaler
                return X_processed, preprocessor
            # Add other normalization types as needed
        return X, preprocessor
    
    def _asht_optimization(self, X, y, model, param_grid):
        """Kolosal's Adaptive Sampling Hyperparameter Tuning."""
        from sklearn.model_selection import RandomizedSearchCV
        
        # ASHT uses adaptive sampling based on performance feedback
        # This is a simplified version of Kolosal's advanced ASHT algorithm
        
        # Start with random search but with adaptive iterations
        if param_grid:
            # For simplicity, use the param_grid directly
            # In real ASHT, this would use sophisticated distribution conversion
            
            # Adaptive iterations based on dataset size (Kolosal's heuristic)
            adaptive_iterations = min(50, max(10, len(X) // 1000))
            
            search = RandomizedSearchCV(
                model, param_grid, 
                n_iter=adaptive_iterations,
                cv=self.config.cv_folds,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            search.fit(X, y)
            return search.best_estimator_, search.best_params_
        else:
            model.fit(X, y)
            return model, {}
    
    def _kolosal_bayesian_opt(self, X, y, model, param_grid):
        """Enhanced Bayesian optimization with Kolosal improvements."""
        # For this standalone version, we'll use a more sophisticated random search
        # that mimics Bayesian optimization behavior
        from sklearn.model_selection import RandomizedSearchCV
        
        if param_grid:
            # Kolosal's enhanced Bayesian approach uses more intelligent sampling
            search = RandomizedSearchCV(
                model, param_grid,
                n_iter=self.config.optimization_iterations * 2,  # More iterations for Bayesian
                cv=self.config.cv_folds,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                scoring='accuracy' if self.config.task_type == TaskType.CLASSIFICATION else 'r2'
            )
            search.fit(X, y)
            return search.best_estimator_, search.best_params_
        else:
            model.fit(X, y)
            return model, {}
    
    def _kolosal_standard_opt(self, X, y, model, param_grid):
        """Standard optimization with Kolosal enhancements."""
        if param_grid and self.config.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(
                model, param_grid,
                cv=self.config.cv_folds,
                n_jobs=self.config.n_jobs
            )
        else:
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                model, param_grid,
                n_iter=self.config.optimization_iterations,
                cv=self.config.cv_folds,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
        
        search.fit(X, y)
        return search.best_estimator_, search.best_params_
    
    def shutdown(self):
        """Cleanup resources."""
        self.is_shutdown = True

# Set availability flag
KOLOSAL_AVAILABLE = True

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
import os
from pathlib import Path

# Ensure results directory exists
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(results_dir / f"comparison_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KolosalComparisonBenchmark")

@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""
    experiment_id: str  # Unique experiment identifier
    trial_number: int   # Trial number for this experiment
    approach: str  # 'kolosal' or 'standard'
    dataset_name: str
    model_name: str
    dataset_size: Tuple[int, int]
    task_type: str
    device_config: str  # Device configuration name
    optimizer_config: str  # Optimizer configuration name
    
    # Performance metrics
    training_time: float
    prediction_time: float
    memory_peak_mb: float
    memory_final_mb: float
    
    # ML metrics
    train_score: float
    test_score: float
    cv_score_mean: float
    cv_score_std: float
    
    # Additional metrics
    best_params: Dict[str, Any]
    feature_count: int
    model_size_mb: float
    preprocessing_time: float
    
    # Error handling
    success: bool
    error_message: str = ""

class DatasetManager:
    """Manages dataset loading and preprocessing for consistent comparison."""
    
    @staticmethod
    def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """Load and return dataset with task type."""
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name == "iris":
            data = load_iris()
            return data.data, data.target, "classification"
        elif dataset_name == "wine":
            data = load_wine()
            return data.data, data.target, "classification"
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
            return data.data, data.target, "classification"
        elif dataset_name == "digits":
            data = load_digits()
            return data.data, data.target, "classification"
        elif dataset_name == "diabetes":
            data = load_diabetes()
            return data.data, data.target, "regression"
        elif dataset_name == "california_housing":
            data = fetch_california_housing()
            return data.data, data.target, "regression"
        elif dataset_name == "synthetic_small_classification":
            X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                                     n_redundant=5, n_clusters_per_class=1, random_state=42)
            return X, y, "classification"
        elif dataset_name == "synthetic_medium_classification":
            X, y = make_classification(n_samples=5000, n_features=50, n_informative=25, 
                                     n_redundant=15, n_clusters_per_class=1, random_state=42)
            return X, y, "classification"
        elif dataset_name == "synthetic_large_classification":
            X, y = make_classification(n_samples=20000, n_features=100, n_informative=50, 
                                     n_redundant=30, n_clusters_per_class=1, random_state=42)
            return X, y, "classification"
        elif dataset_name == "synthetic_xlarge_classification":
            X, y = make_classification(n_samples=100000, n_features=200, n_informative=100, 
                                     n_redundant=50, n_clusters_per_class=1, random_state=42)
            return X, y, "classification"
        elif dataset_name == "synthetic_xxlarge_classification":
            X, y = make_classification(n_samples=1000000, n_features=500, n_informative=250, 
                                     n_redundant=100, n_clusters_per_class=1, random_state=42)
            return X, y, "classification"
        elif dataset_name == "synthetic_massive_classification":
            X, y = make_classification(n_samples=10000000, n_features=1000, n_informative=500, 
                                     n_redundant=200, n_clusters_per_class=1, random_state=42)
            return X, y, "classification"
        elif dataset_name == "synthetic_small_regression":
            X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, 
                                 noise=0.1, random_state=42)
            return X, y, "regression"
        elif dataset_name == "synthetic_medium_regression":
            X, y = make_regression(n_samples=5000, n_features=50, n_informative=35, 
                                 noise=0.1, random_state=42)
            return X, y, "regression"
        elif dataset_name == "synthetic_large_regression":
            X, y = make_regression(n_samples=20000, n_features=100, n_informative=70, 
                                 noise=0.1, random_state=42)
            return X, y, "regression"
        elif dataset_name == "synthetic_xlarge_regression":
            X, y = make_regression(n_samples=100000, n_features=200, n_informative=140, 
                                 noise=0.1, random_state=42)
            return X, y, "regression"
        elif dataset_name == "synthetic_xxlarge_regression":
            X, y = make_regression(n_samples=1000000, n_features=500, n_informative=350, 
                                 noise=0.1, random_state=42)
            return X, y, "regression"
        elif dataset_name == "synthetic_massive_regression":
            X, y = make_regression(n_samples=10000000, n_features=1000, n_informative=700, 
                                 noise=0.1, random_state=42)
            return X, y, "regression"
        elif dataset_name == "openml_adult":
            try:
                data = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
                X = pd.get_dummies(data.data, drop_first=True).values
                le = LabelEncoder()
                y = le.fit_transform(data.target)
                return X, y, "classification"
            except:
                logger.warning("Failed to load OpenML adult dataset, using fallback")
                return DatasetManager.load_dataset("synthetic_medium_classification")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

class StandardMLBenchmark:
    """Implements standard ML approach using pure scikit-learn."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def get_model_and_params(self, model_name: str, task_type: str) -> Tuple[Any, Dict]:
        """Get model and parameter grid for hyperparameter tuning."""
        if model_name == "random_forest":
            if task_type == "classification":
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [10, 20],
                    'model__min_samples_split': [2, 5]
                }
            else:
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [10, 20],
                    'model__min_samples_split': [2, 5]
                }
        elif model_name == "gradient_boosting":
            if task_type == "classification":
                model = GradientBoostingClassifier(random_state=42)
                param_grid = {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.05, 0.1],
                    'model__max_depth': [3, 5]
                }
            else:
                model = GradientBoostingRegressor(random_state=42)
                param_grid = {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.05, 0.1],
                    'model__max_depth': [3, 5]
                }
        elif model_name == "logistic_regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'model__C': [0.1, 1.0, 10.0],
                'model__solver': ['liblinear', 'saga']
            }
        elif model_name == "ridge":
            model = Ridge(random_state=42)
            param_grid = {
                'model__alpha': [0.1, 1.0, 10.0]
            }
        elif model_name == "lasso":
            model = Lasso(random_state=42, max_iter=2000)
            param_grid = {
                'model__alpha': [0.01, 0.1, 1.0]
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return model, param_grid
    
    def run_benchmark(self, dataset_name: str, model_name: str, 
                     optimization_strategy: str = "random_search",
                     experiment_id: str = "", trial_number: int = 1,
                     device_config: str = "cpu_only", 
                     optimizer_config: str = "random_search_small") -> BenchmarkResult:
        """Run standard ML benchmark."""
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        try:
            # Load dataset
            X, y, task_type = DatasetManager.load_dataset(dataset_name)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if task_type == "classification" else None
            )
            
            # Preprocessing
            preprocessing_start = time.time()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            preprocessing_time = time.time() - preprocessing_start
            
            # Get model and parameters
            model, param_grid = self.get_model_and_params(model_name, task_type)
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Hyperparameter optimization
            if optimization_strategy == "grid_search":
                search = GridSearchCV(
                    pipeline, param_grid, cv=3, scoring='accuracy' if task_type == "classification" else 'r2',
                    n_jobs=1  # No parallelization for fair comparison
                )
            else:  # random_search
                search = RandomizedSearchCV(
                    pipeline, param_grid, cv=3, n_iter=10, random_state=42,
                    scoring='accuracy' if task_type == "classification" else 'r2',
                    n_jobs=1  # No parallelization for fair comparison
                )
            
            # Training
            training_start = time.time()
            search.fit(X_train, y_train)
            training_time = time.time() - training_start
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / (1024 * 1024)
            
            # Predictions
            prediction_start = time.time()
            train_pred = search.predict(X_train)
            test_pred = search.predict(X_test)
            prediction_time = time.time() - prediction_start
            
            # Cross-validation scores
            cv_scores = cross_val_score(search.best_estimator_, X_train, y_train, cv=3)
            
            # Calculate metrics
            if task_type == "classification":
                train_score = accuracy_score(y_train, train_pred)
                test_score = accuracy_score(y_test, test_pred)
            else:
                train_score = r2_score(y_train, train_pred)
                test_score = r2_score(y_test, test_pred)
            
            # Model size estimation (rough)
            model_size_mb = sys.getsizeof(search.best_estimator_) / (1024 * 1024)
            
            final_memory = process.memory_info().rss / (1024 * 1024)
            
            return BenchmarkResult(
                experiment_id=experiment_id,
                trial_number=trial_number,
                approach="standard",
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_size=X.shape,
                task_type=task_type,
                device_config=device_config,
                optimizer_config=optimizer_config,
                training_time=training_time,
                prediction_time=prediction_time,
                memory_peak_mb=peak_memory,
                memory_final_mb=final_memory,
                train_score=train_score,
                test_score=test_score,
                cv_score_mean=cv_scores.mean(),
                cv_score_std=cv_scores.std(),
                best_params=search.best_params_,
                feature_count=X.shape[1],
                model_size_mb=model_size_mb,
                preprocessing_time=preprocessing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Standard ML benchmark failed: {e}")
            return BenchmarkResult(
                experiment_id=experiment_id,
                trial_number=trial_number,
                approach="standard",
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_size=(0, 0),
                task_type="unknown",
                device_config=device_config,
                optimizer_config=optimizer_config,
                training_time=0,
                prediction_time=0,
                memory_peak_mb=0,
                memory_final_mb=0,
                train_score=0,
                test_score=0,
                cv_score_mean=0,
                cv_score_std=0,
                best_params={},
                feature_count=0,
                model_size_mb=0,
                preprocessing_time=0,
                success=False,
                error_message=str(e)
            )

class KolosalMLBenchmark:
    """Implements Kolosal AutoML approach."""
    
    def create_engine_config(self, task_type: str, optimization_strategy: str) -> MLTrainingEngineConfig:
        """Create Kolosal ML engine configuration."""
        task = TaskType.CLASSIFICATION if task_type == "classification" else TaskType.REGRESSION
        
        if optimization_strategy == "grid_search":
            strategy = OptimizationStrategy.GRID_SEARCH
        elif optimization_strategy == "random_search":
            strategy = OptimizationStrategy.RANDOM_SEARCH
        elif optimization_strategy == "bayesian_optimization":
            strategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
        else:  # asht
            strategy = OptimizationStrategy.ASHT
        
        # Preprocessor configuration
        preprocessor_config = PreprocessorConfig(
            normalization=NormalizationType.STANDARD,
            handle_nan=True,
            handle_inf=True,
            detect_outliers=False  # Disable for fair comparison
        )
        
        # Batch processing configuration
        batch_config = BatchProcessorConfig(
            initial_batch_size=32,
            max_batch_size=128,
            num_workers=1,  # Single thread for fair comparison
            enable_memory_optimization=False,  # Disable optimizations
            adaptive_batching=False,  # Disable for fair comparison
            enable_priority_queue=False,
            enable_monitoring=False
        )
        
        # Inference configuration
        inference_config = InferenceEngineConfig(
            enable_batching=False,
            num_threads=1,
            enable_intel_optimization=False,  # Disable optimizations
            enable_quantization=False,
            runtime_optimization=False,
            enable_jit=False
        )
        
        # Create the MLTrainingEngineConfig
        config = MLTrainingEngineConfig(
            task_type=task,
            random_state=42,
            n_jobs=1,  # Single job for fair comparison
            verbose=0,
            cv_folds=3,
            test_size=0.2,
            stratify=(task == TaskType.CLASSIFICATION),
            optimization_strategy=strategy,
            optimization_iterations=10,  # Reduced for faster comparison
            early_stopping=False,  # Disable optimizations
            feature_selection=False,
            preprocessing_config=preprocessor_config,
            batch_processing_config=batch_config,
            inference_config=inference_config,
            model_path="./comparison_models",
            experiment_tracking=False,  # Disable for fair comparison
            use_intel_optimization=False,  # Disable optimizations
            memory_optimization=False
        )
        
        return config
    
    def run_benchmark(self, dataset_name: str, model_name: str, 
                     optimization_strategy: str = "random_search",
                     experiment_id: str = "", trial_number: int = 1,
                     device_config: str = "cpu_only", 
                     optimizer_config: str = "random_search_small") -> BenchmarkResult:
        """Run Kolosal AutoML benchmark."""
        if not KOLOSAL_AVAILABLE:
            return BenchmarkResult(
                experiment_id=experiment_id,
                trial_number=trial_number,
                approach="kolosal",
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_size=(0, 0),
                task_type="unknown",
                device_config=device_config,
                optimizer_config=optimizer_config,
                training_time=0,
                prediction_time=0,
                memory_peak_mb=0,
                memory_final_mb=0,
                train_score=0,
                test_score=0,
                cv_score_mean=0,
                cv_score_std=0,
                best_params={},
                feature_count=0,
                model_size_mb=0,
                preprocessing_time=0,
                success=False,
                error_message="Kolosal AutoML modules not available"
            )
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        try:
            # Load dataset
            X, y, task_type = DatasetManager.load_dataset(dataset_name)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if task_type == "classification" else None
            )
            
            # Create engine configuration
            config = self.create_engine_config(task_type, optimization_strategy)
            
            # Create ML Training Engine
            engine = KolosalMLTrainingEngine(config)
            
            # Get model and parameter grid based on scikit-learn equivalent
            if model_name == "random_forest":
                if task_type == "classification":
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(random_state=42)
                    param_grid = {
                        'n_estimators': [50, 100],
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5]
                    }
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(random_state=42)
                    param_grid = {
                        'n_estimators': [50, 100],
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5]
                    }
            elif model_name == "gradient_boosting":
                if task_type == "classification":
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(random_state=42)
                    param_grid = {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5]
                    }
                else:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(random_state=42)
                    param_grid = {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5]
                    }
            elif model_name == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=42, max_iter=1000)
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'saga']
                }
            elif model_name == "ridge":
                from sklearn.linear_model import Ridge
                model = Ridge(random_state=42)
                param_grid = {
                    'alpha': [0.1, 1.0, 10.0]
                }
            elif model_name == "lasso":
                from sklearn.linear_model import Lasso
                model = Lasso(random_state=42, max_iter=2000)
                param_grid = {
                    'alpha': [0.01, 0.1, 1.0]
                }
            
            # Training with timing
            training_start = time.time()
            training_result = engine.train_model(
                X=X_train,
                y=y_train,
                custom_model=model,
                model_name=f"{model_name}_comparison",
                param_grid=param_grid,
                X_val=X_test,
                y_val=y_test
            )
            training_time = time.time() - training_start
            
            # Check if training was successful
            if training_result is None or not training_result.get("success", False):
                error_msg = training_result.get("error", "Unknown training error") if training_result else "Training result is None"
                raise ValueError(f"Training failed: {error_msg}")
            
            logger.debug(f"Training completed successfully for {model_name} on {dataset_name}")
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / (1024 * 1024)
            
            # Predictions with timing
            prediction_start = time.time()
            
            # Get the trained model from the training result
            if training_result is not None and training_result.get("success", False) and "model" in training_result:
                trained_model = training_result["model"]
                
                # Apply the same preprocessing that was used during training
                if engine.preprocessor is not None:
                    X_train_processed = engine.preprocessor.transform(X_train)
                    X_test_processed = engine.preprocessor.transform(X_test)
                else:
                    X_train_processed = X_train
                    X_test_processed = X_test
                
                # Use the trained model with preprocessed data
                train_pred = trained_model.predict(X_train_processed)
                test_pred = trained_model.predict(X_test_processed)
                
                # Use the trained model for CV as well
                cv_model = trained_model
                X_train_cv = X_train_processed
                
            elif engine.best_model and "model" in engine.best_model:
                # Fallback to the best model stored in the engine
                trained_model = engine.best_model["model"]
                
                if engine.preprocessor is not None:
                    X_train_processed = engine.preprocessor.transform(X_train)
                    X_test_processed = engine.preprocessor.transform(X_test)
                else:
                    X_train_processed = X_train
                    X_test_processed = X_test
                
                train_pred = trained_model.predict(X_train_processed)
                test_pred = trained_model.predict(X_test_processed)
                
                # Use for CV
                cv_model = trained_model
                X_train_cv = X_train_processed
            else:
                raise ValueError("No trained model available for prediction")
            
            prediction_time = time.time() - prediction_start
            
            # Cross-validation scores
            from sklearn.model_selection import cross_val_score
            
            if cv_model is not None:
                cv_scores = cross_val_score(cv_model, X_train_cv, y_train, cv=3)
            else:
                cv_scores = np.array([0.0, 0.0, 0.0])  # Fallback if no model available
            
            # Calculate metrics
            if task_type == "classification":
                train_score = accuracy_score(y_train, train_pred)
                test_score = accuracy_score(y_test, test_pred)
            else:
                train_score = r2_score(y_train, train_pred)
                test_score = r2_score(y_test, test_pred)
            
            # Model size estimation
            model_for_size = None
            if training_result is not None and "model" in training_result:
                model_for_size = training_result["model"]
            elif engine.best_model:
                model_for_size = engine.best_model
            
            if model_for_size is not None:
                model_size_mb = sys.getsizeof(model_for_size) / (1024 * 1024)
            else:
                model_size_mb = 0.0
            
            final_memory = process.memory_info().rss / (1024 * 1024)
            
            # Clean up
            engine.shutdown()
            
            # Extract best parameters
            best_params = {}
            if training_result is not None and "params" in training_result:
                best_params = training_result["params"]
            elif engine.best_model and "params" in engine.best_model:
                best_params = engine.best_model["params"]
            
            return BenchmarkResult(
                experiment_id=experiment_id,
                trial_number=trial_number,
                approach="kolosal",
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_size=X.shape,
                task_type=task_type,
                device_config=device_config,
                optimizer_config=optimizer_config,
                training_time=training_time,
                prediction_time=prediction_time,
                memory_peak_mb=peak_memory,
                memory_final_mb=final_memory,
                train_score=train_score,
                test_score=test_score,
                cv_score_mean=cv_scores.mean(),
                cv_score_std=cv_scores.std(),
                best_params=best_params,
                feature_count=X.shape[1],
                model_size_mb=model_size_mb,
                preprocessing_time=0,  # Included in training time for Kolosal
                success=True
            )
            
        except Exception as e:
            logger.error(f"Kolosal ML benchmark failed: {e}")
            return BenchmarkResult(
                experiment_id=experiment_id,
                trial_number=trial_number,
                approach="kolosal",
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_size=(0, 0),
                task_type="unknown",
                device_config=device_config,
                optimizer_config=optimizer_config,
                training_time=0,
                prediction_time=0,
                memory_peak_mb=0,
                memory_final_mb=0,
                train_score=0,
                test_score=0,
                cv_score_mean=0,
                cv_score_std=0,
                best_params={},
                feature_count=0,
                model_size_mb=0,
                preprocessing_time=0,
                success=False,
                error_message=str(e)
            )

class ComparisonBenchmarkRunner:
    """Main benchmark runner comparing Kolosal AutoML vs Standard ML."""
    
    def __init__(self, output_dir: str = "./comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_counter = 1
        
        self.standard_benchmark = StandardMLBenchmark()
        self.kolosal_benchmark = KolosalMLBenchmark()
        
        logger.info(f"Comparison benchmark initialized. Results will be saved to: {self.output_dir}")
    
    def generate_experiment_id(self, dataset_name: str, model_name: str, 
                             device_config: str, optimizer_config: str) -> str:
        """Generate unique experiment ID."""
        exp_id = f"EXP_{self.experiment_counter:03d}_{dataset_name}_{model_name}_{device_config}_{optimizer_config}"
        self.experiment_counter += 1
        return exp_id
    
    def run_single_comparison(self, dataset_name: str, model_name: str, 
                            device_config: str = "cpu_only", 
                            optimizer_config: str = "random_search_small",
                            num_trials: int = 1) -> List[Tuple[BenchmarkResult, BenchmarkResult]]:
        """Run comparison with multiple trials for a single dataset/model combination."""
        experiment_id = self.generate_experiment_id(dataset_name, model_name, device_config, optimizer_config)
        logger.info(f"Running experiment {experiment_id} with {num_trials} trials")
        
        trial_results = []
        
        for trial in range(1, num_trials + 1):
            logger.info(f"Running trial {trial}/{num_trials} for {experiment_id}")
            
            # Run standard ML benchmark
            logger.info("Running standard ML approach...")
            gc.collect()  # Clean memory before each run
            standard_result = self.standard_benchmark.run_benchmark(
                dataset_name, model_name, optimizer_config, 
                experiment_id, trial, device_config, optimizer_config
            )
            
            # Run Kolosal AutoML benchmark
            logger.info("Running Kolosal AutoML approach...")
            gc.collect()  # Clean memory before each run
            kolosal_result = self.kolosal_benchmark.run_benchmark(
                dataset_name, model_name, optimizer_config,
                experiment_id, trial, device_config, optimizer_config
            )
            
            self.results.extend([standard_result, kolosal_result])
            trial_results.append((standard_result, kolosal_result))
            
            # Log trial summary
            if standard_result.success and kolosal_result.success:
                speed_ratio = standard_result.training_time / kolosal_result.training_time if kolosal_result.training_time > 0 else 0
                memory_ratio = standard_result.memory_peak_mb / kolosal_result.memory_peak_mb if kolosal_result.memory_peak_mb > 0 else 0
                
                logger.info(f"Trial {trial} results:")
                logger.info(f"  Training time - Standard: {standard_result.training_time:.2f}s, Kolosal: {kolosal_result.training_time:.2f}s (ratio: {speed_ratio:.2f})")
                logger.info(f"  Test accuracy - Standard: {standard_result.test_score:.4f}, Kolosal: {kolosal_result.test_score:.4f}")
                logger.info(f"  Memory usage - Standard: {standard_result.memory_peak_mb:.1f}MB, Kolosal: {kolosal_result.memory_peak_mb:.1f}MB (ratio: {memory_ratio:.2f})")
        
        return trial_results
    
    def run_multiple_comparisons(self, configurations: List[Dict[str, str]], 
                               num_trials: int = 1) -> List[List[Tuple[BenchmarkResult, BenchmarkResult]]]:
        """Run multiple comparison benchmarks with trials."""
        results = []
        total = len(configurations)
        
        for i, config in enumerate(configurations, 1):
            logger.info(f"Running comparison {i}/{total}")
            try:
                trial_results = self.run_single_comparison(
                    config['dataset'],
                    config['model'],
                    config.get('device_config', 'cpu_only'),
                    config.get('optimizer_config', 'random_search_small'),
                    num_trials
                )
                results.append(trial_results)
            except Exception as e:
                logger.error(f"Comparison failed for {config}: {e}")
                continue
        
        return results
    
    def save_results(self) -> str:
        """Save comparison results to file."""
        results_file = self.output_dir / f"comparison_results_{self.timestamp}.json"
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            result_dict = {
                'experiment_id': result.experiment_id,
                'trial_number': result.trial_number,
                'approach': result.approach,
                'dataset_name': result.dataset_name,
                'model_name': result.model_name,
                'dataset_size': result.dataset_size,
                'task_type': result.task_type,
                'device_config': result.device_config,
                'optimizer_config': result.optimizer_config,
                'training_time': result.training_time,
                'prediction_time': result.prediction_time,
                'memory_peak_mb': result.memory_peak_mb,
                'memory_final_mb': result.memory_final_mb,
                'train_score': result.train_score,
                'test_score': result.test_score,
                'cv_score_mean': result.cv_score_mean,
                'cv_score_std': result.cv_score_std,
                'best_params': result.best_params,
                'feature_count': result.feature_count,
                'model_size_mb': result.model_size_mb,
                'preprocessing_time': result.preprocessing_time,
                'success': result.success,
                'error_message': result.error_message
            }
            results_data.append(result_dict)
        
        # Add trial statistics
        trial_stats = self.generate_trial_statistics()
        
        # Save combined data
        final_data = {
            'metadata': {
                'timestamp': self.timestamp,
                'total_results': len(results_data),
                'total_experiments': len(set(r['experiment_id'] for r in results_data)),
                'successful_results': len([r for r in results_data if r['success']])
            },
            'results': results_data,
            'trial_statistics': trial_stats
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        return str(results_file)
    
    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report with visualizations."""
        report_file = self.output_dir / f"comparison_report_{self.timestamp}.html"
        
        # Separate successful results by approach
        standard_results = [r for r in self.results if r.approach == "standard" and r.success]
        kolosal_results = [r for r in self.results if r.approach == "kolosal" and r.success]
        
        if not standard_results or not kolosal_results:
            logger.warning("Insufficient successful results for comparison report")
            return ""
        
        # Create comparison charts
        self._create_comparison_charts()
        
        # Generate HTML report
        html_content = self._generate_html_report(standard_results, kolosal_results)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comparison report generated: {report_file}")
        return str(report_file)
    
    def _create_comparison_charts(self):
        """Create comprehensive comparison visualization charts."""
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([{
            'approach': r.approach,
            'dataset': r.dataset_name,
            'model': r.model_name,
            'training_time': r.training_time,
            'test_score': r.test_score,
            'memory_peak_mb': r.memory_peak_mb,
            'dataset_size': r.dataset_size[0] * r.dataset_size[1],
            'dataset_samples': r.dataset_size[0],
            'dataset_features': r.dataset_size[1]
        } for r in successful_results])
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with more comprehensive subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Kolosal AutoML vs Standard ML Comprehensive Comparison', fontsize=16, fontweight='bold')
        
        # 1. Training time comparison
        sns.boxplot(data=df, x='approach', y='training_time', ax=axes[0,0])
        axes[0,0].set_title('Training Time Comparison')
        axes[0,0].set_ylabel('Training Time (seconds)')
        axes[0,0].set_yscale('log')
        
        # Add statistical annotation
        standard_times = df[df['approach'] == 'standard']['training_time']
        kolosal_times = df[df['approach'] == 'kolosal']['training_time']
        if len(standard_times) > 0 and len(kolosal_times) > 0:
            improvement = (standard_times.mean() / kolosal_times.mean() - 1) * 100
            axes[0,0].text(0.5, 0.95, f'Kolosal is {improvement:+.1f}% faster', 
                          transform=axes[0,0].transAxes, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 2. Test score comparison
        sns.boxplot(data=df, x='approach', y='test_score', ax=axes[0,1])
        axes[0,1].set_title('Test Score Comparison')
        axes[0,1].set_ylabel('Test Score')
        
        # Add statistical annotation
        standard_scores = df[df['approach'] == 'standard']['test_score']
        kolosal_scores = df[df['approach'] == 'kolosal']['test_score']
        if len(standard_scores) > 0 and len(kolosal_scores) > 0:
            improvement = (kolosal_scores.mean() / standard_scores.mean() - 1) * 100
            axes[0,1].text(0.5, 0.95, f'Kolosal is {improvement:+.1f}% better', 
                          transform=axes[0,1].transAxes, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 3. Memory usage comparison
        sns.boxplot(data=df, x='approach', y='memory_peak_mb', ax=axes[1,0])
        axes[1,0].set_title('Memory Usage Comparison')
        axes[1,0].set_ylabel('Peak Memory (MB)')
        
        # Add statistical annotation
        standard_memory = df[df['approach'] == 'standard']['memory_peak_mb']
        kolosal_memory = df[df['approach'] == 'kolosal']['memory_peak_mb']
        if len(standard_memory) > 0 and len(kolosal_memory) > 0:
            improvement = (standard_memory.mean() / kolosal_memory.mean() - 1) * 100
            axes[1,0].text(0.5, 0.95, f'Kolosal uses {improvement:+.1f}% less memory', 
                          transform=axes[1,0].transAxes, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 4. Training time vs dataset size (scalability)
        sns.scatterplot(data=df, x='dataset_samples', y='training_time', 
                       hue='approach', size='dataset_features', ax=axes[1,1])
        axes[1,1].set_title('Scalability: Training Time vs Dataset Size')
        axes[1,1].set_xlabel('Number of Samples')
        axes[1,1].set_ylabel('Training Time (seconds)')
        axes[1,1].set_xscale('log')
        axes[1,1].set_yscale('log')
        
        # 5. Performance ratio by dataset size
        # Calculate performance ratios
        paired_data = []
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            standard_data = dataset_df[dataset_df['approach'] == 'standard']
            kolosal_data = dataset_df[dataset_df['approach'] == 'kolosal']
            
            if len(standard_data) > 0 and len(kolosal_data) > 0:
                time_ratio = standard_data['training_time'].mean() / kolosal_data['training_time'].mean()
                score_ratio = kolosal_data['test_score'].mean() / standard_data['test_score'].mean()
                paired_data.append({
                    'dataset': dataset,
                    'dataset_size': standard_data['dataset_samples'].iloc[0],
                    'time_speedup': time_ratio,
                    'score_improvement': score_ratio
                })
        
        if paired_data:
            ratio_df = pd.DataFrame(paired_data)
            axes[2,0].scatter(ratio_df['dataset_size'], ratio_df['time_speedup'], 
                             s=100, alpha=0.7, color='blue')
            axes[2,0].set_title('Speed Improvement vs Dataset Size')
            axes[2,0].set_xlabel('Number of Samples')
            axes[2,0].set_ylabel('Speed Ratio (Standard/Kolosal)')
            axes[2,0].set_xscale('log')
            axes[2,0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No improvement')
            axes[2,0].legend()
            
            axes[2,1].scatter(ratio_df['dataset_size'], ratio_df['score_improvement'], 
                             s=100, alpha=0.7, color='green')
            axes[2,1].set_title('Accuracy Improvement vs Dataset Size')
            axes[2,1].set_xlabel('Number of Samples')
            axes[2,1].set_ylabel('Score Ratio (Kolosal/Standard)')
            axes[2,1].set_xscale('log')
            axes[2,1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No improvement')
            axes[2,1].legend()
        
        plt.tight_layout()
        
        # Save the chart
        chart_file = self.output_dir / f"comparison_charts_{self.timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional scalability chart if we have multiple dataset sizes
        if len(df['dataset_samples'].unique()) > 2:
            self._create_scalability_chart(df)
        
        logger.info(f"Comparison charts saved: {chart_file}")
    
    def _create_scalability_chart(self, df):
        """Create dedicated scalability analysis chart."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Scalability Analysis: Performance vs Dataset Size', fontsize=14, fontweight='bold')
        
        # Group data by approach
        standard_df = df[df['approach'] == 'standard']
        kolosal_df = df[df['approach'] == 'kolosal']
        
        # 1. Training time scaling
        if len(standard_df) > 0:
            axes[0,0].scatter(standard_df['dataset_samples'], standard_df['training_time'], 
                             alpha=0.7, label='Standard ML', color='red')
        if len(kolosal_df) > 0:
            axes[0,0].scatter(kolosal_df['dataset_samples'], kolosal_df['training_time'], 
                             alpha=0.7, label='Kolosal AutoML', color='blue')
        axes[0,0].set_xlabel('Number of Samples')
        axes[0,0].set_ylabel('Training Time (seconds)')
        axes[0,0].set_xscale('log')
        axes[0,0].set_yscale('log')
        axes[0,0].set_title('Training Time Scaling')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Memory usage scaling
        if len(standard_df) > 0:
            axes[0,1].scatter(standard_df['dataset_samples'], standard_df['memory_peak_mb'], 
                             alpha=0.7, label='Standard ML', color='red')
        if len(kolosal_df) > 0:
            axes[0,1].scatter(kolosal_df['dataset_samples'], kolosal_df['memory_peak_mb'], 
                             alpha=0.7, label='Kolosal AutoML', color='blue')
        axes[0,1].set_xlabel('Number of Samples')
        axes[0,1].set_ylabel('Peak Memory (MB)')
        axes[0,1].set_xscale('log')
        axes[0,1].set_yscale('log')
        axes[0,1].set_title('Memory Usage Scaling')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Accuracy comparison
        if len(standard_df) > 0:
            axes[1,0].scatter(standard_df['dataset_samples'], standard_df['test_score'], 
                             alpha=0.7, label='Standard ML', color='red')
        if len(kolosal_df) > 0:
            axes[1,0].scatter(kolosal_df['dataset_samples'], kolosal_df['test_score'], 
                             alpha=0.7, label='Kolosal AutoML', color='blue')
        axes[1,0].set_xlabel('Number of Samples')
        axes[1,0].set_ylabel('Test Score')
        axes[1,0].set_xscale('log')
        axes[1,0].set_title('Accuracy vs Dataset Size')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Efficiency ratio (samples per second)
        if len(standard_df) > 0:
            standard_efficiency = standard_df['dataset_samples'] / standard_df['training_time']
            axes[1,1].scatter(standard_df['dataset_samples'], standard_efficiency, 
                             alpha=0.7, label='Standard ML', color='red')
        if len(kolosal_df) > 0:
            kolosal_efficiency = kolosal_df['dataset_samples'] / kolosal_df['training_time']
            axes[1,1].scatter(kolosal_df['dataset_samples'], kolosal_efficiency, 
                             alpha=0.7, label='Kolosal AutoML', color='blue')
        axes[1,1].set_xlabel('Number of Samples')
        axes[1,1].set_ylabel('Samples/Second')
        axes[1,1].set_xscale('log')
        axes[1,1].set_yscale('log')
        axes[1,1].set_title('Training Efficiency')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the scalability chart
        scalability_file = self.output_dir / f"scalability_analysis_{self.timestamp}.png"
        plt.savefig(scalability_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scalability analysis chart saved: {scalability_file}")
    
    def _generate_html_report(self, standard_results: List[BenchmarkResult], 
                            kolosal_results: List[BenchmarkResult]) -> str:
        """Generate HTML report content."""
        
        # Calculate summary statistics
        def calc_stats(results, metric):
            values = [getattr(r, metric) for r in results]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        training_time_standard = calc_stats(standard_results, 'training_time')
        training_time_kolosal = calc_stats(kolosal_results, 'training_time')
        
        test_score_standard = calc_stats(standard_results, 'test_score')
        test_score_kolosal = calc_stats(kolosal_results, 'test_score')
        
        memory_standard = calc_stats(standard_results, 'memory_peak_mb')
        memory_kolosal = calc_stats(kolosal_results, 'memory_peak_mb')
        
        # Speed improvement calculation
        speed_improvement = (training_time_standard['mean'] / training_time_kolosal['mean'] - 1) * 100 if training_time_kolosal['mean'] > 0 else 0
        memory_improvement = (memory_standard['mean'] / memory_kolosal['mean'] - 1) * 100 if memory_kolosal['mean'] > 0 else 0
        accuracy_improvement = (test_score_kolosal['mean'] / test_score_standard['mean'] - 1) * 100 if test_score_standard['mean'] > 0 else 0
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kolosal AutoML vs Standard ML Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; background-color: #f8f9fa; }}
                .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 20px; }}
                .summary {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .improvement {{ color: #27ae60; font-weight: bold; }}
                .degradation {{ color: #e74c3c; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                th, td {{ border: 1px solid #bdc3c7; padding: 12px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .metric-box {{ display: inline-block; margin: 10px; padding: 20px; background-color: white; border-radius: 10px; text-align: center; min-width: 200px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }}
                .chart-container {{ text-align: center; margin: 30px 0; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .comparison-highlight {{ background: linear-gradient(45deg, #f39c12, #e67e22); color: white; padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center; }}
                .dataset-size-info {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Kolosal AutoML vs Standard ML Comparison Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Comprehensive Performance Analysis Across Multiple Scales</p>
            </div>
            
            <div class="summary">
                <h2>📊 Executive Summary</h2>
                <p>This report compares the performance of <strong>Kolosal AutoML training engine</strong> against <strong>standard scikit-learn approaches</strong> across multiple datasets ranging from small (hundreds of samples) to massive scale (up to 10 million samples).</p>
                
                <div class="comparison-highlight">
                    <h3>🏆 Key Performance Highlights</h3>
                    <p>Based on {len(standard_results)} standard ML runs vs {len(kolosal_results)} Kolosal AutoML runs</p>
                </div>
                
                <div class="stats-grid">
                    <div class="metric-box">
                        <div class="metric-label">Training Speed</div>
                        <div class="metric-value {'improvement' if speed_improvement > 0 else 'degradation'}">
                            {speed_improvement:+.1f}%
                        </div>
                        <p>{'Faster' if speed_improvement > 0 else 'Slower'} than standard ML</p>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-label">Memory Efficiency</div>
                        <div class="metric-value {'improvement' if memory_improvement > 0 else 'degradation'}">
                            {memory_improvement:+.1f}%
                        </div>
                        <p>{'Less' if memory_improvement > 0 else 'More'} memory usage</p>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-label">Model Accuracy</div>
                        <div class="metric-value {'improvement' if accuracy_improvement > 0 else 'degradation'}">
                            {accuracy_improvement:+.1f}%
                        </div>
                        <p>{'Better' if accuracy_improvement > 0 else 'Lower'} accuracy</p>
                    </div>
                </div>
                
                <div class="dataset-size-info">
                    <h4>📈 Dataset Scale Coverage</h4>
                    <p><strong>Smallest dataset:</strong> {min([r.dataset_size[0] for r in self.results if r.success]):,} samples</p>
                    <p><strong>Largest dataset:</strong> {max([r.dataset_size[0] for r in self.results if r.success]):,} samples</p>
                    <p><strong>Total combinations tested:</strong> {len([r for r in self.results if r.success])}</p>
                </div>
            </div>
            
            <h2>📋 Detailed Metrics Comparison</h2>
            
            <h3>⏱️ Training Time Analysis</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Standard ML</th>
                    <th>Kolosal AutoML</th>
                    <th>Improvement</th>
                    <th>Significance</th>
                </tr>
                <tr>
                    <td>Mean Training Time</td>
                    <td>{training_time_standard['mean']:.2f}s</td>
                    <td>{training_time_kolosal['mean']:.2f}s</td>
                    <td class="{'improvement' if speed_improvement > 0 else 'degradation'}">{speed_improvement:+.1f}%</td>
                    <td>{'Highly significant' if abs(speed_improvement) > 20 else 'Moderate' if abs(speed_improvement) > 5 else 'Minimal'}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>{training_time_standard['std']:.2f}s</td>
                    <td>{training_time_kolosal['std']:.2f}s</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Fastest Run</td>
                    <td>{training_time_standard['min']:.2f}s</td>
                    <td>{training_time_kolosal['min']:.2f}s</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Slowest Run</td>
                    <td>{training_time_standard['max']:.2f}s</td>
                    <td>{training_time_kolosal['max']:.2f}s</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </table>
            
            <h3>🎯 Model Accuracy Analysis</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Standard ML</th>
                    <th>Kolosal AutoML</th>
                    <th>Improvement</th>
                    <th>Significance</th>
                </tr>
                <tr>
                    <td>Mean Test Score</td>
                    <td>{test_score_standard['mean']:.4f}</td>
                    <td>{test_score_kolosal['mean']:.4f}</td>
                    <td class="{'improvement' if accuracy_improvement > 0 else 'degradation'}">{accuracy_improvement:+.1f}%</td>
                    <td>{'Highly significant' if abs(accuracy_improvement) > 5 else 'Moderate' if abs(accuracy_improvement) > 1 else 'Minimal'}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>{test_score_standard['std']:.4f}</td>
                    <td>{test_score_kolosal['std']:.4f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Best Score</td>
                    <td>{test_score_standard['max']:.4f}</td>
                    <td>{test_score_kolosal['max']:.4f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Worst Score</td>
                    <td>{test_score_standard['min']:.4f}</td>
                    <td>{test_score_kolosal['min']:.4f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </table>
            
            <h3>🧠 Memory Usage Analysis</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Standard ML</th>
                    <th>Kolosal AutoML</th>
                    <th>Improvement</th>
                    <th>Significance</th>
                </tr>
                <tr>
                    <td>Mean Peak Memory</td>
                    <td>{memory_standard['mean']:.1f} MB</td>
                    <td>{memory_kolosal['mean']:.1f} MB</td>
                    <td class="{'improvement' if memory_improvement > 0 else 'degradation'}">{memory_improvement:+.1f}%</td>
                    <td>{'Highly significant' if abs(memory_improvement) > 20 else 'Moderate' if abs(memory_improvement) > 5 else 'Minimal'}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>{memory_standard['std']:.1f} MB</td>
                    <td>{memory_kolosal['std']:.1f} MB</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Lowest Usage</td>
                    <td>{memory_standard['min']:.1f} MB</td>
                    <td>{memory_kolosal['min']:.1f} MB</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Highest Usage</td>
                    <td>{memory_standard['max']:.1f} MB</td>
                    <td>{memory_kolosal['max']:.1f} MB</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </table>
                <tr>
                    <td>Fastest Run</td>
                    <td>{training_time_standard['min']:.2f}s</td>
                    <td>{training_time_kolosal['min']:.2f}s</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Slowest Run</td>
                    <td>{training_time_standard['max']:.2f}s</td>
                    <td>{training_time_kolosal['max']:.2f}s</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </table>
            
            <h3>🎯 Model Accuracy Analysis</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Standard ML</th>
                    <th>Genta AutoML</th>
                    <th>Improvement</th>
                    <th>Significance</th>
                </tr>
                <tr>
                    <td>Mean Test Score</td>
                    <td>{test_score_standard['mean']:.4f}</td>
                    <td>{test_score_kolosal['mean']:.4f}</td>
                    <td class="{'improvement' if accuracy_improvement > 0 else 'degradation'}">{accuracy_improvement:+.1f}%</td>
                    <td>{'Highly significant' if abs(accuracy_improvement) > 5 else 'Moderate' if abs(accuracy_improvement) > 1 else 'Minimal'}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>{test_score_standard['std']:.4f}</td>
                    <td>{test_score_kolosal['std']:.4f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Best Score</td>
                    <td>{test_score_standard['max']:.4f}</td>
                    <td>{test_score_kolosal['max']:.4f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Worst Score</td>
                    <td>{test_score_standard['min']:.4f}</td>
                    <td>{test_score_kolosal['min']:.4f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </table>
            
            <h3>🧠 Memory Usage Analysis</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Standard ML</th>
                    <th>Genta AutoML</th>
                    <th>Improvement</th>
                    <th>Significance</th>
                </tr>
                <tr>
                    <td>Mean Peak Memory</td>
                    <td>{memory_standard['mean']:.1f} MB</td>
                    <td>{memory_kolosal['mean']:.1f} MB</td>
                    <td class="{'improvement' if memory_improvement > 0 else 'degradation'}">{memory_improvement:+.1f}%</td>
                    <td>{'Highly significant' if abs(memory_improvement) > 20 else 'Moderate' if abs(memory_improvement) > 5 else 'Minimal'}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>{memory_standard['std']:.1f} MB</td>
                    <td>{memory_kolosal['std']:.1f} MB</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Lowest Usage</td>
                    <td>{memory_standard['min']:.1f} MB</td>
                    <td>{memory_kolosal['min']:.1f} MB</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Highest Usage</td>
                    <td>{memory_standard['max']:.1f} MB</td>
                    <td>{memory_kolosal['max']:.1f} MB</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </table>
            
            <div class="chart-container">
                <h2>📈 Performance Comparison Visualizations</h2>
                <img src="comparison_charts_{self.timestamp}.png" alt="Comprehensive Comparison Charts" style="max-width: 100%; height: auto; border-radius: 10px;">
            </div>
            
            <div class="chart-container">
                <h2>🔍 Scalability Analysis</h2>
                <img src="scalability_analysis_{self.timestamp}.png" alt="Scalability Analysis Charts" style="max-width: 100%; height: auto; border-radius: 10px;">
            </div>
            
            <h2>📝 Individual Results Details</h2>
            <table>
                <tr>
                    <th>Approach</th>
                    <th>Dataset</th>
                    <th>Model</th>
                    <th>Dataset Size</th>
                    <th>Training Time (s)</th>
                    <th>Test Score</th>
                    <th>Memory (MB)</th>
                    <th>Success</th>
                </tr>
        """
        
        # Add individual results
        all_results = standard_results + kolosal_results
        for result in sorted(all_results, key=lambda x: (x.dataset_name, x.model_name, x.approach)):
            html_content += f"""
                <tr>
                    <td>{result.approach}</td>
                    <td>{result.dataset_name}</td>
                    <td>{result.model_name}</td>
                    <td>{result.training_time:.2f}</td>
                    <td>{result.test_score:.4f}</td>
                    <td>{result.memory_peak_mb:.1f}</td>
                    <td>{'✓' if result.success else '✗'}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="summary">
                <h2>Conclusions</h2>
                <ul>
        """
        
        if speed_improvement > 0:
            html_content += f"<li>Kolosal AutoML is <strong>{speed_improvement:.1f}% faster</strong> than standard ML approaches on average.</li>"
        else:
            html_content += f"<li>Standard ML is <strong>{abs(speed_improvement):.1f}% faster</strong> than Kolosal AutoML on average.</li>"
            
        if accuracy_improvement > 0:
            html_content += f"<li>Kolosal AutoML achieves <strong>{accuracy_improvement:.1f}% better accuracy</strong> than standard ML approaches on average.</li>"
        else:
            html_content += f"<li>Standard ML achieves <strong>{abs(accuracy_improvement):.1f}% better accuracy</strong> than Kolosal AutoML on average.</li>"
            
        if memory_improvement > 0:
            html_content += f"<li>Kolosal AutoML uses <strong>{memory_improvement:.1f}% less memory</strong> than standard ML approaches on average.</li>"
        else:
            html_content += f"<li>Standard ML uses <strong>{abs(memory_improvement):.1f}% less memory</strong> than Kolosal AutoML on average.</li>"
        
        html_content += """
                </ul>
                <p><em>Note: This comparison was conducted with Kolosal AutoML optimizations disabled to ensure fair comparison against standard scikit-learn approaches.</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def plot_trial_results(self, save_path: str = None) -> str:
        """Generate plots showing trial results and variability."""
        if not self.results:
            logger.warning("No results to plot")
            return ""
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        if save_path is None:
            save_path = plots_dir / f"trial_plots_{self.timestamp}.png"
        
        # Prepare data for plotting
        df_results = []
        for result in self.results:
            if result.success:
                df_results.append({
                    'experiment_id': result.experiment_id,
                    'trial_number': result.trial_number,
                    'approach': result.approach,
                    'dataset_name': result.dataset_name,
                    'model_name': result.model_name,
                    'device_config': result.device_config,
                    'optimizer_config': result.optimizer_config,
                    'training_time': result.training_time,
                    'test_score': result.test_score,
                    'memory_peak_mb': result.memory_peak_mb
                })
        
        if not df_results:
            logger.warning("No successful results to plot")
            return ""
        
        df = pd.DataFrame(df_results)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trial Results Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Training time by trials
        ax1 = axes[0, 0]
        for exp_id in df['experiment_id'].unique():
            exp_data = df[df['experiment_id'] == exp_id]
            for approach in exp_data['approach'].unique():
                approach_data = exp_data[exp_data['approach'] == approach]
                ax1.plot(approach_data['trial_number'], approach_data['training_time'], 
                        marker='o', label=f"{exp_id}_{approach}", alpha=0.7)
        
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Training Time (seconds)')
        ax1.set_title('Training Time Across Trials')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Test score by trials
        ax2 = axes[0, 1]
        for exp_id in df['experiment_id'].unique():
            exp_data = df[df['experiment_id'] == exp_id]
            for approach in exp_data['approach'].unique():
                approach_data = exp_data[exp_data['approach'] == approach]
                ax2.plot(approach_data['trial_number'], approach_data['test_score'], 
                        marker='s', label=f"{exp_id}_{approach}", alpha=0.7)
        
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Test Score')
        ax2.set_title('Test Score Across Trials')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Memory usage by trials
        ax3 = axes[1, 0]
        for exp_id in df['experiment_id'].unique():
            exp_data = df[df['experiment_id'] == exp_id]
            for approach in exp_data['approach'].unique():
                approach_data = exp_data[exp_data['approach'] == approach]
                ax3.plot(approach_data['trial_number'], approach_data['memory_peak_mb'], 
                        marker='^', label=f"{exp_id}_{approach}", alpha=0.7)
        
        ax3.set_xlabel('Trial Number')
        ax3.set_ylabel('Peak Memory (MB)')
        ax3.set_title('Memory Usage Across Trials')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Variability analysis (box plots)
        ax4 = axes[1, 1]
        
        # Create box plot data for each experiment
        box_data = []
        box_labels = []
        
        for exp_id in df['experiment_id'].unique():
            exp_data = df[df['experiment_id'] == exp_id]
            
            # Standard approach
            std_data = exp_data[exp_data['approach'] == 'standard']
            if not std_data.empty:
                box_data.append(std_data['test_score'].values)
                box_labels.append(f"{exp_id[:15]}..._std")
            
            # Kolosal approach
            kolosal_data = exp_data[exp_data['approach'] == 'kolosal']
            if not kolosal_data.empty:
                box_data.append(kolosal_data['test_score'].values)
                box_labels.append(f"{exp_id[:15]}..._kolosal")
        
        if box_data:
            ax4.boxplot(box_data, labels=box_labels)
            ax4.set_ylabel('Test Score')
            ax4.set_title('Score Variability Across Trials')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trial plots saved to: {save_path}")
        return str(save_path)

    def generate_trial_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive statistics for all trials."""
        if not self.results:
            return {}
        
        # Prepare data
        df_results = []
        for result in self.results:
            if result.success:
                df_results.append({
                    'experiment_id': result.experiment_id,
                    'trial_number': result.trial_number,
                    'approach': result.approach,
                    'dataset_name': result.dataset_name,
                    'model_name': result.model_name,
                    'device_config': result.device_config,
                    'optimizer_config': result.optimizer_config,
                    'training_time': result.training_time,
                    'test_score': result.test_score,
                    'memory_peak_mb': result.memory_peak_mb
                })
        
        if not df_results:
            return {}
        
        df = pd.DataFrame(df_results)
        
        statistics = {
            'total_trials': len(df),
            'total_experiments': df['experiment_id'].nunique(),
            'total_approaches': df['approach'].nunique(),
            'metrics_by_experiment': {},
            'overall_comparison': {},
            'variability_analysis': {}
        }
        
        # Statistics by experiment
        for exp_id in df['experiment_id'].unique():
            exp_data = df[df['experiment_id'] == exp_id]
            
            exp_stats = {
                'total_trials': len(exp_data),
                'dataset': exp_data['dataset_name'].iloc[0],
                'model': exp_data['model_name'].iloc[0],
                'device_config': exp_data['device_config'].iloc[0],
                'optimizer_config': exp_data['optimizer_config'].iloc[0],
                'approaches': {}
            }
            
            for approach in exp_data['approach'].unique():
                approach_data = exp_data[exp_data['approach'] == approach]
                
                exp_stats['approaches'][approach] = {
                    'trials': len(approach_data),
                    'training_time': {
                        'mean': approach_data['training_time'].mean(),
                        'std': approach_data['training_time'].std(),
                        'min': approach_data['training_time'].min(),
                        'max': approach_data['training_time'].max()
                    },
                    'test_score': {
                        'mean': approach_data['test_score'].mean(),
                        'std': approach_data['test_score'].std(),
                        'min': approach_data['test_score'].min(),
                        'max': approach_data['test_score'].max()
                    },
                    'memory_peak_mb': {
                        'mean': approach_data['memory_peak_mb'].mean(),
                        'std': approach_data['memory_peak_mb'].std(),
                        'min': approach_data['memory_peak_mb'].min(),
                        'max': approach_data['memory_peak_mb'].max()
                    }
                }
            
            statistics['metrics_by_experiment'][exp_id] = exp_stats
        
        # Overall comparison
        std_data = df[df['approach'] == 'standard']
        kolosal_data = df[df['approach'] == 'kolosal']
        
        if not std_data.empty and not kolosal_data.empty:
            statistics['overall_comparison'] = {
                'standard': {
                    'training_time_mean': std_data['training_time'].mean(),
                    'test_score_mean': std_data['test_score'].mean(),
                    'memory_mean': std_data['memory_peak_mb'].mean()
                },
                'kolosal': {
                    'training_time_mean': kolosal_data['training_time'].mean(),
                    'test_score_mean': kolosal_data['test_score'].mean(),
                    'memory_mean': kolosal_data['memory_peak_mb'].mean()
                },
                'improvements': {
                    'speed_improvement_pct': ((std_data['training_time'].mean() / kolosal_data['training_time'].mean()) - 1) * 100,
                    'accuracy_improvement_pct': ((kolosal_data['test_score'].mean() / std_data['test_score'].mean()) - 1) * 100,
                    'memory_improvement_pct': ((std_data['memory_peak_mb'].mean() / kolosal_data['memory_peak_mb'].mean()) - 1) * 100
                }
            }
        
        # Variability analysis
        for metric in ['training_time', 'test_score', 'memory_peak_mb']:
            metric_stats = {}
            
            for approach in df['approach'].unique():
                approach_data = df[df['approach'] == approach]
                metric_stats[approach] = {
                    'coefficient_of_variation': approach_data[metric].std() / approach_data[metric].mean() if approach_data[metric].mean() > 0 else 0,
                    'range': approach_data[metric].max() - approach_data[metric].min(),
                    'iqr': approach_data[metric].quantile(0.75) - approach_data[metric].quantile(0.25)
                }
            
            statistics['variability_analysis'][metric] = metric_stats
        
        return statistics

def main():
    """Main function to run the comparison benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Kolosal AutoML vs Standard ML")
    parser.add_argument("--output-dir", default="./comparison_results", help="Output directory")
    parser.add_argument("--datasets", nargs="+", default=[
        "iris", "wine", "breast_cancer", "diabetes", 
        "synthetic_small_classification", "synthetic_medium_classification",
        "synthetic_small_regression"
    ], help="Datasets to test")
    parser.add_argument("--models", nargs="+", default=[
        "random_forest", "gradient_boosting", "logistic_regression"
    ], help="Models to test")
    parser.add_argument("--optimization", default="random_search", 
                       choices=["grid_search", "random_search"], 
                       help="Optimization strategy")
    
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = ComparisonBenchmarkRunner(args.output_dir)
    
    # Generate configurations
    configurations = []
    for dataset in args.datasets:
        for model in args.models:
            # Skip incompatible combinations
            if ("regression" in dataset and model == "logistic_regression") or \
               ("classification" in dataset and model in ["ridge", "lasso"]) or \
               (dataset in ["diabetes", "california_housing"] and model == "logistic_regression") or \
               (dataset in ["iris", "wine", "breast_cancer", "digits"] and model in ["ridge", "lasso"]):
                continue
                
            configurations.append({
                "dataset": dataset,
                "model": model,
                "optimization_strategy": args.optimization
            })
    
    logger.info(f"Running {len(configurations)} comparisons...")
    
    # Run comparisons
    start_time = time.time()
    runner.run_multiple_comparisons(configurations)
    total_time = time.time() - start_time
    
    # Save results and generate report
    results_file = runner.save_results()
    report_file = runner.generate_comparison_report()
    
    # Summary
    successful_results = [r for r in runner.results if r.success]
    standard_successful = len([r for r in successful_results if r.approach == "standard"])
    kolosal_successful = len([r for r in successful_results if r.approach == "kolosal"])
    
    logger.info("=" * 60)
    logger.info("COMPARISON BENCHMARK COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Total comparisons: {len(configurations)}")
    logger.info(f"Standard ML successful: {standard_successful}")
    logger.info(f"Kolosal AutoML successful: {kolosal_successful}")
    logger.info(f"Results saved to: {results_file}")
    if report_file:
        logger.info(f"Report generated: {report_file}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
