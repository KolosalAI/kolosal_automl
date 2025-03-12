# -----------------------------------------------------------------------------
# Parallel Benchmarking Utilities
# -----------------------------------------------------------------------------

def run_parallel_benchmarks(benchmark_functions, max_workers=None, benchmark_dir=None, logger=None):
    """
    Run multiple benchmarks in parallel using a thread pool.
    
    Parameters:
    -----------
    benchmark_functions : list of callable
        List of benchmark functions to run
    max_workers : int, optional
        Maximum number of parallel workers (default: number of CPU cores)
    benchmark_dir : str
        Directory to save benchmark results
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    dict
        Dictionary with benchmark results
    """
    if not benchmark_functions:
        logger.warning("No benchmark functions provided")
        return {}
    
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    logger.info(f"Running {len(benchmark_functions)} benchmarks in parallel with {max_workers} workers")
    
    # Import concurrent.futures if available
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit benchmark jobs
            future_to_benchmark = {
                executor.submit(func): f"benchmark_{i}" 
                for i, func in enumerate(benchmark_functions)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_benchmark):
                benchmark_name = future_to_benchmark[future]
                try:
                    result = future.result()
                    results[benchmark_name] = result
                    logger.info(f"Completed benchmark: {benchmark_name}")
                except Exception as e:
                    logger.error(f"Benchmark {benchmark_name} failed: {e}")
                    results[benchmark_name] = {"error": str(e)}
        
        return results
    
    except ImportError:
        # Fallback to sequential execution if concurrent.futures is not available
        logger.warning("concurrent.futures not available, running benchmarks sequentially")
        results = {}
        for i, func in enumerate(benchmark_functions):
            benchmark_name = f"benchmark_{i}"
            try:
                results[benchmark_name] = func()
                logger.info(f"Completed benchmark: {benchmark_name}")
            except Exception as e:
                logger.error(f"Benchmark {benchmark_name} failed: {e}")
                results[benchmark_name] = {"error": str(e)}
        
        return results

def create_benchmark_schedule(config, benchmark_dir, logger):
    """
    Create a schedule of benchmark functions to run based on configuration.
    
    Parameters:
    -----------
    config : dict
        Benchmark configuration dictionary
    benchmark_dir : str
        Directory to save benchmark results
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    list
        List of benchmark functions to run
    """
    benchmark_functions = []
    
    # Training speed benchmarks
    if config.get("test_training_speed", True):
        for dataset_name in config.get("training_speed", {}).get("datasets", []):
            for strategy in config.get("training_speed", {}).get("optimization_strategies", []):
                # Convert strategy string to OptimizationStrategy
                strategy_obj = getattr(OptimizationStrategy, strategy)
                
                # Add benchmark function for this dataset and strategy
                benchmark_functions.append(
                    lambda d=dataset_name, s=strategy_obj: benchmark_training_speed(
                        d, s, benchmark_dir=benchmark_dir, logger=logger
                    )
                )
    
    # Feature selection benchmarks
    if config.get("test_feature_selection", True):
        for dataset_name in config.get("feature_selection", {}).get("datasets", []):
            benchmark_functions.append(
                lambda d=dataset_name: benchmark_feature_selection(
                    d, benchmark_dir=benchmark_dir, logger=logger
                )
            )
    
    # Batch processing benchmarks
    if config.get("test_batch_processing", True):
        batch_sizes = config.get("batch_processing", {}).get("batch_sizes", [10, 50, 100, 500])
        benchmark_functions.append(
            lambda: benchmark_batch_processing(
                batch_sizes, benchmark_dir=benchmark_dir, logger=logger
            )
        )
    
    # Model comparison benchmarks
    if config.get("test_model_comparison", True):
        benchmark_functions.append(
            lambda: benchmark_model_comparison(
                benchmark_dir=benchmark_dir, logger=logger
            )
        )
    
    # Standard benchmarks
    if config.get("test_standard_benchmarks", True):
        # PLMB benchmarks
        if config.get("standard_benchmarks", {}).get("plmb", {}).get("enabled", True):
            benchmark_functions.append(
                lambda: run_plmb_benchmarks(
                    benchmark_dir, logger, config.get("standard_benchmarks", {}).get("plmb")
                )
            )
        
        # UCI benchmarks
        if config.get("standard_benchmarks", {}).get("uci", {}).get("enabled", True):
            benchmark_functions.append(
                lambda: run_uci_benchmarks(
                    benchmark_dir, logger, config.get("standard_benchmarks", {}).get("uci")
                )
            )
        
        # OpenML benchmarks
        if config.get("standard_benchmarks", {}).get("openml", {}).get("enabled", True):
            benchmark_functions.append(
                lambda: run_openml_benchmarks(
                    benchmark_dir, logger, config.get("standard_benchmarks", {}).get("openml")
                )
            )
        
        # AutoML benchmarks
        if config.get("standard_benchmarks", {}).get("automl_benchmark", {}).get("enabled", True):
            benchmark_functions.append(
                lambda: run_automl_benchmarks(
                    benchmark_dir, logger, config.get("standard_benchmarks", {}).get("automl_benchmark")
                )
            )
    
    return benchmark_functions#!/usr/bin/env python
"""
Enhanced ML Training Engine Benchmark Script

This script performs comprehensive benchmarking of the ML Training Engine,
testing speed, efficiency, and functionality using standardized ML benchmarks.

Usage:
    python benchmark.py [--output_dir OUTPUT_DIR] [--test_selection TESTS] [--benchmark_suite SUITE]

Example:
    python benchmark.py --output_dir ./benchmark_results --test_selection all
    python benchmark.py --test_selection training,batch --benchmark_suite mlperf
"""

import os
import time
import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import logging
import traceback
import platform
import sys
import pickle
import requests
import tempfile
import zipfile
import io
import re
from pathlib import Path
from memory_profiler import memory_usage
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
from tqdm import tqdm

# Import scikit-learn components for testing
from sklearn.datasets import (
    load_iris, load_breast_cancer, fetch_california_housing, 
    load_digits, load_wine, fetch_openml, load_diabetes
)
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGDClassifier
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default configuration values
DEFAULT_CONFIG = {
    # Basic configuration
    "test_training_speed": True,
    "test_feature_selection": True,
    "test_batch_processing": True,
    "test_model_comparison": True,
    "test_standard_benchmarks": True,
    "save_models": True,
    "generate_plots": True,
    "verbose_output": True,
    
    # Training speed benchmark configuration
    "training_speed": {
        "datasets": [
            "iris",
            "breast_cancer",
            "wine",
            "digits",
            "california_housing",
            "synthetic_classification",
            "synthetic_regression"
        ],
        "optimization_strategies": [
            "RANDOM_SEARCH",
            "GRID_SEARCH",
            "BAYESIAN_OPTIMIZATION"
        ],
        "cv_folds": 3,
        "optimization_iterations": 10
    },
    
    # Feature selection benchmark configuration
    "feature_selection": {
        "datasets": [
            "breast_cancer",
            "synthetic_classification",
            "covtype",
            "mnist"
        ],
        "methods": [
            "mutual_info",
            "recursive_elimination",
            "lasso",
            "tree_importance"
        ],
        "test_with_noise_features": True,
        "noise_feature_count": 20
    },
    
    # Batch processing benchmark configuration
    "batch_processing": {
        "batch_sizes": [10, 20, 50, 100, 200, 500, 1000],
        "test_dataset_size": 10000,
        "feature_count": 20,
        "enable_adaptive_batching": True,
        "test_multiple_threads": True,
        "thread_counts": [1, 2, 4, 8, 16]
    },
    
    # Model comparison benchmark configuration
    "model_comparison": {
        "datasets": [
            "breast_cancer",
            "wine",
            "digits",
            "covtype",
            "synthetic_classification"
        ],
        "models": {
            "random_forest": {
                "class": "RandomForestClassifier",
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "gradient_boosting": {
                "class": "GradientBoostingClassifier",
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10],
                    "learning_rate": [0.01, 0.05, 0.1]
                }
            },
            "svm": {
                "class": "SVC",
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"],
                    "probability": [True]
                }
            },
            "logistic_regression": {
                "class": "LogisticRegression",
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"]
                }
            },
            "knn": {
                "class": "KNeighborsClassifier",
                "params": {
                    "n_neighbors": [3, 5, 10],
                    "weights": ["uniform", "distance"]
                }
            }
        }
    },
    
    # Standard benchmarks configuration
    "standard_benchmarks": {
        # PLMB benchmark settings
        "plmb": {
            "enabled": True,
            "datasets": ["news20", "rcv1", "mnist", "cifar10"],
            "iterations": 3,
            "timeout": 1800  # 30 minutes per benchmark
        },
        
        # UCI repository benchmarks
        "uci": {
            "enabled": True,
            "datasets": [
                "adult", "wine-quality", "bank-marketing", 
                "car-evaluation", "forest-fires", "covertype"
            ],
            "iterations": 3,
            "timeout": 1200  # 20 minutes per benchmark
        },
        
        # OpenML benchmarks
        "openml": {
            "enabled": True,
            "dataset_ids": [
                # Classification datasets
                42178,  # Mice Protein Expression (classification)
                42821,  # higgs (binary classification)
                42731,  # amazon_employee (binary classification)
                42705,  # Australian (binary classification)
                # Regression datasets
                42225,  # diamonds (regression)
                42093,  # house_prices (regression)
                42565,  # medical_charges (regression)
            ],
            "iterations": 2,
            "timeout": 1800  # 30 minutes per benchmark
        },
        
        # Tabular benchmarks for AutoML
        "automl_benchmark": {
            "enabled": True,
            "datasets": ["adult", "credit-g", "nomao", "vehicle"],
            "iterations": 1,
            "timeout": 3600  # 60 minutes per benchmark
        }
    },
    
    # System resource limits (to prevent excessive resource usage)
    "resource_limits": {
        "max_memory_percent": 85,
        "max_cpu_percent": 90,
        "timeout_seconds": 7200  # 2 hours max per benchmark
    }
}

# -----------------------------------------------------------------------------
# Import ML Training Engine components
# -----------------------------------------------------------------------------

# Mock classes to represent the ML Training Engine components
# These should be replaced with actual imports in a real implementation

class TaskType:
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"

class OptimizationStrategy:
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"

class NormalizationType:
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    CUSTOM = "custom"

class PreprocessorConfig:
    def __init__(self, normalization=NormalizationType.STANDARD, handle_nan=True, 
                handle_inf=True, detect_outliers=False, parallel_processing=True, 
                cache_enabled=True):
        self.normalization = normalization
        self.handle_nan = handle_nan
        self.handle_inf = handle_inf
        self.detect_outliers = detect_outliers
        self.parallel_processing = parallel_processing
        self.cache_enabled = cache_enabled

class BatchProcessorConfig:
    def __init__(self, min_batch_size=1, max_batch_size=64, initial_batch_size=16,
                enable_adaptive_batching=True, enable_monitoring=True, 
                enable_memory_optimization=True):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.initial_batch_size = initial_batch_size
        self.enable_adaptive_batching = enable_adaptive_batching
        self.enable_monitoring = enable_monitoring
        self.enable_memory_optimization = enable_memory_optimization

class InferenceEngineConfig:
    def __init__(self):
        pass

class MLTrainingEngineConfig:
    def __init__(self, task_type=TaskType.CLASSIFICATION, optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
                feature_selection=False, feature_selection_method=None, feature_selection_k=None,
                cv_folds=5, optimization_iterations=50, model_path="./models", experiment_tracking=True,
                preprocessing_config=None, batch_processing_config=None, log_level="INFO"):
        self.task_type = task_type
        self.optimization_strategy = optimization_strategy
        self.feature_selection = feature_selection
        self.feature_selection_method = feature_selection_method
        self.feature_selection_k = feature_selection_k
        self.cv_folds = cv_folds
        self.optimization_iterations = optimization_iterations
        self.model_path = model_path
        self.experiment_tracking = experiment_tracking
        self.preprocessing_config = preprocessing_config or PreprocessorConfig()
        self.batch_processing_config = batch_processing_config or BatchProcessorConfig()
        self.log_level = log_level

class MLTrainingEngine:
    """Mock ML Training Engine class for benchmarking purposes."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_model = None
        self.tracker = None
        
    def train_model(self, model, model_name, param_grid, X, y, X_test=None, y_test=None):
        """Simulated model training function."""
        # This is a mock implementation - replace with actual implementation
        time.sleep(0.1 * len(X) / 1000)  # Simulate training time
        
        # Store the model
        self.models[model_name] = model
        
        # Simulate fitting
        model.fit(X, y)
        
        # Simulate metrics calculation
        if hasattr(model, "predict_proba") and len(np.unique(y)) <= 10:
            y_pred = model.predict(X)
            accuracy = np.mean(y_pred == y)
            metrics = {"accuracy": accuracy}
        else:
            y_pred = model.predict(X)
            mse = np.mean((y_pred - y) ** 2)
            metrics = {"mse": mse, "r2_score": 1 - mse / np.var(y)}
        
        self.best_model = model
        return model, metrics
    
    def predict(self, X, model_name=None):
        """Make predictions using the specified or best model."""
        model = self.models.get(model_name, self.best_model)
        if model is None:
            raise ValueError("No model available for prediction")
        return model.predict(X)
    
    def run_batch_inference(self, data_generator, batch_size=None, model_name=None):
        """Run inference in batches."""
        results = []
        for batch in data_generator:
            batch_pred = self.predict(batch, model_name)
            results.append(batch_pred)
        return results
    
    def save_model(self, model_name=None, filepath=None):
        """Save the model to disk."""
        # Simulated save operation
        return True
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models on test data."""
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba") and len(np.unique(y_test)) <= 10:
                accuracy = np.mean(y_pred == y_test)
                results[name] = {"accuracy": accuracy}
            else:
                mse = np.mean((y_pred - y_test) ** 2)
                results[name] = {"mse": mse, "r2_score": 1 - mse / np.var(y_test)}
        return results
    
    def generate_report(self, output_file=None):
        """Generate performance report."""
        # Simulated report generation
        if output_file:
            with open(output_file, 'w') as f:
                f.write("ML Training Engine Performance Report\n")
        return True
    
    def shutdown(self):
        """Release resources."""
        self.models = {}
        self.best_model = None
        gc.collect()

# -----------------------------------------------------------------------------
# Benchmarking Functions
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Dataset Utilities
# -----------------------------------------------------------------------------

def load_dataset(dataset_name, logger, sample_limit=None):
    """
    Load a dataset for benchmarking.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load
    logger : logging.Logger
        Logger instance
    sample_limit : int, optional
        If set, limit the dataset to this many samples
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    task_type : str
        Type of task (classification or regression)
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Standard scikit-learn datasets
    if dataset_name == "iris":
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        task_type = TaskType.CLASSIFICATION
    
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        task_type = TaskType.CLASSIFICATION
    
    elif dataset_name == "wine":
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        task_type = TaskType.CLASSIFICATION
    
    elif dataset_name == "digits":
        data = load_digits()
        X = pd.DataFrame(data.data)
        y = pd.Series(data.target)
        task_type = TaskType.CLASSIFICATION
    
    elif dataset_name == "california_housing":
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        task_type = TaskType.REGRESSION
    
    elif dataset_name == "diabetes":
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        task_type = TaskType.REGRESSION
    
    # OpenML datasets
    elif dataset_name == "mnist":
        logger.info("Fetching MNIST dataset from OpenML (this may take a while)...")
        try:
            data = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')
            X = data.data
            y = LabelEncoder().fit_transform(data.target)
            task_type = TaskType.CLASSIFICATION
        except Exception as e:
            logger.error(f"Error loading MNIST: {e}")
            # Fallback to digits if MNIST fails
            logger.info("Falling back to digits dataset")
            data = load_digits()
            X = pd.DataFrame(data.data)
            y = pd.Series(data.target)
            task_type = TaskType.CLASSIFICATION
    
    elif dataset_name == "covertype" or dataset_name == "covtype":
        logger.info("Fetching Covertype dataset from OpenML (this may take a while)...")
        try:
            data = fetch_openml('covertype', version=1, as_frame=True, parser='auto')
            X = data.data
            y = LabelEncoder().fit_transform(data.target)
            task_type = TaskType.CLASSIFICATION
        except Exception as e:
            logger.error(f"Error loading covertype: {e}")
            # Create a synthetic alternative if covertype fails
            logger.info("Creating synthetic alternative to covertype")
            X, y, task_type = create_synthetic_dataset(n_samples=10000, n_features=50, classification=True)
    
    elif dataset_name.startswith("openml_"):
        # Extract dataset ID from name (e.g., "openml_42178" -> 42178)
        try:
            dataset_id = int(dataset_name.split("_")[1])
            logger.info(f"Fetching OpenML dataset ID {dataset_id}...")
            data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
            X = data.data
            if data.target.dtype == 'object':
                y = LabelEncoder().fit_transform(data.target)
                task_type = TaskType.CLASSIFICATION
            else:
                y = pd.Series(data.target)
                # Determine if classification or regression
                if len(np.unique(y)) < 20:  # Heuristic for classification
                    task_type = TaskType.CLASSIFICATION
                else:
                    task_type = TaskType.REGRESSION
        except Exception as e:
            logger.error(f"Error loading OpenML dataset {dataset_name}: {e}")
            # Fall back to a synthetic dataset
            logger.info("Falling back to synthetic dataset")
            X, y, task_type = create_synthetic_dataset(n_samples=5000, n_features=20, classification=True)
    
    # UCI datasets - we'll download these from a source
    elif dataset_name in ["adult", "wine-quality", "bank-marketing", "car-evaluation", 
                         "forest-fires", "abalone", "housing"]:
        try:
            X, y, task_type = download_uci_dataset(dataset_name, logger)
        except Exception as e:
            logger.error(f"Error downloading UCI dataset {dataset_name}: {e}")
            # Fall back to a synthetic dataset
            logger.info("Falling back to synthetic dataset")
            X, y, task_type = create_synthetic_dataset(n_samples=5000, n_features=20, classification=True)
    
    # PLMB benchmark datasets
    elif dataset_name in ["news20", "rcv1", "cifar10"]:
        try:
            X, y, task_type = download_plmb_dataset(dataset_name, logger)
        except Exception as e:
            logger.error(f"Error downloading PLMB dataset {dataset_name}: {e}")
            # Fall back to a synthetic dataset
            logger.info("Falling back to synthetic dataset")
            X, y, task_type = create_synthetic_dataset(n_samples=5000, n_features=20, classification=True)
    
    # AutoML benchmark datasets
    elif dataset_name in ["credit-g", "nomao", "vehicle", "electricity"]:
        try:
            X, y, task_type = download_automl_dataset(dataset_name, logger)
        except Exception as e:
            logger.error(f"Error downloading AutoML benchmark dataset {dataset_name}: {e}")
            # Fall back to a synthetic dataset
            logger.info("Falling back to synthetic dataset")
            X, y, task_type = create_synthetic_dataset(n_samples=5000, n_features=20, classification=True)
    
    # Synthetic datasets
    elif dataset_name == "synthetic_classification":
        X, y, task_type = create_synthetic_dataset(n_samples=10000, n_features=30, classification=True)
    
    elif dataset_name == "synthetic_regression":
        X, y, task_type = create_synthetic_dataset(n_samples=10000, n_features=30, classification=False)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit samples if specified
    if sample_limit is not None and len(X) > sample_limit:
        logger.info(f"Limiting dataset to {sample_limit} samples (original size: {len(X)})")
        indices = np.random.choice(len(X), sample_limit, replace=False)
        X = X.iloc[indices].reset_index(drop=True)
        y = y.iloc[indices].reset_index(drop=True) if isinstance(y, pd.Series) else pd.Series(y[indices])
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, task_type

def download_uci_dataset(dataset_name, logger):
    """
    Download and prepare datasets from the UCI repository.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the UCI dataset to download
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    task_type : str
        Type of task (classification or regression)
    """
    logger.info(f"Downloading UCI dataset: {dataset_name}")
    
    # Dictionary mapping dataset names to their UCI URLs and processing functions
    uci_datasets = {
        "adult": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "names_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
            "task_type": TaskType.CLASSIFICATION,
            "columns": ["age", "workclass", "fnlwgt", "education", "education-num", 
                      "marital-status", "occupation", "relationship", "race", "sex", 
                      "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        },
        "wine-quality": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "task_type": TaskType.REGRESSION,
            "separator": ";",
        },
        "bank-marketing": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
            "file_in_zip": "bank-additional/bank-additional-full.csv",
            "task_type": TaskType.CLASSIFICATION,
            "separator": ";",
        },
        "housing": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            "task_type": TaskType.REGRESSION,
            "columns": ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
        }
    }
    
    if dataset_name not in uci_datasets:
        raise ValueError(f"UCI dataset {dataset_name} not configured for download")
    
    dataset_info = uci_datasets[dataset_name]
    task_type = dataset_info["task_type"]
    
    try:
        # Handle direct CSV files
        if dataset_name in ["adult", "wine-quality", "housing"]:
            separator = dataset_info.get("separator", ",")
            
            if dataset_name == "housing":
                # Fixed-width format for housing
                df = pd.read_fwf(dataset_info["url"], header=None, names=dataset_info["columns"])
            else:
                # Regular CSV format
                df = pd.read_csv(dataset_info["url"], header=None if dataset_name == "adult" else 0, 
                                 sep=separator, names=dataset_info.get("columns", None))
            
            # For adult dataset, separate target from features
            if dataset_name == "adult":
                y = df["income"].map({" >50K": 1, " <=50K": 0})
                X = df.drop("income", axis=1)
            elif dataset_name == "housing":
                y = df["MEDV"]  # Target is median value
                X = df.drop("MEDV", axis=1)
            else:
                # For wine-quality, last column is the target
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]
        
        # Handle zipped files
        elif dataset_name == "bank-marketing":
            # Download and extract zip file
            r = requests.get(dataset_info["url"])
            with zipfile.ZipFile(io.BytesIO(r.content)) as zip_ref:
                with zip_ref.open(dataset_info["file_in_zip"]) as f:
                    df = pd.read_csv(f, sep=dataset_info["separator"])
            
            # For bank marketing, 'y' column is the target
            y = df["y"].map({"yes": 1, "no": 0})
            X = df.drop("y", axis=1)
        
        # Process categorical features for all datasets
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Convert to pandas DataFrame and Series
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        logger.info(f"UCI dataset {dataset_name} loaded successfully")
        return X, y, task_type
    
    except Exception as e:
        logger.error(f"Error processing UCI dataset {dataset_name}: {e}")
        raise

def download_plmb_dataset(dataset_name, logger):
    """
    Download and prepare datasets from Python Language Model Benchmark (PLMB).
    
    Parameters:
    -----------
    dataset_name : str
        Name of the PLMB dataset to download
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    task_type : str
        Type of task (classification or regression)
    """
    logger.info(f"Preparing PLMB dataset: {dataset_name}")
    
    # For demonstration purposes, we'll create synthetic datasets that mimic 
    # the structure of the real PLMB datasets
    task_type = TaskType.CLASSIFICATION  # All these are classification tasks
    
    if dataset_name == "news20":
        # Create a synthetic dataset similar to news20 (sparse text features)
        n_samples = 2000
        n_features = 500
        X = np.random.randn(n_samples, n_features) * 0.1
        X[X < 0.1] = 0  # Make it sparse
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        y = pd.Series(np.random.randint(0, 20, size=n_samples))  # 20 classes
        
    elif dataset_name == "rcv1":
        # Create a synthetic dataset similar to RCV1 (sparse text features, multi-label)
        n_samples = 2000
        n_features = 500
        X = np.random.randn(n_samples, n_features) * 0.1
        X[X < 0.05] = 0  # Make it sparse
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        y = pd.Series(np.random.randint(0, 2, size=n_samples))  # Binary for simplicity
        
    elif dataset_name == "cifar10":
        # Try to use actual CIFAR-10, but flatten the images
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml('CIFAR-10', as_frame=True)
            X = data.data
            y = LabelEncoder().fit_transform(data.target)
        except:
            # Create a synthetic dataset similar to CIFAR10 (flattened images)
            n_samples = 1000
            n_features = 3 * 32 * 32  # RGB image flattened
            X = np.random.rand(n_samples, n_features)
            X = pd.DataFrame(X, columns=[f"pixel_{i}" for i in range(n_features)])
            y = pd.Series(np.random.randint(0, 10, size=n_samples))  # 10 classes
    
    else:
        raise ValueError(f"Unknown PLMB dataset: {dataset_name}")
    
    logger.info(f"PLMB dataset {dataset_name} prepared: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, task_type

def download_automl_dataset(dataset_name, logger):
    """
    Download and prepare datasets from AutoML Benchmark.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the AutoML benchmark dataset to download
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    task_type : str
        Type of task (classification or regression)
    """
    logger.info(f"Preparing AutoML benchmark dataset: {dataset_name}")
    
    # We'll get these datasets from OpenML by name or ID when possible
    openml_mapping = {
        "credit-g": 31,        # German credit dataset
        "vehicle": 54,         # Vehicle silhouettes 
        "electricity": 151,    # Electricity price prediction
        "nomao": 1486          # Nomao dataset
    }
    
    try:
        if dataset_name in openml_mapping:
            dataset_id = openml_mapping[dataset_name]
            logger.info(f"Fetching {dataset_name} (ID: {dataset_id}) from OpenML...")
            data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
            X = data.data
            
            # Determine task type and process target
            if data.target.dtype == 'object':
                y = LabelEncoder().fit_transform(data.target)
                task_type = TaskType.CLASSIFICATION
            else:
                y = pd.Series(data.target)
                if len(np.unique(y)) < 20:  # Heuristic
                    task_type = TaskType.CLASSIFICATION
                else:
                    task_type = TaskType.REGRESSION
            
            # Process categorical features
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype('category').cat.codes
        else:
            # Fallback to synthetic data if dataset isn't in our mapping
            logger.warning(f"AutoML dataset {dataset_name} not found in OpenML mapping, creating synthetic data")
            X, y, task_type = create_synthetic_dataset(n_samples=2000, n_features=40, classification=True)
    
    except Exception as e:
        logger.error(f"Error fetching AutoML benchmark dataset {dataset_name}: {e}")
        # Fallback to synthetic data
        logger.warning(f"Creating synthetic data as fallback for {dataset_name}")
        X, y, task_type = create_synthetic_dataset(n_samples=2000, n_features=40, classification=True)
    
    logger.info(f"AutoML benchmark dataset {dataset_name} prepared: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, task_type

def create_synthetic_dataset(n_samples=10000, n_features=50, classification=True, n_informative=None, n_redundant=None, n_classes=None, noise=0.1, random_state=42):
    """
    Create a synthetic dataset for benchmarking.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Total number of features
    classification : bool
        If True, create a classification dataset, otherwise regression
    n_informative : int, optional
        Number of informative features (default: n_features // 5)
    n_redundant : int, optional
        Number of redundant features (default: n_features // 10)
    n_classes : int, optional
        Number of classes for classification (default: 2)
    noise : float
        Noise level to add to the data
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    task_type : str
        Type of task (classification or regression)
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Set defaults for optional parameters
    if n_informative is None:
        n_informative = max(1, n_features // 5)
    if n_redundant is None:
        n_redundant = max(0, n_features // 10)
    if n_classes is None:
        n_classes = 2 if classification else None
    
    # Generate data based on task type
    if classification:
        # Try to use scikit-learn's make_classification if available
        try:
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=n_samples, 
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_classes=n_classes,
                n_clusters_per_class=max(1, n_classes // 2),
                random_state=random_state,
                shuffle=True
            )
            task_type = TaskType.CLASSIFICATION
        except ImportError:
            # Fallback to manual implementation if sklearn is not available
            X = np.random.randn(n_samples, n_features)
            
            # Create informative features
            w = np.random.randn(n_informative)
            informative_part = np.dot(X[:, :n_informative], w)
            
            # Generate classes based on thresholds
            if n_classes == 2:
                y = (informative_part + np.random.randn(n_samples) * noise > 0).astype(int)
            else:
                # For multi-class, divide the range into n_classes segments
                informative_part_scaled = (informative_part - np.min(informative_part)) / (np.max(informative_part) - np.min(informative_part))
                y = (informative_part_scaled * n_classes).astype(int)
                y = np.clip(y, 0, n_classes - 1)  # Ensure classes are in the valid range
            
            task_type = TaskType.CLASSIFICATION
    else:
        # Try to use scikit-learn's make_regression if available
        try:
            from sklearn.datasets import make_regression
            X, y = make_regression(
                n_samples=n_samples, 
                n_features=n_features,
                n_informative=n_informative,
                noise=noise,
                random_state=random_state
            )
            task_type = TaskType.REGRESSION
        except ImportError:
            # Fallback to manual implementation if sklearn is not available
            X = np.random.randn(n_samples, n_features)
            
            # Create target using informative features
            w = np.random.randn(n_informative)
            y = np.dot(X[:, :n_informative], w)
            
            # Add noise
            y += np.random.randn(n_samples) * noise
            
            task_type = TaskType.REGRESSION
    
    # Convert to pandas DataFrame and Series
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y)
    
    return X, y, task_type

def benchmark_training_speed(dataset_name, optimization_strategy, feature_selection=False, 
                            benchmark_dir=None, logger=None, timeout=None, sample_limit=None):
    """
    Benchmark the training speed of the ML Training Engine.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to benchmark
    optimization_strategy : str
        Optimization strategy to use (e.g., "RANDOM_SEARCH", "GRID_SEARCH")
    feature_selection : bool, optional
        Whether to enable feature selection
    benchmark_dir : str, optional
        Directory to save benchmark results
    logger : logging.Logger, optional
        Logger instance
    timeout : int, optional
        Maximum time in seconds for the benchmark to run
    sample_limit : int, optional
        Maximum number of samples to use from the dataset
        
    Returns:
    --------
    dict
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking training speed on {dataset_name} with {optimization_strategy}")
    
    # Load dataset
    try:
        if dataset_name == "synthetic_classification":
            X, y, task_type = create_synthetic_dataset(n_samples=10000, n_features=30, classification=True)
        elif dataset_name == "synthetic_regression":
            X, y, task_type = create_synthetic_dataset(n_samples=10000, n_features=30, classification=False)
        else:
            X, y, task_type = load_dataset(dataset_name, logger, sample_limit=sample_limit)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create preprocessing config
        preproc_config = PreprocessorConfig(
            normalization=NormalizationType.STANDARD,
            handle_nan=True,
            handle_inf=True,
            detect_outliers=False,
            parallel_processing=True,
            cache_enabled=True
        )
        
        # Create configuration
        config = MLTrainingEngineConfig(
            task_type=task_type,
            optimization_strategy=optimization_strategy,
            feature_selection=feature_selection,
            feature_selection_method="mutual_info" if feature_selection else None,
            cv_folds=3,
            optimization_iterations=10,
            model_path=os.path.join(benchmark_dir, "models"),
            experiment_tracking=True,
            preprocessing_config=preproc_config,
            log_level="INFO"
        )
        
        # Initialize engine
        engine = MLTrainingEngine(config)
        
        # Define model and params
        if task_type == TaskType.CLASSIFICATION:
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        else:
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        
        # Setup progress tracking with tqdm if available
        try:
            from tqdm import tqdm
            with tqdm(total=100, desc=f"Training on {dataset_name}", unit="%") as pbar:
                # Create a callback to update the progress bar
                def progress_callback(progress):
                    pbar.update(int(progress * 100) - pbar.n)
                
                # Store the callback for later use
                engine.set_progress_callback = progress_callback
        except ImportError:
            # If tqdm is not available, just log progress
            logger.info(f"Starting training on {dataset_name}...")
        
        # Measure training time and memory usage
        start_time = time.time()
        peak_memory_before = memory_usage(-1, interval=0.1, timeout=1)[0]
        
        try:
            # Set a timeout if specified
            if timeout:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Training timed out after {timeout} seconds")
                
                # Set the timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            # Train the model
            best_model, metrics = engine.train_model(
                model=model,
                model_name=f"{dataset_name}_{optimization_strategy}",
                param_grid=param_grid,
                X=X_train,
                y=y_train,
                X_test=X_test,
                y_test=y_test
            )
            
            # Clear the timeout if it was set
            if timeout:
                signal.alarm(0)
            
            # Try to save the model
            try:
                engine.save_model()
            except Exception as e:
                logger.warning(f"Could not save model: {e}")
        
        except TimeoutError as e:
            logger.warning(f"Training timeout: {e}")
            metrics = {"error": "timeout", "error_details": str(e)}
            best_model = None
        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            metrics = {"error": str(e)}
            best_model = None
        
        end_time = time.time()
        peak_memory_after = memory_usage(-1, interval=0.1, timeout=1)[0]
        memory_used = peak_memory_after - peak_memory_before
        
        # Generate results
        results = {
            "dataset": dataset_name,
            "optimization_strategy": str(optimization_strategy),
            "feature_selection": feature_selection,
            "training_time": end_time - start_time,
            "memory_used_mb": memory_used if memory_used > 0 else None,
            "metrics": metrics,
            "dataset_size": len(X),
            "feature_count": X.shape[1],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        os.makedirs(os.path.join(benchmark_dir, "training_speed"), exist_ok=True)
        result_file = os.path.join(benchmark_dir, "training_speed", 
                                  f"{dataset_name}_{optimization_strategy}_{feature_selection}.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
    
    except Exception as e:
        logger.error(f"Failed to benchmark {dataset_name}: {e}")
        logger.error(traceback.format_exc())
        
        # Generate error results
        results = {
            "dataset": dataset_name,
            "optimization_strategy": str(optimization_strategy),
            "feature_selection": feature_selection,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save error results
        os.makedirs(os.path.join(benchmark_dir, "training_speed"), exist_ok=True)
        result_file = os.path.join(benchmark_dir, "training_speed", 
                                  f"{dataset_name}_{optimization_strategy}_{feature_selection}_error.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
    
    finally:
        # Cleanup
        try:
            engine.shutdown()
        except Exception as e:
            logger.warning(f"Error during engine shutdown: {e}")
        
        gc.collect()
    
    logger.info(f"Training benchmark completed for {dataset_name}")
    return results

def benchmark_feature_selection(dataset_name, benchmark_dir=None, logger=None):
    """Benchmark the effectiveness of feature selection."""
    logger.info(f"Benchmarking feature selection on {dataset_name}")
    
    # Load dataset
    if dataset_name == "synthetic_classification":
        X, y, task_type = create_synthetic_dataset(n_samples=5000, n_features=100, classification=True)
    elif dataset_name == "synthetic_regression":
        X, y, task_type = create_synthetic_dataset(n_samples=5000, n_features=100, classification=False)
    else:
        X, y, task_type = load_dataset(dataset_name, logger)
    
    # Add some noise features
    n_features_original = X.shape[1]
    noise = pd.DataFrame(np.random.randn(X.shape[0], 20), 
                         columns=[f"noise_{i}" for i in range(20)])
    X = pd.concat([X, noise], axis=1)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Run with and without feature selection
    results = {}
    
    for fs in [False, True]:
        logger.info(f"Testing with feature_selection={fs}")
        
        # Create configuration
        config = MLTrainingEngineConfig(
            task_type=task_type,
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
            feature_selection=fs,
            feature_selection_method="mutual_info",
            feature_selection_k=n_features_original if fs else None,
            cv_folds=3,
            optimization_iterations=10,
            model_path=os.path.join(benchmark_dir, "models"),
            experiment_tracking=True,
            log_level="INFO"
        )
        
        # Initialize engine
        engine = MLTrainingEngine(config)
        
        # Define model
        if task_type == TaskType.CLASSIFICATION:
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
        
        # Measure training time
        start_time = time.time()
        peak_memory_before = memory_usage(-1, interval=0.1, timeout=1)[0]
        
        try:
            best_model, metrics = engine.train_model(
                model=model,
                model_name=f"{dataset_name}_{fs}",
                param_grid={},  # Use default parameters
                X=X_train,
                y=y_train,
                X_test=X_test,
                y_test=y_test
            )
        except Exception as e:
            logger.error(f"Error during feature selection benchmark: {e}")
            metrics = {"error": str(e)}
            best_model = None
        
        end_time = time.time()
        peak_memory_after = memory_usage(-1, interval=0.1, timeout=1)[0]
        memory_used = peak_memory_after - peak_memory_before
        
        # Store results
        fs_str = "with_fs" if fs else "without_fs"
        results[fs_str] = {
            "training_time": end_time - start_time,
            "memory_used_mb": memory_used if memory_used > 0 else None,
            "metrics": metrics
        }
        
        # Cleanup
        try:
            engine.shutdown()
        except Exception as e:
            logger.warning(f"Error during engine shutdown: {e}")
            
        gc.collect()
    
    # Save results
    os.makedirs(os.path.join(benchmark_dir, "feature_selection"), exist_ok=True)
    result_file = os.path.join(benchmark_dir, "feature_selection", f"{dataset_name}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    # Create visualization
    try:
        times = [results['without_fs']['training_time'], results['with_fs']['training_time']]
        labels = ['Without Feature Selection', 'With Feature Selection']
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, times, color=['blue', 'green'])
        plt.xlabel('Method')
        plt.ylabel('Training Time (s)')
        plt.title(f'Impact of Feature Selection on Training Time - {dataset_name}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(benchmark_dir, "feature_selection", f"{dataset_name}_comparison.png"))
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create feature selection visualization: {e}")
    
    logger.info(f"Feature selection benchmark completed for {dataset_name}")
    return results

def benchmark_batch_processing(batch_sizes=[10, 50, 100, 500], benchmark_dir=None, logger=None):
    """Benchmark batch processing efficiency."""
    logger.info(f"Benchmarking batch processing with sizes: {batch_sizes}")
    
    # Load or create dataset
    try:
        X, y, task_type = create_synthetic_dataset(n_samples=5000, n_features=20, classification=True)
    except Exception as e:
        logger.error(f"Error creating synthetic dataset: {e}")
        # Fallback to a smaller dataset
        X, y, task_type = load_dataset("breast_cancer", logger)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create results container
    results = {
        "batch_sizes": batch_sizes,
        "inference_times": [],
        "throughputs": [],
        "samples_per_batch": []
    }
    
    # Create batch processor configuration
    batch_config = BatchProcessorConfig(
        enable_adaptive_batching=False,
        enable_monitoring=True,
        enable_memory_optimization=True
    )
    
    # Create configuration
    config = MLTrainingEngineConfig(
        task_type=task_type,
        optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
        feature_selection=False,
        cv_folds=3,
        optimization_iterations=5,
        model_path=os.path.join(benchmark_dir, "models"),
        experiment_tracking=False,
        batch_processing_config=batch_config,
        log_level="INFO"
    )
    
    # Initialize engine
    engine = MLTrainingEngine(config)
    
    # Train a base model
    logger.info("Training a base model for batch inference testing")
    if task_type == TaskType.CLASSIFICATION:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    try:
        engine.train_model(model, "batch_model", {}, X_train, y_train)
    except Exception as e:
        logger.error(f"Error training model for batch processing: {e}")
        return {"error": str(e)}
    
    # Benchmark different batch sizes
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        # Try to update batch processor configuration
        try:
            new_batch_config = BatchProcessorConfig(
                min_batch_size=batch_size,
                max_batch_size=batch_size,
                initial_batch_size=batch_size,
                enable_adaptive_batching=False,
                enable_memory_optimization=True
            )
            engine.config.batch_processing_config = new_batch_config
        except Exception as e:
            logger.warning(f"Could not update batch config: {e}")
        
        # Measure batch inference time
        start_time = time.time()
        
        # Create a generator to simulate streaming data
        def data_generator():
            for i in range(0, len(X_test), batch_size):
                end_idx = min(i + batch_size, len(X_test))
                yield X_test.iloc[i:end_idx]
        
        # Try to run batch inference
        predictions = []
        batch_count = 0
        total_samples = 0
        
        try:
            # First try with run_batch_inference if available
            predictions = engine.run_batch_inference(data_generator(), batch_size=batch_size)
            batch_count = len(predictions)
            total_samples = len(X_test)
        except (AttributeError, Exception) as e:
            logger.warning(f"Could not use run_batch_inference: {e}")
            
            # Fallback: use predict method in a loop
            predictions = []
            batch_count = 0
            total_samples = 0
            
            for batch in data_generator():
                try:
                    batch_pred = engine.predict(batch)
                    predictions.append(batch_pred)
                    batch_count += 1
                    total_samples += len(batch)
                except Exception as inner_e:
                    logger.error(f"Error in batch prediction: {inner_e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput (samples per second)
        throughput = total_samples / total_time if total_time > 0 else 0
        samples_per_batch = total_samples / batch_count if batch_count > 0 else 0
        
        # Store results
        results["inference_times"].append(total_time)
        results["throughputs"].append(throughput)
        results["samples_per_batch"].append(samples_per_batch)
        
        logger.info(f"Batch size: {batch_size}, Time: {total_time:.2f}s, Throughput: {throughput:.2f} samples/s")
    
    # Save results
    os.makedirs(os.path.join(benchmark_dir, "batch_processing"), exist_ok=True)
    result_file = os.path.join(benchmark_dir, "batch_processing", "batch_size_comparison.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create visualization
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Inference Time
        plt.subplot(2, 2, 1)
        plt.plot(batch_sizes, results["inference_times"], marker='o', color='blue')
        plt.xlabel('Batch Size')
        plt.ylabel('Inference Time (s)')
        plt.title('Total Inference Time vs Batch Size')
        plt.grid(True)
        
        # Plot 2: Throughput
        plt.subplot(2, 2, 2)
        plt.plot(batch_sizes, results["throughputs"], marker='o', color='green')
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (samples/s)')
        plt.title('Throughput vs Batch Size')
        plt.grid(True)
        
        # Plot 3: Throughput per batch size
        plt.subplot(2, 2, 3)
        plt.bar(range(len(batch_sizes)), results["throughputs"], color='orange')
        plt.xticks(range(len(batch_sizes)), [str(size) for size in batch_sizes])
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (samples/s)')
        plt.title('Throughput Comparison by Batch Size')
        
        # Plot 4: Inference time per sample
        plt.subplot(2, 2, 4)
        inference_per_sample = [time/samples for time, samples in 
                                zip(results["inference_times"], results["samples_per_batch"])]
        plt.plot(batch_sizes, inference_per_sample, marker='s', color='red')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Sample (s)')
        plt.title('Processing Time per Sample vs Batch Size')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(benchmark_dir, "batch_processing", "batch_size_comparison.png"))
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create batch processing visualization: {e}")
    
    # Cleanup
    try:
        engine.shutdown()
    except Exception as e:
        logger.warning(f"Error during engine shutdown: {e}")
    
    gc.collect()
    
    logger.info("Batch processing benchmark completed")
    return results

def benchmark_model_comparison(benchmark_dir=None, logger=None):
    """Benchmark multiple model comparison capabilities."""
    logger.info("Benchmarking model comparison")
    
    # Load dataset
    X, y, task_type = load_dataset("breast_cancer", logger)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create configuration
    config = MLTrainingEngineConfig(
        task_type=task_type,
        optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
        feature_selection=False,
        cv_folds=3,
        optimization_iterations=5,
        model_path=os.path.join(benchmark_dir, "models"),
        experiment_tracking=True,
        log_level="INFO"
    )
    
    # Initialize engine
    engine = MLTrainingEngine(config)
    
    # Define models
    models = {
        "rf": RandomForestClassifier(random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
        "lr": LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    }
    
    param_grids = {
        "rf": {
            'n_estimators': [50, 100],
            'max_depth': [None, 10]
        },
        "gb": {
            'n_estimators': [50, 100],
            'max_depth': [3, 5]
        },
        "lr": {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2']
        }
    }
    
    # Train models and track times
    results = {
        "models": list(models.keys()),
        "training_times": [],
        "memory_usage": [],
        "metrics": {}
    }
    
    for name, model in models.items():
        logger.info(f"Training model: {name}")
        
        # Measure training time and memory
        start_time = time.time()
        peak_memory_before = memory_usage(-1, interval=0.1, timeout=1)[0]
        
        try:
            best_model, metrics = engine.train_model(
                model=model,
                model_name=name,
                param_grid=param_grids[name],
                X=X_train,
                y=y_train
            )
            
            results["metrics"][name] = metrics
        except Exception as e:
            logger.error(f"Error training model {name}: {e}")
            results["metrics"][name] = {"error": str(e)}
        
        end_time = time.time()
        peak_memory_after = memory_usage(-1, interval=0.1, timeout=1)[0]
        memory_used = peak_memory_after - peak_memory_before
        
        results["training_times"].append(end_time - start_time)
        results["memory_usage"].append(memory_used if memory_used > 0 else None)
    
    # Evaluate all models
    try:
        eval_results = engine.evaluate_all_models(X_test, y_test)
        results["evaluation"] = eval_results
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        results["evaluation"] = {"error": str(e)}
    
    # Try to generate comparison report
    try:
        report_path = os.path.join(benchmark_dir, "model_comparison", "comparison_report.html")
        os.makedirs(os.path.join(benchmark_dir, "model_comparison"), exist_ok=True)
        engine.generate_report(report_path)
        results["report_path"] = report_path
    except Exception as e:
        logger.warning(f"Could not generate model comparison report: {e}")
    
    # Save results
    os.makedirs(os.path.join(benchmark_dir, "model_comparison"), exist_ok=True)
    result_file = os.path.join(benchmark_dir, "model_comparison", "results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    # Create visualization
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Training time comparison
        plt.subplot(2, 1, 1)
        bars = plt.bar(results["models"], results["training_times"], color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time Comparison')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        # Plot 2: Accuracy comparison if available
        plt.subplot(2, 1, 2)
        
        # Try to extract accuracy from metrics
        accuracies = []
        for model in results["models"]:
            accuracy = None
            
            # First try evaluation results
            if "evaluation" in results and model in results["evaluation"]:
                accuracy = results["evaluation"][model].get("accuracy")
            
            # If not found, try metrics
            if accuracy is None and model in results["metrics"]:
                accuracy = results["metrics"][model].get("accuracy")
            
            accuracies.append(accuracy if accuracy is not None else 0)
        
        bars = plt.bar(results["models"], accuracies, color='green')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(benchmark_dir, "model_comparison", "model_comparison.png"))
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create model comparison visualization: {e}")
    
    # Cleanup
    try:
        engine.shutdown()
    except Exception as e:
        logger.warning(f"Error during engine shutdown: {e}")
    
    gc.collect()
    
    logger.info("Model comparison benchmark completed")
    return results

# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def create_training_speed_plots(benchmark_dir):
    """
    Create detailed visualizations for training speed benchmarks.
    
    Parameters:
    -----------
    benchmark_dir : str
        Path to the benchmark results directory
    """
    training_speed_dir = os.path.join(benchmark_dir, "training_speed")
    if not os.path.exists(training_speed_dir):
        print(f"Training speed results not found in {training_speed_dir}")
        return
    
    # Collect data from all training speed benchmark files
    results = []
    for file in os.listdir(training_speed_dir):
        if file.endswith('.json'):
            with open(os.path.join(training_speed_dir, file), 'r') as f:
                data = json.load(f)
                results.append({
                    'dataset': data.get('dataset', 'unknown'),
                    'strategy': data.get('optimization_strategy', 'unknown'),
                    'feature_selection': data.get('feature_selection', False),
                    'training_time': data.get('training_time', 0),
                    'memory_used_mb': data.get('memory_used_mb', 0)
                })
    
    if not results:
        print("No training speed results found")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Create directory for advanced plots
    plots_dir = os.path.join(benchmark_dir, "advanced_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Training time by optimization strategy across datasets
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='dataset', y='training_time', hue='strategy', data=df)
    plt.title('Training Time by Optimization Strategy Across Datasets', fontsize=14)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.legend(title='Optimization Strategy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_time_by_strategy.png'))
    plt.close()
    
    # Plot 2: Memory usage by optimization strategy
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='strategy', y='memory_used_mb', data=df)
    plt.title('Memory Usage by Optimization Strategy', fontsize=14)
    plt.xlabel('Optimization Strategy', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'memory_usage_by_strategy.png'))
    plt.close()
    
    # Plot 3: Training time vs Memory usage scatter plot
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x='training_time', y='memory_used_mb', 
                             hue='strategy', size='dataset', 
                             sizes=(100, 200), alpha=0.7, data=df)
    plt.title('Training Time vs Memory Usage', fontsize=14)
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'time_vs_memory_scatter.png'))
    plt.close()
    
    # Plot 4: Impact of feature selection on training time
    if 'feature_selection' in df.columns and any(df['feature_selection']):
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='dataset', y='training_time', hue='feature_selection', data=df)
        plt.title('Impact of Feature Selection on Training Time', fontsize=14)
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Training Time (seconds)', fontsize=12)
        plt.legend(title='Feature Selection')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_selection_impact.png'))
        plt.close()
    
    print(f"Advanced training speed plots generated in {plots_dir}")

def create_batch_processing_plots(benchmark_dir):
    """
    Create detailed visualizations for batch processing benchmarks.
    
    Parameters:
    -----------
    benchmark_dir : str
        Path to the benchmark results directory
    """
    batch_file = os.path.join(benchmark_dir, "batch_processing", "batch_size_comparison.json")
    if not os.path.exists(batch_file):
        print(f"Batch processing results not found at {batch_file}")
        return
    
    # Load batch processing data
    with open(batch_file, 'r') as f:
        data = json.load(f)
    
    batch_sizes = data.get('batch_sizes', [])
    inference_times = data.get('inference_times', [])
    throughputs = data.get('throughputs', [])
    
    if not batch_sizes or len(batch_sizes) != len(inference_times) or len(batch_sizes) != len(throughputs):
        print("Invalid batch processing data")
        return
    
    # Create directory for advanced plots
    plots_dir = os.path.join(benchmark_dir, "advanced_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame({
        'batch_size': batch_sizes,
        'inference_time': inference_times,
        'throughput': throughputs,
        'samples_per_batch': data.get('samples_per_batch', [0] * len(batch_sizes)),
        'time_per_sample': [t/b if b > 0 else 0 for t, b in zip(inference_times, batch_sizes)]
    })
    
    # Plot 1: Advanced throughput visualization
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(x='batch_size', y='throughput', data=df, marker='o', linewidth=2.5)
    
    # Add polynomial trend line
    if len(batch_sizes) > 1:
        z = np.polyfit(batch_sizes, throughputs, 2)
        p = np.poly1d(z)
        batch_range = np.linspace(min(batch_sizes), max(batch_sizes), 100)
        plt.plot(batch_range, p(batch_range), "r--", alpha=0.7)
    
    plt.title('Batch Size vs. Throughput with Trend', fontsize=14)
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Throughput (samples/second)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate optimal batch size
    if throughputs:
        optimal_idx = throughputs.index(max(throughputs))
        optimal_batch = batch_sizes[optimal_idx]
        optimal_throughput = throughputs[optimal_idx]
        plt.annotate(f'Optimal: {optimal_batch}\n({optimal_throughput:.2f} samples/s)',
                    xy=(optimal_batch, optimal_throughput),
                    xytext=(optimal_batch + 20, optimal_throughput),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'batch_throughput_with_trend.png'))
    plt.close()
    
    # Plot 2: Efficiency analysis - time per sample vs batch size
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(x='batch_size', y='time_per_sample', data=df, marker='o', linewidth=2.5)
    plt.title('Processing Efficiency (Time per Sample)', fontsize=14)
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Time per Sample (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'batch_efficiency.png'))
    plt.close()
    
    # Plot 3: Batch size optimization visualization
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color = 'tab:blue'
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Throughput (samples/s)', color=color, fontsize=12)
    ax1.plot(batch_sizes, throughputs, color=color, marker='o', linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Total Inference Time (s)', color=color, fontsize=12)
    ax2.plot(batch_sizes, inference_times, color=color, marker='s', linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Batch Size Optimization: Throughput vs. Total Time', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'batch_optimization.png'))
    plt.close()
    
    print(f"Advanced batch processing plots generated in {plots_dir}")

def create_model_comparison_plots(benchmark_dir):
    """
    Create detailed visualizations for model comparison benchmarks.
    
    Parameters:
    -----------
    benchmark_dir : str
        Path to the benchmark results directory
    """
    model_file = os.path.join(benchmark_dir, "model_comparison", "results.json")
    if not os.path.exists(model_file):
        print(f"Model comparison results not found at {model_file}")
        return
    
    # Load model comparison data
    with open(model_file, 'r') as f:
        data = json.load(f)
    
    models = data.get('models', [])
    training_times = data.get('training_times', [])
    
    if not models or len(models) != len(training_times):
        print("Invalid model comparison data")
        return
    
    # Create directory for advanced plots
    plots_dir = os.path.join(benchmark_dir, "advanced_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract performance metrics if available
    metrics = {}
    if 'evaluation' in data:
        evaluation = data['evaluation']
        for model in models:
            if model in evaluation:
                # Try to extract common performance metrics
                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score', 'rmse']:
                    if metric in evaluation[model]:
                        if metric not in metrics:
                            metrics[metric] = []
                        metrics[metric].append(evaluation[model][metric])
    
    # Plot 1: Training time comparison with advanced styling
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, training_times, color=sns.color_palette("viridis", len(models)))
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Training Time Comparison', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_training_time.png'))
    plt.close()
    
    # Plot 2: Performance metrics comparison (if available)
    if metrics:
        for metric_name, metric_values in metrics.items():
            if len(metric_values) == len(models):
                plt.figure(figsize=(12, 8))
                bars = plt.bar(models, metric_values, 
                              color=sns.color_palette("muted", len(models)))
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
                
                plt.title(f'Model {metric_name.replace("_", " ").title()} Comparison', fontsize=14)
                plt.xlabel('Model', fontsize=12)
                plt.ylabel(metric_name.replace("_", " ").title(), fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'model_{metric_name}.png'))
                plt.close()
    
    print(f"Advanced model comparison plots generated in {plots_dir}")

def create_feature_selection_plots(benchmark_dir):
    """
    Create detailed visualizations for feature selection benchmarks.
    
    Parameters:
    -----------
    benchmark_dir : str
        Path to the benchmark results directory
    """
    feature_selection_dir = os.path.join(benchmark_dir, "feature_selection")
    if not os.path.exists(feature_selection_dir):
        print(f"Feature selection results not found in {feature_selection_dir}")
        return
    
    # Create directory for advanced plots
    plots_dir = os.path.join(benchmark_dir, "advanced_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Collect data from all feature selection benchmark files
    results = []
    for file in os.listdir(feature_selection_dir):
        if file.endswith('.json'):
            dataset = file.split('.')[0]
            with open(os.path.join(feature_selection_dir, file), 'r') as f:
                data = json.load(f)
                
                # Extract training times
                time_without_fs = data.get('without_fs', {}).get('training_time')
                time_with_fs = data.get('with_fs', {}).get('training_time')
                
                # Extract performance metrics if available
                perf_without_fs = None
                perf_with_fs = None
                
                metrics_without = data.get('without_fs', {}).get('metrics', {})
                metrics_with = data.get('with_fs', {}).get('metrics', {})
                
                # Try to find a common metric between both results
                for metric in ['accuracy', 'r2_score', 'f1_score', 'precision', 'recall']:
                    if metric in metrics_without and metric in metrics_with:
                        perf_without_fs = metrics_without[metric]
                        perf_with_fs = metrics_with[metric]
                        metric_name = metric
                        break
                
                # Add to results if we have both training times
                if isinstance(time_without_fs, (int, float)) and isinstance(time_with_fs, (int, float)):
                    results.append({
                        'dataset': dataset,
                        'time_without_fs': time_without_fs,
                        'time_with_fs': time_with_fs,
                        'time_reduction_percent': ((time_without_fs - time_with_fs) / time_without_fs * 100)
                                                if time_without_fs > 0 else 0,
                        'perf_without_fs': perf_without_fs,
                        'perf_with_fs': perf_with_fs,
                        'perf_metric': metric_name if 'metric_name' in locals() else None,
                        'perf_improvement': ((perf_with_fs - perf_without_fs) / abs(perf_without_fs) * 100)
                                        if perf_without_fs and perf_with_fs and perf_without_fs != 0 else None
                    })
    
    if not results:
        print("No valid feature selection results found")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Plot 1: Time reduction percentage
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['dataset'], df['time_reduction_percent'], color='lightgreen')
    
    # Add horizontal line at 0%
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        text_color = 'green' if height > 0 else 'red'
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', color=text_color, fontweight='bold')
    
    plt.title('Training Time Reduction with Feature Selection', fontsize=14)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Time Reduction (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_selection_time_reduction.png'))
    plt.close()
    
    # Plot 2: Performance improvement (if available)
    if 'perf_improvement' in df.columns and df['perf_improvement'].notna().any():
        plt.figure(figsize=(12, 8))
        
        # Get only rows with valid performance improvement data
        perf_df = df[df['perf_improvement'].notna()]
        
        bars = plt.bar(perf_df['dataset'], perf_df['perf_improvement'],
                      color=sns.color_palette("Spectral", len(perf_df)))
        
        # Add horizontal line at 0%
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            text_color = 'green' if height > 0 else 'red'
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', color=text_color, fontweight='bold')
        
        plt.title('Performance Improvement with Feature Selection', fontsize=14)
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Performance Improvement (%)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_selection_perf_improvement.png'))
        plt.close()
    
    print(f"Advanced feature selection plots generated in {plots_dir}")

def generate_advanced_visualizations(benchmark_dir):
    """
    Generate all advanced visualizations for benchmark results.
    
    Parameters:
    -----------
    benchmark_dir : str
        Path to the benchmark results directory
    """
    print(f"Generating advanced visualizations for results in {benchmark_dir}")
    
    if not os.path.exists(benchmark_dir):
        print(f"Benchmark directory not found: {benchmark_dir}")
        return
    
    # Create directory for advanced plots
    plots_dir = os.path.join(benchmark_dir, "advanced_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate individual benchmark visualizations
    create_training_speed_plots(benchmark_dir)
    create_batch_processing_plots(benchmark_dir)
    create_model_comparison_plots(benchmark_dir)
    create_feature_selection_plots(benchmark_dir)
    
    print(f"All advanced visualizations completed. Results saved to {plots_dir}")

# -----------------------------------------------------------------------------
# Reporting Functions
# -----------------------------------------------------------------------------

def generate_summary_report(benchmark_dir):
    """Generate a comprehensive summary report of all benchmarks."""
    print("Generating comprehensive summary report")
    
    # Collect all results
    report = {
        "training_speed": {},
        "feature_selection": {},
        "batch_processing": {},
        "model_comparison": {},
        "system_info": {
            "timestamp": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "cpu_count": os.cpu_count()
        }
    }
    
    # Collect training speed results
    training_speed_dir = os.path.join(benchmark_dir, "training_speed")
    if os.path.exists(training_speed_dir):
        for file in os.listdir(training_speed_dir):
            if file.endswith('.json'):
                try:
                    with open(os.path.join(training_speed_dir, file), 'r') as f:
                        data = json.load(f)
                        key = f"{data['dataset']}_{data['optimization_strategy']}"
                        report['training_speed'][key] = {
                            'training_time': data.get('training_time'),
                            'memory_used_mb': data.get('memory_used_mb')
                        }
                except Exception as e:
                    print(f"Error loading training speed file {file}: {e}")
    
    # Collect feature selection results
    feature_selection_dir = os.path.join(benchmark_dir, "feature_selection")
    if os.path.exists(feature_selection_dir):
        for file in os.listdir(feature_selection_dir):
            if file.endswith('.json'):
                try:
                    with open(os.path.join(feature_selection_dir, file), 'r') as f:
                        data = json.load(f)
                        dataset = file.split('.')[0]
                        report['feature_selection'][dataset] = {
                            'with_fs': data.get('with_fs', {}),
                            'without_fs': data.get('without_fs', {})
                        }
                except Exception as e:
                    print(f"Error loading feature selection file {file}: {e}")
    
    # Collect batch processing results
    batch_processing_file = os.path.join(benchmark_dir, "batch_processing", "batch_size_comparison.json")
    if os.path.exists(batch_processing_file):
        try:
            with open(batch_processing_file, 'r') as f:
                report['batch_processing'] = json.load(f)
        except Exception as e:
            print(f"Error loading batch processing file: {e}")
    
    # Collect model comparison results
    model_comparison_file = os.path.join(benchmark_dir, "model_comparison", "results.json")
    if os.path.exists(model_comparison_file):
        try:
            with open(model_comparison_file, 'r') as f:
                report['model_comparison'] = json.load(f)
        except Exception as e:
            print(f"Error loading model comparison file: {e}")
    
    # Add standard benchmarks to the report if they exist
    # PLMB benchmarks
    plmb_results_file = os.path.join(benchmark_dir, "plmb_benchmarks", "plmb_results.json")
    if os.path.exists(plmb_results_file):
        try:
            with open(plmb_results_file, 'r') as f:
                report['plmb_benchmarks'] = json.load(f)
        except Exception as e:
            print(f"Error loading PLMB results: {e}")
    
    # UCI benchmarks
    uci_results_file = os.path.join(benchmark_dir, "uci_benchmarks", "uci_results.json")
    if os.path.exists(uci_results_file):
        try:
            with open(uci_results_file, 'r') as f:
                report['uci_benchmarks'] = json.load(f)
        except Exception as e:
            print(f"Error loading UCI results: {e}")
    
    # OpenML benchmarks
    openml_results_file = os.path.join(benchmark_dir, "openml_benchmarks", "openml_results.json")
    if os.path.exists(openml_results_file):
        try:
            with open(openml_results_file, 'r') as f:
                report['openml_benchmarks'] = json.load(f)
        except Exception as e:
            print(f"Error loading OpenML results: {e}")
    
    # AutoML benchmarks
    automl_results_file = os.path.join(benchmark_dir, "automl_benchmarks", "automl_results.json")
    if os.path.exists(automl_results_file):
        try:
            with open(automl_results_file, 'r') as f:
                report['automl_benchmarks'] = json.load(f)
        except Exception as e:
            print(f"Error loading AutoML results: {e}")
    
    # Save summary report
    summary_file = os.path.join(benchmark_dir, "summary_report.json")
    with open(summary_file, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    # Generate HTML report
    html_report = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced ML Training Engine Benchmark Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }
            h1, h2, h3 { color: #2c3e50; margin-top: 20px; }
            h4 { color: #3498db; margin-top: 15px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
            th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
            th { background-color: #3498db; color: white; font-weight: bold; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            tr:hover { background-color: #e6f7ff; }
            .section { margin: 40px 0; padding: 20px; border-radius: 5px; background-color: #f9f9f9; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .summary { background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; background-color: white; }
            .card-header { border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 10px; font-weight: bold; }
            .benchmark-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
            .metric-value { font-size: 1.2em; font-weight: bold; color: #2980b9; }
            .tabs { display: flex; border-bottom: 1px solid #ddd; margin-bottom: 20px; }
            .tab { padding: 10px 15px; cursor: pointer; border: 1px solid transparent; }
            .tab.active { border: 1px solid #ddd; border-bottom: none; background-color: white; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            footer { margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.9em; }
            .chart-container { height: 300px; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h1>Enhanced ML Training Engine Benchmark Report</h1>
        <div class="summary">
            <p><strong>Benchmark Time:</strong> %s</p>
            <p><strong>Python Version:</strong> %s</p>
            <p><strong>OS:</strong> %s</p>
            <p><strong>CPU Cores:</strong> %s</p>
        </div>
    """ % (
        report['system_info']['timestamp'], 
        report['system_info']['python_version'],
        report['system_info']['os'],
        report['system_info']['cpu_count']
    )
    
    # Add navigation tabs for different benchmark sections
    html_report += """
        <div class="tabs">
            <div class="tab active" onclick="showTab('standard-benchmarks')">Standard Benchmarks</div>
            <div class="tab" onclick="showTab('plmb-benchmarks')">PLMB Benchmarks</div>
            <div class="tab" onclick="showTab('uci-benchmarks')">UCI Benchmarks</div>
            <div class="tab" onclick="showTab('openml-benchmarks')">OpenML Benchmarks</div>
            <div class="tab" onclick="showTab('automl-benchmarks')">AutoML Benchmarks</div>
        </div>
        
        <div id="standard-benchmarks" class="tab-content active">
    """
    
    # Add training speed section
    html_report += """
        <div class="section">
            <h2>1. Training Speed Benchmark</h2>
            <p>This benchmark evaluates the training speed across different datasets and optimization strategies.</p>
    """
    
    if report['training_speed']:
        html_report += """
            <table>
                <tr>
                    <th>Dataset & Strategy</th>
                    <th>Training Time (s)</th>
                    <th>Memory Usage (MB)</th>
                </tr>
        """
        
        for key, data in report['training_speed'].items():
            training_time = data.get('training_time', 'N/A')
            if isinstance(training_time, (int, float)):
                training_time = f"{training_time:.2f}"
                
            memory_used = data.get('memory_used_mb', 'N/A')
            if isinstance(memory_used, (int, float)):
                memory_used = f"{memory_used:.2f}"
                
            html_report += f"""
                <tr>
                    <td>{key}</td>
                    <td>{training_time}</td>
                    <td>{memory_used}</td>
                </tr>
            """
        
        html_report += """
            </table>
            <div class="chart-container">
                <canvas id="trainingTimeChart"></canvas>
            </div>
            <img src="training_speed/training_time_comparison.png" alt="Training Time Comparison" />
        """
    else:
        html_report += "<p>No training speed benchmark results available.</p>"
    
    html_report += "</div>"
    
    # Add feature selection section
    html_report += """
        <div class="section">
            <h2>2. Feature Selection Benchmark</h2>
            <p>This benchmark evaluates the impact of feature selection on model performance and training time.</p>
    """
    
    if report['feature_selection']:
        html_report += """
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Training Time (Without FS)</th>
                    <th>Training Time (With FS)</th>
                    <th>Time Difference (%)</th>
                    <th>Performance Difference (%)</th>
                </tr>
        """
        
        for dataset, data in report['feature_selection'].items():
            time_without_fs = data.get('without_fs', {}).get('training_time', 'N/A')
            time_with_fs = data.get('with_fs', {}).get('training_time', 'N/A')
            
            # Calculate time difference
            time_diff = "N/A"
            if isinstance(time_without_fs, (int, float)) and isinstance(time_with_fs, (int, float)) and time_without_fs > 0:
                time_diff = f"{((time_without_fs - time_with_fs) / time_without_fs * 100):.2f}%"
            
            # Try to extract performance metrics
            perf_without_fs = "N/A"
            perf_with_fs = "N/A"
            perf_diff = "N/A"
            
            try:
                # For classification tasks, use accuracy; for regression, use r2_score
                metrics_without = data.get('without_fs', {}).get('metrics', {})
                metrics_with = data.get('with_fs', {}).get('metrics', {})
                
                if 'accuracy' in metrics_without and 'accuracy' in metrics_with:
                    perf_without_fs = metrics_without['accuracy']
                    perf_with_fs = metrics_with['accuracy']
                    if isinstance(perf_without_fs, (int, float)) and isinstance(perf_with_fs, (int, float)):
                        perf_diff = f"{((perf_with_fs - perf_without_fs) / perf_without_fs * 100):.2f}%"
                elif 'r2_score' in metrics_without and 'r2_score' in metrics_with:
                    perf_without_fs = metrics_without['r2_score']
                    perf_with_fs = metrics_with['r2_score']
                    if isinstance(perf_without_fs, (int, float)) and isinstance(perf_with_fs, (int, float)):
                        perf_diff = f"{((perf_with_fs - perf_without_fs) / abs(perf_without_fs) * 100):.2f}%"
            except Exception as e:
                print(f"Error calculating performance difference: {e}")
            
            if isinstance(time_without_fs, (int, float)):
                time_without_fs = f"{time_without_fs:.2f}s"
            if isinstance(time_with_fs, (int, float)):
                time_with_fs = f"{time_with_fs:.2f}s"
            
            html_report += f"""
                <tr>
                    <td>{dataset}</td>
                    <td>{time_without_fs}</td>
                    <td>{time_with_fs}</td>
                    <td>{time_diff}</td>
                    <td>{perf_diff}</td>
                </tr>
            """
        
        html_report += """
            </table>
            <img src="feature_selection/feature_selection_comparison.png" alt="Feature Selection Comparison" />
        """
    else:
        html_report += "<p>No feature selection benchmark results available.</p>"
    
    html_report += "</div>"
    
    # Add batch processing section
    html_report += """
        <div class="section">
            <h2>3. Batch Processing Benchmark</h2>
            <p>This benchmark evaluates the impact of batch size on inference throughput and latency.</p>
    """
    
    if report['batch_processing'] and 'batch_sizes' in report['batch_processing']:
        batch_sizes = report['batch_processing'].get('batch_sizes', [])
        throughputs = report['batch_processing'].get('throughputs', [])
        inference_times = report['batch_processing'].get('inference_times', [])
        
        if len(batch_sizes) == len(throughputs) == len(inference_times):
            html_report += """
                <table>
                    <tr>
                        <th>Batch Size</th>
                        <th>Inference Time (s)</th>
                        <th>Throughput (samples/s)</th>
                    </tr>
            """
            
            for i, batch_size in enumerate(batch_sizes):
                inference_time = inference_times[i]
                throughput = throughputs[i]
                
                if isinstance(inference_time, (int, float)):
                    inference_time = f"{inference_time:.2f}"
                if isinstance(throughput, (int, float)):
                    throughput = f"{throughput:.2f}"
                
                html_report += f"""
                    <tr>
                        <td>{batch_size}</td>
                        <td>{inference_time}</td>
                        <td>{throughput}</td>
                    </tr>
                """
            
            html_report += """
                </table>
                <img src="batch_processing/batch_size_comparison.png" alt="Batch Size Impact on Performance" />
            """
        else:
            html_report += "<p>Batch processing data is incomplete or malformed.</p>"
    else:
        html_report += "<p>No batch processing benchmark results available.</p>"
    
    html_report += "</div>"
    
    # Add model comparison section
    html_report += """
        <div class="section">
            <h2>4. Model Comparison Benchmark</h2>
            <p>This benchmark compares the performance and efficiency of different machine learning models.</p>
    """
    
    if report['model_comparison'] and 'models' in report['model_comparison']:
        models = report['model_comparison'].get('models', [])
        training_times = report['model_comparison'].get('training_times', [])
        
        if len(models) == len(training_times):
            html_report += """
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Training Time (s)</th>
                    </tr>
            """
            
            for i, model in enumerate(models):
                training_time = training_times[i]
                
                if isinstance(training_time, (int, float)):
                    training_time = f"{training_time:.2f}"
                
                html_report += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{training_time}</td>
                    </tr>
                """
            
            html_report += """
                </table>
                <p>For detailed model evaluation metrics, please refer to the model comparison JSON results.</p>
                <img src="model_comparison/model_comparison.png" alt="Model Comparison" />
            """
        else:
            html_report += "<p>Model comparison data is incomplete or malformed.</p>"
    else:
        html_report += "<p>No model comparison benchmark results available.</p>"
    
    html_report += "</div>"
    
    # Add conclusions
    html_report += """
        <div class="section">
            <h2>5. Conclusions & Recommendations</h2>
            <p>Based on the benchmark results, here are some observations and recommendations:</p>
            <ul>
    """
    
    # Add recommendation items based on collected data
    recommendations = []
    
    # Training speed recommendation
    if report['training_speed']:
        best_strategy = None
        best_time = float('inf')
        for key, data in report['training_speed'].items():
            time = data.get('training_time')
            if isinstance(time, (int, float)) and time < best_time:
                best_time = time
                best_strategy = key
        
        if best_strategy:
            recommendations.append(f"<li>The fastest training configuration was \"{best_strategy}\" with a training time of {best_time:.2f} seconds.</li>")
    
    # Feature selection recommendation
    if report['feature_selection']:
        avg_time_improvement = []
        for dataset, data in report['feature_selection'].items():
            time_without_fs = data.get('without_fs', {}).get('training_time')
            time_with_fs = data.get('with_fs', {}).get('training_time')
            
            if isinstance(time_without_fs, (int, float)) and isinstance(time_with_fs, (int, float)):
                improvement = (time_without_fs - time_with_fs) / time_without_fs * 100
                avg_time_improvement.append(improvement)
        
        if avg_time_improvement:
            avg_improvement = sum(avg_time_improvement) / len(avg_time_improvement)
            if avg_improvement > 10:
                recommendations.append(f"<li>Feature selection improved training time by an average of {avg_improvement:.2f}% across datasets.</li>")
            else:
                recommendations.append(f"<li>Feature selection had a modest impact on training time (average improvement: {avg_improvement:.2f}%).</li>")
    
    # Batch processing recommendation
    if report['batch_processing'] and 'batch_sizes' in report['batch_processing'] and 'throughputs' in report['batch_processing']:
        batch_sizes = report['batch_processing']['batch_sizes']
        throughputs = report['batch_processing']['throughputs']
        
        if len(batch_sizes) > 0 and len(throughputs) == len(batch_sizes):
            best_batch_idx = throughputs.index(max(throughputs)) if isinstance(throughputs[0], (int, float)) else 0
            if best_batch_idx < len(batch_sizes):
                recommendations.append(f"<li>The optimal batch size for inference is {batch_sizes[best_batch_idx]}, yielding the highest throughput.</li>")
    
    # Model comparison recommendation
    if report['model_comparison'] and 'models' in report['model_comparison'] and 'evaluation' in report['model_comparison']:
        models = report['model_comparison']['models']
        evaluation = report['model_comparison']['evaluation']
        
        best_model = None
        best_score = -float('inf')
        
        for model in models:
            if model in evaluation:
                # For classification, use accuracy; for regression, use r2
                score = None
                if 'accuracy' in evaluation[model]:
                    score = evaluation[model]['accuracy']
                elif 'r2_score' in evaluation[model]:
                    score = evaluation[model]['r2_score']
                
                if isinstance(score, (int, float)) and score > best_score:
                    best_score = score
                    best_model = model
        
        if best_model:
            metric_name = 'accuracy' if 'accuracy' in evaluation[best_model] else 'r2_score'
            recommendations.append(f"<li>The best performing model was {best_model} with {metric_name} of {best_score:.4f}.</li>")
    
    # General recommendations
    recommendations.append("<li>For best performance, consider using adaptive batch sizing based on resource availability.</li>")
    recommendations.append("<li>Memory usage generally scales with batch size - monitor for resource constraints in production.</li>")
    
    # Add recommendations to report
    if recommendations:
        for rec in recommendations:
            html_report += rec
    else:
        html_report += "<li>Insufficient data to generate specific recommendations.</li>"
    
    html_report += """
            </ul>
        </div>
        
        </div><!-- end of standard-benchmarks -->
        
        <!-- PLMB Benchmarks Tab -->
        <div id="plmb-benchmarks" class="tab-content">
            <div class="section">
                <h2>Python Language Model Benchmarks (PLMB)</h2>
                <p>This section shows performance on standard PLMB benchmarks.</p>
                
                <div class="benchmark-grid">
    """
    
    # Add PLMB benchmark cards
    if 'plmb_benchmarks' in report and report['plmb_benchmarks']:
        for dataset, results in report['plmb_benchmarks'].items():
            if isinstance(results, list) and results:
                # Find the best performing model
                best_model = None
                best_score = -float('inf')
                metric_name = "accuracy" if "accuracy" in results[0].get("metrics", {}) else "r2"
                
                for model_result in results:
                    score = model_result.get("metrics", {}).get(metric_name, 0)
                    if isinstance(score, (int, float)) and score > best_score:
                        best_score = score
                        best_model = model_result.get("model", "unknown")
                
                # Calculate average training time
                avg_time = sum(r.get("training_time", 0) for r in results) / len(results)
                
                html_report += f"""
                    <div class="card">
                        <div class="card-header">{dataset}</div>
                        <p><strong>Best Model:</strong> {best_model}</p>
                        <p><strong>{metric_name.capitalize()}:</strong> <span class="metric-value">{best_score:.4f}</span></p>
                        <p><strong>Avg. Training Time:</strong> {avg_time:.2f}s</p>
                        <p><strong>Models Tested:</strong> {len(results)}</p>
                    </div>
                """
            elif isinstance(results, dict) and "error" not in results:
                html_report += f"""
                    <div class="card">
                        <div class="card-header">{dataset}</div>
                        <p>Detailed results available in the benchmark directory.</p>
                    </div>
                """
    else:
        html_report += "<p>No PLMB benchmark results available.</p>"
    
    html_report += """
                </div>
            </div>
        </div>
        
        <!-- UCI Benchmarks Tab -->
        <div id="uci-benchmarks" class="tab-content">
            <div class="section">
                <h2>UCI ML Repository Benchmarks</h2>
                <p>This section shows performance on standard UCI repository datasets.</p>
                
                <div class="benchmark-grid">
    """
    
    # Add UCI benchmark cards
    if 'uci_benchmarks' in report and report['uci_benchmarks']:
        for dataset, result in report['uci_benchmarks'].items():
            if isinstance(result, dict) and "error" not in result:
                # Extract key metrics based on task type
                task_type = result.get("task_type", "")
                metrics = result.get("metrics", {})
                
                if task_type == TaskType.CLASSIFICATION:
                    main_metric = "accuracy"
                    metric_value = metrics.get(main_metric, "N/A")
                    if isinstance(metric_value, (int, float)):
                        metric_value = f"{metric_value:.4f}"
                else:
                    main_metric = "r2"
                    metric_value = metrics.get(main_metric, "N/A")
                    if isinstance(metric_value, (int, float)):
                        metric_value = f"{metric_value:.4f}"
                
                training_time = result.get("training_time", "N/A")
                if isinstance(training_time, (int, float)):
                    training_time = f"{training_time:.2f}s"
                
                dataset_size = result.get("dataset_size", "N/A")
                feature_count = result.get("feature_count", "N/A")
                
                html_report += f"""
                    <div class="card">
                        <div class="card-header">{dataset}</div>
                        <p><strong>Task Type:</strong> {task_type}</p>
                        <p><strong>{main_metric.capitalize()}:</strong> <span class="metric-value">{metric_value}</span></p>
                        <p><strong>Training Time:</strong> {training_time}</p>
                        <p><strong>Dataset Size:</strong> {dataset_size} samples, {feature_count} features</p>
                    </div>
                """
    else:
        html_report += "<p>No UCI benchmark results available.</p>"
    
    html_report += """
                </div>
            </div>
        </div>
        
        <!-- OpenML Benchmarks Tab -->
        <div id="openml-benchmarks" class="tab-content">
            <div class="section">
                <h2>OpenML Benchmarks</h2>
                <p>This section shows performance on standard OpenML datasets.</p>
                
                <div class="benchmark-grid">
    """
    
    # Add OpenML benchmark cards
    if 'openml_benchmarks' in report and report['openml_benchmarks']:
        for dataset_id, result in report['openml_benchmarks'].items():
            if isinstance(result, dict) and "error" not in result:
                # Extract key metrics based on task type
                task_type = result.get("task_type", "")
                metrics = result.get("metrics", {})
                
                if task_type == TaskType.CLASSIFICATION:
                    main_metric = "accuracy"
                    second_metric = "f1_score"
                else:
                    main_metric = "r2"
                    second_metric = "mse"
                
                metric_value = metrics.get(main_metric, "N/A")
                if isinstance(metric_value, (int, float)):
                    metric_value = f"{metric_value:.4f}"
                
                second_value = metrics.get(second_metric, "N/A")
                if isinstance(second_value, (int, float)):
                    second_value = f"{second_value:.4f}"
                
                training_time = result.get("training_time", "N/A")
                if isinstance(training_time, (int, float)):
                    training_time = f"{training_time:.2f}s"
                
                dataset_size = result.get("dataset_size", "N/A")
                feature_count = result.get("feature_count", "N/A")
                
                html_report += f"""
                    <div class="card">
                        <div class="card-header">Dataset ID: {dataset_id}</div>
                        <p><strong>Task Type:</strong> {task_type}</p>
                        <p><strong>{main_metric.capitalize()}:</strong> <span class="metric-value">{metric_value}</span></p>
                        <p><strong>{second_metric.capitalize()}:</strong> {second_value}</p>
                        <p><strong>Training Time:</strong> {training_time}</p>
                        <p><strong>Dataset Size:</strong> {dataset_size} samples, {feature_count} features</p>
                    </div>
                """
    else:
        html_report += "<p>No OpenML benchmark results available.</p>"
    
    html_report += """
                </div>
            </div>
        </div>
        
        <!-- AutoML Benchmarks Tab -->
        <div id="automl-benchmarks" class="tab-content">
            <div class="section">
                <h2>AutoML Benchmarks</h2>
                <p>This section compares ML Training Engine against standard AutoML benchmarks.</p>
                
                <div class="benchmark-grid">
    """
    
    # Add AutoML benchmark cards
    if 'automl_benchmarks' in report and report['automl_benchmarks']:
        for dataset, result in report['automl_benchmarks'].items():
            if isinstance(result, dict) and "error" not in result:
                # Extract key metrics based on task type
                task_type = result.get("task_type", "")
                metrics = result.get("metrics", {})
                
                if task_type == TaskType.CLASSIFICATION:
                    metrics_to_show = [("accuracy", "Accuracy"), ("f1_score", "F1 Score"), 
                                     ("precision", "Precision"), ("recall", "Recall")]
                else:
                    metrics_to_show = [("r2", "R² Score"), ("mse", "MSE"), 
                                     ("mae", "MAE")]
                
                metrics_html = ""
                for key, label in metrics_to_show:
                    value = metrics.get(key, "N/A")
                    if isinstance(value, (int, float)):
                        value = f"{value:.4f}"
                    metrics_html += f"<p><strong>{label}:</strong> <span class='metric-value'>{value}</span></p>"
                
                training_time = result.get("training_time", "N/A")
                if isinstance(training_time, (int, float)):
                    training_time = f"{training_time:.2f}s"
                
                dataset_size = result.get("dataset_size", "N/A")
                feature_count = result.get("feature_count", "N/A")
                
                html_report += f"""
                    <div class="card">
                        <div class="card-header">{dataset}</div>
                        <p><strong>Task Type:</strong> {task_type}</p>
                        {metrics_html}
                        <p><strong>Training Time:</strong> {training_time}</p>
                        <p><strong>Dataset Size:</strong> {dataset_size} samples, {feature_count} features</p>
                    </div>
                """
    else:
        html_report += "<p>No AutoML benchmark results available.</p>"
    
    html_report += """
                </div>
            </div>
        </div>
        
        <script>
        function showTab(tabId) {
            // Hide all tab contents
            var tabContents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
            }
            
            // Deactivate all tabs
            var tabs = document.getElementsByClassName('tab');
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }
            
            // Show the selected tab content
            document.getElementById(tabId).classList.add('active');
            
            // Activate the clicked tab
            var clickedTab = document.querySelector('.tab[onclick="showTab(\\'' + tabId + '\\')"]');
            clickedTab.classList.add('active');
        }
        
        // Generate charts when document is loaded
        document.addEventListener('DOMContentLoaded', function() {
            // Generate training time chart if canvas exists
            var trainingTimeCanvas = document.getElementById('trainingTimeChart');
            if (trainingTimeCanvas) {
                var ctx = trainingTimeCanvas.getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys("""
    
    # Add training time chart data
    chart_data = {}
    for key, data in report.get('training_speed', {}).items():
        training_time = data.get('training_time')
        if isinstance(training_time, (int, float)):
            chart_data[key] = training_time
    
    html_report += json.dumps(chart_data)
    
    html_report += """),
                        datasets: [{
                            label: 'Training Time (s)',
                            data: Object.values("""
    
    html_report += json.dumps(chart_data)
    
    html_report += """),
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Time (seconds)'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Dataset & Strategy'
                                }
                            }
                        }
                    }
                });
            }
        });
        </script>
        
        <footer>
            <p>Enhanced ML Training Engine Benchmark Report<br>Generated on %s</p>
        </footer>
    </body>
    </html>
    """ % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save HTML report
    html_report_path = os.path.join(benchmark_dir, "benchmark_report.html")
    with open(html_report_path, 'w') as f:
        f.write(html_report)
    
    # Create training time comparison visualization
    try:
        if report['training_speed']:
            plt.figure(figsize=(12, 6))
            
            strategies = list(report['training_speed'].keys())
            times = [report['training_speed'][s]['training_time'] for s in strategies 
                    if isinstance(report['training_speed'][s]['training_time'], (int, float))]
            
            if times:
                plt.bar(range(len(times)), times, color='skyblue')
                plt.xticks(range(len(times)), strategies, rotation=45, ha='right')
                plt.xlabel('Dataset & Strategy')
                plt.ylabel('Training Time (s)')
                plt.title('Training Time Comparison')
                plt.grid(axis='y', linestyle='--', alpha=0.6)
                plt.tight_layout()
                
                os.makedirs(os.path.join(benchmark_dir, "training_speed"), exist_ok=True)
                plt.savefig(os.path.join(benchmark_dir, "training_speed", "training_time_comparison.png"))
                plt.close()
    except Exception as e:
        print(f"Could not create training time comparison visualization: {e}")
    
    print(f"Summary report generated: {html_report_path}")
    return html_report_path

# -----------------------------------------------------------------------------
# Main Benchmark Runner
# -----------------------------------------------------------------------------

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='ML Training Engine Benchmark Suite')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for benchmark results')
    parser.add_argument('--test_selection', type=str, default='all',
                        help='Tests to run, comma-separated (options: training,feature,batch,model,all)')
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Standard Benchmark Utilities
# -----------------------------------------------------------------------------

def run_plmb_benchmarks(benchmark_dir, logger, config=None):
    """
    Run Python Language Model Benchmarks (PLMB).
    
    Parameters:
    -----------
    benchmark_dir : str
        Directory to save benchmark results
    logger : logging.Logger
        Logger instance
    config : dict, optional
        Configuration dictionary with PLMB settings
    """
    logger.info("Running Python Language Model Benchmarks (PLMB)")
    
    if config is None:
        config = DEFAULT_CONFIG["standard_benchmarks"]["plmb"]
    
    # Create results directory
    plmb_dir = os.path.join(benchmark_dir, "plmb_benchmarks")
    os.makedirs(plmb_dir, exist_ok=True)
    
    results = {}
    
    for dataset_name in config["datasets"]:
        logger.info(f"Running PLMB benchmark on {dataset_name}")
        
        try:
            # Load dataset
            X, y, task_type = download_plmb_dataset(dataset_name, logger)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Configure engine with pre-defined settings optimized for this benchmark
            preprocessing_config = PreprocessorConfig(
                normalization=NormalizationType.STANDARD,
                handle_nan=True,
                handle_inf=True,
                detect_outliers=True,
                parallel_processing=True
            )
            
            engine_config = MLTrainingEngineConfig(
                task_type=task_type,
                optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
                feature_selection=True,
                feature_selection_method="mutual_info",
                cv_folds=3,
                optimization_iterations=5,
                preprocessing_config=preprocessing_config,
                experiment_tracking=True
            )
            
            # Initialize engine
            engine = MLTrainingEngine(engine_config)
            
            # Define models to test
            classifiers = [
                (RandomForestClassifier(n_estimators=100, random_state=42), "random_forest"),
                (GradientBoostingClassifier(n_estimators=100, random_state=42), "gradient_boosting"),
                (SVC(probability=True, random_state=42), "svc")
            ]
            
            regressors = [
                (RandomForestRegressor(n_estimators=100, random_state=42), "random_forest"),
                (GradientBoostingClassifier(n_estimators=100, random_state=42), "gradient_boosting"),
                (SVR(), "svr")
            ]
            
            models = classifiers if task_type == TaskType.CLASSIFICATION else regressors
            
            # Run benchmarks and collect results
            dataset_results = []
            
            for model, name in models:
                logger.info(f"Benchmarking {name} on {dataset_name}")
                
                # Measure training time
                start_time = time.time()
                peak_memory_before = memory_usage(-1, interval=0.1, timeout=1)[0]
                
                try:
                    # Train model
                    best_model, metrics = engine.train_model(
                        model=model,
                        model_name=f"plmb_{dataset_name}_{name}",
                        param_grid={},  # Use default params for PLMB
                        X=X_train,
                        y=y_train,
                        X_test=X_test, 
                        y_test=y_test
                    )
                    
                    # Predict on test set
                    y_pred = engine.predict(X_test)
                    
                    # Calculate evaluation metrics
                    if task_type == TaskType.CLASSIFICATION:
                        eval_metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "f1_score": f1_score(y_test, y_pred, average='weighted' 
                                               if len(np.unique(y)) > 2 else 'binary')
                        }
                    else:
                        eval_metrics = {
                            "mse": mean_squared_error(y_test, y_pred),
                            "r2": r2_score(y_test, y_pred)
                        }
                    
                except Exception as e:
                    logger.error(f"Error in PLMB benchmark for {name} on {dataset_name}: {e}")
                    eval_metrics = {"error": str(e)}
                
                end_time = time.time()
                peak_memory_after = memory_usage(-1, interval=0.1, timeout=1)[0]
                memory_used = peak_memory_after - peak_memory_before
                
                # Record results
                model_result = {
                    "model": name,
                    "training_time": end_time - start_time,
                    "memory_usage_mb": memory_used if memory_used > 0 else None,
                    "metrics": eval_metrics
                }
                
                dataset_results.append(model_result)
                
                # Clean up
                try:
                    engine.shutdown()
                except:
                    pass
                
                gc.collect()
            
            # Save dataset results
            results[dataset_name] = dataset_results
            
            with open(os.path.join(plmb_dir, f"{dataset_name}_results.json"), 'w') as f:
                json.dump(dataset_results, f, indent=4, default=str)
            
            # Generate comparison plot
            try:
                plt.figure(figsize=(15, 6))
                
                # Training times
                plt.subplot(1, 2, 1)
                model_names = [r["model"] for r in dataset_results]
                times = [r["training_time"] for r in dataset_results]
                
                plt.bar(model_names, times, color='skyblue')
                plt.title(f'Training Time Comparison - {dataset_name}')
                plt.xlabel('Model')
                plt.ylabel('Training Time (s)')
                plt.xticks(rotation=45)
                
                # Performance metric
                plt.subplot(1, 2, 2)
                if task_type == TaskType.CLASSIFICATION:
                    metric_name = "accuracy"
                else:
                    metric_name = "r2"
                
                scores = []
                for r in dataset_results:
                    metric_value = r.get("metrics", {}).get(metric_name, 0)
                    if isinstance(metric_value, (int, float)):
                        scores.append(metric_value)
                    else:
                        scores.append(0)
                
                plt.bar(model_names, scores, color='lightgreen')
                plt.title(f'{metric_name.capitalize()} Comparison - {dataset_name}')
                plt.xlabel('Model')
                plt.ylabel(metric_name.capitalize())
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plmb_dir, f"{dataset_name}_comparison.png"))
                plt.close()
            
            except Exception as e:
                logger.error(f"Error generating plot for {dataset_name}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to run PLMB benchmark on {dataset_name}: {e}")
            results[dataset_name] = {"error": str(e)}
    
    # Save overall results
    with open(os.path.join(plmb_dir, "plmb_results.json"), 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info("PLMB benchmarks completed")
    return results

def run_uci_benchmarks(benchmark_dir, logger, config=None):
    """
    Run benchmarks on UCI repository datasets.
    
    Parameters:
    -----------
    benchmark_dir : str
        Directory to save benchmark results
    logger : logging.Logger
        Logger instance
    config : dict, optional
        Configuration dictionary with UCI settings
    """
    logger.info("Running UCI Repository Benchmarks")
    
    if config is None:
        config = DEFAULT_CONFIG["standard_benchmarks"]["uci"]
    
    # Create results directory
    uci_dir = os.path.join(benchmark_dir, "uci_benchmarks")
    os.makedirs(uci_dir, exist_ok=True)
    
    results = {}
    
    for dataset_name in config["datasets"]:
        logger.info(f"Running UCI benchmark on {dataset_name}")
        
        try:
            # Load dataset
            X, y, task_type = download_uci_dataset(dataset_name, logger)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Configure engine
            engine_config = MLTrainingEngineConfig(
                task_type=task_type,
                optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
                feature_selection=True,
                cv_folds=3,
                optimization_iterations=5,
                experiment_tracking=True
            )
            
            # Initialize engine
            engine = MLTrainingEngine(engine_config)
            
            # Define models based on task type
            if task_type == TaskType.CLASSIFICATION:
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }
            else:
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }
            
            # Measure training time
            start_time = time.time()
            peak_memory_before = memory_usage(-1, interval=0.1, timeout=1)[0]
            
            try:
                # Train model
                best_model, metrics = engine.train_model(
                    model=model,
                    model_name=f"uci_{dataset_name}",
                    param_grid=param_grid,
                    X=X_train,
                    y=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
                
                # Calculate evaluation metrics
                y_pred = engine.predict(X_test)
                
                if task_type == TaskType.CLASSIFICATION:
                    eval_metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "f1_score": f1_score(y_test, y_pred, average='weighted' 
                                           if len(np.unique(y)) > 2 else 'binary')
                    }
                else:
                    eval_metrics = {
                        "mse": mean_squared_error(y_test, y_pred),
                        "r2": r2_score(y_test, y_pred)
                    }
                
                metrics.update(eval_metrics)
                
            except Exception as e:
                logger.error(f"Error in UCI benchmark for {dataset_name}: {e}")
                metrics = {"error": str(e)}
                best_model = None
            
            end_time = time.time()
            peak_memory_after = memory_usage(-1, interval=0.1, timeout=1)[0]
            memory_used = peak_memory_after - peak_memory_before
            
            # Save results
            result = {
                "dataset": dataset_name,
                "task_type": task_type,
                "training_time": end_time - start_time,
                "memory_usage_mb": memory_used if memory_used > 0 else None,
                "metrics": metrics,
                "dataset_size": len(X),
                "feature_count": X.shape[1]
            }
            
            results[dataset_name] = result
            
            with open(os.path.join(uci_dir, f"{dataset_name}_results.json"), 'w') as f:
                json.dump(result, f, indent=4, default=str)
            
            # Clean up
            try:
                engine.shutdown()
            except:
                pass
            
            gc.collect()
        
        except Exception as e:
            logger.error(f"Failed to run UCI benchmark on {dataset_name}: {e}")
            results[dataset_name] = {"error": str(e)}
    
    # Save overall results
    with open(os.path.join(uci_dir, "uci_results.json"), 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    # Generate comparison visualization
    try:
        # Prepare data for visualization
        datasets = []
        times = []
        memory_usage = []
        accuracies = []
        
        for name, result in results.items():
            if not isinstance(result, dict) or "error" in result:
                continue
                
            datasets.append(name)
            times.append(result.get("training_time", 0))
            memory_usage.append(result.get("memory_usage_mb", 0) or 0)
            
            # Get accuracy or R2 score
            metrics = result.get("metrics", {})
            if result.get("task_type") == TaskType.CLASSIFICATION:
                accuracies.append(metrics.get("accuracy", 0))
            else:
                accuracies.append(metrics.get("r2", 0))
        
        if datasets:
            plt.figure(figsize=(15, 10))
            
            # Plot training times
            plt.subplot(2, 1, 1)
            plt.bar(datasets, times, color='skyblue')
            plt.title('Training Time by Dataset')
            plt.xlabel('Dataset')
            plt.ylabel('Training Time (s)')
            plt.xticks(rotation=45)
            
            # Plot performance
            plt.subplot(2, 1, 2)
            plt.bar(datasets, accuracies, color='lightgreen')
            plt.title('Model Performance by Dataset')
            plt.xlabel('Dataset')
            plt.ylabel('Accuracy / R² Score')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(uci_dir, "uci_comparison.png"))
            plt.close()
    except Exception as e:
        logger.error(f"Error generating UCI comparison visualization: {e}")
    
    logger.info("UCI benchmarks completed")
    return results

def run_openml_benchmarks(benchmark_dir, logger, config=None):
    """
    Run benchmarks on OpenML datasets.
    
    Parameters:
    -----------
    benchmark_dir : str
        Directory to save benchmark results
    logger : logging.Logger
        Logger instance
    config : dict, optional
        Configuration dictionary with OpenML settings
    """
    logger.info("Running OpenML Benchmarks")
    
    if config is None:
        config = DEFAULT_CONFIG["standard_benchmarks"]["openml"]
    
    # Create results directory
    openml_dir = os.path.join(benchmark_dir, "openml_benchmarks")
    os.makedirs(openml_dir, exist_ok=True)
    
    results = {}
    
    for dataset_id in config["dataset_ids"]:
        dataset_name = f"openml_{dataset_id}"
        logger.info(f"Running OpenML benchmark on dataset {dataset_id}")
        
        try:
            # Load dataset
            X, y, task_type = load_dataset(dataset_name, logger, sample_limit=10000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Configure engine
            engine_config = MLTrainingEngineConfig(
                task_type=task_type,
                optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
                feature_selection=True,
                cv_folds=3,
                optimization_iterations=3,
                experiment_tracking=True
            )
            
            # Initialize engine
            engine = MLTrainingEngine(engine_config)
            
            # Set up Pipeline with preprocessing
            if task_type == TaskType.CLASSIFICATION:
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier(random_state=42))
                ])
                param_grid = {
                    "clf__n_estimators": [50, 100],
                    "clf__max_depth": [None, 10],
                }
            else:
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('reg', RandomForestRegressor(random_state=42))
                ])
                param_grid = {
                    "reg__n_estimators": [50, 100],
                    "reg__max_depth": [None, 10],
                }
            
            # Measure training time
            start_time = time.time()
            peak_memory_before = memory_usage(-1, interval=0.1, timeout=1)[0]
            
            try:
                # Train model
                best_model, metrics = engine.train_model(
                    model=model,
                    model_name=f"openml_{dataset_id}",
                    param_grid=param_grid,
                    X=X_train,
                    y=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
                
                # Calculate evaluation metrics
                y_pred = engine.predict(X_test)
                
                if task_type == TaskType.CLASSIFICATION:
                    eval_metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "f1_score": f1_score(y_test, y_pred, average='weighted' 
                                           if len(np.unique(y)) > 2 else 'binary')
                    }
                else:
                    eval_metrics = {
                        "mse": mean_squared_error(y_test, y_pred),
                        "r2": r2_score(y_test, y_pred)
                    }
                
                metrics.update(eval_metrics)
                
            except Exception as e:
                logger.error(f"Error in OpenML benchmark for dataset {dataset_id}: {e}")
                metrics = {"error": str(e)}
                best_model = None
            
            end_time = time.time()
            peak_memory_after = memory_usage(-1, interval=0.1, timeout=1)[0]
            memory_used = peak_memory_after - peak_memory_before
            
            # Save results
            result = {
                "dataset_id": dataset_id,
                "task_type": task_type,
                "training_time": end_time - start_time,
                "memory_usage_mb": memory_used if memory_used > 0 else None,
                "metrics": metrics,
                "dataset_size": len(X),
                "feature_count": X.shape[1]
            }
            
            results[str(dataset_id)] = result
            
            with open(os.path.join(openml_dir, f"openml_{dataset_id}_results.json"), 'w') as f:
                json.dump(result, f, indent=4, default=str)
            
            # Clean up
            try:
                engine.shutdown()
            except:
                pass
            
            gc.collect()
        
        except Exception as e:
            logger.error(f"Failed to run OpenML benchmark on dataset {dataset_id}: {e}")
            results[str(dataset_id)] = {"error": str(e)}
    
    # Save overall results
    with open(os.path.join(openml_dir, "openml_results.json"), 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info("OpenML benchmarks completed")
    return results

def run_automl_benchmarks(benchmark_dir, logger, config=None):
    """
    Run AutoML benchmarks to compare against standard AutoML frameworks.
    
    Parameters:
    -----------
    benchmark_dir : str
        Directory to save benchmark results
    logger : logging.Logger
        Logger instance
    config : dict, optional
        Configuration dictionary with AutoML settings
    """
    logger.info("Running AutoML Benchmarks")
    
    if config is None:
        config = DEFAULT_CONFIG["standard_benchmarks"]["automl_benchmark"]
    
    # Create results directory
    automl_dir = os.path.join(benchmark_dir, "automl_benchmarks")
    os.makedirs(automl_dir, exist_ok=True)
    
    results = {}
    
    for dataset_name in config["datasets"]:
        logger.info(f"Running AutoML benchmark on {dataset_name}")
        
        try:
            # Load dataset
            X, y, task_type = download_automl_dataset(dataset_name, logger)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Configure engine for "AutoML" style optimization
            engine_config = MLTrainingEngineConfig(
                task_type=task_type,
                optimization_strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
                feature_selection=True,
                feature_selection_method="mutual_info",
                cv_folds=5,
                optimization_iterations=20,  # More iterations for better optimization
                experiment_tracking=True
            )
            
            # Initialize engine
            engine = MLTrainingEngine(engine_config)
            
            # Define a more complex model with preprocessing
            if task_type == TaskType.CLASSIFICATION:
                # Create a pipeline with preprocessing and multiple model options
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier(random_state=42))
                ])
                
                # Define a rich hyperparameter space
                param_grid = {
                    "scaler": [StandardScaler(), MinMaxScaler(), None],
                    "clf": [
                        RandomForestClassifier(random_state=42),
                        GradientBoostingClassifier(random_state=42),
                        LogisticRegression(random_state=42, max_iter=1000)
                    ],
                    "clf__RandomForestClassifier__n_estimators": [50, 100, 200],
                    "clf__RandomForestClassifier__max_depth": [None, 10, 20, 30],
                    "clf__GradientBoostingClassifier__n_estimators": [50, 100, 200],
                    "clf__GradientBoostingClassifier__learning_rate": [0.01, 0.05, 0.1],
                    "clf__LogisticRegression__C": [0.1, 1.0, 10.0],
                    "clf__LogisticRegression__penalty": ["l1", "l2"]
                }
            else:
                # Create a pipeline for regression
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('reg', RandomForestRegressor(random_state=42))
                ])
                
                # Define a rich hyperparameter space
                param_grid = {
                    "scaler": [StandardScaler(), MinMaxScaler(), None],
                    "reg": [
                        RandomForestRegressor(random_state=42),
                        GradientBoostingClassifier(random_state=42),
                        ElasticNet(random_state=42)
                    ],
                    "reg__RandomForestRegressor__n_estimators": [50, 100, 200],
                    "reg__RandomForestRegressor__max_depth": [None, 10, 20, 30],
                    "reg__GradientBoostingClassifier__n_estimators": [50, 100, 200],
                    "reg__GradientBoostingClassifier__learning_rate": [0.01, 0.05, 0.1],
                    "reg__ElasticNet__alpha": [0.1, 1.0, 10.0],
                    "reg__ElasticNet__l1_ratio": [0.1, 0.5, 0.9]
                }
            
            # Measure training time
            start_time = time.time()
            peak_memory_before = memory_usage(-1, interval=0.1, timeout=1)[0]
            
            try:
                # Train model with "AutoML" approach
                best_model, metrics = engine.train_model(
                    model=model,
                    model_name=f"automl_{dataset_name}",
                    param_grid=param_grid,
                    X=X_train, 
                    y=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
                
                # Calculate evaluation metrics
                y_pred = engine.predict(X_test)
                
                if task_type == TaskType.CLASSIFICATION:
                    eval_metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, average='weighted'),
                        "recall": recall_score(y_test, y_pred, average='weighted'),
                        "f1_score": f1_score(y_test, y_pred, average='weighted')
                    }
                else:
                    eval_metrics = {
                        "mse": mean_squared_error(y_test, y_pred),
                        "mae": mean_absolute_error(y_test, y_pred),
                        "r2": r2_score(y_test, y_pred)
                    }
                
                metrics.update(eval_metrics)
                
            except Exception as e:
                logger.error(f"Error in AutoML benchmark for {dataset_name}: {e}")
                metrics = {"error": str(e)}
                best_model = None
            
            end_time = time.time()
            peak_memory_after = memory_usage(-1, interval=0.1, timeout=1)[0]
            memory_used = peak_memory_after - peak_memory_before
            
            # Save results
            result = {
                "dataset": dataset_name,
                "task_type": task_type,
                "training_time": end_time - start_time,
                "memory_usage_mb": memory_used if memory_used > 0 else None,
                "metrics": metrics,
                "dataset_size": len(X),
                "feature_count": X.shape[1]
            }
            
            results[dataset_name] = result
            
            with open(os.path.join(automl_dir, f"automl_{dataset_name}_results.json"), 'w') as f:
                json.dump(result, f, indent=4, default=str)
            
            # Clean up
            try:
                engine.shutdown()
            except:
                pass
            
            gc.collect()
        
        except Exception as e:
            logger.error(f"Failed to run AutoML benchmark on {dataset_name}: {e}")
            results[dataset_name] = {"error": str(e)}
    
    # Save overall results
    with open(os.path.join(automl_dir, "automl_results.json"), 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info("AutoML benchmarks completed")
    return results

# -----------------------------------------------------------------------------
# Main Functions
# -----------------------------------------------------------------------------

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced ML Training Engine Benchmark Suite')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for benchmark results')
    parser.add_argument('--test_selection', type=str, default='all',
                        help='Tests to run, comma-separated (options: training,feature,batch,model,plmb,uci,openml,automl,all)')
    parser.add_argument('--benchmark_suite', type=str, default='all',
                        help='Benchmark suite to run (options: plmb,uci,openml,automl,standard,all)')
    parser.add_argument('--sample_limit', type=int, default=None,
                        help='Limit datasets to this number of samples')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom configuration file (JSON)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run benchmarks in parallel')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of parallel workers')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum time in seconds for each benchmark')
    parser.add_argument('--skip_report', action='store_true',
                        help='Skip generating the HTML report')
    parser.add_argument('--skip_visualizations', action='store_true',
                        help='Skip generating visualizations')
    return parser.parse_args()

def main():
    """Run the benchmark suite."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load custom configuration if specified
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            print("Using default configuration")
            config = DEFAULT_CONFIG
    else:
        config = DEFAULT_CONFIG
    
    # Create benchmark directory with timestamp if not specified
    if args.output_dir:
        benchmark_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_dir = f"benchmark_result_{timestamp}"
    
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(benchmark_dir, 'benchmark.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("ml_engine_benchmark")
    
    # Parse test selection
    test_selection = args.test_selection.lower().split(',')
    run_all = 'all' in test_selection
    run_training = run_all or 'training' in test_selection
    run_feature = run_all or 'feature' in test_selection
    run_batch = run_all or 'batch' in test_selection
    run_model = run_all or 'model' in test_selection
    
    # Parse benchmark suite selection
    benchmark_suite = args.benchmark_suite.lower().split(',')
    run_all_benchmarks = 'all' in benchmark_suite
    run_plmb = run_all_benchmarks or 'plmb' in benchmark_suite
    run_uci = run_all_benchmarks or 'uci' in benchmark_suite
    run_openml = run_all_benchmarks or 'openml' in benchmark_suite
    run_automl = run_all_benchmarks or 'automl' in benchmark_suite
    
    # Update config based on command-line arguments
    config["test_training_speed"] = run_training
    config["test_feature_selection"] = run_feature
    config["test_batch_processing"] = run_batch
    config["test_model_comparison"] = run_model
    config["standard_benchmarks"]["plmb"]["enabled"] = run_plmb
    config["standard_benchmarks"]["uci"]["enabled"] = run_uci
    config["standard_benchmarks"]["openml"]["enabled"] = run_openml
    config["standard_benchmarks"]["automl_benchmark"]["enabled"] = run_automl
    
    logger.info("Starting Enhanced ML Training Engine Benchmark Suite")
    logger.info(f"Benchmark results will be saved to: {benchmark_dir}")
    
    # Save the configuration used
    with open(os.path.join(benchmark_dir, 'benchmark_config.json'), 'w') as f:
        json.dump(config, f, indent=4, default=str)
    
    try:
        # Check if the user wants to run benchmarks in parallel
        run_parallel = config.get("run_parallel", False)
        max_workers = config.get("max_workers", None)
        
        if run_parallel:
            logger.info("Running benchmarks in parallel")
            
            # Create benchmark schedule
            benchmark_functions = create_benchmark_schedule(config, benchmark_dir, logger)
            
            # Run benchmarks in parallel
            results = run_parallel_benchmarks(
                benchmark_functions, 
                max_workers=max_workers,
                benchmark_dir=benchmark_dir,
                logger=logger
            )
            
            logger.info(f"Completed {len(results)} parallel benchmarks")
            
            # Save parallel benchmark results
            with open(os.path.join(benchmark_dir, 'parallel_results.json'), 'w') as f:
                json.dump(results, f, indent=4, default=str)
        else:
            # Run benchmarks sequentially
            # Benchmark 1: Training speed with different optimization strategies
            if run_training:
                logger.info("Running training speed benchmarks...")
                
                # Test on small datasets
                benchmark_training_speed("iris", OptimizationStrategy.RANDOM_SEARCH, 
                                        benchmark_dir=benchmark_dir, logger=logger,
                                        sample_limit=args.sample_limit)
                benchmark_training_speed("breast_cancer", OptimizationStrategy.RANDOM_SEARCH, 
                                        benchmark_dir=benchmark_dir, logger=logger,
                                        sample_limit=args.sample_limit)
                benchmark_training_speed("wine", OptimizationStrategy.RANDOM_SEARCH, 
                                        benchmark_dir=benchmark_dir, logger=logger,
                                        sample_limit=args.sample_limit)
                
                # Test different optimization strategies
                benchmark_training_speed("breast_cancer", OptimizationStrategy.GRID_SEARCH, 
                                        benchmark_dir=benchmark_dir, logger=logger,
                                        sample_limit=args.sample_limit)
                benchmark_training_speed("breast_cancer", OptimizationStrategy.BAYESIAN_OPTIMIZATION, 
                                        benchmark_dir=benchmark_dir, logger=logger,
                                        sample_limit=args.sample_limit)
                
                # Test on synthetic dataset
                benchmark_training_speed("synthetic_classification", OptimizationStrategy.RANDOM_SEARCH, 
                                        benchmark_dir=benchmark_dir, logger=logger)
                benchmark_training_speed("synthetic_regression", OptimizationStrategy.RANDOM_SEARCH, 
                                        benchmark_dir=benchmark_dir, logger=logger)
            
            # Benchmark 2: Feature selection
            if run_feature:
                logger.info("Running feature selection benchmarks...")
                benchmark_feature_selection("breast_cancer", benchmark_dir=benchmark_dir, logger=logger)
                benchmark_feature_selection("synthetic_classification", benchmark_dir=benchmark_dir, logger=logger)
                # Add a larger dataset if available
                try:
                    benchmark_feature_selection("mnist", benchmark_dir=benchmark_dir, logger=logger,
                                             sample_limit=args.sample_limit)
                except Exception as e:
                    logger.error(f"Failed to run feature selection on mnist: {e}")
            
            # Benchmark 3: Batch processing
            if run_batch:
                logger.info("Running batch processing benchmarks...")
                benchmark_batch_processing([10, 20, 50, 100, 200, 500], benchmark_dir=benchmark_dir, logger=logger)
            
            # Benchmark 4: Model comparison
            if run_model:
                logger.info("Running model comparison benchmarks...")
                benchmark_model_comparison(benchmark_dir=benchmark_dir, logger=logger)
            
            # Standard ML Benchmarks
            if run_plmb:
                logger.info("Running PLMB benchmarks...")
                run_plmb_benchmarks(benchmark_dir, logger, config.get("standard_benchmarks", {}).get("plmb"))
            
            if run_uci:
                logger.info("Running UCI benchmarks...")
                run_uci_benchmarks(benchmark_dir, logger, config.get("standard_benchmarks", {}).get("uci"))
            
            if run_openml:
                logger.info("Running OpenML benchmarks...")
                run_openml_benchmarks(benchmark_dir, logger, config.get("standard_benchmarks", {}).get("openml"))
            
            if run_automl:
                logger.info("Running AutoML benchmarks...")
                run_automl_benchmarks(benchmark_dir, logger, config.get("standard_benchmarks", {}).get("automl_benchmark"))
        
        # Generate advanced visualizations
        logger.info("Generating advanced visualizations...")
        generate_advanced_visualizations(benchmark_dir)
        
        # Generate summary report
        logger.info("Generating benchmark summary report...")
        report_path = generate_summary_report(benchmark_dir)
        
        logger.info(f"Benchmark completed successfully. Report available at: {report_path}")
        print(f"\nBenchmark completed successfully!")
        print(f"Results saved to: {benchmark_dir}")
        print(f"Summary report: {report_path}")
        
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        logger.error(traceback.format_exc())
        print(f"\nBenchmark failed with error: {e}")
        print(f"See logs for details: {os.path.join(benchmark_dir, 'benchmark.log')}")

if __name__ == "__main__":
    main()