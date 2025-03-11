#!/usr/bin/env python
"""
ML Training Engine Benchmark Script

This script performs comprehensive benchmarking of the ML Training Engine,
testing speed, efficiency, and functionality under various conditions.

Usage:
    python benchmark.py [--output_dir OUTPUT_DIR] [--test_selection TESTS]

Example:
    python benchmark.py --output_dir ./benchmark_results --test_selection all
    python benchmark.py --test_selection training,batch
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
from pathlib import Path
from memory_profiler import memory_usage

# Import scikit-learn components for testing
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

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
    "save_models": True,
    "generate_plots": True,
    "verbose_output": True,
    
    # Training speed benchmark configuration
    "training_speed": {
        "datasets": [
            "iris",
            "breast_cancer",
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
            "synthetic_classification"
        ],
        "methods": [
            "mutual_info",
            "recursive_elimination"
        ],
        "test_with_noise_features": True,
        "noise_feature_count": 20
    },
    
    # Batch processing benchmark configuration
    "batch_processing": {
        "batch_sizes": [10, 20, 50, 100, 200, 500],
        "test_dataset_size": 10000,
        "feature_count": 20,
        "enable_adaptive_batching": False,
        "test_multiple_threads": True,
        "thread_counts": [1, 2, 4, 8]
    },
    
    # Model comparison benchmark configuration
    "model_comparison": {
        "datasets": [
            "breast_cancer",
            "synthetic_classification"
        ],
        "models": {
            "random_forest": {
                "class": "RandomForestClassifier",
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "min_samples_split": [2, 5]
                }
            },
            "gradient_boosting": {
                "class": "GradientBoostingClassifier",
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5],
                    "learning_rate": [0.01, 0.1]
                }
            },
            "logistic_regression": {
                "class": "LogisticRegression",
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"]
                }
            }
        }
    },
    
    # System resource limits (to prevent excessive resource usage)
    "resource_limits": {
        "max_memory_percent": 80,
        "max_cpu_percent": 80,
        "timeout_seconds": 3600  # 1 hour max per benchmark
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

def load_dataset(dataset_name, logger):
    """Load a dataset for benchmarking."""
    logger.info(f"Loading dataset: {dataset_name}")
    
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
    elif dataset_name == "california":
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        task_type = TaskType.REGRESSION
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, task_type

def create_synthetic_dataset(n_samples=10000, n_features=50, classification=True):
    """Create a synthetic dataset for large-scale benchmarking."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure to make the problem learnable
    if classification:
        # Binary classification
        w = np.random.randn(n_features)
        noise = np.random.randn(n_samples) * 0.1
        y = (np.dot(X, w) + noise > 0).astype(int)
        task_type = TaskType.CLASSIFICATION
    else:
        # Regression
        w = np.random.randn(n_features)
        noise = np.random.randn(n_samples) * 0.5
        y = np.dot(X, w) + noise
        task_type = TaskType.REGRESSION
    
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y)
    
    return X, y, task_type

def benchmark_training_speed(dataset_name, optimization_strategy, feature_selection=False, 
                            benchmark_dir=None, logger=None):
    """Benchmark the training speed of the ML Training Engine."""
    logger.info(f"Benchmarking training speed on {dataset_name} with {optimization_strategy}")
    
    # Load dataset
    if dataset_name == "synthetic_classification":
        X, y, task_type = create_synthetic_dataset(n_samples=10000, n_features=30, classification=True)
    elif dataset_name == "synthetic_regression":
        X, y, task_type = create_synthetic_dataset(n_samples=10000, n_features=30, classification=False)
    else:
        X, y, task_type = load_dataset(dataset_name, logger)
    
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
    
    # Measure training time and memory usage
    start_time = time.time()
    peak_memory_before = memory_usage(-1, interval=0.1, timeout=1)[0]
    
    try:
        best_model, metrics = engine.train_model(
            model=model,
            model_name=f"{dataset_name}_{optimization_strategy}",
            param_grid=param_grid,
            X=X_train,
            y=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        # Try to save the model
        try:
            engine.save_model()
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
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
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    os.makedirs(os.path.join(benchmark_dir, "training_speed"), exist_ok=True)
    result_file = os.path.join(benchmark_dir, "training_speed", 
                              f"{dataset_name}_{optimization_strategy}_{feature_selection}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
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
    """Generate a summary report of all benchmarks."""
    print("Generating summary report")
    
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
    
    # Save summary report
    summary_file = os.path.join(benchmark_dir, "summary_report.json")
    with open(summary_file, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    # Generate HTML report
    html_report = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Training Engine Benchmark Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }
            h1, h2, h3 { color: #2c3e50; margin-top: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
            th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
            th { background-color: #3498db; color: white; font-weight: bold; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            tr:hover { background-color: #e6f7ff; }
            .section { margin: 40px 0; padding: 20px; border-radius: 5px; background-color: #f9f9f9; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .summary { background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            footer { margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>ML Training Engine Benchmark Report</h1>
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
        
        <footer>
            <p>ML Training Engine Benchmark Report<br>Generated on %s</p>
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

def main():
    """Run the benchmark suite."""
    # Parse command-line arguments
    args = parse_arguments()
    
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
    
    logger.info("Starting ML Training Engine Benchmark Suite")
    logger.info(f"Benchmark results will be saved to: {benchmark_dir}")
    
    try:
        # Benchmark 1: Training speed with different optimization strategies
        if run_training:
            logger.info("Running training speed benchmarks...")
            
            # Test on small datasets
            benchmark_training_speed("iris", OptimizationStrategy.RANDOM_SEARCH, 
                                    benchmark_dir=benchmark_dir, logger=logger)
            benchmark_training_speed("breast_cancer", OptimizationStrategy.RANDOM_SEARCH, 
                                    benchmark_dir=benchmark_dir, logger=logger)
            
            # Test different optimization strategies
            benchmark_training_speed("breast_cancer", OptimizationStrategy.GRID_SEARCH, 
                                    benchmark_dir=benchmark_dir, logger=logger)
            benchmark_training_speed("breast_cancer", OptimizationStrategy.BAYESIAN_OPTIMIZATION, 
                                    benchmark_dir=benchmark_dir, logger=logger)
            
            # Test on synthetic dataset
            benchmark_training_speed("synthetic_classification", OptimizationStrategy.RANDOM_SEARCH, 
                                    benchmark_dir=benchmark_dir, logger=logger)
        
        # Benchmark 2: Feature selection
        if run_feature:
            logger.info("Running feature selection benchmarks...")
            benchmark_feature_selection("breast_cancer", benchmark_dir=benchmark_dir, logger=logger)
            benchmark_feature_selection("synthetic_classification", benchmark_dir=benchmark_dir, logger=logger)
        
        # Benchmark 3: Batch processing
        if run_batch:
            logger.info("Running batch processing benchmarks...")
            benchmark_batch_processing([10, 20, 50, 100, 200], benchmark_dir=benchmark_dir, logger=logger)
        
        # Benchmark 4: Model comparison
        if run_model:
            logger.info("Running model comparison benchmarks...")
            benchmark_model_comparison(benchmark_dir=benchmark_dir, logger=logger)
        
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