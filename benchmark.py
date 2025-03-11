import os
import time
import gc
import json
import logging
import argparse
import platform
import threading
import tracemalloc
import warnings
import traceback
import signal
import functools
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from modules.configs import TaskType, OptimizationStrategy, MLTrainingEngineConfig
from modules.engine.train_engine import MLTrainingEngine
from modules.device_optimizer import DeviceOptimizer

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global LRU cache for model configurations
MODEL_CONFIG_CACHE = {}
DATASET_CACHE = {}

# Function to implement timeout for operations
def timeout_handler(timeout_sec):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if platform.system() == "Windows":
                # Windows doesn't support SIGALRM
                return func(*args, **kwargs)
            
            def handle_timeout(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_sec} seconds")
            
            # Set the timeout handler
            original_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(timeout_sec)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Reset the alarm and restore original handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
                
        return wrapper
    return decorator


class ResourceMonitor:
    """Monitor system resources in a separate thread."""
    
    def __init__(self, interval=0.1):
        self.interval = interval
        self.cpu_percentages = []
        self.memory_usages = []
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process(os.getpid())
        self._lock = threading.Lock()
        
    def start(self):
        """Start monitoring resources."""
        with self._lock:
            if not self.monitoring:
                self.monitoring = True
                self.cpu_percentages = []
                self.memory_usages = []
                self.monitor_thread = threading.Thread(target=self._monitor_loop)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()
                
    def stop(self):
        """Stop monitoring and return results."""
        with self._lock:
            if self.monitoring:
                self.monitoring = False
                if self.monitor_thread and self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=2.0)
                
                return self.get_stats()
        return {}
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self.cpu_percentages.append(psutil.cpu_percent(interval=None))
                self.memory_usages.append(self.process.memory_info().rss / (1024 * 1024))  # MB
                time.sleep(self.interval)
            except Exception as e:
                logger.warning(f"Error in resource monitoring: {str(e)}")
                break
                
    def get_stats(self):
        """Get statistics from collected data."""
        with self._lock:
            if not self.cpu_percentages or not self.memory_usages:
                return {}
                
            return {
                "cpu_percent": {
                    "mean": np.mean(self.cpu_percentages),
                    "max": np.max(self.cpu_percentages),
                    "min": np.min(self.cpu_percentages),
                    "std": np.std(self.cpu_percentages),
                },
                "memory_mb": {
                    "mean": np.mean(self.memory_usages),
                    "max": np.max(self.memory_usages),
                    "min": np.min(self.memory_usages),
                    "std": np.std(self.memory_usages),
                    "final": self.memory_usages[-1] if self.memory_usages else 0,
                },
                "samples": len(self.cpu_percentages),
            }


class DatasetManager:
    """Manage dataset loading and preprocessing with caching."""
    
    def __init__(self, sample_size=50000):
        self.sample_size = sample_size
        self.cache = {}
        
    @lru_cache(maxsize=10)
    def get_openml_datasets(self) -> List[Dict]:
        """Get a list of datasets from OpenML for benchmarking with caching."""
        classification_datasets = [
            {"id": "adult", "name": "Adult Census Income", "task": TaskType.CLASSIFICATION},
            {"id": "blood-transfusion-service-center", "name": "Blood Transfusion", "task": TaskType.CLASSIFICATION},
            {"id": "credit-g", "name": "German Credit", "task": TaskType.CLASSIFICATION},
            {"id": "diabetes", "name": "Diabetes", "task": TaskType.CLASSIFICATION},
            {"id": "breast-cancer", "name": "Breast Cancer Wisconsin", "task": TaskType.CLASSIFICATION},
        ]

        regression_datasets = [
            {"id": "boston", "name": "Boston Housing", "task": TaskType.REGRESSION},
            {"id": "california_housing", "name": "California Housing", "task": TaskType.REGRESSION},
            {"id": "auto_prices", "name": "Auto Prices", "task": TaskType.REGRESSION},
            {"id": "medical_charges", "name": "Medical Charges", "task": TaskType.REGRESSION},
        ]

        return classification_datasets + regression_datasets
    
    def load_dataset(self, dataset_info: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a dataset from OpenML with caching."""
        dataset_id = dataset_info["id"]
        
        # Check if dataset is already in cache
        if dataset_id in self.cache:
            logger.info(f"Using cached dataset: {dataset_info['name']}")
            return self.cache[dataset_id]
            
        logger.info(f"Loading dataset: {dataset_info['name']} (ID: {dataset_id})")
        
        try:
            # Fetch dataset with timeout
            dataset = fetch_openml(name=dataset_id, as_frame=True, parser="auto")
            
            X = dataset.data
            y = dataset.target
            
            # Process categorical features efficiently
            cat_cols = X.select_dtypes(include=["category", "object"]).columns
            if not cat_cols.empty:
                # Process categorical columns in batches to save memory
                for col in cat_cols:
                    X[col] = pd.factorize(X[col])[0]
            
            # Handle missing values for numeric columns efficiently
            num_cols = X.select_dtypes(include=np.number).columns
            if not num_cols.empty:
                imputer = SimpleImputer(strategy="median")
                X[num_cols] = imputer.fit_transform(X[num_cols])
                
            # Sample large datasets
            if X.shape[0] > self.sample_size:
                logger.info(f"Sampling large dataset ({X.shape[0]} rows) to {self.sample_size} rows")
                
                # Efficient sampling
                if dataset_info["task"] == TaskType.CLASSIFICATION:
                    # Convert once to numpy for more efficient processing
                    X_np = X.values
                    y_np = y.values if hasattr(y, 'values') else y
                    
                    # Stratified sampling
                    _, X_sampled, _, y_sampled = train_test_split(
                        X_np, y_np, test_size=self.sample_size/X.shape[0], 
                        stratify=y_np, random_state=42
                    )
                    
                    # Convert back to DataFrame/Series for compatibility
                    X = pd.DataFrame(X_sampled, columns=X.columns)
                    y = pd.Series(y_sampled, name=y.name if hasattr(y, 'name') else None)
                else:
                    # Random sampling for regression (more memory efficient)
                    indices = np.random.RandomState(42).choice(X.shape[0], self.sample_size, replace=False)
                    X = X.iloc[indices]
                    y = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
                    
            # Convert to numpy arrays for better performance with ML algorithms
            X_np = X.values if hasattr(X, 'values') else X
            y_np = y.values if hasattr(y, 'values') else y
            
            # Cache the processed dataset
            self.cache[dataset_id] = (X_np, y_np)
            
            return X_np, y_np
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None, None
    
    def clear_cache(self):
        """Clear the dataset cache to free memory."""
        self.cache.clear()


class ModelManager:
    """Manage model configurations with caching."""
    
    def __init__(self):
        self.cache = {}
        
    @lru_cache(maxsize=2)
    def get_models_for_task(self, task_type: TaskType) -> Dict:
        """Get models and param grids for a task with caching."""
        cache_key = str(task_type)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        if task_type == TaskType.CLASSIFICATION:
            models = {
                "random_forest": {
                    "model": RandomForestClassifier(random_state=42),
                    "param_grid": {
                        "model__n_estimators": [50, 100],
                        "model__max_depth": [None, 10],
                        "model__min_samples_split": [2, 5],
                    },
                },
                "logistic_regression": {
                    "model": LogisticRegression(random_state=42, max_iter=1000),
                    "param_grid": {
                        "model__C": [0.1, 1.0, 10.0],
                        "model__solver": ["liblinear", "saga"],
                    },
                },
                "gradient_boosting": {
                    "model": GradientBoostingClassifier(random_state=42),
                    "param_grid": {
                        "model__n_estimators": [50, 100],
                        "model__learning_rate": [0.01, 0.1],
                    },
                },
            }
        else:  # Regression
            models = {
                "random_forest": {
                    "model": RandomForestRegressor(random_state=42),
                    "param_grid": {
                        "model__n_estimators": [50, 100],
                        "model__max_depth": [None, 10],
                        "model__min_samples_split": [2, 5],
                    },
                },
                "linear_regression": {
                    "model": LinearRegression(),
                    "param_grid": {},  # Linear regression doesn't have hyperparameters to tune
                },
                "svr": {
                    "model": SVR(),
                    "param_grid": {
                        "model__C": [0.1, 1.0, 10.0],
                        "model__kernel": ["linear", "rbf"],
                    },
                },
            }
            
        self.cache[cache_key] = models
        return models


class MLBenchmark:
    """Benchmark the performance of the ML Training Engine using datasets from OpenML."""

    def __init__(self, output_dir: str = "./benchmark_results", config_path: str = "./configs", 
                 sample_size: int = 50000):
        self.output_dir = output_dir
        self.config_path = config_path
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.resource_monitor = ResourceMonitor()
        self.sample_size = sample_size
        
        # Create output directories efficiently
        for directory in [output_dir, config_path, 
                         f"{output_dir}/checkpoints", 
                         f"{output_dir}/models", 
                         f"{output_dir}/plots_{self.timestamp}"]:
            os.makedirs(directory, exist_ok=True)

        # Initialize managers
        self.dataset_manager = DatasetManager(sample_size=sample_size)
        self.model_manager = ModelManager()
        
        # Initialize device optimizer (cached)
        self.device_optimizer = DeviceOptimizer(
            config_path=config_path,
            checkpoint_path=f"{output_dir}/checkpoints",
            model_registry_path=f"{output_dir}/models",
        )

        # Get system info (only once, cache it)
        self.system_info = self._get_system_info()
        logger.info(f"System Information:\n{json.dumps(self.system_info, indent=2)}")
        
        # Cache for engine configurations
        self.engine_config_cache = {}

    def _get_system_info(self) -> Dict:
        """Get detailed system information for benchmarking context."""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            cpu_brand = cpu_info.get("brand_raw", "Unknown")
            cpu_arch = cpu_info.get("arch", "Unknown")
            cpu_bits = cpu_info.get("bits", "Unknown")
            cpu_freq = cpu_info.get("hz_advertised_friendly", "Unknown")
        except (ImportError, Exception):
            cpu_brand = platform.processor()
            cpu_arch = platform.machine()
            cpu_bits = platform.architecture()[0]
            cpu_freq = "Unknown"

        # Memory info
        mem = psutil.virtual_memory()

        # Disk info
        disk = psutil.disk_usage("/")

        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "processor": {
                "brand": cpu_brand,
                "architecture": cpu_arch,
                "bits": cpu_bits,
                "frequency": cpu_freq,
                "physical_cores": psutil.cpu_count(logical=False) or 1,
                "logical_cores": psutil.cpu_count(logical=True) or 1,
            },
            "memory": {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_percent": mem.percent,
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_percent": disk.percent,
            },
            "python_version": platform.python_version(),
            "benchmark_timestamp": self.timestamp,
        }

        return system_info

    @lru_cache(maxsize=10)
    def _create_engine_config(self, task_type: TaskType, optimization_strategy: OptimizationStrategy) -> MLTrainingEngineConfig:
        """Create a training engine configuration with caching."""
        cache_key = f"{task_type}_{optimization_strategy}"
        
        if cache_key in self.engine_config_cache:
            return self.engine_config_cache[cache_key]
            
        # Get optimized configurations from device optimizer (cached)
        quant_config = self.device_optimizer.get_optimal_quantization_config()
        batch_config = self.device_optimizer.get_optimal_batch_processor_config()
        preproc_config = self.device_optimizer.get_optimal_preprocessor_config()
        inference_config = self.device_optimizer.get_optimal_inference_engine_config()

        # Create training engine config
        config = MLTrainingEngineConfig(
            task_type=task_type,
            optimization_strategy=optimization_strategy,
            optimization_iterations=20,
            cv_folds=3,
            test_size=0.2,
            random_state=42,
            n_jobs=-1,  # Use all available cores
            verbose=0,
            model_path=f"{self.output_dir}/models",
            log_level="INFO",
            experiment_tracking=True,
            memory_optimization=True,
            feature_selection=True,
            feature_selection_k=10,
            stratify=(task_type == TaskType.CLASSIFICATION),
            quantization_config=quant_config,
            batch_processing_config=batch_config,
            preprocessing_config=preproc_config,
            inference_config=inference_config,
        )

        # Cache the config
        self.engine_config_cache[cache_key] = config
        
        return config

    @timeout_handler(3600)  # Default 1-hour timeout
    def benchmark_dataset(self, dataset_info: Dict, optimization_strategies: Optional[List[OptimizationStrategy]] = None, 
                         timeout_sec: int = 3600) -> Optional[Dict]:
        """Benchmark the ML Training Engine on a specific dataset with timeout protection."""
        if optimization_strategies is None:
            optimization_strategies = [
                OptimizationStrategy.GRID_SEARCH,
                OptimizationStrategy.RANDOM_SEARCH,
                OptimizationStrategy.ASHT,
            ]

        # Load dataset with caching
        X, y = self.dataset_manager.load_dataset(dataset_info)
        if X is None or y is None:
            logger.warning(f"Skipping dataset {dataset_info['name']} due to loading error")
            return None

        logger.info(f"Benchmarking dataset: {dataset_info['name']} ({X.shape[0]} samples, {X.shape[1]} features)")

        # Split data (only once for all strategies)
        if dataset_info["task"] == TaskType.CLASSIFICATION:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        dataset_results = {
            "dataset_info": {
                "name": dataset_info["name"],
                "id": dataset_info["id"],
                "task": dataset_info["task"].value,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
            },
            "optimization_results": {},
        }

        # Get models for this task (cached)
        models = self.model_manager.get_models_for_task(dataset_info["task"])

        for strategy in optimization_strategies:
            logger.info(f"Testing optimization strategy: {strategy.value}")

            strategy_results = {
                "models": {},
                "overall": {
                    "total_time": 0,
                    "peak_memory_mb": 0,
                    "training_throughput": 0,
                },
            }

            # Create engine config (cached)
            config = self._create_engine_config(dataset_info["task"], strategy)

            # Initialize engine
            engine = MLTrainingEngine(config)

            # Start resource monitoring
            self.resource_monitor.start()

            # Start memory tracking
            tracemalloc.start()

            # Track overall time
            overall_start_time = time.time()

            # Train each model
            for model_name, model_info in models.items():
                logger.info(f"Training model: {model_name}")

                # Track model training time and memory
                start_time = time.time()
                current_memory, peak_memory = tracemalloc.get_traced_memory()

                # Train model
                try:
                    best_model, metrics = engine.train_model(
                        model_info["model"],
                        model_name,
                        model_info["param_grid"],
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                    )

                    # Calculate training time
                    training_time = time.time() - start_time

                    # Get memory usage
                    current_memory, peak_memory = tracemalloc.get_traced_memory()
                    peak_memory_mb = peak_memory / (1024 * 1024)  # MB

                    # Calculate throughput
                    throughput = X_train.shape[0] / training_time  # samples/second

                    # Test inference speed
                    inference_start = time.time()
                    _ = engine.predict(X_test, model_name)
                    inference_time = time.time() - inference_start
                    inference_throughput = X_test.shape[0] / inference_time  # samples/second

                    # Store results
                    strategy_results["models"][model_name] = {
                        "training_time_seconds": training_time,
                        "peak_memory_mb": peak_memory_mb,
                        "training_throughput_samples_per_second": throughput,
                        "inference_time_seconds": inference_time,
                        "inference_throughput_samples_per_second": inference_throughput,
                        "metrics": metrics,
                    }

                    logger.info(f"  - Training time: {training_time:.2f}s")
                    logger.info(f"  - Peak memory: {peak_memory_mb:.2f} MB")
                    logger.info(f"  - Training throughput: {throughput:.2f} samples/s")
                    logger.info(f"  - Inference throughput: {inference_throughput:.2f} samples/s")

                except Exception as e:
                    logger.error(f"Error training model {model_name}: {str(e)}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    strategy_results["models"][model_name] = {"error": str(e)}

            # Calculate overall metrics
            overall_time = time.time() - overall_start_time
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            peak_memory_mb = peak_memory / (1024 * 1024)  # MB
            
            # Avoid division by zero by checking if we have any successful models
            successful_models = [m for m in strategy_results["models"] if "error" not in strategy_results["models"][m]]
            overall_throughput = X_train.shape[0] * len(successful_models) / overall_time if successful_models else 0

            # Stop memory tracking
            tracemalloc.stop()

            # Stop resource monitoring
            resource_data = self.resource_monitor.stop()

            # Store overall results
            strategy_results["overall"] = {
                "total_time_seconds": overall_time,
                "peak_memory_mb": peak_memory_mb,
                "training_throughput_samples_per_second": overall_throughput,
                "resource_monitoring": resource_data,
            }

            # Clean up
            engine.shutdown()
            gc.collect()

            # Store strategy results
            dataset_results["optimization_results"][strategy.value] = strategy_results

            logger.info(f"Completed {strategy.value} optimization in {overall_time:.2f}s")
            logger.info(f"Peak memory usage: {peak_memory_mb:.2f} MB")
            logger.info(f"Overall throughput: {overall_throughput:.2f} samples/s")
            logger.info("-" * 80)

        # Store dataset results
        self.results[dataset_info["id"]] = dataset_results

        # Save intermediate results
        self._save_results()

        return dataset_results

    def run_benchmarks(self, datasets: Optional[List[Dict]] = None, 
                      optimization_strategies: Optional[List[OptimizationStrategy]] = None,
                      timeout_sec: int = 3600) -> Dict:
        """Run benchmarks on all datasets."""
        if datasets is None:
            datasets = self.dataset_manager.get_openml_datasets()

        # Process datasets sequentially to avoid resource contention
        for dataset_info in datasets:
            try:
                # Run benchmark with timeout protection
                self.benchmark_dataset(
                    dataset_info, 
                    optimization_strategies,
                    timeout_sec=timeout_sec
                )
            except TimeoutError as e:
                logger.error(f"Benchmark for dataset {dataset_info['name']} timed out: {str(e)}")
            except Exception as e:
                logger.error(f"Error benchmarking dataset {dataset_info['name']}: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
            # Explicitly clean up after each dataset
            gc.collect()
            
            # Clear any LRU caches that are no longer needed
            if hasattr(self, 'engine_config_cache'):
                self.engine_config_cache.clear()

        # Generate final report
        self._generate_report()

        return self.results

    def _save_results(self) -> str:
        """Save benchmark results to disk."""
        results_file = f"{self.output_dir}/benchmark_results_{self.timestamp}.json"

        # Prepare results with system info
        full_results = {
            "system_info": self.system_info,
            "timestamp": self.timestamp,
            "results": self.results,
        }

        # Save to file with error handling
        try:
            # Use a memory-efficient approach for large results
            with open(results_file, "w") as f:
                json.dump(full_results, f, indent=2)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results to {results_file}: {str(e)}")
            # Try to save to a different location as fallback
            fallback_file = f"./benchmark_results_{self.timestamp}.json"
            try:
                with open(fallback_file, "w") as f:
                    json.dump(full_results, f, indent=2)
                logger.info(f"Results saved to fallback location: {fallback_file}")
                results_file = fallback_file
            except Exception as e2:
                logger.error(f"Failed to save results to fallback location: {str(e2)}")

        return results_file

    def _generate_report(self) -> str:
        """Generate a comprehensive benchmark report more efficiently."""
        report_file = f"{self.output_dir}/benchmark_report_{self.timestamp}.md"

        # Use string concatenation instead of repeated f-strings for better performance
        report_parts = []
        
        # Create report header
        report_parts.append(f"# ML Training Engine Benchmark Report\n\n")
        report_parts.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Add system information
        report_parts.append("## System Information\n\n")
        report_parts.append(f"- **Platform:** {self.system_info['platform']}\n")
        report_parts.append(f"- **Processor:** {self.system_info['processor']['brand']}\n")
        report_parts.append(f"- **CPU Cores:** {self.system_info['processor']['physical_cores']} physical, {self.system_info['processor']['logical_cores']} logical\n")
        report_parts.append(f"- **Memory:** {self.system_info['memory']['total_gb']} GB total, {self.system_info['memory']['available_gb']} GB available\n")
        report_parts.append(f"- **Python Version:** {self.system_info['python_version']}\n\n")

        # Add summary table
        report_parts.append("## Performance Summary\n\n")
        report_parts.append("| Dataset | Task | Samples | Features | Best Strategy | Best Model | Training Time (s) | Throughput (samples/s) | Peak Memory (MB) |\n")
        report_parts.append("|---------|------|---------|----------|---------------|------------|-------------------|------------------------|------------------|\n")

        for dataset_id, dataset_results in self.results.items():
            dataset_info = dataset_results["dataset_info"]

            # Find best strategy and model
            best_strategy = None
            best_model = None
            best_time = float("inf")
            best_throughput = 0
            best_memory = 0

            for strategy, strategy_results in dataset_results["optimization_results"].items():
                for model_name, model_results in strategy_results["models"].items():
                    if "error" in model_results:
                        continue

                    if model_results["training_time_seconds"] < best_time:
                        best_time = model_results["training_time_seconds"]
                        best_throughput = model_results["training_throughput_samples_per_second"]
                        best_memory = model_results["peak_memory_mb"]
                        best_strategy = strategy
                        best_model = model_name

            # Handle case where no successful models were found
            if best_strategy is None:
                report_parts.append(f"| {dataset_info['name']} | {dataset_info['task']} | {dataset_info['n_samples']} | {dataset_info['n_features']} | No successful models | - | - | - | - |\n")
            else:
                report_parts.append(f"| {dataset_info['name']} | {dataset_info['task']} | {dataset_info['n_samples']} | {dataset_info['n_features']} | {best_strategy} | {best_model} | {best_time:.2f} | {best_throughput:.2f} | {best_memory:.2f} |\n")

        # Add detailed results for each dataset
        report_parts.append("\n## Detailed Results\n\n")

        for dataset_id, dataset_results in self.results.items():
            dataset_info = dataset_results["dataset_info"]

            report_parts.append(f"### {dataset_info['name']} ({dataset_info['task']})\n\n")
            report_parts.append(f"- **Samples:** {dataset_info['n_samples']}\n")
            report_parts.append(f"- **Features:** {dataset_info['n_features']}\n\n")

            # Add performance comparison for each optimization strategy
            report_parts.append("#### Performance by Optimization Strategy\n\n")
            report_parts.append("| Strategy | Total Time (s) | Peak Memory (MB) | Training Throughput (samples/s) |\n")
            report_parts.append("|----------|----------------|------------------|--------------------------------|\n")

            for strategy, strategy_results in dataset_results["optimization_results"].items():
                overall = strategy_results["overall"]
                report_parts.append(
                    f"| {strategy} | {overall['total_time_seconds']:.2f} | "
                    f"{overall['peak_memory_mb']:.2f} | "
                    f"{overall['training_throughput_samples_per_second']:.2f} |\n"
                )

            # Add model performance for each strategy
            for strategy, strategy_results in dataset_results["optimization_results"].items():
                report_parts.append(f"\n#### {strategy} Strategy - Model Performance\n\n")
                report_parts.append("| Model | Training Time (s) | Peak Memory (MB) | Training Throughput | Inference Throughput | Metrics |\n")
                report_parts.append("|-------|------------------|------------------|--------------------|--------------------|--------|\n")

                for model_name, model_results in strategy_results["models"].items():
                    if "error" in model_results:
                        report_parts.append(f"| {model_name} | Error: {model_results['error']} | - | - | - | - |\n")
                        continue

                    # Format metrics as a compact string
                    metrics_str = "; ".join([f"{k}: {v:.4f}" for k, v in model_results["metrics"].items()])
                    
                    report_parts.append(
                        f"| {model_name} | {model_results['training_time_seconds']:.2f} | "
                        f"{model_results['peak_memory_mb']:.2f} | "
                        f"{model_results['training_throughput_samples_per_second']:.2f} | "
                        f"{model_results['inference_throughput_samples_per_second']:.2f} | "
                        f"{metrics_str} |\n"
                    )

            # Generate performance plots for this dataset
            self._generate_dataset_plots(dataset_id, dataset_results)
            
            # Add reference to plots in the report
            plot_path = f"plots_{self.timestamp}/{dataset_id}"
            report_parts.append(f"\n#### Performance Visualizations\n\n")
            report_parts.append(f"- [Training Time Comparison]({plot_path}_training_time.png)\n")
            report_parts.append(f"- [Memory Usage Comparison]({plot_path}_memory_usage.png)\n")
            report_parts.append(f"- [Throughput Comparison]({plot_path}_throughput.png)\n\n")
            
            report_parts.append("---\n\n")  # Separator between datasets

        # Append conclusion
        report_parts.append("## Conclusion\n\n")
        report_parts.append("This benchmark report provides a comprehensive performance analysis of different optimization strategies ")
        report_parts.append("and models across various datasets. The results can be used to make informed decisions about which ")
        report_parts.append("combinations work best for specific data characteristics and task types.\n\n")

        # Write report to file efficiently
        try:
            with open(report_file, "w") as f:
                f.write("".join(report_parts))
            logger.info(f"Report generated at {report_file}")
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            # Try fallback location
            fallback_file = f"./benchmark_report_{self.timestamp}.md"
            try:
                with open(fallback_file, "w") as f:
                    f.write("".join(report_parts))
                logger.info(f"Report saved to fallback location: {fallback_file}")
                report_file = fallback_file
            except Exception as e2:
                logger.error(f"Failed to save report to fallback location: {str(e2)}")

        return report_file

    def _generate_dataset_plots(self, dataset_id: str, dataset_results: Dict) -> None:
        """Generate performance visualization plots for a dataset."""
        try:
            dataset_name = dataset_results["dataset_info"]["name"]
            plot_dir = f"{self.output_dir}/plots_{self.timestamp}"
            os.makedirs(plot_dir, exist_ok=True)
            
            # Prepare data for plotting
            strategies = []
            models = []
            training_times = []
            memory_usages = []
            training_throughputs = []
            
            for strategy, strategy_results in dataset_results["optimization_results"].items():
                for model_name, model_results in strategy_results["models"].items():
                    if "error" in model_results:
                        continue
                        
                    strategies.append(strategy)
                    models.append(model_name)
                    training_times.append(model_results["training_time_seconds"])
                    memory_usages.append(model_results["peak_memory_mb"])
                    training_throughputs.append(model_results["training_throughput_samples_per_second"])
            
            if not models:  # Skip plotting if no successful models
                logger.warning(f"No successful models for dataset {dataset_name}, skipping plots")
                return
                
            # Create a DataFrame for easier plotting
            df = pd.DataFrame({
                "Strategy": strategies,
                "Model": models,
                "Training Time (s)": training_times,
                "Memory Usage (MB)": memory_usages,
                "Training Throughput (samples/s)": training_throughputs
            })
            
            # Set plot style
            sns.set(style="whitegrid")
            plt.figure(figsize=(12, 6))
            
            # 1. Training Time Plot
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x="Model", y="Training Time (s)", hue="Strategy", data=df)
            plt.title(f"Training Time by Model and Strategy - {dataset_name}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/{dataset_id}_training_time.png", dpi=300)
            plt.close()
            
            # 2. Memory Usage Plot
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x="Model", y="Memory Usage (MB)", hue="Strategy", data=df)
            plt.title(f"Peak Memory Usage by Model and Strategy - {dataset_name}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/{dataset_id}_memory_usage.png", dpi=300)
            plt.close()
            
            # 3. Throughput Plot
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x="Model", y="Training Throughput (samples/s)", hue="Strategy", data=df)
            plt.title(f"Training Throughput by Model and Strategy - {dataset_name}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/{dataset_id}_throughput.png", dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating plots for dataset {dataset_id}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")

    def cleanup(self) -> None:
        """Release resources and clear caches."""
        logger.info("Cleaning up resources...")
        
        # Stop resource monitor if running
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop()
            
        # Clear dataset cache
        if hasattr(self, 'dataset_manager'):
            self.dataset_manager.clear_cache()
            
        # Clear model cache
        if hasattr(self, 'model_manager') and hasattr(self.model_manager, 'cache'):
            self.model_manager.cache.clear()
            
        # Clear config cache
        if hasattr(self, 'engine_config_cache'):
            self.engine_config_cache.clear()
            
        # Clear LRU caches
        self._create_engine_config.cache_clear()
        self.model_manager.get_models_for_task.cache_clear() if hasattr(self.model_manager, 'get_models_for_task') else None
        self.dataset_manager.get_openml_datasets.cache_clear() if hasattr(self.dataset_manager, 'get_openml_datasets') else None
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Cleanup complete")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark ML Training Engine")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs",
        help="Directory to save configuration files",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Maximum number of samples to use from each dataset",
    )
    parser.add_argument(
        "--dataset-ids",
        type=str,
        nargs="+",
        help="Specific dataset IDs to benchmark (default: all datasets)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each dataset benchmark",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the benchmark script."""
    args = parse_arguments()
    
    # Configure logging based on arguments
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
    logger.info("Starting ML Training Engine Benchmark")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Config path: {args.config_path}")
    logger.info(f"Sample size: {args.sample_size}")
    
    # Initialize benchmark
    benchmark = MLBenchmark(
        output_dir=args.output_dir,
        config_path=args.config_path,
        sample_size=args.sample_size
    )
    
    try:
        # Get datasets
        dataset_manager = DatasetManager(sample_size=args.sample_size)
        all_datasets = dataset_manager.get_openml_datasets()
        
        # Filter datasets if specific IDs are provided
        if args.dataset_ids:
            datasets = [d for d in all_datasets if d["id"] in args.dataset_ids]
            logger.info(f"Running benchmark on {len(datasets)} specified datasets")
        else:
            datasets = all_datasets
            logger.info(f"Running benchmark on all {len(datasets)} datasets")
        
        # Run benchmarks
        results = benchmark.run_benchmarks(
            datasets=datasets,
            timeout_sec=args.timeout
        )
        
        # Save final results
        results_file = benchmark._save_results()
        report_file = benchmark._generate_report()
        
        logger.info(f"Benchmark completed successfully")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
    finally:
        # Clean up resources
        benchmark.cleanup()


if __name__ == "__main__":
    main()