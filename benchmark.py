#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import logging
import traceback
import psutil
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import the benchmark tools 
def get_system_info() -> Dict[str, Any]:
    """
    Collect comprehensive system information.
    
    Returns:
        Dictionary containing system details
    """
    system_info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "processor": {
            "brand": "Unknown",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True)
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }
    }
    return system_info

def save_benchmark_data(
    benchmark_result: Dict[str, Any], 
    output_path: str, 
    model_name: str
) -> str:
    """
    Save benchmark results to a JSON file.
    
    Args:
        benchmark_result: Dictionary containing benchmark data
        output_path: Path to save the JSON file
        model_name: Name of the model being benchmarked
        
    Returns:
        Path to the saved file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the benchmark result
        with open(output_path, 'w') as f:
            json.dump(benchmark_result, f, indent=2)
        
        return output_path
    except Exception as e:
        print(f"Error saving benchmark data for {model_name}: {e}")
        return ""

class MLTrainingEngineConfig:
    """
    Configuration class for MLTrainingEngine.
    """
    def __init__(
        self, 
        task_type: str, 
        random_state: int = 42,
        n_jobs: int = -1,
        optimization_strategy: str = "random_search",
        optimization_iterations: int = 25,
        early_stopping: bool = True,
        preprocessing_config: Optional[Dict] = None,
        batch_processing_config: Optional[Dict] = None,
        inference_config: Optional[Dict] = None,
        quantization_config: Optional[Dict] = None,
        use_intel_optimization: bool = False,
        memory_optimization: bool = True
    ):
        self.task_type = task_type
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.optimization_strategy = optimization_strategy
        self.optimization_iterations = optimization_iterations
        self.early_stopping = early_stopping
        self.preprocessing_config = preprocessing_config or {}
        self.batch_processing_config = batch_processing_config or {}
        self.inference_config = inference_config or {}
        self.quantization_config = quantization_config or {}
        self.use_intel_optimization = use_intel_optimization
        self.memory_optimization = memory_optimization

class MLTrainingEngine:
    """
    Mock implementation of a training engine for benchmarking.
    """
    def __init__(self, config: MLTrainingEngineConfig):
        self.config = config
    
    def train_model(
        self, 
        model, 
        model_name: str, 
        param_grid: Dict[str, Any], 
        X, 
        y, 
        X_test=None, 
        y_test=None
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Mock training method for benchmarking.
        
        Args:
            model: Scikit-learn model to train
            model_name: Name of the model
            param_grid: Hyperparameter grid
            X: Training features
            y: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tuple of (best_model, metrics)
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
        # Determine metrics based on task type
        if "regression" in model_name.lower():
            scorer = 'neg_mean_squared_error'
            metric_func = mean_squared_error
            metric_name = 'mse'
            secondary_metric_func = r2_score
            secondary_metric_name = 'r2'
        else:
            scorer = 'accuracy'
            metric_func = accuracy_score
            metric_name = 'accuracy'
            secondary_metric_func = None
            secondary_metric_name = None
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=param_grid, 
            cv=3, 
            scoring=scorer, 
            n_jobs=self.config.n_jobs
        )
        
        # Fit the model
        grid_search.fit(X, y)
        
        # Get best model and metrics
        best_model = grid_search.best_estimator_
        
        # Compute metrics
        metrics = {
            metric_name: grid_search.best_score_
        }
        
        # Add secondary metric for regression if applicable
        if X_test is not None and y_test is not None:
            y_pred = best_model.predict(X_test)
            
            if secondary_metric_func:
                secondary_metric = secondary_metric_func(y_test, y_pred)
                metrics[secondary_metric_name] = secondary_metric
        
        return best_model, metrics
    
    def predict(self, X):
        """
        Dummy predict method for benchmarking.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        # This is a placeholder. In a real implementation, 
        # this would use the trained model to make predictions
        return X  # Dummy implementation
    
    def shutdown(self):
        """
        Cleanup method for the training engine.
        """
        # Placeholder for any necessary cleanup
        pass

# Enum-like classes for type hinting
class TaskType:
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"

class OptimizationStrategy:
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"
    ASHT = "asht"

class NormalizationType:
    STANDARD = "standard"
    MINMAX = "minmax"

class QuantizationMode:
    DYNAMIC_PER_BATCH = "dynamic_per_batch"

def create_training_engine_config(
    task_type: str = "classification", 
    optimization_strategy: str = "bayesian_optimization",
    n_jobs: int = -1,
    memory_optimization: bool = True,
    use_intel_optimization: bool = True,
    random_state: int = 42
) -> MLTrainingEngineConfig:
    """
    Create a configuration for the MLTrainingEngine.
    """
    preprocessing_config = {
        "normalization": NormalizationType.STANDARD,
        "handle_nan": True,
        "handle_inf": True,
        "detect_outliers": True,
        "parallel_processing": True,
        "n_jobs": n_jobs,
        "cache_enabled": True
    }
    
    batch_processing_config = {
        "initial_batch_size": 100,
        "min_batch_size": 16,
        "max_batch_size": 256,
        "max_queue_size": 1000,
        "enable_priority_queue": True,
        "enable_monitoring": True,
        "enable_memory_optimization": memory_optimization
    }
    
    inference_config = {
        "enable_intel_optimization": use_intel_optimization,
        "enable_batching": True,
        "batch_processing_strategy": "adaptive",
        "max_concurrent_requests": 16,
        "enable_memory_optimization": memory_optimization
    }
    
    quantization_config = {
        "quantization_type": QuantizationMode.DYNAMIC_PER_BATCH,
        "quantization_mode": QuantizationMode.DYNAMIC_PER_BATCH,
        "enable_cache": True,
        "cache_size": 256,
        "optimize_memory": memory_optimization
    }
    
    return MLTrainingEngineConfig(
        task_type=task_type,
        random_state=random_state,
        n_jobs=n_jobs,
        optimization_strategy=optimization_strategy,
        optimization_iterations=25,
        early_stopping=True,
        preprocessing_config=preprocessing_config,
        batch_processing_config=batch_processing_config,
        inference_config=inference_config,
        quantization_config=quantization_config,
        use_intel_optimization=use_intel_optimization,
        memory_optimization=memory_optimization
    )

def get_model_param_grid(model_name: str) -> Dict[str, Any]:
    """
    Get hyperparameter grid for different models.
    """
    param_grids = {
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "gradient_boosting": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        },
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "saga"]
        },
        "random_forest_regressor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "gradient_boosting_regressor": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        },
        "linear_regression": {},
        "ridge": {
            "alpha": [0.1, 1.0, 10.0]
        }
    }
    
    return param_grids.get(model_name, {"random_state": [42]})

def get_model_class(model_name: str):
    """
    Get model class based on name.
    """
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor
    )
    from sklearn.linear_model import (
        LogisticRegression, LinearRegression, Ridge
    )
    from sklearn.svm import SVC

    model_classes = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
        "svm": SVC,
        "random_forest_regressor": RandomForestRegressor,
        "gradient_boosting_regressor": GradientBoostingRegressor,
        "linear_regression": LinearRegression,
        "ridge": Ridge
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model_classes[model_name]

def get_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Get dataset by name.
    """
    from sklearn import datasets
    
    datasets_mapping = {
        "iris": (datasets.load_iris(), "classification"),
        "digits": (datasets.load_digits(), "classification"),
        "wine": (datasets.load_wine(), "classification"),
        "breast_cancer": (datasets.load_breast_cancer(), "classification"),
        "diabetes": (datasets.load_diabetes(), "regression"),
        "california_housing": (datasets.fetch_california_housing(), "regression")
    }
    
    if dataset_name not in datasets_mapping:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    dataset, task_type = datasets_mapping[dataset_name]
    return dataset.data, dataset.target, task_type

def benchmark_training_engine(
    model_name: str,
    dataset_name: str,
    output_dir: str = "benchmark_results",
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
    memory_optimization: bool = True,
    use_intel_optimization: bool = True,
    optimization_strategy: str = "bayesian_optimization"
) -> Dict[str, Any]:
    """
    Benchmark the MLTrainingEngine with a specific model and dataset.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Benchmarking {model_name} on {dataset_name} dataset...")
    
    # Load dataset
    try:
        X, y, task_type = get_dataset(dataset_name)
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {"error": str(e)}
    
    # Get model class
    try:
        model_class = get_model_class(model_name)
        print(f"Model class: {model_class.__name__}")
    except Exception as e:
        print(f"Error getting model class: {e}")
        return {"error": str(e)}
    
    # Check if model is compatible with task type
    is_regression_model = "regressor" in model_name.lower()
    is_regression_task = task_type == "regression"
    
    if (is_regression_model and not is_regression_task) or (not is_regression_model and is_regression_task):
        print(f"Model {model_name} is not compatible with {task_type} task")
        return {"error": f"Model {model_name} is not compatible with {task_type} task"}
    
    # Import train_test_split
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create training engine config
    config = create_training_engine_config(
        task_type=task_type,
        optimization_strategy=optimization_strategy,
        n_jobs=n_jobs,
        memory_optimization=memory_optimization,
        use_intel_optimization=use_intel_optimization,
        random_state=random_state
    )
    
    # Initialize the training engine
    print("Initializing training engine...")
    engine_start_time = time.time()
    training_engine = MLTrainingEngine(config)
    engine_init_time = time.time() - engine_start_time
    print(f"Training engine initialized in {engine_init_time:.2f} seconds")
    
    # Get parameter grid for the model
    param_grid = get_model_param_grid(model_name)
    
    # Start training
    print("Starting model training...")
    start_time = time.time()
    
    try:
        # Train the model
        model = model_class()
        best_model, metrics = training_engine.train_model(
            model=model,
            model_name=model_name,
            param_grid=param_grid,
            X=X_train,
            y=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        training_time = time.time() - start_time
        print(f"Model trained in {training_time:.2f} seconds")
        
        # Get peak memory usage
        import tracemalloc
        tracemalloc.start()
        
        # Benchmark inference
        print("Benchmarking inference...")
        inference_start = time.time()
        
        # Measure inference time using batch processing
        predictions = training_engine.predict(X_test)
        
        inference_time = time.time() - inference_start
        print(f"Inference completed in {inference_time:.4f} seconds")
        
        # Get memory usage
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        peak_memory_mb = peak_mem / (1024 * 1024)
        tracemalloc.stop()
        
        print(f"Peak memory usage: {peak_memory_mb:.2f} MB")
        
        # Calculate throughput
        training_throughput = X_train.shape[0] / training_time if training_time > 0 else 0
        inference_throughput = X_test.shape[0] / inference_time if inference_time > 0 else 0
        
        # Prepare benchmark results
        benchmark_result = {
            "results": {
                dataset_name: {
                    "dataset_info": {
                        "name": dataset_name,
                        "id": dataset_name,
                        "task": task_type,
                        "n_samples": X.shape[0],
                        "n_features": X.shape[1]
                    },
                    model_name: {
                        "training_time_seconds": training_time,
                        "peak_memory_mb": peak_memory_mb,
                        "training_throughput_samples_per_second": training_throughput,
                        "inference_time_seconds": inference_time,
                        "inference_throughput_samples_per_second": inference_throughput,
                        "engine_initialization_time_seconds": engine_init_time,
                        "optimization_strategy": optimization_strategy,
                        "metrics": metrics
                    }
                }
            },
            "config": {
                "task_type": task_type,
                "optimization_strategy": optimization_strategy,
                "n_jobs": n_jobs,
                "memory_optimization": memory_optimization,
                "use_intel_optimization": use_intel_optimization,
                "random_state": random_state,
                "test_size": test_size
            }
        }
        
        # Save benchmark results
        result_path = save_benchmark_data(
            benchmark_result,
            output_path=os.path.join(output_dir, f"{dataset_name}_{model_name}_training_engine.json"),
            model_name=model_name
        )
        
        print(f"Benchmark results saved to: {result_path}")
        
        # Clean up - shutdown the engine to release resources
        training_engine.shutdown()
        
        return benchmark_result
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up
        try:
            training_engine.shutdown()
        except:
            pass
        
        return {"error": str(e)}

def run_training_engine_benchmarks(
    models_to_benchmark: Optional[List[str]] = None,
    datasets_to_benchmark: Optional[List[str]] = None,
    output_dir: str = "training_engine_benchmarks",
    n_jobs: int = -1,
    optimization_strategy: str = "random_search",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, str]:
    """
    Run benchmarks for the MLTrainingEngine with multiple models and datasets.
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"benchmark_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Default model selection if not specified
    if models_to_benchmark is None:
        models_to_benchmark = [
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
            "random_forest_regressor",
            "gradient_boosting_regressor",
            "linear_regression"
        ]
    
    # Default dataset selection if not specified
    if datasets_to_benchmark is None:
        datasets_to_benchmark = [
            "iris",
            "breast_cancer",
            "wine",
            "digits",
            "diabetes",
            "california_housing"
        ]
    
    # Save system info
    system_info = get_system_info()
    with open(os.path.join(results_dir, "system_info.json"), "w") as f:
        json.dump(system_info, f, indent=2)
    
    # Track benchmark results
    benchmark_results = {}
    
    # Run benchmarks
    for dataset_name in datasets_to_benchmark:
        # Get dataset and task type
        try:
            X, y, task_type = get_dataset(dataset_name)
            print(f"\nDataset: {dataset_name}, Task: {task_type}, "
                  f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue
        
        for model_name in models_to_benchmark:
            # Check if model is compatible with task type
            is_regression_model = "regressor" in model_name.lower()
            is_regression_task = task_type == "regression"
            
            if (is_regression_model and not is_regression_task) or (not is_regression_model and is_regression_task):
                print(f"Skipping {model_name} (incompatible with {task_type} task)")
                continue
            
            print(f"\n{'-'*80}")
            print(f"Benchmarking {model_name} on {dataset_name}...")
            
            try:
                result = benchmark_training_engine(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    output_dir=results_dir,
                    test_size=test_size,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    optimization_strategy=optimization_strategy
                )
                
                # Store benchmark result
                benchmark_name = f"{dataset_name}_{model_name}"
                benchmark_results[benchmark_name] = os.path.join(
                    results_dir, f"{dataset_name}_{model_name}_training_engine.json"
                )
                
                # Generate a summary row for this benchmark
                if "results" in result:
                    model_results = result["results"][dataset_name][model_name]
                    print(f"  Training time: {model_results['training_time_seconds']:.2f}s")
                    print(f"  Inference time: {model_results['inference_time_seconds']:.4f}s")
                    print(f"  Peak memory: {model_results['peak_memory_mb']:.2f} MB")
                    print(f"  Training throughput: {model_results['training_throughput_samples_per_second']:.2f} samples/s")
                    print(f"  Inference throughput: {model_results['inference_throughput_samples_per_second']:.2f} samples/s")
                
            except Exception as e:
                print(f"Error benchmarking {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Generate summary report
    try:
        generate_benchmark_report(benchmark_results, results_dir, system_info)
    except Exception as e:
        print(f"Error generating summary report: {e}")
    
    print(f"\nAll benchmarks completed. Results saved to: {results_dir}")
    return benchmark_results

def generate_benchmark_report(
    benchmark_results: Dict[str, str],
    output_dir: str,
    system_info: Dict[str, Any]
) -> str:
    """
    Generate a summary report of benchmark results.
    """
    # Collect benchmark data
    data = []
    
    for benchmark_name, file_path in benchmark_results.items():
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                
            # Parse the benchmark name to get dataset and model
            parts = benchmark_name.split('_')
            dataset_name = parts[0]
            model_name = '_'.join(parts[1:])
            
            # Get the benchmark metrics
            dataset_results = result["results"][dataset_name]
            model_results = dataset_results[model_name]
            
            # Create a data row
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "task": dataset_results["dataset_info"]["task"],
                "samples": dataset_results["dataset_info"]["n_samples"],
                "features": dataset_results["dataset_info"]["n_features"],
                "training_time": model_results["training_time_seconds"],
                "inference_time": model_results["inference_time_seconds"],
                "peak_memory_mb": model_results["peak_memory_mb"],
                "training_throughput": model_results["training_throughput_samples_per_second"],
                "inference_throughput": model_results["inference_throughput_samples_per_second"],
                "engine_init_time": model_results.get("engine_initialization_time_seconds", 0),
                "optimization_strategy": model_results.get("optimization_strategy", "unknown")
            }
            
            # Add performance metrics if available
            metrics = model_results.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                row[f"metric_{metric_name}"] = metric_value
            
            data.append(row)
        except Exception as e:
            print(f"Error processing benchmark result {benchmark_name}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate Markdown report
    report_path = os.path.join(output_dir, "benchmark_report.md")
    
    with open(report_path, "w") as f:
        f.write("# MLTrainingEngine Benchmark Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System information
        f.write("## System Information\n\n")
        f.write(f"- **Platform:** {system_info['platform']}\n")
        f.write(f"- **Processor Cores:** Physical: {system_info['processor']['physical_cores']}, Logical: {system_info['processor']['logical_cores']}\n")
        f.write(f"- **Memory:** {system_info['memory']['total_gb']} GB\n")
        f.write(f"- **Python Version:** {system_info['python_version']}\n\n")
        
        # Benchmark statistics
        f.write("## Benchmark Statistics\n\n")
        f.write(f"- **Total benchmarks:** {len(data)}\n")
        f.write(f"- **Models tested:** {df['model'].nunique()}\n")
        f.write(f"- **Datasets tested:** {df['dataset'].nunique()}\n\n")
        
        # Top 5 fastest models for training
        if not df.empty:
            f.write("## Top 5 Fastest Models (Training)\n\n")
            top_training = df.sort_values('training_time').head(5)
            f.write("| Model | Dataset | Training Time (s) | Memory (MB) | Throughput (samples/s) |\n")
            f.write("|-------|---------|------------------|-------------|------------------------|\n")
            for _, row in top_training.iterrows():
                f.write(f"| {row['model']} | {row['dataset']} | {row['training_time']:.2f} | {row['peak_memory_mb']:.2f} | {row['training_throughput']:.2f} |\n")
            f.write("\n")
            
            # Top 5 fastest models for inference
            f.write("## Top 5 Fastest Models (Inference)\n\n")
            top_inference = df.sort_values('inference_time').head(5)
            f.write("| Model | Dataset | Inference Time (s) | Throughput (samples/s) |\n")
            f.write("|-------|---------|-------------------|------------------------|\n")
            for _, row in top_inference.iterrows():
                f.write(f"| {row['model']} | {row['dataset']} | {row['inference_time']:.4f} | {row['inference_throughput']:.2f} |\n")
            f.write("\n")
    
    print(f"Benchmark report generated: {report_path}")
    return report_path

def main():
    """Main function to run training engine benchmarks from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark MLTrainingEngine with various models and datasets")
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="training_engine_benchmarks",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="List of model names to benchmark (default: a curated selection)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="List of dataset names to benchmark (default: a curated selection)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for testing"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)"
    )
    parser.add_argument(
        "--optimization-strategy",
        type=str,
        default="random_search",
        choices=["grid_search", "random_search", "bayesian_optimization", "evolutionary", "hyperband", "asht"],
        help="Hyperparameter optimization strategy"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single benchmark (with first model and dataset)"
    )
    
    args = parser.parse_args()
    
    if args.single:
            # Run a single benchmark with the first model and dataset
            models = args.models or ["random_forest"]
            datasets = args.datasets or ["iris"]
            
            print(f"Running single benchmark: {models[0]} on {datasets[0]}")
            
            result = benchmark_training_engine(
                model_name=models[0],
                dataset_name=datasets[0],
                output_dir=args.output_dir,
                test_size=args.test_size,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                optimization_strategy=args.optimization_strategy
            )
            
            print(f"Single benchmark completed. Result: {result}")
    else:
        # Run multiple benchmarks
        run_training_engine_benchmarks(
            models_to_benchmark=args.models,
            datasets_to_benchmark=args.datasets,
            output_dir=args.output_dir,
            n_jobs=args.n_jobs,
            optimization_strategy=args.optimization_strategy,
            test_size=args.test_size,
        random_state=args.random_state
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error in benchmark: {e}")
        traceback.print_exc()
        sys.exit(1)