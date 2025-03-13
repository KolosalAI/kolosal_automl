#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import time
import json
import tempfile
import shutil
import logging
import signal
import sys
import psutil
import atexit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from pmlb import fetch_data, classification_dataset_names
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from functools import partial
import argparse
import warnings
import joblib
import re
import gc

from modules.configs import (PreprocessorConfig, 
                             NormalizationType, 
                             InferenceEngineConfig,
                             BatchProcessorConfig,
                             QuantizationConfig,
                             TaskType,
                             OptimizationStrategy,
                             MLTrainingEngineConfig
)
from modules.engine.train_engine import MLTrainingEngine
# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Filter out common warnings that don't affect benchmark functionality
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*resource_tracker.*FileNotFoundError.*", module="joblib.externals.loky.backend.resource_tracker")
warnings.filterwarnings("ignore", message=".*joblib_memmapping_folder.*", module="joblib.externals.loky.backend.resource_tracker")
warnings.filterwarnings("ignore", message="resource_tracker", module="joblib.externals.loky.backend.resource_tracker")
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*", module="sklearn")

# Patch joblib to avoid resource tracker warnings
try:
    from joblib.externals.loky.backend.resource_tracker import _resource_tracker
    # Replace the warning function with a no-op function
    if hasattr(_resource_tracker, 'warn_and_log'):
        original_warn = _resource_tracker.warn_and_log
        def silent_warn(*args, **kwargs):
            pass
        _resource_tracker.warn_and_log = silent_warn
except:
    pass

class BenchmarkEngine:
    """Engine for benchmarking MLTrainingEngine performance across various datasets"""
    
    def __init__(self, output_dir="./benchmark_results", n_datasets=10, seed=42,
                 optimization_iterations=20, n_jobs=-1, test_size=0.2, 
                 use_memory_limit=True, memory_limit_mb=None):
        """
        Initialize the benchmark engine
        
        Args:
            output_dir: Directory to save benchmark results
            n_datasets: Number of datasets to benchmark on
            seed: Random seed for reproducibility
            optimization_iterations: Number of optimization iterations to use
            n_jobs: Number of parallel jobs
            test_size: Test set size for dataset splitting
            use_memory_limit: Whether to limit memory usage for joblib
            memory_limit_mb: Memory limit in MB for joblib (None = auto)
        """
        self.output_dir = output_dir
        self.n_datasets = n_datasets
        self.seed = seed
        self.optimization_iterations = optimization_iterations
        self.n_jobs = n_jobs
        self.test_size = test_size
        self.use_memory_limit = use_memory_limit
        self.memory_limit_mb = memory_limit_mb
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure joblib to use a memory limit if specified
        if self.use_memory_limit:
            # Auto-determine memory limit if not specified
            if self.memory_limit_mb is None:
                import psutil
                total_memory = psutil.virtual_memory().total / (1024 * 1024)  # Convert to MB
                self.memory_limit_mb = int(total_memory * 0.8)  # Use 80% of total memory
                
            # Set joblib memory limit
            # Clean existing temp files with correct API
            try:
                # In newer joblib versions
                if hasattr(joblib, 'disk'):
                    joblib.disk.delete_temporary_files()
                # Fallback for older versions
                elif hasattr(joblib.pool, 'MemmappingPool'):
                    joblib.pool.MemmappingPool.clear_temporary_files()
            except Exception as e:
                print(f"Warning: Could not clean temporary files: {e}")
                
            # Configure to use a specific temp directory we can clean up later
            try:
                self.temp_folder = tempfile.mkdtemp(prefix="benchmark_joblib_")
                print(f"Created temporary directory: {self.temp_folder}")
                joblib.Memory(location=self.temp_folder, verbose=0)
                
                # Set joblib environment variables
                os.environ["JOBLIB_TEMP_FOLDER"] = self.temp_folder
                os.environ["JOBLIB_MULTIPROCESSING"] = "0"  # Force using loky backend with spawn
                
                # Attempt to monkey patch joblib's resource tracker to avoid warnings
                try:
                    from joblib.externals.loky.backend import resource_tracker
                    if hasattr(resource_tracker, '_ResourceTracker'):
                        # Create a subclass that overrides the warning method
                        class SilentResourceTracker(resource_tracker._ResourceTracker):
                            def _warn(self, *args, **kwargs):
                                pass  # Do nothing instead of warning
                        
                        # Replace the resource tracker instance
                        if hasattr(resource_tracker, '_resource_tracker'):
                            original_tracker = resource_tracker._resource_tracker
                            # Create new instance with same state but silent warnings
                            silent_tracker = SilentResourceTracker()
                            for attr in dir(original_tracker):
                                if not attr.startswith('_') and not callable(getattr(original_tracker, attr)):
                                    setattr(silent_tracker, attr, getattr(original_tracker, attr))
                            resource_tracker._resource_tracker = silent_tracker
                except:
                    # If patching fails, just continue without it
                    pass
            except Exception as e:
                print(f"Warning: Could not set up joblib memory: {e}")
        
        # Set up default models and their parameter grids
        self.models = {
            "logistic_regression": {
                "model": LogisticRegression(random_state=seed, max_iter=3000),  # Increased max_iter
                "param_grid": {
                    "model__C": [0.01, 0.1, 1.0, 10.0],
                    "model__solver": ["liblinear", "saga"]
                }
            },
            "random_forest": {
                "model": RandomForestClassifier(random_state=seed),
                "param_grid": {
                    "model__n_estimators": [50, 100],
                    "model__max_depth": [None, 10, 20],
                    "model__min_samples_split": [2, 5]
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier(random_state=seed),
                "param_grid": {
                    "model__n_estimators": [50, 100],
                    "model__learning_rate": [0.01, 0.1],
                    "model__max_depth": [3, 5]
                }
            }
        }
        
    def _benchmark_dataset(self, dataset_name, strategies, max_samples=5000):
        """Benchmark various optimization strategies on a single dataset with detailed timing measurements"""
        try:
            # Fetch the dataset
            try:
                X, y = fetch_data(dataset_name, return_X_y=True, local_cache_dir=tempfile.gettempdir())
            except Exception as e:
                print(f"Error fetching dataset {dataset_name}: {e}")
                return {
                    "dataset": dataset_name,
                    "error": f"Data fetch error: {str(e)}"
                }
            
            # Limit dataset size for speed if needed
            if X.shape[0] > max_samples:
                try:
                    indices = np.random.choice(X.shape[0], max_samples, replace=False)
                    X, y = X[indices], y[indices]
                except Exception as e:
                    print(f"Error sampling dataset {dataset_name}: {e}")
                    # Continue with full dataset if sampling fails
            
            # Basic validation - make sure we have valid data
            if X.shape[0] == 0 or len(np.unique(y)) < 2:
                return {
                    "dataset": dataset_name,
                    "error": "Dataset has no samples or less than 2 classes"
                }
            
            # Split data into train and test sets
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=self.seed, stratify=y
                )
            except Exception as e:
                print(f"Error splitting dataset {dataset_name}: {e}")
                # Try without stratification if that fails
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=self.test_size, random_state=self.seed
                    )
                except Exception as e2:
                    return {
                        "dataset": dataset_name,
                        "error": f"Data split error: {str(e2)}"
                    }
            
            results = {}
            
            # Benchmark each optimization strategy
            for strategy_name, strategy in strategies.items():
                strategy_results = {}
                
                # Create a temporary directory for this run
                temp_dir = tempfile.mkdtemp()
                
                try:
                    # Create engine config with current strategy
                    engine_config = self._create_engine_config(strategy)
                    engine_config.model_path = temp_dir
                    
                    # Initialize training engine
                    engine = MLTrainingEngine(engine_config)
                    
                    # Benchmark each model
                    for model_name, model_info in self.models.items():
                        model_start = time.time()
                        
                        try:
                            # Track detailed timings for each phase
                            phase_times = {}
                            
                            # Training phase timer
                            train_start = time.time()
                            best_model, metrics = engine.train_model(
                                model=model_info["model"],
                                model_name=model_name,
                                param_grid=model_info["param_grid"],
                                X=X_train,
                                y=y_train,
                                X_test=X_test,
                                y_test=y_test
                            )
                            phase_times["train"] = time.time() - train_start
                            
                            # Inference speed evaluation with multiple runs for better accuracy
                            inference_times = []
                            num_repeats = 5  # Number of inference runs to average
                            for _ in range(num_repeats):
                                inference_start = time.time()
                                y_pred = engine.predict(X_test, model_name=model_name)
                                inference_times.append(time.time() - inference_start)
                            
                            # Calculate average inference time and speed
                            inference_time = np.mean(inference_times)
                            phase_times["inference"] = inference_time
                            inference_speed = X_test.shape[0] / inference_time if inference_time > 0 else 0
                            
                            # Calculate prediction accuracy
                            accuracy = np.mean(y_pred == y_test) if y_pred is not None else 0
                            
                            # Add inference metrics to the overall metrics
                            metrics["inference_time"] = inference_time
                            metrics["inference_speed_samples_per_sec"] = inference_speed
                            metrics["inference_accuracy"] = accuracy
                            
                            model_time = time.time() - model_start
                            
                            # Store results
                            strategy_results[model_name] = {
                                "metrics": metrics,
                                "time": model_time,
                                "phase_times": phase_times,
                                "inference_speed": inference_speed
                            }
                            
                        except Exception as e:
                            print(f"Error on dataset {dataset_name}, strategy {strategy_name}, model {model_name}: {str(e)}")
                            strategy_results[model_name] = {
                                "error": str(e)
                            }
                    
                    # Shutdown engine and clean up
                    engine.shutdown()
                    
                finally:
                    # Clean up temporary directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
                
                results[strategy_name] = strategy_results
            
            # Convert class_balance to ensure all dictionary keys are Python types
            return {
                "dataset": dataset_name,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_classes": int(len(np.unique(y))),
                "class_balance": {int(k): int(v) for k, v in dict(zip(*np.unique(y, return_counts=True))).items()},
                "results": results
            }
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            return {
                "dataset": dataset_name,
                "error": str(e)
            }
    
    def _create_engine_config(self, strategy):
        """Create MLTrainingEngine configuration with specified optimization strategy"""
        # Create preprocessor config
        preprocessor_config = PreprocessorConfig(
            normalization=NormalizationType.STANDARD,
            handle_nan=True,
            handle_inf=True,
            detect_outliers=True,
            parallel_processing=True,
            n_jobs=self.n_jobs
        )
        
        # Create batch processor config
        batch_config = BatchProcessorConfig(
            initial_batch_size=32,
            min_batch_size=16,
            max_batch_size=64,
            max_queue_size=100,
            enable_monitoring=False,
            debug_mode=False
        )
        
        # Create inference engine config
        inference_config = InferenceEngineConfig(
            enable_intel_optimization=True,
            enable_batching=True,
            enable_quantization=False,
            debug_mode=False
        )
        
        # Create quantization config
        quantization_config = QuantizationConfig()
        
        # Create main engine config
        engine_config = MLTrainingEngineConfig(
            task_type=TaskType.CLASSIFICATION,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            verbose=0,
            cv_folds=3,
            test_size=self.test_size,
            stratify=True,
            optimization_strategy=strategy,
            optimization_iterations=self.optimization_iterations,
            early_stopping=True,
            feature_selection=True,
            feature_selection_method="mutual_info",
            preprocessing_config=preprocessor_config,
            batch_processing_config=batch_config,
            inference_config=inference_config,
            quantization_config=quantization_config,
            model_path=tempfile.mkdtemp(),
            experiment_tracking=False,
            memory_optimization=True,
            log_level="ERROR"
        )
        
        return engine_config
    
    def run_benchmark(self, n_processes=None):
        """Run the benchmark on multiple datasets using process parallelism with detailed timing metrics"""
        # Register cleanup function to run at exit
        atexit.register(self._cleanup_resources)
        
        # Define optimization strategies to compare
        strategies = {
            "grid_search": OptimizationStrategy.GRID_SEARCH,
            "random_search": OptimizationStrategy.RANDOM_SEARCH,
            "bayesian": OptimizationStrategy.BAYESIAN_OPTIMIZATION,
            "asht": OptimizationStrategy.ASHT
        }
        
        # Select datasets to benchmark
        np.random.seed(self.seed)
        if self.n_datasets < len(classification_dataset_names):
            datasets = np.random.choice(classification_dataset_names, self.n_datasets, replace=False)
        else:
            datasets = classification_dataset_names
        
        print(f"Starting benchmark on {len(datasets)} datasets with {len(strategies)} optimization strategies")
        
        # Initialize results storage and attach it to self for recovery during interruption
        self.partial_results = []
        benchmark_results = self.partial_results
        
        # Determine number of processes
        if n_processes is None:
            n_processes = min(os.cpu_count(), len(datasets))
            
        # Ensure we don't use more processes than is reasonable for available memory
        if self.use_memory_limit:
            try:
                mem_per_process_mb = self.memory_limit_mb / n_processes
                if mem_per_process_mb < 512:  # If less than 512MB per process
                    n_processes_by_mem = max(1, int(self.memory_limit_mb / 512))
                    n_processes = min(n_processes, n_processes_by_mem)
                    print(f"Limiting to {n_processes} processes due to memory constraints")
            except Exception as e:
                print(f"Warning: Could not calculate memory-based process limit: {e}")
                # Fall back to a conservative default
                n_processes = min(n_processes, 4)
        
        # Create a partial function with fixed strategies
        benchmark_fn = partial(self._benchmark_dataset, strategies=strategies)
        
        # Set up signal handlers for the main process
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Run benchmark in parallel with proper interrupt handling
        try:
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                # Restore the signal handler
                signal.signal(signal.SIGINT, original_sigint_handler)
                
                # Submit all tasks
                futures = {executor.submit(benchmark_fn, dataset): dataset for dataset in datasets}
                completed_count = 0
                
                try:
                    for i, future in enumerate(as_completed(futures)):
                        dataset = futures[future]
                        try:
                            result = future.result()
                            benchmark_results.append(result)
                            completed_count += 1
                            print(f"Completed {completed_count}/{len(datasets)}: {dataset}")
                        except Exception as e:
                            print(f"Error on dataset {dataset}: {str(e)}")
                except KeyboardInterrupt:
                    print("\nReceived KeyboardInterrupt. Cancelling pending tasks...")
                    # Cancel pending tasks
                    for future in futures:
                        if not future.done():
                            future.cancel()
                    # Wait briefly for cancellation to complete
                    print("Waiting for running tasks to complete (max 5 seconds)...")
                    try:
                        # Use a short timeout to avoid hanging
                        _, not_done = wait(futures, timeout=5)
                        if not_done:
                            print(f"{len(not_done)} tasks did not complete in time.")
                    except:
                        pass
                    raise  # Re-raise to handle in outer try-block
        except KeyboardInterrupt:
            print("Benchmark interrupted. Saving partial results...")
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # Clean up joblib resources explicitly
            try:
                if hasattr(joblib, 'disk'):
                    joblib.disk.delete_temporary_files()
                elif hasattr(joblib.pool, 'MemmappingPool'):
                    joblib.pool.MemmappingPool.clear_temporary_files()
            except:
                pass
                
            if hasattr(joblib.parallel, "_mp_context"):
                try:
                    joblib.parallel._mp_context = None
                except:
                    pass
            
            # Save results, whether complete or partial
            try:
                status = "completed" if len(benchmark_results) == len(datasets) else "interrupted"
                results_file = os.path.join(self.output_dir, f"benchmark_results_{int(time.time())}.json")
                with open(results_file, 'w') as f:
                    json.dump({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "n_datasets": len(datasets),
                        "datasets_completed": len(benchmark_results),
                        "status": status,
                        "seed": self.seed,
                        "optimization_iterations": self.optimization_iterations,
                        "results": benchmark_results
                    }, f, indent=2, cls=NumpyEncoder)
                
                print(f"Benchmark {status}. Results saved to {results_file}")
            except Exception as e:
                print(f"Error saving benchmark results: {e}")
                # Try to save in a simpler format if regular save fails
                try:
                    simple_results_file = os.path.join(self.output_dir, f"simple_results_{int(time.time())}.json")
                    with open(simple_results_file, 'w') as f:
                        simple_data = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "error_during_save",
                            "datasets_completed": len(benchmark_results),
                            "error": str(e)
                        }
                        json.dump(simple_data, f)
                    print(f"Simplified results saved to {simple_results_file}")
                except:
                    print("Failed to save results in any format.")
        
        return self._analyze_results(benchmark_results)
    
    def _cleanup_resources(self):
        """Clean up any temporary resources created during benchmarking"""
        # Clean up joblib resources
        try:
            if hasattr(joblib, 'disk'):
                joblib.disk.delete_temporary_files()
            elif hasattr(joblib.pool, 'MemmappingPool'):
                joblib.pool.MemmappingPool.clear_temporary_files()
        except Exception as e:
            print(f"Warning: Failed to clean joblib temporary files: {e}")
            
        # Remove temp directory if it exists
        if hasattr(self, 'temp_folder'):
            try:
                # When cleaning up temp folders
                if os.path.exists(self.temp_folder):
                    shutil.rmtree(self.temp_folder, ignore_errors=True)
                    print(f"Removed temporary directory: {self.temp_folder}")
                    
                    # Also remove any memmapping folders that might be leaking
                    # This is where joblib's warnings are coming from
                    folder_pattern = os.path.join(self.temp_folder, "joblib_memmapping_folder_*")
                    import glob
                    for folder in glob.glob(folder_pattern):
                        try:
                            shutil.rmtree(folder, ignore_errors=True)
                        except:
                            pass
            except Exception as e:
                print(f"Warning: Could not remove temp folder {self.temp_folder}: {e}")
            
        # Clean up any orphaned processes
        try:
            current_process = psutil.Process()
            for child in current_process.children(recursive=True):
                try:
                    child.terminate()
                except:
                    pass
        except Exception as e:
            print(f"Warning: Error during process cleanup: {e}")
                
        print("Cleanup complete.")
    
    def _handle_numpy_types(self, obj):
        """Convert NumPy types to Python native types recursively"""
        try:
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return self._handle_numpy_types(obj.tolist())
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {self._handle_numpy_types(k): self._handle_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._handle_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(self._handle_numpy_types(item) for item in obj)
            else:
                # For unhandled types, try conversion to string or dict
                try:
                    return str(obj)
                except:
                    return f"<Unserializable object of type {type(obj).__name__}>"
        except Exception as e:
            print(f"Warning: Error handling type conversion: {e}")
            return str(obj)
    
    def _analyze_results(self, results):
        """Analyze benchmark results and compute summary statistics"""
        # First convert any NumPy types to native Python types
        try:
            results = self._handle_numpy_types(results)
            
            # Initialize counters and aggregators
            strategy_wins = {strategy: 0 for strategy in ["grid_search", "random_search", "bayesian", "asht"]}
            model_wins = {model: 0 for model in self.models.keys()}
            strategy_times = {strategy: [] for strategy in ["grid_search", "random_search", "bayesian", "asht"]}
            strategy_train_times = {strategy: [] for strategy in ["grid_search", "random_search", "bayesian", "asht"]}
            strategy_inference_times = {strategy: [] for strategy in ["grid_search", "random_search", "bayesian", "asht"]}
            strategy_inference_speeds = {strategy: [] for strategy in ["grid_search", "random_search", "bayesian", "asht"]}
            strategy_inference_accuracies = {strategy: [] for strategy in ["grid_search", "random_search", "bayesian", "asht"]}
            strategy_scores = {strategy: [] for strategy in ["grid_search", "random_search", "bayesian", "asht"]}
            
            for result in results:
                if "error" in result:
                    continue
                    
                dataset = result["dataset"]
                
                # Process class_balance to ensure all keys are Python types, not NumPy types
                if "class_balance" in result:
                    # Ensure all keys in class_balance are regular Python types, not NumPy types
                    result["class_balance"] = {int(k): int(v) for k, v in result["class_balance"].items()}
                
                # Find best strategy for this dataset
                best_strategy = None
                best_score = -float('inf')
                
                for strategy, models in result["results"].items():
                    # Get best model for this strategy
                    best_model_score = -float('inf')
                    best_model_name = None
                    
                    for model_name, model_result in models.items():
                        if "error" in model_result:
                            continue
                            
                        # Get primary metric (f1 score for classification)
                        if "metrics" in model_result and "f1" in model_result["metrics"]:
                            score = model_result["metrics"]["f1"]
                            strategy_scores[strategy].append(score)
                            
                            if score > best_model_score:
                                best_model_score = score
                                best_model_name = model_name
                                
                            # Track execution times
                            if "time" in model_result:
                                strategy_times[strategy].append(model_result["time"])
                            
                            # Track detailed timing metrics
                            if "phase_times" in model_result:
                                if "train" in model_result["phase_times"]:
                                    strategy_train_times[strategy].append(model_result["phase_times"]["train"])
                                if "inference" in model_result["phase_times"]:
                                    strategy_inference_times[strategy].append(model_result["phase_times"]["inference"])
                            
                            # Track inference speed
                            if "inference_speed" in model_result:
                                strategy_inference_speeds[strategy].append(model_result["inference_speed"])
                                
                            # Track inference accuracy
                            if "metrics" in model_result and "inference_accuracy" in model_result["metrics"]:
                                strategy_inference_accuracies[strategy].append(model_result["metrics"]["inference_accuracy"])
                    
                    # Update best strategy if this one is better
                    if best_model_score > best_score:
                        best_score = best_model_score
                        best_strategy = strategy
                        
                        # Count model win
                        if best_model_name:
                            model_wins[best_model_name] += 1
                
                # Count strategy win
                if best_strategy:
                    strategy_wins[best_strategy] += 1
            
            # Compute summary statistics
            try:
                summary = {
                    "strategy_wins": strategy_wins,
                    "model_wins": model_wins,
                    "avg_times": {strategy: np.mean(times) if times else 0 for strategy, times in strategy_times.items()},
                    "avg_scores": {strategy: np.mean(scores) if scores else 0 for strategy, scores in strategy_scores.items()},
                    "avg_train_times": {strategy: np.mean(times) if times else 0 for strategy, times in strategy_train_times.items()},
                    "avg_inference_times": {strategy: np.mean(times) if times else 0 for strategy, times in strategy_inference_times.items()},
                    "avg_inference_speeds": {strategy: np.mean(speeds) if speeds else 0 for strategy, speeds in strategy_inference_speeds.items()},
                    "avg_inference_accuracies": {strategy: np.mean(accs) if accs else 0 for strategy, accs in strategy_inference_accuracies.items()},
                }
                
                # Save summary to file
                try:
                    summary_file = os.path.join(self.output_dir, f"benchmark_summary_{int(time.time())}.json")
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2, cls=NumpyEncoder)
                    print(f"\nSummary saved to {summary_file}")
                except Exception as e:
                    print(f"Error saving summary: {e}")
                    
                # Print summary
                print("\n=== Benchmark Summary ===")
                print("\nStrategy Wins:")
                for strategy, wins in strategy_wins.items():
                    print(f"  {strategy}: {wins}")
                    
                print("\nModel Wins:")
                for model, wins in model_wins.items():
                    print(f"  {model}: {wins}")
                    
                print("\nAverage Times (seconds):")
                for strategy, avg_time in summary["avg_times"].items():
                    print(f"  {strategy}: {avg_time:.2f}")
                    
                print("\nAverage F1 Scores:")
                for strategy, avg_score in summary["avg_scores"].items():
                    print(f"  {strategy}: {avg_score:.4f}")
                    
                print("\nAverage Training Times (seconds):")
                for strategy, avg_time in summary["avg_train_times"].items():
                    print(f"  {strategy}: {avg_time:.2f}")
                    
                print("\nAverage Inference Times (seconds):")
                for strategy, avg_time in summary["avg_inference_times"].items():
                    print(f"  {strategy}: {avg_time:.4f}")
                    
                print("\nAverage Inference Speeds (samples/second):")
                for strategy, avg_speed in summary["avg_inference_speeds"].items():
                    print(f"  {strategy}: {avg_speed:.2f}")
                    
                print("\nAverage Inference Accuracies:")
                for strategy, avg_accuracy in summary["avg_inference_accuracies"].items():
                    print(f"  {strategy}: {avg_accuracy:.4f}")
                    
                return summary
            except Exception as e:
                print(f"Error generating summary statistics: {e}")
                return {"error": str(e)}
        except Exception as e:
            print(f"Error generating summary statistics: {e}")
            return {"error": str(e)}
    
    def generate_report(self, output_file=None):
        """Generate a comprehensive report of all models"""
        if not self.models:
            self.logger.warning("No models to generate report")
            return None
                
        if output_file is None:
            output_file = os.path.join(self.config.model_path, "model_report.html")
                
        # Create basic report
        report = f"""
        <html>
        <head>
            <title>ML Training Engine Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .best {{ background-color: #dff0d8; }}
                h1, h2, h3 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>ML Training Engine Report</h1>
            <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        # Add configuration
        for key, value in self.config.to_dict().items():
            report += f"<tr><td>{key}</td><td>{value}</td></tr>"
            
        report += """
            </table>
            
            <h2>Model Performance Summary</h2>
            <table>
                <tr><th>Model</th>
        """
        
        # Collect all metrics across models
        all_metrics = set()
        for model_data in self.models.values():
            all_metrics.update(model_data["metrics"].keys())
            
        # Add metric columns
        for metric in sorted(all_metrics):
            report += f"<th>{metric}</th>"
            
        report += "</tr>"
        
        # Add model rows
        for model_name, model_data in self.models.items():
            is_best = self.best_model and self.best_model["name"] == model_name
            row_class = "best" if is_best else ""
            
            # Use text instead of emoji to avoid encoding issues
            report += f"<tr class='{row_class}'><td>{('BEST ' if is_best else '')}{model_name}</td>"
            
            for metric in sorted(all_metrics):
                value = model_data["metrics"].get(metric, "N/A")
                if isinstance(value, (int, float, np.integer, np.floating)):
                    value = f"{float(value):.4f}"
                report += f"<td>{value}</td>"
                
            report += "</tr>"
            
        # Remainder of report generation...
        # ...
        
        # Write report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"Report generated: {output_file}")
        return output_file

def main():
    # Monkey patch the warnings module to completely suppress specific joblib warnings
    original_showwarning = warnings.showwarning
    def custom_showwarning(message, *args, **kwargs):
        # Skip resource tracker warnings about file not found
        if "resource_tracker" in str(message) and "FileNotFoundError" in str(message):
            return
        if "joblib_memmapping_folder" in str(message):
            return
        # Show all other warnings normally
        original_showwarning(message, *args, **kwargs)
    warnings.showwarning = custom_showwarning
    
    parser = argparse.ArgumentParser(description='Benchmark MLTrainingEngine on PMLB datasets')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results', 
                        help='Directory to save benchmark results')
    parser.add_argument('--n-datasets', type=int, default=10, 
                        help='Number of datasets to benchmark on')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--optimization-iterations', type=int, default=20, 
                        help='Number of optimization iterations')
    parser.add_argument('--n-jobs', type=int, default=-1, 
                        help='Number of parallel jobs for ML model training')
    parser.add_argument('--n-processes', type=int, default=None, 
                        help='Number of parallel processes for benchmark')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='Maximum number of samples to use from each dataset')
    parser.add_argument('--inference-repeat', type=int, default=5,
                        help='Number of times to repeat inference for more accurate timing')
    parser.add_argument('--memory-limit', type=int, default=None,
                        help='Memory limit in MB for joblib (default: 80% of system memory)')
    parser.add_argument('--no-memory-limit', action='store_true',
                        help='Disable memory limiting (may cause more resource leaks)')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = BenchmarkEngine(
        output_dir=args.output_dir,
        n_datasets=args.n_datasets,
        seed=args.seed,
        optimization_iterations=args.optimization_iterations,
        n_jobs=args.n_jobs,
        use_memory_limit=not args.no_memory_limit,
        memory_limit_mb=args.memory_limit
    )
    
    # Update the _benchmark_dataset method to include the max_samples parameter
    benchmark._benchmark_dataset = partial(benchmark._benchmark_dataset, max_samples=args.max_samples)
    
    # Execute the benchmark with keyboard interrupt handling
    start_time = time.time()
    try:
        benchmark.run_benchmark(n_processes=args.n_processes)
        total_time = time.time() - start_time
        print(f"\nTotal benchmark time: {total_time:.2f} seconds")
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user. Shutting down processes...")
        # Save partial results if possible
        if hasattr(benchmark, 'partial_results') and benchmark.partial_results:
            try:
                results_file = os.path.join(args.output_dir, f"partial_results_{int(time.time())}.json")
                with open(results_file, 'w') as f:
                    json.dump({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "interrupted",
                        "elapsed_time": time.time() - start_time,
                        "results": benchmark.partial_results
                    }, f, indent=2, cls=NumpyEncoder)
                print(f"Partial results saved to {results_file}")
            except Exception as e:
                print(f"Error saving partial results: {str(e)}")
        
        # Clean up resources explicitly
        print("Cleaning up resources...")
        benchmark._cleanup_resources()
        
        # Kill any remaining processes
        try:
            import signal
            os.killpg(os.getpgid(0), signal.SIGTERM)  # Send SIGTERM to process group
        except:
            pass
        
        # Force cleanup any joblib resources
        try:
            if hasattr(joblib, 'disk'):
                joblib.disk.delete_temporary_files()
            elif hasattr(joblib.pool, 'MemmappingPool'):
                joblib.pool.MemmappingPool.clear_temporary_files()
        except Exception as e:
            print(f"Warning: Failed to clean joblib temporary files: {e}")
        
        print("All processes terminated.")

if __name__ == "__main__":
    try:
        # On Linux/Unix, ensure the process runs in its own process group
        # This makes it possible to kill all child processes with a single signal
        if hasattr(os, 'setpgrp'):
            os.setpgrp()
            
        # Register signal handlers for graceful shutdown when running as main program
        def signal_handler(sig, frame):
            print(f"\nReceived signal {sig}. Initiating graceful shutdown...")
            # We re-raise KeyboardInterrupt to trigger the exception handlers in the main function
            if sig == signal.SIGINT:
                raise KeyboardInterrupt
            # For other signals, exit more directly
            sys.exit(1)
            
        # Register handlers for common termination signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request
        
        # Run the main function with proper exception handling
        main()
    except KeyboardInterrupt:
        print("\nBenchmark terminated by user.")
        # The keyboard interrupt is handled within main(), so we just need to exit gracefully here
        sys.exit(0)
    except Exception as e:
        print(f"\nUnhandled exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)