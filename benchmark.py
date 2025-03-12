import numpy as np
import pandas as pd
import os
import time
import json
import tempfile
import shutil
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from pmlb import fetch_data, classification_dataset_names
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse
import warnings

# Import the necessary modules from Genta AutoML
from modules.configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig,
    PreprocessorConfig,
    BatchProcessorConfig,
    InferenceEngineConfig,
    QuantizationConfig,
    NormalizationType
)
from modules.engine.train_engine import MLTrainingEngine

# Configure logging to suppress warning messages
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

class BenchmarkEngine:
    """Engine for benchmarking MLTrainingEngine performance across various datasets"""
    
    def __init__(self, output_dir="./benchmark_results", n_datasets=10, seed=42,
                 optimization_iterations=20, n_jobs=-1, test_size=0.2):
        """
        Initialize the benchmark engine
        
        Args:
            output_dir: Directory to save benchmark results
            n_datasets: Number of datasets to benchmark on
            seed: Random seed for reproducibility
            optimization_iterations: Number of optimization iterations to use
            n_jobs: Number of parallel jobs
            test_size: Test set size for dataset splitting
        """
        self.output_dir = output_dir
        self.n_datasets = n_datasets
        self.seed = seed
        self.optimization_iterations = optimization_iterations
        self.n_jobs = n_jobs
        self.test_size = test_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up default models and their parameter grids
        self.models = {
            "logistic_regression": {
                "model": LogisticRegression(random_state=seed, max_iter=1000),
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
    
    def _benchmark_dataset(self, dataset_name, strategies, max_samples=5000):
        """Benchmark various optimization strategies on a single dataset"""
        try:
            # Fetch the dataset
            X, y = fetch_data(dataset_name, return_X_y=True, local_cache_dir=tempfile.gettempdir())
            
            # Limit dataset size for speed if needed
            if X.shape[0] > max_samples:
                indices = np.random.choice(X.shape[0], max_samples, replace=False)
                X, y = X[indices], y[indices]
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.seed, stratify=y
            )
            
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
                            # Train and evaluate model
                            best_model, metrics = engine.train_model(
                                model=model_info["model"],
                                model_name=model_name,
                                param_grid=model_info["param_grid"],
                                X=X_train,
                                y=y_train,
                                X_test=X_test,
                                y_test=y_test
                            )
                            
                            model_time = time.time() - model_start
                            
                            # Store results
                            strategy_results[model_name] = {
                                "metrics": metrics,
                                "time": model_time
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
            
            return {
                "dataset": dataset_name,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_classes": len(np.unique(y)),
                "class_balance": dict(zip(*np.unique(y, return_counts=True))),
                "results": results
            }
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            return {
                "dataset": dataset_name,
                "error": str(e)
            }
    
    def run_benchmark(self, n_processes=None):
        """Run the benchmark on multiple datasets using process parallelism"""
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
        
        benchmark_results = []
        
        # Determine number of processes
        if n_processes is None:
            n_processes = min(os.cpu_count(), len(datasets))
        
        # Create a partial function with fixed strategies
        benchmark_fn = partial(self._benchmark_dataset, strategies=strategies)
        
        # Run benchmark in parallel
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = {executor.submit(benchmark_fn, dataset): dataset for dataset in datasets}
            
            for i, future in enumerate(as_completed(futures)):
                dataset = futures[future]
                try:
                    result = future.result()
                    benchmark_results.append(result)
                    print(f"Completed {i+1}/{len(datasets)}: {dataset}")
                except Exception as e:
                    print(f"Error on dataset {dataset}: {str(e)}")
        
        # Save results to file
        results_file = os.path.join(self.output_dir, f"benchmark_results_{int(time.time())}.json")
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_datasets": len(datasets),
                "seed": self.seed,
                "optimization_iterations": self.optimization_iterations,
                "results": benchmark_results
            }, f, indent=2)
        
        print(f"Benchmark completed. Results saved to {results_file}")
        return self._analyze_results(benchmark_results)
    
    def _analyze_results(self, results):
        """Analyze benchmark results and compute summary statistics"""
        # Initialize counters and aggregators
        strategy_wins = {strategy: 0 for strategy in ["grid_search", "random_search", "bayesian", "asht"]}
        model_wins = {model: 0 for model in self.models.keys()}
        strategy_times = {strategy: [] for strategy in ["grid_search", "random_search", "bayesian", "asht"]}
        strategy_scores = {strategy: [] for strategy in ["grid_search", "random_search", "bayesian", "asht"]}
        
        for result in results:
            if "error" in result:
                continue
                
            dataset = result["dataset"]
            
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
                            
                        # Track execution time
                        if "time" in model_result:
                            strategy_times[strategy].append(model_result["time"])
                
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
        summary = {
            "strategy_wins": strategy_wins,
            "model_wins": model_wins,
            "avg_times": {strategy: np.mean(times) if times else 0 for strategy, times in strategy_times.items()},
            "avg_scores": {strategy: np.mean(scores) if scores else 0 for strategy, scores in strategy_scores.items()}
        }
        
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
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f"benchmark_summary_{int(time.time())}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nSummary saved to {summary_file}")
        return summary

def main():
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
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = BenchmarkEngine(
        output_dir=args.output_dir,
        n_datasets=args.n_datasets,
        seed=args.seed,
        optimization_iterations=args.optimization_iterations,
        n_jobs=args.n_jobs
    )
    
    benchmark.run_benchmark(n_processes=args.n_processes)

if __name__ == "__main__":
    main()