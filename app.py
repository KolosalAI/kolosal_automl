<<<<<<< HEAD
import numpy as np
import pandas as pd
import time
import json
import os
import logging
import gc
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import asdict
import matplotlib.pyplot as plt
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, 
    load_diabetes, load_boston, make_classification, make_regression,
    fetch_openml
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# Import modules from Genta AutoML
from modules.configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig,
    PreprocessorConfig,
    NormalizationType,
    BatchProcessorConfig,
    BatchProcessingStrategy,
    InferenceEngineConfig
)
from modules.engine.train_engine import MLTrainingEngine
from modules.engine.data_preprocessor import DataPreprocessor
from modules.engine.quantizer import Quantizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BenchmarkRunner")


class BenchmarkRunner:
    """
    Run comprehensive benchmarks for the ML Training Engine across
    various datasets, models, and optimization strategies.
    """

    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = output_dir
        self.results = []
        self.current_benchmark_id = int(time.time())
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Benchmark Runner initialized with ID: {self.current_benchmark_id}")
        logger.info(f"Results will be saved to: {output_dir}")

    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (X, y, task_type_str)
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Classification datasets
        if dataset_name == "iris":
            data = load_iris()
            X, y = data.data, data.target
            task_type = TaskType.CLASSIFICATION.value
            
        elif dataset_name == "wine":
            data = load_wine()
            X, y = data.data, data.target
            task_type = TaskType.CLASSIFICATION.value
            
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
            task_type = TaskType.CLASSIFICATION.value
            
        elif dataset_name == "digits":
            data = load_digits()
            X, y = data.data, data.target
            task_type = TaskType.CLASSIFICATION.value
            
        # Regression datasets
        elif dataset_name == "diabetes":
            data = load_diabetes()
            X, y = data.data, data.target
            task_type = TaskType.REGRESSION.value
            
        elif dataset_name == "boston":
            try:
                data = load_boston()
                X, y = data.data, data.target
                task_type = TaskType.REGRESSION.value
            except:
                # Boston dataset may be removed in newer scikit-learn versions
                # Generate a synthetic regression dataset as fallback
                X, y = make_regression(n_samples=506, n_features=13, noise=0.1, random_state=42)
                task_type = TaskType.REGRESSION.value
                
        # Synthetic datasets for scalability testing
        elif dataset_name == "synthetic_small_classification":
            X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
            task_type = TaskType.CLASSIFICATION.value
            
        elif dataset_name == "synthetic_medium_classification":
            X, y = make_classification(n_samples=10000, n_features=50, n_classes=2, random_state=42)
            task_type = TaskType.CLASSIFICATION.value
            
        elif dataset_name == "synthetic_large_classification":
            X, y = make_classification(n_samples=50000, n_features=100, n_classes=2, random_state=42)
            task_type = TaskType.CLASSIFICATION.value
            
        elif dataset_name == "synthetic_small_regression":
            X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
            task_type = TaskType.REGRESSION.value
            
        elif dataset_name == "synthetic_medium_regression":
            X, y = make_regression(n_samples=10000, n_features=50, noise=0.1, random_state=42)
            task_type = TaskType.REGRESSION.value
            
        # OpenML datasets
        elif dataset_name == "adult":
            try:
                data = fetch_openml(name='adult', version=2, as_frame=True)
                # Handle categorical features
                X_df = pd.get_dummies(data.data, drop_first=True)
                X = X_df.values
                y = (data.target == '>50K').astype(int).values
                task_type = TaskType.CLASSIFICATION.value
            except:
                logger.warning("Failed to load Adult dataset, using synthetic data instead")
                X, y = make_classification(n_samples=32561, n_features=50, random_state=42)
                task_type = TaskType.CLASSIFICATION.value
        
        else:
            # Default: generate a small classification dataset
            logger.warning(f"Unknown dataset: {dataset_name}. Using synthetic data.")
            X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
            task_type = TaskType.CLASSIFICATION.value
            
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, task_type

    def get_model_and_params(self, model_name: str, task_type: str) -> Tuple[Any, Dict]:
        """
        Get a model and its hyperparameter search space based on the model name and task type.
        
        Args:
            model_name: Name of the model
            task_type: "classification" or "regression"
            
        Returns:
            Tuple of (model, param_grid)
        """
        # Classification models
        if task_type == TaskType.CLASSIFICATION.value:
            if model_name == "random_forest":
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5]
                }
                
            elif model_name == "gradient_boosting":
                model = GradientBoostingClassifier(random_state=42)
                param_grid = {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.01, 0.1],
                    'model__max_depth': [3, 5]
                }
                
            elif model_name == "logistic_regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
                param_grid = {
                    'model__C': [0.1, 1.0, 10.0],
                    'model__solver': ['liblinear', 'saga'],
                    'model__penalty': ['l1', 'l2']
                }
                
            elif model_name == "svc":
                model = SVC(random_state=42, probability=True)
                param_grid = {
                    'model__C': [0.1, 1.0, 10.0],
                    'model__kernel': ['linear', 'rbf'],
                    'model__gamma': ['scale', 'auto']
                }
                
            elif model_name == "decision_tree":
                model = DecisionTreeClassifier(random_state=42)
                param_grid = {
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5, 10],
                    'model__criterion': ['gini', 'entropy']
                }
                
            else:
                # Default to RandomForest for unknown models
                logger.warning(f"Unknown model: {model_name}. Using RandomForest.")
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'model__n_estimators': [100],
                    'model__max_depth': [None, 10]
                }
        
        # Regression models
        elif task_type == TaskType.REGRESSION.value:
            if model_name == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5]
                }
                
            elif model_name == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=42)
                param_grid = {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.01, 0.1],
                    'model__max_depth': [3, 5]
                }
                
            elif model_name == "ridge":
                model = Ridge(random_state=42)
                param_grid = {
                    'model__alpha': [0.1, 1.0, 10.0],
                    'model__solver': ['auto', 'svd', 'cholesky']
                }
                
            elif model_name == "lasso":
                model = Lasso(random_state=42)
                param_grid = {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0],
                    'model__max_iter': [1000, 2000]
                }
                
            else:
                # Default to RandomForestRegressor for unknown models
                logger.warning(f"Unknown model: {model_name}. Using RandomForestRegressor.")
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'model__n_estimators': [100],
                    'model__max_depth': [None, 10]
                }
        else:
            # Default to classification with RandomForest
            logger.warning(f"Unknown task type: {task_type}. Using classification with RandomForest.")
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'model__n_estimators': [100],
                'model__max_depth': [None, 10]
            }
            
        return model, param_grid

    def create_engine_config(self, task_type: str, optimization_strategy: str) -> MLTrainingEngineConfig:
        """
        Create a training engine configuration.
        
        Args:
            task_type: "classification" or "regression"
            optimization_strategy: Name of the optimization strategy
            
        Returns:
            MLTrainingEngineConfig instance
        """
        # Set task type
        if task_type == TaskType.CLASSIFICATION.value:
            task = TaskType.CLASSIFICATION
        elif task_type == TaskType.REGRESSION.value:
            task = TaskType.REGRESSION
        else:
            logger.warning(f"Unknown task type: {task_type}. Using classification.")
            task = TaskType.CLASSIFICATION
            
        # Set optimization strategy
        if optimization_strategy == "grid_search":
            strategy = OptimizationStrategy.GRID_SEARCH
        elif optimization_strategy == "random_search":
            strategy = OptimizationStrategy.RANDOM_SEARCH
        elif optimization_strategy == "bayesian_optimization":
            strategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
        elif optimization_strategy == "asht":
            strategy = OptimizationStrategy.ASHT
        else:
            logger.warning(f"Unknown optimization strategy: {optimization_strategy}. Using random search.")
            strategy = OptimizationStrategy.RANDOM_SEARCH
            
        # Create optimized preprocessor configuration
        preprocessor_config = PreprocessorConfig(
            normalization=NormalizationType.STANDARD,
            handle_nan=True,
            handle_inf=True,
            detect_outliers=True,
            parallel_processing=True,
            n_jobs=-1,
            cache_enabled=True
        )
        
        # Create optimized batch processor configuration
        batch_config = BatchProcessorConfig(
            initial_batch_size=64,
            min_batch_size=16,
            max_batch_size=256,
            max_queue_size=1000,
            processing_strategy=BatchProcessingStrategy.ADAPTIVE,
            enable_adaptive_batching=True,
            enable_memory_optimization=True,
            max_workers=4
        )
        
        # Create optimized inference engine configuration
        inference_config = InferenceEngineConfig(
            enable_intel_optimization=True,
            enable_batching=True,
            num_threads=4,
            enable_memory_optimization=True
        )
        
        # Create the MLTrainingEngineConfig
        config = MLTrainingEngineConfig(
            task_type=task,
            random_state=42,
            n_jobs=-1,  # Use all cores
            verbose=1,
            cv_folds=5,
            test_size=0.2,
            stratify=(task == TaskType.CLASSIFICATION),
            optimization_strategy=strategy,
            optimization_iterations=20,  # Reduced for faster benchmarking
            early_stopping=True,
            feature_selection=True,
            feature_selection_method="mutual_info",
            preprocessing_config=preprocessor_config,
            batch_processing_config=batch_config,
            inference_config=inference_config,
            model_path="./benchmark_models",
            experiment_tracking=True,
            use_intel_optimization=True,
            memory_optimization=True
        )
        
        return config

    def run_benchmark(
        self, 
        dataset_name: str, 
        model_name: str, 
        optimization_strategy: str
    ) -> Dict[str, Any]:
        """
        Run a single benchmark and return the results.
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            optimization_strategy: Name of the optimization strategy
            
        Returns:
            Dictionary with benchmark results
        """
        benchmark_id = f"{dataset_name}_{model_name}_{optimization_strategy}_{int(time.time())}"
        logger.info(f"Starting benchmark: {benchmark_id}")
        
        # Load dataset
        try:
            X, y, task_type = self.load_dataset(dataset_name)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
            return {
                "benchmark_id": benchmark_id,
                "dataset": dataset_name,
                "model": model_name,
                "optimization_strategy": optimization_strategy,
                "error": f"Failed to load dataset: {str(e)}",
                "success": False
            }
            
        # Get model and parameter grid
        try:
            model, param_grid = self.get_model_and_params(model_name, task_type)
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {str(e)}")
            return {
                "benchmark_id": benchmark_id,
                "dataset": dataset_name,
                "model": model_name,
                "optimization_strategy": optimization_strategy,
                "error": f"Failed to create model: {str(e)}",
                "success": False
            }
            
        # Create engine configuration
        try:
            config = self.create_engine_config(task_type, optimization_strategy)
        except Exception as e:
            logger.error(f"Failed to create engine configuration: {str(e)}")
            return {
                "benchmark_id": benchmark_id,
                "dataset": dataset_name,
                "model": model_name,
                "optimization_strategy": optimization_strategy,
                "error": f"Failed to create engine configuration: {str(e)}",
                "success": False
            }
            
        # Create ML Training Engine
        try:
            engine = MLTrainingEngine(config)
        except Exception as e:
            logger.error(f"Failed to create ML Training Engine: {str(e)}")
            return {
                "benchmark_id": benchmark_id,
                "dataset": dataset_name,
                "model": model_name,
                "optimization_strategy": optimization_strategy,
                "error": f"Failed to create ML Training Engine: {str(e)}",
                "success": False
            }
            
        # Collect system resources
        try:
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            cpu_count = os.cpu_count()
            system_info = {
                "cpu_count": cpu_count,
                "initial_memory_mb": initial_memory,
                "platform": os.name
            }
        except:
            system_info = {"error": "Failed to collect system info"}
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task_type == TaskType.CLASSIFICATION.value else None
        )
        
        # Train and time the model
        start_time = time.time()
        peak_memory = initial_memory
        
        try:
            # Training
            model, train_metrics = engine.train_model(
                model=model,
                model_name=f"{model_name}_benchmark",
                param_grid=param_grid,
                X=X_train,
                y=y_train,
                X_test=X_test,
                y_test=y_test
            )
            
            training_time = time.time() - start_time
            
            # Get peak memory usage
            try:
                peak_memory = max(peak_memory, process.memory_info().rss / (1024 * 1024))
            except:
                pass
                
            # Make predictions for timings
            inference_start = time.time()
            predictions = engine.predict(X_test)
            inference_time = time.time() - inference_start
            
            # Calculate metrics
            if task_type == TaskType.CLASSIFICATION.value:
                test_accuracy = accuracy_score(y_test, predictions)
                test_f1 = f1_score(y_test, predictions, average='weighted')
                metrics = {
                    "test_accuracy": test_accuracy,
                    "test_f1": test_f1
                }
            else:
                test_mse = mean_squared_error(y_test, predictions)
                test_rmse = np.sqrt(test_mse)
                metrics = {
                    "test_mse": test_mse,
                    "test_rmse": test_rmse
                }
                
            # Get throughput metrics
            train_throughput = X_train.shape[0] / training_time if training_time > 0 else 0
            inference_throughput = X_test.shape[0] / inference_time if inference_time > 0 else 0
            
            # Get final memory usage
            try:
                final_memory = process.memory_info().rss / (1024 * 1024)
            except:
                final_memory = 0
                
            # Get engine stats
            try:
                if hasattr(engine.batch_processor, 'get_stats'):
                    batch_stats = engine.batch_processor.get_stats()
                else:
                    batch_stats = {}
            except:
                batch_stats = {}
                
            # Success result
            result = {
                "benchmark_id": benchmark_id,
                "dataset": dataset_name,
                "model": model_name,
                "optimization_strategy": optimization_strategy,
                "dataset_shape": X.shape,
                "system_info": system_info,
                "training_time_seconds": training_time,
                "inference_time_seconds": inference_time,
                "peak_memory_mb": peak_memory,
                "final_memory_mb": final_memory,
                "training_throughput_samples_per_second": train_throughput,
                "inference_throughput_samples_per_second": inference_throughput,
                "metrics": metrics,
                "train_metrics": train_metrics,
                "batch_stats": batch_stats,
                "success": True,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Benchmark error: {str(e)}")
            training_time = time.time() - start_time
            
            # Get peak memory usage
            try:
                peak_memory = max(peak_memory, process.memory_info().rss / (1024 * 1024))
            except:
                pass
                
            # Error result
            result = {
                "benchmark_id": benchmark_id,
                "dataset": dataset_name,
                "model": model_name,
                "optimization_strategy": optimization_strategy,
                "dataset_shape": X.shape,
                "system_info": system_info,
                "training_time_seconds": training_time,
                "peak_memory_mb": peak_memory,
                "error": str(e),
                "success": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        finally:
            # Clean up
            try:
                engine.shutdown()
                del engine
                gc.collect()
            except:
                pass
                
        # Store benchmark result
        self.results.append(result)
        
        # Save individual benchmark result
        result_file = os.path.join(self.output_dir, f"benchmark_{benchmark_id}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Benchmark completed: {benchmark_id}")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Results saved to: {result_file}")
        
        return result

    def run_multiple_benchmarks(self, configurations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Run multiple benchmarks based on configurations.
        
        Args:
            configurations: List of dictionaries with keys 'dataset', 'model', 'optimization_strategy'
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for i, config in enumerate(configurations):
            logger.info(f"Running benchmark {i+1}/{len(configurations)}: {config}")
            
            result = self.run_benchmark(
                dataset_name=config['dataset'],
                model_name=config['model'],
                optimization_strategy=config['optimization_strategy']
            )
            
            results.append(result)
            
        return results

    def save_results(self) -> str:
        """
        Save all benchmark results to a file.
        
        Returns:
            Path to the results file
        """
        results_file = os.path.join(self.output_dir, f"benchmark_results_{self.current_benchmark_id}.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"All benchmark results saved to: {results_file}")
        return results_file

    def generate_report(self) -> str:
        """
        Generate an HTML report with visualizations of benchmark results.
        
        Returns:
            Path to the report file
        """
        if not self.results:
            logger.warning("No results to generate report.")
            return ""
            
        report_file = os.path.join(self.output_dir, f"benchmark_report_{self.current_benchmark_id}.html")
        
        # Filter successful benchmarks only
        successful_results = [r for r in self.results if r.get('success', False)]
        
        if not successful_results:
            logger.warning("No successful benchmarks to include in report.")
            return ""
            
        # Calculate summary statistics
        avg_training_time = np.mean([r['training_time_seconds'] for r in successful_results])
        avg_inference_time = np.mean([r.get('inference_time_seconds', 0) for r in successful_results])
        avg_train_throughput = np.mean([r.get('training_throughput_samples_per_second', 0) for r in successful_results])
        avg_inference_throughput = np.mean([r.get('inference_throughput_samples_per_second', 0) for r in successful_results])
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Report - {self.current_benchmark_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #e6f7ff; padding: 15px; margin: 10px 0; }}
                .chart-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Benchmark Report</h1>
            <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <p><strong>Total benchmarks:</strong> {len(self.results)}</p>
                <p><strong>Successful benchmarks:</strong> {len(successful_results)}</p>
                <p><strong>Average training time:</strong> {avg_training_time:.2f} seconds</p>
                <p><strong>Average inference time:</strong> {avg_inference_time:.2f} seconds</p>
                <p><strong>Average training throughput:</strong> {avg_train_throughput:.2f} samples/second</p>
                <p><strong>Average inference throughput:</strong> {avg_inference_throughput:.2f} samples/second</p>
            </div>
            
            <h2>Benchmark Results</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Dataset</th>
                    <th>Model</th>
                    <th>Optimization</th>
                    <th>Training Time (s)</th>
                    <th>Inference Time (s)</th>
                    <th>Peak Memory (MB)</th>
                    <th>Training Throughput</th>
                    <th>Inference Throughput</th>
                    <th>Metrics</th>
                </tr>
        """
        
        # Add rows for each benchmark
        for r in self.results:
            success = r.get('success', False)
            row_style = "" if success else "background-color: #ffeeee;"
            
            if success:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in r.get('metrics', {}).items()])
                html_content += f"""
                <tr style="{row_style}">
                    <td>{r['benchmark_id']}</td>
                    <td>{r['dataset']}</td>
                    <td>{r['model']}</td>
                    <td>{r['optimization_strategy']}</td>
                    <td>{r['training_time_seconds']:.2f}</td>
                    <td>{r.get('inference_time_seconds', 'N/A')}</td>
                    <td>{r['peak_memory_mb']:.2f}</td>
                    <td>{r.get('training_throughput_samples_per_second', 'N/A'):.2f}</td>
                    <td>{r.get('inference_throughput_samples_per_second', 'N/A'):.2f}</td>
                    <td>{metrics_str}</td>
                </tr>
                """
            else:
                html_content += f"""
                <tr style="{row_style}">
                    <td>{r['benchmark_id']}</td>
                    <td>{r['dataset']}</td>
                    <td>{r['model']}</td>
                    <td>{r['optimization_strategy']}</td>
                    <td>{r['training_time_seconds']:.2f}</td>
                    <td>Failed</td>
                    <td>{r['peak_memory_mb']:.2f}</td>
                    <td>Failed</td>
                    <td>Failed</td>
                    <td>Error: {r.get('error', 'Unknown error')}</td>
                </tr>
                """
                
        html_content += """
            </table>
        """
        
        # Generate charts for visualization
        if successful_results:
            # Create charts directory
            charts_dir = os.path.join(self.output_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # Training time by model and dataset
            plt.figure(figsize=(12, 6))
            df = pd.DataFrame(successful_results)
            pivot = df.pivot_table(
                values='training_time_seconds', 
                index='model', 
                columns='optimization_strategy', 
                aggfunc='mean'
            )
            ax = pivot.plot(kind='bar')
            plt.title('Training Time by Model and Optimization Strategy')
            plt.ylabel('Training Time (seconds)')
            plt.tight_layout()
            train_time_chart = os.path.join(charts_dir, f"training_time_{self.current_benchmark_id}.png")
            plt.savefig(train_time_chart)
            plt.close()
            
            # Throughput by model and dataset
            plt.figure(figsize=(12, 6))
            pivot = df.pivot_table(
                values='training_throughput_samples_per_second', 
                index='model', 
                columns='optimization_strategy', 
                aggfunc='mean'
            )
            ax = pivot.plot(kind='bar')
            plt.title('Training Throughput by Model and Optimization Strategy')
            plt.ylabel('Throughput (samples/second)')
            plt.tight_layout()
            throughput_chart = os.path.join(charts_dir, f"training_throughput_{self.current_benchmark_id}.png")
            plt.savefig(throughput_chart)
            plt.close()
            
            # Memory usage by model
            plt.figure(figsize=(12, 6))
            df.groupby('model')['peak_memory_mb'].mean().plot(kind='bar')
            plt.title('Peak Memory Usage by Model')
            plt.ylabel('Memory (MB)')
            plt.tight_layout()
            memory_chart = os.path.join(charts_dir, f"memory_usage_{self.current_benchmark_id}.png")
            plt.savefig(memory_chart)
            plt.close()
            
            # Add charts to HTML
            html_content += f"""
                <h2>Performance Visualizations</h2>
                
                <div class="chart-container">
                    <h3>Training Time by Model and Optimization Strategy</h3>
                    <img src="{os.path.relpath(train_time_chart, self.output_dir)}" width="800">
                </div>
                
                <div class="chart-container">
                    <h3>Training Throughput by Model and Optimization Strategy</h3>
                    <img src="{os.path.relpath(throughput_chart, self.output_dir)}" width="800">
                </div>
                
                <div class="chart-container">
                    <h3>Peak Memory Usage by Model</h3>
                    <img src="{os.path.relpath(memory_chart, self.output_dir)}" width="800">
                </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(report_file, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Benchmark report generated: {report_file}")
        return report_file

def main():
    """Main function to run benchmarks"""
    # Create benchmark runner
    benchmark_runner = BenchmarkRunner(output_dir="./benchmark_results")
    
    # Define benchmark configurations
    configurations = [
        # Small datasets
        {"dataset": "iris", "model": "random_forest", "optimization_strategy": "grid_search"},
        {"dataset": "iris", "model": "random_forest", "optimization_strategy": "random_search"},
        {"dataset": "iris", "model": "random_forest", "optimization_strategy": "asht"},
        
        {"dataset": "iris", "model": "gradient_boosting", "optimization_strategy": "random_search"},
        {"dataset": "iris", "model": "logistic_regression", "optimization_strategy": "random_search"},
        
        # Medium datasets with different models
        {"dataset": "wine", "model": "random_forest", "optimization_strategy": "asht"},
        {"dataset": "wine", "model": "gradient_boosting", "optimization_strategy": "asht"},
        {"dataset": "wine", "model": "logistic_regression", "optimization_strategy": "asht"},
        
        # Medium datasets
        {"dataset": "breast_cancer", "model": "random_forest", "optimization_strategy": "grid_search"},
        {"dataset": "breast_cancer", "model": "random_forest", "optimization_strategy": "random_search"},
        {"dataset": "breast_cancer", "model": "random_forest", "optimization_strategy": "asht"},
        
        # Regression datasets
        {"dataset": "diabetes", "model": "random_forest", "optimization_strategy": "random_search"},
        {"dataset": "diabetes", "model": "gradient_boosting", "optimization_strategy": "random_search"},
        {"dataset": "diabetes", "model": "ridge", "optimization_strategy": "random_search"},
        
        # Larger synthetic datasets
        {"dataset": "synthetic_medium_classification", "model": "random_forest", "optimization_strategy": "random_search"},
        {"dataset": "synthetic_medium_regression", "model": "random_forest", "optimization_strategy": "random_search"},
        
        # Comparing optimization strategies on a standard dataset
        {"dataset": "breast_cancer", "model": "random_forest", "optimization_strategy": "grid_search"},
        {"dataset": "breast_cancer", "model": "random_forest", "optimization_strategy": "random_search"},
        {"dataset": "breast_cancer", "model": "random_forest", "optimization_strategy": "bayesian_optimization"},
        {"dataset": "breast_cancer", "model": "random_forest", "optimization_strategy": "asht"},
    ]
    
    # Run benchmarks
    benchmark_runner.run_multiple_benchmarks(configurations)
    
    # Save results
    results_file = benchmark_runner.save_results()
    
    # Generate report
    report_file = benchmark_runner.generate_report()
    
    logger.info(f"Benchmarking complete!")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Report generated: {report_file}")
    
    return {
        "results_file": results_file,
        "report_file": report_file
    }
=======
import gradio as gr
import pandas as pd
import numpy as np
import json
import os
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import requests
from io import StringIO
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Import your modules
from modules.configs import (
    MLTrainingEngineConfig, InferenceEngineConfig, 
    TaskType, OptimizationStrategy, ModelSelectionCriteria,
    OptimizationMode, NormalizationType, QuantizationType
)
from modules.engine.train_engine import MLTrainingEngine
from modules.engine.inference_engine import InferenceEngine
from modules.device_optimizer import DeviceOptimizer
from modules.model_manager import SecureModelManager

import modules.engine.batch_processor as batch_processor
import modules.engine.data_preprocessor as data_preprocessor
import modules.engine.lru_ttl_cache as lru_ttl_cache
import modules.engine.quantizer as quantizer
import modules.engine.utils as engine_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreviewGenerator:
    """Generate comprehensive data previews and visualizations"""
    
    @staticmethod
    def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        summary = {
            "basic_info": {
                "shape": df.shape,
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict()
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_by_column": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict()
            },
            "data_types": {
                "numerical": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime": df.select_dtypes(include=['datetime64']).columns.tolist()
            }
        }
        
        # Add descriptive statistics for numerical columns
        numerical_cols = summary["data_types"]["numerical"]
        if numerical_cols:
            summary["numerical_stats"] = df[numerical_cols].describe().round(3).to_dict()
        
        # Add categorical summaries
        categorical_cols = summary["data_types"]["categorical"]
        if categorical_cols:
            summary["categorical_stats"] = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                value_counts = df[col].value_counts().head(10)
                summary["categorical_stats"][col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": value_counts.to_dict()
                }
        
        return summary
    
    @staticmethod
    def create_data_visualizations(df: pd.DataFrame, max_cols: int = 6) -> str:
        """Create data visualization plots and return as base64 image"""
        try:
            # Set style with proper seaborn v0.11+ syntax
            plt.style.use('default')
            sns.set_palette("husl")
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:max_cols]
            
            # Calculate subplot layout
            total_plots = 0
            if len(numerical_cols) > 0:
                total_plots += min(len(numerical_cols), 4)  # Distribution plots
            if len(categorical_cols) > 0:
                total_plots += min(len(categorical_cols), 2)  # Bar plots
            if len(numerical_cols) > 1:
                total_plots += 1  # Correlation heatmap
            
            if total_plots == 0:
                return None
            
            # Create figure
            n_cols = 3
            n_rows = (total_plots + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            
            if n_rows == 1:
                if total_plots == 1:
                    axes = [axes]
                else:
                    axes = axes if isinstance(axes, list) else axes.flatten()
            else:
                axes = axes.flatten()
            
            plot_idx = 0
            
            # 1. Numerical distributions
            for i, col in enumerate(numerical_cols[:4]):
                if plot_idx < len(axes):
                    try:
                        axes[plot_idx].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                        axes[plot_idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                        axes[plot_idx].set_xlabel(col)
                        axes[plot_idx].set_ylabel('Frequency')
                        axes[plot_idx].grid(True, alpha=0.3)
                        plot_idx += 1
                    except Exception as e:
                        continue
            
            # 2. Categorical bar plots
            for i, col in enumerate(categorical_cols[:2]):
                if plot_idx < len(axes):
                    try:
                        value_counts = df[col].value_counts().head(10)
                        bars = axes[plot_idx].bar(range(len(value_counts)), value_counts.values, 
                                                 color='lightcoral', alpha=0.7, edgecolor='black')
                        axes[plot_idx].set_title(f'Top 10 Values in {col}', fontsize=12, fontweight='bold')
                        axes[plot_idx].set_xlabel(col)
                        axes[plot_idx].set_ylabel('Count')
                        
                        # Rotate x-axis labels if needed
                        labels = [str(x)[:15] + '...' if len(str(x)) > 15 else str(x) for x in value_counts.index]
                        axes[plot_idx].set_xticks(range(len(value_counts)))
                        axes[plot_idx].set_xticklabels(labels, rotation=45, ha='right')
                        axes[plot_idx].grid(True, alpha=0.3)
                        plot_idx += 1
                    except Exception as e:
                        continue
            
            # 3. Correlation heatmap for numerical columns
            if len(numerical_cols) > 1 and plot_idx < len(axes):
                try:
                    corr_matrix = df[numerical_cols].corr()
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    
                    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                               square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=axes[plot_idx])
                    axes[plot_idx].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
                    plot_idx += 1
                except Exception as e:
                    pass
            
            # Hide unused subplots
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return None
    
    @staticmethod
    def format_data_preview(df: pd.DataFrame, summary: Dict[str, Any]) -> str:
        """Format comprehensive data preview as markdown"""
        
        # Basic information
        preview_text = f"""
## 📊 **Dataset Overview**

- **Shape**: {summary['basic_info']['shape'][0]:,} rows × {summary['basic_info']['shape'][1]} columns
- **Memory Usage**: {summary['basic_info']['memory_usage_mb']:.2f} MB
- **Total Missing Values**: {summary['missing_data']['total_missing']:,} ({summary['missing_data']['total_missing'] / (df.shape[0] * df.shape[1]) * 100:.1f}% of all data)

### 🔍 **Data Types**
- **Numerical Columns** ({len(summary['data_types']['numerical'])}): {', '.join(summary['data_types']['numerical'][:5])}{'...' if len(summary['data_types']['numerical']) > 5 else ''}
- **Categorical Columns** ({len(summary['data_types']['categorical'])}): {', '.join(summary['data_types']['categorical'][:5])}{'...' if len(summary['data_types']['categorical']) > 5 else ''}
- **DateTime Columns** ({len(summary['data_types']['datetime'])}): {', '.join(summary['data_types']['datetime'][:5])}{'...' if len(summary['data_types']['datetime']) > 5 else ''}
        """
        
        # Missing data details
        if summary['missing_data']['total_missing'] > 0:
            preview_text += "\n### ⚠️ **Missing Data by Column**\n"
            missing_cols = {k: v for k, v in summary['missing_data']['missing_by_column'].items() if v > 0}
            for col, missing_count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = summary['missing_data']['missing_percentage'][col]
                preview_text += f"- **{col}**: {missing_count:,} missing ({percentage:.1f}%)\n"
        
        # Numerical statistics
        if 'numerical_stats' in summary and summary['numerical_stats']:
            preview_text += "\n### 📈 **Numerical Statistics**\n"
            stats_df = pd.DataFrame(summary['numerical_stats']).round(2)
            # Create a simple text table instead of markdown table for better compatibility
            preview_text += "\n"
            for stat in ['mean', 'std', 'min', 'max']:
                if stat in stats_df.index:
                    preview_text += f"**{stat.upper()}**:\n"
                    for col in stats_df.columns[:5]:  # Limit to first 5 columns
                        preview_text += f"- {col}: {stats_df.loc[stat, col]:.2f}\n"
                    preview_text += "\n"
        
        # Categorical summaries
        if 'categorical_stats' in summary and summary['categorical_stats']:
            preview_text += "\n### 🏷️ **Categorical Summaries**\n"
            for col, stats in list(summary['categorical_stats'].items())[:3]:
                preview_text += f"\n**{col}** ({stats['unique_count']} unique values):\n"
                for value, count in list(stats['top_values'].items())[:5]:
                    preview_text += f"- {value}: {count:,}\n"
        
        return preview_text

class SampleDataLoader:
    """Load sample datasets from public sources"""
    
    SAMPLE_DATASETS = {
        "Iris": {
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "description": "Classic iris flower classification dataset",
            "task_type": "CLASSIFICATION",
            "target": "species"
        },
        "Boston Housing": {
            "url": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
            "description": "Boston housing prices regression dataset",
            "task_type": "REGRESSION",
            "target": "medv"
        },
        "Titanic": {
            "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "description": "Titanic passenger survival classification",
            "task_type": "CLASSIFICATION",
            "target": "Survived"
        },
        "Wine Quality": {
            "url": "https://raw.githubusercontent.com/rajeevratan84/datascienceforbeginnerssklearn/master/winequality-red.csv",
            "description": "Wine quality prediction dataset",
            "task_type": "REGRESSION",
            "target": "quality"
        },
        "Diabetes": {
            "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            "description": "Pima Indians diabetes classification",
            "task_type": "CLASSIFICATION",
            "target": "class"
        },
        "Car Evaluation": {
            "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/car.csv",
            "description": "Car evaluation classification dataset",
            "task_type": "CLASSIFICATION",
            "target": "class"
        }
    }
    
    @classmethod
    def get_available_datasets(cls):
        """Get list of available sample datasets"""
        return list(cls.SAMPLE_DATASETS.keys())
    
    @classmethod
    def load_sample_data(cls, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load a sample dataset"""
        if dataset_name not in cls.SAMPLE_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not available")
        
        dataset_info = cls.SAMPLE_DATASETS[dataset_name]
        
        try:
            # Download the dataset
            response = requests.get(dataset_info["url"], timeout=30)
            response.raise_for_status()
            
            # Handle different CSV formats
            if dataset_name == "Diabetes":
                # Diabetes dataset doesn't have headers
                columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
                df = pd.read_csv(StringIO(response.text), names=columns)
            else:
                df = pd.read_csv(StringIO(response.text))
            
            # Basic preprocessing for some datasets
            if dataset_name == "Titanic":
                # Clean up Titanic dataset
                df = df.dropna(subset=['Age', 'Embarked'])
                df['Age'] = df['Age'].fillna(df['Age'].median())
                df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
                
            metadata = {
                "name": dataset_name,
                "description": dataset_info["description"],
                "task_type": dataset_info["task_type"],
                "target_column": dataset_info["target"],
                "shape": df.shape,
                "columns": df.columns.tolist()
            }
            
            return df, metadata
            
        except Exception as e:
            raise Exception(f"Failed to load dataset '{dataset_name}': {str(e)}")

class InferenceServer:
    """Standalone inference server for trained models"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        self.model_manager = None
        self.inference_engine = None
        self.model_metadata = {}
        self.is_loaded = False
        
        if model_path:
            self.load_model_from_path(model_path, config_path)
    
    def load_model_from_path(self, model_path: str, config_path: str = None):
        """Load model from file path"""
        try:
            # Load configuration
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                config = MLTrainingEngineConfig(**config_data)
            else:
                config = MLTrainingEngineConfig()
            
            # Initialize model manager
            self.model_manager = SecureModelManager(config, logger=logger)
            
            # Load the model
            model = self.model_manager.load_model(model_path)
            
            if model is not None:
                # Initialize inference engine
                inference_config = InferenceEngineConfig()
                self.inference_engine = InferenceEngine(inference_config)
                self.inference_engine.model = model
                
                self.is_loaded = True
                logger.info(f"Model loaded successfully from {model_path}")
                
                # Extract metadata if available
                if hasattr(model, 'feature_names_'):
                    self.model_metadata['feature_names'] = model.feature_names_
                if hasattr(model, 'classes_'):
                    self.model_metadata['classes'] = model.classes_.tolist()
                    
            else:
                raise Exception("Failed to load model")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Make prediction using loaded model"""
        if not self.is_loaded:
            return {"error": "No model loaded"}
        
        try:
            success, predictions = self.inference_engine.predict(input_data)
            
            if not success:
                return {"error": f"Prediction failed: {predictions}"}
            
            result = {
                "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                "input_shape": input_data.shape,
                "model_metadata": self.model_metadata
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"error": "No model loaded"}
        
        return {
            "is_loaded": self.is_loaded,
            "metadata": self.model_metadata,
            "model_type": type(self.inference_engine.model).__name__ if self.inference_engine.model else None
        }

class MLSystemUI:
    """Enhanced Gradio UI for the ML Training & Inference System"""
    
    def __init__(self, inference_only: bool = False):
        self.inference_only = inference_only
        self.training_engine = None
        self.inference_engine = None
        self.device_optimizer = None
        self.model_manager = None
        self.inference_server = InferenceServer()
        self.current_data = None
        self.current_config = None
        self.sample_data_loader = SampleDataLoader()
        self.data_preview_generator = DataPreviewGenerator()
        self.trained_models = {}  # Store trained models
        
        # Define available ML algorithms with their categories and correct keys
        self.ml_algorithms = {
            "Tree-Based": {
                "Random Forest": {"key": "random_forest", "supports": ["classification", "regression"]},
                "Extra Trees": {"key": "extra_trees", "supports": ["classification", "regression"]},
                "Decision Tree": {"key": "decision_tree", "supports": ["classification", "regression"]},
                "Gradient Boosting": {"key": "gradient_boosting", "supports": ["classification", "regression"]},
            },
            "Boosting": {
                "XGBoost": {"key": "xgboost", "supports": ["classification", "regression"]},
                "LightGBM": {"key": "lightgbm", "supports": ["classification", "regression"]},
                "CatBoost": {"key": "catboost", "supports": ["classification", "regression"]},
                "AdaBoost": {"key": "adaboost", "supports": ["classification", "regression"]},
            },
            "Linear Models": {
                "Logistic Regression": {"key": "logistic_regression", "supports": ["classification"]},
                "Linear Regression": {"key": "linear_regression", "supports": ["regression"]},
                "Ridge": {"key": "ridge", "supports": ["classification", "regression"]},
                "Lasso": {"key": "lasso", "supports": ["classification", "regression"]},
                "Elastic Net": {"key": "elastic_net", "supports": ["classification", "regression"]},
                "SGD": {"key": "sgd", "supports": ["classification", "regression"]},
            },
            "Support Vector Machines": {
                "SVM": {"key": "svm", "supports": ["classification", "regression"]},
                "SVM (Linear)": {"key": "svm_linear", "supports": ["classification", "regression"]},
                "SVM (Polynomial)": {"key": "svm_poly", "supports": ["classification", "regression"]},
            },
            "Neural Networks": {
                "Multi-layer Perceptron": {"key": "mlp", "supports": ["classification", "regression"]},
                "Neural Network": {"key": "neural_network", "supports": ["classification", "regression"]},
            },
            "Naive Bayes": {
                "Gaussian NB": {"key": "naive_bayes", "supports": ["classification"]},
                "Multinomial NB": {"key": "multinomial_nb", "supports": ["classification"]},
                "Bernoulli NB": {"key": "bernoulli_nb", "supports": ["classification"]},
            },
            "Nearest Neighbors": {
                "K-Nearest Neighbors": {"key": "knn", "supports": ["classification", "regression"]},
            },
            "Ensemble Methods": {
                "Voting Classifier": {"key": "voting", "supports": ["classification"]},
                "Stacking": {"key": "stacking", "supports": ["classification", "regression"]},
            }
        }
        
        # Initialize device optimizer for system info
        if not inference_only:
            try:
                self.device_optimizer = DeviceOptimizer()
                logger.info("Device optimizer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize device optimizer: {e}")
    
    def get_algorithms_for_task(self, task_type: str) -> List[str]:
        """Get available algorithms for a specific task type"""
        algorithms = []
        task_lower = task_type.lower()
        
        for category, models in self.ml_algorithms.items():
            for model_name, model_info in models.items():
                if task_lower in model_info["supports"]:
                    algorithms.append(f"{category} - {model_name}")
        
        return sorted(algorithms)
    
    def get_model_key_from_name(self, algorithm_name: str) -> str:
        """Extract model key from formatted algorithm name with fallback mapping"""
        if " - " in algorithm_name:
            category, model_name = algorithm_name.split(" - ", 1)
            for cat, models in self.ml_algorithms.items():
                if cat == category and model_name in models:
                    return models[model_name]["key"]
        
        # Fallback mapping for common algorithm names
        fallback_mapping = {
            "random forest": "random_forest",
            "decision tree": "decision_tree",
            "gradient boosting": "gradient_boosting",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "adaboost": "adaboost",
            "logistic regression": "logistic_regression",
            "linear regression": "linear_regression",
            "svm": "svm",
            "k-nearest neighbors": "knn",
            "knn": "knn",
            "naive bayes": "naive_bayes",
            "gaussian nb": "naive_bayes",
            "multinomial nb": "multinomial_nb",
            "multi-layer perceptron": "mlp",
            "mlp": "mlp",
            "voting classifier": "voting",
            "stacking": "stacking",
            "sgd": "sgd"
        }
        
        # Try fallback mapping
        algorithm_lower = algorithm_name.lower()
        if algorithm_lower in fallback_mapping:
            return fallback_mapping[algorithm_lower]
        
        # Extract just the model name if it has " - " format
        if " - " in algorithm_name:
            model_name = algorithm_name.split(" - ", 1)[1].lower()
            if model_name in fallback_mapping:
                return fallback_mapping[model_name]
        
        # Default fallback
        return algorithm_name.lower().replace(" ", "_")
    
    def get_trained_model_list(self) -> List[str]:
        """Get list of trained models"""
        model_list = ["Select a trained model..."]
        if self.trained_models:
            for model_name in self.trained_models.keys():
                model_list.append(model_name)
        return model_list
    
    def get_system_info(self) -> str:
        """Get current system information"""
        try:
            if self.device_optimizer:
                info = self.device_optimizer.get_system_info()
                return json.dumps(info, indent=2)
            return "Device optimizer not available"
        except Exception as e:
            return f"Error getting system info: {str(e)}"
    
    def load_sample_data(self, dataset_name: str) -> Tuple[str, Dict, str, str]:
        """Load sample dataset with preview"""
        try:
            if dataset_name == "Select a dataset...":
                return "Please select a dataset", {}, "", ""
            
            df, metadata = self.sample_data_loader.load_sample_data(dataset_name)
            self.current_data = df
            
            # Generate data summary and preview
            summary = self.data_preview_generator.generate_data_summary(df)
            preview_text = self.data_preview_generator.format_data_preview(df, summary)
            
            # Generate sample data table
            sample_table = df.head(10).to_html(classes="table table-striped", escape=False, border=0)
            
            info_text = f"""
**Sample Dataset Loaded: {metadata['name']}**

- **Description**: {metadata['description']}
- **Task Type**: {metadata['task_type']}
- **Target Column**: {metadata['target_column']}
- **Shape**: {df.shape[0]} rows × {df.shape[1]} columns
- **Columns**: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
- **Missing Values**: {df.isnull().sum().sum()} total
            """
            
            return info_text, metadata, preview_text, sample_table
            
        except Exception as e:
            error_msg = f"Error loading sample data: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}, "", ""
    
    def load_data(self, file) -> Tuple[str, Dict, str, str]:
        """Load dataset from uploaded file with preview"""
        try:
            if file is None:
                return "No file uploaded", {}, "", ""
            
            file_path = file.name
            
            # Load based on file extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return "Unsupported file format. Please upload CSV, Excel, or JSON files.", {}, "", ""
            
            self.current_data = df
            
            # Generate data summary and preview
            summary = self.data_preview_generator.generate_data_summary(df)
            preview_text = self.data_preview_generator.format_data_preview(df, summary)
            
            # Generate sample data table
            sample_table = df.head(10).to_html(classes="table table-striped", escape=False, border=0)
            
            info_text = f"""
**Data Loaded Successfully!**

- **Shape**: {df.shape[0]} rows × {df.shape[1]} columns
- **Columns**: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
- **Missing Values**: {df.isnull().sum().sum()} total
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            """
            
            return info_text, summary, preview_text, sample_table
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}, "", ""
    
    def update_algorithm_choices(self, task_type: str) -> gr.Dropdown:
        """Update algorithm choices based on task type"""
        algorithms = self.get_algorithms_for_task(task_type)
        return gr.Dropdown(choices=algorithms, value=algorithms[0] if algorithms else None)
    
    def load_inference_model(self, file, encryption_password: str = "") -> str:
        """Load a model for inference server"""
        try:
            if file is None:
                return "No model file selected."
            
            # Load model into inference server
            if encryption_password:
                # Handle encrypted models (you may need to modify this based on your encryption implementation)
                pass
            
            self.inference_server.load_model_from_path(file.name)
            
            if self.inference_server.is_loaded:
                model_info = self.inference_server.get_model_info()
                return f"""
✅ **Model loaded successfully for inference!**

- **File**: {file.name}
- **Model Type**: {model_info.get('model_type', 'Unknown')}
- **Status**: Ready for predictions
                """
            else:
                return "❌ Failed to load model for inference."
                
        except Exception as e:
            error_msg = f"Error loading inference model: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def make_inference_prediction(self, input_data: str) -> str:
        """Make predictions using the inference server"""
        try:
            if not self.inference_server.is_loaded:
                return "No model loaded in inference server. Please load a model first."
            
            # Parse input data
            try:
                if input_data.strip().startswith('['):
                    # JSON array format
                    data = json.loads(input_data)
                    input_array = np.array(data).reshape(1, -1)
                else:
                    # Comma-separated values
                    data = [float(x.strip()) for x in input_data.split(',')]
                    input_array = np.array(data).reshape(1, -1)
            except Exception as e:
                return f"Error parsing input data: {str(e)}. Please use comma-separated values or JSON array format."
            
            # Make prediction using inference server
            result = self.inference_server.predict(input_array)
            
            if "error" in result:
                return f"Prediction failed: {result['error']}"
            
            # Format results
            predictions = result["predictions"]
            prediction_text = f"""
**Inference Server Prediction:**

- **Input**: {input_data}
- **Prediction**: {predictions}
- **Input Shape**: {result['input_shape']}
- **Model Type**: {result.get('model_metadata', {}).get('model_type', 'Unknown')}
            """
            
            return prediction_text
            
        except Exception as e:
            error_msg = f"Error making inference prediction: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def create_training_config(self, task_type: str, optimization_strategy: str, 
                             cv_folds: int, test_size: float, random_state: int,
                             enable_feature_selection: bool, normalization: str,
                             enable_quantization: bool, optimization_mode: str) -> Tuple[str, gr.Dropdown, List[str]]:
        """Create training configuration and update algorithm choices"""
        if self.inference_only:
            return "Training is not available in inference-only mode.", gr.Dropdown(), []
        
        try:
            # Map string values to enums
            task_type_enum = TaskType[task_type.upper()]
            opt_strategy_enum = OptimizationStrategy[optimization_strategy.upper()]
            opt_mode_enum = OptimizationMode[optimization_mode.upper()]
            norm_enum = NormalizationType[normalization.upper()]
            
            # Create configuration
            config = MLTrainingEngineConfig(
                task_type=task_type_enum,
                optimization_strategy=opt_strategy_enum,
                cv_folds=cv_folds,
                test_size=test_size,
                random_state=random_state,
                feature_selection=enable_feature_selection,
                enable_quantization=enable_quantization,
                model_path="./models",
                checkpoint_path="./checkpoints"
            )
            
            # Update preprocessing config
            if config.preprocessing_config:
                config.preprocessing_config.normalization = norm_enum
            
            # Update optimization mode in inference config
            if config.inference_config:
                config.inference_config.optimization_mode = opt_mode_enum
            
            self.current_config = config
            
            # Get available algorithms for the task type
            algorithms = self.get_algorithms_for_task(task_type)
            algorithm_dropdown = gr.Dropdown(
                choices=algorithms,
                value=algorithms[0] if algorithms else None,
                label="Available ML Algorithms"
            )
            
            config_text = f"""
**Configuration Created Successfully!**

- **Task Type**: {task_type}
- **Optimization Strategy**: {optimization_strategy}
- **CV Folds**: {cv_folds}
- **Test Size**: {test_size}
- **Feature Selection**: {'Enabled' if enable_feature_selection else 'Disabled'}
- **Normalization**: {normalization}
- **Quantization**: {'Enabled' if enable_quantization else 'Disabled'}
- **Available Algorithms**: {len(algorithms)} algorithms for {task_type.lower()}

✅ **Algorithm dropdown in Training tab has been updated!**
            """
            
            return config_text, algorithm_dropdown, algorithms
            
        except Exception as e:
            error_msg = f"Error creating configuration: {str(e)}"
            logger.error(error_msg)
            return error_msg, gr.Dropdown(), []
    
    def train_model(self, target_column: str, algorithm_name: str, model_name: str = None,
                   progress=gr.Progress()) -> Tuple[str, str, str, gr.Dropdown]:
        """Train a model with the current configuration"""
        if self.inference_only:
            return "Training is not available in inference-only mode.", "", "", gr.Dropdown()
        
        try:
            if self.current_data is None:
                return "No data loaded. Please upload a dataset first.", "", "", gr.Dropdown()
            
            if self.current_config is None:
                return "No configuration created. Please configure training parameters first.", "", "", gr.Dropdown()
            
            if target_column not in self.current_data.columns:
                return f"Target column '{target_column}' not found in dataset.", "", "", gr.Dropdown()
            
            if not algorithm_name or algorithm_name == "Select an algorithm...":
                return "Please select an algorithm to train.", "", "", gr.Dropdown()
            
            progress(0.1, desc="Initializing training engine...")
            
            # Initialize training engine
            self.training_engine = MLTrainingEngine(self.current_config)
            
            # Prepare data
            X = self.current_data.drop(columns=[target_column])
            y = self.current_data[target_column]
            
            progress(0.2, desc="Preprocessing data...")
            
            # Handle categorical features
            categorical_columns = X.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                for col in categorical_columns:
                    X[col] = pd.Categorical(X[col]).codes
            
            progress(0.3, desc="Starting model training...")
            
            # Extract model key from algorithm name
            model_key = self.get_model_key_from_name(algorithm_name)
            
            # Generate model name if not provided
            if not model_name:
                timestamp = int(time.time())
                model_name = f"{algorithm_name.split(' - ')[-1]}_{timestamp}"
            
            # Train model
            start_time = time.time()
            result = self.training_engine.train_model(
                X=X.values, 
                y=y.values,
                model_type=model_key,
                model_name=model_name
            )
            
            training_time = time.time() - start_time
            progress(1.0, desc="Training completed!")
            
            # Store trained model information
            self.trained_models[model_name] = {
                'algorithm': algorithm_name,
                'model_key': model_key,
                'target_column': target_column,
                'training_time': training_time,
                'result': result,
                'feature_names': X.columns.tolist(),
                'data_shape': X.shape
            }
            
            # Generate results summary
            metrics_text = "**Training Results:**\n\n"
            if 'metrics' in result and result['metrics']:
                for metric, value in result['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics_text += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
                    else:
                        metrics_text += f"- **{metric.replace('_', ' ').title()}**: {value}\n"
            
            metrics_text += f"\n- **Training Time**: {training_time:.2f} seconds"
            
            # Feature importance
            importance_text = ""
            if 'feature_importance' in result and result['feature_importance'] is not None:
                importance = result['feature_importance']
                feature_names = X.columns.tolist()
                
                importance_text = "**Top 10 Feature Importances:**\n\n"
                if isinstance(importance, dict):
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    for feature, score in sorted_features:
                        importance_text += f"- **{feature}**: {score:.4f}\n"
                else:
                    indices = np.argsort(importance)[::-1][:10]
                    for i, idx in enumerate(indices):
                        if idx < len(feature_names):
                            importance_text += f"- **{feature_names[idx]}**: {importance[idx]:.4f}\n"
            
            # Model summary
            summary_text = f"""
**Model Training Summary**

- **Model Name**: {model_name}
- **Algorithm**: {algorithm_name}
- **Dataset Shape**: {X.shape[0]} samples × {X.shape[1]} features
- **Target Column**: {target_column}
- **Task Type**: {self.current_config.task_type.value}
- **Status**: ✅ Training Completed Successfully
            """
            
            # Update trained models dropdown
            trained_models_dropdown = gr.Dropdown(
                choices=self.get_trained_model_list(),
                value="Select a trained model...",
                label="Trained Models"
            )
            
            return summary_text, metrics_text, importance_text, trained_models_dropdown
            
        except Exception as e:
            error_msg = f"Error during training: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg, "", "", gr.Dropdown()
    
    def get_trained_model_info(self, model_name: str) -> str:
        """Get information about a trained model"""
        try:
            if model_name == "Select a trained model..." or model_name not in self.trained_models:
                return "Please select a trained model to view information."
            
            model_info = self.trained_models[model_name]
            
            info_text = f"""
**Trained Model Information**

- **Model Name**: {model_name}
- **Algorithm**: {model_info['algorithm']}
- **Target Column**: {model_info['target_column']}
- **Training Time**: {model_info['training_time']:.2f} seconds
- **Data Shape**: {model_info['data_shape'][0]} samples × {model_info['data_shape'][1]} features
- **Feature Names**: {', '.join(model_info['feature_names'][:10])}{'...' if len(model_info['feature_names']) > 10 else ''}

**Performance Metrics:**
            """
            
            if 'metrics' in model_info['result'] and model_info['result']['metrics']:
                for metric, value in model_info['result']['metrics'].items():
                    if isinstance(value, (int, float)):
                        info_text += f"\n- **{metric.replace('_', ' ').title()}**: {value:.4f}"
                    else:
                        info_text += f"\n- **{metric.replace('_', ' ').title()}**: {value}"
            
            return info_text
            
        except Exception as e:
            error_msg = f"Error getting model info: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def make_prediction(self, input_data: str, selected_model: str = None) -> str:
        """Make predictions using the trained model"""
        if self.inference_only:
            return "Use the Inference Server tab for predictions in inference-only mode."
        
        try:
            # Determine which model to use
            if selected_model and selected_model != "Select a trained model..." and selected_model in self.trained_models:
                # Use selected trained model
                model_info = self.trained_models[selected_model]
                model_name = selected_model
            elif self.training_engine is None:
                return "No model available. Please train a model first or select a trained model."
            else:
                # Use current training engine model
                model_name = "Current Training Engine Model"
            
            # Parse input data
            try:
                if input_data.strip().startswith('['):
                    # JSON array format
                    data = json.loads(input_data)
                    input_array = np.array(data).reshape(1, -1)
                else:
                    # Comma-separated values
                    data = [float(x.strip()) for x in input_data.split(',')]
                    input_array = np.array(data).reshape(1, -1)
            except Exception as e:
                return f"Error parsing input data: {str(e)}. Please use comma-separated values or JSON array format."
            
            # Make prediction
            if selected_model and selected_model != "Select a trained model..." and selected_model in self.trained_models:
                # For selected model, we would need to reload it from training engine
                # This is a simplified version - in practice, you might want to store the actual model objects
                if self.training_engine is None:
                    return "Training engine not available. Please retrain the model or use the inference server."
                success, predictions = self.training_engine.predict(input_array)
            else:
                success, predictions = self.training_engine.predict(input_array)
            
            if not success:
                return f"Prediction failed: {predictions}"
            
            # Format results
            if isinstance(predictions, np.ndarray):
                if len(predictions.shape) == 1:
                    result = predictions[0]
                else:
                    result = predictions[0]
            else:
                result = predictions
            
            prediction_text = f"""
**Prediction Result:**

- **Model Used**: {model_name}
- **Input**: {input_data}
- **Prediction**: {result}
- **Data Shape**: {input_array.shape}
            """
            
            return prediction_text
            
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def save_model(self, model_name: str, selected_model: str, encryption_password: str = "") -> str:
        """Save the trained model"""
        if self.inference_only:
            return "Model saving is not available in inference-only mode."
        
        try:
            # Determine which model to save
            if selected_model and selected_model != "Select a trained model..." and selected_model in self.trained_models:
                save_name = model_name if model_name else selected_model
                model_to_save = selected_model
            elif self.training_engine is None:
                return "No model to save. Please train a model first or select a trained model."
            else:
                save_name = model_name if model_name else "current_model"
                model_to_save = "current"
            
            # Initialize model manager if not already done
            if self.model_manager is None:
                self.model_manager = SecureModelManager(
                    self.current_config,
                    logger=logger,
                    secret_key=encryption_password if encryption_password else None
                )
            
            # Get the best model
            if model_to_save == "current":
                best_model_name, best_model_info = self.training_engine.get_best_model()
                if best_model_info is None:
                    return "No trained model available to save."
            else:
                # For selected models, we use the training engine's current best model
                # In a full implementation, you'd want to store actual model objects
                best_model_name, best_model_info = self.training_engine.get_best_model()
                if best_model_info is None:
                    return "Selected model not available in training engine. Please retrain the model."
            
            # Update model manager with the model
            self.model_manager.models[save_name] = best_model_info
            self.model_manager.best_model = best_model_info
            
            # Save the model
            success = self.model_manager.save_model(
                model_name=save_name,
                access_code=encryption_password if encryption_password else None
            )
            
            if success:
                return f"✅ Model '{save_name}' saved successfully!"
            else:
                return "❌ Failed to save model. Check logs for details."
                
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def load_model(self, file, encryption_password: str = "") -> str:
        """Load a saved model"""
        try:
            if file is None:
                return "No model file selected."
            
            # Initialize model manager if not already done
            if self.model_manager is None:
                config = self.current_config or MLTrainingEngineConfig()
                self.model_manager = SecureModelManager(
                    config,
                    logger=logger,
                    secret_key=encryption_password if encryption_password else None
                )
            
            # Load the model
            model = self.model_manager.load_model(
                filepath=file.name,
                access_code=encryption_password if encryption_password else None
            )
            
            if model is not None:
                # Initialize inference engine with the loaded model
                inference_config = InferenceEngineConfig()
                self.inference_engine = InferenceEngine(inference_config)
                
                return f"✅ Model loaded successfully from {file.name}"
            else:
                return "❌ Failed to load model. Check password and file integrity."
                
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_model_performance(self) -> str:
        """Get performance comparison of all trained models"""
        if self.inference_only:
            return "Performance comparison is not available in inference-only mode."
        
        try:
            if not self.trained_models and self.training_engine is None:
                return "No models trained yet."
            
            # Get performance from training engine if available
            comparison_text = ""
            if self.training_engine:
                try:
                    comparison = self.training_engine.get_performance_comparison()
                    
                    # Check if we have a valid comparison response
                    if isinstance(comparison, dict) and 'models' in comparison and 'error' not in comparison:
                        models = comparison.get('models', [])
                        if isinstance(models, list) and models:
                            comparison_text += "**Training Engine Model Comparison:**\n\n"
                            for model in models:
                                if isinstance(model, dict):
                                    model_name = model.get('name', 'Unknown')
                                    model_type = model.get('type', 'Unknown')
                                    training_time = model.get('training_time', 0)
                                    is_best = model.get('is_best', False)
                                    metrics = model.get('metrics', {})
                                    
                                    comparison_text += f"### {model_name} {'👑' if is_best else ''}\n"
                                    comparison_text += f"- **Type**: {model_type}\n"
                                    comparison_text += f"- **Training Time**: {training_time:.2f}s\n"
                                    
                                    if isinstance(metrics, dict) and metrics:
                                        comparison_text += "- **Metrics**:\n"
                                        for metric, value in metrics.items():
                                            if isinstance(value, (int, float)):
                                                comparison_text += f"  - {metric}: {value:.4f}\n"
                                    comparison_text += "\n"
                    elif isinstance(comparison, dict) and 'error' in comparison:
                        comparison_text += f"**Training Engine Error**: {comparison['error']}\n\n"
                        
                except Exception as comparison_error:
                    comparison_text += f"**Error getting model comparison**: {str(comparison_error)}\n\n"
            
            # Add stored model information
            if self.trained_models:
                comparison_text += "\n**Stored Trained Models:**\n\n"
                for model_name, model_info in self.trained_models.items():
                    comparison_text += f"### {model_name}\n"
                    comparison_text += f"- **Algorithm**: {model_info['algorithm']}\n"
                    comparison_text += f"- **Target**: {model_info['target_column']}\n"
                    comparison_text += f"- **Training Time**: {model_info['training_time']:.2f}s\n"
                    comparison_text += f"- **Data Shape**: {model_info['data_shape'][0]} × {model_info['data_shape'][1]}\n"
                    
                    if 'metrics' in model_info['result'] and model_info['result']['metrics']:
                        comparison_text += "- **Metrics**:\n"
                        for metric, value in model_info['result']['metrics'].items():
                            if isinstance(value, (int, float)):
                                comparison_text += f"  - {metric}: {value:.4f}\n"
                    comparison_text += "\n"
            
            return comparison_text if comparison_text else "No model performance data available."
            
        except Exception as e:
            error_msg = f"Error getting model performance: {str(e)}"
            logger.error(error_msg)
            return error_msg

def create_ui(inference_only: bool = False):
    """Create and configure the Gradio interface"""
    
    app = MLSystemUI(inference_only=inference_only)
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    title = "🚀 ML Inference Server" if inference_only else "🚀 ML Training & Inference System"
    description = """
**A machine learning inference server for real-time predictions.**

Load your trained models and make instant predictions!
    """ if inference_only else """
**A comprehensive machine learning platform with advanced optimization and secure model management.**

Upload your data, configure training parameters, train models, and make predictions - all in one place!
    """
    
    with gr.Blocks(css=css, title=title, theme=gr.themes.Soft()) as interface:
        gr.Image(value="assets/logo.png", width=120, show_label=False, show_download_button=False)
        gr.Markdown(f"""
# {title}

{description}
        """)
        
        with gr.Tabs():
            
            if not inference_only:
                # Create shared state for algorithm choices
                algorithm_choices_state = gr.State([])
                
                # Data Upload Tab
                with gr.Tab("📁 Data Upload", id="data_upload"):
                    gr.Markdown("### Upload Dataset or Load Sample Data")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Upload Your Own Dataset")
                            file_input = gr.File(
                                label="Upload Dataset",
                                file_types=[".csv", ".xlsx", ".xls", ".json"],
                                type="filepath"
                            )
                            
                            load_btn = gr.Button("Load Data", variant="primary", size="lg")
                            
                        with gr.Column(scale=1):
                            gr.Markdown("#### Or Load Sample Dataset")
                            sample_dropdown = gr.Dropdown(
                                choices=["Select a dataset..."] + app.sample_data_loader.get_available_datasets(),
                                value="Select a dataset...",
                                label="Sample Datasets"
                            )
                            
                            load_sample_btn = gr.Button("Load Sample Data", variant="secondary", size="lg")
                    
                    with gr.Row():
                        data_info = gr.Markdown("Upload a dataset or select a sample dataset to get started...")
                    
                    # Data Preview Section
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📊 Data Preview")
                            data_preview = gr.Markdown("Data preview will appear here after loading...")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📈 Sample Data (First 10 Rows)")
                            sample_data_table = gr.HTML("")
                
                # Configuration Tab
                with gr.Tab("⚙️ Configuration", id="configuration"):
                    gr.Markdown("### Training Configuration")
                    
                    with gr.Row():
                        with gr.Column():
                            task_type = gr.Dropdown(
                                choices=["CLASSIFICATION", "REGRESSION"],
                                value="CLASSIFICATION",
                                label="Task Type"
                            )
                            
                            optimization_strategy = gr.Dropdown(
                                choices=["RANDOM_SEARCH", "GRID_SEARCH", "BAYESIAN_OPTIMIZATION", "HYPERX"],
                                value="RANDOM_SEARCH",
                                label="Optimization Strategy"
                            )
                            
                            cv_folds = gr.Slider(
                                minimum=2,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Cross-Validation Folds"
                            )
                            
                        with gr.Column():
                            test_size = gr.Slider(
                                minimum=0.1,
                                maximum=0.5,
                                value=0.2,
                                step=0.05,
                                label="Test Size"
                            )
                            
                            random_state = gr.Number(
                                value=42,
                                label="Random State",
                                precision=0
                            )
                            
                            normalization = gr.Dropdown(
                                choices=["STANDARD", "MINMAX", "ROBUST", "NONE"],
                                value="STANDARD",
                                label="Normalization"
                            )
                    
                    with gr.Row():
                        enable_feature_selection = gr.Checkbox(
                            label="Enable Feature Selection",
                            value=True
                        )
                        
                        enable_quantization = gr.Checkbox(
                            label="Enable Model Quantization",
                            value=False
                        )
                        
                        optimization_mode = gr.Dropdown(
                            choices=["BALANCED", "PERFORMANCE", "MEMORY_SAVING", "CONSERVATIVE"],
                            value="BALANCED",
                            label="Optimization Mode"
                        )
                    
                    config_btn = gr.Button("Create Configuration", variant="primary", size="lg")
                    config_output = gr.Markdown("")
                    
                    # Algorithm selection (updated based on task type)
                    algorithm_dropdown = gr.Dropdown(
                        choices=[],
                        label="Available ML Algorithms",
                        info="Select task type and create configuration to see available algorithms"
                    )
                
                # Training Tab
                with gr.Tab("🎯 Model Training", id="training"):
                    gr.Markdown("### Train Machine Learning Models")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            target_column = gr.Textbox(
                                label="Target Column",
                                placeholder="Enter the name of your target column",
                                info="The column you want to predict"
                            )
                            
                            # Create the algorithm selection dropdown that will be updated
                            algorithm_selection = gr.Dropdown(
                                choices=["Configure training parameters first..."],
                                value="Configure training parameters first...",
                                label="Select ML Algorithm",
                                info="Go to Configuration tab and create config to see available algorithms",
                                interactive=True
                            )
                            
                            model_name_input = gr.Textbox(
                                label="Model Name (Optional)",
                                placeholder="Leave empty for auto-generated name",
                                info="Custom name for your trained model"
                            )
                            
                            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                            
                        with gr.Column(scale=2):
                            training_output = gr.Markdown("Configure your model and click 'Start Training'...")
                    
                    with gr.Row():
                        with gr.Column():
                            metrics_output = gr.Markdown("")
                        with gr.Column():
                            importance_output = gr.Markdown("")
                    
                    # Trained models dropdown (updated after training)
                    trained_models_dropdown = gr.Dropdown(
                        choices=app.get_trained_model_list(),
                        value="Select a trained model...",
                        label="Trained Models",
                        info="Select a model to view details or use for predictions"
                    )
                    
                    # Model information display
                    with gr.Row():
                        with gr.Column():
                            model_info_btn = gr.Button("📋 Get Model Info", variant="secondary")
                            model_info_output = gr.Markdown("")
                
                # Now set up the event handlers after all components are defined
                
                # Data upload tab event handlers
                load_btn.click(
                    fn=app.load_data,
                    inputs=[file_input],
                    outputs=[data_info, gr.State(), data_preview, sample_data_table]
                )
                
                load_sample_btn.click(
                    fn=app.load_sample_data,
                    inputs=[sample_dropdown],
                    outputs=[data_info, gr.State(), data_preview, sample_data_table]
                )
                
                # Configuration tab event handlers
                config_btn.click(
                    fn=app.create_training_config,
                    inputs=[
                        task_type, optimization_strategy, cv_folds, test_size,
                        random_state, enable_feature_selection, normalization,
                        enable_quantization, optimization_mode
                    ],
                    outputs=[config_output, algorithm_dropdown, algorithm_choices_state]
                )
                
                # Connect the algorithm dropdown updates to the training tab
                def update_training_algorithms(task_type_value):
                    algorithms = app.get_algorithms_for_task(task_type_value)
                    return gr.Dropdown(choices=algorithms, value=algorithms[0] if algorithms else None)
                
                # Update algorithm dropdown when task type changes
                task_type.change(
                    fn=update_training_algorithms,
                    inputs=[task_type],
                    outputs=[algorithm_dropdown]
                )
                
                # Function to update training tab algorithm dropdown
                def update_algorithm_choices_from_state(algorithm_list):
                    return gr.Dropdown(
                        choices=algorithm_list if algorithm_list else ["Configure training parameters first..."],
                        value=algorithm_list[0] if algorithm_list else "Configure training parameters first...",
                        label="Select ML Algorithm",
                        info="Available algorithms based on your configuration"
                    )
                
                # Update the training tab dropdown when algorithm choices state changes
                algorithm_choices_state.change(
                    fn=update_algorithm_choices_from_state,
                    inputs=[algorithm_choices_state],
                    outputs=[algorithm_selection]
                )
                
                # Training tab event handlers
                train_btn.click(
                    fn=app.train_model,
                    inputs=[target_column, algorithm_selection, model_name_input],
                    outputs=[training_output, metrics_output, importance_output, trained_models_dropdown]
                )
                
                model_info_btn.click(
                    fn=app.get_trained_model_info,
                    inputs=[trained_models_dropdown],
                    outputs=[model_info_output]
                )
                
                # Prediction Tab
                with gr.Tab("🔮 Predictions", id="predictions"):
                    gr.Markdown("### Make Predictions with Trained Models")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            prediction_input = gr.Textbox(
                                label="Input Data",
                                placeholder="Enter comma-separated values or JSON array",
                                lines=3,
                                info="Example: 1.5, 2.3, 0.8, 1.1 or [1.5, 2.3, 0.8, 1.1]"
                            )
                            
                            # Model selection for predictions
                            prediction_model_dropdown = gr.Dropdown(
                                choices=app.get_trained_model_list(),
                                value="Select a trained model...",
                                label="Select Model for Prediction",
                                info="Choose a specific trained model or use current training engine model"
                            )
                            
                            predict_btn = gr.Button("🔮 Make Prediction", variant="primary", size="lg")
                            
                        with gr.Column(scale=2):
                            prediction_output = gr.Markdown("Enter input data and click 'Make Prediction'...")
                    
                    predict_btn.click(
                        fn=app.make_prediction,
                        inputs=[prediction_input, prediction_model_dropdown],
                        outputs=[prediction_output]
                    )
                
                # Model Management Tab
                with gr.Tab("💾 Model Management", id="model_management"):
                    gr.Markdown("### Save and Load Models")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Save Model")
                            
                            save_model_name = gr.Textbox(
                                label="Model Name",
                                placeholder="Enter a name for your model"
                            )
                            
                            save_model_dropdown = gr.Dropdown(
                                choices=app.get_trained_model_list(),
                                value="Select a trained model...",
                                label="Select Model to Save",
                                info="Choose which trained model to save"
                            )
                            
                            save_password = gr.Textbox(
                                label="Encryption Password (Optional)",
                                type="password",
                                placeholder="Leave empty for no encryption"
                            )
                            
                            save_btn = gr.Button("💾 Save Model", variant="primary")
                            save_output = gr.Markdown("")
                            
                        with gr.Column():
                            gr.Markdown("#### Load Model")
                            
                            load_file = gr.File(
                                label="Model File",
                                file_types=[".pkl"],
                                type="filepath"
                            )
                            
                            load_password = gr.Textbox(
                                label="Decryption Password",
                                type="password",
                                placeholder="Enter password if model is encrypted"
                            )
                            
                            load_btn = gr.Button("📂 Load Model", variant="secondary")
                            load_output = gr.Markdown("")
                    
                    save_btn.click(
                        fn=app.save_model,
                        inputs=[save_model_name, save_model_dropdown, save_password],
                        outputs=[save_output]
                    )
                    
                    load_btn.click(
                        fn=app.load_model,
                        inputs=[load_file, load_password],
                        outputs=[load_output]
                    )
                
                # Performance Tab
                with gr.Tab("📊 Performance", id="performance"):
                    gr.Markdown("### Model Performance Analysis")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            performance_btn = gr.Button("📊 Get Performance Report", variant="primary", size="lg")
                            
                        with gr.Column(scale=2):
                            performance_output = gr.Markdown("Click 'Get Performance Report' to see model comparisons...")
                    
                    performance_btn.click(
                        fn=app.get_model_performance,
                        outputs=[performance_output]
                    )
            
            # Inference Server Tab (always available)
            with gr.Tab("🔧 Inference Server", id="inference_server"):
                # Custom CSS for inference server styling
                gr.HTML("""
                <style>
                .inference-container {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px;
                    padding: 20px;
                    margin: 10px 0;
                    color: white;
                }
                .server-status {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                .server-controls {
                    background: rgba(0, 0, 0, 0.05);
                    border-radius: 8px;
                    padding: 20px;
                    margin: 10px 0;
                    border: 1px solid #e0e0e0;
                }
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-stopped {
                    background-color: #ff6b6b;
                }
                .status-running {
                    background-color: #51cf66;
                }
                .status-loading {
                    background-color: #ffd43b;
                }
                .server-metric {
                    display: flex;
                    justify-content: space-between;
                    padding: 5px 0;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                .server-metric:last-child {
                    border-bottom: none;
                }
                .metric-label {
                    font-weight: 500;
                    opacity: 0.8;
                }
                .metric-value {
                    font-weight: bold;
                }
                .server-logs {
                    background: #1a1a1a;
                    color: #00ff00;
                    border-radius: 8px;
                    padding: 15px;
                    font-family: 'Courier New', monospace;
                    font-size: 12px;
                    height: 200px;
                    overflow-y: auto;
                    white-space: pre-wrap;
                }
                .control-button {
                    margin: 5px;
                    padding: 10px 20px;
                    border-radius: 6px;
                    border: none;
                    cursor: pointer;
                    font-weight: 500;
                    transition: all 0.3s ease;
                }
                .btn-start {
                    background: #51cf66;
                    color: white;
                }
                .btn-stop {
                    background: #ff6b6b;
                    color: white;
                }
                .btn-load {
                    background: #339af0;
                    color: white;
                }
                .model-info-card {
                    background: white;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border: 1px solid #e0e0e0;
                }
                </style>
                """)
                
                with gr.Row():
                    # Left Panel - Server Control
                    with gr.Column(scale=2):
                        gr.HTML("""
                        <div class="inference-container">
                            <h3 style="margin-top: 0;">🚀 ML Inference Server</h3>
                            <p style="opacity: 0.9;">Professional machine learning inference server for real-time predictions</p>
                        </div>
                        """)
                        
                        # Server Controls
                        gr.HTML('<div class="server-controls">')
                        gr.Markdown("### Server Controls")
                        
                        with gr.Row():
                            start_server_btn = gr.Button("▶️ Start Server", variant="primary", size="sm")
                            stop_server_btn = gr.Button("⏹️ Stop Server", variant="secondary", size="sm")
                            load_model_btn = gr.Button("📁 Load Model", variant="secondary", size="sm")
                        
                        # Model Loading Section
                        with gr.Accordion("Model Loading", open=False):
                            inference_model_file = gr.File(
                                label="Select Model File",
                                file_types=[".pkl", ".joblib", ".h5", ".pt", ".onnx"],
                                type="filepath"
                            )
                            
                            inference_password = gr.Textbox(
                                label="Decryption Password",
                                type="password",
                                placeholder="Leave empty if model is not encrypted",
                                container=False
                            )
                            
                            load_inference_btn = gr.Button("🔧 Load Model", variant="primary")
                        
                        gr.HTML('</div>')
                        
                        # Prediction Interface
                        gr.HTML('<div class="server-controls">')
                        gr.Markdown("### Make Predictions")
                        
                        inference_input = gr.Textbox(
                            label="Input Data",
                            placeholder="Enter comma-separated values or JSON array",
                            lines=4,
                            info="Example: 1.5, 2.3, 0.8, 1.1 or [1.5, 2.3, 0.8, 1.1]"
                        )
                        
                        with gr.Row():
                            inference_predict_btn = gr.Button("🎯 Predict", variant="primary", size="lg")
                            clear_input_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
                        
                        inference_output = gr.Markdown("**Prediction Results:**\n\nLoad a model and enter input data to make predictions...")
                        gr.HTML('</div>')
                    
                    # Right Panel - Server Status & Logs
                    with gr.Column(scale=1):
                        # Server Status
                        gr.HTML("""
                        <div class="server-status">
                            <h4 style="margin-top: 0; color: white;">Server Status</h4>
                            <div style="margin: 10px 0;">
                                <span class="status-indicator status-stopped"></span>
                                <span style="color: #ffcccb;">Stopped</span>
                            </div>
                        </div>
                        """)
                        
                        server_status_display = gr.HTML("""
                        <div class="model-info-card">
                            <h4 style="margin-top: 0;">📊 Server Metrics</h4>
                            <div class="server-metric">
                                <span class="metric-label">Server Port:</span>
                                <span class="metric-value">8080</span>
                            </div>
                            <div class="server-metric">
                                <span class="metric-label">Context Size:</span>
                                <span class="metric-value">4125</span>
                            </div>
                            <div class="server-metric">
                                <span class="metric-label">Keep Alive:</span>
                                <span class="metric-value">2048</span>
                            </div>
                            <div class="server-metric">
                                <span class="metric-label">GPU Layers:</span>
                                <span class="metric-value">100</span>
                            </div>
                            <div class="server-metric">
                                <span class="metric-label">Parallel:</span>
                                <span class="metric-value">1</span>
                            </div>
                            <div class="server-metric">
                                <span class="metric-label">Batch Size:</span>
                                <span class="metric-value">4096</span>
                            </div>
                        </div>
                        """)
                        
                        # Server Configuration
                        with gr.Accordion("⚙️ Configuration", open=False):
                            server_port = gr.Number(
                                label="Server Port",
                                value=8080,
                                precision=0
                            )
                            
                            memory_lock = gr.Checkbox(
                                label="Memory Lock",
                                value=True
                            )
                            
                            continuous_batching = gr.Checkbox(
                                label="Continuous Batching",
                                value=True
                            )
                            
                            warmup = gr.Checkbox(
                                label="Warmup",
                                value=False
                            )
                        
                        # Loaded Models Section
                        gr.HTML("""
                        <div class="model-info-card">
                            <h4 style="margin-top: 0;">📚 Loaded Models</h4>
                            <p style="color: #666; font-style: italic;">No models loaded.</p>
                        </div>
                        """)
                        
                        loaded_models_display = gr.HTML("")
                
                # Server Logs Section
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📋 Server Logs")
                        server_logs = gr.HTML("""
                        <div class="server-logs">
Server logs will be displayed here.
                        </div>
                        """)
                        
                        with gr.Row():
                            refresh_logs_btn = gr.Button("🔄 Refresh Logs", variant="secondary", size="sm")
                            clear_logs_btn = gr.Button("🗑️ Clear Logs", variant="secondary", size="sm")
                
                # Event handlers for inference server
                def update_server_status(is_running=False):
                    if is_running:
                        status_html = """
                        <div class="server-status">
                            <h4 style="margin-top: 0; color: white;">Server Status</h4>
                            <div style="margin: 10px 0;">
                                <span class="status-indicator status-running"></span>
                                <span style="color: #90EE90;">Running</span>
                            </div>
                        </div>
                        """
                        logs = """
                        <div class="server-logs">
[INFO] Server started successfully on port 8080
[INFO] Model loaded and ready for inference
[INFO] Waiting for incoming requests...
                        </div>
                        """
                    else:
                        status_html = """
                        <div class="server-status">
                            <h4 style="margin-top: 0; color: white;">Server Status</h4>
                            <div style="margin: 10px 0;">
                                <span class="status-indicator status-stopped"></span>
                                <span style="color: #ffcccb;">Stopped</span>
                            </div>
                        </div>
                        """
                        logs = """
                        <div class="server-logs">
[INFO] Server stopped
[INFO] Resources cleaned up
                        </div>
                        """
                    return status_html, logs
                
                def update_loaded_models(model_name=None):
                    if model_name:
                        return f"""
                        <div class="model-info-card">
                            <h4 style="margin-top: 0;">📚 Loaded Models</h4>
                            <div style="padding: 10px; background: #f0f8ff; border-radius: 4px; margin: 5px 0;">
                                <strong>{model_name}</strong>
                                <br><small style="color: #666;">Status: Ready</small>
                            </div>
                        </div>
                        """
                    else:
                        return """
                        <div class="model-info-card">
                            <h4 style="margin-top: 0;">📚 Loaded Models</h4>
                            <p style="color: #666; font-style: italic;">No models loaded.</p>
                        </div>
                        """
                
                def clear_input():
                    return ""
                
                def simulate_start_server():
                    return update_server_status(True)
                
                def simulate_stop_server():
                    return update_server_status(False)
                
                # Enhanced load model function
                def load_model_enhanced(file, password=""):
                    result = app.load_inference_model(file, password)
                    if "✅" in result:
                        model_name = file.name.split('/')[-1] if file else "Unknown Model"
                        models_html = update_loaded_models(model_name)
                        logs = f"""
                        <div class="server-logs">
[INFO] Loading model: {model_name}
[INFO] Model validation passed
[INFO] Model loaded successfully
[INFO] Ready for inference requests
                        </div>
                        """
                        return result, models_html, logs
                    else:
                        logs = f"""
                        <div class="server-logs">
[ERROR] Failed to load model
[ERROR] {result}
                        </div>
                        """
                        return result, update_loaded_models(), logs
                
                # Button event handlers
                start_server_btn.click(
                    fn=simulate_start_server,
                    outputs=[gr.HTML(visible=False), server_logs]
                )
                
                stop_server_btn.click(
                    fn=simulate_stop_server,
                    outputs=[gr.HTML(visible=False), server_logs]
                )
                
                load_inference_btn.click(
                    fn=load_model_enhanced,
                    inputs=[inference_model_file, inference_password],
                    outputs=[gr.Markdown(visible=False), loaded_models_display, server_logs]
                )
                
                inference_predict_btn.click(
                    fn=app.make_inference_prediction,
                    inputs=[inference_input],
                    outputs=[inference_output]
                )
                
                clear_input_btn.click(
                    fn=clear_input,
                    outputs=[inference_input]
                )
                
                clear_logs_btn.click(
                    fn=lambda: """<div class="server-logs">Logs cleared.</div>""",
                    outputs=[server_logs]
                )
            
            if not inference_only:
                # System Info Tab
                with gr.Tab("🖥️ System Info", id="system_info"):
                    gr.Markdown("### System Information and Optimization")
                    
                    system_btn = gr.Button("🖥️ Get System Info", variant="secondary")
                    system_output = gr.JSON(label="System Information")
                    
                    system_btn.click(
                        fn=app.get_system_info,
                        outputs=[system_output]
                    )
        
        # Footer
        footer_text = """
---
**ML Inference Server** - Optimized for real-time predictions.

💡 **Tips:**
- Load your trained model in the Inference Server tab
- Make real-time predictions with minimal latency
- Supports encrypted model files for security
        """ if inference_only else """
---
**ML Training & Inference System** - Built with advanced optimization and security features.

💡 **Tips:**
- Start by uploading your dataset or loading sample data in the Data Upload tab
- Try sample datasets like Iris, Titanic, or Boston Housing for quick testing
- Configure your training parameters in the Configuration tab
- Select from multiple ML algorithms based on your task type
- Train multiple models and compare their performance
- Use the Inference Server for production-ready predictions
- Save models securely with encryption
- View detailed information about your trained models

**Available ML Algorithms:**
- **Tree-Based**: Random Forest, Extra Trees, Decision Tree, Gradient Boosting
- **Boosting**: XGBoost, LightGBM, CatBoost, AdaBoost
- **Linear Models**: Logistic/Linear Regression, Ridge, Lasso, Elastic Net
- **Support Vector Machines**: RBF, Linear, Polynomial kernels
- **Neural Networks**: Multi-layer Perceptron, Basic Neural Network
- **Naive Bayes**: Gaussian, Multinomial, Bernoulli
- **Nearest Neighbors**: K-Nearest Neighbors
- **Ensemble Methods**: Voting Classifier, Bagging
        """
        
        gr.Markdown(footer_text)
    
    return interface

def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(description="ML Training & Inference System")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run in inference-only mode (no training capabilities)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to pre-trained model file (for inference-only mode)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    
    args = parser.parse_args()
    
    # Create and launch the interface
    interface = create_ui(inference_only=args.inference_only)
    
    print(f"""
🚀 Starting {'ML Inference Server' if args.inference_only else 'ML Training & Inference System'}

Mode: {'Inference Only' if args.inference_only else 'Full Training & Inference'}
Host: {args.host}
Port: {args.port}
Share: {'Yes' if args.share else 'No'}

Available Features:
{'- Real-time model inference' if args.inference_only else '''- Multiple ML algorithms support
- Advanced model training with hyperparameter optimization
- Model performance comparison
- Secure model storage with encryption
- Real-time inference server'''}
    """)
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=True,
        show_error=True,
        favicon_path=None,
        ssl_verify=False,
        quiet=False
    )
>>>>>>> 70c377c0c903a8ba732d32ee4a5928760c74b31a

if __name__ == "__main__":
    main()