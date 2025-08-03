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
    load_diabetes, make_classification, make_regression,
    fetch_openml, fetch_california_housing
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
            
        elif dataset_name == "california_housing":
            data = fetch_california_housing()
            X, y = data.data, data.target
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

if __name__ == "__main__":
    main()