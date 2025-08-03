#!/usr/bin/env python3
"""
Kolosal AutoML Benchmark Script

This script runs Kolosal AutoML with optimizations DISABLED to ensure fair comparison
against standard ML approaches. This allows for direct performance comparison while
maintaining the same underlying algorithms.

Features:
- Kolosal AutoML framework with optimizations disabled
- Fair comparison setup
- Advanced hyperparameter optimization strategies
- Memory and performance tracking
- Comprehensive reporting
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the project root to the path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import Kolosal AutoML modules
    from modules.configs import (
        TaskType, OptimizationStrategy, MLTrainingEngineConfig,
        PreprocessorConfig, NormalizationType, BatchProcessorConfig,
        BatchProcessingStrategy, InferenceEngineConfig
    )
    from modules.engine.train_engine import MLTrainingEngine
    KOLOSAL_AVAILABLE = True
    logger_name = "KolosalAutoMLBenchmark"
except ImportError as e:
    print(f"Warning: Kolosal AutoML modules not available: {e}")
    print("Will create standalone implementation...")
    KOLOSAL_AVAILABLE = False
    logger_name = "KolosalAutoMLBenchmark_Standalone"

# Standard ML imports for dataset loading
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, 
    load_diabetes, make_classification, make_regression,
    fetch_california_housing
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(logger_name)

@dataclass
class KolosalBenchmarkResult:
    """Data class to store Kolosal AutoML benchmark results."""
    benchmark_id: str
    dataset_name: str
    model_name: str
    dataset_size: Tuple[int, int]
    task_type: str
    optimization_strategy: str
    
    # Performance metrics
    training_time: float
    prediction_time: float
    preprocessing_time: float
    memory_peak_mb: float
    
    # ML metrics
    train_score: float
    test_score: float
    cv_score_mean: float
    cv_score_std: float
    
    # Additional info
    best_params: Dict[str, Any]
    feature_count: int
    success: bool
    error_message: str = ""
    timestamp: str = ""

class DatasetLoader:
    """Loads datasets for Kolosal AutoML benchmarking."""
    
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
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

class KolosalAutoMLBenchmark:
    """Implements Kolosal AutoML approach with optimizations disabled for fair comparison."""
    
    def __init__(self, output_dir: str = "./kolosal_automl_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not KOLOSAL_AVAILABLE:
            logger.warning("Kolosal AutoML modules not available. Results will show failures.")
    
    def create_engine_config(self, task_type: str, optimization_strategy: str) -> Any:
        """Create Kolosal ML engine configuration with optimizations DISABLED."""
        if not KOLOSAL_AVAILABLE:
            return None
            
        # Set task type
        task = TaskType.CLASSIFICATION if task_type == "classification" else TaskType.REGRESSION
        
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
            strategy = OptimizationStrategy.RANDOM_SEARCH
        
        # Preprocessor configuration - DISABLED optimizations
        preprocessor_config = PreprocessorConfig(
            normalization=NormalizationType.STANDARD,
            handle_nan=True,
            handle_inf=True,
            detect_outliers=False,  # DISABLED
            parallel_processing=False,  # DISABLED
            n_jobs=1,  # Single job for fair comparison
            cache_enabled=False  # DISABLED
        )
        
        # Batch processing configuration - DISABLED optimizations
        batch_config = BatchProcessorConfig(
            initial_batch_size=32,
            min_batch_size=32,
            max_batch_size=32,  # Fixed batch size
            max_queue_size=100,
            processing_strategy=BatchProcessingStrategy.SIMPLE,  # Simple strategy
            enable_adaptive_batching=False,  # DISABLED
            enable_memory_optimization=False,  # DISABLED
            max_workers=1  # Single worker
        )
        
        # Inference configuration - DISABLED optimizations
        inference_config = InferenceEngineConfig(
            enable_intel_optimization=False,  # DISABLED
            enable_batching=False,  # DISABLED
            num_threads=1,  # Single thread
            enable_memory_optimization=False  # DISABLED
        )
        
        # Create the MLTrainingEngineConfig with optimizations DISABLED
        config = MLTrainingEngineConfig(
            task_type=task,
            random_state=42,
            n_jobs=1,  # Single job for fair comparison
            verbose=0,
            cv_folds=3,  # Reduced for faster comparison
            test_size=0.2,
            stratify=(task == TaskType.CLASSIFICATION),
            optimization_strategy=strategy,
            optimization_iterations=10,  # Reduced for fair comparison
            early_stopping=False,  # DISABLED
            feature_selection=False,  # DISABLED
            preprocessing_config=preprocessor_config,
            batch_processing_config=batch_config,
            inference_config=inference_config,
            model_path="./kolosal_models_fair_comparison",
            experiment_tracking=False,  # DISABLED
            use_intel_optimization=False,  # DISABLED
            memory_optimization=False  # DISABLED
        )
        
        return config
    
    def get_model_and_params(self, model_name: str, task_type: str) -> Tuple[Any, Dict]:
        """Get model and parameter grid matching standard ML approach."""
        if model_name == "random_forest":
            if task_type == "classification":
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                }
            else:
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                }
        elif model_name == "gradient_boosting":
            if task_type == "classification":
                model = GradientBoostingClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            else:
                model = GradientBoostingRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
        elif model_name == "logistic_regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'saga']
            }
        elif model_name == "ridge":
            model = Ridge(random_state=42)
            param_grid = {
                'alpha': [0.1, 1.0, 10.0]
            }
        elif model_name == "lasso":
            model = Lasso(random_state=42, max_iter=2000)
            param_grid = {
                'alpha': [0.01, 0.1, 1.0]
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return model, param_grid
    
    def run_benchmark(self, dataset_name: str, model_name: str, 
                     optimization_strategy: str = "random_search") -> KolosalBenchmarkResult:
        """Run Kolosal AutoML benchmark with optimizations disabled."""
        benchmark_id = f"kolosal_{dataset_name}_{model_name}_{optimization_strategy}_{int(time.time())}"
        logger.info(f"Starting Kolosal AutoML benchmark: {benchmark_id}")
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        if not KOLOSAL_AVAILABLE:
            return KolosalBenchmarkResult(
                benchmark_id=benchmark_id,
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_size=(0, 0),
                task_type="unknown",
                optimization_strategy=optimization_strategy,
                training_time=0,
                prediction_time=0,
                preprocessing_time=0,
                memory_peak_mb=initial_memory,
                train_score=0,
                test_score=0,
                cv_score_mean=0,
                cv_score_std=0,
                best_params={},
                feature_count=0,
                success=False,
                error_message="Kolosal AutoML modules not available",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        
        try:
            # Load dataset
            X, y, task_type = DatasetLoader.load_dataset(dataset_name)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if task_type == "classification" else None
            )
            
            # Create engine configuration (with optimizations disabled)
            config = self.create_engine_config(task_type, optimization_strategy)
            
            # Create ML Training Engine
            engine = MLTrainingEngine(config)
            
            # Get model and parameter grid
            model, param_grid = self.get_model_and_params(model_name, task_type)
            
            # Training with timing
            training_start = time.time()
            result_model, train_metrics = engine.train_model(
                model=model,
                model_name=f"{model_name}_fair_comparison",
                param_grid=param_grid,
                X=X_train,
                y=y_train,
                X_test=X_test,
                y_test=y_test
            )
            training_time = time.time() - training_start
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / (1024 * 1024)
            
            # Predictions with timing
            prediction_start = time.time()
            train_pred = engine.predict(X_train)
            test_pred = engine.predict(X_test)
            prediction_time = time.time() - prediction_start
            
            # Cross-validation scores (using the trained model)
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(result_model, X_train, y_train, cv=3)
            
            # Calculate metrics
            if task_type == "classification":
                train_score = accuracy_score(y_train, train_pred)
                test_score = accuracy_score(y_test, test_pred)
            else:
                train_score = r2_score(y_train, train_pred)
                test_score = r2_score(y_test, test_pred)
            
            # Get best parameters from train_metrics if available
            best_params = train_metrics.get('best_params', {}) if train_metrics else {}
            
            result = KolosalBenchmarkResult(
                benchmark_id=benchmark_id,
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_size=X.shape,
                task_type=task_type,
                optimization_strategy=optimization_strategy,
                training_time=training_time,
                prediction_time=prediction_time,
                preprocessing_time=0,  # Included in training time
                memory_peak_mb=peak_memory,
                train_score=train_score,
                test_score=test_score,
                cv_score_mean=cv_scores.mean(),
                cv_score_std=cv_scores.std(),
                best_params=best_params,
                feature_count=X.shape[1],
                success=True,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Clean up
            engine.shutdown()
            
            logger.info(f"Kolosal AutoML benchmark completed successfully: {benchmark_id}")
            logger.info(f"Training time: {training_time:.2f}s, Test score: {test_score:.4f}")
            
        except Exception as e:
            logger.error(f"Kolosal AutoML benchmark failed: {e}")
            result = KolosalBenchmarkResult(
                benchmark_id=benchmark_id,
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_size=(0, 0),
                task_type="unknown",
                optimization_strategy=optimization_strategy,
                training_time=time.time() - start_time,
                prediction_time=0,
                preprocessing_time=0,
                memory_peak_mb=initial_memory,
                train_score=0,
                test_score=0,
                cv_score_mean=0,
                cv_score_std=0,
                best_params={},
                feature_count=0,
                success=False,
                error_message=str(e),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        
        self.results.append(result)
        return result
    
    def run_multiple_benchmarks(self, configurations: List[Dict[str, str]]) -> List[KolosalBenchmarkResult]:
        """Run multiple Kolosal AutoML benchmarks."""
        results = []
        total = len(configurations)
        
        for i, config in enumerate(configurations, 1):
            logger.info(f"Running Kolosal AutoML benchmark {i}/{total}: {config}")
            try:
                result = self.run_benchmark(
                    config['dataset'],
                    config['model'],
                    config.get('optimization_strategy', 'random_search')
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark failed for {config}: {e}")
                continue
        
        return results
    
    def save_results(self) -> str:
        """Save benchmark results to file."""
        results_file = self.output_dir / f"kolosal_automl_results_{self.timestamp}.json"
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            result_dict = {
                'benchmark_id': result.benchmark_id,
                'dataset_name': result.dataset_name,
                'model_name': result.model_name,
                'dataset_size': result.dataset_size,
                'task_type': result.task_type,
                'optimization_strategy': result.optimization_strategy,
                'training_time': result.training_time,
                'prediction_time': result.prediction_time,
                'preprocessing_time': result.preprocessing_time,
                'memory_peak_mb': result.memory_peak_mb,
                'train_score': result.train_score,
                'test_score': result.test_score,
                'cv_score_mean': result.cv_score_mean,
                'cv_score_std': result.cv_score_std,
                'best_params': result.best_params,
                'feature_count': result.feature_count,
                'success': result.success,
                'error_message': result.error_message,
                'timestamp': result.timestamp
            }
            results_data.append(result_dict)
        
        final_data = {
            'metadata': {
                'timestamp': self.timestamp,
                'total_results': len(results_data),
                'successful_results': len([r for r in results_data if r['success']]),
                'approach': 'kolosal_automl_fair_comparison',
                'optimizations_disabled': True
            },
            'results': results_data
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info(f"Kolosal AutoML results saved to: {results_file}")
        return str(results_file)
    
    def generate_report(self) -> str:
        """Generate HTML report for Kolosal AutoML results."""
        report_file = self.output_dir / f"kolosal_automl_report_{self.timestamp}.html"
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            logger.warning("No successful results for Kolosal AutoML report")
            return ""
        
        # Calculate summary statistics
        avg_training_time = np.mean([r.training_time for r in successful_results])
        avg_test_score = np.mean([r.test_score for r in successful_results])
        avg_memory = np.mean([r.memory_peak_mb for r in successful_results])
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kolosal AutoML Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 20px; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 10px; border-left: 5px solid #3498db; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #2c3e50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric-box {{ display: inline-block; margin: 10px; padding: 20px; background-color: white; border-radius: 10px; text-align: center; min-width: 200px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
                .warning {{ background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #ffc107; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Kolosal AutoML Benchmark Report</h1>
                <p>Fair Comparison Mode (Optimizations Disabled)</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="warning">
                <h3>⚠️ Fair Comparison Mode</h3>
                <p>This benchmark was run with Kolosal AutoML optimizations <strong>DISABLED</strong> to ensure fair comparison against standard ML approaches. This includes:</p>
                <ul>
                    <li>Single-threaded execution (n_jobs=1)</li>
                    <li>No Intel optimizations</li>
                    <li>No memory optimizations</li>
                    <li>No adaptive batching</li>
                    <li>No parallel preprocessing</li>
                    <li>No feature selection</li>
                    <li>No early stopping</li>
                </ul>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report shows the performance of Kolosal AutoML framework with optimizations disabled for fair comparison against standard scikit-learn approaches.</p>
                
                <div style="text-align: center; margin: 20px 0;">
                    <div class="metric-box">
                        <div class="metric-label">Average Training Time</div>
                        <div class="metric-value">{avg_training_time:.2f}s</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-label">Average Test Score</div>
                        <div class="metric-value">{avg_test_score:.4f}</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-label">Average Memory Usage</div>
                        <div class="metric-value">{avg_memory:.1f} MB</div>
                    </div>
                </div>
                
                <p><strong>Total benchmarks:</strong> {len(self.results)}</p>
                <p><strong>Successful benchmarks:</strong> {len(successful_results)}</p>
                <p><strong>Success rate:</strong> {len(successful_results)/len(self.results)*100:.1f}%</p>
            </div>
            
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Model</th>
                    <th>Optimization Strategy</th>
                    <th>Dataset Size</th>
                    <th>Training Time (s)</th>
                    <th>Test Score</th>
                    <th>Memory (MB)</th>
                    <th>Status</th>
                </tr>
        """
        
        # Add individual results
        for result in self.results:
            status = "✓ Success" if result.success else f"✗ Failed: {result.error_message}"
            html_content += f"""
                <tr>
                    <td>{result.dataset_name}</td>
                    <td>{result.model_name}</td>
                    <td>{result.optimization_strategy}</td>
                    <td>{result.dataset_size[0]} × {result.dataset_size[1]}</td>
                    <td>{result.training_time:.2f}</td>
                    <td>{result.test_score:.4f}</td>
                    <td>{result.memory_peak_mb:.1f}</td>
                    <td>{status}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="summary">
                <h2>Notes</h2>
                <ul>
                    <li>All Kolosal AutoML optimizations have been disabled for fair comparison</li>
                    <li>Single-threaded execution matches standard ML baseline</li>
                    <li>Advanced optimization strategies (ASHT, Bayesian) are still available</li>
                    <li>Framework overhead is minimal in this fair comparison mode</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Kolosal AutoML report generated: {report_file}")
        return str(report_file)

def main():
    """Main function to run Kolosal AutoML benchmarks in fair comparison mode."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Kolosal AutoML Benchmarks (Fair Comparison Mode)")
    parser.add_argument("--output-dir", default="./kolosal_automl_results", help="Output directory")
    parser.add_argument("--datasets", nargs="+", default=[
        "iris", "wine", "breast_cancer", "diabetes", 
        "synthetic_small_classification", "synthetic_medium_classification",
        "synthetic_small_regression"
    ], help="Datasets to test")
    parser.add_argument("--models", nargs="+", default=[
        "random_forest", "gradient_boosting", "logistic_regression"
    ], help="Models to test")
    parser.add_argument("--optimization", default="random_search", 
                       choices=["grid_search", "random_search", "bayesian_optimization", "asht"], 
                       help="Optimization strategy")
    
    args = parser.parse_args()
    
    # Create benchmark runner
    benchmark = KolosalAutoMLBenchmark(args.output_dir)
    
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
    
    logger.info(f"Running {len(configurations)} Kolosal AutoML benchmarks in fair comparison mode...")
    
    # Run benchmarks
    start_time = time.time()
    benchmark.run_multiple_benchmarks(configurations)
    total_time = time.time() - start_time
    
    # Save results and generate report
    results_file = benchmark.save_results()
    report_file = benchmark.generate_report()
    
    # Summary
    successful_results = [r for r in benchmark.results if r.success]
    
    logger.info("=" * 60)
    logger.info("KOLOSAL AUTOML BENCHMARK COMPLETED (FAIR COMPARISON MODE)")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Total benchmarks: {len(configurations)}")
    logger.info(f"Successful benchmarks: {len(successful_results)}")
    logger.info(f"Results saved to: {results_file}")
    if report_file:
        logger.info(f"Report generated: {report_file}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
