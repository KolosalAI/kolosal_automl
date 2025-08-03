#!/usr/bin/env python3
"""
Standard ML Benchmark Script

This script runs pure scikit-learn ML approaches without any optimizations.
It serves as the baseline for comparison against Kolosal AutoML.

Features:
- Pure scikit-learn implementation
- No advanced optimizations
- Standard hyperparameter tuning
- Basic preprocessing
- Memory and performance tracking
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

# Standard ML imports (pure scikit-learn)
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, 
    load_diabetes, make_classification, make_regression,
    fetch_california_housing
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

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StandardMLBenchmark")

@dataclass
class StandardBenchmarkResult:
    """Data class to store standard ML benchmark results."""
    benchmark_id: str
    dataset_name: str
    model_name: str
    dataset_size: Tuple[int, int]
    task_type: str
    
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
    """Loads datasets for standard ML benchmarking."""
    
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

class StandardMLBenchmark:
    """Implements standard ML approach using pure scikit-learn without optimizations."""
    
    def __init__(self, output_dir: str = "./standard_ml_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
                     optimization_strategy: str = "random_search") -> StandardBenchmarkResult:
        """Run standard ML benchmark."""
        benchmark_id = f"std_{dataset_name}_{model_name}_{optimization_strategy}_{int(time.time())}"
        logger.info(f"Starting standard ML benchmark: {benchmark_id}")
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        try:
            # Load dataset
            X, y, task_type = DatasetLoader.load_dataset(dataset_name)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if task_type == "classification" else None
            )
            
            # Preprocessing (basic only)
            preprocessing_start = time.time()
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            preprocessing_time = time.time() - preprocessing_start
            
            # Get model and parameters
            model, param_grid = self.get_model_and_params(model_name, task_type)
            
            # Create pipeline (basic)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Hyperparameter optimization (standard approach)
            if optimization_strategy == "grid_search":
                search = GridSearchCV(
                    pipeline, param_grid, cv=3, 
                    scoring='accuracy' if task_type == "classification" else 'r2',
                    n_jobs=1  # Single job for baseline comparison
                )
            else:  # random_search
                search = RandomizedSearchCV(
                    pipeline, param_grid, cv=3, n_iter=10, random_state=42,
                    scoring='accuracy' if task_type == "classification" else 'r2',
                    n_jobs=1  # Single job for baseline comparison
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
            
            result = StandardBenchmarkResult(
                benchmark_id=benchmark_id,
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_size=X.shape,
                task_type=task_type,
                training_time=training_time,
                prediction_time=prediction_time,
                preprocessing_time=preprocessing_time,
                memory_peak_mb=peak_memory,
                train_score=train_score,
                test_score=test_score,
                cv_score_mean=cv_scores.mean(),
                cv_score_std=cv_scores.std(),
                best_params=search.best_params_,
                feature_count=X.shape[1],
                success=True,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            logger.info(f"Standard ML benchmark completed successfully: {benchmark_id}")
            logger.info(f"Training time: {training_time:.2f}s, Test score: {test_score:.4f}")
            
        except Exception as e:
            logger.error(f"Standard ML benchmark failed: {e}")
            result = StandardBenchmarkResult(
                benchmark_id=benchmark_id,
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_size=(0, 0),
                task_type="unknown",
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
    
    def run_multiple_benchmarks(self, configurations: List[Dict[str, str]]) -> List[StandardBenchmarkResult]:
        """Run multiple standard ML benchmarks."""
        results = []
        total = len(configurations)
        
        for i, config in enumerate(configurations, 1):
            logger.info(f"Running standard ML benchmark {i}/{total}: {config}")
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
        results_file = self.output_dir / f"standard_ml_results_{self.timestamp}.json"
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            result_dict = {
                'benchmark_id': result.benchmark_id,
                'dataset_name': result.dataset_name,
                'model_name': result.model_name,
                'dataset_size': result.dataset_size,
                'task_type': result.task_type,
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
                'approach': 'standard_ml_baseline'
            },
            'results': results_data
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info(f"Standard ML results saved to: {results_file}")
        return str(results_file)
    
    def generate_report(self) -> str:
        """Generate HTML report for standard ML results."""
        report_file = self.output_dir / f"standard_ml_report_{self.timestamp}.html"
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            logger.warning("No successful results for standard ML report")
            return ""
        
        # Calculate summary statistics
        avg_training_time = np.mean([r.training_time for r in successful_results])
        avg_test_score = np.mean([r.test_score for r in successful_results])
        avg_memory = np.mean([r.memory_peak_mb for r in successful_results])
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Standard ML Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #8B4513, #D2B48C); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 20px; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 10px; border-left: 5px solid #8B4513; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #8B4513; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric-box {{ display: inline-block; margin: 10px; padding: 20px; background-color: white; border-radius: 10px; text-align: center; min-width: 200px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; color: #8B4513; }}
                .metric-label {{ font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Standard ML Benchmark Report</h1>
                <p>Pure Scikit-Learn Baseline Performance</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report shows the performance of standard machine learning approaches using pure scikit-learn without any optimizations. This serves as the baseline for comparison with Kolosal AutoML.</p>
                
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
                    <li>All benchmarks use standard scikit-learn implementations without optimizations</li>
                    <li>Single-threaded execution for consistent baseline measurements</li>
                    <li>Basic preprocessing with StandardScaler only</li>
                    <li>Standard hyperparameter search with limited iterations</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Standard ML report generated: {report_file}")
        return str(report_file)

def main():
    """Main function to run standard ML benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Standard ML Benchmarks (Baseline)")
    parser.add_argument("--output-dir", default="./standard_ml_results", help="Output directory")
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
    benchmark = StandardMLBenchmark(args.output_dir)
    
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
    
    logger.info(f"Running {len(configurations)} standard ML benchmarks...")
    
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
    logger.info("STANDARD ML BENCHMARK COMPLETED")
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
