import numpy as np
import pandas as pd
import time
import json
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Import Genta modules
from modules.configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig
)
from modules.engine.train_engine import MLTrainingEngine
from modules.engine.optimizer import ASHTOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ASHTBenchmark")

class ASHTBenchmarkRunner:
    """
    Run benchmarks specifically testing the ASHT optimizer against other strategies.
    """
    
    def __init__(self, output_dir: str = "./asht_benchmark_results"):
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"ASHT Benchmark Runner initialized")
        logger.info(f"Results will be saved to: {output_dir}")

    def run_benchmark(self, dataset_name: str):
        """
        Run a complete benchmark comparing all optimization strategies on a dataset.
        
        Args:
            dataset_name: Name of the dataset from OpenML
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Starting benchmark on dataset: {dataset_name}")
        
        # Load dataset
        try:
            # Fetch from OpenML
            data = fetch_openml(name=dataset_name, as_frame=True)
            X = data.data
            y = data.target
            
            # Handle categorical features
            X = pd.get_dummies(X, drop_first=True)
            
            # Convert to arrays
            X = X.values
            
            if y.dtype == object or y.dtype.name == 'category':
                # For classification, encode labels
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
            else:
                y = y.values
                
            logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
            return {
                "dataset": dataset_name,
                "error": f"Failed to load dataset: {str(e)}",
                "success": False
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if len(np.unique(y)) < 10 else None
        )
        
        # Use RandomForest for all tests
        model = RandomForestClassifier(random_state=42)
        
        # Define parameter grid (identical for all strategies)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        
        # Define optimization strategies to test
        strategies = [
            ("grid_search", OptimizationStrategy.GRID_SEARCH),
            ("random_search", OptimizationStrategy.RANDOM_SEARCH),
            ("bayesian_optimization", OptimizationStrategy.BAYESIAN_OPTIMIZATION),
            ("asht", OptimizationStrategy.ASHT)
        ]
        
        benchmark_results = {
            "dataset": dataset_name,
            "dataset_shape": X.shape,
            "strategies": {},
            "success": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Run benchmarks for each strategy
        for strategy_name, strategy_enum in strategies:
            logger.info(f"Testing optimization strategy: {strategy_name}")
            
            # Create MLTrainingEngineConfig with current strategy
            config = MLTrainingEngineConfig(
                task_type=TaskType.CLASSIFICATION,
                random_state=42,
                n_jobs=-1,  # Use all cores
                verbose=1,
                cv_folds=5,
                test_size=0.2,
                stratify=True,
                optimization_strategy=strategy_enum,
                optimization_iterations=30,  # Reduced for benchmark
                early_stopping=True,
                feature_selection=True,
                model_path=f"./asht_models/{dataset_name}/{strategy_name}",
                experiment_tracking=False,  # Disable to reduce overhead
                memory_optimization=True
            )
            
            try:
                # Create engine
                engine = MLTrainingEngine(config)
                
                # Training with timing
                start_time = time.time()
                best_model, metrics = engine.train_model(
                    model=model,
                    model_name=f"{strategy_name}_benchmark",
                    param_grid=param_grid,
                    X=X_train,
                    y=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
                training_time = time.time() - start_time
                
                # Make predictions with timing
                inference_start = time.time()
                predictions = engine.predict(X_test)
                inference_time = time.time() - inference_start
                
                # Calculate metrics on test set
                if len(np.unique(y)) == 2:  # Binary classification
                    test_accuracy = accuracy_score(y_test, predictions)
                    test_f1 = f1_score(y_test, predictions, average='binary')
                    # Try to get probabilities for AUC
                    try:
                        model_in_pipeline = best_model.named_steps['model']
                        if hasattr(model_in_pipeline, 'predict_proba'):
                            proba = engine.best_model["model"].predict_proba(X_test)
                            test_auc = roc_auc_score(y_test, proba[:, 1])
                        else:
                            test_auc = None
                    except:
                        test_auc = None
                else:  # Multi-class
                    test_accuracy = accuracy_score(y_test, predictions)
                    test_f1 = f1_score(y_test, predictions, average='weighted')
                    test_auc = None
                
                # Calculate throughput
                train_throughput = X_train.shape[0] / training_time if training_time > 0 else 0
                inference_throughput = X_test.shape[0] / inference_time if inference_time > 0 else 0
                
                # Store strategy results
                benchmark_results["strategies"][strategy_name] = {
                    "training_time_seconds": training_time,
                    "inference_time_seconds": inference_time,
                    "train_metrics": metrics,
                    "test_metrics": {
                        "accuracy": test_accuracy,
                        "f1": test_f1,
                        "auc": test_auc
                    },
                    "training_throughput": train_throughput,
                    "inference_throughput": inference_throughput,
                    "best_params": engine.best_model.get("params", {})
                }
                
                logger.info(f"Strategy {strategy_name} completed in {training_time:.2f} seconds")
                logger.info(f"Test metrics: Accuracy={test_accuracy:.4f}, F1={test_f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error with strategy {strategy_name}: {str(e)}")
                benchmark_results["strategies"][strategy_name] = {
                    "error": str(e),
                    "success": False
                }
            
            finally:
                # Clean up
                if 'engine' in locals():
                    engine.shutdown()
                    del engine
                
        # Store and return results
        self.results.append(benchmark_results)
        
        # Save individual benchmark result
        result_file = os.path.join(self.output_dir, f"asht_benchmark_{dataset_name}.json")
        with open(result_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
            
        logger.info(f"Benchmark completed for dataset: {dataset_name}")
        logger.info(f"Results saved to: {result_file}")
        
        return benchmark_results

    def run_multiple_benchmarks(self, dataset_names):
        """
        Run benchmarks on multiple datasets.
        
        Args:
            dataset_names: List of dataset names from OpenML
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for dataset_name in dataset_names:
            logger.info(f"Running benchmark on dataset: {dataset_name}")
            result = self.run_benchmark(dataset_name)
            results.append(result)
            
        return results

    def save_results(self):
        """
        Save all benchmark results to a file.
        
        Returns:
            Path to the results file
        """
        results_file = os.path.join(self.output_dir, f"asht_benchmark_results.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"All benchmark results saved to: {results_file}")
        return results_file

    def generate_report(self):
        """
        Generate visualizations comparing optimization strategies.
        
        Returns:
            Dictionary with paths to generated charts
        """
        if not self.results:
            logger.warning("No results to generate report.")
            return {}
            
        # Filter successful benchmarks only
        successful_results = [r for r in self.results if r.get('success', False)]
        
        if not successful_results:
            logger.warning("No successful benchmarks to include in report.")
            return {}
            
        # Create directory for charts
        charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Prepare data for comparison charts
        training_times = []
        test_accuracies = []
        test_f1s = []
        
        for result in successful_results:
            dataset = result['dataset']
            dataset_shape = result['dataset_shape']
            
            for strategy, stats in result['strategies'].items():
                if 'training_time_seconds' in stats:
                    training_times.append({
                        'dataset': dataset,
                        'samples': dataset_shape[0],
                        'features': dataset_shape[1],
                        'strategy': strategy,
                        'time': stats['training_time_seconds']
                    })
                
                if 'test_metrics' in stats and isinstance(stats['test_metrics'], dict):
                    if 'accuracy' in stats['test_metrics']:
                        test_accuracies.append({
                            'dataset': dataset,
                            'strategy': strategy,
                            'accuracy': stats['test_metrics']['accuracy']
                        })
                    
                    if 'f1' in stats['test_metrics']:
                        test_f1s.append({
                            'dataset': dataset,
                            'strategy': strategy,
                            'f1': stats['test_metrics']['f1']
                        })
        
        chart_paths = {}
        
        # Generate training time comparison
        if training_times:
            df_times = pd.DataFrame(training_times)
            plt.figure(figsize=(12, 8))
            sns.barplot(x='dataset', y='time', hue='strategy', data=df_times)
            plt.title('Training Time by Optimization Strategy')
            plt.ylabel('Training Time (seconds)')
            plt.xlabel('Dataset')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            time_chart = os.path.join(charts_dir, "training_time_comparison.png")
            plt.savefig(time_chart)
            plt.close()
            chart_paths['training_time'] = time_chart
        
        # Generate accuracy comparison
        if test_accuracies:
            df_acc = pd.DataFrame(test_accuracies)
            plt.figure(figsize=(12, 8))
            sns.barplot(x='dataset', y='accuracy', hue='strategy', data=df_acc)
            plt.title('Test Accuracy by Optimization Strategy')
            plt.ylabel('Accuracy')
            plt.xlabel('Dataset')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            acc_chart = os.path.join(charts_dir, "accuracy_comparison.png")
            plt.savefig(acc_chart)
            plt.close()
            chart_paths['accuracy'] = acc_chart
        
        # Generate F1 score comparison
        if test_f1s:
            df_f1 = pd.DataFrame(test_f1s)
            plt.figure(figsize=(12, 8))
            sns.barplot(x='dataset', y='f1', hue='strategy', data=df_f1)
            plt.title('Test F1 Score by Optimization Strategy')
            plt.ylabel('F1 Score')
            plt.xlabel('Dataset')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            f1_chart = os.path.join(charts_dir, "f1_comparison.png")
            plt.savefig(f1_chart)
            plt.close()
            chart_paths['f1'] = f1_chart
        
        # Generate relative performance chart
        if training_times and test_accuracies:
            # Merge data
            df_times_grouped = pd.DataFrame(training_times).groupby(['dataset', 'strategy']).mean().reset_index()
            df_acc_grouped = pd.DataFrame(test_accuracies).groupby(['dataset', 'strategy']).mean().reset_index()
            
            # Calculate relative performance (normalize within each dataset)
            relative_perf = []
            
            for dataset in df_times_grouped['dataset'].unique():
                # Get times for this dataset
                dataset_times = df_times_grouped[df_times_grouped['dataset'] == dataset]
                dataset_acc = df_acc_grouped[df_acc_grouped['dataset'] == dataset]
                
                # Find baseline (random search)
                try:
                    baseline_time = dataset_times[dataset_times['strategy'] == 'random_search']['time'].values[0]
                    
                    # Calculate relative speedup
                    for _, row in dataset_times.iterrows():
                        # Find corresponding accuracy
                        acc_row = dataset_acc[
                            (dataset_acc['dataset'] == row['dataset']) & 
                            (dataset_acc['strategy'] == row['strategy'])
                        ]
                        
                        if not acc_row.empty:
                            accuracy = acc_row['accuracy'].values[0]
                        else:
                            accuracy = None
                            
                        relative_perf.append({
                            'dataset': row['dataset'],
                            'strategy': row['strategy'],
                            'relative_speedup': baseline_time / row['time'],
                            'accuracy': accuracy
                        })
                except:
                    logger.warning(f"Could not calculate relative performance for {dataset}")
            
            if relative_perf:
                df_rel = pd.DataFrame(relative_perf)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='dataset', y='relative_speedup', hue='strategy', data=df_rel)
                plt.axhline(y=1.0, color='r', linestyle='--')
                plt.title('Relative Speedup Compared to Random Search')
                plt.ylabel('Speedup Factor (higher is better)')
                plt.xlabel('Dataset')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                rel_chart = os.path.join(charts_dir, "relative_speedup.png")
                plt.savefig(rel_chart)
                plt.close()
                chart_paths['relative_speedup'] = rel_chart
                
                # Create scatter plot of accuracy vs. speedup
                plt.figure(figsize=(10, 8))
                for strategy in df_rel['strategy'].unique():
                    strategy_data = df_rel[df_rel['strategy'] == strategy]
                    plt.scatter(
                        strategy_data['relative_speedup'], 
                        strategy_data['accuracy'],
                        label=strategy,
                        alpha=0.7,
                        s=100
                    )
                
                plt.axvline(x=1.0, color='r', linestyle='--')
                plt.xlabel('Relative Speedup (vs. Random Search)')
                plt.ylabel('Test Accuracy')
                plt.title('Accuracy vs. Speedup by Optimization Strategy')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                tradeoff_chart = os.path.join(charts_dir, "accuracy_vs_speedup.png")
                plt.savefig(tradeoff_chart)
                plt.close()
                chart_paths['tradeoff'] = tradeoff_chart
        
        # Generate HTML report
        report_path = os.path.join(self.output_dir, "asht_comparison_report.html")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ASHT Optimizer Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .chart-container { margin: 20px 0; text-align: center; }
                .summary { background-color: #f5f5f5; padding: 15px; margin: 15px 0; }
            </style>
        </head>
        <body>
            <h1>ASHT Optimizer Benchmark Report</h1>
            <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>This report compares different optimization strategies including:</p>
                <ul>
                    <li><b>Grid Search</b>: Exhaustive search over parameter grid</li>
                    <li><b>Random Search</b>: Randomly samples parameter combinations</li>
                    <li><b>Bayesian Optimization</b>: Sequential model-based optimization</li>
                    <li><b>ASHT</b>: Adaptive Surrogate-assisted Hyperparameter Tuning</li>
                </ul>
            </div>
        """
        
        # Add charts to report
        for chart_type, chart_path in chart_paths.items():
            if os.path.exists(chart_path):
                relative_path = os.path.relpath(chart_path, self.output_dir)
                html_content += f"""
                <div class="chart-container">
                    <h3>{chart_type.replace('_', ' ').title()}</h3>
                    <img src="{relative_path}" alt="{chart_type}" width="800">
                </div>
                """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated report with charts at: {report_path}")
        chart_paths['html_report'] = report_path
        
        return chart_paths

def main():
    """Main function to run ASHT benchmarks"""
    # Create benchmark runner
    benchmark_runner = ASHTBenchmarkRunner()
    
    # Define datasets to benchmark
    # Using well-known classification datasets from OpenML
    datasets = [
        "iris",                 # Small dataset
        "wine",                 # Small dataset
        "breast_cancer",        # Medium dataset
        "car",                  # Medium categorical dataset
        "credit-g",             # Medium dataset with mixed features
        "blood-transfusion"     # Medical dataset
    ]
    
    # Run benchmarks
    benchmark_runner.run_multiple_benchmarks(datasets)
    
    # Save results
    results_file = benchmark_runner.save_results()
    
    # Generate report
    chart_paths = benchmark_runner.generate_report()
    
    logger.info(f"ASHT benchmarking complete!")
    logger.info(f"Results saved to: {results_file}")
    if 'html_report' in chart_paths:
        logger.info(f"Report generated: {chart_paths['html_report']}")
    
    return {
        "results_file": results_file,
        "chart_paths": chart_paths
    }

if __name__ == "__main__":
    main()