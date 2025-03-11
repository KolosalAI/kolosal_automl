import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from pmlb import fetch_data, classification_dataset_names, regression_dataset_names

from modules.configs import TaskType, OptimizationStrategy, MLTrainingEngineConfig
from modules.ml_training_engine import MLTrainingEngine

def benchmark_training_engine():
    """
    Benchmark the ML Training Engine using PMLB datasets.
    Evaluates performance, training time, and memory usage across multiple datasets.
    """
    # Results storage
    results = {
        'dataset': [],
        'task_type': [],
        'dataset_size': [],
        'n_features': [],
        'optimization_strategy': [],
        'model_type': [],
        'training_time': [],
        'memory_usage_mb': [],
        'best_params': [],
        'accuracy': [],
        'f1_score': [],
        'mse': [],
        'r2': []
    }
    
    # Select a subset of datasets for benchmarking
    classification_datasets = classification_dataset_names[:5]  # First 5 classification datasets
    regression_datasets = regression_dataset_names[:5]  # First 5 regression datasets
    
    # Models to benchmark
    classification_models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    regression_models = {
        'random_forest': RandomForestRegressor(random_state=42),
        'ridge': Ridge(random_state=42)
    }
    
    # Parameter grids
    rf_class_params = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 10],
        'model__min_samples_split': [2, 5]
    }
    
    lr_params = {
        'model__C': [0.1, 1.0],
        'model__solver': ['liblinear', 'saga']
    }
    
    rf_reg_params = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 10],
        'model__min_samples_split': [2, 5]
    }
    
    ridge_params = {
        'model__alpha': [0.1, 1.0, 10.0],
        'model__solver': ['auto', 'svd']
    }
    
    param_grids = {
        'classification': {
            'random_forest': rf_class_params,
            'logistic_regression': lr_params
        },
        'regression': {
            'random_forest': rf_reg_params,
            'ridge': ridge_params
        }
    }
    
    # Optimization strategies to test
    strategies = [
        OptimizationStrategy.GRID_SEARCH,
        OptimizationStrategy.RANDOM_SEARCH
    ]
    
    # Run benchmarks
    for task, datasets in [
        (TaskType.CLASSIFICATION, classification_datasets),
        (TaskType.REGRESSION, regression_datasets)
    ]:
        for dataset_name in datasets:
            print(f"Processing dataset: {dataset_name}")
            
            # Fetch dataset
            try:
                data = fetch_data(dataset_name)
                X = data.drop('target', axis=1)
                y = data['target']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, 
                    stratify=y if task == TaskType.CLASSIFICATION else None
                )
                
                # Get appropriate models for the task
                models = classification_models if task == TaskType.CLASSIFICATION else regression_models
                
                for strategy in strategies:
                    for model_name, model in models.items():
                        # Configure the engine
                        config = MLTrainingEngineConfig(
                            task_type=task,
                            optimization_strategy=strategy,
                            feature_selection=True,
                            feature_selection_k=min(15, X.shape[1]),
                            cv_folds=3,  # Use fewer folds for benchmarking
                            model_path="./benchmark_models",
                            experiment_tracking=True,
                            log_level="INFO"
                        )
                        
                        # Initialize engine
                        engine = MLTrainingEngine(config)
                        
                        # Train and measure performance
                        start_time = time.time()
                        start_memory = get_memory_usage()
                        
                        best_model, metrics = engine.train_model(
                            model=model,
                            model_name=f"{model_name}_{dataset_name}",
                            param_grid=param_grids['classification' if task == TaskType.CLASSIFICATION else 'regression'][model_name],
                            X=X_train,
                            y=y_train,
                            X_test=X_test,
                            y_test=y_test
                        )
                        
                        training_time = time.time() - start_time
                        memory_used = get_memory_usage() - start_memory
                        
                        # Make predictions
                        y_pred = engine.predict(X_test)
                        
                        # Calculate metrics
                        if task == TaskType.CLASSIFICATION:
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            mse = None
                            r2 = None
                        else:
                            acc = None
                            f1 = None
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                        
                        # Store results
                        results['dataset'].append(dataset_name)
                        results['task_type'].append(task.name)
                        results['dataset_size'].append(len(X))
                        results['n_features'].append(X.shape[1])
                        results['optimization_strategy'].append(strategy.name)
                        results['model_type'].append(model_name)
                        results['training_time'].append(training_time)
                        results['memory_usage_mb'].append(memory_used)
                        results['best_params'].append(str(engine.best_params))
                        results['accuracy'].append(acc)
                        results['f1_score'].append(f1)
                        results['mse'].append(mse)
                        results['r2'].append(r2)
                        
                        # Clean up
                        engine.shutdown()
                        
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {str(e)}")
                continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv("ml_engine_benchmark_results.csv", index=False)
    
    # Generate summary visualizations
    generate_benchmark_visualizations(results_df)
    
    return results_df

def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def generate_benchmark_visualizations(results_df):
    """Generate visualizations from benchmark results"""
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Training time by dataset size
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=results_df, 
        x='dataset_size', 
        y='training_time',
        hue='model_type',
        style='optimization_strategy',
        size='n_features',
        sizes=(20, 200),
        alpha=0.7
    )
    plt.title('Training Time vs Dataset Size')
    plt.xlabel('Dataset Size (number of samples)')
    plt.ylabel('Training Time (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('benchmark_training_time.png')
    
    # 2. Performance metrics by model type
    plt.figure(figsize=(14, 6))
    
    # Classification metrics
    class_df = results_df[results_df['task_type'] == 'CLASSIFICATION']
    if not class_df.empty:
        plt.subplot(1, 2, 1)
        sns.boxplot(data=class_df, x='model_type', y='accuracy')
        plt.title('Classification Accuracy by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Accuracy')
    
    # Regression metrics
    reg_df = results_df[results_df['task_type'] == 'REGRESSION']
    if not reg_df.empty:
        plt.subplot(1, 2, 2)
        sns.boxplot(data=reg_df, x='model_type', y='r2')
        plt.title('Regression R² by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('R² Score')
    
    plt.tight_layout()
    plt.savefig('benchmark_performance.png')
    
    # 3. Optimization strategy comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=results_df,
        x='optimization_strategy',
        y='training_time',
        hue='model_type'
    )
    plt.title('Training Time by Optimization Strategy')
    plt.xlabel('Optimization Strategy')
    plt.ylabel('Training Time (seconds)')
    plt.tight_layout()
    plt.savefig('benchmark_optimization_strategy.png')
    
    # 4. Memory usage
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=results_df,
        x='dataset_size',
        y='memory_usage_mb',
        hue='model_type',
        style='optimization_strategy',
        alpha=0.7
    )
    plt.title('Memory Usage vs Dataset Size')
    plt.xlabel('Dataset Size (number of samples)')
    plt.ylabel('Memory Usage (MB)')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('benchmark_memory_usage.png')

def run_comprehensive_benchmark():
    """Run a comprehensive benchmark and generate a report"""
    print("Starting ML Training Engine Benchmark...")
    
    # Run benchmarks
    results = benchmark_training_engine()
    
    # Generate summary statistics
    summary = {
        'classification': {},
        'regression': {}
    }
    
    # Classification summary
    class_df = results[results['task_type'] == 'CLASSIFICATION']
    if not class_df.empty:
        summary['classification']['avg_accuracy'] = class_df['accuracy'].mean()
        summary['classification']['avg_f1'] = class_df['f1_score'].mean()
        summary['classification']['avg_training_time'] = class_df['training_time'].mean()
        
        # Best model by accuracy
        best_idx = class_df['accuracy'].idxmax()
        summary['classification']['best_model'] = {
            'model': class_df.loc[best_idx, 'model_type'],
            'dataset': class_df.loc[best_idx, 'dataset'],
            'accuracy': class_df.loc[best_idx, 'accuracy'],
            'optimization': class_df.loc[best_idx, 'optimization_strategy']
        }
    
    # Regression summary
    reg_df = results[results['task_type'] == 'REGRESSION']
    if not reg_df.empty:
        summary['regression']['avg_mse'] = reg_df['mse'].mean()
        summary['regression']['avg_r2'] = reg_df['r2'].mean()
        summary['regression']['avg_training_time'] = reg_df['training_time'].mean()
        
        # Best model by R2
        best_idx = reg_df['r2'].idxmax()
        summary['regression']['best_model'] = {
            'model': reg_df.loc[best_idx, 'model_type'],
            'dataset': reg_df.loc[best_idx, 'dataset'],
            'r2': reg_df.loc[best_idx, 'r2'],
            'optimization': reg_df.loc[best_idx, 'optimization_strategy']
        }
    
    # Print summary
    print("\n===== ML Training Engine Benchmark Summary =====")
    print(f"Total datasets tested: {results['dataset'].nunique()}")
    print(f"Models tested: {', '.join(results['model_type'].unique())}")
    print(f"Optimization strategies: {', '.join(results['optimization_strategy'].unique())}")
    
    print("\nClassification Results:")
    if 'avg_accuracy' in summary['classification']:
        print(f"  Average Accuracy: {summary['classification']['avg_accuracy']:.4f}")
        print(f"  Average F1 Score: {summary['classification']['avg_f1']:.4f}")
        print(f"  Average Training Time: {summary['classification']['avg_training_time']:.2f} seconds")
        print(f"  Best Model: {summary['classification']['best_model']['model']} on {summary['classification']['best_model']['dataset']}")
        print(f"    Accuracy: {summary['classification']['best_model']['accuracy']:.4f}")
        print(f"    Optimization: {summary['classification']['best_model']['optimization']}")
    
    print("\nRegression Results:")
    if 'avg_r2' in summary['regression']:
        print(f"  Average R² Score: {summary['regression']['avg_r2']:.4f}")
        print(f"  Average MSE: {summary['regression']['avg_mse']:.4f}")
        print(f"  Average Training Time: {summary['regression']['avg_training_time']:.2f} seconds")
        print(f"  Best Model: {summary['regression']['best_model']['model']} on {summary['regression']['best_model']['dataset']}")
        print(f"    R² Score: {summary['regression']['best_model']['r2']:.4f}")
        print(f"    Optimization: {summary['regression']['best_model']['optimization']}")
    
    print("\nDetailed results saved to 'ml_engine_benchmark_results.csv'")
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    run_comprehensive_benchmark()
