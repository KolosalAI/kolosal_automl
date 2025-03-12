#!/usr/bin/env python
# Genta AutoML Comprehensive Benchmark
# This script benchmarks training and inference performance using the PLMB dataset

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pickle
import json
from datetime import datetime
import gc
import logging
import psutil
from tqdm import tqdm
import threading
import multiprocessing
from pathlib import Path
import warnings
import seaborn as sns

# Import Genta AutoML components
from modules.configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig,
    InferenceEngineConfig,
    QuantizationConfig,
    PreprocessorConfig,
    BatchProcessorConfig,
    NormalizationType,
    QuantizationType,
    QuantizationMode,
    ModelType
)
from modules.engine.train_engine import MLTrainingEngine
from modules.engine.inference_engine import InferenceEngine
from modules.engine.data_preprocessor import DataPreprocessor
from modules.engine.quantizer import Quantizer
from modules.device_optimizer import DeviceOptimizer

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GentaBenchmark")

class BenchmarkResults:
    """Class to store, analyze and visualize benchmark results"""
    
    def __init__(self, name: str, output_dir: str = "./benchmark_results"):
        self.name = name
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "training": [],
            "inference": [],
            "quantization": [],
            "memory": [],
            "system_info": self._get_system_info()
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_system_info(self) -> dict:
        """Get system information for the benchmark report"""
        import platform
        
        return {
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3)
        }
    
    def add_training_result(self, model_name: str, task_type: str, dataset_size: tuple, 
                            duration: float, config: dict, metrics: dict,
                            memory_usage: float):
        """Add a training benchmark result"""
        self.results["training"].append({
            "model_name": model_name,
            "task_type": task_type,
            "samples": dataset_size[0],
            "features": dataset_size[1],
            "duration_seconds": duration,
            "config": config,
            "metrics": metrics,
            "memory_usage_mb": memory_usage,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_inference_result(self, model_name: str, task_type: str, dataset_size: tuple,
                             batch_size: int, total_duration: float, avg_latency: float, 
                             throughput: float, config: dict, memory_usage: float):
        """Add an inference benchmark result"""
        self.results["inference"].append({
            "model_name": model_name,
            "task_type": task_type,
            "samples": dataset_size[0],
            "features": dataset_size[1],
            "batch_size": batch_size,
            "duration_seconds": total_duration,
            "avg_latency_ms": avg_latency,
            "throughput_samples_per_second": throughput,
            "config": config,
            "memory_usage_mb": memory_usage,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_quantization_result(self, config: dict, performance: dict, memory: dict, accuracy: dict):
        """Add a quantization benchmark result"""
        self.results["quantization"].append({
            "config": config,
            "performance": performance,
            "memory": memory,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_memory_profile(self, stage: str, model_name: str, memory_profile: list):
        """Add memory profiling data"""
        self.results["memory"].append({
            "stage": stage,
            "model_name": model_name,
            "profile": memory_profile,
            "timestamp": datetime.now().isoformat()
        })
    
    def save(self):
        """Save benchmark results to JSON file"""
        filename = f"{self.output_dir}/{self.name}_{self.timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Benchmark results saved to {filename}")
        return filename
    
    def generate_report(self, include_plots: bool = True):
        """Generate a comprehensive benchmark report"""
        report_file = f"{self.output_dir}/{self.name}_{self.timestamp}_report.md"
        
        with open(report_file, 'w') as f:
            # Write header
            f.write(f"# Benchmark Report: {self.name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System information
            f.write("## System Information\n\n")
            f.write("| Parameter | Value |\n")
            f.write("| --- | --- |\n")
            for key, value in self.results["system_info"].items():
                f.write(f"| {key} | {value} |\n")
            f.write("\n")
            
            # Training results
            if self.results["training"]:
                f.write("## Training Benchmark Results\n\n")
                f.write("| Model | Task | Dataset Size | Duration (s) | Memory (MB) | Key Metric |\n")
                f.write("| --- | --- | --- | --- | --- | --- |\n")
                
                for result in self.results["training"]:
                    dataset_size = f"{result['samples']} × {result['features']}"
                    
                    # Get key metric based on task type
                    key_metric = "N/A"
                    if result["task_type"] == "CLASSIFICATION":
                        key_metric = f"Accuracy: {result['metrics'].get('accuracy', 'N/A'):.4f}"
                    elif result["task_type"] == "REGRESSION":
                        key_metric = f"RMSE: {result['metrics'].get('rmse', 'N/A'):.4f}"
                    
                    f.write(f"| {result['model_name']} | {result['task_type']} | {dataset_size} | "
                           f"{result['duration_seconds']:.2f} | {result['memory_usage_mb']:.1f} | {key_metric} |\n")
                
                f.write("\n")
                
                # Create comparison plots for training if enabled
                if include_plots:
                    training_plot_path = self._create_training_plots()
                    f.write(f"![Training Performance]({os.path.basename(training_plot_path)})\n\n")
            
            # Inference results
            if self.results["inference"]:
                f.write("## Inference Benchmark Results\n\n")
                f.write("| Model | Task | Dataset Size | Batch Size | Total Time (s) | Avg Latency (ms) | "
                        "Throughput (samples/s) | Memory (MB) |\n")
                f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
                
                for result in self.results["inference"]:
                    dataset_size = f"{result['samples']} × {result['features']}"
                    
                    f.write(f"| {result['model_name']} | {result['task_type']} | {dataset_size} | "
                           f"{result['batch_size']} | {result['duration_seconds']:.2f} | "
                           f"{result['avg_latency_ms']:.2f} | {result['throughput_samples_per_second']:.1f} | "
                           f"{result['memory_usage_mb']:.1f} |\n")
                
                f.write("\n")
                
                # Create comparison plots for inference if enabled
                if include_plots:
                    inference_plot_path = self._create_inference_plots()
                    f.write(f"![Inference Performance]({os.path.basename(inference_plot_path)})\n\n")
            
            # Quantization results
            if self.results["quantization"]:
                f.write("## Quantization Benchmark Results\n\n")
                f.write("| Configuration | Quantize (ms) | Dequantize (ms) | Throughput (samples/s) | Error | Compression Ratio |\n")
                f.write("| --- | --- | --- | --- | --- | --- |\n")
                
                for result in self.results["quantization"]:
                    config = f"{result['config']['type']}_{result['config']['mode']}"
                    f.write(f"| {config} | {result['performance']['avg_quantize_time_ms']:.2f} | "
                           f"{result['performance']['avg_dequantize_time_ms']:.2f} | "
                           f"{result['performance']['throughput_samples_per_second']:.2f} | "
                           f"{result['accuracy']['mean_absolute_error']:.6f} | "
                           f"{result['memory']['compression_ratio']:.2f}x |\n")
                
                f.write("\n")
                
                # Create comparison plots for quantization if enabled
                if include_plots:
                    quantization_plot_path = self._create_quantization_plots()
                    f.write(f"![Quantization Performance]({os.path.basename(quantization_plot_path)})\n\n")
            
            # Memory profiling
            if self.results["memory"] and include_plots:
                f.write("## Memory Usage Profiles\n\n")
                memory_plot_path = self._create_memory_plots()
                f.write(f"![Memory Usage]({os.path.basename(memory_plot_path)})\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write(self._generate_recommendations())
            
        logger.info(f"Benchmark report generated: {report_file}")
        return report_file
    
    def _create_training_plots(self):
        """Create comparison plots for training benchmarks"""
        plt.figure(figsize=(12, 6))
        
        # Extract data for plotting
        models = []
        durations = []
        memory_usages = []
        
        for result in self.results["training"]:
            models.append(result["model_name"])
            durations.append(result["duration_seconds"])
            memory_usages.append(result["memory_usage_mb"])
        
        # Create plot
        x = range(len(models))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot training time
        bars1 = ax1.bar(x, durations, width, label='Training Time (s)', color='steelblue')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Training Time (seconds)', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        
        # Plot memory usage on secondary axis
        ax2 = ax1.twinx()
        bars2 = ax2.bar([p + width for p in x], memory_usages, width, label='Memory Usage (MB)', color='coral')
        ax2.set_ylabel('Memory Usage (MB)', color='coral')
        ax2.tick_params(axis='y', labelcolor='coral')
        
        # Set x-axis labels
        ax1.set_xticks([p + width/2 for p in x])
        ax1.set_xticklabels(models)
        
        # Add legend
        fig.tight_layout()
        fig.legend([bars1, bars2], ['Training Time (s)', 'Memory Usage (MB)'], loc='upper right')
        
        # Save plot
        plot_path = f"{self.output_dir}/{self.name}_{self.timestamp}_training_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _create_inference_plots(self):
        """Create comparison plots for inference benchmarks"""
        # Group data by model
        model_data = {}
        for result in self.results["inference"]:
            model_name = result["model_name"]
            if model_name not in model_data:
                model_data[model_name] = {"batch_sizes": [], "latency": [], "throughput": []}
                
            model_data[model_name]["batch_sizes"].append(result["batch_size"])
            model_data[model_name]["latency"].append(result["avg_latency_ms"])
            model_data[model_name]["throughput"].append(result["throughput_samples_per_second"])
        
        # Create plots
        plt.figure(figsize=(16, 12))
        
        # Latency vs batch size
        plt.subplot(2, 2, 1)
        for model, data in model_data.items():
            # Sort by batch size
            indices = np.argsort(data["batch_sizes"])
            batch_sizes = [data["batch_sizes"][i] for i in indices]
            latency = [data["latency"][i] for i in indices]
            plt.plot(batch_sizes, latency, 'o-', label=model)
            
        plt.xlabel('Batch Size')
        plt.ylabel('Average Latency (ms)')
        plt.title('Latency vs Batch Size')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Throughput vs batch size
        plt.subplot(2, 2, 2)
        for model, data in model_data.items():
            # Sort by batch size
            indices = np.argsort(data["batch_sizes"])
            batch_sizes = [data["batch_sizes"][i] for i in indices]
            throughput = [data["throughput"][i] for i in indices]
            plt.plot(batch_sizes, throughput, 'o-', label=model)
            
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (samples/s)')
        plt.title('Throughput vs Batch Size')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Latency comparison for different models
        plt.subplot(2, 2, 3)
        models = []
        latencies = []
        for model, data in model_data.items():
            # Use the batch size with the best throughput
            best_idx = np.argmax(data["throughput"])
            models.append(f"{model}\n(batch={data['batch_sizes'][best_idx]})")
            latencies.append(data["latency"][best_idx])
            
        plt.bar(models, latencies, color='steelblue')
        plt.xlabel('Model')
        plt.ylabel('Best Latency (ms)')
        plt.title('Best Latency by Model')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Throughput comparison for different models
        plt.subplot(2, 2, 4)
        models = []
        throughputs = []
        for model, data in model_data.items():
            # Use the batch size with the best throughput
            best_idx = np.argmax(data["throughput"])
            models.append(f"{model}\n(batch={data['batch_sizes'][best_idx]})")
            throughputs.append(data["throughput"][best_idx])
            
        plt.bar(models, throughputs, color='coral')
        plt.xlabel('Model')
        plt.ylabel('Best Throughput (samples/s)')
        plt.title('Best Throughput by Model')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.output_dir}/{self.name}_{self.timestamp}_inference_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _create_quantization_plots(self):
        """Create comparison plots for quantization benchmarks"""
        plt.figure(figsize=(16, 12))
        
        # Extract data for plotting
        configs = [f"{r['config']['type']}_{r['config']['mode']}" for r in self.results["quantization"]]
        quantize_times = [r['performance']['avg_quantize_time_ms'] for r in self.results["quantization"]]
        dequantize_times = [r['performance']['avg_dequantize_time_ms'] for r in self.results["quantization"]]
        throughputs = [r['performance']['throughput_samples_per_second'] for r in self.results["quantization"]]
        errors = [r['accuracy']['mean_absolute_error'] for r in self.results["quantization"]]
        compression_ratios = [r['memory']['compression_ratio'] for r in self.results["quantization"]]
        
        # Timing plot
        plt.subplot(2, 2, 1)
        x = range(len(configs))
        width = 0.35
        
        plt.bar(x, quantize_times, width, label='Quantize Time')
        plt.bar([i + width for i in x], dequantize_times, width, label='Dequantize Time')
        
        plt.xlabel('Configuration')
        plt.ylabel('Time (ms)')
        plt.title('Quantization and Dequantization Time')
        plt.xticks([i + width/2 for i in x], configs, rotation=45)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Throughput plot
        plt.subplot(2, 2, 2)
        plt.bar(configs, throughputs, color='green')
        plt.xlabel('Configuration')
        plt.ylabel('Throughput (samples/second)')
        plt.title('Throughput Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Error plot
        plt.subplot(2, 2, 3)
        plt.bar(configs, errors, color='red')
        plt.xlabel('Configuration')
        plt.ylabel('Mean Absolute Error')
        plt.title('Quantization Error')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Compression ratio plot
        plt.subplot(2, 2, 4)
        plt.bar(configs, compression_ratios, color='purple')
        plt.xlabel('Configuration')
        plt.ylabel('Compression Ratio')
        plt.title('Memory Efficiency')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.output_dir}/{self.name}_{self.timestamp}_quantization_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _create_memory_plots(self):
        """Create memory profile plots"""
        plt.figure(figsize=(12, 6))
        
        # Plot each memory profile
        for idx, profile_data in enumerate(self.results["memory"]):
            profile = profile_data["profile"]
            timestamps = [p[0] for p in profile]
            memory_usage = [p[1] for p in profile]
            
            # Normalize timestamps to start from 0
            start_time = timestamps[0]
            normalized_time = [(t - start_time) for t in timestamps]
            
            plt.plot(normalized_time, memory_usage, 
                    label=f"{profile_data['stage']} - {profile_data['model_name']}")
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Profile During Benchmarks')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plot_path = f"{self.output_dir}/{self.name}_{self.timestamp}_memory_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _generate_recommendations(self):
        """Generate optimization recommendations based on benchmark results"""
        recommendations = []
        
        # Analyze training results
        if self.results["training"]:
            # Check if training is slow
            avg_training_time = sum(r["duration_seconds"] for r in self.results["training"]) / len(self.results["training"])
            if avg_training_time > 60:  # More than 1 minute on average
                recommendations.append("- **Training Performance**: Consider enabling parallel processing or reducing the search space for hyperparameter optimization.")
            
            # Check memory usage during training
            avg_memory_usage = sum(r["memory_usage_mb"] for r in self.results["training"]) / len(self.results["training"])
            system_memory_gb = self.results["system_info"]["memory_gb"]
            
            if avg_memory_usage > system_memory_gb * 1024 * 0.5:  # Using more than 50% of system memory
                recommendations.append("- **Memory Optimization**: Enable memory optimization in the training configuration and consider reducing batch sizes.")
        
        # Analyze inference results
        if self.results["inference"]:
            # Check inference latency
            latencies = [r["avg_latency_ms"] for r in self.results["inference"]]
            if any(l > 100 for l in latencies):  # Any latency over 100ms
                recommendations.append("- **Inference Latency**: Consider enabling quantization or optimizing batch sizes for better inference performance.")
            
            # Check batch size impact
            batch_impacts = {}
            for r in self.results["inference"]:
                model = r["model_name"]
                batch = r["batch_size"]
                throughput = r["throughput_samples_per_second"]
                
                if model not in batch_impacts:
                    batch_impacts[model] = []
                
                batch_impacts[model].append((batch, throughput))
            
            # Generate recommendations based on batch size impact
            for model, impacts in batch_impacts.items():
                if len(impacts) > 1:
                    impacts.sort()  # Sort by batch size
                    
                    # If larger batch sizes have diminishing returns
                    batches = [i[0] for i in impacts]
                    throughputs = [i[1] for i in impacts]
                    
                    if len(batches) >= 3:
                        # Calculate throughput per batch size unit
                        efficiency = [t/b for t, b in zip(throughputs, batches)]
                        
                        if efficiency[0] > efficiency[-1] * 1.5:  # First batch size is 50% more efficient
                            optimal_batch = batches[efficiency.index(max(efficiency))]
                            recommendations.append(f"- **Batch Size Optimization for {model}**: The optimal batch size appears to be around {optimal_batch}. Larger batches show diminishing returns.")
        
        # Analyze quantization results
        if self.results["quantization"]:
            throughputs = [r['performance']['throughput_samples_per_second'] for r in self.results["quantization"]]
            errors = [r['accuracy']['mean_absolute_error'] for r in self.results["quantization"]]
            configs = [f"{r['config']['type']}_{r['config']['mode']}" for r in self.results["quantization"]]
            
            if throughputs and errors:
                # Find best throughput and best accuracy configurations
                best_throughput_idx = throughputs.index(max(throughputs))
                best_accuracy_idx = errors.index(min(errors))
                
                recommendations.append(f"- **Quantization Optimization**: For best throughput, use {configs[best_throughput_idx]} configuration. For best accuracy, use {configs[best_accuracy_idx]} configuration.")
                
                # If INT8 dynamic is among the top performers, recommend it for its versatility
                int8_dynamic_indices = [i for i, c in enumerate(configs) if 'INT8_DYNAMIC' in c]
                if int8_dynamic_indices:
                    int8_dynamic_throughputs = [throughputs[i] for i in int8_dynamic_indices]
                    best_int8_dynamic_idx = int8_dynamic_indices[int8_dynamic_throughputs.index(max(int8_dynamic_throughputs))]
                    
                    if throughputs[best_int8_dynamic_idx] > 0.8 * max(throughputs):  # If INT8 dynamic is within 80% of best
                        recommendations.append(f"- **INT8 Dynamic Quantization**: {configs[best_int8_dynamic_idx]} provides a good balance between performance and accuracy, and adapts well to different data distributions.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("- No specific optimization recommendations based on current benchmark results.")
        
        return "\n".join(recommendations)


class GentaBenchmark:
    """Comprehensive benchmark for Genta AutoML training and inference engines using PLMB dataset"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        """Initialize the benchmark"""
        self.output_dir = output_dir
        
        # Set up logging
        self.logger = logging.getLogger("GentaBenchmark")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default parameters
        self.default_params = {
            "classification": {
                "RandomForestClassifier": {
                    "n_estimators": [10, 50, 100],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "LogisticRegression": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs", "liblinear", "saga"]
                },
                "DecisionTreeClassifier": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "regression": {
                "RandomForestRegressor": {
                    "n_estimators": [10, 50, 100],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "Ridge": {
                    "alpha": [0.1, 1.0, 10.0],
                    "solver": ["auto", "svd", "cholesky"]
                },
                "DecisionTreeRegressor": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }
            }
        }
    
    def _load_plmb_data(self):
        """Load PLMB dataset from OpenML or from disk cache"""
        cache_dir = Path(self.output_dir) / "data_cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        classification_path = cache_dir / "plmb_classification.npz"
        regression_path = cache_dir / "plmb_regression.npz"
        
        # PLMB datasets on OpenML
        classification_dataset_id = '40981'  # PLMB
        regression_dataset_id = '41021'      # PLMB_reg
        
        # Check if cached data exists
        data_dict = {}
        
        # Try to load classification data
        if classification_path.exists():
            self.logger.info(f"Loading classification data from cache: {classification_path}")
            try:
                data = np.load(classification_path)
                data_dict["classification"] = {
                    "X": data["X"],
                    "y": data["y"]
                }
            except Exception as e:
                self.logger.warning(f"Failed to load cached classification data: {e}")
        
        # Try to load regression data
        if regression_path.exists():
            self.logger.info(f"Loading regression data from cache: {regression_path}")
            try:
                data = np.load(regression_path)
                data_dict["regression"] = {
                    "X": data["X"],
                    "y": data["y"]
                }
            except Exception as e:
                self.logger.warning(f"Failed to load cached regression data: {e}")
        
        # Download data if not in cache
        if "classification" not in data_dict:
            self.logger.info(f"Downloading classification dataset (PLMB)...")
            try:
                dataset = fetch_openml(data_id=classification_dataset_id, as_frame=False)
                X = dataset.data
                y = dataset.target
                
                # Save to cache
                np.savez_compressed(classification_path, X=X, y=y)
                
                data_dict["classification"] = {
                    "X": X,
                    "y": y
                }
                self.logger.info(f"Classification data downloaded and cached")
            except Exception as e:
                self.logger.error(f"Failed to download classification data: {e}")
        
        if "regression" not in data_dict:
            self.logger.info(f"Downloading regression dataset (PLMB_reg)...")
            try:
                dataset = fetch_openml(data_id=regression_dataset_id, as_frame=False)
                X = dataset.data
                y = dataset.target
                
                # Save to cache
                np.savez_compressed(regression_path, X=X, y=y)
                
                data_dict["regression"] = {
                    "X": X,
                    "y": y
                }
                self.logger.info(f"Regression data downloaded and cached")
            except Exception as e:
                self.logger.error(f"Failed to download regression data: {e}")
        
        return data_dict
    
    def _get_model_and_params(self, model_name: str, task_type: str) -> tuple:
        """Get model instance and parameter grid based on name and task type"""
        if task_type == "CLASSIFICATION":
            if model_name == "RandomForestClassifier":
                return (
                    RandomForestClassifier(random_state=42),
                    self.default_params["classification"]["RandomForestClassifier"]
                )
            elif model_name == "LogisticRegression":
                return (
                    LogisticRegression(random_state=42, max_iter=1000),
                    self.default_params["classification"]["LogisticRegression"]
                )
            elif model_name == "DecisionTreeClassifier":
                return (
                    DecisionTreeClassifier(random_state=42),
                    self.default_params["classification"]["DecisionTreeClassifier"]
                )
            else:
                raise ValueError(f"Unsupported classification model: {model_name}")
        else:  # REGRESSION
            if model_name == "RandomForestRegressor":
                return (
                    RandomForestRegressor(random_state=42),
                    self.default_params["regression"]["RandomForestRegressor"]
                )
            elif model_name == "Ridge":
                return (
                    Ridge(random_state=42),
                    self.default_params["regression"]["Ridge"]
                )
            elif model_name == "DecisionTreeRegressor":
                return (
                    DecisionTreeRegressor(random_state=42),
                    self.default_params["regression"]["DecisionTreeRegressor"]
                )
            else:
                raise ValueError(f"Unsupported regression model: {model_name}")
    
    def benchmark_training(self, model_names: list, task_types: list, iterations: int = 1) -> dict:
        """
        Benchmark the training engine performance
        
        Args:
            model_names: List of model names to benchmark
            task_types: List of task types (CLASSIFICATION, REGRESSION)
            iterations: Number of iterations for each configuration
            
        Returns:
            Dictionary with benchmark results
        """
        self.logger.info(f"Starting training benchmark with {len(model_names)} models, {iterations} iterations")
        
        # Create results tracker
        results_tracker = BenchmarkResults(
            name="training_benchmark_plmb",
            output_dir=self.output_dir
        )
        
        # Get optimized configurations
        device_optimizer = DeviceOptimizer()
        optimized_configs = device_optimizer.save_configs()
        
        # Load PLMB data
        data_dict = self._load_plmb_data()
        
        for task_type in task_types:
            task_key = task_type.lower()
            if task_key not in data_dict:
                self.logger.warning(f"No data available for task type {task_type}. Skipping.")
                continue
                
            self.logger.info(f"Benchmarking {task_type} models")
            
            # Get data for this task
            X, y = data_dict[task_key]["X"], data_dict[task_key]["y"]
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Log dataset information
            self.logger.info(f"Dataset shape: {X.shape}, Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            
            for model_name in model_names:
                self.logger.info(f"Benchmarking {model_name}")
                
                # Get model and params
                try:
                    model, param_grid = self._get_model_and_params(model_name, task_type)
                except ValueError as e:
                    self.logger.warning(str(e))
                    continue
                
                # Create training engine config
                engine_config = MLTrainingEngineConfig(
                    task_type=TaskType[task_type],
                    random_state=42,
                    n_jobs=-1,  # Use all cores
                    verbose=0,  # Quiet mode
                    cv_folds=3,
                    test_size=0.2,
                    optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
                    optimization_iterations=5,  # Reduced for benchmarking
                    early_stopping=True,
                    feature_selection=True,
                    model_path=os.path.join(self.output_dir, "models")
                )
                
                # Run iterations
                for i in range(iterations):
                    self.logger.info(f"Iteration {i+1}/{iterations} for {model_name}")
                    
                    # Clear previous memory
                    gc.collect()
                    
                    # Initialize training engine
                    train_engine = MLTrainingEngine(engine_config)
                    
                    # Start memory monitoring
                    memory_profile = []
                    
                    # Track memory usage
                    process = psutil.Process(os.getpid())
                    start_memory = process.memory_info().rss / (1024 * 1024)
                    
                    # Record training time
                    start_time = time.time()
                    
                    try:
                        # Train model
                        best_model, metrics = train_engine.train_model(
                            model=model,
                            model_name=model_name,
                            param_grid=param_grid,
                            X=X_train,
                            y=y_train,
                            X_test=X_test,
                            y_test=y_test
                        )
                        
                        # Record end time and memory
                        end_time = time.time()
                        end_memory = process.memory_info().rss / (1024 * 1024)
                        peak_memory = max(start_memory, end_memory)  # Simplified - ideally track peak memory during training
                        
                        # Add result
                        duration = end_time - start_time
                        results_tracker.add_training_result(
                            model_name=model_name,
                            task_type=task_type,
                            dataset_size=X.shape,
                            duration=duration,
                            config=engine_config.to_dict(),
                            metrics=metrics,
                            memory_usage=peak_memory
                        )
                        
                        self.logger.info(f"Training completed in {duration:.2f}s, peak memory: {peak_memory:.1f}MB")
                        
                        # Save the trained model for inference benchmarking
                        model_path = os.path.join(self.output_dir, "models", f"{model_name}_{task_type}.pkl")
                        train_engine.save_model(
                            model_name=model_name,
                            filepath=model_path
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Error benchmarking {model_name}: {str(e)}")
                    finally:
                        # Shutdown the engine
                        train_engine.shutdown()
                        
                        # Collect garbage
                        gc.collect()
        
        # Save and generate report
        results_tracker.save()
        results_tracker.generate_report()
        return results_tracker
    
    def benchmark_quantization(self, quantization_types: list = None, quantization_modes: list = None, iterations: int = 5) -> dict:
        """
        Benchmark different quantization configurations
        
        Args:
            quantization_types: List of quantization types to benchmark
            quantization_modes: List of quantization modes to benchmark
            iterations: Number of iterations for each configuration
            
        Returns:
            Dictionary with benchmark results
        """
        # Set defaults if not provided
        if quantization_types is None:
            quantization_types = ['INT8', 'UINT8', 'INT16']
        if quantization_modes is None:
            quantization_modes = ['SYMMETRIC', 'ASYMMETRIC', 'DYNAMIC_PER_BATCH', 'DYNAMIC_PER_CHANNEL']
        
        self.logger.info(f"Starting quantization benchmark with {len(quantization_types)} types and {len(quantization_modes)} modes")
        
        # Create results tracker
        results_tracker = BenchmarkResults(
            name="quantization_benchmark_plmb",
            output_dir=self.output_dir
        )
        
        # Load PLMB data
        data_dict = self._load_plmb_data()
        
        # Use classification data for quantization benchmark
        if "classification" in data_dict:
            X = data_dict["classification"]["X"]
        elif "regression" in data_dict:
            X = data_dict["regression"]["X"]
        else:
            self.logger.error("No data available for quantization benchmark")
            return results_tracker
        
        # Use a subset of the data for benchmarking
        X = X[:5000].astype(np.float32)
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        
        # Convert string enums to actual enum values
        q_types = [getattr(QuantizationType, qt) for qt in quantization_types]
        q_modes = [getattr(QuantizationMode, qm) for qm in quantization_modes]
        
        # Generate all combinations of configurations
        configs = []
        for q_type in q_types:
            for q_mode in q_modes:
                # Skip invalid combinations (e.g., symmetric quantization doesn't make sense for uint8)
                if q_type == QuantizationType.UINT8 and q_mode == QuantizationMode.SYMMETRIC:
                    continue
                    
                configs.append({
                    'type': q_type,
                    'mode': q_mode
                })
        
        self.logger.info(f"Benchmarking {len(configs)} quantization configurations...")
        
        for config in configs:
            self.logger.info(f"\nBenchmarking {config['type'].name} {config['mode'].name}")
            
            # Create quantization config
            quant_config = QuantizationConfig(
                quantization_type=config['type'].value,
                quantization_mode=config['mode'].value,
                enable_cache=True,
                cache_size=1024,
                buffer_size=1024,
                use_percentile=True,
                min_percentile=0.1,
                max_percentile=99.9,
                error_on_nan=False,
                error_on_inf=False,
                outlier_threshold=3.0,
                optimize_memory=True
            )
            
            # Initialize quantizer
            quantizer = Quantizer(quant_config)
            
            # Clear cache for accurate memory measurements
            gc.collect()
            start_memory = process.memory_info().rss / (1024 * 1024)
            
            # Benchmark quantization
            quantize_times = []
            dequantize_times = []
            errors = []
            
            self.logger.info(f"Running {iterations} iterations...")
            for i in range(iterations):
                try:
                    # Quantize
                    start_time = time.time()
                    quantized_data = quantizer.quantize(X)
                    quantize_time = time.time() - start_time
                    quantize_times.append(quantize_time)
                    
                    # Dequantize
                    start_time = time.time()
                    dequantized_data = quantizer.dequantize(quantized_data)
                    dequantize_time = time.time() - start_time
                    dequantize_times.append(dequantize_time)
                    
                    # Calculate error
                    error = np.mean(np.abs(X - dequantized_data))
                    errors.append(error)
                    
                except Exception as e:
                    self.logger.error(f"Error during iteration {i}: {str(e)}")
                    # Skip this iteration
                    continue
            
            # Calculate memory usage
            end_memory = process.memory_info().rss / (1024 * 1024)
            memory_growth = end_memory - start_memory
            
            # Get quantizer stats
            quant_stats = quantizer.get_quantization_stats()
            
            # Filter out potential failed iterations
            if quantize_times and dequantize_times and errors:
                # Calculate metrics
                avg_quantize_time = sum(quantize_times) / len(quantize_times)
                avg_dequantize_time = sum(dequantize_times) / len(dequantize_times)
                avg_error = sum(errors) / len(errors)
                
                # Calculate compression ratio
                original_size = X.nbytes
                quantized_size = quantized_data.nbytes
                compression_ratio = original_size / max(1, quantized_size)
                
                # Record results
                performance = {
                    "avg_quantize_time_ms": avg_quantize_time * 1000,
                    "avg_dequantize_time_ms": avg_dequantize_time * 1000,
                    "avg_total_time_ms": (avg_quantize_time + avg_dequantize_time) * 1000,
                    "throughput_samples_per_second": X.shape[0] / (avg_quantize_time + avg_dequantize_time)
                }
                
                memory = {
                    "memory_growth_mb": memory_growth,
                    "compression_ratio": compression_ratio
                }
                
                accuracy = {
                    "mean_absolute_error": float(avg_error),
                    "error_stats": quant_stats
                }
                
                # Add to results tracker
                results_tracker.add_quantization_result(
                    config={"type": config['type'].name, "mode": config['mode'].name},
                    performance=performance,
                    memory=memory,
                    accuracy=accuracy
                )
                
                self.logger.info(f"Results for {config['type'].name} {config['mode'].name}:")
                self.logger.info(f"  Average Quantize Time: {avg_quantize_time * 1000:.2f} ms")
                self.logger.info(f"  Average Dequantize Time: {avg_dequantize_time * 1000:.2f} ms")
                self.logger.info(f"  Throughput: {X.shape[0] / (avg_quantize_time + avg_dequantize_time):.2f} samples/second")
                self.logger.info(f"  Mean Absolute Error: {avg_error:.6f}")
                self.logger.info(f"  Compression Ratio: {compression_ratio:.2f}x")
                self.logger.info(f"  Memory Growth: {memory_growth:.2f} MB")
            
            # Clean up
            del quantizer
            gc.collect()
        
        # Save and generate report
        results_tracker.save()
        results_tracker.generate_report()
        return results_tracker
    
    def benchmark_inference(self, model_names: list, task_types: list, batch_sizes: list = [1, 16, 64, 256]) -> dict:
        """
        Benchmark the inference engine performance
        
        Args:
            model_names: List of model names to benchmark
            task_types: List of task types (CLASSIFICATION, REGRESSION)
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        self.logger.info(f"Starting inference benchmark with {len(model_names)} models")
        
        # Create results tracker
        results_tracker = BenchmarkResults(
            name="inference_benchmark_plmb",
            output_dir=self.output_dir
        )
        
        # Get optimized configurations
        device_optimizer = DeviceOptimizer()
        optimized_configs = device_optimizer.save_configs()
        
        # Load PLMB data
        data_dict = self._load_plmb_data()
        
        for task_type in task_types:
            task_key = task_type.lower()
            if task_key not in data_dict:
                self.logger.warning(f"No data available for task type {task_type}. Skipping.")
                continue
                
            self.logger.info(f"Benchmarking {task_type} inference")
            
            # Get data for this task
            X, _ = data_dict[task_key]["X"], data_dict[task_key]["y"]
            
            # Use a portion of the data for inference testing
            X_test = X[:10000]  # Limit to 10,000 samples for inference
            
            for model_name in model_names:
                self.logger.info(f"Benchmarking {model_name} inference")
                
                # Try to load pre-trained model
                model_path = os.path.join(self.output_dir, "models", f"{model_name}_{task_type}.pkl")
                
                if not os.path.exists(model_path):
                    self.logger.warning(f"Model not found at {model_path}. Training a simple model for inference testing.")
                    
                    # Train a simple model
                    model, _ = self._get_model_and_params(model_name, task_type)
                    model.fit(X[:1000], data_dict[task_key]["y"][:1000])
                    
                    # Save the model
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                
                # Test different batch sizes
                for batch_size in batch_sizes:
                    self.logger.info(f"Testing with batch size: {batch_size}")
                    
                    # Create inference engine config
                    inference_config = InferenceEngineConfig(
                        enable_batching=True,
                        batch_timeout=0.01,
                        initial_batch_size=batch_size,
                        min_batch_size=batch_size,
                        max_batch_size=batch_size,
                        enable_adaptive_batching=False,  # Fixed batch size for benchmarking
                        enable_monitoring=True,
                        num_threads=os.cpu_count(),
                        enable_intel_optimization=True,
                        enable_memory_optimization=True
                    )
                    
                    # Initialize inference engine
                    inference_engine = InferenceEngine(inference_config)
                    
                    # Clear previous memory
                    gc.collect()
                    
                    try:
                        # Load model
                        inference_engine.load_model(model_path)
                        
                        # Track memory usage
                        process = psutil.Process(os.getpid())
                        start_memory = process.memory_info().rss / (1024 * 1024)
                        
                        # Record inference time
                        latencies = []
                        start_time = time.time()
                        
                        # Run inference in batches
                        num_batches = len(X_test) // batch_size
                        
                        for i in range(num_batches):
                            batch_start = i * batch_size
                            batch_end = min((i + 1) * batch_size, len(X_test))
                            
                            # Extract batch
                            X_batch = X_test[batch_start:batch_end]
                            
                            # Measure single batch latency
                            batch_start_time = time.time()
                            success, predictions, metadata = inference_engine.predict(X_batch)
                            batch_end_time = time.time()
                            
                            if success:
                                latencies.append((batch_end_time - batch_start_time) * 1000)  # Convert to ms
                            else:
                                self.logger.warning(f"Batch inference failed: {metadata.get('error', 'Unknown error')}")
                        
                        # Record end time and memory
                        end_time = time.time()
                        end_memory = process.memory_info().rss / (1024 * 1024)
                        peak_memory = max(start_memory, end_memory)  # Simplified
                        
                        # Calculate metrics
                        total_duration = end_time - start_time
                        avg_latency = sum(latencies) / max(len(latencies), 1)
                        throughput = (num_batches * batch_size) / total_duration if total_duration > 0 else 0
                        
                        # Add result
                        results_tracker.add_inference_result(
                            model_name=model_name,
                            task_type=task_type,
                            dataset_size=X_test.shape,
                            batch_size=batch_size,
                            total_duration=total_duration,
                            avg_latency=avg_latency,
                            throughput=throughput,
                            config=inference_config.to_dict(),
                            memory_usage=peak_memory
                        )
                        
                        self.logger.info(
                            f"Inference completed in {total_duration:.2f}s, "
                            f"avg latency: {avg_latency:.2f}ms, "
                            f"throughput: {throughput:.1f} samples/s, "
                            f"memory: {peak_memory:.1f}MB"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Inference error for {model_name}: {str(e)}")
                    finally:
                        # Shutdown the engine
                        inference_engine.shutdown()
                        
                        # Collect garbage
                        gc.collect()
        
        # Save and generate report
        results_tracker.save()
        results_tracker.generate_report()
        return results_tracker
    
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Benchmark Genta AutoML')
    
    # General options
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='Directory to save benchmark results')
    parser.add_argument('--mode', type=str, choices=['training', 'inference', 'quantization', 'all'], default='all',
                        help='Benchmark mode')
    
    # Model options
    parser.add_argument('--models', type=str, nargs='+',
                       default=['RandomForestClassifier', 'LogisticRegression', 'DecisionTreeClassifier'],
                       help='Models to benchmark')
    parser.add_argument('--task-types', type=str, nargs='+',
                       default=['CLASSIFICATION', 'REGRESSION'],
                       help='Task types to benchmark')
    
    # Training options
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of iterations for training benchmark')
    
    # Inference options
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 16, 64, 256],
                       help='Batch sizes for inference benchmark')
    
    # Quantization options
    parser.add_argument('--quantization-types', type=str, nargs='+', 
                       default=['INT8', 'UINT8', 'INT16'],
                       help='Quantization types for quantization benchmark')
    parser.add_argument('--quantization-modes', type=str, nargs='+',
                       default=['SYMMETRIC', 'ASYMMETRIC', 'DYNAMIC_PER_BATCH', 'DYNAMIC_PER_CHANNEL'],
                       help='Quantization modes for quantization benchmark')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print("Genta AutoML Comprehensive Benchmark".center(80))
    print("="*80 + "\n")
    
    # Create benchmark
    benchmark = GentaBenchmark(output_dir=args.output_dir)
    
    # Run benchmarks based on mode
    if args.mode in ['training', 'all']:
        print("\n" + "-"*80)
        print("Training Benchmark".center(80))
        print("-"*80 + "\n")
        benchmark.benchmark_training(
            model_names=args.models,
            task_types=args.task_types,
            iterations=args.iterations
        )
    
    if args.mode in ['inference', 'all']:
        print("\n" + "-"*80)
        print("Inference Benchmark".center(80))
        print("-"*80 + "\n")
        benchmark.benchmark_inference(
            model_names=args.models,
            task_types=args.task_types,
            batch_sizes=args.batch_sizes
        )
    
    if args.mode in ['quantization', 'all']:
        print("\n" + "-"*80)
        print("Quantization Benchmark".center(80))
        print("-"*80 + "\n")
        benchmark.benchmark_quantization(
            quantization_types=args.quantization_types,
            quantization_modes=args.quantization_modes,
            iterations=args.iterations
        )
    
    print("\n" + "="*80)
    print("Benchmark completed!".center(80))
    print(f"Results saved to: {args.output_dir}".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()