"""
Experiment tracking with multiple backends including MLflow support.

This module provides comprehensive experiment tracking capabilities for machine learning
workflows with local storage and optional MLflow integration.
"""

import os
import time
import logging
import json
import pickle
import traceback
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path

# For optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from sklearn.metrics import confusion_matrix
from .utils import _json_safe


class ExperimentTracker:
    """
    Track experiments and model performance metrics with various backends.
    
    This class provides a unified interface for experiment tracking, supporting local
    storage as well as MLflow integration. It keeps a history of experiments,
    metrics, and artifact paths.
    """
    
    def __init__(self, output_dir: str = "./experiments", experiment_name: str = None):
        """
        Initialize the experiment tracker.
        
        Args:
            output_dir: Directory to store experiment results
            experiment_name: Name for this experiment series
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.experiment_id = int(time.time())
        self.metrics_history = []
        self.current_experiment = {}
        self.active_run = None
        self.artifacts = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
            
        # Set up logging using centralized configuration
        log_filename = f"experiment_{self.experiment_id}.log"
        try:
            from modules.logging_config import get_logger
            self.logger = get_logger(
                name=f"Experiment_{self.experiment_id}",
                level=logging.INFO,
                log_file=log_filename,
                enable_console=True
            )
        except ImportError:
            # Fallback to basic logging if centralized logging not available
            log_path = f"{output_dir}/experiment_{self.experiment_id}.log"
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(f"Experiment_{self.experiment_id}")
        
        # Try to set up MLflow if available
        self.mlflow_configured = False
        if MLFLOW_AVAILABLE:
            try:
                # Initialize with default tracking URI if not already set
                if mlflow.get_tracking_uri() == "file:./mlruns":
                    mlflow_dir = os.path.join(output_dir, "mlruns")
                    os.makedirs(mlflow_dir, exist_ok=True)
                    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
                
                # Get or create the experiment
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment:
                    self.mlflow_experiment_id = experiment.experiment_id
                else:
                    self.mlflow_experiment_id = mlflow.create_experiment(
                        self.experiment_name,
                        artifact_location=os.path.abspath(os.path.join(output_dir, "artifacts"))
                    )
                    
                self.mlflow_configured = True
                self.logger.info(f"MLflow tracking configured with experiment: {self.experiment_name}")
            except Exception as e:
                self.logger.warning(f"Failed to configure MLflow: {str(e)}")
                self.mlflow_configured = False
        
    def start_experiment(self, config: Dict, model_info: Dict):
        """
        Begin a new experiment run and attach config / model metadata.

        *config* and *model_info* are first passed through
        ``_make_json_serializable`` so that anything written later
        (log‑file or JSON) is guaranteed to be serialisable.
        """
        self.current_experiment = {
            "experiment_id": self.experiment_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": _json_safe(config),
            "model_info": _json_safe(model_info),
            "metrics": {},
            "feature_importance": {},
            "duration": 0,
            "artifacts": {}
        }
        self.start_time = time.time()
        self.logger.info(f"Started experiment {self.experiment_id}")
        self.logger.info("Configuration:\n%s",
                         json.dumps(self.current_experiment["config"], indent=2))
        self.logger.info(f"Model: {self.current_experiment['model_info']}")

        # ------------ (unchanged MLflow initialisation below) ----------
        if self.mlflow_configured:
            try:
                self.active_run = mlflow.start_run(
                    experiment_id=self.mlflow_experiment_id,
                    run_name=f"{model_info.get('model_type', 'unknown')}_{self.experiment_id}"
                )
                # log params …
                for k, v in self.current_experiment["config"].items():
                    if isinstance(v, (str, int, float, bool)):
                        mlflow.log_param(k, v)
                for k, v in self.current_experiment["model_info"].items():
                    if isinstance(v, (str, int, float, bool)):
                        mlflow.log_param(f"model_{k}", v)
            except Exception as e:
                self.logger.warning(f"Failed to start MLflow run: {e}")
                self.active_run = None

    def _make_json_serializable(self, obj):
        """
        Recursively convert *obj* into a structure that the built‑in
        ``json`` encoder can handle.

        Handles:
        • basic Python scalars (str, int, float, bool, None)  
        • enum instances → their ``value``  
        • numpy scalars / ndarrays → Python scalars / lists  
        • containers (dict / list / tuple / set) – processed element‑wise  
        • ``type`` objects → fully‑qualified class string  
        • anything else → ``str(obj)`` fallback
        """
        from enum import Enum
        import numpy as np

        # -------- JSON‑native types -----------------------------------
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # -------- numpy scalars / arrays ------------------------------
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # -------- Enum -> value ---------------------------------------
        if isinstance(obj, Enum):
            return obj.value

        # -------- containers ------------------------------------------
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, set)):
            return [self._make_json_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._make_json_serializable(v) for v in obj)

        # -------- class / type objects --------------------------------
        if isinstance(obj, type):
            return f"<class '{obj.__module__}.{obj.__name__}'>"

        # -------- graceful fallback -----------------------------------
        try:
            return str(obj)
        except Exception:
            return f"<non‑serializable:{type(obj).__name__}>"
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[str] = None):
        """
        Log metrics for the current experiment.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Optional step name to organize metrics
        """
        if step:
            if "steps" not in self.current_experiment:
                self.current_experiment["steps"] = {}
            self.current_experiment["steps"][step] = metrics
        else:
            self.current_experiment["metrics"].update(metrics)
            
        self.logger.info(f"Metrics {f'for {step}' if step else ''}: {metrics}")
        
        # Log to MLflow if configured
        if self.mlflow_configured and self.active_run:
            try:
                # Prepare step number if needed
                step_num = None
                if step and step.isdigit():
                    step_num = int(step)
                
                # Log each metric
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if step:
                            # Include step in metric name if not a number
                            metric_name = f"{step}_{name}" if not step_num else name
                            mlflow.log_metric(metric_name, value, step=step_num)
                        else:
                            mlflow.log_metric(name, value)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to MLflow: {str(e)}")
        
    def log_feature_importance(self, feature_names: List[str], importance: np.ndarray):
        """
        Log feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance: Array of importance scores
        """
        # Convert to dictionary
        feature_importance = {name: float(score) for name, score in zip(feature_names, importance)}
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Store in experiment
        self.current_experiment["feature_importance"] = sorted_importance
        self.logger.info(f"Feature importance: {json.dumps(dict(list(sorted_importance.items())[:10]), indent=2)}")
        
        # Log to MLflow if configured
        if self.mlflow_configured and self.active_run and PLOTTING_AVAILABLE:
            try:
                # Create and save feature importance plot
                top_n = min(20, len(feature_names))
                plt.figure(figsize=(10, 8))
                top_features = list(sorted_importance.items())[:top_n]
                
                # Plot in horizontal bar chart
                names = [item[0] for item in top_features]
                values = [item[1] for item in top_features]
                
                # Create bar plot with better colors
                plt.barh(range(len(names)), values, align='center', color='#3498db')
                plt.yticks(range(len(names)), names)
                plt.xlabel('Importance')
                plt.title(f'Top {top_n} Feature Importance')
                plt.tight_layout()
                
                # Save locally
                importance_plot_path = os.path.join(self.output_dir, f"feature_importance_{self.experiment_id}.png")
                plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Log to MLflow
                mlflow.log_artifact(importance_plot_path, "feature_importance")
                
                # Add to artifacts
                self.artifacts["feature_importance_plot"] = importance_plot_path
                self.current_experiment["artifacts"]["feature_importance_plot"] = importance_plot_path
                
                # Also log as metrics for top features
                for name, value in top_features:
                    mlflow.log_metric(f"importance_{name}", value)
                    
            except Exception as e:
                self.logger.warning(f"Failed to log feature importance plot: {str(e)}")
        
    def log_model(self, model, model_name: str, path: str = None):
        """
        Log a trained model to the experiment.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            path: Optional path to save the model
        """
        # Determine save path
        if path is None:
            path = os.path.join(self.output_dir, f"{model_name}_{self.experiment_id}.pkl")
            
        # Save model locally
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
                
            self.artifacts[f"model_{model_name}"] = path
            self.current_experiment["artifacts"][f"model_{model_name}"] = path
            self.logger.info(f"Saved model {model_name} to {path}")
            
            # Log to MLflow if configured
            if self.mlflow_configured and self.active_run:
                try:
                    # Log model as artifact
                    mlflow.sklearn.log_model(model, f"model_{model_name}")
                    mlflow.log_artifact(path, "models")
                except Exception as e:
                    self.logger.warning(f"Failed to log model to MLflow: {str(e)}")
                    # Fallback to simple artifact logging
                    try:
                        mlflow.log_artifact(path, "models")
                    except Exception as e2:
                        self.logger.warning(f"Failed to log model artifact to MLflow: {str(e2)}")
                        
        except Exception as e:
            self.logger.error(f"Failed to save model {model_name}: {str(e)}")
            
    def log_confusion_matrix(self, y_true, y_pred, class_names=None):
        """
        Log a confusion matrix for classification tasks.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names
        """
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib and seaborn not available for confusion matrix plotting")
            return
            
        try:
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            
            # Use seaborn for better styling
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            # Save locally
            os.makedirs(self.output_dir, exist_ok=True)  # Ensure directory exists
            cm_path = os.path.join(self.output_dir, f"confusion_matrix_{self.experiment_id}.png")
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Store in artifacts
            self.artifacts["confusion_matrix"] = cm_path
            self.current_experiment["artifacts"]["confusion_matrix"] = cm_path
            
            # Log to MLflow if configured
            if self.mlflow_configured and self.active_run:
                try:
                    mlflow.log_artifact(cm_path, "evaluation")
                except Exception as e:
                    self.logger.warning(f"Failed to log confusion matrix to MLflow: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Failed to create confusion matrix: {str(e)}")
            self.logger.error(traceback.format_exc()) 
    
    def end_experiment(self):
        """
        Finalise the experiment, write all metadata to disk, and
        close any active MLflow run.  Returns the serialisable dict
        that was written to *experiment_{id}.json*.
        """
        self.current_experiment["duration"] = time.time() - self.start_time
        self.metrics_history.append(self.current_experiment)

        serialisable = self._make_json_serializable(self.current_experiment)
        experiment_file = os.path.join(
            self.output_dir, f"experiment_{self.experiment_id}.json"
        )
        with open(experiment_file, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)

        self.logger.info(
            f"Experiment completed in {serialisable['duration']:.2f} seconds"
        )
        self.logger.info(f"Results saved to {experiment_file}")

        # ------------ (unchanged MLflow tear‑down below) --------------
        if self.mlflow_configured and self.active_run:
            try:
                mlflow.log_metric("duration_seconds", serialisable["duration"])
                mlflow.end_run()
                self.logger.info("MLflow run ended")
                self.active_run = None
            except Exception as e:
                self.logger.warning(f"Failed to end MLflow run: {e}")

        return serialisable
    
    def generate_report(self, report_path: Optional[str] = None, include_plots: bool = True):
        """
        Generate a comprehensive report of the experiment in Markdown format.
        
        Args:
            report_path: Path to save the report (defaults to output_dir/report_{experiment_id}.md)
            include_plots: Whether to include plots in the report
            
        Returns:
            Path to the generated report
        """
        if not self.current_experiment:
            self.logger.warning("No active experiment to generate report")
            return None
            
        # Determine report path
        if report_path is None:
            report_path = os.path.join(self.output_dir, f"report_{self.experiment_id}.md")
            
        # Create report content
        report = f"# Experiment Report: {self.experiment_name}\n\n"
        report += f"**Experiment ID:** {self.experiment_id}  \n"
        report += f"**Date:** {self.current_experiment['timestamp']}  \n"
        report += f"**Duration:** {self.current_experiment['duration']:.2f} seconds  \n\n"
        
        # Add configuration section
        report += "## Configuration\n\n"
        report += "| Parameter | Value |\n"
        report += "| --- | --- |\n"
        
        for key, value in self.current_experiment['config'].items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                report += f"| {key} | {value} |\n"
                
        # Add model info section
        report += "\n## Model Information\n\n"
        report += "| Parameter | Value |\n"
        report += "| --- | --- |\n"
        
        for key, value in self.current_experiment['model_info'].items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                report += f"| {key} | {value} |\n"
                
        # Add metrics section
        report += "\n## Performance Metrics\n\n"
        
        if self.current_experiment['metrics']:
            report += "| Metric | Value |\n"
            report += "| --- | --- |\n"
            
            for metric, value in self.current_experiment['metrics'].items():
                if isinstance(value, (int, float)):
                    report += f"| {metric} | {value:.4f} |\n"
                else:
                    report += f"| {metric} | {value} |\n"
        else:
            report += "No overall metrics recorded.\n"
            
        # Add step metrics if available
        if 'steps' in self.current_experiment and self.current_experiment['steps']:
            report += "\n## Step Metrics\n\n"
            
            for step, metrics in self.current_experiment['steps'].items():
                report += f"### Step: {step}\n\n"
                report += "| Metric | Value |\n"
                report += "| --- | --- |\n"
                
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        report += f"| {metric} | {value:.4f} |\n"
                    else:
                        report += f"| {metric} | {value} |\n"
                        
                report += "\n"
                
        # Add feature importance section
        if self.current_experiment['feature_importance']:
            report += "\n## Feature Importance\n\n"
            
            # Get top features (limited to 20)
            top_features = list(self.current_experiment['feature_importance'].items())[:20]
            
            report += "| Feature | Importance |\n"
            report += "| --- | --- |\n"
            
            for feature, importance in top_features:
                report += f"| {feature} | {importance:.4f} |\n"
                
            report += "\n"
            
            # Include feature importance plot if available
            if include_plots and "feature_importance_plot" in self.artifacts:
                plot_path = self.artifacts["feature_importance_plot"]
                rel_path = os.path.relpath(plot_path, os.path.dirname(report_path))
                report += f"![Feature Importance]({rel_path})\n\n"
                
        # Add confusion matrix if available
        if include_plots and "confusion_matrix" in self.artifacts:
            report += "\n## Confusion Matrix\n\n"
            plot_path = self.artifacts["confusion_matrix"]
            rel_path = os.path.relpath(plot_path, os.path.dirname(report_path))
            report += f"![Confusion Matrix]({rel_path})\n\n"
            
        # Add artifacts section
        if self.current_experiment['artifacts']:
            report += "\n## Artifacts\n\n"
            
            for name, path in self.current_experiment['artifacts'].items():
                if os.path.exists(path):
                    rel_path = os.path.relpath(path, os.path.dirname(report_path))
                    report += f"* [{name}]({rel_path})\n"
                else:
                    report += f"* {name}: {path} (file not found)\n"
                    
        # Add MLflow link if configured
        if self.mlflow_configured and self.active_run:
            tracking_uri = mlflow.get_tracking_uri()
            if tracking_uri.startswith("http"):
                report += f"\n## MLflow Tracking\n\n"
                report += f"* [View experiment in MLflow]({tracking_uri}/#/experiments/{self.mlflow_experiment_id})\n"
                if self.active_run:
                    run_id = self.active_run.info.run_id
                    report += f"* [View run in MLflow]({tracking_uri}/#/experiments/{self.mlflow_experiment_id}/runs/{run_id})\n"
                    
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(report)
            
        self.logger.info(f"Report generated: {report_path}")
        return report_path


__all__ = ['ExperimentTracker']
