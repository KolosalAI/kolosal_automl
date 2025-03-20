import numpy as np
import os
import pickle
import joblib
import time
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from enum import Enum
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    StratifiedKFold, KFold, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import warnings
from tqdm import tqdm
import gc
import json
import traceback
# Import custom components from the provided code snippets
from modules.configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig
)
from modules.engine.inference_engine import InferenceEngine
from modules.engine.batch_processor import BatchProcessor
from modules.engine.data_preprocessor import DataPreprocessor
from modules.engine.quantizer import Quantizer
from modules.optimizer.asht import ASHTOptimizer
from modules.optimizer.hyperoptx import HyperOptX

class ExperimentTracker:
    """Track experiments and model performance metrics"""
    
    def __init__(self, output_dir: str = "./experiments"):
        self.output_dir = output_dir
        self.experiment_id = int(time.time())
        self.metrics_history = []
        self.current_experiment = {}
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{output_dir}/experiment_{self.experiment_id}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"Experiment_{self.experiment_id}")
        
    def start_experiment(self, config: Dict, model_info: Dict):
        """Start a new experiment with configuration and model info"""
        self.current_experiment = {
            "experiment_id": self.experiment_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config,
            "model_info": model_info,
            "metrics": {},
            "feature_importance": {},
            "duration": 0
        }
        self.start_time = time.time()
        self.logger.info(f"Started experiment {self.experiment_id}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        self.logger.info(f"Model: {model_info}")
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[str] = None):
        """Log metrics for the current experiment"""
        if step:
            if "steps" not in self.current_experiment:
                self.current_experiment["steps"] = {}
            self.current_experiment["steps"][step] = metrics
        else:
            self.current_experiment["metrics"].update(metrics)
            
        self.logger.info(f"Metrics {f'for {step}' if step else ''}: {metrics}")
        
    def log_feature_importance(self, feature_names: List[str], importance: np.ndarray):
        """Log feature importance scores"""
        feature_importance = {name: float(score) for name, score in zip(feature_names, importance)}
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        self.current_experiment["feature_importance"] = sorted_importance
        self.logger.info(f"Feature importance: {json.dumps(dict(list(sorted_importance.items())[:10]), indent=2)}")
        
    def end_experiment(self):
        """End the current experiment and save results"""
        duration = time.time() - self.start_time
        self.current_experiment["duration"] = duration
        self.metrics_history.append(self.current_experiment)
        
        # Save experiment results
        experiment_file = f"{self.output_dir}/experiment_{self.experiment_id}.json"
        with open(experiment_file, 'w') as f:
            json.dump(self.current_experiment, f, indent=2)
            
        self.logger.info(f"Experiment completed in {duration:.2f} seconds")
        self.logger.info(f"Results saved to {experiment_file}")
        
        return self.current_experiment
    
    def generate_report(self, output_file=None):
        """Generate a comprehensive report of all models in Markdown format"""
        if not self.models:
            self.logger.warning("No models to generate report")
            return None
                
        if output_file is None:
            output_file = os.path.join(self.config.model_path, "model_report.md")
                
        # Create basic report
        report = f"# ML Training Engine Report\n\n"
        report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add configuration section
        report += "## Configuration\n\n"
        report += "| Parameter | Value |\n"
        report += "| --- | --- |\n"
        
        # Add configuration
        for key, value in self.config.to_dict().items():
            report += f"| {key} | {value} |\n"
        
        report += "\n## Model Performance Summary\n\n"
        
        # Collect all metrics across models
        all_metrics = set()
        for model_data in self.models.values():
            all_metrics.update(model_data["metrics"].keys())
        
        # Create metrics table header
        report += "| Model | " + " | ".join(sorted(all_metrics)) + " |\n"
        report += "| --- | " + " | ".join(["---" for _ in all_metrics]) + " |\n"
        
        # Add model rows
        for model_name, model_data in self.models.items():
            is_best = self.best_model and self.best_model == model_name
            model_label = f"{model_name} **[BEST]**" if is_best else model_name
            
            row = f"| {model_label} |"
            for metric in sorted(all_metrics):
                value = model_data["metrics"].get(metric, "N/A")
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
                row += f" {value} |"
            
            report += row + "\n"
        
        # Add model details section
        report += "\n## Model Details\n\n"
        
        for model_name, model_data in self.models.items():
            report += f"### {model_name}\n\n"
            
            # Add model type
            report += f"- **Type**: {type(model_data.get('model', '')).__name__}\n"
            
            # Add if it's the best model
            is_best = self.best_model and self.best_model == model_name
            report += f"- **Best Model**: {'Yes' if is_best else 'No'}\n"
            
            # Add training time if available
            if "training_time" in model_data:
                report += f"- **Training Time**: {model_data['training_time']:.2f}s\n"
            
            # Add hyperparameters if available
            if "params" in model_data and model_data["params"]:
                report += "\n#### Hyperparameters\n\n"
                report += "| Parameter | Value |\n"
                report += "| --- | --- |\n"
                
                for param, value in model_data["params"].items():
                    report += f"| {param} | {value} |\n"
                
                report += "\n"
            
            # Add feature importance if available
            if "feature_importance" in model_data and model_data["feature_importance"] is not None:
                feature_importance = model_data["feature_importance"]
                if isinstance(feature_importance, dict):
                    # Sort by importance and get top 10
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    report += "\n#### Top 10 Features by Importance\n\n"
                    report += "| Feature | Importance |\n"
                    report += "| --- | --- |\n"
                    
                    for feature, importance in sorted_features:
                        report += f"| {feature} | {importance:.4f} |\n"
                    
                    report += "\n"
                elif isinstance(feature_importance, np.ndarray):
                    # For numpy array, we need feature names
                    feature_names = model_data.get("feature_names", [f"feature_{i}" for i in range(len(feature_importance))])
                    
                    # Sort by importance and get top 10
                    indices = np.argsort(feature_importance)[::-1][:10]
                    
                    report += "\n#### Top 10 Features by Importance\n\n"
                    report += "| Feature | Importance |\n"
                    report += "| --- | --- |\n"
                    
                    for idx in indices:
                        report += f"| {feature_names[idx]} | {feature_importance[idx]:.4f} |\n"
                    
                    report += "\n"
        
        # Add conclusion
        report += "\n## Conclusion\n\n"
        
        if self.best_model:
            best_model_name = self.best_model
            best_metrics = self.models[best_model_name]["metrics"]
            
            report += f"The best performing model is **{best_model_name}** with metrics:\n\n"
            
            # List key metrics
            key_metrics = []
            if self.config.task_type == TaskType.CLASSIFICATION:
                key_metrics = ["accuracy", "f1", "precision", "recall"]
            elif self.config.task_type == TaskType.REGRESSION:
                key_metrics = ["rmse", "mse", "r2", "mae"]
                
            for metric in key_metrics:
                if metric in best_metrics:
                    report += f"- **{metric}**: {best_metrics[metric]:.4f}\n"
        else:
            report += "No models have been trained or evaluated yet."
        
        # Write report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"Report generated: {output_file}")
        return output_file
        
    def _generate_plots(self):
        """Generate plots for the experiment"""
        # Create plots directory
        plots_dir = f"{self.output_dir}/plots_{self.experiment_id}"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        # Plot feature importance if available
        if "feature_importance" in self.current_experiment and self.current_experiment["feature_importance"]:
            plt.figure(figsize=(12, 8))
            features = list(self.current_experiment["feature_importance"].keys())[:15]
            importances = [self.current_experiment["feature_importance"][f] for f in features]
            
            sns.barplot(x=importances, y=features)
            plt.title("Top 15 Features by Importance")
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/feature_importance.png")
            plt.close()
            
        # If CV results are available
        if "steps" in self.current_experiment and "cv" in self.current_experiment["steps"]:
            cv_results = self.current_experiment["steps"]["cv"]
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(cv_results)), list(cv_results.values()))
            plt.xticks(range(len(cv_results)), list(cv_results.keys()), rotation=45)
            plt.title("Cross-Validation Scores")
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/cv_scores.png")
            plt.close()


class MLTrainingEngine:
    """Advanced training engine for machine learning models with optimization"""
    VERSION = "0.1.0"  # Add this version attribute
    def __init__(self, config: MLTrainingEngineConfig):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = -float('inf') if config.task_type != TaskType.REGRESSION else float('inf')
        self.preprocessor = None
        self.feature_selector = None
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("MLTrainingEngine")
        
        # Initialize experiment tracker if enabled
        if config.experiment_tracking:
            self.tracker = ExperimentTracker(f"{config.model_path}/experiments")
        else:
            self.tracker = None
            
        # Create model directory if it doesn't exist
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
            
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize all engine components"""
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(self.config.preprocessing_config)
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(self.config.batch_processing_config)
        
        # Initialize inference engine
        self.inference_engine = InferenceEngine(self.config.inference_config)
        
        # Initialize quantizer
        self.quantizer = Quantizer(self.config.quantization_config)
        
        self.logger.info("All components initialized successfully")
        
    def _get_feature_selector(self, X, y):
        """Get appropriate feature selector based on configuration"""
        if not self.config.feature_selection:
            return None
            
        if self.config.task_type == TaskType.CLASSIFICATION:
            if self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_classif, k=self.config.feature_selection_k)
            else:
                selector = SelectKBest(f_classif, k=self.config.feature_selection_k)
        else:
            # For regression, use mutual information regression
            from sklearn.feature_selection import mutual_info_regression, f_regression
            if self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_regression, k=self.config.feature_selection_k)
            else:
                selector = SelectKBest(f_regression, k=self.config.feature_selection_k)
                
        # If k is not specified, select based on percentile
        if self.config.feature_selection_k is None:
            # Use all features but get their scores for later filtering
            selector.k = 'all'
            
        return selector
            
    def _create_pipeline(self, model):
        """Create a pipeline with preprocessing and model"""
        steps = []
        
        if self.preprocessor:
            steps.append(('preprocessor', self.preprocessor))
            
        if self.feature_selector:
            steps.append(('feature_selector', self.feature_selector))
            
        steps.append(('model', model))
        
        return Pipeline(steps)
    
    def _get_cv_splitter(self, y=None):
        """Get appropriate cross-validation splitter based on task type"""
        if self.config.task_type == TaskType.CLASSIFICATION and self.config.stratify and y is not None:
            return StratifiedKFold(n_splits=self.config.cv_folds, 
                                  shuffle=True, 
                                  random_state=self.config.random_state)
        else:
            return KFold(n_splits=self.config.cv_folds, 
                        shuffle=True, 
                        random_state=self.config.random_state)
    
    def _get_optimization_search(self, model, param_grid):
        """Get the appropriate hyperparameter search based on strategy"""
        cv = self._get_cv_splitter()
        
        if self.config.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
            return GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                scoring=self._get_scoring_metric()
            )
        elif self.config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
            return RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=self.config.optimization_iterations,
                cv=cv,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                random_state=self.config.random_state,
                scoring=self._get_scoring_metric()
            )
        elif self.config.optimization_strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
            # Check if scikit-optimize is available
            try:
                from skopt import BayesSearchCV
                return BayesSearchCV(
                    estimator=model,
                    search_spaces=param_grid,
                    n_iter=self.config.optimization_iterations,
                    cv=cv,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state,
                    scoring=self._get_scoring_metric()
                )
            except ImportError:
                self.logger.warning("scikit-optimize not installed. Falling back to RandomizedSearchCV.")
                return RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=self.config.optimization_iterations,
                    cv=cv,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state,
                    scoring=self._get_scoring_metric()
                )
        elif self.config.optimization_strategy == OptimizationStrategy.ASHT:
            # Use our new ASHT optimizer
            return ASHTOptimizer(
                estimator=model,
                param_space=param_grid,
                max_iter=self.config.optimization_iterations,
                cv=self.config.cv_folds,
                scoring=self._get_scoring_metric(),
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            )
        elif self.config.optimization_strategy == OptimizationStrategy.HYPERX:
            # Implementation for HyperOptX optimizer
            return HyperOptX(
                estimator=model,
                param_space=param_grid,
                max_iter=self.config.optimization_iterations,
                cv=self.config.cv_folds,
                scoring=self._get_scoring_metric(),
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            )
        else:
            self.logger.warning(f"Unsupported optimization strategy: {self.config.optimization_strategy}. Using RandomizedSearchCV.")
            return RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=self.config.optimization_iterations,
                cv=cv,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                random_state=self.config.random_state,
                scoring=self._get_scoring_metric()
            )
    
    def _get_scoring_metric(self):
        """Get appropriate scoring metric based on task type"""
        if self.config.task_type == TaskType.CLASSIFICATION:
            return "f1_weighted"
        elif self.config.task_type == TaskType.REGRESSION:
            return "neg_mean_squared_error"
        elif self.config.task_type == TaskType.CLUSTERING:
            return "silhouette"
        else:
            return "accuracy" 

    def evaluate_model(self, model_name=None, X_test=None, y_test=None, detailed=False):
        """
        Evaluate a model with comprehensive metrics and analysis.
        
        Args:
            model_name: Name of the model to evaluate (uses best model if None)
            X_test: Test features (uses cached test data if None)
            y_test: Test targets (uses cached test data if None)
            detailed: Whether to perform detailed evaluation with additional metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Determine which model to use
        if model_name is None and self.best_model is not None:
            model = self.best_model["model"]
            model_name = self.best_model["name"]
        elif model_name in self.models:
            model = self.models[model_name]["model"]
        else:
            self.logger.error(f"Model {model_name} not found")
            return {"error": f"Model {model_name} not found"}
            
        # Use provided test data or fall back to cached data
        if X_test is None or y_test is None:
            if hasattr(self, '_last_X_test') and hasattr(self, '_last_y_test'):
                X_test = self._last_X_test
                y_test = self._last_y_test
                self.logger.info("Using cached test data for evaluation")
            else:
                self.logger.error("No test data provided and no cached test data available")
                return {"error": "No test data available"}
                
        self.logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Apply preprocessing if needed
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                try:
                    X_test = self.preprocessor.transform(X_test)
                except Exception as e:
                    self.logger.error(f"Preprocessing failed during evaluation: {str(e)}")
                    return {"error": f"Preprocessing error: {str(e)}"}
            
            # Time the prediction
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Basic metrics based on task type
            metrics = {"prediction_time": pred_time}
            
            if self.config.task_type == TaskType.CLASSIFICATION:
                # Classification metrics
                metrics.update({
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted"),
                    "recall": recall_score(y_test, y_pred, average="weighted"),
                    "f1": f1_score(y_test, y_pred, average="weighted")
                })
                
                # Add ROC AUC if binary classification
                if len(np.unique(y_test)) == 2:
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_prob = model.predict_proba(X_test)[:, 1]
                            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
                    except (AttributeError, IndexError) as e:
                        self.logger.warning(f"Could not calculate ROC AUC: {str(e)}")
                
                # Add detailed classification metrics if requested
                if detailed:
                    try:
                        from sklearn.metrics import classification_report, confusion_matrix
                        report = classification_report(y_test, y_pred, output_dict=True)
                        metrics["detailed_report"] = report
                        
                        cm = confusion_matrix(y_test, y_pred)
                        metrics["confusion_matrix"] = cm.tolist()
                        
                        # Calculate per-class metrics
                        class_metrics = {}
                        classes = np.unique(y_test)
                        for cls in classes:
                            class_metrics[str(cls)] = {
                                "precision": precision_score(y_test, y_pred, labels=[cls], average=None)[0],
                                "recall": recall_score(y_test, y_pred, labels=[cls], average=None)[0],
                                "f1": f1_score(y_test, y_pred, labels=[cls], average=None)[0],
                                "support": np.sum(y_test == cls)
                            }
                        metrics["per_class"] = class_metrics
                    except Exception as e:
                        self.logger.warning(f"Could not calculate detailed metrics: {str(e)}")
                        
            elif self.config.task_type == TaskType.REGRESSION:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                metrics.update({
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred)
                })
                
                # Add detailed regression metrics if requested
                if detailed:
                    try:
                        # Calculate median absolute error
                        from sklearn.metrics import median_absolute_error, explained_variance_score
                        metrics["median_absolute_error"] = median_absolute_error(y_test, y_pred)
                        metrics["explained_variance"] = explained_variance_score(y_test, y_pred)
                        
                        # Calculate absolute percentage error
                        if not np.any(y_test == 0):  # Avoid division by zero
                            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                            metrics["mape"] = mape
                        
                        # Calculate residuals for analysis
                        residuals = y_test - y_pred
                        metrics["residuals_mean"] = np.mean(residuals)
                        metrics["residuals_std"] = np.std(residuals)
                    except Exception as e:
                        self.logger.warning(f"Could not calculate detailed metrics: {str(e)}")
            
            # Update model metrics
            if model_name in self.models:
                self.models[model_name]["metrics"] = metrics
                self.models[model_name]["last_evaluated"] = time.time()
                
                # Check if this affects best model ranking
                self._update_best_model(model_name)
                
            # Log evaluation results
            self.logger.info(f"Evaluation results for {model_name}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {metric}: {value:.4f}")
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _evaluate_model(self, model, X, y, X_test=None, y_test=None):
        """Evaluate model performance with appropriate metrics"""
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            
            if self.config.task_type == TaskType.CLASSIFICATION:
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted"),
                    "recall": recall_score(y_test, y_pred, average="weighted"),
                    "f1": f1_score(y_test, y_pred, average="weighted")
                }
                
                # Add ROC AUC if binary classification
                if len(np.unique(y)) == 2:
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
                    except (AttributeError, IndexError):
                        pass
                    
            elif self.config.task_type == TaskType.REGRESSION:
                metrics = {
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred)
                }
            else:
                # Default metrics
                metrics = {"score": model.score(X_test, y_test)}
                
            return metrics
            
        else:
            # Use cross-validation if no test set provided
            cv = self._get_cv_splitter(y)
            if self.config.task_type == TaskType.CLASSIFICATION:
                metrics = {
                    "accuracy": np.mean(cross_val_score(model, X, y, cv=cv, scoring="accuracy")),
                    "f1": np.mean(cross_val_score(model, X, y, cv=cv, scoring="f1_weighted"))
                }
            elif self.config.task_type == TaskType.REGRESSION:
                metrics = {
                    "neg_mse": np.mean(cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")),
                    "r2": np.mean(cross_val_score(model, X, y, cv=cv, scoring="r2"))
                }
                # Convert negative MSE to positive and add RMSE
                metrics["mse"] = -metrics["neg_mse"]
                metrics["rmse"] = np.sqrt(metrics["mse"])
                del metrics["neg_mse"]
            else:
                metrics = {"cv_score": np.mean(cross_val_score(model, X, y, cv=cv))}
                
            return metrics
            
    def _get_feature_importance(self, model):
        """Extract feature importance from the model with enhanced handling of different formats"""
        # Try different attributes that might contain feature importance
        for attr in ['feature_importances_', 'coef_', 'feature_importance_']:
            if hasattr(model, attr):
                importance = getattr(model, attr)
                
                # Convert to numpy array if it's not already
                if not isinstance(importance, np.ndarray):
                    try:
                        importance = np.array(importance)
                    except Exception:
                        continue
                
                # Handle different shapes
                if attr == 'coef_':
                    if importance.ndim > 1:
                        # For multi-class models, take the mean absolute coefficient
                        importance = np.mean(np.abs(importance), axis=0)
                    elif importance.ndim == 1:
                        # For binary classification or regression, take absolute values
                        importance = np.abs(importance)
                
                # Normalize importance scores to sum to 1
                if importance.sum() != 0:
                    importance = importance / importance.sum()
                    
                return importance
                
        # Try permutation importance for models without built-in feature importance
        if self.config.compute_permutation_importance and hasattr(model, 'predict'):
            try:
                from sklearn.inspection import permutation_importance
                if hasattr(self, '_last_X_train') and hasattr(self, '_last_y_train'):
                    # Using cached data if available
                    result = permutation_importance(
                        model, self._last_X_train, self._last_y_train,
                        n_repeats=5, random_state=self.config.random_state
                    )
                    return result.importances_mean
            except Exception as e:
                self.logger.warning(f"Permutation importance calculation failed: {str(e)}")
                    
        # If we reach here, model doesn't have built-in feature importance
        self.logger.warning("Model doesn't provide feature importance.")
        return None

    def _validate_training_data(self, X, y):
        """Validate training data to catch issues early"""
        # Check for None
        if X is None or y is None:
            raise ValueError("X and y must not be None")
            
        # Check for empty data
        if hasattr(X, 'shape') and (X.shape[0] == 0 or (len(X.shape) > 1 and X.shape[1] == 0)):
            raise ValueError("X contains no data")
            
        if hasattr(y, 'shape') and y.shape[0] == 0:
            raise ValueError("y contains no data")
            
        # Check for mismatch in sample counts
        if hasattr(X, 'shape') and hasattr(y, 'shape') and X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have different numbers of samples: {X.shape[0]} vs {y.shape[0]}")
            
        # Check for NaN or infinity
        if hasattr(X, 'isnull') and X.isnull().any().any():
            self.logger.warning("X contains NaN values")
            
        if hasattr(y, 'isnull') and y.isnull().any().any():
            self.logger.warning("y contains NaN values")
            
        # For numpy arrays
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.number):
            if np.isnan(X).any():
                self.logger.warning("X contains NaN values")
            if np.isinf(X).any():
                self.logger.warning("X contains infinite values")
                
        if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.number):
            if np.isnan(y).any():
                self.logger.warning("y contains NaN values")
            if np.isinf(y).any():
                self.logger.warning("y contains infinite values")
                
        # Check for sufficient data
        if hasattr(X, 'shape') and X.shape[0] < 10:
            self.logger.warning(f"Very small dataset with only {X.shape[0]} samples")
            
        # For classification, check class imbalance
        if self.config.task_type == TaskType.CLASSIFICATION:
            if hasattr(y, 'value_counts'):
                class_counts = y.value_counts()
                min_class = class_counts.min()
                max_class = class_counts.max()
                if min_class / max_class < 0.1:
                    self.logger.warning(f"Severe class imbalance detected: min_class={min_class}, max_class={max_class}")
            elif hasattr(y, 'shape'):
                unique, counts = np.unique(y, return_counts=True)
                min_class = counts.min()
                max_class = counts.max()
                if min_class / max_class < 0.1:
                    self.logger.warning(f"Severe class imbalance detected: min_class={min_class}, max_class={max_class}")

    def train_model(self, model, model_name, param_grid, X, y, X_test=None, y_test=None, model_metadata=None):
        """
        Train a model with hyperparameter optimization, enhanced error handling, and secure storing.
        
        Args:
            model: The base model to train
            model_name: Name for the model
            param_grid: Grid of hyperparameters to search
            X: Training features
            y: Training target
            X_test: Optional test features
            y_test: Optional test target
            model_metadata: Optional additional metadata to store with the model
            
        Returns:
            Tuple of (best_model, metrics)
        """
        self.logger.info(f"Training model: {model_name}")
        
        if self.tracker:
            self.tracker.start_experiment(
                config=self.config.to_dict(),
                model_info={"name": model_name, "type": str(type(model).__name__)}
            )
            
        # Track timing for performance monitoring
        start_time = time.time()
        
        try:
            # Prepare data splits if not provided
            if X_test is None or y_test is None:
                if self.config.task_type == TaskType.CLASSIFICATION and self.config.stratify:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=self.config.test_size,
                        random_state=self.config.random_state,
                        stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
            else:
                X_train, y_train = X, y
                
            # Input validation
            self._validate_training_data(X_train, y_train)
                
            # Initialize and fit preprocessor with enhanced error handling
            if self.preprocessor:
                self.logger.info("Fitting preprocessor...")
                try:
                    self.preprocessor.fit(X_train)
                except Exception as e:
                    self.logger.error(f"Preprocessor fitting failed: {str(e)}")
                    if self.config.debug_mode:
                        self.logger.error(traceback.format_exc())
                    raise RuntimeError(f"Preprocessor error: {str(e)}")
                
            # Feature selection with enhanced reporting
            selected_features = self._perform_feature_selection(X_train, y_train)
            
            # Apply preprocessing to test data for consistency
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                X_test = self.preprocessor.transform(X_test)
            
            # Create the pipeline with preprocessor, feature selector, and model
            pipeline = self._create_pipeline(model)
            
            # Perform hyperparameter optimization with enhanced monitoring
            self.logger.info(f"Performing hyperparameter optimization with {self.config.optimization_strategy.value}...")
            
            search = self._get_optimization_search(pipeline, param_grid)
            
            # Use a try-except to catch optimization failures
            try:
                # If parallelization is enabled, set up progress tracking
                if self.config.n_jobs > 1 and self.config.verbose > 0:
                    with tqdm(total=100, desc="Hyperparameter optimization") as pbar:
                        # Define callback for progress
                        def update_progress(progress):
                            pbar.n = int(progress * 100)
                            pbar.refresh()
                            
                        # Check if optimizer supports callbacks
                        if hasattr(search, 'set_progress_callback'):
                            search.set_progress_callback(update_progress)
                            
                        search.fit(X_train, y_train)
                else:
                    search.fit(X_train, y_train)
            except Exception as e:
                self.logger.error(f"Hyperparameter optimization failed: {str(e)}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
                raise RuntimeError(f"Optimization error: {str(e)}")
            
            # Get the best model from the search
            best_model = search.best_estimator_
            best_params = search.best_params_
            
            # Extract clean version of best_params without pipeline prefixes
            clean_params = {}
            for param, value in best_params.items():
                # Remove pipeline prefixes like 'model__' or 'preprocessor__'
                if '__' in param:
                    clean_key = param.split('__', 1)[1]
                else:
                    clean_key = param
                clean_params[clean_key] = value
                
            self.logger.info(f"Best parameters: {clean_params}")
            
            # Enhanced model evaluation with timing
            self.logger.info("Evaluating model performance...")
            eval_start_time = time.time()
            metrics = self._evaluate_model(best_model, X_train, y_train, X_test, y_test)
            eval_time = time.time() - eval_start_time
            
            metrics['evaluation_time'] = eval_time
            self.logger.info(f"Performance metrics: {metrics}")
            
            # Get cross-validation results with enhanced reporting
            cv_results = self._extract_cv_results(search)
            
            # Track metrics
            if self.tracker:
                self.tracker.log_metrics(metrics)
                self.tracker.log_metrics(cv_results, step="cv")
                
                # Log feature importance with enhanced error handling
                self._log_feature_importance(best_model, X_train)
                
            # Store model with metadata
            model_info = {
                "model": best_model,
                "params": clean_params,
                "metrics": metrics,
                "cv_results": cv_results,
                "timestamp": time.time(),
                "training_time": time.time() - start_time,
                "feature_names": selected_features if selected_features else 
                                (X_train.columns.tolist() if hasattr(X_train, 'columns') else 
                                 [f"feature_{i}" for i in range(X_train.shape[1])]),
                "dataset_shape": {
                    "X_train": X_train.shape,
                    "X_test": X_test.shape if X_test is not None else None
                }
            }
            
            # Add custom metadata if provided
            if model_metadata:
                model_info.update(model_metadata)
            
            self.models[model_name] = model_info
            
            # Update best model tracking with more informative logging
            old_best = self.best_model if self.best_model else None
            self._update_best_model(model_name)
            
            if self.best_model and self.best_model == model_name and old_best != model_name:
                self.logger.info(f"New best model: {model_name} with score {self.best_score:.4f}")
                if old_best:
                    improvement = ((self.best_score - self._get_model_score(self.models[old_best]["metrics"])) / 
                                self._get_model_score(self.models[old_best]["metrics"]) * 100)
                    # For regression metrics that are minimized, we need to flip the sign
                    if self.config.task_type == TaskType.REGRESSION:
                        improvement = -improvement
                    self.logger.info(f"Improvement over previous best ({old_best}): {improvement:.2f}%")
    
                    
            # Auto-save model if configured
            if getattr(self.config, 'auto_save', False):
                self.save_model(model_name)
            
            # End experiment tracking
            if self.tracker:
                self.tracker.end_experiment()
                
            return best_model, metrics
            
        except Exception as e:
            self.logger.error(f"Training failed for {model_name}: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            
            # End experiment tracking with failure status
            if self.tracker:
                self.tracker.log_metrics({"error": str(e)}, step="failure")
                self.tracker.end_experiment()
                
            raise

    def _log_feature_importance(self, model, X_train):
        """Log feature importance with enhanced error handling"""
        if not self.tracker:
            return
            
        try:
            # Access the model inside the pipeline
            if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                final_model = model.named_steps['model']
            else:
                final_model = model
                
            feature_importance = self._get_feature_importance(final_model)
            
            if feature_importance is not None:
                # Get feature names
                if hasattr(X_train, 'columns'):  # If pandas DataFrame
                    feature_names = X_train.columns.tolist()
                else:
                    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                
                # If we use feature selector, get the selected features
                if self.feature_selector and hasattr(self.feature_selector, 'get_support'):
                    mask = self.feature_selector.get_support()
                    feature_names = [f for i, f in enumerate(feature_names) if mask[i]]
                
                # Ensure lengths match
                if len(feature_names) != len(feature_importance):
                    self.logger.warning(
                        f"Feature names length ({len(feature_names)}) doesn't match "
                        f"importance length ({len(feature_importance)}), using generic names"
                    )
                    feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
                    
                self.tracker.log_feature_importance(feature_names, feature_importance)
        except Exception as e:
            self.logger.warning(f"Failed to log feature importance: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())

    def _extract_cv_results(self, search):
        """Extract and process cross-validation results from the search object"""
        cv_results = {}
        
        # Check if the search object has cv_results_ attribute (most sklearn search methods do)
        if hasattr(search, 'cv_results_'):
            results = search.cv_results_
            
            # Extract fold scores for the best parameters
            best_index = search.best_index_ if hasattr(search, 'best_index_') else None
            
            if best_index is not None:
                # Get the number of splits from the results
                n_splits = 0
                for key in results.keys():
                    if key.startswith('split') and key.endswith('_test_score'):
                        n_splits = max(n_splits, int(key.split('_')[0].replace('split', '')) + 1)
                
                # Extract scores for each fold
                for i in range(n_splits):
                    fold_key = f'split{i}_test_score'
                    if fold_key in results:
                        cv_results[f"fold_{i+1}"] = float(results[fold_key][best_index])
                
                # Add mean and std
                if 'mean_test_score' in results and 'std_test_score' in results:
                    cv_results['mean'] = float(results['mean_test_score'][best_index])
                    cv_results['std'] = float(results['std_test_score'][best_index])
        
        # If cv_results is still empty, try to extract directly from custom optimizers
        if not cv_results and hasattr(search, 'best_scores_'):
            # Some custom optimizers might store scores differently
            for i, score in enumerate(search.best_scores_):
                cv_results[f"fold_{i+1}"] = float(score)
                
            if hasattr(search, 'best_score_'):
                cv_results['mean'] = float(search.best_score_)
                
            if len(search.best_scores_) > 1:
                cv_results['std'] = float(np.std(search.best_scores_))
        
        return cv_results

    def _perform_feature_selection(self, X, y):
        """Enhanced feature selection with better reporting"""
        if not self.config.feature_selection:
            return None
            
        self.logger.info("Performing feature selection...")
        self.feature_selector = self._get_feature_selector(X, y)
        
        if self.feature_selector:
            try:
                # Fit the selector
                self.feature_selector.fit(X, y)
                
                # Get feature mask and feature names
                if hasattr(self.feature_selector, 'get_support'):
                    mask = self.feature_selector.get_support()
                    
                    # Get feature names
                    if hasattr(X, 'columns'):  # If pandas DataFrame
                        all_features = X.columns.tolist()
                    else:
                        all_features = [f"feature_{i}" for i in range(X.shape[1])]
                    
                    selected_features = [f for i, f in enumerate(all_features) if mask[i]]
                    
                    # Calculate percentage of features retained
                    pct_retained = (len(selected_features) / len(all_features)) * 100
                    self.logger.info(f"Selected {len(selected_features)}/{len(all_features)} features ({pct_retained:.1f}%)")
                    
                    # Detailed feature selection logging
                    if hasattr(self.feature_selector, 'scores_'):
                        feature_scores = self.feature_selector.scores_
                        
                        # Create sorted feature importance
                        feature_importance = [(all_features[i], float(score)) for i, score in enumerate(feature_scores)]
                        feature_importance.sort(key=lambda x: x[1], reverse=True)
                        
                        # Log top and bottom features
                        top_k = min(10, len(feature_importance))
                        self.logger.info(f"Top {top_k} features by importance:")
                        for i in range(top_k):
                            self.logger.info(f"  {i+1}. {feature_importance[i][0]}: {feature_importance[i][1]:.4f}")
                            
                        if self.config.feature_selection_k is None:
                            # Filter features based on threshold
                            threshold = np.percentile(feature_scores, 
                                                     100 * self.config.feature_importance_threshold)
                            self.logger.info(f"Feature importance threshold: {threshold:.4f}")
                            selected_indices = np.where(feature_scores > threshold)[0]
                            
                            # Update selector's k
                            self.feature_selector.k = len(selected_indices)
                            self.logger.info(f"Selected {self.feature_selector.k} features based on threshold")
                            
                            # Update selected_features
                            selected_features = [all_features[i] for i in selected_indices]
                    
                    return selected_features
                
            except Exception as e:
                self.logger.error(f"Feature selection failed: {str(e)}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
                # Continue without feature selection
                self.feature_selector = None
                
        return None

    def _get_model_score(self, metrics):
        """Get a single score from metrics based on task type"""
        if self.config.task_type == TaskType.CLASSIFICATION:
            return metrics.get("f1", metrics.get("accuracy", 0.0))
        elif self.config.task_type == TaskType.REGRESSION:
            # For regression, we want to minimize error
            return metrics.get("mse", float('inf'))
        else:
            # Default to the first metric
            return next(iter(metrics.values())) if metrics else 0.0

    def save_model(self, model_name=None, filepath=None, access_code=None, compression_level=5,
                  include_preprocessor=True, include_metadata=True, version_tag=None):
        """
        Enhanced save model function with security integration and comprehensive metadata.
        
        Args:
            model_name: Name of the model to save (uses best model if None)
            filepath: Custom path to save the model (uses default path if None)
            access_code: Optional access code for secure models
            compression_level: Compression level (0-9) for model serialization
            include_preprocessor: Whether to include the preprocessor
            include_metadata: Whether to include model metadata
            version_tag: Optional version tag for the model
            
        Returns:
            Success status (bool)
        """
        # Get model data to save
        if model_name is None and self.best_model is not None:
            model_name = self.best_model
            model_data = self.best_model
        elif model_name in self.models:
            model_data = self.models[model_name]
        else:
            error_msg = f"Model {model_name} not found"
            self.logger.error(error_msg)
            return False, error_msg
            
        # Generate default filepath if not provided
        if filepath is None:
            # Create with timestamp for versioning
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            version = f"{version_tag}-" if version_tag else ""
            filepath = os.path.join(self.config.model_path, f"{model_name}-{version}{timestamp}.pkl")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Prepare the model package with comprehensive metadata
            model_package = {
                "model": model_data["model"],
                "params": model_data.get("params", {}),
                "metrics": model_data.get("metrics", {}),
                "timestamp": int(time.time()),
                "model_name": model_name,
                "version": version_tag or "1.0.0",
                "framework_version": self.VERSION
            }
            
            # Add configuration (sanitized if needed)
            if include_metadata:
                model_package["config"] = self._sanitize_config()
                
                # Add CV results if available
                if "cv_results" in model_data:
                    model_package["cv_results"] = model_data["cv_results"]
                    
                # Add feature importance if available
                if "feature_importance" in model_data:
                    model_package["feature_importance"] = model_data["feature_importance"]
                    
                # Add dataset info
                if "dataset_shape" in model_data:
                    model_package["dataset_shape"] = model_data["dataset_shape"]
                    
                # Add training time if available
                if "training_time" in model_data:
                    model_package["training_time"] = model_data["training_time"]
            
            # Add preprocessor if requested and available
            if include_preprocessor and self.preprocessor:
                model_package["preprocessor"] = self.preprocessor
                
            # Add feature selector if available
            if include_preprocessor and self.feature_selector:
                model_package["feature_selector"] = self.feature_selector
                
            # Use the secure model manager if available
            if hasattr(self, 'secure_manager'):
                return self.secure_manager.save_model(
                    model_package, filepath, access_code=access_code, 
                    compression_level=compression_level
                )
            
            # Otherwise use default serialization
            with open(filepath, 'wb') as f:
                joblib.dump(model_package, filepath, compress=compression_level)
                
            self.logger.info(f"Model saved to {filepath}")
            
            # Optionally quantize and save a separate quantized version
            if getattr(self.config, 'enable_quantization', False) and self.quantizer:
                try:
                    self._save_quantized_model(model_data, model_name, filepath)
                except Exception as e:
                    self.logger.warning(f"Failed to save quantized model: {str(e)}")
            
            return True, filepath
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            self.logger.error(error_msg)
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False, error_msg

    def _save_quantized_model(self, model_data, model_name, original_filepath):
        """Save a quantized version of the model for faster inference"""
        if self.quantizer is None:
            self.logger.warning("Quantizer not initialized but quantization was requested")
            return
            
        try:
            # Prepare quantized filepath
            quantized_filepath = original_filepath.replace('.pkl', '_quantized.pkl')
            if quantized_filepath == original_filepath:
                quantized_filepath = original_filepath + '.quantized'
                
            # Prepare model for quantization - exclude non-numeric parts
            model_bytes = pickle.dumps(model_data["model"])
            model_array = np.frombuffer(model_bytes, dtype=np.uint8)
            
            # Quantize the model bytes
            quantized_data = self.quantizer.quantize(model_array)
            
            # Calculate compression ratio
            compression_ratio = len(model_bytes) / len(quantized_data) if len(quantized_data) > 0 else 0
            
            # Prepare quantized package
            quantized_package = {
                "quantized_data": quantized_data,
                "metadata": {
                    "original_size": len(model_bytes),
                    "quantized_size": len(quantized_data),
                    "compression_ratio": compression_ratio,
                    "config": self.quantizer.get_config(),
                    "model_name": model_name,
                    "timestamp": int(time.time()),
                    "original_filepath": original_filepath
                }
            }
            
            # Save the quantized model
            with open(quantized_filepath, 'wb') as f:
                pickle.dump(quantized_package, f)
                
            self.logger.info(f"Quantized model saved to {quantized_filepath} "
                            f"(compression ratio: {compression_ratio:.2f}x)")
            return quantized_filepath
        except Exception as e:
            self.logger.error(f"Failed to save quantized model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return None

    def load_model(self, filepath, access_code=None, validate_metrics=True):
        """
        Enhanced load model function with security integration and validation.
        
        Args:
            filepath: Path to the model file
            access_code: Optional access code for secure models
            validate_metrics: Whether to validate and update metrics after loading
            
        Returns:
            Loaded model object or None if loading failed
        """
        try:
            # First check if the file exists
            if not os.path.exists(filepath):
                self.logger.error(f"Model file {filepath} does not exist")
                return None
                
            # Try to use secure model manager if available
            if hasattr(self, 'secure_manager'):
                model_package = self.secure_manager.load_model(filepath, access_code)
            else:
                # Otherwise use standard loading
                try:
                    model_package = joblib.load(filepath)
                except Exception as joblib_err:
                    self.logger.error(f"Failed to load with joblib: {str(joblib_err)}")
                    # Try pickle as fallback
                    try:
                        with open(filepath, 'rb') as f:
                            model_package = pickle.load(f)
                    except Exception as pickle_err:
                        self.logger.error(f"Failed to load with pickle: {str(pickle_err)}")
                        return None
                
            # Validate model package
            if not isinstance(model_package, dict):
                self.logger.error("Invalid model package format")
                return None
                
            if "model" not in model_package:
                self.logger.error("Model package missing required 'model' field")
                return None
                
            # Extract model data
            model = model_package["model"]
            model_name = model_package.get("model_name", os.path.basename(filepath).split('.')[0])
            
            # Load preprocessor if available
            if "preprocessor" in model_package and self.preprocessor is None:
                self.preprocessor = model_package["preprocessor"]
                self.logger.info("Loaded preprocessor from model package")
                
            # Load feature selector if available
            if "feature_selector" in model_package and self.feature_selector is None:
                self.feature_selector = model_package["feature_selector"]
                self.logger.info("Loaded feature selector from model package")
                
            # Extract metrics and parameters
            metrics = model_package.get("metrics", {})
            params = model_package.get("params", {})
            cv_results = model_package.get("cv_results", {})
            feature_importance = model_package.get("feature_importance", {})
            
            # Log model details
            self.logger.info(f"Loaded model: {model_name}")
            self.logger.info(f"Model type: {type(model).__name__}")
            if "version" in model_package:
                self.logger.info(f"Model version: {model_package['version']}")
                
            # Store model in our registry
            self.models[model_name] = {
                "name": model_name,
                "model": model,
                "params": params,
                "metrics": metrics,
                "cv_results": cv_results,
                "feature_importance": feature_importance,
                "loaded_from": filepath,
                "loaded_at": time.time(),
                "original_timestamp": model_package.get("timestamp")
            }
            
            # Update best model tracking
            self._update_best_model(model_name)
            
            # Validate metrics if requested and we have test data
            if validate_metrics and hasattr(self, '_last_X_test') and hasattr(self, '_last_y_test'):
                self.logger.info("Validating model metrics with cached test data")
                new_metrics = self._evaluate_model(model, None, None, self._last_X_test, self._last_y_test)
                
                # Log difference between stored and new metrics
                for metric, value in new_metrics.items():
                    if metric in metrics:
                        diff = value - metrics[metric]
                        diff_pct = (diff / metrics[metric]) * 100 if metrics[metric] != 0 else float('inf')
                        self.logger.info(f"Metric {metric}: stored={metrics[metric]:.4f}, "
                                        f"new={value:.4f}, diff={diff_pct:+.1f}%")
                
                # Update metrics
                self.models[model_name]["metrics"] = new_metrics
                self.models[model_name]["metrics_validated"] = True
                
                # Re-check if this is the best model with updated metrics
                self._update_best_model(model_name)
            
            # Return the model
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return None

    def _update_best_model(self, model_name):
        """Update best model tracking with enhanced metrics comparison"""
        if model_name not in self.models:
            return
            
        metrics = self.models[model_name].get("metrics", {})
        model_score = self._get_model_score(metrics)
        
        is_better = False
        if self.config.task_type == TaskType.REGRESSION:
            # For regression, lower score is better
            if model_score < self.best_score or self.best_model is None:
                is_better = True
        else:
            # For classification, higher score is better
            if model_score > self.best_score or self.best_model is None:
                is_better = True
                
        if is_better:
            old_best = None
            if self.best_model:
                old_best = self.best_model
                self.logger.info(f"New best model: {model_name} replacing {old_best}")
                
                # Calculate improvement percentage
                if old_best and old_best in self.models:
                    old_score = self._get_model_score(self.models[old_best]["metrics"])
                    improvement = abs((model_score - old_score) / old_score) * 100 if old_score != 0 else float('inf')
                    trend = "improved" if (
                        (self.config.task_type == TaskType.REGRESSION and model_score < old_score) or
                        (self.config.task_type != TaskType.REGRESSION and model_score > old_score)
                    ) else "decreased"
                    self.logger.info(f"Performance {trend} by {improvement:.2f}%")
            else:
                self.logger.info(f"First model loaded, setting {model_name} as best model")
                
            self.best_score = model_score
            self.best_model = model_name  

    def predict(self, X, model_name=None, return_proba=False, batch_size=None):
        """
        Enhanced prediction method with batching, fallback options and detailed error handling.
        
        Args:
            X: Input features
            model_name: Name of model to use (uses best model if None)
            return_proba: Whether to return probabilities instead of class labels (classification only)
            batch_size: Optional batch size for large datasets
            
        Returns:
            Model predictions
        """
        # Determine which model to use
        if model_name is None and self.best_model is not None:
            model_name = self.best_model
            model = self.models[model_name]["model"]
        elif model_name in self.models:
            model = self.models[model_name]["model"]
        else:
            self.logger.error(f"Model {model_name} not found")
            return None
            
        # Track start time for performance monitoring
        start_time = time.time()
        
        try:
            # Apply preprocessing if needed
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                try:
                    X = self.preprocessor.transform(X)
                except Exception as e:
                    self.logger.error(f"Preprocessing failed during prediction: {str(e)}")
                    if self.config.debug_mode:
                        self.logger.error(traceback.format_exc())
                    raise RuntimeError(f"Preprocessing error: {str(e)}")
            
            # Handle very large datasets with batching
            if batch_size is not None and hasattr(X, 'shape') and X.shape[0] > batch_size:
                self.logger.info(f"Using batched prediction with batch_size={batch_size}")
                return self._predict_in_batches(X, model, return_proba, batch_size)
            
            # First try using the inference engine for high-performance prediction
            if hasattr(self, 'inference_engine') and self.inference_engine is not None:
                try:
                    # Create a temporary model file for the inference engine
                    temp_model_path = os.path.join(self.config.model_path, f"temp_{model_name}_{int(time.time())}.pkl")
                    joblib.dump(model, temp_model_path)
                    
                    # Load model in inference engine
                    success_load = self.inference_engine.load_model(temp_model_path)
                    
                    if success_load:
                        # Make prediction
                        success, predictions, metadata = self.inference_engine.predict(X)
                        
                        # Clean up temp file
                        try:
                            os.remove(temp_model_path)
                        except:
                            pass
                        
                        if success:
                            # Log performance info
                            pred_time = time.time() - start_time
                            self.logger.info(f"Prediction completed via inference engine in {pred_time:.4f}s")
                            if metadata and 'inference_time_ms' in metadata:
                                self.logger.debug(f"Inference engine time: {metadata['inference_time_ms']:.2f}ms")
                            return predictions
                        else:
                            self.logger.warning("Inference engine prediction failed, falling back to direct prediction")
                    else:
                        self.logger.warning("Failed to load model in inference engine, falling back to direct prediction")
                except Exception as e:
                    self.logger.warning(f"Inference engine error: {str(e)}, falling back to direct prediction")
                    # Clean up temp file if it exists
                    try:
                        if os.path.exists(temp_model_path):
                            os.remove(temp_model_path)
                    except:
                        pass
            
            # Fallback: Direct prediction
            try:
                # For classification models, handle probability prediction
                if return_proba and self.config.task_type == TaskType.CLASSIFICATION:
                    if hasattr(model, 'predict_proba'):
                        results = model.predict_proba(X)
                    else:
                        self.logger.warning(f"Model {model_name} does not support probability prediction")
                        results = model.predict(X)
                else:
                    # Standard prediction
                    results = model.predict(X)
                
                # Log performance
                pred_time = time.time() - start_time
                self.logger.info(f"Prediction completed via direct model call in {pred_time:.4f}s")
                
                return results
            except Exception as e:
                self.logger.error(f"Prediction failed: {str(e)}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
                raise
                
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return None

    def _predict_in_batches(self, X, model, return_proba, batch_size):
        """Helper method to perform batch prediction for large datasets"""
        self.logger.info(f"Processing prediction in batches of size {batch_size}")
        
        # Total number of samples
        n_samples = X.shape[0]
        
        # Create a list to store batch results
        all_results = []
        
        # Process data in batches
        with tqdm(total=n_samples, desc="Batch prediction") as pbar:
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch = X[start_idx:end_idx]
                
                # For classification models, handle probability prediction
                if return_proba and self.config.task_type == TaskType.CLASSIFICATION:
                    if hasattr(model, 'predict_proba'):
                        batch_result = model.predict_proba(batch)
                    else:
                        batch_result = model.predict(batch)
                else:
                    # Standard prediction
                    batch_result = model.predict(batch)
                    
                all_results.append(batch_result)
                
                # Update progress bar
                pbar.update(end_idx - start_idx)
                
        # Combine results
        if isinstance(all_results[0], np.ndarray):
            try:
                # Try to concatenate arrays
                return np.concatenate(all_results)
            except Exception as e:
                self.logger.warning(f"Could not concatenate results: {str(e)}")
                return all_results
        else:
            # Handle other result types
            return all_results

    def run_batch_inference(self, data_generator, batch_size=None, model_name=None, 
                           return_proba=False, parallel=True, timeout=None):
        """
        Enhanced batch inference with parallel processing, timeouts, and progress tracking.
        
        Args:
            data_generator: Generator or iterable providing batches of data
            batch_size: Size of batches (overrides generator's batch size)
            model_name: Name of model to use (uses best model if None)
            return_proba: Whether to return probabilities (classification only)
            parallel: Whether to process batches in parallel
            timeout: Maximum time in seconds to wait for each batch
            
        Returns:
            List of batch predictions
        """
        # Determine which model to use
        if model_name is None and self.best_model is not None:
            model = self.best_model["model"]
            model_name = self.best_model["name"]
        elif model_name in self.models:
            model = self.models[model_name]["model"]
        else:
            self.logger.error(f"Model {model_name} not found")
            return None
        
        # Define batch processing function based on model type and return_proba
        def process_batch(batch):
            # Apply preprocessing if needed
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                batch = self.preprocessor.transform(batch)
                
            # For classification models, handle probability prediction
            if return_proba and self.config.task_type == TaskType.CLASSIFICATION:
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(batch)
                else:
                    self.logger.warning(f"Model {model_name} does not support probability prediction")
                    return model.predict(batch)
            else:
                # Standard prediction
                return model.predict(batch)
            
        # Configure batch processor if available and enabled
        batch_processor = None
        if parallel and hasattr(self, 'batch_processor') and self.batch_processor is not None:
            batch_processor = self.batch_processor
            
            # Configure batch size if provided
            if batch_size is not None and hasattr(batch_processor, 'update_batch_size'):
                batch_processor.update_batch_size(batch_size)
                
            # Start the batch processor if not already running
            if hasattr(batch_processor, 'start') and not hasattr(batch_processor, '_worker_thread'):
                batch_processor.start(process_batch)
                
        # Process data in batches
        results = []
        batches = []  # Store original batches for reference
        batch_times = []
        batch_sizes = []
        failed_batches = []
        
        try:
            # Convert generator to list if we need to know total for progress bar
            data_batches = list(data_generator)
            total_batches = len(data_batches)
            
            with tqdm(total=total_batches, desc="Batch inference") as pbar:
                for i, data_chunk in enumerate(data_batches):
                    batch_start_time = time.time()
                    batches.append(data_chunk)
                    
                    try:
                        # Resize batch if needed
                        if batch_size is not None and hasattr(data_chunk, 'shape') and data_chunk.shape[0] > batch_size:
                            # Process large batch in sub-batches
                            chunk_results = []
                            for j in range(0, data_chunk.shape[0], batch_size):
                                sub_chunk = data_chunk[j:j+batch_size]
                                if batch_processor:
                                    # Use batch processor for parallel processing
                                    future = batch_processor.enqueue_predict(sub_chunk)
                                    chunk_results.append(future.result(timeout=timeout))
                                else:
                                    # Process sequentially
                                    chunk_results.append(process_batch(sub_chunk))
                            
                            # Combine sub-batch results
                            if isinstance(chunk_results[0], np.ndarray):
                                batch_result = np.vstack(chunk_results)
                            else:
                                batch_result = chunk_results
                        else:
                            # Process normal-sized batch
                            if batch_processor:
                                # Use batch processor for parallel processing
                                future = batch_processor.enqueue_predict(data_chunk)
                                batch_result = future.result(timeout=timeout)
                            else:
                                # Process sequentially
                                batch_result = process_batch(data_chunk)
                        
                        results.append(batch_result)
                        batch_time = time.time() - batch_start_time
                        batch_times.append(batch_time)
                        batch_sizes.append(data_chunk.shape[0] if hasattr(data_chunk, 'shape') else 1)
                        
                        # Update progress bar with timing info
                        pbar.set_description(f"Batch {i+1}/{total_batches} ({batch_time:.2f}s)")
                        pbar.update(1)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing batch {i+1}: {str(e)}")
                        failed_batches.append(i)
                        # Return None for failed batch
                        results.append(None)
                        pbar.update(1)
                
            # Calculate and log overall statistics
            if batch_times:
                total_time = sum(batch_times)
                avg_time = np.mean(batch_times)
                total_samples = sum(batch_sizes)
                
                self.logger.info(f"Batch inference completed: {total_batches} batches, {total_samples} samples")
                self.logger.info(f"Total time: {total_time:.2f}s, Avg batch time: {avg_time:.2f}s")
                self.logger.info(f"Throughput: {total_samples / total_time:.1f} samples/second")
                
                if failed_batches:
                    self.logger.warning(f"Failed batches: {len(failed_batches)}/{total_batches}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during batch inference: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return results  # Return partial results

    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models on test data"""
        results = {}
        
        for model_name, model_data in self.models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            try:
                model = model_data["model"]
                metrics = self._evaluate_model(model, None, None, X_test, y_test)
                
                results[model_name] = metrics
                self.logger.info(f"Model {model_name} metrics: {metrics}")
                
                # Update stored metrics
                model_data["metrics"] = metrics
                
                # Check if this is better than current best model
                model_score = self._get_model_score(metrics)
                
                is_better = False
                if self.config.task_type == TaskType.REGRESSION:
                    if model_score < self.best_score:
                        is_better = True
                else:
                    if model_score > self.best_score:
                        is_better = True
                        
                if is_better:
                    self.best_score = model_score
                    self.best_model = {
                        "name": model_name,
                        "model": model,
                        "metrics": metrics
                    }
                    self.logger.info(f"New best model: {model_name} with score {self.best_score:.4f}")
            except Exception as e:
                self.logger.error(f"Error evaluating model {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
        
        return results

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
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
                report += f"<td>{value}</td>"
                
            report += "</tr>"
            
        # Remainder of report generation...
        # ...
        
        # Write report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"Report generated: {output_file}")
        return output_file


    def shutdown(self):
        """Enhanced shutdown with resource cleanup and state persistence"""
        self.logger.info("Shutting down ML Training Engine...")
        
        # Auto-save best model if configured
        if getattr(self.config, 'auto_save_on_shutdown', True) and self.best_model:
            self.logger.info(f"Auto-saving best model: {self.best_model['name']}")
            self.save_model(self.best_model["name"])
        
        # Shutdown inference engine
        if hasattr(self, 'inference_engine'):
            try:
                self.inference_engine.shutdown()
                self.logger.info("Inference engine shut down")
            except Exception as e:
                self.logger.error(f"Error shutting down inference engine: {str(e)}")
            
        # Stop batch processor
        if hasattr(self, 'batch_processor'):
            try:
                self.batch_processor.stop()
                self.logger.info("Batch processor stopped")
            except Exception as e:
                self.logger.error(f"Error stopping batch processor: {str(e)}")
            
        # Save state if configured
        if getattr(self.config, 'save_state_on_shutdown', False):
            try:
                state_path = os.path.join(self.config.model_path, "engine_state.pkl")
                # Save minimal state (exclude large objects)
                state = {
                    "models": {name: {
                        k: v for k, v in model_data.items() 
                        if k not in ['model', 'preprocessor', 'feature_selector']
                    } for name, model_data in self.models.items()},
                    "best_model_name": self.best_model["name"] if self.best_model else None,
                    "best_score": self.best_score,
                    "timestamp": time.time()
                }
                with open(state_path, 'wb') as f:
                    pickle.dump(state, f)
                self.logger.info(f"Engine state saved to {state_path}")
            except Exception as e:
                self.logger.error(f"Failed to save engine state: {str(e)}")
        
        # Clean up resources
        if getattr(self.config, 'memory_optimization', True):
            try:
                # Clear models and other large objects
                for model_name in list(self.models.keys()):
                    if 'model' in self.models[model_name]:
                        self.models[model_name]['model'] = None
                
                self.best_model = None
                self.preprocessor = None
                self.feature_selector = None
                
                self.logger.info("Memory resources cleared")
                gc.collect()
            except Exception as e:
                self.logger.error(f"Error during resource cleanup: {str(e)}")
            
        self.logger.info("ML Training Engine shut down successfully")

    def generate_reports(self, output_file=None, include_plots=True):
        """Generate a comprehensive report of all models in Markdown format
        
        Args:
            output_file (str, optional): Path to save the report. Defaults to None.
            include_plots (bool, optional): Whether to include plots in the report. Defaults to True.
            
        Returns:
            str: Path to the generated report file
        """
        if not self.models:
            self.logger.warning("No models to generate report")
            return None
                
        if output_file is None:
            output_file = os.path.join(self.config.model_path, "model_report.md")
                
        # Create basic report
        report = f"# ML Training Engine Report\n\n"
        report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add configuration section
        report += "## Configuration\n\n"
        report += "| Parameter | Value |\n"
        report += "| --- | --- |\n"
        
        # Add configuration details
        for key, value in self.config.to_dict().items():
            report += f"| {key} | {value} |\n"
        
        report += "\n## Model Performance Summary\n\n"
        
        # Collect all metrics across models
        all_metrics = set()
        for model_data in self.models.values():
            all_metrics.update(model_data["metrics"].keys())
        
        # Create table header
        report += "| Model | " + " | ".join(sorted(all_metrics)) + " |\n"
        report += "| --- | " + " | ".join(["---" for _ in all_metrics]) + " |\n"
        
        # Add model rows
        for model_name, model_data in self.models.items():
            is_best = self.best_model and self.best_model["name"] == model_name
            model_label = f"{model_name} **[BEST]**" if is_best else model_name
            
            row = f"| {model_label} |"
            for metric in sorted(all_metrics):
                value = model_data["metrics"].get(metric, "N/A")
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
                row += f" {value} |"
            
            report += row + "\n"
        
        # Add hyperparameter section
        report += "\n## Model Hyperparameters\n\n"
        
        for model_name, model_data in self.models.items():
            report += f"### {model_name}\n\n"
            
            if "params" in model_data and model_data["params"]:
                report += "| Parameter | Value |\n"
                report += "| --- | --- |\n"
                
                # Extract parameters, handling nested dictionaries
                flat_params = {}
                for k, v in model_data["params"].items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            flat_params[f"{k}__{sub_k}"] = sub_v
                    else:
                        flat_params[k] = v
                
                for param, value in flat_params.items():
                    report += f"| {param} | {value} |\n"
            else:
                report += "No hyperparameter information available.\n"
            
            report += "\n"
        
        # Add feature importance section if available
        if include_plots and any("feature_importance" in model_data for model_data in self.models.values()):
            report += "\n## Feature Importance\n\n"
            
            for model_name, model_data in self.models.items():
                if "feature_importance" in model_data and model_data["feature_importance"]:
                    report += f"### {model_name}\n\n"
                    
                    # Get top 15 features
                    feature_importance = model_data["feature_importance"]
                    top_features = dict(sorted(feature_importance.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:15])
                    
                    report += "| Feature | Importance |\n"
                    report += "| --- | --- |\n"
                    
                    for feature, importance in top_features.items():
                        report += f"| {feature} | {importance:.4f} |\n"
                    
                    # Generate and include plot if requested
                    if include_plots:
                        plots_dir = os.path.join(self.config.model_path, "plots")
                        if not os.path.exists(plots_dir):
                            os.makedirs(plots_dir)
                        
                        plot_path = os.path.join(plots_dir, f"{model_name}_feature_importance.png")
                        
                        plt.figure(figsize=(10, 6))
                        features = list(top_features.keys())
                        importances = list(top_features.values())
                        
                        plt.barh(range(len(features)), importances, align='center')
                        plt.yticks(range(len(features)), features)
                        plt.xlabel('Importance')
                        plt.title(f'Feature Importance - {model_name}')
                        plt.tight_layout()
                        plt.savefig(plot_path)
                        plt.close()
                        
                        # Add plot to report
                        report += f"\n![Feature Importance for {model_name}]({os.path.relpath(plot_path, os.path.dirname(output_file))})\n\n"
                    
                    report += "\n"
        
        # Add cross-validation results if available
        if any("cv_results" in model_data for model_data in self.models.values()):
            report += "\n## Cross-Validation Results\n\n"
            
            for model_name, model_data in self.models.items():
                if "cv_results" in model_data and model_data["cv_results"]:
                    report += f"### {model_name}\n\n"
                    
                    cv_results = model_data["cv_results"]
                    report += "| Fold | Score |\n"
                    report += "| --- | --- |\n"
                    
                    for fold, score in cv_results.items():
                        report += f"| {fold} | {score:.4f} |\n"
                    
                    # Calculate summary statistics
                    mean_score = np.mean(list(cv_results.values()))
                    std_score = np.std(list(cv_results.values()))
                    
                    report += f"\n**Mean CV Score:** {mean_score:.4f} ± {std_score:.4f}\n\n"
                    
                    # Generate and include plot if requested
                    if include_plots:
                        plots_dir = os.path.join(self.config.model_path, "plots")
                        if not os.path.exists(plots_dir):
                            os.makedirs(plots_dir)
                        
                        plot_path = os.path.join(plots_dir, f"{model_name}_cv_scores.png")
                        
                        plt.figure(figsize=(10, 6))
                        plt.bar(range(len(cv_results)), list(cv_results.values()))
                        plt.xticks(range(len(cv_results)), list(cv_results.keys()), rotation=45)
                        plt.axhline(y=mean_score, color='r', linestyle='-', label=f'Mean: {mean_score:.4f}')
                        plt.xlabel('Fold')
                        plt.ylabel('Score')
                        plt.title(f'Cross-Validation Scores - {model_name}')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(plot_path)
                        plt.close()
                        
                        # Add plot to report
                        report += f"\n![CV Scores for {model_name}]({os.path.relpath(plot_path, os.path.dirname(output_file))})\n\n"
        
        # Add conclusion section
        report += "\n## Conclusion\n\n"
        
        if self.best_model:
            report += f"The best performing model is **{self.best_model['name']}** "
            
            if self.config.task_type == TaskType.CLASSIFICATION:
                key_metric = "f1" if "f1" in self.best_model["metrics"] else "accuracy"
                report += f"with {key_metric} of {self.best_model['metrics'].get(key_metric, 'N/A'):.4f}.\n\n"
            elif self.config.task_type == TaskType.REGRESSION:
                key_metric = "rmse" if "rmse" in self.best_model["metrics"] else "mse"
                report += f"with {key_metric} of {self.best_model['metrics'].get(key_metric, 'N/A'):.4f}.\n\n"
            else:
                report += ".\n\n"
        else:
            report += "No models were evaluated.\n\n"
        
        # Write report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"Markdown report generated: {output_file}")
        return output_file

    def generate_feature_importance_report(self, model_name=None, top_n=20, 
                                         include_plot=True, output_file=None):
        """
        Generate a detailed feature importance report with analysis and visualization.
        
        Args:
            model_name: Model to analyze (uses best model if None)
            top_n: Number of top features to include
            include_plot: Whether to include visualization
            output_file: Path to save the report (optional)
            
        Returns:
            Dictionary with feature importance information
        """
        # Determine which model to use
        if model_name is None and self.best_model is not None:
            model_data = self.best_model
            model_name = self.best_model["name"]
        elif model_name in self.models:
            model_data = self.models[model_name]
        else:
            self.logger.error(f"Model {model_name} not found")
            return {"error": f"Model {model_name} not found"}
            
        model = model_data.get("model")
        if model is None:
            self.logger.error(f"Model object not found for {model_name}")
            return {"error": f"Model object not found for {model_name}"}
            
        # Get feature names
        feature_names = None
        if "feature_names" in model_data:
            feature_names = model_data["feature_names"]
        elif hasattr(self, '_last_X_train') and hasattr(self._last_X_train, 'columns'):
            feature_names = self._last_X_train.columns.tolist()
        else:
            # Use generic feature names
            n_features = 0
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
            elif 'feature_importances_' in dir(model) and hasattr(model.feature_importances_, 'shape'):
                n_features = model.feature_importances_.shape[0]
            elif 'coef_' in dir(model) and hasattr(model.coef_, 'shape'):
                if model.coef_.ndim > 1:
                    n_features = model.coef_.shape[1]
                else:
                    n_features = model.coef_.shape[0]
            
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Get feature importance
        if "feature_importance" in model_data and model_data["feature_importance"]:
            # Use precomputed feature importance
            feature_importance = model_data["feature_importance"]
            if isinstance(feature_importance, dict):
                # Convert dictionary to lists for easier sorting
                features = list(feature_importance.keys())
                importance_values = list(feature_importance.values())
            else:
                # Assume it's a numpy array
                importance_values = feature_importance
                # Ensure feature names match importance length
                if feature_names and len(feature_names) != len(importance_values):
                    feature_names = [f"feature_{i}" for i in range(len(importance_values))]
                features = feature_names
        else:
            # Calculate feature importance
            importance_values = self._get_feature_importance(model)
            
            if importance_values is None:
                # Try permutation importance as fallback
                importance_values = self._calculate_permutation_importance(model, model_name)
                
                if importance_values is None:
                    self.logger.error(f"Could not calculate feature importance for {model_name}")
                    return {"error": "Could not calculate feature importance"}
            
            features = feature_names[:len(importance_values)] if feature_names else [f"feature_{i}" for i in range(len(importance_values))]
        
        # Create sorted importance pairs
        importance_pairs = list(zip(features, importance_values))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N features
        top_features = importance_pairs[:min(top_n, len(importance_pairs))]
        
        # Prepare output
        result = {
            "model_name": model_name,
            "feature_importance": {name: float(importance) for name, importance in importance_pairs},
            "top_features": {name: float(importance) for name, importance in top_features},
            "timestamp": time.time()
        }
        
        # Generate visualization if requested
        if include_plot:
            try:
                plots_dir = os.path.join(self.config.model_path, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                plot_path = os.path.join(plots_dir, f"{model_name}_feature_importance.png")
                
                plt.figure(figsize=(12, 8))
                features_names = [pair[0] for pair in top_features]
                importance_vals = [pair[1] for pair in top_features]
                
                # Create horizontal bar chart for better readability with long feature names
                plt.barh(range(len(features_names)), importance_vals, align='center')
                plt.yticks(range(len(features_names)), features_names)
                plt.xlabel('Importance')
                plt.title(f'Top {len(top_features)} Features by Importance - {model_name}')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                result["plot_path"] = plot_path
                self.logger.info(f"Feature importance plot saved to {plot_path}")
            except Exception as e:
                self.logger.error(f"Failed to create feature importance plot: {str(e)}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
        
        # Save report to file if requested
        if output_file:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Create report in markdown format
                report = f"# Feature Importance Report - {model_name}\n\n"
                report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                report += f"## Top {len(top_features)} Features\n\n"
                report += "| Rank | Feature | Importance |\n"
                report += "| --- | --- | --- |\n"
                
                for i, (feature, importance) in enumerate(top_features):
                    report += f"| {i+1} | {feature} | {importance:.6f} |\n"
                
                # Add visualization if created
                if "plot_path" in result:
                    report += f"\n\n![Feature Importance Plot]({os.path.relpath(result['plot_path'], os.path.dirname(output_file))})\n\n"
                
                # Add model metadata
                report += "\n## Model Information\n\n"
                report += f"- Model Type: {type(model).__name__}\n"
                if "metrics" in model_data:
                    report += "- Performance Metrics:\n"
                    for metric, value in model_data["metrics"].items():
                        if isinstance(value, (int, float)):
                            report += f"  - {metric}: {value:.4f}\n"
                
                # Add description of importance calculation method
                report += "\n## Importance Calculation Method\n\n"
                if "feature_importance" in model_data:
                    report += "Feature importance was pre-computed and stored with the model.\n"
                elif hasattr(model, 'feature_importances_'):
                    report += "Feature importance was calculated using the model's built-in feature_importances_ attribute.\n"
                elif hasattr(model, 'coef_'):
                    report += "Feature importance was derived from the model's coefficients.\n"
                else:
                    report += "Feature importance was calculated using permutation importance.\n"
                
                # Write report to file
                with open(output_file, 'w') as f:
                    f.write(report)
                
                result["report_path"] = output_file
                self.logger.info(f"Feature importance report saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save report: {str(e)}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
        
        return result

    def _calculate_permutation_importance(self, model, model_name):
        """Calculate permutation importance when built-in methods not available"""
        self.logger.info(f"Calculating permutation importance for {model_name}")
        
        # Need test data for permutation importance
        if not hasattr(self, '_last_X_test') or not hasattr(self, '_last_y_test'):
            self.logger.error("No test data available for permutation importance")
            return None
            
        try:
            from sklearn.inspection import permutation_importance
            
            # Apply preprocessing if needed
            X_test = self._last_X_test
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                X_test = self.preprocessor.transform(X_test)
            
            # Calculate permutation importance with progress reporting
            self.logger.info("Running permutation importance (this may take a while)...")
            
            # For large datasets, use a subset to improve performance
            if hasattr(X_test, 'shape') and X_test.shape[0] > 1000:
                self.logger.info(f"Using subset of data ({min(1000, X_test.shape[0])} samples) for permutation importance")
                indices = np.random.choice(X_test.shape[0], min(1000, X_test.shape[0]), replace=False)
                X_subset = X_test[indices]
                y_subset = self._last_y_test[indices]
            else:
                X_subset = X_test
                y_subset = self._last_y_test
            
            # Calculate permutation importance
            result = permutation_importance(
                model, X_subset, y_subset,
                n_repeats=10,
                random_state=self.config.random_state
            )
            
            # Store results in model data
            if model_name in self.models:
                self.models[model_name]["feature_importance"] = result.importances_mean
                
                # Also store in last_feature_importance for other functions to access
                self._last_feature_importance = result.importances_mean
                
            return result.importances_mean
            
        except Exception as e:
            self.logger.error(f"Failed to calculate permutation importance: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return None

    def export_model(self, model_name=None, format='sklearn', output_dir=None, include_pipeline=True):
        """
        Export a trained model to various formats for deployment.
        
        Args:
            model_name: Name of the model to export (uses best model if None)
            format: Export format ('sklearn', 'onnx', 'pmml', 'tf', or 'torchscript')
            output_dir: Directory to save exported model (uses model_path if None)
            include_pipeline: Whether to include preprocessor in the export
            
        Returns:
            Path to the exported model
        """
        # Determine which model to use
        if model_name is None and self.best_model is not None:
            model_name = self.best_model
            model_data = self.models[model_name]
        elif model_name in self.models:
            model_data = self.models[model_name]
        else:
            self.logger.error(f"Model {model_name} not found")
            return None
            
        model = model_data.get("model")
        if model is None:
            self.logger.error(f"Model object not found for {model_name}")
            return None
            
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(self.config.model_path, "exports")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for versioning
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        try:
            # Export based on format
            if format.lower() == 'sklearn':
                # Standard scikit-learn serialization with joblib
                output_path = os.path.join(output_dir, f"{model_name}_{timestamp}.pkl")
                
                if include_pipeline and self.preprocessor:
                    # Create a full pipeline with preprocessing
                    from sklearn.pipeline import Pipeline
                    pipeline_steps = []
                    
                    if self.preprocessor:
                        pipeline_steps.append(('preprocessor', self.preprocessor))
                    
                    if self.feature_selector:
                        pipeline_steps.append(('feature_selector', self.feature_selector))
                        
                    pipeline_steps.append(('model', model))
                    export_model = Pipeline(pipeline_steps)
                else:
                    export_model = model
                    
                joblib.dump(export_model, output_path)
                self.logger.info(f"Model exported to {output_path}")
                
                # Create a metadata file with additional information
                metadata = {
                    "model_name": model_name,
                    "export_time": timestamp,
                    "format": "sklearn",
                    "includes_pipeline": include_pipeline,
                    "model_type": type(model).__name__,
                    "framework_version": self.VERSION
                }
                
                if "metrics" in model_data:
                    # Include only numeric metrics
                    metadata["metrics"] = {k: v for k, v in model_data["metrics"].items() 
                                         if isinstance(v, (int, float))}
                
                metadata_path = output_path + ".json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return output_path
                
            elif format.lower() == 'onnx':
                # Export to ONNX format for cross-platform deployment
                try:
                    import skl2onnx
                    
                    output_path = os.path.join(output_dir, f"{model_name}_{timestamp}.onnx")
                    
                    # Get sample input data for shape inference
                    if hasattr(self, '_last_X_train'):
                        X_sample = self._last_X_train[:1]
                    else:
                        # Create dummy data based on model's expected input
                        if hasattr(model, 'n_features_in_'):
                            n_features = model.n_features_in_
                        else:
                            n_features = 10  # Default assumption
                        X_sample = np.zeros((1, n_features))
                    
                    # Apply preprocessing if needed and included
                    if include_pipeline and self.preprocessor:
                        X_sample = self.preprocessor.transform(X_sample)
                        
                        # Get feature names after preprocessing
                        if hasattr(X_sample, 'columns'):
                            feature_names = X_sample.columns.tolist()
                        else:
                            feature_names = [f'f{i}' for i in range(X_sample.shape[1])]
                        
                        # Create ONNX pipeline
                        from skl2onnx.common.data_types import FloatTensorType
                        initial_types = [(name, FloatTensorType([None, 1])) for name in feature_names]
                        
                        # Convert pipeline to ONNX
                        from sklearn.pipeline import Pipeline
                        pipeline_steps = [('preprocessor', self.preprocessor), ('model', model)]
                        pipeline = Pipeline(pipeline_steps)
                        
                        onnx_model = skl2onnx.convert_sklearn(
                            pipeline, 
                            initial_types=initial_types,
                            options={type(model): {'zipmap': False}}
                        )
                    else:
                        # Convert just the model
                        from skl2onnx.common.data_types import FloatTensorType
                        initial_types = [('X', FloatTensorType([None, X_sample.shape[1]]))]
                        
                        onnx_model = skl2onnx.convert_sklearn(
                            model, 
                            initial_types=initial_types,
                            options={type(model): {'zipmap': False}}
                        )
                    
                    # Save the ONNX model
                    with open(output_path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                        
                    self.logger.info(f"Model exported to ONNX format: {output_path}")
                    return output_path
                except ImportError:
                    self.logger.error("skl2onnx package is required for ONNX export")
                    return None
                
            elif format.lower() == 'pmml':
                # Export to PMML format
                try:
                    from sklearn2pmml import sklearn2pmml
                    from sklearn2pmml.pipeline import PMMLPipeline
                    
                    output_path = os.path.join(output_dir, f"{model_name}_{timestamp}.pmml")
                    
                    # Create PMML pipeline
                    if include_pipeline and self.preprocessor:
                        pmml_pipeline = PMMLPipeline([
                            ("preprocessor", self.preprocessor),
                            ("model", model)
                        ])
                    else:
                        pmml_pipeline = PMMLPipeline([
                            ("model", model)
                        ])
                    
                    # Export to PMML
                    sklearn2pmml(pmml_pipeline, output_path)
                    
                    self.logger.info(f"Model exported to PMML format: {output_path}")
                    return output_path
                except ImportError:
                    self.logger.error("sklearn2pmml package is required for PMML export")
                    return None
                    
            elif format.lower() in ['tf', 'tensorflow']:
                # Export to TensorFlow SavedModel format
                try:
                    import tensorflow as tf
                    
                    output_path = os.path.join(output_dir, f"{model_name}_{timestamp}_tf")
                    
                    # Create a TensorFlow model wrapper around sklearn model
                    class SklearnModel(tf.Module):
                        def __init__(self, model, preprocessor=None):
                            self.model = model
                            self.preprocessor = preprocessor
                            
                        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
                        def __call__(self, x):
                            # Apply preprocessing if available
                            if self.preprocessor is not None:
                                x_np = x.numpy()
                                x_np = self.preprocessor.transform(x_np)
                                x = tf.convert_to_tensor(x_np, dtype=tf.float32)
                                
                            # Apply model prediction
                            result = self.model.predict(x.numpy())
                            return tf.convert_to_tensor(result, dtype=tf.float32)
                    
                    # Create the wrapper
                    if include_pipeline and self.preprocessor:
                        tf_model = SklearnModel(model, self.preprocessor)
                    else:
                        tf_model = SklearnModel(model)
                        
                    # Save the model
                    tf.saved_model.save(tf_model, output_path)
                    
                    self.logger.info(f"Model exported to TensorFlow format: {output_path}")
                    return output_path
                except ImportError:
                    self.logger.error("tensorflow package is required for TF export")
                    return None
                    
            elif format.lower() in ['torch', 'torchscript']:
                # Export to TorchScript format
                try:
                    import torch
                    import numpy as np
                    
                    output_path = os.path.join(output_dir, f"{model_name}_{timestamp}.pt")
                    
                    # Create a PyTorch wrapper around sklearn model
                    class SklearnModelTorch(torch.nn.Module):
                        def __init__(self, model, preprocessor=None):
                            super().__init__()
                            self.model = model
                            self.preprocessor = preprocessor
                            
                        def forward(self, x):
                            # Convert to numpy for sklearn
                            x_np = x.detach().cpu().numpy()
                            
                            # Apply preprocessing if available
                            if self.preprocessor is not None:
                                x_np = self.preprocessor.transform(x_np)
                                
                            # Apply model prediction
                            result = self.model.predict(x_np)
                            
                            # Convert back to torch tensor
                            return torch.tensor(result, dtype=torch.float32)
                    
                    # Create the wrapper
                    if include_pipeline and self.preprocessor:
                        torch_model = SklearnModelTorch(model, self.preprocessor)
                    else:
                        torch_model = SklearnModelTorch(model)
                        
                    # Get example input for tracing
                    if hasattr(self, '_last_X_train'):
                        X_sample = self._last_X_train[:1]
                    else:
                        if hasattr(model, 'n_features_in_'):
                            n_features = model.n_features_in_
                        else:
                            n_features = 10
                        X_sample = np.zeros((1, n_features))
                        
                    # Convert to torch tensor
                    example_input = torch.tensor(X_sample, dtype=torch.float32)
                    
                    # Trace the model and save
                    traced_model = torch.jit.trace(torch_model, example_input)
                    torch.jit.save(traced_model, output_path)
                    
                    self.logger.info(f"Model exported to TorchScript format: {output_path}")
                    return output_path
                except ImportError:
                    self.logger.error("torch package is required for TorchScript export")
                    return None
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to export model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return None

    def compare_models(self, model_names=None, metrics=None, include_plot=True, output_file=None):
        """
        Generate a comparative analysis of multiple models.
        
        Args:
            model_names: List of models to compare (uses all models if None)
            metrics: List of metrics to compare (uses common metrics if None)
            include_plot: Whether to include visualization
            output_file: Path to save the comparison report
            
        Returns:
            Dictionary with comparison results
        """
        # Determine which models to compare
        if model_names is None:
            model_names = list(self.models.keys())
        
        # Filter to existing models
        model_names = [name for name in model_names if name in self.models]
        
        if not model_names:
            self.logger.error("No valid models to compare")
            return {"error": "No valid models to compare"}
            
        self.logger.info(f"Comparing {len(model_names)} models: {', '.join(model_names)}")
        
        # Collect all metrics from all models
        all_metrics = set()
        for name in model_names:
            if "metrics" in self.models[name]:
                all_metrics.update(self.models[name]["metrics"].keys())
        
        # Filter to numeric metrics and sort
        all_metrics = sorted([m for m in all_metrics 
                            if any(isinstance(self.models[name].get("metrics", {}).get(m), (int, float)) 
                                 for name in model_names)])
        
        # Use specified metrics if provided
        if metrics:
            # Ensure all specified metrics exist
            metrics = [m for m in metrics if m in all_metrics]
            if not metrics:
                self.logger.warning(f"None of the specified metrics {metrics} found in models")
                metrics = all_metrics
        else:
            # Use common metrics appropriate for the task type
            if self.config.task_type == TaskType.CLASSIFICATION:
                preferred = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
            elif self.config.task_type == TaskType.REGRESSION:
                preferred = ['rmse', 'mse', 'mae', 'r2']
            else:
                preferred = []
                
            # Use preferred metrics if available, otherwise use all
            metrics = [m for m in preferred if m in all_metrics]
            if not metrics:
                metrics = all_metrics[:5]  # Limit to 5 metrics if many available
        
        # Prepare comparison data
        comparison = {
            "models": model_names,
            "metrics": metrics,
            "data": {},
            "best_model": None,
            "timestamp": time.time()
        }
        
        # Collect model data
        for name in model_names:
            model_data = self.models[name]
            model_metrics = model_data.get("metrics", {})
            
            comparison["data"][name] = {
                "model_type": type(model_data.get("model", "")).__name__,
                "metrics": {m: model_metrics.get(m) for m in metrics if m in model_metrics},
                "params": model_data.get("params", {}),
                "is_best": (self.best_model is not None and self.best_model == name)
            }
            
            # If this is the best model, mark it
            if comparison["data"][name]["is_best"]:
                comparison["best_model"] = name
        
        # Calculate metric ranges for comparison context
        metric_ranges = {}
        for metric in metrics:
            values = [comparison["data"][name]["metrics"].get(metric) 
                     for name in model_names 
                     if metric in comparison["data"][name]["metrics"]
                     and comparison["data"][name]["metrics"][metric] is not None]
            
            if values:
                metric_ranges[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values) if len(values) > 1 else 0,
                    "mean": np.mean(values)
                }
        
        comparison["metric_ranges"] = metric_ranges
        
        # Create visualization if requested
        if include_plot:
            try:
                plots_dir = os.path.join(self.config.model_path, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                for metric in metrics:
                    # Skip if not enough data points
                    values = [comparison["data"][name]["metrics"].get(metric) 
                             for name in model_names 
                             if metric in comparison["data"][name]["metrics"]
                             and comparison["data"][name]["metrics"][metric] is not None]
                    
                    if len(values) < 2:
                        continue
                        
                    plot_path = os.path.join(plots_dir, f"model_comparison_{metric}.png")
                    
                    plt.figure(figsize=(10, 6))
                    
                    # Get values for this metric
                    names = []
                    metric_values = []
                    colors = []
                    
                    for name in model_names:
                        if metric in comparison["data"][name]["metrics"] and comparison["data"][name]["metrics"][metric] is not None:
                            names.append(name)
                            metric_values.append(comparison["data"][name]["metrics"][metric])
                            colors.append('green' if comparison["data"][name]["is_best"] else 'blue')
                    
                    # Create bar chart
                    plt.bar(names, metric_values, color=colors)
                    plt.title(f'Model Comparison - {metric}')
                    plt.ylabel(metric)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()
                    
                    comparison[f"{metric}_plot"] = plot_path
                
                # Create radar chart for multi-metric comparison if we have enough metrics
                if len(metrics) >= 3:
                    plot_path = os.path.join(plots_dir, "model_comparison_radar.png")
                    
                    # Normalize metric values to 0-1 scale for radar chart
                    normalized_data = {}
                    for name in model_names:
                        normalized_data[name] = []
                        for metric in metrics:
                            if metric in comparison["data"][name]["metrics"] and metric in metric_ranges:
                                value = comparison["data"][name]["metrics"][metric]
                                # For regression metrics like MSE, lower is better, so invert
                                should_invert = self.config.task_type == TaskType.REGRESSION and metric.lower() in ['mse', 'rmse', 'mae']
                                
                                if value is not None and metric_ranges[metric]["range"] > 0:
                                    if should_invert:
                                        # Invert so lower values become higher scores
                                        normalized = 1 - ((value - metric_ranges[metric]["min"]) / metric_ranges[metric]["range"])
                                    else:
                                        normalized = (value - metric_ranges[metric]["min"]) / metric_ranges[metric]["range"]
                                    normalized_data[name].append(normalized)
                                else:
                                    normalized_data[name].append(0)
                            else:
                                normalized_data[name].append(0)
                    
                    # Create radar chart
                    plt.figure(figsize=(10, 8))
                    
                    # Set up the radar chart with correct number of variables
                    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
                    angles += angles[:1]  # Close the loop
                    
                    ax = plt.subplot(111, polar=True)
                    
                    # Add metrics labels
                    plt.xticks(angles[:-1], metrics)
                    
                    # Plot each model
                    for name in model_names:
                        values = normalized_data[name]
                        values += values[:1]  # Close the loop
                        ax.plot(angles, values, linewidth=2, label=name)
                        ax.fill(angles, values, alpha=0.1)
                    
                    plt.title('Model Comparison - Multiple Metrics')
                    plt.legend(loc='upper right')
                    plt.savefig(plot_path)
                    plt.close()
                    
                    comparison["radar_plot"] = plot_path
                
                self.logger.info(f"Model comparison plots saved to {plots_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create comparison plots: {str(e)}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
        
        # Generate report if output file specified
        if output_file:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Generate report if output file specified
                if output_file:
                    try:
                        # Create directory if it doesn't exist
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        
                        # Create report in markdown format
                        report = "# Model Comparison Report\n\n"
                        report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        
                        # Summary section
                        report += "## Summary\n\n"
                        report += f"Comparing {len(model_names)} models: {', '.join(model_names)}\n\n"
                        
                        if comparison["best_model"]:
                            report += f"The best performing model is **{comparison['best_model']}**.\n\n"
                        
                        # Metrics comparison table
                        report += "## Metrics Comparison\n\n"
                        
                        # Header row
                        report += "| Model | " + " | ".join(metrics) + " |\n"
                        report += "| --- | " + " | ".join(["---" for _ in metrics]) + " |\n"
                        
                        # Data rows
                        for name in model_names:
                            # For best model, add highlight
                            model_label = f"**{name}**" if comparison["data"][name]["is_best"] else name
                            row = f"| {model_label} |"
                            
                            for metric in metrics:
                                value = comparison["data"][name]["metrics"].get(metric)
                                if value is not None:
                                    row += f" {value:.4f} |"
                                else:
                                    row += " N/A |"
                                    
                            report += row + "\n"
                        
                        # Add visualizations if created
                        if include_plot:
                            report += "\n## Visualizations\n\n"
                            
                            for metric in metrics:
                                if f"{metric}_plot" in comparison:
                                    plot_path = comparison[f"{metric}_plot"]
                                    rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))
                                    report += f"### {metric}\n\n"
                                    report += f"![{metric} comparison]({rel_path})\n\n"
                            
                            if "radar_plot" in comparison:
                                radar_path = comparison["radar_plot"]
                                rel_path = os.path.relpath(radar_path, os.path.dirname(output_file))
                                report += f"### Multi-metric Comparison\n\n"
                                report += f"![Radar chart comparison]({rel_path})\n\n"
                        
                        # Model details section
                        report += "## Model Details\n\n"
                        
                        for name in model_names:
                            report += f"### {name}\n\n"
                            report += f"- Type: {comparison['data'][name]['model_type']}\n"
                            report += f"- Best model: {'Yes' if comparison['data'][name]['is_best'] else 'No'}\n\n"
                            
                            # Add hyperparameters
                            if comparison["data"][name]["params"]:
                                report += "#### Hyperparameters\n\n"
                                report += "| Parameter | Value |\n"
                                report += "| --- | --- |\n"
                                
                                for param, value in comparison["data"][name]["params"].items():
                                    report += f"| {param} | {value} |\n"
                                
                                report += "\n"
                        
                        # Write report to file
                        with open(output_file, 'w') as f:
                            f.write(report)
                        
                        comparison["report_path"] = output_file
                        self.logger.info(f"Model comparison report saved to {output_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to save report: {str(e)}")
                        if self.config.debug_mode:
                            self.logger.error(traceback.format_exc())
            except:
                pass
        
        return comparison

    def perform_error_analysis(self, model_name=None, X_test=None, y_test=None, 
                             n_samples=100, output_file=None, include_plot=True):
        """
        Perform detailed error analysis on model predictions to identify patterns.
        
        Args:
            model_name: Name of the model to analyze (uses best model if None)
            X_test: Test features (uses cached test data if None)
            y_test: Test targets (uses cached test data if None)
            n_samples: Number of error samples to analyze in detail
            output_file: Path to save analysis report
            
        Returns:
            Dictionary with error analysis results
        """
        # Determine which model to use
        if model_name is None and self.best_model is not None:
            model = self.best_model["model"]
            model_name = self.best_model["name"]
        elif model_name in self.models:
            model = self.models[model_name]["model"]
        else:
            self.logger.error(f"Model {model_name} not found")
            return {"error": f"Model {model_name} not found"}
            
        # Use provided test data or fall back to cached data
        if X_test is None or y_test is None:
            if hasattr(self, '_last_X_test') and hasattr(self, '_last_y_test'):
                X_test = self._last_X_test
                y_test = self._last_y_test
                self.logger.info("Using cached test data for error analysis")
            else:
                self.logger.error("No test data provided and no cached test data available")
                return {"error": "No test data available"}
                
        self.logger.info(f"Performing error analysis on model: {model_name}")
        
        try:
            # Store original data for reference
            original_X_test = X_test
            
            # Apply preprocessing if needed
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                X_test = self.preprocessor.transform(X_test)
            
            # Make predictions
            if self.config.task_type == TaskType.CLASSIFICATION:
                y_pred = model.predict(X_test)
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except Exception as e:
                        self.logger.warning(f"Could not get probability predictions: {str(e)}")
            else:
                y_pred = model.predict(X_test)
            
            # Prepare basic analysis
            analysis = {
                "model_name": model_name,
                "dataset_size": len(y_test),
                "timestamp": time.time()
            }
            
            # Calculate error indices
            error_indices = np.where(y_pred != y_test)[0] if self.config.task_type == TaskType.CLASSIFICATION else None
            
            # For regression, calculate absolute errors
            if self.config.task_type == TaskType.REGRESSION:
                abs_errors = np.abs(y_pred - y_test)
                # Define outliers as errors more than 2 standard deviations from mean
                error_threshold = np.mean(abs_errors) + 2 * np.std(abs_errors)
                error_indices = np.where(abs_errors > error_threshold)[0]
            
            # Add error summary
            if error_indices is not None:
                analysis["error_count"] = len(error_indices)
                analysis["error_rate"] = len(error_indices) / len(y_test)
                
                # If no errors found, return early
                if len(error_indices) == 0:
                    self.logger.info(f"No errors found in test data for model {model_name}")
                    return analysis
            else:
                analysis["error_rate"] = 0.0
                
            # For classification, analyze errors by class
            if self.config.task_type == TaskType.CLASSIFICATION:
                # Get unique classes
                classes = np.unique(np.concatenate([y_test, y_pred]))
                
                # Calculate per-class metrics
                class_metrics = {}
                for cls in classes:
                    # True positives, false positives, etc.
                    true_pos = np.sum((y_test == cls) & (y_pred == cls))
                    false_pos = np.sum((y_test != cls) & (y_pred == cls))
                    false_neg = np.sum((y_test == cls) & (y_pred != cls))
                    true_neg = np.sum((y_test != cls) & (y_pred != cls))
                    
                    # Calculate metrics
                    precision = true_pos / max(1, true_pos + false_pos)
                    recall = true_pos / max(1, true_pos + false_neg)
                    f1 = 2 * precision * recall / max(1e-8, precision + recall)
                    support = np.sum(y_test == cls)
                    
                    class_metrics[str(cls)] = {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "support": int(support),
                        "error_count": int(false_pos + false_neg),
                        "error_rate": (false_pos + false_neg) / max(1, support)
                    }
                
                analysis["class_metrics"] = class_metrics
                
                # Create confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred)
                analysis["confusion_matrix"] = cm.tolist()
                
                # Analyze top misclassifications
                misclassifications = {}
                for i in error_indices:
                    true_label = str(y_test[i])
                    pred_label = str(y_pred[i])
                    key = f"{true_label}->{pred_label}"
                    
                    if key not in misclassifications:
                        misclassifications[key] = {
                            "count": 0,
                            "indices": [],
                            "true_label": true_label,
                            "predicted_label": pred_label
                        }
                    
                    misclassifications[key]["count"] += 1
                    misclassifications[key]["indices"].append(int(i))
                
                # Sort by frequency and get top misclassifications
                sorted_misclass = sorted(misclassifications.items(), key=lambda x: x[1]["count"], reverse=True)
                top_misclass = dict(sorted_misclass[:min(10, len(sorted_misclass))])
                
                analysis["top_misclassifications"] = top_misclass
                
                # Detailed analysis of misclassified examples
                if n_samples > 0:
                    sample_indices = error_indices[:min(n_samples, len(error_indices))]
                    
                    detailed_samples = []
                    for idx in sample_indices:
                        sample = {
                            "index": int(idx),
                            "true_label": str(y_test[idx]),
                            "predicted_label": str(y_pred[idx]),
                            "confidence": None
                        }
                        
                        # Add confidence if available
                        if y_proba is not None:
                            pred_class_idx = np.where(classes == y_pred[idx])[0][0]
                            sample["confidence"] = float(y_proba[idx, pred_class_idx])
                            
                            # Add all class probabilities
                            sample["class_probabilities"] = {
                                str(cls): float(y_proba[idx, i]) 
                                for i, cls in enumerate(classes)
                            }
                        
                        # Add feature values if original data is available
                        if isinstance(original_X_test, (pd.DataFrame, np.ndarray)):
                            # Get feature values
                            if isinstance(original_X_test, pd.DataFrame):
                                feature_values = original_X_test.iloc[idx].to_dict()
                            else:
                                # For numpy arrays, create generic feature names
                                feature_values = {
                                    f"feature_{i}": float(val) 
                                    for i, val in enumerate(original_X_test[idx])
                                }
                            
                            sample["feature_values"] = feature_values
                            
                            # If feature importance is available, add important features
                            if hasattr(self, '_last_feature_importance') and self._last_feature_importance is not None:
                                # Get feature importance
                                importance = self._last_feature_importance
                                if len(importance) == len(feature_values):
                                    # Get top features for this sample
                                    top_features = sorted(
                                        [(name, importance[i], feature_values[name]) 
                                         for i, name in enumerate(feature_values.keys())],
                                        key=lambda x: x[1],
                                        reverse=True
                                    )[:5]  # Top 5 features
                                    
                                    sample["top_features"] = [
                                        {"name": name, "importance": float(imp), "value": float(val)}
                                        for name, imp, val in top_features
                                    ]
                        
                        detailed_samples.append(sample)
                    
                    analysis["detailed_samples"] = detailed_samples
            
            # For regression, analyze error distribution
            if self.config.task_type == TaskType.REGRESSION:
                # Calculate error statistics
                errors = y_test - y_pred
                abs_errors = np.abs(errors)
                
                error_stats = {
                    "mean_error": float(np.mean(errors)),
                    "mean_abs_error": float(np.mean(abs_errors)),
                    "median_abs_error": float(np.median(abs_errors)),
                    "std_error": float(np.std(errors)),
                    "max_error": float(np.max(abs_errors)),
                    "min_error": float(np.min(abs_errors))
                }
                
                # Calculate error percentiles
                percentiles = [10, 25, 50, 75, 90, 95, 99]
                for p in percentiles:
                    error_stats[f"percentile_{p}"] = float(np.percentile(abs_errors, p))
                
                analysis["error_stats"] = error_stats
                
                # Analyze data points with highest errors
                if n_samples > 0:
                    # Get indices of highest errors
                    high_error_indices = np.argsort(abs_errors)[-min(n_samples, len(abs_errors)):]
                    
                    detailed_samples = []
                    for idx in high_error_indices:
                        sample = {
                            "index": int(idx),
                            "true_value": float(y_test[idx]),
                            "predicted_value": float(y_pred[idx]),
                            "error": float(errors[idx]),
                            "abs_error": float(abs_errors[idx])
                        }
                        
                        # Add feature values if original data is available
                        if isinstance(original_X_test, (pd.DataFrame, np.ndarray)):
                            # Get feature values
                            if isinstance(original_X_test, pd.DataFrame):
                                feature_values = original_X_test.iloc[idx].to_dict()
                            else:
                                # For numpy arrays, create generic feature names
                                feature_values = {
                                    f"feature_{i}": float(val) 
                                    for i, val in enumerate(original_X_test[idx])
                                }
                            
                            sample["feature_values"] = feature_values
                            
                            # If feature importance is available, add important features
                            if hasattr(self, '_last_feature_importance') and self._last_feature_importance is not None:
                                # Get feature importance
                                importance = self._last_feature_importance
                                if len(importance) == len(feature_values):
                                    # Get top features for this sample
                                    top_features = sorted(
                                        [(name, importance[i], feature_values[name]) 
                                         for i, name in enumerate(feature_values.keys())],
                                        key=lambda x: x[1],
                                        reverse=True
                                    )[:5]  # Top 5 features
                                    
                                    sample["top_features"] = [
                                        {"name": name, "importance": float(imp), "value": float(val)}
                                        for name, imp, val in top_features
                                    ]
                        
                        detailed_samples.append(sample)
                    
                    analysis["detailed_samples"] = detailed_samples
                    
                # Add error distribution by range
                ranges = np.linspace(0, np.max(abs_errors), 10)
                distribution = []
                
                for i in range(len(ranges)-1):
                    range_start = ranges[i]
                    range_end = ranges[i+1]
                    count = np.sum((abs_errors >= range_start) & (abs_errors < range_end))
                    
                    distribution.append({
                        "range_start": float(range_start),
                        "range_end": float(range_end),
                        "count": int(count),
                        "percentage": float(count / len(abs_errors) * 100)
                    })
                
                analysis["error_distribution"] = distribution
            
            # Create visualizations
            if include_plot:
                try:
                    plots_dir = os.path.join(self.config.model_path, "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    if self.config.task_type == TaskType.CLASSIFICATION:
                        # Plot confusion matrix
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=classes, yticklabels=classes)
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title(f'Confusion Matrix - {model_name}')
                        
                        plot_path = os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
                        plt.savefig(plot_path)
                        plt.close()
                        
                        analysis["confusion_matrix_plot"] = plot_path
                        
                        # Plot class error rates
                        plt.figure(figsize=(12, 6))
                        class_names = []
                        error_rates = []
                        supports = []
                        
                        for cls, metrics in class_metrics.items():
                            class_names.append(cls)
                            error_rates.append(metrics["error_rate"])
                            supports.append(metrics["support"])
                        
                        # Create plot with double y-axis
                        fig, ax1 = plt.subplots(figsize=(12, 6))
                        
                        bars = ax1.bar(class_names, error_rates, alpha=0.7)
                        ax1.set_xlabel('Class')
                        ax1.set_ylabel('Error Rate')
                        ax1.set_ylim(0, min(1.0, max(error_rates) * 1.2))
                        
                        # Add data labels
                        for bar in bars:
                            height = bar.get_height()
                            ax1.annotate(f'{height:.2f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),  # 3 points vertical offset
                                        textcoords="offset points",
                                        ha='center', va='bottom')
                        
                        # Create a second y-axis for the support count
                        ax2 = ax1.twinx()
                        ax2.plot(class_names, supports, 'ro-', alpha=0.7)
                        ax2.set_ylabel('Support (sample count)', color='r')
                        ax2.tick_params(axis='y', labelcolor='r')
                        
                        plt.title(f'Class Error Rates - {model_name}')
                        plt.tight_layout()
                        
                        plot_path = os.path.join(plots_dir, f"{model_name}_class_error_rates.png")
                        plt.savefig(plot_path)
                        plt.close()
                        
                        analysis["class_error_plot"] = plot_path
                        
                    elif self.config.task_type == TaskType.REGRESSION:
                        # Plot error distribution histogram
                        plt.figure(figsize=(10, 6))
                        plt.hist(abs_errors, bins=30, alpha=0.7, color='blue')
                        plt.axvline(error_stats["mean_abs_error"], color='red', linestyle='dashed', linewidth=2,
                                   label=f'Mean = {error_stats["mean_abs_error"]:.2f}')
                        plt.axvline(error_stats["median_abs_error"], color='green', linestyle='dashed', linewidth=2,
                                   label=f'Median = {error_stats["median_abs_error"]:.2f}')
                        plt.xlabel('Absolute Error')
                        plt.ylabel('Count')
                        plt.title(f'Error Distribution - {model_name}')
                        plt.legend()
                        
                        plot_path = os.path.join(plots_dir, f"{model_name}_error_distribution.png")
                        plt.savefig(plot_path)
                        plt.close()
                        
                        analysis["error_distribution_plot"] = plot_path
                        
                        # Plot predicted vs actual values
                        plt.figure(figsize=(10, 10))
                        plt.scatter(y_test, y_pred, alpha=0.5)
                        
                        # Add perfect prediction line
                        min_val = min(np.min(y_test), np.min(y_pred))
                        max_val = max(np.max(y_test), np.max(y_pred))
                        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                        
                        plt.xlabel('Actual Values')
                        plt.ylabel('Predicted Values')
                        plt.title(f'Predicted vs Actual - {model_name}')
                        
                        plot_path = os.path.join(plots_dir, f"{model_name}_predicted_vs_actual.png")
                        plt.savefig(plot_path)
                        plt.close()
                        
                        analysis["predicted_vs_actual_plot"] = plot_path
                        
                        # Plot residuals
                        plt.figure(figsize=(10, 6))
                        plt.scatter(y_pred, errors, alpha=0.5)
                        plt.axhline(y=0, color='r', linestyle='--')
                        plt.xlabel('Predicted Values')
                        plt.ylabel('Residuals')
                        plt.title(f'Residual Plot - {model_name}')
                        
                        plot_path = os.path.join(plots_dir, f"{model_name}_residuals.png")
                        plt.savefig(plot_path)
                        plt.close()
                        
                        analysis["residuals_plot"] = plot_path
                
                except Exception as e:
                    self.logger.error(f"Failed to create error analysis plots: {str(e)}")
                    if self.config.debug_mode:
                        self.logger.error(traceback.format_exc())
            
            # Generate report if output file specified
            if output_file:
                self._generate_error_analysis_report(analysis, output_file)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analysis failed: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _generate_error_analysis_report(self, analysis, output_file):
        """Generate markdown report from error analysis results"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Create report in markdown format
            report = f"# Error Analysis Report - {analysis['model_name']}\n\n"
            report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Summary section
            report += "## Summary\n\n"
            report += f"- **Dataset Size**: {analysis['dataset_size']}\n"
            
            if 'error_count' in analysis:
                report += f"- **Total Errors**: {analysis['error_count']}\n"
                report += f"- **Error Rate**: {analysis['error_rate']:.2%}\n\n"
            
            # Classification specific analysis
            if self.config.task_type == TaskType.CLASSIFICATION:
                # Add confusion matrix plot if available
                if 'confusion_matrix_plot' in analysis:
                    plot_path = analysis['confusion_matrix_plot']
                    rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))
                    report += "## Confusion Matrix\n\n"
                    report += f"![Confusion Matrix]({rel_path})\n\n"
                
                # Add class error rates plot if available
                if 'class_error_plot' in analysis:
                    plot_path = analysis['class_error_plot']
                    rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))
                    report += "## Class Error Rates\n\n"
                    report += f"![Class Error Rates]({rel_path})\n\n"
                
                # Add per-class metrics
                if 'class_metrics' in analysis:
                    report += "## Per-Class Metrics\n\n"
                    report += "| Class | Support | Error Count | Error Rate | Precision | Recall | F1 |\n"
                    report += "| --- | --- | --- | --- | --- | --- | --- |\n"
                    
                    for cls, metrics in analysis['class_metrics'].items():
                        report += f"| {cls} | {metrics['support']} | {metrics['error_count']} | "
                        report += f"{metrics['error_rate']:.2%} | {metrics['precision']:.4f} | "
                        report += f"{metrics['recall']:.4f} | {metrics['f1']:.4f} |\n"
                    
                    report += "\n"
                
                # Add top misclassifications
                if 'top_misclassifications' in analysis:
                    report += "## Top Misclassifications\n\n"
                    report += "| True Label | Predicted Label | Count | Percentage |\n"
                    report += "| --- | --- | --- | --- |\n"
                    
                    for error_type, error_info in analysis['top_misclassifications'].items():
                        true_label = error_info['true_label']
                        pred_label = error_info['predicted_label']
                        count = error_info['count']
                        percentage = count / analysis['error_count'] * 100 if analysis['error_count'] > 0 else 0
                        
                        report += f"| {true_label} | {pred_label} | {count} | {percentage:.2f}% |\n"
                    
                    report += "\n"
                
            # Regression specific analysis
            elif self.config.task_type == TaskType.REGRESSION:
                # Add error statistics
                if 'error_stats' in analysis:
                    report += "## Error Statistics\n\n"
                    report += "| Metric | Value |\n"
                    report += "| --- | --- |\n"
                    
                    for metric, value in analysis['error_stats'].items():
                        if isinstance(value, float):
                            report += f"| {metric} | {value:.4f} |\n"
                        else:
                            report += f"| {metric} | {value} |\n"
                    
                    report += "\n"
                
                # Add plots if available
                for plot_name, title in [
                    ('error_distribution_plot', 'Error Distribution'),
                    ('predicted_vs_actual_plot', 'Predicted vs Actual Values'),
                    ('residuals_plot', 'Residuals')
                ]:
                    if plot_name in analysis:
                        plot_path = analysis[plot_name]
                        rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))
                        report += f"## {title}\n\n"
                        report += f"![{title}]({rel_path})\n\n"
                
                # Add error distribution by range
                if 'error_distribution' in analysis:
                    report += "## Error Distribution by Range\n\n"
                    report += "| Error Range | Count | Percentage |\n"
                    report += "| --- | --- | --- |\n"
                    
                    for range_info in analysis['error_distribution']:
                        range_str = f"{range_info['range_start']:.2f} - {range_info['range_end']:.2f}"
                        report += f"| {range_str} | {range_info['count']} | {range_info['percentage']:.2f}% |\n"
                    
                    report += "\n"
            
            # Add detailed error samples
            if 'detailed_samples' in analysis:
                report += "## Detailed Error Analysis\n\n"
                report += f"Showing {len(analysis['detailed_samples'])} sample errors for detailed analysis.\n\n"
                
                for i, sample in enumerate(analysis['detailed_samples'][:10]):  # Limit to 10 in the report
                    report += f"### Error Sample {i+1}\n\n"
                    
                    if self.config.task_type == TaskType.CLASSIFICATION:
                        report += f"- **True Label**: {sample['true_label']}\n"
                        report += f"- **Predicted Label**: {sample['predicted_label']}\n"
                        
                        if 'confidence' in sample and sample['confidence'] is not None:
                            report += f"- **Confidence**: {sample['confidence']:.4f}\n"
                        
                        if 'class_probabilities' in sample:
                            report += "- **Class Probabilities**:\n"
                            for cls, prob in sorted(sample['class_probabilities'].items(), 
                                                key=lambda x: x[1], reverse=True):
                                report += f"  - {cls}: {prob:.4f}\n"
                    
                    elif self.config.task_type == TaskType.REGRESSION:
                        report += f"- **True Value**: {sample['true_value']:.4f}\n"
                        report += f"- **Predicted Value**: {sample['predicted_value']:.4f}\n"
                        report += f"- **Error**: {sample['error']:.4f}\n"
                        report += f"- **Absolute Error**: {sample['abs_error']:.4f}\n"
                    
                    # Add feature values
                    if 'feature_values' in sample:
                        report += "- **Feature Values**:\n"
                        
                        # If we have top features, prioritize them
                        if 'top_features' in sample:
                            report += "  - **Most Important Features**:\n"
                            for feature in sample['top_features']:
                                report += f"    - {feature['name']}: {feature['value']:.4f} "
                                report += f"(importance: {feature['importance']:.4f})\n"
                            
                            report += "  - **Other Features**:\n"
                            # Find features not in top features
                            top_names = {f['name'] for f in sample['top_features']}
                            other_features = {k: v for k, v in sample['feature_values'].items() 
                                            if k not in top_names}
                            
                            # Show only a few other features
                            for name, value in list(other_features.items())[:5]:
                                if isinstance(value, (int, float)):
                                    report += f"    - {name}: {value:.4f}\n"
                                else:
                                    report += f"    - {name}: {value}\n"
                            
                            if len(other_features) > 5:
                                report += f"    - ... and {len(other_features) - 5} more features\n"
                        else:
                            # Without importance info, just show all features (limited)
                            sorted_features = sorted(sample['feature_values'].items())
                            for name, value in sorted_features[:10]:
                                if isinstance(value, (int, float)):
                                    report += f"  - {name}: {value:.4f}\n"
                                else:
                                    report += f"  - {name}: {value}\n"
                            
                            if len(sorted_features) > 10:
                                report += f"  - ... and {len(sorted_features) - 10} more features\n"
                    
                    report += "\n"
                
                # If there are more samples than shown, indicate this
                if len(analysis['detailed_samples']) > 10:
                    report += f"*Note: Only showing 10 of {len(analysis['detailed_samples'])} error samples.*\n\n"
            
            # Recommendations section
            report += "## Recommendations\n\n"
            
            if self.config.task_type == TaskType.CLASSIFICATION:
                # Add class-specific recommendations
                if 'class_metrics' in analysis:
                    worst_classes = sorted(
                        [(cls, metrics['error_rate']) for cls, metrics in analysis['class_metrics'].items()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]  # Top 3 worst classes
                    
                    if worst_classes:
                        report += "### Class-Specific Issues\n\n"
                        report += "Consider addressing issues with these problematic classes:\n\n"
                        
                        for cls, error_rate in worst_classes:
                            report += f"- **Class {cls}** has a high error rate of {error_rate:.2%}. "
                            
                            # Add more specific recommendations based on precision/recall
                            metrics = analysis['class_metrics'][cls]
                            if metrics['precision'] < metrics['recall']:
                                report += "Low precision indicates the model is incorrectly classifying other classes as this one. "
                                report += "Consider adding more discriminative features or collecting more data for confused classes.\n"
                            else:
                                report += "Low recall indicates the model is missing instances of this class. "
                                report += "Consider collecting more varied examples of this class or applying class weighting.\n"
                        
                        report += "\n"
                
                # Add misclassification-specific recommendations
                if 'top_misclassifications' in analysis and len(analysis['top_misclassifications']) > 0:
                    report += "### Common Misclassification Patterns\n\n"
                    report += "Address these common misclassification patterns:\n\n"
                    
                    for i, (error_type, error_info) in enumerate(list(analysis['top_misclassifications'].items())[:3]):
                        true_label = error_info['true_label']
                        pred_label = error_info['predicted_label']
                        count = error_info['count']
                        
                        report += f"- The model frequently confuses **{true_label}** as **{pred_label}** ({count} instances). "
                        report += "Consider adding features that better distinguish between these classes or "
                        report += "review the training data for potential labeling errors.\n"
                    
                    report += "\n"
            
            elif self.config.task_type == TaskType.REGRESSION:
                report += "### Error Distribution Analysis\n\n"
                
                # Add recommendations based on error distribution
                if 'error_stats' in analysis:
                    mean_error = analysis['error_stats']['mean_error']
                    std_error = analysis['error_stats']['std_error']
                    
                    # Check for bias
                    if abs(mean_error) > std_error * 0.25:
                        direction = "overestimating" if mean_error < 0 else "underestimating"
                        report += f"- **Prediction Bias**: The model is systematically {direction} values "
                        report += f"(mean error: {mean_error:.4f}). Consider adding more features or transforming "
                        report += "existing features to address this systematic bias.\n\n"
                    
                    # Check for high error variance
                    if 'percentile_95' in analysis['error_stats'] and 'mean_abs_error' in analysis['error_stats']:
                        p95 = analysis['error_stats']['percentile_95']
                        mean = analysis['error_stats']['mean_abs_error']
                        
                        if p95 > mean * 3:
                            report += f"- **High Error Variance**: While the average error is {mean:.4f}, 5% of predictions "
                            report += f"have errors above {p95:.4f}. Consider techniques like outlier removal during training "
                            report += "or developing specialized models for high-error regions.\n\n"
                
                # Add recommendations based on residual plot
                if 'residuals_plot' in analysis:
                    report += "- **Residual Analysis**: Review the residual plot to identify any patterns in errors. "
                    report += "If errors increase with predicted values, consider log-transforming the target. "
                    report += "If there are clusters of errors, the model might need additional features to capture "
                    report += "those patterns.\n\n"
            
            # General recommendations
            report += "### General Improvements\n\n"
            report += "Consider these general approaches to improve model performance:\n\n"
            report += "1. **Feature Engineering**: Add new features or transform existing ones to better capture patterns in the data.\n"
            report += "2. **Data Quality**: Review the training data for potential errors or inconsistencies.\n"
            report += "3. **Model Complexity**: Adjust model complexity by tuning hyperparameters or trying different algorithms.\n"
            report += "4. **Ensemble Methods**: Combine multiple models to reduce errors and improve generalization.\n"
            report += "5. **Additional Data**: Collect more training data, especially for poorly performing areas.\n\n"
            
            # Write report to file
            with open(output_file, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Error analysis report saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate error analysis report: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())

    def detect_data_drift(self, new_data, reference_data=None, drift_threshold=0.1):
        """
        Detect distribution shift between new data and reference data or training data.
        
        Args:
            new_data: New data to check for drift
            reference_data: Reference data to compare against (uses training data if None)
            drift_threshold: Threshold for detecting significant drift
            
        Returns:
            Dictionary with drift detection results
        """
        self.logger.info("Performing data drift analysis")
        
        # If no reference data provided, use cached training data if available
        if reference_data is None:
            if hasattr(self, '_last_X_train'):
                reference_data = self._last_X_train
                self.logger.info("Using cached training data as reference")
            else:
                self.logger.error("No reference data provided and no cached training data available")
                return {"error": "No reference data available"}
        
        try:
            # Ensure both datasets are in same format
            if isinstance(new_data, pd.DataFrame) and isinstance(reference_data, pd.DataFrame):
                # Match columns between datasets
                common_cols = list(set(new_data.columns) & set(reference_data.columns))
                if not common_cols:
                    self.logger.error("No common columns between new data and reference data")
                    return {"error": "No common columns between datasets"}
                
                new_data = new_data[common_cols]
                reference_data = reference_data[common_cols]
                feature_names = common_cols
            else:
                # For numpy arrays, assume same structure
                if new_data.shape[1] != reference_data.shape[1]:
                    self.logger.error(f"Feature dimension mismatch: {new_data.shape[1]} vs {reference_data.shape[1]}")
                    return {"error": "Feature dimension mismatch between datasets"}
                
                feature_names = [f"feature_{i}" for i in range(new_data.shape[1])]
            
            # Apply preprocessing if needed
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                try:
                    new_data_proc = self.preprocessor.transform(new_data)
                    reference_data_proc = self.preprocessor.transform(reference_data)
                except Exception as e:
                    self.logger.warning(f"Could not apply preprocessing: {str(e)}")
                    new_data_proc = new_data
                    reference_data_proc = reference_data
            else:
                new_data_proc = new_data
                reference_data_proc = reference_data
            
            # Calculate distribution statistics for each feature
            drift_results = {
                "feature_drift": {},
                "dataset_drift": 0.0,
                "drifted_features": [],
                "drift_detected": False,
                "sample_counts": {
                    "new_data": len(new_data),
                    "reference_data": len(reference_data)
                }
            }
            
            # Metrics to calculate
            metrics = [
                "mean_difference", 
                "std_difference", 
                "ks_statistic", 
                "ks_pvalue",
                "wasserstein_distance",
                "earth_movers_distance"
            ]
            
            # Calculate drift for each feature
            from scipy import stats
            import numpy as np
            
            for i, feature in enumerate(feature_names):
                # Extract feature data
                new_feature = new_data_proc[:, i] if isinstance(new_data_proc, np.ndarray) else new_data_proc[feature]
                ref_feature = reference_data_proc[:, i] if isinstance(reference_data_proc, np.ndarray) else reference_data_proc[feature]
                
                # Calculate basic statistics
                new_mean = np.mean(new_feature)
                ref_mean = np.mean(ref_feature)
                new_std = np.std(new_feature)
                ref_std = np.std(ref_feature)
                
                # Calculate relative differences
                mean_diff_rel = abs(new_mean - ref_mean) / max(abs(ref_mean), 1e-6)
                std_diff_rel = abs(new_std - ref_std) / max(ref_std, 1e-6)
                
                # Statistical tests
                ks_stat, ks_pval = stats.ks_2samp(new_feature, ref_feature)
                
                # Earth mover's distance (Wasserstein)
                wasserstein = stats.wasserstein_distance(new_feature, ref_feature)
                earth_movers = wasserstein  # Alias for clarity
                
                # Store results
                feature_drift = {
                    "new_mean": float(new_mean),
                    "reference_mean": float(ref_mean),
                    "new_std": float(new_std),
                    "reference_std": float(ref_std),
                    "mean_difference": float(mean_diff_rel),
                    "std_difference": float(std_diff_rel),
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pval),
                    "wasserstein_distance": float(wasserstein),
                    "earth_movers_distance": float(earth_movers)
                }
                
                # Determine if this feature has drifted
                # Using multiple criteria for robust detection
                is_drifted = (
                    (mean_diff_rel > drift_threshold) or
                    (std_diff_rel > drift_threshold) or
                    (ks_pval < 0.05 and ks_stat > 0.1) or  # Statistically significant KS test
                    (wasserstein > drift_threshold * np.max([new_std, ref_std]))  # Significant EMD
                )
                
                feature_drift["drift_detected"] = is_drifted
                
                if is_drifted:
                    drift_results["drifted_features"].append(feature)
                
                drift_results["feature_drift"][feature] = feature_drift
            
            # Calculate overall dataset drift
            # We'll use the average of normalized feature drift metrics
            feature_drift_scores = []
            for feature, metrics in drift_results["feature_drift"].items():
                # Combine normalized metrics into a single score
                drift_score = (
                    metrics["mean_difference"] + 
                    metrics["std_difference"] + 
                    metrics["ks_statistic"] +
                    metrics["wasserstein_distance"] / (metrics["new_std"] + metrics["reference_std"] + 1e-6)
                ) / 4.0  # Simple average of 4 normalized metrics
                
                feature_drift_scores.append(drift_score)
            
            # Overall drift is average of feature drift scores
            if feature_drift_scores:
                drift_results["dataset_drift"] = float(np.mean(feature_drift_scores))
                drift_results["max_feature_drift"] = float(np.max(feature_drift_scores))
                drift_results["drift_detected"] = (
                    drift_results["dataset_drift"] > drift_threshold or 
                    len(drift_results["drifted_features"]) / len(feature_names) > 0.3  # Over 30% of features drifted
                )
            
            # Create visualizations
            try:
                plots_dir = os.path.join(self.config.model_path, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                # Plot feature drift scores
                plt.figure(figsize=(12, 6))
                feature_scores = [(feature, drift_results["feature_drift"][feature]["drift_detected"]) 
                                 for feature in feature_names]
                feature_scores.sort(key=lambda x: drift_results["feature_drift"][x[0]]["mean_difference"], reverse=True)
                
                features = [f[0] for f in feature_scores]
                is_drifted = [f[1] for f in feature_scores]
                colors = ['red' if d else 'blue' for d in is_drifted]
                
                scores = [drift_results["feature_drift"][f]["mean_difference"] for f in features]
                
                plt.bar(features, scores, color=colors)
                plt.axhline(y=drift_threshold, color='r', linestyle='--', label=f'Threshold ({drift_threshold})')
                plt.xlabel('Features')
                plt.ylabel('Mean Difference')
                plt.title('Feature Drift Analysis')
                plt.xticks(rotation=90)
                plt.legend()
                plt.tight_layout()
                
                plot_path = os.path.join(plots_dir, "data_drift_analysis.png")
                plt.savefig(plot_path)
                plt.close()
                
                drift_results["drift_plot"] = plot_path
                
                # Plot distribution comparison for top drifted features
                if drift_results["drifted_features"]:
                    n_plots = min(4, len(drift_results["drifted_features"]))
                    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots))
                    
                    # Make axes indexable if there's only one
                    if n_plots == 1:
                        axes = [axes]
                    
                    for i, feature in enumerate(drift_results["drifted_features"][:n_plots]):
                        ax = axes[i]
                        
                        # Extract feature data
                        new_feature = new_data_proc[:, feature_names.index(feature)] if isinstance(new_data_proc, np.ndarray) else new_data_proc[feature]
                        ref_feature = reference_data_proc[:, feature_names.index(feature)] if isinstance(reference_data_proc, np.ndarray) else reference_data_proc[feature]
                        
                        # Plot histograms
                        ax.hist(ref_feature, bins=30, alpha=0.5, label='Reference', density=True)
                        ax.hist(new_feature, bins=30, alpha=0.5, label='New', density=True)
                        
                        # Add statistics to plot
                        metrics = drift_results["feature_drift"][feature]
                        stats_text = f"Mean Diff: {metrics['mean_difference']:.4f}\n"
                        stats_text += f"Std Diff: {metrics['std_difference']:.4f}\n"
                        stats_text += f"KS p-value: {metrics['ks_pvalue']:.4f}"
                        
                        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        ax.set_title(f'Distribution Comparison - {feature}')
                        ax.legend()
                    
                    plt.tight_layout()
                    
                    plot_path = os.path.join(plots_dir, "data_drift_distributions.png")
                    plt.savefig(plot_path)
                    plt.close()
                    
                    drift_results["distribution_plot"] = plot_path
            
            except Exception as e:
                self.logger.error(f"Failed to create data drift plots: {str(e)}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
            
            # Log the results
            if drift_results["drift_detected"]:
                self.logger.warning(
                    f"Data drift detected! Overall drift score: {drift_results['dataset_drift']:.4f}, "
                    f"Drifted features: {len(drift_results['drifted_features'])}/{len(feature_names)}"
                )
            else:
                self.logger.info(
                    f"No significant data drift detected. Overall drift score: {drift_results['dataset_drift']:.4f}"
                )
            
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Data drift analysis failed: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"error": str(e)}