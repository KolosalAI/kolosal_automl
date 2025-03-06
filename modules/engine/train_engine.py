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
    
    def generate_report(self, include_plots: bool = True):
        """Generate a report for the experiment"""
        if not self.current_experiment:
            self.logger.warning("No experiment data to generate report")
            return
            
        report = f"# Experiment Report {self.experiment_id}\n\n"
        report += f"- **Date:** {self.current_experiment['timestamp']}\n"
        report += f"- **Duration:** {self.current_experiment['duration']:.2f} seconds\n"
        report += f"- **Model:** {self.current_experiment['model_info']['name']}\n\n"
        
        report += "## Metrics\n\n"
        metrics_table = "| Metric | Value |\n| --- | --- |\n"
        for metric, value in self.current_experiment["metrics"].items():
            metrics_table += f"| {metric} | {value:.4f} |\n"
        report += metrics_table + "\n\n"
        
        if "feature_importance" in self.current_experiment and self.current_experiment["feature_importance"]:
            report += "## Top 10 Features by Importance\n\n"
            fi_table = "| Feature | Importance |\n| --- | --- |\n"
            for i, (feature, importance) in enumerate(list(self.current_experiment["feature_importance"].items())[:10]):
                fi_table += f"| {feature} | {importance:.4f} |\n"
            report += fi_table + "\n\n"
            
        # Save report
        report_file = f"{self.output_dir}/report_{self.experiment_id}.md"
        with open(report_file, 'w') as f:
            f.write(report)
            
        if include_plots and "metrics" in self.current_experiment:
            self._generate_plots()
            
        self.logger.info(f"Report generated: {report_file}")
        return report
        
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
            return "accuracy"  # Default
    
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
            
    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance from the model if available"""
        # Try different attributes that might contain feature importance
        for attr in ['feature_importances_', 'coef_', 'feature_importance_']:
            if hasattr(model, attr):
                importance = getattr(model, attr)
                if attr == 'coef_' and importance.ndim > 1:
                    # For multi-class models, take the mean absolute coefficient
                    importance = np.mean(np.abs(importance), axis=0)
                return importance
                
        # If we reach here, model doesn't have built-in feature importance
        self.logger.warning("Model doesn't provide feature importance.")
        return None
        
    def train_model(self, model, model_name, param_grid, X, y, X_test=None, y_test=None):
        """Train a model with hyperparameter optimization"""
        self.logger.info(f"Training model: {model_name}")
        
        if self.tracker:
            self.tracker.start_experiment(
                config=self.config.to_dict(),
                model_info={"name": model_name, "type": str(type(model).__name__)}
            )
            
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
            
        # Initialize and fit preprocessor
        if self.preprocessor:
            self.logger.info("Fitting preprocessor...")
            self.preprocessor.fit(X_train)
            
        # Feature selection if enabled
        if self.config.feature_selection:
            self.logger.info("Performing feature selection...")
            self.feature_selector = self._get_feature_selector(X_train, y_train)
            if self.feature_selector:
                self.feature_selector.fit(X_train, y_train)
                
                # Get feature importance scores
                if hasattr(self.feature_selector, 'scores_'):
                    feature_scores = self.feature_selector.scores_
                    if self.config.feature_selection_k is None:
                        # Filter features based on threshold
                        mask = feature_scores > np.percentile(feature_scores, 
                                                             100 * self.config.feature_importance_threshold)
                        selected_indices = np.where(mask)[0]
                        self.feature_selector.k = len(selected_indices)
                        self.logger.info(f"Selected {self.feature_selector.k} features based on threshold")
                        
                # Log initial feature selection
                if self.tracker and hasattr(self.feature_selector, 'get_support'):
                    mask = self.feature_selector.get_support()
                    if hasattr(X_train, 'columns'):  # If pandas DataFrame
                        all_features = X_train.columns
                    else:
                        all_features = [f"feature_{i}" for i in range(X_train.shape[1])]
                    
                    selected_features = [f for i, f in enumerate(all_features) if mask[i]]
                    self.logger.info(f"Selected features: {selected_features}")
        
        # Create the pipeline with preprocessor, feature selector, and model
        pipeline = self._create_pipeline(model)
        
        # Perform hyperparameter optimization
        self.logger.info(f"Performing hyperparameter optimization with {self.config.optimization_strategy.value}...")
        
        search = self._get_optimization_search(pipeline, param_grid)
        search.fit(X_train, y_train)
        
        # Get the best model from the search
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        self.logger.info(f"Best parameters: {best_params}")
        
        # Evaluate model performance
        self.logger.info("Evaluating model performance...")
        metrics = self._evaluate_model(best_model, X_train, y_train, X_test, y_test)
        self.logger.info(f"Performance metrics: {metrics}")
        
        # Get cross-validation results
        cv_results = {}
        for i, score in enumerate(search.cv_results_['split0_test_score']):
            cv_results[f"fold_{i+1}"] = score
        
        # Track metrics
        if self.tracker:
            self.tracker.log_metrics(metrics)
            self.tracker.log_metrics(cv_results, step="cv")
            
            # Log feature importance if available
            try:
                # Access the model inside the pipeline
                final_model = best_model.named_steps['model']
                feature_importance = self._get_feature_importance(final_model, None)
                
                if feature_importance is not None:
                    # Get feature names after preprocessing
                    if hasattr(X_train, 'columns'):  # If pandas DataFrame
                        feature_names = X_train.columns
                    else:
                        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                    
                    # If we use feature selector, get the selected features
                    if self.feature_selector and hasattr(self.feature_selector, 'get_support'):
                        mask = self.feature_selector.get_support()
                        feature_names = [f for i, f in enumerate(feature_names) if mask[i]]
                        
                    self.tracker.log_feature_importance(feature_names, feature_importance)
            except Exception as e:
                self.logger.warning(f"Failed to log feature importance: {str(e)}")
            
        # Store model
        self.models[model_name] = {
            "model": best_model,
            "params": best_params,
            "metrics": metrics,
            "cv_results": cv_results
        }
        
        # Check if this is the best model so far
        model_score = self._get_model_score(metrics)
        
        is_better = False
        if self.config.task_type == TaskType.REGRESSION:
            # For regression, lower error is better
            if model_score < self.best_score:
                is_better = True
        else:
            # For classification, higher score is better
            if model_score > self.best_score:
                is_better = True
                
        if is_better:
            self.best_score = model_score
            self.best_model = {
                "name": model_name,
                "model": best_model,
                "metrics": metrics
            }
            self.logger.info(f"New best model: {model_name} with score {self.best_score:.4f}")
        
        # End experiment tracking
        if self.tracker:
            self.tracker.end_experiment()
            
        return best_model, metrics
    
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

    def save_model(self, model_name=None, filepath=None):
        """Save the model to disk"""
        if model_name is None and self.best_model is not None:
            model_name = self.best_model["name"]
            model_data = self.best_model
        elif model_name in self.models:
            model_data = self.models[model_name]
        else:
            self.logger.error(f"Model {model_name} not found")
            return False
            
        if filepath is None:
            filepath = os.path.join(self.config.model_path, f"{model_name}.pkl")
            
        try:
            # Save the model with metadata
            model_package = {
                "model": model_data["model"],
                "params": model_data.get("params", {}),
                "metrics": model_data.get("metrics", {}),
                "config": self.config.to_dict(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0.0"
            }
            
            # Use joblib for efficient serialization
            joblib.dump(model_package, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
            # Optionally quantize and save a quantized version for faster inference
            if self.config.inference_config.enable_quantization:
                try:
                    # Prepare model for quantization - exclude non-numeric parts
                    model_bytes = pickle.dumps(model_data["model"])
                    model_array = np.frombuffer(model_bytes, dtype=np.uint8)
                    
                    # Quantize the model bytes
                    quantized_data = self.quantizer.quantize(model_array)
                    
                    # Save quantized model
                    quantized_filepath = os.path.join(self.config.model_path, f"{model_name}_quantized.pkl")
                    with open(quantized_filepath, 'wb') as f:
                        pickle.dump({
                            "quantized_data": quantized_data,
                            "metadata": {
                                "original_size": len(model_bytes),
                                "quantized_size": len(quantized_data),
                                "config": self.quantizer.get_config(),
                            }
                        }, f)
                    
                    self.logger.info(f"Quantized model saved to {quantized_filepath}")
                except Exception as e:
                    self.logger.warning(f"Failed to save quantized model: {str(e)}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return False
            
    def load_model(self, filepath):
        """Load a model from disk"""
        try:
            model_package = joblib.load(filepath)
            
            model_name = os.path.basename(filepath).split('.')[0]
            self.models[model_name] = {
                "model": model_package["model"],
                "params": model_package.get("params", {}),
                "metrics": model_package.get("metrics", {})
            }
            
            # Check if this is better than current best model
            metrics = model_package.get("metrics", {})
            model_score = self._get_model_score(metrics)
            
            is_better = False
            if self.config.task_type == TaskType.REGRESSION:
                if model_score < self.best_score:
                    is_better = True
            else:
                if model_score > self.best_score:
                    is_better = True
                    
            if is_better or self.best_model is None:
                self.best_score = model_score
                self.best_model = {
                    "name": model_name,
                    "model": model_package["model"],
                    "metrics": metrics
                }
                
            self.logger.info(f"Model loaded from {filepath}")
            return model_package["model"]
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return None
            
    def predict(self, X, model_name=None):
        """Make predictions using the specified or best model"""
        # Use the inference engine for predictions
        
        if model_name is None and self.best_model is not None:
            model = self.best_model["model"]
            model_name = self.best_model["name"]
        elif model_name in self.models:
            model = self.models[model_name]["model"]
        else:
            self.logger.error(f"Model {model_name} not found")
            return None
            
        # Save model temporarily for inference engine
        temp_model_path = os.path.join(self.config.model_path, f"temp_{model_name}.pkl")
        joblib.dump(model, temp_model_path)
        
        try:
            # Load model in inference engine
            self.inference_engine.load_model(temp_model_path)
            
            # Make prediction
            success, predictions, metadata = self.inference_engine.predict(X)
            
            if success:
                self.logger.info(f"Prediction successful: {metadata.get('timing', {}).get('total_time', 0):.4f}s")
                return predictions
            else:
                self.logger.error(f"Prediction failed: {metadata.get('error', 'Unknown error')}")
                # Fallback to direct prediction
                self.logger.info("Falling back to direct prediction")
                return model.predict(X)
        except Exception as e:
            self.logger.error(f"Inference engine error: {str(e)}")
            # Fallback to direct prediction
            return model.predict(X)
        finally:
            # Clean up temporary model file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                
    def run_batch_inference(self, data_generator, batch_size=None, model_name=None):
        """Run batch inference using the batch processor"""
        if model_name is None and self.best_model is not None:
            model = self.best_model["model"]
        elif model_name in self.models:
            model = self.models[model_name]["model"]
        else:
            self.logger.error(f"Model {model_name} not found")
            return None
            
        # Define batch processing function
        def process_batch(batch):
            return model.predict(batch)
            
        # Configure batch size if provided
        if batch_size is not None and hasattr(self.batch_processor, 'config'):
            self.batch_processor.config.initial_batch_size = batch_size
                
        # Process data in batches
        results = []
        batches = []  # Store original batches for reference
        
        try:
            for data_chunk in data_generator:
                batches.append(data_chunk)
                results.append(process_batch(data_chunk))
        except Exception as e:
            self.logger.error(f"Error during batch inference: {str(e)}")
                
        # Return results
        if not results:
            return None
            
        # Handle the case where arrays have different shapes
        if isinstance(results[0], np.ndarray):
            try:
                return np.vstack(results)
            except ValueError as e:
                self.logger.warning(f"Could not stack results: {str(e)}")
                self.logger.info("Returning list of predictions instead")
                return results
        else:
            return results

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
        """Shut down the engine and release resources"""
        self.logger.info("Shutting down ML Training Engine...")
        
        # Shutdown inference engine
        if hasattr(self, 'inference_engine'):
            self.inference_engine.shutdown()
            
        # Stop batch processor
        if hasattr(self, 'batch_processor'):
            self.batch_processor.stop()
            
        # Clean up resources
        if self.config.memory_optimization:
            self.logger.info("Running garbage collection...")
            gc.collect()
            
        self.logger.info("ML Training Engine shut down successfully")
