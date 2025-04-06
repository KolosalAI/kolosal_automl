import numpy as np
import os
import pickle
import joblib
import time
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
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

# Import configuration classes
from modules.configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig,
    PreprocessorConfig,
    BatchProcessorConfig,
    InferenceEngineConfig,
    NormalizationType,
    ModelSelectionCriteria,
    MonitoringConfig,
    ExplainabilityConfig
)

# Import engine components
from modules.engine.inference_engine import InferenceEngine
from modules.engine.batch_processor import BatchProcessor
from modules.engine.data_preprocessor import DataPreprocessor
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
            report += f"- **Type**: {type(model_data.get('model', '').__name__)}\n"
            
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
            self.logger.info(f"Experiment tracking enabled using {config.experiment_tracking_platform}")
            
            # Configure experiment tracking based on platform
            if config.experiment_tracking_platform == "mlflow":
                try:
                    import mlflow
                    if 'tracking_uri' in config.experiment_tracking_config:
                        mlflow.set_tracking_uri(config.experiment_tracking_config['tracking_uri'])
                    self.logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
                except ImportError:
                    self.logger.warning("MLflow not installed. Falling back to local experiment tracking.")
                except Exception as e:
                    self.logger.warning(f"Failed to configure MLflow: {str(e)}. Using local experiment tracking.")
        else:
            self.tracker = None
            
        # Create model directory if it doesn't exist
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
            
        # Create checkpoint directory if needed
        if config.checkpointing and not os.path.exists(config.checkpoint_path):
            os.makedirs(config.checkpoint_path)
            
        # Initialize components
        self._init_components()
        
        # Log initialization
        self.logger.info(f"ML Training Engine initialized with task type: {config.task_type.value}")
        if config.use_gpu:
            self.logger.info("GPU usage enabled with memory fraction: " 
                           f"{config.gpu_memory_fraction}")
            
        # Register model types for automatic discovery
        self._register_model_types()
        
    def _register_model_types(self):
        """Register built-in model types for auto discovery"""
        self._model_registry = {}
        
        # Register common sklearn model types
        try:
            from sklearn.ensemble import (
                RandomForestClassifier, RandomForestRegressor,
                GradientBoostingClassifier, GradientBoostingRegressor,
                AdaBoostClassifier, AdaBoostRegressor
            )
            from sklearn.linear_model import (
                LogisticRegression, LinearRegression,
                Ridge, Lasso, ElasticNet
            )
            from sklearn.svm import SVC, SVR
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            
            # Classification models
            self._model_registry["classification"] = {
                "random_forest": RandomForestClassifier,
                "gradient_boosting": GradientBoostingClassifier,
                "logistic_regression": LogisticRegression,
                "svm": SVC,
                "knn": KNeighborsClassifier,
                "decision_tree": DecisionTreeClassifier,
                "adaboost": AdaBoostClassifier
            }
            
            # Regression models
            self._model_registry["regression"] = {
                "random_forest": RandomForestRegressor,
                "gradient_boosting": GradientBoostingRegressor,
                "linear_regression": LinearRegression,
                "ridge": Ridge,
                "lasso": Lasso,
                "elastic_net": ElasticNet,
                "svr": SVR,
                "knn": KNeighborsRegressor,
                "decision_tree": DecisionTreeRegressor,
                "adaboost": AdaBoostRegressor
            }
            
            # Try to register other model types if available
            try:
                import xgboost as xgb
                self._model_registry["classification"]["xgboost"] = xgb.XGBClassifier
                self._model_registry["regression"]["xgboost"] = xgb.XGBRegressor
            except ImportError:
                self.logger.debug("XGBoost not available")
                
            try:
                import lightgbm as lgb
                self._model_registry["classification"]["lightgbm"] = lgb.LGBMClassifier
                self._model_registry["regression"]["lightgbm"] = lgb.LGBMRegressor
            except ImportError:
                self.logger.debug("LightGBM not available")
                
            try:
                import catboost as cb
                self._model_registry["classification"]["catboost"] = cb.CatBoostClassifier
                self._model_registry["regression"]["catboost"] = cb.CatBoostRegressor
            except ImportError:
                self.logger.debug("CatBoost not available")
                
        except ImportError as e:
            self.logger.warning(f"Failed to register model types: {str(e)}")
            
        self.logger.debug(f"Registered {len(self._model_registry.get('classification', {}))} classification models and "
                        f"{len(self._model_registry.get('regression', {}))} regression models")
        
    def _init_components(self):
        """Initialize all engine components"""
        # Initialize preprocessor with the provided configuration
        self.preprocessor = DataPreprocessor(self.config.preprocessing_config)
        
        # Initialize batch processor for efficient data handling
        self.batch_processor = BatchProcessor(self.config.batch_processing_config)
        
        # Initialize inference engine for prediction serving
        self.inference_engine = InferenceEngine(self.config.inference_config)
        
        # Initialize monitoring if enabled
        if hasattr(self.config, 'monitoring_config') and self.config.monitoring_config.enable_monitoring:
            self.logger.info("Initializing model monitoring")
            # This would connect to your monitoring implementation
            
        # Initialize explainability tools if enabled
        if hasattr(self.config, 'explainability_config') and self.config.explainability_config.enable_explainability:
            self.logger.info(f"Initializing explainability with method: {self.config.explainability_config.default_method}")
            # Setup explainability tools based on config
            
        self.logger.info("All components initialized successfully")
        
    def _get_feature_selector(self, X, y):
        """Get appropriate feature selector based on configuration"""
        if not self.config.feature_selection:
            return None
        
        # For classification tasks
        if self.config.task_type == TaskType.CLASSIFICATION:
            if self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_classif, k=self.config.feature_selection_k)
            elif self.config.feature_selection_method == "chi2":
                from sklearn.feature_selection import chi2
                # Ensure non-negative values for chi2
                if isinstance(X, np.ndarray) and np.any(X < 0):
                    self.logger.warning("Chi2 requires non-negative values. Using f_classif instead.")
                    selector = SelectKBest(f_classif, k=self.config.feature_selection_k)
                else:
                    selector = SelectKBest(chi2, k=self.config.feature_selection_k)
            elif self.config.feature_selection_method == "rfe":
                # Recursive feature elimination requires a base estimator
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestClassifier
                base_estimator = RandomForestClassifier(n_estimators=10, random_state=self.config.random_state)
                selector = RFE(base_estimator, n_features_to_select=self.config.feature_selection_k)
            else:
                # Default to f_classif
                selector = SelectKBest(f_classif, k=self.config.feature_selection_k)
        
        # For regression tasks
        elif self.config.task_type == TaskType.REGRESSION:
            from sklearn.feature_selection import mutual_info_regression, f_regression
            if self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_regression, k=self.config.feature_selection_k)
            elif self.config.feature_selection_method == "rfe":
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestRegressor
                base_estimator = RandomForestRegressor(n_estimators=10, random_state=self.config.random_state)
                selector = RFE(base_estimator, n_features_to_select=self.config.feature_selection_k)
            else:
                selector = SelectKBest(f_regression, k=self.config.feature_selection_k)
        
        # For other task types, default to mutual information
        else:
            from sklearn.feature_selection import mutual_info_regression
            selector = SelectKBest(mutual_info_regression, k=self.config.feature_selection_k)
                
        # If k is not specified, select based on percentile or threshold
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
        """Get appropriate scoring metric based on task type and configuration"""
        # If a specific optimization metric is set in the config, use it
        if hasattr(self.config, 'optimization_metric') and self.config.optimization_metric:
            if isinstance(self.config.optimization_metric, ModelSelectionCriteria):
                metric_name = self.config.optimization_metric.value
            else:
                metric_name = self.config.optimization_metric
                
            # Map metric names to sklearn scoring strings
            metric_mapping = {
                'accuracy': 'accuracy',
                'f1': 'f1_weighted',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted',
                'roc_auc': 'roc_auc',
                'matthews_correlation': 'matthews_corrcoef',
                'rmse': 'neg_root_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'r2': 'r2',
                'explained_variance': 'explained_variance',
                'silhouette': 'silhouette'
            }
            
            if metric_name.lower() in metric_mapping:
                return metric_mapping[metric_name.lower()]
            else:
                self.logger.warning(f"Unrecognized metric: {metric_name}. Using default for task type.")
        
        # Default metrics based on task type
        if self.config.task_type == TaskType.CLASSIFICATION:
            if hasattr(self.config, 'model_selection_criteria'):
                if self.config.model_selection_criteria == ModelSelectionCriteria.F1:
                    return "f1_weighted"
                elif self.config.model_selection_criteria == ModelSelectionCriteria.PRECISION:
                    return "precision_weighted"
                elif self.config.model_selection_criteria == ModelSelectionCriteria.RECALL:
                    return "recall_weighted"
                elif self.config.model_selection_criteria == ModelSelectionCriteria.ROC_AUC:
                    return "roc_auc"
            return "f1_weighted"  # Default for classification
            
        elif self.config.task_type == TaskType.REGRESSION:
            if hasattr(self.config, 'model_selection_criteria'):
                if self.config.model_selection_criteria == ModelSelectionCriteria.MEAN_ABSOLUTE_ERROR:
                    return "neg_mean_absolute_error"
                elif self.config.model_selection_criteria == ModelSelectionCriteria.R2:
                    return "r2"
                elif self.config.model_selection_criteria == ModelSelectionCriteria.EXPLAINED_VARIANCE:
                    return "explained_variance"
            return "neg_mean_squared_error"  # Default for regression
            
        elif self.config.task_type == TaskType.CLUSTERING:
            return "silhouette"
            
        elif self.config.task_type == TaskType.TIME_SERIES:
            # For time series, typically use MSE or MAE
            return "neg_mean_squared_error"
            
        # Default fallback
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