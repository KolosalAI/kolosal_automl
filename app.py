import pandas as pd
import numpy as np
import json
import os
import time
import traceback
import hmac
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import requests
from io import StringIO
import argparse
import sys
import psutil
from pathlib import Path
import gradio as gr

# Import matplotlib with proper backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# Import optimization integration
try:
    from modules.engine.optimization_integration import (
        OptimizedDataPipeline,
        quick_load_optimized,
        quick_optimize_memory,
        get_optimization_status,
        create_optimized_training_pipeline
    )
    OPTIMIZATION_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Optimization integration not available: {e}")
    OPTIMIZATION_INTEGRATION_AVAILABLE = False

# Import logging configuration
try:
    from modules.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# Import the configuration classes
from modules.configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig,
    PreprocessorConfig,
    NormalizationType,
    BatchProcessorConfig,
    BatchProcessingStrategy,
    InferenceEngineConfig,
    QuantizationConfig,
    QuantizationType,
    QuantizationMode,
    OptimizationMode,
)
from modules.engine.train_engine import MLTrainingEngine

# Import enhanced security components
try:
    from modules.security import (
        SecurityEnvironment, SecurityConfig, EnhancedSecurityManager,
        TLSManager, SecretsManager, generate_secure_api_key,
        generate_jwt_secret, validate_password_strength
    )
    from modules.security.enhanced_security import (
        EnhancedSecurityConfig, DEFAULT_ENHANCED_SECURITY_CONFIG
    )
    from modules.security.security_config import get_security_environment
    
    # Initialize security
    SECURITY_ENV = get_security_environment()
    SECURITY_MANAGER = EnhancedSecurityManager()
except ImportError as e:
    logger.warning(f"Security components not available: {e}")
    SECURITY_ENV = None
    SECURITY_MANAGER = None

# Import other required modules
try:
    from modules.engine.inference_engine import InferenceEngine, InferenceServer
    from modules.ui.data_preview_generator import DataPreviewGenerator
    from modules.ui.sample_data_loader import SampleDataLoader
    from modules.model.model_manager import SecureModelManager
    from modules.security.security_utils import (
        security_wrapper, 
        validate_file_upload,
        secure_data_processing,
        create_security_headers,
        get_auth_config
    )
    from modules.device_optimizer import DeviceOptimizer
    
    # Import advanced features
    from modules.engine.experiment_tracker import ExperimentTracker
    from modules.engine.batch_processor import BatchProcessor
    from modules.engine.adaptive_hyperopt import (
        AdaptiveHyperparameterOptimizer, 
        OptimizationResult,
        SearchSpaceConfig
    )
    from modules.engine.adaptive_preprocessing import (
        PreprocessorConfigOptimizer,
        ProcessingMode,
        PreprocessingStrategy
    )
    from modules.engine.memory_aware_processor import (
        MemoryAwareDataProcessor,
        create_memory_aware_processor
    )
    from modules.api.dashboard import generate_dashboard_html
    
    # Performance monitoring
    try:
        from modules.engine.performance_metrics import PerformanceTracker
        PERFORMANCE_TRACKING_AVAILABLE = True
    except ImportError:
        PERFORMANCE_TRACKING_AVAILABLE = False
    
    ADVANCED_FEATURES_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False
    
    # Create placeholder classes/functions
    class InferenceServer:
        def __init__(self):
            self.is_loaded = False
        def load_model_from_path(self, path, password=None): 
            return "❌ InferenceServer not available"
        def predict(self, data): 
            return "❌ InferenceServer not available"
        def get_model_info(self): 
            return {"error": "Not available"}
    
    class DataPreviewGenerator:
        def generate_data_summary(self, df): 
            return {"shape": df.shape, "columns": df.columns.tolist()}
        def format_data_preview(self, df, summary): 
            return f"Data shape: {df.shape}<br>Columns: {', '.join(df.columns.tolist())}"
    
    class SampleDataLoader:
        def load_sample_data(self, name): 
            if name == "Select a dataset...":
                return pd.DataFrame(), {"name": name, "description": "No dataset selected"}
            # Create a simple dummy dataset
            df = pd.DataFrame({
                'feature_1': [1, 2, 3, 4, 5],
                'feature_2': [2, 4, 6, 8, 10], 
                'target': [0, 1, 0, 1, 0]
            })
            return df, {
                "name": name, 
                "description": "Sample dataset (modules not available)", 
                "task_type": "classification", 
                "target_column": "target"
            }
    
    class SecureModelManager:
        def __init__(self, *args, **kwargs): pass
        
    class DeviceOptimizer:
        def get_system_info(self): 
            return {"error": "Device optimizer not available"}
    
    class ExperimentTracker:
        def __init__(self, *args, **kwargs): pass
        def start_experiment(self, *args, **kwargs): return {}
        def log_metrics(self, *args, **kwargs): pass
        def end_experiment(self, *args, **kwargs): pass
    
    class BatchProcessor:
        def __init__(self, *args, **kwargs): pass
        
    class AdaptiveHyperparameterOptimizer:
        def __init__(self, *args, **kwargs): pass
        def optimize(self, *args, **kwargs): return {}
    
    class PreprocessorConfigOptimizer:
        def __init__(self, *args, **kwargs): pass
        def optimize_config(self, *args, **kwargs): return {}
    
    class MemoryAwareDataProcessor:
        def __init__(self, *args, **kwargs): pass
        def process_data(self, *args, **kwargs): return {}
    
    class PerformanceTracker:
        def __init__(self, *args, **kwargs): pass
        def get_metrics(self, *args, **kwargs): return {}
    
    def security_wrapper(func): return func
    def validate_file_upload(path, extensions): return True
    def secure_data_processing(data): return data
    def create_security_headers(): return {}
    def get_auth_config(): return None
    def generate_dashboard_html(data): return "<h1>Dashboard not available</h1>"
    def create_memory_aware_processor(): return MemoryAwareDataProcessor()
    
    # Create simple search space function
    def create_search_space(task_type, algorithms): 
        return {alg: {"n_estimators": [10, 50, 100]} for alg in algorithms}
    
    ProcessingMode = type('ProcessingMode', (), {'BALANCED': 'balanced'})
    PreprocessingStrategy = type('PreprocessingStrategy', (), {'STANDARD': 'standard'})
    PERFORMANCE_TRACKING_AVAILABLE = False

# Additional function to create proper search spaces
def create_search_space(task_type: str, algorithms: List[str]) -> Dict[str, Dict[str, Any]]:
    """Create hyperparameter search spaces for different algorithms"""
    search_spaces = {}
    
    for algorithm in algorithms:
        if algorithm in ['random_forest', 'extra_trees']:
            search_spaces[algorithm] = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        elif algorithm in ['xgboost', 'lightgbm', 'catboost']:
            search_spaces[algorithm] = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        elif algorithm in ['gradient_boosting', 'adaboost']:
            search_spaces[algorithm] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9]
            }
        elif algorithm in ['svm', 'svm_linear', 'svm_poly']:
            if task_type == 'classification':
                search_spaces[algorithm] = {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
                }
            else:
                search_spaces[algorithm] = {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'epsilon': [0.01, 0.1, 0.2, 0.5],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
                }
        elif algorithm in ['logistic_regression', 'linear_regression', 'ridge', 'lasso']:
            search_spaces[algorithm] = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0] if algorithm in ['logistic_regression', 'ridge'] else [0.1, 1.0, 10.0],
                'alpha': [0.01, 0.1, 1.0, 10.0] if algorithm in ['ridge', 'lasso'] else None,
                'max_iter': [1000, 2000, 5000]
            }
        elif algorithm == 'elastic_net':
            search_spaces[algorithm] = {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'max_iter': [1000, 2000, 5000]
            }
        elif algorithm in ['mlp', 'neural_network']:
            search_spaces[algorithm] = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000, 2000]
            }
        elif algorithm == 'knn':
            search_spaces[algorithm] = {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        elif algorithm in ['naive_bayes', 'multinomial_nb', 'bernoulli_nb']:
            search_spaces[algorithm] = {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0] if algorithm != 'naive_bayes' else None,
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6] if algorithm == 'naive_bayes' else None
            }
        elif algorithm == 'decision_tree':
            search_spaces[algorithm] = {
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy'] if task_type == 'classification' else ['squared_error', 'absolute_error']
            }
        else:
            # Default search space
            search_spaces[algorithm] = {
                'n_estimators': [50, 100, 200] if 'estimators' in algorithm else None,
                'max_depth': [5, 10, None] if 'tree' in algorithm or 'forest' in algorithm else None
            }
        
        # Remove None values from search space
        search_spaces[algorithm] = {k: v for k, v in search_spaces[algorithm].items() if v is not None}
    
    return search_spaces

# Additional imports

def load_css_file():
    """Load CSS file for styling"""
    try:
        css_path = Path(__file__).parent / "static" / "styles.css"
        if css_path.exists():
            return css_path.read_text()
    except Exception:
        pass
    
    # Default CSS
    return """
    /* Default styling */
    .gr-button-primary { background-color: #007bff !important; }
    .gr-box { border-radius: 8px !important; }
    """

class MLSystemUI:
    """Enhanced Gradio UI for the ML Training & Inference System with Advanced Features"""
    
    def __init__(self, inference_only: bool = False):
        self.inference_only = inference_only
        self.training_engine = None
        self.inference_engine = None
        self.device_optimizer = None
        self.model_manager = None
        self.inference_server = InferenceServer()
        self.current_data = None
        self.current_config = None
        self.sample_data_loader = SampleDataLoader()
        self.data_preview_generator = DataPreviewGenerator()
        self.trained_models = {}  # Store trained models
        
        # Advanced features
        self.experiment_tracker = None
        self.batch_processor = None
        self.hpo_optimizer = None
        self.preprocessing_optimizer = None
        self.memory_processor = None
        self.performance_tracker = None
        self.model_comparison_results = {}
        
        # Security integration
        self.security_manager = SECURITY_MANAGER
        self.security_env = SECURITY_ENV
        
        # Initialize optimization pipeline
        self.optimization_pipeline = None
        if OPTIMIZATION_INTEGRATION_AVAILABLE:
            try:
                self.optimization_pipeline = create_optimized_training_pipeline(max_memory_pct=75.0)
                logger.info("🚀 Optimization pipeline initialized")
                if self.security_manager and hasattr(self.security_manager, 'auditor'):
                    self.security_manager.auditor.logger.info("OPTIMIZATION_INIT: Advanced optimization system enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize optimization pipeline: {e}")
                self.optimization_pipeline = None
        
        # Initialize advanced features if available
        if ADVANCED_FEATURES_AVAILABLE and not inference_only:
            try:
                self._initialize_advanced_features()
            except Exception as e:
                logger.warning(f"Failed to initialize advanced features: {e}")
        
        # Initialize security audit logger
        logger.info("🛡️ Initializing ML System UI with enhanced security")
        if self.security_manager and hasattr(self.security_manager, 'auditor'):
            self.security_manager.auditor.logger.info("SYSTEM_INIT: ML System UI initialized")
        
        # Define available ML algorithms with their categories and correct keys
        self.ml_algorithms = {
            "Tree-Based": {
                "Random Forest": {"key": "random_forest", "supports": ["classification", "regression"]},
                "Extra Trees": {"key": "extra_trees", "supports": ["classification", "regression"]},
                "Decision Tree": {"key": "decision_tree", "supports": ["classification", "regression"]},
                "Gradient Boosting": {"key": "gradient_boosting", "supports": ["classification", "regression"]},
            },
            "Boosting": {
                "XGBoost": {"key": "xgboost", "supports": ["classification", "regression"]},
                "LightGBM": {"key": "lightgbm", "supports": ["classification", "regression"]},
                "CatBoost": {"key": "catboost", "supports": ["classification", "regression"]},
                "AdaBoost": {"key": "adaboost", "supports": ["classification", "regression"]},
            },
            "Linear Models": {
                "Logistic Regression": {"key": "logistic_regression", "supports": ["classification"]},
                "Linear Regression": {"key": "linear_regression", "supports": ["regression"]},
                "Ridge": {"key": "ridge", "supports": ["classification", "regression"]},
                "Lasso": {"key": "lasso", "supports": ["classification", "regression"]},
                "Elastic Net": {"key": "elastic_net", "supports": ["classification", "regression"]},
                "SGD": {"key": "sgd", "supports": ["classification", "regression"]},
            },
            "Support Vector Machines": {
                "SVM": {"key": "svm", "supports": ["classification", "regression"]},
                "SVM (Linear)": {"key": "svm_linear", "supports": ["classification", "regression"]},
                "SVM (Polynomial)": {"key": "svm_poly", "supports": ["classification", "regression"]},
            },
            "Neural Networks": {
                "Multi-layer Perceptron": {"key": "mlp", "supports": ["classification", "regression"]},
                "Neural Network": {"key": "neural_network", "supports": ["classification", "regression"]},
            },
            "Naive Bayes": {
                "Gaussian NB": {"key": "naive_bayes", "supports": ["classification"]},
                "Multinomial NB": {"key": "multinomial_nb", "supports": ["classification"]},
                "Bernoulli NB": {"key": "bernoulli_nb", "supports": ["classification"]},
            },
            "Nearest Neighbors": {
                "K-Nearest Neighbors": {"key": "knn", "supports": ["classification", "regression"]},
            },
            "Ensemble Methods": {
                "Voting Classifier": {"key": "voting", "supports": ["classification"]},
                "Stacking": {"key": "stacking", "supports": ["classification", "regression"]},
            }
        }
        
        # Initialize device optimizer for system info
        if not inference_only:
            try:
                self.device_optimizer = DeviceOptimizer()
                logger.info("Device optimizer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize device optimizer: {e}")
                self.device_optimizer = None
    
    def _initialize_advanced_features(self):
        """Initialize advanced features"""
        try:
            # Experiment tracker
            self.experiment_tracker = ExperimentTracker(
                output_dir="./experiments",
                experiment_name=f"automl_session_{int(time.time())}"
            )
            logger.info("✅ Experiment tracker initialized")
            
            # Batch processor
            batch_config = BatchProcessorConfig(
                initial_batch_size=100,
                adaptive_batching=True,
                enable_monitoring=True,
                enable_memory_optimization=True
            )
            self.batch_processor = BatchProcessor(batch_config)
            logger.info("✅ Batch processor initialized")
            
            # Hyperparameter optimizer
            if ADVANCED_FEATURES_AVAILABLE:
                self.hpo_optimizer = AdaptiveHyperparameterOptimizer(
                    n_trials=50,
                    timeout=300
                )
                logger.info("✅ Hyperparameter optimizer initialized")
            
            # Preprocessing optimizer
            self.preprocessing_optimizer = PreprocessorConfigOptimizer()
            logger.info("✅ Preprocessing optimizer initialized")
            
            # Memory-aware processor
            self.memory_processor = create_memory_aware_processor()
            logger.info("✅ Memory-aware processor initialized")
            
            # Performance tracker
            if ADVANCED_FEATURES_AVAILABLE and PERFORMANCE_TRACKING_AVAILABLE:
                self.performance_tracker = PerformanceTracker()
                logger.info("✅ Performance tracker initialized")
            
        except Exception as e:
            logger.error(f"Error initializing advanced features: {e}")
            raise
    
    def get_algorithms_for_task(self, task_type: str) -> List[str]:
        """Get available algorithms for a specific task type"""
        algorithms = []
        task_lower = task_type.lower()
        
        for category, models in self.ml_algorithms.items():
            for model_name, model_info in models.items():
                if task_lower in model_info["supports"]:
                    algorithms.append(f"{category} - {model_name}")
        
        return sorted(algorithms)
    
    def get_model_key_from_name(self, algorithm_name: str) -> str:
        """Extract model key from formatted algorithm name with fallback mapping"""
        if " - " in algorithm_name:
            category, model_name = algorithm_name.split(" - ", 1)
            for cat, models in self.ml_algorithms.items():
                if cat == category and model_name in models:
                    return models[model_name]["key"]
        
        # Fallback mapping for common algorithm names
        fallback_mapping = {
            "random forest": "random_forest",
            "decision tree": "decision_tree",
            "gradient boosting": "gradient_boosting",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "adaboost": "adaboost",
            "logistic regression": "logistic_regression",
            "linear regression": "linear_regression",
            "svm": "svm",
            "k-nearest neighbors": "knn",
            "knn": "knn",
            "naive bayes": "naive_bayes",
            "gaussian nb": "naive_bayes",
            "multinomial nb": "multinomial_nb",
            "multi-layer perceptron": "mlp",
            "mlp": "mlp",
            "voting classifier": "voting",
            "stacking": "stacking",
            "sgd": "sgd"
        }
        
        # Try fallback mapping
        algorithm_lower = algorithm_name.lower()
        if algorithm_lower in fallback_mapping:
            return fallback_mapping[algorithm_lower]
        
        # Extract just the model name if it has " - " format
        if " - " in algorithm_name:
            model_name = algorithm_name.split(" - ", 1)[1].lower()
            if model_name in fallback_mapping:
                return fallback_mapping[model_name]
        
        # Default fallback
        return algorithm_name.lower().replace(" ", "_")

    def get_all_algorithms(self) -> List[str]:
        """Return all algorithm display names without task filtering."""
        algorithms: List[str] = []
        for category, models in self.ml_algorithms.items():
            for model_name in models.keys():
                algorithms.append(f"{category} - {model_name}")
        return sorted(algorithms)

    def get_multi_algorithm_choices(self, prefilter_by_task: bool = True):
        """Return a Gradio update object for algorithm choices using engine registry when possible."""
        try:
            choices: List[str] = []
            if prefilter_by_task and self.current_config is not None:
                # Prefer engine registry for accurate availability
                try:
                    temp_engine = MLTrainingEngine(self.current_config)
                    task_key = self.current_config.task_type.value
                    registry = getattr(temp_engine, '_model_registry', {})
                    choices = sorted(list(registry.get(task_key, {}).keys()))
                except Exception as ie:
                    logger.debug(f"Engine registry unavailable, fallback list used: {ie}")
                    choices = self.get_algorithms_for_task(self.current_config.task_type.value)
            elif prefilter_by_task and self.current_config is None:
                # Fallback to a reasonable default (classification)
                choices = self.get_algorithms_for_task('classification')
            else:
                # All algorithms (union of both tasks if available)
                try:
                    # Use engine registry if a config exists
                    if self.current_config is not None:
                        temp_engine = MLTrainingEngine(self.current_config)
                        registry = getattr(temp_engine, '_model_registry', {})
                        union = set(registry.get('classification', {}).keys()) | set(registry.get('regression', {}).keys())
                        choices = sorted(list(union))
                    else:
                        choices = self.get_all_algorithms()
                except Exception:
                    choices = self.get_all_algorithms()
            return gr.update(choices=choices, value=[])
        except Exception as e:
            logger.warning(f"Failed to get multi algorithm choices: {e}")
            return gr.update()
    
    def get_trained_model_list(self) -> List[str]:
        """Get list of trained models"""
        model_list = ["Select a trained model..."]
        if self.trained_models:
            for model_name in self.trained_models.keys():
                model_list.append(model_name)
        return model_list
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            if self.device_optimizer:
                info = self.device_optimizer.get_system_info()
                return info
            return {"error": "Device optimizer not available"}
        except Exception as e:
            return {"error": f"Error getting system info: {str(e)}"}
    
    def determine_task_type(self, y: pd.Series) -> str:
        """Determine if the task is classification or regression based on target variable"""
        try:
            # Check if target is numeric and has many unique values (likely regression)
            if pd.api.types.is_numeric_dtype(y):
                unique_ratio = y.nunique() / len(y)
                if unique_ratio > 0.05:  # If more than 5% unique values, likely regression
                    return "regression"
                else:
                    return "classification"
            else:
                # Non-numeric targets are typically classification
                return "classification"
        except Exception:
            # Default to classification if unable to determine
            return "classification"
    
    def load_sample_data(self, dataset_name: str) -> Tuple[str, Dict, str, str]:
        """Load sample dataset with preview"""
        try:
            if dataset_name == "Select a dataset...":
                return "Please select a dataset", {}, "", ""
            
            df, metadata = self.sample_data_loader.load_sample_data(dataset_name)
            self.current_data = df
            
            # Generate data summary and preview
            summary = self.data_preview_generator.generate_data_summary(df)
            preview_text = self.data_preview_generator.format_data_preview(df, summary)
            
            # Generate sample data table
            sample_table = df.head(10).to_html(classes="table table-striped", escape=False, border=0)
            
            info_text = f"""
Sample Dataset Loaded: {metadata['name']}

- Description: {metadata['description']}
- Task Type: {metadata['task_type']}
- Target Column: {metadata['target_column']}
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
- Missing Values: {df.isnull().sum().sum()} total
            """
            
            return info_text, metadata, preview_text, sample_table
            
        except Exception as e:
            error_msg = f"Error loading sample data: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}, "", ""

    @security_wrapper
    def load_data(self, file) -> Tuple[str, Dict, str, str]:
        """Load dataset from uploaded file with enhanced security validation"""
        try:
            if file is None:
                return "No file uploaded", {}, "", ""
            
            file_path = file.name
            
            # Security validation
            if not validate_file_upload(file_path, ['.csv', '.xlsx', '.xls', '.json', '.parquet']):
                if self.security_manager and hasattr(self.security_manager, 'auditor'):
                    self.security_manager.auditor.logger.warning(f"FILE_UPLOAD_REJECTED: {file_path}")
                return "❌ File upload rejected for security reasons. Please ensure you're uploading a valid data file.", {}, "", ""
            
            # Log file upload
            if self.security_manager and hasattr(self.security_manager, 'auditor'):
                self.security_manager.auditor.logger.info(f"FILE_UPLOAD: {os.path.basename(file_path)}")
            
            # Load data based on file extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return "Unsupported file format. Please upload CSV, Excel, or JSON files.", {}, "", ""
            
            # Security validation based on dataset size
            if df.shape[0] > 1000000:  # 1M rows limit
                if self.security_manager and hasattr(self.security_manager, 'auditor'):
                    self.security_manager.auditor.logger.warning(f"LARGE_DATASET: {df.shape}")
                return "⚠️ Dataset too large. Please use a dataset with fewer than 1 million rows.", {}, "", ""
            
            if df.shape[1] > 10000:
                if self.security_manager and hasattr(self.security_manager, 'auditor'):
                    self.security_manager.auditor.logger.warning(f"WIDE_DATASET: {df.shape}")
                return "⚠️ Dataset too wide. Please use a dataset with fewer than 10,000 columns.", {}, "", ""
            
            self.current_data = df
            
            # Generate data summary and preview
            summary = self.data_preview_generator.generate_data_summary(df)
            preview_text = self.data_preview_generator.format_data_preview(df, summary)
            
            # Generate sample data table
            sample_table = df.head(10).to_html(classes="table table-striped", escape=False, border=0)
            
            info_text = f"""
Data Loaded Successfully! ✅

📊 Dataset Overview:
- Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns
- Columns: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
- Missing Values: {df.isnull().sum().sum():,} total
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            """
            
            return info_text, summary, preview_text, sample_table
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}, "", ""

    def optimize_preprocessing_config(self, data_characteristics: Dict[str, Any] = None) -> str:
        """Optimize preprocessing configuration based on data characteristics"""
        if not ADVANCED_FEATURES_AVAILABLE:
            return "⚠️ Advanced preprocessing optimization not available"
        
        try:
            if self.current_data is None:
                return "❌ Please load data first"
            
            if self.preprocessing_optimizer is None:
                return "❌ Preprocessing optimizer not initialized"
            
            # Use the available method from PreprocessorConfigOptimizer
            try:
                from modules.engine.adaptive_preprocessing import DatasetSize
                
                # Determine dataset size
                n_samples = len(self.current_data)
                if n_samples < 1000:
                    dataset_size = DatasetSize.SMALL
                elif n_samples < 100000:
                    dataset_size = DatasetSize.MEDIUM
                else:
                    dataset_size = DatasetSize.LARGE
                
                # Estimate memory usage
                memory_mb = self.current_data.memory_usage(deep=True).sum() / 1024**2
                
                # Optimize preprocessing configuration using the correct method
                optimized_config = self.preprocessing_optimizer.optimize_for_dataset(
                    dataset_size=dataset_size,
                    estimated_memory_mb=memory_mb,
                    num_features=len(self.current_data.columns),
                    processing_mode=ProcessingMode.BALANCED,
                    target_memory_pct=75.0
                )
            except ImportError:
                # Fallback if DatasetSize is not available
                optimized_config = self.preprocessing_optimizer.optimize_for_dataset(self.current_data)
            
            # Analyze data characteristics for display
            data_characteristics = self._analyze_data_characteristics(self.current_data)
            
            optimization_text = f"""
Preprocessing Configuration Optimized! ✅

📊 Data Analysis:
- Samples: {data_characteristics.get('n_samples', 0):,}
- Features: {data_characteristics.get('n_features', 0):,}
- Categorical Features: {data_characteristics.get('n_categorical', 0)}
- Numerical Features: {data_characteristics.get('n_numerical', 0)}
- Missing Data: {data_characteristics.get('missing_ratio', 0):.2%}
- Memory Usage: {data_characteristics.get('memory_usage_mb', 0):.2f} MB

⚙️ Optimized Configuration:
- Configuration Type: {type(optimized_config).__name__}
- Normalization: {getattr(optimized_config, 'normalization', 'Auto')}
- Batch Processing: Enabled
- Memory Optimization: Enabled
            """
            
            return optimization_text
            
        except Exception as e:
            error_msg = f"Error optimizing preprocessing: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _analyze_data_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics for preprocessing optimization"""
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            characteristics = {
                'n_samples': len(df),
                'n_features': len(df.columns),
                'n_categorical': len(categorical_cols),
                'n_numerical': len(numerical_cols),
                'missing_ratio': df.isnull().sum().sum() / (len(df) * len(df.columns)),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'sparsity_ratio': 0.0,  # Could be calculated for sparse data
                'outlier_ratio': 0.0,   # Could be calculated using statistical methods
                'skewness_avg': numerical_cols.empty and 0.0 or df[numerical_cols].skew().mean(),
                'cardinality_avg': categorical_cols.empty and 0.0 or df[categorical_cols].nunique().mean()
            }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing data characteristics: {e}")
            return {
                'n_samples': len(df) if df is not None else 0,
                'n_features': len(df.columns) if df is not None else 0,
                'n_categorical': 0,
                'n_numerical': 0,
                'missing_ratio': 0.0,
                'memory_usage_mb': 0.0,
                'sparsity_ratio': 0.0,
                'outlier_ratio': 0.0,
                'skewness_avg': 0.0,
                'cardinality_avg': 0.0
            }
    
    def compare_multiple_models(self, target_column: str, algorithms: List[str], 
                               enable_hpo: bool = True) -> Tuple[str, str, str]:
        """Train and compare multiple models with advanced features"""
        if self.inference_only:
            return "❌ Model comparison not available in inference-only mode.", "", ""
        
        if self.current_data is None:
            return "❌ Please load data first.", "", ""
        
        if target_column not in self.current_data.columns:
            return f"❌ Target column '{target_column}' not found.", "", ""
        
        try:
            # Start experiment tracking
            if self.experiment_tracker and ADVANCED_FEATURES_AVAILABLE:
                try:
                    cfg_dict = self.current_config.to_dict() if hasattr(self.current_config, 'to_dict') else (vars(self.current_config) if self.current_config else {})
                except Exception:
                    cfg_dict = {}
                model_info = {"model_type": "multi_model_comparison", "target": target_column, "models": algorithms}
                self.experiment_tracker.start_experiment(config=cfg_dict, model_info=model_info)
            
            # Prepare data with memory optimization
            X = self.current_data.drop(columns=[target_column])
            y = self.current_data[target_column]
            
            # Use memory-aware processing if available
            if self.memory_processor and ADVANCED_FEATURES_AVAILABLE:
                processed_data = self.memory_processor.process_data(X, y)
                X, y = processed_data['X'], processed_data['y']
            else:
                # Handle categorical features manually
                categorical_columns = X.select_dtypes(include=['object']).columns
                if len(categorical_columns) > 0:
                    for col in categorical_columns:
                        X[col] = pd.Categorical(X[col]).codes
            
            # Initialize training engine with advanced config
            if self.current_config is None:
                task_type = TaskType.CLASSIFICATION if self.determine_task_type(y) == "classification" else TaskType.REGRESSION
                self.current_config = MLTrainingEngineConfig(
                    task_type=task_type,
                    optimization_strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
                    model_path="./models",
                    checkpoint_path="./checkpoints",
                    enable_quantization=True,
                    enable_mixed_precision=True if ADVANCED_FEATURES_AVAILABLE else False
                )
            
            self.training_engine = MLTrainingEngine(self.current_config)
            
            results = {}
            comparison_data = []
            
            # Train each algorithm
            for algorithm in algorithms:
                try:
                    model_key = self.get_model_key_from_name(algorithm)
                    model_name = f"{algorithm}_{int(time.time())}"
                    
                    logger.info(f"Training {algorithm}...")
                    start_time = time.time()
                    
                    # Run hyperparameter optimization if enabled
                    if enable_hpo and self.hpo_optimizer and ADVANCED_FEATURES_AVAILABLE:
                        if hasattr(self.hpo_optimizer, 'optimize'):
                            # Create objective function
                            def objective_function(params):
                                try:
                                    from sklearn.model_selection import cross_val_score
                                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                                    from sklearn.linear_model import LogisticRegression, LinearRegression
                                    
                                    # Create model based on type
                                    if model_key == 'random_forest':
                                        if self.determine_task_type(y) == 'classification':
                                            model = RandomForestClassifier(**params, random_state=42)
                                        else:
                                            model = RandomForestRegressor(**params, random_state=42)
                                    else:
                                        # Default fallback
                                        if self.determine_task_type(y) == 'classification':
                                            model = RandomForestClassifier(**params, random_state=42)
                                        else:
                                            model = RandomForestRegressor(**params, random_state=42)
                                    
                                    # Quick CV
                                    cv_scores = cross_val_score(model, X.values if hasattr(X, 'values') else X, 
                                                              y.values if hasattr(y, 'values') else y, cv=3, 
                                                              scoring='accuracy' if self.determine_task_type(y) == 'classification' else 'r2')
                                    return cv_scores.mean()
                                except Exception as e:
                                    logger.warning(f"HPO objective error: {e}")
                                    return 0.0
                            
                            search_space = create_search_space(self.determine_task_type(y), [model_key])
                            hpo_result = self.hpo_optimizer.optimize(
                                objective_function=objective_function,
                                search_space=search_space,
                                direction='maximize',
                                study_name=f"comparison_{model_key}_{int(time.time())}"
                            )
                            best_params = getattr(hpo_result, 'best_params', search_space.get(model_key, {}))
                        else:
                            # Fallback HPO simulation
                            search_space = create_search_space(self.determine_task_type(y), [model_key])
                            best_params = search_space.get(model_key, {})
                    else:
                        best_params = {}
                    
                    # Train model; convert external best_params to a fixed param_grid for engine
                    fixed_grid = {k: ([v] if not isinstance(v, (list, tuple, np.ndarray)) else list(v)) for k, v in (best_params or {}).items()}
                    result = self.training_engine.train_model(
                        X=X.values if hasattr(X, 'values') else X,
                        y=y.values if hasattr(y, 'values') else y,
                        model_type=model_key,
                        model_name=model_name,
                        param_grid=fixed_grid if fixed_grid else None
                    )
                    
                    training_time = time.time() - start_time
                    
                    # Store results
                    results[algorithm] = {
                        'model_name': model_name,
                        'result': result,
                        'training_time': training_time,
                        'best_params': best_params,
                        'hpo_enabled': enable_hpo and bool(best_params)
                    }
                    
                    # Store in trained models
                    self.trained_models[model_name] = {
                        'algorithm': algorithm,
                        'model_key': model_key,
                        'target_column': target_column,
                        'training_time': training_time,
                        'result': result,
                        'feature_names': X.columns.tolist() if hasattr(X, 'columns') else [],
                        'data_shape': X.shape if hasattr(X, 'shape') else (0, 0),
                        'hpo_params': best_params
                    }
                    
                    # Log experiment metrics
                    if self.experiment_tracker and ADVANCED_FEATURES_AVAILABLE:
                        metrics = result.get('metrics', {})
                        # Use step to distinguish per-model metrics
                        self.experiment_tracker.log_metrics(metrics, step=model_name)
                    
                    logger.info(f"✅ {algorithm} training completed")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to train {algorithm}: {e}")
                    results[algorithm] = {
                        'error': str(e),
                        'training_time': 0
                    }
            
            # Generate comparison results
            comparison_summary = self._generate_model_comparison_summary(results)
            detailed_results = self._generate_detailed_comparison(results)
            visualization_data = self._generate_comparison_visualization(results)
            
            # End experiment
            if self.experiment_tracker and ADVANCED_FEATURES_AVAILABLE:
                self.experiment_tracker.end_experiment()
            
            return comparison_summary, detailed_results, visualization_data
            
        except Exception as e:
            error_msg = f"Error in model comparison: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", ""
    
    def _generate_model_comparison_summary(self, results: Dict[str, Any]) -> str:
        """Generate summary of model comparison results"""
        try:
            summary_lines = [
                "🏆 Model Comparison Results",
                "=" * 50,
                ""
            ]
            
            # Sort by primary metric (assuming classification uses f1, regression uses r2)
            model_scores = []
            for algorithm, result_data in results.items():
                if 'error' not in result_data:
                    result = result_data.get('result', {})
                    metrics = result.get('metrics', {})
                    primary_score = metrics.get('f1', metrics.get('r2', metrics.get('accuracy', 0)))
                    model_scores.append((algorithm, primary_score, result_data))
            
            # Sort by score descending
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (algorithm, score, result_data) in enumerate(model_scores, 1):
                status = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
                hpo_status = " (HPO)" if result_data.get('hpo_enabled', False) else ""
                summary_lines.extend([
                    f"{status} Rank {rank}: {algorithm}{hpo_status}",
                    f"   Score: {score:.4f}",
                    f"   Time: {result_data.get('training_time', 0):.2f}s",
                    ""
                ])
            
            # Add failed models
            failed_models = [alg for alg, res in results.items() if 'error' in res]
            if failed_models:
                summary_lines.extend([
                    "❌ Failed Models:",
                    *[f"   - {alg}: {results[alg]['error'][:50]}..." for alg in failed_models],
                    ""
                ])
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            return f"Error generating comparison summary: {str(e)}"
    
    def _generate_detailed_comparison(self, results: Dict[str, Any]) -> str:
        """Generate detailed comparison of model results"""
        try:
            detail_lines = [
                "📊 Detailed Model Analysis",
                "=" * 50,
                ""
            ]
            
            for algorithm, result_data in results.items():
                if 'error' not in result_data:
                    result = result_data.get('result', {})
                    metrics = result.get('metrics', {})
                    
                    detail_lines.extend([
                        f"🤖 {algorithm}",
                        "-" * (len(algorithm) + 3),
                        f"Training Time: {result_data.get('training_time', 0):.2f} seconds",
                        f"HPO Used: {'Yes' if result_data.get('hpo_enabled', False) else 'No'}",
                        ""
                    ])
                    
                    if metrics:
                        detail_lines.append("Metrics:")
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                detail_lines.append(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")
                            else:
                                detail_lines.append(f"  - {metric.replace('_', ' ').title()}: {value}")
                    
                    # Show best hyperparameters if HPO was used
                    best_params = result_data.get('best_params', {})
                    if best_params:
                        detail_lines.extend([
                            "",
                            "Best Hyperparameters:",
                            json.dumps(best_params, indent=4)
                        ])
                    
                    detail_lines.append("")
            
            return "\n".join(detail_lines)
            
        except Exception as e:
            return f"Error generating detailed comparison: {str(e)}"
    
    def _generate_comparison_visualization(self, results: Dict[str, Any]) -> str:
        """Generate visualization data for model comparison"""
        try:
            # This could generate HTML/plots for visualization
            # For now, return a simple table format
            
            viz_lines = [
                "📈 Performance Visualization",
                "=" * 50,
                "",
                "Algorithm | Score | Time(s) | HPO | Status",
                "-" * 50
            ]
            
            for algorithm, result_data in results.items():
                if 'error' not in result_data:
                    result = result_data.get('result', {})
                    metrics = result.get('metrics', {})
                    primary_score = metrics.get('f1', metrics.get('r2', metrics.get('accuracy', 0)))
                    training_time = result_data.get('training_time', 0)
                    hpo_used = result_data.get('hpo_enabled', False)
                    
                    viz_lines.append(
                        f"{algorithm:<15} | {primary_score:5.3f} | {training_time:6.2f} | {'✓' if hpo_used else '✗':<3} | ✅"
                    )
                else:
                    viz_lines.append(
                        f"{algorithm:<15} | {'N/A':<5} | {'N/A':<6} | {'✗':<3} | ❌"
                    )
            
            return "\n".join(viz_lines)
            
        except Exception as e:
            return f"Error generating visualization: {str(e)}"
    
    def get_system_monitoring_dashboard(self) -> str:
        """Get system monitoring dashboard"""
        try:
            if not ADVANCED_FEATURES_AVAILABLE:
                return "⚠️ Advanced monitoring not available"
            
            # Get system information
            system_info = self.get_system_info()
            
            # Get performance metrics if available
            performance_metrics = {}
            if self.performance_tracker:
                performance_metrics = self.performance_tracker.get_metrics()
            
            # Combine data for dashboard
            dashboard_data = {
                "current_metrics": {
                    "system_info": system_info,
                    "performance": performance_metrics,
                    "trained_models": len(self.trained_models),
                    "active_experiments": 1 if self.experiment_tracker else 0
                },
                "performance_analysis": {
                    "total_training_time": sum(
                        model.get('training_time', 0) 
                        for model in self.trained_models.values()
                    ),
                    "average_accuracy": np.mean([
                        model.get('result', {}).get('metrics', {}).get('accuracy', 0)
                        for model in self.trained_models.values()
                    ]) if self.trained_models else 0
                },
                "resource_analysis": system_info,
                "active_alerts": []
            }
            
            # Generate HTML dashboard
            dashboard_html = generate_dashboard_html(dashboard_data)
            
            return dashboard_html
            
        except Exception as e:
            error_msg = f"Error generating monitoring dashboard: {str(e)}"
            logger.error(error_msg)
            return f"<h1>Dashboard Error</h1><p>{error_msg}</p>"

    def update_algorithm_choices(self, task_type: str) -> gr.CheckboxGroup:
        """Update algorithm choices for given task using engine registry."""
        try:
            if self.current_config is None:
                # Create a minimal config for listing
                task_type_enum = TaskType[task_type.upper()]
                self.current_config = MLTrainingEngineConfig(
                    task_type=task_type_enum,
                    optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
                    model_path="./models",
                    checkpoint_path="./checkpoints"
                )
            temp_engine = MLTrainingEngine(self.current_config)
            choices = sorted(list(getattr(temp_engine, '_model_registry', {}).get(task_type, {})))
        except Exception:
            choices = self.get_algorithms_for_task(task_type)
        return gr.CheckboxGroup(choices=choices, value=[])

    def create_training_config(self, task_type: str, optimization_strategy: str, 
                             cv_folds: int, test_size: float, random_state: int,
                             enable_feature_selection: bool, normalization: str,
                             enable_quantization: bool, optimization_mode: str) -> Tuple[str]:
        """Create training configuration (algorithm selection handled separately)."""
        if self.inference_only:
            return "Training is not available in inference-only mode.", gr.Dropdown(), []
        
        try:
            # Map string values to enums
            task_type_enum = TaskType[task_type.upper()]
            
            # Map UI optimization strategy to actual OptimizationStrategy enum
            strategy_mapping = {
                "speed": OptimizationStrategy.RANDOM_SEARCH,
                "balanced": OptimizationStrategy.BAYESIAN_OPTIMIZATION,
                "accuracy": OptimizationStrategy.GRID_SEARCH
            }
            opt_strategy_enum = strategy_mapping.get(optimization_strategy, OptimizationStrategy.BAYESIAN_OPTIMIZATION)
            
            # Map UI optimization mode to OptimizationMode enum
            mode_mapping = {
                "speed": OptimizationMode.PERFORMANCE,
                "balanced": OptimizationMode.BALANCED,
                "accuracy": OptimizationMode.CONSERVATIVE
            }
            opt_mode_enum = mode_mapping.get(optimization_mode, OptimizationMode.BALANCED)
            
            # Map normalization (should work directly)
            norm_enum = NormalizationType[normalization.upper()]
            
            # Create configuration
            config = MLTrainingEngineConfig(
                task_type=task_type_enum,
                optimization_strategy=opt_strategy_enum,
                cv_folds=cv_folds,
                test_size=test_size,
                random_state=random_state,
                feature_selection=enable_feature_selection,
                enable_quantization=enable_quantization,
                model_path="./models",
                checkpoint_path="./checkpoints"
            )
            
            # Update preprocessing config
            if config.preprocessing_config:
                config.preprocessing_config.normalization = norm_enum
            
            # Update optimization mode in inference config
            if config.inference_config:
                config.inference_config.optimization_mode = opt_mode_enum
            
            self.current_config = config
            
            config_text = f"""
Configuration Created Successfully!

- Task Type: {task_type}
- Optimization Strategy: {optimization_strategy}
- CV Folds: {cv_folds}
- Test Size: {test_size}
- Feature Selection: {'Enabled' if enable_feature_selection else 'Disabled'}
- Normalization: {normalization}
- Quantization: {'Enabled' if enable_quantization else 'Disabled'}
✅ You can now select models in the Multi-Model Training and Comparison tabs.
            """
            
            return (config_text,)
            
        except Exception as e:
            error_msg = f"Error creating configuration: {str(e)}"
            logger.error(error_msg)
            return error_msg, gr.Dropdown(), []

    def train_model(self, target_column: str, algorithm_name: str, model_name: str = None, 
                   enable_hpo: bool = True, hpo_trials: int = 50) -> Tuple[str, str, str, gr.Dropdown]:
        """Train a single model using the unified core training path (merged with multi-model)."""
        if self.inference_only:
            return "❌ Model training is disabled in inference-only mode.", "", "", gr.Dropdown()
        
        if self.current_data is None:
            return "❌ Please load data first.", "", "", gr.Dropdown()
        
        if target_column not in self.current_data.columns:
            return f"❌ Target column '{target_column}' not found in data.", "", "", gr.Dropdown()
        
        try:
            # Use the unified multi-model path for a single algorithm
            results, progress, total_time = self._train_algorithms(
                target_column=target_column,
                algorithms=[algorithm_name],
                enable_hpo=enable_hpo,
                hpo_trials=hpo_trials,
                enable_parallel=False,
                max_workers=1,
                custom_hyperparams={},
                prefilter_by_task=True
            )

            # Extract the only result
            alg = list(results.keys())[0] if results else algorithm_name
            res_entry = results.get(alg, {})
            if res_entry.get('status') != 'success':
                err = res_entry.get('error', 'Unknown error')
                return f"❌ Error during training: {err}", "", "", gr.Dropdown()

            result = res_entry.get('result', {})
            model_name = res_entry.get('model_name', model_name or alg)

            # Build outputs similar to previous single-model format
            hpo_status = f" (HPO: {hpo_trials} trials)" if enable_hpo else " (No HPO)"
            metrics_text = f"Training Results{hpo_status}:\n\n"
            if 'metrics' in result and result['metrics']:
                for metric, value in result['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics_text += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
                    else:
                        metrics_text += f"- {metric.replace('_', ' ').title()}: {value}\n"
            metrics_text += f"\n- Training Time: {res_entry.get('training_time', total_time):.2f} seconds"
            if enable_hpo:
                metrics_text += f"\n- HPO Enabled: Yes ({hpo_trials} max trials)"
                if self.current_config is not None:
                    metrics_text += f"\n- Optimization Strategy: {self.current_config.optimization_strategy.value}"

            importance_text = ""
            if 'feature_importance' in result and result['feature_importance'] is not None:
                importance = result['feature_importance']
                feature_names = res_entry.get('feature_names', [])
                importance_text = "Top 10 Feature Importances:\n\n"
                if isinstance(importance, dict):
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    for feature, score in sorted_features:
                        importance_text += f"- {feature}: {score:.4f}\n"
                else:
                    indices = np.argsort(importance)[::-1][:10]
                    for i, idx in enumerate(indices):
                        if idx < len(feature_names):
                            importance_text += f"- {feature_names[idx]}: {importance[idx]:.4f}\n"

            summary_text = f"""
Model Training Summary

- Model Name: {model_name}
- Algorithm: {alg}
- Dataset Shape: {res_entry.get('data_shape', (0, 0))[0]} samples × {res_entry.get('data_shape', (0, 0))[1]} features
- Target Column: {target_column}
- Status: ✅ Training Completed Successfully
            """

            trained_models_dropdown = gr.Dropdown(
                choices=self.get_trained_model_list(),
                value="Select a trained model...",
                label="Trained Models"
            )

            return summary_text, metrics_text, importance_text, trained_models_dropdown
            
        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", "", gr.Dropdown()

    def _train_algorithms(self, target_column: str, algorithms: List[str],
                          enable_hpo: bool, hpo_trials: int,
                          enable_parallel: bool, max_workers: int,
                          custom_hyperparams: Dict[str, Any],
                          prefilter_by_task: bool) -> Tuple[Dict[str, Any], List[str], float]:
        """Unified core training routine for one or more algorithms.

        Returns: (results_dict, training_progress_lines, total_training_time)
        """
        if self.inference_only:
            return {}, ["❌ Model training is disabled in inference-only mode."], 0.0
        if self.current_data is None:
            return {}, ["❌ Please load data first."], 0.0
        if target_column not in self.current_data.columns:
            return {}, [f"❌ Target column '{target_column}' not found in data."], 0.0
        if not algorithms:
            return {}, ["❌ Please select at least one algorithm to train."], 0.0

        # Initialize training engine config if needed
        if self.current_config is None:
            task_type = TaskType.CLASSIFICATION if self.determine_task_type(self.current_data[target_column]) == "classification" else TaskType.REGRESSION
            self.current_config = MLTrainingEngineConfig(
                task_type=task_type,
                optimization_strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION if enable_hpo else OptimizationStrategy.GRID_SEARCH,
                model_path="./models",
                checkpoint_path="./checkpoints",
                cv_folds=5,
                test_size=0.2
            )

        # Prepare data
        X = self.current_data.drop(columns=[target_column])
        y = self.current_data[target_column]

        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                X[col] = pd.Categorical(X[col]).codes

        # Prefilter algorithms by task, if requested (ensures compatibility)
        training_progress: List[str] = []
        if prefilter_by_task:
            # Use engine registry keys to validate selections, supporting raw keys or display names
            try:
                temp_engine = MLTrainingEngine(self.current_config)
                task_key = self.current_config.task_type.value
                allowed_keys = set(getattr(temp_engine, '_model_registry', {}).get(task_key, {}).keys())
            except Exception:
                allowed_keys = set()
            before = len(algorithms)
            filtered = []
            for a in algorithms:
                key = self.get_model_key_from_name(a)
                if key in allowed_keys:
                    filtered.append(a)
            algorithms = filtered
            after = len(algorithms)
            if after == 0:
                return {}, ["❌ No compatible algorithms after task prefiltering."], 0.0
            if after < before:
                training_progress.append(f"🔎 Prefiltered algorithms by task: kept {after}/{before} compatible models")

        # Start experiment tracking (single or multi)
        if self.experiment_tracker and ADVANCED_FEATURES_AVAILABLE:
            try:
                cfg_dict = self.current_config.to_dict() if hasattr(self.current_config, 'to_dict') else (vars(self.current_config) if self.current_config else {})
            except Exception:
                cfg_dict = {}
            model_info = {"model_type": "multi_model_training", "target": target_column, "num_models": len(algorithms)}
            self.experiment_tracker.start_experiment(config=cfg_dict, model_info=model_info)
        else:
            pass

        start_time = time.time()
        results: Dict[str, Any] = {}

        # Per-model training function
        def train_single_model(algorithm_name):
            try:
                model_key = self.get_model_key_from_name(algorithm_name)
                timestamp = int(time.time() * 1000)
                model_name = f"{algorithm_name.split(' - ')[-1]}_{timestamp}"

                model_custom_params = custom_hyperparams.get(model_key, {}) if custom_hyperparams else {}

                search_space = create_search_space(self.current_config.task_type.value, [model_key])
                if model_custom_params:
                    if model_key in search_space:
                        search_space[model_key].update(model_custom_params)
                    else:
                        search_space[model_key] = model_custom_params

                model_start_time = time.time()
                training_engine = MLTrainingEngine(self.current_config)
                # Preserve reference for prediction convenience
                self.training_engine = training_engine

                # HPO path (best-effort)
                if enable_hpo and self.hpo_optimizer and ADVANCED_FEATURES_AVAILABLE:
                    try:
                        if hasattr(self.hpo_optimizer, 'optimize'):
                            def objective_function(params):
                                try:
                                    from sklearn.model_selection import cross_val_score
                                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                                    from sklearn.linear_model import LogisticRegression, LinearRegression
                                    from sklearn.svm import SVC, SVR

                                    if model_key == 'random_forest':
                                        if self.current_config.task_type == TaskType.CLASSIFICATION:
                                            model = RandomForestClassifier(**params, random_state=42)
                                        else:
                                            model = RandomForestRegressor(**params, random_state=42)
                                    elif model_key == 'logistic_regression':
                                        model = LogisticRegression(**params, random_state=42, max_iter=1000)
                                    elif model_key == 'linear_regression':
                                        model = LinearRegression(**params)
                                    elif model_key == 'svm':
                                        if self.current_config.task_type == TaskType.CLASSIFICATION:
                                            model = SVC(**params, random_state=42)
                                        else:
                                            model = SVR(**params)
                                    else:
                                        if self.current_config.task_type == TaskType.CLASSIFICATION:
                                            model = RandomForestClassifier(**params, random_state=42)
                                        else:
                                            model = RandomForestRegressor(**params, random_state=42)

                                    cv_scores = cross_val_score(
                                        model, X.values, y.values, cv=5,
                                        scoring='accuracy' if self.current_config.task_type == TaskType.CLASSIFICATION else 'r2'
                                    )
                                    return cv_scores.mean()
                                except Exception as e:
                                    logger.warning(f"HPO objective error for {algorithm_name}: {e}")
                                    return 0.0

                            hpo_result = self.hpo_optimizer.optimize(
                                objective_function=objective_function,
                                search_space={model_key: search_space.get(model_key, {})},
                                direction='maximize',
                                study_name=f"training_{model_key}_{timestamp}"
                            )
                            best_params = getattr(hpo_result, 'best_params', search_space.get(model_key, {}))
                        else:
                            best_params = search_space.get(model_key, {})
                    except Exception as e:
                        logger.warning(f"HPO failed for {algorithm_name}: {e}")
                        best_params = search_space.get(model_key, {})
                else:
                    best_params = {}

                # Final training
                fixed_grid = {k: ([v] if not isinstance(v, (list, tuple, np.ndarray)) else list(v)) for k, v in (best_params or {}).items()}
                result = training_engine.train_model(
                    X=X.values,
                    y=y.values,
                    model_type=model_key,
                    model_name=model_name,
                    param_grid=fixed_grid if fixed_grid else None
                )

                model_training_time = time.time() - model_start_time

                # Store trained model information
                self.trained_models[model_name] = {
                    'algorithm': algorithm_name,
                    'model_key': model_key,
                    'target_column': target_column,
                    'training_time': model_training_time,
                    'result': result,
                    'feature_names': X.columns.tolist(),
                    'data_shape': X.shape,
                    'hpo_enabled': enable_hpo,
                    'hpo_trials': hpo_trials if enable_hpo else None,
                    'best_params': best_params,
                    'custom_params': model_custom_params
                }

                # Log experiment metrics
                if self.experiment_tracker and ADVANCED_FEATURES_AVAILABLE:
                    metrics = result.get('metrics', {})
                    self.experiment_tracker.log_metrics(metrics, step=model_name)

                return {
                    'algorithm': algorithm_name,
                    'model_name': model_name,
                    'model_key': model_key,
                    'result': result,
                    'training_time': model_training_time,
                    'best_params': best_params,
                    'custom_params': model_custom_params,
                    'hpo_enabled': enable_hpo and bool(best_params),
                    'status': 'success',
                    'feature_names': X.columns.tolist(),
                    'data_shape': X.shape
                }
            except Exception as e:
                logger.error(f"❌ Failed to train {algorithm_name}: {e}")
                return {
                    'algorithm': algorithm_name,
                    'model_key': self.get_model_key_from_name(algorithm_name),
                    'error': str(e),
                    'training_time': 0,
                    'status': 'failed'
                }

        # Train models (parallel or sequential)
        if enable_parallel and len(algorithms) > 1:
            training_progress.append(f"🚀 Starting parallel training of {len(algorithms)} models...")
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(max_workers, len(algorithms))) as executor:
                future_to_algorithm = {executor.submit(train_single_model, alg): alg for alg in algorithms}
                for future in future_to_algorithm:
                    algorithm_name = future_to_algorithm[future]
                    try:
                        result = future.result()
                        results[algorithm_name] = result
                        if result.get('status') == 'success':
                            training_progress.append(f"✅ {algorithm_name}: Completed in {result['training_time']:.2f}s")
                        else:
                            training_progress.append(f"❌ {algorithm_name}: Failed - {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        results[algorithm_name] = {
                            'algorithm': algorithm_name,
                            'error': str(e),
                            'status': 'failed'
                        }
                        training_progress.append(f"❌ {algorithm_name}: Exception - {str(e)}")
        else:
            training_progress.append(f"🔄 Starting sequential training of {len(algorithms)} models...")
            for algorithm_name in algorithms:
                training_progress.append(f"🎯 Training {algorithm_name}...")
                result = train_single_model(algorithm_name)
                results[algorithm_name] = result
                if result.get('status') == 'success':
                    training_progress.append(f"✅ {algorithm_name}: Completed in {result['training_time']:.2f}s")
                else:
                    training_progress.append(f"❌ {algorithm_name}: Failed - {result.get('error', 'Unknown error')}")

        total_training_time = time.time() - start_time

        # End experiment
        if self.experiment_tracker and ADVANCED_FEATURES_AVAILABLE:
            self.experiment_tracker.end_experiment()

        return results, training_progress, total_training_time

    def train_multiple_models(self, target_column: str, algorithms: List[str], 
                             enable_hpo: bool = True, hpo_trials: int = 50,
                             enable_parallel: bool = True, max_workers: int = 4,
                             custom_hyperparams: str = "",
                             prefilter_by_task: bool = True) -> Tuple[str, str, str, gr.Dropdown]:
        """Train multiple models simultaneously with individual hyperparameter configurations.

        Added: prefilter_by_task to ensure only task-compatible models are trained.
        """
        if self.inference_only:
            return "❌ Model training is disabled in inference-only mode.", "", "", gr.Dropdown()
        
        if self.current_data is None:
            return "❌ Please load data first.", "", "", gr.Dropdown()
        
        if target_column not in self.current_data.columns:
            return f"❌ Target column '{target_column}' not found in data.", "", "", gr.Dropdown()
        
        if not algorithms:
            return "❌ Please select at least one algorithm to train.", "", "", gr.Dropdown()
        
        try:
            # Parse custom hyperparameters
            custom_hp_dict = {}
            if custom_hyperparams.strip():
                try:
                    import json
                    custom_hp_dict = json.loads(custom_hyperparams)
                except json.JSONDecodeError as e:
                    return f"❌ Invalid JSON in custom hyperparameters: {str(e)}", "", ""
            
            # Delegate to unified core
            results, training_progress, total_training_time = self._train_algorithms(
                target_column=target_column,
                algorithms=list(algorithms) if isinstance(algorithms, list) else [],
                enable_hpo=enable_hpo,
                hpo_trials=hpo_trials,
                enable_parallel=enable_parallel,
                max_workers=max_workers,
                custom_hyperparams=custom_hp_dict,
                prefilter_by_task=prefilter_by_task
            )

            # Generate results
            progress_text = "\n".join(training_progress)
            progress_text += f"\n\n🏁 Total Training Time: {total_training_time:.2f} seconds"
            
            summary_text = self._generate_multimodel_summary(results, total_training_time)
            detailed_text = self._generate_multimodel_detailed_results(results)
            
            # Update trained models dropdown
            trained_models_dropdown = gr.Dropdown(
                choices=self.get_trained_model_list(),
                value="Select a trained model...",
                label="Trained Models"
            )
            
            return progress_text, summary_text, detailed_text, trained_models_dropdown
            
        except Exception as e:
            error_msg = f"Error during multi-model training: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", "", gr.Dropdown()
    
    def _generate_multimodel_summary(self, results: Dict[str, Any], total_time: float) -> str:
        """Generate summary of multi-model training results"""
        try:
            summary_lines = [
                "🏆 Multi-Model Training Summary",
                "=" * 50,
                ""
            ]
            
            successful_models = []
            failed_models = []
            
            for algorithm, result_data in results.items():
                if result_data['status'] == 'success':
                    result = result_data.get('result', {})
                    metrics = result.get('metrics', {})
                    primary_score = metrics.get('f1', metrics.get('r2', metrics.get('accuracy', 0)))
                    successful_models.append((algorithm, primary_score, result_data))
                else:
                    failed_models.append((algorithm, result_data))
            
            # Sort successful models by performance
            successful_models.sort(key=lambda x: x[1], reverse=True)
            
            summary_lines.append(f"📊 Training Overview:")
            summary_lines.append(f"   • Total Models: {len(results)}")
            summary_lines.append(f"   • Successful: {len(successful_models)}")
            summary_lines.append(f"   • Failed: {len(failed_models)}")
            summary_lines.append(f"   • Total Time: {total_time:.2f} seconds")
            summary_lines.append("")
            
            if successful_models:
                summary_lines.append("🥇 Top Performing Models:")
                for rank, (algorithm, score, result_data) in enumerate(successful_models[:5], 1):
                    status_icon = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
                    hpo_status = " (HPO)" if result_data.get('hpo_enabled', False) else ""
                    summary_lines.append(f"   {status_icon} {algorithm}{hpo_status}: {score:.4f} ({result_data['training_time']:.2f}s)")
                summary_lines.append("")
            
            if failed_models:
                summary_lines.append("❌ Failed Models:")
                for algorithm, result_data in failed_models:
                    error_preview = result_data.get('error', 'Unknown error')[:50]
                    summary_lines.append(f"   • {algorithm}: {error_preview}...")
                summary_lines.append("")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            return f"Error generating multi-model summary: {str(e)}"
    
    def _generate_multimodel_detailed_results(self, results: Dict[str, Any]) -> str:
        """Generate detailed results for multi-model training"""
        try:
            detail_lines = [
                "📊 Detailed Multi-Model Results",
                "=" * 60,
                ""
            ]
            
            for algorithm, result_data in results.items():
                detail_lines.append(f"🤖 {algorithm}")
                detail_lines.append("-" * (len(algorithm) + 3))
                
                if result_data['status'] == 'success':
                    result = result_data.get('result', {})
                    metrics = result.get('metrics', {})
                    
                    detail_lines.append(f"Status: ✅ Success")
                    detail_lines.append(f"Model Name: {result_data.get('model_name', 'N/A')}")
                    detail_lines.append(f"Training Time: {result_data.get('training_time', 0):.2f} seconds")
                    detail_lines.append(f"HPO Enabled: {'Yes' if result_data.get('hpo_enabled', False) else 'No'}")
                    
                    if metrics:
                        detail_lines.append("")
                        detail_lines.append("📈 Performance Metrics:")
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                detail_lines.append(f"   • {metric.replace('_', ' ').title()}: {value:.4f}")
                            else:
                                detail_lines.append(f"   • {metric.replace('_', ' ').title()}: {value}")
                    
                    # Show hyperparameters
                    best_params = result_data.get('best_params', {})
                    custom_params = result_data.get('custom_params', {})
                    
                    if best_params or custom_params:
                        detail_lines.append("")
                        detail_lines.append("🔧 Hyperparameters:")
                        if custom_params:
                            detail_lines.append("   Custom Parameters:")
                            for param, value in custom_params.items():
                                detail_lines.append(f"     • {param}: {value}")
                        if best_params and result_data.get('hpo_enabled', False):
                            detail_lines.append("   HPO Best Parameters:")
                            for param, value in best_params.items():
                                detail_lines.append(f"     • {param}: {value}")
                    
                else:
                    detail_lines.append(f"Status: ❌ Failed")
                    detail_lines.append(f"Error: {result_data.get('error', 'Unknown error')}")
                
                detail_lines.append("")
            
            return "\n".join(detail_lines)
            
        except Exception as e:
            return f"Error generating detailed multi-model results: {str(e)}"

    def make_prediction(self, input_data: str, selected_model: str = None) -> str:
        """Make predictions using the trained model"""
        if self.inference_only:
            return "Use the Inference Server tab for predictions in inference-only mode."
        
        try:
            # Determine which model to use
            if selected_model and selected_model != "Select a trained model..." and selected_model in self.trained_models:
                model_name = selected_model
            elif self.training_engine is None:
                return "No model available. Please train a model first or select a trained model."
            else:
                model_name = "Current Training Engine Model"
            
            # Parse input data
            try:
                if input_data.strip().startswith('['):
                    # JSON array format
                    data = json.loads(input_data)
                    input_array = np.array(data).reshape(1, -1)
                else:
                    # Comma-separated values
                    data = [float(x.strip()) for x in input_data.split(',')]
                    input_array = np.array(data).reshape(1, -1)
            except Exception as e:
                return f"Error parsing input data: {str(e)}. Please use comma-separated values or JSON array format."
            
            # Make prediction
            success, predictions = self.training_engine.predict(input_array)
            
            if not success:
                return f"Prediction failed: {predictions}"
            
            # Format results
            if isinstance(predictions, np.ndarray):
                if len(predictions.shape) == 1:
                    result = predictions[0]
                else:
                    result = predictions[0]
            else:
                result = predictions
            
            prediction_text = f"""
Prediction Result:

- Model Used: {model_name}
- Input: {input_data}
- Prediction: {result}
- Data Shape: {input_array.shape}
            """
            
            return prediction_text
            
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            logger.error(error_msg)
            return error_msg


def create_ui(inference_only: bool = False):
    """Create and configure the Gradio interface"""
    
    app = MLSystemUI(inference_only=inference_only)
    
    # Load CSS styles
    css = load_css_file()
    
    title = "🚀 ML Inference Server" if inference_only else "🚀 Advanced AutoML Platform with AI Optimization"
    description = """
🎯 Real-time ML inference server with enterprise-grade security and performance optimization.

✨ Load your trained models and get instant predictions with minimal latency!
    """ if inference_only else """
🤖 Complete AutoML platform with cutting-edge AI optimization, automated hyperparameter tuning, and intelligent model comparison.

🎯 Advanced Features:
• 🧠 Adaptive Hyperparameter Optimization with Bayesian & Tree-based methods
• � Multi-Model Training with parallel execution and custom hyperparameters
• �🔄 Multi-Model Comparison with automated benchmarking  
• ⚡ Memory-Aware Processing & Batch Optimization
• 📊 Real-time Performance Monitoring & Experiment Tracking
• 🛡️ Enterprise Security with Audit Logging
• 🚀 Mixed Precision Training & Model Quantization
• 🎛️ Intelligent Preprocessing Configuration Optimization

📈 Quick Start: Upload data → Auto-optimize preprocessing → Train single/multiple models with integrated HPO → Compare performance → Deploy best performer
    """

    with gr.Blocks(css=css, title=title, theme=gr.themes.Soft()) as interface:
        gr.HTML(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2.5em;">{title}</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">{description}</p>
            </div>
        """)
        
        if inference_only:
            # Inference-only interface
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 📁 Model Upload")
                    model_file = gr.File(label="Upload Model File", file_types=[".pkl", ".joblib", ".json"])
                    model_password = gr.Textbox(label="Encryption Password (if required)", type="password")
                    load_model_btn = gr.Button("Load Model", variant="primary")
                    model_status = gr.Textbox(label="Model Status", interactive=False)
                    
                with gr.Column(scale=1):
                    gr.Markdown("## 🔮 Make Predictions")
                    prediction_input = gr.Textbox(
                        label="Input Data", 
                        placeholder="Enter comma-separated values or JSON array: [1.5, 2.3, 0.8] or 1.5,2.3,0.8",
                        lines=3
                    )
                    predict_btn = gr.Button("Predict", variant="secondary")
                    prediction_output = gr.Textbox(label="Prediction Result", lines=10, interactive=False)
            
            # Event handlers for inference-only mode
            load_model_btn.click(
                app.inference_server.load_model_from_path,
                inputs=[model_file, model_password],
                outputs=model_status
            )
            
            predict_btn.click(
                lambda x: app.inference_server.predict(x) if hasattr(app.inference_server, 'predict') else "Model not loaded",
                inputs=prediction_input,
                outputs=prediction_output
            )
            
        else:
            # Full training and inference interface
            with gr.Tabs():
                # Data Upload Tab
                with gr.Tab("📁 Data Upload", id="data"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Upload Your Dataset")
                            data_file = gr.File(label="Choose Data File", file_types=[".csv", ".xlsx", ".json"])
                            
                            gr.Markdown("## Or Try Sample Datasets")
                            sample_datasets = [
                                "Select a dataset...",
                                "Iris Classification",
                                "Boston Housing",
                                "Wine Quality",
                                "Diabetes",
                                "Breast Cancer"
                            ]
                            sample_dropdown = gr.Dropdown(choices=sample_datasets, label="Sample Datasets")
                            load_sample_btn = gr.Button("Load Sample Data", variant="secondary")
                            
                            # Advanced preprocessing
                            gr.Markdown("## 🚀 Advanced Preprocessing")
                            optimize_preprocessing_btn = gr.Button("Optimize Preprocessing Config", variant="primary")
                            preprocessing_output = gr.Textbox(label="Preprocessing Optimization", lines=10, interactive=False)
                            
                        with gr.Column(scale=2):
                            data_info = gr.Textbox(label="Dataset Information", lines=10, interactive=False)
                            data_preview = gr.HTML(label="Data Preview")
                
                # Configuration Tab
                with gr.Tab("⚙️ Configuration", id="config"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Training Configuration")
                            task_type = gr.Dropdown(
                                choices=["classification", "regression"], 
                                label="Task Type",
                                value="classification"
                            )
                            optimization_strategy = gr.Dropdown(
                                choices=["speed", "balanced", "accuracy"], 
                                label="Optimization Strategy",
                                value="balanced"
                            )
                            cv_folds = gr.Slider(minimum=2, maximum=10, value=5, step=1, label="Cross-Validation Folds")
                            test_size = gr.Slider(minimum=0.1, maximum=0.5, value=0.2, step=0.05, label="Test Set Size")
                            random_state = gr.Number(value=42, label="Random State")
                            
                        with gr.Column():
                            gr.Markdown("## Advanced Options")
                            enable_feature_selection = gr.Checkbox(label="Enable Feature Selection", value=False)
                            normalization = gr.Dropdown(
                                choices=["standard", "minmax", "robust", "none"], 
                                label="Normalization",
                                value="standard"
                            )
                            enable_quantization = gr.Checkbox(label="Enable Model Quantization", value=False)
                            optimization_mode = gr.Dropdown(
                                choices=["speed", "balanced", "accuracy"], 
                                label="Optimization Mode",
                                value="balanced"
                            )
                            
                            # Advanced features toggle
                            enable_mixed_precision = gr.Checkbox(label="Enable Mixed Precision", value=False)
                            enable_experiment_tracking = gr.Checkbox(label="Enable Experiment Tracking", value=True)
                    
                    create_config_btn = gr.Button("Create Configuration", variant="primary")
                    config_output = gr.Textbox(label="Configuration Status", lines=15, interactive=False)
                
                # Training Tab
                with gr.Tab("🚀 Training", id="training"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Multi-Model Training Configuration")
                            multi_target_column = gr.Textbox(label="Target Column Name", placeholder="e.g., target, label, price")
                            
                            gr.Markdown("### 📋 Model Selection")
                            multi_prefilter_by_task = gr.Checkbox(label="Prefilter models by task", value=True)
                            multi_algorithms = gr.Dropdown(
                                choices=[],
                                label="Select Models to Train",
                                value=[],
                                multiselect=True
                            )
                            
                            gr.Markdown("### ⚙️ Global Settings")
                            multi_enable_hpo = gr.Checkbox(label="Enable HPO for All Models", value=True)
                            multi_hpo_trials = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="HPO Trials per Model")
                            multi_parallel_training = gr.Checkbox(label="Enable Parallel Training", value=True)
                            multi_max_workers = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Max Parallel Workers")
                            
                            # Model-specific hyperparameters
                            gr.Markdown("### 🔧 Model-Specific Hyperparameters")
                            custom_hyperparams = gr.Textbox(
                                label="Custom Hyperparameters (JSON)",
                                placeholder='{"random_forest": {"n_estimators": [100, 200], "max_depth": [5, 10]}, "xgboost": {"learning_rate": [0.01, 0.1]}}',
                                lines=5
                            )
                            
                            multi_train_btn = gr.Button("Start Multi-Model Training", variant="primary")
                            
                        with gr.Column():
                            multi_training_output = gr.Textbox(label="Multi-Training Progress", lines=12, interactive=False)
                            multi_results_summary = gr.Textbox(label="Training Summary", lines=10, interactive=False)

                    with gr.Row():
                        multi_detailed_results = gr.Textbox(label="Detailed Results", lines=15, interactive=False)
                
                # Model Comparison Tab
                with gr.Tab("🏆 Model Comparison", id="comparison"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Multi-Model Comparison")
                            comp_target_column = gr.Textbox(label="Target Column", placeholder="e.g., target, label, price")
                            comp_algorithms = gr.Dropdown(
                                choices=[],
                                label="Select Algorithms to Compare",
                                value=[],
                                multiselect=True
                            )
                            enable_hpo_comparison = gr.Checkbox(label="Enable HPO for All Models", value=True)
                            
                            compare_models_btn = gr.Button("Compare Models", variant="primary")
                            
                        with gr.Column():
                            comparison_summary = gr.Textbox(label="Comparison Summary", lines=8, interactive=False)
                            
                    with gr.Row():
                        with gr.Column():
                            detailed_comparison = gr.Textbox(label="Detailed Results", lines=12, interactive=False)
                        with gr.Column():
                            comparison_visualization = gr.Textbox(label="Performance Visualization", lines=12, interactive=False)
                
                # Prediction Tab
                with gr.Tab("🔮 Predictions", id="predictions"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Make Predictions")
                            trained_model_dropdown = gr.Dropdown(
                                choices=["Select a trained model..."], 
                                label="Select Trained Model"
                            )
                            prediction_input = gr.Textbox(
                                label="Input Data", 
                                placeholder="Enter comma-separated values: 1.5,2.3,0.8",
                                lines=3
                            )
                            predict_btn = gr.Button("Predict", variant="secondary")
                            
                        with gr.Column():
                            prediction_output = gr.Textbox(label="Prediction Results", lines=10, interactive=False)
                
                # Monitoring Dashboard Tab
                with gr.Tab("📊 Monitoring", id="monitoring"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## System Monitoring")
                            refresh_dashboard_btn = gr.Button("Refresh Dashboard", variant="primary")
                            
                        with gr.Column():
                            pass
                    
                    with gr.Row():
                        monitoring_dashboard = gr.HTML(label="System Dashboard")
            
            # Event handlers
            data_file.change(
                app.load_data,
                inputs=data_file,
                outputs=[data_info, gr.State(), gr.State(), data_preview]
            )
            
            load_sample_btn.click(
                app.load_sample_data,
                inputs=sample_dropdown,
                outputs=[data_info, gr.State(), gr.State(), data_preview]
            )
            
            optimize_preprocessing_btn.click(
                app.optimize_preprocessing_config,
                outputs=preprocessing_output
            )
            
            create_config_btn.click(
                app.create_training_config,
                inputs=[task_type, optimization_strategy, cv_folds, test_size, random_state, 
                       enable_feature_selection, normalization, enable_quantization, optimization_mode],
                outputs=[config_output]
            )
            
            # Update algorithm choices for comparison when config is created
            create_config_btn.click(
                lambda: app.get_multi_algorithm_choices(True),
                outputs=comp_algorithms
            )
            
            # Update multi-model algorithm choices when config is created (default prefilter=True)
            create_config_btn.click(
                lambda: app.get_multi_algorithm_choices(True),
                outputs=multi_algorithms
            )

            # Toggle prefilter updates the multi-model algorithm choices
            multi_prefilter_by_task.change(
                app.get_multi_algorithm_choices,
                inputs=multi_prefilter_by_task,
                outputs=multi_algorithms
            )
            
            # Single-model training removed; use multi-model tab instead
            
            multi_train_btn.click(
                app.train_multiple_models,
                inputs=[multi_target_column, multi_algorithms, multi_enable_hpo, multi_hpo_trials, 
                       multi_parallel_training, multi_max_workers, custom_hyperparams, multi_prefilter_by_task],
                outputs=[multi_training_output, multi_results_summary, multi_detailed_results, trained_model_dropdown]
            )
            
            compare_models_btn.click(
                app.compare_multiple_models,
                inputs=[comp_target_column, comp_algorithms, enable_hpo_comparison],
                outputs=[comparison_summary, detailed_comparison, comparison_visualization]
            )
            
            predict_btn.click(
                app.make_prediction,
                inputs=[prediction_input, trained_model_dropdown],
                outputs=prediction_output
            )
            
            refresh_dashboard_btn.click(
                app.get_system_monitoring_dashboard,
                outputs=monitoring_dashboard
            )
    
    return interface


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(description="ML Training & Inference System")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run in inference-only mode (no training capabilities)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    parser.add_argument(
        "--auth-required",
        action="store_true",
        help="Require authentication"
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=4,
        help="Maximum number of threads"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Security validation
    if args.share and SECURITY_ENV and hasattr(SECURITY_ENV, 'security_level'):
        logger.warning("⚠️  WARNING: --share option should be used with caution!")
    
    # Configure authentication
    auth_config = None
    if args.auth_required:
        auth_config = get_auth_config()
        if auth_config and callable(auth_config):
            logger.info("🔐 Authentication enabled")
        else:
            logger.info("🔓 Authentication disabled")
    
    # Create and launch the interface
    interface = create_ui(inference_only=args.inference_only)
    
    # Display startup information
    print(f"""
🚀 Starting {'ML Inference Server' if args.inference_only else 'ML Training & Inference System'}

🔧 Configuration:
Mode: {'Inference Only' if args.inference_only else 'Full Training & Inference'}
Host: {args.host}
Port: {args.port}
Share: {'Yes' if args.share else 'No'}
Authentication: {'Required' if auth_config else 'Disabled'}

🚀 Available Features:
{'- Real-time model inference with optimized performance' if args.inference_only else '''- 🧠 Advanced ML algorithms with 20+ supported models
- 🚀 Multi-Model Training with parallel execution and custom hyperparameters
- 🎯 Adaptive Hyperparameter Optimization (Bayesian, TPE, CMA-ES)
- 🔄 Automated Multi-Model Comparison & Benchmarking
- ⚡ Memory-Aware Processing & Batch Optimization
- 📊 Real-time System Monitoring & Performance Dashboard  
- 🔬 Experiment Tracking with MLflow Integration
- 🛡️ Enterprise Security with Audit Logging
- 🚀 Mixed Precision Training & Model Quantization
- 🎛️ Intelligent Preprocessing Configuration Optimization
- 📈 Advanced Model Selection & Performance Analytics
- 💾 Secure Model Storage & Version Management
- 🌐 RESTful API for Production Deployment'''}

⚠️  Security Notice:
- Keep your data secure
- Monitor logs regularly
- Use authentication in production
    """)
    
    # Launch configuration
    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "debug": False,
        "show_error": True,
        "max_threads": args.max_threads,
        "quiet": False,
    }
    
    # Add authentication if configured
    if auth_config and callable(auth_config):
        launch_kwargs["auth"] = auth_config
    
    try:
        # Launch the interface
        interface.launch(**launch_kwargs)
        
    except KeyboardInterrupt:
        logger.info("👋 Shutting down gracefully...")
        if SECURITY_MANAGER and hasattr(SECURITY_MANAGER, 'auditor'):
            SECURITY_MANAGER.auditor.logger.info("SYSTEM_SHUTDOWN: Gradio interface stopped by user")
    except Exception as e:
        logger.error(f"Failed to launch interface: {e}")
        if SECURITY_MANAGER and hasattr(SECURITY_MANAGER, 'auditor'):
            SECURITY_MANAGER.auditor.logger.error(f"SYSTEM_ERROR: Failed to launch interface - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
