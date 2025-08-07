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
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
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
    
    def security_wrapper(func): return func
    def validate_file_upload(path, extensions): return True
    def secure_data_processing(data): return data
    def create_security_headers(): return {}
    def get_auth_config(): return None

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
    """Enhanced Gradio UI for the ML Training & Inference System with Security Integration"""
    
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

    def update_algorithm_choices(self, task_type: str) -> gr.Dropdown:
        """Update algorithm choices based on task type"""
        algorithms = self.get_algorithms_for_task(task_type)
        return gr.Dropdown(choices=algorithms, value=algorithms[0] if algorithms else None)

    def create_training_config(self, task_type: str, optimization_strategy: str, 
                             cv_folds: int, test_size: float, random_state: int,
                             enable_feature_selection: bool, normalization: str,
                             enable_quantization: bool, optimization_mode: str) -> Tuple[str, gr.Dropdown, List[str]]:
        """Create training configuration and update algorithm choices"""
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
            
            # Get available algorithms for the task type
            algorithms = self.get_algorithms_for_task(task_type)
            algorithm_dropdown = gr.Dropdown(
                choices=algorithms,
                value=algorithms[0] if algorithms else None,
                label="Available ML Algorithms"
            )
            
            config_text = f"""
Configuration Created Successfully!

- Task Type: {task_type}
- Optimization Strategy: {optimization_strategy}
- CV Folds: {cv_folds}
- Test Size: {test_size}
- Feature Selection: {'Enabled' if enable_feature_selection else 'Disabled'}
- Normalization: {normalization}
- Quantization: {'Enabled' if enable_quantization else 'Disabled'}
- Available Algorithms: {len(algorithms)} algorithms for {task_type.lower()}

✅ Algorithm dropdown in Training tab has been updated!
            """
            
            return config_text, algorithm_dropdown, algorithms
            
        except Exception as e:
            error_msg = f"Error creating configuration: {str(e)}"
            logger.error(error_msg)
            return error_msg, gr.Dropdown(), []

    def train_model(self, target_column: str, algorithm_name: str, model_name: str = None) -> Tuple[str, str, str, gr.Dropdown]:
        """Train model with the current configuration"""
        if self.inference_only:
            return "❌ Model training is disabled in inference-only mode.", "", "", gr.Dropdown()
        
        if self.current_data is None:
            return "❌ Please load data first.", "", "", gr.Dropdown()
        
        if target_column not in self.current_data.columns:
            return f"❌ Target column '{target_column}' not found in data.", "", "", gr.Dropdown()
        
        try:
            # Initialize training engine
            if self.current_config is None:
                # Create a default configuration
                task_type = TaskType.CLASSIFICATION if self.determine_task_type(self.current_data[target_column]) == "classification" else TaskType.REGRESSION
                self.current_config = MLTrainingEngineConfig(
                    task_type=task_type,
                    optimization_strategy=OptimizationStrategy.BALANCED,
                    model_path="./models",
                    checkpoint_path="./checkpoints"
                )
            
            self.training_engine = MLTrainingEngine(self.current_config)
            
            # Prepare data
            X = self.current_data.drop(columns=[target_column])
            y = self.current_data[target_column]
            
            # Handle categorical features
            categorical_columns = X.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                for col in categorical_columns:
                    X[col] = pd.Categorical(X[col]).codes
            
            # Get model key
            model_key = self.get_model_key_from_name(algorithm_name)
            
            # Generate model name if not provided
            if not model_name:
                timestamp = int(time.time())
                model_name = f"{algorithm_name.split(' - ')[-1]}_{timestamp}"
            
            # Train model
            start_time = time.time()
            result = self.training_engine.train_model(
                X=X.values, 
                y=y.values,
                model_type=model_key,
                model_name=model_name
            )
            
            training_time = time.time() - start_time
            logger.info("Training completed!")
            
            # Store trained model information
            self.trained_models[model_name] = {
                'algorithm': algorithm_name,
                'model_key': model_key,
                'target_column': target_column,
                'training_time': training_time,
                'result': result,
                'feature_names': X.columns.tolist(),
                'data_shape': X.shape
            }
            
            # Generate results summary
            metrics_text = "Training Results:\n\n"
            if 'metrics' in result and result['metrics']:
                for metric, value in result['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics_text += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
                    else:
                        metrics_text += f"- {metric.replace('_', ' ').title()}: {value}\n"
            
            metrics_text += f"\n- Training Time: {training_time:.2f} seconds"
            
            # Feature importance
            importance_text = ""
            if 'feature_importance' in result and result['feature_importance'] is not None:
                importance = result['feature_importance']
                feature_names = X.columns.tolist()
                
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
            
            # Model summary
            summary_text = f"""
Model Training Summary

- Model Name: {model_name}
- Algorithm: {algorithm_name}
- Dataset Shape: {X.shape[0]} samples × {X.shape[1]} features
- Target Column: {target_column}
- Status: ✅ Training Completed Successfully
            """
            
            # Update trained models dropdown
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
    
    title = "🚀 ML Inference Server" if inference_only else "🚀 AutoML Training & Inference Platform"
    description = """
🎯 Real-time ML inference server with enterprise-grade security and performance optimization.

✨ Load your trained models and get instant predictions with minimal latency!
    """ if inference_only else """
🤖 Complete AutoML platform with advanced optimization, model comparison, and secure deployment.

🎯 Quick Start: Upload data → Configure training → Train multiple models → Compare results → Deploy predictions
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
                    
                    create_config_btn = gr.Button("Create Configuration", variant="primary")
                    config_output = gr.Textbox(label="Configuration Status", lines=15, interactive=False)
                
                # Training Tab
                with gr.Tab("🚀 Training", id="training"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Model Training")
                            target_column = gr.Textbox(label="Target Column Name", placeholder="e.g., target, label, price")
                            algorithm_dropdown = gr.Dropdown(
                                choices=["Please create configuration first"], 
                                label="Select Algorithm"
                            )
                            model_name = gr.Textbox(label="Model Name (optional)", placeholder="my_model")
                            
                            train_btn = gr.Button("Train Model", variant="primary")
                            
                        with gr.Column():
                            training_output = gr.Textbox(label="Training Results", lines=10, interactive=False)
                            metrics_output = gr.Textbox(label="Model Metrics", lines=8, interactive=False)
                            importance_output = gr.Textbox(label="Feature Importance", lines=6, interactive=False)
                
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
            
            create_config_btn.click(
                app.create_training_config,
                inputs=[task_type, optimization_strategy, cv_folds, test_size, random_state, 
                       enable_feature_selection, normalization, enable_quantization, optimization_mode],
                outputs=[config_output, algorithm_dropdown, gr.State()]
            )
            
            train_btn.click(
                app.train_model,
                inputs=[target_column, algorithm_dropdown, model_name],
                outputs=[training_output, metrics_output, importance_output, trained_model_dropdown]
            )
            
            predict_btn.click(
                app.make_prediction,
                inputs=[prediction_input, trained_model_dropdown],
                outputs=prediction_output
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
{'- Real-time model inference' if args.inference_only else '''- Multiple ML algorithms support
- Advanced model training with hyperparameter optimization
- Model performance comparison
- Secure model storage
- Real-time inference'''}

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
