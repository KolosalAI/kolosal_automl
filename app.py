import gradio as gr
import pandas as pd
import numpy as np
import json
import os
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import requests
from io import StringIO
import argparse
import sys

# Import matplotlib and seaborn with proper backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Import your modules
from modules.configs import (
    MLTrainingEngineConfig, InferenceEngineConfig, 
    TaskType, OptimizationStrategy, ModelSelectionCriteria,
    OptimizationMode, NormalizationType, QuantizationType
)
from modules.engine.train_engine import MLTrainingEngine
from modules.engine.inference_engine import InferenceEngine
from modules.device_optimizer import DeviceOptimizer
from modules.model_manager import SecureModelManager

import modules.engine.batch_processor as batch_processor
import modules.engine.data_preprocessor as data_preprocessor
import modules.engine.lru_ttl_cache as lru_ttl_cache
import modules.engine.quantizer as quantizer
import modules.engine.utils as engine_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_css_file(css_path: str = "static/styles.css") -> str:
    """Load CSS styles from external file"""
    try:
        if os.path.exists(css_path):
            with open(css_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            logger.warning(f"CSS file not found at {css_path}. Using default minimal styles.")
            return ""
    except Exception as e:
        logger.error(f"Error loading CSS file: {e}")
        return ""

class DataPreviewGenerator:
    """Generate comprehensive data previews and visualizations"""
    
    @staticmethod
    def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        summary = {
            "basic_info": {
                "shape": df.shape,
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict()
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_by_column": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict()
            },
            "data_types": {
                "numerical": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime": df.select_dtypes(include=['datetime64']).columns.tolist()
            }
        }
        
        # Add descriptive statistics for numerical columns
        numerical_cols = summary["data_types"]["numerical"]
        if numerical_cols:
            summary["numerical_stats"] = df[numerical_cols].describe().round(3).to_dict()
        
        # Add categorical summaries
        categorical_cols = summary["data_types"]["categorical"]
        if categorical_cols:
            summary["categorical_stats"] = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                value_counts = df[col].value_counts().head(10)
                summary["categorical_stats"][col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": value_counts.to_dict()
                }
        
        return summary
    
    @staticmethod
    def create_data_visualizations(df: pd.DataFrame, max_cols: int = 6) -> str:
        """Create data visualization plots and return as base64 image"""
        try:
            # Set style with proper seaborn v0.11+ syntax
            plt.style.use('default')
            sns.set_palette("husl")
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:max_cols]
            
            # Calculate subplot layout
            total_plots = 0
            if len(numerical_cols) > 0:
                total_plots += min(len(numerical_cols), 4)  # Distribution plots
            if len(categorical_cols) > 0:
                total_plots += min(len(categorical_cols), 2)  # Bar plots
            if len(numerical_cols) > 1:
                total_plots += 1  # Correlation heatmap
            
            if total_plots == 0:
                return None
            
            # Create figure
            n_cols = 3
            n_rows = (total_plots + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            
            if n_rows == 1:
                if total_plots == 1:
                    axes = [axes]
                else:
                    axes = axes if isinstance(axes, list) else axes.flatten()
            else:
                axes = axes.flatten()
            
            plot_idx = 0
            
            # 1. Numerical distributions
            for i, col in enumerate(numerical_cols[:4]):
                if plot_idx < len(axes):
                    try:
                        axes[plot_idx].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                        axes[plot_idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                        axes[plot_idx].set_xlabel(col)
                        axes[plot_idx].set_ylabel('Frequency')
                        axes[plot_idx].grid(True, alpha=0.3)
                        plot_idx += 1
                    except Exception as e:
                        continue
            
            # 2. Categorical bar plots
            for i, col in enumerate(categorical_cols[:2]):
                if plot_idx < len(axes):
                    try:
                        value_counts = df[col].value_counts().head(10)
                        bars = axes[plot_idx].bar(range(len(value_counts)), value_counts.values, 
                                                 color='lightcoral', alpha=0.7, edgecolor='black')
                        axes[plot_idx].set_title(f'Top 10 Values in {col}', fontsize=12, fontweight='bold')
                        axes[plot_idx].set_xlabel(col)
                        axes[plot_idx].set_ylabel('Count')
                        
                        # Rotate x-axis labels if needed
                        labels = [str(x)[:15] + '...' if len(str(x)) > 15 else str(x) for x in value_counts.index]
                        axes[plot_idx].set_xticks(range(len(value_counts)))
                        axes[plot_idx].set_xticklabels(labels, rotation=45, ha='right')
                        axes[plot_idx].grid(True, alpha=0.3)
                        plot_idx += 1
                    except Exception as e:
                        continue
            
            # 3. Correlation heatmap for numerical columns
            if len(numerical_cols) > 1 and plot_idx < len(axes):
                try:
                    corr_matrix = df[numerical_cols].corr()
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    
                    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                               square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=axes[plot_idx])
                    axes[plot_idx].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
                    plot_idx += 1
                except Exception as e:
                    pass
            
            # Hide unused subplots
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return None
    
    @staticmethod
    def format_data_preview(df: pd.DataFrame, summary: Dict[str, Any]) -> str:
        """Format comprehensive data preview as markdown"""
        
        # Basic information
        preview_text = f"""
## 📊 Dataset Overview

- Shape: {summary['basic_info']['shape'][0]:,} rows × {summary['basic_info']['shape'][1]} columns
- Memory Usage: {summary['basic_info']['memory_usage_mb']:.2f} MB
- Total Missing Values: {summary['missing_data']['total_missing']:,} ({summary['missing_data']['total_missing'] / (df.shape[0] * df.shape[1]) * 100:.1f}% of all data)

### 🔍 Data Types
- Numerical Columns ({len(summary['data_types']['numerical'])}): {', '.join(summary['data_types']['numerical'][:5])}{'...' if len(summary['data_types']['numerical']) > 5 else ''}
- Categorical Columns ({len(summary['data_types']['categorical'])}): {', '.join(summary['data_types']['categorical'][:5])}{'...' if len(summary['data_types']['categorical']) > 5 else ''}
- DateTime Columns ({len(summary['data_types']['datetime'])}): {', '.join(summary['data_types']['datetime'][:5])}{'...' if len(summary['data_types']['datetime']) > 5 else ''}
        """
        
        # Missing data details
        if summary['missing_data']['total_missing'] > 0:
            preview_text += "\n### ⚠️ Missing Data by Column\n"
            missing_cols = {k: v for k, v in summary['missing_data']['missing_by_column'].items() if v > 0}
            for col, missing_count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = summary['missing_data']['missing_percentage'][col]
                preview_text += f"- {col}: {missing_count:,} missing ({percentage:.1f}%)\n"
        
        # Numerical statistics
        if 'numerical_stats' in summary and summary['numerical_stats']:
            preview_text += "\n### 📈 Numerical Statistics\n"
            stats_df = pd.DataFrame(summary['numerical_stats']).round(2)
            # Create a simple text table instead of markdown table for better compatibility
            preview_text += "\n"
            for stat in ['mean', 'std', 'min', 'max']:
                if stat in stats_df.index:
                    preview_text += f"{stat.upper()}:\n"
                    for col in stats_df.columns[:5]:  # Limit to first 5 columns
                        preview_text += f"- {col}: {stats_df.loc[stat, col]:.2f}\n"
                    preview_text += "\n"
        
        # Categorical summaries
        if 'categorical_stats' in summary and summary['categorical_stats']:
            preview_text += "\n### 🏷️ Categorical Summaries\n"
            for col, stats in list(summary['categorical_stats'].items())[:3]:
                preview_text += f"\n{col} ({stats['unique_count']} unique values):\n"
                for value, count in list(stats['top_values'].items())[:5]:
                    preview_text += f"- {value}: {count:,}\n"
        
        return preview_text

class SampleDataLoader:
    """Load sample datasets from public sources"""
    
    SAMPLE_DATASETS = {
        "Iris": {
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "description": "Classic iris flower classification dataset",
            "task_type": "CLASSIFICATION",
            "target": "species"
        },
        "Boston Housing": {
            "url": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
            "description": "Boston housing prices regression dataset",
            "task_type": "REGRESSION",
            "target": "medv"
        },
        "Titanic": {
            "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "description": "Titanic passenger survival classification",
            "task_type": "CLASSIFICATION",
            "target": "Survived"
        },
        "Wine Quality": {
            "url": "https://raw.githubusercontent.com/rajeevratan84/datascienceforbeginnerssklearn/master/winequality-red.csv",
            "description": "Wine quality prediction dataset",
            "task_type": "REGRESSION",
            "target": "quality"
        },
        "Diabetes": {
            "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            "description": "Pima Indians diabetes classification",
            "task_type": "CLASSIFICATION",
            "target": "class"
        },
        "Car Evaluation": {
            "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/car.csv",
            "description": "Car evaluation classification dataset",
            "task_type": "CLASSIFICATION",
            "target": "class"
        }
    }
    
    @classmethod
    def get_available_datasets(cls):
        """Get list of available sample datasets"""
        return list(cls.SAMPLE_DATASETS.keys())
    
    @classmethod
    def load_sample_data(cls, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load a sample dataset"""
        if dataset_name not in cls.SAMPLE_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not available")
        
        dataset_info = cls.SAMPLE_DATASETS[dataset_name]
        
        try:
            # Download the dataset
            response = requests.get(dataset_info["url"], timeout=30)
            response.raise_for_status()
            
            # Handle different CSV formats
            if dataset_name == "Diabetes":
                # Diabetes dataset doesn't have headers
                columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
                df = pd.read_csv(StringIO(response.text), names=columns)
            else:
                df = pd.read_csv(StringIO(response.text))
            
            # Basic preprocessing for some datasets
            if dataset_name == "Titanic":
                # Clean up Titanic dataset
                df = df.dropna(subset=['Age', 'Embarked'])
                df['Age'] = df['Age'].fillna(df['Age'].median())
                df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
                
            metadata = {
                "name": dataset_name,
                "description": dataset_info["description"],
                "task_type": dataset_info["task_type"],
                "target_column": dataset_info["target"],
                "shape": df.shape,
                "columns": df.columns.tolist()
            }
            
            return df, metadata
            
        except Exception as e:
            raise Exception(f"Failed to load dataset '{dataset_name}': {str(e)}")

class InferenceServer:
    """Standalone inference server for trained models"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        self.model_manager = None
        self.inference_engine = None
        self.model_metadata = {}
        self.is_loaded = False
        
        if model_path:
            self.load_model_from_path(model_path, config_path)
    
    def load_model_from_path(self, model_path: str, config_path: str = None):
        """Load model from file path"""
        try:
            # Load configuration
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                config = MLTrainingEngineConfig(**config_data)
            else:
                config = MLTrainingEngineConfig()
            
            # Initialize model manager
            self.model_manager = SecureModelManager(config, logger=logger)
            
            # Load the model
            model = self.model_manager.load_model(model_path)
            
            if model is not None:
                # Initialize inference engine
                inference_config = InferenceEngineConfig()
                self.inference_engine = InferenceEngine(inference_config)
                self.inference_engine.model = model
                
                self.is_loaded = True
                logger.info(f"Model loaded successfully from {model_path}")
                
                # Extract metadata if available
                if hasattr(model, 'feature_names_'):
                    self.model_metadata['feature_names'] = model.feature_names_
                if hasattr(model, 'classes_'):
                    self.model_metadata['classes'] = model.classes_.tolist()
                    
            else:
                raise Exception("Failed to load model")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Make prediction using loaded model"""
        if not self.is_loaded:
            return {"error": "No model loaded"}
        
        try:
            success, predictions = self.inference_engine.predict(input_data)
            
            if not success:
                return {"error": f"Prediction failed: {predictions}"}
            
            result = {
                "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                "input_shape": input_data.shape,
                "model_metadata": self.model_metadata
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"error": "No model loaded"}
        
        return {
            "is_loaded": self.is_loaded,
            "metadata": self.model_metadata,
            "model_type": type(self.inference_engine.model).__name__ if self.inference_engine.model else None
        }

class MLSystemUI:
    """Enhanced Gradio UI for the ML Training & Inference System"""
    
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
    
    def load_data(self, file) -> Tuple[str, Dict, str, str]:
        """Load dataset from uploaded file with preview"""
        try:
            if file is None:
                return "No file uploaded", {}, "", ""
            
            file_path = file.name
            
            # Load based on file extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return "Unsupported file format. Please upload CSV, Excel, or JSON files.", {}, "", ""
            
            self.current_data = df
            
            # Generate data summary and preview
            summary = self.data_preview_generator.generate_data_summary(df)
            preview_text = self.data_preview_generator.format_data_preview(df, summary)
            
            # Generate sample data table
            sample_table = df.head(10).to_html(classes="table table-striped", escape=False, border=0)
            
            info_text = f"""
Data Loaded Successfully!

- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
- Missing Values: {df.isnull().sum().sum()} total
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
    
    def load_inference_model(self, file, encryption_password: str = "") -> str:
        """Load a model for inference server"""
        try:
            if file is None:
                return "No model file selected."
            
            # Load model into inference server
            if encryption_password:
                # Handle encrypted models (you may need to modify this based on your encryption implementation)
                pass
            
            self.inference_server.load_model_from_path(file.name)
            
            if self.inference_server.is_loaded:
                model_info = self.inference_server.get_model_info()
                return f"""
✅ Model loaded successfully for inference!

- File: {file.name}
- Model Type: {model_info.get('model_type', 'Unknown')}
- Status: Ready for predictions
                """
            else:
                return "❌ Failed to load model for inference."
                
        except Exception as e:
            error_msg = f"Error loading inference model: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def make_inference_prediction(self, input_data: str) -> str:
        """Make predictions using the inference server"""
        try:
            if not self.inference_server.is_loaded:
                return "No model loaded in inference server. Please load a model first."
            
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
            
            # Make prediction using inference server
            result = self.inference_server.predict(input_array)
            
            if "error" in result:
                return f"Prediction failed: {result['error']}"
            
            # Format results
            predictions = result["predictions"]
            prediction_text = f"""
Inference Server Prediction:

- Input: {input_data}
- Prediction: {predictions}
- Input Shape: {result['input_shape']}
- Model Type: {result.get('model_metadata', {}).get('model_type', 'Unknown')}
            """
            
            return prediction_text
            
        except Exception as e:
            error_msg = f"Error making inference prediction: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
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
            opt_strategy_enum = OptimizationStrategy[optimization_strategy.upper()]
            opt_mode_enum = OptimizationMode[optimization_mode.upper()]
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
        """Train a model with the current configuration"""
        if self.inference_only:
            return "Training is not available in inference-only mode.", "", "", gr.Dropdown()
        
        try:
            if self.current_data is None:
                return "No data loaded. Please upload a dataset first.", "", "", gr.Dropdown()
            
            if self.current_config is None:
                return "No configuration created. Please configure training parameters first.", "", "", gr.Dropdown()
            
            if target_column not in self.current_data.columns:
                return f"Target column '{target_column}' not found in dataset.", "", "", gr.Dropdown()
            
            if not algorithm_name or algorithm_name == "Select an algorithm...":
                return "Please select an algorithm to train.", "", "", gr.Dropdown()
            
            logger.info("Initializing training engine...")
            
            # Initialize training engine
            self.training_engine = MLTrainingEngine(self.current_config)
            
            # Prepare data
            X = self.current_data.drop(columns=[target_column])
            y = self.current_data[target_column]
            
            logger.info("Preprocessing data...")
            
            # Handle categorical features
            categorical_columns = X.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                for col in categorical_columns:
                    X[col] = pd.Categorical(X[col]).codes
            
            logger.info("Starting model training...")
            
            # Extract model key from algorithm name
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
- Task Type: {self.current_config.task_type.value}
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
            error_msg = f"Error during training: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg, "", "", gr.Dropdown()
    
    def get_trained_model_info(self, model_name: str) -> str:
        """Get information about a trained model"""
        try:
            if model_name == "Select a trained model..." or model_name not in self.trained_models:
                return "Please select a trained model to view information."
            
            model_info = self.trained_models[model_name]
            
            info_text = f"""
Trained Model Information

- Model Name: {model_name}
- Algorithm: {model_info['algorithm']}
- Target Column: {model_info['target_column']}
- Training Time: {model_info['training_time']:.2f} seconds
- Data Shape: {model_info['data_shape'][0]} samples × {model_info['data_shape'][1]} features
- Feature Names: {', '.join(model_info['feature_names'][:10])}{'...' if len(model_info['feature_names']) > 10 else ''}

Performance Metrics:
            """
            
            if 'metrics' in model_info['result'] and model_info['result']['metrics']:
                for metric, value in model_info['result']['metrics'].items():
                    if isinstance(value, (int, float)):
                        info_text += f"\n- {metric.replace('_', ' ').title()}: {value:.4f}"
                    else:
                        info_text += f"\n- {metric.replace('_', ' ').title()}: {value}"
            
            return info_text
            
        except Exception as e:
            error_msg = f"Error getting model info: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def make_prediction(self, input_data: str, selected_model: str = None) -> str:
        """Make predictions using the trained model"""
        if self.inference_only:
            return "Use the Inference Server tab for predictions in inference-only mode."
        
        try:
            # Determine which model to use
            if selected_model and selected_model != "Select a trained model..." and selected_model in self.trained_models:
                # Use selected trained model
                model_info = self.trained_models[selected_model]
                model_name = selected_model
            elif self.training_engine is None:
                return "No model available. Please train a model first or select a trained model."
            else:
                # Use current training engine model
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
            if selected_model and selected_model != "Select a trained model..." and selected_model in self.trained_models:
                # For selected model, we would need to reload it from training engine
                # This is a simplified version - in practice, you might want to store the actual model objects
                if self.training_engine is None:
                    return "Training engine not available. Please retrain the model or use the inference server."
                success, predictions = self.training_engine.predict(input_array)
            else:
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
    
    def save_model(self, model_name: str, selected_model: str, encryption_password: str = "") -> str:
        """Save the trained model"""
        if self.inference_only:
            return "Model saving is not available in inference-only mode."
        
        try:
            # Determine which model to save
            if selected_model and selected_model != "Select a trained model..." and selected_model in self.trained_models:
                save_name = model_name if model_name else selected_model
                model_to_save = selected_model
            elif self.training_engine is None:
                return "No model to save. Please train a model first or select a trained model."
            else:
                save_name = model_name if model_name else "current_model"
                model_to_save = "current"
            
            # Initialize model manager if not already done
            if self.model_manager is None:
                self.model_manager = SecureModelManager(
                    self.current_config,
                    logger=logger,
                    secret_key=encryption_password if encryption_password else None
                )
            
            # Get the best model
            if model_to_save == "current":
                best_model_name, best_model_info = self.training_engine.get_best_model()
                if best_model_info is None:
                    return "No trained model available to save."
            else:
                # For selected models, we use the training engine's current best model
                # In a full implementation, you'd want to store actual model objects
                best_model_name, best_model_info = self.training_engine.get_best_model()
                if best_model_info is None:
                    return "Selected model not available in training engine. Please retrain the model."
            
            # Update model manager with the model
            self.model_manager.models[save_name] = best_model_info
            self.model_manager.best_model = best_model_info
            
            # Save the model
            success = self.model_manager.save_model(
                model_name=save_name,
                access_code=encryption_password if encryption_password else None
            )
            
            if success:
                return f"✅ Model '{save_name}' saved successfully!"
            else:
                return "❌ Failed to save model. Check logs for details."
                
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def load_model(self, file, encryption_password: str = "") -> str:
        """Load a saved model"""
        try:
            if file is None:
                return "No model file selected."
            
            # Initialize model manager if not already done
            if self.model_manager is None:
                config = self.current_config or MLTrainingEngineConfig()
                self.model_manager = SecureModelManager(
                    config,
                    logger=logger,
                    secret_key=encryption_password if encryption_password else None
                )
            
            # Load the model
            model = self.model_manager.load_model(
                filepath=file.name,
                access_code=encryption_password if encryption_password else None
            )
            
            if model is not None:
                # Initialize inference engine with the loaded model
                inference_config = InferenceEngineConfig()
                self.inference_engine = InferenceEngine(inference_config)
                
                return f"✅ Model loaded successfully from {file.name}"
            else:
                return "❌ Failed to load model. Check password and file integrity."
                
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_model_performance(self) -> str:
        """Get performance comparison of all trained models"""
        if self.inference_only:
            return "Performance comparison is not available in inference-only mode."
        
        try:
            if not self.trained_models and self.training_engine is None:
                return "No models trained yet."
            
            # Get performance from training engine if available
            comparison_text = ""
            if self.training_engine:
                try:
                    comparison = self.training_engine.get_performance_comparison()
                    
                    # Check if we have a valid comparison response
                    if isinstance(comparison, dict) and 'models' in comparison and 'error' not in comparison:
                        models = comparison.get('models', [])
                        if isinstance(models, list) and models:
                            comparison_text += "Training Engine Model Comparison:\n\n"
                            for model in models:
                                if isinstance(model, dict):
                                    model_name = model.get('name', 'Unknown')
                                    model_type = model.get('type', 'Unknown')
                                    training_time = model.get('training_time', 0)
                                    is_best = model.get('is_best', False)
                                    metrics = model.get('metrics', {})
                                    
                                    comparison_text += f"### {model_name} {'👑' if is_best else ''}\n"
                                    comparison_text += f"- Type: {model_type}\n"
                                    comparison_text += f"- Training Time: {training_time:.2f}s\n"
                                    
                                    if isinstance(metrics, dict) and metrics:
                                        comparison_text += "- Metrics:\n"
                                        for metric, value in metrics.items():
                                            if isinstance(value, (int, float)):
                                                comparison_text += f"  - {metric}: {value:.4f}\n"
                                    comparison_text += "\n"
                    elif isinstance(comparison, dict) and 'error' in comparison:
                        comparison_text += f"Training Engine Error: {comparison['error']}\n\n"
                        
                except Exception as comparison_error:
                    comparison_text += f"Error getting model comparison: {str(comparison_error)}\n\n"
            
            # Add stored model information
            if self.trained_models:
                comparison_text += "\nStored Trained Models:\n\n"
                for model_name, model_info in self.trained_models.items():
                    comparison_text += f"### {model_name}\n"
                    comparison_text += f"- Algorithm: {model_info['algorithm']}\n"
                    comparison_text += f"- Target: {model_info['target_column']}\n"
                    comparison_text += f"- Training Time: {model_info['training_time']:.2f}s\n"
                    comparison_text += f"- Data Shape: {model_info['data_shape'][0]} × {model_info['data_shape'][1]}\n"
                    
                    if 'metrics' in model_info['result'] and model_info['result']['metrics']:
                        comparison_text += "- Metrics:\n"
                        for metric, value in model_info['result']['metrics'].items():
                            if isinstance(value, (int, float)):
                                comparison_text += f"  - {metric}: {value:.4f}\n"
                    comparison_text += "\n"
            
            return comparison_text if comparison_text else "No model performance data available."
            
        except Exception as e:
            error_msg = f"Error getting model performance: {str(e)}"
            logger.error(error_msg)
            return error_msg

def create_ui(inference_only: bool = False):
    """Create and configure the Gradio interface"""
    
    app = MLSystemUI(inference_only=inference_only)
    
    # Load CSS styles from external file
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
        # Header with logo and improved welcome section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Image(value="assets/logo.png", show_label=False, show_download_button=False, height=100, width=150)
            with gr.Column(scale=4):
                gr.Markdown(f"""
# {title}
{description}
                """)
        
        # Quick start guide for new users
        if not inference_only:
            gr.HTML("""
            <div class="quick-start">
                <h3 style="margin-top: 0;">🚀 Quick Start Guide</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div style="text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 5px;">📁</div>
                        <div><strong>1. Load Data</strong></div>
                        <div style="font-size: 0.9em; opacity: 0.9;">Upload CSV/Excel or try sample datasets</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 5px;">⚙️</div>
                        <div><strong>2. Configure</strong></div>
                        <div style="font-size: 0.9em; opacity: 0.9;">Set task type and parameters</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 5px;">🎯</div>
                        <div><strong>3. Train</strong></div>
                        <div style="font-size: 0.9em; opacity: 0.9;">Select algorithms and train models</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 5px;">📊</div>
                        <div><strong>4. Compare</strong></div>
                        <div style="font-size: 0.9em; opacity: 0.9;">Analyze model performance</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 5px;">🔮</div>
                        <div><strong>5. Predict</strong></div>
                        <div style="font-size: 0.9em; opacity: 0.9;">Make real-time predictions</div>
                    </div>
                </div>
            </div>
            """)
        
        with gr.Tabs() as main_tabs:
            
            if not inference_only:
                # Create shared state for algorithm choices and data storage
                algorithm_choices_state = gr.State([])
                data_state = gr.State()
                
                # 🎯 STEP 1: Data Upload Tab - Improved with better guidance
                with gr.Tab("📁 Step 1: Load Data", id="data_upload"):
                    gr.HTML("""
                    <div class="step-indicator">
                        Step 1: Load Your Dataset - Upload files or try sample datasets
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("""
                            ### 📤 Upload Your Dataset
                            Supported formats: CSV, Excel (.xlsx, .xls), JSON
                            Requirements: Clean data with headers
                            """)
                            
                            file_input = gr.File(
                                label="📁 Select Data File",
                                file_count="single",
                                type="filepath",
                                file_types=[".csv", ".xlsx", ".xls", ".json"]
                            )
                            
                            load_btn = gr.Button("📤 Upload & Analyze Data", variant="primary", size="lg")
                            
                        with gr.Column(scale=1):
                            gr.Markdown("""
                            ### 🎲 Try Sample Datasets
                            Perfect for learning: Pre-loaded datasets for different ML tasks
                            """)
                            
                            # Enhanced sample dataset selection with descriptions
                            sample_descriptions = {
                                "Iris": "🌸 Classic classification: Predict flower species (150 samples, 4 features)",
                                "Boston Housing": "🏠 Regression: Predict house prices (506 samples, 13 features)",
                                "Titanic": "🚢 Classification: Predict passenger survival (891 samples, 12 features)",
                                "Wine Quality": "🍷 Regression: Predict wine quality scores (1599 samples, 11 features)",
                                "Diabetes": "💊 Classification: Predict diabetes onset (768 samples, 8 features)",
                                "Car Evaluation": "🚗 Classification: Predict car acceptability (1728 samples, 6 features)"
                            }
                            
                            sample_dropdown = gr.Dropdown(
                                choices=["🎯 Choose a sample dataset..."] + [f"{name}: {desc}" for name, desc in sample_descriptions.items()],
                                value="🎯 Choose a sample dataset...",
                                label="🎲 Sample Datasets",
                                info="Perfect for testing and learning"
                            )
                            
                            load_sample_btn = gr.Button("🎲 Load Sample Dataset", variant="secondary", size="lg")
                    
                    # Data status and preview
                    with gr.Row():
                        data_info = gr.HTML("""
                        <div class="info-card">
                            <h4>📋 Dataset Status</h4>
                            <p>👆 Upload your own data or select a sample dataset to get started</p>
                            <ul>
                                <li>✅ Supports CSV, Excel, and JSON formats</li>
                                <li>✅ Automatic data type detection</li>
                                <li>✅ Missing value analysis</li>
                                <li>✅ Statistical summaries</li>
                            </ul>
                        </div>
                        """)
                    
                    # Enhanced Data Preview Section with tabs
                    with gr.Row():
                        with gr.Tabs():
                            with gr.Tab("📊 Data Overview"):
                                data_preview = gr.Markdown("📊 Data analysis will appear here after loading...")
                            
                            with gr.Tab("📋 Sample Data"):
                                sample_data_table = gr.HTML("""
                                <div class="info-card">
                                    <p style="text-align: center; color: #666;">
                                        📋 First 10 rows of your dataset will appear here
                                    </p>
                                </div>
                                """)
                
                # 🎯 STEP 2: Configuration Tab - Enhanced with smart defaults
                with gr.Tab("⚙️ Step 2: Configure Training", id="configuration"):
                    gr.HTML("""
                    <div class="step-indicator">
                        Step 2: Configure ML Training Parameters - Set up optimization and algorithms
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎯 Core Settings")
                            
                            task_type = gr.Dropdown(
                                choices=["CLASSIFICATION", "REGRESSION"],
                                value="CLASSIFICATION",
                                label="🎯 Task Type",
                                info="Classification: Predict categories | Regression: Predict numbers"
                            )
                            
                            optimization_strategy = gr.Dropdown(
                                choices=[
                                    ("Random Search (Fast)", "RANDOM_SEARCH"),
                                    ("Grid Search (Thorough)", "GRID_SEARCH"),
                                    ("Bayesian Optimization (Smart)", "BAYESIAN_OPTIMIZATION"),
                                    ("HyperX (Advanced)", "HYPERX")
                                ],
                                value="RANDOM_SEARCH",
                                label="🔍 Optimization Strategy",
                                info="How to find the best model parameters"
                            )
                            
                            cv_folds = gr.Slider(
                                minimum=2,
                                maximum=10,
                                value=5,
                                step=1,
                                label="📊 Cross-Validation Folds",
                                info="More folds = better validation, but slower training"
                            )
                            
                        with gr.Column(scale=1):
                            gr.Markdown("### 📊 Data Settings")
                            
                            test_size = gr.Slider(
                                minimum=0.1,
                                maximum=0.5,
                                value=0.2,
                                step=0.05,
                                label="📊 Test Data Proportion",
                                info="Portion of data reserved for final testing"
                            )
                            
                            random_state = gr.Number(
                                value=42,
                                label="🎲 Random Seed",
                                precision=0,
                                info="For reproducible results (use same number for consistent results)"
                            )
                            
                            normalization = gr.Dropdown(
                                choices=[
                                    ("Standard Scaling (Recommended)", "STANDARD"),
                                    ("Min-Max Scaling (0-1 range)", "MINMAX"),
                                    ("Robust Scaling (Outlier-resistant)", "ROBUST"),
                                    ("No Normalization", "NONE")
                                ],
                                value="STANDARD",
                                label="📏 Data Normalization",
                                info="How to scale your features for better model performance"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🚀 Advanced Options")
                            
                            with gr.Row():
                                enable_feature_selection = gr.Checkbox(
                                    label="✨ Smart Feature Selection",
                                    value=True,
                                    info="Automatically identify the most important features"
                                )
                                
                                enable_quantization = gr.Checkbox(
                                    label="⚡ Model Compression",
                                    value=False,
                                    info="Reduce model size for faster inference"
                                )
                            
                            optimization_mode = gr.Dropdown(
                                choices=[
                                    ("Balanced (Recommended)", "BALANCED"),
                                    ("Performance Focus", "PERFORMANCE"),
                                    ("Memory Saving", "MEMORY_SAVING"),
                                    ("Conservative", "CONSERVATIVE")
                                ],
                                value="BALANCED",
                                label="⚖️ Optimization Mode",
                                info="Balance between speed, accuracy, and resource usage"
                            )
                    
                    with gr.Row():
                        config_btn = gr.Button("⚙️ Create Training Configuration", variant="primary", size="lg")
                        
                    config_output = gr.HTML("""
                    <div class="info-card">
                        <h4>📋 Configuration Status</h4>
                        <p>👆 Click "Create Training Configuration" to set up your ML pipeline</p>
                    </div>
                    """)
                    
                    # Algorithm selection display (updated based on task type)
                    algorithm_dropdown = gr.Dropdown(
                        choices=[],
                        label="🤖 Available ML Algorithms",
                        info="Create configuration first to see available algorithms for your task type",
                        interactive=False
                    )
                
                # 🎯 STEP 3: Training Tab - Enhanced with progress tracking
                with gr.Tab("🎯 Step 3: Train Models", id="training"):
                    gr.HTML("""
                    <div class="step-indicator">
                        Step 3: Train ML Models - Select algorithms and start training
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎯 Training Setup")
                            
                            target_column = gr.Textbox(
                                label="🎯 Target Column Name",
                                placeholder="e.g., price, category, outcome",
                                info="The column you want to predict (must match exactly)",
                                lines=1
                            )
                            
                            # Enhanced algorithm selection
                            algorithm_selection = gr.Dropdown(
                                choices=["⚙️ Configure training parameters first..."],
                                value="⚙️ Configure training parameters first...",
                                label="🤖 Select ML Algorithm",
                                info="Complete Step 2 (Configuration) to see available algorithms",
                                interactive=False
                            )
                            
                            model_name_input = gr.Textbox(
                                label="📝 Model Name (Optional)",
                                placeholder="e.g., my_iris_classifier",
                                info="Leave empty for auto-generated name with timestamp",
                                lines=1
                            )
                            
                            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                            
                        with gr.Column(scale=2):
                            training_output = gr.HTML("""
                            <div class="info-card">
                                <h4>🎯 Training Status</h4>
                                <p><strong>Ready to train!</strong> Follow these steps:</p>
                                <ol>
                                    <li>✅ Load your data in Step 1</li>
                                    <li>✅ Configure parameters in Step 2</li>
                                    <li>🎯 Enter your target column name</li>
                                    <li>🤖 Select an ML algorithm</li>
                                    <li>🚀 Click "Start Training"</li>
                                </ol>
                            </div>
                            """)
                    
                    # Results display with better organization
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📊 Model Performance")
                            metrics_output = gr.HTML("""
                            <div class="info-card">
                                <p style="text-align: center; color: #666;">
                                    📊 Training metrics will appear here
                                </p>
                            </div>
                            """)
                            
                        with gr.Column():
                            gr.Markdown("### 🏆 Feature Importance")
                            importance_output = gr.HTML("""
                            <div class="info-card">
                                <p style="text-align: center; color: #666;">
                                    🏆 Feature importance rankings will appear here
                                </p>
                            </div>
                            """)
                    
                    # Enhanced trained models management
                    with gr.Row():
                        with gr.Column(scale=2):
                            trained_models_dropdown = gr.Dropdown(
                                choices=app.get_trained_model_list(),
                                value="📋 Select a trained model...",
                                label="📋 Your Trained Models",
                                info="All your trained models appear here"
                            )
                            
                        with gr.Column(scale=1):
                            model_info_btn = gr.Button("📋 View Model Details", variant="secondary")
                    
                    # Model information display
                    model_info_output = gr.HTML("""
                    <div class="info-card">
                        <h4>📋 Model Information</h4>
                        <p>Select a trained model and click "View Model Details" to see comprehensive information including performance metrics, training time, and feature details.</p>
                    </div>
                    """)
                
                # Enhanced event handlers for better UX
                
                # Data upload tab event handlers with better feedback
                def load_data_enhanced(file):
                    if file is None:
                        return gr.HTML("""
                        <div class="error-card">
                            <h4>❌ No File Selected</h4>
                            <p>Please select a data file to upload.</p>
                        </div>
                        """), {}, gr.Markdown("Please upload a file first."), gr.HTML("")
                    
                    try:
                        result = app.load_data(file)
                        if isinstance(result, tuple) and len(result) >= 4:
                            info, data_state, preview, table = result
                            if "Error" in info:
                                return gr.HTML(f"""
                                <div class="error-card">
                                    <h4>❌ Upload Failed</h4>
                                    <p>{info}</p>
                                </div>
                                """), data_state, preview, table
                            else:
                                return gr.HTML(f"""
                                <div class="success-card">
                                    <h4>✅ Data Loaded Successfully!</h4>
                                    <p>{info}</p>
                                </div>
                                """), data_state, preview, table
                        else:
                            return gr.HTML("""
                            <div class="error-card">
                                <h4>❌ Unexpected Error</h4>
                                <p>Please try again or contact support.</p>
                            </div>
                            """), {}, gr.Markdown("Error occurred"), gr.HTML("")
                    except Exception as e:
                        return gr.HTML(f"""
                        <div class="error-card">
                            <h4>❌ Upload Error</h4>
                            <p>Error: {str(e)}</p>
                        </div>
                        """), {}, gr.Markdown("Error occurred"), gr.HTML("")
                
                def load_sample_data_enhanced(dataset_selection):
                    if dataset_selection.startswith("🎯"):
                        return gr.HTML("""
                        <div class="info-card">
                            <h4>📋 Dataset Status</h4>
                            <p>👆 Select a sample dataset from the dropdown to get started</p>
                        </div>
                        """), {}, gr.Markdown("Please select a dataset"), gr.HTML("")
                    
                    # Extract dataset name from the formatted string
                    dataset_name = dataset_selection.split(":")[0]
                    try:
                        result = app.load_sample_data(dataset_name)
                        if isinstance(result, tuple) and len(result) >= 4:
                            info, data_state, preview, table = result
                            return gr.HTML(f"""
                            <div class="success-card">
                                <h4>✅ Sample Dataset Loaded!</h4>
                                <div>{info}</div>
                            </div>
                            """), data_state, preview, table
                        else:
                            return gr.HTML("""
                            <div class="error-card">
                                <h4>❌ Loading Failed</h4>
                                <p>Could not load the selected dataset.</p>
                            </div>
                            """), {}, gr.Markdown("Error occurred"), gr.HTML("")
                    except Exception as e:
                        return gr.HTML(f"""
                        <div class="error-card">
                            <h4>❌ Loading Error</h4>
                            <p>Error: {str(e)}</p>
                        </div>
                        """), {}, gr.Markdown("Error occurred"), gr.HTML("")
                
                load_btn.click(
                    fn=load_data_enhanced,
                    inputs=[file_input],
                    outputs=[data_info, data_state, data_preview, sample_data_table]
                )
                
                load_sample_btn.click(
                    fn=load_sample_data_enhanced,
                    inputs=[sample_dropdown],
                    outputs=[data_info, data_state, data_preview, sample_data_table]
                )
                
                # Enhanced configuration handler
                def create_config_enhanced(*args):
                    try:
                        result = app.create_training_config(*args)
                        if isinstance(result, tuple) and len(result) >= 3:
                            config_text, algorithm_dropdown_update, algorithm_choices = result
                            if "Error" in config_text:
                                return gr.HTML(f"""
                                <div class="error-card">
                                    <h4>❌ Configuration Failed</h4>
                                    <p>{config_text}</p>
                                </div>
                                """), algorithm_dropdown_update, algorithm_choices
                            else:
                                return gr.HTML(f"""
                                <div class="success-card">
                                    <h4>✅ Configuration Created!</h4>
                                    <div>{config_text}</div>
                                </div>
                                """), algorithm_dropdown_update, algorithm_choices
                        else:
                            return gr.HTML("""
                            <div class="error-card">
                                <h4>❌ Configuration Error</h4>
                                <p>Please check your settings and try again.</p>
                            </div>
                            """), gr.Dropdown(), []
                    except Exception as e:
                        return gr.HTML(f"""
                        <div class="error-card">
                            <h4>❌ Configuration Error</h4>
                            <p>Error: {str(e)}</p>
                        </div>
                        """), gr.Dropdown(), []
                
                # Configuration tab event handlers
                config_btn.click(
                    fn=create_config_enhanced,
                    inputs=[
                        task_type, optimization_strategy, cv_folds, test_size,
                        random_state, enable_feature_selection, normalization,
                        enable_quantization, optimization_mode
                    ],
                    outputs=[config_output, algorithm_dropdown, algorithm_choices_state]
                )
                
                # Connect the algorithm dropdown updates to the training tab
                def update_training_algorithms(task_type_value):
                    algorithms = app.get_algorithms_for_task(task_type_value)
                    return gr.Dropdown(choices=algorithms, value=algorithms[0] if algorithms else None)
                
                # Update algorithm dropdown when task type changes
                task_type.change(
                    fn=update_training_algorithms,
                    inputs=[task_type],
                    outputs=[algorithm_dropdown]
                )
                
                # Function to update training tab algorithm dropdown
                def update_algorithm_choices_from_state(algorithm_list):
                    if algorithm_list and len(algorithm_list) > 0:
                        return gr.Dropdown(
                            choices=algorithm_list,
                            value=algorithm_list[0],
                            label="🤖 Select ML Algorithm",
                            info="Choose from available algorithms for your task type",
                            interactive=True
                        )
                    else:
                        return gr.Dropdown(
                            choices=["⚙️ Configure training parameters first..."],
                            value="⚙️ Configure training parameters first...",
                            label="🤖 Select ML Algorithm",
                            info="Complete Step 2 (Configuration) to see available algorithms",
                            interactive=False
                        )
                
                # Update the training tab dropdown when algorithm choices state changes
                algorithm_choices_state.change(
                    fn=update_algorithm_choices_from_state,
                    inputs=[algorithm_choices_state],
                    outputs=[algorithm_selection]
                )
                
                # Enhanced training handler
                def train_model_enhanced(target_col, algorithm, model_name):
                    if not target_col:
                        return gr.HTML("""
                        <div class="error-card">
                            <h4>❌ Missing Target Column</h4>
                            <p>Please enter the name of your target column.</p>
                        </div>
                        """), gr.HTML(""), gr.HTML(""), gr.Dropdown()
                    
                    if algorithm.startswith("⚙️"):
                        return gr.HTML("""
                        <div class="error-card">
                            <h4>❌ No Algorithm Selected</h4>
                            <p>Please complete Step 2 (Configuration) and select an algorithm.</p>
                        </div>
                        """), gr.HTML(""), gr.HTML(""), gr.Dropdown()
                    
                    try:
                        # Show training in progress
                        progress_html = gr.HTML("""
                        <div class="info-card">
                            <h4>🚀 Training In Progress...</h4>
                            <p>Please wait while we train your model. This may take a few minutes.</p>
                            <div style="margin: 10px 0;">
                                <div style="background: #f0f0f0; border-radius: 10px; overflow: hidden;">
                                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 20px; width: 0%; animation: progress 2s infinite;"></div>
                                </div>
                            </div>
                        </div>
                        """)
                        
                        result = app.train_model(target_col, algorithm, model_name)
                        if isinstance(result, tuple) and len(result) >= 4:
                            summary, metrics, importance, models_dropdown = result
                            if "Error" in summary:
                                return gr.HTML(f"""
                                <div class="error-card">
                                    <h4>❌ Training Failed</h4>
                                    <div>{summary}</div>
                                </div>
                                """), gr.HTML(f"""
                                <div class="error-card">
                                    <p>Training was unsuccessful. Please check your data and configuration.</p>
                                </div>
                                """), gr.HTML(""), models_dropdown
                            else:
                                return gr.HTML(f"""
                                <div class="success-card">
                                    <h4>✅ Training Completed Successfully!</h4>
                                    <div>{summary}</div>
                                </div>
                                """), gr.HTML(f"""
                                <div class="metric-box">
                                    <h4>📊 Performance Metrics</h4>
                                    <div>{metrics}</div>
                                </div>
                                """), gr.HTML(f"""
                                <div class="metric-box">
                                    <h4>🏆 Feature Importance</h4>
                                    <div>{importance}</div>
                                </div>
                                """), models_dropdown
                        else:
                            return gr.HTML("""
                            <div class="error-card">
                                <h4>❌ Training Error</h4>
                                <p>An unexpected error occurred during training.</p>
                            </div>
                            """), gr.HTML(""), gr.HTML(""), gr.Dropdown()
                    except Exception as e:
                        return gr.HTML(f"""
                        <div class="error-card">
                            <h4>❌ Training Error</h4>
                            <p>Error: {str(e)}</p>
                        </div>
                        """), gr.HTML(""), gr.HTML(""), gr.Dropdown()
                
                # Training tab event handlers
                train_btn.click(
                    fn=train_model_enhanced,
                    inputs=[target_column, algorithm_selection, model_name_input],
                    outputs=[training_output, metrics_output, importance_output, trained_models_dropdown]
                )
                
                def get_model_info_enhanced(model_name):
                    try:
                        result = app.get_trained_model_info(model_name)
                        return gr.HTML(f"""
                        <div class="metric-box">
                            <h4>📋 Model Details</h4>
                            <div>{result}</div>
                        </div>
                        """)
                    except Exception as e:
                        return gr.HTML(f"""
                        <div class="error-card">
                            <h4>❌ Error</h4>
                            <p>Could not retrieve model information: {str(e)}</p>
                        </div>
                        """)
                
                model_info_btn.click(
                    fn=get_model_info_enhanced,
                    inputs=[trained_models_dropdown],
                    outputs=[model_info_output]
                )
                
                # 🎯 STEP 4: Predictions Tab - Enhanced with better UX
                with gr.Tab("🔮 Step 4: Make Predictions", id="predictions"):
                    gr.HTML("""
                    <div class="step-indicator">
                        Step 4: Make Real-time Predictions - Test your trained models
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 🔮 Prediction Input")
                            
                            prediction_input = gr.Textbox(
                                label="📊 Input Data",
                                placeholder="Example: 5.1, 3.5, 1.4, 0.2\nOr JSON: [5.1, 3.5, 1.4, 0.2]",
                                lines=4,
                                info="Enter feature values in the same order as your training data"
                            )
                            
                            # Enhanced model selection
                            prediction_model_dropdown = gr.Dropdown(
                                choices=app.get_trained_model_list(),
                                value="📋 Select a trained model...",
                                label="🤖 Select Model",
                                info="Choose which trained model to use for prediction"
                            )
                            
                            with gr.Row():
                                predict_btn = gr.Button("🔮 Make Prediction", variant="primary", size="lg")
                                clear_input_btn = gr.Button("🗑️ Clear Input", variant="secondary")
                            
                        with gr.Column(scale=2):
                            gr.Markdown("### 📊 Prediction Results")
                            prediction_output = gr.HTML("""
                            <div class="info-card">
                                <h4>🔮 Ready for Predictions</h4>
                                <p><strong>How to make predictions:</strong></p>
                                <ol>
                                    <li>✅ Train at least one model in Step 3</li>
                                    <li>📊 Enter your input data (feature values)</li>
                                    <li>🤖 Select a trained model</li>
                                    <li>🔮 Click "Make Prediction"</li>
                                </ol>
                                <div style="margin-top: 15px; padding: 10px; background: #f0f8ff; color: #1e40af; border-radius: 6px;">
                                    <strong>💡 Tip:</strong> Make sure your input features are in the same order and scale as your training data.
                                </div>
                            </div>
                            """)
                    
                    # Enhanced prediction handlers
                    def make_prediction_enhanced(input_data, selected_model):
                        if not input_data.strip():
                            return gr.HTML("""
                            <div class="error-card">
                                <h4>❌ No Input Data</h4>
                                <p>Please enter the feature values for prediction.</p>
                            </div>
                            """)
                        
                        if selected_model.startswith("📋"):
                            return gr.HTML("""
                            <div class="error-card">
                                <h4>❌ No Model Selected</h4>
                                <p>Please select a trained model for prediction.</p>
                            </div>
                            """)
                        
                        try:
                            result = app.make_prediction(input_data, selected_model)
                            if "Error" in result:
                                return gr.HTML(f"""
                                <div class="error-card">
                                    <h4>❌ Prediction Failed</h4>
                                    <p>{result}</p>
                                </div>
                                """)
                            else:
                                return gr.HTML(f"""
                                <div class="success-card">
                                    <h4>✅ Prediction Successful!</h4>
                                    <div style="margin-top: 15px;">
                                        {result}
                                    </div>
                                </div>
                                """)
                        except Exception as e:
                            return gr.HTML(f"""
                            <div class="error-card">
                                <h4>❌ Prediction Error</h4>
                                <p>Error: {str(e)}</p>
                            </div>
                            """)
                    
                    predict_btn.click(
                        fn=make_prediction_enhanced,
                        inputs=[prediction_input, prediction_model_dropdown],
                        outputs=[prediction_output]
                    )
                    
                    clear_input_btn.click(
                        fn=lambda: "",
                        outputs=[prediction_input]
                    )
                
                # 🎯 STEP 5: Model Comparison Tab - Enhanced performance analysis
                with gr.Tab("📊 Step 5: Compare Models", id="performance"):
                    gr.HTML("""
                    <div class="step-indicator">
                        Step 5: Model Performance Analysis - Compare and choose the best model
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📊 Performance Analysis")
                            performance_btn = gr.Button("📊 Generate Performance Report", variant="primary", size="lg")
                            
                            gr.Markdown("""
                            📈 What you'll see:
                            - Model accuracy comparisons
                            - Training time analysis  
                            - Feature importance rankings
                            - Recommendations for best model
                            """)
                            
                        with gr.Column(scale=2):
                            performance_output = gr.HTML("""
                            <div class="info-card">
                                <h4>📊 Performance Dashboard</h4>
                                <p><strong>Ready to analyze your models!</strong></p>
                                <p>Click "Generate Performance Report" to see:</p>
                                <ul>
                                    <li>📈 Accuracy metrics for each model</li>
                                    <li>⏱️ Training time comparisons</li>
                                    <li>🏆 Feature importance analysis</li>
                                    <li>💡 Best model recommendations</li>
                                </ul>
                                <div style="margin-top: 15px; padding: 10px; background: #f0f8ff; color: #1e40af; border-radius: 6px;">
                                    <strong>💡 Note:</strong> Train multiple models in Step 3 to see meaningful comparisons.
                                </div>
                            </div>
                            """)
                    
                    def get_performance_enhanced():
                        try:
                            result = app.get_model_performance()
                            if "Error" in result or "No models" in result:
                                return gr.HTML(f"""
                                <div class="info-card">
                                    <h4>📊 Performance Report</h4>
                                    <p>{result}</p>
                                    <div style="margin-top: 15px; padding: 10px; background: #fff3cd; color: #856404; border-radius: 6px;">
                                        <strong>💡 Tip:</strong> Train multiple models in Step 3 to see performance comparisons.
                                    </div>
                                </div>
                                """)
                            else:
                                return gr.HTML(f"""
                                <div class="metric-box">
                                    <h4>📊 Model Performance Report</h4>
                                    <div>{result}</div>
                                </div>
                                """)
                        except Exception as e:
                            return gr.HTML(f"""
                            <div class="error-card">
                                <h4>❌ Report Generation Failed</h4>
                                <p>Error: {str(e)}</p>
                            </div>
                            """)
                    
                    performance_btn.click(
                        fn=get_performance_enhanced,
                        outputs=[performance_output]
                    )
                
                # 💾 Model Management Tab - Enhanced with better organization
                with gr.Tab("💾 Model Management", id="model_management"):
                    gr.Markdown("### 💾 Save and Load Models")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("""
                            <div class="workflow-step">
                                <h4>💾 Save Trained Models</h4>
                                <p>Securely save your best performing models for future use</p>
                            </div>
                            """)
                            
                            save_model_name = gr.Textbox(
                                label="📝 Model Name",
                                placeholder="e.g., best_iris_classifier_v1",
                                info="Choose a descriptive name for your model"
                            )
                            
                            save_model_dropdown = gr.Dropdown(
                                choices=app.get_trained_model_list(),
                                value="📋 Select a trained model...",
                                label="🤖 Select Model to Save",
                                info="Choose which trained model to save permanently"
                            )
                            
                            save_password = gr.Textbox(
                                label="🔐 Encryption Password (Optional)",
                                type="password",
                                placeholder="Leave empty for no encryption",
                                info="Add password protection for sensitive models"
                            )
                            
                            save_btn = gr.Button("💾 Save Model", variant="primary", size="lg")
                            save_output = gr.HTML("""
                            <div class="info-card">
                                <h4>💾 Save Status</h4>
                                <p>Select a model and click "Save Model" to store it permanently.</p>
                            </div>
                            """)
                            
                        with gr.Column():
                            gr.HTML("""
                            <div class="workflow-step">
                                <h4>📂 Load Saved Models</h4>
                                <p>Load previously saved models for inference or further training</p>
                            </div>
                            """)
                            
                            load_file = gr.File(
                                label="📁 Select Model File",
                                file_count="single",
                                type="filepath",
                                file_types=[".pkl", ".joblib", ".model"]
                            )
                            
                            load_password = gr.Textbox(
                                label="🔐 Decryption Password",
                                type="password",
                                placeholder="Enter password if model is encrypted",
                                info="Only needed for encrypted models"
                            )
                            
                            load_btn = gr.Button("📂 Load Model", variant="secondary", size="lg")
                            load_output = gr.HTML("""
                            <div class="info-card">
                                <h4>📂 Load Status</h4>
                                <p>Select a model file and click "Load Model" to import it.</p>
                            </div>
                            """)
                    
                    # Enhanced save/load handlers
                    def save_model_enhanced(model_name, selected_model, password):
                        if not model_name:
                            return gr.HTML("""
                            <div class="error-card">
                                <h4>❌ Missing Model Name</h4>
                                <p>Please enter a name for your model.</p>
                            </div>
                            """)
                        
                        if selected_model.startswith("📋"):
                            return gr.HTML("""
                            <div class="error-card">
                                <h4>❌ No Model Selected</h4>
                                <p>Please select a trained model to save.</p>
                            </div>
                            """)
                        
                        try:
                            result = app.save_model(model_name, selected_model, password)
                            if "✅" in result:
                                return gr.HTML(f"""
                                <div class="success-card">
                                    <h4>✅ Model Saved Successfully!</h4>
                                    <p>{result}</p>
                                </div>
                                """)
                            else:
                                return gr.HTML(f"""
                                <div class="error-card">
                                    <h4>❌ Save Failed</h4>
                                    <p>{result}</p>
                                </div>
                                """)
                        except Exception as e:
                            return gr.HTML(f"""
                            <div class="error-card">
                                <h4>❌ Save Error</h4>
                                <p>Error: {str(e)}</p>
                            </div>
                            """)
                    
                    def load_model_enhanced(file, password):
                        if file is None:
                            return gr.HTML("""
                            <div class="error-card">
                                <h4>❌ No File Selected</h4>
                                <p>Please select a model file to load.</p>
                            </div>
                            """)
                        
                        try:
                            result = app.load_model(file, password)
                            if "✅" in result:
                                return gr.HTML(f"""
                                <div class="success-card">
                                    <h4>✅ Model Loaded Successfully!</h4>
                                    <p>{result}</p>
                                </div>
                                """)
                            else:
                                return gr.HTML(f"""
                                <div class="error-card">
                                    <h4>❌ Load Failed</h4>
                                    <p>{result}</p>
                                </div>
                                """)
                        except Exception as e:
                            return gr.HTML(f"""
                            <div class="error-card">
                                <h4>❌ Load Error</h4>
                                <p>Error: {str(e)}</p>
                            </div>
                            """)
                    
                    save_btn.click(
                        fn=save_model_enhanced,
                        inputs=[save_model_name, save_model_dropdown, save_password],
                        outputs=[save_output]
                    )
                    
                    load_btn.click(
                        fn=load_model_enhanced,
                        inputs=[load_file, load_password],
                        outputs=[load_output]
                    )
                
                # 🖥️ System Info Tab - Enhanced with useful information
                with gr.Tab("🖥️ System Information", id="system_info"):
                    gr.Markdown("### 🖥️ System Resources and Optimization")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            system_btn = gr.Button("🖥️ Check System Resources", variant="primary")
                            
                            gr.Markdown("""
                            📊 System Information Includes:
                            - CPU and memory usage
                            - Available disk space
                            - GPU information (if available)
                            - Python environment details
                            - Optimization recommendations
                            """)
                            
                        with gr.Column(scale=2):
                            system_output = gr.JSON(
                                label="🖥️ System Information",
                                value={"info": "Click 'Check System Resources' to view system information"}
                            )
                    
                    def get_system_info_enhanced():
                        try:
                            result = app.get_system_info()
                            return result
                        except Exception as e:
                            return {"error": f"Could not retrieve system information: {str(e)}"}
                    
                    system_btn.click(
                        fn=get_system_info_enhanced,
                        outputs=[system_output]
                    )
            
            # Inference Server Tab (always available)
            with gr.Tab("🔧 Inference Server", id="inference_server"):
                with gr.Row():
                    # Left Panel - Server Control
                    with gr.Column(scale=2):
                        gr.HTML("""
                        <div class="inference-container">
                            <h3 style="margin-top: 0;">🚀 ML Inference Server</h3>
                            <p style="opacity: 0.9;">Professional machine learning inference server for real-time predictions</p>
                        </div>
                        """)
                        
                        # Server Controls
                        gr.HTML('<div class="server-controls">')
                        gr.Markdown("### Server Controls")
                        
                        with gr.Row():
                            start_server_btn = gr.Button("▶️ Start Server", variant="primary")
                            stop_server_btn = gr.Button("⏹️ Stop Server", variant="secondary")
                            load_model_btn = gr.Button("📁 Load Model", variant="secondary")
                        
                        # Model Loading Section
                        with gr.Accordion("Model Loading", open=False):
                            inference_model_file = gr.File(
                                label="Select Model File",
                                file_count="single",
                                type="filepath"
                            )
                            
                            inference_password = gr.Textbox(
                                label="Decryption Password",
                                type="password",
                                placeholder="Leave empty if model is not encrypted",
                                container=False
                            )
                            
                            load_inference_btn = gr.Button("🔧 Load Model", variant="primary")
                            
                            # Model loading status
                            model_load_status = gr.Markdown("No model loaded yet...")
                        
                        gr.HTML('</div>')
                        
                        # Prediction Interface
                        gr.HTML('<div class="server-controls">')
                        gr.Markdown("### Make Predictions")
                        
                        inference_input = gr.Textbox(
                            label="Input Data",
                            placeholder="Enter comma-separated values or JSON array",
                            lines=4,
                            info="Example: 1.5, 2.3, 0.8, 1.1 or [1.5, 2.3, 0.8, 1.1]"
                        )
                        
                        with gr.Row():
                            inference_predict_btn = gr.Button("🎯 Predict", variant="primary")
                            clear_input_btn = gr.Button("🗑️ Clear", variant="secondary")
                        
                        inference_output = gr.Markdown("Prediction Results:\n\nLoad a model and enter input data to make predictions...")
                        gr.HTML('</div>')
                    
                    # Right Panel - Server Status & Logs
                    with gr.Column(scale=1):
                        # Server Status
                        gr.HTML("""
                        <div class="server-status">
                            <h4 style="margin-top: 0; color: white;">Server Status</h4>
                            <div style="margin: 10px 0;">
                                <span class="status-indicator status-stopped"></span>
                                <span style="color: #ffcccb;">Stopped</span>
                            </div>
                        </div>
                        """)
                        
                        server_status_display = gr.HTML("""
                        <div class="model-info-card">
                            <h4 style="margin-top: 0;">📊 Server Metrics</h4>
                            <div class="server-metric">
                                <span class="metric-label">Server Port:</span>
                                <span class="metric-value">8080</span>
                            </div>
                            <div class="server-metric">
                                <span class="metric-label">Context Size:</span>
                                <span class="metric-value">4125</span>
                            </div>
                            <div class="server-metric">
                                <span class="metric-label">Keep Alive:</span>
                                <span class="metric-value">2048</span>
                            </div>
                            <div class="server-metric">
                                <span class="metric-label">GPU Layers:</span>
                                <span class="metric-value">100</span>
                            </div>
                            <div class="server-metric">
                                <span class="metric-label">Parallel:</span>
                                <span class="metric-value">1</span>
                            </div>
                            <div class="server-metric">
                                <span class="metric-label">Batch Size:</span>
                                <span class="metric-value">4096</span>
                            </div>
                        </div>
                        """)
                        
                        # Server Configuration
                        with gr.Accordion("⚙️ Configuration", open=False):
                            server_port = gr.Number(
                                label="Server Port",
                                value=8080,
                                precision=0
                            )
                            
                            memory_lock = gr.Checkbox(
                                label="Memory Lock",
                                value=True
                            )
                            
                            continuous_batching = gr.Checkbox(
                                label="Continuous Batching",
                                value=True
                            )
                            
                            warmup = gr.Checkbox(
                                label="Warmup",
                                value=False
                            )
                        
                        # Loaded Models Section
                        gr.HTML("""
                        <div class="model-info-card">
                            <h4 style="margin-top: 0;">📚 Loaded Models</h4>
                            <p style="color: #666; font-style: italic;">No models loaded.</p>
                        </div>
                        """)
                        
                        loaded_models_display = gr.HTML("")
                
                # Server Logs Section
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📋 Server Logs")
                        server_logs = gr.HTML("""
                        <div class="server-logs">
Server logs will be displayed here.
                        </div>
                        """)
                        
                        with gr.Row():
                            refresh_logs_btn = gr.Button("🔄 Refresh Logs", variant="secondary")
                            clear_logs_btn = gr.Button("🗑️ Clear Logs", variant="secondary")
                
                # Event handlers for inference server
                def update_server_status(is_running=False):
                    if is_running:
                        status_html = """
                        <div class="server-status">
                            <h4 style="margin-top: 0; color: white;">Server Status</h4>
                            <div style="margin: 10px 0;">
                                <span class="status-indicator status-running"></span>
                                <span style="color: #90EE90;">Running</span>
                            </div>
                        </div>
                        """
                        logs = """
                        <div class="server-logs">
[INFO] Server started successfully on port 8080
[INFO] Model loaded and ready for inference
[INFO] Waiting for incoming requests...
                        </div>
                        """
                    else:
                        status_html = """
                        <div class="server-status">
                            <h4 style="margin-top: 0; color: white;">Server Status</h4>
                            <div style="margin: 10px 0;">
                                <span class="status-indicator status-stopped"></span>
                                <span style="color: #ffcccb;">Stopped</span>
                            </div>
                        </div>
                        """
                        logs = """
                        <div class="server-logs">
[INFO] Server stopped
[INFO] Resources cleaned up
                        </div>
                        """
                    return status_html, logs
                
                def update_loaded_models(model_name=None):
                    if model_name:
                        return f"""
                        <div class="model-info-card">
                            <h4 style="margin-top: 0;">📚 Loaded Models</h4>
                            <div style="padding: 10px; background: #f0f8ff; color: #1e40af; border-radius: 4px; margin: 5px 0;">
                                <strong>{model_name}</strong>
                                <br><small style="color: #666;">Status: Ready</small>
                            </div>
                        </div>
                        """
                    else:
                        return """
                        <div class="model-info-card">
                            <h4 style="margin-top: 0;">📚 Loaded Models</h4>
                            <p style="color: #666; font-style: italic;">No models loaded.</p>
                        </div>
                        """
                
                def clear_input():
                    return ""
                
                def simulate_start_server():
                    return update_server_status(True)
                
                def simulate_stop_server():
                    return update_server_status(False)
                
                # Enhanced load model function
                def load_model_enhanced(file, password=""):
                    result = app.load_inference_model(file, password)
                    if "✅" in result:
                        model_name = file.name.split('/')[-1] if file else "Unknown Model"
                        models_html = update_loaded_models(model_name)
                        logs = f"""
                        <div class="server-logs">
[INFO] Loading model: {model_name}
[INFO] Model validation passed
[INFO] Model loaded successfully
[INFO] Ready for inference requests
                        </div>
                        """
                        return result, models_html, logs
                    else:
                        logs = f"""
                        <div class="server-logs">
[ERROR] Failed to load model
[ERROR] {result}
                        </div>
                        """
                        return result, update_loaded_models(), logs
                
                # Button event handlers
                start_server_btn.click(
                    fn=simulate_start_server,
                    outputs=[server_status_display, server_logs]
                )
                
                stop_server_btn.click(
                    fn=simulate_stop_server,
                    outputs=[server_status_display, server_logs]
                )
                
                load_inference_btn.click(
                    fn=load_model_enhanced,
                    inputs=[inference_model_file, inference_password],
                    outputs=[model_load_status, loaded_models_display, server_logs]
                )
                
                inference_predict_btn.click(
                    fn=app.make_inference_prediction,
                    inputs=[inference_input],
                    outputs=[inference_output]
                )
                
                clear_input_btn.click(
                    fn=clear_input,
                    outputs=[inference_input]
                )
                
                clear_logs_btn.click(
                    fn=lambda: """<div class="server-logs">Logs cleared.</div>""",
                    outputs=[server_logs]
                )
        
        # Enhanced footer with helpful information
        gr.HTML("""
        <div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px;">
        """ + ("""
                <div>
                    <h3 style="margin-top: 0;">🚀 Quick Start Guide</h3>
                    <ol style="margin: 0;">
                        <li><strong>Load Data:</strong> Upload CSV/Excel or try sample datasets</li>
                        <li><strong>Configure:</strong> Set task type and optimization parameters</li>
                        <li><strong>Train:</strong> Select algorithms and train multiple models</li>
                        <li><strong>Compare:</strong> Analyze performance metrics</li>
                        <li><strong>Predict:</strong> Make real-time predictions</li>
                    </ol>
                </div>
                <div>
                    <h3 style="margin-top: 0;">🤖 Available Algorithms</h3>
                    <div style="font-size: 0.9em;">
                        <strong>Tree-Based:</strong> Random Forest, XGBoost, LightGBM<br>
                        <strong>Linear:</strong> Logistic/Linear Regression, Ridge, Lasso<br>
                        <strong>Advanced:</strong> SVM, Neural Networks, Ensemble Methods<br>
                        <strong>Specialized:</strong> Naive Bayes, KNN, CatBoost
                    </div>
                </div>
                <div>
                    <h3 style="margin-top: 0;">💡 Pro Tips</h3>
                    <ul style="margin: 0; font-size: 0.9em;">
                        <li>Try sample datasets for quick testing</li>
                        <li>Use feature selection for better performance</li>
                        <li>Compare multiple algorithms</li>
                        <li>Save your best models securely</li>
                        <li>Check system resources for optimization</li>
                    </ul>
                </div>
                <div>
                    <h3 style="margin-top: 0;">🔧 Features</h3>
                    <ul style="margin: 0; font-size: 0.9em;">
                        <li>✅ 20+ ML algorithms</li>
                        <li>✅ Automatic hyperparameter tuning</li>
                        <li>✅ Model performance comparison</li>
                        <li>✅ Secure model encryption</li>
                        <li>✅ Real-time inference server</li>
                        <li>✅ Advanced optimization</li>
                    </ul>
                </div>
        """ if not inference_only else """
                <div>
                    <h3 style="margin-top: 0;">🚀 Inference Server</h3>
                    <p style="margin: 0;">High-performance ML inference with enterprise-grade security and optimization.</p>
                </div>
                <div>
                    <h3 style="margin-top: 0;">💡 Quick Guide</h3>
                    <ol style="margin: 0;">
                        <li>Load your trained model</li>
                        <li>Start the inference server</li>
                        <li>Make real-time predictions</li>
                    </ol>
                </div>
                <div>
                    <h3 style="margin-top: 0;">🔧 Features</h3>
                    <ul style="margin: 0; font-size: 0.9em;">
                        <li>✅ Real-time predictions</li>
                        <li>✅ Model encryption support</li>
                        <li>✅ Optimized performance</li>
                        <li>✅ Enterprise security</li>
                    </ul>
                </div>
        """) + """
            </div>
            <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.2);">
                <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">
                    <strong>Kolosal AutoML Platform</strong> - Powered by advanced machine learning and optimization
                </p>
            </div>
        </div>
        """)
    
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
        "--model-path",
        type=str,
        help="Path to pre-trained model file (for inference-only mode)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address (default: 0.0.0.0)"
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
    
    args = parser.parse_args()
    
    # Create and launch the interface
    interface = create_ui(inference_only=args.inference_only)
    
    print(f"""
🚀 Starting {'ML Inference Server' if args.inference_only else 'ML Training & Inference System'}

Mode: {'Inference Only' if args.inference_only else 'Full Training & Inference'}
Host: {args.host}
Port: {args.port}
Share: {'Yes' if args.share else 'No'}

Available Features:
{'- Real-time model inference' if args.inference_only else '''- Multiple ML algorithms support
- Advanced model training with hyperparameter optimization
- Model performance comparison
- Secure model storage with encryption
- Real-time inference server'''}
    """)
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()