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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
## 📊 **Dataset Overview**

- **Shape**: {summary['basic_info']['shape'][0]:,} rows × {summary['basic_info']['shape'][1]} columns
- **Memory Usage**: {summary['basic_info']['memory_usage_mb']:.2f} MB
- **Total Missing Values**: {summary['missing_data']['total_missing']:,} ({summary['missing_data']['total_missing'] / (df.shape[0] * df.shape[1]) * 100:.1f}% of all data)

### 🔍 **Data Types**
- **Numerical Columns** ({len(summary['data_types']['numerical'])}): {', '.join(summary['data_types']['numerical'][:5])}{'...' if len(summary['data_types']['numerical']) > 5 else ''}
- **Categorical Columns** ({len(summary['data_types']['categorical'])}): {', '.join(summary['data_types']['categorical'][:5])}{'...' if len(summary['data_types']['categorical']) > 5 else ''}
- **DateTime Columns** ({len(summary['data_types']['datetime'])}): {', '.join(summary['data_types']['datetime'][:5])}{'...' if len(summary['data_types']['datetime']) > 5 else ''}
        """
        
        # Missing data details
        if summary['missing_data']['total_missing'] > 0:
            preview_text += "\n### ⚠️ **Missing Data by Column**\n"
            missing_cols = {k: v for k, v in summary['missing_data']['missing_by_column'].items() if v > 0}
            for col, missing_count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = summary['missing_data']['missing_percentage'][col]
                preview_text += f"- **{col}**: {missing_count:,} missing ({percentage:.1f}%)\n"
        
        # Numerical statistics
        if 'numerical_stats' in summary and summary['numerical_stats']:
            preview_text += "\n### 📈 **Numerical Statistics**\n"
            stats_df = pd.DataFrame(summary['numerical_stats']).round(2)
            # Create a simple text table instead of markdown table for better compatibility
            preview_text += "\n"
            for stat in ['mean', 'std', 'min', 'max']:
                if stat in stats_df.index:
                    preview_text += f"**{stat.upper()}**:\n"
                    for col in stats_df.columns[:5]:  # Limit to first 5 columns
                        preview_text += f"- {col}: {stats_df.loc[stat, col]:.2f}\n"
                    preview_text += "\n"
        
        # Categorical summaries
        if 'categorical_stats' in summary and summary['categorical_stats']:
            preview_text += "\n### 🏷️ **Categorical Summaries**\n"
            for col, stats in list(summary['categorical_stats'].items())[:3]:
                preview_text += f"\n**{col}** ({stats['unique_count']} unique values):\n"
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
        
        # Initialize device optimizer for system info
        if not inference_only:
            try:
                self.device_optimizer = DeviceOptimizer()
                logger.info("Device optimizer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize device optimizer: {e}")
    
    def get_system_info(self) -> str:
        """Get current system information"""
        try:
            if self.device_optimizer:
                info = self.device_optimizer.get_system_info()
                return json.dumps(info, indent=2)
            return "Device optimizer not available"
        except Exception as e:
            return f"Error getting system info: {str(e)}"
    
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
**Sample Dataset Loaded: {metadata['name']}**

- **Description**: {metadata['description']}
- **Task Type**: {metadata['task_type']}
- **Target Column**: {metadata['target_column']}
- **Shape**: {df.shape[0]} rows × {df.shape[1]} columns
- **Columns**: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
- **Missing Values**: {df.isnull().sum().sum()} total
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
**Data Loaded Successfully!**

- **Shape**: {df.shape[0]} rows × {df.shape[1]} columns
- **Columns**: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
- **Missing Values**: {df.isnull().sum().sum()} total
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            """
            
            return info_text, summary, preview_text, sample_table
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}, "", ""
    
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
✅ **Model loaded successfully for inference!**

- **File**: {file.name}
- **Model Type**: {model_info.get('model_type', 'Unknown')}
- **Status**: Ready for predictions
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
**Inference Server Prediction:**

- **Input**: {input_data}
- **Prediction**: {predictions}
- **Input Shape**: {result['input_shape']}
- **Model Type**: {result.get('model_metadata', {}).get('model_type', 'Unknown')}
            """
            
            return prediction_text
            
        except Exception as e:
            error_msg = f"Error making inference prediction: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def create_training_config(self, task_type: str, optimization_strategy: str, 
                             cv_folds: int, test_size: float, random_state: int,
                             enable_feature_selection: bool, normalization: str,
                             enable_quantization: bool, optimization_mode: str) -> str:
        """Create training configuration"""
        if self.inference_only:
            return "Training is not available in inference-only mode."
        
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
            
            return f"""
**Configuration Created Successfully!**

- **Task Type**: {task_type}
- **Optimization Strategy**: {optimization_strategy}
- **CV Folds**: {cv_folds}
- **Test Size**: {test_size}
- **Feature Selection**: {'Enabled' if enable_feature_selection else 'Disabled'}
- **Normalization**: {normalization}
- **Quantization**: {'Enabled' if enable_quantization else 'Disabled'}
            """
            
        except Exception as e:
            error_msg = f"Error creating configuration: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def train_model(self, target_column: str, model_type: str, 
                   progress=gr.Progress()) -> Tuple[str, str, str]:
        """Train a model with the current configuration"""
        if self.inference_only:
            return "Training is not available in inference-only mode.", "", ""
        
        try:
            if self.current_data is None:
                return "No data loaded. Please upload a dataset first.", "", ""
            
            if self.current_config is None:
                return "No configuration created. Please configure training parameters first.", "", ""
            
            if target_column not in self.current_data.columns:
                return f"Target column '{target_column}' not found in dataset.", "", ""
            
            progress(0.1, desc="Initializing training engine...")
            
            # Initialize training engine
            self.training_engine = MLTrainingEngine(self.current_config)
            
            # Prepare data
            X = self.current_data.drop(columns=[target_column])
            y = self.current_data[target_column]
            
            progress(0.2, desc="Preprocessing data...")
            
            # Handle categorical features
            categorical_columns = X.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                for col in categorical_columns:
                    X[col] = pd.Categorical(X[col]).codes
            
            progress(0.3, desc="Starting model training...")
            
            # Train model
            start_time = time.time()
            result = self.training_engine.train_model(
                X=X.values, 
                y=y.values,
                model_type=model_type.lower().replace(' ', '_'),
                model_name=f"{model_type}_{int(time.time())}"
            )
            
            training_time = time.time() - start_time
            progress(1.0, desc="Training completed!")
            
            # Generate results summary
            metrics_text = "**Training Results:**\n\n"
            if 'metrics' in result and result['metrics']:
                for metric, value in result['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics_text += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
                    else:
                        metrics_text += f"- **{metric.replace('_', ' ').title()}**: {value}\n"
            
            metrics_text += f"\n- **Training Time**: {training_time:.2f} seconds"
            
            # Feature importance
            importance_text = ""
            if 'feature_importance' in result and result['feature_importance'] is not None:
                importance = result['feature_importance']
                feature_names = X.columns.tolist()
                
                importance_text = "**Top 10 Feature Importances:**\n\n"
                if isinstance(importance, dict):
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    for feature, score in sorted_features:
                        importance_text += f"- **{feature}**: {score:.4f}\n"
                else:
                    indices = np.argsort(importance)[::-1][:10]
                    for i, idx in enumerate(indices):
                        if idx < len(feature_names):
                            importance_text += f"- **{feature_names[idx]}**: {importance[idx]:.4f}\n"
            
            # Model summary
            summary_text = f"""
**Model Training Summary**

- **Model Type**: {model_type}
- **Dataset Shape**: {X.shape[0]} samples × {X.shape[1]} features
- **Target Column**: {target_column}
- **Task Type**: {self.current_config.task_type.value}
- **Status**: ✅ Training Completed Successfully
            """
            
            return summary_text, metrics_text, importance_text
            
        except Exception as e:
            error_msg = f"Error during training: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg, "", ""
    
    def make_prediction(self, input_data: str) -> str:
        """Make predictions using the trained model"""
        if self.inference_only:
            return "Use the Inference Server tab for predictions in inference-only mode."
        
        try:
            if self.training_engine is None:
                return "No model trained. Please train a model first."
            
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
**Prediction Result:**

- **Input**: {input_data}
- **Prediction**: {result}
- **Data Shape**: {input_array.shape}
            """
            
            return prediction_text
            
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def save_model(self, model_name: str, encryption_password: str = "") -> str:
        """Save the trained model"""
        if self.inference_only:
            return "Model saving is not available in inference-only mode."
        
        try:
            if self.training_engine is None:
                return "No model to save. Please train a model first."
            
            # Initialize model manager if not already done
            if self.model_manager is None:
                self.model_manager = SecureModelManager(
                    self.current_config,
                    logger=logger,
                    secret_key=encryption_password if encryption_password else None
                )
            
            # Get the best model
            best_model_name, best_model_info = self.training_engine.get_best_model()
            
            if best_model_info is None:
                return "No trained model available to save."
            
            # Update model manager with the model
            self.model_manager.models[model_name] = best_model_info
            self.model_manager.best_model = best_model_info
            
            # Save the model
            success = self.model_manager.save_model(
                model_name=model_name,
                access_code=encryption_password if encryption_password else None
            )
            
            if success:
                return f"✅ Model '{model_name}' saved successfully!"
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
            if self.training_engine is None:
                return "No models trained yet."
            
            comparison = self.training_engine.get_performance_comparison()
            
            if 'error' in comparison:
                return comparison['error']
            
            # Format the comparison
            result_text = "**Model Performance Comparison:**\n\n"
            
            for model in comparison['models']:
                result_text += f"### {model['name']} {'👑' if model['is_best'] else ''}\n"
                result_text += f"- **Type**: {model['type']}\n"
                result_text += f"- **Training Time**: {model['training_time']:.2f}s\n"
                
                if model['metrics']:
                    result_text += "- **Metrics**:\n"
                    for metric, value in model['metrics'].items():
                        if isinstance(value, (int, float)):
                            result_text += f"  - {metric}: {value:.4f}\n"
                
                result_text += "\n"
            
            return result_text
            
        except Exception as e:
            error_msg = f"Error getting model performance: {str(e)}"
            logger.error(error_msg)
            return error_msg

def create_ui(inference_only: bool = False):
    """Create and configure the Gradio interface"""
    
    app = MLSystemUI(inference_only=inference_only)
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    title = "🚀 ML Inference Server" if inference_only else "🚀 ML Training & Inference System"
    description = """
**A machine learning inference server for real-time predictions.**

Load your trained models and make instant predictions!
    """ if inference_only else """
**A comprehensive machine learning platform with advanced optimization and secure model management.**

Upload your data, configure training parameters, train models, and make predictions - all in one place!
    """
    
    with gr.Blocks(css=css, title=title, theme=gr.themes.Soft()) as interface:
        
        gr.Markdown(f"""
# {title}

{description}
        """)
        
        with gr.Tabs():
            
            if not inference_only:
                # Data Upload Tab
                with gr.Tab("📁 Data Upload", id="data_upload"):
                    gr.Markdown("### Upload Dataset or Load Sample Data")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Upload Your Own Dataset")
                            file_input = gr.File(
                                label="Upload Dataset",
                                file_types=[".csv", ".xlsx", ".xls", ".json"],
                                type="filepath"
                            )
                            
                            load_btn = gr.Button("Load Data", variant="primary", size="lg")
                            
                        with gr.Column(scale=1):
                            gr.Markdown("#### Or Load Sample Dataset")
                            sample_dropdown = gr.Dropdown(
                                choices=["Select a dataset..."] + app.sample_data_loader.get_available_datasets(),
                                value="Select a dataset...",
                                label="Sample Datasets"
                            )
                            
                            load_sample_btn = gr.Button("Load Sample Data", variant="secondary", size="lg")
                    
                    with gr.Row():
                        data_info = gr.Markdown("Upload a dataset or select a sample dataset to get started...")
                    
                    # Data Preview Section
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📊 Data Preview")
                            data_preview = gr.Markdown("Data preview will appear here after loading...")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📈 Sample Data (First 10 Rows)")
                            sample_data_table = gr.HTML("")
                        
                    load_btn.click(
                        fn=app.load_data,
                        inputs=[file_input],
                        outputs=[data_info, gr.State(), data_preview, sample_data_table]
                    )
                    
                    load_sample_btn.click(
                        fn=app.load_sample_data,
                        inputs=[sample_dropdown],
                        outputs=[data_info, gr.State(), data_preview, sample_data_table]
                    )
                
                # Configuration Tab
                with gr.Tab("⚙️ Configuration", id="configuration"):
                    gr.Markdown("### Training Configuration")
                    
                    with gr.Row():
                        with gr.Column():
                            task_type = gr.Dropdown(
                                choices=["CLASSIFICATION", "REGRESSION"],
                                value="CLASSIFICATION",
                                label="Task Type"
                            )
                            
                            optimization_strategy = gr.Dropdown(
                                choices=["RANDOM_SEARCH", "GRID_SEARCH", "BAYESIAN_OPTIMIZATION", "HYPERX"],
                                value="RANDOM_SEARCH",
                                label="Optimization Strategy"
                            )
                            
                            cv_folds = gr.Slider(
                                minimum=2,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Cross-Validation Folds"
                            )
                            
                        with gr.Column():
                            test_size = gr.Slider(
                                minimum=0.1,
                                maximum=0.5,
                                value=0.2,
                                step=0.05,
                                label="Test Size"
                            )
                            
                            random_state = gr.Number(
                                value=42,
                                label="Random State",
                                precision=0
                            )
                            
                            normalization = gr.Dropdown(
                                choices=["STANDARD", "MINMAX", "ROBUST", "NONE"],
                                value="STANDARD",
                                label="Normalization"
                            )
                    
                    with gr.Row():
                        enable_feature_selection = gr.Checkbox(
                            label="Enable Feature Selection",
                            value=True
                        )
                        
                        enable_quantization = gr.Checkbox(
                            label="Enable Model Quantization",
                            value=False
                        )
                        
                        optimization_mode = gr.Dropdown(
                            choices=["BALANCED", "PERFORMANCE", "MEMORY_SAVING", "CONSERVATIVE"],
                            value="BALANCED",
                            label="Optimization Mode"
                        )
                    
                    config_btn = gr.Button("Create Configuration", variant="primary", size="lg")
                    config_output = gr.Markdown("")
                    
                    config_btn.click(
                        fn=app.create_training_config,
                        inputs=[
                            task_type, optimization_strategy, cv_folds, test_size,
                            random_state, enable_feature_selection, normalization,
                            enable_quantization, optimization_mode
                        ],
                        outputs=[config_output]
                    )
                
                # Training Tab
                with gr.Tab("🎯 Model Training", id="training"):
                    gr.Markdown("### Train Machine Learning Models")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            target_column = gr.Textbox(
                                label="Target Column",
                                placeholder="Enter the name of your target column",
                                info="The column you want to predict"
                            )
                            
                            model_type = gr.Dropdown(
                                choices=[
                                    "Random Forest", "XGBoost", "LightGBM",
                                    "Logistic Regression", "SVM", "Neural Network",
                                    "Gradient Boosting", "Decision Tree"
                                ],
                                value="Random Forest",
                                label="Model Type"
                            )
                            
                            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                            
                        with gr.Column(scale=2):
                            training_output = gr.Markdown("Configure your model and click 'Start Training'...")
                    
                    with gr.Row():
                        with gr.Column():
                            metrics_output = gr.Markdown("")
                        with gr.Column():
                            importance_output = gr.Markdown("")
                    
                    train_btn.click(
                        fn=app.train_model,
                        inputs=[target_column, model_type],
                        outputs=[training_output, metrics_output, importance_output]
                    )
                
                # Prediction Tab
                with gr.Tab("🔮 Predictions", id="predictions"):
                    gr.Markdown("### Make Predictions with Trained Models")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            prediction_input = gr.Textbox(
                                label="Input Data",
                                placeholder="Enter comma-separated values or JSON array",
                                lines=3,
                                info="Example: 1.5, 2.3, 0.8, 1.1 or [1.5, 2.3, 0.8, 1.1]"
                            )
                            
                            predict_btn = gr.Button("🔮 Make Prediction", variant="primary", size="lg")
                            
                        with gr.Column(scale=2):
                            prediction_output = gr.Markdown("Enter input data and click 'Make Prediction'...")
                    
                    predict_btn.click(
                        fn=app.make_prediction,
                        inputs=[prediction_input],
                        outputs=[prediction_output]
                    )
                
                # Model Management Tab
                with gr.Tab("💾 Model Management", id="model_management"):
                    gr.Markdown("### Save and Load Models")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Save Model")
                            
                            save_model_name = gr.Textbox(
                                label="Model Name",
                                placeholder="Enter a name for your model"
                            )
                            
                            save_password = gr.Textbox(
                                label="Encryption Password (Optional)",
                                type="password",
                                placeholder="Leave empty for no encryption"
                            )
                            
                            save_btn = gr.Button("💾 Save Model", variant="primary")
                            save_output = gr.Markdown("")
                            
                        with gr.Column():
                            gr.Markdown("#### Load Model")
                            
                            load_file = gr.File(
                                label="Model File",
                                file_types=[".pkl"],
                                type="filepath"
                            )
                            
                            load_password = gr.Textbox(
                                label="Decryption Password",
                                type="password",
                                placeholder="Enter password if model is encrypted"
                            )
                            
                            load_btn = gr.Button("📂 Load Model", variant="secondary")
                            load_output = gr.Markdown("")
                    
                    save_btn.click(
                        fn=app.save_model,
                        inputs=[save_model_name, save_password],
                        outputs=[save_output]
                    )
                    
                    load_btn.click(
                        fn=app.load_model,
                        inputs=[load_file, load_password],
                        outputs=[load_output]
                    )
                
                # Performance Tab
                with gr.Tab("📊 Performance", id="performance"):
                    gr.Markdown("### Model Performance Analysis")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            performance_btn = gr.Button("📊 Get Performance Report", variant="primary", size="lg")
                            
                        with gr.Column(scale=2):
                            performance_output = gr.Markdown("Click 'Get Performance Report' to see model comparisons...")
                    
                    performance_btn.click(
                        fn=app.get_model_performance,
                        outputs=[performance_output]
                    )
            
            # Inference Server Tab (always available)
            with gr.Tab("🔧 Inference Server", id="inference_server"):
                gr.Markdown("### Dedicated Inference Server")
                gr.Markdown("Load a trained model and use it for real-time predictions.")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Load Model for Inference")
                        
                        inference_model_file = gr.File(
                            label="Model File",
                            file_types=[".pkl", ".joblib", ".h5", ".pt"],
                            type="filepath"
                        )
                        
                        inference_password = gr.Textbox(
                            label="Decryption Password (if encrypted)",
                            type="password",
                            placeholder="Leave empty if model is not encrypted"
                        )
                        
                        load_inference_btn = gr.Button("🔧 Load for Inference", variant="primary")
                        inference_load_output = gr.Markdown("")
                        
                    with gr.Column():
                        gr.Markdown("#### Make Predictions")
                        
                        inference_input = gr.Textbox(
                            label="Input Data",
                            placeholder="Enter comma-separated values or JSON array",
                            lines=3,
                            info="Example: 1.5, 2.3, 0.8, 1.1"
                        )
                        
                        inference_predict_btn = gr.Button("🎯 Predict", variant="secondary", size="lg")
                        inference_output = gr.Markdown("Load a model and enter input data to make predictions...")
                
                # Model info section
                with gr.Row():
                    model_info_btn = gr.Button("ℹ️ Get Model Info", variant="secondary")
                    model_info_output = gr.Markdown("")
                
                # Event handlers for inference server
                load_inference_btn.click(
                    fn=app.load_inference_model,
                    inputs=[inference_model_file, inference_password],
                    outputs=[inference_load_output]
                )
                
                inference_predict_btn.click(
                    fn=app.make_inference_prediction,
                    inputs=[inference_input],
                    outputs=[inference_output]
                )
                
                model_info_btn.click(
                    fn=lambda: json.dumps(app.inference_server.get_model_info(), indent=2),
                    outputs=[model_info_output]
                )
            
            if not inference_only:
                # System Info Tab
                with gr.Tab("🖥️ System Info", id="system_info"):
                    gr.Markdown("### System Information and Optimization")
                    
                    system_btn = gr.Button("🖥️ Get System Info", variant="secondary")
                    system_output = gr.JSON(label="System Information")
                    
                    system_btn.click(
                        fn=app.get_system_info,
                        outputs=[system_output]
                    )
        
        # Footer
        footer_text = """
---
**ML Inference Server** - Optimized for real-time predictions.

💡 **Tips:**
- Load your trained model in the Inference Server tab
- Make real-time predictions with minimal latency
- Supports encrypted model files for security
        """ if inference_only else """
---
**ML Training & Inference System** - Built with advanced optimization and security features.

💡 **Tips:**
- Start by uploading your dataset or loading sample data in the Data Upload tab
- Try sample datasets like Iris, Titanic, or Boston Housing for quick testing
- Configure your training parameters in the Configuration tab
- Train models and compare their performance
- Use the Inference Server for production-ready predictions
- Save models securely with encryption
        """
        
        gr.Markdown(footer_text)
    
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
    """)
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=True,
        show_error=True,
        favicon_path=None,
        ssl_verify=False,
        quiet=False
    )

if __name__ == "__main__":
    main()