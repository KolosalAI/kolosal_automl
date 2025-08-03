"""
ML Training Engine API

This FastAPI application provides a RESTful API interface for the MLTrainingEngine module.
It allows users to train, evaluate, manage, and make predictions with machine learning models.

Usage:
    uvicorn main:app --reload
"""

import os
import sys
import json
import time
import uuid
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime
from dataclasses import asdict

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, BackgroundTasks, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

# Import MLTrainingEngine components
from modules.configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig,
    PreprocessorConfig,
    InferenceEngineConfig,
    ModelSelectionCriteria,
    NormalizationType
)
from modules.engine.train_engine import MLTrainingEngine

# Configure centralized logging
try:
    from modules.logging_config import get_logger, setup_root_logging
    setup_root_logging()
    logger = get_logger(
        name="MLTrainingEngineAPI",
        level=logging.INFO,
        log_file="ml_api.log",
        enable_console=True
    )
except ImportError:
    # Fallback to basic logging if centralized logging not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("ml_api.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("MLTrainingEngineAPI")

# Create FastAPI app
app = FastAPI(
    title="ML Training Engine API",
    description="A RESTful API for training, evaluating, and deploying machine learning models",
    version="0.1.4"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory for serving files
STATIC_DIR = Path("static")
MODELS_DIR = STATIC_DIR / "models"
REPORTS_DIR = STATIC_DIR / "reports"
UPLOADS_DIR = STATIC_DIR / "uploads"
CHARTS_DIR = STATIC_DIR / "charts"

for directory in [STATIC_DIR, MODELS_DIR, REPORTS_DIR, UPLOADS_DIR, CHARTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global engine instance
ml_engine = None

# Pydantic models for API requests/responses
class TaskTypeEnum(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"

class OptimizationStrategyEnum(str, Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    OPTUNA = "optuna"
    ASHT = "asht"
    HYPERX = "hyperx"

class ModelSelectionCriteriaEnum(str, Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    ROC_AUC = "roc_auc"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    R2 = "r2"
    EXPLAINED_VARIANCE = "explained_variance"

class NormalizationTypeEnum(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"

class EngineConfigRequest(BaseModel):
    task_type: TaskTypeEnum
    model_path: str = "models"
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = -1
    verbose: int = 1
    optimization_strategy: OptimizationStrategyEnum = OptimizationStrategyEnum.RANDOM_SEARCH
    optimization_iterations: int = 20
    model_selection_criteria: ModelSelectionCriteriaEnum = None
    feature_selection: bool = False
    feature_selection_method: str = "mutual_info"
    feature_selection_k: int = None
    early_stopping: bool = False
    early_stopping_rounds: int = 10
    auto_save: bool = True
    checkpointing: bool = True
    checkpoint_path: str = "checkpoints"
    experiment_tracking: bool = True
    generate_model_summary: bool = True
    log_level: str = "INFO"
    
    class Config:
        schema_extra = {
            "example": {
                "task_type": "classification",
                "model_path": "models",
                "random_state": 42,
                "test_size": 0.2,
                "cv_folds": 5,
                "n_jobs": -1,
                "verbose": 1,
                "optimization_strategy": "random_search",
                "optimization_iterations": 20,
                "model_selection_criteria": "f1",
                "feature_selection": True,
                "feature_selection_method": "mutual_info",
                "feature_selection_k": 10,
                "early_stopping": True,
                "early_stopping_rounds": 10,
                "auto_save": True,
                "experiment_tracking": True,
                "generate_model_summary": True,
                "log_level": "INFO"
            }
        }

class PreprocessingConfigRequest(BaseModel):
    handle_nan: bool = True
    nan_strategy: str = "mean"
    detect_outliers: bool = False
    outlier_handling: str = "clip"
    categorical_encoding: str = "one_hot"
    normalization: NormalizationTypeEnum = NormalizationTypeEnum.STANDARD
    feature_interactions: bool = False
    polynomial_features: bool = False
    polynomial_degree: int = 2
    
    class Config:
        schema_extra = {
            "example": {
                "handle_nan": True,
                "nan_strategy": "mean",
                "detect_outliers": True,
                "outlier_handling": "clip",
                "categorical_encoding": "one_hot",
                "normalization": "standard",
                "feature_interactions": False,
                "polynomial_features": False,
                "polynomial_degree": 2
            }
        }

class InitializeEngineRequest(BaseModel):
    engine_config: EngineConfigRequest
    preprocessing_config: Optional[PreprocessingConfigRequest] = None
    
    class Config:
        schema_extra = {
            "example": {
                "engine_config": {
                    "task_type": "classification",
                    "model_path": "models",
                    "random_state": 42,
                    "test_size": 0.2,
                    "cv_folds": 5
                },
                "preprocessing_config": {
                    "handle_nan": True,
                    "categorical_encoding": "one_hot",
                }
            }
        }

class TrainModelRequest(BaseModel):
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    custom_model_class: Optional[str] = None
    param_grid: Optional[Dict[str, List[Any]]] = None
    train_data_file: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    test_size: Optional[float] = None
    validation_data_file: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "random_forest",
                "model_name": "rf_model_1",
                "param_grid": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10]
                },
                "train_data_file": "train_data.csv",
                "target_column": "target",
                "feature_columns": ["feature1", "feature2", "feature3"],
                "test_size": 0.2
            }
        }

class EvaluateModelRequest(BaseModel):
    model_name: Optional[str] = None
    test_data_file: Optional[str] = None
    target_column: str
    feature_columns: Optional[List[str]] = None
    detailed: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "rf_model_1",
                "test_data_file": "test_data.csv",
                "target_column": "target",
                "detailed": True
            }
        }

class PredictRequest(BaseModel):
    model_name: Optional[str] = None
    data_file: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    data: Optional[List[Dict[str, Any]]] = None
    return_probabilities: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "rf_model_1",
                "data_file": "predict_data.csv",
                "return_probabilities": True
            }
        }

class ExplainabilityRequest(BaseModel):
    model_name: Optional[str] = None
    data_file: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    method: str = "shap"
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "rf_model_1",
                "data_file": "explain_data.csv",
                "method": "shap"
            }
        }

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    feature_count: int
    metrics: Dict[str, Any]
    training_time: float
    is_best_model: bool
    top_features: Optional[Dict[str, float]] = None

class TrainingStatusResponse(BaseModel):
    status: str
    model_name: Optional[str] = None
    progress: Optional[float] = None
    eta: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
# Dictionary to track background tasks
training_tasks = {}

def get_ml_engine():
    """Get or initialize the ML engine instance."""
    global ml_engine
    if ml_engine is None:
        raise HTTPException(
            status_code=400,
            detail="ML Engine not initialized. Call /api/initialize first."
        )
    return ml_engine

def convert_to_enum(value, enum_class):
    """Convert string value to enum value."""
    if value is None:
        return None
    for enum_val in enum_class:
        if value.lower() == enum_val.value.lower():
            return enum_val
    raise ValueError(f"Invalid value: {value} for enum {enum_class.__name__}")

def map_config_to_engine_config(config: EngineConfigRequest) -> MLTrainingEngineConfig:
    """Map API config to engine config object."""
    # Map task type
    task_type = TaskType[config.task_type.upper()]
    
    # Map optimization strategy
    optimization_strategy = OptimizationStrategy[config.optimization_strategy.upper()]
    
    # Map model selection criteria if provided
    model_selection_criteria = None
    if config.model_selection_criteria:
        model_selection_criteria = ModelSelectionCriteria[config.model_selection_criteria.upper()]
    
    # Create engine config
    engine_config = MLTrainingEngineConfig(
        task_type=task_type,
        model_path=config.model_path,
        random_state=config.random_state,
        test_size=config.test_size,
        cv_folds=config.cv_folds,
        n_jobs=config.n_jobs,
        verbose=config.verbose,
        optimization_strategy=optimization_strategy,
        optimization_iterations=config.optimization_iterations,
        model_selection_criteria=model_selection_criteria,
        feature_selection=config.feature_selection,
        feature_selection_method=config.feature_selection_method,
        feature_selection_k=config.feature_selection_k,
        early_stopping=config.early_stopping,
        early_stopping_rounds=config.early_stopping_rounds,
        auto_save=config.auto_save,
        checkpointing=config.checkpointing,
        checkpoint_path=config.checkpoint_path,
        experiment_tracking=config.experiment_tracking,
        generate_model_summary=config.generate_model_summary,
        log_level=config.log_level
    )
    
    return engine_config

def map_config_to_preprocessor_config(config_request):
    """Map API config to preprocessor config."""
    # Create a sample PreprocessingConfigRequest
    preprocessor_config = PreprocessorConfig(
        # The parameter is likely named differently in the actual implementation
        # For example, it might be "handle_nan" instead of "handle_nan"
        handle_nan=config_request.handle_nan,
        nan_strategy=config_request.nan_strategy,
        detect_outliers=config_request.detect_outliers,
        outlier_handling=config_request.outlier_handling,
        categorical_encoding=config_request.categorical_encoding,
        normalization=NormalizationType(config_request.normalization)
    )
    return preprocessor_config

def load_data_from_file(file_path: str, columns: List[str] = None) -> pd.DataFrame:
    """Load data from a file."""
    path = Path(file_path)
    if not path.exists():
        path = UPLOADS_DIR / file_path
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")
    
    # Determine file type
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    elif file_path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path)
    elif file_path.lower().endswith('.parquet'):
        df = pd.read_parquet(path)
    elif file_path.lower().endswith('.json'):
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Filter columns if specified
    if columns is not None:
        # Check if columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        return df[columns]
    
    return df

async def save_uploaded_file(file: UploadFile, upload_dir: Path = None) -> str:
    """Save an uploaded file and return the file path."""
    if upload_dir is None:
        upload_dir = UPLOADS_DIR
    
    # Ensure upload directory exists
    upload_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate unique filename to avoid conflicts
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    file_extension = Path(file.filename).suffix if file.filename else ""
    unique_filename = f"{timestamp}_{unique_id}{file_extension}"
    
    file_path = upload_dir / unique_filename
    
    try:
        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return str(file_path)
        
    except Exception as e:
        # Clean up if there was an error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

def train_model_task(request: TrainModelRequest, task_id: str):
    """Background task for model training."""
    try:
        # Update task status
        training_tasks[task_id] = {
            "status": "running",
            "progress": 0.0,
            "started_at": datetime.now().isoformat(),
            "model_name": request.model_name,
            "error": None
        }
        
        # Get engine instance
        engine = get_ml_engine()
        
        # Load training data
        if request.train_data_file:
            df = load_data_from_file(request.train_data_file)
            
            # Split features and target
            if request.target_column not in df.columns:
                raise ValueError(f"Target column '{request.target_column}' not found in data")
            
            # Update progress
            training_tasks[task_id]["progress"] = 0.1
            
            # Extract features and target
            y = df[request.target_column]
            
            # Select feature columns if specified, otherwise use all except target
            if request.feature_columns:
                X = df[request.feature_columns]
            else:
                X = df.drop(request.target_column, axis=1)
                
            # Update progress
            training_tasks[task_id]["progress"] = 0.2
            
            # Load validation data if provided
            X_val, y_val = None, None
            if request.validation_data_file:
                val_df = load_data_from_file(request.validation_data_file)
                
                # Check for target column
                if request.target_column not in val_df.columns:
                    raise ValueError(f"Target column '{request.target_column}' not found in validation data")
                
                # Extract features and target
                y_val = val_df[request.target_column]
                
                # Select features
                if request.feature_columns:
                    X_val = val_df[request.feature_columns]
                else:
                    X_val = val_df.drop(request.target_column, axis=1)
        else:
            raise ValueError("Training data file must be provided")
            
        # Initialize custom model if specified
        custom_model = None
        if request.custom_model_class:
            try:
                # Import the custom model class
                module_parts = request.custom_model_class.split('.')
                class_name = module_parts.pop()
                module_path = '.'.join(module_parts)
                
                # Import the module
                module = __import__(module_path, fromlist=[class_name])
                model_class = getattr(module, class_name)
                
                # Initialize model
                custom_model = model_class()
            except Exception as e:
                raise ValueError(f"Failed to initialize custom model: {str(e)}")
        
        # Apply test size override if provided
        test_size_override = None
        if request.test_size is not None:
            test_size_override = engine.config.test_size
            engine.config.test_size = request.test_size
        
        # Update progress
        training_tasks[task_id]["progress"] = 0.3
        
        # Train the model
        try:
            # Set model name if not specified
            model_name = request.model_name
            if not model_name:
                model_name = f"{request.model_type or 'custom'}_{int(time.time())}"
                
            # Update task with model name
            training_tasks[task_id]["model_name"] = model_name
                
            # Train model
            result = engine.train_model(
                X=X, 
                y=y, 
                model_type=request.model_type,
                custom_model=custom_model,
                param_grid=request.param_grid,
                model_name=model_name,
                X_val=X_val,
                y_val=y_val
            )
            
            # Update progress
            training_tasks[task_id]["progress"] = 1.0
            training_tasks[task_id]["status"] = "completed"
            training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
            
            # Store the result
            training_tasks[task_id]["result"] = {
                "model_name": result["model_name"],
                "metrics": result["metrics"],
                "training_time": result["training_time"]
            }
            
            # Restore original test size if overridden
            if test_size_override is not None:
                engine.config.test_size = test_size_override
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update task with error
            training_tasks[task_id]["status"] = "failed"
            training_tasks[task_id]["error"] = str(e)
            training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
            
            # Restore original test size if overridden
            if test_size_override is not None:
                engine.config.test_size = test_size_override
                
            raise e
            
    except Exception as e:
        logger.error(f"Error in training task: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update task with error
        if task_id in training_tasks:
            training_tasks[task_id]["status"] = "failed"
            training_tasks[task_id]["error"] = str(e)
            training_tasks[task_id]["completed_at"] = datetime.now().isoformat()

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ML Training Engine API",
        "version": "0.1.4",
        "documentation": "/docs",
        "status": "engine_initialized" if ml_engine else "engine_not_initialized"
    }

@app.post("/api/initialize", status_code=200)
async def initialize_engine(request: InitializeEngineRequest):
    """Initialize the ML Training Engine."""
    global ml_engine
    
    try:
        logger.info("Initializing ML Engine")
        
        # Map API config to engine config
        engine_config = map_config_to_engine_config(request.engine_config)
        
        # Map preprocessing config if provided
        preprocessor_config = None
        if request.preprocessing_config:
            preprocessor_config = map_config_to_preprocessor_config(request.preprocessing_config)
            engine_config.preprocessing_config = preprocessor_config
        
        # Create output directories
        os.makedirs(engine_config.model_path, exist_ok=True)
        if engine_config.checkpointing:
            os.makedirs(engine_config.checkpoint_path, exist_ok=True)
        
        # Initialize the engine
        ml_engine = MLTrainingEngine(engine_config)
        
        return {
            "status": "success",
            "message": "ML Engine initialized successfully",
            "config": {
                "task_type": engine_config.task_type.value,
                "model_path": engine_config.model_path,
                "experiment_tracking": engine_config.experiment_tracking
            }
        }
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/api/upload", status_code=200)
async def upload_file(file: UploadFile = File(...)):
    """Upload a data file."""
    try:
        # Create upload directory if not exists
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        # Generate file path
        file_extension = file.filename.split('.')[-1]
        timestamp = int(time.time())
        filename = f"{file.filename.rsplit('.', 1)[0]}_{timestamp}.{file_extension}"
        file_path = UPLOADS_DIR / filename
        
        # Save the file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Get file info
        file_size = os.path.getsize(file_path)
        
        # Try to read file to get row and column count
        try:
            if file_extension.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_extension.lower() in ['xls', 'xlsx']:
                df = pd.read_excel(file_path)
            elif file_extension.lower() == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_extension.lower() == 'json':
                df = pd.read_json(file_path)
            else:
                df = None
                
            if df is not None:
                row_count = len(df)
                column_count = len(df.columns)
                columns = df.columns.tolist()
                
                # Get column types
                column_types = {col: str(df[col].dtype) for col in df.columns}
                
                # Get sample data (first 5 rows)
                sample_data = df.head(5).to_dict(orient='records')
            else:
                row_count = None
                column_count = None
                columns = []
                column_types = {}
                sample_data = []
                
        except Exception as e:
            logger.warning(f"Error reading file: {str(e)}")
            row_count = None
            column_count = None
            columns = []
            column_types = {}
            sample_data = []
        
        return {
            "status": "success",
            "filename": filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "row_count": row_count,
            "column_count": column_count,
            "columns": columns,
            "column_types": column_types,
            "sample_data": sample_data
        }
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/api/train", status_code=202)
async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """Train a model with the given parameters."""
    # Get engine
    engine = get_ml_engine()
    
    # Generate task ID
    task_id = f"train_{int(time.time())}"
    
    # Start training in background
    background_tasks.add_task(train_model_task, request, task_id)
    
    return {
        "status": "training_started",
        "task_id": task_id,
        "message": "Model training started in the background"
    }

@app.get("/api/train/status/{task_id}", response_model=TrainingStatusResponse)
async def get_training_status(task_id: str):
    """Get status of a training task."""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"Training task {task_id} not found")
        
    task = training_tasks[task_id]
    
    # Calculate ETA if running
    eta = None
    if task["status"] == "running" and task["progress"] > 0:
        started = datetime.fromisoformat(task["started_at"])
        elapsed = (datetime.now() - started).total_seconds()
        if task["progress"] < 1.0:
            eta = elapsed * (1.0 - task["progress"]) / task["progress"]
    
    return {
        "status": task["status"],
        "model_name": task.get("model_name"),
        "progress": task.get("progress"),
        "eta": eta,
        "started_at": task.get("started_at"),
        "completed_at": task.get("completed_at"),
        "error": task.get("error")
    }

@app.get("/api/models", status_code=200)
async def list_models():
    """List all trained models."""
    engine = get_ml_engine()
    
    models = []
    for model_name, model_info in engine.models.items():
        models.append({
            "name": model_name,
            "type": type(model_info["model"]).__name__,
            "metrics": model_info.get("metrics", {}),
            "training_time": model_info.get("training_time", 0),
            "is_best": (model_name == engine.best_model_name),
            "feature_count": len(model_info.get("feature_names", [])),
            "loaded_from": model_info.get("loaded_from")
        })
    
    return {
        "models": models,
        "best_model": engine.best_model_name,
        "count": len(models)
    }

@app.post("/api/models/{model_name}/predict", status_code=200)
async def predict(model_name: str, request: PredictRequest):
    """Make predictions using a trained model."""
    engine = get_ml_engine()
    
    # Check if model exists
    if model_name not in engine.models and model_name != "best":
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        # Load prediction data
        if request.data_file:
            df = load_data_from_file(request.data_file)
            
            # Select feature columns if specified
            if request.feature_columns:
                # Check if columns exist
                missing_cols = [col for col in request.feature_columns if col not in df.columns]
                if missing_cols:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Columns not found in data: {missing_cols}"
                    )
                X = df[request.feature_columns]
            else:
                X = df
        elif request.data:
            # Convert list of dicts to DataFrame
            X = pd.DataFrame(request.data)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either data_file or data must be provided"
            )
        
        # Make predictions
        if model_name == "best":
            actual_model_name = engine.best_model_name
        else:
            actual_model_name = model_name
            
        success, predictions = engine.predict(
            X=X,
            model_name=actual_model_name,
            return_proba=request.return_probabilities
        )
        
        # Check for error
        if not success:
            raise HTTPException(status_code=500, detail=str(predictions))
        
        # Convert predictions to list
        if isinstance(predictions, np.ndarray):
            if predictions.ndim == 1:
                # For simple predictions
                predictions_list = predictions.tolist()
            else:
                # For probability predictions
                predictions_list = predictions.tolist()
        else:
            predictions_list = predictions
        
        return {
            "model_name": actual_model_name,
            "predictions": predictions_list,
            "prediction_count": len(predictions_list),
            "probabilities": request.return_probabilities,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/models/{model_name}/explain", status_code=200)
async def explain_model(model_name: str, request: ExplainabilityRequest):
    """Generate model explainability visualizations."""
    engine = get_ml_engine()
    
    # Check if model exists
    if model_name not in engine.models and model_name != "best":
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        # Load data for explanation
        if request.data_file:
            df = load_data_from_file(request.data_file)
            
            # Select feature columns if specified
            if request.feature_columns:
                # Check if columns exist
                missing_cols = [col for col in request.feature_columns if col not in df.columns]
                if missing_cols:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Columns not found in data: {missing_cols}"
                    )
                X = df[request.feature_columns]
            else:
                X = df
        else:
            # Use cached data
            X = None
        
        # Use best model if specified
        if model_name == "best":
            actual_model_name = engine.best_model_name
        else:
            actual_model_name = model_name
            
        # Generate explainability
        explanation = engine.generate_explainability(
            model_name=actual_model_name,
            X=X,
            method=request.method
        )
        
        # Check for error
        if "error" in explanation:
            raise HTTPException(status_code=500, detail=explanation["error"])
        
        # If a plot was generated, copy it to the static directory
        plot_url = None
        if "plot_path" in explanation and explanation["plot_path"]:
            # Create a plot filename
            method = explanation["method"]
            timestamp = int(time.time())
            plot_filename = f"{method}_{actual_model_name}_{timestamp}.png"
            
            # Create destination path
            dest_path = CHARTS_DIR / plot_filename
            
            # Copy file
            try:
                import shutil
                shutil.copy2(explanation["plot_path"], dest_path)
                
                # Create URL
                plot_url = f"/static/charts/{plot_filename}"
                
                # Update explanation
                explanation["plot_url"] = plot_url
            except Exception as e:
                logger.warning(f"Failed to copy plot file: {str(e)}")
        
        return {
            "model_name": actual_model_name,
            "method": explanation["method"],
            "importance": explanation.get("importance", {}),
            "plot_url": plot_url,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explainability error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Explainability generation failed: {str(e)}")

@app.post("/api/models/save/{model_name}", status_code=200)
async def save_model(model_name: str, include_preprocessor: bool = Query(True)):
    """Save a model to disk."""
    engine = get_ml_engine()
    
    # Check if model exists
    if model_name not in engine.models and model_name != "best":
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        # Determine actual model name if "best"
        if model_name == "best":
            actual_model_name = engine.best_model_name
        else:
            actual_model_name = model_name
            
        # Save model
        save_path = engine.save_model(
            model_name=actual_model_name,
            include_preprocessor=include_preprocessor
        )
        
        if not save_path:
            raise HTTPException(status_code=500, detail="Failed to save model")
        
        return {
            "model_name": actual_model_name,
            "save_path": str(save_path),
            "include_preprocessor": include_preprocessor,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model save error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model save failed: {str(e)}")

@app.post("/api/models/load", status_code=200)
async def load_model(
    file: UploadFile = File(None), 
    path: str = Form(None),
    model_name: str = Form(None)
):
    """Load a model from disk or uploaded file."""
    engine = get_ml_engine()
    
    try:
        # Check that either path or file is provided
        if not file and not path:
            raise HTTPException(
                status_code=400,
                detail="Either file or path must be provided"
            )
            
        # If file is provided, save it temporarily
        temp_path = None
        if file:
            # Save uploaded file
            temp_path = UPLOADS_DIR / f"model_{int(time.time())}_{file.filename}"
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # Use this path for loading
            load_path = temp_path
        else:
            # Use provided path
            load_path = path
        
        # Load the model
        success, model = engine.load_model(
            path=load_path,
            model_name=model_name
        )
        
        # Check for error
        if not success:
            raise HTTPException(status_code=500, detail=str(model))
        
        # Get actual model name (in case it was auto-generated)
        for name, info in engine.models.items():
            if info["model"] is model:
                loaded_model_name = name
                break
        else:
            loaded_model_name = model_name
        
        return {
            "status": "success",
            "model_name": loaded_model_name,
            "model_type": type(model).__name__,
            "is_best": (loaded_model_name == engine.best_model_name),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model load error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")
    finally:
        # Clean up temporary file if created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.get("/api/models/compare", status_code=200)
async def compare_models():
    """Compare performance across all trained models."""
    engine = get_ml_engine()
    
    try:
        # Get model comparison
        comparison = engine.get_performance_comparison()
        
        # Check for error
        if "error" in comparison:
            raise HTTPException(status_code=500, detail=comparison["error"])
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model comparison error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

@app.post("/api/reports/generate", status_code=200)
async def generate_report():
    """Generate a comprehensive report of all models."""
    engine = get_ml_engine()
    
    try:
        # Create reports directory
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = int(time.time())
        output_file = REPORTS_DIR / f"model_report_{timestamp}.md"
        
        # Generate report
        report_path = engine.generate_report(output_file=output_file)
        
        if not report_path:
            raise HTTPException(status_code=500, detail="Failed to generate report")
        
        return {
            "report_path": str(report_path),
            "download_url": f"/api/reports/download/{output_file.name}",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/api/reports/download/{filename}", status_code=200)
async def download_report(filename: str):
    """Download a generated report."""
    report_path = REPORTS_DIR / filename
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report {filename} not found")
    
    return FileResponse(
        path=report_path, 
        filename=filename,
        media_type="text/markdown"
    )

@app.post("/api/engine/status", status_code=200)
async def engine_status():
    """Get the status of the ML Engine."""
    try:
        engine = get_ml_engine()
        
        # Get basic status info
        status = {
            "initialized": True,
            "task_type": engine.config.task_type.value,
            "model_count": len(engine.models),
            "best_model": engine.best_model_name,
            "training_complete": engine.training_complete
        }
        
        # Get configuration
        config = {}
        for key, value in vars(engine.config).items():
            if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                config[key] = value
            elif hasattr(value, "value"):  # Handle enums
                config[key] = value.value
            else:
                config[key] = str(value)
                
        status["config"] = config
        
        # If preprocessor is available, add info
        if engine.preprocessor:
            status["preprocessor"] = {
                "type": type(engine.preprocessor).__name__,
                "configured": True
            }
        else:
            status["preprocessor"] = {
                "configured": False
            }
            
        return status
        
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "initialized": False,
            "error": str(e)
        }

@app.post("/api/engine/shutdown", status_code=200)
async def shutdown_engine():
    """Shut down the ML Engine and release resources."""
    global ml_engine
    
    if ml_engine is None:
        return {
            "status": "success",
            "message": "Engine already shut down"
        }
    
    try:
        # Shut down the engine
        ml_engine.shutdown()
        ml_engine = None
        
        return {
            "status": "success",
            "message": "Engine shut down successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Shutdown failed: {str(e)}")
async def get_model_info(model_name: str):
    """Get detailed information about a model."""
    engine = get_ml_engine()
    
    # Check if model exists
    if model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Get model summary
    model_summary = engine.get_model_summary(model_name)
    
    # Check for error
    if "error" in model_summary:
        raise HTTPException(status_code=500, detail=model_summary["error"])
    
    return model_summary

@app.post("/api/models/{model_name}/evaluate", status_code=200)
async def evaluate_model(model_name: str, request: EvaluateModelRequest):
    """Evaluate a model on test data."""
    engine = get_ml_engine()
    
    # Check if model exists
    if model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        # Load test data
        if request.test_data_file:
            df = load_data_from_file(request.test_data_file)
            
            # Split features and target
            if request.target_column not in df.columns:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Target column '{request.target_column}' not found in data"
                )
            
            # Extract features and target
            y_test = df[request.target_column]
            
            # Select feature columns if specified, otherwise use all except target
            if request.feature_columns:
                X_test = df[request.feature_columns]
            else:
                X_test = df.drop(request.target_column, axis=1)
        else:
            raise HTTPException(
                status_code=400,
                detail="Test data file must be provided"
            )
        
        # Evaluate model
        metrics = engine.evaluate_model(
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            detailed=request.detailed
        )
        
        # Check for error
        if "error" in metrics:
            raise HTTPException(status_code=500, detail=metrics["error"])
        
        return {
            "model_name": model_name,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "detailed": request.detailed
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# --- Advanced Training Features ---

@app.post("/api/hyperparameter-optimization")
async def optimize_hyperparameters(
    model_type: str = Body(...),
    data_file: UploadFile = File(...),
    target_column: str = Body(...),
    feature_columns: Optional[List[str]] = Body(None),
    optimization_strategy: OptimizationStrategyEnum = Body(OptimizationStrategyEnum.RANDOM_SEARCH),
    optimization_iterations: int = Body(50),
    cv_folds: int = Body(5),
    param_grid: Optional[Dict[str, Any]] = Body(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Run hyperparameter optimization for a specific model type."""
    if not ml_engine:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    try:
        # Save uploaded file
        file_path = save_uploaded_file(data_file)
        df = load_data_from_file(file_path)
        
        # Prepare data
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        y = df[target_column]
        X = df[feature_columns] if feature_columns else df.drop(target_column, axis=1)
        
        # Create task ID for tracking
        task_id = str(uuid.uuid4())
        training_tasks[task_id] = {
            "task_id": task_id,
            "task_type": "hyperparameter_optimization",
            "status": "running",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "model_type": model_type,
            "optimization_strategy": optimization_strategy
        }
        
        # Run optimization in background
        def run_optimization():
            try:
                # Perform hyperparameter optimization
                result = ml_engine.optimize_hyperparameters(
                    X=X, y=y,
                    model_type=model_type,
                    optimization_strategy=optimization_strategy,
                    optimization_iterations=optimization_iterations,
                    cv_folds=cv_folds,
                    param_grid=param_grid
                )
                
                training_tasks[task_id]["status"] = "completed"
                training_tasks[task_id]["progress"] = 1.0
                training_tasks[task_id]["result"] = result
                training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"Optimization error: {str(e)}")
                training_tasks[task_id]["status"] = "failed"
                training_tasks[task_id]["error"] = str(e)
                training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
        background_tasks.add_task(run_optimization)
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "Hyperparameter optimization started",
            "check_status_url": f"/api/task-status/{task_id}"
        }
        
    except Exception as e:
        logger.error(f"Error starting optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feature-selection")
async def run_feature_selection(
    data_file: UploadFile = File(...),
    target_column: str = Body(...),
    feature_selection_method: str = Body("mutual_info"),
    feature_selection_k: Optional[int] = Body(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Run feature selection on the provided dataset."""
    if not ml_engine:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    try:
        # Save and load data
        file_path = save_uploaded_file(data_file)
        df = load_data_from_file(file_path)
        
        # Prepare data
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        y = df[target_column]
        X = df.drop(target_column, axis=1)
        
        # Create task ID
        task_id = str(uuid.uuid4())
        training_tasks[task_id] = {
            "task_id": task_id,
            "task_type": "feature_selection",
            "status": "running",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "method": feature_selection_method
        }
        
        def run_feature_selection():
            try:
                # Perform feature selection
                result = ml_engine.select_features(
                    X=X, y=y,
                    method=feature_selection_method,
                    k=feature_selection_k
                )
                
                training_tasks[task_id]["status"] = "completed"
                training_tasks[task_id]["progress"] = 1.0
                training_tasks[task_id]["result"] = result
                training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"Feature selection error: {str(e)}")
                training_tasks[task_id]["status"] = "failed"
                training_tasks[task_id]["error"] = str(e)
                training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
        background_tasks.add_task(run_feature_selection)
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "Feature selection started",
            "check_status_url": f"/api/task-status/{task_id}"
        }
        
    except Exception as e:
        logger.error(f"Error starting feature selection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ensemble-training")
async def train_ensemble(
    data_file: UploadFile = File(...),
    target_column: str = Body(...),
    feature_columns: Optional[List[str]] = Body(None),
    ensemble_methods: List[str] = Body(["random_forest", "gradient_boosting", "svm"]),
    ensemble_strategy: str = Body("voting"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Train an ensemble of models."""
    if not ml_engine:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    try:
        # Save and load data
        file_path = save_uploaded_file(data_file)
        df = load_data_from_file(file_path)
        
        # Prepare data
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        y = df[target_column]
        X = df[feature_columns] if feature_columns else df.drop(target_column, axis=1)
        
        # Create task ID
        task_id = str(uuid.uuid4())
        training_tasks[task_id] = {
            "task_id": task_id,
            "task_type": "ensemble_training",
            "status": "running",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "ensemble_methods": ensemble_methods,
            "ensemble_strategy": ensemble_strategy
        }
        
        def run_ensemble_training():
            try:
                # Train ensemble
                result = ml_engine.train_ensemble(
                    X=X, y=y,
                    methods=ensemble_methods,
                    strategy=ensemble_strategy
                )
                
                training_tasks[task_id]["status"] = "completed"
                training_tasks[task_id]["progress"] = 1.0
                training_tasks[task_id]["result"] = result
                training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"Ensemble training error: {str(e)}")
                training_tasks[task_id]["status"] = "failed"
                training_tasks[task_id]["error"] = str(e)
                training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
        background_tasks.add_task(run_ensemble_training)
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "Ensemble training started",
            "check_status_url": f"/api/task-status/{task_id}"
        }
        
    except Exception as e:
        logger.error(f"Error starting ensemble training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cross-validation")
async def perform_cross_validation(
    model_name: str = Body(...),
    data_file: UploadFile = File(...),
    target_column: str = Body(...),
    feature_columns: Optional[List[str]] = Body(None),
    cv_folds: int = Body(5),
    scoring: Optional[str] = Body(None)
):
    """Perform cross-validation on a trained model."""
    if not ml_engine:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    try:
        # Save and load data
        file_path = save_uploaded_file(data_file)
        df = load_data_from_file(file_path)
        
        # Prepare data
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        y = df[target_column]
        X = df[feature_columns] if feature_columns else df.drop(target_column, axis=1)
        
        # Perform cross-validation
        result = ml_engine.cross_validate_model(
            model_name=model_name,
            X=X, y=y,
            cv_folds=cv_folds,
            scoring=scoring
        )
        
        return {
            "model_name": model_name,
            "cv_results": result,
            "cv_folds": cv_folds,
            "scoring": scoring,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cross-validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-explainability/{model_name}")
async def get_model_explainability(
    model_name: str,
    data_file: UploadFile = File(...),
    target_column: str = Body(...),
    feature_columns: Optional[List[str]] = Body(None),
    explainer_type: str = Body("shap"),
    num_samples: int = Body(100)
):
    """Generate model explainability results."""
    if not ml_engine:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    try:
        # Save and load data
        file_path = save_uploaded_file(data_file)
        df = load_data_from_file(file_path)
        
        # Prepare data
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        y = df[target_column]
        X = df[feature_columns] if feature_columns else df.drop(target_column, axis=1)
        
        # Sample data if requested
        if num_samples < len(X):
            sample_indices = np.random.choice(len(X), num_samples, replace=False)
            X = X.iloc[sample_indices]
            y = y.iloc[sample_indices]
        
        # Generate explainability
        result = ml_engine.explain_model(
            model_name=model_name,
            X=X, y=y,
            explainer_type=explainer_type
        )
        
        return {
            "model_name": model_name,
            "explainability": result,
            "explainer_type": explainer_type,
            "num_samples": len(X),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Explainability error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/engine-stats")
async def get_engine_statistics():
    """Get comprehensive engine statistics."""
    if not ml_engine:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    try:
        stats = {
            "engine_version": ml_engine.VERSION,
            "config": asdict(ml_engine.config) if hasattr(ml_engine.config, '__dict__') else str(ml_engine.config),
            "trained_models": list(ml_engine.models.keys()) if hasattr(ml_engine, 'models') else [],
            "best_model": ml_engine.best_model_name if hasattr(ml_engine, 'best_model_name') else None,
            "best_score": ml_engine.best_score if hasattr(ml_engine, 'best_score') else None,
            "training_complete": getattr(ml_engine, 'training_complete', False),
            "active_tasks": len([t for t in training_tasks.values() if t["status"] == "running"]),
            "completed_tasks": len([t for t in training_tasks.values() if t["status"] == "completed"]),
            "failed_tasks": len([t for t in training_tasks.values() if t["status"] == "failed"]),
            "total_tasks": len(training_tasks),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add preprocessing stats if available
        if hasattr(ml_engine, 'preprocessor') and ml_engine.preprocessor:
            if hasattr(ml_engine.preprocessor, '_metrics'):
                stats["preprocessing_metrics"] = ml_engine.preprocessor._metrics
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting engine stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-experiment")
async def save_experiment(
    experiment_name: str = Body(...),
    include_models: bool = Body(True),
    include_data: bool = Body(False)
):
    """Save the current experiment state."""
    if not ml_engine:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    try:
        # Generate experiment package
        experiment_path = ml_engine.save_experiment(
            experiment_name=experiment_name,
            include_models=include_models,
            include_data=include_data
        )
        
        return {
            "experiment_name": experiment_name,
            "experiment_path": experiment_path,
            "include_models": include_models,
            "include_data": include_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error saving experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load-experiment")
async def load_experiment(experiment_file: UploadFile = File(...)):
    """Load a previously saved experiment."""
    global ml_engine
    
    try:
        # Save uploaded experiment file
        experiment_path = save_uploaded_file(experiment_file)
        
        # Load experiment
        ml_engine = MLTrainingEngine.load_experiment(experiment_path)
        
        return {
            "message": "Experiment loaded successfully",
            "experiment_file": experiment_file.filename,
            "models_loaded": list(ml_engine.models.keys()) if hasattr(ml_engine, 'models') else [],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error loading experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for the training engine."""
    return {
        "status": "healthy",
        "service": "ML Training Engine API",
        "version": "0.1.4",
        "engine_initialized": ml_engine is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    for directory in [STATIC_DIR, MODELS_DIR, REPORTS_DIR, UPLOADS_DIR, CHARTS_DIR]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8000)
