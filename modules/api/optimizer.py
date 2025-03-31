from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tempfile
import os
import json
import time
import logging
from enum import Enum
from datetime import datetime

# Import engine and optimization modules
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, OptimizationStrategy, TaskType

router = APIRouter(prefix="/optimizer", tags=["Hyperparameter Optimization"])

# Data models
class OptimizationStrategy(str, Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"
    ASHT = "asht"
    HYPERX = "hyperx"
    OPTUNA = "optuna"

class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"
    KNN = "knn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    
    # Regressor variants
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    SVR = "svr"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    KNN_REGRESSOR = "knn_regressor"
    XGBOOST_REGRESSOR = "xgboost_regressor"
    LIGHTGBM_REGRESSOR = "lightgbm_regressor"

class OptimizationParams(BaseModel):
    model_type: ModelType
    model_name: str
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.HYPERX
    optimization_iterations: int = 50
    optimization_timeout_seconds: Optional[int] = None
    cv_folds: int = 5
    test_size: float = 0.2
    stratify: bool = True
    random_state: Optional[int] = 42
    early_stopping: bool = True
    early_stopping_rounds: int = 5
    params_to_optimize: Optional[Dict[str, Any]] = None

class OptimizationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class OptimizationJobStatus(BaseModel):
    job_id: str
    model_type: ModelType
    model_name: str
    status: OptimizationStatus
    start_time: float
    completion_time: Optional[float] = None
    elapsed_time: Optional[float] = None
    current_iteration: int = 0
    total_iterations: int = 0
    best_score: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Dependency to get ML training engine instance
def get_ml_engine():
    config = MLTrainingEngineConfig()
    return MLTrainingEngine(config)

# Keep track of optimization jobs
optimization_jobs = {}

@router.post("/optimize")
async def start_optimization(
    background_tasks: BackgroundTasks,
    params: OptimizationParams,
    data_file: UploadFile = File(...),
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Start a hyperparameter optimization job"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await data_file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Extract X and y from dataframe (assuming last column is target)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        
        # Generate job ID
        job_id = f"opt_{params.model_name}_{int(time.time())}"
        
        # Create parameter grid based on selected model type
        param_grid = params.params_to_optimize or _get_default_param_grid(params.model_type)
        
        # Set up optimization configuration
        try:
            # Map string enum to actual enum value
            optimization_strategy = getattr(OptimizationStrategy, params.optimization_strategy.upper())
        except (AttributeError, KeyError):
            optimization_strategy = OptimizationStrategy.HYPERX
        
        # Create a temporary directory for optimization artifacts
        opt_dir = os.path.join(os.getcwd(), "optimization_jobs")
        os.makedirs(opt_dir, exist_ok=True)
        job_dir = os.path.join(opt_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Save data copy for the job
        data_copy_path = os.path.join(job_dir, "data.csv")
        df.to_csv(data_copy_path, index=False)
        
        # Create status file
        status = OptimizationJobStatus(
            job_id=job_id,
            model_type=params.model_type,
            model_name=params.model_name,
            status=OptimizationStatus.PENDING,
            start_time=time.time(),
            total_iterations=params.optimization_iterations
        )
        optimization_jobs[job_id] = status
        
        # Save status to file
        with open(os.path.join(job_dir, "status.json"), "w") as f:
            json.dump(status.dict(), f)
        
        # Start optimization in background
        background_tasks.add_task(
            _run_optimization,
            job_id=job_id,
            engine=engine,
            model_type=params.model_type,
            model_name=params.model_name,
            param_grid=param_grid,
            X=X,
            y=y,
            optimization_strategy=optimization_strategy,
            optimization_iterations=params.optimization_iterations,
            optimization_timeout=params.optimization_timeout_seconds,
            cv_folds=params.cv_folds,
            test_size=params.test_size,
            stratify=params.stratify,
            random_state=params.random_state,
            early_stopping=params.early_stopping,
            early_stopping_rounds=params.early_stopping_rounds,
            job_dir=job_dir
        )
        
        return {
            "job_id": job_id,
            "message": f"Optimization job started for model '{params.model_name}'",
            "status_endpoint": f"/optimizer/status/{job_id}",
            "model_type": params.model_type,
            "optimization_strategy": params.optimization_strategy
        }
    
    finally:
        if os.path.exists(job_dir):
            import shutil
            try:
                shutil.rmtree(job_dir)
                return {"message": f"Optimization job {job_id} deleted successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error deleting job directory: {str(e)}")
    
    raise HTTPException(status_code=404, detail=f"Optimization job {job_id} not found")


@router.get("/available-model-types")
async def get_available_model_types():
    """Get list of available model types for optimization"""
    return {
        "classification_models": [
            {"value": "random_forest", "name": "Random Forest"},
            {"value": "gradient_boosting", "name": "Gradient Boosting"},
            {"value": "svm", "name": "Support Vector Machine"},
            {"value": "logistic_regression", "name": "Logistic Regression"},
            {"value": "decision_tree", "name": "Decision Tree"},
            {"value": "knn", "name": "K-Nearest Neighbors"},
            {"value": "xgboost", "name": "XGBoost"},
            {"value": "lightgbm", "name": "LightGBM"}
        ],
        "regression_models": [
            {"value": "random_forest_regressor", "name": "Random Forest Regressor"},
            {"value": "gradient_boosting_regressor", "name": "Gradient Boosting Regressor"},
            {"value": "svr", "name": "Support Vector Regressor"},
            {"value": "linear_regression", "name": "Linear Regression"},
            {"value": "decision_tree_regressor", "name": "Decision Tree Regressor"},
            {"value": "knn_regressor", "name": "K-Nearest Neighbors Regressor"},
            {"value": "xgboost_regressor", "name": "XGBoost Regressor"},
            {"value": "lightgbm_regressor", "name": "LightGBM Regressor"}
        ],
        "optimization_strategies": [
            {"value": "grid_search", "name": "Grid Search"},
            {"value": "random_search", "name": "Random Search"},
            {"value": "bayesian_optimization", "name": "Bayesian Optimization"},
            {"value": "evolutionary", "name": "Evolutionary Algorithm"},
            {"value": "hyperband", "name": "Hyperband"},
            {"value": "asht", "name": "Adaptive Surrogate-Assisted Hyperparameter Tuning"},
            {"value": "hyperx", "name": "HyperX (Advanced)"},
            {"value": "optuna", "name": "Optuna"}
        ]
    }

@router.get("/default-params/{model_type}")
async def get_default_params(model_type: ModelType):
    """Get default hyperparameter grid for a model type"""
    param_grid = _get_default_param_grid(model_type)
    return {
        "model_type": model_type,
        "parameter_grid": param_grid
    }

def _get_default_param_grid(model_type: ModelType) -> Dict[str, Any]:
    """Get default hyperparameter grid based on model type"""
    if model_type == ModelType.RANDOM_FOREST or model_type == ModelType.RANDOM_FOREST_REGRESSOR:
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }
    elif model_type == ModelType.GRADIENT_BOOSTING or model_type == ModelType.GRADIENT_BOOSTING_REGRESSOR:
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0]
        }
    elif model_type == ModelType.SVM or model_type == ModelType.SVR:
        return {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto", 0.1, 0.01]
        }
    elif model_type == ModelType.LOGISTIC_REGRESSION:
        return {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "max_iter": [100, 200, 500]
        }
    elif model_type == ModelType.LINEAR_REGRESSION:
        return {
            "fit_intercept": [True, False],
            "normalize": [True, False]
        }
    elif model_type == ModelType.DECISION_TREE or model_type == ModelType.DECISION_TREE_REGRESSOR:
        return {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"] if model_type == ModelType.DECISION_TREE else ["mse", "friedman_mse", "mae"]
        }
    elif model_type == ModelType.KNN or model_type == ModelType.KNN_REGRESSOR:
        return {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p": [1, 2]
        }
    elif model_type == ModelType.XGBOOST or model_type == ModelType.XGBOOST_REGRESSOR:
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "gamma": [0, 0.1, 0.2]
        }
    elif model_type == ModelType.LIGHTGBM or model_type == ModelType.LIGHTGBM_REGRESSOR:
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7, -1],
            "num_leaves": [31, 50, 100],
            "min_child_samples": [20, 30, 50],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5]
        }
    else:
        # Default parameter grid for unknown model types
        return {
            "param1": [1, 2, 3],
            "param2": [0.1, 0.01, 0.001]
        }

async def _run_optimization(
    job_id: str,
    engine: MLTrainingEngine,
    model_type: ModelType,
    model_name: str,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    optimization_strategy: OptimizationStrategy,
    optimization_iterations: int,
    optimization_timeout: Optional[int],
    cv_folds: int,
    test_size: float,
    stratify: bool,
    random_state: Optional[int],
    early_stopping: bool,
    early_stopping_rounds: int,
    job_dir: str
):
    """Run the optimization process in the background"""
    status = optimization_jobs.get(job_id)
    if not status:
        # Load status from file
        try:
            with open(os.path.join(job_dir, "status.json"), "r") as f:
                status_dict = json.load(f)
                status = OptimizationJobStatus(**status_dict)
                optimization_jobs[job_id] = status
        except Exception as e:
            logging.error(f"Error loading status for job {job_id}: {str(e)}")
            return
    
    # Update status to running
    status.status = OptimizationStatus.RUNNING
    _update_job_status(job_id, status, job_dir)
    
    try:
        # Create model instance based on model_type
        model = _create_model_instance(model_type, random_state)
        
        # Configure the engine
        engine.config.optimization_strategy = optimization_strategy
        engine.config.cv_folds = cv_folds
        engine.config.optimization_iterations = optimization_iterations
        if optimization_timeout:
            engine.config.optimization_timeout = optimization_timeout
        engine.config.early_stopping = early_stopping
        engine.config.early_stopping_rounds = early_stopping_rounds
        
        # Register progress callback
        def progress_callback(iteration, total, best_score, best_params):
            status.current_iteration = iteration
            status.total_iterations = total
            status.best_score = best_score
            status.best_params = best_params
            _update_job_status(job_id, status, job_dir)
        
        # Train the model with hyperparameter optimization
        best_model, metrics = engine.train_model(
            model=model,
            model_name=model_name,
            param_grid=param_grid,
            X=X,
            y=y,
            progress_callback=progress_callback
        )
        
        # Update job status with results
        status.status = OptimizationStatus.COMPLETED
        status.completion_time = time.time()
        status.best_score = _get_best_score(metrics, engine.config.task_type)
        status.best_params = engine.models[model_name].get("params", {}) if model_name in engine.models else {}
        _update_job_status(job_id, status, job_dir)
        
        # Save the results
        results = {
            "job_id": job_id,
            "model_name": model_name,
            "model_type": model_type,
            "best_score": status.best_score,
            "best_params": status.best_params,
            "metrics": metrics,
            "completion_time": status.completion_time,
            "elapsed_time": status.completion_time - status.start_time if status.completion_time else None
        }
        
        with open(os.path.join(job_dir, "results.json"), "w") as f:
            # Handle numpy types for JSON serialization
            json.dump(results, f, default=lambda o: float(o) if isinstance(o, np.number) else str(o))
        
    except Exception as e:
        logging.error(f"Optimization error for job {job_id}: {str(e)}")
        status.status = OptimizationStatus.FAILED
        status.error = str(e)
        status.completion_time = time.time()
        _update_job_status(job_id, status, job_dir)

def _update_job_status(job_id: str, status: OptimizationJobStatus, job_dir: str):
    """Update and save job status"""
    # Update in-memory status
    optimization_jobs[job_id] = status
    
    # Save to file
    try:
        with open(os.path.join(job_dir, "status.json"), "w") as f:
            json.dump(status.dict(), f)
    except Exception as e:
        logging.error(f"Error saving status for job {job_id}: {str(e)}")

def _create_model_instance(model_type: ModelType, random_state: Optional[int] = None):
    """Create a model instance based on the model type"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    
    # Optional model imports
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        XGBClassifier, XGBRegressor = None, None
    
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError:
        LGBMClassifier, LGBMRegressor = None, None
    
    # Model mapping
    model_map = {
        ModelType.RANDOM_FOREST: RandomForestClassifier,
        ModelType.GRADIENT_BOOSTING: GradientBoostingClassifier,
        ModelType.SVM: SVC,
        ModelType.LOGISTIC_REGRESSION: LogisticRegression,
        ModelType.DECISION_TREE: DecisionTreeClassifier,
        ModelType.KNN: KNeighborsClassifier,
        ModelType.XGBOOST: XGBClassifier,
        ModelType.LIGHTGBM: LGBMClassifier,
        
        ModelType.RANDOM_FOREST_REGRESSOR: RandomForestRegressor,
        ModelType.GRADIENT_BOOSTING_REGRESSOR: GradientBoostingRegressor,
        ModelType.SVR: SVR,
        ModelType.LINEAR_REGRESSION: LinearRegression,
        ModelType.DECISION_TREE_REGRESSOR: DecisionTreeRegressor,
        ModelType.KNN_REGRESSOR: KNeighborsRegressor,
        ModelType.XGBOOST_REGRESSOR: XGBRegressor,
        ModelType.LIGHTGBM_REGRESSOR: LGBMRegressor
    }
    
    model_class = model_map.get(model_type)
    
    if model_class is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Initialize the model with random_state if applicable
    try:
        if hasattr(model_class, 'random_state'):
            return model_class(random_state=random_state)
        else:
            return model_class()
    except Exception as e:
        raise ValueError(f"Error initializing model: {str(e)}")

def _get_best_score(metrics: Dict[str, Any], task_type: TaskType) -> float:
    """Extract the best score from metrics based on task type"""
    if task_type == TaskType.CLASSIFICATION:
        # For classification, prioritize common metrics
        for metric in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
            if metric in metrics:
                return float(metrics[metric])
    else:
        # For regression, prioritize common metrics (lower is better)
        for metric in ['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error']:
            if metric in metrics and not metric.startswith('neg_'):
                return float(metrics[metric])
            elif metric in metrics:
                return -float(metrics[metric])  # Convert negative to positive
        
        # For error metrics where lower is better
        for metric in ['mse', 'rmse', 'mae']:
            if metric in metrics:
                return -float(metrics[metric])  # Negate so higher is better
    
    # Default to first metric in the dict
    return float(next(iter(metrics.values()))) if metrics else 0.0

@router.get("/status/{job_id}")
async def get_optimization_status(job_id: str):
    """Get status of an optimization job"""
    # Check if job exists in memory
    if job_id in optimization_jobs:
        status = optimization_jobs[job_id]
        
        # Calculate elapsed time
        if status.completion_time:
            elapsed = status.completion_time - status.start_time
        else:
            elapsed = time.time() - status.start_time
        
        status_dict = status.dict()
        status_dict["elapsed_time"] = elapsed
        
        return status_dict
    
    # If not in memory, try to load from file
    opt_dir = os.path.join(os.getcwd(), "optimization_jobs")
    job_dir = os.path.join(opt_dir, job_id)
    status_file = os.path.join(job_dir, "status.json")
    
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                status_dict = json.load(f)
            
            # Calculate elapsed time
            if status_dict.get("completion_time"):
                elapsed = status_dict["completion_time"] - status_dict["start_time"]
            else:
                elapsed = time.time() - status_dict["start_time"]
            
            status_dict["elapsed_time"] = elapsed
            
            return status_dict
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading status file: {str(e)}")
    
    raise HTTPException(status_code=404, detail=f"Optimization job {job_id} not found")

@router.get("/jobs")
async def list_optimization_jobs():
    """List all optimization jobs"""
    # Combine in-memory jobs with stored jobs
    all_jobs = {}
    
    # Add in-memory jobs
    for job_id, status in optimization_jobs.items():
        all_jobs[job_id] = {
            "job_id": job_id,
            "model_name": status.model_name,
            "status": status.status,
            "start_time": datetime.fromtimestamp(status.start_time).isoformat(),
            "elapsed_time": time.time() - status.start_time if not status.completion_time else status.completion_time - status.start_time
        }
    
    # Add jobs from disk
    opt_dir = os.path.join(os.getcwd(), "optimization_jobs")
    if os.path.exists(opt_dir):
        for job_id in os.listdir(opt_dir):
            if job_id not in all_jobs:
                status_file = os.path.join(opt_dir, job_id, "status.json")
                if os.path.exists(status_file):
                    try:
                        with open(status_file, "r") as f:
                            status = json.load(f)
                        
                        all_jobs[job_id] = {
                            "job_id": job_id,
                            "model_name": status.get("model_name", "unknown"),
                            "status": status.get("status", "unknown"),
                            "start_time": datetime.fromtimestamp(status.get("start_time", 0)).isoformat(),
                            "elapsed_time": time.time() - status.get("start_time", 0) if not status.get("completion_time") 
                                               else status.get("completion_time") - status.get("start_time", 0)
                        }
                    except Exception:
                        pass
    
    return {
        "jobs": list(all_jobs.values()),
        "count": len(all_jobs)
    }

@router.delete("/jobs/{job_id}")
async def delete_optimization_job(job_id: str):
    """Delete an optimization job"""
    # Remove from in-memory tracking
    if job_id in optimization_jobs:
        del optimization_jobs[job_id]
    
    # Delete job directory if it exists
    opt_dir = os.path.join(os.getcwd(), "optimization_jobs")
    job_dir = os.path.join(opt_dir, job_id)
    
    if os.path.exists(job_dir):
        import shutil
        try:
            shutil.rmtree(job_dir)
            return {"message": f"Optimization job {job_id} deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting job directory: {str(e)}")
    
    raise HTTPException(status_code=404, detail=f"Optimization job {job_id} not found")
