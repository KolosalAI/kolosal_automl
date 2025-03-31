from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import tempfile
import joblib
from datetime import datetime
import time

# Import your existing engine
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType, OptimizationStrategy
# Import necessary model classes
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

# Optional model imports (check if available first)
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None
router = APIRouter(prefix="/ml-engine", tags=["ML Training Engine"])

# Dependency to get the ML engine instance
def get_ml_engine():
    # This could be a singleton or loaded from a config
    config = MLTrainingEngineConfig()  # Load your config
    return MLTrainingEngine(config)

# ----- Data Models -----

class ModelMetadata(BaseModel):
    name: str
    description: Optional[str] = None
    tags: List[str] = []
    version: Optional[str] = None
    
class FeatureImportanceParams(BaseModel):
    model_name: Optional[str] = None
    top_n: int = 20
    include_plot: bool = True
    
class ErrorAnalysisParams(BaseModel):
    model_name: Optional[str] = None
    n_samples: int = 100
    include_plot: bool = True
    
class DataDriftParams(BaseModel):
    drift_threshold: float = 0.1

class ModelComparisonParams(BaseModel):
    model_names: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    include_plot: bool = True
    
class ModelExportParams(BaseModel):
    model_name: Optional[str] = None
    format: str = "sklearn"
    include_pipeline: bool = True
    
class TrainModelParams(BaseModel):
    model_type: str
    model_name: str
    param_grid: Dict[str, Any]
    test_size: float = 0.2
    stratify: bool = True
    
class BatchPredictionParams(BaseModel):
    model_name: Optional[str] = None
    batch_size: Optional[int] = None
    return_proba: bool = False
    
# ----- API Endpoints -----

@router.post("/train-model")
async def train_model(
    background_tasks: BackgroundTasks,
    params: TrainModelParams,
    data_file: UploadFile = File(...),
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Train a new model with hyperparameter optimization"""
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
        
        # Create model instance based on model_type
        model_cls = _get_model_class(params.model_type)
        model = model_cls()
        
        # Start training in background
        background_tasks.add_task(
            engine.train_model,
            model=model,
            model_name=params.model_name,
            param_grid=params.param_grid,
            X=X,
            y=y,
            X_test=None,  # Will be split internally
            y_test=None,
            model_metadata={"source": "API", "created_at": datetime.now().isoformat()}
        )
        
        return {
            "status": "training_started",
            "message": f"Model '{params.model_name}' training has been started",
            "model_name": params.model_name
        }
    finally:
        os.unlink(temp_file.name)

@router.get("/models")
async def list_models(engine: MLTrainingEngine = Depends(get_ml_engine)):
    """Get a list of all trained models with their metrics"""
    models_info = {}
    
    for model_name, model_data in engine.models.items():
        # Extract relevant info
        models_info[model_name] = {
            "type": type(model_data.get("model", "")).__name__,
            "metrics": model_data.get("metrics", {}),
            "training_time": model_data.get("training_time"),
            "timestamp": model_data.get("timestamp"),
            "is_best": (engine.best_model == model_name)
        }
    
    return {
        "models": models_info,
        "count": len(models_info),
        "best_model": engine.best_model
    }

@router.get("/models/{model_name}")
async def get_model_details(model_name: str, engine: MLTrainingEngine = Depends(get_ml_engine)):
    """Get detailed information about a specific model"""
    if model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model_data = engine.models[model_name]
    
    # Extract all relevant information
    result = {
        "name": model_name,
        "type": type(model_data.get("model", "")).__name__,
        "metrics": model_data.get("metrics", {}),
        "params": model_data.get("params", {}),
        "cv_results": model_data.get("cv_results", {}),
        "training_time": model_data.get("training_time"),
        "timestamp": model_data.get("timestamp"),
        "is_best": (engine.best_model == model_name),
        "feature_names": model_data.get("feature_names", []),
        "dataset_shape": model_data.get("dataset_shape", {})
    }
    
    return result

@router.post("/predict")
async def predict(
    batch_data: UploadFile = File(...),
    params: BatchPredictionParams = Depends(),
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Make predictions using a trained model"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await batch_data.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Make predictions
        predictions = engine.predict(
            X=df,
            model_name=params.model_name,  # Uses best model if None
            return_proba=params.return_proba,
            batch_size=params.batch_size
        )
        
        # Convert predictions to list
        if isinstance(predictions, np.ndarray):
            if predictions.ndim > 1:
                # For probability predictions
                pred_list = [list(map(float, row)) for row in predictions]
            else:
                # For regular predictions
                pred_list = list(map(float, predictions))
        else:
            pred_list = predictions
            
        return {
            "predictions": pred_list,
            "model_used": params.model_name or engine.best_model,
            "row_count": len(df)
        }
    finally:
        os.unlink(temp_file.name)

@router.post("/evaluate/{model_name}")
async def evaluate_model(
    model_name: str,
    test_data: UploadFile = File(...),
    detailed: bool = False,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Evaluate a model with test data"""
    if model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await test_data.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Extract X and y from dataframe (assuming last column is target)
        y_test = df.iloc[:, -1]
        X_test = df.iloc[:, :-1]
        
        # Evaluate model
        metrics = engine.evaluate_model(
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            detailed=detailed
        )
        
        return {
            "model_name": model_name,
            "metrics": metrics,
            "test_samples": len(X_test)
        }
    finally:
        os.unlink(temp_file.name)

@router.post("/feature-importance")
async def feature_importance(
    params: FeatureImportanceParams,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Generate feature importance report"""
    result = engine.generate_feature_importance_report(
        model_name=params.model_name,  # Uses best model if None
        top_n=params.top_n,
        include_plot=params.include_plot,
        output_file=None  # No file output for API call
    )
    
    # If there was an error
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Return the importance data - remove file paths
    cleaned_result = {k: v for k, v in result.items() if not k.endswith('_path')}
    
    return cleaned_result

@router.post("/error-analysis")
async def error_analysis(
    test_data: UploadFile = File(...),
    params: ErrorAnalysisParams = Depends(),
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Perform error analysis on test data"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await test_data.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Extract X and y from dataframe (assuming last column is target)
        y_test = df.iloc[:, -1]
        X_test = df.iloc[:, :-1]
        
        # Perform error analysis
        analysis = engine.perform_error_analysis(
            model_name=params.model_name,  # Uses best model if None
            X_test=X_test,
            y_test=y_test,
            n_samples=params.n_samples,
            include_plot=params.include_plot,
            output_file=None  # No file output for API call
        )
        
        # If there was an error
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        # Return the analysis - remove file paths that can't be accessed via API
        cleaned_analysis = {k: v for k, v in analysis.items() if not k.endswith('_plot')}
        
        return cleaned_analysis
    finally:
        os.unlink(temp_file.name)

@router.post("/data-drift")
async def data_drift(
    new_data: UploadFile = File(...),
    reference_data: Optional[UploadFile] = File(None),
    params: DataDriftParams = Depends(),
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Detect data drift between new data and reference data"""
    # Save new data temporarily
    new_data_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    reference_data_file = None
    
    try:
        # Handle new data
        contents = await new_data.read()
        new_data_file.write(contents)
        new_data_file.close()
        new_df = pd.read_csv(new_data_file.name)
        
        # Handle reference data if provided
        if reference_data:
            reference_data_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            contents = await reference_data.read()
            reference_data_file.write(contents)
            reference_data_file.close()
            ref_df = pd.read_csv(reference_data_file.name)
        else:
            ref_df = None  # Will use training data
        
        # Detect drift
        drift_results = engine.detect_data_drift(
            new_data=new_df,
            reference_data=ref_df,
            drift_threshold=params.drift_threshold
        )
        
        # If there was an error
        if "error" in drift_results:
            raise HTTPException(status_code=400, detail=drift_results["error"])
        
        # Return the drift results - remove file paths
        cleaned_results = {k: v for k, v in drift_results.items() if not k.endswith('_plot')}
        
        return cleaned_results
    finally:
        os.unlink(new_data_file.name)
        if reference_data_file:
            os.unlink(reference_data_file.name)

@router.post("/compare-models")
async def compare_models(
    params: ModelComparisonParams,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Compare multiple trained models"""
    # Check if the models exist
    all_models = list(engine.models.keys())
    if params.model_names:
        for model in params.model_names:
            if model not in all_models:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
    
    # Compare models
    comparison = engine.compare_models(
        model_names=params.model_names,  # Uses all models if None
        metrics=params.metrics,
        include_plot=params.include_plot,
        output_file=None  # No file output for API call
    )
    
    # If there was an error
    if "error" in comparison:
        raise HTTPException(status_code=400, detail=comparison["error"])
    
    # Return the comparison - remove file paths
    cleaned_comparison = {k: v for k, v in comparison.items() if not (k.endswith('_plot') or k == 'report_path')}
    
    return cleaned_comparison

@router.post("/export-model/{model_name}")
async def export_model(
    model_name: str,
    params: ModelExportParams,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Export a model in various formats"""
    if model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Create temp directory for export
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export model
        output_path = engine.export_model(
            model_name=model_name,
            format=params.format,
            output_dir=temp_dir,
            include_pipeline=params.include_pipeline
        )
        
        if not output_path:
            raise HTTPException(status_code=400, detail=f"Failed to export model in {params.format} format")
            
        # Read the exported model file
        with open(output_path, 'rb') as f:
            model_bytes = f.read()
            
        # Return the model as downloadable file
        return {
            "model_name": model_name,
            "format": params.format,
            "file_size_bytes": len(model_bytes),
            "download_url": f"/ml-engine/download/{model_name}/{params.format}"  # Frontend will need to implement this
        }

@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Delete a trained model"""
    if model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Delete model
    try:
        del engine.models[model_name]
        
        # If this was the best model, update best model
        if engine.best_model == model_name:
            engine.best_model = None
            engine.best_score = -float('inf') if engine.config.task_type != TaskType.REGRESSION else float('inf')
            
            # Find new best model
            for name in engine.models:
                engine._update_best_model(name)
        
        return {"message": f"Model '{model_name}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

@router.post("/generate-report")
async def generate_report(
    include_plots: bool = True,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Generate a comprehensive report of all models"""
    # Create temp file for the report
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
        try:
            output_path = engine.generate_reports(
                output_file=temp_file.name,
                include_plots=include_plots
            )
            
            if not output_path:
                raise HTTPException(status_code=400, detail="Failed to generate report")
                
            # Read report
            with open(output_path, 'r') as f:
                report_content = f.read()
                
            return {
                "report": report_content,
                "model_count": len(engine.models),
                "best_model": engine.best_model
            }
        finally:
            os.unlink(temp_file.name)

@router.post("/save-model/{model_name}")
async def save_model(
    model_name: str,
    version_tag: Optional[str] = None,
    include_preprocessor: bool = True,
    include_metadata: bool = True,
    compression_level: int = 5,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Save a model to disk"""
    if model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Save model
    success, filepath = engine.save_model(
        model_name=model_name,
        filepath=None,  # Use default
        access_code=None,
        compression_level=compression_level,
        include_preprocessor=include_preprocessor,
        include_metadata=include_metadata,
        version_tag=version_tag
    )
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {filepath}")
    
    return {
        "model_name": model_name,
        "filepath": filepath,
        "size_bytes": os.path.getsize(filepath),
        "timestamp": datetime.now().isoformat()
    }

@router.post("/load-model")
async def load_model(
    model_file: UploadFile = File(...),
    validate_metrics: bool = True,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Load a saved model"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    try:
        contents = await model_file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Load model
        model = engine.load_model(
            filepath=temp_file.name,
            access_code=None,
            validate_metrics=validate_metrics
        )
        
        if model is None:
            raise HTTPException(status_code=400, detail="Failed to load model")
        
        # Get model name from loaded model
        model_name = None
        for name, model_data in engine.models.items():
            if model_data.get("loaded_from") == temp_file.name:
                model_name = name
                break
        
        return {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "is_best": (engine.best_model == model_name),
            "metrics": engine.models[model_name].get("metrics", {}) if model_name else {}
        }
    finally:
        os.unlink(temp_file.name)

@router.post("/batch-inference")
async def batch_inference(
    batch_data: List[UploadFile] = File(...),
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    return_proba: bool = False,
    parallel: bool = True,
    timeout: Optional[int] = None,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Run batch inference on multiple files"""
    # Prepare temp files for each batch
    temp_files = []
    try:
        # Process each uploaded file
        batch_dataframes = []
        for data_file in batch_data:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            contents = await data_file.read()
            temp_file.write(contents)
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Read as dataframe
            df = pd.read_csv(temp_file.name)
            batch_dataframes.append(df)
        
        # Run batch inference
        batch_results = engine.run_batch_inference(
            data_generator=batch_dataframes,
            batch_size=batch_size,
            model_name=model_name,  # Uses best model if None
            return_proba=return_proba,
            parallel=parallel,
            timeout=timeout
        )
        
        # Format results for API response
        formatted_results = []
        for i, (batch, result) in enumerate(zip(batch_dataframes, batch_results)):
            # Convert numpy arrays to lists
            if isinstance(result, np.ndarray):
                if result.ndim > 1:
                    result_list = [list(map(float, row)) for row in result]
                else:
                    result_list = list(map(float, result))
            else:
                result_list = result
                
            formatted_results.append({
                "batch_index": i,
                "batch_size": len(batch),
                "predictions": result_list
            })
        
        return {
            "model_used": model_name or engine.best_model,
            "batch_count": len(batch_dataframes),
            "total_samples": sum(len(df) for df in batch_dataframes),
            "results": formatted_results
        }
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            os.unlink(temp_file)

@router.post("/shutdown")
async def shutdown_engine(engine: MLTrainingEngine = Depends(get_ml_engine)):
    """Shut down the ML training engine"""
    try:
        engine.shutdown()
        return {"message": "ML Training Engine shut down successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during shutdown: {str(e)}")

# ----- Utility Functions -----

def _get_model_class(model_type: str):
    """Get the appropriate model class based on model type"""
    # This is a mapping of string identifiers to model classes
    model_map = {
        # Classification models
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "svm": SVC,
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "knn": KNeighborsClassifier,
        "naive_bayes": GaussianNB,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
        
        # Regression models
        "random_forest_regressor": RandomForestRegressor,
        "gradient_boosting_regressor": GradientBoostingRegressor,
        "svr": SVR,
        "linear_regression": LinearRegression,
        "decision_tree_regressor": DecisionTreeRegressor,
        "knn_regressor": KNeighborsRegressor,
        "xgboost_regressor": XGBRegressor,
        "lightgbm_regressor": LGBMRegressor,
    }
    
    if model_type not in model_map:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
        
    return model_map[model_type]

@router.post("/models/{model_name}/quantize")
async def quantize_model(
    model_name: str,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Quantize a model for faster inference"""
    if model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model_data = engine.models[model_name]
    
    # Generate temporary file path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    temp_filepath = os.path.join(tempfile.gettempdir(), f"{model_name}-{timestamp}.pkl")
    
    try:
        # Save model to get original filepath for quantization
        success, filepath = engine.save_model(model_name=model_name, filepath=temp_filepath)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to save model: {filepath}")
        
        # Quantize the model
        quantized_path = engine._save_quantized_model(model_data, model_name, filepath)
        
        if not quantized_path:
            raise HTTPException(status_code=500, detail="Failed to quantize model")
            
        # Get quantized model size
        quantized_size = os.path.getsize(quantized_path)
        original_size = os.path.getsize(filepath)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        
        return {
            "model_name": model_name,
            "original_size_bytes": original_size,
            "quantized_size_bytes": quantized_size,
            "compression_ratio": compression_ratio,
            "quantized_filepath": quantized_path
        }
    finally:
        # Clean up temp files
        if os.path.exists(temp_filepath):
            os.unlink(temp_filepath)

@router.post("/models/{model_name}/transfer-learning")
async def transfer_learning(
    model_name: str,
    new_data: UploadFile = File(...),
    learning_rate: float = 0.01,
    epochs: int = 10,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Adapt a pre-trained model to new data through transfer learning"""
    if model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await new_data.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Extract X and y from dataframe (assuming last column is target)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        
        # Get the original model
        original_model = engine.models[model_name]["model"]
        
        # This is a simplified approach - in a real system, you'd have a proper transfer learning implementation
        # For example, with neural networks you might freeze some layers and only train others
        
        # For simple sklearn models, we'll just do a quick refit with warm_start if available
        if hasattr(original_model, 'warm_start'):
            # Clone the model
            import copy
            new_model = copy.deepcopy(original_model)
            
            # Set warm start
            new_model.warm_start = True
            
            # If model has learning rate, set it
            if hasattr(new_model, 'learning_rate'):
                new_model.learning_rate = learning_rate
            
            # Fit on new data
            new_model.fit(X, y)
            
            # Create a new model entry
            new_model_name = f"{model_name}_transfer"
            engine.models[new_model_name] = {
                "model": new_model,
                "params": engine.models[model_name]["params"],
                "metrics": {},  # Will need evaluation
                "timestamp": time.time(),
                "training_time": 0,  # Not measured
                "feature_names": engine.models[model_name].get("feature_names", []),
                "transferred_from": model_name
            }
            
            return {
                "original_model": model_name,
                "new_model": new_model_name,
                "message": f"Transfer learning complete. New model '{new_model_name}' created.",
                "samples_used": len(X)
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Model type {type(original_model).__name__} doesn't support transfer learning via warm_start"
            )
    finally:
        os.unlink(temp_file.name)

@router.post("/ensemble-models")
async def create_ensemble(
    model_names: List[str],
    ensemble_name: str,
    voting_type: str = "soft",  # 'hard' or 'soft' for classification, 'mean' for regression
    weights: Optional[List[float]] = None,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Create an ensemble model from multiple trained models"""
    # Validate all models exist
    for model_name in model_names:
        if model_name not in engine.models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Check if ensemble name already exists
    if ensemble_name in engine.models:
        raise HTTPException(status_code=400, detail=f"Model '{ensemble_name}' already exists")
    
    try:
        # Get the individual models
        models = [engine.models[name]["model"] for name in model_names]
        
        # Create appropriate ensemble based on task type
        if engine.config.task_type == TaskType.CLASSIFICATION:
            from sklearn.ensemble import VotingClassifier
            ensemble = VotingClassifier(
                estimators=[(f"model_{i}", model) for i, model in enumerate(models)],
                voting=voting_type,
                weights=weights
            )
        else:  # Regression
            from sklearn.ensemble import VotingRegressor
            ensemble = VotingRegressor(
                estimators=[(f"model_{i}", model) for i, model in enumerate(models)],
                weights=weights
            )
        
        # Store ensemble model without training (assumes models are already trained)
        engine.models[ensemble_name] = {
            "model": ensemble,
            "params": {
                "base_models": model_names,
                "voting_type": voting_type,
                "weights": weights
            },
            "metrics": {},  # Will need evaluation
            "timestamp": time.time(),
            "is_ensemble": True
        }
        
        return {
            "ensemble_name": ensemble_name,
            "base_models": model_names,
            "voting_type": voting_type,
            "weights": weights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ensemble: {str(e)}")

@router.post("/models/{model_name}/calibrate")
async def calibrate_model(
    model_name: str,
    calibration_data: UploadFile = File(...),
    method: str = "isotonic",  # 'isotonic' or 'sigmoid'
    cv: int = 5,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Calibrate a classification model's probability estimates"""
    if model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Ensure classification task
    if engine.config.task_type != TaskType.CLASSIFICATION:
        raise HTTPException(status_code=400, detail="Calibration only applies to classification models")
    
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await calibration_data.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Extract X and y from dataframe (assuming last column is target)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        
        # Get the original model
        original_model = engine.models[model_name]["model"]
        
        try:
            from sklearn.calibration import CalibratedClassifierCV
            
            # Create calibrated model
            calibrated_model = CalibratedClassifierCV(
                base_estimator=original_model,
                method=method,
                cv=cv
            )
            
            # Fit the calibrator
            calibrated_model.fit(X, y)
            
            # Create a new model entry
            calibrated_name = f"{model_name}_calibrated"
            engine.models[calibrated_name] = {
                "model": calibrated_model,
                "params": {
                    **engine.models[model_name]["params"],
                    "calibration_method": method,
                    "calibration_cv": cv
                },
                "metrics": {},  # Will need evaluation
                "timestamp": time.time(),
                "calibrated_from": model_name
            }
            
            return {
                "original_model": model_name,
                "calibrated_model": calibrated_name,
                "calibration_method": method,
                "samples_used": len(X)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Calibration error: {str(e)}")
    finally:
        os.unlink(temp_file.name)

@router.post("/interpret-model")
async def interpret_model(
    model_name: Optional[str] = None,
    sample_data: Optional[UploadFile] = File(None),
    method: str = "shap",  # 'shap', 'lime', 'eli5'
    background_samples: int = 100,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Generate model-agnostic interpretations for model predictions"""
    # Get model to interpret
    if model_name is None and engine.best_model is not None:
        model_name = engine.best_model
        model = engine.models[model_name]["model"]
    elif model_name in engine.models:
        model = engine.models[model_name]["model"]
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # If sample data provided, use it for interpretation
    if sample_data:
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        try:
            contents = await sample_data.read()
            temp_file.write(contents)
            temp_file.close()
            
            # Read data
            df = pd.read_csv(temp_file.name)
            
            # If last column might be target, remove it
            X_sample = df.iloc[:, :-1] if df.shape[1] > 1 else df
            
            # Get feature names
            feature_names = engine.models[model_name].get("feature_names", X_sample.columns.tolist())
            
            # For SHAP interpretations
            if method.lower() == 'shap':
                try:
                    import shap
                    
                    # Create explainer based on model type
                    if hasattr(model, 'predict_proba'):
                        # For classification models
                        if hasattr(engine, '_last_X_train'):
                            # Get background data for SHAP
                            background_data = engine._last_X_train.sample(min(background_samples, len(engine._last_X_train)))
                        else:
                            background_data = X_sample  # Use sample as background if no training data
                        
                        # Create explainer
                        if hasattr(model, 'predict_proba'):
                            explainer = shap.KernelExplainer(model.predict_proba, background_data)
                        else:
                            explainer = shap.KernelExplainer(model.predict, background_data)
                        
                        # Calculate SHAP values
                        shap_values = explainer.shap_values(X_sample)
                        
                        # For classification, the result is a list of arrays (one per class)
                        shap_results = []
                        for class_idx, class_shap_values in enumerate(shap_values):
                            class_results = []
                            for i, row in enumerate(class_shap_values):
                                # Create mapping of feature to SHAP value
                                feature_values = {
                                    feature_names[j]: {
                                        "value": float(X_sample.iloc[i, j]) if j < len(X_sample.columns) else None,
                                        "shap_value": float(row[j]),
                                    }
                                    for j in range(len(row))
                                }
                                class_results.append({
                                    "sample_idx": i,
                                    "features": feature_values,
                                    "base_value": float(explainer.expected_value[class_idx]),
                                    "prediction": float(explainer.expected_value[class_idx] + sum(row))
                                })
                            shap_results.append({
                                "class": class_idx,
                                "samples": class_results
                            })
                        
                        return {
                            "model_name": model_name,
                            "method": "shap",
                            "model_type": "classification",
                            "results": shap_results
                        }
                    else:
                        # For regression models
                        if hasattr(engine, '_last_X_train'):
                            # Get background data for SHAP
                            background_data = engine._last_X_train.sample(min(background_samples, len(engine._last_X_train)))
                        else:
                            background_data = X_sample  # Use sample as background if no training data
                        
                        explainer = shap.KernelExplainer(model.predict, background_data)
                        shap_values = explainer.shap_values(X_sample)
                        
                        # Process results
                        shap_results = []
                        for i, row in enumerate(shap_values):
                            # Create mapping of feature to SHAP value
                            feature_values = {
                                feature_names[j]: {
                                    "value": float(X_sample.iloc[i, j]) if j < len(X_sample.columns) else None,
                                    "shap_value": float(row[j]),
                                }
                                for j in range(len(row))
                            }
                            shap_results.append({
                                "sample_idx": i,
                                "features": feature_values,
                                "base_value": float(explainer.expected_value),
                                "prediction": float(explainer.expected_value + sum(row))
                            })
                        
                        return {
                            "model_name": model_name,
                            "method": "shap",
                            "model_type": "regression",
                            "results": shap_results
                        }
                except ImportError:
                    raise HTTPException(status_code=400, detail="SHAP library not available")
            
            # For LIME interpretations 
            elif method.lower() == 'lime':
                try:
                    from lime import lime_tabular
                    
                    # Create explainer
                    explainer = lime_tabular.LimeTabularExplainer(
                        X_sample.values,
                        feature_names=feature_names,
                        class_names=["Negative", "Positive"] if engine.config.task_type == TaskType.CLASSIFICATION else None,
                        mode="classification" if engine.config.task_type == TaskType.CLASSIFICATION else "regression"
                    )
                    
                    # Generate explanations for each sample
                    lime_results = []
                    for i in range(min(10, len(X_sample))):  # Limit to 10 samples for API response
                        # Get explanation
                        if engine.config.task_type == TaskType.CLASSIFICATION and hasattr(model, 'predict_proba'):
                            exp = explainer.explain_instance(
                                X_sample.iloc[i].values, 
                                model.predict_proba, 
                                num_features=10
                            )
                        else:
                            exp = explainer.explain_instance(
                                X_sample.iloc[i].values, 
                                model.predict, 
                                num_features=10
                            )
                        
                        # Extract feature importance
                        feature_importance = {}
                        for feature, importance in exp.as_list():
                            feature_importance[feature] = float(importance)
                        
                        lime_results.append({
                            "sample_idx": i,
                            "feature_importance": feature_importance,
                            "prediction": float(model.predict(X_sample.iloc[i:i+1].values)[0])
                        })
                    
                    return {
                        "model_name": model_name,
                        "method": "lime",
                        "model_type": "classification" if engine.config.task_type == TaskType.CLASSIFICATION else "regression",
                        "results": lime_results
                    }
                except ImportError:
                    raise HTTPException(status_code=400, detail="LIME library not available")
            
            # For ELI5 interpretations
            elif method.lower() == 'eli5':
                try:
                    import eli5
                    from eli5.sklearn import PermutationImportance
                    
                    # Create permutation importance explainer
                    perm = PermutationImportance(
                        model, 
                        random_state=engine.config.random_state,
                        n_iter=10
                    )
                    
                    # Fit the explainer
                    perm.fit(X_sample, y=None)  # Without labels, will use model's predictions
                    
                    # Get feature weights
                    feature_weights = {}
                    for feature_idx, weight in enumerate(perm.feature_importances_):
                        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
                        feature_weights[feature_name] = float(weight)
                    
                    # Format the response
                    return {
                        "model_name": model_name,
                        "method": "eli5",
                        "feature_weights": feature_weights,
                        "feature_importances_std": perm.feature_importances_std_.tolist() if hasattr(perm, 'feature_importances_std_') else None
                    }
                except ImportError:
                    raise HTTPException(status_code=400, detail="ELI5 library not available")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported interpretation method: {method}")
        finally:
            os.unlink(temp_file.name)
    else:
        # Without sample data, just return model's built-in feature importance if available
        feature_importance = engine._get_feature_importance(model)
        
        if feature_importance is None:
            raise HTTPException(status_code=400, detail="Model doesn't provide built-in feature importance. Please provide sample data for model-agnostic interpretation.")
            
        # Get feature names
        feature_names = engine.models[model_name].get("feature_names", [f"feature_{i}" for i in range(len(feature_importance))])
        
        # Create a mapping of feature names to importance
        importance_dict = {
            feature_names[i]: float(importance) 
            for i, importance in enumerate(feature_importance) 
            if i < len(feature_names)
        }
        
        return {
            "model_name": model_name,
            "method": "built_in",
            "feature_importance": importance_dict
        }

@router.post("/explain-prediction")
async def explain_prediction(
    sample_data: UploadFile = File(...),
    model_name: Optional[str] = None,
    method: str = "shap",
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Explain a specific prediction in detail"""
    # Defer to the interpret-model endpoint with specific parameters
    return await interpret_model(
        model_name=model_name,
        sample_data=sample_data,
        method=method,
        background_samples=50,  # Smaller number for faster response
        engine=engine
    )

@router.post("/health-check")
async def model_health_check(
    test_data: Optional[UploadFile] = File(None),
    model_name: Optional[str] = None,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Check model health including performance, drift, and inference time"""
    # Get model to check
    if model_name is None and engine.best_model is not None:
        model_name = engine.best_model
    elif model_name is not None and model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    health_report = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": []
    }
    
    # Basic model check
    model_check = {"name": "model_exists", "status": "passed"}
    if model_name is None or model_name not in engine.models:
        model_check["status"] = "failed"
        model_check["message"] = "No model available"
        health_report["status"] = "unhealthy"
    health_report["checks"].append(model_check)
    
    # If model check failed, return early
    if model_check["status"] == "failed":
        return health_report
    
    # Get model data
    model_data = engine.models[model_name]
    model = model_data["model"]
    
    # Check if model can make predictions
    prediction_check = {"name": "can_predict", "status": "passed"}
    try:
        # Create a small dummy input for testing prediction
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            n_features = 10  # Default guess
            
        dummy_input = np.zeros((1, n_features))
        _ = model.predict(dummy_input)
    except Exception as e:
        prediction_check["status"] = "failed"
        prediction_check["message"] = f"Model prediction failed: {str(e)}"
        health_report["status"] = "unhealthy"
    health_report["checks"].append(prediction_check)
    
    # Inference time check
    inference_time_check = {"name": "inference_time", "status": "passed"}
    try:
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            n_features = 10  # Default guess
            
        dummy_input = np.zeros((100, n_features))  # Use 100 rows for better timing
        
        start_time = time.time()
        _ = model.predict(dummy_input)
        inference_time = time.time() - start_time
        
        inference_time_ms = inference_time * 1000 / 100  # Per-sample time in ms
        inference_time_check["inference_time_ms"] = inference_time_ms
        
        # Flag slow inference (this threshold would be configurable in a real system)
        if inference_time_ms > 10:  # 10ms per sample is arbitrary
            inference_time_check["status"] = "warning"
            inference_time_check["message"] = f"Inference time is high: {inference_time_ms:.2f}ms per sample"
    except Exception as e:
        inference_time_check["status"] = "failed"
        inference_time_check["message"] = f"Inference time check failed: {str(e)}"
    health_report["checks"].append(inference_time_check)
    
    # If test data is provided, run more in-depth checks
    if test_data:
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        try:
            contents = await test_data.read()
            temp_file.write(contents)
            temp_file.close()
            
            # Read data
            df = pd.read_csv(temp_file.name)
            
            # Extract X and y from dataframe (assuming last column is target)
            y_test = df.iloc[:, -1]
            X_test = df.iloc[:, :-1]
            
            # Check for data drift
            drift_check = {"name": "data_drift", "status": "passed"}
            try:
                drift_results = engine.detect_data_drift(
                    new_data=X_test,
                    reference_data=None,  # Use training data as reference
                    drift_threshold=0.2  # Higher threshold for health check
                )
                
                if "error" in drift_results:
                    drift_check["status"] = "skipped"
                    drift_check["message"] = drift_results["error"]
                elif drift_results["drift_detected"]:
                    drift_check["status"] = "warning"
                    drift_check["message"] = f"Data drift detected: {drift_results['dataset_drift']:.4f}"
                    drift_check["drift_score"] = drift_results["dataset_drift"]
                    drift_check["drifted_features"] = drift_results["drifted_features"]
            except Exception as e:
                drift_check["status"] = "failed"
                drift_check["message"] = f"Drift check failed: {str(e)}"
            health_report["checks"].append(drift_check)
            
            # Performance check
            performance_check = {"name": "performance", "status": "passed"}
            try:
                metrics = engine.evaluate_model(
                    model_name=model_name,
                    X_test=X_test,
                    y_test=y_test,
                    detailed=False
                )
                
                if "error" in metrics:
                    performance_check["status"] = "failed"
                    performance_check["message"] = metrics["error"]
                else:
                    performance_check["metrics"] = metrics
                    
                    # Add thresholds for common metrics (these would be configurable)
                    if engine.config.task_type == TaskType.CLASSIFICATION:
                        if "accuracy" in metrics and metrics["accuracy"] < 0.7:
                            performance_check["status"] = "warning"
                            performance_check["message"] = f"Accuracy below threshold: {metrics['accuracy']:.4f}"
                    elif engine.config.task_type == TaskType.REGRESSION:
                        if "r2" in metrics and metrics["r2"] < 0.5:
                            performance_check["status"] = "warning"
                            performance_check["message"] = f"R² below threshold: {metrics['r2']:.4f}"
            except Exception as e:
                performance_check["status"] = "failed"
                performance_check["message"] = f"Performance check failed: {str(e)}"
            health_report["checks"].append(performance_check)
            
            # Calibration check for classification models
            if engine.config.task_type == TaskType.CLASSIFICATION and hasattr(model, 'predict_proba'):
                calibration_check = {"name": "probability_calibration", "status": "passed"}
                try:
                    from sklearn.calibration import calibration_curve
                    
                    # Get probability predictions
                    y_prob = model.predict_proba(X_test)
                    
                    # For binary classification
                    if y_prob.shape[1] == 2:
                        # Get positive class probabilities
                        y_prob = y_prob[:, 1]
                        
                        # Calculate calibration curve
                        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5)
                        
                        # Calculate calibration error
                        calibration_error = np.mean(np.abs(prob_true - prob_pred))
                        
                        calibration_check["calibration_error"] = float(calibration_error)
                        
                        # Flag poor calibration
                        if calibration_error > 0.1:  # Arbitrary threshold
                            calibration_check["status"] = "warning"
                            calibration_check["message"] = f"Model probabilities may be poorly calibrated: {calibration_error:.4f}"
                    else:
                        # For multi-class, we'll skip detailed checks
                        calibration_check["status"] = "skipped"
                        calibration_check["message"] = "Calibration check skipped for multi-class model"
                except Exception as e:
                    calibration_check["status"] = "failed"
                    calibration_check["message"] = f"Calibration check failed: {str(e)}"
                health_report["checks"].append(calibration_check)
        finally:
            os.unlink(temp_file.name)
    
    # Determine overall health status
    failed_checks = [check for check in health_report["checks"] if check["status"] == "failed"]
    warning_checks = [check for check in health_report["checks"] if check["status"] == "warning"]
    
    if failed_checks:
        health_report["status"] = "unhealthy"
    elif warning_checks:
        health_report["status"] = "degraded"
    else:
        health_report["status"] = "healthy"
    
    return health_report

@router.post("/batch-process-pipeline")
async def batch_process_pipeline(
    input_data: UploadFile = File(...),
    model_name: Optional[str] = None,
    steps: List[str] = ["preprocess", "predict", "postprocess"],
    batch_size: Optional[int] = None,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Run a full batch processing pipeline with preprocessing, prediction, and postprocessing"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    
    try:
        contents = await input_data.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        original_df = df.copy()
        
        # Get model to use
        if model_name is None and engine.best_model is not None:
            model_name = engine.best_model
            model = engine.models[model_name]["model"]
        elif model_name in engine.models:
            model = engine.models[model_name]["model"]
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Execute pipeline steps
        pipeline_results = {
            "model_name": model_name,
            "input_rows": len(df),
            "steps": {},
            "execution_time": 0
        }
        
        pipeline_start = time.time()
        
        # Preprocessing step
        if "preprocess" in steps:
            preprocess_start = time.time()
            
            # Apply preprocessing if configured
            if engine.preprocessor and hasattr(engine.preprocessor, 'transform'):
                try:
                    df = engine.preprocessor.transform(df)
                    
                    # Convert back to DataFrame if needed
                    if not isinstance(df, pd.DataFrame):
                        # Try to convert to DataFrame with original column names
                        if isinstance(df, np.ndarray) and df.ndim == 2:
                            if df.shape[1] == len(original_df.columns):
                                df = pd.DataFrame(df, columns=original_df.columns)
                            else:
                                df = pd.DataFrame(df)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")
            
            pipeline_results["steps"]["preprocess"] = {
                "time_seconds": time.time() - preprocess_start,
                "output_rows": len(df),
                "output_columns": df.shape[1]
            }
        
        # Prediction step
        if "predict" in steps:
            predict_start = time.time()
            
            # Make predictions
            predictions = engine.predict(
                X=df,
                model_name=model_name,
                return_proba=False,
                batch_size=batch_size
            )
            
            # Add predictions to DataFrame
            if isinstance(predictions, np.ndarray):
                if predictions.ndim == 1:
                    df['prediction'] = predictions
                else:
                    # For multi-class probabilities
                    for i in range(predictions.shape[1]):
                        df[f'prediction_class_{i}'] = predictions[:, i]
            
            pipeline_results["steps"]["predict"] = {
                "time_seconds": time.time() - predict_start,
                "predictions_shape": list(predictions.shape) if isinstance(predictions, np.ndarray) else None
            }
        
        # Postprocessing step (example: add confidence flags, thresholding, etc.)
        if "postprocess" in steps:
            postprocess_start = time.time()
            
            # Example postprocessing for classification: add confidence flag
            if engine.config.task_type == TaskType.CLASSIFICATION and 'prediction' in df:
                # Check if probability columns exist
                prob_cols = [col for col in df.columns if col.startswith('prediction_class_')]
                
                if prob_cols:
                    # Find highest probability for each row
                    df['confidence'] = df[prob_cols].max(axis=1)
                    
                    # Flag predictions with low confidence
                    df['low_confidence'] = df['confidence'] < 0.7  # Arbitrary threshold
            
            # Example postprocessing for regression: flag outlier predictions
            elif engine.config.task_type == TaskType.REGRESSION and 'prediction' in df:
                # Calculate z-score for predictions
                mean_pred = df['prediction'].mean()
                std_pred = df['prediction'].std()
                
                if std_pred > 0:
                    df['z_score'] = (df['prediction'] - mean_pred) / std_pred
                    
                    # Flag outlier predictions
                    df['outlier_prediction'] = abs(df['z_score']) > 2.0  # 2 standard deviations
            
            pipeline_results["steps"]["postprocess"] = {
                "time_seconds": time.time() - postprocess_start,
                "added_columns": [col for col in df.columns if col not in original_df.columns]
            }
        
        # Record total execution time
        pipeline_results["execution_time"] = time.time() - pipeline_start
        
        # Save the processed data
        df.to_csv(output_file.name, index=False)
        
        # Update results with file info
        pipeline_results["output_rows"] = len(df)
        pipeline_results["output_columns"] = len(df.columns)
        pipeline_results["output_file_size_bytes"] = os.path.getsize(output_file.name)
        
        return pipeline_results
    finally:
        # Clean up temp files
        os.unlink(temp_file.name)
        os.unlink(output_file.name)