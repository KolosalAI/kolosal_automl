
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, Query, Path, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import List, Dict, Optional, Any, Union, Annotated
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import tempfile
import os
import time
import logging
import json
import uuid
from enum import Enum
from datetime import datetime

# Import inference module
from modules.engine.inference_engine import InferenceEngine, ModelType, EngineState
from modules.configs import (
    InferenceEngineConfig, QuantizationConfig, BatchProcessorConfig, 
    BatchProcessingStrategy, NormalizationType, QuantizationMode
)

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/inference", tags=["Model Inference"])

# Constants
MAX_BATCH_SIZE = 1000
DEFAULT_CACHE_TTL = 3600  # 1 hour
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./inference_results")
MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY", "./models")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data models
class InferenceRequest(BaseModel):
    """Request model for single inference requests"""
    model_id: Optional[str] = Field(None, description="ID of the model to use (if not already loaded)")
    model_version: Optional[str] = Field(None, description="Version of the model to use")
    batch_size: Optional[int] = Field(None, description="Batch size for inference")
    return_probabilities: bool = Field(False, description="Return probability scores for each class")
    include_feature_importance: bool = Field(False, description="Include feature importance in response")
    include_explanations: bool = Field(False, description="Include explanations for predictions")
    explanation_method: Optional[str] = Field("shap", description="Method to use for explanations")
    request_id: Optional[str] = Field(None, description="Client-provided request ID for tracking")

class BatchInferenceRequest(BaseModel):
    """Request model for batch inference"""
    model_id: Optional[str] = Field(None, description="ID of the model to use (if not already loaded)")
    model_version: Optional[str] = Field(None, description="Version of the model to use")
    batch_size: Optional[int] = Field(None, description="Batch size for inference")
    parallel: bool = Field(True, description="Process batches in parallel")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")
    return_probabilities: bool = Field(False, description="Return probability scores for each class")
    include_per_file_metrics: bool = Field(True, description="Include metrics for each file")
    save_results: bool = Field(False, description="Save results to file")

class StreamingJobStatus(str, Enum):
    """Status of a streaming inference job"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StreamingInferenceRequest(BaseModel):
    """Request model for streaming inference"""
    model_id: str = Field(..., description="ID of the model to use")
    model_version: Optional[str] = Field(None, description="Version of the model to use")
    batch_identifier: str = Field(..., description="Identifier for the data batch")
    output_format: str = Field("csv", description="Format for output (csv, json, parquet)")
    output_path: Optional[str] = Field(None, description="Custom output path")
    include_probabilities: bool = Field(False, description="Include probability scores")
    notify_on_completion: bool = Field(False, description="Send notification when complete")
    notification_url: Optional[str] = Field(None, description="Webhook URL for notification")

class ExplanationRequest(BaseModel):
    """Request model for prediction explanations"""
    model_id: Optional[str] = Field(None, description="ID of the model to use")
    method: str = Field("shap", description="Explanation method to use")
    n_samples: int = Field(100, description="Number of samples to explain")
    background_samples: Optional[int] = Field(None, description="Number of background samples for methods that require them")
    visualization_type: Optional[str] = Field(None, description="Type of visualization (summary, dependence, etc.)")

class ModelConfigRequest(BaseModel):
    """Request model for updating model configuration"""
    engine_config: Dict[str, Any] = Field({}, description="Engine configuration parameters")
    quantization_config: Optional[Dict[str, Any]] = Field(None, description="Quantization configuration")
    batch_config: Optional[Dict[str, Any]] = Field(None, description="Batch processor configuration")
    feature_names: Optional[List[str]] = Field(None, description="Feature names for the model")

class ModelMetadata(BaseModel):
    """Model for returning model metadata"""
    model_id: str
    model_version: Optional[str] = None
    model_type: str
    feature_count: Optional[int] = None
    feature_names: Optional[List[str]] = None
    model_created: Optional[str] = None
    last_updated: Optional[str] = None
    description: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None

class PerformanceMetrics(BaseModel):
    """Model for engine performance metrics"""
    total_requests: int
    error_count: int
    error_rate: float
    throughput: float
    avg_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    active_requests: int
    memory_mb: float
    cpu_percent: float
    cache_hit_rate: Optional[float] = None
    avg_batch_size: Optional[float] = None
    engine_state: str

# Dependency to get inference engine instance
def get_inference_engine():
    """Dependency to get or create inference engine instance"""
    # In a real implementation, we'd use a singleton pattern or dependency injection
    config = InferenceEngineConfig()
    return InferenceEngine(config)

@router.post("/predict", response_model_exclude_none=True)
async def predict(
    data_file: UploadFile = File(...),
    params: Optional[str] = Form(None),
    model_id: Optional[str] = Query(None, description="Model ID to use for inference"),
    return_probabilities: Optional[bool] = Query(False, description="Return probability scores"),
    include_explanations: Optional[bool] = Query(False, description="Include explanations"),
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Make predictions using a trained model.
    
    This endpoint accepts tabular data in CSV format and returns predictions
    from the specified model. The model must be pre-loaded or specified in the request.
    """
    # Parse request parameters
    if params:
        req_params = InferenceRequest.parse_raw(params)
    else:
        req_params = InferenceRequest(
            model_id=model_id,
            return_probabilities=return_probabilities,
            include_explanations=include_explanations
        )
    
    # Create request ID if not provided
    request_id = req_params.request_id or str(uuid.uuid4())
    
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await data_file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Extract features (assuming target isn't in input)
        features = df.values
        
        # Load model if necessary
        if req_params.model_id:
            model_path = os.path.join(MODEL_REGISTRY, req_params.model_id)
            if req_params.model_version:
                model_path = os.path.join(model_path, req_params.model_version)
                
            model_loaded = inference_engine.load_model(model_path)
            if not model_loaded:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model {req_params.model_id} (version {req_params.model_version}) not found"
                )
        
        # Ensure engine is in valid state
        if inference_engine.get_state() not in (EngineState.READY, EngineState.RUNNING):
            raise HTTPException(
                status_code=503,
                detail=f"Inference engine not ready, current state: {inference_engine.get_state().name}"
            )
        
        # Make prediction
        start_time = time.time()
        success, predictions, metadata = inference_engine.predict(features, request_id=request_id)
        
        if not success:
            raise HTTPException(
                status_code=500, 
                detail=metadata.get("error", "Prediction failed")
            )
        
        # Format predictions for response
        if isinstance(predictions, np.ndarray):
            if predictions.ndim > 1:
                # For probability predictions (2D array)
                predictions_list = [list(map(float, row)) for row in predictions]
            else:
                # For regular predictions (1D array)
                predictions_list = list(map(float, predictions))
        else:
            predictions_list = predictions
        
        # Prepare response
        response = {
            "request_id": request_id,
            "model_id": req_params.model_id or metadata.get("model_id", "unknown"),
            "model_version": req_params.model_version or metadata.get("model_version", "unknown"),
            "predictions": predictions_list,
            "sample_count": len(df),
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "metadata": metadata
        }
        
        # Generate explanations if requested
        if req_params.include_explanations:
            try:
                method = req_params.explanation_method or "shap"
                # In a real implementation, this would call an explanation method
                # For now, we'll just add a placeholder
                response["explanations"] = {
                    "method": method,
                    "note": "Explanations would be generated here"
                }
            except Exception as e:
                response["explanations"] = {"error": str(e)}
        
        return response
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@router.post("/batch-inference")
async def batch_inference(
    files: List[UploadFile] = File(...),
    params: Optional[str] = Form(None),
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Process multiple files for batch inference.
    
    This endpoint accepts multiple CSV files and processes them in batch mode,
    either sequentially or in parallel depending on the settings.
    """
    # Parse request parameters
    if params:
        req_params = BatchInferenceRequest.parse_raw(params)
    else:
        req_params = BatchInferenceRequest()
    
    # Save uploaded files temporarily
    temp_files = []
    try:
        # Process each uploaded file
        batch_dataframes = []
        file_names = []
        
        for data_file in files:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            contents = await data_file.read()
            temp_file.write(contents)
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Read as dataframe
            df = pd.read_csv(temp_file.name)
            batch_dataframes.append(df)
            file_names.append(data_file.filename)
        
        # Load model if necessary
        if req_params.model_id:
            model_path = os.path.join(MODEL_REGISTRY, req_params.model_id)
            if req_params.model_version:
                model_path = os.path.join(model_path, req_params.model_version)
                
            model_loaded = inference_engine.load_model(model_path)
            if not model_loaded:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model {req_params.model_id} (version {req_params.model_version}) not found"
                )
        
        # Process batch using predict_batch
        start_time = time.time()
        
        # Prepare features for each batch
        batch_features = [df.values for df in batch_dataframes]
        
        # Process each batch and collect results
        results = []
        for i, features in enumerate(batch_features):
            # Use predict_batch to get Future
            future = inference_engine.predict_batch(features, priority=i)
            
            try:
                # Get result with timeout if specified
                timeout = req_params.timeout if req_params.timeout else None
                predictions, batch_metadata = future.result(timeout=timeout)
                
                # Format predictions
                if isinstance(predictions, np.ndarray):
                    if predictions.ndim > 1:
                        result_list = [list(map(float, row)) for row in predictions]
                    else:
                        result_list = list(map(float, predictions))
                else:
                    result_list = predictions
                
                results.append({
                    "success": True,
                    "predictions": result_list,
                    "metadata": batch_metadata
                })
                
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e)
                })
        
        # Format results for response
        formatted_results = []
        for i, (batch_df, result) in enumerate(zip(batch_dataframes, results)):
            formatted_results.append({
                "batch_index": i,
                "file_name": file_names[i] if i < len(file_names) else f"batch_{i}",
                "sample_count": len(batch_df),
                "success": result.get("success", False),
                "predictions": result.get("predictions") if result.get("success", False) else None,
                "error": result.get("error") if not result.get("success", False) else None
            })
        
        # Gather overall metrics
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r.get("success", False))
        
        response = {
            "model_id": req_params.model_id,
            "model_version": req_params.model_version,
            "batch_count": len(batch_dataframes),
            "total_samples": sum(len(df) for df in batch_dataframes),
            "successful_batches": success_count,
            "failed_batches": len(batch_dataframes) - success_count,
            "execution_time_ms": int(total_time * 1000),
            "results": formatted_results
        }
        
        # Save results to file if requested
        if req_params.save_results:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            result_file = os.path.join(RESULTS_DIR, f"batch_results_{timestamp}.json")
            with open(result_file, 'w') as f:
                json.dump(response, f, indent=2)
            response["results_file"] = result_file
        
        return response
        
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

@router.post("/streaming-inference")
async def start_streaming_inference(
    background_tasks: BackgroundTasks,
    params: StreamingInferenceRequest,
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Start a streaming inference job that processes data in the background.
    
    This endpoint initiates a long-running job to process large datasets that
    are streamed into the system. It returns a job ID that can be used to check
    the status of the job.
    """
    # Validate model exists
    model_path = os.path.join(MODEL_REGISTRY, params.model_id)
    if params.model_version:
        model_path = os.path.join(model_path, params.model_version)
        
    model_loaded = inference_engine.load_model(model_path)
    if not model_loaded:
        raise HTTPException(
            status_code=404, 
            detail=f"Model {params.model_id} (version {params.model_version}) not found"
        )
    
    # Create a job ID for tracking
    job_id = f"streaming_{params.model_id}_{params.batch_identifier}_{int(time.time())}"
    
    # Set up output path if not provided
    output_path = params.output_path
    if not output_path:
        output_format = params.output_format.lower()
        filename = f"{job_id}_results.{output_format}"
        output_path = os.path.join(RESULTS_DIR, filename)
    
    # Initialize job status
    job_status = {
        "job_id": job_id,
        "model_id": params.model_id,
        "model_version": params.model_version,
        "batch_identifier": params.batch_identifier,
        "status": StreamingJobStatus.PENDING,
        "start_time": time.time(),
        "output_path": output_path,
        "output_format": params.output_format,
        "progress": 0,
        "processed_items": 0,
        "total_items": 0,
        "errors": []
    }
    
    # Store job status for retrieval
    status_path = os.path.join(RESULTS_DIR, f"{job_id}_status.json")
    with open(status_path, 'w') as f:
        json.dump(job_status, f)
    
    # Define the background task function
    async def process_streaming(job_id, batch_id, output_path, status_path):
        """Background task to process streaming inference"""
        import asyncio  # Import here to avoid any issues with sync execution
        try:
            # Update status to processing
            with open(status_path, 'r') as f:
                status = json.load(f)
            
            status["status"] = StreamingJobStatus.PROCESSING
            
            with open(status_path, 'w') as f:
                json.dump(status, f)
            
            # Simulate processing for demo purposes
            # In a real implementation, this would fetch data from a source
            # and process it in batches
            
            # Simulate total items to process
            total_items = 10000
            
            # Update total items in status
            with open(status_path, 'r') as f:
                status = json.load(f)
            
            status["total_items"] = total_items
            
            with open(status_path, 'w') as f:
                json.dump(status, f)
            
            # Process in batches
            batch_size = 1000
            num_batches = (total_items + batch_size - 1) // batch_size  # Ceiling division
            
            for i in range(num_batches):
                # Simulate batch processing delay
                await asyncio.sleep(2)
                
                # Update progress
                processed_items = min((i + 1) * batch_size, total_items)
                progress = round(processed_items / total_items * 100, 1)
                
                with open(status_path, 'r') as f:
                    status = json.load(f)
                
                status["processed_items"] = processed_items
                status["progress"] = progress
                
                with open(status_path, 'w') as f:
                    json.dump(status, f)
            
            # Update status to completed
            with open(status_path, 'r') as f:
                status = json.load(f)
            
            status["status"] = StreamingJobStatus.COMPLETED
            status["end_time"] = time.time()
            status["elapsed_time_seconds"] = status["end_time"] - status["start_time"]
            
            with open(status_path, 'w') as f:
                json.dump(status, f)
            
            # Send notification if requested
            if params.notify_on_completion and params.notification_url:
                # In a real implementation, this would send a POST request to the webhook URL
                pass
                
        except Exception as e:
            # Update status to failed on error
            try:
                with open(status_path, 'r') as f:
                    status = json.load(f)
                
                status["status"] = StreamingJobStatus.FAILED
                status["error"] = str(e)
                status["end_time"] = time.time()
                status["elapsed_time_seconds"] = status["end_time"] - status["start_time"]
                
                with open(status_path, 'w') as f:
                    json.dump(status, f)
            except:
                logger.exception("Failed to update job status on error")
    
    # Start background task for streaming inference
    background_tasks.add_task(
        process_streaming,
        job_id=job_id,
        batch_id=params.batch_identifier,
        output_path=output_path,
        status_path=status_path
    )
    
    return {
        "job_id": job_id,
        "status": StreamingJobStatus.PENDING,
        "message": f"Streaming inference job started for batch {params.batch_identifier}",
        "status_endpoint": f"/api/v1/inference/streaming-status/{job_id}"
    }

@router.get("/streaming-status/{job_id}")
async def get_streaming_status(
    job_id: str = Path(..., description="ID of the streaming job to check")
):
    """
    Get the status of a streaming inference job.
    
    Returns the current status of a previously started streaming job,
    including progress information and any errors that occurred.
    """
    # Look for status file in the results directory
    status_path = os.path.join(RESULTS_DIR, f"{job_id}_status.json")
    
    if not os.path.exists(status_path):
        raise HTTPException(status_code=404, detail=f"Streaming job {job_id} not found")
    
    try:
        with open(status_path, 'r') as f:
            status = json.load(f)
        
        # Check if output file exists and add its size if job is completed
        if status["status"] == StreamingJobStatus.COMPLETED and os.path.exists(status["output_path"]):
            status["output_file_size_bytes"] = os.path.getsize(status["output_path"])
        
        # Update elapsed time
        if "end_time" in status:
            status["elapsed_time_seconds"] = status["end_time"] - status["start_time"]
        else:
            status["elapsed_time_seconds"] = time.time() - status["start_time"]
        
        return status
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")

@router.post("/explain")
async def explain_prediction(
    data_file: UploadFile = File(...),
    params: Optional[str] = Form(None),
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Generate explanations for model predictions.
    
    This endpoint generates explanations (such as SHAP values) for predictions
    to help understand what features are driving the model's decisions.
    """
    # Parse request parameters
    if params:
        req_params = ExplanationRequest.parse_raw(params)
    else:
        req_params = ExplanationRequest()
    
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await data_file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Load model if necessary
        if req_params.model_id:
            model_path = os.path.join(MODEL_REGISTRY, req_params.model_id)
            model_loaded = inference_engine.load_model(model_path)
            if not model_loaded:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model {req_params.model_id} not found"
                )
        
        # In a real implementation, this would call an explanation method
        # For this API design, we'll just return a placeholder
        
        # Check if the current model supports explanations
        model_info = inference_engine.get_model_info()
        if not model_info.get("model_info", {}).get("has_feature_importances", False):
            return {
                "model_id": req_params.model_id,
                "method": req_params.method,
                "message": "Current model does not support feature importance explanations",
                "explanations": None
            }
        
        # Get feature names if available
        feature_names = inference_engine.get_feature_names()
        
        # Simulate generating explanations
        # In a real implementation, this would call model-specific explainers
        sample_count = min(len(df), req_params.n_samples)
        features = df.iloc[:sample_count].values
        
        # Make predictions first
        success, predictions, _ = inference_engine.predict(features)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate predictions for explanation")
        
        # Generate mock explanations based on method
        if req_params.method.lower() == "shap":
            explanations = {
                "type": "shap_values",
                "feature_names": feature_names,
                "sample_count": sample_count,
                "global_importance": {
                    "values": [0.1 * i for i in range(len(feature_names))],
                    "features": feature_names
                },
                "samples": [
                    {
                        "id": i,
                        "prediction": float(predictions[i]) if predictions.ndim == 1 else list(map(float, predictions[i])),
                        "values": [0.05 * j * (i % 5) for j in range(len(feature_names))]
                    }
                    for i in range(sample_count)
                ]
            }
        else:
            explanations = {
                "type": req_params.method,
                "message": f"Explanations using {req_params.method} would be generated here",
                "sample_count": sample_count
            }
        
        return {
            "model_id": req_params.model_id,
            "method": req_params.method,
            "sample_count": sample_count,
            "explanations": explanations
        }
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@router.get("/models")
async def list_available_models(
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    List all models available for inference.
    
    Returns a list of models available in the model registry along with
    their metadata.
    """
    # In a real implementation, this would scan the model registry
    # For this API design, we'll use a simulated approach
    
    # Get current model info if available
    current_model_info = None
    if inference_engine.model is not None:
        current_model_info = inference_engine.get_model_info()
    
    # Scan model registry directory
    models = []
    if os.path.exists(MODEL_REGISTRY):
        for model_dir in os.listdir(MODEL_REGISTRY):
            model_path = os.path.join(MODEL_REGISTRY, model_dir)
            if os.path.isdir(model_path):
                # Check for model info file
                info_path = os.path.join(model_path, "model_info.json")
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f:
                            model_info = json.load(f)
                        
                        # Get available versions
                        versions = []
                        for item in os.listdir(model_path):
                            version_path = os.path.join(model_path, item)
                            if os.path.isdir(version_path) and os.path.exists(os.path.join(version_path, "model_info.json")):
                                versions.append(item)
                        
                        models.append({
                            "model_id": model_dir,
                            "model_type": model_info.get("model_type", "unknown"),
                            "description": model_info.get("description", ""),
                            "feature_count": model_info.get("feature_count"),
                            "versions": versions,
                            "latest_version": versions[-1] if versions else None
                        })
                    except Exception as e:
                        # If we can't read info file, add basic entry
                        models.append({
                            "model_id": model_dir,
                            "model_type": "unknown",
                            "description": "Model information unavailable",
                            "error": str(e)
                        })

@router.post("/load-model/{model_id}")
async def load_model(
    model_id: str = Path(..., description="ID of the model to load"),
    model_version: Optional[str] = Query(None, description="Version of the model to load"),
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Load a specific model for inference.
    
    This endpoint loads a model from the model registry and makes it
    the active model for subsequent inference requests.
    """
    model_path = os.path.join(MODEL_REGISTRY, model_id)
    if model_version:
        model_path = os.path.join(model_path, model_version)
    
    # Check if model exists before trying to load
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Model {model_id}" + (f" version {model_version}" if model_version else "") + " not found"
        )
    
    # Try to load model
    success = inference_engine.load_model(model_path)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model {model_id}" + (f" version {model_version}" if model_version else "")
        )
    
    # Get model metadata
    model_info = inference_engine.get_model_info()
    
    return {
        "message": f"Model {model_id}" + (f" version {model_version}" if model_version else "") + " loaded successfully",
        "model_id": model_id,
        "model_version": model_version,
        "model_info": model_info
    }

@router.get("/model-metadata")
async def get_model_metadata(
    model_id: Optional[str] = Query(None, description="ID of the model to get metadata for"),
    model_version: Optional[str] = Query(None, description="Version of the model to get metadata for"),
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Get metadata for the current model or a specified model.
    
    This endpoint returns detailed information about a model, including
    its structure, feature names, and other relevant metadata.
    """
    # If model_id is provided, try to load it temporarily
    if model_id:
        # Remember the current model
        current_model = None
        if inference_engine.model is not None:
            # Get some identifier for the current model to restore it later
            current_model_info = inference_engine.get_model_info()
            if current_model_info and "model_info" in current_model_info:
                current_model = current_model_info
        
        # Load the specified model
        model_path = os.path.join(MODEL_REGISTRY, model_id)
        if model_version:
            model_path = os.path.join(model_path, model_version)
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id}" + (f" version {model_version}" if model_version else "") + " not found"
            )
        
        success = inference_engine.load_model(model_path)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model {model_id}" + (f" version {model_version}" if model_version else "")
            )
        
        # Get metadata for the specified model
        metadata = inference_engine.get_model_info()
        
        # Restore previous model if there was one
        if current_model:
            # In a real implementation, we'd have a way to restore the previous model
            pass
        
        return {
            "model_id": model_id,
            "model_version": model_version,
            "metadata": metadata
        }
    else:
        # Get metadata for currently loaded model
        if inference_engine.model is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No model currently loaded"
            )
        
        metadata = inference_engine.get_model_info()
        return {
            "metadata": metadata
        }

@router.post("/model-config")
async def update_model_config(
    config: ModelConfigRequest,
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Update configuration settings for the inference engine.
    
    This endpoint allows updating various configuration parameters for
    the inference engine, such as batch size, quantization settings, etc.
    """
    # Check if engine is in a valid state for configuration updates
    engine_state = inference_engine.get_state()
    if engine_state not in (EngineState.READY, EngineState.ERROR):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update configuration while engine is in {engine_state.name} state"
        )
    
    # Update engine configuration if provided
    if config.engine_config:
        # Create a new config object with the updated values
        current_config = inference_engine.config
        
        # In a real implementation, this would update the engine config
        # For now, we'll just log that it would be updated
        logger.info(f"Would update engine config with: {config.engine_config}")
    
    # Update quantization config if provided
    if config.quantization_config:
        # In a real implementation, this would update the quantizer
        logger.info(f"Would update quantization config with: {config.quantization_config}")
    
    # Update batch processor config if provided
    if config.batch_config:
        # In a real implementation, this would update the batch processor
        logger.info(f"Would update batch processor config with: {config.batch_config}")
    
    # Update feature names if provided
    if config.feature_names:
        inference_engine.set_feature_names(config.feature_names)
    
    return {
        "message": "Model configuration updated successfully",
        "engine_state": engine_state.name,
        "updates_applied": {
            "engine_config": bool(config.engine_config),
            "quantization_config": bool(config.quantization_config),
            "batch_config": bool(config.batch_config),
            "feature_names": bool(config.feature_names)
        }
    }

@router.get("/metrics")
async def get_engine_metrics(
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Get performance metrics for the inference engine.
    
    Returns detailed metrics on engine performance, including throughput,
    error rates, and resource usage.
    """
    # Get metrics from engine
    metrics = inference_engine.get_metrics()
    
    # Convert metrics to response format
    response = {
        "engine_state": metrics.get("engine_state", "unknown"),
        "total_requests": metrics.get("total_requests", 0),
        "error_count": metrics.get("error_count", 0),
        "error_rate": metrics.get("error_rate", 0.0),
        "avg_inference_time_ms": metrics.get("avg_inference_time_ms", 0.0),
        "p95_inference_time_ms": metrics.get("p95_inference_time_ms", 0.0),
        "p99_inference_time_ms": metrics.get("p99_inference_time_ms", 0.0),
        "throughput_requests_per_second": metrics.get("throughput_requests_per_second", 0.0),
        "active_requests": metrics.get("active_requests", 0),
        "memory_mb": metrics.get("memory_mb", 0.0),
        "cpu_percent": metrics.get("cpu_percent", 0.0)
    }
    
    # Add cache metrics if available
    if "cache_hit_rate" in metrics:
        response["cache_hit_rate"] = metrics["cache_hit_rate"]
        response["cache_hits"] = metrics.get("cache_hits", 0)
        response["cache_misses"] = metrics.get("cache_misses", 0)
    
    # Add batch metrics if available
    if "avg_batch_size" in metrics:
        response["avg_batch_size"] = metrics["avg_batch_size"]
        response["max_batch_size"] = metrics.get("max_batch_size", 0)
    
    # Add quantization metrics if available
    if "avg_quantize_time_ms" in metrics:
        response["avg_quantize_time_ms"] = metrics["avg_quantize_time_ms"]
        response["avg_dequantize_time_ms"] = metrics.get("avg_dequantize_time_ms", 0.0)
    
    return response

@router.post("/clear-cache")
async def clear_engine_cache(
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Clear the inference engine's cache.
    
    This endpoint clears any cached results and quantization data in the
    inference engine, which can be useful for freeing up memory or
    ensuring fresh predictions.
    """
    inference_engine.clear_cache()
    
    return {
        "message": "Cache cleared successfully",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/status")
async def get_engine_status(
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Get the current status of the inference engine.
    
    Returns information about the engine's state, loaded model, and
    resource usage.
    """
    # Get engine state and basic metrics
    state = inference_engine.get_state()
    memory_usage = inference_engine.get_memory_usage()
    
    # Get model information if a model is loaded
    model_info = {}
    if inference_engine.model is not None:
        model_info = inference_engine.get_model_info()
    
    # Prepare response
    response = {
        "status": state.name,
        "ready_for_inference": state in (EngineState.READY, EngineState.RUNNING),
        "memory_usage_mb": memory_usage.get("rss_mb", 0.0),
        "cpu_percent": memory_usage.get("cpu_percent", 0.0),
        "model_loaded": inference_engine.model is not None,
        "feature_count": len(inference_engine.get_feature_names()) if inference_engine.get_feature_names() else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add model info if available
    if model_info and "model_info" in model_info:
        response["model_type"] = model_info["model_info"].get("model_type", "unknown")
        response["model_class"] = model_info["model_info"].get("model_class", "unknown")
    
    return response

@router.post("/shutdown")
async def shutdown_engine(
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Gracefully shut down the inference engine.
    
    This endpoint initiates a graceful shutdown of the inference engine,
    allowing any in-progress requests to complete before stopping.
    """
    # Check if there are active requests
    metrics = inference_engine.get_metrics()
    active_requests = metrics.get("active_requests", 0)
    
    if active_requests > 0:
        # Return warning that there are active requests
        return {
            "message": f"Initiating shutdown with {active_requests} active requests",
            "status": "SHUTTING_DOWN",
            "active_requests": active_requests
        }
    
    # Initiate shutdown
    inference_engine.shutdown()
    
    return {
        "message": "Inference engine shutdown completed",
        "status": "STOPPED",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/download-results/{job_id}")
async def download_job_results(
    job_id: str = Path(..., description="ID of the job to download results for")
):
    """
    Download results of a completed streaming inference job.
    
    This endpoint returns the results file from a previously run
    streaming inference job.
    """
    # Find the job status file
    status_path = os.path.join(RESULTS_DIR, f"{job_id}_status.json")
    
    if not os.path.exists(status_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Load the job status
    with open(status_path, 'r') as f:
        job_status = json.load(f)
    
    # Check if job is completed
    if job_status.get("status") != StreamingJobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is not completed (current status: {job_status.get('status')})"
        )
    
    # Get the output path from the job status
    output_path = job_status.get("output_path")
    
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Results file for job {job_id} not found"
        )
    
    # Determine file type from extension
    file_extension = os.path.splitext(output_path)[1].lower()
    media_type = "text/csv"  # Default
    
    if file_extension == ".json":
        media_type = "application/json"
    elif file_extension == ".parquet":
        media_type = "application/octet-stream"
    
    # Return the file as a download
    return FileResponse(
        path=output_path,
        media_type=media_type,
        filename=os.path.basename(output_path)
    )