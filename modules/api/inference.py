from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tempfile
import os
import time
import logging
import json
from enum import Enum

# Import inference module
from modules.engine.inference_engine import InferenceEngine
from modules.configs import InferenceEngineConfig

router = APIRouter(prefix="/inference", tags=["Model Inference"])

# Data models
class InferenceRequest(BaseModel):
    model_name: Optional[str] = None
    batch_size: Optional[int] = None
    return_probabilities: bool = False
    include_explanations: bool = False
    explanation_method: Optional[str] = "shap"

class BatchInferenceRequest(BaseModel):
    model_name: Optional[str] = None
    batch_size: Optional[int] = None
    parallel: bool = True
    timeout: Optional[int] = None
    return_probabilities: bool = False

class StreamingInferenceStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class StreamingInferenceRequest(BaseModel):
    model_name: str
    batch_identifier: str
    streaming_mode: bool = True
    output_path: Optional[str] = None

class ExplanationRequest(BaseModel):
    model_name: Optional[str] = None
    method: str = "shap"
    n_samples: int = 100

# Dependency to get inference engine instance
def get_inference_engine():
    config = InferenceEngineConfig()
    return InferenceEngine(config)

@router.post("/predict")
async def predict(
    data_file: UploadFile = File(...),
    params: InferenceRequest = Depends(),
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """Make predictions using a trained model"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await data_file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Load model if necessary
        if params.model_name:
            model_loaded = inference_engine.load_model(params.model_name)
            if not model_loaded:
                raise HTTPException(status_code=404, detail=f"Model {params.model_name} not found or could not be loaded")
        
        # Make prediction
        start_time = time.time()
        success, predictions, metadata = inference_engine.predict(
            df, 
            batch_size=params.batch_size,
            return_proba=params.return_probabilities
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=metadata.get("error", "Prediction failed"))
        
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
        
        # Generate explanations if requested
        explanations = None
        if params.include_explanations:
            try:
                method = params.explanation_method or "shap"
                explanations = inference_engine.explain_predictions(
                    df, 
                    method=method, 
                    n_samples=min(len(df), 10)  # Limit to 10 samples for API response
                )
            except Exception as e:
                explanations = {"error": str(e)}
        
        return {
            "model": params.model_name or inference_engine.current_model_name,
            "predictions": predictions_list,
            "sample_count": len(df),
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "metadata": metadata,
            "explanations": explanations
        }
    finally:
        os.unlink(temp_file.name)

@router.post("/batch-inference")
async def batch_inference(
    files: List[UploadFile] = File(...),
    params: BatchInferenceRequest = Depends(),
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """Process multiple files for batch inference"""
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
        if params.model_name:
            model_loaded = inference_engine.load_model(params.model_name)
            if not model_loaded:
                raise HTTPException(status_code=404, detail=f"Model {params.model_name} not found or could not be loaded")
        
        # Run batch inference
        start_time = time.time()
        results = inference_engine.batch_predict(
            batch_dataframes,
            batch_size=params.batch_size,
            parallel=params.parallel,
            timeout=params.timeout,
            return_proba=params.return_probabilities
        )
        
        # Format results for response
        formatted_results = []
        for i, (batch_df, result) in enumerate(zip(batch_dataframes, results)):
            # Handle result which might be a numpy array
            if isinstance(result, np.ndarray):
                if result.ndim > 1:
                    # For probability predictions (2D array)
                    result_list = [list(map(float, row)) for row in result]
                else:
                    # For regular predictions (1D array)
                    result_list = list(map(float, result))
            elif isinstance(result, dict) and "error" in result:
                # Handle error case
                result_list = {"error": result["error"]}
            else:
                result_list = result
                
            formatted_results.append({
                "batch_index": i,
                "file_name": file_names[i] if i < len(file_names) else f"batch_{i}",
                "batch_size": len(batch_df),
                "predictions": result_list
            })
        
        return {
            "model": params.model_name or inference_engine.current_model_name,
            "batch_count": len(batch_dataframes),
            "total_samples": sum(len(df) for df in batch_dataframes),
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "results": formatted_results
        }
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
    """Start a streaming inference job that processes data in the background"""
    # Validate model exists
    model_loaded = inference_engine.load_model(params.model_name)
    if not model_loaded:
        raise HTTPException(status_code=404, detail=f"Model {params.model_name} not found or could not be loaded")
    
    # Create a job ID for tracking
    job_id = f"{params.model_name}_{params.batch_identifier}_{int(time.time())}"
    
    # Set up output path if not provided
    output_path = params.output_path
    if not output_path:
        output_dir = os.path.join(os.getcwd(), "streaming_results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{job_id}_results.csv")
    
    # Initialize job status
    job_status = {
        "job_id": job_id,
        "model_name": params.model_name,
        "batch_identifier": params.batch_identifier,
        "status": StreamingInferenceStatus.PENDING,
        "start_time": time.time(),
        "output_path": output_path,
        "progress": 0,
        "processed_items": 0,
        "total_items": 0,
        "errors": []
    }
    
    # Store job status for retrieval
    status_path = f"{output_path}_status.json"
    with open(status_path, 'w') as f:
        json.dump(job_status, f)
    
    # Start background task for streaming inference
    background_tasks.add_task(
        inference_engine.process_streaming_inference,
        job_id=job_id,
        batch_identifier=params.batch_identifier,
        output_path=output_path,
        status_path=status_path
    )
    
    return {
        "job_id": job_id,
        "status": StreamingInferenceStatus.PENDING,
        "message": f"Streaming inference job started for batch {params.batch_identifier}",
        "status_endpoint": f"/inference/streaming-status/{job_id}"
    }

@router.get("/streaming-status/{job_id}")
async def get_streaming_status(job_id: str):
    """Get the status of a streaming inference job"""
    # Try to find status file for this job
    status_pattern = f"*{job_id}*_status.json"
    
    # Look in the streaming_results directory first
    streaming_dir = os.path.join(os.getcwd(), "streaming_results")
    
    # If we can't find the status file, search more broadly
    status_files = []
    for root, _, files in os.walk(streaming_dir):
        for file in files:
            if job_id in file and file.endswith("_status.json"):
                status_files.append(os.path.join(root, file))
    
    if not status_files:
        raise HTTPException(status_code=404, detail=f"Streaming job {job_id} not found")
    
    # Use the first matching status file
    status_path = status_files[0]
    
    try:
        with open(status_path, 'r') as f:
            status = json.load(f)
        
        # Calculate elapsed time
        status["elapsed_time_seconds"] = time.time() - status["start_time"]
        
        # If job is completed, add output file information
        if status["status"] == StreamingInferenceStatus.COMPLETED and os.path.exists(status["output_path"]):
            status["output_file_size_bytes"] = os.path.getsize(status["output_path"])
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")

@router.post("/explain")
async def explain_prediction(
    data_file: UploadFile = File(...),
    params: ExplanationRequest = Depends(),
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """Generate explanations for model predictions"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await data_file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Load model if necessary
        if params.model_name:
            model_loaded = inference_engine.load_model(params.model_name)
            if not model_loaded:
                raise HTTPException(status_code=404, detail=f"Model {params.model_name} not found or could not be loaded")
        
        # Generate explanations
        explanations = inference_engine.explain_predictions(
            df, 
            method=params.method, 
            n_samples=min(len(df), params.n_samples)
        )
        
        if isinstance(explanations, dict) and "error" in explanations:
            raise HTTPException(status_code=500, detail=explanations["error"])
        
        return {
            "model": params.model_name or inference_engine.current_model_name,
            "method": params.method,
            "sample_count": min(len(df), params.n_samples),
            "explanations": explanations
        }
    finally:
        os.unlink(temp_file.name)

@router.get("/models")
async def list_available_models(
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """List all models available for inference"""
    models = inference_engine.list_models()
    return {
        "models": models,
        "count": len(models),
        "current_model": inference_engine.current_model_name
    }

@router.post("/load-model/{model_name}")
async def load_model(
    model_name: str,
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """Load a specific model for inference"""
    success = inference_engine.load_model(model_name)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found or could not be loaded")
    
    return {
        "message": f"Model {model_name} loaded successfully",
        "model_name": model_name,
        "model_metadata": inference_engine.get_model_metadata()
    }

@router.get("/model-metadata")
async def get_model_metadata(
    model_name: Optional[str] = None,
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """Get metadata for the current model or a specified model"""
    if model_name:
        # Temporarily load the specified model to get its metadata
        original_model = inference_engine.current_model_name
        success = inference_engine.load_model(model_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found or could not be loaded")
            
        metadata = inference_engine.get_model_metadata()
            
        # Load back the original model if different
        if original_model and original_model != model_name:
            inference_engine.load_model(original_model)
    else:
        # Get metadata for currently loaded model
        metadata = inference_engine.get_model_metadata()
        model_name = inference_engine.current_model_name
    
    if not metadata:
        raise HTTPException(status_code=400, detail="No model currently loaded")
    
    return {
        "model_name": model_name,
        "metadata": metadata
    }