"""
Inference Engine API

A RESTful API for the high-performance InferenceEngine.
Provides endpoints for model loading, inference, batch processing,
and monitoring performance metrics.

Author: Evint Leovonzko
Date: 2025-04-28
"""

import os
import sys
import time
import uuid
import json
import logging
import numpy as np
import asyncio  # Added missing import
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Utility function for parsing boolean environment variables
def parse_bool_env(env_var: str, default: str = "False") -> bool:
    """Parse boolean environment variable safely."""
    value = os.environ.get(env_var, default).lower()
    return value in ("true", "1", "t", "yes", "on")

# Utility function to safely get the engine
def get_engine():
    """Get the inference engine, raising appropriate HTTP exception if not available."""
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized. Please try again later."
        )
    return app.state.engine

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Header, Query, Body, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator, root_validator
import uvicorn

# Import the InferenceEngine and related modules
from modules.engine.inference_engine import (
    InferenceEngine, 
    InferenceEngineConfig,
    BatchPriority, 
    ModelType,
    EngineState
)
from modules.configs import (
    QuantizationConfig, 
    BatchProcessorConfig, 
    BatchProcessingStrategy,
    PreprocessorConfig, 
    NormalizationType, 
    QuantizationMode
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("inference_api.log")
    ]
)
logger = logging.getLogger("inference_api")

# --- Pydantic Models for Request/Response ---

class ModelLoadRequest(BaseModel):
    """Request model for loading a model"""
    model_path: str = Field(..., description="Path to the model file")
    model_type: Optional[str] = Field(None, description="Type of the model (sklearn, xgboost, lightgbm, ensemble, custom)")
    compile_model: Optional[bool] = Field(None, description="Whether to compile the model for faster inference")

    @validator("model_type", pre=True)
    def validate_model_type(cls, v):
        if v is not None:
            allowed_types = ["sklearn", "xgboost", "lightgbm", "ensemble", "custom"]
            if v.lower() not in allowed_types:
                raise ValueError(f"model_type must be one of {allowed_types}")
            return v.lower()
        return v

class InferenceRequest(BaseModel):
    """Request model for making predictions"""
    features: List[List[float]] = Field(..., description="Input features as a 2D array")
    request_id: Optional[str] = Field(None, description="Optional client-provided request ID")

    @validator("features")
    def validate_features(cls, v):
        if not v:
            raise ValueError("features cannot be empty")
        return v

class BatchInferenceRequest(BaseModel):
    """Request model for batch predictions"""
    batch: List[List[List[float]]] = Field(..., description="List of feature arrays for batch processing")
    request_ids: Optional[List[str]] = Field(None, description="Optional client-provided request IDs")

    @validator("batch")
    def validate_batch(cls, v):
        if not v:
            raise ValueError("batch cannot be empty")
        return v

    @validator("request_ids")
    def validate_request_ids(cls, v, values):
        if v is not None and len(v) != len(values["batch"]):
            raise ValueError("If provided, request_ids must have the same length as batch")
        return v

class AsyncInferenceRequest(InferenceRequest):
    """Request model for asynchronous predictions"""
    priority: Optional[str] = Field("normal", description="Processing priority (high, normal, low)")
    timeout_ms: Optional[float] = Field(None, description="Optional timeout in milliseconds")

    @validator("priority")
    def validate_priority(cls, v):
        allowed_priorities = ["high", "normal", "low"]
        if v.lower() not in allowed_priorities:
            raise ValueError(f"priority must be one of {allowed_priorities}")
        return v.lower()

class EngineConfigRequest(BaseModel):
    """Request model for updating engine configuration"""
    enable_batching: Optional[bool] = None
    max_batch_size: Optional[int] = None
    batch_timeout: Optional[float] = None
    enable_cache: Optional[bool] = None
    max_cache_entries: Optional[int] = None
    cache_ttl_seconds: Optional[int] = None
    enable_quantization: Optional[bool] = None
    num_threads: Optional[int] = None
    enable_throttling: Optional[bool] = None
    max_concurrent_requests: Optional[int] = None

    @validator("max_batch_size")
    def validate_max_batch_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_batch_size must be positive")
        return v

    @validator("num_threads")
    def validate_num_threads(cls, v):
        if v is not None and v <= 0:
            raise ValueError("num_threads must be positive")
        return v

class InferenceResponse(BaseModel):
    """Response model for inference results"""
    request_id: str = Field(..., description="Request ID (client-provided or server-generated)")
    success: bool = Field(..., description="Whether the inference was successful")
    predictions: Optional[List[List[float]]] = Field(None, description="Prediction results")
    error: Optional[str] = Field(None, description="Error message if inference failed")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata about the inference")
    timestamp: str = Field(..., description="Timestamp of the response")

class BatchInferenceResponse(BaseModel):
    """Response model for batch inference results"""
    batch_id: str = Field(..., description="Batch request ID")
    results: List[InferenceResponse] = Field(..., description="List of individual inference results")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata about the batch processing")
    timestamp: str = Field(..., description="Timestamp of the response")

class AsyncJobResponse(BaseModel):
    """Response model for asynchronous job submission"""
    job_id: str = Field(..., description="Unique job ID for the asynchronous request")
    status: str = Field("pending", description="Job status")
    eta_seconds: Optional[float] = Field(None, description="Estimated time until completion")
    timestamp: str = Field(..., description="Timestamp of the response")

class EngineHealthResponse(BaseModel):
    """Response model for engine health check"""
    status: str = Field(..., description="Engine status")
    state: str = Field(..., description="Current engine state")
    uptime_seconds: float = Field(..., description="Engine uptime in seconds")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Information about the loaded model")
    active_requests: int = Field(0, description="Number of currently active requests")
    timestamp: str = Field(..., description="Timestamp of the response")

class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics"""
    metrics: Dict[str, Any] = Field(..., description="Engine performance metrics")
    timestamp: str = Field(..., description="Timestamp of the response")

class CacheStatsResponse(BaseModel):
    """Response model for cache statistics"""
    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")
    timestamp: str = Field(..., description="Timestamp of the response")

class BatcherStatsResponse(BaseModel):
    """Response model for batcher statistics"""
    batcher_stats: Dict[str, Any] = Field(..., description="Dynamic batcher statistics")
    timestamp: str = Field(..., description="Timestamp of the response")

class MemoryPoolStatsResponse(BaseModel):
    """Response model for memory pool statistics"""
    memory_stats: Dict[str, Any] = Field(..., description="Memory pool statistics")
    timestamp: str = Field(..., description="Timestamp of the response")

class ModelCompileRequest(BaseModel):
    """Request model for compiling a model"""
    compile_format: Optional[str] = Field("auto", description="Format to compile to (onnx, treelite, auto)")
    optimization_level: Optional[int] = Field(2, description="Optimization level (0-3)")

class WarmupRequest(BaseModel):
    """Request model for model warmup"""
    warmup_samples: Optional[int] = Field(100, description="Number of samples for warmup")
    warmup_batches: Optional[int] = Field(3, description="Number of batches for warmup")

class QuantizationRequest(BaseModel):
    """Request model for model quantization"""
    quantization_type: Optional[str] = Field("int8", description="Type of quantization (int8, uint8, int16, float16)")
    quantization_mode: Optional[str] = Field("dynamic", description="Quantization mode (dynamic, static)")
    calibration_data: Optional[List[List[float]]] = Field(None, description="Optional calibration data for static quantization")

# --- API Configuration ---

# Create configuration
api_config = {
    "title": "Inference Engine API",
    "description": "A high-performance API for machine learning model inference",
    "version": "1.0.0",
    "default_model_dir": os.environ.get("MODEL_DIR", "./models"),
    "api_keys": os.environ.get("API_KEYS", "").split(","),
    "max_workers": int(os.environ.get("MAX_WORKERS", "4")),
    "enable_async": parse_bool_env("ENABLE_ASYNC", "1"),
    "host": os.environ.get("API_HOST", "0.0.0.0"),
    "port": int(os.environ.get("API_PORT", "8000")),
    "debug": parse_bool_env("API_DEBUG", "0"),
    "job_ttl_seconds": int(os.environ.get("JOB_TTL_SECONDS", "3600")),
    "require_api_key": parse_bool_env("REQUIRE_API_KEY", "0")
}

# Create global variables for engine and job tracking
api_security = APIKeyHeader(name="X-API-Key", auto_error=False)
engine_start_time = time.time()
async_jobs = {}  # job_id -> {"status": str, "future": Future, "result": dict, "created_at": float}
job_cleanup_interval = 60  # seconds

# --- Lifecycle Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events for the API
    """
    # Create and initialize the inference engine
    logger.info("Initializing InferenceEngine...")
    
    # Create engine config (using environment variables or defaults)
    engine_config = InferenceEngineConfig(
        enable_batching=parse_bool_env("ENGINE_ENABLE_BATCHING", "1"),
        max_batch_size=int(os.environ.get("ENGINE_MAX_BATCH_SIZE", "64")),
        batch_timeout=float(os.environ.get("ENGINE_BATCH_TIMEOUT", "0.01")),
        max_concurrent_requests=int(os.environ.get("ENGINE_MAX_CONCURRENT_REQUESTS", "100")),
        enable_request_deduplication=parse_bool_env("ENGINE_ENABLE_CACHE", "1"),
        max_cache_entries=int(os.environ.get("ENGINE_MAX_CACHE_ENTRIES", "1000")),
        cache_ttl_seconds=int(os.environ.get("ENGINE_CACHE_TTL_SECONDS", "3600")),
        enable_quantization=parse_bool_env("ENGINE_ENABLE_QUANTIZATION", "0"),
        num_threads=int(os.environ.get("ENGINE_NUM_THREADS", "4")),
        enable_throttling=parse_bool_env("ENGINE_ENABLE_THROTTLING", "0"),
        enable_monitoring=True,
        debug_mode=api_config["debug"]
    )
    
    # Create and store the engine
    app.state.engine = InferenceEngine(engine_config)
    app.state.thread_pool = ThreadPoolExecutor(max_workers=api_config["max_workers"])
    
    # Start job cleanup background task
    app.state.job_cleanup_task = asyncio.create_task(cleanup_expired_jobs())
    
    # Auto-load model if specified in environment variables
    default_model_path = os.environ.get("DEFAULT_MODEL_PATH")
    if default_model_path and os.path.exists(default_model_path):
        try:
            logger.info(f"Auto-loading model from {default_model_path}...")
            model_type_str = os.environ.get("DEFAULT_MODEL_TYPE")
            model_type = None
            if model_type_str:
                try:
                    model_type = ModelType[model_type_str.upper()]
                except KeyError:
                    logger.warning(f"Invalid model type: {model_type_str}")
                    
            app.state.engine.load_model(
                model_path=default_model_path,
                model_type=model_type,
                compile_model=parse_bool_env("DEFAULT_COMPILE_MODEL", "0")
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to auto-load model: {str(e)}")
    
    logger.info("API startup complete")
    yield
    
    # Shutdown the engine and thread pool
    logger.info("Shutting down API...")
    
    # Cancel job cleanup task
    if hasattr(app.state, "job_cleanup_task"):
        app.state.job_cleanup_task.cancel()
        try:
            await app.state.job_cleanup_task
        except asyncio.CancelledError:
            pass
    
    # Shutdown thread pool
    if hasattr(app.state, "thread_pool"):
        app.state.thread_pool.shutdown(wait=False)
    
    # Shutdown engine
    if hasattr(app.state, "engine"):
        app.state.engine.shutdown()
    
    logger.info("API shutdown complete")

# Create the FastAPI app
app = FastAPI(
    title=api_config["title"],
    description=api_config["description"],
    version=api_config["version"],
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# --- Security Functions ---

async def verify_api_key(api_key: str = Depends(api_security)):
    """Verify the API key if required"""
    if not api_config["require_api_key"]:
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )
    
    if api_key not in api_config["api_keys"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return True

# --- Helper Functions ---

def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())

def validate_request_features(features):
    """Validate and convert input features to numpy array"""
    try:
        return np.array(features, dtype=np.float32)
    except Exception as e:
        logger.error(f"Feature validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid feature format: {str(e)}"
        )

def get_priority_enum(priority_str: str) -> BatchPriority:
    """Convert priority string to BatchPriority enum"""
    if priority_str.lower() == "high":
        return BatchPriority.HIGH
    elif priority_str.lower() == "low":
        return BatchPriority.LOW
    else:
        return BatchPriority.NORMAL

def get_model_type_enum(model_type_str: str) -> Optional[ModelType]:
    """Convert model type string to ModelType enum"""
    if model_type_str is None:
        return None
        
    type_map = {
        "sklearn": ModelType.SKLEARN,
        "xgboost": ModelType.XGBOOST,
        "lightgbm": ModelType.LIGHTGBM,
        "ensemble": ModelType.ENSEMBLE,
        "custom": ModelType.CUSTOM
    }
    
    return type_map.get(model_type_str.lower())

async def cleanup_expired_jobs():
    """Background task to clean up expired async jobs"""
    while True:
        try:
            current_time = time.time()
            expired_jobs = []
            
            # Find expired jobs
            for job_id, job_info in async_jobs.items():
                if current_time - job_info["created_at"] > api_config["job_ttl_seconds"]:
                    expired_jobs.append(job_id)
            
            # Remove expired jobs
            for job_id in expired_jobs:
                logger.info(f"Cleaning up expired job {job_id}")
                async_jobs.pop(job_id, None)
                
            await asyncio.sleep(job_cleanup_interval)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in job cleanup: {str(e)}")
            await asyncio.sleep(job_cleanup_interval)

# --- API Endpoints ---

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "name": api_config["title"],
        "version": api_config["version"],
        "description": api_config["description"],
        "status": "running",
        "docs_url": "/docs"
    }

@app.get("/health", response_model=EngineHealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if engine is initialized
        if not hasattr(app.state, 'engine') or app.state.engine is None:
            return {
                "status": "initializing",
                "state": "NOT_READY",
                "uptime_seconds": time.time() - engine_start_time,
                "model_loaded": False,
                "active_requests": 0,
                "timestamp": datetime.now().isoformat(),
                "message": "Engine not yet initialized"
            }
        
        engine = get_engine()
        
        # Get basic engine info
        response = {
            "status": "healthy" if engine.state in (EngineState.READY, EngineState.RUNNING) else "unhealthy",
            "state": engine.state.name if hasattr(engine.state, "name") else str(engine.state),
            "uptime_seconds": time.time() - engine_start_time,
            "model_loaded": engine.model is not None,
            "active_requests": engine.active_requests,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add model info if available
        if engine.model is not None and hasattr(engine, "model_info"):
            response["model_info"] = engine.model_info
        
        return response
        
    except Exception as e:
        # Return error response if health check fails
        return {
            "status": "error",
            "state": "ERROR",
            "uptime_seconds": time.time() - engine_start_time,
            "model_loaded": False,
            "active_requests": 0,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/models/load", response_class=JSONResponse, dependencies=[Depends(verify_api_key)])
async def load_model(request: ModelLoadRequest):
    """Load a model into the inference engine"""
    engine = get_engine()
    
    # Check if path exists
    model_path = request.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(api_config["default_model_dir"], model_path)
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found: {model_path}"
        )
    
    # Convert model_type to enum
    model_type = get_model_type_enum(request.model_type)
    
    # Load the model
    try:
        success = engine.load_model(
            model_path=model_path,
            model_type=model_type,
            compile_model=request.compile_model
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load model. Check server logs for details."
            )
        
        # Return success response with model info
        return {
            "success": True,
            "message": "Model loaded successfully",
            "model_path": model_path,
            "model_type": request.model_type or "auto-detected",
            "model_info": engine.model_info if hasattr(engine, "model_info") else {}
        }
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )

@app.post("/predict", response_model=InferenceResponse, dependencies=[Depends(verify_api_key)])
async def predict(request: InferenceRequest):
    """Make a prediction using the loaded model"""
    engine = get_engine()
    
    # Check if engine is ready
    if engine.state not in (EngineState.READY, EngineState.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Engine not ready. Current state: {engine.state}"
        )
    
    # Check if model is loaded
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded"
        )
    
    # Generate request ID if not provided
    request_id = request.request_id or generate_request_id()
    
    # Validate and convert features
    features = validate_request_features(request.features)
    
    # Make prediction
    try:
        success, predictions, metadata = engine.predict(features)
        
        # Form response
        response = {
            "request_id": request_id,
            "success": success,
            "predictions": predictions.tolist() if success and hasattr(predictions, "tolist") else None,
            "error": metadata.get("error") if not success else None,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            "request_id": request_id,
            "success": False,
            "predictions": None,
            "error": str(e),
            "metadata": {"exception": str(e)},
            "timestamp": datetime.now().isoformat()
        }

@app.post("/predict/batch", response_model=BatchInferenceResponse, dependencies=[Depends(verify_api_key)])
async def predict_batch(request: BatchInferenceRequest):
    """Process a batch of prediction requests"""
    engine = get_engine()
    
    # Check if engine is ready
    if engine.state not in (EngineState.READY, EngineState.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Engine not ready. Current state: {engine.state}"
        )
    
    # Check if model is loaded
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded"
        )
    
    # Generate batch ID and request IDs if not provided
    batch_id = generate_request_id()
    request_ids = request.request_ids or [generate_request_id() for _ in range(len(request.batch))]
    
    # Process each batch item
    start_time = time.time()
    features_list = [validate_request_features(features) for features in request.batch]
    
    try:
        # Use the engine's batch processing capability
        batch_results = engine.predict_batch(features_list)
        
        # Form individual responses
        results = []
        for i, (success, predictions, metadata) in enumerate(batch_results):
            results.append({
                "request_id": request_ids[i],
                "success": success,
                "predictions": predictions.tolist() if success and hasattr(predictions, "tolist") else None,
                "error": metadata.get("error") if not success else None,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            })
        
        # Calculate batch metrics
        total_time = time.time() - start_time
        success_count = sum(1 for result in results if result["success"])
        
        # Form batch response
        response = {
            "batch_id": batch_id,
            "results": results,
            "metadata": {
                "total_time_ms": total_time * 1000,
                "batch_size": len(request.batch),
                "success_rate": success_count / len(request.batch) if request.batch else 0,
                "avg_time_per_item_ms": (total_time * 1000) / len(request.batch) if request.batch else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )

@app.post("/predict/async", response_model=AsyncJobResponse, dependencies=[Depends(verify_api_key)])
async def predict_async(request: AsyncInferenceRequest, background_tasks: BackgroundTasks):
    """Submit an asynchronous prediction request"""
    # Check if async processing is enabled
    if not api_config["enable_async"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Asynchronous processing is disabled"
        )
    
    engine = get_engine()
    
    # Check if engine is ready
    if engine.state not in (EngineState.READY, EngineState.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Engine not ready. Current state: {engine.state}"
        )
    
    # Check if model is loaded
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded"
        )
    
    # Check if engine supports dynamic batching
    if engine.dynamic_batcher is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dynamic batching is not enabled in the engine"
        )
    
    # Generate job ID and request ID
    job_id = generate_request_id()
    request_id = request.request_id or generate_request_id()
    
    # Validate and convert features
    features = validate_request_features(request.features)
    
    # Get priority enum
    priority = get_priority_enum(request.priority)
    
    try:
        # Enqueue the request
        future = engine.enqueue_prediction(
            features=features,
            priority=priority,
            timeout_ms=request.timeout_ms
        )
        
        # Store job information
        async_jobs[job_id] = {
            "status": "pending",
            "future": future,
            "result": None,
            "created_at": time.time(),
            "request_id": request_id
        }
        
        # Set up callback for when the future completes
        def process_result(future):
            try:
                result = future.result()
                async_jobs[job_id]["status"] = "completed"
                async_jobs[job_id]["result"] = {
                    "request_id": request_id,
                    "success": True,
                    "predictions": result.tolist() if hasattr(result, "tolist") else result,
                    "metadata": {},
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                async_jobs[job_id]["status"] = "failed"
                async_jobs[job_id]["result"] = {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "metadata": {"exception": str(e)},
                    "timestamp": datetime.now().isoformat()
                }
        
        future.add_done_callback(process_result)
        
        # Calculate ETA
        eta_seconds = None
        metrics = engine.metrics.get_metrics() if hasattr(engine, "metrics") else {}
        if "avg_inference_time_ms" in metrics:
            # Rough estimate based on current load and average inference time
            batcher_stats = engine.dynamic_batcher.get_stats() if hasattr(engine.dynamic_batcher, "get_stats") else {}
            queue_size = batcher_stats.get("current_queue_size", 0)
            avg_time = metrics["avg_inference_time_ms"] / 1000  # convert to seconds
            eta_seconds = avg_time * (queue_size + 1)  # +1 for this request
        
        # Return job information
        return {
            "job_id": job_id,
            "status": "pending",
            "eta_seconds": eta_seconds,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Async prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue prediction: {str(e)}"
        )

@app.get("/jobs/{job_id}", response_class=JSONResponse, dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str):
    """Get the status of an asynchronous job"""
    # Check if job exists
    if job_id not in async_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )
    
    job_info = async_jobs[job_id]
    
    # Check if job is completed
    if job_info["status"] == "completed" or job_info["status"] == "failed":
        # Return the result
        return {
            "job_id": job_id,
            "status": job_info["status"],
            "result": job_info["result"],
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Job is still pending
        # Calculate ETA if possible
        eta_seconds = None
        try:
            engine = get_engine()
            if hasattr(engine, "metrics"):
                metrics = engine.metrics.get_metrics()
                if "avg_inference_time_ms" in metrics:
                    # Rough estimate based on current load and average inference time
                    if hasattr(engine, "dynamic_batcher") and hasattr(engine.dynamic_batcher, "get_stats"):
                        batcher_stats = engine.dynamic_batcher.get_stats()
                        queue_size = batcher_stats.get("current_queue_size", 0)
                        avg_time = metrics["avg_inference_time_ms"] / 1000  # convert to seconds
                        eta_seconds = avg_time * queue_size
        except HTTPException:
            # Engine not available, skip ETA calculation
            pass
        
        return {
            "job_id": job_id,
            "status": job_info["status"],
            "eta_seconds": eta_seconds,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/metrics", response_model=PerformanceMetricsResponse, dependencies=[Depends(verify_api_key)])
async def get_metrics():
    """Get engine performance metrics"""
    engine = get_engine()
    
    try:
        metrics = engine.get_performance_metrics()
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )

@app.post("/config", response_class=JSONResponse, dependencies=[Depends(verify_api_key)])
async def update_config(request: EngineConfigRequest):
    """Update engine configuration parameters"""
    engine = get_engine()
    
    # Track which parameters are being updated
    updated_params = {}
    
    # Update configuration parameters if provided
    if request.enable_batching is not None:
        engine.config.enable_batching = request.enable_batching
        updated_params["enable_batching"] = request.enable_batching
    
    if request.max_batch_size is not None:
        engine.config.max_batch_size = request.max_batch_size
        updated_params["max_batch_size"] = request.max_batch_size
    
    if request.batch_timeout is not None:
        engine.config.batch_timeout = request.batch_timeout
        updated_params["batch_timeout"] = request.batch_timeout
    
    if request.enable_cache is not None:
        engine.config.enable_request_deduplication = request.enable_cache
        updated_params["enable_cache"] = request.enable_cache
    
    if request.max_cache_entries is not None:
        engine.config.max_cache_entries = request.max_cache_entries
        updated_params["max_cache_entries"] = request.max_cache_entries
        
        # Update cache size if cache exists
        if engine.result_cache:
            engine.result_cache.resize(request.max_cache_entries)
    
    if request.cache_ttl_seconds is not None:
        engine.config.cache_ttl_seconds = request.cache_ttl_seconds
        updated_params["cache_ttl_seconds"] = request.cache_ttl_seconds
        
        # Update cache TTL if cache exists
        if engine.result_cache:
            engine.result_cache.set_ttl(request.cache_ttl_seconds)
    
    if request.enable_quantization is not None:
        engine.config.enable_quantization = request.enable_quantization
        updated_params["enable_quantization"] = request.enable_quantization
    
    if request.num_threads is not None:
        engine.config.num_threads = request.num_threads
        updated_params["num_threads"] = request.num_threads
        # Note: This won't take effect until the engine is restarted
    
    if request.enable_throttling is not None:
        engine.config.enable_throttling = request.enable_throttling
        updated_params["enable_throttling"] = request.enable_throttling
    
    if request.max_concurrent_requests is not None:
        engine.config.max_concurrent_requests = request.max_concurrent_requests
        updated_params["max_concurrent_requests"] = request.max_concurrent_requests
    
    # Return updated configuration
    return {
        "success": True,
        "message": "Configuration updated successfully",
        "updated_parameters": updated_params,
        "current_config": {
            "enable_batching": engine.config.enable_batching,
            "max_batch_size": engine.config.max_batch_size,
            "batch_timeout": engine.config.batch_timeout,
            "enable_cache": engine.config.enable_request_deduplication,
            "max_cache_entries": engine.config.max_cache_entries,
            "cache_ttl_seconds": engine.config.cache_ttl_seconds,
            "enable_quantization": engine.config.enable_quantization,
            "num_threads": engine.config.num_threads,
            "enable_throttling": engine.config.enable_throttling,
            "max_concurrent_requests": engine.config.max_concurrent_requests
        }
    }

@app.post("/validate", response_class=JSONResponse, dependencies=[Depends(verify_api_key)])
async def validate_model():
    """Validate the loaded model to ensure it's functioning correctly"""
    engine = get_engine()
    
    # Check if model is loaded
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded"
        )
    
    try:
        # Run validation using the engine's validation method
        validation_results = engine.validate_model()
        return {
            "success": validation_results.get("valid", False),
            "model_type": validation_results.get("model_type", "Unknown"),
            "results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Model validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )

@app.delete("/models", response_class=JSONResponse, dependencies=[Depends(verify_api_key)])
async def unload_model():
    """Unload the current model from the engine"""
    engine = get_engine()
    
    # Check if model is loaded
    if engine.model is None:
        return {
            "success": True,
            "message": "No model was loaded"
        }
    
    try:
        # Clear the model
        engine.model = None
        engine.compiled_model = None
        engine.model_type = None
        engine.feature_names = []
        
        # Clear caches to free memory
        if engine.result_cache:
            engine.result_cache.clear()
        
        if engine.feature_cache:
            engine.feature_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return {
            "success": True,
            "message": "Model unloaded successfully"
        }
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}"
        )

@app.post("/cache/clear", response_class=JSONResponse, dependencies=[Depends(verify_api_key)])
async def clear_cache():
    """Clear the prediction and feature caches"""
    engine = get_engine()
    
    cache_cleared = False
    feature_cache_cleared = False
    
    try:
        # Clear result cache if it exists
        if engine.result_cache:
            engine.result_cache.clear()
            cache_cleared = True
        
        # Clear feature cache if it exists
        if engine.feature_cache:
            engine.feature_cache.clear()
            feature_cache_cleared = True
        
        return {
            "success": True,
            "result_cache_cleared": cache_cleared,
            "feature_cache_cleared": feature_cache_cleared,
            "message": "Caches cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@app.post("/restart", response_class=JSONResponse, dependencies=[Depends(verify_api_key)])
async def restart_engine():
    """Restart the inference engine with current configuration"""
    try:
        # Get current configuration
        old_engine = get_engine()
        config = old_engine.config
        
        # Create a new engine with the same configuration
        logger.info("Restarting inference engine...")
        app.state.engine = InferenceEngine(config)
        
        # Shutdown the old engine
        old_engine.shutdown()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        new_engine = get_engine()
        return {
            "success": True,
            "message": "Engine restarted successfully",
            "state": new_engine.state.name if hasattr(new_engine.state, "name") else str(new_engine.state)
        }
    except Exception as e:
        logger.error(f"Error restarting engine: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart engine: {str(e)}"
        )

@app.get("/cache/stats", response_class=JSONResponse, dependencies=[Depends(verify_api_key)])
async def cache_stats():
    """Get statistics about cache usage"""
    engine = get_engine()
    
    stats = {
        "result_cache": None,
        "feature_cache": None,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Get result cache stats if available
        if engine.result_cache:
            stats["result_cache"] = engine.result_cache.get_stats()
        
        # Get feature cache stats if available
        if engine.feature_cache:
            stats["feature_cache"] = engine.feature_cache.get_stats()
        
        return stats
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )

@app.get("/jobs", response_class=JSONResponse, dependencies=[Depends(verify_api_key)])
async def list_jobs(limit: int = Query(20, ge=1, le=100), status: Optional[str] = Query(None)):
    """List all async jobs with optional filtering"""
    try:
        jobs_list = []
        
        # Filter jobs by status if specified
        for job_id, job_info in async_jobs.items():
            if status is None or job_info["status"] == status:
                jobs_list.append({
                    "job_id": job_id,
                    "status": job_info["status"],
                    "created_at": datetime.fromtimestamp(job_info["created_at"]).isoformat(),
                    "age_seconds": time.time() - job_info["created_at"]
                })
        
        # Sort by creation time (newest first) and apply limit
        jobs_list.sort(key=lambda job: job["created_at"], reverse=True)
        jobs_list = jobs_list[:limit]
        
        return {
            "jobs": jobs_list,
            "total_count": len(jobs_list),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )

@app.post("/feature-importance", response_class=JSONResponse, dependencies=[Depends(verify_api_key)])
async def feature_importance(request: InferenceRequest):
    """Calculate feature importance for a specific input"""
    engine = get_engine()
    
    # Check if model is loaded
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded"
        )
    
    # Check if feature names are available
    if not engine.feature_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feature names not available for the loaded model"
        )
    
    # Validate and convert features
    features = validate_request_features(request.features)
    
    try:
        # Make a baseline prediction
        success, baseline_pred, _ = engine.predict(features)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to make baseline prediction"
            )
        
        # Calculate feature importance using permutation technique
        importance_values = []
        
        for i in range(features.shape[1]):
            # Create a copy with one feature permuted
            perturbed = features.copy()
            original_value = perturbed[0, i].copy()
            
            # Perturb the feature (set to zero or mean value)
            if hasattr(engine, "preprocessor") and engine.preprocessor is not None:
                # Use mean value if preprocessor is available
                try:
                    perturbed[0, i] = engine.preprocessor.get_feature_mean(i)
                except:
                    perturbed[0, i] = 0
            else:
                perturbed[0, i] = 0
            
            # Make prediction with perturbed feature
            success, perturbed_pred, _ = engine.predict(perturbed)
            if not success:
                # If prediction fails, assign zero importance
                importance = 0
            else:
                # Calculate difference (impact of this feature)
                if hasattr(baseline_pred, "flatten") and hasattr(perturbed_pred, "flatten"):
                    baseline_flat = baseline_pred.flatten()
                    perturbed_flat = perturbed_pred.flatten()
                    importance = np.sum(np.abs(baseline_flat - perturbed_flat))
                else:
                    importance = abs(baseline_pred - perturbed_pred)
            
            importance_values.append(float(importance))
        
        # Normalize importance values
        if sum(importance_values) > 0:
            importance_values = [v / sum(importance_values) for v in importance_values]
        
        # Create feature importance dictionary
        feature_importance_dict = {}
        for i, feature_name in enumerate(engine.feature_names):
            if i < len(importance_values):
                feature_importance_dict[feature_name] = importance_values[i]
        
        # Sort by importance (descending)
        sorted_importance = sorted(
            feature_importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            "success": True,
            "feature_importance": dict(sorted_importance),
            "baseline_prediction": baseline_pred.tolist() if hasattr(baseline_pred, "tolist") else baseline_pred,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate feature importance: {str(e)}"
        )

# --- Advanced Engine Features ---

@app.post("/engine/compile", dependencies=[Depends(verify_api_key)])
async def compile_model(request: ModelCompileRequest):
    """Compile the loaded model for faster inference"""
    engine = get_engine()
    
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded"
        )
    
    try:
        # Compile the model
        success = engine._compile_model(
            compile_format=request.compile_format,
            optimization_level=request.optimization_level
        )
        
        return {
            "success": success,
            "message": "Model compiled successfully" if success else "Model compilation failed",
            "compiled_model_available": engine.compiled_model is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Model compilation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compile model: {str(e)}"
        )

@app.post("/engine/warmup", dependencies=[Depends(verify_api_key)])
async def warmup_model(request: WarmupRequest):
    """Warm up the model for better performance"""
    engine = get_engine()
    
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded"
        )
    
    try:
        # Perform warmup
        engine._warmup_model(
            warmup_samples=request.warmup_samples,
            warmup_batches=request.warmup_batches
        )
        
        return {
            "success": True,
            "message": "Model warmed up successfully",
            "warmup_samples": request.warmup_samples,
            "warmup_batches": request.warmup_batches,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Model warmup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to warm up model: {str(e)}"
        )

@app.post("/engine/quantize", dependencies=[Depends(verify_api_key)])
async def quantize_model(request: QuantizationRequest):
    """Quantize the model for reduced memory usage"""
    engine = get_engine()
    
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded"
        )
    
    try:
        # Setup quantization config
        from modules.configs import QuantizationConfig, QuantizationType, QuantizationMode
        
        # Convert string to enum
        q_type = getattr(QuantizationType, request.quantization_type.upper())
        q_mode = getattr(QuantizationMode, request.quantization_mode.upper())
        
        # Create quantization config
        quant_config = QuantizationConfig(
            quantization_type=q_type,
            quantization_mode=q_mode,
            enable_cache=True
        )
        
        # Initialize quantizer if not already present
        if engine.quantizer is None:
            from modules.engine.quantizer import Quantizer
            engine.quantizer = Quantizer(quant_config)
        
        # Quantize the model
        engine._quantize_model()
        
        return {
            "success": True,
            "message": "Model quantized successfully",
            "quantization_type": request.quantization_type,
            "quantization_mode": request.quantization_mode,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Model quantization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to quantize model: {str(e)}"
        )

@app.get("/engine/cache-stats", response_model=CacheStatsResponse, dependencies=[Depends(verify_api_key)])
async def get_cache_stats():
    """Get cache statistics"""
    engine = get_engine()
    
    try:
        cache_stats = {}
        
        # Get result cache stats
        if engine.result_cache is not None:
            cache_stats["result_cache"] = {
                "size": engine.result_cache.size(),
                "max_size": engine.result_cache.max_size,
                "hit_rate": engine.result_cache.hit_rate(),
                "hits": engine.result_cache.hits,
                "misses": engine.result_cache.misses
            }
        
        # Get feature cache stats
        if engine.feature_cache is not None:
            cache_stats["feature_cache"] = {
                "size": engine.feature_cache.size(),
                "max_size": engine.feature_cache.max_size,
                "hit_rate": engine.feature_cache.hit_rate(),
                "hits": engine.feature_cache.hits,
                "misses": engine.feature_cache.misses
            }
        
        return {
            "cache_stats": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )

@app.get("/engine/batcher-stats", response_model=BatcherStatsResponse, dependencies=[Depends(verify_api_key)])
async def get_batcher_stats():
    """Get dynamic batcher statistics"""
    engine = get_engine()
    
    try:
        batcher_stats = {}
        
        if engine.dynamic_batcher is not None:
            batcher_stats = engine.dynamic_batcher.get_stats()
        
        return {
            "batcher_stats": batcher_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Batcher stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batcher stats: {str(e)}"
        )

@app.get("/engine/memory-stats", response_model=MemoryPoolStatsResponse, dependencies=[Depends(verify_api_key)])
async def get_memory_stats():
    """Get memory pool statistics"""
    engine = get_engine()
    
    try:
        memory_stats = {}
        
        if engine.memory_pool is not None:
            memory_stats = engine.memory_pool.get_stats()
        
        return {
            "memory_stats": memory_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Memory stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory stats: {str(e)}"
        )

@app.post("/engine/clear-cache", dependencies=[Depends(verify_api_key)])
async def clear_cache():
    """Clear all caches"""
    engine = get_engine()
    
    try:
        cleared_caches = []
        
        if engine.result_cache is not None:
            engine.result_cache.clear()
            cleared_caches.append("result_cache")
        
        if engine.feature_cache is not None:
            engine.feature_cache.clear()
            cleared_caches.append("feature_cache")
        
        if engine.memory_pool is not None:
            engine.memory_pool.clear()
            cleared_caches.append("memory_pool")
        
        return {
            "success": True,
            "message": "Caches cleared successfully",
            "cleared_caches": cleared_caches,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Clear cache error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@app.post("/engine/pause", dependencies=[Depends(verify_api_key)])
async def pause_engine():
    """Pause the inference engine"""
    engine = get_engine()
    
    try:
        if engine.dynamic_batcher is not None:
            engine.dynamic_batcher.stop()
        
        return {
            "success": True,
            "message": "Engine paused successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Pause engine error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause engine: {str(e)}"
        )

@app.post("/engine/resume", dependencies=[Depends(verify_api_key)])
async def resume_engine():
    """Resume the inference engine"""
    engine = get_engine()
    
    try:
        if engine.dynamic_batcher is not None:
            engine.dynamic_batcher.start()
        
        return {
            "success": True,
            "message": "Engine resumed successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Resume engine error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume engine: {str(e)}"
        )

@app.get("/engine/model-info", dependencies=[Depends(verify_api_key)])
async def get_model_info():
    """Get detailed information about the loaded model"""
    engine = get_engine()
    
    try:
        if engine.model is None:
            return {
                "success": False,
                "message": "No model loaded",
                "timestamp": datetime.now().isoformat()
            }
        
        model_info = {
            "model_type": str(engine.model_type),
            "model_info": engine.model_info,
            "feature_names": engine.feature_names,
            "compiled_model_available": engine.compiled_model is not None,
            "quantized": engine.quantizer is not None,
            "preprocessor_fitted": engine.preprocessor is not None and engine.preprocessor._fitted,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "model_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

# --- Main Entry Point ---

if __name__ == "__main__":
    # Start the API server
    uvicorn.run(
        "inference_api:app",
        host=api_config["host"],
        port=api_config["port"],
        log_level="debug" if api_config["debug"] else "info",
        reload=api_config["debug"]
    )
