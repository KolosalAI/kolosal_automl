"""
Batch Processor API

A RESTful API for the BatchProcessor engine that provides
advanced batching capabilities for ML workloads.

Features:
- Dynamic batch sizing
- Priority-based processing
- Adaptive batching strategies
- Comprehensive monitoring
- Memory optimization
- Performance metrics

Author: AI Assistant
Date: 2025-06-25
"""

import os
import sys
import time
import uuid
import json
import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Utility function for parsing boolean environment variables
def parse_bool_env(env_var: str, default: str = "False") -> bool:
    """Parse boolean environment variable safely."""
    value = os.environ.get(env_var, default).lower()
    return value in ("true", "1", "t", "yes", "on")

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import uvicorn

# Import the BatchProcessor and related modules
from modules.engine.batch_processor import BatchProcessor, BatchStats
from modules.configs import (
    BatchProcessorConfig, 
    BatchProcessingStrategy, 
    BatchPriority,
    PrioritizedItem
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("batch_processor_api.log")
    ]
)
logger = logging.getLogger("batch_processor_api")

# --- Pydantic Models ---

class BatchProcessorConfigRequest(BaseModel):
    """Request model for batch processor configuration"""
    initial_batch_size: int = Field(8, description="Initial batch size")
    min_batch_size: int = Field(1, description="Minimum batch size")
    max_batch_size: int = Field(64, description="Maximum batch size")
    batch_timeout: float = Field(0.01, description="Batch timeout in seconds")
    max_queue_size: int = Field(1000, description="Maximum queue size")
    enable_priority_queue: bool = Field(True, description="Enable priority queue")
    processing_strategy: str = Field("adaptive", description="Processing strategy")
    enable_adaptive_batching: bool = Field(True, description="Enable adaptive batching")
    enable_monitoring: bool = Field(True, description="Enable monitoring")
    num_workers: int = Field(4, description="Number of worker threads")
    enable_memory_optimization: bool = Field(True, description="Enable memory optimization")
    max_retries: int = Field(3, description="Maximum retries for failed batches")
    
    @validator("processing_strategy")
    def validate_strategy(cls, v):
        allowed = ["adaptive", "fixed", "dynamic"]
        if v.lower() not in allowed:
            raise ValueError(f"Strategy must be one of {allowed}")
        return v.upper()

class BatchItem(BaseModel):
    """Model for a single batch item"""
    data: List[List[float]] = Field(..., description="Data to process")
    priority: str = Field("normal", description="Processing priority (high, normal, low)")
    timeout: Optional[float] = Field(None, description="Optional timeout in seconds")
    
    @validator("priority")
    def validate_priority(cls, v):
        allowed = ["high", "normal", "low"]
        if v.lower() not in allowed:
            raise ValueError(f"Priority must be one of {allowed}")
        return v.upper()

class BatchRequest(BaseModel):
    """Request model for batch processing"""
    items: List[BatchItem] = Field(..., description="Items to process in batch")
    wait_for_completion: bool = Field(True, description="Whether to wait for completion")

class ProcessItemRequest(BaseModel):
    """Request for processing a single item"""
    data: List[List[float]] = Field(..., description="Data to process")
    priority: str = Field("normal", description="Processing priority")
    timeout: Optional[float] = Field(None, description="Optional timeout in seconds")

class BatchStatsResponse(BaseModel):
    """Response model for batch statistics"""
    stats: Dict[str, Any] = Field(..., description="Batch processing statistics")
    timestamp: str = Field(..., description="Response timestamp")

class BatchProcessorStatusResponse(BaseModel):
    """Response model for processor status"""
    status: str = Field(..., description="Processor status")
    is_running: bool = Field(..., description="Whether processor is running")
    queue_size: int = Field(..., description="Current queue size")
    active_batches: int = Field(..., description="Number of active batches")
    current_batch_size: int = Field(..., description="Current target batch size")
    timestamp: str = Field(..., description="Response timestamp")

class ProcessingResponse(BaseModel):
    """Response model for processing results"""
    success: bool = Field(..., description="Whether processing was successful")
    results: Optional[List[List[float]]] = Field(None, description="Processing results")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")
    timestamp: str = Field(..., description="Response timestamp")

# --- API Configuration ---

api_config = {
    "title": "Batch Processor API",
    "description": "High-performance batch processing API for ML workloads",
    "version": "1.0.0",
    "host": os.environ.get("BATCH_API_HOST", "0.0.0.0"),
    "port": int(os.environ.get("BATCH_API_PORT", "8001")),
    "debug": parse_bool_env("BATCH_API_DEBUG", "0"),
    "require_api_key": parse_bool_env("REQUIRE_API_KEY", "0"),
    "api_keys": os.environ.get("API_KEYS", "").split(",")
}

# Security
api_security = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global variables
batch_processor: Optional[BatchProcessor] = None
default_process_func: Optional[Callable] = None

# --- Security Functions ---

async def verify_api_key(api_key: str = Depends(api_security)):
    """Verify API key if required"""
    if not api_config["require_api_key"]:
        return True
    
    if not api_key or api_key not in api_config["api_keys"]:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
    return True

# --- Helper Functions ---

def get_priority_enum(priority_str: str) -> BatchPriority:
    """Convert priority string to BatchPriority enum"""
    if priority_str.upper() == "HIGH":
        return BatchPriority.HIGH
    elif priority_str.upper() == "LOW":
        return BatchPriority.LOW
    else:
        return BatchPriority.NORMAL

def get_strategy_enum(strategy_str: str) -> BatchProcessingStrategy:
    """Convert strategy string to BatchProcessingStrategy enum"""
    try:
        return BatchProcessingStrategy[strategy_str.upper()]
    except KeyError:
        return BatchProcessingStrategy.ADAPTIVE

def default_processing_function(batch_data: np.ndarray) -> np.ndarray:
    """Default processing function that just returns the input"""
    # This is a placeholder - in practice you'd implement your processing logic
    # For demonstration, we'll just return a simple transformation
    return batch_data * 2.0

# --- Lifecycle Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown"""
    global batch_processor, default_process_func
    
    logger.info("Initializing Batch Processor API...")
    
    # Set default processing function
    default_process_func = default_processing_function
    
    # Initialize with default config
    config = BatchProcessorConfig(
        initial_batch_size=int(os.environ.get("BATCH_INITIAL_SIZE", "8")),
        min_batch_size=int(os.environ.get("BATCH_MIN_SIZE", "1")),
        max_batch_size=int(os.environ.get("BATCH_MAX_SIZE", "64")),
        batch_timeout=float(os.environ.get("BATCH_TIMEOUT", "0.01")),
        max_queue_size=int(os.environ.get("BATCH_MAX_QUEUE_SIZE", "1000")),
        enable_priority_queue=parse_bool_env("BATCH_ENABLE_PRIORITY", "1"),
        processing_strategy=BatchProcessingStrategy.ADAPTIVE,
        enable_adaptive_batching=parse_bool_env("BATCH_ENABLE_ADAPTIVE", "1"),
        enable_monitoring=parse_bool_env("BATCH_ENABLE_MONITORING", "1"),
        num_workers=int(os.environ.get("BATCH_NUM_WORKERS", "4")),
        enable_memory_optimization=parse_bool_env("BATCH_ENABLE_MEMORY_OPT", "1"),
        max_retries=int(os.environ.get("BATCH_MAX_RETRIES", "3"))
    )
    
    batch_processor = BatchProcessor(config)
    
    logger.info("Batch Processor API initialized")
    yield
    
    # Cleanup
    logger.info("Shutting down Batch Processor API...")
    if batch_processor:
        batch_processor.stop()
    logger.info("Batch Processor API shutdown complete")

# Create FastAPI app
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

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "batch-processor-api",
        "version": api_config["version"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/configure", dependencies=[Depends(verify_api_key)])
async def configure_processor(config_request: BatchProcessorConfigRequest):
    """Configure the batch processor"""
    global batch_processor
    
    try:
        # Stop existing processor if running
        if batch_processor:
            batch_processor.stop()
        
        # Create new configuration
        config = BatchProcessorConfig(
            initial_batch_size=config_request.initial_batch_size,
            min_batch_size=config_request.min_batch_size,
            max_batch_size=config_request.max_batch_size,
            batch_timeout=config_request.batch_timeout,
            max_queue_size=config_request.max_queue_size,
            enable_priority_queue=config_request.enable_priority_queue,
            processing_strategy=get_strategy_enum(config_request.processing_strategy),
            enable_adaptive_batching=config_request.enable_adaptive_batching,
            enable_monitoring=config_request.enable_monitoring,
            num_workers=config_request.num_workers,
            enable_memory_optimization=config_request.enable_memory_optimization,
            max_retries=config_request.max_retries
        )
        
        # Create new processor
        batch_processor = BatchProcessor(config)
        
        return {
            "success": True,
            "message": "Batch processor configured successfully",
            "config": config_request.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure processor: {str(e)}")

@app.post("/start", dependencies=[Depends(verify_api_key)])
async def start_processor():
    """Start the batch processor"""
    global batch_processor, default_process_func
    
    if not batch_processor:
        raise HTTPException(status_code=400, detail="Processor not configured")
    
    try:
        batch_processor.start(default_process_func)
        return {
            "success": True,
            "message": "Batch processor started successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Start error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start processor: {str(e)}")

@app.post("/stop", dependencies=[Depends(verify_api_key)])
async def stop_processor(timeout: float = Query(5.0, description="Stop timeout in seconds")):
    """Stop the batch processor"""
    global batch_processor
    
    if not batch_processor:
        raise HTTPException(status_code=400, detail="Processor not configured")
    
    try:
        batch_processor.stop(timeout=timeout)
        return {
            "success": True,
            "message": "Batch processor stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stop error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop processor: {str(e)}")

@app.post("/pause", dependencies=[Depends(verify_api_key)])
async def pause_processor():
    """Pause the batch processor"""
    global batch_processor
    
    if not batch_processor:
        raise HTTPException(status_code=400, detail="Processor not configured")
    
    try:
        batch_processor.pause()
        return {
            "success": True,
            "message": "Batch processor paused successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Pause error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to pause processor: {str(e)}")

@app.post("/resume", dependencies=[Depends(verify_api_key)])
async def resume_processor():
    """Resume the batch processor"""
    global batch_processor
    
    if not batch_processor:
        raise HTTPException(status_code=400, detail="Processor not configured")
    
    try:
        batch_processor.resume()
        return {
            "success": True,
            "message": "Batch processor resumed successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Resume error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resume processor: {str(e)}")

@app.post("/process", response_model=ProcessingResponse, dependencies=[Depends(verify_api_key)])
async def process_item(request: ProcessItemRequest):
    """Process a single item"""
    global batch_processor
    
    if not batch_processor:
        raise HTTPException(status_code=400, detail="Processor not configured")
    
    try:
        # Convert data to numpy array
        data = np.array(request.data, dtype=np.float32)
        
        # Get priority
        priority = get_priority_enum(request.priority)
        
        # Submit for processing
        future = batch_processor.enqueue_predict(
            data, 
            timeout=request.timeout,
            priority=priority
        )
        
        # Wait for result
        result = future.result(timeout=request.timeout or 30.0)
        
        return {
            "success": True,
            "results": result.tolist() if hasattr(result, "tolist") else result,
            "error": None,
            "metadata": {
                "priority": request.priority,
                "batch_processed": True
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return {
            "success": False,
            "results": None,
            "error": str(e),
            "metadata": {},
            "timestamp": datetime.now().isoformat()
        }

@app.post("/process-batch", dependencies=[Depends(verify_api_key)])
async def process_batch(request: BatchRequest):
    """Process a batch of items"""
    global batch_processor
    
    if not batch_processor:
        raise HTTPException(status_code=400, detail="Processor not configured")
    
    try:
        futures = []
        
        # Submit all items
        for item in request.items:
            data = np.array(item.data, dtype=np.float32)
            priority = get_priority_enum(item.priority)
            
            future = batch_processor.enqueue_predict(
                data,
                timeout=item.timeout,
                priority=priority
            )
            futures.append(future)
        
        # Wait for results if requested
        if request.wait_for_completion:
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=request.items[i].timeout or 30.0)
                    results.append({
                        "success": True,
                        "result": result.tolist() if hasattr(result, "tolist") else result,
                        "error": None
                    })
                except Exception as e:
                    results.append({
                        "success": False,
                        "result": None,
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "message": f"Processed {len(request.items)} items",
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": True,
                "message": f"Submitted {len(request.items)} items for processing",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process batch: {str(e)}")

@app.get("/status", response_model=BatchProcessorStatusResponse, dependencies=[Depends(verify_api_key)])
async def get_status():
    """Get processor status"""
    global batch_processor
    
    if not batch_processor:
        raise HTTPException(status_code=400, detail="Processor not configured")
    
    try:
        stats = batch_processor.get_stats()
        
        return {
            "status": "running" if not batch_processor.stop_event.is_set() else "stopped",
            "is_running": not batch_processor.stop_event.is_set(),
            "queue_size": stats.get("queue_size", 0),
            "active_batches": stats.get("active_batches", 0),
            "current_batch_size": stats.get("current_batch_size", 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get("/stats", response_model=BatchStatsResponse, dependencies=[Depends(verify_api_key)])
async def get_stats():
    """Get detailed processing statistics"""
    global batch_processor
    
    if not batch_processor:
        raise HTTPException(status_code=400, detail="Processor not configured")
    
    try:
        stats = batch_processor.get_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/update-batch-size", dependencies=[Depends(verify_api_key)])
async def update_batch_size(new_size: int = Body(..., embed=True)):
    """Update the current batch size"""
    global batch_processor
    
    if not batch_processor:
        raise HTTPException(status_code=400, detail="Processor not configured")
    
    if new_size <= 0:
        raise HTTPException(status_code=400, detail="Batch size must be positive")
    
    try:
        batch_processor.update_batch_size(new_size)
        return {
            "success": True,
            "message": f"Batch size updated to {new_size}",
            "new_batch_size": new_size,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Update batch size error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update batch size: {str(e)}")

# --- Main Entry Point ---

if __name__ == "__main__":
    uvicorn.run(
        "batch_processor_api:app",
        host=api_config["host"],
        port=api_config["port"],
        log_level="debug" if api_config["debug"] else "info",
        reload=api_config["debug"]
    )
