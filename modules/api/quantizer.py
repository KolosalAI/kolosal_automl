from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np
import logging
import time
import uvicorn
import io
import json
from contextlib import asynccontextmanager

# Import the Quantizer and related classes
from modules.configs import QuantizationConfig, QuantizationType, QuantizationMode
from modules.engine.quantizer import Quantizer  

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a global quantizer instance
quantizer_instances = {}

# Define lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create a default quantizer instance
    default_config = QuantizationConfig()
    quantizer_instances["default"] = Quantizer(default_config)
    logger.info("Default quantizer instance created")
    yield
    # Clean up on shutdown
    quantizer_instances.clear()
    logger.info("Quantizer instances cleared")

app = FastAPI(
    title="Quantizer API",
    description="API for high-performance data quantization with advanced features",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models for request/response validation
class QuantizerConfigModel(BaseModel):
    quantization_type: str = Field(default="INT8", description="Quantization type (INT8, UINT8, INT16, FLOAT16, NONE, MIXED)")
    quantization_mode: str = Field(default="DYNAMIC", description="Quantization mode (DYNAMIC, DYNAMIC_PER_BATCH, CALIBRATED, etc.)")
    num_bits: int = Field(default=8, description="Number of bits for quantization")
    symmetric: bool = Field(default=False, description="Whether to use symmetric quantization")
    enable_cache: bool = Field(default=True, description="Whether to enable caching")
    cache_size: int = Field(default=1000, description="Size of the cache")
    enable_mixed_precision: bool = Field(default=False, description="Whether to enable mixed precision")
    per_channel: bool = Field(default=False, description="Whether to use per-channel quantization")
    buffer_size: int = Field(default=0, description="Size of preallocated buffers")
    use_percentile: bool = Field(default=False, description="Whether to use percentiles for range calculation")
    min_percentile: float = Field(default=0.01, description="Minimum percentile for range calculation")
    max_percentile: float = Field(default=99.99, description="Maximum percentile for range calculation")
    error_on_nan: bool = Field(default=True, description="Whether to error on NaN values")
    error_on_inf: bool = Field(default=True, description="Whether to error on Inf values")
    optimize_memory: bool = Field(default=False, description="Whether to optimize memory usage")
    enable_requantization: bool = Field(default=False, description="Whether to enable requantization")
    requantization_threshold: float = Field(default=0.01, description="Threshold for requantization")
    outlier_threshold: Optional[float] = Field(default=None, description="Threshold for outlier removal")
    skip_layers: List[str] = Field(default=[], description="Layers to skip during quantization")
    quantize_bias: bool = Field(default=True, description="Whether to quantize bias layers")
    quantize_weights_only: bool = Field(default=False, description="Whether to quantize weights only")
    mixed_precision_layers: List[str] = Field(default=[], description="Layers to use mixed precision for")
    custom_quantization_config: Dict[str, Dict[str, Any]] = Field(default={}, description="Custom quantization configuration")

class QuantizationRequest(BaseModel):
    data: List[List[float]] = Field(..., description="2D array of data to quantize")
    validate: bool = Field(default=True, description="Whether to validate input")
    channel_dim: Optional[int] = Field(default=None, description="Dimension index for per-channel quantization")
    layer_name: Optional[str] = Field(default=None, description="Layer name for mixed precision handling")

class DequantizationRequest(BaseModel):
    data: List[List[int]] = Field(..., description="2D array of quantized data to dequantize")
    channel_dim: Optional[int] = Field(default=None, description="Dimension index for per-channel dequantization")
    layer_name: Optional[str] = Field(default=None, description="Layer name for mixed precision handling")

class CalibrationRequest(BaseModel):
    data: List[List[List[float]]] = Field(..., description="List of 2D arrays for calibration")

class QuantizationParamsRequest(BaseModel):
    data: List[List[float]] = Field(..., description="2D array of data to compute parameters for")
    channel_idx: Optional[int] = Field(default=None, description="Channel index for per-channel quantization")
    layer_name: Optional[str] = Field(default=None, description="Layer name for mixed precision handling")

class ScaleZeroPointResponse(BaseModel):
    scale: float
    zero_point: float

class QuantizationResponse(BaseModel):
    data: List[List[Any]]
    quantization_type: str
    original_shape: List[int]

class StatsResponse(BaseModel):
    clipped_values: int
    total_values: int
    quantize_calls: int
    dequantize_calls: int
    cache_hits: int
    cache_misses: int
    last_scale: float
    last_zero_point: float
    processing_time_ms: float

class ImportExportParams(BaseModel):
    parameters: Dict[str, Any]

# Helper function to get a quantizer instance
def get_quantizer(instance_id: str = "default") -> Quantizer:
    if instance_id not in quantizer_instances:
        raise HTTPException(status_code=404, detail=f"Quantizer instance {instance_id} not found")
    return quantizer_instances[instance_id]

# Helper function to convert between numpy arrays and Python lists
def numpy_to_list(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr

def list_to_numpy(data, dtype=np.float32):
    return np.array(data, dtype=dtype)

# API endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Quantizer API"}

@app.get("/instances", response_model=List[str])
async def get_instances():
    """Get a list of all quantizer instances."""
    return list(quantizer_instances.keys())

@app.post("/instances/{instance_id}")
async def create_instance(instance_id: str, config: QuantizerConfigModel = Body(...)):
    """Create a new quantizer instance with the given ID and configuration."""
    if instance_id in quantizer_instances:
        raise HTTPException(status_code=409, detail=f"Quantizer instance {instance_id} already exists")
    
    try:
        # Convert Pydantic model to dictionary
        config_dict = config.model_dump()
        
        # Convert string enums to actual enum values
        config_dict["quantization_type"] = getattr(QuantizationType, config_dict["quantization_type"])
        config_dict["quantization_mode"] = getattr(QuantizationMode, config_dict["quantization_mode"])
        
        # Create quantization config
        quantization_config = QuantizationConfig(**config_dict)
        
        # Create quantizer instance
        quantizer_instances[instance_id] = Quantizer(quantization_config)
        
        return {"message": f"Quantizer instance {instance_id} created successfully"}
    except Exception as e:
        logger.error(f"Error creating quantizer instance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating quantizer instance: {str(e)}")

@app.delete("/instances/{instance_id}")
async def delete_instance(instance_id: str):
    """Delete a quantizer instance."""
    if instance_id == "default":
        raise HTTPException(status_code=403, detail="Cannot delete the default instance")
    
    if instance_id not in quantizer_instances:
        raise HTTPException(status_code=404, detail=f"Quantizer instance {instance_id} not found")
    
    del quantizer_instances[instance_id]
    return {"message": f"Quantizer instance {instance_id} deleted successfully"}

@app.get("/instances/{instance_id}/config", response_model=Dict[str, Any])
async def get_config(instance_id: str = "default"):
    """Get the configuration of a quantizer instance."""
    quantizer = get_quantizer(instance_id)
    config = quantizer.get_config()
    
    # Convert enum values to strings for JSON response
    if "quantization_type" in config and hasattr(config["quantization_type"], "name"):
        config["quantization_type"] = config["quantization_type"].name
    if "quantization_mode" in config and hasattr(config["quantization_mode"], "name"):
        config["quantization_mode"] = config["quantization_mode"].name
    
    return config

@app.put("/instances/{instance_id}/config")
async def update_config(instance_id: str, config: QuantizerConfigModel):
    """Update the configuration of a quantizer instance."""
    quantizer = get_quantizer(instance_id)
    
    try:
        # Convert Pydantic model to dictionary
        config_dict = config.model_dump()
        
        # Convert string enums to actual enum values
        config_dict["quantization_type"] = getattr(QuantizationType, config_dict["quantization_type"])
        config_dict["quantization_mode"] = getattr(QuantizationMode, config_dict["quantization_mode"])
        
        # Create and update quantization config
        quantization_config = QuantizationConfig(**config_dict)
        quantizer.update_config(quantization_config)
        
        return {"message": f"Configuration for quantizer instance {instance_id} updated successfully"}
    except Exception as e:
        logger.error(f"Error updating quantizer configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating quantizer configuration: {str(e)}")

@app.post("/instances/{instance_id}/quantize", response_model=QuantizationResponse)
async def quantize_data(
    instance_id: str, 
    request: QuantizationRequest
):
    """Quantize input data."""
    quantizer = get_quantizer(instance_id)
    
    try:
        # Convert input data to numpy array
        data = list_to_numpy(request.data)
        original_shape = data.shape
        
        # Perform quantization
        result = quantizer.quantize(
            data, 
            validate=request.validate, 
            channel_dim=request.channel_dim,
            layer_name=request.layer_name
        )
        
        # Get quantization type
        q_type = quantizer.config.quantization_type
        q_type_str = q_type.name if hasattr(q_type, "name") else str(q_type)
        
        # Convert result to list for JSON response
        result_list = numpy_to_list(result)
        
        return {
            "data": result_list,
            "quantization_type": q_type_str,
            "original_shape": list(original_shape)
        }
    except Exception as e:
        logger.error(f"Quantization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quantization error: {str(e)}")

@app.post("/instances/{instance_id}/dequantize", response_model=Dict[str, Any])
async def dequantize_data(
    instance_id: str, 
    request: DequantizationRequest
):
    """Dequantize input data."""
    quantizer = get_quantizer(instance_id)
    
    try:
        # Convert input data to numpy array (using the appropriate dtype based on quantizer config)
        with quantizer._lock:
            dtype = quantizer._qtype
        
        data = list_to_numpy(request.data, dtype=dtype)
        original_shape = data.shape
        
        # Perform dequantization
        result = quantizer.dequantize(
            data, 
            channel_dim=request.channel_dim,
            layer_name=request.layer_name
        )
        
        # Convert result to list for JSON response
        result_list = numpy_to_list(result)
        
        return {
            "data": result_list,
            "original_shape": list(original_shape)
        }
    except Exception as e:
        logger.error(f"Dequantization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dequantization error: {str(e)}")

@app.post("/instances/{instance_id}/quantize_dequantize", response_model=Dict[str, Any])
async def quantize_dequantize_data(
    instance_id: str, 
    request: QuantizationRequest
):
    """Quantize and then dequantize input data to simulate quantization effects."""
    quantizer = get_quantizer(instance_id)
    
    try:
        # Convert input data to numpy array
        data = list_to_numpy(request.data)
        original_shape = data.shape
        
        # Perform quantization and dequantization
        result = quantizer.quantize_dequantize(
            data, 
            channel_dim=request.channel_dim,
            layer_name=request.layer_name
        )
        
        # Convert result to list for JSON response
        result_list = numpy_to_list(result)
        
        return {
            "data": result_list,
            "original_shape": list(original_shape)
        }
    except Exception as e:
        logger.error(f"Quantize-dequantize error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quantize-dequantize error: {str(e)}")

@app.post("/instances/{instance_id}/calibrate")
async def calibrate_quantizer(
    instance_id: str, 
    request: CalibrationRequest
):
    """Calibrate the quantizer using representative data."""
    quantizer = get_quantizer(instance_id)
    
    try:
        # Convert calibration data to numpy arrays
        calibration_data = [list_to_numpy(data_batch) for data_batch in request.data]
        
        # Perform calibration
        quantizer.calibrate(calibration_data)
        
        return {
            "message": "Calibration successful",
            "scale": float(quantizer.scale),
            "zero_point": float(quantizer.zero_point)
        }
    except Exception as e:
        logger.error(f"Calibration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Calibration error: {str(e)}")

@app.post("/instances/{instance_id}/compute_params", response_model=ScaleZeroPointResponse)
async def compute_scale_zero_point(
    instance_id: str, 
    request: QuantizationParamsRequest
):
    """Compute scale and zero point for input data."""
    quantizer = get_quantizer(instance_id)
    
    try:
        # Convert input data to numpy array
        data = list_to_numpy(request.data)
        
        # Compute scale and zero point
        scale, zero_point = quantizer.compute_scale_and_zero_point(
            data, 
            channel_idx=request.channel_idx,
            layer_name=request.layer_name
        )
        
        return {"scale": float(scale), "zero_point": float(zero_point)}
    except Exception as e:
        logger.error(f"Error computing parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing parameters: {str(e)}")

@app.get("/instances/{instance_id}/stats", response_model=StatsResponse)
async def get_stats(instance_id: str):
    """Get quantization statistics."""
    quantizer = get_quantizer(instance_id)
    stats = quantizer.get_stats()
    return stats

@app.post("/instances/{instance_id}/clear_cache")
async def clear_cache(instance_id: str):
    """Clear the dequantization cache."""
    quantizer = get_quantizer(instance_id)
    quantizer.clear_cache()
    return {"message": "Cache cleared successfully"}

@app.post("/instances/{instance_id}/reset_stats")
async def reset_stats(instance_id: str):
    """Reset quantization statistics."""
    quantizer = get_quantizer(instance_id)
    quantizer.reset_stats()
    return {"message": "Statistics reset successfully"}

@app.get("/instances/{instance_id}/parameters", response_model=Dict[str, Any])
async def export_parameters(instance_id: str):
    """Export quantization parameters."""
    quantizer = get_quantizer(instance_id)
    params = quantizer.export_import_parameters()
    
    # Convert numpy types to Python types for JSON serialization
    for key, value in params.items():
        if isinstance(value, np.number):
            params[key] = float(value)
        elif isinstance(value, dict):
            params[key] = {k: float(v) if isinstance(v, np.number) else v for k, v in value.items()}
    
    # Convert enum values to strings
    if "quantization_type" in params and hasattr(params["quantization_type"], "name"):
        params["quantization_type"] = params["quantization_type"].name
    if "quantization_mode" in params and hasattr(params["quantization_mode"], "name"):
        params["quantization_mode"] = params["quantization_mode"].name
    
    return params

@app.post("/instances/{instance_id}/parameters")
async def import_parameters(instance_id: str, request: ImportExportParams):
    """Import quantization parameters."""
    quantizer = get_quantizer(instance_id)
    
    try:
        params = request.parameters
        
        # Convert string enums to actual enum values if needed
        if "quantization_type" in params and isinstance(params["quantization_type"], str):
            params["quantization_type"] = getattr(QuantizationType, params["quantization_type"])
        if "quantization_mode" in params and isinstance(params["quantization_mode"], str):
            params["quantization_mode"] = getattr(QuantizationMode, params["quantization_mode"])
        
        quantizer.load_parameters(params)
        return {"message": "Parameters imported successfully"}
    except Exception as e:
        logger.error(f"Error importing parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error importing parameters: {str(e)}")

@app.get("/instances/{instance_id}/layer_params/{layer_name}", response_model=Dict[str, Any])
async def get_layer_params(instance_id: str, layer_name: str):
    """Get quantization parameters for a specific layer."""
    quantizer = get_quantizer(instance_id)
    params = quantizer.get_layer_quantization_params(layer_name)
    
    # Convert enum values to strings if needed
    if "type" in params and hasattr(params["type"], "name"):
        params["type"] = params["type"].name
    
    return params

@app.post("/instances/{instance_id}/upload_numpy", response_model=Dict[str, Any])
async def upload_numpy(
    instance_id: str,
    file: UploadFile = File(...),
    operation: str = Query("quantize", description="Operation to perform: quantize, dequantize, or quantize_dequantize"),
    validate: bool = Query(True, description="Whether to validate input"),
    channel_dim: Optional[int] = Query(None, description="Dimension index for per-channel operation"),
    layer_name: Optional[str] = Query(None, description="Layer name for mixed precision handling")
):
    """Upload a NumPy array (.npy file) and perform quantization operations."""
    quantizer = get_quantizer(instance_id)
    
    if file.filename.split(".")[-1].lower() != "npy":
        raise HTTPException(status_code=400, detail="Only .npy files are supported")
    
    try:
        # Read the file content
        content = await file.read()
        
        # Load the NumPy array from the binary content
        data = np.load(io.BytesIO(content))
        original_shape = data.shape
        
        # Perform the requested operation
        if operation == "quantize":
            result = quantizer.quantize(data, validate=validate, channel_dim=channel_dim, layer_name=layer_name)
            q_type = quantizer.config.quantization_type
            q_type_str = q_type.name if hasattr(q_type, "name") else str(q_type)
            
            # Return metadata only, not the full array data
            return {
                "message": "Quantization successful",
                "shape": list(result.shape),
                "dtype": str(result.dtype),
                "quantization_type": q_type_str,
                "original_shape": list(original_shape)
            }
        
        elif operation == "dequantize":
            result = quantizer.dequantize(data, channel_dim=channel_dim, layer_name=layer_name)
            
            # Return metadata only, not the full array data
            return {
                "message": "Dequantization successful",
                "shape": list(result.shape),
                "dtype": str(result.dtype),
                "original_shape": list(original_shape)
            }
        
        elif operation == "quantize_dequantize":
            result = quantizer.quantize_dequantize(data, channel_dim=channel_dim, layer_name=layer_name)
            
            # Return metadata only, not the full array data
            return {
                "message": "Quantize-dequantize successful",
                "shape": list(result.shape),
                "dtype": str(result.dtype),
                "original_shape": list(original_shape)
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid operation: {operation}")
    
    except Exception as e:
        logger.error(f"Error processing NumPy file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing NumPy file: {str(e)}")

@app.get("/documentation")
async def get_documentation():
    """Get API documentation and usage examples."""
    docs = {
        "title": "Quantizer API Documentation",
        "description": "API for high-performance data quantization with advanced features",
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Health check endpoint"
            },
            {
                "path": "/instances",
                "method": "GET",
                "description": "Get a list of all quantizer instances"
            },
            # Add other endpoints here...
        ],
        "examples": [
            {
                "title": "Create a quantizer instance with custom configuration",
                "description": "Create a new quantizer instance with INT8 quantization",
                "code": "curl -X POST \"http://localhost:8000/instances/my_quantizer\" -H \"Content-Type: application/json\" -d '{\"quantization_type\": \"INT8\", \"num_bits\": 8, \"symmetric\": true}'"
            },
            {
                "title": "Quantize data",
                "description": "Quantize a 2D array of data",
                "code": "curl -X POST \"http://localhost:8000/instances/default/quantize\" -H \"Content-Type: application/json\" -d '{\"data\": [[1.2, 3.4, 5.6], [7.8, 9.0, 2.1]]}'"
            },
            # Add other examples here...
        ]
    }
    
    return docs

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)