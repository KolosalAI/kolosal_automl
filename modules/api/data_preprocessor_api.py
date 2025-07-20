"""
Data Preprocessor API Module

Enhanced API for the DataPreprocessor class with comprehensive features:
- Advanced configuration management
- Multiple data format support (JSON, CSV, NumPy)
- Batch and streaming processing
- Performance monitoring and statistics
- Memory optimization controls
- Caching and persistence
- Real-time metrics and health monitoring

Author: AI Assistant  
Date: 2025-06-25
"""

import os
import io
import uuid
import logging
import tempfile
import pickle
import json
import asyncio
import csv
import numpy as np
import pandas as pd
import sys
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Depends, Body, Form, status, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum

# Import the DataPreprocessor and related classes
from modules.engine.data_preprocessor import (
    DataPreprocessor,
    PreprocessingError,
    InputValidationError,
    StatisticsError,
    SerializationError
)
from modules.configs import (
    PreprocessorConfig,
    NormalizationType
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_preprocessor_api.log")
    ]
)
logger = logging.getLogger("data_preprocessor_api")

# --- Configuration ---

api_config = {
    "title": "Data Preprocessor API",
    "description": "Enhanced API for advanced data preprocessing operations",
    "version": "0.1.4",
    "host": os.environ.get("PREPROCESSOR_API_HOST", "0.0.0.0"),
    "port": int(os.environ.get("PREPROCESSOR_API_PORT", "8002")),
    "debug": os.environ.get("PREPROCESSOR_API_DEBUG", "False").lower() in ("true", "1", "t"),
    "require_api_key": os.environ.get("REQUIRE_API_KEY", "False").lower() in ("true", "1", "t"),
    "api_keys": os.environ.get("API_KEYS", "").split(","),
    "max_workers": int(os.environ.get("MAX_WORKERS", "4"))
}

# Storage directories
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
TEMP_DATA_DIR = os.environ.get("TEMP_DATA_DIR", "./temp_data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# Security
api_security = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global storage
active_preprocessors: Dict[str, DataPreprocessor] = {}
thread_pool: Optional[ThreadPoolExecutor] = None

# --- Enhanced Pydantic Models ---

class NormalizationTypeEnum(str, Enum):
    NONE = "NONE"
    STANDARD = "STANDARD"
    MINMAX = "MINMAX"
    ROBUST = "ROBUST"
    LOG = "LOG"
    QUANTILE = "QUANTILE"
    POWER = "POWER"
    CUSTOM = "CUSTOM"

class OutlierMethodEnum(str, Enum):
    IQR = "IQR"
    ZSCORE = "ZSCORE"
    PERCENTILE = "PERCENTILE"

class OutlierHandlingEnum(str, Enum):
    CLIP = "CLIP"
    REMOVE = "REMOVE"
    WINSORIZE = "WINSORIZE"
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"

class NanStrategyEnum(str, Enum):
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    MOST_FREQUENT = "MOST_FREQUENT"
    CONSTANT = "CONSTANT"
    ZERO = "ZERO"

class InfStrategyEnum(str, Enum):
    MAX_VALUE = "MAX_VALUE"
    MEDIAN = "MEDIAN"
    CONSTANT = "CONSTANT"
    NAN = "NAN"

# Request Models
class PreprocessorConfigRequest(BaseModel):
    """Configuration for creating a new preprocessor instance"""
    normalization: NormalizationTypeEnum = Field(NormalizationTypeEnum.STANDARD, description="Normalization method to apply")
    handle_nan: bool = Field(True, description="Whether to handle NaN values")
    handle_inf: bool = Field(True, description="Whether to handle infinite values")
    detect_outliers: bool = Field(False, description="Whether to detect and handle outliers")
    nan_strategy: NanStrategyEnum = Field(NanStrategyEnum.MEAN, description="Strategy for handling NaN values")
    inf_strategy: InfStrategyEnum = Field(InfStrategyEnum.MAX_VALUE, description="Strategy for handling infinite values")
    outlier_method: OutlierMethodEnum = Field(OutlierMethodEnum.ZSCORE, description="Method for detecting outliers")
    outlier_handling: OutlierHandlingEnum = Field(OutlierHandlingEnum.CLIP, description="Strategy for handling outliers")
    robust_percentiles: Tuple[float, float] = Field((25.0, 75.0), description="Percentiles for robust scaling")
    outlier_iqr_multiplier: float = Field(1.5, description="Multiplier for IQR outlier detection")
    outlier_zscore_threshold: float = Field(3.0, description="Z-score threshold for outlier detection")
    outlier_percentiles: Tuple[float, float] = Field((1.0, 99.0), description="Percentiles for outlier detection")
    epsilon: float = Field(1e-8, description="Small constant to add for numerical stability")
    clip_values: bool = Field(False, description="Whether to clip values to specified range")
    clip_min: float = Field(-float('inf'), description="Minimum value for clipping")
    clip_max: float = Field(float('inf'), description="Maximum value for clipping")
    nan_fill_value: float = Field(0.0, description="Value to use when filling NaNs with constant strategy")
    copy_X: bool = Field(True, description="Whether to copy the input data")
    dtype: str = Field("float32", description="Data type to use for processing")
    debug_mode: bool = Field(False, description="Whether to enable debug logging")
    parallel_processing: bool = Field(False, description="Whether to enable parallel processing")
    n_jobs: int = Field(-1, description="Number of parallel jobs (-1 for all cores)")
    chunk_size: Optional[int] = Field(10000, description="Chunk size for processing large datasets")
    cache_enabled: bool = Field(True, description="Whether to enable caching")
    enable_input_validation: bool = Field(True, description="Whether to validate input data")
    input_size_limit: Optional[int] = Field(None, description="Maximum input size limit")
    version: str = Field("1.0.0", description="Preprocessor version")
    
    @validator('dtype')
    def validate_dtype(cls, v):
        allowed_dtypes = ['float32', 'float64', 'int32', 'int64']
        if v not in allowed_dtypes:
            raise ValueError(f"dtype must be one of {allowed_dtypes}")
        return v
    
    @validator('robust_percentiles', 'outlier_percentiles')
    def validate_percentiles(cls, v):
        if not (0 <= v[0] < v[1] <= 100):
            raise ValueError("Percentiles must be in [0,100] range with lower < upper")
        return v

class DataRequest(BaseModel):
    """Request model for inline data submission"""
    data: List[List[float]] = Field(..., description="2D array of data")
    feature_names: Optional[List[str]] = Field(None, description="Feature names")
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        
        if not all(isinstance(row, list) for row in v):
            raise ValueError("Data must be a 2D array")
            
        row_lengths = set(len(row) for row in v)
        if len(row_lengths) > 1:
            raise ValueError("All rows must have the same length")
            
        return v
        
    @validator('feature_names')
    def validate_feature_names(cls, v, values):
        if v is not None and 'data' in values:
            if not values['data']:
                return v
                
            n_features = len(values['data'][0])
            if len(v) != n_features:
                raise ValueError(f"Expected {n_features} feature names but got {len(v)}")
        return v

class TransformOptions(BaseModel):
    """Options for transform operations"""
    copy_data: bool = Field(True, description="Whether to copy the input data", alias="copy")

class PreprocessorCreateResponse(BaseModel):
    """Response model for preprocessor creation"""
    preprocessor_id: str = Field(..., description="Unique ID of the created preprocessor")
    config: Dict[str, Any] = Field(..., description="Configuration used for the preprocessor")
    created_at: str = Field(..., description="Creation timestamp")

class StatusResponse(BaseModel):
    """Generic status response"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    
class PreprocessorInfoResponse(BaseModel):
    """Information about a preprocessor instance"""
    preprocessor_id: str = Field(..., description="Unique ID of the preprocessor")
    fitted: bool = Field(..., description="Whether the preprocessor has been fitted")
    n_features: Optional[int] = Field(None, description="Number of features")
    n_samples_seen: Optional[int] = Field(None, description="Number of samples seen during fitting")
    feature_names: Optional[List[str]] = Field(None, description="Feature names")
    config: Dict[str, Any] = Field(..., description="Preprocessor configuration")
    created_at: str = Field(..., description="Creation timestamp")

class StatisticsResponse(BaseModel):
    """Response model for preprocessor statistics"""
    preprocessor_id: str = Field(..., description="Unique ID of the preprocessor")
    statistics: Dict[str, Any] = Field(..., description="Preprocessor statistics")

class MetricsResponse(BaseModel):
    """Response model for preprocessor performance metrics"""
    preprocessor_id: str = Field(..., description="Unique ID of the preprocessor")
    metrics: Dict[str, List[float]] = Field(..., description="Performance metrics")

class AllPreprocessorsResponse(BaseModel):
    """Response listing all available preprocessors"""
    preprocessors: List[PreprocessorInfoResponse] = Field(..., description="List of preprocessor information")

# Helper functions
def get_preprocessor(preprocessor_id: str) -> DataPreprocessor:
    """
    Retrieve a preprocessor instance by ID.
    
    Args:
        preprocessor_id: ID of the preprocessor to retrieve
        
    Returns:
        DataPreprocessor instance
        
    Raises:
        HTTPException: If preprocessor not found
    """
    if preprocessor_id not in active_preprocessors:
        raise HTTPException(
            status_code=404,
            detail=f"Preprocessor with ID {preprocessor_id} not found"
        )
    return active_preprocessors[preprocessor_id]

def parse_csv_data(file_content: bytes, has_header: bool = True) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Parse CSV data from uploaded file content.
    
    Args:
        file_content: Raw bytes of the CSV file
        has_header: Whether the CSV has a header row
        
    Returns:
        Tuple of (numpy array of data, list of feature names or None)
        
    Raises:
        HTTPException: If CSV parsing fails
    """
    try:
        # Create a file-like object from bytes
        file_obj = io.StringIO(file_content.decode('utf-8'))
        
        # Check if we have a header row
        if has_header:
            # Read with pandas to automatically handle headers
            df = pd.read_csv(file_obj)
            feature_names = df.columns.tolist()
            data = df.values
        else:
            # Read without header
            df = pd.read_csv(file_obj, header=None)
            feature_names = None
            data = df.values
            
        return data, feature_names
    except Exception as e:
        logger.error(f"Error parsing CSV data: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV data: {str(e)}"
        )

def create_error_response(exc: Exception, status_code: int) -> JSONResponse:
    """
    Create a standardized error response.
    
    Args:
        exc: The exception that occurred
        status_code: HTTP status code
        
    Returns:
        JSONResponse with error details
    """
    error_type = exc.__class__.__name__
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "type": error_type,
                "message": str(exc),
                "details": getattr(exc, "details", None)
            }
        }
    )

def clean_config_for_creation(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean the configuration dictionary for preprocessor creation.
    
    Args:
        config_dict: Raw configuration dictionary
        
    Returns:
        Cleaned configuration dictionary
    """
    # Convert Enum values to strings
    for key, value in config_dict.items():
        if isinstance(value, Enum):
            config_dict[key] = value.value
    
    return config_dict

def create_preprocessor_config(config_dict: Dict[str, Any]) -> PreprocessorConfig:
    """
    Create a PreprocessorConfig instance from a dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        PreprocessorConfig instance
    """
    cleaned_config = clean_config_for_creation(config_dict)
    
    # Handle normalization enum conversion
    if 'normalization' in cleaned_config:
        cleaned_config['normalization'] = getattr(NormalizationType, cleaned_config['normalization'])
    
    return PreprocessorConfig(**cleaned_config)

# --- Lifecycle Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown"""
    global thread_pool
    
    logger.info("Initializing Data Preprocessor API...")
    
    # Initialize thread pool
    thread_pool = ThreadPoolExecutor(max_workers=api_config["max_workers"])
    
    logger.info("Data Preprocessor API initialized")
    yield
    
    # Cleanup
    logger.info("Shutting down Data Preprocessor API...")
    if thread_pool:
        thread_pool.shutdown(wait=False)
    logger.info("Data Preprocessor API shutdown complete")

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

# --- Security Functions ---

async def verify_api_key(api_key: str = Depends(api_security)):
    """Verify API key if required"""
    if not api_config["require_api_key"]:
        return True
    
    if not api_key or api_key not in api_config["api_keys"]:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
    return True

# Exception handlers
@app.exception_handler(PreprocessingError)
def preprocessing_error_handler(request, exc):
    return create_error_response(exc, status.HTTP_400_BAD_REQUEST)

@app.exception_handler(InputValidationError)
def input_validation_error_handler(request, exc):
    return create_error_response(exc, status.HTTP_400_BAD_REQUEST)

@app.exception_handler(StatisticsError)
def statistics_error_handler(request, exc):
    return create_error_response(exc, status.HTTP_400_BAD_REQUEST)

@app.exception_handler(SerializationError)
def serialization_error_handler(request, exc):
    return create_error_response(exc, status.HTTP_500_INTERNAL_SERVER_ERROR)

# API Endpoints

@app.post(
    "/preprocessors", 
    response_model=PreprocessorCreateResponse, 
    status_code=status.HTTP_201_CREATED,
    summary="Create a new preprocessor instance"
)
def create_preprocessor(config: PreprocessorConfigRequest = Body(...)):
    """
    Create a new preprocessor instance with the specified configuration.
    
    The created preprocessor is assigned a unique ID and stored for later use.
    
    Returns:
        PreprocessorCreateResponse: Details of the created preprocessor instance
    """
    try:
        # Generate a unique ID for the preprocessor
        preprocessor_id = str(uuid.uuid4())
        
        # Create preprocessor config
        config_dict = config.dict()
        preprocessor_config = create_preprocessor_config(config_dict)
        
        # Create preprocessor instance
        preprocessor = DataPreprocessor(config=preprocessor_config)
        
        # Store the preprocessor
        active_preprocessors[preprocessor_id] = preprocessor
        
        # Create response with safe JSON values
        safe_config = {}
        for key, value in config_dict.items():
            if isinstance(value, float):
                if np.isnan(value):
                    safe_config[key] = None
                elif np.isinf(value):
                    safe_config[key] = 1e10 if value > 0 else -1e10
                else:
                    safe_config[key] = value
            else:
                safe_config[key] = value
        
        return PreprocessorCreateResponse(
            preprocessor_id=preprocessor_id,
            config=safe_config,
            created_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error creating preprocessor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create preprocessor: {str(e)}"
        )

@app.get(
    "/preprocessors",
    response_model=AllPreprocessorsResponse,
    summary="List all active preprocessors"
)
def list_preprocessors():
    """
    List all currently active preprocessor instances.
    
    Returns:
        AllPreprocessorsResponse: List of active preprocessor information
    """
    preprocessors_info = []
    
    def make_json_safe(obj):
        """Convert value to JSON-safe format"""
        if isinstance(obj, (int, str, bool, type(None))):
            return obj
        elif isinstance(obj, float):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return obj
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, type):
            return str(obj.__name__)
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        else:
            return str(obj)
    
    for preprocessor_id, preprocessor in active_preprocessors.items():
        # Convert config to dict correctly, handling the Enum values and infinity
        config_dict = {}
        for key, value in asdict(preprocessor.config).items():
            config_dict[key] = make_json_safe(value)
        
        # Get basic info for each preprocessor
        info = {
            "preprocessor_id": preprocessor_id,
            "fitted": preprocessor._fitted,
            "config": config_dict,
            "created_at": datetime.now().isoformat()  # We don't store creation time, so use current time
        }
        
        # Add additional info if fitted
        if preprocessor._fitted:
            info["n_features"] = preprocessor._n_features
            info["n_samples_seen"] = preprocessor._n_samples_seen
            info["feature_names"] = preprocessor._feature_names
            
        preprocessors_info.append(info)
    
    return AllPreprocessorsResponse(preprocessors=preprocessors_info)

@app.get(
    "/preprocessors/{preprocessor_id}",
    response_model=PreprocessorInfoResponse,
    summary="Get preprocessor information"
)
def get_preprocessor_info(preprocessor_id: str):
    """
    Get detailed information about a specific preprocessor instance.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        
    Returns:
        PreprocessorInfoResponse: Detailed preprocessor information
    """
    preprocessor = get_preprocessor(preprocessor_id)
    
    def make_json_safe(obj):
        """Convert value to JSON-safe format"""
        if isinstance(obj, (int, str, bool, type(None))):
            return obj
        elif isinstance(obj, float):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return obj
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, type):
            return str(obj.__name__)
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        else:
            return str(obj)
    
    # Build response with JSON-safe config
    config_dict = {}
    for key, value in asdict(preprocessor.config).items():
        config_dict[key] = make_json_safe(value)
    
    response = {
        "preprocessor_id": preprocessor_id,
        "fitted": preprocessor._fitted,
        "config": config_dict,
        "created_at": datetime.now().isoformat()  # We don't store creation time, so use current time
    }
    
    # Add additional info if fitted
    if preprocessor._fitted:
        response["n_features"] = preprocessor._n_features
        response["n_samples_seen"] = preprocessor._n_samples_seen
        response["feature_names"] = preprocessor._feature_names
        
    return response

@app.delete(
    "/preprocessors/{preprocessor_id}",
    response_model=StatusResponse,
    summary="Delete a preprocessor instance"
)
def delete_preprocessor(preprocessor_id: str):
    """
    Delete a preprocessor instance from memory.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor to delete
        
    Returns:
        StatusResponse: Status of the delete operation
    """
    # Check if preprocessor exists
    if preprocessor_id not in active_preprocessors:
        raise HTTPException(
            status_code=404,
            detail=f"Preprocessor with ID {preprocessor_id} not found"
        )
    
    # Remove the preprocessor
    del active_preprocessors[preprocessor_id]
    
    return StatusResponse(
        success=True,
        message=f"Preprocessor {preprocessor_id} deleted successfully"
    )

@app.post(
    "/preprocessors/{preprocessor_id}/fit",
    response_model=StatusResponse,
    summary="Fit a preprocessor to data"
)
async def fit_preprocessor(
    preprocessor_id: str,
    request: Request,
    csv_file: Optional[UploadFile] = File(None),
    has_header: bool = Query(True, description="Whether the CSV file has a header row")
):
    """
    Fit a preprocessor to the provided data.
    
    The data can be provided either as a JSON structure or as a CSV file upload.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        data: JSON data structure (optional)
        csv_file: CSV file upload (optional)
        has_header: Whether the CSV file has a header row
        
    Returns:
        StatusResponse: Status of the fit operation
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    try:
        data = None
        # Check if this is a multipart request (has file) or JSON request
        content_type = request.headers.get("content-type", "")
        
        if csv_file is not None:
            # Use CSV file
            file_content = await csv_file.read()
            X, feature_names = parse_csv_data(file_content, has_header)
        elif "application/json" in content_type:
            # Handle JSON request
            json_data = await request.json()
            if isinstance(json_data, dict):
                X = np.array(json_data.get("data", []))
                feature_names = json_data.get("feature_names", None)
            else:
                # If it's just an array, assume it's the data
                X = np.array(json_data)
                feature_names = None
        elif "multipart/form-data" in content_type:
            # Handle form data
            form = await request.form()
            data_str = form.get("data")
            if data_str:
                try:
                    data_dict = json.loads(data_str)
                    if isinstance(data_dict, dict):
                        X = np.array(data_dict.get("data", []))
                        feature_names = data_dict.get("feature_names", None)
                    else:
                        X = np.array(data_dict)
                        feature_names = None
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid JSON format in data field"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No data provided. Please provide either JSON data or a CSV file."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No data provided. Please provide either JSON data or a CSV file."
            )
        
        # Fit the preprocessor
        preprocessor.fit(X, feature_names)
        
        return StatusResponse(
            success=True,
            message=f"Preprocessor {preprocessor_id} fitted successfully to {X.shape[0]} samples with {X.shape[1]} features"
        )
    except InputValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fitting preprocessor: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fit preprocessor: {str(e)}"
        )

@app.post(
    "/preprocessors/{preprocessor_id}/transform",
    summary="Transform data using a fitted preprocessor"
)
async def transform_data(
    preprocessor_id: str,
    request: Request,
    csv_file: Optional[UploadFile] = File(None),
    has_header: bool = Query(True, description="Whether the CSV file has a header row"),
    output_format: str = Query("json", description="Output format (json or csv)")
):
    """
    Transform data using a fitted preprocessor.
    
    The data can be provided either as a JSON structure or as a CSV file upload.
    The transformed data can be returned as JSON or CSV.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        options: Transform options
        data: JSON data structure (optional)
        csv_file: CSV file upload (optional)
        has_header: Whether the CSV file has a header row
        output_format: Output format (json or csv)
        
    Returns:
        Transformed data in the specified format
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    # Check if preprocessor is fitted
    if not preprocessor._fitted:
        raise HTTPException(
            status_code=400,
            detail="Preprocessor is not fitted. Please fit the preprocessor first."
        )
    
    try:
        # Parse options if provided
        transform_options = TransformOptions()
        data = None
        options = None
        
        # Check if this is a multipart request (has file) or JSON request
        content_type = request.headers.get("content-type", "")
        
        if csv_file is not None:
            # Use CSV file
            file_content = await csv_file.read()
            X, _ = parse_csv_data(file_content, has_header)
        elif "application/json" in content_type:
            # Handle JSON request
            json_data = await request.json()
            if isinstance(json_data, dict):
                # Check if it's wrapped in a "data" field
                if "data" in json_data:
                    X = np.array(json_data["data"])
                else:
                    X = np.array(json_data)
            else:
                # If it's just an array, assume it's the data
                X = np.array(json_data)
        elif "multipart/form-data" in content_type:
            # Handle form data
            form = await request.form()
            data_str = form.get("data")
            options_str = form.get("options")
            
            if options_str:
                try:
                    options_dict = json.loads(options_str)
                    if isinstance(options_dict, dict):
                        transform_options.copy_data = options_dict.get("copy", True)
                except json.JSONDecodeError:
                    # Use default options if JSON is invalid
                    pass
            
            if data_str:
                try:
                    data_dict = json.loads(data_str)
                    if isinstance(data_dict, dict):
                        X = np.array(data_dict.get("data", []))
                    else:
                        X = np.array(data_dict)
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid JSON format in data field"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No data provided. Please provide either JSON data or a CSV file."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No data provided. Please provide either JSON data or a CSV file."
            )
        
        # Transform the data
        transformed_data = preprocessor.transform(X, copy=transform_options.copy_data)
        
        # Prepare response based on output format
        if output_format.lower() == "json":
            # Return JSON response
            return {
                "transformed_data": transformed_data.tolist(),
                "shape": transformed_data.shape
            }
        elif output_format.lower() == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            if preprocessor._feature_names:
                writer.writerow(preprocessor._feature_names)
            for row in transformed_data:
                writer.writerow(row)
            output.seek(0)
            csv_bytes = output.getvalue().encode("utf-8")          # ← encode to bytes
            return StreamingResponse(
                iter([csv_bytes]),                                 # ← yield bytes
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f"attachment; filename=transformed_{preprocessor_id}.csv"}
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output format: {output_format}. Supported formats: json, csv"
            )
    except InputValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error transforming data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transform data: {str(e)}"
        )
@app.post(
    "/preprocessors/{preprocessor_id}/fit-transform",
    summary="Fit and transform data in one operation"
)
async def fit_transform_data(
    preprocessor_id: str,
    request: Request,
    has_header: bool = Query(True, description="Whether the CSV file has a header row"),
    output_format: str = Query("json", description="Output format (json or csv)")
):
    """
    Fit the preprocessor to data and transform it in one operation.
    
    The data can be provided either as a JSON structure or as a CSV file upload.
    The transformed data can be returned as JSON or CSV.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        request: HTTP request containing options and data
        has_header: Whether the CSV file has a header row
        output_format: Output format (json or csv)
        
    Returns:
        Transformed data in the specified format
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    try:
        content_type = request.headers.get("content-type", "")
        
        # Parse options and data based on content type
        transform_options = TransformOptions()
        X = None
        feature_names = None
        
        if "application/json" in content_type:
            # Handle JSON request
            body = await request.json()
            
            # Parse options if provided
            if "options" in body:
                options_data = body["options"]
                if isinstance(options_data, dict):
                    transform_options.copy = options_data.get("copy", True)
            
            # Parse data
            if "data" in body:
                X = np.array(body["data"])
                feature_names = body.get("feature_names", None)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No data provided in JSON request. Please provide 'data' field."
                )
                
        elif "multipart/form-data" in content_type:
            # Handle multipart form data
            form = await request.form()
            
            # Parse options if provided
            if "options" in form:
                try:
                    options_dict = json.loads(form["options"])
                    if isinstance(options_dict, dict):
                        transform_options.copy = options_dict.get("copy", True)
                except json.JSONDecodeError:
                    # Use default options if JSON is invalid
                    pass
            
            # Process data based on input method
            if "data" in form:
                # Parse JSON data from form field
                try:
                    data_dict = json.loads(form["data"])
                    if isinstance(data_dict, dict):
                        X = np.array(data_dict.get("data", []))
                        feature_names = data_dict.get("feature_names", None)
                    else:
                        # If it's just an array, assume it's the data
                        X = np.array(data_dict)
                        feature_names = None
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid JSON format in data field"
                    )
            elif "csv_file" in form:
                # Use CSV file
                csv_file = form["csv_file"]
                if hasattr(csv_file, 'read'):
                    file_content = await csv_file.read()
                    X, feature_names = parse_csv_data(file_content, has_header)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid CSV file upload"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No data provided. Please provide either JSON data or a CSV file."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No data provided. Please provide either JSON data or a CSV file."
            )
        
        # Fit and transform the data
        transformed_data = preprocessor.fit_transform(X, feature_names=feature_names, copy=transform_options.copy_data)
        
        # Prepare response based on output format
        if output_format.lower() == "json":
            # Return JSON response
            return {
                "transformed_data": transformed_data.tolist(),
                "shape": transformed_data.shape
            }
        elif output_format.lower() == "csv":
            # Return CSV response
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header if we have feature names
            if preprocessor._feature_names:
                writer.writerow(preprocessor._feature_names)
                
            # Write data rows
            for row in transformed_data:
                writer.writerow(row)
                
            output.seek(0)
            csv_bytes = output.getvalue().encode("utf-8")  # Convert string to bytes
            
            return StreamingResponse(
                iter([csv_bytes]),  # Provide bytes
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=fit_transformed_{preprocessor_id}.csv"
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output format: {output_format}. Supported formats: json, csv"
            )
    except InputValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in fit_transform operation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed in fit_transform operation: {str(e)}"
        )

@app.post(
    "/preprocessors/{preprocessor_id}/reverse-transform",
    summary="Reverse transform data using a fitted preprocessor"
)
async def reverse_transform_data(
    preprocessor_id: str,
    request: Request,
    has_header: bool = Query(True, description="Whether the CSV file has a header row"),
    output_format: str = Query("json", description="Output format (json or csv)")
):
    """
    Reverse transform data using a fitted preprocessor.
    
    The data can be provided either as a JSON structure or as a CSV file upload.
    This allows converting normalized data back to its original scale.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        request: HTTP request containing options and data
        has_header: Whether the CSV file has a header row
        output_format: Output format (json or csv)
        
    Returns:
        Reverse transformed data in the specified format
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    # Check if preprocessor is fitted
    if not preprocessor._fitted:
        raise HTTPException(
            status_code=400,
            detail="Preprocessor is not fitted. Please fit the preprocessor first."
        )
    
    try:
        content_type = request.headers.get("content-type", "")
        
        # Parse options and data based on content type
        transform_options = TransformOptions()
        X = None
        
        if "application/json" in content_type:
            # Handle JSON request
            body = await request.json()
            
            # Parse options if provided
            if "options" in body:
                options_data = body["options"]
                if isinstance(options_data, dict):
                    transform_options.copy = options_data.get("copy", True)
            
            # Parse data
            if "data" in body:
                X = np.array(body["data"])
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No data provided in JSON request. Please provide 'data' field."
                )
                
        elif "multipart/form-data" in content_type:
            # Handle multipart form data
            form = await request.form()
            
            # Parse options if provided
            if "options" in form:
                try:
                    options_dict = json.loads(form["options"])
                    if isinstance(options_dict, dict):
                        transform_options.copy = options_dict.get("copy", True)
                except json.JSONDecodeError:
                    # Use default options if JSON is invalid
                    pass
            
            # Process data based on input method
            if "data" in form:
                # Parse JSON data from form field
                try:
                    data_dict = json.loads(form["data"])
                    if isinstance(data_dict, dict):
                        X = np.array(data_dict.get("data", []))
                    else:
                        # If it's just an array, assume it's the data
                        X = np.array(data_dict)
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid JSON format in data field"
                    )
            elif "csv_file" in form:
                # Use CSV file
                csv_file = form["csv_file"]
                if hasattr(csv_file, 'read'):
                    file_content = await csv_file.read()
                    X, _ = parse_csv_data(file_content, has_header)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid CSV file upload"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No data provided. Please provide either JSON data or a CSV file."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No data provided. Please provide either JSON data or a CSV file."
            )
        
        # Reverse transform the data
        reverse_transformed_data = preprocessor.reverse_transform(X, copy=transform_options.copy_data)
        
        # Prepare response based on output format
        if output_format.lower() == "json":
            # Return JSON response
            return {
                "reverse_transformed_data": reverse_transformed_data.tolist(),
                "shape": reverse_transformed_data.shape
            }
        elif output_format.lower() == "csv":
            # Return CSV response
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header if we have feature names
            if preprocessor._feature_names:
                writer.writerow(preprocessor._feature_names)
                
            # Write data rows
            for row in reverse_transformed_data:
                writer.writerow(row)
                
            output.seek(0)
            csv_bytes = output.getvalue().encode("utf-8")  # Convert string to bytes
            
            return StreamingResponse(
                iter([csv_bytes]),  # Provide bytes
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=reverse_transformed_{preprocessor_id}.csv"
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output format: {output_format}. Supported formats: json, csv"
            )
    except InputValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in reverse transform: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reverse transform data: {str(e)}"
        )

@app.post(
    "/preprocessors/{preprocessor_id}/partial-fit",
    response_model=StatusResponse,
    summary="Partially fit a preprocessor with new data"
)
async def partial_fit_preprocessor(
    preprocessor_id: str,
    request: Request,
    has_header: bool = Query(True, description="Whether the CSV file has a header row")
):
    """
    Update a preprocessor with new data incrementally.
    
    The data can be provided either as a JSON structure or as a CSV file upload.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        request: HTTP request containing data and feature_names
        has_header: Whether the CSV file has a header row
        
    Returns:
        StatusResponse: Status of the partial fit operation
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    try:
        content_type = request.headers.get("content-type", "")
        X = None
        feature_names = None
        
        if "application/json" in content_type:
            # Handle JSON request
            body = await request.json()
            
            # Parse data
            if "data" in body:
                X = np.array(body["data"])
                feature_names = body.get("feature_names", None)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No data provided in JSON request. Please provide 'data' field."
                )
                
        elif "multipart/form-data" in content_type:
            # Handle multipart form data
            form = await request.form()
            
            # Process data based on input method
            if "data" in form:
                # Parse JSON data from form field
                try:
                    data_dict = json.loads(form["data"])
                    if isinstance(data_dict, dict):
                        X = np.array(data_dict.get("data", []))
                        feature_names = data_dict.get("feature_names", None)
                    else:
                        # If it's just an array, assume it's the data
                        X = np.array(data_dict)
                        feature_names = None
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid JSON format in data field"
                    )
            elif "csv_file" in form:
                # Use CSV file
                csv_file = form["csv_file"]
                if hasattr(csv_file, 'read'):
                    file_content = await csv_file.read()
                    X, feature_names = parse_csv_data(file_content, has_header)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid CSV file upload"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No data provided. Please provide either JSON data or a CSV file."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No data provided. Please provide either JSON data or a CSV file."
            )
        
        # Partial fit the preprocessor
        preprocessor.partial_fit(X, feature_names=feature_names)
        
        return StatusResponse(
            success=True,
            message=f"Preprocessor {preprocessor_id} partially fitted with {X.shape[0]} additional samples"
        )
    except InputValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in partial_fit: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed in partial_fit operation: {str(e)}"
        )

@app.post(
    "/preprocessors/{preprocessor_id}/reset",
    response_model=StatusResponse,
    summary="Reset a preprocessor to its initial state"
)
def reset_preprocessor(preprocessor_id: str):
    """
    Reset a preprocessor to its initial state.
    
    This clears all fitted statistics and state, allowing the preprocessor to be fitted again.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        
    Returns:
        StatusResponse: Status of the reset operation
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    try:
        # Reset the preprocessor
        preprocessor.reset()
        
        return StatusResponse(
            success=True,
            message=f"Preprocessor {preprocessor_id} reset successfully"
        )
    except Exception as e:
        logger.error(f"Error resetting preprocessor: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset preprocessor: {str(e)}"
        )

@app.post(
    "/preprocessors/{preprocessor_id}/update-config",
    response_model=StatusResponse,
    summary="Update preprocessor configuration"
)
def update_preprocessor_config(
    preprocessor_id: str,
    config: PreprocessorConfigRequest
):
    """
    Update the configuration of an existing preprocessor.
    
    This may reset the preprocessor state if the configuration changes affect the preprocessing pipeline.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        config: New configuration
        
    Returns:
        StatusResponse: Status of the update operation
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    try:
        # Create preprocessor config
        config_dict = config.dict()
        preprocessor_config = create_preprocessor_config(config_dict)
        
        # Update the preprocessor configuration
        preprocessor.update_config(preprocessor_config)
        
        return StatusResponse(
            success=True,
            message=f"Preprocessor {preprocessor_id} configuration updated successfully"
        )
    except Exception as e:
        logger.error(f"Error updating preprocessor config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update preprocessor configuration: {str(e)}"
        )

@app.get(
    "/preprocessors/{preprocessor_id}/statistics",
    response_model=StatisticsResponse,
    summary="Get preprocessor statistics"
)
def get_preprocessor_statistics(preprocessor_id: str):
    """
    Get the statistics computed by a fitted preprocessor.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        
    Returns:
        StatisticsResponse: Preprocessor statistics
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    # Check if preprocessor is fitted
    if not preprocessor._fitted:
        raise HTTPException(
            status_code=400,
            detail="Preprocessor is not fitted. Please fit the preprocessor first."
        )
    
    try:
        # Get statistics
        statistics = preprocessor.get_statistics()
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in statistics.items():
            if isinstance(value, np.ndarray):
                statistics[key] = value.tolist()
        
        return StatisticsResponse(
            preprocessor_id=preprocessor_id,
            statistics=statistics
        )
    except Exception as e:
        logger.error(f"Error getting preprocessor statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get preprocessor statistics: {str(e)}"
        )

@app.get(
    "/preprocessors/{preprocessor_id}/metrics",
    response_model=MetricsResponse,
    summary="Get preprocessor performance metrics"
)
def get_preprocessor_metrics(preprocessor_id: str):
    """
    Get performance metrics collected during preprocessing operations.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        
    Returns:
        MetricsResponse: Preprocessor performance metrics
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    try:
        # Get metrics
        metrics = preprocessor.get_performance_metrics()
        
        return MetricsResponse(
            preprocessor_id=preprocessor_id,
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Error getting preprocessor metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get preprocessor metrics: {str(e)}"
        )

@app.post(
    "/preprocessors/{preprocessor_id}/serialize",
    response_model=StatusResponse,
    summary="Save preprocessor state to disk"
)
def serialize_preprocessor(
    preprocessor_id: str,
    filename: Optional[str] = Query(None, description="Custom filename to use (without extension)")
):
    """
    Serialize (save) a preprocessor to disk.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        filename: Custom filename to use (without extension)
        
    Returns:
        StatusResponse: Status of the serialization operation with the path to the saved file
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    try:
        # Generate filename if not provided
        if not filename:
            filename = f"preprocessor_{preprocessor_id}"
        
        # Ensure it has the .pkl extension
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        # Construct the full path
        file_path = os.path.join(MODEL_DIR, filename)
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Serialize the preprocessor
        success = preprocessor.serialize(file_path)
        
        if not success:
            raise SerializationError("Failed to serialize preprocessor")
        
        return StatusResponse(
            success=True,
            message=f"Preprocessor serialized successfully to {file_path}"
        )
    except Exception as e:
        logger.error(f"Error serializing preprocessor: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to serialize preprocessor: {str(e)}"
        )

@app.post(
    "/preprocessors/deserialize",
    response_model=PreprocessorCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Load preprocessor from disk"
)
def deserialize_preprocessor(
    file: UploadFile = File(..., description="Serialized preprocessor file"),
    custom_id: Optional[str] = Query(None, description="Custom ID to assign to the loaded preprocessor")
):
    """
    Deserialize (load) a preprocessor from a file.
    
    Args:
        file: Uploaded serialized preprocessor file
        custom_id: Custom ID to assign to the loaded preprocessor
        
    Returns:
        PreprocessorCreateResponse: Details of the loaded preprocessor
    """
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(file.file.read())
        
        try:
            # Deserialize the preprocessor
            preprocessor = DataPreprocessor.deserialize(temp_file_path)
            
            # Generate or use custom ID
            preprocessor_id = custom_id or str(uuid.uuid4())
            
            # Store the preprocessor
            active_preprocessors[preprocessor_id] = preprocessor
            
            # Convert config to dict correctly, handling the Enum values
            config_dict = {}
            for key, value in asdict(preprocessor.config).items():
                if isinstance(value, Enum):
                    config_dict[key] = value.value
                elif isinstance(value, type):
                    config_dict[key] = str(value.__name__)
                else:
                    config_dict[key] = value
            
            return PreprocessorCreateResponse(
                preprocessor_id=preprocessor_id,
                config=config_dict,
                created_at=datetime.now().isoformat()
            )
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    except Exception as e:
        logger.error(f"Error deserializing preprocessor: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to deserialize preprocessor: {str(e)}"
        )

@app.get(
    "/preprocessors/{preprocessor_id}/download",
    summary="Download serialized preprocessor"
)
def download_preprocessor(preprocessor_id: str):
    """
    Serialize and download a preprocessor.
    
    Args:
        preprocessor_id: Unique ID of the preprocessor
        
    Returns:
        Serialized preprocessor file as a download
    """
    # Get the preprocessor
    preprocessor = get_preprocessor(preprocessor_id)
    
    try:
        # Create a temporary file to save the preprocessor
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_file_path = temp_file.name
        
        # Serialize the preprocessor
        success = preprocessor.serialize(temp_file_path)
        
        if not success:
            raise SerializationError("Failed to serialize preprocessor")
        
        # Create a background tasks object
        background_tasks = BackgroundTasks()
        # Add the cleanup task
        background_tasks.add_task(os.unlink, temp_file_path)
        
        # Read the file into memory as bytes
        with open(temp_file_path, 'rb') as f:
            file_content = f.read()
        
        # Return a streaming response with bytes content
        return StreamingResponse(
            iter([file_content]),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=preprocessor_{preprocessor_id}.pkl"
            },
            background=background_tasks
        )
    except Exception as e:
        logger.error(f"Error downloading preprocessor: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download preprocessor: {str(e)}"
        )

@app.get("/health", response_model=StatusResponse)
def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        StatusResponse: API status
    """
    return StatusResponse(
        success=True,
        message="Data Preprocessor API is operational"
    )

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
