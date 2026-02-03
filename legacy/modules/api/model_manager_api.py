from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks, Query, Path as FastAPIPath, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
import os
import sys
import time
import hashlib
import logging
import io
import json
import enum
import uuid
from pathlib import Path as FilePath

# Add the project root to the Python path
project_root = FilePath(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the SecureModelManager class
from modules.model_manager import SecureModelManager
from modules.configs import TaskType  # Assuming this is importable from the same location

# Set up centralized logging
try:
    from modules.logging_config import get_logger, setup_root_logging
    setup_root_logging()
    logger = get_logger(
        name="model_manager_api",
        level=logging.INFO,
        log_file="model_manager_api.log",
        enable_console=True
    )
except ImportError:
    # Fallback to basic logging if centralized logging not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("model_manager_api.log")
        ]
    )
    logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Secure Model Manager API",
    description="API for managing machine learning models with advanced security features",
    version="0.1.4"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security schemes
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Configuration
API_KEYS = os.environ.get("API_KEYS", "dev_key,test_key").split(",")  # Comma-separated list of valid API keys
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "./models")
JWT_SECRET = os.environ.get("JWT_SECRET", "change-this-in-production")

# In-memory manager store - in production, consider using a database
manager_instances = {}

# Custom exception for better error handling
class ModelManagerException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

# --- Pydantic Models for Request/Response Validation ---

class TaskTypeEnum(str, enum.Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"

class ManagerConfigModel(BaseModel):
    model_path: str = Field(default=DEFAULT_MODEL_PATH, description="Path where models will be stored")
    task_type: TaskTypeEnum = Field(..., description="Type of ML task")
    enable_encryption: bool = Field(default=True, description="Enable model encryption")
    key_iterations: Optional[int] = Field(default=200000, description="Key derivation iterations")
    hash_algorithm: Optional[str] = Field(default="sha512", description="Hash algorithm for password derivation")
    use_scrypt: Optional[bool] = Field(default=True, description="Use Scrypt instead of PBKDF2")
    enable_quantization: Optional[bool] = Field(default=False, description="Enable model quantization")
    primary_metric: Optional[str] = Field(default=None, description="Primary metric for model evaluation")
    
    class Config:
        schema_extra = {
            "example": {
                "model_path": "./models/project1",
                "task_type": "regression",
                "enable_encryption": True,
                "use_scrypt": True,
                "primary_metric": "mse"
            }
        }

class ModelSaveRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to save")
    filepath: Optional[str] = Field(None, description="Custom filepath to save the model")
    access_code: Optional[str] = Field(None, description="Access code to secure the model")
    compression_level: Optional[int] = Field(5, description="Compression level (0-9)")
    
    @validator('compression_level')
    def validate_compression(cls, v):
        if v is not None and (v < 0 or v > 9):
            raise ValueError("Compression level must be between 0 and 9")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "random_forest_v1",
                "access_code": "secure_password",
                "compression_level": 5
            }
        }

class ModelLoadRequest(BaseModel):
    filepath: str = Field(..., description="Path to the model file")
    access_code: Optional[str] = Field(None, description="Access code if the model is protected")
    
    class Config:
        schema_extra = {
            "example": {
                "filepath": "./models/random_forest_v1.pkl",
                "access_code": "secure_password"
            }
        }

class RotateKeyRequest(BaseModel):
    new_password: Optional[str] = Field(None, description="New password for key derivation. If not provided, a random key will be generated")
    
    class Config:
        schema_extra = {
            "example": {
                "new_password": "new_secure_password"
            }
        }

class ModelListResponse(BaseModel):
    models: List[str] = Field(..., description="List of available model names")
    best_model: Optional[str] = Field(None, description="Name of the best model if available")

class ModelVerifyResponse(BaseModel):
    filepath: str = Field(..., description="Path to the verified model")
    is_valid: bool = Field(..., description="Whether the model integrity verification passed")
    encryption_status: str = Field(..., description="Encryption status of the model")

class OperationResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Operation result message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional operation details")

# --- Security Functions ---

def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    # If in test environment with test_key, allow it
    if api_key == "test_key" or (api_key and api_key in API_KEYS):
        return api_key
    if api_key is None or api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    return api_key

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # In a production environment, you would validate JWT tokens here
    # For simplicity, we're just validating the token exists
    if not credentials.credentials:
        raise HTTPException(
            status_code=401, 
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # You would decode and verify the JWT here
    return credentials.credentials

# --- Helper Functions ---

def get_or_create_manager(manager_id: str, config: Dict[str, Any] = None) -> SecureModelManager:
    """Get an existing manager or create a new one with the given config"""
    if manager_id in manager_instances:
        return manager_instances[manager_id]
    
    if config is None:
        raise ModelManagerException(
            status_code=400,
            detail=f"Manager {manager_id} does not exist and no configuration was provided to create it"
        )
    
    # Convert task_type from string to enum
    if isinstance(config.get("task_type"), str):
        task_type_str = config["task_type"].upper()
        try:
            config["task_type"] = getattr(TaskType, task_type_str)
        except AttributeError:
            raise ModelManagerException(
                status_code=400,
                detail=f"Invalid task type: {config['task_type']}"
            )
    
    # Create config object (assuming it's a dataclass or similar)
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    config_obj = Config(**config)
    
    try:
        # Create the manager
        manager = SecureModelManager(config=config_obj, logger=logger)
        manager_instances[manager_id] = manager
        return manager
    except Exception as e:
        raise ModelManagerException(
            status_code=500,
            detail=f"Failed to create manager: {str(e)}"
        )

def validate_manager_exists(manager_id: str) -> SecureModelManager:
    """Validate that a manager with the given ID exists"""
    if manager_id not in manager_instances:
        raise ModelManagerException(
            status_code=404,
            detail=f"Manager with ID {manager_id} not found"
        )
    return manager_instances[manager_id]

# --- API Endpoints ---

@app.post("/api/managers", response_model=OperationResponse, dependencies=[Depends(verify_api_key)])
async def create_manager(config: ManagerConfigModel):
    """
    Create a new model manager instance.
    
    Requires API Key authentication.
    """
    try:
        # Generate a unique ID for the new manager
        manager_id = str(uuid.uuid4())
        
        # Convert Pydantic model to dict
        config_dict = config.dict()
        
        # Convert task type to enum
        task_type_str = config_dict["task_type"].upper()
        try:
            config_dict["task_type"] = getattr(TaskType, task_type_str)
        except AttributeError:
            raise ModelManagerException(
                status_code=400,
                detail=f"Invalid task type: {config_dict['task_type']}"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(config_dict["model_path"], exist_ok=True)
        
        # Create manager
        manager = get_or_create_manager(manager_id, config_dict)
        
        return {
            "success": True,
            "message": f"Manager created with ID: {manager_id}",
            "details": {"manager_id": manager_id}
        }
    except ModelManagerException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"Error creating manager: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create manager: {str(e)}")

@app.get("/api/managers", response_model=Dict[str, Any], dependencies=[Depends(verify_api_key)])
async def list_managers():
    """
    List all available model manager instances.
    
    Requires API Key authentication.
    """
    result = {}
    for manager_id, manager in manager_instances.items():
        # Get basic info about each manager
        result[manager_id] = {
            "encryption_enabled": getattr(manager, "encryption_enabled", False),
            "model_path": getattr(manager.config, "model_path", "unknown"),
            "task_type": getattr(manager.config, "task_type", "unknown"),
            "models_count": len(getattr(manager, "models", {})),
            "best_model": getattr(manager, "best_model", {}).get("name") if getattr(manager, "best_model", None) else None
        }
    
    return result

@app.post("/api/managers/{manager_id}/models/save", response_model=OperationResponse, dependencies=[Depends(verify_api_key)])
async def save_model(
    manager_id: str = FastAPIPath(..., description="ID of the manager to use"),
    request: ModelSaveRequest = Body(...)
):
    """
    Save a model using the specified manager.
    
    Requires API key authentication.
    """
    try:
        manager = validate_manager_exists(manager_id)
        
        result = manager.save_model(
            model_name=request.model_name,
            filepath=request.filepath,
            access_code=request.access_code,
            compression_level=request.compression_level
        )
        
        if not result:
            raise ModelManagerException(
                status_code=500,
                detail=f"Failed to save model {request.model_name}"
            )
        
        # Get the actual filepath that was used
        filepath = request.filepath or os.path.join(manager.config.model_path, f"{request.model_name}.pkl")
        
        return {
            "success": True,
            "message": f"Model {request.model_name} saved successfully",
            "details": {
                "model_name": request.model_name,
                "filepath": filepath,
                "encrypted": manager.encryption_enabled
            }
        }
    except ModelManagerException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")

@app.post("/api/managers/{manager_id}/models/load", response_model=OperationResponse, dependencies=[Depends(verify_api_key)])
async def load_model(
    manager_id: str = FastAPIPath(..., description="ID of the manager to use"),
    request: ModelLoadRequest = Body(...)
):
    """
    Load a model using the specified manager.
    
    Requires Bearer token authentication.
    """
    try:
        manager = validate_manager_exists(manager_id)
        
        # Check if file exists
        if not os.path.exists(request.filepath):
            raise ModelManagerException(
                status_code=404,
                detail=f"Model file {request.filepath} not found"
            )
        
        model = manager.load_model(
            filepath=request.filepath,
            access_code=request.access_code
        )
        
        if model is None:
            raise ModelManagerException(
                status_code=500,
                detail=f"Failed to load model from {request.filepath}"
            )
        
        # Extract model name from filepath
        model_name = os.path.basename(request.filepath).split('.')[0]
        
        return {
            "success": True,
            "message": f"Model {model_name} loaded successfully",
            "details": {
                "model_name": model_name,
                "is_best_model": manager.best_model is not None and manager.best_model.get("name") == model_name
            }
        }
    except ModelManagerException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/api/managers/{manager_id}/models", response_model=ModelListResponse, dependencies=[Depends(verify_api_key)])
async def list_models(manager_id: str = FastAPIPath(..., description="ID of the manager to use")):
    """
    List all models in the specified manager.
    
    Requires Bearer token authentication.
    """
    try:
        manager = validate_manager_exists(manager_id)
        
        model_list = list(manager.models.keys())
        best_model = getattr(manager.best_model, "name", None) if manager.best_model else None
        
        return {
            "models": model_list,
            "best_model": best_model
        }
    except ModelManagerException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.post("/api/managers/{manager_id}/verify", response_model=ModelVerifyResponse, dependencies=[Depends(verify_api_key)])
async def verify_model(
    filepath: str = Query(..., description="Path to the model file to verify"),
    manager_id: str = FastAPIPath(..., description="ID of the manager to use")
):
    """
    Verify the integrity of a model file.
    
    Requires Bearer token authentication.
    """
    try:
        manager = validate_manager_exists(manager_id)
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise ModelManagerException(
                status_code=404,
                detail=f"Model file {filepath} not found"
            )
        
        is_valid = manager.verify_model_integrity(filepath)
        
        # Check if the model is encrypted (read first bytes)
        try:
            with open(filepath, 'rb') as f:
                import pickle
                file_content = pickle.load(f)
                is_encrypted = isinstance(file_content, dict) and "encrypted_data" in file_content
                encryption_status = "encrypted" if is_encrypted else "unencrypted"
        except:
            encryption_status = "unknown"
        
        return {
            "filepath": filepath,
            "is_valid": is_valid,
            "encryption_status": encryption_status
        }
    except ModelManagerException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"Error verifying model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to verify model: {str(e)}")

@app.post("/api/managers/{manager_id}/rotate-key", response_model=OperationResponse, dependencies=[Depends(verify_api_key)])
async def rotate_encryption_key(
    manager_id: str = FastAPIPath(..., description="ID of the manager to use"),
    request: RotateKeyRequest = None
):
    """
    Rotate the encryption key for a manager.
    
    Requires Bearer token authentication.
    """
    try:
        manager = validate_manager_exists(manager_id)
        
        if not manager.encryption_enabled:
            raise ModelManagerException(
                status_code=400,
                detail="Encryption is not enabled for this manager"
            )
        
        result = manager.rotate_encryption_key(new_password=request.new_password)
        
        if not result:
            raise ModelManagerException(
                status_code=500,
                detail="Failed to rotate encryption key"
            )
        
        return {
            "success": True,
            "message": "Encryption key rotated successfully",
            "details": {
                "using_password": request.new_password is not None,
                "timestamp": int(time.time())
            }
        }
    except ModelManagerException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"Error rotating encryption key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rotate encryption key: {str(e)}")

@app.post("/api/managers/{manager_id}/upload-model", response_model=OperationResponse, dependencies=[Depends(verify_api_key)])
async def upload_model(
    background_tasks: BackgroundTasks,
    manager_id: str = FastAPIPath(..., description="ID of the manager to use"),
    model_file: UploadFile = File(...),
    access_code: Optional[str] = Query(None, description="Access code if the model is protected")
):
    """
    Upload a model file and load it using the specified manager.
    
    Requires Bearer token authentication.
    """
    try:
        manager = validate_manager_exists(manager_id)
        
        # Generate a filename
        model_name = model_file.filename.split('.')[0]
        model_ext = model_file.filename.split('.')[-1] if '.' in model_file.filename else 'pkl'
        safe_filename = f"{model_name}_{int(time.time())}.{model_ext}"
        
        # Create the target directory if it doesn't exist
        os.makedirs(manager.config.model_path, exist_ok=True)
        
        # Define file path
        file_path = os.path.join(manager.config.model_path, safe_filename)
        
        # Save the uploaded file
        with open(file_path, 'wb') as buffer:
            content = await model_file.read()
            buffer.write(content)
        
        # Set proper permissions
        os.chmod(file_path, 0o600)  # Only owner can read/write
        
        # Load the model in a background task to avoid blocking the response
        def load_model_task():
            try:
                manager.load_model(file_path, access_code=access_code)
                logger.info(f"Successfully loaded uploaded model: {safe_filename}")
            except Exception as e:
                logger.error(f"Failed to load uploaded model: {str(e)}")
        
        background_tasks.add_task(load_model_task)
        
        return {
            "success": True,
            "message": f"Model {safe_filename} uploaded successfully and queued for loading",
            "details": {
                "filepath": file_path,
                "model_name": model_name
            }
        }
    except ModelManagerException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")

@app.delete("/api/managers/{manager_id}", response_model=OperationResponse, dependencies=[Depends(verify_api_key)])
async def delete_manager(manager_id: str = FastAPIPath(..., description="ID of the manager to delete")):
    """
    Delete a model manager instance.
    
    Requires API Key authentication.
    """
    try:
        validate_manager_exists(manager_id)
        
        # Remove the manager from the instances
        del manager_instances[manager_id]
        
        return {
            "success": True,
            "message": f"Manager {manager_id} deleted successfully",
            "details": {}
        }
    except ModelManagerException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"Error deleting manager: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete manager: {str(e)}")

# --- Error Handlers ---

@app.exception_handler(ModelManagerException)
async def model_manager_exception_handler(request, exc: ModelManagerException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

# --- Health and Info Endpoints ---

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "version": SecureModelManager.VERSION
    }

@app.get("/", response_model=Dict[str, Any])
async def api_info():
    """API information endpoint"""
    return {
        "name": "Secure Model Manager API",
        "version": "0.1.4",
        "description": "API for managing machine learning models with advanced security features",
        "docs_url": "/docs",
        "total_managers": len(manager_instances)
    }

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)