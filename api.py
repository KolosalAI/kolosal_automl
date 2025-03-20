import os
import io
import json
import time
import uuid
import logging
import base64
from contextlib import contextmanager
import numpy as np
import pandas as pd
import threading
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, Query, Header, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timedelta
import secrets
import jwt
import pickle
import uvicorn
from enum import Enum
# Import status codes from starlette
from starlette.status import (
    HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_500_INTERNAL_SERVER_ERROR
)

# Import dotenv for environment variable loading
from dotenv import load_dotenv

# Secure password hashing
import bcrypt

# Import custom modules
from modules.configs import (
    TaskType, 
    OptimizationStrategy,
    MLTrainingEngineConfig,
    PreprocessorConfig,
    InferenceEngineConfig, 
    QuantizationConfig,
    NormalizationType
)
from modules.engine.train_engine import MLTrainingEngine
from modules.engine.inference_engine import InferenceEngine
from modules.engine.data_preprocessor import DataPreprocessor
from modules.engine.quantizer import Quantizer
from modules.model_manager import SecureModelManager

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("ml-api")

# Configuration from environment variables
API_VERSION = os.getenv("API_VERSION", "1.0.0")
MODEL_PATH = os.getenv("MODEL_PATH", "./models")
TEMP_UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
ALLOWED_EXTENSIONS = set(os.getenv("ALLOWED_EXTENSIONS", "csv,txt,json,xlsx,pkl,joblib").split(","))

# Security configuration - fail if not provided in production
if os.getenv("ENVIRONMENT") == "production" and not os.getenv("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY environment variable is required in production")

SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
TOKEN_EXPIRATION = int(os.getenv("TOKEN_EXPIRATION", "86400"))  # 24 hours default

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Admin credentials
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# User credentials
USER_USERNAME = os.getenv("USER_USERNAME", "user")
USER_PASSWORD = os.getenv("USER_PASSWORD", "user123")

# Ensure directories exist
Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(TEMP_UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Machine Learning API",
    description="API for machine learning model training, inference, and management",
    version=API_VERSION
)

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OAuth2 password bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

# Thread-safe lazy initialization
class LazyInitializer:
    _lock = threading.Lock()
    _instances = {}
    
    @classmethod
    def get_instance(cls, name, initializer):
        if name not in cls._instances:
            with cls._lock:
                if name not in cls._instances:
                    cls._instances[name] = initializer()
        return cls._instances[name]

# User management (in a real app, you'd use a database)
# Use bcrypt for password hashing instead of SHA-256
USERS = {
    ADMIN_USERNAME: {
        "password": bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode(),
        "roles": ["admin"]
    },
    USER_USERNAME: {
        "password": bcrypt.hashpw(USER_PASSWORD.encode(), bcrypt.gensalt()).decode(),
        "roles": ["user"]
    }
}

# The rest of the code remains the same...
# Pydantic Data Models, Authentication/Authorization, Service Initialization, etc.

# -------------------------------------------------------------------
# Pydantic Data Models
# -------------------------------------------------------------------

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    username: str
    roles: List[str]

class TokenResponse(BaseModel):
    token: str
    username: str
    roles: List[str]
    expires_in: int

class UserCreate(BaseModel):
    username: str
    password: str
    roles: List[str] = ["user"]
    
    @field_validator('username')
    @classmethod
    def username_must_be_valid(cls, v):
        if not v or len(v) < 3 or not v.isalnum():
            raise ValueError('Username must be at least 3 alphanumeric characters')
        return v
    
    @field_validator('password')
    @classmethod
    def password_must_be_strong(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class ModelInfo(BaseModel):
    name: str
    path: str
    size: Optional[int] = None
    modified: Optional[str] = None

class ModelsList(BaseModel):
    models: List[ModelInfo]
    count: int

class ModelMetrics(BaseModel):
    model_name: str
    metrics: Dict[str, Any]

class PredictionData(BaseModel):
    model: str
    data: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[Any]
    model: str
    sample_count: int
    processing_time_ms: int
    metadata: Dict[str, Any]

class AccessCode(BaseModel):
    access_code: str

class DriftResult(BaseModel):
    dataset_drift: float
    drift_detected: bool
    drifted_features_count: int
    total_features: int
    drift_threshold: float
    drifted_features: List[str]
    timestamp: str
    drift_plot_base64: Optional[str] = None
    distribution_plot_base64: Optional[str] = None

class QuantizationOptions(BaseModel):
    quantization_type: str = "int8"
    quantization_mode: str = "dynamic_per_batch"

class QuantizationResult(BaseModel):
    message: str
    original_size_bytes: int
    quantized_size_bytes: int
    compression_ratio: float
    quantized_path: str

class ModelComparison(BaseModel):
    models: List[str]
    metrics: Optional[List[str]] = None

# -------------------------------------------------------------------
# Authentication and Authorization
# -------------------------------------------------------------------

def generate_token(username: str, roles: List[str]) -> str:
    """Generate a JWT token for authenticated users"""
    expiration = datetime.utcnow() + timedelta(seconds=TOKEN_EXPIRATION)
    payload = {
        "sub": username,
        "roles": roles,
        "exp": expiration,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Authenticate and get the current user based on JWT token.
    Used as a dependency for protected endpoints.
    """
    credentials_exception = HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")
        roles = payload.get("roles", [])
        
        if username is None:
            raise credentials_exception
            
        # Check if user still exists
        if username not in USERS:
            raise credentials_exception
        
        return {"username": username, "roles": roles}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise credentials_exception

async def get_admin_user(current_user: dict = Depends(get_current_user)):
    """
    Check if the current user has admin role.
    Used as a dependency for admin-only endpoints.
    """
    if "admin" not in current_user["roles"]:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this resource"
        )
    return current_user

# -------------------------------------------------------------------
# Service Initialization
# -------------------------------------------------------------------

def init_training_engine():
    """Initialize the training engine"""
    config = MLTrainingEngineConfig(
        task_type=TaskType.CLASSIFICATION,  # Default, can be changed via API
        model_path=MODEL_PATH,
        experiment_tracking=True,
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )
    engine = MLTrainingEngine(config)
    logger.info("Training engine initialized")
    return engine

def init_inference_engine():
    """Initialize the inference engine"""
    config = InferenceEngineConfig(
        model_version=os.getenv("MODEL_VERSION", "1.0"),
        enable_batching=os.getenv("ENABLE_BATCHING", "true").lower() == "true",
        enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() == "true",
        enable_quantization=os.getenv("ENABLE_QUANTIZATION", "true").lower() == "true"
    )
    engine = InferenceEngine(config)
    logger.info("Inference engine initialized")
    return engine

def init_secure_manager():
    """Initialize the secure model manager"""
    # Create a minimal config for the secure manager
    class SecureConfig:
        def __init__(self):
            self.model_path = MODEL_PATH
            self.task_type = TaskType.CLASSIFICATION
            self.enable_encryption = os.getenv("ENABLE_ENCRYPTION", "true").lower() == "true"
    
    manager = SecureModelManager(SecureConfig(), logger=logger)
    logger.info("Secure model manager initialized")
    return manager

def init_preprocessor():
    """Initialize the data preprocessor"""
    config = PreprocessorConfig()
    processor = DataPreprocessor(config)
    logger.info("Data preprocessor initialized")
    return processor

def init_quantizer():
    """Initialize the quantizer"""
    config = QuantizationConfig()
    quantizer = Quantizer(config)
    logger.info("Quantizer initialized")
    return quantizer

# Thread-safe service getters
def get_training_engine():
    return LazyInitializer.get_instance("training_engine", init_training_engine)

def get_inference_engine():
    return LazyInitializer.get_instance("inference_engine", init_inference_engine)

def get_secure_manager():
    return LazyInitializer.get_instance("secure_manager", init_secure_manager)

def get_preprocessor():
    return LazyInitializer.get_instance("preprocessor", init_preprocessor)

def get_quantizer():
    return LazyInitializer.get_instance("quantizer", init_quantizer)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def validate_model_name(model_name: str) -> str:
    """Validate model name to prevent directory traversal and other issues"""
    if not model_name or not model_name.isalnum() and not all(c in "-_." for c in model_name if not c.isalnum()):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Invalid model name. Only alphanumeric characters, dash, underscore, and period allowed."
        )
    return model_name

def allowed_file(filename: str) -> bool:
    """Check if a file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@contextmanager
def temp_file_manager(upload_file: UploadFile):
    """Context manager for handling temporary file uploads"""
    temp_path = None
    try:
        # Generate a secure random filename to prevent path traversal
        file_ext = upload_file.filename.rsplit('.', 1)[1].lower() if '.' in upload_file.filename else ''
        secure_filename = f"{uuid.uuid4().hex}.{file_ext}"
        temp_path = os.path.join(TEMP_UPLOAD_FOLDER, secure_filename)
        
        # Save file
        with open(temp_path, "wb") as f:
            content = upload_file.file.read()
            f.write(content)
        
        # Reset file pointer for potential further reads
        upload_file.file.seek(0)
        
        yield temp_path
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_path}: {str(e)}")

async def parse_data(file_path: str) -> pd.DataFrame:
    """Parse uploaded data file into a DataFrame"""
    file_ext = file_path.rsplit('.', 1)[1].lower()
    
    try:
        if file_ext == 'csv':
            return pd.read_csv(file_path)
        elif file_ext == 'xlsx':
            return pd.read_excel(file_path)
        elif file_ext == 'json':
            return pd.read_json(file_path)
        elif file_ext in ['pkl', 'joblib']:
            import joblib
            return joblib.load(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {str(e)}")
        raise

def get_model_path(model_name: str) -> str:
    """Get validated model path"""
    # Validate model name
    clean_name = validate_model_name(model_name)
    
    # Construct path without using string interpolation
    model_path = os.path.join(MODEL_PATH, f"{clean_name}.pkl")
    
    # Verify the path is within the models directory (prevent path traversal)
    if not os.path.abspath(model_path).startswith(os.path.abspath(MODEL_PATH)):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Invalid model path"
        )
    
    return model_path

def check_model_exists(model_path: str):
    """Check if a model file exists, raise HTTP 404 if not"""
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Model not found at {os.path.basename(model_path)}"
        )

# -------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------

@app.get("/api/status", tags=["Utility"])
async def status():
    """Get API status and version information"""
    return {
        "status": "online",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/login", response_model=TokenResponse, tags=["Authentication"])
async def login(user: UserLogin):
    """Authenticate user and issue JWT token"""
    if user.username not in USERS:
        # Use a consistent error message to prevent username enumeration
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    stored_password = USERS[user.username]['password']
    
    # Verify password using bcrypt
    if not bcrypt.checkpw(user.password.encode(), stored_password.encode()):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Generate token
    token = generate_token(user.username, USERS[user.username]['roles'])
    
    return {
        "token": token,
        "username": user.username,
        "roles": USERS[user.username]['roles'],
        "expires_in": TOKEN_EXPIRATION
    }

@app.get("/api/models", response_model=ModelsList, tags=["Model Management"])
async def list_models(current_user: dict = Depends(get_current_user)):
    """List all available models"""
    models = []
    
    # Get model directory
    model_dir = Path(MODEL_PATH)
    
    # List all model files
    for file_path in model_dir.glob("*.pkl"):
        if not file_path.name.startswith('.'):
            models.append({
                "name": file_path.stem,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    return {
        "models": models,
        "count": len(models)
    }

@app.get("/api/models/{model_name}", tags=["Model Management"])
async def get_model_info(model_name: str, current_user: dict = Depends(get_current_user)):
    """Get detailed information about a specific model"""
    # Get validated model path
    model_path = get_model_path(model_name)
    
    # Check if the model exists
    check_model_exists(model_path)
    
    # Try to load minimal model metadata without full deserialization
    try:
        engine = get_inference_engine()
        metadata = {"name": model_name, "path": model_path}
        
        # Try loading metadata without loading full model
        if hasattr(engine, 'get_model_info'):
            engine.load_model(model_path)
            model_info = engine.get_model_info()
            metadata.update(model_info)
        
        return metadata
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {
            "name": model_name,
            "path": model_path,
            "error": "Could not load detailed model information",
            "size": os.path.getsize(model_path),
            "modified": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
        }

@app.delete("/api/models/{model_name}", tags=["Model Management"])
async def delete_model(model_name: str, current_user: dict = Depends(get_admin_user)):
    """Delete a model from storage (admin only)"""
    # Get validated model path
    model_path = get_model_path(model_name)
    
    # Check if the model exists
    check_model_exists(model_path)
    
    try:
        # Delete the model file
        os.remove(model_path)
        
        # Also delete any associated files (quantized version, metadata, etc.)
        quantized_path = os.path.join(MODEL_PATH, f"{model_name}_quantized.pkl")
        if os.path.exists(quantized_path):
            os.remove(quantized_path)
            
        # Return success
        return {
            "message": f"Model {model_name} successfully deleted",
            "name": model_name
        }
    except Exception as e:
        logger.error(f"Error deleting model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )

@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    model: str = Query(None, description="Model name to use for prediction"),
    batch_size: int = Query(0, description="Batch size for large datasets (0 means direct prediction)"),
    file: Optional[UploadFile] = File(None, description="File containing data for prediction"),
    data: Optional[PredictionData] = None,
    current_user: dict = Depends(get_current_user)
):
    """Make predictions using a loaded model"""
    # Check if data is provided
    if file is None and data is None:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="No data provided. Send JSON data or file."
        )
    
    # Get model name if not provided as query param
    model_name = model
    if not model_name and data is not None:
        model_name = data.model
    
    if not model_name:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Model name not specified"
        )
    
    # Validate model name and get path
    model_path = get_model_path(model_name)
    check_model_exists(model_path)
    
    # Handle JSON input data
    if data is not None:
        try:
            input_data = data.data
            
            if not input_data:
                raise HTTPException(
                    status_code=HTTP_400_BAD_REQUEST,
                    detail="No input data provided"
                )
                
            # Convert to numpy array
            features = np.array(input_data)
        except Exception as e:
            logger.error(f"Error processing JSON input: {str(e)}")
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Error processing input data: {str(e)}"
            )
    
    # Handle file upload
    elif file is not None:
        if file.filename == '':
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="No file selected"
            )
            
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
            
        # Use context manager for temporary file handling
        with temp_file_manager(file) as temp_path:
            try:
                # Parse the file
                features = await parse_data(temp_path)
            except Exception as e:
                logger.error(f"Error processing file upload: {str(e)}")
                raise HTTPException(
                    status_code=HTTP_400_BAD_REQUEST,
                    detail=f"Error processing uploaded file: {str(e)}"
                )
    else:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="No data provided"
        )
    
    # Make predictions
    try:
        # Load the model and make prediction
        engine = get_inference_engine()
            
        # Load the model
        success = engine.load_model(model_path)
        if not success:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model {model_name}"
            )
        
        # Make prediction based on options
        start_time = time.time()
        if batch_size > 0:
            # Use batch prediction for large datasets
            try:
                future = engine.predict_batch(features, batch_size=batch_size)
                predictions, metadata = future.result()
            except Exception as e:
                raise HTTPException(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Batch prediction failed: {str(e)}"
                )
        else:
            # Use direct prediction
            success, predictions, metadata = engine.predict(features)
            
            if not success:
                raise HTTPException(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Prediction failed: {metadata}"
                )
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Format the results
        if isinstance(predictions, np.ndarray):
            predictions_list = predictions.tolist()
        else:
            predictions_list = predictions
            
        # Return predictions with metadata
        return {
            "predictions": predictions_list,
            "model": model_name,
            "sample_count": len(features),
            "processing_time_ms": int(total_time * 1000),
            "metadata": metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

# Rest of the code remains the same...
# Continue with train_model, get_model_metrics, preprocess_data endpoints, etc.

# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)