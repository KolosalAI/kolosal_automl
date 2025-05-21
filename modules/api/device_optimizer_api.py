"""
CPU Device Optimizer API

This API provides RESTful endpoints to utilize the DeviceOptimizer for hardware detection,
configuration generation, and optimization based on system capabilities.
"""
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Query, Path
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import Dict, Any, Optional, Union, List, Set
import json
import os
import shutil
import datetime
import uuid
import logging
from pathlib import Path as FilePath

# Import the DeviceOptimizer and related classes
from modules.device_optimizer import (
    DeviceOptimizer, get_system_information,
    optimize_for_environment, optimize_for_workload,
    apply_configs_to_pipeline, get_default_config
)
# Import classes from configs module rather than from device_optimizer
from modules.configs import (
    OptimizationMode, QuantizationType, QuantizationMode,
    QuantizationConfig, BatchProcessorConfig, BatchProcessingStrategy,
    PreprocessorConfig, NormalizationType,
    InferenceEngineConfig, MLTrainingEngineConfig, TaskType, 
    OptimizationStrategy as TrainingOptimizationStrategy,
    ModelSelectionCriteria, AutoMLMode, ExplainabilityConfig, MonitoringConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cpu_device_optimizer_api")

# ------------------ Pydantic models for request/response validation ------------------

class Environment(str, Enum):
    AUTO = "auto"
    CLOUD = "cloud"
    DESKTOP = "desktop"
    EDGE = "edge"

class WorkloadType(str, Enum):
    MIXED = "mixed"
    INFERENCE = "inference"
    TRAINING = "training"

class OptimizerRequest(BaseModel):
    """Base request model for the optimizer API"""
    config_path: str = Field(default="./configs", description="Path to save configuration files")
    checkpoint_path: str = Field(default="./checkpoints", description="Path for model checkpoints")
    model_registry_path: str = Field(default="./model_registry", description="Path for model registry")
    optimization_mode: str = Field(default="BALANCED", description="Mode for optimization strategy")
    workload_type: WorkloadType = Field(default=WorkloadType.MIXED, description="Type of workload")
    environment: Environment = Field(default=Environment.AUTO, description="Computing environment")
    enable_specialized_accelerators: bool = Field(default=True, description="Whether to enable detection of specialized hardware")
    memory_reservation_percent: float = Field(default=10.0, description="Percentage of memory to reserve for the system", ge=0, le=50)
    power_efficiency: bool = Field(default=False, description="Whether to optimize for power efficiency")
    resilience_level: int = Field(default=1, description="Level of fault tolerance", ge=0, le=3)
    auto_tune: bool = Field(default=True, description="Whether to enable automatic parameter tuning")
    config_id: Optional[str] = Field(default=None, description="Optional identifier for the configuration set")
    debug_mode: bool = Field(default=False, description="Enable debug mode for more verbose logging")
    
    @validator('optimization_mode')
    def validate_optimization_mode(cls, v):
        try:
            return OptimizationMode[v].name
        except KeyError:
            valid_modes = [mode.name for mode in OptimizationMode]
            raise ValueError(f"Invalid optimization mode: {v}. Must be one of {valid_modes}")
        return v

class SystemInfoRequest(BaseModel):
    """Request model for system information endpoint"""
    enable_specialized_accelerators: bool = Field(default=True, description="Whether to detect specialized hardware")

class LoadConfigRequest(BaseModel):
    """Request model for loading configurations"""
    config_path: str = Field(description="Path where configuration files are stored")
    config_id: str = Field(description="Identifier for the configuration set")

class ApplyConfigRequest(BaseModel):
    """Request model for applying configurations to a pipeline"""
    configs: Dict[str, Any] = Field(description="Dictionary with configurations to apply")

class DefaultConfigRequest(BaseModel):
    """Request model for getting default configurations"""
    optimization_mode: str = Field(default="BALANCED", description="The optimization strategy to use")
    workload_type: WorkloadType = Field(default=WorkloadType.MIXED, description="Type of workload to optimize for")
    environment: Environment = Field(default=Environment.AUTO, description="Computing environment")
    output_dir: str = Field(default="./configs/default", description="Directory where configuration files will be saved")
    enable_specialized_accelerators: bool = Field(default=True, description="Whether to enable detection of specialized hardware")
    
    @validator('optimization_mode')
    def validate_optimization_mode(cls, v):
        try:
            return OptimizationMode[v].name
        except KeyError:
            valid_modes = [mode.name for mode in OptimizationMode]
            raise ValueError(f"Invalid optimization mode: {v}. Must be one of {valid_modes}")
        return v

class ConfigListResponse(BaseModel):
    """Response model for listing available configurations"""
    configs: List[Dict[str, Any]] = Field(description="List of available configurations")

# ------------------ FastAPI Application Setup ------------------

app = FastAPI(
    title="CPU Device Optimizer API",
    description="""
    This API provides access to the CPU Device Optimizer functionality for hardware detection, 
    configuration generation, and optimization based on system capabilities.
    
    Features:
    - System information detection
    - Configuration generation for different ML pipeline components
    - Configuration saving and loading
    - Creating optimized configs for different modes and environments
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Simple API key authentication - for production use a more robust method
API_KEY = os.environ.get("API_KEY", "dev_key_12345")

def verify_api_key(api_key: str = Query(..., description="API Key for authentication")):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# ------------------ API Endpoints ------------------

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "api": "CPU Device Optimizer API",
        "version": "1.0.0",
        "description": "API for hardware detection, configuration generation, and optimization"
    }

@app.get("/system-info", tags=["System Information"])
async def get_system_info(
    request: SystemInfoRequest = Depends(),
    api_key: str = Depends(verify_api_key)
):
    """
    Get comprehensive information about the current system.
    
    This endpoint detects hardware capabilities, memory information, and other
    system details to provide a complete overview of the current environment.
    """
    try:
        # Create a temporary DeviceOptimizer to get system info
        optimizer = DeviceOptimizer(
            enable_specialized_accelerators=request.enable_specialized_accelerators
        )
        system_info = optimizer.get_system_info()
        return JSONResponse(content=system_info)
    except Exception as e:
        logger.error(f"Error getting system information: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system information: {str(e)}"
        )

@app.post("/optimize", tags=["Configuration Generation"])
async def create_optimized_configurations(
    request: OptimizerRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate and save optimized configurations based on device capabilities.
    
    This endpoint creates configurations for all ML pipeline components 
    (quantization, batch processing, preprocessing, inference, and training)
    optimized for the current system.
    """
    try:
        # Convert string to enum
        opt_mode = OptimizationMode[request.optimization_mode]
        
        # Create DeviceOptimizer with all parameters
        optimizer = DeviceOptimizer(
            config_path=request.config_path,
            checkpoint_path=request.checkpoint_path,
            model_registry_path=request.model_registry_path,
            optimization_mode=opt_mode,
            workload_type=request.workload_type.value,
            environment=request.environment.value,
            enable_specialized_accelerators=request.enable_specialized_accelerators,
            memory_reservation_percent=request.memory_reservation_percent,
            power_efficiency=request.power_efficiency,
            resilience_level=request.resilience_level,
            auto_tune=request.auto_tune,
            debug_mode=request.debug_mode
        )
        
        # Generate optimized configurations
        master_config = optimizer.save_configs(config_id=request.config_id)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Optimized configurations generated successfully",
            "config_id": master_config.get("config_id", request.config_id),
            "master_config": master_config
        })
    except Exception as e:
        logger.error(f"Error creating optimized configurations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create optimized configurations: {str(e)}"
        )

@app.post("/optimize/all-modes", tags=["Configuration Generation"])
async def create_all_mode_configurations(
    request: OptimizerRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate and save configurations for all optimization modes.
    
    This endpoint creates separate configurations for each optimization mode
    (BALANCED, PERFORMANCE, MEMORY_SAVING, etc.) optimized for the current system.
    """
    try:
        # Initialize DeviceOptimizer with parameters from request
        optimizer = DeviceOptimizer(
            config_path=request.config_path,
            checkpoint_path=request.checkpoint_path,
            model_registry_path=request.model_registry_path,
            workload_type=request.workload_type.value,
            environment=request.environment.value,
            enable_specialized_accelerators=request.enable_specialized_accelerators,
            memory_reservation_percent=request.memory_reservation_percent,
            power_efficiency=request.power_efficiency,
            resilience_level=request.resilience_level,
            auto_tune=request.auto_tune,
            debug_mode=request.debug_mode
        )
        
        # Generate configurations for all modes
        all_configs = optimizer.create_configs_for_all_modes()
        
        return JSONResponse(content={
            "status": "success",
            "message": "Configurations for all optimization modes generated successfully",
            "configs": all_configs
        })
    except Exception as e:
        logger.error(f"Error creating configurations for all modes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create configurations for all modes: {str(e)}"
        )

@app.post("/optimize/environment/{environment}", tags=["Configuration Generation"])
async def optimize_for_specific_environment(
    environment: str = Path(..., description="Target environment (cloud, desktop, edge)"),
    api_key: str = Depends(verify_api_key)
):
    """
    Create optimized configurations specifically for a given environment type.
    
    This endpoint generates configurations optimized for a specific environment:
    - cloud: High-performance settings for cloud environments
    - desktop: Balanced settings for desktop environments
    - edge: Memory-saving settings for edge devices
    """
    try:
        # Validate environment
        if environment not in ["cloud", "desktop", "edge"]:
            raise ValueError(f"Invalid environment: {environment}. Must be 'cloud', 'desktop', or 'edge'.")
        
        # Create a DeviceOptimizer with auto-detected environment to get system info
        optimizer = DeviceOptimizer(environment=environment)
        
        # Save configs with this environment setting
        master_config = optimizer.save_configs(config_id=f"env_{environment}_{uuid.uuid4().hex[:6]}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Optimized configurations for {environment} environment generated successfully",
            "master_config": master_config
        })
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating optimized configurations for environment {environment}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create optimized configurations for environment: {str(e)}"
        )

@app.post("/optimize/workload/{workload_type}", tags=["Configuration Generation"])
async def optimize_for_specific_workload(
    workload_type: str = Path(..., description="Target workload type (inference, training, mixed)"),
    api_key: str = Depends(verify_api_key)
):
    """
    Create optimized configurations specifically for a given workload type.
    
    This endpoint generates configurations optimized for a specific workload:
    - inference: Settings optimized for inference tasks
    - training: Settings optimized for training tasks
    - mixed: Balanced settings for mixed workloads
    """
    try:
        # Validate workload type
        if workload_type not in ["inference", "training", "mixed"]:
            raise ValueError(f"Invalid workload type: {workload_type}. Must be 'inference', 'training', or 'mixed'.")
        
        # Create DeviceOptimizer with the specified workload type
        optimizer = DeviceOptimizer(workload_type=workload_type)
        
        # Save configs with this workload setting
        master_config = optimizer.save_configs(config_id=f"workload_{workload_type}_{uuid.uuid4().hex[:6]}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Optimized configurations for {workload_type} workload generated successfully",
            "master_config": master_config
        })
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating optimized configurations for workload {workload_type}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create optimized configurations for workload: {str(e)}"
        )

@app.post("/configs/load", tags=["Configuration Management"])
async def load_configurations(
    request: LoadConfigRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Load previously saved configurations.
    
    This endpoint loads configurations saved with a specific config_id,
    including all component configurations (quantization, batch processing, etc.).
    """
    try:
        # Create DeviceOptimizer and load configs
        optimizer = DeviceOptimizer(config_path=request.config_path)
        loaded_configs = optimizer.load_configs(request.config_id)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Configurations with ID {request.config_id} loaded successfully",
            "configs": loaded_configs
        })
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error loading configurations with ID {request.config_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load configurations: {str(e)}"
        )

@app.post("/configs/apply", tags=["Configuration Management"])
async def apply_configurations(
    request: ApplyConfigRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Apply loaded configurations to ML pipeline components.
    
    This endpoint applies the provided configurations to the appropriate
    ML pipeline components (quantizer, batch processor, preprocessor, etc.).
    """
    try:
        # Apply the configurations - assume this function is implemented elsewhere
        success = apply_configs_to_pipeline(request.configs)
        
        if success:
            return JSONResponse(content={
                "status": "success",
                "message": "Configurations successfully applied to pipeline components"
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to apply configurations to pipeline components"
            )
    except Exception as e:
        logger.error(f"Error applying configurations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply configurations: {str(e)}"
        )

@app.post("/configs/default", tags=["Configuration Management"])
async def get_default_configurations(
    request: DefaultConfigRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Get a set of default configurations optimized for the current system.
    
    This endpoint creates optimized configurations based on the specified
    parameters and current system capabilities, without auto-tuning.
    """
    try:
        # Convert string to enum
        opt_mode = OptimizationMode[request.optimization_mode]
        
        # Create DeviceOptimizer with minimal parameters
        optimizer = DeviceOptimizer(
            optimization_mode=opt_mode,
            workload_type=request.workload_type.value,
            environment=request.environment.value,
            enable_specialized_accelerators=request.enable_specialized_accelerators,
            auto_tune=False  # Default configs don't use auto-tuning
        )
        
        # Generate basic configs and return them
        quant_config = optimizer.get_optimal_quantization_config()
        batch_config = optimizer.get_optimal_batch_processor_config()
        preproc_config = optimizer.get_optimal_preprocessor_config()
        infer_config = optimizer.get_optimal_inference_engine_config()
        train_config = optimizer.get_optimal_training_engine_config()
        
        configs = {
            "quantization_config": quant_config.to_dict() if hasattr(quant_config, 'to_dict') else jsonable_encoder(quant_config),
            "batch_processor_config": batch_config.to_dict() if hasattr(batch_config, 'to_dict') else jsonable_encoder(batch_config),
            "preprocessor_config": preproc_config.to_dict() if hasattr(preproc_config, 'to_dict') else jsonable_encoder(preproc_config),
            "inference_engine_config": infer_config.to_dict() if hasattr(infer_config, 'to_dict') else jsonable_encoder(infer_config),
            "training_engine_config": train_config.to_dict() if hasattr(train_config, 'to_dict') else jsonable_encoder(train_config)
        }
        
        # Save to output_dir if specified
        if request.output_dir:
            os.makedirs(request.output_dir, exist_ok=True)
            for config_name, config_data in configs.items():
                config_path = os.path.join(request.output_dir, f"{config_name}.json")
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
            
        return JSONResponse(content={
            "status": "success",
            "message": "Default configurations generated successfully",
            "configs": configs
        })
    except Exception as e:
        logger.error(f"Error getting default configurations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get default configurations: {str(e)}"
        )

@app.get("/configs/list", tags=["Configuration Management"])
async def list_configurations(
    config_path: str = Query("./configs", description="Path where configuration files are stored"),
    api_key: str = Depends(verify_api_key)
):
    """
    List all available configuration sets.
    
    This endpoint scans the configuration directory and returns information
    about all available configuration sets, including their IDs and creation dates.
    """
    try:
        configs_dir = FilePath(config_path)
        
        if not configs_dir.exists() or not configs_dir.is_dir():
            return JSONResponse(content={
                "configs": []
            })
        
        configs = []
        
        # Iterate through subdirectories (each should be a config set)
        for config_dir in configs_dir.iterdir():
            if config_dir.is_dir():
                # Look for master_config.json
                master_config_path = config_dir / "master_config.json"
                
                if master_config_path.exists():
                    try:
                        with open(master_config_path, "r") as f:
                            master_config = json.load(f)
                            
                        # Extract info
                        config_info = {
                            "config_id": master_config.get("config_id", config_dir.name),
                            "optimization_mode": master_config.get("optimization_mode", "unknown"),
                            "creation_timestamp": master_config.get("creation_timestamp", "unknown"),
                            "path": str(config_dir)
                        }
                        
                        configs.append(config_info)
                    except Exception as e:
                        logger.warning(f"Failed to load master config from {master_config_path}: {e}")
                        # Add basic info if master config can't be loaded
                        configs.append({
                            "config_id": config_dir.name,
                            "path": str(config_dir),
                            "status": "incomplete"
                        })
        
        return JSONResponse(content={
            "configs": configs
        })
    except Exception as e:
        logger.error(f"Error listing configurations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list configurations: {str(e)}"
        )

@app.delete("/configs/{config_id}", tags=["Configuration Management"])
async def delete_configuration(
    config_id: str = Path(..., description="Identifier for the configuration set to delete"),
    config_path: str = Query("./configs", description="Path where configuration files are stored"),
    api_key: str = Depends(verify_api_key)
):
    """
    Delete a specific configuration set.
    
    This endpoint removes a configuration set and all its files from the filesystem.
    """
    try:
        config_dir = FilePath(config_path) / config_id
        
        if not config_dir.exists() or not config_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Configuration with ID {config_id} not found"
            )
        
        # Remove the directory and all its contents
        shutil.rmtree(config_dir)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Configuration with ID {config_id} deleted successfully"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting configuration with ID {config_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete configuration: {str(e)}"
        )

# ------------------ Background Tasks ------------------

def cleanup_old_configs(older_than_days: int, config_path: str):
    """Background task to clean up old configuration files"""
    try:
        configs_dir = FilePath(config_path)
        if not configs_dir.exists() or not configs_dir.is_dir():
            logger.warning(f"Configuration directory {config_path} does not exist")
            return
        
        current_time = datetime.datetime.now()
        threshold = current_time - datetime.timedelta(days=older_than_days)
        
        # Iterate through subdirectories
        for config_dir in configs_dir.iterdir():
            if config_dir.is_dir():
                # Look for master_config.json to get creation timestamp
                master_config_path = config_dir / "master_config.json"
                
                if master_config_path.exists():
                    try:
                        with open(master_config_path, "r") as f:
                            master_config = json.load(f)
                        
                        # Get creation timestamp
                        creation_str = master_config.get("creation_timestamp")
                        if creation_str:
                            creation_time = datetime.datetime.fromisoformat(creation_str)
                            
                            # If older than threshold, delete
                            if creation_time < threshold:
                                logger.info(f"Deleting old configuration: {config_dir.name}")
                                shutil.rmtree(config_dir)
                    except Exception as e:
                        logger.error(f"Error processing {master_config_path}: {e}")
        
        logger.info(f"Cleanup completed. Removed configurations older than {older_than_days} days")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@app.post("/maintenance/cleanup", tags=["Maintenance"])
async def schedule_cleanup(
    background_tasks: BackgroundTasks,
    older_than_days: int = Query(30, description="Delete configurations older than this many days", ge=1),
    config_path: str = Query("./configs", description="Path where configuration files are stored"),
    api_key: str = Depends(verify_api_key)
):
    """
    Schedule a background task to clean up old configuration files.
    
    This endpoint initiates a background task that deletes configuration files
    older than the specified number of days.
    """
    try:
        # Schedule the cleanup task
        background_tasks.add_task(cleanup_old_configs, older_than_days, config_path)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Cleanup scheduled for configurations older than {older_than_days} days"
        })
    except Exception as e:
        logger.error(f"Error scheduling cleanup task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule cleanup task: {str(e)}"
        )

# Run the application with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)