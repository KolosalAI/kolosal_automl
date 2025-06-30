import os
import time
import json
import logging
import uuid
import sys
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field
import uvicorn

# Import API modules
from modules.api.data_preprocessor_api import app as data_preprocessor_app
from modules.api.device_optimizer_api import app as device_optimizer_app
from modules.api.inference_engine_api import app as inference_engine_app
from modules.api.model_manager_api import app as model_manager_app
from modules.api.quantizer_api import app as quantizer_app
from modules.api.train_engine_api import app as train_engine_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kolosal_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("kolosal_api")

# API configuration
API_VERSION = "1.0.0"
API_TITLE = "kolosal AutoML API"
API_DESCRIPTION = """
kolosal AutoML provides a comprehensive suite of machine learning tools for data preprocessing,
model training, inference, quantization, and deployment with a focus on performance
optimization and ease of use.
"""

# Environment-based configuration
API_ENV = os.environ.get("API_ENV", "development")
API_DEBUG = os.environ.get("API_DEBUG", "False").lower() in ("true", "1", "t")
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
API_WORKERS = int(os.environ.get("API_WORKERS", "1"))

# Auth configuration
API_KEY_HEADER = os.environ.get("API_KEY_HEADER", "X-API-Key")
API_KEYS = os.environ.get("API_KEYS", "dev_key").split(",")
REQUIRE_API_KEY = os.environ.get("REQUIRE_API_KEY", "False").lower() in ("true", "1", "t")

# Static files configuration
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True, parents=True)


# Security setup
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


# Define lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup logic
    logger.info(f"Starting kolosal AutoML API v{API_VERSION} in {API_ENV} mode")
    
    # Initialize app state
    app.state.start_time = time.time()
    app.state.request_count = 0
    app.state.error_count = 0
    
    # Create metrics collector
    app.state.metrics = {
        "requests_per_endpoint": {},
        "average_response_time": {},
        "error_rate": {},
        "active_connections": 0
    }
    
    yield
    
    # Shutdown logic
    logger.info("Shutting down kolosal AutoML API")


async def verify_api_key(api_key: str = Header(None, alias=API_KEY_HEADER)):
    """Verify that the API key is valid."""
    # Check if API key is required
    if REQUIRE_API_KEY:
        # Check if API key is provided
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key"
            )
        # Check if API key is valid
        if api_key not in API_KEYS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
    return True


# Response models
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="API environment")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    components: Dict[str, str] = Field(..., description="Status of individual components")
    timestamp: str = Field(..., description="Response timestamp")


class MetricsResponse(BaseModel):
    """API metrics response model"""
    total_requests: int = Field(..., description="Total number of requests")
    errors: int = Field(..., description="Total number of errors")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    requests_per_endpoint: Dict[str, int] = Field(..., description="Request count per endpoint")
    average_response_time: Dict[str, float] = Field(..., description="Average response time per endpoint")
    active_connections: int = Field(..., description="Number of active connections")
    timestamp: str = Field(..., description="Response timestamp")


# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=None,  # Custom docs URL to require auth
    redoc_url=None,  # Custom redoc URL to require auth
    openapi_url="/openapi.json" if not REQUIRE_API_KEY else None,  # Conditional schema
    lifespan=lifespan
)


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Create request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next: Callable) -> Response:
    """
    Middleware to add a unique request ID to each request and track metrics.
    
    Args:
        request: The incoming request
        call_next: The next middleware or route handler
        
    Returns:
        Response: The processed response
    """
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Track request start time
    start_time = time.time()
    
    # Update metrics
    app.state.request_count += 1
    app.state.metrics["active_connections"] += 1
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Update endpoint metrics
        endpoint = f"{request.method}:{request.url.path}"
        if endpoint not in app.state.metrics["requests_per_endpoint"]:
            app.state.metrics["requests_per_endpoint"][endpoint] = 0
            app.state.metrics["average_response_time"][endpoint] = 0.0
        
        # Update metrics with exponential moving average for response time
        current_count = app.state.metrics["requests_per_endpoint"][endpoint]
        current_avg = app.state.metrics["average_response_time"][endpoint]
        
        app.state.metrics["requests_per_endpoint"][endpoint] += 1
        if current_count > 0:
            # Use EMA with alpha=0.05 for smoother averaging
            alpha = 0.05
            app.state.metrics["average_response_time"][endpoint] = (
                (1 - alpha) * current_avg + alpha * response_time
            )
        else:
            app.state.metrics["average_response_time"][endpoint] = response_time
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        # Track errors
        app.state.error_count += 1
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise
    finally:
        # Update connection count
        app.state.metrics["active_connections"] -= 1


# API routers
main_router = APIRouter(prefix="/api")
health_router = APIRouter(tags=["Health & Monitoring"])


# Authenticated documentation routes
@app.get("/docs", include_in_schema=False)
async def get_docs(authorized: bool = Depends(verify_api_key)):
    """Custom Swagger UI that requires API key authentication."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{API_TITLE} - Interactive Docs",
        swagger_favicon_url="/static/favicon.ico"
    )


@app.get("/redoc", include_in_schema=False)
async def get_redoc(authorized: bool = Depends(verify_api_key)):
    """Custom ReDoc that requires API key authentication."""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{API_TITLE} - ReDoc",
        redoc_favicon_url="/static/favicon.ico"
    )


# Improved function to include component routes in the main application
def include_component_routes(app: FastAPI, router: APIRouter, prefix: str, tags: List[str]):
    """
    Include routes from a FastAPI app into a router with specific prefix and tags.
    
    Args:
        app: The FastAPI app containing the routes
        router: The router to add the routes to
        prefix: URL prefix for the component
        tags: OpenAPI tags for the component
    """
    component_router = APIRouter(prefix=prefix, tags=tags)
    
    # Extract routes from the app
    for route in app.routes:
        # Skip non-API routes
        if hasattr(route, "methods"):
            # Create a new route with the same endpoint but modified path and add to component router
            path = route.path
            if path.startswith("/"):
                path = path[1:]  # Remove leading slash to avoid double slashes
                
            component_router.add_api_route(
                path=f"/{path}" if path else "",  # Add leading slash back
                endpoint=route.endpoint,
                methods=route.methods,
                response_model=getattr(route, "response_model", None),
                status_code=getattr(route, "status_code", 200),
                tags=getattr(route, "tags", None),
                dependencies=getattr(route, "dependencies", None),
                summary=getattr(route, "summary", None),
                description=getattr(route, "description", None),
                response_description=getattr(route, "response_description", "Successful Response"),
                responses=getattr(route, "responses", None),
                deprecated=getattr(route, "deprecated", None),
                operation_id=getattr(route, "operation_id", None),
                include_in_schema=getattr(route, "include_in_schema", True),
                response_class=getattr(route, "response_class", JSONResponse),
                name=getattr(route, "name", None),
            )
    
    # Include the component router in the provided router
    router.include_router(component_router)


# Health check endpoints
@health_router.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "documentation": "/docs",
        "health_check": "/health"
    }


@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API and components are operational."""
    # Check each component
    components = {
        "data_preprocessor": "healthy",
        "device_optimizer": "healthy",
        "inference_engine": "healthy",
        "model_manager": "healthy",
        "quantizer": "healthy",
        "train_engine": "healthy"
    }
    
    # Simple health check - in a real implementation you'd do a deeper check
    
    return {
        "status": "healthy",
        "version": API_VERSION,
        "environment": API_ENV,
        "uptime_seconds": time.time() - app.state.start_time,
        "components": components,
        "timestamp": datetime.now().isoformat()
    }


@health_router.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(verify_api_key)])
async def get_metrics():
    """Get API performance metrics."""
    return {
        "total_requests": app.state.request_count,
        "errors": app.state.error_count,
        "uptime_seconds": time.time() - app.state.start_time,
        "requests_per_endpoint": app.state.metrics["requests_per_endpoint"],
        "average_response_time": app.state.metrics["average_response_time"],
        "active_connections": app.state.metrics["active_connections"],
        "timestamp": datetime.now().isoformat()
    }


# Include component routers with improved function
include_component_routes(
    app=data_preprocessor_app,
    router=main_router,
    prefix="/preprocessor", 
    tags=["Data Preprocessing"]
)
include_component_routes(
    app=device_optimizer_app,
    router=main_router,
    prefix="/device",
    tags=["Device Optimization"]
)
include_component_routes(
    app=inference_engine_app,
    router=main_router,
    prefix="/inference",
    tags=["Inference Engine"]
)
include_component_routes(
    app=model_manager_app,
    router=main_router,
    prefix="/models",
    tags=["Model Management"]
)
include_component_routes(
    app=quantizer_app,
    router=main_router,
    prefix="/quantizer",
    tags=["Quantization"]
)
include_component_routes(
    app=train_engine_app,
    router=main_router,
    prefix="/train",
    tags=["Training Engine"]
)


# Include routers in the main app
app.include_router(health_router)
app.include_router(main_router)


# Improved global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for all unhandled exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The exception that was raised
        
    Returns:
        JSONResponse: A standardized error response
    """
    # Get request ID if available
    request_id = getattr(request.state, "request_id", None)
    
    # Determine status code - use HTTP exception status if available, else 500
    status_code = getattr(exc, "status_code", 500)
    
    # Log the error with traceback
    logger.exception(f"Unhandled exception on request {request_id}: {str(exc)}")
    
    # Increment error count
    app.state.error_count += 1
    
    # Get exception details
    error_details = None
    if hasattr(exc, "__dict__"):
        try:
            # Try to extract relevant details from the exception
            error_details = {k: v for k, v in exc.__dict__.items() 
                            if not k.startswith('_') and not callable(v)}
        except:
            # If extraction fails, don't include details
            pass
    
    # Create error response
    error_response = {
        "status": "error",
        "message": str(exc),
        "type": exc.__class__.__name__,
        "request_id": request_id,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add details if available
    if error_details:
        error_response["details"] = error_details
    
    # Return JSON response
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


# Main entry point
if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "modules.api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_DEBUG,
        workers=API_WORKERS,
        log_level="debug" if API_DEBUG else "info"
    )