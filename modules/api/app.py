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
from fastapi.responses import HTMLResponse
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
from modules.api.batch_processor_api import app as batch_processor_app

# Import enhanced security and error handling
from modules.api.security import (
    SecurityManager, SecurityConfig, DEFAULT_SECURITY_CONFIG,
    create_security_middleware, create_auth_dependency
)
from modules.api.error_handling import (
    ErrorHandler, default_error_handler, create_error_middleware,
    KolosalException, ValidationError, AuthenticationError
)
from modules.api.monitoring import default_monitoring
from modules.api.dashboard import generate_dashboard_html

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
API_VERSION = "0.1.4"
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

# Security configuration
security_config = SecurityConfig(
    require_api_key=REQUIRE_API_KEY,
    api_keys=API_KEYS,
    enable_rate_limiting=True,
    rate_limit_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "100")),
    rate_limit_window=int(os.environ.get("RATE_LIMIT_WINDOW", "60")),
    enable_input_validation=True,
    enable_security_headers=True,
    enable_audit_logging=True,
    max_request_size=int(os.environ.get("MAX_REQUEST_SIZE", str(10 * 1024 * 1024)))  # 10MB
)

# Initialize security manager
security_manager = SecurityManager(security_config)

# Initialize enhanced error handler
error_handler = ErrorHandler(debug_mode=API_DEBUG, log_errors=True)

# Static files configuration
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True, parents=True)


# Define enhanced authentication
auth_dependency = create_auth_dependency(security_manager)

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
    app.state.security_manager = security_manager
    app.state.error_handler = error_handler
    app.state.monitoring = default_monitoring
    
    # Start monitoring system
    default_monitoring.start()
    
    # Create metrics collector
    app.state.metrics = {
        "requests_per_endpoint": {},
        "average_response_time": {},
        "error_rate": {},
        "active_connections": 0,
        "security_events": 0,
        "rate_limit_hits": 0
    }
    
    yield
    
    # Shutdown logic
    logger.info("Shutting down kolosal AutoML API")
    default_monitoring.stop()


# Enhanced authentication function
async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Verify API key with enhanced security"""
    return await auth_dependency(None, api_key)  # Request will be injected by FastAPI
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

# Enhanced middleware for security and error handling
@app.middleware("http")
async def security_and_error_middleware(request: Request, call_next):
    """Combined security and error handling middleware"""
    start_time = time.time()
    
    try:
        # Apply security middleware
        security_middleware = create_security_middleware(security_manager)
        response = await security_middleware(request, call_next)
        
        # Update metrics
        app.state.request_count += 1
        processing_time = time.time() - start_time
        
        endpoint = request.url.path
        if endpoint not in app.state.metrics["requests_per_endpoint"]:
            app.state.metrics["requests_per_endpoint"][endpoint] = 0
        app.state.metrics["requests_per_endpoint"][endpoint] += 1
        
        # Update average response time
        if endpoint not in app.state.metrics["average_response_time"]:
            app.state.metrics["average_response_time"][endpoint] = []
        app.state.metrics["average_response_time"][endpoint].append(processing_time)
        
        # Keep only last 100 response times for average calculation
        if len(app.state.metrics["average_response_time"][endpoint]) > 100:
            app.state.metrics["average_response_time"][endpoint] = \
                app.state.metrics["average_response_time"][endpoint][-100:]
        
        return response
        
    except Exception as e:
        # Handle with enhanced error handler
        app.state.error_count += 1
        error_response = error_handler.handle_exception(e, request)
        
        # Determine status code
        status_code = 500
        if isinstance(e, HTTPException):
            status_code = e.status_code
        elif isinstance(e, AuthenticationError):
            status_code = 401
        elif isinstance(e, ValidationError):
            status_code = 400
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict()
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
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Record metrics in monitoring system
        default_monitoring.record_api_request(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            processing_time=processing_time
        )
        
        # Update application metrics
        endpoint = str(request.url.path)
        if endpoint not in app.state.metrics["requests_per_endpoint"]:
            app.state.metrics["requests_per_endpoint"][endpoint] = 0
            app.state.metrics["average_response_time"][endpoint] = 0
        
        app.state.metrics["requests_per_endpoint"][endpoint] += 1
        
        # Update average response time
        old_avg = app.state.metrics["average_response_time"][endpoint]
        count = app.state.metrics["requests_per_endpoint"][endpoint]
        app.state.metrics["average_response_time"][endpoint] = (
            (old_avg * (count - 1) + processing_time) / count
        )
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = str(processing_time)
        
        return response
        
    except Exception as e:
        # Record error in monitoring system
        processing_time = time.time() - start_time
        default_monitoring.record_api_request(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=500,
            processing_time=processing_time
        )
        
        # Update error metrics
        app.state.error_count += 1
        raise e
        
    finally:
        # Decrease active connections
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
        "train_engine": "healthy",
        "batch_processor": "healthy"
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
include_component_routes(
    app=batch_processor_app,
    router=main_router,
    prefix="/batch",
    tags=["Batch Processing"]
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


# Monitoring endpoints
@app.get("/monitoring/health", tags=["Monitoring"])
async def get_health_check():
    """Get system health status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - app.state.start_time,
        "version": API_VERSION
    }


@app.get("/monitoring/metrics", tags=["Monitoring"], dependencies=[Depends(verify_api_key)])
async def get_metrics():
    """Get current system metrics"""
    try:
        metrics_data = default_monitoring.get_dashboard_data()
        return {
            "status": "success",
            "data": metrics_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@app.get("/monitoring/alerts", tags=["Monitoring"], dependencies=[Depends(verify_api_key)])
async def get_active_alerts():
    """Get active alerts"""
    try:
        active_alerts = default_monitoring.alert_manager.get_active_alerts()
        return {
            "status": "success",
            "active_alerts": [
                {
                    "name": alert.name,
                    "level": alert.level.value,
                    "message": alert.message,
                    "last_triggered": alert.last_triggered
                }
                for alert in active_alerts
            ],
            "alert_count": len(active_alerts),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alerts: {str(e)}")


@app.get("/monitoring/performance", tags=["Monitoring"], dependencies=[Depends(verify_api_key)])
async def get_performance_analysis():
    """Get performance analysis"""
    try:
        performance_data = default_monitoring.performance_analyzer.analyze_throughput_trends(24)
        resource_data = default_monitoring.performance_analyzer.analyze_resource_utilization(6)
        
        return {
            "status": "success",
            "performance_analysis": performance_data,
            "resource_analysis": resource_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")


@app.get("/monitoring/dashboard", tags=["Monitoring"], response_class=HTMLResponse)
async def get_monitoring_dashboard():
    """Get HTML monitoring dashboard (no auth required for dashboard view)"""
    try:
        metrics_data = default_monitoring.get_dashboard_data()
        html_content = generate_dashboard_html(metrics_data)
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body><h1>Dashboard Error</h1><p>Failed to load dashboard: {str(e)}</p></body></html>",
            status_code=500
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