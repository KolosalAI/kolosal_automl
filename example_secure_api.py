"""
Example API Integration with Enhanced Security

This module demonstrates how to integrate the enhanced security framework
into existing kolosal AutoML APIs.

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import enhanced security components
try:
    from modules.security.security_integration import (
        setup_application_security,
        require_api_key,
        secure_api_route
    )
    from modules.security.security_config import get_security_environment
    from modules.security.enhanced_security import EnhancedSecurityManager
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logging.warning("Enhanced security not available, using basic security")

# Create FastAPI app
app = FastAPI(
    title="kolosal AutoML API",
    description="Machine Learning API with Enhanced Security",
    version="0.2.0",
    docs_url="/docs" if SECURITY_AVAILABLE else None,  # Disable docs in production
    redoc_url="/redoc" if SECURITY_AVAILABLE else None
)

# Setup security if available
if SECURITY_AVAILABLE:
    security_integrator = setup_application_security(app)
    security_env = get_security_environment()
    enhanced_security = EnhancedSecurityManager()
else:
    # Fallback CORS for basic setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Health check endpoint (no security required)
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.2.0",
        "security_enabled": SECURITY_AVAILABLE
    }


# Public endpoint
@app.get("/api/v1/info")
async def api_info():
    """Get API information"""
    return {
        "name": "kolosal AutoML API",
        "version": "0.2.0",
        "description": "Machine Learning API with Enhanced Security",
        "security_features": [
            "API Key Authentication",
            "Rate Limiting",
            "Input Validation",
            "HTTPS Enforcement",
            "CORS Protection",
            "Security Headers",
            "Audit Logging"
        ] if SECURITY_AVAILABLE else ["Basic Security"]
    }


# Secured endpoint example
if SECURITY_AVAILABLE:
    @app.get("/api/v1/models", dependencies=[Depends(require_api_key())])
    @secure_api_route()
    async def list_models(request: Request):
        """List available models (secured endpoint)"""
        
        # Log API access
        client_ip = request.client.host if request.client else "unknown"
        logging.info(f"Models accessed from {client_ip}")
        
        return {
            "models": [
                {
                    "id": "model_1",
                    "name": "Classification Model",
                    "type": "classification",
                    "status": "active"
                },
                {
                    "id": "model_2", 
                    "name": "Regression Model",
                    "type": "regression",
                    "status": "active"
                }
            ]
        }
else:
    @app.get("/api/v1/models")
    async def list_models():
        """List available models (basic endpoint)"""
        return {
            "models": [
                {
                    "id": "model_1",
                    "name": "Classification Model", 
                    "type": "classification",
                    "status": "active"
                }
            ]
        }


# Prediction endpoint with enhanced security
if SECURITY_AVAILABLE:
    @app.post("/api/v1/predict", dependencies=[Depends(require_api_key())])
    @secure_api_route()
    async def predict(request: Request, data: dict):
        """Make predictions (secured endpoint)"""
        
        # Validate input data
        valid, issues = enhanced_security.validate_input(data)
        if not valid:
            logging.warning(f"Invalid input detected: {issues}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input data"
            )
        
        # Mock prediction logic
        prediction_result = {
            "prediction": "positive",
            "confidence": 0.95,
            "model_used": "model_1",
            "timestamp": "2025-07-24T10:00:00Z"
        }
        
        # Log prediction request
        client_ip = request.client.host if request.client else "unknown"
        logging.info(f"Prediction made from {client_ip}")
        
        return prediction_result
else:
    @app.post("/api/v1/predict")
    async def predict(data: dict):
        """Make predictions (basic endpoint)"""
        return {
            "prediction": "positive",
            "confidence": 0.95,
            "model_used": "model_1"
        }


# Admin endpoint with enhanced security
if SECURITY_AVAILABLE:
    @app.get("/api/v1/admin/security-status", dependencies=[Depends(require_api_key())])
    async def security_status(request: Request):
        """Get security status (admin endpoint)"""
        
        # Additional admin verification could be added here
        
        return {
            "security_level": security_env.security_level.value,
            "https_enforced": security_env.enforce_https,
            "rate_limiting_enabled": security_env.enable_rate_limiting,
            "api_key_required": security_env.require_api_key,
            "jwt_enabled": security_env.enable_jwt,
            "audit_logging": security_env.enable_audit_logging,
            "input_validation": security_env.enable_input_validation
        }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with security logging"""
    
    if SECURITY_AVAILABLE:
        # Log security-related errors
        if exc.status_code in [401, 403, 429]:
            client_ip = request.client.host if request.client else "unknown"
            logging.warning(f"Security error {exc.status_code} from {client_ip}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    
    logging.error(f"Unexpected error: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    if SECURITY_AVAILABLE:
        logging.info("kolosal AutoML API started with enhanced security")
        logging.info(f"Security level: {security_env.security_level.value}")
    else:
        logging.info("kolosal AutoML API started with basic security")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    if SECURITY_AVAILABLE:
        security_integrator.shutdown()
    logging.info("kolosal AutoML API shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    # Configure server based on security environment
    server_config = {
        "app": app,
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "info"
    }
    
    if SECURITY_AVAILABLE and security_env.enforce_https:
        # Add TLS configuration for HTTPS
        server_config.update({
            "ssl_keyfile": "certs/server.key",
            "ssl_certfile": "certs/server.crt",
            "ssl_ca_certs": "certs/ca.crt"
        })
        logging.info("Starting with HTTPS enabled")
    
    uvicorn.run(**server_config)
