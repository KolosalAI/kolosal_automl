"""
CORS Security Middleware for kolosal AutoML

Implements secure CORS policies with:
- Dynamic origin validation
- Strict credential handling
- Method and header filtering
- Preflight optimization
- Security logging

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import logging
import re
from typing import List, Optional, Set, Dict, Any
from urllib.parse import urlparse

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from .security_config import get_security_environment


class SecureCORSMiddleware(BaseHTTPMiddleware):
    """
    Secure CORS middleware with enhanced validation and logging
    """
    
    def __init__(
        self,
        app,
        allowed_origins: Optional[List[str]] = None,
        allowed_methods: Optional[List[str]] = None,
        allowed_headers: Optional[List[str]] = None,
        allow_credentials: bool = False,
        max_age: int = 86400,
        expose_headers: Optional[List[str]] = None,
        allow_origin_regex: Optional[str] = None,
        strict_mode: bool = True
    ):
        super().__init__(app)
        
        self.security_env = get_security_environment()
        self.logger = logging.getLogger("kolosal_security.cors")
        
        # Use security environment configuration if not explicitly provided
        self.allowed_origins = allowed_origins or self.security_env.allowed_origins
        self.allowed_methods = allowed_methods or self.security_env.allowed_methods
        self.allowed_headers = allowed_headers or self.security_env.allowed_headers
        self.allow_credentials = allow_credentials or self.security_env.allow_credentials
        self.max_age = max_age
        self.expose_headers = expose_headers or []
        self.allow_origin_regex = allow_origin_regex
        self.strict_mode = strict_mode
        
        # Compile regex patterns for performance
        self._compiled_origin_regex = None
        if self.allow_origin_regex:
            try:
                self._compiled_origin_regex = re.compile(self.allow_origin_regex)
            except re.error as e:
                self.logger.error(f"Invalid origin regex pattern: {e}")
        
        # Convert to sets for faster lookup
        self._allowed_methods_set = set(method.upper() for method in self.allowed_methods)
        self._allowed_headers_set = set(header.lower() for header in self.allowed_headers if header != "*")
        
        # Validate configuration
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate CORS configuration for security issues"""
        
        # Check for insecure wildcard origins with credentials
        if "*" in self.allowed_origins and self.allow_credentials:
            self.logger.error("CORS: Wildcard origins cannot be used with credentials")
            raise ValueError("CORS: Wildcard origins cannot be used with credentials")
        
        # Warn about wildcard origins in production
        if ("*" in self.allowed_origins and 
            self.security_env.security_level.value == "production"):
            self.logger.warning("CORS: Wildcard origins should not be used in production")
        
        # Validate methods
        valid_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}
        invalid_methods = self._allowed_methods_set - valid_methods
        if invalid_methods:
            self.logger.warning(f"CORS: Invalid HTTP methods: {invalid_methods}")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process CORS for incoming requests"""
        
        origin = request.headers.get("origin")
        method = request.method.upper()
        
        # Log CORS requests in strict mode
        if self.strict_mode and origin:
            self.logger.info(f"CORS request: {method} {request.url.path} from {origin}")
        
        # Handle preflight requests
        if method == "OPTIONS":
            return await self._handle_preflight(request, origin)
        
        # Handle actual requests
        response = await call_next(request)
        return self._add_cors_headers(response, origin, method)
    
    async def _handle_preflight(self, request: Request, origin: Optional[str]) -> StarletteResponse:
        """Handle CORS preflight requests"""
        
        # Check if origin is allowed
        if not self._is_origin_allowed(origin):
            self.logger.warning(f"CORS: Blocked preflight from disallowed origin: {origin}")
            return StarletteResponse(status_code=403)
        
        # Check requested method
        requested_method = request.headers.get("access-control-request-method", "").upper()
        if requested_method and requested_method not in self._allowed_methods_set:
            self.logger.warning(f"CORS: Blocked preflight for disallowed method: {requested_method}")
            return StarletteResponse(status_code=403)
        
        # Check requested headers
        requested_headers = request.headers.get("access-control-request-headers", "")
        if requested_headers and not self._are_headers_allowed(requested_headers):
            self.logger.warning(f"CORS: Blocked preflight for disallowed headers: {requested_headers}")
            return StarletteResponse(status_code=403)
        
        # Create preflight response
        response = StarletteResponse(status_code=200)
        
        # Add CORS headers
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
        
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        
        if "*" in self.allowed_headers:
            if requested_headers:
                response.headers["Access-Control-Allow-Headers"] = requested_headers
        else:
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        
        response.headers["Access-Control-Max-Age"] = str(self.max_age)
        
        # Add security headers
        response.headers["Vary"] = "Origin, Access-Control-Request-Method, Access-Control-Request-Headers"
        
        return response
    
    def _add_cors_headers(self, response: Response, origin: Optional[str], method: str) -> Response:
        """Add CORS headers to actual response"""
        
        # Only add headers if origin is allowed
        if not self._is_origin_allowed(origin):
            if origin and self.strict_mode:
                self.logger.warning(f"CORS: Blocked response to disallowed origin: {origin}")
            return response
        
        # Add origin header
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
        
        # Add credentials header
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        # Add exposed headers
        if self.expose_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        
        # Add Vary header for proper caching
        vary_header = "Origin"
        if "Vary" in response.headers:
            existing_vary = response.headers["Vary"]
            if "Origin" not in existing_vary:
                vary_header = f"{existing_vary}, Origin"
            else:
                vary_header = existing_vary
        response.headers["Vary"] = vary_header
        
        return response
    
    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """Check if origin is allowed"""
        
        if not origin:
            return True  # Allow same-origin requests
        
        # Check wildcard
        if "*" in self.allowed_origins:
            return True
        
        # Check exact match
        if origin in self.allowed_origins:
            return True
        
        # Check regex pattern
        if self._compiled_origin_regex and self._compiled_origin_regex.match(origin):
            return True
        
        # Additional origin validation
        return self._validate_origin_security(origin)
    
    def _validate_origin_security(self, origin: str) -> bool:
        """Additional security validation for origins"""
        
        try:
            parsed = urlparse(origin)
            
            # Must be HTTPS in production (except localhost)
            if (self.security_env.security_level.value == "production" and
                parsed.scheme != "https" and
                parsed.hostname not in ["localhost", "127.0.0.1"]):
                self.logger.warning(f"CORS: Non-HTTPS origin rejected in production: {origin}")
                return False
            
            # Block suspicious origins
            suspicious_patterns = [
                r".*\.ngrok\.io$",  # Tunneling services
                r".*\.localtunnel\.me$",
                r".*\.serveo\.net$",
                r".*\.burpcollaborator\.net$",  # Security testing tools
                r".*\.requestcatcher\.com$",
                r".*\.webhook\.site$"
            ]
            
            for pattern in suspicious_patterns:
                if re.match(pattern, parsed.hostname or ""):
                    self.logger.warning(f"CORS: Suspicious origin blocked: {origin}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"CORS: Error validating origin {origin}: {e}")
            return False
    
    def _are_headers_allowed(self, requested_headers: str) -> bool:
        """Check if requested headers are allowed"""
        
        # Allow all headers if wildcard is configured
        if "*" in self.allowed_headers:
            return True
        
        # Parse requested headers
        headers = [h.strip().lower() for h in requested_headers.split(",") if h.strip()]
        
        # Check each header
        for header in headers:
            # Always allow simple headers
            if header in {"accept", "accept-language", "content-language", "content-type"}:
                continue
            
            # Check against allowed headers
            if header not in self._allowed_headers_set:
                return False
        
        return True


def setup_secure_cors(app, **kwargs) -> None:
    """
    Setup secure CORS middleware on FastAPI app
    
    Args:
        app: FastAPI application instance
        **kwargs: Additional CORS configuration
    """
    
    # Get security environment for defaults
    security_env = get_security_environment()
    
    # Merge with provided configuration
    cors_config = {
        "allowed_origins": security_env.allowed_origins,
        "allowed_methods": security_env.allowed_methods,
        "allowed_headers": security_env.allowed_headers,
        "allow_credentials": security_env.allow_credentials,
        "strict_mode": security_env.security_level.value in ["production", "staging"],
        **kwargs
    }
    
    # Add middleware
    app.add_middleware(SecureCORSMiddleware, **cors_config)
    
    # Log configuration
    logger = logging.getLogger("kolosal_security.cors")
    logger.info(f"CORS configured with origins: {cors_config['allowed_origins']}")
    
    if cors_config.get('strict_mode'):
        logger.info("CORS strict mode enabled")


# Convenience function for common secure CORS configurations
def get_production_cors_config() -> Dict[str, Any]:
    """Get production-ready CORS configuration"""
    return {
        "allowed_origins": [],  # Must be explicitly configured
        "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
        "allowed_headers": [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-API-Key"
        ],
        "allow_credentials": False,
        "max_age": 3600,  # 1 hour
        "strict_mode": True
    }


def get_development_cors_config() -> Dict[str, Any]:
    """Get development CORS configuration"""
    return {
        "allowed_origins": [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001"
        ],
        "allowed_methods": ["*"],
        "allowed_headers": ["*"],
        "allow_credentials": False,
        "max_age": 86400,  # 24 hours
        "strict_mode": False
    }
