"""
Enhanced Error Handling Module for kolosal AutoML APIs

Provides comprehensive error handling including:
- Custom exception classes
- Error recovery mechanisms
- Graceful degradation strategies
- Comprehensive logging
- User-friendly error responses

Author: AI Assistant
Date: 2025-07-20
"""

import logging
import traceback
import sys
import time
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
import json

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    PROCESSING_ERROR = "processing_error"
    CONFIGURATION_ERROR = "configuration_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"


class ErrorResponse(BaseModel):
    """Standardized error response model"""
    error: bool = True
    error_code: str
    message: str
    detail: Optional[str] = None
    category: str
    severity: str
    timestamp: str
    request_id: Optional[str] = None
    suggestions: List[str] = []
    debug_info: Optional[Dict[str, Any]] = None


class KolosalException(Exception):
    """Base exception class for kolosal AutoML"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "KOLOSAL_ERROR",
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        detail: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.detail = detail
        self.suggestions = suggestions or []
        self.original_exception = original_exception
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "detail": self.detail,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "suggestions": self.suggestions
        }


class ValidationError(KolosalException):
    """Validation error exception"""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )
        self.field = field


class AuthenticationError(KolosalException):
    """Authentication error exception"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message,
            error_code="AUTH_ERROR",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            suggestions=["Check your API key", "Ensure proper authentication headers"],
            **kwargs
        )


class AuthorizationError(KolosalException):
    """Authorization error exception"""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message,
            error_code="AUTHZ_ERROR",
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            suggestions=["Check your permissions", "Contact administrator"],
            **kwargs
        )


class ResourceNotFoundError(KolosalException):
    """Resource not found error"""
    
    def __init__(self, resource_type: str, resource_id: str = "", **kwargs):
        message = f"{resource_type} not found"
        if resource_id:
            message += f": {resource_id}"
        
        super().__init__(
            message,
            error_code="RESOURCE_NOT_FOUND",
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[f"Check if {resource_type} exists", "Verify the identifier"],
            **kwargs
        )


class ResourceExhaustedError(KolosalException):
    """Resource exhausted error"""
    
    def __init__(self, resource_type: str, **kwargs):
        super().__init__(
            f"{resource_type} resource exhausted",
            error_code="RESOURCE_EXHAUSTED",
            category=ErrorCategory.RESOURCE_EXHAUSTED,
            severity=ErrorSeverity.HIGH,
            suggestions=["Try again later", "Reduce request size", "Check system capacity"],
            **kwargs
        )


class ProcessingError(KolosalException):
    """Processing error exception"""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        if operation:
            message = f"Processing error in {operation}: {message}"
        
        super().__init__(
            message,
            error_code="PROCESSING_ERROR",
            category=ErrorCategory.PROCESSING_ERROR,
            severity=ErrorSeverity.MEDIUM,
            suggestions=["Check input data", "Verify configuration", "Try again"],
            **kwargs
        )


class ConfigurationError(KolosalException):
    """Configuration error exception"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        if config_key:
            message = f"Configuration error for '{config_key}': {message}"
        
        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            category=ErrorCategory.CONFIGURATION_ERROR,
            severity=ErrorSeverity.HIGH,
            suggestions=["Check configuration file", "Verify environment variables"],
            **kwargs
        )


class ExternalServiceError(KolosalException):
    """External service error exception"""
    
    def __init__(self, service_name: str, message: str, **kwargs):
        super().__init__(
            f"External service error ({service_name}): {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            category=ErrorCategory.EXTERNAL_SERVICE_ERROR,
            severity=ErrorSeverity.HIGH,
            suggestions=["Check service availability", "Verify service configuration"],
            **kwargs
        )


class ErrorHandler:
    """Comprehensive error handler"""
    
    def __init__(self, debug_mode: bool = False, log_errors: bool = True):
        self.debug_mode = debug_mode
        self.log_errors = log_errors
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        
        # Setup error logger using centralized logging
        try:
            from modules.logging_config import get_logger
            self.logger = get_logger(
                name="kolosal_errors",
                level=logging.ERROR,
                log_file="kolosal_errors.log",
                enable_console=True
            )
        except ImportError:
            # Fallback to basic logging if centralized logging not available
            self.logger = logging.getLogger("kolosal_errors")
            self.logger.setLevel(logging.ERROR)
            
            # Only add handler if none exists
            if not self.logger.handlers:
                try:
                    error_handler = logging.FileHandler("kolosal_errors.log")
                    error_handler.setFormatter(
                        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    )
                    self.logger.addHandler(error_handler)
                except Exception as e:
                    # If file handler fails, just use console
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(
                        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    )
                    self.logger.addHandler(console_handler)
    
    def handle_exception(
        self,
        exception: Exception,
        request: Optional[Request] = None,
        include_debug: bool = None
    ) -> ErrorResponse:
        """Handle any exception and return standardized error response"""
        
        if include_debug is None:
            include_debug = self.debug_mode
        
        # Extract request information
        request_info = {}
        if request:
            request_info = {
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("User-Agent", "unknown")
            }
        
        # Handle kolosal exceptions
        if isinstance(exception, KolosalException):
            error_response = ErrorResponse(
                error_code=exception.error_code,
                message=exception.message,
                detail=exception.detail,
                category=exception.category.value,
                severity=exception.severity.value,
                timestamp=exception.timestamp,
                suggestions=exception.suggestions
            )
            
            if include_debug and exception.original_exception:
                error_response.debug_info = {
                    "original_exception": str(exception.original_exception),
                    "traceback": traceback.format_exc()
                }
        
        # Handle HTTP exceptions
        elif isinstance(exception, HTTPException):
            error_response = ErrorResponse(
                error_code="HTTP_ERROR",
                message=exception.detail,
                category=self._categorize_http_error(exception.status_code),
                severity=self._determine_http_severity(exception.status_code),
                timestamp=datetime.now().isoformat()
            )
        
        # Handle general exceptions
        else:
            error_response = ErrorResponse(
                error_code="UNEXPECTED_ERROR",
                message="An unexpected error occurred",
                detail=str(exception) if include_debug else None,
                category=ErrorCategory.SYSTEM_ERROR.value,
                severity=ErrorSeverity.HIGH.value,
                timestamp=datetime.now().isoformat(),
                suggestions=["Contact support if the problem persists"]
            )
            
            if include_debug:
                error_response.debug_info = {
                    "exception_type": type(exception).__name__,
                    "traceback": traceback.format_exc(),
                    "request_info": request_info
                }
        
        # Log the error
        if self.log_errors:
            self._log_error(exception, error_response, request_info)
        
        # Track error statistics
        self._track_error(error_response.error_code)
        
        return error_response
    
    def _categorize_http_error(self, status_code: int) -> str:
        """Categorize HTTP error based on status code"""
        if status_code == 400:
            return ErrorCategory.VALIDATION.value
        elif status_code == 401:
            return ErrorCategory.AUTHENTICATION.value
        elif status_code == 403:
            return ErrorCategory.AUTHORIZATION.value
        elif status_code == 404:
            return ErrorCategory.RESOURCE_NOT_FOUND.value
        elif status_code == 429:
            return ErrorCategory.RESOURCE_EXHAUSTED.value
        elif 400 <= status_code < 500:
            return ErrorCategory.VALIDATION.value
        else:
            return ErrorCategory.SYSTEM_ERROR.value
    
    def _determine_http_severity(self, status_code: int) -> str:
        """Determine severity based on HTTP status code"""
        if status_code < 400:
            return ErrorSeverity.LOW.value
        elif status_code < 500:
            return ErrorSeverity.MEDIUM.value
        else:
            return ErrorSeverity.HIGH.value
    
    def _log_error(self, exception: Exception, error_response: ErrorResponse, 
                  request_info: Dict[str, Any]):
        """Log error details"""
        log_entry = {
            "timestamp": error_response.timestamp,
            "error_code": error_response.error_code,
            "message": error_response.message,
            "category": error_response.category,
            "severity": error_response.severity,
            "exception_type": type(exception).__name__,
            "request_info": request_info
        }
        
        self.logger.error(json.dumps(log_entry))
        
        # Keep error history (limited size)
        self.error_history.append(log_entry)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]  # Keep last 500
    
    def _track_error(self, error_code: str):
        """Track error statistics"""
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "unique_error_types": len(self.error_counts),
            "recent_errors": len([
                e for e in self.error_history 
                if (datetime.now() - datetime.fromisoformat(e["timestamp"])).seconds < 3600
            ])
        }
    
    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent error history"""
        return self.error_history[-limit:]


class CircuitBreaker:
    """Circuit breaker for handling repeated failures"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        recovery_timeout: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker"""
        current_time = time.time()
        
        if self.state == "open":
            if current_time - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise ResourceExhaustedError(
                    "Service temporarily unavailable",
                    detail="Circuit breaker is open"
                )
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e


@contextmanager
def error_context(operation: str, reraise: bool = True):
    """Context manager for error handling"""
    try:
        yield
    except Exception as e:
        error_handler = ErrorHandler()
        error_response = error_handler.handle_exception(e)
        
        if reraise:
            raise ProcessingError(
                error_response.message,
                operation=operation,
                original_exception=e
            )
        else:
            return error_response


def create_error_middleware(error_handler: ErrorHandler):
    """Create error handling middleware for FastAPI"""
    
    async def error_middleware(request: Request, call_next):
        """Error handling middleware"""
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            error_response = error_handler.handle_exception(e, request)
            
            # Determine HTTP status code
            status_code = 500
            if isinstance(e, HTTPException):
                status_code = e.status_code
            elif isinstance(e, AuthenticationError):
                status_code = 401
            elif isinstance(e, AuthorizationError):
                status_code = 403
            elif isinstance(e, ResourceNotFoundError):
                status_code = 404
            elif isinstance(e, ValidationError):
                status_code = 400
            elif isinstance(e, ResourceExhaustedError):
                status_code = 429
            
            return JSONResponse(
                status_code=status_code,
                content=error_response.dict()
            )
    
    return error_middleware


# Retry decorator with exponential backoff
def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        break
                    
                    # Exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    time.sleep(delay)
            
            # All retries failed
            raise ProcessingError(
                f"Function failed after {max_attempts} attempts",
                original_exception=last_exception
            )
        
        return wrapper
    return decorator


# Default error handler instance
default_error_handler = ErrorHandler(debug_mode=False, log_errors=True)


# Helper functions for common error scenarios

def validate_required_fields(data: Dict[str, Any], required_fields: List[str]):
    """Validate required fields in data"""
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            detail=f"The following fields are required: {', '.join(missing_fields)}"
        )


def validate_data_types(data: Dict[str, Any], type_specs: Dict[str, type]):
    """Validate data types"""
    for field, expected_type in type_specs.items():
        if field in data and not isinstance(data[field], expected_type):
            raise ValidationError(
                f"Invalid type for field '{field}': expected {expected_type.__name__}, got {type(data[field]).__name__}",
                field=field
            )


def check_resource_exists(resource: Any, resource_type: str, resource_id: str = ""):
    """Check if resource exists"""
    if resource is None:
        raise ResourceNotFoundError(resource_type, resource_id)


def handle_external_service_call(service_name: str, func: Callable, *args, **kwargs):
    """Handle external service calls with proper error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        raise ExternalServiceError(
            service_name,
            str(e),
            original_exception=e
        )
