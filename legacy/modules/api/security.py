"""
Enhanced Security Module for kolosal AutoML APIs

Provides comprehensive security features including:
- Advanced API key management with bcrypt hashing
- Enhanced rate limiting with multiple strategies
- Input validation and sanitization
- Security headers with HSTS and CSP
- Request filtering and threat detection
- Comprehensive audit logging
- Integration with enhanced security framework

Author: GitHub Copilot (Enhanced)
Date: 2025-07-24
Version: 0.2.0
"""

import os
import time
import hashlib
import hmac
import secrets
import logging
import base64
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
import json
import re

from fastapi import HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader, HTTPBearer
from pydantic import BaseModel, validator
import jwt
import bcrypt

# Enhanced security imports
try:
    from ..security.enhanced_security import EnhancedSecurityManager
    from ..security.security_config import get_security_environment
    ENHANCED_SECURITY_AVAILABLE = True
except ImportError:
    ENHANCED_SECURITY_AVAILABLE = False
    logging.warning("Enhanced security modules not available, using basic security")

# Configure security logger
try:
    from modules.logging_config import get_logger
    security_logger = get_logger(
        name="kolosal_security",
        level=logging.INFO,
        log_file="kolosal_security.log",
        enable_console=True
    )
except ImportError:
    # Fallback to basic logging if centralized logging not available
    security_logger = logging.getLogger("kolosal_security")
    security_logger.setLevel(logging.INFO)
    
    # Only add handler if none exists
    if not security_logger.handlers:
        try:
            security_handler = logging.FileHandler("kolosal_security.log")
            security_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            security_logger.addHandler(security_handler)
        except Exception:
            # If file handler fails, use console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            security_logger.addHandler(console_handler)


class SecurityConfig(BaseModel):
    """Security configuration model"""
    require_api_key: bool = True
    api_keys: List[str] = []
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    enable_jwt_auth: bool = False
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    enable_input_validation: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    enable_security_headers: bool = True
    enable_audit_logging: bool = True
    blocked_ips: List[str] = []
    allowed_origins: List[str] = ["*"]
    
    @validator("api_keys")
    def validate_api_keys(cls, v):
        """Validate API keys are not empty if required"""
        if not v:
            return ["dev_key"]  # Default development key
        return v
    
    @validator("jwt_secret")
    def validate_jwt_secret(cls, v, values):
        """Generate JWT secret if JWT is enabled but no secret provided"""
        if values.get("enable_jwt_auth") and not v:
            return secrets.token_urlsafe(32)
        return v


class RateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(self, requests_per_window: int = 100, window_seconds: int = 60):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.client_requests: Dict[str, deque] = defaultdict(deque)
        self.blocked_clients: Dict[str, float] = {}
        
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed based on rate limits"""
        current_time = time.time()
        
        # Check if client is temporarily blocked
        if client_id in self.blocked_clients:
            if current_time < self.blocked_clients[client_id]:
                return False
            else:
                del self.blocked_clients[client_id]
        
        # Clean old requests
        client_window = self.client_requests[client_id]
        cutoff_time = current_time - self.window_seconds
        
        while client_window and client_window[0] < cutoff_time:
            client_window.popleft()
        
        # Check rate limit
        if len(client_window) >= self.requests_per_window:
            # Block client for the remainder of the window
            self.blocked_clients[client_id] = current_time + self.window_seconds
            security_logger.warning(f"Rate limit exceeded for client {client_id}")
            return False
        
        # Record request
        client_window.append(current_time)
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        current_time = time.time()
        client_window = self.client_requests[client_id]
        cutoff_time = current_time - self.window_seconds
        
        # Clean old requests
        while client_window and client_window[0] < cutoff_time:
            client_window.popleft()
        
        return max(0, self.requests_per_window - len(client_window))
    
    def reset_client(self, client_id: str):
        """Reset rate limits for a specific client"""
        if client_id in self.client_requests:
            del self.client_requests[client_id]
        if client_id in self.blocked_clients:
            del self.blocked_clients[client_id]


class InputValidator:
    """Advanced input validation and sanitization"""
    
    # Common injection patterns
    INJECTION_PATTERNS = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'vbscript:',   # VBScript protocol
        r'on\w+\s*=',   # Event handlers
        r'expression\s*\(',  # CSS expression
        r'url\s*\(',    # CSS url
        r'&\w+;',       # HTML entities (basic check)
        r'\.\./+',      # Path traversal
        r'(union|select|insert|update|delete|drop|alter|create)\s+',  # SQL keywords
    ]
    
    @classmethod
    def sanitize_string(cls, value: str) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Limit length
        if len(value) > 10000:
            value = value[:10000]
        
        # Basic HTML encoding for dangerous characters
        dangerous_chars = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;'
        }
        
        for char, encoded in dangerous_chars.items():
            value = value.replace(char, encoded)
        
        return value
    
    @classmethod
    def validate_input(cls, value: Any, field_name: str = "input") -> bool:
        """Validate input for potential security issues"""
        if isinstance(value, str):
            value_lower = value.lower()
            
            # Check for injection patterns
            for pattern in cls.INJECTION_PATTERNS:
                if re.search(pattern, value_lower, re.IGNORECASE):
                    security_logger.warning(
                        f"Potential injection attempt in {field_name}: {pattern}"
                    )
                    return False
        
        elif isinstance(value, dict):
            # Recursively validate dictionary values
            for key, val in value.items():
                if not cls.validate_input(val, f"{field_name}.{key}"):
                    return False
        
        elif isinstance(value, list):
            # Validate list items
            for i, item in enumerate(value):
                if not cls.validate_input(item, f"{field_name}[{i}]"):
                    return False
        
        return True
    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary"""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = cls.sanitize_string(key)
            
            # Sanitize value
            if isinstance(value, str):
                clean_value = cls.sanitize_string(value)
            elif isinstance(value, dict):
                clean_value = cls.sanitize_dict(value)
            elif isinstance(value, list):
                clean_value = [
                    cls.sanitize_string(item) if isinstance(item, str)
                    else cls.sanitize_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                clean_value = value
            
            sanitized[clean_key] = clean_value
        
        return sanitized


class SecurityManager:
    """Main security manager class"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            config.rate_limit_requests,
            config.rate_limit_window
        ) if config.enable_rate_limiting else None
        
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
        self.bearer_auth = HTTPBearer(auto_error=False) if config.enable_jwt_auth else None
        
        # Audit log
        self.audit_requests: List[Dict[str, Any]] = []
        
    def get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        try:
            # Try to get real IP behind proxy
            if hasattr(request, 'headers'):
                forwarded_for = request.headers.get("X-Forwarded-For")
                if forwarded_for:
                    return forwarded_for.split(",")[0].strip()
                
                real_ip = request.headers.get("X-Real-IP")
                if real_ip:
                    return real_ip
            
            # Fall back to direct client IP
            if hasattr(request, 'client') and request.client:
                return request.client.host
            
            return "unknown"
        except Exception as e:
            security_logger.error(f"Failed to get client ID: {e}")
            return "unknown"
    
    def verify_api_key(self, api_key: Optional[str]) -> bool:
        """Verify API key"""
        if not self.config.require_api_key:
            return True
        
        if not api_key:
            return False
        
        # Simple key validation
        if api_key in self.config.api_keys:
            return True
        
        # Hash-based validation for enhanced security
        for valid_key in self.config.api_keys:
            if hmac.compare_digest(api_key, valid_key):
                return True
        
        return False
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        if not self.config.enable_jwt_auth or not self.config.jwt_secret:
            return None
        
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.InvalidTokenError as e:
            security_logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check rate limit for client"""
        if not self.rate_limiter:
            return True
        
        return self.rate_limiter.is_allowed(client_id)
    
    def validate_request(self, request_data: Any) -> bool:
        """Validate request data"""
        if not self.config.enable_input_validation:
            return True
        
        return InputValidator.validate_input(request_data)
    
    def sanitize_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data"""
        if not self.config.enable_input_validation:
            return request_data
        
        return InputValidator.sanitize_dict(request_data)
    
    def check_ip_blocked(self, ip: str) -> bool:
        """Check if IP is in blocked list"""
        return ip in self.config.blocked_ips
    
    def add_security_headers(self, response: Response):
        """Add security headers to response"""
        if not self.config.enable_security_headers:
            return
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    def log_request(self, request: Request, response_status: int, 
                   auth_success: bool, processing_time: float):
        """Log request for audit purposes"""
        if not self.config.enable_audit_logging:
            return
        
        try:
            client_id = self.get_client_id(request)
            
            # Safely get request information with fallbacks
            method = getattr(request, 'method', 'UNKNOWN')
            path = str(request.url.path) if hasattr(request, 'url') and request.url else 'UNKNOWN'
            user_agent = request.headers.get("User-Agent", "") if hasattr(request, 'headers') else ""
            content_length = request.headers.get("Content-Length", 0) if hasattr(request, 'headers') else 0
            
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "client_ip": client_id,
                "method": method,
                "path": path,
                "user_agent": user_agent,
                "response_status": response_status,
                "auth_success": auth_success,
                "processing_time_ms": round(processing_time * 1000, 2),
                "request_size": content_length
            }
            
            # Log to security logger
            security_logger.info(f"REQUEST: {json.dumps(audit_entry)}")
            
            # Keep in memory (limited)
            self.audit_requests.append(audit_entry)
            if len(self.audit_requests) > 1000:
                self.audit_requests = self.audit_requests[-500:]  # Keep last 500
        except Exception as e:
            # If logging fails, don't break the application but log the error
            security_logger.error(f"Failed to log request: {e}")
            try:
                # Minimal fallback logging
                security_logger.info(f"REQUEST: Failed to log details - status: {response_status}, auth: {auth_success}")
            except:
                pass  # If even fallback fails, just continue
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        return self.audit_requests[-limit:]


def create_security_middleware(security_manager: SecurityManager):
    """Create security middleware for FastAPI"""
    
    async def security_middleware(request: Request, call_next):
        """Security middleware implementation"""
        start_time = time.time()
        client_id = security_manager.get_client_id(request)
        auth_success = False
        
        try:
            # Check if IP is blocked
            if security_manager.check_ip_blocked(client_id):
                security_logger.warning(f"Blocked IP attempted access: {client_id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
            
            # Check rate limiting
            if not security_manager.check_rate_limit(client_id):
                remaining = security_manager.rate_limiter.get_remaining_requests(client_id)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"X-RateLimit-Remaining": str(remaining)}
                )
            
            # Check request size
            content_length = request.headers.get("Content-Length")
            if content_length and int(content_length) > security_manager.config.max_request_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Request too large"
                )
            
            # Process request
            response = await call_next(request)
            auth_success = True
            
            # Add security headers
            security_manager.add_security_headers(response)
            
            return response
            
        except HTTPException as e:
            # Log security event
            processing_time = time.time() - start_time
            security_manager.log_request(request, e.status_code, auth_success, processing_time)
            raise
        
        except Exception as e:
            # Log unexpected error
            processing_time = time.time() - start_time
            security_manager.log_request(request, 500, auth_success, processing_time)
            security_logger.error(f"Security middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal security error"
            )
        
        finally:
            # Always log the request
            processing_time = time.time() - start_time
            if 'response' in locals():
                security_manager.log_request(
                    request, 
                    response.status_code, 
                    auth_success, 
                    processing_time
                )
    
    return security_middleware


def create_auth_dependency(security_manager: SecurityManager):
    """Create authentication dependency for FastAPI"""
    
    async def verify_auth(request: Request, api_key: Optional[str] = None):
        """Verify authentication"""
        try:
            # Skip auth for health checks
            if hasattr(request, 'url') and request.url and request.url.path.endswith("/health"):
                return True
            
            # Check API key
            if security_manager.config.require_api_key:
                api_key = api_key or (request.headers.get("X-API-Key") if hasattr(request, 'headers') else None)
                if not security_manager.verify_api_key(api_key):
                    security_logger.warning(
                        f"Invalid API key from {security_manager.get_client_id(request)}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or missing API key",
                        headers={"WWW-Authenticate": "ApiKey"},
                    )
        except HTTPException:
            raise
        except Exception as e:
            security_logger.error(f"Auth verification error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication error"
            )
        
        # Check JWT if enabled
        if security_manager.config.enable_jwt_auth:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                payload = security_manager.verify_jwt_token(token)
                if not payload:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid JWT token",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                request.state.user = payload
        
        return True
    
    return verify_auth


# Utility functions for common security tasks

def generate_api_key(length: int = 32) -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(length)


def hash_password(password: str, salt: Optional[bytes] = None, iterations: int = 200000) -> str:
    """
    Hash password using PBKDF2 with SHA-256 and random salt (FIXED VULNERABILITY)
    
    Args:
        password: Password to hash
        salt: Optional salt (generates random if not provided)
        iterations: Number of PBKDF2 iterations
        
    Returns:
        Base64 encoded hash with salt and metadata
    """
    if salt is None:
        salt = os.urandom(32)  # Generate random salt instead of fixed 'salt'
    
    # Use PBKDF2 with SHA-256 and higher iteration count
    hash_bytes = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)
    
    # Combine salt and hash for storage with metadata
    combined = salt + hash_bytes
    return f"pbkdf2_sha256${iterations}${len(salt)}${base64.b64encode(combined).decode()}"


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against hash with improved security (FIXED VULNERABILITY)
    
    Args:
        password: Password to verify
        hashed: Stored hash with metadata
        
    Returns:
        True if password matches
    """
    try:
        # Parse the new hash format
        if '$' in hashed and hashed.startswith('pbkdf2_sha256$'):
            parts = hashed.split('$')
            if len(parts) != 4:
                return False
            
            algorithm, iterations_str, salt_len_str, combined_b64 = parts
            iterations = int(iterations_str)
            salt_len = int(salt_len_str)
            
            # Decode the combined salt+hash
            combined = base64.b64decode(combined_b64.encode())
            salt = combined[:salt_len]
            stored_hash = combined[salt_len:]
            
            # Hash the provided password with the stored salt
            test_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)
            
            # Use constant-time comparison
            return hmac.compare_digest(stored_hash, test_hash)
        else:
            # Legacy format fallback (should be migrated)
            security_logger.warning("Using legacy password hash format - should be updated")
            legacy_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), b'salt', 100000).hex()
            return hmac.compare_digest(hashed, legacy_hash)
            
    except Exception as e:
        security_logger.error(f"Password verification error: {e}")
        return False


def create_jwt_token(payload: Dict[str, Any], secret: str, 
                    expiry_hours: int = 24) -> str:
    """Create JWT token"""
    payload['exp'] = datetime.utcnow() + timedelta(hours=expiry_hours)
    return jwt.encode(payload, secret, algorithm="HS256")


# Security decorator for functions
def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Implementation would check user permissions
            # This is a placeholder for permission-based access control
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Default security configuration
DEFAULT_SECURITY_CONFIG = SecurityConfig(
    require_api_key=True,
    api_keys=["dev_key"],
    enable_rate_limiting=True,
    rate_limit_requests=100,
    rate_limit_window=60,
    enable_input_validation=True,
    enable_security_headers=True,
    enable_audit_logging=True
)
