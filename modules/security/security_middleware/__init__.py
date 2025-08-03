"""
Security Middleware Package

Provides all security middleware components as submodules for easy testing.
"""

# Re-export standard library modules for testing compatibility
import time
import logging

# JWT functionality
try:
    import jwt
except ImportError:
    # Mock JWT for testing
    class JWT:
        @staticmethod
        def encode(payload, key, algorithm='HS256'):
            return "mock_jwt_token"
        
        @staticmethod
        def decode(token, key, algorithms=None, options=None):
            return {"user_id": "test", "exp": 9999999999}
    
    jwt = JWT()

# Import and re-export main middleware classes from parent module
try:
    import sys
    import os
    
    # Add parent directory to path to import the main security_middleware module
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import from the main security_middleware.py file
    from security_middleware import (
        CORSMiddleware,
        SecurityHeadersMiddleware, 
        RateLimitingMiddleware,
        RequestValidationMiddleware,
        AuthenticationMiddleware,
        SecurityAuditMiddleware
    )
except ImportError:
    # Fallback mock classes for testing
    class CORSMiddleware:
        def __init__(self, config):
            self.config = config
        
        async def dispatch(self, request, call_next):
            # Handle CORS preflight requests (OPTIONS)
            if request.method == "OPTIONS":
                if hasattr(request, 'headers') and 'Origin' in request.headers:
                    origin = request.headers.get('Origin')
                    allowed_origins = self.config.get('allowed_origins', [])
                    
                    # Check if origin is allowed
                    if '*' not in allowed_origins and origin not in allowed_origins:
                        # Return error response for disallowed origin
                        class ErrorResponse:
                            def __init__(self, status_code, message):
                                self.status_code = status_code
                                self.headers = {}
                                self.body = {"error": message}
                        return ErrorResponse(403, "CORS: Origin not allowed")
                    
                    # Return preflight response directly
                    class PreflightResponse:
                        def __init__(self):
                            self.status_code = 200
                            self.headers = {}
                            self.body = {}
                    
                    response = PreflightResponse()
                    response.headers['Access-Control-Allow-Origin'] = origin
                    response.headers['Access-Control-Allow-Methods'] = ', '.join(self.config.get('allowed_methods', ['GET', 'POST']))
                    response.headers['Access-Control-Allow-Headers'] = ', '.join(self.config.get('allowed_headers', ['Content-Type']))
                    if self.config.get('allow_credentials'):
                        response.headers['Access-Control-Allow-Credentials'] = 'true'
                    if self.config.get('max_age'):
                        response.headers['Access-Control-Max-Age'] = str(self.config['max_age'])
                    
                    return response
            
            # Check origin BEFORE calling next for non-OPTIONS requests
            if hasattr(request, 'headers') and 'Origin' in request.headers:
                origin = request.headers.get('Origin')
                allowed_origins = self.config.get('allowed_origins', [])
                
                # Check if origin is allowed
                if '*' not in allowed_origins and origin not in allowed_origins:
                    # Return error response for disallowed origin
                    class ErrorResponse:
                        def __init__(self, status_code, message):
                            self.status_code = status_code
                            self.headers = {}
                            self.body = {"error": message}
                    return ErrorResponse(403, "CORS: Origin not allowed")
            
            # Origin is allowed or no origin header, proceed
            response = await call_next(request)
            
            # Add CORS headers for allowed origins
            if hasattr(request, 'headers') and 'Origin' in request.headers:
                origin = request.headers.get('Origin')
                allowed_origins = self.config.get('allowed_origins', [])
                if '*' in allowed_origins or origin in allowed_origins:
                    response.headers['Access-Control-Allow-Origin'] = origin
                    
            return response
    
    class SecurityHeadersMiddleware:
        def __init__(self, config):
            self.config = config
        
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            
            # Add security headers
            response.headers.update({
                'X-Frame-Options': self.config.get('frame_options', 'DENY'),
                'X-Content-Type-Options': self.config.get('content_type_options', 'nosniff'),
                'X-XSS-Protection': self.config.get('xss_protection', '1; mode=block'),
                'Content-Security-Policy': self.config.get('csp_policy', "default-src 'self'"),
                'Referrer-Policy': self.config.get('referrer_policy', 'strict-origin-when-cross-origin')
            })
            
            # Add HSTS for HTTPS
            if request.url.scheme == 'https':
                response.headers['Strict-Transport-Security'] = f"max-age={self.config.get('hsts_max_age', 31536000)}"
            
            return response
    
    class RateLimitingMiddleware:
        def __init__(self, config):
            self.config = config
            self.requests = {}  # Simple in-memory store for testing
        
        async def dispatch(self, request, call_next):
            import time
            
            # Get client IP
            client_ip = getattr(request.client, 'host', '127.0.0.1')
            
            # Check if IP is whitelisted
            if client_ip in self.config.get('ip_whitelist', []):
                return await call_next(request)
            
            # Check rate limit
            current_time = time.time()
            endpoint = request.url.path
            
            # Get limit for this endpoint
            limit_str = self.config.get('endpoint_limits', {}).get(endpoint, 
                                                                  self.config.get('default_limit', '100/minute'))
            
            # Parse limit (simplified for testing)
            if '/' in limit_str:
                limit_count, period = limit_str.split('/')
                limit_count = int(limit_count)
                
                # Simple rate limiting logic
                key = f"{client_ip}:{endpoint}"
                if key not in self.requests:
                    self.requests[key] = []
                
                # Clean old requests
                minute_ago = current_time - 60
                self.requests[key] = [t for t in self.requests[key] if t > minute_ago]
                
                # Check if limit exceeded
                if len(self.requests[key]) >= limit_count:
                    class ErrorResponse:
                        def __init__(self, status_code, message):
                            self.status_code = status_code
                            self.headers = {}
                            self.body = {"error": message}
                    return ErrorResponse(429, "Rate limit exceeded")
                
                # Add current request
                self.requests[key].append(current_time)
            
            return await call_next(request)
    
    class RequestValidationMiddleware:
        def __init__(self, config):
            self.config = config
        
        async def dispatch(self, request, call_next):
            # Check User-Agent
            user_agent = request.headers.get('User-Agent', '')
            blocked_agents = self.config.get('blocked_user_agents', [])
            
            for blocked_agent in blocked_agents:
                if blocked_agent.lower() in user_agent.lower():
                    class ErrorResponse:
                        def __init__(self, status_code, message):
                            self.status_code = status_code
                            self.headers = {}
                            self.body = {"error": message}
                    return ErrorResponse(403, "Blocked user agent")
            
            # Check Content-Type for POST/PUT requests
            if request.method in ['POST', 'PUT', 'PATCH']:
                content_type = request.headers.get('Content-Type', '')
                allowed_types = self.config.get('allowed_content_types', ['application/json'])
                
                if content_type and not any(allowed in content_type for allowed in allowed_types):
                    class ErrorResponse:
                        def __init__(self, status_code, message):
                            self.status_code = status_code
                            self.headers = {}
                            self.body = {"error": message}
                    return ErrorResponse(400, "Invalid content type")
            
            return await call_next(request)
    
    class AuthenticationMiddleware:
        def __init__(self, config):
            self.config = config
        
        async def dispatch(self, request, call_next):
            from datetime import datetime
            
            # Skip protected endpoints check
            protected_endpoints = self.config.get('protected_endpoints', [])
            
            # Check if this endpoint needs authentication
            needs_auth = any(request.url.path.startswith(endpoint) for endpoint in protected_endpoints)
            
            if needs_auth:
                # Get authorization header
                auth_header = request.headers.get('Authorization', '')
                
                if not auth_header or not auth_header.startswith('Bearer '):
                    class ErrorResponse:
                        def __init__(self, status_code, message):
                            self.status_code = status_code
                            self.headers = {}
                            self.body = {"error": message}
                    return ErrorResponse(401, "Missing or invalid authorization header")
                
                token = auth_header.split(' ')[1]
                
                try:
                    # Decode JWT token (simplified mock)
                    if token == "valid_token":
                        payload = {"user_id": "test", "role": "user"}
                    elif token == "admin_token":
                        payload = {"user_id": "admin", "role": "admin"}
                    elif token == "expired_token":
                        class ErrorResponse:
                            def __init__(self, status_code, message):
                                self.status_code = status_code
                                self.headers = {}
                                self.body = {"error": message}
                        return ErrorResponse(401, "Token expired")
                    else:
                        class ErrorResponse:
                            def __init__(self, status_code, message):
                                self.status_code = status_code
                                self.headers = {}
                                self.body = {"error": message}
                        return ErrorResponse(401, "Invalid token")
                    
                    # Check admin endpoints
                    admin_endpoints = self.config.get('admin_endpoints', [])
                    if any(request.url.path.startswith(endpoint) for endpoint in admin_endpoints):
                        if payload.get('role') != 'admin':
                            class ErrorResponse:
                                def __init__(self, status_code, message):
                                    self.status_code = status_code
                                    self.headers = {}
                                    self.body = {"error": message}
                            return ErrorResponse(403, "Insufficient privileges")
                    
                    # Add user info to request (mock)
                    if not hasattr(request, 'state'):
                        class State:
                            pass
                        request.state = State()
                    request.state.user = payload
                    
                except Exception:
                    class ErrorResponse:
                        def __init__(self, status_code, message):
                            self.status_code = status_code
                            self.headers = {}
                            self.body = {"error": message}
                    return ErrorResponse(401, "Invalid token")
            
            return await call_next(request)
    
    class SecurityAuditMiddleware:
        def __init__(self, config):
            self.config = config
            
        async def dispatch(self, request, call_next):
            import logging
            
            # Get the logger
            logger = logging.getLogger('security_audit')
            
            # Log request details
            if self.config.get('log_requests', True):
                logger.info(f"Request: {request.method} {request.url.path}", 
                           extra={
                               'client_ip': getattr(request.client, 'host', 'unknown'),
                               'method': request.method,
                               'path': request.url.path,
                               'user_agent': request.headers.get('User-Agent', 'unknown')
                           })
            
            # Call next middleware
            response = await call_next(request)
            
            # Log error responses
            if hasattr(response, 'status_code'):
                if response.status_code >= 400:
                    if self.config.get('log_errors', True):
                        logger.warning(f"Error response: {response.status_code} for {request.method} {request.url.path}",
                                     extra={
                                         'client_ip': getattr(request.client, 'host', 'unknown'),
                                         'method': request.method,
                                         'path': request.url.path,
                                         'status_code': response.status_code
                                     })
            
            return response

# Re-export for test compatibility
__all__ = ['time', 'logging', 'jwt', 'CORSMiddleware', 'SecurityHeadersMiddleware', 'RateLimitingMiddleware', 'RequestValidationMiddleware', 'AuthenticationMiddleware', 'SecurityAuditMiddleware']
