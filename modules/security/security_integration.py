"""
Security Integration Module for kolosal AutoML

Integrates all security components and provides unified security management:
- Enhanced security framework integration
- Secure API initialization
- Security middleware setup
- Configuration validation
- Security monitoring

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, status

# Security component imports
from .security_config import get_security_environment, setup_secure_environment
from .enhanced_security import EnhancedSecurityManager
from .https_middleware import HTTPSEnforcementMiddleware
from .cors_middleware import SecureCORSMiddleware, setup_secure_cors
from .tls_manager import TLSManager
from .secrets_manager import get_secrets_manager


class SecurityIntegrator:
    """
    Central security integrator for kolosal AutoML
    
    Manages all security components and provides unified interface
    """
    
    def __init__(self, app: Optional[FastAPI] = None):
        self.app = app
        self.security_env = get_security_environment()
        self.enhanced_security = EnhancedSecurityManager()
        self.tls_manager = TLSManager()
        self.secrets_manager = get_secrets_manager()
        self.logger = logging.getLogger("kolosal_security.integrator")
        
        # Validate security configuration
        self._validate_security_setup()
    
    def _validate_security_setup(self) -> None:
        """Validate security configuration and components"""
        
        issues = self.security_env.validate()
        if issues:
            self.logger.warning(f"Security configuration issues: {issues}")
            
            # Critical issues that should stop startup
            critical_issues = [
                issue for issue in issues 
                if any(keyword in issue.lower() for keyword in 
                      ["production", "https", "api key", "wildcard"])
            ]
            
            if critical_issues and self.security_env.security_level.value == "production":
                raise ValueError(f"Critical security issues in production: {critical_issues}")
    
    def setup_security(self, app: FastAPI) -> None:
        """
        Setup comprehensive security for FastAPI application
        
        Args:
            app: FastAPI application instance
        """
        self.app = app
        
        # Setup security environment
        setup_secure_environment()
        
        # Add security middleware in correct order
        self._add_security_middleware()
        
        # Setup TLS/SSL if required
        if self.security_env.enforce_https:
            self._setup_tls()
        
        # Add security headers
        self._setup_security_headers()
        
        # Setup security routes
        self._setup_security_routes()
        
        # Initialize monitoring
        if self.security_env.enable_audit_logging:
            self._setup_security_monitoring()
        
        self.logger.info("Security framework initialized successfully")
    
    def _add_security_middleware(self) -> None:
        """Add security middleware to application"""
        
        if not self.app:
            return
        
        # 1. HTTPS enforcement (must be first)
        if self.security_env.enforce_https:
            self.app.add_middleware(
                HTTPSEnforcementMiddleware,
                redirect_http=self.security_env.redirect_http,
                hsts_enabled=self.security_env.hsts_enabled,
                hsts_max_age=self.security_env.hsts_max_age
            )
        
        # 2. CORS middleware
        setup_secure_cors(self.app)
        
        # 3. Rate limiting middleware (handled by enhanced security)
        if self.security_env.enable_rate_limiting:
            @self.app.middleware("http")
            async def rate_limiting_middleware(request: Request, call_next):
                allowed, details = self.enhanced_security.check_rate_limit(request)
                if not allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )
                return await call_next(request)
        
        # 4. Input validation middleware
        if self.security_env.enable_input_validation:
            @self.app.middleware("http")
            async def input_validation_middleware(request: Request, call_next):
                # Validate request data if JSON
                if request.headers.get("content-type") == "application/json":
                    try:
                        body = await request.body()
                        if body:
                            import json
                            data = json.loads(body.decode())
                            valid, issues = self.enhanced_security.validate_input(data)
                            if not valid:
                                self.logger.warning(f"Malicious input detected: {issues}")
                                raise HTTPException(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    detail="Invalid input detected"
                                )
                        # Re-create request with original body
                        request._body = body
                    except json.JSONDecodeError:
                        pass  # Not JSON, continue
                    except Exception as e:
                        self.logger.error(f"Input validation error: {e}")
                
                return await call_next(request)
    
    def _setup_tls(self) -> None:
        """Setup TLS/SSL configuration"""
        
        try:
            # Ensure certificates exist
            cert_config = self.tls_manager.get_certificate_config()
            
            if not cert_config:
                # Generate self-signed certificates for development
                if self.security_env.security_level.value in ["development", "testing"]:
                    self.logger.info("Generating self-signed certificates for development")
                    cert_config = self.tls_manager.generate_self_signed_certificate()
                else:
                    self.logger.error("No TLS certificates configured for production")
                    raise ValueError("TLS certificates required for production")
            
            self.logger.info("TLS configuration validated")
            
        except Exception as e:
            self.logger.error(f"TLS setup failed: {e}")
            if self.security_env.security_level.value == "production":
                raise
    
    def _setup_security_headers(self) -> None:
        """Setup security headers middleware"""
        
        if not self.app:
            return
        
        @self.app.middleware("http")
        async def security_headers_middleware(request: Request, call_next):
            response = await call_next(request)
            
            # Add security headers
            headers = self.enhanced_security.get_security_headers()
            for key, value in headers.items():
                response.headers[key] = value
            
            return response
    
    def _setup_security_routes(self) -> None:
        """Setup security-related routes"""
        
        if not self.app:
            return
        
        @self.app.get("/security/health")
        async def security_health():
            """Security health check endpoint"""
            return {
                "status": "healthy",
                "security_level": self.security_env.security_level.value,
                "https_enforced": self.security_env.enforce_https,
                "rate_limiting_enabled": self.security_env.enable_rate_limiting,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/security/configuration")
        async def security_configuration(request: Request):
            """Get security configuration (non-sensitive parts)"""
            
            # Verify admin access (basic check)
            api_key = request.headers.get("x-api-key")
            if not api_key or not self.enhanced_security.validate_api_key(
                api_key, self.security_env.get_api_keys()
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Admin access required"
                )
            
            config = self.security_env.to_dict()
            
            # Remove sensitive information
            sensitive_keys = [
                "api_keys", "jwt_secret", "blocked_ips", 
                "blocked_countries", "secret_key"
            ]
            for key in sensitive_keys:
                config.pop(key, None)
            
            return config
    
    def _setup_security_monitoring(self) -> None:
        """Setup security monitoring and alerting"""
        
        # Log security startup
        self.logger.info(f"Security monitoring enabled - Level: {self.security_env.security_level.value}")
        
        # Setup periodic security checks
        import asyncio
        
        async def security_monitor():
            """Periodic security monitoring"""
            while True:
                try:
                    # Check for security events
                    events = self.enhanced_security.get_recent_security_events()
                    
                    # Alert on critical events
                    critical_events = [
                        event for event in events 
                        if event.get("severity") == "CRITICAL"
                    ]
                    
                    if critical_events and self.security_env.alert_on_threats:
                        self.logger.critical(f"Critical security events detected: {len(critical_events)}")
                        # Could integrate with alerting system here
                    
                    # Sleep for monitoring interval
                    await asyncio.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    self.logger.error(f"Security monitoring error: {e}")
                    await asyncio.sleep(60)  # Retry in 1 minute
        
        # Start monitoring task
        if hasattr(self.app, "state"):
            self.app.state.security_monitor_task = asyncio.create_task(security_monitor())
    
    def get_api_security_dependencies(self):
        """Get security dependencies for API routes"""
        
        dependencies = []
        
        # Rate limiting dependency
        if self.security_env.enable_rate_limiting:
            async def rate_limit_dependency(request: Request):
                allowed, details = self.enhanced_security.check_rate_limit(request)
                if not allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )
            dependencies.append(rate_limit_dependency)
        
        # API key validation dependency
        if self.security_env.require_api_key:
            async def api_key_dependency(request: Request):
                api_key = request.headers.get("x-api-key")
                if not api_key:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="API key required"
                    )
                
                valid_keys = self.security_env.get_api_keys()
                if not self.enhanced_security.validate_api_key(api_key, valid_keys):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid API key"
                    )
            dependencies.append(api_key_dependency)
        
        return dependencies
    
    def shutdown(self) -> None:
        """Cleanup security resources"""
        
        try:
            # Cancel monitoring tasks
            if hasattr(self.app, "state") and hasattr(self.app.state, "security_monitor_task"):
                self.app.state.security_monitor_task.cancel()
            
            # Cleanup secrets manager
            self.secrets_manager.cleanup()
            
            self.logger.info("Security framework shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Security shutdown error: {e}")


# Global security integrator instance
_security_integrator: Optional[SecurityIntegrator] = None


def get_security_integrator() -> SecurityIntegrator:
    """Get or create global security integrator"""
    global _security_integrator
    
    if _security_integrator is None:
        _security_integrator = SecurityIntegrator()
    
    return _security_integrator


def setup_application_security(app: FastAPI, **kwargs) -> SecurityIntegrator:
    """
    Setup comprehensive security for FastAPI application
    
    Args:
        app: FastAPI application
        **kwargs: Additional security configuration
        
    Returns:
        SecurityIntegrator instance
    """
    integrator = get_security_integrator()
    integrator.setup_security(app)
    
    return integrator


# Convenience functions for easy integration
def secure_api_route(dependencies: Optional[List] = None):
    """
    Decorator to add security to API routes
    
    Args:
        dependencies: Additional dependencies for the route
        
    Returns:
        Decorator function
    """
    def decorator(func):
        integrator = get_security_integrator()
        security_deps = integrator.get_api_security_dependencies()
        
        if dependencies:
            security_deps.extend(dependencies)
        
        # Add dependencies to function
        if hasattr(func, "__annotations__"):
            func.__annotations__["dependencies"] = security_deps
        
        return func
    
    return decorator


def require_api_key():
    """Dependency to require API key authentication"""
    async def api_key_dependency(request: Request):
        integrator = get_security_integrator()
        api_key = request.headers.get("x-api-key")
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        valid_keys = integrator.security_env.get_api_keys()
        if not integrator.enhanced_security.validate_api_key(api_key, valid_keys):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
    
    return api_key_dependency
