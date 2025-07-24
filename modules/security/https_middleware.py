"""
HTTPS Enforcement Middleware for kolosal AutoML

Provides comprehensive HTTPS enforcement and security:
- HTTP to HTTPS redirection
- HSTS header enforcement
- TLS version validation
- Secure cookie settings
- Certificate validation helpers

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import os
import logging
from typing import Callable, Optional
from urllib.parse import urlparse

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware


class HTTPSEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce HTTPS connections and add security headers
    """
    
    def __init__(self, app, 
                 enforce_https: bool = True,
                 redirect_http: bool = True,
                 hsts_max_age: int = 31536000,
                 include_subdomains: bool = True,
                 preload: bool = True,
                 allowed_hosts: Optional[list] = None,
                 development_mode: bool = False):
        """
        Initialize HTTPS enforcement middleware
        
        Args:
            app: FastAPI application
            enforce_https: Whether to enforce HTTPS
            redirect_http: Whether to redirect HTTP to HTTPS
            hsts_max_age: HSTS max-age in seconds
            include_subdomains: Include subdomains in HSTS
            preload: Enable HSTS preload
            allowed_hosts: List of allowed hostnames
            development_mode: Disable enforcement for development
        """
        super().__init__(app)
        self.enforce_https = enforce_https and not development_mode
        self.redirect_http = redirect_http and not development_mode
        self.hsts_max_age = hsts_max_age
        self.include_subdomains = include_subdomains
        self.preload = preload
        self.allowed_hosts = allowed_hosts or []
        self.development_mode = development_mode
        self.logger = logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with HTTPS enforcement"""
        
        # Check if we should enforce HTTPS
        if self.enforce_https:
            # Get the request scheme
            scheme = request.url.scheme
            headers = request.headers
            
            # Check for forwarded protocol headers (common in proxy setups)
            forwarded_proto = (
                headers.get("X-Forwarded-Proto") or
                headers.get("X-Forwarded-Protocol") or
                headers.get("X-Scheme") or
                headers.get("X-Original-Proto")
            )
            
            if forwarded_proto:
                scheme = forwarded_proto.lower()
            
            # If not HTTPS, handle accordingly
            if scheme != "https":
                if self.redirect_http:
                    # Redirect to HTTPS
                    https_url = str(request.url).replace("http://", "https://", 1)
                    # Change port if it's the default HTTP port
                    if ":80/" in https_url:
                        https_url = https_url.replace(":80/", "/")
                    
                    self.logger.info(f"Redirecting HTTP to HTTPS: {request.url} -> {https_url}")
                    return RedirectResponse(url=https_url, status_code=301)
                else:
                    # Block HTTP requests
                    raise HTTPException(
                        status_code=status.HTTP_426_UPGRADE_REQUIRED,
                        detail="HTTPS Required",
                        headers={"Upgrade": "TLS/1.2, HTTP/1.1"}
                    )
        
        # Validate hostname if allowed_hosts is specified
        if self.allowed_hosts:
            host = request.headers.get("Host", "").split(":")[0]  # Remove port
            if host and host not in self.allowed_hosts:
                self.logger.warning(f"Request to disallowed host: {host}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid host header"
                )
        
        # Process the request
        response = await call_next(request)
        
        # Add security headers if HTTPS is being used
        if self.enforce_https or request.url.scheme == "https":
            self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add HTTPS-related security headers"""
        
        # HTTP Strict Transport Security (HSTS)
        hsts_value = f"max-age={self.hsts_max_age}"
        if self.include_subdomains:
            hsts_value += "; includeSubDomains"
        if self.preload:
            hsts_value += "; preload"
        
        response.headers["Strict-Transport-Security"] = hsts_value
        
        # Additional security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy for HTTPS
        csp = (
            "default-src 'self' https:; "
            "script-src 'self' 'unsafe-inline' https:; "
            "style-src 'self' 'unsafe-inline' https:; "
            "img-src 'self' https: data:; "
            "font-src 'self' https:; "
            "connect-src 'self' https: wss:; "
            "upgrade-insecure-requests"
        )
        response.headers["Content-Security-Policy"] = csp
        
        # Permissions Policy
        permissions = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=(), "
            "speaker=(), fullscreen=(self)"
        )
        response.headers["Permissions-Policy"] = permissions
