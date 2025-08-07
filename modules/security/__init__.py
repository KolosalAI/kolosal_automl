"""
Security Module for kolosal AutoML

This module provides comprehensive security features including:
- Enhanced security management
- TLS/SSL certificate management  
- Secrets management with encryption
- Security configuration and environment management
- Security utilities and helpers
- Security middleware components

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

from .security_config import SecurityEnvironment, SecurityConfig, get_security_environment
from .enhanced_security import AdvancedRateLimiter, EnhancedSecurityManager, EnhancedSecurityConfig
from .tls_manager import TLSManager, TLSConfig
from .secrets_manager import SecretsManager, SecretMetadata
from .security_utils import (
    generate_secure_password,
    validate_password_strength,
    generate_secure_api_key,
    generate_jwt_secret,
    hash_sensitive_data,
    verify_sensitive_data,
    security_wrapper,
    validate_file_upload,
    secure_data_processing,
    create_security_headers,
    get_auth_config
)

__version__ = "0.2.0"
__author__ = "GitHub Copilot"

__all__ = [
    "SecurityEnvironment",
    "SecurityConfig",
    "get_security_environment",
    "AdvancedRateLimiter",
    "EnhancedSecurityManager",
    "EnhancedSecurityConfig",
    "TLSManager",
    "TLSConfig",
    "SecretsManager",
    "SecretMetadata",
    "generate_secure_password",
    "validate_password_strength",
    "generate_secure_api_key",
    "generate_jwt_secret",
    "hash_sensitive_data",
    "verify_sensitive_data",
    "security_wrapper",
    "validate_file_upload", 
    "secure_data_processing",
    "create_security_headers",
    "get_auth_config"
]
