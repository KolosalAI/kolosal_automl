#!/usr/bin/env python3
"""
Genta AutoML API Startup Script with Enhanced Security

This script provides a convenient way to start the Genta AutoML API server
with proper configuration, environment setup, and comprehensive security.

Features:
- Environment-based security configuration
- TLS/HTTPS setup
- Security validation
- Development/production modes

Usage:
    python start_api.py
    # or
    uv run python start_api.py
    
Environment Variables:
    SECURITY_ENV: Security environment (development/testing/staging/production)
    API_ENV: API environment
    ENABLE_HTTPS: Enable HTTPS (true/false)
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_security_environment():
    """Setup security environment variables"""
    
    # Determine security level from environment
    security_env = os.getenv("SECURITY_ENV", "development").lower()
    api_env = os.getenv("API_ENV", "development").lower()
    
    # Security configuration based on environment
    if security_env in ["production", "prod"]:
        security_config = {
            "SECURITY_ENV": "production",
            "SECURITY_ENFORCE_HTTPS": "true",
            "SECURITY_REQUIRE_API_KEY": "true",
            "SECURITY_ENABLE_RATE_LIMITING": "true",
            "SECURITY_RATE_LIMIT_REQUESTS": "50",
            "SECURITY_ENABLE_JWT": "true",
            "SECURITY_JWT_EXPIRY_HOURS": "1",
            "SECURITY_ENABLE_AUDIT_LOGGING": "true",
            "SECURITY_HSTS_MAX_AGE": "31536000",
            "SECURITY_ALLOWED_ORIGINS": "",  # Must be explicitly set
            "API_PORT": "8000",
        }
    elif security_env in ["staging", "stage"]:
        security_config = {
            "SECURITY_ENV": "staging",
            "SECURITY_ENFORCE_HTTPS": "true",
            "SECURITY_REQUIRE_API_KEY": "true",
            "SECURITY_ENABLE_RATE_LIMITING": "true",
            "SECURITY_RATE_LIMIT_REQUESTS": "75",
            "SECURITY_ENABLE_JWT": "true",
            "SECURITY_ENABLE_AUDIT_LOGGING": "true",
            "SECURITY_ALLOWED_ORIGINS": "https://staging.example.com",
            "API_PORT": "8000",
        }
    elif security_env in ["testing", "test"]:
        security_config = {
            "SECURITY_ENV": "testing",
            "SECURITY_ENFORCE_HTTPS": "false",
            "SECURITY_REQUIRE_API_KEY": "false",
            "SECURITY_ENABLE_RATE_LIMITING": "false",
            "SECURITY_ENABLE_JWT": "false",
            "SECURITY_ENABLE_AUDIT_LOGGING": "false",
            "SECURITY_ALLOWED_ORIGINS": "*",
            "API_PORT": "8000",
        }
    else:  # development
        security_config = {
            "SECURITY_ENV": "development",
            "SECURITY_ENFORCE_HTTPS": "false",
            "SECURITY_REQUIRE_API_KEY": "false",
            "SECURITY_ENABLE_RATE_LIMITING": "false",
            "SECURITY_ENABLE_JWT": "false",
            "SECURITY_ENABLE_AUDIT_LOGGING": "true",
            "SECURITY_ALLOWED_ORIGINS": "*",
            "API_PORT": "8000",
        }
    
    # Set security environment variables if not already set
    for key, value in security_config.items():
        if key not in os.environ:
            os.environ[key] = value
    
    logger.info(f"Security environment configured: {security_env}")
    
    # Validate production security
    if security_env == "production":
        validate_production_security()


def validate_production_security():
    """Validate production security configuration"""
    
    issues = []
    
    # Check for required production settings
    if os.getenv("SECURITY_ALLOWED_ORIGINS") in ["", "*"]:
        issues.append("SECURITY_ALLOWED_ORIGINS must be explicitly configured in production")
    
    if not os.getenv("API_KEYS") and not os.path.exists("secrets"):
        issues.append("API keys must be configured in production")
    
    if not os.getenv("JWT_SECRET") and not os.path.exists("secrets"):
        issues.append("JWT secret must be configured in production")
    
    # Check for TLS certificates if HTTPS is enabled
    if os.getenv("SECURITY_ENFORCE_HTTPS", "").lower() == "true":
        cert_paths = [
            os.getenv("TLS_CERT_PATH", "certs/server.crt"),
            os.getenv("TLS_KEY_PATH", "certs/server.key")
        ]
        
        for cert_path in cert_paths:
            if not os.path.exists(cert_path):
                issues.append(f"TLS certificate not found: {cert_path}")
    
    if issues:
        logger.error("Production security validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        
        # Allow override with environment variable
        if os.getenv("IGNORE_SECURITY_WARNINGS", "").lower() != "true":
            logger.error("Set IGNORE_SECURITY_WARNINGS=true to bypass these checks")
            sys.exit(1)
        else:
            logger.warning("Security warnings ignored - not recommended for production")


def main():
    """Main function to start the API server."""
    
    # Setup security environment
    setup_security_environment()
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Set environment variables if not already set
    env_vars = {
        "API_ENV": "development",
        "API_DEBUG": "True",
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000",
        "API_WORKERS": "1",
        "REQUIRE_API_KEY": "False",
        "API_KEYS": "dev_key,test_key",
    }
    
    # Update environment
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Change to project directory
    os.chdir(project_root)
    
    # Start the API server
    print("Starting Genta AutoML API...")
    print(f"Environment: {os.environ.get('API_ENV', 'development')}")
    print(f"Debug mode: {os.environ.get('API_DEBUG', 'False')}")
    print(f"Host: {os.environ.get('API_HOST', '0.0.0.0')}")
    print(f"Port: {os.environ.get('API_PORT', '8000')}")
    print(f"API Key required: {os.environ.get('REQUIRE_API_KEY', 'False')}")
    print()
    print("API will be available at:")
    print(f"  - Health check: http://{os.environ.get('API_HOST', '0.0.0.0')}:{os.environ.get('API_PORT', '8000')}/health")
    print(f"  - Documentation: http://{os.environ.get('API_HOST', '0.0.0.0')}:{os.environ.get('API_PORT', '8000')}/docs")
    print(f"  - API root: http://{os.environ.get('API_HOST', '0.0.0.0')}:{os.environ.get('API_PORT', '8000')}/")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run the API server
        subprocess.run([
            sys.executable, 
            "modules/api/app.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\nShutting down API server...")
    except subprocess.CalledProcessError as e:
        print(f"Error starting API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
