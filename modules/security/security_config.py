"""
Security Configuration and Environment Manager for kolosal AutoML

Manages security configurations with:
- Environment-based settings
- Secure defaults
- Configuration validation
- Runtime security checks
- Secret management integration

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
from dataclasses import dataclass, field
from enum import Enum


class SecurityLevel(Enum):
    """Security levels for different environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SecurityEnvironment:
    """Security environment configuration"""
    
    # Environment settings
    security_level: SecurityLevel = SecurityLevel.DEVELOPMENT
    debug_mode: bool = False
    
    # API Security
    require_api_key: bool = True
    api_key_rotation_enabled: bool = True
    api_key_min_length: int = 32
    
    # Authentication
    enable_jwt: bool = False
    jwt_expiry_hours: int = 24
    enable_refresh_tokens: bool = True
    
    # Rate Limiting
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    rate_limit_strategy: str = "sliding_window"
    
    # TLS/HTTPS
    enforce_https: bool = True
    redirect_http: bool = True
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000
    
    # Request Security
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    enable_input_validation: bool = True
    enable_sql_injection_protection: bool = True
    enable_xss_protection: bool = True
    
    # CORS Settings
    allowed_origins: List[str] = field(default_factory=lambda: ["https://localhost:3000"])
    allow_credentials: bool = False
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allowed_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Security Headers
    enable_security_headers: bool = True
    enable_csp: bool = True
    csp_policy: str = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    
    # Monitoring and Logging
    enable_audit_logging: bool = True
    log_security_events: bool = True
    alert_on_threats: bool = True
    
    # IP and Access Control
    enable_ip_blocking: bool = True
    blocked_ips: List[str] = field(default_factory=list)
    enable_geo_blocking: bool = False
    blocked_countries: List[str] = field(default_factory=list)
    
    # Session Management
    session_timeout: int = 3600  # 1 hour
    enable_session_rotation: bool = True
    concurrent_sessions_limit: int = 5
    
    @classmethod
    def from_environment(cls, env_name: Optional[str] = None) -> 'SecurityEnvironment':
        """Create security environment from environment variables"""
        
        # Determine environment
        env_name = env_name or os.getenv("SECURITY_ENV", "development")
        try:
            security_level = SecurityLevel(env_name.lower())
        except ValueError:
            security_level = SecurityLevel.DEVELOPMENT
        
        # Base configuration based on security level
        if security_level == SecurityLevel.PRODUCTION:
            config = cls._production_config()
        elif security_level == SecurityLevel.STAGING:
            config = cls._staging_config()
        elif security_level == SecurityLevel.TESTING:
            config = cls._testing_config()
        else:
            config = cls._development_config()
        
        # Override with environment variables
        config._apply_env_overrides()
        
        return config
    
    @classmethod
    def _production_config(cls) -> 'SecurityEnvironment':
        """Production security configuration"""
        return cls(
            security_level=SecurityLevel.PRODUCTION,
            debug_mode=False,
            require_api_key=True,
            api_key_rotation_enabled=True,
            enable_jwt=True,
            jwt_expiry_hours=1,  # Shorter expiry in production
            enable_rate_limiting=True,
            rate_limit_requests=50,  # Stricter rate limiting
            rate_limit_window=60,
            enforce_https=True,
            redirect_http=True,
            hsts_enabled=True,
            allowed_origins=[],  # Must be explicitly configured
            allow_credentials=False,
            enable_audit_logging=True,
            log_security_events=True,
            alert_on_threats=True,
            enable_ip_blocking=True,
            session_timeout=1800,  # 30 minutes
            concurrent_sessions_limit=3
        )
    
    @classmethod
    def _staging_config(cls) -> 'SecurityEnvironment':
        """Staging security configuration"""
        return cls(
            security_level=SecurityLevel.STAGING,
            debug_mode=False,
            require_api_key=True,
            enable_jwt=True,
            jwt_expiry_hours=4,
            enable_rate_limiting=True,
            rate_limit_requests=75,
            enforce_https=True,
            redirect_http=True,
            allowed_origins=["https://staging.example.com"],
            enable_audit_logging=True,
            log_security_events=True
        )
    
    @classmethod
    def _testing_config(cls) -> 'SecurityEnvironment':
        """Testing security configuration"""
        return cls(
            security_level=SecurityLevel.TESTING,
            debug_mode=True,
            require_api_key=False,  # Simplified for testing
            enable_jwt=False,
            enable_rate_limiting=False,
            enforce_https=False,
            redirect_http=False,
            allowed_origins=["*"],
            enable_audit_logging=False,
            log_security_events=False
        )
    
    @classmethod
    def _development_config(cls) -> 'SecurityEnvironment':
        """Development security configuration"""
        return cls(
            security_level=SecurityLevel.DEVELOPMENT,
            debug_mode=True,
            require_api_key=False,
            enable_jwt=False,
            enable_rate_limiting=False,
            enforce_https=False,
            redirect_http=False,
            allowed_origins=["*"],
            enable_audit_logging=True,
            log_security_events=True
        )
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        
        # Boolean environment variables
        bool_vars = {
            "SECURITY_REQUIRE_API_KEY": "require_api_key",
            "SECURITY_ENABLE_JWT": "enable_jwt",
            "SECURITY_ENABLE_RATE_LIMITING": "enable_rate_limiting",
            "SECURITY_ENFORCE_HTTPS": "enforce_https",
            "SECURITY_REDIRECT_HTTP": "redirect_http",
            "SECURITY_ENABLE_HSTS": "hsts_enabled",
            "SECURITY_ENABLE_AUDIT_LOGGING": "enable_audit_logging",
            "SECURITY_DEBUG_MODE": "debug_mode",
        }
        
        for env_var, attr_name in bool_vars.items():
            value = os.getenv(env_var)
            if value is not None:
                setattr(self, attr_name, value.lower() in ("true", "1", "yes", "on"))
        
        # Integer environment variables
        int_vars = {
            "SECURITY_RATE_LIMIT_REQUESTS": "rate_limit_requests",
            "SECURITY_RATE_LIMIT_WINDOW": "rate_limit_window",
            "SECURITY_JWT_EXPIRY_HOURS": "jwt_expiry_hours",
            "SECURITY_HSTS_MAX_AGE": "hsts_max_age",
            "SECURITY_MAX_REQUEST_SIZE": "max_request_size",
            "SECURITY_SESSION_TIMEOUT": "session_timeout",
        }
        
        for env_var, attr_name in int_vars.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    setattr(self, attr_name, int(value))
                except ValueError:
                    logging.warning(f"Invalid integer value for {env_var}: {value}")
        
        # String environment variables
        string_vars = {
            "SECURITY_RATE_LIMIT_STRATEGY": "rate_limit_strategy",
            "SECURITY_CSP_POLICY": "csp_policy",
        }
        
        for env_var, attr_name in string_vars.items():
            value = os.getenv(env_var)
            if value is not None:
                setattr(self, attr_name, value)
        
        # List environment variables
        list_vars = {
            "SECURITY_ALLOWED_ORIGINS": "allowed_origins",
            "SECURITY_BLOCKED_IPS": "blocked_ips",
            "SECURITY_BLOCKED_COUNTRIES": "blocked_countries",
            "SECURITY_ALLOWED_METHODS": "allowed_methods",
        }
        
        for env_var, attr_name in list_vars.items():
            value = os.getenv(env_var)
            if value is not None:
                # Parse comma-separated values
                parsed_list = [item.strip() for item in value.split(",") if item.strip()]
                setattr(self, attr_name, parsed_list)
    
    def validate(self) -> List[str]:
        """Validate security configuration and return any issues"""
        issues = []
        
        # Production-specific validations
        if self.security_level == SecurityLevel.PRODUCTION:
            if not self.require_api_key:
                issues.append("API key authentication should be required in production")
            
            if not self.enforce_https:
                issues.append("HTTPS should be enforced in production")
            
            if not self.enable_rate_limiting:
                issues.append("Rate limiting should be enabled in production")
            
            if "*" in self.allowed_origins:
                issues.append("Wildcard CORS origins should not be used in production")
            
            if self.debug_mode:
                issues.append("Debug mode should be disabled in production")
        
        # General validations
        if self.rate_limit_requests <= 0:
            issues.append("Rate limit requests must be positive")
        
        if self.rate_limit_window <= 0:
            issues.append("Rate limit window must be positive")
        
        if self.jwt_expiry_hours <= 0:
            issues.append("JWT expiry hours must be positive")
        
        if self.max_request_size <= 0:
            issues.append("Max request size must be positive")
        
        return issues
    
    def get_api_keys(self) -> List[str]:
        """Get API keys from secrets manager or environment"""
        from .secrets_manager import get_secrets_manager, SecretType
        secrets_manager = get_secrets_manager()
        
        # Try to get from secrets manager first
        api_keys = []
        
        # Check for stored API keys
        stored_keys = secrets_manager.list_secrets(secret_type=SecretType.API_KEY)
        for metadata in stored_keys:
            key_value = secrets_manager.get_secret(metadata.secret_id)
            if key_value:
                api_keys.append(key_value)
        
        # Fallback to environment variable
        if not api_keys:
            env_keys = os.getenv("API_KEYS", "")
            if env_keys:
                api_keys = [key.strip() for key in env_keys.split(",") if key.strip()]
        
        # Generate default key if none found and required
        if not api_keys and self.require_api_key:
            from .enhanced_security import generate_secure_api_key
            default_key = generate_secure_api_key()
            
            # Store in secrets manager
            secrets_manager.store_secret(
                secret_id="default_api_key",
                secret=default_key,
                secret_type=SecretType.API_KEY,
                tags=["auto-generated", "default"]
            )
            
            api_keys = [default_key]
            logging.warning(f"Generated default API key. Please configure proper keys.")
        
        return api_keys
    
    def get_jwt_secret(self) -> Optional[str]:
        """Get JWT secret from secrets manager or environment"""
        if not self.enable_jwt:
            return None
        
        from .secrets_manager import get_secrets_manager, SecretType
        secrets_manager = get_secrets_manager()
        
        # Try to get from secrets manager
        jwt_secret = secrets_manager.get_secret("jwt_secret")
        
        # Fallback to environment variable
        if not jwt_secret:
            jwt_secret = os.getenv("JWT_SECRET")
        
        # Generate if needed
        if not jwt_secret:
            import secrets
            jwt_secret = secrets.token_urlsafe(64)
            
            # Store in secrets manager
            secrets_manager.store_secret(
                secret_id="jwt_secret",
                secret=jwt_secret,
                secret_type=SecretType.JWT_SECRET,
                tags=["auto-generated"]
            )
            
            logging.warning("Generated default JWT secret. Please configure a proper secret.")
        
        return jwt_secret
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, SecurityLevel):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    def save_to_file(self, file_path: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        
        # Remove sensitive information
        sensitive_keys = ["api_keys", "jwt_secret"]
        for key in sensitive_keys:
            config_dict.pop(key, None)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'SecurityEnvironment':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert security_level back to enum
        if 'security_level' in config_dict:
            config_dict['security_level'] = SecurityLevel(config_dict['security_level'])
        
        return cls(**config_dict)


# Global security environment instance
_security_env: Optional[SecurityEnvironment] = None


def get_security_environment(env_name: Optional[str] = None, 
                           force_reload: bool = False) -> SecurityEnvironment:
    """
    Get or create the global security environment instance
    
    Args:
        env_name: Environment name to load
        force_reload: Force reloading the environment
        
    Returns:
        SecurityEnvironment instance
    """
    global _security_env
    
    if _security_env is None or force_reload:
        _security_env = SecurityEnvironment.from_environment(env_name)
        
        # Validate configuration
        issues = _security_env.validate()
        if issues:
            logger = logging.getLogger(__name__)
            logger.warning(f"Security configuration issues found: {issues}")
    
    return _security_env


def setup_secure_environment(env_name: Optional[str] = None) -> SecurityEnvironment:
    """
    Setup and configure secure environment
    
    Args:
        env_name: Environment name
        
    Returns:
        Configured SecurityEnvironment
    """
    env = get_security_environment(env_name)
    
    # Setup logging for security events
    if env.log_security_events:
        security_logger = logging.getLogger("kolosal_security")
        if not security_logger.handlers:
            handler = logging.FileHandler("security_events.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            security_logger.addHandler(handler)
            security_logger.setLevel(logging.INFO)
    
    return env


# Alias for backward compatibility and convenience
SecurityConfig = SecurityEnvironment


# Global convenience functions for external compatibility
def get_secrets_manager():
    """Global function to get secrets manager - for compatibility with tests"""
    try:
        from .secrets_manager import get_secrets_manager as _get_secrets_manager
        return _get_secrets_manager()
    except ImportError:
        return None
