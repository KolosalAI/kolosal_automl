"""
Security Utilities and Helper Functions for kolosal AutoML

Provides utility functions for:
- Password generation and validation
- Token management
- Security configuration helpers
- Validation utilities
- Security constants

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import re
import secrets
import string
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import base64


# Security constants
MINIMUM_PASSWORD_LENGTH = 8
MAXIMUM_PASSWORD_LENGTH = 128
API_KEY_LENGTH = 32
JWT_SECRET_LENGTH = 64

# Password strength requirements
PASSWORD_PATTERNS = {
    'lowercase': re.compile(r'[a-z]'),
    'uppercase': re.compile(r'[A-Z]'),
    'digit': re.compile(r'\d'),
    'special': re.compile(r'[!@#$%^&*(),.?":{}|<>]'),
    'length': lambda x: len(x) >= MINIMUM_PASSWORD_LENGTH
}


def generate_secure_password(length: int = 16, 
                           include_symbols: bool = True,
                           exclude_ambiguous: bool = True) -> str:
    """
    Generate a cryptographically secure password
    
    Args:
        length: Password length (minimum 8)
        include_symbols: Include special characters
        exclude_ambiguous: Exclude ambiguous characters (0, O, l, 1, etc.)
        
    Returns:
        Secure password string
    """
    if length < MINIMUM_PASSWORD_LENGTH:
        length = MINIMUM_PASSWORD_LENGTH
    
    # Character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    symbols = "!@#$%^&*(),.?\":{}|<>"
    
    # Remove ambiguous characters if requested
    if exclude_ambiguous:
        lowercase = lowercase.replace('l', '')
        uppercase = uppercase.replace('O', '')
        digits = digits.replace('0', '').replace('1', '')
        symbols = symbols.replace('|', '').replace(':', '')
    
    # Build character pool
    char_pool = lowercase + uppercase + digits
    if include_symbols:
        char_pool += symbols
    
    # Ensure at least one character from each required set
    password = []
    password.append(secrets.choice(lowercase))
    password.append(secrets.choice(uppercase))
    password.append(secrets.choice(digits))
    
    if include_symbols:
        password.append(secrets.choice(symbols))
    
    # Fill remaining length
    remaining_length = length - len(password)
    for _ in range(remaining_length):
        password.append(secrets.choice(char_pool))
    
    # Shuffle the password
    secrets.SystemRandom().shuffle(password)
    
    return ''.join(password)


def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password strength against security requirements
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Length check
    if len(password) < MINIMUM_PASSWORD_LENGTH:
        issues.append(f"Password must be at least {MINIMUM_PASSWORD_LENGTH} characters long")
    
    if len(password) > MAXIMUM_PASSWORD_LENGTH:
        issues.append(f"Password must be no more than {MAXIMUM_PASSWORD_LENGTH} characters long")
    
    # Character type checks
    if not PASSWORD_PATTERNS['lowercase'].search(password):
        issues.append("Password must contain at least one lowercase letter")
    
    if not PASSWORD_PATTERNS['uppercase'].search(password):
        issues.append("Password must contain at least one uppercase letter")
    
    if not PASSWORD_PATTERNS['digit'].search(password):
        issues.append("Password must contain at least one digit")
    
    if not PASSWORD_PATTERNS['special'].search(password):
        issues.append("Password must contain at least one special character")
    
    # Common password checks
    common_passwords = [
        'password', '123456', 'password123', 'admin', 'qwerty',
        'letmein', 'welcome', 'monkey', '1234567890', 'password1'
    ]
    
    if password.lower() in common_passwords:
        issues.append("Password is too common")
    
    # Sequential character check
    if _has_sequential_chars(password):
        issues.append("Password should not contain sequential characters")
    
    # Repeated character check
    if _has_repeated_chars(password):
        issues.append("Password should not contain too many repeated characters")
    
    return len(issues) == 0, issues


def _has_sequential_chars(password: str, max_sequence: int = 3) -> bool:
    """Check for sequential characters"""
    for i in range(len(password) - max_sequence + 1):
        chars = password[i:i + max_sequence]
        
        # Check if characters are sequential (ascending or descending)
        if len(set(ord(c) for c in chars)) == len(chars):
            ascii_values = [ord(c) for c in chars]
            if (all(ascii_values[i] + 1 == ascii_values[i + 1] for i in range(len(ascii_values) - 1)) or
                all(ascii_values[i] - 1 == ascii_values[i + 1] for i in range(len(ascii_values) - 1))):
                return True
    
    return False


def _has_repeated_chars(password: str, max_repeat: int = 3) -> bool:
    """Check for repeated characters"""
    for i in range(len(password) - max_repeat + 1):
        if password[i] * max_repeat == password[i:i + max_repeat]:
            return True
    return False


def generate_secure_api_key(length: int = API_KEY_LENGTH, 
                          prefix: Optional[str] = None) -> str:
    """
    Generate a secure API key
    
    Args:
        length: Key length
        prefix: Optional prefix (e.g., 'sk_', 'pk_')
        
    Returns:
        Secure API key
    """
    # Use URL-safe base64 encoding for API keys
    key_bytes = secrets.token_bytes(length)
    key = base64.urlsafe_b64encode(key_bytes).decode('utf-8').rstrip('=')
    
    if prefix:
        key = f"{prefix}{key}"
    
    return key


def generate_jwt_secret(length: int = JWT_SECRET_LENGTH) -> str:
    """
    Generate a secure JWT secret
    
    Args:
        length: Secret length in bytes
        
    Returns:
        Base64 encoded secret
    """
    secret_bytes = secrets.token_bytes(length)
    return base64.b64encode(secret_bytes).decode('utf-8')


def generate_session_token(length: int = 32) -> str:
    """
    Generate a secure session token
    
    Args:
        length: Token length
        
    Returns:
        URL-safe session token
    """
    return secrets.token_urlsafe(length)


def hash_sensitive_data(data: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash sensitive data with salt
    
    Args:
        data: Data to hash
        salt: Optional salt (will be generated if not provided)
        
    Returns:
        Tuple of (hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Combine data and salt
    combined = f"{data}{salt}".encode('utf-8')
    
    # Hash with SHA-256
    hash_object = hashlib.sha256(combined)
    hash_hex = hash_object.hexdigest()
    
    return hash_hex, salt


def verify_sensitive_data(data: str, hash_value: str, salt: str) -> bool:
    """
    Verify sensitive data against hash
    
    Args:
        data: Original data
        hash_value: Expected hash
        salt: Salt used for hashing
        
    Returns:
        True if data matches hash
    """
    calculated_hash, _ = hash_sensitive_data(data, salt)
    return secrets.compare_digest(calculated_hash, hash_value)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for security
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace dangerous characters
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Remove path traversal attempts
    sanitized = sanitized.replace('..', '_')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:240] + ('.' + ext if ext else '')
    
    return sanitized


def validate_email(email: str) -> bool:
    """
    Validate email address format
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    if not email or '@' not in email:
        return False
    
    # Check for consecutive dots
    if '..' in email:
        return False
    
    # Check basic pattern
    pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    if not pattern.match(email):
        return False
    
    local, domain = email.split('@', 1)
    
    # Local part checks
    if not local or len(local) > 64:
        return False
    if local.startswith('.') or local.endswith('.'):
        return False
    
    # Domain part checks  
    if not domain or len(domain) > 255:
        return False
    if domain.startswith('.') or domain.endswith('.'):
        return False
    if domain.startswith('-') or domain.endswith('-'):
        return False
    
    return True


def validate_ip_address(ip: str) -> bool:
    """
    Validate IP address format
    
    Args:
        ip: IP address to validate
        
    Returns:
        True if valid IP address
    """
    try:
        import ipaddress
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def validate_url(url: str) -> bool:
    """
    Validate URL format
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL format
    """
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(pattern.match(url))


def get_entropy_score(data: str) -> float:
    """
    Calculate entropy score for password strength assessment
    
    Args:
        data: String to analyze
        
    Returns:
        Entropy score (higher is better)
    """
    if not data:
        return 0.0
    
    # Count character frequencies
    frequencies = {}
    for char in data:
        frequencies[char] = frequencies.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    length = len(data)
    
    for count in frequencies.values():
        probability = count / length
        if probability > 0:
            import math
            entropy -= probability * math.log2(probability)
    
    return entropy


def generate_nonce(length: int = 16) -> str:
    """
    Generate a cryptographic nonce
    
    Args:
        length: Nonce length in bytes
        
    Returns:
        Base64 encoded nonce
    """
    nonce_bytes = secrets.token_bytes(length)
    return base64.b64encode(nonce_bytes).decode('utf-8')


def time_constant_compare(a: str, b: str) -> bool:
    """
    Time-constant string comparison to prevent timing attacks
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal
    """
    return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


def mask_sensitive_data(data: str, mask_char: str = '*', 
                       visible_start: int = 2, visible_end: int = 2) -> str:
    """
    Mask sensitive data for logging/display
    
    Args:
        data: Sensitive data to mask
        mask_char: Character to use for masking
        visible_start: Number of characters to show at start
        visible_end: Number of characters to show at end
        
    Returns:
        Masked string
    """
    if len(data) <= visible_start + visible_end:
        return mask_char * len(data)
    
    start = data[:visible_start]
    end = data[-visible_end:] if visible_end > 0 else ''
    middle_length = len(data) - visible_start - visible_end
    middle = mask_char * middle_length
    
    return f"{start}{middle}{end}"


# Security configuration helpers
def get_security_headers_config() -> Dict[str, str]:
    """Get recommended security headers configuration"""
    return {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        'Cache-Control': 'no-store, no-cache, must-revalidate, private',
        'Pragma': 'no-cache',
    }


def get_csp_policy(mode: str = 'strict') -> str:
    """
    Get Content Security Policy configuration
    
    Args:
        mode: 'strict', 'moderate', or 'relaxed'
        
    Returns:
        CSP policy string
    """
    policies = {
        'strict': (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        ),
        'moderate': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'self'; "
            "base-uri 'self'"
        ),
        'relaxed': (
            "default-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "img-src * data:; "
            "font-src *; "
            "connect-src *"
        )
    }
    
    return policies.get(mode, policies['strict'])


def create_error_response(status_code: int, message: str) -> dict:
    """
    Create a standardized error response
    
    Args:
        status_code: HTTP status code
        message: Error message
        
    Returns:
        Standardized error response dictionary
    """
    class ErrorResponse:
        def __init__(self, status_code: int, message: str):
            self.status_code = status_code
            self.headers = {}
            self.body = {
                "error": {
                    "code": status_code,
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    return ErrorResponse(status_code, message)
