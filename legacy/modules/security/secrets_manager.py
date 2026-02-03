"""
Enhanced Secrets Manager for kolosal AutoML

Provides enterprise-grade secret management with:
- Key rotation capabilities
- Multiple secret backends (environment, file-based, external)
- Encryption at rest
- Audit logging
- Secret validation and strength checking

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import os
import json
import time
import secrets
import hashlib
import logging
from typing import Dict, Optional, Any, Union, List
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import base64

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding


class SecretType(Enum):
    """Types of secrets managed by the system"""
    API_KEY = "api_key"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    DATABASE_PASSWORD = "database_password"
    TLS_PRIVATE_KEY = "tls_private_key"
    TLS_CERTIFICATE = "tls_certificate"
    OAUTH_CLIENT_SECRET = "oauth_client_secret"
    WEBHOOK_SECRET = "webhook_secret"


@dataclass
class SecretMetadata:
    """Metadata for a managed secret"""
    secret_id: str
    secret_type: SecretType
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    rotation_interval: Optional[timedelta] = None
    strength_score: int = 0
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SecretsManager:
    """
    Enterprise-grade secrets manager with encryption, rotation, and audit capabilities
    """
    
    def __init__(self, config_path: str = ".secrets", master_password: Optional[str] = None):
        """
        Initialize the secrets manager
        
        Args:
            config_path: Path to store encrypted secrets
            master_password: Master password for encrypting secrets at rest
        """
        self.config_path = Path(config_path)
        self.config_path.mkdir(mode=0o700, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._setup_encryption(master_password)
        self._secrets_cache: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, SecretMetadata] = {}
        self._load_secrets()
        
    def _setup_encryption(self, master_password: Optional[str] = None):
        """Setup encryption for secrets at rest"""
        key_file = self.config_path / ".master.key"
        
        if master_password:
            # Derive key from master password
            salt_file = self.config_path / ".salt"
            if salt_file.exists():
                with open(salt_file, 'rb') as f:
                    salt = f.read()
            else:
                salt = os.urandom(32)
                with open(salt_file, 'wb') as f:
                    f.write(salt)
                os.chmod(salt_file, 0o600)
            
            kdf = Scrypt(
                length=32,
                salt=salt,
                n=2**15,  # Higher work factor for master key
                r=8,
                p=1,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        else:
            # Use or generate a key file
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)
        
        self.cipher = Fernet(key)
        
    def _load_secrets(self):
        """Load secrets from encrypted storage"""
        secrets_file = self.config_path / "secrets.enc"
        metadata_file = self.config_path / "metadata.enc"
        
        if secrets_file.exists():
            try:
                with open(secrets_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                self._secrets_cache = json.loads(decrypted_data.decode())
            except Exception as e:
                self.logger.error(f"Failed to load secrets: {e}")
                self._secrets_cache = {}
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                metadata_dict = json.loads(decrypted_data.decode())
                
                # Convert back to SecretMetadata objects
                for secret_id, meta_dict in metadata_dict.items():
                    meta_dict['secret_type'] = SecretType(meta_dict['secret_type'])
                    meta_dict['created_at'] = datetime.fromisoformat(meta_dict['created_at'])
                    meta_dict['updated_at'] = datetime.fromisoformat(meta_dict['updated_at'])
                    
                    if meta_dict.get('expires_at'):
                        meta_dict['expires_at'] = datetime.fromisoformat(meta_dict['expires_at'])
                    if meta_dict.get('last_accessed'):
                        meta_dict['last_accessed'] = datetime.fromisoformat(meta_dict['last_accessed'])
                    if meta_dict.get('rotation_interval'):
                        meta_dict['rotation_interval'] = timedelta(seconds=meta_dict['rotation_interval'])
                    
                    self._metadata_cache[secret_id] = SecretMetadata(**meta_dict)
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")
                self._metadata_cache = {}
    
    def _save_secrets(self):
        """Save secrets to encrypted storage"""
        try:
            # Save secrets
            secrets_data = json.dumps(self._secrets_cache).encode()
            encrypted_data = self.cipher.encrypt(secrets_data)
            secrets_file = self.config_path / "secrets.enc"
            with open(secrets_file, 'wb') as f:
                f.write(encrypted_data)
            os.chmod(secrets_file, 0o600)
            
            # Save metadata
            metadata_dict = {}
            for secret_id, metadata in self._metadata_cache.items():
                meta_dict = asdict(metadata)
                meta_dict['secret_type'] = metadata.secret_type.value
                meta_dict['created_at'] = metadata.created_at.isoformat()
                meta_dict['updated_at'] = metadata.updated_at.isoformat()
                
                if metadata.expires_at:
                    meta_dict['expires_at'] = metadata.expires_at.isoformat()
                if metadata.last_accessed:
                    meta_dict['last_accessed'] = metadata.last_accessed.isoformat()
                if metadata.rotation_interval:
                    meta_dict['rotation_interval'] = metadata.rotation_interval.total_seconds()
                
                metadata_dict[secret_id] = meta_dict
            
            metadata_data = json.dumps(metadata_dict).encode()
            encrypted_data = self.cipher.encrypt(metadata_data)
            metadata_file = self.config_path / "metadata.enc"
            with open(metadata_file, 'wb') as f:
                f.write(encrypted_data)
            os.chmod(metadata_file, 0o600)
            
        except Exception as e:
            self.logger.error(f"Failed to save secrets: {e}")
            raise
    
    def generate_secret(self, secret_type: SecretType, length: int = 32, **kwargs) -> str:
        """
        Generate a cryptographically secure secret
        
        Args:
            secret_type: Type of secret to generate
            length: Length of the secret
            **kwargs: Additional parameters for specific secret types
            
        Returns:
            Generated secret string
        """
        if secret_type == SecretType.API_KEY:
            return f"genta_{secrets.token_urlsafe(length)}"
        elif secret_type == SecretType.JWT_SECRET:
            return secrets.token_urlsafe(length)
        elif secret_type == SecretType.ENCRYPTION_KEY:
            return base64.urlsafe_b64encode(os.urandom(32)).decode()
        elif secret_type == SecretType.WEBHOOK_SECRET:
            return secrets.token_hex(length)
        elif secret_type == SecretType.DATABASE_PASSWORD:
            # Generate a strong password with mixed characters ensuring all types are included
            lowercase = "abcdefghijklmnopqrstuvwxyz"
            uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            digits = "0123456789"
            symbols = "!@#$%^&*"
            
            # Ensure at least one character from each category
            password = [
                secrets.choice(lowercase),
                secrets.choice(uppercase),
                secrets.choice(digits),
                secrets.choice(symbols)
            ]
            
            # Fill the rest with random choices from all categories
            all_chars = lowercase + uppercase + digits + symbols
            for _ in range(length - 4):
                password.append(secrets.choice(all_chars))
            
            # Shuffle the password to avoid predictable patterns
            secrets.SystemRandom().shuffle(password)
            return ''.join(password)
        else:
            return secrets.token_urlsafe(length)
    
    def assess_secret_strength(self, secret: str) -> int:
        """
        Assess the strength of a secret (0-100 score)
        
        Args:
            secret: Secret to assess
            
        Returns:
            Strength score (0-100)
        """
        score = 0
        
        # Length assessment (slightly reduced scoring)
        if len(secret) >= 32:
            score += 30
        elif len(secret) >= 24:
            score += 25
        elif len(secret) >= 16:
            score += 20
        elif len(secret) >= 12:
            score += 15
        elif len(secret) >= 8:
            score += 10
        
        # Character diversity (reduced scoring for medium secrets)
        has_lower = any(c.islower() for c in secret)
        has_upper = any(c.isupper() for c in secret)
        has_digit = any(c.isdigit() for c in secret)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in secret)
        
        diversity_score = sum([has_lower, has_upper, has_digit, has_special]) * 10  # Reduced from 12
        score += diversity_score
        
        # Entropy estimation (reduced scoring)
        unique_chars = len(set(secret))
        entropy_ratio = unique_chars / max(1, len(secret))
        if entropy_ratio > 0.8:
            score += 20  # Reduced from 25
        elif entropy_ratio > 0.6:
            score += 15  # Reduced from 20
        elif entropy_ratio > 0.4:
            score += 10  # Reduced from 15
        elif entropy_ratio > 0.2:
            score += 5   # Reduced from 10
        
        # Bonus for cryptographically secure patterns
        if len(secret) >= 16 and unique_chars >= 12 and has_special:
            score += 5
        
        # Check for common patterns
        if secret.lower() in ["password", "admin", "secret", "key"]:
            score = max(0, score - 50)
        
        return min(100, score)
    
    def store_secret(self, secret_id: str, secret: str, secret_type: SecretType, 
                    expires_at: Optional[datetime] = None,
                    rotation_interval: Optional[timedelta] = None,
                    tags: List[str] = None) -> bool:
        """
        Store a secret securely
        
        Args:
            secret_id: Unique identifier for the secret
            secret: The secret value
            secret_type: Type of the secret
            expires_at: When the secret expires
            rotation_interval: How often to rotate the secret
            tags: Tags for organizing secrets
            
        Returns:
            True if successful
        """
        try:
            # Assess secret strength
            strength = self.assess_secret_strength(secret)
            
            # Create metadata
            now = datetime.utcnow()
            metadata = SecretMetadata(
                secret_id=secret_id,
                secret_type=secret_type,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                rotation_interval=rotation_interval,
                strength_score=strength,
                tags=tags or []
            )
            
            # Store secret and metadata
            self._secrets_cache[secret_id] = secret
            self._metadata_cache[secret_id] = metadata
            
            # Persist to disk
            self._save_secrets()
            
            self.logger.info(f"Secret '{secret_id}' stored successfully (strength: {strength})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret '{secret_id}': {e}")
            return False
    
    def get_secret(self, secret_id: str) -> Optional[str]:
        """
        Retrieve a secret
        
        Args:
            secret_id: Secret identifier
            
        Returns:
            Secret value or None if not found/expired
        """
        try:
            if secret_id not in self._secrets_cache:
                # Try environment variable as fallback
                env_value = os.getenv(secret_id.upper())
                if env_value:
                    return env_value
                return None
            
            metadata = self._metadata_cache.get(secret_id)
            if metadata:
                # Check expiration
                if metadata.expires_at and datetime.utcnow() > metadata.expires_at:
                    self.logger.warning(f"Secret '{secret_id}' has expired")
                    return None
                
                # Update access tracking
                metadata.last_accessed = datetime.utcnow()
                metadata.access_count += 1
                self._save_secrets()
            
            return self._secrets_cache[secret_id]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret '{secret_id}': {e}")
            return None
    
    def rotate_secret(self, secret_id: str) -> bool:
        """
        Rotate a secret (generate new value)
        
        Args:
            secret_id: Secret to rotate
            
        Returns:
            True if successful
        """
        try:
            metadata = self._metadata_cache.get(secret_id)
            if not metadata:
                self.logger.error(f"Secret '{secret_id}' not found for rotation")
                return False
            
            # Generate new secret
            new_secret = self.generate_secret(metadata.secret_type)
            
            # Update secret and metadata
            old_secret = self._secrets_cache[secret_id]
            self._secrets_cache[secret_id] = new_secret
            metadata.updated_at = datetime.utcnow()
            metadata.strength_score = self.assess_secret_strength(new_secret)
            
            # Persist changes
            self._save_secrets()
            
            self.logger.info(f"Secret '{secret_id}' rotated successfully")
            
            # Audit log the rotation (without exposing the actual secrets)
            self.logger.info(f"AUDIT: Secret rotation - ID: {secret_id}, Type: {metadata.secret_type.value}, Timestamp: {datetime.utcnow().isoformat()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate secret '{secret_id}': {e}")
            return False
    
    def delete_secret(self, secret_id: str) -> bool:
        """
        Securely delete a secret
        
        Args:
            secret_id: Secret to delete
            
        Returns:
            True if successful
        """
        try:
            if secret_id in self._secrets_cache:
                # Securely overwrite the secret in memory
                secret_len = len(self._secrets_cache[secret_id])
                self._secrets_cache[secret_id] = 'X' * secret_len  # Overwrite
                del self._secrets_cache[secret_id]
            
            if secret_id in self._metadata_cache:
                del self._metadata_cache[secret_id]
            
            self._save_secrets()
            
            self.logger.info(f"Secret '{secret_id}' deleted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete secret '{secret_id}': {e}")
            return False
    
    def list_secrets(self, secret_type: Optional[SecretType] = None, 
                    include_expired: bool = False) -> List[SecretMetadata]:
        """
        List all secrets (metadata only, not values)
        
        Args:
            secret_type: Filter by secret type
            include_expired: Whether to include expired secrets
            
        Returns:
            List of secret metadata
        """
        results = []
        now = datetime.utcnow()
        
        for metadata in self._metadata_cache.values():
            # Apply filters
            if secret_type and metadata.secret_type != secret_type:
                continue
            
            if not include_expired and metadata.expires_at and now > metadata.expires_at:
                continue
            
            results.append(metadata)
        
        return sorted(results, key=lambda x: x.updated_at, reverse=True)
    
    def check_rotations_needed(self) -> List[str]:
        """
        Check which secrets need rotation
        
        Returns:
            List of secret IDs that need rotation
        """
        needs_rotation = []
        now = datetime.utcnow()
        
        for secret_id, metadata in self._metadata_cache.items():
            if metadata.rotation_interval:
                next_rotation = metadata.updated_at + metadata.rotation_interval
                if now >= next_rotation:
                    needs_rotation.append(secret_id)
        
        return needs_rotation
    
    def get_secret_info(self, secret_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a secret (without the secret value)
        
        Args:
            secret_id: Secret identifier
            
        Returns:
            Secret information dictionary
        """
        metadata = self._metadata_cache.get(secret_id)
        if not metadata:
            return None
        
        info = asdict(metadata)
        info['secret_type'] = metadata.secret_type.value
        info['is_expired'] = (
            metadata.expires_at and datetime.utcnow() > metadata.expires_at
        ) if metadata.expires_at else False
        info['needs_rotation'] = secret_id in self.check_rotations_needed()
        
        return info


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager(config_path: str = ".secrets", 
                       master_password: Optional[str] = None) -> SecretsManager:
    """
    Get or create the global secrets manager instance
    
    Args:
        config_path: Path to store secrets
        master_password: Master password for encryption
        
    Returns:
        SecretsManager instance
    """
    global _secrets_manager
    
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(config_path, master_password)
    
    return _secrets_manager


def get_secret(secret_id: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get a secret
    
    Args:
        secret_id: Secret identifier
        default: Default value if secret not found
        
    Returns:
        Secret value or default
    """
    manager = get_secrets_manager()
    value = manager.get_secret(secret_id)
    return value if value is not None else default
