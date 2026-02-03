"""
Unit tests for the enhanced secrets manager

Tests cover:
- Secret storage and retrieval
- Encryption and decryption
- Secret rotation
- Strength assessment
- Metadata management
- Error handling

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.security.secrets_manager import (
    SecretsManager,
    SecretType,
    SecretMetadata,
    get_secrets_manager,
    get_secret
)


class TestSecretsManager(unittest.TestCase):
    """Test cases for SecretsManager class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.secrets_manager = SecretsManager(config_path=self.test_dir)
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test SecretsManager initialization"""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertTrue(hasattr(self.secrets_manager, 'cipher'))
        self.assertTrue(hasattr(self.secrets_manager, '_secrets_cache'))
        self.assertTrue(hasattr(self.secrets_manager, '_metadata_cache'))
    
    def test_generate_secret_api_key(self):
        """Test API key generation"""
        secret = self.secrets_manager.generate_secret(SecretType.API_KEY, length=32)
        
        self.assertIsInstance(secret, str)
        self.assertTrue(secret.startswith('genta_'))
        self.assertGreaterEqual(len(secret), 32)
    
    def test_generate_secret_jwt(self):
        """Test JWT secret generation"""
        secret = self.secrets_manager.generate_secret(SecretType.JWT_SECRET, length=64)
        
        self.assertIsInstance(secret, str)
        self.assertGreaterEqual(len(secret), 64)
    
    def test_generate_secret_database_password(self):
        """Test database password generation"""
        secret = self.secrets_manager.generate_secret(SecretType.DATABASE_PASSWORD, length=16)
        
        self.assertIsInstance(secret, str)
        self.assertEqual(len(secret), 16)
        # Should contain mixed characters
        self.assertTrue(any(c.isupper() for c in secret))
        self.assertTrue(any(c.islower() for c in secret))
        self.assertTrue(any(c.isdigit() for c in secret))
    
    def test_assess_secret_strength_strong(self):
        """Test secret strength assessment for strong secrets"""
        strong_secret = "Th1s!sAVery$tr0ngP@ssw0rd123"
        score = self.secrets_manager.assess_secret_strength(strong_secret)
        
        self.assertGreaterEqual(score, 80)
    
    def test_assess_secret_strength_weak(self):
        """Test secret strength assessment for weak secrets"""
        weak_secret = "password"
        score = self.secrets_manager.assess_secret_strength(weak_secret)
        
        self.assertLessEqual(score, 30)
    
    def test_assess_secret_strength_medium(self):
        """Test secret strength assessment for medium secrets"""
        medium_secret = "Test123!"
        score = self.secrets_manager.assess_secret_strength(medium_secret)
        
        self.assertGreater(score, 30)
        self.assertLess(score, 80)
    
    def test_store_and_retrieve_secret(self):
        """Test storing and retrieving secrets"""
        secret_id = "test_secret"
        secret_value = "my_secret_value_123"
        
        # Store secret
        result = self.secrets_manager.store_secret(
            secret_id=secret_id,
            secret=secret_value,
            secret_type=SecretType.API_KEY,
            tags=["test", "api"]
        )
        
        self.assertTrue(result)
        
        # Retrieve secret
        retrieved = self.secrets_manager.get_secret(secret_id)
        self.assertEqual(retrieved, secret_value)
    
    def test_store_secret_with_expiration(self):
        """Test storing secret with expiration"""
        secret_id = "expiring_secret"
        secret_value = "temporary_secret"
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        # Store secret with expiration
        result = self.secrets_manager.store_secret(
            secret_id=secret_id,
            secret=secret_value,
            secret_type=SecretType.JWT_SECRET,
            expires_at=expires_at
        )
        
        self.assertTrue(result)
        
        # Should be retrievable before expiration
        retrieved = self.secrets_manager.get_secret(secret_id)
        self.assertEqual(retrieved, secret_value)
        
        # Test with expired secret
        metadata = self.secrets_manager._metadata_cache[secret_id]
        metadata.expires_at = datetime.utcnow() - timedelta(hours=1)
        
        retrieved_expired = self.secrets_manager.get_secret(secret_id)
        self.assertIsNone(retrieved_expired)
    
    def test_secret_rotation(self):
        """Test secret rotation"""
        secret_id = "rotatable_secret"
        original_secret = "original_value"
        
        # Store original secret
        self.secrets_manager.store_secret(
            secret_id=secret_id,
            secret=original_secret,
            secret_type=SecretType.API_KEY
        )
        
        # Rotate secret
        result = self.secrets_manager.rotate_secret(secret_id)
        self.assertTrue(result)
        
        # Verify secret changed
        new_secret = self.secrets_manager.get_secret(secret_id)
        self.assertNotEqual(new_secret, original_secret)
        self.assertIsNotNone(new_secret)
    
    def test_secret_deletion(self):
        """Test secret deletion"""
        secret_id = "deletable_secret"
        secret_value = "to_be_deleted"
        
        # Store secret
        self.secrets_manager.store_secret(
            secret_id=secret_id,
            secret=secret_value,
            secret_type=SecretType.API_KEY
        )
        
        # Verify it exists
        self.assertIsNotNone(self.secrets_manager.get_secret(secret_id))
        
        # Delete secret
        result = self.secrets_manager.delete_secret(secret_id)
        self.assertTrue(result)
        
        # Verify it's gone
        self.assertIsNone(self.secrets_manager.get_secret(secret_id))
    
    def test_list_secrets(self):
        """Test listing secrets"""
        # Store multiple secrets
        secrets = [
            ("api_key_1", SecretType.API_KEY),
            ("jwt_secret_1", SecretType.JWT_SECRET),
            ("api_key_2", SecretType.API_KEY),
        ]
        
        for secret_id, secret_type in secrets:
            self.secrets_manager.store_secret(
                secret_id=secret_id,
                secret=f"value_{secret_id}",
                secret_type=secret_type
            )
        
        # List all secrets
        all_secrets = self.secrets_manager.list_secrets()
        self.assertEqual(len(all_secrets), 3)
        
        # List API keys only
        api_keys = self.secrets_manager.list_secrets(secret_type=SecretType.API_KEY)
        self.assertEqual(len(api_keys), 2)
        
        # List JWT secrets only
        jwt_secrets = self.secrets_manager.list_secrets(secret_type=SecretType.JWT_SECRET)
        self.assertEqual(len(jwt_secrets), 1)
    
    def test_check_rotations_needed(self):
        """Test checking which secrets need rotation"""
        secret_id = "rotation_test"
        
        # Store secret with rotation interval
        self.secrets_manager.store_secret(
            secret_id=secret_id,
            secret="test_value",
            secret_type=SecretType.API_KEY,
            rotation_interval=timedelta(hours=1)
        )
        
        # Should not need rotation immediately
        needs_rotation = self.secrets_manager.check_rotations_needed()
        self.assertNotIn(secret_id, needs_rotation)
        
        # Simulate time passing
        metadata = self.secrets_manager._metadata_cache[secret_id]
        metadata.updated_at = datetime.utcnow() - timedelta(hours=2)
        
        # Should need rotation now
        needs_rotation = self.secrets_manager.check_rotations_needed()
        self.assertIn(secret_id, needs_rotation)
    
    def test_get_secret_info(self):
        """Test getting secret information"""
        secret_id = "info_test"
        
        self.secrets_manager.store_secret(
            secret_id=secret_id,
            secret="test_value",
            secret_type=SecretType.API_KEY,
            tags=["test", "info"]
        )
        
        info = self.secrets_manager.get_secret_info(secret_id)
        
        self.assertIsNotNone(info)
        self.assertEqual(info['secret_id'], secret_id)
        self.assertEqual(info['secret_type'], SecretType.API_KEY.value)
        self.assertIn('strength_score', info)
        self.assertIn('created_at', info)
        self.assertIn('tags', info)
        self.assertEqual(info['tags'], ["test", "info"])
    
    def test_persistence(self):
        """Test that secrets persist across manager instances"""
        secret_id = "persistent_secret"
        secret_value = "persistent_value"
        
        # Store secret in first manager
        self.secrets_manager.store_secret(
            secret_id=secret_id,
            secret=secret_value,
            secret_type=SecretType.API_KEY
        )
        
        # Create new manager with same config path
        new_manager = SecretsManager(config_path=self.test_dir)
        
        # Should be able to retrieve the secret
        retrieved = new_manager.get_secret(secret_id)
        self.assertEqual(retrieved, secret_value)
    
    def test_environment_variable_fallback(self):
        """Test fallback to environment variables"""
        secret_id = "ENV_SECRET"
        env_value = "environment_value"
        
        with patch.dict(os.environ, {secret_id: env_value}):
            retrieved = self.secrets_manager.get_secret(secret_id.lower())
            self.assertEqual(retrieved, env_value)
    
    def test_master_password_encryption(self):
        """Test encryption with master password"""
        master_password = "test_master_password"
        
        # Create manager with master password
        manager_with_password = SecretsManager(
            config_path=os.path.join(self.test_dir, "password_test"),
            master_password=master_password
        )
        
        secret_id = "password_protected"
        secret_value = "super_secret"
        
        # Store secret
        manager_with_password.store_secret(
            secret_id=secret_id,
            secret=secret_value,
            secret_type=SecretType.API_KEY
        )
        
        # Retrieve secret
        retrieved = manager_with_password.get_secret(secret_id)
        self.assertEqual(retrieved, secret_value)
    
    def test_access_tracking(self):
        """Test access tracking for secrets"""
        secret_id = "tracked_secret"
        secret_value = "tracked_value"
        
        # Store secret
        self.secrets_manager.store_secret(
            secret_id=secret_id,
            secret=secret_value,
            secret_type=SecretType.API_KEY
        )
        
        # Access secret multiple times
        for _ in range(3):
            self.secrets_manager.get_secret(secret_id)
        
        # Check access count
        metadata = self.secrets_manager._metadata_cache[secret_id]
        self.assertEqual(metadata.access_count, 3)
        self.assertIsNotNone(metadata.last_accessed)
    
    def test_error_handling_invalid_secret_id(self):
        """Test error handling for invalid secret ID"""
        result = self.secrets_manager.rotate_secret("nonexistent_secret")
        self.assertFalse(result)
        
        info = self.secrets_manager.get_secret_info("nonexistent_secret")
        self.assertIsNone(info)
    
    def test_encryption_decryption_error_handling(self):
        """Test handling of encryption/decryption errors"""
        # Create corrupted secrets file
        secrets_file = Path(self.test_dir) / "secrets.enc"
        with open(secrets_file, 'wb') as f:
            f.write(b"corrupted_data")
        
        # Should handle corrupted data gracefully
        manager = SecretsManager(config_path=self.test_dir)
        self.assertEqual(len(manager._secrets_cache), 0)


class TestGlobalSecretsFunctions(unittest.TestCase):
    """Test global secrets manager functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        
        # Reset global manager
        import modules.security.secrets_manager as sm
        sm._secrets_manager = None
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
        # Reset global manager
        import modules.security.secrets_manager as sm
        sm._secrets_manager = None
    
    def test_get_secrets_manager_singleton(self):
        """Test that get_secrets_manager returns singleton"""
        manager1 = get_secrets_manager(config_path=self.test_dir)
        manager2 = get_secrets_manager(config_path=self.test_dir)
        
        self.assertIs(manager1, manager2)
    
    def test_get_secret_convenience_function(self):
        """Test convenience get_secret function"""
        manager = get_secrets_manager(config_path=self.test_dir)
        
        # Store a secret
        manager.store_secret(
            secret_id="test_convenience",
            secret="convenience_value",
            secret_type=SecretType.API_KEY
        )
        
        # Use convenience function
        value = get_secret("test_convenience")
        self.assertEqual(value, "convenience_value")
        
        # Test default value
        default_value = get_secret("nonexistent", default="default_val")
        self.assertEqual(default_value, "default_val")


class TestSecretMetadata(unittest.TestCase):
    """Test SecretMetadata dataclass"""
    
    def test_metadata_creation(self):
        """Test creating SecretMetadata"""
        now = datetime.utcnow()
        metadata = SecretMetadata(
            secret_id="test_id",
            secret_type=SecretType.API_KEY,
            created_at=now,
            updated_at=now,
            tags=["test", "metadata"]
        )
        
        self.assertEqual(metadata.secret_id, "test_id")
        self.assertEqual(metadata.secret_type, SecretType.API_KEY)
        self.assertEqual(metadata.tags, ["test", "metadata"])
        self.assertEqual(metadata.access_count, 0)
    
    def test_metadata_default_tags(self):
        """Test default tags initialization"""
        now = datetime.utcnow()
        metadata = SecretMetadata(
            secret_id="test_id",
            secret_type=SecretType.API_KEY,
            created_at=now,
            updated_at=now
        )
        
        self.assertEqual(metadata.tags, [])


class TestSecretType(unittest.TestCase):
    """Test SecretType enum"""
    
    def test_secret_types(self):
        """Test all secret types are available"""
        expected_types = [
            "api_key",
            "encryption_key", 
            "jwt_secret",
            "database_password",
            "tls_private_key",
            "tls_certificate",
            "oauth_client_secret",
            "webhook_secret"
        ]
        
        for expected_type in expected_types:
            self.assertTrue(any(st.value == expected_type for st in SecretType))


if __name__ == '__main__':
    unittest.main()
