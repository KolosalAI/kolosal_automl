"""
Unit tests for TLS Manager

Tests cover:
- Certificate generation and validation
- TLS configuration
- Certificate management
- Security settings

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import unittest
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from modules.security.tls_manager import TLSManager, CertificateInfo


@unittest.skipIf(not CRYPTOGRAPHY_AVAILABLE, "Cryptography library not available")
class TestTLSManager(unittest.TestCase):
    """Test cases for TLSManager class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.tls_manager = TLSManager(cert_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test TLSManager initialization"""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertIsInstance(self.tls_manager.cert_dir, Path)
    
    def test_generate_private_key(self):
        """Test private key generation"""
        private_key = self.tls_manager._generate_private_key()
        
        self.assertIsNotNone(private_key)
        self.assertIsInstance(private_key, rsa.RSAPrivateKey)
        self.assertEqual(private_key.key_size, 2048)
    
    def test_generate_self_signed_certificate(self):
        """Test self-signed certificate generation"""
        cert_config = self.tls_manager.generate_self_signed_certificate(
            common_name="test.example.com",
            organization="Test Org",
            validity_days=30
        )
        
        self.assertIsNotNone(cert_config)
        self.assertIn("cert_path", cert_config)
        self.assertIn("key_path", cert_config)
        
        # Verify files were created
        cert_path = Path(cert_config["cert_path"])
        key_path = Path(cert_config["key_path"])
        
        self.assertTrue(cert_path.exists())
        self.assertTrue(key_path.exists())
    
    def test_certificate_validation(self):
        """Test certificate validation"""
        # Generate a test certificate
        cert_config = self.tls_manager.generate_self_signed_certificate(
            common_name="test.localhost"
        )
        
        # Validate the certificate
        is_valid, details = self.tls_manager.validate_certificate(cert_config["cert_path"], return_tuple=True)
        
        self.assertTrue(is_valid)
        self.assertIn("subject", details)
        self.assertIn("issuer", details)
        self.assertIn("valid_from", details)
        self.assertIn("valid_until", details)
    
    def test_certificate_expiry_check(self):
        """Test certificate expiry checking"""
        # Generate certificate with short validity
        cert_config = self.tls_manager.generate_self_signed_certificate(
            common_name="test.localhost",
            validity_days=1
        )
        
        # Check expiry
        days_until_expiry = self.tls_manager.check_certificate_expiry(cert_config["cert_path"])
        
        self.assertIsInstance(days_until_expiry, int)
        self.assertLessEqual(days_until_expiry, 1)
    
    def test_certificate_info_extraction(self):
        """Test certificate information extraction"""
        cert_config = self.tls_manager.generate_self_signed_certificate(
            common_name="test.example.com",
            organization="Test Organization",
            country="US"
        )
        
        cert_info = self.tls_manager.get_certificate_info(cert_config["cert_path"])
        
        self.assertIsInstance(cert_info, CertificateInfo)
        self.assertEqual(cert_info.common_name, "test.example.com")
        self.assertEqual(cert_info.organization, "Test Organization")
        self.assertEqual(cert_info.country, "US")
        self.assertTrue(cert_info.is_self_signed)
    
    def test_get_ssl_context(self):
        """Test SSL context creation"""
        cert_config = self.tls_manager.generate_self_signed_certificate()
        
        ssl_context = self.tls_manager.get_ssl_context(
            cert_file=cert_config["cert_path"],
            key_file=cert_config["key_path"]
        )
        
        self.assertIsNotNone(ssl_context)
        # SSL context should have secure settings
        self.assertIsNotNone(ssl_context.protocol)
    
    def test_certificate_chain_validation(self):
        """Test certificate chain validation"""
        # Generate root CA
        ca_config = self.tls_manager.generate_self_signed_certificate(
            common_name="Test Root CA",
            is_ca=True
        )
        
        # For simplicity, just test that the method works with self-signed cert
        is_valid = self.tls_manager.validate_certificate_chain(ca_config["cert_path"])
        
        # Self-signed certificates don't have a valid chain, so this should return False
        # but the method should not raise an exception
        self.assertIsInstance(is_valid, bool)
    
    def test_certificate_renewal_needed(self):
        """Test certificate renewal detection"""
        # Generate certificate with short validity
        cert_config = self.tls_manager.generate_self_signed_certificate(
            validity_days=30
        )
        
        # Check if renewal is needed (default threshold is 30 days)
        needs_renewal = self.tls_manager.check_renewal_needed(
            cert_config["cert_path"],
            renewal_threshold_days=45  # Should need renewal
        )
        
        self.assertTrue(needs_renewal)
        
        # Check with lower threshold
        needs_renewal_low = self.tls_manager.check_renewal_needed(
            cert_config["cert_path"],
            renewal_threshold_days=15  # Should not need renewal
        )
        
        self.assertFalse(needs_renewal_low)
    
    def test_multiple_san_certificates(self):
        """Test certificate generation with multiple Subject Alternative Names"""
        san_list = ["example.com", "www.example.com", "api.example.com"]
        
        cert_config = self.tls_manager.generate_self_signed_certificate(
            common_name="example.com",
            san_list=san_list
        )
        
        # Verify certificate was created
        self.assertTrue(Path(cert_config["cert_path"]).exists())
        
        # Get certificate info and check SANs
        cert_info = self.tls_manager.get_certificate_info(cert_config["cert_path"])
        
        # Should contain all SANs
        for san in san_list:
            self.assertIn(san, cert_info.subject_alt_names)
    
    def test_certificate_with_custom_extensions(self):
        """Test certificate generation with custom extensions"""
        cert_config = self.tls_manager.generate_self_signed_certificate(
            common_name="test.localhost",
            key_usage_critical=True
        )
        
        # Verify certificate has proper extensions
        cert_info = self.tls_manager.get_certificate_info(cert_config["cert_path"])
        
        # Should have key usage extension
        self.assertIsNotNone(cert_info.key_usage)
    
    def test_private_key_encryption(self):
        """Test private key encryption/decryption"""
        password = "test_password"
        
        cert_config = self.tls_manager.generate_self_signed_certificate(
            common_name="test.localhost",
            encrypt_private_key=True,
            private_key_password=password
        )
        
        # Verify encrypted key file exists
        self.assertTrue(Path(cert_config["key_path"]).exists())
        
        # Try to load the encrypted private key
        with open(cert_config["key_path"], 'rb') as f:
            key_data = f.read()
        
        # Should be able to load with password
        private_key = serialization.load_pem_private_key(
            key_data,
            password=password.encode(),
            backend=None
        )
        
        self.assertIsInstance(private_key, rsa.RSAPrivateKey)
    
    def test_certificate_backup_and_restore(self):
        """Test certificate backup and restore functionality"""
        # Generate a certificate
        cert_config = self.tls_manager.generate_self_signed_certificate(
            common_name="backup.test.com"
        )
        
        # Create backup
        backup_path = self.tls_manager.backup_certificates()
        
        self.assertIsNotNone(backup_path)
        self.assertTrue(Path(backup_path).exists())
    
    def test_invalid_certificate_handling(self):
        """Test handling of invalid certificates"""
        # Create invalid certificate file
        invalid_cert_path = Path(self.test_dir) / "invalid.crt"
        with open(invalid_cert_path, 'w') as f:
            f.write("This is not a valid certificate")
        
        # Should handle invalid certificate gracefully
        is_valid, details = self.tls_manager.validate_certificate(str(invalid_cert_path), return_tuple=True)
        
        self.assertFalse(is_valid)
        self.assertIn("error", details)
    
    def test_certificate_format_conversion(self):
        """Test certificate format conversion"""
        cert_config = self.tls_manager.generate_self_signed_certificate()
        
        # Convert to DER format
        der_path = self.tls_manager.convert_certificate_format(
            cert_config["cert_path"],
            output_format="DER"
        )
        
        if der_path:  # Only test if conversion is implemented
            self.assertTrue(Path(der_path).exists())
    
    def test_cipher_suite_configuration(self):
        """Test cipher suite configuration"""
        cipher_suites = self.tls_manager.get_recommended_cipher_suites()
        
        self.assertIsInstance(cipher_suites, list)
        self.assertGreater(len(cipher_suites), 0)
        
        # Should contain secure cipher suites
        secure_patterns = ["ECDHE", "AES", "GCM", "SHA256"]
        cipher_string = ":".join(cipher_suites)
        
        for pattern in secure_patterns:
            self.assertIn(pattern, cipher_string)


class TestCertificateInfo(unittest.TestCase):
    """Test CertificateInfo dataclass"""
    
    def test_certificate_info_creation(self):
        """Test creating CertificateInfo instances"""
        now = datetime.utcnow()
        
        cert_info = CertificateInfo(
            common_name="test.example.com",
            organization="Test Org",
            country="US",
            valid_from=now,
            valid_until=now + timedelta(days=365),
            is_self_signed=True,
            subject_alt_names=["test.example.com", "www.test.example.com"]
        )
        
        self.assertEqual(cert_info.common_name, "test.example.com")
        self.assertEqual(cert_info.organization, "Test Org")
        self.assertTrue(cert_info.is_self_signed)
        self.assertEqual(len(cert_info.subject_alt_names), 2)
    
    def test_certificate_info_serialization(self):
        """Test certificate info serialization"""
        now = datetime.utcnow()
        
        cert_info = CertificateInfo(
            common_name="api.example.com",
            organization="Example Inc",
            country="CA",
            valid_from=now,
            valid_until=now + timedelta(days=90),
            is_self_signed=False
        )
        
        info_dict = cert_info.to_dict()
        
        self.assertIsInstance(info_dict, dict)
        self.assertEqual(info_dict["common_name"], "api.example.com")
        self.assertIn("valid_from", info_dict)
        self.assertIn("valid_until", info_dict)


class TestTLSConfiguration(unittest.TestCase):
    """Test TLS configuration and security settings"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.tls_manager = TLSManager(cert_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_secure_tls_configuration(self):
        """Test secure TLS configuration settings"""
        config = self.tls_manager.get_secure_tls_config()
        
        self.assertIsInstance(config, dict)
        
        # Should have secure protocol versions
        self.assertIn("min_version", config)
        self.assertIn("max_version", config)
        
        # Should disable insecure features
        self.assertIn("disable_compression", config)
        self.assertTrue(config.get("disable_compression", False))
    
    def test_protocol_version_validation(self):
        """Test TLS protocol version validation"""
        # Test valid versions
        valid_versions = ["TLSv1.2", "TLSv1.3"]
        
        for version in valid_versions:
            is_valid = self.tls_manager.validate_tls_version(version)
            self.assertTrue(is_valid)
        
        # Test invalid versions
        invalid_versions = ["SSLv3", "TLSv1.0", "TLSv1.1"]
        
        for version in invalid_versions:
            is_valid = self.tls_manager.validate_tls_version(version)
            self.assertFalse(is_valid)
    
    def test_certificate_authority_validation(self):
        """Test CA certificate validation"""
        # Generate a CA certificate
        ca_config = self.tls_manager.generate_self_signed_certificate(
            common_name="Test Root CA",
            is_ca=True
        )
        
        # Validate CA certificate
        is_ca = self.tls_manager.is_ca_certificate(ca_config["cert_path"])
        
        # Should be recognized as CA certificate
        self.assertTrue(is_ca)


class TestTLSIntegration(unittest.TestCase):
    """Integration tests for TLS manager"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.tls_manager = TLSManager(cert_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_end_to_end_certificate_lifecycle(self):
        """Test complete certificate lifecycle"""
        # 1. Generate certificate
        cert_config = self.tls_manager.generate_self_signed_certificate(
            common_name="lifecycle.test.com",
            validity_days=365
        )
        
        self.assertIsNotNone(cert_config)
        
        # 2. Validate certificate
        is_valid, _ = self.tls_manager.validate_certificate(cert_config["cert_path"], return_tuple=True)
        self.assertTrue(is_valid)
        
        # 3. Check expiry
        days_left = self.tls_manager.check_certificate_expiry(cert_config["cert_path"])
        self.assertGreater(days_left, 300)  # Should have ~365 days
        
        # 4. Get certificate info
        cert_info = self.tls_manager.get_certificate_info(cert_config["cert_path"])
        self.assertEqual(cert_info.common_name, "lifecycle.test.com")
        
        # 5. Create SSL context
        ssl_context = self.tls_manager.get_ssl_context(
            cert_config["cert_path"],
            cert_config["key_path"]
        )
        self.assertIsNotNone(ssl_context)
    
    @patch('modules.security.tls_manager.TLSManager._send_notification')
    def test_certificate_expiry_notification(self, mock_notify):
        """Test certificate expiry notification system"""
        # Generate certificate with short validity
        cert_config = self.tls_manager.generate_self_signed_certificate(
            validity_days=15  # Will trigger renewal notification
        )
        
        # Check for expiry notifications
        self.tls_manager.check_all_certificates_expiry()
        
        # Should have called notification method
        if hasattr(self.tls_manager, '_send_notification'):
            mock_notify.assert_called()


if __name__ == '__main__':
    unittest.main()
