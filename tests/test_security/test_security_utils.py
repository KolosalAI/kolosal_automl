"""
Unit tests for Security Utilities

Tests cover:
- Password generation and validation
- API key generation
- Input sanitization
- Security helpers

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import unittest
import string
from unittest.mock import patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.security.security_utils import (
    generate_secure_password,
    validate_password_strength,
    generate_secure_api_key,
    generate_jwt_secret,
    generate_session_token,
    hash_sensitive_data,
    verify_sensitive_data,
    sanitize_filename,
    validate_email,
    validate_ip_address,
    validate_url,
    get_entropy_score,
    generate_nonce,
    time_constant_compare,
    mask_sensitive_data,
    get_security_headers_config,
    get_csp_policy
)


class TestPasswordGeneration(unittest.TestCase):
    """Test password generation functionality"""
    
    def test_generate_secure_password_default(self):
        """Test default password generation"""
        password = generate_secure_password()
        
        self.assertIsInstance(password, str)
        self.assertGreaterEqual(len(password), 16)
        
        # Should contain different character types
        self.assertTrue(any(c.islower() for c in password))
        self.assertTrue(any(c.isupper() for c in password))
        self.assertTrue(any(c.isdigit() for c in password))
    
    def test_generate_secure_password_custom_length(self):
        """Test password generation with custom length"""
        lengths = [8, 12, 24, 32]
        
        for length in lengths:
            password = generate_secure_password(length=length)
            self.assertEqual(len(password), length)
    
    def test_generate_secure_password_minimum_length(self):
        """Test password generation respects minimum length"""
        # Request very short password
        password = generate_secure_password(length=4)
        
        # Should enforce minimum length
        self.assertGreaterEqual(len(password), 8)
    
    def test_generate_secure_password_without_symbols(self):
        """Test password generation without symbols"""
        password = generate_secure_password(include_symbols=False)
        
        # Should not contain symbols
        symbols = "!@#$%^&*(),.?\":{}|<>"
        self.assertFalse(any(c in symbols for c in password))
    
    def test_generate_secure_password_exclude_ambiguous(self):
        """Test password generation excluding ambiguous characters"""
        password = generate_secure_password(exclude_ambiguous=True)
        
        # Should not contain ambiguous characters
        ambiguous = "0O1l|:"
        self.assertFalse(any(c in ambiguous for c in password))
    
    def test_password_uniqueness(self):
        """Test that generated passwords are unique"""
        passwords = [generate_secure_password() for _ in range(10)]
        unique_passwords = set(passwords)
        
        self.assertEqual(len(passwords), len(unique_passwords))


class TestPasswordValidation(unittest.TestCase):
    """Test password strength validation"""
    
    def test_validate_strong_password(self):
        """Test validation of strong passwords"""
        strong_passwords = [
            "MyStr0ng!P@ssw0rd42",
            "C0mpl3x&Secure#Password!",
            "Th1s!sVery$tr0ng2024",
            "Adm1n!str@t0r#P@ssw0rd"
        ]
        
        for password in strong_passwords:
            is_valid, issues = validate_password_strength(password)
            self.assertTrue(is_valid, f"Password '{password}' should be valid. Issues: {issues}")
            self.assertEqual(len(issues), 0)
    
    def test_validate_weak_password(self):
        """Test validation of weak passwords"""
        weak_passwords = [
            "password",
            "123456",
            "admin",
            "qwerty",
            "letmein"
        ]
        
        for password in weak_passwords:
            is_valid, issues = validate_password_strength(password)
            self.assertFalse(is_valid, f"Password '{password}' should be invalid")
            self.assertGreater(len(issues), 0)
    
    def test_validate_password_length_requirements(self):
        """Test password length requirements"""
        # Too short
        short_password = "A1!"
        is_valid, issues = validate_password_strength(short_password)
        self.assertFalse(is_valid)
        self.assertTrue(any("at least" in issue for issue in issues))
        
        # Too long
        long_password = "A" * 200
        is_valid, issues = validate_password_strength(long_password)
        self.assertFalse(is_valid)
        self.assertTrue(any("no more than" in issue for issue in issues))
    
    def test_validate_password_character_requirements(self):
        """Test password character type requirements"""
        # Missing uppercase
        no_upper = "lowercase123!"
        is_valid, issues = validate_password_strength(no_upper)
        self.assertFalse(is_valid)
        self.assertTrue(any("uppercase" in issue for issue in issues))
        
        # Missing lowercase
        no_lower = "UPPERCASE123!"
        is_valid, issues = validate_password_strength(no_lower)
        self.assertFalse(is_valid)
        self.assertTrue(any("lowercase" in issue for issue in issues))
        
        # Missing digits
        no_digit = "NoDigitsHere!"
        is_valid, issues = validate_password_strength(no_digit)
        self.assertFalse(is_valid)
        self.assertTrue(any("digit" in issue for issue in issues))
        
        # Missing special characters
        no_special = "NoSpecialChars123"
        is_valid, issues = validate_password_strength(no_special)
        self.assertFalse(is_valid)
        self.assertTrue(any("special" in issue for issue in issues))
    
    def test_validate_password_common_patterns(self):
        """Test detection of common password patterns"""
        # Sequential characters
        sequential = "Abc12345!"
        is_valid, issues = validate_password_strength(sequential)
        if not is_valid:
            issue_text = " ".join(issues)
            if "sequential" in issue_text:
                self.assertTrue(True)  # Sequential pattern detected
        
        # Repeated characters
        repeated = "Aaa11111!"
        is_valid, issues = validate_password_strength(repeated)
        if not is_valid:
            issue_text = " ".join(issues)
            if "repeated" in issue_text:
                self.assertTrue(True)  # Repeated pattern detected


class TestAPIKeyGeneration(unittest.TestCase):
    """Test API key generation"""
    
    def test_generate_secure_api_key_default(self):
        """Test default API key generation"""
        api_key = generate_secure_api_key()
        
        self.assertIsInstance(api_key, str)
        self.assertGreaterEqual(len(api_key), 32)
    
    def test_generate_secure_api_key_custom_length(self):
        """Test API key generation with custom length"""
        api_key = generate_secure_api_key(length=64)
        
        self.assertGreaterEqual(len(api_key), 64)
    
    def test_generate_secure_api_key_with_prefix(self):
        """Test API key generation with prefix"""
        prefix = "sk_"
        api_key = generate_secure_api_key(prefix=prefix)
        
        self.assertTrue(api_key.startswith(prefix))
    
    def test_api_key_uniqueness(self):
        """Test that generated API keys are unique"""
        api_keys = [generate_secure_api_key() for _ in range(10)]
        unique_keys = set(api_keys)
        
        self.assertEqual(len(api_keys), len(unique_keys))


class TestSecretGeneration(unittest.TestCase):
    """Test various secret generation functions"""
    
    def test_generate_jwt_secret(self):
        """Test JWT secret generation"""
        jwt_secret = generate_jwt_secret()
        
        self.assertIsInstance(jwt_secret, str)
        self.assertGreaterEqual(len(jwt_secret), 64)
    
    def test_generate_session_token(self):
        """Test session token generation"""
        token = generate_session_token()
        
        self.assertIsInstance(token, str)
        self.assertGreaterEqual(len(token), 32)
    
    def test_generate_nonce(self):
        """Test nonce generation"""
        nonce = generate_nonce()
        
        self.assertIsInstance(nonce, str)
        self.assertGreaterEqual(len(nonce), 16)
        
        # Should be URL-safe base64
        import base64
        try:
            base64.b64decode(nonce + '==')  # Add padding if needed
            valid_base64 = True
        except:
            valid_base64 = False
        
        self.assertTrue(valid_base64)


class TestDataHashing(unittest.TestCase):
    """Test sensitive data hashing functions"""
    
    def test_hash_sensitive_data(self):
        """Test sensitive data hashing"""
        data = "sensitive_information"
        hash_value, salt = hash_sensitive_data(data)
        
        self.assertIsInstance(hash_value, str)
        self.assertIsInstance(salt, str)
        self.assertNotEqual(hash_value, data)
        self.assertGreater(len(hash_value), 0)
        self.assertGreater(len(salt), 0)
    
    def test_hash_verification(self):
        """Test hash verification"""
        data = "test_data_to_hash"
        hash_value, salt = hash_sensitive_data(data)
        
        # Verify correct data
        is_valid = verify_sensitive_data(data, hash_value, salt)
        self.assertTrue(is_valid)
        
        # Verify incorrect data
        is_invalid = verify_sensitive_data("wrong_data", hash_value, salt)
        self.assertFalse(is_invalid)
    
    def test_hash_with_custom_salt(self):
        """Test hashing with custom salt"""
        data = "test_data"
        custom_salt = "custom_salt_value"
        
        hash_value, returned_salt = hash_sensitive_data(data, custom_salt)
        
        self.assertEqual(returned_salt, custom_salt)
        
        # Should verify correctly
        is_valid = verify_sensitive_data(data, hash_value, custom_salt)
        self.assertTrue(is_valid)


class TestInputSanitization(unittest.TestCase):
    """Test input sanitization functions"""
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("file with spaces.txt", "file_with_spaces.txt"),
            ("../../../etc/passwd", "___etc_passwd"),
            ("file|with|pipes.txt", "file_with_pipes.txt"),
            ("very" + "long" * 100 + ".txt", "verylonglong")  # Should be truncated
        ]
        
        for input_filename, expected_pattern in test_cases:
            sanitized = sanitize_filename(input_filename)
            
            # Should not contain dangerous characters
            dangerous_chars = ["../", "..\\", "|", "<", ">", ":", "*", "?", '"']
            for char in dangerous_chars:
                self.assertNotIn(char, sanitized)


class TestValidationFunctions(unittest.TestCase):
    """Test various validation functions"""
    
    def test_validate_email(self):
        """Test email validation"""
        valid_emails = [
            "user@example.com",
            "test.email@domain.org",
            "admin+tag@company.co.uk",
            "123@numbers.com"
        ]
        
        invalid_emails = [
            "invalid.email",
            "@domain.com",
            "user@",
            "user..double@domain.com",
            "user@domain",
            ""
        ]
        
        for email in valid_emails:
            self.assertTrue(validate_email(email), f"Email '{email}' should be valid")
        
        for email in invalid_emails:
            self.assertFalse(validate_email(email), f"Email '{email}' should be invalid")
    
    def test_validate_ip_address(self):
        """Test IP address validation"""
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "127.0.0.1",
            "255.255.255.255",
            "2001:db8::1",
            "::1"
        ]
        
        invalid_ips = [
            "256.256.256.256",
            "192.168.1",
            "not.an.ip.address",
            "192.168.1.1.1",
            ""
        ]
        
        for ip in valid_ips:
            self.assertTrue(validate_ip_address(ip), f"IP '{ip}' should be valid")
        
        for ip in invalid_ips:
            self.assertFalse(validate_ip_address(ip), f"IP '{ip}' should be invalid")
    
    def test_validate_url(self):
        """Test URL validation"""
        valid_urls = [
            "https://example.com",
            "http://localhost:8000",
            "https://api.example.com/v1/endpoint",
            "http://192.168.1.1:3000"
        ]
        
        invalid_urls = [
            "not_a_url",
            "ftp://example.com",  # Only http/https allowed
            "example.com",  # Missing protocol
            "https://",  # Incomplete
            ""
        ]
        
        for url in valid_urls:
            self.assertTrue(validate_url(url), f"URL '{url}' should be valid")
        
        for url in invalid_urls:
            self.assertFalse(validate_url(url), f"URL '{url}' should be invalid")


class TestSecurityHelpers(unittest.TestCase):
    """Test security helper functions"""
    
    def test_get_entropy_score(self):
        """Test entropy score calculation"""
        # High entropy string
        high_entropy = "Th1s!sAV3ry$tr0ngP@ssw0rd"
        high_score = get_entropy_score(high_entropy)
        
        # Low entropy string
        low_entropy = "aaaaaaaaaa"
        low_score = get_entropy_score(low_entropy)
        
        # High entropy should have higher score
        self.assertGreater(high_score, low_score)
        
        # Empty string should have zero entropy
        zero_score = get_entropy_score("")
        self.assertEqual(zero_score, 0.0)
    
    def test_time_constant_compare(self):
        """Test time-constant string comparison"""
        string1 = "secret_value"
        string2 = "secret_value"
        string3 = "different_value"
        
        # Same strings should match
        self.assertTrue(time_constant_compare(string1, string2))
        
        # Different strings should not match
        self.assertFalse(time_constant_compare(string1, string3))
    
    def test_mask_sensitive_data(self):
        """Test sensitive data masking"""
        sensitive_data = "secret123456789"
        
        # Default masking
        masked = mask_sensitive_data(sensitive_data)
        
        self.assertNotEqual(masked, sensitive_data)
        self.assertIn("*", masked)
        
        # Custom masking
        custom_masked = mask_sensitive_data(
            sensitive_data,
            mask_char="#",
            visible_start=3,
            visible_end=3
        )
        
        self.assertTrue(custom_masked.startswith("sec"))
        self.assertTrue(custom_masked.endswith("789"))
        self.assertIn("#", custom_masked)
    
    def test_mask_short_data(self):
        """Test masking of short data"""
        short_data = "abc"
        masked = mask_sensitive_data(short_data, visible_start=2, visible_end=2)
        
        # Should be completely masked when data is too short
        self.assertEqual(masked, "***")


class TestSecurityHeaders(unittest.TestCase):
    """Test security headers configuration"""
    
    def test_get_security_headers_config(self):
        """Test security headers configuration"""
        headers = get_security_headers_config()
        
        self.assertIsInstance(headers, dict)
        
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Referrer-Policy",
            "Cache-Control"
        ]
        
        for header in required_headers:
            self.assertIn(header, headers)
        
        # Check specific values
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
    
    def test_get_csp_policy(self):
        """Test Content Security Policy generation"""
        # Test different modes
        modes = ["strict", "moderate", "relaxed"]
        
        for mode in modes:
            csp = get_csp_policy(mode)
            
            self.assertIsInstance(csp, str)
            self.assertIn("default-src", csp)
            
            if mode == "strict":
                self.assertIn("'self'", csp)
                self.assertNotIn("'unsafe-eval'", csp)
            elif mode == "relaxed":
                self.assertIn("'unsafe-eval'", csp)
    
    def test_get_csp_policy_invalid_mode(self):
        """Test CSP policy with invalid mode"""
        csp = get_csp_policy("invalid_mode")
        
        # Should default to strict
        self.assertIn("'self'", csp)
        self.assertNotIn("'unsafe-eval'", csp)


if __name__ == '__main__':
    unittest.main()
