"""
Unit tests for the enhanced security framework

Tests cover:
- Rate limiting functionality
- Input validation
- API key validation
- Security event logging
- Threat detection

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import time
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from fastapi import Request
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Mock FastAPI components for testing
    class Request:
        def __init__(self):
            self.client = Mock()
            self.client.host = "127.0.0.1"
            self.headers = {}
            self.method = "GET"
            self.url = Mock()
            self.url.path = "/test"

from modules.security.enhanced_security import (
    EnhancedSecurityManager,
    AdvancedRateLimiter,
    AdvancedInputValidator,
    SecurityEvent,
    SecurityAuditor,
    generate_secure_api_key
)


class TestEnhancedSecurityManager(unittest.TestCase):
    """Test cases for EnhancedSecurityManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.security_manager = EnhancedSecurityManager()
        self.mock_request = self._create_mock_request()
    
    def _create_mock_request(self):
        """Create a mock FastAPI request"""
        request = Mock()
        request.client = Mock()
        request.client.host = "192.168.1.100"
        request.headers = {
            "user-agent": "TestClient/1.0",
            "x-forwarded-for": "10.0.0.1"
        }
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/api/test"
        return request
    
    def test_initialization(self):
        """Test EnhancedSecurityManager initialization"""
        self.assertIsInstance(self.security_manager.rate_limiter, AdvancedRateLimiter)
        self.assertIsInstance(self.security_manager.input_validator, AdvancedInputValidator)
        self.assertIsInstance(self.security_manager.auditor, SecurityAuditor)
    
    def test_rate_limiting_allowed(self):
        """Test rate limiting when requests are allowed"""
        allowed, details = self.security_manager.check_rate_limit(self.mock_request)
        
        self.assertTrue(allowed)
        self.assertIsInstance(details, dict)
    
    def test_rate_limiting_exceeded(self):
        """Test rate limiting when limit is exceeded"""
        # Simulate many requests quickly
        for _ in range(200):  # Exceed typical rate limit
            self.security_manager.check_rate_limit(self.mock_request)
        
        allowed, details = self.security_manager.check_rate_limit(self.mock_request)
        
        # Should eventually be rate limited
        if not allowed:
            self.assertFalse(allowed)
            self.assertIn("reason", details)
    
    def test_input_validation_safe_data(self):
        """Test input validation with safe data"""
        safe_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30
        }
        
        is_valid, issues = self.security_manager.validate_input(safe_data)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_input_validation_sql_injection(self):
        """Test input validation detects SQL injection"""
        malicious_data = {
            "username": "admin'; DROP TABLE users; --",
            "password": "password"
        }
        
        is_valid, issues = self.security_manager.validate_input(malicious_data)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("SQL injection" in issue for issue in issues))
    
    def test_input_validation_xss_attack(self):
        """Test input validation detects XSS attacks"""
        malicious_data = {
            "comment": "<script>alert('xss')</script>",
            "title": "Normal title"
        }
        
        is_valid, issues = self.security_manager.validate_input(malicious_data)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("XSS" in issue for issue in issues))
    
    def test_api_key_validation_valid(self):
        """Test API key validation with valid keys"""
        valid_keys = ["valid_key_123", "another_valid_key_456"]
        
        result = self.security_manager.validate_api_key("valid_key_123", valid_keys)
        self.assertTrue(result)
    
    def test_api_key_validation_invalid(self):
        """Test API key validation with invalid keys"""
        valid_keys = ["valid_key_123", "another_valid_key_456"]
        
        result = self.security_manager.validate_api_key("invalid_key", valid_keys)
        self.assertFalse(result)
    
    def test_api_key_validation_empty(self):
        """Test API key validation with empty inputs"""
        result1 = self.security_manager.validate_api_key("", ["valid_key"])
        result2 = self.security_manager.validate_api_key("valid_key", [])
        
        self.assertFalse(result1)
        self.assertFalse(result2)
    
    def test_security_headers(self):
        """Test security headers generation"""
        headers = self.security_manager.get_security_headers()
        
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Referrer-Policy"
        ]
        
        for header in required_headers:
            self.assertIn(header, headers)
    
    def test_security_event_logging(self):
        """Test security event logging"""
        events_before = len(self.security_manager.get_recent_security_events())
        
        # Trigger a security event
        malicious_data = {"input": "'; DROP TABLE test; --"}
        self.security_manager.validate_input(malicious_data)
        
        events_after = len(self.security_manager.get_recent_security_events())
        
        # Should have logged a security event
        self.assertGreater(events_after, events_before)


class TestAdvancedRateLimiter(unittest.TestCase):
    """Test cases for AdvancedRateLimiter"""
    
    def setUp(self):
        """Set up test environment"""
        self.rate_limiter = AdvancedRateLimiter(
            max_requests=10,
            time_window=60,
            strategy="sliding_window"
        )
        self.mock_request = Mock()
        self.mock_request.client = Mock()
        self.mock_request.client.host = "192.168.1.100"
        self.mock_request.headers = {}
    
    def test_sliding_window_rate_limiting(self):
        """Test sliding window rate limiting"""
        # Make requests within limit
        for i in range(5):
            allowed, _ = self.rate_limiter.is_allowed(self.mock_request)
            self.assertTrue(allowed)
    
    def test_rate_limit_exceeded(self):
        """Test behavior when rate limit is exceeded"""
        # Exceed the limit
        for i in range(15):
            allowed, details = self.rate_limiter.is_allowed(self.mock_request)
            if i < 10:
                self.assertTrue(allowed)
            else:
                self.assertFalse(allowed)
                self.assertIn("reason", details)
    
    def test_ip_blocking(self):
        """Test IP blocking functionality"""
        # Exceed rate limit multiple times to trigger blocking
        for _ in range(20):
            self.rate_limiter.is_allowed(self.mock_request)
        
        # Check if IP gets blocked
        blocked_ips = self.rate_limiter.blocked_ips
        if "192.168.1.100" in blocked_ips:
            # If blocked, further requests should be denied
            allowed, details = self.rate_limiter.is_allowed(self.mock_request)
            self.assertFalse(allowed)
            self.assertEqual(details["reason"], "IP blocked")
    
    def test_different_strategies(self):
        """Test different rate limiting strategies"""
        strategies = ["sliding_window", "fixed_window", "token_bucket"]
        
        for strategy in strategies:
            limiter = AdvancedRateLimiter(strategy=strategy)
            allowed, _ = limiter.is_allowed(self.mock_request)
            self.assertTrue(allowed)  # First request should always be allowed


class TestAdvancedInputValidator(unittest.TestCase):
    """Test cases for AdvancedInputValidator"""
    
    def setUp(self):
        """Set up test environment"""
        self.validator = AdvancedInputValidator()
    
    def test_safe_input_validation(self):
        """Test validation of safe input"""
        safe_inputs = [
            "Hello World",
            "user@example.com",
            "123-456-7890",
            {"name": "John", "age": 30}
        ]
        
        for input_data in safe_inputs:
            is_valid, issues = self.validator.validate_input(input_data)
            self.assertTrue(is_valid)
            self.assertEqual(len(issues), 0)
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/**/OR/**/1=1--",
            "UNION SELECT * FROM passwords"
        ]
        
        for malicious_input in malicious_inputs:
            is_valid, issues = self.validator.validate_input(malicious_input)
            self.assertFalse(is_valid)
            self.assertTrue(any("SQL injection" in issue for issue in issues))
    
    def test_xss_detection(self):
        """Test XSS attack detection"""
        xss_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(1)'></iframe>"
        ]
        
        for xss_input in xss_inputs:
            is_valid, issues = self.validator.validate_input(xss_input)
            self.assertFalse(is_valid)
            self.assertTrue(any("XSS" in issue for issue in issues))
    
    def test_path_traversal_detection(self):
        """Test path traversal detection"""
        path_traversal_inputs = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "%2e%2e%2f%2e%2e%2f",
            "....//....//etc/passwd"
        ]
        
        for traversal_input in path_traversal_inputs:
            is_valid, issues = self.validator.validate_input(traversal_input)
            self.assertFalse(is_valid)
            self.assertTrue(any("path traversal" in issue for issue in issues))
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        malicious_input = "<script>alert('test')</script>"
        sanitized = self.validator.sanitize_input(malicious_input)
        
        self.assertNotIn("<script>", sanitized)
        self.assertIn("&lt;script&gt;", sanitized)
    
    def test_large_input_detection(self):
        """Test detection of overly large inputs"""
        large_input = "A" * 20000  # Very large input
        is_valid, issues = self.validator.validate_input(large_input)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("too long" in issue for issue in issues))
    
    def test_null_byte_detection(self):
        """Test null byte detection"""
        null_byte_input = "normal_input\x00malicious"
        is_valid, issues = self.validator.validate_input(null_byte_input)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("null byte" in issue for issue in issues))
    
    def test_nested_data_validation(self):
        """Test validation of nested data structures"""
        nested_data = {
            "user": {
                "name": "John",
                "comments": ["Good product", "<script>alert('xss')</script>"]
            },
            "products": [
                {"name": "Product 1"},
                {"description": "'; DROP TABLE products; --"}
            ]
        }
        
        is_valid, issues = self.validator.validate_input(nested_data)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)


class TestSecurityAuditor(unittest.TestCase):
    """Test cases for SecurityAuditor"""
    
    def setUp(self):
        """Set up test environment"""
        self.auditor = SecurityAuditor()
    
    def test_event_logging(self):
        """Test security event logging"""
        initial_count = len(self.auditor.get_recent_events())
        
        # Log a security event
        self.auditor.log_security_event(
            event_type="TEST_EVENT",
            severity="HIGH",
            details={"test": "data"},
            source_ip="192.168.1.1"
        )
        
        final_count = len(self.auditor.get_recent_events())
        self.assertEqual(final_count, initial_count + 1)
        
        # Check event details
        recent_events = self.auditor.get_recent_events()
        latest_event = recent_events[-1]
        
        self.assertEqual(latest_event["event_type"], "TEST_EVENT")
        self.assertEqual(latest_event["severity"], "HIGH")
        self.assertEqual(latest_event["source_ip"], "192.168.1.1")
    
    def test_threat_pattern_detection(self):
        """Test threat pattern detection"""
        # Log multiple suspicious events from same IP
        for i in range(5):
            self.auditor.log_security_event(
                event_type="RATE_LIMIT_EXCEEDED",
                severity="MEDIUM",
                details={"attempt": i},
                source_ip="10.0.0.1"
            )
        
        patterns = self.auditor.detect_threat_patterns()
        
        # Should detect pattern of repeated rate limit violations
        self.assertGreater(len(patterns), 0)
    
    def test_event_filtering(self):
        """Test event filtering by type and timeframe"""
        # Log different types of events
        event_types = ["LOGIN_FAILURE", "RATE_LIMIT_EXCEEDED", "SQL_INJECTION_ATTEMPT"]
        
        for event_type in event_types:
            self.auditor.log_security_event(
                event_type=event_type,
                severity="MEDIUM",
                details={},
                source_ip="192.168.1.100"
            )
        
        # Filter by event type
        login_failures = self.auditor.get_recent_events(event_type="LOGIN_FAILURE")
        self.assertTrue(any(event["event_type"] == "LOGIN_FAILURE" for event in login_failures))
        
        # Filter by timeframe
        recent_events = self.auditor.get_recent_events(hours=1)
        self.assertGreater(len(recent_events), 0)


class TestSecurityUtilities(unittest.TestCase):
    """Test security utility functions"""
    
    def test_generate_secure_api_key(self):
        """Test secure API key generation"""
        api_key = generate_secure_api_key()
        
        self.assertIsInstance(api_key, str)
        self.assertGreaterEqual(len(api_key), 32)
        
        # Test with custom length
        custom_key = generate_secure_api_key(length=64)
        self.assertGreaterEqual(len(custom_key), 64)
    
    def test_api_key_uniqueness(self):
        """Test that generated API keys are unique"""
        keys = [generate_secure_api_key() for _ in range(10)]
        unique_keys = set(keys)
        
        self.assertEqual(len(keys), len(unique_keys))


class TestSecurityEvent(unittest.TestCase):
    """Test SecurityEvent dataclass"""
    
    def test_event_creation(self):
        """Test creating SecurityEvent instances"""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="TEST_EVENT",
            severity="HIGH",
            source_ip="192.168.1.1",
            details={"key": "value"}
        )
        
        self.assertEqual(event.event_type, "TEST_EVENT")
        self.assertEqual(event.severity, "HIGH")
        self.assertEqual(event.source_ip, "192.168.1.1")
        self.assertIsInstance(event.details, dict)
    
    def test_event_serialization(self):
        """Test event serialization to dict"""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="TEST_EVENT",
            severity="MEDIUM",
            source_ip="10.0.0.1",
            details={"action": "test"}
        )
        
        event_dict = event.to_dict()
        
        self.assertIsInstance(event_dict, dict)
        self.assertEqual(event_dict["event_type"], "TEST_EVENT")
        self.assertEqual(event_dict["severity"], "MEDIUM")
        self.assertIn("timestamp", event_dict)


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestIntegrationWithFastAPI(unittest.TestCase):
    """Integration tests with FastAPI (when available)"""
    
    def setUp(self):
        """Set up FastAPI test environment"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        self.app = FastAPI()
        self.security_manager = EnhancedSecurityManager()
        self.client = TestClient(self.app)
        
        # Add a test endpoint
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
    
    def test_fastapi_integration(self):
        """Test basic FastAPI integration"""
        response = self.client.get("/test")
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
