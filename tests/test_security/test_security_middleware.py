"""
Unit tests for Security Middleware

Tests cover:
- CORS middleware functionality
- Security headers middleware
- Rate limiting middleware
- Request validation middleware
- Authentication middleware

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.security.security_middleware import (
    CORSMiddleware,
    SecurityHeadersMiddleware,
    RateLimitingMiddleware,
    RequestValidationMiddleware,
    AuthenticationMiddleware,
    SecurityAuditMiddleware
)

# Mock FastAPI request and response objects
class MockRequest:
    def __init__(self, method="GET", url="http://localhost:8000/test", 
                 headers=None, client=None, json_body=None):
        self.method = method
        self.url = Mock()
        self.url.path = url.replace("http://localhost:8000", "")
        self.url.scheme = "http"
        self.url.hostname = "localhost"
        self.url.port = 8000
        self.headers = headers or {}
        self.client = client or Mock(host="127.0.0.1")
        self._json_body = json_body
        self.state = Mock()  # Add state attribute for FastAPI compatibility
        
    async def json(self):
        return self._json_body or {}
    
    async def body(self):
        return json.dumps(self._json_body or {}).encode() if self._json_body else b""

class MockResponse:
    def __init__(self, status_code=200, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class TestCORSMiddleware(unittest.TestCase):
    """Test CORS middleware functionality"""
    
    def setUp(self):
        self.cors_config = {
            "allowed_origins": ["http://localhost:3000", "https://app.example.com"],
            "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
            "allowed_headers": ["Content-Type", "Authorization"],
            "allow_credentials": True,
            "max_age": 86400
        }
        self.middleware = CORSMiddleware(self.cors_config)
    
    def test_cors_preflight_request(self):
        """Test CORS preflight request handling"""
        request = MockRequest(
            method="OPTIONS",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Mock call_next function
        call_next = AsyncMock()
        
        # Run the middleware
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should handle preflight without calling next
        call_next.assert_not_called()
        
        # Check response has correct CORS headers
        self.assertEqual(response.status_code, 200)
        self.assertIn("Access-Control-Allow-Origin", response.headers)
        self.assertIn("Access-Control-Allow-Methods", response.headers)
    
    def test_cors_simple_request(self):
        """Test CORS simple request handling"""
        request = MockRequest(
            method="GET",
            headers={"Origin": "http://localhost:3000"}
        )
        
        call_next = AsyncMock()
        mock_response = MockResponse()
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should call next middleware
        call_next.assert_called_once()
        
        # Should add CORS headers to response
        self.assertIn("Access-Control-Allow-Origin", response.headers)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:3000")
    
    def test_cors_disallowed_origin(self):
        """Test CORS with disallowed origin"""
        request = MockRequest(
            method="GET",
            headers={"Origin": "https://malicious.com"}
        )
        
        call_next = AsyncMock()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should reject the request
        self.assertEqual(response.status_code, 403)
        call_next.assert_not_called()
    
    def test_cors_wildcard_origin(self):
        """Test CORS with wildcard origin"""
        cors_config = self.cors_config.copy()
        cors_config["allowed_origins"] = ["*"]
        middleware = CORSMiddleware(cors_config)
        
        request = MockRequest(
            method="GET",
            headers={"Origin": "https://any-domain.com"}
        )
        
        call_next = AsyncMock()
        mock_response = MockResponse()
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should allow any origin
        call_next.assert_called_once()
        self.assertIn("Access-Control-Allow-Origin", response.headers)


class TestSecurityHeadersMiddleware(unittest.TestCase):
    """Test security headers middleware"""
    
    def setUp(self):
        self.security_config = {
            "frame_options": "DENY",
            "content_type_options": "nosniff",
            "xss_protection": "1; mode=block",
            "csp_policy": "default-src 'self'",
            "hsts_max_age": 31536000,
            "referrer_policy": "strict-origin-when-cross-origin"
        }
        self.middleware = SecurityHeadersMiddleware(self.security_config)
    
    def test_security_headers_added(self):
        """Test that security headers are added to response"""
        request = MockRequest()
        call_next = AsyncMock()
        mock_response = MockResponse()
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Check that security headers are present
        expected_headers = [
            "X-Frame-Options",
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "Content-Security-Policy",
            "Referrer-Policy"
        ]
        
        for header in expected_headers:
            self.assertIn(header, response.headers)
        
        # Check specific values
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
    
    def test_hsts_header_https_only(self):
        """Test that HSTS header is only added for HTTPS"""
        # HTTP request
        http_request = MockRequest()
        http_request.url.scheme = "http"
        
        call_next = AsyncMock()
        mock_response = MockResponse()
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(http_request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        
        # HSTS should not be added for HTTP
        self.assertNotIn("Strict-Transport-Security", response.headers)
        
        # HTTPS request
        https_request = MockRequest()
        https_request.url.scheme = "https"
        
        async def run_test_https():
            response = await self.middleware.dispatch(https_request, call_next)
            return response
        
        response = loop.run_until_complete(run_test_https())
        loop.close()
        
        # HSTS should be added for HTTPS
        self.assertIn("Strict-Transport-Security", response.headers)


class TestRateLimitingMiddleware(unittest.TestCase):
    """Test rate limiting middleware"""
    
    def setUp(self):
        self.rate_config = {
            "default_limit": "100/minute",
            "endpoint_limits": {
                "/api/login": "5/minute",
                "/api/register": "3/minute"
            },
            "ip_whitelist": ["127.0.0.1"],
            "enable_burst": True
        }
        self.middleware = RateLimitingMiddleware(self.rate_config)
    
    def test_rate_limiting_within_limit(self):
        """Test request within rate limit"""
        request = MockRequest()
        call_next = AsyncMock()
        mock_response = MockResponse()
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should allow request through
        call_next.assert_called_once()
        self.assertEqual(response.status_code, 200)
    
    def test_rate_limiting_whitelisted_ip(self):
        """Test that whitelisted IPs bypass rate limiting"""
        request = MockRequest()
        request.client.host = "127.0.0.1"  # Whitelisted IP
        
        call_next = AsyncMock()
        mock_response = MockResponse()
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should allow whitelisted IP through
        call_next.assert_called_once()
    
    @patch('modules.security.security_middleware.time.time')
    def test_rate_limiting_exceeded(self, mock_time):
        """Test rate limit exceeded scenario"""
        # Mock time to control rate limiting
        mock_time.return_value = 1000.0
        
        # Create middleware with very low limit for testing
        test_config = {
            "default_limit": "1/minute",
            "endpoint_limits": {},
            "ip_whitelist": [],
            "enable_burst": False
        }
        middleware = RateLimitingMiddleware(test_config)
        
        request = MockRequest()
        request.client.host = "192.168.1.100"
        
        call_next = AsyncMock()
        mock_response = MockResponse()
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            # First request should pass
            response1 = await middleware.dispatch(request, call_next)
            
            # Second request should be rate limited
            response2 = await middleware.dispatch(request, call_next)
            
            return response1, response2
        
        response1, response2 = loop.run_until_complete(run_test())
        loop.close()
        
        # First request should succeed
        self.assertEqual(response1.status_code, 200)
        
        # Second request should be rate limited
        self.assertEqual(response2.status_code, 429)


class TestRequestValidationMiddleware(unittest.TestCase):
    """Test request validation middleware"""
    
    def setUp(self):
        self.validation_config = {
            "max_content_length": 1024 * 1024,  # 1MB
            "allowed_content_types": ["application/json", "text/plain"],
            "validate_json": True,
            "max_json_depth": 10,
            "blocked_user_agents": ["BadBot", "Scraper"]
        }
        self.middleware = RequestValidationMiddleware(self.validation_config)
    
    def test_valid_request(self):
        """Test valid request passes through"""
        request = MockRequest(
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"
            },
            json_body={"key": "value"}
        )
        
        call_next = AsyncMock()
        mock_response = MockResponse()
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should allow valid request through
        call_next.assert_called_once()
        self.assertEqual(response.status_code, 200)
    
    def test_blocked_user_agent(self):
        """Test blocked user agent rejection"""
        request = MockRequest(
            headers={"User-Agent": "BadBot/1.0"}
        )
        
        call_next = AsyncMock()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should block request
        self.assertEqual(response.status_code, 403)
        call_next.assert_not_called()
    
    def test_invalid_content_type(self):
        """Test invalid content type rejection"""
        request = MockRequest(
            method="POST",
            headers={"Content-Type": "application/xml"}
        )
        
        call_next = AsyncMock()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should reject invalid content type
        self.assertEqual(response.status_code, 400)
        call_next.assert_not_called()


class TestAuthenticationMiddleware(unittest.TestCase):
    """Test authentication middleware"""
    
    def setUp(self):
        self.auth_config = {
            "jwt_secret": "test_secret_key",
            "jwt_algorithm": "HS256",
            "token_expire_minutes": 30,
            "public_endpoints": ["/health", "/docs", "/openapi.json"],
            "admin_endpoints": ["/admin"],
            "require_https": False  # For testing
        }
        self.middleware = AuthenticationMiddleware(self.auth_config)
    
    def test_public_endpoint_access(self):
        """Test access to public endpoints without authentication"""
        request = MockRequest(url="http://localhost:8000/health")
        
        call_next = AsyncMock()
        mock_response = MockResponse()
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should allow access to public endpoint
        call_next.assert_called_once()
        self.assertEqual(response.status_code, 200)
    
    def test_protected_endpoint_no_token(self):
        """Test access to protected endpoint without token"""
        request = MockRequest(url="http://localhost:8000/api/protected")
        
        call_next = AsyncMock()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should reject request without token
        self.assertEqual(response.status_code, 401)
        call_next.assert_not_called()
    
    @patch('modules.security.security_middleware.jwt.decode')
    def test_protected_endpoint_valid_token(self, mock_jwt_decode):
        """Test access to protected endpoint with valid token"""
        # Mock valid JWT decode
        mock_jwt_decode.return_value = {
            "sub": "user123",
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "role": "user"
        }
        
        request = MockRequest(
            url="http://localhost:8000/api/protected",
            headers={"Authorization": "Bearer valid_token"}
        )
        
        call_next = AsyncMock()
        mock_response = MockResponse()
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should allow access with valid token
        call_next.assert_called_once()
        self.assertEqual(response.status_code, 200)
    
    @patch('modules.security.security_middleware.jwt.decode')
    def test_admin_endpoint_insufficient_privileges(self, mock_jwt_decode):
        """Test access to admin endpoint with insufficient privileges"""
        # Mock JWT decode for regular user
        mock_jwt_decode.return_value = {
            "sub": "user123",
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "role": "user"
        }
        
        request = MockRequest(
            url="http://localhost:8000/admin/users",
            headers={"Authorization": "Bearer user_token"}
        )
        
        call_next = AsyncMock()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should reject access to admin endpoint
        self.assertEqual(response.status_code, 403)
        call_next.assert_not_called()


class TestSecurityAuditMiddleware(unittest.TestCase):
    """Test security audit middleware"""
    
    def setUp(self):
        self.audit_config = {
            "log_all_requests": True,
            "log_authentication_events": True,
            "log_authorization_failures": True,
            "log_suspicious_activity": True,
            "max_log_size": 1024 * 1024,  # 1MB
            "rotate_logs": True
        }
        self.middleware = SecurityAuditMiddleware(self.audit_config)
    
    @patch('modules.security.security_middleware.logging.getLogger')
    def test_request_logging(self, mock_get_logger):
        """Test that requests are logged for audit"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        request = MockRequest(
            method="POST",
            url="http://localhost:8000/api/login",
            headers={"User-Agent": "TestAgent"}
        )
        
        call_next = AsyncMock()
        mock_response = MockResponse(status_code=200)
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should log the request
        mock_logger.info.assert_called()
        
        # Should continue with request
        call_next.assert_called_once()
        self.assertEqual(response.status_code, 200)
    
    @patch('modules.security.security_middleware.logging.getLogger')
    def test_error_response_logging(self, mock_get_logger):
        """Test logging of error responses"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        request = MockRequest()
        call_next = AsyncMock()
        mock_response = MockResponse(status_code=403)
        call_next.return_value = mock_response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_test():
            response = await self.middleware.dispatch(request, call_next)
            return response
        
        response = loop.run_until_complete(run_test())
        loop.close()
        
        # Should log error response
        mock_logger.warning.assert_called()
        self.assertEqual(response.status_code, 403)


if __name__ == '__main__':
    unittest.main()
