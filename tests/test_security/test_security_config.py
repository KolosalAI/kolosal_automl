"""
Unit tests for Security Configuration Management

Tests cover:
- Environment-based configuration
- Security validation
- Configuration loading and saving
- Environment variable handling

Author: GitHub Copilot
Date: 2025-07-24
Version: 0.2.0
"""

import unittest
import tempfile
import shutil
import os
import json
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import directly from security_config to avoid __init__.py issues
from modules.security.security_config import (
    SecurityEnvironment,
    SecurityLevel,
    get_security_environment,
    setup_secure_environment
)


class TestSecurityLevel(unittest.TestCase):
    """Test SecurityLevel enum"""
    
    def test_security_levels(self):
        """Test all security levels are available"""
        expected_levels = ["development", "testing", "staging", "production"]
        
        for level in expected_levels:
            self.assertTrue(any(sl.value == level for sl in SecurityLevel))


class TestSecurityEnvironment(unittest.TestCase):
    """Test SecurityEnvironment configuration class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_default_initialization(self):
        """Test default SecurityEnvironment initialization"""
        env = SecurityEnvironment()
        
        # Default constructor uses secure defaults, not development defaults
        self.assertEqual(env.security_level, SecurityLevel.DEVELOPMENT)
        self.assertFalse(env.debug_mode)  # Secure default is False
        self.assertTrue(env.require_api_key)
        self.assertTrue(env.enable_rate_limiting)
    
    def test_production_config(self):
        """Test production security configuration"""
        env = SecurityEnvironment._production_config()
        
        self.assertEqual(env.security_level, SecurityLevel.PRODUCTION)
        self.assertFalse(env.debug_mode)
        self.assertTrue(env.require_api_key)
        self.assertTrue(env.api_key_rotation_enabled)
        self.assertTrue(env.enforce_https)
        self.assertEqual(env.rate_limit_requests, 50)  # Stricter in production
        self.assertEqual(env.jwt_expiry_hours, 1)  # Shorter expiry
        self.assertEqual(env.session_timeout, 1800)  # 30 minutes
    
    def test_development_config(self):
        """Test development security configuration"""
        env = SecurityEnvironment._development_config()
        
        self.assertEqual(env.security_level, SecurityLevel.DEVELOPMENT)
        self.assertTrue(env.debug_mode)
        self.assertFalse(env.require_api_key)
        self.assertFalse(env.enable_jwt)
        self.assertFalse(env.enable_rate_limiting)
        self.assertFalse(env.enforce_https)
        self.assertIn("*", env.allowed_origins)
    
    def test_staging_config(self):
        """Test staging security configuration"""
        env = SecurityEnvironment._staging_config()
        
        self.assertEqual(env.security_level, SecurityLevel.STAGING)
        self.assertFalse(env.debug_mode)
        self.assertTrue(env.require_api_key)
        self.assertTrue(env.enable_jwt)
        self.assertTrue(env.enforce_https)
        self.assertEqual(env.rate_limit_requests, 75)
    
    def test_testing_config(self):
        """Test testing security configuration"""
        env = SecurityEnvironment._testing_config()
        
        self.assertEqual(env.security_level, SecurityLevel.TESTING)
        self.assertTrue(env.debug_mode)
        self.assertFalse(env.require_api_key)
        self.assertFalse(env.enable_jwt)
        self.assertFalse(env.enable_rate_limiting)
        self.assertFalse(env.enforce_https)
    
    @patch.dict(os.environ, {
        'SECURITY_ENV': 'production',
        'SECURITY_REQUIRE_API_KEY': 'true',
        'SECURITY_ENABLE_HTTPS': 'true',
        'SECURITY_RATE_LIMIT_REQUESTS': '100',
        'SECURITY_ALLOWED_ORIGINS': 'https://example.com,https://api.example.com'
    })
    def test_from_environment_with_overrides(self):
        """Test loading configuration from environment variables"""
        env = SecurityEnvironment.from_environment()
        
        self.assertEqual(env.security_level, SecurityLevel.PRODUCTION)
        self.assertTrue(env.require_api_key)
        self.assertTrue(env.enforce_https)
        self.assertEqual(env.rate_limit_requests, 100)
        self.assertIn('https://example.com', env.allowed_origins)
        self.assertIn('https://api.example.com', env.allowed_origins)
    
    @patch.dict(os.environ, {'SECURITY_ENV': 'invalid_env'})
    def test_from_environment_invalid_env(self):
        """Test handling of invalid environment names"""
        env = SecurityEnvironment.from_environment()
        
        # Should default to development
        self.assertEqual(env.security_level, SecurityLevel.DEVELOPMENT)
    
    def test_validate_production_security(self):
        """Test validation of production security configuration"""
        # Valid production config
        env = SecurityEnvironment._production_config()
        env.allowed_origins = ["https://secure.example.com"]
        
        issues = env.validate()
        
        # Should have no issues for proper production config
        self.assertEqual(len(issues), 0)
        
        # Invalid production config
        env.require_api_key = False
        env.enforce_https = False
        env.enable_rate_limiting = False
        env.allowed_origins = ["*"]
        env.debug_mode = True
        
        issues = env.validate()
        
        # Should have multiple issues
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("API key" in issue for issue in issues))
        self.assertTrue(any("HTTPS" in issue for issue in issues))
        self.assertTrue(any("Rate limiting" in issue for issue in issues))
        self.assertTrue(any("Wildcard" in issue for issue in issues))
        self.assertTrue(any("Debug mode" in issue for issue in issues))
    
    def test_validate_general_settings(self):
        """Test validation of general settings"""
        env = SecurityEnvironment()
        
        # Invalid settings
        env.rate_limit_requests = -10
        env.rate_limit_window = 0
        env.jwt_expiry_hours = -5
        env.max_request_size = 0
        
        issues = env.validate()
        
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("Rate limit requests" in issue for issue in issues))
        self.assertTrue(any("Rate limit window" in issue for issue in issues))
        self.assertTrue(any("JWT expiry" in issue for issue in issues))
        self.assertTrue(any("Max request size" in issue for issue in issues))
    
    @patch('modules.security.security_config.get_secrets_manager')
    def test_get_api_keys(self, mock_secrets_manager):
        """Test API key retrieval from secrets manager"""
        # Mock secrets manager
        mock_manager = MagicMock()
        mock_manager.list_secrets.return_value = []
        mock_manager.get_secret.return_value = None
        mock_secrets_manager.return_value = mock_manager
        
        env = SecurityEnvironment()
        env.require_api_key = True
        
        with patch.dict(os.environ, {'API_KEYS': 'key1,key2,key3'}):
            api_keys = env.get_api_keys()
            
            self.assertEqual(len(api_keys), 3)
            self.assertIn('key1', api_keys)
            self.assertIn('key2', api_keys)
            self.assertIn('key3', api_keys)
    
    @patch('modules.security.security_config.get_secrets_manager')
    def test_get_jwt_secret(self, mock_secrets_manager):
        """Test JWT secret retrieval"""
        # Mock secrets manager
        mock_manager = MagicMock()
        mock_manager.get_secret.return_value = None
        mock_secrets_manager.return_value = mock_manager
        
        env = SecurityEnvironment()
        env.enable_jwt = True
        
        with patch.dict(os.environ, {'JWT_SECRET': 'test_jwt_secret'}):
            jwt_secret = env.get_jwt_secret()
            
            self.assertEqual(jwt_secret, 'test_jwt_secret')
    
    def test_to_dict_serialization(self):
        """Test serialization to dictionary"""
        env = SecurityEnvironment()
        
        config_dict = env.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['security_level'], SecurityLevel.DEVELOPMENT.value)
        self.assertIn('debug_mode', config_dict)
        self.assertIn('require_api_key', config_dict)
        self.assertIn('allowed_origins', config_dict)
    
    def test_save_and_load_from_file(self):
        """Test saving and loading configuration from file"""
        env = SecurityEnvironment._production_config()
        config_file = os.path.join(self.test_dir, "security_config.json")
        
        # Save configuration
        env.save_to_file(config_file)
        
        self.assertTrue(os.path.exists(config_file))
        
        # Load configuration
        loaded_env = SecurityEnvironment.load_from_file(config_file)
        
        self.assertEqual(loaded_env.security_level, SecurityLevel.PRODUCTION)
        self.assertEqual(loaded_env.require_api_key, env.require_api_key)
        self.assertEqual(loaded_env.enforce_https, env.enforce_https)
    
    def test_boolean_environment_parsing(self):
        """Test parsing of boolean environment variables"""
        env = SecurityEnvironment()
        
        # Test various boolean representations
        boolean_tests = [
            ('true', True),
            ('false', False),
            ('1', True),
            ('0', False),
            ('yes', True),
            ('no', False),
            ('on', True),
            ('off', False),
            ('TRUE', True),
            ('FALSE', False)
        ]
        
        for env_value, expected in boolean_tests:
            with patch.dict(os.environ, {'SECURITY_REQUIRE_API_KEY': env_value}):
                test_env = SecurityEnvironment.from_environment()
                self.assertEqual(test_env.require_api_key, expected,
                               f"Failed for value '{env_value}', expected {expected}")
    
    def test_list_environment_parsing(self):
        """Test parsing of list environment variables"""
        test_origins = "https://app.example.com,https://api.example.com,https://admin.example.com"
        
        with patch.dict(os.environ, {'SECURITY_ALLOWED_ORIGINS': test_origins}):
            env = SecurityEnvironment.from_environment()
            
            expected_origins = [
                "https://app.example.com",
                "https://api.example.com", 
                "https://admin.example.com"
            ]
            
            self.assertEqual(env.allowed_origins, expected_origins)
    
    def test_integer_environment_parsing(self):
        """Test parsing of integer environment variables"""
        with patch.dict(os.environ, {
            'SECURITY_RATE_LIMIT_REQUESTS': '200',
            'SECURITY_JWT_EXPIRY_HOURS': '24',
            'SECURITY_SESSION_TIMEOUT': '7200'
        }):
            env = SecurityEnvironment.from_environment()
            
            self.assertEqual(env.rate_limit_requests, 200)
            self.assertEqual(env.jwt_expiry_hours, 24)
            self.assertEqual(env.session_timeout, 7200)
    
    def test_invalid_integer_environment_handling(self):
        """Test handling of invalid integer environment variables"""
        with patch.dict(os.environ, {'SECURITY_RATE_LIMIT_REQUESTS': 'invalid_number'}):
            # Should not raise exception, should use default value
            env = SecurityEnvironment.from_environment()
            
            # Should use default value, not the invalid one
            self.assertIsInstance(env.rate_limit_requests, int)


class TestGlobalSecurityEnvironment(unittest.TestCase):
    """Test global security environment functions"""
    
    def setUp(self):
        """Set up test environment"""
        # Reset global environment
        import modules.security.security_config as sc
        sc._security_env = None
    
    def tearDown(self):
        """Clean up test environment"""
        # Reset global environment
        import modules.security.security_config as sc
        sc._security_env = None
    
    def test_get_security_environment_singleton(self):
        """Test that get_security_environment returns singleton"""
        env1 = get_security_environment()
        env2 = get_security_environment()
        
        self.assertIs(env1, env2)
    
    def test_get_security_environment_force_reload(self):
        """Test force reloading security environment"""
        env1 = get_security_environment()
        env2 = get_security_environment(force_reload=True)
        
        # Should be different instances after force reload
        self.assertIsNot(env1, env2)
    
    @patch.dict(os.environ, {'SECURITY_ENV': 'production'})
    def test_get_security_environment_with_env_name(self):
        """Test getting security environment with specific environment name"""
        env = get_security_environment(env_name='staging')
        
        # Should use the provided env_name, not the environment variable
        self.assertEqual(env.security_level, SecurityLevel.STAGING)
    
    @patch('modules.security.security_config.logging')
    def test_setup_secure_environment_logging(self, mock_logging):
        """Test that setup_secure_environment configures logging"""
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger
        
        env = setup_secure_environment()
        
        # Should have configured logging if log_security_events is True
        if env.log_security_events:
            mock_logging.getLogger.assert_called_with("kolosal_security")


class TestSecurityEnvironmentIntegration(unittest.TestCase):
    """Integration tests for security environment"""
    
    def test_production_environment_complete_setup(self):
        """Test complete production environment setup"""
        with patch.dict(os.environ, {
            'SECURITY_ENV': 'production',
            'SECURITY_ALLOWED_ORIGINS': 'https://secure.example.com',
            'API_KEYS': 'prod_key_123,prod_key_456',
            'JWT_SECRET': 'production_jwt_secret_very_long_and_secure'
        }):
            env = get_security_environment(force_reload=True)  # Force reload to pick up env vars
            
            # Validate production setup
            self.assertEqual(env.security_level, SecurityLevel.PRODUCTION)
            self.assertTrue(env.require_api_key)
            self.assertTrue(env.enforce_https)
            self.assertTrue(env.enable_rate_limiting)
            
            # Check that API keys are accessible
            api_keys = env.get_api_keys()
            self.assertGreater(len(api_keys), 0)
            
            # Check JWT secret
            jwt_secret = env.get_jwt_secret()
            self.assertIsNotNone(jwt_secret)
    
    def test_development_environment_complete_setup(self):
        """Test complete development environment setup"""
        with patch.dict(os.environ, {'SECURITY_ENV': 'development'}):
            env = get_security_environment(force_reload=True)
            
            # Validate development setup
            self.assertEqual(env.security_level, SecurityLevel.DEVELOPMENT)
            self.assertFalse(env.require_api_key)
            self.assertFalse(env.enforce_https)
            self.assertFalse(env.enable_rate_limiting)
            self.assertTrue(env.debug_mode)
    
    def test_configuration_validation_workflow(self):
        """Test complete configuration validation workflow"""
        # Create production config with issues
        with patch.dict(os.environ, {
            'SECURITY_ENV': 'production',
            'SECURITY_ALLOWED_ORIGINS': '*',  # Security issue
            'SECURITY_DEBUG_MODE': 'true'     # Security issue
        }):
            env = get_security_environment()
            issues = env.validate()
            
            # Should identify security issues
            self.assertGreater(len(issues), 0)
            
            # Check specific issues
            issue_text = " ".join(issues)
            self.assertIn("wildcard", issue_text.lower())
            self.assertIn("debug", issue_text.lower())


if __name__ == '__main__':
    unittest.main()
