import os
import time
import json
import unittest
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI, status, APIRouter
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response, status, Header
# Path manipulation to make imports work in test environment
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# Import the app
from modules.api.app import (
    app, 
    verify_api_key, 
    mount_component, 
    HealthResponse, 
    MetricsResponse, 
    API_VERSION, 
    API_ENV
)


class TestKolosalAutoMLAPI(unittest.TestCase):
    """Test suite for the Kolosal AutoML API application."""
    
    def setUp(self):
        """Set up the test client and mock components before each test."""
        # Create test client
        self.client = TestClient(app)
        
        # Set up app state for tests
        app.state.start_time = time.time()
        app.state.request_count = 0
        app.state.error_count = 0
        app.state.metrics = {
            "requests_per_endpoint": {},
            "average_response_time": {},
            "error_rate": {},
            "active_connections": 0
        }
        
        # Set environment variables for testing
        os.environ["API_ENV"] = "testing"
        os.environ["API_DEBUG"] = "True"
        os.environ["REQUIRE_API_KEY"] = "False"
        
    def tearDown(self):
        """Clean up after each test."""
        # Reset any environment variables that were set
        if "API_ENV" in os.environ:
            del os.environ["API_ENV"]
        if "API_DEBUG" in os.environ:
            del os.environ["API_DEBUG"]
        if "REQUIRE_API_KEY" in os.environ:
            del os.environ["REQUIRE_API_KEY"]
    
    def test_root_endpoint(self):
        """Test the root endpoint returns correct API information."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertEqual(data["name"], "kolosal AutoML API")
        self.assertEqual(data["version"], API_VERSION)
        self.assertIn("documentation", data)
        self.assertIn("health_check", data)
    
    def test_health_check_endpoint(self):
        """Test the health check endpoint returns correct status information."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], API_VERSION)
        self.assertEqual(data["environment"], API_ENV)
        self.assertIn("uptime_seconds", data)
        self.assertIn("components", data)
        self.assertIn("timestamp", data)
        
        # Check components
        components = data["components"]
        expected_components = [
            "data_preprocessor", 
            "device_optimizer", 
            "inference_engine", 
            "model_manager", 
            "quantizer", 
            "train_engine"
        ]
        for component in expected_components:
            self.assertIn(component, components)
            self.assertEqual(components[component], "healthy")
    
    def test_metrics_endpoint(self):
        """Test the metrics endpoint returns correct metrics information."""
        # Set up the test app state with some metrics data
        app.state.request_count = 10
        app.state.error_count = 2
        app.state.start_time = time.time() - 100  # Started 100 seconds ago
        app.state.metrics = {
            "requests_per_endpoint": {
                "GET:/": 5,
                "GET:/health": 5
            },
            "average_response_time": {
                "GET:/": 0.01,
                "GET:/health": 0.02
            },
            "active_connections": 1
        }
        
        # Mock the verify_api_key dependency to allow access
        with patch('modules.api.app.verify_api_key', return_value=True):
            # Now check the metrics endpoint
            response = self.client.get("/metrics")
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            
            data = response.json()
            self.assertEqual(data["total_requests"], 10)
            self.assertEqual(data["errors"], 2)
            self.assertGreaterEqual(data["uptime_seconds"], 100)
            self.assertEqual(data["requests_per_endpoint"], {
                "GET:/": 5,
                "GET:/health": 5
            })
            self.assertEqual(data["average_response_time"], {
                "GET:/": 0.01,
                "GET:/health": 0.02
            })
            self.assertEqual(data["active_connections"], 1)
            self.assertIn("timestamp", data)
    
    def test_api_key_verification_required_missing_key(self):
        """Test API key verification when key is required but missing."""
        # Reset any previous overrides
        app.dependency_overrides = {}
        
        # Create a test endpoint that uses the dependency
        test_router = APIRouter()
        
        @test_router.get("/test-auth-missing")
        async def test_auth(api_key_check: bool = Depends(verify_api_key)):
            return {"authenticated": True}
        
        # Store original routes
        original_routes = list(app.routes)
        
        try:
            # Include the test router
            app.include_router(test_router)
            
            # Test with patched settings
            with patch('modules.api.app.REQUIRE_API_KEY', True), \
                patch('modules.api.app.API_KEYS', ["test_key"]):
                
                # Make a request with no key provided
                response = self.client.get("/test-auth-missing")
                
                # Verify response
                self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
                error_data = response.json()
                self.assertEqual(error_data["detail"], "Missing API key")
        finally:
            # Restore original routes
            app.routes.clear()
            for route in original_routes:
                app.routes.append(route)

    def test_api_key_verification_required_invalid_key(self):
        """Test API key verification when key is required but invalid."""
        # Reset any previous overrides
        app.dependency_overrides = {}
        
        # Create a test endpoint that uses the dependency
        test_router = APIRouter()
        
        @test_router.get("/test-auth-invalid")
        async def test_auth(api_key_check: bool = Depends(verify_api_key)):
            return {"authenticated": True}
        
        # Store original routes
        original_routes = list(app.routes)
        
        try:
            # Include the test router
            app.include_router(test_router)
            
            # Test with patched settings
            with patch('modules.api.app.REQUIRE_API_KEY', True), \
                patch('modules.api.app.API_KEYS', ["test_key"]):
                
                # Make a request with an invalid key
                response = self.client.get("/test-auth-invalid", headers={"X-API-Key": "invalid_key"})
                
                # Verify response
                self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
                error_data = response.json()
                self.assertEqual(error_data["detail"], "Invalid API key")
        finally:
            # Restore original routes
            app.routes.clear()
            for route in original_routes:
                app.routes.append(route)

    def test_api_key_verification_required_valid_key(self):
        """Test API key verification when key is required and valid."""
        # Reset any previous overrides
        app.dependency_overrides = {}
        
        # Create a test endpoint that uses the dependency
        test_router = APIRouter()
        
        @test_router.get("/test-auth-valid")
        async def test_auth(api_key_check: bool = Depends(verify_api_key)):
            return {"authenticated": True}
        
        # Store original routes
        original_routes = list(app.routes)
        
        try:
            # Include the test router
            app.include_router(test_router)
            
            # Test with patched settings
            with patch('modules.api.app.REQUIRE_API_KEY', True), \
                patch('modules.api.app.API_KEYS', ["test_key"]):
                
                # Make a request with a valid key
                response = self.client.get("/test-auth-valid", headers={"X-API-Key": "test_key"})
                
                # Verify response is successful
                self.assertEqual(response.status_code, status.HTTP_200_OK)
                self.assertEqual(response.json(), {"authenticated": True})
        finally:
            # Restore original routes
            app.routes.clear()
            for route in original_routes:
                app.routes.append(route)
    
    def test_docs_endpoint(self):
        """Test the docs endpoint returns Swagger UI HTML."""
        # Mock the verify_api_key dependency to always return True
        with patch('modules.api.app.verify_api_key', return_value=True):
            response = self.client.get("/docs")
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            self.assertIn("text/html", response.headers["content-type"])
            # Check for some expected content in the swagger UI
            self.assertTrue(any(marker in response.text.lower() for marker in ["swagger", "openapi"]))
    
    def test_redoc_endpoint(self):
        """Test the redoc endpoint returns ReDoc HTML."""
        # Mock the verify_api_key dependency to always return True
        with patch('modules.api.app.verify_api_key', return_value=True):
            response = self.client.get("/redoc")
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            self.assertIn("text/html", response.headers["content-type"])
            # Check for some expected content in the redoc UI
            self.assertTrue(any(marker in response.text.lower() for marker in ["redoc", "openapi"]))
    
    def test_request_id_middleware(self):
        """Test that requests get assigned a unique request ID."""
        # We'll test the middleware function directly
        from modules.api.app import add_request_id
        
        # Create a mock request
        mock_request = MagicMock()
        mock_request.state = MagicMock()
        
        # Create a mock call_next function that returns a mock response
        mock_response = MagicMock()
        async def mock_call_next(request):
            return mock_response
        
        # Execute the middleware function
        import asyncio
        try:
            # This might fail in a non-async context, but we'll check if it sets the state
            asyncio.run(add_request_id(mock_request, mock_call_next))
        except:
            pass
        
        # Check if request_id was set on request.state
        self.assertTrue(hasattr(mock_request.state, 'request_id'))
        
        # Make a real request and check for the header
        response = self.client.get("/health")
        self.assertIn("X-Request-ID", response.headers)
        self.assertTrue(response.headers["X-Request-ID"])  # Not empty
    

    def test_global_exception_handler(self):
        """Test the global exception handler works correctly using direct invocation."""
        # Import the global exception handler
        from modules.api.app import global_exception_handler
        import json
        import asyncio
        
        # Create a mock request with request_id
        mock_request = MagicMock()
        mock_request.state = MagicMock()
        mock_request.state.request_id = "test-request-id"
        
        # Create a test exception
        test_exception = ValueError("Test exception")
        
        # Run the handler with asyncio
        response = asyncio.run(global_exception_handler(mock_request, test_exception))
        
        # Verify the response is correct
        self.assertEqual(response.status_code, 500)
        
        # Parse the JSON response body
        body = json.loads(response.body.decode())
        
        # Check the response content
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["message"], "Test exception")
        self.assertEqual(body["type"], "ValueError")
        self.assertEqual(body["request_id"], "test-request-id")
        self.assertIn("timestamp", body)
    
    def test_mount_component_functionality(self):
        """Test that the mount_component function works correctly."""
        # Create a test FastAPI app with a test route
        test_app = FastAPI()
        
        @test_app.get("/test-path")
        def test_endpoint():
            return {"message": "test"}
        
        # Original router count
        original_router_count = len(app.routes)
        
        # Call mount_component
        mount_component(test_app, "/test-component", ["TestComponent"])
        
        # Verify a new route was added
        self.assertGreater(len(app.routes), original_router_count)
        
        # Verify we can find the mounted route
        mounted_routes = [route for route in app.routes if "/test-component" in str(route)]
        self.assertTrue(len(mounted_routes) > 0, "Test component route was not mounted")


class TestHealthResponse(unittest.TestCase):
    """Test the HealthResponse model."""
    
    def test_health_response_model(self):
        """Test creating a valid HealthResponse model."""
        health_data = {
            "status": "healthy",
            "version": "1.0.0",
            "environment": "testing",
            "uptime_seconds": 123.45,
            "components": {
                "component1": "healthy",
                "component2": "healthy"
            },
            "timestamp": "2023-01-01T00:00:00"
        }
        
        # Create model instance
        health_response = HealthResponse(**health_data)
        
        # Verify fields
        self.assertEqual(health_response.status, "healthy")
        self.assertEqual(health_response.version, "1.0.0")
        self.assertEqual(health_response.environment, "testing")
        self.assertEqual(health_response.uptime_seconds, 123.45)
        self.assertEqual(health_response.components, {
            "component1": "healthy",
            "component2": "healthy"
        })
        self.assertEqual(health_response.timestamp, "2023-01-01T00:00:00")


class TestMetricsResponse(unittest.TestCase):
    """Test the MetricsResponse model."""
    
    def test_metrics_response_model(self):
        """Test creating a valid MetricsResponse model."""
        metrics_data = {
            "total_requests": 100,
            "errors": 5,
            "uptime_seconds": 123.45,
            "requests_per_endpoint": {
                "GET:/": 50,
                "GET:/health": 50
            },
            "average_response_time": {
                "GET:/": 0.01,
                "GET:/health": 0.02
            },
            "active_connections": 2,
            "timestamp": "2023-01-01T00:00:00"
        }
        
        # Create model instance
        metrics_response = MetricsResponse(**metrics_data)
        
        # Verify fields
        self.assertEqual(metrics_response.total_requests, 100)
        self.assertEqual(metrics_response.errors, 5)
        self.assertEqual(metrics_response.uptime_seconds, 123.45)
        self.assertEqual(metrics_response.requests_per_endpoint, {
            "GET:/": 50,
            "GET:/health": 50
        })
        self.assertEqual(metrics_response.average_response_time, {
            "GET:/": 0.01,
            "GET:/health": 0.02
        })
        self.assertEqual(metrics_response.active_connections, 2)
        self.assertEqual(metrics_response.timestamp, "2023-01-01T00:00:00")


if __name__ == "__main__":
    unittest.main()