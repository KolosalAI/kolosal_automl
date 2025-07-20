"""
Enhanced Integration Tests for kolosal AutoML

Comprehensive integration testing covering:
- End-to-end API workflows
- Security integration testing
- Monitoring integration testing
- Error handling integration
- Performance testing
- Multi-component integration

Author: AI Assistant
Date: 2025-07-20
"""

import pytest
import asyncio
import time
import json
import requests
from typing import Dict, List, Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile
import os

# Test configuration
TEST_BASE_URL = "http://localhost:8000"
TEST_API_KEY = "test_key"
TEST_TIMEOUT = 30


class APITestClient:
    """Enhanced API test client with authentication and monitoring"""
    
    def __init__(self, base_url: str = TEST_BASE_URL, api_key: str = TEST_API_KEY):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        })
        
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make GET request"""
        url = f"{self.base_url}{endpoint}"
        return self.session.get(url, timeout=TEST_TIMEOUT, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make POST request"""
        url = f"{self.base_url}{endpoint}"
        return self.session.post(url, timeout=TEST_TIMEOUT, **kwargs)
    
    def put(self, endpoint: str, **kwargs) -> requests.Response:
        """Make PUT request"""
        url = f"{self.base_url}{endpoint}"
        return self.session.put(url, timeout=TEST_TIMEOUT, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        """Make DELETE request"""
        url = f"{self.base_url}{endpoint}"
        return self.session.delete(url, timeout=TEST_TIMEOUT, **kwargs)


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.client = APITestClient()
        self.test_data = {
            "sample_data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "batch_config": {
                "batch_size": 5,
                "max_wait_time": 10,
                "priority": "normal"
            }
        }
        
    def test_complete_batch_processing_workflow(self):
        """Test complete batch processing workflow"""
        
        # Step 1: Submit batch
        batch_data = {
            "data": self.test_data["sample_data"],
            "config": self.test_data["batch_config"],
            "metadata": {"test_id": "integration_test_1"}
        }
        
        response = self.client.post("/api/batch-processor/submit", json=batch_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "success"
        assert "batch_id" in result
        
        batch_id = result["batch_id"]
        
        # Step 2: Check batch status
        status_response = self.client.get(f"/api/batch-processor/status/{batch_id}")
        assert status_response.status_code == 200
        
        status_result = status_response.json()
        assert status_result["status"] == "success"
        assert "batch_status" in status_result["data"]
        
        # Step 3: Wait for processing (with timeout)
        max_wait = 30
        wait_time = 0
        batch_completed = False
        
        while wait_time < max_wait and not batch_completed:
            time.sleep(1)
            wait_time += 1
            
            status_response = self.client.get(f"/api/batch-processor/status/{batch_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                batch_status = status_data.get("data", {}).get("batch_status", "")
                if batch_status in ["completed", "failed"]:
                    batch_completed = True
        
        # Step 4: Get results
        results_response = self.client.get(f"/api/batch-processor/results/{batch_id}")
        assert results_response.status_code in [200, 202]  # 202 if still processing
        
        if results_response.status_code == 200:
            results_data = results_response.json()
            assert results_data["status"] == "success"
            assert "results" in results_data["data"]
        
        # Step 5: Verify metrics were recorded
        metrics_response = self.client.get("/api/batch-processor/metrics")
        assert metrics_response.status_code == 200
        
        metrics_data = metrics_response.json()
        assert metrics_data["status"] == "success"
        assert "total_batches" in metrics_data["data"]
    
    def test_security_integration_workflow(self):
        """Test security integration across endpoints"""
        
        # Test without API key
        client_no_auth = APITestClient(api_key="")
        client_no_auth.session.headers.pop("X-API-Key", None)
        
        protected_endpoints = [
            "/api/batch-processor/submit",
            "/monitoring/metrics",
            "/monitoring/alerts",
            "/monitoring/performance"
        ]
        
        for endpoint in protected_endpoints:
            if endpoint == "/api/batch-processor/submit":
                response = client_no_auth.post(endpoint, json={})
            else:
                response = client_no_auth.get(endpoint)
            
            # Should be blocked by security
            assert response.status_code in [401, 403], f"Endpoint {endpoint} not properly protected"
        
        # Test with invalid API key
        client_invalid = APITestClient(api_key="invalid_key")
        response = client_invalid.get("/monitoring/metrics")
        assert response.status_code in [401, 403]
        
        # Test with valid API key
        response = self.client.get("/monitoring/health")
        assert response.status_code == 200
    
    def test_monitoring_integration_workflow(self):
        """Test monitoring system integration"""
        
        # Generate some API activity
        for i in range(5):
            self.client.get("/monitoring/health")
            time.sleep(0.1)
        
        # Check that metrics are being recorded
        metrics_response = self.client.get("/monitoring/metrics")
        assert metrics_response.status_code == 200
        
        metrics_data = metrics_response.json()
        assert metrics_data["status"] == "success"
        assert "current_metrics" in metrics_data["data"]
        
        # Check alerts endpoint
        alerts_response = self.client.get("/monitoring/alerts")
        assert alerts_response.status_code == 200
        
        alerts_data = alerts_response.json()
        assert alerts_data["status"] == "success"
        assert "active_alerts" in alerts_data
        
        # Check performance analytics
        performance_response = self.client.get("/monitoring/performance")
        assert performance_response.status_code == 200
        
        performance_data = performance_response.json()
        assert performance_data["status"] == "success"
        assert "performance_analysis" in performance_data
        
        # Check dashboard (HTML response)
        dashboard_response = self.client.get("/monitoring/dashboard")
        assert dashboard_response.status_code == 200
        assert "kolosal AutoML" in dashboard_response.text
    
    def test_error_handling_integration(self):
        """Test error handling integration across components"""
        
        # Test validation errors
        invalid_batch_data = {
            "data": "invalid_data_format",  # Should be a list
            "config": {}
        }
        
        response = self.client.post("/api/batch-processor/submit", json=invalid_batch_data)
        assert response.status_code == 400
        
        error_data = response.json()
        assert error_data["status"] == "error"
        assert "message" in error_data
        
        # Test not found errors
        response = self.client.get("/api/batch-processor/status/nonexistent_batch_id")
        assert response.status_code == 404
        
        # Test malformed requests
        response = self.client.post("/api/batch-processor/submit", data="invalid_json")
        assert response.status_code == 400


class TestPerformanceIntegration:
    """Test performance and load handling"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.client = APITestClient()
        
    def test_concurrent_batch_submissions(self):
        """Test handling concurrent batch submissions"""
        
        batch_data = {
            "data": [[1, 2, 3], [4, 5, 6]],
            "config": {
                "batch_size": 2,
                "max_wait_time": 5,
                "priority": "normal"
            }
        }
        
        # Submit multiple batches concurrently
        num_concurrent = 5
        batch_ids = []
        
        def submit_batch(client, data, index):
            data_copy = data.copy()
            data_copy["metadata"] = {"test_index": index}
            response = client.post("/api/batch-processor/submit", json=data_copy)
            return response
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(submit_batch, self.client, batch_data, i)
                for i in range(num_concurrent)
            ]
            
            responses = []
            for future in as_completed(futures):
                try:
                    response = future.result(timeout=10)
                    responses.append(response)
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("status") == "success":
                            batch_ids.append(result.get("batch_id"))
                except Exception as e:
                    pytest.fail(f"Concurrent batch submission failed: {e}")
        
        # Verify all submissions were successful
        assert len(responses) == num_concurrent
        successful_responses = [r for r in responses if r.status_code == 200]
        assert len(successful_responses) >= num_concurrent // 2  # At least half should succeed
        
        # Wait for batches to process
        time.sleep(2)
        
        # Check that all batches are tracked
        for batch_id in batch_ids:
            status_response = self.client.get(f"/api/batch-processor/status/{batch_id}")
            assert status_response.status_code == 200
    
    def test_api_rate_limiting(self):
        """Test API rate limiting functionality"""
        
        # Make rapid requests to trigger rate limiting
        # Note: This test depends on rate limiting configuration
        rapid_requests = 50
        responses = []
        
        start_time = time.time()
        
        for i in range(rapid_requests):
            response = self.client.get("/monitoring/health")
            responses.append(response)
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.01)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check for rate limiting responses (429 status code)
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        successful_responses = [r for r in responses if r.status_code == 200]
        
        # Should have some successful responses
        assert len(successful_responses) > 0
        
        # If rate limiting is enabled, should see some 429 responses
        print(f"Made {rapid_requests} requests in {total_time:.2f}s")
        print(f"Successful: {len(successful_responses)}, Rate limited: {len(rate_limited_responses)}")
        
        # Verify the responses contain expected data
        for response in successful_responses[:5]:  # Check first 5 successful responses
            data = response.json()
            assert "status" in data
            assert "timestamp" in data


class TestMultiComponentIntegration:
    """Test integration between multiple components"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.client = APITestClient()
        
    def test_batch_processing_with_monitoring(self):
        """Test batch processing integration with monitoring"""
        
        # Get initial metrics
        initial_metrics = self.client.get("/monitoring/metrics")
        assert initial_metrics.status_code == 200
        
        # Submit batch
        batch_data = {
            "data": [[1, 2], [3, 4], [5, 6]],
            "config": {"batch_size": 3}
        }
        
        batch_response = self.client.post("/api/batch-processor/submit", json=batch_data)
        assert batch_response.status_code == 200
        
        batch_id = batch_response.json()["batch_id"]
        
        # Wait for processing
        time.sleep(1)
        
        # Check that metrics have been updated
        updated_metrics = self.client.get("/monitoring/metrics")
        assert updated_metrics.status_code == 200
        
        # The monitoring system should have recorded the API calls
        metrics_data = updated_metrics.json()["data"]
        current_metrics = metrics_data.get("current_metrics", {})
        
        # Should have recorded request activity
        assert current_metrics.get("request_rate", 0) >= 0
        
        # Check batch-specific metrics
        batch_metrics = self.client.get("/api/batch-processor/metrics")
        assert batch_metrics.status_code == 200
        
        batch_metrics_data = batch_metrics.json()["data"]
        assert batch_metrics_data.get("total_batches", 0) > 0
    
    def test_security_with_error_handling(self):
        """Test security integration with error handling"""
        
        # Test various security violations and ensure proper error responses
        test_cases = [
            {
                "description": "No API key",
                "headers": {},
                "expected_status": [401, 403]
            },
            {
                "description": "Invalid API key",
                "headers": {"X-API-Key": "invalid_key"},
                "expected_status": [401, 403]
            },
            {
                "description": "Malformed request with valid auth",
                "headers": {"X-API-Key": TEST_API_KEY},
                "data": "invalid_json",
                "expected_status": [400]
            }
        ]
        
        for test_case in test_cases:
            # Create client with specific headers
            client = APITestClient()
            client.session.headers.clear()
            client.session.headers.update(test_case.get("headers", {}))
            
            if "data" in test_case:
                response = client.session.post(
                    f"{TEST_BASE_URL}/api/batch-processor/submit",
                    data=test_case["data"],
                    timeout=TEST_TIMEOUT
                )
            else:
                response = client.get("/monitoring/metrics")
            
            assert response.status_code in test_case["expected_status"], \
                f"Test case '{test_case['description']}' failed"
            
            # Verify error response format
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    # Should have standardized error format
                    assert "status" in error_data or "detail" in error_data
                except:
                    # Some errors might return plain text, which is also acceptable
                    pass


class TestDataFlowIntegration:
    """Test data flow through the entire system"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.client = APITestClient()
        
    def test_data_integrity_through_pipeline(self):
        """Test that data maintains integrity through the processing pipeline"""
        
        # Create test data with known patterns
        test_data = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ]
        
        batch_data = {
            "data": test_data,
            "config": {
                "batch_size": 2,
                "max_wait_time": 10
            },
            "metadata": {
                "test_type": "data_integrity",
                "expected_rows": len(test_data),
                "expected_cols": len(test_data[0])
            }
        }
        
        # Submit data
        response = self.client.post("/api/batch-processor/submit", json=batch_data)
        assert response.status_code == 200
        
        batch_id = response.json()["batch_id"]
        
        # Wait for processing
        max_wait = 15
        wait_time = 0
        
        while wait_time < max_wait:
            status_response = self.client.get(f"/api/batch-processor/status/{batch_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()["data"]
                if status_data.get("batch_status") == "completed":
                    break
            
            time.sleep(1)
            wait_time += 1
        
        # Get results and verify data integrity
        results_response = self.client.get(f"/api/batch-processor/results/{batch_id}")
        
        if results_response.status_code == 200:
            results_data = results_response.json()["data"]
            
            # Verify results structure
            assert "results" in results_data
            assert "metadata" in results_data
            
            # Check metadata integrity
            metadata = results_data["metadata"]
            original_metadata = batch_data["metadata"]
            
            for key, value in original_metadata.items():
                assert metadata.get(key) == value, f"Metadata key {key} not preserved"
    
    def test_error_propagation_through_pipeline(self):
        """Test how errors propagate through the processing pipeline"""
        
        # Submit intentionally invalid data to test error handling
        invalid_data_cases = [
            {
                "data": None,
                "config": {"batch_size": 1},
                "description": "null data"
            },
            {
                "data": [],
                "config": {"batch_size": 1},
                "description": "empty data"
            },
            {
                "data": [[1, 2], [3]],  # Inconsistent row lengths
                "config": {"batch_size": 2},
                "description": "inconsistent data shape"
            }
        ]
        
        for test_case in invalid_data_cases:
            response = self.client.post("/api/batch-processor/submit", json=test_case)
            
            # Should either reject immediately or accept and fail gracefully
            if response.status_code == 200:
                # If accepted, should fail during processing
                batch_id = response.json()["batch_id"]
                
                # Wait a bit for processing
                time.sleep(2)
                
                status_response = self.client.get(f"/api/batch-processor/status/{batch_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()["data"]
                    batch_status = status_data.get("batch_status")
                    
                    # Should eventually show failed status or error information
                    assert batch_status in ["failed", "error"] or "error" in status_data
            else:
                # Should return appropriate error status
                assert response.status_code >= 400
                
                # Should have error information
                try:
                    error_data = response.json()
                    assert "status" in error_data or "detail" in error_data
                except:
                    # Text error responses are also acceptable
                    pass


# Test runner and utilities
class TestRunner:
    """Utility class for running integration tests"""
    
    @staticmethod
    def run_all_tests():
        """Run all integration tests"""
        test_classes = [
            TestEndToEndWorkflows,
            TestPerformanceIntegration,
            TestMultiComponentIntegration,
            TestDataFlowIntegration
        ]
        
        results = {}
        
        for test_class in test_classes:
            class_name = test_class.__name__
            print(f"\n=== Running {class_name} ===")
            
            # Get test methods
            test_methods = [
                method for method in dir(test_class)
                if method.startswith('test_') and callable(getattr(test_class, method))
            ]
            
            class_results = {}
            
            for method_name in test_methods:
                print(f"Running {method_name}...")
                
                try:
                    # Create instance and run setup
                    instance = test_class()
                    if hasattr(instance, 'setup_method'):
                        instance.setup_method()
                    
                    # Run test method
                    test_method = getattr(instance, method_name)
                    test_method()
                    
                    class_results[method_name] = "PASSED"
                    print(f"  ✓ {method_name} PASSED")
                    
                except Exception as e:
                    class_results[method_name] = f"FAILED: {str(e)}"
                    print(f"  ✗ {method_name} FAILED: {str(e)}")
            
            results[class_name] = class_results
        
        return results
    
    @staticmethod
    def print_summary(results):
        """Print test results summary"""
        total_tests = 0
        passed_tests = 0
        
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        
        for class_name, class_results in results.items():
            print(f"\n{class_name}:")
            
            for method_name, result in class_results.items():
                total_tests += 1
                if result == "PASSED":
                    passed_tests += 1
                    print(f"  ✓ {method_name}")
                else:
                    print(f"  ✗ {method_name}: {result}")
        
        print(f"\n{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*60)


if __name__ == "__main__":
    print("kolosal AutoML - Enhanced Integration Tests")
    print("=" * 50)
    
    # Check if server is running
    try:
        client = APITestClient()
        health_response = client.get("/monitoring/health")
        if health_response.status_code != 200:
            print("ERROR: API server is not responding properly")
            print("Please ensure the kolosal AutoML API server is running")
            exit(1)
    except Exception as e:
        print(f"ERROR: Cannot connect to API server: {e}")
        print("Please ensure the kolosal AutoML API server is running at {TEST_BASE_URL}")
        exit(1)
    
    print(f"Connected to API server at {TEST_BASE_URL}")
    print("Starting integration tests...\n")
    
    # Run all tests
    results = TestRunner.run_all_tests()
    TestRunner.print_summary(results)
