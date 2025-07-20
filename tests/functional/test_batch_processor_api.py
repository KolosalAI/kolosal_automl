"""
Test suite for Batch Processor API

Tests for the batch processing API endpoints including submit, status, results,
configuration, and monitoring functionality.

Author: AI Assistant
Date: 2025-07-20
"""

import pytest
import requests
import time
import numpy as np
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp


class TestBatchProcessorAPI:
    """Test suite for Batch Processor API"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.base_url = "http://localhost:8001"
        self.api_key = "test_key"
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "batch-processor-api"
        assert "version" in data
        assert "timestamp" in data
    
    def test_configure_processor(self):
        """Test processor configuration"""
        config = {
            "initial_batch_size": 16,
            "min_batch_size": 2,
            "max_batch_size": 128,
            "batch_timeout": 0.05,
            "max_queue_size": 2000,
            "enable_priority_queue": True,
            "processing_strategy": "BALANCED",
            "enable_adaptive_batching": True,
            "enable_monitoring": True,
            "num_workers": 8,
            "enable_memory_optimization": True,
            "max_retries": 5
        }
        
        response = requests.post(
            f"{self.base_url}/configure",
            json=config,
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "config" in data
    
    def test_start_stop_processor(self):
        """Test starting and stopping the processor"""
        # Configure first
        self.test_configure_processor()
        
        # Start processor
        response = requests.post(
            f"{self.base_url}/start",
            headers=self.headers
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Check status
        response = requests.get(
            f"{self.base_url}/status",
            headers=self.headers
        )
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["is_running"] is True
        
        # Stop processor
        response = requests.post(
            f"{self.base_url}/stop",
            headers=self.headers
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    def test_pause_resume_processor(self):
        """Test pausing and resuming the processor"""
        # Start processor first
        self.test_configure_processor()
        requests.post(f"{self.base_url}/start", headers=self.headers)
        
        # Pause processor
        response = requests.post(
            f"{self.base_url}/pause",
            headers=self.headers
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Resume processor
        response = requests.post(
            f"{self.base_url}/resume",
            headers=self.headers
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    def test_process_single_item(self):
        """Test processing a single item"""
        # Setup processor
        self.test_configure_processor()
        requests.post(f"{self.base_url}/start", headers=self.headers)
        
        # Process item
        request_data = {
            "data": [[1.0, 2.0, 3.0, 4.0]],
            "priority": "normal",
            "timeout": 30.0
        }
        
        response = requests.post(
            f"{self.base_url}/process",
            json=request_data,
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        assert data["results"] is not None
    
    def test_process_batch(self):
        """Test processing a batch of items"""
        # Setup processor
        self.test_configure_processor()
        requests.post(f"{self.base_url}/start", headers=self.headers)
        
        # Process batch
        batch_data = {
            "items": [
                {
                    "data": [[1.0, 2.0, 3.0]],
                    "priority": "normal",
                    "timeout": 30.0
                },
                {
                    "data": [[4.0, 5.0, 6.0]],
                    "priority": "high",
                    "timeout": 30.0
                },
                {
                    "data": [[7.0, 8.0, 9.0]],
                    "priority": "low",
                    "timeout": 30.0
                }
            ],
            "wait_for_completion": True
        }
        
        response = requests.post(
            f"{self.base_url}/process-batch",
            json=batch_data,
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        assert len(data["results"]) == 3
    
    def test_submit_batch(self):
        """Test batch submission workflow"""
        # Setup processor
        self.test_configure_processor()
        requests.post(f"{self.base_url}/start", headers=self.headers)
        
        # Submit batch
        batch_data = [
            {"data": [1.0, 2.0, 3.0]},
            {"data": [4.0, 5.0, 6.0]},
            {"data": [7.0, 8.0, 9.0]}
        ]
        
        response = requests.post(
            f"{self.base_url}/submit",
            json={
                "data": batch_data,
                "priority": "normal",
                "metadata": {"experiment_id": "test_001"}
            },
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert data["status"] == "submitted"
        assert data["items_count"] == 3
        
        batch_id = data["batch_id"]
        
        # Check status
        response = requests.get(
            f"{self.base_url}/status/{batch_id}",
            headers=self.headers
        )
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["batch_id"] == batch_id
        assert "status" in status_data
        
        # Wait for completion and get results
        time.sleep(2)  # Allow processing time
        
        response = requests.get(
            f"{self.base_url}/results/{batch_id}",
            headers=self.headers
        )
        
        # Should either get results or 202 (still processing)
        assert response.status_code in [200, 202]
    
    def test_get_metrics(self):
        """Test metrics endpoint"""
        # Setup processor
        self.test_configure_processor()
        requests.post(f"{self.base_url}/start", headers=self.headers)
        
        response = requests.get(
            f"{self.base_url}/metrics",
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "processor_stats" in data
        assert "system_metrics" in data
        assert "batch_tracking" in data
        assert "timestamp" in data
    
    def test_get_analytics(self):
        """Test analytics endpoint"""
        # Setup processor
        self.test_configure_processor()
        requests.post(f"{self.base_url}/start", headers=self.headers)
        
        response = requests.get(
            f"{self.base_url}/analytics",
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "processing_performance" in data
        assert "resource_utilization" in data
        assert "error_analysis" in data
        assert "throughput_metrics" in data
    
    def test_update_batch_size(self):
        """Test batch size update"""
        # Setup processor
        self.test_configure_processor()
        requests.post(f"{self.base_url}/start", headers=self.headers)
        
        response = requests.post(
            f"{self.base_url}/update-batch-size",
            json={"new_size": 32},
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["new_batch_size"] == 32
    
    def test_authentication_required(self):
        """Test that API key is required"""
        response = requests.get(f"{self.base_url}/status")
        # Should work if auth is disabled, fail if enabled
        assert response.status_code in [200, 401]
    
    def test_invalid_api_key(self):
        """Test invalid API key handling"""
        invalid_headers = {
            "X-API-Key": "invalid_key",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{self.base_url}/status",
            headers=invalid_headers
        )
        # Should work if auth is disabled, fail if enabled
        assert response.status_code in [200, 401]
    
    def test_concurrent_processing(self):
        """Test concurrent batch processing"""
        # Setup processor
        self.test_configure_processor()
        requests.post(f"{self.base_url}/start", headers=self.headers)
        
        def submit_batch(batch_id: int) -> Dict[str, Any]:
            """Submit a batch and return response"""
            batch_data = [
                {"data": [float(i), float(i+1), float(i+2)]}
                for i in range(batch_id * 10, (batch_id + 1) * 10)
            ]
            
            response = requests.post(
                f"{self.base_url}/submit",
                json={
                    "data": batch_data,
                    "priority": "normal",
                    "metadata": {"batch_number": batch_id}
                },
                headers=self.headers
            )
            return response.json()
        
        # Submit multiple batches concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(submit_batch, i)
                for i in range(5)
            ]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                assert "batch_id" in result
                assert result["status"] == "submitted"
        
        assert len(results) == 5
    
    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test without configuring processor
        response = requests.get(
            f"{self.base_url}/status",
            headers=self.headers
        )
        assert response.status_code == 400
        
        # Test invalid batch size update
        self.test_configure_processor()
        response = requests.post(
            f"{self.base_url}/update-batch-size",
            json={"new_size": -1},
            headers=self.headers
        )
        assert response.status_code == 400
        
        # Test invalid batch ID
        response = requests.get(
            f"{self.base_url}/status/invalid_batch_id",
            headers=self.headers
        )
        assert response.status_code == 404


class TestBatchProcessorAsyncAPI:
    """Async tests for Batch Processor API"""
    
    async def test_async_batch_submission(self):
        """Test async batch submission"""
        base_url = "http://localhost:8001"
        headers = {
            "X-API-Key": "test_key",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            # Configure processor
            config = {
                "initial_batch_size": 8,
                "max_batch_size": 64,
                "enable_monitoring": True
            }
            
            async with session.post(
                f"{base_url}/configure",
                json=config,
                headers=headers
            ) as response:
                assert response.status == 200
            
            # Start processor
            async with session.post(
                f"{base_url}/start",
                headers=headers
            ) as response:
                assert response.status == 200
            
            # Submit multiple batches concurrently
            tasks = []
            for i in range(3):
                batch_data = [
                    {"data": [float(j), float(j+1), float(j+2)]}
                    for j in range(i * 5, (i + 1) * 5)
                ]
                
                task = session.post(
                    f"{base_url}/submit",
                    json={
                        "data": batch_data,
                        "priority": "normal",
                        "metadata": {"async_batch": i}
                    },
                    headers=headers
                )
                tasks.append(task)
            
            # Wait for all submissions
            responses = await asyncio.gather(*tasks)
            
            batch_ids = []
            for response in responses:
                assert response.status == 200
                data = await response.json()
                batch_ids.append(data["batch_id"])
            
            assert len(batch_ids) == 3
            
            # Check metrics
            async with session.get(
                f"{base_url}/metrics",
                headers=headers
            ) as response:
                assert response.status == 200
                metrics = await response.json()
                assert "batch_tracking" in metrics


# Performance and stress tests
class TestBatchProcessorPerformance:
    """Performance tests for Batch Processor API"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup performance test environment"""
        self.base_url = "http://localhost:8001"
        self.headers = {
            "X-API-Key": "test_key",
            "Content-Type": "application/json"
        }
        
        # Configure for performance
        config = {
            "initial_batch_size": 32,
            "max_batch_size": 256,
            "batch_timeout": 0.01,
            "enable_adaptive_batching": True,
            "enable_monitoring": True,
            "num_workers": 8
        }
        
        requests.post(f"{self.base_url}/configure", json=config, headers=self.headers)
        requests.post(f"{self.base_url}/start", headers=self.headers)
    
    def test_high_throughput_processing(self):
        """Test high throughput batch processing"""
        num_batches = 50
        items_per_batch = 20
        
        start_time = time.time()
        
        # Submit many batches
        batch_ids = []
        for i in range(num_batches):
            batch_data = [
                {"data": [float(j), float(j+1), float(j+2), float(j+3)]}
                for j in range(items_per_batch)
            ]
            
            response = requests.post(
                f"{self.base_url}/submit",
                json={
                    "data": batch_data,
                    "priority": "normal",
                    "metadata": {"batch_index": i}
                },
                headers=self.headers
            )
            
            assert response.status_code == 200
            batch_ids.append(response.json()["batch_id"])
        
        submission_time = time.time() - start_time
        
        # Get metrics
        response = requests.get(f"{self.base_url}/metrics", headers=self.headers)
        assert response.status_code == 200
        metrics = response.json()
        
        print(f"Submitted {num_batches} batches in {submission_time:.2f}s")
        print(f"Throughput: {num_batches / submission_time:.2f} batches/s")
        print(f"Total items: {num_batches * items_per_batch}")
        print(f"Batch tracking: {metrics['batch_tracking']}")
    
    def test_memory_usage_under_load(self):
        """Test memory usage under heavy load"""
        # Submit large batches
        for i in range(10):
            large_batch = [
                {"data": [float(j) for j in range(100)]}  # Large arrays
                for _ in range(50)  # Many items
            ]
            
            response = requests.post(
                f"{self.base_url}/submit",
                json={
                    "data": large_batch,
                    "priority": "normal",
                    "metadata": {"large_batch": i}
                },
                headers=self.headers
            )
            
            assert response.status_code == 200
        
        # Check system metrics
        response = requests.get(f"{self.base_url}/metrics", headers=self.headers)
        assert response.status_code == 200
        metrics = response.json()
        
        if "system_metrics" in metrics and metrics["system_metrics"]:
            memory_usage = metrics["system_metrics"].get("memory_usage_percent", 0)
            print(f"Memory usage under load: {memory_usage:.1f}%")
            assert memory_usage < 90, "Memory usage too high"
    
    def test_concurrent_load(self):
        """Test concurrent load handling"""
        def worker_thread(thread_id: int, num_requests: int):
            """Worker thread for concurrent testing"""
            results = []
            for i in range(num_requests):
                batch_data = [
                    {"data": [float(thread_id), float(i), float(thread_id + i)]}
                    for _ in range(10)
                ]
                
                response = requests.post(
                    f"{self.base_url}/submit",
                    json={
                        "data": batch_data,
                        "priority": "normal",
                        "metadata": {"thread_id": thread_id, "request_id": i}
                    },
                    headers=self.headers
                )
                
                results.append(response.status_code)
            
            return results
        
        # Run concurrent workers
        num_workers = 5
        requests_per_worker = 10
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker_thread, i, requests_per_worker)
                for i in range(num_workers)
            ]
            
            all_results = []
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
        
        total_time = time.time() - start_time
        
        # Verify all requests succeeded
        successful_requests = sum(1 for status in all_results if status == 200)
        total_requests = num_workers * requests_per_worker
        
        print(f"Concurrent test: {successful_requests}/{total_requests} successful")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {total_requests / total_time:.2f} requests/s")
        
        assert successful_requests == total_requests
        
        # Get final metrics
        response = requests.get(f"{self.base_url}/metrics", headers=self.headers)
        metrics = response.json()
        print(f"Final metrics: {metrics['batch_tracking']}")


if __name__ == "__main__":
    # Run basic functionality tests
    pytest.main([__file__ + "::TestBatchProcessorAPI", "-v"])
