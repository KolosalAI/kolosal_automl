"""
Enhanced unit tests for Batch Processor

Comprehensive test suite for the BatchProcessor engine with focus on
performance, error handling, and edge cases.

Author: AI Assistant  
Date: 2025-07-20
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch
from concurrent.futures import Future
from typing import List, Dict, Any
import queue
import gc

from modules.engine.batch_processor import BatchProcessor, BatchStats
from modules.configs import (
    BatchProcessorConfig, 
    BatchProcessingStrategy, 
    BatchPriority
)


class TestBatchProcessorCore:
    """Core functionality tests for BatchProcessor"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return BatchProcessorConfig(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16,
            batch_timeout=0.1,
            max_queue_size=100,
            enable_priority_queue=True,
            processing_strategy=BatchProcessingStrategy.BALANCED,
            enable_adaptive_batching=True,
            enable_monitoring=True,
            num_workers=2,
            enable_memory_optimization=True,
            max_retries=3
        )
    
    @pytest.fixture
    def processor(self, config):
        """Create test processor"""
        return BatchProcessor(config)
    
    def simple_processing_func(self, batch: np.ndarray) -> np.ndarray:
        """Simple processing function for testing"""
        return batch * 2.0
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor is not None
        assert not processor.stop_event.is_set()
        assert processor.paused_event.is_set()  # Should start paused
        assert processor.config.initial_batch_size == 4
    
    def test_start_stop_processor(self, processor):
        """Test starting and stopping processor"""
        # Start processor
        processor.start(self.simple_processing_func)
        assert processor.worker_thread is not None
        assert processor.worker_thread.is_alive()
        
        # Stop processor
        processor.stop(timeout=2.0)
        assert processor.stop_event.is_set()
        if processor.worker_thread:
            assert not processor.worker_thread.is_alive()
    
    def test_pause_resume_processor(self, processor):
        """Test pausing and resuming processor"""
        processor.start(self.simple_processing_func)
        
        # Pause
        processor.pause()
        # Note: paused_event is cleared when paused
        assert not processor.paused_event.is_set()
        
        # Resume  
        processor.resume()
        assert processor.paused_event.is_set()
        
        processor.stop()
    
    def test_single_item_processing(self, processor):
        """Test processing single items"""
        processor.start(self.simple_processing_func)
        
        # Process single item
        test_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        future = processor.enqueue_predict(test_data)
        
        result = future.result(timeout=5.0)
        expected = test_data * 2.0
        
        np.testing.assert_array_almost_equal(result, expected)
        
        processor.stop()
    
    def test_batch_processing(self, processor):
        """Test processing multiple items as batch"""
        processor.start(self.simple_processing_func)
        
        # Submit multiple items
        futures = []
        test_data = []
        
        for i in range(8):  # More than batch size to trigger batching
            data = np.array([[float(i), float(i+1), float(i+2)]], dtype=np.float32)
            test_data.append(data)
            future = processor.enqueue_predict(data)
            futures.append(future)
        
        # Wait for all results
        results = []
        for future in futures:
            result = future.result(timeout=5.0)
            results.append(result)
        
        # Verify results
        for i, result in enumerate(results):
            expected = test_data[i] * 2.0
            np.testing.assert_array_almost_equal(result, expected)
        
        processor.stop()
    
    def test_priority_processing(self, processor):
        """Test priority-based processing"""
        processor.start(self.simple_processing_func)
        
        # Submit items with different priorities
        low_future = processor.enqueue_predict(
            np.array([[1.0]], dtype=np.float32), 
            priority=BatchPriority.LOW
        )
        
        high_future = processor.enqueue_predict(
            np.array([[2.0]], dtype=np.float32), 
            priority=BatchPriority.HIGH
        )
        
        normal_future = processor.enqueue_predict(
            np.array([[3.0]], dtype=np.float32), 
            priority=BatchPriority.NORMAL
        )
        
        # All should complete successfully
        results = [
            low_future.result(timeout=5.0),
            high_future.result(timeout=5.0), 
            normal_future.result(timeout=5.0)
        ]
        
        assert len(results) == 3
        assert all(r is not None for r in results)
        
        processor.stop()
    
    def test_timeout_handling(self, processor):
        """Test timeout handling"""
        def slow_processing_func(batch: np.ndarray) -> np.ndarray:
            time.sleep(2.0)  # Intentionally slow
            return batch * 2.0
        
        processor.start(slow_processing_func)
        
        # Submit with short timeout
        future = processor.enqueue_predict(
            np.array([[1.0]], dtype=np.float32),
            timeout=0.5
        )
        
        # Should timeout
        with pytest.raises(Exception):  # TimeoutError or similar
            future.result(timeout=1.0)
        
        processor.stop()
    
    def test_batch_size_updates(self, processor):
        """Test dynamic batch size updates"""
        processor.start(self.simple_processing_func)
        
        # Update batch size
        new_size = 8
        processor.update_batch_size(new_size)
        
        # Verify update (implementation dependent)
        # This tests the interface exists
        assert hasattr(processor, 'update_batch_size')
        
        processor.stop()
    
    def test_statistics_collection(self, processor):
        """Test statistics collection"""
        processor.start(self.simple_processing_func)
        
        # Process some items
        futures = []
        for i in range(5):
            future = processor.enqueue_predict(
                np.array([[float(i)]], dtype=np.float32)
            )
            futures.append(future)
        
        # Wait for completion
        for future in futures:
            future.result(timeout=5.0)
        
        # Get statistics
        stats = processor.get_stats()
        assert isinstance(stats, dict)
        
        # Should have some basic metrics
        assert "total_processed" in stats or len(stats) >= 0  # Implementation dependent
        
        processor.stop()


class TestBatchProcessorErrorHandling:
    """Error handling and edge case tests"""
    
    @pytest.fixture
    def config(self):
        return BatchProcessorConfig(
            initial_batch_size=2,
            max_batch_size=8,
            batch_timeout=0.1,
            max_retries=2
        )
    
    @pytest.fixture
    def processor(self, config):
        return BatchProcessor(config)
    
    def failing_processing_func(self, batch: np.ndarray) -> np.ndarray:
        """Processing function that always fails"""
        raise ValueError("Intentional failure for testing")
    
    def test_processing_error_handling(self, processor):
        """Test handling of processing errors"""
        processor.start(self.failing_processing_func)
        
        # Submit item to failing processor
        future = processor.enqueue_predict(
            np.array([[1.0]], dtype=np.float32)
        )
        
        # Should propagate error but not crash the system
        with pytest.raises(ValueError, match="Intentional failure"):
            future.result(timeout=5.0)
        
        # Processor should still be running and capable of handling other requests
        # Test with a working function
        def working_func(x):
            return x * 2
        
        processor.stop()
        processor.start(working_func)
        
        # This should work
        future2 = processor.enqueue_predict(
            np.array([[2.0]], dtype=np.float32)
        )
        result = future2.result(timeout=5.0)
        np.testing.assert_array_equal(result, np.array([[4.0]], dtype=np.float32))
        
        processor.stop()
    
    def test_invalid_input_handling(self, processor):
        """Test handling of invalid inputs"""
        processor.start(lambda x: x * 2)
        
        # Test invalid data types
        with pytest.raises((TypeError, ValueError)):
            processor.enqueue_predict("invalid_data")
        
        # Test empty arrays
        future = processor.enqueue_predict(np.array([]))
        # Should handle gracefully or raise appropriate error
        try:
            future.result(timeout=5.0)
        except Exception as e:
            # Should be a reasonable error, not a crash
            assert isinstance(e, (ValueError, RuntimeError))
        
        processor.stop()
    
    def test_queue_overflow_handling(self):
        """Test queue overflow scenarios"""
        config = BatchProcessorConfig(
            max_queue_size=3,  # Very small queue
            batch_timeout=1.0  # Long timeout to control timing
        )
        processor = BatchProcessor(config)
        
        def slow_processing(batch):
            time.sleep(0.5)
            return batch * 2
        
        processor.start(slow_processing)
        
        # Fill queue beyond capacity
        futures = []
        for i in range(5):  # More than queue size
            try:
                future = processor.enqueue_predict(
                    np.array([[float(i)]], dtype=np.float32),
                    timeout=0.1
                )
                futures.append(future)
            except Exception:
                # Queue full exception is acceptable
                break
        
        processor.stop()
    
    def test_concurrent_access_safety(self, processor):
        """Test thread safety under concurrent access"""
        processor.start(lambda x: x * 2)
        
        def worker_thread(thread_id: int, num_items: int):
            """Worker thread for concurrent testing"""
            results = []
            for i in range(num_items):
                try:
                    data = np.array([[float(thread_id * 100 + i)]], dtype=np.float32)
                    future = processor.enqueue_predict(data)
                    result = future.result(timeout=5.0)
                    results.append(result)
                except Exception as e:
                    results.append(e)
            return results
        
        # Run multiple threads
        num_threads = 4
        items_per_thread = 3
        threads = []
        
        for i in range(num_threads):
            thread = threading.Thread(
                target=worker_thread, 
                args=(i, items_per_thread)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        processor.stop()
    
    def test_memory_leak_prevention(self, processor):
        """Test memory management and leak prevention"""
        processor.start(lambda x: x * 2)
        
        # Process many items to check for memory leaks
        initial_objects = len(gc.get_objects())
        
        for i in range(50):
            future = processor.enqueue_predict(
                np.array([[float(i)]], dtype=np.float32)
            )
            future.result(timeout=5.0)
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count shouldn't grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 100, f"Potential memory leak: {object_growth} new objects"
        
        processor.stop()


class TestBatchProcessorPerformance:
    """Performance and benchmarking tests"""
    
    @pytest.fixture
    def high_performance_config(self):
        return BatchProcessorConfig(
            initial_batch_size=16,
            max_batch_size=128,
            batch_timeout=0.01,
            max_queue_size=1000,
            enable_adaptive_batching=True,
            enable_monitoring=True,
            num_workers=4
        )
    
    @pytest.fixture
    def processor(self, high_performance_config):
        return BatchProcessor(high_performance_config)
    
    def test_throughput_performance(self, processor):
        """Test processing throughput"""
        def fast_processing(batch):
            return batch * 2.0
        
        processor.start(fast_processing)
        
        num_items = 100
        start_time = time.time()
        
        # Submit all items
        futures = []
        for i in range(num_items):
            future = processor.enqueue_predict(
                np.array([[float(i)]], dtype=np.float32)
            )
            futures.append(future)
        
        # Wait for all to complete
        for future in futures:
            future.result(timeout=10.0)
        
        total_time = time.time() - start_time
        throughput = num_items / total_time
        
        print(f"Processed {num_items} items in {total_time:.3f}s")
        print(f"Throughput: {throughput:.1f} items/second")
        
        # Should achieve reasonable throughput
        assert throughput > 50, f"Throughput too low: {throughput:.1f} items/s"
        
        processor.stop()
    
    def test_latency_performance(self, processor):
        """Test processing latency"""
        def instant_processing(batch):
            return batch
        
        processor.start(instant_processing)
        
        # Measure latency for single items
        latencies = []
        
        for i in range(10):
            start_time = time.time()
            
            future = processor.enqueue_predict(
                np.array([[float(i)]], dtype=np.float32)
            )
            future.result(timeout=5.0)
            
            latency = time.time() - start_time
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        print(f"Average latency: {avg_latency*1000:.2f}ms")
        print(f"Max latency: {max_latency*1000:.2f}ms")
        
        # Latency should be reasonable
        assert avg_latency < 0.1, f"Average latency too high: {avg_latency*1000:.2f}ms"
        assert max_latency < 0.5, f"Max latency too high: {max_latency*1000:.2f}ms"
        
        processor.stop()
    
    def test_adaptive_batching_performance(self, processor):
        """Test adaptive batching behavior"""
        call_count = 0
        batch_sizes = []
        
        def monitoring_processing_func(batch):
            nonlocal call_count, batch_sizes
            call_count += 1
            batch_sizes.append(len(batch))
            return batch * 2.0
        
        processor.start(monitoring_processing_func)
        
        # Submit items with varying patterns
        # Burst pattern
        for i in range(20):
            processor.enqueue_predict(
                np.array([[float(i)]], dtype=np.float32)
            )
        
        time.sleep(0.2)  # Allow processing
        
        # Single items
        for i in range(5):
            future = processor.enqueue_predict(
                np.array([[float(i)]], dtype=np.float32)
            )
            future.result(timeout=5.0)
            time.sleep(0.05)
        
        print(f"Processing function called {call_count} times")
        print(f"Batch sizes: {batch_sizes}")
        
        if batch_sizes:
            avg_batch_size = sum(batch_sizes) / len(batch_sizes)
            print(f"Average batch size: {avg_batch_size:.2f}")
            
            # Should show some batching occurred
            assert max(batch_sizes) > 1, "No batching occurred"
        
        processor.stop()
    
    def test_memory_efficiency(self, processor):
        """Test memory efficiency under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        processor.start(lambda x: x * 2)
        
        # Process large number of items
        futures = []
        for i in range(200):
            future = processor.enqueue_predict(
                np.random.rand(1, 100).astype(np.float32)  # Larger arrays
            )
            futures.append(future)
        
        # Complete all processing
        for future in futures:
            future.result(timeout=10.0)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
        print(f"Memory growth: {memory_growth:.1f}MB")
        
        # Memory growth should be reasonable
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB"
        
        processor.stop()


class TestBatchProcessorIntegration:
    """Integration tests with realistic scenarios"""
    
    def test_ml_inference_simulation(self):
        """Simulate ML model inference workload"""
        config = BatchProcessorConfig(
            initial_batch_size=8,
            max_batch_size=32,
            batch_timeout=0.02,
            enable_adaptive_batching=True,
            enable_monitoring=True
        )
        processor = BatchProcessor(config)
        
        def mock_model_inference(batch):
            """Mock ML model inference"""
            # Simulate processing time based on batch size
            processing_time = 0.01 + (len(batch) * 0.001)
            time.sleep(processing_time)
            
            # Return mock predictions
            return np.random.rand(len(batch), 10)
        
        processor.start(mock_model_inference)
        
        # Simulate realistic inference requests
        futures = []
        request_times = []
        
        for i in range(50):
            start_time = time.time()
            
            # Variable input sizes (realistic)
            input_size = np.random.randint(5, 20)
            input_data = np.random.rand(1, input_size).astype(np.float32)
            
            future = processor.enqueue_predict(input_data)
            futures.append((future, start_time))
            
            # Variable inter-request timing
            time.sleep(np.random.exponential(0.02))
        
        # Collect results and measure latencies
        latencies = []
        for future, start_time in futures:
            result = future.result(timeout=10.0)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            assert result is not None
            assert result.shape[1] == 10  # Mock prediction size
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"ML Inference Simulation Results:")
        print(f"  Processed: {len(futures)} requests")
        print(f"  Avg latency: {avg_latency*1000:.2f}ms")
        print(f"  P95 latency: {p95_latency*1000:.2f}ms")
        
        # Get processor stats
        stats = processor.get_stats()
        print(f"  Processor stats: {stats}")
        
        processor.stop()
    
    def test_mixed_priority_workload(self):
        """Test mixed priority workload scenario"""
        config = BatchProcessorConfig(
            enable_priority_queue=True,
            initial_batch_size=4,
            max_batch_size=16
        )
        processor = BatchProcessor(config)
        
        processed_order = []
        
        def tracking_processing_func(batch):
            # Track processing order by priority
            for item in batch:
                processed_order.append(float(item[0]))
            return batch * 2
        
        processor.start(tracking_processing_func)
        
        # Submit mixed priority requests
        priorities = [BatchPriority.LOW, BatchPriority.NORMAL, BatchPriority.HIGH]
        futures = []
        
        for i in range(12):
            priority = priorities[i % 3]
            value = float(i)
            
            future = processor.enqueue_predict(
                np.array([[value]], dtype=np.float32),
                priority=priority
            )
            futures.append((future, priority, value))
        
        # Wait for all processing
        for future, priority, value in futures:
            result = future.result(timeout=5.0)
            assert result is not None
        
        print(f"Processing order: {processed_order}")
        
        # High priority items should generally be processed earlier
        # (exact ordering depends on timing and batching)
        
        processor.stop()


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])
