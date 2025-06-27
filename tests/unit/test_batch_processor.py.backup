import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future
import tempfile
import os
import sys
from queue import Queue, Empty
from typing import List, Dict, Any

# Add the modules path to import the batch processor
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules", "engine"))

try:
    from modules.engine.batch_processor import (
        BatchProcessor, BatchProcessorConfig, BatchProcessingStrategy, BatchPriority
    )
    # Try to import optional components that might not exist
    try:
        from modules.engine.batch_processor import (
            HybridConfig, ProcessingMode, CacheStrategy,
            ResultCache, StreamProcessor, PipelineStage, LoadBalancer,
            BatchStats
        )
    except ImportError:
        # Create minimal mock classes for missing components
        class ProcessingMode:
            HYBRID = "hybrid"
            BATCH = "batch"
            STREAM = "stream"
        
        class HybridConfig:
            def __init__(self):
                self.processing_mode = ProcessingMode.HYBRID
        
        class CacheStrategy:
            LRU = "lru"
            TTL = "ttl"
        
        class ResultCache:
            def __init__(self, **kwargs):
                pass
        
        class StreamProcessor:
            def __init__(self, **kwargs):
                pass
        
        class PipelineStage:
            def __init__(self, **kwargs):
                pass
        
        class LoadBalancer:
            def __init__(self, **kwargs):
                pass
        
        class BatchStats:
            def __init__(self, **kwargs):
                pass
                
except ImportError as e:
    pytest.skip(f"Batch processor module not available: {e}", allow_module_level=True)


@pytest.mark.unit
class TestHybridConfig:
    """Test HybridConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = HybridConfig()
        
        assert config.processing_mode == ProcessingMode.HYBRID
        self.assertEqual(config.cache_strategy, CacheStrategy.LRU)
        self.assertEqual(config.cache_size, 1000)
        self.assertEqual(config.cache_ttl, 3600)
        self.assertTrue(config.enable_streaming)
        self.assertTrue(config.enable_compression)
        self.assertTrue(config.enable_prefetching)
    
    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = HybridConfig(
            processing_mode=ProcessingMode.ASYNC,
            cache_strategy=CacheStrategy.TTL,
            cache_size=500,
            enable_gpu_processing=True
        )
        
        self.assertEqual(config.processing_mode, ProcessingMode.ASYNC)
        self.assertEqual(config.cache_strategy, CacheStrategy.TTL)
        self.assertEqual(config.cache_size, 500)
        self.assertTrue(config.enable_gpu_processing)


class TestResultCache(unittest.TestCase):
    """Test ResultCache functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HybridConfig(cache_strategy=CacheStrategy.LRU, cache_size=3)
        self.cache = ResultCache(self.config)
    
    def test_lru_cache_basic_operations(self):
        """Test basic LRU cache put and get operations."""
        # Test putting and getting items
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key2"), "value2")
        self.assertIsNone(self.cache.get("key3"))
    
    def test_lru_cache_eviction(self):
        """Test LRU cache eviction when size limit is exceeded."""
        # Fill cache to capacity
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        
        # Add one more item, should evict oldest
        self.cache.put("key4", "value4")
        
        # key1 should be evicted
        self.assertIsNone(self.cache.get("key1"))
        self.assertEqual(self.cache.get("key2"), "value2")
        self.assertEqual(self.cache.get("key3"), "value3")
        self.assertEqual(self.cache.get("key4"), "value4")
    
    def test_lru_access_order_update(self):
        """Test that accessing items updates LRU order."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        
        # Access key1 to make it most recently used
        self.cache.get("key1")
        
        # Add key4, should evict key2 (now oldest)
        self.cache.put("key4", "value4")
        
        self.assertEqual(self.cache.get("key1"), "value1")  # Should still be there
        self.assertIsNone(self.cache.get("key2"))  # Should be evicted
        self.assertEqual(self.cache.get("key3"), "value3")
        self.assertEqual(self.cache.get("key4"), "value4")
    
    def test_ttl_cache(self):
        """Test TTL cache with expiration."""
        config = HybridConfig(cache_strategy=CacheStrategy.TTL, cache_ttl=1)
        cache = ResultCache(config)
        
        cache.put("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")
        
        # Wait for expiration
        time.sleep(1.1)
        self.assertIsNone(cache.get("key1"))
    
    def test_hash_key_generation(self):
        """Test cache key generation for different data types."""
        # Test with numpy array
        array = np.array([1, 2, 3])
        key1 = self.cache._hash_key(array)
        key2 = self.cache._hash_key(array)
        self.assertEqual(key1, key2)  # Same array should generate same key
        
        # Test with different array
        different_array = np.array([1, 2, 4])
        key3 = self.cache._hash_key(different_array)
        self.assertNotEqual(key1, key3)
        
        # Test with regular object
        key4 = self.cache._hash_key("test_string")
        self.assertIsInstance(key4, str)
    
    def test_none_cache_strategy(self):
        """Test that NONE strategy doesn't cache anything."""
        config = HybridConfig(cache_strategy=CacheStrategy.NONE)
        cache = ResultCache(config)
        
        cache.put("key1", "value1")
        self.assertIsNone(cache.get("key1"))


class TestStreamProcessor(unittest.TestCase):
    """Test StreamProcessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HybridConfig(stream_chunk_size=2)
        self.processor = StreamProcessor(self.config)
    
    @patch('asyncio.get_event_loop')
    @patch('concurrent.futures.ThreadPoolExecutor')
    async def test_stream_process_chunking(self, mock_executor, mock_loop):
        """Test that stream processing chunks data correctly."""
        # Mock the executor and loop
        mock_loop.return_value = Mock()
        mock_executor.return_value.__enter__ = Mock(return_value=Mock())
        mock_executor.return_value.__exit__ = Mock(return_value=None)
        
        # Create a simple async generator
        async def mock_stream():
            for i in range(5):
                yield i
        
        def mock_process_func(x):
            return x * 2
        
        # This test verifies the chunking logic structure
        # Full async testing would require more complex setup
        self.assertEqual(self.processor.chunk_size, 2)


class TestPipelineStage(unittest.TestCase):
    """Test PipelineStage functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_queue = Queue()
        self.output_queue = Queue()
        
        def simple_process_func(item):
            return item * 2
        
        self.stage = PipelineStage(
            stage_id=0,
            process_func=simple_process_func,
            input_queue=self.input_queue,
            output_queue=self.output_queue
        )
    
    def test_pipeline_stage_initialization(self):
        """Test pipeline stage initialization."""
        self.assertEqual(self.stage.stage_id, 0)
        self.assertIsNotNone(self.stage.process_func)
        self.assertEqual(self.stage.input_queue, self.input_queue)
        self.assertEqual(self.stage.output_queue, self.output_queue)
        self.assertFalse(self.stage.stop_event.is_set())
    
    def test_pipeline_stage_processing(self):
        """Test pipeline stage processing items."""
        # Put item in input queue
        self.input_queue.put(5)
        self.input_queue.put(None)  # Poison pill
        
        # Start stage
        self.stage.start()
        
        # Wait briefly for processing
        time.sleep(0.1)
        
        # Stop stage
        self.stage.stop()
        
        # Check output
        try:
            result = self.output_queue.get(timeout=1.0)
            self.assertEqual(result, 10)  # 5 * 2
        except:
            self.fail("Pipeline stage did not produce expected output")


class TestLoadBalancer(unittest.TestCase):
    """Test LoadBalancer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.balancer = LoadBalancer(num_workers=3)
    
    def test_initialization(self):
        """Test load balancer initialization."""
        self.assertEqual(self.balancer.num_workers, 3)
        self.assertEqual(len(self.balancer.worker_loads), 3)
        self.assertEqual(sum(self.balancer.worker_loads), 0)
    
    def test_least_loaded_worker_selection(self):
        """Test selection of least loaded worker."""
        # Initially all workers have zero load
        worker_id = self.balancer.get_least_loaded_worker()
        self.assertIn(worker_id, [0, 1, 2])
        
        # Add load to worker 0
        self.balancer.update_worker_load(0, 10)
        
        # Now worker 1 or 2 should be selected
        worker_id = self.balancer.get_least_loaded_worker()
        self.assertIn(worker_id, [1, 2])
        
        # Add more load to worker 1
        self.balancer.update_worker_load(1, 5)
        
        # Worker 2 should now be least loaded
        worker_id = self.balancer.get_least_loaded_worker()
        self.assertEqual(worker_id, 2)
    
    def test_load_updates(self):
        """Test worker load updates."""
        self.balancer.update_worker_load(0, 5)
        self.balancer.update_worker_load(1, 3)
        self.balancer.update_worker_load(2, 7)
        
        self.assertEqual(self.balancer.worker_loads[0], 5)
        self.assertEqual(self.balancer.worker_loads[1], 3)
        self.assertEqual(self.balancer.worker_loads[2], 7)
        
        # Test negative updates (reducing load)
        self.balancer.update_worker_load(2, -2)
        self.assertEqual(self.balancer.worker_loads[2], 5)


class TestBatchStats(unittest.TestCase):
    """Test BatchStats functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stats = BatchStats(window_size=10)
    
    def test_stats_initialization(self):
        """Test stats initialization."""
        self.assertEqual(self.stats.window_size, 10)
        self.assertEqual(self.stats.error_counts, 0)
        self.assertEqual(self.stats.batch_counts, 0)
        self.assertEqual(self.stats.total_processed, 0)
    
    def test_update_stats(self):
        """Test updating statistics."""
        self.stats.update(
            processing_time=0.1,
            batch_size=5,
            queue_time=0.05,
            latency=0.15,
            cache_hit=True
        )
        
        self.assertEqual(self.stats.batch_counts, 1)
        self.assertEqual(self.stats.total_processed, 5)
        self.assertEqual(self.stats.total_cache_hits, 1)
        self.assertEqual(len(self.stats.processing_times), 1)
        self.assertEqual(self.stats.processing_times[0], 0.1)
    
    def test_get_stats_calculation(self):
        """Test statistics calculation."""
        # Add some sample data
        for i in range(3):
            self.stats.update(
                processing_time=0.1 + i * 0.05,
                batch_size=5,
                queue_time=0.02,
                cache_hit=(i % 2 == 0)
            )
        
        stats = self.stats.get_stats()
        
        # Check basic stats
        self.assertAlmostEqual(stats['avg_processing_time_ms'], 125.0, places=1)
        self.assertEqual(stats['avg_batch_size'], 5)
        self.assertEqual(stats['batch_count'], 3)
        self.assertEqual(stats['total_processed'], 15)
        
        # Check cache hit rate (2 hits out of 3 total)
        self.assertAlmostEqual(stats['cache_hit_rate'], 2/3, places=2)
    
    def test_error_recording(self):
        """Test error recording."""
        self.stats.record_error(3)
        self.assertEqual(self.stats.error_counts, 3)
        
        self.stats.record_error()  # Default count of 1
        self.assertEqual(self.stats.error_counts, 4)
    
    def test_stats_caching(self):
        """Test that stats are cached for performance."""
        self.stats.update(0.1, 5, 0.02)
        
        # First call should calculate stats
        stats1 = self.stats.get_stats()
        
        # Second call should return cached stats
        stats2 = self.stats.get_stats()
        
        self.assertEqual(stats1, stats2)


class TestBatchProcessor(unittest.TestCase):
    """Test main BatchProcessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create basic config
        self.config = BatchProcessorConfig(
            initial_batch_size=2,
            max_batch_size=10,
            min_batch_size=1,
            batch_timeout=0.1,
            max_queue_size=100,
            enable_monitoring=True,
            enable_health_monitoring=False  # Disable for tests
        )
        
        self.hybrid_config = HybridConfig(
            processing_mode=ProcessingMode.SYNC,
            enable_streaming=False,
            enable_pipeline_parallelism=False,
            enable_gpu_processing=False
        )
        
        self.processor = BatchProcessor(
            config=self.config,
            hybrid_config=self.hybrid_config
        )
        
        # Simple processing function for tests
        def simple_process_func(data):
            if isinstance(data, np.ndarray):
                return data * 2
            return np.array([data * 2])
        
        self.process_func = simple_process_func
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'processor'):
            try:
                self.processor.stop(timeout=1.0)
            except:
                pass
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.config, self.config)
        self.assertEqual(self.processor.hybrid_config, self.hybrid_config)
        self.assertIsNotNone(self.processor.queue)
        self.assertIsNotNone(self.processor.cache)
        self.assertIsNotNone(self.processor.stats)
    
    def test_start_and_stop(self):
        """Test starting and stopping the processor."""
        self.assertFalse(self.processor.stop_event.is_set())
        
        # Start processor
        self.processor.start(self.process_func)
        self.assertTrue(self.processor.worker_thread.is_alive())
        
        # Stop processor
        self.processor.stop(timeout=2.0)
        self.assertTrue(self.processor.stop_event.is_set())
    
    def test_pause_and_resume(self):
        """Test pausing and resuming the processor."""
        self.processor.start(self.process_func)
        
        # Test pause
        self.processor.pause()
        self.assertTrue(self.processor.paused_event.is_set())
        
        # Test resume
        self.processor.resume()
        self.assertFalse(self.processor.paused_event.is_set())
        
        self.processor.stop()
    
    def test_enqueue_basic(self):
        """Test basic enqueue functionality."""
        self.processor.start(self.process_func)
        
        # Test that enqueue doesn't raise exceptions
        try:
            self.processor.enqueue(np.array([1, 2, 3]))
            time.sleep(0.2)  # Allow processing
        except Exception as e:
            self.fail(f"Basic enqueue failed: {e}")
        finally:
            self.processor.stop()
    
    def test_enqueue_predict(self):
        """Test enqueue_predict with future return."""
        self.processor.start(self.process_func)
        
        try:
            # Submit prediction request
            future = self.processor.enqueue_predict(np.array([1, 2, 3]))
            self.assertIsInstance(future, Future)
            
            # Wait for result
            result = future.result(timeout=2.0)
            expected = np.array([2, 4, 6])  # Input * 2
            np.testing.assert_array_equal(result, expected)
            
        except Exception as e:
            self.fail(f"enqueue_predict failed: {e}")
        finally:
            self.processor.stop()
    
    def test_batch_processing_multiple_items(self):
        """Test processing multiple items in a batch."""
        self.processor.start(self.process_func)
        
        try:
            # Submit multiple prediction requests
            futures = []
            inputs = [np.array([i]) for i in range(5)]
            
            for inp in inputs:
                future = self.processor.enqueue_predict(inp)
                futures.append(future)
            
            # Wait for all results
            results = []
            for future in futures:
                result = future.result(timeout=2.0)
                results.append(result)
            
            # Verify results
            for i, result in enumerate(results):
                expected = np.array([i * 2])
                np.testing.assert_array_equal(result, expected)
                
        except Exception as e:
            self.fail(f"Batch processing failed: {e}")
        finally:
            self.processor.stop()
    
    def test_queue_full_handling(self):
        """Test handling of full queue."""
        # Create processor with small queue
        small_config = BatchProcessorConfig(
            max_queue_size=2,
            batch_timeout=10.0  # Long timeout to prevent processing
        )
        small_processor = BatchProcessor(
            config=small_config,
            hybrid_config=self.hybrid_config
        )
        
        try:
            # Fill the queue
            for i in range(2):
                small_processor.enqueue(np.array([i]), timeout=0.1)
            
            # Next enqueue should timeout or raise exception
            with self.assertRaises((Exception, TimeoutError)):
                small_processor.enqueue(np.array([3]), timeout=0.1)
                
        finally:
            small_processor.stop()
    
    def test_cache_functionality(self):
        """Test caching of results."""
        cache_config = HybridConfig(cache_strategy=CacheStrategy.LRU)
        cached_processor = BatchProcessor(
            config=self.config,
            hybrid_config=cache_config
        )
        
        cached_processor.start(self.process_func)
        
        try:
            # Submit same input twice
            input_data = np.array([1, 2, 3])
            
            future1 = cached_processor.enqueue_predict(input_data, use_cache=True)
            result1 = future1.result(timeout=2.0)
            
            # Second request should use cache
            future2 = cached_processor.enqueue_predict(input_data, use_cache=True)
            result2 = future2.result(timeout=2.0)
            
            # Results should be identical
            np.testing.assert_array_equal(result1, result2)
            
            # Check cache stats
            cache_stats = cached_processor.get_cache_stats()
            self.assertGreater(cache_stats.get('total_cache_hits', 0), 0)
            
        finally:
            cached_processor.stop()
    
    def test_priority_queue(self):
        """Test priority queue functionality."""
        priority_config = BatchProcessorConfig(
            enable_priority_queue=True,
            batch_timeout=0.5
        )
        priority_processor = BatchProcessor(
            config=priority_config,
            hybrid_config=self.hybrid_config
        )
        
        priority_processor.start(self.process_func)
        
        try:
            # Submit items with different priorities
            futures = []
            
            # Submit low priority first
            future_low = priority_processor.enqueue_predict(
                np.array([1]), priority=BatchPriority.LOW
            )
            futures.append(future_low)
            
            # Submit high priority
            future_high = priority_processor.enqueue_predict(
                np.array([2]), priority=BatchPriority.HIGH
            )
            futures.append(future_high)
            
            # Wait for results
            for future in futures:
                result = future.result(timeout=2.0)
                self.assertIsNotNone(result)
                
        finally:
            priority_processor.stop()
    
    def test_get_stats(self):
        """Test statistics collection."""
        self.processor.start(self.process_func)
        
        try:
            # Process some items
            future = self.processor.enqueue_predict(np.array([1, 2, 3]))
            future.result(timeout=2.0)
            
            # Get stats
            stats = self.processor.get_stats()
            
            # Verify stats structure
            self.assertIsInstance(stats, dict)
            self.assertIn('total_processed', stats)
            self.assertIn('batch_count', stats)
            self.assertIn('processing_mode', stats)
            self.assertIn('cache_strategy', stats)
            
        finally:
            self.processor.stop()
    
    def test_update_batch_size(self):
        """Test dynamic batch size updates."""
        initial_size = self.processor._current_batch_size
        new_size = initial_size + 5
        
        self.processor.update_batch_size(new_size)
        self.assertEqual(self.processor._current_batch_size, new_size)
        
        # Test bounds checking
        self.processor.update_batch_size(1000)  # Above max
        self.assertEqual(self.processor._current_batch_size, self.config.max_batch_size)
        
        self.processor.update_batch_size(0)  # Below min
        self.assertEqual(self.processor._current_batch_size, self.config.min_batch_size)
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        cache_config = HybridConfig(cache_strategy=CacheStrategy.LRU)
        cached_processor = BatchProcessor(
            config=self.config,
            hybrid_config=cache_config
        )
        
        # Add something to cache
        cached_processor.cache.put("test_key", "test_value")
        self.assertEqual(cached_processor.cache.get("test_key"), "test_value")
        
        # Clear cache
        cached_processor.clear_cache()
        self.assertIsNone(cached_processor.cache.get("test_key"))
    
    def test_processing_mode_changes(self):
        """Test changing processing modes."""
        initial_mode = self.processor.hybrid_config.processing_mode
        
        # Change to async mode
        self.processor.set_processing_mode(ProcessingMode.ASYNC)
        self.assertEqual(self.processor.hybrid_config.processing_mode, ProcessingMode.ASYNC)
        
        # Change back
        self.processor.set_processing_mode(initial_mode)
        self.assertEqual(self.processor.hybrid_config.processing_mode, initial_mode)
    
    def test_memory_management(self):
        """Test memory management features."""
        # Test memory map creation for large arrays
        large_array = np.random.rand(1000, 100)
        
        # This tests the internal memory mapping functionality
        if self.processor.hybrid_config.memory_mapping:
            compressed = self.processor._maybe_compress(large_array)
            if isinstance(compressed, str):
                # Memory mapping was used
                decompressed = self.processor._decompress_item(compressed)
                np.testing.assert_array_equal(large_array, decompressed)
    
    def test_error_handling(self):
        """Test error handling in processing."""
        def error_func(data):
            raise ValueError("Test error")
        
        self.processor.start(error_func)
        
        try:
            future = self.processor.enqueue_predict(np.array([1, 2, 3]))
            
            # Should get exception
            with self.assertRaises(ValueError):
                future.result(timeout=2.0)
                
        finally:
            self.processor.stop()
    
    def test_shutdown_during_processing(self):
        """Test graceful shutdown during active processing."""
        def slow_func(data):
            time.sleep(0.5)
            return data * 2
        
        self.processor.start(slow_func)
        
        # Submit work
        future = self.processor.enqueue_predict(np.array([1, 2, 3]))
        
        # Shutdown immediately
        self.processor.stop(timeout=1.0)
        
        # Future should be cancelled or completed
        try:
            result = future.result(timeout=0.1)
            # If we get here, processing completed
        except:
            # Future was cancelled due to shutdown
            pass
    
    @patch('modules.engine.batch_processor.psutil')
    def test_health_monitoring_disabled(self, mock_psutil):
        """Test that health monitoring can be disabled."""
        # Processor was created with health monitoring disabled
        self.assertIsNone(self.processor._health_monitor_thread)


class TestProcessorIntegration(unittest.TestCase):
    """Integration tests for the complete batch processor system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = BatchProcessorConfig(
            initial_batch_size=3,
            max_batch_size=10,
            batch_timeout=0.2,
            enable_monitoring=True,
            enable_health_monitoring=False
        )
        
        self.hybrid_config = HybridConfig(
            cache_strategy=CacheStrategy.LRU,
            cache_size=100
        )
    
    def test_end_to_end_processing_pipeline(self):
        """Test complete processing pipeline with multiple features."""
        processor = BatchProcessor(
            config=self.config,
            hybrid_config=self.hybrid_config
        )
        
        def complex_process_func(data):
            """More complex processing function."""
            if isinstance(data, np.ndarray):
                return np.sum(data, axis=1, keepdims=True) * 2
            return np.array([[data * 2]])
        
        processor.start(complex_process_func)
        
        try:
            # Submit various types of requests
            futures = []
            test_data = [
                np.array([[1, 2], [3, 4]]),
                np.array([[5, 6]]),
                np.array([[7, 8], [9, 10], [11, 12]])
            ]
            
            for data in test_data:
                future = processor.enqueue_predict(data)
                futures.append((future, data))
            
            # Wait for all results and verify
            for future, original_data in futures:
                result = future.result(timeout=3.0)
                expected = np.sum(original_data, axis=1, keepdims=True) * 2
                np.testing.assert_array_equal(result, expected)
            
            # Check final statistics
            stats = processor.get_stats()
            self.assertGreater(stats['total_processed'], 0)
            self.assertGreater(stats['batch_count'], 0)
            
            performance = processor.get_performance_summary()
            self.assertIn('overall_stats', performance)
            self.assertIn('efficiency_metrics', performance)
            
        finally:
            processor.stop()
    
    def test_stress_test_with_concurrent_requests(self):
        """Stress test with many concurrent requests."""
        processor = BatchProcessor(
            config=self.config,
            hybrid_config=self.hybrid_config
        )
        
        def simple_func(data):
            return data + 1
        
        processor.start(simple_func)
        
        try:
            # Submit many concurrent requests
            num_requests = 50
            futures = []
            
            for i in range(num_requests):
                data = np.array([i])
                future = processor.enqueue_predict(data)
                futures.append((future, i))
            
            # Wait for all results
            completed = 0
            for future, expected_input in futures:
                try:
                    result = future.result(timeout=5.0)
                    expected = np.array([expected_input + 1])
                    np.testing.assert_array_equal(result, expected)
                    completed += 1
                except Exception as e:
                    self.fail(f"Request {expected_input} failed: {e}")
            
            self.assertEqual(completed, num_requests)
            
        finally:
            processor.stop()


if __name__ == "__main__":
    unittest.main()