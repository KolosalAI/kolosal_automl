import unittest
import numpy as np
import time
from concurrent.futures import TimeoutError

# Import the module under test and the configuration classes.
# Adjust the import paths as needed.
from modules.engine.batch_processor import BatchProcessor, BatchStats
from modules.configs import (
    BatchProcessorConfig,
    BatchProcessingStrategy,
    BatchPriority,
    PrioritizedItem
)

# Create a minimal configuration for testing.
def get_test_config(priority_queue: bool = False) -> BatchProcessorConfig:
    # Assume BatchProcessorConfig is a simple container (e.g., a dataclass)
    # with attributes as used by the processor.
    config = BatchProcessorConfig()
    config.enable_priority_queue = priority_queue
    config.max_queue_size = 10
    config.initial_batch_size = 3
    config.max_batch_memory_mb = 10  # 10 MB
    config.batch_timeout = 0.5  # seconds
    config.debug_mode = True
    config.max_workers = 2
    config.monitoring_window = 5
    config.item_timeout = 1.0  # seconds
    config.retry_delay = 0.1  # seconds
    config.max_retries = 2
    config.reduce_batch_on_failure = True
    config.min_batch_interval = 0.05  # seconds
    config.gc_batch_threshold = 5
    config.enable_adaptive_batching = True
    config.processing_strategy = BatchProcessingStrategy.ADAPTIVE
    config.enable_monitoring = True
    config.enable_health_monitoring = False  # Disable for tests to avoid extra threads
    config.health_check_interval = 0.2  # seconds
    config.memory_warning_threshold = 80.0  # percent
    config.memory_critical_threshold = 90.0  # percent
    config.queue_warning_threshold = 5
    config.queue_critical_threshold = 8
    config.min_batch_size = 1
    config.max_batch_size = 10
    config.enable_memory_optimization = True
    return config

# A simple process function for numpy arrays: doubles the array.
def process_numpy_func(arr: np.ndarray) -> np.ndarray:
    return arr * 2

# A generic process function that works for non-numpy items.
def process_generic_func(item):
    # For testing purposes, just return a string representation.
    return f"processed-{item}"

class TestBatchStats(unittest.TestCase):
    def setUp(self):
        self.stats = BatchStats(window_size=3)

    def test_update_and_get_stats(self):
        # Update stats a few times.
        self.stats.update(processing_time=0.1, batch_size=2, queue_time=0.05, latency=0.15, queue_length=5)
        self.stats.update(processing_time=0.2, batch_size=3, queue_time=0.04, latency=0.24, queue_length=4)
        stats = self.stats.get_stats()
        
        # Check average processing time in ms
        self.assertAlmostEqual(stats['avg_processing_time_ms'], ((0.1+0.2)/2)*1000, places=2)
        # Check error rate is zero initially.
        self.assertEqual(stats['error_rate'], 0)
        # Check total processed count
        self.assertEqual(stats['total_processed'], 5)

    def test_record_error(self):
        # Record error for one batch.
        self.stats.record_error(2)
        stats = self.stats.get_stats()
        # Since total_processed is 0, error_rate should be calculated using max(1, total_processed)
        self.assertEqual(stats['error_rate'], 2 / 1)

class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        # Create two configurations: one with and one without a priority queue.
        self.config = get_test_config(priority_queue=False)
        self.config_priority = get_test_config(priority_queue=True)
        
    def tearDown(self):
        # Ensure that any running processor is stopped.
        pass

    def test_enqueue_and_process_numpy_batch(self):
        processor = BatchProcessor(self.config)
        # Start the processor using the numpy processing function.
        processor.start(process_numpy_func)
        
        # Create some numpy arrays to enqueue.
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6]])
        # Enqueue without expecting a return.
        processor.enqueue(arr1)
        # Enqueue with prediction, expecting a Future.
        future = processor.enqueue_predict(arr2)
        
        # Allow some time for the batch to be processed.
        time.sleep(1)
        
        # Check that the future has a result.
        try:
            result = future.result(timeout=1)
            # Since our function doubles the input, verify accordingly.
            np.testing.assert_array_equal(result, arr2 * 2)
        except TimeoutError:
            self.fail("Future did not complete in expected time.")
        
        # Stop the processor.
        processor.stop()

    def test_enqueue_and_process_generic_batch(self):
        processor = BatchProcessor(self.config)
        processor.start(process_generic_func)
        
        # Enqueue generic items.
        processor.enqueue("item1")
        future = processor.enqueue_predict("item2")
        
        # Allow processing time.
        time.sleep(1)
        
        # Get the result from the future.
        try:
            result = future.result(timeout=1)
            self.assertEqual(result, "processed-item2")
        except TimeoutError:
            self.fail("Future for generic processing did not complete.")
        
        processor.stop()

    def test_pause_resume(self):
        processor = BatchProcessor(self.config)
        processor.start(process_generic_func)
        
        # Pause the processor.
        processor.pause()
        self.assertTrue(processor.paused_event.is_set())
        
        # Enqueue an item while paused.
        future = processor.enqueue_predict("paused_item")
        
        # Give a little time to ensure it is not processed.
        time.sleep(0.3)
        # The future should not be done since the processor is paused.
        self.assertFalse(future.done())
        
        # Resume processing.
        processor.resume()
        # Wait for processing.
        time.sleep(1)
        self.assertTrue(future.done())
        self.assertEqual(future.result(), "processed-paused_item")
        
        processor.stop()

    def test_update_batch_size(self):
        processor = BatchProcessor(self.config)
        original_size = processor._current_batch_size
        # Update batch size and verify the change.
        processor.update_batch_size(5)
        self.assertEqual(processor._current_batch_size, 5)
        
        # Ensure that the batch size is clamped to configuration limits.
        processor.update_batch_size(1000)
        self.assertLessEqual(processor._current_batch_size, self.config.max_batch_size)
        processor.stop()

    def test_get_stats(self):
        processor = BatchProcessor(self.config)
        processor.start(process_generic_func)
        
        # Enqueue a couple of items.
        processor.enqueue("stats1")
        processor.enqueue("stats2")
        
        # Allow processing.
        time.sleep(1)
        stats = processor.get_stats()
        # Check that certain keys exist in the stats.
        for key in ["current_batch_size", "system_load", "active_batches", "queue_size", "is_paused", "is_stopping"]:
            self.assertIn(key, stats)
        
        processor.stop()

    def test_shutdown_with_pending_items(self):
        processor = BatchProcessor(self.config)
        processor.start(process_generic_func)
        
        # Enqueue an item but immediately stop the processor.
        future = processor.enqueue_predict("shutdown_item")
        processor.stop(timeout=1)
        
        # Since the processor is shutting down, the future should have an exception.
        with self.assertRaises(Exception):
            future.result(timeout=1)

    def test_priority_queue_enqueue(self):
        # Test processor with a priority queue enabled.
        processor = BatchProcessor(self.config_priority)
        processor.start(process_generic_func)
        
        # Enqueue items with different priorities.
        future_normal = processor.enqueue_predict("normal", priority=BatchPriority.NORMAL)
        future_high = processor.enqueue_predict("high", priority=BatchPriority.HIGH)
        
        # Allow processing time.
        time.sleep(1)
        # Check that both futures completed correctly.
        self.assertEqual(future_normal.result(), "processed-normal")
        self.assertEqual(future_high.result(), "processed-high")
        
        processor.stop()

if __name__ == '__main__':
    unittest.main()
