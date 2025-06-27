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

try:
    from modules.engine.batch_processor import (
        BatchProcessor, BatchProcessorConfig, BatchProcessingStrategy, BatchPriority
    )
except ImportError as e:
    pytest.skip(f"Batch processor module not available: {e}", allow_module_level=True)


@pytest.mark.unit
class TestBatchProcessor:
    """Test the main BatchProcessor class."""
    
    @pytest.fixture(autouse=True)
    def setup_processor(self):
        """Setup batch processor for each test."""
        config = BatchProcessorConfig(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=8,
            batch_timeout=0.1,
            max_queue_size=100,
            enable_priority_queue=True,
            processing_strategy=BatchProcessingStrategy.ADAPTIVE,
            enable_adaptive_batching=True,
            enable_monitoring=False,
            num_workers=2,
            enable_memory_optimization=False,
            max_retries=3
        )
        
        # Mock process function
        def mock_process_function(batch):
            """Mock function that processes a batch."""
            return [f"processed_{item}" for item in batch]
        
        self.processor = BatchProcessor(config)
        self.mock_process_function = mock_process_function
        
        yield
        
        # Cleanup
        if hasattr(self, 'processor') and self.processor:
            try:
                self.processor.stop()
            except:
                pass
    
    def test_processor_initialization(self):
        """Test that processor initializes correctly."""
        assert self.processor is not None
        assert self.processor.config.initial_batch_size == 4
        assert self.processor.config.max_batch_size == 8
    
    def test_start_and_stop(self):
        """Test starting and stopping the processor."""
        # Start the processor
        self.processor.start(self.mock_process_function)
        
        # Should be running
        assert self.processor.running
        
        # Stop the processor
        self.processor.stop()
        
        # Should be stopped
        assert not self.processor.running
    
    def test_submit_single_item(self):
        """Test submitting a single item for processing."""
        self.processor.start(self.mock_process_function)
        
        # Submit an item
        future = self.processor.submit("test_item")
        
        # Should get a result
        result = future.result(timeout=1.0)
        assert result == "processed_test_item"
        
        self.processor.stop()
    
    def test_submit_multiple_items(self):
        """Test submitting multiple items."""
        self.processor.start(self.mock_process_function)
        
        # Submit multiple items
        futures = []
        for i in range(5):
            future = self.processor.submit(f"item_{i}")
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            result = future.result(timeout=1.0)
            results.append(result)
        
        # Check results
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result == f"processed_item_{i}"
        
        self.processor.stop()
    
    def test_batch_processing(self):
        """Test that items are processed in batches."""
        processed_batches = []
        
        def batch_tracking_function(batch):
            """Function that tracks batch sizes."""
            processed_batches.append(len(batch))
            return [f"processed_{item}" for item in batch]
        
        self.processor.start(batch_tracking_function)
        
        # Submit multiple items quickly
        futures = []
        for i in range(8):
            future = self.processor.submit(f"item_{i}")
            futures.append(future)
        
        # Wait for processing
        for future in futures:
            future.result(timeout=1.0)
        
        self.processor.stop()
        
        # Should have processed items in batches
        assert len(processed_batches) > 0
        assert max(processed_batches) > 1  # At least one batch should have multiple items
    
    def test_timeout_processing(self):
        """Test that items are processed even if batch isn't full after timeout."""
        processed_batches = []
        
        def batch_tracking_function(batch):
            processed_batches.append(len(batch))
            return [f"processed_{item}" for item in batch]
        
        self.processor.start(batch_tracking_function)
        
        # Submit just one item
        future = self.processor.submit("single_item")
        
        # Should still get processed due to timeout
        result = future.result(timeout=1.0)
        assert result == "processed_single_item"
        
        self.processor.stop()
        
        # Should have processed at least one batch
        assert len(processed_batches) > 0
    
    @pytest.mark.slow
    def test_error_handling(self):
        """Test error handling in batch processing."""
        def error_function(batch):
            """Function that raises an error."""
            raise ValueError("Test error")
        
        self.processor.start(error_function)
        
        # Submit an item
        future = self.processor.submit("error_item")
        
        # Should get an exception
        with pytest.raises(ValueError, match="Test error"):
            future.result(timeout=1.0)
        
        self.processor.stop()
    
    def test_stats_collection(self):
        """Test that statistics are collected."""
        self.processor.start(self.mock_process_function)
        
        # Submit some items
        for i in range(3):
            future = self.processor.submit(f"item_{i}")
            future.result(timeout=1.0)
        
        # Get stats
        stats = self.processor.get_stats()
        
        self.processor.stop()
        
        # Should have collected some stats
        assert "total_items_processed" in stats
        assert stats["total_items_processed"] >= 3
        assert "total_batches_processed" in stats
        assert stats["total_batches_processed"] > 0


@pytest.mark.unit
class TestBatchProcessorConfig:
    """Test the BatchProcessorConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BatchProcessorConfig()
        
        assert config.initial_batch_size > 0
        assert config.max_batch_size >= config.min_batch_size
        assert config.batch_timeout > 0
        assert config.max_queue_size > 0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BatchProcessorConfig(
            initial_batch_size=10,
            max_batch_size=50,
            batch_timeout=0.5,
            max_queue_size=1000
        )
        
        assert config.initial_batch_size == 10
        assert config.max_batch_size == 50
        assert config.batch_timeout == 0.5
        assert config.max_queue_size == 1000


if __name__ == "__main__":
    pytest.main([__file__])
