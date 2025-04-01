# Batch Processor Documentation

The `BatchProcessor` is a highly optimized, asynchronous batch processing system designed for handling large volumes of data with adaptive batching, priority queuing, and comprehensive monitoring. This document provides an overview of the key features, usage, and configuration options.

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Statistics and Monitoring](#statistics-and-monitoring)
6. [Error Handling](#error-handling)
7. [Health Monitoring](#health-monitoring)
8. [API Reference](#api-reference)

## Overview

The `BatchProcessor` is designed to efficiently process batches of data, particularly when dealing with large arrays or high-throughput systems. It supports adaptive batching, priority-based queuing, and detailed performance monitoring. The processor is thread-safe and optimized for memory usage, making it suitable for both CPU-bound and I/O-bound tasks.

## Key Features

- **Adaptive Batching**: Dynamically adjusts batch sizes based on system load and memory usage.
- **Priority Queuing**: Supports priority-based processing of items.
- **Comprehensive Monitoring**: Tracks processing times, queue lengths, error rates, and more.
- **Graceful Shutdown**: Ensures all pending items are processed before shutting down.
- **Error Handling**: Robust error handling with retries and automatic batch size adjustment.
- **Health Monitoring**: Background thread monitors system health, including memory usage and queue growth.

## Configuration

The `BatchProcessor` is configured using the `BatchProcessorConfig` class, which includes the following parameters:

- `initial_batch_size`: Initial size of each batch.
- `min_batch_size`: Minimum allowed batch size.
- `max_batch_size`: Maximum allowed batch size.
- `max_queue_size`: Maximum number of items in the queue.
- `batch_timeout`: Maximum time to wait for a batch to fill.
- `min_batch_interval`: Minimum time between batch processing.
- `max_retries`: Maximum number of retries for failed batches.
- `retry_delay`: Delay between retries.
- `enable_priority_queue`: Whether to use priority-based queuing.
- `enable_monitoring`: Whether to enable performance monitoring.
- `monitoring_window`: Number of batches to include in statistics.
- `enable_health_monitoring`: Whether to enable health monitoring.
- `health_check_interval`: Interval between health checks.
- `memory_warning_threshold`: Memory usage threshold for warnings.
- `memory_critical_threshold`: Memory usage threshold for critical actions.
- `queue_warning_threshold`: Queue size threshold for warnings.
- `queue_critical_threshold`: Queue size threshold for critical actions.
- `enable_memory_optimization`: Whether to enable memory optimization.
- `gc_batch_threshold`: Batch size threshold for garbage collection.

## Usage

### Initialization

```python
from configs import BatchProcessorConfig
from batch_processor import BatchProcessor

config = BatchProcessorConfig(
    initial_batch_size=100,
    min_batch_size=50,
    max_batch_size=200,
    max_queue_size=1000,
    batch_timeout=1.0,
    min_batch_interval=0.1,
    max_retries=3,
    retry_delay=0.5,
    enable_priority_queue=True,
    enable_monitoring=True,
    monitoring_window=100,
    enable_health_monitoring=True,
    health_check_interval=10.0,
    memory_warning_threshold=80.0,
    memory_critical_threshold=90.0,
    queue_warning_threshold=500,
    queue_critical_threshold=800,
    enable_memory_optimization=True,
    gc_batch_threshold=100
)

processor = BatchProcessor(config)
```

### Starting the Processor

```python
def process_batch(batch: np.ndarray) -> np.ndarray:
    # Process the batch and return results
    return batch * 2

processor.start(process_batch)
```

### Enqueuing Items

```python
# Enqueue an item without expecting a result
processor.enqueue(np.array([1, 2, 3]))

# Enqueue an item and get a future for the result
future = processor.enqueue_predict(np.array([4, 5, 6]))
result = future.result()  # Block until result is available
```

### Stopping the Processor

```python
processor.stop(timeout=5.0)
```

## Statistics and Monitoring

The `BatchProcessor` provides detailed statistics on processing performance, which can be accessed using the `get_stats` method:

```python
stats = processor.get_stats()
print(stats)
```

The statistics include:

- `avg_processing_time_ms`: Average processing time per batch.
- `p95_processing_time_ms`: 95th percentile processing time.
- `avg_batch_size`: Average batch size.
- `max_batch_size`: Maximum batch size.
- `min_batch_size`: Minimum batch size.
- `avg_queue_time_ms`: Average time items spend in the queue.
- `avg_latency_ms`: Average latency (queue time + processing time).
- `p95_latency_ms`: 95th percentile latency.
- `p99_latency_ms`: 99th percentile latency.
- `avg_queue_length`: Average queue length.
- `max_queue_length`: Maximum queue length.
- `error_rate`: Error rate (errors per processed item).
- `throughput`: Throughput in items per second.
- `batch_count`: Total number of batches processed.
- `total_processed`: Total number of items processed.
- `current_batch_size`: Current batch size.
- `system_load`: Current system load (0-1).
- `active_batches`: Number of batches currently being processed.
- `queue_size`: Current queue size.
- `is_paused`: Whether the processor is paused.
- `is_stopping`: Whether the processor is stopping.

## Error Handling

The `BatchProcessor` includes robust error handling with automatic retries and batch size adjustment. If a batch fails, the processor will retry up to `max_retries` times. If the error persists, the batch size may be reduced to prevent further failures.

## Health Monitoring

The processor includes a background health monitoring thread that checks for:

- **Stuck Batches**: Batches that have been processing for too long.
- **Memory Usage**: High memory usage, which may trigger garbage collection or batch size reduction.
- **Queue Growth**: Rapidly growing queue, which may trigger batch size increases.

## API Reference

### `BatchStats`

Tracks batch processing statistics with thread-safety and performance optimizations.

#### Methods

- `update(processing_time, batch_size, queue_time, latency, queue_length)`: Update statistics with batch processing information.
- `record_error(count)`: Record processing errors.
- `get_stats()`: Get processing statistics with caching for performance.

### `BatchProcessor`

Enhanced asynchronous batch processor with adaptive batching, monitoring, priority queuing, and efficient resource management.

#### Methods

- `start(process_func)`: Start the batch processing thread.
- `stop(timeout)`: Stop the batch processor with graceful shutdown.
- `pause()`: Pause processing temporarily.
- `resume()`: Resume processing after pause.
- `enqueue(item, timeout, priority)`: Enqueue item for processing without expecting results.
- `enqueue_predict(item, timeout, priority)`: Enqueue item for processing and return future for result.
- `update_batch_size(new_size)`: Update the current batch size.
- `get_stats()`: Get current processing statistics.

#### Internal Methods

- `_queue_is_prioritized()`: Check if the queue is using priority-based queuing.
- `_get_item_size(item)`: Get the size of an item for batching decisions.
- `_get_batch_memory_size(item)`: Estimate memory size of an item in bytes.
- `_wrapped_worker_loop(process_func)`: Wrapper for worker loop with error handling and cleanup.
- `_worker_loop(process_func)`: Enhanced worker loop with adaptive batching and monitoring.
- `_collect_batch()`: Collect items for a batch with adaptive sizing.
- `_process_batch(batch_requests, process_func)`: Process a batch of requests with retries and error handling.
- `_process_numpy_batch(batch_requests, process_func, batch_id, start_time, queue_time)`: Process a batch of numpy arrays.
- `_process_generic_batch(batch_requests, process_func, batch_id, start_time, queue_time)`: Process a batch of generic items individually.
- `_get_next_batch_id()`: Get a unique batch ID.
- `_update_system_load(processing_time)`: Update system load estimation based on processing time.
- `_adjust_batch_size_adaptive()`: Dynamically adjust batch size based on system load.
- `_handle_batch_error(error, batch_requests)`: Handle errors in batch processing.
- `_process_remaining_items(timeout)`: Process remaining items in the queue before shutdown.
- `_start_health_monitor()`: Start a background thread to monitor processor health.
- `_health_check_loop()`: Background thread for monitoring system health.
- `_check_stuck_batches()`: Check for batches that have been processing too long.
- `_check_memory_usage()`: Monitor memory usage and take action if it's too high.
- `_check_queue_growth()`: Monitor queue size and take action if it's growing too quickly.
- `_cleanup()`: Clean up resources during shutdown

Certainly! Here’s the continuation of the documentation:

---

### `_cleanup()` (continued)

Cleans up resources during shutdown, including failing any pending futures, clearing data structures, and signaling all waiting threads.

---

### `get_stats()`

Retrieves current processing statistics, including both basic and extended metrics.

---

## Example Use Case

### Processing Batches of Numpy Arrays

```python
import numpy as np
from configs import BatchProcessorConfig
from batch_processor import BatchProcessor

# Define a simple processing function
def process_batch(batch: np.ndarray) -> np.ndarray:
    return batch * 2  # Example: Double each element in the batch

# Configure the processor
config = BatchProcessorConfig(
    initial_batch_size=100,
    min_batch_size=50,
    max_batch_size=200,
    max_queue_size=1000,
    batch_timeout=1.0,
    min_batch_interval=0.1,
    max_retries=3,
    retry_delay=0.5,
    enable_priority_queue=True,
    enable_monitoring=True,
    monitoring_window=100,
    enable_health_monitoring=True,
    health_check_interval=10.0,
    memory_warning_threshold=80.0,
    memory_critical_threshold=90.0,
    queue_warning_threshold=500,
    queue_critical_threshold=800,
    enable_memory_optimization=True,
    gc_batch_threshold=100
)

# Initialize the processor
processor = BatchProcessor(config)

# Start the processor
processor.start(process_batch)

# Enqueue items for processing
for i in range(1000):
    item = np.random.rand(10)  # Example: 10-element numpy array
    processor.enqueue(item)

# Retrieve and print statistics
stats = processor.get_stats()
print("Processing Statistics:", stats)

# Stop the processor
processor.stop(timeout=5.0)
```

---

## Advanced Features

### Adaptive Batching

The processor dynamically adjusts the batch size based on system load and memory usage. If the system is under heavy load, the batch size is reduced to prevent overloading. Conversely, if the system is underutilized, the batch size is increased to improve throughput.

### Priority Queuing

When `enable_priority_queue` is set to `True`, items can be enqueued with different priorities (`BatchPriority.LOW`, `BatchPriority.NORMAL`, `BatchPriority.HIGH`). Higher-priority items are processed before lower-priority ones.

### Memory Optimization

When `enable_memory_optimization` is enabled, the processor uses pre-allocated arrays for numpy batches to minimize memory fragmentation and improve performance. Garbage collection is also triggered opportunistically to free up memory.

### Health Monitoring

The health monitoring thread continuously checks for:

- **Stuck Batches**: Batches that exceed the expected processing time.
- **Memory Usage**: High memory usage triggers garbage collection and batch size reduction.
- **Queue Growth**: Rapidly growing queues trigger batch size increases to prevent backlogs.

---

## Error Handling and Recovery

The processor includes robust error handling mechanisms:

- **Retries**: Failed batches are retried up to `max_retries` times.
- **Batch Size Reduction**: If a batch fails repeatedly, the batch size is reduced to prevent further failures.
- **Graceful Shutdown**: During shutdown, all pending items are processed, and any remaining futures are marked as failed.

---

## Performance Considerations

- **Thread Safety**: The processor uses `RLock` and `Lock` to ensure thread safety during concurrent operations.
- **Caching**: Statistics are cached to avoid expensive recalculations, improving performance.
- **Efficient Batching**: For numpy arrays, the processor uses `np.vstack` or pre-allocated arrays to efficiently combine batches.

---

## Troubleshooting

### High Memory Usage

If memory usage exceeds the configured thresholds:

1. Enable `enable_memory_optimization` to reduce memory fragmentation.
2. Reduce the `max_batch_size` to limit the amount of data processed at once.
3. Trigger garbage collection manually using `gc.collect()`.

### Queue Backlogs

If the queue grows too quickly:

1. Increase the `max_batch_size` to process more items per batch.
2. Reduce the `min_batch_interval` to process batches more frequently.
3. Check the processing function for bottlenecks.

### Stuck Batches

If batches are taking too long to process:

1. Increase the `batch_timeout` to allow more time for processing.
2. Optimize the processing function to reduce execution time.
3. Check for system-level issues (e.g., CPU or I/O bottlenecks).

---

## Conclusion

The `BatchProcessor` is a powerful tool for handling large-scale batch processing tasks with adaptive batching, priority queuing, and comprehensive monitoring. Its robust error handling and health monitoring features make it suitable for production environments where reliability and performance are critical.

For further customization, refer to the source code and adjust the configuration parameters as needed.

--- 
