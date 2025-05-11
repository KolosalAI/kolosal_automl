# Module: `batch_processor.py`

## Overview
A high-performance asynchronous batch processing system with adaptive batching capabilities, designed for efficient processing of data items (particularly NumPy arrays) in batches. The system supports priority queuing, comprehensive monitoring, memory optimization, and fault tolerance.

## Prerequisites
- Python ≥3.8
- Required packages:
  ```bash
  pip install numpy~=1.20.0 psutil
  ```
- Optional dependencies:
  - `psutil`: For memory monitoring (automatically detected)
- Custom module dependencies:
  - Local module `configs` with classes: `BatchProcessorConfig`, `BatchProcessingStrategy`, `BatchPriority`, `PrioritizedItem`

## Installation
Ensure all dependencies are installed and the module is accessible in your Python path. The module is designed to be used as part of a larger system.

## Usage
```python
from modules.engine.batch_processor import BatchProcessor
from configs import BatchProcessorConfig, BatchProcessingStrategy

# Create a configuration
config = BatchProcessorConfig(
    initial_batch_size=32,
    max_batch_size=128,
    enable_priority_queue=True
)

# Initialize the processor
processor = BatchProcessor(config)

# Start processing with a function that handles batched inputs
processor.start(process_func=my_batch_processing_function)

# Enqueue items for processing without waiting for results
processor.enqueue(item)

# Enqueue items and get a Future for the result
future = processor.enqueue_predict(item, priority=BatchPriority.HIGH)
result = future.result()  # Wait for result

# Stop the processor when done
processor.stop(timeout=10.0)
```

## Configuration
The processor is configured using a `BatchProcessorConfig` object with the following settings:

| Parameter | Description |
|-----------|-------------|
| `initial_batch_size` | Starting batch size for processing |
| `min_batch_size` | Minimum batch size allowed during adaptive sizing |
| `max_batch_size` | Maximum batch size allowed |
| `max_queue_size` | Maximum items in queue before blocking |
| `batch_timeout` | Maximum time to wait while forming a batch (seconds) |
| `item_timeout` | Maximum time to process a single item (seconds) |
| `enable_priority_queue` | Whether to enable priority-based processing |
| `max_workers` | Maximum worker threads in thread pool |
| `enable_monitoring` | Whether to collect performance statistics |
| `monitoring_window` | Number of batches to keep statistics for |
| `max_retries` | Maximum retry attempts for failed batches |
| `retry_delay` | Delay between retries (seconds) |
| `enable_adaptive_batching` | Whether to dynamically adjust batch size |
| `processing_strategy` | Strategy for batch processing (e.g., ADAPTIVE) |
| `min_batch_interval` | Minimum time between batch processing start (seconds) |
| `max_batch_memory_mb` | Maximum memory per batch in MB |
| `enable_memory_optimization` | Whether to optimize memory usage |
| `gc_batch_threshold` | Batch size threshold for garbage collection |
| `debug_mode` | Enable verbose logging |
| `enable_health_monitoring` | Whether to monitor system health |
| `health_check_interval` | Interval between health checks (seconds) |
| `memory_warning_threshold` | Memory percentage to trigger warnings |
| `memory_critical_threshold` | Memory percentage to trigger critical actions |
| `queue_warning_threshold` | Queue size to trigger warnings |
| `queue_critical_threshold` | Queue size to trigger critical actions |
| `reduce_batch_on_failure` | Whether to reduce batch size after failures |

---

## Classes

### `BatchStats`
```python
class BatchStats:
```
- **Description**:  
  Tracks and calculates batch processing statistics with thread-safety and performance optimizations through caching.

- **Attributes**:  
  - `window_size (int)`: Number of samples to keep in the statistics window.
  - `processing_times (deque)`: Recent batch processing times in seconds.
  - `batch_sizes (deque)`: Recent batch sizes.
  - `queue_times (deque)`: Recent queue waiting times in seconds.
  - `latencies (deque)`: Recent end-to-end latencies in seconds.
  - `queue_lengths (deque)`: Recent queue lengths.
  - `error_counts (int)`: Count of processing errors.
  - `batch_counts (int)`: Total number of batches processed.
  - `total_processed (int)`: Total number of items processed.
  - `lock (RLock)`: Thread synchronization lock.
  - `_stats_cache (dict)`: Cache for computed statistics.
  - `_last_update_time (float)`: Time of last cache update.
  - `_cache_valid_time (float)`: How long cache remains valid in seconds.

- **Constructor**:
  ```python
  def __init__(self, window_size: int)
  ```
  - **Parameters**:
    - `window_size (int)`: Number of recent measurements to keep for statistics.

- **Methods**:  
  - `update(processing_time: float, batch_size: int, queue_time: float, latency: float = 0.0, queue_length: int = 0) -> None`:
    ```python
    def update(self, processing_time: float, batch_size: int, queue_time: float, 
               latency: float = 0.0, queue_length: int = 0) -> None
    ```
    - **Description**:  
      Update statistics with batch processing information.
    - **Parameters**:
      - `processing_time (float)`: Time taken to process the batch in seconds.
      - `batch_size (int)`: Number of items in the batch.
      - `queue_time (float)`: Time items spent in queue in seconds.
      - `latency (float)`: End-to-end latency in seconds. Default is 0.0.
      - `queue_length (int)`: Current queue length. Default is 0.
    - **Returns**:  
      - `None`

  - `record_error(count: int = 1) -> None`:
    ```python
    def record_error(self, count: int = 1) -> None
    ```
    - **Description**:  
      Record processing errors.
    - **Parameters**:
      - `count (int)`: Number of errors to record. Default is 1.
    - **Returns**:  
      - `None`

  - `get_stats() -> Dict[str, float]`:
    ```python
    def get_stats(self) -> Dict[str, float]
    ```
    - **Description**:  
      Get a dictionary of current processing statistics with caching for performance.
    - **Returns**:  
      - `Dict[str, float]`: Dictionary containing statistics like average processing time, latency percentiles, batch sizes, error rates, and throughput.

### `BatchProcessor`
```python
class BatchProcessor(Generic[T, U]):
```
- **Description**:  
  Enhanced asynchronous batch processor with adaptive batching, monitoring, priority queuing, and efficient resource management. Optimized for processing data items (particularly NumPy arrays) in batches.

- **Attributes**:  
  - `config (BatchProcessorConfig)`: Configuration parameters.
  - `metrics (Any)`: Optional external metrics collection system.
  - `queue (Union[Queue, PriorityQueue])`: Queue for holding items to be processed.
  - `stop_event (Event)`: Signal to stop processing.
  - `paused_event (Event)`: Signal that processing is paused.
  - `shutdown_event (Event)`: Signal for complete shutdown.
  - `batch_ready_event (Event)`: Signal when batch is ready for processing.
  - `worker_thread (Optional[Thread])`: Main processing thread.
  - `worker_lock (RLock)`: Lock for thread management.
  - `_current_batch_size (int)`: Current target batch size.
  - `_batch_size_lock (RLock)`: Lock for batch size adjustments.
  - `_max_batch_memory (Optional[int])`: Maximum memory per batch in bytes.
  - `_system_load (float)`: Estimate of current system load (0-1).
  - `executor (ThreadPoolExecutor)`: Thread pool for parallel processing.
  - `stats (Optional[BatchStats])`: Statistics collection.
  - `_futures (weakref.WeakSet[Future])`: Weak references to all pending futures.
  - `_active_batches (Set[int])`: Set of batch IDs currently processing.
  - `logger (Logger)`: Logger instance.

- **Constructor**:
  ```python
  def __init__(self, config: BatchProcessorConfig, metrics_collector: Optional[Any] = None)
  ```
  - **Parameters**:
    - `config (BatchProcessorConfig)`: Configuration parameters for the processor.
    - `metrics_collector (Optional[Any])`: Optional external metrics collection system. Default is None.

- **Methods**:  
  - `start(process_func: Callable[[NDArray[T]], NDArray[U]]) -> None`:
    ```python
    def start(self, process_func: Callable[[NDArray[T]], NDArray[U]]) -> None
    ```
    - **Description**:  
      Start the batch processing thread.
    - **Parameters**:
      - `process_func (Callable[[NDArray[T]], NDArray[U]])`: Function that processes batched inputs.
    - **Returns**:  
      - `None`

  - `stop(timeout: Optional[float] = 5.0) -> None`:
    ```python
    def stop(self, timeout: Optional[float] = 5.0) -> None
    ```
    - **Description**:  
      Stop the batch processor with graceful shutdown.
    - **Parameters**:
      - `timeout (Optional[float])`: Maximum time to wait for shutdown in seconds. Default is 5.0.
    - **Returns**:  
      - `None`

  - `pause() -> None`:
    ```python
    def pause(self) -> None
    ```
    - **Description**:  
      Pause processing temporarily.
    - **Returns**:  
      - `None`

  - `resume() -> None`:
    ```python
    def resume(self) -> None
    ```
    - **Description**:  
      Resume processing after pause.
    - **Returns**:  
      - `None`

  - `enqueue(item: T, timeout: Optional[float] = None, priority: BatchPriority = BatchPriority.NORMAL) -> None`:
    ```python
    def enqueue(self, item: T, timeout: Optional[float] = None, 
                priority: BatchPriority = BatchPriority.NORMAL) -> None
    ```
    - **Description**:  
      Enqueue item for processing without expecting results.
    - **Parameters**:
      - `item (T)`: Item to process.
      - `timeout (Optional[float])`: Maximum time to wait if queue is full in seconds. Default is None.
      - `priority (BatchPriority)`: Processing priority for this item. Default is NORMAL.
    - **Returns**:  
      - `None`

  - `enqueue_predict(item: T, timeout: Optional[float] = None, priority: BatchPriority = BatchPriority.NORMAL) -> Future[U]`:
    ```python
    def enqueue_predict(self, item: T, timeout: Optional[float] = None,
                       priority: BatchPriority = BatchPriority.NORMAL) -> Future[U]
    ```
    - **Description**:  
      Enqueue item for processing and return future for result.
    - **Parameters**:
      - `item (T)`: Item to process.
      - `timeout (Optional[float])`: Maximum time to wait if queue is full in seconds. Default is None.
      - `priority (BatchPriority)`: Processing priority for this item. Default is NORMAL.
    - **Returns**:  
      - `Future[U]`: Future object that will contain the result.
    - **Raises**:
      - `RuntimeError`: If the processor is shutting down or queue is full.

  - `update_batch_size(new_size: int) -> None`:
    ```python
    def update_batch_size(self, new_size: int) -> None
    ```
    - **Description**:  
      Update the current batch size.
    - **Parameters**:
      - `new_size (int)`: New target batch size.
    - **Returns**:  
      - `None`

  - `get_stats() -> Dict[str, Any]`:
    ```python
    def get_stats(self) -> Dict[str, Any]
    ```
    - **Description**:  
      Get current processing statistics.
    - **Returns**:  
      - `Dict[str, Any]`: Dictionary of statistics or empty dict if monitoring disabled.

  - *Internal methods* (not typically called directly by users):
    - `_queue_is_prioritized() -> bool`: Check if the queue is using priority-based queuing.
    - `_get_item_size(item: T) -> int`: Get the size of an item for batching decisions.
    - `_get_batch_memory_size(item: T) -> int`: Estimate memory size of an item in bytes.
    - `_wrapped_worker_loop(process_func: Callable[[NDArray[T]], NDArray[U]]) -> None`: Wrapper for worker loop with error handling.
    - `_worker_loop(process_func: Callable[[NDArray[T]], NDArray[U]]) -> None`: Main processing loop.
    - `_collect_batch() -> List[Tuple[T, Optional[Future[U]], int]]`: Collect items for a batch with adaptive sizing.
    - `_process_batch(batch_requests: List[Tuple[T, Optional[Future[U]], int]], process_func: Callable[[NDArray[T]], NDArray[U]]) -> None`: Process a batch of requests.
    - `_process_numpy_batch(batch_requests: List[Tuple[T, Optional[Future[U]], int]], process_func: Callable[[NDArray[T]], NDArray[U]], batch_id: int, start_time: float, queue_time: float) -> None`: Process a batch of numpy arrays.
    - `_process_generic_batch(batch_requests: List[Tuple[T, Optional[Future[U]], int]], process_func: Callable[[T], U], batch_id: int, start_time: float, queue_time: float) -> None`: Process a batch of generic items.
    - `_get_next_batch_id() -> int`: Get a unique batch ID.
    - `_update_system_load(processing_time: float) -> None`: Update system load estimation.
    - `_adjust_batch_size_adaptive() -> None`: Dynamically adjust batch size.
    - `_handle_batch_error(error: Exception, batch_requests: List[Tuple[T, Optional[Future[U]], int]]) -> None`: Handle errors in batch processing.
    - `_process_remaining_items(timeout: float) -> None`: Process remaining items before shutdown.
    - `_start_health_monitor() -> None`: Start health monitoring thread.
    - `_health_check_loop() -> None`: Background health monitoring thread.
    - `_check_stuck_batches() -> None`: Check for batches processing too long.
    - `_check_memory_usage() -> None`: Monitor memory usage.
    - `_check_queue_growth() -> None`: Monitor queue size growth.
    - `_cleanup() -> None`: Clean up resources during shutdown.

## Architecture

### Data Flow
1. Items are enqueued for processing with optional priorities
2. Worker thread collects items into batches adaptively
3. For NumPy arrays, batches are efficiently combined using vectorized operations
4. Other items are processed individually through a thread pool
5. Results are delivered through Futures
6. Performance metrics are collected and used for dynamic optimization

### Performance Optimizations
- Adaptive batch sizing based on system load
- Memory footprint management and garbage collection
- Caching of statistics calculations
- Efficient numpy array handling with pre-allocation
- Priority-based processing for important items
- Thread-safe queuing with locking

## Testing
```python
# Basic test to ensure processor works as expected
def test_batch_processor():
    from configs import BatchProcessorConfig, BatchProcessingStrategy, BatchPriority
    
    # Create test configuration
    config = BatchProcessorConfig(
        initial_batch_size=10,
        max_batch_size=100,
        enable_priority_queue=True
    )
    
    # Initialize processor
    processor = BatchProcessor(config)
    
    # Define process function
    def process_func(batch):
        return batch * 2  # Double each value
    
    # Start processor
    processor.start(process_func)
    
    # Enqueue items and get results
    futures = []
    for i in range(50):
        item = np.array([i])
        future = processor.enqueue_predict(item)
        futures.append(future)
    
    # Wait for results
    results = [future.result() for future in futures]
    
    # Check results
    for i, result in enumerate(results):
        assert result[0] == i * 2
    
    # Stop processor
    processor.stop()

# Run the test
if __name__ == "__main__":
    test_batch_processor()
```

## Security & Compliance
- Thread-safe implementation with proper locking
- Graceful error handling and recovery
- Memory usage monitoring to prevent out-of-memory conditions
- Resource cleanup on shutdown
- Configurable retry policies for handling transient failures

## Optimization Considerations
- The batch processor is optimized for processing numpy arrays
- For very large batches, memory optimization is critical
- Performance is best when batch sizes are tuned to the specific workload
- Consider enabling priority queuing for mixed-priority workloads
- Monitor memory usage when processing large datasets

> Last Updated: 2025-05-11