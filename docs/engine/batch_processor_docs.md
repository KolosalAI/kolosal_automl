# Module: `modules.engine.batch_processor`

## Overview
This module implements an advanced asynchronous batch processing engine designed for
high-throughput and low-latency batch processing. It includes features such as adaptive
batch sizing, system load tracking, priority queuing, error handling, monitoring, and
resource optimization.

It supports both numpy-based and generic data processing, ensuring flexibility across
a wide range of applications.

## Prerequisites
- Python ≥3.8
- Required Packages:
  ```bash
  pip install numpy psutil
  ```
- Hardware: Multi-core CPU recommended
- Optional: `psutil` library for enhanced memory monitoring

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from modules.engine.batch_processor import BatchProcessor

# Initialize configuration
config = BatchProcessorConfig(...)

# Create batch processor
processor = BatchProcessor(config=config)

# Start processing with a processing function
def my_process_func(batch):
    return model.predict(batch)

processor.start(my_process_func)

# Enqueue an item
processor.enqueue_predict(item)

# Stop processor
processor.stop()
```

## Configuration
| Config Option                  | Type          | Description                                         |
|---------------------------------|---------------|-----------------------------------------------------|
| `max_queue_size`                | int           | Max items allowed in the queue                      |
| `initial_batch_size`            | int           | Starting batch size                                |
| `max_batch_memory_mb`           | Optional[int] | Limit for memory size of batches                   |
| `batch_timeout`                 | float         | Max time to wait for batch completion (seconds)     |
| `min_batch_size`, `max_batch_size` | int        | Bounds for batch size during adaptation            |
| `processing_strategy`           | Enum          | Batching strategy (e.g., Adaptive, Fixed)           |
| `enable_priority_queue`         | bool          | Enable priority queueing                           |
| `enable_monitoring`             | bool          | Enable metrics collection                          |
| `debug_mode`                    | bool          | Enable debug logging                               |

_(Refer to `BatchProcessorConfig` in `configs.py` for full details.)_

## Architecture
**Core Components:**
- `BatchStats`: Tracks processing time, latency, throughput.
- `BatchProcessor`: Manages batch lifecycle, adaptive control, memory optimization.
- Threading model: Dedicated worker and health monitor threads.
- System load estimation based on processing time.
- Health monitoring: memory, queue size, stuck batch detection.

## Testing
```bash
pytest tests/engine/test_batch_processor.py
```
Recommended to test:
- Normal batch processing
- Priority-based queue handling
- Error recovery under simulated failures
- Memory optimization impact

## Security/Compliance
- Data is processed in-memory.
- No external network calls.
- Sensitive data handling should be added at the application level.

## Versioning and Metadata
> Last Updated: 2025-04-28

- Compatible with Python 3.8+
- Designed for modular extension (custom processors, monitoring hooks)

---

# Classes and Functions

## Class: `BatchStats`
```python
class BatchStats:
```
### Description
Tracks and computes moving-window statistics for batch processing performance, with thread-safe
updates and caching for efficiency.

### Attributes
| Name | Type | Purpose |
|-----|------|---------|
| `window_size` | int | Number of recent samples to keep |
| `processing_times` | deque | Recorded processing times |
| `batch_sizes` | deque | Batch size per operation |
| `queue_times` | deque | Time spent in queue |
| `latencies` | deque | Total latency per batch |
| `queue_lengths` | deque | Length of queue over time |
| `error_counts` | int | Number of processing errors |
| `batch_counts` | int | Number of batches processed |
| `total_processed` | int | Total items processed |

### Methods

#### `update`
```python
def update(self, processing_time: float, batch_size: int, queue_time: float, latency: float = 0.0, queue_length: int = 0) -> None
```
Update statistics with a new batch.

#### `record_error`
```python
def record_error(self, count: int = 1) -> None
```
Increment error counter.

#### `get_stats`
```python
def get_stats(self) -> Dict[str, float]
```
Retrieve current statistics (averages, percentiles, throughput, error rate).

---

## Class: `BatchProcessor`
```python
class BatchProcessor(Generic[T, U]):
```

### Description
Manages adaptive, asynchronous batch processing with options for priority queuing,
performance monitoring, dynamic batch sizing, and robust error recovery.

### Constructor
```python
def __init__(self, config: BatchProcessorConfig, metrics_collector: Optional[Any] = None)
```
Initialize processor with provided config and optional external metrics system.

### Main Methods

#### `start`
```python
def start(self, process_func: Callable[[NDArray[T]], NDArray[U]]) -> None
```
Start batch processor with a given processing function.

#### `stop`
```python
def stop(self, timeout: Optional[float] = 5.0) -> None
```
Gracefully stop processor with timeout.

#### `pause`
```python
def pause(self) -> None
```
Pause batch processing.

#### `resume`
```python
def resume(self) -> None
```
Resume batch processing.

#### `enqueue`
```python
def enqueue(self, item: T, timeout: Optional[float] = None, priority: BatchPriority = BatchPriority.NORMAL) -> None
```
Submit item without waiting for results.

#### `enqueue_predict`
```python
def enqueue_predict(self, item: T, timeout: Optional[float] = None, priority: BatchPriority = BatchPriority.NORMAL) -> Future[U]
```
Submit item and return a future to retrieve result.

#### `update_batch_size`
```python
def update_batch_size(self, new_size: int) -> None
```
Update batch size manually.

#### `get_stats`
```python
def get_stats(self) -> Dict[str, Any]
```
Fetch runtime performance statistics.

### Internal and Support Methods
- `_worker_loop`, `_wrapped_worker_loop`
- `_collect_batch`, `_process_batch`, `_process_numpy_batch`, `_process_generic_batch`
- `_handle_batch_error`
- `_adjust_batch_size_adaptive`, `_update_system_load`
- `_start_health_monitor`, `_health_check_loop`

### Examples
```python
# Using BatchProcessor
processor.enqueue_predict(np.random.randn(10, 128))

# Pause and resume
processor.pause()
time.sleep(5)
processor.resume()
```

---

# Notes
- **BatchProcessor** is optimized for ML inference workloads but can be extended for any batched processing.
- **BatchStats** helps in tracking and diagnosing performance bottlenecks.
- **Adaptive batching** ensures robustness even under system load fluctuations.

---
