# Module: `batch_processor.py`

## Overview
A high-performance asynchronous batch processing system with advanced adaptive batching capabilities, designed for efficient processing of machine learning workloads. The system supports priority queuing, comprehensive monitoring, memory optimization, health monitoring, and fault tolerance with production-ready features for ML inference and training pipelines.

## Prerequisites
- Python ≥3.10
- Required packages:
  ```bash
  pip install numpy>=1.20.0 
  ```
- Optional dependencies:
  - `psutil`: For advanced memory and system monitoring (automatically detected)
- Custom module dependencies:
  - Local module `configs` with classes: `BatchProcessorConfig`, `BatchProcessingStrategy`, `BatchPriority`, `PrioritizedItem`
  - Local module `batch_stats` with class: `BatchStats`

## Installation
Ensure all dependencies are installed and the module is accessible in your Python path. The module is designed to be used as part of a larger ML system.

## Quick Start
```python
from modules.engine.batch_processor import BatchProcessor
from modules.configs import BatchProcessorConfig, BatchProcessingStrategy, BatchPriority
import numpy as np

# Create a production-ready configuration
config = BatchProcessorConfig(
    initial_batch_size=32,
    max_batch_size=256,
    enable_priority_queue=True,
    enable_adaptive_batching=True,
    enable_monitoring=True,
    enable_health_monitoring=True,
    processing_strategy=BatchProcessingStrategy.ADAPTIVE,
    max_batch_memory_mb=512,
    enable_memory_optimization=True
)

# Initialize the processor
processor = BatchProcessor(config)

# Define your ML processing function
def ml_inference_func(batch_data):
    # Your ML model inference here
    # batch_data is a numpy array with shape (batch_size, features)
    return model.predict(batch_data)

# Start processing
processor.start(process_func=ml_inference_func)

# Enqueue high-priority inference requests
future = processor.enqueue_predict(
    data, 
    priority=BatchPriority.HIGH, 
    timeout=30.0
)

# Get result asynchronously
result = future.result()

# Stop the processor when done
processor.stop(timeout=10.0)
```

## Advanced Configuration
The processor is configured using a `BatchProcessorConfig` object with comprehensive settings for production ML workloads:

### Core Batching Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `initial_batch_size` | `int` | Starting batch size for processing | 32 |
| `min_batch_size` | `int` | Minimum batch size during adaptive sizing | 1 |
| `max_batch_size` | `int` | Maximum batch size allowed | 128 |
| `batch_timeout` | `float` | Maximum time to wait while forming a batch (seconds) | 0.1 |
| `item_timeout` | `float` | Maximum time to process a single item (seconds) | 30.0 |

### Queue & Threading Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_queue_size` | `int` | Maximum items in queue before blocking | 1000 |
| `enable_priority_queue` | `bool` | Enable priority-based processing | False |
| `max_workers` | `int` | Maximum worker threads in thread pool | 4 |
| `min_batch_interval` | `float` | Minimum time between batch processing start (seconds) | 0.0 |

### Memory Management Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_batch_memory_mb` | `Optional[int]` | Maximum memory per batch in MB | None |
| `enable_memory_optimization` | `bool` | Enable memory usage optimization | True |
| `gc_batch_threshold` | `int` | Batch size threshold for garbage collection | 100 |
| `memory_warning_threshold` | `float` | Memory percentage to trigger warnings | 80.0 |
| `memory_critical_threshold` | `float` | Memory percentage to trigger critical actions | 90.0 |

### Adaptive Processing Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enable_adaptive_batching` | `bool` | Enable dynamic batch size adjustment | True |
| `processing_strategy` | `BatchProcessingStrategy` | Strategy for batch processing | ADAPTIVE |
| `reduce_batch_on_failure` | `bool` | Reduce batch size after failures | True |

### Monitoring & Health Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enable_monitoring` | `bool` | Collect performance statistics | True |
| `monitoring_window` | `int` | Number of batches to keep statistics for | 1000 |
| `enable_health_monitoring` | `bool` | Enable system health monitoring | True |
| `health_check_interval` | `float` | Interval between health checks (seconds) | 30.0 |
| `queue_warning_threshold` | `int` | Queue size to trigger warnings | 500 |
| `queue_critical_threshold` | `int` | Queue size to trigger critical actions | 800 |

### Error Handling & Retry Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_retries` | `int` | Maximum retry attempts for failed batches | 3 |
| `retry_delay` | `float` | Delay between retries (seconds) | 1.0 |
| `debug_mode` | `bool` | Enable verbose logging | False |

### Example Configurations

#### High-Throughput Configuration
```python
high_throughput_config = BatchProcessorConfig(
    initial_batch_size=128,
    max_batch_size=512,
    batch_timeout=0.05,  # Faster batching
    enable_priority_queue=False,  # Skip priority overhead
    max_workers=8,
    enable_adaptive_batching=True,
    max_batch_memory_mb=1024,
    enable_memory_optimization=True
)
```

#### Low-Latency Configuration
```python
low_latency_config = BatchProcessorConfig(
    initial_batch_size=8,
    max_batch_size=32,
    batch_timeout=0.01,  # Very fast batching
    enable_priority_queue=True,
    min_batch_interval=0.001,
    enable_adaptive_batching=True,
    enable_health_monitoring=True
)
```

#### Memory-Constrained Configuration
```python
memory_constrained_config = BatchProcessorConfig(
    initial_batch_size=16,
    max_batch_size=64,
    max_batch_memory_mb=256,  # Strict memory limit
    enable_memory_optimization=True,
    gc_batch_threshold=32,  # Frequent GC
    memory_warning_threshold=70.0,
    memory_critical_threshold=85.0,
    reduce_batch_on_failure=True
)
```

---

## Core Classes

### `BatchProcessor[T, U]`
```python
class BatchProcessor(Generic[T, U]):
```

A production-ready asynchronous batch processor with advanced features for ML workloads.

#### **Key Features**
- **🚀 Adaptive Batching**: Dynamic batch size adjustment based on system load
- **⚡ Priority Processing**: Priority-based queue for urgent requests
- **🧠 Memory Management**: Advanced memory optimization with automatic GC
- **📊 Health Monitoring**: Real-time system health checks and alerts
- **🔄 Fault Tolerance**: Comprehensive error handling and retry mechanisms
- **📈 Performance Analytics**: Detailed metrics collection and monitoring
- **🎯 Load Balancing**: Intelligent resource utilization optimization

#### **Constructor**
```python
def __init__(self, config: BatchProcessorConfig, metrics_collector: Optional[Any] = None)
```

**Parameters:**
- `config`: Configuration parameters for the processor
- `metrics_collector`: Optional external metrics collection system

#### **Core Methods**

##### Lifecycle Management
```python
def start(self, process_func: Callable[[NDArray[T]], NDArray[U]]) -> None
```
Start the batch processing with your ML function.

**Parameters:**
- `process_func`: Function that processes batched inputs (e.g., ML model inference)

```python
def stop(self, timeout: Optional[float] = 5.0) -> None
```
Gracefully stop the processor with proper cleanup.

**Parameters:**
- `timeout`: Maximum time to wait for shutdown (seconds)

```python
def pause() -> None
def resume() -> None
```
Temporarily pause/resume processing without stopping the processor.

##### Request Processing
```python
def enqueue_predict(
    self, 
    item: T, 
    timeout: Optional[float] = None,
    priority: BatchPriority = BatchPriority.NORMAL
) -> Future[U]
```
Enqueue item for processing and get a Future for the result.

**Parameters:**
- `item`: Data to process (numpy array, tensor, etc.)
- `timeout`: Maximum wait time if queue is full
- `priority`: Processing priority (LOW, NORMAL, HIGH, CRITICAL)

**Returns:**
- `Future[U]`: Future object containing the result

**Example:**
```python
# High-priority inference request
future = processor.enqueue_predict(
    data=numpy_array, 
    priority=BatchPriority.HIGH,
    timeout=30.0
)
result = future.result()  # Blocks until result is ready
```

```python
def enqueue(
    self, 
    item: T, 
    timeout: Optional[float] = None, 
    priority: BatchPriority = BatchPriority.NORMAL
) -> None
```
Fire-and-forget processing (no result tracking).

##### Monitoring & Control
```python
def get_stats() -> Dict[str, Any]
```
Get comprehensive processing statistics.

**Returns:**
```python
{
    "avg_processing_time": 0.015,      # Average batch processing time
    "avg_batch_size": 45.2,            # Average batch size
    "avg_queue_time": 0.008,           # Average time in queue
    "p95_latency": 0.025,              # 95th percentile latency
    "p99_latency": 0.040,              # 99th percentile latency
    "throughput": 2840.5,              # Items per second
    "error_rate": 0.001,               # Error percentage
    "current_batch_size": 48,          # Current adaptive batch size
    "system_load": 0.65,               # System load estimate (0-1)
    "active_batches": 2,               # Currently processing batches
    "queue_size": 127,                 # Current queue length
    "is_paused": False,                # Pause status
    "is_stopping": False               # Shutdown status
}
```

```python
def update_batch_size(self, new_size: int) -> None
```
Manually override the current batch size.

#### **Advanced Features**

##### Adaptive Batch Sizing
The processor automatically adjusts batch size based on:
- System load and processing times
- Memory usage patterns
- Queue length dynamics
- Error rates and retry patterns

```python
# System automatically adjusts batch size:
# High load (slow processing) → Smaller batches
# Low load (fast processing) → Larger batches  
# High memory usage → Smaller batches
# Queue backing up → Larger batches
```

##### Memory Optimization
```python
# Automatic memory management features:
# - Pre-allocated arrays for numpy batching
# - Garbage collection at configurable thresholds
# - Memory usage monitoring and alerts
# - Automatic batch size reduction on memory pressure
```

##### Health Monitoring
```python
# Background health monitoring checks:
# - Memory usage tracking
# - Queue growth monitoring  
# - Stuck batch detection
# - System resource utilization
```

##### Error Handling & Recovery
```python
# Comprehensive error handling:
# - Automatic retries with exponential backoff
# - Graceful degradation (smaller batches on errors)
# - Individual item error isolation
# - Detailed error logging and metrics
```

### `BatchStats`
```python
class BatchStats:
```

Thread-safe statistics collection with performance optimizations.

#### **Constructor**
```python
def __init__(self, window_size: int)
```

**Parameters:**
- `window_size`: Number of recent measurements to keep

#### **Methods**
```python
def update(
    self, 
    processing_time: float, 
    batch_size: int, 
    queue_time: float, 
    latency: float = 0.0, 
    queue_length: int = 0
) -> None
```
Update statistics with batch processing information.

```python
def record_error(self, count: int = 1) -> None
```
Record processing errors for error rate calculation.

```python
def get_stats() -> Dict[str, float]
```
Get cached statistics dictionary with performance metrics.

---

## Architecture & Design

### System Architecture
```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Client Code   │───▶│    Queue     │───▶│ Batch Processor │
│                 │    │   System     │    │    Worker       │
│ • enqueue_predict│    │              │    │                 │
│ • enqueue       │    │ • Priority   │    │ • Adaptive      │
│ • get results   │    │ • Memory     │    │ • Monitoring    │
└─────────────────┘    │ • Backpres.  │    │ • Health Check  │
                       └──────────────┘    └─────────────────┘
                              │                       │
                              ▼                       ▼
                       ┌──────────────┐    ┌─────────────────┐
                       │   Priority   │    │   Processing    │
                       │   Handler    │    │    Engine       │
                       │              │    │                 │
                       │ • HIGH       │    │ • NumPy Batch   │
                       │ • NORMAL     │    │ • Generic Items │
                       │ • LOW        │    │ • Thread Pool   │
                       │ • CRITICAL   │    │ • Memory Mgmt   │
                       └──────────────┘    └─────────────────┘
```

### Data Flow Pipeline

1. **Request Ingestion**
   - Items enqueued with priority levels
   - Queue capacity and backpressure management
   - Memory-based batching decisions

2. **Adaptive Batch Formation**
   - Dynamic size adjustment based on system load
   - Memory constraints and optimization
   - Timeout-based batch completion

3. **Intelligent Processing**
   - NumPy array vectorization for ML workloads
   - Generic item processing via thread pools
   - Retry logic with exponential backoff

4. **Result Distribution**
   - Future-based asynchronous result delivery
   - Error propagation and handling
   - Performance metrics collection

### Performance Optimizations

#### Memory Management
- **Pre-allocated Arrays**: For NumPy batching to avoid allocation overhead
- **Memory Monitoring**: Real-time memory usage tracking with psutil
- **Adaptive GC**: Garbage collection triggered at configurable thresholds
- **Memory Limits**: Per-batch memory constraints to prevent OOM

#### Concurrency & Threading
- **Thread Pool**: Optimized worker threads for generic processing
- **Lock-free Operations**: Minimal locking for high-throughput scenarios
- **Async Futures**: Non-blocking result delivery
- **Queue Optimization**: Priority queues with efficient insertion/removal

#### Adaptive Intelligence
- **System Load Monitoring**: Real-time load estimation and batch size adjustment
- **Queue Length Tracking**: Dynamic batching based on queue backlog
- **Error-based Adaptation**: Automatic batch size reduction on failures
- **Memory-aware Batching**: Memory usage influences batch formation

---

## Production Examples

### ML Model Inference Server
```python
import numpy as np
from modules.engine.batch_processor import BatchProcessor
from modules.configs import BatchProcessorConfig, BatchPriority

class MLInferenceServer:
    def __init__(self, model):
        self.model = model
        
        # Production-ready configuration
        config = BatchProcessorConfig(
            initial_batch_size=64,
            max_batch_size=256,
            batch_timeout=0.05,  # 50ms max latency
            enable_priority_queue=True,
            enable_adaptive_batching=True,
            enable_health_monitoring=True,
            max_batch_memory_mb=512,
            memory_warning_threshold=75.0,
            queue_warning_threshold=200
        )
        
        self.processor = BatchProcessor(config)
        self.processor.start(self._batch_inference)
    
    def _batch_inference(self, batch_data):
        """Process a batch of inference requests"""
        return self.model.predict(batch_data)
    
    async def predict(self, data, priority=BatchPriority.NORMAL):
        """Async inference endpoint"""
        future = self.processor.enqueue_predict(data, priority=priority)
        return future.result()
    
    def get_health_status(self):
        """Health check endpoint"""
        stats = self.processor.get_stats()
        return {
            "status": "healthy" if stats.get("error_rate", 0) < 0.01 else "degraded",
            "throughput": stats.get("throughput", 0),
            "latency_p95": stats.get("p95_latency", 0),
            "queue_size": stats.get("queue_size", 0),
            "batch_size": stats.get("current_batch_size", 0)
        }

# Usage
server = MLInferenceServer(your_ml_model)

# High-priority request
result = await server.predict(urgent_data, BatchPriority.HIGH)

# Health monitoring
health = server.get_health_status()
```

### Real-time Data Processing Pipeline
```python
from concurrent.futures import as_completed
import time

class DataProcessingPipeline:
    def __init__(self):
        config = BatchProcessorConfig(
            initial_batch_size=100,
            max_batch_size=500,
            enable_adaptive_batching=True,
            enable_monitoring=True,
            processing_strategy=BatchProcessingStrategy.ADAPTIVE
        )
        
        self.processor = BatchProcessor(config)
        self.processor.start(self._process_data_batch)
    
    def _process_data_batch(self, batch):
        """Your data processing logic here"""
        # Example: Feature extraction, transformation, etc.
        return np.array([self._transform_item(item) for item in batch])
    
    def _transform_item(self, item):
        # Your transformation logic
        return item * 2  # Simplified example
    
    def process_stream(self, data_stream):
        """Process continuous data stream"""
        futures = []
        
        for data_item in data_stream:
            future = self.processor.enqueue_predict(data_item)
            futures.append(future)
            
            # Process completed results
            if len(futures) > 100:  # Batch result collection
                completed = [f for f in futures if f.done()]
                for future in completed:
                    try:
                        result = future.result()
                        yield result
                    except Exception as e:
                        print(f"Processing error: {e}")
                
                # Remove completed futures
                futures = [f for f in futures if not f.done()]
        
        # Process remaining futures
        for future in as_completed(futures):
            try:
                yield future.result()
            except Exception as e:
                print(f"Processing error: {e}")

# Usage
pipeline = DataProcessingPipeline()
for result in pipeline.process_stream(your_data_stream):
    handle_result(result)
```

### Load Testing and Benchmarking
```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

class BatchProcessorBenchmark:
    def __init__(self):
        self.results = []
    
    async def benchmark_latency(self, processor, num_requests=1000):
        """Benchmark request latency"""
        start_time = time.time()
        
        # Submit all requests
        futures = []
        for i in range(num_requests):
            data = np.random.rand(10)  # Sample data
            future = processor.enqueue_predict(data)
            futures.append((time.time(), future))
        
        # Collect results and measure latency
        latencies = []
        for submit_time, future in futures:
            result = await asyncio.wrap_future(future)
            latency = time.time() - submit_time
            latencies.append(latency)
        
        total_time = time.time() - start_time
        
        return {
            "total_time": total_time,
            "throughput": num_requests / total_time,
            "avg_latency": statistics.mean(latencies),
            "p95_latency": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
            "p99_latency": statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        }
    
    def benchmark_throughput(self, processor, duration_seconds=60):
        """Benchmark maximum throughput"""
        end_time = time.time() + duration_seconds
        processed_count = 0
        
        while time.time() < end_time:
            # Submit batch of requests
            futures = []
            for _ in range(100):  # Submit 100 at a time
                data = np.random.rand(10)
                future = processor.enqueue_predict(data)
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                future.result()
                processed_count += 1
        
        return processed_count / duration_seconds

# Run benchmarks
def run_benchmark():
    config = BatchProcessorConfig(
        initial_batch_size=32,
        max_batch_size=128,
        enable_adaptive_batching=True
    )
    
    processor = BatchProcessor(config)
    processor.start(lambda x: x * 2)  # Simple processing function
    
    benchmark = BatchProcessorBenchmark()
    
    # Latency benchmark
    latency_results = asyncio.run(benchmark.benchmark_latency(processor))
    print(f"Latency Results: {latency_results}")
    
    # Throughput benchmark
    throughput = benchmark.benchmark_throughput(processor)
    print(f"Throughput: {throughput:.2f} requests/second")
    
    processor.stop()

if __name__ == "__main__":
    run_benchmark()
```

---

## Testing & Validation

### Unit Testing with pytest
```python
import pytest
import numpy as np
from modules.engine.batch_processor import BatchProcessor
from modules.configs import BatchProcessorConfig, BatchPriority, BatchProcessingStrategy

class TestBatchProcessor:
    @pytest.fixture
    def basic_config(self):
        return BatchProcessorConfig(
            initial_batch_size=4,
            max_batch_size=16,
            batch_timeout=0.1,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def processor(self, basic_config):
        processor = BatchProcessor(basic_config)
        yield processor
        processor.stop()
    
    def test_basic_processing(self, processor):
        """Test basic batch processing functionality"""
        def double_func(batch):
            return batch * 2
        
        processor.start(double_func)
        
        # Submit test data
        test_data = np.array([1, 2, 3])
        future = processor.enqueue_predict(test_data)
        result = future.result()
        
        np.testing.assert_array_equal(result, np.array([2, 4, 6]))
    
    def test_priority_processing(self):
        """Test priority-based processing"""
        config = BatchProcessorConfig(
            initial_batch_size=2,
            max_batch_size=4,
            enable_priority_queue=True
        )
        
        processor = BatchProcessor(config)
        processor.start(lambda x: x)
        
        try:
            # Submit normal and high priority items
            normal_future = processor.enqueue_predict(
                np.array([1]), priority=BatchPriority.NORMAL
            )
            high_future = processor.enqueue_predict(
                np.array([2]), priority=BatchPriority.HIGH
            )
            
            # High priority should be processed first
            # (timing dependent, but generally true)
            high_result = high_future.result()
            normal_result = normal_future.result()
            
            assert high_result[0] == 2
            assert normal_result[0] == 1
        finally:
            processor.stop()
    
    def test_adaptive_batching(self):
        """Test adaptive batch size adjustment"""
        config = BatchProcessorConfig(
            initial_batch_size=4,
            max_batch_size=16,
            enable_adaptive_batching=True,
            processing_strategy=BatchProcessingStrategy.ADAPTIVE
        )
        
        processor = BatchProcessor(config)
        
        # Slow processing function to trigger adaptation
        def slow_func(batch):
            import time
            time.sleep(0.1)  # Simulate slow processing
            return batch
        
        processor.start(slow_func)
        
        try:
            initial_size = processor._current_batch_size
            
            # Submit several batches to trigger adaptation
            futures = []
            for _ in range(10):
                future = processor.enqueue_predict(np.array([1]))
                futures.append(future)
            
            # Wait for processing
            for future in futures:
                future.result()
            
            # Batch size should adapt (likely decrease due to slow processing)
            # This is a timing-dependent test, so we just check it's reasonable
            final_size = processor._current_batch_size
            assert 1 <= final_size <= 16
        finally:
            processor.stop()
    
    def test_error_handling(self, processor):
        """Test error handling and recovery"""
        def error_func(batch):
            if len(batch) > 2:
                raise ValueError("Batch too large")
            return batch
        
        processor.start(error_func)
        
        # Small batch should work
        small_future = processor.enqueue_predict(np.array([1]))
        result = small_future.result()
        assert result[0] == 1
        
        # Large batch should fail
        large_future = processor.enqueue_predict(np.array([1, 2, 3, 4]))
        with pytest.raises(ValueError):
            large_future.result()
    
    def test_statistics_collection(self, processor):
        """Test statistics and monitoring"""
        processor.start(lambda x: x)
        
        # Process some items
        futures = []
        for i in range(5):
            future = processor.enqueue_predict(np.array([i]))
            futures.append(future)
        
        # Wait for completion
        for future in futures:
            future.result()
        
        # Check statistics
        stats = processor.get_stats()
        assert "avg_processing_time" in stats
        assert "throughput" in stats
        assert "current_batch_size" in stats
        assert stats["throughput"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, processor):
        """Test concurrent request handling"""
        import asyncio
        
        processor.start(lambda x: x * 2)
        
        async def submit_request(value):
            future = processor.enqueue_predict(np.array([value]))
            return await asyncio.wrap_future(future)
        
        # Submit concurrent requests
        tasks = [submit_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify all results
        for i, result in enumerate(results):
            assert result[0] == i * 2

# Run tests with: pytest test_batch_processor.py -v
```

### Integration Testing
```python
import pytest
import numpy as np
from modules.engine.batch_processor import BatchProcessor
from modules.configs import BatchProcessorConfig

def test_ml_pipeline_integration():
    """Integration test simulating ML pipeline"""
    
    # Mock ML model
    class MockMLModel:
        def predict(self, X):
            # Simulate model inference
            return X.sum(axis=1, keepdims=True)
    
    model = MockMLModel()
    
    # Configure processor for ML workload
    config = BatchProcessorConfig(
        initial_batch_size=32,
        max_batch_size=128,
        enable_adaptive_batching=True,
        enable_monitoring=True,
        max_batch_memory_mb=256
    )
    
    processor = BatchProcessor(config)
    processor.start(model.predict)
    
    try:
        # Simulate real ML inference requests
        test_data = [
            np.random.rand(10) for _ in range(100)
        ]
        
        futures = []
        for data in test_data:
            future = processor.enqueue_predict(data)
            futures.append(future)
        
        # Collect results
        results = [future.result() for future in futures]
        
        # Verify results
        assert len(results) == 100
        for i, result in enumerate(results):
            expected = test_data[i].sum()
            assert abs(result[0] - expected) < 1e-6
        
        # Check performance metrics
        stats = processor.get_stats()
        assert stats["throughput"] > 0
        assert stats["error_rate"] == 0
        
    finally:
        processor.stop()

def test_load_and_stress():
    """Stress test under high load"""
    config = BatchProcessorConfig(
        initial_batch_size=16,
        max_batch_size=64,
        enable_adaptive_batching=True
    )
    
    processor = BatchProcessor(config)
    processor.start(lambda x: x)
    
    try:
        # High load test
        futures = []
        for _ in range(1000):
            data = np.random.rand(5)
            future = processor.enqueue_predict(data)
            futures.append(future)
        
        # Wait for all to complete
        for future in futures:
            future.result()
        
        stats = processor.get_stats()
        print(f"Processed 1000 requests with throughput: {stats['throughput']:.2f}/s")
        
    finally:
        processor.stop()
```

---

## Performance Tuning Guide

### Configuration Optimization

#### High-Throughput Scenarios
```python
high_throughput_config = BatchProcessorConfig(
    initial_batch_size=128,
    max_batch_size=512,
    batch_timeout=0.02,  # Lower timeout for faster batching
    enable_adaptive_batching=True,
    max_workers=8,  # More workers
    enable_priority_queue=False,  # Skip priority overhead
    max_batch_memory_mb=1024
)
```

#### Low-Latency Requirements
```python
low_latency_config = BatchProcessorConfig(
    initial_batch_size=8,
    max_batch_size=32,
    batch_timeout=0.005,  # Very low timeout
    enable_adaptive_batching=True,
    min_batch_interval=0.001,
    enable_health_monitoring=True
)
```

#### Memory-Constrained Environments
```python
memory_optimized_config = BatchProcessorConfig(
    initial_batch_size=16,
    max_batch_size=64,
    max_batch_memory_mb=128,  # Strict memory limit
    enable_memory_optimization=True,
    gc_batch_threshold=32,
    memory_warning_threshold=70.0
)
```

### Monitoring & Alerting
```python
def setup_monitoring(processor):
    """Setup comprehensive monitoring"""
    
    def check_health():
        stats = processor.get_stats()
        
        # Performance alerts
        if stats.get("p95_latency", 0) > 0.5:  # 500ms
            alert("High latency detected")
        
        if stats.get("error_rate", 0) > 0.05:  # 5%
            alert("High error rate")
        
        if stats.get("queue_size", 0) > 500:
            alert("Queue backing up")
        
        # Log metrics
        logger.info(f"Throughput: {stats.get('throughput', 0):.2f}/s, "
                   f"Latency P95: {stats.get('p95_latency', 0)*1000:.2f}ms")
    
    # Setup periodic health checks
    import threading
    import time
    
    def monitor_loop():
        while not processor.shutdown_event.is_set():
            try:
                check_health()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    return monitor_thread
```

---

## Security & Production Considerations

### Resource Management
- **Memory Limits**: Configure `max_batch_memory_mb` to prevent OOM conditions
- **Queue Size**: Set `max_queue_size` to avoid unbounded memory growth
- **Thread Limits**: Configure `max_workers` based on system capacity
- **Timeout Controls**: Set appropriate timeouts to prevent hanging requests

### Error Handling
- **Graceful Degradation**: Automatic batch size reduction on errors
- **Retry Logic**: Configurable retry attempts with exponential backoff
- **Circuit Breaking**: Built-in error rate monitoring
- **Isolation**: Individual request failures don't affect the batch

### Monitoring & Observability
- **Health Checks**: Background system health monitoring
- **Metrics Collection**: Comprehensive performance statistics
- **Logging**: Detailed debug and operational logging
- **Alerting**: Configurable thresholds for alerts

### Deployment Best Practices
1. **Start Small**: Begin with conservative batch sizes and adjust based on metrics
2. **Monitor Closely**: Watch memory usage, latency, and error rates
3. **Load Test**: Validate performance under expected production load
4. **Gradual Rollout**: Deploy with monitoring and rollback capabilities
5. **Resource Planning**: Ensure adequate CPU, memory, and I/O capacity

---

## Troubleshooting

### Common Issues

#### High Memory Usage
```python
# Solutions:
# 1. Reduce max_batch_size
# 2. Enable memory optimization
# 3. Lower max_batch_memory_mb
# 4. Increase gc_batch_threshold frequency

config.max_batch_size = 32  # Reduce from default
config.enable_memory_optimization = True
config.max_batch_memory_mb = 256
config.gc_batch_threshold = 16
```

#### High Latency
```python
# Solutions:
# 1. Reduce batch_timeout
# 2. Optimize processing function
# 3. Enable adaptive batching
# 4. Check system load

config.batch_timeout = 0.01  # Faster batching
config.enable_adaptive_batching = True
```

#### Queue Backup
```python
# Solutions:
# 1. Increase max_batch_size
# 2. Add more workers
# 3. Optimize processing function
# 4. Enable priority queuing for urgent requests

config.max_batch_size = 256
config.max_workers = 8
```

### Debug Mode
```python
# Enable detailed logging
config = BatchProcessorConfig(
    debug_mode=True,  # Verbose logging
    enable_monitoring=True,
    enable_health_monitoring=True
)

# Check logs for detailed information
import logging
logging.basicConfig(level=logging.DEBUG)
```

> **Last Updated**: June 27, 2025 (v0.1.4)  
> **Compatibility**: Python 3.10+, NumPy 1.20+  
> **Status**: Production Ready ✅