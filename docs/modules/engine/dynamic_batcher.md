# Dynamic Batcher (`modules/engine/dynamic_batcher.py`)

## Overview

The Dynamic Batcher provides an advanced dynamic batching system with priority queues and adaptive sizing for optimal hardware utilization. It intelligently batches requests to maximize throughput while minimizing latency in machine learning inference workloads.

## Features

- **Priority-Based Queuing**: Multi-level priority queues for request scheduling
- **Adaptive Batch Sizing**: Dynamic optimization of batch sizes based on performance
- **Intelligent Scheduling**: Smart request grouping and timing
- **Hardware Optimization**: GPU/CPU utilization optimization
- **Latency Management**: Configurable timeout and priority handling
- **Performance Monitoring**: Real-time metrics and optimization feedback

## Core Classes

### DynamicBatcher

Main dynamic batching class with adaptive capabilities:

```python
class DynamicBatcher:
    def __init__(
        self,
        batch_processor: Callable,
        max_batch_size: int = 64,
        max_wait_time_ms: float = 10.0,
        max_queue_size: int = 1000,
        enable_adaptive_sizing: bool = True,
        enable_priority_queues: bool = True,
        hardware_type: str = "auto"
    )
```

**Parameters:**
- `batch_processor`: Function to process batches of requests
- `max_batch_size`: Maximum number of requests per batch
- `max_wait_time_ms`: Maximum wait time before processing batch
- `max_queue_size`: Maximum total queue size across all priorities
- `enable_adaptive_sizing`: Enable dynamic batch size optimization
- `enable_priority_queues`: Enable priority-based request scheduling
- `hardware_type`: Target hardware ("gpu", "cpu", "auto")

## Usage Examples

### Basic Dynamic Batching

```python
from modules.engine.dynamic_batcher import DynamicBatcher
from modules.engine.prediction_request import PredictionRequest
import numpy as np
import time

# Define batch processing function
def process_batch(requests):
    """Process a batch of prediction requests"""
    # Extract input data from requests
    inputs = [req.data for req in requests]
    batch_data = np.vstack(inputs)
    
    # Simulate model inference
    time.sleep(0.01)  # 10ms processing time
    predictions = np.random.rand(len(inputs), 10)
    
    # Return results for each request
    results = []
    for i, req in enumerate(requests):
        results.append({
            'request_id': req.request_id,
            'prediction': predictions[i],
            'processing_time': 0.01
        })
    return results

# Initialize dynamic batcher
batcher = DynamicBatcher(
    batch_processor=process_batch,
    max_batch_size=32,
    max_wait_time_ms=15.0,
    enable_adaptive_sizing=True
)

# Start the batcher
batcher.start()

# Submit requests
for i in range(100):
    request = PredictionRequest(
        request_id=f"req_{i}",
        data=np.random.rand(1, 20),
        priority=1,  # Normal priority
        timeout_ms=1000
    )
    
    # Submit request and get future result
    future = batcher.submit_request(request)
    
    # Get result (non-blocking with timeout)
    try:
        result = future.result(timeout=2.0)
        print(f"Request {i}: {result['request_id']} completed")
    except TimeoutError:
        print(f"Request {i} timed out")

# Shutdown batcher
batcher.shutdown()
```

### Priority-Based Request Handling

```python
from modules.engine.dynamic_batcher import DynamicBatcher, RequestPriority
import threading
import time

# Initialize batcher with priority queues
batcher = DynamicBatcher(
    batch_processor=process_batch,
    max_batch_size=64,
    max_wait_time_ms=20.0,
    enable_priority_queues=True
)

batcher.start()

# Submit requests with different priorities
def submit_priority_requests():
    # Critical priority requests (processed first)
    for i in range(10):
        request = PredictionRequest(
            request_id=f"critical_{i}",
            data=np.random.rand(1, 20),
            priority=RequestPriority.CRITICAL,
            timeout_ms=500
        )
        batcher.submit_request(request)
    
    # Normal priority requests
    for i in range(50):
        request = PredictionRequest(
            request_id=f"normal_{i}",
            data=np.random.rand(1, 20),
            priority=RequestPriority.NORMAL,
            timeout_ms=2000
        )
        batcher.submit_request(request)
    
    # Low priority requests (background processing)
    for i in range(20):
        request = PredictionRequest(
            request_id=f"low_{i}",
            data=np.random.rand(1, 20),
            priority=RequestPriority.LOW,
            timeout_ms=5000
        )
        batcher.submit_request(request)

# Submit requests from multiple threads
threads = []
for i in range(3):
    thread = threading.Thread(target=submit_priority_requests)
    thread.start()
    threads.append(thread)

# Wait for completion
for thread in threads:
    thread.join()

# Check batcher statistics
stats = batcher.get_statistics()
print(f"Total processed: {stats['total_processed']}")
print(f"Average batch size: {stats['avg_batch_size']:.1f}")
print(f"Average wait time: {stats['avg_wait_time']:.2f}ms")
print(f"Priority distribution: {stats['priority_distribution']}")

batcher.shutdown()
```

### Adaptive Batch Sizing

```python
from modules.engine.dynamic_batcher import DynamicBatcher
import matplotlib.pyplot as plt

# Configure adaptive batcher for performance optimization
batcher = DynamicBatcher(
    batch_processor=process_batch,
    max_batch_size=128,
    max_wait_time_ms=25.0,
    enable_adaptive_sizing=True,
    adaptation_interval=5.0,  # Adapt every 5 seconds
    optimization_target="throughput"  # or "latency"
)

# Monitor adaptive behavior
def monitor_adaptation():
    adaptation_history = []
    
    for minute in range(10):  # Monitor for 10 minutes
        time.sleep(60)  # Wait 1 minute
        
        stats = batcher.get_adaptation_stats()
        adaptation_history.append({
            'minute': minute,
            'optimal_batch_size': stats['current_optimal_batch_size'],
            'throughput': stats['current_throughput'],
            'avg_latency': stats['avg_latency'],
            'adaptation_score': stats['adaptation_score']
        })
        
        print(f"Minute {minute}: Optimal batch size = {stats['current_optimal_batch_size']}")
    
    return adaptation_history

batcher.start()

# Start monitoring in background
monitor_thread = threading.Thread(target=monitor_adaptation)
monitor_thread.start()

# Simulate varying load patterns
def simulate_load_patterns():
    # Phase 1: Low load
    for i in range(100):
        request = PredictionRequest(f"low_load_{i}", np.random.rand(1, 20))
        batcher.submit_request(request)
        time.sleep(0.1)  # 100ms between requests
    
    # Phase 2: High burst load
    for i in range(500):
        request = PredictionRequest(f"burst_{i}", np.random.rand(1, 20))
        batcher.submit_request(request)
        time.sleep(0.01)  # 10ms between requests
    
    # Phase 3: Medium sustained load
    for i in range(300):
        request = PredictionRequest(f"sustained_{i}", np.random.rand(1, 20))
        batcher.submit_request(request)
        time.sleep(0.05)  # 50ms between requests

# Run load simulation
load_thread = threading.Thread(target=simulate_load_patterns)
load_thread.start()

# Wait for completion
load_thread.join()
monitor_thread.join()

# Plot adaptation results
final_stats = batcher.get_comprehensive_stats()
batcher.shutdown()
```

## Advanced Features

### Hardware-Aware Batching

```python
# GPU-optimized configuration
gpu_batcher = DynamicBatcher(
    batch_processor=gpu_process_batch,
    max_batch_size=256,  # Larger batches for GPU
    max_wait_time_ms=30.0,  # Longer wait for more batching
    hardware_type="gpu",
    enable_tensor_batching=True,
    memory_limit_mb=8192
)

# CPU-optimized configuration
cpu_batcher = DynamicBatcher(
    batch_processor=cpu_process_batch,
    max_batch_size=32,  # Smaller batches for CPU
    max_wait_time_ms=10.0,  # Shorter wait for responsiveness
    hardware_type="cpu",
    enable_multiprocessing=True,
    num_workers=4
)

# Automatic hardware detection and optimization
auto_batcher = DynamicBatcher(
    batch_processor=auto_process_batch,
    hardware_type="auto",  # Automatically detect and optimize
    auto_tune=True,
    tune_duration=300  # 5-minute tuning period
)
```

### Custom Batching Strategies

```python
from modules.engine.dynamic_batcher import BatchingStrategy

class LatencyOptimizedStrategy(BatchingStrategy):
    """Custom strategy optimizing for low latency"""
    
    def should_process_batch(self, queue_state):
        # Process immediately if any critical priority requests
        if queue_state.has_critical_requests():
            return True
        
        # Process if wait time exceeds threshold
        if queue_state.max_wait_time > self.max_wait_threshold:
            return True
        
        # Process if batch is reasonably sized
        if queue_state.total_requests >= self.min_batch_size:
            return True
        
        return False
    
    def select_batch(self, priority_queues):
        # Always prioritize critical requests
        batch = []
        
        # First, add all critical requests
        batch.extend(priority_queues[RequestPriority.CRITICAL])
        
        # Fill remaining space with high priority
        remaining_space = self.max_batch_size - len(batch)
        batch.extend(priority_queues[RequestPriority.HIGH][:remaining_space])
        
        # Fill any remaining space with normal priority
        remaining_space = self.max_batch_size - len(batch)
        batch.extend(priority_queues[RequestPriority.NORMAL][:remaining_space])
        
        return batch

# Use custom strategy
batcher = DynamicBatcher(
    batch_processor=process_batch,
    batching_strategy=LatencyOptimizedStrategy(
        max_wait_threshold=5.0,
        min_batch_size=8
    )
)
```

### Request Deduplication

```python
from modules.engine.dynamic_batcher import DynamicBatcher

# Enable request deduplication for identical inputs
batcher = DynamicBatcher(
    batch_processor=process_batch,
    enable_deduplication=True,
    deduplication_cache_size=1000,
    deduplication_ttl=60  # 60 seconds TTL
)

# Submit duplicate requests
data = np.random.rand(1, 20)
for i in range(10):
    # Same data, different request IDs
    request = PredictionRequest(
        request_id=f"dup_{i}",
        data=data,  # Identical data
        enable_deduplication=True
    )
    future = batcher.submit_request(request)

# Only one actual computation will be performed
# All requests will receive the same result
```

## Performance Optimization

### Batch Size Optimization

```python
class BatchSizeOptimizer:
    def __init__(self, batcher):
        self.batcher = batcher
        self.performance_history = deque(maxlen=100)
        
    def optimize_batch_size(self):
        """Automatically optimize batch size based on performance"""
        current_stats = self.batcher.get_recent_performance()
        
        # Calculate performance metrics
        throughput = current_stats['throughput']
        latency = current_stats['avg_latency']
        gpu_utilization = current_stats['gpu_utilization']
        
        # Performance score combining throughput and latency
        score = throughput / (1 + latency * 0.1)
        
        self.performance_history.append({
            'batch_size': self.batcher.current_optimal_batch_size,
            'score': score,
            'throughput': throughput,
            'latency': latency
        })
        
        if len(self.performance_history) >= 10:
            # Find optimal batch size from recent history
            best_config = max(self.performance_history, key=lambda x: x['score'])
            optimal_size = best_config['batch_size']
            
            # Adjust current batch size towards optimal
            current_size = self.batcher.current_optimal_batch_size
            if optimal_size > current_size:
                new_size = min(current_size + 4, self.batcher.max_batch_size)
            elif optimal_size < current_size:
                new_size = max(current_size - 4, 1)
            else:
                new_size = current_size
                
            self.batcher.set_optimal_batch_size(new_size)
            return new_size
        
        return self.batcher.current_optimal_batch_size

# Use optimizer
optimizer = BatchSizeOptimizer(batcher)

# Periodic optimization
def optimization_loop():
    while batcher.is_running():
        optimal_size = optimizer.optimize_batch_size()
        print(f"Optimal batch size updated to: {optimal_size}")
        time.sleep(30)  # Optimize every 30 seconds

optimization_thread = threading.Thread(target=optimization_loop)
optimization_thread.daemon = True
optimization_thread.start()
```

### Memory Management

```python
from modules.engine.dynamic_batcher import MemoryAwareBatcher

# Memory-aware batching with automatic size adjustment
memory_batcher = MemoryAwareBatcher(
    batch_processor=process_batch,
    max_memory_usage_mb=4096,  # 4GB limit
    memory_monitoring_interval=5.0,  # Check every 5 seconds
    enable_memory_profiling=True
)

def memory_intensive_processor(requests):
    """Processor that uses significant memory"""
    # Large intermediate computations
    batch_size = len(requests)
    intermediate_data = np.random.rand(batch_size, 10000, 1000)  # Large arrays
    
    # Simulate processing
    results = []
    for i, req in enumerate(requests):
        # Process with large memory footprint
        result = np.mean(intermediate_data[i])
        results.append({
            'request_id': req.request_id,
            'result': result
        })
    
    # Memory cleanup
    del intermediate_data
    
    return results

# The batcher will automatically reduce batch sizes if memory usage is high
memory_batcher.set_processor(memory_intensive_processor)
memory_batcher.start()
```

## Monitoring and Diagnostics

### Real-time Monitoring

```python
from modules.engine.dynamic_batcher import BatcherMonitor
import json

class RealTimeMonitor:
    def __init__(self, batcher):
        self.batcher = batcher
        self.monitor = BatcherMonitor(batcher)
        
    def start_monitoring(self):
        """Start real-time monitoring dashboard"""
        while self.batcher.is_running():
            stats = self.monitor.get_real_time_stats()
            
            dashboard_data = {
                'timestamp': time.time(),
                'queue_status': {
                    'total_requests': stats['total_queued'],
                    'priority_breakdown': stats['priority_distribution'],
                    'average_wait_time': stats['avg_wait_time']
                },
                'processing': {
                    'current_batch_size': stats['current_batch_size'],
                    'optimal_batch_size': stats['optimal_batch_size'],
                    'throughput': stats['throughput'],
                    'latency_p95': stats['latency_p95']
                },
                'performance': {
                    'cpu_usage': stats['cpu_usage'],
                    'memory_usage': stats['memory_usage'],
                    'gpu_utilization': stats.get('gpu_utilization', 0)
                },
                'health': {
                    'error_rate': stats['error_rate'],
                    'timeout_rate': stats['timeout_rate'],
                    'adaptation_score': stats['adaptation_score']
                }
            }
            
            # Output to monitoring system
            print(json.dumps(dashboard_data, indent=2))
            
            # Export to external monitoring
            self.export_to_prometheus(dashboard_data)
            
            time.sleep(10)  # Update every 10 seconds
    
    def export_to_prometheus(self, data):
        """Export metrics to Prometheus"""
        # Implementation for Prometheus metrics export
        pass

# Start monitoring
monitor = RealTimeMonitor(batcher)
monitor_thread = threading.Thread(target=monitor.start_monitoring)
monitor_thread.daemon = True
monitor_thread.start()
```

### Performance Profiling

```python
from modules.engine.dynamic_batcher import BatcherProfiler

# Enable detailed profiling
profiler = BatcherProfiler(batcher, enable_detailed_timing=True)

# Profile batch processing performance
with profiler:
    # Run workload
    for i in range(1000):
        request = PredictionRequest(f"profile_{i}", np.random.rand(1, 20))
        batcher.submit_request(request)

# Get profiling results
profile_results = profiler.get_results()
print("Profiling Results:")
print(f"  Total requests: {profile_results['total_requests']}")
print(f"  Average batch size: {profile_results['avg_batch_size']:.1f}")
print(f"  Time breakdown:")
print(f"    Queue wait: {profile_results['queue_wait_pct']:.1f}%")
print(f"    Batch formation: {profile_results['batch_formation_pct']:.1f}%")
print(f"    Processing: {profile_results['processing_pct']:.1f}%")
print(f"    Result distribution: {profile_results['result_dist_pct']:.1f}%")

# Generate performance report
profiler.generate_report("batch_performance_report.html")
```

## Best Practices

### 1. Batch Size Selection
```python
# Start with hardware-appropriate defaults
if hardware_type == "gpu":
    initial_batch_size = 128  # Larger for GPU
elif hardware_type == "cpu":
    initial_batch_size = 32   # Smaller for CPU
else:
    initial_batch_size = 64   # Balanced default

batcher = DynamicBatcher(
    batch_processor=process_batch,
    max_batch_size=initial_batch_size * 2,  # Allow growth
    enable_adaptive_sizing=True
)
```

### 2. Priority Management
```python
# Set appropriate priorities based on use case
def get_request_priority(request_metadata):
    if request_metadata.get('user_type') == 'premium':
        return RequestPriority.HIGH
    elif request_metadata.get('real_time') == True:
        return RequestPriority.CRITICAL
    elif request_metadata.get('background') == True:
        return RequestPriority.LOW
    else:
        return RequestPriority.NORMAL
```

### 3. Error Handling
```python
def robust_batch_processor(requests):
    """Robust batch processor with error handling"""
    results = []
    
    try:
        # Process successful requests
        batch_results = model.predict_batch([req.data for req in requests])
        
        for i, req in enumerate(requests):
            results.append({
                'request_id': req.request_id,
                'result': batch_results[i],
                'status': 'success'
            })
            
    except Exception as e:
        # Handle batch-level errors
        logging.error(f"Batch processing failed: {e}")
        
        # Return error results for all requests
        for req in requests:
            results.append({
                'request_id': req.request_id,
                'error': str(e),
                'status': 'error'
            })
    
    return results
```

## Related Documentation

- [Batch Processor Documentation](batch_processor.md)
- [Batch Statistics Documentation](batch_stats.md)
- [Prediction Request Documentation](prediction_request.md)
- [Inference Engine Documentation](inference_engine.md)
- [Performance Metrics Documentation](performance_metrics.md)
