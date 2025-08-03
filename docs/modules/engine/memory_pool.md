# Memory Pool (`modules/engine/memory_pool.py`)

## Overview

The Memory Pool module provides NUMA-aware memory buffer pooling with intelligent management for optimal memory locality and reduced allocation overhead. It's designed to minimize memory allocation/deallocation costs in high-performance ML workloads.

## Features

- **NUMA-Aware Allocation**: Memory allocation optimized for NUMA topology
- **Hierarchical Buffer Pools**: Size-based buffer categorization (small/medium/large)
- **Zero-Copy Operations**: Efficient buffer reuse without copying
- **Thread-Safe Operations**: Concurrent access support
- **Memory Pressure Handling**: Automatic cleanup under memory pressure
- **Performance Monitoring**: Memory usage and allocation statistics

## Core Classes

### MemoryPool

Main memory pool manager with NUMA awareness:

```python
class MemoryPool:
    def __init__(
        self,
        max_buffers: int = 32,
        numa_aware: bool = True,
        enable_monitoring: bool = True,
        cleanup_threshold: float = 0.8
    )
```

**Parameters:**
- `max_buffers`: Maximum number of buffers per pool category
- `numa_aware`: Enable NUMA-aware memory allocation
- `enable_monitoring`: Enable memory usage monitoring
- `cleanup_threshold`: Memory pressure threshold for cleanup (0.0-1.0)

## Usage Examples

### Basic Memory Pool Operations

```python
from modules.engine.memory_pool import MemoryPool
import numpy as np
import time

# Initialize memory pool
memory_pool = MemoryPool(
    max_buffers=64,
    numa_aware=True,
    enable_monitoring=True
)

# Initialize the pool
memory_pool.initialize()

# Allocate buffers of different sizes
def test_basic_operations():
    # Allocate small buffer (< 1KB)
    small_buffer = memory_pool.get_buffer(shape=(100,), dtype=np.float32)
    print(f"Small buffer shape: {small_buffer.shape}, size: {small_buffer.nbytes} bytes")
    
    # Allocate medium buffer (1KB - 1MB)
    medium_buffer = memory_pool.get_buffer(shape=(1000, 100), dtype=np.float64)
    print(f"Medium buffer shape: {medium_buffer.shape}, size: {medium_buffer.nbytes} bytes")
    
    # Allocate large buffer (> 1MB)
    large_buffer = memory_pool.get_buffer(shape=(2000, 2000), dtype=np.float32)
    print(f"Large buffer shape: {large_buffer.shape}, size: {large_buffer.nbytes} bytes")
    
    # Use buffers for computation
    small_buffer[:] = np.random.rand(100)
    medium_buffer[:] = np.random.rand(1000, 100)
    large_buffer[:] = np.random.rand(2000, 2000)
    
    # Return buffers to pool for reuse
    memory_pool.return_buffer(small_buffer)
    memory_pool.return_buffer(medium_buffer)
    memory_pool.return_buffer(large_buffer)
    
    print("All buffers returned to pool")

# Run test
test_basic_operations()

# Check pool statistics
stats = memory_pool.get_statistics()
print(f"Pool Statistics:")
print(f"  Total allocations: {stats['total_allocations']}")
print(f"  Cache hits: {stats['cache_hits']}")
print(f"  Cache hit ratio: {stats['cache_hit_ratio']:.2%}")
print(f"  Memory usage: {stats['memory_usage_mb']:.1f} MB")
```

### NUMA-Aware Allocation

```python
from modules.engine.memory_pool import MemoryPool, NUMAOptimizer
import threading
import numpy as np

# Create NUMA-aware memory pool
numa_pool = MemoryPool(
    max_buffers=128,
    numa_aware=True,
    enable_monitoring=True
)

numa_optimizer = NUMAOptimizer(numa_pool)

def numa_aware_computation(thread_id, numa_node):
    """Computation bound to specific NUMA node"""
    # Bind thread to NUMA node
    numa_optimizer.bind_thread_to_node(numa_node)
    
    print(f"Thread {thread_id} bound to NUMA node {numa_node}")
    
    # Allocate memory on the same NUMA node
    data_buffer = numa_pool.get_buffer(
        shape=(5000, 1000), 
        dtype=np.float32,
        numa_node=numa_node
    )
    
    # Perform computation (simulated)
    for i in range(10):
        # Fill buffer with computation results
        data_buffer[:] = np.random.rand(5000, 1000) * i
        
        # Simulate computation
        result = np.mean(data_buffer, axis=1)
        
        # Use result
        print(f"Thread {thread_id}, iteration {i}: Mean result = {np.mean(result):.4f}")
    
    # Return buffer to pool
    numa_pool.return_buffer(data_buffer)
    print(f"Thread {thread_id} completed")

# Run multi-threaded NUMA-aware computation
numa_nodes = numa_optimizer.get_numa_nodes()
threads = []

for i, numa_node in enumerate(numa_nodes[:4]):  # Use up to 4 NUMA nodes
    thread = threading.Thread(
        target=numa_aware_computation,
        args=(i, numa_node)
    )
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Check NUMA statistics
numa_stats = numa_optimizer.get_numa_statistics()
print("NUMA Statistics:")
for node_id, stats in numa_stats.items():
    print(f"  Node {node_id}:")
    print(f"    Allocations: {stats['allocations']}")
    print(f"    Memory usage: {stats['memory_usage_mb']:.1f} MB")
    print(f"    Local access ratio: {stats['local_access_ratio']:.2%}")
```

### High-Performance Batch Processing

```python
from modules.engine.memory_pool import MemoryPool
import numpy as np
import time

# Create memory pool optimized for batch processing
batch_pool = MemoryPool(
    max_buffers=256,
    numa_aware=True,
    enable_monitoring=True
)

class BatchProcessor:
    """High-performance batch processor using memory pool"""
    
    def __init__(self, memory_pool, batch_size=32):
        self.memory_pool = memory_pool
        self.batch_size = batch_size
        
        # Pre-allocate common buffer shapes
        self.input_buffer = memory_pool.get_buffer(
            shape=(batch_size, 224, 224, 3), 
            dtype=np.float32
        )
        self.feature_buffer = memory_pool.get_buffer(
            shape=(batch_size, 2048), 
            dtype=np.float32
        )
        self.output_buffer = memory_pool.get_buffer(
            shape=(batch_size, 1000), 
            dtype=np.float32
        )
        
    def process_batch(self, batch_data):
        """Process a batch using pre-allocated buffers"""
        # Copy input data to input buffer (zero-copy when possible)
        self.input_buffer[:len(batch_data)] = batch_data
        
        # Simulate feature extraction
        for i in range(len(batch_data)):
            self.feature_buffer[i] = np.mean(self.input_buffer[i].reshape(-1, 3), axis=0).repeat(2048//3)[:2048]
        
        # Simulate model inference
        self.output_buffer[:len(batch_data)] = np.random.rand(len(batch_data), 1000)
        
        # Return results (copy only the needed portion)
        return self.output_buffer[:len(batch_data)].copy()
    
    def cleanup(self):
        """Return buffers to pool"""
        self.memory_pool.return_buffer(self.input_buffer)
        self.memory_pool.return_buffer(self.feature_buffer)
        self.memory_pool.return_buffer(self.output_buffer)

# Create batch processor
processor = BatchProcessor(batch_pool, batch_size=32)

# Process multiple batches
n_batches = 100
batch_times = []

for batch_idx in range(n_batches):
    # Generate random batch data
    batch_data = np.random.rand(32, 224, 224, 3).astype(np.float32)
    
    # Time batch processing
    start_time = time.time()
    results = processor.process_batch(batch_data)
    end_time = time.time()
    
    batch_times.append(end_time - start_time)
    
    if batch_idx % 20 == 0:
        print(f"Batch {batch_idx}: {batch_times[-1]:.4f}s")

# Cleanup
processor.cleanup()

# Performance analysis
avg_batch_time = np.mean(batch_times)
throughput = 32 / avg_batch_time  # samples per second

print(f"Performance Summary:")
print(f"  Average batch time: {avg_batch_time:.4f}s")
print(f"  Throughput: {throughput:.1f} samples/sec")
print(f"  Total time: {sum(batch_times):.2f}s")

# Memory pool statistics
final_stats = batch_pool.get_statistics()
print(f"Memory Pool Statistics:")
print(f"  Peak memory usage: {final_stats['peak_memory_mb']:.1f} MB")
print(f"  Buffer reuse ratio: {final_stats['buffer_reuse_ratio']:.2%}")
print(f"  Allocation efficiency: {final_stats['allocation_efficiency']:.2%}")
```

### Memory Pressure Handling

```python
from modules.engine.memory_pool import MemoryPool, MemoryPressureMonitor
import psutil
import time

# Create memory pool with pressure monitoring
pressure_pool = MemoryPool(
    max_buffers=512,
    numa_aware=True,
    cleanup_threshold=0.8,  # Clean up when 80% memory used
    enable_monitoring=True
)

# Initialize pressure monitor
pressure_monitor = MemoryPressureMonitor(
    memory_pool=pressure_pool,
    check_interval=1.0,  # Check every second
    pressure_threshold=0.75,  # 75% memory usage threshold
    emergency_threshold=0.9   # 90% emergency threshold
)

def memory_intensive_workload():
    """Workload that gradually increases memory usage"""
    buffers = []
    
    try:
        for i in range(100):
            # Allocate increasingly large buffers
            size_factor = 1 + i * 0.1
            buffer_size = int(1000000 * size_factor)  # Start at 1MB
            
            buffer = pressure_pool.get_buffer(
                shape=(buffer_size,), 
                dtype=np.float32
            )
            
            # Fill buffer with data
            buffer[:] = np.random.rand(buffer_size)
            buffers.append(buffer)
            
            # Check system memory
            memory_percent = psutil.virtual_memory().percent
            print(f"Iteration {i}: Buffer size={buffer_size/1000000:.1f}MB, "
                  f"System memory: {memory_percent:.1f}%")
            
            # Let pressure monitor handle cleanup if needed
            if memory_percent > 80:
                print("High memory pressure detected, triggering cleanup...")
                pressure_monitor.handle_memory_pressure()
                
                # Return some buffers
                for j in range(min(10, len(buffers))):
                    pressure_pool.return_buffer(buffers.pop())
            
            time.sleep(0.1)
    
    except MemoryError:
        print("Memory allocation failed - returning all buffers")
    
    finally:
        # Return all remaining buffers
        for buffer in buffers:
            pressure_pool.return_buffer(buffer)

# Start pressure monitoring
pressure_monitor.start()

# Run memory-intensive workload
memory_intensive_workload()

# Stop monitoring
pressure_monitor.stop()

# Get pressure statistics
pressure_stats = pressure_monitor.get_statistics()
print("Memory Pressure Statistics:")
print(f"  Pressure events: {pressure_stats['pressure_events']}")
print(f"  Emergency cleanups: {pressure_stats['emergency_cleanups']}")
print(f"  Buffers released: {pressure_stats['buffers_released']}")
print(f"  Memory saved: {pressure_stats['memory_saved_mb']:.1f} MB")
```

## Advanced Features

### Custom Memory Allocators

```python
from modules.engine.memory_pool import CustomAllocator, AlignedAllocator

# Create aligned memory allocator for SIMD operations
simd_allocator = AlignedAllocator(
    alignment=64,  # 64-byte alignment for AVX-512
    memory_pool=memory_pool
)

# Allocate SIMD-optimized buffers
def simd_optimized_computation():
    """Computation using SIMD-aligned memory"""
    # Allocate aligned buffers
    a = simd_allocator.allocate_aligned(shape=(10000,), dtype=np.float32)
    b = simd_allocator.allocate_aligned(shape=(10000,), dtype=np.float32)
    result = simd_allocator.allocate_aligned(shape=(10000,), dtype=np.float32)
    
    # Fill with test data
    a[:] = np.random.rand(10000)
    b[:] = np.random.rand(10000)
    
    # SIMD-optimized operations (would use actual SIMD instructions)
    result[:] = a * b + np.sin(a) * np.cos(b)
    
    print(f"SIMD computation result: mean={np.mean(result):.4f}")
    
    # Return aligned buffers
    simd_allocator.deallocate_aligned(a)
    simd_allocator.deallocate_aligned(b)
    simd_allocator.deallocate_aligned(result)

simd_optimized_computation()

# Create GPU memory allocator (if CUDA available)
try:
    import cupy
    
    gpu_allocator = CustomAllocator(
        device_type="gpu",
        memory_pool=memory_pool
    )
    
    def gpu_computation():
        """GPU computation with custom allocator"""
        # Allocate GPU memory
        gpu_buffer = gpu_allocator.allocate_device(
            shape=(10000, 10000), 
            dtype=np.float32
        )
        
        # Perform GPU computation
        gpu_result = cupy.sum(gpu_buffer ** 2)
        
        print(f"GPU computation result: {gpu_result}")
        
        # Return GPU memory
        gpu_allocator.deallocate_device(gpu_buffer)
    
    gpu_computation()
    
except ImportError:
    print("CuPy not available, skipping GPU allocator test")
```

### Memory Pool Analytics

```python
from modules.engine.memory_pool import MemoryAnalytics
import matplotlib.pyplot as plt

# Create analytics module
analytics = MemoryAnalytics(memory_pool)

# Run workload with analytics
analytics.start_collection()

# Simulate various workloads
workloads = [
    {"name": "small_buffers", "shape": (1000,), "count": 100},
    {"name": "medium_buffers", "shape": (10000, 100), "count": 50},
    {"name": "large_buffers", "shape": (5000, 5000), "count": 10}
]

for workload in workloads:
    print(f"Running workload: {workload['name']}")
    
    buffers = []
    for i in range(workload['count']):
        buffer = memory_pool.get_buffer(
            shape=workload['shape'], 
            dtype=np.float32
        )
        buffers.append(buffer)
        
        # Simulate work
        time.sleep(0.01)
    
    # Return buffers
    for buffer in buffers:
        memory_pool.return_buffer(buffer)

analytics.stop_collection()

# Generate analytics report
report = analytics.generate_report()

print("Memory Pool Analytics Report:")
print(f"  Total allocation events: {report['total_allocations']}")
print(f"  Peak memory usage: {report['peak_memory_mb']:.1f} MB")
print(f"  Average allocation time: {report['avg_allocation_time_ms']:.3f} ms")
print(f"  Memory fragmentation: {report['fragmentation_ratio']:.2%}")
print(f"  NUMA efficiency: {report['numa_efficiency']:.2%}")

# Plot memory usage over time
analytics.plot_memory_usage_timeline("memory_usage.png")

# Plot allocation size distribution
analytics.plot_allocation_size_distribution("allocation_distribution.png")

# Plot NUMA node utilization
analytics.plot_numa_utilization("numa_utilization.png")

print("Analytics plots saved to disk")
```

### Integration with ML Frameworks

```python
from modules.engine.memory_pool import MLFrameworkIntegration
import numpy as np

# Create framework integration
ml_integration = MLFrameworkIntegration(memory_pool)

# Scikit-learn integration
def sklearn_memory_optimization():
    """Optimize sklearn operations with memory pool"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate data using memory pool
    X, y = ml_integration.generate_sklearn_data(
        n_samples=10000,
        n_features=100,
        n_classes=2
    )
    
    # Create memory-optimized estimator
    clf = ml_integration.create_memory_optimized_estimator(
        RandomForestClassifier,
        n_estimators=100,
        memory_pool=memory_pool
    )
    
    # Train with memory optimization
    clf.fit(X, y)
    
    # Predict with memory reuse
    predictions = clf.predict(X)
    
    print(f"Sklearn integration: Accuracy = {clf.score(X, y):.4f}")
    
    # Return memory to pool
    ml_integration.cleanup_sklearn_data(X, y)

sklearn_memory_optimization()

# PyTorch integration (if available)
try:
    import torch
    
    def pytorch_memory_optimization():
        """Optimize PyTorch operations with memory pool"""
        # Create PyTorch tensors using memory pool
        tensor_a = ml_integration.create_torch_tensor(
            shape=(1000, 1000),
            dtype=torch.float32,
            device='cpu'
        )
        
        tensor_b = ml_integration.create_torch_tensor(
            shape=(1000, 1000),
            dtype=torch.float32,
            device='cpu'
        )
        
        # Perform operations
        result = torch.matmul(tensor_a, tensor_b)
        
        print(f"PyTorch integration: Result shape = {result.shape}")
        
        # Return tensors to pool
        ml_integration.return_torch_tensor(tensor_a)
        ml_integration.return_torch_tensor(tensor_b)
        ml_integration.return_torch_tensor(result)
    
    pytorch_memory_optimization()
    
except ImportError:
    print("PyTorch not available, skipping PyTorch integration test")
```

## Best Practices

### 1. Pool Sizing and Configuration

```python
# Configure pool based on workload characteristics
def configure_memory_pool(workload_type):
    """Configure memory pool for specific workload types"""
    
    if workload_type == "inference":
        # Inference workloads: small pools, fast allocation
        return MemoryPool(
            max_buffers=64,
            numa_aware=True,
            cleanup_threshold=0.9,  # Keep more memory
            enable_monitoring=False  # Reduce overhead
        )
    
    elif workload_type == "training":
        # Training workloads: large pools, aggressive cleanup
        return MemoryPool(
            max_buffers=256,
            numa_aware=True,
            cleanup_threshold=0.7,  # More aggressive cleanup
            enable_monitoring=True
        )
    
    elif workload_type == "batch_processing":
        # Batch processing: very large pools
        return MemoryPool(
            max_buffers=512,
            numa_aware=True,
            cleanup_threshold=0.8,
            enable_monitoring=True
        )
    
    else:
        # Default configuration
        return MemoryPool()

# Use appropriate configuration
inference_pool = configure_memory_pool("inference")
training_pool = configure_memory_pool("training")
```

### 2. Error Handling and Recovery

```python
def robust_memory_operations(memory_pool):
    """Robust memory operations with error handling"""
    buffer = None
    
    try:
        # Attempt to allocate buffer
        buffer = memory_pool.get_buffer(shape=(10000, 10000), dtype=np.float32)
        
        # Perform operations
        buffer[:] = np.random.rand(10000, 10000)
        result = np.mean(buffer)
        
        return result
        
    except MemoryError:
        # Handle memory allocation failure
        print("Memory allocation failed, trying smaller buffer...")
        try:
            buffer = memory_pool.get_buffer(shape=(5000, 5000), dtype=np.float32)
            buffer[:] = np.random.rand(5000, 5000)
            return np.mean(buffer)
        except MemoryError:
            print("Even smaller allocation failed, falling back to standard allocation")
            data = np.random.rand(1000, 1000)
            return np.mean(data)
    
    finally:
        # Always return buffer to pool
        if buffer is not None:
            memory_pool.return_buffer(buffer)
```

### 3. Performance Monitoring

```python
def monitor_memory_performance(memory_pool):
    """Monitor memory pool performance"""
    
    # Set up monitoring
    start_stats = memory_pool.get_statistics()
    start_time = time.time()
    
    # Run workload
    for i in range(1000):
        buffer = memory_pool.get_buffer(shape=(1000,), dtype=np.float32)
        buffer[:] = i
        memory_pool.return_buffer(buffer)
    
    # Check performance
    end_stats = memory_pool.get_statistics()
    end_time = time.time()
    
    # Calculate metrics
    allocations = end_stats['total_allocations'] - start_stats['total_allocations']
    duration = end_time - start_time
    allocation_rate = allocations / duration
    
    print(f"Memory Pool Performance:")
    print(f"  Allocations: {allocations}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Allocation rate: {allocation_rate:.1f} allocs/sec")
    print(f"  Cache hit ratio: {end_stats['cache_hit_ratio']:.2%}")

monitor_memory_performance(memory_pool)
```

## Related Documentation

- [Dynamic Batcher Documentation](dynamic_batcher.md)
- [SIMD Optimizer Documentation](simd_optimizer.md)
- [Performance Metrics Documentation](performance_metrics.md)
- [JIT Compiler Documentation](jit_compiler.md)
- [Inference Engine Documentation](inference_engine.md)
