# JIT Compiler (`modules/engine/jit_compiler.py`)

## Overview

The JIT Compiler module provides Just-In-Time compilation capabilities for frequent model patterns and hot paths in ML workloads using Numba. It automatically identifies performance-critical code sections and applies optimal compilation strategies to accelerate computation.

## Features

- **Automatic Hot Path Detection**: Identifies frequently called functions for optimization
- **Intelligent Compilation**: Selective JIT compilation based on performance profiling
- **Performance Monitoring**: Real-time tracking of compilation benefits
- **Memory Optimization**: Efficient memory usage patterns for compiled functions
- **Fallback Mechanisms**: Graceful degradation when JIT compilation fails
- **Multi-threading Support**: Thread-safe compilation and execution

## Core Classes

### JITCompiler

Main JIT compilation manager with automatic optimization:

```python
class JITCompiler:
    def __init__(
        self,
        min_calls_for_compilation: int = 10,
        enable_parallel: bool = True,
        cache_compiled_functions: bool = True,
        profile_performance: bool = True,
        auto_optimize: bool = True
    )
```

**Parameters:**
- `min_calls_for_compilation`: Minimum calls before considering JIT compilation
- `enable_parallel`: Enable parallel execution with prange
- `cache_compiled_functions`: Cache compiled functions for reuse
- `profile_performance`: Track performance improvements
- `auto_optimize`: Automatically optimize hot paths

### HotPathTracker

Tracks function call patterns to identify optimization candidates:

```python
class HotPathTracker:
    def __init__(
        self,
        min_calls_for_compilation: int = 10,
        performance_threshold: float = 0.001
    )
```

## Usage Examples

### Basic JIT Compilation

```python
from modules.engine.jit_compiler import JITCompiler
import numpy as np
import time

# Initialize JIT compiler
jit_compiler = JITCompiler(
    min_calls_for_compilation=5,
    enable_parallel=True,
    auto_optimize=True
)

# Define functions that will be JIT compiled
@jit_compiler.auto_jit
def matrix_multiply_optimized(a, b):
    """Matrix multiplication with automatic JIT compilation"""
    return np.dot(a, b)

@jit_compiler.auto_jit
def element_wise_operations(x, y, z):
    """Element-wise operations with JIT optimization"""
    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = x[i] * y[i] + z[i] ** 2
    return result

# Use functions - they will be compiled after enough calls
data_a = np.random.rand(1000, 1000)
data_b = np.random.rand(1000, 1000)

# First few calls will use Python interpreter
for i in range(3):
    result = matrix_multiply_optimized(data_a, data_b)
    print(f"Call {i+1}: Not yet compiled")

# After min_calls_for_compilation, function will be JIT compiled
for i in range(5):
    start_time = time.time()
    result = matrix_multiply_optimized(data_a, data_b)
    end_time = time.time()
    print(f"Call {i+4}: Execution time: {end_time - start_time:.4f}s")

# Get compilation statistics
stats = jit_compiler.get_compilation_stats()
print("Compilation Statistics:")
for func_name, stat in stats.items():
    print(f"  {func_name}:")
    print(f"    Call count: {stat.call_count}")
    print(f"    Average speedup: {stat.average_speedup:.2f}x")
    print(f"    Compile time: {stat.total_compile_time:.4f}s")
```

### Advanced JIT Optimization Patterns

```python
from modules.engine.jit_compiler import JITCompiler, jit_optimize
import numba

# Create compiler with custom configuration
compiler = JITCompiler(
    min_calls_for_compilation=3,
    enable_parallel=True,
    profile_performance=True
)

# Manual JIT compilation with specific optimizations
@compiler.jit_compile(
    signature="float64[:](float64[:], float64[:])",
    parallel=True,
    cache=True,
    nogil=True
)
def parallel_vector_operations(x, y):
    """Parallel vector operations with explicit JIT configuration"""
    result = np.empty(len(x))
    for i in numba.prange(len(x)):  # Parallel loop
        result[i] = np.sqrt(x[i] ** 2 + y[i] ** 2)
    return result

# Conditional JIT compilation based on data size
@compiler.conditional_jit(size_threshold=1000)
def adaptive_computation(data):
    """JIT compile only for large datasets"""
    if len(data) < 1000:
        # Use regular Python for small data
        return np.sum(data ** 2)
    else:
        # Use JIT compilation for large data
        return jit_sum_squares(data)

@numba.njit
def jit_sum_squares(data):
    """Pre-compiled function for large datasets"""
    total = 0.0
    for i in range(len(data)):
        total += data[i] ** 2
    return total

# Test with different data sizes
small_data = np.random.rand(100)
large_data = np.random.rand(10000)

small_result = adaptive_computation(small_data)  # Uses Python
large_result = adaptive_computation(large_data)  # Uses JIT

print(f"Small data result: {small_result:.4f}")
print(f"Large data result: {large_result:.4f}")
```

### Machine Learning Specific Optimizations

```python
from modules.engine.jit_compiler import MLJITOptimizer
import numpy as np

# ML-specific JIT optimizer
ml_optimizer = MLJITOptimizer(
    enable_vectorization=True,
    optimize_for_inference=True
)

# Optimized distance calculations
@ml_optimizer.optimize_distance_function
def euclidean_distance_batch(X, centroids):
    """Optimized batch Euclidean distance calculation"""
    n_samples, n_features = X.shape
    n_centroids = centroids.shape[0]
    distances = np.zeros((n_samples, n_centroids))
    
    for i in numba.prange(n_samples):
        for j in range(n_centroids):
            dist = 0.0
            for k in range(n_features):
                diff = X[i, k] - centroids[j, k]
                dist += diff * diff
            distances[i, j] = np.sqrt(dist)
    
    return distances

# Optimized activation functions
@ml_optimizer.optimize_activation
def relu_batch(x):
    """Optimized ReLU activation for batch processing"""
    result = np.zeros_like(x)
    for i in numba.prange(x.size):
        result.flat[i] = max(0.0, x.flat[i])
    return result

@ml_optimizer.optimize_activation  
def sigmoid_batch(x):
    """Optimized sigmoid activation for batch processing"""
    result = np.zeros_like(x)
    for i in numba.prange(x.size):
        val = x.flat[i]
        if val > 500:  # Prevent overflow
            result.flat[i] = 1.0
        elif val < -500:
            result.flat[i] = 0.0
        else:
            result.flat[i] = 1.0 / (1.0 + np.exp(-val))
    return result

# Optimized loss functions
@ml_optimizer.optimize_loss_function
def mse_loss_batch(y_true, y_pred):
    """Optimized mean squared error loss"""
    n = y_true.size
    total_error = 0.0
    
    for i in numba.prange(n):
        diff = y_true.flat[i] - y_pred.flat[i]
        total_error += diff * diff
    
    return total_error / n

# Test ML optimizations
X = np.random.rand(10000, 50)
centroids = np.random.rand(10, 50)

# This will be JIT compiled for fast execution
distances = euclidean_distance_batch(X, centroids)
print(f"Distance matrix shape: {distances.shape}")

# Test activation functions
activations = np.random.randn(10000, 100)
relu_output = relu_batch(activations)
sigmoid_output = sigmoid_batch(activations)

print(f"ReLU output range: [{relu_output.min():.4f}, {relu_output.max():.4f}]")
print(f"Sigmoid output range: [{sigmoid_output.min():.4f}, {sigmoid_output.max():.4f}]")
```

### Performance Profiling and Optimization

```python
from modules.engine.jit_compiler import PerformanceProfiler
import time

# Create performance profiler
profiler = PerformanceProfiler(jit_compiler)

# Profile function performance
@profiler.profile_function
def expensive_computation(n):
    """Computationally expensive function for profiling"""
    result = 0.0
    for i in range(n):
        for j in range(n):
            result += np.sin(i) * np.cos(j)
    return result

# Run profiling
sizes = [100, 200, 500, 1000]
for size in sizes:
    # Time without JIT
    start = time.time()
    result_python = expensive_computation(size)
    python_time = time.time() - start
    
    # Enable JIT for this function
    jit_compiler.force_compile('expensive_computation')
    
    # Time with JIT
    start = time.time()
    result_jit = expensive_computation(size)
    jit_time = time.time() - start
    
    speedup = python_time / jit_time if jit_time > 0 else 0
    print(f"Size {size}: Python={python_time:.4f}s, JIT={jit_time:.4f}s, Speedup={speedup:.2f}x")

# Generate performance report
report = profiler.generate_report()
print("\nPerformance Report:")
print(f"Functions profiled: {report['functions_profiled']}")
print(f"Average speedup: {report['average_speedup']:.2f}x")
print(f"Best speedup: {report['best_speedup']:.2f}x")
print(f"Total compilation time: {report['total_compile_time']:.4f}s")
```

## Advanced Features

### Automatic Function Inlining

```python
from modules.engine.jit_compiler import InlineOptimizer

# Create inline optimizer
inline_optimizer = InlineOptimizer(jit_compiler)

# Functions that will be automatically inlined
@inline_optimizer.inline_candidate
def fast_dot_product(a, b):
    """Small function that benefits from inlining"""
    return np.sum(a * b)

@inline_optimizer.inline_candidate
def normalize_vector(x):
    """Small normalization function"""
    norm = np.sqrt(np.sum(x * x))
    return x / norm if norm > 0 else x

# Main function that uses inlined functions
@jit_compiler.auto_jit
def complex_vector_operation(vectors):
    """Complex operation using inlined functions"""
    results = []
    for i in range(len(vectors)):
        normalized = normalize_vector(vectors[i])
        if i > 0:
            similarity = fast_dot_product(normalized, results[-1])
            results.append(normalized * similarity)
        else:
            results.append(normalized)
    return np.array(results)

# The inline optimizer will automatically inline small functions
test_vectors = [np.random.rand(10) for _ in range(100)]
result = complex_vector_operation(test_vectors)
```

### Memory-Optimized Compilation

```python
from modules.engine.jit_compiler import MemoryOptimizer

# Memory-aware JIT compilation
memory_optimizer = MemoryOptimizer(
    max_memory_usage_mb=2048,
    enable_memory_pooling=True,
    optimize_for_cache=True
)

@memory_optimizer.memory_efficient_jit
def large_matrix_operations(matrices):
    """Memory-efficient operations on large matrices"""
    n_matrices = len(matrices)
    results = []
    
    # Process matrices in chunks to manage memory
    chunk_size = memory_optimizer.calculate_optimal_chunk_size(matrices[0].shape)
    
    for i in range(0, n_matrices, chunk_size):
        chunk_end = min(i + chunk_size, n_matrices)
        chunk_matrices = matrices[i:chunk_end]
        
        # Process chunk
        chunk_result = process_matrix_chunk_jit(chunk_matrices)
        results.extend(chunk_result)
        
        # Force garbage collection for large chunks
        if chunk_size > 10:
            memory_optimizer.cleanup_memory()
    
    return results

@numba.njit
def process_matrix_chunk_jit(chunk):
    """JIT-compiled chunk processing"""
    results = []
    for matrix in chunk:
        # In-place operations to save memory
        matrix_copy = matrix.copy()
        # Perform operations...
        results.append(matrix_copy)
    return results

# Test memory-efficient processing
large_matrices = [np.random.rand(1000, 1000) for _ in range(50)]
processed = large_matrix_operations(large_matrices)

# Monitor memory usage
memory_stats = memory_optimizer.get_memory_stats()
print(f"Peak memory usage: {memory_stats['peak_usage_mb']:.1f} MB")
print(f"Memory efficiency: {memory_stats['efficiency_score']:.2f}")
```

### Dynamic Recompilation

```python
from modules.engine.jit_compiler import DynamicRecompiler

# Dynamic recompilation based on runtime characteristics
recompiler = DynamicRecompiler(
    recompilation_threshold=100,  # Recompile after 100 calls
    performance_degradation_threshold=0.1  # 10% performance drop
)

@recompiler.dynamic_jit
def adaptive_algorithm(data, algorithm_type="auto"):
    """Algorithm that adapts compilation based on data characteristics"""
    if algorithm_type == "auto":
        # Automatically select algorithm based on data properties
        if len(data) < 1000:
            return small_data_algorithm(data)
        elif np.std(data) < 0.1:
            return uniform_data_algorithm(data)
        else:
            return general_algorithm(data)
    else:
        return general_algorithm(data)

@numba.njit
def small_data_algorithm(data):
    """Optimized for small datasets"""
    return np.sum(data ** 2)

@numba.njit
def uniform_data_algorithm(data):
    """Optimized for uniform data"""
    return len(data) * np.mean(data) ** 2

@numba.njit
def general_algorithm(data):
    """General-purpose algorithm"""
    result = 0.0
    for x in data:
        result += x * x
    return result

# The recompiler will adapt based on data characteristics
test_datasets = [
    np.random.rand(100),              # Small data
    np.full(5000, 0.5) + np.random.normal(0, 0.01, 5000),  # Uniform data
    np.random.exponential(2, 10000)   # General data
]

for i, dataset in enumerate(test_datasets):
    result = adaptive_algorithm(dataset)
    print(f"Dataset {i+1}: Result = {result:.4f}")

# Check recompilation statistics
recompilation_stats = recompiler.get_stats()
print(f"Recompilations performed: {recompilation_stats['recompilations']}")
print(f"Performance improvements: {recompilation_stats['improvements']}")
```

## Integration with ML Pipelines

### Scikit-learn Integration

```python
from modules.engine.jit_compiler import SklearnJITOptimizer
from sklearn.base import BaseEstimator, TransformerMixin

class JITOptimizedTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer with JIT optimization"""
    
    def __init__(self):
        self.jit_optimizer = SklearnJITOptimizer()
        self.fitted_ = False
    
    def fit(self, X, y=None):
        """Fit the transformer and compile JIT functions"""
        self.feature_means_ = np.mean(X, axis=0)
        self.feature_stds_ = np.std(X, axis=0)
        
        # Compile JIT functions based on data characteristics
        self.jit_optimizer.compile_for_data_shape(X.shape)
        self.fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data using JIT-optimized functions"""
        if not self.fitted_:
            raise ValueError("Transformer not fitted")
        
        return self.jit_optimizer.transform_jit(
            X, self.feature_means_, self.feature_stds_
        )

# The JIT optimizer will compile transform function based on data shape
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=10000, n_features=50, random_state=42)

# Create pipeline with JIT-optimized transformer
pipeline = Pipeline([
    ('jit_transform', JITOptimizedTransformer()),
    ('classifier', RandomForestClassifier())
])

# Training will trigger JIT compilation
pipeline.fit(X, y)

# Subsequent transforms will use compiled functions
X_transformed = pipeline.named_steps['jit_transform'].transform(X)
print(f"Transformed data shape: {X_transformed.shape}")
```

## Best Practices

### 1. Profile Before Optimizing

```python
# Always profile to identify true bottlenecks
profiler = PerformanceProfiler(jit_compiler)

# Profile your code first
@profiler.profile_function
def candidate_for_optimization(data):
    # Your function here
    return np.sum(data ** 2)

# Run profiling to see if JIT helps
profiler.run_comparison_benchmark(candidate_for_optimization, test_data)
```

### 2. Use Appropriate JIT Strategies

```python
# For small, frequently called functions
@jit_compiler.auto_jit(min_calls=5)
def small_frequent_function(x):
    return x * 2 + 1

# For large, compute-intensive functions
@jit_compiler.eager_jit(parallel=True)
def large_compute_function(data):
    # Heavy computation
    pass

# For functions with varying input sizes
@jit_compiler.conditional_jit(size_threshold=1000)
def size_dependent_function(data):
    # Function that benefits from JIT only for large inputs
    pass
```

### 3. Handle JIT Compilation Failures

```python
@jit_compiler.safe_jit(fallback_to_python=True)
def potentially_problematic_function(data):
    """Function that might not compile successfully"""
    try:
        # Complex operations that might not be JIT-compatible
        return complex_operation(data)
    except Exception as e:
        # Log compilation issues
        logging.warning(f"JIT compilation failed: {e}")
        # Fallback to Python implementation
        return python_implementation(data)
```

## Related Documentation

- [Performance Metrics Documentation](performance_metrics.md)
- [SIMD Optimizer Documentation](simd_optimizer.md)
- [Mixed Precision Documentation](mixed_precision.md)
- [Memory Pool Documentation](memory_pool.md)
- [Training Engine Documentation](train_engine.md)
