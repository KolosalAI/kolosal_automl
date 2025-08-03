# SIMD Optimizer (`modules/engine/simd_optimizer.py`)

## Overview

The SIMD Optimizer module provides vectorized operations optimized for Single Instruction, Multiple Data (SIMD) CPU instructions. It leverages CPU-specific features like AVX, AVX2, and AVX-512 to accelerate linear algebra operations beyond basic NumPy optimizations, particularly for machine learning workloads.

## Features

- **CPU Feature Detection**: Automatic detection of available SIMD instruction sets
- **Optimized Linear Operations**: Accelerated matrix operations for ML workloads
- **Numba Integration**: JIT compilation for performance-critical code paths
- **Memory Layout Optimization**: Ensures optimal memory alignment for SIMD operations
- **Fallback Support**: Graceful degradation when advanced features aren't available

## Core Classes

### SIMDOptimizer

Main optimizer class that provides SIMD-accelerated operations:

```python
class SIMDOptimizer:
    def __init__(self):
        self.use_numba = NUMBA_AVAILABLE
        self.cpu_features = self._detect_cpu_features()
```

**Key Methods:**
- `optimized_linear_forward()`: Accelerated linear layer forward pass
- `get_optimization_info()`: Information about available optimizations
- `_detect_cpu_features()`: CPU capability detection

## Usage Examples

### Basic SIMD Optimization

```python
from modules.engine.simd_optimizer import SIMDOptimizer
import numpy as np
import time

# Initialize optimizer
optimizer = SIMDOptimizer()

# Check available optimizations
optimization_info = optimizer.get_optimization_info()
print("SIMD Optimization Info:")
print(f"  Numba available: {optimization_info['numba_available']}")
print(f"  CPU features: {optimization_info['cpu_features']}")
print(f"  Optimized functions: {optimization_info['optimized_functions']}")

# Create test data for linear operations
batch_size = 1000
input_dim = 512
output_dim = 256

# Input data (batch_size, input_dim)
x = np.random.randn(batch_size, input_dim).astype(np.float32)
# Weight matrix (output_dim, input_dim)
weight = np.random.randn(output_dim, input_dim).astype(np.float32)
# Bias vector (output_dim,)
bias = np.random.randn(output_dim).astype(np.float32)

print(f"\nTest data shapes:")
print(f"  Input: {x.shape}")
print(f"  Weight: {weight.shape}")
print(f"  Bias: {bias.shape}")

# Benchmark standard NumPy operation
start_time = time.time()
for _ in range(100):
    numpy_result = np.dot(x, weight.T) + bias
numpy_time = time.time() - start_time

# Benchmark SIMD-optimized operation
start_time = time.time()
for _ in range(100):
    simd_result = optimizer.optimized_linear_forward(x, weight, bias)
simd_time = time.time() - start_time

print(f"\nPerformance Comparison (100 iterations):")
print(f"  NumPy time: {numpy_time:.4f}s")
print(f"  SIMD time: {simd_time:.4f}s")
print(f"  Speedup: {numpy_time/simd_time:.2f}x")

# Verify correctness
difference = np.abs(numpy_result - simd_result).max()
print(f"  Max difference: {difference:.2e}")
print(f"  Results match: {np.allclose(numpy_result, simd_result)}")
```

### Neural Network Layer Optimization

```python
from modules.engine.simd_optimizer import SIMDOptimizer
import numpy as np
import time

class OptimizedLinearLayer:
    """Linear layer with SIMD optimization"""
    
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # Initialize optimizer
        self.optimizer = SIMDOptimizer()
        
        # Initialize parameters
        self.weight = np.random.randn(output_dim, input_dim).astype(np.float32) * 0.1
        self.bias = np.random.randn(output_dim).astype(np.float32) * 0.1 if use_bias else None
        
        print(f"Layer initialized with SIMD optimizer")
        print(f"  CPU features: {self.optimizer.cpu_features}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with SIMD optimization"""
        return self.optimizer.optimized_linear_forward(x, self.weight, self.bias)
    
    def forward_baseline(self, x: np.ndarray) -> np.ndarray:
        """Baseline forward pass for comparison"""
        result = np.dot(x, self.weight.T)
        if self.bias is not None:
            result += self.bias
        return result

class OptimizedMLP:
    """Multi-layer perceptron with SIMD-optimized layers"""
    
    def __init__(self, layer_sizes: list):
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            layer = OptimizedLinearLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i + 1],
                use_bias=True
            )
            self.layers.append(layer)
        
        print(f"MLP created with {len(self.layers)} SIMD-optimized layers")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        current = x
        
        for i, layer in enumerate(self.layers):
            current = layer.forward(current)
            
            # Apply ReLU activation (except last layer)
            if i < len(self.layers) - 1:
                current = np.maximum(0, current)
        
        return current
    
    def forward_baseline(self, x: np.ndarray) -> np.ndarray:
        """Baseline forward pass for comparison"""
        current = x
        
        for i, layer in enumerate(self.layers):
            current = layer.forward_baseline(current)
            
            # Apply ReLU activation (except last layer)
            if i < len(self.layers) - 1:
                current = np.maximum(0, current)
        
        return current

# Create optimized MLP
layer_sizes = [784, 512, 256, 128, 10]  # MNIST-like architecture
mlp = OptimizedMLP(layer_sizes)

# Create test batch
batch_size = 1000
x = np.random.randn(batch_size, 784).astype(np.float32)

print(f"\nTesting MLP with batch size {batch_size}")

# Benchmark optimized version
start_time = time.time()
for _ in range(50):
    optimized_output = mlp.forward(x)
optimized_time = time.time() - start_time

# Benchmark baseline version
start_time = time.time()
for _ in range(50):
    baseline_output = mlp.forward_baseline(x)
baseline_time = time.time() - start_time

print(f"\nMLP Performance (50 iterations):")
print(f"  Optimized time: {optimized_time:.4f}s")
print(f"  Baseline time: {baseline_time:.4f}s")
print(f"  Speedup: {baseline_time/optimized_time:.2f}x")

# Verify correctness
max_diff = np.abs(optimized_output - baseline_output).max()
print(f"  Max difference: {max_diff:.2e}")
print(f"  Results match: {np.allclose(optimized_output, baseline_output, atol=1e-5)}")
```

### Memory Layout Optimization

```python
from modules.engine.simd_optimizer import SIMDOptimizer
import numpy as np
import time

def benchmark_memory_layouts():
    """Benchmark different memory layouts for SIMD operations"""
    
    optimizer = SIMDOptimizer()
    
    # Test parameters
    batch_size = 2000
    input_dim = 1000
    output_dim = 500
    
    # Create data with different layouts
    print("Creating test data with different memory layouts...")
    
    # Contiguous arrays (C-order)
    x_contiguous = np.random.randn(batch_size, input_dim).astype(np.float32)
    weight_contiguous = np.random.randn(output_dim, input_dim).astype(np.float32)
    
    # Non-contiguous arrays (created by slicing)
    x_large = np.random.randn(batch_size * 2, input_dim * 2).astype(np.float32)
    weight_large = np.random.randn(output_dim * 2, input_dim * 2).astype(np.float32)
    
    x_noncontiguous = x_large[::2, ::2]  # Non-contiguous view
    weight_noncontiguous = weight_large[::2, ::2]  # Non-contiguous view
    
    # Fortran-order arrays
    x_fortran = np.asfortranarray(x_contiguous)
    weight_fortran = np.asfortranarray(weight_contiguous)
    
    print(f"Array properties:")
    print(f"  Contiguous - x.flags.c_contiguous: {x_contiguous.flags.c_contiguous}")
    print(f"  Non-contiguous - x.flags.c_contiguous: {x_noncontiguous.flags.c_contiguous}")
    print(f"  Fortran - x.flags.f_contiguous: {x_fortran.flags.f_contiguous}")
    
    # Benchmark each layout
    layouts = [
        ("Contiguous (C-order)", x_contiguous, weight_contiguous),
        ("Non-contiguous", x_noncontiguous, weight_noncontiguous),
        ("Fortran-order", x_fortran, weight_fortran)
    ]
    
    results = {}
    iterations = 20
    
    for layout_name, x_data, weight_data in layouts:
        print(f"\nBenchmarking {layout_name}...")
        
        # Warm up
        for _ in range(5):
            _ = optimizer.optimized_linear_forward(x_data, weight_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            result = optimizer.optimized_linear_forward(x_data, weight_data)
        elapsed_time = time.time() - start_time
        
        results[layout_name] = {
            'time': elapsed_time,
            'time_per_iteration': elapsed_time / iterations,
            'result_shape': result.shape
        }
        
        print(f"  Time: {elapsed_time:.4f}s ({elapsed_time/iterations:.6f}s per iteration)")
    
    # Compare results
    contiguous_time = results["Contiguous (C-order)"]["time"]
    print(f"\nMemory Layout Performance Comparison:")
    for layout_name, layout_results in results.items():
        speedup = contiguous_time / layout_results["time"]
        print(f"  {layout_name}: {speedup:.2f}x relative to contiguous")
    
    return results

# Run memory layout benchmark
layout_results = benchmark_memory_layouts()
```

### Advanced SIMD Techniques

```python
from modules.engine.simd_optimizer import SIMDOptimizer
import numpy as np
import time

class AdvancedSIMDOperations:
    """Advanced SIMD optimization techniques"""
    
    def __init__(self):
        self.optimizer = SIMDOptimizer()
        self.cpu_features = self.optimizer.cpu_features
    
    def optimized_batch_operations(self, data_batches: list) -> list:
        """Optimize operations across multiple batches"""
        results = []
        
        # Process each batch with SIMD optimization
        for batch in data_batches:
            if len(batch) == 3:  # (x, weight, bias)
                x, weight, bias = batch
                result = self.optimizer.optimized_linear_forward(x, weight, bias)
            elif len(batch) == 2:  # (x, weight)
                x, weight = batch
                result = self.optimizer.optimized_linear_forward(x, weight)
            else:
                raise ValueError(f"Invalid batch format: {len(batch)} elements")
            
            results.append(result)
        
        return results
    
    def memory_prefetch_optimization(self, x: np.ndarray, weights: list) -> list:
        """Optimize memory access patterns for multiple weight matrices"""
        # Ensure input is contiguous
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x)
        
        results = []
        
        # Process weights in order to optimize cache usage
        for weight in weights:
            # Ensure weight is contiguous
            if not weight.flags.c_contiguous:
                weight = np.ascontiguousarray(weight)
            
            # Perform computation
            result = self.optimizer.optimized_linear_forward(x, weight)
            results.append(result)
        
        return results
    
    def vectorized_activation_functions(self, x: np.ndarray, activation: str) -> np.ndarray:
        """SIMD-optimized activation functions"""
        # Ensure contiguous memory layout
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x)
        
        if activation == 'relu':
            # Use NumPy's vectorized maximum (SIMD optimized)
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            # Prevent overflow in exponential
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'gelu':
            # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        else:
            raise ValueError(f"Unsupported activation: {activation}")

# Example usage of advanced SIMD operations
advanced_simd = AdvancedSIMDOperations()

print("Advanced SIMD Operations Demo")
print(f"CPU Features: {advanced_simd.cpu_features}")

# Test batch operations
batch_size = 500
input_dim = 256
output_dims = [128, 64, 32, 16]

# Create multiple batches for processing
batches = []
for output_dim in output_dims:
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
    weight = np.random.randn(output_dim, input_dim).astype(np.float32)
    bias = np.random.randn(output_dim).astype(np.float32)
    batches.append((x, weight, bias))

print(f"\nProcessing {len(batches)} batches with SIMD optimization...")

start_time = time.time()
batch_results = advanced_simd.optimized_batch_operations(batches)
batch_time = time.time() - start_time

print(f"Batch processing completed in {batch_time:.4f}s")
print(f"Result shapes: {[result.shape for result in batch_results]}")

# Test memory prefetch optimization
x = np.random.randn(batch_size, input_dim).astype(np.float32)
weights = [np.random.randn(64, input_dim).astype(np.float32) for _ in range(10)]

print(f"\nTesting memory prefetch optimization with {len(weights)} weight matrices...")

start_time = time.time()
prefetch_results = advanced_simd.memory_prefetch_optimization(x, weights)
prefetch_time = time.time() - start_time

print(f"Prefetch optimization completed in {prefetch_time:.4f}s")

# Test vectorized activations
test_data = np.random.randn(1000, 512).astype(np.float32)
activations = ['relu', 'sigmoid', 'tanh', 'gelu']

print(f"\nTesting SIMD-optimized activation functions...")

for activation in activations:
    start_time = time.time()
    for _ in range(100):
        result = advanced_simd.vectorized_activation_functions(test_data, activation)
    activation_time = time.time() - start_time
    
    print(f"  {activation}: {activation_time:.4f}s (100 iterations)")
```

### CPU Feature Detection and Optimization

```python
from modules.engine.simd_optimizer import SIMDOptimizer
import numpy as np

def detailed_cpu_analysis():
    """Detailed analysis of CPU capabilities for SIMD optimization"""
    
    optimizer = SIMDOptimizer()
    features = optimizer.cpu_features
    
    print("=== CPU SIMD Feature Analysis ===")
    
    # Feature descriptions
    feature_descriptions = {
        'sse4': 'SSE4.1 - Basic SIMD (128-bit vectors)',
        'avx': 'AVX - Advanced Vector Extensions (256-bit vectors)',
        'avx2': 'AVX2 - Enhanced AVX with integer operations',
        'avx512': 'AVX-512 - 512-bit vector operations',
        'fma': 'FMA - Fused Multiply-Add operations'
    }
    
    print("Available SIMD Instructions:")
    for feature, available in features.items():
        status = "✓ Available" if available else "✗ Not available"
        description = feature_descriptions.get(feature, "Unknown feature")
        print(f"  {feature.upper()}: {status} - {description}")
    
    # Recommend optimal data types and operations
    print("\n=== Optimization Recommendations ===")
    
    if features.get('avx512', False):
        print("🚀 AVX-512 detected - Optimal for large batch processing")
        print("   - Use 512-bit aligned arrays")
        print("   - Batch sizes multiple of 16 (float32) or 8 (float64)")
        recommended_dtype = np.float32
        optimal_batch_multiple = 16
    elif features.get('avx2', False):
        print("⚡ AVX2 detected - Good for medium-large operations")
        print("   - Use 256-bit aligned arrays")
        print("   - Batch sizes multiple of 8 (float32) or 4 (float64)")
        recommended_dtype = np.float32
        optimal_batch_multiple = 8
    elif features.get('avx', False):
        print("📈 AVX detected - Suitable for moderate acceleration")
        print("   - Use 256-bit aligned arrays")
        print("   - Focus on float operations")
        recommended_dtype = np.float32
        optimal_batch_multiple = 8
    elif features.get('sse4', False):
        print("📊 SSE4 detected - Basic SIMD acceleration available")
        print("   - Use 128-bit aligned arrays")
        print("   - Batch sizes multiple of 4 (float32) or 2 (float64)")
        recommended_dtype = np.float32
        optimal_batch_multiple = 4
    else:
        print("⚠️  No advanced SIMD features detected")
        print("   - Limited acceleration available")
        print("   - Focus on algorithm optimization")
        recommended_dtype = np.float32
        optimal_batch_multiple = 1
    
    # Test optimal configurations
    print(f"\n=== Performance Testing ===")
    print(f"Recommended dtype: {recommended_dtype}")
    print(f"Optimal batch multiple: {optimal_batch_multiple}")
    
    # Test different batch sizes
    base_batch_size = 1000
    test_sizes = [
        base_batch_size,
        base_batch_size - (base_batch_size % optimal_batch_multiple),  # Aligned
        base_batch_size + optimal_batch_multiple - (base_batch_size % optimal_batch_multiple)  # Next aligned
    ]
    
    input_dim = 512
    output_dim = 256
    
    weight = np.random.randn(output_dim, input_dim).astype(recommended_dtype)
    
    print(f"Testing batch sizes: {test_sizes}")
    
    for batch_size in test_sizes:
        x = np.random.randn(batch_size, input_dim).astype(recommended_dtype)
        
        # Ensure optimal memory alignment
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x)
        if not weight.flags.c_contiguous:
            weight = np.ascontiguousarray(weight)
        
        # Benchmark
        import time
        start_time = time.time()
        for _ in range(50):
            result = optimizer.optimized_linear_forward(x, weight)
        elapsed = time.time() - start_time
        
        aligned = "✓" if batch_size % optimal_batch_multiple == 0 else "✗"
        print(f"  Batch {batch_size:4d} {aligned}: {elapsed:.4f}s")
    
    return {
        'features': features,
        'recommended_dtype': recommended_dtype,
        'optimal_batch_multiple': optimal_batch_multiple,
        'optimization_info': optimizer.get_optimization_info()
    }

# Run detailed CPU analysis
cpu_analysis = detailed_cpu_analysis()
```

## Best Practices

### 1. Memory Alignment

```python
# Ensure arrays are contiguous for optimal SIMD performance
if not array.flags.c_contiguous:
    array = np.ascontiguousarray(array)
```

### 2. Optimal Data Types

```python
# Use float32 for best SIMD performance (more elements per vector)
data = data.astype(np.float32)
```

### 3. Batch Size Optimization

```python
# Align batch sizes to SIMD vector width
def optimize_batch_size(batch_size, vector_width=8):
    return ((batch_size + vector_width - 1) // vector_width) * vector_width
```

### 4. Pre-compilation

```python
# Pre-compile optimization functions during initialization
optimizer = SIMDOptimizer()
# Functions are automatically compiled when first used
```

## Related Documentation

- [JIT Compiler Documentation](jit_compiler.md)
- [Performance Metrics Documentation](performance_metrics.md)
- [Memory Pool Documentation](memory_pool.md)
- [Training Engine Documentation](train_engine.md)
