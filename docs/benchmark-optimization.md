# Optimized Benchmark System

This document describes the enhanced benchmark system that automatically applies optimal device configuration for maximum performance during benchmark execution.

## Overview

The optimized benchmark system provides:

- **Automatic Hardware Detection**: Detects CPU capabilities, memory, and specialized accelerators
- **Optimal Configuration**: Applies performance-optimized settings based on system capabilities
- **Threading Optimization**: Configures optimal thread counts for ML operations
- **Memory Management**: Optimizes memory allocation and usage patterns
- **Process Priority**: Sets high process priority for benchmark execution
- **Environment Control**: Manages environment variables for optimal performance

## Quick Start

### Basic Usage

```bash
# Run all benchmarks with automatic optimization
python run_optimized_benchmarks.py

# Run specific category with optimization
python run_optimized_benchmarks.py --category ml

# Run quick benchmarks with optimization
python run_optimized_benchmarks.py --quick
```

### Advanced Usage

```bash
# Skip optimization (use default settings)
python run_optimized_benchmarks.py --no-optimize

# Apply optimization manually
python scripts/benchmark_optimize.py

# View system information
python scripts/benchmark_optimize.py --info-only

# Restore environment after benchmarks
python scripts/benchmark_optimize.py --restore
```

## Architecture

### Components

1. **DeviceOptimizer**: Core optimization engine that detects hardware and generates optimal configurations
2. **BenchmarkOptimizer**: System-level optimization utility that applies CPU, memory, and process optimizations
3. **Enhanced Fixtures**: Pytest fixtures that provide optimal device configuration to tests
4. **Optimized Runners**: Enhanced benchmark runners that apply optimizations automatically

### Optimization Modes

- **PERFORMANCE**: Maximum performance, uses all available resources
- **BALANCED**: Balanced performance and resource usage
- **MEMORY_SAVING**: Optimized for memory-constrained environments
- **CONSERVATIVE**: Safe settings for stable execution

## Key Optimizations Applied

### CPU Optimizations

- **Thread Configuration**: Sets optimal thread counts for ML libraries (OpenMP, MKL, OpenBLAS)
- **CPU Affinity**: Binds process to all available CPU cores
- **SIMD Instructions**: Leverages AVX, AVX2, AVX512, and NEON instructions where available
- **Hardware Accelerators**: Detects and utilizes Intel MKL, Intel IPEX, and ARM NEON

### Memory Optimizations

- **Memory Allocation**: Configures malloc settings for performance
- **Memory Reservation**: Reserves optimal amount of memory for benchmarks
- **Garbage Collection**: Strategic garbage collection timing
- **Memory Pool**: Optimizes memory pool sizes

### Process Optimizations

- **Process Priority**: Sets high priority for benchmark processes
- **Environment Variables**: Configures performance-critical environment variables
- **Resource Limits**: Adjusts resource limits for optimal performance

## Configuration Examples

### Example System Output

```
🔧 Configuring optimal device settings for benchmarks...
✅ Optimal configuration applied:
   CPU cores: 16 (physical: 8)
   Memory: 32.0 GB
   Accelerators: ['INTEL_MKL', 'INTEL_IPEX']

Applied optimizations:
  • CPU threading optimized for 8 cores
  • CPU affinity set to use all available cores
  • Memory optimized (28.5GB available)
  • Process priority set to HIGH
  • Hardware accelerators detected: ['INTEL_MKL', 'INTEL_IPEX']
  • CPU features enabled: ['avx', 'avx2', 'avx512', 'fma']
  • Device optimizer configuration applied
```

### Generated Configuration

The system automatically generates optimal configurations for:

- **Quantization**: INT8/FP16 precision based on hardware capabilities
- **Batch Processing**: Optimal batch sizes and worker counts
- **Inference Engine**: Optimized for throughput and latency
- **Training Engine**: Parallel processing and memory optimization

## Performance Improvements

Expected performance improvements with optimization:

- **CPU-bound workloads**: 40-80% faster execution
- **Memory-intensive tasks**: 25-50% better memory utilization
- **Multi-threaded operations**: 60-120% improvement with proper thread configuration
- **Inference throughput**: 30-70% higher predictions per second

## Environment Variables Set

The optimization system configures these environment variables:

```bash
# Threading Configuration
OMP_NUM_THREADS=8              # OpenMP thread count
MKL_NUM_THREADS=8              # Intel MKL thread count
OPENBLAS_NUM_THREADS=8         # OpenBLAS thread count
NUMEXPR_NUM_THREADS=8          # NumExpr thread count
VECLIB_MAXIMUM_THREADS=8       # macOS Accelerate framework

# Performance Tuning
OMP_DYNAMIC=FALSE              # Disable dynamic threading
MKL_DYNAMIC=FALSE              # Disable dynamic MKL threading
MALLOC_TRIM_THRESHOLD_=0       # Disable malloc trimming
MALLOC_MMAP_THRESHOLD_=1048576 # Use mmap for large allocations
```

## Hardware Support

### CPU Architectures

- **x86_64**: Intel and AMD processors with AVX/AVX2/AVX512 support
- **ARM64**: ARM processors with NEON instruction support
- **Apple Silicon**: M1/M2 processors with optimized Accelerate framework

### Specialized Accelerators

- **Intel MKL**: Math Kernel Library for Intel CPUs
- **Intel IPEX**: Intel Extension for PyTorch
- **ARM NEON**: ARM Advanced SIMD instructions
- **ONNX Runtime**: Cross-platform inference optimization

## Monitoring and Debugging

### Performance Metrics

The system collects comprehensive performance metrics:

- **Execution Time**: Detailed timing for each benchmark
- **Memory Usage**: Peak and delta memory consumption
- **CPU Utilization**: CPU usage patterns during execution
- **Throughput**: Operations per second for different workloads

### Debug Information

Enable verbose output to see detailed optimization information:

```bash
python run_optimized_benchmarks.py --verbose
```

### Log Files

Benchmark results and optimization details are saved to:

- `tests/benchmark/results/benchmark_results_*.json`
- `tests/test.log`
- `logs/device_optimizer.log`

## Best Practices

### Before Running Benchmarks

1. **Close unnecessary applications** to free up system resources
2. **Ensure stable power supply** (disable power saving on laptops)
3. **Cool down the system** if it has been under heavy load
4. **Run on dedicated hardware** when possible for consistent results

### During Benchmarks

1. **Avoid system usage** while benchmarks are running
2. **Monitor system temperature** to prevent thermal throttling
3. **Ensure adequate disk space** for result files

### After Benchmarks

1. **Restore environment** using `--restore` flag or `--restore-only`
2. **Analyze results** using generated JSON reports
3. **Compare across runs** to identify performance trends

## Troubleshooting

### Common Issues

1. **Permission Errors**: Run with elevated privileges if needed for process priority
2. **Memory Errors**: Reduce batch sizes or enable memory optimization mode
3. **Threading Issues**: Check for conflicting thread configurations
4. **Missing Dependencies**: Ensure all required libraries are installed

### Performance Issues

1. **Thermal Throttling**: Monitor CPU temperature and ensure adequate cooling
2. **Memory Pressure**: Close other applications or increase available memory
3. **Disk I/O**: Use SSD storage for better performance
4. **Background Processes**: Disable unnecessary services during benchmarks

## API Reference

### BenchmarkOptimizer Class

```python
from scripts.benchmark_optimize import BenchmarkOptimizer

optimizer = BenchmarkOptimizer()

# Apply all optimizations
result = optimizer.optimize_for_benchmarks()

# Restore original settings
optimizer.restore_environment()

# Get system information
system_info = optimizer.get_system_summary()
```

### DeviceOptimizer Integration

```python
from modules.device_optimizer import DeviceOptimizer, OptimizationMode

# Create performance-optimized configuration
optimizer = DeviceOptimizer(
    optimization_mode=OptimizationMode.PERFORMANCE,
    workload_type="inference",
    enable_specialized_accelerators=True
)

# Get optimal configurations
inference_config = optimizer.get_optimal_inference_engine_config()
batch_config = optimizer.get_optimal_batch_processor_config()
```

## Contributing

When adding new benchmarks:

1. **Use the optimal_device_config fixture** to access optimized settings
2. **Apply n_jobs configuration** for scikit-learn models
3. **Include optimization metadata** in benchmark results
4. **Test across different optimization modes**
5. **Document performance expectations**

Example benchmark function:

```python
@pytest.mark.benchmark
def test_my_benchmark(benchmark_result, optimal_device_config):
    benchmark_result.start()
    
    # Get optimal configuration
    optimal_n_jobs = -1
    if optimal_device_config:
        batch_config = optimal_device_config.get('batch_processor_config')
        if batch_config and hasattr(batch_config, 'num_workers'):
            optimal_n_jobs = batch_config.num_workers
    
    # Use optimal configuration in your code
    model = RandomForestClassifier(n_jobs=optimal_n_jobs)
    
    # ... benchmark code ...
    
    benchmark_result.stop()
    benchmark_result.metadata['optimization_config'] = {
        'n_jobs_used': optimal_n_jobs,
        'device_config_available': optimal_device_config is not None
    }
```
