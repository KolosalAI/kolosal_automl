# Performance Metrics (`modules/engine/performance_metrics.py`)

## Overview

The Performance Metrics module provides thread-safe performance monitoring capabilities with comprehensive metric collection and statistical analysis. It's designed for real-time monitoring of inference operations, training processes, and system performance.

## Features

- **Thread-Safe Metrics**: Safe concurrent access from multiple threads
- **Real-Time Monitoring**: Live performance tracking and updates
- **Statistical Analysis**: Comprehensive statistical calculations
- **Memory Efficient**: Rolling window approach for metric storage
- **Configurable Windows**: Adjustable time windows for different metrics
- **Export Capabilities**: Multiple export formats for monitoring systems

## Core Classes

### PerformanceMetrics

Main performance metrics tracker with thread-safe operations:

```python
class PerformanceMetrics:
    def __init__(
        self,
        window_size: int = 100,
        enable_detailed_stats: bool = True,
        track_percentiles: bool = True
    )
```

**Parameters:**
- `window_size`: Number of recent measurements to track
- `enable_detailed_stats`: Enable detailed statistical calculations
- `track_percentiles`: Track percentile statistics (P50, P95, P99)

## Usage Examples

### Basic Performance Tracking

```python
from modules.engine.performance_metrics import PerformanceMetrics
import time
import threading
import numpy as np

# Initialize performance metrics tracker
metrics = PerformanceMetrics(
    window_size=1000,
    enable_detailed_stats=True,
    track_percentiles=True
)

# Example: Tracking inference performance
def simulate_inference_workload():
    """Simulate ML inference with performance tracking"""
    
    for i in range(100):
        # Simulate preprocessing
        preprocessing_start = time.time()
        time.sleep(0.001)  # 1ms preprocessing
        preprocessing_time = time.time() - preprocessing_start
        
        # Simulate inference
        inference_start = time.time()
        batch_size = np.random.randint(1, 64)  # Variable batch size
        time.sleep(0.01 * batch_size / 32)  # Inference time scales with batch size
        inference_time = time.time() - inference_start
        
        # Simulate quantization (optional)
        quantization_time = 0.0
        if np.random.random() > 0.5:  # 50% of requests use quantization
            quantization_start = time.time()
            time.sleep(0.0005)  # 0.5ms quantization
            quantization_time = time.time() - quantization_start
        
        # Update metrics
        metrics.update_inference(
            inference_time=inference_time,
            batch_size=batch_size,
            preprocessing_time=preprocessing_time,
            quantization_time=quantization_time
        )
        
        # Occasionally record errors
        if np.random.random() < 0.01:  # 1% error rate
            metrics.record_error()
        
        # Record cache hits/misses
        if np.random.random() > 0.2:  # 80% cache hit rate
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()

# Run simulation
simulate_inference_workload()

# Get performance statistics
stats = metrics.get_statistics()
print("Performance Statistics:")
print(f"  Total requests: {stats['total_requests']}")
print(f"  Average inference time: {stats['avg_inference_time']:.4f}s")
print(f"  Median inference time: {stats['median_inference_time']:.4f}s")
print(f"  P95 inference time: {stats['p95_inference_time']:.4f}s")
print(f"  P99 inference time: {stats['p99_inference_time']:.4f}s")
print(f"  Throughput: {stats['throughput']:.1f} requests/sec")
print(f"  Error rate: {stats['error_rate']:.3%}")
print(f"  Cache hit rate: {stats['cache_hit_rate']:.3%}")
```

### Multi-Threaded Performance Monitoring

```python
from modules.engine.performance_metrics import PerformanceMetrics
import threading
import time
import random

# Shared metrics instance
shared_metrics = PerformanceMetrics(
    window_size=5000,
    enable_detailed_stats=True
)

def worker_thread(thread_id, duration_seconds=30):
    """Worker thread that generates load and tracks metrics"""
    print(f"Worker {thread_id} started")
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        # Simulate varying workload
        batch_size = random.randint(1, 128)
        
        # Simulate inference with varying complexity
        complexity_factor = random.uniform(0.5, 2.0)
        inference_time = 0.01 * complexity_factor * (batch_size / 32)
        
        # Simulate actual work
        time.sleep(inference_time)
        
        # Update shared metrics (thread-safe)
        shared_metrics.update_inference(
            inference_time=inference_time,
            batch_size=batch_size,
            preprocessing_time=random.uniform(0.001, 0.005),
            quantization_time=random.uniform(0.0, 0.002)
        )
        
        # Random errors and cache events
        if random.random() < 0.02:  # 2% error rate
            shared_metrics.record_error()
        
        if random.random() > 0.15:  # 85% cache hit rate
            shared_metrics.record_cache_hit()
        else:
            shared_metrics.record_cache_miss()
        
        # Small delay between requests
        time.sleep(random.uniform(0.001, 0.01))
    
    print(f"Worker {thread_id} finished")

# Start multiple worker threads
threads = []
num_workers = 8

for i in range(num_workers):
    thread = threading.Thread(target=worker_thread, args=(i, 60))  # 60 seconds each
    threads.append(thread)
    thread.start()

# Monitor performance in real-time
def performance_monitor():
    """Monitor and report performance every 5 seconds"""
    for _ in range(12):  # Monitor for 60 seconds (12 * 5 seconds)
        time.sleep(5)
        
        stats = shared_metrics.get_statistics()
        print(f"\n=== Performance Update ===")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Current throughput: {stats['current_throughput']:.1f} req/sec")
        print(f"Average latency: {stats['avg_total_time']:.4f}s")
        print(f"Error rate: {stats['error_rate']:.2%}")
        print(f"Workers active: {sum(1 for t in threads if t.is_alive())}")

# Start monitoring
monitor_thread = threading.Thread(target=performance_monitor)
monitor_thread.start()

# Wait for all workers to complete
for thread in threads:
    thread.join()

monitor_thread.join()

# Final performance report
final_stats = shared_metrics.get_comprehensive_statistics()
print("\n=== Final Performance Report ===")
print(f"Total runtime: {final_stats['total_runtime']:.1f} seconds")
print(f"Total requests processed: {final_stats['total_requests']}")
print(f"Average throughput: {final_stats['average_throughput']:.1f} req/sec")
print(f"Peak throughput: {final_stats['peak_throughput']:.1f} req/sec")
print(f"Overall error rate: {final_stats['error_rate']:.3%}")
print(f"Performance stability: {final_stats['stability_score']:.2f}")
```

### Performance Analytics and Reporting

```python
from modules.engine.performance_metrics import PerformanceAnalyzer
import matplotlib.pyplot as plt
import pandas as pd

# Create performance analyzer
analyzer = PerformanceAnalyzer(shared_metrics)

# Generate comprehensive performance analysis
analysis = analyzer.analyze_performance_trends()

print("Performance Trend Analysis:")
print(f"  Throughput trend: {analysis['throughput_trend']}")  # increasing/decreasing/stable
print(f"  Latency trend: {analysis['latency_trend']}")
print(f"  Error trend: {analysis['error_trend']}")
print(f"  System stability: {analysis['stability_rating']}")

# Performance bottleneck analysis
bottlenecks = analyzer.identify_bottlenecks()
print("\nBottleneck Analysis:")
for bottleneck in bottlenecks:
    print(f"  {bottleneck['component']}: {bottleneck['impact']} impact")
    print(f"    Recommendation: {bottleneck['recommendation']}")

# Generate performance dashboard data
dashboard_data = analyzer.generate_dashboard_data()

# Export to different formats
# JSON for APIs
json_export = analyzer.export_json()
print(f"JSON export size: {len(json_export)} characters")

# CSV for analysis
csv_export = analyzer.export_csv()
print(f"CSV export: {len(csv_export.split(','))} columns")

# Prometheus metrics
prometheus_export = analyzer.export_prometheus()
print("Prometheus metrics:")
print(prometheus_export[:500] + "..." if len(prometheus_export) > 500 else prometheus_export)
```

### Real-Time Performance Alerts

```python
from modules.engine.performance_metrics import PerformanceAlerting
import logging

# Configure alerting system
alerting = PerformanceAlerting(
    metrics=shared_metrics,
    alert_thresholds={
        'error_rate_threshold': 0.05,       # 5% error rate
        'latency_p95_threshold': 0.1,       # 100ms P95 latency
        'throughput_min_threshold': 100,    # 100 req/sec minimum
        'cache_hit_rate_threshold': 0.8     # 80% cache hit rate
    },
    alert_window_size=50,  # Check last 50 measurements
    cooldown_period=300    # 5 minutes between same alert types
)

# Custom alert handlers
def critical_error_handler(alert):
    """Handle critical performance alerts"""
    logging.critical(f"CRITICAL ALERT: {alert['message']}")
    # Send to monitoring system, page ops team, etc.
    print(f"🚨 CRITICAL: {alert['message']}")
    print(f"   Current value: {alert['current_value']}")
    print(f"   Threshold: {alert['threshold']}")

def warning_handler(alert):
    """Handle warning alerts"""
    logging.warning(f"WARNING: {alert['message']}")
    print(f"⚠️  WARNING: {alert['message']}")

def info_handler(alert):
    """Handle informational alerts"""
    logging.info(f"INFO: {alert['message']}")
    print(f"ℹ️  INFO: {alert['message']}")

# Register alert handlers
alerting.register_handler('critical', critical_error_handler)
alerting.register_handler('warning', warning_handler)
alerting.register_handler('info', info_handler)

# Start alert monitoring
alerting.start_monitoring()

# Simulate performance degradation
def simulate_performance_issues():
    """Simulate various performance issues to trigger alerts"""
    
    # Simulate high error rate
    for _ in range(20):
        shared_metrics.record_error()
        time.sleep(0.1)
    
    # Simulate slow requests
    for _ in range(10):
        shared_metrics.update_inference(
            inference_time=0.2,  # Very slow
            batch_size=1,
            preprocessing_time=0.05,
            quantization_time=0.01
        )
        time.sleep(0.1)
    
    # Simulate cache misses
    for _ in range(30):
        shared_metrics.record_cache_miss()
        time.sleep(0.05)

# Run performance issue simulation
simulate_performance_issues()

# Stop alert monitoring
time.sleep(2)  # Let alerts process
alerting.stop_monitoring()

# Get alert summary
alert_summary = alerting.get_alert_summary()
print("\nAlert Summary:")
print(f"  Total alerts: {alert_summary['total_alerts']}")
print(f"  Critical alerts: {alert_summary['critical_alerts']}")
print(f"  Warning alerts: {alert_summary['warning_alerts']}")
print(f"  Most frequent alert: {alert_summary['most_frequent_alert']}")
```

## Advanced Analytics

### Performance Profiling

```python
from modules.engine.performance_metrics import PerformanceProfiler
import cProfile
import pstats

# Create performance profiler
profiler = PerformanceProfiler(shared_metrics)

# Profile a function
@profiler.profile_function
def complex_ml_operation(data_size=10000):
    """Complex ML operation for profiling"""
    import numpy as np
    
    # Simulate feature extraction
    features = np.random.rand(data_size, 100)
    
    # Simulate model inference
    weights = np.random.rand(100, 50)
    intermediate = np.dot(features, weights)
    
    # Simulate post-processing
    result = np.mean(intermediate, axis=1)
    
    return result

# Run profiled operation
result = complex_ml_operation(50000)

# Get profiling results
profile_results = profiler.get_profile_results('complex_ml_operation')
print("Profiling Results:")
print(f"  Function calls: {profile_results['total_calls']}")
print(f"  Total time: {profile_results['total_time']:.4f}s")
print(f"  Average time per call: {profile_results['avg_time_per_call']:.6f}s")
print(f"  Time breakdown:")
for component, time_spent in profile_results['time_breakdown'].items():
    print(f"    {component}: {time_spent:.4f}s ({time_spent/profile_results['total_time']:.1%})")

# Generate detailed profile report
profiler.generate_detailed_report("performance_profile.html")
```

### Custom Metrics

```python
from modules.engine.performance_metrics import CustomMetrics

# Create custom metrics tracker
custom_metrics = CustomMetrics(shared_metrics)

# Define custom metrics
custom_metrics.define_metric(
    name="model_accuracy",
    metric_type="gauge",
    description="Current model accuracy",
    unit="percentage"
)

custom_metrics.define_metric(
    name="data_freshness",
    metric_type="histogram",
    description="Age of training data",
    unit="hours",
    buckets=[1, 6, 12, 24, 48, 168]  # 1h, 6h, 12h, 1d, 2d, 1w
)

custom_metrics.define_metric(
    name="feature_drift",
    metric_type="counter",
    description="Feature drift detection events",
    labels=["feature_name", "drift_type"]
)

# Update custom metrics
custom_metrics.update_gauge("model_accuracy", 0.945)
custom_metrics.observe_histogram("data_freshness", 8.5)  # 8.5 hours
custom_metrics.increment_counter("feature_drift", labels={"feature_name": "age", "drift_type": "statistical"})

# Get custom metrics
custom_stats = custom_metrics.get_all_metrics()
print("Custom Metrics:")
for metric_name, metric_data in custom_stats.items():
    print(f"  {metric_name}: {metric_data}")
```

## Integration with Monitoring Systems

### Prometheus Integration

```python
from modules.engine.performance_metrics import PrometheusExporter
from prometheus_client import start_http_server

# Create Prometheus exporter
prometheus_exporter = PrometheusExporter(shared_metrics)

# Start Prometheus metrics server
start_http_server(8000)
print("Prometheus metrics available at http://localhost:8000/metrics")

# The exporter automatically creates these metrics:
# - inference_duration_seconds (histogram)
# - batch_size_total (histogram) 
# - requests_total (counter)
# - errors_total (counter)
# - cache_hits_total (counter)
# - cache_misses_total (counter)
# - throughput_requests_per_second (gauge)

# Custom Prometheus metrics
prometheus_exporter.add_custom_metric(
    "model_memory_usage_bytes",
    "Memory usage of loaded models",
    metric_type="gauge"
)

# Update custom metrics
prometheus_exporter.set_custom_metric("model_memory_usage_bytes", 1024*1024*512)  # 512MB
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "ML Performance Metrics",
    "panels": [
      {
        "title": "Request Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time Percentiles",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.50, inference_duration_seconds)",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, inference_duration_seconds)", 
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, inference_duration_seconds)",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(errors_total[5m]) / rate(requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

### 1. Metric Window Sizing

```python
# Choose appropriate window sizes based on use case
realtime_metrics = PerformanceMetrics(window_size=100)    # Real-time monitoring
shortterm_metrics = PerformanceMetrics(window_size=1000)  # Short-term trends  
longterm_metrics = PerformanceMetrics(window_size=10000)  # Long-term analysis
```

### 2. Thread Safety

```python
# Always use thread-safe updates in multi-threaded environments
def safe_metric_update(metrics, inference_time, batch_size):
    """Thread-safe metric update with error handling"""
    try:
        metrics.update_inference(inference_time, batch_size)
    except Exception as e:
        logging.error(f"Failed to update metrics: {e}")
        # Don't let metric failures break the main application
```

### 3. Performance Impact Minimization

```python
# Minimize performance impact of metrics collection
class LightweightMetrics(PerformanceMetrics):
    """Lightweight metrics for high-frequency operations"""
    
    def __init__(self):
        super().__init__(
            window_size=50,           # Smaller window
            enable_detailed_stats=False,  # Disable expensive calculations
            track_percentiles=False       # Disable percentile tracking
        )
    
    def quick_update(self, inference_time):
        """Minimal overhead update"""
        with self.lock:
            if len(self.inference_times) >= self.window_size:
                self.inference_times.pop(0)
            self.inference_times.append(inference_time)
            self.total_requests += 1
```

## Related Documentation

- [Inference Engine Documentation](inference_engine.md)
- [Batch Processor Documentation](batch_processor.md)
- [Experiment Tracker Documentation](experiment_tracker.md)
- [JIT Compiler Documentation](jit_compiler.md)
- [Training Engine Documentation](train_engine.md)
