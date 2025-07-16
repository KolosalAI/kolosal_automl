# Batch Processing Statistics (`modules/engine/batch_stats.py`)

## Overview

The Batch Statistics module provides comprehensive tracking and analysis of batch processing operations with thread-safety and performance optimizations. It enables real-time monitoring of batch processing performance, queue management, and system throughput.

## Features

- **Thread-Safe Statistics**: Safe concurrent access from multiple threads
- **Performance Metrics**: Processing time, throughput, latency tracking
- **Queue Management**: Queue length and wait time monitoring
- **Intelligent Caching**: Cached statistical calculations for performance
- **Memory Efficient**: Rolling window approach for memory management
- **Real-time Analysis**: Live performance monitoring and alerts

## Core Classes

### BatchStats

Main class for tracking batch processing statistics:

```python
class BatchStats:
    def __init__(
        self,
        window_size: int = 1000,
        cache_valid_time: float = 1.0
    )
```

**Parameters:**
- `window_size`: Number of recent batches to track (default: 1000)
- `cache_valid_time`: Cache validity duration in seconds (default: 1.0)

## Usage Examples

### Basic Statistics Tracking

```python
from modules.engine.batch_stats import BatchStats
import time

# Initialize statistics tracker
stats = BatchStats(window_size=500)

# Simulate batch processing
for i in range(100):
    start_time = time.time()
    
    # Simulate batch processing
    batch_size = 32
    queue_time = 0.05  # 50ms queue wait
    
    # Process batch (simulation)
    time.sleep(0.1)  # 100ms processing
    processing_time = time.time() - start_time
    
    # Update statistics
    stats.update(
        processing_time=processing_time,
        batch_size=batch_size,
        queue_time=queue_time,
        latency=processing_time + queue_time,
        queue_length=10
    )

# Get current statistics
current_stats = stats.get_stats()
print(f"Average processing time: {current_stats['avg_processing_time']:.3f}s")
print(f"Throughput: {current_stats['throughput']:.1f} items/sec")
print(f"Total processed: {current_stats['total_processed']} items")
```

### Advanced Performance Monitoring

```python
from modules.engine.batch_stats import BatchStats, BatchStatsMonitor
import threading
import time

# Initialize statistics tracker with larger window
stats = BatchStats(window_size=2000, cache_valid_time=0.5)

# Performance monitoring in separate thread
monitor = BatchStatsMonitor(stats, alert_threshold=0.5)
monitor_thread = threading.Thread(target=monitor.start_monitoring)
monitor_thread.daemon = True
monitor_thread.start()

# Simulate varying batch processing load
def simulate_batch_processing():
    for i in range(1000):
        # Simulate varying batch sizes and processing times
        batch_size = np.random.randint(8, 64)
        base_processing_time = 0.1
        load_factor = 1 + 0.5 * np.sin(i * 0.1)  # Varying load
        
        start_time = time.time()
        time.sleep(base_processing_time * load_factor)
        processing_time = time.time() - start_time
        
        queue_time = np.random.exponential(0.02)  # Exponential queue times
        queue_length = max(0, int(np.random.normal(15, 5)))
        
        stats.update(
            processing_time=processing_time,
            batch_size=batch_size,
            queue_time=queue_time,
            latency=processing_time + queue_time,
            queue_length=queue_length
        )
        
        time.sleep(0.01)  # Small delay between batches

# Run simulation
simulate_batch_processing()

# Get comprehensive statistics
final_stats = stats.get_comprehensive_stats()
print(f"Performance Summary:")
print(f"  Total Batches: {final_stats['total_batches']}")
print(f"  Total Items: {final_stats['total_processed']}")
print(f"  Average Throughput: {final_stats['avg_throughput']:.1f} items/sec")
print(f"  P95 Processing Time: {final_stats['p95_processing_time']:.3f}s")
print(f"  Error Rate: {final_stats['error_rate']:.2%}")
```

### Real-time Dashboard Integration

```python
from modules.engine.batch_stats import BatchStats
import json
import time

class BatchStatsDashboard:
    def __init__(self, stats: BatchStats):
        self.stats = stats
        
    def get_dashboard_data(self) -> dict:
        """Get formatted data for dashboard display"""
        stats = self.stats.get_stats()
        
        return {
            'timestamp': time.time(),
            'metrics': {
                'throughput': round(stats.get('throughput', 0), 2),
                'avg_processing_time': round(stats.get('avg_processing_time', 0), 3),
                'avg_queue_time': round(stats.get('avg_queue_time', 0), 3),
                'avg_latency': round(stats.get('avg_latency', 0), 3),
                'queue_length': round(stats.get('avg_queue_length', 0), 1),
                'total_processed': stats.get('total_processed', 0),
                'error_rate': round(stats.get('error_rate', 0), 4)
            },
            'status': 'healthy' if stats.get('error_rate', 1) < 0.01 else 'warning'
        }
    
    def export_metrics(self, format='json') -> str:
        """Export metrics in specified format"""
        data = self.get_dashboard_data()
        
        if format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'prometheus':
            metrics = data['metrics']
            return f"""
# HELP batch_throughput Current batch processing throughput
# TYPE batch_throughput gauge
batch_throughput {metrics['throughput']}

# HELP batch_processing_time_avg Average batch processing time in seconds
# TYPE batch_processing_time_avg gauge
batch_processing_time_avg {metrics['avg_processing_time']}

# HELP batch_queue_time_avg Average queue time in seconds
# TYPE batch_queue_time_avg gauge
batch_queue_time_avg {metrics['avg_queue_time']}

# HELP batch_error_rate Batch processing error rate
# TYPE batch_error_rate gauge
batch_error_rate {metrics['error_rate']}
"""

# Usage with dashboard
stats = BatchStats(window_size=1000)
dashboard = BatchStatsDashboard(stats)

# Generate dashboard data
dashboard_data = dashboard.get_dashboard_data()
prometheus_metrics = dashboard.export_metrics('prometheus')
```

## Statistical Metrics

### Core Metrics

The `BatchStats` class tracks the following core metrics:

```python
# Basic metrics
stats = batch_stats.get_stats()

print(f"Processing Time:")
print(f"  Average: {stats['avg_processing_time']:.3f}s")
print(f"  Median: {stats['median_processing_time']:.3f}s") 
print(f"  P95: {stats['p95_processing_time']:.3f}s")
print(f"  P99: {stats['p99_processing_time']:.3f}s")

print(f"Throughput:")
print(f"  Current: {stats['throughput']:.1f} items/sec")
print(f"  Peak: {stats['peak_throughput']:.1f} items/sec")

print(f"Queue Metrics:")
print(f"  Average Wait Time: {stats['avg_queue_time']:.3f}s")
print(f"  Average Queue Length: {stats['avg_queue_length']:.1f}")

print(f"Error Metrics:")
print(f"  Error Rate: {stats['error_rate']:.2%}")
print(f"  Total Errors: {stats['total_errors']}")
```

### Advanced Analytics

```python
# Get comprehensive analysis
analysis = stats.get_comprehensive_stats()

print(f"Performance Analysis:")
print(f"  Efficiency Score: {analysis['efficiency_score']:.2f}")
print(f"  Stability Index: {analysis['stability_index']:.2f}")
print(f"  Load Distribution: {analysis['load_distribution']}")

print(f"Trend Analysis:")
print(f"  Performance Trend: {analysis['performance_trend']}")
print(f"  Throughput Trend: {analysis['throughput_trend']}")
print(f"  Error Trend: {analysis['error_trend']}")

print(f"Capacity Analysis:")
print(f"  Current Utilization: {analysis['utilization']:.1%}")
print(f"  Recommended Capacity: {analysis['recommended_capacity']}")
print(f"  Bottleneck Analysis: {analysis['bottlenecks']}")
```

## Performance Optimization

### Efficient Statistics Calculation

```python
from modules.engine.batch_stats import BatchStats
import numpy as np

# Initialize with optimized settings
stats = BatchStats(
    window_size=5000,  # Larger window for better statistics
    cache_valid_time=2.0  # Longer cache time for high-frequency updates
)

# Batch updates for better performance
def batch_update_stats(stats, batch_data):
    """Efficiently update statistics with multiple batches"""
    with stats.lock:  # Single lock acquisition for multiple updates
        for data in batch_data:
            stats.processing_times.append(data['processing_time'])
            stats.batch_sizes.append(data['batch_size'])
            stats.queue_times.append(data['queue_time'])
            stats.latencies.append(data['latency'])
            stats.queue_lengths.append(data['queue_length'])
            stats.total_processed += data['batch_size']
            stats.batch_counts += 1
        
        # Invalidate cache once after all updates
        stats._last_update_time = time.monotonic()
        stats._stats_cache.clear()

# Use batch updates for better performance
batch_data = [
    {
        'processing_time': 0.15,
        'batch_size': 32,
        'queue_time': 0.03,
        'latency': 0.18,
        'queue_length': 12
    }
    # ... more batch data
]

batch_update_stats(stats, batch_data)
```

### Memory-Efficient Operation

```python
# Configure for memory efficiency
stats = BatchStats(
    window_size=1000,  # Reasonable window size
    cache_valid_time=5.0  # Longer cache to reduce computation
)

# Periodically clear old statistics
def cleanup_stats(stats, max_age_hours=24):
    """Clean up old statistics to prevent memory growth"""
    current_time = time.time()
    cutoff_time = current_time - (max_age_hours * 3600)
    
    with stats.lock:
        # Remove old entries (implementation depends on timestamp tracking)
        stats._cleanup_old_entries(cutoff_time)
        
        # Force garbage collection of cache
        stats._stats_cache.clear()

# Schedule periodic cleanup
import threading
cleanup_timer = threading.Timer(3600, cleanup_stats, args=[stats])
cleanup_timer.start()
```

## Integration with Monitoring Systems

### Prometheus Integration

```python
from prometheus_client import Gauge, Counter, Histogram
from modules.engine.batch_stats import BatchStats

class PrometheusMetrics:
    def __init__(self, stats: BatchStats):
        self.stats = stats
        
        # Define Prometheus metrics
        self.throughput_gauge = Gauge('batch_throughput', 'Batch processing throughput')
        self.processing_time_hist = Histogram('batch_processing_time', 'Batch processing time distribution')
        self.queue_time_hist = Histogram('batch_queue_time', 'Batch queue time distribution')
        self.error_counter = Counter('batch_errors_total', 'Total batch processing errors')
        self.queue_length_gauge = Gauge('batch_queue_length', 'Current batch queue length')
        
    def update_prometheus_metrics(self):
        """Update Prometheus metrics from BatchStats"""
        stats_data = self.stats.get_stats()
        
        self.throughput_gauge.set(stats_data.get('throughput', 0))
        self.queue_length_gauge.set(stats_data.get('avg_queue_length', 0))
        
        # Update histograms with recent samples
        recent_processing_times = list(self.stats.processing_times)[-100:]
        for pt in recent_processing_times:
            self.processing_time_hist.observe(pt)
        
        recent_queue_times = list(self.stats.queue_times)[-100:]
        for qt in recent_queue_times:
            self.queue_time_hist.observe(qt)

# Usage
stats = BatchStats()
prometheus_metrics = PrometheusMetrics(stats)

# Periodically update Prometheus metrics
def update_metrics_loop():
    while True:
        prometheus_metrics.update_prometheus_metrics()
        time.sleep(30)  # Update every 30 seconds

metrics_thread = threading.Thread(target=update_metrics_loop)
metrics_thread.daemon = True
metrics_thread.start()
```

### Custom Alerting

```python
from modules.engine.batch_stats import BatchStats
import logging

class BatchStatsAlerting:
    def __init__(self, stats: BatchStats):
        self.stats = stats
        self.alert_thresholds = {
            'max_processing_time': 1.0,  # 1 second
            'min_throughput': 100,       # items/sec
            'max_error_rate': 0.05,      # 5%
            'max_queue_length': 50       # items
        }
        
    def check_alerts(self):
        """Check for alert conditions"""
        stats_data = self.stats.get_stats()
        alerts = []
        
        # Processing time alert
        if stats_data.get('avg_processing_time', 0) > self.alert_thresholds['max_processing_time']:
            alerts.append({
                'type': 'performance',
                'message': f"High processing time: {stats_data['avg_processing_time']:.3f}s",
                'severity': 'warning'
            })
        
        # Throughput alert
        if stats_data.get('throughput', 0) < self.alert_thresholds['min_throughput']:
            alerts.append({
                'type': 'throughput',
                'message': f"Low throughput: {stats_data['throughput']:.1f} items/sec",
                'severity': 'warning'
            })
        
        # Error rate alert
        if stats_data.get('error_rate', 0) > self.alert_thresholds['max_error_rate']:
            alerts.append({
                'type': 'error',
                'message': f"High error rate: {stats_data['error_rate']:.2%}",
                'severity': 'critical'
            })
        
        # Queue length alert
        if stats_data.get('avg_queue_length', 0) > self.alert_thresholds['max_queue_length']:
            alerts.append({
                'type': 'queue',
                'message': f"High queue length: {stats_data['avg_queue_length']:.1f}",
                'severity': 'warning'
            })
        
        return alerts
    
    def process_alerts(self, alerts):
        """Process and send alerts"""
        for alert in alerts:
            if alert['severity'] == 'critical':
                logging.critical(f"CRITICAL ALERT: {alert['message']}")
                # Send to pager/SMS/Slack
                self.send_critical_alert(alert)
            else:
                logging.warning(f"WARNING: {alert['message']}")
                # Send to monitoring dashboard
                self.send_warning_alert(alert)

# Usage
stats = BatchStats()
alerting = BatchStatsAlerting(stats)

# Periodic alert checking
def alert_monitoring_loop():
    while True:
        alerts = alerting.check_alerts()
        if alerts:
            alerting.process_alerts(alerts)
        time.sleep(60)  # Check every minute

alert_thread = threading.Thread(target=alert_monitoring_loop)
alert_thread.daemon = True
alert_thread.start()
```

## Best Practices

### 1. Appropriate Window Sizes
```python
# For real-time monitoring (short window)
realtime_stats = BatchStats(window_size=100)

# For trend analysis (medium window)
trend_stats = BatchStats(window_size=1000)

# For long-term analysis (large window)
longterm_stats = BatchStats(window_size=10000)
```

### 2. Efficient Updates
```python
# Good: Batch multiple updates
with stats.lock:
    for batch in recent_batches:
        stats.update(batch.processing_time, batch.size, batch.queue_time)

# Avoid: Frequent individual updates without batching
for batch in recent_batches:
    stats.update(batch.processing_time, batch.size, batch.queue_time)  # Multiple lock acquisitions
```

### 3. Resource Management
```python
# Monitor memory usage
def monitor_memory_usage(stats):
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 500:  # 500MB threshold
        logging.warning(f"High memory usage: {memory_mb:.1f}MB")
        # Consider reducing window size or clearing cache
        stats._stats_cache.clear()

# Periodic memory monitoring
timer = threading.Timer(300, monitor_memory_usage, args=[stats])  # Every 5 minutes
timer.start()
```

## Related Documentation

- [Batch Processor Documentation](batch_processor.md)
- [Dynamic Batcher Documentation](dynamic_batcher.md)
- [Performance Metrics Documentation](performance_metrics.md)
- [Inference Engine Documentation](inference_engine.md)
