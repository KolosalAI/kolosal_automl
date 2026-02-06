//! Performance Metrics
//!
//! Latency, throughput, and resource usage tracking.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Type of metric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Latency in milliseconds
    Latency,
    /// Throughput (items per second)
    Throughput,
    /// Error rate
    ErrorRate,
    /// Queue size
    QueueSize,
    /// Memory usage in bytes
    MemoryUsage,
    /// CPU usage percentage
    CpuUsage,
    /// Custom metric
    Custom,
}

/// Histogram bucket for latency distribution
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Upper bound of this bucket (in milliseconds)
    pub le: f64,
    /// Count of observations in this bucket
    pub count: u64,
}

/// Performance metrics collector
pub struct PerformanceMetrics {
    /// Window size for rolling metrics
    window_size: usize,
    
    // Latency tracking
    latencies: RwLock<VecDeque<f64>>,
    latency_histogram: RwLock<Vec<HistogramBucket>>,
    
    // Counters
    total_requests: AtomicU64,
    total_errors: AtomicU64,
    total_items: AtomicU64,
    
    // Timing
    start_time: Instant,
    last_reset: RwLock<Instant>,
    
    // Throughput tracking
    throughput_samples: RwLock<VecDeque<(Instant, u64)>>,
}

impl PerformanceMetrics {
    /// Create a new metrics collector
    pub fn new(window_size: usize) -> Self {
        // Default histogram buckets (in milliseconds)
        let histogram_buckets = vec![
            HistogramBucket { le: 1.0, count: 0 },
            HistogramBucket { le: 5.0, count: 0 },
            HistogramBucket { le: 10.0, count: 0 },
            HistogramBucket { le: 25.0, count: 0 },
            HistogramBucket { le: 50.0, count: 0 },
            HistogramBucket { le: 100.0, count: 0 },
            HistogramBucket { le: 250.0, count: 0 },
            HistogramBucket { le: 500.0, count: 0 },
            HistogramBucket { le: 1000.0, count: 0 },
            HistogramBucket { le: f64::INFINITY, count: 0 },
        ];
        
        Self {
            window_size,
            latencies: RwLock::new(VecDeque::with_capacity(window_size)),
            latency_histogram: RwLock::new(histogram_buckets),
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_items: AtomicU64::new(0),
            start_time: Instant::now(),
            last_reset: RwLock::new(Instant::now()),
            throughput_samples: RwLock::new(VecDeque::with_capacity(100)),
        }
    }
    
    /// Record a latency observation
    pub fn record_latency(&self, latency_ms: f64) {
        // Add to rolling window
        if let Ok(mut latencies) = self.latencies.write() {
            latencies.push_back(latency_ms);
            if latencies.len() > self.window_size {
                latencies.pop_front();
            }
        }
        
        // Update histogram
        if let Ok(mut histogram) = self.latency_histogram.write() {
            for bucket in histogram.iter_mut() {
                if latency_ms <= bucket.le {
                    bucket.count += 1;
                    break;
                }
            }
        }
        
        self.total_requests.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Record a batch of items processed
    pub fn record_items(&self, count: u64) {
        self.total_items.fetch_add(count, Ordering::SeqCst);
        
        if let Ok(mut samples) = self.throughput_samples.write() {
            samples.push_back((Instant::now(), count));
            // Keep last 100 samples
            while samples.len() > 100 {
                samples.pop_front();
            }
        }
    }
    
    /// Record an error
    pub fn record_error(&self) {
        self.total_errors.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Get the average latency
    pub fn avg_latency(&self) -> f64 {
        self.latencies.read()
            .map(|l| {
                if l.is_empty() {
                    0.0
                } else {
                    l.iter().sum::<f64>() / l.len() as f64
                }
            })
            .unwrap_or(0.0)
    }
    
    /// Get the minimum latency
    pub fn min_latency(&self) -> f64 {
        self.latencies.read()
            .map(|l| l.iter().copied().fold(f64::INFINITY, f64::min))
            .unwrap_or(0.0)
    }
    
    /// Get the maximum latency
    pub fn max_latency(&self) -> f64 {
        self.latencies.read()
            .map(|l| l.iter().copied().fold(0.0, f64::max))
            .unwrap_or(0.0)
    }
    
    /// Get a percentile latency (e.g., p50, p95, p99)
    pub fn percentile_latency(&self, percentile: f64) -> f64 {
        self.latencies.read()
            .map(|l| {
                if l.is_empty() {
                    return 0.0;
                }
                
                let mut sorted: Vec<f64> = l.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                let idx = ((percentile / 100.0) * (sorted.len() - 1) as f64) as usize;
                sorted[idx]
            })
            .unwrap_or(0.0)
    }
    
    /// Get P50 latency
    pub fn p50_latency(&self) -> f64 {
        self.percentile_latency(50.0)
    }
    
    /// Get P95 latency
    pub fn p95_latency(&self) -> f64 {
        self.percentile_latency(95.0)
    }
    
    /// Get P99 latency
    pub fn p99_latency(&self) -> f64 {
        self.percentile_latency(99.0)
    }
    
    /// Get the current throughput (items per second)
    pub fn throughput(&self) -> f64 {
        self.throughput_samples.read()
            .map(|samples| {
                if samples.len() < 2 {
                    return 0.0;
                }
                
                let first = samples.front().unwrap();
                let last = samples.back().unwrap();
                
                let duration = last.0.duration_since(first.0).as_secs_f64();
                if duration <= 0.0 {
                    return 0.0;
                }
                
                let total_items: u64 = samples.iter().map(|(_, c)| c).sum();
                total_items as f64 / duration
            })
            .unwrap_or(0.0)
    }
    
    /// Get the error rate
    pub fn error_rate(&self) -> f64 {
        let requests = self.total_requests.load(Ordering::SeqCst);
        let errors = self.total_errors.load(Ordering::SeqCst);
        
        if requests > 0 {
            errors as f64 / requests as f64
        } else {
            0.0
        }
    }
    
    /// Get the latency histogram
    pub fn histogram(&self) -> Vec<HistogramBucket> {
        self.latency_histogram.read()
            .map(|h| h.clone())
            .unwrap_or_default()
    }
    
    /// Get total requests
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::SeqCst)
    }
    
    /// Get total errors
    pub fn total_errors(&self) -> u64 {
        self.total_errors.load(Ordering::SeqCst)
    }
    
    /// Get total items processed
    pub fn total_items(&self) -> u64 {
        self.total_items.load(Ordering::SeqCst)
    }
    
    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
    
    /// Get a summary of all metrics
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_requests: self.total_requests(),
            total_errors: self.total_errors(),
            total_items: self.total_items(),
            error_rate: self.error_rate(),
            avg_latency_ms: self.avg_latency(),
            min_latency_ms: self.min_latency(),
            max_latency_ms: self.max_latency(),
            p50_latency_ms: self.p50_latency(),
            p95_latency_ms: self.p95_latency(),
            p99_latency_ms: self.p99_latency(),
            throughput: self.throughput(),
            uptime_secs: self.uptime_secs(),
        }
    }
    
    /// Reset all metrics
    pub fn reset(&self) {
        if let Ok(mut latencies) = self.latencies.write() {
            latencies.clear();
        }
        if let Ok(mut histogram) = self.latency_histogram.write() {
            for bucket in histogram.iter_mut() {
                bucket.count = 0;
            }
        }
        if let Ok(mut samples) = self.throughput_samples.write() {
            samples.clear();
        }
        if let Ok(mut last_reset) = self.last_reset.write() {
            *last_reset = Instant::now();
        }
        
        self.total_requests.store(0, Ordering::SeqCst);
        self.total_errors.store(0, Ordering::SeqCst);
        self.total_items.store(0, Ordering::SeqCst);
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Summary of performance metrics
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    /// Total requests
    pub total_requests: u64,
    /// Total errors
    pub total_errors: u64,
    /// Total items processed
    pub total_items: u64,
    /// Error rate
    pub error_rate: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Minimum latency in milliseconds
    pub min_latency_ms: f64,
    /// Maximum latency in milliseconds
    pub max_latency_ms: f64,
    /// P50 latency in milliseconds
    pub p50_latency_ms: f64,
    /// P95 latency in milliseconds
    pub p95_latency_ms: f64,
    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,
    /// Throughput (items per second)
    pub throughput: f64,
    /// Uptime in seconds
    pub uptime_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_basic() {
        let metrics = PerformanceMetrics::new(100);
        
        metrics.record_latency(10.0);
        metrics.record_latency(20.0);
        metrics.record_latency(30.0);
        
        assert_eq!(metrics.total_requests(), 3);
        assert!((metrics.avg_latency() - 20.0).abs() < 0.01);
    }
    
    #[test]
    fn test_percentiles() {
        let metrics = PerformanceMetrics::new(100);
        
        for i in 1..=100 {
            metrics.record_latency(i as f64);
        }
        
        assert!((metrics.p50_latency() - 50.0).abs() < 1.0);
        assert!((metrics.p95_latency() - 95.0).abs() < 1.0);
        assert!((metrics.p99_latency() - 99.0).abs() < 1.0);
    }
    
    #[test]
    fn test_error_rate() {
        let metrics = PerformanceMetrics::new(100);
        
        for _ in 0..100 {
            metrics.record_latency(10.0);
        }
        for _ in 0..10 {
            metrics.record_error();
        }
        
        assert!((metrics.error_rate() - 0.1).abs() < 0.01);
    }
    
    #[test]
    fn test_histogram() {
        let metrics = PerformanceMetrics::new(100);
        
        metrics.record_latency(0.5);   // <= 1ms bucket
        metrics.record_latency(3.0);   // <= 5ms bucket
        metrics.record_latency(100.0); // <= 100ms bucket
        
        let histogram = metrics.histogram();
        assert!(histogram[0].count >= 1); // <= 1ms
        assert!(histogram[1].count >= 1); // <= 5ms
    }
}
