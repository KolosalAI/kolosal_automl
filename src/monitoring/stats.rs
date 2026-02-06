//! Batch Statistics
//!
//! Statistics for batch processing operations.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Instant;

/// Statistics summary
#[derive(Debug, Clone, Default)]
pub struct StatsSummary {
    /// Number of observations
    pub count: u64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Average value
    pub avg: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Sum of all values
    pub sum: f64,
}

impl StatsSummary {
    /// Create a summary from a slice of values
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        
        let count = values.len() as u64;
        let sum: f64 = values.iter().sum();
        let avg = sum / count as f64;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        
        let variance: f64 = values.iter()
            .map(|&x| (x - avg).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        
        Self {
            count,
            min,
            max,
            avg,
            std_dev,
            sum,
        }
    }
}

/// Batch processing statistics
pub struct BatchStats {
    /// Window size for rolling statistics
    window_size: usize,
    
    // Processing times
    processing_times: RwLock<VecDeque<f64>>,
    
    // Batch sizes
    batch_sizes: RwLock<VecDeque<usize>>,
    
    // Queue sizes
    queue_sizes: RwLock<VecDeque<usize>>,
    
    // Counters
    total_batches: AtomicU64,
    total_items: AtomicU64,
    total_errors: AtomicU64,
    
    // Timing
    start_time: Instant,
}

impl BatchStats {
    /// Create new batch statistics with given window size
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            processing_times: RwLock::new(VecDeque::with_capacity(window_size)),
            batch_sizes: RwLock::new(VecDeque::with_capacity(window_size)),
            queue_sizes: RwLock::new(VecDeque::with_capacity(window_size)),
            total_batches: AtomicU64::new(0),
            total_items: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }
    
    /// Record a batch processing event
    pub fn record_batch(&self, batch_size: usize, processing_time_ms: f64) {
        self.total_batches.fetch_add(1, Ordering::SeqCst);
        self.total_items.fetch_add(batch_size as u64, Ordering::SeqCst);
        
        // Record processing time
        if let Ok(mut times) = self.processing_times.write() {
            times.push_back(processing_time_ms);
            if times.len() > self.window_size {
                times.pop_front();
            }
        }
        
        // Record batch size
        if let Ok(mut sizes) = self.batch_sizes.write() {
            sizes.push_back(batch_size);
            if sizes.len() > self.window_size {
                sizes.pop_front();
            }
        }
    }
    
    /// Record current queue size
    pub fn record_queue_size(&self, size: usize) {
        if let Ok(mut sizes) = self.queue_sizes.write() {
            sizes.push_back(size);
            if sizes.len() > self.window_size {
                sizes.pop_front();
            }
        }
    }
    
    /// Record an error
    pub fn record_error(&self) {
        self.total_errors.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Get processing time statistics
    pub fn processing_time_stats(&self) -> StatsSummary {
        self.processing_times.read()
            .map(|times| {
                let values: Vec<f64> = times.iter().copied().collect();
                StatsSummary::from_values(&values)
            })
            .unwrap_or_default()
    }
    
    /// Get batch size statistics
    pub fn batch_size_stats(&self) -> StatsSummary {
        self.batch_sizes.read()
            .map(|sizes| {
                let values: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
                StatsSummary::from_values(&values)
            })
            .unwrap_or_default()
    }
    
    /// Get queue size statistics
    pub fn queue_size_stats(&self) -> StatsSummary {
        self.queue_sizes.read()
            .map(|sizes| {
                let values: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
                StatsSummary::from_values(&values)
            })
            .unwrap_or_default()
    }
    
    /// Get average processing time
    pub fn avg_processing_time(&self) -> f64 {
        self.processing_time_stats().avg
    }
    
    /// Get average batch size
    pub fn avg_batch_size(&self) -> f64 {
        self.batch_size_stats().avg
    }
    
    /// Get total batches processed
    pub fn total_batches(&self) -> u64 {
        self.total_batches.load(Ordering::SeqCst)
    }
    
    /// Get total items processed
    pub fn total_items(&self) -> u64 {
        self.total_items.load(Ordering::SeqCst)
    }
    
    /// Get total errors
    pub fn total_errors(&self) -> u64 {
        self.total_errors.load(Ordering::SeqCst)
    }
    
    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        let batches = self.total_batches();
        let errors = self.total_errors();
        
        if batches > 0 {
            errors as f64 / batches as f64
        } else {
            0.0
        }
    }
    
    /// Get throughput (items per second)
    pub fn throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_items() as f64 / elapsed
        } else {
            0.0
        }
    }
    
    /// Get batches per second
    pub fn batches_per_second(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_batches() as f64 / elapsed
        } else {
            0.0
        }
    }
    
    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
    
    /// Get a comprehensive summary
    pub fn summary(&self) -> BatchStatsSummary {
        BatchStatsSummary {
            total_batches: self.total_batches(),
            total_items: self.total_items(),
            total_errors: self.total_errors(),
            error_rate: self.error_rate(),
            avg_processing_time_ms: self.avg_processing_time(),
            avg_batch_size: self.avg_batch_size(),
            throughput: self.throughput(),
            batches_per_second: self.batches_per_second(),
            uptime_secs: self.uptime_secs(),
            processing_time_stats: self.processing_time_stats(),
            batch_size_stats: self.batch_size_stats(),
            queue_size_stats: self.queue_size_stats(),
        }
    }
    
    /// Reset all statistics
    pub fn reset(&self) {
        if let Ok(mut times) = self.processing_times.write() {
            times.clear();
        }
        if let Ok(mut sizes) = self.batch_sizes.write() {
            sizes.clear();
        }
        if let Ok(mut sizes) = self.queue_sizes.write() {
            sizes.clear();
        }
        
        self.total_batches.store(0, Ordering::SeqCst);
        self.total_items.store(0, Ordering::SeqCst);
        self.total_errors.store(0, Ordering::SeqCst);
    }
}

impl Default for BatchStats {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Comprehensive batch statistics summary
#[derive(Debug, Clone)]
pub struct BatchStatsSummary {
    /// Total batches processed
    pub total_batches: u64,
    /// Total items processed
    pub total_items: u64,
    /// Total errors
    pub total_errors: u64,
    /// Error rate
    pub error_rate: f64,
    /// Average processing time in milliseconds
    pub avg_processing_time_ms: f64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Throughput (items per second)
    pub throughput: f64,
    /// Batches per second
    pub batches_per_second: f64,
    /// Uptime in seconds
    pub uptime_secs: f64,
    /// Processing time statistics
    pub processing_time_stats: StatsSummary,
    /// Batch size statistics
    pub batch_size_stats: StatsSummary,
    /// Queue size statistics
    pub queue_size_stats: StatsSummary,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_stats_basic() {
        let stats = BatchStats::new(100);
        
        stats.record_batch(10, 5.0);
        stats.record_batch(20, 10.0);
        stats.record_batch(30, 15.0);
        
        assert_eq!(stats.total_batches(), 3);
        assert_eq!(stats.total_items(), 60);
        assert!((stats.avg_processing_time() - 10.0).abs() < 0.01);
        assert!((stats.avg_batch_size() - 20.0).abs() < 0.01);
    }
    
    #[test]
    fn test_stats_summary() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let summary = StatsSummary::from_values(&values);
        
        assert_eq!(summary.count, 5);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 5.0);
        assert!((summary.avg - 3.0).abs() < 0.01);
        assert!((summary.sum - 15.0).abs() < 0.01);
    }
    
    #[test]
    fn test_error_tracking() {
        let stats = BatchStats::new(100);
        
        stats.record_batch(10, 5.0);
        stats.record_batch(10, 5.0);
        stats.record_error();
        
        assert_eq!(stats.total_errors(), 1);
        assert!((stats.error_rate() - 0.5).abs() < 0.01);
    }
}
