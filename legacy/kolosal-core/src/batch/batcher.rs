//! Dynamic Batcher Implementation
//!
//! Intelligent request batching with adaptive sizing for optimal hardware utilization.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::priority::{Priority, PriorityQueue};

/// Configuration for dynamic batcher
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum wait time in milliseconds before forcing batch processing
    pub max_wait_time_ms: u64,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Enable adaptive batch sizing
    pub enable_adaptive_sizing: bool,
    /// Adaptation interval in seconds
    pub adaptation_interval_secs: f64,
    /// Number of priority levels
    pub priority_levels: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_wait_time_ms: 10,
            max_queue_size: 1000,
            enable_adaptive_sizing: true,
            adaptation_interval_secs: 5.0,
            priority_levels: 4,
        }
    }
}

/// Statistics for the dynamic batcher
#[derive(Debug, Clone, Default)]
pub struct BatcherStats {
    /// Number of batches processed
    pub processed_batches: u64,
    /// Number of requests processed
    pub processed_requests: u64,
    /// Maximum observed queue size
    pub max_observed_queue_size: usize,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average wait time in milliseconds
    pub avg_wait_time_ms: f64,
    /// Average processing time in milliseconds
    pub avg_processing_time_ms: f64,
    /// Current optimal batch size
    pub current_optimal_batch_size: usize,
    /// Throughput (requests per second)
    pub throughput: f64,
}

/// A request to be batched
#[derive(Debug)]
pub struct BatchRequest<T> {
    /// The request data
    pub data: T,
    /// Priority level
    pub priority: Priority,
    /// Timestamp when request was created
    pub created_at: Instant,
}

impl<T> BatchRequest<T> {
    /// Create a new batch request
    pub fn new(data: T, priority: Priority) -> Self {
        Self {
            data,
            priority,
            created_at: Instant::now(),
        }
    }
    
    /// Create a normal priority request
    pub fn normal(data: T) -> Self {
        Self::new(data, Priority::Normal)
    }
    
    /// Get the age of this request in milliseconds
    pub fn age_ms(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64() * 1000.0
    }
}

/// Result of forming a batch
#[derive(Debug)]
pub struct FormedBatch<T> {
    /// The batch items
    pub items: Vec<T>,
    /// Average wait time for items in this batch
    pub avg_wait_time_ms: f64,
    /// Time taken to form the batch
    pub formation_time_ms: f64,
}

/// Dynamic batcher with intelligent batching strategies
pub struct DynamicBatcher<T: Send + 'static> {
    config: BatcherConfig,
    
    // Multi-priority queues
    priority_queues: Arc<Mutex<Vec<VecDeque<BatchRequest<T>>>>>,
    
    // Adaptive batch sizing
    current_optimal_batch_size: AtomicUsize,
    last_adaptation_time: Arc<Mutex<Instant>>,
    batch_performance_history: Arc<Mutex<VecDeque<f64>>>,
    
    // Control
    running: AtomicBool,
    batch_trigger: Arc<(Mutex<bool>, Condvar)>,
    
    // Statistics
    processed_batches: AtomicU64,
    processed_requests: AtomicU64,
    max_observed_queue_size: AtomicUsize,
    batch_sizes: Arc<Mutex<VecDeque<usize>>>,
    batch_wait_times: Arc<Mutex<VecDeque<f64>>>,
    batch_processing_times: Arc<Mutex<VecDeque<f64>>>,
    
    // Worker
    worker_thread: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl<T: Send + 'static> DynamicBatcher<T> {
    /// Create a new dynamic batcher
    pub fn new(config: BatcherConfig) -> Self {
        let num_priorities = config.priority_levels;
        let max_batch_size = config.max_batch_size;
        
        let priority_queues: Vec<VecDeque<BatchRequest<T>>> = (0..num_priorities)
            .map(|_| VecDeque::new())
            .collect();
        
        Self {
            config,
            priority_queues: Arc::new(Mutex::new(priority_queues)),
            current_optimal_batch_size: AtomicUsize::new(max_batch_size),
            last_adaptation_time: Arc::new(Mutex::new(Instant::now())),
            batch_performance_history: Arc::new(Mutex::new(VecDeque::with_capacity(50))),
            running: AtomicBool::new(false),
            batch_trigger: Arc::new((Mutex::new(false), Condvar::new())),
            processed_batches: AtomicU64::new(0),
            processed_requests: AtomicU64::new(0),
            max_observed_queue_size: AtomicUsize::new(0),
            batch_sizes: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            batch_wait_times: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            batch_processing_times: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            worker_thread: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Enqueue a request for batching
    pub fn enqueue(&self, request: BatchRequest<T>) -> bool {
        let priority_idx = (request.priority as usize).min(self.config.priority_levels - 1);
        
        let mut queues = match self.priority_queues.lock() {
            Ok(q) => q,
            Err(_) => return false,
        };
        
        // Check total queue size
        let total_size: usize = queues.iter().map(|q| q.len()).sum();
        if total_size >= self.config.max_queue_size {
            return false;
        }
        
        queues[priority_idx].push_back(request);
        
        // Update max observed queue size
        let new_total = total_size + 1;
        let current_max = self.max_observed_queue_size.load(Ordering::SeqCst);
        if new_total > current_max {
            self.max_observed_queue_size.store(new_total, Ordering::SeqCst);
        }
        
        // Signal batch trigger if we have enough items
        let optimal_size = self.current_optimal_batch_size.load(Ordering::SeqCst);
        if new_total >= optimal_size {
            let (ref lock, ref cvar) = *self.batch_trigger;
            if let Ok(_guard) = lock.lock() {
                cvar.notify_one();
            }
        }
        
        true
    }
    
    /// Enqueue data with normal priority
    pub fn enqueue_normal(&self, data: T) -> bool {
        self.enqueue(BatchRequest::normal(data))
    }
    
    /// Get the total queue size
    pub fn queue_size(&self) -> usize {
        self.priority_queues
            .lock()
            .map(|q| q.iter().map(|pq| pq.len()).sum())
            .unwrap_or(0)
    }
    
    /// Form a batch from the queues
    pub fn form_batch(&self) -> Option<FormedBatch<T>> {
        let start = Instant::now();
        let optimal_size = self.current_optimal_batch_size.load(Ordering::SeqCst);
        
        let mut queues = match self.priority_queues.lock() {
            Ok(q) => q,
            Err(_) => return None,
        };
        
        let mut items = Vec::with_capacity(optimal_size);
        let mut total_wait_time = 0.0;
        
        // Collect from highest to lowest priority
        for queue in queues.iter_mut() {
            while items.len() < optimal_size {
                match queue.pop_front() {
                    Some(request) => {
                        total_wait_time += request.age_ms();
                        items.push(request.data);
                    }
                    None => break,
                }
            }
            if items.len() >= optimal_size {
                break;
            }
        }
        
        if items.is_empty() {
            return None;
        }
        
        let avg_wait_time_ms = total_wait_time / items.len() as f64;
        let formation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        // Update statistics
        if let Ok(mut sizes) = self.batch_sizes.lock() {
            sizes.push_back(items.len());
            if sizes.len() > 100 {
                sizes.pop_front();
            }
        }
        
        if let Ok(mut times) = self.batch_wait_times.lock() {
            times.push_back(avg_wait_time_ms);
            if times.len() > 100 {
                times.pop_front();
            }
        }
        
        Some(FormedBatch {
            items,
            avg_wait_time_ms,
            formation_time_ms,
        })
    }
    
    /// Record processing time for a batch
    pub fn record_processing_time(&self, processing_time_ms: f64) {
        self.processed_batches.fetch_add(1, Ordering::SeqCst);
        
        if let Ok(mut times) = self.batch_processing_times.lock() {
            times.push_back(processing_time_ms);
            if times.len() > 100 {
                times.pop_front();
            }
        }
        
        // Update performance history for adaptation
        if let Ok(mut history) = self.batch_performance_history.lock() {
            history.push_back(processing_time_ms);
            if history.len() > 50 {
                history.pop_front();
            }
        }
        
        // Check if we should adapt
        if self.config.enable_adaptive_sizing {
            self.maybe_adapt();
        }
    }
    
    /// Maybe adapt batch size based on performance
    fn maybe_adapt(&self) {
        let should_adapt = {
            let last = self.last_adaptation_time.lock().ok();
            last.map(|t| t.elapsed().as_secs_f64() >= self.config.adaptation_interval_secs)
                .unwrap_or(false)
        };
        
        if should_adapt {
            self.adapt_batch_size();
        }
    }
    
    /// Adapt batch size based on performance history
    fn adapt_batch_size(&self) {
        let history = match self.batch_performance_history.lock() {
            Ok(h) => h.clone(),
            Err(_) => return,
        };
        
        if history.len() < 10 {
            return; // Not enough data
        }
        
        // Calculate average processing time
        let avg_time: f64 = history.iter().sum::<f64>() / history.len() as f64;
        let target_time = self.config.max_wait_time_ms as f64;
        
        let current_size = self.current_optimal_batch_size.load(Ordering::SeqCst);
        
        let new_size = if avg_time < target_time * 0.5 {
            // Processing is fast, can increase batch size
            ((current_size as f64) * 1.1).ceil() as usize
        } else if avg_time > target_time * 0.9 {
            // Processing is slow, decrease batch size
            ((current_size as f64) * 0.9).floor() as usize
        } else {
            current_size
        };
        
        let new_size = new_size.max(1).min(self.config.max_batch_size);
        self.current_optimal_batch_size.store(new_size, Ordering::SeqCst);
        
        // Update last adaptation time
        if let Ok(mut last) = self.last_adaptation_time.lock() {
            *last = Instant::now();
        }
    }
    
    /// Get batcher statistics
    pub fn stats(&self) -> BatcherStats {
        let processed_batches = self.processed_batches.load(Ordering::SeqCst);
        let processed_requests = self.processed_requests.load(Ordering::SeqCst);
        
        let avg_batch_size = self.batch_sizes
            .lock()
            .map(|sizes| {
                if sizes.is_empty() {
                    0.0
                } else {
                    sizes.iter().sum::<usize>() as f64 / sizes.len() as f64
                }
            })
            .unwrap_or(0.0);
        
        let avg_wait_time_ms = self.batch_wait_times
            .lock()
            .map(|times| {
                if times.is_empty() {
                    0.0
                } else {
                    times.iter().sum::<f64>() / times.len() as f64
                }
            })
            .unwrap_or(0.0);
        
        let avg_processing_time_ms = self.batch_processing_times
            .lock()
            .map(|times| {
                if times.is_empty() {
                    0.0
                } else {
                    times.iter().sum::<f64>() / times.len() as f64
                }
            })
            .unwrap_or(0.0);
        
        let throughput = if avg_processing_time_ms > 0.0 {
            (avg_batch_size * 1000.0) / avg_processing_time_ms
        } else {
            0.0
        };
        
        BatcherStats {
            processed_batches,
            processed_requests,
            max_observed_queue_size: self.max_observed_queue_size.load(Ordering::SeqCst),
            avg_batch_size,
            avg_wait_time_ms,
            avg_processing_time_ms,
            current_optimal_batch_size: self.current_optimal_batch_size.load(Ordering::SeqCst),
            throughput,
        }
    }
    
    /// Check if the batcher is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
    
    /// Stop the batcher
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        
        // Wake up the condition variable
        let (ref _lock, ref cvar) = *self.batch_trigger;
        cvar.notify_all();
    }
    
    /// Clear all queues
    pub fn clear(&self) {
        if let Ok(mut queues) = self.priority_queues.lock() {
            for queue in queues.iter_mut() {
                queue.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dynamic_batcher_enqueue() {
        let config = BatcherConfig::default();
        let batcher: DynamicBatcher<i32> = DynamicBatcher::new(config);
        
        for i in 0..10 {
            assert!(batcher.enqueue_normal(i));
        }
        
        assert_eq!(batcher.queue_size(), 10);
    }
    
    #[test]
    fn test_dynamic_batcher_form_batch() {
        let mut config = BatcherConfig::default();
        config.max_batch_size = 5;
        let batcher: DynamicBatcher<i32> = DynamicBatcher::new(config);
        
        for i in 0..10 {
            batcher.enqueue_normal(i);
        }
        
        let batch = batcher.form_batch().unwrap();
        assert_eq!(batch.items.len(), 5);
        assert_eq!(batcher.queue_size(), 5);
    }
    
    #[test]
    fn test_dynamic_batcher_priority() {
        let config = BatcherConfig::default();
        let batcher: DynamicBatcher<&str> = DynamicBatcher::new(config);
        
        batcher.enqueue(BatchRequest::new("low", Priority::Low));
        batcher.enqueue(BatchRequest::new("critical", Priority::Critical));
        batcher.enqueue(BatchRequest::new("normal", Priority::Normal));
        
        let batch = batcher.form_batch().unwrap();
        
        // Critical should come first
        assert_eq!(batch.items[0], "critical");
    }
}
