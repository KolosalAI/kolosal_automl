//! Backpressure Controller
//!
//! Controls backpressure in streaming pipelines.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Configuration for backpressure control
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Maximum queue size before applying backpressure
    pub max_queue_size: usize,
    /// Memory threshold in MB for backpressure activation
    pub memory_threshold_mb: usize,
    /// Enable adaptive batching
    pub adaptive_batching: bool,
    /// Initial batch size
    pub initial_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Backoff factor when backpressure is applied
    pub backoff_factor: f64,
    /// Recovery factor when backpressure is released
    pub recovery_factor: f64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 100,
            memory_threshold_mb: 1000,
            adaptive_batching: true,
            initial_batch_size: 1000,
            min_batch_size: 100,
            max_batch_size: 10000,
            backoff_factor: 0.5,
            recovery_factor: 1.2,
        }
    }
}

/// Controller for backpressure in streaming pipelines
pub struct BackpressureController {
    config: BackpressureConfig,
    
    // Current state
    current_queue_size: AtomicUsize,
    current_batch_size: AtomicUsize,
    backpressure_active: AtomicBool,
    
    // Statistics
    backpressure_events: AtomicU64,
    total_wait_time_ms: AtomicU64,
    
    // Synchronization
    capacity_available: Arc<(Mutex<bool>, Condvar)>,
}

impl BackpressureController {
    /// Create a new backpressure controller
    pub fn new(config: BackpressureConfig) -> Self {
        let initial_batch_size = config.initial_batch_size;
        
        Self {
            config,
            current_queue_size: AtomicUsize::new(0),
            current_batch_size: AtomicUsize::new(initial_batch_size),
            backpressure_active: AtomicBool::new(false),
            backpressure_events: AtomicU64::new(0),
            total_wait_time_ms: AtomicU64::new(0),
            capacity_available: Arc::new((Mutex::new(true), Condvar::new())),
        }
    }
    
    /// Check if backpressure should be applied
    pub fn should_apply_backpressure(&self) -> bool {
        let queue_size = self.current_queue_size.load(Ordering::SeqCst);
        let threshold = (self.config.max_queue_size as f64 * 0.8) as usize;
        
        queue_size >= threshold
    }
    
    /// Update the current queue size
    pub fn set_queue_size(&self, size: usize) {
        self.current_queue_size.store(size, Ordering::SeqCst);
        
        let should_apply = self.should_apply_backpressure();
        let was_active = self.backpressure_active.swap(should_apply, Ordering::SeqCst);
        
        if should_apply && !was_active {
            // Just activated backpressure
            self.backpressure_events.fetch_add(1, Ordering::SeqCst);
            self.reduce_batch_size();
        } else if !should_apply && was_active {
            // Backpressure released
            self.increase_batch_size();
            self.signal_capacity_available();
        }
    }
    
    /// Increment queue size
    pub fn increment_queue(&self) -> usize {
        let new_size = self.current_queue_size.fetch_add(1, Ordering::SeqCst) + 1;
        self.set_queue_size(new_size);
        new_size
    }
    
    /// Decrement queue size (uses CAS loop to prevent underflow)
    pub fn decrement_queue(&self) -> usize {
        loop {
            let current = self.current_queue_size.load(Ordering::SeqCst);
            if current == 0 {
                return 0;
            }
            match self.current_queue_size.compare_exchange(
                current, current - 1, Ordering::SeqCst, Ordering::SeqCst
            ) {
                Ok(_) => {
                    let new_size = current - 1;
                    self.set_queue_size(new_size);
                    return new_size;
                }
                Err(_) => continue,
            }
        }
    }
    
    /// Wait for capacity to be available
    pub fn wait_for_capacity(&self, timeout: Option<Duration>) -> Result<(), BackpressureTimeout> {
        if !self.should_apply_backpressure() {
            return Ok(());
        }
        
        let start = Instant::now();
        let (lock, cvar) = &*self.capacity_available;
        
        let mut guard = lock.lock().map_err(|_| BackpressureTimeout)?;
        
        loop {
            if !self.should_apply_backpressure() {
                return Ok(());
            }
            
            if let Some(timeout) = timeout {
                let elapsed = start.elapsed();
                if elapsed >= timeout {
                    return Err(BackpressureTimeout);
                }
                
                let remaining = timeout - elapsed;
                let result = cvar.wait_timeout(guard, remaining).map_err(|_| BackpressureTimeout)?;
                guard = result.0;
                
                if result.1.timed_out() {
                    return Err(BackpressureTimeout);
                }
            } else {
                guard = cvar.wait(guard).map_err(|_| BackpressureTimeout)?;
            }
        }
    }
    
    fn signal_capacity_available(&self) {
        let (lock, cvar) = &*self.capacity_available;
        if let Ok(mut guard) = lock.lock() {
            *guard = true;
            cvar.notify_all();
        }
    }
    
    fn reduce_batch_size(&self) {
        if !self.config.adaptive_batching {
            return;
        }
        
        let current = self.current_batch_size.load(Ordering::SeqCst);
        let new_size = ((current as f64) * self.config.backoff_factor) as usize;
        let new_size = new_size.max(self.config.min_batch_size);
        self.current_batch_size.store(new_size, Ordering::SeqCst);
    }
    
    fn increase_batch_size(&self) {
        if !self.config.adaptive_batching {
            return;
        }
        
        let current = self.current_batch_size.load(Ordering::SeqCst);
        let new_size = ((current as f64) * self.config.recovery_factor) as usize;
        let new_size = new_size.min(self.config.max_batch_size);
        self.current_batch_size.store(new_size, Ordering::SeqCst);
    }
    
    /// Get the current recommended batch size
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size.load(Ordering::SeqCst)
    }
    
    /// Check if backpressure is currently active
    pub fn is_backpressure_active(&self) -> bool {
        self.backpressure_active.load(Ordering::SeqCst)
    }
    
    /// Get backpressure statistics
    pub fn stats(&self) -> BackpressureStats {
        BackpressureStats {
            backpressure_events: self.backpressure_events.load(Ordering::SeqCst),
            current_queue_size: self.current_queue_size.load(Ordering::SeqCst),
            current_batch_size: self.current_batch_size.load(Ordering::SeqCst),
            is_active: self.backpressure_active.load(Ordering::SeqCst),
            total_wait_time_ms: self.total_wait_time_ms.load(Ordering::SeqCst),
        }
    }
    
    /// Reset the controller
    pub fn reset(&self) {
        self.current_queue_size.store(0, Ordering::SeqCst);
        self.current_batch_size.store(self.config.initial_batch_size, Ordering::SeqCst);
        self.backpressure_active.store(false, Ordering::SeqCst);
        self.signal_capacity_available();
    }
}

/// Backpressure timeout error
#[derive(Debug, Clone)]
pub struct BackpressureTimeout;

impl std::fmt::Display for BackpressureTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Backpressure timeout exceeded")
    }
}

impl std::error::Error for BackpressureTimeout {}

/// Backpressure statistics
#[derive(Debug, Clone, Default)]
pub struct BackpressureStats {
    /// Number of times backpressure was activated
    pub backpressure_events: u64,
    /// Current queue size
    pub current_queue_size: usize,
    /// Current batch size
    pub current_batch_size: usize,
    /// Whether backpressure is currently active
    pub is_active: bool,
    /// Total wait time in milliseconds
    pub total_wait_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_backpressure_basic() {
        let config = BackpressureConfig {
            max_queue_size: 10,
            ..Default::default()
        };
        let controller = BackpressureController::new(config);
        
        // Initially no backpressure
        assert!(!controller.should_apply_backpressure());
        assert!(!controller.is_backpressure_active());
        
        // Fill queue to trigger backpressure
        for _ in 0..9 {
            controller.increment_queue();
        }
        
        assert!(controller.should_apply_backpressure());
        assert!(controller.is_backpressure_active());
    }
    
    #[test]
    fn test_adaptive_batch_sizing() {
        let config = BackpressureConfig {
            max_queue_size: 10,
            initial_batch_size: 1000,
            min_batch_size: 100,
            backoff_factor: 0.5,
            ..Default::default()
        };
        let controller = BackpressureController::new(config);
        
        let initial = controller.current_batch_size();
        assert_eq!(initial, 1000);
        
        // Trigger backpressure
        controller.set_queue_size(9);
        
        // Batch size should be reduced
        let reduced = controller.current_batch_size();
        assert!(reduced < initial);
    }
    
    #[test]
    fn test_backpressure_release() {
        let config = BackpressureConfig {
            max_queue_size: 10,
            ..Default::default()
        };
        let controller = BackpressureController::new(config);
        
        // Trigger backpressure
        controller.set_queue_size(9);
        assert!(controller.is_backpressure_active());
        
        // Release backpressure
        controller.set_queue_size(2);
        assert!(!controller.is_backpressure_active());
    }
}
