//! Timing and memory tracking utilities

use std::time::{Duration, Instant};
use tracing::info;

/// Timer for measuring execution time
#[derive(Debug)]
pub struct Timer {
    name: String,
    start: Instant,
    checkpoints: Vec<(String, Duration)>,
}

impl Timer {
    /// Create and start a new timer
    pub fn start(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    /// Add a checkpoint
    pub fn checkpoint(&mut self, name: impl Into<String>) {
        self.checkpoints.push((name.into(), self.start.elapsed()));
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get elapsed time in seconds
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Stop and log the timer
    pub fn stop(self) -> Duration {
        let elapsed = self.start.elapsed();
        info!(
            "{} completed in {:.3}s",
            self.name,
            elapsed.as_secs_f64()
        );
        elapsed
    }

    /// Stop with detailed checkpoint report
    pub fn stop_with_report(self) -> Duration {
        let elapsed = self.start.elapsed();
        
        info!("=== {} Timing Report ===", self.name);
        
        let mut prev_time = Duration::ZERO;
        for (name, time) in &self.checkpoints {
            let delta = *time - prev_time;
            info!("  {}: {:.3}s (+{:.3}s)", name, time.as_secs_f64(), delta.as_secs_f64());
            prev_time = *time;
        }
        
        info!("  Total: {:.3}s", elapsed.as_secs_f64());
        info!("========================");
        
        elapsed
    }
}

/// Memory tracker for monitoring memory usage
#[derive(Debug)]
pub struct MemoryTracker {
    name: String,
    initial_bytes: Option<usize>,
    peak_bytes: Option<usize>,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            initial_bytes: Self::current_memory(),
            peak_bytes: None,
        }
    }

    /// Update peak memory usage
    pub fn update(&mut self) {
        if let Some(current) = Self::current_memory() {
            match self.peak_bytes {
                None => self.peak_bytes = Some(current),
                Some(peak) if current > peak => self.peak_bytes = Some(current),
                _ => {}
            }
        }
    }

    /// Get current memory usage (platform-specific)
    fn current_memory() -> Option<usize> {
        // This is a simplified implementation
        // In production, would use platform-specific APIs
        None
    }

    /// Report memory usage
    pub fn report(&self) {
        info!(
            "{} memory: initial={:?}, peak={:?}",
            self.name, self.initial_bytes, self.peak_bytes
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_timer() {
        let timer = Timer::start("test");
        sleep(Duration::from_millis(10));
        let elapsed = timer.stop();
        
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_timer_checkpoints() {
        let mut timer = Timer::start("test");
        
        sleep(Duration::from_millis(5));
        timer.checkpoint("step1");
        
        sleep(Duration::from_millis(5));
        timer.checkpoint("step2");
        
        assert_eq!(timer.checkpoints.len(), 2);
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new("test");
        tracker.update();
        // Just verify it doesn't panic
    }
}
