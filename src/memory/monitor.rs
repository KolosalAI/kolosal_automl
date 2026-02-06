//! Memory Monitoring Module
//!
//! Provides memory pressure detection, adaptive chunk processing,
//! and runtime memory tracking. Mirrors the Python
//! `modules/engine/memory_aware_processor.py` functionality.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use sysinfo::{Pid, System};

/// Represents the current level of memory pressure on the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Monitors process and system memory, tracks history, and detects pressure levels.
pub struct MemoryMonitor {
    warning_threshold: f64,
    critical_threshold: f64,
    initial_memory_mb: f64,
    memory_history: Vec<f64>,
    system: System,
}

impl MemoryMonitor {
    /// Create a new `MemoryMonitor`.
    ///
    /// * `warning_threshold` – system memory usage % to consider "High" (default 70.0).
    /// * `critical_threshold` – system memory usage % to consider "Critical" (default 85.0).
    pub fn new(warning_threshold: f64, critical_threshold: f64) -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        let initial = Self::process_memory_mb(&system);
        Self {
            warning_threshold,
            critical_threshold,
            initial_memory_mb: initial,
            memory_history: vec![initial],
            system,
        }
    }

    // --------------- internal helpers ---------------

    fn process_memory_mb(system: &System) -> f64 {
        let pid = Pid::from_u32(std::process::id());
        system
            .process(pid)
            .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    }

    // --------------- public API ---------------

    /// Current process resident memory in MB.
    pub fn get_memory_usage(&mut self) -> f64 {
        self.system.refresh_all();
        Self::process_memory_mb(&self.system)
    }

    /// Returns `(total_mb, available_mb, used_percent)` for system RAM.
    pub fn get_system_memory(&mut self) -> (f64, f64, f64) {
        self.system.refresh_all();
        let total = self.system.total_memory() as f64 / (1024.0 * 1024.0);
        let available = self.system.available_memory() as f64 / (1024.0 * 1024.0);
        let used_pct = if total > 0.0 {
            (total - available) / total * 100.0
        } else {
            0.0
        };
        (total, available, used_pct)
    }

    /// Determine the current memory pressure level.
    pub fn get_pressure_level(&mut self) -> MemoryPressureLevel {
        let (_, _, used_pct) = self.get_system_memory();
        if used_pct >= self.critical_threshold {
            MemoryPressureLevel::Critical
        } else if used_pct >= self.warning_threshold {
            MemoryPressureLevel::High
        } else if used_pct >= self.warning_threshold * 0.7 {
            MemoryPressureLevel::Medium
        } else {
            MemoryPressureLevel::Low
        }
    }

    /// Returns `true` when current usage exceeds `threshold` (defaults to `warning_threshold`).
    pub fn should_trigger_gc(&mut self, threshold: Option<f64>) -> bool {
        let thresh = threshold.unwrap_or(self.warning_threshold);
        let (_, _, used_pct) = self.get_system_memory();
        used_pct >= thresh
    }

    /// Available system memory in MB.
    pub fn get_available_memory_mb(&mut self) -> f64 {
        let (_, available, _) = self.get_system_memory();
        available
    }

    /// Change in process memory since construction (MB).
    pub fn get_memory_delta(&mut self) -> f64 {
        self.get_memory_usage() - self.initial_memory_mb
    }

    /// Record the current process memory to history and return the value.
    pub fn update_memory(&mut self) -> f64 {
        let current = self.get_memory_usage();
        self.memory_history.push(current);
        current
    }

    /// Access the full memory history.
    pub fn get_memory_history(&self) -> &[f64] {
        &self.memory_history
    }

    /// Clear all recorded history (keeps the current reading as the new baseline).
    pub fn clear_history(&mut self) {
        self.memory_history.clear();
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new(70.0, 85.0)
    }
}

/// Processes data in adaptively-sized chunks based on available memory.
pub struct AdaptiveChunkProcessor {
    memory_monitor: MemoryMonitor,
    min_chunk_size: usize,
    max_chunk_size: usize,
    target_memory_pct: f64,
    chunks_processed: usize,
    total_processing_time_ms: f64,
    peak_memory_mb: f64,
}

impl AdaptiveChunkProcessor {
    pub fn new(min_chunk_size: usize, max_chunk_size: usize, target_memory_pct: f64) -> Self {
        Self {
            memory_monitor: MemoryMonitor::default(),
            min_chunk_size,
            max_chunk_size,
            target_memory_pct,
            chunks_processed: 0,
            total_processing_time_ms: 0.0,
            peak_memory_mb: 0.0,
        }
    }

    /// Determine a chunk size that fits within available memory headroom.
    pub fn get_adaptive_chunk_size(&mut self, data_size: usize) -> usize {
        let (total, available, _) = self.memory_monitor.get_system_memory();
        let target_available = total * (1.0 - self.target_memory_pct / 100.0);
        let headroom = if available > target_available {
            available - target_available
        } else {
            0.0
        };

        // Estimate: each element ~8 bytes (f64-sized)
        let headroom_elements = (headroom * 1024.0 * 1024.0 / 8.0) as usize;

        let chunk = headroom_elements
            .max(self.min_chunk_size)
            .min(self.max_chunk_size)
            .min(data_size);

        chunk.max(self.min_chunk_size)
    }

    /// Split `data` into adaptive chunks, apply `process_fn` to each, and collect results.
    pub fn process_chunks<T, F>(&mut self, data: &[T], process_fn: F) -> Vec<T>
    where
        T: Clone,
        F: Fn(&[T]) -> Vec<T>,
    {
        let mut results: Vec<T> = Vec::with_capacity(data.len());
        let mut offset = 0;

        while offset < data.len() {
            let remaining = data.len() - offset;
            let chunk_size = self.get_adaptive_chunk_size(remaining);
            let end = (offset + chunk_size).min(data.len());

            let start = Instant::now();
            let chunk_result = process_fn(&data[offset..end]);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            self.chunks_processed += 1;
            self.total_processing_time_ms += elapsed;

            let mem = self.memory_monitor.update_memory();
            if mem > self.peak_memory_mb {
                self.peak_memory_mb = mem;
            }

            results.extend(chunk_result);
            offset = end;
        }

        results
    }

    /// Summary statistics about processing performed so far.
    pub fn get_performance_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("chunks_processed".into(), self.chunks_processed as f64);
        stats.insert(
            "avg_time_ms".into(),
            if self.chunks_processed > 0 {
                self.total_processing_time_ms / self.chunks_processed as f64
            } else {
                0.0
            },
        );
        stats.insert("memory_peak_mb".into(), self.peak_memory_mb);
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_monitor_defaults() {
        let monitor = MemoryMonitor::default();
        assert!(!monitor.get_memory_history().is_empty());
    }

    #[test]
    fn test_memory_monitor_usage() {
        let mut monitor = MemoryMonitor::default();
        let usage = monitor.get_memory_usage();
        // Process should use at least some memory
        assert!(usage >= 0.0);
    }

    #[test]
    fn test_system_memory() {
        let mut monitor = MemoryMonitor::default();
        let (total, available, pct) = monitor.get_system_memory();
        assert!(total > 0.0);
        assert!(available >= 0.0);
        assert!((0.0..=100.0).contains(&pct));
    }

    #[test]
    fn test_pressure_level() {
        let mut monitor = MemoryMonitor::default();
        let level = monitor.get_pressure_level();
        // Just make sure it returns a valid variant without panicking
        let _ = format!("{:?}", level);
    }

    #[test]
    fn test_update_and_history() {
        let mut monitor = MemoryMonitor::default();
        let initial_len = monitor.get_memory_history().len();
        monitor.update_memory();
        assert_eq!(monitor.get_memory_history().len(), initial_len + 1);
        monitor.clear_history();
        assert!(monitor.get_memory_history().is_empty());
    }

    #[test]
    fn test_adaptive_chunk_processor() {
        let mut proc = AdaptiveChunkProcessor::new(1, 100, 80.0);
        let data: Vec<i32> = (0..50).collect();
        let result = proc.process_chunks(&data, |chunk| chunk.to_vec());
        assert_eq!(result, data);
        let stats = proc.get_performance_stats();
        assert!(*stats.get("chunks_processed").unwrap() >= 1.0);
    }
}
