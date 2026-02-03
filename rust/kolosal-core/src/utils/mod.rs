//! Utility functions and types

mod parallel;
mod metrics;

pub use parallel::{ParallelConfig, parallel_map};
pub use metrics::{Timer, MemoryTracker};

use std::time::{Duration, Instant};
