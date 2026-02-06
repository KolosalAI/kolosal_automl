//! Utility functions and types

mod parallel;
mod metrics;
pub mod data_loader;
pub mod simd;
pub mod memory;

pub use parallel::{ParallelConfig, parallel_map};
pub use metrics::{Timer, MemoryTracker};
pub use data_loader::{DataLoader, DataSaver, ChunkedReader, FileInfo};
pub use simd::SimdOps;
pub use memory::{MemoryPool, LruCache, Arena, PoolStats, CacheStats};

use std::time::{Duration, Instant};
