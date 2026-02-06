//! Memory Module
//!
//! Provides efficient buffer pooling with NUMA-awareness
//! for reduced allocation overhead, plus memory monitoring
//! and adaptive chunk processing.

mod pool;
pub mod monitor;

pub use pool::{MemoryPool, PooledBuffer, BufferCategory, PoolStats};
pub use monitor::{MemoryPressureLevel, MemoryMonitor, AdaptiveChunkProcessor};
