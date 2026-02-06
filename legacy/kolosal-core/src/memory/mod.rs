//! Memory Pool Module
//!
//! Provides efficient buffer pooling with NUMA-awareness
//! for reduced allocation overhead.

mod pool;

pub use pool::{MemoryPool, PooledBuffer, BufferCategory, PoolStats};
