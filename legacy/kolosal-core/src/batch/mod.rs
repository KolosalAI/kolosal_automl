//! Batch Processing Module
//!
//! Provides high-performance batch processing with priority queues,
//! adaptive batch sizing, and intelligent request scheduling.

mod processor;
mod batcher;
mod priority;

pub use processor::{BatchProcessor, BatchProcessorConfig, BatchResult};
pub use batcher::{DynamicBatcher, BatcherConfig};
pub use priority::{PriorityQueue, Priority, PrioritizedItem};
