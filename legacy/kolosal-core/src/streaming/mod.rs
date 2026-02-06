//! Streaming Pipeline Module
//!
//! Provides streaming data processing with backpressure control
//! and memory optimization.

mod pipeline;
mod backpressure;

pub use pipeline::{StreamingPipeline, StreamConfig, StreamStats, ChunkResult};
pub use backpressure::{BackpressureController, BackpressureConfig};
