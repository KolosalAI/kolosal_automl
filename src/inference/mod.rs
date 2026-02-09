//! Inference engine module
//!
//! Provides high-performance model inference with:
//! - Batch processing (sequential and parallel via rayon)
//! - Model quantization support
//! - Parallel prediction with configurable worker count
//! - Memory-efficient streaming
//! - Latency tracking and performance metrics
//! - Model warmup for consistent latency
//! - Classification probability support (predict_proba)
//! - Configurable classification threshold

mod config;
mod engine;

pub use config::{InferenceConfig, QuantizationType};
pub use engine::{InferenceEngine, InferenceStats};

use crate::error::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
