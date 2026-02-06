//! Inference engine module
//!
//! Provides high-performance model inference with:
//! - Batch processing
//! - Model quantization support
//! - Parallel prediction
//! - Memory-efficient streaming

mod config;
mod engine;

pub use config::{InferenceConfig, QuantizationType};
pub use engine::InferenceEngine;

use crate::error::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
