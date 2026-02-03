//! Kolosal Core - High-performance AutoML engine
//!
//! This crate provides the core functionality for the Kolosal AutoML framework,
//! including data preprocessing, model training, inference, and hyperparameter optimization.

pub mod error;
pub mod preprocessing;
pub mod training;
pub mod inference;
pub mod optimizer;
pub mod utils;

pub use error::{KolosalError, Result};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::error::{KolosalError, Result};
    pub use crate::preprocessing::{DataPreprocessor, PreprocessingConfig};
    pub use crate::training::{TrainEngine, TrainingConfig, ModelType};
    pub use crate::inference::{InferenceEngine, InferenceConfig};
    pub use crate::optimizer::{HyperOptX, OptimizationConfig, SearchSpace};
}
