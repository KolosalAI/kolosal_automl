//! Hyperparameter optimization module (HyperOptX)
//!
//! Provides advanced hyperparameter optimization including:
//! - Bayesian optimization with Gaussian Processes
//! - Tree-structured Parzen Estimators (TPE)
//! - Random search
//! - Grid search
//! - Evolutionary algorithms
//! - Multi-fidelity optimization (Hyperband, BOHB)

mod config;
mod search_space;
mod optimizer;
mod samplers;

pub use config::OptimizationConfig;
pub use search_space::{SearchSpace, Parameter, ParameterType};
pub use optimizer::HyperOptX;
pub use samplers::{Sampler, SamplerType};

use crate::error::Result;
use serde::{Deserialize, Serialize};
