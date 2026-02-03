//! Hyperparameter optimization module (HyperOptX)
//!
//! Provides advanced hyperparameter optimization including:
//! - Bayesian optimization with Gaussian Processes
//! - Tree-structured Parzen Estimators (TPE)
//! - Random search
//! - Grid search
//! - Evolutionary algorithms
//! - Multi-fidelity optimization (Hyperband, BOHB)
//! - Adaptive Successive Halving (ASHT)

mod config;
mod search_space;
mod optimizer;
mod samplers;
pub mod pruners;
pub mod asht;

pub use config::{OptimizationConfig, OptimizeDirection};
pub use search_space::{SearchSpace, Parameter, ParameterType, TrialParams, ParameterValue};
pub use optimizer::HyperOptX;
pub use samplers::{Sampler, SamplerType, SamplerConfig, create_sampler};
pub use pruners::{Pruner, PrunerConfig, MedianPruner, PercentilePruner, HyperbandPruner, NoPruner, TrialHistory, create_pruner};
pub use asht::{ASHT, ASHTConfig, ASHTResult, TrialResult};

use crate::error::Result;
use serde::{Deserialize, Serialize};
