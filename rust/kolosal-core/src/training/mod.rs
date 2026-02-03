//! Model training module
//!
//! Provides training functionality for various ML models including:
//! - Decision trees and Random Forests
//! - Gradient boosting
//! - Linear models
//! - Neural networks (future)

mod config;
mod engine;
mod models;

pub use config::{TrainingConfig, ModelType, TaskType};
pub use engine::TrainEngine;
pub use models::{Model, ModelMetrics};

use crate::error::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
