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
pub mod cross_validation;
pub mod linear_models;
pub mod decision_tree;
pub mod random_forest;

pub use config::{TrainingConfig, ModelType, TaskType};
pub use engine::TrainEngine;
pub use models::{Model, ModelMetrics};
pub use cross_validation::{CrossValidator, CVStrategy, CVSplit, CVResults};
pub use linear_models::{LinearRegression, LogisticRegression};
pub use decision_tree::{DecisionTree, TreeNode, Criterion};
pub use random_forest::{RandomForest, MaxFeatures};

use crate::error::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
