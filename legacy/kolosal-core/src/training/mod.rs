//! Model training module
//!
//! Provides training functionality for various ML models including:
//! - Decision trees and Random Forests
//! - Gradient boosting
//! - Linear models
//! - K-Nearest Neighbors
//! - Naive Bayes
//! - Neural networks (MLP)
//! - Support Vector Machines

mod config;
mod engine;
mod models;
pub mod cross_validation;
pub mod linear_models;
pub mod decision_tree;
pub mod random_forest;
pub mod gradient_boosting;
pub mod knn;
pub mod naive_bayes;
pub mod neural_network;
pub mod svm;

pub use config::{TrainingConfig, ModelType, TaskType};
pub use engine::{TrainEngine, TrainedModel};
pub use models::{Model, ModelMetrics};
pub use cross_validation::{CrossValidator, CVStrategy, CVSplit, CVResults};
pub use linear_models::{LinearRegression, LogisticRegression};
pub use decision_tree::{DecisionTree, TreeNode, Criterion};
pub use random_forest::{RandomForest, MaxFeatures};
pub use gradient_boosting::{GradientBoostingRegressor, GradientBoostingClassifier, GradientBoostingConfig};
pub use knn::{KNNClassifier, KNNRegressor, KNNConfig, DistanceMetric, WeightScheme};
pub use naive_bayes::{GaussianNaiveBayes, MultinomialNaiveBayes};
pub use neural_network::{MLPRegressor, MLPClassifier, MLPConfig, Activation};
pub use svm::{SVMClassifier, SVMRegressor, SVMConfig, KernelType};

use crate::error::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
