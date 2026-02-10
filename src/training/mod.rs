//! Model training module
//!
//! Provides training functionality for various ML models including:
//! - Decision trees and Random Forests
//! - Gradient boosting, XGBoost, LightGBM, CatBoost
//! - Linear models (OLS, Ridge, Lasso, ElasticNet, Polynomial)
//! - K-Nearest Neighbors
//! - Naive Bayes
//! - Neural networks (MLP)
//! - Support Vector Machines
//! - AdaBoost
//! - Extra Trees (Extremely Randomized Trees)
//! - Stochastic Gradient Descent (SGD)
//! - Gaussian Process Regression
//! - Clustering (KMeans, DBSCAN)

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
pub mod adaboost;
pub mod extra_trees;
pub mod xgboost;
pub mod lightgbm;
pub mod catboost;
pub mod sgd;
pub mod gaussian_process_regression;
pub mod clustering;

pub use config::{TrainingConfig, ModelType, TaskType};
pub use engine::{TrainEngine, TrainedModel, EnsembleStrategy, EnsembleResult, ModelComparison};
pub use models::{Model, ModelMetrics};
pub use cross_validation::{CrossValidator, CVStrategy, CVSplit, CVResults};
pub use linear_models::{LinearRegression, LogisticRegression, RidgeRegression, LassoRegression, ElasticNetRegression, PolynomialRegression};
pub use decision_tree::{DecisionTree, TreeNode, Criterion};
pub use random_forest::{RandomForest, MaxFeatures};
pub use gradient_boosting::{GradientBoostingRegressor, GradientBoostingClassifier, GradientBoostingConfig};
pub use knn::{KNNClassifier, KNNRegressor, KNNConfig, DistanceMetric, WeightScheme};
pub use naive_bayes::{GaussianNaiveBayes, MultinomialNaiveBayes};
pub use neural_network::{MLPRegressor, MLPClassifier, MLPConfig, Activation};
pub use svm::{SVMClassifier, SVMRegressor, SVMConfig, KernelType};
pub use adaboost::AdaBoostClassifier;
pub use extra_trees::ExtraTrees;
pub use xgboost::{XGBoostRegressor, XGBoostClassifier, XGBoostConfig};
pub use lightgbm::{LightGBMRegressor, LightGBMClassifier, LightGBMConfig};
pub use catboost::{CatBoostRegressor, CatBoostClassifier, CatBoostConfig};
pub use sgd::{SGDRegressor, SGDClassifier, SGDConfig};
pub use gaussian_process_regression::{GaussianProcessRegressor, GPConfig};
pub use clustering::{KMeans, DBSCAN};

use crate::error::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
