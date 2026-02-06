//! Advanced imputation module
//!
//! Provides sophisticated imputation methods including:
//! - MICE (Multiple Imputation by Chained Equations)
//! - KNN Imputer
//! - Iterative Imputer

mod mice;
mod knn;
mod iterative;

pub use mice::MICEImputer;
pub use knn::KNNImputer;
pub use iterative::{IterativeImputer, ImputerEstimator};

use crate::error::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Strategy for initial imputation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitialStrategy {
    /// Mean imputation
    Mean,
    /// Median imputation
    Median,
    /// Most frequent value
    MostFrequent,
    /// Constant value (0)
    Constant,
}

/// Trait for imputers
pub trait Imputer: Send + Sync {
    /// Fit the imputer on data with missing values
    fn fit(&mut self, x: &Array2<f64>) -> Result<()>;
    
    /// Transform data by imputing missing values
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    
    /// Fit and transform in one step
    fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }
}

/// Check if value is missing (NaN)
#[inline]
pub fn is_missing(v: f64) -> bool {
    v.is_nan()
}
