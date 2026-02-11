//! Synthetic data generation module
//!
//! Provides techniques for generating synthetic samples:
//! - SMOTE (Synthetic Minority Over-sampling Technique)
//! - ADASYN (Adaptive Synthetic Sampling)
//! - Random oversampling
//! - Undersampling methods

mod smote;
mod adasyn;
mod random_sampling;

pub use smote::{SMOTE, SMOTEVariant, BorderlineSMOTE};
pub use adasyn::ADASYN;
pub use random_sampling::{RandomOverSampler, RandomUnderSampler, NearMiss};

use crate::error::Result;
use ndarray::{Array1, Array2};

/// Result of resampling
#[derive(Debug, Clone)]
pub struct ResampleResult {
    /// Resampled features
    pub x: Array2<f64>,
    /// Resampled labels
    pub y: Array1<i64>,
    /// Number of synthetic samples generated per class
    pub n_synthetic: Vec<usize>,
}

/// Trait for samplers
pub trait Sampler: Send + Sync {
    /// Fit the sampler on data
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<i64>) -> Result<()>;
    
    /// Resample data
    fn resample(&self, x: &Array2<f64>, y: &Array1<i64>) -> Result<ResampleResult>;
    
    /// Fit and resample in one step
    fn fit_resample(&mut self, x: &Array2<f64>, y: &Array1<i64>) -> Result<ResampleResult> {
        self.fit(x, y)?;
        self.resample(x, y)
    }
}

/// Get class distribution
pub fn class_counts(y: &Array1<i64>) -> std::collections::HashMap<i64, usize> {
    let mut counts = std::collections::HashMap::new();
    for &label in y.iter() {
        *counts.entry(label).or_insert(0) += 1;
    }
    counts
}

/// Get indices for each class
pub fn class_indices(y: &Array1<i64>) -> std::collections::HashMap<i64, Vec<usize>> {
    let mut indices = std::collections::HashMap::new();
    for (i, &label) in y.iter().enumerate() {
        indices.entry(label).or_insert_with(Vec::new).push(i);
    }
    indices
}
