//! Anomaly detection module
//!
//! Provides anomaly/outlier detection algorithms including:
//! - Isolation Forest
//! - Local Outlier Factor (LOF)
//! - One-Class SVM
//! - Elliptic Envelope
//! - DBSCAN-based detection

mod isolation_forest;
mod lof;
mod statistical;

pub use isolation_forest::{IsolationForest, IsolationTree};
pub use lof::{LocalOutlierFactor, LOFResult};
pub use statistical::{EllipticEnvelope, StatisticalDetector, ZScoreDetector};

use crate::error::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Anomaly scores (higher = more anomalous)
    pub scores: Array1<f64>,
    /// Binary labels (-1 = anomaly, 1 = normal)
    pub labels: Array1<i32>,
    /// Threshold used for classification
    pub threshold: f64,
    /// Number of anomalies detected
    pub n_anomalies: usize,
}

/// Trait for anomaly detectors
pub trait AnomalyDetector: Send + Sync {
    /// Fit the detector on training data
    fn fit(&mut self, x: &Array2<f64>) -> Result<()>;
    
    /// Compute anomaly scores for new data
    fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
    
    /// Predict labels (-1 = anomaly, 1 = normal)
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>>;
    
    /// Fit and predict in one step
    fn fit_predict(&mut self, x: &Array2<f64>) -> Result<Array1<i32>> {
        self.fit(x)?;
        self.predict(x)
    }
    
    /// Get detection results with scores and labels
    fn detect(&self, x: &Array2<f64>) -> Result<AnomalyResult> {
        let scores = self.score_samples(x)?;
        let labels = self.predict(x)?;
        let threshold = self.threshold();
        let n_anomalies = labels.iter().filter(|&&l| l == -1).count();
        
        Ok(AnomalyResult {
            scores,
            labels,
            threshold,
            n_anomalies,
        })
    }
    
    /// Get the decision threshold
    fn threshold(&self) -> f64;
}
