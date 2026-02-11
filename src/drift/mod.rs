//! Drift detection module
//!
//! Detects data drift, concept drift, and feature drift for monitoring
//! ML models in production.

mod data_drift;
mod concept_drift;
mod feature_drift;

pub use data_drift::{DataDriftDetector, KolmogorovSmirnovTest, PopulationStabilityIndex, JensenShannonDivergence};
pub use concept_drift::{ConceptDriftDetector, DDM, ADWIN, PageHinkley, EDDM};
pub use feature_drift::{FeatureDriftMonitor, DriftReport, FeatureDriftResult};

use crate::error::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Drift detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftResult {
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Drift score/statistic
    pub score: f64,
    /// P-value (if applicable)
    pub p_value: Option<f64>,
    /// Threshold used for detection
    pub threshold: f64,
    /// Drift severity (0=none, 1=warning, 2=critical)
    pub severity: u8,
    /// Additional information
    pub message: String,
}

impl DriftResult {
    /// Create a result indicating no drift
    pub fn no_drift(score: f64, threshold: f64) -> Self {
        Self {
            drift_detected: false,
            score,
            p_value: None,
            threshold,
            severity: 0,
            message: "No drift detected".to_string(),
        }
    }

    /// Create a result indicating drift
    pub fn drift(score: f64, threshold: f64, severity: u8, message: &str) -> Self {
        Self {
            drift_detected: true,
            score,
            p_value: None,
            threshold,
            severity,
            message: message.to_string(),
        }
    }
}

/// Trait for drift detectors
pub trait DriftDetector: Send + Sync {
    /// Detect drift between reference and test data
    fn detect(&self, reference: &Array1<f64>, test: &Array1<f64>) -> Result<DriftResult>;
    
    /// Get the threshold used for detection
    fn threshold(&self) -> f64;
    
    /// Reset the detector state (for online methods)
    fn reset(&mut self);
}
