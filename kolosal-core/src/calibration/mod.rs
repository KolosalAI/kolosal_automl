//! Model calibration module
//!
//! Provides probability calibration methods including:
//! - Platt scaling (sigmoid calibration)
//! - Isotonic regression
//! - Temperature scaling
//! - Beta calibration
//! - Calibration metrics (ECE, MCE, Brier score)

mod platt;
mod isotonic;
mod temperature;
mod metrics;

pub use platt::PlattScaling;
pub use isotonic::IsotonicRegression;
pub use temperature::TemperatureScaling;
pub use metrics::{CalibrationMetrics, expected_calibration_error, maximum_calibration_error, brier_score, reliability_diagram};

use crate::error::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Calibration method type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CalibrationType {
    /// Platt scaling (logistic regression)
    Platt,
    /// Isotonic regression (non-parametric)
    Isotonic,
    /// Temperature scaling (single parameter)
    Temperature,
    /// Beta calibration
    Beta,
}

/// Trait for probability calibrators
pub trait Calibrator: Send + Sync {
    /// Fit the calibrator on predicted probabilities and true labels
    fn fit(&mut self, probs: &Array1<f64>, labels: &Array1<f64>) -> Result<()>;
    
    /// Calibrate probabilities
    fn calibrate(&self, probs: &Array1<f64>) -> Result<Array1<f64>>;
    
    /// Fit and calibrate in one step
    fn fit_calibrate(&mut self, probs: &Array1<f64>, labels: &Array1<f64>) -> Result<Array1<f64>> {
        self.fit(probs, labels)?;
        self.calibrate(probs)
    }
}
