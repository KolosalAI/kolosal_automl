//! Python bindings for data drift detection

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Data drift detector for Python
#[pyclass(name = "DataDriftDetector")]
pub struct PyDataDriftDetector {
    reference_stats: Option<Vec<(f64, f64)>>,  // (mean, std) per feature
    threshold: f64,
}

#[pymethods]
impl PyDataDriftDetector {
    #[new]
    #[pyo3(signature = (threshold=0.1))]
    fn new(threshold: f64) -> Self {
        Self {
            reference_stats: None,
            threshold,
        }
    }

    /// Fit on reference data
    fn fit(&mut self, reference: Vec<Vec<f64>>) -> PyResult<()> {
        if reference.is_empty() {
            return Err(PyValueError::new_err("Empty reference data"));
        }
        
        let n_features = reference[0].len();
        let n_samples = reference.len() as f64;
        
        let mut stats = Vec::with_capacity(n_features);
        
        for feat in 0..n_features {
            let values: Vec<f64> = reference.iter().map(|r| r[feat]).collect();
            let mean = values.iter().sum::<f64>() / n_samples;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_samples;
            stats.push((mean, variance.sqrt()));
        }
        
        self.reference_stats = Some(stats);
        Ok(())
    }

    /// Detect drift in current data
    fn detect(&self, py: Python<'_>, current: Vec<Vec<f64>>) -> PyResult<PyObject> {
        let stats = self.reference_stats.as_ref()
            .ok_or_else(|| PyValueError::new_err("Not fitted"))?;
        
        if current.is_empty() {
            return Err(PyValueError::new_err("Empty current data"));
        }
        
        let n_features = current[0].len();
        let n_samples = current.len() as f64;
        
        let mut drifted_features = Vec::new();
        let mut drift_scores = Vec::with_capacity(n_features);
        
        for feat in 0..n_features {
            let values: Vec<f64> = current.iter().map(|r| r[feat]).collect();
            let curr_mean = values.iter().sum::<f64>() / n_samples;
            let curr_var = values.iter().map(|v| (v - curr_mean).powi(2)).sum::<f64>() / n_samples;
            let curr_std = curr_var.sqrt();
            
            let (ref_mean, ref_std) = stats[feat];
            
            // PSI approximation
            let mean_diff = (curr_mean - ref_mean).abs() / (ref_std + 1e-10);
            let std_ratio = (curr_std / (ref_std + 1e-10) - 1.0).abs();
            let score = mean_diff + std_ratio;
            
            drift_scores.push(score);
            if score > self.threshold {
                drifted_features.push(feat);
            }
        }
        
        let is_drift = !drifted_features.is_empty();
        
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("is_drift", is_drift)?;
        dict.set_item("drifted_features", drifted_features)?;
        dict.set_item("drift_scores", drift_scores)?;
        
        Ok(dict.into())
    }

    /// Get threshold
    fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Check if fitted
    fn is_fitted(&self) -> bool {
        self.reference_stats.is_some()
    }
}

/// Concept drift detector for Python (Page-Hinkley test)
#[pyclass(name = "ConceptDriftDetector")]
pub struct PyConceptDriftDetector {
    delta: f64,
    lambda_: f64,
    cumulative_sum: f64,
    min_sum: f64,
    mean: f64,
    n_samples: usize,
}

#[pymethods]
impl PyConceptDriftDetector {
    #[new]
    #[pyo3(signature = (delta=0.005, lambda_=50.0))]
    fn new(delta: f64, lambda_: f64) -> Self {
        Self {
            delta,
            lambda_,
            cumulative_sum: 0.0,
            min_sum: 0.0,
            mean: 0.0,
            n_samples: 0,
        }
    }

    /// Update with new error/metric value and check for drift
    fn update(&mut self, value: f64) -> bool {
        self.n_samples += 1;
        self.mean = self.mean + (value - self.mean) / self.n_samples as f64;
        self.cumulative_sum = self.cumulative_sum + value - self.mean - self.delta;
        self.min_sum = self.min_sum.min(self.cumulative_sum);
        
        self.cumulative_sum - self.min_sum > self.lambda_
    }

    /// Batch update
    fn update_batch(&mut self, values: Vec<f64>) -> Vec<bool> {
        values.into_iter().map(|v| self.update(v)).collect()
    }

    /// Reset the detector
    fn reset(&mut self) {
        self.cumulative_sum = 0.0;
        self.min_sum = 0.0;
        self.mean = 0.0;
        self.n_samples = 0;
    }

    /// Get current state
    fn get_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("n_samples", self.n_samples)?;
        dict.set_item("mean", self.mean)?;
        dict.set_item("cumulative_sum", self.cumulative_sum)?;
        dict.set_item("min_sum", self.min_sum)?;
        Ok(dict.into())
    }
}

/// Feature drift monitor for Python
#[pyclass(name = "FeatureDriftMonitor")]
pub struct PyFeatureDriftMonitor {
    reference_distributions: Option<Vec<Vec<f64>>>,
    n_bins: usize,
}

#[pymethods]
impl PyFeatureDriftMonitor {
    #[new]
    #[pyo3(signature = (n_bins=10))]
    fn new(n_bins: usize) -> Self {
        Self {
            reference_distributions: None,
            n_bins,
        }
    }

    /// Fit on reference data
    fn fit(&mut self, reference: Vec<Vec<f64>>) -> PyResult<()> {
        if reference.is_empty() {
            return Err(PyValueError::new_err("Empty reference data"));
        }
        
        let n_features = reference[0].len();
        let mut distributions = Vec::with_capacity(n_features);
        
        for feat in 0..n_features {
            let values: Vec<f64> = reference.iter().map(|r| r[feat]).collect();
            let hist = self.compute_histogram(&values);
            distributions.push(hist);
        }
        
        self.reference_distributions = Some(distributions);
        Ok(())
    }

    /// Compute PSI for each feature
    fn compute_psi(&self, current: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let ref_dists = self.reference_distributions.as_ref()
            .ok_or_else(|| PyValueError::new_err("Not fitted"))?;
        
        if current.is_empty() {
            return Err(PyValueError::new_err("Empty current data"));
        }
        
        let n_features = current[0].len();
        let mut psi_scores = Vec::with_capacity(n_features);
        
        for feat in 0..n_features {
            let values: Vec<f64> = current.iter().map(|r| r[feat]).collect();
            let curr_hist = self.compute_histogram(&values);
            
            let psi = self.compute_psi_score(&ref_dists[feat], &curr_hist);
            psi_scores.push(psi);
        }
        
        Ok(psi_scores)
    }

    /// Check for significant drift
    fn check_drift(&self, py: Python<'_>, current: Vec<Vec<f64>>, threshold: f64) -> PyResult<PyObject> {
        let psi_scores = self.compute_psi(current)?;
        
        let drifted_features: Vec<usize> = psi_scores.iter()
            .enumerate()
            .filter(|(_, &psi)| psi > threshold)
            .map(|(i, _)| i)
            .collect();
        
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("psi_scores", psi_scores)?;
        dict.set_item("drifted_features", drifted_features.clone())?;
        dict.set_item("is_drift", !drifted_features.is_empty())?;
        
        Ok(dict.into())
    }
}

impl PyFeatureDriftMonitor {
    fn compute_histogram(&self, values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return vec![0.0; self.n_bins];
        }
        
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val + 1e-10;
        
        let mut counts = vec![0usize; self.n_bins];
        for &v in values {
            let bin = ((v - min_val) / range * self.n_bins as f64) as usize;
            let bin = bin.min(self.n_bins - 1);
            counts[bin] += 1;
        }
        
        let total = values.len() as f64;
        counts.into_iter().map(|c| c as f64 / total + 1e-10).collect()
    }
    
    fn compute_psi_score(&self, expected: &[f64], actual: &[f64]) -> f64 {
        expected.iter().zip(actual.iter())
            .map(|(&e, &a)| (a - e) * (a / e).ln())
            .sum()
    }
}
