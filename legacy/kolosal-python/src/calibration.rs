//! Python bindings for probability calibration

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Platt Scaling for Python
#[pyclass(name = "PlattScaling")]
pub struct PyPlattScaling {
    a: Option<f64>,
    b: Option<f64>,
}

#[pymethods]
impl PyPlattScaling {
    #[new]
    fn new() -> Self {
        Self { a: None, b: None }
    }

    /// Fit the calibrator
    fn fit(&mut self, probas: Vec<f64>, y_true: Vec<f64>) -> PyResult<()> {
        // Simple logistic regression fitting via gradient descent
        let mut a = 0.0f64;
        let mut b = 0.0f64;
        let lr = 0.01;
        
        for _ in 0..1000 {
            let mut grad_a = 0.0;
            let mut grad_b = 0.0;
            
            for (p, y) in probas.iter().zip(y_true.iter()) {
                let logit = a * p + b;
                let pred = 1.0 / (1.0 + (-logit).exp());
                let error = pred - y;
                grad_a += error * p;
                grad_b += error;
            }
            
            a -= lr * grad_a / probas.len() as f64;
            b -= lr * grad_b / probas.len() as f64;
        }
        
        self.a = Some(a);
        self.b = Some(b);
        Ok(())
    }

    /// Calibrate probabilities
    fn transform(&self, probas: Vec<f64>) -> PyResult<Vec<f64>> {
        let a = self.a.ok_or_else(|| PyValueError::new_err("Not fitted"))?;
        let b = self.b.ok_or_else(|| PyValueError::new_err("Not fitted"))?;
        
        Ok(probas.into_iter()
            .map(|p| 1.0 / (1.0 + (-(a * p + b)).exp()))
            .collect())
    }

    /// Fit and transform
    fn fit_transform(&mut self, probas: Vec<f64>, y_true: Vec<f64>) -> PyResult<Vec<f64>> {
        self.fit(probas.clone(), y_true)?;
        self.transform(probas)
    }

    /// Get calibration parameters (A, B)
    fn parameters(&self) -> Option<(f64, f64)> {
        match (self.a, self.b) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }

    /// Check if fitted
    fn is_fitted(&self) -> bool {
        self.a.is_some() && self.b.is_some()
    }
}

/// Isotonic Regression for Python
#[pyclass(name = "IsotonicRegression")]
pub struct PyIsotonicRegression {
    x_thresholds: Vec<f64>,
    y_thresholds: Vec<f64>,
}

#[pymethods]
impl PyIsotonicRegression {
    #[new]
    fn new() -> Self {
        Self {
            x_thresholds: Vec::new(),
            y_thresholds: Vec::new(),
        }
    }

    /// Fit the calibrator using pool adjacent violators algorithm
    fn fit(&mut self, probas: Vec<f64>, y_true: Vec<f64>) -> PyResult<()> {
        // Sort by probas
        let mut pairs: Vec<(f64, f64)> = probas.into_iter().zip(y_true.into_iter()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let x: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let y: Vec<f64> = pairs.iter().map(|p| p.1).collect();
        
        // Pool adjacent violators
        let mut blocks: Vec<(usize, usize, f64)> = vec![];  // (start, end, mean)
        
        for (i, &yi) in y.iter().enumerate() {
            blocks.push((i, i + 1, yi));
            
            while blocks.len() > 1 {
                let n = blocks.len();
                if blocks[n - 1].2 < blocks[n - 2].2 {
                    // Merge blocks
                    let last = blocks.pop().unwrap();
                    let prev = blocks.pop().unwrap();
                    let total_weight = (last.1 - last.0 + prev.1 - prev.0) as f64;
                    let merged_mean = (prev.2 * (prev.1 - prev.0) as f64 + last.2 * (last.1 - last.0) as f64) / total_weight;
                    blocks.push((prev.0, last.1, merged_mean));
                } else {
                    break;
                }
            }
        }
        
        // Extract thresholds
        self.x_thresholds = blocks.iter().map(|b| x[b.0]).collect();
        self.y_thresholds = blocks.iter().map(|b| b.2).collect();
        
        Ok(())
    }

    /// Calibrate probabilities
    fn transform(&self, probas: Vec<f64>) -> PyResult<Vec<f64>> {
        if self.x_thresholds.is_empty() {
            return Err(PyValueError::new_err("Not fitted"));
        }
        
        Ok(probas.into_iter()
            .map(|p| {
                // Find appropriate threshold
                let idx = self.x_thresholds.iter()
                    .position(|&x| x > p)
                    .unwrap_or(self.x_thresholds.len());
                
                if idx == 0 {
                    self.y_thresholds[0]
                } else if idx >= self.y_thresholds.len() {
                    *self.y_thresholds.last().unwrap()
                } else {
                    // Linear interpolation
                    let x0 = self.x_thresholds[idx - 1];
                    let x1 = self.x_thresholds[idx];
                    let y0 = self.y_thresholds[idx - 1];
                    let y1 = self.y_thresholds[idx];
                    y0 + (y1 - y0) * (p - x0) / (x1 - x0)
                }
            })
            .collect())
    }

    /// Fit and transform
    fn fit_transform(&mut self, probas: Vec<f64>, y_true: Vec<f64>) -> PyResult<Vec<f64>> {
        self.fit(probas.clone(), y_true)?;
        self.transform(probas)
    }

    /// Check if fitted
    fn is_fitted(&self) -> bool {
        !self.x_thresholds.is_empty()
    }
}

/// Temperature Scaling for Python
#[pyclass(name = "TemperatureScaling")]
pub struct PyTemperatureScaling {
    temperature: Option<f64>,
}

#[pymethods]
impl PyTemperatureScaling {
    #[new]
    fn new() -> Self {
        Self { temperature: None }
    }

    /// Fit to find optimal temperature
    fn fit(&mut self, logits: Vec<Vec<f64>>, y_true: Vec<usize>) -> PyResult<()> {
        // Grid search for optimal temperature
        let mut best_temp = 1.0;
        let mut best_loss = f64::INFINITY;
        
        for t in (1..100).map(|i| i as f64 * 0.1) {
            let loss = self.compute_nll(&logits, &y_true, t);
            if loss < best_loss {
                best_loss = loss;
                best_temp = t;
            }
        }
        
        self.temperature = Some(best_temp);
        Ok(())
    }

    /// Apply temperature scaling
    fn transform(&self, logits: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let temp = self.temperature.ok_or_else(|| PyValueError::new_err("Not fitted"))?;
        
        Ok(logits.into_iter()
            .map(|row| {
                let scaled: Vec<f64> = row.iter().map(|&l| l / temp).collect();
                let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = scaled.iter().map(|&s| (s - max_val).exp()).sum();
                scaled.iter().map(|&s| (s - max_val).exp() / exp_sum).collect()
            })
            .collect())
    }

    /// Get the learned temperature
    fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    /// Check if fitted
    fn is_fitted(&self) -> bool {
        self.temperature.is_some()
    }
}

impl PyTemperatureScaling {
    fn compute_nll(&self, logits: &[Vec<f64>], y_true: &[usize], temp: f64) -> f64 {
        let mut nll = 0.0;
        
        for (row, &y) in logits.iter().zip(y_true.iter()) {
            let scaled: Vec<f64> = row.iter().map(|&l| l / temp).collect();
            let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = scaled.iter().map(|&s| (s - max_val).exp()).sum();
            let log_prob = scaled[y] - max_val - exp_sum.ln();
            nll -= log_prob;
        }
        
        nll / logits.len() as f64
    }
}

/// Calibrator type enum for Python
#[pyclass(name = "CalibratorType")]
#[derive(Clone)]
pub enum PyCalibratorType {
    Platt,
    Isotonic,
    Temperature,
}
