//! Linear model implementations

use crate::error::{KolosalError, Result};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Simple matrix inversion for small matrices using Gauss-Jordan elimination
fn matrix_inverse(m: &Array2<f64>) -> Option<Array2<f64>> {
    let n = m.nrows();
    if n != m.ncols() {
        return None;
    }
    
    // Create augmented matrix [M | I]
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = m[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }
    
    // Gauss-Jordan elimination
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        for row in col + 1..n {
            if aug[[row, col]].abs() > aug[[max_row, col]].abs() {
                max_row = row;
            }
        }
        
        // Swap rows
        if max_row != col {
            for j in 0..2 * n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        
        // Check for singularity
        if aug[[col, col]].abs() < 1e-10 {
            return None;
        }
        
        // Scale pivot row
        let pivot = aug[[col, col]];
        for j in 0..2 * n {
            aug[[col, j]] /= pivot;
        }
        
        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..2 * n {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }
    
    // Extract inverse from right half
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    
    Some(inv)
}

/// Solve least squares via pseudo-inverse: (X^T X)^-1 X^T y
fn solve_least_squares(x: &Array2<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);
    
    match matrix_inverse(&xtx) {
        Some(inv) => Some(inv.dot(&xty)),
        None => None,
    }
}

/// Linear regression model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    /// Fitted coefficients (weights)
    pub coefficients: Option<Array1<f64>>,
    /// Fitted intercept (bias)
    pub intercept: Option<f64>,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Regularization strength (L2)
    pub alpha: f64,
    /// Whether model is fitted
    pub is_fitted: bool,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegression {
    /// Create a new linear regression model
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            alpha: 0.0,
            is_fitted: false,
        }
    }

    /// Enable/disable fitting intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set regularization strength (Ridge regression)
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Fit the model to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(KolosalError::ShapeError {
                expected: format!("y length = {}", n_samples),
                actual: format!("y length = {}", y.len()),
            });
        }

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap_or(0.0);
            
            let x_centered = x - &x_mean.clone().insert_axis(Axis(0));
            let y_centered = y - y_mean;
            
            (x_centered, y_centered, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        // Solve normal equations: (X^T X + alpha*I) * w = X^T y
        let coefficients = if self.alpha > 0.0 {
            // Ridge regression: add regularization
            let mut xtx = x_centered.t().dot(&x_centered);
            for i in 0..n_features {
                xtx[[i, i]] += self.alpha;
            }
            let xty = x_centered.t().dot(&y_centered);
            
            match matrix_inverse(&xtx) {
                Some(inv) => inv.dot(&xty),
                None => {
                    return Err(KolosalError::ComputationError(
                        "Matrix is singular, cannot compute inverse".to_string()
                    ));
                }
            }
        } else {
            // OLS
            match solve_least_squares(&x_centered, &y_centered) {
                Some(coef) => coef,
                None => {
                    return Err(KolosalError::ComputationError(
                        "Matrix is singular, cannot solve least squares".to_string()
                    ));
                }
            }
        };

        // Compute intercept
        let intercept = if self.fit_intercept {
            let x_mean = x_mean.unwrap();
            let y_mean = y_mean.unwrap();
            Some(y_mean - coefficients.dot(&x_mean))
        } else {
            Some(0.0)
        };

        self.coefficients = Some(coefficients);
        self.intercept = intercept;
        self.is_fitted = true;

        Ok(self)
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }

        let coefficients = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        let predictions = x.dot(coefficients) + intercept;
        Ok(predictions)
    }

    /// Get R² score
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let y_pred = self.predict(x)?;
        
        let y_mean = y.mean().unwrap_or(0.0);
        let ss_res = (&y_pred - y).mapv(|v| v * v).sum();
        let ss_tot = y.mapv(|v| (v - y_mean) * (v - y_mean)).sum();
        
        if ss_tot == 0.0 {
            return Ok(1.0);
        }
        
        Ok(1.0 - ss_res / ss_tot)
    }
}

/// Logistic regression for binary classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    /// Fitted coefficients
    pub coefficients: Option<Array1<f64>>,
    /// Fitted intercept
    pub intercept: Option<f64>,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Regularization strength (L2)
    pub alpha: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Whether model is fitted
    pub is_fitted: bool,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LogisticRegression {
    /// Create a new logistic regression model
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            alpha: 0.01,
            max_iter: 1000,
            tol: 1e-6,
            learning_rate: 0.1,
            is_fitted: false,
        }
    }

    /// Set regularization strength
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sigmoid function
    fn sigmoid(z: &Array1<f64>) -> Array1<f64> {
        z.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// Fit the model using gradient descent
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(KolosalError::ShapeError {
                expected: format!("y length = {}", n_samples),
                actual: format!("y length = {}", y.len()),
            });
        }

        // Initialize weights
        let mut weights = Array1::zeros(n_features);
        let mut bias = 0.0;

        let lr = self.learning_rate;
        let alpha = self.alpha;

        // Gradient descent
        for _iter in 0..self.max_iter {
            // Forward pass
            let linear = x.dot(&weights) + bias;
            let predictions = Self::sigmoid(&linear);

            // Compute gradients
            let errors = &predictions - y;
            let dw = (x.t().dot(&errors) / n_samples as f64) + (alpha * &weights);
            let db = errors.mean().unwrap_or(0.0);

            // Check convergence
            let grad_norm = (dw.mapv(|v| v * v).sum() + db * db).sqrt();
            if grad_norm < self.tol {
                break;
            }

            // Update weights
            weights = weights - lr * dw;
            bias = bias - lr * db;
        }

        self.coefficients = Some(weights);
        self.intercept = Some(bias);
        self.is_fitted = true;

        Ok(self)
    }

    /// Predict probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }

        let coefficients = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        let linear = x.dot(coefficients) + intercept;
        Ok(Self::sigmoid(&linear))
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let proba = self.predict_proba(x)?;
        Ok(proba.mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 }))
    }

    /// Get accuracy score
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let y_pred = self.predict(x)?;
        
        let correct = y_pred.iter().zip(y.iter())
            .filter(|(pred, actual)| (*pred - *actual).abs() < 0.5)
            .count();
        
        Ok(correct as f64 / y.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_regression_simple() {
        // y = 2*x1 + 3*x2 + 1
        let x = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 1.0],
        ];
        // y = 2*x1 + 3*x2 + 1
        let y = array![6.0, 8.0, 9.0, 11.0, 10.0];

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted);
        
        let r2 = model.score(&x, &y).unwrap();
        assert!(r2 > 0.99, "R² should be close to 1, got {}", r2);
    }

    #[test]
    fn test_ridge_regression() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ];
        let y = array![2.0, 4.0, 6.0];

        let mut model = LinearRegression::new().with_alpha(0.1);
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted);
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_logistic_regression() {
        // Simple linearly separable data
        let x = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0],
            [5.0, 5.0],
            [5.5, 5.5],
            [6.0, 6.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new()
            .with_max_iter(1000)
            .with_learning_rate(0.5);
        
        model.fit(&x, &y).unwrap();
        assert!(model.is_fitted);

        let accuracy = model.score(&x, &y).unwrap();
        assert!(accuracy >= 0.8, "Accuracy should be >= 0.8, got {}", accuracy);
    }

    #[test]
    fn test_predict_proba() {
        let x = array![
            [0.0, 0.0],
            [10.0, 10.0],
        ];
        let y = array![0.0, 1.0];

        let mut model = LogisticRegression::new().with_max_iter(500);
        model.fit(&x, &y).unwrap();

        let proba = model.predict_proba(&x).unwrap();
        
        // First sample should have low probability, second should have high
        assert!(proba[0] < 0.5);
        assert!(proba[1] > 0.5);
    }
}
