//! Linear model implementations

use crate::error::{KolosalError, Result};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Solve symmetric positive-definite system Ax = b using Cholesky decomposition.
/// Falls back to regularized solve if matrix is near-singular.
fn cholesky_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return None;
    }

    // Cholesky decomposition: A = L * L^T
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }

            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    // Not positive definite — add regularization and retry
                    let mut a_reg = a.clone();
                    let ridge = 1e-8 * a.diag().iter().map(|v| v.abs()).sum::<f64>() / n as f64;
                    for k in 0..n {
                        a_reg[[k, k]] += ridge;
                    }
                    return cholesky_solve_inner(&a_reg, b);
                }
                l[[i, j]] = diag.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    // Forward substitution: L * y = b
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * y[j];
        }
        y[i] = (b[i] - sum) / l[[i, i]];
    }

    // Backward substitution: L^T * x = y
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j];
        }
        x[i] = (y[i] - sum) / l[[i, i]];
    }

    Some(x)
}

/// Inner Cholesky solve (no retry) for regularized matrix
fn cholesky_solve_inner(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    return None; // Still not PD even with regularization — fall back to Gauss-Jordan
                }
                l[[i, j]] = diag.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * y[j];
        }
        y[i] = (b[i] - sum) / l[[i, i]];
    }

    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j];
        }
        x[i] = (y[i] - sum) / l[[i, i]];
    }

    Some(x)
}

/// Simple matrix inversion for small matrices using Gauss-Jordan elimination (fallback)
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

/// Solve least squares via normal equations: (X^T X) w = X^T y
/// Uses Cholesky decomposition (O(n^3/3)) with fallback to Gauss-Jordan
fn solve_least_squares(x: &Array2<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);

    // Try Cholesky first (faster, more stable for well-conditioned systems)
    if let Some(result) = cholesky_solve(&xtx, &xty) {
        return Some(result);
    }

    // Fallback to Gauss-Jordan via explicit inverse
    match matrix_inverse(&xtx) {
        Some(inv) => Some(inv.dot(&xty)),
        None => None,
    }
}

/// Solve regularized least squares: (X^T X + alpha*I) w = X^T y
/// Uses Cholesky first (the regularized system is usually positive definite), falls back to inverse.
fn solve_ridge(x: &Array2<f64>, y: &Array1<f64>, alpha: f64) -> Option<Array1<f64>> {
    let n_features = x.ncols();
    let mut xtx = x.t().dot(x);
    for i in 0..n_features {
        xtx[[i, i]] += alpha;
    }
    let xty = x.t().dot(y);

    if let Some(result) = cholesky_solve(&xtx, &xty) {
        return Some(result);
    }

    match matrix_inverse(&xtx) {
        Some(inv) => Some(inv.dot(&xty)),
        None => None,
    }
}

/// Center X and y for intercept fitting, returning (x_centered, y_centered, x_mean, y_mean).
/// For fit_intercept=false, returns views of the original data (zero-cost via clone of the view).
fn center_data(
    x: &Array2<f64>,
    y: &Array1<f64>,
    fit_intercept: bool,
) -> (Array2<f64>, Array1<f64>, Option<Array1<f64>>, Option<f64>) {
    if fit_intercept {
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean().unwrap_or(0.0);
        let x_centered = x - &x_mean.clone().insert_axis(Axis(0));
        let y_centered = y - y_mean;
        (x_centered, y_centered, Some(x_mean), Some(y_mean))
    } else {
        (x.view().to_owned(), y.view().to_owned(), None, None)
    }
}

/// Compute intercept from centered coefficients
fn compute_intercept(coefficients: &Array1<f64>, x_mean: Option<&Array1<f64>>, y_mean: Option<f64>) -> f64 {
    match (x_mean, y_mean) {
        (Some(xm), Some(ym)) => ym - coefficients.dot(xm),
        _ => 0.0,
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

        if n_samples != y.len() {
            return Err(KolosalError::ShapeError {
                expected: format!("y length = {}", n_samples),
                actual: format!("y length = {}", y.len()),
            });
        }

        let (x_c, y_c, x_mean, y_mean) = center_data(x, y, self.fit_intercept);

        let coefficients = if self.alpha > 0.0 {
            // Ridge regression: use cholesky_solve first, fall back to inverse
            match solve_ridge(&x_c, &y_c, self.alpha) {
                Some(coef) => coef,
                None => {
                    return Err(KolosalError::ComputationError(
                        "Matrix is singular, cannot compute inverse".to_string()
                    ));
                }
            }
        } else {
            match solve_least_squares(&x_c, &y_c) {
                Some(coef) => coef,
                None => {
                    return Err(KolosalError::ComputationError(
                        "Matrix is singular, cannot solve least squares".to_string()
                    ));
                }
            }
        };

        let intercept = compute_intercept(&coefficients, x_mean.as_ref(), y_mean);
        self.coefficients = Some(coefficients);
        self.intercept = Some(intercept);
        self.is_fitted = true;

        Ok(self)
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }

        let coefficients = self.coefficients.as_ref().ok_or(KolosalError::ModelNotFitted)?;
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

        let coefficients = self.coefficients.as_ref().ok_or(KolosalError::ModelNotFitted)?;
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

/// Ridge Regression (L2-regularized linear regression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeRegression {
    pub coefficients: Option<Array1<f64>>,
    pub intercept: Option<f64>,
    pub fit_intercept: bool,
    /// L2 regularization strength
    pub alpha: f64,
    pub is_fitted: bool,
}

impl Default for RidgeRegression {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl RidgeRegression {
    pub fn new(alpha: f64) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            alpha,
            is_fitted: false,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        if n_samples != y.len() {
            return Err(KolosalError::ShapeError {
                expected: format!("y length = {}", n_samples),
                actual: format!("y length = {}", y.len()),
            });
        }

        let (x_c, y_c, x_mean, y_mean) = center_data(x, y, self.fit_intercept);

        let coefficients = match solve_ridge(&x_c, &y_c, self.alpha) {
            Some(coef) => coef,
            None => return Err(KolosalError::ComputationError("Singular matrix".to_string())),
        };

        self.intercept = Some(compute_intercept(&coefficients, x_mean.as_ref(), y_mean));
        self.coefficients = Some(coefficients);
        self.is_fitted = true;
        Ok(self)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }
        Ok(x.dot(self.coefficients.as_ref().ok_or(KolosalError::ModelNotFitted)?) + self.intercept.unwrap_or(0.0))
    }

    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let p = self.predict(x)?;
        let ym = y.mean().unwrap_or(0.0);
        let ss_res = (&p - y).mapv(|v| v * v).sum();
        let ss_tot = y.mapv(|v| (v - ym).powi(2)).sum();
        Ok(if ss_tot == 0.0 { 1.0 } else { 1.0 - ss_res / ss_tot })
    }
}

/// Lasso Regression (L1-regularized via coordinate descent)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LassoRegression {
    pub coefficients: Option<Array1<f64>>,
    pub intercept: Option<f64>,
    pub fit_intercept: bool,
    /// L1 regularization strength
    pub alpha: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub is_fitted: bool,
}

impl Default for LassoRegression {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LassoRegression {
    pub fn new(alpha: f64) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            alpha,
            max_iter: 1000,
            tol: 1e-6,
            is_fitted: false,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Soft-threshold operator for L1 proximal step
    fn soft_threshold(val: f64, threshold: f64) -> f64 {
        if val > threshold {
            val - threshold
        } else if val < -threshold {
            val + threshold
        } else {
            0.0
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if n_samples != y.len() {
            return Err(KolosalError::ShapeError {
                expected: format!("y length = {}", n_samples),
                actual: format!("y length = {}", y.len()),
            });
        }

        let (x_c, y_c, x_mean, y_mean) = center_data(x, y, self.fit_intercept);

        // Pre-compute column norms
        let col_norms: Vec<f64> = (0..n_features)
            .map(|j| x_c.column(j).mapv(|v| v * v).sum())
            .collect();

        let mut w = Array1::zeros(n_features);
        let lambda = self.alpha * n_samples as f64;

        for _iter in 0..self.max_iter {
            let w_old = w.clone();

            // Coordinate descent
            for j in 0..n_features {
                if col_norms[j] < 1e-15 {
                    w[j] = 0.0;
                    continue;
                }
                // Compute partial residual without feature j
                let r = &y_c - &x_c.dot(&w) + &(&x_c.column(j) * w[j]);
                let rho = x_c.column(j).dot(&r);
                w[j] = Self::soft_threshold(rho, lambda) / col_norms[j];
            }

            // Check convergence
            let diff = (&w - &w_old).mapv(|v| v.abs()).sum();
            if diff < self.tol {
                break;
            }
        }

        self.intercept = Some(compute_intercept(&w, x_mean.as_ref(), y_mean));
        self.coefficients = Some(w);
        self.is_fitted = true;
        Ok(self)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }
        Ok(x.dot(self.coefficients.as_ref().ok_or(KolosalError::ModelNotFitted)?) + self.intercept.unwrap_or(0.0))
    }

    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let p = self.predict(x)?;
        let ym = y.mean().unwrap_or(0.0);
        let ss_res = (&p - y).mapv(|v| v * v).sum();
        let ss_tot = y.mapv(|v| (v - ym).powi(2)).sum();
        Ok(if ss_tot == 0.0 { 1.0 } else { 1.0 - ss_res / ss_tot })
    }
}

/// Elastic Net Regression (L1 + L2 regularization via coordinate descent)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticNetRegression {
    pub coefficients: Option<Array1<f64>>,
    pub intercept: Option<f64>,
    pub fit_intercept: bool,
    /// Overall regularization strength
    pub alpha: f64,
    /// L1 ratio (0.0 = pure L2/Ridge, 1.0 = pure L1/Lasso)
    pub l1_ratio: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub is_fitted: bool,
}

impl Default for ElasticNetRegression {
    fn default() -> Self {
        Self::new(1.0, 0.5)
    }
}

impl ElasticNetRegression {
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            alpha,
            l1_ratio: l1_ratio.clamp(0.0, 1.0),
            max_iter: 1000,
            tol: 1e-6,
            is_fitted: false,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio.clamp(0.0, 1.0);
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if n_samples != y.len() {
            return Err(KolosalError::ShapeError {
                expected: format!("y length = {}", n_samples),
                actual: format!("y length = {}", y.len()),
            });
        }

        let (x_c, y_c, x_mean, y_mean) = center_data(x, y, self.fit_intercept);

        let col_norms: Vec<f64> = (0..n_features)
            .map(|j| x_c.column(j).mapv(|v| v * v).sum())
            .collect();

        let mut w = Array1::zeros(n_features);
        let n = n_samples as f64;
        let l1_penalty = self.alpha * self.l1_ratio * n;
        let l2_penalty = self.alpha * (1.0 - self.l1_ratio) * n;

        for _iter in 0..self.max_iter {
            let w_old = w.clone();

            for j in 0..n_features {
                let denom = col_norms[j] + l2_penalty;
                if denom < 1e-15 {
                    w[j] = 0.0;
                    continue;
                }
                let r = &y_c - &x_c.dot(&w) + &(&x_c.column(j) * w[j]);
                let rho = x_c.column(j).dot(&r);
                w[j] = LassoRegression::soft_threshold(rho, l1_penalty) / denom;
            }

            let diff = (&w - &w_old).mapv(|v| v.abs()).sum();
            if diff < self.tol {
                break;
            }
        }

        self.intercept = Some(compute_intercept(&w, x_mean.as_ref(), y_mean));
        self.coefficients = Some(w);
        self.is_fitted = true;
        Ok(self)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }
        Ok(x.dot(self.coefficients.as_ref().ok_or(KolosalError::ModelNotFitted)?) + self.intercept.unwrap_or(0.0))
    }

    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let p = self.predict(x)?;
        let ym = y.mean().unwrap_or(0.0);
        let ss_res = (&p - y).mapv(|v| v * v).sum();
        let ss_tot = y.mapv(|v| (v - ym).powi(2)).sum();
        Ok(if ss_tot == 0.0 { 1.0 } else { 1.0 - ss_res / ss_tot })
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

    #[test]
    fn test_ridge_regression_dedicated() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let mut model = RidgeRegression::new(0.1);
        model.fit(&x, &y).unwrap();
        assert!(model.is_fitted);
        let r2 = model.score(&x, &y).unwrap();
        assert!(r2 > 0.95, "Ridge R² = {}", r2);
    }

    #[test]
    fn test_lasso_regression() {
        let x = array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let mut model = LassoRegression::new(0.01);
        model.fit(&x, &y).unwrap();
        assert!(model.is_fitted);
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        // With small alpha, should fit well
        let r2 = model.score(&x, &y).unwrap();
        assert!(r2 > 0.9, "Lasso R² = {}", r2);
    }

    #[test]
    fn test_elastic_net() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0];
        let mut model = ElasticNetRegression::new(0.01, 0.5);
        model.fit(&x, &y).unwrap();
        assert!(model.is_fitted);
        let r2 = model.score(&x, &y).unwrap();
        assert!(r2 > 0.9, "ElasticNet R² = {}", r2);
    }
}

// ============ Polynomial Regression ============

/// Polynomial Regression — expands features to polynomial degree then fits with Ridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialRegression {
    pub degree: usize,
    pub alpha: f64,
    inner: RidgeRegression,
    pub is_fitted: bool,
    n_original_features: usize,
}

impl PolynomialRegression {
    pub fn new(degree: usize, alpha: f64) -> Self {
        Self {
            degree: degree.max(1),
            alpha,
            inner: RidgeRegression::new(alpha),
            is_fitted: false,
            n_original_features: 0,
        }
    }

    /// Expand features to polynomial terms up to given degree.
    /// Uses Array2::from_shape_fn to fill directly — no intermediate Vec<Array1> allocation.
    fn expand_features(x: &Array2<f64>, degree: usize) -> Array2<f64> {
        let n = x.nrows();
        let p = x.ncols();

        if degree == 1 {
            return x.clone();
        }

        // Count total output columns
        let mut n_cols = p; // degree 1
        if degree >= 2 {
            n_cols += p * (p + 1) / 2; // degree 2 interactions
        }
        for _d in 3..=degree {
            n_cols += p; // higher powers
        }

        // Build a column descriptor: (type, params) for each output column
        // Instead of materializing columns, we compute on the fly in from_shape_fn
        // by mapping output column index → source computation.
        //
        // Column layout:
        //   [0..p)                              → x[:,j]
        //   [p..p+p*(p+1)/2)                    → x[:,i]*x[:,j] for i<=j
        //   [p+p*(p+1)/2..p+p*(p+1)/2+p)       → x[:,j]^3
        //   ...and so on for higher degrees

        let deg2_start = p;
        let deg2_count = if degree >= 2 { p * (p + 1) / 2 } else { 0 };
        let higher_start = deg2_start + deg2_count;

        Array2::from_shape_fn((n, n_cols), |(row, col)| {
            if col < p {
                // Degree 1: original feature
                x[[row, col]]
            } else if col < higher_start {
                // Degree 2: interaction term
                let offset = col - deg2_start;
                // Map linear offset → (i, j) pair where i <= j
                // offset = i*p - i*(i-1)/2 + (j - i)
                let mut i = 0;
                let mut remaining = offset;
                loop {
                    let row_len = p - i;
                    if remaining < row_len {
                        let j = i + remaining;
                        return x[[row, i]] * x[[row, j]];
                    }
                    remaining -= row_len;
                    i += 1;
                }
            } else {
                // Degree 3+: higher power
                let offset = col - higher_start;
                let d = 3 + offset / p;
                let j = offset % p;
                x[[row, j]].powi(d as i32)
            }
        })
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.n_original_features = x.ncols();
        let x_poly = Self::expand_features(x, self.degree);
        self.inner.fit(&x_poly, y)?;
        self.is_fitted = true;
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::TrainingError("Model not fitted".into()));
        }
        let x_poly = Self::expand_features(x, self.degree);
        self.inner.predict(&x_poly)
    }

    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let preds = self.predict(x)?;
        let mean_y = y.mean().unwrap_or(0.0);
        let ss_res: f64 = y.iter().zip(preds.iter()).map(|(&yi, &pi)| (yi - pi).powi(2)).sum();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        if ss_tot == 0.0 { return Ok(1.0); }
        Ok(1.0 - ss_res / ss_tot)
    }
}

#[cfg(test)]
mod poly_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_polynomial_regression_quadratic() {
        // y = x^2
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0];
        let mut model = PolynomialRegression::new(2, 0.001);
        model.fit(&x, &y).unwrap();
        let r2 = model.score(&x, &y).unwrap();
        assert!(r2 > 0.99, "Poly R² = {}", r2);
    }

    #[test]
    fn test_polynomial_regression_multivariate() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![5.0, 13.0, 25.0, 41.0]; // ~ x1^2 + x2^2
        let mut model = PolynomialRegression::new(2, 0.01);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_polynomial_feature_expansion() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let expanded = PolynomialRegression::expand_features(&x, 2);
        // degree 1: x1, x2 (2) + degree 2: x1^2, x1*x2, x2^2 (3) = 5 total
        assert_eq!(expanded.ncols(), 5);
    }
}
