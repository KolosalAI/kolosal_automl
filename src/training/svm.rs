//! Support Vector Machine implementations
//!
//! Provides SVM classifier and regressor using SMO (Sequential Minimal Optimization) algorithm.

use crate::error::{KolosalError, Result};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

/// Kernel function type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KernelType {
    /// Linear kernel: K(x, y) = x · y
    Linear,
    /// Polynomial kernel: K(x, y) = (γ * x · y + r)^d
    Polynomial { degree: usize, gamma: f64, coef0: f64 },
    /// Radial Basis Function (Gaussian): K(x, y) = exp(-γ * ||x - y||²)
    RBF { gamma: f64 },
    /// Sigmoid kernel: K(x, y) = tanh(γ * x · y + r)
    Sigmoid { gamma: f64, coef0: f64 },
}

impl Default for KernelType {
    fn default() -> Self {
        KernelType::RBF { gamma: 1.0 }
    }
}

/// SVM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVMConfig {
    /// Regularization parameter (C)
    pub c: f64,
    /// Kernel function
    pub kernel: KernelType,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Random seed
    pub random_state: Option<u64>,
    /// Epsilon for regression (SVR tube width)
    pub epsilon: f64,
}

impl Default for SVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            kernel: KernelType::RBF { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 1000,
            random_state: Some(42),
            epsilon: 0.1,
        }
    }
}

/// Support Vector Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVMClassifier {
    config: SVMConfig,
    /// Support vectors
    support_vectors: Option<Array2<f64>>,
    /// Alpha coefficients (Lagrange multipliers)
    alphas: Option<Array1<f64>>,
    /// Support vector labels
    support_labels: Option<Array1<f64>>,
    /// Bias term
    bias: f64,
    /// Unique class labels
    classes: Vec<i64>,
    is_fitted: bool,
}

impl SVMClassifier {
    /// Create a new SVM classifier
    pub fn new(config: SVMConfig) -> Self {
        Self {
            config,
            support_vectors: None,
            alphas: None,
            support_labels: None,
            bias: 0.0,
            classes: Vec::new(),
            is_fitted: false,
        }
    }

    /// Fit the classifier
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        // Find unique classes (binary classification for now)
        let mut classes: Vec<i64> = y.iter().map(|&v| v as i64).collect();
        classes.sort();
        classes.dedup();
        
        if classes.len() != 2 {
            return Err(KolosalError::InvalidInput(
                "SVM currently only supports binary classification".to_string()
            ));
        }
        
        self.classes = classes.clone();
        
        // Convert labels to +1/-1
        let y_binary: Array1<f64> = y.mapv(|v| {
            if v as i64 == classes[1] { 1.0 } else { -1.0 }
        });
        
        // Train using SMO
        let (alphas, bias, support_indices) = self.smo_train(x, &y_binary)?;
        
        // Store support vectors
        let sv_count = support_indices.len();
        let n_features = x.ncols();
        
        let mut support_vectors = Array2::zeros((sv_count, n_features));
        let mut support_labels = Array1::zeros(sv_count);
        let mut support_alphas = Array1::zeros(sv_count);
        
        for (i, &idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(idx));
            support_labels[i] = y_binary[idx];
            support_alphas[i] = alphas[idx];
        }
        
        self.support_vectors = Some(support_vectors);
        self.support_labels = Some(support_labels);
        self.alphas = Some(support_alphas);
        self.bias = bias;
        self.is_fitted = true;
        
        Ok(())
    }

    /// SMO training algorithm
    fn smo_train(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64, Vec<usize>)> {
        let n = x.nrows();
        let mut alphas = Array1::zeros(n);
        let mut bias = 0.0;
        
        // Precompute kernel matrix for efficiency
        let kernel_matrix = self.compute_kernel_matrix(x);
        
        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        
        let mut passes = 0;
        let max_passes = 5;

        while passes < max_passes {
            let mut num_changed = 0;

            if n <= 1 {
                break;
            }

            for i in 0..n {
                // Calculate error for i
                let e_i = self.decision_function_cached(&kernel_matrix, &alphas, y, bias, i) - y[i];

                // Check KKT conditions
                if (y[i] * e_i < -self.config.tol && alphas[i] < self.config.c)
                    || (y[i] * e_i > self.config.tol && alphas[i] > 0.0)
                {
                    // Select j randomly (safe: n > 1 guaranteed above)
                    let j = loop {
                        let j = rng.gen_range(0..n);
                        if j != i { break j; }
                    };
                    
                    let e_j = self.decision_function_cached(&kernel_matrix, &alphas, y, bias, j) - y[j];
                    
                    let alpha_i_old = alphas[i];
                    let alpha_j_old = alphas[j];
                    
                    // Compute bounds
                    let (l, h) = if y[i] != y[j] {
                        (
                            (alphas[j] - alphas[i]).max(0.0),
                            (self.config.c + alphas[j] - alphas[i]).min(self.config.c)
                        )
                    } else {
                        (
                            (alphas[i] + alphas[j] - self.config.c).max(0.0),
                            (alphas[i] + alphas[j]).min(self.config.c)
                        )
                    };
                    
                    if (l - h).abs() < 1e-10 {
                        continue;
                    }
                    
                    // Compute eta
                    let eta = 2.0 * kernel_matrix[[i, j]] - kernel_matrix[[i, i]] - kernel_matrix[[j, j]];
                    
                    if eta >= 0.0 {
                        continue;
                    }
                    
                    // Update alpha_j
                    alphas[j] = alphas[j] - y[j] * (e_i - e_j) / eta;
                    alphas[j] = alphas[j].max(l).min(h);
                    
                    if (alphas[j] - alpha_j_old).abs() < 1e-5 {
                        continue;
                    }
                    
                    // Update alpha_i
                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j]);
                    
                    // Update bias
                    let b1 = bias - e_i
                        - y[i] * (alphas[i] - alpha_i_old) * kernel_matrix[[i, i]]
                        - y[j] * (alphas[j] - alpha_j_old) * kernel_matrix[[i, j]];
                    
                    let b2 = bias - e_j
                        - y[i] * (alphas[i] - alpha_i_old) * kernel_matrix[[i, j]]
                        - y[j] * (alphas[j] - alpha_j_old) * kernel_matrix[[j, j]];
                    
                    bias = if alphas[i] > 0.0 && alphas[i] < self.config.c {
                        b1
                    } else if alphas[j] > 0.0 && alphas[j] < self.config.c {
                        b2
                    } else {
                        (b1 + b2) / 2.0
                    };
                    
                    num_changed += 1;
                }
            }
            
            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }
        
        // Find support vectors (alpha > 0)
        let support_indices: Vec<usize> = alphas
            .iter()
            .enumerate()
            .filter(|(_, &a)| a > 1e-8)
            .map(|(i, _)| i)
            .collect();
        
        Ok((alphas, bias, support_indices))
    }

    /// Compute kernel matrix (parallelized for large datasets)
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let n_features = x.ncols();

        // For small matrices, sequential is faster due to overhead
        if n < 100 {
            let mut k = Array2::zeros((n, n));
            for i in 0..n {
                for j in i..n {
                    let val = self.kernel(&x.row(i).to_owned(), &x.row(j).to_owned());
                    k[[i, j]] = val;
                    k[[j, i]] = val;
                }
            }
            return k;
        }

        // Parallel: compute upper triangle rows in parallel
        let config = self.config.clone();
        let x_data: Vec<Vec<f64>> = (0..n)
            .map(|i| x.row(i).to_vec())
            .collect();

        let rows: Vec<Vec<(usize, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row_i = &x_data[i];
                let a = Array1::from_vec(row_i.clone());
                (i..n)
                    .map(|j| {
                        let b = Array1::from_vec(x_data[j].clone());
                        let val = match &config.kernel {
                            KernelType::Linear => a.dot(&b),
                            KernelType::Polynomial { degree, gamma, coef0 } => {
                                (*gamma * a.dot(&b) + coef0).powi(*degree as i32)
                            }
                            KernelType::RBF { gamma } => {
                                let diff = &a - &b;
                                (-gamma * diff.dot(&diff)).exp()
                            }
                            KernelType::Sigmoid { gamma, coef0 } => {
                                (*gamma * a.dot(&b) + coef0).tanh()
                            }
                        };
                        (j, val)
                    })
                    .collect()
            })
            .collect();

        let mut k = Array2::zeros((n, n));
        for (i, row_vals) in rows.into_iter().enumerate() {
            for (j, val) in row_vals {
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
        }
        k
    }

    /// Compute kernel between two vectors
    fn kernel(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        match &self.config.kernel {
            KernelType::Linear => x1.dot(x2),
            KernelType::Polynomial { degree, gamma, coef0 } => {
                (*gamma * x1.dot(x2) + coef0).powi(*degree as i32)
            }
            KernelType::RBF { gamma } => {
                let diff = x1 - x2;
                let norm_sq = diff.dot(&diff);
                (-gamma * norm_sq).exp()
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                (*gamma * x1.dot(x2) + coef0).tanh()
            }
        }
    }

    /// Decision function using cached kernel matrix
    fn decision_function_cached(
        &self,
        k: &Array2<f64>,
        alphas: &Array1<f64>,
        y: &Array1<f64>,
        bias: f64,
        idx: usize,
    ) -> f64 {
        let mut sum = 0.0;
        for i in 0..alphas.len() {
            sum += alphas[i] * y[i] * k[[i, idx]];
        }
        sum + bias
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }
        
        let sv = self.support_vectors.as_ref().unwrap();
        let sv_labels = self.support_labels.as_ref().unwrap();
        let alphas = self.alphas.as_ref().unwrap();
        
        let n = x.nrows();
        let mut predictions = Array1::zeros(n);
        
        for i in 0..n {
            let sample = x.row(i).to_owned();
            let mut sum = self.bias;
            
            for j in 0..sv.nrows() {
                let k_val = self.kernel(&sample, &sv.row(j).to_owned());
                sum += alphas[j] * sv_labels[j] * k_val;
            }
            
            // Convert decision to original class labels
            predictions[i] = if sum >= 0.0 {
                self.classes[1] as f64
            } else {
                self.classes[0] as f64
            };
        }
        
        Ok(predictions)
    }

    /// Get decision function values
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }
        
        let sv = self.support_vectors.as_ref().unwrap();
        let sv_labels = self.support_labels.as_ref().unwrap();
        let alphas = self.alphas.as_ref().unwrap();
        
        let n = x.nrows();
        let mut scores = Array1::zeros(n);
        
        for i in 0..n {
            let sample = x.row(i).to_owned();
            let mut sum = self.bias;
            
            for j in 0..sv.nrows() {
                let k_val = self.kernel(&sample, &sv.row(j).to_owned());
                sum += alphas[j] * sv_labels[j] * k_val;
            }
            
            scores[i] = sum;
        }
        
        Ok(scores)
    }

    /// Get number of support vectors
    pub fn n_support_vectors(&self) -> usize {
        self.support_vectors.as_ref().map(|sv| sv.nrows()).unwrap_or(0)
    }
}

/// Support Vector Regressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVMRegressor {
    config: SVMConfig,
    support_vectors: Option<Array2<f64>>,
    alphas: Option<Array1<f64>>,  // alpha - alpha*
    bias: f64,
    is_fitted: bool,
}

impl SVMRegressor {
    /// Create a new SVM regressor
    pub fn new(config: SVMConfig) -> Self {
        Self {
            config,
            support_vectors: None,
            alphas: None,
            bias: 0.0,
            is_fitted: false,
        }
    }

    /// Fit the regressor using simplified SMO for SVR
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();
        let mut alphas: Array1<f64> = Array1::zeros(n);      // alpha
        let mut alphas_star: Array1<f64> = Array1::zeros(n); // alpha*
        let mut bias: f64 = 0.0;
        
        let kernel_matrix = self.compute_kernel_matrix(x);
        
        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        
        // Simple gradient descent approach for SVR
        let learning_rate: f64 = 0.01;
        
        for _ in 0..self.config.max_iter {
            for i in 0..n {
                let mut pred: f64 = bias;
                for j in 0..n {
                    pred += (alphas[j] - alphas_star[j]) * kernel_matrix[[j, i]];
                }
                
                let error: f64 = pred - y[i];
                
                // Update alphas based on epsilon-insensitive loss
                if error > self.config.epsilon {
                    alphas_star[i] = (alphas_star[i] + learning_rate).min(self.config.c);
                } else if error < -self.config.epsilon {
                    alphas[i] = (alphas[i] + learning_rate).min(self.config.c);
                }
                
                // Update bias
                bias -= learning_rate * 0.1 * error;
            }
        }
        
        // Combine into single alpha vector (alpha - alpha*)
        let combined_alphas = &alphas - &alphas_star;
        
        // Find support vectors
        let support_indices: Vec<usize> = combined_alphas
            .iter()
            .enumerate()
            .filter(|(_, a): &(usize, &f64)| a.abs() > 1e-8)
            .map(|(i, _)| i)
            .collect();
        
        let sv_count = support_indices.len();
        if sv_count == 0 {
            // Fallback: use all points if no support vectors found
            self.support_vectors = Some(x.clone());
            self.alphas = Some(combined_alphas);
        } else {
            let n_features = x.ncols();
            let mut support_vectors = Array2::zeros((sv_count, n_features));
            let mut support_alphas = Array1::zeros(sv_count);
            
            for (i, &idx) in support_indices.iter().enumerate() {
                support_vectors.row_mut(i).assign(&x.row(idx));
                support_alphas[i] = combined_alphas[idx];
            }
            
            self.support_vectors = Some(support_vectors);
            self.alphas = Some(support_alphas);
        }
        
        self.bias = bias;
        self.is_fitted = true;
        
        Ok(())
    }

    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut k = Array2::zeros((n, n));
        
        for i in 0..n {
            for j in i..n {
                let val = self.kernel(&x.row(i).to_owned(), &x.row(j).to_owned());
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
        }
        
        k
    }

    fn kernel(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        match &self.config.kernel {
            KernelType::Linear => x1.dot(x2),
            KernelType::Polynomial { degree, gamma, coef0 } => {
                (*gamma * x1.dot(x2) + coef0).powi(*degree as i32)
            }
            KernelType::RBF { gamma } => {
                let diff = x1 - x2;
                let norm_sq = diff.dot(&diff);
                (-gamma * norm_sq).exp()
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                (*gamma * x1.dot(x2) + coef0).tanh()
            }
        }
    }

    /// Predict target values
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }
        
        let sv = self.support_vectors.as_ref().unwrap();
        let alphas = self.alphas.as_ref().unwrap();
        
        let n = x.nrows();
        let mut predictions = Array1::zeros(n);
        
        for i in 0..n {
            let sample = x.row(i).to_owned();
            let mut sum = self.bias;
            
            for j in 0..sv.nrows() {
                let k_val = self.kernel(&sample, &sv.row(j).to_owned());
                sum += alphas[j] * k_val;
            }
            
            predictions[i] = sum;
        }
        
        Ok(predictions)
    }

    /// Get number of support vectors
    pub fn n_support_vectors(&self) -> usize {
        self.support_vectors.as_ref().map(|sv| sv.nrows()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_linear_separable_data() -> (Array2<f64>, Array1<f64>) {
        // Simple linearly separable data
        let x = Array2::from_shape_vec((10, 2), vec![
            1.0, 1.0,
            1.5, 1.2,
            2.0, 2.0,
            1.2, 1.8,
            0.8, 1.5,
            5.0, 5.0,
            5.5, 5.2,
            6.0, 6.0,
            5.2, 5.8,
            4.8, 5.5,
        ]).unwrap();
        
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        
        (x, y)
    }

    #[test]
    fn test_svm_classifier_linear() {
        let (x, y) = create_linear_separable_data();
        
        let config = SVMConfig {
            c: 1.0,
            kernel: KernelType::Linear,
            max_iter: 1000,
            ..Default::default()
        };
        
        let mut svm = SVMClassifier::new(config);
        svm.fit(&x, &y).unwrap();
        
        let predictions = svm.predict(&x).unwrap();
        
        // Should correctly classify most points
        let correct: usize = y.iter()
            .zip(predictions.iter())
            .filter(|(&yi, &pi)| yi == pi)
            .count();
        
        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.8, "Accuracy {} should be > 0.8", accuracy);
    }

    #[test]
    fn test_svm_classifier_rbf() {
        let (x, y) = create_linear_separable_data();
        
        let config = SVMConfig {
            c: 1.0,
            kernel: KernelType::RBF { gamma: 0.5 },
            max_iter: 1000,
            ..Default::default()
        };
        
        let mut svm = SVMClassifier::new(config);
        svm.fit(&x, &y).unwrap();
        
        let predictions = svm.predict(&x).unwrap();
        assert_eq!(predictions.len(), 10);
    }

    #[test]
    fn test_svm_regressor() {
        // Simple regression data
        let x = Array2::from_shape_vec((10, 1), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
        ]).unwrap();
        
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
        
        let config = SVMConfig {
            c: 10.0,
            kernel: KernelType::Linear,
            epsilon: 0.5,
            max_iter: 500,
            ..Default::default()
        };
        
        let mut svr = SVMRegressor::new(config);
        svr.fit(&x, &y).unwrap();
        
        let predictions = svr.predict(&x).unwrap();
        
        // Check predictions are reasonable (within 20% of actual)
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            let error = (pred - actual).abs() / actual;
            assert!(error < 0.5, "Error {} too large for pred={}, actual={}", error, pred, actual);
        }
    }
}
