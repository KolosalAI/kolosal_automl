//! Neural Network (Multi-Layer Perceptron) implementation
//!
//! A simple feedforward neural network with backpropagation.

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Activation function
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    /// Rectified Linear Unit
    ReLU,
    /// Sigmoid
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Linear (identity)
    Linear,
    /// Softmax (for output layer)
    Softmax,
}

impl Default for Activation {
    fn default() -> Self {
        Self::ReLU
    }
}

/// Neural Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPConfig {
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Activation function for hidden layers
    pub activation: Activation,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub max_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// L2 regularization
    pub alpha: f64,
    /// Random seed
    pub random_state: Option<u64>,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Validation split for early stopping
    pub validation_split: f64,
    /// Momentum
    pub momentum: f64,
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![100],
            activation: Activation::ReLU,
            learning_rate: 0.001,
            max_epochs: 200,
            batch_size: 32,
            alpha: 0.0001,
            random_state: Some(42),
            early_stopping_patience: 10,
            validation_split: 0.1,
            momentum: 0.9,
        }
    }
}

/// Multi-Layer Perceptron Regressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPRegressor {
    config: MLPConfig,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    n_features: usize,
    is_fitted: bool,
}

impl MLPRegressor {
    pub fn new(config: MLPConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            biases: Vec::new(),
            n_features: 0,
            is_fitted: false,
        }
    }

    /// Fit the model
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        self.n_features = x.ncols();
        
        // Initialize weights
        self.initialize_weights(1); // Single output for regression
        
        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        
        // Split for validation
        let val_size = (n_samples as f64 * self.config.validation_split) as usize;
        let train_size = n_samples - val_size;
        
        let x_train = x.slice(ndarray::s![..train_size, ..]).to_owned();
        let y_train = y.slice(ndarray::s![..train_size]).to_owned();
        let x_val = x.slice(ndarray::s![train_size.., ..]).to_owned();
        let y_val = y.slice(ndarray::s![train_size..]).to_owned();
        
        // Initialize velocity for momentum
        let mut velocities_w: Vec<Array2<f64>> = self.weights.iter()
            .map(|w| Array2::zeros(w.raw_dim()))
            .collect();
        let mut velocities_b: Vec<Array1<f64>> = self.biases.iter()
            .map(|b| Array1::zeros(b.len()))
            .collect();
        
        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;
        
        for _epoch in 0..self.config.max_epochs {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..train_size).collect();
            indices.shuffle(&mut rng);
            
            // Mini-batch training
            for batch_start in (0..train_size).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(train_size);
                let batch_indices = &indices[batch_start..batch_end];
                
                let x_batch = self.gather_rows(&x_train, batch_indices);
                let y_batch: Array1<f64> = batch_indices.iter().map(|&i| y_train[i]).collect();
                
                // Forward and backward pass
                let (activations, z_values) = self.forward(&x_batch);
                let gradients = self.backward(&x_batch, &y_batch, &activations, &z_values);
                
                // Update weights with momentum
                for (i, (grad_w, grad_b)) in gradients.into_iter().enumerate() {
                    velocities_w[i] = &velocities_w[i] * self.config.momentum
                        - &grad_w * self.config.learning_rate;
                    velocities_b[i] = &velocities_b[i] * self.config.momentum
                        - &grad_b * self.config.learning_rate;
                    
                    self.weights[i] = &self.weights[i] + &velocities_w[i];
                    self.biases[i] = &self.biases[i] + &velocities_b[i];
                    
                    // L2 regularization
                    self.weights[i] = &self.weights[i] * (1.0 - self.config.alpha * self.config.learning_rate);
                }
            }
            
            // Early stopping check
            if val_size > 0 {
                let val_pred = self.predict(&x_val);
                let val_loss = self.mse(&y_val, &val_pred);
                
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.config.early_stopping_patience {
                        break;
                    }
                }
            }
        }
        
        self.is_fitted = true;
        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let (activations, _) = self.forward(x);
        activations.last().unwrap().column(0).to_owned()
    }

    fn initialize_weights(&mut self, n_outputs: usize) {
        self.weights.clear();
        self.biases.clear();
        
        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        
        let mut layer_sizes = vec![self.n_features];
        layer_sizes.extend(&self.config.hidden_layers);
        layer_sizes.push(n_outputs);
        
        for i in 0..layer_sizes.len() - 1 {
            let n_in = layer_sizes[i];
            let n_out = layer_sizes[i + 1];
            
            // Xavier/Glorot initialization
            let scale = (2.0 / (n_in + n_out) as f64).sqrt();
            
            let weights: Vec<f64> = (0..n_in * n_out)
                .map(|_| rng.gen::<f64>() * 2.0 * scale - scale)
                .collect();
            
            self.weights.push(Array2::from_shape_vec((n_in, n_out), weights).unwrap());
            self.biases.push(Array1::zeros(n_out));
        }
    }

    fn forward(&self, x: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut activations = vec![x.clone()];
        let mut z_values = Vec::new();
        
        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let z = activations.last().unwrap().dot(w) + b;
            z_values.push(z.clone());
            
            let a = if i < self.weights.len() - 1 {
                self.activate(&z, self.config.activation)
            } else {
                z // Linear output for regression
            };
            
            activations.push(a);
        }
        
        (activations, z_values)
    }

    fn backward(
        &self,
        _x: &Array2<f64>,
        y: &Array1<f64>,
        activations: &[Array2<f64>],
        z_values: &[Array2<f64>],
    ) -> Vec<(Array2<f64>, Array1<f64>)> {
        let n = y.len() as f64;
        let mut gradients = Vec::new();
        
        // Output layer error (MSE gradient)
        let y_2d = y.clone().insert_axis(Axis(1));
        let output = activations.last().unwrap();
        let mut delta = (output - &y_2d) / n;
        
        // Backpropagate
        for i in (0..self.weights.len()).rev() {
            let a_prev = &activations[i];
            
            let grad_w = a_prev.t().dot(&delta);
            let grad_b = delta.sum_axis(Axis(0));
            
            gradients.push((grad_w, grad_b));
            
            if i > 0 {
                let z = &z_values[i - 1];
                delta = delta.dot(&self.weights[i].t()) * self.activate_derivative(z, self.config.activation);
            }
        }
        
        gradients.reverse();
        gradients
    }

    fn activate(&self, z: &Array2<f64>, activation: Activation) -> Array2<f64> {
        match activation {
            Activation::ReLU => z.mapv(|v| v.max(0.0)),
            Activation::Sigmoid => z.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => z.mapv(|v| v.tanh()),
            Activation::Linear => z.clone(),
            Activation::Softmax => {
                let mut result = z.clone();
                for mut row in result.rows_mut() {
                    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_sum: f64 = row.iter().map(|&v| (v - max).exp()).sum();
                    for v in row.iter_mut() {
                        *v = (*v - max).exp() / exp_sum;
                    }
                }
                result
            }
        }
    }

    fn activate_derivative(&self, z: &Array2<f64>, activation: Activation) -> Array2<f64> {
        match activation {
            Activation::ReLU => z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => {
                let sig = self.activate(z, Activation::Sigmoid);
                &sig * &(1.0 - &sig)
            }
            Activation::Tanh => {
                let t = z.mapv(|v| v.tanh());
                1.0 - &t * &t
            }
            Activation::Linear => Array2::ones(z.raw_dim()),
            Activation::Softmax => Array2::ones(z.raw_dim()), // Simplified
        }
    }

    fn gather_rows(&self, x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let n_cols = x.ncols();
        let mut rows = Vec::with_capacity(indices.len() * n_cols);
        for &i in indices {
            rows.extend(x.row(i).iter().copied());
        }
        Array2::from_shape_vec((indices.len(), n_cols), rows).unwrap()
    }

    fn mse(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        y_true.iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>() / y_true.len() as f64
    }
}

/// Multi-Layer Perceptron Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPClassifier {
    config: MLPConfig,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    n_features: usize,
    n_classes: usize,
    classes: Vec<i64>,
    is_fitted: bool,
}

impl MLPClassifier {
    pub fn new(config: MLPConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            biases: Vec::new(),
            n_features: 0,
            n_classes: 0,
            classes: Vec::new(),
            is_fitted: false,
        }
    }

    /// Fit the model
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        self.n_features = x.ncols();
        
        // Find unique classes
        let mut classes: Vec<i64> = y.iter().map(|&v| v as i64).collect();
        classes.sort();
        classes.dedup();
        self.classes = classes;
        self.n_classes = self.classes.len();
        
        // Initialize weights
        self.initialize_weights();
        
        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        
        // Convert y to one-hot encoding
        let y_onehot = self.to_onehot(y);
        
        // Split for validation
        let val_size = (n_samples as f64 * self.config.validation_split) as usize;
        let train_size = n_samples - val_size;
        
        let x_train = x.slice(ndarray::s![..train_size, ..]).to_owned();
        let y_train = y_onehot.slice(ndarray::s![..train_size, ..]).to_owned();
        
        // Initialize velocity for momentum
        let mut velocities_w: Vec<Array2<f64>> = self.weights.iter()
            .map(|w| Array2::zeros(w.raw_dim()))
            .collect();
        let mut velocities_b: Vec<Array1<f64>> = self.biases.iter()
            .map(|b| Array1::zeros(b.len()))
            .collect();
        
        for _epoch in 0..self.config.max_epochs {
            let mut indices: Vec<usize> = (0..train_size).collect();
            indices.shuffle(&mut rng);
            
            for batch_start in (0..train_size).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(train_size);
                let batch_indices = &indices[batch_start..batch_end];
                
                let x_batch = self.gather_rows(&x_train, batch_indices);
                let y_batch = self.gather_rows(&y_train, batch_indices);
                
                let (activations, z_values) = self.forward(&x_batch);
                let gradients = self.backward_classification(&y_batch, &activations, &z_values);
                
                for (i, (grad_w, grad_b)) in gradients.into_iter().enumerate() {
                    velocities_w[i] = &velocities_w[i] * self.config.momentum
                        - &grad_w * self.config.learning_rate;
                    velocities_b[i] = &velocities_b[i] * self.config.momentum
                        - &grad_b * self.config.learning_rate;
                    
                    self.weights[i] = &self.weights[i] + &velocities_w[i];
                    self.biases[i] = &self.biases[i] + &velocities_b[i];
                    
                    self.weights[i] = &self.weights[i] * (1.0 - self.config.alpha * self.config.learning_rate);
                }
            }
        }
        
        self.is_fitted = true;
        Ok(())
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let proba = self.predict_proba(x);
        
        proba.rows().into_iter()
            .map(|row| {
                let max_idx = row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.classes[max_idx] as f64
            })
            .collect()
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let (activations, _) = self.forward(x);
        activations.last().unwrap().clone()
    }

    fn initialize_weights(&mut self) {
        self.weights.clear();
        self.biases.clear();
        
        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        
        let mut layer_sizes = vec![self.n_features];
        layer_sizes.extend(&self.config.hidden_layers);
        layer_sizes.push(self.n_classes);
        
        for i in 0..layer_sizes.len() - 1 {
            let n_in = layer_sizes[i];
            let n_out = layer_sizes[i + 1];
            
            let scale = (2.0 / (n_in + n_out) as f64).sqrt();
            
            let weights: Vec<f64> = (0..n_in * n_out)
                .map(|_| rng.gen::<f64>() * 2.0 * scale - scale)
                .collect();
            
            self.weights.push(Array2::from_shape_vec((n_in, n_out), weights).unwrap());
            self.biases.push(Array1::zeros(n_out));
        }
    }

    fn forward(&self, x: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut activations = vec![x.clone()];
        let mut z_values = Vec::new();
        
        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let z = activations.last().unwrap().dot(w) + b;
            z_values.push(z.clone());
            
            let a = if i < self.weights.len() - 1 {
                self.activate(&z, self.config.activation)
            } else {
                self.activate(&z, Activation::Softmax) // Softmax for classification output
            };
            
            activations.push(a);
        }
        
        (activations, z_values)
    }

    fn backward_classification(
        &self,
        y_onehot: &Array2<f64>,
        activations: &[Array2<f64>],
        z_values: &[Array2<f64>],
    ) -> Vec<(Array2<f64>, Array1<f64>)> {
        let n = y_onehot.nrows() as f64;
        let mut gradients = Vec::new();
        
        // Cross-entropy gradient with softmax
        let mut delta = (activations.last().unwrap() - y_onehot) / n;
        
        for i in (0..self.weights.len()).rev() {
            let a_prev = &activations[i];
            
            let grad_w = a_prev.t().dot(&delta);
            let grad_b = delta.sum_axis(Axis(0));
            
            gradients.push((grad_w, grad_b));
            
            if i > 0 {
                let z = &z_values[i - 1];
                delta = delta.dot(&self.weights[i].t()) * self.activate_derivative(z, self.config.activation);
            }
        }
        
        gradients.reverse();
        gradients
    }

    fn activate(&self, z: &Array2<f64>, activation: Activation) -> Array2<f64> {
        match activation {
            Activation::ReLU => z.mapv(|v| v.max(0.0)),
            Activation::Sigmoid => z.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => z.mapv(|v| v.tanh()),
            Activation::Linear => z.clone(),
            Activation::Softmax => {
                let mut result = z.clone();
                for mut row in result.rows_mut() {
                    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_sum: f64 = row.iter().map(|&v| (v - max).exp()).sum();
                    for v in row.iter_mut() {
                        *v = (*v - max).exp() / exp_sum;
                    }
                }
                result
            }
        }
    }

    fn activate_derivative(&self, z: &Array2<f64>, activation: Activation) -> Array2<f64> {
        match activation {
            Activation::ReLU => z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => {
                let sig = self.activate(z, Activation::Sigmoid);
                &sig * &(1.0 - &sig)
            }
            Activation::Tanh => {
                let t = z.mapv(|v| v.tanh());
                1.0 - &t * &t
            }
            _ => Array2::ones(z.raw_dim()),
        }
    }

    fn to_onehot(&self, y: &Array1<f64>) -> Array2<f64> {
        let n = y.len();
        let mut onehot = Array2::zeros((n, self.n_classes));
        
        for (i, &label) in y.iter().enumerate() {
            let class_idx = self.classes.iter().position(|&c| c == label as i64).unwrap_or(0);
            onehot[[i, class_idx]] = 1.0;
        }
        
        onehot
    }

    fn gather_rows(&self, x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let n_cols = x.ncols();
        let mut rows = Vec::with_capacity(indices.len() * n_cols);
        for &i in indices {
            rows.extend(x.row(i).iter().copied());
        }
        Array2::from_shape_vec((indices.len(), n_cols), rows).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((100, 2), 
            (0..200).map(|i| (i as f64) * 0.05).collect()
        ).unwrap();
        
        let y: Array1<f64> = x.rows().into_iter()
            .map(|row| row[0] * 2.0 + row[1] + 0.5)
            .collect();
        
        (x, y)
    }

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((100, 2), 
            (0..200).map(|i| (i as f64) * 0.05).collect()
        ).unwrap();
        
        let y: Array1<f64> = x.rows().into_iter()
            .map(|row| if row[0] + row[1] > 5.0 { 1.0 } else { 0.0 })
            .collect();
        
        (x, y)
    }

    #[test]
    fn test_mlp_regressor() {
        let (x, y) = create_regression_data();
        
        let config = MLPConfig {
            hidden_layers: vec![32, 16],
            max_epochs: 500,  // More epochs for better convergence
            learning_rate: 0.001,  // Smaller learning rate
            ..Default::default()
        };
        
        let mut mlp = MLPRegressor::new(config);
        mlp.fit(&x, &y).unwrap();
        
        let predictions = mlp.predict(&x);
        assert_eq!(predictions.len(), 100);
        
        // Check that predictions are reasonable
        let mse: f64 = y.iter()
            .zip(predictions.iter())
            .map(|(yi, pi)| (yi - pi).powi(2))
            .sum::<f64>() / y.len() as f64;
        
        let y_var = y.var(0.0);
        assert!(mse < y_var, "MSE ({}) should be less than variance ({})", mse, y_var);
    }

    #[test]
    fn test_mlp_classifier() {
        let (x, y) = create_classification_data();
        
        let config = MLPConfig {
            hidden_layers: vec![32, 16],
            max_epochs: 100,
            learning_rate: 0.01,
            ..Default::default()
        };
        
        let mut mlp = MLPClassifier::new(config);
        mlp.fit(&x, &y).unwrap();
        
        let predictions = mlp.predict(&x);
        assert_eq!(predictions.len(), 100);
        
        // Check accuracy
        let correct: usize = y.iter()
            .zip(predictions.iter())
            .filter(|(&yi, &pi)| (yi - pi).abs() < 0.5)
            .count();
        
        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.7, "Accuracy ({}) should be above 70%", accuracy);
    }

    #[test]
    fn test_activation_functions() {
        let mlp = MLPRegressor::new(MLPConfig::default());
        let z = Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, -2.0, 0.5, 2.0]).unwrap();
        
        // Test ReLU
        let relu = mlp.activate(&z, Activation::ReLU);
        assert_eq!(relu[[0, 0]], 0.0);
        assert_eq!(relu[[0, 2]], 1.0);
        
        // Test Sigmoid
        let sigmoid = mlp.activate(&z, Activation::Sigmoid);
        assert!((sigmoid[[0, 1]] - 0.5).abs() < 0.001); // sigmoid(0) = 0.5
    }
}
