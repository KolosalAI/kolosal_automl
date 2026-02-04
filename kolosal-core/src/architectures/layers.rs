//! Common neural network layers for tabular architectures

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Ghost Batch Normalization
/// 
/// Applies batch normalization over virtual mini-batches for improved
/// regularization with large batch sizes.
#[derive(Debug, Clone)]
pub struct GhostBatchNorm {
    /// Number of features
    num_features: usize,
    /// Virtual batch size
    virtual_batch_size: usize,
    /// Momentum for running stats
    momentum: f64,
    /// Epsilon for numerical stability
    eps: f64,
    /// Running mean
    running_mean: Array1<f64>,
    /// Running variance
    running_var: Array1<f64>,
    /// Learnable scale (gamma)
    gamma: Array1<f64>,
    /// Learnable shift (beta)
    beta: Array1<f64>,
    /// Whether in training mode
    training: bool,
}

impl GhostBatchNorm {
    /// Create new GhostBatchNorm
    pub fn new(num_features: usize, virtual_batch_size: usize) -> Self {
        Self {
            num_features,
            virtual_batch_size,
            momentum: 0.01,
            eps: 1e-5,
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            training: true,
        }
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Forward pass
    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let batch_size = x.nrows();
        
        if self.training && batch_size > self.virtual_batch_size {
            // Process in virtual batches
            let num_chunks = (batch_size + self.virtual_batch_size - 1) / self.virtual_batch_size;
            let mut outputs = Vec::new();
            
            for i in 0..num_chunks {
                let start = i * self.virtual_batch_size;
                let end = (start + self.virtual_batch_size).min(batch_size);
                let chunk = x.slice(ndarray::s![start..end, ..]).to_owned();
                let normalized = self.normalize_batch(&chunk);
                outputs.push(normalized);
            }
            
            // Concatenate results
            let views: Vec<_> = outputs.iter().map(|a| a.view()).collect();
            ndarray::concatenate(Axis(0), &views).unwrap()
        } else {
            self.normalize_batch(x)
        }
    }

    fn normalize_batch(&mut self, x: &Array2<f64>) -> Array2<f64> {
        if self.training {
            let mean = x.mean_axis(Axis(0)).unwrap();
            let var = x.var_axis(Axis(0), 0.0);
            
            // Update running stats
            self.running_mean = &self.running_mean * (1.0 - self.momentum) + &mean * self.momentum;
            self.running_var = &self.running_var * (1.0 - self.momentum) + &var * self.momentum;
            
            // Normalize
            let std = var.mapv(|v| (v + self.eps).sqrt());
            let normalized = (x - &mean) / &std;
            &normalized * &self.gamma + &self.beta
        } else {
            let std = self.running_var.mapv(|v| (v + self.eps).sqrt());
            let normalized = (x - &self.running_mean) / &std;
            &normalized * &self.gamma + &self.beta
        }
    }
}

/// Sparsemax activation function
/// 
/// Projects onto the probability simplex, producing sparse outputs.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sparsemax;

impl Sparsemax {
    /// Apply sparsemax to input
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let n = x.len();
        
        // Sort in descending order
        let mut sorted: Vec<f64> = x.iter().cloned().collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        // Find the threshold
        let mut cumsum = 0.0;
        let mut k = 0;
        for (i, &val) in sorted.iter().enumerate() {
            cumsum += val;
            if val > (cumsum - 1.0) / (i + 1) as f64 {
                k = i + 1;
            }
        }
        
        let tau = (sorted[..k].iter().sum::<f64>() - 1.0) / k as f64;
        
        // Apply threshold
        x.mapv(|xi| (xi - tau).max(0.0))
    }

    /// Apply sparsemax to each row of a 2D array
    pub fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.raw_dim());
        for (i, row) in x.rows().into_iter().enumerate() {
            let sparse_row = self.forward(&row.to_owned());
            result.row_mut(i).assign(&sparse_row);
        }
        result
    }
}

/// Gated Linear Unit Block
/// 
/// GLU(x) = x * sigmoid(gate(x))
#[derive(Debug, Clone)]
pub struct GLUBlock {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Linear weights
    fc_weights: Array2<f64>,
    /// Gate weights
    gate_weights: Array2<f64>,
    /// Linear bias
    fc_bias: Array1<f64>,
    /// Gate bias
    gate_bias: Array1<f64>,
    /// Batch normalization
    bn: Option<GhostBatchNorm>,
}

impl GLUBlock {
    /// Create new GLU block
    pub fn new(input_dim: usize, output_dim: usize, use_bn: bool, virtual_batch_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        
        let fc_weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            (rng.gen::<f64>() - 0.5) * scale
        });
        let gate_weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            (rng.gen::<f64>() - 0.5) * scale
        });
        
        let bn = if use_bn {
            Some(GhostBatchNorm::new(output_dim * 2, virtual_batch_size))
        } else {
            None
        };

        Self {
            input_dim,
            output_dim,
            fc_weights,
            gate_weights,
            fc_bias: Array1::zeros(output_dim),
            gate_bias: Array1::zeros(output_dim),
            bn,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let fc_out = x.dot(&self.fc_weights) + &self.fc_bias;
        let gate_out = x.dot(&self.gate_weights) + &self.gate_bias;
        
        // Apply batch norm if present
        let (fc_normed, gate_normed) = if let Some(ref mut bn) = self.bn {
            let combined = ndarray::concatenate(Axis(1), &[fc_out.view(), gate_out.view()]).unwrap();
            let normed = bn.forward(&combined);
            let mid = normed.ncols() / 2;
            (
                normed.slice(ndarray::s![.., ..mid]).to_owned(),
                normed.slice(ndarray::s![.., mid..]).to_owned(),
            )
        } else {
            (fc_out, gate_out)
        };
        
        // GLU: fc * sigmoid(gate)
        let sigmoid_gate = gate_normed.mapv(|g| 1.0 / (1.0 + (-g).exp()));
        fc_normed * sigmoid_gate
    }
}

/// Attention Transformer for feature selection
#[derive(Debug, Clone)]
pub struct AttentionTransformer {
    /// Input dimension
    input_dim: usize,
    /// Output dimension (attention weights)
    output_dim: usize,
    /// FC layer weights
    fc_weights: Array2<f64>,
    /// FC layer bias
    fc_bias: Array1<f64>,
    /// Batch normalization
    bn: GhostBatchNorm,
}

impl AttentionTransformer {
    /// Create new attention transformer
    pub fn new(input_dim: usize, output_dim: usize, virtual_batch_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        
        let fc_weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            (rng.gen::<f64>() - 0.5) * scale
        });

        Self {
            input_dim,
            output_dim,
            fc_weights,
            fc_bias: Array1::zeros(output_dim),
            bn: GhostBatchNorm::new(output_dim, virtual_batch_size),
        }
    }

    /// Forward pass - returns attention weights
    pub fn forward(&mut self, x: &Array2<f64>, priors: &Array2<f64>) -> Array2<f64> {
        let out = x.dot(&self.fc_weights) + &self.fc_bias;
        let out = self.bn.forward(&out);
        
        // Multiply by prior scales
        let scaled = &out * priors;
        
        // Apply sparsemax
        let sparsemax = Sparsemax;
        sparsemax.forward_batch(&scaled)
    }
}

/// Feature Attention Layer
#[derive(Debug, Clone)]
pub struct FeatureAttention {
    /// Number of features
    num_features: usize,
    /// Attention dimension
    attention_dim: usize,
    /// Query projection
    query_weights: Array2<f64>,
    /// Key projection
    key_weights: Array2<f64>,
    /// Value projection  
    value_weights: Array2<f64>,
    /// Output projection
    output_weights: Array2<f64>,
    /// Number of attention heads
    num_heads: usize,
}

impl FeatureAttention {
    /// Create new feature attention
    pub fn new(num_features: usize, attention_dim: usize, num_heads: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (num_features + attention_dim) as f64).sqrt();
        
        let query_weights = Array2::from_shape_fn((num_features, attention_dim), |_| {
            (rng.gen::<f64>() - 0.5) * scale
        });
        let key_weights = Array2::from_shape_fn((num_features, attention_dim), |_| {
            (rng.gen::<f64>() - 0.5) * scale
        });
        let value_weights = Array2::from_shape_fn((num_features, attention_dim), |_| {
            (rng.gen::<f64>() - 0.5) * scale
        });
        let output_weights = Array2::from_shape_fn((attention_dim, num_features), |_| {
            (rng.gen::<f64>() - 0.5) * scale
        });

        Self {
            num_features,
            attention_dim,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            num_heads,
        }
    }

    /// Forward pass with self-attention
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let batch_size = x.nrows();
        
        // Project to Q, K, V
        let q = x.dot(&self.query_weights);
        let k = x.dot(&self.key_weights);
        let v = x.dot(&self.value_weights);
        
        // Scaled dot-product attention
        let scale = (self.attention_dim as f64).sqrt();
        let scores = q.dot(&k.t()) / scale;
        
        // Softmax
        let attention_weights = softmax_2d(&scores);
        
        // Apply attention to values
        let attended = attention_weights.dot(&v);
        
        // Output projection
        attended.dot(&self.output_weights)
    }
}

/// Softmax over rows of 2D array
fn softmax_2d(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    for mut row in result.rows_mut() {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Array1<f64> = row.mapv(|v| (v - max_val).exp());
        let sum: f64 = exp.sum();
        row.assign(&(exp / sum));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ghost_batch_norm() {
        let mut gbn = GhostBatchNorm::new(10, 16);
        let x = Array2::from_shape_fn((32, 10), |_| rand::random::<f64>());
        
        let out = gbn.forward(&x);
        assert_eq!(out.shape(), &[32, 10]);
    }

    #[test]
    fn test_sparsemax() {
        let sparsemax = Sparsemax;
        let x = Array1::from_vec(vec![2.0, 1.0, 0.1, -1.0]);
        
        let out = sparsemax.forward(&x);
        
        // Output should sum to 1
        assert!((out.sum() - 1.0).abs() < 1e-6);
        // Should be sparse (some zeros)
        assert!(out.iter().any(|&v| v == 0.0));
    }

    #[test]
    fn test_glu_block() {
        let mut glu = GLUBlock::new(10, 8, true, 16);
        let x = Array2::from_shape_fn((32, 10), |_| rand::random::<f64>());
        
        let out = glu.forward(&x);
        assert_eq!(out.shape(), &[32, 8]);
    }

    #[test]
    fn test_feature_attention() {
        let attention = FeatureAttention::new(10, 32, 4);
        let x = Array2::from_shape_fn((16, 10), |_| rand::random::<f64>());
        
        let out = attention.forward(&x);
        assert_eq!(out.shape(), &[16, 10]);
    }

    #[test]
    fn test_attention_transformer() {
        let mut attn = AttentionTransformer::new(32, 10, 16);
        let x = Array2::from_shape_fn((16, 32), |_| rand::random::<f64>());
        let priors = Array2::ones((16, 10));
        
        let out = attn.forward(&x, &priors);
        assert_eq!(out.shape(), &[16, 10]);
        
        // Each row should sum to ~1 (sparsemax output)
        for row in out.rows() {
            assert!((row.sum() - 1.0).abs() < 0.1);
        }
    }
}
