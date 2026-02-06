//! FT-Transformer Architecture
//!
//! Feature Tokenizer + Transformer for tabular data
//! Based on the paper: "Revisiting Deep Learning Models for Tabular Data"

use std::collections::HashMap;

/// FT-Transformer configuration
#[derive(Debug, Clone)]
pub struct FTTransformerConfig {
    /// Input dimension (number of features)
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Embedding dimension for each feature
    pub d_token: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Feedforward dimension multiplier
    pub ffn_d_hidden_multiplier: f64,
    /// Attention dropout rate
    pub attention_dropout: f64,
    /// FFN dropout rate
    pub ffn_dropout: f64,
    /// Residual dropout rate
    pub residual_dropout: f64,
}

impl Default for FTTransformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 0,
            output_dim: 0,
            d_token: 192,
            n_layers: 3,
            n_heads: 8,
            ffn_d_hidden_multiplier: 4.0 / 3.0,
            attention_dropout: 0.2,
            ffn_dropout: 0.1,
            residual_dropout: 0.0,
        }
    }
}

/// Feature Tokenizer for converting features to embeddings
#[derive(Debug, Clone)]
pub struct FeatureTokenizer {
    /// Number of numerical features
    pub n_numerical: usize,
    /// Number of categorical features
    pub n_categorical: usize,
    /// Embedding dimension
    pub d_token: usize,
    /// Numerical feature embeddings (linear projection weights)
    numerical_weights: Vec<Vec<f64>>,
    /// Categorical feature embeddings
    categorical_embeddings: HashMap<usize, Vec<Vec<f64>>>,
}

impl FeatureTokenizer {
    /// Create a new feature tokenizer
    pub fn new(n_numerical: usize, n_categorical: usize, d_token: usize) -> Self {
        // Initialize numerical weights randomly
        let numerical_weights: Vec<Vec<f64>> = (0..n_numerical)
            .map(|_| (0..d_token).map(|_| 0.1).collect())
            .collect();
        
        Self {
            n_numerical,
            n_categorical,
            d_token,
            numerical_weights,
            categorical_embeddings: HashMap::new(),
        }
    }
    
    /// Tokenize numerical features
    pub fn tokenize_numerical(&self, x: &[f64]) -> Vec<Vec<f64>> {
        x.iter().enumerate().map(|(i, &val)| {
            if i < self.numerical_weights.len() {
                self.numerical_weights[i].iter().map(|&w| w * val).collect()
            } else {
                vec![0.0; self.d_token]
            }
        }).collect()
    }
    
    /// Tokenize a single sample
    pub fn tokenize(&self, numerical: &[f64], _categorical: &[usize]) -> Vec<Vec<f64>> {
        let mut tokens = self.tokenize_numerical(numerical);
        
        // Add CLS token
        let cls_token = vec![0.0; self.d_token];
        tokens.insert(0, cls_token);
        
        tokens
    }
}

/// FT-Transformer model
#[derive(Debug)]
pub struct FTTransformer {
    config: FTTransformerConfig,
    tokenizer: FeatureTokenizer,
    is_fitted: bool,
}

impl FTTransformer {
    /// Create a new FT-Transformer
    pub fn new(config: FTTransformerConfig) -> Self {
        let tokenizer = FeatureTokenizer::new(config.input_dim, 0, config.d_token);
        
        Self {
            config,
            tokenizer,
            is_fitted: false,
        }
    }
    
    /// Fit the model
    pub fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64]) {
        // Placeholder - actual implementation would train the model
        self.is_fitted = true;
    }
    
    /// Predict
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        if !self.is_fitted {
            return vec![0.0; x.len()];
        }
        
        x.iter().map(|row| {
            let tokens = self.tokenizer.tokenize(row, &[]);
            // Use CLS token for prediction (simplified)
            tokens[0].iter().sum::<f64>()
        }).collect()
    }
    
    /// Get the tokenizer
    pub fn tokenizer(&self) -> &FeatureTokenizer {
        &self.tokenizer
    }
    
    /// Check if fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ft_transformer_basic() {
        let config = FTTransformerConfig {
            input_dim: 10,
            output_dim: 1,
            ..Default::default()
        };
        
        let mut model = FTTransformer::new(config);
        
        let x = vec![vec![1.0; 10]; 100];
        let y = vec![1.0; 100];
        
        model.fit(&x, &y);
        
        assert!(model.is_fitted());
        
        let predictions = model.predict(&x);
        assert_eq!(predictions.len(), 100);
    }
    
    #[test]
    fn test_feature_tokenizer() {
        let tokenizer = FeatureTokenizer::new(5, 0, 32);
        
        let numerical = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tokens = tokenizer.tokenize(&numerical, &[]);
        
        // Should have 6 tokens: 1 CLS + 5 features
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].len(), 32);
    }
}
