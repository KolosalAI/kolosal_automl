//! Preprocessing configuration

use serde::{Deserialize, Serialize};
use super::{ScalerType, EncoderType, ImputeStrategy};

/// Configuration for data preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Strategy for handling missing numeric values
    pub numeric_impute_strategy: ImputeStrategy,
    
    /// Strategy for handling missing categorical values
    pub categorical_impute_strategy: ImputeStrategy,
    
    /// Type of scaler to use for numeric features
    pub scaler_type: ScalerType,
    
    /// Type of encoder to use for categorical features
    pub encoder_type: EncoderType,
    
    /// Maximum number of categories for one-hot encoding
    /// Categories beyond this will use label encoding
    pub max_onehot_categories: usize,
    
    /// Whether to detect and handle outliers
    pub handle_outliers: bool,
    
    /// Outlier threshold (number of standard deviations)
    pub outlier_threshold: f64,
    
    /// Whether to generate polynomial features
    pub polynomial_features: bool,
    
    /// Degree for polynomial features
    pub polynomial_degree: u32,
    
    /// Whether to generate interaction features
    pub interaction_features: bool,
    
    /// Number of threads for parallel processing
    pub n_jobs: Option<usize>,
    
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            numeric_impute_strategy: ImputeStrategy::Mean,
            categorical_impute_strategy: ImputeStrategy::MostFrequent,
            scaler_type: ScalerType::Standard,
            encoder_type: EncoderType::OneHot,
            max_onehot_categories: 10,
            handle_outliers: false,
            outlier_threshold: 3.0,
            polynomial_features: false,
            polynomial_degree: 2,
            interaction_features: false,
            n_jobs: None,
            random_state: None,
        }
    }
}

impl PreprocessingConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to set numeric impute strategy
    pub fn with_numeric_impute(mut self, strategy: ImputeStrategy) -> Self {
        self.numeric_impute_strategy = strategy;
        self
    }

    /// Builder method to set scaler type
    pub fn with_scaler(mut self, scaler_type: ScalerType) -> Self {
        self.scaler_type = scaler_type;
        self
    }

    /// Builder method to set encoder type
    pub fn with_encoder(mut self, encoder_type: EncoderType) -> Self {
        self.encoder_type = encoder_type;
        self
    }

    /// Builder method to enable outlier handling
    pub fn with_outlier_handling(mut self, threshold: f64) -> Self {
        self.handle_outliers = true;
        self.outlier_threshold = threshold;
        self
    }

    /// Builder method to set number of threads
    pub fn with_n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PreprocessingConfig::default();
        assert_eq!(config.max_onehot_categories, 10);
        assert!(!config.handle_outliers);
    }

    #[test]
    fn test_builder_pattern() {
        let config = PreprocessingConfig::new()
            .with_scaler(ScalerType::MinMax)
            .with_outlier_handling(2.5)
            .with_n_jobs(4);
        
        assert!(matches!(config.scaler_type, ScalerType::MinMax));
        assert!(config.handle_outliers);
        assert_eq!(config.outlier_threshold, 2.5);
        assert_eq!(config.n_jobs, Some(4));
    }
}
