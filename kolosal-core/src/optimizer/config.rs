//! Optimization configuration

use serde::{Deserialize, Serialize};
use super::SamplerType;

/// Direction of optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizeDirection {
    Minimize,
    Maximize,
}

/// Configuration for hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Number of trials to run
    pub n_trials: usize,
    
    /// Maximum time in seconds
    pub timeout_secs: Option<f64>,
    
    /// Optimization direction
    pub direction: OptimizeDirection,
    
    /// Sampler type
    pub sampler: SamplerType,
    
    /// Number of initial random samples before optimization
    pub n_startup_trials: usize,
    
    /// Number of parallel workers
    pub n_jobs: usize,
    
    /// Random seed
    pub random_state: Option<u64>,
    
    /// Whether to prune unpromising trials
    pub pruning: bool,
    
    /// Patience for early stopping
    pub early_stopping_patience: Option<usize>,
    
    /// Minimum improvement to consider
    pub min_improvement: f64,
    
    /// Whether to show progress
    pub verbose: bool,
    
    /// Cross-validation folds for evaluation
    pub cv_folds: usize,
    
    /// Metric to optimize
    pub metric: String,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            n_trials: 100,
            timeout_secs: None,
            direction: OptimizeDirection::Minimize,
            sampler: SamplerType::TPE,
            n_startup_trials: 10,
            n_jobs: 1,
            random_state: Some(42),
            pruning: true,
            early_stopping_patience: Some(20),
            min_improvement: 1e-6,
            verbose: true,
            cv_folds: 5,
            metric: "auto".to_string(),
        }
    }
}

impl OptimizationConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to set number of trials
    pub fn with_n_trials(mut self, n: usize) -> Self {
        self.n_trials = n;
        self
    }

    /// Builder method to set timeout
    pub fn with_timeout(mut self, secs: f64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Builder method to set direction
    pub fn with_direction(mut self, direction: OptimizeDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Builder method to set sampler
    pub fn with_sampler(mut self, sampler: SamplerType) -> Self {
        self.sampler = sampler;
        self
    }

    /// Builder method to enable parallel execution
    pub fn with_n_jobs(mut self, n: usize) -> Self {
        self.n_jobs = n;
        self
    }

    /// Builder method to set metric
    pub fn with_metric(mut self, metric: impl Into<String>) -> Self {
        self.metric = metric.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OptimizationConfig::default();
        assert_eq!(config.n_trials, 100);
        assert!(matches!(config.sampler, SamplerType::TPE));
    }

    #[test]
    fn test_builder() {
        let config = OptimizationConfig::new()
            .with_n_trials(50)
            .with_sampler(SamplerType::Random)
            .with_direction(OptimizeDirection::Maximize);
        
        assert_eq!(config.n_trials, 50);
        assert!(matches!(config.sampler, SamplerType::Random));
        assert!(matches!(config.direction, OptimizeDirection::Maximize));
    }
}
