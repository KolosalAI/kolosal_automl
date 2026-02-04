//! Sampling strategies for hyperparameter optimization

use crate::error::Result;
use super::search_space::{SearchSpace, TrialParams, ParameterValue};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of sampler to use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SamplerType {
    /// Random sampling
    Random,
    /// Tree-structured Parzen Estimator
    TPE,
    /// Bayesian optimization with Gaussian Process
    GaussianProcess,
    /// Grid search
    Grid,
    /// Evolutionary algorithm
    Evolutionary,
    /// CMA-ES
    CmaEs,
}

/// Sampler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplerConfig {
    /// Random sampling
    Random,
    /// TPE with parameters
    TPE { n_startup_trials: usize },
    /// Grid search
    Grid,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig::TPE { n_startup_trials: 10 }
    }
}

/// Trait for hyperparameter samplers
pub trait Sampler: Send + Sync {
    /// Sample the next set of hyperparameters
    fn sample(
        &mut self,
        search_space: &SearchSpace,
        history: &[(TrialParams, f64)],
    ) -> TrialParams;
}

/// Random sampler
#[derive(Debug)]
pub struct RandomSampler {
    rng: Xoshiro256PlusPlus,
}

impl RandomSampler {
    /// Create a new random sampler
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => Xoshiro256PlusPlus::seed_from_u64(s),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        Self { rng }
    }
}

impl Sampler for RandomSampler {
    fn sample(
        &mut self,
        search_space: &SearchSpace,
        _history: &[(TrialParams, f64)],
    ) -> TrialParams {
        search_space.sample(&mut self.rng)
    }
}

/// Tree-structured Parzen Estimator sampler
#[derive(Debug)]
pub struct TPESampler {
    rng: Xoshiro256PlusPlus,
    n_startup_trials: usize,
    gamma: f64,
    n_candidates: usize,
}

impl TPESampler {
    /// Create a new TPE sampler
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => Xoshiro256PlusPlus::seed_from_u64(s),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        Self {
            rng,
            n_startup_trials: 10,
            gamma: 0.25,
            n_candidates: 24,
        }
    }

    /// Set number of startup trials
    pub fn with_n_startup(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Set gamma (quantile for splitting good/bad)
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }
}

impl Sampler for TPESampler {
    fn sample(
        &mut self,
        search_space: &SearchSpace,
        history: &[(TrialParams, f64)],
    ) -> TrialParams {
        // Use random sampling for startup trials
        if history.len() < self.n_startup_trials {
            return search_space.sample(&mut self.rng);
        }

        // Simplified TPE implementation
        // In production, would implement full TPE with KDE
        
        // Sort history by objective value
        let mut sorted: Vec<_> = history.iter().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Split into good and bad
        let n_good = ((sorted.len() as f64 * self.gamma).ceil() as usize).max(1);
        let good_trials: Vec<_> = sorted[..n_good].iter().map(|(p, _)| p).collect();

        // Sample candidates and pick the one most similar to good trials
        let mut best_params = search_space.sample(&mut self.rng);
        let mut best_score = f64::MIN;

        for _ in 0..self.n_candidates {
            let candidate = search_space.sample(&mut self.rng);
            
            // Score based on similarity to good trials (simplified)
            let score = self.compute_similarity(&candidate, &good_trials);
            
            if score > best_score {
                best_score = score;
                best_params = candidate;
            }
        }

        best_params
    }
}

impl TPESampler {
    fn compute_similarity(
        &self,
        candidate: &TrialParams,
        good_trials: &[&TrialParams],
    ) -> f64 {
        if good_trials.is_empty() {
            return 0.0;
        }

        // Simplified similarity: average inverse distance
        let mut total_sim = 0.0;

        for good in good_trials {
            let mut dist = 0.0;
            let mut count = 0;

            for (key, val) in candidate {
                if let Some(good_val) = good.get(key) {
                    let d = Self::param_distance(val, good_val);
                    dist += d * d;
                    count += 1;
                }
            }

            if count > 0 {
                dist = (dist / count as f64).sqrt();
                total_sim += 1.0 / (1.0 + dist);
            }
        }

        total_sim / good_trials.len() as f64
    }

    fn param_distance(a: &ParameterValue, b: &ParameterValue) -> f64 {
        match (a, b) {
            (ParameterValue::Float(va), ParameterValue::Float(vb)) => (va - vb).abs(),
            (ParameterValue::Int(va), ParameterValue::Int(vb)) => (va - vb).abs() as f64,
            (ParameterValue::String(va), ParameterValue::String(vb)) => {
                if va == vb { 0.0 } else { 1.0 }
            }
            (ParameterValue::Bool(va), ParameterValue::Bool(vb)) => {
                if va == vb { 0.0 } else { 1.0 }
            }
            _ => 1.0,
        }
    }
}

/// Create a sampler from type
pub fn create_sampler(sampler_type: SamplerType, seed: Option<u64>) -> Box<dyn Sampler> {
    match sampler_type {
        SamplerType::Random => Box::new(RandomSampler::new(seed)),
        SamplerType::TPE => Box::new(TPESampler::new(seed)),
        // Other samplers would be implemented similarly
        _ => Box::new(TPESampler::new(seed)), // Default to TPE
    }
}

/// Create a sampler from config
pub fn create_sampler_from_config(config: SamplerConfig, seed: Option<u64>) -> Box<dyn Sampler> {
    match config {
        SamplerConfig::Random => Box::new(RandomSampler::new(seed)),
        SamplerConfig::TPE { n_startup_trials } => {
            Box::new(TPESampler::new(seed).with_n_startup(n_startup_trials))
        }
        SamplerConfig::Grid => Box::new(RandomSampler::new(seed)), // Fallback for now
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_sampler() {
        let space = SearchSpace::new()
            .float("lr", 0.001, 0.1)
            .int("n", 10, 100);

        let mut sampler = RandomSampler::new(Some(42));
        let params = sampler.sample(&space, &[]);

        assert!(params.contains_key("lr"));
        assert!(params.contains_key("n"));
    }

    #[test]
    fn test_tpe_sampler_startup() {
        let space = SearchSpace::new()
            .float("lr", 0.001, 0.1);

        let mut sampler = TPESampler::new(Some(42));
        
        // During startup, should just do random sampling
        for _ in 0..5 {
            let params = sampler.sample(&space, &[]);
            assert!(params.contains_key("lr"));
        }
    }

    #[test]
    fn test_tpe_sampler_with_history() {
        let space = SearchSpace::new()
            .float("lr", 0.001, 0.1);

        let mut sampler = TPESampler::new(Some(42));
        
        // Create some history
        let history: Vec<(TrialParams, f64)> = (0..20)
            .map(|i| {
                let mut params = HashMap::new();
                params.insert("lr".to_string(), ParameterValue::Float(0.001 + i as f64 * 0.005));
                (params, i as f64 * 0.1)
            })
            .collect();

        let params = sampler.sample(&space, &history);
        assert!(params.contains_key("lr"));
    }
}
