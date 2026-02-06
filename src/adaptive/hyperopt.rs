use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Configuration for a single parameter in the search space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpaceConfig {
    pub name: String,
    /// "float", "int", or "categorical"
    pub param_type: String,
    pub low: Option<f64>,
    pub high: Option<f64>,
    pub choices: Option<Vec<f64>>,
    pub log_scale: bool,
}

/// Tracks optimization performance across trials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTracker {
    pub window_size: usize,
    pub scores: Vec<f64>,
    pub params: Vec<HashMap<String, f64>>,
}

impl PerformanceTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            scores: Vec::new(),
            params: Vec::new(),
        }
    }

    /// Record a trial result.
    pub fn add_result(&mut self, score: f64, params: HashMap<String, f64>) {
        self.scores.push(score);
        self.params.push(params);
    }

    /// Check whether optimization has converged (std of recent scores below threshold).
    pub fn is_converged(&self) -> bool {
        if self.scores.len() < self.window_size {
            return false;
        }
        let window = &self.scores[self.scores.len() - self.window_size..];
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let variance = window.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / window.len() as f64;
        variance.sqrt() < 1e-4
    }

    /// Identify promising parameter ranges from the top-performing trials.
    pub fn get_promising_regions(&self) -> HashMap<String, (f64, f64)> {
        let mut regions = HashMap::new();
        if self.scores.is_empty() {
            return regions;
        }

        // Select top 20% of trials
        let mut indexed: Vec<(usize, f64)> =
            self.scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_n = (indexed.len() as f64 * 0.2).ceil() as usize;
        let top_n = top_n.max(1);
        let top_indices: Vec<usize> = indexed[..top_n].iter().map(|(i, _)| *i).collect();

        // Collect all parameter names from top trials
        let mut param_names: Vec<String> = Vec::new();
        for &idx in &top_indices {
            for key in self.params[idx].keys() {
                if !param_names.contains(key) {
                    param_names.push(key.clone());
                }
            }
        }

        for name in &param_names {
            let values: Vec<f64> = top_indices
                .iter()
                .filter_map(|&idx| self.params[idx].get(name).copied())
                .collect();
            if let (Some(&min), Some(&max)) = (
                values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
                values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
            ) {
                // Add 10% margin
                let margin = (max - min) * 0.1;
                regions.insert(name.clone(), (min - margin, max + margin));
            }
        }

        regions
    }
}

/// Adaptive search space that narrows over time based on trial results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSearchSpace {
    pub initial_space: HashMap<String, SearchSpaceConfig>,
    pub current_space: HashMap<String, SearchSpaceConfig>,
}

impl AdaptiveSearchSpace {
    pub fn new(space: HashMap<String, SearchSpaceConfig>) -> Self {
        Self {
            initial_space: space.clone(),
            current_space: space,
        }
    }

    /// Narrow the search space based on promising regions. Returns true if adapted.
    pub fn adapt_search_space(&mut self, tracker: &PerformanceTracker) -> bool {
        let regions = tracker.get_promising_regions();
        if regions.is_empty() {
            return false;
        }

        let mut adapted = false;
        for (name, config) in self.current_space.iter_mut() {
            if config.param_type == "categorical" {
                continue;
            }
            if let Some(&(new_low, new_high)) = regions.get(name) {
                if let (Some(cur_low), Some(cur_high)) = (config.low, config.high) {
                    let clamped_low = new_low.max(cur_low);
                    let clamped_high = new_high.min(cur_high);
                    if clamped_low < clamped_high {
                        config.low = Some(clamped_low);
                        config.high = Some(clamped_high);
                        adapted = true;
                    }
                }
            }
        }
        adapted
    }

    /// Reset the search space to its initial configuration.
    pub fn reset(&mut self) {
        self.current_space = self.initial_space.clone();
    }
}

/// Configuration for the adaptive hyperparameter optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveHyperoptConfig {
    pub n_trials: usize,
    pub timeout_secs: f64,
    pub n_startup: usize,
    pub adaptation_interval: usize,
}

impl Default for AdaptiveHyperoptConfig {
    fn default() -> Self {
        Self {
            n_trials: 100,
            timeout_secs: 300.0,
            n_startup: 10,
            adaptation_interval: 20,
        }
    }
}

/// Result of an optimization run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub best_params: HashMap<String, f64>,
    pub best_score: f64,
    pub n_trials: usize,
    pub duration_secs: f64,
    pub converged: bool,
}

/// Adaptive hyperparameter optimizer that combines random search with search-space narrowing.
#[derive(Debug, Clone)]
pub struct AdaptiveHyperparameterOptimizer {
    pub config: AdaptiveHyperoptConfig,
    pub tracker: PerformanceTracker,
    pub search_space: AdaptiveSearchSpace,
}

impl AdaptiveHyperparameterOptimizer {
    pub fn new(config: AdaptiveHyperoptConfig) -> Self {
        Self {
            config,
            tracker: PerformanceTracker::new(20),
            search_space: AdaptiveSearchSpace::new(HashMap::new()),
        }
    }

    /// Run optimization with the given objective function and search space.
    pub fn optimize<F>(
        &mut self,
        objective: F,
        search_space: HashMap<String, SearchSpaceConfig>,
    ) -> OptimizationResult
    where
        F: Fn(&HashMap<String, f64>) -> f64,
    {
        self.search_space = AdaptiveSearchSpace::new(search_space);
        self.tracker = PerformanceTracker::new(20);

        let start = Instant::now();
        let mut rng = rand::thread_rng();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = HashMap::new();
        let mut trials_run = 0;

        for trial in 0..self.config.n_trials {
            if start.elapsed().as_secs_f64() > self.config.timeout_secs {
                break;
            }

            let params = self.sample_params(&mut rng);
            let score = objective(&params);

            self.tracker.add_result(score, params.clone());
            trials_run = trial + 1;

            if score > best_score {
                best_score = score;
                best_params = params;
            }

            // Adapt search space periodically after startup phase
            if trial >= self.config.n_startup
                && trial % self.config.adaptation_interval == 0
            {
                self.search_space.adapt_search_space(&self.tracker);
            }

            if self.tracker.is_converged() {
                break;
            }
        }

        OptimizationResult {
            best_params,
            best_score,
            n_trials: trials_run,
            duration_secs: start.elapsed().as_secs_f64(),
            converged: self.tracker.is_converged(),
        }
    }

    /// Get the full optimization history as (score, params) pairs.
    pub fn get_optimization_history(&self) -> Vec<(f64, HashMap<String, f64>)> {
        self.tracker
            .scores
            .iter()
            .copied()
            .zip(self.tracker.params.iter().cloned())
            .collect()
    }

    /// Clear cached trial history.
    pub fn clear_cache(&mut self) {
        self.tracker = PerformanceTracker::new(self.tracker.window_size);
        self.search_space.reset();
    }

    fn sample_params(&self, rng: &mut impl Rng) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        for (name, config) in &self.search_space.current_space {
            let value = match config.param_type.as_str() {
                "categorical" => {
                    if let Some(ref choices) = config.choices {
                        if choices.is_empty() {
                            0.0
                        } else {
                            let idx = rng.gen_range(0..choices.len());
                            choices[idx]
                        }
                    } else {
                        0.0
                    }
                }
                "int" => {
                    let low = config.low.unwrap_or(0.0) as i64;
                    let high = config.high.unwrap_or(1.0) as i64;
                    if low >= high {
                        low as f64
                    } else {
                        rng.gen_range(low..=high) as f64
                    }
                }
                _ => {
                    // float
                    let low = config.low.unwrap_or(0.0);
                    let high = config.high.unwrap_or(1.0);
                    if config.log_scale && low > 0.0 && high > 0.0 {
                        let log_low = low.ln();
                        let log_high = high.ln();
                        (rng.gen_range(log_low..=log_high)).exp()
                    } else if low < high {
                        rng.gen_range(low..=high)
                    } else {
                        low
                    }
                }
            };
            params.insert(name.clone(), value);
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_space() -> HashMap<String, SearchSpaceConfig> {
        let mut space = HashMap::new();
        space.insert(
            "learning_rate".into(),
            SearchSpaceConfig {
                name: "learning_rate".into(),
                param_type: "float".into(),
                low: Some(0.001),
                high: Some(1.0),
                choices: None,
                log_scale: true,
            },
        );
        space.insert(
            "n_estimators".into(),
            SearchSpaceConfig {
                name: "n_estimators".into(),
                param_type: "int".into(),
                low: Some(10.0),
                high: Some(500.0),
                choices: None,
                log_scale: false,
            },
        );
        space
    }

    #[test]
    fn test_performance_tracker_add_and_converge() {
        let mut tracker = PerformanceTracker::new(5);
        for _ in 0..10 {
            tracker.add_result(0.9, HashMap::new());
        }
        assert!(tracker.is_converged());
    }

    #[test]
    fn test_performance_tracker_not_converged() {
        let mut tracker = PerformanceTracker::new(5);
        for i in 0..10 {
            tracker.add_result(i as f64, HashMap::new());
        }
        assert!(!tracker.is_converged());
    }

    #[test]
    fn test_promising_regions() {
        let mut tracker = PerformanceTracker::new(5);
        for i in 0..10 {
            let mut params = HashMap::new();
            params.insert("lr".into(), 0.01 * (i as f64 + 1.0));
            tracker.add_result(i as f64, params);
        }
        let regions = tracker.get_promising_regions();
        assert!(regions.contains_key("lr"));
    }

    #[test]
    fn test_adaptive_search_space_adapt() {
        let space = make_space();
        let mut adaptive = AdaptiveSearchSpace::new(space);

        let mut tracker = PerformanceTracker::new(5);
        for i in 0..20 {
            let mut params = HashMap::new();
            params.insert("learning_rate".into(), 0.01 + 0.001 * i as f64);
            params.insert("n_estimators".into(), 100.0 + i as f64);
            tracker.add_result(0.8 + 0.01 * i as f64, params);
        }

        let adapted = adaptive.adapt_search_space(&tracker);
        assert!(adapted);
    }

    #[test]
    fn test_adaptive_search_space_reset() {
        let space = make_space();
        let mut adaptive = AdaptiveSearchSpace::new(space.clone());
        let tracker = PerformanceTracker::new(5);
        adaptive.adapt_search_space(&tracker);
        adaptive.reset();
        for (name, config) in &adaptive.current_space {
            assert_eq!(config.low, space[name].low);
            assert_eq!(config.high, space[name].high);
        }
    }

    #[test]
    fn test_optimizer_runs() {
        let space = make_space();
        let config = AdaptiveHyperoptConfig {
            n_trials: 30,
            timeout_secs: 10.0,
            n_startup: 5,
            adaptation_interval: 10,
        };
        let mut optimizer = AdaptiveHyperparameterOptimizer::new(config);

        let result = optimizer.optimize(
            |params| {
                let lr = params.get("learning_rate").copied().unwrap_or(0.1);
                // Simple parabolic objective: peak at lr=0.1
                -(lr - 0.1_f64).powi(2)
            },
            space,
        );

        assert!(result.n_trials > 0);
        assert!(result.best_params.contains_key("learning_rate"));
    }

    #[test]
    fn test_optimizer_history_and_clear() {
        let space = make_space();
        let config = AdaptiveHyperoptConfig {
            n_trials: 10,
            timeout_secs: 5.0,
            n_startup: 3,
            adaptation_interval: 5,
        };
        let mut optimizer = AdaptiveHyperparameterOptimizer::new(config);
        optimizer.optimize(|_| 0.5, space);

        let history = optimizer.get_optimization_history();
        assert!(!history.is_empty());

        optimizer.clear_cache();
        assert!(optimizer.get_optimization_history().is_empty());
    }
}
