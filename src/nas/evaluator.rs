//! Architecture Evaluator
//!
//! Provides utilities for evaluating neural architecture performance.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::search_space::NetworkArchitecture;
use crate::error::Result;

/// Evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Number of epochs for full evaluation
    pub epochs: usize,
    /// Number of epochs for quick evaluation (proxy)
    pub proxy_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Whether to use early stopping
    pub early_stopping: bool,
    /// Validation split ratio
    pub val_split: f64,
    /// Number of cross-validation folds (0 = no CV)
    pub cv_folds: usize,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            proxy_epochs: 10,
            batch_size: 32,
            patience: 10,
            early_stopping: true,
            val_split: 0.2,
            cv_folds: 0,
        }
    }
}

/// Result of architecture evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Architecture identifier/hash
    pub arch_id: u64,
    /// Training loss
    pub train_loss: f64,
    /// Validation loss
    pub val_loss: f64,
    /// Validation accuracy (or other metric)
    pub val_metric: f64,
    /// Test metric (if available)
    pub test_metric: Option<f64>,
    /// Number of parameters
    pub num_params: usize,
    /// Training time in seconds
    pub train_time: f64,
    /// Number of FLOPs (estimated)
    pub flops: Option<usize>,
    /// Additional metrics
    pub extra_metrics: HashMap<String, f64>,
}

impl EvaluationResult {
    /// Create new result
    pub fn new(arch_id: u64) -> Self {
        Self {
            arch_id,
            train_loss: 0.0,
            val_loss: 0.0,
            val_metric: 0.0,
            test_metric: None,
            num_params: 0,
            train_time: 0.0,
            flops: None,
            extra_metrics: HashMap::new(),
        }
    }

    /// Set losses
    pub fn with_losses(mut self, train: f64, val: f64) -> Self {
        self.train_loss = train;
        self.val_loss = val;
        self
    }

    /// Set validation metric
    pub fn with_val_metric(mut self, metric: f64) -> Self {
        self.val_metric = metric;
        self
    }

    /// Set test metric
    pub fn with_test_metric(mut self, metric: f64) -> Self {
        self.test_metric = Some(metric);
        self
    }

    /// Set number of parameters
    pub fn with_num_params(mut self, n: usize) -> Self {
        self.num_params = n;
        self
    }

    /// Set training time
    pub fn with_train_time(mut self, t: f64) -> Self {
        self.train_time = t;
        self
    }

    /// Add extra metric
    pub fn add_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.extra_metrics.insert(name.into(), value);
        self
    }
}

/// Architecture evaluator
pub struct ArchitectureEvaluator {
    /// Configuration
    config: EvaluationConfig,
    /// Evaluation cache
    cache: HashMap<u64, EvaluationResult>,
    /// Number of evaluations performed
    eval_count: usize,
}

impl ArchitectureEvaluator {
    /// Create new evaluator
    pub fn new(config: EvaluationConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            eval_count: 0,
        }
    }

    /// Create with default config
    pub fn default_evaluator() -> Self {
        Self::new(EvaluationConfig::default())
    }

    /// Estimate number of parameters in architecture
    pub fn estimate_params(&self, arch: &NetworkArchitecture) -> usize {
        let mut params = 0;
        let hidden = arch.hidden_dim;

        // Input projection
        params += arch.input_dim * hidden;

        // Each cell
        for cell in &arch.cells {
            for op in &cell.operations {
                let op_hidden = op.hidden_dim.unwrap_or(hidden);
                
                match op.op_type {
                    super::search_space::OperationType::Dense => {
                        params += hidden * op_hidden + op_hidden;
                    }
                    super::search_space::OperationType::MultiHeadAttention => {
                        let _heads = op.num_heads.unwrap_or(4);
                        // Q, K, V projections + output
                        params += 4 * hidden * op_hidden;
                    }
                    super::search_space::OperationType::LayerNorm | 
                    super::search_space::OperationType::BatchNorm => {
                        params += 2 * hidden; // gamma, beta
                    }
                    super::search_space::OperationType::Conv1D => {
                        let kernel = op.kernel_size.unwrap_or(3);
                        params += kernel * hidden * op_hidden + op_hidden;
                    }
                    _ => {}
                }
            }
        }

        // Output projection
        params += hidden * arch.output_dim + arch.output_dim;

        params
    }

    /// Estimate FLOPs for architecture
    pub fn estimate_flops(&self, arch: &NetworkArchitecture, seq_len: usize) -> usize {
        let mut flops = 0;
        let hidden = arch.hidden_dim;

        // Input projection
        flops += 2 * seq_len * arch.input_dim * hidden;

        // Each cell
        for cell in &arch.cells {
            for op in &cell.operations {
                let op_hidden = op.hidden_dim.unwrap_or(hidden);
                
                match op.op_type {
                    super::search_space::OperationType::Dense => {
                        flops += 2 * seq_len * hidden * op_hidden;
                    }
                    super::search_space::OperationType::MultiHeadAttention => {
                        // Attention: O(n^2 * d)
                        flops += 4 * seq_len * seq_len * hidden;
                    }
                    super::search_space::OperationType::Conv1D => {
                        let kernel = op.kernel_size.unwrap_or(3);
                        flops += 2 * seq_len * kernel * hidden * op_hidden;
                    }
                    _ => {}
                }
            }
        }

        // Output projection
        flops += 2 * seq_len * hidden * arch.output_dim;

        flops
    }

    /// Check cache for previous evaluation
    pub fn get_cached(&self, arch: &NetworkArchitecture) -> Option<&EvaluationResult> {
        let hash = arch.compute_hash();
        self.cache.get(&hash)
    }

    /// Add result to cache
    pub fn cache_result(&mut self, result: EvaluationResult) {
        self.cache.insert(result.arch_id, result);
    }

    /// Quick proxy evaluation (few epochs)
    pub fn proxy_evaluate(
        &mut self,
        arch: &NetworkArchitecture,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<EvaluationResult> {
        let arch_id = arch.compute_hash();

        // Check cache
        if let Some(cached) = self.cache.get(&arch_id) {
            return Ok(cached.clone());
        }

        self.eval_count += 1;

        // Estimate parameters
        let num_params = self.estimate_params(arch);

        // Simulate training (placeholder for actual neural network training)
        let train_loss = self.simulate_training(arch, x_train, y_train, self.config.proxy_epochs);
        let (val_loss, val_metric) = self.simulate_validation(arch, x_val, y_val);

        let result = EvaluationResult::new(arch_id)
            .with_losses(train_loss, val_loss)
            .with_val_metric(val_metric)
            .with_num_params(num_params);

        self.cache_result(result.clone());
        Ok(result)
    }

    /// Full evaluation with more epochs
    pub fn full_evaluate(
        &mut self,
        arch: &NetworkArchitecture,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<EvaluationResult> {
        let arch_id = arch.compute_hash();
        self.eval_count += 1;

        let start_time = std::time::Instant::now();

        // Estimate parameters
        let num_params = self.estimate_params(arch);
        let _flops = self.estimate_flops(arch, x_train.nrows());

        // Simulate training
        let train_loss = self.simulate_training(arch, x_train, y_train, self.config.epochs);
        let (val_loss, val_metric) = self.simulate_validation(arch, x_val, y_val);

        let train_time = start_time.elapsed().as_secs_f64();

        let result = EvaluationResult::new(arch_id)
            .with_losses(train_loss, val_loss)
            .with_val_metric(val_metric)
            .with_num_params(num_params)
            .with_train_time(train_time);

        Ok(result)
    }

    /// Simulate training (placeholder - returns synthetic loss based on architecture)
    fn simulate_training(
        &self,
        arch: &NetworkArchitecture,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
        epochs: usize,
    ) -> f64 {
        // Synthetic loss based on architecture complexity
        let complexity = arch.cells.iter()
            .map(|c| c.operations.len())
            .sum::<usize>() as f64;
        
        let base_loss = 1.0 / (1.0 + 0.1 * complexity);
        let decay = 0.95_f64.powi(epochs as i32);
        
        base_loss * decay
    }

    /// Simulate validation (placeholder - returns synthetic metrics)
    fn simulate_validation(
        &self,
        arch: &NetworkArchitecture,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> (f64, f64) {
        // Synthetic validation based on architecture
        let complexity = arch.cells.iter()
            .map(|c| c.operations.len())
            .sum::<usize>() as f64;
        
        let val_loss = 0.5 / (1.0 + 0.05 * complexity);
        let val_acc = 0.5 + 0.05 * complexity.min(10.0);
        
        (val_loss, val_acc)
    }

    /// Get number of evaluations
    pub fn eval_count(&self) -> usize {
        self.eval_count
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Multi-fidelity evaluator for efficient search
pub struct MultiFidelityEvaluator {
    /// Fidelity levels (epochs per level)
    fidelity_levels: Vec<usize>,
    /// Base evaluator
    evaluator: ArchitectureEvaluator,
    /// Results at each fidelity level
    results: HashMap<u64, Vec<EvaluationResult>>,
}

impl MultiFidelityEvaluator {
    /// Create new multi-fidelity evaluator
    pub fn new(fidelity_levels: Vec<usize>) -> Self {
        Self {
            fidelity_levels,
            evaluator: ArchitectureEvaluator::default_evaluator(),
            results: HashMap::new(),
        }
    }

    /// Create with Hyperband-style fidelities
    pub fn hyperband_style(max_epochs: usize, eta: usize) -> Self {
        let mut levels = Vec::new();
        let mut epochs = max_epochs;
        while epochs >= 1 {
            levels.push(epochs);
            epochs /= eta;
        }
        levels.reverse();
        Self::new(levels)
    }

    /// Evaluate at specific fidelity level
    pub fn evaluate_at_fidelity(
        &mut self,
        arch: &NetworkArchitecture,
        fidelity: usize,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<EvaluationResult> {
        let epochs = self.fidelity_levels.get(fidelity)
            .copied()
            .unwrap_or(self.fidelity_levels.last().copied().unwrap_or(10));

        self.evaluator.config.proxy_epochs = epochs;
        self.evaluator.proxy_evaluate(arch, x_train, y_train, x_val, y_val)
    }

    /// Get fidelity levels
    pub fn levels(&self) -> &[usize] {
        &self.fidelity_levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::search_space::*;

    fn create_test_arch() -> NetworkArchitecture {
        NetworkArchitecture::new(10, 2)
            .with_hidden_dim(64)
            .with_num_layers(2)
            .add_cell(
                Cell::new(CellType::Normal)
                    .add_operation(Operation::dense(64), vec![0])
            )
    }

    #[test]
    fn test_evaluator_creation() {
        let evaluator = ArchitectureEvaluator::default_evaluator();
        assert_eq!(evaluator.eval_count(), 0);
    }

    #[test]
    fn test_estimate_params() {
        let evaluator = ArchitectureEvaluator::default_evaluator();
        let arch = create_test_arch();
        
        let params = evaluator.estimate_params(&arch);
        assert!(params > 0);
    }

    #[test]
    fn test_proxy_evaluate() {
        let mut evaluator = ArchitectureEvaluator::default_evaluator();
        let arch = create_test_arch();
        
        let x_train = Array2::zeros((100, 10));
        let y_train = Array1::zeros(100);
        let x_val = Array2::zeros((20, 10));
        let y_val = Array1::zeros(20);
        
        let result = evaluator.proxy_evaluate(&arch, &x_train, &y_train, &x_val, &y_val).unwrap();
        
        assert!(result.val_metric > 0.0);
        assert!(result.num_params > 0);
    }

    #[test]
    fn test_caching() {
        let mut evaluator = ArchitectureEvaluator::default_evaluator();
        let arch = create_test_arch();
        
        let x = Array2::zeros((100, 10));
        let y = Array1::zeros(100);
        
        // First evaluation
        let _result1 = evaluator.proxy_evaluate(&arch, &x, &y, &x, &y).unwrap();
        assert_eq!(evaluator.eval_count(), 1);
        
        // Should use cache
        let _result2 = evaluator.proxy_evaluate(&arch, &x, &y, &x, &y).unwrap();
        assert_eq!(evaluator.eval_count(), 1); // No new evaluation
    }

    #[test]
    fn test_multi_fidelity() {
        let evaluator = MultiFidelityEvaluator::hyperband_style(81, 3);
        
        // Should have levels: 1, 3, 9, 27, 81
        assert!(!evaluator.levels().is_empty());
    }

    #[test]
    fn test_evaluation_result_builder() {
        let result = EvaluationResult::new(12345)
            .with_losses(0.5, 0.4)
            .with_val_metric(0.85)
            .with_num_params(10000)
            .add_metric("f1_score", 0.82);
        
        assert_eq!(result.train_loss, 0.5);
        assert_eq!(result.val_metric, 0.85);
        assert_eq!(result.extra_metrics.get("f1_score"), Some(&0.82));
    }
}
