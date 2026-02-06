//! DARTS - Differentiable Architecture Search
//!
//! Implements differentiable architecture search using continuous relaxation
//! of the discrete architecture space.

use ndarray::{Array1, Array2, Array3, Axis};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use super::search_space::{NASSearchSpace, NetworkArchitecture, Operation, OperationType, Cell, CellType};

/// DARTS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DARTSConfig {
    /// Number of nodes per cell
    pub nodes_per_cell: usize,
    /// Architecture learning rate
    pub arch_learning_rate: f64,
    /// Weight learning rate
    pub weight_learning_rate: f64,
    /// Architecture weight decay
    pub arch_weight_decay: f64,
    /// Number of epochs for search
    pub search_epochs: usize,
    /// Number of warmup epochs (train weights only)
    pub warmup_epochs: usize,
    /// Unrolled optimization steps
    pub unrolled_steps: usize,
}

impl Default for DARTSConfig {
    fn default() -> Self {
        Self {
            nodes_per_cell: 4,
            arch_learning_rate: 0.001,
            weight_learning_rate: 0.01,
            arch_weight_decay: 0.001,
            search_epochs: 50,
            warmup_epochs: 10,
            unrolled_steps: 1,
        }
    }
}

/// Architecture weights for differentiable search
#[derive(Debug, Clone)]
pub struct ArchitectureWeights {
    /// Alpha weights for normal cell edges
    /// Shape: [num_nodes, num_prev_nodes, num_ops]
    pub alpha_normal: Array3<f64>,
    /// Alpha weights for reduction cell edges
    pub alpha_reduce: Array3<f64>,
    /// Beta weights for input selection (optional)
    pub beta: Option<Array2<f64>>,
}

impl ArchitectureWeights {
    /// Create new architecture weights
    pub fn new(num_nodes: usize, num_ops: usize, rng: &mut impl Rng) -> Self {
        let scale = 0.001;
        
        // For each node, we have connections from all previous nodes + 2 inputs
        let max_inputs = num_nodes + 2;
        
        let mut alpha_normal = Array3::zeros((num_nodes, max_inputs, num_ops));
        let mut alpha_reduce = Array3::zeros((num_nodes, max_inputs, num_ops));

        for val in alpha_normal.iter_mut() {
            *val = (rng.gen::<f64>() - 0.5) * scale;
        }
        for val in alpha_reduce.iter_mut() {
            *val = (rng.gen::<f64>() - 0.5) * scale;
        }

        Self {
            alpha_normal,
            alpha_reduce,
            beta: None,
        }
    }

    /// Get softmax probabilities for operations on an edge
    pub fn edge_probs(&self, node: usize, prev: usize, is_reduce: bool) -> Array1<f64> {
        let alpha = if is_reduce { &self.alpha_reduce } else { &self.alpha_normal };
        let logits = alpha.slice(ndarray::s![node, prev, ..]).to_owned();
        softmax(&logits)
    }

    /// Get the most likely operation for an edge
    pub fn best_op(&self, node: usize, prev: usize, is_reduce: bool) -> usize {
        let probs = self.edge_probs(node, prev, is_reduce);
        probs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update weights using gradient
    pub fn update(&mut self, grad_normal: &Array3<f64>, grad_reduce: &Array3<f64>, lr: f64, weight_decay: f64) {
        // SGD update with weight decay
        self.alpha_normal.zip_mut_with(grad_normal, |a, g| {
            *a -= lr * (g + weight_decay * *a);
        });
        self.alpha_reduce.zip_mut_with(grad_reduce, |a, g| {
            *a -= lr * (g + weight_decay * *a);
        });
    }
}

/// DARTS search algorithm
#[derive(Debug)]
pub struct DARTSSearch {
    /// Configuration
    config: DARTSConfig,
    /// Search space
    search_space: NASSearchSpace,
    /// Architecture weights
    arch_weights: ArchitectureWeights,
    /// Current epoch
    epoch: usize,
    /// Best validation accuracy seen
    best_val_acc: f64,
    /// Best architecture found
    best_arch: Option<NetworkArchitecture>,
    /// Random number generator
    rng: Xoshiro256PlusPlus,
    /// Search history
    history: Vec<SearchStep>,
}

/// A single search step record
#[derive(Debug, Clone)]
pub struct SearchStep {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub val_acc: f64,
}

impl DARTSSearch {
    /// Create new DARTS search
    pub fn new(search_space: NASSearchSpace, config: DARTSConfig, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => Xoshiro256PlusPlus::seed_from_u64(s),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        let num_ops = search_space.num_operations();
        let arch_weights = ArchitectureWeights::new(config.nodes_per_cell, num_ops, &mut rng);

        Self {
            config,
            search_space,
            arch_weights,
            epoch: 0,
            best_val_acc: 0.0,
            best_arch: None,
            rng,
            history: Vec::new(),
        }
    }

    /// Get architecture weights
    pub fn weights(&self) -> &ArchitectureWeights {
        &self.arch_weights
    }

    /// Derive discrete architecture from continuous weights
    pub fn derive_architecture(&self) -> NetworkArchitecture {
        let num_nodes = self.config.nodes_per_cell;
        let ops = &self.search_space.config.operations;
        
        // Get default hidden dim
        let hidden_dims = self.search_space.hidden_dim_choices();
        let hidden_dim = hidden_dims[hidden_dims.len() / 2];

        let mut arch = NetworkArchitecture::new(0, 0)
            .with_hidden_dim(hidden_dim)
            .with_num_layers(2);

        // Build normal cell
        let mut normal_cell = Cell::new(CellType::Normal);
        for node in 0..num_nodes {
            // Select top-2 input edges for this node
            let mut edge_scores: Vec<(usize, usize, f64)> = Vec::new();
            
            for prev in 0..(node + 2) {
                let probs = self.arch_weights.edge_probs(node, prev, false);
                let best_op = probs.iter()
                    .enumerate()
                    .filter(|(i, _)| ops[*i] != OperationType::None)
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap());
                
                if let Some((op_idx, &score)) = best_op {
                    edge_scores.push((prev, op_idx, score));
                }
            }

            // Sort by score and take top 2
            edge_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            
            for (prev, op_idx, _) in edge_scores.into_iter().take(2) {
                let op_type = ops[op_idx];
                let mut op = Operation::new(op_type);
                if matches!(op_type, OperationType::Dense | OperationType::Attention) {
                    op = op.with_hidden_dim(hidden_dim);
                }
                normal_cell = normal_cell.add_operation(op, vec![prev]);
            }
        }
        arch = arch.add_cell(normal_cell);

        // Build reduction cell
        let mut reduce_cell = Cell::new(CellType::Reduction);
        for node in 0..num_nodes {
            let mut edge_scores: Vec<(usize, usize, f64)> = Vec::new();
            
            for prev in 0..(node + 2) {
                let probs = self.arch_weights.edge_probs(node, prev, true);
                let best_op = probs.iter()
                    .enumerate()
                    .filter(|(i, _)| ops[*i] != OperationType::None)
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap());
                
                if let Some((op_idx, &score)) = best_op {
                    edge_scores.push((prev, op_idx, score));
                }
            }

            edge_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            
            for (prev, op_idx, _) in edge_scores.into_iter().take(2) {
                let op_type = ops[op_idx];
                let mut op = Operation::new(op_type);
                if matches!(op_type, OperationType::Dense | OperationType::Attention) {
                    op = op.with_hidden_dim(hidden_dim);
                }
                reduce_cell = reduce_cell.add_operation(op, vec![prev]);
            }
        }
        arch = arch.add_cell(reduce_cell);

        arch
    }

    /// Perform one step of architecture search
    /// 
    /// In a full implementation, this would:
    /// 1. Train network weights on training data
    /// 2. Update architecture weights on validation data
    pub fn step(&mut self, train_loss: f64, val_loss: f64, val_acc: f64) {
        self.epoch += 1;

        // Record history
        self.history.push(SearchStep {
            epoch: self.epoch,
            train_loss,
            val_loss,
            val_acc,
        });

        // Update best architecture
        if val_acc > self.best_val_acc {
            self.best_val_acc = val_acc;
            self.best_arch = Some(self.derive_architecture());
        }

        // Simulate architecture weight update (in practice, use actual gradients)
        if self.epoch > self.config.warmup_epochs {
            let num_nodes = self.config.nodes_per_cell;
            let num_ops = self.search_space.num_operations();
            let max_inputs = num_nodes + 2;

            // Create random gradients (placeholder for actual gradient computation)
            let grad_scale = 0.01 * (1.0 - val_acc);
            let mut grad_normal = Array3::zeros((num_nodes, max_inputs, num_ops));
            let mut grad_reduce = Array3::zeros((num_nodes, max_inputs, num_ops));

            for val in grad_normal.iter_mut() {
                *val = (self.rng.gen::<f64>() - 0.5) * grad_scale;
            }
            for val in grad_reduce.iter_mut() {
                *val = (self.rng.gen::<f64>() - 0.5) * grad_scale;
            }

            self.arch_weights.update(
                &grad_normal,
                &grad_reduce,
                self.config.arch_learning_rate,
                self.config.arch_weight_decay,
            );
        }
    }

    /// Check if search is complete
    pub fn is_complete(&self) -> bool {
        self.epoch >= self.config.search_epochs
    }

    /// Get best architecture found
    pub fn best_architecture(&self) -> Option<&NetworkArchitecture> {
        self.best_arch.as_ref()
    }

    /// Get search history
    pub fn history(&self) -> &[SearchStep] {
        &self.history
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> usize {
        self.epoch
    }

    /// Run complete search with evaluation function
    pub fn search<F>(&mut self, mut evaluate: F) -> NetworkArchitecture
    where
        F: FnMut(&NetworkArchitecture) -> (f64, f64, f64), // (train_loss, val_loss, val_acc)
    {
        while !self.is_complete() {
            let arch = self.derive_architecture();
            let (train_loss, val_loss, val_acc) = evaluate(&arch);
            self.step(train_loss, val_loss, val_acc);
        }

        self.best_architecture()
            .cloned()
            .unwrap_or_else(|| self.derive_architecture())
    }
}

/// Softmax function
fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Array1<f64> = logits.mapv(|x| (x - max_val).exp());
    let sum: f64 = exp.sum();
    if sum > 0.0 {
        exp / sum
    } else {
        Array1::from_elem(logits.len(), 1.0 / logits.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_darts_creation() {
        let space = NASSearchSpace::tabular();
        let config = DARTSConfig::default();
        let darts = DARTSSearch::new(space, config, Some(42));
        
        assert_eq!(darts.current_epoch(), 0);
        assert!(!darts.is_complete());
    }

    #[test]
    fn test_arch_weights() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let weights = ArchitectureWeights::new(4, 9, &mut rng);
        
        let probs = weights.edge_probs(0, 0, false);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_derive_architecture() {
        let space = NASSearchSpace::tabular();
        let config = DARTSConfig::default();
        let darts = DARTSSearch::new(space, config, Some(42));
        
        let arch = darts.derive_architecture();
        assert!(!arch.cells.is_empty());
    }

    #[test]
    fn test_darts_step() {
        let space = NASSearchSpace::tabular();
        let config = DARTSConfig {
            search_epochs: 5,
            warmup_epochs: 1,
            ..Default::default()
        };
        let mut darts = DARTSSearch::new(space, config, Some(42));
        
        for i in 0..5 {
            let val_acc = 0.5 + 0.1 * i as f64;
            darts.step(0.5, 0.4, val_acc);
        }
        
        assert!(darts.is_complete());
        assert!(darts.best_architecture().is_some());
    }

    #[test]
    fn test_darts_search() {
        let space = NASSearchSpace::tabular();
        let config = DARTSConfig {
            search_epochs: 3,
            warmup_epochs: 1,
            ..Default::default()
        };
        let mut darts = DARTSSearch::new(space, config, Some(42));
        
        let mut epoch = 0;
        let arch = darts.search(|_arch| {
            epoch += 1;
            (0.5 / epoch as f64, 0.4 / epoch as f64, 0.5 + 0.1 * epoch as f64)
        });
        
        assert!(!arch.cells.is_empty());
    }
}
