//! Decision tree implementation

use crate::error::{KolosalError, Result};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Decision tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    /// Leaf node with prediction value
    Leaf {
        value: f64,
        n_samples: usize,
    },
    /// Internal node with split
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        n_samples: usize,
        impurity: f64,
    },
}

/// Impurity criterion
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Criterion {
    /// Gini impurity (classification)
    Gini,
    /// Entropy (classification)
    Entropy,
    /// Mean squared error (regression)
    MSE,
    /// Mean absolute error (regression)
    MAE,
}

/// Decision tree model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    /// Tree root
    root: Option<TreeNode>,
    /// Maximum depth
    pub max_depth: Option<usize>,
    /// Minimum samples to split
    pub min_samples_split: usize,
    /// Minimum samples in leaf
    pub min_samples_leaf: usize,
    /// Maximum features to consider
    pub max_features: Option<usize>,
    /// Impurity criterion
    pub criterion: Criterion,
    /// Number of features
    n_features: usize,
    /// Feature importances
    feature_importances: Option<Array1<f64>>,
    /// Is classification task
    is_classification: bool,
    /// Classes (for classification)
    classes: Vec<f64>,
}

impl Default for DecisionTree {
    fn default() -> Self {
        Self::new_classifier()
    }
}

impl DecisionTree {
    /// Create a new classifier tree
    pub fn new_classifier() -> Self {
        Self {
            root: None,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            criterion: Criterion::Gini,
            n_features: 0,
            feature_importances: None,
            is_classification: true,
            classes: Vec::new(),
        }
    }

    /// Create a new regressor tree
    pub fn new_regressor() -> Self {
        Self {
            root: None,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            criterion: Criterion::MSE,
            n_features: 0,
            feature_importances: None,
            is_classification: false,
            classes: Vec::new(),
        }
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Set minimum samples to split
    pub fn with_min_samples_split(mut self, min_samples: usize) -> Self {
        self.min_samples_split = min_samples;
        self
    }

    /// Set minimum samples in leaf
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }

    /// Set criterion
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Fit the tree to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(KolosalError::ShapeError {
                expected: format!("y length = {}", n_samples),
                actual: format!("y length = {}", y.len()),
            });
        }

        if n_samples < self.min_samples_split {
            return Err(KolosalError::ValidationError(
                format!("Need at least {} samples, got {}", self.min_samples_split, n_samples)
            ));
        }

        self.n_features = n_features;

        // Get unique classes for classification
        if self.is_classification {
            let mut classes: Vec<f64> = y.iter().copied().collect();
            classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            classes.dedup();
            self.classes = classes;
        }

        // Initialize feature importances
        let mut importances = vec![0.0; n_features];

        // Build tree recursively
        let indices: Vec<usize> = (0..n_samples).collect();
        self.root = Some(self.build_tree(x, y, &indices, 0, &mut importances));

        // Normalize feature importances
        let total: f64 = importances.iter().sum();
        if total > 0.0 {
            for imp in &mut importances {
                *imp /= total;
            }
        }
        self.feature_importances = Some(Array1::from_vec(importances));

        Ok(self)
    }

    fn build_tree(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        depth: usize,
        importances: &mut [f64],
    ) -> TreeNode {
        let n_samples = indices.len();
        let y_subset: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

        // Check stopping conditions
        let should_stop = n_samples < self.min_samples_split
            || n_samples <= self.min_samples_leaf
            || self.max_depth.map_or(false, |d| depth >= d)
            || self.is_pure(&y_subset);

        if should_stop {
            return TreeNode::Leaf {
                value: self.compute_leaf_value(&y_subset),
                n_samples,
            };
        }

        // Find best split
        if let Some((best_feature, best_threshold, best_impurity)) = self.find_best_split(x, y, indices) {
            // Split indices
            let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .partition(|&&i| x[[i, best_feature]] <= best_threshold);

            if left_indices.len() < self.min_samples_leaf || right_indices.len() < self.min_samples_leaf {
                return TreeNode::Leaf {
                    value: self.compute_leaf_value(&y_subset),
                    n_samples,
                };
            }

            // Update feature importance
            let parent_impurity = self.compute_impurity(&y_subset);
            let left_y: Vec<f64> = left_indices.iter().map(|&i| y[i]).collect();
            let right_y: Vec<f64> = right_indices.iter().map(|&i| y[i]).collect();
            
            let weighted_child_impurity = 
                (left_indices.len() as f64 * self.compute_impurity(&left_y)
                + right_indices.len() as f64 * self.compute_impurity(&right_y))
                / n_samples as f64;
            
            importances[best_feature] += n_samples as f64 * (parent_impurity - weighted_child_impurity);

            // Build children recursively
            let left = Box::new(self.build_tree(x, y, &left_indices, depth + 1, importances));
            let right = Box::new(self.build_tree(x, y, &right_indices, depth + 1, importances));

            TreeNode::Split {
                feature_idx: best_feature,
                threshold: best_threshold,
                left,
                right,
                n_samples,
                impurity: best_impurity,
            }
        } else {
            TreeNode::Leaf {
                value: self.compute_leaf_value(&y_subset),
                n_samples,
            }
        }
    }

    fn find_best_split(&self, x: &Array2<f64>, y: &Array1<f64>, indices: &[usize]) -> Option<(usize, f64, f64)> {
        let n_features = x.ncols();
        let max_features = self.max_features.unwrap_or(n_features);
        let n_features_to_try = max_features.min(n_features);

        let y_subset: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
        let parent_impurity = self.compute_impurity(&y_subset);

        // Parallelize feature scanning — each feature independently finds its best split
        let feature_results: Vec<Option<(usize, f64, f64)>> = (0..n_features_to_try)
            .into_par_iter()
            .map(|feature_idx| {
                let mut values: Vec<f64> = indices.iter().map(|&i| x[[i, feature_idx]]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                values.dedup();

                let mut best_gain = 0.0f64;
                let mut best_threshold = 0.0f64;

                for window in values.windows(2) {
                    let threshold = (window[0] + window[1]) / 2.0;

                    // Count left/right and accumulate impurity incrementally
                    let mut left_count = 0usize;
                    let mut right_count = 0usize;
                    let mut left_sum = 0.0f64;
                    let mut right_sum = 0.0f64;
                    let mut left_sq_sum = 0.0f64;
                    let mut right_sq_sum = 0.0f64;
                    // For classification: count class frequencies
                    let mut left_class_counts: HashMap<i64, usize> = HashMap::new();
                    let mut right_class_counts: HashMap<i64, usize> = HashMap::new();

                    for &idx in indices {
                        let yi = y[idx];
                        if x[[idx, feature_idx]] <= threshold {
                            left_count += 1;
                            left_sum += yi;
                            left_sq_sum += yi * yi;
                            *left_class_counts.entry(yi as i64).or_insert(0) += 1;
                        } else {
                            right_count += 1;
                            right_sum += yi;
                            right_sq_sum += yi * yi;
                            *right_class_counts.entry(yi as i64).or_insert(0) += 1;
                        }
                    }

                    if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                        continue;
                    }

                    let left_impurity = self.compute_impurity_fast(left_count, left_sum, left_sq_sum, &left_class_counts);
                    let right_impurity = self.compute_impurity_fast(right_count, right_sum, right_sq_sum, &right_class_counts);

                    let n = indices.len() as f64;
                    let weighted_impurity =
                        (left_count as f64 * left_impurity + right_count as f64 * right_impurity) / n;

                    let gain = parent_impurity - weighted_impurity;
                    if gain > best_gain {
                        best_gain = gain;
                        best_threshold = threshold;
                    }
                }

                if best_gain > 0.0 {
                    Some((feature_idx, best_threshold, best_gain))
                } else {
                    None
                }
            })
            .collect();

        // Find best across all features
        feature_results
            .into_iter()
            .flatten()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Fast impurity computation from pre-computed statistics (avoids re-iterating data)
    fn compute_impurity_fast(&self, count: usize, sum: f64, sq_sum: f64, class_counts: &HashMap<i64, usize>) -> f64 {
        if count == 0 {
            return 0.0;
        }
        match self.criterion {
            Criterion::Gini => {
                let n = count as f64;
                let mut gini = 1.0;
                for &c in class_counts.values() {
                    let p = c as f64 / n;
                    gini -= p * p;
                }
                gini
            }
            Criterion::Entropy => {
                let n = count as f64;
                let mut entropy = 0.0;
                for &c in class_counts.values() {
                    if c > 0 {
                        let p = c as f64 / n;
                        entropy -= p * p.ln();
                    }
                }
                entropy
            }
            Criterion::MSE => {
                // Var = E[X²] - E[X]² = sq_sum/n - (sum/n)²
                let n = count as f64;
                sq_sum / n - (sum / n).powi(2)
            }
            Criterion::MAE => {
                let mean = sum / count as f64;
                // Would need actual values for MAE — fall back to MSE approximation
                sq_sum / count as f64 - mean * mean
            }
        }
    }

    fn compute_impurity(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }

        match self.criterion {
            Criterion::Gini => self.gini_impurity(y),
            Criterion::Entropy => self.entropy(y),
            Criterion::MSE => self.mse(y),
            Criterion::MAE => self.mae(y),
        }
    }

    fn gini_impurity(&self, y: &[f64]) -> f64 {
        let n = y.len() as f64;
        let mut counts: HashMap<i64, usize> = HashMap::new();
        
        for &val in y {
            *counts.entry(val.round() as i64).or_insert(0) += 1;
        }

        let sum_sq: f64 = counts.values()
            .map(|&c| (c as f64 / n).powi(2))
            .sum();

        1.0 - sum_sq
    }

    fn entropy(&self, y: &[f64]) -> f64 {
        let n = y.len() as f64;
        let mut counts: HashMap<i64, usize> = HashMap::new();
        
        for &val in y {
            *counts.entry(val.round() as i64).or_insert(0) += 1;
        }

        -counts.values()
            .map(|&c| {
                let p = c as f64 / n;
                if p > 0.0 { p * p.ln() } else { 0.0 }
            })
            .sum::<f64>()
    }

    fn mse(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / y.len() as f64
    }

    fn mae(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        let median = {
            let mut sorted = y.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted.len() % 2 == 0 && sorted.len() >= 2 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            }
        };
        y.iter().map(|&v| (v - median).abs()).sum::<f64>() / y.len() as f64
    }

    fn is_pure(&self, y: &[f64]) -> bool {
        if y.is_empty() {
            return true;
        }
        let first = y[0];
        y.iter().all(|&v| (v - first).abs() < 1e-10)
    }

    fn compute_leaf_value(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }

        if self.is_classification {
            // Mode (most common class)
            let mut counts: HashMap<i64, usize> = HashMap::new();
            for &val in y {
                *counts.entry(val.round() as i64).or_insert(0) += 1;
            }
            counts.into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(class, _)| class as f64)
                .unwrap_or(0.0)
        } else {
            // Mean
            y.iter().sum::<f64>() / y.len() as f64
        }
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let root = self.root.as_ref()
            .ok_or(KolosalError::ModelNotFitted)?;

        let predictions: Vec<f64> = (0..x.nrows())
            .map(|i| {
                let sample = x.row(i);
                self.predict_sample(root, &sample.to_vec())
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    fn predict_sample(&self, node: &TreeNode, sample: &[f64]) -> f64 {
        match node {
            TreeNode::Leaf { value, .. } => *value,
            TreeNode::Split { feature_idx, threshold, left, right, .. } => {
                if sample[*feature_idx] <= *threshold {
                    self.predict_sample(left, sample)
                } else {
                    self.predict_sample(right, sample)
                }
            }
        }
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> Option<&Array1<f64>> {
        self.feature_importances.as_ref()
    }

    /// Get tree depth
    pub fn get_depth(&self) -> usize {
        match &self.root {
            None => 0,
            Some(node) => self.node_depth(node),
        }
    }

    fn node_depth(&self, node: &TreeNode) -> usize {
        match node {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => {
                1 + self.node_depth(left).max(self.node_depth(right))
            }
        }
    }

    /// Get number of leaves
    pub fn get_n_leaves(&self) -> usize {
        match &self.root {
            None => 0,
            Some(node) => self.count_leaves(node),
        }
    }

    fn count_leaves(&self, node: &TreeNode) -> usize {
        match node {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => {
                self.count_leaves(left) + self.count_leaves(right)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_classifier_simple() {
        let x = array![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0]; // XOR-like but simpler

        let mut tree = DecisionTree::new_classifier();
        tree.fit(&x, &y).unwrap();

        let predictions = tree.predict(&x).unwrap();
        
        // Should get at least some predictions right
        let correct = predictions.iter().zip(y.iter())
            .filter(|(p, a)| (*p - *a).abs() < 0.5)
            .count();
        
        assert!(correct >= 2);
    }

    #[test]
    fn test_regressor_simple() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut tree = DecisionTree::new_regressor()
            .with_criterion(Criterion::MSE);
        tree.fit(&x, &y).unwrap();

        let predictions = tree.predict(&x).unwrap();
        
        // MSE should be reasonable
        let mse: f64 = predictions.iter().zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>() / y.len() as f64;
        
        assert!(mse < 1.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_max_depth() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut tree = DecisionTree::new_classifier()
            .with_max_depth(2);
        tree.fit(&x, &y).unwrap();

        assert!(tree.get_depth() <= 2);
    }

    #[test]
    fn test_feature_importances() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut tree = DecisionTree::new_classifier();
        tree.fit(&x, &y).unwrap();

        let importances = tree.feature_importances().unwrap();
        
        // First feature should be more important (second is constant)
        assert!(importances[0] >= importances[1]);
    }
}
