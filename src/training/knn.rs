//! K-Nearest Neighbors implementation
//!
//! KNN classifier and regressor with distance metrics.
//! Uses a KD-tree for Euclidean/Manhattan distance (O(log n) queries)
//! and brute-force for other metrics.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::error::Result;

/// Distance metric for KNN
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance (L2)
    Euclidean,
    /// Manhattan distance (L1)
    Manhattan,
    /// Minkowski distance with parameter p
    Minkowski(f64),
    /// Cosine similarity (converted to distance)
    Cosine,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Euclidean
    }
}

/// KNN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNConfig {
    /// Number of neighbors
    pub n_neighbors: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Weighting scheme
    pub weights: WeightScheme,
}

/// Weighting scheme for neighbors
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WeightScheme {
    /// All neighbors have equal weight
    Uniform,
    /// Closer neighbors have more weight (inverse distance)
    Distance,
}

impl Default for WeightScheme {
    fn default() -> Self {
        Self::Uniform
    }
}

impl Default for KNNConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            metric: DistanceMetric::Euclidean,
            weights: WeightScheme::Uniform,
        }
    }
}

// ============================================================================
// KD-tree for O(log n) nearest neighbor queries
// ============================================================================

/// A node in the KD-tree
#[derive(Debug, Clone, Serialize, Deserialize)]
enum KdNode {
    Leaf {
        /// Indices of training points stored in this leaf
        indices: Vec<usize>,
    },
    Split {
        /// Dimension to split on
        dim: usize,
        /// Split value (median along dim)
        median: f64,
        /// Training point index at the split
        idx: usize,
        left: Box<KdNode>,
        right: Box<KdNode>,
    },
}

/// KD-tree for fast nearest neighbor search
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KdTree {
    root: Option<KdNode>,
    n_dims: usize,
}

/// Maximum leaf size — below this, we store a flat list and brute-force scan
const KD_LEAF_SIZE: usize = 16;

impl KdTree {
    /// Build a KD-tree from training data
    fn build(data: &Array2<f64>) -> Self {
        let n = data.nrows();
        let n_dims = data.ncols();
        if n == 0 {
            return Self { root: None, n_dims };
        }

        let mut indices: Vec<usize> = (0..n).collect();
        let root = Self::build_recursive(data, &mut indices, 0, n_dims);
        Self { root: Some(root), n_dims }
    }

    fn build_recursive(data: &Array2<f64>, indices: &mut [usize], depth: usize, n_dims: usize) -> KdNode {
        if indices.len() <= KD_LEAF_SIZE {
            return KdNode::Leaf { indices: indices.to_vec() };
        }

        let dim = depth % n_dims;

        // Partition by median along current dimension
        indices.sort_unstable_by(|&a, &b| {
            data[[a, dim]].partial_cmp(&data[[b, dim]]).unwrap_or(Ordering::Equal)
        });

        let mid = indices.len() / 2;
        let idx = indices[mid];
        let median = data[[idx, dim]];

        let (left_slice, rest) = indices.split_at_mut(mid);
        let right_slice = &mut rest[1..]; // skip the median element

        let left = Box::new(Self::build_recursive(data, left_slice, depth + 1, n_dims));
        let right = Box::new(Self::build_recursive(data, right_slice, depth + 1, n_dims));

        KdNode::Split { dim, median, idx, left, right }
    }

    /// Find k nearest neighbors to a query point.
    /// Returns Vec<(distance, training_index)>.
    fn query_k_nearest(&self, data: &Array2<f64>, query: &[f64], k: usize, metric: DistanceMetric) -> Vec<(f64, usize)> {
        let mut heap = BinaryHeap::with_capacity(k + 1);
        if let Some(ref root) = self.root {
            self.search(root, data, query, k, metric, &mut heap);
        }
        heap.into_iter().map(|di| (di.0, di.1)).collect()
    }

    fn search(
        &self,
        node: &KdNode,
        data: &Array2<f64>,
        query: &[f64],
        k: usize,
        metric: DistanceMetric,
        heap: &mut BinaryHeap<DistIdx>,
    ) {
        match node {
            KdNode::Leaf { indices } => {
                for &idx in indices {
                    let row = data.row(idx);
                    let dist = compute_distance(query, row.as_slice().unwrap(), metric);
                    if heap.len() < k {
                        heap.push(DistIdx(dist, idx));
                    } else if let Some(top) = heap.peek() {
                        if dist < top.0 {
                            heap.pop();
                            heap.push(DistIdx(dist, idx));
                        }
                    }
                }
            }
            KdNode::Split { dim, median, idx, left, right } => {
                // Check the split point itself
                let row = data.row(*idx);
                let dist = compute_distance(query, row.as_slice().unwrap(), metric);
                if heap.len() < k {
                    heap.push(DistIdx(dist, *idx));
                } else if let Some(top) = heap.peek() {
                    if dist < top.0 {
                        heap.pop();
                        heap.push(DistIdx(dist, *idx));
                    }
                }

                // Determine which side to search first (nearer side)
                let diff = query[*dim] - *median;
                let (near, far) = if diff <= 0.0 {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                // Always search near side
                self.search(near, data, query, k, metric, heap);

                // Search far side only if the splitting plane is closer than current worst
                let plane_dist = diff.abs();
                let should_search_far = heap.len() < k
                    || heap.peek().map_or(true, |top| plane_dist < top.0);

                if should_search_far {
                    self.search(far, data, query, k, metric, heap);
                }
            }
        }
    }
}

// ============================================================================
// Classifier and Regressor
// ============================================================================

/// K-Nearest Neighbors Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNClassifier {
    config: KNNConfig,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
    classes: Vec<i64>,
    /// KD-tree index (built for Euclidean/Manhattan metrics)
    kd_tree: Option<KdTree>,
}

impl KNNClassifier {
    pub fn new(config: KNNConfig) -> Self {
        Self {
            config,
            x_train: None,
            y_train: None,
            classes: Vec::new(),
            kd_tree: None,
        }
    }

    /// Create with default config and specified k
    pub fn with_k(k: usize) -> Self {
        Self::new(KNNConfig {
            n_neighbors: k,
            ..Default::default()
        })
    }

    /// Fit the classifier (stores training data, builds KD-tree for supported metrics)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());

        // Build KD-tree for metrics where plane-distance pruning is valid
        self.kd_tree = match self.config.metric {
            DistanceMetric::Euclidean | DistanceMetric::Manhattan => Some(KdTree::build(x)),
            _ => None,
        };

        // Find unique classes
        let mut class_set: Vec<i64> = y.iter()
            .map(|&v| v as i64)
            .collect();
        class_set.sort();
        class_set.dedup();
        self.classes = class_set;

        Ok(())
    }

    /// Predict class labels (parallelized over test samples)
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let x_train = self.x_train.as_ref().ok_or(crate::error::KolosalError::ModelNotFitted)?;
        let y_train = self.y_train.as_ref().ok_or(crate::error::KolosalError::ModelNotFitted)?;
        let k = self.config.n_neighbors;
        let metric = self.config.metric;
        let weights = self.config.weights;
        let classes = &self.classes;
        let kd_tree = self.kd_tree.as_ref();

        let predictions: Vec<f64> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x.row(i);
                let query = row.as_slice().unwrap();
                let neighbors = match kd_tree {
                    Some(tree) => tree.query_k_nearest(x_train, query, k, metric),
                    None => find_k_nearest_brute(query, x_train, k, metric),
                };
                vote_classify(&neighbors, y_train, classes, weights)
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    /// Predict class probabilities (parallelized)
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let x_train = self.x_train.as_ref().ok_or(crate::error::KolosalError::ModelNotFitted)?;
        let y_train = self.y_train.as_ref().ok_or(crate::error::KolosalError::ModelNotFitted)?;
        let n_classes = self.classes.len();
        let k = self.config.n_neighbors;
        let metric = self.config.metric;
        let weights = self.config.weights;
        let classes = &self.classes;
        let kd_tree = self.kd_tree.as_ref();

        let probs: Vec<Vec<f64>> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x.row(i);
                let query = row.as_slice().unwrap();
                let neighbors = match kd_tree {
                    Some(tree) => tree.query_k_nearest(x_train, query, k, metric),
                    None => find_k_nearest_brute(query, x_train, k, metric),
                };
                class_probs_from(&neighbors, y_train, classes, n_classes, weights)
            })
            .collect();

        let flat: Vec<f64> = probs.into_iter().flatten().collect();
        Ok(Array2::from_shape_vec((x.nrows(), n_classes), flat)?)
    }

    fn distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        compute_distance(a.as_slice().unwrap(), b.as_slice().unwrap(), self.config.metric)
    }
}

/// K-Nearest Neighbors Regressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNRegressor {
    config: KNNConfig,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
    kd_tree: Option<KdTree>,
}

impl KNNRegressor {
    pub fn new(config: KNNConfig) -> Self {
        Self {
            config,
            x_train: None,
            y_train: None,
            kd_tree: None,
        }
    }

    /// Create with default config and specified k
    pub fn with_k(k: usize) -> Self {
        Self::new(KNNConfig {
            n_neighbors: k,
            ..Default::default()
        })
    }

    /// Fit the regressor (stores training data, builds KD-tree)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());

        self.kd_tree = match self.config.metric {
            DistanceMetric::Euclidean | DistanceMetric::Manhattan => Some(KdTree::build(x)),
            _ => None,
        };

        Ok(())
    }

    /// Predict target values (parallelized over test samples)
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let x_train = self.x_train.as_ref().ok_or(crate::error::KolosalError::ModelNotFitted)?;
        let y_train = self.y_train.as_ref().ok_or(crate::error::KolosalError::ModelNotFitted)?;
        let k = self.config.n_neighbors;
        let metric = self.config.metric;
        let weights = self.config.weights;
        let kd_tree = self.kd_tree.as_ref();

        let predictions: Vec<f64> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x.row(i);
                let query = row.as_slice().unwrap();
                let neighbors = match kd_tree {
                    Some(tree) => tree.query_k_nearest(x_train, query, k, metric),
                    None => find_k_nearest_brute(query, x_train, k, metric),
                };
                weighted_mean_from(&neighbors, y_train, weights)
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

// ============================================================================
// Shared optimized helpers
// ============================================================================

/// Max-heap entry for partial sort (keeps k smallest distances).
#[derive(PartialEq)]
struct DistIdx(f64, usize);

impl Eq for DistIdx {}
impl PartialOrd for DistIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for DistIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Brute-force k nearest neighbors — O(n log k). Fallback for non-tree-compatible metrics.
fn find_k_nearest_brute(
    point: &[f64],
    x_train: &Array2<f64>,
    k: usize,
    metric: DistanceMetric,
) -> Vec<(f64, usize)> {
    let mut heap = BinaryHeap::with_capacity(k + 1);

    for (i, row) in x_train.rows().into_iter().enumerate() {
        let dist = compute_distance(point, row.as_slice().unwrap(), metric);
        if heap.len() < k {
            heap.push(DistIdx(dist, i));
        } else if let Some(top) = heap.peek() {
            if dist < top.0 {
                heap.pop();
                heap.push(DistIdx(dist, i));
            }
        }
    }

    heap.into_iter().map(|dl| (dl.0, dl.1)).collect()
}

/// Compute distance between two points using the specified metric
#[inline]
fn compute_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => {
            a.iter()
                .zip(b.iter())
                .map(|(ai, bi)| {
                    let d = ai - bi;
                    d * d
                })
                .sum::<f64>()
                .sqrt()
        }
        DistanceMetric::Manhattan => {
            a.iter()
                .zip(b.iter())
                .map(|(ai, bi)| (ai - bi).abs())
                .sum()
        }
        DistanceMetric::Minkowski(p) => {
            a.iter()
                .zip(b.iter())
                .map(|(ai, bi)| (ai - bi).abs().powf(p))
                .sum::<f64>()
                .powf(1.0 / p)
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0;
            let mut norm_a = 0.0;
            let mut norm_b = 0.0;
            for (ai, bi) in a.iter().zip(b.iter()) {
                dot += ai * bi;
                norm_a += ai * ai;
                norm_b += bi * bi;
            }
            let denom = norm_a.sqrt() * norm_b.sqrt();
            if denom > 0.0 { 1.0 - (dot / denom) } else { 1.0 }
        }
    }
}

/// Classify by weighted majority vote using Vec indexed by class position
fn vote_classify(neighbors: &[(f64, usize)], y_train: &Array1<f64>, classes: &[i64], weights: WeightScheme) -> f64 {
    let mut votes = vec![0.0f64; classes.len()];
    for &(dist, idx) in neighbors {
        let label = y_train[idx] as i64;
        let weight = match weights {
            WeightScheme::Uniform => 1.0,
            WeightScheme::Distance => 1.0 / (dist + 1e-10),
        };
        if let Some(pos) = classes.iter().position(|&c| c == label) {
            votes[pos] += weight;
        }
    }
    let mut best_pos = 0;
    let mut best_weight = votes[0];
    for (i, &w) in votes.iter().enumerate().skip(1) {
        if w > best_weight {
            best_weight = w;
            best_pos = i;
        }
    }
    classes[best_pos] as f64
}

/// Compute class probabilities
fn class_probs_from(neighbors: &[(f64, usize)], y_train: &Array1<f64>, classes: &[i64], n_classes: usize, weights: WeightScheme) -> Vec<f64> {
    let mut counts = vec![0.0; n_classes];
    let mut total = 0.0;
    for &(dist, idx) in neighbors {
        let label = y_train[idx] as i64;
        let weight = match weights {
            WeightScheme::Uniform => 1.0,
            WeightScheme::Distance => 1.0 / (dist + 1e-10),
        };
        let class_idx = classes.iter().position(|&c| c == label).unwrap_or(0);
        counts[class_idx] += weight;
        total += weight;
    }
    if total > 0.0 {
        counts.iter_mut().for_each(|c| *c /= total);
    }
    counts
}

/// Compute weighted mean for regression
fn weighted_mean_from(neighbors: &[(f64, usize)], y_train: &Array1<f64>, weights: WeightScheme) -> f64 {
    match weights {
        WeightScheme::Uniform => {
            neighbors.iter().map(|&(_, idx)| y_train[idx]).sum::<f64>() / neighbors.len() as f64
        }
        WeightScheme::Distance => {
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;
            for &(dist, idx) in neighbors {
                let w = 1.0 / (dist + 1e-10);
                weighted_sum += w * y_train[idx];
                weight_total += w;
            }
            if weight_total > 0.0 { weighted_sum / weight_total }
            else { neighbors.iter().map(|&(_, idx)| y_train[idx]).sum::<f64>() / neighbors.len() as f64 }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((20, 2), vec![
            1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 1.0, 2.0,
            1.5, 2.5, 2.0, 1.5, 2.5, 1.0, 1.2, 1.8, 1.8, 1.2,
            8.0, 8.0, 8.5, 8.5, 9.0, 9.0, 9.5, 9.5, 8.0, 9.0,
            8.5, 9.5, 9.0, 8.5, 9.5, 8.0, 8.2, 8.8, 8.8, 8.2,
        ]).unwrap();

        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);

        (x, y)
    }

    fn create_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((10, 2),
            (0..20).map(|i| i as f64).collect()
        ).unwrap();

        let y: Array1<f64> = x.rows().into_iter()
            .map(|row| row[0] + row[1])
            .collect();

        (x, y)
    }

    #[test]
    fn test_knn_classifier() {
        let (x, y) = create_classification_data();

        let mut knn = KNNClassifier::with_k(3);
        knn.fit(&x, &y).unwrap();

        let predictions = knn.predict(&x).unwrap();

        let correct: usize = y.iter()
            .zip(predictions.iter())
            .filter(|(&yi, &pi)| (yi - pi).abs() < 0.5)
            .count();

        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.9, "Accuracy ({}) should be above 90%", accuracy);
    }

    #[test]
    fn test_knn_regressor() {
        let (x, y) = create_regression_data();

        let mut knn = KNNRegressor::with_k(3);
        knn.fit(&x, &y).unwrap();

        let predictions = knn.predict(&x).unwrap();

        let mse: f64 = y.iter()
            .zip(predictions.iter())
            .map(|(yi, pi)| (yi - pi).powi(2))
            .sum::<f64>() / y.len() as f64;

        assert!(mse < 10.0, "MSE ({}) should be low", mse);
    }

    #[test]
    fn test_distance_metrics() {
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);

        let knn = KNNClassifier::new(KNNConfig {
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        });

        let dist = knn.distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001, "Euclidean distance should be 5.0");
    }

    #[test]
    fn test_weighted_knn() {
        let (x, y) = create_classification_data();

        let mut knn = KNNClassifier::new(KNNConfig {
            n_neighbors: 5,
            weights: WeightScheme::Distance,
            ..Default::default()
        });
        knn.fit(&x, &y).unwrap();

        let predictions = knn.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);
    }

    #[test]
    fn test_kd_tree_vs_brute_force() {
        let (x, y) = create_classification_data();

        // KD-tree (Euclidean)
        let mut knn_tree = KNNClassifier::with_k(3);
        knn_tree.fit(&x, &y).unwrap();
        let pred_tree = knn_tree.predict(&x).unwrap();

        // Brute force (Cosine — no KD-tree)
        let mut knn_brute = KNNClassifier::new(KNNConfig {
            n_neighbors: 3,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        });
        knn_brute.fit(&x, &y).unwrap();
        let pred_brute = knn_brute.predict(&x).unwrap();

        // Both should produce reasonable predictions
        assert_eq!(pred_tree.len(), 20);
        assert_eq!(pred_brute.len(), 20);
    }

    #[test]
    fn test_kd_tree_single_point() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);

        let mut knn = KNNClassifier::with_k(1);
        knn.fit(&x, &y).unwrap();

        let test = Array2::from_shape_vec((1, 2), vec![1.1, 1.1]).unwrap();
        let pred = knn.predict(&test).unwrap();
        assert_eq!(pred[0], 0.0); // Closest to (1,1)
    }
}
