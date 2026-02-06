//! K-Nearest Neighbors implementation
//!
//! KNN classifier and regressor with distance metrics.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

/// K-Nearest Neighbors Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNClassifier {
    config: KNNConfig,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
    classes: Vec<i64>,
}

impl KNNClassifier {
    pub fn new(config: KNNConfig) -> Self {
        Self {
            config,
            x_train: None,
            y_train: None,
            classes: Vec::new(),
        }
    }

    /// Create with default config and specified k
    pub fn with_k(k: usize) -> Self {
        Self::new(KNNConfig {
            n_neighbors: k,
            ..Default::default()
        })
    }

    /// Fit the classifier (stores training data)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        
        // Find unique classes
        let mut class_set: Vec<i64> = y.iter()
            .map(|&v| v as i64)
            .collect();
        class_set.sort();
        class_set.dedup();
        self.classes = class_set;
        
        Ok(())
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");
        
        x.rows().into_iter()
            .map(|row| {
                let neighbors = self.find_neighbors(&row.to_owned(), x_train, y_train);
                self.vote(&neighbors)
            })
            .collect()
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");
        let n_classes = self.classes.len();
        
        let probs: Vec<f64> = x.rows().into_iter()
            .flat_map(|row| {
                let neighbors = self.find_neighbors(&row.to_owned(), x_train, y_train);
                self.class_probs(&neighbors, n_classes)
            })
            .collect();
        
        Array2::from_shape_vec((x.nrows(), n_classes), probs).unwrap()
    }

    fn find_neighbors(
        &self,
        point: &Array1<f64>,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
    ) -> Vec<(f64, f64)> {
        // Compute all distances
        let mut distances: Vec<(f64, f64)> = x_train
            .rows()
            .into_iter()
            .zip(y_train.iter())
            .map(|(row, &label)| {
                let dist = self.distance(point, &row.to_owned());
                (dist, label)
            })
            .collect();
        
        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(self.config.n_neighbors);
        
        distances
    }

    fn vote(&self, neighbors: &[(f64, f64)]) -> f64 {
        let mut votes: HashMap<i64, f64> = HashMap::new();
        
        for &(dist, label) in neighbors {
            let weight = match self.config.weights {
                WeightScheme::Uniform => 1.0,
                WeightScheme::Distance => 1.0 / (dist + 1e-10),
            };
            *votes.entry(label as i64).or_insert(0.0) += weight;
        }
        
        votes.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, _)| label as f64)
            .unwrap_or(0.0)
    }

    fn class_probs(&self, neighbors: &[(f64, f64)], n_classes: usize) -> Vec<f64> {
        let mut counts = vec![0.0; n_classes];
        let mut total = 0.0;
        
        for &(dist, label) in neighbors {
            let weight = match self.config.weights {
                WeightScheme::Uniform => 1.0,
                WeightScheme::Distance => 1.0 / (dist + 1e-10),
            };
            let class_idx = self.classes.iter().position(|&c| c == label as i64).unwrap_or(0);
            counts[class_idx] += weight;
            total += weight;
        }
        
        if total > 0.0 {
            counts.iter_mut().for_each(|c| *c /= total);
        }
        
        counts
    }

    fn distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        match self.config.metric {
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(ai, bi)| (ai - bi).powi(2))
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
                let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
                let norm_a: f64 = a.iter().map(|ai| ai.powi(2)).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|bi| bi.powi(2)).sum::<f64>().sqrt();
                
                if norm_a > 0.0 && norm_b > 0.0 {
                    1.0 - (dot / (norm_a * norm_b))
                } else {
                    1.0
                }
            }
        }
    }
}

/// K-Nearest Neighbors Regressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNRegressor {
    config: KNNConfig,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
}

impl KNNRegressor {
    pub fn new(config: KNNConfig) -> Self {
        Self {
            config,
            x_train: None,
            y_train: None,
        }
    }

    /// Create with default config and specified k
    pub fn with_k(k: usize) -> Self {
        Self::new(KNNConfig {
            n_neighbors: k,
            ..Default::default()
        })
    }

    /// Fit the regressor (stores training data)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        Ok(())
    }

    /// Predict target values
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");
        
        x.rows().into_iter()
            .map(|row| {
                let neighbors = self.find_neighbors(&row.to_owned(), x_train, y_train);
                self.weighted_mean(&neighbors)
            })
            .collect()
    }

    fn find_neighbors(
        &self,
        point: &Array1<f64>,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
    ) -> Vec<(f64, f64)> {
        let mut distances: Vec<(f64, f64)> = x_train
            .rows()
            .into_iter()
            .zip(y_train.iter())
            .map(|(row, &label)| {
                let dist = self.distance(point, &row.to_owned());
                (dist, label)
            })
            .collect();
        
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(self.config.n_neighbors);
        
        distances
    }

    fn weighted_mean(&self, neighbors: &[(f64, f64)]) -> f64 {
        match self.config.weights {
            WeightScheme::Uniform => {
                let sum: f64 = neighbors.iter().map(|(_, y)| y).sum();
                sum / neighbors.len() as f64
            }
            WeightScheme::Distance => {
                let mut weighted_sum = 0.0;
                let mut weight_total = 0.0;
                
                for &(dist, y) in neighbors {
                    let weight = 1.0 / (dist + 1e-10);
                    weighted_sum += weight * y;
                    weight_total += weight;
                }
                
                if weight_total > 0.0 {
                    weighted_sum / weight_total
                } else {
                    neighbors.iter().map(|(_, y)| y).sum::<f64>() / neighbors.len() as f64
                }
            }
        }
    }

    fn distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        match self.config.metric {
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(ai, bi)| (ai - bi).powi(2))
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
                let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
                let norm_a: f64 = a.iter().map(|ai| ai.powi(2)).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|bi| bi.powi(2)).sum::<f64>().sqrt();
                
                if norm_a > 0.0 && norm_b > 0.0 {
                    1.0 - (dot / (norm_a * norm_b))
                } else {
                    1.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        // Create linearly separable data
        let x = Array2::from_shape_vec((20, 2), vec![
            // Class 0 (low values)
            1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 1.0, 2.0,
            1.5, 2.5, 2.0, 1.5, 2.5, 1.0, 1.2, 1.8, 1.8, 1.2,
            // Class 1 (high values)
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
        
        let predictions = knn.predict(&x);
        
        // Check accuracy (should be perfect for this separable data)
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
        
        let predictions = knn.predict(&x);
        
        // Check that predictions are reasonable
        let mse: f64 = y.iter()
            .zip(predictions.iter())
            .map(|(yi, pi)| (yi - pi).powi(2))
            .sum::<f64>() / y.len() as f64;
        
        // MSE should be low for this simple data
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
        
        let predictions = knn.predict(&x);
        assert_eq!(predictions.len(), 20);
    }
}
