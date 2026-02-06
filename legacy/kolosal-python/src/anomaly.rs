//! Python bindings for anomaly detection

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rand::prelude::*;

/// Isolation Forest for Python
#[pyclass(name = "IsolationForest")]
pub struct PyIsolationForest {
    n_estimators: usize,
    max_samples: Option<usize>,
    contamination: f64,
    seed: Option<u64>,
    trees: Vec<IsolationTree>,
    threshold: Option<f64>,
}

struct IsolationTree {
    // Simplified tree structure
    split_feature: Option<usize>,
    split_value: Option<f64>,
    left: Option<Box<IsolationTree>>,
    right: Option<Box<IsolationTree>>,
    size: usize,
}

impl IsolationTree {
    fn new_leaf(size: usize) -> Self {
        Self {
            split_feature: None,
            split_value: None,
            left: None,
            right: None,
            size,
        }
    }
}

#[pymethods]
impl PyIsolationForest {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_samples=None, contamination=0.1, random_state=None))]
    fn new(
        n_estimators: usize,
        max_samples: Option<usize>,
        contamination: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_estimators,
            max_samples,
            contamination,
            seed: random_state,
            trees: Vec::new(),
            threshold: None,
        }
    }

    /// Fit the model
    fn fit(&mut self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<()> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        let n_samples = x_list.len();
        let n_features = x_list.first().map(|r| r.len()).unwrap_or(0);
        
        let sample_size = self.max_samples.unwrap_or(n_samples.min(256));
        let max_depth = (sample_size as f64).log2().ceil() as usize;
        
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        self.trees.clear();
        
        for _ in 0..self.n_estimators {
            // Sample indices
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            let sample_indices: Vec<usize> = indices.into_iter().take(sample_size).collect();
            
            // Build tree
            let tree = self.build_tree(&x_list, &sample_indices, 0, max_depth, n_features, &mut rng);
            self.trees.push(tree);
        }
        
        // Compute threshold based on contamination
        let scores = self.compute_scores(&x_list);
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = ((1.0 - self.contamination) * n_samples as f64) as usize;
        self.threshold = Some(sorted_scores.get(threshold_idx).copied().unwrap_or(0.5));
        
        Ok(())
    }

    /// Predict anomaly labels (-1 for outliers, 1 for inliers)
    fn predict(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        let scores = self.compute_scores(&x_list);
        let threshold = self.threshold.unwrap_or(0.5);
        
        Ok(scores.into_iter()
            .map(|s| if s > threshold { -1 } else { 1 })
            .collect())
    }

    /// Compute anomaly scores (higher = more anomalous)
    fn decision_function(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        Ok(self.compute_scores(&x_list))
    }

    /// Fit and predict
    fn fit_predict(&mut self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
        self.fit(py, X)?;
        self.predict(py, X)
    }

    /// Get contamination parameter
    fn contamination(&self) -> f64 {
        self.contamination
    }

    /// Get number of estimators
    fn n_estimators(&self) -> usize {
        self.n_estimators
    }
}

impl PyIsolationForest {
    fn build_tree(
        &self,
        data: &[Vec<f64>],
        indices: &[usize],
        depth: usize,
        max_depth: usize,
        n_features: usize,
        rng: &mut StdRng,
    ) -> IsolationTree {
        if depth >= max_depth || indices.len() <= 1 {
            return IsolationTree::new_leaf(indices.len());
        }
        
        // Random feature and split
        let feature = rng.gen_range(0..n_features);
        let values: Vec<f64> = indices.iter().map(|&i| data[i][feature]).collect();
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        if (max_val - min_val).abs() < 1e-10 {
            return IsolationTree::new_leaf(indices.len());
        }
        
        let split_value = rng.gen_range(min_val..max_val);
        
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices.iter()
            .partition(|&&i| data[i][feature] < split_value);
        
        IsolationTree {
            split_feature: Some(feature),
            split_value: Some(split_value),
            left: Some(Box::new(self.build_tree(data, &left_indices, depth + 1, max_depth, n_features, rng))),
            right: Some(Box::new(self.build_tree(data, &right_indices, depth + 1, max_depth, n_features, rng))),
            size: indices.len(),
        }
    }
    
    fn path_length(&self, point: &[f64], tree: &IsolationTree, depth: usize) -> f64 {
        if tree.split_feature.is_none() {
            return depth as f64 + self.c(tree.size);
        }
        
        let feature = tree.split_feature.unwrap();
        let split_value = tree.split_value.unwrap();
        
        if point[feature] < split_value {
            if let Some(ref left) = tree.left {
                return self.path_length(point, left, depth + 1);
            }
        } else {
            if let Some(ref right) = tree.right {
                return self.path_length(point, right, depth + 1);
            }
        }
        
        depth as f64
    }
    
    fn c(&self, n: usize) -> f64 {
        if n <= 1 {
            return 0.0;
        }
        2.0 * ((n as f64 - 1.0).ln() + 0.5772156649) - (2.0 * (n as f64 - 1.0) / n as f64)
    }
    
    fn compute_scores(&self, data: &[Vec<f64>]) -> Vec<f64> {
        let n_samples = self.max_samples.unwrap_or(256);
        let c_n = self.c(n_samples);
        
        data.iter().map(|point| {
            let avg_path_length: f64 = self.trees.iter()
                .map(|tree| self.path_length(point, tree, 0))
                .sum::<f64>() / self.trees.len() as f64;
            
            2.0_f64.powf(-avg_path_length / c_n)
        }).collect()
    }
}

/// Local Outlier Factor for Python
#[pyclass(name = "LocalOutlierFactor")]
pub struct PyLocalOutlierFactor {
    n_neighbors: usize,
    contamination: f64,
    lof_scores: Vec<f64>,
}

#[pymethods]
impl PyLocalOutlierFactor {
    #[new]
    #[pyo3(signature = (n_neighbors=20, contamination=0.1))]
    fn new(n_neighbors: usize, contamination: f64) -> Self {
        Self {
            n_neighbors,
            contamination,
            lof_scores: Vec::new(),
        }
    }

    /// Fit and predict (LOF is transductive)
    fn fit_predict(&mut self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        let n_samples = x_list.len();
        
        // Compute pairwise distances
        let mut distances: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut dists: Vec<(usize, f64)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = self.euclidean_distance(&x_list[i], &x_list[j]);
                    (j, dist)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.push(dists);
        }
        
        // Compute k-distance and reachability distance
        let k = self.n_neighbors.min(n_samples - 1);
        let k_distances: Vec<f64> = distances.iter()
            .map(|d| d.get(k - 1).map(|x| x.1).unwrap_or(0.0))
            .collect();
        
        // Compute local reachability density
        let mut lrd: Vec<f64> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let neighbors: Vec<usize> = distances[i].iter().take(k).map(|x| x.0).collect();
            let sum_reach_dist: f64 = neighbors.iter()
                .map(|&j| k_distances[j].max(distances[i].iter().find(|x| x.0 == j).unwrap().1))
                .sum();
            lrd.push(k as f64 / sum_reach_dist.max(1e-10));
        }
        
        // Compute LOF
        self.lof_scores = (0..n_samples).map(|i| {
            let neighbors: Vec<usize> = distances[i].iter().take(k).map(|x| x.0).collect();
            let avg_lrd_neighbors: f64 = neighbors.iter().map(|&j| lrd[j]).sum::<f64>() / k as f64;
            avg_lrd_neighbors / lrd[i].max(1e-10)
        }).collect();
        
        // Determine threshold
        let mut sorted_scores = self.lof_scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold_idx = (self.contamination * n_samples as f64) as usize;
        let threshold = sorted_scores.get(threshold_idx).copied().unwrap_or(1.0);
        
        Ok(self.lof_scores.iter()
            .map(|&s| if s > threshold { -1 } else { 1 })
            .collect())
    }

    /// Get local outlier factors
    fn negative_outlier_factor(&self) -> Vec<f64> {
        self.lof_scores.iter().map(|&x| -x).collect()
    }

    /// Get the number of neighbors used
    fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Get contamination
    fn contamination(&self) -> f64 {
        self.contamination
    }
}

impl PyLocalOutlierFactor {
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}
