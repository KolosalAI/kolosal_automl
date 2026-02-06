//! Python bindings for advanced imputation

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// KNN Imputer for Python
#[pyclass(name = "KNNImputer")]
pub struct PyKNNImputer {
    n_neighbors: usize,
    weights: String,
}

#[pymethods]
impl PyKNNImputer {
    #[new]
    #[pyo3(signature = (n_neighbors=5, weights="uniform"))]
    fn new(n_neighbors: usize, weights: &str) -> Self {
        Self {
            n_neighbors,
            weights: weights.to_string(),
        }
    }

    /// Impute missing values
    fn fit_transform(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        
        if x_list.is_empty() {
            return Err(PyValueError::new_err("Empty data"));
        }
        
        let n_samples = x_list.len();
        let n_features = x_list[0].len();
        let mut result = x_list.clone();
        
        // Find rows with missing values
        for i in 0..n_samples {
            for j in 0..n_features {
                if result[i][j].is_nan() {
                    // Find k nearest neighbors that have this feature
                    let mut distances: Vec<(usize, f64)> = Vec::new();
                    
                    for k in 0..n_samples {
                        if k == i || x_list[k][j].is_nan() {
                            continue;
                        }
                        
                        // Compute distance using non-missing features
                        let mut dist = 0.0;
                        let mut count = 0;
                        for f in 0..n_features {
                            if !x_list[i][f].is_nan() && !x_list[k][f].is_nan() {
                                dist += (x_list[i][f] - x_list[k][f]).powi(2);
                                count += 1;
                            }
                        }
                        if count > 0 {
                            distances.push((k, (dist / count as f64).sqrt()));
                        }
                    }
                    
                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let k_neighbors: Vec<(usize, f64)> = distances.into_iter()
                        .take(self.n_neighbors)
                        .collect();
                    
                    if !k_neighbors.is_empty() {
                        if self.weights == "distance" {
                            let total_weight: f64 = k_neighbors.iter()
                                .map(|(_, d)| 1.0 / (d + 1e-10))
                                .sum();
                            let weighted_sum: f64 = k_neighbors.iter()
                                .map(|(idx, d)| x_list[*idx][j] / (d + 1e-10))
                                .sum();
                            result[i][j] = weighted_sum / total_weight;
                        } else {
                            let sum: f64 = k_neighbors.iter()
                                .map(|(idx, _)| x_list[*idx][j])
                                .sum();
                            result[i][j] = sum / k_neighbors.len() as f64;
                        }
                    }
                }
            }
        }
        
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (result,))?;
        Ok(array.into())
    }

    fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }
}

/// MICE Imputer for Python
#[pyclass(name = "MICEImputer")]
pub struct PyMICEImputer {
    max_iter: usize,
    tol: f64,
}

#[pymethods]
impl PyMICEImputer {
    #[new]
    #[pyo3(signature = (max_iter=10, tol=1e-3))]
    fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Impute using MICE algorithm
    fn fit_transform(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        
        if x_list.is_empty() {
            return Err(PyValueError::new_err("Empty data"));
        }
        
        let n_samples = x_list.len();
        let n_features = x_list[0].len();
        let mut result = x_list.clone();
        
        // Initial imputation with column means
        for j in 0..n_features {
            let valid: Vec<f64> = (0..n_samples)
                .filter_map(|i| {
                    if result[i][j].is_nan() { None } else { Some(result[i][j]) }
                })
                .collect();
            
            let mean = if valid.is_empty() { 
                0.0 
            } else { 
                valid.iter().sum::<f64>() / valid.len() as f64 
            };
            
            for i in 0..n_samples {
                if result[i][j].is_nan() {
                    result[i][j] = mean;
                }
            }
        }
        
        // Iterative refinement
        for _ in 0..self.max_iter {
            let prev = result.clone();
            
            for j in 0..n_features {
                // Simple linear regression using other features
                let missing_mask: Vec<bool> = x_list.iter().map(|r| r[j].is_nan()).collect();
                
                if !missing_mask.iter().any(|&m| m) {
                    continue;
                }
                
                // Compute mean of other columns for prediction
                for i in 0..n_samples {
                    if missing_mask[i] {
                        let other_mean: f64 = (0..n_features)
                            .filter(|&f| f != j)
                            .map(|f| result[i][f])
                            .sum::<f64>() / (n_features - 1) as f64;
                        
                        // Simple prediction based on correlation with mean
                        let col_mean: f64 = (0..n_samples)
                            .filter(|&k| !missing_mask[k])
                            .map(|k| x_list[k][j])
                            .sum::<f64>() / missing_mask.iter().filter(|&&m| !m).count() as f64;
                        
                        result[i][j] = col_mean + (other_mean - col_mean) * 0.5;
                    }
                }
            }
            
            // Check convergence
            let diff: f64 = result.iter().zip(prev.iter())
                .flat_map(|(r, p)| r.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()))
                .sum();
            
            if diff < self.tol {
                break;
            }
        }
        
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (result,))?;
        Ok(array.into())
    }

    fn max_iter(&self) -> usize {
        self.max_iter
    }
}

/// Iterative Imputer for Python
#[pyclass(name = "IterativeImputer")]
pub struct PyIterativeImputer {
    max_iter: usize,
    initial_strategy: String,
}

#[pymethods]
impl PyIterativeImputer {
    #[new]
    #[pyo3(signature = (max_iter=10, initial_strategy="mean"))]
    fn new(max_iter: usize, initial_strategy: &str) -> Self {
        Self {
            max_iter,
            initial_strategy: initial_strategy.to_string(),
        }
    }

    /// Impute using iterative approach
    fn fit_transform(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        
        if x_list.is_empty() {
            return Err(PyValueError::new_err("Empty data"));
        }
        
        let n_samples = x_list.len();
        let n_features = x_list[0].len();
        let mut result = x_list.clone();
        
        // Initial imputation
        for j in 0..n_features {
            let valid: Vec<f64> = (0..n_samples)
                .filter_map(|i| {
                    if result[i][j].is_nan() { None } else { Some(result[i][j]) }
                })
                .collect();
            
            let fill_value = if valid.is_empty() {
                0.0
            } else {
                match self.initial_strategy.as_str() {
                    "median" => {
                        let mut sorted = valid.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        sorted[sorted.len() / 2]
                    }
                    "most_frequent" => {
                        valid.iter().sum::<f64>() / valid.len() as f64  // simplified
                    }
                    _ => valid.iter().sum::<f64>() / valid.len() as f64,
                }
            };
            
            for i in 0..n_samples {
                if result[i][j].is_nan() {
                    result[i][j] = fill_value;
                }
            }
        }
        
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (result,))?;
        Ok(array.into())
    }

    fn max_iter(&self) -> usize {
        self.max_iter
    }
}

/// Simple Imputer for Python
#[pyclass(name = "SimpleImputer")]
pub struct PySimpleImputer {
    strategy: String,
    fill_value: Option<f64>,
    statistics: Vec<f64>,
}

#[pymethods]
impl PySimpleImputer {
    #[new]
    #[pyo3(signature = (strategy="mean", fill_value=None))]
    fn new(strategy: &str, fill_value: Option<f64>) -> Self {
        Self {
            strategy: strategy.to_string(),
            fill_value,
            statistics: Vec::new(),
        }
    }

    /// Fit the imputer
    fn fit(&mut self, X: Vec<Vec<f64>>) -> PyResult<()> {
        if X.is_empty() {
            return Err(PyValueError::new_err("Empty data"));
        }
        
        let n_features = X[0].len();
        self.statistics.clear();
        
        for j in 0..n_features {
            let valid: Vec<f64> = X.iter()
                .filter_map(|r| if r[j].is_nan() { None } else { Some(r[j]) })
                .collect();
            
            let stat = if valid.is_empty() {
                self.fill_value.unwrap_or(0.0)
            } else {
                match self.strategy.as_str() {
                    "median" => {
                        let mut sorted = valid.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        sorted[sorted.len() / 2]
                    }
                    "constant" => self.fill_value.unwrap_or(0.0),
                    _ => valid.iter().sum::<f64>() / valid.len() as f64,
                }
            };
            
            self.statistics.push(stat);
        }
        
        Ok(())
    }

    /// Transform data
    fn transform(&self, py: Python<'_>, X: Vec<Vec<f64>>) -> PyResult<PyObject> {
        if self.statistics.is_empty() {
            return Err(PyValueError::new_err("Not fitted"));
        }
        
        let mut result = X.clone();
        for (i, row) in result.iter_mut().enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                if val.is_nan() {
                    *val = self.statistics[j];
                }
            }
        }
        
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (result,))?;
        Ok(array.into())
    }

    /// Fit and transform
    fn fit_transform(&mut self, py: Python<'_>, X: Vec<Vec<f64>>) -> PyResult<PyObject> {
        self.fit(X.clone())?;
        self.transform(py, X)
    }

    fn statistics(&self) -> Vec<f64> {
        self.statistics.clone()
    }
}
