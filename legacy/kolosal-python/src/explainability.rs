//! Python bindings for model explainability

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use ndarray::{Array1, Array2};

/// Permutation importance result
#[pyclass(name = "ImportanceResult")]
pub struct PyImportanceResult {
    importances: Vec<f64>,
    importances_std: Vec<f64>,
    feature_names: Vec<String>,
}

#[pymethods]
impl PyImportanceResult {
    /// Get importance scores
    fn importances(&self) -> Vec<f64> {
        self.importances.clone()
    }

    /// Get importance standard deviations
    fn importances_std(&self) -> Vec<f64> {
        self.importances_std.clone()
    }

    /// Get feature names
    fn feature_names(&self) -> Vec<String> {
        self.feature_names.clone()
    }

    /// Get sorted indices (most important first)
    fn sorted_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.importances.len()).collect();
        indices.sort_by(|&a, &b| {
            self.importances[b].partial_cmp(&self.importances[a]).unwrap()
        });
        indices
    }

    /// Get top k features
    fn top_k(&self, k: usize) -> Vec<(String, f64)> {
        let indices = self.sorted_indices();
        indices.into_iter()
            .take(k)
            .map(|i| (self.feature_names.get(i).cloned().unwrap_or_default(), self.importances[i]))
            .collect()
    }
}

/// Permutation importance calculator
#[pyclass(name = "PermutationImportance")]
pub struct PyPermutationImportance {
    n_repeats: usize,
    seed: Option<u64>,
    feature_names: Vec<String>,
}

#[pymethods]
impl PyPermutationImportance {
    #[new]
    #[pyo3(signature = (n_repeats=5, random_state=None))]
    fn new(n_repeats: usize, random_state: Option<u64>) -> Self {
        Self {
            n_repeats,
            seed: random_state,
            feature_names: Vec::new(),
        }
    }

    /// Set feature names
    fn with_feature_names(&mut self, names: Vec<String>) {
        self.feature_names = names;
    }

    /// Compute feature importances using a model's predict function
    fn compute(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        X: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<PyImportanceResult> {
        use rand::prelude::*;
        
        // Convert inputs
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        let y_list: Vec<f64> = y.call_method0("tolist")?.extract()?;
        
        let n_samples = x_list.len();
        let n_features = x_list.first().map(|r| r.len()).unwrap_or(0);
        
        // Get baseline score
        let baseline_pred = model.call_method1("predict", (X,))?;
        let baseline_score = self.compute_score(py, &baseline_pred, y)?;
        
        let mut importances = vec![0.0; n_features];
        let mut importances_std = vec![0.0; n_features];
        
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        // For each feature, permute and measure score drop
        for feat_idx in 0..n_features {
            let mut scores = Vec::with_capacity(self.n_repeats);
            
            for _ in 0..self.n_repeats {
                // Create permuted copy
                let mut x_perm = x_list.clone();
                let mut perm_indices: Vec<usize> = (0..n_samples).collect();
                perm_indices.shuffle(&mut rng);
                
                for (i, &perm_i) in perm_indices.iter().enumerate() {
                    x_perm[i][feat_idx] = x_list[perm_i][feat_idx];
                }
                
                // Convert to numpy and predict
                let numpy = py.import("numpy")?;
                let x_perm_arr = numpy.call_method1("array", (x_perm,))?;
                let perm_pred = model.call_method1("predict", (&x_perm_arr,))?;
                let perm_score = self.compute_score(py, &perm_pred, y)?;
                
                scores.push(baseline_score - perm_score);
            }
            
            importances[feat_idx] = scores.iter().sum::<f64>() / scores.len() as f64;
            let mean = importances[feat_idx];
            importances_std[feat_idx] = (scores.iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f64>() / scores.len() as f64).sqrt();
        }
        
        let feature_names = if self.feature_names.is_empty() {
            (0..n_features).map(|i| format!("feature_{}", i)).collect()
        } else {
            self.feature_names.clone()
        };
        
        Ok(PyImportanceResult {
            importances,
            importances_std,
            feature_names,
        })
    }
}

impl PyPermutationImportance {
    fn compute_score(&self, py: Python<'_>, y_pred: &Bound<'_, PyAny>, y_true: &Bound<'_, PyAny>) -> PyResult<f64> {
        // Simple MSE-based score (negative so higher is better)
        let pred: Vec<f64> = y_pred.call_method0("tolist")?.extract()?;
        let true_vals: Vec<f64> = y_true.call_method0("tolist")?.extract()?;
        
        let mse: f64 = pred.iter().zip(true_vals.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / pred.len() as f64;
        
        Ok(-mse) // Negative so that lower error = higher score
    }
}

/// Partial dependence result
#[pyclass(name = "PDPResult")]
pub struct PyPDPResult {
    grid: Vec<f64>,
    values: Vec<f64>,
    feature_idx: usize,
}

#[pymethods]
impl PyPDPResult {
    fn grid(&self) -> Vec<f64> {
        self.grid.clone()
    }
    
    fn values(&self) -> Vec<f64> {
        self.values.clone()
    }
    
    fn feature_idx(&self) -> usize {
        self.feature_idx
    }
}

/// Partial dependence calculator
#[pyclass(name = "PartialDependence")]
pub struct PyPartialDependence {
    grid_resolution: usize,
}

#[pymethods]
impl PyPartialDependence {
    #[new]
    #[pyo3(signature = (grid_resolution=100))]
    fn new(grid_resolution: usize) -> Self {
        Self { grid_resolution }
    }

    /// Compute partial dependence for a feature
    fn compute(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        X: &Bound<'_, PyAny>,
        feature: usize,
    ) -> PyResult<PyPDPResult> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        let n_samples = x_list.len();
        
        // Get feature values and create grid
        let feature_vals: Vec<f64> = x_list.iter().map(|r| r[feature]).collect();
        let min_val = feature_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = feature_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        let step = (max_val - min_val) / (self.grid_resolution - 1) as f64;
        let grid: Vec<f64> = (0..self.grid_resolution)
            .map(|i| min_val + i as f64 * step)
            .collect();
        
        let numpy = py.import("numpy")?;
        let mut pdp_values = Vec::with_capacity(self.grid_resolution);
        
        for grid_val in &grid {
            // Replace feature with grid value for all samples
            let mut x_mod = x_list.clone();
            for row in &mut x_mod {
                row[feature] = *grid_val;
            }
            
            let x_arr = numpy.call_method1("array", (x_mod,))?;
            let pred = model.call_method1("predict", (&x_arr,))?;
            let pred_vals: Vec<f64> = pred.call_method0("tolist")?.extract()?;
            
            // Average prediction
            let avg = pred_vals.iter().sum::<f64>() / pred_vals.len() as f64;
            pdp_values.push(avg);
        }
        
        Ok(PyPDPResult {
            grid,
            values: pdp_values,
            feature_idx: feature,
        })
    }
}

/// Local explanation result
#[pyclass(name = "LocalExplanation")]
pub struct PyLocalExplanation {
    shap_values: Vec<f64>,
    expected_value: f64,
    prediction: f64,
    feature_names: Vec<String>,
}

#[pymethods]
impl PyLocalExplanation {
    fn shap_values(&self) -> Vec<f64> {
        self.shap_values.clone()
    }
    
    fn expected_value(&self) -> f64 {
        self.expected_value
    }
    
    fn prediction(&self) -> f64 {
        self.prediction
    }
    
    fn feature_names(&self) -> Vec<String> {
        self.feature_names.clone()
    }
    
    /// Get feature contributions sorted by absolute value
    fn sorted_contributions(&self) -> Vec<(String, f64)> {
        let mut contrib: Vec<(String, f64)> = self.feature_names.iter()
            .zip(self.shap_values.iter())
            .map(|(n, v)| (n.clone(), *v))
            .collect();
        contrib.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        contrib
    }
}

/// Local explainer (SHAP-like approximation)
#[pyclass(name = "LocalExplainer")]
pub struct PyLocalExplainer {
    n_samples: usize,
    seed: Option<u64>,
}

#[pymethods]
impl PyLocalExplainer {
    #[new]
    #[pyo3(signature = (n_samples=100, random_state=None))]
    fn new(n_samples: usize, random_state: Option<u64>) -> Self {
        Self { n_samples, seed: random_state }
    }

    /// Explain a single prediction using sampling-based approximation
    fn explain(
        &self,
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        X: &Bound<'_, PyAny>,
        instance_idx: usize,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<PyLocalExplanation> {
        use rand::prelude::*;
        
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        let n_features = x_list.first().map(|r| r.len()).unwrap_or(0);
        let instance = &x_list[instance_idx];
        
        let numpy = py.import("numpy")?;
        
        // Get prediction for this instance
        let instance_arr = numpy.call_method1("array", (vec![instance.clone()],))?;
        let pred = model.call_method1("predict", (&instance_arr,))?;
        let pred_vals: Vec<f64> = pred.call_method0("tolist")?.extract()?;
        let prediction = pred_vals[0];
        
        // Get baseline (average) prediction
        let baseline_pred = model.call_method1("predict", (X,))?;
        let baseline_vals: Vec<f64> = baseline_pred.call_method0("tolist")?.extract()?;
        let expected_value = baseline_vals.iter().sum::<f64>() / baseline_vals.len() as f64;
        
        // Simple feature attribution by masking
        let mut shap_values = vec![0.0; n_features];
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        for feat_idx in 0..n_features {
            let mut contributions = Vec::new();
            
            for _ in 0..self.n_samples {
                // Sample a random instance as baseline
                let baseline_idx = rng.gen_range(0..x_list.len());
                let baseline = &x_list[baseline_idx];
                
                // Create masked versions
                let mut with_feature = baseline.clone();
                with_feature[feat_idx] = instance[feat_idx];
                
                let arr_with = numpy.call_method1("array", (vec![with_feature],))?;
                let arr_without = numpy.call_method1("array", (vec![baseline.clone()],))?;
                
                let pred_with: Vec<f64> = model.call_method1("predict", (&arr_with,))?
                    .call_method0("tolist")?.extract()?;
                let pred_without: Vec<f64> = model.call_method1("predict", (&arr_without,))?
                    .call_method0("tolist")?.extract()?;
                
                contributions.push(pred_with[0] - pred_without[0]);
            }
            
            shap_values[feat_idx] = contributions.iter().sum::<f64>() / contributions.len() as f64;
        }
        
        let names = feature_names.unwrap_or_else(|| {
            (0..n_features).map(|i| format!("feature_{}", i)).collect()
        });
        
        Ok(PyLocalExplanation {
            shap_values,
            expected_value,
            prediction,
            feature_names: names,
        })
    }
}
