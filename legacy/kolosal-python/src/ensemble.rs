//! Python bindings for ensemble methods

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

/// Voting strategy enum for Python
#[pyclass(name = "VotingType")]
#[derive(Clone)]
pub enum PyVotingType {
    Hard,
    Soft,
}

/// Voting classifier for Python
#[pyclass(name = "VotingClassifier")]
pub struct PyVotingClassifier {
    voting: PyVotingType,
    weights: Option<Vec<f64>>,
    models: Vec<(String, PyObject)>,
}

#[pymethods]
impl PyVotingClassifier {
    #[new]
    #[pyo3(signature = (voting=None, weights=None))]
    fn new(voting: Option<PyVotingType>, weights: Option<Vec<f64>>) -> Self {
        Self {
            voting: voting.unwrap_or(PyVotingType::Hard),
            weights,
            models: Vec::new(),
        }
    }

    /// Add a model to the ensemble
    fn add_model(&mut self, py: Python<'_>, name: String, model: PyObject) -> PyResult<()> {
        self.models.push((name, model));
        Ok(())
    }

    /// Fit all models (models should already be fitted)
    fn fit(&mut self, _py: Python<'_>, _X: &Bound<'_, PyAny>, _y: &Bound<'_, PyAny>) -> PyResult<()> {
        // Models are expected to be pre-fitted
        Ok(())
    }

    /// Predict using voting
    fn predict(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        if self.models.is_empty() {
            return Err(PyValueError::new_err("No models added to ensemble"));
        }
        
        let numpy = py.import("numpy")?;
        let mut all_predictions: Vec<Vec<f64>> = Vec::new();
        
        for (_, model) in &self.models {
            let pred = model.call_method1(py, "predict", (X,))?;
            let pred_list: Vec<f64> = pred.call_method0(py, "tolist")?.extract(py)?;
            all_predictions.push(pred_list);
        }
        
        let n_samples = all_predictions[0].len();
        let n_models = all_predictions.len();
        let weights = self.weights.clone().unwrap_or_else(|| vec![1.0; n_models]);
        
        let mut final_predictions = vec![0.0; n_samples];
        
        match self.voting {
            PyVotingType::Hard => {
                // Majority voting
                for i in 0..n_samples {
                    let mut votes: HashMap<i64, f64> = HashMap::new();
                    for (j, pred) in all_predictions.iter().enumerate() {
                        let class = pred[i] as i64;
                        *votes.entry(class).or_insert(0.0) += weights[j];
                    }
                    final_predictions[i] = *votes.iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap()
                        .0 as f64;
                }
            }
            PyVotingType::Soft => {
                // Weighted average
                for i in 0..n_samples {
                    let mut weighted_sum = 0.0;
                    let mut weight_sum = 0.0;
                    for (j, pred) in all_predictions.iter().enumerate() {
                        weighted_sum += pred[i] * weights[j];
                        weight_sum += weights[j];
                    }
                    final_predictions[i] = (weighted_sum / weight_sum).round();
                }
            }
        }
        
        Ok(final_predictions)
    }

    /// Get model names
    fn model_names(&self) -> Vec<String> {
        self.models.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Get number of models
    fn n_models(&self) -> usize {
        self.models.len()
    }
}

/// Voting regressor for Python
#[pyclass(name = "VotingRegressor")]
pub struct PyVotingRegressor {
    weights: Option<Vec<f64>>,
    models: Vec<(String, PyObject)>,
}

#[pymethods]
impl PyVotingRegressor {
    #[new]
    #[pyo3(signature = (weights=None))]
    fn new(weights: Option<Vec<f64>>) -> Self {
        Self {
            weights,
            models: Vec::new(),
        }
    }

    /// Add a model to the ensemble
    fn add_model(&mut self, py: Python<'_>, name: String, model: PyObject) -> PyResult<()> {
        self.models.push((name, model));
        Ok(())
    }

    /// Fit (models should already be fitted)
    fn fit(&mut self, _py: Python<'_>, _X: &Bound<'_, PyAny>, _y: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(())
    }

    /// Predict using averaging
    fn predict(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        if self.models.is_empty() {
            return Err(PyValueError::new_err("No models added to ensemble"));
        }
        
        let mut all_predictions: Vec<Vec<f64>> = Vec::new();
        
        for (_, model) in &self.models {
            let pred = model.call_method1(py, "predict", (X,))?;
            let pred_list: Vec<f64> = pred.call_method0(py, "tolist")?.extract(py)?;
            all_predictions.push(pred_list);
        }
        
        let n_samples = all_predictions[0].len();
        let n_models = all_predictions.len();
        let weights = self.weights.clone().unwrap_or_else(|| vec![1.0; n_models]);
        let weight_sum: f64 = weights.iter().sum();
        
        let mut final_predictions = vec![0.0; n_samples];
        
        for i in 0..n_samples {
            let mut weighted_sum = 0.0;
            for (j, pred) in all_predictions.iter().enumerate() {
                weighted_sum += pred[i] * weights[j];
            }
            final_predictions[i] = weighted_sum / weight_sum;
        }
        
        Ok(final_predictions)
    }

    /// Get model names
    fn model_names(&self) -> Vec<String> {
        self.models.iter().map(|(n, _)| n.clone()).collect()
    }
}

/// Stacking configuration for Python
#[pyclass(name = "StackingConfig")]
#[derive(Clone)]
pub struct PyStackingConfig {
    cv_folds: usize,
    use_probas: bool,
    passthrough: bool,
}

#[pymethods]
impl PyStackingConfig {
    #[new]
    fn new() -> Self {
        Self {
            cv_folds: 5,
            use_probas: false,
            passthrough: false,
        }
    }

    /// Set CV folds for stacking
    fn with_cv(&self, folds: usize) -> Self {
        Self { cv_folds: folds, ..*self }
    }

    /// Use probabilities as features (classification)
    fn with_use_probas(&self, use_probas: bool) -> Self {
        Self { use_probas, ..*self }
    }

    /// Passthrough original features
    fn with_passthrough(&self, passthrough: bool) -> Self {
        Self { passthrough, ..*self }
    }
}

/// Stacking classifier for Python
#[pyclass(name = "StackingClassifier")]
pub struct PyStackingClassifier {
    config: PyStackingConfig,
    base_models: Vec<(String, PyObject)>,
    meta_model: Option<PyObject>,
    is_fitted: bool,
}

#[pymethods]
impl PyStackingClassifier {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyStackingConfig>) -> Self {
        Self {
            config: config.unwrap_or_else(PyStackingConfig::new),
            base_models: Vec::new(),
            meta_model: None,
            is_fitted: false,
        }
    }

    /// Add a base model
    fn add_model(&mut self, py: Python<'_>, name: String, model: PyObject) -> PyResult<()> {
        self.base_models.push((name, model));
        Ok(())
    }

    /// Set the meta-learner
    fn set_meta_model(&mut self, py: Python<'_>, model: PyObject) -> PyResult<()> {
        self.meta_model = Some(model);
        Ok(())
    }

    /// Fit the stacking ensemble
    fn fit(&mut self, py: Python<'_>, X: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        // Simple implementation: fit all base models and collect predictions
        let numpy = py.import("numpy")?;
        
        // Fit base models
        for (_, model) in &self.base_models {
            model.call_method1(py, "fit", (X, y))?;
        }
        
        // Get base model predictions for meta-features
        let mut meta_features: Vec<Vec<f64>> = Vec::new();
        for (_, model) in &self.base_models {
            let pred = model.call_method1(py, "predict", (X,))?;
            let pred_list: Vec<f64> = pred.call_method0(py, "tolist")?.extract(py)?;
            meta_features.push(pred_list);
        }
        
        // Transpose to get samples x models
        let n_samples = meta_features[0].len();
        let mut meta_X: Vec<Vec<f64>> = vec![Vec::new(); n_samples];
        for preds in &meta_features {
            for (i, &p) in preds.iter().enumerate() {
                meta_X[i].push(p);
            }
        }
        
        // Fit meta-model
        if let Some(ref meta) = self.meta_model {
            let meta_arr = numpy.call_method1("array", (meta_X,))?;
            meta.call_method1(py, "fit", (&meta_arr, y))?;
        }
        
        self.is_fitted = true;
        Ok(())
    }

    /// Predict using stacking
    fn predict(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("Model not fitted"));
        }
        
        let numpy = py.import("numpy")?;
        
        // Get base model predictions
        let mut meta_features: Vec<Vec<f64>> = Vec::new();
        for (_, model) in &self.base_models {
            let pred = model.call_method1(py, "predict", (X,))?;
            let pred_list: Vec<f64> = pred.call_method0(py, "tolist")?.extract(py)?;
            meta_features.push(pred_list);
        }
        
        // Transpose
        let n_samples = meta_features[0].len();
        let mut meta_X: Vec<Vec<f64>> = vec![Vec::new(); n_samples];
        for preds in &meta_features {
            for (i, &p) in preds.iter().enumerate() {
                meta_X[i].push(p);
            }
        }
        
        // Predict with meta-model
        if let Some(ref meta) = self.meta_model {
            let meta_arr = numpy.call_method1("array", (meta_X,))?;
            let pred = meta.call_method1(py, "predict", (&meta_arr,))?;
            let pred_list: Vec<f64> = pred.call_method0(py, "tolist")?.extract(py)?;
            Ok(pred_list)
        } else {
            // Fall back to voting
            let weights = vec![1.0; self.base_models.len()];
            let mut final_predictions = vec![0.0; n_samples];
            for i in 0..n_samples {
                let avg: f64 = meta_X[i].iter().sum::<f64>() / meta_X[i].len() as f64;
                final_predictions[i] = avg.round();
            }
            Ok(final_predictions)
        }
    }
}

/// Stacking regressor for Python
#[pyclass(name = "StackingRegressor")]
pub struct PyStackingRegressor {
    config: PyStackingConfig,
    base_models: Vec<(String, PyObject)>,
    meta_model: Option<PyObject>,
    is_fitted: bool,
}

#[pymethods]
impl PyStackingRegressor {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyStackingConfig>) -> Self {
        Self {
            config: config.unwrap_or_else(PyStackingConfig::new),
            base_models: Vec::new(),
            meta_model: None,
            is_fitted: false,
        }
    }

    /// Add a base model
    fn add_model(&mut self, py: Python<'_>, name: String, model: PyObject) -> PyResult<()> {
        self.base_models.push((name, model));
        Ok(())
    }

    /// Set the meta-learner
    fn set_meta_model(&mut self, py: Python<'_>, model: PyObject) -> PyResult<()> {
        self.meta_model = Some(model);
        Ok(())
    }

    /// Fit the stacking ensemble
    fn fit(&mut self, py: Python<'_>, X: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let numpy = py.import("numpy")?;
        
        for (_, model) in &self.base_models {
            model.call_method1(py, "fit", (X, y))?;
        }
        
        let mut meta_features: Vec<Vec<f64>> = Vec::new();
        for (_, model) in &self.base_models {
            let pred = model.call_method1(py, "predict", (X,))?;
            let pred_list: Vec<f64> = pred.call_method0(py, "tolist")?.extract(py)?;
            meta_features.push(pred_list);
        }
        
        let n_samples = meta_features[0].len();
        let mut meta_X: Vec<Vec<f64>> = vec![Vec::new(); n_samples];
        for preds in &meta_features {
            for (i, &p) in preds.iter().enumerate() {
                meta_X[i].push(p);
            }
        }
        
        if let Some(ref meta) = self.meta_model {
            let meta_arr = numpy.call_method1("array", (meta_X,))?;
            meta.call_method1(py, "fit", (&meta_arr, y))?;
        }
        
        self.is_fitted = true;
        Ok(())
    }

    /// Predict using stacking
    fn predict(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("Model not fitted"));
        }
        
        let numpy = py.import("numpy")?;
        
        let mut meta_features: Vec<Vec<f64>> = Vec::new();
        for (_, model) in &self.base_models {
            let pred = model.call_method1(py, "predict", (X,))?;
            let pred_list: Vec<f64> = pred.call_method0(py, "tolist")?.extract(py)?;
            meta_features.push(pred_list);
        }
        
        let n_samples = meta_features[0].len();
        let mut meta_X: Vec<Vec<f64>> = vec![Vec::new(); n_samples];
        for preds in &meta_features {
            for (i, &p) in preds.iter().enumerate() {
                meta_X[i].push(p);
            }
        }
        
        if let Some(ref meta) = self.meta_model {
            let meta_arr = numpy.call_method1("array", (meta_X,))?;
            let pred = meta.call_method1(py, "predict", (&meta_arr,))?;
            let pred_list: Vec<f64> = pred.call_method0(py, "tolist")?.extract(py)?;
            Ok(pred_list)
        } else {
            let mut final_predictions = vec![0.0; n_samples];
            for i in 0..n_samples {
                final_predictions[i] = meta_X[i].iter().sum::<f64>() / meta_X[i].len() as f64;
            }
            Ok(final_predictions)
        }
    }
}
