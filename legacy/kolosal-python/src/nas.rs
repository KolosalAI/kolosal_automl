//! Python bindings for Neural Architecture Search

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use kolosal_core::nas::{NASController, SearchStrategy, ArchitectureEvaluator};

use crate::utils::python_df_to_polars;

/// NAS strategy enum for Python
#[pyclass(name = "NASStrategy")]
#[derive(Clone)]
pub enum PyNASStrategy {
    RandomSearch,
    EvolutionarySearch,
    BayesianSearch,
    ENAS,
    DARTS,
}

impl From<PyNASStrategy> for SearchStrategy {
    fn from(py: PyNASStrategy) -> Self {
        match py {
            PyNASStrategy::RandomSearch => SearchStrategy::RandomSearch,
            PyNASStrategy::EvolutionarySearch => SearchStrategy::EvolutionarySearch,
            PyNASStrategy::BayesianSearch => SearchStrategy::BayesianSearch,
            PyNASStrategy::ENAS => SearchStrategy::ENAS,
            PyNASStrategy::DARTS => SearchStrategy::DARTS,
        }
    }
}

/// NAS Controller for Python
#[pyclass(name = "NASController")]
pub struct PyNASController {
    inner: NASController,
    best_architecture: Option<std::collections::HashMap<String, serde_json::Value>>,
    history: Vec<(std::collections::HashMap<String, serde_json::Value>, f64)>,
}

#[pymethods]
impl PyNASController {
    #[new]
    #[pyo3(signature = (strategy=None, n_trials=100, time_budget=None, random_state=None))]
    fn new(
        strategy: Option<PyNASStrategy>,
        n_trials: usize,
        time_budget: Option<f64>,
        random_state: Option<u64>,
    ) -> Self {
        let strat = strategy.map(|s| s.into()).unwrap_or(SearchStrategy::RandomSearch);
        
        let mut controller = NASController::new(strat).with_n_trials(n_trials);
        
        if let Some(tb) = time_budget {
            controller = controller.with_time_budget(tb);
        }
        if let Some(seed) = random_state {
            controller = controller.with_random_state(seed);
        }
        
        Self {
            inner: controller,
            best_architecture: None,
            history: Vec::new(),
        }
    }

    /// Define search space for architecture
    fn define_search_space(&mut self, py: Python<'_>, space: &Bound<'_, PyAny>) -> PyResult<()> {
        let space_dict: std::collections::HashMap<String, PyObject> = space.extract()?;
        
        let mut rust_space = std::collections::HashMap::new();
        for (key, value) in space_dict {
            if let Ok(v) = value.extract::<Vec<String>>(py) {
                rust_space.insert(key, serde_json::json!(v));
            } else if let Ok(v) = value.extract::<Vec<i64>>(py) {
                rust_space.insert(key, serde_json::json!(v));
            } else if let Ok(v) = value.extract::<Vec<f64>>(py) {
                rust_space.insert(key, serde_json::json!(v));
            } else if let Ok(v) = value.extract::<String>(py) {
                rust_space.insert(key, serde_json::json!(v));
            }
        }
        
        self.inner
            .define_search_space(rust_space)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Run architecture search
    fn search(
        &mut self,
        py: Python<'_>,
        X: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let x_df = python_df_to_polars(py, X)?;
        let y_vec: Vec<f64> = y.call_method0("tolist")?.extract()?;
        
        let result = self.inner
            .search(&x_df, &y_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        self.best_architecture = Some(result.best_architecture.clone());
        self.history = result.history.clone();
        
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("best_score", result.best_score)?;
        dict.set_item("n_trials", result.n_trials)?;
        dict.set_item("total_time", result.total_time)?;
        
        let arch_dict = pyo3::types::PyDict::new(py);
        for (k, v) in &result.best_architecture {
            arch_dict.set_item(k, v.to_string())?;
        }
        dict.set_item("best_architecture", arch_dict)?;
        
        Ok(dict.into())
    }

    /// Get best architecture found
    fn best_architecture(&self, py: Python<'_>) -> PyResult<PyObject> {
        let arch = self.best_architecture.as_ref()
            .ok_or_else(|| PyValueError::new_err("No search performed yet"))?;
        
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in arch {
            dict.set_item(k, v.to_string())?;
        }
        Ok(dict.into())
    }

    /// Get search history
    fn history(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = pyo3::types::PyList::empty(py);
        
        for (arch, score) in &self.history {
            let item = pyo3::types::PyDict::new(py);
            
            let arch_dict = pyo3::types::PyDict::new(py);
            for (k, v) in arch {
                arch_dict.set_item(k, v.to_string())?;
            }
            item.set_item("architecture", arch_dict)?;
            item.set_item("score", *score)?;
            
            list.append(item)?;
        }
        
        Ok(list.into())
    }

    /// Export best architecture as model config
    fn export_config(&self) -> PyResult<String> {
        let arch = self.best_architecture.as_ref()
            .ok_or_else(|| PyValueError::new_err("No search performed yet"))?;
        
        serde_json::to_string_pretty(arch)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get number of trials
    fn n_trials(&self) -> usize {
        self.inner.n_trials()
    }

    /// Get time budget
    fn time_budget(&self) -> Option<f64> {
        self.inner.time_budget()
    }

    /// Get strategy
    fn strategy(&self) -> String {
        match self.inner.strategy() {
            SearchStrategy::RandomSearch => "random".to_string(),
            SearchStrategy::EvolutionarySearch => "evolutionary".to_string(),
            SearchStrategy::BayesianSearch => "bayesian".to_string(),
            SearchStrategy::ENAS => "enas".to_string(),
            SearchStrategy::DARTS => "darts".to_string(),
        }
    }
}

/// Architecture evaluator for Python
#[pyclass(name = "ArchitectureEvaluator")]
pub struct PyArchitectureEvaluator {
    inner: ArchitectureEvaluator,
}

#[pymethods]
impl PyArchitectureEvaluator {
    #[new]
    #[pyo3(signature = (metric="accuracy", cv_folds=3, early_stopping=true))]
    fn new(metric: &str, cv_folds: usize, early_stopping: bool) -> Self {
        Self {
            inner: ArchitectureEvaluator::new()
                .with_metric(metric)
                .with_cv_folds(cv_folds)
                .with_early_stopping(early_stopping),
        }
    }

    /// Evaluate an architecture
    fn evaluate(
        &self,
        py: Python<'_>,
        architecture: &Bound<'_, PyAny>,
        X: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let arch_dict: std::collections::HashMap<String, PyObject> = architecture.extract()?;
        
        let mut rust_arch = std::collections::HashMap::new();
        for (key, value) in arch_dict {
            if let Ok(v) = value.extract::<String>(py) {
                rust_arch.insert(key, serde_json::json!(v));
            } else if let Ok(v) = value.extract::<i64>(py) {
                rust_arch.insert(key, serde_json::json!(v));
            } else if let Ok(v) = value.extract::<f64>(py) {
                rust_arch.insert(key, serde_json::json!(v));
            }
        }
        
        let x_df = python_df_to_polars(py, X)?;
        let y_vec: Vec<f64> = y.call_method0("tolist")?.extract()?;
        
        self.inner
            .evaluate(&rust_arch, &x_df, &y_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get metric
    fn metric(&self) -> String {
        self.inner.metric().to_string()
    }

    /// Get CV folds
    fn cv_folds(&self) -> usize {
        self.inner.cv_folds()
    }

    /// Check if early stopping enabled
    fn early_stopping(&self) -> bool {
        self.inner.early_stopping()
    }
}

/// Layer search space for NAS
#[pyclass(name = "LayerSearchSpace")]
pub struct PyLayerSearchSpace {
    layers: Vec<std::collections::HashMap<String, serde_json::Value>>,
}

#[pymethods]
impl PyLayerSearchSpace {
    #[new]
    fn new() -> Self {
        Self {
            layers: Vec::new(),
        }
    }

    /// Add a dense layer option
    fn add_dense(&mut self, units: Vec<usize>, activation: Vec<String>) -> &mut Self {
        let mut layer = std::collections::HashMap::new();
        layer.insert("type".to_string(), serde_json::json!("dense"));
        layer.insert("units".to_string(), serde_json::json!(units));
        layer.insert("activation".to_string(), serde_json::json!(activation));
        self.layers.push(layer);
        self
    }

    /// Add a dropout layer option
    fn add_dropout(&mut self, rates: Vec<f64>) -> &mut Self {
        let mut layer = std::collections::HashMap::new();
        layer.insert("type".to_string(), serde_json::json!("dropout"));
        layer.insert("rate".to_string(), serde_json::json!(rates));
        self.layers.push(layer);
        self
    }

    /// Add a batch norm layer option
    fn add_batch_norm(&mut self) -> &mut Self {
        let mut layer = std::collections::HashMap::new();
        layer.insert("type".to_string(), serde_json::json!("batch_norm"));
        self.layers.push(layer);
        self
    }

    /// Get number of layer options
    fn __len__(&self) -> usize {
        self.layers.len()
    }

    /// Convert to dict
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = pyo3::types::PyList::empty(py);
        
        for layer in &self.layers {
            let dict = pyo3::types::PyDict::new(py);
            for (k, v) in layer {
                dict.set_item(k, v.to_string())?;
            }
            list.append(dict)?;
        }
        
        Ok(list.into())
    }
}
