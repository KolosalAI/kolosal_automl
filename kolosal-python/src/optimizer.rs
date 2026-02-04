//! Python bindings for optimizer module

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use kolosal_core::optimizer::{
    HyperOptX, OptimizationConfig, OptimizeDirection, SearchSpace, Parameter, SamplerType,
};
use std::collections::HashMap;

/// Optimization configuration for Python
#[pyclass(name = "OptimizationConfig")]
#[derive(Clone)]
pub struct PyOptimizationConfig {
    inner: OptimizationConfig,
}

#[pymethods]
impl PyOptimizationConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: OptimizationConfig::default(),
        }
    }

    /// Set number of trials
    fn with_n_trials(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().with_n_trials(n),
        }
    }

    /// Set timeout in seconds
    fn with_timeout(&self, secs: f64) -> Self {
        Self {
            inner: self.inner.clone().with_timeout(secs),
        }
    }

    /// Set to minimize (default)
    fn minimize(&self) -> Self {
        Self {
            inner: self.inner.clone().with_direction(OptimizeDirection::Minimize),
        }
    }

    /// Set to maximize
    fn maximize(&self) -> Self {
        Self {
            inner: self.inner.clone().with_direction(OptimizeDirection::Maximize),
        }
    }

    /// Set number of parallel jobs
    fn with_n_jobs(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().with_n_jobs(n),
        }
    }

    /// Set metric to optimize
    fn with_metric(&self, metric: String) -> Self {
        Self {
            inner: self.inner.clone().with_metric(metric),
        }
    }
}

/// Search space for Python
#[pyclass(name = "SearchSpace")]
#[derive(Clone)]
pub struct PySearchSpace {
    inner: SearchSpace,
}

#[pymethods]
impl PySearchSpace {
    #[new]
    fn new() -> Self {
        Self {
            inner: SearchSpace::new(),
        }
    }

    /// Add a float parameter
    fn float(&self, name: String, low: f64, high: f64) -> Self {
        Self {
            inner: self.inner.clone().float(name, low, high),
        }
    }

    /// Add a log-scale float parameter
    fn log_float(&self, name: String, low: f64, high: f64) -> Self {
        Self {
            inner: self.inner.clone().log_float(name, low, high),
        }
    }

    /// Add an integer parameter
    fn int(&self, name: String, low: i64, high: i64) -> Self {
        Self {
            inner: self.inner.clone().int(name, low, high),
        }
    }

    /// Add a categorical parameter
    fn categorical(&self, name: String, choices: Vec<String>) -> Self {
        let choices_refs: Vec<&str> = choices.iter().map(|s| s.as_str()).collect();
        Self {
            inner: self.inner.clone().categorical(name, choices_refs),
        }
    }

    /// Add a boolean parameter
    fn boolean(&self, name: String) -> Self {
        Self {
            inner: self.inner.clone().boolean(name),
        }
    }

    /// Get number of parameters
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// HyperOptX optimizer for Python
#[pyclass(name = "HyperOptX")]
pub struct PyHyperOptX {
    config: OptimizationConfig,
    search_space: SearchSpace,
}

#[pymethods]
impl PyHyperOptX {
    #[new]
    fn new(config: PyOptimizationConfig, search_space: PySearchSpace) -> Self {
        Self {
            config: config.inner,
            search_space: search_space.inner,
        }
    }

    /// Run optimization with a Python objective function
    fn optimize(&self, py: Python<'_>, objective: PyObject) -> PyResult<PyObject> {
        let mut optimizer = HyperOptX::new(self.config.clone(), self.search_space.clone());
        
        // Wrap Python objective
        let py_objective = |params: &HashMap<String, kolosal_core::optimizer::ParameterValue>| 
            -> kolosal_core::error::Result<f64> {
            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                for (key, value) in params {
                    match value {
                        kolosal_core::optimizer::ParameterValue::Float(v) => {
                            dict.set_item(key, *v).ok();
                        }
                        kolosal_core::optimizer::ParameterValue::Int(v) => {
                            dict.set_item(key, *v).ok();
                        }
                        kolosal_core::optimizer::ParameterValue::String(v) => {
                            dict.set_item(key, v).ok();
                        }
                        kolosal_core::optimizer::ParameterValue::Bool(v) => {
                            dict.set_item(key, *v).ok();
                        }
                    }
                }
                
                let result = objective.call1(py, (dict,))
                    .map_err(|e| kolosal_core::error::KolosalError::OptimizationError(e.to_string()))?;
                
                result.extract::<f64>(py)
                    .map_err(|e| kolosal_core::error::KolosalError::OptimizationError(e.to_string()))
            })
        };
        
        let study = optimizer.optimize(py_objective)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Convert study to Python dict
        let result = pyo3::types::PyDict::new(py);
        
        // Best parameters
        if let Some(best_params) = study.best_params() {
            let params_dict = pyo3::types::PyDict::new(py);
            for (key, value) in best_params {
                match value {
                    kolosal_core::optimizer::ParameterValue::Float(v) => {
                        params_dict.set_item(key, *v)?;
                    }
                    kolosal_core::optimizer::ParameterValue::Int(v) => {
                        params_dict.set_item(key, *v)?;
                    }
                    kolosal_core::optimizer::ParameterValue::String(v) => {
                        params_dict.set_item(key, v)?;
                    }
                    kolosal_core::optimizer::ParameterValue::Bool(v) => {
                        params_dict.set_item(key, *v)?;
                    }
                }
            }
            result.set_item("best_params", params_dict)?;
        }
        
        // Best value
        if let Some(best_value) = study.best_value() {
            result.set_item("best_value", best_value)?;
        }
        
        // Number of trials
        result.set_item("n_trials", study.trials.len())?;
        
        // Total duration
        result.set_item("total_duration_secs", study.total_duration_secs)?;
        
        Ok(result.into())
    }

    /// Save study to file
    fn save_study(&self, path: &str) -> PyResult<()> {
        let optimizer = HyperOptX::new(self.config.clone(), self.search_space.clone());
        optimizer.save_study(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
