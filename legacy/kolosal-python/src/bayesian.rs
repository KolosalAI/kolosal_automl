//! Python bindings for Bayesian optimization

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use kolosal_core::optimizer::{BayesianOptimizer, GaussianProcess, AcquisitionFunction};

use crate::optimizer::PySearchSpace;

/// Gaussian Process for Python
#[pyclass(name = "GaussianProcess")]
pub struct PyGaussianProcess {
    inner: GaussianProcess,
}

#[pymethods]
impl PyGaussianProcess {
    #[new]
    #[pyo3(signature = (kernel="rbf", length_scale=1.0, noise=1e-10))]
    fn new(kernel: &str, length_scale: f64, noise: f64) -> Self {
        Self {
            inner: GaussianProcess::new()
                .with_kernel(kernel)
                .with_length_scale(length_scale)
                .with_noise(noise),
        }
    }

    /// Fit the GP to data
    fn fit(&mut self, X: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        self.inner
            .fit(&X, &y)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Predict mean and std
    fn predict(&self, py: Python<'_>, X: Vec<Vec<f64>>) -> PyResult<PyObject> {
        let (mean, std) = self.inner
            .predict(&X)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let numpy = py.import("numpy")?;
        let mean_arr = numpy.call_method1("array", (mean,))?;
        let std_arr = numpy.call_method1("array", (std,))?;
        
        let tuple = pyo3::types::PyTuple::new(py, &[mean_arr, std_arr])?;
        Ok(tuple.into())
    }

    /// Get log marginal likelihood
    fn log_marginal_likelihood(&self) -> f64 {
        self.inner.log_marginal_likelihood()
    }

    /// Get kernel name
    fn kernel(&self) -> String {
        self.inner.kernel().to_string()
    }

    /// Get length scale
    fn length_scale(&self) -> f64 {
        self.inner.length_scale()
    }

    /// Get noise
    fn noise(&self) -> f64 {
        self.inner.noise()
    }

    /// Check if fitted
    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    /// Get number of training samples
    fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }
}

/// Acquisition function enum for Python
#[pyclass(name = "AcquisitionFunction")]
#[derive(Clone)]
pub enum PyAcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    LowerConfidenceBound,
}

impl From<PyAcquisitionFunction> for AcquisitionFunction {
    fn from(py: PyAcquisitionFunction) -> Self {
        match py {
            PyAcquisitionFunction::ExpectedImprovement => AcquisitionFunction::ExpectedImprovement,
            PyAcquisitionFunction::ProbabilityOfImprovement => AcquisitionFunction::ProbabilityOfImprovement,
            PyAcquisitionFunction::UpperConfidenceBound => AcquisitionFunction::UpperConfidenceBound(2.0),
            PyAcquisitionFunction::LowerConfidenceBound => AcquisitionFunction::LowerConfidenceBound(2.0),
        }
    }
}

/// Bayesian Optimizer for Python
#[pyclass(name = "BayesianOptimizer")]
pub struct PyBayesianOptimizer {
    inner: BayesianOptimizer,
    history_x: Vec<Vec<f64>>,
    history_y: Vec<f64>,
}

#[pymethods]
impl PyBayesianOptimizer {
    #[new]
    #[pyo3(signature = (search_space, n_initial=5, acquisition="ei", xi=0.01, kappa=2.0))]
    fn new(
        search_space: PySearchSpace,
        n_initial: usize,
        acquisition: &str,
        xi: f64,
        kappa: f64,
    ) -> Self {
        let acq = match acquisition {
            "ei" | "expected_improvement" => AcquisitionFunction::ExpectedImprovement,
            "pi" | "probability_of_improvement" => AcquisitionFunction::ProbabilityOfImprovement,
            "ucb" | "upper_confidence_bound" => AcquisitionFunction::UpperConfidenceBound(kappa),
            "lcb" | "lower_confidence_bound" => AcquisitionFunction::LowerConfidenceBound(kappa),
            _ => AcquisitionFunction::ExpectedImprovement,
        };
        
        Self {
            inner: BayesianOptimizer::new(search_space.inner)
                .with_n_initial(n_initial)
                .with_acquisition(acq)
                .with_xi(xi),
            history_x: Vec::new(),
            history_y: Vec::new(),
        }
    }

    /// Suggest next point to evaluate
    fn suggest(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let params = self.inner
            .suggest()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let dict = pyo3::types::PyDict::new(py);
        for (key, value) in params {
            match value {
                kolosal_core::optimizer::ParameterValue::Float(v) => {
                    dict.set_item(&key, v)?;
                }
                kolosal_core::optimizer::ParameterValue::Int(v) => {
                    dict.set_item(&key, v)?;
                }
                kolosal_core::optimizer::ParameterValue::String(v) => {
                    dict.set_item(&key, v)?;
                }
                kolosal_core::optimizer::ParameterValue::Bool(v) => {
                    dict.set_item(&key, v)?;
                }
            }
        }
        Ok(dict.into())
    }

    /// Register observation
    fn register(&mut self, py: Python<'_>, params: &Bound<'_, PyAny>, value: f64) -> PyResult<()> {
        let params_dict: std::collections::HashMap<String, PyObject> = params.extract()?;
        
        let mut rust_params = std::collections::HashMap::new();
        for (key, value_obj) in params_dict {
            if let Ok(v) = value_obj.extract::<f64>(py) {
                rust_params.insert(key, kolosal_core::optimizer::ParameterValue::Float(v));
            } else if let Ok(v) = value_obj.extract::<i64>(py) {
                rust_params.insert(key, kolosal_core::optimizer::ParameterValue::Int(v));
            } else if let Ok(v) = value_obj.extract::<bool>(py) {
                rust_params.insert(key, kolosal_core::optimizer::ParameterValue::Bool(v));
            } else if let Ok(v) = value_obj.extract::<String>(py) {
                rust_params.insert(key, kolosal_core::optimizer::ParameterValue::String(v));
            }
        }
        
        self.inner
            .register(rust_params, value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        self.history_y.push(value);
        
        Ok(())
    }

    /// Run optimization with objective function
    fn optimize(
        &mut self,
        py: Python<'_>,
        objective: PyObject,
        n_iter: usize,
    ) -> PyResult<PyObject> {
        for _ in 0..n_iter {
            let params = self.suggest(py)?;
            let value: f64 = objective.call1(py, (&params,))?.extract(py)?;
            self.register(py, params.bind(py), value)?;
        }
        
        self.best_params(py)
    }

    /// Get best parameters found
    fn best_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let params = self.inner
            .best_params()
            .ok_or_else(|| PyValueError::new_err("No observations yet"))?;
        
        let dict = pyo3::types::PyDict::new(py);
        for (key, value) in params {
            match value {
                kolosal_core::optimizer::ParameterValue::Float(v) => {
                    dict.set_item(key, *v)?;
                }
                kolosal_core::optimizer::ParameterValue::Int(v) => {
                    dict.set_item(key, *v)?;
                }
                kolosal_core::optimizer::ParameterValue::String(v) => {
                    dict.set_item(key, v)?;
                }
                kolosal_core::optimizer::ParameterValue::Bool(v) => {
                    dict.set_item(key, *v)?;
                }
            }
        }
        Ok(dict.into())
    }

    /// Get best value found
    fn best_value(&self) -> Option<f64> {
        self.inner.best_value()
    }

    /// Get optimization history
    fn history(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("values", &self.history_y)?;
        dict.set_item("n_iterations", self.history_y.len())?;
        
        if let Some(best) = self.best_value() {
            dict.set_item("best_value", best)?;
        }
        
        Ok(dict.into())
    }

    /// Get number of observations
    fn n_observations(&self) -> usize {
        self.inner.n_observations()
    }

    /// Get number of initial random samples
    fn n_initial(&self) -> usize {
        self.inner.n_initial()
    }
}

/// Tree-structured Parzen Estimator for Python
#[pyclass(name = "TPEOptimizer")]
pub struct PyTPEOptimizer {
    search_space: kolosal_core::optimizer::SearchSpace,
    n_startup: usize,
    gamma: f64,
    history: Vec<(std::collections::HashMap<String, kolosal_core::optimizer::ParameterValue>, f64)>,
}

#[pymethods]
impl PyTPEOptimizer {
    #[new]
    #[pyo3(signature = (search_space, n_startup=10, gamma=0.25))]
    fn new(search_space: PySearchSpace, n_startup: usize, gamma: f64) -> Self {
        Self {
            search_space: search_space.inner,
            n_startup,
            gamma,
            history: Vec::new(),
        }
    }

    /// Suggest next point to evaluate
    fn suggest(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        // TPE implementation would go here
        // For now, delegate to random sampling
        let params = self.search_space
            .sample()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let dict = pyo3::types::PyDict::new(py);
        for (key, value) in params {
            match value {
                kolosal_core::optimizer::ParameterValue::Float(v) => {
                    dict.set_item(&key, v)?;
                }
                kolosal_core::optimizer::ParameterValue::Int(v) => {
                    dict.set_item(&key, v)?;
                }
                kolosal_core::optimizer::ParameterValue::String(v) => {
                    dict.set_item(&key, v)?;
                }
                kolosal_core::optimizer::ParameterValue::Bool(v) => {
                    dict.set_item(&key, v)?;
                }
            }
        }
        Ok(dict.into())
    }

    /// Register observation
    fn register(&mut self, py: Python<'_>, params: &Bound<'_, PyAny>, value: f64) -> PyResult<()> {
        let params_dict: std::collections::HashMap<String, PyObject> = params.extract()?;
        
        let mut rust_params = std::collections::HashMap::new();
        for (key, value_obj) in params_dict {
            if let Ok(v) = value_obj.extract::<f64>(py) {
                rust_params.insert(key, kolosal_core::optimizer::ParameterValue::Float(v));
            } else if let Ok(v) = value_obj.extract::<i64>(py) {
                rust_params.insert(key, kolosal_core::optimizer::ParameterValue::Int(v));
            } else if let Ok(v) = value_obj.extract::<bool>(py) {
                rust_params.insert(key, kolosal_core::optimizer::ParameterValue::Bool(v));
            } else if let Ok(v) = value_obj.extract::<String>(py) {
                rust_params.insert(key, kolosal_core::optimizer::ParameterValue::String(v));
            }
        }
        
        self.history.push((rust_params, value));
        Ok(())
    }

    /// Get best parameters
    fn best_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let best = self.history
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .ok_or_else(|| PyValueError::new_err("No observations yet"))?;
        
        let dict = pyo3::types::PyDict::new(py);
        for (key, value) in &best.0 {
            match value {
                kolosal_core::optimizer::ParameterValue::Float(v) => {
                    dict.set_item(key, *v)?;
                }
                kolosal_core::optimizer::ParameterValue::Int(v) => {
                    dict.set_item(key, *v)?;
                }
                kolosal_core::optimizer::ParameterValue::String(v) => {
                    dict.set_item(key, v)?;
                }
                kolosal_core::optimizer::ParameterValue::Bool(v) => {
                    dict.set_item(key, *v)?;
                }
            }
        }
        Ok(dict.into())
    }

    /// Get best value
    fn best_value(&self) -> Option<f64> {
        self.history.iter().map(|(_, v)| *v).min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Get number of observations
    fn n_observations(&self) -> usize {
        self.history.len()
    }
}
