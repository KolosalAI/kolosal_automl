//! Python bindings for automatic pipeline construction

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use kolosal_core::autopipeline::{AutoPipeline, PipelineConfig, DataTypeDetector};
use kolosal_core::training::TaskType;

use crate::utils::{python_df_to_polars, polars_to_python_df};
use crate::training::PyTaskType;

/// Data type detector for Python
#[pyclass(name = "DataTypeDetector")]
pub struct PyDataTypeDetector {
    inner: DataTypeDetector,
    numeric_cols: Vec<String>,
    categorical_cols: Vec<String>,
    datetime_cols: Vec<String>,
    text_cols: Vec<String>,
}

#[pymethods]
impl PyDataTypeDetector {
    #[new]
    fn new() -> Self {
        Self {
            inner: DataTypeDetector::new(),
            numeric_cols: Vec::new(),
            categorical_cols: Vec::new(),
            datetime_cols: Vec::new(),
            text_cols: Vec::new(),
        }
    }

    /// Detect column types from a DataFrame
    fn detect(&mut self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let polars_df = python_df_to_polars(py, df)?;
        
        let result = self.inner
            .detect(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        self.numeric_cols = result.numeric.clone();
        self.categorical_cols = result.categorical.clone();
        self.datetime_cols = result.datetime.clone();
        self.text_cols = result.text.clone();
        
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("numeric", &result.numeric)?;
        dict.set_item("categorical", &result.categorical)?;
        dict.set_item("datetime", &result.datetime)?;
        dict.set_item("text", &result.text)?;
        
        Ok(dict.into())
    }

    /// Get numeric columns
    fn numeric_columns(&self) -> Vec<String> {
        self.numeric_cols.clone()
    }

    /// Get categorical columns
    fn categorical_columns(&self) -> Vec<String> {
        self.categorical_cols.clone()
    }

    /// Get datetime columns
    fn datetime_columns(&self) -> Vec<String> {
        self.datetime_cols.clone()
    }

    /// Get text columns
    fn text_columns(&self) -> Vec<String> {
        self.text_cols.clone()
    }
}

/// Pipeline configuration for Python
#[pyclass(name = "PipelineConfig")]
#[derive(Clone)]
pub struct PyPipelineConfig {
    inner: PipelineConfig,
}

#[pymethods]
impl PyPipelineConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: PipelineConfig::default(),
        }
    }

    /// Set task type
    fn with_task(&self, task: PyTaskType) -> Self {
        Self {
            inner: self.inner.clone().with_task(task.into()),
        }
    }

    /// Set target column
    fn with_target(&self, target: String) -> Self {
        Self {
            inner: self.inner.clone().with_target(target),
        }
    }

    /// Enable auto feature selection
    fn with_feature_selection(&self, enabled: bool) -> Self {
        Self {
            inner: self.inner.clone().with_feature_selection(enabled),
        }
    }

    /// Enable auto hyperparameter tuning
    fn with_auto_tune(&self, enabled: bool) -> Self {
        Self {
            inner: self.inner.clone().with_auto_tune(enabled),
        }
    }

    /// Set optimization metric
    fn with_metric(&self, metric: String) -> Self {
        Self {
            inner: self.inner.clone().with_metric(metric),
        }
    }

    /// Set time budget in seconds
    fn with_time_budget(&self, seconds: f64) -> Self {
        Self {
            inner: self.inner.clone().with_time_budget(seconds),
        }
    }

    /// Set number of trials for tuning
    fn with_n_trials(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().with_n_trials(n),
        }
    }

    /// Set random state
    fn with_random_state(&self, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().with_random_state(seed),
        }
    }
}

/// Auto pipeline for Python
#[pyclass(name = "AutoPipeline")]
pub struct PyAutoPipeline {
    inner: AutoPipeline,
    best_model: Option<String>,
    best_params: Option<std::collections::HashMap<String, f64>>,
    cv_scores: Option<Vec<f64>>,
    feature_importances: Option<Vec<f64>>,
}

#[pymethods]
impl PyAutoPipeline {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyPipelineConfig>) -> Self {
        let inner = match config {
            Some(c) => AutoPipeline::with_config(c.inner),
            None => AutoPipeline::new(),
        };
        Self {
            inner,
            best_model: None,
            best_params: None,
            cv_scores: None,
            feature_importances: None,
        }
    }

    /// Fit the auto pipeline
    fn fit(&mut self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<()> {
        let polars_df = python_df_to_polars(py, df)?;
        
        let result = self.inner
            .fit(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        self.best_model = Some(result.best_model.clone());
        self.best_params = Some(result.best_params.clone());
        self.cv_scores = Some(result.cv_scores.clone());
        self.feature_importances = result.feature_importances.clone();
        
        Ok(())
    }

    /// Make predictions
    fn predict(&self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let polars_df = python_df_to_polars(py, df)?;
        
        self.inner
            .predict(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Predict probabilities (classification)
    fn predict_proba(&self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let polars_df = python_df_to_polars(py, df)?;
        
        self.inner
            .predict_proba(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get the best model type
    fn best_model(&self) -> Option<String> {
        self.best_model.clone()
    }

    /// Get the best hyperparameters
    fn best_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        if let Some(ref params) = self.best_params {
            for (k, v) in params {
                dict.set_item(k, *v)?;
            }
        }
        Ok(dict.into())
    }

    /// Get pipeline steps
    fn steps(&self) -> Vec<String> {
        self.inner.steps()
    }

    /// Get feature importances
    fn feature_importances(&self) -> Option<Vec<f64>> {
        self.feature_importances.clone()
    }

    /// Get cross-validation scores
    fn cv_scores(&self) -> Option<Vec<f64>> {
        self.cv_scores.clone()
    }

    /// Export pipeline to file
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load pipeline from file
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = AutoPipeline::load(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            best_model: None,
            best_params: None,
            cv_scores: None,
            feature_importances: None,
        })
    }

    /// Get best score
    fn best_score(&self) -> Option<f64> {
        self.cv_scores.as_ref().map(|scores| {
            scores.iter().sum::<f64>() / scores.len() as f64
        })
    }
}
