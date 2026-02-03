//! Python bindings for inference module

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use kolosal_core::inference::{InferenceEngine, InferenceConfig, QuantizationType};
use kolosal_core::preprocessing::DataPreprocessor;
use kolosal_core::training::TrainEngine;

/// Inference configuration for Python
#[pyclass(name = "InferenceConfig")]
#[derive(Clone)]
pub struct PyInferenceConfig {
    inner: InferenceConfig,
}

#[pymethods]
impl PyInferenceConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: InferenceConfig::default(),
        }
    }

    /// Set batch size
    fn with_batch_size(&self, size: usize) -> Self {
        Self {
            inner: self.inner.clone().with_batch_size(size),
        }
    }

    /// Set number of workers
    fn with_n_workers(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().with_n_workers(n),
        }
    }

    /// Enable streaming mode
    fn with_streaming(&self, chunk_size: usize) -> Self {
        Self {
            inner: self.inner.clone().with_streaming(chunk_size),
        }
    }
}

/// Inference engine for Python
#[pyclass(name = "InferenceEngine")]
pub struct PyInferenceEngine {
    inner: Option<InferenceEngine>,
    config: InferenceConfig,
}

#[pymethods]
impl PyInferenceEngine {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyInferenceConfig>) -> Self {
        let config = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: Some(InferenceEngine::new(config.clone())),
            config,
        }
    }

    /// Load a model from file
    fn load_model(&mut self, model_path: &str, preprocessor_path: Option<&str>) -> PyResult<()> {
        let engine = InferenceEngine::load(self.config.clone(), preprocessor_path, model_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner = Some(engine);
        Ok(())
    }

    /// Make predictions on a pandas DataFrame
    fn predict(&self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let engine = self.inner.as_ref()
            .ok_or_else(|| PyValueError::new_err("Engine not loaded"))?;
        
        let polars_df = python_df_to_polars(py, df)?;
        let predictions = engine
            .predict(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(predictions.to_vec())
    }

    /// Make probability predictions (for classification)
    fn predict_proba(&self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let engine = self.inner.as_ref()
            .ok_or_else(|| PyValueError::new_err("Engine not loaded"))?;
        
        let polars_df = python_df_to_polars(py, df)?;
        let proba = engine
            .predict_proba(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Convert to Vec<Vec<f64>>
        let result: Vec<Vec<f64>> = proba
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        
        Ok(result)
    }
}

/// Convert Python DataFrame (pandas) to Polars DataFrame
fn python_df_to_polars(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<polars::prelude::DataFrame> {
    let columns: Vec<String> = df.getattr("columns")?.extract()?;
    
    let mut series_vec = Vec::new();
    
    for col in &columns {
        let col_data = df.get_item(col)?;
        let values: Vec<f64> = col_data.call_method0("tolist")?.extract().unwrap_or_default();
        let series = polars::prelude::Series::new(col.clone().into(), values);
        series_vec.push(series);
    }
    
    polars::prelude::DataFrame::new(series_vec)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
