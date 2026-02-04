//! Python bindings for preprocessing module

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use polars::prelude::NamedFrom;
use kolosal_core::preprocessing::{
    DataPreprocessor, PreprocessingConfig, ScalerType, EncoderType, ImputeStrategy,
};

/// Scaler type enum for Python
#[pyclass(name = "ScalerType")]
#[derive(Clone)]
pub enum PyScalerType {
    Standard,
    MinMax,
    Robust,
    MaxAbs,
    None,
}

impl From<PyScalerType> for ScalerType {
    fn from(py: PyScalerType) -> Self {
        match py {
            PyScalerType::Standard => ScalerType::Standard,
            PyScalerType::MinMax => ScalerType::MinMax,
            PyScalerType::Robust => ScalerType::Robust,
            PyScalerType::MaxAbs => ScalerType::MaxAbs,
            PyScalerType::None => ScalerType::None,
        }
    }
}

/// Encoder type enum for Python
#[pyclass(name = "EncoderType")]
#[derive(Clone)]
pub enum PyEncoderType {
    OneHot,
    Label,
    Target,
    Binary,
    Frequency,
}

impl From<PyEncoderType> for EncoderType {
    fn from(py: PyEncoderType) -> Self {
        match py {
            PyEncoderType::OneHot => EncoderType::OneHot,
            PyEncoderType::Label => EncoderType::Label,
            PyEncoderType::Target => EncoderType::Target,
            PyEncoderType::Binary => EncoderType::Binary,
            PyEncoderType::Frequency => EncoderType::Frequency,
        }
    }
}

/// Preprocessing configuration for Python
#[pyclass(name = "PreprocessingConfig")]
#[derive(Clone)]
pub struct PyPreprocessingConfig {
    inner: PreprocessingConfig,
}

#[pymethods]
impl PyPreprocessingConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: PreprocessingConfig::default(),
        }
    }

    /// Set scaler type
    fn with_scaler(&self, scaler: PyScalerType) -> Self {
        Self {
            inner: self.inner.clone().with_scaler(scaler.into()),
        }
    }

    /// Set encoder type
    fn with_encoder(&self, encoder: PyEncoderType) -> Self {
        Self {
            inner: self.inner.clone().with_encoder(encoder.into()),
        }
    }

    /// Enable outlier handling
    fn with_outlier_handling(&self, threshold: f64) -> Self {
        Self {
            inner: self.inner.clone().with_outlier_handling(threshold),
        }
    }

    /// Set number of parallel jobs
    fn with_n_jobs(&self, n_jobs: usize) -> Self {
        Self {
            inner: self.inner.clone().with_n_jobs(n_jobs),
        }
    }
}

/// Data preprocessor for Python
#[pyclass(name = "DataPreprocessor")]
pub struct PyDataPreprocessor {
    inner: DataPreprocessor,
}

#[pymethods]
impl PyDataPreprocessor {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyPreprocessingConfig>) -> Self {
        let inner = match config {
            Some(c) => DataPreprocessor::with_config(c.inner),
            None => DataPreprocessor::new(),
        };
        Self { inner }
    }

    /// Fit the preprocessor to a pandas DataFrame
    fn fit(&mut self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<()> {
        let polars_df = python_df_to_polars(py, df)?;
        self.inner
            .fit(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Transform a pandas DataFrame
    fn transform(&self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let polars_df = python_df_to_polars(py, df)?;
        let result = self
            .inner
            .transform(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        polars_to_python_df(py, &result)
    }

    /// Fit and transform in one step
    fn fit_transform(&mut self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let polars_df = python_df_to_polars(py, df)?;
        let result = self
            .inner
            .fit_transform(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        polars_to_python_df(py, &result)
    }

    /// Get numeric column names
    fn numeric_columns(&self) -> Vec<String> {
        self.inner.numeric_columns().to_vec()
    }

    /// Get categorical column names
    fn categorical_columns(&self) -> Vec<String> {
        self.inner.categorical_columns().to_vec()
    }

    /// Save preprocessor to file
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load preprocessor from file
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = DataPreprocessor::load(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

/// Convert Python DataFrame (pandas) to Polars DataFrame
fn python_df_to_polars(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<polars::prelude::DataFrame> {
    // This is a simplified implementation
    // In production, would use polars' Python integration or arrow
    
    // Get column names
    let columns: Vec<String> = df.getattr("columns")?.extract()?;
    
    let mut col_vec = Vec::new();
    
    for col in &columns {
        let col_data = df.get_item(col)?;
        let values: Vec<f64> = col_data.call_method0("tolist")?.extract().unwrap_or_default();
        let series = polars::prelude::Series::new(col.clone().into(), values);
        col_vec.push(series.into());
    }
    
    polars::prelude::DataFrame::new(col_vec)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Convert Polars DataFrame to Python DataFrame (pandas)
fn polars_to_python_df(py: Python<'_>, df: &polars::prelude::DataFrame) -> PyResult<PyObject> {
    // This is a simplified implementation
    // In production, would use polars' Python integration or arrow
    
    let pandas = py.import("pandas")?;
    let dict = pyo3::types::PyDict::new(py);
    
    for series in df.get_columns() {
        let name = series.name().to_string();
        let values: Vec<f64> = series
            .f64()
            .map(|ca| ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            .unwrap_or_default();
        dict.set_item(name, values)?;
    }
    
    let df = pandas.call_method1("DataFrame", (dict,))?;
    Ok(df.into())
}
