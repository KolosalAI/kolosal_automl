//! Python bindings for feature selection

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use kolosal_core::preprocessing::{FeatureSelector, SelectionMethod};
use ndarray::{Array1, Array2};

use crate::utils::python_df_to_polars;

/// Selection method enum for Python
#[pyclass(name = "SelectionMethod")]
#[derive(Clone)]
pub enum PySelectionMethod {
    VarianceThreshold,
    MutualInformation,
    CorrelationThreshold,
    RFE,
    Percentile,
    ImportanceThreshold,
}

/// Feature selector for Python
#[pyclass(name = "FeatureSelector")]
pub struct PyFeatureSelector {
    inner: FeatureSelector,
    selected_indices: Option<Vec<usize>>,
    feature_scores: Option<Vec<f64>>,
}

#[pymethods]
impl PyFeatureSelector {
    #[new]
    #[pyo3(signature = (method=None, k=None, threshold=None, percentile=None))]
    fn new(
        method: Option<PySelectionMethod>,
        k: Option<usize>,
        threshold: Option<f64>,
        percentile: Option<f64>,
    ) -> Self {
        let selection_method = match method {
            Some(PySelectionMethod::VarianceThreshold) => {
                SelectionMethod::VarianceThreshold { threshold: threshold.unwrap_or(0.0) }
            }
            Some(PySelectionMethod::MutualInformation) => {
                SelectionMethod::MutualInformation { k: k.unwrap_or(10) }
            }
            Some(PySelectionMethod::CorrelationThreshold) => {
                SelectionMethod::CorrelationThreshold { threshold: threshold.unwrap_or(0.9) }
            }
            Some(PySelectionMethod::RFE) => {
                SelectionMethod::RFE { n_features_to_select: k.unwrap_or(10), step: 1 }
            }
            Some(PySelectionMethod::Percentile) => {
                SelectionMethod::Percentile { percentile: percentile.unwrap_or(50.0) }
            }
            Some(PySelectionMethod::ImportanceThreshold) => {
                SelectionMethod::ImportanceThreshold { threshold: threshold.unwrap_or(0.01) }
            }
            None => SelectionMethod::VarianceThreshold { threshold: threshold.unwrap_or(0.0) },
        };
        
        Self {
            inner: FeatureSelector::new(selection_method),
            selected_indices: None,
            feature_scores: None,
        }
    }

    /// Fit the selector to data
    fn fit(&mut self, py: Python<'_>, X: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        // Convert to ndarray
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        let y_list: Vec<f64> = y.call_method0("tolist")?.extract()?;
        
        let n_rows = x_list.len();
        let n_cols = x_list.first().map(|r| r.len()).unwrap_or(0);
        
        let x_flat: Vec<f64> = x_list.into_iter().flatten().collect();
        let x_arr = Array2::from_shape_vec((n_rows, n_cols), x_flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let y_arr = Array1::from_vec(y_list);
        
        self.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        self.selected_indices = self.inner.selected_indices().map(|s| s.to_vec());
        self.feature_scores = self.inner.scores().map(|s| s.to_vec());
        
        Ok(())
    }

    /// Transform data (select features)
    fn transform(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        
        let n_rows = x_list.len();
        let n_cols = x_list.first().map(|r| r.len()).unwrap_or(0);
        
        let x_flat: Vec<f64> = x_list.into_iter().flatten().collect();
        let x_arr = Array2::from_shape_vec((n_rows, n_cols), x_flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let result = self.inner
            .transform(&x_arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Convert back to Python list
        let result_vec: Vec<Vec<f64>> = result.outer_iter()
            .map(|row| row.to_vec())
            .collect();
        
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (result_vec,))?;
        Ok(array.into())
    }

    /// Fit and transform in one step
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        X: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        self.fit(py, X, y)?;
        self.transform(py, X)
    }

    /// Get selected feature indices
    fn get_support(&self) -> Vec<usize> {
        self.selected_indices.clone().unwrap_or_default()
    }

    /// Get feature importances/scores
    fn feature_scores(&self) -> Option<Vec<f64>> {
        self.feature_scores.clone()
    }

    /// Get selected feature names
    fn get_feature_names(&self, input_names: Vec<String>) -> Vec<String> {
        let indices = self.selected_indices.as_ref().cloned().unwrap_or_default();
        indices
            .into_iter()
            .filter_map(|i| input_names.get(i).cloned())
            .collect()
    }

    /// Get number of selected features
    fn n_features_selected(&self) -> usize {
        self.selected_indices.as_ref().map(|v| v.len()).unwrap_or(0)
    }
}
