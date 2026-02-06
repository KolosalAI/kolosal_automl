//! Python bindings for time series utilities

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

/// Time series feature generator for Python
#[pyclass(name = "TimeSeriesFeatures")]
pub struct PyTimeSeriesFeatures {
    lags: Vec<usize>,
    rolling_windows: Vec<usize>,
    ewm_spans: Vec<usize>,
}

#[pymethods]
impl PyTimeSeriesFeatures {
    #[new]
    #[pyo3(signature = (lags=None, rolling_windows=None, ewm_spans=None))]
    fn new(
        lags: Option<Vec<usize>>,
        rolling_windows: Option<Vec<usize>>,
        ewm_spans: Option<Vec<usize>>,
    ) -> Self {
        Self {
            lags: lags.unwrap_or_else(|| vec![1, 2, 3, 7, 14]),
            rolling_windows: rolling_windows.unwrap_or_else(|| vec![7, 14, 30]),
            ewm_spans: ewm_spans.unwrap_or_else(|| vec![7, 14, 30]),
        }
    }

    /// Generate lag features
    fn create_lags(&self, py: Python<'_>, series: Vec<f64>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        let n = series.len();
        
        for &lag in &self.lags {
            let mut lagged = vec![f64::NAN; n];
            for i in lag..n {
                lagged[i] = series[i - lag];
            }
            dict.set_item(format!("lag_{}", lag), lagged)?;
        }
        
        Ok(dict.into())
    }

    /// Generate rolling statistics
    fn create_rolling(&self, py: Python<'_>, series: Vec<f64>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        let n = series.len();
        
        for &window in &self.rolling_windows {
            let mut rolling_mean = vec![f64::NAN; n];
            let mut rolling_std = vec![f64::NAN; n];
            let mut rolling_min = vec![f64::NAN; n];
            let mut rolling_max = vec![f64::NAN; n];
            
            for i in (window - 1)..n {
                let win: Vec<f64> = series[(i + 1 - window)..=i].to_vec();
                let mean = win.iter().sum::<f64>() / win.len() as f64;
                let variance = win.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / win.len() as f64;
                
                rolling_mean[i] = mean;
                rolling_std[i] = variance.sqrt();
                rolling_min[i] = win.iter().cloned().fold(f64::INFINITY, f64::min);
                rolling_max[i] = win.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            }
            
            dict.set_item(format!("rolling_mean_{}", window), rolling_mean)?;
            dict.set_item(format!("rolling_std_{}", window), rolling_std)?;
            dict.set_item(format!("rolling_min_{}", window), rolling_min)?;
            dict.set_item(format!("rolling_max_{}", window), rolling_max)?;
        }
        
        Ok(dict.into())
    }

    /// Generate exponential moving statistics
    fn create_ewm(&self, py: Python<'_>, series: Vec<f64>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        let n = series.len();
        
        for &span in &self.ewm_spans {
            let alpha = 2.0 / (span as f64 + 1.0);
            let mut ewm = vec![0.0; n];
            
            if !series.is_empty() {
                ewm[0] = series[0];
                for i in 1..n {
                    ewm[i] = alpha * series[i] + (1.0 - alpha) * ewm[i - 1];
                }
            }
            
            dict.set_item(format!("ewm_{}", span), ewm)?;
        }
        
        Ok(dict.into())
    }

    /// Get feature names that will be created
    fn get_feature_names(&self, column: &str) -> Vec<String> {
        let mut names = Vec::new();
        
        for &lag in &self.lags {
            names.push(format!("{}_lag_{}", column, lag));
        }
        for &window in &self.rolling_windows {
            names.push(format!("{}_rolling_mean_{}", column, window));
            names.push(format!("{}_rolling_std_{}", column, window));
            names.push(format!("{}_rolling_min_{}", column, window));
            names.push(format!("{}_rolling_max_{}", column, window));
        }
        for &span in &self.ewm_spans {
            names.push(format!("{}_ewm_{}", column, span));
        }
        
        names
    }
}

/// Time series cross-validation for Python
#[pyclass(name = "TimeSeriesCV")]
pub struct PyTimeSeriesCV {
    n_splits: usize,
    gap: usize,
    test_size: Option<usize>,
}

#[pymethods]
impl PyTimeSeriesCV {
    #[new]
    #[pyo3(signature = (n_splits=5, gap=0, test_size=None))]
    fn new(n_splits: usize, gap: usize, test_size: Option<usize>) -> Self {
        Self { n_splits, gap, test_size }
    }

    /// Generate train/test splits
    fn split(&self, py: Python<'_>, n_samples: usize) -> PyResult<PyObject> {
        let test_size = self.test_size.unwrap_or(n_samples / (self.n_splits + 1));
        let min_train_size = test_size;
        
        let list = pyo3::types::PyList::empty(py);
        
        for i in 0..self.n_splits {
            let test_end = n_samples - (self.n_splits - i - 1) * test_size;
            let test_start = test_end - test_size;
            let train_end = test_start - self.gap;
            
            if train_end < min_train_size {
                continue;
            }
            
            let train_indices: Vec<usize> = (0..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            
            let tuple = pyo3::types::PyTuple::new(py, &[
                train_indices.into_pyobject(py)?.into_any(),
                test_indices.into_pyobject(py)?.into_any(),
            ])?;
            list.append(tuple)?;
        }
        
        Ok(list.into())
    }

    /// Get number of splits
    fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    /// Get gap size
    fn gap(&self) -> usize {
        self.gap
    }

    /// Get test size
    fn test_size(&self) -> Option<usize> {
        self.test_size
    }
}

/// Differencer for time series stationarity
#[pyclass(name = "Differencer")]
pub struct PyDifferencer {
    order: usize,
    seasonal_order: usize,
    seasonal_period: usize,
    initial_values: Vec<f64>,
}

#[pymethods]
impl PyDifferencer {
    #[new]
    #[pyo3(signature = (order=1, seasonal_order=0, seasonal_period=1))]
    fn new(order: usize, seasonal_order: usize, seasonal_period: usize) -> Self {
        Self {
            order,
            seasonal_order,
            seasonal_period,
            initial_values: Vec::new(),
        }
    }

    /// Difference the series
    fn fit_transform(&mut self, series: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut result = series.clone();
        self.initial_values.clear();
        
        // Regular differencing
        for _ in 0..self.order {
            if result.len() <= 1 {
                break;
            }
            self.initial_values.push(result[0]);
            result = result.windows(2).map(|w| w[1] - w[0]).collect();
        }
        
        // Seasonal differencing
        for _ in 0..self.seasonal_order {
            if result.len() <= self.seasonal_period {
                break;
            }
            for i in 0..self.seasonal_period {
                self.initial_values.push(result[i]);
            }
            let new_result: Vec<f64> = (self.seasonal_period..result.len())
                .map(|i| result[i] - result[i - self.seasonal_period])
                .collect();
            result = new_result;
        }
        
        Ok(result)
    }

    /// Inverse difference to recover original scale
    fn inverse_transform(&self, diffed: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut result = diffed;
        let mut init_idx = self.initial_values.len();
        
        // Inverse seasonal differencing
        for _ in 0..self.seasonal_order {
            init_idx -= self.seasonal_period;
            let mut new_result = Vec::with_capacity(result.len() + self.seasonal_period);
            for i in 0..self.seasonal_period {
                new_result.push(self.initial_values[init_idx + i]);
            }
            for (i, &d) in result.iter().enumerate() {
                new_result.push(d + new_result[i]);
            }
            result = new_result;
        }
        
        // Inverse regular differencing
        for _ in 0..self.order {
            init_idx -= 1;
            let mut new_result = vec![self.initial_values[init_idx]];
            for &d in &result {
                new_result.push(new_result.last().unwrap() + d);
            }
            result = new_result;
        }
        
        Ok(result)
    }

    /// Check if series is stationary (simple variance test)
    fn is_stationary(&self, series: Vec<f64>) -> PyResult<bool> {
        if series.len() < 10 {
            return Ok(true);
        }
        
        let n = series.len();
        let half = n / 2;
        
        let first_half = &series[..half];
        let second_half = &series[half..];
        
        let mean1: f64 = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let mean2: f64 = second_half.iter().sum::<f64>() / second_half.len() as f64;
        
        let var1: f64 = first_half.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / first_half.len() as f64;
        let var2: f64 = second_half.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / second_half.len() as f64;
        
        // Simple check: means and variances should be similar
        let mean_ratio = (mean1 - mean2).abs() / (mean1.abs() + mean2.abs() + 1e-10);
        let var_ratio = var1.max(var2) / (var1.min(var2) + 1e-10);
        
        Ok(mean_ratio < 0.2 && var_ratio < 2.0)
    }

    /// Get recommended differencing order
    #[staticmethod]
    fn recommend_order(series: Vec<f64>) -> PyResult<usize> {
        let mut temp = series;
        
        for order in 0..3 {
            // Check stationarity
            if temp.len() < 10 {
                return Ok(order);
            }
            
            let n = temp.len();
            let half = n / 2;
            let mean1: f64 = temp[..half].iter().sum::<f64>() / half as f64;
            let mean2: f64 = temp[half..].iter().sum::<f64>() / (n - half) as f64;
            let mean_ratio = (mean1 - mean2).abs() / (mean1.abs() + mean2.abs() + 1e-10);
            
            if mean_ratio < 0.1 {
                return Ok(order);
            }
            
            // Difference
            temp = temp.windows(2).map(|w| w[1] - w[0]).collect();
        }
        
        Ok(2)
    }

    /// Get order
    fn order(&self) -> usize {
        self.order
    }

    /// Get seasonal order
    fn seasonal_order(&self) -> usize {
        self.seasonal_order
    }

    /// Get seasonal period
    fn seasonal_period(&self) -> usize {
        self.seasonal_period
    }

    /// Get initial values needed for inverse transform
    fn initial_values(&self) -> Vec<f64> {
        self.initial_values.clone()
    }
}
