//! Feature scaling implementations

use crate::error::{KolosalError, Result};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of scaler to use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalerType {
    /// Standard scaling (z-score normalization): (x - mean) / std
    Standard,
    /// Min-Max scaling: (x - min) / (max - min)
    MinMax,
    /// Robust scaling using median and IQR
    Robust,
    /// Max absolute scaling: x / max(|x|)
    MaxAbs,
    /// No scaling
    None,
}

/// Parameters for a fitted scaler
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalerParams {
    center: f64,    // mean, min, or median
    scale: f64,     // std, range, or IQR
}

/// Feature scaler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scaler {
    scaler_type: ScalerType,
    params: HashMap<String, ScalerParams>,
    is_fitted: bool,
}

impl Scaler {
    /// Create a new scaler
    pub fn new(scaler_type: ScalerType) -> Self {
        Self {
            scaler_type,
            params: HashMap::new(),
            is_fitted: false,
        }
    }

    /// Fit the scaler to the data
    pub fn fit(&mut self, df: &DataFrame, columns: &[&str]) -> Result<&mut Self> {
        for col_name in columns {
            let column = df
                .column(col_name)
                .map_err(|_| KolosalError::FeatureNotFound(col_name.to_string()))?;
            let series = column.as_materialized_series();

            let params = self.compute_params(series)?;
            self.params.insert(col_name.to_string(), params);
        }

        self.is_fitted = true;
        Ok(self)
    }

    /// Transform the data.
    /// Builds all replacement columns first, then applies them in a single pass
    /// (avoids N DataFrame clones for N columns).
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }

        // Build all scaled columns first
        let replacements: Vec<Series> = self.params.iter()
            .filter_map(|(col_name, params)| {
                df.column(col_name).ok().map(|column| {
                    let series = column.as_materialized_series();
                    self.scale_series(series, params)
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Apply all at once (single clone + N in-place replacements)
        let mut result = df.clone();
        for scaled in replacements {
            result = result
                .with_column(scaled)
                .map_err(|e| KolosalError::DataError(e.to_string()))?
                .clone();
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        self.fit(df, columns)?;
        self.transform(df)
    }

    /// Inverse transform the data.
    /// Same batch-apply pattern as transform.
    pub fn inverse_transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }

        let replacements: Vec<Series> = self.params.iter()
            .filter_map(|(col_name, params)| {
                df.column(col_name).ok().map(|column| {
                    let series = column.as_materialized_series();
                    self.unscale_series(series, params)
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut result = df.clone();
        for unscaled in replacements {
            result = result
                .with_column(unscaled)
                .map_err(|e| KolosalError::DataError(e.to_string()))?
                .clone();
        }

        Ok(result)
    }

    fn compute_params(&self, series: &Series) -> Result<ScalerParams> {
        let ca = series
            .f64()
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        match self.scaler_type {
            ScalerType::Standard => {
                let mean = ca.mean().unwrap_or(0.0);
                let std = ca.std(1).unwrap_or(1.0);
                Ok(ScalerParams {
                    center: mean,
                    scale: if std == 0.0 { 1.0 } else { std },
                })
            }
            ScalerType::MinMax => {
                let min = ca.min().unwrap_or(0.0);
                let max = ca.max().unwrap_or(1.0);
                let range = max - min;
                Ok(ScalerParams {
                    center: min,
                    scale: if range == 0.0 { 1.0 } else { range },
                })
            }
            ScalerType::Robust => {
                let median = ca.median().unwrap_or(0.0);
                let q1 = ca.quantile(0.25, QuantileMethod::Linear).unwrap_or(Some(0.0)).unwrap_or(0.0);
                let q3 = ca.quantile(0.75, QuantileMethod::Linear).unwrap_or(Some(1.0)).unwrap_or(1.0);
                let iqr = q3 - q1;
                Ok(ScalerParams {
                    center: median,
                    scale: if iqr == 0.0 { 1.0 } else { iqr },
                })
            }
            ScalerType::MaxAbs => {
                let max_abs = ca
                    .into_iter()
                    .filter_map(|v| v.map(|x| x.abs()))
                    .fold(0.0f64, |a, b| a.max(b));
                Ok(ScalerParams {
                    center: 0.0,
                    scale: if max_abs == 0.0 { 1.0 } else { max_abs },
                })
            }
            ScalerType::None => Ok(ScalerParams {
                center: 0.0,
                scale: 1.0,
            }),
        }
    }

    fn scale_series(&self, series: &Series, params: &ScalerParams) -> Result<Series> {
        let ca = series
            .f64()
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        let scaled: Float64Chunked = ca
            .into_iter()
            .map(|opt| opt.map(|v| (v - params.center) / params.scale))
            .collect();

        Ok(scaled.with_name(series.name().clone()).into_series())
    }

    fn unscale_series(&self, series: &Series, params: &ScalerParams) -> Result<Series> {
        let ca = series
            .f64()
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        let unscaled: Float64Chunked = ca
            .into_iter()
            .map(|opt| opt.map(|v| v * params.scale + params.center))
            .collect();

        Ok(unscaled.with_name(series.name().clone()).into_series())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler() {
        let df = DataFrame::new(vec![
            Series::new("a".into(), &[1.0, 2.0, 3.0, 4.0, 5.0]).into(),
        ])
        .unwrap();

        let mut scaler = Scaler::new(ScalerType::Standard);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        let mean: f64 = col.mean().unwrap();
        assert!(mean.abs() < 1e-10); // Mean should be ~0
    }

    #[test]
    fn test_minmax_scaler() {
        let df = DataFrame::new(vec![
            Series::new("a".into(), &[1.0, 2.0, 3.0, 4.0, 5.0]).into(),
        ])
        .unwrap();

        let mut scaler = Scaler::new(ScalerType::MinMax);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        assert!((col.min().unwrap() - 0.0).abs() < 1e-10);
        assert!((col.max().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_transform() {
        let df = DataFrame::new(vec![
            Series::new("a".into(), &[1.0, 2.0, 3.0, 4.0, 5.0]).into(),
        ])
        .unwrap();

        let mut scaler = Scaler::new(ScalerType::Standard);
        let scaled = scaler.fit_transform(&df, &["a"]).unwrap();
        let unscaled = scaler.inverse_transform(&scaled).unwrap();

        let original = df.column("a").unwrap().f64().unwrap();
        let restored = unscaled.column("a").unwrap().f64().unwrap();

        for (o, r) in original.into_iter().zip(restored.into_iter()) {
            assert!((o.unwrap() - r.unwrap()).abs() < 1e-10);
        }
    }
}
