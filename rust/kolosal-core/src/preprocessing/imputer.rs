//! Missing value imputation strategies

use crate::error::{KolosalError, Result};
use ndarray::{Array1, ArrayView1, Axis};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy for imputing missing values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImputeStrategy {
    /// Replace with mean (numeric only)
    Mean,
    /// Replace with median (numeric only)
    Median,
    /// Replace with mode / most frequent value
    MostFrequent,
    /// Replace with a constant value
    Constant(f64),
    /// Replace with a constant string (categorical)
    ConstantString(String),
    /// Forward fill
    ForwardFill,
    /// Backward fill
    BackwardFill,
    /// KNN imputation
    Knn { n_neighbors: usize },
    /// Drop rows with missing values
    Drop,
}

/// Imputer for handling missing values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Imputer {
    strategy: ImputeStrategy,
    fill_values: HashMap<String, ImputeValue>,
    is_fitted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ImputeValue {
    Numeric(f64),
    String(String),
}

impl Imputer {
    /// Create a new imputer with the specified strategy
    pub fn new(strategy: ImputeStrategy) -> Self {
        Self {
            strategy,
            fill_values: HashMap::new(),
            is_fitted: false,
        }
    }

    /// Fit the imputer to the data
    pub fn fit(&mut self, df: &DataFrame, columns: &[&str]) -> Result<&mut Self> {
        for col_name in columns {
            let series = df
                .column(col_name)
                .map_err(|_| KolosalError::FeatureNotFound(col_name.to_string()))?;

            let fill_value = self.compute_fill_value(series)?;
            self.fill_values.insert(col_name.to_string(), fill_value);
        }

        self.is_fitted = true;
        Ok(self)
    }

    /// Transform the data by imputing missing values
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }

        let mut result = df.clone();

        for (col_name, fill_value) in &self.fill_values {
            if let Ok(series) = df.column(col_name) {
                let filled = self.fill_series(series, fill_value)?;
                result = result
                    .with_column(filled)
                    .map_err(|e| KolosalError::DataError(e.to_string()))?
                    .clone();
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        self.fit(df, columns)?;
        self.transform(df)
    }

    fn compute_fill_value(&self, series: &Series) -> Result<ImputeValue> {
        match &self.strategy {
            ImputeStrategy::Mean => {
                let mean = series
                    .f64()
                    .map_err(|e| KolosalError::DataError(e.to_string()))?
                    .mean()
                    .unwrap_or(0.0);
                Ok(ImputeValue::Numeric(mean))
            }
            ImputeStrategy::Median => {
                let median = series
                    .f64()
                    .map_err(|e| KolosalError::DataError(e.to_string()))?
                    .median()
                    .unwrap_or(0.0);
                Ok(ImputeValue::Numeric(median))
            }
            ImputeStrategy::MostFrequent => {
                // Get mode (most frequent value)
                if series.dtype().is_numeric() {
                    let mode = series
                        .mode()
                        .map_err(|e| KolosalError::DataError(e.to_string()))?;
                    let val = mode
                        .f64()
                        .map_err(|e| KolosalError::DataError(e.to_string()))?
                        .get(0)
                        .unwrap_or(0.0);
                    Ok(ImputeValue::Numeric(val))
                } else {
                    let mode = series
                        .mode()
                        .map_err(|e| KolosalError::DataError(e.to_string()))?;
                    let val = mode
                        .str()
                        .map_err(|e| KolosalError::DataError(e.to_string()))?
                        .get(0)
                        .unwrap_or("")
                        .to_string();
                    Ok(ImputeValue::String(val))
                }
            }
            ImputeStrategy::Constant(val) => Ok(ImputeValue::Numeric(*val)),
            ImputeStrategy::ConstantString(val) => Ok(ImputeValue::String(val.clone())),
            _ => Ok(ImputeValue::Numeric(0.0)), // Placeholder for other strategies
        }
    }

    fn fill_series(&self, series: &Series, fill_value: &ImputeValue) -> Result<Series> {
        match fill_value {
            ImputeValue::Numeric(val) => {
                let filled = series
                    .f64()
                    .map_err(|e| KolosalError::DataError(e.to_string()))?
                    .fill_null_with_values(*val)
                    .map_err(|e| KolosalError::DataError(e.to_string()))?;
                Ok(filled.into_series())
            }
            ImputeValue::String(val) => {
                let filled = series
                    .str()
                    .map_err(|e| KolosalError::DataError(e.to_string()))?
                    .fill_null_with_values(val)
                    .map_err(|e| KolosalError::DataError(e.to_string()))?;
                Ok(filled.into_series())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imputer_creation() {
        let imputer = Imputer::new(ImputeStrategy::Mean);
        assert!(!imputer.is_fitted);
    }

    #[test]
    fn test_impute_strategy_serialize() {
        let strategy = ImputeStrategy::Knn { n_neighbors: 5 };
        let json = serde_json::to_string(&strategy).unwrap();
        assert!(json.contains("Knn"));
        assert!(json.contains("5"));
    }

    #[test]
    fn test_mean_imputation() {
        let df = DataFrame::new(vec![
            Series::new("a".into(), &[Some(1.0), None, Some(3.0), Some(4.0)]),
        ])
        .unwrap();

        let mut imputer = Imputer::new(ImputeStrategy::Mean);
        let result = imputer.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        // Mean of [1, 3, 4] = 8/3 ≈ 2.67
        assert!((col.get(1).unwrap() - 2.666666666666667).abs() < 0.001);
    }
}
