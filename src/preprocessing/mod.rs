//! Data preprocessing module
//!
//! Provides high-performance data preprocessing capabilities including:
//! - Missing value imputation
//! - Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
//! - Categorical encoding (OneHot, Label, Target, Hash encoding)
//! - Feature transformation (Log, Power, Box-Cox, Yeo-Johnson)
//! - Binning and discretization
//! - Outlier detection and handling

mod config;
mod imputer;
mod scaler;
mod encoder;
mod pipeline;
pub mod outlier;
pub mod transforms;
pub mod feature_selection;

pub use config::PreprocessingConfig;
pub use imputer::{Imputer, ImputeStrategy};
pub use scaler::{Scaler, ScalerType};
pub use encoder::{Encoder, EncoderType};
pub use pipeline::DataPreprocessor;
pub use outlier::{OutlierDetector, OutlierMethod, OutlierStrategy, OutlierBounds};
pub use transforms::{Transformer, TransformType, Binner, BinningStrategy, BinEncoding};
pub use feature_selection::{FeatureSelector, SelectionMethod, CorrelationFilter};

use crate::error::Result;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Column data type for preprocessing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnType {
    Numeric,
    Categorical,
    DateTime,
    Text,
    Unknown,
}

/// Feature statistics computed during fit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub name: String,
    pub dtype: ColumnType,
    pub count: usize,
    pub null_count: usize,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub median: Option<f64>,
    pub unique_count: Option<usize>,
    pub categories: Option<Vec<String>>,
}

impl FeatureStats {
    /// Create new feature statistics
    pub fn new(name: impl Into<String>, dtype: ColumnType) -> Self {
        Self {
            name: name.into(),
            dtype,
            count: 0,
            null_count: 0,
            mean: None,
            std: None,
            min: None,
            max: None,
            median: None,
            unique_count: None,
            categories: None,
        }
    }

    /// Compute statistics from a numeric series
    pub fn from_numeric_series(name: &str, series: &Series) -> Result<Self> {
        let mut stats = Self::new(name, ColumnType::Numeric);
        stats.count = series.len();
        stats.null_count = series.null_count();

        if let Ok(ca) = series.cast(&DataType::Float64).and_then(|s| Ok(s.f64().unwrap().clone())) {
            stats.mean = ca.mean();
            stats.std = ca.std(1);
            stats.min = ca.min();
            stats.max = ca.max();
            stats.median = ca.median();
        }

        Ok(stats)
    }

    /// Compute statistics from a categorical series
    pub fn from_categorical_series(name: &str, series: &Series) -> Result<Self> {
        let mut stats = Self::new(name, ColumnType::Categorical);
        stats.count = series.len();
        stats.null_count = series.null_count();
        stats.unique_count = Some(series.n_unique().unwrap_or(0));

        if let Ok(ca) = series.str() {
            let categories: Vec<String> = ca
                .unique()
                .unwrap_or_else(|_| ca.clone())
                .into_iter()
                .filter_map(|s| s.map(|s| s.to_string()))
                .collect();
            stats.categories = Some(categories);
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_stats_new() {
        let stats = FeatureStats::new("test_feature", ColumnType::Numeric);
        assert_eq!(stats.name, "test_feature");
        assert_eq!(stats.dtype, ColumnType::Numeric);
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_column_type_serialize() {
        let dtype = ColumnType::Numeric;
        let json = serde_json::to_string(&dtype).unwrap();
        assert_eq!(json, "\"Numeric\"");
    }
}
