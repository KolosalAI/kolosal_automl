//! Time series module
//!
//! Provides time series preprocessing and feature engineering including:
//! - Lag features
//! - Rolling statistics
//! - Differencing
//! - Seasonal decomposition
//! - Time-based features
//! - Trend extraction

mod features;
mod transforms;
mod validation;

pub use features::{TimeSeriesFeatures, TimeFeatureConfig, LagConfig, RollingConfig};
pub use transforms::{TimeSeriesTransformer, Differencer, SeasonalDecomposer, DecomposeMethod};
pub use validation::{TimeSeriesCV, TimeSeriesSplit, WalkForwardCV};

