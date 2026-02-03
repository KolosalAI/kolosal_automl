//! Data preprocessing pipeline

use crate::error::{KolosalError, Result};
use super::{
    config::PreprocessingConfig,
    imputer::{Imputer, ImputeStrategy},
    scaler::{Scaler, ScalerType},
    encoder::{Encoder, EncoderType},
    FeatureStats, ColumnType,
};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main data preprocessing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPreprocessor {
    config: PreprocessingConfig,
    numeric_columns: Vec<String>,
    categorical_columns: Vec<String>,
    numeric_imputer: Option<Imputer>,
    categorical_imputer: Option<Imputer>,
    scaler: Option<Scaler>,
    encoder: Option<Encoder>,
    feature_stats: HashMap<String, FeatureStats>,
    is_fitted: bool,
}

impl DataPreprocessor {
    /// Create a new preprocessor with default configuration
    pub fn new() -> Self {
        Self::with_config(PreprocessingConfig::default())
    }

    /// Create a new preprocessor with custom configuration
    pub fn with_config(config: PreprocessingConfig) -> Self {
        Self {
            config,
            numeric_columns: Vec::new(),
            categorical_columns: Vec::new(),
            numeric_imputer: None,
            categorical_imputer: None,
            scaler: None,
            encoder: None,
            feature_stats: HashMap::new(),
            is_fitted: false,
        }
    }

    /// Fit the preprocessor to the data
    pub fn fit(&mut self, df: &DataFrame) -> Result<&mut Self> {
        // Detect column types
        self.detect_column_types(df)?;

        // Compute feature statistics
        self.compute_statistics(df)?;

        // Fit imputers
        if !self.numeric_columns.is_empty() {
            let mut imputer = Imputer::new(self.config.numeric_impute_strategy.clone());
            let cols: Vec<&str> = self.numeric_columns.iter().map(|s| s.as_str()).collect();
            imputer.fit(df, &cols)?;
            self.numeric_imputer = Some(imputer);
        }

        if !self.categorical_columns.is_empty() {
            let mut imputer = Imputer::new(self.config.categorical_impute_strategy.clone());
            let cols: Vec<&str> = self.categorical_columns.iter().map(|s| s.as_str()).collect();
            imputer.fit(df, &cols)?;
            self.categorical_imputer = Some(imputer);
        }

        // Fit scaler for numeric columns
        if !self.numeric_columns.is_empty() {
            let mut scaler = Scaler::new(self.config.scaler_type.clone());
            let cols: Vec<&str> = self.numeric_columns.iter().map(|s| s.as_str()).collect();
            
            // Need to impute first before fitting scaler
            let imputed = if let Some(ref imputer) = self.numeric_imputer {
                imputer.transform(df)?
            } else {
                df.clone()
            };
            
            scaler.fit(&imputed, &cols)?;
            self.scaler = Some(scaler);
        }

        // Fit encoder for categorical columns
        if !self.categorical_columns.is_empty() {
            let mut encoder = Encoder::new(self.config.encoder_type.clone());
            let cols: Vec<&str> = self.categorical_columns.iter().map(|s| s.as_str()).collect();
            
            // Need to impute first before fitting encoder
            let imputed = if let Some(ref imputer) = self.categorical_imputer {
                imputer.transform(df)?
            } else {
                df.clone()
            };
            
            encoder.fit(&imputed, &cols)?;
            self.encoder = Some(encoder);
        }

        self.is_fitted = true;
        Ok(self)
    }

    /// Transform the data
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }

        let mut result = df.clone();

        // Apply numeric imputation
        if let Some(ref imputer) = self.numeric_imputer {
            result = imputer.transform(&result)?;
        }

        // Apply categorical imputation
        if let Some(ref imputer) = self.categorical_imputer {
            result = imputer.transform(&result)?;
        }

        // Apply scaling
        if let Some(ref scaler) = self.scaler {
            result = scaler.transform(&result)?;
        }

        // Apply encoding
        if let Some(ref encoder) = self.encoder {
            result = encoder.transform(&result)?;
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }

    /// Get feature statistics
    pub fn feature_stats(&self) -> &HashMap<String, FeatureStats> {
        &self.feature_stats
    }

    /// Get numeric column names
    pub fn numeric_columns(&self) -> &[String] {
        &self.numeric_columns
    }

    /// Get categorical column names
    pub fn categorical_columns(&self) -> &[String] {
        &self.categorical_columns
    }

    /// Save the preprocessor to a file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a preprocessor from a file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let preprocessor: Self = serde_json::from_str(&json)?;
        Ok(preprocessor)
    }

    fn detect_column_types(&mut self, df: &DataFrame) -> Result<()> {
        self.numeric_columns.clear();
        self.categorical_columns.clear();

        for col in df.get_columns() {
            let name = col.name().to_string();
            
            match col.dtype() {
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
                DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 |
                DataType::Float32 | DataType::Float64 => {
                    self.numeric_columns.push(name);
                }
                DataType::String | DataType::Categorical(_, _) => {
                    self.categorical_columns.push(name);
                }
                _ => {
                    // Try to infer type from data
                    if col.str().is_ok() {
                        self.categorical_columns.push(name);
                    }
                }
            }
        }

        Ok(())
    }

    fn compute_statistics(&mut self, df: &DataFrame) -> Result<()> {
        self.feature_stats.clear();

        for col_name in &self.numeric_columns {
            if let Ok(series) = df.column(col_name) {
                let stats = FeatureStats::from_numeric_series(col_name, series)?;
                self.feature_stats.insert(col_name.clone(), stats);
            }
        }

        for col_name in &self.categorical_columns {
            if let Ok(series) = df.column(col_name) {
                let stats = FeatureStats::from_categorical_series(col_name, series)?;
                self.feature_stats.insert(col_name.clone(), stats);
            }
        }

        Ok(())
    }
}

impl Default for DataPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataframe() -> DataFrame {
        DataFrame::new(vec![
            Series::new("age".into(), &[25.0, 30.0, 35.0, 40.0, 45.0]),
            Series::new("income".into(), &[50000.0, 60000.0, 70000.0, 80000.0, 90000.0]),
            Series::new("city".into(), &["NYC", "LA", "NYC", "SF", "LA"]),
        ])
        .unwrap()
    }

    #[test]
    fn test_preprocessor_creation() {
        let preprocessor = DataPreprocessor::new();
        assert!(!preprocessor.is_fitted);
    }

    #[test]
    fn test_column_detection() {
        let df = create_test_dataframe();
        let mut preprocessor = DataPreprocessor::new();
        preprocessor.fit(&df).unwrap();

        assert_eq!(preprocessor.numeric_columns().len(), 2);
        assert_eq!(preprocessor.categorical_columns().len(), 1);
        assert!(preprocessor.numeric_columns().contains(&"age".to_string()));
        assert!(preprocessor.numeric_columns().contains(&"income".to_string()));
        assert!(preprocessor.categorical_columns().contains(&"city".to_string()));
    }

    #[test]
    fn test_fit_transform() {
        let df = create_test_dataframe();
        let mut preprocessor = DataPreprocessor::with_config(
            PreprocessingConfig::new()
                .with_scaler(ScalerType::Standard)
                .with_encoder(EncoderType::OneHot),
        );

        let result = preprocessor.fit_transform(&df).unwrap();
        
        // Original numeric columns should be scaled
        assert!(result.column("age").is_ok());
        assert!(result.column("income").is_ok());
        
        // Original categorical column should be removed (one-hot)
        assert!(result.column("city").is_err());
    }

    #[test]
    fn test_feature_statistics() {
        let df = create_test_dataframe();
        let mut preprocessor = DataPreprocessor::new();
        preprocessor.fit(&df).unwrap();

        let stats = preprocessor.feature_stats();
        
        let age_stats = stats.get("age").unwrap();
        assert_eq!(age_stats.dtype, ColumnType::Numeric);
        assert!(age_stats.mean.is_some());
        
        let city_stats = stats.get("city").unwrap();
        assert_eq!(city_stats.dtype, ColumnType::Categorical);
        assert!(city_stats.categories.is_some());
    }
}
