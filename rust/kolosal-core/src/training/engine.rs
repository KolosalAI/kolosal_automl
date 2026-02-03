//! Training engine implementation

use crate::error::{KolosalError, Result};
use super::{TrainingConfig, ModelType, TaskType, ModelMetrics};
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Main training engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainEngine {
    config: TrainingConfig,
    feature_names: Vec<String>,
    model_bytes: Option<Vec<u8>>,
    metrics: Option<ModelMetrics>,
    is_fitted: bool,
}

impl TrainEngine {
    /// Create a new training engine
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            feature_names: Vec::new(),
            model_bytes: None,
            metrics: None,
            is_fitted: false,
        }
    }

    /// Fit the model to the data
    pub fn fit(&mut self, df: &DataFrame) -> Result<&mut Self> {
        let start = Instant::now();

        // Extract features and target
        let (x, y) = self.prepare_data(df)?;

        // Split data
        let (x_train, x_val, y_train, y_val) = self.train_val_split(&x, &y)?;

        // Train model based on type
        let model_bytes = self.train_model(&x_train, &y_train)?;
        self.model_bytes = Some(model_bytes);

        // Compute metrics on validation set
        let y_pred = self.predict_internal(&x_val)?;
        let mut metrics = match self.config.task_type {
            TaskType::Regression | TaskType::TimeSeries => {
                ModelMetrics::compute_regression(&y_val, &y_pred)
            }
            _ => ModelMetrics::compute_classification(&y_val, &y_pred, None),
        };

        metrics.training_time_secs = start.elapsed().as_secs_f64();
        metrics.n_features = x.ncols();
        metrics.n_samples = x.nrows();
        self.metrics = Some(metrics);

        self.is_fitted = true;
        Ok(self)
    }

    /// Make predictions on new data
    pub fn predict(&self, df: &DataFrame) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(KolosalError::ModelNotFitted);
        }

        let x = self.extract_features(df)?;
        self.predict_internal(&x)
    }

    /// Get training metrics
    pub fn metrics(&self) -> Option<&ModelMetrics> {
        self.metrics.as_ref()
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Save the engine to a file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load an engine from a file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let engine: Self = serde_json::from_str(&json)?;
        Ok(engine)
    }

    fn prepare_data(&mut self, df: &DataFrame) -> Result<(Array2<f64>, Array1<f64>)> {
        // Get feature columns
        let feature_cols: Vec<String> = match &self.config.feature_columns {
            Some(cols) => cols.clone(),
            None => df
                .get_column_names()
                .into_iter()
                .filter(|name| *name != self.config.target_column)
                .map(|s| s.to_string())
                .collect(),
        };

        self.feature_names = feature_cols.clone();

        // Extract target
        let target_series = df
            .column(&self.config.target_column)
            .map_err(|_| KolosalError::FeatureNotFound(self.config.target_column.clone()))?;

        let y: Array1<f64> = target_series
            .f64()
            .map_err(|e| KolosalError::DataError(e.to_string()))?
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();

        // Extract features
        let n_rows = df.height();
        let n_cols = feature_cols.len();
        let mut x_data = Vec::with_capacity(n_rows * n_cols);

        for col_name in &feature_cols {
            let series = df
                .column(col_name)
                .map_err(|_| KolosalError::FeatureNotFound(col_name.clone()))?;

            let values: Vec<f64> = series
                .f64()
                .map_err(|e| KolosalError::DataError(e.to_string()))?
                .into_iter()
                .map(|v| v.unwrap_or(0.0))
                .collect();

            x_data.extend(values);
        }

        let x = Array2::from_shape_vec((n_cols, n_rows), x_data)
            .map_err(|e| KolosalError::ShapeError {
                expected: format!("({}, {})", n_cols, n_rows),
                actual: e.to_string(),
            })?
            .t()
            .to_owned();

        Ok((x, y))
    }

    fn extract_features(&self, df: &DataFrame) -> Result<Array2<f64>> {
        let n_rows = df.height();
        let n_cols = self.feature_names.len();
        let mut x_data = Vec::with_capacity(n_rows * n_cols);

        for col_name in &self.feature_names {
            let series = df
                .column(col_name)
                .map_err(|_| KolosalError::FeatureNotFound(col_name.clone()))?;

            let values: Vec<f64> = series
                .f64()
                .map_err(|e| KolosalError::DataError(e.to_string()))?
                .into_iter()
                .map(|v| v.unwrap_or(0.0))
                .collect();

            x_data.extend(values);
        }

        let x = Array2::from_shape_vec((n_cols, n_rows), x_data)
            .map_err(|e| KolosalError::ShapeError {
                expected: format!("({}, {})", n_cols, n_rows),
                actual: e.to_string(),
            })?
            .t()
            .to_owned();

        Ok(x)
    }

    fn train_val_split(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>)> {
        let n = x.nrows();
        let val_size = (n as f64 * self.config.validation_split) as usize;
        let train_size = n - val_size;

        // Simple split (TODO: add stratified split for classification)
        let x_train = x.slice(ndarray::s![..train_size, ..]).to_owned();
        let x_val = x.slice(ndarray::s![train_size.., ..]).to_owned();
        let y_train = y.slice(ndarray::s![..train_size]).to_owned();
        let y_val = y.slice(ndarray::s![train_size..]).to_owned();

        Ok((x_train, x_val, y_train, y_val))
    }

    fn train_model(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Vec<u8>> {
        // Placeholder implementation - in production would use actual ML algorithms
        // For now, compute and store basic statistics for a simple linear model
        
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Simple mean-based model (placeholder)
        let y_mean = y.mean().unwrap_or(0.0);
        
        // Compute feature means
        let mut feature_means = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col = x.column(j);
            feature_means.push(col.mean().unwrap_or(0.0));
        }

        // Serialize model data
        let model_data = SimpleModelData {
            y_mean,
            feature_means,
            n_samples,
        };

        let bytes = bincode::serialize(&model_data)
            .map_err(|e| KolosalError::SerializationError(e.to_string()))?;

        Ok(bytes)
    }

    fn predict_internal(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let model_bytes = self
            .model_bytes
            .as_ref()
            .ok_or(KolosalError::ModelNotFitted)?;

        let model_data: SimpleModelData = bincode::deserialize(model_bytes)
            .map_err(|e| KolosalError::SerializationError(e.to_string()))?;

        // Simple prediction (placeholder - just return mean)
        let predictions = Array1::from_elem(x.nrows(), model_data.y_mean);

        Ok(predictions)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SimpleModelData {
    y_mean: f64,
    feature_means: Vec<f64>,
    n_samples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> DataFrame {
        DataFrame::new(vec![
            Series::new("feature1".into(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            Series::new("feature2".into(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]),
            Series::new("target".into(), &[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]),
        ])
        .unwrap()
    }

    #[test]
    fn test_engine_creation() {
        let config = TrainingConfig::new(TaskType::Regression, "target");
        let engine = TrainEngine::new(config);
        assert!(!engine.is_fitted);
    }

    #[test]
    fn test_fit_predict() {
        let df = create_test_data();
        let config = TrainingConfig::new(TaskType::Regression, "target");
        let mut engine = TrainEngine::new(config);

        engine.fit(&df).unwrap();
        assert!(engine.is_fitted);
        assert!(engine.metrics().is_some());

        let predictions = engine.predict(&df).unwrap();
        assert_eq!(predictions.len(), 10);
    }

    #[test]
    fn test_feature_names() {
        let df = create_test_data();
        let config = TrainingConfig::new(TaskType::Regression, "target");
        let mut engine = TrainEngine::new(config);

        engine.fit(&df).unwrap();

        let feature_names = engine.feature_names();
        assert_eq!(feature_names.len(), 2);
        assert!(feature_names.contains(&"feature1".to_string()));
        assert!(feature_names.contains(&"feature2".to_string()));
    }
}
