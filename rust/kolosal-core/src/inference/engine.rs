//! Inference engine implementation

use crate::error::{KolosalError, Result};
use crate::preprocessing::DataPreprocessor;
use crate::training::TrainEngine;
use super::InferenceConfig;
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// High-performance inference engine
#[derive(Debug)]
pub struct InferenceEngine {
    config: InferenceConfig,
    preprocessor: Option<Arc<DataPreprocessor>>,
    model: Option<Arc<TrainEngine>>,
    is_loaded: bool,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            config,
            preprocessor: None,
            model: None,
            is_loaded: false,
        }
    }

    /// Load a preprocessor
    pub fn with_preprocessor(mut self, preprocessor: DataPreprocessor) -> Self {
        self.preprocessor = Some(Arc::new(preprocessor));
        self
    }

    /// Load a trained model
    pub fn with_model(mut self, model: TrainEngine) -> Self {
        self.model = Some(Arc::new(model));
        self.is_loaded = true;
        self
    }

    /// Load from files
    pub fn load(
        config: InferenceConfig,
        preprocessor_path: Option<&str>,
        model_path: &str,
    ) -> Result<Self> {
        let mut engine = Self::new(config);

        if let Some(path) = preprocessor_path {
            let preprocessor = DataPreprocessor::load(path)?;
            engine.preprocessor = Some(Arc::new(preprocessor));
        }

        let model = TrainEngine::load(model_path)?;
        engine.model = Some(Arc::new(model));
        engine.is_loaded = true;

        Ok(engine)
    }

    /// Make predictions on a DataFrame
    pub fn predict(&self, df: &DataFrame) -> Result<Array1<f64>> {
        if !self.is_loaded {
            return Err(KolosalError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(KolosalError::ModelNotFitted)?;

        // Apply preprocessing if available
        let processed = if let Some(ref preprocessor) = self.preprocessor {
            preprocessor.transform(df)?
        } else {
            df.clone()
        };

        // Predict in batches
        if processed.height() <= self.config.batch_size {
            return model.predict(&processed);
        }

        self.predict_batched(&processed, model)
    }

    /// Make predictions on raw feature array
    pub fn predict_array(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_loaded {
            return Err(KolosalError::ModelNotFitted);
        }

        // For array input, we need to convert to DataFrame
        // This is a simplified version - in production, would handle more efficiently
        Err(KolosalError::DataError(
            "Direct array prediction not yet implemented. Use DataFrame input.".to_string(),
        ))
    }

    /// Stream predictions for large datasets
    pub fn predict_streaming<'a>(
        &'a self,
        df: &'a DataFrame,
    ) -> Result<impl Iterator<Item = Result<Array1<f64>>> + 'a> {
        if !self.is_loaded {
            return Err(KolosalError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(KolosalError::ModelNotFitted)?;
        let chunk_size = self.config.stream_chunk_size;
        let n_chunks = (df.height() + chunk_size - 1) / chunk_size;

        Ok((0..n_chunks).map(move |i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(df.height());
            let chunk = df.slice(start as i64, end - start);
            
            // Apply preprocessing
            let processed = if let Some(ref preprocessor) = self.preprocessor {
                preprocessor.transform(&chunk)?
            } else {
                chunk
            };

            model.predict(&processed)
        }))
    }

    /// Get probability predictions (for classification)
    pub fn predict_proba(&self, df: &DataFrame) -> Result<Array2<f64>> {
        if !self.is_loaded {
            return Err(KolosalError::ModelNotFitted);
        }

        // Placeholder - would implement actual probability prediction
        let predictions = self.predict(df)?;
        let n = predictions.len();
        
        // Convert to probability format [p_negative, p_positive]
        let mut proba = Array2::zeros((n, 2));
        for (i, p) in predictions.iter().enumerate() {
            let prob = 1.0 / (1.0 + (-p).exp()); // Sigmoid
            proba[[i, 0]] = 1.0 - prob;
            proba[[i, 1]] = prob;
        }

        Ok(proba)
    }

    fn predict_batched(&self, df: &DataFrame, model: &TrainEngine) -> Result<Array1<f64>> {
        let n_rows = df.height();
        let batch_size = self.config.batch_size;
        let n_batches = (n_rows + batch_size - 1) / batch_size;

        let mut all_predictions = Vec::with_capacity(n_rows);

        for i in 0..n_batches {
            let start = i * batch_size;
            let end = (start + batch_size).min(n_rows);
            let batch = df.slice(start as i64, end - start);
            
            let predictions = model.predict(&batch)?;
            all_predictions.extend(predictions.iter().cloned());
        }

        Ok(Array1::from_vec(all_predictions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{TrainingConfig, TaskType};

    fn create_test_data() -> DataFrame {
        DataFrame::new(vec![
            Series::new("feature1".into(), &[1.0, 2.0, 3.0, 4.0, 5.0]).into(),
            Series::new("feature2".into(), &[2.0, 4.0, 6.0, 8.0, 10.0]).into(),
            Series::new("target".into(), &[3.0, 6.0, 9.0, 12.0, 15.0]).into(),
        ])
        .unwrap()
    }

    #[test]
    fn test_engine_creation() {
        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config);
        assert!(!engine.is_loaded);
    }

    #[test]
    fn test_inference_pipeline() {
        let df = create_test_data();

        // Train a model
        let train_config = TrainingConfig::new(TaskType::Regression, "target");
        let mut model = TrainEngine::new(train_config);
        model.fit(&df).unwrap();

        // Create inference engine
        let config = InferenceConfig::new().with_batch_size(2);
        let engine = InferenceEngine::new(config).with_model(model);

        // Predict
        let predictions = engine.predict(&df).unwrap();
        assert_eq!(predictions.len(), 5);
    }
}
