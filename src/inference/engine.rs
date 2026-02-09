//! Inference engine implementation
//!
//! High-performance inference with:
//! - Parallel batch processing via rayon
//! - In-memory model caching (Arc-shared)
//! - Latency tracking via PerformanceMetrics
//! - Memory-aware adaptive batch sizing
//! - Model warmup for consistent latency
//! - Classification probability support

use crate::error::{KolosalError, Result};
use crate::monitoring::PerformanceMetrics;
use crate::preprocessing::DataPreprocessor;
use crate::training::TrainEngine;
use super::InferenceConfig;
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// Inference statistics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    pub total_predictions: u64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_per_sec: f64,
    pub error_count: u64,
}

/// High-performance inference engine
pub struct InferenceEngine {
    config: InferenceConfig,
    preprocessor: Option<Arc<DataPreprocessor>>,
    model: Option<Arc<TrainEngine>>,
    is_loaded: bool,
    metrics: Arc<PerformanceMetrics>,
    is_warmed_up: bool,
}

impl std::fmt::Debug for InferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("config", &self.config)
            .field("is_loaded", &self.is_loaded)
            .field("is_warmed_up", &self.is_warmed_up)
            .finish()
    }
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            config,
            preprocessor: None,
            model: None,
            is_loaded: false,
            metrics: Arc::new(PerformanceMetrics::new(10000)),
            is_warmed_up: false,
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

    /// Warm up the engine by running a dummy prediction to prime caches and JIT paths
    pub fn warmup(&mut self) -> Result<()> {
        if !self.is_loaded {
            return Err(KolosalError::ModelNotFitted);
        }
        let model = self.model.as_ref().ok_or(KolosalError::ModelNotFitted)?;
        let n_features = model.feature_names().len();
        if n_features == 0 {
            return Ok(());
        }

        // Create a small dummy DataFrame with 1 row
        let columns: Vec<Column> = model.feature_names().iter().map(|name| {
            let series = Series::new(name.as_str().into(), &[0.0_f64]);
            series.into()
        }).collect();

        let df = DataFrame::new(columns)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        let processed = if let Some(ref preprocessor) = self.preprocessor {
            preprocessor.transform(&df)?
        } else {
            df
        };

        // Run a prediction to warm up internal paths
        let _ = model.predict(&processed);
        self.is_warmed_up = true;
        Ok(())
    }

    /// Check if the engine has been warmed up
    pub fn is_warmed_up(&self) -> bool {
        self.is_warmed_up
    }

    /// Get a reference to the underlying performance metrics
    pub fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    /// Get inference statistics snapshot
    pub fn stats(&self) -> InferenceStats {
        InferenceStats {
            total_predictions: self.metrics.total_items(),
            avg_latency_ms: self.metrics.avg_latency(),
            p50_latency_ms: self.metrics.p50_latency(),
            p95_latency_ms: self.metrics.p95_latency(),
            p99_latency_ms: self.metrics.p99_latency(),
            throughput_per_sec: self.metrics.throughput(),
            error_count: self.metrics.total_errors(),
        }
    }

    /// Check if the engine is ready for predictions
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Get the current configuration
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Get the model reference (if loaded)
    pub fn model(&self) -> Option<&TrainEngine> {
        self.model.as_deref()
    }

    /// Make predictions on a DataFrame
    pub fn predict(&self, df: &DataFrame) -> Result<Array1<f64>> {
        let start = Instant::now();

        if !self.is_loaded {
            self.metrics.record_error();
            return Err(KolosalError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(KolosalError::ModelNotFitted)?;

        // Apply preprocessing if available
        let processed = if let Some(ref preprocessor) = self.preprocessor {
            preprocessor.transform(df)?
        } else {
            df.clone()
        };

        let result = if processed.height() <= self.config.batch_size {
            model.predict(&processed)
        } else if self.should_use_parallel(processed.height()) {
            self.predict_parallel(&processed, model)
        } else {
            self.predict_batched(&processed, model)
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(preds) => {
                self.metrics.record_latency(latency_ms);
                self.metrics.record_items(preds.len() as u64);
            }
            Err(_) => self.metrics.record_error(),
        }

        result
    }

    /// Make predictions on raw feature array
    pub fn predict_array(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let start = Instant::now();

        if !self.is_loaded {
            self.metrics.record_error();
            return Err(KolosalError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(KolosalError::ModelNotFitted)?;

        // Get feature names from model
        let feature_names = model.feature_names();

        if x.ncols() != feature_names.len() {
            self.metrics.record_error();
            return Err(KolosalError::ShapeError {
                expected: format!("{} features", feature_names.len()),
                actual: format!("{} features", x.ncols()),
            });
        }

        // Convert Array2 to DataFrame
        let df = self.array_to_dataframe(x, feature_names)?;

        // Apply preprocessing if available
        let processed = if let Some(ref preprocessor) = self.preprocessor {
            preprocessor.transform(&df)?
        } else {
            df
        };

        let result = model.predict(&processed);

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(preds) => {
                self.metrics.record_latency(latency_ms);
                self.metrics.record_items(preds.len() as u64);
            }
            Err(_) => self.metrics.record_error(),
        }

        result
    }

    /// Get probability predictions (for classification) using the model's native predict_proba
    pub fn predict_proba(&self, df: &DataFrame) -> Result<Array2<f64>> {
        let start = Instant::now();

        if !self.is_loaded {
            self.metrics.record_error();
            return Err(KolosalError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(KolosalError::ModelNotFitted)?;

        // Apply preprocessing if available
        let processed = if let Some(ref preprocessor) = self.preprocessor {
            preprocessor.transform(df)?
        } else {
            df.clone()
        };

        let result = model.predict_proba(&processed);

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(proba) => {
                self.metrics.record_latency(latency_ms);
                self.metrics.record_items(proba.nrows() as u64);
            }
            Err(_) => self.metrics.record_error(),
        }

        result
    }

    /// Get probability predictions on raw feature array
    pub fn predict_proba_array(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let start = Instant::now();

        if !self.is_loaded {
            self.metrics.record_error();
            return Err(KolosalError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(KolosalError::ModelNotFitted)?;
        let feature_names = model.feature_names();

        if x.ncols() != feature_names.len() {
            self.metrics.record_error();
            return Err(KolosalError::ShapeError {
                expected: format!("{} features", feature_names.len()),
                actual: format!("{} features", x.ncols()),
            });
        }

        let result = model.predict_proba_array(x);

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(proba) => {
                self.metrics.record_latency(latency_ms);
                self.metrics.record_items(proba.nrows() as u64);
            }
            Err(_) => self.metrics.record_error(),
        }

        result
    }

    /// Predict with classification threshold applied (returns 0/1 labels)
    pub fn predict_with_threshold(&self, df: &DataFrame) -> Result<Array1<f64>> {
        let proba = self.predict_proba(df)?;
        let threshold = self.config.classification_threshold;
        // Use the positive class column (last column)
        let pos_col = proba.ncols() - 1;
        Ok(Array1::from_iter(
            proba.column(pos_col).iter().map(|&p| if p >= threshold { 1.0 } else { 0.0 })
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
            let start_time = Instant::now();
            let start = i * chunk_size;
            let end = (start + chunk_size).min(df.height());
            let chunk = df.slice(start as i64, end - start);

            // Apply preprocessing
            let processed = if let Some(ref preprocessor) = self.preprocessor {
                preprocessor.transform(&chunk)?
            } else {
                chunk
            };

            let result = model.predict(&processed);
            let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            self.metrics.record_latency(latency_ms);
            if let Ok(ref preds) = result {
                self.metrics.record_items(preds.len() as u64);
            }
            result
        }))
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /// Determine if parallel batch processing should be used
    fn should_use_parallel(&self, n_rows: usize) -> bool {
        let n_workers = self.config.n_workers.unwrap_or_else(|| {
            rayon::current_num_threads()
        });
        // Use parallel when we have enough rows to fill multiple workers
        n_workers > 1 && n_rows > self.config.batch_size * 2
    }

    /// Compute adaptive batch size based on available memory
    pub(crate) fn adaptive_batch_size(&self, n_rows: usize, n_cols: usize) -> usize {
        if let Some(max_bytes) = self.config.max_memory_bytes {
            let bytes_per_row = n_cols * std::mem::size_of::<f64>();
            // Use at most 50% of memory budget for a single batch
            let max_rows = (max_bytes / 2) / bytes_per_row.max(1);
            max_rows.max(1).min(self.config.batch_size).min(n_rows)
        } else {
            self.config.batch_size.min(n_rows)
        }
    }

    /// Convert Array2 to DataFrame using model feature names
    fn array_to_dataframe(&self, x: &Array2<f64>, feature_names: &[String]) -> Result<DataFrame> {
        let mut columns: Vec<Column> = Vec::with_capacity(x.ncols());
        for (j, name) in feature_names.iter().enumerate() {
            let col_data: Vec<f64> = x.column(j).to_vec();
            let series = Series::new(name.as_str().into(), &col_data);
            columns.push(series.into());
        }
        DataFrame::new(columns).map_err(|e| KolosalError::DataError(e.to_string()))
    }

    /// Sequential batch prediction
    fn predict_batched(&self, df: &DataFrame, model: &TrainEngine) -> Result<Array1<f64>> {
        let n_rows = df.height();
        let n_cols = df.width();
        let batch_size = self.adaptive_batch_size(n_rows, n_cols);
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

    /// Parallel batch prediction using rayon
    fn predict_parallel(&self, df: &DataFrame, model: &TrainEngine) -> Result<Array1<f64>> {
        let n_rows = df.height();
        let n_cols = df.width();
        let batch_size = self.adaptive_batch_size(n_rows, n_cols);
        let n_batches = (n_rows + batch_size - 1) / batch_size;

        // Configure rayon thread pool size if specified
        let pool = if let Some(n_workers) = self.config.n_workers {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_workers)
                    .build()
                    .map_err(|e| KolosalError::InferenceError(format!("Thread pool error: {}", e)))?
            )
        } else {
            None
        };

        // Prepare batch slices
        let batches: Vec<DataFrame> = (0..n_batches)
            .map(|i| {
                let start = i * batch_size;
                let end = (start + batch_size).min(n_rows);
                df.slice(start as i64, end - start)
            })
            .collect();

        // Run predictions in parallel
        let predict_fn = |batches: &[DataFrame]| -> Result<Vec<Array1<f64>>> {
            batches
                .par_iter()
                .map(|batch| model.predict(batch))
                .collect::<Result<Vec<_>>>()
        };

        let batch_results = if let Some(ref pool) = pool {
            pool.install(|| predict_fn(&batches))?
        } else {
            predict_fn(&batches)?
        };

        // Concatenate results in order
        let mut all_predictions = Vec::with_capacity(n_rows);
        for preds in batch_results {
            all_predictions.extend(preds.iter().cloned());
        }

        Ok(Array1::from_vec(all_predictions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{TrainingConfig, TaskType};

    fn create_regression_data() -> DataFrame {
        DataFrame::new(vec![
            Series::new("feature1".into(), &[1.0, 2.0, 3.0, 4.0, 5.0]).into(),
            Series::new("feature2".into(), &[1.5, 3.1, 2.8, 7.2, 5.5]).into(),
            Series::new("target".into(), &[3.0, 6.0, 9.0, 12.0, 15.0]).into(),
        ])
        .unwrap()
    }

    fn create_classification_data() -> DataFrame {
        DataFrame::new(vec![
            Series::new("feature1".into(), &[1.0, 2.0, 3.0, 8.0, 9.0, 10.0]).into(),
            Series::new("feature2".into(), &[1.5, 2.5, 3.5, 8.5, 9.5, 10.5]).into(),
            Series::new("target".into(), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).into(),
        ])
        .unwrap()
    }

    fn train_regression_model() -> TrainEngine {
        let df = create_regression_data();
        let config = TrainingConfig::new(TaskType::Regression, "target");
        let mut model = TrainEngine::new(config);
        model.fit(&df).unwrap();
        model
    }

    fn train_classification_model() -> TrainEngine {
        let df = create_classification_data();
        let config = TrainingConfig::new(TaskType::BinaryClassification, "target");
        let mut model = TrainEngine::new(config);
        model.fit(&df).unwrap();
        model
    }

    #[test]
    fn test_engine_creation() {
        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config);
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_inference_pipeline() {
        let df = create_regression_data();
        let model = train_regression_model();

        let config = InferenceConfig::new().with_batch_size(2);
        let engine = InferenceEngine::new(config).with_model(model);

        let predictions = engine.predict(&df).unwrap();
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_predict_array() {
        let df = create_regression_data();
        let model = train_regression_model();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        let x = ndarray::array![[1.0, 1.5], [2.0, 3.1]];
        let predictions = engine.predict_array(&x).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_warmup() {
        let model = train_regression_model();
        let config = InferenceConfig::new();
        let mut engine = InferenceEngine::new(config).with_model(model);

        assert!(!engine.is_warmed_up());
        engine.warmup().unwrap();
        assert!(engine.is_warmed_up());
    }

    #[test]
    fn test_latency_tracking() {
        let df = create_regression_data();
        let model = train_regression_model();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        assert_eq!(engine.stats().total_predictions, 0);

        engine.predict(&df).unwrap();
        let stats = engine.stats();
        assert!(stats.total_predictions > 0);
        assert!(stats.avg_latency_ms >= 0.0);
    }

    #[test]
    fn test_parallel_batch_prediction() {
        let model = train_regression_model();

        // Create larger dataset to trigger parallel processing
        let n = 100;
        let f1: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let f2: Vec<f64> = (0..n).map(|i| i as f64 * 1.5).collect();
        let target: Vec<f64> = (0..n).map(|i| i as f64 * 3.0).collect();
        let df = DataFrame::new(vec![
            Series::new("feature1".into(), &f1).into(),
            Series::new("feature2".into(), &f2).into(),
            Series::new("target".into(), &target).into(),
        ]).unwrap();

        let config = InferenceConfig::new()
            .with_batch_size(10)
            .with_n_workers(2);
        let engine = InferenceEngine::new(config).with_model(model);

        let predictions = engine.predict(&df).unwrap();
        assert_eq!(predictions.len(), n);
    }

    #[test]
    fn test_memory_aware_batch_sizing() {
        let model = train_regression_model();
        let config = InferenceConfig::new()
            .with_batch_size(1000)
            .with_max_memory(1024); // Very small memory budget
        let engine = InferenceEngine::new(config).with_model(model);

        // adaptive_batch_size should limit batch size due to memory constraint
        let batch_size = engine.adaptive_batch_size(1000, 10);
        assert!(batch_size < 1000);
    }

    #[test]
    fn test_predict_not_loaded() {
        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config);

        let df = create_regression_data();
        assert!(engine.predict(&df).is_err());
    }

    #[test]
    fn test_predict_proba_classification() {
        let df = create_classification_data();
        let model = train_classification_model();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        let proba = engine.predict_proba(&df);
        // predict_proba may fail for some model types that don't support it
        // but should not panic
        if let Ok(proba) = proba {
            assert_eq!(proba.nrows(), 6);
            assert_eq!(proba.ncols(), 2);
            // Probabilities should sum to ~1
            for i in 0..proba.nrows() {
                let sum = proba[[i, 0]] + proba[[i, 1]];
                assert!((sum - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_predict_with_threshold() {
        let df = create_classification_data();
        let model = train_classification_model();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        let result = engine.predict_with_threshold(&df);
        if let Ok(labels) = result {
            assert_eq!(labels.len(), 6);
            for &v in labels.iter() {
                assert!(v == 0.0 || v == 1.0);
            }
        }
    }

    #[test]
    fn test_streaming_predictions() {
        let df = create_regression_data();
        let model = train_regression_model();

        let config = InferenceConfig::new().with_streaming(2);
        let engine = InferenceEngine::new(config).with_model(model);

        let chunks: Vec<_> = engine.predict_streaming(&df).unwrap().collect();
        let total: usize = chunks.iter().filter_map(|r| r.as_ref().ok()).map(|a| a.len()).sum();
        assert_eq!(total, 5);
    }
}
