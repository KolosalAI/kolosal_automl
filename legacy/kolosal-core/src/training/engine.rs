//! Training engine implementation

use crate::error::{KolosalError, Result};
use super::{TrainingConfig, ModelType, TaskType, ModelMetrics};
use super::linear_models::{LinearRegression, LogisticRegression};
use super::decision_tree::DecisionTree;
use super::random_forest::RandomForest;
use super::gradient_boosting::{GradientBoostingRegressor, GradientBoostingClassifier, GradientBoostingConfig};
use super::knn::{KNNClassifier, KNNRegressor, KNNConfig};
use super::neural_network::{MLPClassifier, MLPRegressor, MLPConfig};
use super::naive_bayes::{GaussianNaiveBayes, MultinomialNaiveBayes};
use super::svm::{SVMClassifier, SVMRegressor, SVMConfig, KernelType};
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Enum to hold trained model variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainedModel {
    LinearRegression(LinearRegression),
    LogisticRegression(LogisticRegression),
    DecisionTreeClassifier(DecisionTree),
    DecisionTreeRegressor(DecisionTree),
    RandomForestClassifier(RandomForest),
    RandomForestRegressor(RandomForest),
    GradientBoostingClassifier(GradientBoostingClassifier),
    GradientBoostingRegressor(GradientBoostingRegressor),
    KNNClassifier(KNNClassifier),
    KNNRegressor(KNNRegressor),
    MLPClassifier(MLPClassifier),
    MLPRegressor(MLPRegressor),
    GaussianNaiveBayes(GaussianNaiveBayes),
    MultinomialNaiveBayes(MultinomialNaiveBayes),
    SVMClassifier(SVMClassifier),
    SVMRegressor(SVMRegressor),
}

/// Main training engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainEngine {
    config: TrainingConfig,
    feature_names: Vec<String>,
    model: Option<TrainedModel>,
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
            model: None,
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
        self.train_model(&x_train, &y_train)?;

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
                .filter(|name| name.as_str() != self.config.target_column)
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

    fn train_model(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let model = match (&self.config.task_type, &self.config.model_type) {
            // Regression models
            (TaskType::Regression, ModelType::LinearRegression) => {
                let mut model = LinearRegression::new();
                model.fit(x, y)?;
                TrainedModel::LinearRegression(model)
            }
            (TaskType::Regression, ModelType::DecisionTree) => {
                let mut model = DecisionTree::new_regressor();
                if let Some(depth) = self.config.max_depth {
                    model = model.with_max_depth(depth);
                }
                model.fit(x, y)?;
                TrainedModel::DecisionTreeRegressor(model)
            }
            (TaskType::Regression, ModelType::RandomForest) => {
                let mut model = RandomForest::new_regressor(self.config.n_estimators.unwrap_or(100));
                if let Some(depth) = self.config.max_depth {
                    model = model.with_max_depth(depth);
                }
                if let Some(seed) = self.config.random_seed {
                    model = model.with_random_state(seed);
                }
                model.fit(x, y)?;
                TrainedModel::RandomForestRegressor(model)
            }
            (TaskType::Regression, ModelType::GradientBoosting) => {
                let config = GradientBoostingConfig {
                    n_estimators: self.config.n_estimators.unwrap_or(100),
                    learning_rate: self.config.learning_rate.unwrap_or(0.1),
                    max_depth: self.config.max_depth.unwrap_or(6),
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = GradientBoostingRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::GradientBoostingRegressor(model)
            }
            (TaskType::Regression, ModelType::KNN) => {
                let config = KNNConfig {
                    n_neighbors: self.config.n_neighbors.unwrap_or(5),
                    ..Default::default()
                };
                let mut model = KNNRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::KNNRegressor(model)
            }
            
            // Classification models
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::LogisticRegression) => {
                let mut model = LogisticRegression::new()
                    .with_max_iter(self.config.max_iter.unwrap_or(1000));
                if let Some(lr) = self.config.learning_rate {
                    model = model.with_learning_rate(lr);
                }
                model.fit(x, y)?;
                TrainedModel::LogisticRegression(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::DecisionTree) => {
                let mut model = DecisionTree::new_classifier();
                if let Some(depth) = self.config.max_depth {
                    model = model.with_max_depth(depth);
                }
                model.fit(x, y)?;
                TrainedModel::DecisionTreeClassifier(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::RandomForest) => {
                let mut model = RandomForest::new_classifier(self.config.n_estimators.unwrap_or(100));
                if let Some(depth) = self.config.max_depth {
                    model = model.with_max_depth(depth);
                }
                if let Some(seed) = self.config.random_seed {
                    model = model.with_random_state(seed);
                }
                model.fit(x, y)?;
                TrainedModel::RandomForestClassifier(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::GradientBoosting) => {
                let config = GradientBoostingConfig {
                    n_estimators: self.config.n_estimators.unwrap_or(100),
                    learning_rate: self.config.learning_rate.unwrap_or(0.1),
                    max_depth: self.config.max_depth.unwrap_or(6),
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = GradientBoostingClassifier::new(config);
                model.fit(x, y)?;
                TrainedModel::GradientBoostingClassifier(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::KNN) => {
                let config = KNNConfig {
                    n_neighbors: self.config.n_neighbors.unwrap_or(5),
                    ..Default::default()
                };
                let mut model = KNNClassifier::new(config);
                model.fit(x, y)?;
                TrainedModel::KNNClassifier(model)
            }
            
            // Neural Network models
            (TaskType::Regression, ModelType::NeuralNetwork) => {
                let config = MLPConfig {
                    hidden_layers: vec![100, 50],
                    max_epochs: self.config.max_iter.unwrap_or(200),
                    learning_rate: self.config.learning_rate.unwrap_or(0.001),
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = MLPRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::MLPRegressor(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::NeuralNetwork) => {
                let config = MLPConfig {
                    hidden_layers: vec![100, 50],
                    max_epochs: self.config.max_iter.unwrap_or(200),
                    learning_rate: self.config.learning_rate.unwrap_or(0.001),
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = MLPClassifier::new(config);
                model.fit(x, y)?;
                TrainedModel::MLPClassifier(model)
            }
            
            // Naive Bayes (classification only, use Gaussian variant)
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::NaiveBayes) => {
                let mut model = GaussianNaiveBayes::new();
                model.fit(x, y)?;
                TrainedModel::GaussianNaiveBayes(model)
            }
            
            // SVM models
            (TaskType::Regression, ModelType::SVM) => {
                let config = SVMConfig {
                    c: 1.0,
                    kernel: KernelType::RBF { gamma: 1.0 / x.ncols() as f64 },
                    max_iter: self.config.max_iter.unwrap_or(1000),
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = SVMRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::SVMRegressor(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::SVM) => {
                let config = SVMConfig {
                    c: 1.0,
                    kernel: KernelType::RBF { gamma: 1.0 / x.ncols() as f64 },
                    max_iter: self.config.max_iter.unwrap_or(1000),
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = SVMClassifier::new(config);
                model.fit(x, y)?;
                TrainedModel::SVMClassifier(model)
            }
            
            // Default cases - use appropriate model based on task type
            (TaskType::Regression, _) => {
                let mut model = LinearRegression::new();
                model.fit(x, y)?;
                TrainedModel::LinearRegression(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, _) => {
                let mut model = LogisticRegression::new();
                model.fit(x, y)?;
                TrainedModel::LogisticRegression(model)
            }
            
            // TimeSeries defaults to linear regression for now
            (TaskType::TimeSeries, _) => {
                let mut model = LinearRegression::new();
                model.fit(x, y)?;
                TrainedModel::LinearRegression(model)
            }
        };

        self.model = Some(model);
        Ok(())
    }

    fn predict_internal(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let model = self.model.as_ref().ok_or(KolosalError::ModelNotFitted)?;

        let predictions = match model {
            TrainedModel::LinearRegression(m) => m.predict(x)?,
            TrainedModel::LogisticRegression(m) => m.predict(x)?,
            TrainedModel::DecisionTreeClassifier(m) => m.predict(x)?,
            TrainedModel::DecisionTreeRegressor(m) => m.predict(x)?,
            TrainedModel::RandomForestClassifier(m) => m.predict(x)?,
            TrainedModel::RandomForestRegressor(m) => m.predict(x)?,
            TrainedModel::GradientBoostingClassifier(m) => m.predict(x)?,
            TrainedModel::GradientBoostingRegressor(m) => m.predict(x)?,
            TrainedModel::KNNClassifier(m) => m.predict(x),
            TrainedModel::KNNRegressor(m) => m.predict(x),
            TrainedModel::MLPClassifier(m) => m.predict(x),
            TrainedModel::MLPRegressor(m) => m.predict(x),
            TrainedModel::GaussianNaiveBayes(m) => m.predict(x),
            TrainedModel::MultinomialNaiveBayes(m) => m.predict(x),
            TrainedModel::SVMClassifier(m) => m.predict(x)?,
            TrainedModel::SVMRegressor(m) => m.predict(x)?,
        };

        Ok(predictions)
    }

    /// Get feature importances (if available)
    pub fn feature_importances(&self) -> Option<Array1<f64>> {
        let model = self.model.as_ref()?;
        
        match model {
            TrainedModel::DecisionTreeClassifier(m) | TrainedModel::DecisionTreeRegressor(m) => {
                m.feature_importances().cloned()
            }
            TrainedModel::RandomForestClassifier(m) | TrainedModel::RandomForestRegressor(m) => {
                m.feature_importances().cloned()
            }
            TrainedModel::GradientBoostingRegressor(m) => {
                Some(Array1::from_vec(m.feature_importances().to_vec()))
            }
            TrainedModel::GradientBoostingClassifier(m) => {
                Some(Array1::from_vec(m.feature_importances().to_vec()))
            }
            _ => None,
        }
    }

    /// Get the trained model
    pub fn model(&self) -> Option<&TrainedModel> {
        self.model.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> DataFrame {
        DataFrame::new(vec![
            Series::new("feature1".into(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).into(),
            Series::new("feature2".into(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]).into(),
            Series::new("target".into(), &[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]).into(),
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
