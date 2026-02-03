//! Python bindings for training module

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use polars::prelude::NamedFrom;
use kolosal_core::training::{TrainEngine, TrainingConfig, TaskType, ModelType};

/// Task type enum for Python
#[pyclass(name = "TaskType")]
#[derive(Clone)]
pub enum PyTaskType {
    BinaryClassification,
    MultiClassification,
    Regression,
    TimeSeries,
}

impl From<PyTaskType> for TaskType {
    fn from(py: PyTaskType) -> Self {
        match py {
            PyTaskType::BinaryClassification => TaskType::BinaryClassification,
            PyTaskType::MultiClassification => TaskType::MultiClassification,
            PyTaskType::Regression => TaskType::Regression,
            PyTaskType::TimeSeries => TaskType::TimeSeries,
        }
    }
}

/// Model type enum for Python
#[pyclass(name = "ModelType")]
#[derive(Clone)]
pub enum PyModelType {
    DecisionTree,
    RandomForest,
    GradientBoosting,
    XGBoost,
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SVM,
    KNN,
    NeuralNetwork,
    Auto,
}

impl From<PyModelType> for ModelType {
    fn from(py: PyModelType) -> Self {
        match py {
            PyModelType::DecisionTree => ModelType::DecisionTree,
            PyModelType::RandomForest => ModelType::RandomForest,
            PyModelType::GradientBoosting => ModelType::GradientBoosting,
            PyModelType::XGBoost => ModelType::XGBoost,
            PyModelType::LinearRegression => ModelType::LinearRegression,
            PyModelType::LogisticRegression => ModelType::LogisticRegression,
            PyModelType::Ridge => ModelType::Ridge,
            PyModelType::Lasso => ModelType::Lasso,
            PyModelType::ElasticNet => ModelType::ElasticNet,
            PyModelType::SVM => ModelType::SVM,
            PyModelType::KNN => ModelType::KNN,
            PyModelType::NeuralNetwork => ModelType::NeuralNetwork,
            PyModelType::Auto => ModelType::Auto,
        }
    }
}

/// Training configuration for Python
#[pyclass(name = "TrainingConfig")]
#[derive(Clone)]
pub struct PyTrainingConfig {
    inner: TrainingConfig,
}

#[pymethods]
impl PyTrainingConfig {
    #[new]
    fn new(task_type: PyTaskType, target: String) -> Self {
        Self {
            inner: TrainingConfig::new(task_type.into(), target),
        }
    }

    /// Set model type
    fn with_model(&self, model_type: PyModelType) -> Self {
        Self {
            inner: self.inner.clone().with_model(model_type.into()),
        }
    }

    /// Set number of estimators
    fn with_n_estimators(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().with_n_estimators(n),
        }
    }

    /// Set learning rate
    fn with_learning_rate(&self, lr: f64) -> Self {
        Self {
            inner: self.inner.clone().with_learning_rate(lr),
        }
    }

    /// Set max depth
    fn with_max_depth(&self, depth: usize) -> Self {
        Self {
            inner: self.inner.clone().with_max_depth(depth),
        }
    }

    /// Set CV folds
    fn with_cv(&self, folds: usize) -> Self {
        Self {
            inner: self.inner.clone().with_cv(folds),
        }
    }

    /// Set random state
    fn with_random_state(&self, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().with_random_state(seed),
        }
    }
}

/// Train engine for Python
#[pyclass(name = "TrainEngine")]
pub struct PyTrainEngine {
    inner: TrainEngine,
}

#[pymethods]
impl PyTrainEngine {
    #[new]
    fn new(config: PyTrainingConfig) -> Self {
        Self {
            inner: TrainEngine::new(config.inner),
        }
    }

    /// Fit the model to a pandas DataFrame
    fn fit(&mut self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<()> {
        let polars_df = python_df_to_polars(py, df)?;
        self.inner
            .fit(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Make predictions on a pandas DataFrame
    fn predict(&self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let polars_df = python_df_to_polars(py, df)?;
        let predictions = self
            .inner
            .predict(&polars_df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(predictions.to_vec())
    }

    /// Get training metrics
    fn metrics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        
        if let Some(metrics) = self.inner.metrics() {
            if let Some(v) = metrics.accuracy {
                dict.set_item("accuracy", v)?;
            }
            if let Some(v) = metrics.precision {
                dict.set_item("precision", v)?;
            }
            if let Some(v) = metrics.recall {
                dict.set_item("recall", v)?;
            }
            if let Some(v) = metrics.f1_score {
                dict.set_item("f1_score", v)?;
            }
            if let Some(v) = metrics.mse {
                dict.set_item("mse", v)?;
            }
            if let Some(v) = metrics.rmse {
                dict.set_item("rmse", v)?;
            }
            if let Some(v) = metrics.mae {
                dict.set_item("mae", v)?;
            }
            if let Some(v) = metrics.r2 {
                dict.set_item("r2", v)?;
            }
            dict.set_item("training_time_secs", metrics.training_time_secs)?;
            dict.set_item("n_features", metrics.n_features)?;
            dict.set_item("n_samples", metrics.n_samples)?;
        }
        
        Ok(dict.into())
    }

    /// Get feature names
    fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names().to_vec()
    }

    /// Save engine to file
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load engine from file
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = TrainEngine::load(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

/// Convert Python DataFrame (pandas) to Polars DataFrame
fn python_df_to_polars(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<polars::prelude::DataFrame> {
    let columns: Vec<String> = df.getattr("columns")?.extract()?;
    
    let mut col_vec = Vec::new();
    
    for col in &columns {
        let col_data = df.get_item(col)?;
        let values: Vec<f64> = col_data.call_method0("tolist")?.extract().unwrap_or_default();
        let series = polars::prelude::Series::new(col.clone().into(), values);
        col_vec.push(series.into());
    }
    
    polars::prelude::DataFrame::new(col_vec)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
