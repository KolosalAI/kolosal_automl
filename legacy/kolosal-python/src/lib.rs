//! Python bindings for kolosal-core via PyO3
//!
//! This module exposes the Rust implementation to Python, providing
//! high-performance alternatives to the Python implementations.

use pyo3::prelude::*;

// Shared utilities
mod utils;

// Core modules
mod preprocessing;
mod training;
mod inference;
mod optimizer;

// Feature selection
mod feature_selector;

// Explainability
mod explainability;

// Ensemble methods
mod ensemble;

// Anomaly detection
mod anomaly;

// Probability calibration
mod calibration;

// Time series
mod timeseries;

// Drift detection
mod drift;

// Feature engineering
mod feature_engineering;

// Imputation
mod imputation;

// Synthetic data / oversampling
mod synthetic;

/// Kolosal AutoML - High-performance machine learning in Rust
#[pymodule]
fn kolosal_automl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__rust_version__", "1.75+")?;

    // ===== Preprocessing =====
    m.add_class::<preprocessing::PyDataPreprocessor>()?;
    m.add_class::<preprocessing::PyPreprocessingConfig>()?;
    m.add_class::<preprocessing::PyScalerType>()?;
    m.add_class::<preprocessing::PyEncoderType>()?;
    m.add_class::<feature_selector::PyFeatureSelector>()?;
    m.add_class::<feature_selector::PySelectionMethod>()?;

    // ===== Training =====
    m.add_class::<training::PyTrainEngine>()?;
    m.add_class::<training::PyTrainingConfig>()?;
    m.add_class::<training::PyTaskType>()?;
    m.add_class::<training::PyModelType>()?;

    // ===== Inference =====
    m.add_class::<inference::PyInferenceEngine>()?;
    m.add_class::<inference::PyInferenceConfig>()?;

    // ===== Optimization =====
    m.add_class::<optimizer::PyHyperOptX>()?;
    m.add_class::<optimizer::PyOptimizationConfig>()?;
    m.add_class::<optimizer::PySearchSpace>()?;

    // ===== Explainability =====
    m.add_class::<explainability::PyPermutationImportance>()?;
    m.add_class::<explainability::PyPartialDependence>()?;
    m.add_class::<explainability::PyLocalExplainer>()?;
    m.add_class::<explainability::PyImportanceResult>()?;
    m.add_class::<explainability::PyPDPResult>()?;
    m.add_class::<explainability::PyLocalExplanation>()?;

    // ===== Ensemble =====
    m.add_class::<ensemble::PyVotingClassifier>()?;
    m.add_class::<ensemble::PyVotingRegressor>()?;
    m.add_class::<ensemble::PyVotingType>()?;
    m.add_class::<ensemble::PyStackingConfig>()?;
    m.add_class::<ensemble::PyStackingClassifier>()?;
    m.add_class::<ensemble::PyStackingRegressor>()?;

    // ===== Anomaly Detection =====
    m.add_class::<anomaly::PyIsolationForest>()?;
    m.add_class::<anomaly::PyLocalOutlierFactor>()?;

    // ===== Calibration =====
    m.add_class::<calibration::PyPlattScaling>()?;
    m.add_class::<calibration::PyIsotonicRegression>()?;
    m.add_class::<calibration::PyTemperatureScaling>()?;
    m.add_class::<calibration::PyCalibratorType>()?;

    // ===== Time Series =====
    m.add_class::<timeseries::PyTimeSeriesFeatures>()?;
    m.add_class::<timeseries::PyTimeSeriesCV>()?;
    m.add_class::<timeseries::PyDifferencer>()?;

    // ===== Drift Detection =====
    m.add_class::<drift::PyDataDriftDetector>()?;
    m.add_class::<drift::PyConceptDriftDetector>()?;
    m.add_class::<drift::PyFeatureDriftMonitor>()?;

    // ===== Feature Engineering =====
    m.add_class::<feature_engineering::PyPolynomialFeatures>()?;
    m.add_class::<feature_engineering::PyFeatureInteractions>()?;
    m.add_class::<feature_engineering::PyTfidfVectorizer>()?;
    m.add_class::<feature_engineering::PyFeatureHasher>()?;

    // ===== Imputation =====
    m.add_class::<imputation::PyKNNImputer>()?;
    m.add_class::<imputation::PyMICEImputer>()?;
    m.add_class::<imputation::PyIterativeImputer>()?;
    m.add_class::<imputation::PySimpleImputer>()?;

    // ===== Synthetic Data / Oversampling =====
    m.add_class::<synthetic::PySMOTE>()?;
    m.add_class::<synthetic::PyADASYN>()?;
    m.add_class::<synthetic::PyRandomOverSampler>()?;
    m.add_class::<synthetic::PyRandomUnderSampler>()?;

    Ok(())
}
