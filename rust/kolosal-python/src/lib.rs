//! Python bindings for kolosal-core via PyO3
//!
//! This module exposes the Rust implementation to Python, providing
//! high-performance alternatives to the Python implementations.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::{Array1, Array2};

mod preprocessing;
mod training;
mod inference;
mod optimizer;

use preprocessing::PyDataPreprocessor;
use training::PyTrainEngine;
use inference::PyInferenceEngine;
use optimizer::PyHyperOptX;

/// Kolosal AutoML - High-performance machine learning in Rust
#[pymodule]
fn kolosal_automl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__rust_version__", "1.75+")?;

    // Preprocessing
    m.add_class::<PyDataPreprocessor>()?;
    m.add_class::<preprocessing::PyPreprocessingConfig>()?;
    m.add_class::<preprocessing::PyScalerType>()?;
    m.add_class::<preprocessing::PyEncoderType>()?;

    // Training
    m.add_class::<PyTrainEngine>()?;
    m.add_class::<training::PyTrainingConfig>()?;
    m.add_class::<training::PyTaskType>()?;
    m.add_class::<training::PyModelType>()?;

    // Inference
    m.add_class::<PyInferenceEngine>()?;
    m.add_class::<inference::PyInferenceConfig>()?;

    // Optimizer
    m.add_class::<PyHyperOptX>()?;
    m.add_class::<optimizer::PyOptimizationConfig>()?;
    m.add_class::<optimizer::PySearchSpace>()?;

    Ok(())
}
