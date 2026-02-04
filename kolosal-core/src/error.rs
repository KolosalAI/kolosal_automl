//! Error types for the Kolosal AutoML framework

use thiserror::Error;

/// Result type alias for Kolosal operations
pub type Result<T> = std::result::Result<T, KolosalError>;

/// Main error type for the Kolosal framework
#[derive(Error, Debug)]
pub enum KolosalError {
    #[error("Data error: {0}")]
    DataError(String),

    #[error("Preprocessing error: {0}")]
    PreprocessingError(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid shape: expected {expected}, got {actual}")]
    ShapeError { expected: String, actual: String },

    #[error("Feature not found: {0}")]
    FeatureNotFound(String),

    #[error("Model not fitted")]
    ModelNotFitted,

    #[error("Invalid parameter: {name} = {value}, {reason}")]
    InvalidParameter {
        name: String,
        value: String,
        reason: String,
    },

    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceError { iterations: usize },

    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    #[error("Thread pool error: {0}")]
    ThreadPoolError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl From<polars::error::PolarsError> for KolosalError {
    fn from(err: polars::error::PolarsError) -> Self {
        KolosalError::DataError(err.to_string())
    }
}

impl From<serde_json::Error> for KolosalError {
    fn from(err: serde_json::Error) -> Self {
        KolosalError::SerializationError(err.to_string())
    }
}

impl From<ndarray::ShapeError> for KolosalError {
    fn from(err: ndarray::ShapeError) -> Self {
        KolosalError::ShapeError {
            expected: "valid shape".to_string(),
            actual: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = KolosalError::DataError("test error".to_string());
        assert_eq!(err.to_string(), "Data error: test error");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: KolosalError = io_err.into();
        assert!(matches!(err, KolosalError::IoError(_)));
    }
}
