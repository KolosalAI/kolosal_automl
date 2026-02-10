//! Error types for the server

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Polars error: {0}")]
    Polars(#[from] polars::prelude::PolarsError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Training error: {0}")]
    Training(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ServerError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            ServerError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            ServerError::Internal(msg) => {
                tracing::error!(detail = %msg, "Internal server error");
                (StatusCode::INTERNAL_SERVER_ERROR, "An internal error occurred".to_string())
            }
            ServerError::Io(e) => {
                tracing::error!(detail = %e, "IO error");
                (StatusCode::INTERNAL_SERVER_ERROR, "A file system error occurred".to_string())
            }
            ServerError::Polars(e) => {
                let msg = e.to_string();
                // Only expose safe parts of Polars errors
                let safe_msg = if msg.contains("not found") || msg.contains("column") {
                    msg.clone()
                } else {
                    "Data processing error. Check your data format.".to_string()
                };
                (StatusCode::BAD_REQUEST, safe_msg)
            }
            ServerError::Json(e) => (StatusCode::BAD_REQUEST, "Invalid JSON format".to_string()),
            ServerError::Training(msg) => {
                tracing::error!(detail = %msg, "Training error");
                (StatusCode::INTERNAL_SERVER_ERROR, "Training failed. Check server logs for details.".to_string())
            }
        };

        let body = Json(json!({
            "error": true,
            "message": message,
        }));

        (status, body).into_response()
    }
}

pub type Result<T> = std::result::Result<T, ServerError>;
