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
            ServerError::BadRequest(msg) => (StatusCode::BAD_REQUEST, sanitize_error(msg)),
            ServerError::NotFound(msg) => (StatusCode::NOT_FOUND, sanitize_error(msg)),
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
                    sanitize_error(&msg)
                } else {
                    "Data processing error. Check your data format.".to_string()
                };
                (StatusCode::BAD_REQUEST, safe_msg)
            }
            ServerError::Json(_e) => (StatusCode::BAD_REQUEST, "Invalid JSON format".to_string()),
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

/// Sanitize error messages to avoid leaking internal details to clients.
fn sanitize_error(msg: &str) -> String {
    if msg.contains('/') || msg.contains("src/") || msg.contains("thread") || msg.contains("panicked") {
        "An internal error occurred. Please check the server logs for details.".to_string()
    } else if msg.len() > 200 {
        let end = msg.char_indices().nth(200).map(|(i, _)| i).unwrap_or(msg.len());
        format!("{}...", &msg[..end])
    } else {
        msg.to_string()
    }
}

pub type Result<T> = std::result::Result<T, ServerError>;
