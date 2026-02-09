//! API route definitions

use std::sync::Arc;
use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};

use super::{handlers, state::AppState, ServerConfig};

async fn handle_404() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": true,
            "message": "Not found. Visit / for the web UI or /api/health to check API status.",
        })),
    )
}

async fn handle_405() -> impl IntoResponse {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        Json(json!({
            "error": true,
            "message": "Method not allowed. Check the API documentation for supported methods.",
        })),
    )
}

/// Create the main application router
pub fn create_router(state: Arc<AppState>, config: &ServerConfig) -> Router {
    // API routes
    let api_routes = Router::new()
        // Data endpoints
        .route("/data/upload", post(handlers::upload_data))
        .route("/data/preview", get(handlers::get_data_preview))
        .route("/data/info", get(handlers::get_data_info))
        .route("/data/sample/:name", get(handlers::load_sample_data))
        .route("/data/clean", post(handlers::auto_clean_data))
        .route("/data/import/kaggle", post(handlers::import_kaggle))
        // Preprocessing
        .route("/preprocess", post(handlers::run_preprocessing))
        .route("/preprocess/config", get(handlers::get_preprocessing_config))
        // Training
        .route("/train", post(handlers::start_training))
        .route("/train/status/:job_id", get(handlers::get_training_status))
        .route("/train/compare", post(handlers::compare_models))
        // Models
        .route("/models", get(handlers::list_models))
        .route("/models/:model_id", get(handlers::get_model).delete(handlers::delete_model))
        .route("/models/:model_id/download", get(handlers::download_model))
        // Inference
        .route("/predict", post(handlers::predict))
        .route("/predict/batch", post(handlers::predict_batch))
        .route("/predict/proba", post(handlers::predict_proba))
        .route("/predict/stats/:model_id", get(handlers::get_inference_stats))
        .route("/predict/cache/:model_id", axum::routing::delete(handlers::evict_model_cache))
        // System
        .route("/system/status", get(handlers::get_system_status))
        .route("/health", get(handlers::health_check))
        // Batch processor
        .route("/batch/config", post(handlers::configure_batch_processor))
        .route("/batch/stats", get(handlers::get_batch_stats))
        // Quantizer
        .route("/quantize", post(handlers::quantize_data))
        .route("/dequantize", post(handlers::dequantize_data))
        .route("/quantize/calibrate", post(handlers::calibrate_quantizer))
        // Device optimizer
        .route("/device/info", get(handlers::get_device_info))
        .route("/device/optimal-configs", get(handlers::get_optimal_configs))
        // Model management
        .route("/models/:model_id/metrics", get(handlers::get_model_metrics))
        // Monitoring
        .route("/monitoring/dashboard", get(handlers::get_monitoring_dashboard))
        .route("/monitoring/alerts", get(handlers::get_alerts))
        .route("/monitoring/performance", get(handlers::get_performance_analysis))
        .fallback(handle_404)
        .method_not_allowed_fallback(handle_405);

    // Build main router
    let mut app = Router::new()
        .nest("/api", api_routes)
        .route("/", get(handlers::serve_index))
        .fallback(handle_404)
        .method_not_allowed_fallback(handle_405)
        .with_state(state);

    // Serve static files if directory exists
    if let Some(ref static_dir) = config.static_dir {
        let static_path = std::path::Path::new(static_dir);
        if static_path.exists() {
            app = app.nest_service("/static", ServeDir::new(static_path));
        } else {
            // Fallback: try CARGO_MANIFEST_DIR-relative path
            let manifest_path = format!("{}/kolosal-web/static", env!("CARGO_MANIFEST_DIR"));
            let fallback_path = std::path::Path::new(&manifest_path);
            if fallback_path.exists() {
                app = app.nest_service("/static", ServeDir::new(fallback_path));
            }
        }
    }

    // Add middleware
    app.layer(CompressionLayer::new())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
}
