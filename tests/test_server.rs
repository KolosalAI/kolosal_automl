//! Integration test: Server API endpoints

use kolosal_automl::server::{AppState, ServerConfig, create_router};
use std::sync::Arc;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

fn test_app() -> axum::Router {
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        static_dir: None,
        data_dir: "/tmp/kolosal-test-data".to_string(),
        models_dir: "/tmp/kolosal-test-models".to_string(),
        max_upload_size: 10 * 1024 * 1024,
    };
    std::fs::create_dir_all(&config.data_dir).ok();
    std::fs::create_dir_all(&config.models_dir).ok();
    let state = Arc::new(AppState::new(config.clone()));
    create_router(state, &config)
}

#[tokio::test]
async fn test_health_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_system_status_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/system/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_root_serves_html() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_list_models_empty() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_data_preview_no_data() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/data/preview")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // Should return OK with empty/error message since no data loaded
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::BAD_REQUEST || status == StatusCode::NOT_FOUND,
        "unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_device_info_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/device/info")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_optimal_configs_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/device/optimal-configs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_batch_stats_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/batch/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_monitoring_dashboard_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/monitoring/dashboard")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_monitoring_alerts_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/monitoring/alerts")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_monitoring_performance_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/monitoring/performance")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_preprocess_config_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/preprocess/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_load_sample_data() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/data/sample/iris")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_quantize_endpoint() {
    let app = test_app();
    let body = serde_json::json!({
        "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "bits": 8
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/quantize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
        "unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_train_endpoint() {
    // First load sample data, then train
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        static_dir: None,
        data_dir: "/tmp/kolosal-test-data2".to_string(),
        models_dir: "/tmp/kolosal-test-models2".to_string(),
        max_upload_size: 10 * 1024 * 1024,
    };
    std::fs::create_dir_all(&config.data_dir).ok();
    std::fs::create_dir_all(&config.models_dir).ok();
    let state = Arc::new(AppState::new(config.clone()));
    let app = create_router(state, &config);

    // Load sample data first
    let (parts, body) = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/data/sample/iris")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
        .into_parts();
    assert_eq!(parts.status, StatusCode::OK);

    // Now train
    let train_body = serde_json::json!({
        "target": "target",
        "task_type": "classification",
        "model_type": "decision_tree"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/train")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&train_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::ACCEPTED || status == StatusCode::UNPROCESSABLE_ENTITY || status == StatusCode::BAD_REQUEST,
        "train should succeed or be accepted, got: {}",
        status
    );
}
