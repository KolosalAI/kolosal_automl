//! Integration test: Web UI structure, tabs, and serve tab

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
        data_dir: "/tmp/kolosal-test-webui-data".to_string(),
        models_dir: "/tmp/kolosal-test-webui-models".to_string(),
        max_upload_size: 10 * 1024 * 1024,
    };
    std::fs::create_dir_all(&config.data_dir).ok();
    std::fs::create_dir_all(&config.models_dir).ok();
    let state = Arc::new(AppState::new(config.clone()));
    create_router(state, &config)
}

async fn get_index_html() -> String {
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
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

// ============================================================================
// Tab Structure Tests
// ============================================================================

#[tokio::test]
async fn test_index_has_seven_navigation_tabs() {
    let html = get_index_html().await;
    let expected_tabs = ["data", "automl", "train", "predict", "analyze", "serve", "system"];
    for tab in &expected_tabs {
        assert!(
            html.contains(&format!("data-tab=\"{}\"", tab)),
            "Missing nav tab: {}",
            tab
        );
    }
}

#[tokio::test]
async fn test_index_has_seven_tab_panels() {
    let html = get_index_html().await;
    let expected_panels = [
        "tab-data", "tab-automl", "tab-train", "tab-predict",
        "tab-analyze", "tab-serve", "tab-system",
    ];
    for panel in &expected_panels {
        assert!(
            html.contains(&format!("id=\"{}\"", panel)),
            "Missing tab panel: {}",
            panel
        );
    }
}

#[tokio::test]
async fn test_data_tab_is_active_by_default() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"tab-data\" class=\"tab-panel active\""));
}

// ============================================================================
// Serve Tab Tests
// ============================================================================

#[tokio::test]
async fn test_serve_tab_has_model_list_section() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"serve-models-list\""), "Serve tab missing models list element");
    assert!(html.contains("id=\"serve-empty\""), "Serve tab missing empty state element");
}

#[tokio::test]
async fn test_serve_tab_has_quick_test_section() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"serve-test-model\""), "Serve tab missing model select");
    assert!(html.contains("id=\"serve-test-input\""), "Serve tab missing test input textarea");
    assert!(html.contains("id=\"serve-test-result\""), "Serve tab missing test result area");
    assert!(html.contains("id=\"serve-test-output\""), "Serve tab missing test output pre");
}

#[tokio::test]
async fn test_serve_tab_has_api_documentation() {
    let html = get_index_html().await;
    // Must document all prediction endpoints
    assert!(html.contains("/api/predict"), "API docs missing /api/predict");
    assert!(html.contains("/api/predict/batch"), "API docs missing /api/predict/batch");
    assert!(html.contains("/api/predict/proba"), "API docs missing /api/predict/proba");
    assert!(html.contains("/api/predict/stats/"), "API docs missing /api/predict/stats");
    assert!(html.contains("/api/models"), "API docs missing /api/models");
    assert!(html.contains("/api/health"), "API docs missing /api/health");
}

#[tokio::test]
async fn test_serve_tab_has_curl_examples() {
    let html = get_index_html().await;
    assert!(html.contains("curl -X POST"), "API docs missing POST curl example");
    assert!(html.contains("curl -X DELETE"), "API docs missing DELETE curl example");
    assert!(html.contains("Content-Type: application/json"), "API docs missing Content-Type header");
}

#[tokio::test]
async fn test_serve_tab_has_http_method_badges() {
    let html = get_index_html().await;
    assert!(html.contains(">POST</span>"), "API docs missing POST method badge");
    assert!(html.contains(">GET</span>"), "API docs missing GET method badge");
    assert!(html.contains(">DELETE</span>"), "API docs missing DELETE method badge");
}

#[tokio::test]
async fn test_serve_tab_documents_request_response_format() {
    let html = get_index_html().await;
    // Prediction request must document the data format
    assert!(html.contains("\"model_id\""), "API docs should show model_id field");
    assert!(html.contains("\"data\""), "API docs should show data field");
    assert!(html.contains("\"predictions\""), "API docs should show predictions response");
    assert!(html.contains("\"probabilities\""), "API docs should show probabilities response");
    assert!(html.contains("\"latency_ms\""), "API docs should show latency_ms response");
}

#[tokio::test]
async fn test_serve_tab_javascript_functions_exist() {
    let html = get_index_html().await;
    assert!(html.contains("function loadServeModels"), "Missing loadServeModels JS function");
    assert!(html.contains("function serveTestPredict"), "Missing serveTestPredict JS function");
    assert!(html.contains("function serveViewStats"), "Missing serveViewStats JS function");
    assert!(html.contains("function serveEvictCache"), "Missing serveEvictCache JS function");
}

// ============================================================================
// Train Sub-Tab Tests
// ============================================================================

#[tokio::test]
async fn test_train_tab_has_subtabs() {
    let html = get_index_html().await;
    let subtabs = ["setup", "models", "compare", "hyperopt", "ensemble", "explain"];
    for sub in &subtabs {
        assert!(
            html.contains(&format!("id=\"sub-{}\"", sub)),
            "Missing sub-panel: {}",
            sub
        );
    }
}

// ============================================================================
// Visualization Canvas Tests
// ============================================================================

#[tokio::test]
async fn test_chart_canvases_present() {
    let html = get_index_html().await;
    let canvases = [
        "columnTypesChart", "trainingMetricsChart", "comparisonChart",
        "convergenceChart", "importanceChart", "anomalyChart",
        "clusteringChart", "cpuGaugeChart", "memGaugeChart",
        "umapChart", "pcaChart", "trainUmapChart",
    ];
    for canvas in &canvases {
        assert!(
            html.contains(&format!("id=\"{}\"", canvas)),
            "Missing chart canvas: {}",
            canvas
        );
    }
}

#[tokio::test]
async fn test_chart_js_loaded() {
    let html = get_index_html().await;
    assert!(html.contains("chart.min.js"), "Chart.js should be loaded");
}

// ============================================================================
// Auto-Tune Integration Tests
// ============================================================================

#[tokio::test]
async fn test_auto_tune_wired_in_poll_jobs() {
    let html = get_index_html().await;
    assert!(html.contains("hyperopt_job_id"), "pollJobs must track hyperopt job id");
    assert!(html.contains("hyperopt_status"), "pollJobs must track hyperopt status");
    assert!(html.contains("/api/hyperopt"), "Auto-tune must call /api/hyperopt");
}

// ============================================================================
// Content-Security-Policy Integration Test
// ============================================================================

#[tokio::test]
async fn test_index_response_headers() {
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

    let ct = response.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.contains("text/html"), "Index should be served as HTML");

    let cc = response.headers().get("cache-control").unwrap().to_str().unwrap();
    assert!(cc.contains("no-cache"), "Index should have no-cache header");
}

// ============================================================================
// CSS & Styling Tests
// ============================================================================

#[tokio::test]
async fn test_index_has_inline_styles() {
    let html = get_index_html().await;
    assert!(html.contains("<style>"), "Index must have inline styles");
    assert!(html.contains("--color-text"), "Index must define CSS custom properties");
}

#[tokio::test]
async fn test_index_has_sub_tab_styles() {
    let html = get_index_html().await;
    assert!(html.contains(".sub-tab"), "Must have sub-tab CSS class");
    assert!(html.contains(".sub-panel"), "Must have sub-panel CSS class");
}

// ============================================================================
// Essential Element IDs
// ============================================================================

#[tokio::test]
async fn test_critical_element_ids_present() {
    let html = get_index_html().await;
    let ids = [
        "cfg-targetColumn", "cfg-taskType",
        "btn-compare", "btn-hyperopt", "btn-anomaly", "btn-clustering",
        "prediction-input", "prediction-output",
        "serve-base-url",
    ];
    for id in &ids {
        assert!(
            html.contains(&format!("id=\"{}\"", id)),
            "Missing critical element: {}",
            id
        );
    }
}

// ============================================================================
// UMAP Visualization Tests
// ============================================================================

#[tokio::test]
async fn test_umap_chart_canvas_present() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"umapChart\""), "Missing UMAP chart canvas in Analyze tab");
    assert!(html.contains("id=\"trainUmapChart\""), "Missing UMAP chart canvas in Train tab");
}

#[tokio::test]
async fn test_umap_controls_present() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"btn-umap\""), "Missing UMAP run button");
    assert!(html.contains("id=\"umap-neighbors\""), "Missing UMAP n_neighbors input");
    assert!(html.contains("id=\"umap-mindist\""), "Missing UMAP min_dist input");
    assert!(html.contains("id=\"umap-colorby\""), "Missing UMAP color-by select");
}

#[tokio::test]
async fn test_umap_result_elements_present() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"umap-results\""), "Missing UMAP results container");
    assert!(html.contains("id=\"umap-chart-wrap\""), "Missing UMAP chart wrapper");
    assert!(html.contains("id=\"umap-empty\""), "Missing UMAP empty state");
}

#[tokio::test]
async fn test_umap_train_card_present() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"train-umap-card\""), "Missing live UMAP card in Train tab");
    assert!(html.contains("id=\"train-umap-badge\""), "Missing live UMAP badge");
    assert!(html.contains("id=\"train-umap-status\""), "Missing live UMAP status container");
}

#[tokio::test]
async fn test_umap_javascript_functions_exist() {
    let html = get_index_html().await;
    assert!(html.contains("function runUmap"), "Missing runUmap JS function");
    assert!(html.contains("function renderUmapChart"), "Missing renderUmapChart JS function");
    assert!(html.contains("function convexHull"), "Missing convexHull JS function");
    assert!(html.contains("function lerpColor"), "Missing lerpColor JS function");
}

// ============================================================================
// PCA Visualization Tests
// ============================================================================

#[tokio::test]
async fn test_pca_chart_canvas_present() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"pcaChart\""), "Missing PCA chart canvas in Analyze tab");
}

#[tokio::test]
async fn test_pca_controls_present() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"btn-pca\""), "Missing PCA run button");
    assert!(html.contains("id=\"pca-scale\""), "Missing PCA scale select");
    assert!(html.contains("id=\"pca-colorby\""), "Missing PCA color-by select");
    assert!(html.contains("id=\"pca-variance\""), "Missing PCA variance display");
}

#[tokio::test]
async fn test_pca_result_elements_present() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"pca-results\""), "Missing PCA results container");
    assert!(html.contains("id=\"pca-chart-wrap\""), "Missing PCA chart wrapper");
    assert!(html.contains("id=\"pca-empty\""), "Missing PCA empty state");
}

#[tokio::test]
async fn test_pca_javascript_functions_exist() {
    let html = get_index_html().await;
    assert!(html.contains("function runPca"), "Missing runPca JS function");
    assert!(html.contains("function renderPcaChart"), "Missing renderPcaChart JS function");
}
