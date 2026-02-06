//! Kolosal AutoML Server
//!
//! High-performance web server for the Kolosal AutoML platform.
//! Provides REST API and web UI for data preprocessing, model training,
//! and inference - all in pure Rust.

mod api;
mod error;
mod state;
mod handlers;

pub use api::create_router;
pub use error::ServerError;
pub use state::AppState;

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub static_dir: Option<String>,
    pub data_dir: String,
    pub models_dir: String,
    pub max_upload_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            static_dir: Some("kolosal-web/static".to_string()),
            data_dir: "./data".to_string(),
            models_dir: "./models".to_string(),
            max_upload_size: 100 * 1024 * 1024, // 100MB
        }
    }
}

/// Start the server with the given configuration
pub async fn run_server(config: ServerConfig) -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "kolosal_server=info,tower_http=debug".into()),
        )
        .init();

    // Create directories
    std::fs::create_dir_all(&config.data_dir)?;
    std::fs::create_dir_all(&config.models_dir)?;

    // Create app state
    let state = Arc::new(AppState::new(config.clone()));

    // Build router
    let app = create_router(state, &config);

    // Start server
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    info!("🚀 Kolosal AutoML Server starting on http://{}", addr);
    info!("📊 Web UI available at http://{}/", addr);
    info!("📡 API available at http://{}/api", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_upload_size, 100 * 1024 * 1024);
    }
}
