//! Application state management

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use polars::prelude::*;
use crate::inference::{InferenceConfig, InferenceEngine};
use crate::security::{SecurityManager, SecurityConfig, RateLimiter, RateLimitConfig, AuditTrail};
use crate::preprocessing::DataPreprocessor;
use crate::training::TrainEngine;
use crate::provenance::ProvenanceTracker;
use crate::privacy::RetentionManager;

use super::inflight_batcher::{InflightBatcherConfig, InflightBatcherHandle, spawn_inflight_batcher};
use super::scaling::ScalingState;
use super::ServerConfig;

/// Training job status
#[derive(Debug, Clone, serde::Serialize)]
pub enum JobStatus {
    Pending,
    Running {
        progress: f64,
        message: String,
        /// Partial results streamed as they arrive (e.g. leaderboard entries, trial results)
        #[serde(skip_serializing_if = "Option::is_none")]
        partial_results: Option<Vec<serde_json::Value>>,
    },
    Completed { metrics: serde_json::Value },
    Failed { error: String },
}

/// Training job information
#[derive(Debug, Clone)]
pub struct TrainingJob {
    pub id: String,
    pub status: JobStatus,
    pub config: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub model_path: Option<PathBuf>,
}

/// Trained model information
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub task_type: String,
    pub metrics: serde_json::Value,
    pub created_at: String,
    pub path: PathBuf,
}

/// Uploaded dataset information
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub rows: usize,
    pub columns: usize,
    pub column_names: Vec<String>,
    pub dtypes: Vec<String>,
}

/// In-memory model registry for caching loaded inference engines
pub struct ModelRegistry {
    engines: RwLock<HashMap<String, Arc<InferenceEngine>>>,
    max_cached: usize,
}

impl ModelRegistry {
    pub fn new(max_cached: usize) -> Self {
        Self {
            engines: RwLock::new(HashMap::new()),
            max_cached,
        }
    }

    /// Get a cached inference engine, or load from disk and cache it
    pub async fn get_or_load(
        &self,
        model_id: &str,
        model_path: &str,
    ) -> crate::error::Result<Arc<InferenceEngine>> {
        // Fast path: check cache with read lock (no blocking on concurrent reads)
        {
            let engines = self.engines.read().await;
            if let Some(engine) = engines.get(model_id) {
                return Ok(Arc::clone(engine));
            }
        }

        // Load from disk WITHOUT holding any lock — this is potentially slow I/O
        // and we don't want to block concurrent cache reads.
        let mut engine = InferenceEngine::load(InferenceConfig::new(), None, model_path)?;
        engine.warmup()?;
        let engine = Arc::new(engine);

        // Now take write lock and double-check (another task may have loaded
        // the same model concurrently).
        let mut engines = self.engines.write().await;
        if let Some(existing) = engines.get(model_id) {
            return Ok(Arc::clone(existing));
        }

        // Evict least-used entry if at capacity
        if engines.len() >= self.max_cached {
            let least_used = engines.iter()
                .min_by_key(|(_, e)| e.stats().total_predictions)
                .map(|(k, _)| k.clone());
            if let Some(key) = least_used {
                engines.remove(&key);
            }
        }
        engines.insert(model_id.to_string(), Arc::clone(&engine));

        Ok(engine)
    }

    /// Evict a model from the cache
    pub async fn evict(&self, model_id: &str) {
        let mut engines = self.engines.write().await;
        engines.remove(model_id);
    }

    /// Clear all cached models
    pub async fn clear(&self) {
        let mut engines = self.engines.write().await;
        engines.clear();
    }

    /// Number of currently cached models
    pub async fn cached_count(&self) -> usize {
        self.engines.read().await.len()
    }

    /// Get stats for a cached model
    pub async fn get_model_stats(&self, model_id: &str) -> Option<crate::inference::InferenceStats> {
        let engines = self.engines.read().await;
        engines.get(model_id).map(|e| e.stats())
    }
}

/// Application state shared across handlers
pub struct AppState {
    pub config: ServerConfig,
    pub datasets: RwLock<HashMap<String, DatasetInfo>>,
    pub jobs: RwLock<HashMap<String, TrainingJob>>,
    pub models: RwLock<HashMap<String, ModelInfo>>,
    pub current_data: RwLock<Option<Arc<DataFrame>>>,
    pub preprocessor: RwLock<Option<DataPreprocessor>>,
    pub model_registry: Arc<ModelRegistry>,
    /// Inflight batcher for coalescing concurrent inference requests
    pub inflight_batcher: InflightBatcherHandle,
    /// Scaling state for zero-autoscaling
    pub scaling: ScalingState,
    /// Store trained engines for post-training analysis (explainability, etc.)
    pub train_engines: RwLock<HashMap<String, TrainEngine>>,
    /// Store completed hyperopt studies for history
    pub completed_studies: RwLock<Vec<serde_json::Value>>,
    /// Security manager for auth, input validation, audit logging
    pub security_manager: Arc<SecurityManager>,
    /// Rate limiter for API throttling
    pub rate_limiter: Arc<RateLimiter>,
    /// Data provenance tracker (ISO 5259, 5338)
    pub provenance_tracker: ProvenanceTracker,
    /// Tamper-evident audit trail (ISO 27001, TR 24028)
    pub audit_trail: AuditTrail,
    /// Data retention manager (ISO 27701)
    pub retention_manager: RetentionManager,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        // Initialize security from env vars
        let security_config = SecurityConfig {
            api_keys: std::env::var("SECURITY_API_KEYS")
                .map(|s| s.split(',').map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).collect())
                .unwrap_or_default(),
            jwt_secret: std::env::var("SECURITY_JWT_SECRET").ok(),
            jwt_expiry_secs: std::env::var("SECURITY_JWT_EXPIRY")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(3600),
            enable_api_key_auth: std::env::var("SECURITY_ENABLE_API_KEY")
                .map(|v| v == "true" || v == "1").unwrap_or(false),
            enable_jwt_auth: std::env::var("SECURITY_ENABLE_JWT")
                .map(|v| v == "true" || v == "1").unwrap_or(false),
            blocked_ips: std::env::var("SECURITY_BLOCKED_IPS")
                .map(|s| s.split(',').map(|ip| ip.trim().to_string()).filter(|ip| !ip.is_empty()).collect())
                .unwrap_or_default(),
            audit_log_enabled: std::env::var("SECURITY_AUDIT_LOG")
                .map(|v| v != "false" && v != "0").unwrap_or(true),
        };

        let rate_limit_config = RateLimitConfig {
            requests_per_window: std::env::var("SECURITY_RATE_LIMIT")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(200),
            window_seconds: std::env::var("SECURITY_RATE_WINDOW")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(60),
            burst_size: std::env::var("SECURITY_RATE_BURST")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(20),
            ..Default::default()
        };

        let model_registry = Arc::new(ModelRegistry::new(10));
        let batcher_config = InflightBatcherConfig::default();
        let inflight_batcher = spawn_inflight_batcher(batcher_config, Arc::clone(&model_registry));
        let scaling = ScalingState::new();

        Self {
            config,
            datasets: RwLock::new(HashMap::new()),
            jobs: RwLock::new(HashMap::new()),
            models: RwLock::new(HashMap::new()),
            current_data: RwLock::new(None),
            preprocessor: RwLock::new(None),
            model_registry,
            inflight_batcher,
            scaling,
            train_engines: RwLock::new(HashMap::new()),
            completed_studies: RwLock::new(Vec::new()),
            security_manager: Arc::new(SecurityManager::new(security_config)),
            rate_limiter: Arc::new(RateLimiter::new(rate_limit_config)),
            provenance_tracker: ProvenanceTracker::default(),
            audit_trail: AuditTrail::default(),
            retention_manager: RetentionManager::new(),
        }
    }

    /// Maximum number of completed jobs to retain in memory
    const MAX_JOBS: usize = 1000;
    /// Maximum number of train engines to retain
    const MAX_TRAIN_ENGINES: usize = 50;
    /// Maximum number of completed studies to retain
    const MAX_COMPLETED_STUDIES: usize = 100;

    pub fn generate_id() -> String {
        // Use full UUID to avoid collisions (previously only 8 hex chars = 32 bits)
        Uuid::new_v4().to_string()
    }

    /// Store a dataset and return its ID
    pub async fn store_dataset(&self, name: String, df: DataFrame, path: PathBuf) -> String {
        let id = Self::generate_id();
        
        let column_names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        let dtypes: Vec<String> = df.dtypes().iter().map(|d| format!("{:?}", d)).collect();
        
        let info = DatasetInfo {
            id: id.clone(),
            name,
            path,
            rows: df.height(),
            columns: df.width(),
            column_names,
            dtypes,
        };

        // Store info
        self.datasets.write().await.insert(id.clone(), info);
        
        // Set as current data (wrapped in Arc for cheap cloning in handlers)
        *self.current_data.write().await = Some(Arc::new(df));

        id
    }

    /// Evict completed/failed jobs and old train engines to prevent unbounded memory growth.
    /// Called automatically after adding new entries.
    pub async fn evict_stale_entries(&self) {
        // Evict oldest completed/failed jobs if over limit.
        // Collect keys under read lock, then take write lock only if removal needed.
        let jobs_to_remove = {
            let jobs = self.jobs.read().await;
            if jobs.len() > Self::MAX_JOBS {
                let mut completed_ids: Vec<(String, chrono::DateTime<chrono::Utc>)> = jobs.iter()
                    .filter(|(_, j)| matches!(j.status, JobStatus::Completed { .. } | JobStatus::Failed { .. }))
                    .map(|(id, j)| (id.clone(), j.created_at))
                    .collect();
                completed_ids.sort_by_key(|(_, t)| *t);
                let to_remove = jobs.len() - Self::MAX_JOBS;
                completed_ids.into_iter().take(to_remove).map(|(id, _)| id).collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        };
        if !jobs_to_remove.is_empty() {
            let mut jobs = self.jobs.write().await;
            for id in jobs_to_remove {
                jobs.remove(&id);
            }
        }

        // Evict oldest train engines if over limit
        let engine_keys_to_remove = {
            let engines = self.train_engines.read().await;
            if engines.len() > Self::MAX_TRAIN_ENGINES {
                let to_remove = engines.len() - Self::MAX_TRAIN_ENGINES;
                engines.keys().cloned().take(to_remove).collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        };
        if !engine_keys_to_remove.is_empty() {
            let mut engines = self.train_engines.write().await;
            for key in engine_keys_to_remove {
                engines.remove(&key);
            }
        }

        // Truncate completed studies
        {
            let studies = self.completed_studies.read().await;
            if studies.len() <= Self::MAX_COMPLETED_STUDIES {
                return;
            }
        }
        let mut studies = self.completed_studies.write().await;
        if studies.len() > Self::MAX_COMPLETED_STUDIES {
            let drain_count = studies.len() - Self::MAX_COMPLETED_STUDIES;
            studies.drain(..drain_count);
        }
    }

    /// Get system information
    pub fn get_system_info(&self) -> serde_json::Value {
        use sysinfo::System;

        let mut sys = System::new_all();
        sys.refresh_all();

        // Calculate overall CPU usage
        let cpu_count = sys.cpus().len().max(1);
        let cpu_usage: f32 = sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() / cpu_count as f32;

        let total_memory = sys.total_memory().max(1); // avoid division by zero

        serde_json::json!({
            "cpu_count": sys.cpus().len(),
            "cpu_usage": cpu_usage,
            "total_memory_gb": total_memory as f64 / 1024.0 / 1024.0 / 1024.0,
            "used_memory_gb": sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
            "memory_usage_percent": (sys.used_memory() as f64 / total_memory as f64) * 100.0,
        })
    }
}
