//! Application state management

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::RwLock;
use uuid::Uuid;
use polars::prelude::*;
use crate::cache::LruTtlCache;
use crate::inference::{InferenceConfig, InferenceEngine, InferenceStats};
use crate::security::{SecurityManager, SecurityConfig, RateLimiter, RateLimitConfig, AuditTrail};
use crate::preprocessing::DataPreprocessor;
use crate::training::TrainEngine;
use crate::provenance::ProvenanceTracker;
use crate::privacy::RetentionManager;

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

/// Cached evaluation data for Metrics Deep Dive insights
#[derive(Debug, Clone)]
pub struct InsightsEvaluation {
    pub task: crate::training::TaskType,
    pub classes: Vec<String>,
    pub y_true: Vec<f64>,
    pub y_pred: Vec<f64>,
    pub y_prob: Option<Vec<Vec<f64>>>,
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
    engines: DashMap<String, Arc<InferenceEngine>>,
    max_cached: usize,
    /// Shared prediction cache across all models
    prediction_cache: Option<Arc<LruTtlCache<u64, Vec<f64>>>>,
}

impl ModelRegistry {
    pub fn new(max_cached: usize) -> Self {
        Self {
            engines: DashMap::new(),
            max_cached,
            prediction_cache: None,
        }
    }

    /// Create a new model registry with a shared prediction cache
    pub fn with_prediction_cache(mut self, cache: Arc<LruTtlCache<u64, Vec<f64>>>) -> Self {
        self.prediction_cache = Some(cache);
        self
    }

    /// Insert a pre-built engine into the registry (e.g. right after training).
    /// The shared prediction cache is injected automatically.
    pub fn insert(&self, model_id: String, mut engine: InferenceEngine) {
        if let Some(ref cache) = self.prediction_cache {
            engine.set_prediction_cache(Arc::clone(cache));
        }
        // Evict least-used entry if at capacity
        if self.engines.len() >= self.max_cached {
            let least_used = self.engines.iter()
                .min_by_key(|e| e.value().stats().total_predictions)
                .map(|e| e.key().clone());
            if let Some(key) = least_used {
                self.engines.remove(&key);
            }
        }
        self.engines.insert(model_id, Arc::new(engine));
    }

    /// Get a cached inference engine, or load from disk and cache it.
    /// Uses DashMap for lock-free fast-path reads (no async overhead on cache hit).
    pub async fn get_or_load(
        &self,
        model_id: &str,
        model_path: &str,
    ) -> crate::error::Result<Arc<InferenceEngine>> {
        // Fast path: DashMap read — no async overhead
        if let Some(engine) = self.engines.get(model_id) {
            return Ok(Arc::clone(engine.value()));
        }

        // Slow path: load from disk (blocking I/O, done without holding any lock)
        let mut engine = InferenceEngine::load(InferenceConfig::new(), None, model_path)?;

        // Inject shared prediction cache
        if let Some(ref cache) = self.prediction_cache {
            engine.set_prediction_cache(Arc::clone(cache));
        }

        engine.warmup()?;
        let engine = Arc::new(engine);

        // Evict least-used entry if at capacity
        if self.engines.len() >= self.max_cached {
            let least_used = self.engines.iter()
                .min_by_key(|e| e.value().stats().total_predictions)
                .map(|e| e.key().clone());
            if let Some(key) = least_used {
                self.engines.remove(&key);
            }
        }

        // entry().or_insert() is atomic — if another task raced us, we get their Arc back
        let entry = self.engines.entry(model_id.to_string()).or_insert_with(|| Arc::clone(&engine));
        Ok(Arc::clone(entry.value()))
    }

    /// Evict a model from the cache
    pub fn evict(&self, model_id: &str) {
        self.engines.remove(model_id);
    }

    /// Clear all cached models
    pub fn clear(&self) {
        self.engines.clear();
    }

    /// Number of currently cached models
    pub fn cached_count(&self) -> usize {
        self.engines.len()
    }

    /// Get stats for a cached model
    pub fn get_model_stats(&self, model_id: &str) -> Option<InferenceStats> {
        self.engines.get(model_id).map(|e| e.value().stats())
    }
}

/// Application state shared across handlers
pub struct AppState {
    pub config: ServerConfig,
    pub datasets: RwLock<HashMap<String, DatasetInfo>>,
    pub jobs: RwLock<HashMap<String, TrainingJob>>,
    pub models: DashMap<String, ModelInfo>,
    pub current_data: RwLock<Option<DataFrame>>,
    pub preprocessor: RwLock<Option<DataPreprocessor>>,
    pub model_registry: ModelRegistry,
    /// Shared prediction cache for all inference engines
    pub prediction_cache: Arc<LruTtlCache<u64, Vec<f64>>>,
    /// Store trained engines for post-training analysis (explainability, etc.)
    pub train_engines: DashMap<String, TrainEngine>,
    /// Cache of evaluation data (y_true/y_pred) for Metrics Deep Dive
    pub insights_cache: DashMap<String, InsightsEvaluation>,
    /// Quality reports produced by QualityPipeline, keyed by model_id.
    pub quality_reports: dashmap::DashMap<String, crate::quality::QualityReport>,
    /// Fitted OOD detectors, keyed by model_id.
    pub ood_detectors: dashmap::DashMap<String, crate::quality::post_training::OodDetector>,
    /// Fitted conformal predictors for regression, keyed by model_id.
    pub conformal_predictors: dashmap::DashMap<String, crate::quality::post_training::ConformalPredictor>,
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
                .ok().and_then(|s| s.parse().ok()).unwrap_or(200),
            algorithm: crate::security::RateLimitAlgorithm::TokenBucket,
            ..Default::default()
        };

        let prediction_cache = Arc::new(LruTtlCache::new(1000, 300));

        Self {
            config,
            datasets: RwLock::new(HashMap::new()),
            jobs: RwLock::new(HashMap::new()),
            models: DashMap::new(),
            current_data: RwLock::new(None),
            preprocessor: RwLock::new(None),
            model_registry: ModelRegistry::new(10)
                .with_prediction_cache(Arc::clone(&prediction_cache)),
            prediction_cache,
            train_engines: DashMap::new(),
            insights_cache: DashMap::new(),
            quality_reports: dashmap::DashMap::new(),
            ood_detectors: dashmap::DashMap::new(),
            conformal_predictors: dashmap::DashMap::new(),
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
        
        // Set as current data
        *self.current_data.write().await = Some(df);

        id
    }

    /// Evict completed/failed jobs and old train engines to prevent unbounded memory growth.
    /// Called automatically after adding new entries.
    pub async fn evict_stale_entries(&self) {
        // Evict jobs: only acquire write lock if over limit
        {
            let jobs_len = self.jobs.read().await.len();
            if jobs_len > Self::MAX_JOBS {
                let mut jobs = self.jobs.write().await;
                if jobs.len() > Self::MAX_JOBS {
                    let to_remove = jobs.len() - Self::MAX_JOBS;
                    // Single pass to find oldest completed/failed jobs (O(n) not O(n log n))
                    let mut removable: Vec<(String, chrono::DateTime<chrono::Utc>)> = jobs.iter()
                        .filter(|(_, j)| matches!(j.status, JobStatus::Completed { .. } | JobStatus::Failed { .. }))
                        .map(|(id, j)| (id.clone(), j.created_at))
                        .collect();
                    if !removable.is_empty() {
                        // Partial sort: only bring the `to_remove` oldest to front
                        let pivot = to_remove.min(removable.len()) - 1;
                        removable.select_nth_unstable_by_key(pivot, |(_, t)| *t);
                        for (id, _) in removable.into_iter().take(to_remove) {
                            jobs.remove(&id);
                        }
                    }
                }
            }
        }

        // Evict train_engines and insights_cache in lockstep: same keys removed from both
        if self.train_engines.len() > Self::MAX_TRAIN_ENGINES {
            let to_remove = self.train_engines.len() - Self::MAX_TRAIN_ENGINES;
            let keys: Vec<String> = self.train_engines.iter()
                .take(to_remove)
                .map(|entry| entry.key().clone())
                .collect();
            for key in &keys {
                self.train_engines.remove(key);
                self.insights_cache.remove(key);
            }
        }

        // Truncate completed studies
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

        let (cache_hits, cache_misses, cache_hit_rate) = self.prediction_cache.stats();

        serde_json::json!({
            "cpu_count": sys.cpus().len(),
            "cpu_usage": cpu_usage,
            "total_memory_gb": total_memory as f64 / 1024.0 / 1024.0 / 1024.0,
            "used_memory_gb": sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
            "memory_usage_percent": (sys.used_memory() as f64 / total_memory as f64) * 100.0,
            "prediction_cache": {
                "size": self.prediction_cache.len(),
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": cache_hit_rate,
            },
        })
    }
}

/// Scan `models_dir` for `model_*.json` files and populate `models` with
/// restored `ModelInfo` entries. Called once at server startup (synchronously,
/// before axum::serve). Skips autotune_ and optimized_ files. Logs and skips
/// malformed files without panicking.
pub fn restore_models_from_disk(models_dir: &str, models: &dashmap::DashMap<String, ModelInfo>) {
    use crate::training::TrainEngine;
    use tracing::{info, warn};

    let dir = match std::fs::read_dir(models_dir) {
        Ok(d) => d,
        Err(e) => {
            warn!(models_dir = %models_dir, error = %e, "Cannot read models directory during restore");
            return;
        }
    };

    for entry in dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }

        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };

        if stem.starts_with("autotune_") || stem.starts_with("optimized_") { continue; }
        if !stem.starts_with("model_") { continue; }

        let path_str = match path.to_str() {
            Some(s) => s.to_string(),
            None => continue,
        };

        let engine = match TrainEngine::load(&path_str) {
            Ok(e) => e,
            Err(err) => {
                warn!(path = %path_str, error = %err, "Skipping model file — failed to load");
                continue;
            }
        };

        let display_type = {
            let parts: Vec<&str> = stem.splitn(3, '_').collect();
            if parts.len() >= 2 { parts[1].to_string() } else { stem.clone() }
        };

        let metrics = engine.metrics()
            .cloned()
            .and_then(|m| serde_json::to_value(m).ok())
            .unwrap_or(serde_json::Value::Null);

        let created_at = std::fs::metadata(&path)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(|t| chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339())
            .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

        let task_type = format!("{:?}", engine.config().task_type);

        let model_info = ModelInfo {
            id: stem.clone(),
            name: format!("{} (restored)", display_type),
            task_type,
            metrics,
            created_at,
            path: path.clone(),
        };

        models.insert(stem.clone(), model_info);
        info!(model_id = %stem, path = %path_str, "Restored model from disk");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{TrainEngine, TrainingConfig, TaskType, ModelType};
    use polars::prelude::*;

    fn make_test_engine() -> TrainEngine {
        let df = df! {
            "x" => [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y" => [0.0f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }.unwrap();
        let config = TrainingConfig::new(TaskType::BinaryClassification, "y")
            .with_model(ModelType::LogisticRegression);
        let mut engine = TrainEngine::new(config);
        engine.fit(&df).unwrap();
        engine
    }

    #[test]
    fn test_restore_models_from_disk_loads_valid_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model_LogisticRegression_abc123.json");
        let engine = make_test_engine();
        engine.save(path.to_str().unwrap()).unwrap();

        let models: DashMap<String, ModelInfo> = DashMap::new();
        restore_models_from_disk(dir.path().to_str().unwrap(), &models);

        assert_eq!(models.len(), 1);
        let entry = models.iter().next().unwrap();
        assert_eq!(entry.key(), "model_LogisticRegression_abc123");
        assert!(entry.value().name.contains("restored"));
        assert_eq!(entry.value().path, path);
    }

    #[test]
    fn test_restore_models_from_disk_skips_autotune() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("autotune_RandomForest_xyz.json");
        let engine = make_test_engine();
        engine.save(path.to_str().unwrap()).unwrap();

        let models: DashMap<String, ModelInfo> = DashMap::new();
        restore_models_from_disk(dir.path().to_str().unwrap(), &models);
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_restore_models_from_disk_skips_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model_Bad_zzz.json"),
            b"not valid json {{{"
        ).unwrap();

        let models: DashMap<String, ModelInfo> = DashMap::new();
        // Must not panic
        restore_models_from_disk(dir.path().to_str().unwrap(), &models);
        assert_eq!(models.len(), 0);
    }
}

#[cfg(test)]
mod quality_tests {
    use super::*;
    use crate::quality::QualityReport;

    #[test]
    fn test_app_state_stores_and_retrieves_quality_report() {
        let reports: dashmap::DashMap<String, QualityReport> = dashmap::DashMap::new();
        let report = QualityReport { model_id: "m1".into(), overall_quality_score: 0.85, ..Default::default() };
        reports.insert("m1".to_string(), report);
        let retrieved = reports.get("m1").unwrap();
        assert_eq!(retrieved.model_id, "m1");
        assert!((retrieved.overall_quality_score - 0.85).abs() < 1e-10);
    }
}
