//! Application state management

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use polars::prelude::*;
use crate::inference::{InferenceConfig, InferenceEngine};
use crate::preprocessing::DataPreprocessor;
use crate::training::TrainEngine;

use super::ServerConfig;

/// Training job status
#[derive(Debug, Clone, serde::Serialize)]
pub enum JobStatus {
    Pending,
    Running { progress: f64, message: String },
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
        // Fast path: check cache
        {
            let engines = self.engines.read().await;
            if let Some(engine) = engines.get(model_id) {
                return Ok(Arc::clone(engine));
            }
        }

        // Slow path: load from disk
        let mut engine = InferenceEngine::load(InferenceConfig::new(), None, model_path)?;
        engine.warmup()?;
        let engine = Arc::new(engine);

        // Insert into cache, evict oldest if needed
        {
            let mut engines = self.engines.write().await;
            if engines.len() >= self.max_cached {
                // Evict first entry (simple LRU approximation)
                if let Some(key) = engines.keys().next().cloned() {
                    engines.remove(&key);
                }
            }
            engines.insert(model_id.to_string(), Arc::clone(&engine));
        }

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
    pub current_data: RwLock<Option<DataFrame>>,
    pub preprocessor: RwLock<Option<DataPreprocessor>>,
    pub model_registry: ModelRegistry,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            datasets: RwLock::new(HashMap::new()),
            jobs: RwLock::new(HashMap::new()),
            models: RwLock::new(HashMap::new()),
            current_data: RwLock::new(None),
            preprocessor: RwLock::new(None),
            model_registry: ModelRegistry::new(10), // Cache up to 10 models
        }
    }

    pub fn generate_id() -> String {
        Uuid::new_v4().to_string()[..8].to_string()
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

    /// Get system information
    pub fn get_system_info(&self) -> serde_json::Value {
        use sysinfo::System;
        
        let mut sys = System::new_all();
        sys.refresh_all();

        // Calculate overall CPU usage
        let cpu_usage: f32 = sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() / sys.cpus().len() as f32;

        serde_json::json!({
            "cpu_count": sys.cpus().len(),
            "cpu_usage": cpu_usage,
            "total_memory_gb": sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
            "used_memory_gb": sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
            "memory_usage_percent": (sys.used_memory() as f64 / sys.total_memory() as f64) * 100.0,
        })
    }
}
