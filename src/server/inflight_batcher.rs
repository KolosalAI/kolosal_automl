//! Inflight batching for inference requests
//!
//! A tokio-native async batcher that sits between HTTP handlers and the inference
//! engine. Concurrent requests are transparently collected into batches, reducing
//! per-request overhead and improving throughput under load.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ndarray::{Array1, Array2};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, warn};

use super::state::ModelRegistry;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the inflight batcher, populated from env vars.
#[derive(Debug, Clone)]
pub struct InflightBatcherConfig {
    /// Maximum number of requests to coalesce into a single batch.
    pub max_batch_size: usize,
    /// Maximum milliseconds to wait for additional requests after the first arrives.
    pub max_wait_time_ms: u64,
    /// Bounded channel capacity for incoming requests.
    pub queue_capacity: usize,
}

impl Default for InflightBatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: std::env::var("BATCH_MAX_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(32),
            max_wait_time_ms: std::env::var("BATCH_MAX_WAIT_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            queue_capacity: std::env::var("BATCH_QUEUE_CAPACITY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1024),
        }
    }
}

// ---------------------------------------------------------------------------
// Request / Handle types
// ---------------------------------------------------------------------------

/// A single inference request submitted by a handler.
pub struct InferenceRequest {
    pub model_id: String,
    pub model_path: String,
    pub features: Array2<f64>,
    pub response_tx: oneshot::Sender<Result<Array1<f64>, String>>,
}

/// Cloneable handle used by HTTP handlers to submit requests.
#[derive(Clone)]
pub struct InflightBatcherHandle {
    tx: mpsc::Sender<InferenceRequest>,
    /// Number of requests currently inflight (submitted but not yet responded).
    inflight_count: Arc<AtomicU64>,
    /// Monotonic total request counter.
    total_requests: Arc<AtomicU64>,
}

impl InflightBatcherHandle {
    /// Submit a prediction request through the batcher and await the result.
    pub async fn predict(
        &self,
        model_id: String,
        model_path: String,
        features: Array2<f64>,
    ) -> Result<Array1<f64>, String> {
        let (response_tx, response_rx) = oneshot::channel();

        let req = InferenceRequest {
            model_id,
            model_path,
            features,
            response_tx,
        };

        // Increment counters AFTER successful send to avoid inflated counts on failure
        if self.tx.send(req).await.is_err() {
            return Err("Inflight batcher channel closed".to_string());
        }

        self.inflight_count.fetch_add(1, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let result = response_rx
            .await
            .map_err(|_| "Inflight batcher dropped response channel".to_string())?;

        self.inflight_count.fetch_sub(1, Ordering::Relaxed);
        result
    }

    /// Current number of inflight requests.
    pub fn inflight_count(&self) -> u64 {
        self.inflight_count.load(Ordering::Relaxed)
    }

    /// Total requests submitted since startup.
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Approximate queue depth (requests waiting but not yet picked up by the batch loop).
    /// This is a lower bound since we can't inspect the channel directly, but
    /// `inflight - being_processed` is a reasonable proxy. We expose the channel's
    /// max capacity minus remaining capacity when available.
    pub fn queue_depth(&self) -> usize {
        self.tx.max_capacity() - self.tx.capacity()
    }
}

// ---------------------------------------------------------------------------
// Batcher spawn + loop
// ---------------------------------------------------------------------------

/// Spawn the background batch loop and return a handle for submitting requests.
pub fn spawn_inflight_batcher(
    config: InflightBatcherConfig,
    model_registry: Arc<ModelRegistry>,
) -> InflightBatcherHandle {
    let (tx, rx) = mpsc::channel(config.queue_capacity);
    let inflight_count = Arc::new(AtomicU64::new(0));
    let total_requests = Arc::new(AtomicU64::new(0));

    tokio::spawn(batch_loop(config, rx, model_registry));

    InflightBatcherHandle {
        tx,
        inflight_count,
        total_requests,
    }
}

/// The main batch loop: waits for the first request, then collects more within
/// the deadline window before dispatching the batch.
async fn batch_loop(
    config: InflightBatcherConfig,
    mut rx: mpsc::Receiver<InferenceRequest>,
    model_registry: Arc<ModelRegistry>,
) {
    loop {
        // Wait (park) until the first request arrives.
        let first = match rx.recv().await {
            Some(req) => req,
            None => {
                debug!("Inflight batcher channel closed, exiting batch loop");
                return;
            }
        };

        let mut batch = vec![first];
        let deadline = tokio::time::Instant::now()
            + tokio::time::Duration::from_millis(config.max_wait_time_ms);

        // Collect more requests until batch is full or deadline expires.
        while batch.len() < config.max_batch_size {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(req)) => batch.push(req),
                Ok(None) => {
                    // Channel closed — process what we have and exit.
                    debug!("Channel closed during batch collection");
                    process_batch(batch, &model_registry).await;
                    return;
                }
                Err(_) => break, // timeout
            }
        }

        debug!(batch_size = batch.len(), "Dispatching inflight batch");
        process_batch(batch, &model_registry).await;
    }
}

/// Process a collected batch: group by model_id, call engine.predict_array()
/// per model, then split results back to individual oneshot channels.
async fn process_batch(requests: Vec<InferenceRequest>, model_registry: &Arc<ModelRegistry>) {
    // Group requests by model_id so we can do a single predict_array per model.
    let mut groups: HashMap<String, Vec<(usize, InferenceRequest)>> = HashMap::new();
    for (idx, req) in requests.into_iter().enumerate() {
        groups
            .entry(req.model_id.clone())
            .or_default()
            .push((idx, req));
    }

    for (_model_id, group) in groups {
        let model_id = group[0].1.model_id.clone();
        let model_path = group[0].1.model_path.clone();

        // Record the number of rows each request contributes so we can split later.
        let mut row_counts: Vec<usize> = Vec::with_capacity(group.len());
        let mut all_features: Vec<Array2<f64>> = Vec::with_capacity(group.len());
        let mut senders: Vec<oneshot::Sender<Result<Array1<f64>, String>>> =
            Vec::with_capacity(group.len());

        for (_idx, req) in group {
            row_counts.push(req.features.nrows());
            all_features.push(req.features);
            senders.push(req.response_tx);
        }

        // Pre-allocate concatenated array with known total rows
        let total_rows: usize = row_counts.iter().sum();
        let n_cols = all_features.first().map(|a| a.ncols()).unwrap_or(0);
        let mut concatenated = Array2::zeros((total_rows, n_cols));
        let mut row_offset = 0;
        for features in &all_features {
            let nrows = features.nrows();
            concatenated
                .slice_mut(ndarray::s![row_offset..row_offset + nrows, ..])
                .assign(&features.view());
            row_offset += nrows;
        }

        // Load engine and run prediction on a blocking thread (it's CPU-bound).
        let registry = Arc::clone(model_registry);
        let result = tokio::task::spawn_blocking(move || {
            // We need a runtime handle to call the async get_or_load from a blocking context.
            let rt = tokio::runtime::Handle::current();
            let engine = rt.block_on(registry.get_or_load(&model_id, &model_path))
                .map_err(|e| format!("Failed to load model: {}", e))?;
            engine
                .predict_array(&concatenated)
                .map_err(|e| format!("Prediction failed: {}", e))
        })
        .await;

        match result {
            Ok(Ok(predictions)) => {
                // Split the concatenated predictions back to individual requests.
                let mut offset = 0;
                for (tx, count) in senders.into_iter().zip(row_counts.iter()) {
                    let slice = predictions.slice(ndarray::s![offset..offset + count]).to_owned();
                    offset += count;
                    let _ = tx.send(Ok(slice));
                }
            }
            Ok(Err(e)) => {
                warn!(error = %e, "Batch prediction failed");
                for tx in senders {
                    let _ = tx.send(Err(e.clone()));
                }
            }
            Err(join_err) => {
                let msg = format!("Batch task panicked: {}", join_err);
                error!("{}", msg);
                for tx in senders {
                    let _ = tx.send(Err(msg.clone()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = InflightBatcherConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_wait_time_ms, 10);
        assert_eq!(config.queue_capacity, 1024);
    }
}
