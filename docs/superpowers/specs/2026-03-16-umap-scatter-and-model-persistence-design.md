# Design: UMAP Streaming Scatter & Model Persistence Fix

**Date:** 2026-03-16
**Status:** Approved

---

## 1. Problem Statement

### 1a — Model persistence bug
Trained models disappear after a page refresh and do not reappear in the models list. Root cause: `AppState::new()` initialises `models: DashMap::new()` as empty and nothing scans `models_dir` on startup to repopulate it. Model JSON files are correctly written to disk by `engine.save()`, but the in-memory `state.models` map is never restored after a server restart.

### 1b — UMAP scatter in training results
After a model finishes training the results card shows metrics but no spatial view of the data. Users have no way to see how well the learned representation separates classes, spreads regression targets, or groups clusters.

---

## 2. Scope

| Feature | In scope |
|---|---|
| Scan `models_dir` on startup and restore `ModelInfo` entries | Yes |
| SSE endpoint streaming partial UMAP embeddings | Yes |
| Canvas scatter in training results card (classification / regression / clustering) | Yes |
| Auto-trigger UMAP stream when training result card appears | Yes |
| Modifying UMAP hyper-parameters from the UI | No |
| Persisting insights_cache or train_engines across restarts | No |

---

## 3. Architecture

### 3a — Model persistence fix

**Location:** `src/server/state.rs` (function definition) + `src/server/mod.rs` (call site, inside `run_server` after `AppState::new` and before `axum::serve`). The function must be called synchronously; `main.rs` is not modified.

Add a **synchronous** function:

```rust
pub fn restore_models_from_disk(models_dir: &str, models: &DashMap<String, ModelInfo>)
```

Steps:
1. `std::fs::read_dir(models_dir)` — iterate entries.
2. Accept only files matching the pattern `model_<type>_<job_id>.json`. Files with prefixes `autotune_` or `optimized_` are skipped silently (they use a different naming convention and may not be independently loadable as standalone inference models).
3. For each accepted file call `TrainEngine::load(path)` to deserialise the engine.
4. Build `ModelInfo`:
   - `id` — use the filename stem directly (e.g. `model_LogisticRegression_abc123`). This is stable across restarts with zero extra dependencies and no hashing. The ID will not match any session UUID from the original run; that is acceptable and expected.
   - `name` — parse the `<type>` segment from the filename: `"{type} (restored)"`.
   - `task_type` — `format!("{:?}", engine.config.task_type)`.
   - `metrics` — `engine.metrics().cloned().map(|m| serde_json::to_value(m).unwrap_or_default()).unwrap_or(serde_json::Value::Null)`.
   - `created_at` — file modification time in RFC 3339, fallback to `Utc::now()`.
   - `path` — canonical `PathBuf` of the file.
5. Insert into `models`.

Failures for individual files are logged as `warn!` and skipped; they must not prevent the server from starting.

### 3b — UMAP streaming scatter

#### UMAP epoch callback (backend)

`src/visualization/umap.rs` — add `fit_transform_with_cb` that matches the existing type conventions in that module (`&[Vec<f64>]` input, `Vec<[f64; 2]>` output):

```rust
pub fn fit_transform_with_cb<F>(
    &self,
    data: &[Vec<f64>],
    mut on_epoch: F,
) -> crate::error::Result<Vec<[f64; 2]>>
where
    F: FnMut(usize, &[[f64; 2]]),  // (epoch_index, current_embedding_snapshot)
```

Emit the callback every `UMAP_STREAM_EMIT_EVERY` epochs, where:

```rust
const UMAP_STREAM_EMIT_EVERY: usize = 20;
```

This is a module-level constant, not a UI-configurable parameter (consistent with the "no UI hyper-param tuning" scope). The existing `fit_transform` delegates to this with a no-op closure.

After the optimization loop completes, `fit_transform_with_cb` invokes `on_epoch` one final time with the last epoch index and the completed embedding — regardless of whether `n_epochs % UMAP_STREAM_EMIT_EVERY == 0`. This guarantees the async SSE side always receives the fully-converged layout.

#### SSE endpoint (backend)

New route: `GET /api/visualization/umap/stream`

Query params:
- `model_id` — required
- `max_samples` — optional, default 2000

**Bridging `spawn_blocking` to async SSE:**
Use a `tokio::sync::mpsc::channel` to pass epoch snapshots from the blocking thread back to the async SSE stream:

```
spawn_blocking closure:
  fit_transform_with_cb(data, |epoch, embedding| {
      let _ = tx.blocking_send(UmapChunk { epoch, points: embedding.to_vec() });
  })

async SSE task:
  while let Some(chunk) = rx.recv().await {
      yield SSE event from chunk
  }
```

Full handler behaviour:
1. Look up `ModelInfo` by `model_id`.
2. Load `TrainEngine::load(&info.path)` — this recovers `config.target_column` needed for step 5.
3. Load dataset from `state.current_data`.
4. Sample up to `max_samples` rows:
   - **Classification / Clustering** — stratified per class/cluster label; guarantee at least 1 sample per class (rare classes with count ≥ 1 are always included, even if the per-class quota rounds down to 0).
   - **Regression / TimeSeries** — uniform random sample.
5. Build feature matrix by dropping `engine.config.target_column` from the sampled DataFrame.
6. Extract labels in parallel with the feature matrix:
   - **BinaryClassification / MultiClassification** — string class name from the target column (e.g. `"class_0"`).
   - **Regression / TimeSeries** — float target value (used as `value` field, not `label`).
   - **Clustering** — call `engine.predict(sampled_df)` which returns cluster assignments (KMeans/DBSCAN both implement `predict` returning cluster indices via `TrainEngine::predict`); cast each index to `"cluster_N"` string.
7. Spawn `spawn_blocking` + channel pair as described above.
8. Stream SSE events:
   - Each intermediate event: `progress`, `done: false`, `task_type`, `points` array. Compute `progress = epoch_index as f64 / config.n_epochs as f64` where `epoch_index` is the first argument passed by `fit_transform_with_cb` to the callback.
   - Final event: same structure with `progress: 1.0`, `done: true`, and the **complete final embedding** in `points`.

**SSE event payload:**

Classification / Clustering point:
```json
{
  "progress": 0.45,
  "done": false,
  "task_type": "BinaryClassification",
  "points": [
    { "x": 1.23, "y": -0.45, "label": "class_0" }
  ]
}
```

Regression / TimeSeries point (`label` field is absent; `value` is used instead; `value_range` is included in every event so the frontend can scale the gradient consistently):
```json
{
  "progress": 0.45,
  "done": false,
  "task_type": "Regression",
  "value_range": [0.0, 10.5],
  "points": [
    { "x": 1.23, "y": -0.45, "value": 4.2 }
  ]
}
```

#### Frontend — canvas scatter in training results card

After training completes and the results card renders:

1. Inject a `<canvas id="umap-scatter-{model_id}" style="width:100%;height:280px">` below the metrics table inside the card.
2. Render a `"Computing UMAP…"` label and a thin progress bar above the canvas.
3. Open `new EventSource('/api/visualization/umap/stream?model_id=' + id)`.
4. On each message: parse JSON, update the progress bar, clear canvas, redraw all points.
5. On `done: true`: close `EventSource`, hide the progress bar.
6. Set `es.onerror`: close `EventSource`, hide the progress bar, show an inline `"UMAP failed"` message in place of the canvas.

**Canvas drawing — per task type:**

| Task type | Point color | Legend |
|---|---|---|
| BinaryClassification, MultiClassification | Palette of up to 10 distinct hues, one per class name | Dot + class name, bottom-left corner |
| Regression, TimeSeries | Continuous gradient `hsl(240,80%,55%)` (blue, min) → `hsl(0,80%,55%)` (red, max); `t = (value − min) / (max − min)` | Vertical gradient bar with min/max value labels, right edge |
| Clustering | Same palette as classification, one color per cluster ID | Dot + "Cluster N" label |

Common:
- Point radius 3 px, alpha 0.75.
- No axis lines (UMAP coords are not meaningful).
- No hover tooltip.

---

## 4. Error Handling

| Scenario | Behaviour |
|---|---|
| `model_id` not found in `state.models` | SSE: `{"error":"model not found"}` then close |
| `current_data` is `None` | SSE: `{"error":"no dataset loaded"}` then close |
| `TrainEngine::load` fails | SSE: `{"error":"failed to load model: …"}` then close |
| UMAP computation error | SSE: `{"error":"umap failed: …"}` then close |
| File load fails during startup scan | `warn!` log, skip file, continue |
| `autotune_` / `optimized_` filenames | Silent skip during startup scan |

---

## 5. Testing

- **Unit:** `restore_models_from_disk` with a temp dir containing one valid `model_*.json`, one `autotune_*.json` (must be skipped), and one malformed JSON (must be skipped with a warning).
- **Integration:** train a model, call `restore_models_from_disk` on a fresh `DashMap`, assert the model appears with the correct `task_type` and `path`.
- **E2E (Playwright):** train a model, wait for `canvas#umap-scatter-*` to be visible, assert the progress bar disappears after `done: true`.

---

## 6. Files Changed

| File | Change |
|---|---|
| `src/server/state.rs` | Add `restore_models_from_disk()` |
| `src/server/mod.rs` | Call `restore_models_from_disk` synchronously inside `run_server`, after `AppState::new` and before `axum::serve` |
| `src/visualization/umap.rs` | Add `UMAP_STREAM_EMIT_EVERY` constant; add `fit_transform_with_cb`; delegate existing `fit_transform` to it |
| `src/server/api.rs` | Register `GET /api/visualization/umap/stream` route |
| `src/server/handlers.rs` | Add SSE handler `umap_stream_handler`; inject UMAP canvas + EventSource JS into training results card |
