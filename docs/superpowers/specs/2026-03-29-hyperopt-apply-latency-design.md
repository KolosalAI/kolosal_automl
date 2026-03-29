# HyperOpt Apply & Real Latency Tracking — Design

**Date:** 2026-03-29
**Status:** Approved

## Goal

Two independent improvements that complete and polish the last round of work:

1. **Feature A — Apply Best Params:** Close the hyperopt loop by wiring the existing `/hyperopt/apply` backend endpoint to a frontend "Apply & Retrain" button. The resulting model is fully first-class: appears in the dashboard, triggers quality fetch and UMAP, enables predict/explain.

2. **Feature B — Real Latency Tracking:** Replace the three hardcoded SLO latency values (`5.0 / 25.0 / 100.0 ms`) with percentiles computed from a ring buffer of actual prediction latency measurements.

---

## Feature A — Apply Best Params & Retrain

### Placement

Everything lives inside the existing `#e-ho-result` div. No new HTML panels. The result card already renders dynamically via `eHoPoll` on completion — we extend that rendered HTML.

### Best Params Display (improvement)

Replace the current `JSON.stringify(m.best_trial.params||{})` with a two-column table:

```
┌─────────────────────┬───────────┐
│ n_estimators        │ 150       │
│ max_depth           │ 8         │
│ learning_rate       │ 0.0500    │
└─────────────────────┴───────────┘
```

Monospace values, compact row height, no border-collapse visible. Renders only numeric params (same filter as the 3D chart).

### Apply Button

Styled with `background:#8b5cf6` (matching the HyperOpt purple). Appears below the params table. Label: `Apply & Retrain`. Disabled once clicked. Generated inline by `eHoPoll` with a closure capturing `m.best_trial`.

### JS Function: `eHoApply(bestTrial)`

1. Disables button, replaces label with `<span class="spinner"></span> Retraining…`
2. POSTs to `/hyperopt/apply`:
   ```json
   {
     "target_column": "<from e-target>",
     "task_type":     "<from e-task>",
     "model_type":    "<from e-ho-model>",
     "params":        bestTrial.params
   }
   ```
3. On `{job_id}` response → polls `/api/train/status/:id` every 1500 ms
4. On `Completed`:
   - Reads `metrics.model_id` from the completed job
   - Sets `eLastModelId = metrics.model_id`
   - Enables `#e-explain-btn`
   - Calls `eQualityFetch(eLastModelId, 'e-quality-panel')`
   - Builds UMAP canvas (same DOM construction as normal `ePoll`)
   - Calls `eDashModels()`
   - Appends a compact metrics table into `#e-ho-result`
   - Shows success notification via `eNotify`
   - Pushes to history via `eHistPush`
5. On `Failed` → shows error in red, re-enables button

### Error Handling

- If `e-target` or `e-task` are empty (no dataset loaded), show a toast via `eNotify` and do nothing. The "Optimize" button is already disabled without data, so this is belt-and-suspenders.
- Network errors: catch block re-enables the button and shows the error message.

---

## Feature B — Real Latency Tracking

### Storage

Add to `AppState` in `src/server/state.rs`:

```rust
pub prediction_latencies: tokio::sync::Mutex<std::collections::VecDeque<u64>>,
```

- Values in **microseconds** (u64)
- Hard cap: **1000 entries** (pop front when full)
- Initialized as `Mutex::new(VecDeque::new())`

Add helper method on `AppState`:

```rust
pub async fn record_prediction_latency(&self, micros: u64) {
    let mut q = self.prediction_latencies.lock().await;
    if q.len() >= 1000 {
        q.pop_front();
    }
    q.push_back(micros);
}
```

### Measurement Point

In the `predict` handler (`src/server/handlers.rs`), wrap the inference call:

```rust
let t0 = std::time::Instant::now();
let prediction = engine.predict(&row)?;
let elapsed_us = t0.elapsed().as_micros() as u64;
drop(engine_ref);  // release read lock before async call
state.record_prediction_latency(elapsed_us).await;
```

Captures **pure model compute time** — no HTTP overhead, no JSON serialization.

### Percentile Computation

Add a free function in `handlers.rs`:

```rust
fn percentile(sorted: &[u64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted[idx] as f64 / 1000.0  // microseconds → milliseconds
}
```

### SLO Handler Change

Replace hardcoded values with:

```rust
let latencies = {
    let q = state.prediction_latencies.lock().await;
    let mut v: Vec<u64> = q.iter().copied().collect();
    v.sort_unstable();
    v
};

let metrics = SloMetrics {
    latency_p50_ms: percentile(&latencies, 50.0),
    latency_p95_ms: percentile(&latencies, 95.0),
    latency_p99_ms: percentile(&latencies, 99.0),
    availability_percent: availability,
    error_rate,
    total_requests,
};
```

### Fallback Behaviour

If no predictions have been made yet (`latencies` is empty), `percentile` returns `0.0`. The SLO dashboard will show `0.00 ms` — accurate (nothing measured yet) rather than misleading fictional values.

---

## Files Changed

| File | Change |
|---|---|
| `src/server/state.rs` | Add `prediction_latencies` field + `record_prediction_latency` method |
| `src/server/handlers.rs` | (A) Extend `eHoPoll` completion HTML + add `eHoApply` JS function; (B) wrap predict inference + add `percentile` fn + fix SLO handler |

No new files. No new routes (backend `/hyperopt/apply` already exists).

---

## Security

All user-derived strings in the new JS (model_id, param keys/values) are passed through the existing `escHtml()` helper before being placed in `innerHTML`. Param values rendered in the table use `toFixed(4)` for numbers, never raw string injection.
