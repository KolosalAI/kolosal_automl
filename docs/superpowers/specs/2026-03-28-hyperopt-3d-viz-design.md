# HyperOpt 3D Visualization — Design Spec

**Date:** 2026-03-28
**Status:** Approved

## Goal

Add an interactive 3D scatter plot to the HyperOpt panel that shows the optimization search space: two hyperparameters on X/Y axes and the trial score on the Z axis. The best trial is highlighted with a gold star so users can immediately see where the optimum was found.

## Chosen Approach

**Plotly.js interactive `scatter3d`** — loaded from CDN. Gives rotate/zoom/hover for free with minimal code.

---

## Changes Required

### 1. Backend — Fix params serialization (`src/server/handlers.rs`)

**Problem:** Trial params are currently serialized as a Rust debug string (`format!("{:?}", t.params)`), which is opaque to the frontend.

**Fix:** Serialize params as a proper JSON object. In the `run_hyperopt` handler where trials are mapped to `serde_json::Value`:

```rust
// Before (~line 5358):
"params": format!("{:?}", t.params),

// After:
"params": t.params.iter()
    .filter_map(|(k, v)| {
        let num = match v {
            ParameterValue::Float(f) => Some(serde_json::Value::from(*f)),
            ParameterValue::Int(i)   => Some(serde_json::Value::from(*i)),
            _                        => None,
        };
        num.map(|n| (k.clone(), n))
    })
    .collect::<serde_json::Map<_, _>>(),
```

This makes each trial's `params` a JSON object like `{"learning_rate": 0.05, "max_depth": 6}`.

### 2. Frontend — Load Plotly.js (`src/server/handlers.rs`)

Add a single `<script>` tag in the HTML head (or just before the closing `</body>`) to load Plotly from CDN:

```html
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" defer></script>
```

### 3. Frontend — Add 3D chart container (`src/server/handlers.rs`)

In the `#e-ho-charts` div (currently contains two `<canvas>` cards for convergence and best score), add a third card below the grid:

```html
<div id="e-ho-3d-wrap" style="display:none;margin-top:16px">
  <div class="card">
    <p style="font-size:14px;font-weight:500;margin-bottom:12px;color:#6a6f73">
      Search Space (3D)
    </p>
    <div id="e-ho-3d" style="width:100%;height:380px"></div>
  </div>
</div>
```

### 4. Frontend — Update `eHoCharts(trials)` function (`src/server/handlers.rs`)

Extend the existing `eHoCharts` function to also render the 3D plot after the existing 2D charts.

**Logic:**

1. Collect numeric param keys from the first valid trial.
2. If **0 numeric params**: skip 3D entirely.
3. If **1 numeric param**: render a Plotly 2D scatter (param vs score) — fallback for models like KNN.
4. If **≥2 numeric params**: render a Plotly `scatter3d` using the first two params as X/Y, score as Z.

**Best trial detection:** The trial with the lowest `value` (since the optimizer minimizes) or the one matching `best_trial.trial_id` passed alongside the trials array. Use a gold star marker (`symbol: 'diamond'`, size 14, color `#ffd700`).

**Visual spec:**
- Paper/plot background: `#0f0f1a` (matches existing dark chart style)
- Points colored by score using the `RdYlGn` colorscale
- Best trial: `star` symbol, size 16, color `#ffd700`, text annotation `"BEST"`
- Axes labeled with actual param names
- Hover template: `Trial #%{customdata}<br>Score: %{z:.4f}<br>%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}`
- Margin: `{l:40,r:20,t:20,b:40}`

**Fallback (1 param):** A Plotly `scatter` with the param on X, score on Y, same color scheme, best trial highlighted.

---

## Scope

- No new files created; all changes are in `src/server/handlers.rs`.
- Does not affect any other chart or panel.
- The `params` serialization fix also benefits the `get_hyperopt_history` endpoint (currently stores the same debug string).

## Out of Scope

- Real-time streaming of trials during optimization (3D only shown after completion).
- Bundling Plotly as a static asset (CDN is sufficient for now).
- Showing more than 2 params simultaneously.
