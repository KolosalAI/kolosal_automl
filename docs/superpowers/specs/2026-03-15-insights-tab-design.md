# Insights Tab — Educational ML Visualizations

**Date:** 2026-03-15
**Status:** Approved

---

## Overview

Add a top-level **Insights** tab to the Kolosal AutoML frontend, sitting alongside Train / Predict / Analysis. It contains four sub-tabs, each a Canvas 2D visualization connected to the user's live session (trained model + uploaded dataset). The goal is to let users understand *how their model thinks* — not just what it outputs.

No external charting libraries. All rendering uses the native HTML5 Canvas 2D API, consistent with the rest of the frontend.

---

## Architecture

### Frontend

- **Location:** `src/server/handlers.rs` — the embedded HTML/JS/CSS string (~6500 lines). The Insights tab is added as a new top-level tab alongside the existing Train/Predict/Analysis tabs.
- **Tab structure:** Top-level `Insights` tab → 4 sub-tabs inside: Step-Through, Concepts, Playground, Metrics.
- **Rendering:** Native Canvas 2D API. Reuse existing helper functions (`eDrawLine`, `eDrawBars`, `eDrawScatter`, `eDrawDonut`, `eDrawMiniHist`, `eDrawSparkline`, `eDrawRadar`) where applicable. New canvas drawing logic added inline.
- **Data connection:** All visualizations are driven by the current session state (loaded model + dataset). When no model is loaded, each sub-tab shows a clear "Train a model first" placeholder.
- **State:** Uses existing JS state pattern (`appState` / `currentModel` globals already present in the frontend).

### Backend (Rust / Axum)

Two new API endpoints added to `src/server/handlers.rs` (Axum router):

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/insights/model-structure` | GET | Returns tree structure (nodes, splits, conditions, impurities) or convergence history for non-tree models |
| `/api/insights/evaluation` | GET | Returns confusion matrix, per-class metrics, ROC points, residuals for the current model evaluated on its test split |

`/api/predict` (existing) is reused as-is by the Playground sub-tab.
`/api/explain/importance` (existing) is reused by the Concepts sub-tab.

---

## Sub-tab 1 — Step-Through (`▶ How It Learned`)

### Purpose
Animate the internals of the trained model — show the user *how* it was built, one step at a time.

### Behaviour
- **Decision Tree / Random Forest:** Renders the tree on Canvas as a node-link diagram (root at top, leaves at bottom). Steps through splits one by one:
  - Highlights the active node in blue
  - Shows the split condition (`petal_length ≤ 2.45`), Gini impurity before/after, and sample counts left/right
  - Sidebar annotation explains the split in plain English: *"The model chose this feature because splitting here reduced uncertainty the most"*
- **Other model types (Linear, SVM, etc.):** Renders the training loss/accuracy curve animating frame by frame. Annotations appear at key inflection points: *"Learning accelerated here"*, *"Validation loss stopped improving — training ended"*.
- **Controls:** Play / Pause / Step Back / Step Forward / Reset. Speed slider (slow / normal / fast).

### Data source
`/api/insights/model-structure` — returns `{ type: "tree", nodes: [...] }` or `{ type: "convergence", history: [...] }`.

### Canvas layout
- Left ~70%: animated diagram
- Right ~30%: annotation panel (plain-text callouts for current step)

---

## Sub-tab 2 — Concept Explainer (`📐 Understand the Model`)

### Purpose
Three annotated diagrams drawn from real session data that explain foundational ML concepts in context of the user's model.

### Panels (stacked vertically, each with a heading and plain-English caption)

**1. Bias-Variance (Training Curve)**
- Line chart: training accuracy vs validation accuracy across epochs/iterations (from training history in `model-structure` response)
- Overfitting zone shaded red if validation diverges from training
- Underfitting zone shaded yellow if both curves plateau early
- Labels: *"Your model fits the training data well but struggles on unseen data"* or *"Good balance"* depending on curve shape

**2. Feature Importance**
- Horizontal bar chart (reuses `/api/explain/importance` data)
- Top 10 features, sorted descending
- Bars colour-coded by magnitude
- Plain-English caption: *"petal_length drove predictions 3× more than the next feature"*

**3. Data Split Breakdown**
- Visual stacked bar showing train / validation / test row counts
- Colour-coded segments
- Caption: *"Your model trained on N rows and was evaluated on M rows it had never seen"*

### Data sources
`/api/insights/model-structure` (training history), `/api/explain/importance` (feature importance), dataset row counts from existing session state.

---

## Sub-tab 3 — Live Playground (`🎛 What If?`)

### Purpose
Feature sliders tied to the real model. Every change fires a prediction and updates the display live — letting users develop intuition for *which inputs matter*.

### Behaviour
- Sliders are generated at load time from the dataset's actual feature names, min/max values, and dtypes (numeric → range slider; categorical → dropdown)
- On every input change: POST to `/api/predict` with current slider values
- Updates:
  - Large prediction label with confidence badge (e.g. *"Iris-virginica — 92%"*)
  - Horizontal confidence bar per class on Canvas
  - *"What pushed the prediction"* panel: top 3 contributing features rendered as arrows with magnitude and direction on Canvas (approximated from feature importance × input deviation from mean)
- Debounced 150ms to avoid hammering the API on rapid slider drags

### Canvas layout
- Top: prediction output (label + confidence bars)
- Bottom: feature attribution arrows panel

### Data sources
`/api/predict` (existing), `/api/explain/importance` (for attribution approximation), dataset feature metadata from session state.

---

## Sub-tab 4 — Metrics Deep Dive (`📊 How Well Did It Do?`)

### Purpose
Visual explanation of model evaluation metrics using the model's actual test-set predictions.

### Classification models

**Confusion Matrix**
- Grid on Canvas, cells colour-coded: TP (green), TN (blue), FP (orange), FN (red)
- Cells fill in one prediction at a time on initial animation
- Click a cell to see sample count and a tooltip explaining what that cell means (*"These 3 samples were actually Setosa but predicted as Virginica"*)

**ROC Curve**
- Drawn incrementally on Canvas, AUC area shaded
- Draggable threshold marker: as user drags, the confusion matrix updates to show the impact of the threshold change
- Diagonal baseline (random classifier) for reference

**Precision-Recall Curve**
- Drawn on Canvas alongside ROC in a split layout
- Current threshold highlighted with a dot and labelled values

### Regression models

**Residual Plot**
- Scatter: actual vs predicted on Canvas
- Perfect prediction line (y=x) drawn in grey
- Points colour-coded by residual magnitude

**Error Distribution**
- Histogram of residuals on Canvas
- Gaussian bell curve overlay

**Summary metrics** (R², MAE, RMSE) displayed as stat cards with a plain-English size comparison: *"On average, predictions are off by ±0.3 — about 4% of the target range"*.

### Data source
`/api/insights/evaluation` — returns `{ task: "classification", confusion_matrix: [...], roc_points: [...], pr_points: [...], auc: 0.97 }` or `{ task: "regression", residuals: [...], r2: 0.94, mae: 0.31, rmse: 0.45 }`.

---

## New API Endpoints

### `GET /api/insights/model-structure`

Returns the internal structure of the current model for visualization.

**Response (decision tree):**
```json
{
  "type": "tree",
  "algorithm": "DecisionTreeClassifier",
  "nodes": [
    {
      "id": 0,
      "feature": "petal_length",
      "threshold": 2.45,
      "gini": 0.667,
      "samples": 120,
      "left_child": 1,
      "right_child": 2
    }
  ],
  "training_history": {
    "train_accuracy": [0.6, 0.75, 0.88, 0.92],
    "val_accuracy": [0.58, 0.72, 0.85, 0.89]
  }
}
```

**Response (non-tree):**
```json
{
  "type": "convergence",
  "algorithm": "LogisticRegression",
  "training_history": {
    "train_loss": [...],
    "val_loss": [...],
    "train_accuracy": [...],
    "val_accuracy": [...]
  }
}
```

### `GET /api/insights/evaluation`

Returns evaluation data computed on the test split held out during training.

**Response (classification):**
```json
{
  "task": "classification",
  "classes": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
  "confusion_matrix": [[13, 0, 0], [0, 10, 1], [0, 2, 14]],
  "roc_points": [{"fpr": 0.0, "tpr": 0.0}, ...],
  "pr_points": [{"precision": 1.0, "recall": 0.0}, ...],
  "auc": 0.972,
  "accuracy": 0.925,
  "per_class": [{"class": "Iris-setosa", "precision": 1.0, "recall": 1.0, "f1": 1.0}]
}
```

**Response (regression):**
```json
{
  "task": "regression",
  "residuals": [{"actual": 1.2, "predicted": 1.1}, ...],
  "r2": 0.94,
  "mae": 0.31,
  "rmse": 0.45,
  "target_range": [0.1, 7.9]
}
```

---

## Error States

- **No model loaded:** Each sub-tab shows a centred placeholder: *"Train a model first to unlock this visualization"* with an arrow pointing to the Train tab.
- **API error:** Toast notification + canvas shows error text inline.
- **Model type not supported for step-through:** Step-Through shows convergence chart fallback for any non-tree model.

---

## Implementation Constraints

- All Canvas rendering uses the existing Canvas 2D API pattern in `handlers.rs` — no new dependencies.
- Animations use `requestAnimationFrame` for smooth playback.
- The Rust backend computes evaluation metrics at prediction/training time and caches the result — the `/api/insights/evaluation` endpoint is a cheap read from cached state, not a recomputation.
- The frontend string in `handlers.rs` will grow by approximately 800–1200 lines.

---

## Out of Scope

- Exporting visualizations as images (future)
- Side-by-side model comparison (future)
- SHAP exact values (approximated from feature importance × input deviation; exact SHAP requires a separate library)
