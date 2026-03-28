# Pipeline Quality — Frontend Visibility Design

**Date:** 2026-03-28
**Branch:** feat/pipeline-quality
**Status:** Approved

## Goal

Surface all pipeline quality improvements in the frontend so users can see what the system did, understand why each decision was made, and trust the resulting model.

The quality pipeline already runs four gates (pre-training, training, hyperopt, post-training) and produces a `QualityReport` struct with sub-scores and detailed findings. None of this data is currently displayed in the UI. This spec describes exactly how to make it visible.

---

## Placement

**Option chosen: quality score badge everywhere + full detail panel in the Train tab.**

- A **quality score badge** (colored pill) appears next to each model name in the Train tab leaderboard.
- A **Quality Report panel** appears below the training results table in the Train tab, populated when the user selects a model.
- An **OOD warning banner** appears in the Predict area when a prediction input is flagged as out-of-distribution.

No new tab is added. Everything lives where users already look after training.

---

## Badge Colors

| Score range | Color  | Meaning                        |
|-------------|--------|--------------------------------|
| ≥ 80        | Green  | Pipeline ran well               |
| 60–79       | Yellow | Minor issues, usable            |
| < 60        | Red    | Notable issues, review findings |

Score is `overall_quality_score * 100` (rounded to integer).

---

## Quality Report Panel

Located in the Train tab, below the existing training results table. Rendered by JS after a `GET /api/quality/report/:model_id` call. Shown only when a model is selected from the leaderboard.

### Trigger

The quality panel fetches when a model row is clicked in the leaderboard table. If an existing row-click handler is present in the Train tab JS, hook into it. If no click handler exists, add one that: (a) highlights the selected row, (b) reads the `model_id` from a `data-model-id` attribute on the row, (c) calls `fetchQualityReport(modelId)`. If training just completed and there is only one model, auto-fetch without requiring a click.

### States

- **Loading**: skeleton shimmer while fetch is in flight.
- **Pending**: `{ "error": "not found" }` response → show "Quality analysis pending..." (training may still be running or quality pipeline was not run).
- **Loaded**: render full panel.

### Panel Layout

```
┌─────────────────────────────────────────────────────────┐
│  Pipeline Quality Report          Overall Score: [87]   │
│  Model: xgboost_v1                                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐                     │
│  │ Pre-training │  │   Training   │                     │
│  │    100 / 100 │  │    70 / 100  │                     │
│  │ ▼ findings   │  │ ▼ findings   │                     │
│  └──────────────┘  └──────────────┘                     │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │   Hyperopt   │  │ Post-training│                     │
│  │   100 / 100  │  │    75 / 100  │                     │
│  │ ▼ findings   │  │ ▼ findings   │                     │
│  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

### Gate Card Contents

**Pre-training card** (sub-score derived from critical findings):
- Sub-score pill
- Findings list: each `QualityFinding` shown with severity icon (🔴 Critical, 🟡 Warning, ℹ️ Info), category, message, and optional column name
- Dropped features list (if any)
- Leakage suspects list (if any)
- High-cardinality columns list (if any)

**Training card** (sub-score 1.0 if ensemble beat single, 0.7 otherwise):
- Sub-score pill
- CV strategy used (e.g. "Stratified 5-Fold", "Repeated 3×5 Fold", "Group K-Fold (patient_id)")
- Algorithms excluded list with reasons (parsed from `algorithms_excluded` entries that start with "EXCLUDED ")
- Ensemble result: "Ensemble beat single model ✓" or "Ensemble did not improve on best single model"

**Hyperopt card** (sub-score 1.0 if converged, 0.6 otherwise):
- Sub-score pill
- Trials run count
- Converged: yes/no
- Pareto front: a small scatter plot (canvas, ~200×120px) with metric score on X axis and latency on Y axis; each point labeled with trial ID on hover. Only rendered if `pareto_front` has ≥ 2 points.

**Post-training card** (sub-score = ECE improvement ratio or 0.5 if skipped):
- Sub-score pill
- Calibration method used (Platt / Isotonic / None)
- ECE before → after (e.g. "0.200 → 0.050")
- OOD threshold value (if set), with tooltip: "Inputs with normalised distance above this threshold are flagged as out-of-distribution"
- Calibration set fraction (if set)

### Gate Card Interaction

Each gate card is **expanded by default** (all findings visible). A chevron toggle collapses/expands the findings list. This ensures quality data is seen on first load without requiring any click.

---

## OOD Warning Banner

Location: the Predict tab / inline prediction result area.

When `/api/predict` returns `"ood_warning": true` (already implemented in the backend), display:

```
⚠ This input may be out of distribution.
  The model was trained on different data patterns — prediction confidence may be lower than expected.
```

Style: yellow background banner, dismissible with ×. Shown above the prediction result, not replacing it.

---

## API Used

| Endpoint | When called | Notes |
|---|---|---|
| `GET /api/quality/report/:model_id` | User selects model in Train tab leaderboard | Returns `QualityReport` JSON or 404 |
| `GET /api/predict` (existing) | User submits prediction | Already returns `ood_warning: bool` |

No new API endpoints needed.

---

## Files Changed

| File | Change |
|---|---|
| `src/quality/mod.rs` | Add `GateScores { pre: f64, training: f64, hyperopt: f64, post_training: f64 }` field to `QualityReport`; populate it in `QualityReport::from_context` |
| `src/server/handlers.rs` | Add quality panel HTML/CSS, quality fetch JS, OOD banner JS — all inline within the existing monolithic template |

The `GateScores` addition is required because the four sub-scores are computed transiently in `from_context` but not currently stored. Without them in the JSON response, the frontend cannot display per-gate scores. This is a minimal struct change — no scoring logic changes.

---

## Sub-score Computation Reference

These are the scoring rules already implemented in `src/quality/mod.rs`:

| Gate | Score formula |
|---|---|
| Pre-training | `(1.0 - critical_count × 0.2).clamp(0, 1)` |
| Training | `1.0` if ensemble beat single, `0.7` otherwise |
| Hyperopt | `1.0` if converged, `0.6` otherwise |
| Post-training | `(ece_before - ece_after) / ece_before` clamped to [0,1]; `1.0` if ece_before=0; `0.5` if skipped |
| **Overall** | Average of the four sub-scores |

Displayed sub-scores are multiplied by 100 and rounded to integers.

---

## Out of Scope

- No changes to `QualityReport` struct or backend scoring logic.
- No real-time quality score updates during training (quality report is only available after training completes).
- No export or download of quality report (can be added later).
- No quality data in the Dashboard tab (may be added later as a summary card).
