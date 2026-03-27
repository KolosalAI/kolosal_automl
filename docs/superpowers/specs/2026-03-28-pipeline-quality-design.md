# Pipeline Quality Improvement Design

**Date:** 2026-03-28
**Status:** Approved

## Overview

This spec describes a `QualityPipeline` orchestrator and four quality gates that wrap the existing `TrainEngine` and `HyperOptX` to produce better-trained models, more reliable predictions, and smarter automation — without replacing any existing subsystem.

The core insight: quality improvements compound when they share context. A leakage detector that removes features informs algorithm selection. Algorithm selection informs calibration method. Calibration informs OOD thresholds. A shared `QualityContext` struct carries this information forward through the pipeline so each stage makes decisions informed by what came before.

---

## Architecture

```
Upload → [Pre-training Gate] → [Training Gate] → [Hyperopt Gate] → [Post-training Gate] → Model
              ↓                      ↓                   ↓                    ↓
         QualityContext ─────────────────────────────────────────────────────→ QualityReport
```

### New Module: `src/quality/`

```
src/quality/
├── mod.rs              # QualityPipeline orchestrator, QualityContext, QualityReport
├── pre_training.rs     # Gate 1: leakage, encoding safety, cardinality, distribution
├── training.rs         # Gate 2: algorithm selection, CV upgrades, ensemble/stacking
├── hyperopt.rs         # Gate 3: multi-objective, EI-noise, adaptive early stopping
└── post_training.rs    # Gate 4: calibration, confidence intervals, OOD detection
```

### `QualityContext`

Shared mutable state passed through all four gates:

```rust
pub struct QualityContext {
    // Dataset fingerprint (set by pre-training gate)
    pub n_samples: usize,
    pub n_features: usize,
    pub sparsity: f64,
    pub class_balance: Option<f64>,         // None for regression
    pub feature_types: Vec<FeatureType>,    // Numeric, Categorical, Text
    pub training_distribution: Vec<ColumnStats>, // mean, var, skew, kurtosis per column

    // Pre-training decisions
    pub pre_training_findings: Vec<QualityFinding>,
    pub dropped_features: Vec<String>,      // Leakage suspects removed
    pub algorithm_selection_log: Vec<String>, // Rationale for inclusions/exclusions

    // Training decisions
    pub cv_strategy: CvStrategy,            // Stratified, Repeated, Group
    pub cv_scores: Vec<(String, f64, f64)>, // (model_name, mean, std)
    pub ensemble_attempted: bool,
    pub ensemble_beat_single: bool,

    // Hyperopt decisions
    pub pareto_front: Vec<ParetoPoint>,     // (metric_score, latency_ms)
    pub hyperopt_converged: bool,
    pub trials_run: usize,

    // Post-training decisions
    pub calibration_method: Option<CalibrationMethod>, // Platt, Isotonic, None
    pub ece_before: Option<f64>,
    pub ece_after: Option<f64>,
    pub ood_threshold: Option<f64>,         // 99th percentile Mahalanobis distance
}
```

### `QualityReport`

Returned from training API alongside `ModelMetrics`. Retrievable via `GET /api/quality/report/:model_id`.

```rust
pub struct QualityReport {
    pub model_id: String,
    pub overall_quality_score: f64,         // 0.0–1.0 composite
    pub pre_training: PreTrainingReport,
    pub training: TrainingReport,
    pub hyperopt: HyperoptReport,
    pub post_training: PostTrainingReport,
}
```

---

## Gate 1: Pre-training (`src/quality/pre_training.rs`)

Runs before any model training. Takes the raw dataset and a mutable `QualityContext`.

### 1.1 Leakage Detection

- Compute Pearson correlation (numeric) and Cramér's V (categorical) between each feature and the target.
- Features with |correlation| > 0.95 are flagged as leakage suspects and added to `QualityContext.dropped_features`.
- Features whose column name contains `{"date", "time", "timestamp", "created_at", "updated_at"}` on non-time-series tasks are flagged with severity Warning.
- Findings written to `pre_training_findings` with severity Info / Warning / Critical and human-readable rationale.

### 1.2 Target Encoding Safety

- Detect any column being encoded with `TargetEncoder`.
- Replace with `SafeTargetEncoder`: fits only on the training fold, transforms validation fold using smoothing fallback for unseen categories.
- Smoothing formula: `encoded = (count * target_mean + m * global_mean) / (count + m)` where `m` is a configurable smoothing factor (default: 10).
- Unseen categories at inference time fall back to the global target mean.

### 1.3 High-Cardinality Auto-Handling

- Columns with cardinality > 50 unique values currently being one-hot encoded are automatically switched:
  - If target is available: switch to `SafeTargetEncoder`
  - If target is not available (unsupervised): switch to `HashingEncoder` with 32 buckets
- Prevents feature explosion. Logged to `pre_training_findings` as Info.

### 1.4 Distribution Fingerprint

- Compute per-column statistics: mean, variance, skew, kurtosis.
- Store as `QualityContext.training_distribution`.
- Used at inference time to detect distribution shift: columns outside 3σ emit a `DistributionShiftWarning` in the prediction response.

---

## Gate 2: Training (`src/quality/training.rs`)

Runs after pre-training gate, before model fitting begins.

### 2.1 Smart Algorithm Selection

Reads `QualityContext` to exclude models unlikely to perform well, narrowing the 26-model candidate set:

| Condition | Excluded | Preferred |
|---|---|---|
| n_samples < 1,000 | GradientBoosting, XGBoost, LightGBM, CatBoost, Neural | Linear, KNN, SVM, NaiveBayes |
| n_samples > 100,000 | KNN, SVM | RandomForest, GradientBoosting, Linear |
| n_features > n_samples | All nonlinear models | Ridge, Lasso, ElasticNet |
| class_balance > 10:1 | Accuracy-optimized variants | F1/AUC-optimized + SMOTE pre-processing |
| n_categorical_features / n_features > 0.7 | Linear, SVM | DecisionTree, RandomForest, GradientBoosting |

- Rationale for each exclusion written to `QualityContext.algorithm_selection_log`.
- User can override via `training_config.force_algorithms: Vec<String>`.

### 2.2 Upgraded Cross-Validation

Replace the existing k-fold with a context-aware CV strategy:

| Condition | Strategy |
|---|---|
| Default | Stratified 5-fold (classification) / 5-fold (regression) |
| n_samples < 5,000 | Repeated 3×5-fold (15 total folds) |
| group_column specified in config | Group k-fold (no same-group leakage) |

- CV mean ± std stored per model in `QualityContext.cv_scores`.
- Strategy written to `QualityContext.cv_strategy`.

### 2.3 Automatic Ensemble/Stacking

After individual model training, take top-3 models by CV score and build two additional candidates:

**Soft-voting ensemble:**
- Classification: average predicted probabilities across top-3 models
- Regression: average predictions across top-3 models

**Stacking ensemble:**
- Generate out-of-fold predictions from top-3 models (5-fold)
- Train a linear meta-learner (Ridge for regression, LogisticRegression for classification) on the OOF predictions
- Final predictions: meta-learner applied to base model outputs

Both candidates compete in the same evaluation. If neither beats the best single model by > 0.001, they are discarded and not returned as the best model. Uses and extends `src/ensemble/` module.

---

## Gate 3: Hyperopt (`src/quality/hyperopt.rs`)

Wraps and upgrades `HyperOptX` in `src/optimizer/`.

### 3.1 Multi-Objective Optimization

- Add a second objective: prediction latency (measured per trial by timing a batch of 1,000 predictions on held-out data).
- Each trial result stored as `(metric_score, latency_ms)` on a Pareto front in `QualityContext.pareto_front`.
- Best trial selected by weighted scalarization: `score = 0.8 * normalized_metric + 0.2 * (1 - normalized_latency)`.
- Weights configurable via `OptimizationConfig.objective_weights`.
- Full Pareto front returned in `QualityReport.hyperopt.pareto_front` — users can pick a different trade-off after training.

### 3.2 Expected Improvement with Noise (EI-noise)

Replace the current acquisition function in `src/optimizer/gaussian_process.rs`:

- Current: GP posterior mean (pure exploitation)
- New: EI-noise = `EI(x) = (μ(x) - f* - ξ) * Φ(Z) + σ(x) * φ(Z)` where `Z = (μ(x) - f* - ξ) / σ(x)`, `ξ` is exploration weight (default: 0.01), decayed over trials
- Effect: more exploration early in the search, more exploitation later. Fewer wasted trials on marginal improvements.

### 3.3 Adaptive Early Stopping

- Add successive halving with warm starts: trials that survive pruning start from the best known hyperparameters rather than sampling fresh from the prior.
- Add convergence detection: if best score hasn't improved by > 0.001 in the last 20 trials, halt and set `QualityContext.hyperopt_converged = true`.
- Distinguish converged (good) from timed-out (budget exhausted) in `QualityReport`.

---

## Gate 4: Post-training (`src/quality/post_training.rs`)

Runs after the best model is selected. Uses a held-out calibration set (15% of training data, stratified, not used in CV).

### 4.1 Probability Calibration (Classification only)

- Fit two calibrators on the calibration set: Platt scaling (logistic) and Isotonic regression.
- Select the calibrator with lower Expected Calibration Error (ECE) on the calibration set.
- ECE before/after stored in `QualityContext.ece_before` and `QualityContext.ece_after`.
- Calibration applied transparently: prediction endpoint returns calibrated probabilities.
- If neither calibrator improves ECE by > 0.005, no calibration is applied (model is already well-calibrated).

**ECE formula:** `ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|` with 10 equal-width bins.

### 4.2 Prediction Confidence Intervals (Regression)

- Conformal prediction using calibration set residuals.
- For each prediction `ŷ`, return `[ŷ - q̂, ŷ + q̂]` where `q̂` is the `⌈(1 + coverage) / 2⌉`-th quantile of calibration residuals.
- Default coverage: 90%. Configurable via `inference_config.conformal_coverage`.
- Interval width stored in `InferenceStats.avg_interval_width` as a quality signal.
- Response JSON adds `lower_bound` and `upper_bound` fields to regression predictions.

### 4.3 Entropy-Based Confidence (Classification)

- Return `confidence = 1 - H(p) / log(n_classes)` alongside each prediction.
- `confidence = 1.0` means the model is certain; `confidence = 0.0` means maximum uncertainty.
- Added as a `confidence` field in classification prediction responses.

### 4.4 Out-of-Distribution Detection

- Fit Mahalanobis distance detector on training features:
  - Compute training set centroid and covariance matrix
  - Threshold = 99th percentile of training distances
- At inference time: inputs beyond threshold get `ood_warning: true` in response.
- `QualityContext.ood_threshold` stores the threshold value.
- `QualityReport` includes the fraction of calibration set within threshold (sanity check — should be ~99%).

---

## API Changes

### Updated Training Response

`POST /api/train` response gains a `quality_report` field alongside `metrics`:

```json
{
  "model_id": "...",
  "metrics": { ... },
  "quality_report": {
    "overall_quality_score": 0.84,
    "pre_training": { "findings": [...], "dropped_features": [...] },
    "training": { "cv_strategy": "Repeated3x5", "ensemble_used": true },
    "hyperopt": { "converged": true, "trials_run": 47, "pareto_front": [...] },
    "post_training": { "calibration_method": "Isotonic", "ece_before": 0.12, "ece_after": 0.04, "ood_threshold": 3.2 }
  }
}
```

### New Endpoint

`GET /api/quality/report/:model_id` — retrieve the quality report for any trained model.

### Updated Prediction Response

Classification:
```json
{ "prediction": "class_a", "probabilities": {...}, "confidence": 0.91, "ood_warning": false }
```

Regression:
```json
{ "prediction": 42.3, "lower_bound": 38.1, "upper_bound": 46.5, "ood_warning": false }
```

---

## Testing

- **Unit tests** per gate: each gate tested in isolation with mock `QualityContext`
- **Integration test**: full `QualityPipeline` on iris (classification) and boston housing (regression) — assert ECE improves, ensemble beats single model on held-out test set
- **Regression test**: benchmark suite must not regress below current 16/18 wins vs Python baseline
- **Property test**: conformal intervals must achieve stated coverage on held-out data (empirical coverage ≥ 90% when configured at 90%)

---

## Out of Scope

- Time-series-specific quality improvements (separate future spec)
- Unsupervised model quality (clustering quality metrics)
- UI changes for displaying the quality report (separate frontend spec)
- Distributed/multi-node training
