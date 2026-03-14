# Codebase Optimization Design

**Date:** 2026-03-15
**Scope:** Full codebase performance optimization — algorithms, memory, parallelism, concurrency
**Approach:** File-by-file ordered by impact severity; build + test once at the end
**Parallelism target:** Conservative (container-safe) — parallelize over columns only

---

## Goals

- Eliminate O(n²)/O(n³) algorithms in preprocessing and feature engineering
- Remove unnecessary DataFrame clones throughout the pipeline
- Add conservative rayon parallelism where operations are column-independent
- Reduce async lock contention in the server layer
- Move CPU-bound training off the async executor

## Non-Goals

- No API or struct interface changes
- No new features
- No aggressive thread-pool sizing (container-safe only)
- No SIMD intrinsics

---

## Section 1: Algorithmic Fixes

### `src/preprocessing/feature_selection.rs`

**Problem:** `CorrelationFilter::fit()` is O(n³) — nested O(n²) loop over feature pairs, each calling `mean_correlation()` which is itself O(n).

**Fix:**
- Precompute all column means and stds once before the loop
- Compute the full pairwise correlation matrix in one O(n²) pass and store it in an `Array2<f64>`
- Replace `mean_correlation()` calls with direct lookups into the precomputed matrix
- RFE: cache importance scores per feature; remove the worst scoring feature(s) per step without recomputing all scores
- `discretize()`: single pass computing min, max, and bin index together instead of three separate iterations

**Expected gain:** O(n³) → O(n²) for correlation filter; O(n·k) → O(n) per RFE step

---

### `src/feature_engineering/interactions.rs`

**Problem:** Element-wise `result[[i, j]]` access in the sample loop destroys cache locality. Correlation precomputation in `AutoCrossing::fit()` recomputes means/stds redundantly per pair.

**Fix:**
- Replace element-wise row/column writes with ndarray column-slice assignments (`s![.., col_idx]`)
- Precompute all column means and stds once before the crossing correlation loop
- Replace `Vec::clone()` calls in `get_feature_names()` with reference iteration

**Expected gain:** 2–5× on large interaction matrices due to cache locality

---

### `src/autopipeline/composer.rs`

**Problem:** Column list is filtered 3+ times in sequence to partition into `high_missing`, `remaining_missing`, and `ok` buckets. `Vec::contains()` used for membership testing.

**Fix:**
- Single-pass partition into three buckets using conditional collection
- Replace `Vec::contains()` with `HashSet` for O(1) membership checks

---

## Section 2: Memory / Clone Elimination

### `src/preprocessing/encoder.rs`

**Problem:** Every `transform_*()` method starts with `df.clone()` then adds columns one-by-one in a loop, causing O(columns) full DataFrame reallocations. Hash encoding re-iterates all rows once per component.

**Fix:**
- Remove `df.clone()` at entry of each transform method
- Accumulate all new columns into `Vec<Column>` during the loop
- Build the result `DataFrame` once at the end via `DataFrame::new()`
- Hash encoding: compute all component values per row in a single row-iteration pass

**Expected gain:** O(columns) allocations → O(1) per transform call

---

### `src/preprocessing/scaler.rs`

**Problem:** `transform()` clones the entire DataFrame then mutates it column-by-column. Robust scaler calls `quantile()` twice on the same sorted data.

**Fix:**
- Collect all scaled columns into `Vec<Column>`, then build result DataFrame once
- For robust scaler: sort values once, read Q1 from index `n/4` and Q3 from index `3n/4` in the same sorted slice

---

### `src/preprocessing/pipeline.rs`

**Problem:** Defensive `df.clone()` in `else` branches of imputer chain creates copies of data that are immediately discarded. Column order preservation in `cast_numeric_to_f64` uses O(n·m) post-sort.

**Fix:**
- Pass a mutable reference through the imputer chain; only clone when an actual transform occurs
- Column order: build an index map `{col_name → original_position}` once before casting; use it to order output without sorting

---

## Section 3: Conservative Parallelism

All parallelism added uses `rayon::par_iter()` over **columns only** (not rows). Column count is bounded (typically 10–200), preventing over-subscription in containerized environments. Rayon and ndarray's rayon feature are already declared dependencies.

| File | Parallelized operation | Safety |
|------|----------------------|--------|
| `encoder.rs` | Frequency map computation and hash component generation per column | Each column fully independent |
| `scaler.rs` | `self.params.par_iter()` when scaling columns | No shared mutable state |
| `pipeline.rs` | Column stat computation in `fit()` (mean, std, null-count per column) | Each column fully independent |
| `feature_selection.rs` | Outer loop of correlation matrix computation | Each `(i, j)` pair writes to unique index in pre-allocated `Array2` |
| `interactions.rs` | Crossing pair loop in `transform()` | Each crossing writes to a distinct column slice |

**Not parallelized:**
- `engine.rs` training: uses `tokio::task::spawn_blocking` instead (avoids rayon/tokio thread pool conflict)
- `handlers.rs` request handlers: already async
- `state.rs`: restructured for lock efficiency instead

---

## Section 4: Concurrency & Server Layer

### `src/server/state.rs`

**Problem:** `RwLock<HashMap>` for model/engine caches serializes all concurrent read accesses during writes. `evict_stale_entries()` sorts the full jobs list to find oldest entries.

**Fix:**
- Add `dashmap` dependency; replace `RwLock<HashMap<String, TrainEngine>>` with `DashMap` for engine/model caches
- `evict_stale_entries()`: replace sort with `min_by_key` pass — O(n) instead of O(n log n)
- Guard eviction write-lock acquisition behind a count check to avoid locking on every call

---

### `src/training/engine.rs`

**Problem:** `train_model()` runs CPU-intensive training synchronously in an async context, blocking the tokio executor. DataFrame → Array2 conversion reads column-by-column into a row-major buffer causing cache misses.

**Fix:**
- Wrap `train_model()` body in `tokio::task::spawn_blocking`
- DataFrame → Array2: build column-major `Vec<Vec<f64>>` first, then transpose in one allocation to row-major layout

---

### `src/server/handlers.rs`

**Problem:** Skewness and kurtosis computation in `analyze_data()` iterates column values twice (once for `Σ((x-μ)/σ)³`, once for `Σ((x-μ)/σ)⁴`). Lock scopes held across awaited operations increase contention window.

**Fix:**
- Merge skewness and kurtosis into a single pass computing both `sum3` and `sum4` in the same iterator
- Reduce lock scope: drop read locks before any `.await` call

---

## Execution Order

Files worked in order of impact severity:

1. `src/preprocessing/feature_selection.rs` — O(n³) → O(n²), CRITICAL
2. `src/preprocessing/encoder.rs` — clone storm, CRITICAL
3. `src/preprocessing/scaler.rs` — redundant clone, HIGH
4. `src/preprocessing/pipeline.rs` — chain clones, HIGH
5. `src/feature_engineering/interactions.rs` — cache-hostile O(n²), HIGH
6. `src/training/engine.rs` — blocking async + cache-miss layout, MEDIUM-HIGH
7. `src/autopipeline/composer.rs` — repeated filtering, MEDIUM
8. `src/server/state.rs` — lock contention, MEDIUM
9. `src/server/handlers.rs` — per-request inefficiencies, MEDIUM

**Verification:** `cargo build` after all files, then full `cargo test` once at the end.

---

## Dependencies to Add

| Crate | Version | Reason |
|-------|---------|--------|
| `dashmap` | `"6"` | Fine-grained concurrent HashMap for model/engine caches |

All other optimizations use existing dependencies (rayon, ndarray, tokio).
