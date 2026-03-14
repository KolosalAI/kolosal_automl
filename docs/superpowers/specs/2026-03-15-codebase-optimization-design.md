# Codebase Optimization Design

**Date:** 2026-03-15
**Scope:** Full codebase performance optimization — algorithms, memory, parallelism, concurrency
**Approach:** File-by-file ordered by impact severity; `cargo check` after each file, `cargo test` + `cargo bench` at the end
**Parallelism target:** Conservative (container-safe) — parallelize over columns only

---

## Goals

- Eliminate O(n²)/O(n³) algorithms in preprocessing and feature engineering
- Remove unnecessary DataFrame clones throughout the pipeline
- Add conservative rayon parallelism where operations are column-independent
- Reduce async lock contention in the server layer

## Non-Goals

- No API or struct interface changes
- No new features
- No aggressive thread-pool sizing (container-safe only)
- No SIMD intrinsics

---

## Floating-Point Reproducibility Note

Adding `par_iter` to reduction operations (sum of correlations, skewness/kurtosis) breaks floating-point associativity — partial sums are reduced in non-deterministic thread order, meaning the same input can produce slightly different values across runs. This affects:

- Correlation matrix values (and therefore which feature is eliminated on a tie)
- Skewness/kurtosis values in `analyze_data`

**Accepted trade-off:** The pipeline already has non-deterministic feature selection (RFE uses `HashSet` iteration order for final output). The tiny floating-point delta from parallel reductions is acceptable for this use case. This is documented here and must be noted in commit messages for affected files.

---

## Section 1: Algorithmic Fixes

### `src/preprocessing/feature_selection.rs`

**Problem:** `CorrelationFilter::fit()` is O(n³) — an O(n²) loop over feature pairs, each calling `mean_correlation()` which is itself O(n).

**Fix:**
- Precompute all column means and stds once before the loop
- Compute the full pairwise correlation matrix in one O(n²) pass, storing it in an `Array2<f64>`
- Replace each `mean_correlation()` call with: sum row `i` of the precomputed matrix, divide by `(n_features - n_excluded)` — **not a single-cell lookup**, but a row-mean over the already-computed values (O(n) read, no recomputation)
- RFE: the current loop recomputes importance scores for all remaining features on each iteration. Change to compute all scores once per iteration pass (they already depend only on the `remaining` set, not on each other); the saving is eliminating the redundant full-pass recomputation
- `discretize()`: single pass computing min, max, and bin index together instead of three separate `.fold()` / `.map()` iterations

**Expected gain:** O(n³) → O(n²) for correlation filter

---

### `src/feature_engineering/interactions.rs`

**Problem:** Element-wise `result[[i, j]]` access in the sample loop destroys cache locality. `AutoCrossing::correlation()` recomputes column means and stds once per pair call.

**Fix:**
- Replace element-wise row/column writes with ndarray column-slice assignments using `s![.., col_idx]` — eliminates the cache-hostile row-major strided access
- Precompute all column means and stds once before the crossing correlation loop in `AutoCrossing::fit()`
- Replace `names.extend(input_names.clone())` with `names.extend_from_slice(input_names)` — same allocation, clearer intent

**Expected gain:** 2–5× on large interaction matrices due to cache locality

---

### `src/autopipeline/composer.rs`

**Problem:** Column list is filtered 3+ times in sequence. `Vec::contains()` used for membership testing (O(n) per check).

**Fix:**
- Single-pass partition into `high_missing` / `remaining_missing` / `ok` buckets
- Replace `Vec::contains()` with `HashSet` for O(1) membership checks

---

## Section 2: Memory / Clone Elimination

### `src/preprocessing/encoder.rs`

**Problem:** Every `transform_*()` method starts with `df.clone()` then adds columns one-by-one in a loop via `.with_column(...).clone()`, causing O(columns) full DataFrame reallocations.

**Fix:**
- Remove `df.clone()` at entry of each transform method
- Accumulate all new columns into `Vec<Column>` during the loop
- Build the result `DataFrame` once at the end via `DataFrame::new()`
- Hash encoding: flip loop order to row-outer / component-inner so each row is read once and all component values are computed together; column names must still be generated consistently for downstream compatibility

**Expected gain:** O(columns) allocations → O(1) per transform call

---

### `src/preprocessing/scaler.rs`

**Note:** The current `transform()` already accumulates scaled columns into `Vec<Series>`. However, it still ends with `let mut result = df.clone()` followed by N `with_column` calls — so there is one full DataFrame clone plus N mutations.

**Fix:**
- After collecting scaled columns into `Vec<Column>`, build result directly from the original non-scaled columns plus the new scaled ones — no `df.clone()` needed
- For robust scaler: sort values once, read Q1 from index `n/4` and Q3 from index `3n/4` in the same sorted slice (currently calls `quantile()` twice)

---

### `src/preprocessing/pipeline.rs`

**Problem (1):** When no imputer is configured, `fit()` passes `df.clone()` to the next stage. The clone is used (not discarded), but it is unnecessary — a borrow would suffice.

**Fix:** Restructure the imputer chain to work with a `&DataFrame` reference; only materialise an owned `DataFrame` when a transform actually occurs.

**Problem (2):** `cast_numeric_to_f64` preserves column order via `sort_by_key` — O(n log n). The existing `orig_order` lookup is already an index map; the sort can be replaced by direct-index placement into a pre-sized output `Vec` for O(n).

---

## Section 3: Conservative Parallelism

All parallelism uses `rayon::par_iter()` over **columns only** (not rows). Column count is bounded (typically 10–200). Rayon and ndarray's rayon feature are already declared dependencies (`Cargo.toml` lines 23, 27).

### Array2 parallel writes — borrow checker constraint

Naive `par_iter` over `(i, j)` index pairs writing into a shared `Array2` will not compile — `Array2::column_mut` borrows the whole array. Safe parallel writes require one of:

- `ndarray::parallel::par_azip!` macro (element-wise parallel assignment)
- `array.axis_iter_mut(Axis(0)).into_par_iter()` to split by rows into independent mutable slices

All `Array2` parallel write targets in this spec must use one of these two patterns.

| File | Parallelized operation | Pattern |
|------|----------------------|--------|
| `encoder.rs` | Frequency map computation and hash component generation per column | `par_iter` over column map entries (read-only input, independent output `Vec`s) |
| `scaler.rs` | Scaling each column | `self.params.par_iter()` → collect `Vec<Column>` (no shared mutable state) |
| `pipeline.rs` | Column stat computation in `fit()` | `par_iter` over column names → collect stats (each column fully independent) |
| `feature_selection.rs` | Outer loop of correlation matrix computation | `axis_iter_mut(Axis(0)).into_par_iter()` — each row is an independent mutable slice |
| `interactions.rs` | Crossing pair loop in `transform()` | `axis_iter_mut(Axis(1)).into_par_iter()` — each output column is an independent mutable slice |

**Not parallelized:**
- `engine.rs` training: already wrapped in `tokio::task::spawn_blocking` at the handler level in `handlers.rs` — applying it again inside `train_model` itself would risk deadlock on the blocking thread pool. No change needed here.
- `handlers.rs` request handlers: already async
- `state.rs`: restructured for lock efficiency instead

---

## Section 4: Concurrency & Server Layer

### `src/server/state.rs`

**Problem:** `RwLock<HashMap<String, TrainEngine>>` for engine/model caches serializes all concurrent readers during any write. `evict_stale_entries()` sorts the full jobs list to find oldest entries.

**Note:** `dashmap` is already present in `Cargo.toml` at line 71 (`dashmap = "6.1"`). No Cargo.toml change needed — only usage migration.

**Also note:** `DashMap` iteration order is non-deterministic (shard-based). The existing `train_engines` eviction already has no ordering guarantee; DashMap does not make this worse. The `jobs` eviction is being changed to `min_by_key` which is deterministic regardless of map type.

**Fix:**
- Migrate `train_engines: RwLock<HashMap<...>>` and `models: RwLock<HashMap<...>>` to `DashMap` — eliminates write-lock blocking on concurrent reads
- `evict_stale_entries()` for jobs: replace `sort_by_key` with `min_by_key` pass — O(n) instead of O(n log n)
- Guard eviction write-lock acquisition behind a count check to avoid locking on every periodic call

---

### `src/server/handlers.rs`

**Problem (1):** Skewness and kurtosis computation in `analyze_data()` iterates column values twice (one pass for `Σ((x-μ)/σ)³`, a second pass for `Σ((x-μ)/σ)⁴`). The same double-pass exists in the `get_data_quality` handler.

**Fix:** Merge into a single iterator pass computing both `sum3` and `sum4` simultaneously. Apply to both `analyze_data` and `get_data_quality`.

**Problem (2):** `analyze_data` holds `state.current_data.read()` for the entire function body, blocking all writers during CPU-intensive analysis of large datasets. There are no `.await` points in this function — the standard "drop before await" guidance does not apply, but the lock scope is still too wide.

**Fix:** Clone the DataFrame immediately after acquiring the read lock, then drop the lock before beginning analysis — the same pattern already used correctly in `preprocess_data` (handlers.rs lines 671–676).

---

## Execution Order

| # | File | Severity | Primary Fix |
|---|------|----------|-------------|
| 1 | `src/preprocessing/feature_selection.rs` | CRITICAL | O(n³) → O(n²) |
| 2 | `src/preprocessing/encoder.rs` | CRITICAL | Clone storm elimination |
| 3 | `src/preprocessing/scaler.rs` | HIGH | Residual clone + double quantile |
| 4 | `src/preprocessing/pipeline.rs` | HIGH | Imputer chain borrow + cast sort |
| 5 | `src/feature_engineering/interactions.rs` | HIGH | Cache-hostile loop + correlation precompute |
| 6 | `src/autopipeline/composer.rs` | MEDIUM | Single-pass partition + HashSet |
| 7 | `src/server/state.rs` | MEDIUM | DashMap migration + O(n) eviction |
| 8 | `src/server/handlers.rs` | MEDIUM | Single-pass skew/kurt in `analyze_data` + `get_data_quality`; clone-then-drop lock in both handlers |

---

## Verification Plan

- **After each file:** `cargo check` — catch compile errors (especially borrow checker issues with `Array2` parallelism) before moving on
- **After all 8 files:** `cargo build --release` then `cargo test`
- **Performance validation:** Run `cargo bench` before starting (baseline) and after all changes (comparison) using the existing `benches/preprocessing.rs` and `benches/training.rs`
- **Numerical regression:** Assert that existing test assertions on feature selection output still hold — if any test checks specific selected feature indices, a flip is a signal to investigate the float-ordering trade-off
