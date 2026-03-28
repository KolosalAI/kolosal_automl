//! Gate 2: Training quality — algorithm selection, CV strategy, ensemble.

use crate::quality::{CvStrategy, QualityContext};

// ── AlgorithmSelector ─────────────────────────────────────────────────────────

pub struct AlgorithmSelector {
    pub small_dataset_threshold: usize,   // default 1_000
    pub large_dataset_threshold: usize,   // default 100_000
    pub high_dim_ratio: f64,              // n_features/n_samples threshold, default 1.0
}

impl Default for AlgorithmSelector {
    fn default() -> Self {
        Self {
            small_dataset_threshold: 1_000,
            large_dataset_threshold: 100_000,
            high_dim_ratio: 1.0,
        }
    }
}

impl AlgorithmSelector {
    /// Returns a list of algorithm names that should be excluded for this dataset.
    pub fn excluded_algorithms(&self, ctx: &QualityContext) -> Vec<String> {
        let mut excluded: Vec<String> = Vec::new();
        let n = ctx.n_samples;
        let d = ctx.n_features;

        if n < self.small_dataset_threshold {
            let slow = ["GradientBoosting", "XGBoost", "LightGBM", "CatBoost", "SGDClassifier", "SGDRegressor"];
            for alg in &slow {
                excluded.push(alg.to_string());
            }
        }

        if n > self.large_dataset_threshold {
            let expensive = ["KNN", "SVM"];
            for alg in &expensive {
                excluded.push(alg.to_string());
            }
        }

        if d > 0 && n > 0 && (d as f64 / n as f64) > self.high_dim_ratio {
            let nonlinear = ["RandomForest", "ExtraTrees", "GradientBoosting",
                             "XGBoost", "LightGBM", "CatBoost", "KNN", "SVM"];
            for alg in &nonlinear {
                if !excluded.contains(&alg.to_string()) {
                    excluded.push(alg.to_string());
                }
            }
        }

        excluded
    }

    /// Write selection rationale into ctx.algorithm_selection_log and return excluded list.
    pub fn apply(&self, ctx: &mut QualityContext) -> Vec<String> {
        let excluded = self.excluded_algorithms(ctx);
        for alg in &excluded {
            ctx.algorithm_selection_log.push(format!(
                "EXCLUDED {} — dataset characteristics: n_samples={}, n_features={}",
                alg, ctx.n_samples, ctx.n_features
            ));
        }
        excluded
    }
}

// ── CvStrategyChooser ─────────────────────────────────────────────────────────

pub struct CvStrategyChooser {
    /// n_samples below which repeated CV is used. Default: 5_000.
    pub repeated_threshold: usize,
}

impl Default for CvStrategyChooser {
    fn default() -> Self { Self { repeated_threshold: 5_000 } }
}

impl CvStrategyChooser {
    /// Choose the appropriate CV strategy for the dataset.
    pub fn choose(&self, ctx: &QualityContext, group_column: Option<String>) -> CvStrategy {
        if let Some(col) = group_column {
            return CvStrategy::GroupKFold { group_column: col };
        }
        if ctx.n_samples < self.repeated_threshold {
            CvStrategy::Repeated3x5Fold
        } else {
            CvStrategy::Stratified5Fold
        }
    }

    /// Choose and write the strategy into ctx.cv_strategy.
    pub fn apply(&self, ctx: &mut QualityContext, group_column: Option<String>) {
        let strategy = self.choose(ctx, group_column);
        ctx.cv_strategy = Some(strategy);
    }
}

// ── TopKSelector ──────────────────────────────────────────────────────────────

pub struct TopKSelector;

impl TopKSelector {
    /// Sort `scores` by mean score descending and return the top k.
    /// `scores` is a slice of (model_name, mean_score, std_score).
    pub fn select(scores: &[(String, f64, f64)], k: usize) -> Vec<(String, f64, f64)> {
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(k).collect()
    }
}

/// Returns true if the ensemble score beats the best single model score by at
/// least `margin`. This is the gating condition for using the ensemble.
pub fn ensemble_beats_single(ensemble_score: f64, single_best: f64, margin: f64) -> bool {
    ensemble_score > single_best + margin
}

// ── TrainingGate ──────────────────────────────────────────────────────────────

/// Orchestrates all training-gate decisions. In the full pipeline this is called
/// after PreTrainingGate and before HyperoptGate.
pub struct TrainingGate {
    pub algorithm_selector: AlgorithmSelector,
    pub cv_chooser: CvStrategyChooser,
    pub top_k: usize,
    pub ensemble_margin: f64,
}

impl Default for TrainingGate {
    fn default() -> Self {
        Self {
            algorithm_selector: AlgorithmSelector::default(),
            cv_chooser: CvStrategyChooser::default(),
            top_k: 3,
            ensemble_margin: 0.001,
        }
    }
}

impl TrainingGate {
    /// Apply algorithm selection and CV strategy to ctx.
    /// Returns the list of excluded algorithm names.
    pub fn prepare(&self, ctx: &mut QualityContext, group_column: Option<String>) -> Vec<String> {
        self.cv_chooser.apply(ctx, group_column);
        self.algorithm_selector.apply(ctx)
    }

    /// Call after all models are trained. `cv_scores` is (model_name, mean, std).
    /// Sets ensemble flags on ctx.
    pub fn finalize(
        &self,
        ctx: &mut QualityContext,
        cv_scores: Vec<(String, f64, f64)>,
        ensemble_score: Option<f64>,
    ) {
        let single_best = cv_scores.iter().map(|(_, m, _)| *m)
            .fold(f64::NEG_INFINITY, f64::max);
        ctx.cv_scores = cv_scores;
        ctx.ensemble_attempted = ensemble_score.is_some();
        ctx.ensemble_beat_single = ensemble_score
            .map(|s| ensemble_beats_single(s, single_best, self.ensemble_margin))
            .unwrap_or(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_selector_small_dataset_excludes_boosting() {
        let mut ctx = QualityContext::default();
        ctx.n_samples = 500;
        let selector = AlgorithmSelector::default();
        let excluded = selector.excluded_algorithms(&ctx);
        assert!(excluded.contains(&"GradientBoosting".to_string()));
        assert!(excluded.contains(&"XGBoost".to_string()));
    }

    #[test]
    fn test_algorithm_selector_large_dataset_excludes_knn() {
        let mut ctx = QualityContext::default();
        ctx.n_samples = 200_000;
        let selector = AlgorithmSelector::default();
        let excluded = selector.excluded_algorithms(&ctx);
        assert!(excluded.contains(&"KNN".to_string()));
        assert!(excluded.contains(&"SVM".to_string()));
    }

    #[test]
    fn test_cv_strategy_small_dataset_uses_repeated() {
        let mut ctx = QualityContext::default();
        ctx.n_samples = 3000;
        let chooser = CvStrategyChooser::default();
        let strategy = chooser.choose(&ctx, None);
        assert!(matches!(strategy, CvStrategy::Repeated3x5Fold));
    }

    #[test]
    fn test_cv_strategy_large_dataset_uses_stratified() {
        let mut ctx = QualityContext::default();
        ctx.n_samples = 10_000;
        let chooser = CvStrategyChooser::default();
        let strategy = chooser.choose(&ctx, None);
        assert!(matches!(strategy, CvStrategy::Stratified5Fold));
    }

    #[test]
    fn test_cv_strategy_group_column() {
        let mut ctx = QualityContext::default();
        ctx.n_samples = 10_000;
        let chooser = CvStrategyChooser::default();
        let strategy = chooser.choose(&ctx, Some("patient_id".to_string()));
        assert!(matches!(strategy, CvStrategy::GroupKFold { .. }));
    }

    #[test]
    fn test_top_k_selector_returns_top_3() {
        let scores = vec![
            ("ModelA".to_string(), 0.80_f64, 0.02),
            ("ModelB".to_string(), 0.95, 0.01),
            ("ModelC".to_string(), 0.70, 0.03),
            ("ModelD".to_string(), 0.88, 0.01),
        ];
        let top = TopKSelector::select(&scores, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, "ModelB"); // highest first
        assert_eq!(top[1].0, "ModelD");
        assert_eq!(top[2].0, "ModelA");
    }

    #[test]
    fn test_ensemble_beats_single_detection() {
        let single_best = 0.90_f64;
        let ensemble_score = 0.92_f64;
        let margin = 0.001;
        assert!(ensemble_beats_single(ensemble_score, single_best, margin));
        assert!(!ensemble_beats_single(0.90, single_best, margin));
    }
}
