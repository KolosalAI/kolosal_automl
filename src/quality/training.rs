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
}
