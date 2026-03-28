//! Pipeline quality gates and orchestrator.
//!
//! Provides a [`QualityPipeline`] that wraps training with four sequential
//! gates sharing a [`QualityContext`], producing a [`QualityReport`].

pub mod pre_training;
pub mod training;
pub mod hyperopt;
pub mod post_training;

use serde::{Deserialize, Serialize};

// ── Shared types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FeatureType {
    Numeric,
    Categorical,
    Text,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub name: String,
    pub mean: f64,
    pub variance: f64,
    pub skew: f64,
    pub kurtosis: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FindingSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFinding {
    pub severity: FindingSeverity,
    pub category: String,
    pub message: String,
    pub column: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CvStrategy {
    Stratified5Fold,
    Repeated3x5Fold,
    GroupKFold { group_column: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub metric_score: f64,
    pub latency_ms: f64,
    pub trial_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CalibrationMethod {
    Platt,
    Isotonic,
}

// ── QualityContext ────────────────────────────────────────────────────────────

/// Shared mutable state passed through all four quality gates.
/// Each gate reads earlier decisions and writes its own outputs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityContext {
    // Dataset fingerprint (set by pre-training gate)
    pub n_samples: usize,
    pub n_features: usize,
    pub sparsity: f64,
    pub class_balance: Option<f64>,
    pub feature_types: Vec<FeatureType>,
    pub training_distribution: Vec<FeatureStats>,

    // Pre-training decisions
    pub pre_training_findings: Vec<QualityFinding>,
    pub dropped_features: Vec<String>,
    /// Log entries for algorithm selection decisions.
    /// Exclusion entries MUST start with "EXCLUDED " (e.g., "EXCLUDED XGBoost — reason").
    /// This prefix is used by `QualityReport::from_context` to extract `algorithms_excluded`.
    pub algorithm_selection_log: Vec<String>,

    // Training decisions
    pub cv_strategy: Option<CvStrategy>,
    pub cv_scores: Vec<(String, f64, f64)>, // (model_name, mean, std)
    pub ensemble_attempted: bool,
    pub ensemble_beat_single: bool,

    // Hyperopt decisions
    pub pareto_front: Vec<ParetoPoint>,
    pub hyperopt_converged: bool,
    pub trials_run: usize,

    // Post-training decisions
    pub calibration_method: Option<CalibrationMethod>,
    pub ece_before: Option<f64>,
    pub ece_after: Option<f64>,
    pub ood_threshold: Option<f64>,
    pub calibration_in_set_fraction: Option<f64>,
}

// ── QualityReport ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreTrainingReport {
    pub findings: Vec<QualityFinding>,
    pub dropped_features: Vec<String>,
    pub leakage_suspects: Vec<String>,
    pub high_cardinality_columns: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingReport {
    pub cv_strategy: String,
    pub cv_scores: Vec<(String, f64, f64)>,
    pub ensemble_attempted: bool,
    pub ensemble_beat_single: bool,
    pub algorithms_excluded: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HyperoptReport {
    pub converged: bool,
    pub trials_run: usize,
    pub pareto_front: Vec<ParetoPoint>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PostTrainingReport {
    pub calibration_method: Option<String>,
    pub ece_before: Option<f64>,
    pub ece_after: Option<f64>,
    pub ood_threshold: Option<f64>,
    pub calibration_in_set_fraction: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateScores {
    pub pre: f64,
    pub training: f64,
    pub hyperopt: f64,
    pub post_training: f64,
}

/// Full quality report returned alongside ModelMetrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityReport {
    pub model_id: String,
    pub overall_quality_score: f64,
    pub gate_scores: GateScores,
    pub pre_training: PreTrainingReport,
    pub training: TrainingReport,
    pub hyperopt: HyperoptReport,
    pub post_training: PostTrainingReport,
}

impl QualityReport {
    /// Build from a completed QualityContext. Computes overall_quality_score as
    /// the average of four sub-scores (0.0–1.0 each).
    pub fn from_context(model_id: String, ctx: &QualityContext) -> Self {
        // Sub-score 1: pre-training — penalise critical findings
        let critical_count = ctx.pre_training_findings.iter()
            .filter(|f| f.severity == FindingSeverity::Critical)
            .count();
        let pre_score = (1.0 - (critical_count as f64 * 0.2)).clamp(0.0, 1.0);

        // Sub-score 2: training — reward ensemble beating single model
        let training_score = if ctx.ensemble_beat_single { 1.0 } else { 0.7 };

        // Sub-score 3: hyperopt — reward convergence
        let hyperopt_score = if ctx.hyperopt_converged { 1.0 } else { 0.6 };

        // Sub-score 4: post-training — reward calibration improvement
        let post_score = match (ctx.ece_before, ctx.ece_after) {
            (Some(before), Some(after)) if before > 0.0 => {
                ((before - after) / before).clamp(0.0, 1.0)
            }
            (Some(_), Some(_)) => 1.0, // already perfectly calibrated (ece_before == 0.0)
            _ => 0.5, // calibration was skipped
        };

        let overall = (pre_score + training_score + hyperopt_score + post_score) / 4.0;

        QualityReport {
            model_id,
            overall_quality_score: overall,
            gate_scores: GateScores {
                pre: pre_score,
                training: training_score,
                hyperopt: hyperopt_score,
                post_training: post_score,
            },
            pre_training: PreTrainingReport {
                findings: ctx.pre_training_findings.clone(),
                dropped_features: ctx.dropped_features.clone(),
                leakage_suspects: ctx.pre_training_findings.iter()
                    .filter(|f| f.category == "LeakageDetection")
                    .filter_map(|f| f.column.clone())
                    .collect(),
                high_cardinality_columns: ctx.pre_training_findings.iter()
                    .filter(|f| f.category == "HighCardinality")
                    .filter_map(|f| f.column.clone())
                    .collect(),
            },
            training: TrainingReport {
                cv_strategy: match &ctx.cv_strategy {
                    Some(CvStrategy::Stratified5Fold) => "Stratified5Fold".into(),
                    Some(CvStrategy::Repeated3x5Fold) => "Repeated3x5Fold".into(),
                    Some(CvStrategy::GroupKFold { group_column }) => format!("GroupKFold({})", group_column),
                    None => "Default".into(),
                },
                cv_scores: ctx.cv_scores.clone(),
                ensemble_attempted: ctx.ensemble_attempted,
                ensemble_beat_single: ctx.ensemble_beat_single,
                algorithms_excluded: ctx.algorithm_selection_log.iter()
                    .filter(|s| s.starts_with("EXCLUDED"))
                    .cloned()
                    .collect(),
            },
            hyperopt: HyperoptReport {
                converged: ctx.hyperopt_converged,
                trials_run: ctx.trials_run,
                pareto_front: ctx.pareto_front.clone(),
            },
            post_training: PostTrainingReport {
                calibration_method: ctx.calibration_method.as_ref().map(|m| match m {
                    CalibrationMethod::Platt => "Platt".to_string(),
                    CalibrationMethod::Isotonic => "Isotonic".to_string(),
                }),
                ece_before: ctx.ece_before,
                ece_after: ctx.ece_after,
                ood_threshold: ctx.ood_threshold,
                calibration_in_set_fraction: ctx.calibration_in_set_fraction,
            },
        }
    }
}

// ── QualityPipeline ───────────────────────────────────────────────────────────

/// Top-level orchestrator. Wraps TrainEngine and HyperOptX with four quality
/// gates sharing a QualityContext. Call `run_pre_training` before fitting,
/// then `run_post_training` after the best model is selected.
///
/// The caller (training handler in handlers.rs) is responsible for fitting
/// models via TrainEngine; this struct handles the quality wrapping.
#[derive(Debug, Clone)]
pub struct QualityPipeline {
    pub pre_training_leakage_threshold: f64,
    pub cv_small_dataset_threshold: usize,
    pub ensemble_top_k: usize,
    pub ensemble_margin: f64,
    pub ood_percentile: f64,
    pub conformal_coverage: f64,
}

impl Default for QualityPipeline {
    fn default() -> Self {
        Self {
            pre_training_leakage_threshold: 0.95,
            cv_small_dataset_threshold: 5_000,
            ensemble_top_k: 3,
            ensemble_margin: 0.001,
            ood_percentile: 0.99,
            conformal_coverage: 0.90,
        }
    }
}

// TODO: uncomment when gate files are implemented
// impl QualityPipeline {
//     /// Phase 1: Run pre-training gate. Returns indices of features to keep.
//     /// Call before fitting any model.
//     pub fn run_pre_training(
//         &self,
//         features: &ndarray::Array2<f64>,
//         target: &ndarray::Array1<f64>,
//         feature_names: &[String],
//         ctx: &mut QualityContext,
//     ) -> Vec<usize> {
//         use crate::quality::pre_training::PreTrainingGate;
//         let gate = PreTrainingGate {
//             leakage_threshold: self.pre_training_leakage_threshold,
//             cardinality_threshold: 50,
//             shift_sigma: 3.0,
//         };
//         gate.run(features, target, feature_names, ctx)
//     }
//
//     /// Phase 2: Record training gate decisions into ctx.
//     /// Call after all CV scores are known, before hyperopt.
//     pub fn record_training_decisions(
//         &self,
//         ctx: &mut QualityContext,
//         cv_scores: Vec<(String, f64, f64)>,
//         ensemble_score: Option<f64>,
//         group_column: Option<String>,
//     ) {
//         use crate::quality::training::{TrainingGate, AlgorithmSelector, CvStrategyChooser};
//         let gate = TrainingGate {
//             algorithm_selector: AlgorithmSelector::default(),
//             cv_chooser: CvStrategyChooser { repeated_threshold: self.cv_small_dataset_threshold },
//             top_k: self.ensemble_top_k,
//             ensemble_margin: self.ensemble_margin,
//         };
//         gate.prepare(ctx, group_column);
//         gate.finalize(ctx, cv_scores, ensemble_score);
//     }
//
//     /// Phase 3: Record hyperopt decisions into ctx.
//     /// Call after hyperopt completes.
//     pub fn record_hyperopt_decisions(
//         &self,
//         ctx: &mut QualityContext,
//         pareto_points: Vec<ParetoPoint>,
//         converged: bool,
//         trials_run: usize,
//     ) {
//         ctx.pareto_front = pareto_points;
//         ctx.hyperopt_converged = converged;
//         ctx.trials_run = trials_run;
//     }
//
//     /// Phase 4: Run post-training gate on calibration set.
//     /// `raw_scores`: model's raw output on calibration set (probabilities or regression preds).
//     /// `cal_targets`: true labels/values on calibration set.
//     /// `features`: calibration feature matrix (for OOD detector).
//     /// Returns the finalised QualityReport.
//     pub fn run_post_training(
//         &self,
//         model_id: String,
//         raw_scores: &[f64],
//         cal_targets: &[f64],
//         features: &ndarray::Array2<f64>,
//         ctx: &mut QualityContext,
//     ) -> QualityReport {
//         use crate::quality::post_training::{PostTrainingGate, OodDetector};
//         let gate = PostTrainingGate {
//             ece_improvement_threshold: 0.005,
//             conformal_coverage: self.conformal_coverage,
//             ood_percentile: self.ood_percentile,
//             n_ece_bins: 10,
//         };
//         let cal_labels: Vec<f64> = cal_targets.to_vec();
//         gate.select_calibration(raw_scores, &cal_labels, ctx);
//
//         let detector = OodDetector::fit(features, self.ood_percentile);
//         ctx.ood_threshold = Some(detector.threshold);
//
//         QualityReport::from_context(model_id, ctx)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_context_default() {
        let ctx = QualityContext::default();
        assert_eq!(ctx.n_samples, 0);
        assert!(ctx.pre_training_findings.is_empty());
        assert!(ctx.dropped_features.is_empty());
        assert!(!ctx.hyperopt_converged);
        assert!(ctx.calibration_method.is_none());
        assert!(ctx.ood_threshold.is_none());
    }

    #[test]
    fn test_quality_report_serialization() {
        let report = QualityReport::default();
        let json = serde_json::to_string(&report).unwrap();
        let _: QualityReport = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_report_from_context_overall_score() {
        let ctx = QualityContext {
            hyperopt_converged: true,
            ensemble_beat_single: true,
            ece_before: Some(0.20),
            ece_after: Some(0.05),
            ..Default::default()
        };
        let report = QualityReport::from_context("m1".into(), &ctx);
        // pre_score=1.0 (no critical findings), training_score=1.0 (ensemble beats single),
        // hyperopt_score=1.0 (converged), post_score=(0.20-0.05)/0.20=0.75 → overall=0.9375
        assert!((report.overall_quality_score - 0.9375).abs() < 1e-9,
            "expected 0.9375, got {}", report.overall_quality_score);
        assert!(report.overall_quality_score >= 0.0 && report.overall_quality_score <= 1.0);
    }

    #[test]
    fn test_report_from_context_perfect_calibration_scores_full() {
        // When ece_before=0.0 (already perfect), post_score should be 1.0, not 0.5
        let ctx = QualityContext {
            ece_before: Some(0.0),
            ece_after: Some(0.0),
            ..Default::default()
        };
        let report = QualityReport::from_context("m2".into(), &ctx);
        // post_score=1.0 (perfect calibration), training_score=0.7, hyperopt_score=0.6, pre_score=1.0
        // overall = (1.0 + 0.7 + 0.6 + 1.0) / 4.0 = 0.825
        assert!((report.overall_quality_score - 0.825).abs() < 1e-9,
            "expected 0.825, got {}", report.overall_quality_score);
    }

    #[test]
    fn test_report_gate_scores_populated() {
        let ctx = QualityContext {
            hyperopt_converged: true,
            ensemble_beat_single: true,
            ece_before: Some(0.20),
            ece_after: Some(0.05),
            ..Default::default()
        };
        let report = QualityReport::from_context("m1".into(), &ctx);
        // pre_score=1.0 (no critical findings)
        assert!((report.gate_scores.pre - 1.0).abs() < 1e-9);
        // training_score=1.0 (ensemble beat single)
        assert!((report.gate_scores.training - 1.0).abs() < 1e-9);
        // hyperopt_score=1.0 (converged)
        assert!((report.gate_scores.hyperopt - 1.0).abs() < 1e-9);
        // post_score=(0.20-0.05)/0.20=0.75
        assert!((report.gate_scores.post_training - 0.75).abs() < 1e-9);
    }
}
