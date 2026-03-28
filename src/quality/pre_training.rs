//! Gate 1: Pre-training quality checks.

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use crate::quality::{FindingSeverity, QualityContext, QualityFinding};
use crate::quality::FeatureStats;

// ── Pearson correlation ───────────────────────────────────────────────────────

pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 { return 0.0; }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let numerator: f64 = x.iter().zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    let var_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
    let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    if var_x == 0.0 || var_y == 0.0 { return 0.0; }
    numerator / (var_x.sqrt() * var_y.sqrt())
}

// ── LeakageDetector ───────────────────────────────────────────────────────────

pub struct LeakageDetector {
    /// Correlation threshold above which a feature is flagged. Default: 0.95.
    pub threshold: f64,
}

impl Default for LeakageDetector {
    fn default() -> Self { Self { threshold: 0.95 } }
}

const TEMPORAL_KEYWORDS: &[&str] = &[
    "date", "time", "timestamp", "created_at", "updated_at",
];

impl LeakageDetector {
    /// Returns one QualityFinding per flagged feature.
    pub fn detect(
        &self,
        features: &Array2<f64>,
        target: &Array1<f64>,
        feature_names: &[String],
    ) -> Vec<QualityFinding> {
        let target_slice = target.as_slice().unwrap();
        let mut findings = Vec::new();

        for (i, name) in feature_names.iter().enumerate() {
            let col = features.column(i).to_owned();
            let col_slice = col.as_slice().unwrap();
            let corr = pearson_correlation(col_slice, target_slice).abs();

            if corr > self.threshold {
                findings.push(QualityFinding {
                    severity: FindingSeverity::Critical,
                    category: "LeakageDetection".into(),
                    message: format!(
                        "Feature '{}' has |correlation| {:.3} with target (threshold: {:.2}). \
                         Likely target leakage — dropping.",
                        name, corr, self.threshold
                    ),
                    column: Some(name.clone()),
                });
            }

            let lower = name.to_lowercase();
            if TEMPORAL_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
                findings.push(QualityFinding {
                    severity: FindingSeverity::Warning,
                    category: "TemporalLeakage".into(),
                    message: format!(
                        "Feature '{}' appears to contain temporal information which may cause \
                         leakage on non-time-series tasks.",
                        name
                    ),
                    column: Some(name.clone()),
                });
            }
        }

        findings
    }
}

// ── SafeTargetEncoder ─────────────────────────────────────────────────────────

/// Target encoder that fits only on the training fold and uses smoothing to
/// prevent leakage. Unseen categories fall back to the global target mean.
#[derive(Debug, Clone)]
pub struct SafeTargetEncoder {
    /// Smoothing factor m. Higher = more regularisation toward global mean.
    pub smoothing: f64,
    pub global_mean: f64,
    /// (sum_of_targets, count) per category
    category_stats: HashMap<String, (f64, usize)>,
}

impl Default for SafeTargetEncoder {
    fn default() -> Self {
        Self { smoothing: 10.0, global_mean: 0.0, category_stats: HashMap::new() }
    }
}

impl SafeTargetEncoder {
    /// Fit on a single fold's training data. Call this before transforming the
    /// corresponding validation fold.
    pub fn fit_fold(&mut self, categories: &[String], targets: &[f64]) {
        assert_eq!(categories.len(), targets.len());
        self.global_mean = targets.iter().sum::<f64>() / targets.len().max(1) as f64;
        self.category_stats.clear();
        for (cat, &t) in categories.iter().zip(targets.iter()) {
            let entry = self.category_stats.entry(cat.clone()).or_insert((0.0, 0));
            entry.0 += t;
            entry.1 += 1;
        }
    }

    /// Transform a single category value using smoothed estimate.
    /// encoded = (count * cat_mean + m * global_mean) / (count + m)
    pub fn transform(&self, category: &str) -> f64 {
        match self.category_stats.get(category) {
            Some(&(sum, count)) => {
                let count_f = count as f64;
                let cat_mean = sum / count_f;
                (count_f * cat_mean + self.smoothing * self.global_mean)
                    / (count_f + self.smoothing)
            }
            None => self.global_mean,
        }
    }
}

// ── HighCardinalityHandler ────────────────────────────────────────────────────

pub struct HighCardinalityHandler {
    /// Columns with more unique values than this are considered high-cardinality.
    pub threshold: usize,
}

impl Default for HighCardinalityHandler {
    fn default() -> Self { Self { threshold: 50 } }
}

impl HighCardinalityHandler {
    /// Returns true if the column values have more unique values than the threshold.
    pub fn is_high_cardinality(&self, values: &[String]) -> bool {
        let unique: std::collections::HashSet<&String> = values.iter().collect();
        unique.len() > self.threshold
    }

    /// Returns a QualityFinding for a high-cardinality column.
    pub fn finding(&self, column_name: &str, unique_count: usize) -> QualityFinding {
        QualityFinding {
            severity: FindingSeverity::Info,
            category: "HighCardinality".into(),
            message: format!(
                "Column '{}' has {} unique values (threshold: {}). \
                 Switching from one-hot encoding to target/hash encoding.",
                column_name, unique_count, self.threshold
            ),
            column: Some(column_name.to_string()),
        }
    }
}

// ── DistributionFingerprint ───────────────────────────────────────────────────

impl FeatureStats {
    /// Returns true if `value` is more than `sigma_threshold` standard deviations
    /// from the training mean. Used at inference time for shift detection.
    pub fn is_shifted(&self, value: &f64, sigma_threshold: f64) -> bool {
        let std = self.variance.sqrt();
        if std < 1e-10 { return false; } // constant column — can't detect shift
        ((value - self.mean) / std).abs() > sigma_threshold
    }
}

pub struct DistributionFingerprint;

impl DistributionFingerprint {
    /// Compute per-column statistics from a feature matrix.
    pub fn compute(features: &Array2<f64>, names: &[String]) -> Vec<FeatureStats> {
        let n = features.nrows() as f64;
        names.iter().enumerate().map(|(j, name)| {
            let col: Vec<f64> = features.column(j).iter().copied().collect();
            let mean = col.iter().sum::<f64>() / n;
            let variance = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt().max(1e-10);
            let skew = col.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n;
            let kurtosis = col.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n - 3.0;
            FeatureStats { name: name.clone(), mean, variance, skew, kurtosis }
        }).collect()
    }
}

// ── PreTrainingGate ───────────────────────────────────────────────────────────

/// Runs all pre-training quality checks and writes results into `QualityContext`.
pub struct PreTrainingGate {
    pub leakage_threshold: f64,
    pub cardinality_threshold: usize,
    pub shift_sigma: f64,
}

impl Default for PreTrainingGate {
    fn default() -> Self {
        Self { leakage_threshold: 0.95, cardinality_threshold: 50, shift_sigma: 3.0 }
    }
}

impl PreTrainingGate {
    /// Run all pre-training checks. Populates `ctx` with findings, dropped_features,
    /// and training_distribution. Returns the indices of features to keep.
    pub fn run(
        &self,
        features: &Array2<f64>,
        target: &Array1<f64>,
        feature_names: &[String],
        ctx: &mut QualityContext,
    ) -> Vec<usize> {
        ctx.n_samples = features.nrows();
        ctx.n_features = features.ncols();

        // 1. Leakage detection
        let detector = LeakageDetector { threshold: self.leakage_threshold };
        let leakage_findings = detector.detect(features, target, feature_names);
        let leaked: std::collections::HashSet<String> = leakage_findings.iter()
            .filter(|f| f.severity == FindingSeverity::Critical)
            .filter_map(|f| f.column.clone())
            .collect();
        ctx.dropped_features.extend(leaked.iter().cloned());
        ctx.pre_training_findings.extend(leakage_findings);

        // 2. High-cardinality findings (numeric matrix — flag by name heuristic only;
        //    full categorical detection happens in preprocessing pipeline)
        let hc = HighCardinalityHandler { threshold: self.cardinality_threshold };
        for name in feature_names.iter() {
            // Flag column names that suggest ID columns (likely high cardinality)
            if name.to_lowercase().contains("_id") || name.to_lowercase() == "id" {
                ctx.pre_training_findings.push(hc.finding(name, self.cardinality_threshold + 1));
            }
        }

        // 3. Distribution fingerprint
        let fingerprint = DistributionFingerprint::compute(features, feature_names);
        ctx.training_distribution = fingerprint;

        // Return indices of features to keep (not in dropped_features)
        feature_names.iter().enumerate()
            .filter(|(_, name)| !leaked.contains(*name))
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_leakage_detector_catches_perfect_correlation() {
        let features = array![[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [4.0, 2.0]];
        let target = array![1.0, 2.0, 3.0, 4.0]; // perfectly correlated with col 0, uncorrelated with col 1
        let names = vec!["perfect".to_string(), "ok".to_string()];
        let detector = LeakageDetector { threshold: 0.95 };
        let findings = detector.detect(&features, &target, &names);
        assert!(findings.iter().any(|f| f.column.as_deref() == Some("perfect") && f.severity == FindingSeverity::Critical));
        assert!(!findings.iter().any(|f| f.column.as_deref() == Some("ok") && f.severity == FindingSeverity::Critical));
    }

    #[test]
    fn test_leakage_detector_flags_temporal_names() {
        let features = array![[1.0], [2.0], [3.0]];
        let target = array![0.0, 1.0, 0.0];
        let names = vec!["created_at".to_string()];
        let detector = LeakageDetector { threshold: 0.95 };
        let findings = detector.detect(&features, &target, &names);
        assert!(findings.iter().any(|f| f.category == "TemporalLeakage"));
    }

    #[test]
    fn test_pearson_correlation_known_value() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "Perfect linear relationship should give r=1");
    }

    #[test]
    fn test_pearson_correlation_zero_variance() {
        let x = [1.0, 1.0, 1.0];
        let y = [1.0, 2.0, 3.0];
        let r = pearson_correlation(&x, &y);
        assert_eq!(r, 0.0, "Zero variance should return 0");
    }

    #[test]
    fn test_safe_target_encoder_smoothing() {
        let mut enc = SafeTargetEncoder::default();
        let categories = vec!["a".to_string(), "a".to_string(), "b".to_string()];
        let targets = [1.0, 3.0, 5.0];
        enc.fit_fold(&categories, &targets);
        // global_mean = (1+3+5)/3 = 3.0
        // "a": count=2, sum=4 → cat_mean=2.0 → smoothed = (2*2 + 10*3)/(2+10) = 34/12 ≈ 2.833
        let val_a = enc.transform("a");
        assert!((val_a - 2.833).abs() < 0.01, "got {}", val_a);
        // Unknown category → global_mean = 3.0
        let val_unknown = enc.transform("unseen");
        assert!((val_unknown - 3.0).abs() < 1e-10, "got {}", val_unknown);
    }

    #[test]
    fn test_high_cardinality_handler_detects_over_threshold() {
        let col_values: Vec<String> = (0..60).map(|i| format!("cat_{}", i)).collect();
        let handler = HighCardinalityHandler { threshold: 50 };
        assert!(handler.is_high_cardinality(&col_values));
        let small: Vec<String> = (0..10).map(|i| format!("v{}", i)).collect();
        assert!(!handler.is_high_cardinality(&small));
    }

    #[test]
    fn test_distribution_fingerprint_stats() {
        let features = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
        let names = vec!["a".to_string(), "b".to_string()];
        let fp = DistributionFingerprint::compute(&features, &names);
        assert_eq!(fp.len(), 2);
        assert!((fp[0].mean - 2.5).abs() < 1e-10, "mean of [1,2,3,4] = 2.5, got {}", fp[0].mean);
        assert!(fp[0].variance > 0.0);
    }

    #[test]
    fn test_distribution_fingerprint_shift_detection() {
        // Training: mean=0, std≈1; Inference: column shifted to mean=100
        let features = array![[-1.0], [0.0], [1.0], [0.0]];
        let names = vec!["x".to_string()];
        let fp = DistributionFingerprint::compute(&features, &names);
        let ok_sample = 0.5_f64;
        let shifted_sample = 100.0_f64;
        assert!(!fp[0].is_shifted(&ok_sample, 3.0));
        assert!(fp[0].is_shifted(&shifted_sample, 3.0));
    }

    #[test]
    fn test_pre_training_gate_drops_leaked_features() {
        // Feature 0 is perfectly correlated with target → should be dropped
        let features = array![[1.0, 5.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
        let target = array![1.0, 2.0, 3.0, 4.0];
        let names = vec!["perfect".to_string(), "noise".to_string()];
        let mut ctx = QualityContext::default();
        let gate = PreTrainingGate::default();
        let kept = gate.run(&features, &target, &names, &mut ctx);
        assert!(!kept.contains(&0), "leaked feature should be dropped");
        assert!(kept.contains(&1), "non-leaked feature should be kept");
        assert_eq!(ctx.n_samples, 4);
        assert_eq!(ctx.n_features, 2);
        assert!(!ctx.dropped_features.is_empty());
    }
}
