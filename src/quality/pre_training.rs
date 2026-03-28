//! Gate 1: Pre-training quality checks.

use ndarray::{Array1, Array2};
use crate::quality::{FindingSeverity, QualityContext, QualityFinding};

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
}
