//! Gate 4: Post-training quality — calibration, conformal intervals, OOD.

use crate::quality::{CalibrationMethod, QualityContext};

// ── ECE ───────────────────────────────────────────────────────────────────────

/// Expected Calibration Error over `n_bins` equal-width bins.
pub fn expected_calibration_error(probs: &[f64], labels: &[bool], n_bins: usize) -> f64 {
    let n = probs.len() as f64;
    let bin_width = 1.0 / n_bins as f64;
    let mut ece = 0.0_f64;
    for b in 0..n_bins {
        let lower = b as f64 * bin_width;
        let upper = lower + bin_width;
        let bin: Vec<(f64, bool)> = probs.iter().zip(labels.iter())
            .filter(|(p, _)| **p >= lower && (**p < upper || (b == n_bins - 1 && **p <= 1.0)))
            .map(|(p, l)| (*p, *l))
            .collect();
        if bin.is_empty() { continue; }
        let bin_n = bin.len() as f64;
        let acc = bin.iter().filter(|(_, l)| *l).count() as f64 / bin_n;
        let conf = bin.iter().map(|(p, _)| p).sum::<f64>() / bin_n;
        ece += (bin_n / n) * (acc - conf).abs();
    }
    ece
}

// ── PlattCalibrator ───────────────────────────────────────────────────────────

/// Logistic (Platt) calibration: P_cal = 1 / (1 + exp(a·score + b)).
/// Fit via gradient descent on the calibration set.
pub struct PlattCalibrator {
    pub a: f64,
    pub b: f64,
}

impl PlattCalibrator {
    pub fn fit(scores: &[f64], labels: &[f64]) -> Self {
        let mut a = 0.0_f64;
        let mut b = 0.0_f64;
        let lr = 0.01;
        let n = scores.len() as f64;
        for _ in 0..1_000 {
            let mut grad_a = 0.0_f64;
            let mut grad_b = 0.0_f64;
            for (&s, &y) in scores.iter().zip(labels.iter()) {
                let p = 1.0 / (1.0 + (-(a * s + b)).exp());
                grad_a += (p - y) * s;
                grad_b += p - y;
            }
            a -= lr * grad_a / n;
            b -= lr * grad_b / n;
        }
        PlattCalibrator { a, b }
    }

    pub fn predict(&self, score: f64) -> f64 {
        1.0 / (1.0 + (-(self.a * score + self.b)).exp())
    }
}

// ── IsotonicCalibrator ────────────────────────────────────────────────────────

/// Isotonic regression calibration via Pool Adjacent Violators (PAV).
pub struct IsotonicCalibrator {
    /// Sorted score thresholds (midpoints of each block).
    pub thresholds: Vec<f64>,
    /// Calibrated probability for each block.
    pub values: Vec<f64>,
}

impl IsotonicCalibrator {
    pub fn fit(scores: &[f64], labels: &[f64]) -> Self {
        assert_eq!(scores.len(), labels.len());
        let mut pairs: Vec<(f64, f64)> = scores.iter()
            .zip(labels.iter())
            .map(|(&s, &l)| (s, l))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // PAV: blocks of (label_sum, score_sum, count)
        let mut blocks: Vec<(f64, f64, usize)> = Vec::new();
        for (score, label) in &pairs {
            blocks.push((*label, *score, 1));
            while blocks.len() > 1 {
                let n = blocks.len();
                let prev_mean = blocks[n - 2].0 / blocks[n - 2].2 as f64;
                let curr_mean = blocks[n - 1].0 / blocks[n - 1].2 as f64;
                if prev_mean > curr_mean {
                    let last = blocks.pop().unwrap();
                    let prev = blocks.last_mut().unwrap();
                    prev.0 += last.0;
                    prev.1 += last.1;
                    prev.2 += last.2;
                } else {
                    break;
                }
            }
        }

        let thresholds: Vec<f64> = blocks.iter().map(|(_, s, c)| s / *c as f64).collect();
        let values: Vec<f64> = blocks.iter().map(|(l, _, c)| l / *c as f64).collect();
        IsotonicCalibrator { thresholds, values }
    }

    /// Step-function interpolation.
    pub fn predict(&self, score: f64) -> f64 {
        match self.thresholds.binary_search_by(|t| t.partial_cmp(&score).unwrap_or(std::cmp::Ordering::Less)) {
            Ok(i) => self.values[i],
            Err(i) => {
                if i == 0 { self.values[0] }
                else if i >= self.values.len() { *self.values.last().unwrap_or(&0.5) }
                else { self.values[i - 1] }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platt_calibrator_moves_scores_toward_labels() {
        let scores = vec![0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.7];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let cal = PlattCalibrator::fit(&scores, &labels);
        assert!(cal.predict(0.9) > 0.5, "high score should give high probability");
        assert!(cal.predict(0.1) < 0.5, "low score should give low probability");
    }

    #[test]
    fn test_isotonic_calibrator_is_monotone() {
        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let cal = IsotonicCalibrator::fit(&scores, &labels);
        let p1 = cal.predict(0.2);
        let p2 = cal.predict(0.6);
        let p3 = cal.predict(0.8);
        assert!(p1 <= p2, "isotonic must be non-decreasing: {} <= {}", p1, p2);
        assert!(p2 <= p3, "isotonic must be non-decreasing: {} <= {}", p2, p3);
    }

    #[test]
    fn test_ece_perfect_calibration_is_zero() {
        // Use midpoint probabilities (0.05, 0.95) so each falls in a single bin
        // with accuracy matching confidence: acc=0/50=0 ≈ conf=0.05, acc=50/50=1 ≈ conf=0.95
        let probs: Vec<f64> = (0..50).map(|_| 0.05).chain((0..50).map(|_| 0.95)).collect();
        let labels: Vec<bool> = (0..50).map(|_| false).chain((0..50).map(|_| true)).collect();
        let ece = expected_calibration_error(&probs, &labels, 10);
        assert!(ece < 0.05, "ECE should be near 0 for well-calibrated model, got {}", ece);
    }

    #[test]
    fn test_ece_overconfident_model_has_high_ece() {
        let probs: Vec<f64> = vec![0.99; 100];
        let labels: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let ece = expected_calibration_error(&probs, &labels, 10);
        assert!(ece > 0.4, "Overconfident model should have high ECE, got {}", ece);
    }
}
