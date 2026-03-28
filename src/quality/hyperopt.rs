//! Gate 3: Hyperopt quality — multi-objective, EI-noise, convergence.

use crate::quality::{ParetoPoint, QualityContext};

// ── Multi-objective scalarization ─────────────────────────────────────────────

pub struct MultiObjectiveConfig {
    pub metric_weight: f64,  // default 0.8
    pub latency_weight: f64, // default 0.2
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self { Self { metric_weight: 0.8, latency_weight: 0.2 } }
}

/// Normalize and combine metric score + latency into a single scalar.
/// `metric_range` and `latency_range` are (min, max) across all observed trials.
/// Returns 0.0–1.0.
pub fn scalarized_score(
    metric: f64,
    latency_ms: f64,
    metric_range: (f64, f64),
    latency_range: (f64, f64),
    config: &MultiObjectiveConfig,
) -> f64 {
    let norm_metric = if (metric_range.1 - metric_range.0).abs() > 1e-10 {
        (metric - metric_range.0) / (metric_range.1 - metric_range.0)
    } else { 0.5 };
    let norm_latency = if (latency_range.1 - latency_range.0).abs() > 1e-10 {
        1.0 - (latency_ms - latency_range.0) / (latency_range.1 - latency_range.0)
    } else { 0.5 };
    (config.metric_weight * norm_metric + config.latency_weight * norm_latency)
        .clamp(0.0, 1.0)
}

// ── Pareto front ──────────────────────────────────────────────────────────────

/// Returns the non-dominated subset of `trials` (maximising both metric_score and minimising latency_ms).
/// A point is dominated if another point has both equal-or-better metric AND equal-or-lower latency,
/// with at least one strictly better.
pub fn compute_pareto_front(trials: &[ParetoPoint]) -> Vec<ParetoPoint> {
    let mut front: Vec<ParetoPoint> = Vec::new();
    for candidate in trials {
        let dominated = front.iter().any(|p| {
            p.metric_score >= candidate.metric_score && p.latency_ms <= candidate.latency_ms
                && (p.metric_score > candidate.metric_score || p.latency_ms < candidate.latency_ms)
        });
        if !dominated {
            // Remove any existing front members dominated by candidate
            front.retain(|p| {
                !(candidate.metric_score >= p.metric_score && candidate.latency_ms <= p.latency_ms
                    && (candidate.metric_score > p.metric_score || candidate.latency_ms < p.latency_ms))
            });
            front.push(candidate.clone());
        }
    }
    front
}

// ── ConvergenceDetector ───────────────────────────────────────────────────────

/// Tracks convergence across multiple calls to `observe`.
pub struct ConvergenceTracker {
    patience: usize,
    min_delta: f64,
    best: f64,
    stagnant_count: usize,
}

impl ConvergenceTracker {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self { patience, min_delta, best: f64::NEG_INFINITY, stagnant_count: 0 }
    }

    /// Call after each trial with the best score seen so far.
    /// Returns true when patience is exhausted (convergence declared).
    pub fn observe(&mut self, best_so_far: f64) -> bool {
        if best_so_far > self.best + self.min_delta {
            self.best = best_so_far;
            self.stagnant_count = 0;
        } else {
            self.stagnant_count += 1;
        }
        self.stagnant_count >= self.patience
    }
}

/// Stateful convergence detector — use this in tests and in the orchestration loop.
pub struct StatefulConvergenceDetector {
    tracker: ConvergenceTracker,
}

impl StatefulConvergenceDetector {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self { tracker: ConvergenceTracker::new(patience, min_delta) }
    }

    pub fn update(&mut self, best_so_far: f64) -> bool {
        self.tracker.observe(best_so_far)
    }
}

/// Apply convergence detection across a list of best-so-far scores.
/// Returns the index at which convergence was declared, or None.
pub fn detect_convergence(
    best_scores: &[f64],
    patience: usize,
    min_delta: f64,
) -> Option<usize> {
    let mut tracker = ConvergenceTracker::new(patience, min_delta);
    for (i, &score) in best_scores.iter().enumerate() {
        if tracker.observe(score) {
            return Some(i);
        }
    }
    None
}

// ── HyperoptGate ──────────────────────────────────────────────────────────────

/// Configuration for the hyperopt quality gate.
pub struct HyperoptGate {
    pub config: MultiObjectiveConfig,
    pub patience: usize,
    pub min_delta: f64,
}

impl Default for HyperoptGate {
    fn default() -> Self {
        Self {
            config: MultiObjectiveConfig::default(),
            patience: 20,
            min_delta: 0.001,
        }
    }
}

impl HyperoptGate {
    /// Record hyperopt results into ctx.
    pub fn record(
        &self,
        ctx: &mut QualityContext,
        trials: Vec<ParetoPoint>,
        best_scores_over_time: &[f64],
    ) {
        ctx.trials_run = trials.len();
        ctx.pareto_front = compute_pareto_front(&trials);
        ctx.hyperopt_converged = detect_convergence(
            best_scores_over_time,
            self.patience,
            self.min_delta,
        ).is_some();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalarized_score_weights() {
        let config = MultiObjectiveConfig::default(); // metric_weight=0.8, latency_weight=0.2
        // Best case: metric=1.0, latency at minimum → score=1.0
        let score = scalarized_score(1.0, 10.0, (0.5, 1.0), (10.0, 100.0), &config);
        assert!((score - 1.0).abs() < 1e-10, "got {}", score);
        // Worst case: metric=0.5, latency at maximum → score=0.0
        let score2 = scalarized_score(0.5, 100.0, (0.5, 1.0), (10.0, 100.0), &config);
        assert!((score2 - 0.0).abs() < 1e-10, "got {}", score2);
    }

    #[test]
    fn test_pareto_front_non_dominated() {
        let trials = vec![
            ParetoPoint { metric_score: 0.9, latency_ms: 50.0, trial_id: 0 },
            ParetoPoint { metric_score: 0.8, latency_ms: 20.0, trial_id: 1 },
            ParetoPoint { metric_score: 0.7, latency_ms: 100.0, trial_id: 2 }, // dominated by trial 0
        ];
        let front = compute_pareto_front(&trials);
        assert_eq!(front.len(), 2);
        let ids: Vec<usize> = front.iter().map(|p| p.trial_id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_convergence_detector_detects_plateau() {
        let mut detector = StatefulConvergenceDetector::new(5, 0.001);
        for i in 0..6 {
            // All scores within min_delta of 0.90
            let converged = detector.update(0.9 + i as f64 * 0.0001);
            if i < 5 { assert!(!converged); }
            else { assert!(converged, "should converge after patience exhausted"); }
        }
    }

    #[test]
    fn test_convergence_detector_resets_on_improvement() {
        let mut detector = StatefulConvergenceDetector::new(3, 0.001);
        detector.update(0.80);
        detector.update(0.80);
        let converged = detector.update(0.85); // improvement > min_delta — reset
        assert!(!converged, "should not converge after improvement");
    }
}
