//! Service Level Objectives (SLO) for ISO 25010 compliance.
//!
//! Defines and monitors SLOs for inference latency, availability,
//! and error rates.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Service Level Objectives configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLevelObjectives {
    /// p50 latency target in milliseconds
    pub latency_p50_ms: f64,
    /// p95 latency target in milliseconds
    pub latency_p95_ms: f64,
    /// p99 latency target in milliseconds
    pub latency_p99_ms: f64,
    /// Availability target as percentage (e.g., 99.9)
    pub availability_percent: f64,
    /// Maximum acceptable error rate (e.g., 0.001 for 0.1%)
    pub error_rate_max: f64,
}

impl Default for ServiceLevelObjectives {
    fn default() -> Self {
        Self {
            latency_p50_ms: 10.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 200.0,
            availability_percent: 99.9,
            error_rate_max: 0.001,
        }
    }
}

/// Current SLO compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloStatus {
    /// SLO definitions
    pub objectives: ServiceLevelObjectives,
    /// Current metrics
    pub current: SloMetrics,
    /// Per-SLO compliance
    pub compliance: Vec<SloCompliance>,
    /// Overall SLO compliance (all passing)
    pub all_met: bool,
    /// Remaining error budget as percentage (0-100)
    pub error_budget_remaining: f64,
    /// Last checked timestamp
    pub checked_at: DateTime<Utc>,
}

/// Current measured metrics for SLO comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloMetrics {
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub availability_percent: f64,
    pub error_rate: f64,
    pub total_requests: u64,
    pub total_errors: u64,
}

/// Compliance status for a single SLO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloCompliance {
    pub name: String,
    pub target: f64,
    pub actual: f64,
    pub met: bool,
    pub margin: f64,
}

/// SLO monitor that compares actual metrics against targets
pub struct SloMonitor {
    objectives: ServiceLevelObjectives,
}

impl SloMonitor {
    /// Create a new SLO monitor with the given objectives
    pub fn new(objectives: ServiceLevelObjectives) -> Self {
        Self { objectives }
    }

    /// Evaluate current metrics against SLOs
    pub fn evaluate(&self, metrics: SloMetrics) -> SloStatus {
        let mut compliance = Vec::new();

        // Latency SLOs (actual should be <= target)
        compliance.push(SloCompliance {
            name: "latency_p50".to_string(),
            target: self.objectives.latency_p50_ms,
            actual: metrics.latency_p50_ms,
            met: metrics.latency_p50_ms <= self.objectives.latency_p50_ms,
            margin: self.objectives.latency_p50_ms - metrics.latency_p50_ms,
        });

        compliance.push(SloCompliance {
            name: "latency_p95".to_string(),
            target: self.objectives.latency_p95_ms,
            actual: metrics.latency_p95_ms,
            met: metrics.latency_p95_ms <= self.objectives.latency_p95_ms,
            margin: self.objectives.latency_p95_ms - metrics.latency_p95_ms,
        });

        compliance.push(SloCompliance {
            name: "latency_p99".to_string(),
            target: self.objectives.latency_p99_ms,
            actual: metrics.latency_p99_ms,
            met: metrics.latency_p99_ms <= self.objectives.latency_p99_ms,
            margin: self.objectives.latency_p99_ms - metrics.latency_p99_ms,
        });

        // Availability SLO (actual should be >= target)
        compliance.push(SloCompliance {
            name: "availability".to_string(),
            target: self.objectives.availability_percent,
            actual: metrics.availability_percent,
            met: metrics.availability_percent >= self.objectives.availability_percent,
            margin: metrics.availability_percent - self.objectives.availability_percent,
        });

        // Error rate SLO (actual should be <= target)
        compliance.push(SloCompliance {
            name: "error_rate".to_string(),
            target: self.objectives.error_rate_max,
            actual: metrics.error_rate,
            met: metrics.error_rate <= self.objectives.error_rate_max,
            margin: self.objectives.error_rate_max - metrics.error_rate,
        });

        let all_met = compliance.iter().all(|c| c.met);

        // Error budget: how much of allowed error rate has been consumed
        let error_budget_remaining = if self.objectives.error_rate_max > 0.0 {
            ((self.objectives.error_rate_max - metrics.error_rate) / self.objectives.error_rate_max * 100.0)
                .max(0.0)
                .min(100.0)
        } else {
            if metrics.error_rate == 0.0 { 100.0 } else { 0.0 }
        };

        SloStatus {
            objectives: self.objectives.clone(),
            current: metrics,
            compliance,
            all_met,
            error_budget_remaining,
            checked_at: Utc::now(),
        }
    }
}

impl Default for SloMonitor {
    fn default() -> Self {
        Self::new(ServiceLevelObjectives::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_slos_met() {
        let monitor = SloMonitor::default();
        let metrics = SloMetrics {
            latency_p50_ms: 5.0,
            latency_p95_ms: 30.0,
            latency_p99_ms: 100.0,
            availability_percent: 99.99,
            error_rate: 0.0001,
            total_requests: 10000,
            total_errors: 1,
        };

        let status = monitor.evaluate(metrics);
        assert!(status.all_met);
        assert!(status.error_budget_remaining > 0.0);
    }

    #[test]
    fn test_latency_slo_breach() {
        let monitor = SloMonitor::new(ServiceLevelObjectives {
            latency_p99_ms: 100.0,
            ..Default::default()
        });
        let metrics = SloMetrics {
            latency_p50_ms: 5.0,
            latency_p95_ms: 30.0,
            latency_p99_ms: 250.0, // exceeds 100ms target
            availability_percent: 99.99,
            error_rate: 0.0001,
            total_requests: 10000,
            total_errors: 1,
        };

        let status = monitor.evaluate(metrics);
        assert!(!status.all_met);
        let p99 = status.compliance.iter().find(|c| c.name == "latency_p99").unwrap();
        assert!(!p99.met);
        assert!(p99.margin < 0.0);
    }

    #[test]
    fn test_error_budget() {
        let monitor = SloMonitor::new(ServiceLevelObjectives {
            error_rate_max: 0.01,
            ..Default::default()
        });
        let metrics = SloMetrics {
            latency_p50_ms: 5.0,
            latency_p95_ms: 30.0,
            latency_p99_ms: 100.0,
            availability_percent: 99.99,
            error_rate: 0.005, // 50% of budget consumed
            total_requests: 10000,
            total_errors: 50,
        };

        let status = monitor.evaluate(metrics);
        assert!((status.error_budget_remaining - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_availability_breach() {
        let monitor = SloMonitor::default(); // 99.9% availability target
        let metrics = SloMetrics {
            latency_p50_ms: 5.0,
            latency_p95_ms: 30.0,
            latency_p99_ms: 100.0,
            availability_percent: 98.0, // Below 99.9% target
            error_rate: 0.0001,
            total_requests: 10000,
            total_errors: 1,
        };

        let status = monitor.evaluate(metrics);
        assert!(!status.all_met);
        let avail = status.compliance.iter().find(|c| c.name == "availability").unwrap();
        assert!(!avail.met);
        assert!(avail.margin < 0.0); // actual - target = 98.0 - 99.9 < 0
    }

    #[test]
    fn test_error_rate_breach() {
        let monitor = SloMonitor::default(); // 0.001 max error rate
        let metrics = SloMetrics {
            latency_p50_ms: 5.0,
            latency_p95_ms: 30.0,
            latency_p99_ms: 100.0,
            availability_percent: 99.99,
            error_rate: 0.05, // Way above 0.001 max
            total_requests: 10000,
            total_errors: 500,
        };

        let status = monitor.evaluate(metrics);
        assert!(!status.all_met);
        let err = status.compliance.iter().find(|c| c.name == "error_rate").unwrap();
        assert!(!err.met);
        assert_eq!(status.error_budget_remaining, 0.0); // Budget exhausted
    }

    #[test]
    fn test_p50_breach() {
        let monitor = SloMonitor::default(); // 10ms p50 target
        let metrics = SloMetrics {
            latency_p50_ms: 15.0, // Exceeds 10ms
            latency_p95_ms: 30.0,
            latency_p99_ms: 100.0,
            availability_percent: 99.99,
            error_rate: 0.0001,
            total_requests: 10000,
            total_errors: 1,
        };

        let status = monitor.evaluate(metrics);
        assert!(!status.all_met);
        let p50 = status.compliance.iter().find(|c| c.name == "latency_p50").unwrap();
        assert!(!p50.met);
    }

    #[test]
    fn test_p95_breach() {
        let monitor = SloMonitor::default(); // 50ms p95 target
        let metrics = SloMetrics {
            latency_p50_ms: 5.0,
            latency_p95_ms: 75.0, // Exceeds 50ms
            latency_p99_ms: 100.0,
            availability_percent: 99.99,
            error_rate: 0.0001,
            total_requests: 10000,
            total_errors: 1,
        };

        let status = monitor.evaluate(metrics);
        assert!(!status.all_met);
        let p95 = status.compliance.iter().find(|c| c.name == "latency_p95").unwrap();
        assert!(!p95.met);
    }

    #[test]
    fn test_all_slos_breach() {
        let monitor = SloMonitor::default();
        let metrics = SloMetrics {
            latency_p50_ms: 100.0,   // >> 10ms
            latency_p95_ms: 500.0,   // >> 50ms
            latency_p99_ms: 1000.0,  // >> 200ms
            availability_percent: 90.0, // << 99.9%
            error_rate: 0.1,         // >> 0.001
            total_requests: 1000,
            total_errors: 100,
        };

        let status = monitor.evaluate(metrics);
        assert!(!status.all_met);
        assert!(status.compliance.iter().all(|c| !c.met));
    }

    #[test]
    fn test_exact_threshold_values() {
        let monitor = SloMonitor::default();
        let metrics = SloMetrics {
            latency_p50_ms: 10.0,    // Exactly at target
            latency_p95_ms: 50.0,    // Exactly at target
            latency_p99_ms: 200.0,   // Exactly at target
            availability_percent: 99.9, // Exactly at target
            error_rate: 0.001,       // Exactly at target
            total_requests: 10000,
            total_errors: 10,
        };

        let status = monitor.evaluate(metrics);
        assert!(status.all_met); // <= and >= should pass at exact boundary
    }

    #[test]
    fn test_zero_error_rate_max() {
        let monitor = SloMonitor::new(ServiceLevelObjectives {
            error_rate_max: 0.0,
            ..Default::default()
        });

        // Zero errors => budget should be 100%
        let metrics = SloMetrics {
            latency_p50_ms: 5.0, latency_p95_ms: 30.0, latency_p99_ms: 100.0,
            availability_percent: 99.99, error_rate: 0.0,
            total_requests: 1000, total_errors: 0,
        };
        let status = monitor.evaluate(metrics);
        assert_eq!(status.error_budget_remaining, 100.0);

        // Any errors => budget should be 0%
        let metrics2 = SloMetrics {
            latency_p50_ms: 5.0, latency_p95_ms: 30.0, latency_p99_ms: 100.0,
            availability_percent: 99.99, error_rate: 0.001,
            total_requests: 1000, total_errors: 1,
        };
        let status2 = monitor.evaluate(metrics2);
        assert_eq!(status2.error_budget_remaining, 0.0);
    }

    #[test]
    fn test_error_budget_fully_consumed() {
        let monitor = SloMonitor::new(ServiceLevelObjectives {
            error_rate_max: 0.01,
            ..Default::default()
        });
        let metrics = SloMetrics {
            latency_p50_ms: 5.0, latency_p95_ms: 30.0, latency_p99_ms: 100.0,
            availability_percent: 99.99,
            error_rate: 0.02, // Double the max
            total_requests: 1000, total_errors: 20,
        };

        let status = monitor.evaluate(metrics);
        assert_eq!(status.error_budget_remaining, 0.0); // Clamped to 0
    }

    #[test]
    fn test_error_budget_100_percent() {
        let monitor = SloMonitor::new(ServiceLevelObjectives {
            error_rate_max: 0.01,
            ..Default::default()
        });
        let metrics = SloMetrics {
            latency_p50_ms: 5.0, latency_p95_ms: 30.0, latency_p99_ms: 100.0,
            availability_percent: 99.99,
            error_rate: 0.0, // No errors at all
            total_requests: 1000, total_errors: 0,
        };

        let status = monitor.evaluate(metrics);
        assert_eq!(status.error_budget_remaining, 100.0);
    }

    #[test]
    fn test_compliance_has_five_items() {
        let monitor = SloMonitor::default();
        let metrics = SloMetrics {
            latency_p50_ms: 5.0, latency_p95_ms: 30.0, latency_p99_ms: 100.0,
            availability_percent: 99.99, error_rate: 0.0001,
            total_requests: 1000, total_errors: 0,
        };
        let status = monitor.evaluate(metrics);
        assert_eq!(status.compliance.len(), 5);

        let names: Vec<&str> = status.compliance.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"latency_p50"));
        assert!(names.contains(&"latency_p95"));
        assert!(names.contains(&"latency_p99"));
        assert!(names.contains(&"availability"));
        assert!(names.contains(&"error_rate"));
    }

    #[test]
    fn test_default_slo_values() {
        let slos = ServiceLevelObjectives::default();
        assert_eq!(slos.latency_p50_ms, 10.0);
        assert_eq!(slos.latency_p95_ms, 50.0);
        assert_eq!(slos.latency_p99_ms, 200.0);
        assert_eq!(slos.availability_percent, 99.9);
        assert_eq!(slos.error_rate_max, 0.001);
    }

    #[test]
    fn test_margin_calculation() {
        let monitor = SloMonitor::default();
        let metrics = SloMetrics {
            latency_p50_ms: 5.0,     // margin = 10.0 - 5.0 = 5.0
            latency_p95_ms: 60.0,    // margin = 50.0 - 60.0 = -10.0
            latency_p99_ms: 100.0,
            availability_percent: 99.95, // margin = 99.95 - 99.9 = 0.05
            error_rate: 0.0001,
            total_requests: 1000,
            total_errors: 0,
        };

        let status = monitor.evaluate(metrics);
        let p50 = status.compliance.iter().find(|c| c.name == "latency_p50").unwrap();
        assert!((p50.margin - 5.0).abs() < 0.01);

        let p95 = status.compliance.iter().find(|c| c.name == "latency_p95").unwrap();
        assert!((p95.margin - (-10.0)).abs() < 0.01);

        let avail = status.compliance.iter().find(|c| c.name == "availability").unwrap();
        assert!((avail.margin - 0.05).abs() < 0.01);
    }
}
