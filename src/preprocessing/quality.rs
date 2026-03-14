//! Data Quality Scoring for ISO 5259 and ISO 25012 compliance.
//!
//! Provides systematic data quality assessment including completeness,
//! uniqueness, consistency, and validity metrics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete data quality report for a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityReport {
    /// Overall quality score (0.0 - 1.0)
    pub overall_score: f64,
    /// Completeness: ratio of non-missing values
    pub completeness: f64,
    /// Uniqueness: ratio of unique rows
    pub uniqueness: f64,
    /// Consistency: ratio of columns matching expected types
    pub consistency: f64,
    /// Validity: ratio of values within expected ranges
    pub validity: f64,
    /// Per-column quality metrics
    pub per_column: Vec<ColumnQuality>,
    /// Detected quality warnings
    pub warnings: Vec<QualityWarning>,
    /// Number of rows
    pub num_rows: usize,
    /// Number of columns
    pub num_columns: usize,
}

/// Quality metrics for a single column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnQuality {
    /// Column name
    pub column: String,
    /// Completeness (1 - missing_ratio)
    pub completeness: f64,
    /// Ratio of distinct values to total
    pub distinct_ratio: f64,
    /// Ratio of outlier values (if numeric)
    pub outlier_ratio: f64,
    /// Type consistency (ratio of values matching the inferred type)
    pub type_consistency: f64,
    /// Ratio of zero values (if numeric)
    pub zeros_ratio: f64,
    /// Coefficient of variation = std / |mean| (None if mean is zero or not numeric)
    pub cv: Option<f64>,
    /// Interquartile range (Q75 - Q25)
    pub iqr: Option<f64>,
    /// Value range (max - min)
    pub value_range: Option<f64>,
    /// Skewness of the distribution
    pub skewness: Option<f64>,
    /// Excess kurtosis of the distribution
    pub kurtosis: Option<f64>,
    /// Sum of all non-null values
    pub sum: Option<f64>,
    /// Most frequent value as a fraction of non-null rows
    pub mode_frequency: Option<f64>,
    /// Shannon entropy (categorical columns)
    pub entropy: Option<f64>,
}

/// Quality warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityWarning {
    /// Column has high proportion of missing values
    HighMissingness { column: String, ratio: f64 },
    /// Column has very low variance
    LowVariance { column: String, variance: f64 },
    /// Column has very high cardinality (potential ID column)
    HighCardinality { column: String, unique_count: usize, total_count: usize },
    /// Column appears to be an ID/index column
    SuspectedId { column: String },
    /// Target variable has class imbalance
    ClassImbalance { column: String, ratio: f64 },
    /// Duplicate rows detected
    DuplicateRows { count: usize, total: usize },
    /// Constant column (zero variance)
    ConstantColumn { column: String },
    /// Column has high absolute skewness (|skew| > 2) indicating non-normal distribution
    HighSkewness { column: String, skewness: f64 },
    /// Column has very high excess kurtosis (|kurt| > 7) indicating heavy tails or spikes
    HighKurtosis { column: String, kurtosis: f64 },
    /// Numeric column has a high proportion of zero values (>50%)
    HighZeroRatio { column: String, ratio: f64 },
    /// Column has a high proportion of outliers by IQR rule (>10%)
    HighOutlierRatio { column: String, ratio: f64 },
}

/// Data quality scorer
pub struct DataQualityScorer;

impl DataQualityScorer {
    /// Compute quality report from column statistics.
    ///
    /// # Arguments
    /// * `num_rows` - Number of rows in the dataset
    /// * `column_stats` - Map of column name to (null_count, unique_count, mean, std, min, max)
    /// * `duplicate_row_count` - Number of duplicate rows
    pub fn score(
        num_rows: usize,
        column_stats: &HashMap<String, ColumnStatistics>,
        duplicate_row_count: usize,
    ) -> DataQualityReport {
        let num_columns = column_stats.len();
        let mut per_column = Vec::new();
        let mut warnings = Vec::new();

        let mut total_completeness = 0.0;
        let mut total_consistency = 0.0;
        let mut total_validity = 0.0;

        for (col_name, stats) in column_stats {
            let completeness = if num_rows > 0 {
                1.0 - (stats.null_count as f64 / num_rows as f64)
            } else {
                1.0
            };

            let distinct_ratio = if num_rows > 0 {
                stats.unique_count as f64 / num_rows as f64
            } else {
                0.0
            };

            let outlier_ratio = stats.outlier_count
                .map(|oc| if num_rows > 0 { oc as f64 / num_rows as f64 } else { 0.0 })
                .unwrap_or(0.0);

            let type_consistency = stats.type_consistency.unwrap_or(1.0);

            let zeros_ratio = stats.zeros_count
                .map(|zc| if num_rows > 0 { zc as f64 / num_rows as f64 } else { 0.0 })
                .unwrap_or(0.0);

            let cv = match (stats.mean, stats.std) {
                (Some(m), Some(s)) if m.abs() > 1e-10 => Some(s / m.abs()),
                _ => None,
            };

            let iqr = match (stats.q25, stats.q75) {
                (Some(q1), Some(q3)) => Some(q3 - q1),
                _ => None,
            };

            let value_range = match (stats.min, stats.max) {
                (Some(mn), Some(mx)) => Some(mx - mn),
                _ => None,
            };

            // Validity: non-null, non-outlier, type-consistent
            let validity = completeness * (1.0 - outlier_ratio) * type_consistency;

            total_completeness += completeness;
            total_consistency += type_consistency;
            total_validity += validity;

            per_column.push(ColumnQuality {
                column: col_name.clone(),
                completeness,
                distinct_ratio,
                outlier_ratio,
                type_consistency,
                zeros_ratio,
                cv,
                iqr,
                value_range,
                skewness: stats.skewness,
                kurtosis: stats.kurtosis,
                sum: stats.sum,
                mode_frequency: stats.mode_frequency,
                entropy: stats.entropy,
            });

            // Warnings
            if completeness < 0.7 {
                warnings.push(QualityWarning::HighMissingness {
                    column: col_name.clone(),
                    ratio: 1.0 - completeness,
                });
            }

            if let Some(std) = stats.std {
                if std < 1e-10 && num_rows > 1 {
                    warnings.push(QualityWarning::ConstantColumn {
                        column: col_name.clone(),
                    });
                } else if std < 1e-6 {
                    warnings.push(QualityWarning::LowVariance {
                        column: col_name.clone(),
                        variance: std * std,
                    });
                }
            }

            if let Some(skew) = stats.skewness {
                if skew.abs() > 2.0 {
                    warnings.push(QualityWarning::HighSkewness {
                        column: col_name.clone(),
                        skewness: skew,
                    });
                }
            }

            if let Some(kurt) = stats.kurtosis {
                if kurt.abs() > 7.0 {
                    warnings.push(QualityWarning::HighKurtosis {
                        column: col_name.clone(),
                        kurtosis: kurt,
                    });
                }
            }

            if zeros_ratio > 0.5 && stats.mean.is_some() {
                warnings.push(QualityWarning::HighZeroRatio {
                    column: col_name.clone(),
                    ratio: zeros_ratio,
                });
            }

            if outlier_ratio > 0.1 {
                warnings.push(QualityWarning::HighOutlierRatio {
                    column: col_name.clone(),
                    ratio: outlier_ratio,
                });
            }

            if distinct_ratio > 0.95 && num_rows > 20 {
                warnings.push(QualityWarning::HighCardinality {
                    column: col_name.clone(),
                    unique_count: stats.unique_count,
                    total_count: num_rows,
                });

                // If it's also sequential, likely an ID
                if distinct_ratio > 0.99 && stats.is_sequential.unwrap_or(false) {
                    warnings.push(QualityWarning::SuspectedId {
                        column: col_name.clone(),
                    });
                }
            }
        }

        if duplicate_row_count > 0 {
            warnings.push(QualityWarning::DuplicateRows {
                count: duplicate_row_count,
                total: num_rows,
            });
        }

        let n = num_columns.max(1) as f64;
        let avg_completeness = total_completeness / n;
        let avg_consistency = total_consistency / n;
        let avg_validity = total_validity / n;

        let uniqueness = if num_rows > 0 {
            1.0 - (duplicate_row_count as f64 / num_rows as f64)
        } else {
            1.0
        };

        // Overall score: weighted average
        let overall_score = avg_completeness * 0.3
            + uniqueness * 0.2
            + avg_consistency * 0.25
            + avg_validity * 0.25;

        DataQualityReport {
            overall_score,
            completeness: avg_completeness,
            uniqueness,
            consistency: avg_consistency,
            validity: avg_validity,
            per_column,
            warnings,
            num_rows,
            num_columns,
        }
    }
}

/// Statistics for a single column used in quality scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStatistics {
    pub null_count: usize,
    pub unique_count: usize,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub q25: Option<f64>,
    pub q75: Option<f64>,
    pub median: Option<f64>,
    pub skewness: Option<f64>,
    pub kurtosis: Option<f64>,
    pub zeros_count: Option<usize>,
    pub sum: Option<f64>,
    pub mode_frequency: Option<f64>,
    pub entropy: Option<f64>,
    pub outlier_count: Option<usize>,
    pub type_consistency: Option<f64>,
    pub is_sequential: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(
        null_count: usize, unique_count: usize,
        mean: Option<f64>, std: Option<f64>,
        min: Option<f64>, max: Option<f64>,
        outlier_count: Option<usize>,
        type_consistency: Option<f64>,
        is_sequential: Option<bool>,
    ) -> ColumnStatistics {
        ColumnStatistics {
            null_count, unique_count, mean, std, min, max,
            q25: None, q75: None, median: None,
            skewness: None, kurtosis: None,
            zeros_count: None, sum: None,
            mode_frequency: None, entropy: None,
            outlier_count, type_consistency, is_sequential,
        }
    }

    #[test]
    fn test_perfect_quality() {
        let mut stats = HashMap::new();
        stats.insert("age".to_string(), make_stats(
            0, 50, Some(35.0), Some(10.0), Some(18.0), Some(65.0),
            Some(0), Some(1.0), Some(false),
        ));
        let report = DataQualityScorer::score(100, &stats, 0);
        assert!(report.overall_score > 0.9);
        assert_eq!(report.completeness, 1.0);
    }

    #[test]
    fn test_poor_quality() {
        let mut stats = HashMap::new();
        stats.insert("messy_col".to_string(), make_stats(
            80, 5, None, None, None, None, Some(10), Some(0.5), Some(false),
        ));
        let report = DataQualityScorer::score(100, &stats, 20);
        assert!(report.overall_score < 0.5);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::HighMissingness { .. })));
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::DuplicateRows { .. })));
    }

    #[test]
    fn test_suspected_id_column() {
        let mut stats = HashMap::new();
        stats.insert("row_id".to_string(), make_stats(
            0, 100, Some(50.0), Some(29.0), Some(1.0), Some(100.0),
            Some(0), Some(1.0), Some(true),
        ));
        let report = DataQualityScorer::score(100, &stats, 0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::SuspectedId { .. })));
    }

    #[test]
    fn test_constant_column_warning() {
        let mut stats = HashMap::new();
        stats.insert("constant".to_string(), make_stats(
            0, 1, Some(5.0), Some(0.0), Some(5.0), Some(5.0),
            Some(0), Some(1.0), Some(false),
        ));
        let report = DataQualityScorer::score(100, &stats, 0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::ConstantColumn { .. })));
    }

    #[test]
    fn test_empty_dataset() {
        let stats = HashMap::new();
        let report = DataQualityScorer::score(0, &stats, 0);
        assert_eq!(report.num_rows, 0);
        assert_eq!(report.num_columns, 0);
        assert_eq!(report.uniqueness, 1.0);
        assert!(report.per_column.is_empty());
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn test_zero_rows_with_columns() {
        let mut stats = HashMap::new();
        stats.insert("col".to_string(), make_stats(0, 0, None, None, None, None, None, None, None));
        let report = DataQualityScorer::score(0, &stats, 0);
        assert_eq!(report.num_columns, 1);
        assert_eq!(report.completeness, 1.0);
    }

    #[test]
    fn test_all_null_column() {
        let mut stats = HashMap::new();
        stats.insert("all_null".to_string(), make_stats(
            100, 0, None, None, None, None, Some(0), Some(1.0), Some(false),
        ));
        let report = DataQualityScorer::score(100, &stats, 0);
        assert_eq!(report.completeness, 0.0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::HighMissingness { .. })));
    }

    #[test]
    fn test_all_duplicates() {
        let mut stats = HashMap::new();
        stats.insert("val".to_string(), make_stats(
            0, 1, Some(42.0), Some(0.0), Some(42.0), Some(42.0),
            Some(0), Some(1.0), Some(false),
        ));
        let report = DataQualityScorer::score(100, &stats, 100);
        assert_eq!(report.uniqueness, 0.0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::DuplicateRows { count: 100, total: 100 })));
    }

    #[test]
    fn test_low_variance_warning() {
        let mut stats = HashMap::new();
        stats.insert("low_var".to_string(), make_stats(
            0, 3, Some(1.0), Some(0.0000001), Some(0.9999999), Some(1.0000001),
            Some(0), Some(1.0), Some(false),
        ));
        let report = DataQualityScorer::score(100, &stats, 0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::LowVariance { .. })));
    }

    #[test]
    fn test_multiple_columns_quality() {
        let mut stats = HashMap::new();
        stats.insert("good".to_string(), make_stats(
            0, 80, Some(50.0), Some(15.0), Some(10.0), Some(90.0),
            Some(0), Some(1.0), Some(false),
        ));
        stats.insert("bad".to_string(), make_stats(
            50, 5, None, None, None, None, Some(20), Some(0.3), Some(false),
        ));
        let report = DataQualityScorer::score(100, &stats, 5);
        assert!(report.overall_score > 0.0);
        assert!(report.overall_score < 1.0);
        assert_eq!(report.per_column.len(), 2);
        assert_eq!(report.num_columns, 2);
    }

    #[test]
    fn test_high_cardinality_without_sequential() {
        let mut stats = HashMap::new();
        stats.insert("high_card".to_string(), make_stats(
            0, 99, Some(50.0), Some(30.0), Some(1.0), Some(99.0),
            Some(0), Some(1.0), Some(false),
        ));
        let report = DataQualityScorer::score(100, &stats, 0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::HighCardinality { .. })));
        assert!(!report.warnings.iter().any(|w| matches!(w, QualityWarning::SuspectedId { .. })));
    }

    #[test]
    fn test_validity_accounts_for_outliers() {
        let mut stats = HashMap::new();
        stats.insert("outlier_col".to_string(), make_stats(
            0, 50, Some(50.0), Some(10.0), Some(0.0), Some(1000.0),
            Some(30), Some(1.0), Some(false),
        ));
        let report = DataQualityScorer::score(100, &stats, 0);
        let col_quality = &report.per_column[0];
        assert_eq!(col_quality.outlier_ratio, 0.3);
        assert!(report.validity < 1.0);
    }

    #[test]
    fn test_overall_score_weights() {
        let mut stats = HashMap::new();
        stats.insert("col".to_string(), make_stats(
            0, 50, Some(50.0), Some(10.0), Some(0.0), Some(100.0),
            Some(0), Some(1.0), Some(false),
        ));
        let report = DataQualityScorer::score(100, &stats, 0);
        assert!((report.overall_score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_high_skewness_warning() {
        let mut stats = HashMap::new();
        let mut s = make_stats(
            0, 50, Some(5.0), Some(2.0), Some(0.0), Some(100.0),
            Some(0), Some(1.0), Some(false),
        );
        s.skewness = Some(3.5);
        stats.insert("skewed".to_string(), s);
        let report = DataQualityScorer::score(100, &stats, 0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::HighSkewness { .. })));
    }

    #[test]
    fn test_high_zero_ratio_warning() {
        let mut stats = HashMap::new();
        let mut s = make_stats(
            0, 3, Some(0.5), Some(1.0), Some(0.0), Some(10.0),
            Some(0), Some(1.0), Some(false),
        );
        s.zeros_count = Some(70);
        stats.insert("sparse".to_string(), s);
        let report = DataQualityScorer::score(100, &stats, 0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::HighZeroRatio { .. })));
    }

    #[test]
    fn test_column_quality_cv_iqr() {
        let mut stats = HashMap::new();
        let mut s = make_stats(
            0, 50, Some(20.0), Some(4.0), Some(10.0), Some(30.0),
            Some(0), Some(1.0), Some(false),
        );
        s.q25 = Some(17.0);
        s.q75 = Some(23.0);
        stats.insert("col".to_string(), s);
        let report = DataQualityScorer::score(100, &stats, 0);
        let cq = &report.per_column[0];
        // cv = 4.0 / 20.0 = 0.2
        assert!((cq.cv.unwrap() - 0.2).abs() < 1e-9);
        // iqr = 23.0 - 17.0 = 6.0
        assert!((cq.iqr.unwrap() - 6.0).abs() < 1e-9);
        // range = 30.0 - 10.0 = 20.0
        assert!((cq.value_range.unwrap() - 20.0).abs() < 1e-9);
    }
}
