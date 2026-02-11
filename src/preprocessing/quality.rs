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
    pub outlier_count: Option<usize>,
    pub type_consistency: Option<f64>,
    pub is_sequential: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_quality() {
        let mut stats = HashMap::new();
        stats.insert("age".to_string(), ColumnStatistics {
            null_count: 0,
            unique_count: 50,
            mean: Some(35.0),
            std: Some(10.0),
            min: Some(18.0),
            max: Some(65.0),
            outlier_count: Some(0),
            type_consistency: Some(1.0),
            is_sequential: Some(false),
        });

        let report = DataQualityScorer::score(100, &stats, 0);
        assert!(report.overall_score > 0.9);
        assert_eq!(report.completeness, 1.0);
    }

    #[test]
    fn test_poor_quality() {
        let mut stats = HashMap::new();
        stats.insert("messy_col".to_string(), ColumnStatistics {
            null_count: 80,
            unique_count: 5,
            mean: None,
            std: None,
            min: None,
            max: None,
            outlier_count: Some(10),
            type_consistency: Some(0.5),
            is_sequential: Some(false),
        });

        let report = DataQualityScorer::score(100, &stats, 20);
        assert!(report.overall_score < 0.5);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::HighMissingness { .. })));
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::DuplicateRows { .. })));
    }

    #[test]
    fn test_suspected_id_column() {
        let mut stats = HashMap::new();
        stats.insert("row_id".to_string(), ColumnStatistics {
            null_count: 0,
            unique_count: 100,
            mean: Some(50.0),
            std: Some(29.0),
            min: Some(1.0),
            max: Some(100.0),
            outlier_count: Some(0),
            type_consistency: Some(1.0),
            is_sequential: Some(true),
        });

        let report = DataQualityScorer::score(100, &stats, 0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::SuspectedId { .. })));
    }

    #[test]
    fn test_constant_column_warning() {
        let mut stats = HashMap::new();
        stats.insert("constant".to_string(), ColumnStatistics {
            null_count: 0,
            unique_count: 1,
            mean: Some(5.0),
            std: Some(0.0),
            min: Some(5.0),
            max: Some(5.0),
            outlier_count: Some(0),
            type_consistency: Some(1.0),
            is_sequential: Some(false),
        });

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
        stats.insert("col".to_string(), ColumnStatistics {
            null_count: 0,
            unique_count: 0,
            mean: None,
            std: None,
            min: None,
            max: None,
            outlier_count: None,
            type_consistency: None,
            is_sequential: None,
        });

        let report = DataQualityScorer::score(0, &stats, 0);
        assert_eq!(report.num_columns, 1);
        // With 0 rows, completeness = 1.0 (no missing data in 0 rows)
        assert_eq!(report.completeness, 1.0);
    }

    #[test]
    fn test_all_null_column() {
        let mut stats = HashMap::new();
        stats.insert("all_null".to_string(), ColumnStatistics {
            null_count: 100,
            unique_count: 0,
            mean: None,
            std: None,
            min: None,
            max: None,
            outlier_count: Some(0),
            type_consistency: Some(1.0),
            is_sequential: Some(false),
        });

        let report = DataQualityScorer::score(100, &stats, 0);
        assert_eq!(report.completeness, 0.0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::HighMissingness { .. })));
    }

    #[test]
    fn test_all_duplicates() {
        let mut stats = HashMap::new();
        stats.insert("val".to_string(), ColumnStatistics {
            null_count: 0,
            unique_count: 1,
            mean: Some(42.0),
            std: Some(0.0),
            min: Some(42.0),
            max: Some(42.0),
            outlier_count: Some(0),
            type_consistency: Some(1.0),
            is_sequential: Some(false),
        });

        let report = DataQualityScorer::score(100, &stats, 100);
        assert_eq!(report.uniqueness, 0.0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::DuplicateRows { count: 100, total: 100 })));
    }

    #[test]
    fn test_low_variance_warning() {
        let mut stats = HashMap::new();
        stats.insert("low_var".to_string(), ColumnStatistics {
            null_count: 0,
            unique_count: 3,
            mean: Some(1.0),
            std: Some(0.0000001), // Very low but not zero
            min: Some(0.9999999),
            max: Some(1.0000001),
            outlier_count: Some(0),
            type_consistency: Some(1.0),
            is_sequential: Some(false),
        });

        let report = DataQualityScorer::score(100, &stats, 0);
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::LowVariance { .. })));
    }

    #[test]
    fn test_multiple_columns_quality() {
        let mut stats = HashMap::new();
        // Perfect column
        stats.insert("good".to_string(), ColumnStatistics {
            null_count: 0, unique_count: 80, mean: Some(50.0), std: Some(15.0),
            min: Some(10.0), max: Some(90.0), outlier_count: Some(0),
            type_consistency: Some(1.0), is_sequential: Some(false),
        });
        // Bad column
        stats.insert("bad".to_string(), ColumnStatistics {
            null_count: 50, unique_count: 5, mean: None, std: None,
            min: None, max: None, outlier_count: Some(20),
            type_consistency: Some(0.3), is_sequential: Some(false),
        });

        let report = DataQualityScorer::score(100, &stats, 5);
        // Overall score should be between 0 and 1
        assert!(report.overall_score > 0.0);
        assert!(report.overall_score < 1.0);
        assert_eq!(report.per_column.len(), 2);
        assert_eq!(report.num_columns, 2);
    }

    #[test]
    fn test_high_cardinality_without_sequential() {
        let mut stats = HashMap::new();
        stats.insert("high_card".to_string(), ColumnStatistics {
            null_count: 0, unique_count: 99, mean: Some(50.0), std: Some(30.0),
            min: Some(1.0), max: Some(99.0), outlier_count: Some(0),
            type_consistency: Some(1.0), is_sequential: Some(false),
        });

        let report = DataQualityScorer::score(100, &stats, 0);
        // High cardinality warning but NOT suspected ID (not sequential)
        assert!(report.warnings.iter().any(|w| matches!(w, QualityWarning::HighCardinality { .. })));
        assert!(!report.warnings.iter().any(|w| matches!(w, QualityWarning::SuspectedId { .. })));
    }

    #[test]
    fn test_validity_accounts_for_outliers() {
        let mut stats = HashMap::new();
        stats.insert("outlier_col".to_string(), ColumnStatistics {
            null_count: 0, unique_count: 50, mean: Some(50.0), std: Some(10.0),
            min: Some(0.0), max: Some(1000.0), outlier_count: Some(30),
            type_consistency: Some(1.0), is_sequential: Some(false),
        });

        let report = DataQualityScorer::score(100, &stats, 0);
        let col_quality = &report.per_column[0];
        assert_eq!(col_quality.outlier_ratio, 0.3);
        // Validity should be reduced by outlier ratio
        assert!(report.validity < 1.0);
    }

    #[test]
    fn test_overall_score_weights() {
        // Perfect data should give score near 1.0
        let mut stats = HashMap::new();
        stats.insert("col".to_string(), ColumnStatistics {
            null_count: 0, unique_count: 50, mean: Some(50.0), std: Some(10.0),
            min: Some(0.0), max: Some(100.0), outlier_count: Some(0),
            type_consistency: Some(1.0), is_sequential: Some(false),
        });

        let report = DataQualityScorer::score(100, &stats, 0);
        // completeness=1.0 * 0.3 + uniqueness=1.0 * 0.2 + consistency=1.0 * 0.25 + validity=1.0 * 0.25 = 1.0
        assert!((report.overall_score - 1.0).abs() < 0.01);
    }
}
