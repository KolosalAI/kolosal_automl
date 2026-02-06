use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dataset size classification based on row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetSize {
    Tiny,      // < 1K rows
    Small,     // < 10K rows
    Medium,    // < 100K rows
    Large,     // < 1M rows
    VeryLarge, // >= 1M rows
}

impl DatasetSize {
    pub fn from_row_count(n_rows: usize) -> Self {
        match n_rows {
            0..=999 => DatasetSize::Tiny,
            1_000..=9_999 => DatasetSize::Small,
            10_000..=99_999 => DatasetSize::Medium,
            100_000..=999_999 => DatasetSize::Large,
            _ => DatasetSize::VeryLarge,
        }
    }
}

/// Processing mode that determines the optimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingMode {
    Speed,
    Memory,
    Quality,
    Balanced,
    LargeScale,
}

/// Characteristics of a dataset used to drive adaptive preprocessing decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetCharacteristics {
    pub n_rows: usize,
    pub n_cols: usize,
    pub n_numeric: usize,
    pub n_categorical: usize,
    pub n_datetime: usize,
    pub n_text: usize,
    pub sparsity: f64,
    pub skewness_avg: f64,
    pub missing_ratio: f64,
    pub outlier_ratio: f64,
    pub high_cardinality_count: usize,
    pub memory_mb: f64,
    pub dataset_size: DatasetSize,
}

/// Adaptive preprocessor configuration that selects strategies based on dataset characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePreprocessorConfig;

impl AdaptivePreprocessorConfig {
    /// Determine the optimal processing mode for the given dataset characteristics.
    pub fn determine_strategy(characteristics: &DatasetCharacteristics) -> ProcessingMode {
        let size = characteristics.dataset_size;
        let memory_mb = characteristics.memory_mb;

        if matches!(size, DatasetSize::VeryLarge) || memory_mb > 4096.0 {
            return ProcessingMode::LargeScale;
        }
        if memory_mb > 2048.0 {
            return ProcessingMode::Memory;
        }
        if matches!(size, DatasetSize::Tiny | DatasetSize::Small) {
            return ProcessingMode::Quality;
        }
        if characteristics.missing_ratio > 0.3 || characteristics.outlier_ratio > 0.1 {
            return ProcessingMode::Quality;
        }
        if matches!(size, DatasetSize::Large) && characteristics.n_cols > 100 {
            return ProcessingMode::Speed;
        }
        ProcessingMode::Balanced
    }

    /// Select the best normalization method based on data characteristics.
    pub fn select_normalization(characteristics: &DatasetCharacteristics) -> String {
        if characteristics.outlier_ratio > 0.05 || characteristics.skewness_avg.abs() > 2.0 {
            "robust".to_string()
        } else if characteristics.sparsity > 0.5 {
            "minmax".to_string()
        } else {
            "standard".to_string()
        }
    }

    /// Suggest an appropriate chunk size for batch processing.
    pub fn suggest_chunk_size(characteristics: &DatasetCharacteristics) -> usize {
        match characteristics.dataset_size {
            DatasetSize::Tiny => characteristics.n_rows,
            DatasetSize::Small => characteristics.n_rows,
            DatasetSize::Medium => 10_000,
            DatasetSize::Large => 50_000,
            DatasetSize::VeryLarge => 100_000,
        }
    }

    /// Suggest the number of parallel workers based on dataset size and available resources.
    pub fn suggest_num_workers(characteristics: &DatasetCharacteristics) -> usize {
        let available = num_cpus_estimate();
        match characteristics.dataset_size {
            DatasetSize::Tiny | DatasetSize::Small => 1,
            DatasetSize::Medium => (available / 2).max(1),
            DatasetSize::Large => available.max(1),
            DatasetSize::VeryLarge => available.max(1),
        }
    }
}

fn num_cpus_estimate() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Optimizer that analyzes datasets and produces optimized configuration maps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigOptimizer;

impl ConfigOptimizer {
    pub fn new() -> Self {
        ConfigOptimizer
    }

    /// Analyze raw numeric data to compute dataset characteristics.
    pub fn analyze_dataset(
        data: &[Vec<f64>],
        n_rows: usize,
        n_cols: usize,
    ) -> DatasetCharacteristics {
        let total_cells = n_rows * n_cols;

        // Count missing (NaN) values
        let mut nan_count: usize = 0;
        let mut zero_count: usize = 0;
        let mut col_means = vec![0.0f64; n_cols];
        let mut col_counts = vec![0usize; n_cols];

        for row in data.iter().take(n_rows) {
            for (j, val) in row.iter().enumerate().take(n_cols) {
                if val.is_nan() {
                    nan_count += 1;
                } else {
                    col_means[j] += val;
                    col_counts[j] += 1;
                    if *val == 0.0 {
                        zero_count += 1;
                    }
                }
            }
        }

        for j in 0..n_cols {
            if col_counts[j] > 0 {
                col_means[j] /= col_counts[j] as f64;
            }
        }

        // Compute average skewness across columns
        let mut skewness_sum = 0.0f64;
        let mut skewness_cols = 0usize;
        let mut outlier_count: usize = 0;

        for j in 0..n_cols {
            if col_counts[j] < 3 {
                continue;
            }
            let mean = col_means[j];
            let n = col_counts[j] as f64;
            let mut m2 = 0.0f64;
            let mut m3 = 0.0f64;

            for row in data.iter().take(n_rows) {
                if j < row.len() && !row[j].is_nan() {
                    let d = row[j] - mean;
                    m2 += d * d;
                    m3 += d * d * d;
                }
            }

            let variance = m2 / n;
            let std_dev = variance.sqrt();
            if std_dev > 1e-10 {
                let skew = (m3 / n) / (std_dev * std_dev * std_dev);
                skewness_sum += skew;
                skewness_cols += 1;

                // Count outliers (beyond 3 std deviations)
                for row in data.iter().take(n_rows) {
                    if j < row.len() && !row[j].is_nan() {
                        if (row[j] - mean).abs() > 3.0 * std_dev {
                            outlier_count += 1;
                        }
                    }
                }
            }
        }

        let skewness_avg = if skewness_cols > 0 {
            skewness_sum / skewness_cols as f64
        } else {
            0.0
        };

        let valid_cells = total_cells.saturating_sub(nan_count);
        let missing_ratio = if total_cells > 0 {
            nan_count as f64 / total_cells as f64
        } else {
            0.0
        };
        let sparsity = if total_cells > 0 {
            zero_count as f64 / total_cells as f64
        } else {
            0.0
        };
        let outlier_ratio = if valid_cells > 0 {
            outlier_count as f64 / valid_cells as f64
        } else {
            0.0
        };

        let memory_mb = (total_cells * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        DatasetCharacteristics {
            n_rows,
            n_cols,
            n_numeric: n_cols,
            n_categorical: 0,
            n_datetime: 0,
            n_text: 0,
            sparsity,
            skewness_avg,
            missing_ratio,
            outlier_ratio,
            high_cardinality_count: 0,
            memory_mb,
            dataset_size: DatasetSize::from_row_count(n_rows),
        }
    }

    /// Produce an optimized configuration map for the given dataset characteristics.
    pub fn optimize_for_dataset(
        characteristics: &DatasetCharacteristics,
    ) -> HashMap<String, serde_json::Value> {
        let mode = AdaptivePreprocessorConfig::determine_strategy(characteristics);
        let normalization = AdaptivePreprocessorConfig::select_normalization(characteristics);
        let chunk_size = AdaptivePreprocessorConfig::suggest_chunk_size(characteristics);
        let num_workers = AdaptivePreprocessorConfig::suggest_num_workers(characteristics);

        let mut config = HashMap::new();
        config.insert(
            "processing_mode".into(),
            serde_json::to_value(format!("{:?}", mode)).unwrap(),
        );
        config.insert(
            "normalization".into(),
            serde_json::Value::String(normalization),
        );
        config.insert(
            "chunk_size".into(),
            serde_json::to_value(chunk_size).unwrap(),
        );
        config.insert(
            "num_workers".into(),
            serde_json::to_value(num_workers).unwrap(),
        );
        config.insert(
            "use_sparse".into(),
            serde_json::Value::Bool(characteristics.sparsity > 0.5),
        );
        config.insert(
            "imputation_strategy".into(),
            serde_json::Value::String(
                if characteristics.missing_ratio > 0.2 {
                    "iterative"
                } else if characteristics.missing_ratio > 0.05 {
                    "knn"
                } else {
                    "median"
                }
                .into(),
            ),
        );
        config
    }

    /// Estimate processing time in seconds for each pipeline stage.
    pub fn estimate_processing_time(
        characteristics: &DatasetCharacteristics,
    ) -> HashMap<String, f64> {
        let base = (characteristics.n_rows as f64 * characteristics.n_cols as f64) / 1_000_000.0;
        let mut estimates = HashMap::new();

        estimates.insert("loading".into(), base * 0.5);
        estimates.insert("cleaning".into(), base * (1.0 + characteristics.missing_ratio * 2.0));
        estimates.insert(
            "normalization".into(),
            base * if characteristics.outlier_ratio > 0.05 { 1.5 } else { 0.8 },
        );
        estimates.insert(
            "encoding".into(),
            base * 0.3 * (1.0 + characteristics.high_cardinality_count as f64 * 0.5),
        );
        estimates.insert(
            "total".into(),
            estimates.values().sum::<f64>(),
        );
        estimates
    }

    /// Provide memory optimization suggestions based on dataset characteristics.
    pub fn get_memory_optimization_suggestions(
        characteristics: &DatasetCharacteristics,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        if characteristics.memory_mb > 1024.0 {
            suggestions.push("Use chunked processing to limit peak memory usage".into());
        }
        if characteristics.sparsity > 0.5 {
            suggestions.push("Convert to sparse matrix representation to save memory".into());
        }
        if characteristics.high_cardinality_count > 0 {
            suggestions.push(
                "Apply target encoding or hashing for high-cardinality categorical features".into(),
            );
        }
        if matches!(characteristics.dataset_size, DatasetSize::VeryLarge) {
            suggestions.push("Consider memory-mapped file I/O for very large datasets".into());
            suggestions.push("Use streaming preprocessing to avoid loading full dataset".into());
        }
        if characteristics.n_text > 0 {
            suggestions.push("Process text features in batches to control memory".into());
        }
        if characteristics.missing_ratio > 0.3 {
            suggestions.push("Drop columns with >50% missing before imputation to reduce work".into());
        }

        suggestions
    }
}

impl Default for ConfigOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_characteristics() -> DatasetCharacteristics {
        DatasetCharacteristics {
            n_rows: 50_000,
            n_cols: 20,
            n_numeric: 15,
            n_categorical: 3,
            n_datetime: 1,
            n_text: 1,
            sparsity: 0.1,
            skewness_avg: 0.5,
            missing_ratio: 0.02,
            outlier_ratio: 0.01,
            high_cardinality_count: 1,
            memory_mb: 8.0,
            dataset_size: DatasetSize::Medium,
        }
    }

    #[test]
    fn test_dataset_size_classification() {
        assert_eq!(DatasetSize::from_row_count(500), DatasetSize::Tiny);
        assert_eq!(DatasetSize::from_row_count(5_000), DatasetSize::Small);
        assert_eq!(DatasetSize::from_row_count(50_000), DatasetSize::Medium);
        assert_eq!(DatasetSize::from_row_count(500_000), DatasetSize::Large);
        assert_eq!(DatasetSize::from_row_count(2_000_000), DatasetSize::VeryLarge);
    }

    #[test]
    fn test_determine_strategy_balanced() {
        let chars = sample_characteristics();
        let mode = AdaptivePreprocessorConfig::determine_strategy(&chars);
        assert_eq!(mode, ProcessingMode::Balanced);
    }

    #[test]
    fn test_determine_strategy_large_scale() {
        let mut chars = sample_characteristics();
        chars.dataset_size = DatasetSize::VeryLarge;
        chars.n_rows = 2_000_000;
        assert_eq!(
            AdaptivePreprocessorConfig::determine_strategy(&chars),
            ProcessingMode::LargeScale
        );
    }

    #[test]
    fn test_select_normalization_robust() {
        let mut chars = sample_characteristics();
        chars.outlier_ratio = 0.1;
        assert_eq!(AdaptivePreprocessorConfig::select_normalization(&chars), "robust");
    }

    #[test]
    fn test_select_normalization_standard() {
        let chars = sample_characteristics();
        assert_eq!(AdaptivePreprocessorConfig::select_normalization(&chars), "standard");
    }

    #[test]
    fn test_chunk_size() {
        let chars = sample_characteristics();
        assert_eq!(AdaptivePreprocessorConfig::suggest_chunk_size(&chars), 10_000);
    }

    #[test]
    fn test_analyze_dataset() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, f64::NAN, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let chars = ConfigOptimizer::analyze_dataset(&data, 3, 3);
        assert_eq!(chars.n_rows, 3);
        assert_eq!(chars.n_cols, 3);
        assert!(chars.missing_ratio > 0.0);
        assert_eq!(chars.dataset_size, DatasetSize::Tiny);
    }

    #[test]
    fn test_optimize_for_dataset() {
        let chars = sample_characteristics();
        let config = ConfigOptimizer::optimize_for_dataset(&chars);
        assert!(config.contains_key("processing_mode"));
        assert!(config.contains_key("normalization"));
        assert!(config.contains_key("chunk_size"));
        assert!(config.contains_key("num_workers"));
    }

    #[test]
    fn test_estimate_processing_time() {
        let chars = sample_characteristics();
        let times = ConfigOptimizer::estimate_processing_time(&chars);
        assert!(times.contains_key("total"));
        assert!(*times.get("total").unwrap() > 0.0);
    }

    #[test]
    fn test_memory_suggestions_large() {
        let mut chars = sample_characteristics();
        chars.memory_mb = 2048.0;
        chars.sparsity = 0.7;
        let suggestions = ConfigOptimizer::get_memory_optimization_suggestions(&chars);
        assert!(suggestions.len() >= 2);
    }
}
