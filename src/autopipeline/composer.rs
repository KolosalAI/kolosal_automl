//! Pipeline composition

use crate::error::Result;
use crate::autopipeline::detector::DetectedSchema;
use crate::training::TaskType;
use serde::{Deserialize, Serialize};

/// Preprocessing step types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PreprocessingStep {
    /// Impute missing values
    Impute {
        strategy: ImputeStrategy,
        columns: Vec<usize>,
    },
    /// Scale numeric features
    Scale {
        method: ScaleMethod,
        columns: Vec<usize>,
    },
    /// Encode categorical features
    Encode {
        method: EncodeMethod,
        columns: Vec<usize>,
    },
    /// Handle outliers
    HandleOutliers {
        method: OutlierMethod,
        columns: Vec<usize>,
    },
    /// Transform skewed features
    Transform {
        method: TransformMethod,
        columns: Vec<usize>,
    },
    /// Select features
    SelectFeatures {
        method: FeatureSelectionMethod,
        n_features: Option<usize>,
    },
    /// Drop columns
    DropColumns {
        columns: Vec<usize>,
    },
}

/// Imputation strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ImputeStrategy {
    Mean,
    Median,
    Mode,
    Constant(f64),
    KNN,
}

/// Scaling methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ScaleMethod {
    Standard,
    MinMax,
    Robust,
    MaxAbs,
}

/// Encoding methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EncodeMethod {
    OneHot,
    Label,
    Target,
    Frequency,
    Hash,
}

/// Outlier handling methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OutlierMethod {
    Clip,
    Remove,
    Replace,
    IsolationForest,
}

/// Transform methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TransformMethod {
    Log,
    Sqrt,
    BoxCox,
    YeoJohnson,
}

/// Feature selection methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FeatureSelectionMethod {
    VarianceThreshold,
    MutualInformation,
    Correlation,
    RFE,
}

/// Model step in pipeline
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStep {
    /// Decision tree
    DecisionTree,
    /// Random forest
    RandomForest,
    /// Gradient boosting
    GradientBoosting,
    /// Linear model (regression or logistic)
    LinearModel,
    /// K-nearest neighbors
    KNN,
    /// Naive Bayes
    NaiveBayes,
    /// Support vector machine
    SVM,
}

/// Configuration for pipeline composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposerConfig {
    /// Task type
    pub task_type: TaskType,
    /// Enable feature selection
    pub enable_feature_selection: bool,
    /// Enable outlier handling
    pub enable_outlier_handling: bool,
    /// Enable transformations for skewed features
    pub enable_skew_transform: bool,
    /// Maximum preprocessing steps
    pub max_preprocessing_steps: usize,
    /// Preferred models
    pub preferred_models: Option<Vec<ModelStep>>,
    /// Skewness threshold for transformation
    pub skewness_threshold: f64,
    /// Missing threshold for column drop
    pub missing_threshold: f64,
}

impl Default for ComposerConfig {
    fn default() -> Self {
        Self {
            task_type: TaskType::BinaryClassification,
            enable_feature_selection: true,
            enable_outlier_handling: true,
            enable_skew_transform: true,
            max_preprocessing_steps: 10,
            preferred_models: None,
            skewness_threshold: 2.0,
            missing_threshold: 0.5,
        }
    }
}

/// Composed pipeline blueprint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineBlueprint {
    /// Preprocessing steps in order
    pub preprocessing_steps: Vec<PreprocessingStep>,
    /// Recommended models to try
    pub recommended_models: Vec<ModelStep>,
    /// Estimated complexity score
    pub complexity_score: f64,
    /// Warnings or notes
    pub notes: Vec<String>,
}

/// Pipeline composer that creates optimal pipelines from data
pub struct PipelineComposer {
    config: ComposerConfig,
}

impl PipelineComposer {
    /// Create new composer with config
    pub fn new(config: ComposerConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(ComposerConfig::default())
    }

    /// Compose pipeline from detected schema
    pub fn compose(&self, schema: &DetectedSchema) -> Result<PipelineBlueprint> {
        let mut steps = Vec::new();
        let mut notes = Vec::new();

        // Step 1: Drop unusable columns
        if !schema.drop_columns.is_empty() {
            steps.push(PreprocessingStep::DropColumns {
                columns: schema.drop_columns.clone(),
            });
            notes.push(format!(
                "Dropping {} columns (constant or ID-like)",
                schema.drop_columns.len()
            ));
        }

        // Step 2: Handle missing values
        let columns_with_missing: Vec<usize> = schema
            .columns
            .iter()
            .enumerate()
            .filter(|(i, col)| col.n_missing > 0 && !schema.drop_columns.contains(i))
            .map(|(i, _)| i)
            .collect();

        if !columns_with_missing.is_empty() {
            let high_missing: Vec<usize> = columns_with_missing
                .iter()
                .filter(|&&i| schema.columns[i].pct_missing > self.config.missing_threshold)
                .copied()
                .collect();

            if !high_missing.is_empty() {
                steps.push(PreprocessingStep::DropColumns {
                    columns: high_missing.clone(),
                });
                notes.push(format!(
                    "Dropping {} columns with >{:.0}% missing values",
                    high_missing.len(),
                    self.config.missing_threshold * 100.0
                ));
            }

            let remaining_missing: Vec<usize> = columns_with_missing
                .iter()
                .filter(|i| !high_missing.contains(i))
                .copied()
                .collect();

            if !remaining_missing.is_empty() {
                // Use median for numeric (robust to outliers)
                let numeric_missing: Vec<usize> = remaining_missing
                    .iter()
                    .filter(|&&i| schema.numeric_columns.contains(&i))
                    .copied()
                    .collect();

                if !numeric_missing.is_empty() {
                    steps.push(PreprocessingStep::Impute {
                        strategy: ImputeStrategy::Median,
                        columns: numeric_missing,
                    });
                }

                // Use mode for categorical
                let categorical_missing: Vec<usize> = remaining_missing
                    .iter()
                    .filter(|&&i| schema.categorical_columns.contains(&i))
                    .copied()
                    .collect();

                if !categorical_missing.is_empty() {
                    steps.push(PreprocessingStep::Impute {
                        strategy: ImputeStrategy::Mode,
                        columns: categorical_missing,
                    });
                }
            }
        }

        // Step 3: Encode categorical features
        if !schema.categorical_columns.is_empty() {
            let active_categorical: Vec<usize> = schema
                .categorical_columns
                .iter()
                .filter(|i| !schema.drop_columns.contains(i))
                .copied()
                .collect();

            if !active_categorical.is_empty() {
                // Split by cardinality
                let low_cardinality: Vec<usize> = active_categorical
                    .iter()
                    .filter(|&&i| schema.columns[i].n_unique <= 10)
                    .copied()
                    .collect();

                let high_cardinality: Vec<usize> = active_categorical
                    .iter()
                    .filter(|&&i| schema.columns[i].n_unique > 10)
                    .copied()
                    .collect();

                if !low_cardinality.is_empty() {
                    steps.push(PreprocessingStep::Encode {
                        method: EncodeMethod::OneHot,
                        columns: low_cardinality,
                    });
                }

                if !high_cardinality.is_empty() {
                    let method = if self.config.task_type == TaskType::Regression {
                        EncodeMethod::Target
                    } else {
                        EncodeMethod::Frequency
                    };
                    steps.push(PreprocessingStep::Encode {
                        method,
                        columns: high_cardinality,
                    });
                    notes.push("Using target/frequency encoding for high-cardinality features".to_string());
                }
            }
        }

        // Step 4: Handle outliers
        if self.config.enable_outlier_handling && !schema.numeric_columns.is_empty() {
            let active_numeric: Vec<usize> = schema
                .numeric_columns
                .iter()
                .filter(|i| !schema.drop_columns.contains(i))
                .copied()
                .collect();

            if !active_numeric.is_empty() {
                steps.push(PreprocessingStep::HandleOutliers {
                    method: OutlierMethod::Clip,
                    columns: active_numeric.clone(),
                });
            }
        }

        // Step 5: Transform skewed features
        if self.config.enable_skew_transform {
            let skewed: Vec<usize> = schema.skewed_columns(self.config.skewness_threshold);
            let active_skewed: Vec<usize> = skewed
                .iter()
                .filter(|i| !schema.drop_columns.contains(i))
                .copied()
                .collect();

            if !active_skewed.is_empty() {
                steps.push(PreprocessingStep::Transform {
                    method: TransformMethod::YeoJohnson,
                    columns: active_skewed,
                });
                notes.push("Applying Yeo-Johnson transform to skewed features".to_string());
            }
        }

        // Step 6: Scale features
        if !schema.numeric_columns.is_empty() {
            let active_numeric: Vec<usize> = schema
                .numeric_columns
                .iter()
                .filter(|i| !schema.drop_columns.contains(i))
                .copied()
                .collect();

            if !active_numeric.is_empty() {
                steps.push(PreprocessingStep::Scale {
                    method: ScaleMethod::Standard,
                    columns: active_numeric,
                });
            }
        }

        // Step 7: Feature selection
        if self.config.enable_feature_selection && schema.n_cols > 20 {
            steps.push(PreprocessingStep::SelectFeatures {
                method: FeatureSelectionMethod::MutualInformation,
                n_features: Some(20),
            });
            notes.push("Applying feature selection (>20 features)".to_string());
        }

        // Determine recommended models
        let recommended_models = self.recommend_models(schema);

        // Compute complexity score
        let complexity_score = self.compute_complexity(schema, steps.len());

        Ok(PipelineBlueprint {
            preprocessing_steps: steps,
            recommended_models,
            complexity_score,
            notes,
        })
    }

    fn recommend_models(&self, schema: &DetectedSchema) -> Vec<ModelStep> {
        if let Some(ref preferred) = self.config.preferred_models {
            return preferred.clone();
        }

        let mut models = Vec::new();
        let n_samples = schema.n_rows;
        let n_features = schema.n_cols - schema.drop_columns.len();

        match self.config.task_type {
            TaskType::BinaryClassification => {
                // Always include gradient boosting (strong baseline)
                models.push(ModelStep::GradientBoosting);
                
                // Random forest for stability
                models.push(ModelStep::RandomForest);

                // Logistic regression for interpretability
                if n_features < 100 {
                    models.push(ModelStep::LinearModel);
                }

                // KNN for smaller datasets
                if n_samples < 10000 {
                    models.push(ModelStep::KNN);
                }
            }
            TaskType::Regression => {
                models.push(ModelStep::GradientBoosting);
                models.push(ModelStep::RandomForest);
                models.push(ModelStep::LinearModel);
            }
            _ => {
                models.push(ModelStep::GradientBoosting);
                models.push(ModelStep::RandomForest);
            }
        }

        models
    }

    fn compute_complexity(&self, schema: &DetectedSchema, n_steps: usize) -> f64 {
        let mut score = 0.0;

        // Base complexity from data size
        score += (schema.n_rows as f64).log10() * 0.1;
        score += (schema.n_cols as f64).log10() * 0.2;

        // Complexity from preprocessing steps
        score += n_steps as f64 * 0.1;

        // Complexity from missing values
        let avg_missing: f64 = schema.columns.iter().map(|c| c.pct_missing).sum::<f64>()
            / schema.columns.len() as f64;
        score += avg_missing * 0.3;

        // Complexity from categorical features
        let n_categorical = schema.categorical_columns.len();
        score += (n_categorical as f64 / schema.n_cols.max(1) as f64) * 0.2;

        score.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autopipeline::detector::{DataTypeDetector, ColumnStats};
    use crate::preprocessing::ColumnType;

    fn create_test_schema() -> DetectedSchema {
        DetectedSchema {
            columns: vec![
                crate::autopipeline::detector::ColumnInfo {
                    name: "num1".to_string(),
                    dtype: ColumnType::Numeric,
                    n_unique: 100,
                    n_missing: 5,
                    pct_missing: 0.05,
                    is_constant: false,
                    is_id_like: false,
                    recommended_encoding: None,
                    stats: ColumnStats::default(),
                },
                crate::autopipeline::detector::ColumnInfo {
                    name: "cat1".to_string(),
                    dtype: ColumnType::Categorical,
                    n_unique: 5,
                    n_missing: 0,
                    pct_missing: 0.0,
                    is_constant: false,
                    is_id_like: false,
                    recommended_encoding: Some("onehot".to_string()),
                    stats: ColumnStats::default(),
                },
            ],
            n_rows: 100,
            n_cols: 2,
            numeric_columns: vec![0],
            categorical_columns: vec![1],
            datetime_columns: vec![],
            text_columns: vec![],
            drop_columns: vec![],
            quality_score: 0.9,
        }
    }

    #[test]
    fn test_compose_basic() {
        let schema = create_test_schema();
        let composer = PipelineComposer::default();
        let blueprint = composer.compose(&schema).unwrap();

        assert!(!blueprint.preprocessing_steps.is_empty());
        assert!(!blueprint.recommended_models.is_empty());
    }

    #[test]
    fn test_imputation_step() {
        let mut schema = create_test_schema();
        schema.columns[0].n_missing = 10;
        schema.columns[0].pct_missing = 0.1;

        let composer = PipelineComposer::default();
        let blueprint = composer.compose(&schema).unwrap();

        let has_impute = blueprint.preprocessing_steps.iter().any(|s| {
            matches!(s, PreprocessingStep::Impute { .. })
        });
        assert!(has_impute);
    }

    #[test]
    fn test_categorical_encoding() {
        let schema = create_test_schema();
        let composer = PipelineComposer::default();
        let blueprint = composer.compose(&schema).unwrap();

        let has_encode = blueprint.preprocessing_steps.iter().any(|s| {
            matches!(s, PreprocessingStep::Encode { .. })
        });
        assert!(has_encode);
    }
}
