//! Datasheet generation for ISO 5259 (Data Quality) compliance.
//!
//! Auto-generates dataset documentation following the "Datasheets for Datasets"
//! framework to document dataset provenance, composition, and quality.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete datasheet for a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Datasheet {
    /// Dataset identification
    pub name: String,
    pub description: String,
    pub generated_at: DateTime<Utc>,

    /// Composition
    pub composition: DatasetComposition,

    /// Collection process
    pub collection: CollectionInfo,

    /// Preprocessing / cleaning
    pub preprocessing: PreprocessingInfo,

    /// Distribution information
    pub distribution: DistributionInfo,

    /// Known issues and limitations
    pub limitations: Vec<String>,
}

/// Dataset composition details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetComposition {
    pub num_samples: usize,
    pub num_features: usize,
    pub feature_descriptions: Vec<FeatureDescription>,
    pub target_variable: Option<String>,
    pub class_distribution: HashMap<String, usize>,
    pub missing_value_summary: HashMap<String, f64>,
}

/// Description of a single feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDescription {
    pub name: String,
    pub dtype: String,
    pub description: String,
    pub missing_ratio: f64,
    pub unique_count: Option<usize>,
}

/// How the data was collected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub source: String,
    pub collection_method: String,
    pub time_period: Option<String>,
}

/// Preprocessing applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingInfo {
    pub steps_applied: Vec<String>,
    pub rows_before: usize,
    pub rows_after: usize,
}

/// Distribution/sharing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionInfo {
    pub license: String,
    pub access_level: String,
}

impl Datasheet {
    /// Generate a datasheet from dataset statistics
    pub fn generate(
        name: &str,
        source: &str,
        num_samples: usize,
        num_features: usize,
        feature_names: &[String],
        feature_dtypes: &[String],
        missing_ratios: &HashMap<String, f64>,
    ) -> Self {
        let feature_descriptions: Vec<FeatureDescription> = feature_names.iter()
            .enumerate()
            .map(|(i, name)| {
                let dtype = feature_dtypes.get(i)
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                let missing_ratio = missing_ratios.get(name).copied().unwrap_or(0.0);
                FeatureDescription {
                    name: name.clone(),
                    dtype,
                    description: String::new(),
                    missing_ratio,
                    unique_count: None,
                }
            })
            .collect();

        Self {
            name: name.to_string(),
            description: format!("Auto-generated datasheet for {}", name),
            generated_at: Utc::now(),
            composition: DatasetComposition {
                num_samples,
                num_features,
                feature_descriptions,
                target_variable: None,
                class_distribution: HashMap::new(),
                missing_value_summary: missing_ratios.clone(),
            },
            collection: CollectionInfo {
                source: source.to_string(),
                collection_method: "Not specified".to_string(),
                time_period: None,
            },
            preprocessing: PreprocessingInfo {
                steps_applied: Vec::new(),
                rows_before: num_samples,
                rows_after: num_samples,
            },
            distribution: DistributionInfo {
                license: "Not specified".to_string(),
                access_level: "Internal".to_string(),
            },
            limitations: vec![
                "This datasheet was auto-generated; manual review is recommended".to_string(),
            ],
        }
    }

    /// Render the datasheet as Markdown
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# Datasheet: {}\n\n", self.name));
        md.push_str(&format!("*Generated: {}*\n\n", self.generated_at.format("%Y-%m-%d %H:%M UTC")));
        md.push_str(&format!("{}\n\n", self.description));

        md.push_str("## Composition\n\n");
        md.push_str(&format!("- **Samples:** {}\n", self.composition.num_samples));
        md.push_str(&format!("- **Features:** {}\n\n", self.composition.num_features));

        if !self.composition.feature_descriptions.is_empty() {
            md.push_str("| Feature | Type | Missing % |\n|---------|------|----------|\n");
            for f in &self.composition.feature_descriptions {
                md.push_str(&format!("| {} | {} | {:.1}% |\n", f.name, f.dtype, f.missing_ratio * 100.0));
            }
            md.push('\n');
        }

        md.push_str("## Collection\n\n");
        md.push_str(&format!("- **Source:** {}\n", self.collection.source));
        md.push_str(&format!("- **Method:** {}\n\n", self.collection.collection_method));

        if !self.limitations.is_empty() {
            md.push_str("## Limitations\n\n");
            for item in &self.limitations {
                md.push_str(&format!("- {}\n", item));
            }
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datasheet_generation() {
        let features = vec!["age".to_string(), "income".to_string()];
        let dtypes = vec!["Float64".to_string(), "Float64".to_string()];
        let mut missing = HashMap::new();
        missing.insert("income".to_string(), 0.05);

        let ds = Datasheet::generate("test_data", "file upload", 100, 2, &features, &dtypes, &missing);
        assert_eq!(ds.composition.num_samples, 100);
        assert_eq!(ds.composition.feature_descriptions.len(), 2);
    }

    #[test]
    fn test_datasheet_markdown() {
        let ds = Datasheet::generate("iris", "sample", 150, 4, &vec!["f1".to_string()], &vec!["Float64".to_string()], &HashMap::new());
        let md = ds.to_markdown();
        assert!(md.contains("# Datasheet: iris"));
        assert!(md.contains("150"));
    }

    #[test]
    fn test_datasheet_empty_features() {
        let ds = Datasheet::generate("empty", "unknown", 0, 0, &[], &[], &HashMap::new());
        assert_eq!(ds.composition.num_samples, 0);
        assert_eq!(ds.composition.num_features, 0);
        assert!(ds.composition.feature_descriptions.is_empty());

        let md = ds.to_markdown();
        // Should not have a feature table since no features
        assert!(!md.contains("| Feature |"));
    }

    #[test]
    fn test_datasheet_mismatched_feature_lengths() {
        // More feature names than dtypes
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let dtypes = vec!["Float64".to_string()]; // Only 1 dtype

        let ds = Datasheet::generate("test", "src", 100, 3, &names, &dtypes, &HashMap::new());
        assert_eq!(ds.composition.feature_descriptions.len(), 3);
        // First feature gets dtype, others get "unknown"
        assert_eq!(ds.composition.feature_descriptions[0].dtype, "Float64");
        assert_eq!(ds.composition.feature_descriptions[1].dtype, "unknown");
        assert_eq!(ds.composition.feature_descriptions[2].dtype, "unknown");
    }

    #[test]
    fn test_datasheet_missing_ratios() {
        let names = vec!["clean".to_string(), "messy".to_string()];
        let dtypes = vec!["Float64".to_string(), "Float64".to_string()];
        let mut missing = HashMap::new();
        missing.insert("messy".to_string(), 0.75);

        let ds = Datasheet::generate("test", "src", 100, 2, &names, &dtypes, &missing);
        let clean_feat = ds.composition.feature_descriptions.iter().find(|f| f.name == "clean").unwrap();
        let messy_feat = ds.composition.feature_descriptions.iter().find(|f| f.name == "messy").unwrap();

        assert_eq!(clean_feat.missing_ratio, 0.0);
        assert_eq!(messy_feat.missing_ratio, 0.75);

        let md = ds.to_markdown();
        assert!(md.contains("75.0%"));
    }

    #[test]
    fn test_datasheet_markdown_has_all_sections() {
        let ds = Datasheet::generate("test", "upload", 50, 3,
            &vec!["a".to_string()], &vec!["Int64".to_string()], &HashMap::new());
        let md = ds.to_markdown();

        assert!(md.contains("## Composition"));
        assert!(md.contains("## Collection"));
        assert!(md.contains("## Limitations"));
    }

    #[test]
    fn test_datasheet_serialization() {
        let ds = Datasheet::generate("test", "src", 100, 2,
            &vec!["a".to_string()], &vec!["Float64".to_string()], &HashMap::new());

        let json = serde_json::to_string(&ds).unwrap();
        let deserialized: Datasheet = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test");
        assert_eq!(deserialized.composition.num_samples, 100);
    }

    #[test]
    fn test_datasheet_default_fields() {
        let ds = Datasheet::generate("mydata", "kaggle", 1000, 10,
            &[], &[], &HashMap::new());
        assert_eq!(ds.collection.source, "kaggle");
        assert_eq!(ds.collection.collection_method, "Not specified");
        assert!(ds.collection.time_period.is_none());
        assert_eq!(ds.distribution.license, "Not specified");
        assert_eq!(ds.distribution.access_level, "Internal");
        assert_eq!(ds.preprocessing.rows_before, 1000);
        assert_eq!(ds.preprocessing.rows_after, 1000);
        assert!(ds.preprocessing.steps_applied.is_empty());
    }
}
