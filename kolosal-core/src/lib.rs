//! Kolosal Core - High-performance AutoML engine
//!
//! This crate provides the core functionality for the Kolosal AutoML framework,
//! including data preprocessing, model training, inference, and hyperparameter optimization.

pub mod error;
pub mod preprocessing;
pub mod training;
pub mod inference;
pub mod optimizer;
pub mod utils;
pub mod explainability;
pub mod ensemble;
pub mod timeseries;
pub mod autopipeline;
pub mod calibration;
pub mod anomaly;
pub mod drift;
pub mod feature_engineering;
pub mod imputation;
pub mod synthetic;
pub mod export;
pub mod nas;

pub use error::{KolosalError, Result};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::error::{KolosalError, Result};
    pub use crate::preprocessing::{DataPreprocessor, PreprocessingConfig, FeatureSelector, SelectionMethod};
    pub use crate::training::{TrainEngine, TrainingConfig, ModelType};
    pub use crate::inference::{InferenceEngine, InferenceConfig};
    pub use crate::optimizer::{HyperOptX, OptimizationConfig, SearchSpace, BayesianOptimizer, GaussianProcess};
    pub use crate::explainability::{PermutationImportance, PartialDependence, LocalExplainer};
    pub use crate::ensemble::{VotingClassifier, VotingRegressor, StackingConfig};
    pub use crate::timeseries::{TimeSeriesFeatures, TimeSeriesCV, Differencer};
    pub use crate::autopipeline::{AutoPipeline, PipelineConfig, DataTypeDetector};
    pub use crate::calibration::{PlattScaling, IsotonicRegression, TemperatureScaling, Calibrator};
    pub use crate::anomaly::{IsolationForest, LocalOutlierFactor, AnomalyDetector};
    pub use crate::drift::{DataDriftDetector, ConceptDriftDetector, FeatureDriftMonitor, DriftResult};
    pub use crate::feature_engineering::{PolynomialFeatures, FeatureInteractions, TfidfVectorizer, FeatureTransformer};
    pub use crate::imputation::{MICEImputer, KNNImputer, IterativeImputer, Imputer};
    pub use crate::synthetic::{SMOTE, ADASYN, RandomOverSampler, Sampler};
    pub use crate::export::{ModelSerializer, ModelMetadata, SerializationFormat, ONNXExporter, PMMLExporter, ModelRegistry};
}
