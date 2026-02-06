//! Kolosal Core - High-performance AutoML engine
//!
//! This crate provides the core functionality for the Kolosal AutoML framework,
//! including data preprocessing, model training, inference, and hyperparameter optimization.
//!
//! # Modules
//!
//! ## Core ML Modules
//! - [`preprocessing`] - Data preprocessing, scaling, encoding
//! - [`training`] - Model training with multiple algorithms
//! - [`inference`] - High-performance inference engine
//! - [`optimizer`] - Hyperparameter optimization (Bayesian, ASHT)
//!
//! ## Advanced ML Features
//! - [`explainability`] - Model interpretability (SHAP-like, PDP)
//! - [`ensemble`] - Voting and stacking ensembles
//! - [`calibration`] - Probability calibration
//! - [`anomaly`] - Anomaly detection (Isolation Forest, LOF)
//! - [`drift`] - Data and concept drift detection
//!
//! ## Data Processing
//! - [`feature_engineering`] - Feature generation and transformation
//! - [`imputation`] - Missing value imputation
//! - [`synthetic`] - Synthetic data generation (SMOTE, ADASYN)
//! - [`timeseries`] - Time series features and validation
//!
//! ## Infrastructure
//! - [`batch`] - Batch processing with priority queues
//! - [`cache`] - Multi-level caching with LRU and TTL
//! - [`memory`] - Memory pooling for buffer management
//! - [`streaming`] - Streaming pipelines with backpressure
//! - [`quantization`] - Data quantization for efficiency
//! - [`monitoring`] - Performance metrics and statistics
//! - [`tracking`] - Experiment tracking
//!
//! ## Utilities
//! - [`autopipeline`] - Automatic pipeline construction
//! - [`export`] - Model serialization (ONNX, PMML)
//! - [`nas`] - Neural architecture search

// Core error handling
pub mod error;

// Core ML modules
pub mod preprocessing;
pub mod training;
pub mod inference;
pub mod optimizer;

// Advanced ML features
pub mod explainability;
pub mod ensemble;
pub mod calibration;
pub mod anomaly;
pub mod drift;

// Data processing
pub mod feature_engineering;
pub mod imputation;
pub mod synthetic;
pub mod timeseries;

// Infrastructure - Performance & Optimization
pub mod batch;
pub mod cache;
pub mod memory;
pub mod streaming;
pub mod quantization;
pub mod monitoring;
pub mod tracking;

// Utilities
pub mod autopipeline;
pub mod export;
pub mod nas;
pub mod utils;
pub mod architectures;

pub use error::{KolosalError, Result};

/// Re-export commonly used types
pub mod prelude {
    // Error handling
    pub use crate::error::{KolosalError, Result};
    
    // Preprocessing
    pub use crate::preprocessing::{DataPreprocessor, PreprocessingConfig, FeatureSelector, SelectionMethod};
    
    // Training
    pub use crate::training::{TrainEngine, TrainingConfig, ModelType};
    
    // Inference
    pub use crate::inference::{InferenceEngine, InferenceConfig};
    
    // Optimization
    pub use crate::optimizer::{HyperOptX, OptimizationConfig, SearchSpace, BayesianOptimizer, GaussianProcess};
    
    // Explainability
    pub use crate::explainability::{PermutationImportance, PartialDependence, LocalExplainer};
    
    // Ensemble
    pub use crate::ensemble::{VotingClassifier, VotingRegressor, StackingConfig};
    
    // Time series
    pub use crate::timeseries::{TimeSeriesFeatures, TimeSeriesCV, Differencer};
    
    // Auto pipeline
    pub use crate::autopipeline::{AutoPipeline, PipelineConfig, DataTypeDetector};
    
    // Calibration
    pub use crate::calibration::{PlattScaling, IsotonicRegression, TemperatureScaling, Calibrator};
    
    // Anomaly detection
    pub use crate::anomaly::{IsolationForest, LocalOutlierFactor, AnomalyDetector};
    
    // Drift detection
    pub use crate::drift::{DataDriftDetector, ConceptDriftDetector, FeatureDriftMonitor, DriftResult};
    
    // Feature engineering
    pub use crate::feature_engineering::{PolynomialFeatures, FeatureInteractions, TfidfVectorizer, FeatureTransformer};
    
    // Imputation
    pub use crate::imputation::{MICEImputer, KNNImputer, IterativeImputer, Imputer};
    
    // Synthetic data
    pub use crate::synthetic::{SMOTE, ADASYN, RandomOverSampler, Sampler};
    
    // Export
    pub use crate::export::{ModelSerializer, ModelMetadata, SerializationFormat, ONNXExporter, PMMLExporter, ModelRegistry};
    
    // Batch processing
    pub use crate::batch::{BatchProcessor, BatchProcessorConfig, DynamicBatcher, Priority, PriorityQueue};
    
    // Caching
    pub use crate::cache::{LruTtlCache, MultiLevelCache, CacheLevel};
    
    // Memory management
    pub use crate::memory::{MemoryPool, PooledBuffer};
    
    // Streaming
    pub use crate::streaming::{StreamingPipeline, StreamConfig, BackpressureController};
    
    // Quantization
    pub use crate::quantization::{Quantizer, QuantizationConfig, QuantizationType, QuantizationMode};
    
    // Monitoring
    pub use crate::monitoring::{PerformanceMetrics, BatchStats};
    
    // Experiment tracking
    pub use crate::tracking::{ExperimentTracker, ExperimentConfig, Experiment, Run};
}
