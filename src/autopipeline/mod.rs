//! Automated ML Pipeline module
//!
//! Provides automatic pipeline composition and optimization including:
//! - Automatic data type detection
//! - Preprocessing step selection
//! - Model selection
//! - Pipeline optimization

mod detector;
mod composer;
mod pipeline;

pub use detector::{DataTypeDetector, DetectedSchema, ColumnInfo};
pub use composer::{PipelineComposer, ComposerConfig, PreprocessingStep, ModelStep};
pub use pipeline::{AutoPipeline, PipelineConfig, PipelineResult};

