//! Model export and serialization module
//!
//! Provides functionality to export trained models to various formats:
//! - Native binary format (efficient for Rust)
//! - JSON format (portable, human-readable)
//! - ONNX format (interoperability with other ML frameworks)
//! - PMML format (Predictive Model Markup Language)

mod serializer;
mod onnx;
mod pmml;
mod versioning;

pub use serializer::{
    ModelSerializer, SerializationFormat, ModelMetadata,
    save_model, load_model, save_model_json, load_model_json,
};
pub use onnx::{ONNXExporter, ONNXConfig};
pub use pmml::{PMMLExporter};
pub use versioning::{ModelVersion, VersionedModel, ModelRegistry};
