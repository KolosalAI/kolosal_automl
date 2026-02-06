//! Python bindings for model export

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use kolosal_core::export::{ModelSerializer, ModelMetadata, SerializationFormat, ONNXExporter, PMMLExporter, ModelRegistry};

use crate::training::PyTrainEngine;

/// Serialization format enum for Python
#[pyclass(name = "SerializationFormat")]
#[derive(Clone)]
pub enum PySerializationFormat {
    Bincode,
    JSON,
    MessagePack,
    ONNX,
    PMML,
}

impl From<PySerializationFormat> for SerializationFormat {
    fn from(py: PySerializationFormat) -> Self {
        match py {
            PySerializationFormat::Bincode => SerializationFormat::Bincode,
            PySerializationFormat::JSON => SerializationFormat::JSON,
            PySerializationFormat::MessagePack => SerializationFormat::MessagePack,
            PySerializationFormat::ONNX => SerializationFormat::ONNX,
            PySerializationFormat::PMML => SerializationFormat::PMML,
        }
    }
}

/// Model serializer for Python
#[pyclass(name = "ModelSerializer")]
pub struct PyModelSerializer {
    inner: ModelSerializer,
}

#[pymethods]
impl PyModelSerializer {
    #[new]
    fn new() -> Self {
        Self {
            inner: ModelSerializer::new(),
        }
    }

    /// Save model to file
    fn save(
        &self,
        model: &PyTrainEngine,
        path: &str,
        format: Option<PySerializationFormat>,
    ) -> PyResult<()> {
        let fmt = format.map(|f| f.into()).unwrap_or(SerializationFormat::Bincode);
        
        self.inner
            .save(model.inner_ref(), path, fmt)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load model from file
    fn load(&self, path: &str) -> PyResult<PyTrainEngine> {
        let engine = self.inner
            .load(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(PyTrainEngine::from_inner(engine))
    }

    /// Save with metadata
    fn save_with_metadata(
        &self,
        model: &PyTrainEngine,
        path: &str,
        metadata: &PyModelMetadata,
    ) -> PyResult<()> {
        self.inner
            .save_with_metadata(model.inner_ref(), path, &metadata.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get supported formats
    fn supported_formats(&self) -> Vec<String> {
        vec![
            "bincode".to_string(),
            "json".to_string(),
            "messagepack".to_string(),
            "onnx".to_string(),
            "pmml".to_string(),
        ]
    }
}

/// Model metadata for Python
#[pyclass(name = "ModelMetadata")]
#[derive(Clone)]
pub struct PyModelMetadata {
    pub inner: ModelMetadata,
}

#[pymethods]
impl PyModelMetadata {
    #[new]
    fn new() -> Self {
        Self {
            inner: ModelMetadata::new(),
        }
    }

    /// Set model name
    fn with_name(&self, name: String) -> Self {
        Self {
            inner: self.inner.clone().with_name(name),
        }
    }

    /// Set model version
    fn with_version(&self, version: String) -> Self {
        Self {
            inner: self.inner.clone().with_version(version),
        }
    }

    /// Set description
    fn with_description(&self, description: String) -> Self {
        Self {
            inner: self.inner.clone().with_description(description),
        }
    }

    /// Add custom metadata
    fn with_custom(&self, key: String, value: String) -> Self {
        Self {
            inner: self.inner.clone().with_custom(key, value),
        }
    }

    /// Set feature names
    fn with_feature_names(&self, names: Vec<String>) -> Self {
        Self {
            inner: self.inner.clone().with_feature_names(names),
        }
    }

    /// Set target name
    fn with_target_name(&self, name: String) -> Self {
        Self {
            inner: self.inner.clone().with_target_name(name),
        }
    }

    /// To dict
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        
        if let Some(ref name) = self.inner.name {
            dict.set_item("name", name)?;
        }
        if let Some(ref version) = self.inner.version {
            dict.set_item("version", version)?;
        }
        if let Some(ref desc) = self.inner.description {
            dict.set_item("description", desc)?;
        }
        if let Some(ref features) = self.inner.feature_names {
            dict.set_item("feature_names", features)?;
        }
        if let Some(ref target) = self.inner.target_name {
            dict.set_item("target_name", target)?;
        }
        
        let custom = pyo3::types::PyDict::new(py);
        for (k, v) in &self.inner.custom {
            custom.set_item(k, v)?;
        }
        dict.set_item("custom", custom)?;
        
        Ok(dict.into())
    }

    /// Get name
    fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// Get version
    fn version(&self) -> Option<String> {
        self.inner.version.clone()
    }

    /// Get description
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
}

/// ONNX exporter for Python
#[pyclass(name = "ONNXExporter")]
pub struct PyONNXExporter {
    inner: ONNXExporter,
}

#[pymethods]
impl PyONNXExporter {
    #[new]
    fn new() -> Self {
        Self {
            inner: ONNXExporter::new(),
        }
    }

    /// Export model to ONNX format
    fn export(&self, model: &PyTrainEngine, path: &str) -> PyResult<()> {
        self.inner
            .export(model.inner_ref(), path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Export with input/output names
    fn export_with_names(
        &self,
        model: &PyTrainEngine,
        path: &str,
        input_names: Vec<String>,
        output_names: Vec<String>,
    ) -> PyResult<()> {
        self.inner
            .export_with_names(model.inner_ref(), path, &input_names, &output_names)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get ONNX opset version
    fn opset_version(&self) -> i64 {
        self.inner.opset_version()
    }

    /// Set ONNX opset version
    fn with_opset_version(&self, version: i64) -> Self {
        Self {
            inner: self.inner.clone().with_opset_version(version),
        }
    }
}

/// PMML exporter for Python
#[pyclass(name = "PMMLExporter")]
pub struct PyPMMLExporter {
    inner: PMMLExporter,
}

#[pymethods]
impl PyPMMLExporter {
    #[new]
    fn new() -> Self {
        Self {
            inner: PMMLExporter::new(),
        }
    }

    /// Export model to PMML format
    fn export(&self, model: &PyTrainEngine, path: &str) -> PyResult<()> {
        self.inner
            .export(model.inner_ref(), path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Export with metadata
    fn export_with_metadata(
        &self,
        model: &PyTrainEngine,
        path: &str,
        metadata: &PyModelMetadata,
    ) -> PyResult<()> {
        self.inner
            .export_with_metadata(model.inner_ref(), path, &metadata.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get PMML version
    fn pmml_version(&self) -> String {
        self.inner.pmml_version().to_string()
    }
}

/// Model registry for Python
#[pyclass(name = "ModelRegistry")]
pub struct PyModelRegistry {
    inner: ModelRegistry,
}

#[pymethods]
impl PyModelRegistry {
    #[new]
    fn new(base_path: &str) -> PyResult<Self> {
        let inner = ModelRegistry::new(base_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Register a model
    fn register(
        &mut self,
        name: String,
        model: &PyTrainEngine,
        metadata: Option<&PyModelMetadata>,
    ) -> PyResult<String> {
        let meta = metadata.map(|m| m.inner.clone());
        
        self.inner
            .register(&name, model.inner_ref(), meta.as_ref())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get model by name and version
    fn get(&self, name: &str, version: Option<&str>) -> PyResult<PyTrainEngine> {
        let engine = self.inner
            .get(name, version)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(PyTrainEngine::from_inner(engine))
    }

    /// List all models
    fn list_models(&self) -> Vec<String> {
        self.inner.list_models()
    }

    /// List versions for a model
    fn list_versions(&self, name: &str) -> Vec<String> {
        self.inner.list_versions(name)
    }

    /// Delete a model version
    fn delete(&mut self, name: &str, version: &str) -> PyResult<()> {
        self.inner
            .delete(name, version)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get latest version for a model
    fn latest_version(&self, name: &str) -> Option<String> {
        self.inner.latest_version(name)
    }

    /// Get model metadata
    fn get_metadata(&self, name: &str, version: Option<&str>) -> PyResult<PyModelMetadata> {
        let meta = self.inner
            .get_metadata(name, version)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(PyModelMetadata { inner: meta })
    }

    /// Get base path
    fn base_path(&self) -> String {
        self.inner.base_path().to_string()
    }
}
