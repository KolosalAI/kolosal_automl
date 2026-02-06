//! HTTP request handlers

use std::sync::Arc;
use axum::{
    extract::{Multipart, Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    Json,
};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use tracing::{info, error};

use crate::preprocessing::{DataPreprocessor, PreprocessingConfig, ScalerType, ImputeStrategy};
use crate::training::{TrainEngine, TrainingConfig, TaskType, ModelType};
use crate::inference::{InferenceEngine, InferenceConfig};

use super::error::{Result, ServerError};
use super::state::{AppState, JobStatus, ModelInfo, TrainingJob};

// ============================================================================
// Data Handlers
// ============================================================================

/// Upload and parse a data file
pub async fn upload_data(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>> {
    while let Some(field) = multipart.next_field().await.map_err(|e| ServerError::BadRequest(e.to_string()))? {
        let name = field.name().unwrap_or("file").to_string();
        let file_name = field.file_name().unwrap_or("data.csv").to_string();
        let content_type = field.content_type().unwrap_or("text/csv").to_string();
        let data = field.bytes().await.map_err(|e| ServerError::BadRequest(e.to_string()))?;

        info!("Received file: {} ({} bytes)", file_name, data.len());

        // Parse based on file extension
        let df = if file_name.ends_with(".csv") {
            CsvReadOptions::default()
                .with_infer_schema_length(Some(1000))
                .with_has_header(true)
                .into_reader_with_file_handle(Cursor::new(&data))
                .finish()?
        } else if file_name.ends_with(".json") {
            JsonReader::new(Cursor::new(&data))
                .finish()?
        } else if file_name.ends_with(".parquet") {
            ParquetReader::new(Cursor::new(&data))
                .finish()?
        } else {
            return Err(ServerError::BadRequest(
                "Unsupported file format. Use CSV, JSON, or Parquet.".to_string()
            ));
        };

        // Store dataset
        let temp_path = std::env::temp_dir().join(&file_name);
        let dataset_id = state.store_dataset(file_name.clone(), df.clone(), temp_path).await;

        let column_names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        return Ok(Json(serde_json::json!({
            "success": true,
            "dataset_id": dataset_id,
            "name": file_name,
            "rows": df.height(),
            "columns": df.width(),
            "column_names": column_names,
        })));
    }

    Err(ServerError::BadRequest("No file uploaded".to_string()))
}

/// Get data preview (first N rows)
#[derive(Deserialize)]
pub struct PreviewQuery {
    rows: Option<usize>,
}

pub async fn get_data_preview(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PreviewQuery>,
) -> Result<Json<serde_json::Value>> {
    let data = state.current_data.read().await;
    
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?;

    let n_rows = query.rows.unwrap_or(10).min(100);
    let preview = df.head(Some(n_rows));

    // Convert to JSON-serializable format
    let columns: Vec<serde_json::Value> = preview.get_columns().iter().map(|col| {
        let values: Vec<serde_json::Value> = (0..col.len()).map(|i| {
            match col.get(i) {
                Ok(AnyValue::Float64(v)) => serde_json::json!(v),
                Ok(AnyValue::Float32(v)) => serde_json::json!(v),
                Ok(AnyValue::Int64(v)) => serde_json::json!(v),
                Ok(AnyValue::Int32(v)) => serde_json::json!(v),
                Ok(AnyValue::String(v)) => serde_json::json!(v),
                Ok(AnyValue::Boolean(v)) => serde_json::json!(v),
                Ok(AnyValue::Null) => serde_json::Value::Null,
                _ => serde_json::json!(col.get(i).map(|v| format!("{:?}", v)).unwrap_or_default()),
            }
        }).collect();
        
        let col_name = col.name().to_string();
        serde_json::json!({
            "name": col_name,
            "dtype": format!("{:?}", col.dtype()),
            "values": values,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "rows": preview.height(),
        "columns": columns,
    })))
}

/// Get dataset info
pub async fn get_data_info(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let data = state.current_data.read().await;
    
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?;

    let columns: Vec<serde_json::Value> = df.get_columns().iter().map(|col| {
        let null_count = col.null_count();
        let unique_count = col.n_unique().unwrap_or(0);
        let col_name = col.name().to_string();
        
        serde_json::json!({
            "name": col_name,
            "dtype": format!("{:?}", col.dtype()),
            "null_count": null_count,
            "unique_count": unique_count,
            "null_percent": (null_count as f64 / df.height() as f64) * 100.0,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "rows": df.height(),
        "columns": df.width(),
        "column_info": columns,
        "memory_usage_bytes": df.estimated_size(),
    })))
}

/// Load a sample dataset
pub async fn load_sample_data(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>> {
    // Generate sample datasets
    let df = match name.as_str() {
        "iris" => generate_iris_dataset()?,
        "diabetes" => generate_diabetes_dataset()?,
        "boston" => generate_boston_dataset()?,
        "wine" => generate_wine_dataset()?,
        "breast_cancer" => generate_breast_cancer_dataset()?,
        _ => return Err(ServerError::NotFound(format!("Unknown sample dataset: {}", name))),
    };

    let dataset_id = state.store_dataset(name.clone(), df.clone(), std::env::temp_dir().join(&name)).await;
    let column_names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();

    Ok(Json(serde_json::json!({
        "success": true,
        "dataset_id": dataset_id,
        "name": name,
        "rows": df.height(),
        "columns": df.width(),
        "column_names": column_names,
    })))
}

// ============================================================================
// Preprocessing Handlers
// ============================================================================

#[derive(Deserialize)]
pub struct PreprocessRequest {
    scaler: Option<String>,
    imputation: Option<String>,
    target_column: Option<String>,
}

pub async fn run_preprocessing(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PreprocessRequest>,
) -> Result<Json<serde_json::Value>> {
    let mut data = state.current_data.write().await;
    
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
        .clone();

    // Build preprocessing config
    let mut config = PreprocessingConfig::default();
    
    if let Some(scaler) = &request.scaler {
        config = config.with_scaler(match scaler.as_str() {
            "standard" => ScalerType::Standard,
            "minmax" => ScalerType::MinMax,
            "robust" => ScalerType::Robust,
            _ => ScalerType::None,
        });
    }

    if let Some(imputation) = &request.imputation {
        config = config.with_numeric_impute(match imputation.as_str() {
            "mean" => ImputeStrategy::Mean,
            "median" => ImputeStrategy::Median,
            "mode" => ImputeStrategy::MostFrequent,
            _ => ImputeStrategy::Drop,
        });
    }

    // Run preprocessing
    let mut preprocessor = DataPreprocessor::with_config(config);
    let processed = preprocessor.fit_transform(&df)
        .map_err(|e| ServerError::Internal(format!("Preprocessing failed: {:?}", e)))?;

    // Update state
    *data = Some(processed.clone());
    *state.preprocessor.write().await = Some(preprocessor);

    Ok(Json(serde_json::json!({
        "success": true,
        "rows": processed.height(),
        "columns": processed.width(),
        "message": "Preprocessing completed successfully",
    })))
}

pub async fn get_preprocessing_config() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "scalers": ["none", "standard", "minmax", "robust"],
        "imputation_strategies": ["drop", "mean", "median", "mode"],
        "encoders": ["label", "onehot"],
    }))
}

// ============================================================================
// Training Handlers
// ============================================================================

#[derive(Deserialize)]
pub struct TrainRequest {
    target_column: String,
    task_type: String,
    model_type: String,
    cv_folds: Option<usize>,
    test_size: Option<f64>,
    hyperparameters: Option<serde_json::Value>,
}

pub async fn start_training(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TrainRequest>,
) -> Result<Json<serde_json::Value>> {
    let data = state.current_data.read().await;
    
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
        .clone();

    // Parse task type
    let task_type = match request.task_type.as_str() {
        "classification" => TaskType::BinaryClassification,
        "multiclass" => TaskType::MultiClassification,
        "regression" => TaskType::Regression,
        _ => return Err(ServerError::BadRequest("Invalid task type".to_string())),
    };

    // Parse model type
    let model_type = match request.model_type.as_str() {
        "linear" | "linear_regression" => ModelType::LinearRegression,
        "logistic" | "logistic_regression" => ModelType::LogisticRegression,
        "decision_tree" => ModelType::DecisionTree,
        "random_forest" => ModelType::RandomForest,
        _ => return Err(ServerError::BadRequest(format!("Invalid model type: {}", request.model_type))),
    };

    // Create job ID
    let job_id = AppState::generate_id();
    let job_id_for_response = job_id.clone();
    
    // Store job
    let job = TrainingJob {
        id: job_id.clone(),
        status: JobStatus::Running { 
            progress: 0.0, 
            message: "Starting training...".to_string() 
        },
        config: serde_json::json!({
            "target_column": request.target_column,
            "task_type": request.task_type,
            "model_type": request.model_type,
        }),
        created_at: chrono::Utc::now(),
        model_path: None,
    };
    state.jobs.write().await.insert(job_id.clone(), job);

    // Run training in background
    let state_clone = state.clone();
    let target_col = request.target_column.clone();
    
    tokio::spawn(async move {
        let result = run_training_job(
            &df, 
            &target_col, 
            task_type, 
            model_type,
        ).await;

        let mut jobs = state_clone.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            match result {
                Ok(metrics) => {
                    job.status = JobStatus::Completed { metrics };
                    info!("Training job {} completed successfully", job_id);
                }
                Err(e) => {
                    job.status = JobStatus::Failed { error: e.to_string() };
                    error!("Training job {} failed: {}", job_id, e);
                }
            }
        }
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "job_id": job_id_for_response,
        "message": "Training started",
    })))
}

async fn run_training_job(
    df: &DataFrame,
    target_column: &str,
    task_type: TaskType,
    model_type: ModelType,
) -> anyhow::Result<serde_json::Value> {
    let config = TrainingConfig::new(task_type.clone(), target_column)
        .with_model(model_type);
    
    let mut engine = TrainEngine::new(config);
    
    engine.fit(df)?;
    
    let metrics = engine.metrics().cloned().unwrap_or_default();
    
    let result = match task_type {
        TaskType::BinaryClassification | TaskType::MultiClassification => serde_json::json!({
            "accuracy": metrics.accuracy.unwrap_or(0.0),
            "precision": metrics.precision.unwrap_or(0.0),
            "recall": metrics.recall.unwrap_or(0.0),
            "f1": metrics.f1_score.unwrap_or(0.0),
            "training_time_secs": metrics.training_time_secs,
        }),
        TaskType::Regression | TaskType::TimeSeries => serde_json::json!({
            "r2": metrics.r2.unwrap_or(0.0),
            "mse": metrics.mse.unwrap_or(0.0),
            "mae": metrics.mae.unwrap_or(0.0),
            "training_time_secs": metrics.training_time_secs,
        }),
    };

    Ok(result)
}

pub async fn get_training_status(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<serde_json::Value>> {
    let jobs = state.jobs.read().await;
    
    let job = jobs.get(&job_id)
        .ok_or_else(|| ServerError::NotFound(format!("Job not found: {}", job_id)))?;

    Ok(Json(serde_json::json!({
        "job_id": job.id,
        "status": job.status,
        "config": job.config,
        "created_at": job.created_at.to_rfc3339(),
    })))
}

#[derive(Deserialize)]
pub struct CompareRequest {
    target_column: String,
    task_type: String,
    models: Vec<String>,
    cv_folds: Option<usize>,
}

pub async fn compare_models(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompareRequest>,
) -> Result<Json<serde_json::Value>> {
    let data = state.current_data.read().await;
    
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
        .clone();

    let task_type = match request.task_type.as_str() {
        "classification" => TaskType::BinaryClassification,
        "multiclass" => TaskType::MultiClassification,
        "regression" => TaskType::Regression,
        _ => return Err(ServerError::BadRequest("Invalid task type".to_string())),
    };

    let mut results = Vec::new();

    for model_name in &request.models {
        let model_type = match model_name.as_str() {
            "linear" | "linear_regression" => ModelType::LinearRegression,
            "logistic" | "logistic_regression" => ModelType::LogisticRegression,
            "decision_tree" => ModelType::DecisionTree,
            "random_forest" => ModelType::RandomForest,
            _ => continue,
        };

        if let Ok(metrics) = run_training_job(&df, &request.target_column, task_type.clone(), model_type).await {
            results.push(serde_json::json!({
                "model": model_name,
                "metrics": metrics,
            }));
        }
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "comparison": results,
    })))
}

// ============================================================================
// Model Handlers
// ============================================================================

pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let models = state.models.read().await;
    let model_list: Vec<&ModelInfo> = models.values().collect();
    
    Json(serde_json::json!({
        "models": model_list,
    }))
}

pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>> {
    let models = state.models.read().await;
    
    let model = models.get(&model_id)
        .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", model_id)))?;

    Ok(Json(serde_json::json!(model)))
}

pub async fn download_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<impl IntoResponse> {
    let models = state.models.read().await;

    let model = models.get(&model_id)
        .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", model_id)))?;

    let json_bytes = serde_json::to_vec_pretty(model)
        .map_err(|e| ServerError::Internal(format!("Failed to serialize model: {}", e)))?;

    let disposition = format!("attachment; filename=\"model_{}.json\"", model_id);

    Ok((
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, axum::http::HeaderValue::from_static("application/json")),
            (axum::http::header::CONTENT_DISPOSITION, axum::http::HeaderValue::from_str(&disposition)
                .map_err(|e| ServerError::Internal(format!("Invalid header: {}", e)))?),
        ],
        json_bytes,
    ))
}

// ============================================================================
// Prediction Handlers
// ============================================================================

#[derive(Deserialize)]
pub struct PredictRequest {
    model_id: Option<String>,
    data: serde_json::Value,
}

pub async fn predict(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<serde_json::Value>> {
    let models = state.models.read().await;

    // Find the model: use provided model_id or most recently created
    let model_info = if let Some(ref id) = request.model_id {
        models.get(id)
            .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", id)))?
    } else {
        models.values()
            .max_by_key(|m| &m.created_at)
            .ok_or_else(|| ServerError::NotFound("No trained models available".to_string()))?
    };

    // Parse features from data field
    let features: Vec<Vec<f64>> = serde_json::from_value(request.data)
        .map_err(|e| ServerError::BadRequest(format!("Invalid features format, expected array of arrays of f64: {}", e)))?;

    if features.is_empty() {
        return Err(ServerError::BadRequest("Features array is empty".to_string()));
    }

    // Load the trained model
    let model_path = model_info.path.to_str()
        .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?;

    let engine = InferenceEngine::load(InferenceConfig::new(), None, model_path)
        .map_err(|e| ServerError::Internal(format!("Failed to load model: {}", e)))?;

    // Convert Vec<Vec<f64>> to Array2
    let n_rows = features.len();
    let n_cols = features[0].len();
    let flat: Vec<f64> = features.into_iter().flatten().collect();
    let x = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| ServerError::BadRequest(format!("Invalid feature dimensions: {}", e)))?;

    // Run prediction
    let predictions = engine.predict_array(&x)
        .map_err(|e| ServerError::Internal(format!("Prediction failed: {}", e)))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "predictions": predictions.to_vec(),
    })))
}

#[derive(Deserialize)]
pub struct BatchPredictRequest {
    model_id: Option<String>,
    data: Vec<Vec<f64>>,
}

pub async fn predict_batch(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchPredictRequest>,
) -> Result<Json<serde_json::Value>> {
    let models = state.models.read().await;

    // Find the model: use provided model_id or most recently created
    let model_info = if let Some(ref id) = request.model_id {
        models.get(id)
            .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", id)))?
    } else {
        models.values()
            .max_by_key(|m| &m.created_at)
            .ok_or_else(|| ServerError::NotFound("No trained models available".to_string()))?
    };

    if request.data.is_empty() {
        return Err(ServerError::BadRequest("Data array is empty".to_string()));
    }

    // Load the trained model
    let model_path = model_info.path.to_str()
        .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?;

    let engine = InferenceEngine::load(InferenceConfig::new(), None, model_path)
        .map_err(|e| ServerError::Internal(format!("Failed to load model: {}", e)))?;

    // Convert Vec<Vec<f64>> to Array2
    let n_rows = request.data.len();
    let n_cols = request.data[0].len();
    let flat: Vec<f64> = request.data.into_iter().flatten().collect();
    let x = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| ServerError::BadRequest(format!("Invalid data dimensions: {}", e)))?;

    // Run batch prediction
    let predictions = engine.predict_array(&x)
        .map_err(|e| ServerError::Internal(format!("Batch prediction failed: {}", e)))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "predictions": predictions.to_vec(),
        "count": predictions.len(),
    })))
}

// ============================================================================
// System Handlers
// ============================================================================

pub async fn get_system_status(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let system_info = state.get_system_info();
    let datasets = state.datasets.read().await;
    let jobs = state.jobs.read().await;
    let models = state.models.read().await;

    Json(serde_json::json!({
        "system": system_info,
        "datasets_count": datasets.len(),
        "jobs_count": jobs.len(),
        "models_count": models.len(),
        "status": "healthy",
    }))
}

pub async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

// ============================================================================
// UI Handler
// ============================================================================

pub async fn serve_index() -> Html<String> {
    // Embedded HTML for portability
    Html(EMBEDDED_INDEX_HTML.to_string())
}

const EMBEDDED_INDEX_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kolosal AutoML</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>[x-cloak]{display:none!important}.tab-active{background-color:rgb(59 130 246);color:white}</style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen" x-data="app()">
    <header class="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div class="flex items-center justify-between">
            <h1 class="text-xl font-bold">🚀 Kolosal AutoML</h1>
            <span class="text-sm text-gray-400">v0.2.0 - Pure Rust</span>
        </div>
    </header>
    <nav class="bg-gray-800 px-6 py-2 border-b border-gray-700">
        <div class="flex space-x-1">
            <button @click="tab='data'" :class="tab==='data'?'tab-active':'hover:bg-gray-700'" class="px-4 py-2 rounded-md text-sm">📊 Data</button>
            <button @click="tab='config'" :class="tab==='config'?'tab-active':'hover:bg-gray-700'" class="px-4 py-2 rounded-md text-sm">⚙️ Config</button>
            <button @click="tab='train'" :class="tab==='train'?'tab-active':'hover:bg-gray-700'" class="px-4 py-2 rounded-md text-sm">🚀 Train</button>
            <button @click="tab='monitor'" :class="tab==='monitor'?'tab-active':'hover:bg-gray-700'" class="px-4 py-2 rounded-md text-sm">📡 Monitor</button>
        </div>
    </nav>
    <main class="p-6">
        <div x-show="tab==='data'" x-cloak>
            <div class="grid grid-cols-2 gap-6">
                <div class="bg-gray-800 rounded-lg p-6">
                    <h2 class="text-lg font-semibold mb-4">Sample Datasets</h2>
                    <div class="grid grid-cols-2 gap-3">
                        <button @click="loadSample('iris')" class="p-3 bg-gray-700 rounded hover:bg-gray-600">Iris</button>
                        <button @click="loadSample('diabetes')" class="p-3 bg-gray-700 rounded hover:bg-gray-600">Diabetes</button>
                        <button @click="loadSample('boston')" class="p-3 bg-gray-700 rounded hover:bg-gray-600">Boston</button>
                        <button @click="loadSample('wine')" class="p-3 bg-gray-700 rounded hover:bg-gray-600">Wine</button>
                    </div>
                </div>
                <div class="bg-gray-800 rounded-lg p-6" x-show="data">
                    <h2 class="text-lg font-semibold mb-4">Data Info</h2>
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-gray-700 p-3 rounded"><div class="text-xl font-bold" x-text="data?.rows"></div><div class="text-sm text-gray-400">Rows</div></div>
                        <div class="bg-gray-700 p-3 rounded"><div class="text-xl font-bold" x-text="data?.columns"></div><div class="text-sm text-gray-400">Columns</div></div>
                    </div>
                </div>
            </div>
        </div>
        <div x-show="tab==='config'" x-cloak class="bg-gray-800 rounded-lg p-6">
            <h2 class="text-lg font-semibold mb-4">Configuration</h2>
            <div class="grid grid-cols-2 gap-4">
                <div><label class="block text-sm mb-1">Target Column</label><select x-model="cfg.target" class="w-full bg-gray-700 rounded p-2"><template x-for="c in data?.column_names||[]"><option :value="c" x-text="c"></option></template></select></div>
                <div><label class="block text-sm mb-1">Task Type</label><select x-model="cfg.task" class="w-full bg-gray-700 rounded p-2"><option value="binary_classification">Classification</option><option value="regression">Regression</option></select></div>
                <div><label class="block text-sm mb-1">Model</label><select x-model="cfg.model" class="w-full bg-gray-700 rounded p-2"><option value="random_forest">Random Forest</option><option value="decision_tree">Decision Tree</option><option value="logistic_regression">Logistic Regression</option></select></div>
            </div>
        </div>
        <div x-show="tab==='train'" x-cloak class="bg-gray-800 rounded-lg p-6">
            <div class="flex justify-between mb-6"><h2 class="text-lg font-semibold">Training</h2><button @click="train()" :disabled="!data||training" class="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded"><span x-show="!training">🚀 Train</span><span x-show="training">⏳ Training...</span></button></div>
            <div x-show="result" class="grid grid-cols-4 gap-4">
                <div class="bg-gray-700 p-4 rounded"><div class="text-2xl font-bold text-green-500" x-text="result?.accuracy?.toFixed(4)||'-'"></div><div class="text-sm text-gray-400">Accuracy</div></div>
                <div class="bg-gray-700 p-4 rounded"><div class="text-2xl font-bold text-blue-500" x-text="result?.r2?.toFixed(4)||'-'"></div><div class="text-sm text-gray-400">R²</div></div>
                <div class="bg-gray-700 p-4 rounded"><div class="text-2xl font-bold text-yellow-500" x-text="result?.mse?.toFixed(4)||'-'"></div><div class="text-sm text-gray-400">MSE</div></div>
                <div class="bg-gray-700 p-4 rounded"><div class="text-2xl font-bold" x-text="result?.training_time_secs?.toFixed(2)+'s'||'-'"></div><div class="text-sm text-gray-400">Time</div></div>
            </div>
        </div>
        <div x-show="tab==='monitor'" x-cloak class="bg-gray-800 rounded-lg p-6">
            <h2 class="text-lg font-semibold mb-4">System Status</h2>
            <div class="space-y-4">
                <div><div class="flex justify-between mb-1"><span>CPU</span><span x-text="sys.cpu_percent+'%'"></span></div><div class="w-full bg-gray-700 rounded h-2"><div class="bg-blue-500 h-2 rounded" :style="'width:'+sys.cpu_percent+'%'"></div></div></div>
                <div><div class="flex justify-between mb-1"><span>Memory</span><span x-text="sys.memory_percent+'%'"></span></div><div class="w-full bg-gray-700 rounded h-2"><div class="bg-green-500 h-2 rounded" :style="'width:'+sys.memory_percent+'%'"></div></div></div>
            </div>
        </div>
    </main>
    <script>
    function app(){return{tab:'data',data:null,training:false,result:null,cfg:{target:'',task:'binary_classification',model:'random_forest'},sys:{cpu_percent:0,memory_percent:0},
    init(){this.fetchSys();setInterval(()=>this.fetchSys(),5000)},
    async loadSample(n){try{const r=await fetch('/api/data/sample/'+n,{method:'POST'});this.data=await r.json()}catch(e){console.error(e)}},
    async train(){this.training=true;this.result=null;try{const r=await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target_column:this.cfg.target,task_type:this.cfg.task,model_type:this.cfg.model})});const d=await r.json();this.result=d.metrics}catch(e){console.error(e)}finally{this.training=false}},
    async fetchSys(){try{const r=await fetch('/api/system/status');this.sys=await r.json()}catch(e){}}}}
    </script>
</body>
</html>"#;

// ============================================================================
// Sample Data Generators
// ============================================================================

fn generate_iris_dataset() -> Result<DataFrame> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = 150;

    let sepal_length: Vec<f64> = (0..n).map(|i| {
        let base = match i / 50 { 0 => 5.0, 1 => 5.9, _ => 6.6 };
        base + rng.gen::<f64>() * 0.8
    }).collect();

    let sepal_width: Vec<f64> = (0..n).map(|i| {
        let base = match i / 50 { 0 => 3.4, 1 => 2.8, _ => 3.0 };
        base + rng.gen::<f64>() * 0.5
    }).collect();

    let petal_length: Vec<f64> = (0..n).map(|i| {
        let base = match i / 50 { 0 => 1.4, 1 => 4.3, _ => 5.5 };
        base + rng.gen::<f64>() * 0.5
    }).collect();

    let petal_width: Vec<f64> = (0..n).map(|i| {
        let base = match i / 50 { 0 => 0.2, 1 => 1.3, _ => 2.0 };
        base + rng.gen::<f64>() * 0.3
    }).collect();

    let species: Vec<i32> = (0..n).map(|i| (i / 50) as i32).collect();

    Ok(DataFrame::new(vec![
        Series::new("sepal_length".into(), sepal_length).into(),
        Series::new("sepal_width".into(), sepal_width).into(),
        Series::new("petal_length".into(), petal_length).into(),
        Series::new("petal_width".into(), petal_width).into(),
        Series::new("species".into(), species).into(),
    ])?)
}

fn generate_diabetes_dataset() -> Result<DataFrame> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = 442;

    let age: Vec<f64> = (0..n).map(|_| rng.gen_range(19.0..79.0)).collect();
    let bmi: Vec<f64> = (0..n).map(|_| rng.gen_range(18.0..42.0)).collect();
    let bp: Vec<f64> = (0..n).map(|_| rng.gen_range(60.0..130.0)).collect();
    let s1: Vec<f64> = (0..n).map(|_| rng.gen_range(90.0..300.0)).collect();
    let s2: Vec<f64> = (0..n).map(|_| rng.gen_range(40.0..240.0)).collect();
    
    let target: Vec<f64> = (0..n).enumerate().map(|(i, _)| {
        let base = age[i] * 0.5 + bmi[i] * 3.0 + bp[i] * 0.5;
        base + rng.gen::<f64>() * 50.0
    }).collect();

    Ok(DataFrame::new(vec![
        Series::new("age".into(), age).into(),
        Series::new("bmi".into(), bmi).into(),
        Series::new("bp".into(), bp).into(),
        Series::new("s1".into(), s1).into(),
        Series::new("s2".into(), s2).into(),
        Series::new("target".into(), target).into(),
    ])?)
}

fn generate_boston_dataset() -> Result<DataFrame> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = 506;

    let crim: Vec<f64> = (0..n).map(|_| rng.gen_range(0.01..90.0)).collect();
    let zn: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..100.0)).collect();
    let indus: Vec<f64> = (0..n).map(|_| rng.gen_range(0.5..28.0)).collect();
    let nox: Vec<f64> = (0..n).map(|_| rng.gen_range(0.38..0.87)).collect();
    let rm: Vec<f64> = (0..n).map(|_| rng.gen_range(3.5..8.8)).collect();
    let age: Vec<f64> = (0..n).map(|_| rng.gen_range(2.0..100.0)).collect();
    
    let price: Vec<f64> = (0..n).enumerate().map(|(i, _)| {
        let base = rm[i] * 5.0 - crim[i] * 0.1 - nox[i] * 10.0;
        (base + rng.gen::<f64>() * 5.0).max(5.0).min(50.0)
    }).collect();

    Ok(DataFrame::new(vec![
        Series::new("crim".into(), crim).into(),
        Series::new("zn".into(), zn).into(),
        Series::new("indus".into(), indus).into(),
        Series::new("nox".into(), nox).into(),
        Series::new("rm".into(), rm).into(),
        Series::new("age".into(), age).into(),
        Series::new("price".into(), price).into(),
    ])?)
}

fn generate_wine_dataset() -> Result<DataFrame> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = 178;

    let alcohol: Vec<f64> = (0..n).map(|i| {
        let base = match i % 3 { 0 => 13.7, 1 => 12.4, _ => 13.2 };
        base + rng.gen::<f64>() * 0.8
    }).collect();
    
    let malic_acid: Vec<f64> = (0..n).map(|_| rng.gen_range(0.7..5.8)).collect();
    let ash: Vec<f64> = (0..n).map(|_| rng.gen_range(1.4..3.2)).collect();
    let flavanoids: Vec<f64> = (0..n).map(|_| rng.gen_range(0.3..5.1)).collect();
    
    let class: Vec<i32> = (0..n).map(|i| (i % 3) as i32).collect();

    Ok(DataFrame::new(vec![
        Series::new("alcohol".into(), alcohol).into(),
        Series::new("malic_acid".into(), malic_acid).into(),
        Series::new("ash".into(), ash).into(),
        Series::new("flavanoids".into(), flavanoids).into(),
        Series::new("class".into(), class).into(),
    ])?)
}

fn generate_breast_cancer_dataset() -> Result<DataFrame> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = 569;

    let radius_mean: Vec<f64> = (0..n).map(|i| {
        if i < 212 { rng.gen_range(17.0..28.0) } else { rng.gen_range(6.0..17.0) }
    }).collect();
    
    let texture_mean: Vec<f64> = (0..n).map(|_| rng.gen_range(9.0..40.0)).collect();
    let perimeter_mean: Vec<f64> = (0..n).map(|i| radius_mean[i] * 6.28 + rng.gen::<f64>() * 10.0).collect();
    let area_mean: Vec<f64> = (0..n).map(|i| radius_mean[i].powi(2) * 3.14 + rng.gen::<f64>() * 100.0).collect();
    let smoothness_mean: Vec<f64> = (0..n).map(|_| rng.gen_range(0.05..0.16)).collect();
    
    let diagnosis: Vec<i32> = (0..n).map(|i| if i < 212 { 1 } else { 0 }).collect();

    Ok(DataFrame::new(vec![
        Series::new("radius_mean".into(), radius_mean).into(),
        Series::new("texture_mean".into(), texture_mean).into(),
        Series::new("perimeter_mean".into(), perimeter_mean).into(),
        Series::new("area_mean".into(), area_mean).into(),
        Series::new("smoothness_mean".into(), smoothness_mean).into(),
        Series::new("diagnosis".into(), diagnosis).into(),
    ])?)
}

// ============================================================================
// Batch Processor Handlers
// ============================================================================

/// Configure the batch processor
pub async fn configure_batch_processor(
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let max_batch_size = body.get("max_batch_size").and_then(|v| v.as_u64()).unwrap_or(32);
    let num_workers = body.get("num_workers").and_then(|v| v.as_u64()).unwrap_or(4);
    let enable_priority = body.get("enable_priority").and_then(|v| v.as_bool()).unwrap_or(false);
    let adaptive_sizing = body.get("adaptive_sizing").and_then(|v| v.as_bool()).unwrap_or(false);

    Json(serde_json::json!({
        "success": true,
        "message": "Batch processor configured",
        "config": {
            "max_batch_size": max_batch_size,
            "num_workers": num_workers,
            "enable_priority": enable_priority,
            "adaptive_sizing": adaptive_sizing,
        }
    }))
}

/// Get batch processor statistics
pub async fn get_batch_stats(
    State(_state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "total_batches_processed": 0,
        "total_items_processed": 0,
        "average_batch_size": 0.0,
        "average_processing_time_ms": 0.0,
        "queue_depth": 0,
        "active_workers": 0,
    }))
}

// ============================================================================
// Quantizer Handlers
// ============================================================================

/// Quantize input data
pub async fn quantize_data(
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>> {
    use crate::quantization::{Quantizer, QuantizationConfig, QuantizationType};

    let data: Vec<Vec<f64>> = serde_json::from_value(
        body.get("data").cloned().ok_or_else(|| ServerError::BadRequest("missing 'data' field".into()))?
    ).map_err(|e| ServerError::BadRequest(format!("invalid data format: {}", e)))?;

    let bits = body.get("bits").and_then(|v| v.as_u64()).unwrap_or(8) as u8;

    let quantization_type = match body.get("quantization_type").and_then(|v| v.as_str()).unwrap_or("int8") {
        "uint8" => QuantizationType::UInt8,
        "int16" => QuantizationType::Int16,
        "float16" => QuantizationType::Float16,
        _ => QuantizationType::Int8,
    };

    let config = QuantizationConfig {
        quantization_type,
        num_bits: bits,
        ..Default::default()
    };

    let flat_data: Vec<f64> = data.iter().flat_map(|row| row.iter().copied()).collect();
    let original_size = flat_data.len() * 8; // f64 = 8 bytes

    let mut quantizer = Quantizer::new(config);
    quantizer.calibrate(&flat_data);
    let quantized = quantizer.quantize(&flat_data);

    let stats = quantizer.stats();

    Ok(Json(serde_json::json!({
        "quantized_data": quantized.data,
        "scale": quantized.scale,
        "zero_point": quantized.zero_point,
        "shape": quantized.shape,
        "compression_ratio": quantized.compression_ratio(),
        "stats": {
            "original_bytes": original_size,
            "quantized_bytes": quantized.size_bytes(),
            "clip_rate": stats.clip_rate,
            "is_calibrated": stats.is_calibrated,
        }
    })))
}

/// Dequantize data back to f64
pub async fn dequantize_data(
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>> {
    use crate::quantization::{QuantizedData, QuantizationType};

    let quantized_values: Vec<i32> = serde_json::from_value(
        body.get("data").cloned().ok_or_else(|| ServerError::BadRequest("missing 'data' field".into()))?
    ).map_err(|e| ServerError::BadRequest(format!("invalid data format: {}", e)))?;

    let scale = body.get("scale").and_then(|v| v.as_f64())
        .ok_or_else(|| ServerError::BadRequest("missing 'scale' field".into()))?;
    let zero_point = body.get("zero_point").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let quantized = QuantizedData {
        data: quantized_values,
        scale,
        zero_point,
        shape: vec![],
        quantization_type: QuantizationType::Int8,
    };

    let dequantized = quantized.dequantize();

    Ok(Json(serde_json::json!({
        "data": dequantized,
        "count": dequantized.len(),
    })))
}

/// Calibrate a quantizer with sample data
pub async fn calibrate_quantizer(
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>> {
    use crate::quantization::{Quantizer, QuantizationConfig};

    let calibration_data: Vec<f64> = serde_json::from_value(
        body.get("calibration_data").cloned()
            .ok_or_else(|| ServerError::BadRequest("missing 'calibration_data' field".into()))?
    ).map_err(|e| ServerError::BadRequest(format!("invalid calibration data: {}", e)))?;

    let config = QuantizationConfig::default();
    let mut quantizer = Quantizer::new(config);
    quantizer.calibrate(&calibration_data);

    let stats = quantizer.stats();

    Ok(Json(serde_json::json!({
        "success": true,
        "is_calibrated": stats.is_calibrated,
        "scale": stats.scale,
        "zero_point": stats.zero_point,
    })))
}

// ============================================================================
// Device Optimizer Handlers
// ============================================================================

/// Get device information (CPU, memory, disk)
pub async fn get_device_info() -> Json<serde_json::Value> {
    use sysinfo::{System, Disks};

    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_usage: f32 = sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
        / sys.cpus().len().max(1) as f32;

    let disks = Disks::new_with_refreshed_list();
    let disk_info: Vec<serde_json::Value> = disks.iter().map(|d| {
        serde_json::json!({
            "name": d.name().to_string_lossy(),
            "total_gb": d.total_space() as f64 / 1024.0 / 1024.0 / 1024.0,
            "available_gb": d.available_space() as f64 / 1024.0 / 1024.0 / 1024.0,
        })
    }).collect();

    Json(serde_json::json!({
        "cpu": {
            "count": sys.cpus().len(),
            "usage_percent": cpu_usage,
            "brand": sys.cpus().first().map(|c| c.brand().to_string()).unwrap_or_default(),
        },
        "memory": {
            "total_gb": sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
            "used_gb": sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
            "usage_percent": (sys.used_memory() as f64 / sys.total_memory().max(1) as f64) * 100.0,
        },
        "disks": disk_info,
    }))
}

/// Get auto-tuned optimal configurations
pub async fn get_optimal_configs() -> Json<serde_json::Value> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_count = sys.cpus().len();
    let total_memory_gb = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

    let recommended_workers = (cpu_count as f64 * 0.75).ceil() as usize;
    let recommended_batch_size = if total_memory_gb > 16.0 { 64 } else if total_memory_gb > 8.0 { 32 } else { 16 };
    let recommended_cache_mb = (total_memory_gb * 0.1 * 1024.0) as usize;

    Json(serde_json::json!({
        "batch_processor": {
            "max_batch_size": recommended_batch_size,
            "num_workers": recommended_workers,
        },
        "quantization": {
            "type": "int8",
            "bits": 8,
        },
        "cache": {
            "max_size_mb": recommended_cache_mb,
        },
        "training": {
            "parallelism": cpu_count,
        },
    }))
}

// ============================================================================
// Model Manager Handlers
// ============================================================================

/// Delete a model by ID
pub async fn delete_model(
    Path(model_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let mut models = state.models.write().await;
    if models.remove(&model_id).is_some() {
        Ok(Json(serde_json::json!({
            "success": true,
            "message": format!("Model '{}' deleted", model_id),
        })))
    } else {
        Err(ServerError::NotFound(format!("Model '{}' not found", model_id)))
    }
}

/// Get detailed metrics for a specific model
pub async fn get_model_metrics(
    Path(model_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let models = state.models.read().await;
    let model = models.get(&model_id)
        .ok_or_else(|| ServerError::NotFound(format!("Model '{}' not found", model_id)))?;

    Ok(Json(serde_json::json!({
        "model_id": model.id,
        "name": model.name,
        "task_type": model.task_type,
        "metrics": model.metrics,
        "created_at": model.created_at,
    })))
}

// ============================================================================
// Monitoring Handlers
// ============================================================================

/// Serve an HTML monitoring dashboard
pub async fn get_monitoring_dashboard() -> Html<String> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_usage: f32 = sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
        / sys.cpus().len().max(1) as f32;
    let mem_usage = (sys.used_memory() as f64 / sys.total_memory().max(1) as f64) * 100.0;

    let html = format!(r#"<!DOCTYPE html>
<html><head><title>Kolosal AutoML - Monitoring</title>
<style>
body {{ font-family: sans-serif; margin: 2em; background: #f5f5f5; }}
.card {{ background: white; border-radius: 8px; padding: 1.5em; margin: 1em 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
h1 {{ color: #333; }} h2 {{ color: #555; margin-top: 0; }}
.metric {{ font-size: 2em; font-weight: bold; color: #2563eb; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1em; }}
</style></head><body>
<h1>Kolosal AutoML Monitoring</h1>
<div class="grid">
  <div class="card"><h2>CPU Usage</h2><div class="metric">{:.1}%</div></div>
  <div class="card"><h2>Memory Usage</h2><div class="metric">{:.1}%</div></div>
  <div class="card"><h2>CPUs</h2><div class="metric">{}</div></div>
  <div class="card"><h2>Total Memory</h2><div class="metric">{:.1} GB</div></div>
</div>
</body></html>"#,
        cpu_usage, mem_usage, sys.cpus().len(),
        sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
    );

    Html(html)
}

/// Get active alerts (placeholder)
pub async fn get_alerts() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "alerts": [],
        "total": 0,
    }))
}

/// Get performance analysis
pub async fn get_performance_analysis(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let models = state.models.read().await;
    let jobs = state.jobs.read().await;

    let total_models = models.len();
    let total_jobs = jobs.len();
    let completed_jobs = jobs.values().filter(|j| matches!(j.status, JobStatus::Completed { .. })).count();
    let failed_jobs = jobs.values().filter(|j| matches!(j.status, JobStatus::Failed { .. })).count();

    Json(serde_json::json!({
        "models_count": total_models,
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
            "error_rate": if total_jobs > 0 { failed_jobs as f64 / total_jobs as f64 } else { 0.0 },
        },
        "throughput": {
            "description": "Requests per second (placeholder)",
            "value": 0.0,
        },
    }))
}
