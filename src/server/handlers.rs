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
use tracing::{info, warn, error, debug};

use calamine;

use crate::preprocessing::{DataPreprocessor, PreprocessingConfig, ScalerType, ImputeStrategy};
use crate::training::{TrainEngine, TrainingConfig, TaskType, ModelType};
use crate::inference::{InferenceEngine, InferenceConfig};

use super::error::{Result, ServerError};
use super::state::{AppState, JobStatus, ModelInfo, TrainingJob};

fn parse_task_type(s: &str) -> std::result::Result<TaskType, ServerError> {
    match s {
        "classification" => Ok(TaskType::BinaryClassification),
        "multiclass" => Ok(TaskType::MultiClassification),
        "regression" => Ok(TaskType::Regression),
        _ => Err(ServerError::BadRequest(format!("Invalid task type: '{}'. Expected: classification, multiclass, regression", s))),
    }
}

fn parse_model_type(s: &str) -> std::result::Result<ModelType, ServerError> {
    match s {
        "linear" | "linear_regression" => Ok(ModelType::LinearRegression),
        "logistic" | "logistic_regression" => Ok(ModelType::LogisticRegression),
        "decision_tree" => Ok(ModelType::DecisionTree),
        "random_forest" => Ok(ModelType::RandomForest),
        "gradient_boosting" => Ok(ModelType::GradientBoosting),
        "knn" => Ok(ModelType::KNN),
        "naive_bayes" => Ok(ModelType::NaiveBayes),
        "svm" => Ok(ModelType::SVM),
        "neural_network" => Ok(ModelType::NeuralNetwork),
        _ => Err(ServerError::BadRequest(format!(
            "Invalid model type: '{}'. Expected: linear_regression, logistic_regression, decision_tree, random_forest, gradient_boosting, knn, naive_bayes, svm, neural_network",
            s
        ))),
    }
}

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
        } else if file_name.ends_with(".xlsx") || file_name.ends_with(".xls") {
            read_excel_bytes(&data)?
        } else {
            return Err(ServerError::BadRequest(
                "Unsupported file format. Use CSV, JSON, Parquet, or Excel (.xlsx/.xls).".to_string()
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

    info!(dataset = %name, dataset_id = %dataset_id, rows = df.height(), columns = df.width(), "Sample dataset loaded");

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

    let task_type = parse_task_type(&request.task_type)?;
    let model_type = parse_model_type(&request.model_type)?;

    // Create job ID
    let job_id = AppState::generate_id();
    let job_id_for_response = job_id.clone();
    
    info!(
        job_id = %job_id,
        model = %request.model_type,
        task = %request.task_type,
        target = %request.target_column,
        "Training job created"
    );

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
    let model_type_name = request.model_type.clone();
    let task_type_name = request.task_type.clone();
    let models_dir = state.config.models_dir.clone();
    
    tokio::spawn(async move {
        let start = std::time::Instant::now();
        let result = run_training_job(
            &df, 
            &target_col, 
            task_type, 
            model_type,
        ).await;

        match result {
            Ok((metrics, engine)) => {
                let elapsed = start.elapsed();
                info!(
                    job_id = %job_id,
                    model = %model_type_name,
                    elapsed_secs = elapsed.as_secs_f64(),
                    "Training job completed, saving model"
                );

                // Save model to disk
                let model_path = format!("{}/model_{}_{}.json", models_dir, model_type_name, job_id);
                let model_id = AppState::generate_id();

                if let Err(e) = engine.save(&model_path) {
                    error!(job_id = %job_id, error = %e, "Failed to save model to disk");
                } else {
                    info!(job_id = %job_id, model_id = %model_id, path = %model_path, "Model saved to disk");
                }

                // Register model in state
                let model_info = ModelInfo {
                    id: model_id.clone(),
                    name: format!("{} (job {})", model_type_name, job_id),
                    task_type: task_type_name,
                    metrics: metrics.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    path: std::path::PathBuf::from(&model_path),
                };
                state_clone.models.write().await.insert(model_id, model_info);

                // Update job status
                let mut jobs = state_clone.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Completed { metrics };
                    job.model_path = Some(std::path::PathBuf::from(model_path));
                }
            }
            Err(e) => {
                error!(job_id = %job_id, error = %e, "Training job failed");
                let mut jobs = state_clone.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Failed { error: e.to_string() };
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
) -> anyhow::Result<(serde_json::Value, TrainEngine)> {
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

    Ok((result, engine))
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

    let task_type = parse_task_type(&request.task_type)?;

    info!(
        task = %request.task_type,
        target = %request.target_column,
        model_count = request.models.len(),
        "Starting model comparison"
    );

    let mut results = Vec::new();

    for model_name in &request.models {
        let model_type = match parse_model_type(model_name) {
            Ok(mt) => mt,
            Err(_) => {
                warn!(model = %model_name, "Skipping unknown model type in comparison");
                continue;
            }
        };

        match run_training_job(&df, &request.target_column, task_type.clone(), model_type).await {
            Ok((metrics, _engine)) => {
                info!(model = %model_name, "Comparison model trained successfully");
                results.push(serde_json::json!({
                    "model": model_name,
                    "metrics": metrics,
                }));
            }
            Err(e) => {
                error!(model = %model_name, error = %e, "Comparison model training failed");
                results.push(serde_json::json!({
                    "model": model_name,
                    "error": e.to_string(),
                }));
            }
        }
    }

    info!(total = results.len(), "Model comparison completed");

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

    let model_count = models.len();
    debug!(model_count = model_count, requested_id = ?request.model_id, "Predict request received");

    // Find the model: use provided model_id or most recently created
    let model_info = if let Some(ref id) = request.model_id {
        models.get(id)
            .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", id)))?
    } else {
        models.values()
            .max_by_key(|m| &m.created_at)
            .ok_or_else(|| ServerError::NotFound("No trained models available".to_string()))?
    };

    let model_id = model_info.id.clone();
    let model_path = model_info.path.to_str()
        .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?
        .to_string();
    drop(models);

    // Parse features from data field
    let features: Vec<Vec<f64>> = serde_json::from_value(request.data)
        .map_err(|e| ServerError::BadRequest(format!("Invalid features format, expected array of arrays of f64: {}", e)))?;

    if features.is_empty() {
        return Err(ServerError::BadRequest("Features array is empty".to_string()));
    }

    // Get or load the model from registry (cached)
    let engine = state.model_registry.get_or_load(&model_id, &model_path)
        .await
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

    let stats = engine.stats();

    Ok(Json(serde_json::json!({
        "success": true,
        "predictions": predictions.to_vec(),
        "latency_ms": stats.avg_latency_ms,
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

    let model_id = model_info.id.clone();
    let model_path = model_info.path.to_str()
        .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?
        .to_string();
    drop(models);

    if request.data.is_empty() {
        return Err(ServerError::BadRequest("Data array is empty".to_string()));
    }

    // Get or load the model from registry (cached)
    let engine = state.model_registry.get_or_load(&model_id, &model_path)
        .await
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

    let stats = engine.stats();

    Ok(Json(serde_json::json!({
        "success": true,
        "predictions": predictions.to_vec(),
        "count": predictions.len(),
        "avg_latency_ms": stats.avg_latency_ms,
    })))
}

/// Predict probabilities for classification models
pub async fn predict_proba(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<serde_json::Value>> {
    let models = state.models.read().await;

    let model_info = if let Some(ref id) = request.model_id {
        models.get(id)
            .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", id)))?
    } else {
        models.values()
            .max_by_key(|m| &m.created_at)
            .ok_or_else(|| ServerError::NotFound("No trained models available".to_string()))?
    };

    let model_id = model_info.id.clone();
    let model_path = model_info.path.to_str()
        .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?
        .to_string();
    drop(models);

    let features: Vec<Vec<f64>> = serde_json::from_value(request.data)
        .map_err(|e| ServerError::BadRequest(format!("Invalid features format: {}", e)))?;

    if features.is_empty() {
        return Err(ServerError::BadRequest("Features array is empty".to_string()));
    }

    let engine = state.model_registry.get_or_load(&model_id, &model_path)
        .await
        .map_err(|e| ServerError::Internal(format!("Failed to load model: {}", e)))?;

    let n_rows = features.len();
    let n_cols = features[0].len();
    let flat: Vec<f64> = features.into_iter().flatten().collect();
    let x = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| ServerError::BadRequest(format!("Invalid feature dimensions: {}", e)))?;

    let proba = engine.predict_proba_array(&x)
        .map_err(|e| ServerError::Internal(format!("Probability prediction failed: {}", e)))?;

    // Convert Array2 to Vec<Vec<f64>> for JSON
    let proba_vec: Vec<Vec<f64>> = proba.rows().into_iter()
        .map(|row| row.to_vec())
        .collect();

    Ok(Json(serde_json::json!({
        "success": true,
        "probabilities": proba_vec,
        "count": proba.nrows(),
    })))
}

/// Get inference statistics for a model
pub async fn get_inference_stats(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>> {
    match state.model_registry.get_model_stats(&model_id).await {
        Some(stats) => Ok(Json(serde_json::json!({
            "success": true,
            "model_id": model_id,
            "stats": {
                "total_predictions": stats.total_predictions,
                "avg_latency_ms": stats.avg_latency_ms,
                "p50_latency_ms": stats.p50_latency_ms,
                "p95_latency_ms": stats.p95_latency_ms,
                "p99_latency_ms": stats.p99_latency_ms,
                "throughput_per_sec": stats.throughput_per_sec,
                "error_count": stats.error_count,
            }
        }))),
        None => Err(ServerError::NotFound(format!("Model {} not cached", model_id))),
    }
}

/// Evict a model from the inference cache
pub async fn evict_model_cache(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>> {
    state.model_registry.evict(&model_id).await;
    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("Model {} evicted from cache", model_id),
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

pub async fn serve_index() -> impl IntoResponse {
    // Try multiple paths to find index.html
    let mut paths = vec![
        "kolosal-web/static/index.html".to_string(),
        format!("{}/kolosal-web/static/index.html", env!("CARGO_MANIFEST_DIR")),
    ];

    // Also check STATIC_DIR env var
    if let Ok(static_dir) = std::env::var("STATIC_DIR") {
        paths.insert(0, format!("{}/index.html", static_dir));
    }

    let html = paths.iter()
        .find_map(|path| std::fs::read_to_string(path).ok())
        .unwrap_or_else(|| EMBEDDED_INDEX_HTML.to_string());

    (
        [
            ("content-type", "text/html; charset=utf-8"),
            ("cache-control", "no-cache, no-store, must-revalidate"),
            ("pragma", "no-cache"),
            ("expires", "0"),
        ],
        html,
    )
}

const EMBEDDED_INDEX_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kolosal AutoML</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box;font-family:system-ui,-apple-system,sans-serif}
        body{background:#f8f9f9;color:#0d0e0f;min-height:100vh}
        header{background:#fff;border-bottom:1px solid #dde1e3;padding:1rem 1.5rem;position:sticky;top:0;z-index:15}
        .flex{display:flex}.items-center{align-items:center}.justify-between{justify-content:space-between}
        nav{background:#fff;padding:0 1.5rem;border-bottom:1px solid #e4e7e9}
        .nav-tab{display:inline-flex;align-items:center;gap:6px;height:40px;padding:0 12px;font-size:14px;font-weight:500;color:#6a6f73;background:none;border:none;border-bottom:2px solid transparent;cursor:pointer}
        .nav-tab:hover{color:#0d0e0f;background:#f1f3f4}.nav-tab.active{color:#0d0e0f;border-bottom-color:#0d0e0f}
        main{max-width:1080px;margin:0 auto;padding:24px 20px}
        .card{background:#fff;border:1px solid #e4e7e9;border-radius:12px;padding:24px;margin-bottom:16px}
        .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
        .btn{display:inline-flex;align-items:center;gap:6px;height:36px;padding:0 14px;font-size:14px;font-weight:500;border-radius:10px;border:none;cursor:pointer;background:#0d0e0f;color:#fff}
        .btn:disabled{opacity:.5;cursor:not-allowed}
        .badge{display:inline-flex;align-items:center;height:24px;padding:0 8px;font-size:12px;font-weight:500;border-radius:6px}
        .badge-ok{color:#2e9632;background:#f3fbf4}.badge-err{color:#cc2727;background:#fff3f3}
        .mono{font-family:"Geist Mono",monospace}
        .tab-panel{display:none}.tab-panel.active{display:block}
        .empty{text-align:center;padding:40px;color:#9c9fa1}
        select{height:36px;padding:0 14px;border:1px solid #dde1e3;border-radius:10px;width:100%;font-size:14px;appearance:none;background:#fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%236A6F73' viewBox='0 0 24 24'%3E%3Cpath d='M12 16l-6-6h12z'/%3E%3C/svg%3E") no-repeat right 12px center;padding-right:36px}
        .pill{display:inline-flex;align-items:center;height:28px;padding:0 10px;font-size:12px;font-weight:500;border-radius:999px;border:none;cursor:pointer;background:#f0f6fe;color:#0052c4;margin:0 4px 4px 0}
        .pill:hover{background:#b4d2fc}
        .toast-box{position:fixed;bottom:20px;right:20px;z-index:100}
        .toast{padding:10px 16px;border-radius:10px;font-size:14px;font-weight:500;color:#fff;background:#0d0e0f;margin-top:8px;box-shadow:0 4px 12px rgba(0,0,0,.15)}
        .toast-ok{background:#3abc3f}.toast-err{background:#ff3131}
        .stat{font-size:32px;line-height:44px;font-family:"Geist Mono",monospace;font-weight:500}
        .stat-info{color:#0066f5}.stat-ok{color:#3abc3f}
        .progress{width:100%;height:6px;background:#e4e7e9;border-radius:999px;overflow:hidden}
        .progress-fill{height:100%;background:#0066f5;border-radius:999px;transition:width .3s}
        table{width:100%;border-collapse:collapse}th{padding:8px 12px;text-align:left;font-size:12px;font-weight:500;color:#6a6f73;background:#f8f9f9;border-bottom:1px solid #e4e7e9}
        td{padding:8px 12px;font-size:14px;border-bottom:1px solid #ebedee}
        textarea{width:100%;min-height:120px;padding:8px 12px;border:1px solid #dde1e3;border-radius:12px;font-size:14px;resize:none;font-family:inherit}
        pre.code{background:#f8f9f9;border:1px solid #e4e7e9;border-radius:10px;padding:16px;font-size:13px;font-family:"Geist Mono",monospace;overflow:auto}
        label.form{display:block;font-size:14px;font-weight:500;margin-bottom:6px}
        .form-stack>*+*{margin-top:16px}
    </style>
</head>
<body style="background:#f8f9f9;color:#0d0e0f;font-family:system-ui,-apple-system,sans-serif;margin:0;padding:0;min-height:100vh;display:block;visibility:visible;opacity:1">
    <header>
        <div class="flex items-center justify-between">
            <div class="flex items-center" style="gap:10px"><strong class="mono" style="font-size:20px">Kolosal AutoML</strong><span style="color:#6a6f73;font-size:14px">Rust-Powered</span></div>
            <div class="flex items-center" style="gap:8px"><span id="e-badge" class="badge badge-ok">healthy</span><span id="e-dot" style="width:8px;height:8px;border-radius:50%;background:#3abc3f"></span></div>
        </div>
    </header>
    <nav>
        <button class="nav-tab active" data-tab="data" onclick="eTab('data')">Data</button>
        <button class="nav-tab" data-tab="config" onclick="eTab('config')">Config</button>
        <button class="nav-tab" data-tab="train" onclick="eTab('train')">Train</button>
        <button class="nav-tab" data-tab="monitor" onclick="eTab('monitor')">Monitor</button>
    </nav>
    <main>
        <div id="et-data" class="tab-panel active">
            <div class="grid-2">
                <div class="card">
                    <h2 style="font-size:16px;font-weight:500;margin-bottom:16px">Sample Datasets</h2>
                    <div id="e-pills"></div>
                </div>
                <div class="card">
                    <h2 style="font-size:16px;font-weight:500;margin-bottom:16px">Data Info</h2>
                    <div id="e-info" class="empty">No data loaded</div>
                </div>
            </div>
        </div>
        <div id="et-config" class="tab-panel">
            <div class="card form-stack" style="max-width:500px">
                <div><label class="form">Target Column</label><select id="e-target"><option value="">Select...</option></select></div>
                <div><label class="form">Task Type</label><select id="e-task"><option value="classification">Classification</option><option value="regression">Regression</option></select></div>
                <div><label class="form">Model</label><select id="e-model"><option value="random_forest">Random Forest</option><option value="decision_tree">Decision Tree</option><option value="logistic_regression">Logistic Regression</option></select></div>
            </div>
        </div>
        <div id="et-train" class="tab-panel">
            <div class="card">
                <div class="flex items-center justify-between" style="margin-bottom:16px">
                    <h2 style="font-size:16px;font-weight:500">Training</h2>
                    <button id="e-train-btn" class="btn" onclick="eTrain()" disabled>Train</button>
                </div>
                <div id="e-result"></div>
            </div>
        </div>
        <div id="et-monitor" class="tab-panel">
            <div class="grid-2">
                <div class="card"><p style="font-size:12px;color:#6a6f73;margin-bottom:8px">CPU Usage</p><p id="e-cpu" class="stat stat-info">0.0%</p></div>
                <div class="card"><p style="font-size:12px;color:#6a6f73;margin-bottom:8px">Memory</p><p id="e-mem" class="stat stat-ok">0.0%</p><p id="e-mem2" style="font-size:12px;color:#9c9fa1;margin-top:4px"></p></div>
            </div>
        </div>
    </main>
    <div id="e-toasts" class="toast-box"></div>
    <script>
    var eData=null,eTraining=false;
    function $(i){return document.getElementById(i)}
    function eNotify(m,t){var d=document.createElement('div');d.className='toast'+(t==='ok'?' toast-ok':t==='err'?' toast-err':'');d.textContent=m;$('e-toasts').appendChild(d);setTimeout(function(){d.remove()},3000)}
    function eTab(n){document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active')});document.querySelectorAll('.nav-tab').forEach(function(t){t.classList.remove('active')});$('et-'+n).classList.add('active');document.querySelector('[data-tab="'+n+'"]').classList.add('active')}
    function eSys(){fetch('/api/system/status').then(function(r){return r.json()}).then(function(d){var s=d.system||{};$('e-badge').textContent=d.status||'error';$('e-badge').className='badge '+(d.status==='healthy'?'badge-ok':'badge-err');$('e-dot').style.background=d.status==='healthy'?'#3abc3f':'#ff3131';$('e-cpu').textContent=(s.cpu_usage||0).toFixed(1)+'%';$('e-mem').textContent=(s.memory_usage_percent||0).toFixed(1)+'%';$('e-mem2').textContent=(s.used_memory_gb||0).toFixed(1)+' / '+(s.total_memory_gb||0).toFixed(1)+' GB'}).catch(function(){})}
    function eLoad(n){fetch('/api/data/sample/'+n).then(function(r){return r.json()}).then(function(d){if(d.success){eData=d;$('e-info').innerHTML='<div class="grid-2"><div><span class="stat stat-info">'+d.rows+'</span><p style="font-size:12px;color:#6a6f73">Rows</p></div><div><span class="stat stat-info">'+d.columns+'</span><p style="font-size:12px;color:#6a6f73">Columns</p></div></div>';var sel=$('e-target');sel.innerHTML='<option value="">Select...</option>';(d.column_names||[]).forEach(function(c){var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o)});$('e-train-btn').disabled=false;eNotify('Loaded '+n+'!','ok')}}).catch(function(e){eNotify('Failed: '+e.message,'err')})}
    function eTrain(){if(!eData||eTraining)return;eTraining=true;$('e-train-btn').disabled=true;$('e-result').innerHTML='<p>Training...</p>';var cfg={target_column:$('e-target').value,task_type:$('e-task').value,model_type:$('e-model').value};fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)}).then(function(r){return r.json()}).then(function(d){if(d.job_id){ePoll(d.job_id)}}).catch(function(e){$('e-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';eTraining=false;$('e-train-btn').disabled=false})}
    function ePoll(id){fetch('/api/train/status/'+id).then(function(r){return r.json()}).then(function(d){var s=d.status;if(s.Completed){var m=s.Completed.metrics||{};var h='<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';Object.keys(m).forEach(function(k){h+='<tr><td>'+k+'</td><td class="mono">'+(typeof m[k]==='number'?m[k].toFixed(4):m[k])+'</td></tr>'});h+='</tbody></table>';$('e-result').innerHTML=h;eTraining=false;$('e-train-btn').disabled=false;eNotify('Training complete!','ok')}else if(s.Failed){$('e-result').innerHTML='<p style="color:#ff3131">'+(s.Failed.error||'Failed')+'</p>';eTraining=false;$('e-train-btn').disabled=false}else{var p=s.Running?s.Running.progress||0:0;$('e-result').innerHTML='<div class="progress"><div class="progress-fill" style="width:'+(p*100)+'%"></div></div><p style="font-size:12px;color:#6a6f73;margin-top:8px">'+(s.Running?s.Running.message||'':'')+'</p>';setTimeout(function(){ePoll(id)},1000)}}).catch(function(){setTimeout(function(){ePoll(id)},2000)})}
    document.addEventListener('DOMContentLoaded',function(){['iris','diabetes','boston','wine','breast_cancer'].forEach(function(n){var b=document.createElement('button');b.className='pill';b.textContent=n.charAt(0).toUpperCase()+n.slice(1).replace('_',' ');b.onclick=function(){eLoad(n)};$('e-pills').appendChild(b)});eSys();setInterval(eSys,5000)});
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
// Excel Reader Helper
// ============================================================================

/// Read Excel (.xlsx/.xls) bytes into a Polars DataFrame
fn read_excel_bytes(data: &[u8]) -> Result<DataFrame> {
    use calamine::{Reader, open_workbook_auto_from_rs};

    let cursor = Cursor::new(data);
    let mut workbook = open_workbook_auto_from_rs(cursor)
        .map_err(|e| ServerError::BadRequest(format!("Failed to read Excel file: {}", e)))?;

    let sheet_names = workbook.sheet_names().to_vec();
    let first_sheet = sheet_names.first()
        .ok_or_else(|| ServerError::BadRequest("Excel file has no sheets".to_string()))?;

    let range = workbook.worksheet_range(first_sheet)
        .map_err(|e| ServerError::BadRequest(format!("Failed to read sheet: {}", e)))?;

    let (rows, cols) = range.get_size();
    if rows < 2 || cols == 0 {
        return Err(ServerError::BadRequest("Excel sheet is empty or has no data rows".to_string()));
    }

    // First row as headers
    let headers: Vec<String> = (0..cols).map(|c| {
        range.get((0, c))
            .map(|v| format!("{}", v))
            .unwrap_or_else(|| format!("col_{}", c))
    }).collect();

    // Build columns as f64 or string
    let mut series_vec: Vec<Column> = Vec::new();
    for col_idx in 0..cols {
        let mut float_values: Vec<Option<f64>> = Vec::new();
        let mut all_numeric = true;

        for row_idx in 1..rows {
            match range.get((row_idx, col_idx)) {
                Some(calamine::Data::Float(v)) => float_values.push(Some(*v)),
                Some(calamine::Data::Int(v)) => float_values.push(Some(*v as f64)),
                Some(calamine::Data::Empty) => float_values.push(None),
                _ => { all_numeric = false; break; }
            }
        }

        if all_numeric {
            let series = Series::new(headers[col_idx].clone().into(), &float_values);
            series_vec.push(series.into());
        } else {
            let str_values: Vec<Option<String>> = (1..rows).map(|row_idx| {
                range.get((row_idx, col_idx)).map(|v| format!("{}", v))
            }).collect();
            let series = Series::new(headers[col_idx].clone().into(), &str_values);
            series_vec.push(series.into());
        }
    }

    Ok(DataFrame::new(series_vec)?)
}

// ============================================================================
// Data Auto-Clean Handler
// ============================================================================

#[derive(Deserialize)]
pub struct CleanRequest {
    /// Remove duplicate rows (default: true)
    remove_duplicates: Option<bool>,
    /// Drop columns with more than this % of nulls (default: 80.0)
    null_threshold_percent: Option<f64>,
    /// Fill numeric nulls with mean (default: true)
    fill_numeric_nulls: Option<bool>,
    /// Fill string nulls with mode (default: true)
    fill_string_nulls: Option<bool>,
    /// Trim whitespace from string columns (default: true)
    trim_whitespace: Option<bool>,
}

/// Auto-clean the currently loaded dataset
pub async fn auto_clean_data(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CleanRequest>,
) -> Result<Json<serde_json::Value>> {
    let mut data = state.current_data.write().await;

    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
        .clone();

    let original_rows = df.height();
    let original_cols = df.width();
    let mut cleaned = df.clone();
    let mut actions: Vec<String> = Vec::new();

    info!(rows = original_rows, cols = original_cols, "Starting auto-clean");

    // 1. Drop columns with too many nulls
    let null_threshold = request.null_threshold_percent.unwrap_or(80.0) / 100.0;
    let mut drop_cols: Vec<String> = Vec::new();
    for col in cleaned.get_columns() {
        let null_ratio = col.null_count() as f64 / cleaned.height().max(1) as f64;
        if null_ratio > null_threshold {
            drop_cols.push(col.name().to_string());
        }
    }
    if !drop_cols.is_empty() {
        cleaned = cleaned.drop_many(drop_cols.iter().map(|s| s.as_str()));
        actions.push(format!("Dropped {} columns with >{:.0}% nulls: {:?}", drop_cols.len(), null_threshold * 100.0, drop_cols));
    }

    // 2. Remove duplicate rows
    if request.remove_duplicates.unwrap_or(true) {
        let before = cleaned.height();
        let no_subset: Option<&[String]> = None;
        cleaned = cleaned.unique_stable(no_subset, polars::frame::UniqueKeepStrategy::First, None)
            .map_err(|e| ServerError::Internal(format!("Dedup failed: {}", e)))?;
        let removed = before - cleaned.height();
        if removed > 0 {
            actions.push(format!("Removed {} duplicate rows", removed));
        }
    }

    // 3. Fill numeric nulls with column mean
    if request.fill_numeric_nulls.unwrap_or(true) {
        let mut filled_count = 0usize;
        let col_names: Vec<String> = cleaned.get_column_names().iter().map(|s| s.to_string()).collect();
        for col_name in &col_names {
            let col = cleaned.column(col_name).unwrap();
            if col.null_count() > 0 && col.dtype().is_float() {
                let filled = col.fill_null(polars::prelude::FillNullStrategy::Mean)
                    .map_err(|e| ServerError::Internal(format!("Fill null failed: {}", e)))?;
                let _ = cleaned.with_column(filled)
                    .map_err(|e| ServerError::Internal(format!("Replace column failed: {}", e)))?;
                filled_count += 1;
            }
        }
        if filled_count > 0 {
            actions.push(format!("Filled nulls with mean in {} numeric columns", filled_count));
        }
    }

    // 4. Fill string nulls with forward fill
    if request.fill_string_nulls.unwrap_or(true) {
        let mut filled_count = 0usize;
        let col_names: Vec<String> = cleaned.get_column_names().iter().map(|s| s.to_string()).collect();
        for col_name in &col_names {
            let col = cleaned.column(col_name).unwrap();
            if col.null_count() > 0 && col.dtype() == &DataType::String {
                let filled = col.fill_null(polars::prelude::FillNullStrategy::Forward(None))
                    .unwrap_or_else(|_| col.clone());
                let _ = cleaned.with_column(filled)
                    .map_err(|e| ServerError::Internal(format!("Replace column failed: {}", e)))?;
                filled_count += 1;
            }
        }
        if filled_count > 0 {
            actions.push(format!("Forward-filled nulls in {} string columns", filled_count));
        }
    }

    let final_rows = cleaned.height();
    let final_cols = cleaned.width();

    info!(
        original_rows, original_cols, final_rows, final_cols,
        actions_count = actions.len(),
        "Auto-clean completed"
    );

    *data = Some(cleaned);

    Ok(Json(serde_json::json!({
        "success": true,
        "original": { "rows": original_rows, "columns": original_cols },
        "cleaned": { "rows": final_rows, "columns": final_cols },
        "rows_removed": original_rows - final_rows,
        "columns_removed": original_cols - final_cols,
        "actions": actions,
    })))
}

// ============================================================================
// Kaggle Dataset Import Handler
// ============================================================================

#[derive(Deserialize)]
pub struct KaggleImportRequest {
    /// Kaggle dataset URL or slug (e.g., "username/dataset-name")
    dataset: String,
}

/// Import a dataset from Kaggle.
/// Returns a download link and instructions since direct download requires Kaggle API credentials.
pub async fn import_kaggle(
    Json(request): Json<KaggleImportRequest>,
) -> Result<Json<serde_json::Value>> {
    let dataset = request.dataset.trim().to_string();

    if dataset.is_empty() {
        return Err(ServerError::BadRequest("Dataset slug or URL is required".to_string()));
    }

    // Extract slug from full URL if provided
    let slug = if dataset.starts_with("http") {
        // Parse: https://www.kaggle.com/datasets/username/dataset-name
        dataset
            .split("/datasets/")
            .nth(1)
            .map(|s| s.trim_end_matches('/').to_string())
            .unwrap_or(dataset.clone())
    } else {
        dataset.clone()
    };

    // Validate slug format: "owner/dataset-name"
    let parts: Vec<&str> = slug.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err(ServerError::BadRequest(format!(
            "Invalid Kaggle dataset: '{}'. Expected format: 'username/dataset-name' or full Kaggle URL.",
            slug
        )));
    }

    let download_url = format!("https://www.kaggle.com/datasets/{}/download?datasetVersionNumber=1", slug);
    let browse_url = format!("https://www.kaggle.com/datasets/{}", slug);
    let api_command = format!("kaggle datasets download -d {}", slug);

    info!(slug = %slug, "Kaggle dataset import requested");

    Ok(Json(serde_json::json!({
        "success": true,
        "dataset": slug,
        "owner": parts[0],
        "name": parts[1],
        "download_url": download_url,
        "browse_url": browse_url,
        "api_command": api_command,
        "instructions": {
            "step1": "Install the Kaggle CLI: pip install kaggle",
            "step2": format!("Download: {}", api_command),
            "step3": "Unzip the downloaded file",
            "step4": "Upload the CSV/Excel file via /api/data/upload"
        },
        "message": format!(
            "Kaggle dataset '{}' found. Download it using the link or CLI command, then upload via /api/data/upload. Direct download requires Kaggle API credentials.",
            slug
        )
    })))
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
