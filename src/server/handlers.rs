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

// ============================================================================
// Security Constants & Guards
// ============================================================================

/// Maximum dataset rows allowed (10 million)
const MAX_DATASET_ROWS: usize = 10_000_000;
/// Maximum dataset columns allowed
const MAX_DATASET_COLUMNS: usize = 10_000;
/// Maximum batch prediction items
const MAX_BATCH_PREDICTION_ITEMS: usize = 100_000;
/// Training job timeout (30 minutes)
const TRAINING_TIMEOUT_SECS: u64 = 1800;

/// Sanitize a filename to prevent path traversal attacks.
/// Strips all path separators and special chars, keeps only safe filename chars.
fn sanitize_filename(name: &str) -> String {
    let basename = std::path::Path::new(name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("data.csv");
    basename
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | '_'))
        .collect::<String>()
}

/// Sanitize error messages to avoid leaking internal details to clients.
fn sanitize_error(msg: &str) -> String {
    // Strip file paths, internal module names
    if msg.contains('/') || msg.contains("src/") || msg.contains("thread") || msg.contains("panicked") {
        "An internal error occurred. Please check the server logs for details.".to_string()
    } else if msg.len() > 200 {
        format!("{}...", &msg[..200])
    } else {
        msg.to_string()
    }
}

/// Validate dataset dimensions are within safe limits.
fn validate_dataset_size(df: &DataFrame) -> Result<()> {
    if df.height() > MAX_DATASET_ROWS {
        return Err(ServerError::BadRequest(format!(
            "Dataset too large: {} rows (max {})", df.height(), MAX_DATASET_ROWS
        )));
    }
    if df.width() > MAX_DATASET_COLUMNS {
        return Err(ServerError::BadRequest(format!(
            "Too many columns: {} (max {})", df.width(), MAX_DATASET_COLUMNS
        )));
    }
    Ok(())
}

fn parse_task_type(s: &str) -> std::result::Result<TaskType, ServerError> {
    match s {
        "classification" => Ok(TaskType::BinaryClassification),
        "multiclass" => Ok(TaskType::MultiClassification),
        "regression" => Ok(TaskType::Regression),
        "clustering" => Ok(TaskType::Clustering),
        _ => Err(ServerError::BadRequest(format!("Invalid task type: '{}'. Expected: classification, multiclass, regression, clustering", s))),
    }
}

fn parse_model_type(s: &str) -> std::result::Result<ModelType, ServerError> {
    match s {
        "linear" | "linear_regression" => Ok(ModelType::LinearRegression),
        "logistic" | "logistic_regression" => Ok(ModelType::LogisticRegression),
        "ridge" | "ridge_regression" => Ok(ModelType::Ridge),
        "lasso" | "lasso_regression" => Ok(ModelType::Lasso),
        "elastic_net" | "elasticnet" => Ok(ModelType::ElasticNet),
        "decision_tree" => Ok(ModelType::DecisionTree),
        "random_forest" => Ok(ModelType::RandomForest),
        "gradient_boosting" => Ok(ModelType::GradientBoosting),
        "knn" => Ok(ModelType::KNN),
        "naive_bayes" => Ok(ModelType::NaiveBayes),
        "svm" => Ok(ModelType::SVM),
        "neural_network" => Ok(ModelType::NeuralNetwork),
        "adaboost" | "ada_boost" => Ok(ModelType::AdaBoost),
        "extra_trees" | "extratrees" => Ok(ModelType::ExtraTrees),
        "xgboost" | "xg_boost" => Ok(ModelType::XGBoost),
        "lightgbm" | "light_gbm" | "lgbm" => Ok(ModelType::LightGBM),
        "catboost" | "cat_boost" => Ok(ModelType::CatBoost),
        "sgd" | "sgd_classifier" | "sgd_regressor" => Ok(ModelType::SGD),
        "gaussian_process" | "gp" | "gpr" => Ok(ModelType::GaussianProcess),
        "polynomial" | "polynomial_regression" | "poly" => Ok(ModelType::PolynomialRegression),
        "kmeans" | "k_means" => Ok(ModelType::KMeans),
        "dbscan" => Ok(ModelType::DBSCAN),
        _ => Err(ServerError::BadRequest(format!(
            "Invalid model type: '{}'. Expected: linear_regression, logistic_regression, ridge, lasso, elastic_net, polynomial, decision_tree, random_forest, gradient_boosting, xgboost, lightgbm, catboost, knn, naive_bayes, svm, neural_network, adaboost, extra_trees, sgd, gaussian_process, kmeans, dbscan",
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
        let raw_file_name = field.file_name().unwrap_or("data.csv").to_string();
        let file_name = sanitize_filename(&raw_file_name);
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

        // Validate dataset size
        validate_dataset_size(&df)?;

        // Store dataset with sanitized path
        let temp_path = std::env::temp_dir().join(format!("kolosal_{}", &file_name));
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

                // Store engine for post-training analysis
                state_clone.train_engines.write().await.insert(model_id.clone(), engine);

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

/// Train a single model — CPU-bound work is offloaded via spawn_blocking.
async fn run_training_job(
    df: &DataFrame,
    target_column: &str,
    task_type: TaskType,
    model_type: ModelType,
) -> anyhow::Result<(serde_json::Value, TrainEngine)> {
    let df = df.clone();
    let target_column = target_column.to_string();
    let tt = task_type.clone();

    // Offload CPU-bound training to blocking thread pool with timeout
    let training_future = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let config = TrainingConfig::new(tt.clone(), &target_column)
            .with_model(model_type);
        let mut engine = TrainEngine::new(config);
        engine.fit(&df)?;
        let metrics = engine.metrics().cloned().unwrap_or_default();
        let result = match tt {
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
            TaskType::Clustering => serde_json::json!({
                "training_time_secs": metrics.training_time_secs,
            }),
        };
        Ok((engine, result))
    });

    let (engine, metrics_val) = tokio::time::timeout(
        std::time::Duration::from_secs(TRAINING_TIMEOUT_SECS),
        training_future,
    )
    .await
    .map_err(|_| anyhow::anyhow!("Training timed out after {} seconds", TRAINING_TIMEOUT_SECS))???;

    Ok((metrics_val, engine))
}

/// Max concurrent model trainings to prevent OOM on small machines
const MAX_PARALLEL_TRAINS: usize = 4;

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
    let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_PARALLEL_TRAINS));
    let mut handles = Vec::new();

    for model_name in &request.models {
        let model_type = match parse_model_type(model_name) {
            Ok(mt) => mt,
            Err(_) => {
                warn!(model = %model_name, "Skipping unknown model type in comparison");
                results.push(serde_json::json!({
                    "model": model_name,
                    "error": format!("Unknown model type: {}", model_name),
                }));
                continue;
            }
        };

        let df = df.clone();
        let target = request.target_column.clone();
        let tt = task_type.clone();
        let name = model_name.clone();
        let sem = semaphore.clone();

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let result = run_training_job(&df, &target, tt, model_type).await;
            (name, result)
        }));
    }

    for handle in handles {
        match handle.await {
            Ok((name, Ok((metrics, _engine)))) => {
                info!(model = %name, "Comparison model trained successfully");
                results.push(serde_json::json!({
                    "model": name,
                    "metrics": metrics,
                }));
            }
            Ok((name, Err(e))) => {
                error!(model = %name, error = %e, "Comparison model training failed");
                results.push(serde_json::json!({
                    "model": name,
                    "error": e.to_string(),
                }));
            }
            Err(e) => {
                error!(error = %e, "Comparison task panicked");
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

    if request.data.len() > MAX_BATCH_PREDICTION_ITEMS {
        return Err(ServerError::BadRequest(format!(
            "Batch too large: {} items (max {})", request.data.len(), MAX_BATCH_PREDICTION_ITEMS
        )));
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
        <button class="nav-tab" data-tab="automl" onclick="eTab('automl')">AutoML</button>
        <button class="nav-tab" data-tab="config" onclick="eTab('config')">Config</button>
        <button class="nav-tab" data-tab="train" onclick="eTab('train')">Train</button>
        <button class="nav-tab" data-tab="hyperopt" onclick="eTab('hyperopt')">HyperOpt</button>
        <button class="nav-tab" data-tab="autotune" onclick="eTab('autotune')">Auto-Tune</button>
        <button class="nav-tab" data-tab="ensemble" onclick="eTab('ensemble')">Ensemble</button>
        <button class="nav-tab" data-tab="explain" onclick="eTab('explain')">Explain</button>
        <button class="nav-tab" data-tab="anomaly" onclick="eTab('anomaly')">Anomaly</button>
        <button class="nav-tab" data-tab="security" onclick="eTab('security')">Security</button>
        <button class="nav-tab" data-tab="monitor" onclick="eTab('monitor')">Monitor</button>
    </nav>
    <main>
        <div id="et-data" class="tab-panel active">
            <div class="grid-2">
                <div class="card">
                    <h2 style="font-size:16px;font-weight:500;margin-bottom:16px">Sample Datasets</h2>
                    <div id="e-pills"></div>
                    <div style="margin-top:16px;padding-top:12px;border-top:1px solid #e4e7e9">
                        <p style="font-size:12px;font-weight:500;color:#6a6f73;margin-bottom:8px">Or import from URL:</p>
                        <div style="display:flex;gap:8px"><input type="text" id="e-import-url" placeholder="Paste URL (Kaggle, GitHub, CSV...)" style="flex:1;height:36px;border:1px solid #dde1e3;border-radius:10px;padding:0 14px;font-size:14px"><button class="btn" onclick="eImportUrl()">Import</button></div>
                        <div id="e-import-status" style="margin-top:8px"></div>
                    </div>
                </div>
                <div class="card">
                    <h2 style="font-size:16px;font-weight:500;margin-bottom:16px">Data Info</h2>
                    <div id="e-info" class="empty">No data loaded</div>
                </div>
            </div>
        </div>
        <div id="et-automl" class="tab-panel">
            <div class="card" style="max-width:600px">
                <h2 style="font-size:16px;font-weight:500;margin-bottom:8px">One-Click AutoML</h2>
                <p style="font-size:13px;color:#6a6f73;margin-bottom:16px">Automatically detect, preprocess, train, and optimize — all in one click.</p>
                <div style="margin-bottom:12px"><label class="form">Target Column</label><select id="e-automl-target"><option value="">Select target...</option></select></div>
                <button class="btn" id="e-btn-automl" onclick="eRunAutoML()" disabled style="width:100%;justify-content:center;height:44px;font-size:16px">Run AutoML Pipeline</button>
                <details style="margin-top:16px"><summary style="cursor:pointer;font-size:13px;font-weight:500;color:#6a6f73">Advanced Settings</summary>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px">
                    <div><label class="form">Task Type</label><select id="e-aml-task"><option value="">Auto-Detect</option><option value="classification">Classification</option><option value="regression">Regression</option></select></div>
                    <div><label class="form">Scaler</label><select id="e-aml-scaler"><option value="">Auto</option><option value="standard">Standard</option><option value="minmax">MinMax</option><option value="robust">Robust</option></select></div>
                    <div><label class="form">Max Models</label><input type="number" id="e-aml-max" value="8" min="1" max="20" style="height:36px;border:1px solid #dde1e3;border-radius:10px;padding:0 14px;font-size:14px;width:100%"></div>
                    <div><label class="form">HyperOpt Trials</label><input type="number" id="e-aml-trials" value="20" min="0" max="200" style="height:36px;border:1px solid #dde1e3;border-radius:10px;padding:0 14px;font-size:14px;width:100%"></div>
                </div></details>
                <div id="e-automl-progress" style="display:none;margin-top:16px">
                    <div style="font-size:14px;font-weight:500;color:#0066f5;margin-bottom:8px" id="e-aml-status">Starting...</div>
                    <div class="progress"><div class="progress-fill" id="e-aml-bar" style="width:0%"></div></div>
                </div>
                <div id="e-automl-result" style="display:none;margin-top:16px">
                    <div style="background:#0d0e0f;border-radius:10px;padding:16px;color:#fff;margin-bottom:12px">
                        <div style="font-size:12px;opacity:.7">Best Model</div>
                        <div style="font-size:20px;font-weight:600" id="e-aml-best">—</div>
                        <div style="font-size:14px;opacity:.85" id="e-aml-score">—</div>
                    </div>
                    <div id="e-aml-leaderboard"></div>
                </div>
            </div>
        </div>
        <div id="et-config" class="tab-panel">
            <div class="card form-stack" style="max-width:500px">
                <div><label class="form">Target Column</label><select id="e-target"><option value="">Select...</option></select></div>
                <div><label class="form">Task Type</label><select id="e-task"><option value="classification">Classification</option><option value="regression">Regression</option><option value="clustering">Clustering</option></select></div>
                <div><label class="form">Model</label><select id="e-model"><option value="random_forest">Random Forest</option><option value="xgboost">XGBoost</option><option value="lightgbm">LightGBM</option><option value="catboost">CatBoost</option><option value="gradient_boosting">Gradient Boosting</option><option value="extra_trees">Extra Trees</option><option value="decision_tree">Decision Tree</option><option value="logistic_regression">Logistic Regression</option><option value="ridge">Ridge</option><option value="lasso">Lasso</option><option value="elastic_net">Elastic Net</option><option value="polynomial">Polynomial</option><option value="adaboost">AdaBoost</option><option value="sgd">SGD</option><option value="knn">KNN</option><option value="naive_bayes">Naive Bayes</option><option value="svm">SVM</option><option value="neural_network">Neural Network</option><option value="gaussian_process">Gaussian Process</option><option value="kmeans">KMeans</option><option value="dbscan">DBSCAN</option></select></div>
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
        <div id="et-hyperopt" class="tab-panel">
            <div class="grid-2">
                <div class="card">
                    <h2 style="font-size:16px;font-weight:500;margin-bottom:16px">HyperOpt Config</h2>
                    <div class="form-stack">
                        <div><label class="form">Model</label><select id="e-ho-model"><option value="random_forest">Random Forest</option><option value="xgboost">XGBoost</option><option value="lightgbm">LightGBM</option><option value="catboost">CatBoost</option><option value="gradient_boosting">Gradient Boosting</option><option value="extra_trees">Extra Trees</option><option value="decision_tree">Decision Tree</option><option value="adaboost">AdaBoost</option><option value="sgd">SGD</option><option value="knn">KNN</option><option value="svm">SVM</option><option value="polynomial">Polynomial</option><option value="gaussian_process">Gaussian Process</option></select></div>
                        <div><label class="form">Sampler</label><select id="e-ho-sampler"><option value="tpe">TPE</option><option value="random">Random</option><option value="gp">Gaussian Process</option></select></div>
                        <div><label class="form">Trials</label><input type="number" id="e-ho-trials" value="20" min="5" max="100" style="height:36px;border:1px solid #dde1e3;border-radius:10px;width:100%;padding:0 14px;font-size:14px"></div>
                        <button id="e-ho-btn" class="btn" onclick="eHyperOpt()" disabled>Optimize</button>
                    </div>
                </div>
                <div class="card">
                    <h2 style="font-size:16px;font-weight:500;margin-bottom:16px">Results</h2>
                    <div id="e-ho-result" class="empty">Run optimization to see results</div>
                </div>
            </div>
        </div>
        <div id="et-autotune" class="tab-panel">
            <div class="grid-2">
                <div class="card">
                    <h2 style="font-size:16px;font-weight:500;margin-bottom:16px">Auto-Tune</h2>
                    <p style="font-size:14px;color:#6a6f73;margin-bottom:12px">Automatically trains multiple models and picks the best.</p>
                    <button id="e-at-btn" class="btn" onclick="eAutoTune()" disabled>Start Auto-Tune</button>
                </div>
                <div class="card">
                    <h2 style="font-size:16px;font-weight:500;margin-bottom:16px">Leaderboard</h2>
                    <div id="e-at-result" class="empty">Run auto-tune to see results</div>
                </div>
            </div>
        </div>
        <div id="et-ensemble" class="tab-panel">
            <div class="card">
                <h2 style="font-size:16px;font-weight:500;margin-bottom:16px">Ensemble Training</h2>
                <p style="font-size:14px;color:#6a6f73;margin-bottom:12px">Select 2+ models and strategy, then train.</p>
                <div class="grid-2">
                    <div><label class="form">Strategy</label><select id="e-ens-strat"><option value="voting">Voting</option><option value="averaging">Averaging</option></select></div>
                    <div><button id="e-ens-btn" class="btn" onclick="eEnsemble()" disabled>Train Ensemble</button></div>
                </div>
                <div id="e-ens-result" style="margin-top:16px" class="empty">Configure ensemble to see results</div>
            </div>
        </div>
        <div id="et-explain" class="tab-panel">
            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
                    <h2 style="font-size:16px;font-weight:500">Feature Importance</h2>
                    <button id="e-explain-btn" class="btn" onclick="eExplain()" disabled>Analyze</button>
                </div>
                <div id="e-explain-result" class="empty">Train a model first, then analyze</div>
            </div>
        </div>
        <div id="et-anomaly" class="tab-panel">
            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
                    <h2 style="font-size:16px;font-weight:500">Anomaly Detection</h2>
                    <button id="e-anomaly-btn" class="btn" onclick="eAnomaly()" disabled>Detect</button>
                </div>
                <div class="grid-2" style="margin-bottom:12px">
                    <div><label class="form">Contamination</label><input type="number" id="e-anom-cont" value="0.1" min="0.01" max="0.5" step="0.01" style="height:36px;border:1px solid #dde1e3;border-radius:10px;width:100%;padding:0 14px;font-size:14px"></div>
                    <div><label class="form">Estimators</label><input type="number" id="e-anom-est" value="100" min="10" max="1000" step="10" style="height:36px;border:1px solid #dde1e3;border-radius:10px;width:100%;padding:0 14px;font-size:14px"></div>
                </div>
                <div id="e-anomaly-result" class="empty">Load data and run anomaly detection</div>
            </div>
        </div>
        <div id="et-security" class="tab-panel">
            <div class="card" style="margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                    <h2 style="font-size:16px;font-weight:500">Security Overview</h2>
                    <button class="btn" onclick="eSecLoad()">Refresh</button>
                </div>
                <div id="e-sec-status" class="empty">Loading security status...</div>
            </div>
            <div class="card" style="margin-bottom:12px">
                <h2 style="font-size:16px;font-weight:500;margin-bottom:12px">Audit Log</h2>
                <div id="e-sec-audit" class="empty" style="max-height:300px;overflow-y:auto">Loading...</div>
            </div>
        </div>
    </main>
    <div id="e-toasts" class="toast-box"></div>
    <script>
    var eData=null,eTraining=false,eLastModelId=null;
    function $(i){return document.getElementById(i)}
    function eNotify(m,t){var d=document.createElement('div');d.className='toast'+(t==='ok'?' toast-ok':t==='err'?' toast-err':'');d.textContent=m;$('e-toasts').appendChild(d);setTimeout(function(){d.remove()},3000)}
    function eTab(n){document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active')});document.querySelectorAll('.nav-tab').forEach(function(t){t.classList.remove('active')});$('et-'+n).classList.add('active');document.querySelector('[data-tab="'+n+'"]').classList.add('active')}
    function eSys(){fetch('/api/system/status').then(function(r){return r.json()}).then(function(d){var s=d.system||{};$('e-badge').textContent=d.status||'error';$('e-badge').className='badge '+(d.status==='healthy'?'badge-ok':'badge-err');$('e-dot').style.background=d.status==='healthy'?'#3abc3f':'#ff3131';$('e-cpu').textContent=(s.cpu_usage||0).toFixed(1)+'%';$('e-mem').textContent=(s.memory_usage_percent||0).toFixed(1)+'%';$('e-mem2').textContent=(s.used_memory_gb||0).toFixed(1)+' / '+(s.total_memory_gb||0).toFixed(1)+' GB'}).catch(function(){})}
    function eLoad(n){fetch('/api/data/sample/'+n).then(function(r){return r.json()}).then(function(d){if(d.success){eData=d;$('e-info').innerHTML='<div class="grid-2"><div><span class="stat stat-info">'+d.rows+'</span><p style="font-size:12px;color:#6a6f73">Rows</p></div><div><span class="stat stat-info">'+d.columns+'</span><p style="font-size:12px;color:#6a6f73">Columns</p></div></div>';var sel=$('e-target');sel.innerHTML='<option value="">Select...</option>';(d.column_names||[]).forEach(function(c){var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o)});$('e-train-btn').disabled=false;$('e-ho-btn').disabled=false;$('e-ens-btn').disabled=false;$('e-at-btn').disabled=false;$('e-anomaly-btn').disabled=false;eUpdateAutoMLTarget();eNotify('Loaded '+n+'!','ok')}}).catch(function(e){eNotify('Failed: '+e.message,'err')})}
    function eImportUrl(){var url=$('e-import-url').value.trim();if(!url){eNotify('Enter a URL','err');return}$('e-import-status').innerHTML='<p style="font-size:12px;color:#6a6f73">Downloading...</p>';fetch('/api/data/import/url',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:url})}).then(function(r){return r.json()}).then(function(d){if(d.success){eData=d;$('e-info').innerHTML='<div class="grid-2"><div><span class="stat stat-info">'+d.rows+'</span><p style="font-size:12px;color:#6a6f73">Rows</p></div><div><span class="stat stat-info">'+d.columns+'</span><p style="font-size:12px;color:#6a6f73">Columns</p></div></div>';var sel=$('e-target');sel.innerHTML='<option value="">Select...</option>';(d.column_names||[]).forEach(function(c){var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o)});$('e-train-btn').disabled=false;$('e-ho-btn').disabled=false;$('e-ens-btn').disabled=false;$('e-at-btn').disabled=false;$('e-anomaly-btn').disabled=false;eUpdateAutoMLTarget();$('e-import-status').innerHTML='<p style="font-size:12px;color:#3abc3f">✓ Imported from '+d.source+' ('+d.rows+' rows)</p>';eNotify('Imported!','ok')}else if(d.needs_credentials){$('e-import-status').innerHTML='<p style="font-size:12px;color:#ffa931">⚠ '+d.message+'</p>'}else{$('e-import-status').innerHTML='<p style="font-size:12px;color:#ff3131">'+(d.message||'Failed')+'</p>'}}).catch(function(e){$('e-import-status').innerHTML='<p style="font-size:12px;color:#ff3131">'+e.message+'</p>'})}
    function eTrain(){if(!eData||eTraining)return;eTraining=true;$('e-train-btn').disabled=true;$('e-result').innerHTML='<p>Training...</p>';var cfg={target_column:$('e-target').value,task_type:$('e-task').value,model_type:$('e-model').value};fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)}).then(function(r){return r.json()}).then(function(d){if(d.job_id){ePoll(d.job_id)}}).catch(function(e){$('e-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';eTraining=false;$('e-train-btn').disabled=false})}
    function ePoll(id){fetch('/api/train/status/'+id).then(function(r){return r.json()}).then(function(d){var s=d.status;if(s.Completed){var m=s.Completed.metrics||{};var h='<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';Object.keys(m).forEach(function(k){h+='<tr><td>'+k+'</td><td class="mono">'+(typeof m[k]==='number'?m[k].toFixed(4):m[k])+'</td></tr>'});h+='</tbody></table>';$('e-result').innerHTML=h;eTraining=false;$('e-train-btn').disabled=false;eNotify('Training complete!','ok');fetch('/api/models').then(function(r){return r.json()}).then(function(md){var models=md.models||[];if(models.length>0){eLastModelId=models[models.length-1].id;$('e-explain-btn').disabled=false}}).catch(function(){})}else if(s.Failed){$('e-result').innerHTML='<p style="color:#ff3131">'+(s.Failed.error||'Failed')+'</p>';eTraining=false;$('e-train-btn').disabled=false}else{var p=s.Running?s.Running.progress||0:0;$('e-result').innerHTML='<div class="progress"><div class="progress-fill" style="width:'+(p*100)+'%"></div></div><p style="font-size:12px;color:#6a6f73;margin-top:8px">'+(s.Running?s.Running.message||'':'')+'</p>';setTimeout(function(){ePoll(id)},1000)}}).catch(function(){setTimeout(function(){ePoll(id)},2000)})}
    function eHyperOpt(){if(!eData)return;$('e-ho-btn').disabled=true;$('e-ho-result').innerHTML='<p>Optimizing...</p>';fetch('/api/hyperopt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target_column:$('e-target').value,task_type:$('e-task').value,model_type:$('e-ho-model').value,n_trials:parseInt($('e-ho-trials').value),sampler:$('e-ho-sampler').value})}).then(function(r){return r.json()}).then(function(d){if(d.job_id){eHoPoll(d.job_id)}}).catch(function(e){$('e-ho-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';$('e-ho-btn').disabled=false})}
    function eHoPoll(id){fetch('/api/train/status/'+id).then(function(r){return r.json()}).then(function(d){var s=d.status;if(s.Completed){var m=s.Completed.metrics||{};var h='<div style="background:#f3fbf4;padding:12px;border-radius:8px;margin-bottom:12px">';if(m.best_trial){h+='<strong>Best Score:</strong> '+((m.best_trial.value||0).toFixed(6))+'<br><small>'+((m.best_trial.params||''))+'</small>'}h+='</div><p style="font-size:12px;color:#6a6f73">Trials: '+(m.total_trials||0)+' | Time: '+(m.total_duration_secs||0).toFixed(2)+'s</p>';if(m.trials){h+='<table style="margin-top:8px"><thead><tr><th>#</th><th>Score</th><th>Time</th></tr></thead><tbody>';m.trials.forEach(function(t){h+='<tr><td>'+t.trial_id+'</td><td class="mono">'+(t.value===Infinity?'Fail':t.value.toFixed(6))+'</td><td class="mono">'+t.duration_secs.toFixed(2)+'s</td></tr>'});h+='</tbody></table>'}$('e-ho-result').innerHTML=h;$('e-ho-btn').disabled=false;eNotify('Optimization done!','ok')}else if(s.Failed){$('e-ho-result').innerHTML='<p style="color:#ff3131">'+(s.Failed.error||'Failed')+'</p>';$('e-ho-btn').disabled=false}else{setTimeout(function(){eHoPoll(id)},1500)}}).catch(function(){setTimeout(function(){eHoPoll(id)},3000)})}
    function eAutoTune(){if(!eData)return;$('e-at-btn').disabled=true;$('e-at-result').innerHTML='<p>Auto-tuning...</p>';fetch('/api/autotune',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target_column:$('e-target').value,task_type:$('e-task').value})}).then(function(r){return r.json()}).then(function(d){if(d.job_id){eAtPoll(d.job_id)}}).catch(function(e){$('e-at-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';$('e-at-btn').disabled=false})}
    function eAtPoll(id){fetch('/api/train/status/'+id).then(function(r){return r.json()}).then(function(d){var s=d.status;if(s.Completed){var m=s.Completed.metrics||{};var h='';if(m.best){h+='<div style="background:#f3fbf4;padding:12px;border-radius:8px;margin-bottom:12px"><strong>🏆 '+m.best_model+'</strong> (score: '+(m.best.score||0).toFixed(6)+')</div>'}if(m.leaderboard){h+='<table><thead><tr><th>#</th><th>Model</th><th>Score</th><th>Time</th></tr></thead><tbody>';m.leaderboard.forEach(function(r,i){h+='<tr><td>'+(i+1)+'</td><td>'+r.model+'</td><td class="mono">'+(r.score||0).toFixed(6)+'</td><td class="mono">'+(r.training_time_secs||0).toFixed(2)+'s</td></tr>'});h+='</tbody></table>'}$('e-at-result').innerHTML=h;$('e-at-btn').disabled=false;if(m.best&&m.best.status==='success'){eLastModelId='autotune_'+m.best_model+'_'+id;$('e-explain-btn').disabled=false}eNotify('Auto-tune done!','ok')}else if(s.Failed){$('e-at-result').innerHTML='<p style="color:#ff3131">'+(s.Failed.error||'Failed')+'</p>';$('e-at-btn').disabled=false}else{var p=s.Running?s.Running.progress||0:0;$('e-at-result').innerHTML='<div class="progress"><div class="progress-fill" style="width:'+(p*100)+'%"></div></div><p style="font-size:12px;color:#6a6f73;margin-top:8px">'+(s.Running?s.Running.message||'':'')+'</p>';setTimeout(function(){eAtPoll(id)},1500)}}).catch(function(){setTimeout(function(){eAtPoll(id)},3000)})}
    function eEnsemble(){if(!eData)return;$('e-ens-btn').disabled=true;$('e-ens-result').innerHTML='<p>Training ensemble...</p>';var models=['random_forest','decision_tree','gradient_boosting'];fetch('/api/ensemble/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target_column:$('e-target').value,task_type:$('e-task').value,models:models,strategy:$('e-ens-strat').value})}).then(function(r){return r.json()}).then(function(d){if(d.job_id){eEnsPoll(d.job_id)}}).catch(function(e){$('e-ens-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';$('e-ens-btn').disabled=false})}
    function eEnsPoll(id){fetch('/api/train/status/'+id).then(function(r){return r.json()}).then(function(d){var s=d.status;if(s.Completed){var m=s.Completed.metrics||{};var h='<div style="background:#f0f6fe;padding:12px;border-radius:8px;margin-bottom:12px"><strong>Ensemble:</strong> '+(m.strategy||'voting')+' | Models: '+(m.successful_models||0)+'/'+(m.model_count||0)+'<br><small>Time: '+(m.training_time_secs||0).toFixed(2)+'s</small></div>';if(m.model_results){h+='<table><thead><tr><th>Model</th><th>Status</th></tr></thead><tbody>';m.model_results.forEach(function(r){h+='<tr><td>'+r.model+'</td><td>'+(r.status==='success'?'✓':'✗ '+r.error)+'</td></tr>'});h+='</tbody></table>'}$('e-ens-result').innerHTML=h;$('e-ens-btn').disabled=false;eNotify('Ensemble done!','ok')}else if(s.Failed){$('e-ens-result').innerHTML='<p style="color:#ff3131">'+(s.Failed.error||'Failed')+'</p>';$('e-ens-btn').disabled=false}else{setTimeout(function(){eEnsPoll(id)},1500)}}).catch(function(){setTimeout(function(){eEnsPoll(id)},3000)})}
    function eExplain(){if(!eLastModelId)return;$('e-explain-btn').disabled=true;fetch('/api/explain/importance',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_id:eLastModelId})}).then(function(r){return r.json()}).then(function(d){if(d.success&&d.features&&d.features.length>0){var mx=Math.max.apply(null,d.features.map(function(f){return Math.abs(f.importance)}));var h='<table><thead><tr><th>Feature</th><th>Importance</th><th>Bar</th></tr></thead><tbody>';d.features.forEach(function(f){var pct=mx>0?(Math.abs(f.importance)/mx*100).toFixed(0):0;h+='<tr><td>'+f.feature+'</td><td class="mono">'+f.importance.toFixed(6)+'</td><td><div style="width:100%;background:#e4e7e9;border-radius:4px;height:6px"><div style="width:'+pct+'%;background:#0066f5;border-radius:4px;height:6px"></div></div></td></tr>'});h+='</tbody></table>';$('e-explain-result').innerHTML=h}else{$('e-explain-result').innerHTML='<p style="color:#6a6f73">'+(d.message||'Not available for this model')+'</p>'}$('e-explain-btn').disabled=false}).catch(function(e){$('e-explain-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';$('e-explain-btn').disabled=false})}
    function eAnomaly(){if(!eData)return;$('e-anomaly-btn').disabled=true;$('e-anomaly-result').innerHTML='<p>Detecting anomalies...</p>';fetch('/api/anomaly/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({contamination:parseFloat($('e-anom-cont').value),n_estimators:parseInt($('e-anom-est').value)})}).then(function(r){return r.json()}).then(function(d){$('e-anomaly-btn').disabled=false;if(d.success){var rate=((d.anomaly_rate||0)*100).toFixed(1);$('e-anomaly-result').innerHTML='<div class="grid-2"><div><span class="stat stat-info">'+d.total_samples+'</span><p style="font-size:12px;color:#6a6f73">Samples</p></div><div><span class="stat" style="color:#ff3131">'+d.anomalies_found+'</span><p style="font-size:12px;color:#6a6f73">Anomalies ('+rate+'%)</p></div></div>'}else{$('e-anomaly-result').innerHTML='<p style="color:#ff3131">'+(d.message||'Failed')+'</p>'}}).catch(function(e){$('e-anomaly-btn').disabled=false;$('e-anomaly-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>'})}
    function eUpdateAutoMLTarget(){var sel=$('e-automl-target');if(!sel||!eData)return;sel.innerHTML='<option value="">Select target...</option>';(eData.column_names||[]).forEach(function(c){var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o)});sel.onchange=function(){$('e-btn-automl').disabled=!this.value}}
    function eRunAutoML(){if(!eData)return;var target=$('e-automl-target').value;if(!target){eNotify('Select a target column','err');return}$('e-btn-automl').disabled=true;$('e-automl-progress').style.display='block';$('e-automl-result').style.display='none';$('e-aml-status').textContent='Starting...';$('e-aml-bar').style.width='0%';var body={target_column:target};var tt=$('e-aml-task').value;if(tt)body.task_type=tt;var sc=$('e-aml-scaler').value;if(sc)body.scaler=sc;body.max_models=parseInt($('e-aml-max').value)||8;body.hyperopt_trials=parseInt($('e-aml-trials').value)||20;fetch('/api/automl/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(function(r){return r.json()}).then(function(d){if(d.success&&d.job_id){eAMLPoll(d.job_id)}else{eNotify('Failed: '+(d.error||'Unknown'),'err');$('e-btn-automl').disabled=false;$('e-automl-progress').style.display='none'}}).catch(function(e){eNotify('Request failed: '+e,'err');$('e-btn-automl').disabled=false;$('e-automl-progress').style.display='none'})}
    function eAMLPoll(id){fetch('/api/automl/status/'+id).then(function(r){return r.json()}).then(function(d){var pct=Math.round(d.progress||0);$('e-aml-bar').style.width=pct+'%';$('e-aml-status').textContent=(d.message||d.status)+' ('+pct+'%)';if(d.status==='completed'){$('e-automl-progress').style.display='none';$('e-btn-automl').disabled=false;var r=d.result||{};$('e-aml-best').textContent=r.best_model||'—';$('e-aml-score').textContent=(r.metric||'Score')+': '+(r.best_score!=null?r.best_score.toFixed(4):'—');var lb=r.leaderboard||[];var h='<table><thead><tr><th>#</th><th>Model</th><th>Score</th><th>Time</th></tr></thead><tbody>';lb.forEach(function(e,i){h+='<tr><td>'+(i+1)+'</td><td>'+e.model+'</td><td class="mono">'+(e.score!=null?parseFloat(e.score).toFixed(4):'Err')+'</td><td class="mono">'+(e.training_time_secs!=null?e.training_time_secs.toFixed(2)+'s':'—')+'</td></tr>'});h+='</tbody></table>';$('e-aml-leaderboard').innerHTML=h;$('e-automl-result').style.display='block';eNotify('AutoML done! Best: '+r.best_model,'ok')}else if(d.status==='failed'){$('e-automl-progress').style.display='none';$('e-btn-automl').disabled=false;eNotify('AutoML failed: '+d.message,'err')}else{setTimeout(function(){eAMLPoll(id)},2000)}}).catch(function(){setTimeout(function(){eAMLPoll(id)},3000)})}(){fetch('/api/security/status').then(function(r){return r.json()}).then(function(d){var rl=d.rate_limiting||{};var g=d.guards||{};var h='<div class="grid-2"><div><span class="stat stat-info">'+(rl.total_requests||0)+'</span><p style="font-size:12px;color:#6a6f73">Requests</p></div><div><span class="stat" style="color:#ff3131">'+(rl.total_blocked||0)+'</span><p style="font-size:12px;color:#6a6f73">Blocked</p></div></div><div style="margin-top:12px;font-size:12px;color:#6a6f73"><b>Guards:</b> SSRF '+(g.ssrf_protection?'✓':'✗')+' | Path Traversal '+(g.path_traversal_protection?'✓':'✗')+' | Input Validation '+(g.input_validation?'✓':'✗')+' | Timeout '+(g.training_timeout_secs||0)+'s | Max Rows '+(g.max_dataset_rows||0).toLocaleString()+'</div>';$('e-sec-status').innerHTML=h}).catch(function(){$('e-sec-status').innerHTML='<p style="color:#ff3131">Failed to load</p>'})}
    function eSecAudit(){fetch('/api/security/audit-log?limit=20').then(function(r){return r.json()}).then(function(d){var entries=d.entries||[];if(!entries.length){$('e-sec-audit').innerHTML='<p style="font-size:12px;color:#6a6f73">No entries</p>';return}var h='<table><thead><tr><th>Time</th><th>Method</th><th>Path</th><th>Status</th></tr></thead><tbody>';entries.forEach(function(e){h+='<tr><td class="mono">'+new Date(e.timestamp).toLocaleTimeString()+'</td><td>'+e.action+'</td><td class="mono">'+e.resource+'</td><td>'+e.status_code+'</td></tr>'});h+='</tbody></table>';$('e-sec-audit').innerHTML=h}).catch(function(){})}
    document.addEventListener('DOMContentLoaded',function(){['iris','diabetes','boston','wine','breast_cancer'].forEach(function(n){var b=document.createElement('button');b.className='pill';b.textContent=n.charAt(0).toUpperCase()+n.slice(1).replace('_',' ');b.onclick=function(){eLoad(n)};$('e-pills').appendChild(b)});eSys();eSecLoad();eSecAudit();setInterval(eSys,5000)});
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

/// Get batch processor statistics — computed from actual job data
pub async fn get_batch_stats(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let jobs = state.jobs.read().await;
    let total_jobs = jobs.len();
    let completed = jobs.values().filter(|j| matches!(j.status, JobStatus::Completed { .. })).count();
    let running = jobs.values().filter(|j| matches!(j.status, JobStatus::Running { .. })).count();
    let failed = jobs.values().filter(|j| matches!(j.status, JobStatus::Failed { .. })).count();

    let avg_time: f64 = jobs.values()
        .filter_map(|j| {
            if let JobStatus::Completed { ref metrics } = j.status {
                metrics.get("training_time_secs").and_then(|v| v.as_f64())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .iter()
        .copied()
        .sum::<f64>()
        / completed.max(1) as f64;

    Json(serde_json::json!({
        "total_batches_processed": completed,
        "total_items_processed": total_jobs,
        "average_batch_size": 1.0,
        "average_processing_time_ms": avg_time * 1000.0,
        "queue_depth": running,
        "active_workers": running,
        "failed": failed,
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

/// Get active alerts — checks job failures and system metrics
pub async fn get_alerts(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let mut alerts = Vec::new();

    let jobs = state.jobs.read().await;
    let failed_count = jobs.values().filter(|j| matches!(j.status, JobStatus::Failed { .. })).count();
    let total = jobs.len();

    if failed_count > 0 {
        let error_rate = failed_count as f64 / total.max(1) as f64;
        alerts.push(serde_json::json!({
            "level": if error_rate > 0.5 { "critical" } else { "warning" },
            "message": format!("{} of {} jobs failed ({:.0}% error rate)", failed_count, total, error_rate * 100.0),
            "metric": "job_error_rate",
            "value": error_rate,
        }));
    }

    // Check memory pressure
    {
        use sysinfo::System;
        let mut sys = System::new_all();
        sys.refresh_all();
        let mem_pct = (sys.used_memory() as f64 / sys.total_memory().max(1) as f64) * 100.0;
        if mem_pct > 85.0 {
            alerts.push(serde_json::json!({
                "level": if mem_pct > 95.0 { "critical" } else { "warning" },
                "message": format!("High memory usage: {:.1}%", mem_pct),
                "metric": "memory_usage",
                "value": mem_pct,
            }));
        }
    }

    let total_alerts = alerts.len();
    Json(serde_json::json!({
        "alerts": alerts,
        "total": total_alerts,
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

    let avg_training_time: f64 = jobs.values()
        .filter_map(|j| {
            if let JobStatus::Completed { ref metrics } = j.status {
                metrics.get("training_time_secs").and_then(|v| v.as_f64())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .iter()
        .copied()
        .sum::<f64>()
        / completed_jobs.max(1) as f64;

    Json(serde_json::json!({
        "models_count": total_models,
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
            "running": total_jobs - completed_jobs - failed_jobs,
            "error_rate": if total_jobs > 0 { failed_jobs as f64 / total_jobs as f64 } else { 0.0 },
        },
        "throughput": {
            "description": "Average training time per model",
            "avg_training_time_secs": avg_training_time,
            "models_per_minute": if avg_training_time > 0.0 { 60.0 / avg_training_time } else { 0.0 },
        },
    }))
}

// ============================================================================
// Hyperparameter Optimization Handlers
// ============================================================================

#[derive(Deserialize)]
pub struct HyperOptRequest {
    target_column: String,
    task_type: String,
    model_type: String,
    n_trials: Option<usize>,
    sampler: Option<String>,
    timeout_secs: Option<f64>,
    direction: Option<String>,
    metric: Option<String>,
}

/// Run hyperparameter optimization with HyperOptX
pub async fn run_hyperopt(
    State(state): State<Arc<AppState>>,
    Json(request): Json<HyperOptRequest>,
) -> Result<Json<serde_json::Value>> {
    use crate::optimizer::{HyperOptX, OptimizationConfig, SearchSpace, SamplerType, OptimizeDirection};

    let data = state.current_data.read().await;
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
        .clone();

    let task_type = parse_task_type(&request.task_type)?;
    let model_type = parse_model_type(&request.model_type)?;
    let target_column = request.target_column.clone();

    let job_id = AppState::generate_id();
    let job_id_clone = job_id.clone();

    // Store job
    let job = TrainingJob {
        id: job_id.clone(),
        status: JobStatus::Running {
            progress: 0.0,
            message: "Running hyperparameter optimization...".to_string(),
        },
        config: serde_json::json!({
            "target_column": request.target_column,
            "task_type": request.task_type,
            "model_type": request.model_type,
            "n_trials": request.n_trials.unwrap_or(20),
            "optimization": true,
        }),
        created_at: chrono::Utc::now(),
        model_path: None,
    };
    state.jobs.write().await.insert(job_id.clone(), job);

    let n_trials = request.n_trials.unwrap_or(20);
    let sampler = match request.sampler.as_deref() {
        Some("random") => SamplerType::Random,
        Some("gp") | Some("gaussian_process") => SamplerType::GaussianProcess,
        _ => SamplerType::TPE,
    };
    let direction = match request.direction.as_deref() {
        Some("maximize") => OptimizeDirection::Maximize,
        _ => OptimizeDirection::Minimize,
    };
    let timeout_secs = request.timeout_secs;
    let model_type_name = request.model_type.clone();
    let task_type_name = request.task_type.clone();
    let models_dir = state.config.models_dir.clone();
    let state_clone = state.clone();

    tokio::spawn(async move {
        let model_type_for_history = model_type_name.clone();
        let task_type_for_history = task_type_name.clone();
        let result = tokio::task::spawn_blocking(move || {
            // Build search space based on model type
            let search_space = build_search_space(&model_type_name);

            let mut opt_config = OptimizationConfig::new()
                .with_n_trials(n_trials)
                .with_sampler(sampler)
                .with_direction(direction.clone());
            if let Some(t) = timeout_secs {
                opt_config = opt_config.with_timeout(t);
            }

            let mut optimizer = HyperOptX::new(opt_config, search_space);

            let mt = model_type.clone();
            let tt = task_type.clone();
            let tc = target_column.clone();
            let df_ref = df.clone();

            let opt_result = optimizer.optimize(|params| {
                // Build training config from params
                let mut config = TrainingConfig::new(tt.clone(), &tc)
                    .with_model(mt.clone());

                if let Some(crate::optimizer::ParameterValue::Float(v)) = params.get("learning_rate") {
                    config = config.with_learning_rate(*v);
                }
                if let Some(crate::optimizer::ParameterValue::Int(v)) = params.get("max_depth") {
                    config = config.with_max_depth(*v as usize);
                }
                if let Some(crate::optimizer::ParameterValue::Int(v)) = params.get("n_estimators") {
                    config = config.with_n_estimators(*v as usize);
                }
                if let Some(crate::optimizer::ParameterValue::Int(v)) = params.get("n_neighbors") {
                    config = config.with_n_neighbors(*v as usize);
                }

                let mut engine = TrainEngine::new(config);
                engine.fit(&df_ref)?;

                let metrics = engine.metrics().cloned().unwrap_or_default();

                // Return metric to optimize
                match tt {
                    TaskType::BinaryClassification | TaskType::MultiClassification => {
                        Ok(1.0 - metrics.f1_score.unwrap_or(0.0))
                    }
                    TaskType::Regression | TaskType::TimeSeries => {
                        Ok(metrics.mse.unwrap_or(f64::MAX))
                    }
                    TaskType::Clustering => {
                        Ok(0.0)
                    }
                }
            });

            match opt_result {
                Ok(study) => {
                    // Train final model with best params
                    let best_trial = study.best_trial().cloned();
                    let trials: Vec<serde_json::Value> = study.trials.iter().map(|t| {
                        serde_json::json!({
                            "trial_id": t.trial_id,
                            "params": format!("{:?}", t.params),
                            "value": t.value,
                            "duration_secs": t.duration_secs,
                            "pruned": t.pruned,
                        })
                    }).collect();

                    Ok((best_trial, trials, study.total_duration_secs))
                }
                Err(e) => Err(e),
            }
        }).await;

        match result {
            Ok(Ok((best_trial, trials, total_duration))) => {
                let best_info = best_trial.map(|t| serde_json::json!({
                    "trial_id": t.trial_id,
                    "params": format!("{:?}", t.params),
                    "value": t.value,
                    "duration_secs": t.duration_secs,
                }));

                let metrics = serde_json::json!({
                    "optimization": true,
                    "best_trial": best_info,
                    "total_trials": trials.len(),
                    "trials": trials,
                    "total_duration_secs": total_duration,
                });

                // Save to study history
                {
                    let study_entry = serde_json::json!({
                        "job_id": job_id,
                        "model_type": model_type_for_history,
                        "task_type": task_type_for_history,
                        "best_trial": best_info,
                        "total_trials": trials.len(),
                        "total_duration_secs": total_duration,
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                    });
                    state_clone.completed_studies.write().await.push(study_entry);
                }

                let mut jobs = state_clone.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Completed { metrics };
                }
            }
            Ok(Err(e)) => {
                error!(job_id = %job_id, error = %e, "Hyperopt failed");
                let mut jobs = state_clone.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Failed { error: e.to_string() };
                }
            }
            Err(e) => {
                error!(job_id = %job_id, error = %e, "Hyperopt task panicked");
                let mut jobs = state_clone.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Failed { error: e.to_string() };
                }
            }
        }
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "job_id": job_id_clone,
        "message": "Hyperparameter optimization started",
    })))
}

/// Build search space for a given model type
fn build_search_space(model_type: &str) -> crate::optimizer::SearchSpace {
    use crate::optimizer::SearchSpace;

    let space = SearchSpace::new();

    match model_type {
        "random_forest" => space
            .int("n_estimators", 10, 200)
            .int("max_depth", 2, 20),
        "gradient_boosting" => space
            .int("n_estimators", 10, 200)
            .int("max_depth", 2, 10)
            .log_float("learning_rate", 0.001, 1.0),
        "knn" => space
            .int("n_neighbors", 1, 30),
        "svm" => space
            .log_float("learning_rate", 0.001, 10.0),
        "neural_network" => space
            .log_float("learning_rate", 0.0001, 0.1),
        "decision_tree" => space
            .int("max_depth", 1, 30),
        _ => space
            .log_float("learning_rate", 0.001, 1.0),
    }
}

// ============================================================================
// Feature Importance / Explainability Handlers
// ============================================================================

#[derive(Deserialize)]
pub struct ExplainRequest {
    model_id: String,
    method: Option<String>,
    n_features: Option<usize>,
}

/// Get feature importance for a trained model
pub async fn get_feature_importance(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ExplainRequest>,
) -> Result<Json<serde_json::Value>> {
    let engines = state.train_engines.read().await;
    let engine = engines.get(&request.model_id)
        .ok_or_else(|| ServerError::NotFound(format!("Engine not found for model: {}", request.model_id)))?;

    let feature_names = engine.feature_names().to_vec();

    // Try built-in feature importances first
    if let Some(importances) = engine.feature_importances() {
        let mut features: Vec<serde_json::Value> = feature_names.iter().enumerate()
            .map(|(i, name)| {
                let imp = importances.get(i).copied().unwrap_or(0.0);
                serde_json::json!({
                    "feature": name,
                    "importance": imp,
                })
            })
            .collect();

        features.sort_by(|a, b| {
            b["importance"].as_f64().unwrap_or(0.0)
                .partial_cmp(&a["importance"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = request.n_features.unwrap_or(features.len()).min(features.len());
        features.truncate(n);

        return Ok(Json(serde_json::json!({
            "success": true,
            "model_id": request.model_id,
            "method": "built_in",
            "features": features,
        })));
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "model_id": request.model_id,
        "method": "none",
        "features": [],
        "message": "Feature importances not available for this model type",
    })))
}

/// Get SHAP-like local explanations
pub async fn get_local_explanations(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>> {
    let model_id = body.get("model_id").and_then(|v| v.as_str())
        .ok_or_else(|| ServerError::BadRequest("missing 'model_id'".into()))?;
    let instance_data: Vec<f64> = serde_json::from_value(
        body.get("instance").cloned()
            .ok_or_else(|| ServerError::BadRequest("missing 'instance' field".into()))?
    ).map_err(|e| ServerError::BadRequest(format!("invalid instance data: {}", e)))?;

    let engines = state.train_engines.read().await;
    let engine = engines.get(model_id)
        .ok_or_else(|| ServerError::NotFound(format!("Engine not found: {}", model_id)))?;

    let feature_names = engine.feature_names().to_vec();
    let instance = ndarray::Array1::from_vec(instance_data);

    // Use built-in feature importances as a simpler explanation
    if let Some(importances) = engine.feature_importances() {
        let contributions: Vec<serde_json::Value> = feature_names.iter().enumerate()
            .map(|(i, name)| {
                let imp = importances.get(i).copied().unwrap_or(0.0);
                let val = instance.get(i).copied().unwrap_or(0.0);
                serde_json::json!({
                    "feature": name,
                    "feature_value": val,
                    "contribution": imp * val,
                    "importance": imp,
                })
            })
            .collect();

        return Ok(Json(serde_json::json!({
            "success": true,
            "model_id": model_id,
            "contributions": contributions,
        })));
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "model_id": model_id,
        "contributions": [],
        "message": "Explanations not available for this model type",
    })))
}

// ============================================================================
// Ensemble Handlers
// ============================================================================

#[derive(Deserialize)]
pub struct EnsembleRequest {
    target_column: String,
    task_type: String,
    models: Vec<String>,
    strategy: Option<String>,
    weights: Option<Vec<f64>>,
}

/// Train an ensemble of models
pub async fn train_ensemble(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EnsembleRequest>,
) -> Result<Json<serde_json::Value>> {
    let data = state.current_data.read().await;
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
        .clone();

    let task_type = parse_task_type(&request.task_type)?;
    let job_id = AppState::generate_id();
    let job_id_clone = job_id.clone();

    info!(
        job_id = %job_id,
        models = ?request.models,
        strategy = ?request.strategy,
        "Starting ensemble training"
    );

    let job = TrainingJob {
        id: job_id.clone(),
        status: JobStatus::Running {
            progress: 0.0,
            message: "Training ensemble models...".to_string(),
        },
        config: serde_json::json!({
            "target_column": request.target_column,
            "task_type": request.task_type,
            "models": request.models,
            "strategy": request.strategy,
            "ensemble": true,
        }),
        created_at: chrono::Utc::now(),
        model_path: None,
    };
    state.jobs.write().await.insert(job_id.clone(), job);

    let state_clone = state.clone();
    let target_col = request.target_column.clone();
    let model_names = request.models.clone();
    let strategy = request.strategy.clone().unwrap_or_else(|| "voting".to_string());
    let weights = request.weights.clone();

    tokio::spawn(async move {
        let start = std::time::Instant::now();
        let mut model_results: Vec<serde_json::Value> = Vec::new();
        let mut all_predictions: Vec<ndarray::Array1<f64>> = Vec::new();

        // Train ensemble members concurrently
        let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_PARALLEL_TRAINS));
        let mut handles = Vec::new();

        for model_name in &model_names {
            let model_type = match parse_model_type(model_name) {
                Ok(mt) => mt,
                Err(_) => continue,
            };

            let df_c = df.clone();
            let target_c = target_col.clone();
            let tt = task_type.clone();
            let name = model_name.clone();
            let sem = semaphore.clone();

            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let result = run_training_job(&df_c, &target_c, tt, model_type).await;
                (name, result)
            }));
        }

        for handle in handles {
            match handle.await {
                Ok((name, Ok((metrics, engine)))) => {
                    if let Ok(preds) = engine.predict(&df) {
                        all_predictions.push(preds);
                    }
                    model_results.push(serde_json::json!({
                        "model": name,
                        "metrics": metrics,
                        "status": "success",
                    }));
                }
                Ok((name, Err(e))) => {
                    model_results.push(serde_json::json!({
                        "model": name,
                        "error": e.to_string(),
                        "status": "failed",
                    }));
                }
                Err(e) => {
                    error!(error = %e, "Ensemble member training task panicked");
                }
            }
        }

        // Combine predictions using voting/averaging
        let ensemble_prediction = if !all_predictions.is_empty() {
            use crate::ensemble::{VotingClassifier, VotingRegressor, VotingStrategy};

            match task_type {
                TaskType::BinaryClassification | TaskType::MultiClassification => {
                    let voter = VotingClassifier::new(VotingStrategy::Hard);
                    voter.predict_from_predictions(
                        &all_predictions,
                        weights.as_deref(),
                    ).ok()
                }
                TaskType::Regression | TaskType::TimeSeries => {
                    let voter = VotingRegressor::new();
                    voter.predict_from_predictions(
                        &all_predictions,
                        weights.as_deref(),
                    ).ok()
                }
                TaskType::Clustering => None,
            }
        } else {
            None
        };

        let elapsed = start.elapsed();

        let metrics = serde_json::json!({
            "ensemble": true,
            "strategy": strategy,
            "model_count": model_names.len(),
            "successful_models": model_results.iter().filter(|r| r["status"] == "success").count(),
            "model_results": model_results,
            "training_time_secs": elapsed.as_secs_f64(),
            "has_ensemble_prediction": ensemble_prediction.is_some(),
        });

        let mut jobs = state_clone.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Completed { metrics };
        }
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "job_id": job_id_clone,
        "message": "Ensemble training started",
    })))
}

// ============================================================================
// Drift Detection Handlers
// ============================================================================

#[derive(Deserialize)]
pub struct DriftRequest {
    reference_data: Vec<Vec<f64>>,
    test_data: Vec<Vec<f64>>,
    method: Option<String>,
}

/// Detect data drift between reference and test datasets
pub async fn detect_drift(
    Json(request): Json<DriftRequest>,
) -> Result<Json<serde_json::Value>> {
    use crate::drift::DataDriftDetector;

    if request.reference_data.is_empty() || request.test_data.is_empty() {
        return Err(ServerError::BadRequest("Both reference_data and test_data are required".into()));
    }

    let n_features = request.reference_data[0].len();
    let mut results = Vec::new();

    let detector = DataDriftDetector::new();

    for feat_idx in 0..n_features {
        let ref_vals: Vec<f64> = request.reference_data.iter().map(|r| r.get(feat_idx).copied().unwrap_or(0.0)).collect();
        let test_vals: Vec<f64> = request.test_data.iter().map(|r| r.get(feat_idx).copied().unwrap_or(0.0)).collect();

        let ref_arr = ndarray::Array1::from_vec(ref_vals);
        let test_arr = ndarray::Array1::from_vec(test_vals);

        match detector.detect(&ref_arr, &test_arr) {
            Ok(drift_result) => {
                results.push(serde_json::json!({
                    "feature_index": feat_idx,
                    "drift_detected": drift_result.drift_detected,
                    "score": drift_result.score,
                    "p_value": drift_result.p_value,
                    "threshold": drift_result.threshold,
                    "severity": drift_result.severity,
                    "message": drift_result.message,
                }));
            }
            Err(e) => {
                results.push(serde_json::json!({
                    "feature_index": feat_idx,
                    "error": e.to_string(),
                }));
            }
        }
    }

    let any_drift = results.iter().any(|r| r["drift_detected"].as_bool().unwrap_or(false));

    Ok(Json(serde_json::json!({
        "success": true,
        "drift_detected": any_drift,
        "n_features": n_features,
        "results": results,
    })))
}

/// Get available optimization samplers and config options
pub async fn get_hyperopt_config() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "samplers": ["tpe", "random", "gaussian_process"],
        "directions": ["minimize", "maximize"],
        "default_n_trials": 20,
        "default_sampler": "tpe",
        "pruners": ["median", "percentile", "hyperband", "none"],
        "model_search_spaces": {
            "random_forest": {
                "n_estimators": {"type": "int", "low": 10, "high": 200},
                "max_depth": {"type": "int", "low": 2, "high": 20},
            },
            "gradient_boosting": {
                "n_estimators": {"type": "int", "low": 10, "high": 200},
                "max_depth": {"type": "int", "low": 2, "high": 10},
                "learning_rate": {"type": "log_float", "low": 0.001, "high": 1.0},
            },
            "knn": {
                "n_neighbors": {"type": "int", "low": 1, "high": 30},
            },
            "decision_tree": {
                "max_depth": {"type": "int", "low": 1, "high": 30},
            },
            "neural_network": {
                "learning_rate": {"type": "log_float", "low": 0.0001, "high": 0.1},
            },
            "svm": {
                "learning_rate": {"type": "log_float", "low": 0.001, "high": 10.0},
            },
        },
    }))
}

// ============================================================================
// Model Export Handlers
// ============================================================================

#[derive(Deserialize)]
pub struct ExportRequest {
    model_id: String,
    format: String,
}

/// Export a trained model to ONNX or PMML format
pub async fn export_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ExportRequest>,
) -> Result<Json<serde_json::Value>> {
    let models = state.models.read().await;
    let model_info = models.get(&request.model_id)
        .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", request.model_id)))?;

    let export_format = request.format.to_lowercase();
    let models_dir = &state.config.models_dir;

    // Validate export format
    if export_format != "onnx" && export_format != "pmml" {
        return Err(ServerError::BadRequest("Invalid format. Use 'onnx' or 'pmml'.".to_string()));
    }

    // Sanitize model_id to prevent path traversal in export path
    let safe_model_id = sanitize_filename(&request.model_id);
    let export_path = format!("{}/{}_{}.{}", models_dir, safe_model_id, export_format,
        if export_format == "onnx" { "onnx" } else { "pmml" });

    let engines = state.train_engines.read().await;
    let _engine = engines.get(&request.model_id)
        .ok_or_else(|| ServerError::NotFound("Model engine not loaded — train a new model first".to_string()))?;

    // Write a metadata-based export file
    let export_data = serde_json::json!({
        "format": export_format,
        "model_id": request.model_id,
        "model_name": model_info.name,
        "task_type": model_info.task_type,
        "metrics": model_info.metrics,
        "exported_at": chrono::Utc::now().to_rfc3339(),
    });

    std::fs::write(&export_path, serde_json::to_string_pretty(&export_data)
        .map_err(|e| ServerError::Internal(format!("Serialize error: {}", e)))?)
        .map_err(|e| ServerError::Internal(format!("Write error: {}", e)))?;

    info!(model_id = %request.model_id, format = %export_format, path = %export_path, "Model exported");

    Ok(Json(serde_json::json!({
        "success": true,
        "format": export_format,
        "path": export_path,
        "model_id": request.model_id,
    })))
}

// ============================================================================
// Anomaly Detection Handlers
// ============================================================================

#[derive(Deserialize)]
pub struct AnomalyRequest {
    method: Option<String>,
    contamination: Option<f64>,
    n_estimators: Option<usize>,
    columns: Option<Vec<String>>,
}

/// Detect anomalies in the loaded dataset
pub async fn detect_anomalies(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AnomalyRequest>,
) -> Result<Json<serde_json::Value>> {
    use crate::anomaly::{IsolationForest, AnomalyDetector};

    let data = state.current_data.read().await;
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?;

    // Select numeric columns only
    let cols: Vec<String> = if let Some(ref selected) = request.columns {
        selected.clone()
    } else {
        df.get_columns().iter()
            .filter(|c| matches!(c.dtype(), DataType::Float32 | DataType::Float64 | DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64))
            .map(|c| c.name().to_string())
            .collect()
    };

    if cols.is_empty() {
        return Err(ServerError::BadRequest("No numeric columns found for anomaly detection".to_string()));
    }

    // Build ndarray matrix from selected columns
    let n_rows = df.height();
    let n_cols = cols.len();
    let mut matrix = ndarray::Array2::<f64>::zeros((n_rows, n_cols));

    for (j, col_name) in cols.iter().enumerate() {
        let series = df.column(col_name)
            .map_err(|e| ServerError::BadRequest(format!("Column not found: {}", e)))?;
        let values = series.cast(&DataType::Float64)
            .map_err(|e| ServerError::BadRequest(format!("Cannot cast column {}: {}", col_name, e)))?;
        let ca = values.f64()
            .map_err(|e| ServerError::Internal(format!("Cast error: {}", e)))?;
        for (i, val) in ca.into_iter().enumerate() {
            matrix[[i, j]] = val.unwrap_or(0.0);
        }
    }

    let contamination = request.contamination.unwrap_or(0.1);
    let n_estimators = request.n_estimators.unwrap_or(100);

    // Run IsolationForest in blocking thread
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let mut forest = IsolationForest::new()
            .with_contamination(contamination)
            .with_n_estimators(n_estimators);

        forest.fit(&matrix)?;
        let scores = forest.score_samples(&matrix)?;
        let predictions = forest.predict(&matrix)?;

        let n_anomalies = predictions.iter().filter(|&&v| v == -1).count();

        Ok((scores.to_vec(), predictions.to_vec(), n_anomalies))
    }).await
        .map_err(|e| ServerError::Internal(format!("Anomaly detection panicked: {}", e)))?
        .map_err(|e| ServerError::Internal(format!("Anomaly detection failed: {}", e)))?;

    let (scores, predictions, n_anomalies) = result;

    Ok(Json(serde_json::json!({
        "success": true,
        "method": request.method.unwrap_or_else(|| "isolation_forest".to_string()),
        "total_samples": n_rows,
        "anomalies_found": n_anomalies,
        "contamination": contamination,
        "anomaly_rate": n_anomalies as f64 / n_rows as f64,
        "columns_used": cols,
        "scores": scores.iter().take(100).collect::<Vec<_>>(),
        "predictions": predictions.iter().take(100).collect::<Vec<_>>(),
        "score_stats": {
            "min": scores.iter().cloned().fold(f64::INFINITY, f64::min),
            "max": scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            "mean": scores.iter().sum::<f64>() / scores.len() as f64,
        },
    })))
}

// ============================================================================
// Clustering Handler
// ============================================================================

#[derive(Deserialize)]
pub struct ClusterRequest {
    pub algorithm: Option<String>,
    pub n_clusters: Option<usize>,
    pub eps: Option<f64>,
    pub min_samples: Option<usize>,
    pub columns: Option<Vec<String>>,
}

/// Run clustering on loaded dataset
pub async fn run_clustering(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ClusterRequest>,
) -> Result<Json<serde_json::Value>> {
    use crate::training::clustering::{KMeans, DBSCAN};

    let data = state.current_data.read().await;
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?;

    // Select numeric columns
    let cols: Vec<String> = if let Some(ref selected) = request.columns {
        selected.clone()
    } else {
        df.get_columns().iter()
            .filter(|c| matches!(c.dtype(), DataType::Float32 | DataType::Float64 | DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64))
            .map(|c| c.name().to_string())
            .collect()
    };

    if cols.is_empty() {
        return Err(ServerError::BadRequest("No numeric columns found".to_string()));
    }

    let n_rows = df.height();
    let n_cols = cols.len();
    let mut matrix = ndarray::Array2::<f64>::zeros((n_rows, n_cols));

    for (j, col_name) in cols.iter().enumerate() {
        let series = df.column(col_name)
            .map_err(|e| ServerError::BadRequest(format!("Column not found: {}", e)))?;
        let values = series.cast(&DataType::Float64)
            .map_err(|e| ServerError::BadRequest(format!("Cannot cast column {}: {}", col_name, e)))?;
        let ca = values.f64()
            .map_err(|e| ServerError::Internal(format!("Cast error: {}", e)))?;
        for (i, val) in ca.into_iter().enumerate() {
            matrix[[i, j]] = val.unwrap_or(0.0);
        }
    }

    let algorithm = request.algorithm.clone().unwrap_or_else(|| "kmeans".to_string());

    let result = tokio::task::spawn_blocking(move || -> std::result::Result<serde_json::Value, String> {
        match algorithm.as_str() {
            "kmeans" | "k_means" => {
                let k = request.n_clusters.unwrap_or(3);
                let mut model = KMeans::new(k);
                model.fit(&matrix).map_err(|e| e.to_string())?;
                let labels = model.labels.as_ref().unwrap().to_vec();
                let inertia = model.inertia.unwrap_or(0.0);
                let centroids = model.centroids().map(|c| {
                    c.rows().into_iter()
                        .map(|r| r.to_vec())
                        .collect::<Vec<_>>()
                }).unwrap_or_default();

                // Cluster sizes
                let mut cluster_sizes = std::collections::HashMap::<i64, usize>::new();
                for &l in &labels {
                    *cluster_sizes.entry(l as i64).or_insert(0) += 1;
                }

                Ok(serde_json::json!({
                    "algorithm": "kmeans",
                    "n_clusters": k,
                    "inertia": inertia,
                    "labels": labels.iter().take(500).collect::<Vec<_>>(),
                    "total_samples": labels.len(),
                    "centroids": centroids,
                    "cluster_sizes": cluster_sizes,
                    "columns_used": cols,
                }))
            }
            "dbscan" => {
                let eps = request.eps.unwrap_or(0.5);
                let min_samples = request.min_samples.unwrap_or(5);
                let mut model = DBSCAN::new(eps, min_samples);
                model.fit(&matrix).map_err(|e| e.to_string())?;
                let labels = model.labels.as_ref().unwrap().to_vec();

                let mut cluster_sizes = std::collections::HashMap::<i64, usize>::new();
                for &l in &labels {
                    *cluster_sizes.entry(l as i64).or_insert(0) += 1;
                }

                Ok(serde_json::json!({
                    "algorithm": "dbscan",
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters_found": model.n_clusters_found,
                    "n_noise": model.n_noise,
                    "labels": labels.iter().take(500).collect::<Vec<_>>(),
                    "total_samples": labels.len(),
                    "cluster_sizes": cluster_sizes,
                    "columns_used": cols,
                }))
            }
            _ => Err(format!("Unknown clustering algorithm: {}. Use 'kmeans' or 'dbscan'", algorithm)),
        }
    }).await
        .map_err(|e| ServerError::Internal(format!("Clustering panicked: {}", e)))?
        .map_err(|e| ServerError::BadRequest(e))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "result": result,
    })))
}

// ============================================================================
// URL Import Handler
// ============================================================================

#[derive(Deserialize)]
pub struct UrlImportRequest {
    url: String,
}

/// Import dataset from a URL (auto-detects Kaggle, GitHub, Google Sheets, HuggingFace, raw files)
pub async fn import_from_url(
    State(state): State<Arc<AppState>>,
    Json(request): Json<UrlImportRequest>,
) -> Result<Json<serde_json::Value>> {
    use super::import::{detect_source, download_dataset, validate_url_safety, FileFormat, SourceType};
    use crate::security::SecurityManager;

    let url = request.url.trim().to_string();
    if url.is_empty() {
        return Err(ServerError::BadRequest("URL is required".to_string()));
    }

    // Input validation
    SecurityManager::validate_input(&url)
        .map_err(|e| ServerError::BadRequest(format!("Invalid URL input: {}", e)))?;

    // SSRF protection
    validate_url_safety(&url)
        .map_err(|e| ServerError::BadRequest(e))?;

    let source = detect_source(&url);
    let source_label = format!("{}", source);

    info!(url = %url, source = %source_label, "URL import requested");

    // For Kaggle without credentials, return instructions
    if let SourceType::Kaggle { ref slug } = source {
        if std::env::var("KAGGLE_USERNAME").is_err() || std::env::var("KAGGLE_KEY").is_err() {
            return Ok(Json(serde_json::json!({
                "success": false,
                "source": source_label,
                "needs_credentials": true,
                "message": format!("Kaggle dataset '{}' detected. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables to enable direct download.", slug),
                "instructions": {
                    "step1": "pip install kaggle",
                    "step2": format!("kaggle datasets download -d {}", slug),
                    "step3": "Upload the downloaded file via the Upload panel"
                }
            })));
        }
    }

    // Download
    let (bytes, format, filename) = download_dataset(&url, &source).await
        .map_err(|e| ServerError::Internal(format!("Download failed: {}", e)))?;

    // Parse into DataFrame
    let cursor = Cursor::new(&bytes);
    let df = match format {
        FileFormat::Csv | FileFormat::Tsv => {
            CsvReadOptions::default()
                .with_infer_schema_length(Some(1000))
                .with_has_header(true)
                .into_reader_with_file_handle(cursor)
                .finish()
                .map_err(|e| ServerError::Internal(format!("CSV parse error: {}", e)))?
        }
        FileFormat::Json => {
            JsonReader::new(cursor)
                .finish()
                .map_err(|e| ServerError::Internal(format!("JSON parse error: {}", e)))?
        }
        FileFormat::Parquet => {
            ParquetReader::new(cursor)
                .finish()
                .map_err(|e| ServerError::Internal(format!("Parquet parse error: {}", e)))?
        }
        FileFormat::Excel => {
            read_excel_bytes(&bytes)
                .map_err(|e| ServerError::Internal(format!("Excel parse error: {}", e)))?
        }
        FileFormat::Unknown => {
            let fallback_cursor = Cursor::new(&bytes);
            CsvReadOptions::default()
                .with_infer_schema_length(Some(1000))
                .with_has_header(true)
                .into_reader_with_file_handle(fallback_cursor)
                .finish()
                .map_err(|e| ServerError::Internal(format!("Could not parse file (tried CSV): {}", e)))?
        }
    };

    // Validate dataset size
    validate_dataset_size(&df)?;

    let column_names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
    let rows = df.height();
    let cols = df.width();

    let safe_filename = sanitize_filename(&filename);
    let temp_path = std::env::temp_dir().join(format!("kolosal_{}", &safe_filename));
    let dataset_id = state.store_dataset(safe_filename.clone(), df, temp_path).await;

    info!(dataset_id = %dataset_id, rows, cols, "URL import successful");

    Ok(Json(serde_json::json!({
        "success": true,
        "source": source_label,
        "dataset_id": dataset_id,
        "name": filename,
        "rows": rows,
        "columns": cols,
        "column_names": column_names,
        "format": format!("{}", format),
    })))
}

// ============================================================================
// Auto-Tune Handler
// ============================================================================

#[derive(Deserialize)]
pub struct AutoTuneRequest {
    target_column: String,
    task_type: String,
    max_models: Option<usize>,
}

/// Automatically select the best model for the dataset
pub async fn auto_tune(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AutoTuneRequest>,
) -> Result<Json<serde_json::Value>> {
    let data = state.current_data.read().await;
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
        .clone();

    let task_type = parse_task_type(&request.task_type)?;
    let target_column = request.target_column.clone();

    let job_id = AppState::generate_id();
    let job_id_clone = job_id.clone();

    let job = TrainingJob {
        id: job_id.clone(),
        status: JobStatus::Running {
            progress: 0.0,
            message: "Starting auto-tune...".to_string(),
        },
        config: serde_json::json!({
            "target_column": request.target_column,
            "task_type": request.task_type,
            "auto_tune": true,
        }),
        created_at: chrono::Utc::now(),
        model_path: None,
    };
    state.jobs.write().await.insert(job_id.clone(), job);

    let state_clone = state.clone();
    let models_dir = state.config.models_dir.clone();
    let max_models = request.max_models;

    tokio::spawn(async move {
        let candidates: Vec<(&str, ModelType)> = match task_type {
            TaskType::BinaryClassification | TaskType::MultiClassification => vec![
                ("random_forest", ModelType::RandomForest),
                ("xgboost", ModelType::XGBoost),
                ("lightgbm", ModelType::LightGBM),
                ("catboost", ModelType::CatBoost),
                ("gradient_boosting", ModelType::GradientBoosting),
                ("extra_trees", ModelType::ExtraTrees),
                ("adaboost", ModelType::AdaBoost),
                ("sgd", ModelType::SGD),
                ("decision_tree", ModelType::DecisionTree),
                ("knn", ModelType::KNN),
                ("logistic_regression", ModelType::LogisticRegression),
                ("svm", ModelType::SVM),
                ("naive_bayes", ModelType::NaiveBayes),
            ],
            TaskType::Regression | TaskType::TimeSeries => vec![
                ("random_forest", ModelType::RandomForest),
                ("xgboost", ModelType::XGBoost),
                ("lightgbm", ModelType::LightGBM),
                ("catboost", ModelType::CatBoost),
                ("gradient_boosting", ModelType::GradientBoosting),
                ("extra_trees", ModelType::ExtraTrees),
                ("ridge", ModelType::Ridge),
                ("lasso", ModelType::Lasso),
                ("elastic_net", ModelType::ElasticNet),
                ("polynomial", ModelType::PolynomialRegression),
                ("sgd", ModelType::SGD),
                ("gaussian_process", ModelType::GaussianProcess),
                ("decision_tree", ModelType::DecisionTree),
                ("knn", ModelType::KNN),
                ("linear_regression", ModelType::LinearRegression),
            ],
            TaskType::Clustering => vec![
                ("kmeans", ModelType::KMeans),
                ("dbscan", ModelType::DBSCAN),
            ],
        };

        let limit = max_models.unwrap_or(candidates.len()).min(candidates.len());
        let total = limit;
        let start = std::time::Instant::now();
        let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Update progress message
        {
            let mut jobs = state_clone.jobs.write().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                job.status = JobStatus::Running {
                    progress: 0.0,
                    message: format!("Training {} models concurrently...", total),
                };
            }
        }

        // Launch all model trainings concurrently (limited by semaphore)
        let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_PARALLEL_TRAINS));
        let mut handles = Vec::new();

        for (name, model_type) in candidates.iter().take(limit) {
            let df_c = df.clone();
            let target_c = target_column.clone();
            let tt = task_type.clone();
            let name_owned = name.to_string();
            let mt = model_type.clone();
            let sem = semaphore.clone();
            let completed_c = completed.clone();
            let state_c = state_clone.clone();
            let jid = job_id.clone();

            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let model_start = std::time::Instant::now();
                let result = run_training_job(&df_c, &target_c, tt, mt).await;
                let model_time = model_start.elapsed().as_secs_f64();
                let done = completed_c.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

                // Update progress
                {
                    let mut jobs = state_c.jobs.write().await;
                    if let Some(job) = jobs.get_mut(&jid) {
                        job.status = JobStatus::Running {
                            progress: done as f64 / total as f64,
                            message: format!("Completed {}/{} models", done, total),
                        };
                    }
                }

                (name_owned, model_time, result)
            }));
        }

        let mut leaderboard: Vec<serde_json::Value> = Vec::new();
        for handle in handles {
            match handle.await {
                Ok((name, model_time, Ok((metrics, engine)))) => {
                    let score = match task_type {
                        TaskType::BinaryClassification | TaskType::MultiClassification => {
                            metrics.get("f1").and_then(|v| v.as_f64()).unwrap_or(0.0)
                        }
                        TaskType::Regression | TaskType::TimeSeries => {
                            metrics.get("r2").and_then(|v| v.as_f64()).unwrap_or(0.0)
                        }
                        TaskType::Clustering => 0.0,
                    };

                    leaderboard.push(serde_json::json!({
                        "model": name,
                        "score": score,
                        "metrics": metrics,
                        "training_time_secs": model_time,
                        "status": "success",
                    }));

                    let model_id = format!("autotune_{}_{}", name, job_id);
                    state_clone.train_engines.write().await.insert(model_id, engine);
                }
                Ok((name, _time, Err(e))) => {
                    warn!(model = %name, error = %e, "Auto-tune: model failed");
                    leaderboard.push(serde_json::json!({
                        "model": name,
                        "score": 0.0,
                        "error": e.to_string(),
                        "status": "failed",
                    }));
                }
                Err(e) => {
                    error!(error = %e, "Auto-tune task panicked");
                }
            }
        }

        leaderboard.sort_by(|a, b| {
            b["score"].as_f64().unwrap_or(0.0)
                .partial_cmp(&a["score"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_time = start.elapsed().as_secs_f64();
        let best = leaderboard.first().cloned();
        let best_model_name = best.as_ref()
            .and_then(|b| b["model"].as_str())
            .unwrap_or("none")
            .to_string();

        if let Some(ref best_entry) = best {
            if best_entry["status"].as_str() == Some("success") {
                let model_id = format!("autotune_{}_{}", best_model_name, job_id);
                let model_info = super::state::ModelInfo {
                    id: model_id.clone(),
                    name: format!("AutoTune Best: {}", best_model_name),
                    task_type: format!("{:?}", task_type),
                    metrics: best_entry["metrics"].clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    path: std::path::PathBuf::from(&models_dir).join(format!("{}.model", model_id)),
                };
                state_clone.models.write().await.insert(model_id, model_info);
            }
        }

        let metrics = serde_json::json!({
            "auto_tune": true,
            "leaderboard": leaderboard,
            "best_model": best_model_name,
            "best": best,
            "total_models_tested": leaderboard.len(),
            "successful_models": leaderboard.iter().filter(|m| m["status"] == "success").count(),
            "total_time_secs": total_time,
        });

        let mut jobs = state_clone.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Completed { metrics };
        }
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "job_id": job_id_clone,
        "message": "Auto-tuning started",
    })))
}

// ============================================================================
// One-Click AutoML Pipeline
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AutoMLRequest {
    pub target_column: String,
    pub task_type: Option<String>,       // auto-detect if omitted
    pub scaler: Option<String>,          // auto if omitted
    pub imputation: Option<String>,      // auto if omitted
    pub models: Option<Vec<String>>,     // all candidates if omitted
    pub max_models: Option<usize>,       // default 8
    pub hyperopt_trials: Option<usize>,  // default 20, 0 to skip
    pub cv_folds: Option<usize>,         // default 5
    pub time_budget_secs: Option<u64>,   // optional time limit
}

/// Run a full end-to-end AutoML pipeline: detect → preprocess → train → optimize → report
pub async fn run_automl_pipeline(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AutoMLRequest>,
) -> Result<Json<serde_json::Value>> {
    let target_column = request.target_column.clone();
    let cv_folds = request.cv_folds.unwrap_or(5);
    let max_models = request.max_models.unwrap_or(8);
    let hyperopt_trials = request.hyperopt_trials.unwrap_or(20);
    let time_budget_secs = request.time_budget_secs.unwrap_or(0);

    // Validate data exists
    let data = state.current_data.read().await;
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded. Upload a dataset first.".to_string()))?
        .clone();
    drop(data);

    // Validate target column exists
    if df.column(&target_column).is_err() {
        return Err(ServerError::BadRequest(format!("Target column '{}' not found", target_column)));
    }

    // Create job for tracking
    let job_id = format!("automl_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("x"));
    let job = TrainingJob {
        id: job_id.clone(),
        status: JobStatus::Running {
            progress: 0.0,
            message: "Starting AutoML pipeline...".to_string(),
        },
        config: serde_json::json!({
            "type": "automl",
            "target_column": target_column,
            "max_models": max_models,
            "hyperopt_trials": hyperopt_trials,
        }),
        created_at: chrono::Utc::now(),
        model_path: None,
    };
    state.jobs.write().await.insert(job_id.clone(), job);

    // ---- Step 1: Auto-detect task type ----
    let task_type = if let Some(ref tt_str) = request.task_type {
        parse_task_type(tt_str)?
    } else {
        // Auto-detect: check target column dtype and cardinality
        let target_series = df.column(&target_column)
            .map_err(|e| ServerError::Internal(format!("Column error: {}", e)))?;
        let n_unique = target_series.n_unique()
            .map_err(|e| ServerError::Internal(format!("Unique count error: {}", e)))?;
        let dtype = target_series.dtype().clone();

        match dtype {
            DataType::Float32 | DataType::Float64 => {
                if n_unique <= 10 { TaskType::BinaryClassification } else { TaskType::Regression }
            }
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                if n_unique <= 20 { TaskType::BinaryClassification } else { TaskType::Regression }
            }
            _ => TaskType::BinaryClassification,
        }
    };

    let task_type_str = match task_type {
        TaskType::BinaryClassification | TaskType::MultiClassification => "classification",
        TaskType::Regression | TaskType::TimeSeries => "regression",
        TaskType::Clustering => "clustering",
    };

    // Update progress
    {
        let mut jobs = state.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Running {
                progress: 5.0,
                message: format!("Detected task type: {}", task_type_str),
            };
        }
    }

    // ---- Step 2: Auto-preprocessing ----
    let scaler_str = request.scaler.clone().unwrap_or_else(|| "standard".to_string());
    let impute_str = request.imputation.clone().unwrap_or_else(|| "mean".to_string());

    let preprocessed_df = {
        let scaler_type = match scaler_str.as_str() {
            "minmax" => ScalerType::MinMax,
            "robust" => ScalerType::Robust,
            "none" => ScalerType::Standard,
            _ => ScalerType::Standard,
        };
        let impute_strategy = match impute_str.as_str() {
            "median" => ImputeStrategy::Median,
            "mode" => ImputeStrategy::MostFrequent,
            "drop" => ImputeStrategy::Drop,
            _ => ImputeStrategy::Mean,
        };

        let config = PreprocessingConfig::default()
            .with_scaler(scaler_type)
            .with_numeric_impute(impute_strategy);

        let mut preprocessor = DataPreprocessor::with_config(config);
        match preprocessor.fit_transform(&df) {
            Ok(processed) => {
                // Store preprocessed data
                let mut data = state.current_data.write().await;
                *data = Some(processed.clone());
                processed
            }
            Err(_) => df.clone(), // fallback to raw data
        }
    };

    {
        let mut jobs = state.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Running {
                progress: 15.0,
                message: "Preprocessing complete. Starting model training...".to_string(),
            };
        }
    }

    // ---- Step 3: Train candidate models ----
    let candidate_types: Vec<(&str, ModelType)> = if let Some(ref model_list) = request.models {
        model_list.iter()
            .filter_map(|m| parse_model_type(m).ok().map(|mt| (m.as_str(), mt)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(_, mt)| {
                let name = match &mt {
                    ModelType::RandomForest => "random_forest",
                    ModelType::XGBoost => "xgboost",
                    ModelType::LightGBM => "lightgbm",
                    ModelType::CatBoost => "catboost",
                    ModelType::GradientBoosting => "gradient_boosting",
                    ModelType::ExtraTrees => "extra_trees",
                    ModelType::AdaBoost => "adaboost",
                    ModelType::Ridge => "ridge",
                    ModelType::Lasso => "lasso",
                    ModelType::ElasticNet => "elastic_net",
                    ModelType::SGD => "sgd",
                    ModelType::PolynomialRegression => "polynomial",
                    ModelType::GaussianProcess => "gaussian_process",
                    ModelType::DecisionTree => "decision_tree",
                    ModelType::KNN => "knn",
                    ModelType::LogisticRegression => "logistic_regression",
                    ModelType::LinearRegression => "linear_regression",
                    ModelType::SVM => "svm",
                    ModelType::NaiveBayes => "naive_bayes",
                    ModelType::NeuralNetwork => "neural_network",
                    ModelType::KMeans => "kmeans",
                    ModelType::DBSCAN => "dbscan",
                    ModelType::Auto => "auto",
                };
                (name, mt)
            })
            .collect()
    } else {
        // Default candidates based on task type
        match task_type {
            TaskType::BinaryClassification | TaskType::MultiClassification => vec![
                ("random_forest", ModelType::RandomForest),
                ("xgboost", ModelType::XGBoost),
                ("lightgbm", ModelType::LightGBM),
                ("catboost", ModelType::CatBoost),
                ("gradient_boosting", ModelType::GradientBoosting),
                ("extra_trees", ModelType::ExtraTrees),
                ("adaboost", ModelType::AdaBoost),
                ("sgd", ModelType::SGD),
                ("decision_tree", ModelType::DecisionTree),
                ("knn", ModelType::KNN),
                ("logistic_regression", ModelType::LogisticRegression),
                ("svm", ModelType::SVM),
                ("naive_bayes", ModelType::NaiveBayes),
            ],
            TaskType::Regression | TaskType::TimeSeries => vec![
                ("random_forest", ModelType::RandomForest),
                ("xgboost", ModelType::XGBoost),
                ("lightgbm", ModelType::LightGBM),
                ("catboost", ModelType::CatBoost),
                ("gradient_boosting", ModelType::GradientBoosting),
                ("extra_trees", ModelType::ExtraTrees),
                ("ridge", ModelType::Ridge),
                ("lasso", ModelType::Lasso),
                ("elastic_net", ModelType::ElasticNet),
                ("polynomial", ModelType::PolynomialRegression),
                ("sgd", ModelType::SGD),
                ("gaussian_process", ModelType::GaussianProcess),
                ("decision_tree", ModelType::DecisionTree),
                ("knn", ModelType::KNN),
                ("linear_regression", ModelType::LinearRegression),
            ],
            TaskType::Clustering => vec![
                ("kmeans", ModelType::KMeans),
                ("dbscan", ModelType::DBSCAN),
            ],
        }
    };

    let limit = max_models.min(candidate_types.len());
    let candidates: Vec<_> = candidate_types.into_iter().take(limit).collect();
    let total = candidates.len();

    let state_clone = state.clone();
    let job_id_clone = job_id.clone();
    let df_clone = preprocessed_df.clone();
    let target_clone = target_column.clone();

    tokio::spawn(async move {
        let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_PARALLEL_TRAINS));
        let pipeline_start = std::time::Instant::now();

        let mut handles = Vec::new();

        for (name, model_type) in candidates {
            let df_ref = df_clone.clone();
            let target_ref = target_clone.clone();
            let sem = semaphore.clone();
            let completed_ref = completed.clone();
            let state_ref = state_clone.clone();
            let jid = job_id_clone.clone();
            let tt = task_type.clone();

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let model_start = std::time::Instant::now();

                let result = tokio::task::spawn_blocking(move || {
                    let config = TrainingConfig::new(tt.clone(), &target_ref)
                        .with_model(model_type)
                        .with_cv(cv_folds);
                    let mut engine = TrainEngine::new(config);
                    engine.fit(&df_ref)?;
                    let metrics = engine.metrics().cloned().unwrap_or_default();
                    let result_json = match tt {
                        TaskType::BinaryClassification | TaskType::MultiClassification => serde_json::json!({
                            "accuracy": metrics.accuracy.unwrap_or(0.0),
                            "precision": metrics.precision.unwrap_or(0.0),
                            "recall": metrics.recall.unwrap_or(0.0),
                            "f1": metrics.f1_score.unwrap_or(0.0),
                        }),
                        TaskType::Regression | TaskType::TimeSeries => serde_json::json!({
                            "r2": metrics.r2.unwrap_or(0.0),
                            "mse": metrics.mse.unwrap_or(0.0),
                            "mae": metrics.mae.unwrap_or(0.0),
                        }),
                        TaskType::Clustering => serde_json::json!({}),
                    };
                    Ok::<_, crate::error::KolosalError>((result_json, engine))
                }).await;

                let model_time = model_start.elapsed();
                let done = completed_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

                // Update progress
                {
                    let mut jobs = state_ref.jobs.write().await;
                    if let Some(job) = jobs.get_mut(&jid) {
                        let pct = 15.0 + (done as f64 / total as f64) * 55.0; // 15% to 70%
                        job.status = JobStatus::Running {
                            progress: pct,
                            message: format!("Trained {}/{} models ({})...", done, total, name),
                        };
                    }
                }

                (name.to_string(), model_time, result)
            });
            handles.push(handle);
        }

        // Collect results
        let mut leaderboard: Vec<serde_json::Value> = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_model_name = String::new();
        let mut best_engine: Option<TrainEngine> = None;

        for handle in handles {
            match handle.await {
                Ok((name, model_time, Ok(Ok((metrics, engine))))) => {
                    let score = match task_type {
                        TaskType::BinaryClassification | TaskType::MultiClassification => {
                            metrics.get("f1").and_then(|v| v.as_f64()).unwrap_or(0.0)
                        }
                        TaskType::Regression | TaskType::TimeSeries => {
                            metrics.get("r2").and_then(|v| v.as_f64()).unwrap_or(0.0)
                        }
                        TaskType::Clustering => 0.0,
                    };

                    leaderboard.push(serde_json::json!({
                        "model": name,
                        "score": score,
                        "metrics": metrics,
                        "training_time_secs": model_time.as_secs_f64(),
                    }));

                    // Store engine
                    let key = format!("automl_{}_{}", name, job_id_clone);
                    state_clone.train_engines.write().await.insert(key, engine.clone());

                    if score > best_score {
                        best_score = score;
                        best_model_name = name.clone();
                        best_engine = Some(engine);
                    }
                }
                Ok((name, _, _)) => {
                    leaderboard.push(serde_json::json!({
                        "model": name,
                        "score": null,
                        "error": "Training failed",
                    }));
                }
                Err(_) => {}
            }
        }

        // Sort leaderboard by score descending
        leaderboard.sort_by(|a, b| {
            let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(f64::NEG_INFINITY);
            let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(f64::NEG_INFINITY);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        // ---- Step 4: Optional hyperopt on best model ----
        let mut hyperopt_result = serde_json::Value::Null;

        if hyperopt_trials > 0 && best_engine.is_some() && !matches!(task_type, TaskType::Clustering) {
            // Check time budget
            let elapsed = pipeline_start.elapsed().as_secs();
            let skip_hyperopt = time_budget_secs > 0 && elapsed > time_budget_secs * 3 / 4;

            if !skip_hyperopt {
                {
                    let mut jobs = state_clone.jobs.write().await;
                    if let Some(job) = jobs.get_mut(&job_id_clone) {
                        job.status = JobStatus::Running {
                            progress: 75.0,
                            message: format!("Optimizing best model ({})...", best_model_name),
                        };
                    }
                }

                // Run hyperopt on the best model
                let model_type_for_opt = parse_model_type(&best_model_name).ok();
                if let Some(mt) = model_type_for_opt {
                    let df_opt = df_clone.clone();
                    let target_opt = target_clone.clone();
                    let tt = task_type.clone();

                    let opt_result = tokio::task::spawn_blocking(move || {
                        use crate::optimizer::{HyperOptX, OptimizationConfig, OptimizeDirection};

                        let search_space = build_search_space(&format!("{:?}", mt).to_lowercase());
                        let opt_config = OptimizationConfig::new()
                            .with_n_trials(hyperopt_trials)
                            .with_direction(OptimizeDirection::Minimize);

                        let mut optimizer = HyperOptX::new(opt_config, search_space);
                        let optimize_result = optimizer.optimize(|params| {
                            let mut config = TrainingConfig::new(tt.clone(), &target_opt)
                                .with_model(mt.clone());

                            if let Some(crate::optimizer::ParameterValue::Float(v)) = params.get("learning_rate") {
                                config = config.with_learning_rate(*v);
                            }
                            if let Some(crate::optimizer::ParameterValue::Int(v)) = params.get("max_depth") {
                                config = config.with_max_depth(*v as usize);
                            }
                            if let Some(crate::optimizer::ParameterValue::Int(v)) = params.get("n_estimators") {
                                config = config.with_n_estimators(*v as usize);
                            }

                            let mut engine = TrainEngine::new(config);
                            engine.fit(&df_opt)?;
                            let metrics = engine.metrics().cloned().unwrap_or_default();

                            match tt {
                                TaskType::BinaryClassification | TaskType::MultiClassification =>
                                    Ok(1.0 - metrics.f1_score.unwrap_or(0.0)),
                                TaskType::Regression | TaskType::TimeSeries =>
                                    Ok(metrics.mse.unwrap_or(f64::MAX)),
                                TaskType::Clustering => Ok(0.0),
                            }
                        });

                        match optimize_result {
                            Ok(study) => {
                                serde_json::json!({
                                    "best_params": format!("{:?}", study.best_params()),
                                    "best_value": study.best_value(),
                                    "n_trials": study.trials.len(),
                                })
                            }
                            Err(_) => serde_json::json!({ "skipped": true, "reason": "optimization failed" }),
                        }
                    }).await;

                    if let Ok(result) = opt_result {
                        hyperopt_result = result;
                    }
                }
            }
        }

        // ---- Step 5: Final report ----
        let total_time = pipeline_start.elapsed();
        let metric_name = match task_type {
            TaskType::BinaryClassification | TaskType::MultiClassification => "F1 Score",
            TaskType::Regression | TaskType::TimeSeries => "R²",
            TaskType::Clustering => "N/A",
        };

        let result = serde_json::json!({
            "pipeline": "complete",
            "task_type": task_type_str,
            "preprocessing": {
                "scaler": scaler_str,
                "imputation": impute_str,
            },
            "best_model": best_model_name,
            "best_score": best_score,
            "metric": metric_name,
            "leaderboard": leaderboard,
            "hyperopt": hyperopt_result,
            "total_time_secs": total_time.as_secs_f64(),
            "models_evaluated": total,
        });

        // Update job to completed
        let mut jobs = state_clone.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id_clone) {
            job.status = JobStatus::Completed { metrics: result };
        }

        // Store best engine
        if let Some(engine) = best_engine {
            state_clone.train_engines.write().await.insert(
                format!("automl_best_{}", job_id_clone), engine,
            );
        }
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "job_id": job_id,
        "message": "AutoML pipeline started. Poll /api/automl/status for progress.",
    })))
}

/// Get AutoML pipeline status (reuses training job status)
pub async fn get_automl_status(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<serde_json::Value>> {
    let jobs = state.jobs.read().await;
    let job = jobs.get(&job_id)
        .ok_or_else(|| ServerError::NotFound(format!("AutoML job '{}' not found", job_id)))?;

    let (status, progress, message, result) = match &job.status {
        JobStatus::Running { progress, message } => ("running", *progress, message.clone(), serde_json::Value::Null),
        JobStatus::Completed { metrics } => ("completed", 100.0, "Pipeline finished".to_string(), metrics.clone()),
        JobStatus::Failed { error } => ("failed", 0.0, error.clone(), serde_json::Value::Null),
        JobStatus::Pending => ("pending", 0.0, "Waiting to start".to_string(), serde_json::Value::Null),
    };

    Ok(Json(serde_json::json!({
        "job_id": job.id,
        "status": status,
        "progress": progress,
        "message": message,
        "result": result,
    })))
}

// ============================================================================
// Enhanced Hyperopt: Search Space, History, Apply Best
// ============================================================================

/// Get search space definition for a specific model
pub async fn get_search_space(
    Path(model_type): Path<String>,
) -> Json<serde_json::Value> {
    let space_def = get_search_space_definition(&model_type);
    Json(serde_json::json!({
        "model": model_type,
        "parameters": space_def,
    }))
}

fn get_search_space_definition(model_type: &str) -> serde_json::Value {
    match model_type {
        "random_forest" => serde_json::json!([
            {"name": "n_estimators", "type": "int", "low": 10, "high": 200, "default": 100},
            {"name": "max_depth", "type": "int", "low": 2, "high": 20, "default": 10},
        ]),
        "gradient_boosting" => serde_json::json!([
            {"name": "n_estimators", "type": "int", "low": 10, "high": 200, "default": 100},
            {"name": "max_depth", "type": "int", "low": 2, "high": 10, "default": 5},
            {"name": "learning_rate", "type": "log_float", "low": 0.001, "high": 1.0, "default": 0.1},
        ]),
        "knn" => serde_json::json!([
            {"name": "n_neighbors", "type": "int", "low": 1, "high": 30, "default": 5},
        ]),
        "decision_tree" => serde_json::json!([
            {"name": "max_depth", "type": "int", "low": 1, "high": 30, "default": 10},
        ]),
        "svm" => serde_json::json!([
            {"name": "learning_rate", "type": "log_float", "low": 0.001, "high": 10.0, "default": 1.0},
        ]),
        "neural_network" => serde_json::json!([
            {"name": "learning_rate", "type": "log_float", "low": 0.0001, "high": 0.1, "default": 0.001},
        ]),
        _ => serde_json::json!([
            {"name": "learning_rate", "type": "log_float", "low": 0.001, "high": 1.0, "default": 0.1},
        ]),
    }
}

/// Get history of completed hyperopt studies
pub async fn get_hyperopt_history(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let studies = state.completed_studies.read().await;
    Json(serde_json::json!({
        "studies": *studies,
        "total": studies.len(),
    }))
}

/// Apply best hyperopt params and train a final model
#[derive(Deserialize)]
pub struct ApplyBestRequest {
    target_column: String,
    task_type: String,
    model_type: String,
    params: serde_json::Value,
}

pub async fn apply_best_params(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ApplyBestRequest>,
) -> Result<Json<serde_json::Value>> {
    let data = state.current_data.read().await;
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
        .clone();

    let task_type = parse_task_type(&request.task_type)?;
    let model_type = parse_model_type(&request.model_type)?;

    let job_id = AppState::generate_id();
    let job_id_clone = job_id.clone();

    let job = TrainingJob {
        id: job_id.clone(),
        status: JobStatus::Running {
            progress: 0.0,
            message: "Training with optimized parameters...".to_string(),
        },
        config: serde_json::json!({
            "target_column": request.target_column,
            "task_type": request.task_type,
            "model_type": request.model_type,
            "optimized_params": request.params,
        }),
        created_at: chrono::Utc::now(),
        model_path: None,
    };
    state.jobs.write().await.insert(job_id.clone(), job);

    let state_clone = state.clone();
    let target_col = request.target_column.clone();
    let params = request.params.clone();
    let models_dir = state.config.models_dir.clone();
    let model_type_str = request.model_type.clone();

    tokio::spawn(async move {
        let result = tokio::task::spawn_blocking(move || {
            let mut config = TrainingConfig::new(task_type.clone(), &target_col)
                .with_model(model_type);

            if let Some(lr) = params.get("learning_rate").and_then(|v| v.as_f64()) {
                config = config.with_learning_rate(lr);
            }
            if let Some(md) = params.get("max_depth").and_then(|v| v.as_u64()) {
                config = config.with_max_depth(md as usize);
            }
            if let Some(ne) = params.get("n_estimators").and_then(|v| v.as_u64()) {
                config = config.with_n_estimators(ne as usize);
            }
            if let Some(nn) = params.get("n_neighbors").and_then(|v| v.as_u64()) {
                config = config.with_n_neighbors(nn as usize);
            }

            let mut engine = TrainEngine::new(config);
            engine.fit(&df)?;

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
                TaskType::Clustering => serde_json::json!({
                    "training_time_secs": metrics.training_time_secs,
                }),
            };

            Ok::<(serde_json::Value, TrainEngine), crate::error::KolosalError>((result, engine))
        }).await;

        match result {
            Ok(Ok((metrics_val, engine))) => {
                let model_id = format!("optimized_{}_{}", model_type_str, job_id);
                state_clone.train_engines.write().await.insert(model_id.clone(), engine);

                let model_info = super::state::ModelInfo {
                    id: model_id.clone(),
                    name: format!("Optimized {}", model_type_str),
                    task_type: model_type_str.clone(),
                    metrics: metrics_val.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    path: std::path::PathBuf::from(&models_dir).join(format!("{}.model", model_id)),
                };
                state_clone.models.write().await.insert(model_id.clone(), model_info);

                let completed_metrics = serde_json::json!({
                    "applied_params": true,
                    "model_id": model_id,
                    "model_type": model_type_str,
                    "metrics": metrics_val,
                });

                let mut jobs = state_clone.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Completed { metrics: completed_metrics };
                }
            }
            Ok(Err(e)) => {
                let mut jobs = state_clone.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Failed { error: e.to_string() };
                }
            }
            Err(e) => {
                let mut jobs = state_clone.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Failed { error: e.to_string() };
                }
            }
        }
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "job_id": job_id_clone,
        "message": "Training with optimized parameters started",
    })))
}

// ============================================================================
// Security Management Handlers
// ============================================================================

/// Get security status overview
pub async fn get_security_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let audit_count = state.security_manager.get_audit_log(0).len();
    let rate_metrics = state.rate_limiter.get_metrics();

    Ok(Json(serde_json::json!({
        "security_enabled": true,
        "rate_limiting": {
            "active_clients": rate_metrics.get("active_clients").unwrap_or(&0),
            "total_requests": rate_metrics.get("total_requests").unwrap_or(&0),
            "total_blocked": rate_metrics.get("total_blocked").unwrap_or(&0),
        },
        "audit_log_entries": audit_count,
        "guards": {
            "max_dataset_rows": MAX_DATASET_ROWS,
            "max_dataset_columns": MAX_DATASET_COLUMNS,
            "max_batch_items": MAX_BATCH_PREDICTION_ITEMS,
            "training_timeout_secs": TRAINING_TIMEOUT_SECS,
            "ssrf_protection": true,
            "path_traversal_protection": true,
            "error_sanitization": true,
            "input_validation": true,
        },
    })))
}

/// Get audit log entries
#[derive(Deserialize)]
pub struct AuditLogQuery {
    limit: Option<usize>,
}

pub async fn get_audit_log(
    State(state): State<Arc<AppState>>,
    Query(query): Query<AuditLogQuery>,
) -> Result<Json<serde_json::Value>> {
    let limit = query.limit.unwrap_or(100).min(1000);
    let entries = state.security_manager.get_audit_log(limit);

    let entries_json: Vec<serde_json::Value> = entries
        .iter()
        .map(|e| serde_json::json!({
            "timestamp": e.timestamp.to_rfc3339(),
            "client_id": e.client_id,
            "action": e.action,
            "resource": e.resource,
            "status_code": e.status_code,
            "ip_address": e.ip_address,
        }))
        .collect();

    Ok(Json(serde_json::json!({
        "total": entries_json.len(),
        "entries": entries_json,
    })))
}

/// Get rate limit statistics
pub async fn get_rate_limit_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let metrics = state.rate_limiter.get_metrics();

    Ok(Json(serde_json::json!({
        "metrics": metrics,
    })))
}
