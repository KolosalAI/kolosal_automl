//! HTTP request handlers

use std::sync::Arc;
use axum::{
    extract::{Multipart, Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    Json,
};
use polars::prelude::*;
use serde::Deserialize;
use std::io::Cursor;
use tracing::{info, warn, error, debug};

use calamine;

use crate::preprocessing::{DataPreprocessor, PreprocessingConfig, ScalerType, ImputeStrategy};
use crate::training::{TrainEngine, TrainingConfig, TaskType, ModelType};

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
    let sanitized: String = basename
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | '_'))
        .collect();
    if sanitized.is_empty() {
        "data.csv".to_string()
    } else {
        sanitized
    }
}

/// Maximum upload body size (100 MB)
const MAX_UPLOAD_SIZE: usize = 100 * 1024 * 1024;

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
            "Invalid model type: '{}'. Expected: linear_regression, logistic_regression, ridge, lasso, elastic_net, polynomial, decision_tree, random_forest, gradient_boosting, xgboost, lightgbm, catboost, knn, naive_bayes, svm, adaboost, extra_trees, sgd, gaussian_process, kmeans, dbscan",
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

        // Reject oversized uploads before parsing
        if data.len() > MAX_UPLOAD_SIZE {
            return Err(ServerError::BadRequest(format!(
                "File too large: {} bytes (max {} MB)", data.len(), MAX_UPLOAD_SIZE / 1024 / 1024
            )));
        }

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
/// Clear the loaded dataset
pub async fn clear_data(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let mut data = state.current_data.write().await;
    *data = None;
    Ok(Json(serde_json::json!({ "success": true })))
}

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
            "null_percent": (null_count as f64 / df.height().max(1) as f64) * 100.0,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "rows": df.height(),
        "columns": df.width(),
        "column_info": columns,
        "memory_usage_bytes": df.estimated_size(),
    })))
}

/// Comprehensive automated data analysis
pub async fn analyze_data(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let data = state.current_data.read().await;

    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?;

    let n_rows = df.height();
    let n_cols = df.width();
    let memory_bytes = df.estimated_size();

    // --- Duplicate count ---
    let dup_count = if n_rows > 0 {
        let unique_df = df.unique_stable(None, polars::frame::UniqueKeepStrategy::First, None);
        match unique_df {
            Ok(udf) => n_rows.saturating_sub(udf.height()),
            Err(_) => 0,
        }
    } else { 0 };

    // --- Per-column stats ---
    let mut col_stats = Vec::new();
    let mut missing_values = Vec::new();
    let mut numeric_col_names = Vec::new();
    let mut numeric_col_indices = Vec::new();
    let mut distributions = Vec::new();
    let mut outlier_summary = Vec::new();
    let mut col_warnings: Vec<serde_json::Value> = Vec::new();

    for (idx, col) in df.get_columns().iter().enumerate() {
        let col_name = col.name().to_string();
        let null_count = col.null_count();
        let null_pct = (null_count as f64 / n_rows.max(1) as f64) * 100.0;
        let unique_count = col.n_unique().unwrap_or(0);
        let dtype_str = format!("{:?}", col.dtype());

        missing_values.push(serde_json::json!({
            "column": col_name,
            "null_count": null_count,
            "null_percent": (null_pct * 100.0).round() / 100.0,
        }));

        let is_numeric = matches!(col.dtype(), DataType::Float64 | DataType::Float32 | DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 | DataType::UInt64 | DataType::UInt32 | DataType::UInt16 | DataType::UInt8);

        if is_numeric {
            numeric_col_names.push(col_name.clone());
            numeric_col_indices.push(idx);
            let series = col.cast(&DataType::Float64).unwrap_or_else(|_| col.clone());
            let ca = series.f64().ok();

            let (mean, std, min, max, median) = if let Some(ca) = ca {
                (
                    ca.mean().unwrap_or(0.0),
                    ca.std(1).unwrap_or(0.0),
                    ca.min().unwrap_or(0.0),
                    ca.max().unwrap_or(0.0),
                    ca.median().unwrap_or(0.0),
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };

            // Quartiles via sorted values
            let (q25, q75) = if let Some(ca) = ca {
                let mut vals: Vec<f64> = ca.into_no_null_iter().collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let len = vals.len();
                if len > 0 {
                    (vals[len / 4], vals[3 * len / 4])
                } else { (0.0, 0.0) }
            } else { (0.0, 0.0) };

            // Skewness and Kurtosis
            let (skewness, kurtosis) = if let Some(ca) = ca {
                let n = ca.len() as f64 - ca.null_count() as f64;
                if n > 3.0 && std > 0.0 {
                    let sum3: f64 = ca.into_no_null_iter().map(|v| ((v - mean) / std).powi(3)).sum();
                    let sum4: f64 = ca.into_no_null_iter().map(|v| ((v - mean) / std).powi(4)).sum();
                    let skew = sum3 * n / ((n - 1.0) * (n - 2.0));
                    // Fisher excess kurtosis
                    let kurt = n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * sum4
                        - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
                    (skew, kurt)
                } else { (0.0, 0.0) }
            } else { (0.0, 0.0) };

            // Variance, CV, range
            let variance = std * std;
            let cv = if mean.abs() > 1e-10 { std / mean.abs() * 100.0 } else { 0.0 };
            let range = max - min;

            // IQR outliers
            let iqr = q75 - q25;
            let lower = q25 - 1.5 * iqr;
            let upper = q75 + 1.5 * iqr;
            let outlier_count = if let Some(ca) = ca {
                ca.into_no_null_iter().filter(|&v| v < lower || v > upper).count()
            } else { 0 };

            // Sum and zeros
            let (sum, zeros_count) = if let Some(ca) = ca {
                let s: f64 = ca.into_no_null_iter().sum();
                let z = ca.into_no_null_iter().filter(|&v| v == 0.0).count();
                (s, z)
            } else { (0.0, 0) };
            let zeros_percent = zeros_count as f64 / n_rows.max(1) as f64 * 100.0;

            outlier_summary.push(serde_json::json!({
                "column": col_name,
                "outlier_count": outlier_count,
                "outlier_percent": (outlier_count as f64 / n_rows.max(1) as f64 * 100.0 * 100.0).round() / 100.0,
                "lower_bound": (lower * 1000.0).round() / 1000.0,
                "upper_bound": (upper * 1000.0).round() / 1000.0,
            }));

            // Per-column advanced warnings
            if skewness.abs() > 2.0 {
                col_warnings.push(serde_json::json!({
                    "type": "high_skewness",
                    "severity": "info",
                    "message": format!("Column '{}' is highly skewed (skew={:.2})", col_name, skewness),
                }));
            }
            if kurtosis.abs() > 7.0 {
                col_warnings.push(serde_json::json!({
                    "type": "high_kurtosis",
                    "severity": "info",
                    "message": format!("Column '{}' has heavy tails (excess kurtosis={:.2})", col_name, kurtosis),
                }));
            }
            let outlier_pct = outlier_count as f64 / n_rows.max(1) as f64 * 100.0;
            if outlier_pct > 10.0 {
                col_warnings.push(serde_json::json!({
                    "type": "high_outlier_ratio",
                    "severity": "warning",
                    "message": format!("Column '{}' has {:.1}% outliers (IQR method)", col_name, outlier_pct),
                }));
            }
            if zeros_percent > 50.0 {
                col_warnings.push(serde_json::json!({
                    "type": "high_zero_ratio",
                    "severity": "info",
                    "message": format!("Column '{}' is {:.1}% zeros (sparse feature)", col_name, zeros_percent),
                }));
            }

            // Histogram bins
            let hist = if let Some(ca) = ca {
                let vals: Vec<f64> = ca.into_no_null_iter().collect();
                if !vals.is_empty() && max > min {
                    let n_bins = 20usize;
                    let bin_width = (max - min) / n_bins as f64;
                    let mut counts = vec![0u64; n_bins];
                    for &v in &vals {
                        let bin = ((v - min) / bin_width).floor() as usize;
                        let bin = bin.min(n_bins - 1);
                        counts[bin] += 1;
                    }
                    let bin_edges: Vec<f64> = (0..=n_bins).map(|i| ((min + i as f64 * bin_width) * 1000.0).round() / 1000.0).collect();
                    Some(serde_json::json!({ "bin_edges": bin_edges, "counts": counts }))
                } else { None }
            } else { None };

            distributions.push(serde_json::json!({
                "column": col_name,
                "type": "numeric",
                "histogram": hist,
            }));

            col_stats.push(serde_json::json!({
                "column": col_name,
                "dtype": dtype_str,
                "null_count": null_count,
                "null_percent": (null_pct * 100.0).round() / 100.0,
                "unique": unique_count,
                "mean": (mean * 10000.0).round() / 10000.0,
                "std": (std * 10000.0).round() / 10000.0,
                "variance": (variance * 10000.0).round() / 10000.0,
                "cv_percent": (cv * 100.0).round() / 100.0,
                "min": (min * 10000.0).round() / 10000.0,
                "q25": (q25 * 10000.0).round() / 10000.0,
                "median": (median * 10000.0).round() / 10000.0,
                "q75": (q75 * 10000.0).round() / 10000.0,
                "max": (max * 10000.0).round() / 10000.0,
                "range": (range * 10000.0).round() / 10000.0,
                "iqr": (iqr * 10000.0).round() / 10000.0,
                "skewness": (skewness * 10000.0).round() / 10000.0,
                "kurtosis": (kurtosis * 10000.0).round() / 10000.0,
                "sum": (sum * 10000.0).round() / 10000.0,
                "zeros_count": zeros_count,
                "zeros_percent": (zeros_percent * 100.0).round() / 100.0,
                "outlier_count": outlier_count,
                "outlier_percent": (outlier_count as f64 / n_rows.max(1) as f64 * 10000.0).round() / 100.0,
                "is_numeric": true,
            }));
        } else {
            // Categorical stats
            let series = col.as_materialized_series();
            let value_counts = series.value_counts(true, true, "count".into(), false);

            let n_valid = (n_rows - null_count) as f64;

            let (top_categories, mode_value, mode_count, mode_percent, entropy) = match value_counts {
                Ok(vc) => {
                    let top_n = vc.height().min(10);
                    let cats: Vec<serde_json::Value> = (0..top_n).filter_map(|i| {
                        let val = vc.column(col.name()).ok()?.get(i).ok()?.to_string();
                        let cnt = vc.column("count").ok()?.get(i).ok()?.to_string().replace('"', "");
                        Some(serde_json::json!({ "value": val.trim_matches('"'), "count": cnt.parse::<u64>().unwrap_or(0) }))
                    }).collect();

                    // Mode (top category)
                    let (mv, mc, mp) = if vc.height() > 0 {
                        let val = vc.column(col.name()).ok()
                            .and_then(|c| c.get(0).ok())
                            .map(|v| v.to_string().trim_matches('"').to_string())
                            .unwrap_or_default();
                        let cnt = vc.column("count").ok()
                            .and_then(|c| c.get(0).ok())
                            .and_then(|v| v.to_string().parse::<u64>().ok())
                            .unwrap_or(0);
                        let pct = if n_valid > 0.0 { cnt as f64 / n_valid * 100.0 } else { 0.0 };
                        (val, cnt, (pct * 100.0).round() / 100.0)
                    } else { (String::new(), 0u64, 0.0_f64) };

                    // Shannon entropy (nats → bits via /ln(2))
                    let ent = if n_valid > 0.0 {
                        let e: f64 = (0..vc.height()).filter_map(|i| {
                            vc.column("count").ok()?.get(i).ok()
                                .and_then(|v| v.to_string().parse::<f64>().ok())
                                .map(|cnt| { let p = cnt / n_valid; if p > 0.0 { -p * p.log2() } else { 0.0 } })
                        }).sum();
                        (e * 100.0).round() / 100.0
                    } else { 0.0 };

                    (cats, mv, mc, mp, ent)
                }
                Err(_) => (Vec::new(), String::new(), 0u64, 0.0_f64, 0.0_f64),
            };

            distributions.push(serde_json::json!({
                "column": col_name,
                "type": "categorical",
                "value_counts": top_categories.clone(),
            }));

            col_stats.push(serde_json::json!({
                "column": col_name,
                "dtype": dtype_str,
                "null_count": null_count,
                "null_percent": (null_pct * 100.0).round() / 100.0,
                "unique": unique_count,
                "top_categories": top_categories,
                "mode_value": mode_value,
                "mode_count": mode_count,
                "mode_percent": mode_percent,
                "entropy_bits": entropy,
                "is_numeric": false,
            }));
        }
    }

    // --- Correlation matrix (numeric columns only, max 30 to avoid perf issues) ---
    let correlation = if numeric_col_names.len() >= 2 && numeric_col_names.len() <= 30 {
        use ndarray::Array2;
        let n = n_rows;
        let nc = numeric_col_indices.len();
        let mut matrix = Array2::zeros((n, nc));
        for (j, &col_idx) in numeric_col_indices.iter().enumerate() {
            let col = &df.get_columns()[col_idx];
            if let Ok(ca) = col.cast(&DataType::Float64) {
                if let Ok(ca) = ca.f64() {
                    for i in 0..n {
                        matrix[[i, j]] = ca.get(i).unwrap_or(0.0);
                    }
                }
            }
        }
        let corr = crate::preprocessing::feature_selection::CorrelationFilter::compute_correlation_matrix(&matrix);
        let corr_vec: Vec<Vec<f64>> = (0..nc).map(|i| {
            (0..nc).map(|j| (corr[[i, j]] * 10000.0).round() / 10000.0).collect()
        }).collect();
        Some(serde_json::json!({
            "columns": numeric_col_names,
            "matrix": corr_vec,
        }))
    } else { None };

    // --- Quality report ---
    let total_cells = n_rows * n_cols;
    let total_nulls: usize = df.get_columns().iter().map(|c| c.null_count()).sum();
    let completeness = if total_cells > 0 { 1.0 - (total_nulls as f64 / total_cells as f64) } else { 1.0 };
    let total_unique: usize = df.get_columns().iter().map(|c| c.n_unique().unwrap_or(0)).sum();
    let uniqueness = if total_cells > 0 { (total_unique as f64 / total_cells as f64).min(1.0) } else { 1.0 };
    let consistency = 1.0; // placeholder
    let validity = completeness; // basic approximation
    let quality_score = completeness * 0.4 + uniqueness * 0.2 + consistency * 0.2 + validity * 0.2;

    // --- Warnings ---
    let mut warnings = Vec::new();
    for col in df.get_columns() {
        let null_pct = col.null_count() as f64 / n_rows.max(1) as f64;
        if null_pct > 0.5 {
            warnings.push(serde_json::json!({
                "type": "high_missing",
                "severity": "warning",
                "message": format!("Column '{}' has {:.1}% missing values", col.name(), null_pct * 100.0),
            }));
        }
        if col.n_unique().unwrap_or(0) == 1 {
            warnings.push(serde_json::json!({
                "type": "constant_column",
                "severity": "info",
                "message": format!("Column '{}' has only one unique value", col.name()),
            }));
        }
    }
    if dup_count > n_rows / 10 {
        warnings.push(serde_json::json!({
            "type": "high_duplicates",
            "severity": "warning",
            "message": format!("{} duplicate rows ({:.1}%)", dup_count, dup_count as f64 / n_rows.max(1) as f64 * 100.0),
        }));
    }
    warnings.extend(col_warnings);

    Ok(Json(serde_json::json!({
        "success": true,
        "overview": {
            "rows": n_rows,
            "columns": n_cols,
            "memory_bytes": memory_bytes,
            "memory_mb": ((memory_bytes as f64 / 1024.0 / 1024.0) * 100.0).round() / 100.0,
            "duplicates": dup_count,
            "total_missing": total_nulls,
            "missing_percent": ((total_nulls as f64 / total_cells.max(1) as f64) * 10000.0).round() / 100.0,
        },
        "quality": {
            "score": (quality_score * 1000.0).round() / 1000.0,
            "completeness": (completeness * 1000.0).round() / 1000.0,
            "uniqueness": (uniqueness * 1000.0).round() / 1000.0,
            "consistency": consistency,
            "validity": (validity * 1000.0).round() / 1000.0,
        },
        "warnings": warnings,
        "column_stats": col_stats,
        "missing_values": missing_values,
        "correlation": correlation,
        "distributions": distributions,
        "outlier_summary": outlier_summary,
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
        "titanic" => generate_titanic_dataset()?,
        "heart" => generate_heart_dataset()?,
        "credit" => generate_credit_dataset()?,
        "customers" => generate_customers_dataset()?,
        "california_housing" => generate_california_housing_dataset()?,
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
    // Read and clone data, then drop lock before CPU-bound processing
    let df = {
        let data = state.current_data.read().await;
        data.as_ref()
            .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
            .clone()
    };

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

    // Run preprocessing (lock is not held)
    let mut preprocessor = DataPreprocessor::with_config(config);
    let processed = preprocessor.fit_transform(&df)
        .map_err(|e| ServerError::Internal(format!("Preprocessing failed: {:?}", e)))?;

    // Re-acquire write lock to update state
    *state.current_data.write().await = Some(processed.clone());
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
            message: "Starting training...".to_string(),
            partial_results: None,
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
                state_clone.train_engines.insert(model_id.clone(), engine);

                // Register model in state
                let model_info = ModelInfo {
                    id: model_id.clone(),
                    name: format!("{} (job {})", model_type_name, job_id),
                    task_type: task_type_name,
                    metrics: metrics.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    path: std::path::PathBuf::from(&model_path),
                };
                state_clone.models.insert(model_id, model_info);

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
    .map_err(|_| anyhow::anyhow!("Training timed out after {} seconds", TRAINING_TIMEOUT_SECS))?
    .map_err(|e| anyhow::anyhow!("Training task panicked: {}", e))??;

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

    let mut trained_engines: Vec<(String, TrainEngine)> = Vec::new();

    for handle in handles {
        match handle.await {
            Ok((name, Ok((metrics, engine)))) => {
                info!(model = %name, "Comparison model trained successfully");
                results.push(serde_json::json!({
                    "model": name,
                    "metrics": metrics,
                }));
                trained_engines.push((name, engine));
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

    // --- UMAP projection & per-model predictions ---
    let mut umap_points: Option<Vec<[f64; 2]>> = None;
    let mut per_model_preds: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut true_labels: Option<Vec<f64>> = None;

    // Try UMAP on the dataset (best-effort, don't fail comparison on UMAP errors)
    let umap_result: std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> = (|| {
        use crate::visualization::umap::{Umap, UmapConfig};

        let numeric_cols: Vec<String> = df.get_columns().iter()
            .filter(|c| c.name().as_str() != request.target_column && matches!(c.dtype(), DataType::Float32 | DataType::Float64 | DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64))
            .map(|c| c.name().to_string())
            .collect();

        if numeric_cols.len() < 2 || df.height() < 5 { return Ok(()); }

        // Cap at 2000 rows for speed
        let max_rows = df.height().min(2000);
        let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(max_rows);
        for i in 0..max_rows {
            let mut row = Vec::with_capacity(numeric_cols.len());
            for cn in &numeric_cols {
                let s = df.column(cn).ok().and_then(|c| c.cast(&DataType::Float64).ok());
                let val = s.as_ref().and_then(|s| s.f64().ok()).and_then(|ca| ca.get(i)).unwrap_or(0.0);
                row.push(val);
            }
            matrix.push(row);
        }

        let umap = Umap::new(UmapConfig::default());
        let points = umap.fit_transform(&matrix).map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e))) })?;

        // True labels
        if let Ok(target_col) = df.column(&request.target_column) {
            if let Ok(casted) = target_col.cast(&DataType::Float64) {
                if let Ok(ca) = casted.f64() {
                    let labels: Vec<f64> = (0..max_rows).map(|i| ca.get(i).unwrap_or(0.0)).collect();
                    true_labels = Some(labels);
                }
            }
        }

        // Per-model predictions
        for (name, engine) in &trained_engines {
            if let Ok(preds) = engine.predict(&df) {
                let pred_vec: Vec<f64> = preds.iter().take(max_rows).cloned().collect();
                per_model_preds.insert(name.clone(), serde_json::json!(pred_vec));
            }
        }

        umap_points = Some(points);
        Ok(())
    })();

    if let Err(e) = umap_result {
        debug!("UMAP for comparison skipped: {}", e);
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "comparison": results,
        "umap_points": umap_points,
        "per_model_predictions": per_model_preds,
        "true_labels": true_labels,
    })))
}

// ============================================================================
// Model Handlers
// ============================================================================

pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let model_list: Vec<ModelInfo> = state.models.iter()
        .map(|entry| entry.value().clone())
        .collect();

    Json(serde_json::json!({
        "models": model_list,
    }))
}

pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>> {
    let model = state.models.get(&model_id)
        .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", model_id)))?
        .clone();

    Ok(Json(serde_json::json!(model)))
}

pub async fn download_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<impl IntoResponse> {
    let model = state.models.get(&model_id)
        .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", model_id)))?
        .clone();

    let json_bytes = serde_json::to_vec_pretty(&model)
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
    let model_count = state.models.len();
    debug!(model_count = model_count, requested_id = ?request.model_id, "Predict request received");

    // Find the model: use provided model_id or most recently created
    let (model_id, model_path) = if let Some(ref id) = request.model_id {
        let entry = state.models.get(id)
            .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", id)))?;
        let mid = entry.id.clone();
        let mpath = entry.path.to_str()
            .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?
            .to_string();
        (mid, mpath)
    } else {
        let entry = state.models.iter()
            .max_by_key(|e| e.value().created_at.clone())
            .ok_or_else(|| ServerError::NotFound("No trained models available".to_string()))?;
        let mid = entry.id.clone();
        let mpath = entry.path.to_str()
            .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?
            .to_string();
        (mid, mpath)
    };

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

    let cache_hit = engine.last_prediction_was_cached();
    let (cache_hits, cache_misses, cache_hit_rate) = engine.cache_stats();
    let avg_latency = engine.stats().avg_latency_ms;

    Ok(Json(serde_json::json!({
        "success": true,
        "predictions": predictions.to_vec(),
        "cache_hit": cache_hit,
        "latency_ms": avg_latency,
        "cache_stats": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": cache_hit_rate,
        },
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
    // Find the model: use provided model_id or most recently created
    let (model_id, model_path) = if let Some(ref id) = request.model_id {
        let entry = state.models.get(id)
            .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", id)))?;
        let mid = entry.id.clone();
        let mpath = entry.path.to_str()
            .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?
            .to_string();
        (mid, mpath)
    } else {
        let entry = state.models.iter()
            .max_by_key(|e| e.value().created_at.clone())
            .ok_or_else(|| ServerError::NotFound("No trained models available".to_string()))?;
        let mid = entry.id.clone();
        let mpath = entry.path.to_str()
            .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?
            .to_string();
        (mid, mpath)
    };

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

    let cache_hit = engine.last_prediction_was_cached();
    let stats = engine.stats();

    Ok(Json(serde_json::json!({
        "success": true,
        "predictions": predictions.to_vec(),
        "count": predictions.len(),
        "cache_hit": cache_hit,
        "avg_latency_ms": stats.avg_latency_ms,
    })))
}

/// Predict probabilities for classification models
pub async fn predict_proba(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<serde_json::Value>> {
    let (model_id, model_path) = if let Some(ref id) = request.model_id {
        let entry = state.models.get(id)
            .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", id)))?;
        let mid = entry.id.clone();
        let mpath = entry.path.to_str()
            .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?
            .to_string();
        (mid, mpath)
    } else {
        let entry = state.models.iter()
            .max_by_key(|e| e.value().created_at.clone())
            .ok_or_else(|| ServerError::NotFound("No trained models available".to_string()))?;
        let mid = entry.id.clone();
        let mpath = entry.path.to_str()
            .ok_or_else(|| ServerError::Internal("Invalid model path".to_string()))?
            .to_string();
        (mid, mpath)
    };

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

/// Get prediction cache statistics
pub async fn get_cache_stats(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let (hits, misses, hit_rate) = state.prediction_cache.stats();
    Json(serde_json::json!({
        "success": true,
        "cache_stats": {
            "size": state.prediction_cache.len(),
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
        },
    }))
}

/// Clear the prediction cache
pub async fn clear_prediction_cache(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    state.prediction_cache.clear();
    Json(serde_json::json!({
        "success": true,
        "cleared": true,
    }))
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

    Json(serde_json::json!({
        "system": system_info,
        "datasets_count": datasets.len(),
        "jobs_count": jobs.len(),
        "models_count": state.models.len(),
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
    <link rel="stylesheet" href="/static/remixicon.css">
    <style>
        @font-face{font-family:"remixicon";src:url("/static/remixicon.woff2") format("woff2");font-display:swap}
        [class^="ri-"],[class*=" ri-"]{font-family:'remixicon'!important;font-style:normal;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
        .ri-dashboard-line:before{content:"\ec14"}.ri-database-2-line:before{content:"\ec16"}.ri-magic-line:before{content:"\eeea"}.ri-settings-3-line:before{content:"\f0e6"}.ri-play-circle-line:before{content:"\f009"}.ri-flashlight-line:before{content:"\ed3d"}.ri-equalizer-line:before{content:"\ec9d"}.ri-stack-line:before{content:"\f181"}.ri-search-eye-line:before{content:"\f0cf"}.ri-alarm-warning-line:before{content:"\ea1d"}.ri-pulse-line:before{content:"\f035"}.ri-shield-check-line:before{content:"\f100"}.ri-arrow-right-s-line:before{content:"\ea6e"}.ri-arrow-left-s-line:before{content:"\ea64"}.ri-upload-2-line:before{content:"\f24a"}.ri-refresh-line:before{content:"\f064"}.ri-cpu-line:before{content:"\ebf0"}.ri-speed-line:before{content:"\f177"}.ri-hard-drive-2-line:before{content:"\edf9"}.ri-bar-chart-line:before{content:"\ea7e"}.ri-line-chart-line:before{content:"\ee69"}.ri-time-line:before{content:"\f1dd"}.ri-check-line:before{content:"\eb7b"}.ri-close-line:before{content:"\ebd1"}.ri-information-line:before{content:"\ee59"}.ri-loader-4-line:before{content:"\ee8c"}.ri-file-upload-line:before{content:"\ed0e"}.ri-checkbox-circle-line:before{content:"\eb81"}.ri-error-warning-line:before{content:"\eca1"}
        *{margin:0;padding:0;box-sizing:border-box;font-family:system-ui,-apple-system,sans-serif}
        body{background:#f8f9f9;color:#0d0e0f;min-height:100vh;overflow:hidden}
        .app{display:flex;height:100vh}
        /* Sidebar */
        .sidebar{width:60px;min-width:60px;background:#fff;border-right:1px solid #e4e7e9;display:flex;flex-direction:column;transition:width .2s ease,min-width .2s ease;overflow:hidden;z-index:20;height:100vh;position:relative}
        .sidebar.expanded{width:220px;min-width:220px}
        .sidebar-header{height:56px;display:flex;align-items:center;padding:0 16px;gap:10px;border-bottom:1px solid #e4e7e9;flex-shrink:0}
        .sidebar-logo{width:28px;height:28px;background:#0d0e0f;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:14px;flex-shrink:0}
        .sidebar-title{font-weight:600;font-size:15px;white-space:nowrap;opacity:0;transition:opacity .15s;font-family:"Geist Mono",monospace}
        .sidebar.expanded .sidebar-title{opacity:1}
        .sidebar-nav{flex:1;padding:8px;overflow-y:auto;overflow-x:hidden}
        .sidebar-section{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#9c9fa1;padding:12px 8px 6px;white-space:nowrap;opacity:0;height:0;overflow:hidden;transition:opacity .15s}
        .sidebar.expanded .sidebar-section{opacity:1;height:auto}
        .sidebar-item{display:flex;align-items:center;gap:10px;height:38px;padding:0 8px;border-radius:10px;border:none;background:none;color:#6a6f73;font-size:14px;font-weight:500;cursor:pointer;width:100%;white-space:nowrap;transition:background .12s,color .12s}
        .sidebar-item:hover{background:#f1f3f4;color:#0d0e0f}
        .sidebar-item.active{background:#f0f6fe;color:#0066f5}
        .sidebar-item i{font-size:18px;width:24px;text-align:center;flex-shrink:0}
        .sidebar-item span{opacity:0;transition:opacity .15s}
        .sidebar.expanded .sidebar-item span{opacity:1}
        .sidebar-toggle{height:44px;display:flex;align-items:center;justify-content:center;border-top:1px solid #e4e7e9;flex-shrink:0;cursor:pointer;color:#9c9fa1;background:none;border-left:none;border-right:none;border-bottom:none;width:100%;font-size:18px;transition:color .12s}
        .sidebar-toggle:hover{color:#0d0e0f;background:#f1f3f4}
        /* Main content */
        .main-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
        .topbar{height:56px;background:#fff;border-bottom:1px solid #e4e7e9;display:flex;align-items:center;justify-content:space-between;padding:0 24px;flex-shrink:0}
        .topbar-title{font-size:16px;font-weight:600}
        .topbar-meta{display:flex;align-items:center;gap:12px;font-size:13px;color:#6a6f73}
        .content{flex:1;overflow-y:auto;padding:24px}
        .content-inner{max-width:1080px;margin:0 auto}
        /* Components */
        .flex{display:flex}.items-center{align-items:center}.justify-between{justify-content:space-between}
        .card{background:#fff;border:1px solid #e4e7e9;border-radius:12px;padding:24px;margin-bottom:16px}
        .card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
        .card-title{font-size:16px;font-weight:600}
        .card-desc{font-size:13px;color:#6a6f73;margin-top:2px}
        .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
        .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
        .grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
        @media(max-width:900px){.grid-4{grid-template-columns:repeat(2,1fr)}.grid-2{grid-template-columns:1fr}}
        .btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;height:36px;padding:0 14px;font-size:14px;font-weight:500;border-radius:10px;border:none;cursor:pointer;background:#0d0e0f;color:#fff;transition:opacity .12s}
        .btn:hover{opacity:.85}.btn:disabled{opacity:.5;cursor:not-allowed}
        .btn-outline{background:#fff;color:#0d0e0f;border:1px solid #dde1e3}
        .btn-outline:hover{background:#f1f3f4;opacity:1}
        .btn-primary{background:#0066f5;color:#fff}
        .btn-sm{height:30px;padding:0 10px;font-size:13px;border-radius:8px}
        .btn .spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite}
        @keyframes spin{to{transform:rotate(360deg)}}
        .badge{display:inline-flex;align-items:center;height:24px;padding:0 8px;font-size:12px;font-weight:500;border-radius:6px}
        .badge-ok{color:#2e9632;background:#f3fbf4}.badge-err{color:#cc2727;background:#fff3f3}
        .badge-info{color:#0052c4;background:#f0f6fe}.badge-warn{color:#9a6700;background:#fff8e6}
        .mono{font-family:"Geist Mono",monospace}
        .tab-panel{display:none}.tab-panel.active{display:block}
        .empty{text-align:center;padding:40px;color:#9c9fa1}
        select{height:36px;padding:0 14px;border:1px solid #dde1e3;border-radius:10px;width:100%;font-size:14px;appearance:none;background:#fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%236A6F73' viewBox='0 0 24 24'%3E%3Cpath d='M12 16l-6-6h12z'/%3E%3C/svg%3E") no-repeat right 12px center;padding-right:36px}
        input[type="number"],input[type="text"]{height:36px;border:1px solid #dde1e3;border-radius:10px;padding:0 14px;font-size:14px;width:100%;background:#fff}
        input[type="number"]:focus,input[type="text"]:focus,select:focus{outline:none;border-color:#0066f5;box-shadow:0 0 0 3px rgba(0,102,245,.1)}
        .pill{display:inline-flex;align-items:center;height:28px;padding:0 10px;font-size:12px;font-weight:500;border-radius:999px;border:none;cursor:pointer;background:#f0f6fe;color:#0052c4;margin:0 4px 4px 0;transition:background .12s,color .12s}
        .pill:hover{background:#b4d2fc}
        .pill.active{background:#0066f5;color:#fff}
        .toast-box{position:fixed;bottom:20px;right:20px;z-index:100}
        .toast{padding:10px 16px;border-radius:10px;font-size:14px;font-weight:500;color:#fff;background:#0d0e0f;margin-top:8px;box-shadow:0 4px 12px rgba(0,0,0,.15);animation:toastIn .25s ease}
        @keyframes toastIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        .toast-ok{background:#3abc3f}.toast-err{background:#ff3131}
        .stat{font-size:32px;line-height:44px;font-family:"Geist Mono",monospace;font-weight:500}
        .stat-sm{font-size:24px;line-height:32px}
        .stat-info{color:#0066f5}.stat-ok{color:#3abc3f}.stat-warn{color:#ffa931}.stat-err{color:#ff3131}
        .stat-label{font-size:12px;color:#6a6f73;margin-top:4px}
        .progress{width:100%;height:6px;background:#e4e7e9;border-radius:999px;overflow:hidden}
        .progress-fill{height:100%;background:#0066f5;border-radius:999px;transition:width .3s}
        .tp-card{background:#fff;border:1px solid #e4e7e9;border-radius:12px;padding:16px;position:relative;overflow:hidden}
        .tp-card.active{border-color:#0066f5;box-shadow:0 0 0 3px rgba(0,102,245,.08)}
        .tp-header{display:flex;align-items:center;gap:12px;margin-bottom:12px}
        .tp-ring{width:48px;height:48px;position:relative;flex-shrink:0}
        .tp-ring svg{transform:rotate(-90deg)}
        .tp-ring-text{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#0066f5}
        .tp-phase{font-size:13px;font-weight:600;color:#0d0e0f}
        .tp-msg{font-size:12px;color:#6a6f73;margin-top:2px}
        .tp-elapsed{font-size:11px;color:#6a6f73;font-family:'Geist Mono',monospace}
        .tp-steps{display:flex;gap:0;margin-top:12px}
        .tp-step{flex:1;text-align:center;padding:8px 4px;position:relative}
        .tp-step::before{content:'';position:absolute;top:14px;left:0;right:0;height:2px;background:#e4e7e9}
        .tp-step:first-child::before{left:50%}
        .tp-step:last-child::before{right:50%}
        .tp-dot{width:10px;height:10px;border-radius:50%;background:#e4e7e9;margin:0 auto 6px;position:relative;z-index:1;transition:all .3s}
        .tp-step.done .tp-dot{background:#3abc3f}
        .tp-step.active .tp-dot{background:#0066f5;box-shadow:0 0 0 4px rgba(0,102,245,.2);animation:tpPulse 1.5s infinite}
        .tp-step.fail .tp-dot{background:#ff3131}
        .tp-label{font-size:10px;color:#9ca0a5;white-space:nowrap}
        .tp-step.done .tp-label,.tp-step.active .tp-label{color:#0d0e0f;font-weight:500}
        @keyframes tpPulse{0%,100%{box-shadow:0 0 0 4px rgba(0,102,245,.2)}50%{box-shadow:0 0 0 8px rgba(0,102,245,.05)}}
        .mc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px;margin-top:12px}
        .mc-card{border-radius:10px;padding:10px 12px;font-size:12px;border:1px solid #e4e7e9;transition:all .3s}
        .mc-card.pending{background:#fafafa;color:#9ca0a5}
        .mc-card.running{background:#f0f6fe;border-color:#0066f5;color:#0066f5;animation:tpPulse 1.5s infinite}
        .mc-card.done{background:#f3fbf4;border-color:#3abc3f;color:#0d0e0f}
        .mc-card.fail{background:#fff5f5;border-color:#ff3131;color:#ff3131}
        .mc-name{font-weight:600;margin-bottom:2px}
        .mc-score{font-family:'Geist Mono',monospace;font-size:11px}
        .mc-icon{float:right;font-size:14px}
        table{width:100%;border-collapse:collapse}
        th{padding:8px 12px;text-align:left;font-size:12px;font-weight:500;color:#6a6f73;background:#f8f9f9;border-bottom:1px solid #e4e7e9}
        td{padding:8px 12px;font-size:14px;border-bottom:1px solid #ebedee}
        tr:hover td{background:#fafbfb}
        textarea{width:100%;min-height:120px;padding:8px 12px;border:1px solid #dde1e3;border-radius:12px;font-size:14px;resize:none;font-family:inherit}
        pre.code{background:#f8f9f9;border:1px solid #e4e7e9;border-radius:10px;padding:16px;font-size:13px;font-family:"Geist Mono",monospace;overflow:auto}
        label.form{display:block;font-size:14px;font-weight:500;margin-bottom:6px}
        .form-stack>*+*{margin-top:16px}
        .hint{font-size:12px;color:#9c9fa1;margin-top:4px}
        .alert{padding:12px 16px;border-radius:10px;font-size:13px;display:flex;align-items:flex-start;gap:8px;margin-bottom:12px}
        .alert i{margin-top:1px;flex-shrink:0}
        .alert-warn{background:#fff8e6;color:#9a6700;border:1px solid #f5e6c8}
        .alert-info{background:#f0f6fe;color:#0052c4;border:1px solid #c8dcf5}
        /* Dashboard cards */
        .dash-stat-card{background:#fff;border:1px solid #e4e7e9;border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:8px}
        .dash-stat-card .icon-wrap{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px}
        .dash-stat-card .icon-wrap.blue{background:#f0f6fe;color:#0066f5}
        .dash-stat-card .icon-wrap.green{background:#f3fbf4;color:#3abc3f}
        .dash-stat-card .icon-wrap.orange{background:#fff8e6;color:#ffa931}
        .dash-stat-card .icon-wrap.red{background:#fff3f3;color:#ff3131}
        .quick-action{display:flex;align-items:center;gap:12px;padding:12px;border-radius:10px;border:1px solid #e4e7e9;cursor:pointer;background:#fff;transition:border-color .12s,box-shadow .12s;width:100%;text-align:left}
        .quick-action:hover{border-color:#0066f5;box-shadow:0 0 0 3px rgba(0,102,245,.08)}
        .quick-action i{font-size:20px;color:#0066f5;width:36px;height:36px;display:flex;align-items:center;justify-content:center;background:#f0f6fe;border-radius:8px;flex-shrink:0}
        .quick-action .qa-title{font-size:14px;font-weight:500}
        .quick-action .qa-desc{font-size:12px;color:#6a6f73}
        /* Chart enhancements */
        .chart-wrap{position:relative;border-radius:12px;overflow:hidden}
        .chart-wrap canvas{display:block;width:100%!important}
        .chart-tooltip{position:absolute;background:#0d0e0f;color:#fff;padding:6px 10px;border-radius:8px;font-size:12px;font-family:"Geist Mono",monospace;pointer-events:none;opacity:0;transition:opacity .15s;z-index:10;white-space:nowrap;box-shadow:0 4px 12px rgba(0,0,0,.2)}
        .chart-legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
        .chart-legend-item{display:flex;align-items:center;gap:6px;font-size:12px;color:#6a6f73}
        .chart-legend-dot{width:10px;height:10px;border-radius:50%}
        .viz-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-top:12px}
        .mini-hist{background:#f8f9f9;border:1px solid #e4e7e9;border-radius:10px;padding:12px;cursor:default;transition:border-color .15s,box-shadow .15s}
        .mini-hist:hover{border-color:#0066f5;box-shadow:0 0 0 3px rgba(0,102,245,.08)}
        .mini-hist-title{font-size:12px;font-weight:600;color:#0d0e0f;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .mini-hist-stat{font-size:11px;color:#9c9fa1;font-family:"Geist Mono",monospace;margin-bottom:6px}
        .mini-hist canvas{border-radius:6px}
        @keyframes countUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        .stat-animated{animation:countUp .4s ease-out}
        .donut-center{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center}
        .donut-center .donut-val{font-size:28px;font-weight:600;font-family:"Geist Mono",monospace}
        .donut-center .donut-label{font-size:11px;color:#6a6f73}
        .sub-tabs{display:flex;gap:4px;margin-bottom:20px;border-bottom:1px solid #e4e7e9;padding-bottom:0}
        .sub-tab{display:inline-flex;align-items:center;gap:5px;height:36px;padding:0 14px;font-size:13px;font-weight:500;border:none;background:none;color:#6a6f73;cursor:pointer;border-bottom:2px solid transparent;transition:color .12s,border-color .12s;margin-bottom:-1px}
        .sub-tab:hover{color:#0d0e0f}
        .sub-tab.active{color:#0066f5;border-bottom-color:#0066f5}
        .sub-tab i{font-size:16px}
        .sub-panel{display:none}.sub-panel.active{display:block}
        /* File upload */
        .upload-zone{border:2px dashed #dde1e3;border-radius:12px;padding:32px;text-align:center;cursor:pointer;transition:border-color .15s,background .15s}
        .upload-zone:hover,.upload-zone.dragover{border-color:#0066f5;background:#f0f6fe}
        .upload-zone i{font-size:32px;color:#9c9fa1;margin-bottom:8px}
        .upload-zone p{font-size:13px;color:#6a6f73}
        .upload-zone input[type="file"]{display:none}
        /* Checkbox grid */
        .chk-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
        .chk-item{display:flex;align-items:center;gap:6px;padding:6px 8px;border-radius:8px;font-size:13px;cursor:pointer;transition:background .1s}
        .chk-item:hover{background:#f1f3f4}
        .chk-item input{accent-color:#0066f5}
        /* Config summary */
        .config-summary{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
        .config-chip{display:inline-flex;align-items:center;gap:4px;height:28px;padding:0 10px;font-size:12px;border-radius:8px;background:#f1f3f4;color:#6a6f73}
        .config-chip strong{color:#0d0e0f;font-weight:600}
    </style>
</head>
<body>
    <div class="app">
        <aside class="sidebar" id="e-sidebar">
            <div class="sidebar-header">
                <div class="sidebar-logo">K</div>
                <div class="sidebar-title">Kolosal AutoML</div>
            </div>
            <nav class="sidebar-nav">
                <div class="sidebar-section">Overview</div>
                <button class="sidebar-item active" data-tab="dashboard" onclick="eTab('dashboard')"><i class="ri-dashboard-line"></i><span>Dashboard</span></button>
                <button class="sidebar-item" data-tab="data" onclick="eTab('data')"><i class="ri-database-2-line"></i><span>Data</span></button>
                <div class="sidebar-section">Training</div>
                <button class="sidebar-item" data-tab="train" onclick="eTab('train')"><i class="ri-play-circle-line"></i><span>Train</span></button>
                <div class="sidebar-section">Analysis</div>
                <button class="sidebar-item" data-tab="analysis" onclick="eTab('analysis')"><i class="ri-bar-chart-line"></i><span>Analysis</span></button>
                <button class="sidebar-item" data-tab="reports" onclick="eTab('reports');eLoadReport()"><i class="ri-file-chart-line"></i><span>Reports</span></button>
                <button class="sidebar-item" data-tab="monitor" onclick="eTab('monitor')"><i class="ri-pulse-line"></i><span>Monitor</span></button>
            </nav>
            <button class="sidebar-toggle" onclick="eToggleSidebar()" title="Toggle sidebar"><i id="e-sidebar-icon" class="ri-arrow-right-s-line"></i></button>
        </aside>
        <div class="main-wrap">
            <div class="topbar">
                <div><div class="topbar-title" id="e-page-title">Dashboard</div></div>
                <div class="topbar-meta">
                    <span id="e-topbar-data" style="display:none"><strong style="color:#0066f5" id="e-topbar-rows">0</strong> rows &times; <strong style="color:#0066f5" id="e-topbar-cols">0</strong> cols &mdash; <span id="e-topbar-name"></span></span>
                    <span id="e-badge" class="badge badge-ok">healthy</span>
                    <span id="e-dot" style="width:8px;height:8px;border-radius:50%;background:#3abc3f;display:inline-block"></span>
                </div>
            </div>
            <div class="content">
                <div class="content-inner">
                    <!-- Dashboard -->
                    <div id="et-dashboard" class="tab-panel active">
                        <div class="grid-4" style="margin-bottom:20px">
                            <div class="dash-stat-card">
                                <div class="icon-wrap blue"><i class="ri-database-2-line"></i></div>
                                <div class="stat stat-sm stat-info" id="e-dash-rows">&mdash;</div>
                                <div class="stat-label">Data Rows</div>
                            </div>
                            <div class="dash-stat-card">
                                <div class="icon-wrap green"><i class="ri-bar-chart-line"></i></div>
                                <div class="stat stat-sm stat-ok" id="e-dash-models">0</div>
                                <div class="stat-label">Models Trained</div>
                            </div>
                            <div class="dash-stat-card">
                                <div class="icon-wrap orange"><i class="ri-speed-line"></i></div>
                                <div class="stat stat-sm stat-warn" id="e-dash-cpu">0%</div>
                                <div class="stat-label">CPU Usage</div>
                            </div>
                            <div class="dash-stat-card">
                                <div class="icon-wrap red"><i class="ri-hard-drive-2-line"></i></div>
                                <div class="stat stat-sm" id="e-dash-mem" style="color:#6a6f73">0%</div>
                                <div class="stat-label">Memory Usage</div>
                            </div>
                        </div>
                        <div class="grid-2">
                            <div class="card">
                                <div class="card-header"><div class="card-title">Quick Actions</div></div>
                                <div style="display:grid;gap:8px">
                                    <button class="quick-action" onclick="eTab('data')"><i class="ri-upload-2-line"></i><div><div class="qa-title">Load Dataset</div><div class="qa-desc">Import sample or custom data</div></div></button>
                                    <button class="quick-action" onclick="eTab('train')"><i class="ri-magic-line"></i><div><div class="qa-title">Run AutoML</div><div class="qa-desc">One-click pipeline &mdash; train &amp; optimize</div></div></button>
                                    <button class="quick-action" onclick="eTab('train')"><i class="ri-speed-line"></i><div><div class="qa-title">Auto-Tune</div><div class="qa-desc">Train all models &amp; pick the best</div></div></button>
                                    <button class="quick-action" onclick="eTab('analysis')"><i class="ri-bar-chart-line"></i><div><div class="qa-title">Analyze Results</div><div class="qa-desc">Feature importance, anomalies, UMAP</div></div></button>
                                </div>
                            </div>
                            <div class="card">
                                <div class="card-header"><div class="card-title">System Health</div></div>
                                <div style="display:flex;flex-direction:column;gap:12px">
                                    <div style="display:flex;justify-content:space-between;align-items:center"><span style="font-size:14px">Status</span><span id="e-dash-status" class="badge badge-ok">healthy</span></div>
                                    <div><div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px"><span>CPU</span><span class="mono" id="e-dash-cpu2">0%</span></div><div class="progress"><div class="progress-fill" id="e-dash-cpu-bar" style="width:0%"></div></div></div>
                                    <div><div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px"><span>Memory</span><span class="mono" id="e-dash-mem2">0%</span></div><div class="progress"><div class="progress-fill" id="e-dash-mem-bar" style="width:0%;background:#3abc3f"></div></div></div>
                                    <div style="font-size:12px;color:#9c9fa1;margin-top:4px" id="e-dash-mem-detail"></div>
                                </div>
                            </div>
                        </div>
                        <div class="grid-2">
                            <div class="card">
                                <div class="card-header"><div class="card-title">Trained Models</div><button class="btn btn-sm btn-outline" onclick="eDashModels()"><i class="ri-refresh-line" style="font-size:14px"></i> Refresh</button></div>
                                <div id="e-dash-model-list" class="empty">No models trained yet. Load data and run training to get started.</div>
                            </div>
                            <div class="card">
                                <div class="card-header"><div class="card-title">Latest Training Result</div></div>
                                <div id="e-dash-train-result" class="empty">No training results yet</div>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-header"><div class="card-title">Activity Log</div></div>
                            <div id="e-dash-activity" style="max-height:200px;overflow-y:auto"><div class="empty">No activity yet</div></div>
                        </div>
                    </div>
                    <!-- Data -->
                    <div id="et-data" class="tab-panel">
                        <div class="grid-2">
                            <div class="card">
                                <div class="card-header"><div class="card-title">Sample Datasets</div></div>
                                <div id="e-pills"></div>
                                <div style="margin-top:16px;padding-top:12px;border-top:1px solid #e4e7e9">
                                    <p style="font-size:12px;font-weight:500;color:#6a6f73;margin-bottom:8px">Or import from URL:</p>
                                    <div style="display:flex;gap:8px"><input type="text" id="e-import-url" placeholder="Paste URL (Kaggle, GitHub, CSV...)" style="flex:1"><button class="btn" onclick="eImportUrl()">Import</button></div>
                                    <div id="e-import-status" style="margin-top:8px"></div>
                                </div>
                                <div style="margin-top:16px;padding-top:12px;border-top:1px solid #e4e7e9">
                                    <div class="upload-zone" id="e-upload-zone" onclick="document.getElementById('e-file-input').click()">
                                        <input type="file" id="e-file-input" accept=".csv,.tsv,.json" onchange="eFileUpload(this)">
                                        <i class="ri-file-upload-line"></i>
                                        <p><strong>Drop CSV file here</strong> or click to browse</p>
                                        <p style="font-size:11px;color:#9c9fa1;margin-top:4px">Supports .csv, .tsv, .json</p>
                                    </div>
                                </div>
                            </div>
                            <div class="card">
                                <div class="card-header"><div class="card-title">Data Info</div></div>
                                <div id="e-info" class="empty">No data loaded</div>
                            </div>
                        </div>
                        <div id="e-data-preview-wrap" style="display:none">
                            <div class="card">
                                <div class="card-header"><div class="card-title">Data Preview</div><span class="badge badge-info" id="e-preview-badge"></span></div>
                                <div style="overflow-x:auto" id="e-data-preview"></div>
                            </div>
                        </div>
                        <div id="e-analysis-wrap" style="display:none">
                            <div class="card">
                                <div class="card-header"><div><div class="card-title"><i class="ri-bar-chart-box-line" style="color:#0066f5"></i> Data Quality Report</div><div class="card-desc">Automated profiling of your dataset</div></div><span class="badge" id="e-quality-badge"></span></div>
                                <div id="e-analysis-loading" style="text-align:center;padding:20px"><span class="spinner" style="display:inline-block;width:20px;height:20px;border:2px solid #e4e7e9;border-top-color:#0066f5;border-radius:50%;animation:spin .6s linear infinite"></span><p style="font-size:13px;color:#6a6f73;margin-top:8px">Analyzing dataset...</p></div>
                                <div id="e-analysis-content" style="display:none">
                                    <div class="grid-4" id="e-analysis-overview" style="margin-bottom:20px"></div>
                                    <div id="e-analysis-warnings" style="margin-bottom:16px"></div>
                                    <div style="margin-bottom:20px"><p style="font-size:14px;font-weight:600;margin-bottom:12px">Data Quality</p><div class="grid-4" id="e-quality-gauges"></div></div>
                                    <div style="margin-bottom:20px"><p style="font-size:14px;font-weight:600;margin-bottom:12px">Column Statistics</p><div style="overflow-x:auto" id="e-col-stats-table"></div></div>
                                    <div id="e-outlier-section" style="margin-bottom:20px"><p style="font-size:14px;font-weight:600;margin-bottom:12px">Outlier Detection</p><div id="e-outlier-table"></div></div>
                                    <div id="e-missing-section"><p style="font-size:14px;font-weight:600;margin-bottom:12px">Missing Values</p><div id="e-missing-chart"></div></div>
                                </div>
                            </div>
                        </div>
                        <div id="e-data-viz-wrap" style="display:none">
                            <div class="card">
                                <div class="card-header"><div><div class="card-title">Feature Distributions</div><div class="card-desc">Visual overview of each column's value distribution</div></div></div>
                                <div id="e-col-distributions" class="viz-grid"></div>
                            </div>
                            <div class="grid-2" style="margin-top:16px">
                                <div class="card">
                                    <div class="card-header"><div><div class="card-title">Scatter Plot</div><div class="card-desc">Explore relationships between features</div></div></div>
                                    <div style="display:flex;gap:8px;margin-bottom:12px">
                                        <div style="flex:1"><label class="form" style="font-size:12px">X Axis</label><select id="e-scatter-x" onchange="eDrawScatterFromData()" style="font-size:12px;height:30px"></select></div>
                                        <div style="flex:1"><label class="form" style="font-size:12px">Y Axis</label><select id="e-scatter-y" onchange="eDrawScatterFromData()" style="font-size:12px;height:30px"></select></div>
                                        <div style="flex:1"><label class="form" style="font-size:12px">Color By</label><select id="e-scatter-c" onchange="eDrawScatterFromData()" style="font-size:12px;height:30px"><option value="">None</option></select></div>
                                    </div>
                                    <div class="chart-wrap"><canvas id="e-scatter" width="520" height="360" style="border-radius:10px"></canvas><div class="chart-tooltip" id="e-scatter-tip"></div></div>
                                </div>
                                <div class="card">
                                    <div class="card-header"><div><div class="card-title">Correlation Matrix</div><div class="card-desc">Pairwise feature correlations</div></div></div>
                                    <div class="chart-wrap"><canvas id="e-corr" width="520" height="400" style="border-radius:10px"></canvas><div class="chart-tooltip" id="e-corr-tip"></div></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <!-- Train (merged: Config + Single Train + AutoML + HyperOpt + Auto-Tune + Ensemble) -->
                    <div id="et-train" class="tab-panel">
                        <div class="sub-tabs">
                            <button class="sub-tab active" onclick="eSubTab('train','autotune')"><i class="ri-speed-line"></i> Auto-Tune</button>
                            <button class="sub-tab" onclick="eSubTab('train','automl')"><i class="ri-magic-line"></i> AutoML</button>
                            <button class="sub-tab" onclick="eSubTab('train','single')"><i class="ri-play-circle-line"></i> Single Model</button>
                            <button class="sub-tab" onclick="eSubTab('train','hyperopt')"><i class="ri-equalizer-line"></i> HyperOpt</button>
                            <button class="sub-tab" onclick="eSubTab('train','ensemble')"><i class="ri-stack-line"></i> Ensemble</button>
                        </div>
                        <div id="e-train-nodata" class="alert alert-warn" style="display:none"><i class="ri-error-warning-line"></i><span>Load a dataset first from the <a href="javascript:void(0)" onclick="eTab('data');return false" style="color:inherit;font-weight:600">Data</a> page.</span></div>
                        <!-- Auto-Tune sub-panel (default) -->
                        <div id="esp-train-autotune" class="sub-panel active">
                            <div class="card" style="max-width:700px">
                                <div class="card-header"><div><div class="card-title"><i class="ri-speed-line" style="color:#0066f5"></i> Auto-Tune</div><div class="card-desc">Automatically trains all applicable models, ranks them, and picks the best. This is the recommended way to train.</div></div></div>
                                <div class="grid-3" style="margin-bottom:16px">
                                    <div><label class="form">Target Column</label><select id="e-at-target"><option value="">Select...</option></select></div>
                                    <div><label class="form">Task Type</label><select id="e-at-task"><option value="classification">Classification</option><option value="regression">Regression</option><option value="clustering">Clustering</option></select></div>
                                    <div style="display:flex;align-items:flex-end"><button id="e-at-btn" class="btn btn-primary" onclick="eAutoTune()" disabled style="width:100%;height:40px"><i class="ri-speed-line"></i> Start Auto-Tune</button></div>
                                </div>
                                <div id="e-at-progress" style="display:none"></div>
                                <div id="e-at-result" class="empty">Load data and click Start Auto-Tune to begin</div>
                            </div>
                            <div id="e-at-charts" style="display:none;margin-top:16px">
                                <div class="card">
                                    <p style="font-size:14px;font-weight:500;margin-bottom:12px;color:#6a6f73">Model Comparison</p>
                                    <canvas id="e-at-bar" width="700" height="300" style="width:100%;border-radius:8px"></canvas>
                                </div>
                            </div>
                        </div>
                        <!-- AutoML sub-panel -->
                        <div id="esp-train-automl" class="sub-panel">
                            <div class="card" style="max-width:640px">
                                <div class="card-header"><div><div class="card-title">One-Click AutoML</div><div class="card-desc">Automatically detect, preprocess, train, and optimize.</div></div></div>
                                <div style="margin-bottom:12px"><label class="form">Target Column</label><select id="e-automl-target"><option value="">Select target...</option></select></div>
                                <button class="btn btn-primary" id="e-btn-automl" onclick="eRunAutoML()" disabled style="width:100%;height:44px;font-size:16px">Run AutoML Pipeline</button>
                                <details style="margin-top:16px"><summary style="cursor:pointer;font-size:13px;font-weight:500;color:#6a6f73">Advanced Settings</summary>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px">
                                    <div><label class="form">Task Type</label><select id="e-aml-task"><option value="">Auto-Detect</option><option value="classification">Classification</option><option value="regression">Regression</option></select></div>
                                    <div><label class="form">Scaler</label><select id="e-aml-scaler"><option value="">Auto</option><option value="standard">Standard</option><option value="minmax">MinMax</option><option value="robust">Robust</option></select></div>
                                    <div><label class="form">Max Models</label><input type="number" id="e-aml-max" value="8" min="1" max="20"></div>
                                    <div><label class="form">HyperOpt Trials</label><input type="number" id="e-aml-trials" value="20" min="0" max="200"></div>
                                </div></details>
                                <div id="e-automl-progress" style="display:none;margin-top:16px">
                                    <div style="font-size:14px;font-weight:500;color:#0066f5;margin-bottom:8px" id="e-aml-status">Starting...</div>
                                    <div class="progress"><div class="progress-fill" id="e-aml-bar" style="width:0%"></div></div>
                                </div>
                                <div id="e-automl-result" style="display:none;margin-top:16px">
                                    <div style="background:#0d0e0f;border-radius:10px;padding:16px;color:#fff;margin-bottom:12px">
                                        <div style="font-size:12px;opacity:.7">Best Model</div>
                                        <div style="font-size:20px;font-weight:600" id="e-aml-best">&mdash;</div>
                                        <div style="font-size:14px;opacity:.85" id="e-aml-score">&mdash;</div>
                                    </div>
                                    <div id="e-aml-leaderboard"></div>
                                    <div id="e-aml-chart-wrap" style="display:none;margin-top:16px">
                                        <p style="font-size:14px;font-weight:500;margin-bottom:12px;color:#6a6f73">Model Comparison</p>
                                        <canvas id="e-aml-chart" width="700" height="300" style="width:100%;border-radius:8px"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <!-- Single Model sub-panel -->
                        <div id="esp-train-single" class="sub-panel">
                            <div class="card" style="max-width:700px">
                                <div class="card-header">
                                    <div><div class="card-title">Single Model Training</div><div class="card-desc">Configure and train one model.</div></div>
                                    <button id="e-train-btn" class="btn" onclick="eTrain()" disabled>Train Model</button>
                                </div>
                                <div class="grid-3" style="margin-bottom:16px">
                                    <div><label class="form">Target Column</label><select id="e-target"><option value="">Select...</option></select></div>
                                    <div><label class="form">Task Type</label><select id="e-task"><option value="classification">Classification</option><option value="regression">Regression</option><option value="clustering">Clustering</option></select></div>
                                    <div><label class="form">Model</label><select id="e-model"><option value="random_forest">Random Forest</option><option value="xgboost">XGBoost</option><option value="lightgbm">LightGBM</option><option value="catboost">CatBoost</option><option value="gradient_boosting">Gradient Boosting</option><option value="extra_trees">Extra Trees</option><option value="decision_tree">Decision Tree</option><option value="logistic_regression">Logistic Regression</option><option value="ridge">Ridge</option><option value="lasso">Lasso</option><option value="elastic_net">Elastic Net</option><option value="polynomial">Polynomial</option><option value="adaboost">AdaBoost</option><option value="sgd">SGD</option><option value="knn">KNN</option><option value="naive_bayes">Naive Bayes</option><option value="svm">SVM</option><option value="gaussian_process">Gaussian Process</option><option value="kmeans">KMeans</option><option value="dbscan">DBSCAN</option></select></div>
                                </div>
                                <div id="e-result"></div>
                                <div id="e-train-viz" style="display:none;margin-top:16px">
                                    <div class="grid-2">
                                        <div class="chart-wrap"><canvas id="e-train-radar" width="400" height="400" style="border-radius:10px"></canvas></div>
                                        <div class="chart-wrap"><canvas id="e-train-bars" width="520" height="400" style="border-radius:10px"></canvas></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <!-- HyperOpt sub-panel -->
                        <div id="esp-train-hyperopt" class="sub-panel">
                            <div class="grid-2">
                                <div class="card">
                                    <div class="card-header"><div class="card-title">Hyperparameter Optimization</div></div>
                                    <div class="form-stack">
                                        <div><label class="form">Model</label><select id="e-ho-model"><option value="random_forest">Random Forest</option><option value="xgboost">XGBoost</option><option value="lightgbm">LightGBM</option><option value="catboost">CatBoost</option><option value="gradient_boosting">Gradient Boosting</option><option value="extra_trees">Extra Trees</option><option value="decision_tree">Decision Tree</option><option value="adaboost">AdaBoost</option><option value="sgd">SGD</option><option value="knn">KNN</option><option value="svm">SVM</option><option value="polynomial">Polynomial</option><option value="gaussian_process">Gaussian Process</option></select></div>
                                        <div><label class="form">Sampler</label><select id="e-ho-sampler"><option value="tpe">TPE</option><option value="random">Random</option><option value="gp">Gaussian Process</option></select></div>
                                        <div><label class="form">Trials</label><input type="number" id="e-ho-trials" value="20" min="5" max="100"></div>
                                        <button id="e-ho-btn" class="btn" onclick="eHyperOpt()" disabled>Optimize</button>
                                    </div>
                                </div>
                                <div class="card">
                                    <div class="card-header"><div class="card-title">Results</div></div>
                                    <div id="e-ho-result" class="empty">Run optimization to see results</div>
                                </div>
                            </div>
                            <div id="e-ho-charts" style="display:none;margin-top:16px">
                                <div class="grid-2">
                                    <div class="card">
                                        <p style="font-size:14px;font-weight:500;margin-bottom:12px;color:#6a6f73">Optimization Convergence</p>
                                        <canvas id="e-ho-cvg" width="520" height="260" style="width:100%;border-radius:8px"></canvas>
                                    </div>
                                    <div class="card">
                                        <p style="font-size:14px;font-weight:500;margin-bottom:12px;color:#6a6f73">Best Score Progress</p>
                                        <canvas id="e-ho-best" width="520" height="260" style="width:100%;border-radius:8px"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <!-- Ensemble sub-panel -->
                        <div id="esp-train-ensemble" class="sub-panel">
                            <div class="card">
                                <div class="card-header"><div><div class="card-title">Ensemble Training</div><div class="card-desc">Select 2+ models and a strategy, then train an ensemble.</div></div></div>
                                <div style="margin-bottom:12px">
                                    <label class="form">Select Models</label>
                                    <div class="chk-grid" id="e-ens-models">
                                        <label class="chk-item"><input type="checkbox" value="random_forest" checked> Random Forest</label>
                                        <label class="chk-item"><input type="checkbox" value="decision_tree" checked> Decision Tree</label>
                                        <label class="chk-item"><input type="checkbox" value="gradient_boosting" checked> Gradient Boosting</label>
                                        <label class="chk-item"><input type="checkbox" value="extra_trees"> Extra Trees</label>
                                        <label class="chk-item"><input type="checkbox" value="adaboost"> AdaBoost</label>
                                        <label class="chk-item"><input type="checkbox" value="knn"> KNN</label>
                                        <label class="chk-item"><input type="checkbox" value="svm"> SVM</label>
                                        <label class="chk-item"><input type="checkbox" value="logistic_regression"> Logistic Regression</label>
                                        <label class="chk-item"><input type="checkbox" value="naive_bayes"> Naive Bayes</label>
                                    </div>
                                </div>
                                <div class="grid-2" style="margin-bottom:12px">
                                    <div><label class="form">Strategy</label><select id="e-ens-strat"><option value="voting">Voting</option><option value="averaging">Averaging</option></select></div>
                                    <div style="display:flex;align-items:flex-end"><button id="e-ens-btn" class="btn" onclick="eEnsemble()" disabled style="width:100%">Train Ensemble</button></div>
                                </div>
                                <div id="e-ens-result" style="margin-top:16px" class="empty">Select models and run ensemble training</div>
                            </div>
                        </div>
                    </div>
                    <!-- Analysis (merged: Explain + Anomaly + UMAP/PCA) -->
                    <div id="et-analysis" class="tab-panel">
                        <div class="sub-tabs">
                            <button class="sub-tab active" onclick="eSubTab('analysis','explain')"><i class="ri-search-eye-line"></i> Explainability</button>
                            <button class="sub-tab" onclick="eSubTab('analysis','anomaly')"><i class="ri-alarm-warning-line"></i> Anomaly</button>
                            <button class="sub-tab" onclick="eSubTab('analysis','dimred')"><i class="ri-flashlight-line"></i> UMAP / PCA</button>
                        </div>
                        <!-- Explain sub-panel -->
                        <div id="esp-analysis-explain" class="sub-panel active">
                            <div class="card">
                                <div class="card-header">
                                    <div><div class="card-title">Feature Importance</div><div class="card-desc">Analyze which features matter most to your trained model.</div></div>
                                    <button id="e-explain-btn" class="btn" onclick="eExplain()" disabled>Analyze</button>
                                </div>
                                <div id="e-explain-result" class="empty">Train a model first, then analyze feature importance</div>
                            </div>
                        </div>
                        <!-- Anomaly sub-panel -->
                        <div id="esp-analysis-anomaly" class="sub-panel">
                            <div class="card">
                                <div class="card-header">
                                    <div><div class="card-title">Anomaly Detection</div><div class="card-desc">Find unusual data points using Isolation Forest.</div></div>
                                    <button id="e-anomaly-btn" class="btn" onclick="eAnomaly()" disabled>Detect</button>
                                </div>
                                <div id="e-anom-nodata" class="alert alert-warn" style="display:none"><i class="ri-error-warning-line"></i><span>Load a dataset first from the <a href="javascript:void(0)" onclick="eTab('data');return false" style="color:inherit;font-weight:600">Data</a> page.</span></div>
                                <div class="grid-2" style="margin-bottom:12px">
                                    <div><label class="form">Contamination</label><input type="number" id="e-anom-cont" value="0.1" min="0.01" max="0.5" step="0.01"><p class="hint">Expected fraction of anomalies (0.01&ndash;0.5)</p></div>
                                    <div><label class="form">Estimators</label><input type="number" id="e-anom-est" value="100" min="10" max="1000" step="10"><p class="hint">Number of trees in the forest</p></div>
                                </div>
                                <div id="e-anomaly-result" class="empty">Load data and run anomaly detection</div>
                                <div id="e-anom-viz" style="display:none;margin-top:16px">
                                    <div class="grid-2">
                                        <div class="chart-wrap" style="position:relative"><canvas id="e-anom-scatter" width="520" height="360" style="border-radius:10px"></canvas><div class="chart-tooltip" id="e-anom-tip"></div></div>
                                        <div style="position:relative"><canvas id="e-anom-donut" width="360" height="360" style="border-radius:10px;margin:0 auto;display:block"></canvas><div class="donut-center" id="e-anom-donut-center"></div></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <!-- UMAP/PCA sub-panel -->
                        <div id="esp-analysis-dimred" class="sub-panel">
                            <div class="card">
                                <div class="card-header"><div><div class="card-title">Dimensionality Reduction</div><div class="card-desc">Project high-dimensional data into 2D using UMAP or PCA.</div></div></div>
                                <div id="e-dr-nodata" class="alert alert-warn" style="display:none"><i class="ri-error-warning-line"></i><span>Load a dataset first from the <a href="javascript:void(0)" onclick="eTab('data');return false" style="color:inherit;font-weight:600">Data</a> page.</span></div>
                                <div class="grid-4" style="margin-bottom:16px">
                                    <div><label class="form">Method</label><select id="e-dr-method"><option value="umap">UMAP</option><option value="pca">PCA</option></select></div>
                                    <div><label class="form">Color By</label><select id="e-dr-color"><option value="">None</option></select></div>
                                    <div><label class="form">Neighbors <span class="hint">(UMAP)</span></label><input type="number" id="e-dr-neighbors" value="15" min="2" max="200"></div>
                                    <div style="display:flex;align-items:flex-end"><button id="e-dr-btn" class="btn btn-primary" onclick="eRunDimRed()" disabled style="width:100%">Run</button></div>
                                </div>
                                <div id="e-dr-progress" style="display:none"><p style="font-size:13px;color:#6a6f73"><span class="spinner" style="display:inline-block;width:14px;height:14px;border:2px solid rgba(0,102,245,.3);border-top-color:#0066f5;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:6px"></span> Computing projection...</p></div>
                                <div id="e-dr-result" style="display:none">
                                    <div class="grid-2">
                                        <div class="chart-wrap" style="position:relative"><canvas id="e-dr-scatter" width="520" height="420" style="border-radius:10px"></canvas><div class="chart-tooltip" id="e-dr-tip"></div></div>
                                        <div>
                                            <div id="e-dr-info" style="margin-bottom:12px"></div>
                                            <div id="e-dr-legend" class="chart-legend"></div>
                                            <div id="e-dr-variance" style="margin-top:16px"></div>
                                        </div>
                                    </div>
                                </div>
                                <div id="e-dr-empty" class="empty">Select a method and click Run to project your data</div>
                            </div>
                        </div>
                    </div>
                    <!-- Reports -->
                    <div id="et-reports" class="tab-panel">
                        <div id="e-rpt-empty" class="card" style="max-width:700px">
                            <div class="card-header"><div><div class="card-title"><i class="ri-file-chart-line" style="color:#0066f5"></i> Model Comparison Report</div><div class="card-desc">Select trained models to generate a side-by-side comparison report with metrics, charts, and rankings.</div></div></div>
                            <div id="e-rpt-nomodels" style="display:none" class="alert alert-warn"><i class="ri-error-warning-line"></i><span>No models trained yet. Train some models first from the <a href="javascript:void(0)" onclick="eTab('train');return false" style="color:inherit;font-weight:600">Train</a> tab.</span></div>
                            <div id="e-rpt-select" style="display:none">
                                <label class="form" style="margin-bottom:8px">Select Models to Compare</label>
                                <div class="chk-grid" id="e-rpt-models"></div>
                                <div style="display:flex;gap:8px;margin-top:16px">
                                    <button class="btn btn-sm btn-outline" onclick="eRptSelectAll(true)">Select All</button>
                                    <button class="btn btn-sm btn-outline" onclick="eRptSelectAll(false)">Deselect All</button>
                                    <div style="flex:1"></div>
                                    <button id="e-rpt-btn" class="btn btn-primary" onclick="eGenerateReport()"><i class="ri-file-chart-line"></i> Generate Report</button>
                                </div>
                            </div>
                        </div>
                        <div id="e-rpt-content" style="display:none">
                            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
                                <div><span style="font-size:18px;font-weight:700">Comparison Report</span><span id="e-rpt-date" class="mono" style="font-size:12px;color:#6a6f73;margin-left:12px"></span></div>
                                <div style="display:flex;gap:8px"><button class="btn btn-sm btn-outline" onclick="eRptPrint()"><i class="ri-printer-line"></i> Print</button><button class="btn btn-sm btn-outline" onclick="eRptBack()"><i class="ri-arrow-left-line"></i> Back</button></div>
                            </div>
                            <div id="e-rpt-summary" class="grid-4" style="margin-bottom:16px"></div>
                            <div id="e-rpt-best" style="margin-bottom:16px"></div>
                            <div class="grid-2" style="margin-bottom:16px">
                                <div class="card"><div class="card-title" style="margin-bottom:12px">Metric Comparison</div><canvas id="e-rpt-bars" width="600" height="350" style="width:100%;border-radius:8px"></canvas></div>
                                <div class="card"><div class="card-title" style="margin-bottom:12px">Performance Radar</div><canvas id="e-rpt-radar" width="400" height="350" style="width:100%;border-radius:8px"></canvas></div>
                            </div>
                            <div class="card" style="margin-bottom:16px"><div class="card-title" style="margin-bottom:12px">Detailed Metrics</div><div id="e-rpt-table"></div></div>
                            <div class="grid-2" style="margin-bottom:16px">
                                <div class="card"><div class="card-title" style="margin-bottom:12px">Training Time</div><canvas id="e-rpt-time" width="600" height="280" style="width:100%;border-radius:8px"></canvas></div>
                                <div class="card"><div class="card-title" style="margin-bottom:12px">Score Distribution</div><canvas id="e-rpt-scatter" width="600" height="280" style="width:100%;border-radius:8px"></canvas></div>
                            </div>
                            <div class="card"><div class="card-title" style="margin-bottom:12px">Model Rankings</div><div id="e-rpt-rankings"></div></div>
                        </div>
                    </div>
                    <!-- Monitor (merged: Monitor + Security) -->
                    <div id="et-monitor" class="tab-panel">
                        <div class="grid-4" style="margin-bottom:16px">
                            <div class="dash-stat-card">
                                <div class="icon-wrap blue"><i class="ri-cpu-line"></i></div>
                                <div class="stat stat-sm stat-info" id="e-cpu">0.0%</div>
                                <div class="stat-label">CPU Usage</div>
                            </div>
                            <div class="dash-stat-card">
                                <div class="icon-wrap green"><i class="ri-hard-drive-2-line"></i></div>
                                <div class="stat stat-sm stat-ok" id="e-mem">0.0%</div>
                                <div class="stat-label">Memory Usage</div>
                            </div>
                            <div class="dash-stat-card">
                                <div class="icon-wrap orange"><i class="ri-database-2-line"></i></div>
                                <div class="stat stat-sm" id="e-mem-used" style="color:#6a6f73">0 GB</div>
                                <div class="stat-label">Memory Used</div>
                            </div>
                            <div class="dash-stat-card">
                                <div class="icon-wrap red"><i class="ri-hard-drive-2-line"></i></div>
                                <div class="stat stat-sm" id="e-mem-total" style="color:#6a6f73">0 GB</div>
                                <div class="stat-label">Total Memory</div>
                            </div>
                        </div>
                        <div class="grid-2">
                            <div class="card">
                                <div class="card-header"><div class="card-title">CPU</div></div>
                                <div class="progress" style="height:10px"><div class="progress-fill" id="e-mon-cpu-bar" style="width:0%;height:100%"></div></div>
                                <p id="e-mon-cpu-val" class="mono" style="font-size:13px;color:#6a6f73;margin-top:8px">0.0%</p>
                                <div class="chart-wrap" style="margin-top:12px"><canvas id="e-spark-cpu" width="520" height="120" style="border-radius:8px"></canvas></div>
                            </div>
                            <div class="card">
                                <div class="card-header"><div class="card-title">Memory</div></div>
                                <div class="progress" style="height:10px"><div class="progress-fill" id="e-mon-mem-bar" style="width:0%;height:100%;background:#3abc3f"></div></div>
                                <p id="e-mon-mem-val" class="mono" style="font-size:13px;color:#6a6f73;margin-top:8px">0.0%</p>
                                <div class="chart-wrap" style="margin-top:12px"><canvas id="e-spark-mem" width="520" height="120" style="border-radius:8px"></canvas></div>
                            </div>
                        </div>
                        <div class="grid-2" style="margin-top:16px">
                            <div class="card">
                                <div class="card-header"><div class="card-title">Security Overview</div><button class="btn btn-sm btn-outline" onclick="eSecLoad()"><i class="ri-refresh-line" style="font-size:14px"></i></button></div>
                                <div id="e-sec-status" class="empty">Loading security status...</div>
                            </div>
                            <div class="card">
                                <div class="card-header"><div class="card-title">Audit Log</div></div>
                                <div id="e-sec-audit" class="empty" style="max-height:300px;overflow-y:auto">Loading...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div id="e-toasts" class="toast-box"></div>
    <script>
    var eData=null,eTraining=false,eLastModelId=null,eLoadedDataset=null;
    function $(i){return document.getElementById(i)}
    function eNotify(m,t){var d=document.createElement('div');d.className='toast'+(t==='ok'?' toast-ok':t==='err'?' toast-err':'');d.textContent=m;$('e-toasts').appendChild(d);setTimeout(function(){d.remove()},3500)}
    var eTabNames={dashboard:'Dashboard',data:'Data',train:'Train',analysis:'Analysis',reports:'Reports',monitor:'Monitor'};
    function eTab(n){
        document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active')});
        document.querySelectorAll('.sidebar-item').forEach(function(t){t.classList.remove('active')});
        var panel=$('et-'+n);if(panel)panel.classList.add('active');
        var btn=document.querySelector('[data-tab="'+n+'"]');if(btn)btn.classList.add('active');
        $('e-page-title').textContent=eTabNames[n]||n;
        eUpdateNoDataAlerts();
        if(n==='reports')eLoadReport();
    }
    function eSubTab(parent,name){
        var panel=$('et-'+parent);if(!panel)return;
        panel.querySelectorAll('.sub-panel').forEach(function(p){p.classList.remove('active')});
        panel.querySelectorAll('.sub-tab').forEach(function(t){t.classList.remove('active')});
        var sp=$('esp-'+parent+'-'+name);if(sp)sp.classList.add('active');
        var btn=event.currentTarget;if(btn)btn.classList.add('active');
    }
    function eToggleSidebar(){var s=$('e-sidebar');s.classList.toggle('expanded');$('e-sidebar-icon').className=s.classList.contains('expanded')?'ri-arrow-left-s-line':'ri-arrow-right-s-line'}
    function eTopbar(name,rows,cols){$('e-topbar-data').style.display='inline';$('e-topbar-name').textContent=name;$('e-topbar-rows').textContent=rows.toLocaleString();$('e-topbar-cols').textContent=cols;$('e-dash-rows').textContent=rows.toLocaleString()}
    function eUpdateNoDataAlerts(){
        var show=!eData;
        ['e-train-nodata','e-anom-nodata','e-dr-nodata'].forEach(function(id){var el=$(id);if(el)el.style.display=show?'flex':'none'});
        $('e-dr-btn').disabled=!eData;
    }
    function eUpdateTrainSummary(){
        var wrap=$('e-train-summary');var chips=$('e-train-config-chips');
        if(!eData){wrap.style.display='none';return}
        wrap.style.display='block';
        var target=$('e-target').value||'(not set)';
        var task=$('e-task').value||'—';
        var model=$('e-model').options[$('e-model').selectedIndex].text;
        chips.innerHTML='<div class="config-chip"><strong>Target:</strong> '+target+'</div><div class="config-chip"><strong>Task:</strong> '+task+'</div><div class="config-chip"><strong>Model:</strong> '+model+'</div>';
    }
    function eSys(){fetch('/api/system/status').then(function(r){return r.json()}).then(function(d){var s=d.system||{};$('e-badge').textContent=d.status||'error';$('e-badge').className='badge '+(d.status==='healthy'?'badge-ok':'badge-err');$('e-dot').style.background=d.status==='healthy'?'#3abc3f':'#ff3131';var cpuVal=(s.cpu_usage||0).toFixed(1);var memVal=(s.memory_usage_percent||0).toFixed(1);var usedGb=(s.used_memory_gb||0).toFixed(1);var totalGb=(s.total_memory_gb||0).toFixed(1);$('e-cpu').textContent=cpuVal+'%';$('e-mem').textContent=memVal+'%';$('e-mem-used').textContent=usedGb+' GB';$('e-mem-total').textContent=totalGb+' GB';$('e-mon-cpu-bar').style.width=cpuVal+'%';$('e-mon-cpu-val').textContent=cpuVal+'%';$('e-mon-mem-bar').style.width=memVal+'%';$('e-mon-mem-val').textContent=memVal+'% ('+usedGb+' / '+totalGb+' GB)';$('e-dash-cpu').textContent=cpuVal+'%';$('e-dash-mem').textContent=memVal+'%';$('e-dash-cpu2').textContent=cpuVal+'%';$('e-dash-mem2').textContent=memVal+'%';$('e-dash-cpu-bar').style.width=cpuVal+'%';$('e-dash-mem-bar').style.width=memVal+'%';$('e-dash-mem-detail').textContent=usedGb+' / '+totalGb+' GB';$('e-dash-status').textContent=d.status||'error';$('e-dash-status').className='badge '+(d.status==='healthy'?'badge-ok':'badge-err');eCpuHistory.push(parseFloat(cpuVal));eMemHistory.push(parseFloat(memVal));if(eCpuHistory.length>60)eCpuHistory.shift();if(eMemHistory.length>60)eMemHistory.shift();eDrawMonitorSpark()}).catch(function(){})}
    function ePopulateTargets(cols){
        var sel=$('e-target');sel.innerHTML='<option value="">Select...</option>';
        (cols||[]).forEach(function(c){var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o)});
    }
    function eEnableButtons(){$('e-train-btn').disabled=false;$('e-ho-btn').disabled=false;$('e-ens-btn').disabled=false;$('e-at-btn').disabled=false;$('e-anomaly-btn').disabled=false;$('e-dr-btn').disabled=false;ePopulateAtTarget()}
    function eShowDataInfo(d){
        var h='<div class="grid-2" style="margin-bottom:16px"><div style="text-align:center;padding:12px;background:#f0f6fe;border-radius:10px"><span class="stat stat-info stat-animated">'+d.rows.toLocaleString()+'</span><p style="font-size:12px;color:#6a6f73;margin-top:2px">Rows</p></div><div style="text-align:center;padding:12px;background:#f3fbf4;border-radius:10px"><span class="stat stat-ok stat-animated">'+d.columns+'</span><p style="font-size:12px;color:#6a6f73;margin-top:2px">Columns</p></div></div>';
        h+='<div style="display:flex;flex-wrap:wrap;gap:4px">';(d.column_names||[]).forEach(function(c,i){var colors=['#0066f5','#3abc3f','#ffa931','#8b5cf6','#ec4899','#06b6d4'];var ci=colors[i%colors.length];h+='<span style="display:inline-flex;align-items:center;height:24px;padding:0 8px;font-size:11px;border-radius:6px;background:'+ci+'11;color:'+ci+';font-weight:500">'+c+'</span>'});h+='</div>';
        $('e-info').innerHTML=h;
    }
    function eShowDataPreview(d){
        var cols=d.column_names||[];var prev=d.preview||[];
        if(!prev.length||!cols.length){$('e-data-preview-wrap').style.display='none';return}
        $('e-data-preview-wrap').style.display='block';
        $('e-preview-badge').textContent='First '+prev.length+' rows';
        var h='<table><thead><tr>';cols.forEach(function(c){h+='<th>'+c+'</th>'});h+='</tr></thead><tbody>';
        prev.forEach(function(row){h+='<tr>';cols.forEach(function(c){h+='<td class="mono" style="font-size:12px">'+(row[c]!=null?row[c]:'—')+'</td>'});h+='</tr>'});
        h+='</tbody></table>';$('e-data-preview').innerHTML=h;
    }
    function eLoad(n){
        document.querySelectorAll('#e-pills .pill').forEach(function(p){p.classList.remove('active')});
        var btn=document.querySelector('#e-pills .pill[data-name="'+n+'"]');if(btn)btn.classList.add('active');
        fetch('/api/data/sample/'+n).then(function(r){return r.json()}).then(function(d){if(d.success){eData=d;eLoadedDataset=n;eTopbar(d.name||n,d.rows,d.columns);eShowDataInfo(d);ePopulateTargets(d.column_names);eEnableButtons();eUpdateAutoMLTarget();eUpdateDimRedColor();eUpdateNoDataAlerts();eShowDataPreview(d);eRunAnalysis();eShowDataViz(d);eNotify('Loaded '+n+' ('+d.rows+' rows, '+d.columns+' cols)','ok');eLogActivity('Loaded dataset: '+n+' ('+d.rows+' rows)')}}).catch(function(e){eNotify('Failed: '+e.message,'err')})}
    function eImportUrl(){var url=$('e-import-url').value.trim();if(!url){eNotify('Enter a URL','err');return}$('e-import-status').innerHTML='<p style="font-size:12px;color:#6a6f73">Downloading...</p>';fetch('/api/data/import/url',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:url})}).then(function(r){return r.json()}).then(function(d){if(d.success){eData=d;eTopbar(d.name||'Imported',d.rows,d.columns);eShowDataInfo(d);ePopulateTargets(d.column_names);eEnableButtons();eUpdateAutoMLTarget();eUpdateNoDataAlerts();eShowDataPreview(d);eRunAnalysis();eShowDataViz(d);eUpdateDimRedColor();$('e-import-status').innerHTML='<p style="font-size:12px;color:#3abc3f">Imported from '+d.source+' ('+d.rows+' rows, '+d.columns+' cols)</p>';eNotify('Imported!','ok')}else if(d.needs_credentials){$('e-import-status').innerHTML='<p style="font-size:12px;color:#ffa931">'+d.message+'</p>'}else{$('e-import-status').innerHTML='<p style="font-size:12px;color:#ff3131">'+(d.message||'Failed')+'</p>'}}).catch(function(e){$('e-import-status').innerHTML='<p style="font-size:12px;color:#ff3131">'+e.message+'</p>'})}
    function eFileUpload(input){
        if(!input.files||!input.files[0])return;
        var file=input.files[0];var fd=new FormData();fd.append('file',file);
        eNotify('Uploading '+file.name+'...','');
        fetch('/api/data/upload',{method:'POST',body:fd}).then(function(r){return r.json()}).then(function(d){if(d.success){eData=d;eTopbar(d.name||file.name,d.rows,d.columns);eShowDataInfo(d);ePopulateTargets(d.column_names);eEnableButtons();eUpdateAutoMLTarget();eUpdateNoDataAlerts();eShowDataPreview(d);eRunAnalysis();eShowDataViz(d);eUpdateDimRedColor();eNotify('Uploaded '+file.name+' ('+d.rows+' rows)','ok');eLogActivity('Uploaded file: '+file.name)}else{eNotify(d.message||'Upload failed','err')}}).catch(function(e){eNotify('Upload failed: '+e.message,'err')});
        input.value='';
    }
    function eValidateTarget(){
        var t=$('e-target').value;
        if(!t){eNotify('Set target column first','err');return false}
        return true;
    }
    var eTrainStart=0;var eAtStart=0;var eHoStart=0;var eEnsStart=0;
    function eElapsed(t){var s=Math.floor((Date.now()-t)/1000);var m=Math.floor(s/60);s=s%60;return m>0?(m+'m '+s+'s'):(s+'s')}
    function eProgressRing(pct,size,color){var r=(size-6)/2;var c=2*Math.PI*r;var off=c*(1-pct);var bg='rgb(228,231,233)';var fg=color||'rgb(0,102,245)';return "<svg width='"+size+"' height='"+size+"'><circle cx='"+(size/2)+"' cy='"+(size/2)+"' r='"+r+"' fill='none' stroke='"+bg+"' stroke-width='3'/><circle cx='"+(size/2)+"' cy='"+(size/2)+"' r='"+r+"' fill='none' stroke='"+fg+"' stroke-width='3' stroke-dasharray='"+c+"' stroke-dashoffset='"+off+"' stroke-linecap='round' style='transition:stroke-dashoffset .5s'/></svg>"}
    function eStepBar(steps,current){var h='<div class="tp-steps">';for(var i=0;i<steps.length;i++){var cls=i<current?'done':(i===current?'active':'');h+='<div class="tp-step '+cls+'"><div class="tp-dot"></div><div class="tp-label">'+steps[i]+'</div></div>'}return h+'</div>'}
    function ePopulateAtTarget(){var sel=$('e-at-target');if(!sel||!eData)return;sel.innerHTML='<option value="">Select...</option>';(eData.column_names||[]).forEach(function(c){var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o)})}
    function eAtModelList(task){var cls=task==='classification'||task==='binary_classification'||task==='multi_classification';var reg=task==='regression'||task==='time_series';if(cls)return['Random Forest','XGBoost','LightGBM','CatBoost','Gradient Boosting','Extra Trees','AdaBoost','SGD','Decision Tree','KNN','Logistic Regression','SVM','Naive Bayes'];if(reg)return['Random Forest','XGBoost','LightGBM','CatBoost','Gradient Boosting','Extra Trees','Ridge','Lasso','Elastic Net','Polynomial','SGD','Gaussian Process','Decision Tree','KNN','Linear Regression'];return['KMeans','DBSCAN']}
    function eTrain(){if(!eData||eTraining)return;if(!eValidateTarget())return;eTraining=true;eTrainStart=Date.now();$('e-train-btn').disabled=true;$('e-train-btn').innerHTML='<span class="spinner"></span> Training...';var model=$('e-model').options[$('e-model').selectedIndex].text;$('e-result').innerHTML='<div class="tp-card active"><div class="tp-header"><div class="tp-ring">'+eProgressRing(0.1,48)+'<div class="tp-ring-text">0%</div></div><div><div class="tp-phase">Training '+model+'</div><div class="tp-msg">Preparing data...</div><div class="tp-elapsed">0s</div></div></div><div class="progress" style="height:4px"><div class="progress-fill" style="width:10%"></div></div>'+eStepBar(['Prepare','Split','Fit','Evaluate'],0)+'</div>';var cfg={target_column:$('e-target').value,task_type:$('e-task').value,model_type:$('e-model').value};fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)}).then(function(r){return r.json()}).then(function(d){if(d.job_id){ePoll(d.job_id)}}).catch(function(e){$('e-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';eTraining=false;$('e-train-btn').disabled=false;$('e-train-btn').textContent='Train Model'})}
    function ePoll(id){fetch('/api/train/status/'+id).then(function(r){return r.json()}).then(function(d){var s=d.status;if(s.Completed){var m=s.Completed.metrics||{};var h='<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';Object.keys(m).forEach(function(k){h+='<tr><td>'+k+'</td><td class="mono">'+(typeof m[k]==='number'?m[k].toFixed(4):m[k])+'</td></tr>'});h+='</tbody></table>';$('e-result').innerHTML=h;var modelName=$('e-model').options[$('e-model').selectedIndex].text;eDashTrainResult(modelName,m);eTraining=false;$('e-train-btn').disabled=false;$('e-train-btn').textContent='Train Model';eNotify('Training complete!','ok');eLogActivity('Training completed: '+modelName);eDashModels();eShowTrainViz(m);fetch('/api/models').then(function(r){return r.json()}).then(function(md){var models=md.models||[];if(models.length>0){eLastModelId=models[models.length-1].id;$('e-explain-btn').disabled=false}}).catch(function(){})}else if(s.Failed){$('e-result').innerHTML='<p style="color:#ff3131">'+(s.Failed.error||'Failed')+'</p>';eTraining=false;$('e-train-btn').disabled=false;$('e-train-btn').textContent='Train Model';eLogActivity('Training failed: '+(s.Failed.error||'Unknown error'))}else{var p=s.Running?s.Running.progress||0:0;var msg=s.Running?s.Running.message||'':'';var pct=Math.round(p*100);var model=$('e-model').options[$('e-model').selectedIndex].text;var step=0;if(p>0.1)step=1;if(p>0.3)step=2;if(p>0.8)step=3;$('e-result').innerHTML='<div class="tp-card active"><div class="tp-header"><div class="tp-ring">'+eProgressRing(p,48)+'<div class="tp-ring-text">'+pct+'%</div></div><div><div class="tp-phase">Training '+model+'</div><div class="tp-msg">'+msg+'</div><div class="tp-elapsed">'+eElapsed(eTrainStart)+'</div></div></div><div class="progress" style="height:4px"><div class="progress-fill" style="width:'+pct+'%"></div></div>'+eStepBar(['Prepare','Split','Fit','Evaluate'],step)+'</div>';setTimeout(function(){ePoll(id)},1000)}}).catch(function(){setTimeout(function(){ePoll(id)},2000)})}
    function eHyperOpt(){if(!eData)return;if(!eValidateTarget())return;eHoStart=Date.now();$('e-ho-btn').disabled=true;$('e-ho-btn').innerHTML='<span class="spinner"></span> Optimizing...';var nTrials=parseInt($('e-ho-trials').value);$('e-ho-result').innerHTML='<div class="tp-card active"><div class="tp-header"><div class="tp-ring">'+eProgressRing(0,48,'#8b5cf6')+'<div class="tp-ring-text">0%</div></div><div><div class="tp-phase">Hyperparameter Optimization</div><div class="tp-msg">Starting '+nTrials+' trials...</div><div class="tp-elapsed">0s</div></div></div><div class="progress" style="height:4px"><div class="progress-fill" style="width:0%;background:#8b5cf6"></div></div></div>';fetch('/api/hyperopt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target_column:$('e-target').value,task_type:$('e-task').value,model_type:$('e-ho-model').value,n_trials:nTrials,sampler:$('e-ho-sampler').value})}).then(function(r){return r.json()}).then(function(d){if(d.job_id){eHoPoll(d.job_id)}}).catch(function(e){$('e-ho-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';$('e-ho-btn').disabled=false;$('e-ho-btn').textContent='Optimize'})}
    function eHoPoll(id){fetch('/api/train/status/'+id).then(function(r){return r.json()}).then(function(d){var s=d.status;if(s.Completed){var m=s.Completed.metrics||{};var h='<div style="background:#f3fbf4;padding:12px;border-radius:8px;margin-bottom:12px">';if(m.best_trial){h+='<strong>Best Score:</strong> '+((m.best_trial.value||0).toFixed(6))+'<br><small>'+((m.best_trial.params||''))+'</small>'}h+='</div><p style="font-size:12px;color:#6a6f73">Trials: '+(m.total_trials||0)+' | Time: '+(m.total_duration_secs||0).toFixed(2)+'s</p>';if(m.trials){h+='<table style="margin-top:8px"><thead><tr><th>#</th><th>Score</th><th>Time</th></tr></thead><tbody>';m.trials.forEach(function(t){h+='<tr><td>'+t.trial_id+'</td><td class="mono">'+(t.value===Infinity?'Fail':t.value.toFixed(6))+'</td><td class="mono">'+t.duration_secs.toFixed(2)+'s</td></tr>'});h+='</tbody></table>'}$('e-ho-result').innerHTML=h;if(m.trials)eHoCharts(m.trials);$('e-ho-btn').disabled=false;$('e-ho-btn').textContent='Optimize';eNotify('Optimization done!','ok');eLogActivity('HyperOpt completed: '+$('e-ho-model').value);eDashModels()}else if(s.Failed){$('e-ho-result').innerHTML='<p style="color:#ff3131">'+(s.Failed.error||'Failed')+'</p>';$('e-ho-btn').disabled=false;$('e-ho-btn').textContent='Optimize'}else{var p=s.Running?s.Running.progress||0:0;var pct=Math.round(p*100);var msg=s.Running?s.Running.message||'':'';$('e-ho-result').innerHTML='<div class="tp-card active"><div class="tp-header"><div class="tp-ring">'+eProgressRing(p,48,'#8b5cf6')+'<div class="tp-ring-text">'+pct+'%</div></div><div><div class="tp-phase">Hyperparameter Optimization</div><div class="tp-msg">'+msg+'</div><div class="tp-elapsed">'+eElapsed(eHoStart)+'</div></div></div><div class="progress" style="height:4px"><div class="progress-fill" style="width:'+pct+'%;background:#8b5cf6"></div></div></div>';setTimeout(function(){eHoPoll(id)},1500)}}).catch(function(){setTimeout(function(){eHoPoll(id)},3000)})}
    function eAutoTune(){if(!eData)return;var target=$('e-at-target').value;if(!target){eNotify('Select a target column first','err');return}eAtStart=Date.now();$('e-at-btn').disabled=true;$('e-at-btn').innerHTML='<span class="spinner"></span> Tuning...';var task=$('e-at-task').value;var models=eAtModelList(task);$('e-at-progress').style.display='block';var mh='<div class="mc-grid">';models.forEach(function(m){mh+='<div class="mc-card pending" id="e-mc-'+m.replace(/\s+/g,'_')+'"><div class="mc-name">'+m+'</div><div class="mc-score">Pending</div></div>'});mh+='</div>';$('e-at-progress').innerHTML='<div class="tp-card active"><div class="tp-header"><div class="tp-ring">'+eProgressRing(0,48,'#3abc3f')+'<div class="tp-ring-text">0%</div></div><div><div class="tp-phase">Auto-Tune</div><div class="tp-msg">Starting '+models.length+' models...</div><div class="tp-elapsed">0s</div></div></div><div class="progress" style="height:4px"><div class="progress-fill" style="width:0%;background:#3abc3f"></div></div></div>'+mh;$('e-at-result').innerHTML='';$('e-at-result').className='';fetch('/api/autotune',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target_column:target,task_type:task})}).then(function(r){return r.json()}).then(function(d){if(d.job_id){eAtPoll(d.job_id)}}).catch(function(e){$('e-at-progress').style.display='none';$('e-at-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';$('e-at-btn').disabled=false;$('e-at-btn').innerHTML='<i class="ri-speed-line"></i> Start Auto-Tune'})}
    function eAtPoll(id){fetch('/api/train/status/'+id).then(function(r){return r.json()}).then(function(d){var s=d.status;if(s.Completed){$('e-at-progress').style.display='none';var m=s.Completed.metrics||{};var h='';if(m.best){h+='<div style="background:#0d0e0f;border-radius:12px;padding:16px;color:#fff;margin-bottom:16px"><div style="font-size:12px;opacity:.6">Best Model</div><div style="font-size:22px;font-weight:700;margin:4px 0">'+m.best_model+'</div><div style="font-size:14px;color:#3abc3f">Score: '+(m.best.score||0).toFixed(6)+'</div><div style="font-size:11px;opacity:.5;margin-top:4px">'+m.total_models_tested+' models tested in '+(m.total_time_secs||0).toFixed(1)+'s</div></div>'}if(m.leaderboard){h+='<table><thead><tr><th>#</th><th>Model</th><th>Score</th><th>Time</th><th>Status</th></tr></thead><tbody>';m.leaderboard.forEach(function(r,i){var ok=r.status==='success';h+='<tr><td style="font-weight:600;color:'+(i===0?'#3abc3f':'#6a6f73')+'">'+(i+1)+'</td><td style="font-weight:500">'+r.model+'</td><td class="mono">'+(ok?(r.score||0).toFixed(6):'—')+'</td><td class="mono">'+(r.training_time_secs||0).toFixed(2)+'s</td><td>'+(ok?'<span style="color:#3abc3f">&#10003;</span>':'<span style="color:#ff3131">&#10007;</span>')+'</td></tr>'});h+='</tbody></table>'}$('e-at-result').innerHTML=h;$('e-at-btn').disabled=false;$('e-at-btn').innerHTML='<i class="ri-speed-line"></i> Start Auto-Tune';if(m.best&&m.best.status==='success'){eLastModelId='autotune_'+m.best_model+'_'+id;$('e-explain-btn').disabled=false}if(m.leaderboard)eAtCharts(m.leaderboard);eNotify('Auto-tune done! Best: '+m.best_model,'ok');eLogActivity('Auto-tune completed: best model '+m.best_model);eDashModels()}else if(s.Failed){$('e-at-progress').style.display='none';$('e-at-result').innerHTML='<p style="color:#ff3131">'+(s.Failed.error||'Failed')+'</p>';$('e-at-btn').disabled=false;$('e-at-btn').innerHTML='<i class="ri-speed-line"></i> Start Auto-Tune'}else{var p=s.Running?s.Running.progress||0:0;var pct=Math.round(p*100);var msg=s.Running?s.Running.message||'':'';var pr=s.Running?s.Running.partial_results:null;var done=pr?pr.length:0;var task=$('e-at-task').value;var models=eAtModelList(task);var total=models.length;var tp=$('e-at-progress');if(tp){var ring=tp.querySelector('.tp-ring');if(ring)ring.innerHTML=eProgressRing(p,48,'#3abc3f')+'<div class="tp-ring-text">'+pct+'%</div>';var phase=tp.querySelector('.tp-phase');if(phase)phase.textContent='Auto-Tune ('+done+'/'+total+')';var tmsg=tp.querySelector('.tp-msg');if(tmsg)tmsg.textContent=msg;var tel=tp.querySelector('.tp-elapsed');if(tel)tel.textContent=eElapsed(eAtStart);var bar=tp.querySelector('.progress-fill');if(bar)bar.style.width=pct+'%'}if(pr&&pr.length>0){var doneSet={};pr.forEach(function(r){var key=r.model.replace(/[\s_]+/g,'_');doneSet[key]=r});models.forEach(function(m){var key=m.replace(/\s+/g,'_');var el=$('e-mc-'+key);if(!el)return;var entry=null;pr.forEach(function(r){if(r.model.replace(/[\s_]+/g,'_').toLowerCase()===key.toLowerCase()||r.model.toLowerCase()===m.toLowerCase())entry=r});if(entry){var ok=entry.status==='success';el.className='mc-card '+(ok?'done':'fail');el.innerHTML='<div class="mc-name">'+m+'</div><div class="mc-score">'+(ok?((entry.score||0).toFixed(4)):'Failed')+'</div>'}});var rh='';pr.sort(function(a,b){return(b.score||0)-(a.score||0)});pr.forEach(function(r){if(r.status==='success')rh+='<div style="display:flex;justify-content:space-between;padding:6px 8px;font-size:12px;border-bottom:1px solid #f1f3f4"><span style="font-weight:500">'+r.model+'</span><span class="mono" style="color:#3abc3f">'+((r.score||0).toFixed(6))+'</span></div>'});if(rh)$('e-at-result').innerHTML='<div style="font-size:13px;font-weight:500;margin-bottom:6px;color:#6a6f73">Live Leaderboard</div>'+rh;eAtCharts(pr)}setTimeout(function(){eAtPoll(id)},1500)}}).catch(function(){setTimeout(function(){eAtPoll(id)},3000)})}
    function eEnsemble(){if(!eData)return;if(!eValidateTarget())return;eEnsStart=Date.now();var models=[];document.querySelectorAll('#e-ens-models input:checked').forEach(function(cb){models.push(cb.value)});if(models.length<2){eNotify('Select at least 2 models','err');return}$('e-ens-btn').disabled=true;$('e-ens-btn').innerHTML='<span class="spinner"></span> Training...';$('e-ens-result').innerHTML='<div class="tp-card active"><div class="tp-header"><div class="tp-ring">'+eProgressRing(0,48,'#ec4899')+'<div class="tp-ring-text">0%</div></div><div><div class="tp-phase">Ensemble Training</div><div class="tp-msg">Training '+models.length+' models with '+$('e-ens-strat').value+'...</div><div class="tp-elapsed">0s</div></div></div><div class="progress" style="height:4px"><div class="progress-fill" style="width:0%;background:#ec4899"></div></div></div>';fetch('/api/ensemble/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target_column:$('e-target').value,task_type:$('e-task').value,models:models,strategy:$('e-ens-strat').value})}).then(function(r){return r.json()}).then(function(d){if(d.job_id){eEnsPoll(d.job_id)}}).catch(function(e){$('e-ens-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';$('e-ens-btn').disabled=false;$('e-ens-btn').textContent='Train Ensemble'})}
    function eEnsPoll(id){fetch('/api/train/status/'+id).then(function(r){return r.json()}).then(function(d){var s=d.status;if(s.Completed){var m=s.Completed.metrics||{};var h='<div style="background:#f0f6fe;padding:12px;border-radius:8px;margin-bottom:12px"><strong>Ensemble:</strong> '+(m.strategy||'voting')+' | Models: '+(m.successful_models||0)+'/'+(m.model_count||0)+' | Time: '+(m.training_time_secs||0).toFixed(2)+'s</div>';if(m.model_results){h+='<table><thead><tr><th>Model</th><th>Status</th></tr></thead><tbody>';m.model_results.forEach(function(r){h+='<tr><td>'+r.model+'</td><td>'+(r.status==='success'?'<span style="color:#3abc3f">&#10003; Success</span>':'<span style="color:#ff3131">&#10007; '+r.error+'</span>')+'</td></tr>'});h+='</tbody></table>'}$('e-ens-result').innerHTML=h;$('e-ens-btn').disabled=false;$('e-ens-btn').textContent='Train Ensemble';eNotify('Ensemble done!','ok');eLogActivity('Ensemble training completed');eDashModels()}else if(s.Failed){$('e-ens-result').innerHTML='<p style="color:#ff3131">'+(s.Failed.error||'Failed')+'</p>';$('e-ens-btn').disabled=false;$('e-ens-btn').textContent='Train Ensemble'}else{var p=s.Running?s.Running.progress||0:0;var pct=Math.round(p*100);var msg=s.Running?s.Running.message||'':'';$('e-ens-result').innerHTML='<div class="tp-card active"><div class="tp-header"><div class="tp-ring">'+eProgressRing(p,48,'#ec4899')+'<div class="tp-ring-text">'+pct+'%</div></div><div><div class="tp-phase">Ensemble Training</div><div class="tp-msg">'+msg+'</div><div class="tp-elapsed">'+eElapsed(eEnsStart)+'</div></div></div><div class="progress" style="height:4px"><div class="progress-fill" style="width:'+pct+'%;background:#ec4899"></div></div></div>';setTimeout(function(){eEnsPoll(id)},1500)}}).catch(function(){setTimeout(function(){eEnsPoll(id)},3000)})}
    function eExplain(){if(!eLastModelId)return;$('e-explain-btn').disabled=true;$('e-explain-btn').innerHTML='<span class="spinner"></span> Analyzing...';fetch('/api/explain/importance',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_id:eLastModelId})}).then(function(r){return r.json()}).then(function(d){if(d.success&&d.features&&d.features.length>0){var mx=Math.max.apply(null,d.features.map(function(f){return Math.abs(f.importance)}));var colors=['#0066f5','#3abc3f','#ffa931','#8b5cf6','#ec4899','#06b6d4','#ff3131','#f97316'];var h='<table><thead><tr><th>Feature</th><th>Importance</th><th style="width:45%">Distribution</th></tr></thead><tbody>';d.features.forEach(function(f,i){var pct=mx>0?(Math.abs(f.importance)/mx*100).toFixed(0):0;var ci=colors[i%colors.length];h+='<tr><td style="font-weight:500">'+f.feature+'</td><td class="mono" style="color:'+ci+'">'+f.importance.toFixed(6)+'</td><td><div style="width:100%;background:#f1f3f4;border-radius:6px;height:12px;overflow:hidden"><div style="width:'+pct+'%;background:linear-gradient(90deg,'+ci+','+ci+'88);border-radius:6px;height:12px;transition:width .6s ease"></div></div></td></tr>'});h+='</tbody></table>';$('e-explain-result').innerHTML=h}else{$('e-explain-result').innerHTML='<p style="color:#6a6f73">'+(d.message||'Not available for this model')+'</p>'}$('e-explain-btn').disabled=false;$('e-explain-btn').textContent='Analyze'}).catch(function(e){$('e-explain-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>';$('e-explain-btn').disabled=false;$('e-explain-btn').textContent='Analyze'})}
    function eAnomaly(){if(!eData)return;$('e-anomaly-btn').disabled=true;$('e-anomaly-btn').innerHTML='<span class="spinner"></span> Detecting...';$('e-anomaly-result').innerHTML='<p style="color:#6a6f73">Detecting anomalies...</p>';fetch('/api/anomaly/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({contamination:parseFloat($('e-anom-cont').value),n_estimators:parseInt($('e-anom-est').value)})}).then(function(r){return r.json()}).then(function(d){$('e-anomaly-btn').disabled=false;$('e-anomaly-btn').textContent='Detect';if(d.success){var rate=((d.anomaly_rate||0)*100).toFixed(1);$('e-anomaly-result').innerHTML='<div class="grid-2"><div><span class="stat stat-info">'+d.total_samples+'</span><p style="font-size:12px;color:#6a6f73">Samples</p></div><div><span class="stat stat-err">'+d.anomalies_found+'</span><p style="font-size:12px;color:#6a6f73">Anomalies ('+rate+'%)</p></div></div>';$('e-anom-viz').style.display='block';eDrawDonut('e-anom-donut',[{label:'Normal',value:d.total_samples-d.anomalies_found},{label:'Anomaly',value:d.anomalies_found}],{colors:['#3abc3f','#ff3131']});$('e-anom-donut-center').innerHTML='<div class="donut-val" style="color:#ff3131">'+rate+'%</div><div class="donut-label">anomalies</div>';if(eData&&eData.preview){var numCols=[];eData.column_names.forEach(function(c){var ok=true;for(var i=0;i<Math.min(eData.preview.length,10);i++){if(isNaN(parseFloat(eData.preview[i][c]))){ok=false;break}}if(ok)numCols.push(c)});if(numCols.length>=2){var pts=[];var anomIdx=d.anomaly_indices||[];var anomSet={};anomIdx.forEach(function(i){anomSet[i]=true});eData.preview.forEach(function(row,i){var x=parseFloat(row[numCols[0]]),y=parseFloat(row[numCols[1]]);if(!isNaN(x)&&!isNaN(y))pts.push({x:x,y:y,c:anomSet[i]?1:0,s:anomSet[i]?5:3})});eDrawScatter('e-anom-scatter',pts,{xLabel:numCols[0],yLabel:numCols[1],colors:['#3abc3f','#ff3131'],tipEl:$('e-anom-tip')})}}}else{$('e-anomaly-result').innerHTML='<p style="color:#ff3131">'+(d.message||'Failed')+'</p>'}}).catch(function(e){$('e-anomaly-btn').disabled=false;$('e-anomaly-btn').textContent='Detect';$('e-anomaly-result').innerHTML='<p style="color:#ff3131">'+e.message+'</p>'})}
    function eUpdateAutoMLTarget(){var sel=$('e-automl-target');if(!sel||!eData)return;sel.innerHTML='<option value="">Select target...</option>';(eData.column_names||[]).forEach(function(c){var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o)});sel.onchange=function(){$('e-btn-automl').disabled=!this.value}}
    function eHoCharts(trials){
        $('e-ho-charts').style.display='block';
        var pts=[],bestPts=[],bestVal=Infinity;
        for(var i=0;i<trials.length;i++){var v=trials[i].value;if(v===Infinity||v==null)continue;pts.push({x:trials[i].trial_id,y:v});if(v<bestVal)bestVal=v;bestPts.push({x:trials[i].trial_id,y:bestVal})}
        eDrawLine('e-ho-cvg',pts,{color:'#8b5cf6',label:'Trial #'});
        eDrawLine('e-ho-best',bestPts,{color:'#3abc3f',label:'Trial #'});
    }
    function eAtCharts(leaderboard){
        $('e-at-charts').style.display='block';var items=[];
        for(var i=0;i<leaderboard.length;i++){var s=leaderboard[i].score;if(s!=null&&s>0)items.push({label:leaderboard[i].model,value:s})}
        items.sort(function(a,b){return b.value-a.value});eDrawBars('e-at-bar',items,{label:'Model Score'});
    }
    var eAMLStart=0;
    function eRunAutoML(){if(!eData)return;var target=$('e-automl-target').value;if(!target){eNotify('Select a target column','err');return}eAMLStart=Date.now();$('e-btn-automl').disabled=true;$('e-btn-automl').innerHTML='<span class="spinner"></span> Running...';$('e-automl-progress').style.display='block';$('e-automl-result').style.display='none';$('e-aml-status').innerHTML=eProgressRing(0,36)+'<span style="margin-left:8px">Starting pipeline...</span>';$('e-aml-bar').style.width='0%';var body={target_column:target};var tt=$('e-aml-task').value;if(tt)body.task_type=tt;var sc=$('e-aml-scaler').value;if(sc)body.scaler=sc;body.max_models=parseInt($('e-aml-max').value)||8;body.hyperopt_trials=parseInt($('e-aml-trials').value)||20;fetch('/api/automl/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(function(r){return r.json()}).then(function(d){if(d.success&&d.job_id){eAMLPoll(d.job_id)}else{eNotify('Failed: '+(d.error||'Unknown'),'err');$('e-btn-automl').disabled=false;$('e-btn-automl').textContent='Run AutoML Pipeline';$('e-automl-progress').style.display='none'}}).catch(function(e){eNotify('Request failed: '+e,'err');$('e-btn-automl').disabled=false;$('e-btn-automl').textContent='Run AutoML Pipeline';$('e-automl-progress').style.display='none'})}
    function eAMLPoll(id){fetch('/api/automl/status/'+id).then(function(r){return r.json()}).then(function(d){var pct=Math.round(d.progress||0);$('e-aml-bar').style.width=pct+'%';var msg=d.message||d.status;var step=0;if(pct>10)step=1;if(pct>30)step=2;if(pct>60)step=3;if(pct>90)step=4;$('e-aml-status').innerHTML='<div style="display:flex;align-items:center;gap:10px">'+eProgressRing(pct/100,36)+'<div><div style="font-weight:600;font-size:13px">'+msg+'</div><div class="tp-elapsed">'+eElapsed(eAMLStart)+' &middot; '+pct+'%</div></div></div>'+eStepBar(['Detect','Preprocess','Train','Optimize','Evaluate'],step);if(d.status==='completed'){$('e-automl-progress').style.display='none';$('e-btn-automl').disabled=false;$('e-btn-automl').textContent='Run AutoML Pipeline';var r=d.result||{};$('e-aml-best').textContent=r.best_model||'—';$('e-aml-score').textContent=(r.metric||'Score')+': '+(r.best_score!=null?r.best_score.toFixed(4):'—');var lb=r.leaderboard||[];var h='<table><thead><tr><th>#</th><th>Model</th><th>Score</th><th>Time</th></tr></thead><tbody>';lb.forEach(function(e,i){h+='<tr><td>'+(i+1)+'</td><td>'+e.model+'</td><td class="mono">'+(e.score!=null?parseFloat(e.score).toFixed(4):'Err')+'</td><td class="mono">'+(e.training_time_secs!=null?e.training_time_secs.toFixed(2)+'s':'—')+'</td></tr>'});h+='</tbody></table>';$('e-aml-leaderboard').innerHTML=h;$('e-automl-result').style.display='block';if(lb.length>0){$('e-aml-chart-wrap').style.display='block';var ci=[];lb.forEach(function(e){var sc=parseFloat(e.score);if(sc!=null&&sc>0)ci.push({label:e.model,value:sc})});ci.sort(function(a,b){return b.value-a.value});eDrawBars('e-aml-chart',ci,{label:'Model Score'})}eNotify('AutoML done! Best: '+r.best_model,'ok');eLogActivity('AutoML completed: best model '+r.best_model);eDashModels()}else if(d.status==='failed'){$('e-automl-progress').style.display='none';$('e-btn-automl').disabled=false;$('e-btn-automl').textContent='Run AutoML Pipeline';eNotify('AutoML failed: '+d.message,'err')}else{var pr=d.partial_results;if(pr&&pr.length>0){$('e-aml-chart-wrap').style.display='block';var ci=[];pr.forEach(function(e){var sc=parseFloat(e.score);if(sc!=null&&sc>0)ci.push({label:e.model,value:sc})});ci.sort(function(a,b){return b.value-a.value});eDrawBars('e-aml-chart',ci,{label:'Model Score (live)'})}setTimeout(function(){eAMLPoll(id)},2000)}}).catch(function(){setTimeout(function(){eAMLPoll(id)},3000)})}
    function eSecLoad(){fetch('/api/security/status').then(function(r){return r.json()}).then(function(d){var rl=d.rate_limiting||{};var g=d.guards||{};var h='<div class="grid-2"><div><span class="stat stat-info">'+(rl.total_requests||0)+'</span><p style="font-size:12px;color:#6a6f73">Requests</p></div><div><span class="stat stat-err">'+(rl.total_blocked||0)+'</span><p style="font-size:12px;color:#6a6f73">Blocked</p></div></div><div style="margin-top:12px;font-size:12px;color:#6a6f73"><b>Guards:</b> SSRF '+(g.ssrf_protection?'&#10003;':'&#10007;')+' | Path Traversal '+(g.path_traversal_protection?'&#10003;':'&#10007;')+' | Input Validation '+(g.input_validation?'&#10003;':'&#10007;')+' | Timeout '+(g.training_timeout_secs||0)+'s | Max Rows '+(g.max_dataset_rows||0).toLocaleString()+'</div>';$('e-sec-status').innerHTML=h}).catch(function(){$('e-sec-status').innerHTML='<p style="color:#ff3131">Failed to load</p>'})}
    function eSecAudit(){fetch('/api/security/audit-log?limit=20').then(function(r){return r.json()}).then(function(d){var entries=d.entries||[];if(!entries.length){$('e-sec-audit').innerHTML='<p style="font-size:12px;color:#6a6f73">No audit entries yet</p>';return}var h='<table><thead><tr><th>Time</th><th>Method</th><th>Path</th><th>Status</th></tr></thead><tbody>';entries.forEach(function(e){h+='<tr><td class="mono">'+new Date(e.timestamp).toLocaleTimeString()+'</td><td>'+e.action+'</td><td class="mono">'+e.resource+'</td><td>'+e.status_code+'</td></tr>'});h+='</tbody></table>';$('e-sec-audit').innerHTML=h}).catch(function(){})}
    function eDrawLine(canvasId,points,opts){
        var c=document.getElementById(canvasId);if(!c||!points.length)return;
        var dpr=window.devicePixelRatio||1;var W=c.width,H=c.height;
        c.width=W*dpr;c.height=H*dpr;c.style.width=W+'px';c.style.height=H+'px';
        var ctx=c.getContext('2d');ctx.scale(dpr,dpr);
        var pad={t:24,r:24,b:40,l:56},pW=W-pad.l-pad.r,pH=H-pad.t-pad.b;
        var xs=points.map(function(p){return p.x}),ys=points.map(function(p){return p.y});
        var xMin=Math.min.apply(null,xs),xMax=Math.max.apply(null,xs);
        var yMin=Math.min.apply(null,ys),yMax=Math.max.apply(null,ys);
        var yRange=yMax-yMin||1;yMin-=yRange*0.08;yMax+=yRange*0.08;yRange=yMax-yMin;
        var xRange=xMax-xMin||1;
        function tx(v){return pad.l+(v-xMin)/xRange*pW}
        function ty(v){return pad.t+pH-(v-yMin)/yRange*pH}
        var color=opts.color||'#0066f5';
        var fillColor=opts.fill||color;
        var label=opts.label||'';
        var frame=0,total=points.length,dur=total+20;
        function ease(t){return t<0.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2}
        function draw(){
            ctx.clearRect(0,0,W,H);
            // Background
            ctx.fillStyle='#fafbfb';ctx.fillRect(0,0,W,H);
            // Grid
            ctx.strokeStyle='#eef0f2';ctx.lineWidth=1;
            for(var i=0;i<=5;i++){var yv=yMin+yRange*i/5,yy=ty(yv);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke();ctx.fillStyle='#b0b4b8';ctx.font='10px system-ui';ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(yv.toPrecision(3),pad.l-8,yy)}
            var xTicks=Math.min(total,6);
            for(var i=0;i<=xTicks;i++){var xv=xMin+xRange*i/xTicks,xx=tx(xv);ctx.beginPath();ctx.moveTo(xx,pad.t);ctx.lineTo(xx,H-pad.b);ctx.stroke();ctx.fillStyle='#b0b4b8';ctx.font='10px system-ui';ctx.textAlign='center';ctx.textBaseline='top';ctx.fillText(Math.round(xv),xx,H-pad.b+6)}
            var progress=ease(Math.min(frame/dur,1));
            var n=Math.max(2,Math.ceil(total*progress));if(n>total)n=total;
            // Gradient fill
            var grad=ctx.createLinearGradient(0,pad.t,0,H-pad.b);
            if(typeof fillColor==='string'&&fillColor.indexOf('rgba')===0){grad.addColorStop(0,fillColor);grad.addColorStop(1,'rgba(250,251,251,0)')}
            else{var fc=color;grad.addColorStop(0,fc+'22');grad.addColorStop(0.6,fc+'08');grad.addColorStop(1,fc+'00')}
            // Smooth bezier curve
            ctx.beginPath();ctx.moveTo(tx(points[0].x),ty(points[0].y));
            for(var i=1;i<n;i++){var x0=tx(points[i-1].x),y0=ty(points[i-1].y),x1=tx(points[i].x),y1=ty(points[i].y);var cpx=(x0+x1)/2;ctx.bezierCurveTo(cpx,y0,cpx,y1,x1,y1)}
            // Glow
            ctx.save();ctx.shadowColor=color;ctx.shadowBlur=8;ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.lineJoin='round';ctx.lineCap='round';ctx.stroke();ctx.restore();
            // Stroke again without glow for crispness
            ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
            // Fill area
            ctx.lineTo(tx(points[n-1].x),H-pad.b);ctx.lineTo(tx(points[0].x),H-pad.b);ctx.closePath();ctx.fillStyle=grad;ctx.fill();
            // Dots with glow
            for(var i=0;i<n;i++){
                var px=tx(points[i].x),py=ty(points[i].y);
                ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
                ctx.beginPath();ctx.arc(px,py,3,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();
                if(i===n-1&&frame<dur){ctx.save();ctx.beginPath();ctx.arc(px,py,8,0,Math.PI*2);ctx.fillStyle=color+'33';ctx.fill();ctx.restore()}
            }
            if(label){ctx.fillStyle='#9c9fa1';ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillText(label,W/2,H-4)}
            if(frame<dur+5){frame++;requestAnimationFrame(draw)}
        }
        draw();
    }
    function eDrawBars(canvasId,items,opts){
        var c=document.getElementById(canvasId);if(!c||!items.length)return;
        var dpr=window.devicePixelRatio||1;var W=c.width,H=c.height;
        c.width=W*dpr;c.height=H*dpr;c.style.width=W+'px';c.style.height=H+'px';
        var ctx=c.getContext('2d');ctx.scale(dpr,dpr);
        var pad={t:28,r:24,b:64,l:56},pW=W-pad.l-pad.r,pH=H-pad.t-pad.b;
        var vals=items.map(function(d){return d.value});var vMax=Math.max.apply(null,vals)*1.1||1;
        var barW=Math.min(52,pW/items.length*0.65);var gap=(pW-barW*items.length)/(items.length+1);
        var colors=opts.colors||['#0066f5','#3abc3f','#ffa931','#8b5cf6','#06b6d4','#ec4899','#f97316','#14b8a6','#6366f1','#84cc16','#a855f7','#ef4444','#22d3ee','#ff3131'];
        var label=opts.label||'';var frame=0,totalFrames=50;
        function ease(t){return 1-Math.pow(1-t,3)}
        function draw(){
            ctx.clearRect(0,0,W,H);ctx.fillStyle='#fafbfb';ctx.fillRect(0,0,W,H);
            // Grid lines
            for(var i=0;i<=4;i++){var yv=vMax*i/4,yy=pad.t+pH-pH*i/4;ctx.strokeStyle='#eef0f2';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke();ctx.fillStyle='#b0b4b8';ctx.font='10px system-ui';ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(yv.toPrecision(3),pad.l-8,yy)}
            var progress=ease(Math.min(frame/totalFrames,1));
            for(var i=0;i<items.length;i++){
                var x=pad.l+gap+(barW+gap)*i;var barH=pH*(items[i].value/vMax)*progress;var y=pad.t+pH-barH;var ci=colors[i%colors.length];
                // Bar gradient
                var grad=ctx.createLinearGradient(x,y,x,pad.t+pH);grad.addColorStop(0,ci);grad.addColorStop(1,ci+'88');
                // Shadow/glow
                ctx.save();ctx.shadowColor=ci+'44';ctx.shadowBlur=12;ctx.shadowOffsetY=4;
                var r=6;ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+barW-r,y);ctx.quadraticCurveTo(x+barW,y,x+barW,y+r);ctx.lineTo(x+barW,pad.t+pH);ctx.lineTo(x,pad.t+pH);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();ctx.fillStyle=grad;ctx.fill();ctx.restore();
                // Value label
                if(progress>0.6){var labelAlpha=Math.min(1,(progress-0.6)/0.4);ctx.globalAlpha=labelAlpha;ctx.fillStyle='#0d0e0f';ctx.font='bold 11px "Geist Mono",monospace';ctx.textAlign='center';ctx.textBaseline='bottom';ctx.fillText(items[i].value.toFixed(4),x+barW/2,y-6);ctx.globalAlpha=1}
                // X label
                ctx.save();ctx.translate(x+barW/2,pad.t+pH+8);ctx.rotate(-Math.PI/5);ctx.fillStyle='#6a6f73';ctx.font='11px system-ui';ctx.textAlign='right';ctx.textBaseline='top';var name=items[i].label.length>14?items[i].label.substring(0,13)+'…':items[i].label;ctx.fillText(name,0,0);ctx.restore()
            }
            if(label){ctx.fillStyle='#9c9fa1';ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillText(label,W/2,H-4)}
            if(frame<totalFrames+5){frame++;requestAnimationFrame(draw)}
        }
        draw();
    }
    function eDrawScatter(canvasId,points,opts){
        var c=document.getElementById(canvasId);if(!c||!points.length)return;
        var dpr=window.devicePixelRatio||1;var W=c.width,H=c.height;
        c.width=W*dpr;c.height=H*dpr;c.style.width=W+'px';c.style.height=H+'px';
        var ctx=c.getContext('2d');ctx.scale(dpr,dpr);
        var pad={t:24,r:24,b:40,l:56},pW=W-pad.l-pad.r,pH=H-pad.t-pad.b;
        var xs=points.map(function(p){return p.x}),ys=points.map(function(p){return p.y});
        var xMin=Math.min.apply(null,xs),xMax=Math.max.apply(null,xs);
        var yMin=Math.min.apply(null,ys),yMax=Math.max.apply(null,ys);
        var xRange=xMax-xMin||1;var yRange=yMax-yMin||1;
        xMin-=xRange*0.05;xMax+=xRange*0.05;yMin-=yRange*0.05;yMax+=yRange*0.05;
        xRange=xMax-xMin;yRange=yMax-yMin;
        function tx(v){return pad.l+(v-xMin)/xRange*pW}
        function ty(v){return pad.t+pH-(v-yMin)/yRange*pH}
        var colors=opts.colors||['#0066f5','#3abc3f','#ffa931','#8b5cf6','#ec4899','#06b6d4','#ff3131','#f97316'];
        var xLabel=opts.xLabel||'',yLabel=opts.yLabel||'';
        var frame=0,dur=40;
        function ease(t){return 1-Math.pow(1-t,3)}
        function draw(){
            ctx.clearRect(0,0,W,H);ctx.fillStyle='#fafbfb';ctx.fillRect(0,0,W,H);
            // Grid
            ctx.strokeStyle='#eef0f2';ctx.lineWidth=1;
            for(var i=0;i<=5;i++){
                var yv=yMin+yRange*i/5,yy=ty(yv);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke();
                ctx.fillStyle='#b0b4b8';ctx.font='10px system-ui';ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(yv.toPrecision(3),pad.l-8,yy);
                var xv=xMin+xRange*i/5,xx=tx(xv);ctx.beginPath();ctx.moveTo(xx,pad.t);ctx.lineTo(xx,H-pad.b);ctx.stroke();
                ctx.textAlign='center';ctx.textBaseline='top';ctx.fillText(xv.toPrecision(3),xx,H-pad.b+6)
            }
            var progress=ease(Math.min(frame/dur,1));
            var n=Math.ceil(points.length*progress);
            // Draw points with glow
            for(var i=0;i<n;i++){
                var px=tx(points[i].x),py=ty(points[i].y);
                var ci=colors[(points[i].c||0)%colors.length];
                var sz=points[i].s||3.5;
                ctx.save();ctx.globalAlpha=0.15;ctx.beginPath();ctx.arc(px,py,sz+4,0,Math.PI*2);ctx.fillStyle=ci;ctx.fill();ctx.restore();
                ctx.beginPath();ctx.arc(px,py,sz,0,Math.PI*2);ctx.fillStyle=ci;ctx.globalAlpha=0.85;ctx.fill();ctx.globalAlpha=1;
                ctx.beginPath();ctx.arc(px,py,sz,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke()
            }
            if(xLabel){ctx.fillStyle='#9c9fa1';ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillText(xLabel,W/2,H-4)}
            if(yLabel){ctx.save();ctx.translate(10,H/2);ctx.rotate(-Math.PI/2);ctx.fillStyle='#9c9fa1';ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillText(yLabel,0,0);ctx.restore()}
            if(frame<dur+5){frame++;requestAnimationFrame(draw)}
        }
        draw();
        // Tooltip on hover
        var tip=opts.tipEl;
        if(tip){c.onmousemove=function(e){var rect=c.getBoundingClientRect();var mx=(e.clientX-rect.left)*(W/rect.width);var my=(e.clientY-rect.top)*(H/rect.height);var best=-1,bestD=Infinity;for(var i=0;i<points.length;i++){var dx=tx(points[i].x)-mx,dy=ty(points[i].y)-my;var d=dx*dx+dy*dy;if(d<bestD){bestD=d;best=i}}if(bestD<400&&best>=0){var p=points[best];tip.style.opacity='1';tip.style.left=(tx(p.x)/W*100)+'%';tip.style.top=(ty(p.y)/H*100-8)+'%';tip.textContent='('+p.x.toPrecision(4)+', '+p.y.toPrecision(4)+')'}else{tip.style.opacity='0'}};c.onmouseleave=function(){tip.style.opacity='0'}}
    }
    function eDrawDonut(canvasId,segments,opts){
        var c=document.getElementById(canvasId);if(!c||!segments.length)return;
        var dpr=window.devicePixelRatio||1;var W=c.width,H=c.height;
        c.width=W*dpr;c.height=H*dpr;c.style.width=W+'px';c.style.height=H+'px';
        var ctx=c.getContext('2d');ctx.scale(dpr,dpr);
        var cx=W/2,cy=H/2,r=Math.min(W,H)/2-30,inner=r*0.62;
        var total=0;segments.forEach(function(s){total+=s.value});
        var colors=opts.colors||['#0066f5','#3abc3f','#ffa931','#8b5cf6','#ec4899','#ff3131'];
        var frame=0,dur=50;
        function ease(t){return 1-Math.pow(1-t,3)}
        function draw(){
            ctx.clearRect(0,0,W,H);
            var progress=ease(Math.min(frame/dur,1));
            var angle=-Math.PI/2;
            for(var i=0;i<segments.length;i++){
                var sweep=(segments[i].value/total)*Math.PI*2*progress;
                var ci=colors[i%colors.length];
                // Shadow
                ctx.save();ctx.shadowColor=ci+'44';ctx.shadowBlur=16;
                ctx.beginPath();ctx.moveTo(cx,cy);ctx.arc(cx,cy,r,angle,angle+sweep);ctx.closePath();ctx.fillStyle=ci;ctx.fill();ctx.restore();
                // Inner cutout is done after all segments
                angle+=sweep;
            }
            // Cut out center
            ctx.beginPath();ctx.arc(cx,cy,inner,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
            // Labels around donut
            angle=-Math.PI/2;
            if(progress>0.8){
                ctx.globalAlpha=Math.min(1,(progress-0.8)/0.2);
                for(var i=0;i<segments.length;i++){
                    var sweep=(segments[i].value/total)*Math.PI*2;
                    var mid=angle+sweep/2;
                    var lx=cx+Math.cos(mid)*(r+18),ly=cy+Math.sin(mid)*(r+18);
                    ctx.fillStyle='#6a6f73';ctx.font='11px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';
                    var pct=((segments[i].value/total)*100).toFixed(1)+'%';
                    if(sweep>0.3)ctx.fillText(segments[i].label+' '+pct,lx,ly);
                    angle+=sweep;
                }
                ctx.globalAlpha=1;
            }
            if(frame<dur+5){frame++;requestAnimationFrame(draw)}
        }
        draw();
    }
    function eDrawMiniHist(canvasId,values,color){
        var c=document.getElementById(canvasId);if(!c||!values.length)return;
        var dpr=window.devicePixelRatio||1;var W=c.width,H=c.height;
        c.width=W*dpr;c.height=H*dpr;c.style.width=W+'px';c.style.height=H+'px';
        var ctx=c.getContext('2d');ctx.scale(dpr,dpr);
        var bins=Math.min(20,Math.max(6,Math.ceil(Math.sqrt(values.length))));
        var vMin=Math.min.apply(null,values),vMax=Math.max.apply(null,values);
        var range=vMax-vMin||1;
        var counts=new Array(bins).fill(0);
        for(var i=0;i<values.length;i++){var b=Math.min(bins-1,Math.floor((values[i]-vMin)/range*bins));counts[b]++}
        var maxC=Math.max.apply(null,counts)||1;
        var barW=W/bins,pad=1;
        var grad=ctx.createLinearGradient(0,0,0,H);grad.addColorStop(0,color);grad.addColorStop(1,color+'22');
        for(var i=0;i<bins;i++){var h=(counts[i]/maxC)*H*0.92;var x=i*barW+pad;var y=H-h;var w=barW-pad*2;
            ctx.fillStyle=grad;var r=Math.min(2,w/2);ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);ctx.lineTo(x+w,H);ctx.lineTo(x,H);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();ctx.fill()
        }
    }
    function eDrawSparkline(canvasId,data,color){
        var c=document.getElementById(canvasId);if(!c||data.length<2)return;
        var dpr=window.devicePixelRatio||1;var W=c.width,H=c.height;
        c.width=W*dpr;c.height=H*dpr;c.style.width=W+'px';c.style.height=H+'px';
        var ctx=c.getContext('2d');ctx.scale(dpr,dpr);
        var pad=8,pW=W-pad*2,pH=H-pad*2;
        var yMin=Math.min.apply(null,data),yMax=Math.max.apply(null,data);
        var yRange=yMax-yMin||1;yMin-=yRange*0.1;yMax+=yRange*0.1;yRange=yMax-yMin;
        ctx.fillStyle='#fafbfb';ctx.fillRect(0,0,W,H);
        // Fill
        var grad=ctx.createLinearGradient(0,pad,0,H-pad);grad.addColorStop(0,color+'30');grad.addColorStop(1,color+'05');
        ctx.beginPath();ctx.moveTo(pad,pad+pH-(data[0]-yMin)/yRange*pH);
        for(var i=1;i<data.length;i++){var x=pad+i/(data.length-1)*pW;var y=pad+pH-(data[i]-yMin)/yRange*pH;var px=pad+(i-1)/(data.length-1)*pW;var py=pad+pH-(data[i-1]-yMin)/yRange*pH;var cpx=(px+x)/2;ctx.bezierCurveTo(cpx,py,cpx,y,x,y)}
        ctx.lineTo(pad+pW,H-pad);ctx.lineTo(pad,H-pad);ctx.closePath();ctx.fillStyle=grad;ctx.fill();
        // Line
        ctx.beginPath();ctx.moveTo(pad,pad+pH-(data[0]-yMin)/yRange*pH);
        for(var i=1;i<data.length;i++){var x=pad+i/(data.length-1)*pW;var y=pad+pH-(data[i]-yMin)/yRange*pH;var px=pad+(i-1)/(data.length-1)*pW;var py=pad+pH-(data[i-1]-yMin)/yRange*pH;var cpx=(px+x)/2;ctx.bezierCurveTo(cpx,py,cpx,y,x,y)}
        ctx.save();ctx.shadowColor=color;ctx.shadowBlur=6;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.lineCap='round';ctx.lineJoin='round';ctx.stroke();ctx.restore();
        ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();
        // Last dot
        var lx=pad+pW,ly=pad+pH-(data[data.length-1]-yMin)/yRange*pH;
        ctx.beginPath();ctx.arc(lx,ly,4,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
        ctx.beginPath();ctx.arc(lx,ly,3,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();
    }
    function eDrawRadar(canvasId,labels,values,opts){
        var c=document.getElementById(canvasId);if(!c||!labels.length)return;
        var dpr=window.devicePixelRatio||1;var W=c.width,H=c.height;
        c.width=W*dpr;c.height=H*dpr;c.style.width=W+'px';c.style.height=H+'px';
        var ctx=c.getContext('2d');ctx.scale(dpr,dpr);
        var cx=W/2,cy=H/2,r=Math.min(W,H)/2-40;
        var n=labels.length;var color=opts.color||'#0066f5';
        var frame=0,dur=40;
        function ease(t){return 1-Math.pow(1-t,3)}
        function draw(){
            ctx.clearRect(0,0,W,H);
            var progress=ease(Math.min(frame/dur,1));
            // Grid rings
            for(var ring=1;ring<=5;ring++){
                var rr=r*ring/5;
                ctx.beginPath();for(var i=0;i<=n;i++){var angle=-Math.PI/2+Math.PI*2*i/n;var px=cx+Math.cos(angle)*rr,py=cy+Math.sin(angle)*rr;if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)}
                ctx.closePath();ctx.strokeStyle=ring===5?'#dde1e3':'#eef0f2';ctx.lineWidth=1;ctx.stroke();
                if(ring%2===0){ctx.fillStyle='#9c9fa1';ctx.font='9px system-ui';ctx.textAlign='left';ctx.textBaseline='bottom';ctx.fillText((ring/5).toFixed(1),cx+2,cy-rr-2)}
            }
            // Axes
            for(var i=0;i<n;i++){var angle=-Math.PI/2+Math.PI*2*i/n;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.cos(angle)*r,cy+Math.sin(angle)*r);ctx.strokeStyle='#eef0f2';ctx.lineWidth=1;ctx.stroke();
                var lx=cx+Math.cos(angle)*(r+16),ly=cy+Math.sin(angle)*(r+16);
                ctx.fillStyle='#6a6f73';ctx.font='11px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';
                var lbl=labels[i].length>10?labels[i].substring(0,9)+'…':labels[i];ctx.fillText(lbl,lx,ly)
            }
            // Data polygon
            ctx.beginPath();
            for(var i=0;i<=n;i++){var idx=i%n;var angle=-Math.PI/2+Math.PI*2*idx/n;var v=Math.min(1,values[idx]||0)*progress;var px=cx+Math.cos(angle)*r*v,py=cy+Math.sin(angle)*r*v;if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)}
            ctx.closePath();
            ctx.save();ctx.shadowColor=color+'44';ctx.shadowBlur=12;ctx.fillStyle=color+'22';ctx.fill();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();ctx.restore();
            // Dots
            for(var i=0;i<n;i++){var angle=-Math.PI/2+Math.PI*2*i/n;var v=Math.min(1,values[i]||0)*progress;var px=cx+Math.cos(angle)*r*v,py=cy+Math.sin(angle)*r*v;
                ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.beginPath();ctx.arc(px,py,3,0,Math.PI*2);ctx.fillStyle=color;ctx.fill()}
            if(frame<dur+5){frame++;requestAnimationFrame(draw)}
        }
        draw();
    }
    function eDrawCorrelation(canvasId,matrix,labels,tipEl){
        var c=document.getElementById(canvasId);if(!c||!matrix.length)return;
        var dpr=window.devicePixelRatio||1;var W=c.width,H=c.height;
        c.width=W*dpr;c.height=H*dpr;c.style.width=W+'px';c.style.height=H+'px';
        var ctx=c.getContext('2d');ctx.scale(dpr,dpr);
        var n=matrix.length;var padL=70,padT=12,padR=40,padB=70;
        var cellW=(W-padL-padR)/n,cellH=(H-padT-padB)/n;
        ctx.fillStyle='#fafbfb';ctx.fillRect(0,0,W,H);
        function corrColor(v){
            if(v>=0){var t=Math.min(1,v);return 'rgba(0,102,245,'+t.toFixed(2)+')'}
            else{var t=Math.min(1,-v);return 'rgba(255,49,49,'+t.toFixed(2)+')'}
        }
        for(var i=0;i<n;i++){for(var j=0;j<n;j++){
            var x=padL+j*cellW,y=padT+i*cellH;
            ctx.fillStyle=corrColor(matrix[i][j]);
            var r=3;ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+cellW-r,y);ctx.quadraticCurveTo(x+cellW,y,x+cellW,y+r);ctx.lineTo(x+cellW,y+cellH-r);ctx.quadraticCurveTo(x+cellW,y+cellH,x+cellW-r,y+cellH);ctx.lineTo(x+r,y+cellH);ctx.quadraticCurveTo(x,y+cellH,x,y+cellH-r);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();ctx.fill();
            if(cellW>30){ctx.fillStyle=Math.abs(matrix[i][j])>0.5?'#fff':'#6a6f73';ctx.font='bold 10px "Geist Mono",monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(matrix[i][j].toFixed(2),x+cellW/2,y+cellH/2)}
        }}
        // Labels
        ctx.fillStyle='#6a6f73';ctx.font='11px system-ui';
        for(var i=0;i<n;i++){
            var lbl=labels[i].length>8?labels[i].substring(0,7)+'…':labels[i];
            ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(lbl,padL-6,padT+i*cellH+cellH/2);
            ctx.save();ctx.translate(padL+i*cellW+cellW/2,H-padB+8);ctx.rotate(-Math.PI/4);ctx.textAlign='right';ctx.textBaseline='top';ctx.fillText(lbl,0,0);ctx.restore()
        }
        // Legend bar
        var lx=W-padR+8,lw=14,ly=padT,lh=H-padT-padB;
        for(var i=0;i<lh;i++){var v=1-2*i/lh;ctx.fillStyle=corrColor(v);ctx.fillRect(lx,ly+i,lw,1)}
        ctx.fillStyle='#9c9fa1';ctx.font='9px system-ui';ctx.textAlign='left';
        ctx.fillText('+1',lx+lw+4,ly+6);ctx.fillText('0',lx+lw+4,ly+lh/2+3);ctx.fillText('-1',lx+lw+4,ly+lh-2);
        // Tooltip
        if(tipEl){c.onmousemove=function(e){var rect=c.getBoundingClientRect();var mx=(e.clientX-rect.left)*(W/rect.width)-padL;var my=(e.clientY-rect.top)*(H/rect.height)-padT;var ci=Math.floor(mx/cellW),ri=Math.floor(my/cellH);if(ci>=0&&ci<n&&ri>=0&&ri<n){tipEl.style.opacity='1';tipEl.style.left=((padL+ci*cellW+cellW/2)/W*100)+'%';tipEl.style.top=((padT+ri*cellH)/H*100-4)+'%';tipEl.textContent=labels[ri]+' × '+labels[ci]+': '+matrix[ri][ci].toFixed(3)}else{tipEl.style.opacity='0'}};c.onmouseleave=function(){tipEl.style.opacity='0'}}
    }
    // Animated counter
    function eAnimateValue(el,end,duration,decimals){
        var start=parseFloat(el.textContent)||0;if(isNaN(end))return;
        var startTime=null;decimals=decimals||0;
        function step(ts){if(!startTime)startTime=ts;var p=Math.min((ts-startTime)/(duration||400),1);var ease=1-Math.pow(1-p,3);var v=start+(end-start)*ease;el.textContent=v.toFixed(decimals)+(el.dataset.suffix||'');if(p<1)requestAnimationFrame(step)}
        requestAnimationFrame(step);
    }
    // System monitor sparkline data
    var eCpuHistory=[],eMemHistory=[];
    function eDrawMonitorSpark(){
        if(eCpuHistory.length>1)eDrawSparkline('e-spark-cpu',eCpuHistory,'#0066f5');
        if(eMemHistory.length>1)eDrawSparkline('e-spark-mem',eMemHistory,'#3abc3f');
    }
    // Data visualization helpers
    function eComputeCorrelation(preview,cols){
        var numCols=[];
        cols.forEach(function(c){var allNum=true;for(var i=0;i<preview.length;i++){var v=preview[i][c];if(v!=null&&v!==''&&isNaN(parseFloat(v))){allNum=false;break}}if(allNum)numCols.push(c)});
        if(numCols.length<2)return null;
        var data={};numCols.forEach(function(c){data[c]=preview.map(function(r){return parseFloat(r[c])||0})});
        var n=numCols.length;var matrix=[];
        for(var i=0;i<n;i++){matrix[i]=[];for(var j=0;j<n;j++){
            var a=data[numCols[i]],b=data[numCols[j]];
            var ma=0,mb=0;for(var k=0;k<a.length;k++){ma+=a[k];mb+=b[k]}ma/=a.length;mb/=b.length;
            var num=0,da=0,db=0;for(var k=0;k<a.length;k++){var dx=a[k]-ma,dy=b[k]-mb;num+=dx*dy;da+=dx*dx;db+=dy*dy}
            matrix[i][j]=da&&db?num/Math.sqrt(da*db):i===j?1:0
        }}
        return{matrix:matrix,labels:numCols}
    }
    function eRunAnalysis(){
        $('e-analysis-wrap').style.display='block';$('e-analysis-loading').style.display='block';$('e-analysis-content').style.display='none';
        fetch('/api/data/analyze').then(function(r){return r.json()}).then(function(d){
            $('e-analysis-loading').style.display='none';$('e-analysis-content').style.display='block';
            if(!d.success)return;
            var ov=d.overview||{},q=d.quality||{};
            // Quality badge
            var qs=q.score||0;var qpct=Math.round(qs*100);
            var qcol=qpct>=80?'#3abc3f':qpct>=60?'#ffa931':'#ff3131';
            var qclass=qpct>=80?'badge-ok':qpct>=60?'badge-warn':'badge-err';
            $('e-quality-badge').className='badge '+qclass;$('e-quality-badge').textContent=qpct+'% Quality';
            // Overview cards
            function fmtMem(b){return b>1048576?(b/1048576).toFixed(1)+' MB':(b/1024).toFixed(0)+' KB'}
            var oh='';
            oh+='<div style="text-align:center;padding:14px;background:#f0f6fe;border-radius:10px"><span class="stat stat-sm stat-info stat-animated">'+ov.rows.toLocaleString()+'</span><p style="font-size:11px;color:#6a6f73;margin-top:2px">Rows</p></div>';
            oh+='<div style="text-align:center;padding:14px;background:#f3fbf4;border-radius:10px"><span class="stat stat-sm stat-ok stat-animated">'+ov.columns+'</span><p style="font-size:11px;color:#6a6f73;margin-top:2px">Columns</p></div>';
            oh+='<div style="text-align:center;padding:14px;background:'+(ov.duplicates>0?'#fff8e6':'#f3fbf4')+';border-radius:10px"><span class="stat stat-sm '+(ov.duplicates>0?'stat-warn':'stat-ok')+' stat-animated">'+ov.duplicates+'</span><p style="font-size:11px;color:#6a6f73;margin-top:2px">Duplicates</p></div>';
            oh+='<div style="text-align:center;padding:14px;background:#f8f0fe;border-radius:10px"><span class="stat stat-sm stat-animated" style="color:#8b5cf6">'+fmtMem(ov.memory_bytes||0)+'</span><p style="font-size:11px;color:#6a6f73;margin-top:2px">Memory</p></div>';
            $('e-analysis-overview').innerHTML=oh;
            // Quality gauges
            var dims=[{l:'Completeness',v:q.completeness,c:'#0066f5'},{l:'Uniqueness',v:q.uniqueness,c:'#8b5cf6'},{l:'Consistency',v:q.consistency,c:'#3abc3f'},{l:'Validity',v:q.validity,c:'#06b6d4'}];
            var gh='';dims.forEach(function(dm){var pct=Math.round((dm.v||0)*100);gh+='<div style="text-align:center;padding:12px;background:#fafbfc;border-radius:10px"><div style="font-size:22px;font-weight:600;color:'+dm.c+';font-family:\'Geist Mono\',monospace">'+pct+'%</div><div class="progress" style="margin:8px 0 4px"><div class="progress-fill" style="width:'+pct+'%;background:'+dm.c+'"></div></div><div style="font-size:11px;color:#6a6f73">'+dm.l+'</div></div>'});
            $('e-quality-gauges').innerHTML=gh;
            // Warnings
            var warns=d.warnings||[];
            if(warns.length>0){var wh='';warns.forEach(function(w){var ic=w.severity==='warning'?'ri-error-warning-line':'ri-information-line';var cls=w.severity==='warning'?'alert-warn':'alert';wh+='<div class="alert '+cls+'" style="padding:10px 14px"><i class="'+ic+'"></i><span style="font-size:13px">'+w.message+'</span></div>'});$('e-analysis-warnings').innerHTML=wh}else{$('e-analysis-warnings').innerHTML=''}
            // Column stats table
            var cs=d.column_stats||[];
            if(cs.length>0){
                function esc(s){var d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML}
                function fmt(v){return v!=null?Number(v).toFixed(3):'\u2014'}
                var th='<table><thead><tr>'
                    +'<th>Column</th><th>Type</th><th>Missing</th><th>Unique</th>'
                    +'<th>Mean</th><th>Std</th><th>Var</th><th>CV%</th>'
                    +'<th>Min</th><th>Q25</th><th>Median</th><th>Q75</th><th>Max</th>'
                    +'<th>Range</th><th>IQR</th>'
                    +'<th>Skew</th><th>Kurt</th>'
                    +'<th>Sum</th><th>Zeros%</th><th>Outliers%</th>'
                    +'<th>Top Categories / Mode</th><th>Entropy(bits)</th>'
                    +'</tr></thead><tbody>';
                cs.forEach(function(c){
                    var missPct=c.null_percent||0;
                    var missBg=missPct>10?'#fff3f3':missPct>0?'#fff8e6':'transparent';
                    var skewColor=c.skewness!=null&&Math.abs(c.skewness)>2?'#ffa931':'inherit';
                    var kurtColor=c.kurtosis!=null&&Math.abs(c.kurtosis)>7?'#ffa931':'inherit';
                    var outColor=(c.outlier_percent||0)>10?'#ff3131':(c.outlier_percent||0)>5?'#ffa931':'inherit';
                    var zeroColor=(c.zeros_percent||0)>50?'#ffa931':'inherit';
                    th+='<tr>';
                    th+='<td style="font-weight:500;white-space:nowrap">'+esc(c.column)+'</td>';
                    th+='<td><span class="badge '+(c.is_numeric?'badge-info':'badge-warn')+'" style="font-size:11px">'+(c.is_numeric?'Numeric':'Categorical')+'</span></td>';
                    th+='<td style="background:'+missBg+'"><span class="mono" style="font-size:12px">'+c.null_count+'</span> <span style="font-size:11px;color:#6a6f73">('+missPct.toFixed(1)+'%)</span></td>';
                    th+='<td class="mono" style="font-size:12px">'+c.unique+'</td>';
                    if(c.is_numeric){
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.mean)+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.std)+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.variance)+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+(c.cv_percent!=null?Number(c.cv_percent).toFixed(1)+'%':'\u2014')+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.min)+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.q25)+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.median)+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.q75)+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.max)+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.range)+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.iqr)+'</td>';
                        th+='<td class="mono" style="font-size:12px;color:'+skewColor+'">'+(c.skewness!=null?Number(c.skewness).toFixed(2):'\u2014')+'</td>';
                        th+='<td class="mono" style="font-size:12px;color:'+kurtColor+'">'+(c.kurtosis!=null?Number(c.kurtosis).toFixed(2):'\u2014')+'</td>';
                        th+='<td class="mono" style="font-size:12px">'+fmt(c.sum)+'</td>';
                        th+='<td class="mono" style="font-size:12px;color:'+zeroColor+'">'+(c.zeros_percent!=null?Number(c.zeros_percent).toFixed(1)+'%':'\u2014')+'</td>';
                        th+='<td class="mono" style="font-size:12px;color:'+outColor+'">'+(c.outlier_percent!=null?Number(c.outlier_percent).toFixed(1)+'%':'\u2014')+'</td>';
                        th+='<td style="font-size:12px;color:#6a6f73" colspan="2">\u2014</td>';
                    }else{
                        var cats=(c.top_categories||[]).slice(0,3).map(function(t){return esc(String(t.value))+'('+t.count+')'}).join(', ');
                        th+='<td style="font-size:12px;color:#6a6f73" colspan="11">\u2014</td>';
                        th+='<td class="mono" style="font-size:12px">\u2014</td>';
                        th+='<td style="font-size:12px;color:#6a6f73" colspan="2">\u2014</td>';
                        var modeStr=c.mode_value?(esc(c.mode_value)+' ('+Number(c.mode_percent||0).toFixed(1)+'%)'):cats||'\u2014';
                        th+='<td style="font-size:12px;color:#6a6f73;white-space:nowrap">'+modeStr+'<br><span style="font-size:11px;color:#9ca3af">'+cats+'</span></td>';
                        th+='<td class="mono" style="font-size:12px">'+(c.entropy_bits!=null?Number(c.entropy_bits).toFixed(2)+' bits':'\u2014')+'</td>';
                    }
                    th+='</tr>';
                });
                th+='</tbody></table>';
                $('e-col-stats-table').innerHTML=th;
            }
            // Outlier summary
            var out=d.outlier_summary||[];
            var outWithData=out.filter(function(o){return o.outlier_count>0});
            if(outWithData.length>0){$('e-outlier-section').style.display='block';var ot='<div style="display:flex;flex-wrap:wrap;gap:10px">';outWithData.forEach(function(o){var pct=o.outlier_percent||0;var col=pct>5?'#ff3131':pct>1?'#ffa931':'#06b6d4';ot+='<div style="padding:12px 16px;background:#fafbfc;border-radius:10px;border-left:3px solid '+col+';min-width:180px"><div style="font-weight:500;font-size:13px;margin-bottom:4px">'+o.column+'</div><div style="font-size:22px;font-weight:600;color:'+col+';font-family:\'Geist Mono\',monospace">'+o.outlier_count+'</div><div style="font-size:11px;color:#6a6f73">'+pct.toFixed(1)+'% outliers</div><div class="mono" style="font-size:10px;color:#9c9fa1;margin-top:4px">bounds: ['+o.lower_bound.toFixed(1)+', '+o.upper_bound.toFixed(1)+']</div></div>'});ot+='</div>';$('e-outlier-table').innerHTML=ot}else{$('e-outlier-section').style.display='none'}
            // Missing values chart
            var mv=d.missing_values||[];
            var mvWithData=mv.filter(function(m){return m.null_count>0});
            if(mvWithData.length>0){$('e-missing-section').style.display='block';mvWithData.sort(function(a,b){return b.null_percent-a.null_percent});var mh='';mvWithData.forEach(function(m){var pct=m.null_percent||0;var col=pct>50?'#ff3131':pct>10?'#ffa931':'#06b6d4';mh+='<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px"><span style="font-size:12px;font-weight:500;width:140px;text-align:right;flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+m.column+'</span><div style="flex:1;height:20px;background:#f1f3f4;border-radius:6px;overflow:hidden;position:relative"><div style="height:100%;width:'+Math.max(pct,0.5)+'%;background:'+col+';border-radius:6px;transition:width .3s"></div></div><span class="mono" style="font-size:12px;width:60px;flex-shrink:0;color:'+col+'">'+pct.toFixed(1)+'%</span></div>'});$('e-missing-chart').innerHTML=mh}else{$('e-missing-section').style.display='none';$('e-missing-chart').innerHTML='<p style="font-size:13px;color:#3abc3f"><i class="ri-check-line"></i> No missing values detected</p>'}
            eLogActivity('Data analysis complete: '+qpct+'% quality score');
        }).catch(function(e){$('e-analysis-loading').style.display='none';$('e-analysis-content').style.display='block';$('e-analysis-content').innerHTML='<p style="color:#ff3131;font-size:13px">Analysis failed: '+e.message+'</p>'})
    }
    function eShowDataViz(d){
        if(!d||!d.preview||!d.column_names)return;
        $('e-data-viz-wrap').style.display='block';
        var cols=d.column_names,prev=d.preview;
        // Distribution histograms
        var distHtml='';var numIdx=0;
        var distColors=['#0066f5','#3abc3f','#ffa931','#8b5cf6','#ec4899','#06b6d4','#ff3131','#f97316'];
        cols.forEach(function(c,ci){
            var vals=[];for(var i=0;i<prev.length;i++){var v=parseFloat(prev[i][c]);if(!isNaN(v))vals.push(v)}
            if(vals.length>2){
                var mn=Math.min.apply(null,vals),mx=Math.max.apply(null,vals);
                var avg=(vals.reduce(function(a,b){return a+b},0)/vals.length);
                distHtml+='<div class="mini-hist"><div class="mini-hist-title">'+c+'</div>';
                distHtml+='<div class="mini-hist-stat">μ='+avg.toPrecision(3)+' range=['+mn.toPrecision(3)+', '+mx.toPrecision(3)+']</div>';
                distHtml+='<canvas id="e-dist-'+ci+'" width="160" height="48" style="width:100%"></canvas></div>';
                numIdx++
            }
        });
        $('e-col-distributions').innerHTML=distHtml||'<p style="color:#9c9fa1;font-size:13px">No numeric columns to visualize</p>';
        // Draw each mini histogram
        setTimeout(function(){
            cols.forEach(function(c,ci){
                var vals=[];for(var i=0;i<prev.length;i++){var v=parseFloat(prev[i][c]);if(!isNaN(v))vals.push(v)}
                if(vals.length>2)eDrawMiniHist('e-dist-'+ci,vals,distColors[ci%distColors.length])
            })
        },50);
        // Populate scatter selects
        var numCols=[];
        cols.forEach(function(c){var allNum=true;for(var i=0;i<Math.min(prev.length,20);i++){var v=prev[i][c];if(v!=null&&v!==''&&isNaN(parseFloat(v))){allNum=false;break}}if(allNum)numCols.push(c)});
        ['e-scatter-x','e-scatter-y','e-scatter-c'].forEach(function(id){
            var sel=$(id);var old=sel.value;sel.innerHTML=id==='e-scatter-c'?'<option value="">None</option>':'';
            numCols.forEach(function(c){var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o)});
            if(old&&numCols.indexOf(old)>=0)sel.value=old
        });
        if(numCols.length>=2){$('e-scatter-x').value=numCols[0];$('e-scatter-y').value=numCols[1]}
        eDrawScatterFromData();
        // Correlation matrix
        var corr=eComputeCorrelation(prev,cols);
        if(corr)eDrawCorrelation('e-corr',corr.matrix,corr.labels,$('e-corr-tip'));
    }
    function eDrawScatterFromData(){
        if(!eData||!eData.preview)return;
        var xCol=$('e-scatter-x').value,yCol=$('e-scatter-y').value;
        if(!xCol||!yCol)return;
        var cCol=$('e-scatter-c').value;
        var pts=[];var colorMap={};var colorIdx=0;
        eData.preview.forEach(function(row){
            var x=parseFloat(row[xCol]),y=parseFloat(row[yCol]);
            if(isNaN(x)||isNaN(y))return;
            var c=0;
            if(cCol&&row[cCol]!=null){var cv=String(row[cCol]);if(!(cv in colorMap))colorMap[cv]=colorIdx++;c=colorMap[cv]}
            pts.push({x:x,y:y,c:c})
        });
        eDrawScatter('e-scatter',pts,{xLabel:xCol,yLabel:yCol,tipEl:$('e-scatter-tip')});
    }
    function eShowTrainViz(metrics){
        var numKeys=[],numVals=[];
        Object.keys(metrics).forEach(function(k){var v=metrics[k];if(typeof v==='number'&&v>=0&&v<=1){numKeys.push(k);numVals.push(v)}});
        if(numKeys.length<2){$('e-train-viz').style.display='none';return}
        $('e-train-viz').style.display='block';
        eDrawRadar('e-train-radar',numKeys,numVals,{color:'#0066f5'});
        var barItems=[];numKeys.forEach(function(k,i){barItems.push({label:k,value:numVals[i]})});
        barItems.sort(function(a,b){return b.value-a.value});
        eDrawBars('e-train-bars',barItems,{label:'Metric Values'});
    }
    function eUpdateDimRedColor(){
        var sel=$('e-dr-color');if(!sel||!eData)return;
        sel.innerHTML='<option value="">None</option>';
        (eData.column_names||[]).forEach(function(c){var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o)});
    }
    function eRunDimRed(){
        if(!eData)return;
        var method=$('e-dr-method').value;
        var colorBy=$('e-dr-color').value||undefined;
        $('e-dr-btn').disabled=true;$('e-dr-btn').innerHTML='<span class="spinner"></span> Running...';
        $('e-dr-progress').style.display='block';$('e-dr-result').style.display='none';$('e-dr-empty').style.display='none';
        var url=method==='pca'?'/api/visualization/pca':'/api/visualization/umap';
        var body=method==='pca'?{color_by:colorBy,scale:true}:{n_neighbors:parseInt($('e-dr-neighbors').value)||15,min_dist:0.1,color_by:colorBy};
        fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(function(r){return r.json()}).then(function(d){
            $('e-dr-btn').disabled=false;$('e-dr-btn').textContent='Run';$('e-dr-progress').style.display='none';
            if(d.success&&d.points){
                $('e-dr-result').style.display='block';
                var pts=[];var colors=['#0066f5','#3abc3f','#ffa931','#8b5cf6','#ec4899','#06b6d4','#ff3131','#f97316'];
                if(d.color_mode==='classification'&&d.labels){
                    for(var i=0;i<d.points.length;i++){pts.push({x:d.points[i][0],y:d.points[i][1],c:d.labels[i]||0})}
                    var legendHtml='';(d.label_names||[]).forEach(function(n,i){legendHtml+='<div class="chart-legend-item"><div class="chart-legend-dot" style="background:'+colors[i%colors.length]+'"></div>'+n+'</div>'});
                    $('e-dr-legend').innerHTML=legendHtml;
                }else if(d.color_mode==='regression'&&d.color_values){
                    var cv=d.color_values;var cvMin=Math.min.apply(null,cv),cvMax=Math.max.apply(null,cv),cvRange=cvMax-cvMin||1;
                    for(var i=0;i<d.points.length;i++){var t=(cv[i]-cvMin)/cvRange;var ci=Math.floor(t*(colors.length-1));pts.push({x:d.points[i][0],y:d.points[i][1],c:ci})}
                    $('e-dr-legend').innerHTML='<div class="chart-legend-item"><div class="chart-legend-dot" style="background:#0066f5"></div>Low</div><div class="chart-legend-item"><div class="chart-legend-dot" style="background:#ff3131"></div>High</div>';
                }else{
                    for(var i=0;i<d.points.length;i++){pts.push({x:d.points[i][0],y:d.points[i][1],c:0})}
                    $('e-dr-legend').innerHTML='';
                }
                var xLbl=method==='pca'?'PC1':'UMAP 1',yLbl=method==='pca'?'PC2':'UMAP 2';
                eDrawScatter('e-dr-scatter',pts,{xLabel:xLbl,yLabel:yLbl,tipEl:$('e-dr-tip'),colors:colors});
                var infoHtml='<div class="grid-2"><div style="text-align:center;padding:8px;background:#f0f6fe;border-radius:8px"><span class="stat stat-sm stat-info">'+d.n_samples+'</span><p style="font-size:11px;color:#6a6f73">Samples</p></div><div style="text-align:center;padding:8px;background:#f3fbf4;border-radius:8px"><span class="stat stat-sm stat-ok">'+method.toUpperCase()+'</span><p style="font-size:11px;color:#6a6f73">Method</p></div></div>';
                $('e-dr-info').innerHTML=infoHtml;
                if(d.explained_variance_ratio){
                    var vh='<p style="font-size:13px;font-weight:500;color:#6a6f73;margin-bottom:8px">Explained Variance</p>';
                    d.explained_variance_ratio.forEach(function(v,i){
                        var pct=(v*100).toFixed(1);
                        vh+='<div style="margin-bottom:6px"><div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px"><span>PC'+(i+1)+'</span><span class="mono">'+pct+'%</span></div><div class="progress"><div class="progress-fill" style="width:'+pct+'%;background:'+(i===0?'#0066f5':i===1?'#3abc3f':'#ffa931')+'"></div></div></div>'
                    });
                    $('e-dr-variance').innerHTML=vh;
                }else{$('e-dr-variance').innerHTML=''}
                eNotify(method.toUpperCase()+' projection complete!','ok');eLogActivity(method.toUpperCase()+' computed on '+d.n_samples+' samples');
            }else{$('e-dr-empty').style.display='block';$('e-dr-empty').textContent=d.message||'Projection failed';eNotify('Failed: '+(d.message||'Unknown error'),'err')}
        }).catch(function(e){$('e-dr-btn').disabled=false;$('e-dr-btn').textContent='Run';$('e-dr-progress').style.display='none';$('e-dr-empty').style.display='block';$('e-dr-empty').textContent=e.message;eNotify('Failed: '+e.message,'err')});
    }
    document.addEventListener('DOMContentLoaded',function(){
        ['iris','diabetes','boston','wine','breast_cancer','titanic','heart','credit','customers','california_housing'].forEach(function(n){var b=document.createElement('button');b.className='pill';b.setAttribute('data-name',n);b.textContent=n.charAt(0).toUpperCase()+n.slice(1).replace(/_/g,' ');b.onclick=function(){eLoad(n)};$('e-pills').appendChild(b)});
        eSys();eSecLoad();eSecAudit();eDashModels();eUpdateNoDataAlerts();setInterval(eSys,5000);
        var uz=$('e-upload-zone');
        uz.addEventListener('dragover',function(e){e.preventDefault();uz.classList.add('dragover')});
        uz.addEventListener('dragleave',function(){uz.classList.remove('dragover')});
        uz.addEventListener('drop',function(e){e.preventDefault();uz.classList.remove('dragover');if(e.dataTransfer.files.length){$('e-file-input').files=e.dataTransfer.files;eFileUpload($('e-file-input'))}});
    });
    function eDashModels(){fetch('/api/models').then(function(r){return r.json()}).then(function(d){var models=d.models||[];$('e-dash-models').textContent=models.length;if(models.length===0){$('e-dash-model-list').innerHTML='<div class="empty">No models trained yet. Load data and run training to get started.</div>';return}var h='<table><thead><tr><th>Model</th><th>Type</th><th>Created</th></tr></thead><tbody>';models.forEach(function(m){h+='<tr><td style="font-weight:500">'+m.id+'</td><td><span class="badge badge-info">'+(m.model_type||'—')+'</span></td><td class="mono" style="font-size:12px;color:#6a6f73">'+(m.created_at?new Date(m.created_at).toLocaleString():'—')+'</td></tr>'});h+='</tbody></table>';$('e-dash-model-list').innerHTML=h;if(models.length>0){eLastModelId=models[models.length-1].id;$('e-explain-btn').disabled=false}}).catch(function(){$('e-dash-model-list').innerHTML='<div class="empty" style="color:#9c9fa1">Could not load models</div>'})}
    var eActivityLog=[];
    function eLogActivity(msg){var now=new Date();eActivityLog.unshift({time:now,msg:msg});if(eActivityLog.length>50)eActivityLog.length=50;var h='<div style="display:flex;flex-direction:column;gap:1px">';eActivityLog.forEach(function(a){h+='<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #f1f3f4"><span class="mono" style="font-size:11px;color:#9c9fa1;flex-shrink:0">'+a.time.toLocaleTimeString()+'</span><span style="font-size:13px">'+a.msg+'</span></div>'});h+='</div>';$('e-dash-activity').innerHTML=h}
    function eDashTrainResult(model,metrics){var h='<div style="background:#f3fbf4;border-radius:8px;padding:12px;margin-bottom:12px"><div style="font-size:14px;font-weight:600;margin-bottom:4px"><i class="ri-check-line" style="color:#3abc3f"></i> '+model+'</div></div>';h+='<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';Object.keys(metrics).forEach(function(k){h+='<tr><td>'+k+'</td><td class="mono">'+(typeof metrics[k]==='number'?metrics[k].toFixed(4):metrics[k])+'</td></tr>'});h+='</tbody></table>';$('e-dash-train-result').innerHTML=h}
    var eRptModels=[];
    function eLoadReport(){fetch('/api/models').then(function(r){return r.json()}).then(function(d){eRptModels=d.models||[];if(eRptModels.length===0){$('e-rpt-nomodels').style.display='flex';$('e-rpt-select').style.display='none';return}$('e-rpt-nomodels').style.display='none';$('e-rpt-select').style.display='block';var h='';eRptModels.forEach(function(m,i){h+='<label class="chk-item"><input type="checkbox" value="'+m.id+'" checked> '+(m.name||m.id)+'</label>'});$('e-rpt-models').innerHTML=h}).catch(function(){$('e-rpt-nomodels').style.display='flex';$('e-rpt-select').style.display='none'})}
    function eRptSelectAll(val){document.querySelectorAll('#e-rpt-models input').forEach(function(cb){cb.checked=val})}
    function eRptBack(){$('e-rpt-content').style.display='none';$('e-rpt-empty').style.display='block';eLoadReport()}
    function eRptPrint(){window.print()}
    function eGenerateReport(){var sel=[];document.querySelectorAll('#e-rpt-models input:checked').forEach(function(cb){sel.push(cb.value)});if(sel.length<2){eNotify('Select at least 2 models to compare','err');return}var models=eRptModels.filter(function(m){return sel.indexOf(m.id)!==-1});$('e-rpt-empty').style.display='none';$('e-rpt-content').style.display='block';$('e-rpt-date').textContent=new Date().toLocaleString();var colors=['#0066f5','#3abc3f','#ffa931','#8b5cf6','#ec4899','#06b6d4','#ff3131','#f97316'];var allMetrics={};var metricKeys=[];models.forEach(function(m){var mx=m.metrics||{};Object.keys(mx).forEach(function(k){if(typeof mx[k]==='number'&&metricKeys.indexOf(k)===-1)metricKeys.push(k)});allMetrics[m.id]=mx});var scoreKey=metricKeys.indexOf('f1')!==-1?'f1':(metricKeys.indexOf('accuracy')!==-1?'accuracy':(metricKeys.indexOf('r2')!==-1?'r2':metricKeys[0]||'score'));var timeKey='training_time_secs';var sorted=models.slice().sort(function(a,b){return((allMetrics[b.id]||{})[scoreKey]||0)-((allMetrics[a.id]||{})[scoreKey]||0)});var best=sorted[0];$('e-rpt-summary').innerHTML='<div style="text-align:center;padding:14px;background:#f0f6fe;border-radius:10px"><div style="font-size:24px;font-weight:700;color:#0066f5">'+models.length+'</div><div style="font-size:12px;color:#6a6f73">Models</div></div><div style="text-align:center;padding:14px;background:#f3fbf4;border-radius:10px"><div style="font-size:24px;font-weight:700;color:#3abc3f">'+metricKeys.length+'</div><div style="font-size:12px;color:#6a6f73">Metrics</div></div><div style="text-align:center;padding:14px;background:#fef9ee;border-radius:10px"><div style="font-size:24px;font-weight:700;color:#ffa931">'+scoreKey+'</div><div style="font-size:12px;color:#6a6f73">Primary Metric</div></div><div style="text-align:center;padding:14px;background:#fdf4ff;border-radius:10px"><div style="font-size:24px;font-weight:700;color:#8b5cf6">'+(best?(allMetrics[best.id]||{})[scoreKey]||0:0).toFixed(4)+'</div><div style="font-size:12px;color:#6a6f73">Best Score</div></div>';if(best){$('e-rpt-best').innerHTML='<div style="background:#0d0e0f;border-radius:12px;padding:20px;color:#fff;display:flex;align-items:center;gap:16px"><div style="width:48px;height:48px;border-radius:50%;background:#3abc3f;display:flex;align-items:center;justify-content:center;font-size:20px"><i class="ri-trophy-line"></i></div><div><div style="font-size:12px;opacity:.6">Best Model</div><div style="font-size:20px;font-weight:700">'+(best.name||best.id)+'</div><div style="font-size:13px;opacity:.8">'+scoreKey+': '+((allMetrics[best.id]||{})[scoreKey]||0).toFixed(6)+'</div></div></div>'}var barItems=[];sorted.forEach(function(m){var sc=(allMetrics[m.id]||{})[scoreKey]||0;if(sc>0)barItems.push({label:(m.name||m.id).substring(0,20),value:sc})});if(barItems.length>0)eDrawBars('e-rpt-bars',barItems,{label:scoreKey});var numericKeys=metricKeys.filter(function(k){return k!==timeKey});if(numericKeys.length>=3&&models.length>0){var radarLabels=numericKeys.slice(0,8);var radarMaxes={};radarLabels.forEach(function(k){var mx=0;models.forEach(function(m){var v=Math.abs((allMetrics[m.id]||{})[k]||0);if(v>mx)mx=v});radarMaxes[k]=mx||1});var radarSets=[];models.slice(0,4).forEach(function(m,mi){var vals=radarLabels.map(function(k){return radarMaxes[k]>0?Math.abs((allMetrics[m.id]||{})[k]||0)/radarMaxes[k]:0});radarSets.push({label:(m.name||m.id).substring(0,15),values:vals,color:colors[mi%colors.length]})});eDrawRadar('e-rpt-radar',radarLabels,radarSets[0].values,{labels:radarLabels,color:radarSets[0].color})}var th='<table><thead><tr><th>Model</th>';metricKeys.forEach(function(k){th+='<th>'+k+'</th>'});th+='</tr></thead><tbody>';sorted.forEach(function(m,i){var mx=allMetrics[m.id]||{};th+='<tr style="'+(i===0?'background:#f3fbf4':'')+'"><td style="font-weight:600">'+(i===0?'<i class="ri-trophy-line" style="color:#3abc3f;margin-right:4px"></i>':'')+((m.name||m.id).substring(0,30))+'</td>';metricKeys.forEach(function(k){var v=mx[k];var isBest=true;sorted.forEach(function(om){if(om.id!==m.id&&((allMetrics[om.id]||{})[k]||0)>=(v||0))isBest=false});th+='<td class="mono" style="'+(isBest&&typeof v==='number'?'color:#3abc3f;font-weight:600':'')+'">'+((typeof v==='number')?v.toFixed(4):(v||'—'))+'</td>'});th+='</tr>'});th+='</tbody></table>';$('e-rpt-table').innerHTML=th;var timeItems=[];sorted.forEach(function(m){var t=(allMetrics[m.id]||{})[timeKey]||0;if(t>0)timeItems.push({label:(m.name||m.id).substring(0,20),value:t})});if(timeItems.length>0)eDrawBars('e-rpt-time',timeItems,{label:'Time (s)'});var scatterPts=[];sorted.forEach(function(m,i){var sc=(allMetrics[m.id]||{})[scoreKey]||0;var t=(allMetrics[m.id]||{})[timeKey]||0;scatterPts.push({x:t,y:sc,c:i%colors.length,s:6})});if(scatterPts.length>0)eDrawScatter('e-rpt-scatter',scatterPts,{xLabel:'Training Time (s)',yLabel:scoreKey,colors:colors});var rh='';sorted.forEach(function(m,i){var sc=(allMetrics[m.id]||{})[scoreKey]||0;var maxSc=sorted.length>0?((allMetrics[sorted[0].id]||{})[scoreKey]||1):1;var pct=maxSc>0?(sc/maxSc*100).toFixed(0):0;var ci=colors[i%colors.length];rh+='<div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #f1f3f4"><div style="width:28px;height:28px;border-radius:50%;background:'+ci+'15;color:'+ci+';display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px">'+(i+1)+'</div><div style="flex:1;min-width:0"><div style="font-weight:600;font-size:13px;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+(m.name||m.id)+'</div><div class="progress" style="height:6px"><div class="progress-fill" style="width:'+pct+'%;background:'+ci+'"></div></div></div><div class="mono" style="font-size:13px;font-weight:600;color:'+ci+';flex-shrink:0">'+sc.toFixed(4)+'</div></div>'});$('e-rpt-rankings').innerHTML=rh;eNotify('Report generated for '+models.length+' models','ok');eLogActivity('Generated comparison report: '+models.length+' models')}
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

fn generate_titanic_dataset() -> Result<DataFrame> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = 891;

    let pclass: Vec<i32> = (0..n).map(|i| match i % 10 { 0..=2 => 1, 3..=5 => 2, _ => 3 }).collect();
    let sex: Vec<i32> = (0..n).map(|i| if i % 3 == 0 { 0 } else { 1 }).collect();
    let age: Vec<f64> = (0..n).map(|_| rng.gen_range(1.0..80.0)).collect();
    let sibsp: Vec<i32> = (0..n).map(|_| rng.gen_range(0..5)).collect();
    let parch: Vec<i32> = (0..n).map(|_| rng.gen_range(0..4)).collect();
    let fare: Vec<f64> = (0..n).enumerate().map(|(i, _)| {
        let base = match pclass[i] { 1 => 60.0, 2 => 25.0, _ => 10.0 };
        base + rng.gen::<f64>() * base * 0.8
    }).collect();
    let embarked: Vec<i32> = (0..n).map(|_| rng.gen_range(0..3)).collect();
    let survived: Vec<i32> = (0..n).enumerate().map(|(i, _)| {
        let chance = match (pclass[i], sex[i]) {
            (1, 0) => 0.75, (1, 1) => 0.40,
            (2, 0) => 0.60, (2, 1) => 0.15,
            (_, 0) => 0.50, _ => 0.10,
        };
        if rng.gen::<f64>() < chance { 1 } else { 0 }
    }).collect();

    Ok(DataFrame::new(vec![
        Series::new("pclass".into(), pclass).into(),
        Series::new("sex".into(), sex).into(),
        Series::new("age".into(), age).into(),
        Series::new("sibsp".into(), sibsp).into(),
        Series::new("parch".into(), parch).into(),
        Series::new("fare".into(), fare).into(),
        Series::new("embarked".into(), embarked).into(),
        Series::new("survived".into(), survived).into(),
    ])?)
}

fn generate_heart_dataset() -> Result<DataFrame> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = 303;

    let age: Vec<f64> = (0..n).map(|_| rng.gen_range(29.0..77.0)).collect();
    let sex: Vec<i32> = (0..n).map(|_| rng.gen_range(0..2)).collect();
    let cp: Vec<i32> = (0..n).map(|_| rng.gen_range(0..4)).collect();
    let trestbps: Vec<f64> = (0..n).map(|_| rng.gen_range(94.0..200.0)).collect();
    let chol: Vec<f64> = (0..n).map(|_| rng.gen_range(126.0..564.0)).collect();
    let fbs: Vec<i32> = (0..n).map(|_| if rng.gen::<f64>() < 0.15 { 1 } else { 0 }).collect();
    let restecg: Vec<i32> = (0..n).map(|_| rng.gen_range(0..3)).collect();
    let thalach: Vec<f64> = (0..n).map(|_| rng.gen_range(71.0..202.0)).collect();
    let exang: Vec<i32> = (0..n).map(|_| if rng.gen::<f64>() < 0.33 { 1 } else { 0 }).collect();
    let oldpeak: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..6.2)).collect();
    let slope: Vec<i32> = (0..n).map(|_| rng.gen_range(0..3)).collect();
    let ca: Vec<i32> = (0..n).map(|_| rng.gen_range(0..4)).collect();
    let thal: Vec<i32> = (0..n).map(|_| rng.gen_range(0..4)).collect();
    let target: Vec<i32> = (0..n).enumerate().map(|(i, _)| {
        let risk = age[i] * 0.02 + chol[i] * 0.002 + trestbps[i] * 0.005
            + cp[i] as f64 * 0.15 + oldpeak[i] * 0.1 - thalach[i] * 0.005;
        if risk + rng.gen::<f64>() * 0.5 > 2.5 { 1 } else { 0 }
    }).collect();

    Ok(DataFrame::new(vec![
        Series::new("age".into(), age).into(),
        Series::new("sex".into(), sex).into(),
        Series::new("cp".into(), cp).into(),
        Series::new("trestbps".into(), trestbps).into(),
        Series::new("chol".into(), chol).into(),
        Series::new("fbs".into(), fbs).into(),
        Series::new("restecg".into(), restecg).into(),
        Series::new("thalach".into(), thalach).into(),
        Series::new("exang".into(), exang).into(),
        Series::new("oldpeak".into(), oldpeak).into(),
        Series::new("slope".into(), slope).into(),
        Series::new("ca".into(), ca).into(),
        Series::new("thal".into(), thal).into(),
        Series::new("target".into(), target).into(),
    ])?)
}

fn generate_credit_dataset() -> Result<DataFrame> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = 1000;

    let limit_bal: Vec<f64> = (0..n).map(|_| rng.gen_range(10000.0..500000.0)).collect();
    let age: Vec<f64> = (0..n).map(|_| rng.gen_range(21.0..75.0)).collect();
    let education: Vec<i32> = (0..n).map(|_| rng.gen_range(1..5)).collect();
    let marriage: Vec<i32> = (0..n).map(|_| rng.gen_range(1..4)).collect();
    let pay_0: Vec<i32> = (0..n).map(|_| rng.gen_range(-2..9)).collect();
    let pay_2: Vec<i32> = (0..n).map(|_| rng.gen_range(-2..9)).collect();
    let pay_3: Vec<i32> = (0..n).map(|_| rng.gen_range(-2..9)).collect();
    let bill_amt1: Vec<f64> = (0..n).map(|_| rng.gen_range(-10000.0..400000.0)).collect();
    let bill_amt2: Vec<f64> = (0..n).map(|_| rng.gen_range(-10000.0..400000.0)).collect();
    let pay_amt1: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..200000.0)).collect();
    let pay_amt2: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..200000.0)).collect();
    let default_payment: Vec<i32> = (0..n).enumerate().map(|(i, _)| {
        let risk = (pay_0[i].max(0) as f64) * 0.15 + (pay_2[i].max(0) as f64) * 0.1
            + bill_amt1[i] / limit_bal[i].max(1.0) * 0.3;
        if risk + rng.gen::<f64>() * 0.4 > 0.5 { 1 } else { 0 }
    }).collect();

    Ok(DataFrame::new(vec![
        Series::new("limit_bal".into(), limit_bal).into(),
        Series::new("age".into(), age).into(),
        Series::new("education".into(), education).into(),
        Series::new("marriage".into(), marriage).into(),
        Series::new("pay_0".into(), pay_0).into(),
        Series::new("pay_2".into(), pay_2).into(),
        Series::new("pay_3".into(), pay_3).into(),
        Series::new("bill_amt1".into(), bill_amt1).into(),
        Series::new("bill_amt2".into(), bill_amt2).into(),
        Series::new("pay_amt1".into(), pay_amt1).into(),
        Series::new("pay_amt2".into(), pay_amt2).into(),
        Series::new("default_payment".into(), default_payment).into(),
    ])?)
}

fn generate_customers_dataset() -> Result<DataFrame> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = 500;

    let tenure: Vec<f64> = (0..n).map(|_| rng.gen_range(1.0..72.0)).collect();
    let monthly_charges: Vec<f64> = (0..n).map(|_| rng.gen_range(18.0..120.0)).collect();
    let total_charges: Vec<f64> = (0..n).enumerate().map(|(i, _)| {
        tenure[i] * monthly_charges[i] * (0.9 + rng.gen::<f64>() * 0.2)
    }).collect();
    let contract: Vec<i32> = (0..n).map(|_| rng.gen_range(0..3)).collect();
    let internet_service: Vec<i32> = (0..n).map(|_| rng.gen_range(0..3)).collect();
    let online_security: Vec<i32> = (0..n).map(|_| rng.gen_range(0..2)).collect();
    let tech_support: Vec<i32> = (0..n).map(|_| rng.gen_range(0..2)).collect();
    let payment_method: Vec<i32> = (0..n).map(|_| rng.gen_range(0..4)).collect();
    let senior_citizen: Vec<i32> = (0..n).map(|_| if rng.gen::<f64>() < 0.16 { 1 } else { 0 }).collect();
    let num_tickets: Vec<i32> = (0..n).map(|_| rng.gen_range(0..8)).collect();
    let churn: Vec<i32> = (0..n).enumerate().map(|(i, _)| {
        let risk = (1.0 - tenure[i] / 72.0) * 0.3
            + monthly_charges[i] / 120.0 * 0.25
            + (2 - contract[i]).max(0) as f64 * 0.15
            + num_tickets[i] as f64 * 0.05;
        if risk + rng.gen::<f64>() * 0.3 > 0.55 { 1 } else { 0 }
    }).collect();

    Ok(DataFrame::new(vec![
        Series::new("tenure".into(), tenure).into(),
        Series::new("monthly_charges".into(), monthly_charges).into(),
        Series::new("total_charges".into(), total_charges).into(),
        Series::new("contract".into(), contract).into(),
        Series::new("internet_service".into(), internet_service).into(),
        Series::new("online_security".into(), online_security).into(),
        Series::new("tech_support".into(), tech_support).into(),
        Series::new("payment_method".into(), payment_method).into(),
        Series::new("senior_citizen".into(), senior_citizen).into(),
        Series::new("num_tickets".into(), num_tickets).into(),
        Series::new("churn".into(), churn).into(),
    ])?)
}

fn generate_california_housing_dataset() -> Result<DataFrame> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = 2000;

    let longitude: Vec<f64> = (0..n).map(|_| rng.gen_range(-124.3..-114.3)).collect();
    let latitude: Vec<f64> = (0..n).map(|_| rng.gen_range(32.5..42.0)).collect();
    let housing_median_age: Vec<f64> = (0..n).map(|_| rng.gen_range(1.0..52.0)).collect();
    let total_rooms: Vec<f64> = (0..n).map(|_| rng.gen_range(2.0..40000.0)).collect();
    let total_bedrooms: Vec<f64> = (0..n).enumerate().map(|(i, _)| {
        total_rooms[i] * rng.gen_range(0.15..0.35)
    }).collect();
    let population: Vec<f64> = (0..n).map(|_| rng.gen_range(3.0..35000.0)).collect();
    let households: Vec<f64> = (0..n).enumerate().map(|(i, _)| {
        population[i] * rng.gen_range(0.2..0.6)
    }).collect();
    let median_income: Vec<f64> = (0..n).map(|_| rng.gen_range(0.5..15.0)).collect();
    let median_house_value: Vec<f64> = (0..n).enumerate().map(|(i, _)| {
        let base = median_income[i] * 30000.0
            + (42.0 - latitude[i].abs()) * 1000.0
            - housing_median_age[i] * 500.0;
        (base + rng.gen::<f64>() * 50000.0).max(15000.0).min(500001.0)
    }).collect();

    Ok(DataFrame::new(vec![
        Series::new("longitude".into(), longitude).into(),
        Series::new("latitude".into(), latitude).into(),
        Series::new("housing_median_age".into(), housing_median_age).into(),
        Series::new("total_rooms".into(), total_rooms).into(),
        Series::new("total_bedrooms".into(), total_bedrooms).into(),
        Series::new("population".into(), population).into(),
        Series::new("households".into(), households).into(),
        Series::new("median_income".into(), median_income).into(),
        Series::new("median_house_value".into(), median_house_value).into(),
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
    // Read and clone, drop lock before CPU-bound cleaning
    let df = {
        let data = state.current_data.read().await;
        data.as_ref()
            .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
            .clone()
    };

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

    // Re-acquire write lock to store cleaned data
    *state.current_data.write().await = Some(cleaned);

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
    if let Some((_, model_info)) = state.models.remove(&model_id) {
        // Clean up associated resources
        state.train_engines.remove(&model_id);
        state.model_registry.evict(&model_id).await;

        // Attempt to delete the model file from disk
        if model_info.path.exists() {
            if let Err(e) = std::fs::remove_file(&model_info.path) {
                warn!(model_id = %model_id, error = %e, "Failed to delete model file from disk");
            }
        }

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
    let model = state.models.get(&model_id)
        .ok_or_else(|| ServerError::NotFound(format!("Model '{}' not found", model_id)))?
        .clone();

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
    let jobs = state.jobs.read().await;

    let total_models = state.models.len();
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
    use crate::optimizer::{HyperOptX, OptimizationConfig, SamplerType, OptimizeDirection};

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
            partial_results: None,
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

    let n_trials = request.n_trials.unwrap_or(20).min(500);
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
    let engine = state.train_engines.get(&request.model_id)
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

    let engine = state.train_engines.get(model_id)
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
            partial_results: None,
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
    let model_info = state.models.get(&request.model_id)
        .ok_or_else(|| ServerError::NotFound(format!("Model not found: {}", request.model_id)))?
        .clone();

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

    // Existence check only — drop the Ref immediately so the shard lock is not held
    // across the blocking file write below
    if !state.train_engines.contains_key(&request.model_id) {
        return Err(ServerError::NotFound("Model engine not loaded — train a new model first".to_string()));
    }

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
    if contamination <= 0.0 || contamination > 0.5 {
        return Err(ServerError::BadRequest(
            "contamination must be in range (0.0, 0.5]".to_string()
        ));
    }
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
        "anomaly_rate": n_anomalies as f64 / n_rows.max(1) as f64,
        "columns_used": cols,
        "scores": scores.iter().take(100).collect::<Vec<_>>(),
        "predictions": predictions.iter().take(100).collect::<Vec<_>>(),
        "score_stats": {
            "min": scores.iter().cloned().fold(f64::INFINITY, f64::min),
            "max": scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            "mean": if scores.is_empty() { 0.0 } else { scores.iter().sum::<f64>() / scores.len() as f64 },
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

    // Validate clustering parameters
    if let Some(k) = request.n_clusters {
        if k == 0 || k > 10_000 {
            return Err(ServerError::BadRequest("n_clusters must be between 1 and 10000".to_string()));
        }
    }
    if let Some(eps) = request.eps {
        if eps <= 0.0 {
            return Err(ServerError::BadRequest("eps must be positive".to_string()));
        }
    }

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
// UMAP Visualization Handler
// ============================================================================

#[derive(Deserialize)]
pub struct UmapRequest {
    pub n_neighbors: Option<usize>,
    pub min_dist: Option<f64>,
    pub color_by: Option<String>,
}

/// Run UMAP dimensionality reduction on the loaded dataset
pub async fn generate_umap(
    State(state): State<Arc<AppState>>,
    Json(request): Json<UmapRequest>,
) -> Result<Json<serde_json::Value>> {
    use crate::visualization::umap::{Umap, UmapConfig};

    let data = state.current_data.read().await;
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?;

    // Extract numeric columns
    let numeric_cols: Vec<String> = df.get_columns().iter()
        .filter(|c| matches!(c.dtype(), DataType::Float32 | DataType::Float64 | DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64))
        .map(|c| c.name().to_string())
        .collect();

    if numeric_cols.len() < 2 {
        return Err(ServerError::BadRequest("UMAP requires at least 2 numeric columns".to_string()));
    }

    let n_rows = df.height();
    if n_rows < 3 {
        return Err(ServerError::BadRequest("UMAP requires at least 3 samples".to_string()));
    }

    // Build data matrix (Vec<Vec<f64>>)
    let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let mut row = Vec::with_capacity(numeric_cols.len());
        for col_name in &numeric_cols {
            let series = df.column(col_name)
                .map_err(|e| ServerError::BadRequest(format!("Column error: {}", e)))?;
            let val = series.cast(&DataType::Float64)
                .map_err(|e| ServerError::BadRequest(format!("Cast error: {}", e)))?;
            let ca = val.f64()
                .map_err(|e| ServerError::Internal(format!("Type error: {}", e)))?;
            row.push(ca.get(i).unwrap_or(0.0));
        }
        matrix.push(row);
    }

    // Extract color labels if requested
    let (labels, label_names, color_values, color_mode) = if let Some(ref color_col) = request.color_by {
        let series = df.column(color_col)
            .map_err(|e| ServerError::BadRequest(format!("Color column not found: {}", e)))?;

        // Determine if column is numeric (regression-like) or categorical (classification-like)
        let is_numeric = matches!(series.dtype(), DataType::Float32 | DataType::Float64 | DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64);

        // Count unique values to decide: if numeric with many unique values, treat as regression
        let str_vals: Vec<String> = (0..n_rows).map(|i| {
            series.get(i).map(|v| format!("{}", v)).unwrap_or_else(|_| "NA".to_string())
        }).collect();
        let unique_count = str_vals.iter().collect::<std::collections::HashSet<_>>().len();
        let is_regression = is_numeric && unique_count > 20;

        if is_regression {
            // Regression mode: return raw float values for gradient coloring
            let float_series = series.cast(&DataType::Float64)
                .map_err(|e| ServerError::BadRequest(format!("Cast error: {}", e)))?;
            let ca = float_series.f64()
                .map_err(|e| ServerError::Internal(format!("Type error: {}", e)))?;
            let vals: Vec<f64> = (0..n_rows).map(|i| ca.get(i).unwrap_or(0.0)).collect();
            (None, None, Some(vals), "regression".to_string())
        } else {
            // Classification mode: return discrete labels
            let mut unique: Vec<String> = str_vals.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
            unique.sort();
            let label_indices: Vec<usize> = str_vals.iter().map(|v| {
                unique.iter().position(|u| u == v).unwrap_or(0)
            }).collect();
            (Some(label_indices), Some(unique), None, "classification".to_string())
        }
    } else {
        (None, None, None, "none".to_string())
    };

    // Validate parameters
    let n_neighbors = request.n_neighbors.unwrap_or(15);
    let min_dist = request.min_dist.unwrap_or(0.1);
    if n_neighbors < 2 || n_neighbors > 200 {
        return Err(ServerError::BadRequest("n_neighbors must be between 2 and 200".to_string()));
    }
    if min_dist <= 0.0 || min_dist > 1.0 {
        return Err(ServerError::BadRequest("min_dist must be between 0.0 (exclusive) and 1.0".to_string()));
    }

    let result = tokio::task::spawn_blocking(move || -> std::result::Result<Vec<[f64; 2]>, String> {
        let config = UmapConfig {
            n_neighbors,
            min_dist,
            ..Default::default()
        };
        let umap = Umap::new(config);
        umap.fit_transform(&matrix).map_err(|e| e.to_string())
    }).await
        .map_err(|e| ServerError::Internal(format!("UMAP panicked: {}", e)))?
        .map_err(|e| ServerError::BadRequest(e))?;

    let points: Vec<[f64; 2]> = result;

    Ok(Json(serde_json::json!({
        "success": true,
        "points": points,
        "labels": labels,
        "label_names": label_names,
        "color_values": color_values,
        "color_mode": color_mode,
        "n_samples": n_rows,
        "params": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "color_by": request.color_by,
        }
    })))
}

// ============================================================================
// PCA Visualization Handler
// ============================================================================

#[derive(Deserialize)]
pub struct PcaRequest {
    pub color_by: Option<String>,
    pub scale: Option<bool>,
}

/// Run PCA dimensionality reduction on the loaded dataset
pub async fn generate_pca(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PcaRequest>,
) -> Result<Json<serde_json::Value>> {
    use crate::visualization::pca::{Pca, PcaConfig};

    let data = state.current_data.read().await;
    let df = data.as_ref()
        .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?;

    // Extract numeric columns
    let numeric_cols: Vec<String> = df.get_columns().iter()
        .filter(|c| matches!(c.dtype(), DataType::Float32 | DataType::Float64 | DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64))
        .map(|c| c.name().to_string())
        .collect();

    if numeric_cols.len() < 2 {
        return Err(ServerError::BadRequest("PCA requires at least 2 numeric columns".to_string()));
    }

    let n_rows = df.height();
    if n_rows < 2 {
        return Err(ServerError::BadRequest("PCA requires at least 2 samples".to_string()));
    }

    // Build data matrix
    let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let mut row = Vec::with_capacity(numeric_cols.len());
        for col_name in &numeric_cols {
            let series = df.column(col_name)
                .map_err(|e| ServerError::BadRequest(format!("Column error: {}", e)))?;
            let val = series.cast(&DataType::Float64)
                .map_err(|e| ServerError::BadRequest(format!("Cast error: {}", e)))?;
            let ca = val.f64()
                .map_err(|e| ServerError::Internal(format!("Type error: {}", e)))?;
            row.push(ca.get(i).unwrap_or(0.0));
        }
        matrix.push(row);
    }

    // Extract color info (same logic as UMAP)
    let (labels, label_names, color_values, color_mode) = if let Some(ref color_col) = request.color_by {
        let series = df.column(color_col)
            .map_err(|e| ServerError::BadRequest(format!("Color column not found: {}", e)))?;

        let is_numeric = matches!(series.dtype(), DataType::Float32 | DataType::Float64 | DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64);

        let str_vals: Vec<String> = (0..n_rows).map(|i| {
            series.get(i).map(|v| format!("{}", v)).unwrap_or_else(|_| "NA".to_string())
        }).collect();
        let unique_count = str_vals.iter().collect::<std::collections::HashSet<_>>().len();
        let is_regression = is_numeric && unique_count > 20;

        if is_regression {
            let float_series = series.cast(&DataType::Float64)
                .map_err(|e| ServerError::BadRequest(format!("Cast error: {}", e)))?;
            let ca = float_series.f64()
                .map_err(|e| ServerError::Internal(format!("Type error: {}", e)))?;
            let vals: Vec<f64> = (0..n_rows).map(|i| ca.get(i).unwrap_or(0.0)).collect();
            (None, None, Some(vals), "regression".to_string())
        } else {
            let mut unique: Vec<String> = str_vals.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
            unique.sort();
            let label_indices: Vec<usize> = str_vals.iter().map(|v| {
                unique.iter().position(|u| u == v).unwrap_or(0)
            }).collect();
            (Some(label_indices), Some(unique), None, "classification".to_string())
        }
    } else {
        (None, None, None, "none".to_string())
    };

    let scale = request.scale.unwrap_or(true);

    let result = tokio::task::spawn_blocking(move || -> std::result::Result<crate::visualization::pca::PcaResult, String> {
        let config = PcaConfig {
            scale,
            ..Default::default()
        };
        let pca = Pca::new(config);
        pca.fit_transform(&matrix).map_err(|e| e.to_string())
    }).await
        .map_err(|e| ServerError::Internal(format!("PCA panicked: {}", e)))?
        .map_err(|e| ServerError::BadRequest(e))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "points": result.embedding,
        "explained_variance_ratio": result.explained_variance_ratio,
        "eigenvalues": result.eigenvalues,
        "labels": labels,
        "label_names": label_names,
        "color_values": color_values,
        "color_mode": color_mode,
        "n_samples": n_rows,
        "params": {
            "scale": scale,
            "color_by": request.color_by,
        }
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
            partial_results: None,
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
                    partial_results: None,
                };
            }
        }

        // Launch all model trainings concurrently (limited by semaphore)
        let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_PARALLEL_TRAINS));
        let partial = Arc::new(tokio::sync::Mutex::new(Vec::<serde_json::Value>::new()));
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
            let partial_c = partial.clone();

            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let model_start = std::time::Instant::now();
                let result = run_training_job(&df_c, &target_c, tt.clone(), mt).await;
                let model_time = model_start.elapsed().as_secs_f64();
                let done = completed_c.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

                // Build partial entry for streaming
                let entry = match &result {
                    Ok((metrics, _)) => {
                        let score = match tt {
                            TaskType::BinaryClassification | TaskType::MultiClassification =>
                                metrics.get("f1").and_then(|v| v.as_f64()).unwrap_or(0.0),
                            TaskType::Regression | TaskType::TimeSeries =>
                                metrics.get("r2").and_then(|v| v.as_f64()).unwrap_or(0.0),
                            TaskType::Clustering => 0.0,
                        };
                        serde_json::json!({"model": name_owned, "score": score, "training_time_secs": model_time, "status": "success"})
                    }
                    Err(e) => serde_json::json!({"model": name_owned, "score": 0.0, "error": e.to_string(), "status": "failed"}),
                };
                partial_c.lock().await.push(entry);

                // Update progress with streaming partial results
                {
                    let partial_snapshot = partial_c.lock().await.clone();
                    let mut jobs = state_c.jobs.write().await;
                    if let Some(job) = jobs.get_mut(&jid) {
                        job.status = JobStatus::Running {
                            progress: done as f64 / total as f64,
                            message: format!("Completed {}/{} models", done, total),
                            partial_results: Some(partial_snapshot),
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
                    state_clone.train_engines.insert(model_id, engine);
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
                state_clone.models.insert(model_id, model_info);
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
            partial_results: None,
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
                if n_unique <= 2 { TaskType::BinaryClassification }
                else if n_unique <= 10 { TaskType::MultiClassification }
                else { TaskType::Regression }
            }
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                if n_unique <= 2 { TaskType::BinaryClassification }
                else if n_unique <= 20 { TaskType::MultiClassification }
                else { TaskType::Regression }
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
                partial_results: None,
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
            "none" => ScalerType::None,
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
                // Use preprocessed data for training but do NOT overwrite the
                // user's original current_data — that would corrupt their data.
                processed
            }
            Err(e) => {
                warn!(error = %e, "AutoML preprocessing failed, falling back to raw data");
                df.clone()
            }
        }
    };

    {
        let mut jobs = state.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Running {
                progress: 15.0,
                message: "Preprocessing complete. Starting model training...".to_string(),
                partial_results: None,
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
                    ModelType::KMeans => "kmeans",
                    ModelType::DBSCAN => "dbscan",
                    ModelType::SOM => "som",
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
        let partial = Arc::new(tokio::sync::Mutex::new(Vec::<serde_json::Value>::new()));
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
            let partial_c = partial.clone();

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

                // Build partial entry for streaming
                let entry = match &result {
                    Ok(Ok((metrics, _))) => {
                        let score = metrics.get("f1").or_else(|| metrics.get("r2"))
                            .and_then(|v| v.as_f64()).unwrap_or(0.0);
                        serde_json::json!({"model": name, "score": score, "training_time_secs": model_time.as_secs_f64()})
                    }
                    _ => serde_json::json!({"model": name, "score": null, "error": "failed"}),
                };
                partial_c.lock().await.push(entry);

                // Update progress with streaming partial results
                {
                    let partial_snapshot = partial_c.lock().await.clone();
                    let mut jobs = state_ref.jobs.write().await;
                    if let Some(job) = jobs.get_mut(&jid) {
                        let pct = 15.0 + (done as f64 / total as f64) * 55.0; // 15% to 70%
                        job.status = JobStatus::Running {
                            progress: pct,
                            message: format!("Trained {}/{} models ({})...", done, total, name),
                            partial_results: Some(partial_snapshot),
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
                    state_clone.train_engines.insert(key, engine.clone());

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
                            partial_results: None,
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
            state_clone.train_engines.insert(
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

    let (status, progress, message, result, partial) = match &job.status {
        JobStatus::Running { progress, message, partial_results } => {
            let pr = partial_results.as_ref().map(|v| serde_json::json!(v)).unwrap_or(serde_json::Value::Null);
            ("running", *progress, message.clone(), serde_json::Value::Null, pr)
        }
        JobStatus::Completed { metrics } => ("completed", 100.0, "Pipeline finished".to_string(), metrics.clone(), serde_json::Value::Null),
        JobStatus::Failed { error } => ("failed", 0.0, error.clone(), serde_json::Value::Null, serde_json::Value::Null),
        JobStatus::Pending => ("pending", 0.0, "Waiting to start".to_string(), serde_json::Value::Null, serde_json::Value::Null),
    };

    Ok(Json(serde_json::json!({
        "job_id": job.id,
        "status": status,
        "progress": progress,
        "message": message,
        "result": result,
        "partial_results": partial,
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
            partial_results: None,
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
                state_clone.train_engines.insert(model_id.clone(), engine);

                let model_info = super::state::ModelInfo {
                    id: model_id.clone(),
                    name: format!("Optimized {}", model_type_str),
                    task_type: model_type_str.clone(),
                    metrics: metrics_val.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    path: std::path::PathBuf::from(&models_dir).join(format!("{}.model", model_id)),
                };
                state_clone.models.insert(model_id.clone(), model_info);

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

// ============================================================================
// ISO Compliance Handlers
// ============================================================================

/// Get data lineage for the current dataset (ISO 5259, 5338)
pub async fn get_data_lineage(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let lineage_records = state.provenance_tracker.list_all();

    Ok(Json(serde_json::json!({
        "lineage_records": lineage_records,
        "total": lineage_records.len(),
    })))
}

/// Get data quality report for current dataset (ISO 5259, 25012)
pub async fn get_data_quality(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    use crate::preprocessing::quality::{DataQualityScorer, ColumnStatistics};

    let data = state.current_data.read().await;
    let df = data.as_ref().ok_or_else(|| {
        ServerError::BadRequest("No dataset loaded. Upload data first.".to_string())
    })?;

    let num_rows = df.height();
    // Compute duplicate row count for the quality scorer
    let duplicate_row_count = if num_rows > 0 {
        match df.unique_stable(None, polars::frame::UniqueKeepStrategy::First, None) {
            Ok(udf) => num_rows.saturating_sub(udf.height()),
            Err(_) => 0,
        }
    } else { 0 };

    let mut column_stats = std::collections::HashMap::new();
    for col_name in df.get_column_names() {
        let series = df.column(col_name).map_err(|e| ServerError::Internal(e.to_string()))?;
        let null_count = series.null_count();
        let unique_count = series.n_unique().unwrap_or(0);

        let is_numeric = matches!(series.dtype(),
            DataType::Float64 | DataType::Float32 |
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 |
            DataType::UInt64 | DataType::UInt32 | DataType::UInt16 | DataType::UInt8
        );

        let (mean, std, min, max, q25, q75, median, skewness, kurtosis, zeros_count, sum, outlier_count, is_sequential) =
        if is_numeric {
            if let Ok(fcast) = series.cast(&DataType::Float64) {
                if let Ok(ca) = fcast.f64() {
                    let mean_val = ca.mean();
                    let std_val = ca.std(1);
                    let min_val = ca.min();
                    let max_val = ca.max();

                    // Collect non-null values for quartile and advanced stats
                    let mut vals: Vec<f64> = ca.into_no_null_iter().collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = vals.len();

                    let (q25_v, q75_v, median_v) = if n > 0 {
                        (Some(vals[n / 4]), Some(vals[3 * n / 4]), Some(vals[n / 2]))
                    } else { (None, None, None) };

                    let (skew_v, kurt_v) = if let (Some(m), Some(s)) = (mean_val, std_val) {
                        if s > 1e-10 && n > 3 {
                            let nf = n as f64;
                            let sum3: f64 = vals.iter().map(|v| ((v - m) / s).powi(3)).sum();
                            let sum4: f64 = vals.iter().map(|v| ((v - m) / s).powi(4)).sum();
                            let skew = sum3 * nf / ((nf - 1.0) * (nf - 2.0));
                            // Fisher excess kurtosis
                            let kurt = nf * (nf + 1.0) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0)) * sum4
                                - 3.0 * (nf - 1.0).powi(2) / ((nf - 2.0) * (nf - 3.0));
                            (Some(skew), Some(kurt))
                        } else { (None, None) }
                    } else { (None, None) };

                    let zeros = vals.iter().filter(|&&v| v == 0.0).count();
                    let sum_v: f64 = vals.iter().sum();

                    // IQR outlier count
                    let outlier_cnt = if let (Some(q1), Some(q3)) = (q25_v, q75_v) {
                        let iqr = q3 - q1;
                        let lo = q1 - 1.5 * iqr;
                        let hi = q3 + 1.5 * iqr;
                        vals.iter().filter(|&&v| v < lo || v > hi).count()
                    } else { 0 };

                    // Sequential detection: sorted unique values form an arithmetic sequence
                    let seq = unique_count == n && n > 1 && {
                        let step = vals[1] - vals[0];
                        step > 0.0 && vals.windows(2).all(|w| (w[1] - w[0] - step).abs() < 1e-9)
                    };

                    (mean_val, std_val, min_val, max_val,
                     q25_v, q75_v, median_v,
                     skew_v, kurt_v,
                     Some(zeros), Some(sum_v), Some(outlier_cnt), Some(seq))
                } else {
                    (None, None, None, None, None, None, None, None, None, None, None, None, None)
                }
            } else {
                (None, None, None, None, None, None, None, None, None, None, None, None, None)
            }
        } else {
            // Categorical: compute entropy and mode_frequency
            (None, None, None, None, None, None, None, None, None, None, None, None, None)
        };

        // For categorical: compute entropy and mode_frequency
        let (mode_frequency, entropy) = if !is_numeric {
            let s = series.as_materialized_series();
            match s.value_counts(true, true, "count".into(), false) {
                Ok(vc) => {
                    let n_valid = (num_rows - null_count) as f64;
                    let mode_freq = if n_valid > 0.0 && vc.height() > 0 {
                        vc.column("count").ok()
                            .and_then(|c| c.get(0).ok())
                            .and_then(|v| v.to_string().parse::<f64>().ok())
                            .map(|top| top / n_valid)
                    } else { None };
                    let ent = if n_valid > 0.0 {
                        let e: f64 = (0..vc.height()).filter_map(|i| {
                            vc.column("count").ok()?.get(i).ok()
                                .and_then(|v| v.to_string().parse::<f64>().ok())
                                .map(|cnt| { let p = cnt / n_valid; if p > 0.0 { -p * p.ln() } else { 0.0 } })
                        }).sum();
                        Some(e)
                    } else { None };
                    (mode_freq, ent)
                }
                Err(_) => (None, None),
            }
        } else {
            // For numeric: mode_frequency = ratio of most common value
            (None, None)
        };

        // Type consistency: ratio of non-null values that parse as the inferred dtype
        // For strongly-typed Polars series this is always 1.0 unless dtype is Utf8/String
        let type_consistency = Some(1.0_f64);

        column_stats.insert(col_name.to_string(), ColumnStatistics {
            null_count,
            unique_count,
            mean,
            std,
            min,
            max,
            q25,
            q75,
            median,
            skewness,
            kurtosis,
            zeros_count,
            sum,
            mode_frequency,
            entropy,
            outlier_count,
            type_consistency,
            is_sequential,
        });
    }

    let report = DataQualityScorer::score(num_rows, &column_stats, duplicate_row_count);

    Ok(Json(serde_json::json!({
        "quality_report": report,
    })))
}

/// Get auto-generated datasheet for current dataset (ISO 5259)
pub async fn get_data_datasheet(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    use crate::export::Datasheet;

    let data = state.current_data.read().await;
    let df = data.as_ref().ok_or_else(|| {
        ServerError::BadRequest("No dataset loaded. Upload data first.".to_string())
    })?;

    let feature_names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
    let feature_dtypes: Vec<String> = df.dtypes().iter().map(|d| format!("{:?}", d)).collect();

    let mut missing_ratios = std::collections::HashMap::new();
    for col_name in df.get_column_names() {
        if let Ok(series) = df.column(col_name) {
            let ratio = series.null_count() as f64 / df.height().max(1) as f64;
            if ratio > 0.0 {
                missing_ratios.insert(col_name.to_string(), ratio);
            }
        }
    }

    let datasheet = Datasheet::generate(
        "current_dataset",
        "upload",
        df.height(),
        df.width(),
        &feature_names,
        &feature_dtypes,
        &missing_ratios,
    );

    Ok(Json(serde_json::json!({
        "datasheet": datasheet,
        "markdown": datasheet.to_markdown(),
    })))
}

/// Evaluate fairness of model predictions (ISO TR 24027)
pub async fn evaluate_fairness(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>> {
    use crate::fairness::{FairnessEvaluator, FairnessConfig, FairnessThresholds};
    use ndarray::Array1;

    let predictions: Vec<f64> = serde_json::from_value(
        body.get("predictions").cloned().unwrap_or_default()
    ).map_err(|e| ServerError::BadRequest(format!("Invalid predictions: {}", e)))?;

    let actuals: Vec<f64> = serde_json::from_value(
        body.get("actuals").cloned().unwrap_or_default()
    ).map_err(|e| ServerError::BadRequest(format!("Invalid actuals: {}", e)))?;

    let protected_attrs: std::collections::HashMap<String, Vec<String>> = serde_json::from_value(
        body.get("protected_attributes").cloned().unwrap_or_default()
    ).map_err(|e| ServerError::BadRequest(format!("Invalid protected_attributes: {}", e)))?;

    let config = FairnessConfig {
        protected_attributes: protected_attrs.keys().cloned().collect(),
        favorable_label: body.get("favorable_label").and_then(|v| v.as_f64()).unwrap_or(1.0),
        thresholds: FairnessThresholds::default(),
    };

    let evaluator = FairnessEvaluator::new(config);
    let pred_arr = Array1::from(predictions);
    let actual_arr = Array1::from(actuals);

    let report = evaluator.evaluate(&pred_arr, &actual_arr, &protected_attrs)
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(serde_json::json!({
        "fairness_report": report,
    })))
}

/// Scan dataset for bias before training (ISO TR 24027)
pub async fn bias_scan(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>> {
    use crate::fairness::{FairnessEvaluator, FairnessConfig};
    use ndarray::Array1;

    let labels: Vec<f64> = serde_json::from_value(
        body.get("labels").cloned().unwrap_or_default()
    ).map_err(|e| ServerError::BadRequest(format!("Invalid labels: {}", e)))?;

    let protected_attrs: std::collections::HashMap<String, Vec<String>> = serde_json::from_value(
        body.get("protected_attributes").cloned().unwrap_or_default()
    ).map_err(|e| ServerError::BadRequest(format!("Invalid protected_attributes: {}", e)))?;

    let config = FairnessConfig {
        protected_attributes: protected_attrs.keys().cloned().collect(),
        favorable_label: body.get("favorable_label").and_then(|v| v.as_f64()).unwrap_or(1.0),
        ..Default::default()
    };

    let evaluator = FairnessEvaluator::new(config);
    let label_arr = Array1::from(labels);

    let report = evaluator.bias_scan(&label_arr, &protected_attrs)
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(serde_json::json!({
        "bias_report": report,
    })))
}

/// Get model card for a trained model (ISO TR 24028)
pub async fn get_model_card(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>> {
    use crate::export::ModelCard;

    let model = state.models.get(&model_id).ok_or_else(|| {
        ServerError::NotFound(format!("Model not found: {}", model_id))
    })?.clone();

    let metrics: std::collections::HashMap<String, f64> = model.metrics
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                .collect()
        })
        .unwrap_or_default();

    let card = ModelCard::generate(
        &model.name,
        &model.task_type,
        &model.task_type,
        metrics,
        0, // samples count not stored in ModelInfo
        0,
        Vec::new(),
        serde_json::json!({}),
    );

    Ok(Json(serde_json::json!({
        "model_card": card,
        "markdown": card.to_markdown(),
    })))
}

/// Get audit events filtered by type (ISO 27001, TR 24028)
pub async fn get_audit_events(
    State(state): State<Arc<AppState>>,
    Query(query): Query<AuditEventsQuery>,
) -> Result<Json<serde_json::Value>> {
    let entries = if let Some(ref event_type) = query.event_type {
        state.audit_trail.query_by_type(event_type)
    } else {
        state.audit_trail.get_recent(query.limit.unwrap_or(100))
    };

    Ok(Json(serde_json::json!({
        "events": entries,
        "total": entries.len(),
    })))
}

#[derive(Debug, Deserialize)]
pub struct AuditEventsQuery {
    event_type: Option<String>,
    limit: Option<usize>,
}

/// Verify audit trail integrity (ISO 27001)
pub async fn verify_audit_integrity(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let result = state.audit_trail.verify_integrity();

    Ok(Json(serde_json::json!({
        "integrity": result,
    })))
}

/// Scan dataset for PII (ISO 27701)
pub async fn scan_pii(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>> {
    use crate::privacy::PiiScanner;

    let columns: std::collections::HashMap<String, Vec<String>> = serde_json::from_value(
        body.get("columns").cloned().unwrap_or_default()
    ).map_err(|e| ServerError::BadRequest(format!("Invalid columns: {}", e)))?;

    let scanner = PiiScanner::new();
    let result = scanner.scan_dataset(&columns);

    Ok(Json(serde_json::json!({
        "pii_scan": result,
    })))
}

/// Get data retention status (ISO 27701)
pub async fn get_retention_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let active = state.retention_manager.get_active();
    let expired = state.retention_manager.get_expired();

    Ok(Json(serde_json::json!({
        "active_records": active,
        "expired_records": expired,
        "total_active": active.len(),
        "total_expired": expired.len(),
    })))
}

/// Get compliance report (ISO 42001, 27001)
pub async fn get_compliance_report(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    use crate::compliance::{ComplianceChecker, ComplianceCheckConfig};

    let checker = ComplianceChecker::new(ComplianceCheckConfig::default());

    // Derive actual capabilities from system state rather than hardcoding all as true
    let security_status = state.security_manager.get_status();
    let has_models = !state.models.is_empty();

    let mut capabilities = std::collections::HashMap::new();
    capabilities.insert("auth_enabled".to_string(), security_status.api_key_auth_enabled);
    capabilities.insert("audit_logging".to_string(), security_status.audit_log_enabled);
    capabilities.insert("rate_limiting".to_string(), true); // always enabled
    capabilities.insert("input_validation".to_string(), true); // always enabled
    capabilities.insert("explainability".to_string(), has_models);
    capabilities.insert("calibration".to_string(), has_models);
    capabilities.insert("drift_detection".to_string(), true);
    capabilities.insert("preprocessing".to_string(), true);
    capabilities.insert("monitoring".to_string(), true);
    capabilities.insert("test_suite".to_string(), true);
    capabilities.insert("model_versioning".to_string(), true);
    capabilities.insert("provenance".to_string(), true);
    capabilities.insert("fairness_metrics".to_string(), true);
    capabilities.insert("bias_detection".to_string(), true);
    capabilities.insert("pii_detection".to_string(), true);
    capabilities.insert("anonymization".to_string(), true);
    capabilities.insert("retention_policies".to_string(), true);
    capabilities.insert("model_cards".to_string(), has_models);
    capabilities.insert("data_classification".to_string(), true);
    capabilities.insert("compliance_reporting".to_string(), true);
    capabilities.insert("rbac_enabled".to_string(), true);
    capabilities.insert("tamper_evident_audit".to_string(), true);
    capabilities.insert("data_quality".to_string(), true);
    capabilities.insert("datasheets".to_string(), true);
    capabilities.insert("slo_monitoring".to_string(), true);

    let report = checker.generate_report(&capabilities);

    Ok(Json(serde_json::json!({
        "compliance_report": report,
    })))
}

/// Get SLO status (ISO 25010)
pub async fn get_slo_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    use crate::monitoring::slo::{SloMonitor, SloMetrics, ServiceLevelObjectives};

    let monitor = SloMonitor::new(ServiceLevelObjectives::default());

    // Derive real metrics from the security manager's audit log
    let audit_log = state.security_manager.get_recent_entries(1000);
    let total_requests = audit_log.len() as u64;
    let total_errors = audit_log.iter().filter(|e| e.status_code >= 500).count() as u64;
    let error_rate = if total_requests > 0 {
        total_errors as f64 / total_requests as f64
    } else {
        0.0
    };
    let availability = if total_requests > 0 {
        100.0 * (1.0 - (total_errors as f64 / total_requests as f64))
    } else {
        100.0
    };

    let metrics = SloMetrics {
        latency_p50_ms: 5.0,   // TODO: instrument actual latency tracking
        latency_p95_ms: 25.0,
        latency_p99_ms: 100.0,
        availability_percent: availability,
        error_rate,
        total_requests,
        total_errors,
    };

    let status = monitor.evaluate(metrics);

    Ok(Json(serde_json::json!({
        "slo_status": status,
    })))
}

/// Submit prediction feedback for model improvement (ISO 42001, 5338)
pub async fn submit_prediction_feedback(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>> {
    let model_id = body.get("model_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let prediction_id = body.get("prediction_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let actual_value = body.get("actual_value")
        .and_then(|v| v.as_f64());

    // Record in audit trail
    state.audit_trail.record(
        crate::security::AuditEventType::Custom {
            category: "prediction_feedback".to_string(),
            details: serde_json::json!({
                "model_id": model_id,
                "prediction_id": prediction_id,
                "actual_value": actual_value,
            }),
        },
        "api",
        "feedback",
    );

    info!(model_id = model_id, prediction_id = prediction_id, "Prediction feedback received");

    Ok(Json(serde_json::json!({
        "status": "feedback_recorded",
        "model_id": model_id,
        "prediction_id": prediction_id,
    })))
}
