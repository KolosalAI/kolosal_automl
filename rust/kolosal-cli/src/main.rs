//! Kolosal AutoML CLI
//!
//! Command-line interface for training, prediction, and data processing.

use clap::{Parser, Subcommand};
use colored::*;
use polars::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

use kolosal_core::preprocessing::{DataPreprocessor, PreprocessingConfig, ScalerType, ImputeStrategy};
use kolosal_core::training::{TrainEngine, TrainingConfig, TaskType, ModelType};

#[derive(Parser)]
#[command(name = "kolosal")]
#[command(author = "KolosalAI")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "High-performance AutoML framework", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model on data
    Train {
        /// Input data file (CSV, JSON, or Parquet)
        #[arg(short, long)]
        data: PathBuf,

        /// Target column name
        #[arg(short, long)]
        target: String,

        /// Model type (linear, logistic, decision_tree, random_forest)
        #[arg(short, long, default_value = "random_forest")]
        model: String,

        /// Task type (classification, regression)
        #[arg(long, default_value = "classification")]
        task: String,

        /// Number of cross-validation folds
        #[arg(long, default_value = "5")]
        cv_folds: usize,

        /// Output model file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Make predictions using a trained model
    Predict {
        /// Trained model file
        #[arg(short, long)]
        model: PathBuf,

        /// Input data file
        #[arg(short, long)]
        data: PathBuf,

        /// Output predictions file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Preprocess data
    Preprocess {
        /// Input data file
        #[arg(short, long)]
        data: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Scaler type (none, standard, minmax, robust)
        #[arg(long, default_value = "standard")]
        scaler: String,

        /// Imputation strategy (drop, mean, median, mode)
        #[arg(long, default_value = "mean")]
        imputation: String,
    },

    /// Benchmark multiple models
    Benchmark {
        /// Input data file
        #[arg(short, long)]
        data: PathBuf,

        /// Target column name
        #[arg(short, long)]
        target: String,

        /// Task type (classification, regression)
        #[arg(long, default_value = "classification")]
        task: String,

        /// Number of cross-validation folds
        #[arg(long, default_value = "5")]
        cv_folds: usize,
    },

    /// Show data information
    Info {
        /// Input data file
        #[arg(short, long)]
        data: PathBuf,
    },

    /// Start the web server
    Serve {
        /// Server port
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Server host
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
    },
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "kolosal=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train { data, target, model, task, cv_folds, output } => {
            cmd_train(&data, &target, &model, &task, cv_folds, output.as_deref())?;
        }
        Commands::Predict { model, data, output } => {
            cmd_predict(&model, &data, output.as_deref())?;
        }
        Commands::Preprocess { data, output, scaler, imputation } => {
            cmd_preprocess(&data, &output, &scaler, &imputation)?;
        }
        Commands::Benchmark { data, target, task, cv_folds } => {
            cmd_benchmark(&data, &target, &task, cv_folds)?;
        }
        Commands::Info { data } => {
            cmd_info(&data)?;
        }
        Commands::Serve { port, host } => {
            cmd_serve(&host, port)?;
        }
    }

    Ok(())
}

fn load_data(path: &PathBuf) -> anyhow::Result<DataFrame> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    
    let df = match ext {
        "csv" => CsvReadOptions::default()
            .with_infer_schema_length(Some(1000))
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path.clone()))?
            .finish()?,
        "json" => JsonReader::new(std::fs::File::open(path)?)
            .finish()?,
        "parquet" => ParquetReader::new(std::fs::File::open(path)?)
            .finish()?,
        _ => anyhow::bail!("Unsupported file format: {}", ext),
    };

    Ok(df)
}

fn cmd_train(
    data_path: &PathBuf,
    target: &str,
    model_type: &str,
    task_type: &str,
    _cv_folds: usize,
    _output: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    println!("{}", "🚀 Kolosal AutoML - Training".blue().bold());
    println!();

    // Load data
    print!("Loading data... ");
    let start = Instant::now();
    let df = load_data(data_path)?;
    println!("{} ({} rows × {} cols in {:?})", "✓".green(), df.height(), df.width(), start.elapsed());

    // Parse task type
    let task = match task_type {
        "classification" => TaskType::BinaryClassification,
        "multiclass" => TaskType::MultiClassification,
        "regression" => TaskType::Regression,
        _ => anyhow::bail!("Invalid task type: {}", task_type),
    };

    // Parse model type
    let model = match model_type {
        "linear" | "linear_regression" => ModelType::LinearRegression,
        "logistic" | "logistic_regression" => ModelType::LogisticRegression,
        "decision_tree" => ModelType::DecisionTree,
        "random_forest" => ModelType::RandomForest,
        _ => anyhow::bail!("Invalid model type: {}", model_type),
    };

    // Create training config
    let config = TrainingConfig::new(task.clone(), target)
        .with_model(model);

    // Train
    print!("Training {}... ", model_type.cyan());
    let start = Instant::now();
    
    let mut engine = TrainEngine::new(config);
    engine.fit(&df)?;
    
    println!("{} ({:?})", "✓".green(), start.elapsed());

    // Print results
    println!();
    println!("{}", "📊 Results".yellow().bold());
    println!("─────────────────────────────");
    
    let metrics = engine.metrics().cloned().unwrap_or_default();

    let metric_name = match task {
        TaskType::BinaryClassification | TaskType::MultiClassification => "Accuracy",
        TaskType::Regression | TaskType::TimeSeries => "R² Score",
    };

    let score = match task {
        TaskType::BinaryClassification | TaskType::MultiClassification => metrics.accuracy.unwrap_or(0.0),
        TaskType::Regression | TaskType::TimeSeries => metrics.r2.unwrap_or(0.0),
    };

    println!("{}: {:.4}", metric_name, score);
    println!("Training time: {:.3}s", metrics.training_time_secs);
    println!();

    Ok(())
}

fn cmd_predict(
    _model_path: &PathBuf,
    _data_path: &PathBuf,
    _output: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    println!("{}", "🔮 Kolosal AutoML - Prediction".blue().bold());
    println!();
    println!("{}", "Model persistence not yet implemented".yellow());
    Ok(())
}

fn cmd_preprocess(
    data_path: &PathBuf,
    output_path: &PathBuf,
    scaler: &str,
    imputation: &str,
) -> anyhow::Result<()> {
    println!("{}", "⚙️ Kolosal AutoML - Preprocessing".blue().bold());
    println!();

    // Load data
    print!("Loading data... ");
    let df = load_data(data_path)?;
    println!("{} ({} rows × {} cols)", "✓".green(), df.height(), df.width());

    // Build config
    let scaler_type = match scaler {
        "standard" => ScalerType::Standard,
        "minmax" => ScalerType::MinMax,
        "robust" => ScalerType::Robust,
        _ => ScalerType::None,
    };

    let imputation_strategy = match imputation {
        "mean" => ImputeStrategy::Mean,
        "median" => ImputeStrategy::Median,
        "mode" => ImputeStrategy::MostFrequent,
        _ => ImputeStrategy::Drop,
    };

    let config = PreprocessingConfig::default()
        .with_scaler(scaler_type)
        .with_numeric_impute(imputation_strategy);

    // Preprocess
    print!("Preprocessing... ");
    let start = Instant::now();
    let mut preprocessor = DataPreprocessor::with_config(config);
    let processed = preprocessor.fit_transform(&df)?;
    println!("{} ({:?})", "✓".green(), start.elapsed());

    // Save
    print!("Saving to {}... ", output_path.display());
    let mut file = std::fs::File::create(output_path)?;
    CsvWriter::new(&mut file).finish(&mut processed.clone())?;
    println!("{}", "✓".green());

    println!();
    println!("Output: {} rows × {} cols", processed.height(), processed.width());

    Ok(())
}

fn cmd_benchmark(
    data_path: &PathBuf,
    target: &str,
    task_type: &str,
    _cv_folds: usize,
) -> anyhow::Result<()> {
    println!("{}", "🏆 Kolosal AutoML - Benchmark".blue().bold());
    println!();

    // Load data
    print!("Loading data... ");
    let df = load_data(data_path)?;
    println!("{} ({} rows × {} cols)", "✓".green(), df.height(), df.width());
    println!();

    // Parse task type
    let task = match task_type {
        "classification" => TaskType::BinaryClassification,
        "multiclass" => TaskType::MultiClassification,
        "regression" => TaskType::Regression,
        _ => anyhow::bail!("Invalid task type: {}", task_type),
    };

    let models: Vec<(&str, ModelType)> = match task {
        TaskType::BinaryClassification | TaskType::MultiClassification => vec![
            ("Logistic Regression", ModelType::LogisticRegression),
            ("Decision Tree", ModelType::DecisionTree),
            ("Random Forest", ModelType::RandomForest),
        ],
        TaskType::Regression | TaskType::TimeSeries => vec![
            ("Linear Regression", ModelType::LinearRegression),
            ("Decision Tree", ModelType::DecisionTree),
            ("Random Forest", ModelType::RandomForest),
        ],
    };

    let metric_name = match task {
        TaskType::BinaryClassification | TaskType::MultiClassification => "Accuracy",
        TaskType::Regression | TaskType::TimeSeries => "R² Score",
    };

    println!("{:<25} {:>12} {:>10}", "Model", metric_name, "Time");
    println!("{}", "─".repeat(50));

    let mut results: Vec<(String, f64, std::time::Duration)> = Vec::new();

    for (name, model_type) in models {
        let config = TrainingConfig::new(task.clone(), target)
            .with_model(model_type);
        
        let mut engine = TrainEngine::new(config);
        let start = Instant::now();
        
        match engine.fit(&df) {
            Ok(_) => {
                let elapsed = start.elapsed();
                let metrics = engine.metrics().cloned().unwrap_or_default();
                let score = match task {
                    TaskType::BinaryClassification | TaskType::MultiClassification => metrics.accuracy.unwrap_or(0.0),
                    TaskType::Regression | TaskType::TimeSeries => metrics.r2.unwrap_or(0.0),
                };
                
                println!("{:<25} {:>12.4} {:>10.2?}", name, score, elapsed);
                results.push((name.to_string(), score, elapsed));
            }
            Err(e) => {
                println!("{:<25} {:>12}", name, format!("Error: {}", e).red());
            }
        }
    }

    println!("{}", "─".repeat(50));
    
    // Find best model
    if let Some((name, score, _)) = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
        println!();
        println!("🥇 Best model: {} ({}: {:.4})", name.green().bold(), metric_name, score);
    }

    Ok(())
}

fn cmd_info(data_path: &PathBuf) -> anyhow::Result<()> {
    println!("{}", "📋 Kolosal AutoML - Data Info".blue().bold());
    println!();

    let df = load_data(data_path)?;

    println!("File: {}", data_path.display());
    println!("Rows: {}", df.height());
    println!("Columns: {}", df.width());
    println!("Memory: {:.2} MB", df.estimated_size() as f64 / 1024.0 / 1024.0);
    println!();

    println!("{:<20} {:<15} {:>10} {:>10}", "Column", "Type", "Nulls", "Unique");
    println!("{}", "─".repeat(60));

    for col in df.get_columns() {
        let null_count = col.null_count();
        let unique_count = col.n_unique().unwrap_or(0);
        println!(
            "{:<20} {:<15} {:>10} {:>10}",
            col.name(),
            format!("{:?}", col.dtype()),
            null_count,
            unique_count
        );
    }

    Ok(())
}

fn cmd_serve(host: &str, port: u16) -> anyhow::Result<()> {
    println!("{}", "🚀 Starting Kolosal AutoML Server...".blue().bold());
    println!();
    println!("Run with: cargo run --package kolosal-server -- --host {} --port {}", host, port);
    println!();
    println!("Or use the kolosal-server binary directly.");
    Ok(())
}
