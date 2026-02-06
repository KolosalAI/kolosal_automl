//! Kolosal AutoML - Main Entry Point
//!
//! A high-performance AutoML framework in Rust with CLI and server modes.

use clap::Parser;
use kolosal_automl::cli::{Cli, Commands, cmd_train, cmd_predict, cmd_preprocess, cmd_benchmark, cmd_info, cmd_serve, cmd_interactive};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "kolosal=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Train { data, target, model, task, cv_folds, output }) => {
            cmd_train(&data, &target, &model, &task, cv_folds, output.as_deref())?;
        }
        Some(Commands::Predict { model, data, output }) => {
            cmd_predict(&model, &data, output.as_deref())?;
        }
        Some(Commands::Preprocess { data, output, scaler, imputation }) => {
            cmd_preprocess(&data, &output, &scaler, &imputation)?;
        }
        Some(Commands::Benchmark { data, target, task, cv_folds }) => {
            cmd_benchmark(&data, &target, &task, cv_folds)?;
        }
        Some(Commands::Info { data }) => {
            cmd_info(&data)?;
        }
        Some(Commands::Serve { port, host }) => {
            cmd_serve(&host, port).await?;
        }
        None => {
            // Default: interactive mode (matches Python main.py behavior)
            cmd_interactive().await?;
        }
    }

    Ok(())
}
