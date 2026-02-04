//! Hyperparameter Optimization Example
//!
//! Demonstrates using HyperOptX for automatic hyperparameter tuning.

use polars::prelude::*;
use kolosal_core::optimizer::{HyperOptX, HyperOptConfig};
use kolosal_core::training::{TrainEngine, TrainingConfig, TaskType, ModelType};

fn main() -> anyhow::Result<()> {
    // Create sample classification data
    let n = 200;
    let x1: Vec<f64> = (0..n).map(|i| (i as f64 / 20.0).cos() + rand_simple(i)).collect();
    let x2: Vec<f64> = (0..n).map(|i| (i as f64 / 20.0).sin() + rand_simple(i + 100)).collect();
    let y: Vec<f64> = x1.iter().zip(x2.iter())
        .map(|(a, b)| if *a + *b > 1.0 { 1.0 } else { 0.0 })
        .collect();

    let df = DataFrame::new(vec![
        Series::new("x1".into(), &x1).into(),
        Series::new("x2".into(), &x2).into(),
        Series::new("target".into(), &y).into(),
    ])?;

    println!("Dataset: {} samples for classification\n", df.height());

    // Configure hyperparameter optimization
    let opt_config = HyperOptConfig::default()
        .with_n_trials(20)
        .with_timeout_secs(60)
        .with_random_seed(42);

    // Create optimizer for Random Forest
    let mut optimizer = HyperOptX::new(opt_config);

    println!("Running hyperparameter optimization (20 trials)...\n");

    // Define the objective function
    let best_result = optimizer.optimize(|params| {
        let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
            .with_model(ModelType::RandomForest)
            .with_n_estimators(params.get("n_estimators").unwrap_or(&100.0) as usize)
            .with_max_depth(params.get("max_depth").unwrap_or(&6.0) as usize);

        let mut engine = TrainEngine::new(config);
        if let Err(_) = engine.fit(&df) {
            return 0.0;  // Return worst score on error
        }

        let metrics = engine.metrics().cloned().unwrap_or_default();
        metrics.accuracy.unwrap_or(0.0)
    });

    println!("Optimization complete!");
    println!("Best score: {:.4}", best_result.best_score);
    println!("Best parameters: {:?}", best_result.best_params);
    println!("Trials completed: {}", best_result.n_trials);

    Ok(())
}

// Simple deterministic pseudo-random for reproducibility
fn rand_simple(seed: usize) -> f64 {
    ((seed * 1103515245 + 12345) % (1 << 31)) as f64 / (1 << 31) as f64 - 0.5
}
