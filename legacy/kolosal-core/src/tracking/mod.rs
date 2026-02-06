//! Experiment Tracking Module
//!
//! Provides experiment tracking capabilities similar to MLflow.

mod tracker;
mod storage;

pub use tracker::{ExperimentTracker, ExperimentConfig, Experiment, Run, Metric};
pub use storage::{LocalStorage, StorageBackend};
