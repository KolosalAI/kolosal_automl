//! Experiment Tracker Implementation
//!
//! Track experiments, metrics, and artifacts.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use super::storage::{LocalStorage, StorageBackend};

/// Configuration for experiment tracking
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Output directory for experiments
    pub output_dir: PathBuf,
    /// Experiment name
    pub experiment_name: String,
    /// Enable artifact logging
    pub enable_artifacts: bool,
    /// Enable metrics history
    pub enable_metrics_history: bool,
    /// Auto-save interval in seconds (0 to disable)
    pub auto_save_interval_secs: u64,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./experiments"),
            experiment_name: format!("experiment_{}", current_timestamp()),
            enable_artifacts: true,
            enable_metrics_history: true,
            auto_save_interval_secs: 60,
        }
    }
}

/// A single metric value
#[derive(Debug, Clone)]
pub struct Metric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Step/epoch number
    pub step: u64,
    /// Timestamp
    pub timestamp: u64,
}

impl Metric {
    /// Create a new metric
    pub fn new(name: impl Into<String>, value: f64, step: u64) -> Self {
        Self {
            name: name.into(),
            value,
            step,
            timestamp: current_timestamp(),
        }
    }
}

/// A run within an experiment
#[derive(Debug, Clone)]
pub struct Run {
    /// Run ID
    pub run_id: String,
    /// Run name
    pub run_name: String,
    /// Start time
    pub start_time: u64,
    /// End time (None if still running)
    pub end_time: Option<u64>,
    /// Parameters
    pub params: HashMap<String, String>,
    /// Latest metrics
    pub metrics: HashMap<String, f64>,
    /// Metrics history
    pub metrics_history: Vec<Metric>,
    /// Tags
    pub tags: HashMap<String, String>,
    /// Artifact paths
    pub artifacts: Vec<String>,
    /// Status
    pub status: RunStatus,
}

impl Run {
    /// Create a new run
    pub fn new(run_name: impl Into<String>) -> Self {
        let run_id = generate_run_id();
        Self {
            run_id,
            run_name: run_name.into(),
            start_time: current_timestamp(),
            end_time: None,
            params: HashMap::new(),
            metrics: HashMap::new(),
            metrics_history: Vec::new(),
            tags: HashMap::new(),
            artifacts: Vec::new(),
            status: RunStatus::Running,
        }
    }
    
    /// Get run duration in seconds
    pub fn duration_secs(&self) -> f64 {
        let end = self.end_time.unwrap_or_else(current_timestamp);
        (end - self.start_time) as f64
    }
}

/// Status of a run
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// Run is currently running
    Running,
    /// Run completed successfully
    Finished,
    /// Run failed
    Failed,
    /// Run was killed/stopped
    Killed,
}

/// An experiment containing multiple runs
#[derive(Debug, Clone)]
pub struct Experiment {
    /// Experiment ID
    pub experiment_id: String,
    /// Experiment name
    pub name: String,
    /// Creation time
    pub created_at: u64,
    /// Runs in this experiment
    pub runs: Vec<Run>,
    /// Tags
    pub tags: HashMap<String, String>,
}

impl Experiment {
    /// Create a new experiment
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            experiment_id: generate_experiment_id(),
            name: name.into(),
            created_at: current_timestamp(),
            runs: Vec::new(),
            tags: HashMap::new(),
        }
    }
    
    /// Get the best run by a metric (higher is better by default)
    pub fn best_run(&self, metric_name: &str, maximize: bool) -> Option<&Run> {
        self.runs.iter()
            .filter(|r| r.metrics.contains_key(metric_name))
            .max_by(|a, b| {
                let val_a = a.metrics.get(metric_name).unwrap_or(&0.0);
                let val_b = b.metrics.get(metric_name).unwrap_or(&0.0);
                
                if maximize {
                    val_a.partial_cmp(val_b).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    val_b.partial_cmp(val_a).unwrap_or(std::cmp::Ordering::Equal)
                }
            })
    }
}

/// Experiment tracker
pub struct ExperimentTracker {
    config: ExperimentConfig,
    storage: Box<dyn StorageBackend + Send + Sync>,
    
    // Current state
    current_experiment: RwLock<Option<Experiment>>,
    current_run: RwLock<Option<Run>>,
    
    // All experiments
    experiments: RwLock<HashMap<String, Experiment>>,
}

impl ExperimentTracker {
    /// Create a new experiment tracker
    pub fn new(config: ExperimentConfig) -> Self {
        let storage = Box::new(LocalStorage::new(config.output_dir.clone()));
        
        Self {
            config,
            storage,
            current_experiment: RwLock::new(None),
            current_run: RwLock::new(None),
            experiments: RwLock::new(HashMap::new()),
        }
    }
    
    /// Create with default configuration
    pub fn with_dir(output_dir: impl Into<PathBuf>) -> Self {
        let config = ExperimentConfig {
            output_dir: output_dir.into(),
            ..Default::default()
        };
        Self::new(config)
    }
    
    /// Create or get an experiment
    pub fn create_experiment(&self, name: impl Into<String>) -> String {
        let name = name.into();
        let experiment = Experiment::new(&name);
        let experiment_id = experiment.experiment_id.clone();
        
        if let Ok(mut experiments) = self.experiments.write() {
            experiments.insert(experiment_id.clone(), experiment.clone());
        }
        
        if let Ok(mut current) = self.current_experiment.write() {
            *current = Some(experiment);
        }
        
        experiment_id
    }
    
    /// Start a new run
    pub fn start_run(&self, run_name: impl Into<String>) -> String {
        let run = Run::new(run_name);
        let run_id = run.run_id.clone();
        
        if let Ok(mut current) = self.current_run.write() {
            *current = Some(run);
        }
        
        run_id
    }
    
    /// Log a parameter
    pub fn log_param(&self, key: impl Into<String>, value: impl Into<String>) {
        if let Ok(mut run) = self.current_run.write() {
            if let Some(ref mut r) = *run {
                r.params.insert(key.into(), value.into());
            }
        }
    }
    
    /// Log multiple parameters
    pub fn log_params(&self, params: HashMap<String, String>) {
        if let Ok(mut run) = self.current_run.write() {
            if let Some(ref mut r) = *run {
                r.params.extend(params);
            }
        }
    }
    
    /// Log a metric
    pub fn log_metric(&self, name: impl Into<String>, value: f64, step: Option<u64>) {
        let name = name.into();
        let step = step.unwrap_or(0);
        
        if let Ok(mut run) = self.current_run.write() {
            if let Some(ref mut r) = *run {
                r.metrics.insert(name.clone(), value);
                
                if self.config.enable_metrics_history {
                    r.metrics_history.push(Metric::new(&name, value, step));
                }
            }
        }
    }
    
    /// Log multiple metrics
    pub fn log_metrics(&self, metrics: HashMap<String, f64>, step: Option<u64>) {
        for (name, value) in metrics {
            self.log_metric(name, value, step);
        }
    }
    
    /// Log a tag
    pub fn log_tag(&self, key: impl Into<String>, value: impl Into<String>) {
        if let Ok(mut run) = self.current_run.write() {
            if let Some(ref mut r) = *run {
                r.tags.insert(key.into(), value.into());
            }
        }
    }
    
    /// Log an artifact path
    pub fn log_artifact(&self, path: impl Into<String>) {
        if let Ok(mut run) = self.current_run.write() {
            if let Some(ref mut r) = *run {
                r.artifacts.push(path.into());
            }
        }
    }
    
    /// End the current run
    pub fn end_run(&self, status: RunStatus) {
        let completed_run = {
            let mut run_guard = match self.current_run.write() {
                Ok(g) => g,
                Err(_) => return,
            };
            
            if let Some(ref mut r) = *run_guard {
                r.end_time = Some(current_timestamp());
                r.status = status;
            }
            
            run_guard.take()
        };
        
        // Add to experiment
        if let Some(run) = completed_run {
            if let Ok(mut exp) = self.current_experiment.write() {
                if let Some(ref mut e) = *exp {
                    e.runs.push(run.clone());
                    
                    // Update in experiments map
                    if let Ok(mut experiments) = self.experiments.write() {
                        if let Some(stored_exp) = experiments.get_mut(&e.experiment_id) {
                            stored_exp.runs.push(run);
                        }
                    }
                }
            }
        }
    }
    
    /// End run as finished
    pub fn end_run_success(&self) {
        self.end_run(RunStatus::Finished);
    }
    
    /// End run as failed
    pub fn end_run_failed(&self) {
        self.end_run(RunStatus::Failed);
    }
    
    /// Get the current run
    pub fn current_run(&self) -> Option<Run> {
        self.current_run.read().ok().and_then(|r| r.clone())
    }
    
    /// Get the current experiment
    pub fn current_experiment(&self) -> Option<Experiment> {
        self.current_experiment.read().ok().and_then(|e| e.clone())
    }
    
    /// Get an experiment by ID
    pub fn get_experiment(&self, experiment_id: &str) -> Option<Experiment> {
        self.experiments.read()
            .ok()
            .and_then(|e| e.get(experiment_id).cloned())
    }
    
    /// List all experiments
    pub fn list_experiments(&self) -> Vec<Experiment> {
        self.experiments.read()
            .map(|e| e.values().cloned().collect())
            .unwrap_or_default()
    }
    
    /// Save current state to storage
    pub fn save(&self) -> Result<(), String> {
        let experiments: Vec<Experiment> = self.experiments.read()
            .map(|e| e.values().cloned().collect())
            .map_err(|e| e.to_string())?;
        
        self.storage.save_experiments(&experiments)
    }
    
    /// Load state from storage
    pub fn load(&self) -> Result<(), String> {
        let experiments = self.storage.load_experiments()?;
        
        if let Ok(mut stored) = self.experiments.write() {
            for exp in experiments {
                stored.insert(exp.experiment_id.clone(), exp);
            }
        }
        
        Ok(())
    }
}

// Helper functions

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn generate_run_id() -> String {
    format!("run_{}", current_timestamp())
}

fn generate_experiment_id() -> String {
    format!("exp_{}", current_timestamp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_tracker_basic() {
        let config = ExperimentConfig {
            output_dir: PathBuf::from("/tmp/test_experiments"),
            ..Default::default()
        };
        let tracker = ExperimentTracker::new(config);
        
        // Create experiment
        let exp_id = tracker.create_experiment("test_experiment");
        assert!(!exp_id.is_empty());
        
        // Start run
        let run_id = tracker.start_run("run_1");
        assert!(!run_id.is_empty());
        
        // Log params
        tracker.log_param("learning_rate", "0.01");
        tracker.log_param("batch_size", "32");
        
        // Log metrics
        tracker.log_metric("accuracy", 0.95, Some(1));
        tracker.log_metric("loss", 0.05, Some(1));
        
        // End run
        tracker.end_run_success();
        
        // Check experiment has the run
        let exp = tracker.current_experiment().unwrap();
        assert_eq!(exp.runs.len(), 1);
        assert_eq!(exp.runs[0].status, RunStatus::Finished);
    }
    
    #[test]
    fn test_metrics_history() {
        let config = ExperimentConfig {
            enable_metrics_history: true,
            ..Default::default()
        };
        let tracker = ExperimentTracker::new(config);
        
        tracker.create_experiment("test");
        tracker.start_run("run");
        
        tracker.log_metric("loss", 1.0, Some(0));
        tracker.log_metric("loss", 0.5, Some(1));
        tracker.log_metric("loss", 0.1, Some(2));
        
        let run = tracker.current_run().unwrap();
        assert_eq!(run.metrics_history.len(), 3);
        assert_eq!(run.metrics.get("loss"), Some(&0.1)); // Latest value
    }
    
    #[test]
    fn test_best_run() {
        let mut exp = Experiment::new("test");
        
        let mut run1 = Run::new("run1");
        run1.metrics.insert("accuracy".to_string(), 0.8);
        
        let mut run2 = Run::new("run2");
        run2.metrics.insert("accuracy".to_string(), 0.95);
        
        let mut run3 = Run::new("run3");
        run3.metrics.insert("accuracy".to_string(), 0.85);
        
        exp.runs = vec![run1, run2, run3];
        
        let best = exp.best_run("accuracy", true).unwrap();
        assert_eq!(best.metrics.get("accuracy"), Some(&0.95));
    }
}
