//! Training configuration

use serde::{Deserialize, Serialize};

/// Type of ML task
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    /// Binary classification
    BinaryClassification,
    /// Multi-class classification
    MultiClassification,
    /// Regression
    Regression,
    /// Time series forecasting
    TimeSeries,
}

/// Type of model to train
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelType {
    /// Decision Tree
    DecisionTree,
    /// Random Forest
    RandomForest,
    /// Gradient Boosted Trees (LightGBM-style)
    GradientBoosting,
    /// XGBoost
    XGBoost,
    /// Linear Regression
    LinearRegression,
    /// Logistic Regression
    LogisticRegression,
    /// Ridge Regression
    Ridge,
    /// Lasso Regression
    Lasso,
    /// Elastic Net
    ElasticNet,
    /// Support Vector Machine
    SVM,
    /// K-Nearest Neighbors
    KNN,
    /// Neural Network (MLP)
    NeuralNetwork,
    /// Auto (automatically select best model)
    Auto,
}

/// Configuration for model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Type of ML task
    pub task_type: TaskType,
    
    /// Model type to train
    pub model_type: ModelType,
    
    /// Target column name
    pub target_column: String,
    
    /// Feature column names (None = all except target)
    pub feature_columns: Option<Vec<String>>,
    
    /// Train/validation split ratio
    pub validation_split: f64,
    
    /// Number of cross-validation folds (0 = no CV)
    pub cv_folds: usize,
    
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    
    /// Whether to use early stopping
    pub early_stopping: bool,
    
    /// Number of rounds without improvement before stopping
    pub early_stopping_rounds: usize,
    
    /// Metric to optimize
    pub metric: String,
    
    /// Number of parallel jobs
    pub n_jobs: Option<usize>,
    
    /// Verbosity level
    pub verbose: bool,
    
    // Tree-specific parameters
    /// Maximum depth of trees
    pub max_depth: Option<usize>,
    
    /// Minimum samples per leaf
    pub min_samples_leaf: usize,
    
    /// Number of trees (for ensemble methods)
    pub n_estimators: usize,
    
    /// Learning rate (for boosting)
    pub learning_rate: f64,
    
    /// Subsample ratio
    pub subsample: f64,
    
    /// Column sample ratio
    pub colsample_bytree: f64,
    
    // Regularization
    /// L1 regularization
    pub reg_alpha: f64,
    
    /// L2 regularization
    pub reg_lambda: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            task_type: TaskType::Regression,
            model_type: ModelType::Auto,
            target_column: "target".to_string(),
            feature_columns: None,
            validation_split: 0.2,
            cv_folds: 5,
            random_state: Some(42),
            early_stopping: true,
            early_stopping_rounds: 50,
            metric: "auto".to_string(),
            n_jobs: None,
            verbose: false,
            max_depth: Some(6),
            min_samples_leaf: 1,
            n_estimators: 100,
            learning_rate: 0.1,
            subsample: 1.0,
            colsample_bytree: 1.0,
            reg_alpha: 0.0,
            reg_lambda: 1.0,
        }
    }
}

impl TrainingConfig {
    /// Create a new configuration
    pub fn new(task_type: TaskType, target: impl Into<String>) -> Self {
        Self {
            task_type,
            target_column: target.into(),
            ..Default::default()
        }
    }

    /// Builder method to set model type
    pub fn with_model(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Builder method to set number of estimators
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Builder method to set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder method to set max depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Builder method to set CV folds
    pub fn with_cv(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }

    /// Builder method to set random state
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.validation_split, 0.2);
        assert_eq!(config.cv_folds, 5);
    }

    #[test]
    fn test_builder_pattern() {
        let config = TrainingConfig::new(TaskType::BinaryClassification, "label")
            .with_model(ModelType::RandomForest)
            .with_n_estimators(200)
            .with_max_depth(10);
        
        assert!(matches!(config.task_type, TaskType::BinaryClassification));
        assert!(matches!(config.model_type, ModelType::RandomForest));
        assert_eq!(config.n_estimators, 200);
        assert_eq!(config.max_depth, Some(10));
    }
}
