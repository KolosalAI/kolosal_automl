//! Search space definition for hyperparameters

use crate::error::{KolosalError, Result};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of parameter
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterType {
    /// Continuous float parameter
    Float {
        low: f64,
        high: f64,
        log_scale: bool,
    },
    /// Integer parameter
    Int {
        low: i64,
        high: i64,
        log_scale: bool,
    },
    /// Categorical parameter
    Categorical {
        choices: Vec<String>,
    },
    /// Boolean parameter
    Boolean,
}

/// A single hyperparameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: ParameterType,
}

impl Parameter {
    /// Create a float parameter
    pub fn float(name: impl Into<String>, low: f64, high: f64) -> Self {
        Self {
            name: name.into(),
            param_type: ParameterType::Float {
                low,
                high,
                log_scale: false,
            },
        }
    }

    /// Create a log-scale float parameter
    pub fn log_float(name: impl Into<String>, low: f64, high: f64) -> Self {
        Self {
            name: name.into(),
            param_type: ParameterType::Float {
                low,
                high,
                log_scale: true,
            },
        }
    }

    /// Create an integer parameter
    pub fn int(name: impl Into<String>, low: i64, high: i64) -> Self {
        Self {
            name: name.into(),
            param_type: ParameterType::Int {
                low,
                high,
                log_scale: false,
            },
        }
    }

    /// Create a categorical parameter
    pub fn categorical(name: impl Into<String>, choices: Vec<&str>) -> Self {
        Self {
            name: name.into(),
            param_type: ParameterType::Categorical {
                choices: choices.into_iter().map(String::from).collect(),
            },
        }
    }

    /// Create a boolean parameter
    pub fn boolean(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: ParameterType::Boolean,
        }
    }

    /// Sample a random value
    pub fn sample(&self, rng: &mut impl Rng) -> ParameterValue {
        match &self.param_type {
            ParameterType::Float { low, high, log_scale } => {
                let val = if *log_scale {
                    let log_low = low.ln();
                    let log_high = high.ln();
                    (rng.gen::<f64>() * (log_high - log_low) + log_low).exp()
                } else {
                    rng.gen::<f64>() * (high - low) + low
                };
                ParameterValue::Float(val)
            }
            ParameterType::Int { low, high, log_scale } => {
                let val = if *log_scale {
                    let log_low = (*low as f64).ln();
                    let log_high = (*high as f64).ln();
                    (rng.gen::<f64>() * (log_high - log_low) + log_low).exp() as i64
                } else {
                    rng.gen_range(*low..=*high)
                };
                ParameterValue::Int(val)
            }
            ParameterType::Categorical { choices } => {
                let idx = rng.gen_range(0..choices.len());
                ParameterValue::String(choices[idx].clone())
            }
            ParameterType::Boolean => {
                ParameterValue::Bool(rng.gen())
            }
        }
    }
}

/// Sampled parameter value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
}

impl ParameterValue {
    /// Get as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(v) => Some(*v),
            ParameterValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get as int
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParameterValue::Int(v) => Some(*v),
            ParameterValue::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            ParameterValue::String(v) => Some(v),
            _ => None,
        }
    }

    /// Get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Search space for hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    parameters: Vec<Parameter>,
}

impl SearchSpace {
    /// Create a new empty search space
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
        }
    }

    /// Add a parameter to the search space
    pub fn add(mut self, param: Parameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Add a float parameter
    pub fn float(self, name: impl Into<String>, low: f64, high: f64) -> Self {
        self.add(Parameter::float(name, low, high))
    }

    /// Add a log-scale float parameter
    pub fn log_float(self, name: impl Into<String>, low: f64, high: f64) -> Self {
        self.add(Parameter::log_float(name, low, high))
    }

    /// Add an integer parameter
    pub fn int(self, name: impl Into<String>, low: i64, high: i64) -> Self {
        self.add(Parameter::int(name, low, high))
    }

    /// Add a categorical parameter
    pub fn categorical(self, name: impl Into<String>, choices: Vec<&str>) -> Self {
        self.add(Parameter::categorical(name, choices))
    }

    /// Add a boolean parameter
    pub fn boolean(self, name: impl Into<String>) -> Self {
        self.add(Parameter::boolean(name))
    }

    /// Get all parameters
    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    /// Sample a random configuration
    pub fn sample(&self, rng: &mut impl Rng) -> HashMap<String, ParameterValue> {
        self.parameters
            .iter()
            .map(|p| (p.name.clone(), p.sample(rng)))
            .collect()
    }

    /// Number of parameters
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Get parameter names in order
    pub fn param_names(&self) -> Vec<String> {
        self.parameters.iter().map(|p| p.name.clone()).collect()
    }
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Alias for sampled configuration
pub type TrialParams = HashMap<String, ParameterValue>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_space_builder() {
        let space = SearchSpace::new()
            .float("learning_rate", 0.001, 0.1)
            .int("n_estimators", 10, 1000)
            .categorical("model", vec!["rf", "gbm", "linear"])
            .boolean("early_stopping");

        assert_eq!(space.len(), 4);
    }

    #[test]
    fn test_parameter_sampling() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let param = Parameter::float("lr", 0.0, 1.0);
        let val = param.sample(&mut rng);
        
        if let ParameterValue::Float(v) = val {
            assert!(v >= 0.0 && v <= 1.0);
        } else {
            panic!("Expected float value");
        }
    }

    #[test]
    fn test_log_scale_sampling() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let param = Parameter::log_float("lr", 0.0001, 0.1);
        
        // Sample multiple times to check distribution
        let mut samples = Vec::new();
        for _ in 0..100 {
            if let ParameterValue::Float(v) = param.sample(&mut rng) {
                samples.push(v);
            }
        }

        // All samples should be in range
        assert!(samples.iter().all(|&v| v >= 0.0001 && v <= 0.1));
    }

    #[test]
    fn test_categorical_sampling() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let param = Parameter::categorical("model", vec!["a", "b", "c"]);
        let val = param.sample(&mut rng);

        if let ParameterValue::String(s) = val {
            assert!(["a", "b", "c"].contains(&s.as_str()));
        } else {
            panic!("Expected string value");
        }
    }
}
