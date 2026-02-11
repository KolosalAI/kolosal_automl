//! Model explainability module
//!
//! Provides model interpretation and explanation methods including:
//! - Permutation feature importance
//! - Partial dependence plots (PDP)
//! - Individual conditional expectation (ICE)
//! - SHAP-like local explanations
//! - Feature contribution analysis

mod importance;
mod pdp;
mod local_explanations;

pub use importance::{PermutationImportance, ImportanceResult};
pub use pdp::{PartialDependence, PDPResult, ICEResult};
pub use local_explanations::{LocalExplainer, LocalExplanation, FeatureContribution};

