//! Ensemble methods module
//!
//! Provides ensemble learning algorithms including:
//! - Voting ensembles (hard and soft voting)
//! - Stacking (meta-learning)
//! - Blending (holdout-based stacking)
//! - Model averaging

mod voting;
mod stacking;
mod blending;

pub use voting::{VotingClassifier, VotingRegressor, VotingStrategy};
pub use stacking::{StackingClassifier, StackingRegressor, StackingConfig};
pub use blending::{BlendingClassifier, BlendingRegressor, BlendingConfig};

