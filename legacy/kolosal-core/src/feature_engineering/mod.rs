//! Advanced Feature Engineering module
//!
//! Provides automated feature generation including:
//! - Polynomial features
//! - Feature interactions
//! - Arithmetic combinations
//! - Text feature extraction

mod polynomial;
mod interactions;
mod text_features;

pub use polynomial::{PolynomialFeatures, PolynomialConfig};
pub use interactions::{FeatureInteractions, InteractionType, FeatureCrossing, AutoCrossing};
pub use text_features::{TfidfVectorizer, CountVectorizer, TextTokenizer, HashingVectorizer};

use crate::error::Result;
use ndarray::Array2;

/// Trait for feature transformers
pub trait FeatureTransformer: Send + Sync {
    /// Fit the transformer
    fn fit(&mut self, x: &Array2<f64>) -> Result<()>;
    
    /// Transform data
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    
    /// Fit and transform in one step
    fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }
    
    /// Get output feature names
    fn get_feature_names(&self) -> Vec<String>;
}
