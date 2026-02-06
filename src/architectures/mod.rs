//! Advanced Neural Network Architectures for Tabular Data
//!
//! This module provides implementations of state-of-the-art neural network
//! architectures designed specifically for tabular data:
//! - TabNet: Attention-based feature selection
//! - FT-Transformer: Feature Tokenizer + Transformer

mod tabnet;
mod ft_transformer;
mod layers;

pub use tabnet::{TabNet, TabNetConfig, TabNetEncoder, TabNetDecoder};
pub use ft_transformer::{FTTransformer, FTTransformerConfig, FeatureTokenizer};
pub use layers::{
    GhostBatchNorm, Sparsemax, GLUBlock, 
    AttentionTransformer, FeatureAttention,
};
