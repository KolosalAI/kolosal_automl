//! Visualization module — dimensionality reduction for plotting.

pub mod pca;
pub mod umap;
pub use pca::{Pca, PcaConfig, PcaResult};
pub use umap::{Umap, UmapConfig};
