//! Parallel processing utilities

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for parallel processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of threads (None = use all available)
    pub n_threads: Option<usize>,
    /// Chunk size for parallel iteration
    pub chunk_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            n_threads: None,
            chunk_size: 1000,
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of threads
    pub fn with_threads(mut self, n: usize) -> Self {
        self.n_threads = Some(n);
        self
    }

    /// Set chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Get the number of threads to use
    pub fn num_threads(&self) -> usize {
        self.n_threads.unwrap_or_else(rayon::current_num_threads)
    }
}

/// Parallel map operation
pub fn parallel_map<T, U, F>(items: Vec<T>, f: F) -> Vec<U>
where
    T: Send + Sync,
    U: Send,
    F: Fn(T) -> U + Send + Sync,
{
    items.into_par_iter().map(f).collect()
}

/// Parallel map with configuration
pub fn parallel_map_with_config<T, U, F>(items: Vec<T>, config: &ParallelConfig, f: F) -> Vec<U>
where
    T: Send + Sync,
    U: Send,
    F: Fn(T) -> U + Send + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.num_threads())
        .build()
        .unwrap();

    pool.install(|| items.into_par_iter().map(f).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_map() {
        let items: Vec<i32> = (0..1000).collect();
        let results = parallel_map(items, |x| x * 2);
        
        assert_eq!(results.len(), 1000);
        assert_eq!(results[0], 0);
        assert_eq!(results[500], 1000);
    }

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::new()
            .with_threads(4)
            .with_chunk_size(500);
        
        assert_eq!(config.n_threads, Some(4));
        assert_eq!(config.chunk_size, 500);
    }
}
