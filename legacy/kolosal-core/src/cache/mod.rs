//! Cache Module
//!
//! Provides high-performance caching with LRU eviction, TTL support,
//! and multi-level caching with predictive warming.

mod lru_ttl;
mod multi_level;

pub use lru_ttl::{LruTtlCache, CacheEntry};
pub use multi_level::{MultiLevelCache, CacheLevel, CacheStats};
