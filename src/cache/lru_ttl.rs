//! LRU Cache with TTL Support
//!
//! A thread-safe LRU cache with time-to-live expiration.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// A cache entry with value and metadata
#[derive(Debug, Clone)]
pub struct CacheEntry<V> {
    /// The cached value
    pub value: V,
    /// When the entry was created
    pub created_at: Instant,
    /// When the entry was last accessed
    pub last_accessed: Instant,
    /// Number of times this entry was accessed
    pub access_count: u64,
}

impl<V> CacheEntry<V> {
    /// Create a new cache entry
    pub fn new(value: V) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }
    
    /// Check if this entry has expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
    
    /// Get the age of this entry
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Node in the LRU linked list
struct LruNode<K> {
    key: K,
    prev: Option<usize>,
    next: Option<usize>,
}

/// LRU Cache with TTL support
pub struct LruTtlCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Maximum cache size
    max_size: usize,
    /// Time-to-live for entries
    ttl: Duration,
    /// The cache storage
    cache: Arc<RwLock<HashMap<K, (usize, CacheEntry<V>)>>>,
    /// LRU order tracking (index -> node)
    lru_list: Arc<RwLock<Vec<LruNode<K>>>>,
    /// Head of LRU list (most recently used)
    head: Arc<RwLock<Option<usize>>>,
    /// Tail of LRU list (least recently used)
    tail: Arc<RwLock<Option<usize>>>,
    /// Free list for recycling nodes
    free_list: Arc<RwLock<Vec<usize>>>,
    /// Statistics
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
}

impl<K, V> LruTtlCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create a new LRU-TTL cache
    pub fn new(max_size: usize, ttl_seconds: u64) -> Self {
        Self {
            max_size,
            ttl: Duration::from_secs(ttl_seconds),
            cache: Arc::new(RwLock::new(HashMap::with_capacity(max_size))),
            lru_list: Arc::new(RwLock::new(Vec::with_capacity(max_size))),
            head: Arc::new(RwLock::new(None)),
            tail: Arc::new(RwLock::new(None)),
            free_list: Arc::new(RwLock::new(Vec::new())),
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Get an entry from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        // First check if key exists
        let cache = self.cache.read().ok()?;
        
        if let Some((node_idx, entry)) = cache.get(key) {
            // Check if expired
            if entry.is_expired(self.ttl) {
                drop(cache);
                self.remove(key);
                if let Ok(mut misses) = self.misses.write() {
                    *misses += 1;
                }
                return None;
            }
            
            let value = entry.value.clone();
            let node_idx = *node_idx;
            drop(cache);
            
            // Update LRU order and access count
            self.move_to_head(node_idx);
            if let Ok(mut cache) = self.cache.write() {
                if let Some((_, entry)) = cache.get_mut(key) {
                    entry.last_accessed = Instant::now();
                    entry.access_count += 1;
                }
            }
            
            if let Ok(mut hits) = self.hits.write() {
                *hits += 1;
            }
            
            Some(value)
        } else {
            drop(cache);
            if let Ok(mut misses) = self.misses.write() {
                *misses += 1;
            }
            None
        }
    }
    
    /// Set an entry in the cache
    pub fn set(&self, key: K, value: V) {
        let mut cache = match self.cache.write() {
            Ok(c) => c,
            Err(_) => return,
        };
        
        // If key already exists, update it
        if let Some((node_idx, entry)) = cache.get_mut(&key) {
            entry.value = value;
            entry.last_accessed = Instant::now();
            let idx = *node_idx;
            drop(cache);
            self.move_to_head(idx);
            return;
        }
        
        // If at capacity, evict LRU entry
        if cache.len() >= self.max_size {
            drop(cache);
            self.evict_lru();
            cache = match self.cache.write() {
                Ok(c) => c,
                Err(_) => return,
            };
        }
        
        // Add new entry
        let entry = CacheEntry::new(value);
        let node_idx = self.allocate_node(key.clone());
        cache.insert(key, (node_idx, entry));
        drop(cache);
        
        self.push_to_head(node_idx);
    }
    
    /// Remove an entry from the cache
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write().ok()?;
        
        if let Some((node_idx, entry)) = cache.remove(key) {
            drop(cache);
            self.remove_node(node_idx);
            self.free_node(node_idx);
            Some(entry.value)
        } else {
            None
        }
    }
    
    /// Check if a key exists in the cache
    pub fn contains(&self, key: &K) -> bool {
        self.cache.read()
            .map(|c| c.contains_key(key))
            .unwrap_or(false)
    }
    
    /// Get the current cache size
    pub fn len(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        if let Ok(mut list) = self.lru_list.write() {
            list.clear();
        }
        if let Ok(mut head) = self.head.write() {
            *head = None;
        }
        if let Ok(mut tail) = self.tail.write() {
            *tail = None;
        }
        if let Ok(mut free) = self.free_list.write() {
            free.clear();
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> (u64, u64, f64) {
        let hits = self.hits.read().map(|h| *h).unwrap_or(0);
        let misses = self.misses.read().map(|m| *m).unwrap_or(0);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        (hits, misses, hit_rate)
    }
    
    /// Prune expired entries
    pub fn prune_expired(&self) -> usize {
        let keys_to_remove: Vec<K> = {
            let cache = match self.cache.read() {
                Ok(c) => c,
                Err(_) => return 0,
            };
            
            cache.iter()
                .filter(|(_, (_, entry))| entry.is_expired(self.ttl))
                .map(|(k, _)| k.clone())
                .collect()
        };
        
        let count = keys_to_remove.len();
        for key in keys_to_remove {
            self.remove(&key);
        }
        count
    }
    
    // Internal methods
    
    fn allocate_node(&self, key: K) -> usize {
        // Try to reuse from free list
        if let Ok(mut free) = self.free_list.write() {
            if let Some(idx) = free.pop() {
                if let Ok(mut list) = self.lru_list.write() {
                    if idx < list.len() {
                        list[idx] = LruNode {
                            key,
                            prev: None,
                            next: None,
                        };
                        return idx;
                    }
                }
            }
        }
        
        // Allocate new node
        if let Ok(mut list) = self.lru_list.write() {
            let idx = list.len();
            list.push(LruNode {
                key,
                prev: None,
                next: None,
            });
            idx
        } else {
            0
        }
    }
    
    fn free_node(&self, idx: usize) {
        if let Ok(mut free) = self.free_list.write() {
            free.push(idx);
        }
    }
    
    fn push_to_head(&self, idx: usize) {
        let mut head = match self.head.write() {
            Ok(h) => h,
            Err(_) => return,
        };
        let mut tail = match self.tail.write() {
            Ok(t) => t,
            Err(_) => return,
        };
        let mut list = match self.lru_list.write() {
            Ok(l) => l,
            Err(_) => return,
        };
        
        if let Some(old_head) = *head {
            if old_head < list.len() {
                list[old_head].prev = Some(idx);
            }
            if idx < list.len() {
                list[idx].next = Some(old_head);
                list[idx].prev = None;
            }
        } else {
            // List was empty
            *tail = Some(idx);
            if idx < list.len() {
                list[idx].next = None;
                list[idx].prev = None;
            }
        }
        
        *head = Some(idx);
    }
    
    fn move_to_head(&self, idx: usize) {
        self.remove_node(idx);
        self.push_to_head(idx);
    }
    
    fn remove_node(&self, idx: usize) {
        let mut head = match self.head.write() {
            Ok(h) => h,
            Err(_) => return,
        };
        let mut tail = match self.tail.write() {
            Ok(t) => t,
            Err(_) => return,
        };
        let mut list = match self.lru_list.write() {
            Ok(l) => l,
            Err(_) => return,
        };
        
        if idx >= list.len() {
            return;
        }
        
        let prev = list[idx].prev;
        let next = list[idx].next;
        
        if let Some(prev_idx) = prev {
            if prev_idx < list.len() {
                list[prev_idx].next = next;
            }
        } else {
            // This was the head
            *head = next;
        }
        
        if let Some(next_idx) = next {
            if next_idx < list.len() {
                list[next_idx].prev = prev;
            }
        } else {
            // This was the tail
            *tail = prev;
        }
        
        list[idx].prev = None;
        list[idx].next = None;
    }
    
    fn evict_lru(&self) {
        let key_to_remove: Option<K> = (|| {
            let tail = self.tail.read().ok()?;
            let list = self.lru_list.read().ok()?;
            
            tail.and_then(|idx| {
                if idx < list.len() {
                    Some(list[idx].key.clone())
                } else {
                    None
                }
            })
        })();
        
        if let Some(key) = key_to_remove {
            self.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_lru_cache_basic() {
        let cache: LruTtlCache<String, i32> = LruTtlCache::new(3, 60);
        
        cache.set("a".to_string(), 1);
        cache.set("b".to_string(), 2);
        cache.set("c".to_string(), 3);
        
        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.get(&"c".to_string()), Some(3));
    }
    
    #[test]
    fn test_lru_eviction() {
        let cache: LruTtlCache<String, i32> = LruTtlCache::new(3, 60);
        
        cache.set("a".to_string(), 1);
        cache.set("b".to_string(), 2);
        cache.set("c".to_string(), 3);
        
        // Access "a" to make it recently used
        cache.get(&"a".to_string());
        
        // Add "d", should evict "b" (LRU)
        cache.set("d".to_string(), 4);
        
        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), None); // Evicted
        assert_eq!(cache.get(&"c".to_string()), Some(3));
        assert_eq!(cache.get(&"d".to_string()), Some(4));
    }
    
    #[test]
    fn test_ttl_expiration() {
        let cache: LruTtlCache<String, i32> = LruTtlCache::new(10, 1); // 1 second TTL
        
        cache.set("a".to_string(), 1);
        assert_eq!(cache.get(&"a".to_string()), Some(1));
        
        // Wait for expiration
        thread::sleep(Duration::from_secs(2));
        
        assert_eq!(cache.get(&"a".to_string()), None);
    }
    
    #[test]
    fn test_cache_stats() {
        let cache: LruTtlCache<String, i32> = LruTtlCache::new(10, 60);
        
        cache.set("a".to_string(), 1);
        cache.get(&"a".to_string()); // Hit
        cache.get(&"a".to_string()); // Hit
        cache.get(&"b".to_string()); // Miss
        
        let (hits, misses, hit_rate) = cache.stats();
        assert_eq!(hits, 2);
        assert_eq!(misses, 1);
        assert!((hit_rate - 0.666).abs() < 0.01);
    }
}
