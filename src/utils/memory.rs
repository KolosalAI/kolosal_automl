//! Memory management utilities

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

/// Memory pool for efficient allocation
pub struct MemoryPool {
    /// Pool of available buffers by size class
    pools: Mutex<HashMap<usize, Vec<Vec<u8>>>>,
    /// Maximum buffers per size class
    max_per_class: usize,
    /// Statistics
    stats: Mutex<PoolStats>,
}

#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    pub allocations: usize,
    pub deallocations: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub bytes_allocated: usize,
    pub bytes_in_pool: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(max_per_class: usize) -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            max_per_class,
            stats: Mutex::new(PoolStats::default()),
        }
    }

    /// Get a buffer of at least the specified size
    pub fn get(&self, size: usize) -> Vec<u8> {
        let size_class = Self::size_class(size);
        
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(pool) = pools.get_mut(&size_class) {
            if let Some(mut buffer) = pool.pop() {
                stats.pool_hits += 1;
                stats.bytes_in_pool -= buffer.capacity();
                buffer.clear();
                return buffer;
            }
        }

        stats.pool_misses += 1;
        stats.allocations += 1;
        stats.bytes_allocated += size_class;

        Vec::with_capacity(size_class)
    }

    /// Return a buffer to the pool
    pub fn put(&self, buffer: Vec<u8>) {
        let size_class = Self::size_class(buffer.capacity());
        
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let pool = pools.entry(size_class).or_insert_with(Vec::new);
        
        if pool.len() < self.max_per_class {
            stats.bytes_in_pool += buffer.capacity();
            pool.push(buffer);
        } else {
            stats.deallocations += 1;
            stats.bytes_allocated = stats.bytes_allocated.saturating_sub(buffer.capacity());
            // Buffer is dropped here
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all pooled buffers
    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        for (_, pool) in pools.drain() {
            for buffer in pool {
                stats.bytes_allocated = stats.bytes_allocated.saturating_sub(buffer.capacity());
                stats.deallocations += 1;
            }
        }
        stats.bytes_in_pool = 0;
    }

    /// Round up to nearest size class (power of 2)
    fn size_class(size: usize) -> usize {
        let min_size = 64;
        if size <= min_size {
            return min_size;
        }
        size.next_power_of_two()
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new(16)
    }
}

/// LRU cache with optional TTL
pub struct LruCache<K, V> {
    capacity: usize,
    ttl: Option<Duration>,
    map: RwLock<HashMap<K, CacheEntry<V>>>,
    access_order: Mutex<Vec<K>>,
    stats: Mutex<CacheStats>,
}

struct CacheEntry<V> {
    value: V,
    created_at: Instant,
    last_accessed: Instant,
}

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub expirations: usize,
}

impl<K: Clone + Eq + std::hash::Hash, V: Clone> LruCache<K, V> {
    /// Create a new LRU cache
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ttl: None,
            map: RwLock::new(HashMap::new()),
            access_order: Mutex::new(Vec::new()),
            stats: Mutex::new(CacheStats::default()),
        }
    }

    /// Create a cache with TTL
    pub fn with_ttl(capacity: usize, ttl: Duration) -> Self {
        Self {
            capacity,
            ttl: Some(ttl),
            map: RwLock::new(HashMap::new()),
            access_order: Mutex::new(Vec::new()),
            stats: Mutex::new(CacheStats::default()),
        }
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        let now = Instant::now();
        
        // Check if key exists and is not expired
        {
            let map = self.map.read().unwrap();
            if let Some(entry) = map.get(key) {
                if let Some(ttl) = self.ttl {
                    if now.duration_since(entry.created_at) > ttl {
                        drop(map);
                        self.remove(key);
                        self.stats.lock().unwrap().expirations += 1;
                        self.stats.lock().unwrap().misses += 1;
                        return None;
                    }
                }
                
                self.stats.lock().unwrap().hits += 1;
                
                // Update access order
                let mut order = self.access_order.lock().unwrap();
                if let Some(pos) = order.iter().position(|k| k == key) {
                    let k = order.remove(pos);
                    order.push(k);
                }
                
                return Some(entry.value.clone());
            }
        }

        self.stats.lock().unwrap().misses += 1;
        None
    }

    /// Insert a value into the cache
    pub fn insert(&self, key: K, value: V) {
        let now = Instant::now();

        // Evict if necessary
        {
            let map = self.map.read().unwrap();
            if map.len() >= self.capacity && !map.contains_key(&key) {
                drop(map);
                self.evict_lru();
            }
        }

        // Insert new entry
        let entry = CacheEntry {
            value,
            created_at: now,
            last_accessed: now,
        };

        {
            let mut map = self.map.write().unwrap();
            let is_new = !map.contains_key(&key);
            map.insert(key.clone(), entry);

            if is_new {
                let mut order = self.access_order.lock().unwrap();
                order.push(key);
            }
        }
    }

    /// Remove a key from the cache
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut map = self.map.write().unwrap();
        let mut order = self.access_order.lock().unwrap();

        if let Some(pos) = order.iter().position(|k| k == key) {
            order.remove(pos);
        }

        map.remove(key).map(|e| e.value)
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut map = self.map.write().unwrap();
        let mut order = self.access_order.lock().unwrap();
        map.clear();
        order.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.map.read().unwrap().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.map.read().unwrap().is_empty()
    }

    fn evict_lru(&self) {
        let order = self.access_order.lock().unwrap();
        if let Some(oldest_key) = order.first().cloned() {
            drop(order);
            self.remove(&oldest_key);
            self.stats.lock().unwrap().evictions += 1;
        }
    }

    /// Remove expired entries
    pub fn cleanup_expired(&self) {
        if self.ttl.is_none() {
            return;
        }

        let ttl = self.ttl.unwrap();
        let now = Instant::now();

        let expired_keys: Vec<K> = {
            let map = self.map.read().unwrap();
            map.iter()
                .filter(|(_, entry)| now.duration_since(entry.created_at) > ttl)
                .map(|(k, _)| k.clone())
                .collect()
        };

        let mut stats = self.stats.lock().unwrap();
        for key in expired_keys {
            self.remove(&key);
            stats.expirations += 1;
        }
    }
}

/// Arena allocator for temporary allocations
pub struct Arena {
    chunks: Mutex<Vec<Vec<u8>>>,
    current_chunk: Mutex<Vec<u8>>,
    chunk_size: usize,
    offset: Mutex<usize>,
}

impl Arena {
    /// Create a new arena with the specified chunk size
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: Mutex::new(Vec::new()),
            current_chunk: Mutex::new(Vec::with_capacity(chunk_size)),
            chunk_size,
            offset: Mutex::new(0),
        }
    }

    /// Allocate memory from the arena
    pub fn alloc(&self, size: usize) -> *mut u8 {
        let aligned_size = (size + 7) & !7; // 8-byte alignment
        
        let mut offset = self.offset.lock().unwrap();
        let mut current = self.current_chunk.lock().unwrap();

        if *offset + aligned_size > current.capacity() {
            // Need a new chunk
            let new_chunk_size = self.chunk_size.max(aligned_size);
            let old_chunk = std::mem::replace(&mut *current, Vec::with_capacity(new_chunk_size));
            
            if !old_chunk.is_empty() {
                self.chunks.lock().unwrap().push(old_chunk);
            }
            
            *offset = 0;
        }

        // Extend current chunk
        current.resize(*offset + aligned_size, 0);
        let ptr = current.as_mut_ptr().wrapping_add(*offset);
        *offset += aligned_size;

        ptr
    }

    /// Allocate and initialize a value
    pub fn alloc_val<T: Copy>(&self, value: T) -> &mut T {
        let ptr = self.alloc(std::mem::size_of::<T>()) as *mut T;
        unsafe {
            std::ptr::write(ptr, value);
            &mut *ptr
        }
    }

    /// Allocate a slice
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> &mut [T] {
        let size = std::mem::size_of::<T>() * values.len();
        let ptr = self.alloc(size) as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
            std::slice::from_raw_parts_mut(ptr, values.len())
        }
    }

    /// Reset the arena (reuse memory)
    pub fn reset(&self) {
        let mut chunks = self.chunks.lock().unwrap();
        let mut current = self.current_chunk.lock().unwrap();
        let mut offset = self.offset.lock().unwrap();

        // Keep the largest chunk as current
        if let Some(largest) = chunks.iter().max_by_key(|c| c.capacity()) {
            let largest_cap = largest.capacity();
            if largest_cap > current.capacity() {
                let idx = chunks.iter().position(|c| c.capacity() == largest_cap).unwrap();
                *current = chunks.swap_remove(idx);
            }
        }

        chunks.clear();
        current.clear();
        *offset = 0;
    }

    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        let chunks = self.chunks.lock().unwrap();
        let current = self.current_chunk.lock().unwrap();
        
        chunks.iter().map(|c| c.capacity()).sum::<usize>() + current.capacity()
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new(64 * 1024) // 64KB default chunk size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(4);
        
        let buf1 = pool.get(100);
        assert!(buf1.capacity() >= 100);
        
        pool.put(buf1);
        
        let buf2 = pool.get(100);
        assert!(buf2.capacity() >= 100);
        
        let stats = pool.stats();
        assert!(stats.pool_hits > 0 || stats.pool_misses > 0);
    }

    #[test]
    fn test_lru_cache() {
        let cache: LruCache<String, i32> = LruCache::new(3);
        
        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);
        
        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        
        // Insert 4th item - should evict LRU (c, since a and b were accessed)
        cache.insert("d".to_string(), 4);
        
        assert_eq!(cache.get(&"d".to_string()), Some(4));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_lru_cache_ttl() {
        let cache: LruCache<String, i32> = LruCache::with_ttl(10, Duration::from_millis(50));
        
        cache.insert("a".to_string(), 1);
        assert_eq!(cache.get(&"a".to_string()), Some(1));
        
        std::thread::sleep(Duration::from_millis(100));
        
        assert_eq!(cache.get(&"a".to_string()), None);
    }

    #[test]
    fn test_arena() {
        let arena = Arena::new(1024);
        
        let val1 = arena.alloc_val(42i32);
        assert_eq!(*val1, 42);
        
        let slice = arena.alloc_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
        
        let bytes = arena.allocated_bytes();
        assert!(bytes > 0);
        
        arena.reset();
    }
}
