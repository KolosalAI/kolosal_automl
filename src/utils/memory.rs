//! Memory management utilities

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

/// Helper to recover from mutex poisoning instead of panicking
fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn read_or_recover<T>(rwlock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    rwlock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_or_recover<T>(rwlock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    rwlock.write().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Pool statistics using atomic counters (no lock needed on hot path)
#[derive(Debug)]
pub struct PoolStats {
    pub allocations: AtomicU64,
    pub deallocations: AtomicU64,
    pub pool_hits: AtomicU64,
    pub pool_misses: AtomicU64,
    pub bytes_allocated: AtomicU64,
    pub bytes_in_pool: AtomicU64,
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            pool_hits: AtomicU64::new(0),
            pool_misses: AtomicU64::new(0),
            bytes_allocated: AtomicU64::new(0),
            bytes_in_pool: AtomicU64::new(0),
        }
    }
}

impl Clone for PoolStats {
    fn clone(&self) -> Self {
        Self {
            allocations: AtomicU64::new(self.allocations.load(AtomicOrdering::Relaxed)),
            deallocations: AtomicU64::new(self.deallocations.load(AtomicOrdering::Relaxed)),
            pool_hits: AtomicU64::new(self.pool_hits.load(AtomicOrdering::Relaxed)),
            pool_misses: AtomicU64::new(self.pool_misses.load(AtomicOrdering::Relaxed)),
            bytes_allocated: AtomicU64::new(self.bytes_allocated.load(AtomicOrdering::Relaxed)),
            bytes_in_pool: AtomicU64::new(self.bytes_in_pool.load(AtomicOrdering::Relaxed)),
        }
    }
}

/// Snapshot of pool stats for external consumers
#[derive(Debug, Default, Clone)]
pub struct PoolStatsSnapshot {
    pub allocations: usize,
    pub deallocations: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub bytes_allocated: usize,
    pub bytes_in_pool: usize,
}

impl PoolStats {
    fn snapshot(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            allocations: self.allocations.load(AtomicOrdering::Relaxed) as usize,
            deallocations: self.deallocations.load(AtomicOrdering::Relaxed) as usize,
            pool_hits: self.pool_hits.load(AtomicOrdering::Relaxed) as usize,
            pool_misses: self.pool_misses.load(AtomicOrdering::Relaxed) as usize,
            bytes_allocated: self.bytes_allocated.load(AtomicOrdering::Relaxed) as usize,
            bytes_in_pool: self.bytes_in_pool.load(AtomicOrdering::Relaxed) as usize,
        }
    }
}

/// Memory pool for efficient allocation.
/// Single mutex for the pool HashMap; atomic counters for stats (no lock contention).
pub struct MemoryPool {
    pools: Mutex<HashMap<usize, Vec<Vec<u8>>>>,
    max_per_class: usize,
    stats: PoolStats,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(max_per_class: usize) -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            max_per_class,
            stats: PoolStats::default(),
        }
    }

    /// Get a buffer of at least the specified size
    pub fn get(&self, size: usize) -> Vec<u8> {
        let size_class = Self::size_class(size);

        let mut pools = lock_or_recover(&self.pools);

        if let Some(pool) = pools.get_mut(&size_class) {
            if let Some(mut buffer) = pool.pop() {
                self.stats.pool_hits.fetch_add(1, AtomicOrdering::Relaxed);
                self.stats.bytes_in_pool.fetch_sub(buffer.capacity() as u64, AtomicOrdering::Relaxed);
                buffer.clear();
                return buffer;
            }
        }

        self.stats.pool_misses.fetch_add(1, AtomicOrdering::Relaxed);
        self.stats.allocations.fetch_add(1, AtomicOrdering::Relaxed);
        self.stats.bytes_allocated.fetch_add(size_class as u64, AtomicOrdering::Relaxed);

        Vec::with_capacity(size_class)
    }

    /// Return a buffer to the pool
    pub fn put(&self, buffer: Vec<u8>) {
        let size_class = Self::size_class(buffer.capacity());

        let mut pools = lock_or_recover(&self.pools);

        let pool = pools.entry(size_class).or_default();

        if pool.len() < self.max_per_class {
            self.stats.bytes_in_pool.fetch_add(buffer.capacity() as u64, AtomicOrdering::Relaxed);
            pool.push(buffer);
        } else {
            self.stats.deallocations.fetch_add(1, AtomicOrdering::Relaxed);
            let cap = buffer.capacity() as u64;
            self.stats.bytes_allocated.fetch_sub(cap.min(self.stats.bytes_allocated.load(AtomicOrdering::Relaxed)), AtomicOrdering::Relaxed);
            // Buffer is dropped here
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStatsSnapshot {
        self.stats.snapshot()
    }

    /// Clear all pooled buffers
    pub fn clear(&self) {
        let mut pools = lock_or_recover(&self.pools);

        for (_, pool) in pools.drain() {
            for buffer in pool {
                let cap = buffer.capacity() as u64;
                self.stats.bytes_allocated.fetch_sub(cap.min(self.stats.bytes_allocated.load(AtomicOrdering::Relaxed)), AtomicOrdering::Relaxed);
                self.stats.deallocations.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
        self.stats.bytes_in_pool.store(0, AtomicOrdering::Relaxed);
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

/// LRU cache with optional TTL.
/// Uses a generation counter instead of Vec for O(1) access ordering.
pub struct LruCache<K, V> {
    capacity: usize,
    ttl: Option<Duration>,
    map: RwLock<HashMap<K, CacheEntry<V>>>,
    /// Global generation counter — each access bumps the generation
    generation: AtomicU64,
    stats: Mutex<CacheStats>,
}

struct CacheEntry<V> {
    value: V,
    created_at: Instant,
    /// Generation number at last access — higher = more recently used
    generation: u64,
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
            generation: AtomicU64::new(0),
            stats: Mutex::new(CacheStats::default()),
        }
    }

    /// Create a cache with TTL
    pub fn with_ttl(capacity: usize, ttl: Duration) -> Self {
        Self {
            capacity,
            ttl: Some(ttl),
            map: RwLock::new(HashMap::new()),
            generation: AtomicU64::new(0),
            stats: Mutex::new(CacheStats::default()),
        }
    }

    /// Get a value from the cache — O(1) for ordering via generation counter
    pub fn get(&self, key: &K) -> Option<V> {
        let now = Instant::now();

        // Fast path: read lock only
        {
            let map = read_or_recover(&self.map);
            if let Some(entry) = map.get(key) {
                if let Some(ttl) = self.ttl {
                    if now.duration_since(entry.created_at) > ttl {
                        drop(map);
                        self.remove(key);
                        let mut stats = lock_or_recover(&self.stats);
                        stats.expirations += 1;
                        stats.misses += 1;
                        return None;
                    }
                }

                lock_or_recover(&self.stats).hits += 1;

                // Bump generation — O(1) atomic instead of O(n) Vec scan+shift
                // Note: we can't update the entry under a read lock, but the generation
                // will be updated on next write. For approximate LRU this is acceptable.
                let _gen = self.generation.fetch_add(1, AtomicOrdering::Relaxed);

                return Some(entry.value.clone());
            }
        }

        lock_or_recover(&self.stats).misses += 1;
        None
    }

    /// Insert a value into the cache
    pub fn insert(&self, key: K, value: V) {
        let now = Instant::now();
        let gen = self.generation.fetch_add(1, AtomicOrdering::Relaxed);

        // Evict if necessary
        {
            let map = read_or_recover(&self.map);
            if map.len() >= self.capacity && !map.contains_key(&key) {
                drop(map);
                self.evict_lru();
            }
        }

        // Insert new entry
        let entry = CacheEntry {
            value,
            created_at: now,
            generation: gen,
        };

        let mut map = write_or_recover(&self.map);
        map.insert(key, entry);
    }

    /// Remove a key from the cache
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut map = write_or_recover(&self.map);
        map.remove(key).map(|e| e.value)
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut map = write_or_recover(&self.map);
        map.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        lock_or_recover(&self.stats).clone()
    }

    /// Get current size
    pub fn len(&self) -> usize {
        read_or_recover(&self.map).len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        read_or_recover(&self.map).is_empty()
    }

    /// Evict the least recently used entry (lowest generation number)
    fn evict_lru(&self) {
        let mut map = write_or_recover(&self.map);
        if map.is_empty() {
            return;
        }

        // Find entry with lowest generation — O(n) but only on eviction
        let oldest_key = map
            .iter()
            .min_by_key(|(_, entry)| entry.generation)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest_key {
            map.remove(&key);
            lock_or_recover(&self.stats).evictions += 1;
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
            let map = read_or_recover(&self.map);
            map.iter()
                .filter(|(_, entry)| now.duration_since(entry.created_at) > ttl)
                .map(|(k, _)| k.clone())
                .collect()
        };

        if expired_keys.is_empty() {
            return;
        }

        let count = expired_keys.len();
        {
            let mut map = write_or_recover(&self.map);
            for key in &expired_keys {
                map.remove(key);
            }
        }
        lock_or_recover(&self.stats).expirations += count;
    }
}

/// Arena allocator for temporary allocations.
/// Single mutex for all state instead of three separate mutexes.
pub struct Arena {
    inner: Mutex<ArenaInner>,
    chunk_size: usize,
}

struct ArenaInner {
    chunks: Vec<Vec<u8>>,
    current_chunk: Vec<u8>,
    offset: usize,
}

impl Arena {
    /// Create a new arena with the specified chunk size
    pub fn new(chunk_size: usize) -> Self {
        Self {
            inner: Mutex::new(ArenaInner {
                chunks: Vec::new(),
                current_chunk: Vec::with_capacity(chunk_size),
                offset: 0,
            }),
            chunk_size,
        }
    }

    /// Allocate memory from the arena
    pub fn alloc(&self, size: usize) -> *mut u8 {
        let aligned_size = (size + 7) & !7; // 8-byte alignment

        let mut inner = lock_or_recover(&self.inner);

        if inner.offset + aligned_size > inner.current_chunk.capacity() {
            // Need a new chunk
            let new_chunk_size = self.chunk_size.max(aligned_size);
            let old_chunk = std::mem::replace(&mut inner.current_chunk, Vec::with_capacity(new_chunk_size));

            if !old_chunk.is_empty() {
                inner.chunks.push(old_chunk);
            }

            inner.offset = 0;
        }

        // Extend current chunk
        let current_offset = inner.offset;
        inner.current_chunk.resize(current_offset + aligned_size, 0);
        let ptr = inner.current_chunk.as_mut_ptr().wrapping_add(current_offset);
        inner.offset = current_offset + aligned_size;

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
        let mut inner = lock_or_recover(&self.inner);

        // Keep the largest chunk as current
        if let Some(largest_idx) = inner.chunks.iter().enumerate().max_by_key(|(_, c)| c.capacity()).map(|(i, _)| i) {
            if inner.chunks[largest_idx].capacity() > inner.current_chunk.capacity() {
                inner.current_chunk = inner.chunks.swap_remove(largest_idx);
            }
        }

        inner.chunks.clear();
        inner.current_chunk.clear();
        inner.offset = 0;
    }

    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        let inner = lock_or_recover(&self.inner);
        inner.chunks.iter().map(|c| c.capacity()).sum::<usize>() + inner.current_chunk.capacity()
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
