from collections import OrderedDict
import time
import threading
from typing import Any, Dict, Tuple, Optional

class LRUTTLCache:
    """
    Thread-safe LRU cache with time-to-live (TTL) expiration.
    
    This implementation combines LRU eviction policy with time-based expiration.
    Keys are evicted when:
    1. They expire based on TTL
    2. The cache exceeds max_size (oldest used items are removed first)
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300, cleanup_interval: int = 60):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
            ttl_seconds: Time-to-live for cache items in seconds
            cleanup_interval: How often to run background cleanup in seconds (0 to disable)
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
            
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.hits = 0
        self.misses = 0
        
        # Background cleanup
        self.cleanup_thread = None
        self.should_stop = threading.Event()
        
        if cleanup_interval > 0:
            self.cleanup_thread = threading.Thread(
                target=self._background_cleanup,
                args=(cleanup_interval,),
                daemon=True  # Make thread daemon so it doesn't block program exit
            )
            self.cleanup_thread.start()
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (hit_status, value)
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return False, None
                
            # Check if expired
            timestamp = self.timestamps.get(key)
            if timestamp is None or time.time() - timestamp > self.ttl_seconds:
                # Expired item
                self.cache.pop(key, None)
                self.timestamps.pop(key, None)
                self.misses += 1
                return False, None
            
            # Move to end to track as most recently used
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return True, value
    
    def get_or_set(self, key: str, value_func: callable) -> Any:
        """
        Get item from cache or set it if missing/expired.
        
        Args:
            key: Cache key
            value_func: Function to call to generate value if not in cache
            
        Returns:
            The cached or newly generated value
        """
        hit, value = self.get(key)
        if hit:
            return value
            
        # Cache miss - generate value and store it
        value = value_func()
        self.set(key, value)
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Add or update item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # If key exists, update it
            if key in self.cache:
                self.cache.pop(key)
            
            # Add new entry
            self.cache[key] = value
            self.timestamps[key] = time.time()
            
            # Enforce size limit
            while len(self.cache) > self.max_size:
                # Remove oldest item (first in OrderedDict)
                oldest_key, _ = self.cache.popitem(last=False)
                self.timestamps.pop(oldest_key, None)
    
    def _cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of items removed
        """
        current_time = time.time()
        expired_keys = [
            k for k, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for k in expired_keys:
            self.cache.pop(k, None)
            self.timestamps.pop(k, None)
            
        return len(expired_keys)
    
    def _background_cleanup(self, interval: int) -> None:
        """Run periodic cleanup in background thread."""
        while not self.should_stop.wait(interval):
            with self.lock:
                self._cleanup_expired()
    
    def invalidate(self, key: str) -> bool:
        """
        Remove a specific key from cache.
        
        Args:
            key: Key to remove
            
        Returns:
            True if key was in cache and removed, False otherwise
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                self.timestamps.pop(key, None)
                return True
            return False
    
    def clear(self) -> None:
        """Clear the entire cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / max(1, total) * 100
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_percent": hit_rate
            }
    
    def __len__(self) -> int:
        """Return the number of items in the cache."""
        with self.lock:
            return len(self.cache)
            
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache and not expired."""
        with self.lock:
            if key not in self.cache:
                return False
                
            # Check if expired
            timestamp = self.timestamps.get(key)
            if timestamp is None or time.time() - timestamp > self.ttl_seconds:
                return False
                
            return True
    
    def __del__(self) -> None:
        """Clean up resources when instance is garbage collected."""
        if self.cleanup_thread is not None:
            self.should_stop.set()
            # We don't join the thread here as it could lead to deadlocks
            # The daemon=True flag ensures the thread doesn't block program exit
