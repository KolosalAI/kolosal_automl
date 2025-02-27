from collections import deque, OrderedDict
import time
import threading
from typing import Any, Dict, Tuple

class LRUTTLCache:
    """
    Thread-safe LRU cache with time-to-live (TTL) expiration.
    
    This implementation combines LRU eviction policy with time-based expiration.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (hit_status, value)
        """
        with self.lock:
            self._cleanup_expired()
            
            if key in self.cache:
                # Check if expired
                timestamp = self.timestamps[key]
                if time.time() - timestamp > self.ttl_seconds:
                    # Expired item
                    self.cache.pop(key)
                    self.timestamps.pop(key)
                    self.misses += 1
                    return False, None
                
                # Move to end to track as most recently used
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return True, value
            
            # Cache miss
            self.misses += 1
            return False, None
    
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
            if len(self.cache) > self.max_size:
                # Remove oldest item (first in OrderedDict)
                oldest_key, _ = self.cache.popitem(last=False)
                self.timestamps.pop(oldest_key)
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            k for k, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for k in expired_keys:
            if k in self.cache:
                self.cache.pop(k)
            if k in self.timestamps:
                self.timestamps.pop(k)
    
    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                self.timestamps.pop(key)
    
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

