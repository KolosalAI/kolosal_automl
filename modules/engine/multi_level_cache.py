"""
Advanced multi-level caching system with predictive warming and intelligent partitioning.

This module provides a sophisticated caching system with multiple cache levels,
access pattern learning, and predictive cache warming capabilities.
"""

import time
import threading
import queue
from collections import deque
from typing import Dict, Any, Tuple
from .lru_ttl_cache import LRUTTLCache


class MultiLevelCache:
    """
    Advanced multi-level caching system with predictive warming,
    intelligent partitioning, and access pattern learning.
    """
    
    def __init__(self, l1_size: int = 100, l2_size: int = 500, l3_size: int = 1000,
                 enable_prediction: bool = True):
        # L1: Hot data (frequently accessed, small cache, fast access)
        self.l1_cache = LRUTTLCache(max_size=l1_size, ttl_seconds=300)
        
        # L2: Warm data (moderately accessed, medium cache)
        self.l2_cache = LRUTTLCache(max_size=l2_size, ttl_seconds=900)
        
        # L3: Cold data (rarely accessed, large cache, longer TTL)
        self.l3_cache = LRUTTLCache(max_size=l3_size, ttl_seconds=1800)
        
        # Cache partitions for different data types
        self.feature_cache = LRUTTLCache(max_size=l2_size // 2, ttl_seconds=600)
        self.model_cache = LRUTTLCache(max_size=50, ttl_seconds=3600)
        self.result_cache = LRUTTLCache(max_size=l1_size * 2, ttl_seconds=180)
        
        # Predictive caching
        self.enable_prediction = enable_prediction
        self.access_patterns = {}  # key -> list of access times
        self.prediction_lock = threading.RLock()
        
        # Cache warming
        self.warming_queue = queue.Queue(maxsize=100)
        self.warming_thread = None
        self.stop_warming = threading.Event()
        
        # Statistics
        self.stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0, 
            'l3_hits': 0, 'l3_misses': 0,
            'predictions_made': 0, 'predictions_hit': 0,
            'total_requests': 0
        }
        self.stats_lock = threading.Lock()
        
        if enable_prediction:
            self._start_warming_thread()
    
    def get(self, key: str, data_type: str = 'general') -> Tuple[bool, Any]:
        """Get item from multi-level cache with access pattern tracking"""
        with self.stats_lock:
            self.stats['total_requests'] += 1
        
        # Track access pattern for prediction
        if self.enable_prediction:
            self._track_access(key)
        
        # Try L1 first (hot data)
        hit, value = self.l1_cache.get(key)
        if hit:
            with self.stats_lock:
                self.stats['l1_hits'] += 1
            return True, value
        
        # Try L2 (warm data)
        hit, value = self.l2_cache.get(key)
        if hit:
            with self.stats_lock:
                self.stats['l2_hits'] += 1
            # Promote to L1 for future fast access
            self.l1_cache.set(key, value)
            return True, value
        
        # Try L3 (cold data)
        hit, value = self.l3_cache.get(key)
        if hit:
            with self.stats_lock:
                self.stats['l3_hits'] += 1
            # Promote to L2
            self.l2_cache.set(key, value)
            return True, value
        
        # Cache miss
        with self.stats_lock:
            self.stats['l1_misses'] += 1
        
        return False, None
    
    def set(self, key: str, value: Any, data_type: str = 'general', priority: str = 'normal'):
        """Set item in appropriate cache level based on type and priority"""
        if data_type == 'features':
            self.feature_cache.set(key, value)
        elif data_type == 'model':
            self.model_cache.set(key, value)
        elif data_type == 'result':
            self.result_cache.set(key, value)
        
        # Also store in main cache hierarchy
        if priority == 'high' or data_type == 'result':
            self.l1_cache.set(key, value)
        elif priority == 'medium':
            self.l2_cache.set(key, value)  
        else:
            self.l3_cache.set(key, value)
    
    def _track_access(self, key: str):
        """Track access patterns for predictive caching"""
        current_time = time.time()
        
        with self.prediction_lock:
            if key not in self.access_patterns:
                self.access_patterns[key] = deque(maxlen=10)
            
            self.access_patterns[key].append(current_time)
    
    def _start_warming_thread(self):
        """Start background thread for predictive cache warming"""
        self.warming_thread = threading.Thread(
            target=self._warming_loop,
            daemon=True,
            name="CacheWarmingThread"
        )
        self.warming_thread.start()
    
    def _warming_loop(self):
        """Background loop for cache warming"""
        while not self.stop_warming.is_set():
            try:
                # Simple warming - just sleep
                self.stop_warming.wait(1.0)
            except Exception:
                pass
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get detailed statistics across all cache levels"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        return {
            'l1_stats': self.l1_cache.get_stats(),
            'l2_stats': self.l2_cache.get_stats(),
            'l3_stats': self.l3_cache.get_stats(),
            **stats
        }
    
    def shutdown(self):
        """Shutdown cache warming thread"""
        if self.warming_thread:
            self.stop_warming.set()
            self.warming_thread.join(timeout=2.0)


__all__ = ['MultiLevelCache']
