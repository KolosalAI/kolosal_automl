"""
Batch processing statistics tracking with thread-safety and performance optimizations.

This module provides comprehensive statistics collection for batch processing operations
with intelligent caching and efficient metric calculation.
"""

import time
import statistics
import numpy as np
from threading import RLock
from collections import deque
from typing import Dict


class BatchStats:
    """Tracks batch processing statistics with thread-safety and performance optimizations."""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.queue_times = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.queue_lengths = deque(maxlen=window_size)
        self.error_counts = 0
        self.batch_counts = 0
        self.total_processed = 0
        self.lock = RLock()  # Use RLock for nested lock acquisitions
        
        # Cache for expensive statistical calculations
        self._stats_cache = {}
        self._last_update_time = 0
        self._cache_valid_time = 1.0  # Cache valid for 1 second

    def update(self, processing_time: float, batch_size: int, queue_time: float, 
               latency: float = 0.0, queue_length: int = 0) -> None:
        """Update statistics with batch processing information."""
        with self.lock:
            self.processing_times.append(processing_time)
            self.batch_sizes.append(batch_size)
            self.queue_times.append(queue_time)
            self.latencies.append(latency)
            self.queue_lengths.append(queue_length)
            self.total_processed += batch_size
            self.batch_counts += 1
            
            # Invalidate cache
            self._last_update_time = time.monotonic()
            self._stats_cache.clear()

    def record_error(self, count: int = 1) -> None:
        """Record processing errors."""
        with self.lock:
            self.error_counts += count
            # Invalidate cache
            self._stats_cache.clear()

    def get_stats(self) -> Dict[str, float]:
        """Get processing statistics with caching for performance."""
        current_time = time.monotonic()
        
        with self.lock:
            # Return cached stats if valid
            if self._stats_cache and (current_time - self._last_update_time) < self._cache_valid_time:
                return self._stats_cache.copy()
            
            # Calculate new stats
            stats = {}
            
            if self.processing_times:
                stats['avg_processing_time_ms'] = statistics.mean(self.processing_times) * 1000
                stats['p95_processing_time_ms'] = np.percentile(self.processing_times, 95) * 1000 if len(self.processing_times) > 1 else 0
            else:
                stats['avg_processing_time_ms'] = 0
                stats['p95_processing_time_ms'] = 0
                
            if self.batch_sizes:
                stats['avg_batch_size'] = statistics.mean(self.batch_sizes)
                stats['max_batch_size'] = max(self.batch_sizes)
                stats['min_batch_size'] = min(self.batch_sizes)
            else:
                stats['avg_batch_size'] = 0
                stats['max_batch_size'] = 0
                stats['min_batch_size'] = 0
                
            if self.queue_times:
                stats['avg_queue_time_ms'] = statistics.mean(self.queue_times) * 1000
            else:
                stats['avg_queue_time_ms'] = 0
                
            if self.latencies:
                stats['avg_latency_ms'] = statistics.mean(self.latencies) * 1000
                stats['p95_latency_ms'] = np.percentile(self.latencies, 95) * 1000 if len(self.latencies) > 1 else 0
                stats['p99_latency_ms'] = np.percentile(self.latencies, 99) * 1000 if len(self.latencies) > 1 else 0
            else:
                stats['avg_latency_ms'] = 0
                stats['p95_latency_ms'] = 0
                stats['p99_latency_ms'] = 0
                
            if self.queue_lengths:
                stats['avg_queue_length'] = statistics.mean(self.queue_lengths)
                stats['max_queue_length'] = max(self.queue_lengths)
            else:
                stats['avg_queue_length'] = 0
                stats['max_queue_length'] = 0
                
            stats['error_rate'] = self.error_counts / max(1, self.total_processed)
            
            # Calculate throughput (items/second)
            total_processing_time = sum(self.processing_times)
            if total_processing_time > 0:
                stats['throughput'] = self.total_processed / total_processing_time
            else:
                stats['throughput'] = 0
                
            stats['batch_count'] = self.batch_counts
            stats['total_processed'] = self.total_processed
            
            # Cache the results
            self._stats_cache = stats.copy()
            return stats


__all__ = ['BatchStats']
