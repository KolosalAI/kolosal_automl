"""
Thread-safe performance metrics tracking for inference operations.

This module provides comprehensive performance monitoring capabilities
with thread-safe metric collection and statistical analysis.
"""

import time
import threading
import numpy as np
from typing import Dict, Any


class PerformanceMetrics:
    """Thread-safe container for tracking performance metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.lock = threading.RLock()
        
        # Initialize metrics storage
        self.inference_times = []
        self.batch_sizes = []
        self.preprocessing_times = []
        self.quantization_times = []
        self.total_requests = 0
        self.total_errors = 0
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        self.last_updated = time.time()
    
    def update_inference(self, inference_time: float, batch_size: int, 
                         preprocessing_time: float = 0.0, quantization_time: float = 0.0):
        """Update inference metrics with thread safety"""
        with self.lock:
            # Keep fixed window of metrics
            if len(self.inference_times) >= self.window_size:
                self.inference_times.pop(0)
                self.batch_sizes.pop(0)
                self.preprocessing_times.pop(0)
                self.quantization_times.pop(0)
            
            self.inference_times.append(inference_time)
            self.batch_sizes.append(batch_size)
            self.preprocessing_times.append(preprocessing_time)
            self.quantization_times.append(quantization_time)
            self.total_requests += batch_size
            self.last_updated = time.time()
    
    def record_error(self):
        """Record an inference error"""
        with self.lock:
            self.total_errors += 1
    
    def record_cache_hit(self):
        """Record a cache hit"""
        with self.lock:
            self.total_cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        with self.lock:
            self.total_cache_misses += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics as a dictionary"""
        with self.lock:
            metrics = {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": self.total_errors / max(1, self.total_requests),
                "cache_hit_rate": self.total_cache_hits / max(1, self.total_cache_hits + self.total_cache_misses),
                "last_updated": self.last_updated
            }
            
            # Add time-based metrics if we have data
            if self.inference_times:
                metrics.update({
                    "avg_inference_time_ms": np.mean(self.inference_times) * 1000,
                    "p95_inference_time_ms": np.percentile(self.inference_times, 95) * 1000,
                    "p99_inference_time_ms": np.percentile(self.inference_times, 99) * 1000,
                    "max_inference_time_ms": np.max(self.inference_times) * 1000,
                })
            
            if self.batch_sizes:
                metrics.update({
                    "avg_batch_size": np.mean(self.batch_sizes),
                    "max_batch_size": np.max(self.batch_sizes),
                })
            
            if self.preprocessing_times:
                metrics.update({
                    "avg_preprocessing_time_ms": np.mean(self.preprocessing_times) * 1000,
                })
            
            if self.quantization_times:
                metrics.update({
                    "avg_quantization_time_ms": np.mean(self.quantization_times) * 1000,
                })
                
            # Calculate throughput
            if self.inference_times:
                total_time = sum(self.inference_times)
                if total_time > 0:
                    metrics["throughput_requests_per_second"] = sum(self.batch_sizes) / total_time
            
            return metrics


__all__ = ['PerformanceMetrics']
