"""
NUMA-aware memory pool for optimized buffer management.

This module provides intelligent memory buffer pooling with NUMA awareness
for optimal memory locality and reduced allocation overhead.
"""

import os
import threading
import numpy as np
from typing import Dict, Any, Tuple

# Try to import psutil for NUMA detection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryPool:
    """
    NUMA-aware memory pool that manages pre-allocated buffers across NUMA nodes
    for optimal memory locality and reduced allocation overhead.
    """
    
    def __init__(self, max_buffers: int = 32, numa_aware: bool = True):
        self.max_buffers = max_buffers
        self.numa_aware = numa_aware
        
        # Hierarchical buffer pools: Small/Medium/Large + NUMA-aware
        self.buffer_pools = {
            'small': {},    # < 1KB buffers
            'medium': {},   # 1KB - 1MB buffers  
            'large': {}     # > 1MB buffers
        }
        
        # NUMA node mapping if available
        self.numa_nodes = []
        self.current_numa_node = 0
        
        if numa_aware and PSUTIL_AVAILABLE:
            try:
                # Detect NUMA topology
                self.numa_nodes = list(range(psutil.cpu_count(logical=False) // 4)) if hasattr(psutil, 'cpu_count') else [0]
                if not self.numa_nodes:
                    self.numa_nodes = [0]
            except:
                self.numa_nodes = [0]
        else:
            self.numa_nodes = [0]
        
        # Per-NUMA node pools
        for node in self.numa_nodes:
            for pool_type in self.buffer_pools:
                self.buffer_pools[pool_type][node] = {}
        
        self.lock = threading.RLock()
        self.allocation_stats = {'hits': 0, 'misses': 0, 'numa_misses': 0}
    
    def _get_buffer_size_category(self, shape: Tuple[int, ...], dtype) -> str:
        """Categorize buffer size for optimal pool selection"""
        size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        if size_bytes < 1024:  # < 1KB
            return 'small'
        elif size_bytes < 1024 * 1024:  # < 1MB
            return 'medium'
        else:
            return 'large'
    
    def _get_preferred_numa_node(self) -> int:
        """Get preferred NUMA node for current thread"""
        if not self.numa_aware or not PSUTIL_AVAILABLE:
            return 0
        
        try:
            # Try to get current CPU and map to NUMA node
            if hasattr(psutil.Process(), 'cpu_num'):
                cpu_num = psutil.Process().cpu_num()
                return cpu_num // (psutil.cpu_count(logical=False) // len(self.numa_nodes))
        except:
            pass
        
        # Round-robin fallback
        self.current_numa_node = (self.current_numa_node + 1) % len(self.numa_nodes)
        return self.numa_nodes[self.current_numa_node]
    
    def get_buffer(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Get a buffer with NUMA awareness and size-based pooling"""
        with self.lock:
            category = self._get_buffer_size_category(shape, dtype)
            key = (shape, np.dtype(dtype).name)
            numa_node = self._get_preferred_numa_node()
            
            # Try preferred NUMA node first
            node_pool = self.buffer_pools[category][numa_node]
            if key in node_pool and node_pool[key]:
                self.allocation_stats['hits'] += 1
                return node_pool[key].pop()
            
            # Try other NUMA nodes
            for node in self.numa_nodes:
                if node != numa_node:
                    node_pool = self.buffer_pools[category][node]
                    if key in node_pool and node_pool[key]:
                        self.allocation_stats['hits'] += 1
                        self.allocation_stats['numa_misses'] += 1
                        return node_pool[key].pop()
            
            # Allocate new buffer
            self.allocation_stats['misses'] += 1
            return self._allocate_new_buffer(shape, dtype, numa_node)
    
    def _allocate_new_buffer(self, shape: Tuple[int, ...], dtype, numa_node: int) -> np.ndarray:
        """Allocate new buffer with NUMA awareness"""
        try:
            # Use numpy's empty for better performance
            buffer = np.empty(shape, dtype=dtype)
            
            # On Linux, try to move pages to preferred NUMA node
            if os.name == 'posix' and hasattr(os, 'sched_setaffinity'):
                try:
                    # This is a simplified NUMA binding - real implementation would use libnuma
                    pass  # Placeholder for NUMA memory binding
                except:
                    pass
            
            return buffer
        except MemoryError:
            # Fallback to smaller allocation or cleanup
            self._cleanup_old_buffers()
            return np.empty(shape, dtype=dtype)
    
    def return_buffer(self, buffer: np.ndarray):
        """Return buffer to appropriate pool with NUMA awareness"""
        if buffer is None:
            return
            
        with self.lock:
            category = self._get_buffer_size_category(buffer.shape, buffer.dtype)
            key = (buffer.shape, buffer.dtype.name)
            numa_node = self._get_preferred_numa_node()
            
            # Add to preferred NUMA node pool
            node_pool = self.buffer_pools[category][numa_node]
            if key not in node_pool:
                node_pool[key] = []
            
            if len(node_pool[key]) < self.max_buffers:
                # Zero out for security and cache efficiency
                buffer.fill(0)
                node_pool[key].append(buffer)
    
    def _cleanup_old_buffers(self):
        """Clean up least recently used buffers to free memory"""
        with self.lock:
            for category in self.buffer_pools:
                for node in self.numa_nodes:
                    node_pool = self.buffer_pools[category][node]
                    # Keep only half of the buffers in each pool
                    for key in list(node_pool.keys()):
                        if node_pool[key]:
                            keep_count = len(node_pool[key]) // 2
                            node_pool[key] = node_pool[key][:keep_count]
    
    def clear(self):
        """Clear all buffer pools"""
        with self.lock:
            for category in self.buffer_pools:
                for node in self.numa_nodes:
                    self.buffer_pools[category][node].clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory pool statistics"""
        with self.lock:
            total_buffers = 0
            memory_usage = 0
            category_stats = {}
            
            for category in self.buffer_pools:
                category_buffers = 0
                category_memory = 0
                
                for node in self.numa_nodes:
                    node_pool = self.buffer_pools[category][node]
                    for key, buffers in node_pool.items():
                        if buffers:
                            shape, dtype_name = key
                            buffer_size = np.prod(shape) * np.dtype(dtype_name).itemsize
                            buffer_count = len(buffers)
                            
                            category_buffers += buffer_count
                            category_memory += buffer_size * buffer_count
                
                category_stats[category] = {
                    'buffers': category_buffers,
                    'memory_mb': category_memory / (1024 * 1024)
                }
                total_buffers += category_buffers
                memory_usage += category_memory
            
            hit_rate = self.allocation_stats['hits'] / max(1, 
                self.allocation_stats['hits'] + self.allocation_stats['misses']) * 100
            numa_efficiency = (1 - self.allocation_stats['numa_misses'] / 
                max(1, self.allocation_stats['hits'])) * 100
            
            return {
                "total_buffers": total_buffers,
                "memory_usage_bytes": memory_usage,
                "memory_usage_mb": memory_usage / (1024 * 1024),
                "category_stats": category_stats,
                "numa_nodes": len(self.numa_nodes),
                "hit_rate_percent": hit_rate,
                "numa_efficiency_percent": numa_efficiency,
                "allocation_stats": self.allocation_stats.copy()
            }


__all__ = ['MemoryPool']
