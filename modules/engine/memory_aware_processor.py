"""
Memory-Aware Data Processing Manager for Large Datasets

This module provides intelligent memory management during data processing,
integrating with the existing memory pool and providing adaptive processing
strategies based on available memory and dataset characteristics.
"""

import gc
import logging
import threading
import time
import psutil
from typing import Dict, Any, Optional, List, Callable, Generator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from contextlib import contextmanager

try:
    from modules.engine.memory_pool import MemoryPool
    MEMORY_POOL_AVAILABLE = True
except (ImportError, SystemError, Exception):
    MEMORY_POOL_AVAILABLE = False


class MemoryPressureLevel(Enum):
    """Memory pressure levels for adaptive processing"""
    LOW = "low"           # < 50% memory usage
    MODERATE = "moderate" # 50-70% memory usage
    HIGH = "high"         # 70-85% memory usage
    CRITICAL = "critical" # > 85% memory usage


class ChunkingStrategy(Enum):
    """Chunking strategies for different processing scenarios"""
    FIXED = "fixed"               # Fixed chunk size
    ADAPTIVE = "adaptive"         # Adaptive chunk size based on performance
    MEMORY_BASED = "memory_based" # Chunk size based on memory pressure


@dataclass
class MemoryStats:
    """Memory statistics for monitoring"""
    total_gb: float
    available_gb: float
    used_gb: float
    usage_percentage: float
    pressure_level: MemoryPressureLevel
    gc_count: int
    peak_usage_gb: float
    
    # Additional properties for compatibility
    current_usage_mb: float = field(init=False)
    available_mb: float = field(init=False)
    
    def __post_init__(self):
        self.current_usage_mb = self.used_gb * 1024
        self.available_mb = self.available_gb * 1024


@dataclass 
class ProcessingStats:
    """Processing statistics for performance monitoring"""
    total_processing_time: float = 0.0
    chunks_processed: int = 0
    memory_peak_mb: float = 0.0
    avg_chunk_processing_time: float = 0.0
    total_data_processed_mb: float = 0.0
    gc_count: int = 0


class MemoryMonitor:
    """Real-time memory monitoring with adaptive thresholds"""
    
    def __init__(self, warning_threshold: float = 70.0, critical_threshold: float = 85.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = logging.getLogger(__name__)
        
        self._process = psutil.Process()
        self._initial_memory = self._get_current_memory()
        self._peak_memory = self._initial_memory
        self._gc_count = 0
        self._memory_history: List[float] = []
        self._lock = threading.RLock()
        
    def _get_current_memory(self) -> float:
        """Get current memory usage in GB"""
        return self._process.memory_info().rss / (1024**3)
    
    def _get_system_memory(self) -> Tuple[float, float, float]:
        """Get system memory info in GB"""
        memory = psutil.virtual_memory()
        total = memory.total / (1024**3)
        available = memory.available / (1024**3)
        used = memory.used / (1024**3)
        return total, available, used
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics"""
        with self._lock:
            current_memory = self._get_current_memory()
            total, available, used = self._get_system_memory()
            usage_pct = (used / total) * 100
            
            # Update peak
            self._peak_memory = max(self._peak_memory, current_memory)
            self._memory_history.append(current_memory)
            
            # Keep only recent history
            if len(self._memory_history) > 100:
                self._memory_history = self._memory_history[-50:]
            
            # Determine pressure level
            if usage_pct < 50:
                pressure = MemoryPressureLevel.LOW
            elif usage_pct < self.warning_threshold:
                pressure = MemoryPressureLevel.MODERATE
            elif usage_pct < self.critical_threshold:
                pressure = MemoryPressureLevel.HIGH
            else:
                pressure = MemoryPressureLevel.CRITICAL
            
            return MemoryStats(
                total_gb=total,
                available_gb=available,
                used_gb=used,
                usage_percentage=usage_pct,
                pressure_level=pressure,
                gc_count=self._gc_count,
                peak_usage_gb=self._peak_memory
            )
    
    def should_trigger_gc(self, force_threshold: float = None) -> bool:
        """Check if garbage collection should be triggered"""
        stats = self.get_memory_stats()
        threshold = force_threshold or self.warning_threshold
        return stats.usage_percentage > threshold
    
    def trigger_gc(self, aggressive: bool = False) -> float:
        """Trigger garbage collection and return memory freed"""
        before_memory = self._get_current_memory()
        
        if aggressive:
            # Aggressive GC
            for _ in range(3):
                gc.collect()
        else:
            gc.collect()
        
        after_memory = self._get_current_memory()
        freed_gb = before_memory - after_memory
        
        with self._lock:
            self._gc_count += 1
        
        if freed_gb > 0.1:  # Log if significant memory was freed
            self.logger.info(f"Garbage collection freed {freed_gb:.2f} GB")
        
        return freed_gb


class AdaptiveChunkProcessor:
    """Adaptive chunk processor that adjusts chunk size based on memory pressure"""
    
    def __init__(self, 
                 initial_chunk_size: int = 10000,
                 min_chunk_size: int = 1000,
                 max_chunk_size: int = 100000,
                 memory_monitor: Optional[MemoryMonitor] = None):
        
        self.current_chunk_size = initial_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Adaptive parameters
        self._performance_history: List[Dict[str, float]] = []
        self._chunk_adjustment_factor = 1.2
        
    def get_adaptive_chunk_size(self, data_size: int) -> int:
        """Get chunk size adapted to current memory pressure and data size"""
        stats = self.memory_monitor.get_memory_stats()
        
        # Base adjustment based on memory pressure
        if stats.pressure_level == MemoryPressureLevel.CRITICAL:
            adjusted_size = max(self.min_chunk_size, int(self.current_chunk_size * 0.5))
        elif stats.pressure_level == MemoryPressureLevel.HIGH:
            adjusted_size = max(self.min_chunk_size, int(self.current_chunk_size * 0.7))
        elif stats.pressure_level == MemoryPressureLevel.LOW:
            adjusted_size = min(self.max_chunk_size, int(self.current_chunk_size * 1.3))
        else:  # MODERATE
            adjusted_size = self.current_chunk_size
        
        # Ensure chunk size is not larger than data size
        adjusted_size = min(adjusted_size, data_size)
        
        # Update current chunk size for next iteration
        self.current_chunk_size = adjusted_size
        
        return adjusted_size
    
    def process_chunks(self, 
                      data: pd.DataFrame, 
                      processing_func: Callable[[pd.DataFrame], pd.DataFrame],
                      progress_callback: Optional[Callable] = None) -> Generator[pd.DataFrame, None, None]:
        """Process data in adaptive chunks with memory monitoring"""
        
        total_rows = len(data)
        processed_rows = 0
        chunk_count = 0
        
        while processed_rows < total_rows:
            # Get adaptive chunk size
            remaining_rows = total_rows - processed_rows
            chunk_size = self.get_adaptive_chunk_size(remaining_rows)
            
            # Extract chunk
            end_idx = min(processed_rows + chunk_size, total_rows)
            chunk = data.iloc[processed_rows:end_idx].copy()
            
            # Process chunk with memory monitoring
            start_time = time.time()
            start_memory = self.memory_monitor._get_current_memory()
            
            try:
                processed_chunk = processing_func(chunk)
                
                # Update performance history
                processing_time = time.time() - start_time
                memory_used = self.memory_monitor._get_current_memory() - start_memory
                
                self._performance_history.append({
                    'chunk_size': chunk_size,
                    'processing_time': processing_time,
                    'memory_used': memory_used,
                    'rows_per_second': len(chunk) / processing_time if processing_time > 0 else 0
                })
                
                # Keep only recent history
                if len(self._performance_history) > 20:
                    self._performance_history = self._performance_history[-10:]
                
                yield processed_chunk
                
                # Update progress
                processed_rows = end_idx
                chunk_count += 1
                
                if progress_callback:
                    progress_callback(processed_rows, total_rows, chunk_count)
                
                # Check memory pressure and trigger GC if needed
                if self.memory_monitor.should_trigger_gc():
                    freed = self.memory_monitor.trigger_gc()
                    self.logger.debug(f"Triggered GC after chunk {chunk_count}, freed {freed:.2f} GB")
                
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_count}: {e}")
                # Try with smaller chunk size
                if chunk_size > self.min_chunk_size:
                    self.current_chunk_size = max(self.min_chunk_size, int(chunk_size * 0.5))
                    self.logger.info(f"Reduced chunk size to {self.current_chunk_size} after error")
                raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self._performance_history:
            return {}
        
        recent_perf = self._performance_history[-10:]
        
        return {
            'avg_chunk_size': np.mean([p['chunk_size'] for p in recent_perf]),
            'avg_processing_time': np.mean([p['processing_time'] for p in recent_perf]),
            'avg_memory_used': np.mean([p['memory_used'] for p in recent_perf]),
            'avg_rows_per_second': np.mean([p['rows_per_second'] for p in recent_perf]),
            'total_chunks_processed': len(self._performance_history)
        }


class MemoryAwareDataProcessor:
    """
    Main data processor that integrates memory monitoring, adaptive chunking,
    and memory pool management for efficient large dataset processing
    """
    
    def __init__(self, 
                 memory_pool_size: int = 32,
                 enable_numa: bool = True,
                 warning_threshold: float = 70.0,
                 critical_threshold: float = 85.0):
        
        self.logger = logging.getLogger(__name__)
        self.memory_monitor = MemoryMonitor(warning_threshold, critical_threshold)
        
        # Initialize memory pool if available
        self.memory_pool = None
        if MEMORY_POOL_AVAILABLE:
            try:
                self.memory_pool = MemoryPool(max_buffers=memory_pool_size, numa_aware=enable_numa)
                self.logger.info("Memory pool initialized with NUMA awareness")
            except Exception as e:
                self.logger.warning(f"Failed to initialize memory pool: {e}")
        
        # Initialize adaptive chunk processor
        self.chunk_processor = AdaptiveChunkProcessor(memory_monitor=self.memory_monitor)
        
        # Processing statistics
        self._processing_stats = ProcessingStats()
        self._start_time = None
        
    @contextmanager
    def memory_context(self, operation_name: str = "data_processing"):
        """Context manager for memory-aware processing"""
        start_time = time.time()
        start_stats = self.memory_monitor.get_memory_stats()
        
        self.logger.info(f"Starting {operation_name} - Memory: {start_stats.usage_percentage:.1f}%")
        
        try:
            yield self.memory_monitor
        finally:
            end_time = time.time()
            end_stats = self.memory_monitor.get_memory_stats()
            
            processing_time = end_time - start_time
            memory_change = end_stats.used_gb - start_stats.used_gb
            
            self.logger.info(f"Completed {operation_name} - "
                           f"Time: {processing_time:.2f}s, "
                           f"Memory change: {memory_change:+.2f} GB, "
                           f"Final usage: {end_stats.usage_percentage:.1f}%")
            
            # Update stats
            self._processing_stats.total_processing_time += processing_time
            if memory_change > 0:
                self._processing_stats.total_data_processed_mb += memory_change * 1024
    
    def process_large_dataframe(self,
                               df: pd.DataFrame,
                               processing_func: Callable[[pd.DataFrame], pd.DataFrame],
                               progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Process large DataFrame with memory-aware chunking
        
        Args:
            df: Input DataFrame
            processing_func: Function to apply to each chunk
            progress_callback: Optional progress callback function
            
        Returns:
            Processed DataFrame
        """
        with self.memory_context(f"large_dataframe_processing_{len(df)}_rows"):
            initial_memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Check if chunking is needed
            stats = self.memory_monitor.get_memory_stats()
            
            if len(df) < 10000 or stats.pressure_level == MemoryPressureLevel.LOW:
                # Process directly if small dataset or low memory pressure
                self.logger.info("Processing DataFrame directly (no chunking needed)")
                return processing_func(df)
            
            # Process in chunks
            self.logger.info(f"Processing DataFrame in chunks (initial memory: {initial_memory_mb:.1f} MB)")
            
            processed_chunks = []
            chunk_count = 0
            
            for processed_chunk in self.chunk_processor.process_chunks(
                df, processing_func, progress_callback
            ):
                processed_chunks.append(processed_chunk)
                chunk_count += 1
                
                # Monitor memory pressure
                current_stats = self.memory_monitor.get_memory_stats()
                if current_stats.pressure_level == MemoryPressureLevel.CRITICAL:
                    self.logger.warning(f"Critical memory pressure detected at chunk {chunk_count}")
            
            # Combine chunks
            self.logger.info(f"Combining {len(processed_chunks)} processed chunks")
            final_df = pd.concat(processed_chunks, ignore_index=True)
            
            # Cleanup
            del processed_chunks
            if self.memory_monitor.should_trigger_gc():
                self.memory_monitor.trigger_gc()
                self._processing_stats.gc_count += 1
            
            final_memory_mb = final_df.memory_usage(deep=True).sum() / 1024 / 1024
            self.logger.info(f"Processing complete. Final memory: {final_memory_mb:.1f} MB")
            
            return final_df
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage through dtype optimization"""
        with self.memory_context("memory_optimization"):
            original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize object columns
            for col in df.select_dtypes(include=['object']).columns:
                num_unique_values = df[col].nunique()
                num_total_values = len(df[col])
                
                # Convert to category if less than 50% unique values
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            reduction = (1 - optimized_memory / original_memory) * 100
            
            self.logger.info(f"Memory optimization: {original_memory:.1f} MB -> "
                           f"{optimized_memory:.1f} MB ({reduction:.1f}% reduction)")
            
            return df
    
    def get_processing_stats(self) -> ProcessingStats:
        """Get comprehensive processing statistics"""
        memory_stats = self.memory_monitor.get_memory_stats()
        
        # Update peak memory
        self._processing_stats.memory_peak_mb = max(
            self._processing_stats.memory_peak_mb,
            memory_stats.used_gb * 1024
        )
        
        return self._processing_stats
    
    def cleanup(self):
        """Cleanup resources and trigger final garbage collection"""
        self.logger.info("Cleaning up memory-aware data processor")
        
        if self.memory_pool:
            # Memory pool cleanup would go here if the pool supports it
            pass
        
        # Final garbage collection
        freed = self.memory_monitor.trigger_gc(aggressive=True)
        self.logger.info(f"Final cleanup freed {freed:.2f} GB")


# Integration functions
def create_memory_aware_processor(warning_threshold: float = 70.0,
                                critical_threshold: float = 85.0,
                                enable_numa: bool = True) -> MemoryAwareDataProcessor:
    """Create a memory-aware data processor with optimized settings"""
    return MemoryAwareDataProcessor(
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        enable_numa=enable_numa
    )


# Integration functions
def create_memory_aware_processor(warning_threshold: float = 70.0,
                                critical_threshold: float = 85.0,
                                enable_numa: bool = True) -> MemoryAwareDataProcessor:
    """Create a memory-aware data processor with optimized settings"""
    return MemoryAwareDataProcessor(
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        enable_numa=enable_numa
    )


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to optimize DataFrame memory usage
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    processor = create_memory_aware_processor()
    return processor.optimize_dataframe_memory(df)


def monitor_memory_usage(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage of a function
    
    Args:
        func: Function to monitor
        
    Returns:
        Decorated function with memory monitoring
    """
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        start_stats = monitor.get_memory_stats()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_stats = monitor.get_memory_stats()
            memory_change = end_stats.used_gb - start_stats.used_gb
            
            if memory_change > 0.1:  # Log significant memory changes
                logging.getLogger(__name__).info(
                    f"Function {func.__name__} used {memory_change:.2f} GB of memory"
                )
    
    return wrapper


def process_dataframe_with_memory_management(df: pd.DataFrame,
                                           processing_func: Callable,
                                           memory_threshold: float = 70.0) -> pd.DataFrame:
    """
    Convenience function to process DataFrame with automatic memory management
    
    Args:
        df: Input DataFrame
        processing_func: Function to apply to the data
        memory_threshold: Memory usage threshold for triggering chunked processing
        
    Returns:
        Processed DataFrame
    """
    processor = create_memory_aware_processor(warning_threshold=memory_threshold)
    
    try:
        return processor.process_large_dataframe(df, processing_func)
    finally:
        processor.cleanup()
