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

    # Additional methods expected by tests
    @property
    def initial_memory_mb(self) -> float:
        """Get initial memory usage in MB"""
        return self._initial_memory * 1024

    @property
    def current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self._get_current_memory() * 1024

    @property
    def memory_history(self) -> List[float]:
        """Get memory usage history"""
        return self._memory_history.copy()

    def get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        return self.current_memory_mb

    def update_memory(self) -> float:
        """Update memory tracking and return current usage in MB"""
        current = self._get_current_memory()
        with self._lock:
            self._memory_history.append(current)
            if len(self._memory_history) > 100:
                self._memory_history = self._memory_history[-50:]
        return current * 1024

    def get_memory_pressure(self) -> float:
        """Get current memory pressure as percentage"""
        stats = self.get_memory_stats()
        return stats.usage_percentage

    def is_memory_pressure_high(self, threshold: float = None) -> bool:
        """Check if memory pressure is high"""
        threshold = threshold or self.warning_threshold
        return self.get_memory_pressure() > threshold

    def is_memory_pressure_critical(self, threshold: float = None) -> bool:
        """Check if memory pressure is critical"""
        threshold = threshold or self.critical_threshold
        return self.get_memory_pressure() > threshold

    def get_available_memory(self) -> float:
        """Get available memory in MB"""
        stats = self.get_memory_stats()
        return stats.available_mb

    def get_memory_delta(self) -> float:
        """Get memory delta since last measurement in MB"""
        if len(self._memory_history) < 2:
            return 0.0
        return (self._memory_history[-1] - self._memory_history[-2]) * 1024

    def clear_history(self):
        """Clear memory history"""
        with self._lock:
            self._memory_history.clear()

    def get_memory_history(self) -> List[float]:
        """Get memory history in MB"""
        return [m * 1024 for m in self._memory_history]


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
            return {
                'total_time_seconds': 0.0,
                'avg_chunk_size': 0.0,
                'avg_processing_time': 0.0,
                'avg_memory_used': 0.0,
                'avg_rows_per_second': 0.0,
                'total_chunks_processed': 0
            }
        
        recent_perf = self._performance_history[-10:]
        total_time = sum(p['processing_time'] for p in recent_perf)
        
        return {
            'total_time_seconds': total_time,
            'avg_chunk_size': np.mean([p['chunk_size'] for p in recent_perf]),
            'avg_processing_time': np.mean([p['processing_time'] for p in recent_perf]),
            'avg_memory_used': np.mean([p['memory_used'] for p in recent_perf]),
            'avg_rows_per_second': np.mean([p['rows_per_second'] for p in recent_perf]),
            'total_chunks_processed': len(self._performance_history)
        }

    def select_chunking_strategy(self, data):
        """Select appropriate chunking strategy based on data characteristics"""
        if len(data) < 1000:
            return ChunkingStrategy.FIXED
        elif len(data) < 100000:
            return ChunkingStrategy.ADAPTIVE
        else:
            return ChunkingStrategy.MEMORY_BASED

    def calculate_adaptive_chunk_size(self, data):
        """Calculate adaptive chunk size based on data and memory"""
        stats = self.memory_monitor.get_memory_stats()
        
        # Base chunk size based on available memory
        available_memory_mb = stats.available_gb * 1024
        target_memory_mb = min(available_memory_mb * 0.1, 100)  # Use 10% of available memory or 100MB max
        
        # Estimate memory per row
        if len(data) > 0:
            sample_size = min(100, len(data))
            sample_memory_mb = data.head(sample_size).memory_usage(deep=True).sum() / 1024 / 1024
            memory_per_row = sample_memory_mb / sample_size if sample_size > 0 else 0.001
            
            chunk_size = int(target_memory_mb / memory_per_row) if memory_per_row > 0 else 1000
        else:
            chunk_size = 1000
        
        # Ensure reasonable bounds
        return max(self.min_chunk_size, min(self.max_chunk_size, chunk_size))

    def calculate_memory_based_chunk_size(self, data, target_memory_mb=None):
        """Calculate chunk size based on memory constraints"""
        if target_memory_mb is None:
            target_memory_mb = 100  # Default 100MB limit
        
        if len(data) == 0:
            return self.min_chunk_size
        
        # Estimate memory per row
        sample_size = min(100, len(data))
        sample_memory_mb = data.head(sample_size).memory_usage(deep=True).sum() / 1024 / 1024
        memory_per_row = sample_memory_mb / sample_size if sample_size > 0 else 0.001
        
        chunk_size = int(target_memory_mb / memory_per_row) if memory_per_row > 0 else self.min_chunk_size
        
        # Ensure reasonable bounds and don't exceed data length
        return max(self.min_chunk_size, min(len(data), min(self.max_chunk_size, chunk_size)))

    def process_with_adaptive_chunking(self, data, process_func=None):
        """Process data with adaptive chunking strategy"""
        if process_func is None:
            process_func = lambda x: x  # Default identity operation
        
        chunk_size = self.calculate_adaptive_chunk_size(data)
        self.current_chunk_size = chunk_size
        
        results = []
        for chunk in self.process_chunks(data, process_func):
            results.append(chunk)
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return data.iloc[0:0] if hasattr(data, 'iloc') else data


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
    
    def process_data(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Process features and target data with memory optimization and categorical encoding
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dict containing processed X and y
        """
        with self.memory_context("data_preprocessing"):
            # Make a copy to avoid modifying the original data
            X_processed = X.copy()
            
            # Handle categorical features first (before memory optimization)
            categorical_columns = X_processed.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                for col in categorical_columns:
                    X_processed[col] = pd.Categorical(X_processed[col]).codes
            
            # Then optimize memory usage
            X_processed = self.optimize_dataframe_memory(X_processed)
            
            return {'X': X_processed, 'y': y}
    
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

    def select_chunking_strategy(self, data):
        """Select appropriate chunking strategy based on data characteristics"""
        return self.chunk_processor.select_chunking_strategy(data)

    def calculate_adaptive_chunk_size(self, data):
        """Calculate adaptive chunk size based on data and memory"""
        return self.chunk_processor.calculate_adaptive_chunk_size(data)

    def calculate_memory_based_chunk_size(self, data, target_memory_mb=None):
        """Calculate chunk size based on memory constraints"""
        return self.chunk_processor.calculate_memory_based_chunk_size(data, target_memory_mb)

    def process_with_adaptive_chunking(self, data, process_func=None):
        """Process data with adaptive chunking strategy"""
        return self.chunk_processor.process_with_adaptive_chunking(data, process_func)

    def get_performance_stats(self):
        """Get performance statistics"""
        processing_stats = self.get_processing_stats()
        chunk_stats = self.chunk_processor.get_performance_stats()
        memory_stats = self.memory_monitor.get_memory_stats()
        
        return {
            'processing_stats': {
                'total_processing_time': processing_stats.total_processing_time,
                'total_data_processed_mb': processing_stats.total_data_processed_mb,
                'memory_peak_mb': processing_stats.memory_peak_mb,
                'gc_count': processing_stats.gc_count
            },
            'chunk_stats': chunk_stats,
            'memory_stats': {
                'current_usage_gb': memory_stats.used_gb,
                'usage_percentage': memory_stats.usage_percentage,
                'pressure_level': memory_stats.pressure_level.value
            }
        }

    # Additional methods expected by tests
    @property
    def monitor(self):
        """Get the memory monitor instance"""
        return self.memory_monitor

    @property 
    def warning_threshold(self):
        """Get the warning threshold"""
        return self.memory_monitor.warning_threshold

    @property
    def enable_numa(self):
        """Check if NUMA is enabled"""
        return self.memory_pool is not None and getattr(self.memory_pool, 'numa_aware', False)

    @property
    def critical_threshold(self):
        """Get the critical threshold"""
        return self.memory_monitor.critical_threshold

    def process_in_chunks(self, df: pd.DataFrame, process_func: Callable, chunk_size: int = None) -> pd.DataFrame:
        """Process DataFrame in chunks"""
        start_time = time.time()
        
        if chunk_size is None:
            chunk_size = self.calculate_optimal_chunk_size(df)
        
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            processed_chunk = process_func(chunk)
            chunks.append(processed_chunk)
        
        result = pd.concat(chunks, ignore_index=True)
        
        # Update processing stats
        processing_time = time.time() - start_time
        self._processing_stats.total_processing_time += processing_time
        self._processing_stats.chunks_processed += len(chunks)
        
        return result

    def should_use_chunking(self, df: pd.DataFrame) -> bool:
        """Determine if chunking should be used based on data size and memory pressure"""
        # If data is large (>=50000 rows), use chunking
        if len(df) >= 50000:
            return True
            
        # If data is small (<10000 rows), don't use chunking
        if len(df) < 10000:
            return False
        
        # For medium data, check memory pressure
        stats = self.memory_monitor.get_memory_stats()
        return stats.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]

    def calculate_optimal_chunk_size(self, df: pd.DataFrame, target_memory_mb: float = 100) -> int:
        """Calculate optimal chunk size based on memory constraints"""
        if len(df) == 0:
            return 1000
        
        # Estimate memory per row
        sample_size = min(1000, len(df))
        sample_memory_mb = df.head(sample_size).memory_usage(deep=True).sum() / 1024 / 1024
        memory_per_row = sample_memory_mb / sample_size
        
        # Calculate chunk size to fit in target memory
        chunk_size = int(target_memory_mb / memory_per_row) if memory_per_row > 0 else 1000
        
        # Ensure reasonable bounds and don't exceed DataFrame length
        chunk_size = max(100, min(len(df), chunk_size))
        
        return chunk_size

    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check current memory pressure"""
        stats = self.memory_monitor.get_memory_stats()
        return {
            "pressure_level": stats.pressure_level.value,
            "usage_percentage": stats.usage_percentage,
            "available_gb": stats.available_gb
        }

    def cleanup_memory(self):
        """Cleanup memory and trigger garbage collection"""
        freed = self.memory_monitor.trigger_gc(aggressive=True)
        return freed

    def get_detailed_memory_usage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed memory usage information for a DataFrame"""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        # Create column_memory dict for backward compatibility
        column_memory = {col: mem / 1024 / 1024 for col, mem in memory_usage.items()}
        
        # Create dtype distribution
        dtype_counts = df.dtypes.value_counts()
        dtype_distribution = {str(dtype): count for dtype, count in dtype_counts.items()}
        
        return {
            "total_memory_mb": total_memory / 1024 / 1024,
            "memory_per_column": column_memory,
            "column_memory": column_memory,  # For test compatibility
            "dtype_distribution": dtype_distribution,
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict()
        }

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
