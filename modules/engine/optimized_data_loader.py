"""
Optimized Data Loader for handling both small and large datasets (>1M rows)

This module provides intelligent data loading strategies based on dataset size,
memory constraints, and system capabilities.

Features:
- Automatic dataset size detection and optimization strategy selection
- Memory-efficient chunked loading for large datasets
- Streaming data processing with adaptive batch sizes
- Memory monitoring and garbage collection
- Support for multiple data formats with optimization
- Integration with existing preprocessing pipeline
"""

import os
import gc
import psutil
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Generator, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from contextlib import contextmanager

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    from .memory_aware_processor import MemoryAwareDataProcessor, create_memory_aware_processor
    MEMORY_PROCESSOR_AVAILABLE = True
except ImportError:
    MEMORY_PROCESSOR_AVAILABLE = False


class DatasetSize(Enum):
    """Dataset size categories for optimization strategy selection"""
    TINY = "tiny"           # < 10K rows
    SMALL = "small"         # 10K - 100K rows  
    MEDIUM = "medium"       # 100K - 1M rows
    LARGE = "large"         # 1M - 10M rows
    HUGE = "huge"           # > 10M rows


class LoadingStrategy(Enum):
    """Data loading strategies based on dataset characteristics"""
    DIRECT = "direct"                    # Load entire dataset into memory
    CHUNKED = "chunked"                  # Process in chunks
    STREAMING = "streaming"              # Stream processing with minimal memory
    DISTRIBUTED = "distributed"         # Use distributed computing
    MEMORY_MAPPED = "memory_mapped"      # Use memory mapping for large files


@dataclass
class LoadingConfig:
    """Configuration for optimized data loading"""
    strategy: LoadingStrategy = LoadingStrategy.DIRECT
    chunk_size: int = 10000
    max_memory_usage_pct: float = 70.0  # Max % of available memory to use
    enable_memory_mapping: bool = True
    enable_compression: bool = True
    use_categorical_optimization: bool = True
    dtype_optimization: bool = True
    enable_parallel_io: bool = True
    cache_stats: bool = True
    enable_progress_tracking: bool = True
    
    # Advanced options
    use_polars: bool = POLARS_AVAILABLE
    use_dask: bool = DASK_AVAILABLE
    memory_map_threshold_mb: float = 1000.0  # Use memory mapping for files > 1GB
    

@dataclass
class DatasetInfo:
    """Information about loaded dataset"""
    size_category: DatasetSize
    loading_strategy: LoadingStrategy
    estimated_memory_mb: float
    actual_memory_mb: float
    rows: int
    columns: int
    loading_time: float
    optimization_applied: List[str]
    dtype_info: Dict[str, str]
    missing_data_summary: Dict[str, Any]


class MemoryMonitor:
    """Monitor memory usage during data loading"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.memory_history = []
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_available_memory(self) -> float:
        """Get available system memory in MB"""
        return psutil.virtual_memory().available / 1024 / 1024
    
    def update_peak(self):
        """Update peak memory usage"""
        current = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, current)
        self.memory_history.append(current)
        return current
    
    def get_memory_pressure(self) -> float:
        """Get memory pressure as percentage of available memory"""
        used = self.get_memory_usage() - self.initial_memory
        available = self.get_available_memory()
        return (used / available) * 100 if available > 0 else 100.0
    
    def should_trigger_gc(self, threshold_pct: float = 80.0) -> bool:
        """Check if garbage collection should be triggered"""
        return self.get_memory_pressure() > threshold_pct


class OptimizedDataLoader:
    """
    Intelligent data loader that adapts strategy based on dataset size and system resources
    """
    
    def __init__(self, config: Optional[LoadingConfig] = None):
        self.config = config or LoadingConfig()
        self.logger = logging.getLogger(__name__)
        self.memory_monitor = MemoryMonitor()
        self._lock = threading.RLock()
        
        # Initialize memory-aware processor if available
        self.memory_processor = None
        if MEMORY_PROCESSOR_AVAILABLE:
            try:
                self.memory_processor = create_memory_aware_processor()
                self.logger.info("Memory-aware processor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize memory processor: {e}")
        
        # Cache for dataset statistics
        self._stats_cache = {} if self.config.cache_stats else None
        
        # Supported file formats and their optimized readers
        self._readers = {
            '.csv': self._read_csv_optimized,
            '.xlsx': self._read_excel_optimized,
            '.xls': self._read_excel_optimized,
            '.json': self._read_json_optimized,
            '.parquet': self._read_parquet_optimized,
            '.feather': self._read_feather_optimized,
        }
        
        self.logger.info(f"OptimizedDataLoader initialized with strategy: {self.config.strategy}")
    
    def estimate_file_complexity(self, file_path: str) -> Tuple[DatasetSize, float]:
        """
        Estimate dataset size and memory requirements without full loading
        
        Returns:
            Tuple of (size_category, estimated_memory_mb)
        """
        file_path = Path(file_path)
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        
        # Check cache first
        cache_key = f"{file_path}_{file_path.stat().st_mtime}"
        if self._stats_cache and cache_key in self._stats_cache:
            cached = self._stats_cache[cache_key]
            return cached['size_category'], cached['estimated_memory_mb']
        
        # Estimate based on file extension and size
        ext = file_path.suffix.lower()
        
        if ext == '.csv':
            estimated_rows, estimated_memory = self._estimate_csv_size(file_path, file_size_mb)
        elif ext in ['.xlsx', '.xls']:
            estimated_rows, estimated_memory = self._estimate_excel_size(file_path, file_size_mb)
        elif ext == '.json':
            estimated_rows, estimated_memory = self._estimate_json_size(file_path, file_size_mb)
        elif ext == '.parquet':
            estimated_rows, estimated_memory = self._estimate_parquet_size(file_path, file_size_mb)
        else:
            # Conservative estimate
            estimated_rows = file_size_mb * 1000  # ~1000 rows per MB
            estimated_memory = file_size_mb * 3   # 3x file size in memory
        
        # Determine size category
        size_category = self._categorize_dataset_size(estimated_rows)
        
        # Cache the result
        if self._stats_cache:
            self._stats_cache[cache_key] = {
                'size_category': size_category,
                'estimated_memory_mb': estimated_memory,
                'estimated_rows': estimated_rows
            }
        
        return size_category, estimated_memory
    
    def _estimate_csv_size(self, file_path: Path, file_size_mb: float) -> Tuple[int, float]:
        """Estimate CSV file size by sampling"""
        try:
            # Sample first 1000 lines to estimate structure
            sample_df = pd.read_csv(file_path, nrows=1000)
            sample_memory_mb = sample_df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Estimate total based on file size ratio
            lines_sampled = len(sample_df)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Count total lines (approximate)
                total_lines = sum(1 for _ in f)
            
            scale_factor = total_lines / lines_sampled if lines_sampled > 0 else 1
            estimated_memory = sample_memory_mb * scale_factor
            
            return total_lines, estimated_memory
            
        except Exception as e:
            self.logger.warning(f"Failed to sample CSV file: {e}")
            # Fallback to conservative estimate
            return int(file_size_mb * 500), file_size_mb * 2
    
    def _estimate_excel_size(self, file_path: Path, file_size_mb: float) -> Tuple[int, float]:
        """Estimate Excel file size"""
        try:
            # Excel files are typically more memory intensive
            xl_file = pd.ExcelFile(file_path)
            total_rows = 0
            
            for sheet_name in xl_file.sheet_names[:3]:  # Check max 3 sheets
                sheet_info = xl_file.book[sheet_name]
                if hasattr(sheet_info, 'max_row'):
                    total_rows += sheet_info.max_row
                else:
                    # Fallback: sample the sheet
                    sample = pd.read_excel(file_path, sheet_name=sheet_name, nrows=100)
                    total_rows += len(sample) * 10  # Rough estimate
            
            estimated_memory = file_size_mb * 4  # Excel typically expands 4x in memory
            return total_rows, estimated_memory
            
        except Exception as e:
            self.logger.warning(f"Failed to estimate Excel file: {e}")
            return int(file_size_mb * 200), file_size_mb * 4
    
    def _estimate_json_size(self, file_path: Path, file_size_mb: float) -> Tuple[int, float]:
        """Estimate JSON file size"""
        try:
            # Sample first few lines for JSON
            sample_df = pd.read_json(file_path, lines=True, nrows=1000)
            if len(sample_df) > 0:
                sample_memory_mb = sample_df.memory_usage(deep=True).sum() / 1024 / 1024
                scale_factor = file_size_mb / (sample_memory_mb if sample_memory_mb > 0 else 1)
                estimated_rows = len(sample_df) * scale_factor
                estimated_memory = file_size_mb * 2  # JSON typically 2x in memory
                return int(estimated_rows), estimated_memory
        except:
            pass
        
        # Fallback
        return int(file_size_mb * 100), file_size_mb * 2
    
    def _estimate_parquet_size(self, file_path: Path, file_size_mb: float) -> Tuple[int, float]:
        """Estimate Parquet file size"""
        try:
            # Parquet files are already optimized
            parquet_file = pd.read_parquet(file_path, engine='pyarrow')
            estimated_memory = parquet_file.memory_usage(deep=True).sum() / 1024 / 1024
            return len(parquet_file), estimated_memory
        except:
            # Fallback - Parquet is typically very efficient
            return int(file_size_mb * 1000), file_size_mb * 1.2
    
    def _categorize_dataset_size(self, estimated_rows: int) -> DatasetSize:
        """Categorize dataset size based on row count"""
        if estimated_rows < 10_000:
            return DatasetSize.TINY
        elif estimated_rows < 100_000:
            return DatasetSize.SMALL
        elif estimated_rows < 1_000_000:
            return DatasetSize.MEDIUM
        elif estimated_rows < 10_000_000:
            return DatasetSize.LARGE
        else:
            return DatasetSize.HUGE
    
    def select_loading_strategy(self, size_category: DatasetSize, 
                              estimated_memory_mb: float) -> LoadingStrategy:
        """Select optimal loading strategy based on dataset characteristics"""
        available_memory_mb = self.memory_monitor.get_available_memory()
        memory_threshold = available_memory_mb * (self.config.max_memory_usage_pct / 100)
        
        # Force strategy if explicitly set and not DIRECT
        if self.config.strategy != LoadingStrategy.DIRECT:
            return self.config.strategy
        
        # Strategy selection logic
        if size_category in [DatasetSize.TINY, DatasetSize.SMALL]:
            return LoadingStrategy.DIRECT
        
        elif size_category == DatasetSize.MEDIUM:
            if estimated_memory_mb <= memory_threshold:
                return LoadingStrategy.DIRECT
            else:
                return LoadingStrategy.CHUNKED
        
        elif size_category == DatasetSize.LARGE:
            if estimated_memory_mb <= memory_threshold * 0.5:  # Conservative for large
                return LoadingStrategy.DIRECT
            elif self.config.use_dask and DASK_AVAILABLE:
                return LoadingStrategy.DISTRIBUTED
            else:
                return LoadingStrategy.STREAMING
        
        else:  # HUGE
            if self.config.use_dask and DASK_AVAILABLE:
                return LoadingStrategy.DISTRIBUTED
            elif estimated_memory_mb > self.config.memory_map_threshold_mb:
                return LoadingStrategy.MEMORY_MAPPED
            else:
                return LoadingStrategy.STREAMING
    
    def load_data(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
        """
        Load data using the optimal strategy
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for specific readers
            
        Returns:
            Tuple of (dataframe, dataset_info)
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        if ext not in self._readers:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Estimate complexity
        size_category, estimated_memory_mb = self.estimate_file_complexity(str(file_path))
        
        # Select strategy
        strategy = self.select_loading_strategy(size_category, estimated_memory_mb)
        
        self.logger.info(f"Loading {file_path.name} using {strategy.value} strategy "
                        f"(estimated: {size_category.value}, {estimated_memory_mb:.1f}MB)")
        
        # Load data using selected strategy
        initial_memory = self.memory_monitor.get_memory_usage()
        
        try:
            df, optimization_applied = self._load_with_strategy(
                file_path, strategy, estimated_memory_mb, **kwargs
            )
            
            # Monitor memory usage
            final_memory = self.memory_monitor.update_peak()
            actual_memory_mb = final_memory - initial_memory
            
            # Generate dataset info
            dataset_info = DatasetInfo(
                size_category=size_category,
                loading_strategy=strategy,
                estimated_memory_mb=estimated_memory_mb,
                actual_memory_mb=actual_memory_mb,
                rows=len(df),
                columns=len(df.columns),
                loading_time=time.time() - start_time,
                optimization_applied=optimization_applied,
                dtype_info={col: str(dtype) for col, dtype in df.dtypes.items()},
                missing_data_summary=self._analyze_missing_data(df)
            )
            
            self.logger.info(f"Data loaded successfully: {dataset_info.rows:,} rows, "
                           f"{dataset_info.columns} columns, "
                           f"{dataset_info.actual_memory_mb:.1f}MB, "
                           f"{dataset_info.loading_time:.2f}s")
            
            # Apply memory optimization if memory processor is available and dataset is large
            if (self.memory_processor and 
                size_category in [DatasetSize.LARGE, DatasetSize.HUGE] and
                actual_memory_mb > 100):  # Only optimize if > 100MB
                
                self.logger.info("Applying memory optimization to large dataset")
                original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
                df = self.memory_processor.optimize_dataframe_memory(df)
                optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
                
                # Update dataset info with optimization results
                dataset_info.optimization_applied.append("memory_optimization")
                dataset_info.actual_memory_mb = optimized_memory
                
                self.logger.info(f"Memory optimization completed: "
                               f"{original_memory:.1f}MB -> {optimized_memory:.1f}MB "
                               f"({((original_memory - optimized_memory) / original_memory * 100):.1f}% reduction)")
            
            return df, dataset_info
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def _load_with_strategy(self, file_path: Path, strategy: LoadingStrategy, 
                          estimated_memory_mb: float, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        """Load data using the specified strategy"""
        optimization_applied = []
        
        if strategy == LoadingStrategy.DIRECT:
            df = self._readers[file_path.suffix.lower()](file_path, **kwargs)
            
        elif strategy == LoadingStrategy.CHUNKED:
            df = self._load_chunked(file_path, **kwargs)
            optimization_applied.append("chunked_loading")
            
        elif strategy == LoadingStrategy.STREAMING:
            df = self._load_streaming(file_path, **kwargs)
            optimization_applied.append("streaming_loading")
            
        elif strategy == LoadingStrategy.DISTRIBUTED:
            df = self._load_distributed(file_path, **kwargs)
            optimization_applied.append("distributed_loading")
            
        elif strategy == LoadingStrategy.MEMORY_MAPPED:
            df = self._load_memory_mapped(file_path, **kwargs)
            optimization_applied.append("memory_mapped_loading")
        
        # Apply post-loading optimizations
        df, post_opts = self._apply_post_loading_optimizations(df)
        optimization_applied.extend(post_opts)
        
        return df, optimization_applied
    
    def _apply_post_loading_optimizations(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Apply optimizations after loading data"""
        optimizations = []
        
        if self.config.dtype_optimization:
            df = self._optimize_dtypes(df)
            optimizations.append("dtype_optimization")
        
        if self.config.use_categorical_optimization:
            df = self._optimize_categorical(df)
            optimizations.append("categorical_optimization")
        
        # Trigger garbage collection if memory pressure is high
        if self.memory_monitor.should_trigger_gc():
            gc.collect()
            optimizations.append("garbage_collection")
        
        return df, optimizations
        
        if self.config.dtype_optimization:
            df = self._optimize_dtypes(df)
            optimizations.append("dtype_optimization")
        
        if self.config.use_categorical_optimization:
            df = self._optimize_categorical(df)
            optimizations.append("categorical_optimization")
        
        # Trigger garbage collection if memory pressure is high
        if self.memory_monitor.should_trigger_gc():
            gc.collect()
            optimizations.append("garbage_collection")
        
        return df, optimizations
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage"""
        original_memory = df.memory_usage(deep=True).sum()
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                except:
                    try:
                        df[col] = pd.to_numeric(df[col], downcast='float')
                    except:
                        pass  # Keep as object
            
            elif 'int' in str(col_type):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            elif 'float' in str(col_type):
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        new_memory = df.memory_usage(deep=True).sum()
        reduction_pct = (1 - new_memory / original_memory) * 100
        
        if reduction_pct > 5:  # Only log if significant reduction
            self.logger.info(f"Dtype optimization reduced memory by {reduction_pct:.1f}%")
        
        return df
    
    def _optimize_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert suitable string columns to categorical"""
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        return {
            'total_missing': missing_count.sum(),
            'missing_percentage': missing_pct.sum() / len(df.columns),
            'columns_with_missing': missing_count[missing_count > 0].to_dict(),
            'missing_percentages': missing_pct[missing_pct > 0].to_dict()
        }
    
    # Reader implementations
    def _read_csv_optimized(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Optimized CSV reader"""
        # Use efficient defaults for pandas
        read_kwargs = {
            'low_memory': False,
            'engine': 'c',  # Use C engine for speed
            **kwargs
        }
        
        if self.config.use_polars and POLARS_AVAILABLE:
            try:
                # Use Polars for faster CSV reading, then convert to pandas
                pl_df = pl.read_csv(file_path)
                return pl_df.to_pandas()
            except Exception as e:
                self.logger.warning(f"Polars CSV read failed, falling back to pandas: {e}")
        
        return pd.read_csv(file_path, **read_kwargs)
    
    def _read_excel_optimized(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Optimized Excel reader"""
        read_kwargs = {
            'engine': 'openpyxl' if file_path.suffix == '.xlsx' else 'xlrd',
            **kwargs
        }
        return pd.read_excel(file_path, **read_kwargs)
    
    def _read_json_optimized(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Optimized JSON reader"""
        read_kwargs = {
            'lines': True,  # Assume JSONL format for better performance
            **kwargs
        }
        return pd.read_json(file_path, **read_kwargs)
    
    def _read_parquet_optimized(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Optimized Parquet reader"""
        read_kwargs = {
            'engine': 'pyarrow',  # PyArrow is typically faster
            **kwargs
        }
        return pd.read_parquet(file_path, **read_kwargs)
    
    def _read_feather_optimized(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Optimized Feather reader"""
        return pd.read_feather(file_path, **kwargs)
    
    def _load_chunked(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data in chunks and combine"""
        chunks = []
        chunk_size = self.config.chunk_size
        
        ext = file_path.suffix.lower()
        
        if ext == '.csv':
            reader = pd.read_csv(file_path, chunksize=chunk_size, **kwargs)
        else:
            # For non-CSV files, simulate chunking by loading full data
            # and splitting it (not ideal but works for medium datasets)
            full_df = self._readers[ext](file_path, **kwargs)
            for i in range(0, len(full_df), chunk_size):
                chunks.append(full_df.iloc[i:i+chunk_size])
            return pd.concat(chunks, ignore_index=True)
        
        for chunk in reader:
            # Apply optimizations to each chunk
            chunk, _ = self._apply_post_loading_optimizations(chunk)
            chunks.append(chunk)
            
            # Monitor memory and trigger GC if needed
            if self.memory_monitor.should_trigger_gc():
                gc.collect()
        
        return pd.concat(chunks, ignore_index=True)
    
    def _load_streaming(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data using streaming approach with minimal memory footprint"""
        # This is a simplified streaming loader
        # In practice, this would return a generator or iterator
        return self._load_chunked(file_path, **kwargs)
    
    def _load_distributed(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data using distributed computing (Dask)"""
        if not DASK_AVAILABLE:
            self.logger.warning("Dask not available, falling back to chunked loading")
            return self._load_chunked(file_path, **kwargs)
        
        ext = file_path.suffix.lower()
        
        try:
            if ext == '.csv':
                ddf = dd.read_csv(file_path, **kwargs)
            elif ext == '.parquet':
                ddf = dd.read_parquet(file_path, **kwargs)
            else:
                # Fallback to chunked for unsupported formats
                return self._load_chunked(file_path, **kwargs)
            
            # Convert back to pandas DataFrame
            return ddf.compute()
            
        except Exception as e:
            self.logger.warning(f"Distributed loading failed, falling back to chunked: {e}")
            return self._load_chunked(file_path, **kwargs)
    
    def _load_memory_mapped(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data using memory mapping (for very large files)"""
        # This is a placeholder - actual implementation would depend on file format
        # For now, fall back to streaming
        self.logger.info("Memory mapping not fully implemented, using streaming")
        return self._load_streaming(file_path, **kwargs)


# Integration functions for existing codebase
def create_optimized_loader(max_memory_pct: float = 70.0, 
                          chunk_size: int = 10000,
                          enable_distributed: bool = True) -> OptimizedDataLoader:
    """Create an optimized data loader with sensible defaults"""
    config = LoadingConfig(
        chunk_size=chunk_size,
        max_memory_usage_pct=max_memory_pct,
        use_dask=enable_distributed and DASK_AVAILABLE,
        use_polars=POLARS_AVAILABLE
    )
    return OptimizedDataLoader(config)


def load_data_optimized(file_path: str, 
                       max_memory_pct: float = 70.0,
                       **kwargs) -> Tuple[pd.DataFrame, DatasetInfo]:
    """
    Convenience function to load data with automatic optimization
    
    Args:
        file_path: Path to the data file
        max_memory_pct: Maximum percentage of available memory to use
        **kwargs: Additional arguments passed to the file reader
        
    Returns:
        Tuple of (dataframe, dataset_info)
    """
    loader = create_optimized_loader(max_memory_pct=max_memory_pct)
    return loader.load_data(file_path, **kwargs)
