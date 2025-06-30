"""
Streaming Data Processing Pipeline
Provides efficient streaming data processing with backpressure control and memory optimization.
"""

import logging
import time
import threading
import queue
import gc
from typing import Dict, List, Any, Optional, Union, Callable, Iterator, Generator
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from contextlib import contextmanager
import psutil
import weakref

try:
    import dask.dataframe as dd
    from dask.distributed import Client
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


@dataclass
class StreamingStats:
    """Statistics for streaming pipeline performance."""
    total_chunks_processed: int
    total_rows_processed: int
    total_processing_time: float
    average_chunk_size: float
    memory_peak_mb: float
    throughput_rows_per_second: float
    backpressure_events: int
    errors_encountered: int


class BackpressureController:
    """Controls backpressure in streaming pipeline."""
    
    def __init__(self,
                 max_queue_size: int = 100,
                 memory_threshold_mb: float = 1000,
                 adaptive_batching: bool = True):
        """
        Initialize backpressure controller.
        
        Args:
            max_queue_size: Maximum number of items in processing queue
            memory_threshold_mb: Memory threshold for backpressure activation
            adaptive_batching: Whether to adapt batch sizes based on memory
        """
        self.max_queue_size = max_queue_size
        self.memory_threshold_mb = memory_threshold_mb
        self.adaptive_batching = adaptive_batching
        
        self.processing_queue = queue.Queue(maxsize=max_queue_size)
        self.backpressure_events = 0
        self.current_batch_size = 1000  # Default batch size
        self.lock = threading.RLock()
        
    def should_apply_backpressure(self) -> bool:
        """Check if backpressure should be applied."""
        # Check queue size
        if self.processing_queue.qsize() >= self.max_queue_size * 0.8:
            return True
        
        # Check memory usage
        try:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            if current_memory > self.memory_threshold_mb:
                return True
        except:
            pass
        
        return False
    
    def wait_for_capacity(self, timeout: Optional[float] = None):
        """Wait until there's capacity in the pipeline."""
        start_time = time.time()
        while self.should_apply_backpressure():
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Backpressure timeout exceeded")
            
            with self.lock:
                self.backpressure_events += 1
            
            time.sleep(0.1)  # Brief pause
    
    def adapt_batch_size(self):
        """Adapt batch size based on current conditions."""
        if not self.adaptive_batching:
            return
        
        try:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_ratio = current_memory / self.memory_threshold_mb
            
            if memory_ratio > 0.9:
                # High memory usage, reduce batch size
                self.current_batch_size = max(100, int(self.current_batch_size * 0.8))
            elif memory_ratio < 0.5:
                # Low memory usage, increase batch size
                self.current_batch_size = min(10000, int(self.current_batch_size * 1.2))
        except:
            pass


class ChunkProcessor:
    """Processes data chunks with various optimization strategies."""
    
    def __init__(self,
                 processing_function: Callable,
                 enable_parallel: bool = True,
                 num_workers: Optional[int] = None,
                 enable_caching: bool = True):
        """
        Initialize chunk processor.
        
        Args:
            processing_function: Function to process each chunk
            enable_parallel: Whether to enable parallel processing
            num_workers: Number of worker threads (auto-detect if None)
            enable_caching: Whether to enable result caching
        """
        self.processing_function = processing_function
        self.enable_parallel = enable_parallel
        self.num_workers = num_workers or min(8, (psutil.cpu_count() or 1))
        self.enable_caching = enable_caching
        
        self.logger = logging.getLogger(__name__)
        
        # Thread pool for parallel processing
        if self.enable_parallel:
            from concurrent.futures import ThreadPoolExecutor
            self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        else:
            self.thread_pool = None
        
        # Simple cache for repeated chunks
        if self.enable_caching:
            self.chunk_cache = weakref.WeakValueDictionary()
        else:
            self.chunk_cache = None
    
    def process_chunk(self, chunk: Any, chunk_id: Optional[str] = None) -> Any:
        """Process a single chunk of data."""
        # Check cache first
        if self.enable_caching and chunk_id and chunk_id in self.chunk_cache:
            return self.chunk_cache[chunk_id]
        
        # Process the chunk
        try:
            result = self.processing_function(chunk)
            
            # Cache result if enabled
            if self.enable_caching and chunk_id:
                self.chunk_cache[chunk_id] = result
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            raise
    
    def process_chunks_parallel(self, chunks: Iterator[Any]) -> Generator[Any, None, None]:
        """Process chunks in parallel."""
        if not self.thread_pool:
            # Fallback to sequential processing
            for chunk in chunks:
                yield self.process_chunk(chunk)
            return
        
        # Submit chunks to thread pool
        futures = []
        chunk_buffer = []
        
        for i, chunk in enumerate(chunks):
            future = self.thread_pool.submit(self.process_chunk, chunk, f"chunk_{i}")
            futures.append(future)
            chunk_buffer.append(chunk)
            
            # Process in batches to avoid memory buildup
            if len(futures) >= self.num_workers * 2:
                # Yield completed results
                for future in futures:
                    yield future.result()
                futures.clear()
                chunk_buffer.clear()
        
        # Process remaining futures
        for future in futures:
            yield future.result()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


class StreamingDataPipeline:
    """
    High-performance streaming data processing pipeline.
    
    Features:
    - Memory-efficient chunk-based processing
    - Adaptive batching based on memory pressure
    - Backpressure control for stable operation
    - Parallel processing with thread pools
    - Integration with Dask, Ray, and Spark for distributed processing
    - Real-time performance monitoring
    - Automatic garbage collection and memory management
    """
    
    def __init__(self,
                 chunk_size: int = 1000,
                 max_memory_mb: float = 1000,
                 enable_parallel: bool = True,
                 num_workers: Optional[int] = None,
                 enable_distributed: bool = False,
                 distributed_backend: str = 'dask',
                 enable_monitoring: bool = True):
        """
        Initialize streaming data pipeline.
        
        Args:
            chunk_size: Default chunk size for processing
            max_memory_mb: Maximum memory usage before applying backpressure
            enable_parallel: Whether to enable parallel processing
            num_workers: Number of worker threads
            enable_distributed: Whether to enable distributed processing
            distributed_backend: Distributed processing backend ('dask', 'ray', 'spark')
            enable_monitoring: Whether to enable performance monitoring
        """
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.enable_parallel = enable_parallel
        self.num_workers = num_workers or min(8, (psutil.cpu_count() or 1))
        self.enable_distributed = enable_distributed
        self.distributed_backend = distributed_backend
        self.enable_monitoring = enable_monitoring
        
        self.logger = logging.getLogger(__name__)
        
        # Pipeline components
        self.backpressure_controller = BackpressureController(
            max_queue_size=100,
            memory_threshold_mb=max_memory_mb,
            adaptive_batching=True
        )
        
        # Performance monitoring
        self.stats = StreamingStats(
            total_chunks_processed=0,
            total_rows_processed=0,
            total_processing_time=0.0,
            average_chunk_size=0.0,
            memory_peak_mb=0.0,
            throughput_rows_per_second=0.0,
            backpressure_events=0,
            errors_encountered=0
        )
        
        # State management
        self.is_running = False
        self.lock = threading.RLock()
        
        # Distributed processing setup
        self.distributed_client = None
        if enable_distributed:
            self._setup_distributed_processing()
        
        self.logger.info(f"Streaming pipeline initialized with chunk_size={chunk_size}, "
                        f"parallel={enable_parallel}, distributed={enable_distributed}")
    
    def _setup_distributed_processing(self):
        """Setup distributed processing backend."""
        if self.distributed_backend == 'dask' and DASK_AVAILABLE:
            try:
                self.distributed_client = Client(processes=False, threads_per_worker=2)
                self.logger.info("Dask client initialized for distributed processing")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Dask client: {str(e)}")
        
        elif self.distributed_backend == 'ray' and RAY_AVAILABLE:
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self.logger.info("Ray initialized for distributed processing")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Ray: {str(e)}")
        
        elif self.distributed_backend == 'spark' and SPARK_AVAILABLE:
            try:
                self.distributed_client = SparkSession.builder \
                    .appName("StreamingPipeline") \
                    .getOrCreate()
                self.logger.info("Spark session initialized for distributed processing")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Spark: {str(e)}")
    
    def process_dataframe_stream(self,
                                df: pd.DataFrame,
                                processing_function: Callable,
                                output_path: Optional[str] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Process a DataFrame in streaming fashion.
        
        Args:
            df: Input DataFrame
            processing_function: Function to apply to each chunk
            output_path: Optional path to save processed chunks
            
        Yields:
            Processed DataFrame chunks
        """
        start_time = time.perf_counter()
        self.is_running = True
        
        try:
            # Create chunk processor
            chunk_processor = ChunkProcessor(
                processing_function=processing_function,
                enable_parallel=self.enable_parallel,
                num_workers=self.num_workers
            )
            
            # Process DataFrame in chunks
            total_rows = len(df)
            chunks_processed = 0
            
            for i in range(0, total_rows, self.chunk_size):
                if not self.is_running:
                    break
                
                # Apply backpressure if needed
                self.backpressure_controller.wait_for_capacity(timeout=30)
                
                # Extract chunk
                end_idx = min(i + self.chunk_size, total_rows)
                chunk = df.iloc[i:end_idx].copy()
                
                # Monitor memory usage
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                self.stats.memory_peak_mb = max(self.stats.memory_peak_mb, current_memory)
                
                # Process chunk
                try:
                    processed_chunk = chunk_processor.process_chunk(chunk, f"df_chunk_{chunks_processed}")
                    
                    # Update statistics
                    with self.lock:
                        self.stats.total_chunks_processed += 1
                        self.stats.total_rows_processed += len(chunk)
                        chunks_processed += 1
                    
                    # Save to file if specified
                    if output_path:
                        chunk_file = Path(output_path) / f"chunk_{chunks_processed}.parquet"
                        chunk_file.parent.mkdir(parents=True, exist_ok=True)
                        processed_chunk.to_parquet(chunk_file)
                    
                    yield processed_chunk
                    
                except Exception as e:
                    with self.lock:
                        self.stats.errors_encountered += 1
                    self.logger.error(f"Error processing chunk {chunks_processed}: {str(e)}")
                    raise
                
                # Adapt chunk size based on performance
                self.backpressure_controller.adapt_batch_size()
                self.chunk_size = self.backpressure_controller.current_batch_size
                
                # Periodic garbage collection
                if chunks_processed % 10 == 0:
                    gc.collect()
            
            # Cleanup
            chunk_processor.cleanup()
            
        finally:
            self.is_running = False
            
            # Update final statistics
            total_time = time.perf_counter() - start_time
            with self.lock:
                self.stats.total_processing_time = total_time
                if self.stats.total_chunks_processed > 0:
                    self.stats.average_chunk_size = self.stats.total_rows_processed / self.stats.total_chunks_processed
                if total_time > 0:
                    self.stats.throughput_rows_per_second = self.stats.total_rows_processed / total_time
                self.stats.backpressure_events = self.backpressure_controller.backpressure_events
    
    def process_file_stream(self,
                           file_path: str,
                           processing_function: Callable,
                           file_format: str = 'auto',
                           output_path: Optional[str] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Process a file in streaming fashion.
        
        Args:
            file_path: Path to input file
            processing_function: Function to apply to each chunk
            file_format: File format ('csv', 'parquet', 'json', 'auto')
            output_path: Optional path to save processed chunks
            
        Yields:
            Processed DataFrame chunks
        """
        file_path = Path(file_path)
        
        # Auto-detect file format
        if file_format == 'auto':
            file_format = file_path.suffix.lower().lstrip('.')
        
        # Read file in chunks
        if file_format == 'csv':
            chunk_reader = pd.read_csv(file_path, chunksize=self.chunk_size)
        elif file_format == 'parquet':
            # For parquet, we need to read the entire file and then chunk it
            df = pd.read_parquet(file_path)
            yield from self.process_dataframe_stream(df, processing_function, output_path)
            return
        elif file_format == 'json':
            chunk_reader = pd.read_json(file_path, lines=True, chunksize=self.chunk_size)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Process chunks
        start_time = time.perf_counter()
        self.is_running = True
        
        try:
            chunk_processor = ChunkProcessor(
                processing_function=processing_function,
                enable_parallel=self.enable_parallel,
                num_workers=self.num_workers
            )
            
            chunks_processed = 0
            
            for chunk in chunk_reader:
                if not self.is_running:
                    break
                
                # Apply backpressure if needed
                self.backpressure_controller.wait_for_capacity(timeout=30)
                
                # Monitor memory
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                self.stats.memory_peak_mb = max(self.stats.memory_peak_mb, current_memory)
                
                # Process chunk
                try:
                    processed_chunk = chunk_processor.process_chunk(chunk, f"file_chunk_{chunks_processed}")
                    
                    # Update statistics
                    with self.lock:
                        self.stats.total_chunks_processed += 1
                        self.stats.total_rows_processed += len(chunk)
                        chunks_processed += 1
                    
                    # Save to file if specified
                    if output_path:
                        chunk_file = Path(output_path) / f"chunk_{chunks_processed}.parquet"
                        chunk_file.parent.mkdir(parents=True, exist_ok=True)
                        processed_chunk.to_parquet(chunk_file)
                    
                    yield processed_chunk
                    
                except Exception as e:
                    with self.lock:
                        self.stats.errors_encountered += 1
                    self.logger.error(f"Error processing chunk {chunks_processed}: {str(e)}")
                    raise
                
                # Periodic cleanup
                if chunks_processed % 10 == 0:
                    gc.collect()
            
            chunk_processor.cleanup()
            
        finally:
            self.is_running = False
            
            # Update final statistics
            total_time = time.perf_counter() - start_time
            with self.lock:
                self.stats.total_processing_time = total_time
                if self.stats.total_chunks_processed > 0:
                    self.stats.average_chunk_size = self.stats.total_rows_processed / self.stats.total_chunks_processed
                if total_time > 0:
                    self.stats.throughput_rows_per_second = self.stats.total_rows_processed / total_time
                self.stats.backpressure_events = self.backpressure_controller.backpressure_events
    
    def process_distributed_dataframe(self,
                                     df: pd.DataFrame,
                                     processing_function: Callable) -> pd.DataFrame:
        """
        Process DataFrame using distributed computing.
        
        Args:
            df: Input DataFrame
            processing_function: Function to apply
            
        Returns:
            Processed DataFrame
        """
        if not self.enable_distributed or not self.distributed_client:
            self.logger.warning("Distributed processing not available, falling back to local processing")
            # Process locally
            chunks = list(self.process_dataframe_stream(df, processing_function))
            return pd.concat(chunks, ignore_index=True)
        
        if self.distributed_backend == 'dask' and DASK_AVAILABLE:
            # Convert to Dask DataFrame
            ddf = dd.from_pandas(df, npartitions=self.num_workers)
            
            # Apply function to each partition
            result_ddf = ddf.map_partitions(processing_function)
            
            # Compute result
            return result_ddf.compute()
        
        elif self.distributed_backend == 'ray' and RAY_AVAILABLE:
            # Split DataFrame into chunks
            chunks = np.array_split(df, self.num_workers)
            
            # Create remote function
            @ray.remote
            def process_chunk_remote(chunk):
                return processing_function(chunk)
            
            # Process chunks in parallel
            futures = [process_chunk_remote.remote(chunk) for chunk in chunks]
            results = ray.get(futures)
            
            # Combine results
            return pd.concat(results, ignore_index=True)
        
        elif self.distributed_backend == 'spark' and SPARK_AVAILABLE:
            # Convert to Spark DataFrame
            spark_df = self.distributed_client.createDataFrame(df)
            
            # This would require UDF creation for the processing function
            # Simplified implementation - convert back to Pandas for processing
            self.logger.warning("Spark distributed processing not fully implemented, using local processing")
            chunks = list(self.process_dataframe_stream(df, processing_function))
            return pd.concat(chunks, ignore_index=True)
        
        else:
            raise ValueError(f"Unsupported distributed backend: {self.distributed_backend}")
    
    @contextmanager
    def monitoring_context(self):
        """Context manager for performance monitoring."""
        if not self.enable_monitoring:
            yield
            return
        
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            self.logger.info(f"Processing completed in {end_time - start_time:.2f}s")
            self.logger.info(f"Memory usage: {start_memory:.1f}MB -> {end_memory:.1f}MB")
            if self.stats.total_rows_processed > 0:
                self.logger.info(f"Throughput: {self.stats.throughput_rows_per_second:.0f} rows/second")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.lock:
            return {
                'total_chunks_processed': self.stats.total_chunks_processed,
                'total_rows_processed': self.stats.total_rows_processed,
                'total_processing_time': self.stats.total_processing_time,
                'average_chunk_size': self.stats.average_chunk_size,
                'memory_peak_mb': self.stats.memory_peak_mb,
                'throughput_rows_per_second': self.stats.throughput_rows_per_second,
                'backpressure_events': self.stats.backpressure_events,
                'errors_encountered': self.stats.errors_encountered,
                'current_chunk_size': self.chunk_size
            }
    
    def stop_processing(self):
        """Stop the streaming pipeline."""
        self.is_running = False
        self.logger.info("Streaming pipeline stopped")
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_processing()
        
        if self.distributed_client:
            if self.distributed_backend == 'dask':
                self.distributed_client.close()
            elif self.distributed_backend == 'spark':
                self.distributed_client.stop()
            elif self.distributed_backend == 'ray':
                ray.shutdown()


# Utility functions for common streaming operations
def create_streaming_preprocessor(scaler_type: str = 'standard') -> Callable:
    """Create a streaming-compatible preprocessor."""
    if scaler_type == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")
    
    # Fit on first chunk, transform subsequent chunks
    is_fitted = False
    
    def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        nonlocal is_fitted
        
        numeric_columns = chunk.select_dtypes(include=[np.number]).columns
        chunk_processed = chunk.copy()
        
        if not is_fitted:
            # Fit on first chunk
            chunk_processed[numeric_columns] = scaler.fit_transform(chunk[numeric_columns])
            is_fitted = True
        else:
            # Transform subsequent chunks
            chunk_processed[numeric_columns] = scaler.transform(chunk[numeric_columns])
        
        return chunk_processed
    
    return preprocess_chunk


def create_streaming_feature_selector(n_features: int = 10) -> Callable:
    """Create a streaming-compatible feature selector."""
    from sklearn.feature_selection import SelectKBest, f_classif
    
    selector = SelectKBest(score_func=f_classif, k=n_features)
    is_fitted = False
    
    def select_features_chunk(chunk: pd.DataFrame, target_column: str = 'target') -> pd.DataFrame:
        nonlocal is_fitted
        
        if target_column not in chunk.columns:
            return chunk
        
        feature_columns = [col for col in chunk.columns if col != target_column]
        
        if not is_fitted:
            # Fit on first chunk
            X = chunk[feature_columns]
            y = chunk[target_column]
            selector.fit(X, y)
            is_fitted = True
        
        # Transform chunk
        X = chunk[feature_columns]
        X_selected = selector.transform(X)
        
        # Create new DataFrame with selected features
        selected_feature_names = [feature_columns[i] for i in selector.get_support(indices=True)]
        result_df = pd.DataFrame(X_selected, columns=selected_feature_names, index=chunk.index)
        result_df[target_column] = chunk[target_column]
        
        return result_df
    
    return select_features_chunk


# Global streaming pipeline instance
_global_streaming_pipeline = None

def get_global_streaming_pipeline() -> StreamingDataPipeline:
    """Get or create the global streaming pipeline."""
    global _global_streaming_pipeline
    if _global_streaming_pipeline is None:
        _global_streaming_pipeline = StreamingDataPipeline()
    return _global_streaming_pipeline
