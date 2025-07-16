# Streaming Pipeline (`modules/engine/streaming_pipeline.py`)

## Overview

The Streaming Pipeline module provides a high-performance, memory-efficient data processing pipeline designed for handling large datasets that don't fit in memory. It features adaptive batching, backpressure control, parallel processing, and integration with distributed computing frameworks.

## Features

- **Memory-Efficient Processing**: Chunk-based processing for large datasets
- **Adaptive Batching**: Dynamic batch size adjustment based on memory pressure
- **Backpressure Control**: Prevents memory overflow during high-throughput processing
- **Parallel Processing**: Multi-threaded chunk processing for improved performance
- **Distributed Computing**: Integration with Dask, Ray, and Spark
- **Real-Time Monitoring**: Performance metrics and resource usage tracking
- **Automatic Memory Management**: Garbage collection and memory optimization

## Core Classes

### StreamingDataPipeline

Main pipeline class for streaming data processing:

```python
class StreamingDataPipeline:
    def __init__(
        self,
        chunk_size: int = 1000,
        max_memory_mb: float = 1000,
        enable_parallel: bool = True,
        num_workers: Optional[int] = None,
        enable_distributed: bool = False,
        distributed_backend: str = 'dask'
    )
```

### BackpressureController

Controls flow control and memory pressure:

```python
class BackpressureController:
    def __init__(
        self,
        max_queue_size: int = 100,
        memory_threshold_mb: float = 1000,
        adaptive_batching: bool = True
    )
```

### StreamingStats

Performance statistics tracking:

```python
@dataclass
class StreamingStats:
    total_chunks_processed: int
    total_rows_processed: int
    total_processing_time: float
    throughput_rows_per_second: float
    memory_peak_mb: float
    backpressure_events: int
```

## Usage Examples

### Basic Streaming Processing

```python
from modules.engine.streaming_pipeline import StreamingDataPipeline
import pandas as pd
import numpy as np
import time

# Create sample large dataset
def create_large_dataset(num_rows=100000):
    """Create a large dataset for streaming processing demo"""
    return pd.DataFrame({
        'feature_1': np.random.randn(num_rows),
        'feature_2': np.random.randn(num_rows),
        'feature_3': np.random.randn(num_rows),
        'category': np.random.choice(['A', 'B', 'C'], num_rows),
        'target': np.random.randint(0, 2, num_rows)
    })

# Define processing function
def data_preprocessing(chunk: pd.DataFrame) -> pd.DataFrame:
    """Example preprocessing function for each chunk"""
    # Simulate some processing time
    time.sleep(0.01)
    
    # Example transformations
    processed = chunk.copy()
    
    # Feature engineering
    processed['feature_interaction'] = processed['feature_1'] * processed['feature_2']
    processed['feature_sum'] = processed['feature_1'] + processed['feature_2'] + processed['feature_3']
    
    # Categorical encoding
    processed = pd.get_dummies(processed, columns=['category'], prefix='cat')
    
    # Normalization (simple example)
    numeric_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_interaction', 'feature_sum']
    for col in numeric_columns:
        mean_val = processed[col].mean()
        std_val = processed[col].std()
        if std_val > 0:
            processed[col] = (processed[col] - mean_val) / std_val
    
    return processed

# Initialize streaming pipeline
pipeline = StreamingDataPipeline(
    chunk_size=5000,        # Process 5000 rows at a time
    max_memory_mb=500,      # 500MB memory limit
    enable_parallel=True,   # Enable parallel processing
    num_workers=4          # Use 4 worker threads
)

# Create test dataset
print("Creating test dataset...")
df = create_large_dataset(100000)
print(f"Dataset created: {df.shape}")

# Process dataset in streaming fashion
print("Starting streaming processing...")
processed_chunks = []
start_time = time.time()

for i, processed_chunk in enumerate(pipeline.process_dataframe_stream(
    df=df,
    processing_function=data_preprocessing,
    output_path="processed_data"  # Save chunks to files
)):
    processed_chunks.append(processed_chunk)
    
    # Print progress every 5 chunks
    if (i + 1) % 5 == 0:
        print(f"Processed chunk {i + 1}, shape: {processed_chunk.shape}")

total_time = time.time() - start_time

# Combine results (if memory allows)
if len(processed_chunks) < 50:  # Only combine if reasonable number of chunks
    final_result = pd.concat(processed_chunks, ignore_index=True)
    print(f"Final result shape: {final_result.shape}")
else:
    print(f"Processed {len(processed_chunks)} chunks (too many to combine in memory)")

# Get performance statistics
stats = pipeline.get_stats()
print(f"\nPerformance Statistics:")
print(f"  Total processing time: {total_time:.2f}s")
print(f"  Chunks processed: {stats.total_chunks_processed}")
print(f"  Rows processed: {stats.total_rows_processed}")
print(f"  Throughput: {stats.throughput_rows_per_second:.1f} rows/sec")
print(f"  Peak memory usage: {stats.memory_peak_mb:.1f} MB")
print(f"  Backpressure events: {stats.backpressure_events}")
```

### Advanced Memory Management

```python
from modules.engine.streaming_pipeline import StreamingDataPipeline, BackpressureController
import pandas as pd
import numpy as np
import psutil
import time

class MemoryAwareStreamingPipeline:
    """Enhanced pipeline with advanced memory management"""
    
    def __init__(self, memory_limit_percentage=80):
        # Calculate memory limit based on system memory
        total_memory_mb = psutil.virtual_memory().total / 1024 / 1024
        memory_limit = total_memory_mb * (memory_limit_percentage / 100)
        
        self.pipeline = StreamingDataPipeline(
            chunk_size=2000,
            max_memory_mb=memory_limit,
            enable_parallel=True,
            num_workers=psutil.cpu_count()
        )
        
        print(f"Memory limit set to {memory_limit:.1f} MB ({memory_limit_percentage}% of {total_memory_mb:.1f} MB)")
    
    def process_with_memory_monitoring(self, df, processing_func):
        """Process data with real-time memory monitoring"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        memory_history = []
        chunk_count = 0
        
        for processed_chunk in self.pipeline.process_dataframe_stream(df, processing_func):
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_history.append(current_memory)
            chunk_count += 1
            
            # Print memory usage every 10 chunks
            if chunk_count % 10 == 0:
                print(f"Chunk {chunk_count}: Memory usage {current_memory:.1f} MB")
                
                # Force garbage collection if memory usage is high
                if current_memory > initial_memory * 2:
                    import gc
                    gc.collect()
                    after_gc_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    print(f"  After GC: {after_gc_memory:.1f} MB")
            
            yield processed_chunk
        
        print(f"Memory usage history: min={min(memory_history):.1f}, max={max(memory_history):.1f}, avg={sum(memory_history)/len(memory_history):.1f} MB")

# Complex processing function that uses significant memory
def memory_intensive_processing(chunk: pd.DataFrame) -> pd.DataFrame:
    """Processing function that creates temporary large objects"""
    
    # Create temporary large arrays for processing
    temp_data = np.random.randn(len(chunk), 100)  # Large temporary array
    
    # Simulate complex feature engineering
    processed = chunk.copy()
    
    # Add polynomial features (memory intensive)
    for i in range(3):
        for col in ['feature_1', 'feature_2', 'feature_3']:
            processed[f'{col}_poly_{i+2}'] = processed[col] ** (i + 2)
    
    # Add rolling statistics (using temporary data)
    window_size = min(50, len(chunk))
    if window_size > 1:
        for col in processed.select_dtypes(include=[np.number]).columns:
            processed[f'{col}_rolling_mean'] = processed[col].rolling(window=window_size, min_periods=1).mean()
    
    # Simulate computation with temporary array
    correlation_matrix = np.corrcoef(temp_data.T)
    
    # Add some features based on correlation
    if correlation_matrix.shape[0] > 10:
        processed['temp_feature'] = np.sum(correlation_matrix[:10, :10])
    
    # Clean up temporary data explicitly
    del temp_data, correlation_matrix
    
    return processed

# Test memory-aware processing
memory_pipeline = MemoryAwareStreamingPipeline(memory_limit_percentage=60)

# Create large dataset
large_df = create_large_dataset(50000)
print(f"Processing dataset with {len(large_df)} rows...")

# Process with memory monitoring
processed_count = 0
for processed_chunk in memory_pipeline.process_with_memory_monitoring(
    large_df, 
    memory_intensive_processing
):
    processed_count += 1
    
    # Only keep last few chunks in memory to simulate real streaming
    if processed_count > 5:
        # In real scenario, you'd save to disk or send to next stage
        pass

print(f"Successfully processed {processed_count} chunks")
```

### Distributed Processing with Dask

```python
from modules.engine.streaming_pipeline import StreamingDataPipeline
import pandas as pd
import numpy as np

try:
    import dask.dataframe as dd
    from dask.distributed import Client, as_completed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Dask not available, skipping distributed processing example")

if DASK_AVAILABLE:
    class DistributedStreamingPipeline:
        """Streaming pipeline with Dask distributed processing"""
        
        def __init__(self, scheduler_address=None):
            # Initialize Dask client
            if scheduler_address:
                self.client = Client(scheduler_address)
            else:
                self.client = Client(processes=False, threads_per_worker=2, n_workers=4)
            
            print(f"Dask client initialized: {self.client}")
            
            # Initialize streaming pipeline
            self.pipeline = StreamingDataPipeline(
                chunk_size=3000,
                enable_distributed=True,
                distributed_backend='dask',
                num_workers=4
            )
        
        def process_large_dataset_distributed(self, data_path_or_df, processing_func):
            """Process large dataset using Dask distributed computing"""
            
            if isinstance(data_path_or_df, str):
                # Load data as Dask DataFrame
                ddf = dd.read_parquet(data_path_or_df)
                print(f"Loaded Dask DataFrame with {len(ddf)} partitions")
            else:
                # Convert pandas DataFrame to Dask DataFrame
                ddf = dd.from_pandas(data_path_or_df, npartitions=8)
            
            # Process each partition
            def process_partition(partition):
                """Process a single partition"""
                return processing_func(partition)
            
            # Apply processing to all partitions
            processed_ddf = ddf.map_partitions(process_partition, meta=ddf)
            
            # Compute results partition by partition (streaming)
            partitions = processed_ddf.to_delayed()
            
            futures = self.client.compute(partitions)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    yield result
                except Exception as e:
                    print(f"Error processing partition: {e}")
        
        def cleanup(self):
            """Clean up Dask resources"""
            self.client.close()
    
    # Example usage of distributed processing
    def distributed_processing_example():
        """Example of distributed streaming processing"""
        
        # Create distributed pipeline
        dist_pipeline = DistributedStreamingPipeline()
        
        try:
            # Create test dataset
            large_df = create_large_dataset(200000)
            
            # Define processing function
            def advanced_feature_engineering(chunk: pd.DataFrame) -> pd.DataFrame:
                """Advanced feature engineering for distributed processing"""
                processed = chunk.copy()
                
                # Complex transformations
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    # Statistical features
                    processed[f'{col}_zscore'] = (processed[col] - processed[col].mean()) / processed[col].std()
                    processed[f'{col}_rank'] = processed[col].rank(pct=True)
                    
                    # Binning
                    processed[f'{col}_bin'] = pd.cut(processed[col], bins=5, labels=False)
                
                # Interaction features
                if len(numeric_cols) > 1:
                    for i, col1 in enumerate(numeric_cols[:3]):
                        for col2 in numeric_cols[i+1:4]:
                            processed[f'{col1}_{col2}_interaction'] = processed[col1] * processed[col2]
                
                return processed
            
            # Process with distributed pipeline
            print("Starting distributed processing...")
            processed_partitions = []
            
            for i, partition_result in enumerate(dist_pipeline.process_large_dataset_distributed(
                large_df, 
                advanced_feature_engineering
            )):
                processed_partitions.append(partition_result)
                print(f"Processed partition {i+1}, shape: {partition_result.shape}")
            
            print(f"Distributed processing completed: {len(processed_partitions)} partitions")
            
        finally:
            dist_pipeline.cleanup()
    
    # Run distributed processing example
    if __name__ == "__main__":
        distributed_processing_example()
```

### Real-Time Stream Processing

```python
from modules.engine.streaming_pipeline import StreamingDataPipeline
import pandas as pd
import numpy as np
import time
import threading
import queue
from typing import Generator

class RealTimeStreamProcessor:
    """Real-time streaming processor with continuous data ingestion"""
    
    def __init__(self):
        self.pipeline = StreamingDataPipeline(
            chunk_size=1000,
            max_memory_mb=200,
            enable_parallel=True,
            num_workers=2
        )
        
        self.data_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.processing_thread = None
        
    def start_data_ingestion(self):
        """Start continuous data ingestion simulation"""
        def data_generator():
            """Simulate continuous data stream"""
            while self.is_running:
                # Generate batch of new data
                batch = pd.DataFrame({
                    'timestamp': [time.time()] * 100,
                    'sensor_1': np.random.randn(100),
                    'sensor_2': np.random.randn(100),
                    'sensor_3': np.random.randn(100),
                    'device_id': np.random.choice(['device_A', 'device_B', 'device_C'], 100)
                })
                
                try:
                    self.data_queue.put(batch, timeout=1)
                except queue.Full:
                    print("Data queue full, dropping batch")
                
                time.sleep(0.1)  # 100ms interval
        
        self.is_running = True
        self.ingestion_thread = threading.Thread(target=data_generator)
        self.ingestion_thread.start()
        print("Data ingestion started")
    
    def start_processing(self, processing_func, output_callback=None):
        """Start real-time processing of incoming data"""
        def process_stream():
            """Process data from queue continuously"""
            buffer = []
            buffer_size = 1000  # Accumulate data before processing
            
            while self.is_running:
                try:
                    # Get data from queue
                    batch = self.data_queue.get(timeout=1)
                    buffer.append(batch)
                    
                    # Process when buffer is full
                    if len(buffer) >= buffer_size // 100:  # Adjust threshold
                        combined_data = pd.concat(buffer, ignore_index=True)
                        buffer = []
                        
                        # Process the accumulated data
                        for processed_chunk in self.pipeline.process_dataframe_stream(
                            combined_data, 
                            processing_func
                        ):
                            if output_callback:
                                output_callback(processed_chunk)
                            else:
                                print(f"Processed chunk: {processed_chunk.shape}")
                
                except queue.Empty:
                    # Process any remaining data in buffer
                    if buffer:
                        combined_data = pd.concat(buffer, ignore_index=True)
                        buffer = []
                        
                        for processed_chunk in self.pipeline.process_dataframe_stream(
                            combined_data, 
                            processing_func
                        ):
                            if output_callback:
                                output_callback(processed_chunk)
        
        self.processing_thread = threading.Thread(target=process_stream)
        self.processing_thread.start()
        print("Stream processing started")
    
    def stop(self):
        """Stop the real-time processor"""
        self.is_running = False
        
        if hasattr(self, 'ingestion_thread'):
            self.ingestion_thread.join()
        
        if self.processing_thread:
            self.processing_thread.join()
        
        print("Real-time processor stopped")

# Example usage of real-time processing
def real_time_example():
    """Example of real-time stream processing"""
    
    # Define real-time processing function
    def real_time_analytics(chunk: pd.DataFrame) -> pd.DataFrame:
        """Real-time analytics processing"""
        processed = chunk.copy()
        
        # Add derived features
        processed['sensor_sum'] = processed['sensor_1'] + processed['sensor_2'] + processed['sensor_3']
        processed['sensor_variance'] = processed[['sensor_1', 'sensor_2', 'sensor_3']].var(axis=1)
        
        # Anomaly detection (simple threshold-based)
        processed['anomaly_score'] = np.abs(processed['sensor_sum'])
        processed['is_anomaly'] = processed['anomaly_score'] > 2.0
        
        # Device-specific aggregations
        device_stats = processed.groupby('device_id').agg({
            'sensor_sum': ['mean', 'std'],
            'anomaly_score': 'max'
        }).round(3)
        
        # Add device statistics back to the chunk
        for device in processed['device_id'].unique():
            device_mask = processed['device_id'] == device
            processed.loc[device_mask, 'device_mean'] = device_stats.loc[device, ('sensor_sum', 'mean')]
            processed.loc[device_mask, 'device_std'] = device_stats.loc[device, ('sensor_sum', 'std')]
        
        return processed
    
    # Define output callback
    def handle_processed_data(processed_chunk: pd.DataFrame):
        """Handle processed streaming data"""
        anomalies = processed_chunk[processed_chunk['is_anomaly']]
        
        if len(anomalies) > 0:
            print(f"⚠️  ALERT: {len(anomalies)} anomalies detected in chunk")
            print(f"   Max anomaly score: {anomalies['anomaly_score'].max():.3f}")
        
        # In real scenario, you might:
        # - Save to database
        # - Send alerts
        # - Update dashboards
        # - Trigger other processes
    
    # Create and start real-time processor
    processor = RealTimeStreamProcessor()
    
    try:
        # Start data ingestion
        processor.start_data_ingestion()
        
        # Start processing
        processor.start_processing(
            processing_func=real_time_analytics,
            output_callback=handle_processed_data
        )
        
        # Let it run for 30 seconds
        print("Real-time processing running for 30 seconds...")
        time.sleep(30)
        
        # Get final statistics
        stats = processor.pipeline.get_stats()
        print(f"\nFinal Statistics:")
        print(f"  Chunks processed: {stats.total_chunks_processed}")
        print(f"  Rows processed: {stats.total_rows_processed}")
        print(f"  Throughput: {stats.throughput_rows_per_second:.1f} rows/sec")
        
    finally:
        processor.stop()

# Run real-time example
if __name__ == "__main__":
    real_time_example()
```

## Advanced Features

### Custom Backpressure Strategies

```python
from modules.engine.streaming_pipeline import BackpressureController
import psutil
import time

class AdaptiveBackpressureController(BackpressureController):
    """Enhanced backpressure controller with adaptive strategies"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_history = []
        self.last_adjustment_time = time.time()
        
    def should_apply_backpressure(self) -> bool:
        """Enhanced backpressure logic with multiple factors"""
        
        # Original checks
        base_backpressure = super().should_apply_backpressure()
        
        # Additional checks
        
        # CPU usage check
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 90:
            return True
        
        # Disk I/O check
        try:
            disk_io = psutil.disk_io_counters()
            if hasattr(self, '_last_disk_io'):
                read_speed = (disk_io.read_bytes - self._last_disk_io.read_bytes) / 1024 / 1024  # MB/s
                if read_speed > 100:  # More than 100 MB/s
                    return True
            self._last_disk_io = disk_io
        except:
            pass
        
        # Network I/O check (if applicable)
        try:
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                net_speed = (net_io.bytes_sent + net_io.bytes_recv - 
                           self._last_net_io.bytes_sent - self._last_net_io.bytes_recv) / 1024 / 1024
                if net_speed > 50:  # More than 50 MB/s
                    return True
            self._last_net_io = net_io
        except:
            pass
        
        return base_backpressure
    
    def adapt_batch_size(self):
        """Enhanced adaptive batch sizing with performance history"""
        current_time = time.time()
        
        # Only adjust every 5 seconds to avoid oscillation
        if current_time - self.last_adjustment_time < 5:
            return
        
        super().adapt_batch_size()
        
        # Additional adaptations based on performance history
        if len(self.performance_history) > 10:
            recent_performance = self.performance_history[-10:]
            avg_performance = sum(recent_performance) / len(recent_performance)
            
            # If performance is consistently poor, reduce batch size more aggressively
            if avg_performance < 100:  # Less than 100 rows/sec
                self.current_batch_size = max(50, int(self.current_batch_size * 0.5))
            
            # If performance is excellent, increase batch size more aggressively
            elif avg_performance > 1000:  # More than 1000 rows/sec
                self.current_batch_size = min(50000, int(self.current_batch_size * 1.5))
        
        self.last_adjustment_time = current_time
    
    def record_performance(self, rows_per_second: float):
        """Record performance metric for adaptive optimization"""
        self.performance_history.append(rows_per_second)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
```

## Best Practices

### 1. Memory Management

```python
# Configure appropriate memory limits
pipeline = StreamingDataPipeline(
    max_memory_mb=min(1000, psutil.virtual_memory().total / 1024 / 1024 * 0.5)
)
```

### 2. Chunk Size Optimization

```python
# Start with reasonable chunk size and let adaptive batching optimize
initial_chunk_size = max(100, min(10000, data_size // 100))
```

### 3. Error Handling

```python
def safe_processing_function(chunk):
    try:
        return your_processing_function(chunk)
    except Exception as e:
        logging.error(f"Processing error: {e}")
        return chunk  # Return original chunk on error
```

### 4. Resource Cleanup

```python
try:
    for result in pipeline.process_dataframe_stream(df, processing_func):
        handle_result(result)
finally:
    pipeline.cleanup()  # Always cleanup resources
```

## Related Documentation

- [Dynamic Batcher Documentation](dynamic_batcher.md)
- [Memory Pool Documentation](memory_pool.md)
- [Performance Metrics Documentation](performance_metrics.md)
- [Batch Processor Documentation](batch_processor.md)
