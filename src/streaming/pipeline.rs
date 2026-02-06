//! Streaming Pipeline Implementation
//!
//! Efficient streaming data processing with backpressure.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::VecDeque;

use super::backpressure::{BackpressureController, BackpressureConfig};

/// Configuration for streaming pipeline
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Enable backpressure control
    pub enable_backpressure: bool,
    /// Memory threshold in MB
    pub memory_threshold_mb: usize,
    /// Buffer size
    pub buffer_size: usize,
    /// Timeout for operations in milliseconds
    pub timeout_ms: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            max_queue_size: 100,
            enable_backpressure: true,
            memory_threshold_mb: 1000,
            buffer_size: 10,
            timeout_ms: 30000,
        }
    }
}

/// Statistics for streaming pipeline
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Total chunks processed
    pub chunks_processed: u64,
    /// Total rows processed
    pub rows_processed: u64,
    /// Total processing time in milliseconds
    pub total_processing_time_ms: f64,
    /// Average chunk size
    pub avg_chunk_size: f64,
    /// Memory peak in MB
    pub memory_peak_mb: f64,
    /// Throughput (rows per second)
    pub throughput: f64,
    /// Number of backpressure events
    pub backpressure_events: u64,
    /// Number of errors
    pub errors: u64,
}

/// Result of processing a chunk
#[derive(Debug)]
pub struct ChunkResult<T> {
    /// The processed data
    pub data: T,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Chunk index
    pub chunk_index: u64,
    /// Number of rows in this chunk
    pub row_count: usize,
}

impl<T> ChunkResult<T> {
    /// Create a new chunk result
    pub fn new(data: T, processing_time_ms: f64, chunk_index: u64, row_count: usize) -> Self {
        Self {
            data,
            processing_time_ms,
            chunk_index,
            row_count,
        }
    }
}

/// Streaming data pipeline with backpressure control
pub struct StreamingPipeline<T: Send + 'static> {
    config: StreamConfig,
    backpressure: Arc<BackpressureController>,
    
    // Buffer
    buffer: Arc<Mutex<VecDeque<T>>>,
    
    // State
    running: AtomicBool,
    chunk_index: AtomicU64,
    
    // Statistics
    chunks_processed: AtomicU64,
    rows_processed: AtomicU64,
    total_processing_time_ns: AtomicU64,
    errors: AtomicU64,
    memory_peak_bytes: AtomicUsize,
}

impl<T: Send + 'static> StreamingPipeline<T> {
    /// Create a new streaming pipeline
    pub fn new(config: StreamConfig) -> Self {
        let backpressure_config = BackpressureConfig {
            max_queue_size: config.max_queue_size,
            memory_threshold_mb: config.memory_threshold_mb,
            initial_batch_size: config.chunk_size,
            ..Default::default()
        };
        
        Self {
            config,
            backpressure: Arc::new(BackpressureController::new(backpressure_config)),
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            running: AtomicBool::new(false),
            chunk_index: AtomicU64::new(0),
            chunks_processed: AtomicU64::new(0),
            rows_processed: AtomicU64::new(0),
            total_processing_time_ns: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            memory_peak_bytes: AtomicUsize::new(0),
        }
    }
    
    /// Start the pipeline
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
    }
    
    /// Stop the pipeline
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
    
    /// Check if the pipeline is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
    
    /// Push data into the pipeline
    pub fn push(&self, data: T) -> Result<(), StreamError> {
        if self.config.enable_backpressure {
            let timeout = Duration::from_millis(self.config.timeout_ms);
            self.backpressure.wait_for_capacity(Some(timeout))
                .map_err(|_| StreamError::Timeout)?;
        }
        
        let mut buffer = self.buffer.lock()
            .map_err(|_| StreamError::LockError)?;
        
        if buffer.len() >= self.config.max_queue_size {
            return Err(StreamError::BufferFull);
        }
        
        buffer.push_back(data);
        self.backpressure.increment_queue();
        
        Ok(())
    }
    
    /// Pop data from the pipeline
    pub fn pop(&self) -> Option<T> {
        let mut buffer = self.buffer.lock().ok()?;
        let item = buffer.pop_front()?;
        self.backpressure.decrement_queue();
        Some(item)
    }
    
    /// Process a chunk of data with the given function
    pub fn process_chunk<F, R>(&self, data: Vec<T>, process_fn: F) -> Result<ChunkResult<Vec<R>>, StreamError>
    where
        F: Fn(T) -> R,
    {
        let start = Instant::now();
        let row_count = data.len();
        let chunk_idx = self.chunk_index.fetch_add(1, Ordering::SeqCst);
        
        let results: Vec<R> = data.into_iter().map(process_fn).collect();
        
        let elapsed = start.elapsed();
        let processing_time_ms = elapsed.as_secs_f64() * 1000.0;
        
        // Update statistics
        self.chunks_processed.fetch_add(1, Ordering::SeqCst);
        self.rows_processed.fetch_add(row_count as u64, Ordering::SeqCst);
        self.total_processing_time_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::SeqCst);
        
        Ok(ChunkResult::new(results, processing_time_ms, chunk_idx, row_count))
    }
    
    /// Drain the buffer into chunks and process
    pub fn drain_and_process<F, R>(&self, process_fn: F) -> Vec<ChunkResult<Vec<R>>>
    where
        F: Fn(T) -> R + Clone,
    {
        let mut results = Vec::new();
        
        loop {
            let chunk: Vec<T> = {
                let mut buffer = match self.buffer.lock() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                
                let chunk_size = self.config.chunk_size.min(buffer.len());
                if chunk_size == 0 {
                    break;
                }
                
                buffer.drain(..chunk_size).collect()
            };
            
            // Update backpressure
            for _ in 0..chunk.len() {
                self.backpressure.decrement_queue();
            }
            
            match self.process_chunk(chunk, process_fn.clone()) {
                Ok(result) => results.push(result),
                Err(_) => {
                    self.errors.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
        
        results
    }
    
    /// Get pipeline statistics
    pub fn stats(&self) -> StreamStats {
        let chunks = self.chunks_processed.load(Ordering::SeqCst);
        let rows = self.rows_processed.load(Ordering::SeqCst);
        let total_time_ns = self.total_processing_time_ns.load(Ordering::SeqCst);
        
        let total_time_ms = total_time_ns as f64 / 1_000_000.0;
        let total_time_sec = total_time_ns as f64 / 1_000_000_000.0;
        
        let avg_chunk_size = if chunks > 0 {
            rows as f64 / chunks as f64
        } else {
            0.0
        };
        
        let throughput = if total_time_sec > 0.0 {
            rows as f64 / total_time_sec
        } else {
            0.0
        };
        
        let bp_stats = self.backpressure.stats();
        
        StreamStats {
            chunks_processed: chunks,
            rows_processed: rows,
            total_processing_time_ms: total_time_ms,
            avg_chunk_size,
            memory_peak_mb: self.memory_peak_bytes.load(Ordering::SeqCst) as f64 / (1024.0 * 1024.0),
            throughput,
            backpressure_events: bp_stats.backpressure_events,
            errors: self.errors.load(Ordering::SeqCst),
        }
    }
    
    /// Get the current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.lock().map(|b| b.len()).unwrap_or(0)
    }
    
    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer_size() == 0
    }
    
    /// Clear the buffer
    pub fn clear(&self) {
        if let Ok(mut buffer) = self.buffer.lock() {
            buffer.clear();
        }
        self.backpressure.reset();
    }
    
    /// Get the current recommended chunk size
    pub fn recommended_chunk_size(&self) -> usize {
        self.backpressure.current_batch_size()
    }
}

/// Stream processing error
#[derive(Debug, Clone)]
pub enum StreamError {
    /// Buffer is full
    BufferFull,
    /// Operation timed out
    Timeout,
    /// Failed to acquire lock
    LockError,
    /// Processing error
    ProcessingError(String),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::BufferFull => write!(f, "Buffer is full"),
            StreamError::Timeout => write!(f, "Operation timed out"),
            StreamError::LockError => write!(f, "Failed to acquire lock"),
            StreamError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl std::error::Error for StreamError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_streaming_pipeline_basic() {
        let config = StreamConfig::default();
        let pipeline: StreamingPipeline<i32> = StreamingPipeline::new(config);
        
        pipeline.start();
        assert!(pipeline.is_running());
        
        // Push some data
        for i in 0..10 {
            pipeline.push(i).unwrap();
        }
        
        assert_eq!(pipeline.buffer_size(), 10);
        
        // Pop data
        let item = pipeline.pop();
        assert_eq!(item, Some(0));
        assert_eq!(pipeline.buffer_size(), 9);
    }
    
    #[test]
    fn test_process_chunk() {
        let config = StreamConfig::default();
        let pipeline: StreamingPipeline<i32> = StreamingPipeline::new(config);
        
        let data = vec![1, 2, 3, 4, 5];
        let result = pipeline.process_chunk(data, |x| x * 2).unwrap();
        
        assert_eq!(result.data, vec![2, 4, 6, 8, 10]);
        assert_eq!(result.row_count, 5);
    }
    
    #[test]
    fn test_drain_and_process() {
        let mut config = StreamConfig::default();
        config.chunk_size = 3;
        let pipeline: StreamingPipeline<i32> = StreamingPipeline::new(config);
        
        for i in 0..10 {
            pipeline.push(i).unwrap();
        }
        
        let results = pipeline.drain_and_process(|x| x * 2);
        
        // Should have 4 chunks: 3, 3, 3, 1
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].data.len(), 3);
        assert_eq!(results[3].data.len(), 1);
        
        assert!(pipeline.is_empty());
    }
}
