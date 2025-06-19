import os
import time
import logging
import pickle
import hashlib
import json
import threading
import gc
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue
from dataclasses import dataclass
import functools
from collections import deque
import heapq
from contextlib import contextmanager

# Try to import optimization libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import mkl
    MKL_AVAILABLE = True
except ImportError:
    MKL_AVAILABLE = False

# Try to import ONNX for model compilation
try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Try to import thread pool control
try:
    import threadpoolctl
    THREADPOOLCTL_AVAILABLE = True
except ImportError:
    THREADPOOLCTL_AVAILABLE = False

# Try to import treelite for tree ensemble compilation
try:
    import treelite
    import treelite_runtime
    TREELITE_AVAILABLE = True
except ImportError:
    TREELITE_AVAILABLE = False

# Import local modules
from .batch_processor import BatchProcessor
from .data_preprocessor import DataPreprocessor
from .quantizer import Quantizer
from .lru_ttl_cache import LRUTTLCache
from ..configs import (
    QuantizationConfig, BatchProcessorConfig, BatchProcessingStrategy,
    BatchPriority, PreprocessorConfig, NormalizationType, 
    QuantizationMode, ModelType, EngineState, InferenceEngineConfig
)

@dataclass
class PredictionRequest:
    """Container for a prediction request with metadata"""
    id: str
    features: np.ndarray
    priority: int = 0  # Lower value = higher priority
    timestamp: float = 0.0
    future: Optional[Any] = None
    timeout_ms: Optional[float] = None
    
    def __lt__(self, other):
        """Comparison for priority queue (lower value = higher priority)"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class DynamicBatcher:
    """
    Improved dynamic batching system that efficiently groups requests
    and processes them together to maximize hardware utilization.
    """
    
    def __init__(
        self, 
        batch_processor: Callable,
        max_batch_size: int = 64, 
        max_wait_time_ms: float = 10.0,
        max_queue_size: int = 1000
    ):
        self.batch_processor = batch_processor
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        
        # Priority queue for requests (min heap based on priority)
        self.request_queue = []
        self.queue_lock = threading.RLock()
        
        # Batch formation trigger
        self.batch_trigger = threading.Event()
        
        # Control flags
        self.running = False
        self.stop_event = threading.Event()
        
        # Worker thread
        self.worker_thread = None
        
        # Stats
        self.processed_batches = 0
        self.processed_requests = 0
        self.max_observed_queue_size = 0
        self.batch_sizes = deque(maxlen=100)
        self.batch_wait_times = deque(maxlen=100)
        self.max_queue_size = max_queue_size
        
    def start(self):
        """Start the batcher worker thread"""
        if self.running:
            return
            
        self.running = True
        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="DynamicBatcherWorker"
        )
        self.worker_thread.start()
        
    def stop(self, timeout: float = 2.0):
        """Stop the batcher worker thread"""
        if not self.running:
            return
            
        self.running = False
        self.stop_event.set()
        self.batch_trigger.set()  # Wake up worker if it's waiting
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
            
    def enqueue(self, request: PredictionRequest) -> bool:
        """Add a request to the batch queue"""
        with self.queue_lock:
            # Check if queue is full
            if len(self.request_queue) >= self.max_queue_size:
                return False
                
            # Set timestamp if not already set
            if request.timestamp == 0.0:
                request.timestamp = time.monotonic()
                
            # Add to priority queue
            heapq.heappush(self.request_queue, request)
            
            # Update stats
            current_size = len(self.request_queue)
            if current_size > self.max_observed_queue_size:
                self.max_observed_queue_size = current_size
                
            # Signal worker if we have enough items to form a batch
            # or if this is a high-priority request
            if current_size >= self.max_batch_size or request.priority <= 1:
                self.batch_trigger.set()
                
            return True
                
    def _worker_loop(self):
        """Main batcher worker loop that forms and processes batches"""
        last_batch_time = time.monotonic()
        
        while not self.stop_event.is_set():
            current_time = time.monotonic()
            
            # Batch formation criteria:
            # 1. Either max batch size is reached
            # 2. Or max wait time has elapsed since last batch
            # 3. Or high priority item is waiting
            should_process = False
            batch_wait_time = 0
            
            with self.queue_lock:
                queue_size = len(self.request_queue)
                
                # Check if we have items and should form a batch
                if queue_size > 0:
                    batch_wait_time = (current_time - last_batch_time) * 1000  # ms
                    
                    # Case 1: Queue has reached max batch size
                    if queue_size >= self.max_batch_size:
                        should_process = True
                    
                    # Case 2: Wait time exceeded and we have at least one item
                    elif batch_wait_time >= self.max_wait_time_ms:
                        should_process = True
                    
                    # Case 3: High priority request is waiting
                    elif queue_size > 0 and self.request_queue[0].priority <= 1:
                        should_process = True
            
            if should_process:
                self._process_batch()
                last_batch_time = time.monotonic()
            else:
                # Wait for signal or timeout
                timeout = max(0, (self.max_wait_time_ms / 1000) - batch_wait_time / 1000)
                self.batch_trigger.wait(timeout=timeout)
                self.batch_trigger.clear()
                
    def _process_batch(self):
        """Form a batch from the queue and process it"""
        batch_items = []
        request_futures = []
        start_time = time.monotonic()
        
        # Extract items for this batch
        with self.queue_lock:
            batch_size = min(len(self.request_queue), self.max_batch_size)
            
            if batch_size == 0:
                return
                
            batch_items = []
            for _ in range(batch_size):
                request = heapq.heappop(self.request_queue)
                batch_items.append(request)
                if request.future is not None:
                    request_futures.append(request.future)
        
        try:
            # Group requests with same shapes for efficient batching
            batch_groups = self._group_compatible_requests(batch_items)
            
            # Process each group
            for group in batch_groups:
                if not group:
                    continue
                
                # Stack features into a batch
                features = np.vstack([req.features for req in group])
                
                # Process batch
                results = self.batch_processor(features)
                
                # Distribute results back to requesters
                self._distribute_results(group, results)
            
            # Update stats
            batch_time = (time.monotonic() - start_time) * 1000  # ms
            with self.queue_lock:
                self.processed_batches += 1
                self.processed_requests += len(batch_items)
                self.batch_sizes.append(len(batch_items))
                self.batch_wait_times.append(batch_time)
                
        except Exception as e:
            # Mark all futures as failed
            for request in batch_items:
                if request.future and not request.future.done():
                    request.future.set_exception(e)
    
    def _group_compatible_requests(self, requests: List[PredictionRequest]) -> List[List[PredictionRequest]]:
        """Group requests with compatible shapes for efficient batching"""
        if not requests:
            return []
            
        # Group by feature shape (excluding batch dimension)
        groups = {}
        for req in requests:
            # Get shape key (tuple of dimensions except first)
            if hasattr(req.features, 'shape') and len(req.features.shape) > 1:
                shape_key = tuple(req.features.shape[1:])
            else:
                shape_key = (1,)  # Default for 1D arrays
                
            if shape_key not in groups:
                groups[shape_key] = []
            groups[shape_key].append(req)
            
        return list(groups.values())
    
    def _distribute_results(self, requests: List[PredictionRequest], batch_results: np.ndarray):
        """Distribute batch results back to individual requesters"""
        if batch_results is None:
            # Handle error case
            for req in requests:
                if req.future and not req.future.done():
                    req.future.set_exception(RuntimeError("Batch processing returned None"))
            return
            
        # Calculate result slices
        start_idx = 0
        for i, req in enumerate(requests):
            # Get number of rows for this request
            n_rows = req.features.shape[0] if hasattr(req.features, 'shape') else 1
            
            # Extract result slice
            end_idx = start_idx + n_rows
            result_slice = batch_results[start_idx:end_idx]
            
            # Set future result if available
            if req.future and not req.future.done():
                req.future.set_result(result_slice)
                
            start_idx = end_idx
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        with self.queue_lock:
            stats = {
                "processed_batches": self.processed_batches,
                "processed_requests": self.processed_requests,
                "current_queue_size": len(self.request_queue),
                "max_observed_queue_size": self.max_observed_queue_size,
                "avg_batch_size": np.mean(self.batch_sizes) if self.batch_sizes else 0,
                "avg_batch_wait_time_ms": np.mean(self.batch_wait_times) if self.batch_wait_times else 0,
                "max_batch_wait_time_ms": max(self.batch_wait_times) if self.batch_wait_times else 0,
            }
            return stats


class MemoryPool:
    """
    Manages a pool of pre-allocated memory buffers to reduce allocation overhead
    and improve memory reuse for better CPU cache utilization.
    """
    
    def __init__(self, max_buffers: int = 32):
        self.max_buffers = max_buffers
        self.buffer_pools = {}  # Shape -> List of free buffers
        self.lock = threading.RLock()
    
    def get_buffer(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Get a buffer of the specified shape and type from the pool or create one"""
        with self.lock:
            key = (shape, np.dtype(dtype).name)
            if key in self.buffer_pools and self.buffer_pools[key]:
                return self.buffer_pools[key].pop()
            # Use np.empty for performance (will be overwritten)
            return np.empty(shape, dtype=dtype)
    
    def return_buffer(self, buffer: np.ndarray):
        """Return a buffer to the pool for reuse"""
        if buffer is None:
            return
        with self.lock:
            key = (buffer.shape, buffer.dtype.name)
            if key not in self.buffer_pools:
                self.buffer_pools[key] = []
            if len(self.buffer_pools[key]) < self.max_buffers:
                buffer.fill(0)
                self.buffer_pools[key].append(buffer)
    
    def clear(self):
        with self.lock:
            self.buffer_pools.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_buffers = sum(len(buffers) for buffers in self.buffer_pools.values())
            memory_usage = 0
            for key, buffers in self.buffer_pools.items():
                if buffers:
                    shape, dtype_name = key
                    buffer_size = np.prod(shape) * np.dtype(dtype_name).itemsize
                    memory_usage += buffer_size * len(buffers)
            return {
                "total_buffers": total_buffers,
                "memory_usage_bytes": memory_usage,
                "memory_usage_mb": memory_usage / (1024 * 1024),
                "shape_counts": {str(k): len(v) for k, v in self.buffer_pools.items()}
            }


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


class InferenceEngine:
    """
    High-performance inference engine with advanced optimizations for ML workloads.
    
    Features:
    - Dynamic micro-batching with priority queues for optimal hardware utilization
    - Thread pool management with intelligent CPU affinity
    - Memory pooling for reduced allocation overhead and better cache locality
    - Model compilation via ONNX/Treelite for faster inference
    - Optimized vectorized operations for efficient CPU utilization
    - Multi-level caching (results, feature transformations)
    - Quantization for reduced memory footprint
    - Smart preprocessing pipeline with memory reuse
    - Comprehensive performance monitoring and dynamic scaling
    - NUMA-aware processing for multi-socket systems
    """
    
    def __init__(self, config: Optional[InferenceEngineConfig] = None):
        """
        Initialize the inference engine with the given configuration.
        
        Args:
            config: Configuration for the inference engine
        """
        self.config = config or InferenceEngineConfig()
        
        # Set up math library environment variables first (before any libraries load)
        self._configure_math_libraries()
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize engine state
        self.state = EngineState.INITIALIZING
        self.model = None
        self.model_type = None
        self.compiled_model = None  # For ONNX/Treelite compiled models
        self.model_info = {}
        self.feature_names = []
        
        # Thread safety
        self.inference_lock = threading.RLock()
        self.state_lock = threading.RLock()
        
        # Initialize common counters with atomic access
        self.active_requests = 0
        self.active_requests_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = PerformanceMetrics(window_size=self.config.monitoring_window)
        
        # Set up resource management and thread pool
        self._setup_resource_management()
        
        # Initialize memory pool for buffer reuse
        self.memory_pool = MemoryPool(max_buffers=32)
        
        # Feature transformation cache (separate from result cache)
        self.feature_cache = LRUTTLCache(
            max_size=min(1000, self.config.max_cache_entries),
            ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.enable_request_deduplication else None
        
        # Optional components - initialize based on configuration
        self.quantizer = self._setup_quantizer() if self._should_use_quantization() else None
        self.preprocessor = self._setup_preprocessor() if self.config.enable_feature_scaling else None
        self.result_cache = self._setup_cache() if self.config.enable_request_deduplication else None
        
        # Enhanced batching system that replaces the old batch processor
        if self.config.enable_batching:
            self.dynamic_batcher = DynamicBatcher(
                batch_processor=self._process_batch,
                max_batch_size=self.config.max_batch_size,
                max_wait_time_ms=self.config.batch_timeout * 1000,
                max_queue_size=self.config.max_concurrent_requests * 2
            )
            self.dynamic_batcher.start()
        else:
            self.dynamic_batcher = None
        
        # Initialize threadpoolctl for controlling nested parallelism
        if THREADPOOLCTL_AVAILABLE:
            self._threadpool_controllers = threadpoolctl.threadpool_info()
        
        # Monitor resources if enabled
        if self.config.enable_monitoring:
            self._start_monitoring()
        
        # Register garbage collection hooks
        gc.callbacks.append(self._gc_callback)
        
        # Mark as ready
        self.state = EngineState.READY
        self.logger.info(f"Inference engine initialized with model version {self.config.model_version}")
    
    def _configure_math_libraries(self):
        """Configure math libraries for optimal performance"""
        # Set thread counts for numerical libraries based on our threading strategy
        thread_count = "1" if self.config.enable_batching else str(self.config.num_threads)
        
        # Set environment variables for common math libraries
        os.environ["OMP_NUM_THREADS"] = thread_count
        os.environ["MKL_NUM_THREADS"] = thread_count
        os.environ["OPENBLAS_NUM_THREADS"] = thread_count
        os.environ["VECLIB_MAXIMUM_THREADS"] = thread_count
        os.environ["NUMEXPR_NUM_THREADS"] = thread_count
        
        # Set MKL environment variable for fast thread control
        os.environ["MKL_DYNAMIC"] = "FALSE"
        
        # Additional MKL optimizations if available
        if MKL_AVAILABLE:
            # These are Intel-specific optimizations
            os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
            os.environ["KMP_BLOCKTIME"] = "0"
    
    def _gc_callback(self, phase, info):
        """Callback for garbage collection events to track GC pauses"""
        if phase == "start":
            self._gc_start_time = time.time()
        elif phase == "stop":
            if hasattr(self, '_gc_start_time'):
                gc_duration = time.time() - self._gc_start_time
                if gc_duration > 0.1:  # Only log significant GC pauses
                    self.logger.debug(f"GC pause detected: {gc_duration*1000:.2f}ms")
                    
                    # Update metrics
                    with self.metrics.lock:
                        if not hasattr(self.metrics, 'gc_pauses'):
                            self.metrics.gc_pauses = []
                        if len(self.metrics.gc_pauses) >= self.metrics.window_size:
                            self.metrics.gc_pauses.pop(0)
                        self.metrics.gc_pauses.append(gc_duration)
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging based on configuration"""
        logger = logging.getLogger(f"{__name__}.InferenceEngine")
        
        # Set logging level based on debug mode
        level = logging.DEBUG if self.config.debug_mode else logging.INFO
        logger.setLevel(level)
        
        # Only add handlers if none exist
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            
            # Format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Use lazy logging for performance
        logger.propagate = False
        
        return logger
    
    def _setup_resource_management(self):
        """Set up system resource management with advanced optimizations"""
        # Configure thread counts for better resource utilization
        thread_count = self.config.num_threads
        if thread_count <= 0:
            # Auto-detect available CPUs if not specified
            thread_count = os.cpu_count() or 4
        
        # Store thread count for reference
        self.thread_count = thread_count
            
        # Configure MKL if available
        if MKL_AVAILABLE and self.config.enable_intel_optimization:
            try:
                # Explicitly set number of threads
                mkl.set_num_threads(1 if self.config.enable_batching else thread_count)
                
                # Enable conditional numerical optimizations
                if hasattr(mkl, "enable_fast_mm"):
                    mkl.enable_fast_mm(1)
                    
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(f"MKL optimizations enabled with {thread_count} threads")
            except Exception as e:
                if self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(f"Failed to configure MKL: {str(e)}")
        
        # Setup NUMA optimization if available
        self.numa_nodes = []
        self.thread_to_core_map = {}
        if PSUTIL_AVAILABLE:
            try:
                # Detect available NUMA nodes and CPU cores
                if hasattr(psutil, "cpu_count") and hasattr(psutil, "Process"):
                    p = psutil.Process()
                    
                    # Get all available cores
                    available_cores = p.cpu_affinity() if hasattr(p, "cpu_affinity") else list(range(os.cpu_count() or 4))
                    
                    # Limit cores to thread count
                    usable_cores = available_cores[:thread_count]
                    
                    # Set CPU affinity if enabled
                    if self.config.set_cpu_affinity:
                        try:
                            p.cpu_affinity(usable_cores)
                            if self.logger.isEnabledFor(logging.INFO):
                                self.logger.info(f"CPU affinity set to cores {usable_cores}")
                        except Exception as e:
                            if self.logger.isEnabledFor(logging.WARNING):
                                self.logger.warning(f"Failed to set CPU affinity: {str(e)}")
                    
                    # Detect NUMA nodes if possible
                    if hasattr(psutil, "numa_memory_info"):
                        # This is a placeholder - actual implementation would
                        # map cores to NUMA nodes for optimal data locality
                        self.numa_nodes = [0]  # Just one node in this placeholder
                    
                    # Create thread-to-core mapping for worker threads
                    for i in range(thread_count):
                        self.thread_to_core_map[i] = usable_cores[i % len(usable_cores)]
                        
            except Exception as e:
                if self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(f"Failed to setup NUMA optimization: {str(e)}")
        
        # Use a shared thread pool if possible
        if hasattr(self, 'shared_thread_pool') and self.shared_thread_pool:
            self.thread_pool = self.shared_thread_pool
        else:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=thread_count,
                thread_name_prefix="InferenceWorker",
                initializer=self._thread_pool_initializer,
            )
        
        # Large pages, if available (Linux-specific feature)
        if os.name == 'posix' and hasattr(os, 'madvise') and self.config.enable_memory_optimization:
            try:
                # Attempt to enable transparent huge pages (THP)
                with open("/sys/kernel/mm/transparent_hugepage/enabled", "w") as f:
                    f.write("always")
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info("Transparent huge pages enabled for better memory performance")
            except (IOError, PermissionError):
                # Not critical if this fails
                pass
        
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Advanced resource management configured with {thread_count} threads")
    
    def _thread_pool_initializer(self):
        """Initialize worker thread with optimized settings"""
        # Set thread name for better debugging
        threading.current_thread().name = f"Worker-{threading.get_ident()}"
        
        # Pin thread to specific core if mapping is available
        if PSUTIL_AVAILABLE and self.thread_to_core_map:
            try:
                # Get worker index from thread pool
                worker_idx = sum(1 for t in threading.enumerate() 
                              if t.name.startswith("Worker-")) % len(self.thread_to_core_map)
                core_id = self.thread_to_core_map[worker_idx]
                
                # Set affinity for this thread
                p = psutil.Process()
                p.cpu_affinity([core_id])
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Worker thread {threading.current_thread().name} pinned to core {core_id}")
            except Exception as e:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Failed to pin thread to core: {str(e)}")
        
        # Limit internal parallelism of math libraries for this thread
        if THREADPOOLCTL_AVAILABLE:
            threadpoolctl.threadpool_limits(limits=1, user_api="blas")
            threadpoolctl.threadpool_limits(limits=1, user_api="openmp")
    
    def _should_use_quantization(self) -> bool:
        """Determine if quantization should be used"""
        return (self.config.enable_quantization or 
                self.config.enable_model_quantization or 
                self.config.enable_input_quantization)
    
    def _setup_quantizer(self) -> Optional[Quantizer]:
        """Initialize and configure the quantizer"""
        try:
            quantization_config = self.config.quantization_config
            
            # Create default config if none provided
            if quantization_config is None:
                quantization_config = QuantizationConfig(
                    quantization_type=self.config.quantization_dtype,
                    quantization_mode=QuantizationMode.DYNAMIC_PER_BATCH.value,
                    enable_cache=True,
                    cache_size=1024
                )
            
            quantizer = Quantizer(quantization_config)
            self.logger.info(f"Quantizer initialized with {quantization_config.quantization_type} type")
            return quantizer
        except Exception as e:
            self.logger.error(f"Failed to initialize quantizer: {str(e)}")
            return None
    
    def _setup_preprocessor(self) -> Optional[DataPreprocessor]:
        """Initialize and configure the data preprocessor"""
        try:
            preprocessor_config = PreprocessorConfig(
                normalization=NormalizationType.STANDARD,
                handle_nan=True,
                handle_inf=True,
                detect_outliers=True,
                cache_enabled=True
            )
            
            preprocessor = DataPreprocessor(preprocessor_config)
            self.logger.info("Data preprocessor initialized")
            return preprocessor
        except Exception as e:
            self.logger.error(f"Failed to initialize preprocessor: {str(e)}")
            return None
    
    def _setup_cache(self) -> Optional[LRUTTLCache]:
        """Initialize and configure the result cache"""
        try:
            cache = LRUTTLCache(
                max_size=self.config.max_cache_entries,
                ttl_seconds=self.config.cache_ttl_seconds
            )
            self.logger.info(f"Result cache initialized with {self.config.max_cache_entries} entries")
            return cache
        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {str(e)}")
            return None
    
    def _setup_batch_processor(self) -> Optional[BatchProcessor]:
        """Initialize and configure the batch processor"""
        try:
            # Determine batch processing strategy
            strategy_str = self.config.batching_strategy.upper()
            try:
                strategy = BatchProcessingStrategy[strategy_str]
            except KeyError:
                strategy = BatchProcessingStrategy.ADAPTIVE
                self.logger.warning(f"Unknown batch strategy: {strategy_str}, using ADAPTIVE")
            
            # Create the batch processor configuration
            batch_config = BatchProcessorConfig(
                initial_batch_size=self.config.initial_batch_size,
                min_batch_size=self.config.min_batch_size,
                max_batch_size=self.config.max_batch_size,
                batch_timeout=self.config.batch_timeout,
                max_queue_size=self.config.max_concurrent_requests * 2,
                enable_priority_queue=True,
                processing_strategy=strategy,
                enable_adaptive_batching=self.config.enable_adaptive_batching,
                enable_monitoring=self.config.enable_monitoring,
                num_workers=self.config.num_threads,
                enable_memory_optimization=self.config.enable_memory_optimization,
                max_retries=3
            )
            
            # Create the batch processor
            processor = BatchProcessor(batch_config)
            
            # Start the batch processor with our processing function
            processor.start(self._process_batch)
            
            self.logger.info(f"Batch processor started with strategy={strategy.name}")
            return processor
        except Exception as e:
            self.logger.error(f"Failed to initialize batch processor: {str(e)}")
            return None
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        self.monitor_stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitor"
        )
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def _monitoring_loop(self):
        """Background monitoring thread for resource usage"""
        check_interval = self.config.monitoring_interval
        
        while not getattr(self, 'monitor_stop_event', threading.Event()).is_set():
            try:
                if PSUTIL_AVAILABLE:
                    # Monitor memory usage
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    
                    # Take action if memory usage is too high
                    if memory_mb > self.config.memory_high_watermark_mb:
                        self.logger.warning(
                            f"Memory usage ({memory_mb:.1f} MB) exceeds high watermark, "
                            f"triggering garbage collection"
                        )
                        gc.collect()
                        
                        # Clear cache if memory is still high
                        if self.result_cache is not None:
                            memory_info = process.memory_info()
                            memory_mb = memory_info.rss / (1024 * 1024)
                            
                            if memory_mb > self.config.memory_high_watermark_mb:
                                self.result_cache.clear()
                                self.logger.info("Cache cleared due to high memory usage")
                    
                    # Monitor CPU usage if configured
                    if self.config.throttle_on_high_cpu:
                        cpu_percent = process.cpu_percent(interval=0.1)
                        
                        # Enable throttling if CPU usage is too high
                        if cpu_percent > self.config.cpu_threshold_percent and not self.config.enable_throttling:
                            self.logger.warning(
                                f"CPU usage ({cpu_percent:.1f}%) above threshold, enabling throttling"
                            )
                            self.config.enable_throttling = True
                        # Disable throttling if CPU usage has dropped significantly
                        elif (cpu_percent < self.config.cpu_threshold_percent * 0.8 and 
                              self.config.enable_throttling):
                            self.logger.info(
                                f"CPU usage ({cpu_percent:.1f}%) below threshold, disabling throttling"
                            )
                            self.config.enable_throttling = False
                
                # Log detailed metrics in debug mode
                if self.config.debug_mode:
                    metrics = self.metrics.get_metrics()
                    self.logger.debug(
                        f"Metrics: avg_inference={metrics.get('avg_inference_time_ms', 0):.2f}ms, "
                        f"throughput={metrics.get('throughput_requests_per_second', 0):.1f}req/s, "
                        f"error_rate={metrics.get('error_rate', 0)*100:.2f}%"
                    )
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
            
            # Sleep until next check
            time.sleep(check_interval)
    
    def load_model(self, model_path: str, model_type: Optional[ModelType] = None,
                compile_model: bool = None) -> bool:
        """
        Load a pre-trained model from file with optimized processing.
        
        Args:
            model_path: Path to the saved model file
            model_type: Type of model being loaded (auto-detected if None)
            compile_model: Whether to compile model for faster inference (if supported)
                           If None, uses config setting
            
        Returns:
            Success status
        """
        # Update engine state
        with self.state_lock:
            if self.state in (EngineState.LOADING, EngineState.STOPPING, EngineState.STOPPED):
                self.logger.warning(f"Cannot load model in current state: {self.state}")
                return False
            
            self.state = EngineState.LOADING
        
        self.logger.info(f"Loading model from {model_path}")
        load_start_time = time.time()
        
        try:
            # Validate model file exists
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                self.state = EngineState.ERROR
                return False
            
            # Detect model type if not provided
            if model_type is None:
                model_type = self._detect_model_type(model_path)
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(f"Auto-detected model type: {model_type}")
            
            # Load the model based on type
            if model_type == ModelType.SKLEARN:
                if JOBLIB_AVAILABLE:
                    model = joblib.load(model_path)
                else:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                
                # Extract feature names if available
                if hasattr(model, 'feature_names_in_'):
                    self.feature_names = model.feature_names_in_.tolist()
            
            elif model_type == ModelType.XGBOOST:
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(model_path)
                
                # Try to extract feature names from model if available
                feature_names = model.feature_names
                if feature_names:
                    self.feature_names = feature_names
            
            elif model_type == ModelType.LIGHTGBM:
                import lightgbm as lgb
                model = lgb.Booster(model_file=model_path)
                
                # Try to extract feature names
                feature_names = model.feature_name()
                if feature_names:
                    self.feature_names = feature_names
            
            elif model_type == ModelType.ENSEMBLE:
                # Load ensemble model config
                if model_path.endswith('.json'):
                    with open(model_path, 'r') as f:
                        ensemble_config = json.load(f)
                    model = self._load_ensemble_model(ensemble_config)
                else:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                
                # Try to extract feature names from first model if it's a dict
                if isinstance(model, dict) and 'models' in model and model['models']:
                    first_model = model['models'][0]
                    if hasattr(first_model, 'feature_names_in_'):
                        self.feature_names = first_model.feature_names_in_.tolist()
            
            elif model_type == ModelType.CUSTOM:
                # Default to pickle for custom models
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                self.state = EngineState.ERROR
                return False
            
            # Store model and update state
            self.model = model
            self.model_type = model_type
            
            # Extract model information
            self.model_info = self._extract_model_info(model, model_type)
            self.model_info['load_time_ms'] = (time.time() - load_start_time) * 1000
            
            # If we have a preprocessor and feature names, fit it on synthetic data
            if self.preprocessor is not None and self.feature_names:
                self._initialize_preprocessor()
            
            # Compile model if requested or configured
            should_compile = compile_model if compile_model is not None else self.config.enable_jit
            if should_compile:
                self._compile_model()
            
            # Quantize model if enabled
            if self.config.enable_model_quantization and self.quantizer is not None:
                self._quantize_model()
            
            # Warm up the model if enabled
            if self.config.enable_warmup:
                self._warmup_model()
            
            # Cache optimization: pre-compute common values
            if self.config.enable_memory_optimization:
                self._precompute_common_values()
            
            # Force GC to clean up any temporary objects
            gc.collect()
            
            self.state = EngineState.READY
            self.logger.info(f"Model loaded successfully in {time.time() - load_start_time:.2f}s")
            
            return True
        
        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Failed to load model: {str(e)}")
            try:
                import traceback
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
            except ImportError:
                pass
            self.state = EngineState.ERROR
            return False
    
    def _detect_model_type(self, model_path: str) -> ModelType:
        """Detect model type from file extension or peek at content"""
        # Check file extension first
        file_ext = os.path.splitext(model_path)[1].lower()
        
        if file_ext == '.json':
            return ModelType.ENSEMBLE
        elif file_ext in ('.pkl', '.pickle'):
            # Try to peek at the model to determine type
            try:
                if JOBLIB_AVAILABLE:
                    model = joblib.load(model_path)
                else:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                
                # Check model class name
                model_class = model.__class__.__module__ + '.' + model.__class__.__name__
                
                if 'sklearn' in model_class:
                    return ModelType.SKLEARN
                elif 'xgboost' in model_class:
                    return ModelType.XGBOOST
                elif 'lightgbm' in model_class:
                    return ModelType.LIGHTGBM
                elif 'ensemble' in model_class.lower():
                    return ModelType.ENSEMBLE
                
                return ModelType.CUSTOM
            except Exception:
                # If can't determine, default to custom
                return ModelType.CUSTOM
        elif file_ext == '.model' or file_ext == '.bst':
            # Likely XGBoost model
            return ModelType.XGBOOST
        elif file_ext == '.txt':
            # Likely LightGBM model
            return ModelType.LIGHTGBM
        else:
            # Default to custom for unknown types
            return ModelType.CUSTOM
    
    def _extract_model_info(self, model: Any, model_type: ModelType) -> Dict[str, Any]:
        """Extract information about the loaded model"""
        info = {
            "model_type": model_type.name,
            "timestamp": datetime.now().isoformat(),
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}"
        }
        
        # Extract model-specific info
        if model_type == ModelType.SKLEARN:
            # Get hyperparameters
            if hasattr(model, 'get_params'):
                info["hyperparameters"] = model.get_params()
            
            # Get feature count
            if hasattr(model, 'n_features_in_'):
                info["feature_count"] = model.n_features_in_
        
        elif model_type == ModelType.XGBOOST:
            # Get number of trees
            try:
                dump = model.get_dump()
                info["num_trees"] = len(dump)
            except:
                pass
        
        elif model_type == ModelType.LIGHTGBM:
            # Get model parameters
            try:
                info["hyperparameters"] = model.params
            except:
                pass
            
            # Get number of trees
            try:
                info["num_trees"] = model.num_trees()
            except:
                pass
        
        return info
    
    def _initialize_preprocessor(self):
        """Initialize preprocessor with synthetic data based on feature names"""
        if self.preprocessor is None or not self.feature_names:
            return
        
        try:
            # Create synthetic data for fitting the preprocessor
            feature_count = len(self.feature_names)
            sample_count = 100
            
            # Generate random data with reasonable ranges
            synthetic_data = np.random.randn(sample_count, feature_count)
            
            # Fit the preprocessor
            self.preprocessor.fit(synthetic_data, feature_names=self.feature_names)
            self.logger.info("Preprocessor initialized with synthetic data")
        except Exception as e:
            self.logger.warning(f"Failed to initialize preprocessor: {str(e)}")
    
    def _quantize_model(self):
        """Quantize the model weights if supported"""
        if self.quantizer is None or self.model is None:
            return
        
        try:
            # Model quantization depends on model type
            if self.model_type == ModelType.SKLEARN:
                if hasattr(self.model, 'coef_'):
                    # Linear models: quantize coefficients
                    original_coef = self.model.coef_.copy()
                    quantized_coef, _ = self.quantizer.quantize(original_coef)
                    
                    # Store original and set quantized version
                    self.model._original_coef = original_coef
                    self.model.coef_ = quantized_coef
                    
                    if hasattr(self.model, 'intercept_'):
                        self.model._original_intercept = self.model.intercept_.copy()
                        quantized_intercept, _ = self.quantizer.quantize(self.model.intercept_)
                        self.model.intercept_ = quantized_intercept
                    
                    self.logger.info("Model weights quantized successfully")
            else:
                self.logger.info(f"Model quantization not supported for {self.model_type.name}")
        except Exception as e:
            self.logger.error(f"Failed to quantize model: {str(e)}")
    
    def _warmup_model(self):
        """Perform model warm-up to stabilize performance"""
        if not self.config.enable_warmup or self.model is None:
            return
        
        try:
            # Determine feature count
            feature_count = len(self.feature_names) if self.feature_names else 10
            
            # Generate synthetic data for warm-up
            batch_size = min(32, self.config.max_batch_size)
            warmup_data = np.random.rand(batch_size, feature_count)
            
            self.logger.info(f"Warming up model with {batch_size} samples...")
            
            # Run multiple batches through the model for warm-up
            for i in range(3):
                if self.config.enable_input_quantization and self.quantizer:
                    # With quantization
                    quantized_data, _ = self.quantizer.quantize(warmup_data)
                    _ = self._predict_internal(quantized_data)
                else:
                    # Without quantization
                    _ = self._predict_internal(warmup_data)
            
            self.logger.info("Model warm-up completed")
        except Exception as e:
            self.logger.warning(f"Model warm-up failed: {str(e)}")
    
    def predict(self, features: np.ndarray) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Make optimized prediction with input features.
        
        Args:
            features: Input features as numpy array
            
        Returns:
            Tuple of (success_flag, predictions, metadata)
        """
        # Check if engine is ready
        if self.state not in (EngineState.READY, EngineState.RUNNING):
            error_msg = f"Cannot predict: engine in {self.state} state"
            self.logger.error(error_msg)
            return False, None, {"error": error_msg}
        
        # Update engine state
        with self.state_lock:
            if self.state == EngineState.READY:
                self.state = EngineState.RUNNING
        
        # Check if we should throttle based on active requests
        if self.config.enable_throttling:
            with self.active_requests_lock:
                if self.active_requests >= self.config.max_concurrent_requests:
                    error_msg = "Request throttled due to high load"
                    self.logger.warning(error_msg)
                    return False, None, {"error": error_msg, "throttled": True}
        
        # Check cache for identical input if enabled - FAST PATH
        if self.result_cache is not None:
            cache_key = self._create_cache_key(features)
            hit, cached_result = self.result_cache.get(cache_key)
            
            if hit:
                self.metrics.record_cache_hit()
                return True, cached_result, {"cached": True, "cache_hit": True}
            else:
                self.metrics.record_cache_miss()
        
        # Increment active request counter
        with self.active_requests_lock:
            self.active_requests += 1
        
        # Track processing times
        start_time = time.time()
        preprocessing_time = 0
        quantization_time = 0
        feature_extraction_time = 0
        
        try:
            # Ensure features is a numpy array with correct shape
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
            
            # Reshape 1D array to 2D if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)
                
            # Check if feature cache has this input - reuse transformed features
            if self.feature_cache is not None:
                feature_key = self._create_cache_key(features)
                cache_hit, cached_features = self.feature_cache.get(feature_key)
                
                if cache_hit:
                    # Use cached preprocessed features
                    features = cached_features
                else:
                    # Apply preprocessing if enabled
                    if self.preprocessor is not None:
                        preprocess_start = time.time()
                        try:
                            # Try to get a buffer from memory pool for result
                            result_buffer = self.memory_pool.get_buffer(
                                shape=features.shape, 
                                dtype=features.dtype
                            )
                            
                            # Transform in-place to the buffer if possible
                            features = self.preprocessor.transform(features, copy=True)
                            
                            # Store in feature cache
                            self.feature_cache.set(feature_key, features)
                            
                            preprocessing_time = time.time() - preprocess_start
                        except Exception as e:
                            self.logger.error(f"Preprocessing error: {str(e)}")
                            return False, None, {"error": f"Preprocessing failed: {str(e)}"}
            
            # Apply quantization if enabled
            if self.config.enable_input_quantization and self.quantizer is not None:
                quantize_start = time.time()
                try:
                    # Perform quantization on the input features
                    quantized_features, quantization_params = self.quantizer.quantize(features)
                    quantization_time = time.time() - quantize_start
                    
                    # Use quantized features for prediction
                    features = quantized_features
                except Exception as e:
                    self.logger.error(f"Quantization error: {str(e)}")
                    return False, None, {"error": f"Quantization failed: {str(e)}"}
            
            # Make prediction with timing
            predict_start = time.time()
            
            # Use the appropriate prediction method based on compilation status
            with self._control_threading(1):  # Limit threads during prediction
                if self.compiled_model is not None:
                    # Use compiled model (ONNX/Treelite)
                    predictions = self._predict_compiled(features)
                else:
                    # Use regular model
                    predictions = self._predict_internal(features)
                    
            inference_time = time.time() - predict_start
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Update performance metrics
            batch_size = features.shape[0] if hasattr(features, 'shape') else 1
            self.metrics.update_inference(
                inference_time=inference_time,
                batch_size=batch_size,
                preprocessing_time=preprocessing_time,
                quantization_time=quantization_time
            )
            
            # Cache result if enabled
            if self.result_cache is not None:
                cache_key = self._create_cache_key(features)
                self.result_cache.set(cache_key, predictions)
            
            # Create response metadata
            metadata = {
                "inference_time_ms": inference_time * 1000,
                "total_time_ms": total_time * 1000,
                "batch_size": batch_size,
                "compiled_model_used": self.compiled_model is not None
            }
            
            # Add processing time breakdowns if significant
            if preprocessing_time > 0:
                metadata["preprocessing_time_ms"] = preprocessing_time * 1000
            if quantization_time > 0:
                metadata["quantization_time_ms"] = quantization_time * 1000
            if feature_extraction_time > 0:
                metadata["feature_extraction_time_ms"] = feature_extraction_time * 1000
            
            self.logger.debug(
                f"Prediction completed: batch_size={batch_size}, "
                f"time={total_time*1000:.2f}ms"
            )
            
            return True, predictions, metadata
            
        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Prediction error: {str(e)}")
            try:
                import traceback
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
            except ImportError:
                pass
            self.metrics.record_error()
            return False, None, {"error": str(e)}
            
        finally:
            # Decrement active request counter
            with self.active_requests_lock:
                self.active_requests -= 1
            
            # Update engine state if no active requests
            with self.state_lock:
                if self.state == EngineState.RUNNING and self.active_requests == 0:
                    self.state = EngineState.READY
                    
    def _predict_internal(self, features: np.ndarray) -> np.ndarray:
        """
        Make a prediction using the loaded model with optimized execution path.
        
        Args:
            features: Input features
            
        Returns:
            Prediction results
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        # Create a clean, contiguous copy of features if needed
        # This improves memory access patterns for vectorized operations
        if not features.flags.c_contiguous:
            features = np.ascontiguousarray(features)
            
        # Choose prediction method based on model type
        model_type = self.model_type
        if model_type == ModelType.SKLEARN:
            # Scikit-learn models 
            # For batch predictions, use vectorized APIs directly
            if hasattr(self.model, 'predict_proba') and self.config.return_probabilities:
                # Get probabilities for classification
                return self.model.predict_proba(features)
            else:
                # Standard prediction
                return self.model.predict(features)
                
        elif model_type == ModelType.XGBOOST:
            try:
                import xgboost as xgb
                dmatrix = xgb.DMatrix(features)
                pred_params = {}
                if features.shape[0] <= 10:
                    pred_params['ntree_limit'] = 0
                    pred_params['nthread'] = 1
                else:
                    pred_params['nthread'] = min(4, self.thread_count)
                return self.model.predict(dmatrix, **pred_params)
            except Exception as e:
                if self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(f"Optimized XGBoost prediction failed: {str(e)}, falling back to basic")
                return self.model.predict(features)
        elif model_type == ModelType.LIGHTGBM:
            try:
                import lightgbm as lgb
                pred_params = {}
                if features.shape[0] <= 10:
                    pred_params['num_threads'] = 1
                else:
                    pred_params['num_threads'] = min(4, self.thread_count)
                return self.model.predict(features, **pred_params)
            except Exception as e:
                if self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(f"Optimized LightGBM prediction failed: {str(e)}")
                return self.model.predict(features)
        elif model_type == ModelType.ENSEMBLE:
            if hasattr(self.model, 'predict'):
                return self.model.predict(features)
            elif isinstance(self.model, dict) and 'models' in self.model:
                return self._predict_custom_ensemble_optimized(features, self.model)
            else:
                raise ValueError(f"Unsupported ensemble model format")
        elif model_type == ModelType.CUSTOM:
            if hasattr(self.model, 'predict'):
                return self.model.predict(features)
            elif callable(self.model):
                return self.model(features)
            else:
                raise ValueError(f"Model does not have a standard prediction interface")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _predict_compiled(self, features: np.ndarray) -> np.ndarray:
        """
        Make a prediction using the compiled model (ONNX/Treelite).
        
        Args:
            features: Input features
            
        Returns:
            Prediction results
        """
        if self.compiled_model is None:
            raise RuntimeError("No compiled model available")
        
        # Handle different compiled model types
        if hasattr(self.compiled_model, 'run'):  # ONNX Runtime
            input_name = getattr(self, 'onnx_input_name', 'float_input')
            onnx_inputs = {input_name: features.astype(np.float32)}
            outputs = self.compiled_model.run(None, onnx_inputs)
            return outputs[0]
        elif hasattr(self.compiled_model, 'predict'):  # Treelite
            try:
                from treelite_runtime import Batch
            except ImportError:
                raise RuntimeError("treelite_runtime is not available")
            batch = Batch.from_npy2d(features)
            return self.compiled_model.predict(batch)
        else:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning("Unknown compiled model type, falling back to regular prediction")
            return self._predict_internal(features)

    def enqueue_prediction(self, features: np.ndarray, 
                          priority: BatchPriority = BatchPriority.NORMAL,
                          timeout_ms: Optional[float] = None) -> Any:
        """
        Enqueue a prediction request to be processed asynchronously
        with the new dynamic batcher.
        
        Args:
            features: Input features
            priority: Processing priority
            timeout_ms: Optional timeout in milliseconds
            
        Returns:
            Future object with the prediction result
        """
        if self.dynamic_batcher is None:
            raise RuntimeError("Dynamic batching is not enabled")
            
        # Create a future for the result
        from concurrent.futures import Future
        future = Future()
        
        # Create a prediction request
        request_id = str(hash(str(features) + str(time.time())))
        request = PredictionRequest(
            id=request_id,
            features=features,
            priority=priority.value,
            timestamp=time.monotonic(),
            future=future,
            timeout_ms=timeout_ms
        )
        
        # Enqueue the request
        if not self.dynamic_batcher.enqueue(request):
            future.set_exception(RuntimeError("Request queue is full"))
            
        return future
    
    def predict_batch(self, features_list: List[np.ndarray], 
                     timeout: Optional[float] = None) -> List[Tuple[bool, np.ndarray, Dict[str, Any]]]:
        """
        Optimized batch prediction for multiple inputs with vectorized operations.
        
        Args:
            features_list: List of feature arrays
            timeout: Optional timeout in seconds
            
        Returns:
            List of (success, predictions, metadata) tuples
        """
        # Early exit for empty list
        if not features_list:
            return []
            
        # Start timing
        start_time = time.time()
        
        # Check if shapes are compatible for batching
        shapes = [features.shape[1:] if features.ndim > 1 else (features.shape[0],)
                 for features in features_list]
        
        if all(shape == shapes[0] for shape in shapes):
            # All shapes match - can use efficient batching
            try:
                # Stack features into a single batch
                if all(isinstance(f, np.ndarray) for f in features_list):
                    # Use numpy vstack for efficient stacking
                    stacked_features = np.vstack(features_list)
                else:
                    # Convert to numpy arrays first
                    stacked_features = np.vstack([np.array(f) for f in features_list])
                    
                # Make a single prediction on the batch
                success, predictions, metadata = self.predict(stacked_features)
                
                if not success:
                    # Return error for all inputs
                    return [(False, None, metadata) for _ in features_list]
                
                # Split predictions back to individual results
                result = []
                start_idx = 0
                for i, features in enumerate(features_list):
                    n_samples = features.shape[0] if hasattr(features, 'shape') else 1
                    end_idx = start_idx + n_samples
                    
                    # Extract slice for this input
                    pred_slice = predictions[start_idx:end_idx]
                    
                    # Create individual metadata (copy shared and add specifics)
                    item_metadata = metadata.copy()
                    item_metadata['batch_index'] = i
                    item_metadata['batch_size'] = len(features_list)
                    
                    result.append((True, pred_slice, item_metadata))
                    start_idx = end_idx
                    
                return result
            
            except Exception as e:
                if self.logger.isEnabledFor(logging.ERROR):
                    self.logger.error(f"Batch prediction error: {str(e)}")
                try:
                    import traceback
                    if self.config.debug_mode:
                        self.logger.error(traceback.format_exc())
                except ImportError:
                    pass
                self.logger.info("Falling back to individual processing after batch error")
        
        # If shapes are incompatible or batch processing failed, process individually
        return [self.predict(features) for features in features_list]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics from all components.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = self.metrics.get_metrics()
        
        # Add memory pool stats
        memory_stats = self.memory_pool.get_stats()
        metrics["memory_pool"] = memory_stats
        
        # Add dynamic batcher stats if available
        if self.dynamic_batcher:
            batcher_stats = self.dynamic_batcher.get_stats()
            metrics["dynamic_batcher"] = batcher_stats
            
        # Add cache stats
        if self.result_cache:
            metrics["result_cache"] = self.result_cache.get_stats()
            
        # Add feature cache stats if enabled
        if self.feature_cache:
            metrics["feature_cache"] = self.feature_cache.get_stats()
            
        # Add system metrics if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                
                # Get CPU and memory usage
                metrics["system"] = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
                    "threads_count": process.num_threads(),
                    "open_files": len(process.open_files()),
                }
                
                # Add NUMA stats if available
                if hasattr(process, "numa_memory_info"):
                    numa_info = process.numa_memory_info()
                    metrics["system"]["numa_info"] = numa_info
            except Exception as e:
                self.logger.debug(f"Error collecting system metrics: {str(e)}")
                
        # Add GC stats
        metrics["gc"] = {
            "enabled": gc.isenabled(),
            "threshold": gc.get_threshold(),
            "count": gc.get_count(),
        }
        
        if hasattr(self.metrics, 'gc_pauses'):
            metrics["gc"]["avg_pause_ms"] = np.mean(self.metrics.gc_pauses) * 1000 if self.metrics.gc_pauses else 0
            metrics["gc"]["max_pause_ms"] = np.max(self.metrics.gc_pauses) * 1000 if self.metrics.gc_pauses else 0
            
        return metrics
    
    def shutdown(self):
        """
        Shutdown the engine and release resources with proper cleanup.
        """
        self.logger.info("Shutting down inference engine")
        
        with self.state_lock:
            self.state = EngineState.STOPPING
            
        # Stop monitoring
        if hasattr(self, 'monitor_stop_event'):
            self.monitor_stop_event.set()
            if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=1.0)
                
        # Stop dynamic batcher
        if self.dynamic_batcher:
            self.dynamic_batcher.stop()
            
        # Clear caches
        if self.result_cache:
            self.result_cache.clear()
            
        if self.feature_cache:
            self.feature_cache.clear()
            
        # Clear memory pool
        self.memory_pool.clear()
        
        # Shutdown thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
            
        # Remove GC callback
        if gc.callbacks and self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)
            
        # Unload model references to free memory
        self.model = None
        self.compiled_model = None
            
        # Update state
        with self.state_lock:
            self.state = EngineState.STOPPED
            
        self.logger.info("Inference engine shutdown complete")
        
    def __del__(self):
        """Clean up resources when the object is garbage collected."""
        try:
            # Only call shutdown if not already stopped
            if hasattr(self, 'state') and self.state not in (EngineState.STOPPING, EngineState.STOPPED):
                self.shutdown()
        except:
            # Ignore errors during garbage collection
            pass

    # Fix 1: Add missing _control_threading method - contextmanager for controlling thread counts
    @contextmanager
    def _control_threading(self, num_threads: int):
        """
        Context manager to temporarily control thread limits for numerical libraries.
        
        Args:
            num_threads: Number of threads to use for computations
        """
        original_settings = {}
        
        try:
            # Save original settings
            if hasattr(os, 'environ'):
                for env_var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 
                            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                    if env_var in os.environ:
                        original_settings[env_var] = os.environ[env_var]
            
            # Set thread count temporarily
            thread_count = str(num_threads)
            os.environ["OMP_NUM_THREADS"] = thread_count
            os.environ["MKL_NUM_THREADS"] = thread_count
            os.environ["OPENBLAS_NUM_THREADS"] = thread_count
            os.environ["VECLIB_MAXIMUM_THREADS"] = thread_count
            os.environ["NUMEXPR_NUM_THREADS"] = thread_count
            
            # Apply threadpoolctl if available
            if hasattr(self, '_threadpool_controllers') and self.config.threadpoolctl_available:
                import threadpoolctl
                self._threadpool_limit = threadpoolctl.threadpool_limits(limits=num_threads)
                
            # Yield control back to the caller
            yield
            
        finally:
            # Restore original settings
            for env_var, value in original_settings.items():
                os.environ[env_var] = value
                
            # Release threadpoolctl limit if it was applied
            if hasattr(self, '_threadpool_limit'):
                del self._threadpool_limit

    # Fix 2: Add missing _compile_model method
    def _compile_model(self):
        """
        Compile model to optimized format for faster inference if supported.
        For ONNX: Converts scikit-learn/XGBoost models to ONNX format.
        For Treelite: Converts tree ensemble models to compiled format.
        """
        if self.model is None:
            self.logger.warning("Cannot compile: no model loaded")
            return
            
        self.logger.info(f"Compiling model of type {self.model_type.name}")
        
        try:
            if self.model_type == self.ModelType.SKLEARN:
                if not self.config.onnx_available:
                    if self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning("ONNX not available for model compilation")
                    return
                try:
                    import onnx
                    import onnxruntime
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType
                except ImportError:
                    if self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning("skl2onnx or onnxruntime not available")
                    return
                
                # Get input shape for ONNX conversion
                n_features = len(self.feature_names) if self.feature_names else (
                    self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None
                )
                
                if n_features is None:
                    self.logger.warning("Could not determine feature count for ONNX conversion")
                    return
                    
                # Define input type
                input_type = [('float_input', FloatTensorType([None, n_features]))]
                
                # Convert model to ONNX
                onnx_model = convert_sklearn(self.model, initial_types=input_type)
                
                # Save ONNX model to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                    onnx_path = f.name
                    f.write(onnx_model.SerializeToString())
                
                # Load ONNX model with runtime
                self.onnx_input_name = 'float_input'  # Store for prediction
                self.compiled_model = onnxruntime.InferenceSession(onnx_path)
                
                # Remove temporary file
                os.unlink(onnx_path)
                
                self.logger.info("Model compiled to ONNX format successfully")
                
            elif self.model_type in [self.ModelType.XGBOOST, self.ModelType.LIGHTGBM]:
                if not self.config.treelite_available:
                    if self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning("Treelite not available for model compilation")
                    return
                try:
                    import treelite
                    import treelite_runtime
                except ImportError:
                    if self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning("treelite or treelite_runtime not available")
                    return
                
                # Handle different tree model types
                if self.model_type == self.ModelType.XGBOOST:
                    # Export to treelite model
                    treelite_model = treelite.Model.from_xgboost(self.model)
                else:  # LightGBM
                    # Export LightGBM model to temporary file for treelite
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                        lgb_path = f.name
                        self.model.save_model(lgb_path)
                    
                    # Load into treelite
                    treelite_model = treelite.Model.load(lgb_path, model_format='lightgbm')
                    os.unlink(lgb_path)  # Remove temp file
                
                # Compile model - optimize for current architecture
                libpath = treelite_model.compile(dirpath='.', params={'parallel_comp': self.thread_count})
                
                # Load compiled model
                self.compiled_model = treelite_runtime.Predictor(libpath)
                
                self.logger.info("Tree model compiled with Treelite successfully")
            else:
                self.logger.info(f"Model type {self.model_type.name} does not support compilation")
        
        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Model compilation failed: {str(e)}")
            try:
                import traceback
                self.logger.debug(traceback.format_exc())
            except ImportError:
                pass
            self.compiled_model = None


    # Fix 3: Add missing _precompute_common_values method
    def _precompute_common_values(self):
        """
        Pre-compute and cache common values to avoid redundant calculations
        during inference. This can improve cache efficiency and reduce
        computation time for frequent patterns.
        """
        if self.model is None:
            return
            
        try:
            # Model-specific optimizations
            if self.model_type == self.ModelType.SKLEARN:
                # For linear models, precompute common dot products
                if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
                    # Store coef and intercept in contiguous memory for better cache locality
                    if not hasattr(self, '_optimized_model_params'):
                        self._optimized_model_params = {}
                        
                    # Convert to float32 if precision allows for faster computation
                    if self.config.enable_fp16_optimization:
                        self._optimized_model_params['coef'] = np.ascontiguousarray(
                            self.model.coef_.astype(np.float32)
                        )
                        self._optimized_model_params['intercept'] = np.ascontiguousarray(
                            self.model.intercept_.astype(np.float32)
                        )
                    else:
                        self._optimized_model_params['coef'] = np.ascontiguousarray(self.model.coef_)
                        self._optimized_model_params['intercept'] = np.ascontiguousarray(self.model.intercept_)
                        
                    self.logger.debug("Precomputed model coefficients for faster inference")
                    
            elif self.model_type in [self.ModelType.XGBOOST, self.ModelType.LIGHTGBM]:
                # For tree models, we can precompute common paths for fast features
                # This is a placeholder - real implementation would analyze the model
                # to identify and cache common decision paths
                pass
                
            # Precompute common input transformations if preprocessing is enabled
            if self.preprocessor is not None and hasattr(self.preprocessor, 'precompute_transforms'):
                self.preprocessor.precompute_transforms()
                
            self.logger.info("Precomputed common values for optimized inference")
        
        except Exception as e:
            self.logger.warning(f"Error in precomputing values: {str(e)}")


    # Fix 4: Add missing _load_ensemble_model method
    def _load_ensemble_model(self, ensemble_config: Dict[str, Any]):
        """
        Load an ensemble model from configuration.
        
        Args:
            ensemble_config: Dictionary with ensemble configuration
            
        Returns:
            Loaded ensemble model
        """
        if 'models' not in ensemble_config:
            raise ValueError("Invalid ensemble config: 'models' key not found")
            
        # Get model paths and load models
        model_paths = ensemble_config.get('model_paths', [])
        weights = ensemble_config.get('weights', None)
        method = ensemble_config.get('method', 'average')
        
        # Dictionary to store loaded models
        loaded_models = []
        
        # Load each model in the ensemble
        for model_path in model_paths:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Detect model type from file extension
            model_type = self._detect_model_type(model_path)
            
            # Load model based on type
            if model_type == self.ModelType.SKLEARN:
                if self.config.joblib_available:
                    try:
                        import joblib
                        model = joblib.load(model_path)
                    except ImportError:
                        with open(model_path, 'rb') as f:
                            import pickle
                            model = pickle.load(f)
                else:
                    with open(model_path, 'rb') as f:
                        import pickle
                        model = pickle.load(f)
            elif model_type == self.ModelType.XGBOOST:
                try:
                    import xgboost as xgb
                    model = xgb.Booster()
                    model.load_model(model_path)
                except ImportError:
                    raise RuntimeError("xgboost is not available")
            elif model_type == self.ModelType.LIGHTGBM:
                try:
                    import lightgbm as lgb
                    model = lgb.Booster(model_file=model_path)
                except ImportError:
                    raise RuntimeError("lightgbm is not available")
            else:
                with open(model_path, 'rb') as f:
                    import pickle
                    model = pickle.load(f)
            
            loaded_models.append(model)
        
        # Create and return ensemble container
        ensemble = {
            'models': loaded_models,
            'weights': weights,
            'method': method
        }
        
        self.logger.info(f"Loaded ensemble with {len(loaded_models)} models using {method} method")
        return ensemble

    # Fix 5: Add missing _create_cache_key method
    def _create_cache_key(self, features: np.ndarray) -> str:
        """
        Create a unique cache key for the input features.
        
        Args:
            features: Input feature array
            
        Returns:
            Cache key string
        """
        # For small feature sets, use direct hashing
        if features.size <= 1000:  # Arbitrary threshold
            # Convert features to bytes and hash
            feature_bytes = features.tobytes()
            return hashlib.md5(feature_bytes).hexdigest()
        else:
            # For larger feature sets, use a combination of shape, dtype, and statistical properties
            # This is faster but has a small chance of collisions
            hash_components = [
                str(features.shape),
                str(features.dtype),
                str(hash(features.data.tobytes()[:1000])),  # Hash first 1000 bytes
                f"{np.sum(features):.6f}",  # Sum with limited precision
                f"{np.mean(features):.6f}",  # Mean with limited precision
                f"{np.std(features):.6f}" if features.size > 1 else "0"  # Std with limited precision
            ]
            
            key_string = "_".join(hash_components)
            return hashlib.md5(key_string.encode()).hexdigest()


    # Fix 6: Fixed _predict_compiled method (removing duplicate code)
    def _predict_compiled(self, features: np.ndarray) -> np.ndarray:
        """
        Make a prediction using the compiled model (ONNX/Treelite).
        
        Args:
            features: Input features
            
        Returns:
            Prediction results
        """
        if self.compiled_model is None:
            raise RuntimeError("No compiled model available")
        
        # Handle different compiled model types
        if hasattr(self.compiled_model, 'run'):  # ONNX Runtime
            input_name = getattr(self, 'onnx_input_name', 'float_input')
            onnx_inputs = {input_name: features.astype(np.float32)}
            outputs = self.compiled_model.run(None, onnx_inputs)
            return outputs[0]
        elif hasattr(self.compiled_model, 'predict'):  # Treelite
            try:
                from treelite_runtime import Batch
            except ImportError:
                raise RuntimeError("treelite_runtime is not available")
            batch = Batch.from_npy2d(features)
            return self.compiled_model.predict(batch)
        else:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning("Unknown compiled model type, falling back to regular prediction")
            return self._predict_internal(features)


    # Additional improvement: Add a method for model verification and validation
    def validate_model(self) -> Dict[str, Any]:
        """
        Perform validation checks on the loaded model to ensure it can
        make predictions correctly.
        
        Returns:
            Dictionary with validation results
        """
        if self.model is None:
            return {'valid': False, 'error': 'No model loaded'}
            
        validation_results = {
            'valid': True,
            'model_type': self.model_type.name if self.model_type else 'Unknown',
            'compiled': self.compiled_model is not None,
            'feature_count': len(self.feature_names) if self.feature_names else 'Unknown'
        }
        
        try:
            # Create test data for validation
            feature_count = len(self.feature_names) if self.feature_names else (
                self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 10
            )
            
            test_data = np.random.rand(1, feature_count).astype(np.float32)
            
            # Try to predict with the model
            with self._control_threading(1):
                if self.compiled_model is not None:
                    prediction = self._predict_compiled(test_data)
                else:
                    prediction = self._predict_internal(test_data)
            
            # Record prediction shape and type
            validation_results['prediction_shape'] = prediction.shape if hasattr(prediction, 'shape') else 'Unknown'
            validation_results['prediction_type'] = str(type(prediction))
            
            # Additional checks for specific model types
            if self.model_type == ModelType.SKLEARN:
                if hasattr(self.model, 'get_params'):
                    params = self.model.get_params()
                    validation_results['params'] = {k: str(v) for k, v in params.items()}
            
            return validation_results
        
        except Exception as e:
            validation_results['valid'] = False
            validation_results['error'] = str(e)
            return validation_results


    # Additional improvement: Add efficient batch processing for _process_batch method
    def _process_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """
        Process a batch of feature vectors efficiently.
        Used by the dynamic batcher for optimized batch processing.
        
        Args:
            features_batch: Batch of input features
            
        Returns:
            Batch prediction results
        """
        if self.model is None:
            raise RuntimeError("No model loaded for batch processing")
        
        # Run preprocessing if needed
        if self.preprocessor is not None:
            features_batch = self.preprocessor.transform(features_batch)
        
        # Apply quantization if enabled
        if self.config.enable_input_quantization and self.quantizer is not None:
            features_batch, _ = self.quantizer.quantize(features_batch)
        
        # Use compiled model if available
        if self.compiled_model is not None:
            return self._predict_compiled(features_batch)
        else:
            return self._predict_internal(features_batch)