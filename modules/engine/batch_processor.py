from typing import Tuple, Optional, Callable, List, Dict, TypeVar, Generic, Union, Any, Set
from queue import Queue, Empty, PriorityQueue, Full
from threading import Event, Thread, Lock, RLock
import numpy as np
from numpy.typing import NDArray, ArrayLike
import time
import logging
from concurrent.futures import Future, ThreadPoolExecutor
import weakref
from collections import deque
import statistics
import threading
import gc

# Try to import psutil for memory monitoring
try:
    import psutil
except ImportError:
    psutil = None

# local custom modules imports
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import BatchProcessorConfig, BatchProcessingStrategy, BatchPriority, PrioritizedItem

# Type variables for generic typing
T = TypeVar('T')
U = TypeVar('U')

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


class BatchProcessor(Generic[T, U]):
    """
    Enhanced asynchronous batch processor with adaptive batching, monitoring, priority queuing,
    and efficient resource management.

    Features:
    - Priority-based processing
    - Adaptive batch sizing based on system load
    - Efficient memory usage for large arrays
    - Comprehensive error handling and recovery
    - Detailed performance metrics
    - Graceful shutdown with cleanup
    """

    def __init__(self, config: BatchProcessorConfig, metrics_collector: Optional[Any] = None):
        """
        Initialize the batch processor.
        
        Args:
            config: Configuration parameters
            metrics_collector: Optional external metrics collection system
        """
        self.config = config
        self.metrics = metrics_collector
        
        # Initialize queues with priority support
        self._queue_class = PriorityQueue if config.enable_priority_queue else Queue
        self.queue: Union[Queue, PriorityQueue] = self._queue_class(maxsize=config.max_queue_size)
        
        # Control events
        self.stop_event = Event()
        self.paused_event = Event()
        self.shutdown_event = Event()
        self.batch_ready_event = Event()  # Signal when batch is ready for processing

        # Thread management
        self.worker_thread: Optional[Thread] = None
        self.worker_lock = RLock()
        
        # Batch data management
        self._current_batch_size = config.initial_batch_size
        self._batch_size_lock = RLock()
        self._max_batch_memory = config.max_batch_memory_mb * 1024 * 1024 if config.max_batch_memory_mb else None
        
        # Dynamic system load tracking
        self._system_load = 0.5  # Initial normalized load (0-1)
        self._load_alpha = 0.2  # Smoothing factor for load updates
        
        # Optimize thread pool based on workload
        worker_count = min(
            config.max_workers,
            max(1, os.cpu_count() or 4)
        )
        self.executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="BatchWorker"
        )
        
        # Statistics and monitoring
        self.stats = BatchStats(config.monitoring_window) if config.enable_monitoring else None
        
        # Track all futures for cleanup
        self._futures: weakref.WeakSet[Future] = weakref.WeakSet()
        self._active_batches: Set[int] = set()  # Track batch IDs currently processing
        self._next_batch_id = 0
        self._batch_id_lock = Lock()
        
        # Rate limiting
        self._last_batch_time = 0
        self._min_batch_interval = config.min_batch_interval
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if config.debug_mode else logging.INFO
        self.logger.setLevel(log_level)
        
        # Memory optimization
        self._enable_memory_optimization = config.enable_memory_optimization
        
        # Health check
        self._health_monitor_thread = None
        if config.enable_health_monitoring:
            self._start_health_monitor()

    def start(self, process_func: Callable[[NDArray[T]], NDArray[U]]) -> None:
        """
        Start the batch processing thread.
        
        Args:
            process_func: Function that processes batched inputs
        """
        with self.worker_lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return

            self.stop_event.clear()
            self.paused_event.clear()
            self.shutdown_event.clear()
            
            self._last_batch_time = time.monotonic()
            
            # Create and start the worker thread
            self.worker_thread = Thread(
                target=self._wrapped_worker_loop,
                args=(process_func,),
                daemon=True,
                name=f"BatchProcessor-{id(self)}"
            )
            self.worker_thread.start()
            
            self.logger.info("Batch processor started")

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """
        Stop the batch processor with graceful shutdown.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        self.logger.info("Stopping batch processor...")
        self.stop_event.set()
        self.shutdown_event.set()
        
        # Signal any waiting threads
        self.batch_ready_event.set()
        
        # Process remaining items with timeout
        timeout_time = time.monotonic() + (timeout or 5.0)
        try:
            self._process_remaining_items(timeout_time - time.monotonic())
        except Exception as e:
            self.logger.error(f"Error processing remaining items: {e}")

        # Wait for worker thread to finish
        if self.worker_thread:
            remaining_time = max(0.1, timeout_time - time.monotonic())
            self.worker_thread.join(timeout=remaining_time)
            if self.worker_thread.is_alive():
                self.logger.warning("Worker thread did not stop within timeout")
            self.worker_thread = None

        # Stop health monitor if running
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=1.0)
            
        # Clean shutdown of thread pool
        self.executor.shutdown(wait=True)
        
        self.logger.info("Batch processor stopped")

    def pause(self) -> None:
        """Pause processing temporarily."""
        if not self.paused_event.is_set():
            self.logger.info("Pausing batch processor")
            self.paused_event.set()

    def resume(self) -> None:
        """Resume processing after pause."""
        if self.paused_event.is_set():
            self.logger.info("Resuming batch processor")
            self.paused_event.clear()
            # Signal batch processing to resume immediately
            self.batch_ready_event.set()

    def enqueue(self, item: T, timeout: Optional[float] = None, 
                priority: BatchPriority = BatchPriority.NORMAL) -> None:
        """
        Enqueue item for processing without expecting results.
        
        Args:
            item: Item to process
            timeout: Maximum time to wait if queue is full
            priority: Processing priority for this item
        """
        # Create a null future - we're not tracking the result
        if self._queue_is_prioritized():
            queue_item = PrioritizedItem(
                priority=priority.value,
                timestamp=time.monotonic(),
                item=(item, None, self._get_item_size(item))
            )
            self.queue.put(queue_item, timeout=timeout)
        else:
            self.queue.put((item, None, self._get_item_size(item)), timeout=timeout)
        
        # Signal that a new item is available
        self.batch_ready_event.set()

    def enqueue_predict(self, item: T, timeout: Optional[float] = None,
                       priority: BatchPriority = BatchPriority.NORMAL) -> Future[U]:
        """
        Enqueue item for processing and return future for result.
        
        Args:
            item: Item to process
            timeout: Maximum time to wait if queue is full
            priority: Processing priority for this item
            
        Returns:
            Future object that will contain the result
        """
        # Validate we're running
        if self.shutdown_event.is_set():
            future: Future[U] = Future()
            future.set_exception(RuntimeError("BatchProcessor is shutting down"))
            return future
            
        # Create and track the future
        future: Future[U] = Future()
        # Add creation time attribute for latency tracking
        future.creation_time = time.monotonic()
        self._futures.add(future)
        
        # Get item size for batching decisions
        item_size = self._get_item_size(item)
        
        try:
            # Put in priority queue if enabled
            if self._queue_is_prioritized():
                queue_item = PrioritizedItem(
                    priority=priority.value,
                    timestamp=time.monotonic(),
                    item=(item, future, item_size)
                )
                self.queue.put(queue_item, timeout=timeout)
            else:
                self.queue.put((item, future, item_size), timeout=timeout)
                
            # Signal that a new item is available
            self.batch_ready_event.set()
            
            return future
        except Full:
            future.set_exception(RuntimeError("Queue is full, request rejected"))
            return future
        except Exception as e:
            future.set_exception(e)
            return future

    def update_batch_size(self, new_size: int) -> None:
        """
        Update the current batch size.
        
        Args:
            new_size: New target batch size
        """
        with self._batch_size_lock:
            self._current_batch_size = max(
                self.config.min_batch_size,
                min(new_size, self.config.max_batch_size)
            )
            self.logger.debug(f"Batch size updated to {self._current_batch_size}")

    def _queue_is_prioritized(self) -> bool:
        """Check if the queue is using priority-based queuing."""
        return isinstance(self.queue, PriorityQueue)

    def _get_item_size(self, item: T) -> int:
        """
        Get the size of an item for batching decisions.
        
        Args:
            item: The item to measure
            
        Returns:
            Size metric (samples count or other relevant measure)
        """
        if isinstance(item, np.ndarray):
            return item.shape[0]  # Number of samples
        elif hasattr(item, '__len__'):
            return len(item)  # Length for sequence-like objects
        return 1  # Default size for scalar items

    def _get_batch_memory_size(self, item: T) -> int:
        """
        Estimate memory size of an item in bytes.
        
        Args:
            item: Item to measure
            
        Returns:
            Estimated memory size in bytes
        """
        if isinstance(item, np.ndarray):
            return item.nbytes
        # For other types, make a conservative estimate
        return sys.getsizeof(item)

    def _wrapped_worker_loop(self, process_func: Callable[[NDArray[T]], NDArray[U]]) -> None:
        """
        Wrapper for worker loop with error handling and cleanup.
        
        Args:
            process_func: Function to process batched data
        """
        try:
            self._worker_loop(process_func)
        except Exception as e:
            self.logger.error(f"Fatal error in worker loop: {e}", exc_info=True)
            # Record the error
            if self.stats:
                self.stats.record_error()
        finally:
            self._cleanup()

    def _worker_loop(self, process_func: Callable[[NDArray[T]], NDArray[U]]) -> None:
        """
        Enhanced worker loop with adaptive batching and monitoring.
        
        Args:
            process_func: Function to process batched data
        """
        while not self.shutdown_event.is_set():
            # Handle pause state
            if self.paused_event.is_set():
                # Wait for resume signal or shutdown
                self.batch_ready_event.wait(timeout=0.1)
                self.batch_ready_event.clear()
                continue

            # Rate limiting - ensure minimum interval between batches
            if self._min_batch_interval > 0:
                time_since_last = time.monotonic() - self._last_batch_time
                if time_since_last < self._min_batch_interval:
                    time.sleep(max(0, self._min_batch_interval - time_since_last))

            # Collect batch items with timeout
            batch_requests = self._collect_batch()
            
            # Process batch if we have items
            if batch_requests:
                self._last_batch_time = time.monotonic()
                self._process_batch(batch_requests, process_func)
            else:
                # If queue is empty, wait for signal or timeout
                self.batch_ready_event.wait(timeout=0.01)
                self.batch_ready_event.clear()

    def _collect_batch(self) -> List[Tuple[T, Optional[Future[U]], int]]:
        """
        Collect items for a batch with adaptive sizing.
        
        Returns:
            List of items to process in a batch
        """
        batch_requests: List[Tuple[T, Optional[Future[U]], int]] = []
        start_time = time.monotonic()
        current_batch_size = 0
        total_memory_size = 0
        
        # Get the current target batch size (thread-safe)
        with self._batch_size_lock:
            target_batch_size = self._current_batch_size
            
        # Collect batch items
        while (
            len(batch_requests) < target_batch_size and
            (time.monotonic() - start_time) < self.config.batch_timeout and
            not self.shutdown_event.is_set() and
            not self.paused_event.is_set()
        ):
            try:
                # Calculate remaining timeout
                remaining_time = max(
                    0.001,  # 1ms minimum to avoid CPU spinning
                    self.config.batch_timeout - (time.monotonic() - start_time)
                )
                
                # Get next item from queue
                if self._queue_is_prioritized():
                    prioritized_item = self.queue.get(timeout=remaining_time)
                    item_tuple = prioritized_item.item
                else:
                    item_tuple = self.queue.get(timeout=remaining_time)
                
                batch_requests.append(item_tuple)
                current_batch_size += item_tuple[2]
                
                # If memory limits are enabled, check batch memory size
                if self._max_batch_memory is not None:
                    item_memory = self._get_batch_memory_size(item_tuple[0])
                    total_memory_size += item_memory
                    
                    # Stop collecting if we exceed memory limit
                    if total_memory_size > self._max_batch_memory:
                        self.logger.debug(f"Batch memory limit reached: {total_memory_size/1024/1024:.2f}MB")
                        break
                        
                # Check if we have a full batch based on samples count
                if current_batch_size >= target_batch_size:
                    break
                    
            except Empty:
                # No more items available in queue
                break
                
        # Update queue length metrics
        if self.stats:
            self.stats.update(
                processing_time=0,  # Will be updated after processing
                batch_size=len(batch_requests),
                queue_time=0,  # Will be updated after processing
                queue_length=self.queue.qsize() if hasattr(self.queue, 'qsize') else 0
            )
            
        return batch_requests

    def _process_batch(
        self,
        batch_requests: List[Tuple[T, Optional[Future[U]], int]],
        process_func: Callable[[NDArray[T]], NDArray[U]]
    ) -> None:
        """
        Process a batch of requests with retries and error handling.
        
        Args:
            batch_requests: List of items to process
            process_func: Function to process the batch
        """
        if not batch_requests:
            return

        # Generate batch ID for tracking
        batch_id = self._get_next_batch_id()
        batch_size = len(batch_requests)
        self._active_batches.add(batch_id)
        
        start_time = time.monotonic()
        
        # Calculate average queue time
        queue_time = 0
        queue_time_count = 0
        for _, future, _ in batch_requests:
            if future is not None and hasattr(future, 'creation_time'):
                queue_time += (start_time - future.creation_time)
                queue_time_count += 1
        
        avg_queue_time = queue_time / max(1, queue_time_count) if queue_time_count > 0 else 0
        
        self.logger.debug(f"Processing batch {batch_id} with {batch_size} items")

        # Process different types of batches appropriately
        try:
            # For numpy arrays, use vstack for efficient batching
            if batch_size > 0 and isinstance(batch_requests[0][0], np.ndarray):
                self._process_numpy_batch(batch_requests, process_func, batch_id, start_time, avg_queue_time)
            else:
                # For non-numpy types, process individually
                self._process_generic_batch(batch_requests, process_func, batch_id, start_time, avg_queue_time)
                
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
            self._handle_batch_error(e, batch_requests)
        finally:
            # Clean up batch tracking
            self._active_batches.discard(batch_id)
            
            # Update system load estimation based on processing time
            processing_time = time.monotonic() - start_time
            self._update_system_load(processing_time)
            
            # Opportunistic garbage collection if memory optimization is enabled
            if self._enable_memory_optimization and batch_size > self.config.gc_batch_threshold:
                gc.collect()

    def _process_numpy_batch(
        self,
        batch_requests: List[Tuple[T, Optional[Future[U]], int]],
        process_func: Callable[[NDArray[T]], NDArray[U]],
        batch_id: int,
        start_time: float,
        queue_time: float
    ) -> None:
        """
        Process a batch of numpy arrays.
        
        Args:
            batch_requests: List of items to process
            process_func: Function to process the batch
            batch_id: Unique ID for this batch
            start_time: Time when batch processing started
            queue_time: Average time items spent in queue
        """
        # Extract the numpy arrays
        try:
            arrays = [req[0] for req in batch_requests]
            
            # Check if arrays can be stacked (same shape except first dimension)
            shapes_compatible = all(arr.shape[1:] == arrays[0].shape[1:] for arr in arrays)
            
            if shapes_compatible:
                # Efficiently combine arrays with pre-allocation if possible
                if self._enable_memory_optimization:
                    # Calculate total first dimension
                    total_dim = sum(arr.shape[0] for arr in arrays)
                    # Get shape of first array
                    rest_dims = arrays[0].shape[1:]
                    # Create pre-allocated array
                    combined = np.empty((total_dim, *rest_dims), dtype=arrays[0].dtype)
                    
                    # Copy arrays into pre-allocated array
                    pos = 0
                    for arr in arrays:
                        arr_len = arr.shape[0]
                        combined[pos:pos+arr_len] = arr
                        pos += arr_len
                else:
                    # Standard vstack
                    combined = np.vstack(arrays)
            else:
                # If shapes aren't compatible, process individually
                self._process_generic_batch(batch_requests, process_func, batch_id, start_time, queue_time)
                return
                
            # Process the combined array
            retries = 0
            while retries <= self.config.max_retries:
                try:
                    # Process the batch
                    predictions = process_func(combined)
                    
                    # Distribute results back to futures
                    start_idx = 0
                    for i, (data, future, n_samples) in enumerate(batch_requests):
                        if future is not None and not future.done():
                            end_idx = start_idx + n_samples
                            result = predictions[start_idx:end_idx]
                            future.set_result(result)
                        start_idx += n_samples
                    
                    # Update statistics
                    processing_time = time.monotonic() - start_time
                    if self.stats:
                        self.stats.update(
                            processing_time=processing_time,
                            batch_size=len(batch_requests),
                            queue_time=queue_time,
                            latency=processing_time + queue_time,
                            queue_length=self.queue.qsize() if hasattr(self.queue, 'qsize') else 0
                        )
                        
                    self.logger.debug(
                        f"Completed batch {batch_id} in {processing_time*1000:.2f}ms with {len(batch_requests)} items"
                    )
                    break
                    
                except Exception as e:
                    retries += 1
                    if retries > self.config.max_retries:
                        raise
                    self.logger.warning(f"Retry {retries}/{self.config.max_retries} for batch {batch_id}: {e}")
                    time.sleep(self.config.retry_delay)
                    
        except Exception as e:
            self.logger.error(f"Error in numpy batch {batch_id}: {e}", exc_info=True)
            self._handle_batch_error(e, batch_requests)

    def _process_generic_batch(
        self,
        batch_requests: List[Tuple[T, Optional[Future[U]], int]],
        process_func: Callable[[T], U],
        batch_id: int,
        start_time: float,
        queue_time: float
    ) -> None:
        """
        Process a batch of generic items individually.
        
        Args:
            batch_requests: List of items to process
            process_func: Function to process the items
            batch_id: Unique ID for this batch
            start_time: Time when batch processing started
            queue_time: Average time items spent in queue
        """
        futures = []
        
        # Submit each item to the thread pool
        for data, result_future, _ in batch_requests:
            if result_future is not None and not result_future.done():
                # Submit to thread pool and track future
                future = self.executor.submit(process_func, data)
                futures.append((future, result_future))
        
        # Wait for all processing to complete
        for executor_future, result_future in futures:
            try:
                result = executor_future.result(timeout=self.config.item_timeout)
                if not result_future.done():
                    result_future.set_result(result)
            except Exception as e:
                if not result_future.done():
                    result_future.set_exception(e)
                
                # Record error
                if self.stats:
                    self.stats.record_error()
        
        # Update statistics
        processing_time = time.monotonic() - start_time
        if self.stats:
            self.stats.update(
                processing_time=processing_time,
                batch_size=len(batch_requests),
                queue_time=queue_time,
                latency=processing_time + queue_time,
                queue_length=self.queue.qsize() if hasattr(self.queue, 'qsize') else 0
            )
            
        self.logger.debug(
            f"Completed generic batch {batch_id} in {processing_time*1000:.2f}ms with {len(batch_requests)} items"
        )

    def _get_next_batch_id(self) -> int:
        """Get a unique batch ID."""
        with self._batch_id_lock:
            batch_id = self._next_batch_id
            self._next_batch_id += 1
            return batch_id

    def _update_system_load(self, processing_time: float) -> None:
        """
        Update system load estimation based on processing time.
        
        Args:
            processing_time: Time taken to process the last batch
        """
        # Calculate normalized load (0-1) based on processing time relative to timeout
        normalized_load = min(1.0, processing_time / max(0.001, self.config.batch_timeout))
        
        # Update running average with smoothing
        self._system_load = (1 - self._load_alpha) * self._system_load + self._load_alpha * normalized_load
        
        # Adjust batch size based on system load if adaptive strategy is enabled
        if (self.config.processing_strategy == BatchProcessingStrategy.ADAPTIVE and 
            self.config.enable_adaptive_batching):
            self._adjust_batch_size_adaptive()

    def _adjust_batch_size_adaptive(self) -> None:
        """Dynamically adjust batch size based on system load."""
        with self._batch_size_lock:
            current_size = self._current_batch_size
            
            # If system load is high, decrease batch size
            if self._system_load > 0.8:
                new_size = int(current_size * 0.8)  # Reduce by 20%
                self.logger.debug(f"System load high ({self._system_load:.2f}), reducing batch size")
            # If system load is low, increase batch size
            elif self._system_load < 0.4:
                new_size = int(current_size * 1.2)  # Increase by 20%
                self.logger.debug(f"System load low ({self._system_load:.2f}), increasing batch size")
            else:
                # Load is moderate, maintain current size
                return
                
            # Apply limits
            new_size = max(
                self.config.min_batch_size,
                min(new_size, self.config.max_batch_size)
            )
            
            # Only update if there's a meaningful change
            if new_size != current_size:
                self._current_batch_size = new_size
                self.logger.info(f"Adjusted batch size to {new_size} based on system load: {self._system_load:.2f}")

    def _handle_batch_error(self, error: Exception, batch_requests: List[Tuple[T, Optional[Future[U]], int]]) -> None:
        """
        Handle errors in batch processing.
        
        Args:
            error: The exception that occurred
            batch_requests: The batch that caused the error
        """
        # Record the error in statistics
        if self.stats:
            self.stats.record_error(len(batch_requests))
        
        # Mark all futures as failed
        for _, future, _ in batch_requests:
            if future is not None and not future.done():
                future.set_exception(error)
                
        self.logger.error(f"Batch processing error: {error}", exc_info=True)
        
        # If configured to break large batches on failure, adjust batch size
        if self.config.reduce_batch_on_failure and len(batch_requests) > self.config.min_batch_size:
            with self._batch_size_lock:
                new_size = max(
                    self.config.min_batch_size,
                    int(self._current_batch_size * 0.7)  # Reduce by 30%
                )
                if new_size < self._current_batch_size:
                    self._current_batch_size = new_size
                    self.logger.info(f"Reduced batch size to {new_size} after error")

    def _process_remaining_items(self, timeout: float) -> None:
        """
        Process remaining items in the queue before shutdown.
        
        Args:
            timeout: Maximum time to spend processing remaining items
        """
        start_time = time.monotonic()
        processed = 0
        
        while not self.queue.empty() and time.monotonic() - start_time < timeout:
            try:
                # Get a single item
                if self._queue_is_prioritized():
                    prioritized_item = self.queue.get(block=False)
                    item, future, _ = prioritized_item.item
                else:
                    item, future, _ = self.queue.get(block=False)
                
                if future is not None and not future.done():
                    future.set_exception(RuntimeError("BatchProcessor shutting down"))
                
                processed += 1
            except Empty:
                break
            except Exception as e:
                self.logger.error(f"Error during queue drain: {e}")
        
        self.logger.info(f"Processed {processed} remaining items during shutdown")

    def _start_health_monitor(self) -> None:
        """Start a background thread to monitor processor health."""
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            return
            
        self._health_monitor_thread = Thread(
            target=self._health_check_loop,
            daemon=True,
            name="BatchProcessorHealthMonitor"
        )
        self._health_monitor_thread.start()
        
    def _health_check_loop(self) -> None:
        """Background thread for monitoring system health."""
        check_interval = self.config.health_check_interval
        
        while not self.shutdown_event.is_set():
            try:
                # Check for stuck batches
                self._check_stuck_batches()
                
                # Check memory usage
                if self._enable_memory_optimization:
                    self._check_memory_usage()
                    
                # Check queue growth
                self._check_queue_growth()
                
                # Sleep until next check
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                time.sleep(max(1.0, check_interval / 2))  # Reduced sleep on error
    
    def _check_stuck_batches(self) -> None:
        """Check for batches that have been processing too long."""
        # Implementation would track batch start times and alert if any exceed thresholds
        pass
        
    def _check_memory_usage(self) -> None:
        """Monitor memory usage and take action if it's too high."""
        try:
            # This is a simplified approach - a production version would use 
            # a memory profiling library for more accurate measurements
            if psutil and hasattr(psutil, 'Process'):
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                if memory_percent > self.config.memory_warning_threshold:
                    self.logger.warning(
                        f"High memory usage: {memory_percent:.1f}% ({memory_info.rss / 1024 / 1024:.1f} MB)"
                    )
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Reduce batch size if very high memory
                    if memory_percent > self.config.memory_critical_threshold:
                        with self._batch_size_lock:
                            new_size = max(
                                self.config.min_batch_size,
                                int(self._current_batch_size * 0.6)  # Reduce by 40%
                            )
                            if new_size < self._current_batch_size:
                                self._current_batch_size = new_size
                                self.logger.info(
                                    f"Reduced batch size to {new_size} due to high memory usage"
                                )
        except (ImportError, NameError, AttributeError):
            # psutil not available
            pass
        except Exception as e:
            self.logger.error(f"Error checking memory usage: {e}")
    
    def _check_queue_growth(self) -> None:
        """Monitor queue size and take action if it's growing too quickly."""
        try:
            queue_size = self.queue.qsize() if hasattr(self.queue, 'qsize') else -1
            
            if queue_size > self.config.queue_warning_threshold:
                self.logger.warning(f"Queue size growing: {queue_size} items")
                
                # If queue is severely backed up, consider increasing batch size
                if queue_size > self.config.queue_critical_threshold:
                    with self._batch_size_lock:
                        new_size = min(
                            self.config.max_batch_size,
                            int(self._current_batch_size * 1.5)  # Increase by 50%
                        )
                        if new_size > self._current_batch_size:
                            self._current_batch_size = new_size
                            self.logger.info(
                                f"Increased batch size to {new_size} to handle queue backlog"
                            )
        except Exception as e:
            self.logger.error(f"Error checking queue growth: {e}")
    
    def _cleanup(self) -> None:
        """Clean up resources during shutdown."""
        # Fail any pending futures
        for future in list(self._futures):
            if not future.done():
                future.set_exception(RuntimeError("BatchProcessor shutting down"))
        
        # Clear data structures
        self._futures.clear()
        self._active_batches.clear()
        
        # Signal all waiting threads
        self.batch_ready_event.set()
        
        # Final log message
        self.logger.info("BatchProcessor cleanup complete")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary of statistics or empty dict if monitoring disabled
        """
        if not self.stats:
            return {}
            
        basic_stats = self.stats.get_stats()
        
        # Add additional runtime stats
        extended_stats = {
            **basic_stats,
            "current_batch_size": self._current_batch_size,
            "system_load": self._system_load,
            "active_batches": len(self._active_batches),
            "queue_size": self.queue.qsize() if hasattr(self.queue, 'qsize') else -1,
            "is_paused": self.paused_event.is_set(),
            "is_stopping": self.stop_event.is_set(),
        }
        
        return extended_stats
