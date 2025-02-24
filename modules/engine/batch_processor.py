from typing import Tuple, Optional, Callable, List, Dict, TypeVar, Generic, Union
from queue import Queue, Empty
from threading import Event, Thread, Lock
import numpy as np
from numpy.typing import NDArray
import time
import logging
from concurrent.futures import Future, ThreadPoolExecutor
import weakref
from collections import deque
import statistics
import threading
from modules.configs import BatchProcessorConfig, BatchProcessingStrategy

# Type variables for generic typing
T = TypeVar('T')
U = TypeVar('U')

class BatchStats:
    """Tracks batch processing statistics."""
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.queue_times = deque(maxlen=window_size)
        self.error_counts = 0
        self.total_processed = 0
        self.lock = Lock()

    def update(self, processing_time: float, batch_size: int, queue_time: float) -> None:
        with self.lock:
            self.processing_times.append(processing_time)
            self.batch_sizes.append(batch_size)
            self.queue_times.append(queue_time)
            self.total_processed += batch_size

    def get_stats(self) -> Dict[str, float]:
        with self.lock:
            return {
                'avg_processing_time': statistics.mean(self.processing_times) if self.processing_times else 0,
                'avg_batch_size': statistics.mean(self.batch_sizes) if self.batch_sizes else 0,
                'avg_queue_time': statistics.mean(self.queue_times) if self.queue_times else 0,
                'error_rate': self.error_counts / max(1, self.total_processed),
                'throughput': sum(self.batch_sizes) / max(1e-6, sum(self.processing_times))
            }

class BatchProcessor(Generic[T, U]):
    """
    Enhanced asynchronous batch processor with adaptive batching, monitoring, and error handling.
    Supports generic input and output types with numpy array processing.
    """

    def __init__(self, config: BatchProcessorConfig, metrics_collector: Optional['MetricsCollector'] = None):
        self.config = config
        self.metrics = metrics_collector
        
        # Initialize queues and events
        self.queue: Queue[Tuple[T, Optional[Future[U]], int]] = Queue(maxsize=config.max_queue_size)
        self.stop_event = Event()
        self.paused_event = Event()
        self.shutdown_event = Event() # New event for cleaner shutdown

        # Thread management
        self.worker_thread: Optional[Thread] = None
        self.worker_lock = Lock()

        # Use a thread pool for processing batches in parallel
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)  # Configurable number of workers

        # Batch size management
        self._current_batch_size = config.initial_batch_size
        self._batch_size_lock = Lock()
        
        # Statistics and monitoring
        self.stats = BatchStats(config.monitoring_window) if config.enable_monitoring else None
        
        # Cleanup management
        self._futures: weakref.WeakSet[Future] = weakref.WeakSet()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def start(self, process_func: Callable[[NDArray[T]], NDArray[U]]) -> None:
        """Start the batch processing thread with enhanced error handling."""
        with self.worker_lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return

            self.stop_event.clear()
            self.paused_event.clear()
            self.shutdown_event.clear()
            self.worker_thread = Thread(
                target=self._wrapped_worker_loop,
                args=(process_func,),
                daemon=True,
                name="BatchProcessorWorker"
            )
            self.worker_thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the batch processor with graceful shutdown."""
        self.stop_event.set()
        self.shutdown_event.set() # Signal shutdown

        # Process remaining items
        self._process_remaining_items()

        # Shutdown the executor
        self.executor.shutdown(wait=True)  # Wait for all tasks to complete

        if self.worker_thread:
            self.worker_thread.join(timeout=timeout)
            if self.worker_thread.is_alive():
                self.logger.warning("Worker thread did not stop within timeout")
            self.worker_thread = None

    def pause(self) -> None:
        """Pause processing temporarily."""
        self.paused_event.set()

    def resume(self) -> None:
        """Resume processing."""
        self.paused_event.clear()

    def enqueue(self, item: T, timeout: Optional[float] = None) -> None:
        """Enqueue item for processing without expecting results."""
        if isinstance(item, np.ndarray):
            self.queue.put((item, None, item.shape[0]), timeout=timeout)
        else:
            self.queue.put((item, None, 1), timeout=timeout)

    def enqueue_predict(self, item: T, timeout: Optional[float] = None) -> Future[U]:
        """Enqueue item for processing and return future for result."""
        future: Future[U] = Future()
        self._futures.add(future)
        
        if isinstance(item, np.ndarray):
            self.queue.put((item, future, item.shape[0]), timeout=timeout)
        else:
            self.queue.put((item, future, 1), timeout=timeout)
            
        return future

    def _wrapped_worker_loop(self, process_func: Callable[[NDArray[T]], NDArray[U]]) -> None:
        """Wrapper for worker loop with error handling and cleanup."""
        try:
            self._worker_loop(process_func)
        except Exception as e:
            self.logger.error("Fatal error in worker loop", exc_info=e)
        finally:
            self._cleanup()

    def _worker_loop(self, process_func: Callable[[NDArray[T]], NDArray[U]]) -> None:
        """Enhanced worker loop with adaptive batching and monitoring."""
        while not self.shutdown_event.is_set(): # Use shutdown_event
            if self.paused_event.is_set():
                time.sleep(0.1)
                continue

            batch_requests: List[Tuple[T, Optional[Future[U]], int]] = []
            start_time = time.monotonic()
            total_samples = 0

            # Collect batch items
            while (
                len(batch_requests) < self._current_batch_size and
                (time.monotonic() - start_time) < self.config.batch_timeout and
                not self.shutdown_event.is_set() # Check shutdown here too
            ):
                try:
                    remaining_time = max(
                        0.0,
                        self.config.batch_timeout - (time.monotonic() - start_time)
                    )
                    req = self.queue.get(timeout=remaining_time)
                    batch_requests.append(req)
                    total_samples += req[2]
                except Empty:
                    break

            if batch_requests:
                self._process_batch(batch_requests, process_func, start_time)
                self._adjust_batch_size(total_samples, time.monotonic() - start_time)

    def _process_batch(
        self,
        batch_requests: List[Tuple[T, Optional[Future[U]], int]],
        process_func: Callable[[NDArray[T]], NDArray[U]],
        start_time: float
    ) -> None:
        """Process a batch of requests with retries and error handling."""

        # Combine batch items outside the retry loop
        try:
            combined = np.vstack([req[0] for req in batch_requests])
        except Exception as e:
            self._handle_batch_error(e, batch_requests)
            return

        retries = 0
        while retries < self.config.max_retries:
            try:
                # Process batch using the thread pool
                future = self.executor.submit(process_func, combined)
                predictions = future.result()  # Wait for the result

                # Distribute results
                start_idx = 0
                for data, future, n_samples in batch_requests:
                    end_idx = start_idx + n_samples
                    result = predictions[start_idx:end_idx]
                    if future is not None:
                        future.set_result(result)
                    start_idx = end_idx
                
                # Update statistics
                if self.stats:
                    self.stats.update(
                        processing_time=time.monotonic() - start_time,
                        batch_size=len(batch_requests),
                        queue_time=start_time - time.monotonic()
                    )
                
                break
            except Exception as e:
                retries += 1
                if retries >= self.config.max_retries:
                    self._handle_batch_error(e, batch_requests)
                else:
                    time.sleep(self.config.retry_delay)

    def _adjust_batch_size(self, total_samples: int, processing_time: float) -> None:
        """Dynamically adjust batch size based on processing metrics."""
        if self.config.processing_strategy != BatchProcessingStrategy.ADAPTIVE:
            return

        with self._batch_size_lock:
            target_time = self.config.batch_timeout * 0.8
            if processing_time > target_time:
                self._current_batch_size = max(
                    self.config.min_batch_size,
                    int(self._current_batch_size * 0.8)
                )
            elif processing_time < target_time * 0.5:
                self._current_batch_size = min(
                    self.config.max_batch_size,
                    int(self._current_batch_size * 1.2)
                )

    def _handle_batch_error(
        self,
        error: Exception,
        batch_requests: List[Tuple[T, Optional[Future[U]], int]]
    ) -> None:
        """Handle batch processing errors."""
        self.logger.error(f"Batch processing error: {error}", exc_info=error)
        if self.stats:
            self.stats.error_counts += len(batch_requests)
        
        for _, future, _ in batch_requests:
            if future is not None:
                future.set_exception(error)

    def _process_remaining_items(self) -> None:
        """Process remaining items in queue during shutdown."""
        remaining_items = []
        while True:
            try:
                remaining_items.append(self.queue.get_nowait())
            except Empty:
                break
        
        if remaining_items:
            self.logger.info(f"Processing {len(remaining_items)} remaining items")
            for item, future, _ in remaining_items:
                if future is not None:
                    future.cancel()

    def _cleanup(self) -> None:
        """Cleanup resources and cancel pending futures."""
        with self.worker_lock:
            for future in self._futures:
                if not future.done():
                    future.cancel()
            self._futures.clear()

    def get_stats(self) -> Optional[Dict[str, float]]:
        """Get current processing statistics."""
        return self.stats.get_stats() if self.stats else None

    @property
    def is_running(self) -> bool:
        """Check if the processor is running."""
        return bool(self.worker_thread and self.worker_thread.is_alive())

    @property
    def current_batch_size(self) -> int:
        """Get current batch size."""
        return self._current_batch_size
