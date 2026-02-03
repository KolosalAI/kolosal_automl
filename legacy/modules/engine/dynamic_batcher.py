"""
Advanced dynamic batching system with priority queues and adaptive sizing.

This module provides intelligent request batching for optimal hardware utilization
and improved throughput in machine learning inference workloads.
"""

import time
import threading
import heapq
import numpy as np
from collections import deque
from typing import List, Dict, Any, Callable
from .prediction_request import PredictionRequest


class DynamicBatcher:
    """
    Advanced dynamic batching system with priority queues, adaptive sizing,
    and intelligent request scheduling for optimal hardware utilization.
    """
    
    def __init__(
        self, 
        batch_processor: Callable,
        max_batch_size: int = 64, 
        max_wait_time_ms: float = 10.0,
        max_queue_size: int = 1000,
        enable_adaptive_sizing: bool = True
    ):
        self.batch_processor = batch_processor
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.enable_adaptive_sizing = enable_adaptive_sizing
        
        # Multi-priority queues for better scheduling
        self.priority_queues = {
            0: [],  # Critical priority
            1: [],  # High priority  
            2: [],  # Normal priority
            3: []   # Low priority
        }
        self.queue_lock = threading.RLock()
        
        # Adaptive batch sizing
        self.current_optimal_batch_size = max_batch_size
        self.batch_performance_history = deque(maxlen=50)
        self.last_adaptation_time = time.monotonic()
        self.adaptation_interval = 5.0  # seconds
        
        # Batch formation trigger
        self.batch_trigger = threading.Event()
        
        # Control flags
        self.running = False
        self.stop_event = threading.Event()
        
        # Worker thread
        self.worker_thread = None
        
        # Enhanced stats
        self.processed_batches = 0
        self.processed_requests = 0
        self.max_observed_queue_size = 0
        self.batch_sizes = deque(maxlen=100)
        self.batch_wait_times = deque(maxlen=100)
        self.batch_processing_times = deque(maxlen=100)
        self.throughput_history = deque(maxlen=20)
        self.max_queue_size = max_queue_size
        
        # Request type tracking for intelligent batching
        self.request_type_stats = {}
        
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
        """Add request to appropriate priority queue with intelligent scheduling"""
        with self.queue_lock:
            # Check total queue size across all priorities
            total_queue_size = sum(len(q) for q in self.priority_queues.values())
            if total_queue_size >= self.max_queue_size:
                return False
                
            # Set timestamp if not already set
            if request.timestamp == 0.0:
                request.timestamp = time.monotonic()
                
            # Ensure valid priority
            priority = max(0, min(3, request.priority))
            
            # Add to appropriate priority queue
            heapq.heappush(self.priority_queues[priority], request)
            
            # Update stats
            if total_queue_size > self.max_observed_queue_size:
                self.max_observed_queue_size = total_queue_size
                
            # Track request characteristics for batching optimization
            self._track_request_type(request)
            
            # Intelligent triggering based on queue state and priorities
            should_trigger = self._should_trigger_batch()
            if should_trigger:
                self.batch_trigger.set()
                
            return True
    
    def _track_request_type(self, request: PredictionRequest):
        """Track request characteristics for intelligent batching"""
        request_signature = f"{request.features.shape}_{request.features.dtype}"
        
        if request_signature not in self.request_type_stats:
            self.request_type_stats[request_signature] = {
                'count': 0,
                'avg_processing_time': 0.0,
                'last_seen': time.monotonic()
            }
        
        stats = self.request_type_stats[request_signature]
        stats['count'] += 1
        stats['last_seen'] = time.monotonic()
    
    def _should_trigger_batch(self) -> bool:
        """Intelligent batch triggering based on queue state and request characteristics"""
        # Check for critical/high priority requests
        critical_high_count = len(self.priority_queues[0]) + len(self.priority_queues[1])
        if critical_high_count > 0:
            return True
        
        # Check if we have enough requests for optimal batch size
        total_requests = sum(len(q) for q in self.priority_queues.values())
        if total_requests >= self.current_optimal_batch_size:
            return True
        
        # Check for request diversity that would benefit from batching
        if self._has_diverse_requests() and total_requests >= 4:
            return True
        
        return False
    
    def _has_diverse_requests(self) -> bool:
        """Check if current queue has diverse request types that benefit from batching"""
        request_types = set()
        for queue in self.priority_queues.values():
            for req in queue:
                if hasattr(req, 'features'):
                    request_types.add(f"{req.features.shape}_{req.features.dtype}")
        return len(request_types) > 1
    
    def _adapt_batch_size(self):
        """Adapt batch size based on recent performance"""
        if not self.enable_adaptive_sizing:
            return
        
        current_time = time.monotonic()
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return
        
        if len(self.batch_performance_history) < 10:
            return
        
        # Analyze recent performance
        recent_performance = list(self.batch_performance_history)[-10:]
        avg_throughput = sum(p['throughput'] for p in recent_performance) / len(recent_performance)
        
        # Calculate performance trend
        if len(self.throughput_history) >= 2:
            recent_throughput = sum(self.throughput_history[-5:]) / min(5, len(self.throughput_history))
            older_throughput = sum(self.throughput_history[-10:-5]) / min(5, len(self.throughput_history) - 5)
            
            if recent_throughput > older_throughput * 1.05:
                # Performance improving, try larger batches
                self.current_optimal_batch_size = min(
                    self.max_batch_size, 
                    int(self.current_optimal_batch_size * 1.1)
                )
            elif recent_throughput < older_throughput * 0.95:
                # Performance degrading, try smaller batches
                self.current_optimal_batch_size = max(
                    4,  # Minimum batch size
                    int(self.current_optimal_batch_size * 0.9)
                )
        
        self.last_adaptation_time = current_time
                
    def _worker_loop(self):
        """Enhanced worker loop with adaptive batching and priority handling"""
        last_batch_time = time.monotonic()
        
        while not self.stop_event.is_set():
            current_time = time.monotonic()
            
            # Adaptive batch size adjustment
            self._adapt_batch_size()
            
            # Batch formation with priority awareness
            batch_requests = []
            batch_wait_time = 0
            should_process = False
            
            with self.queue_lock:
                # Process high-priority requests first
                for priority in sorted(self.priority_queues.keys()):
                    queue = self.priority_queues[priority]
                    
                    while queue and len(batch_requests) < self.current_optimal_batch_size:
                        request = heapq.heappop(queue)
                        
                        # Check request timeout
                        if request.timeout_ms:
                            elapsed_ms = (current_time - request.timestamp) * 1000
                            if elapsed_ms > request.timeout_ms:
                                # Skip expired request
                                continue
                        
                        batch_requests.append(request)
                
                # Determine if we should process the batch
                if batch_requests:
                    oldest_request_time = min(req.timestamp for req in batch_requests)
                    batch_wait_time = (current_time - oldest_request_time) * 1000
                    
                    should_process = (
                        len(batch_requests) >= self.current_optimal_batch_size or
                        batch_wait_time >= self.max_wait_time_ms or
                        any(req.priority <= 1 for req in batch_requests) or  # High priority
                        (current_time - last_batch_time) * 1000 >= self.max_wait_time_ms
                    )
            
            if should_process and batch_requests:
                # Process the batch
                try:
                    batch_start_time = time.monotonic()
                    self._process_batch(batch_requests)
                    batch_processing_time = (time.monotonic() - batch_start_time) * 1000
                    
                    # Update statistics
                    self.processed_batches += 1
                    self.processed_requests += len(batch_requests)
                    self.batch_sizes.append(len(batch_requests))
                    self.batch_wait_times.append(batch_wait_time)
                    self.batch_processing_times.append(batch_processing_time)
                    
                    # Calculate throughput (requests/second)
                    if batch_processing_time > 0:
                        throughput = len(batch_requests) / (batch_processing_time / 1000)
                        self.throughput_history.append(throughput)
                        
                        # Update performance history for adaptation
                        self.batch_performance_history.append({
                            'batch_size': len(batch_requests),
                            'processing_time': batch_processing_time,
                            'throughput': throughput,
                            'wait_time': batch_wait_time
                        })
                    
                    last_batch_time = time.monotonic()
                    
                except Exception as e:
                    # Handle batch processing errors gracefully
                    for request in batch_requests:
                        if request.future and not request.future.done():
                            request.future.set_exception(e)
                
            else:
                # Wait for more requests or timeout
                wait_time = max(0.001, (self.max_wait_time_ms - batch_wait_time) / 1000) if batch_requests else 0.01
                self.batch_trigger.wait(timeout=wait_time)
                self.batch_trigger.clear()
    
    def _process_batch(self, requests: List[PredictionRequest]):
        """Process a batch of requests with optimized data handling"""
        if not requests:
            return
        
        try:
            # Prepare batch input with memory efficiency
            batch_features = []
            feature_shapes = set()
            
            for request in requests:
                batch_features.append(request.features)
                feature_shapes.add(request.features.shape[1:])  # Exclude batch dimension
            
            # Check for uniform shapes (enables optimizations)
            if len(feature_shapes) == 1:
                # Uniform batch - can use vectorized operations
                batch_array = np.vstack(batch_features)
            else:
                # Mixed shapes - handle individually or pad to common shape
                max_shape = max(feature_shapes, key=lambda x: np.prod(x))
                padded_features = []
                for features in batch_features:
                    if features.shape[1:] != max_shape:
                        # Pad smaller arrays
                        pad_width = [(0, 0)]  # No padding for batch dimension
                        for i, (current, target) in enumerate(zip(features.shape[1:], max_shape)):
                            pad_width.append((0, target - current))
                        features = np.pad(features, pad_width, mode='constant')
                    padded_features.append(features)
                batch_array = np.vstack(padded_features)
            
            # Process batch
            results = self.batch_processor(batch_array)
            
            # Distribute results to futures
            if isinstance(results, np.ndarray) and len(results) == len(requests):
                for i, request in enumerate(requests):
                    if request.future and not request.future.done():
                        request.future.set_result(results[i])
            else:
                # Handle non-array results or size mismatches
                for request in requests:
                    if request.future and not request.future.done():
                        request.future.set_result(results)
                        
        except Exception as e:
            # Set exception for all requests in batch
            for request in requests:
                if request.future and not request.future.done():
                    request.future.set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive batching statistics"""
        with self.queue_lock:
            queue_sizes = {f"priority_{p}": len(q) for p, q in self.priority_queues.items()}
            total_queue_size = sum(queue_sizes.values())
            
            avg_batch_size = np.mean(self.batch_sizes) if self.batch_sizes else 0
            avg_wait_time = np.mean(self.batch_wait_times) if self.batch_wait_times else 0
            avg_processing_time = np.mean(self.batch_processing_times) if self.batch_processing_times else 0
            avg_throughput = np.mean(self.throughput_history) if self.throughput_history else 0
            
            return {
                "processed_batches": self.processed_batches,
                "processed_requests": self.processed_requests,
                "current_queue_size": total_queue_size,
                "max_observed_queue_size": self.max_observed_queue_size,
                "queue_sizes_by_priority": queue_sizes,
                "avg_batch_size": avg_batch_size,
                "current_optimal_batch_size": self.current_optimal_batch_size,
                "avg_wait_time_ms": avg_wait_time,
                "avg_processing_time_ms": avg_processing_time,
                "avg_throughput_rps": avg_throughput,
                "request_types_tracked": len(self.request_type_stats),
                "adaptive_sizing_enabled": self.enable_adaptive_sizing
            }


__all__ = ['DynamicBatcher']
