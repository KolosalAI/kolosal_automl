import os
import time
import logging
import pickle
import hashlib
import json
import threading
import gc
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Iterator
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue
from dataclasses import dataclass
import functools
from collections import deque
import heapq
from contextlib import contextmanager

# Import optimization modules
from .jit_compiler import get_global_jit_compiler
from .mixed_precision import get_global_mixed_precision_manager
from .streaming_pipeline import get_global_streaming_pipeline

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
from .simd_optimizer import SIMDOptimizer
from .multi_level_cache import MultiLevelCache
from .prediction_request import PredictionRequest
from .dynamic_batcher import DynamicBatcher
from .memory_pool import MemoryPool
from .performance_metrics import PerformanceMetrics
from ..configs import (
    QuantizationConfig, BatchProcessorConfig, BatchProcessingStrategy,
    BatchPriority, PreprocessorConfig, NormalizationType, 
    QuantizationMode, ModelType, EngineState, InferenceEngineConfig
)

# SIMD Optimizations for vectorized operations
NUMBA_AVAILABLE = False
NUMBA_ERROR = None

try:
    import numba
    from numba import njit, prange
    
    # Test basic numba functionality
    @njit
    def _test_numba_inference():
        return 1.0
    
    try:
        _test_numba_inference()
        NUMBA_AVAILABLE = True
    except Exception as e:
        NUMBA_AVAILABLE = False
        NUMBA_ERROR = f"Numba test failed: {str(e)}"
        
except ImportError as e:
    NUMBA_AVAILABLE = False
    NUMBA_ERROR = f"Numba import failed: {str(e)}"
except Exception as e:
    NUMBA_AVAILABLE = False
    NUMBA_ERROR = f"Numba initialization failed: {str(e)}"

# Define fallback decorators if numba is not available
if not NUMBA_AVAILABLE:
    def njit(*args, **kwargs):
        """Fallback njit decorator that does nothing."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    
    def prange(*args, **kwargs):
        """Fallback prange that uses regular range."""
        return range(*args, **kwargs)

















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
        
        # Initialize memory pool for buffer reuse with NUMA awareness
        self.memory_pool = MemoryPool(max_buffers=64, numa_aware=True)
        
        # Initialize multi-level caching system
        self.advanced_cache = MultiLevelCache(
            l1_size=100,
            l2_size=500, 
            l3_size=1000,
            enable_prediction=True
        ) if self.config.enable_request_deduplication else None
        
        # Initialize SIMD optimizer for vectorized operations  
        try:
            self.simd_optimizer = SIMDOptimizer()
            self.logger.info(f"SIMD optimizer initialized")
        except Exception as e:
            self.logger.warning(f"SIMD optimizer not available, using numpy fallback: {str(e)}")
            self.simd_optimizer = None
        
        # Feature transformation cache (separate from result cache) - kept for backward compatibility
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
        
        # Initialize optimization components
        self._init_optimization_components()
        
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
        # Set logging level based on debug mode
        level = logging.DEBUG if self.config.debug_mode else logging.INFO
        
        try:
            from modules.logging_config import get_logger
            logger = get_logger(
                name=f"InferenceEngine",
                level=level,
                log_file="inference_engine.log",
                enable_console=True
            )
        except ImportError:
            # Fallback to basic logging if centralized logging not available
            logger = logging.getLogger(f"{__name__}.InferenceEngine")
            logger.setLevel(level)
            
            # Only add handlers if none exist
            if not logger.handlers:
                # Safe console handler that handles closed streams
                try:
                    from modules.logging_config import SafeStreamHandler
                    console_handler = SafeStreamHandler()
                except ImportError:
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
    
    def _control_threading(self, max_threads: int = 1):
        """Context manager to control threading for numerical libraries during prediction."""
        from contextlib import contextmanager
        
        @contextmanager
        def thread_controller():
            if THREADPOOLCTL_AVAILABLE:
                # Control threadpoolctl libraries
                with threadpoolctl.threadpool_limits(limits=max_threads, user_api="blas"):
                    with threadpoolctl.threadpool_limits(limits=max_threads, user_api="openmp"):
                        yield
            else:
                # Fallback - just yield without thread control
                yield
        
        return thread_controller()
    
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
        
        # Check cache using advanced multi-level cache if available - FAST PATH
        cache_key = None
        if self.advanced_cache is not None or self.result_cache is not None:
            cache_key = self._create_cache_key(features)
            
            # Try advanced cache first
            if self.advanced_cache is not None:
                hit, cached_result = self.advanced_cache.get(cache_key, data_type='result')
                if hit:
                    self.metrics.record_cache_hit()
                    return True, cached_result, {"cached": True, "cache_hit": True, "cache_level": "advanced"}
            
            # Fallback to simple cache
            elif self.result_cache is not None:
                hit, cached_result = self.result_cache.get(cache_key)
                if hit:
                    self.metrics.record_cache_hit()
                    return True, cached_result, {"cached": True, "cache_hit": True, "cache_level": "simple"}
            
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
            # Apply mixed precision optimization for numerical features
            if hasattr(self, 'mixed_precision_manager') and hasattr(self.config, 'enable_mixed_precision') and self.config.enable_mixed_precision:
                features = self.mixed_precision_manager.optimize_numpy_precision(features, target_precision='auto')
            
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
            
            # Use optimized prediction with all enhancements
            with self._control_threading(1):  # Limit threads during prediction
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
            
            # Cache result using advanced cache if available
            if cache_key:
                if self.advanced_cache is not None:
                    self.advanced_cache.set(cache_key, predictions, data_type='result', priority='high')
                elif self.result_cache is not None:
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
        
        # Get the model type early
        model_type = self.model_type
        
        # Apply JIT compilation for frequently called prediction methods
        if hasattr(self, 'jit_compiler') and hasattr(self.config, 'enable_jit_compilation') and self.config.enable_jit_compilation:
            # Use JIT compiler for model prediction if supported
            if model_type == ModelType.SKLEARN:
                # JIT compile prediction for sklearn models
                if hasattr(self.model, 'predict_proba') and self.config.return_probabilities:
                    return self.jit_compiler.compile_if_hot(self.model.predict_proba, features)
                else:
                    return self.jit_compiler.compile_if_hot(self.model.predict, features)
        
        # Create a clean, contiguous copy of features if needed
        # This improves memory access patterns for vectorized operations
        if not features.flags.c_contiguous:
            features = np.ascontiguousarray(features)
            
        # Choose prediction method based on model type
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
    
    def _process_batch(self, batch_requests):
        """
        Process a batch of prediction requests.
        
        Args:
            batch_requests: List of request dictionaries
            
        Returns:
            List of prediction results
        """
        try:
            # Extract features from batch requests
            features_list = []
            request_ids = []
            
            for request in batch_requests:
                if isinstance(request, dict):
                    features_list.append(request.get('features'))
                    request_ids.append(request.get('id', None))
                else:
                    # Assume request is features directly
                    features_list.append(request)
                    request_ids.append(None)
            
            # Stack features into batch
            if features_list:
                batch_features = np.vstack(features_list)
                
                # Run batch prediction
                success, predictions, metadata = self.predict(batch_features)
                
                # Package results
                results = []
                if success and predictions is not None:
                    for i, (pred, req_id) in enumerate(zip(predictions, request_ids)):
                        result = {
                            'success': True,
                            'prediction': pred,
                            'metadata': metadata,
                            'request_id': req_id
                        }
                        results.append(result)
                else:
                    # All requests failed
                    for req_id in request_ids:
                        result = {
                            'success': False,
                            'prediction': None,
                            'metadata': metadata,
                            'request_id': req_id
                        }
                        results.append(result)
                
                return results
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            # Return error results for all requests
            results = []
            for i, request in enumerate(batch_requests):
                req_id = request.get('id', None) if isinstance(request, dict) else None
                result = {
                    'success': False,
                    'prediction': None,
                    'metadata': {'error': str(e)},
                    'request_id': req_id
                }
                results.append(result)
            return results

    def _init_optimization_components(self):
        """Initialize high-impact optimization components for inference."""
        try:
            # Initialize JIT compiler for hot inference paths
            self.jit_compiler = get_global_jit_compiler()
            if hasattr(self.config, 'enable_jit_compilation') and self.config.enable_jit_compilation:
                self.jit_compiler.enable_compilation = True
                self.logger.info("JIT compiler enabled for inference optimization")
            
            # Initialize mixed precision manager for inference
            self.mixed_precision_manager = get_global_mixed_precision_manager()
            if hasattr(self.config, 'enable_mixed_precision') and self.config.enable_mixed_precision:
                self.mixed_precision_manager.enable_mixed_precision()
                self.logger.info("Mixed precision enabled for inference")
            
            # Initialize streaming pipeline for batch inference
            self.streaming_pipeline = get_global_streaming_pipeline()
            if hasattr(self.config, 'enable_streaming') and self.config.enable_streaming:
                self.logger.info("Streaming pipeline available for batch inference")
                
        except Exception as e:
            self.logger.warning(f"Some optimization components could not be initialized: {str(e)}")

    def predict_batch_streaming(self, features_df: pd.DataFrame, 
                               batch_size: Optional[int] = None,
                               output_path: Optional[str] = None) -> Iterator[Tuple[bool, np.ndarray, Dict[str, Any]]]:
        """
        Streaming batch prediction for large datasets using streaming pipeline.
        
        Args:
            features_df: Input features as DataFrame
            batch_size: Batch size for streaming (uses config default if None)
            output_path: Optional path to save predictions
            
        Yields:
            Prediction results for each chunk
        """
        if not hasattr(self, 'streaming_pipeline'):
            raise RuntimeError("Streaming pipeline not available")
        
        batch_size = batch_size or getattr(self.config, 'streaming_batch_size', 1000)
        
        # Update streaming pipeline chunk size
        original_chunk_size = self.streaming_pipeline.chunk_size
        self.streaming_pipeline.chunk_size = batch_size
        
        try:
            def predict_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
                """Predict on a single chunk."""
                features_array = chunk.values
                success, predictions, metadata = self.predict(features_array)
                
                if not success:
                    raise RuntimeError(f"Prediction failed: {metadata.get('error', 'Unknown error')}")
                
                # Create result DataFrame
                result_df = chunk.copy()
                if predictions.ndim == 1:
                    result_df['prediction'] = predictions
                else:
                    # Multi-output predictions
                    for i, pred_col in enumerate(predictions.T):
                        result_df[f'prediction_{i}'] = pred_col
                
                result_df['prediction_metadata'] = str(metadata)
                return result_df
            
            # Process using streaming pipeline
            with self.streaming_pipeline.monitoring_context():
                for chunk_result in self.streaming_pipeline.process_dataframe_stream(
                    features_df, predict_chunk, output_path
                ):
                    # Extract predictions from chunk result
                    prediction_columns = [col for col in chunk_result.columns if col.startswith('prediction')]
                    if len(prediction_columns) == 1:
                        predictions = chunk_result['prediction'].values
                    else:
                        predictions = chunk_result[prediction_columns].values
                    
                    yield True, predictions, {"streaming": True, "chunk_size": len(chunk_result)}
                    
        except Exception as e:
            self.logger.error(f"Streaming batch prediction failed: {str(e)}")
            yield False, None, {"error": str(e), "streaming": True}
        
        finally:
            # Restore original chunk size
            self.streaming_pipeline.chunk_size = original_chunk_size
    
    def _safe_log(self, level, message):
        """Safely log a message, avoiding I/O errors during shutdown."""
        try:
            if hasattr(self, 'logger') and self.logger:
                # Check if logger handlers are still valid
                for handler in self.logger.handlers:
                    if hasattr(handler, 'stream') and hasattr(handler.stream, 'closed'):
                        if handler.stream.closed:
                            return  # Skip logging if stream is closed
                
                getattr(self.logger, level)(message)
        except (ValueError, OSError, AttributeError):
            # Silently ignore logging errors during shutdown
            pass
    
    def shutdown(self):
        """Shutdown the inference engine and clean up resources."""
        try:
            self._safe_log("info", "Shutting down inference engine")
            
            # Stop monitoring
            if hasattr(self, '_monitoring_active'):
                self._monitoring_active = False
            
            # Stop dynamic batcher
            if hasattr(self, 'dynamic_batcher') and self.dynamic_batcher:
                self.dynamic_batcher.stop()
            
            # Clear caches
            if hasattr(self, 'result_cache') and self.result_cache:
                self.result_cache.clear()
            if hasattr(self, 'feature_cache') and self.feature_cache:
                self.feature_cache.clear()
            
            # Clear memory pool
            if hasattr(self, 'memory_pool') and self.memory_pool:
                self.memory_pool.clear()
            
            # Cleanup models
            if hasattr(self, 'model'):
                self.model = None
            if hasattr(self, 'compiled_model'):
                self.compiled_model = None
                
            # Set state to stopped
            self.state = EngineState.STOPPED
            
            self._safe_log("info", "Inference engine shutdown complete")
            
        except Exception as e:
            self._safe_log("error", f"Error during shutdown: {str(e)}")
    
    def _compile_model(self):
        """Compile the model for faster inference if supported."""
        if self.model is None:
            return
        
        try:
            # Check if ONNX compilation is available and enabled
            if self.config.enable_onnx and hasattr(self.config, 'onnx_available') and self.config.onnx_available:
                self._compile_with_onnx()
            # Check if Treelite compilation is available for tree models  
            elif hasattr(self.config, 'treelite_available') and self.config.treelite_available:
                self._compile_with_treelite()
            else:
                # Use JIT compilation for supported models
                if hasattr(self, 'jit_compiler') and self.jit_compiler:
                    self.logger.info("Using JIT compilation for model acceleration")
                    # JIT compilation happens dynamically during prediction
                else:
                    self.logger.info("Model compilation not available, using standard inference")
        except Exception as e:
            self.logger.warning(f"Failed to compile model: {str(e)}. Using standard inference.")
    
    def _compile_with_onnx(self):
        """Compile model using ONNX Runtime."""
        try:
            import onnx
            import onnxruntime as ort
            from sklearn import __version__ as sklearn_version
            
            # Only support sklearn models for now
            if self.model_type != ModelType.SKLEARN:
                raise ValueError(f"ONNX compilation not supported for {self.model_type}")
            
            # Convert sklearn model to ONNX
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Define input shape
            feature_count = len(self.feature_names) if self.feature_names else 10
            initial_type = [('float_input', FloatTensorType([None, feature_count]))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(self.model, initial_types=initial_type)
            
            # Create ONNX Runtime session
            self.compiled_model = ort.InferenceSession(onnx_model.SerializeToString())
            self.onnx_input_name = 'float_input'
            
            self.logger.info("Model successfully compiled with ONNX Runtime")
        except Exception as e:
            self.logger.warning(f"ONNX compilation failed: {str(e)}")
            raise
    
    def _compile_with_treelite(self):
        """Compile tree-based models using Treelite."""
        try:
            import treelite
            import treelite_runtime
            
            # Only support tree-based models
            if self.model_type not in (ModelType.XGBOOST, ModelType.LIGHTGBM):
                # Check if sklearn model is tree-based
                if self.model_type == ModelType.SKLEARN:
                    model_name = self.model.__class__.__name__.lower()
                    if 'forest' not in model_name and 'tree' not in model_name:
                        raise ValueError(f"Treelite compilation not supported for {model_name}")
                else:
                    raise ValueError(f"Treelite compilation not supported for {self.model_type}")
            
            # Convert model to Treelite
            if self.model_type == ModelType.XGBOOST:
                tl_model = treelite.Model.from_xgboost(self.model)
            elif self.model_type == ModelType.LIGHTGBM:
                tl_model = treelite.Model.from_lightgbm(self.model)
            elif self.model_type == ModelType.SKLEARN:
                tl_model = treelite.Model.from_sklearn(self.model)
            
            # Compile the model
            tl_model.compile(dirpath='./temp_treelite_model', verbose=False)
            
            # Load compiled model
            self.compiled_model = treelite_runtime.Predictor('./temp_treelite_model')
            
            self.logger.info("Model successfully compiled with Treelite")
        except Exception as e:
            self.logger.warning(f"Treelite compilation failed: {str(e)}")
            raise
    
    def _precompute_common_values(self):
        """Pre-compute common values for cache optimization."""
        try:
            # Pre-allocate common array shapes in memory pool
            if hasattr(self, 'memory_pool') and self.memory_pool:
                common_shapes = [
                    (1, len(self.feature_names)) if self.feature_names else (1, 10),
                    (32, len(self.feature_names)) if self.feature_names else (32, 10),
                    (64, len(self.feature_names)) if self.feature_names else (64, 10),
                ]
                
                for shape in common_shapes:
                    try:
                        buffer = self.memory_pool.get_buffer(shape=shape, dtype=np.float32)
                        # Return buffer to pool for reuse
                        self.memory_pool.return_buffer(buffer)
                    except Exception:
                        pass  # Non-critical optimization
            
            # Pre-warm any caches
            if hasattr(self, 'result_cache') and self.result_cache:
                # Cache is already initialized, no pre-warming needed
                pass
            
            self.logger.debug("Common values pre-computed for optimization")
        except Exception as e:
            self.logger.debug(f"Pre-computation optimization failed: {str(e)}")
    
    def _create_cache_key(self, features):
        """Create a cache key for features."""
        if isinstance(features, np.ndarray):
            # For numpy arrays, create a hash of the values
            return hash(features.tobytes())
        elif isinstance(features, (list, tuple)):
            # For lists/tuples, convert to tuple and hash
            return hash(tuple(features))
        elif hasattr(features, 'values'):
            # For DataFrames, use the underlying values
            return hash(features.values.tobytes())
        else:
            # Fallback: convert to string and hash
            return hash(str(features))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the inference engine."""
        try:
            # Basic metrics
            metrics = {
                'state': self.state.value if hasattr(self.state, 'value') else str(self.state),
                'model_loaded': self.model is not None,
                'total_predictions': getattr(self, '_prediction_count', 0),
                'cache_stats': {}
            }
            
            # Add cache statistics if cache is available
            if hasattr(self, 'cache') and self.cache is not None:
                if hasattr(self.cache, 'get_stats'):
                    metrics['cache_stats'] = self.cache.get_stats()
                else:
                    metrics['cache_stats'] = {'enabled': True}
            
            # Add batch processor metrics if available
            if hasattr(self, 'batch_processor') and self.batch_processor is not None:
                if hasattr(self.batch_processor, 'get_metrics'):
                    metrics['batch_processor'] = self.batch_processor.get_metrics()
                else:
                    metrics['batch_processor'] = {'enabled': True}
            
            # Add memory pool metrics if available
            if hasattr(self, 'memory_pool') and self.memory_pool is not None:
                if hasattr(self.memory_pool, 'get_stats'):
                    metrics['memory_pool'] = self.memory_pool.get_stats()
                else:
                    metrics['memory_pool'] = {'enabled': True}
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    def validate_model(self) -> Dict[str, Any]:
        """Validate that the loaded model is working correctly."""
        try:
            if self.model is None:
                self.logger.warning("No model loaded for validation")
                return {
                    "valid": False,
                    "error": "No model loaded",
                    "model_type": "None"
                }
            
            # Check if model has required attributes/methods
            if hasattr(self.model, 'predict'):
                # Try a simple prediction with dummy data
                try:
                    # Create simple test data based on model info
                    num_features = 5  # default
                    
                    # Try to detect the number of features from the model
                    if hasattr(self.model, 'n_features_in_'):
                        num_features = self.model.n_features_in_
                    elif hasattr(self, 'model_info') and 'input_shape' in self.model_info:
                        input_shape = self.model_info['input_shape']
                        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 1:
                            num_features = input_shape[1]
                    
                    test_data = np.random.random((1, num_features)).astype(np.float32)
                    
                    # Try prediction
                    _ = self.model.predict(test_data)
                    self.logger.info("Model validation successful")
                    return {
                        "valid": True,
                        "model_type": self.model_type.name if self.model_type else "Unknown",
                        "test_prediction_successful": True
                    }
                except Exception as pred_error:
                    self.logger.warning(f"Model prediction test failed: {pred_error}")
                    return {
                        "valid": False,
                        "error": f"Model prediction test failed: {pred_error}",
                        "model_type": self.model_type.name if self.model_type else "Unknown"
                    }
            else:
                self.logger.warning("Model does not have predict method")
                return {
                    "valid": False,
                    "error": "Model does not have predict method",
                    "model_type": self.model_type.name if self.model_type else "Unknown"
                }
                
                
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            return {
                "valid": False,
                "error": f"Validation error: {e}",
                "model_type": "Unknown"
            }


class InferenceServer:
    """
    High-level inference server for model serving
    Provides a simple interface for loading and serving models
    """
    
    def __init__(self):
        """Initialize the inference server"""
        self.inference_engine = None
        self.is_loaded = False
        self.current_model_path = None
        self.logger = logging.getLogger(__name__)
        
    def load_model_from_path(self, model_file, password: Optional[str] = None) -> str:
        """
        Load a model from file path
        
        Args:
            model_file: Model file (either path string or file object)
            password: Optional password for encrypted models
            
        Returns:
            Status message
        """
        try:
            if hasattr(model_file, 'name'):
                # It's a file object from Gradio
                model_path = model_file.name
            else:
                # It's a string path
                model_path = model_file
                
            if not os.path.exists(model_path):
                return "❌ Model file not found"
            
            # Create inference engine with default config
            from ..configs import InferenceEngineConfig, OptimizationMode
            config = InferenceEngineConfig(
                optimization_mode=OptimizationMode.BALANCED,
                enable_caching=True,
                cache_size=100
            )
            
            self.inference_engine = InferenceEngine(config)
            
            # Load the model
            success = self.inference_engine.load_model(model_path)
            
            if success:
                self.is_loaded = True
                self.current_model_path = model_path
                model_info = self.inference_engine.get_model_info()
                return f"✅ Model loaded successfully!\n\nModel Info:\n{json.dumps(model_info, indent=2)}"
            else:
                return "❌ Failed to load model"
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return f"❌ Error loading model: {str(e)}"
    
    def predict(self, input_data: str) -> str:
        """
        Make prediction with the loaded model
        
        Args:
            input_data: Input data as string (comma-separated or JSON)
            
        Returns:
            Prediction result as formatted string
        """
        try:
            if not self.is_loaded or not self.inference_engine:
                return "❌ No model loaded. Please load a model first."
            
            # Parse input data
            try:
                if input_data.strip().startswith('[') or input_data.strip().startswith('{'):
                    # JSON format
                    data = json.loads(input_data)
                    if isinstance(data, list):
                        input_array = np.array(data).reshape(1, -1)
                    elif isinstance(data, dict):
                        # Convert dict to array (assuming it's feature dict)
                        input_array = np.array(list(data.values())).reshape(1, -1)
                    else:
                        return "❌ Invalid JSON format"
                else:
                    # Comma-separated values
                    data = [float(x.strip()) for x in input_data.split(',') if x.strip()]
                    input_array = np.array(data).reshape(1, -1)
                    
            except Exception as parse_error:
                return f"❌ Error parsing input: {str(parse_error)}\nExpected: comma-separated values or JSON array/object"
            
            # Make prediction
            result = self.inference_engine.predict(input_array)
            
            if result is None:
                return "❌ Prediction failed"
            
            # Format result
            if isinstance(result, np.ndarray):
                if len(result.shape) == 1:
                    prediction = result[0]
                else:
                    prediction = result[0]  # First row
            else:
                prediction = result
            
            # Get model info for context
            model_info = self.inference_engine.get_model_info()
            
            formatted_result = f"""
🔮 Prediction Result:

📥 Input: {input_data}
📤 Prediction: {prediction}
🎯 Model: {model_info.get('model_type', 'Unknown')}
📊 Input Shape: {input_array.shape}
⏱️ Model Path: {os.path.basename(self.current_model_path)}

✅ Prediction completed successfully!
            """
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return f"❌ Prediction error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        try:
            if not self.is_loaded or not self.inference_engine:
                return {"status": "No model loaded"}
            
            model_info = self.inference_engine.get_model_info()
            model_info["model_path"] = self.current_model_path
            model_info["status"] = "Model loaded and ready"
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}