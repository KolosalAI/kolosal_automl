import os
import time
import logging
import pickle
import hashlib
import threading
import gc
import json
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import traceback
import signal
import psutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
import copy

# Optional imports for specific optimizations
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        SKLEARN_OPTIMIZED = True
    except ImportError:
        SKLEARN_OPTIMIZED = False
except ImportError:
    SKLEARN_AVAILABLE = False
    SKLEARN_OPTIMIZED = False

try:
    import intel
    INTEL_PYTHON = True
except ImportError:
    INTEL_PYTHON = False

try:
    import mkl
    MKL_AVAILABLE = True
except ImportError:
    MKL_AVAILABLE = False

# Import local modules
from modules.engine.batch_processor import BatchProcessor
from modules.engine.data_preprocessor import DataPreprocessor
from modules.engine.quantizer import Quantizer
from modules.engine.lru_ttl_cache import LRUTTLCache
from modules.configs import (
    QuantizationConfig, BatchProcessorConfig, BatchProcessingStrategy,
    BatchPriority, PreprocessorConfig, NormalizationType, 
    QuantizationMode, ModelType, EngineState, InferenceEngineConfig
)

# Define constants
MODEL_REGISTRY_PATH = os.environ.get("MODEL_REGISTRY_PATH", "./models")
DEFAULT_LOG_PATH = os.environ.get("LOG_PATH", "./logs")
MAX_RETRY_ATTEMPTS = 3
MEMORY_CHECK_INTERVAL = 10  # seconds


class ModelMetrics:
    """Class to track and report model performance metrics with thread safety."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.lock = threading.RLock()
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.inference_times = deque(maxlen=self.window_size)
            self.batch_sizes = deque(maxlen=self.window_size)
            self.queue_times = deque(maxlen=self.window_size)
            self.quantize_times = deque(maxlen=self.window_size)
            self.dequantize_times = deque(maxlen=self.window_size)
            self.total_requests = 0
            self.error_count = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.throttled_requests = 0
            self.last_updated = time.time()
    
    def update_inference(self, inference_time: float, batch_size: int, queue_time: float = 0,
                         quantize_time: float = 0, dequantize_time: float = 0) -> None:
        """Update inference metrics including quantization timing."""
        with self.lock:
            self.inference_times.append(inference_time)
            self.batch_sizes.append(batch_size)
            self.queue_times.append(queue_time)
            self.quantize_times.append(quantize_time)
            self.dequantize_times.append(dequantize_time)
            self.total_requests += batch_size
            self.last_updated = time.time()
    
    def record_error(self) -> None:
        """Record an inference error."""
        with self.lock:
            self.error_count += 1
            self.last_updated = time.time()
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self.lock:
            self.cache_hits += 1
            self.last_updated = time.time()
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self.lock:
            self.cache_misses += 1
            self.last_updated = time.time()
    
    def record_throttled(self) -> None:
        """Record a throttled request."""
        with self.lock:
            self.throttled_requests += 1
            self.last_updated = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as a dictionary."""
        with self.lock:
            metrics = {
                "total_requests": self.total_requests,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(1, self.total_requests),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                "throttled_requests": self.throttled_requests,
                "last_updated": self.last_updated
            }
            
            if self.inference_times:
                metrics.update({
                    "avg_inference_time_ms": np.mean(self.inference_times) * 1000,
                    "p50_inference_time_ms": np.percentile(self.inference_times, 50) * 1000,
                    "p95_inference_time_ms": np.percentile(self.inference_times, 95) * 1000,
                    "p99_inference_time_ms": np.percentile(self.inference_times, 99) * 1000,
                    "min_inference_time_ms": np.min(self.inference_times) * 1000,
                    "max_inference_time_ms": np.max(self.inference_times) * 1000,
                })
            
            if self.batch_sizes:
                metrics.update({
                    "avg_batch_size": np.mean(self.batch_sizes),
                    "max_batch_size": np.max(self.batch_sizes),
                })
            
            if self.queue_times:
                metrics.update({
                    "avg_queue_time_ms": np.mean(self.queue_times) * 1000,
                    "p95_queue_time_ms": np.percentile(self.queue_times, 95) * 1000,
                })
            
            if self.quantize_times:
                metrics.update({
                    "avg_quantize_time_ms": np.mean(self.quantize_times) * 1000,
                    "p95_quantize_time_ms": np.percentile(self.quantize_times, 95) * 1000,
                })
                
            if self.dequantize_times:
                metrics.update({
                    "avg_dequantize_time_ms": np.mean(self.dequantize_times) * 1000,
                    "p95_dequantize_time_ms": np.percentile(self.dequantize_times, 95) * 1000,
                })
            
            if self.inference_times:
                total_processing_time = sum(self.inference_times)
                if total_processing_time > 0:
                    metrics["throughput_requests_per_second"] = self.total_requests / total_processing_time
            
            return metrics


class InferenceEngine:
    """
    High-performance CPU-optimized inference engine with Intel optimizations, 
    adaptive batching, quantization support, and comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[InferenceEngineConfig] = None):
        """
        Initialize the inference engine.
        
        Args:
            config: Configuration for the inference engine
        """
        self.config = config or InferenceEngineConfig()
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize state
        self.state = EngineState.INITIALIZING
        self.model = None
        self.model_type = None
        self.model_info = {}
        self.feature_names = []
        
        # Thread safety
        self.inference_lock = threading.RLock()
        self.state_lock = threading.RLock()
        
        # Initialize quantizer if quantization is enabled
        self.quantizer = None
        if (self.config.enable_quantization or self.config.enable_model_quantization or 
            self.config.enable_input_quantization):
            self._setup_quantizer()
        
        # Preprocessing
        self.preprocessor = None
        
        # Performance monitoring
        self.metrics = ModelMetrics(window_size=self.config.monitoring_window)
        
        # Cache for inference results
        if self.config.enable_request_deduplication:
            self.result_cache = LRUTTLCache(
                max_size=self.config.max_cache_entries,
                ttl_seconds=self.config.cache_ttl_seconds
            )
        else:
            self.result_cache = None
        
        # Track active requests for throttling and shutdown
        self.active_requests = 0
        self.active_requests_lock = threading.Lock()
        
        # Initialize batch processor attribute to ensure it always exists
        self.batch_processor = None
        # Set up threading
        self._setup_threading()
        
        # Set up monitoring and resource management
        self._setup_monitoring()
        
        # Configure Intel optimizations if enabled
        if self.config.enable_intel_optimization:
            self._setup_intel_optimizations()
        
        # Initialize batch processor if batching is enabled
        if self.config.enable_batching:
            self._setup_batch_processor()
        
        # Signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Change state to ready
        self.state = EngineState.READY
        self.logger.info(f"Inference engine initialized with version {self.config.model_version}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the inference engine."""
        logger = logging.getLogger(f"{__name__}.InferenceEngine")
        
        # Configure logging based on debug mode
        level = logging.DEBUG if self.config.debug_mode else logging.INFO
        logger.setLevel(level)
        
        # Ensure log directory exists
        os.makedirs(DEFAULT_LOG_PATH, exist_ok=True)
        
        # Only add handlers if there are none
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            
            # File handler
            file_handler = logging.FileHandler(
                os.path.join(DEFAULT_LOG_PATH, f"engine_{self.config.model_version}.log")
            )
            file_handler.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_quantizer(self) -> None:
        """Initialize and configure the quantizer."""
        quantization_config = self.config.quantization_config
        if quantization_config is None:
            # Create default config if none provided
            quantization_config = QuantizationConfig(
                quantization_type=self.config.quantization_dtype,
                quantization_mode=QuantizationMode.DYNAMIC_PER_BATCH.value,
                enable_cache=True,
                cache_size=1024,
                use_percentile=True,
                min_percentile=0.1,
                max_percentile=99.9,
                error_on_nan=False,
                error_on_inf=False
            )
        
        # Initialize the quantizer
        self.quantizer = Quantizer(quantization_config)
        self.logger.info(f"Quantizer initialized with {quantization_config.quantization_type} type and "
                         f"{quantization_config.quantization_mode} mode")
    
    def _setup_threading(self) -> None:
        """Configure threading and parallelism."""
        # Set number of threads for numpy/MKL if available
        thread_count = self.config.num_threads
        
        # Set environment variables for better thread control
        os.environ["OMP_NUM_THREADS"] = str(thread_count)
        os.environ["MKL_NUM_THREADS"] = str(thread_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
        
        # Set thread count for numpy if using OpenBLAS/MKL
        try:
            import numpy as np
            np.set_num_threads(thread_count)
        except (ImportError, AttributeError):
            pass
        
        # Configure MKL settings if available
        if MKL_AVAILABLE:
            try:
                mkl.set_num_threads(thread_count)
                self.logger.info(f"MKL configured with {thread_count} threads")
            except Exception as e:
                self.logger.warning(f"Failed to configure MKL: {str(e)}")
        
        # Set CPU affinity if enabled and supported
        if self.config.set_cpu_affinity:
            try:
                p = psutil.Process(os.getpid())
                # Use the specified number of CPUs
                cpu_count = os.cpu_count() or 4
                cpu_list = list(range(min(thread_count, cpu_count)))
                p.cpu_affinity(cpu_list)
                self.logger.info(f"CPU affinity set to cores {cpu_list}")
            except (AttributeError, NotImplementedError):
                self.logger.warning("CPU affinity setting not supported on this platform")
        
        # Create thread pool for parallel tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=thread_count,
            thread_name_prefix="InferenceWorker"
        )
        
        self.logger.info(f"Threading configured with {thread_count} threads")
    
    def _setup_intel_optimizations(self) -> None:
        """Configure Intel-specific optimizations."""
        if not self.config.enable_intel_optimization:
            return
        
        # Log status of Intel optimizations
        optimizations = []
        
        # Configure Intel MKL if available
        if MKL_AVAILABLE:
            try:
                # Configure MKL for best performance
                mkl.set_threading_layer("intel")
                mkl.set_dynamic(False)
                optimizations.append("MKL")
            except Exception as e:
                self.logger.warning(f"Failed to configure MKL optimizations: {str(e)}")
        
        # Check for scikit-learn optimizations
        if SKLEARN_AVAILABLE and SKLEARN_OPTIMIZED:
            optimizations.append("scikit-learn-intelex")
        
        # Check if we're running Intel Python Distribution
        if INTEL_PYTHON:
            optimizations.append("Intel Python Distribution")
        
        if optimizations:
            self.logger.info(f"Intel optimizations enabled: {', '.join(optimizations)}")
        else:
            self.logger.warning("No Intel optimizations are available")
    
    def _setup_batch_processor(self) -> None:
        """Initialize the batch processor for efficient inference."""
        # Convert batch processing strategy string to enum
        strategy_str = self.config.batch_processing_strategy.upper()
        try:
            strategy = BatchProcessingStrategy[strategy_str]
        except KeyError:
            self.logger.warning(
                f"Unknown batch processing strategy: {strategy_str}, defaulting to ADAPTIVE"
            )
            strategy = BatchProcessingStrategy.ADAPTIVE
        
        # Create batch processor configuration
        batch_config = BatchProcessorConfig(
            batch_timeout=self.config.batch_timeout,
            max_queue_size=self.config.max_concurrent_requests * 2,
            initial_batch_size=self.config.initial_batch_size,
            min_batch_size=self.config.min_batch_size,
            max_batch_size=self.config.max_batch_size,
            enable_monitoring=self.config.enable_monitoring,
            monitoring_window=self.config.monitoring_window,
            max_retries=MAX_RETRY_ATTEMPTS,
            retry_delay=0.01,
            processing_strategy=strategy,
            max_workers=self.config.num_threads,
            enable_adaptive_batching=self.config.enable_adaptive_batching,
            max_batch_memory_mb=None,  # Determine dynamically based on available memory
            enable_memory_optimization=self.config.enable_memory_optimization,
            enable_priority_queue=True,
            debug_mode=self.config.debug_mode
        )
        
        # Create batch processor
        self.batch_processor = BatchProcessor(batch_config)
        
        # Start the batch processor with our processing function
        self.batch_processor.start(self._process_batch)
        
        self.logger.info(
            f"Batch processor started with strategy={strategy.name}, "
            f"batch_size={self.config.initial_batch_size}"
        )
    
    def _setup_monitoring(self) -> None:
        """Set up monitoring and resource management."""
        if not self.config.enable_monitoring:
            return
        
        # Create memory monitor thread
        self.monitor_stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="InferenceMonitor"
        )
        
        # Start monitoring thread
        self.monitor_thread.start()
        self.logger.info("Monitoring thread started")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.logger.info("Signal handlers registered for graceful shutdown")
        except (AttributeError, ValueError) as e:
            # Signal handling might not work in certain environments (e.g., Windows)
            self.logger.warning(f"Could not set up signal handlers: {e}")
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle termination signals."""
        self.logger.info(f"Received signal {sig}, initiating graceful shutdown")
        self.shutdown()
    
    def _monitoring_loop(self) -> None:
        """Background thread for monitoring system resources."""
        check_interval = min(MEMORY_CHECK_INTERVAL, self.config.monitoring_interval)
        
        while not self.monitor_stop_event.is_set():
            try:
                # Check memory usage
                self._check_memory_usage()
                
                # Check CPU usage if throttling enabled
                if self.config.throttle_on_high_cpu:
                    self._check_cpu_usage()
                
                # Log periodic metrics if in debug mode
                if self.config.debug_mode:
                    self._log_metrics()
                
                # Sleep until next check
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
                
                # Sleep with reduced interval on error
                time.sleep(max(1.0, check_interval / 2))
    
    def _check_memory_usage(self) -> None:
        """Monitor memory usage and take action if too high."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Check if we've exceeded high watermark
            if memory_mb > self.config.memory_high_watermark_mb:
                self.logger.warning(
                    f"Memory usage ({memory_mb:.1f} MB) exceeds high watermark "
                    f"({self.config.memory_high_watermark_mb} MB), triggering GC"
                )
                # Force garbage collection
                gc.collect()
                
                # Clear cache if it exists and memory is still high
                if self.result_cache is not None:
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    
                    if memory_mb > self.config.memory_high_watermark_mb:
                        self.logger.warning("Memory still high after GC, clearing result cache")
                        self.result_cache.clear()
                
                # Clear quantizer cache if it exists and memory is still high
                if self.quantizer is not None:
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    
                    if memory_mb > self.config.memory_high_watermark_mb:
                        self.logger.warning("Memory still high, clearing quantizer cache")
                        self.quantizer.clear_cache()
            
            # Check against absolute limit if configured
            if self.config.memory_limit_gb is not None:
                memory_limit_mb = self.config.memory_limit_gb * 1024
                if memory_mb > memory_limit_mb:
                    self.logger.error(
                        f"Memory usage ({memory_mb:.1f} MB) exceeds limit "
                        f"({memory_limit_mb:.1f} MB), entering error state"
                    )
                    self.state = EngineState.ERROR
                    # Trigger cache clearing and emergency cleanup
                    if self.result_cache is not None:
                        self.result_cache.clear()
                    if self.quantizer is not None:
                        self.quantizer.clear_cache()
                    gc.collect()
                    
        except Exception as e:
            self.logger.error(f"Failed to check memory usage: {str(e)}")
    
    def _check_cpu_usage(self) -> None:
        """Monitor CPU usage and throttle requests if needed."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Check if we need to enable/disable throttling based on CPU usage
            if cpu_percent > self.config.cpu_threshold_percent:
                if not self.config.enable_throttling:
                    self.logger.warning(
                        f"CPU usage ({cpu_percent:.1f}%) above threshold "
                        f"({self.config.cpu_threshold_percent}%), enabling throttling"
                    )
                    self.config.enable_throttling = True
            elif self.config.enable_throttling and cpu_percent < self.config.cpu_threshold_percent * 0.8:
                # Hysteresis: only disable throttling when significantly below threshold
                self.logger.info(
                    f"CPU usage ({cpu_percent:.1f}%) below threshold, disabling throttling"
                )
                self.config.enable_throttling = False
                
        except Exception as e:
            self.logger.error(f"Failed to check CPU usage: {str(e)}")
    
    def _log_metrics(self) -> None:
        """Log current performance metrics."""
        try:
            metrics = self.metrics.get_metrics()
            
            # Log main metrics with cache hit rate converted to a percentage
            self.logger.debug(
                f"Performance metrics: avg_inference={metrics.get('avg_inference_time_ms', 0):.2f}ms, "
                f"p95_inference={metrics.get('p95_inference_time_ms', 0):.2f}ms, "
                f"avg_batch_size={metrics.get('avg_batch_size', 0):.1f}, "
                f"throughput={metrics.get('throughput_requests_per_second', 0):.1f}req/s, "
                f"cache_hit_rate={metrics.get('cache_hit_rate', 0)*100:.1f}%"
            )
            
            # Additional debug metrics when quantization is enabled
            if (self.config.enable_quantization or self.config.enable_model_quantization or 
                self.config.enable_input_quantization):
                self.logger.debug(
                    f"Quantization metrics: avg_quantize={metrics.get('avg_quantize_time_ms', 0):.2f}ms, "
                    f"avg_dequantize={metrics.get('avg_dequantize_time_ms', 0):.2f}ms"
                )
                
            # Log memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.logger.debug(f"Memory usage: {memory_mb:.1f} MB")
            
            # Log cache stats if enabled
            if self.result_cache is not None:
                cache_stats = self.result_cache.get_stats()
                self.logger.debug(
                    f"Cache stats: size={cache_stats['size']}/{cache_stats['max_size']}, "
                    f"hit_rate={cache_stats['hit_rate_percent']:.1f}%"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {str(e)}")
    
    def load_model(self, model_path: str, model_type: ModelType = None) -> bool:
        """
        Load a pre-trained model from file.
        
        Args:
            model_path: Path to the saved model file
            model_type: Type of model being loaded
            
        Returns:
            Success status
        """
        with self.state_lock:
            if self.state in (EngineState.LOADING, EngineState.STOPPING, EngineState.STOPPED):
                self.logger.warning(f"Cannot load model in current state: {self.state}")
                return False
            
            self.state = EngineState.LOADING
        
        self.logger.info(f"Loading model from {model_path}")
        
        try:
            # If file doesn't exist, abort
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                self.state = EngineState.ERROR
                return False
            
            # Determine model type if not provided
            if model_type is None:
                model_type = self._detect_model_type(model_path)
            
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
                
            elif model_type == ModelType.LIGHTGBM:
                import lightgbm as lgb
                model = lgb.Booster(model_file=model_path)
                
            elif model_type == ModelType.CUSTOM:
                # Load using pickle for custom models
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    
            elif model_type == ModelType.ENSEMBLE:
                # For ensemble models we expect a specialized format
                if model_path.endswith('.json'):
                    with open(model_path, 'r') as f:
                        ensemble_config = json.load(f)
                    model = self._load_ensemble_from_config(ensemble_config)
                else:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
            
            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                self.state = EngineState.ERROR
                return False
            
            # Store model and update state
            self.model = model
            self.model_type = model_type
            
            # Get model info
            self.model_info = self._extract_model_info(model, model_type)
            
            # Set up preprocessor if feature scaling is enabled
            if self.config.enable_feature_scaling:
                self._setup_preprocessor()
            
            # Quantize model if enabled
            if self.config.enable_model_quantization and self.quantizer is not None:
                self._quantize_model()
            
            # Warm up the model if enabled
            if self.config.enable_warmup:
                self._warmup_model()
            
            self.state = EngineState.READY
            self.logger.info(f"Model loaded successfully: {os.path.basename(model_path)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            self.state = EngineState.ERROR
            return False
    
    def _detect_model_type(self, model_path: str) -> ModelType:
        """
        Attempt to detect model type from file extension or content.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Detected model type
        """
        file_ext = os.path.splitext(model_path)[1].lower()
        
        if file_ext == '.json':
            return ModelType.ENSEMBLE
        elif file_ext == '.pkl' or file_ext == '.joblib':
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
                # If we can't determine, default to CUSTOM
                return ModelType.CUSTOM
        elif file_ext == '.model' or file_ext == '.bst':
            # Likely XGBoost model
            return ModelType.XGBOOST
        elif file_ext == '.txt':
            # Likely LightGBM model
            return ModelType.LIGHTGBM
        else:
            # Default to custom
            return ModelType.CUSTOM
    
    def _extract_model_info(self, model: Any, model_type: ModelType) -> Dict[str, Any]:
        """
        Extract information about the loaded model.
        
        Args:
            model: The loaded model object
            model_type: Type of the model
            
        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": model_type.name,
            "timestamp": datetime.now().isoformat(),
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}"
        }
        
        # Extract sklearn-specific info
        if model_type == ModelType.SKLEARN:
            # Add hyperparameters
            if hasattr(model, 'get_params'):
                info["hyperparameters"] = model.get_params()
            
            # Get feature count
            if hasattr(model, 'n_features_in_'):
                info["feature_count"] = model.n_features_in_
            
            # Get feature importances if available
            if hasattr(model, 'feature_importances_'):
                info["has_feature_importances"] = True
            
        # Extract XGBoost-specific info
        elif model_type == ModelType.XGBOOST:
            # Get number of trees
            info["num_trees"] = getattr(model, 'num_boosted_rounds', 
                                       getattr(model, 'n_estimators', None))
            
            # Try to get feature count
            try:
                dump = model.get_dump()
                info["num_trees"] = len(dump)
            except:
                pass
            
        # Extract LightGBM-specific info
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
    
    def _setup_preprocessor(self) -> None:
        """Set up data preprocessing pipeline."""
        if not self.config.enable_feature_scaling:
            return
        
        # Create preprocessor config
        preprocessor_config = PreprocessorConfig(
            normalization_type=NormalizationType.STANDARD,  # Default to standard scaling
            enable_outlier_detection=True,
            outlier_threshold=3.0,
            clip_values=True,
            min_clip_percentile=0.1,
            max_clip_percentile=99.9,
            handle_missing=True,
            missing_strategy="mean",
            handle_categorical=True,
            max_categories=20,
            enable_dimensionality_reduction=False,
            cache_preprocessing=True,
            cache_size=1000
        )
        
        # Create data preprocessor
        self.preprocessor = DataPreprocessor(preprocessor_config)
        self.logger.info("Data preprocessor initialized")
    
    def _quantize_model(self) -> None:
        """Quantize the model weights if supported."""
        if self.quantizer is None or self.model is None:
            return
        
        try:
            # Model quantization is highly model-specific and may not be supported
            # for all model types. Here we'll implement a few common cases.
            if self.model_type == ModelType.SKLEARN:
                if hasattr(self.model, 'coef_'):
                    # Linear models: quantize coefficients
                    self.logger.info("Quantizing linear model coefficients")
                    original_coef = self.model.coef_.copy()
                    quantized_coef, _ = self.quantizer.quantize(original_coef)
                    # Store original and set quantized version
                    self.model._original_coef = original_coef
                    self.model.coef_ = quantized_coef
                    
                    if hasattr(self.model, 'intercept_'):
                        self.model._original_intercept = self.model.intercept_.copy()
                        quantized_intercept, _ = self.quantizer.quantize(self.model.intercept_)
                        self.model.intercept_ = quantized_intercept
                
                elif hasattr(self.model, 'tree_'):
                    # Decision trees: more complex, typically not quantized
                    self.logger.warning("Quantization not supported for this tree-based sklearn model")
                
            elif self.model_type == ModelType.XGBOOST or self.model_type == ModelType.LIGHTGBM:
                # Tree-based models usually don't benefit from weight quantization
                # Their predictions are based on threshold comparisons
                self.logger.warning(f"Model quantization not supported for {self.model_type.name} models")
                
            else:
                self.logger.warning(f"Model quantization not supported for {self.model_type.name} models")
                
        except Exception as e:
            self.logger.error(f"Failed to quantize model: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
    
    def _warmup_model(self) -> None:
        """Perform model warm-up to stabilize performance."""
        if not self.config.enable_warmup or self.model is None:
            return
        
        self.logger.info("Warming up model...")
        try:
            # Create synthetic data for warmup
            feature_count = 0
            
            # Try to determine feature count from model or feature names
            if self.feature_names:
                feature_count = len(self.feature_names)
            elif 'feature_count' in self.model_info:
                feature_count = self.model_info['feature_count']
            elif hasattr(self.model, 'n_features_in_'):
                feature_count = self.model.n_features_in_
            else:
                # Default to 10 features if we can't determine
                feature_count = 10
                self.logger.warning(f"Couldn't determine feature count, using default: {feature_count}")
            
            # Generate synthetic data - uniformly distributed to cover the input space
            warmup_size = 32
            warmup_data = np.random.rand(warmup_size, feature_count)
            
            # Run a few batches through the model
            for i in range(3):  # Run multiple times to warm up caches
                start_time = time.time()
                
                # Handle potential quantization
                if (self.config.enable_input_quantization and self.quantizer is not None):
                    quantize_start = time.time()
                    quantized_data, _ = self.quantizer.quantize(warmup_data)
                    quantize_time = time.time() - quantize_start
                    
                    _ = self._predict_internal(quantized_data)
                    
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"Warmup batch {i+1}: quantization_time={quantize_time*1000:.2f}ms")
                else:
                    _ = self._predict_internal(warmup_data)
                
                batch_time = time.time() - start_time
                self.logger.debug(f"Warmup batch {i+1} completed in {batch_time*1000:.2f}ms")
            
            self.logger.info("Model warmup completed")
            
        except Exception as e:
            self.logger.error(f"Error during model warmup: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
    
    def predict(self, features: np.ndarray, request_id: Optional[str] = None) -> Tuple[bool, Union[np.ndarray, None], Dict[str, Any]]:
        """
        Make prediction with a single input batch.
        
        Args:
            features: Input features as numpy array
            request_id: Optional identifier for the request
            
        Returns:
            Tuple of (success_flag, predictions, metadata)
        """
        # Check if engine is in a valid state
        if self.state not in (EngineState.READY, EngineState.RUNNING):
            error_msg = f"Cannot make prediction: engine in {self.state} state"
            self.logger.error(error_msg)
            return False, None, {"error": error_msg}
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Update engine state
        with self.state_lock:
            if self.state == EngineState.READY:
                self.state = EngineState.RUNNING
        
        # Check if we should throttle based on active requests
        if self.config.enable_throttling:
            with self.active_requests_lock:
                if self.active_requests >= self.config.max_concurrent_requests:
                    self.metrics.record_throttled()
                    error_msg = "Request throttled due to high load"
                    self.logger.warning(f"{error_msg} (active_requests={self.active_requests})")
                    return False, None, {"error": error_msg, "throttled": True}
        
        # Check for cache hit if deduplication is enabled
        if self.config.enable_request_deduplication and self.result_cache is not None:
            # Create a cache key from the input features
            cache_key = self._create_cache_key(features)
            
            # Check cache
            hit, cached_result = self.result_cache.get(cache_key)
            if hit:
                self.metrics.record_cache_hit()
                self.logger.debug(f"Cache hit for request {request_id}")
                return True, cached_result, {"cached": True, "request_id": request_id}
            else:
                self.metrics.record_cache_miss()
        
        # Track active request count
        with self.active_requests_lock:
            self.active_requests += 1
        
        try:
            # Process data through the preprocessor if available
            if self.preprocessor is not None:
                try:
                    features = self.preprocessor.transform(features)
                except Exception as e:
                    self.logger.error(f"Preprocessing error: {str(e)}")
                    return False, None, {"error": f"Preprocessing failed: {str(e)}"}
            
            # Handle quantization if enabled
            quantize_time = 0
            dequantize_time = 0
            
            if self.config.enable_input_quantization and self.quantizer is not None:
                try:
                    q_start = time.time()
                    quantized_features, quantization_params = self.quantizer.quantize(features)
                    quantize_time = time.time() - q_start
                    
                    # Use quantized features for prediction
                    features = quantized_features
                except Exception as e:
                    self.logger.error(f"Quantization error: {str(e)}")
                    return False, None, {"error": f"Quantization failed: {str(e)}"}
            
            # Make prediction with timing
            start_time = time.time()
            predictions = self._predict_internal(features)
            inference_time = time.time() - start_time
            
            # Dequantize predictions if needed
            if (self.config.enable_input_quantization and self.quantizer is not None and 
                self.config.enable_quantization_aware_inference):
                try:
                    dq_start = time.time()
                    predictions = self.quantizer.dequantize(predictions, quantization_params)
                    dequantize_time = time.time() - dq_start
                except Exception as e:
                    self.logger.error(f"Dequantization error: {str(e)}")
                    # Continue with quantized predictions as fallback
            
            # Calculate batch size (number of samples)
            batch_size = features.shape[0] if hasattr(features, 'shape') else 1
            
            # Update metrics
            self.metrics.update_inference(
                inference_time=inference_time,
                batch_size=batch_size,
                queue_time=0,  # No queue time for direct predict calls
                quantize_time=quantize_time,
                dequantize_time=dequantize_time
            )
            
            # Cache result if deduplication is enabled
            if self.config.enable_request_deduplication and self.result_cache is not None:
                cache_key = self._create_cache_key(features)
                self.result_cache.set(cache_key, predictions)
            
            # Return predictions and metadata
            metadata = {
                "request_id": request_id,
                "inference_time_ms": inference_time * 1000,
                "batch_size": batch_size
            }
            
            # Add quantization info if used
            if quantize_time > 0:
                metadata["quantize_time_ms"] = quantize_time * 1000
            if dequantize_time > 0:
                metadata["dequantize_time_ms"] = dequantize_time * 1000
            
            self.logger.debug(
                f"Prediction successful for request {request_id}: "
                f"batch_size={batch_size}, inference_time={inference_time*1000:.2f}ms"
            )
            
            return True, predictions, metadata
            
        except Exception as e:
            self.logger.error(f"Prediction error for request {request_id}: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            
            self.metrics.record_error()
            return False, None, {"error": str(e), "request_id": request_id}
            
        finally:
            # Decrement active request count
            with self.active_requests_lock:
                self.active_requests -= 1
            
            # Restore state if no active requests
            with self.state_lock:
                if self.state == EngineState.RUNNING and self.active_requests == 0:
                    self.state = EngineState.READY
    
    def predict_batch(self, features_batch: np.ndarray, priority: int = 0) -> Future:
        """
        Asynchronously process a batch of features using the batch processor.
        
        Args:
            features_batch: Batch of input features
            priority: Priority level (higher values = higher priority)
            
        Returns:
            Future object for retrieving the result
        """
        if not self.config.enable_batching or self.batch_processor is None:
            # If batching is disabled, process synchronously
            success, predictions, metadata = self.predict(features_batch)
            future = Future()
            if success:
                future.set_result((predictions, metadata))
            else:
                future.set_exception(Exception(metadata.get("error", "Unknown error")))
            return future
        
        # Determine batch priority
        batch_priority = BatchPriority.NORMAL
        if priority > 0:
            batch_priority = BatchPriority.HIGH
        elif priority < 0:
            batch_priority = BatchPriority.LOW
        
        # Submit to batch processor
        return self.batch_processor.enqueue_predict(features_batch, priority=batch_priority)
    
    def _process_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Process a batch of features for the batch processor.
        
        Args:
            batch: Batch of input features
            
        Returns:
            Batch predictions
        """
        # Make direct prediction
        success, predictions, _ = self.predict(batch)
        
        if not success:
            raise RuntimeError("Batch prediction failed")
        
        return predictions
    
    def _predict_internal(self, features: np.ndarray) -> np.ndarray:
        """
        Internal prediction method that handles different model types.
        
        Args:
            features: Input features
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        # Use model-specific prediction methods based on model type
        if self.model_type == ModelType.SKLEARN:
            # Most sklearn models use predict method
            with self.inference_lock:
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(features)
                else:
                    return self.model.predict(features)
                
        elif self.model_type == ModelType.XGBOOST:
            # XGBoost requires data to be in DMatrix format
            import xgboost as xgb
            dmatrix = xgb.DMatrix(features)
            with self.inference_lock:
                return self.model.predict(dmatrix)
                
        elif self.model_type == ModelType.LIGHTGBM:
            # LightGBM can predict directly from numpy arrays
            with self.inference_lock:
                return self.model.predict(features)
                
        elif self.model_type == ModelType.ENSEMBLE:
            # Ensemble models might have custom prediction logic
            if hasattr(self.model, 'predict'):
                with self.inference_lock:
                    return self.model.predict(features)
            else:
                raise RuntimeError("Ensemble model doesn't have a predict method")
                
        elif self.model_type == ModelType.CUSTOM:
            # For custom models, try predict or __call__
            with self.inference_lock:
                if hasattr(self.model, 'predict'):
                    return self.model.predict(features)
                elif callable(self.model):
                    return self.model(features)
                else:
                    raise RuntimeError("Custom model doesn't have a predict method or is not callable")
        
        else:
            raise NotImplementedError(f"Prediction not implemented for model type: {self.model_type}")
    
    def _create_cache_key(self, features: np.ndarray) -> str:
        """
        Create a cache key from input features.
        
        Args:
            features: Input features
            
        Returns:
            Cache key string
        """
        try:
            # Use faster alternatives if available
            key = hashlib.md5(features.tobytes()).hexdigest()
            return key
        except:
            # Fallback to pickle-based solution
            try:
                pickled = pickle.dumps(features)
                return hashlib.md5(pickled).hexdigest()
            except:
                # Last resort, use string representation (slower)
                return hashlib.md5(str(features).encode()).hexdigest()
    
    def shutdown(self) -> None:
        """Gracefully shut down the inference engine."""
        self.logger.info("Shutting down inference engine")
        
        with self.state_lock:
            if self.state in (EngineState.STOPPED, EngineState.STOPPING):
                self.logger.info("Engine already stopped or stopping")
                return
            self.state = EngineState.STOPPING
        
        # Shut down monitoring
        if hasattr(self, 'monitor_stop_event'):
            self.monitor_stop_event.set()
            if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
        
        # Shut down batch processor
        #if hasattr(self, 'batch_processor') and self.batch_processor is not None:
        #    self.batch_processor.shutdown()
        
        # Shut down thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        # Update state to stopped
        self.state = EngineState.STOPPED
        self.logger.info("Inference engine shutdown complete")
    
    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.metrics.get_metrics()
        
        # Add engine state and model info
        metrics.update({
            "engine_state": self.state.name,
            "model_type": self.model_type.name if self.model_type else None,
            "active_requests": self.active_requests,
        })
        
        # Add cache stats if available
        if self.result_cache is not None:
            metrics["cache_stats"] = self.result_cache.get_stats()
        
        # Add system metrics
        try:
            process = psutil.Process(os.getpid())
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            metrics.update({
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
            })
        except:
            pass
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_info:
            return {"error": "No model loaded"}
        
        # Include config information
        info = {
            "model_info": self.model_info,
            "config": self.config.to_dict()
        }
        
        return info
    
    def _load_ensemble_from_config(self, ensemble_config: Dict[str, Any]) -> Any:
        """
        Load an ensemble model from configuration.
        
        Args:
            ensemble_config: Ensemble model configuration dictionary
            
        Returns:
            Loaded ensemble model
        """
        if 'models' not in ensemble_config:
            raise ValueError("Ensemble config must contain 'models' section")
        
        self.logger.info(f"Loading ensemble with {len(ensemble_config['models'])} models")
        
        # Load each model in the ensemble
        models = []
        weights = []
        
        for model_cfg in ensemble_config['models']:
            if 'path' not in model_cfg:
                raise ValueError("Each model in ensemble must specify a 'path'")
            
            model_path = model_cfg['path']
            # Resolve path if not absolute
            if not os.path.isabs(model_path):
                model_path = os.path.join(MODEL_REGISTRY_PATH, model_path)
            
            # Determine model type if specified
            model_type_str = model_cfg.get('type', None)
            if model_type_str:
                try:
                    model_type = ModelType[model_type_str.upper()]
                except KeyError:
                    raise ValueError(f"Unknown model type: {model_type_str}")
            else:
                model_type = self._detect_model_type(model_path)
            
            # Load the model
            if model_type == ModelType.SKLEARN:
                if JOBLIB_AVAILABLE:
                    model = joblib.load(model_path)
                else:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
            
            elif model_type == ModelType.XGBOOST:
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(model_path)
            
            elif model_type == ModelType.LIGHTGBM:
                import lightgbm as lgb
                model = lgb.Booster(model_file=model_path)
            
            elif model_type == ModelType.CUSTOM:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            else:
                raise ValueError(f"Unsupported model type in ensemble: {model_type}")
            
            # Add model and its weight to lists
            models.append(model)
            weights.append(model_cfg.get('weight', 1.0))
        
        # Create ensemble object based on ensemble type
        ensemble_type = ensemble_config.get('ensemble_type', 'average').lower()
        
        # Create a simple wrapper class for the ensemble
        class EnsembleModel:
            """Simple ensemble model wrapper."""
            
            def __init__(self, models, weights, ensemble_type, parent_engine):
                self.models = models
                self.weights = weights
                self.ensemble_type = ensemble_type
                self.parent_engine = parent_engine  # Reference to inference engine
                self.normalized_weights = np.array(weights) / sum(weights) if weights else None
            
            def predict(self, features):
                """Make predictions with the ensemble."""
                predictions = []
                
                # Get predictions from each model
                for i, model in enumerate(self.models):
                    model_type = self.parent_engine._detect_model_type_from_object(model)
                    
                    if model_type == ModelType.SKLEARN:
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict_proba(features)
                        else:
                            pred = model.predict(features)
                    
                    elif model_type == ModelType.XGBOOST:
                        import xgboost as xgb
                        dmatrix = xgb.DMatrix(features)
                        pred = model.predict(dmatrix)
                    
                    elif model_type == ModelType.LIGHTGBM:
                        pred = model.predict(features)
                    
                    elif model_type == ModelType.CUSTOM:
                        if hasattr(model, 'predict'):
                            pred = model.predict(features)
                        elif callable(model):
                            pred = model(features)
                        else:
                            raise RuntimeError(f"Model {i} in ensemble has no predict method")
                    
                    predictions.append(pred)
                
                # Combine predictions based on ensemble type
                if self.ensemble_type == 'average':
                    # Weighted average of predictions
                    if self.normalized_weights is not None:
                        combined = np.zeros_like(predictions[0])
                        for i, pred in enumerate(predictions):
                            combined += pred * self.normalized_weights[i]
                        return combined
                    else:
                        return np.mean(predictions, axis=0)
                
                elif self.ensemble_type == 'vote':
                    # Hard voting (for classification)
                    votes = np.zeros((features.shape[0], len(np.unique(np.concatenate(predictions)))))
                    for i, pred in enumerate(predictions):
                        weight = self.normalized_weights[i] if self.normalized_weights is not None else 1.0/len(predictions)
                        for j, cls in enumerate(np.unique(pred)):
                            votes[:, j] += (pred == cls) * weight
                    return np.argmax(votes, axis=1)
                
                elif self.ensemble_type == 'stack':
                    # Simple stacking: concatenate predictions
                    return np.column_stack(predictions)
                
                else:
                    raise ValueError(f"Unsupported ensemble type: {self.ensemble_type}")
        
        # Create the ensemble model
        return EnsembleModel(models, weights, ensemble_type, self)

    def _detect_model_type_from_object(self, model: Any) -> ModelType:
        """
        Determine model type from a model object.
        
        Args:
            model: Model object
            
        Returns:
            Model type
        """
        model_class = model.__class__.__module__ + '.' + model.__class__.__name__
        
        if 'sklearn' in model_class:
            return ModelType.SKLEARN
        elif 'xgboost' in model_class:
            return ModelType.XGBOOST
        elif 'lightgbm' in model_class:
            return ModelType.LIGHTGBM
        elif 'ensemble' in model_class.lower():
            return ModelType.ENSEMBLE
        else:
            return ModelType.CUSTOM

    def clear_cache(self) -> None:
        """Clear inference result cache."""
        if self.result_cache is not None:
            self.result_cache.clear()
            self.logger.info("Inference cache cleared")
        
        if self.quantizer is not None:
            self.quantizer.clear_cache()
            self.logger.info("Quantizer cache cleared")
    
    def set_config(self, config: InferenceEngineConfig) -> None:
        """
        Update configuration settings.
        
        Args:
            config: New configuration
        """
        old_config = self.config
        self.config = config
        
        # Update components based on config changes
        if config.enable_quantization != old_config.enable_quantization:
            if config.enable_quantization:
                self._setup_quantizer()
            else:
                self.quantizer = None
        
        if config.enable_feature_scaling != old_config.enable_feature_scaling:
            if config.enable_feature_scaling:
                self._setup_preprocessor()
            else:
                self.preprocessor = None
        
        if (config.enable_request_deduplication != old_config.enable_request_deduplication or
            config.max_cache_entries != old_config.max_cache_entries or
            config.cache_ttl_seconds != old_config.cache_ttl_seconds):
            if config.enable_request_deduplication:
                self.result_cache = LRUTTLCache(
                    max_size=config.max_cache_entries,
                    ttl_seconds=config.cache_ttl_seconds
                )
            else:
                self.result_cache = None
        
        # Update batch processor configuration if needed
        if self.batch_processor is not None:
            if (config.initial_batch_size != old_config.initial_batch_size or
                config.max_batch_size != old_config.max_batch_size or
                config.min_batch_size != old_config.min_batch_size or
                config.batch_timeout != old_config.batch_timeout or
                config.batch_processing_strategy != old_config.batch_processing_strategy):
                
                # Convert batch processing strategy string to enum
                strategy_str = config.batch_processing_strategy.upper()
                try:
                    strategy = BatchProcessingStrategy[strategy_str]
                except KeyError:
                    self.logger.warning(
                        f"Unknown batch processing strategy: {strategy_str}, defaulting to ADAPTIVE"
                    )
                    strategy = BatchProcessingStrategy.ADAPTIVE
                
                # Update batch processor config
                batch_config = self.batch_processor.get_config()
                batch_config.batch_timeout = config.batch_timeout
                batch_config.initial_batch_size = config.initial_batch_size
                batch_config.min_batch_size = config.min_batch_size
                batch_config.max_batch_size = config.max_batch_size
                batch_config.processing_strategy = strategy
                batch_config.enable_adaptive_batching = config.enable_adaptive_batching
                
                self.batch_processor.update_config(batch_config)
        
        self.logger.info("Configuration updated")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names if available."""
        return self.feature_names
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set feature names.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        self.logger.info(f"Feature names set: {len(feature_names)} features")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
                "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
                "percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(interval=0.1)
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {str(e)}")
            return {"error": str(e)}

    def __enter__(self):
        """Support for context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure engine is properly shutdown when exiting context."""
        self.shutdown()
    
    def __getstate__(self):
        """
        Customize serialization behavior to exclude unpicklable attributes.
        
        Returns:
            Dictionary with picklable state
        """
        state = self.__dict__.copy()
        
        # Remove unpicklable thread and synchronization objects
        unpicklable_attrs = [
            'inference_lock', 'state_lock', 'active_requests_lock',
            'monitor_thread', 'monitor_stop_event', 'thread_pool',
            'logger', 'batch_processor'
        ]
        
        for attr in unpicklable_attrs:
            if attr in state:
                state[attr] = None
        
        # Note that we're serializing but when deserialized,
        # these resources will need to be reinitialized
        
        # Add a flag to indicate this is a restored state
        state['_is_restored'] = True
        
        return state
    def save(self, save_path: str, save_format: str = 'joblib', include_config: bool = True, 
            include_metrics: bool = False, overwrite: bool = False) -> bool:
        """
        Save the inference engine state to disk.
        
        Args:
            save_path: Directory path to save the model and configuration
            save_format: Format to use for saving ('joblib', 'pickle', or 'json')
            include_config: Whether to save configuration separately
            include_metrics: Whether to include current metrics in the saved state
            overwrite: Whether to overwrite existing files
            
        Returns:
            Success status
        """
        self.logger.info(f"Saving inference engine to {save_path}")
        
        # Check if the model is loaded
        if self.model is None:
            self.logger.error("Cannot save: no model loaded")
            return False
        
        # Create the directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Determine file paths
        model_filename = f"model.{save_format}"
        config_filename = "config.json"
        metrics_filename = "metrics.json"
        info_filename = "model_info.json"
        
        model_path = os.path.join(save_path, model_filename)
        config_path = os.path.join(save_path, config_filename)
        metrics_path = os.path.join(save_path, metrics_filename)
        info_path = os.path.join(save_path, info_filename)
        
        # Check if files exist and handle overwrite flag
        if not overwrite:
            for path in [model_path, config_path, metrics_path, info_path]:
                if os.path.exists(path):
                    self.logger.error(f"File {path} already exists and overwrite=False")
                    return False
        
        try:
            # Save the model using the specified format
            if save_format == 'joblib':
                if not JOBLIB_AVAILABLE:
                    self.logger.warning("joblib not available, falling back to pickle")
                    save_format = 'pickle'
                else:
                    joblib.dump(self.model, model_path)
                    self.logger.info(f"Model saved to {model_path} using joblib")
            
            if save_format == 'pickle':
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info(f"Model saved to {model_path} using pickle")
            
            # Save configuration if requested
            if include_config:
                with open(config_path, 'w') as f:
                    json.dump(self.config.to_dict(), f, indent=2)
                self.logger.info(f"Configuration saved to {config_path}")
            
            # Save metrics if requested
            if include_metrics:
                with open(metrics_path, 'w') as f:
                    # Get current metrics and convert any non-serializable values to strings
                    metrics = self.metrics.get_metrics()
                    serializable_metrics = {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) 
                                        else v for k, v in metrics.items()}
                    json.dump(serializable_metrics, f, indent=2)
                self.logger.info(f"Metrics saved to {metrics_path}")
            
            # Save model info
            with open(info_path, 'w') as f:
                # Make a copy of model_info and ensure it's JSON serializable
                info_copy = copy.deepcopy(self.model_info)
                # Convert non-serializable types to strings
                for k, v in info_copy.items():
                    if not isinstance(v, (int, float, bool, str, list, dict, type(None))):
                        info_copy[k] = str(v)
                
                # Add feature names
                info_copy['feature_names'] = self.feature_names
                
                # Add model type
                info_copy['model_type'] = self.model_type.name if self.model_type else None
                
                # Add timestamp
                info_copy['saved_at'] = datetime.now().isoformat()
                
                json.dump(info_copy, f, indent=2)
            self.logger.info(f"Model info saved to {info_path}")
            
            # Create a README file with basic information
            readme_path = os.path.join(save_path, "README.md")
            with open(readme_path, 'w') as f:
                f.write(f"# Inference Engine Model\n\n")
                f.write(f"Saved on: {datetime.now().isoformat()}\n")
                f.write(f"Model type: {self.model_type.name if self.model_type else 'Unknown'}\n")
                f.write(f"Model version: {self.config.model_version}\n\n")
                f.write(f"## Files\n\n")
                f.write(f"- `{model_filename}`: The serialized model\n")
                if include_config:
                    f.write(f"- `{config_filename}`: Engine configuration\n")
                if include_metrics:
                    f.write(f"- `{metrics_filename}`: Performance metrics at time of saving\n")
                f.write(f"- `{info_filename}`: Model metadata\n")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving inference engine: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False
