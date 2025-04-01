# Inference Engine Documentation

## Overview

The `InferenceEngine` class is a high-performance, CPU-optimized inference engine designed for efficient model inference. It supports various optimizations, including Intel-specific optimizations, adaptive batching, quantization, and comprehensive monitoring. The engine is designed to handle multiple concurrent requests, manage system resources efficiently, and provide detailed performance metrics.

## Key Features

- **Intel Optimizations**: Utilizes Intel MKL and other optimizations for enhanced performance.
- **Adaptive Batching**: Dynamically adjusts batch sizes based on system load and resource availability.
- **Quantization Support**: Supports model and input quantization to reduce memory usage and improve inference speed.
- **Comprehensive Monitoring**: Tracks performance metrics, system resource usage, and error rates.
- **Thread Safety**: Ensures thread-safe operations with multiple concurrent requests.
- **Graceful Shutdown**: Handles termination signals for a graceful shutdown process.
- **Caching**: Implements request deduplication and result caching to improve efficiency.

## Class Structure

### `InferenceEngine`

The main class that encapsulates the inference engine functionality.

#### Attributes

- `config`: Configuration for the inference engine.
- `logger`: Logger instance for logging messages.
- `state`: Current state of the engine.
- `model`: Loaded model for inference.
- `model_type`: Type of the loaded model.
- `model_info`: Information about the loaded model.
- `feature_names`: List of feature names.
- `inference_lock`: Thread lock for inference operations.
- `state_lock`: Thread lock for state management.
- `quantizer`: Quantizer instance for quantization operations.
- `preprocessor`: Data preprocessor instance.
- `metrics`: Instance of `ModelMetrics` for tracking performance metrics.
- `result_cache`: Cache for inference results.
- `active_requests`: Number of active inference requests.
- `active_requests_lock`: Thread lock for managing active requests.
- `batch_processor`: Batch processor instance for handling batch inference.
- `thread_pool`: Thread pool for parallel tasks.
- `monitor_stop_event`: Event for stopping the monitoring thread.
- `monitor_thread`: Thread for monitoring system resources.

#### Methods

- `__init__(self, config: Optional[InferenceEngineConfig] = None)`: Initializes the inference engine.
- `_setup_logging(self) -> logging.Logger`: Sets up logging for the inference engine.
- `_setup_quantizer(self) -> None`: Initializes and configures the quantizer.
- `_setup_threading(self) -> None`: Configures threading and parallelism.
- `_setup_intel_optimizations(self) -> None`: Configures Intel-specific optimizations.
- `_setup_batch_processor(self) -> None`: Initializes the batch processor for efficient inference.
- `_setup_monitoring(self) -> None`: Sets up monitoring and resource management.
- `_setup_signal_handlers(self) -> None`: Sets up signal handlers for graceful shutdown.
- `_signal_handler(self, sig, frame) -> None`: Handles termination signals.
- `_monitoring_loop(self) -> None`: Background thread for monitoring system resources.
- `_check_memory_usage(self) -> None`: Monitors memory usage and takes action if too high.
- `_check_cpu_usage(self) -> None`: Monitors CPU usage and throttles requests if needed.
- `_log_metrics(self) -> None`: Logs current performance metrics.
- `load_model(self, model_path: str, model_type: ModelType = None) -> bool`: Loads a pre-trained model from file.
- `_detect_model_type(self, model_path: str) -> ModelType`: Attempts to detect model type from file extension or content.
- `_extract_model_info(self, model: Any, model_type: ModelType) -> Dict[str, Any]`: Extracts information about the loaded model.
- `_setup_preprocessor(self) -> None`: Sets up data preprocessing pipeline.
- `_quantize_model(self) -> None`: Quantizes the model weights if supported.
- `_warmup_model(self) -> None`: Performs model warm-up to stabilize performance.
- `predict(self, features: np.ndarray, request_id: Optional[str] = None) -> Tuple[bool, Union[np.ndarray, None], Dict[str, Any]]`: Makes prediction with a single input batch.
- `predict_batch(self, features_batch: np.ndarray, priority: int = 0) -> Future`: Asynchronously processes a batch of features using the batch processor.
- `_process_batch(self, batch: np.ndarray) -> np.ndarray`: Processes a batch of features for the batch processor.
- `_predict_internal(self, features: np.ndarray) -> np.ndarray`: Internal prediction method that handles different model types.
- `_create_cache_key(self, features: np.ndarray) -> str`: Creates a cache key from input features.
- `shutdown(self) -> None`: Gracefully shuts down the inference engine.
- `get_state(self) -> EngineState`: Gets current engine state.
- `get_metrics(self) -> Dict[str, Any]`: Gets current performance metrics.
- `get_model_info(self) -> Dict[str, Any]`: Gets information about the loaded model.
- `_load_ensemble_from_config(self, ensemble_config: Dict[str, Any]) -> Any`: Loads an ensemble model from configuration.
- `_detect_model_type_from_object(self, model: Any) -> ModelType`: Determines model type from a model object.
- `clear_cache(self) -> None`: Clears inference result cache.
- `set_config(self, config: InferenceEngineConfig) -> None`: Updates configuration settings.
- `get_feature_names(self) -> List[str]`: Gets feature names if available.
- `set_feature_names(self, feature_names: List[str]) -> None`: Sets feature names.
- `get_memory_usage(self) -> Dict[str, float]`: Gets current memory usage statistics.
- `__enter__(self)`: Supports context manager protocol.
- `__exit__(self, exc_type, exc_val, exc_tb)`: Ensures engine is properly shutdown when exiting context.
- `__getstate__(self)`: Customizes serialization behavior to exclude unpicklable attributes.
- `save(self, save_path: str, save_format: str = 'joblib', include_config: bool = True, include_metrics: bool = False, overwrite: bool = False) -> bool`: Saves the inference engine state to disk.

### `ModelMetrics`

Class to track and report model performance metrics with thread safety.

#### Attributes

- `window_size`: Size of the window for tracking metrics.
- `lock`: Thread lock for thread-safe operations.
- `inference_times`: Deque to store inference times.
- `batch_sizes`: Deque to store batch sizes.
- `queue_times`: Deque to store queue times.
- `quantize_times`: Deque to store quantization times.
- `dequantize_times`: Deque to store dequantization times.
- `total_requests`: Total number of requests processed.
- `error_count`: Number of errors encountered.
- `cache_hits`: Number of cache hits.
- `cache_misses`: Number of cache misses.
- `throttled_requests`: Number of throttled requests.
- `last_updated`: Timestamp of the last update.

#### Methods

- `__init__(self, window_size: int = 100)`: Initializes the `ModelMetrics` instance.
- `reset(self) -> None`: Resets all metrics.
- `update_inference(self, inference_time: float, batch_size: int, queue_time: float = 0, quantize_time: float = 0, dequantize_time: float = 0) -> None`: Updates inference metrics including quantization timing.
- `record_error(self) -> None`: Records an inference error.
- `record_cache_hit(self) -> None`: Records a cache hit.
- `record_cache_miss(self) -> None`: Records a cache miss.
- `record_throttled(self) -> None`: Records a throttled request.
- `get_metrics(self) -> Dict[str, Any]`: Gets current metrics as a dictionary.

## Usage Example

```python
# Initialize the inference engine
config = InferenceEngineConfig(
    enable_intel_optimization=True,
    enable_batching=True,
    enable_quantization=True,
    debug_mode=True
)
engine = InferenceEngine(config)

# Load a model
model_path = "./models/my_model.pkl"
engine.load_model(model_path)

# Make a prediction
features = np.random.rand(10, 5)  # Example input features
success, predictions, metadata = engine.predict(features)

if success:
    print("Predictions:", predictions)
else:
    print("Prediction failed:", metadata.get("error"))

# Shutdown the engine
engine.shutdown()
```

## Configuration

The `InferenceEngineConfig` class is used to configure the inference engine. Key configuration options include:

- `enable_intel_optimization`: Enables Intel-specific optimizations.
- `enable_batching`: Enables adaptive batching.
- `enable_quantization`: Enables quantization support.
- `debug_mode`: Enables debug logging.
- `num_threads`: Number of threads to use for parallel tasks.
- `max_concurrent_requests`: Maximum number of concurrent inference requests.
- `memory_high_watermark_mb`: Memory usage threshold for triggering garbage collection.
- `memory_limit_gb`: Absolute memory limit in GB.

## Performance Metrics

The `ModelMetrics` class tracks various performance metrics```markdown
, including:

- **Inference Times**: Average, median (p50), 95th percentile (p95), and 99th percentile (p99) inference times.
- **Batch Sizes**: Average and maximum batch sizes.
- **Queue Times**: Average and 95th percentile queue times.
- **Quantization Times**: Average and 95th percentile quantization and dequantization times.
- **Error Rate**: Ratio of errors to total requests.
- **Cache Hit Rate**: Ratio of cache hits to total cache accesses.
- **Throughput**: Number of requests processed per second.
- **System Metrics**: CPU and memory usage.

## Monitoring and Resource Management

The inference engine includes a background monitoring thread that periodically checks system resources and takes action if thresholds are exceeded. Key monitoring features include:

- **Memory Usage**: Triggers garbage collection and cache clearing if memory usage exceeds the high watermark.
- **CPU Usage**: Enables request throttling if CPU usage exceeds a configurable threshold.
- **Periodic Logging**: Logs performance metrics and system resource usage at regular intervals (debug mode only).

## Caching

The engine supports request deduplication and result caching to improve efficiency. The cache is implemented using an `LRUTTLCache` (Least Recently Used with Time-to-Live) and can be configured with a maximum size and TTL (Time-to-Live) for entries.

### Cache Operations

- **Cache Key Generation**: A cache key is generated from the input features using a hash function.
- **Cache Hit/Miss**: The engine checks the cache before processing a request. If a cache hit occurs, the cached result is returned immediately.
- **Cache Clearing**: The cache can be cleared manually or automatically when memory usage is high.

## Quantization

Quantization is supported for both model weights and input features. The `Quantizer` class handles quantization operations and can be configured with various quantization modes and parameters.

### Quantization Modes

- **Dynamic Per Batch**: Quantization parameters are calculated dynamically for each batch.
- **Static**: Quantization parameters are precomputed and reused for all batches.

### Quantization Types

- **INT8**: 8-bit integer quantization.
- **FP16**: 16-bit floating-point quantization.

## Batch Processing

The engine supports adaptive batching to improve throughput. The `BatchProcessor` class manages batch processing and can be configured with various strategies:

### Batch Processing Strategies

- **Adaptive**: Dynamically adjusts batch sizes based on system load and resource availability.
- **Fixed**: Uses a fixed batch size for all requests.
- **Priority**: Processes high-priority requests first.

## Signal Handling

The engine handles termination signals (`SIGINT` and `SIGTERM`) for graceful shutdown. When a signal is received, the engine:

1. Stops accepting new requests.
2. Waits for active requests to complete.
3. Shuts down the batch processor and thread pool.
4. Clears caches and releases resources.

## Serialization

The engine can be serialized to disk using various formats (`joblib`, `pickle`, or `json`). Serialization includes:

- **Model**: The loaded model is saved in the specified format.
- **Configuration**: The engine configuration is saved as a JSON file.
- **Metrics**: Current performance metrics can be optionally saved as a JSON file.
- **Model Info**: Metadata about the loaded model is saved as a JSON file.

## Example: Saving the Engine

```python
save_path = "./saved_engine"
success = engine.save(save_path, save_format='joblib', include_config=True, include_metrics=True, overwrite=True)

if success:
    print("Engine saved successfully")
else:
    print("Failed to save engine")
```

## Example: Loading the Engine

```python
# Initialize the engine
engine = InferenceEngine()

# Load the saved model and configuration
model_path = os.path.join(save_path, "model.joblib")
engine.load_model(model_path)

# Load configuration (optional)
config_path = os.path.join(save_path, "config.json")
with open(config_path, 'r') as f:
    config_data = json.load(f)
engine.set_config(InferenceEngineConfig.from_dict(config_data))
```

## Error Handling

The engine includes comprehensive error handling and logging. Errors are logged with detailed information, and the engine state is updated to reflect errors. Key error handling features include:

- **Model Loading Errors**: Handles errors during model loading and updates the engine state to `ERROR`.
- **Inference Errors**: Logs errors during inference and updates performance metrics.
- **Resource Errors**: Handles memory and CPU usage errors by triggering cleanup and throttling.

## Debug Mode

When debug mode is enabled, the engine logs detailed information, including:

- Performance metrics.
- System resource usage.
- Cache statistics.
- Quantization and dequantization times.
- Batch processing details.

## Thread Safety

The engine is designed to be thread-safe and supports multiple concurrent requests. Thread safety is achieved using:

- **Locks**: Thread locks (`RLock`) are used to protect shared resources.
- **Thread Pool**: A `ThreadPoolExecutor` is used for parallel tasks.
- **Atomic Operations**: Critical operations are performed atomically to avoid race conditions.

## Best Practices

1. **Warm Up the Model**: Use the `_warmup_model` method to stabilize performance before handling real requests.
2. **Monitor System Resources**: Enable monitoring to detect and handle resource constraints.
3. **Use Caching**: Enable request deduplication and result caching to improve efficiency.
4. **Configure Quantization**: Use quantization to reduce memory usage and improve inference speed.
5. **Handle Errors Gracefully**: Implement error handling and retry logic for robust operation.

## Conclusion

The `InferenceEngine` class provides a powerful and flexible framework for high-performance model inference. With support for Intel optimizations, adaptive batching, quantization, and comprehensive monitoring, it is well-suited for production environments. The engine's thread safety, graceful shutdown, and error handling features ensure reliable and efficient operation.

For more details, refer to the source code and inline documentation.