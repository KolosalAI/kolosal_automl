# InferenceEngine

## Overview
This module provides a high-performance ML inference engine designed for optimal hardware utilization, efficient memory management, and minimal inference latency. The system implements numerous optimization techniques including dynamic batching, memory pooling, model compilation, quantization, and extensive performance monitoring.

## Prerequisites
- Python ≥3.8
- Core dependencies:
  - `numpy`
  - `threading`
  - `logging`
  - `concurrent.futures`

- Optional dependencies:
  ```python
  # Performance optimization
  psutil
  joblib
  threadpoolctl
  
  # ML frameworks
  scikit-learn
  xgboost
  lightgbm
  
  # Model compilation
  onnx
  onnxruntime
  treelite
  treelite_runtime
  ```

## Installation
```bash
pip install numpy
pip install -r requirements.txt  # For optional dependencies
```

## Usage
```python
from inference_engine import InferenceEngine, InferenceEngineConfig

# Create configuration with desired settings
config = InferenceEngineConfig(
    enable_batching=True,
    max_batch_size=32,
    enable_feature_scaling=True,
    enable_request_deduplication=True
)

# Initialize inference engine
engine = InferenceEngine(config)

# Load model
engine.load_model("models/my_model.pkl")

# Make predictions
features = np.array([[1.2, 3.4, 5.6, 7.8]])
success, predictions, metadata = engine.predict(features)

# For batch processing
features_batch = np.random.random((10, 4))
futures = [engine.enqueue_prediction(features, timeout_ms=100) 
          for features in features_batch]
results = [future.result() for future in futures]

# Shutdown engine when done
engine.shutdown()
```

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_batch_size` | 64 | Maximum number of requests to batch together |
| `batch_timeout` | 0.01 | Max wait time (seconds) for batch formation |
| `enable_feature_scaling` | False | Apply preprocessing to inputs |
| `enable_quantization` | False | Use quantized operations |
| `enable_jit` | False | JIT compile model if possible |
| `enable_batching` | True | Enable dynamic batching of requests |
| `enable_request_deduplication` | True | Cache results for duplicate inputs |
| `max_concurrent_requests` | 1000 | Max concurrent inference requests |
| `num_threads` | CPU count | Number of threads for compute |

## Architecture
The InferenceEngine architecture consists of several core components:

1. **Dynamic Batcher**: Groups prediction requests and processes them together to maximize hardware utilization
2. **Memory Pool**: Pre-allocates and reuses memory buffers to reduce allocation overhead
3. **Model Compilation**: Converts models to optimized formats via ONNX or Treelite
4. **Result/Feature Caching**: Multi-level caching for deduplication and faster processing
5. **Performance Metrics**: Comprehensive monitoring and resource management

**Data Flow**: Request → Batching → Preprocessing → Quantization → Inference → Result Distribution

---

## Classes

### `InferenceEngine`
```python
class InferenceEngine:
```
- **Description**:  
  High-performance inference engine with advanced optimizations for ML workloads.

- **Attributes**:  
  - `config (InferenceEngineConfig)`: Configuration for the inference engine
  - `state (EngineState)`: Current state of the engine
  - `model`: Loaded ML model
  - `model_type (ModelType)`: Type of the loaded model
  - `compiled_model`: Optimized compiled version of the model
  - `feature_names (List[str])`: Names of input features

- **Constructor**:
  ```python
  def __init__(self, config: Optional[InferenceEngineConfig] = None)
  ```
  - **Parameters**:
    - `config (Optional[InferenceEngineConfig])`: Configuration for the inference engine. Default is None (uses default config).
  - **Raises**:  
    - Does not explicitly raise exceptions.

- **Methods**:  
  - `load_model(model_path: str, model_type: Optional[ModelType] = None, compile_model: bool = None) -> bool`:
    - **Description**:  
      Loads a pre-trained model from file with optimized processing.
    - **Parameters**:
      - `model_path (str)`: Path to the saved model file
      - `model_type (Optional[ModelType])`: Type of model being loaded (auto-detected if None)
      - `compile_model (bool)`: Whether to compile model for faster inference (uses config setting if None)
    - **Returns**:
      - `bool`: Success status

  - `predict(features: np.ndarray) -> Tuple[bool, np.ndarray, Dict[str, Any]]`:
    - **Description**:  
      Makes optimized prediction with input features.
    - **Parameters**:
      - `features (np.ndarray)`: Input features as numpy array
    - **Returns**:
      - `Tuple[bool, np.ndarray, Dict[str, Any]]`: Tuple of (success_flag, predictions, metadata)

  - `enqueue_prediction(features: np.ndarray, priority: BatchPriority = BatchPriority.NORMAL, timeout_ms: Optional[float] = None) -> Any`:
    - **Description**:  
      Enqueues a prediction request to be processed asynchronously with the dynamic batcher.
    - **Parameters**:
      - `features (np.ndarray)`: Input features
      - `priority (BatchPriority)`: Processing priority
      - `timeout_ms (Optional[float])`: Optional timeout in milliseconds
    - **Returns**:
      - `Future`: Future object with the prediction result

  - `predict_batch(features_list: List[np.ndarray], timeout: Optional[float] = None) -> List[Tuple[bool, np.ndarray, Dict[str, Any]]]`:
    - **Description**:  
      Optimized batch prediction for multiple inputs with vectorized operations.
    - **Parameters**:
      - `features_list (List[np.ndarray])`: List of feature arrays
      - `timeout (Optional[float])`: Optional timeout in seconds
    - **Returns**:
      - `List[Tuple[bool, np.ndarray, Dict[str, Any]]]`: List of (success, predictions, metadata) tuples

  - `get_performance_metrics() -> Dict[str, Any]`:
    - **Description**:  
      Gets comprehensive performance metrics from all components.
    - **Parameters**:
      - None
    - **Returns**:
      - `Dict[str, Any]`: Dictionary of performance metrics

  - `shutdown()`:
    - **Description**:  
      Shuts down the engine and releases resources with proper cleanup.
    - **Parameters**:
      - None
    - **Returns**:
      - None

  - `validate_model() -> Dict[str, Any]`:
    - **Description**:  
      Performs validation checks on the loaded model to ensure it can make predictions correctly.
    - **Parameters**:
      - None
    - **Returns**:
      - `Dict[str, Any]`: Dictionary with validation results

---

### `DynamicBatcher`
```python
class DynamicBatcher:
```
- **Description**:  
  Improved dynamic batching system that efficiently groups requests and processes them together to maximize hardware utilization.

- **Attributes**:  
  - `batch_processor (Callable)`: Function that processes batches
  - `max_batch_size (int)`: Maximum size of a batch
  - `max_wait_time_ms (float)`: Maximum wait time in milliseconds for batch formation
  - `request_queue`: Priority queue for pending requests

- **Constructor**:
  ```python
  def __init__(self, batch_processor: Callable, max_batch_size: int = 64, max_wait_time_ms: float = 10.0, max_queue_size: int = 1000)
  ```
  - **Parameters**:
    - `batch_processor (Callable)`: Function that processes batches
    - `max_batch_size (int)`: Maximum batch size. Default is 64.
    - `max_wait_time_ms (float)`: Maximum wait time in milliseconds. Default is 10.0.
    - `max_queue_size (int)`: Maximum size of the request queue. Default is 1000.
  - **Raises**:  
    - Does not explicitly raise exceptions.

- **Methods**:  
  - `start()`:
    - **Description**:  
      Starts the batcher worker thread.
    - **Parameters**:
      - None
    - **Returns**:
      - None

  - `stop(timeout: float = 2.0)`:
    - **Description**:  
      Stops the batcher worker thread.
    - **Parameters**:
      - `timeout (float)`: Maximum time to wait for thread termination. Default is 2.0.
    - **Returns**:
      - None

  - `enqueue(request: PredictionRequest) -> bool`:
    - **Description**:  
      Adds a request to the batch queue.
    - **Parameters**:
      - `request (PredictionRequest)`: The prediction request to add to the queue
    - **Returns**:
      - `bool`: Success status (False if queue is full)

  - `get_stats() -> Dict[str, Any]`:
    - **Description**:  
      Gets batcher statistics.
    - **Parameters**:
      - None
    - **Returns**:
      - `Dict[str, Any]`: Dictionary of batcher statistics

---

### `MemoryPool`
```python
class MemoryPool:
```
- **Description**:  
  Manages a pool of pre-allocated memory buffers to reduce allocation overhead and improve memory reuse for better CPU cache utilization.

- **Attributes**:  
  - `max_buffers (int)`: Maximum number of buffers to keep per shape
  - `buffer_pools (Dict)`: Dictionary mapping shapes to lists of available buffers

- **Constructor**:
  ```python
  def __init__(self, max_buffers: int = 32)
  ```
  - **Parameters**:
    - `max_buffers (int)`: Maximum number of buffers per shape. Default is 32.
  - **Raises**:  
    - Does not explicitly raise exceptions.

- **Methods**:  
  - `get_buffer(shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray`:
    - **Description**:  
      Gets a buffer of the specified shape and type from the pool or creates one.
    - **Parameters**:
      - `shape (Tuple[int, ...])`: Shape of the buffer
      - `dtype`: Data type of the buffer. Default is np.float32.
    - **Returns**:
      - `np.ndarray`: Buffer of the requested shape and type

  - `return_buffer(buffer: np.ndarray)`:
    - **Description**:  
      Returns a buffer to the pool for reuse.
    - **Parameters**:
      - `buffer (np.ndarray)`: Buffer to return to the pool
    - **Returns**:
      - None

  - `clear()`:
    - **Description**:  
      Clears all buffers from the pool.
    - **Parameters**:
      - None
    - **Returns**:
      - None

  - `get_stats() -> Dict[str, Any]`:
    - **Description**:  
      Gets statistics about the memory pool.
    - **Parameters**:
      - None
    - **Returns**:
      - `Dict[str, Any]`: Dictionary of memory pool statistics

---

### `PerformanceMetrics`
```python
class PerformanceMetrics:
```
- **Description**:  
  Thread-safe container for tracking performance metrics.

- **Attributes**:  
  - `window_size (int)`: Number of measurements to keep in the sliding window
  - `inference_times (List[float])`: List of recent inference times
  - `batch_sizes (List[int])`: List of recent batch sizes
  - `total_requests (int)`: Total number of processed requests
  - `total_errors (int)`: Total number of errors

- **Constructor**:
  ```python
  def __init__(self, window_size: int = 100)
  ```
  - **Parameters**:
    - `window_size (int)`: Size of the sliding window for metrics. Default is 100.
  - **Raises**:  
    - Does not explicitly raise exceptions.

- **Methods**:  
  - `update_inference(inference_time: float, batch_size: int, preprocessing_time: float = 0.0, quantization_time: float = 0.0)`:
    - **Description**:  
      Updates inference metrics with thread safety.
    - **Parameters**:
      - `inference_time (float)`: Time taken for inference in seconds
      - `batch_size (int)`: Size of the processed batch
      - `preprocessing_time (float)`: Time taken for preprocessing. Default is 0.0.
      - `quantization_time (float)`: Time taken for quantization. Default is 0.0.
    - **Returns**:
      - None

  - `record_error()`:
    - **Description**:  
      Records an inference error.
    - **Parameters**:
      - None
    - **Returns**:
      - None

  - `record_cache_hit()`:
    - **Description**:  
      Records a cache hit.
    - **Parameters**:
      - None
    - **Returns**:
      - None

  - `record_cache_miss()`:
    - **Description**:  
      Records a cache miss.
    - **Parameters**:
      - None
    - **Returns**:
      - None

  - `get_metrics() -> Dict[str, Any]`:
    - **Description**:  
      Gets current performance metrics as a dictionary.
    - **Parameters**:
      - None
    - **Returns**:
      - `Dict[str, Any]`: Dictionary of performance metrics

---

### `PredictionRequest`
```python
@dataclass
class PredictionRequest:
```
- **Description**:  
  Container for a prediction request with metadata.

- **Attributes**:  
  - `id (str)`: Unique identifier for the request
  - `features (np.ndarray)`: Input features for prediction
  - `priority (int)`: Priority of the request (lower value = higher priority). Default is 0.
  - `timestamp (float)`: When the request was created. Default is 0.0.
  - `future (Optional[Any])`: Future object to store the result. Default is None.
  - `timeout_ms (Optional[float])`: Optional timeout in milliseconds. Default is None.

- **Methods**:  
  - `__lt__(other)`:
    - **Description**:  
      Comparison for priority queue (lower value = higher priority).
    - **Parameters**:
      - `other (PredictionRequest)`: Another request to compare with
    - **Returns**:
      - `bool`: True if this request has higher priority than the other

## Testing
```bash
# Run unit tests
python -m unittest tests/test_inference_engine.py

# Run integration tests
python -m unittest tests/test_integration.py
```

## Security & Compliance
- Thread-safe operations for concurrent processing
- Memory zeroing before buffer reuse to prevent data leakage
- Garbage collection hooks to manage memory usage
- CPU and memory monitors to prevent resource exhaustion

> Last Updated: 2025-05-11