# Module: `inference_engine`

## Overview
This module implements a **high-performance CPU-based inference engine** designed to
handle machine learning workloads with **dynamic batching**, **quantization**,
**memory pooling**, **NUMA-aware scheduling**, and **real-time performance monitoring**.

The engine supports various model types (Scikit-learn, XGBoost, LightGBM, ONNX, Treelite),
and is equipped for **low-latency predictions** with **caching, preprocessing, and compilation** features.

---

## Prerequisites
- Python >= 3.8
- Recommended packages:
  ```bash
  pip install numpy psutil joblib onnxruntime xgboost lightgbm skl2onnx treelite
  ```
- OS: Linux/macOS/Windows

## Installation
Include the module in your project and ensure dependencies are met:
```bash
pip install -r requirements.txt
```

---

## Usage
```python
from inference_engine import InferenceEngine

engine = InferenceEngine()
engine.load_model("model.pkl")
success, prediction, metadata = engine.predict(input_features)
```

---

## Configuration
`InferenceEngineConfig` provides the parameters for configuration. Here are key options:

| Parameter | Default | Description |
|----------|---------|-------------|
| `enable_batching` | `True` | Enables dynamic micro-batching |
| `enable_quantization` | `False` | Applies input quantization |
| `enable_model_quantization` | `False` | Applies model weight quantization |
| `enable_request_deduplication` | `True` | Caches request outputs |
| `num_threads` | `os.cpu_count()` | Thread pool size |
| `max_batch_size` | `64` | Maximum batch size for prediction |
| `max_concurrent_requests` | `100` | Max inflight requests |

---

## Architecture

### Major Components:
- **InferenceEngine**: Main interface for loading models and executing inference.
- **DynamicBatcher**: Groups prediction requests into batches based on priority and timing.
- **MemoryPool**: Caches preallocated `numpy.ndarray` for fast reuse.
- **PerformanceMetrics**: Collects stats on latency, throughput, errors.
- **Quantizer / Preprocessor**: Optional modules for preprocessing and quantizing features.
- **LRUTTLCache**: Used for result and feature caching.

### Supported Model Types:
- Scikit-learn
- XGBoost
- LightGBM
- Custom Python models (with `predict` method)
- ONNX compiled models
- Treelite compiled models

---

## Testing
```bash
pytest tests/
```
To validate engine status and readiness:
```python
engine.validate_model()
```

---

## Security & Compliance
- Prediction results and cache keys are hashed.
- Buffers are zeroed out before reuse.
- GC callback tracking is enabled to monitor memory behavior.

---

## Versioning and Metadata
> Last Updated: 2025-04-28

---

## Classes and Functions

### `PredictionRequest`
```python
@dataclass
class PredictionRequest:
```
- **Description**: Container for individual prediction requests.
- **Attributes**:
  - `id (str)`: Unique ID.
  - `features (np.ndarray)`: Input data.
  - `priority (int)`: Priority level.
  - `timestamp (float)`: Request time.
  - `future (Future)`: Future to hold async result.
  - `timeout_ms (float)`: Optional timeout.

---

### `DynamicBatcher`
```python
class DynamicBatcher:
```
- **Description**: Asynchronously batches prediction requests by priority and latency constraints.
- **Methods**:
  - `start()` / `stop()`: Start or stop worker thread.
  - `enqueue(request)`: Add a request to the batch queue.
  - `get_stats()`: Get batching statistics.

---

### `MemoryPool`
```python
class MemoryPool:
```
- **Description**: Manages reusable memory buffers to avoid repeated allocations.
- **Methods**:
  - `get_buffer(shape)`: Fetch or allocate a buffer.
  - `return_buffer(buffer)`: Recycle a buffer.
  - `clear()`: Reset all buffers.
  - `get_stats()`: Return buffer pool stats.

---

### `PerformanceMetrics`
```python
class PerformanceMetrics:
```
- **Description**: Records inference times, batch sizes, and error rates.
- **Methods**:
  - `update_inference()`, `record_error()`, `get_metrics()`

---

### `InferenceEngine`
```python
class InferenceEngine:
```
- **Description**: Full-featured CPU inference engine with pluggable components.
- **Constructor**:
  ```python
  def __init__(self, config: Optional[InferenceEngineConfig] = None)
  ```
- **Key Methods**:
  - `load_model(model_path, model_type, compile_model)`: Load and optionally compile a model.
  - `predict(features)`: Make synchronous prediction.
  - `enqueue_prediction(features)`: Submit async prediction request.
  - `predict_batch(features_list)`: Batch prediction interface.
  - `get_performance_metrics()`: Retrieve full stats.
  - `shutdown()`: Clean up resources.
  - `validate_model()`: Run internal model verification.

---

## Advanced Features
- **Model Compilation**:
  - ONNX for Scikit-learn
  - Treelite for XGBoost/LightGBM
- **Cache Deduplication**:
  - Caches hash of input features to return repeated results faster
- **Thread Affinity**:
  - Optionally pins threads to CPU cores using `psutil`
- **GC Monitoring**:
  - Tracks significant garbage collection pauses

---

## Monitoring Metrics Snapshot (Example)
```json
{
  "avg_inference_time_ms": 3.4,
  "total_requests": 1245,
  "cache_hit_rate": 0.83,
  "memory_pool": {
    "total_buffers": 12,
    "memory_usage_mb": 0.84
  },
  "dynamic_batcher": {
    "processed_batches": 102,
    "avg_batch_size": 7.3
  }
}
```

---

