# Prediction Request (`modules/engine/prediction_request.py`)

## Overview

The Prediction Request module provides a standardized data container for representing prediction requests in the dynamic batching system. It encapsulates all necessary information for processing requests through the inference pipeline with proper prioritization and metadata tracking.

## Features

- **Request Encapsulation**: Complete request data packaging
- **Priority Queue Support**: Built-in comparison for priority-based processing
- **Metadata Tracking**: Timestamps, timeouts, and request identifiers
- **Async Integration**: Support for async futures and callbacks
- **Lightweight Design**: Minimal overhead data structure

## Core Classes

### PredictionRequest

A dataclass that represents a single prediction request:

```python
@dataclass
class PredictionRequest:
    id: str                           # Unique request identifier
    features: np.ndarray             # Input features for prediction
    priority: int = 0                # Request priority (lower = higher priority)
    timestamp: float = 0.0           # Request creation timestamp
    future: Optional[Any] = None     # Async future for result delivery
    timeout_ms: Optional[float] = None  # Request timeout in milliseconds
```

## Usage Examples

### Basic Request Creation

```python
from modules.engine.prediction_request import PredictionRequest
import numpy as np
import time
import uuid

# Create basic prediction request
features = np.array([[1.0, 2.0, 3.0, 4.0]])  # Single sample
request = PredictionRequest(
    id=str(uuid.uuid4()),
    features=features,
    timestamp=time.time()
)

print(f"Request ID: {request.id}")
print(f"Features shape: {request.features.shape}")
print(f"Priority: {request.priority}")
print(f"Timestamp: {request.timestamp}")
```

### Priority-Based Request Processing

```python
from modules.engine.prediction_request import PredictionRequest
import numpy as np
import time
import heapq
import uuid

# Create requests with different priorities
requests = []

# High priority request (emergency/real-time)
high_priority_request = PredictionRequest(
    id=f"high_{uuid.uuid4()}",
    features=np.random.rand(1, 10),
    priority=0,  # Highest priority
    timestamp=time.time()
)

# Normal priority request
normal_request = PredictionRequest(
    id=f"normal_{uuid.uuid4()}",
    features=np.random.rand(1, 10),
    priority=5,  # Normal priority
    timestamp=time.time()
)

# Low priority request (batch processing)
low_priority_request = PredictionRequest(
    id=f"low_{uuid.uuid4()}",
    features=np.random.rand(1, 10),
    priority=10,  # Low priority
    timestamp=time.time()
)

# Add to priority queue
priority_queue = []
heapq.heappush(priority_queue, high_priority_request)
heapq.heappush(priority_queue, low_priority_request)
heapq.heappush(priority_queue, normal_request)

# Process in priority order
print("Processing requests in priority order:")
while priority_queue:
    request = heapq.heappop(priority_queue)
    print(f"Processing: {request.id}, Priority: {request.priority}")
```

### Async Request Processing

```python
import asyncio
from concurrent.futures import Future
from modules.engine.prediction_request import PredictionRequest
import numpy as np
import time
import uuid

async def async_prediction_processing():
    """Example of async request processing with futures"""
    
    # Create request with future for async result delivery
    future = asyncio.Future()
    
    request = PredictionRequest(
        id=str(uuid.uuid4()),
        features=np.random.rand(5, 20),  # Batch of 5 samples
        priority=1,
        timestamp=time.time(),
        future=future,
        timeout_ms=5000  # 5 second timeout
    )
    
    # Simulate async processing
    async def process_request(req):
        """Simulate async inference processing"""
        print(f"Starting processing for request {req.id}")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Simulate prediction result
        batch_size = req.features.shape[0]
        predictions = np.random.rand(batch_size, 1)
        
        # Complete the future with result
        if not req.future.done():
            req.future.set_result({
                'predictions': predictions,
                'request_id': req.id,
                'processing_time': 0.1,
                'batch_size': batch_size
            })
        
        print(f"Completed processing for request {req.id}")
    
    # Start processing
    task = asyncio.create_task(process_request(request))
    
    try:
        # Wait for result with timeout
        result = await asyncio.wait_for(request.future, timeout=request.timeout_ms/1000)
        print(f"Result received: {result}")
        
    except asyncio.TimeoutError:
        print(f"Request {request.id} timed out")
        # Cancel the processing task
        task.cancel()
    
    await task

# Run async processing
asyncio.run(async_prediction_processing())
```

### Batch Request Management

```python
from modules.engine.prediction_request import PredictionRequest
import numpy as np
import time
import uuid
from typing import List, Dict
from dataclasses import asdict

class BatchRequestManager:
    """Manager for handling batches of prediction requests"""
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.pending_requests: List[PredictionRequest] = []
    
    def add_request(self, features: np.ndarray, priority: int = 5, 
                   timeout_ms: float = 10000) -> PredictionRequest:
        """Add a new prediction request"""
        request = PredictionRequest(
            id=str(uuid.uuid4()),
            features=features,
            priority=priority,
            timestamp=time.time(),
            timeout_ms=timeout_ms
        )
        
        self.pending_requests.append(request)
        return request
    
    def get_next_batch(self) -> List[PredictionRequest]:
        """Get the next batch of requests for processing"""
        if not self.pending_requests:
            return []
        
        # Sort by priority and timestamp
        self.pending_requests.sort()
        
        # Take up to max_batch_size requests
        batch_size = min(self.max_batch_size, len(self.pending_requests))
        batch = self.pending_requests[:batch_size]
        self.pending_requests = self.pending_requests[batch_size:]
        
        return batch
    
    def check_timeouts(self) -> List[PredictionRequest]:
        """Check for and remove timed out requests"""
        current_time = time.time()
        timed_out = []
        active_requests = []
        
        for request in self.pending_requests:
            if (request.timeout_ms and 
                (current_time - request.timestamp) * 1000 > request.timeout_ms):
                timed_out.append(request)
            else:
                active_requests.append(request)
        
        self.pending_requests = active_requests
        return timed_out
    
    def get_stats(self) -> Dict:
        """Get statistics about pending requests"""
        if not self.pending_requests:
            return {'pending_count': 0, 'avg_wait_time': 0, 'priority_distribution': {}}
        
        current_time = time.time()
        wait_times = [(current_time - req.timestamp) for req in self.pending_requests]
        priorities = [req.priority for req in self.pending_requests]
        
        priority_dist = {}
        for priority in priorities:
            priority_dist[priority] = priority_dist.get(priority, 0) + 1
        
        return {
            'pending_count': len(self.pending_requests),
            'avg_wait_time': sum(wait_times) / len(wait_times),
            'max_wait_time': max(wait_times),
            'priority_distribution': priority_dist
        }

# Example usage
manager = BatchRequestManager(max_batch_size=16)

# Add various requests
for i in range(50):
    features = np.random.rand(1, 10)
    priority = np.random.choice([0, 1, 5, 10], p=[0.1, 0.2, 0.6, 0.1])  # Priority distribution
    timeout = np.random.choice([5000, 10000, 30000])  # Different timeouts
    
    request = manager.add_request(features, priority, timeout)
    
    # Simulate processing delay
    if i % 10 == 0:
        time.sleep(0.01)

# Process batches
print("Processing batches:")
batch_num = 1
while True:
    # Check for timeouts
    timed_out = manager.check_timeouts()
    if timed_out:
        print(f"Timed out requests: {len(timed_out)}")
    
    # Get next batch
    batch = manager.get_next_batch()
    if not batch:
        break
    
    print(f"Batch {batch_num}: {len(batch)} requests")
    print(f"  Priorities: {[req.priority for req in batch]}")
    print(f"  Average wait time: {sum((time.time() - req.timestamp) for req in batch) / len(batch):.3f}s")
    
    batch_num += 1
    
    # Simulate batch processing time
    time.sleep(0.05)

# Final stats
stats = manager.get_stats()
print(f"Final stats: {stats}")
```

### Request Serialization and Persistence

```python
import json
import pickle
from modules.engine.prediction_request import PredictionRequest
import numpy as np
import time
import uuid

def serialize_request_json(request: PredictionRequest) -> str:
    """Serialize request to JSON (features as list)"""
    request_dict = {
        'id': request.id,
        'features': request.features.tolist(),
        'features_shape': request.features.shape,
        'features_dtype': str(request.features.dtype),
        'priority': request.priority,
        'timestamp': request.timestamp,
        'timeout_ms': request.timeout_ms
    }
    return json.dumps(request_dict)

def deserialize_request_json(json_str: str) -> PredictionRequest:
    """Deserialize request from JSON"""
    data = json.loads(json_str)
    
    # Reconstruct numpy array
    features = np.array(data['features'], dtype=data['features_dtype'])
    features = features.reshape(data['features_shape'])
    
    return PredictionRequest(
        id=data['id'],
        features=features,
        priority=data['priority'],
        timestamp=data['timestamp'],
        timeout_ms=data['timeout_ms']
    )

def serialize_request_pickle(request: PredictionRequest) -> bytes:
    """Serialize request to pickle (more efficient for numpy arrays)"""
    return pickle.dumps(request)

def deserialize_request_pickle(pickle_bytes: bytes) -> PredictionRequest:
    """Deserialize request from pickle"""
    return pickle.loads(pickle_bytes)

# Example usage
original_request = PredictionRequest(
    id=str(uuid.uuid4()),
    features=np.random.rand(10, 50).astype(np.float32),
    priority=3,
    timestamp=time.time(),
    timeout_ms=15000
)

# Test JSON serialization
json_serialized = serialize_request_json(original_request)
json_deserialized = deserialize_request_json(json_serialized)

print("JSON Serialization Test:")
print(f"  Original features shape: {original_request.features.shape}")
print(f"  Deserialized features shape: {json_deserialized.features.shape}")
print(f"  Features equal: {np.array_equal(original_request.features, json_deserialized.features)}")
print(f"  Serialized size: {len(json_serialized)} bytes")

# Test pickle serialization  
pickle_serialized = serialize_request_pickle(original_request)
pickle_deserialized = deserialize_request_pickle(pickle_serialized)

print("\nPickle Serialization Test:")
print(f"  Original features shape: {original_request.features.shape}")
print(f"  Deserialized features shape: {pickle_deserialized.features.shape}")
print(f"  Features equal: {np.array_equal(original_request.features, pickle_deserialized.features)}")
print(f"  Serialized size: {len(pickle_serialized)} bytes")
```

## Advanced Usage Patterns

### Request Validation

```python
from modules.engine.prediction_request import PredictionRequest
import numpy as np
from typing import Optional, Tuple

class RequestValidator:
    """Validator for prediction requests"""
    
    def __init__(self, expected_feature_dim: int, max_batch_size: int = 128):
        self.expected_feature_dim = expected_feature_dim
        self.max_batch_size = max_batch_size
    
    def validate_request(self, request: PredictionRequest) -> Tuple[bool, Optional[str]]:
        """Validate a prediction request"""
        
        # Check ID
        if not request.id or not isinstance(request.id, str):
            return False, "Invalid request ID"
        
        # Check features
        if not isinstance(request.features, np.ndarray):
            return False, "Features must be numpy array"
        
        if request.features.ndim != 2:
            return False, "Features must be 2D array (batch_size, features)"
        
        if request.features.shape[1] != self.expected_feature_dim:
            return False, f"Expected {self.expected_feature_dim} features, got {request.features.shape[1]}"
        
        if request.features.shape[0] > self.max_batch_size:
            return False, f"Batch size {request.features.shape[0]} exceeds maximum {self.max_batch_size}"
        
        # Check priority
        if not isinstance(request.priority, int) or request.priority < 0:
            return False, "Priority must be non-negative integer"
        
        # Check timestamp
        if request.timestamp < 0:
            return False, "Timestamp must be non-negative"
        
        # Check timeout
        if request.timeout_ms is not None and request.timeout_ms <= 0:
            return False, "Timeout must be positive"
        
        return True, None

# Example usage
validator = RequestValidator(expected_feature_dim=10, max_batch_size=64)

# Valid request
valid_request = PredictionRequest(
    id="valid_001",
    features=np.random.rand(5, 10),
    priority=1,
    timestamp=time.time(),
    timeout_ms=5000
)

is_valid, error_msg = validator.validate_request(valid_request)
print(f"Valid request: {is_valid}, Error: {error_msg}")

# Invalid request (wrong feature dimension)
invalid_request = PredictionRequest(
    id="invalid_001", 
    features=np.random.rand(5, 15),  # Wrong dimension
    priority=1,
    timestamp=time.time()
)

is_valid, error_msg = validator.validate_request(invalid_request)
print(f"Invalid request: {is_valid}, Error: {error_msg}")
```

## Best Practices

### 1. Request ID Generation

```python
import uuid
from datetime import datetime

def generate_request_id(prefix: str = "req") -> str:
    """Generate unique request ID with timestamp and UUID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{short_uuid}"

# Example usage
request_id = generate_request_id("inference")
print(f"Generated ID: {request_id}")
```

### 2. Memory-Efficient Feature Handling

```python
# Use appropriate data types to minimize memory usage
features_float32 = np.array(data, dtype=np.float32)  # Instead of float64
features_int16 = np.array(indices, dtype=np.int16)    # For small integers

# For large batch requests, consider memory mapping
def create_memory_efficient_request(data_file: str, request_id: str) -> PredictionRequest:
    """Create request with memory-mapped features"""
    features = np.memmap(data_file, dtype=np.float32, mode='r')
    features = features.reshape(-1, 10)  # Reshape to proper dimensions
    
    return PredictionRequest(
        id=request_id,
        features=features,
        priority=5,
        timestamp=time.time()
    )
```

### 3. Priority Management

```python
# Define priority levels as constants
class Priority:
    EMERGENCY = 0     # Real-time critical requests
    HIGH = 1          # High priority interactive requests  
    NORMAL = 5        # Standard requests
    LOW = 10          # Batch processing requests
    BACKGROUND = 20   # Background/analytical requests

# Usage
emergency_request = PredictionRequest(
    id="emergency_001",
    features=features,
    priority=Priority.EMERGENCY
)
```

## Related Documentation

- [Dynamic Batcher Documentation](dynamic_batcher.md)
- [Inference Engine Documentation](inference_engine.md)
- [Batch Processor Documentation](batch_processor.md)
- [Performance Metrics Documentation](performance_metrics.md)
