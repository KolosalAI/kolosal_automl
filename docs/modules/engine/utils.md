# Engine Utils (`modules/engine/utils.py`)

## Overview

The Engine Utils module provides essential utility functions for safe serialization, JSON conversion, and handling of complex Python objects in a machine learning pipeline. It includes specialized functions for dealing with threading locks, unpicklable objects, and safe JSON serialization of ML-specific data types.

## Features

- **Safe Serialization**: Handles unpicklable objects (locks, queues) during serialization
- **Object Scrubbing**: Recursively removes problematic objects while preserving structure
- **JSON Conversion**: Type-safe conversion of complex objects to JSON-serializable formats
- **Thread Safety**: Proper handling of threading primitives in serialization
- **ML Integration**: Specialized support for NumPy arrays and ML objects

## Core Functions

### Object Scrubbing and Serialization

#### `_scrub(obj, memo=None)`

Recursively removes unpicklable objects (locks, queues) from nested data structures:

```python
def _scrub(obj, memo=None):
    """
    Return a version of *obj* with every nested Lock/RLock replaced by None,
    leaving primitives, classes, functions, modules, etc. unchanged.
    Uses *memo* to stay cycle‑safe.
    """
```

#### `_patch_pickle_for_locks()`

Registers pickle reducers to handle unpicklable objects safely:

```python
def _patch_pickle_for_locks():
    """Register reducers that replace unpicklable objects with None when pickling."""
```

### JSON Serialization

#### `_json_safe(obj)`

Single-dispatch function for safe JSON conversion of complex objects:

```python
@singledispatch
def _json_safe(obj):
    """Best‑effort conversion to something the std‑lib json encoder accepts."""
```

## Usage Examples

### Safe Object Serialization

```python
from modules.engine.utils import _scrub, _patch_pickle_for_locks
import threading
import pickle
import json

# Apply pickle patches for thread safety
_patch_pickle_for_locks()

class MLModel:
    """Example ML model with threading components"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.weights = {'layer1': [1.0, 2.0, 3.0], 'layer2': [4.0, 5.0]}
        self.config = {'learning_rate': 0.01, 'batch_size': 32}
        self.training_thread = None
        self.results_queue = None
        
    def setup_training(self):
        """Setup training components with threading"""
        import queue
        self.results_queue = queue.Queue()
        self.training_thread = threading.Thread(target=self._dummy_training)
    
    def _dummy_training(self):
        """Dummy training method"""
        pass

# Create model with unpicklable components
model = MLModel()
model.setup_training()

print("Original model components:")
print(f"  Lock: {model.lock}")
print(f"  Queue: {model.results_queue}")
print(f"  Thread: {model.training_thread}")

# Scrub unpicklable objects
scrubbed_model = _scrub(model)

print("\nScrubbed model components:")
print(f"  Lock: {scrubbed_model.lock}")
print(f"  Queue: {scrubbed_model.results_queue}")
print(f"  Thread: {scrubbed_model.training_thread}")
print(f"  Weights preserved: {scrubbed_model.weights}")
print(f"  Config preserved: {scrubbed_model.config}")

# Now the model can be safely pickled
try:
    pickled_data = pickle.dumps(scrubbed_model)
    unpickled_model = pickle.loads(pickled_data)
    print("✓ Serialization successful")
except Exception as e:
    print(f"✗ Serialization failed: {e}")
```

### Complex Data Structure Scrubbing

```python
from modules.engine.utils import _scrub
import threading
import queue
import numpy as np

# Create complex nested structure with unpicklable objects
complex_data = {
    'model_configs': [
        {
            'name': 'model_1',
            'lock': threading.Lock(),
            'weights': np.array([1.0, 2.0, 3.0]),
            'metadata': {
                'trained_at': '2023-01-01',
                'performance_queue': queue.Queue(),
                'nested_data': {
                    'sub_lock': threading.RLock(),
                    'values': [1, 2, 3]
                }
            }
        },
        {
            'name': 'model_2', 
            'lock': threading.Lock(),
            'weights': np.array([4.0, 5.0, 6.0])
        }
    ],
    'global_lock': threading.Lock(),
    'settings': {
        'batch_size': 32,
        'learning_rate': 0.01,
        'worker_lock': threading.Lock()
    }
}

print("Original structure:")
print(f"  Global lock: {complex_data['global_lock']}")
print(f"  Model 1 lock: {complex_data['model_configs'][0]['lock']}")
print(f"  Performance queue: {complex_data['model_configs'][0]['metadata']['performance_queue']}")

# Scrub the entire structure
scrubbed_data = _scrub(complex_data)

print("\nScrubbed structure:")
print(f"  Global lock: {scrubbed_data['global_lock']}")
print(f"  Model 1 lock: {scrubbed_data['model_configs'][0]['lock']}")
print(f"  Performance queue: {scrubbed_data['model_configs'][0]['metadata']['performance_queue']}")
print(f"  Values preserved: {scrubbed_data['model_configs'][0]['metadata']['values']}")
print(f"  Weights preserved: {scrubbed_data['model_configs'][0]['weights']}")

# Verify serializability
try:
    import pickle
    pickle.dumps(scrubbed_data)
    print("✓ Complex structure is now serializable")
except Exception as e:
    print(f"✗ Serialization failed: {e}")
```

### JSON-Safe Conversion

```python
from modules.engine.utils import _json_safe
import numpy as np
from enum import Enum
import json

# Define example enum
class ModelType(Enum):
    LINEAR = "linear"
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"

# Create complex ML data structure
ml_data = {
    'model_type': ModelType.NEURAL_NETWORK,
    'weights': np.array([[1.0, 2.0], [3.0, 4.0]]),
    'biases': np.array([0.5, -0.2]),
    'metrics': {
        'accuracy': np.float64(0.95),
        'loss': np.float32(0.02),
        'epochs': np.int32(100)
    },
    'predictions': np.array([0.1, 0.9, 0.3, 0.8]),
    'feature_names': ['feature_1', 'feature_2'],
    'model_class': type(object),  # Non-serializable class
    'training_data_shape': (1000, 2),
    'custom_object': object()  # Non-serializable object
}

print("Original data types:")
for key, value in ml_data.items():
    print(f"  {key}: {type(value)} = {value}")

# Convert to JSON-safe format
json_safe_data = _json_safe(ml_data)

print("\nJSON-safe conversion:")
for key, value in json_safe_data.items():
    print(f"  {key}: {type(value)} = {value}")

# Verify JSON serializability
try:
    json_string = json.dumps(json_safe_data, indent=2)
    print("✓ Successfully converted to JSON")
    print(f"JSON preview: {json_string[:200]}...")
except Exception as e:
    print(f"✗ JSON conversion failed: {e}")
```

### Advanced Serialization Patterns

```python
from modules.engine.utils import _scrub, _json_safe, _patch_pickle_for_locks
import threading
import pickle
import json
import numpy as np
from typing import Any, Dict

class SerializationManager:
    """Advanced serialization manager using utils functions"""
    
    def __init__(self):
        # Apply pickle patches
        _patch_pickle_for_locks()
    
    def safe_pickle_dump(self, obj: Any, filepath: str) -> bool:
        """Safely pickle an object, scrubbing unpicklable components"""
        try:
            # Scrub unpicklable objects
            clean_obj = _scrub(obj)
            
            # Pickle the cleaned object
            with open(filepath, 'wb') as f:
                pickle.dump(clean_obj, f)
            
            print(f"✓ Object safely pickled to {filepath}")
            return True
            
        except Exception as e:
            print(f"✗ Pickle dump failed: {e}")
            return False
    
    def safe_json_dump(self, obj: Any, filepath: str) -> bool:
        """Safely convert object to JSON and save"""
        try:
            # Convert to JSON-safe format
            json_safe_obj = _json_safe(obj)
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(json_safe_obj, f, indent=2)
            
            print(f"✓ Object safely converted to JSON at {filepath}")
            return True
            
        except Exception as e:
            print(f"✗ JSON dump failed: {e}")
            return False
    
    def serialize_ml_experiment(self, experiment_data: Dict) -> Dict[str, bool]:
        """Serialize ML experiment data in multiple formats"""
        results = {}
        
        # Try pickle serialization
        results['pickle'] = self.safe_pickle_dump(
            experiment_data, 
            f"experiment_{experiment_data.get('id', 'unknown')}.pkl"
        )
        
        # Try JSON serialization  
        results['json'] = self.safe_json_dump(
            experiment_data,
            f"experiment_{experiment_data.get('id', 'unknown')}.json"
        )
        
        return results

# Example ML experiment with mixed data types
experiment = {
    'id': 'exp_001',
    'model': {
        'weights': np.random.randn(10, 5),
        'biases': np.random.randn(5),
        'lock': threading.Lock(),  # Unpicklable
        'training_queue': None     # Will be set to Queue later
    },
    'metrics': {
        'train_accuracy': np.float64(0.95),
        'val_accuracy': np.float64(0.92),
        'train_loss': np.float32(0.05),
        'val_loss': np.float32(0.08)
    },
    'config': {
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100,
        'optimizer': 'adam'
    },
    'data_stats': {
        'train_shape': (10000, 10),
        'val_shape': (2000, 10),
        'feature_means': np.random.randn(10),
        'feature_stds': np.random.randn(10)
    },
    'runtime_objects': {
        'main_lock': threading.Lock(),
        'worker_locks': [threading.Lock() for _ in range(4)],
        'thread_pool': None  # Would be ThreadPoolExecutor
    }
}

# Add queue to make it more complex
import queue
experiment['model']['training_queue'] = queue.Queue()
experiment['runtime_objects']['results_queue'] = queue.PriorityQueue()

# Serialize using manager
manager = SerializationManager()
results = manager.serialize_ml_experiment(experiment)

print(f"Serialization results: {results}")
```

### Custom Object Handling

```python
from modules.engine.utils import _json_safe, _scrub
import numpy as np
from typing import Any, Dict
import threading

class CustomMLObject:
    """Custom ML object with complex attributes"""
    
    def __init__(self, name: str):
        self.name = name
        self.lock = threading.Lock()
        self.weights = np.random.randn(5, 3)
        self.metadata = {
            'created_at': '2023-01-01',
            'version': '1.0',
            'internal_lock': threading.RLock()
        }
        self.nested_objects = [
            {'data': np.array([1, 2, 3]), 'lock': threading.Lock()},
            {'data': np.array([4, 5, 6]), 'lock': threading.Lock()}
        ]

# Register custom JSON conversion for our object
@_json_safe.register(CustomMLObject)
def _(obj: CustomMLObject):
    """Custom JSON conversion for CustomMLObject"""
    return {
        'object_type': 'CustomMLObject',
        'name': obj.name,
        'weights': _json_safe(obj.weights),
        'metadata': _json_safe(obj.metadata),
        'nested_objects': _json_safe(obj.nested_objects),
        'lock': '<Lock object removed>'
    }

# Test custom object handling
custom_obj = CustomMLObject("test_model")

print("Original object:")
print(f"  Name: {custom_obj.name}")
print(f"  Lock: {custom_obj.lock}")
print(f"  Weights shape: {custom_obj.weights.shape}")
print(f"  Nested locks: {[item['lock'] for item in custom_obj.nested_objects]}")

# Test scrubbing
scrubbed_obj = _scrub(custom_obj)
print("\nScrubbed object:")
print(f"  Name: {scrubbed_obj.name}")
print(f"  Lock: {scrubbed_obj.lock}")
print(f"  Weights shape: {scrubbed_obj.weights.shape}")
print(f"  Nested locks: {[item['lock'] for item in scrubbed_obj.nested_objects]}")

# Test JSON conversion
json_safe_obj = _json_safe(custom_obj)
print("\nJSON-safe object:")
print(f"  Type: {json_safe_obj['object_type']}")
print(f"  Name: {json_safe_obj['name']}")
print(f"  Weights: {type(json_safe_obj['weights'])}")
print(f"  Lock: {json_safe_obj['lock']}")

# Verify JSON serializability
import json
try:
    json_string = json.dumps(json_safe_obj, indent=2)
    print("✓ Custom object successfully converted to JSON")
except Exception as e:
    print(f"✗ JSON conversion failed: {e}")
```

## Advanced Usage Patterns

### Memory-Safe Deep Copy with Scrubbing

```python
from modules.engine.utils import _scrub
import threading
import copy
import numpy as np

def memory_safe_deepcopy(obj, scrub_locks=True):
    """Create a deep copy that's memory safe and optionally scrubs locks"""
    try:
        # First attempt a regular deep copy
        copied = copy.deepcopy(obj)
        
        # If requested, scrub locks from the copy
        if scrub_locks:
            copied = _scrub(copied)
        
        return copied
        
    except Exception as e:
        print(f"Deep copy failed: {e}")
        # Fallback to scrubbing first, then copying
        try:
            scrubbed = _scrub(obj)
            return copy.deepcopy(scrubbed)
        except Exception as e2:
            print(f"Fallback copy failed: {e2}")
            return _scrub(obj)  # Return scrubbed version without deep copy

# Example usage
class ComplexModel:
    def __init__(self):
        self.data = np.random.randn(1000, 100)  # Large array
        self.lock = threading.Lock()
        self.nested = {
            'sub_data': np.random.randn(500, 50),
            'sub_lock': threading.RLock(),
            'params': {'lr': 0.01, 'momentum': 0.9}
        }

model = ComplexModel()
safe_copy = memory_safe_deepcopy(model, scrub_locks=True)

print("Original model lock:", model.lock)
print("Copy model lock:", safe_copy.lock)
print("Data shapes match:", model.data.shape == safe_copy.data.shape)
print("Nested params match:", model.nested['params'] == safe_copy.nested['params'])
```

### Batch Serialization Utilities

```python
from modules.engine.utils import _scrub, _json_safe, _patch_pickle_for_locks
import pickle
import json
import threading
from typing import List, Dict, Any
from pathlib import Path

class BatchSerializationManager:
    """Manager for batch serialization operations"""
    
    def __init__(self, output_dir: str = "serialized_objects"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        _patch_pickle_for_locks()
    
    def batch_pickle_objects(self, objects: Dict[str, Any]) -> Dict[str, bool]:
        """Batch pickle multiple objects safely"""
        results = {}
        
        for name, obj in objects.items():
            try:
                # Scrub and pickle
                clean_obj = _scrub(obj)
                filepath = self.output_dir / f"{name}.pkl"
                
                with open(filepath, 'wb') as f:
                    pickle.dump(clean_obj, f)
                
                results[name] = True
                print(f"✓ {name} pickled successfully")
                
            except Exception as e:
                results[name] = False
                print(f"✗ {name} failed: {e}")
        
        return results
    
    def batch_json_objects(self, objects: Dict[str, Any]) -> Dict[str, bool]:
        """Batch convert multiple objects to JSON safely"""
        results = {}
        
        for name, obj in objects.items():
            try:
                # Convert to JSON-safe and save
                json_obj = _json_safe(obj)
                filepath = self.output_dir / f"{name}.json"
                
                with open(filepath, 'w') as f:
                    json.dump(json_obj, f, indent=2)
                
                results[name] = True
                print(f"✓ {name} converted to JSON successfully")
                
            except Exception as e:
                results[name] = False
                print(f"✗ {name} failed: {e}")
        
        return results
    
    def serialize_training_state(self, models: Dict[str, Any], 
                               metrics: Dict[str, Any],
                               configs: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
        """Serialize complete training state"""
        all_results = {}
        
        # Serialize models (usually pickled due to complex objects)
        print("Serializing models...")
        all_results['models'] = self.batch_pickle_objects(models)
        
        # Serialize metrics (usually JSON for readability)
        print("Serializing metrics...")
        all_results['metrics'] = self.batch_json_objects(metrics)
        
        # Serialize configs (JSON for human readability)
        print("Serializing configs...")
        all_results['configs'] = self.batch_json_objects(configs)
        
        return all_results

# Example: Serialize training session with multiple models
training_models = {
    'linear_model': {
        'weights': np.random.randn(10, 1),
        'bias': np.random.randn(1),
        'lock': threading.Lock(),
        'optimizer_state': {'momentum': np.random.randn(10, 1)}
    },
    'neural_net': {
        'layers': [
            {'weights': np.random.randn(10, 5), 'bias': np.random.randn(5)},
            {'weights': np.random.randn(5, 1), 'bias': np.random.randn(1)}
        ],
        'lock': threading.Lock(),
        'training_queue': None  # Would be a queue object
    }
}

training_metrics = {
    'linear_model_metrics': {
        'train_loss': np.array([0.5, 0.3, 0.2, 0.1]),
        'val_loss': np.array([0.6, 0.4, 0.3, 0.2]),
        'epochs': np.int32(4)
    },
    'neural_net_metrics': {
        'train_accuracy': np.array([0.7, 0.8, 0.85, 0.9]),
        'val_accuracy': np.array([0.65, 0.75, 0.8, 0.85]),
        'learning_rates': np.array([0.01, 0.008, 0.006, 0.004])
    }
}

training_configs = {
    'experiment_config': {
        'batch_size': 32,
        'learning_rate': 0.01,
        'epochs': 4,
        'model_types': ['linear', 'neural_net']
    },
    'data_config': {
        'train_size': 10000,
        'val_size': 2000,
        'feature_count': 10,
        'normalization': 'standard'
    }
}

# Serialize everything
manager = BatchSerializationManager("training_session_001")
results = manager.serialize_training_state(
    training_models, 
    training_metrics, 
    training_configs
)

print("\nSerialization Summary:")
for category, category_results in results.items():
    successful = sum(category_results.values())
    total = len(category_results)
    print(f"  {category}: {successful}/{total} successful")
```

## Best Practices

### 1. Always Apply Pickle Patches

```python
from modules.engine.utils import _patch_pickle_for_locks

# Apply patches early in your application
_patch_pickle_for_locks()

# Then use normal pickle operations safely
import pickle
```

### 2. Scrub Before Serialization

```python
from modules.engine.utils import _scrub

# Always scrub complex objects before serialization
def safe_serialize(obj, filepath):
    clean_obj = _scrub(obj)
    with open(filepath, 'wb') as f:
        pickle.dump(clean_obj, f)
```

### 3. Use JSON-Safe for API Responses

```python
from modules.engine.utils import _json_safe

# Convert ML results to JSON-safe format for APIs
def create_api_response(ml_results):
    return {
        'status': 'success',
        'data': _json_safe(ml_results),
        'timestamp': time.time()
    }
```

### 4. Custom Object Registration

```python
from modules.engine.utils import _json_safe

# Register custom conversions for your objects
@_json_safe.register(YourCustomClass)
def _(obj):
    return {
        'type': 'YourCustomClass',
        'data': obj.serialize()  # Your custom serialization
    }
```

## Related Documentation

- [Experiment Tracker Documentation](experiment_tracker.md)
- [Memory Pool Documentation](memory_pool.md)
- [JIT Compiler Documentation](jit_compiler.md)
- [Training Engine Documentation](train_engine.md)
