# Multi-Level Cache (`modules/engine/multi_level_cache.py`)

## Overview

The Multi-Level Cache module provides a sophisticated caching system with multiple cache levels, access pattern learning, and predictive cache warming capabilities. It's designed to optimize data access patterns in machine learning workloads through intelligent cache hierarchies and automated optimization.

## Features

- **Multi-Level Hierarchy**: L1 (hot), L2 (warm), and L3 (cold) cache levels
- **Specialized Partitions**: Dedicated caches for features, models, and results
- **Access Pattern Learning**: Tracks usage patterns for predictive optimization
- **Automatic Promotion**: Frequently accessed data moves to faster cache levels
- **Predictive Warming**: Background cache warming based on access patterns
- **Comprehensive Statistics**: Detailed performance metrics across all levels

## Core Classes

### MultiLevelCache

Main cache system with intelligent hierarchical storage:

```python
class MultiLevelCache:
    def __init__(
        self,
        l1_size: int = 100,          # Hot data cache size
        l2_size: int = 500,          # Warm data cache size
        l3_size: int = 1000,         # Cold data cache size
        enable_prediction: bool = True
    )
```

**Cache Hierarchy:**
- **L1 Cache**: Hot data (frequently accessed, 5-minute TTL)
- **L2 Cache**: Warm data (moderately accessed, 15-minute TTL)
- **L3 Cache**: Cold data (rarely accessed, 30-minute TTL)

**Specialized Partitions:**
- **Feature Cache**: Input features and preprocessed data
- **Model Cache**: Trained models and model artifacts
- **Result Cache**: Inference results and computed outputs

## Usage Examples

### Basic Multi-Level Caching

```python
from modules.engine.multi_level_cache import MultiLevelCache
import numpy as np
import time
import pickle

# Initialize multi-level cache
cache = MultiLevelCache(
    l1_size=50,      # 50 items in hot cache
    l2_size=200,     # 200 items in warm cache
    l3_size=500,     # 500 items in cold cache
    enable_prediction=True
)

# Example data types
def create_sample_data():
    """Create sample data for caching demonstration"""
    return {
        'features': np.random.rand(1000, 100),
        'model_weights': np.random.rand(100, 10),
        'predictions': np.random.rand(1000, 10),
        'metadata': {'version': '1.0', 'timestamp': time.time()}
    }

# Store data with different priorities and types
sample_data = create_sample_data()

# High-priority result data (goes to L1)
cache.set(
    key='recent_predictions_batch_001',
    value=sample_data['predictions'],
    data_type='result',
    priority='high'
)

# Medium-priority feature data (goes to L2)
cache.set(
    key='preprocessed_features_dataset_A',
    value=sample_data['features'],
    data_type='features',
    priority='medium'
)

# Normal priority model data (goes to L3, but also model cache)
cache.set(
    key='trained_model_v1.2',
    value=sample_data['model_weights'],
    data_type='model',
    priority='normal'
)

# Retrieve data and observe cache behavior
test_keys = [
    'recent_predictions_batch_001',
    'preprocessed_features_dataset_A', 
    'trained_model_v1.2',
    'nonexistent_key'
]

print("Cache Access Test:")
for key in test_keys:
    hit, value = cache.get(key)
    if hit:
        print(f"✓ Cache HIT for '{key}': {type(value)} shape {getattr(value, 'shape', 'N/A')}")
    else:
        print(f"✗ Cache MISS for '{key}'")

# Get initial statistics
stats = cache.get_comprehensive_stats()
print(f"\nInitial Cache Statistics:")
print(f"  L1 hits: {stats['l1_hits']}, L1 misses: {stats['l1_misses']}")
print(f"  L2 hits: {stats['l2_hits']}, L2 misses: {stats['l2_misses']}")
print(f"  L3 hits: {stats['l3_hits']}, L3 misses: {stats['l3_misses']}")
```

### Cache Promotion and Access Pattern Learning

```python
from modules.engine.multi_level_cache import MultiLevelCache
import time
import random

def demonstrate_cache_promotion():
    """Demonstrate automatic cache promotion based on access patterns"""
    
    cache = MultiLevelCache(
        l1_size=10,
        l2_size=30, 
        l3_size=100,
        enable_prediction=True
    )
    
    # Store test data in L3 (cold storage)
    test_data = {}
    for i in range(20):
        key = f'data_item_{i}'
        value = {'id': i, 'data': list(range(i * 10, (i + 1) * 10))}
        test_data[key] = value
        cache.set(key, value, priority='normal')  # Goes to L3
    
    print("Initial storage: All items in L3 cache")
    
    # Simulate access patterns
    # Some items accessed frequently (should promote to L1)
    hot_keys = ['data_item_5', 'data_item_10', 'data_item_15']
    # Some items accessed moderately (should promote to L2)  
    warm_keys = ['data_item_3', 'data_item_8', 'data_item_12', 'data_item_18']
    # Some items rarely accessed (stay in L3)
    cold_keys = ['data_item_0', 'data_item_1', 'data_item_2']
    
    print("\nSimulating access patterns...")
    
    # Phase 1: Create access patterns
    for round_num in range(5):
        print(f"\nAccess round {round_num + 1}:")
        
        # Frequently access hot keys
        for key in hot_keys:
            for _ in range(3):  # Access multiple times
                hit, value = cache.get(key)
                print(f"  Hot access: {key} -> {'HIT' if hit else 'MISS'}")
                time.sleep(0.01)  # Small delay
        
        # Moderately access warm keys
        for key in warm_keys:
            hit, value = cache.get(key)
            print(f"  Warm access: {key} -> {'HIT' if hit else 'MISS'}")
            time.sleep(0.01)
        
        # Rarely access cold keys
        if round_num % 3 == 0:  # Only access every 3rd round
            for key in random.sample(cold_keys, 1):
                hit, value = cache.get(key)
                print(f"  Cold access: {key} -> {'HIT' if hit else 'MISS'}")
    
    # Show final statistics
    final_stats = cache.get_comprehensive_stats()
    print(f"\nFinal Cache Statistics:")
    print(f"  L1: {final_stats['l1_hits']} hits, {final_stats['l1_misses']} misses")
    print(f"  L2: {final_stats['l2_hits']} hits, {final_stats['l2_misses']} misses")
    print(f"  L3: {final_stats['l3_hits']} hits, {final_stats['l3_misses']} misses")
    print(f"  Total requests: {final_stats['total_requests']}")
    
    # Calculate hit rates
    total_hits = final_stats['l1_hits'] + final_stats['l2_hits'] + final_stats['l3_hits']
    total_requests = final_stats['total_requests']
    hit_rate = (total_hits / total_requests) * 100 if total_requests > 0 else 0
    
    print(f"  Overall hit rate: {hit_rate:.1f}%")
    
    # Test that hot data is now in L1
    print(f"\nVerifying promotion - Testing hot keys in L1:")
    for key in hot_keys:
        hit, value = cache.get(key)
        # If it's a hit and we get it fast, it's likely in L1
        print(f"  {key}: {'✓ Likely in L1' if hit else '✗ Not found'}")

# Run cache promotion demonstration
demonstrate_cache_promotion()
```

### Specialized Cache Partitions

```python
from modules.engine.multi_level_cache import MultiLevelCache
import numpy as np
import pickle
import time

class MLWorkflowCache:
    """Example ML workflow using specialized cache partitions"""
    
    def __init__(self):
        self.cache = MultiLevelCache(
            l1_size=20,   # Small L1 for immediate results
            l2_size=100,  # Medium L2 for features and intermediate data
            l3_size=500,  # Large L3 for models and historical data
            enable_prediction=True
        )
    
    def cache_feature_data(self, dataset_id: str, features: np.ndarray):
        """Cache preprocessed feature data"""
        feature_key = f'features_{dataset_id}'
        
        # Store in specialized feature cache
        self.cache.set(
            key=feature_key,
            value=features,
            data_type='features',
            priority='medium'
        )
        
        # Also cache metadata
        metadata_key = f'features_meta_{dataset_id}'
        metadata = {
            'shape': features.shape,
            'dtype': str(features.dtype),
            'cached_at': time.time(),
            'dataset_id': dataset_id
        }
        
        self.cache.set(
            key=metadata_key,
            value=metadata,
            data_type='general',
            priority='medium'
        )
        
        print(f"Cached features for {dataset_id}: shape {features.shape}")
    
    def cache_model(self, model_id: str, model_data: dict):
        """Cache trained model data"""
        model_key = f'model_{model_id}'
        
        # Store in specialized model cache (high priority for trained models)
        self.cache.set(
            key=model_key,
            value=model_data,
            data_type='model',
            priority='high'
        )
        
        print(f"Cached model {model_id}")
    
    def cache_predictions(self, prediction_id: str, predictions: np.ndarray, confidence: np.ndarray = None):
        """Cache model predictions"""
        pred_key = f'predictions_{prediction_id}'
        
        prediction_data = {
            'predictions': predictions,
            'confidence': confidence,
            'generated_at': time.time(),
            'prediction_id': prediction_id
        }
        
        # Store in specialized result cache (high priority for recent results)
        self.cache.set(
            key=pred_key,
            value=prediction_data,
            data_type='result',
            priority='high'
        )
        
        print(f"Cached predictions {prediction_id}: {predictions.shape}")
    
    def get_features(self, dataset_id: str) -> tuple:
        """Retrieve cached feature data"""
        feature_key = f'features_{dataset_id}'
        metadata_key = f'features_meta_{dataset_id}'
        
        # Try to get features
        hit, features = self.cache.get(feature_key, data_type='features')
        meta_hit, metadata = self.cache.get(metadata_key)
        
        if hit:
            print(f"✓ Retrieved cached features for {dataset_id}")
            return features, metadata if meta_hit else None
        else:
            print(f"✗ Features not found for {dataset_id}")
            return None, None
    
    def get_model(self, model_id: str):
        """Retrieve cached model"""
        model_key = f'model_{model_id}'
        
        hit, model_data = self.cache.get(model_key, data_type='model')
        
        if hit:
            print(f"✓ Retrieved cached model {model_id}")
            return model_data
        else:
            print(f"✗ Model not found {model_id}")
            return None
    
    def get_predictions(self, prediction_id: str):
        """Retrieve cached predictions"""
        pred_key = f'predictions_{prediction_id}'
        
        hit, prediction_data = self.cache.get(pred_key, data_type='result')
        
        if hit:
            age_seconds = time.time() - prediction_data['generated_at']
            print(f"✓ Retrieved cached predictions {prediction_id} (age: {age_seconds:.1f}s)")
            return prediction_data
        else:
            print(f"✗ Predictions not found {prediction_id}")
            return None
    
    def get_cache_statistics(self):
        """Get comprehensive cache statistics"""
        return self.cache.get_comprehensive_stats()

# Demonstrate specialized cache usage
def ml_workflow_demo():
    """Demonstrate ML workflow with specialized caching"""
    
    workflow = MLWorkflowCache()
    
    # Step 1: Cache feature data for different datasets
    print("=== Caching Feature Data ===")
    datasets = ['train_set_A', 'val_set_A', 'test_set_A', 'train_set_B']
    
    for dataset_id in datasets:
        features = np.random.randn(1000, 50)  # Random feature data
        workflow.cache_feature_data(dataset_id, features)
    
    # Step 2: Cache trained models
    print("\n=== Caching Models ===")
    models = ['linear_model_v1', 'neural_net_v2', 'ensemble_v1']
    
    for model_id in models:
        model_data = {
            'weights': np.random.randn(50, 10),
            'bias': np.random.randn(10),
            'hyperparameters': {'lr': 0.01, 'epochs': 100},
            'performance': {'accuracy': 0.95, 'loss': 0.05}
        }
        workflow.cache_model(model_id, model_data)
    
    # Step 3: Cache prediction results
    print("\n=== Caching Predictions ===")
    predictions = ['pred_batch_001', 'pred_batch_002', 'pred_realtime_001']
    
    for pred_id in predictions:
        preds = np.random.rand(100, 10)  # Random predictions
        confidence = np.random.rand(100)  # Confidence scores
        workflow.cache_predictions(pred_id, preds, confidence)
    
    # Step 4: Simulate retrieval operations
    print("\n=== Retrieving Cached Data ===")
    
    # Retrieve features
    features, metadata = workflow.get_features('train_set_A')
    if features is not None:
        print(f"  Features shape: {features.shape}")
        if metadata:
            print(f"  Metadata: {metadata}")
    
    # Retrieve model
    model = workflow.get_model('neural_net_v2')
    if model:
        print(f"  Model performance: {model['performance']}")
    
    # Retrieve predictions
    pred_data = workflow.get_predictions('pred_batch_001')
    if pred_data:
        print(f"  Predictions shape: {pred_data['predictions'].shape}")
    
    # Step 5: Show cache statistics
    print("\n=== Cache Statistics ===")
    stats = workflow.get_cache_statistics()
    
    print(f"Cache Performance:")
    print(f"  L1 Cache: {stats['l1_hits']} hits, {stats['l1_misses']} misses")
    print(f"  L2 Cache: {stats['l2_hits']} hits, {stats['l2_misses']} misses")
    print(f"  L3 Cache: {stats['l3_hits']} hits, {stats['l3_misses']} misses")
    
    print(f"Specialized Caches:")
    print(f"  Feature Cache: {stats['l2_stats']}")
    print(f"  Model Cache: {stats['l3_stats']}")
    print(f"  Result Cache: {stats['l1_stats']}")

# Run ML workflow demonstration
ml_workflow_demo()
```

### Performance Optimization and Monitoring

```python
from modules.engine.multi_level_cache import MultiLevelCache
import time
import threading
import random
import numpy as np
from typing import Dict, List

class CachePerformanceMonitor:
    """Monitor and optimize cache performance"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.performance_history = []
        self.access_time_history = []
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start performance monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        print("Cache performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Cache performance monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                stats = self.cache.get_comprehensive_stats()
                
                # Calculate current hit rates
                total_hits = stats['l1_hits'] + stats['l2_hits'] + stats['l3_hits']
                total_requests = stats['total_requests']
                
                if total_requests > 0:
                    hit_rate = (total_hits / total_requests) * 100
                    
                    performance_snapshot = {
                        'timestamp': time.time(),
                        'hit_rate': hit_rate,
                        'l1_hit_rate': (stats['l1_hits'] / total_requests) * 100,
                        'l2_hit_rate': (stats['l2_hits'] / total_requests) * 100,
                        'l3_hit_rate': (stats['l3_hits'] / total_requests) * 100,
                        'total_requests': total_requests
                    }
                    
                    self.performance_history.append(performance_snapshot)
                    
                    # Keep only recent history
                    if len(self.performance_history) > 100:
                        self.performance_history = self.performance_history[-50:]
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def measure_access_time(self, key: str, data_type: str = 'general') -> float:
        """Measure cache access time"""
        start_time = time.perf_counter()
        hit, value = self.cache.get(key, data_type)
        end_time = time.perf_counter()
        
        access_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        self.access_time_history.append({
            'key': key,
            'hit': hit,
            'access_time_ms': access_time,
            'timestamp': time.time()
        })
        
        return access_time
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.performance_history:
            return {'error': 'No performance data available'}
        
        recent_performance = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        # Calculate averages
        avg_hit_rate = sum(p['hit_rate'] for p in recent_performance) / len(recent_performance)
        avg_l1_rate = sum(p['l1_hit_rate'] for p in recent_performance) / len(recent_performance)
        avg_l2_rate = sum(p['l2_hit_rate'] for p in recent_performance) / len(recent_performance)
        avg_l3_rate = sum(p['l3_hit_rate'] for p in recent_performance) / len(recent_performance)
        
        # Access time analysis
        access_times = [a['access_time_ms'] for a in self.access_time_history[-100:]]
        hit_times = [a['access_time_ms'] for a in self.access_time_history[-100:] if a['hit']]
        miss_times = [a['access_time_ms'] for a in self.access_time_history[-100:] if not a['hit']]
        
        report = {
            'performance_summary': {
                'average_hit_rate': avg_hit_rate,
                'l1_hit_rate': avg_l1_rate,
                'l2_hit_rate': avg_l2_rate,
                'l3_hit_rate': avg_l3_rate,
                'monitoring_duration': len(self.performance_history)
            },
            'access_time_analysis': {
                'total_accesses': len(access_times),
                'average_access_time_ms': sum(access_times) / len(access_times) if access_times else 0,
                'hit_access_time_ms': sum(hit_times) / len(hit_times) if hit_times else 0,
                'miss_access_time_ms': sum(miss_times) / len(miss_times) if miss_times else 0,
                'min_access_time_ms': min(access_times) if access_times else 0,
                'max_access_time_ms': max(access_times) if access_times else 0
            }
        }
        
        return report
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest cache optimizations based on performance data"""
        suggestions = []
        
        if not self.performance_history:
            return ['No performance data available for optimization suggestions']
        
        recent_stats = self.performance_history[-5:] if len(self.performance_history) >= 5 else self.performance_history
        avg_hit_rate = sum(p['hit_rate'] for p in recent_stats) / len(recent_stats)
        avg_l1_rate = sum(p['l1_hit_rate'] for p in recent_stats) / len(recent_stats)
        
        # Analyze hit rates and suggest improvements
        if avg_hit_rate < 70:
            suggestions.append("Overall hit rate is low (<70%). Consider increasing cache sizes or improving cache key strategies.")
        
        if avg_l1_rate < 20:
            suggestions.append("L1 hit rate is low (<20%). Consider increasing L1 cache size for hot data.")
        
        if avg_l1_rate > 80:
            suggestions.append("L1 hit rate is very high (>80%). L1 cache might be oversized relative to L2/L3.")
        
        # Analyze access times
        if self.access_time_history:
            recent_access_times = [a['access_time_ms'] for a in self.access_time_history[-50:]]
            avg_access_time = sum(recent_access_times) / len(recent_access_times)
            
            if avg_access_time > 1.0:  # More than 1ms average
                suggestions.append(f"Average access time is high ({avg_access_time:.2f}ms). Consider optimizing cache lookup algorithms.")
        
        if not suggestions:
            suggestions.append("Cache performance looks good! No specific optimizations suggested.")
        
        return suggestions

# Comprehensive performance testing
def performance_testing_demo():
    """Comprehensive cache performance testing"""
    
    # Create cache with moderate sizes for testing
    cache = MultiLevelCache(
        l1_size=50,
        l2_size=150,
        l3_size=300,
        enable_prediction=True
    )
    
    # Create performance monitor
    monitor = CachePerformanceMonitor(cache)
    monitor.start_monitoring(interval_seconds=0.5)
    
    try:
        # Phase 1: Populate cache with test data
        print("=== Phase 1: Populating Cache ===")
        test_data = {}
        
        for i in range(500):
            key = f'test_key_{i}'
            value = np.random.randn(100, 10)  # Some test data
            test_data[key] = value
            
            # Vary priority to test different cache levels
            if i < 50:
                priority = 'high'
            elif i < 200:
                priority = 'medium'
            else:
                priority = 'normal'
            
            cache.set(key, value, priority=priority)
        
        print(f"Populated cache with {len(test_data)} items")
        
        # Phase 2: Simulate realistic access patterns
        print("\n=== Phase 2: Simulating Access Patterns ===")
        
        # Hot data (frequently accessed)
        hot_keys = [f'test_key_{i}' for i in range(20)]
        # Warm data (moderately accessed)
        warm_keys = [f'test_key_{i}' for i in range(20, 100)]
        # Cold data (rarely accessed)
        cold_keys = [f'test_key_{i}' for i in range(100, 500)]
        
        # Simulate 1000 cache accesses with realistic patterns
        for round_num in range(10):
            print(f"  Access round {round_num + 1}/10")
            
            for _ in range(100):
                # 60% hot data access, 30% warm data, 10% cold data
                rand = random.random()
                
                if rand < 0.6:  # Hot data
                    key = random.choice(hot_keys)
                elif rand < 0.9:  # Warm data
                    key = random.choice(warm_keys)
                else:  # Cold data
                    key = random.choice(cold_keys)
                
                # Measure access time
                access_time = monitor.measure_access_time(key)
                
                # Occasional miss (access non-existent key)
                if random.random() < 0.05:  # 5% miss rate
                    monitor.measure_access_time(f'nonexistent_key_{random.randint(1000, 2000)}')
            
            time.sleep(0.1)  # Brief pause between rounds
        
        # Phase 3: Performance analysis
        print("\n=== Phase 3: Performance Analysis ===")
        
        time.sleep(2)  # Let monitoring collect final data
        
        # Get performance report
        report = monitor.get_performance_report()
        
        print("Performance Report:")
        print(f"  Average hit rate: {report['performance_summary']['average_hit_rate']:.1f}%")
        print(f"  L1 hit rate: {report['performance_summary']['l1_hit_rate']:.1f}%")
        print(f"  L2 hit rate: {report['performance_summary']['l2_hit_rate']:.1f}%")
        print(f"  L3 hit rate: {report['performance_summary']['l3_hit_rate']:.1f}%")
        
        print(f"\nAccess Time Analysis:")
        print(f"  Total accesses: {report['access_time_analysis']['total_accesses']}")
        print(f"  Average access time: {report['access_time_analysis']['average_access_time_ms']:.3f}ms")
        print(f"  Hit access time: {report['access_time_analysis']['hit_access_time_ms']:.3f}ms")
        print(f"  Miss access time: {report['access_time_analysis']['miss_access_time_ms']:.3f}ms")
        
        # Get optimization suggestions
        suggestions = monitor.suggest_optimizations()
        print(f"\nOptimization Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
        
    finally:
        monitor.stop_monitoring()
        cache.shutdown()

# Run performance testing
performance_testing_demo()
```

## Best Practices

### 1. Cache Size Configuration

```python
# Configure cache sizes based on memory constraints and access patterns
total_memory_mb = 1000  # Available memory
cache = MultiLevelCache(
    l1_size=int(total_memory_mb * 0.1),    # 10% for hot data
    l2_size=int(total_memory_mb * 0.3),    # 30% for warm data
    l3_size=int(total_memory_mb * 0.6),    # 60% for cold data
)
```

### 2. Key Design

```python
# Use hierarchical, descriptive keys
def create_cache_key(data_type: str, identifier: str, version: str = "v1") -> str:
    return f"{data_type}:{identifier}:{version}"

# Examples:
feature_key = create_cache_key("features", "dataset_A", "v2")
model_key = create_cache_key("model", "neural_net", "v1.5")
```

### 3. Data Type Specification

```python
# Always specify data type for optimal cache placement
cache.set(key, feature_data, data_type='features', priority='medium')
cache.set(key, model_data, data_type='model', priority='high')
cache.set(key, result_data, data_type='result', priority='high')
```

### 4. Resource Cleanup

```python
try:
    # Use cache
    pass
finally:
    cache.shutdown()  # Always cleanup resources
```

## Related Documentation

- [LRU TTL Cache Documentation](lru_ttl_cache.md)
- [Memory Pool Documentation](memory_pool.md)
- [Performance Metrics Documentation](performance_metrics.md)
- [Dynamic Batcher Documentation](dynamic_batcher.md)
