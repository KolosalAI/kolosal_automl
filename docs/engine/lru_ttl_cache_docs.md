# Module: `LRUTTLCache`

## Overview
A thread-safe implementation of a Least Recently Used (LRU) cache with Time-To-Live (TTL) functionality. This cache combines two eviction policies: LRU-based removal when size limits are reached, and time-based expiration of cached items.

## Prerequisites
- Python ≥3.7
- Standard library modules:
  - `collections` (OrderedDict)
  - `time`
  - `threading`
  - `typing`

## Installation
No additional installation required beyond standard Python libraries.

## Usage
```python
from lru_ttl_cache import LRUTTLCache

# Create a cache with 1000 items max, 5-minute TTL
cache = LRUTTLCache(max_size=1000, ttl_seconds=300)

# Set a value
cache.set("key1", "value1")

# Get a value
hit, value = cache.get("key1")
if hit:
    print(f"Found: {value}")
else:
    print("Cache miss")

# Get or compute value if missing
value = cache.get_or_set("key2", lambda: expensive_computation())

# Check cache stats
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']}%")
```

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_size` | `1000` | Maximum number of items to store in the cache |
| `ttl_seconds` | `300` | Time-to-live for cache items in seconds |
| `cleanup_interval` | `60` | How often to run background cleanup in seconds (0 to disable) |

---

## Classes

### `LRUTTLCache`
```python
class LRUTTLCache:
```
- **Description**:  
  A thread-safe cache implementation that combines Least Recently Used (LRU) eviction policy with time-based expiration. Keys are evicted when they expire based on TTL or when the cache exceeds its maximum size (oldest used items are removed first).

- **Attributes**:  
  - `max_size (int)`: Maximum number of items to store in the cache.
  - `ttl_seconds (int)`: Time-to-live for cache items in seconds.
  - `cache (OrderedDict)`: The underlying ordered dictionary that stores the cached values.
  - `timestamps (dict)`: Dictionary mapping keys to their insertion timestamps.
  - `lock (threading.RLock)`: Reentrant lock for ensuring thread safety.
  - `hits (int)`: Counter for cache hits.
  - `misses (int)`: Counter for cache misses.
  - `cleanup_thread (threading.Thread)`: Background thread for periodic cleanup.
  - `should_stop (threading.Event)`: Event to signal the cleanup thread to stop.

- **Constructor**:
  ```python
  def __init__(self, max_size: int = 1000, ttl_seconds: int = 300, cleanup_interval: int = 60)
  ```
  - **Parameters**:
    - `max_size (int)`: Maximum number of items to store in the cache. Default is 1000.
    - `ttl_seconds (int)`: Time-to-live for cache items in seconds. Default is 300.
    - `cleanup_interval (int)`: How often to run background cleanup in seconds. Default is 60. Set to 0 to disable.
  - **Raises**:  
    - `ValueError`: If max_size or ttl_seconds is not positive.

- **Methods**:

  ### `get(self, key: str) -> Tuple[bool, Any]`
  ```python
  def get(self, key: str) -> Tuple[bool, Any]
  ```
  - **Description**:  
    Retrieves an item from the cache if it exists and is not expired.

  - **Parameters**:  
    - `key (str)`: The cache key to retrieve.

  - **Returns**:  
    - `Tuple[bool, Any]`: A tuple containing:
      - A boolean indicating whether the item was found (hit status).
      - The cached value (or None if not found).

  ### `get_or_set(self, key: str, value_func: callable) -> Any`
  ```python
  def get_or_set(self, key: str, value_func: callable) -> Any
  ```
  - **Description**:  
    Gets an item from the cache or sets it if missing or expired.

  - **Parameters**:  
    - `key (str)`: The cache key to retrieve or set.
    - `value_func (callable)`: Function to call to generate the value if not in cache.

  - **Returns**:  
    - `Any`: The cached or newly generated value.

  ### `set(self, key: str, value: Any) -> None`
  ```python
  def set(self, key: str, value: Any) -> None
  ```
  - **Description**:  
    Adds or updates an item in the cache.

  - **Parameters**:  
    - `key (str)`: The cache key.
    - `value (Any)`: The value to cache.

  - **Returns**:  
    - `None`

  ### `_cleanup_expired(self) -> int`
  ```python
  def _cleanup_expired(self) -> int
  ```
  - **Description**:  
    Internal method that removes expired entries from the cache.

  - **Returns**:  
    - `int`: Number of items removed.

  ### `_background_cleanup(self, interval: int) -> None`
  ```python
  def _background_cleanup(self, interval: int) -> None
  ```
  - **Description**:  
    Internal method that runs periodic cleanup in a background thread.

  - **Parameters**:  
    - `interval (int)`: Time in seconds between cleanup operations.

  - **Returns**:  
    - `None`

  ### `invalidate(self, key: str) -> bool`
  ```python
  def invalidate(self, key: str) -> bool
  ```
  - **Description**:  
    Removes a specific key from the cache.

  - **Parameters**:  
    - `key (str)`: The key to remove.

  - **Returns**:  
    - `bool`: True if the key was in cache and removed, False otherwise.

  ### `clear(self) -> None`
  ```python
  def clear(self) -> None
  ```
  - **Description**:  
    Clears the entire cache.

  - **Returns**:  
    - `None`

  ### `get_stats(self) -> Dict[str, Any]`
  ```python
  def get_stats(self) -> Dict[str, Any]
  ```
  - **Description**:  
    Returns statistics about the cache's performance.

  - **Returns**:  
    - `Dict[str, Any]`: Dictionary containing:
      - `size`: Current number of items in the cache.
      - `max_size`: Maximum capacity of the cache.
      - `ttl_seconds`: Time-to-live setting in seconds.
      - `hits`: Number of cache hits.
      - `misses`: Number of cache misses.
      - `hit_rate_percent`: Cache hit rate as a percentage.

  ### `__len__(self) -> int`
  ```python
  def __len__(self) -> int
  ```
  - **Description**:  
    Returns the number of items in the cache.

  - **Returns**:  
    - `int`: Current number of items in the cache.

  ### `__contains__(self, key: str) -> bool`
  ```python
  def __contains__(self, key: str) -> bool
  ```
  - **Description**:  
    Checks if a key is in the cache and not expired. Enables the `in` operator.

  - **Parameters**:  
    - `key (str)`: The key to check.

  - **Returns**:  
    - `bool`: True if the key exists in the cache and is not expired, False otherwise.

  ### `__del__(self) -> None`
  ```python
  def __del__(self) -> None
  ```
  - **Description**:  
    Cleans up resources when the instance is garbage collected. Signals the cleanup thread to stop.

  - **Returns**:  
    - `None`

## Architecture
The cache uses an `OrderedDict` for maintaining LRU order and a separate dictionary for tracking timestamps. The implementation is made thread-safe with an `RLock` to allow nested access from the same thread.

**Key Design Patterns**:
1. **Decorator Pattern** (functional): The `get_or_set` method enables memoization patterns.
2. **Background Worker**: A daemon thread performs periodic cleanup of expired items.

## Testing
Unit tests for this class should include tests for:
```bash
python -m unittest tests/test_lru_ttl_cache.py
```

- Thread safety with concurrent access
- Correctness of LRU eviction policy
- Proper expiration of items based on TTL
- Performance under different load conditions

## Security & Compliance
- Thread-safe implementation suitable for multi-threaded environments
- No external network calls or file system access
- Memory usage is bounded by the `max_size` parameter

> Last Updated: 2025-05-11