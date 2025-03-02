# LRUTTLCache Documentation

## Overview

The `LRUTTLCache` class implements a thread-safe Least Recently Used (LRU) cache with Time-to-Live (TTL) expiration. This cache combines the LRU eviction policy with time-based expiration to manage cached items efficiently.

## Class: `LRUTTLCache`

### Initialization

```python
def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
```

- **max_size**: Maximum number of items the cache can hold. Default is `1000`.
- **ttl_seconds**: Time-to-live duration in seconds for each cached item. Default is `300` seconds (5 minutes).

### Methods

#### `get(key: str) -> Tuple[bool, Any]`

Retrieves an item from the cache.

- **key**: The key of the item to retrieve.
- **Returns**: A tuple `(hit_status, value)` where `hit_status` is `True` if the item was found and not expired, and `False` otherwise. `value` is the cached item if found, otherwise `None`.

#### `set(key: str, value: Any) -> None`

Adds or updates an item in the cache.

- **key**: The key of the item to add or update.
- **value**: The value to cache.

#### `_cleanup_expired() -> None`

Internal method to remove expired entries from the cache.

#### `invalidate(key: str) -> None`

Removes a specific key from the cache.

- **key**: The key of the item to remove.

#### `clear() -> None`

Clears the entire cache, removing all items.

#### `get_stats() -> Dict[str, Any]`

Retrieves cache statistics.

- **Returns**: A dictionary containing the following statistics:
  - `size`: Current number of items in the cache.
  - `max_size`: Maximum number of items the cache can hold.
  - `ttl_seconds`: Time-to-live duration for cached items.
  - `hits`: Number of cache hits.
  - `misses`: Number of cache misses.
  - `hit_rate_percent`: Cache hit rate as a percentage.

### Attributes

- **max_size**: Maximum number of items the cache can hold.
- **ttl_seconds**: Time-to-live duration for cached items.
- **cache**: An `OrderedDict` that stores the cached items.
- **timestamps**: A dictionary that stores the timestamps of when items were added or updated.
- **lock**: A reentrant lock (`threading.RLock`) to ensure thread safety.
- **hits**: Counter for cache hits.
- **misses**: Counter for cache misses.

## Example Usage

```python
cache = LRUTTLCache(max_size=100, ttl_seconds=60)

# Add an item to the cache
cache.set("key1", "value1")

# Retrieve an item from the cache
hit, value = cache.get("key1")
if hit:
    print(f"Cache hit: {value}")
else:
    print("Cache miss")

# Invalidate a specific key
cache.invalidate("key1")

# Clear the entire cache
cache.clear()

# Get cache statistics
stats = cache.get_stats()
print(stats)
```

## Notes

- The cache is thread-safe, ensuring that concurrent access to the cache is handled correctly.
- Expired items are automatically removed from the cache when accessed or during cleanup.
- The cache uses an LRU eviction policy to remove the least recently used items when the cache reaches its maximum size.