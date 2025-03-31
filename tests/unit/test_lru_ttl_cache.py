"""
Tests for the LRUTTLCache module.

This file demonstrates the conversion from unittest to pytest.
"""
import pytest
import time
from modules.engine.lru_ttl_cache import LRUTTLCache  # Adjust the import path as necessary

# Define a fixture for creating a cache with short TTL for testing
@pytest.fixture
def cache():
    """Create a cache with a small TTL for testing expiration easily."""
    ttl = 1  # seconds
    return LRUTTLCache(max_size=3, ttl_seconds=ttl)

def test_set_and_get(cache):
    """Test basic set and get functionality."""
    cache.set("key1", "value1")
    hit, value = cache.get("key1")
    assert hit
    assert value == "value1"

def test_get_miss(cache):
    """Test that getting a key that doesn't exist results in a miss."""
    hit, value = cache.get("non_existent")
    assert not hit
    assert value is None

def test_expiration(cache):
    """Test that items expire after TTL."""
    # Get the TTL from the cache object for tests
    ttl = cache._ttl_seconds
    
    # Set a key
    cache.set("key_expire", "value_expire")
    
    # Immediately available
    hit, _ = cache.get("key_expire")
    assert hit
    
    # Wait for TTL to expire
    time.sleep(ttl + 0.1)
    hit, value = cache.get("key_expire")
    assert not hit
    assert value is None

def test_invalidate(cache):
    """Test that invalidate removes a key."""
    cache.set("key_inv", "value_inv")
    
    # Confirm it's present
    hit, value = cache.get("key_inv")
    assert hit
    assert value == "value_inv"
    
    # Invalidate and check
    cache.invalidate("key_inv")
    hit, value = cache.get("key_inv")
    assert not hit
    assert value is None

def test_clear(cache):
    """Test that clear removes all items."""
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.clear()
    assert cache.get_stats()["size"] == 0
    
    # All keys should be missing
    hit, _ = cache.get("key1")
    assert not hit
    hit, _ = cache.get("key2")
    assert not hit

def test_max_size_eviction(cache):
    """Test that the cache evicts the least-recently used item once max_size is exceeded."""
    # Set three items (full capacity)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    
    # Access 'a' to mark it as recently used
    hit, value = cache.get("a")
    assert hit
    assert value == 1
    
    # Now add a new item; expect eviction of 'b' (the least recently used)
    cache.set("d", 4)
    
    # 'b' should have been evicted
    hit, _ = cache.get("b")
    assert not hit
    
    # 'a', 'c', and 'd' should be present
    for key, expected in [("a", 1), ("c", 3), ("d", 4)]:
        hit, value = cache.get(key)
        assert hit
        assert value == expected

def test_stats(cache):
    """Test that cache statistics are correctly updated."""
    # Initially, stats should show zero hits and misses
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    
    # Set a key and get it
    cache.set("stat_key", "stat_value")
    hit, _ = cache.get("stat_key")
    assert hit
    
    # Get a missing key
    hit, _ = cache.get("no_key")
    assert not hit
    
    # Check that stats reflect one hit and one miss
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    
    # Now let a key expire and try to get it
    ttl = cache._ttl_seconds
    cache.set("expire_key", "value")
    time.sleep(ttl + 0.1)
    hit, _ = cache.get("expire_key")
    assert not hit
    
    # Stats should increase the miss counter
    stats = cache.get_stats()
    assert stats["misses"] == 2

# Example of a parametrized test (new pytest feature not in original)
@pytest.mark.parametrize(
    "key,value", 
    [
        ("simple_key", "simple_value"),
        ("unicode_key_€£¥", "unicode_value_€£¥"),
        ("empty_value", ""),
        (123, "numeric_key"),  # Non-string key
    ]
)
def test_various_key_types(cache, key, value):
    """Test that the cache can handle various key types."""
    cache.set(key, value)
    hit, retrieved = cache.get(key)
    assert hit
    assert retrieved == value