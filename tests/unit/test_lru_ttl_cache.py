"""
Tests for the LRUTTLCache module.

This file uses the pytest framework.
"""
import pytest
import time

try:
    from modules.engine.lru_ttl_cache import LRUTTLCache
except ImportError as e:
    pytest.skip(f"LRU TTL Cache module not available: {e}", allow_module_level=True)


@pytest.mark.unit
class TestLRUTTLCache:
    """Test cases for the LRUTTLCache class."""

    @pytest.fixture(autouse=True)
    def setup_cache(self):
        """Set up a cache with a small TTL for testing expiration easily."""
        self.ttl = 1  # seconds
        self.cache = LRUTTLCache(max_size=3, ttl_seconds=self.ttl)

    def test_set_and_get(self):
        """Test basic set and get functionality."""
        self.cache.set("key1", "value1")
        hit, value = self.cache.get("key1")
        assert hit
        assert value == "value1"

    def test_get_miss(self):
        """Test that getting a key that doesn't exist results in a miss."""
        hit, value = self.cache.get("non_existent")
        assert not hit
        assert value is None

    def test_expiration(self):
        """Test that items expire after TTL."""
        # Set a key
        self.cache.set("key_expire", "value_expire")
        
        # Immediately available
        hit, _ = self.cache.get("key_expire")
        assert hit
        
        # Wait for TTL to expire
        time.sleep(self.ttl + 0.1)
        hit, value = self.cache.get("key_expire")
        assert not hit
        assert value is None

    def test_invalidate(self):
        """Test that invalidate removes a key."""
        self.cache.set("key_inv", "value_inv")
        
        # Confirm it's present
        hit, value = self.cache.get("key_inv")
        self.assertTrue(hit)
        self.assertEqual(value, "value_inv")
        
        # Invalidate and check
        self.cache.invalidate("key_inv")
        hit, value = self.cache.get("key_inv")
        self.assertFalse(hit)
        self.assertIsNone(value)

    def test_clear(self):
        """Test that clear removes all items."""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.clear()
        self.assertEqual(self.cache.get_stats()["size"], 0)
        
        # All keys should be missing
        hit, _ = self.cache.get("key1")
        self.assertFalse(hit)
        hit, _ = self.cache.get("key2")
        self.assertFalse(hit)

    def test_max_size_eviction(self):
        """Test that the cache evicts the least-recently used item once max_size is exceeded."""
        # Set three items (full capacity)
        self.cache.set("a", 1)
        self.cache.set("b", 2)
        self.cache.set("c", 3)
        
        # Access 'a' to mark it as recently used
        hit, value = self.cache.get("a")
        self.assertTrue(hit)
        self.assertEqual(value, 1)
        
        # Now add a new item; expect eviction of 'b' (the least recently used)
        self.cache.set("d", 4)
        
        # 'b' should have been evicted
        hit, _ = self.cache.get("b")
        self.assertFalse(hit)
        
        # 'a', 'c', and 'd' should be present
        for key, expected in [("a", 1), ("c", 3), ("d", 4)]:
            hit, value = self.cache.get(key)
            self.assertTrue(hit)
            self.assertEqual(value, expected)

    def test_stats(self):
        """Test that cache statistics are correctly updated."""
        # Initially, stats should show zero hits and misses
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        
        # Set a key and get it
        self.cache.set("stat_key", "stat_value")
        hit, _ = self.cache.get("stat_key")
        self.assertTrue(hit)
        
        # Get a missing key
        hit, _ = self.cache.get("no_key")
        self.assertFalse(hit)
        
        # Check that stats reflect one hit and one miss
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        
        # Now let a key expire and try to get it
        self.cache.set("expire_key", "value")
        time.sleep(self.ttl + 0.1)
        hit, _ = self.cache.get("expire_key")
        self.assertFalse(hit)
        
        # Stats should increase the miss counter
        stats = self.cache.get_stats()
        self.assertEqual(stats["misses"], 2)

    def test_various_key_types(self):
        """Test that the cache can handle various key types."""
        test_cases = [
            ("simple_key", "simple_value"),
            ("unicode_key_€£¥", "unicode_value_€£¥"),
            ("empty_value", ""),
            (123, "numeric_key"),  # Non-string key
        ]
        
        for key, value in test_cases:
            self.cache.set(key, value)
            hit, retrieved = self.cache.get(key)
            assert hit
            assert retrieved == value


if __name__ == '__main__':
    pytest.main([__file__])