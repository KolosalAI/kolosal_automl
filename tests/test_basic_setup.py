"""
Simple test to verify pytest setup is working correctly.
"""
import pytest
import numpy as np


@pytest.mark.unit
class TestBasicSetup:
    """Basic tests to verify pytest setup."""
    
    def test_imports(self):
        """Test that basic imports work."""
        import sys
        import os
        import numpy as np
        assert True

    def test_numpy_operations(self):
        """Test basic numpy operations."""
        arr = np.array([1, 2, 3, 4, 5])
        assert len(arr) == 5
        assert arr.sum() == 15
        assert arr.mean() == 3.0

    def test_basic_assertions(self):
        """Test basic assertion patterns."""
        # Equality assertions
        assert 2 + 2 == 4
        assert "hello" == "hello"
        
        # Boolean assertions
        assert True
        assert not False
        
        # Membership assertions
        assert "a" in "abc"
        assert 1 in [1, 2, 3]
        
        # Type assertions
        assert isinstance("hello", str)
        assert isinstance(42, int)
        assert isinstance(3.14, float)

    def test_exception_handling(self):
        """Test exception handling."""
        with pytest.raises(ZeroDivisionError):
            1 / 0
        
        with pytest.raises(KeyError):
            d = {}
            d["nonexistent"]

    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
        (0, 0),
        (-1, -2),
    ])
    def test_parametrized(self, input_val, expected):
        """Test parametrized test functionality."""
        def double(x):
            return x * 2
        
        assert double(input_val) == expected

    def test_fixtures(self, sample_data):
        """Test that fixtures work."""
        assert "X" in sample_data
        assert "y" in sample_data
        assert sample_data["n_samples"] > 0
        assert sample_data["n_features"] > 0


@pytest.mark.functional  
def test_example_functional():
    """Example functional test."""
    # This would typically test API endpoints or integration scenarios
    result = {"status": "ok", "data": [1, 2, 3]}
    assert result["status"] == "ok"
    assert len(result["data"]) == 3


if __name__ == "__main__":
    pytest.main([__file__])
