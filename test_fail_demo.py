#!/usr/bin/env python3
"""
Quick test to demonstrate the fixed logging behavior
"""

def test_will_pass():
    """This test will pass"""
    assert True

def test_will_fail():
    """This test will fail to demonstrate the [INFO] [FAIL] log"""
    assert False, "This test intentionally fails to show the logging fix"

def test_will_be_skipped():
    """This test will be skipped"""
    import pytest
    pytest.skip("Skipping to show logging")

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
