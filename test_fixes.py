#!/usr/bin/env python3
"""
Simple test script to verify specific fixes without running the full test suite.
This avoids dependency conflicts and focuses on the core fixes.
"""

import sys
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_secrets_manager_password():
    """Test the secrets manager database password generation fix"""
    print("Testing secrets manager password generation...")
    try:
        from modules.security.secrets_manager import SecretsManager, SecretType
        
        manager = SecretsManager()
        password = manager.generate_secret(SecretType.DATABASE_PASSWORD, 16)
        
        # Check that password has all required character types
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_symbol = any(c in "!@#$%^&*" for c in password)
        
        assert len(password) == 16, f"Password length is {len(password)}, expected 16"
        assert has_upper, f"Password {password} missing uppercase letters"
        assert has_lower, f"Password {password} missing lowercase letters"
        assert has_digit, f"Password {password} missing digits"
        assert has_symbol, f"Password {password} missing symbols"
        
        print(f"✓ Password generation test passed: {password}")
        return True
    except Exception as e:
        print(f"✗ Password generation test failed: {e}")
        return False

def test_inference_engine_validation():
    """Test the inference engine validation fix"""
    print("Testing inference engine validation...")
    try:
        from modules.engine.inference_engine import InferenceEngine
        from modules.configs import InferenceEngineConfig
        
        config = InferenceEngineConfig(
            enable_batching=False,
            enable_feature_scaling=False,
            enable_quantization=False,
            debug_mode=True
        )
        
        engine = InferenceEngine(config)
        result = engine.validate_model()
        
        assert 'valid' in result, f"Result missing 'valid' key: {result.keys()}"
        assert result['valid'] is False, f"Expected valid=False for no model, got {result['valid']}"
        
        print(f"✓ Inference engine validation test passed")
        return True
    except Exception as e:
        print(f"✗ Inference engine validation test failed: {e}")
        return False

def test_memory_processor_timing():
    """Test the memory processor timing fix"""
    print("Testing memory processor timing...")
    try:
        from modules.engine.memory_aware_processor import MemoryAwareDataProcessor
        import pandas as pd
        import numpy as np
        import time
        
        processor = MemoryAwareDataProcessor()
        
        # Create test data
        df = pd.DataFrame({
            'col1': np.random.randn(1000),
            'col2': np.random.randn(1000),
            'col3': np.random.randn(1000)
        })
        
        def slow_process(chunk):
            time.sleep(0.001)  # Small delay to ensure measurable time
            return chunk * 2
        
        # Process data
        result = processor.process_with_adaptive_chunking(
            df,
            process_func=slow_process
        )
        
        # Check stats
        stats = processor.chunk_processor.get_performance_stats()
        
        assert 'total_time_seconds' in stats, f"Stats missing 'total_time_seconds': {stats.keys()}"
        assert stats['total_time_seconds'] >= 0, f"Total time is negative: {stats['total_time_seconds']}"
        assert stats['total_chunks_processed'] > 0, f"No chunks processed: {stats['total_chunks_processed']}"
        
        print(f"✓ Memory processor timing test passed: {stats['total_time_seconds']}s")
        return True
    except Exception as e:
        print(f"✗ Memory processor timing test failed: {e}")
        return False

def main():
    """Run all test functions"""
    print("Running targeted tests for specific fixes...\n")
    
    tests = [
        test_secrets_manager_password,
        test_inference_engine_validation,
        test_memory_processor_timing,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All targeted tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
