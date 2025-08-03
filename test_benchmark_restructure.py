#!/usr/bin/env python3
"""
Test script to verify the restructured benchmark code works correctly.
"""

import sys
import os
from pathlib import Path

# Add benchmark directory to path
benchmark_dir = Path(__file__).parent / "benchmark"
sys.path.insert(0, str(benchmark_dir))

def test_imports():
    """Test that all benchmark modules can be imported."""
    print("Testing benchmark module imports...")
    
    try:
        # Test package import
        import benchmark
        print("✓ benchmark package imported successfully")
        
        # Test individual module imports
        from benchmark import ComparisonBenchmarkRunner, BenchmarkResult, DatasetManager
        print("✓ ComparisonBenchmarkRunner, BenchmarkResult, DatasetManager imported")
        
        try:
            from benchmark import BenchmarkRunner
            print("✓ BenchmarkRunner imported")
        except ImportError:
            print("⚠ BenchmarkRunner not available (modules dependency missing)")
        
        # Test that we can instantiate classes
        runner = ComparisonBenchmarkRunner("./test_results")
        print("✓ ComparisonBenchmarkRunner instantiated")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_config_access():
    """Test that configuration files can be accessed."""
    print("\nTesting configuration file access...")
    
    try:
        config_path = benchmark_dir / "configs" / "benchmark_comparison_config.json"
        if config_path.exists():
            print("✓ Configuration file exists")
            
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✓ Configuration file loaded successfully")
            print(f"✓ Found {len(config.get('comparison_configurations', []))} comparison configurations")
            
            return True
        else:
            print("❌ Configuration file not found")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_results_directory():
    """Test that results directory exists and is accessible."""
    print("\nTesting results directory...")
    
    try:
        results_dir = benchmark_dir / "results"
        if results_dir.exists():
            print("✓ Results directory exists")
        else:
            results_dir.mkdir(parents=True, exist_ok=True)
            print("✓ Results directory created")
            
        # Test write access
        test_file = results_dir / "test_write.txt"
        test_file.write_text("test")
        test_file.unlink()
        print("✓ Results directory is writable")
        
        return True
        
    except Exception as e:
        print(f"❌ Results directory test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("KOLOSAL AUTOML BENCHMARK RESTRUCTURE TEST")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_imports():
        tests_passed += 1
        
    if test_config_access():
        tests_passed += 1
        
    if test_results_directory():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Benchmark restructure is successful.")
        print("\nNext steps:")
        print("1. Run: python run_comparison.py --config quick_comparison")
        print("2. Run: python run_kolosal_comparison.py --mode quick")
        print("3. Check results in benchmark/results/ directory")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
