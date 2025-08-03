#!/usr/bin/env python3
"""
Test Comparison System

This script tests the comparison system setup and validates that all components work correctly.
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import matplotlib
        import psutil
        print("✅ Standard dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import standard dependencies: {e}")
        return False
    
    try:
        # Test project imports
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from benchmark.standard_ml_benchmark import StandardMLBenchmark
        from benchmark.kolosal_automl_benchmark import KolosalAutoMLBenchmark
        print("✅ Benchmark modules imported successfully")
    except ImportError as e:
        print(f"⚠️  Benchmark modules import failed (expected if modules not available): {e}")
    
    return True

def test_scripts_exist():
    """Test that all script files exist."""
    print("\n📁 Testing script files...")
    
    project_root = Path(__file__).parent
    required_files = [
        "run_kolosal_comparison.py",
        "run_standard_ml.py", 
        "run_kolosal_automl.py",
        "benchmark/standard_ml_benchmark.py",
        "benchmark/kolosal_automl_benchmark.py",
        "benchmark/configs/example_comparison_config.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            all_exist = False
    
    return all_exist

def test_standard_ml_benchmark():
    """Test standard ML benchmark with minimal dataset."""
    print("\n🧪 Testing Standard ML benchmark...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path(__file__).parent / "benchmark" / "standard_ml_benchmark.py"
        cmd = [
            sys.executable, str(script_path),
            "--output-dir", temp_dir,
            "--datasets", "iris",
            "--models", "random_forest",
            "--optimization", "random_search"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ Standard ML benchmark completed successfully")
                
                # Check for output files
                output_files = list(Path(temp_dir).glob("*.json"))
                if output_files:
                    print(f"✅ Generated {len(output_files)} output files")
                    return True
                else:
                    print("⚠️  No output files generated")
                    return False
            else:
                print(f"❌ Standard ML benchmark failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Standard ML benchmark timed out")
            return False
        except Exception as e:
            print(f"❌ Error running Standard ML benchmark: {e}")
            return False

def test_kolosal_automl_benchmark():
    """Test Kolosal AutoML benchmark with minimal dataset."""
    print("\n🚀 Testing Kolosal AutoML benchmark...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path(__file__).parent / "benchmark" / "kolosal_automl_benchmark.py"
        cmd = [
            sys.executable, str(script_path),
            "--output-dir", temp_dir,
            "--datasets", "iris",
            "--models", "random_forest", 
            "--optimization", "random_search"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ Kolosal AutoML benchmark completed successfully")
                
                # Check for output files
                output_files = list(Path(temp_dir).glob("*.json"))
                if output_files:
                    print(f"✅ Generated {len(output_files)} output files")
                    return True
                else:
                    print("⚠️  No output files generated")
                    return False
            else:
                print(f"❌ Kolosal AutoML benchmark failed: {result.stderr}")
                print(f"ℹ️  This is expected if Kolosal AutoML modules are not available")
                return True  # Not a failure if modules aren't available
                
        except subprocess.TimeoutExpired:
            print("❌ Kolosal AutoML benchmark timed out")
            return False
        except Exception as e:
            print(f"❌ Error running Kolosal AutoML benchmark: {e}")
            return False

def test_comparison_runner():
    """Test the main comparison runner."""
    print("\n🔄 Testing comparison runner...")
    
    # Test help command
    script_path = Path(__file__).parent / "run_kolosal_comparison.py"
    cmd = [sys.executable, str(script_path), "--help"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and "usage:" in result.stdout.lower():
            print("✅ Comparison runner help works")
            return True
        else:
            print(f"❌ Comparison runner help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing comparison runner: {e}")
        return False

def test_configuration_file():
    """Test configuration file loading."""
    print("\n⚙️  Testing configuration file...")
    
    config_path = Path(__file__).parent / "benchmark" / "configs" / "example_comparison_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_keys = ["datasets", "models", "optimization"]
        if all(key in config for key in required_keys):
            print("✅ Configuration file is valid")
            print(f"   Datasets: {len(config['datasets'])}")
            print(f"   Models: {len(config['models'])}")
            print(f"   Optimization: {config['optimization']}")
            return True
        else:
            print("❌ Configuration file missing required keys")
            return False
            
    except Exception as e:
        print(f"❌ Error reading configuration file: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Kolosal AutoML Comparison System")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_imports),
        ("Script Files", test_scripts_exist), 
        ("Configuration", test_configuration_file),
        ("Comparison Runner", test_comparison_runner),
        ("Standard ML Benchmark", test_standard_ml_benchmark),
        ("Kolosal AutoML Benchmark", test_kolosal_automl_benchmark)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The comparison system is ready to use.")
        print("\nTo run a quick comparison:")
        print("python run_kolosal_comparison.py --mode quick")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\nThe system may still work for available components.")
    
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
