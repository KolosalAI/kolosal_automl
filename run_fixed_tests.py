#!/usr/bin/env python3
"""
Script to run the originally failing tests to check how many are now fixed.
"""

import subprocess
import sys

# List of originally failing tests with correct paths and names
failing_tests = [
    # Integration tests
    "tests/integration/test_optimization_system.py::TestCompleteOptimizationWorkflow::test_large_dataset_workflow",
    
    # Security tests
    "tests/unit/test_security_config.py::TestSecurityConfig::test_production_config",
    
    # Inference engine tests  
    "tests/unit/test_inference_engine.py::TestInferenceEngine::test_validation_failure",
    
    # Memory processor tests
    "tests/unit/test_memory_aware_processor.py::TestMemoryAwareProcessor::test_numa_stats",
    "tests/unit/test_memory_aware_processor.py::TestMemoryAwareProcessor::test_memory_optimization_selection",
    
    # Model manager tests
    "tests/unit/test_model_manager.py::TestModelManager::test_secure_save_load",
    
    # Data loader tests  
    "tests/unit/test_optimized_data_loader.py::TestOptimizedDataLoader::test_load_medium_csv_chunked",
    "tests/unit/test_optimized_data_loader.py::TestErrorHandling::test_corrupted_csv_handling",
    "tests/unit/test_optimized_data_loader.py::TestOptimizedDataLoader::test_file_not_found_error",
]

def run_test(test_path):
    """Run a single test and return the result."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test
        )
        return test_path, result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return test_path, False, "", "Test timed out"
    except Exception as e:
        return test_path, False, "", str(e)

def main():
    """Run all tests and report results."""
    print("Running originally failing tests to check fixes...")
    print("=" * 80)
    
    passed = []
    failed = []
    
    for test in failing_tests:
        print(f"\nRunning: {test}")
        test_name, success, stdout, stderr = run_test(test)
        
        if success:
            print(f"✅ PASSED: {test}")
            passed.append(test)
        else:
            print(f"❌ FAILED: {test}")
            failed.append(test)
            if stderr:
                print(f"Error: {stderr}")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Total tests: {len(failing_tests)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(passed)/len(failing_tests)*100:.1f}%")
    
    if passed:
        print(f"\n✅ PASSING TESTS ({len(passed)}):")
        for test in passed:
            print(f"  - {test}")
    
    if failed:
        print(f"\n❌ FAILING TESTS ({len(failed)}):")
        for test in failed:
            print(f"  - {test}")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
