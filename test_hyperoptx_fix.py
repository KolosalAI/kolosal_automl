#!/usr/bin/env python3
"""
Test script to verify that the HyperOptX get_params() fix is working correctly.
"""

import sys
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add the modules path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

try:
    from optimizer.hyperoptx import HyperOptX
    print("✓ Successfully imported HyperOptX")
except ImportError as e:
    print(f"✗ Failed to import HyperOptX: {e}")
    sys.exit(1)

def test_get_params():
    """Test that HyperOptX has a working get_params method."""
    print("\n1. Testing get_params() method...")
    
    # Create a simple estimator and parameter space
    estimator = RandomForestClassifier(random_state=42)
    param_space = {
        'n_estimators': [10, 50],
        'max_depth': [3, 5, None],
    }
    
    # Create HyperOptX instance
    optimizer = HyperOptX(
        estimator=estimator,
        param_space=param_space,
        max_iter=5,  # Very small for testing
        random_state=42,
        verbose=0
    )
    
    # Test get_params method
    try:
        params = optimizer.get_params()
        print(f"✓ get_params() works: returned {len(params)} parameters")
        
        # Check that basic parameters are included
        expected_params = ['estimator', 'param_space', 'max_iter', 'random_state']
        for param in expected_params:
            if param in params:
                print(f"✓ Parameter '{param}' found in get_params()")
            else:
                print(f"✗ Parameter '{param}' missing from get_params()")
                
    except Exception as e:
        print(f"✗ get_params() failed: {e}")
        return False
    
    return True

def test_set_params():
    """Test that HyperOptX has a working set_params method."""
    print("\n2. Testing set_params() method...")
    
    # Create a simple estimator and parameter space
    estimator = RandomForestClassifier(random_state=42)
    param_space = {
        'n_estimators': [10, 50],
        'max_depth': [3, 5, None],
    }
    
    # Create HyperOptX instance
    optimizer = HyperOptX(
        estimator=estimator,
        param_space=param_space,
        max_iter=5,
        random_state=42,
        verbose=0
    )
    
    # Test set_params method
    try:
        # Change some parameters
        optimizer.set_params(max_iter=10, verbose=1)
        
        # Verify the changes
        if optimizer.max_iter == 10:
            print("✓ set_params() successfully changed max_iter")
        else:
            print(f"✗ set_params() failed to change max_iter: {optimizer.max_iter}")
            return False
            
        if optimizer.verbose == 1:
            print("✓ set_params() successfully changed verbose")
        else:
            print(f"✗ set_params() failed to change verbose: {optimizer.verbose}")
            return False
            
    except Exception as e:
        print(f"✗ set_params() failed: {e}")
        return False
    
    return True

def test_basic_optimization():
    """Test that HyperOptX can perform basic optimization without the get_params error."""
    print("\n3. Testing basic optimization...")
    
    # Generate simple dataset
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create a simple estimator and parameter space
    estimator = RandomForestClassifier(random_state=42)
    param_space = {
        'n_estimators': [10, 20],
        'max_depth': [3, 5],
    }
    
    # Create HyperOptX instance with minimal iterations
    optimizer = HyperOptX(
        estimator=estimator,
        param_space=param_space,
        max_iter=3,  # Very small for quick testing
        cv=2,  # Minimal CV folds
        random_state=42,
        verbose=0
    )
    
    try:
        # Fit the optimizer
        print("   Running optimization...")
        optimizer.fit(X_train, y_train)
        
        # Check that we have best parameters
        if hasattr(optimizer, 'best_params_') and optimizer.best_params_:
            print(f"✓ Optimization completed. Best params: {optimizer.best_params_}")
        else:
            print("✗ Optimization completed but no best_params_ found")
            return False
            
        # Test the get_params method on the fitted optimizer
        params = optimizer.get_params()
        if params:
            print("✓ get_params() works on fitted optimizer")
        else:
            print("✗ get_params() returned empty result on fitted optimizer")
            return False
            
    except Exception as e:
        import traceback
        print(f"✗ Optimization failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing HyperOptX get_params() fix...")
    
    tests = [
        test_get_params,
        test_set_params,
        test_basic_optimization
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n--- Test Results ---")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 All tests passed! The HyperOptX get_params() fix is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
