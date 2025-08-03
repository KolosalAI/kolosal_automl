#!/usr/bin/env python3
"""
Run Standard ML Benchmark Script

This script provides a simple interface to run standard ML benchmarks.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the standard ML benchmark with default settings."""
    
    # Path to the benchmark script
    script_path = Path(__file__).parent / "benchmark" / "standard_ml_benchmark.py"
    
    # Default arguments
    default_args = [
        "--datasets", "iris", "wine", "breast_cancer",
        "--models", "random_forest", "logistic_regression", 
        "--optimization", "random_search"
    ]
    
    # Combine with any provided arguments
    cmd = [sys.executable, str(script_path)] + default_args + sys.argv[1:]
    
    print("Running Standard ML Benchmark...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the benchmark
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
