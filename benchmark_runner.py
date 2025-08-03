#!/usr/bin/env python3
"""
Convenience script to run benchmarks from the root directory.
This script forwards all arguments to the actual benchmark script.
"""

import sys
import os
from pathlib import Path

# Add benchmark directory to path
benchmark_dir = Path(__file__).parent / "benchmark"
sys.path.insert(0, str(benchmark_dir))

# Change to benchmark directory for proper relative paths
original_cwd = os.getcwd()
os.chdir(benchmark_dir)

try:
    # Import and run the actual benchmark script
    from benchmark_runner import main
    main()
finally:
    # Restore original working directory
    os.chdir(original_cwd)
