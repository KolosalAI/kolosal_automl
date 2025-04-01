#!/usr/bin/env python
"""
Script to run all the tests in the ML Modules project.
This script provides options to run all tests or specific test types,
with configurable verbosity and parallel execution.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ML Modules tests")
    
    parser.add_argument(
        "--test-type", 
        choices=["all", "unit", "functional", "integration"], 
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="count", 
        default=0,
        help="Increase verbosity (can be used multiple times, e.g., -vv for more verbosity)"
    )
    
    parser.add_argument(
        "--parallel", "-p", 
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=None,
        help="Number of parallel jobs (default: auto)"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate HTML test report"
    )
    
    parser.add_argument(
        "--specific-test",
        help="Run a specific test file, class, or function"
    )
    
    return parser.parse_args()


def build_command(args):
    """Build the pytest command based on arguments."""
    cmd = ["pytest"]
    
    # Verbosity
    for _ in range(args.verbose):
        cmd.append("-v")
    
    # Fail fast
    if args.fail_fast:
        cmd.append("-x")
    
    # Coverage
    if args.coverage:
        cmd.extend(["--cov=modules", "--cov-report=term", "--cov-report=html"])
    
    # HTML Report
    if args.report:
        cmd.append("--html=test_report.html")
    
    # Parallel execution
    if args.parallel:
        cmd.append("-xvs")
        if args.jobs:
            cmd.append(f"-n={args.jobs}")
        else:
            cmd.append("-n=auto")
    
    # Test type
    if args.test_type == "unit":
        cmd.append("tests/unit/")
    elif args.test_type == "functional":
        cmd.append("tests/functional/")
    elif args.test_type == "integration":
        cmd.append("tests/integration/")
    else:  # all
        cmd.append("tests/")
    
    # Specific test
    if args.specific_test:
        cmd.append(args.specific_test)
    
    return cmd


def run_tests(cmd):
    """Run the tests with the given command."""
    try:
        # Set up PYTHONPATH to include the current directory
        env = os.environ.copy()
        current_path = os.getcwd()
        python_path = env.get("PYTHONPATH", "")
        if python_path:
            env["PYTHONPATH"] = f"{current_path}:{python_path}"
        else:
            env["PYTHONPATH"] = current_path
        
        # Run the command
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import pytest
        return True
    except ImportError:
        print("Error: pytest is not installed. Please install it with: pip install pytest")
        return False


def main():
    """Main function to run tests."""
    if not check_dependencies():
        return 1
    
    args = parse_args()
    cmd = build_command(args)
    return run_tests(cmd)


if __name__ == "__main__":
    sys.exit(main())