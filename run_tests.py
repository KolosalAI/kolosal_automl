#!/usr/bin/env python3
"""
Test runner script for the Kolosal AutoML project.

This script provides easy commands to run different types of tests:
- Unit tests
- Functional tests
- All tests
- Individual test files

Usage:
    python run_tests.py unit                    # Run all unit tests
    python run_tests.py functional              # Run all functional tests
    python run_tests.py all                     # Run all tests
    python run_tests.py tests/unit/test_*.py    # Run specific test files
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, verbose=False):
    """Run a command and handle the output."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, check=False)
        
        if verbose:
            return result.returncode
        else:
            if result.returncode == 0:
                print("✅ Tests passed!")
            else:
                print("❌ Tests failed!")
                print("\nSTDOUT:")
                print(result.stdout)
                print("\nSTDERR:")
                print(result.stderr)
            
            return result.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest first:")
        print("   pip install pytest")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run tests for Kolosal AutoML")
    parser.add_argument(
        "target", 
        choices=["unit", "functional", "all"], 
        nargs="?", 
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "-k", "--keyword", 
        help="Run tests matching keyword expression"
    )
    parser.add_argument(
        "-m", "--marker", 
        help="Run tests with specific marker (unit, functional, slow, etc.)"
    )
    parser.add_argument(
        "--file", 
        help="Run specific test file"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.extend(["-v", "--tb=short"])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=modules", "--cov-report=term-missing", "--cov-report=html"])
    
    # Add keyword filtering
    if args.keyword:
        cmd.extend(["-k", args.keyword])
    
    # Add marker filtering
    if args.marker:
        cmd.extend(["-m", args.marker])
    
    # Determine test paths
    if args.file:
        cmd.append(args.file)
    elif args.target == "unit":
        cmd.extend(["-m", "unit", "tests/unit/"])
    elif args.target == "functional":
        cmd.extend(["-m", "functional", "tests/functional/"])
    elif args.target == "all":
        cmd.append("tests/")
    
    # Run the tests
    print(f"🧪 Running {args.target} tests...")
    return_code = run_command(cmd, verbose=args.verbose)
    
    if return_code == 0:
        print(f"\n🎉 All {args.target} tests passed!")
    else:
        print(f"\n💥 Some {args.target} tests failed!")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())
