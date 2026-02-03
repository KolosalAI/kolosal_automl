#!/usr/bin/env python3
"""
Standalone Kolosal AutoML Bytecode Compiler

This script compiles the entire Kolosal AutoML project to bytecode for improved
startup performance. It can be run independently or as part of a build process.

Usage:
    python compile.py                    # Compile with default settings
    python compile.py --force            # Force recompile all files
    python compile.py --clean            # Clean all bytecode files
    python compile.py --workers 8        # Use 8 worker threads
    python compile.py --verbose          # Verbose output
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for standalone compilation."""
    try:
        from modules.compiler import cli_compile_command
        cli_compile_command()
    except ImportError as e:
        print(f"❌ Error: Could not import compiler module: {e}")
        print("   Make sure you're running this from the project root directory")
        return 1
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
