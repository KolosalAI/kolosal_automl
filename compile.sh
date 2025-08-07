#!/bin/bash
# Kolosal AutoML Bytecode Compiler for Unix/Linux
# This script compiles the Python code to bytecode for better performance

echo "============================================"
echo "Kolosal AutoML Bytecode Compiler"
echo "============================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed or not in PATH"
    echo "   Please install Python and try again"
    exit 1
fi

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Check if uv is available (preferred)
if command -v uv &> /dev/null; then
    echo "📦 Using uv to run compilation..."
    uv run $PYTHON_CMD compile.py "$@"
else
    echo "🐍 Using $PYTHON_CMD to run compilation..."
    $PYTHON_CMD compile.py "$@"
fi

echo
echo "✅ Compilation complete!"
