# Kolosal AutoML Bytecode Compilation

This document explains how to use the bytecode compilation features in Kolosal AutoML to improve startup performance and import speed.

## Overview

Python bytecode compilation converts `.py` source files to `.pyc` bytecode files, which Python can load and execute faster than parsing source files each time. This is especially beneficial for:

- **Faster startup times** - Reduced import overhead
- **Better performance** - Pre-compiled bytecode loads faster
- **Production deployments** - Consistent performance across runs
- **Large codebases** - Significant improvement for projects with many modules

## Usage Methods

### 1. Command Line Interface (Recommended)

#### Using the main CLI:
```bash
# Compile the entire project
python main.py --compile

# Clean all bytecode files  
python main.py --clean-bytecode

# Interactive menu (includes compile options)
python main.py
# Then navigate to: System Information > Compile to Bytecode
```

#### Using the standalone compiler:
```bash
# Basic compilation
python compile.py

# Force recompile all files (even if up to date)
python compile.py --force

# Clean all bytecode files
python compile.py --clean

# Use more worker threads for faster compilation
python compile.py --workers 8

# Verbose output showing each file
python compile.py --verbose

# Compile specific directory only
python compile.py --directory modules
```

#### Using the batch/shell scripts:
```bash
# On Windows
compile.bat

# On Unix/Linux/macOS (after making executable with chmod +x compile.sh)
./compile.sh

# Pass arguments to the scripts
compile.bat --force --verbose
./compile.sh --clean
```

### 2. Automatic Compilation

Enable automatic compilation on first import by setting an environment variable:

```bash
# Windows (PowerShell)
$env:KOLOSAL_AUTO_COMPILE = "true"
python main.py

# Windows (Command Prompt)
set KOLOSAL_AUTO_COMPILE=true
python main.py

# Unix/Linux/macOS
export KOLOSAL_AUTO_COMPILE=true
python main.py
```

### 3. Programmatic Usage

```python
from modules.compiler import BytecodeCompiler

# Create compiler instance
compiler = BytecodeCompiler()

# Compile entire project
results = compiler.compile_project()

# Compile specific directory
results = compiler.compile_directory(Path("modules"))

# Clean bytecode files
removed_count = compiler.clean_bytecode()

# Check if file needs compilation
needs_compile = compiler.needs_compilation(Path("modules/engine.py"))

# Get compilation statistics
stats = compiler.get_compilation_stats()
```

## Performance Benefits

After compilation, you should notice:

- **Faster startup** - Initial imports will be significantly faster
- **Reduced I/O** - Less file system access during imports
- **Consistent performance** - No parsing overhead on subsequent runs
- **Better caching** - Python's import system works more efficiently

Typical performance improvements:
- **Small projects**: 10-30% faster startup
- **Medium projects**: 30-50% faster startup  
- **Large projects**: 50%+ faster startup

## When to Recompile

The compiler automatically detects when recompilation is needed:

- **Source file modified** - `.py` file is newer than `.pyc` file
- **Missing bytecode** - No `.pyc` file exists
- **Python version change** - Bytecode is version-specific
- **Force compilation** - Using `--force` flag

## File Structure

Compiled bytecode files are stored in `__pycache__` directories:

```
modules/
├── engine.py
├── configs.py
└── __pycache__/
    ├── engine.cpython-311.pyc
    ├── configs.cpython-311.pyc
    └── ...
```

The naming convention includes the Python version (e.g., `cpython-311` for Python 3.11).

## Troubleshooting

### Common Issues

1. **Import errors during compilation**:
   - Check that all dependencies are installed
   - Ensure you're in the correct directory
   - Run with `--verbose` to see detailed error messages

2. **Permission errors**:
   - Ensure write permissions to project directories
   - On Unix/Linux, you might need appropriate file permissions

3. **Compilation failures**:
   - Syntax errors in source files will prevent compilation
   - Check the error messages for specific file issues
   - Use `--clean` to remove corrupted bytecode files

### Debugging

```bash
# Verbose output to see what's happening
python compile.py --verbose

# Check compilation statistics
python -c "from modules.compiler import BytecodeCompiler; c = BytecodeCompiler(); print(c.get_compilation_stats())"

# Manual file-by-file testing
python -c "import py_compile; py_compile.compile('modules/engine.py', doraise=True)"
```

## Integration with Build Process

### uv (Recommended)

Add to your workflow:

```bash
# Install dependencies and compile
uv install
uv run python compile.py

# Run with compiled code
uv run python main.py
```

### Docker

Add to your Dockerfile:

```dockerfile
# Copy source code
COPY . /app
WORKDIR /app

# Install dependencies
RUN uv install

# Compile to bytecode
RUN python compile.py --force

# The compiled bytecode will be included in the image
CMD ["python", "main.py"]
```

### CI/CD Pipelines

Example GitHub Actions step:

```yaml
- name: Compile to bytecode
  run: |
    python compile.py --force
    # Optionally verify compilation
    python -c "import modules; print('Compilation successful')"
```

## Environment Variables

- `KOLOSAL_AUTO_COMPILE=true` - Enable automatic compilation on import
- `PYTHONOPTIMIZE=2` - Enable Python's built-in optimization level 2
- `PYTHONDONTWRITEBYTECODE=1` - Disable bytecode generation (opposite of what we want)

## Best Practices

1. **Compile before production deployment**
2. **Use `--force` in CI/CD pipelines** to ensure clean compilation
3. **Include bytecode in Docker images** for faster container startup
4. **Clean and recompile after major changes**
5. **Monitor compilation errors** - they indicate potential code issues
6. **Use multiple workers** (`--workers`) for large projects

## Performance Monitoring

Track the impact of compilation:

```python
import time

# Time import without compilation
start = time.time()
import modules
import_time = time.time() - start

print(f"Import time: {import_time:.3f} seconds")
```

Run this before and after compilation to measure the improvement.
