@echo off
REM Kolosal AutoML Bytecode Compiler for Windows
REM This batch file compiles the Python code to bytecode for better performance

echo ============================================
echo Kolosal AutoML Bytecode Compiler
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if uv is available (preferred)
uv --version >nul 2>&1
if not errorlevel 1 (
    echo Using uv to run compilation...
    uv run python compile.py %*
) else (
    echo Using python to run compilation...
    python compile.py %*
)

echo.
echo Compilation complete!
if "%1" neq "--no-pause" pause
