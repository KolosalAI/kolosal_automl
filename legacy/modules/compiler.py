"""
Python Bytecode Compiler for Kolosal AutoML

This module provides functionality to compile Python source files to bytecode (.pyc files)
for improved startup performance and import speed.
"""

import os
import sys
import py_compile
import compileall
import importlib.util
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class BytecodeCompiler:
    """
    Handles compilation of Python source files to bytecode.
    
    Features:
    - Compile individual files or entire directories
    - Parallel compilation for better performance
    - Error handling and logging
    - Cache validation and recompilation
    - Integration with project structure
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize the bytecode compiler.
        
        Args:
            project_root: Root directory of the project. If None, uses current directory.
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        self.project_root = Path(project_root).resolve()
        self.compiled_files: Dict[str, float] = {}  # file_path -> compile_time
        self.lock = threading.Lock()
        
        # Directories to include in compilation
        self.include_dirs = [
            self.project_root / "modules",
            self.project_root / "benchmark", 
            self.project_root,  # For main.py, app.py, etc.
        ]
        
        # Files/patterns to exclude
        self.exclude_patterns = [
            "test_*.py",
            "*_test.py", 
            "tests/*",
            "__pycache__/*",
            ".git/*",
            ".venv/*",
            "venv/*",
            "env/*",
            "build/*",
            "dist/*",
            "*.egg-info/*",
        ]
    
    def should_exclude(self, file_path: Path) -> bool:
        """Check if a file should be excluded from compilation."""
        from fnmatch import fnmatch
        
        # Convert to relative path for pattern matching
        try:
            rel_path = file_path.relative_to(self.project_root)
            rel_path_str = str(rel_path)
        except ValueError:
            # File is outside project root
            return True
        
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if fnmatch(rel_path_str, pattern):
                return True
            if fnmatch(file_path.name, pattern):
                return True
        
        return False
    
    def get_python_files(self, directory: Path) -> List[Path]:
        """Get all Python files in a directory recursively."""
        python_files = []
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return python_files
        
        for file_path in directory.rglob("*.py"):
            if not self.should_exclude(file_path):
                python_files.append(file_path)
        
        return python_files
    
    def needs_compilation(self, source_file: Path) -> bool:
        """
        Check if a source file needs compilation.
        
        Args:
            source_file: Path to the source .py file
            
        Returns:
            True if compilation is needed, False otherwise
        """
        # Get the expected .pyc file path
        pyc_file = self._get_pyc_path(source_file)
        
        # If .pyc doesn't exist, need to compile
        if not pyc_file.exists():
            return True
        
        # Check if source is newer than bytecode
        try:
            source_mtime = source_file.stat().st_mtime
            pyc_mtime = pyc_file.stat().st_mtime
            return source_mtime > pyc_mtime
        except OSError:
            # If we can't stat, assume we need to compile
            return True
    
    def _get_pyc_path(self, source_file: Path) -> Path:
        """Get the path where the .pyc file should be located."""
        # Python stores .pyc files in __pycache__ subdirectory
        pycache_dir = source_file.parent / "__pycache__"
        filename = source_file.stem + f".cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
        return pycache_dir / filename
    
    def compile_file(self, source_file: Path, force: bool = False) -> bool:
        """
        Compile a single Python file to bytecode.
        
        Args:
            source_file: Path to the source .py file
            force: If True, compile even if not needed
            
        Returns:
            True if compilation succeeded, False otherwise
        """
        try:
            # Check if compilation is needed
            if not force and not self.needs_compilation(source_file):
                logger.debug(f"Skipping {source_file} - already up to date")
                return True
            
            # Compile the file
            logger.debug(f"Compiling {source_file}")
            py_compile.compile(
                str(source_file),
                doraise=True,
                optimize=2  # Maximum optimization
            )
            
            # Track compilation
            with self.lock:
                self.compiled_files[str(source_file)] = time.time()
            
            return True
            
        except py_compile.PyCompileError as e:
            logger.error(f"Failed to compile {source_file}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error compiling {source_file}: {e}")
            return False
    
    def compile_directory(self, directory: Path, max_workers: Optional[int] = None, force: bool = False) -> Dict[str, bool]:
        """
        Compile all Python files in a directory using parallel processing.
        
        Args:
            directory: Directory to compile
            max_workers: Maximum number of worker threads
            force: If True, compile all files even if not needed
            
        Returns:
            Dictionary mapping file paths to compilation success status
        """
        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 1)
        
        # Get all Python files
        python_files = self.get_python_files(directory)
        
        if not python_files:
            logger.info(f"No Python files found in {directory}")
            return {}
        
        logger.info(f"Compiling {len(python_files)} files in {directory} using {max_workers} workers")
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel compilation
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit compilation tasks
            future_to_file = {
                executor.submit(self.compile_file, file_path, force): file_path
                for file_path in python_files
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success = future.result()
                    results[str(file_path)] = success
                except Exception as e:
                    logger.error(f"Failed to compile {file_path}: {e}")
                    results[str(file_path)] = False
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Compilation complete: {successful}/{len(results)} files successful")
        
        return results
    
    def compile_project(self, max_workers: Optional[int] = None, force: bool = False) -> Dict[str, Dict[str, bool]]:
        """
        Compile the entire project.
        
        Args:
            max_workers: Maximum number of worker threads
            force: If True, compile all files even if not needed
            
        Returns:
            Dictionary mapping directory names to compilation results
        """
        logger.info("Starting project compilation...")
        start_time = time.time()
        
        all_results = {}
        
        for directory in self.include_dirs:
            if directory.exists():
                logger.info(f"Compiling directory: {directory}")
                results = self.compile_directory(directory, max_workers, force)
                all_results[str(directory)] = results
            else:
                logger.warning(f"Directory not found: {directory}")
                all_results[str(directory)] = {}
        
        # Calculate totals
        total_files = sum(len(results) for results in all_results.values())
        total_successful = sum(
            sum(1 for success in results.values() if success)
            for results in all_results.values()
        )
        
        duration = time.time() - start_time
        logger.info(f"Project compilation complete: {total_successful}/{total_files} files in {duration:.2f}s")
        
        return all_results
    
    def clean_bytecode(self, directory: Optional[Path] = None) -> int:
        """
        Clean all bytecode files in the project or specified directory.
        
        Args:
            directory: Directory to clean. If None, cleans entire project.
            
        Returns:
            Number of files removed
        """
        if directory is None:
            directories = self.include_dirs
        else:
            directories = [directory]
        
        removed_count = 0
        
        for dir_path in directories:
            if not dir_path.exists():
                continue
            
            # Find all __pycache__ directories
            for pycache_dir in dir_path.rglob("__pycache__"):
                try:
                    # Remove all .pyc files
                    for pyc_file in pycache_dir.glob("*.pyc"):
                        pyc_file.unlink()
                        removed_count += 1
                    
                    # Remove the directory if it's empty
                    if not any(pycache_dir.iterdir()):
                        pycache_dir.rmdir()
                        
                except Exception as e:
                    logger.error(f"Error cleaning {pycache_dir}: {e}")
        
        logger.info(f"Cleaned {removed_count} bytecode files")
        return removed_count
    
    def get_compilation_stats(self) -> Dict[str, any]:
        """Get statistics about compiled files."""
        with self.lock:
            stats = {
                "total_compiled": len(self.compiled_files),
                "last_compilation": max(self.compiled_files.values()) if self.compiled_files else None,
                "files": dict(self.compiled_files)
            }
        
        return stats


def compile_on_import():
    """
    Compile modules automatically when this module is imported.
    This can be called from __init__.py files for automatic compilation.
    """
    try:
        compiler = BytecodeCompiler()
        
        # Only compile if we detect we're in a development/first-run scenario
        main_module_file = compiler.project_root / "modules" / "__init__.py"
        if compiler.needs_compilation(main_module_file):
            logger.info("Performing automatic bytecode compilation...")
            compiler.compile_project(max_workers=2, force=False)  # Conservative settings
        
    except Exception as e:
        logger.warning(f"Automatic compilation failed: {e}")
        # Don't fail the import if compilation fails


def cli_compile_command():
    """Command-line interface for compilation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compile Kolosal AutoML to bytecode")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force recompilation of all files"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true", 
        help="Clean all bytecode files"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker threads"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    parser.add_argument(
        "--directory", "-d",
        type=str,
        help="Compile specific directory only"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    compiler = BytecodeCompiler()
    
    if args.clean:
        if args.directory:
            compiler.clean_bytecode(Path(args.directory))
        else:
            compiler.clean_bytecode()
        return
    
    # Compile
    if args.directory:
        directory = Path(args.directory)
        results = compiler.compile_directory(directory, args.workers, args.force)
    else:
        results = compiler.compile_project(args.workers, args.force)
    
    # Print results
    if args.verbose:
        print("\nDetailed Results:")
        if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            # Project results
            for dir_name, dir_results in results.items():
                print(f"\n{dir_name}:")
                for file_path, success in dir_results.items():
                    status = "✓" if success else "✗"
                    print(f"  {status} {file_path}")
        else:
            # Directory results
            for file_path, success in results.items():
                status = "✓" if success else "✗"
                print(f"  {status} {file_path}")


if __name__ == "__main__":
    cli_compile_command()
