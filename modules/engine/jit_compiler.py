"""
JIT Compilation for Frequent Model Patterns
Provides JIT compilation capabilities for hot paths in ML workloads using Numba.
"""

import logging
import functools
import inspect
import hashlib
import time
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import threading

try:
    import numba
    from numba import njit, jit, prange
    from numba.typed import Dict as NumbaDict
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class CompilationStats:
    """Statistics for JIT compilation performance."""
    function_name: str
    call_count: int
    total_compile_time: float
    total_execution_time: float
    average_speedup: float
    last_compiled: float


class HotPathTracker:
    """Tracks function call frequency to identify hot paths for JIT compilation."""
    
    def __init__(self, min_calls_for_compilation: int = 10):
        self.min_calls_for_compilation = min_calls_for_compilation
        self.call_counts = defaultdict(int)
        self.execution_times = defaultdict(list)
        self.lock = threading.RLock()
        
    def track_call(self, func_name: str, execution_time: float):
        """Track a function call for hot path analysis."""
        with self.lock:
            self.call_counts[func_name] += 1
            self.execution_times[func_name].append(execution_time)
            
    def should_compile(self, func_name: str) -> bool:
        """Determine if a function should be JIT compiled."""
        with self.lock:
            return self.call_counts[func_name] >= self.min_calls_for_compilation
    
    def get_hot_functions(self) -> List[str]:
        """Get list of functions that are hot paths."""
        with self.lock:
            return [
                func_name for func_name, count in self.call_counts.items()
                if count >= self.min_calls_for_compilation
            ]


class JITCompiler:
    """
    JIT compilation manager for frequent model patterns.
    
    Features:
    - Automatic hot path detection
    - Numba JIT compilation for numerical operations
    - Performance tracking and adaptive compilation
    - Memory-efficient compilation caching
    - Fallback to interpreted execution if compilation fails
    """
    
    def __init__(self, 
                 enable_compilation: bool = True,
                 min_calls_for_compilation: int = 10,
                 cache_size: int = 100):
        """
        Initialize the JIT compiler.
        
        Args:
            enable_compilation: Whether to enable JIT compilation
            min_calls_for_compilation: Minimum calls before compiling a function
            cache_size: Maximum number of compiled functions to cache
        """
        self.enable_compilation = enable_compilation and NUMBA_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # Hot path tracking
        self.hot_path_tracker = HotPathTracker(min_calls_for_compilation)
        
        # Compiled function cache
        self.compiled_functions: Dict[str, Callable] = {}
        self.compilation_stats: Dict[str, CompilationStats] = {}
        self.cache_size = cache_size
        
        # Thread safety
        self.compilation_lock = threading.RLock()
        
        if not NUMBA_AVAILABLE:
            self.logger.warning("Numba not available. JIT compilation disabled.")
        else:
            self.logger.info("JIT compiler initialized with Numba support")
    
    def compile_if_hot(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function and compile if it's identified as a hot path.
        
        Args:
            func: Function to potentially compile and execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function execution result
        """
        if not self.enable_compilation:
            return func(*args, **kwargs)
        
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Track execution time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        # Track the call
        self.hot_path_tracker.track_call(func_name, execution_time)
        
        # Check if we should compile this function
        if (self.hot_path_tracker.should_compile(func_name) and 
            func_name not in self.compiled_functions):
            self._attempt_compilation(func, func_name)
        
        return result
    
    def _attempt_compilation(self, func: Callable, func_name: str):
        """Attempt to JIT compile a function."""
        with self.compilation_lock:
            # Double-check to avoid race condition
            if func_name in self.compiled_functions:
                return
            
            try:
                compile_start = time.perf_counter()
                
                # Determine compilation strategy based on function characteristics
                compiled_func = self._compile_function(func)
                
                compile_time = time.perf_counter() - compile_start
                
                if compiled_func is not None:
                    self.compiled_functions[func_name] = compiled_func
                    self.compilation_stats[func_name] = CompilationStats(
                        function_name=func_name,
                        call_count=self.hot_path_tracker.call_counts[func_name],
                        total_compile_time=compile_time,
                        total_execution_time=0.0,
                        average_speedup=0.0,
                        last_compiled=time.time()
                    )
                    
                    self.logger.info(f"Successfully compiled function {func_name} in {compile_time:.4f}s")
                    
                    # Cleanup old entries if cache is full
                    self._cleanup_cache()
                
            except Exception as e:
                self.logger.warning(f"Failed to compile function {func_name}: {str(e)}")
    
    def _compile_function(self, func: Callable) -> Optional[Callable]:
        """Compile a function using appropriate Numba decorators."""
        try:
            # Analyze function signature and body to determine compilation strategy
            signature = inspect.signature(func)
            
            # For numerical operations, use njit with optimization
            if self._is_numerical_function(func):
                return njit(
                    func,
                    cache=True,
                    fastmath=True,
                    parallel=True if self._supports_parallel(func) else False
                )
            
            # For general functions, use standard jit
            return jit(func, cache=True, forceobj=True)
            
        except Exception as e:
            self.logger.debug(f"Compilation failed for {func.__name__}: {str(e)}")
            return None
    
    def _is_numerical_function(self, func: Callable) -> bool:
        """Determine if a function is suitable for numerical JIT compilation."""
        # Check function source for numerical operations
        try:
            source = inspect.getsource(func)
            numerical_keywords = ['numpy', 'np.', 'array', 'dot', 'sum', 'mean', 'std', 'var']
            return any(keyword in source for keyword in numerical_keywords)
        except:
            return False
    
    def _supports_parallel(self, func: Callable) -> bool:
        """Determine if a function can benefit from parallel compilation."""
        try:
            source = inspect.getsource(func)
            parallel_keywords = ['for', 'while', 'sum', 'mean', 'dot']
            return any(keyword in source for keyword in parallel_keywords)
        except:
            return False
    
    def _cleanup_cache(self):
        """Remove oldest compiled functions if cache is full."""
        if len(self.compiled_functions) > self.cache_size:
            # Remove the oldest compiled function
            oldest_func = min(
                self.compilation_stats.keys(),
                key=lambda x: self.compilation_stats[x].last_compiled
            )
            del self.compiled_functions[oldest_func]
            del self.compilation_stats[oldest_func]
    
    def get_compiled_function(self, func_name: str) -> Optional[Callable]:
        """Get a compiled version of a function if available."""
        return self.compiled_functions.get(func_name)
    
    def get_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for compiled functions."""
        stats = {}
        for func_name, stat in self.compilation_stats.items():
            stats[func_name] = {
                'call_count': stat.call_count,
                'compile_time': stat.total_compile_time,
                'execution_time': stat.total_execution_time,
                'average_speedup': stat.average_speedup,
                'last_compiled': stat.last_compiled
            }
        return stats
    
    def clear_cache(self):
        """Clear all compiled functions from cache."""
        with self.compilation_lock:
            self.compiled_functions.clear()
            self.compilation_stats.clear()
            self.hot_path_tracker.call_counts.clear()
            self.hot_path_tracker.execution_times.clear()


# Decorators for easy JIT compilation
def jit_if_hot(compiler: Optional[JITCompiler] = None, 
               min_calls: int = 10):
    """
    Decorator to automatically JIT compile functions when they become hot paths.
    
    Args:
        compiler: JIT compiler instance (creates default if None)
        min_calls: Minimum calls before compilation
    """
    if compiler is None:
        compiler = JITCompiler(min_calls_for_compilation=min_calls)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return compiler.compile_if_hot(func, *args, **kwargs)
        return wrapper
    return decorator


# Pre-compiled numerical operations for common ML patterns
if NUMBA_AVAILABLE:
    
    @njit(cache=True, fastmath=True, parallel=True)
    def fast_dot_product(a: np.ndarray, b: np.ndarray) -> float:
        """Fast dot product computation."""
        result = 0.0
        for i in prange(len(a)):
            result += a[i] * b[i]
        return result
    
    @njit(cache=True, fastmath=True, parallel=True)
    def fast_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fast matrix multiplication."""
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        
        result = np.zeros((m, n))
        for i in prange(m):
            for j in range(n):
                for l in range(k):
                    result[i, j] += a[i, l] * b[l, j]
        return result
    
    @njit(cache=True, fastmath=True, parallel=True)
    def fast_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Fast Euclidean distance computation."""
        result = 0.0
        for i in prange(len(a)):
            diff = a[i] - b[i]
            result += diff * diff
        return np.sqrt(result)
    
    @njit(cache=True, fastmath=True, parallel=True)
    def fast_standardize(data: np.ndarray) -> np.ndarray:
        """Fast data standardization."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    
    @njit(cache=True, fastmath=True, parallel=True)
    def fast_sigmoid(x: np.ndarray) -> np.ndarray:
        """Fast sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-x))
    
    @njit(cache=True, fastmath=True, parallel=True)
    def fast_softmax(x: np.ndarray) -> np.ndarray:
        """Fast softmax computation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

else:
    # Fallback implementations using NumPy
    def fast_dot_product(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b)
    
    def fast_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b)
    
    def fast_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)
    
    def fast_standardize(data: np.ndarray) -> np.ndarray:
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    
    def fast_sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def fast_softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


# Global JIT compiler instance
_global_jit_compiler = None

def get_global_jit_compiler() -> JITCompiler:
    """Get or create the global JIT compiler instance."""
    global _global_jit_compiler
    if _global_jit_compiler is None:
        _global_jit_compiler = JITCompiler()
    return _global_jit_compiler
