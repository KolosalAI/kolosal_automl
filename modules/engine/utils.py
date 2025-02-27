from contextlib import contextmanager
import time
import logging
from datetime import datetime
from functools import wraps
import traceback


@contextmanager
def timer(name: str = None) -> float:
    """
    Context manager for timing code blocks.
    
    Args:
        name: Optional name for the timed operation
        
    Yields:
        Elapsed time in seconds
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    
    if name:
        print(f"{name}: {elapsed:.4f} seconds")
    
    return elapsed


def log_operation(func):
    """Decorator to log preprocessing operations."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Executing {func.__name__}")
        start_time = datetime.now()
        try:
            result = func(self, *args, **kwargs)
            if self.config.enable_monitoring:
                elapsed = (datetime.now() - start_time).total_seconds()
                self._update_metrics(func.__name__, elapsed)
            return result
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            raise
    return wrapper