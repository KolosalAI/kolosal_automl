"""
Centralized logging configuration for the Kolosal AutoML project.
This module provides a single place to configure logging to avoid conflicts
and ensure proper cleanup of file handlers.
"""
import logging
import logging.handlers
import os
import atexit
import threading
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager


class LoggingManager:
    """Centralized logging manager to handle all logging configuration."""
    
    _instance: Optional['LoggingManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'LoggingManager':
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._file_handlers: Dict[str, logging.FileHandler] = {}
        self._loggers: Dict[str, logging.Logger] = {}
        self._shutdown_registered = False
        self._is_shutdown = False
        
        # Create logs directory
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Register cleanup on exit
        self._register_cleanup()
    
    def _register_cleanup(self):
        """Register cleanup handler to be called on exit."""
        if not self._shutdown_registered:
            atexit.register(self.cleanup)
            self._shutdown_registered = True
    
    def get_logger(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        enable_console: bool = True,
        formatter: Optional[logging.Formatter] = None
    ) -> logging.Logger:
        """
        Get or create a logger with proper configuration.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file name (will be placed in logs directory)
            enable_console: Whether to enable console output
            formatter: Custom formatter (uses default if None)
            
        Returns:
            Configured logger instance
        """
        if self._is_shutdown:
            # Return a basic logger if already shut down
            return logging.getLogger(name)
        
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        logger.propagate = False
        
        # Default formatter
        if formatter is None:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # Add console handler if requested
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_file:
            file_handler = self._get_or_create_file_handler(log_file, level, formatter)
            if file_handler:
                logger.addHandler(file_handler)
        
        self._loggers[name] = logger
        return logger
    
    def _get_or_create_file_handler(
        self,
        log_file: str,
        level: int,
        formatter: logging.Formatter
    ) -> Optional[logging.FileHandler]:
        """Get or create a file handler with proper error handling."""
        if self._is_shutdown:
            return None
            
        if log_file in self._file_handlers:
            return self._file_handlers[log_file]
        
        try:
            log_path = self.log_dir / log_file
            
            # Use RotatingFileHandler to avoid issues with large log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            
            self._file_handlers[log_file] = file_handler
            return file_handler
            
        except Exception as e:
            # If file handler creation fails, log to console and continue
            console_logger = logging.getLogger('logging_manager')
            console_logger.error(f"Failed to create file handler for {log_file}: {e}")
            return None
    
    def cleanup(self):
        """Clean up all file handlers properly."""
        if self._is_shutdown:
            return
            
        self._is_shutdown = True
        
        # Close all file handlers
        for handler_name, handler in self._file_handlers.items():
            try:
                handler.close()
            except Exception as e:
                # Use print instead of logging since we're shutting down
                print(f"Error closing file handler {handler_name}: {e}")
        
        # Clear all loggers
        for logger_name, logger in self._loggers.items():
            try:
                # Remove all handlers
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    if hasattr(handler, 'close'):
                        handler.close()
            except Exception as e:
                print(f"Error cleaning up logger {logger_name}: {e}")
        
        self._file_handlers.clear()
        self._loggers.clear()
    
    @contextmanager
    def safe_logging_context(self):
        """Context manager for safe logging operations."""
        try:
            yield self
        except Exception as e:
            # Use print for errors during logging setup
            print(f"Logging error: {e}")
        finally:
            # Ensure cleanup happens
            pass


# Global logging manager instance
_logging_manager = LoggingManager()


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    formatter: Optional[logging.Formatter] = None
) -> logging.Logger:
    """
    Get a properly configured logger.
    
    This is the main function that should be used throughout the project
    instead of logging.getLogger() or logging.basicConfig().
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file name
        enable_console: Whether to enable console output
        formatter: Custom formatter
        
    Returns:
        Configured logger instance
    """
    return _logging_manager.get_logger(name, level, log_file, enable_console, formatter)


def setup_root_logging(level: int = logging.INFO):
    """
    Set up root logging configuration.
    This should be called once at application startup.
    
    Args:
        level: Root logging level
    """
    # Configure root logger to avoid basicConfig conflicts
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add a null handler to prevent unwanted output
    root_logger.addHandler(logging.NullHandler())


def cleanup_logging():
    """Clean up all logging resources."""
    _logging_manager.cleanup()


# Register cleanup on import
atexit.register(cleanup_logging)
