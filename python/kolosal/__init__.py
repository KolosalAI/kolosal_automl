"""
Kolosal AutoML - High-performance AutoML framework.

This package provides a unified interface to both the Rust and Python backends.
By default, the Rust backend is used for maximum performance. Set the environment
variable KOLOSAL_USE_RUST=0 to use the Python backend.

Example:
    >>> from kolosal import DataPreprocessor, TrainEngine
    >>> preprocessor = DataPreprocessor()
    >>> engine = TrainEngine()
"""

import os
import warnings
from typing import TYPE_CHECKING

__version__ = "0.2.0"
__author__ = "KolosalAI"

# Feature flag for backend selection
USE_RUST_BACKEND = os.environ.get("KOLOSAL_USE_RUST", "1") != "0"
_rust_available = False

# Try to import Rust backend
try:
    import kolosal_rust as _rust
    _rust_available = True
except ImportError:
    if USE_RUST_BACKEND:
        warnings.warn(
            "Rust backend not available. Falling back to Python backend. "
            "Install with `pip install kolosal[rust]` for better performance.",
            RuntimeWarning,
        )
    USE_RUST_BACKEND = False

# Select backend based on availability and preference
if USE_RUST_BACKEND and _rust_available:
    # Rust backend - high performance
    from kolosal._rust_backend import (
        DataPreprocessor,
        TrainEngine,
        InferenceEngine,
        HyperOptX,
        # Data loading
        load_csv,
        load_parquet,
        save_csv,
        get_file_info,
    )
    BACKEND = "rust"
else:
    # Python backend - fallback
    from kolosal._python_backend import (
        DataPreprocessor,
        TrainEngine,
        InferenceEngine,
        HyperOptX,
        load_csv,
        load_parquet,
        save_csv,
        get_file_info,
    )
    BACKEND = "python"

# Public API
__all__ = [
    # Version info
    "__version__",
    "BACKEND",
    "USE_RUST_BACKEND",
    # Core classes
    "DataPreprocessor",
    "TrainEngine", 
    "InferenceEngine",
    "HyperOptX",
    # Data I/O
    "load_csv",
    "load_parquet",
    "save_csv",
    "get_file_info",
]


def get_backend_info() -> dict:
    """Get information about the current backend.
    
    Returns:
        dict: Backend information including name and availability.
    """
    return {
        "backend": BACKEND,
        "rust_available": _rust_available,
        "rust_requested": os.environ.get("KOLOSAL_USE_RUST", "1") != "0",
        "version": __version__,
    }
