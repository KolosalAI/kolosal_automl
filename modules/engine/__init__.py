"""
ML Engine Components.

This subpackage contains all the core engine components for machine learning:
- Inference engine for model serving
- Training engine for model development
- Batch processor for efficient data handling
- Data preprocessor for feature engineering
- Quantizer for model optimization
- Various utilities and caching mechanisms
"""

from .inference_engine import InferenceEngine
from .train_engine import MLTrainingEngine
from .batch_processor import BatchProcessor
from .data_preprocessor import DataPreprocessor
from .quantizer import Quantizer
from .lru_ttl_cache import LRUTTLCache

# Import optimization modules
try:
    from .optimized_data_loader import OptimizedDataLoader, DatasetSize, LoadingStrategy, load_data_optimized
    from .adaptive_preprocessing import AdaptivePreprocessorConfig, PreprocessorConfigOptimizer
    from .memory_aware_processor import MemoryAwareDataProcessor, create_memory_aware_processor
    OPTIMIZATION_MODULES_AVAILABLE = True
except ImportError as e:
    # Graceful fallback if optimization modules have dependency issues
    print(f"Warning: Optimization modules not available: {e}")
    OptimizedDataLoader = None
    DatasetSize = None
    LoadingStrategy = None
    load_data_optimized = None
    AdaptivePreprocessorConfig = None
    PreprocessorConfigOptimizer = None
    MemoryAwareDataProcessor = None
    create_memory_aware_processor = None
    OPTIMIZATION_MODULES_AVAILABLE = False

# Import utilities
from .utils import (
    _json_safe,
    _scrub,
    _patch_pickle_for_locks,
    _return_none,
)

# Build __all__ dynamically based on what's available
__all__ = [
    # Core engines
    "InferenceEngine",
    "MLTrainingEngine",
    
    # Processors
    "BatchProcessor",
    "DataPreprocessor",
    "Quantizer",
    
    # Utilities
    "LRUTTLCache",
    
    # Helper functions
    "_json_safe",
    "_scrub",
    "_patch_pickle_for_locks",
    "_return_none",
    
    # Availability flag
    "OPTIMIZATION_MODULES_AVAILABLE",
]

# Add optimization modules to __all__ if they're available
if OPTIMIZATION_MODULES_AVAILABLE:
    __all__.extend([
        "OptimizedDataLoader",
        "DatasetSize", 
        "LoadingStrategy",
        "load_data_optimized",
        "AdaptivePreprocessorConfig",
        "PreprocessorConfigOptimizer", 
        "MemoryAwareDataProcessor",
        "create_memory_aware_processor",
    ])

# Version info
__version__ = "0.1.4"