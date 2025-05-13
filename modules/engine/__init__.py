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

from modules.engine.inference_engine import InferenceEngine
from modules.engine.train_engine import MLTrainingEngine
from modules.engine.batch_processor import BatchProcessor
from modules.engine.data_preprocessor import DataPreprocessor
from modules.engine.quantizer import Quantizer
from modules.engine.lru_ttl_cache import LRUTTLCache

# Import utilities
from modules.engine.utils import (
    _json_safe,
    _scrub,
    _patch_pickle_for_locks,
    _return_none,
)

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
]

# Version info
__version__ = "1.0.0"