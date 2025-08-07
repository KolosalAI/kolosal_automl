"""
Machine Learning Engine Modules.

This package contains all the core modules for the ML pipeline including:
- Configurations and settings
- Inference and training engines
- Data preprocessing and optimization
- Batch processing and quantization
"""

# Optional: Compile to bytecode on first import for better performance
import os
_AUTO_COMPILE = os.environ.get("KOLOSAL_AUTO_COMPILE", "false").lower() == "true"

if _AUTO_COMPILE:
    try:
        from .compiler import compile_on_import
        compile_on_import()
    except ImportError:
        # Silently continue if compiler module is not available
        pass

from .configs import *
from .engine import *
from .model_manager import SecureModelManager
from .ui import DataPreviewGenerator, SampleDataLoader

__version__ = "0.1.4"
__author__ = "Kolosal AI Team"

# Package metadata - build dynamically based on what's available
__all__ = [
    # From configs
    "TaskType",
    "OptimizationStrategy",
    "ModelType",
    "EngineState",
    "QuantizationType",
    "QuantizationMode",
    "NormalizationType",
    "BatchProcessingStrategy",
    "BatchPriority",
    "ModelSelectionCriteria",
    
    # Configuration classes
    "QuantizationConfig",
    "BatchProcessorConfig",
    "PreprocessorConfig",
    "InferenceEngineConfig",
    "MLTrainingEngineConfig",
    "MonitoringConfig",
    "ExplainabilityConfig",
    
    # From engine - core components
    "InferenceEngine",
    "MLTrainingEngine",
    "BatchProcessor",
    "DataPreprocessor",
    "Quantizer",
    
    # From model_manager
    "SecureModelManager",
    
    # From UI
    "DataPreviewGenerator",
    "SampleDataLoader",
    
    # Utils
    "LRUTTLCache",
    
    # Availability flag
    "OPTIMIZATION_MODULES_AVAILABLE",
]

# Add optimization modules if they're available
try:
    # These imports will only succeed if the optimization modules are properly available
    from .engine import OPTIMIZATION_MODULES_AVAILABLE
    if OPTIMIZATION_MODULES_AVAILABLE:
        # Import the optimization components
        from .engine import (
            OptimizedDataLoader, DatasetSize, LoadingStrategy, load_data_optimized,
            AdaptivePreprocessorConfig, PreprocessorConfigOptimizer,
            MemoryAwareDataProcessor, create_memory_aware_processor
        )
        
        # Add them to __all__
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
except (ImportError, AttributeError):
    # If optimization modules aren't available, that's okay
    OPTIMIZATION_MODULES_AVAILABLE = False